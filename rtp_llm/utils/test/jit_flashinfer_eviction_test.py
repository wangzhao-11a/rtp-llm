"""Unit tests for FlashInfer stale-op detection and eviction.

Covers the eviction chain used by setup_jit_cache():
- _read_ninja_graph follows include/subninja fragments (relative paths, cycles, max_files);
- _flashinfer_op_is_stale flags runfiles-anchored rules anywhere in the reachable graph,
  dangling include paths, and dangling `src` symlinks;
- _evict_dangling_flashinfer_ops removes stale ops under cached_ops/<op>/ while keeping
  healthy ones and never touching build.ninja files outside that layout.
"""

import os
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.utils.jit_sys_path_setup import (
    _evict_dangling_flashinfer_ops,
    _flashinfer_op_is_stale,
    _read_ninja_graph,
)


class NinjaGraphTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="flashinfer_graph_test_")
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, rel, text):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def test_follows_included_fragments(self):
        op = self.root / "op"
        self._write("op/build.ninja", "include extra.ninja\nrule main\n")
        self._write("op/extra.ninja", "build foo.o: rule\n  include_flags = -I/tmp/x\n")
        graph = _read_ninja_graph(op)
        self.assertIn("rule main", graph)
        self.assertIn("build foo.o", graph)
        self.assertIn("-I/tmp/x", graph)

    def test_follows_subninja_and_nested_relative_paths(self):
        op = self.root / "op"
        self._write("op/build.ninja", "subninja sub/dir/one.ninja\n")
        self._write("op/sub/dir/one.ninja", "include two.ninja\n")
        self._write("op/sub/dir/two.ninja", "depfile = inputs.d\n")
        graph = _read_ninja_graph(op)
        self.assertIn("depfile = inputs.d", graph)

    def test_cycle_is_bounded_by_max_files(self):
        op = self.root / "op"
        self._write("op/build.ninja", "include a.ninja\n")
        self._write("op/a.ninja", "include build.ninja\n")
        graph = _read_ninja_graph(op, max_files=2)
        self.assertEqual(graph.count("include a.ninja"), 1)
        self.assertEqual(graph.count("include build.ninja"), 1)

    def test_missing_include_is_ignored(self):
        op = self.root / "op"
        self._write("op/build.ninja", "include missing.ninja\nrule keep\n")
        self.assertEqual(_read_ninja_graph(op), "include missing.ninja\nrule keep\n")

    def test_no_ninja_returns_empty(self):
        op = self.root / "op"
        op.mkdir(parents=True)
        self.assertEqual(_read_ninja_graph(op), "")


class StaleDetectionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="flashinfer_stale_test_")
        self.root = Path(self._tmp.name)
        self.include_dir = self.root / "stable" / "include"
        self.include_dir.mkdir(parents=True)

    def tearDown(self):
        self._tmp.cleanup()

    def _op(self, name="op"):
        op = self.root / "cached_ops" / name
        op.mkdir(parents=True)
        return op

    def test_runfiles_anchor_in_build_ninja_is_stale(self):
        op = self._op()
        (op / "build.ninja").write_text(
            "build x.o: cxx /tmp/bazel/abc.runfiles/rtp/x.cu\n"
        )
        self.assertTrue(_flashinfer_op_is_stale(op))

    def test_runfiles_anchor_in_included_fragment_is_stale(self):
        op = self._op()
        (op / "build.ninja").write_text("include extra.ninja\n")
        (op / "extra.ninja").write_text(
            "build x.o: cxx /tmp/bazel/abc.runfiles/rtp/x.cu\n"
        )
        self.assertTrue(_flashinfer_op_is_stale(op))

    def test_dangling_include_is_stale(self):
        op = self._op()
        (op / "build.ninja").write_text(
            "build x.o: cxx y.cu\n  includes = -I{}\n".format(self.root / "gone")
        )
        self.assertTrue(_flashinfer_op_is_stale(op))

    def test_healthy_op_is_not_stale(self):
        op = self._op()
        (op / "build.ninja").write_text(
            "build x.o: cxx y.cu\n  includes = -I{}\n  depfile = x.d\n".format(
                self.include_dir
            )
        )
        self.assertFalse(_flashinfer_op_is_stale(op))

    def test_dangling_src_symlink_is_stale(self):
        op = self._op()
        (op / "build.ninja").write_text(
            "build x.o: cxx y.cu\n  includes = -I{}\n".format(self.include_dir)
        )
        os.symlink(self.root / "gone-src", op / "src")
        self.assertTrue(_flashinfer_op_is_stale(op))

    def test_missing_ninja_is_not_stale(self):
        op = self._op()
        self.assertFalse(_flashinfer_op_is_stale(op))


class EvictionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="flashinfer_evict_test_")
        self.root = Path(self._tmp.name)
        self.cache = self.root / "cache" / "flashinfer"
        self.include_dir = self.root / "stable" / "include"
        self.include_dir.mkdir(parents=True)

    def tearDown(self):
        self._tmp.cleanup()

    def _make_op(self, name, ninja_text):
        op = self.cache / "cached_ops" / name
        op.mkdir(parents=True)
        (op / "build.ninja").write_text(ninja_text)
        return op

    def test_evicts_stale_and_keeps_healthy(self):
        stale = self._make_op(
            "stale_op",
            "build x.o: cxx /tmp/bazel/abc.runfiles/rtp/x.cu\n",
        )
        healthy = self._make_op(
            "healthy_op",
            "build x.o: cxx y.cu\n  includes = -I{}\n".format(self.include_dir),
        )
        unrelated = self.cache / "other" / "deep" / "build.ninja"
        unrelated.parent.mkdir(parents=True)
        unrelated.write_text("build x.o: cxx /tmp/bazel/abc.runfiles/rtp/x.cu\n")

        _evict_dangling_flashinfer_ops(self.cache)

        self.assertFalse(stale.exists())
        self.assertTrue(healthy.exists())
        self.assertTrue(unrelated.exists())  # not a cached_ops/<op> layout entry


if __name__ == "__main__":
    unittest.main()