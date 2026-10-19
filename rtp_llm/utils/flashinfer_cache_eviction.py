"""FlashInfer JIT cache eviction shared by the launcher and the JIT manager.

FlashInfer records the absolute paths of the package instance that generated
each cached op's ninja rules. A bazel action's runfiles tree only lives for
that action, so ops generated from a runfiles import dangle for every later
consumer of the shared cache -- and a JIT snapshot can restore that poisoned
state onto a fresh worker. Both entry points (the test launcher, which owns
``$HOME/.cache/flashinfer``, and the JIT cache manager, which owns the managed
component directory) sweep with this module so their rules cannot drift.
"""

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Tuple

from filelock import FileLock, Timeout


def read_ninja_graph(op_dir: Path, max_files: int = 16) -> str:
    """Concatenate the op's ninja files, following include/subninja statements.

    FlashInfer splits the generated graph across the op's build.ninja and
    included fragments; the fragments may sit outside the op directory, so a
    plain directory glob misses the paths written into them.
    """
    seen: set[Path] = set()
    queue: list[Path] = [op_dir / "build.ninja"]
    chunks: list[str] = []
    while queue and len(seen) < max_files:
        path = queue.pop(0)
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        chunks.append(text)
        for m in re.finditer(r"^(?:include|subninja)\s+(\S+)", text, re.M):
            target = Path(m.group(1))
            if not target.is_absolute():
                target = (path.parent / target).resolve()
            queue.append(target)
    return "".join(chunks)


def flashinfer_op_is_stale(op_dir: Path) -> bool:
    # The compile inputs / depfiles may live in an included ninja fragment
    # rather than the top-level build.ninja, so scan the full reachable
    # graph -- the runfiles-anchor rule applies to whichever file the
    # generator wrote the paths into.
    rules = read_ninja_graph(op_dir)
    if not rules:
        return False
    # FlashInfer embeds the absolute paths of the package instance that
    # generated the rules -- include flags, compile inputs, depfiles and the
    # module source list alike. A bazel action's runfiles tree only lives for
    # that action, so rules anchored there dangle for every later consumer of
    # the shared workspace -- whether or not the tree still exists at eviction
    # time -- and only rules anchored to stable locations (package-copy cache,
    # toolkit dirs) are safe to keep.
    # Scanning only the -I/-isystem flags misses ops whose includes point at
    # the stable package-copy cache while the compile inputs anchor into a dead
    # per-action runfiles tree (ninja then fails on the missing input file).
    # Treat any .runfiles-anchored path anywhere in the rules as stale.
    if re.search(r"/[^/\s]+\.runfiles(/|$|\s)", rules):
        return True
    include_paths = re.findall(r"(?:-I|-isystem\s+)(/\S+)", rules)
    if any(".runfiles" in path for path in include_paths):
        return True
    if any(not Path(path).exists() for path in include_paths):
        return True
    src_link = op_dir / "src"
    return src_link.is_symlink() and not src_link.exists()


def evict_dangling_flashinfer_ops(
    cache_dir: Path, lock_timeout_s: float = 300.0
) -> Tuple[int, int]:
    """Drop cached FlashInfer ops whose build rules anchor to include paths that
    are not stable for later consumers of the shared cache directory.

    FlashInfer's generated FMHA ninja rules embed the absolute include paths of the
    flashinfer package instance that generated them. A bazel test action's runfiles
    sandbox is per-action, so rules generated from a runfiles import dangle for every
    later action that reuses the shared cache directory and fail with missing-header
    errors -- including when the generating action is still alive at eviction time,
    because its sandbox disappears when the action completes. Evicting the affected
    ops makes FlashInfer regenerate them against the current stable package copy.

    FlashInfer serializes each op's build and load with a FileLock at
    ``cached_ops/tmp/<op_name>.lock``; eviction takes that same lock and re-judges
    staleness only inside it, so it cannot delete an op directory that a concurrent
    compile or load is using or has just refreshed.
    """
    stale_candidates = []
    for root, _dirs, files in os.walk(cache_dir, followlinks=False):
        op_dir = Path(root)
        # FlashInfer layout only: cached_ops/<op_name>/build.ninja (never trust an
        # unrelated build.ninja deeper in the tree).
        if op_dir.parent.name != "cached_ops" or "build.ninja" not in files:
            continue
        if flashinfer_op_is_stale(op_dir):
            stale_candidates.append(op_dir)
    evicted = 0
    skipped = 0
    for op_dir in sorted(stale_candidates):
        lock_dir = op_dir.parent / "tmp"
        try:
            lock_dir.mkdir(parents=True, exist_ok=True)
            lock = FileLock(
                str(lock_dir / f"{op_dir.name}.lock"), timeout=lock_timeout_s
            )
            with lock:
                # Re-read the rules under the op's own lock: a compile that ran
                # since the first pass may have replaced the stale rules already.
                if flashinfer_op_is_stale(op_dir):
                    shutil.rmtree(op_dir, ignore_errors=True)
                    if op_dir.exists():
                        skipped += 1
                        logging.warning(
                            f"[Package Setup] Eviction left {op_dir} in place "
                            "(rmtree failed); it will be swept again next run"
                        )
                        continue
                    evicted += 1
                    logging.info(
                        f"[Package Setup] Evicted stale FlashInfer cache op: {op_dir}"
                    )
        except Timeout:
            # Somebody else owns the op (likely recompiling it); leave it alone.
            skipped += 1
            logging.info(f"[Package Setup] Skipped eviction of {op_dir}: lock busy")
    if stale_candidates:
        logging.info(
            f"[Package Setup] FlashInfer stale-op sweep: {evicted} evicted, "
            f"{skipped} skipped out of {len(stale_candidates)} candidates "
            f"under {cache_dir}"
        )
    else:
        logging.info(f"[Package Setup] FlashInfer stale-op sweep: no candidates under {cache_dir}")
    return evicted, skipped




def flashinfer_cache_dir(workspace_base: Path) -> Path:
    """FlashInfer derives its cache from the workspace base this way."""
    return Path(workspace_base) / ".cache" / "flashinfer"
