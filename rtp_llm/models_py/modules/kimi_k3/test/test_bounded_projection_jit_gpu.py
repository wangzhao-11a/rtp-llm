"""PROPOSAL: real CUDA wrapper/key contracts; requires private dependency manifest.

No mocks, dependency installation, global setup changes or production requests.
The manifest distinguishes synthetic fixtures from model-derived geometries.
Missing CUDA/dependencies are failures, never a successful skipped GPU result.
"""

import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def wheel_member_sha256(artifact, path):
    relative = Path(path).resolve().relative_to(Path(artifact["root"]).resolve())
    with zipfile.ZipFile(artifact["wheel"]) as wheel:
        expected = hashlib.sha256(wheel.read(relative.as_posix())).hexdigest()
    observed = sha256(path)
    if observed != expected:
        raise AssertionError(
            f"extracted member differs from approved wheel: {relative}"
        )
    return observed


def independent_probe_m_values(maximum, kind, extra=()):
    """Probe policy intentionally independent of the warmup representative grid."""
    if maximum <= 0:
        raise AssertionError("positive probe upper bound required")
    if kind == "synthetic_unit_fixture":
        return tuple(range(1, maximum + 1))
    values = {1, maximum, max(1, maximum - 1)}
    power = 1
    while power <= maximum:
        values.update(
            value for value in (power - 1, power, power + 1) if 0 < value <= maximum
        )
        power *= 2
    for value in extra:
        if type(value) is not int or not 0 < value <= maximum:
            raise AssertionError("explicit probe is outside the declared bound")
        values.add(value)
    return tuple(sorted(values))


@contextlib.contextmanager
def triton_keys():
    from triton import knobs

    previous, keys = knobs.runtime.jit_cache_hook, set()

    def listen(**event):
        function = event["fn"].jit_function
        keys.add((function.__module__, function.__name__, str(event["key"])))
        if previous is not None:
            return previous(**event)

    knobs.runtime.jit_cache_hook = listen
    try:
        yield keys
    finally:
        knobs.runtime.jit_cache_hook = previous


class BoundedProjectionGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(
            Path(os.environ["RTP_K3_JIT_TEST_MANIFEST"]).read_text()
        )
        provenance = cls.manifest["geometry_provenance"]
        kind = provenance["kind"]
        if kind not in ("synthetic_unit_fixture", "model_derived_geometry"):
            raise AssertionError("unsupported geometry provenance kind")
        if not provenance.get("description") or not provenance.get("public_source_sha"):
            raise AssertionError(
                "geometry description and tested source SHA are required"
            )
        if kind == "model_derived_geometry" and not provenance.get(
            "model_config_sha256"
        ):
            raise AssertionError(
                "model-derived geometry requires its config source SHA"
            )
        if kind == "synthetic_unit_fixture" and provenance.get("model_config_sha256"):
            raise AssertionError("synthetic fixtures must not claim a model config SHA")
        families = ("fp8", "fp8_kv_up", "bf16_kv_up", "fla")
        geometry = cls.manifest["geometry"]
        for family in families:
            if family not in geometry:
                raise AssertionError(f"explicit geometry family required: {family}")
            if not geometry[family] and (
                kind == "synthetic_unit_fixture"
                or not cls.manifest.get("not_applicable_families", {}).get(family)
            ):
                raise AssertionError(
                    f"nonempty geometry or model N/A reason required: {family}"
                )
        if not any(geometry[family] for family in families):
            raise AssertionError("at least one real geometry family must execute")
        cls.temporary = tempfile.TemporaryDirectory(prefix="k3-directed-gpu-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.cache = Path(cls.temporary.name)
        # Must precede the first DG/Triton import/runtime creation in this process.
        if any(
            name in sys.modules
            for name in ("deep_gemm", "fla", "rtp_kernel", "rtp_ops")
        ):
            raise RuntimeError("dependency overlay must precede wrapper imports")
        os.environ["DG_JIT_CACHE_DIR"] = str(cls.cache / "dg")
        os.environ["TRITON_CACHE_DIR"] = str(cls.cache / "triton")
        cls.evidence = {
            "dependencies": {},
            "checks": [],
            "dg_cache_environment": "DG_JIT_CACHE_DIR",
            "dg_cache_root": str(cls.cache / "dg"),
            "geometry_provenance": cls.manifest["geometry_provenance"],
            "not_applicable_families": cls.manifest.get("not_applicable_families", {}),
        }
        expected = {
            "deep-gemm": "2.8.2+98a672e.cu132",
            "flash-linear-attention": "0.5.0+rtp.3a9ce1c.6",
            "rtp-kernel": "0.1.0+94fd20eb.cu132",
        }
        for distribution, version in expected.items():
            item = cls.manifest["dependencies"][distribution]
            observed = sha256(item["wheel"])
            if observed != item["sha256"]:
                raise AssertionError(f"artifact SHA256 mismatch: {distribution}")
            sys.path.insert(0, str(Path(item["root"]).resolve()))
            cls.evidence["dependencies"][distribution] = {
                "wheel_sha256": observed,
                "expected_version": version,
            }
        vendor = cls.manifest["dependencies"]["cutlass-site"]
        provenance_hash = sha256(vendor["provenance"])
        if provenance_hash != vendor["sha256"]:
            raise AssertionError("Cutlass merged-wheel provenance SHA256 mismatch")
        vendor_root = Path(vendor["root"]).resolve()
        # Match the real wheel's .pth entry before Bazel's parent namespaces.
        sys.path[:0] = [
            str(vendor_root / "nvidia_cutlass_dsl" / "dsl_packages"),
            str(vendor_root),
        ]
        import cutlass

        cutlass_path = Path(cutlass.__file__).resolve()
        cutlass_version = importlib.metadata.version("nvidia-cutlass-dsl")
        if not cutlass_path.is_relative_to(vendor_root) or cutlass_version != "4.7.1":
            raise AssertionError("expected real merged nvidia-cutlass-dsl4.7.1 package")
        cls.evidence["dependencies"]["cutlass-site"] = {
            "provenance_sha256": provenance_hash,
            "module": str(cutlass_path),
            "module_sha256": sha256(cutlass_path),
            "version": cutlass_version,
        }
        import torch
        import triton

        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            raise AssertionError("real SM10 CUDA execution required")
        cls.torch, cls.device = torch, torch.device("cuda")
        for distribution, module_name in (
            ("deep-gemm", "deep_gemm"),
            ("flash-linear-attention", "fla"),
            ("rtp-kernel", "rtp_kernel"),
        ):
            module = importlib.import_module(module_name)
            root = Path(cls.manifest["dependencies"][distribution]["root"]).resolve()
            resolved = Path(module.__file__).resolve()
            if not resolved.is_relative_to(root):
                raise AssertionError(
                    f"wrong resolved module: {distribution}: {resolved}"
                )
            actual = importlib.metadata.version(distribution)
            if actual != expected[distribution]:
                raise AssertionError(
                    f"wrong distribution version: {distribution}: {actual}"
                )
            cls.evidence["dependencies"][distribution].update(
                version=actual,
                module=str(resolved),
                module_sha256=wheel_member_sha256(
                    cls.manifest["dependencies"][distribution], resolved
                ),
            )
        import deep_gemm
        import rtp_ops
        from fla.modules import l2norm

        from rtp_llm.models_py.modules.kimi_k3 import bounded_projection_jit_warmup

        cls.warmup, cls.l2norm = bounded_projection_jit_warmup, l2norm
        rtp_native = Path(rtp_ops.__file__).resolve()
        rtp_root = Path(cls.manifest["dependencies"]["rtp-kernel"]["root"]).resolve()
        if not rtp_native.is_relative_to(rtp_root) or rtp_native.suffix != ".so":
            raise AssertionError(
                "expected real rtp_ops native extension from approved wheel"
            )
        cls.evidence["dependencies"]["rtp-kernel"].update(
            native_module=str(rtp_native),
            native_sha256=wheel_member_sha256(
                cls.manifest["dependencies"]["rtp-kernel"], rtp_native
            ),
        )
        if list(cls.cache.rglob("kernel.cubin")):
            raise AssertionError("the DG test process must start with no compiled keys")
        cls.evidence.update(
            torch=torch.__version__,
            triton=triton.__version__,
            gpu=torch.cuda.get_device_name(),
            capability=torch.cuda.get_device_capability(),
            dg_extension_sha256=wheel_member_sha256(
                cls.manifest["dependencies"]["deep-gemm"], deep_gemm._C.__file__
            ),
            fla_l2norm_sha256=wheel_member_sha256(
                cls.manifest["dependencies"]["flash-linear-attention"], l2norm.__file__
            ),
        )
        output = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", cls.temporary.name))
        cls.evidence_path = output / "k3_directed_gpu_evidence.json"
        cls.addClassCleanup(cls.write_evidence)

    @classmethod
    def write_evidence(cls):
        cls.evidence["dg_key_files"] = [
            {
                "cubin": str(path.resolve()),
                "source_sha256": sha256(path.parent / "kernel.cu"),
            }
            for path in cls.cache.rglob("kernel.cubin")
        ]
        cls.evidence_path.write_text(json.dumps(cls.evidence, indent=2, sort_keys=True))

    def dg_keys(self):
        # DG98 exposes no Python compile listener. A fresh, invocation-private
        # DeepJIT cache makes new compiled source keys observable on disk.
        return {
            (
                str(path.parent.relative_to(self.cache)),
                sha256(path.parent / "kernel.cu"),
            )
            for path in self.cache.rglob("kernel.cubin")
        }

    def sampled_rows(self, rows):
        if rows <= 256:
            return self.torch.arange(rows, device=self.device)
        return (
            self.torch.linspace(0, rows - 1, min(rows, 17), device=self.device)
            .long()
            .unique()
        )

    def family_items(self, family):
        items = self.manifest["geometry"][family]
        if not items:
            self.skipTest(
                f"model-derived {family} not applicable: "
                f"{self.manifest['not_applicable_families'][family]}"
            )
        return items

    def test_fp8_actual_outer_wrapper_preserves_sampled_keys_and_values(self):
        items = self.family_items("fp8")
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
            CudaFp8GEMMLinear,
        )

        torch = self.torch
        for index, item in enumerate(items):
            n, k = item["n"], item["k"]
            weight = (
                torch.empty((n, k), device=self.device, dtype=torch.bfloat16)
                .random_(-1, 2)
                .to(torch.float8_e4m3fn)
            )
            # Packed UE8M0 SF is MN-major/TMA aligned, including SF_K=1.
            scales = torch.empty_strided(
                (n, (k + 511) // 512),
                (1, (n + 3) // 4 * 4),
                device=self.device,
                dtype=torch.int32,
            ).fill_(0x7F7F7F7F)
            self.assertEqual(scales.stride(-2), 1)
            self.assertGreaterEqual(scales.stride(-1), (n + 3) // 4 * 4)
            self.assertEqual(scales.stride(-1) % 4, 0)
            projection = CudaFp8GEMMLinear(weight=weight, weight_scales=scales)
            import deep_gemm

            samples = self.warmup._sample_m_values(
                item["max_m"],
                n,
                k,
                fp8=True,
                k_major=True,
                num_sms=deep_gemm.get_num_sms(),
            )
            self.assertTrue(samples)
            probes = independent_probe_m_values(
                item["max_m"],
                self.manifest["geometry_provenance"]["kind"],
                item.get("probe_m", ()),
            )
            self.warmup._check_projection_memory(
                self.device,
                self.warmup._projection_scratch_bytes(max(samples), k, n, fp8=True),
            )
            if index == 0:
                before = self.dg_keys()
                unprepared = torch.ones(
                    (1, k), device=self.device, dtype=torch.bfloat16
                )
                projection(unprepared)
                torch.cuda.synchronize()
                observed = self.dg_keys() - before
                self.assertTrue(
                    observed,
                    "cold actual FP8 call must create observable DG keys before preparation",
                )
                self.evidence["checks"].append(
                    {"kind": "fp8_cold_negative_control", "new_keys": sorted(observed)}
                )
                del unprepared
            with triton_keys() as keys:
                self.warmup._warmup_fp8(projection, samples)
                torch.cuda.synchronize()
                prepared, prepared_triton = self.dg_keys(), set(keys)
                self.assertTrue(
                    prepared, "DG cache routing must yield real compiled artifacts"
                )
                for rows in probes:
                    hidden = torch.empty(
                        (rows, k), device=self.device, dtype=torch.bfloat16
                    ).random_(-1, 2)
                    actual = projection(hidden)
                    selected = self.sampled_rows(rows)
                    # {-1,0,1} values and unit UE8M0 scales are exactly representable.
                    expected = hidden[selected].float() @ weight.float().T
                    torch.testing.assert_close(
                        actual[selected],
                        expected.to(torch.bfloat16),
                        rtol=0.015625,
                        atol=0.03125,
                    )
                    del hidden, actual, expected
                torch.cuda.synchronize()
                observed = self.dg_keys()
                self.assertEqual(
                    observed,
                    prepared,
                    f"unseen DG keys under independent FP8 probes: {sorted(observed - prepared)}",
                )
                self.assertEqual(keys, prepared_triton)
            self.evidence["checks"].append(
                {
                    "kind": "fp8",
                    "geometry": item,
                    "sampled_m": samples,
                    "independent_probe_m": probes,
                    "dg_keys": sorted(prepared),
                    "triton_keys": sorted(keys),
                }
            )
            del projection, weight, scales
            torch.cuda.empty_cache()

    def test_bf16_actual_skip_head_wrapper_preserves_sampled_keys_and_values(self):
        items = self.family_items("bf16_kv_up")
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import (
            CudaF16Linear,
        )

        torch = self.torch
        for index, item in enumerate(items):
            n, k, splits = item["n"], item["k"], tuple(item["head_splits"])
            heads = n // (splits[0] + splits[2])
            raw = torch.empty((k, n), device=self.device, dtype=torch.bfloat16).random_(
                -1, 2
            )
            projection = CudaF16Linear(raw)
            probe = torch.zeros((1, k), device=self.device, dtype=torch.bfloat16)
            self.assertTrue(projection.supports_skip_head_mid(probe, splits))
            del probe
            import deep_gemm

            samples = self.warmup._sample_m_values(
                item["max_m"],
                n,
                k,
                fp8=False,
                k_major=(
                    projection.weight.T.is_contiguous()
                    and torch.cuda.get_device_capability(self.device)[0] == 10
                ),
                num_sms=deep_gemm.get_num_sms(),
            )
            self.assertTrue(samples)
            probes = independent_probe_m_values(
                item["max_m"],
                self.manifest["geometry_provenance"]["kind"],
                (
                    *item.get("probe_m", ()),
                    *range(1, min(item["max_m"], 32768) + 1, 127),
                ),
            )
            self.warmup._check_projection_memory(
                self.device,
                self.warmup._projection_scratch_bytes(
                    max(samples), k, heads * sum(splits), fp8=False
                ),
            )
            if index == 0:
                before = self.dg_keys()
                unprepared = torch.ones(
                    (1, k), device=self.device, dtype=torch.bfloat16
                )
                projection.forward_skip_head_mid(unprepared, splits)
                torch.cuda.synchronize()
                observed = self.dg_keys() - before
                self.assertTrue(
                    observed,
                    "cold actual BF16 call must create observable DG keys before preparation",
                )
                self.evidence["checks"].append(
                    {"kind": "bf16_cold_negative_control", "new_keys": sorted(observed)}
                )
                del unprepared
            self.assertTrue(self.warmup._warmup_bf16(projection, splits, samples))
            torch.cuda.synchronize()
            prepared = self.dg_keys()
            self.assertTrue(prepared)
            for rows in probes:
                hidden = torch.empty(
                    (rows, k), device=self.device, dtype=torch.bfloat16
                ).random_(-1, 2)
                packed = torch.full(
                    (rows, heads * sum(splits)),
                    float("nan"),
                    device=self.device,
                    dtype=torch.bfloat16,
                )
                projection.forward_skip_head_mid(hidden, splits, output=packed)
                selected = self.sampled_rows(rows)
                expected = (hidden[selected].float() @ raw.float()).reshape(
                    -1, heads, splits[0] + splits[2]
                )
                viewed = packed[selected].reshape(-1, heads, sum(splits))
                torch.testing.assert_close(
                    viewed[..., : splits[0]],
                    expected[..., : splits[0]].to(torch.bfloat16),
                    rtol=0.015625,
                    atol=0.03125,
                )
                torch.testing.assert_close(
                    viewed[..., -splits[2] :],
                    expected[..., -splits[2] :].to(torch.bfloat16),
                    rtol=0.015625,
                    atol=0.03125,
                )
                self.assertTrue(
                    torch.isnan(viewed[..., splits[0] : splits[0] + splits[1]]).all()
                )
                del hidden, packed, expected, viewed
            torch.cuda.synchronize()
            observed = self.dg_keys()
            self.evidence["checks"].append(
                {
                    "kind": "bf16_kv_up",
                    "geometry": item,
                    "sampled_m": samples,
                    "independent_probe_m": probes,
                    "dg_keys": sorted(prepared),
                    "observed_dg_keys": sorted(observed),
                    "unseen_dg_keys": sorted(observed - prepared),
                }
            )
            self.assertEqual(
                observed,
                prepared,
                f"unseen DG keys under independent BF16 probes: {sorted(observed - prepared)}",
            )
            del projection, raw
            torch.cuda.empty_cache()

    def test_fp8_kv_up_actual_backend_preserves_independent_probe_keys_and_values(self):
        items = self.family_items("fp8_kv_up")
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
            CudaFp8GEMMLinear,
        )

        torch = self.torch
        for index, item in enumerate(items):
            n, k, splits = item["n"], item["k"], tuple(item["head_splits"])
            heads = n // (splits[0] + splits[2])
            weight = (
                torch.empty((n, k), device=self.device, dtype=torch.bfloat16)
                .random_(-1, 2)
                .to(torch.float8_e4m3fn)
            )
            scales = torch.empty_strided(
                (n, (k + 511) // 512),
                (1, (n + 3) // 4 * 4),
                device=self.device,
                dtype=torch.int32,
            ).fill_(0x7F7F7F7F)
            projection = CudaFp8GEMMLinear(weight=weight, weight_scales=scales)
            probe = torch.zeros((1, k), device=self.device, dtype=torch.bfloat16)
            supported = projection.supports_skip_head_mid(probe, splits)
            self.assertIs(type(item["expected_skip_head_mid"]), bool)
            self.assertEqual(supported, item["expected_skip_head_mid"])
            del probe
            import deep_gemm

            backend = "skip_head_mid" if supported else "ordinary_fallback"
            samples = self.warmup._sample_m_values(
                item["max_m"],
                n,
                k,
                fp8=True,
                k_major=(
                    not supported
                    and weight.stride(-1) == 1
                    and torch.cuda.get_device_capability(self.device)[0] == 10
                ),
                num_sms=deep_gemm.get_num_sms(),
            )
            probes = independent_probe_m_values(
                item["max_m"],
                self.manifest["geometry_provenance"]["kind"],
                item.get("probe_m", ()),
            )
            self.assertTrue(samples)
            self.warmup._check_projection_memory(
                self.device,
                self.warmup._projection_scratch_bytes(
                    max(samples), k, heads * sum(splits) if supported else n, fp8=True
                ),
            )
            if index == 0:
                before = self.dg_keys()
                unprepared = torch.ones(
                    (1, k), device=self.device, dtype=torch.bfloat16
                )
                if supported:
                    projection.forward_skip_head_mid(unprepared, splits)
                else:
                    projection(unprepared)
                torch.cuda.synchronize()
                observed = self.dg_keys() - before
                if supported or observed:
                    self.assertTrue(
                        observed, "cold FP8 skip-head call must add observable keys"
                    )
                else:
                    self.assertTrue(
                        any(
                            check["kind"] == "fp8"
                            and check["geometry"]["n"] == n
                            and check["geometry"]["k"] == k
                            for check in self.evidence["checks"]
                        ),
                        "ordinary fallback reuse requires completed ordinary FP8 checks with matching N/K",
                    )
                self.evidence["checks"].append(
                    {
                        "kind": "fp8_kv_up_initial_call",
                        "actual_backend": backend,
                        "reused_ordinary_keys": not observed,
                        "new_keys": sorted(observed),
                    }
                )
                del unprepared
            with triton_keys() as keys:
                self.warmup._warmup_fp8(projection, samples, head_splits=splits)
                torch.cuda.synchronize()
                prepared, prepared_triton = self.dg_keys(), set(keys)
                self.assertTrue(prepared)
                for rows in probes:
                    hidden = torch.empty(
                        (rows, k), device=self.device, dtype=torch.bfloat16
                    ).random_(-1, 2)
                    selected = self.sampled_rows(rows)
                    expected = hidden[selected].float() @ weight.float().T
                    if supported:
                        actual = torch.full(
                            (rows, heads * sum(splits)),
                            float("nan"),
                            device=self.device,
                            dtype=torch.bfloat16,
                        )
                        projection.forward_skip_head_mid(hidden, splits, output=actual)
                        viewed = actual[selected].reshape(-1, heads, sum(splits))
                        reference = expected.reshape(-1, heads, splits[0] + splits[2])
                        torch.testing.assert_close(
                            viewed[..., : splits[0]],
                            reference[..., : splits[0]].to(torch.bfloat16),
                            rtol=0.015625,
                            atol=0.03125,
                        )
                        torch.testing.assert_close(
                            viewed[..., -splits[2] :],
                            reference[..., -splits[2] :].to(torch.bfloat16),
                            rtol=0.015625,
                            atol=0.03125,
                        )
                        self.assertTrue(
                            torch.isnan(
                                viewed[..., splits[0] : splits[0] + splits[1]]
                            ).all()
                        )
                        del viewed, reference
                    else:
                        actual = projection(hidden)
                        torch.testing.assert_close(
                            actual[selected],
                            expected.to(torch.bfloat16),
                            rtol=0.015625,
                            atol=0.03125,
                        )
                    del hidden, actual, expected
                torch.cuda.synchronize()
                observed = self.dg_keys()
                self.assertEqual(
                    observed,
                    prepared,
                    f"unseen DG keys under independent FP8 {backend} probes: {sorted(observed - prepared)}",
                )
                self.assertEqual(keys, prepared_triton)
            self.evidence["checks"].append(
                {
                    "kind": "fp8_kv_up",
                    "actual_backend": backend,
                    "supports_skip_head_mid": supported,
                    "geometry": item,
                    "sampled_m": samples,
                    "independent_probe_m": probes,
                    "dg_keys": sorted(prepared),
                    "triton_keys": sorted(keys),
                }
            )
            del projection, weight, scales
            torch.cuda.empty_cache()

    def test_tiny_fla_preparation_covers_real_wrapper_sampled_nb_keys_and_values(self):
        torch = self.torch
        kernel = self.l2norm.l2norm_fwd_kernel
        for item in self.family_items("fla"):
            kernel.device_caches.clear()
            heads, maximum = item["heads"], item["max_call_rows"]
            buckets = tuple(range(1, (heads * maximum + 65535) // 65536 + 1))
            with triton_keys() as keys:
                self.warmup._warmup_fla(self.device, buckets)
                torch.cuda.synchronize()
                prepared = set(keys)
                self.assertEqual(
                    len(prepared),
                    len(buckets),
                    "each prepared NB must produce one observable FLA compile key",
                )
                for nb in buckets:
                    first = (nb - 1) * 65536 // heads + 1
                    last = min(nb * 65536 // heads, maximum)
                    for rows in sorted({first, last}):
                        self.assertGreaterEqual(rows, 1)
                        self.assertEqual((rows * heads + 65535) // 65536, nb)
                        self.warmup._check_projection_memory(
                            self.device, rows * heads * (128 * 4 + 4)
                        )
                        source = torch.empty(
                            (1, rows, heads, 128),
                            device=self.device,
                            dtype=torch.bfloat16,
                        ).uniform_(-1, 1)
                        actual, rstd = self.l2norm.l2norm_fwd(source)
                        selected = self.sampled_rows(rows * heads)
                        values = source.reshape(-1, 128)[selected].float()
                        expected_rstd = (values.square().sum(-1) + 1e-6).rsqrt()
                        expected = values * expected_rstd[:, None]
                        torch.testing.assert_close(
                            actual.reshape(-1, 128)[selected],
                            expected.to(torch.bfloat16),
                            rtol=0.015625,
                            atol=0.00390625,
                        )
                        torch.testing.assert_close(
                            rstd.flatten()[selected],
                            expected_rstd,
                            rtol=1e-5,
                            atol=1e-6,
                        )
                        del source, actual, rstd, values, expected
                torch.cuda.synchronize()
                self.assertEqual(keys, prepared)
            self.evidence["checks"].append(
                {"kind": "fla", "geometry": item, "triton_keys": sorted(keys)}
            )


if __name__ == "__main__":
    unittest.main()
