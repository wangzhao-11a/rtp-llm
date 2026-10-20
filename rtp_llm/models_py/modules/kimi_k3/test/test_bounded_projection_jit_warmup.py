"""CPU planning/dispatch checks; these do not claim GPU compilation coverage."""

import ast
import importlib.util
import math
import os
import sys
import types
import unittest
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence
from unittest import mock

HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "bounded_projection_jit_warmup.py"
if not MODULE_PATH.exists():
    MODULE_PATH = HERE.parent / "bounded_projection_jit_warmup.py"


def load_warmup():
    spec = importlib.util.spec_from_file_location("k3_directed_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeviceMetadata:
    def __init__(self, name):
        self.name, self.type = name, name.split(":")[0]

    def __str__(self):
        return self.name


class TensorMetadata:
    def __init__(
        self, shape, *, strides=None, dtype="torch.float8_e4m3fn", device="cuda:0"
    ):
        self.shape = shape
        self._stride = strides if strides is not None else (shape[1], 1)
        self.dtype = dtype
        self.device = DeviceMetadata(device)

    def stride(self, dim=None):
        return self._stride if dim is None else self._stride[dim]


class CudaFp8GEMMLinear:
    def __init__(self, *, scale_strides=(1, 128), device="cuda:0"):
        self.weight = TensorMetadata((128, 512), device=device)
        self.weight_scales = TensorMetadata(
            (128, 1), strides=scale_strides, dtype="torch.int32", device=device
        )
        self.N, self.K, self.scale_ue8m0 = 128, 512, True
        self._flashinfer_linear = None
        self.rows = []

    def __call__(self, hidden):
        self.rows.append(hidden.shape[0])
        return object()


class KimiK3KDA:
    def __init__(self, heads=12):
        self.prefill = SimpleNamespace(
            local_num_k_heads=heads,
            head_k_dim=128,
            conv_weights=TensorMetadata((128, 4), dtype="torch.bfloat16"),
        )


class BoundedProjectionWarmupTest(unittest.TestCase):
    def setUp(self):
        self.warmup = load_warmup()

    def test_collects_outer_wrappers_once_and_keeps_scale_layouts(self):
        first = CudaFp8GEMMLinear()
        same = CudaFp8GEMMLinear()
        other = CudaFp8GEMMLinear(scale_strides=(1, 256))
        nested = SimpleNamespace(weight=first.weight)
        model = SimpleNamespace(
            named_modules=lambda: [
                ("a", first),
                ("a._deepgemm_linear", nested),
                ("b", same),
                ("c", other),
                ("cpu", CudaFp8GEMMLinear(device="cpu")),
            ]
        )
        self.assertEqual(
            [name for name, _ in self.warmup._fp8_projections(model)], ["a", "c"]
        )

    def test_uses_scheduler_bound_and_verify_padding(self):
        model = SimpleNamespace(
            parallelism_config=SimpleNamespace(tp_size=8),
            config=SimpleNamespace(
                moe_prefill_max_tokens_per_rank=65536,
                max_seq_len=1048576,
                gen_num_per_cycle=4,
            ),
            _max_generate_batch_size=9,
        )
        self.assertEqual(
            self.warmup._projection_rows(model, SimpleNamespace(is_decode_role=False)),
            65536,
        )
        self.assertEqual(
            self.warmup._projection_rows(model, SimpleNamespace(is_decode_role=True)),
            80,
        )

    def test_fla_buckets_respect_actual_heads_and_checkpoint_grouping(self):
        dependency = types.ModuleType(
            "rtp_llm.models_py.modules.factory.attention.cuda_kda_impl"
        )
        dependency._MAX_PACKED_CULA_CHECKPOINT_BYTES = 32 << 20
        model = SimpleNamespace(
            layers=[SimpleNamespace(index=3, attention=KimiK3KDA(12))],
            kv_cache=SimpleNamespace(
                get_layer_cache_groups=lambda _: [
                    SimpleNamespace(seq_size_per_block=4096)
                ]
            ),
        )
        with mock.patch.dict(sys.modules, {dependency.__name__: dependency}):
            specs = list(self.warmup._fla_l2norm_specs(model, 65536))
        self.assertEqual([buckets for _, buckets in specs], [tuple(range(1, 13))])
        model.layers[0].attention = KimiK3KDA(96)
        model.kv_cache.get_layer_cache_groups = lambda _: [
            SimpleNamespace(seq_size_per_block=128)
        ]
        with mock.patch.dict(sys.modules, {dependency.__name__: dependency}):
            specs = list(self.warmup._fla_l2norm_specs(model, 65536))
        self.assertEqual([buckets for _, buckets in specs], [(1,)])

    def test_fla_deduplicates_compile_buckets_across_head_geometries(self):
        dependency = types.ModuleType(
            "rtp_llm.models_py.modules.factory.attention.cuda_kda_impl"
        )
        dependency._MAX_PACKED_CULA_CHECKPOINT_BYTES = 32 << 20
        model = SimpleNamespace(
            layers=[
                SimpleNamespace(index=i, attention=KimiK3KDA(heads))
                for i, heads in enumerate((12, 16))
            ],
            kv_cache=SimpleNamespace(
                get_layer_cache_groups=lambda _: [
                    SimpleNamespace(seq_size_per_block=4096)
                ]
            ),
        )
        with mock.patch.dict(sys.modules, {dependency.__name__: dependency}):
            specs = list(self.warmup._fla_l2norm_specs(model, 65536))
        self.assertEqual([buckets for _, buckets in specs], [tuple(range(1, 17))])

    def test_uses_existing_dsv4_bf16_values_and_real_fp8_grid(self):
        names = {
            "_DENSE_GEMM_FALLBACK_M_GRID",
            "_DENSE_GEMM_HEURISTIC_SCAN_LIMIT",
            "_SM100_DENSE_BLOCK_M_CANDIDATES",
            "_SM100_DENSE_BLOCK_N_CANDIDATES",
            "_ceil_div",
            "_align",
            "_sm100_swizzle_mode",
            "_sm100_dense_storage_signature",
            "_sm100_dense_layout_info",
            "_sm100_dense_layout_is_better",
            "_sm100_dense_layout_signature",
            "_candidate_dense_gemm_m_values",
            "_generate_dense_gemm_warmup_m_grid",
            "_dense_gemm_m_grid",
        }
        source_path = HERE / "baseline_dsv4_kernel_jit_warmup.py"
        if not source_path.exists():
            source_path = HERE.parent.parent / "dsv41" / "dsv4_kernel_jit_warmup.py"
        source = ast.parse(source_path.read_text())
        nodes = [
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef)
            and node.name in names
            or isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
        ]
        dependency = types.ModuleType(
            "rtp_llm.models_py.modules.dsv41.dsv4_kernel_jit_warmup"
        )
        dependency.lru_cache = lru_cache
        exec(
            compile(
                ast.Module(body=nodes, type_ignores=[]), "actual_dsv4_sampling", "exec"
            ),
            dependency.__dict__,
        )
        with mock.patch.dict(sys.modules, {dependency.__name__: dependency}):
            samples = self.warmup._sample_m_values(
                64, 128, 512, fp8=False, k_major=False, num_sms=148
            )
            fp8 = self.warmup._sample_m_values(
                65536, 128, 512, fp8=True, k_major=True, num_sms=148
            )
        self.assertEqual(samples, (1, 2, 4, 8, 16, 24, 32, 48, 64))
        self.assertTrue(fp8)
        self.assertEqual(fp8, tuple(sorted(set(fp8))))
        self.assertLessEqual(max(fp8), 65536)
        self.assertGreater(max(fp8), 8192)

    def test_nonanchor_sampling_includes_reachable_upper_endpoint(self):
        dependency = types.ModuleType(
            "rtp_llm.models_py.modules.dsv41.dsv4_kernel_jit_warmup"
        )
        dependency._DENSE_GEMM_FALLBACK_M_GRID = [
            1,
            2,
            4,
            8,
            16,
            24,
            32,
            48,
            64,
            96,
            128,
            192,
            256,
            384,
            512,
            768,
            1024,
        ]
        dependency._generate_dense_gemm_warmup_m_grid = lambda **_: ()
        dependency._dense_gemm_m_grid = lambda maximum: [
            m for m in dependency._DENSE_GEMM_FALLBACK_M_GRID if m <= maximum
        ] + [maximum]
        with mock.patch.dict(sys.modules, {dependency.__name__: dependency}):
            samples = self.warmup._sample_m_values(
                1003, 1024, 512, fp8=False, k_major=False, num_sms=148
            )
        self.assertEqual(samples[-1], 1003)

    def test_bf16_prefix_sampling_uses_budget_not_query_chunk(self):
        self.assertEqual(
            self._prepared_bf16_maximum("1.0", "torch.bfloat16", 1000000, 2), 419430
        )

    def test_lazy_fp8_kv_up_uses_historical_bound_and_fp8_launch_family(self):
        self.assertEqual(
            self._prepared_bf16_maximum(
                "6.0", "torch.float8_e4m3fn", 1048576, 1, fp8=True
            ),
            559240,
        )

    def test_lazy_fp8_unsupported_skip_still_prepares_prefix_outer_fallback(self):
        self.assertEqual(
            self._prepared_bf16_maximum(
                "6.0",
                "torch.float8_e4m3fn",
                1048576,
                1,
                fp8=True,
                supported=False,
            ),
            559240,
        )

    def test_lazy_fp8_unsupported_skip_cannot_bypass_large_memory_guard(self):
        with self.assertRaisesRegex(RuntimeError, "needs at least"):
            self._prepared_bf16_maximum(
                "6.0",
                "torch.float8_e4m3fn",
                1048576,
                1,
                fp8=True,
                supported=False,
                free=0,
            )

    def test_bf16_prefix_bound_uses_actual_cache_precision(self):
        self.assertEqual(
            self._prepared_bf16_maximum("1.0", "torch.float8_e4m3fn", 1000000, 2),
            279620,
        )

    def test_bf16_zero_budget_respects_request_batch_and_tp_padding(self):
        self.assertEqual(
            self._prepared_bf16_maximum("0", "torch.bfloat16", 500, 2), 1007
        )

    def test_bf16_positive_budget_is_clamped_to_request_batch_limit(self):
        self.assertEqual(
            self._prepared_bf16_maximum("1.0", "torch.bfloat16", 1000, 3), 3007
        )

    def test_full_warmup_skips_unsupported_backend_before_large_memory_check(self):
        try:
            prepared = self._prepared_bf16_maximum(
                "6.0", "torch.bfloat16", 1000000, 2, supported=False, free=0
            )
        except RuntimeError as error:
            self.fail(f"unsupported backend entered large-scratch preparation: {error}")
        self.assertIsNone(prepared)

    def test_unsupported_backend_is_checked_before_deriving_large_bound(self):
        self.assertIsNone(
            self._prepared_bf16_maximum(
                "unparsed", "torch.bfloat16", 1000000, 2, supported=False, free=0
            )
        )

    def _prepared_bf16_maximum(
        self, budget, dtype, sequence, batch, *, supported=True, free=1 << 50, fp8=False
    ):
        dependency = types.ModuleType(
            "rtp_llm.models_py.modules.dsv41.dsv4_kernel_jit_warmup"
        )
        dependency._get_deep_gemm_num_sms = lambda *_: 148
        dependency._release_cuda_cache = lambda *_: None
        dependency._run_deepgemm_warmup_launches_serialized = (
            lambda _, function: function()
        )
        switch = types.ModuleType("rtp_llm.utils.warmup")
        switch.model_warm_up_enabled = lambda: True
        plan = types.ModuleType(
            "rtp_llm.models_py.modules.factory.attention.mla_prefill_plan"
        )
        plan.math, plan.Sequence = math, Sequence
        plan_path = HERE / "baseline_mla_prefill_plan.py"
        if not plan_path.exists():
            plan_path = (
                HERE.parent.parent / "factory" / "attention" / "mla_prefill_plan.py"
            )
        nodes = [
            node
            for node in ast.parse(plan_path.read_text()).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "plan_expanded_kv_capacity"
        ]
        exec(
            compile(
                ast.Module(body=nodes, type_ignores=[]),
                "actual_prefix_capacity",
                "exec",
            ),
            plan.__dict__,
        )
        torch = types.ModuleType("torch")
        torch.inference_mode = lambda: lambda function: function
        torch.bfloat16 = "torch.bfloat16"
        torch.zeros = lambda shape, **_: SimpleNamespace(shape=shape)
        torch.cuda = SimpleNamespace(
            is_available=lambda: True,
            is_current_stream_capturing=lambda: False,
            synchronize=lambda *_: None,
            mem_get_info=lambda *_: (free, 1 << 50),
            get_device_capability=lambda *_: (10, 3),
        )

        def check_support(hidden, splits):
            self.assertEqual(hidden.shape, (1, 512))
            self.assertEqual(splits, (128, 64, 128))
            return supported

        projection = SimpleNamespace(
            weight=TensorMetadata(
                (3072 if fp8 else 1024, 512),
                dtype="torch.float8_e4m3fn" if fp8 else "torch.bfloat16",
            ),
            supports_skip_head_mid=check_support,
        )
        if fp8:
            projection.N, projection.K = 3072, 512
        cache = SimpleNamespace(
            seq_size_per_block=128, kv_cache_base=TensorMetadata((1, 128), dtype=dtype)
        )
        model = SimpleNamespace(
            parallelism_config=SimpleNamespace(tp_size=8),
            config=SimpleNamespace(
                moe_prefill_max_tokens_per_rank=65536, max_seq_len=sequence
            ),
            kv_cache=SimpleNamespace(get_layer_cache_groups=lambda _: [cache]),
        )
        seen = []
        with (
            mock.patch.dict(
                sys.modules, {m.__name__: m for m in (dependency, switch, torch, plan)}
            ),
            mock.patch.dict(
                os.environ, {"RTP_MLA_PREFILL_EXPANDED_KV_BUDGET_GIB": budget}
            ),
            mock.patch.object(self.warmup, "_fp8_projections", return_value=()),
            mock.patch.object(
                self.warmup,
                "_kv_up_projections",
                return_value=[(0, projection, (128, 64, 128))],
            ),
            mock.patch.object(self.warmup, "_fla_l2norm_specs", return_value=()),
            mock.patch.object(
                self.warmup,
                "_sample_m_values",
                side_effect=lambda maximum, *_args, **_kwargs: (maximum,),
            ),
            mock.patch.object(
                self.warmup,
                "_warmup_bf16",
                side_effect=lambda _, _s, samples: (
                    self.fail("FP8 KV-up sent to BF16 launch family")
                    if fp8
                    else seen.extend(samples) or True
                ),
            ),
            mock.patch.object(
                self.warmup,
                "_warmup_fp8",
                side_effect=lambda _, samples, **_kwargs: seen.extend(samples),
            ),
        ):
            self.warmup.warmup_kimi_k3_projection_jit(
                model,
                SimpleNamespace(is_decode_role=False, max_context_batch_size=batch),
            )
        return seen[-1] if seen else None

    def test_larger_projection_rejects_insufficient_free_memory_before_allocation(self):
        torch = types.ModuleType("torch")
        torch.cuda = SimpleNamespace(mem_get_info=lambda _: (4096, 8192))
        with mock.patch.dict(sys.modules, {"torch": torch}):
            self.warmup._check_projection_memory("cuda:0", 4096)
            with self.assertRaisesRegex(RuntimeError, "needs at least 8192 bytes"):
                self.warmup._check_projection_memory("cuda:0", 8192)

    def test_scratch_estimate_includes_quantized_input_scales_and_packed_output(self):
        # M=8 K=512 N=128: BF16 source+output 10240, FP8 source4096, SF128.
        self.assertEqual(
            self.warmup._projection_scratch_bytes(8, 512, 128, fp8=True), 14464
        )
        self.assertEqual(
            self.warmup._projection_scratch_bytes(8, 512, 320, fp8=False), 13312
        )

    def test_fp8_preparation_calls_actual_outer_dispatch_and_retries_failure(self):
        torch = types.ModuleType("torch")
        torch.bfloat16 = "torch.bfloat16"
        torch.zeros = lambda shape, **_: SimpleNamespace(shape=shape)
        module = CudaFp8GEMMLinear()
        with mock.patch.dict(sys.modules, {"torch": torch}):
            self.warmup._warmup_fp8(module, (1, 24, 65536))
        self.assertEqual(module.rows, [1, 24, 65536])
        with mock.patch.dict(sys.modules, {"torch": torch}):
            with mock.patch.object(
                torch, "zeros", side_effect=RuntimeError("allocation failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "allocation failed"):
                    self.warmup._warmup_fp8(module, (1,))
            self.warmup._warmup_fp8(module, (1,))
        self.assertEqual(module.rows[-1], 1)

    def test_disabled_preparation_has_no_tensor_or_dependency_work(self):
        dependency = types.ModuleType("rtp_llm.utils.warmup")
        dependency.model_warm_up_enabled = lambda: False
        with mock.patch.dict(
            sys.modules, {dependency.__name__: dependency, "torch": None}
        ):
            self.warmup.warmup_kimi_k3_projection_jit(object(), object())

    def test_directed_failure_prevents_successful_startup(self):
        torch = types.ModuleType("torch")
        torch.inference_mode = lambda: lambda function: function
        torch.cuda = SimpleNamespace(
            is_available=lambda: True, is_current_stream_capturing=lambda: False
        )
        replay = types.ModuleType("rtp_llm.models_py.triton_kernels.linear_replay")
        replay.finalize_linear_replay = replay.linear_serial_replay = (
            lambda *args, **kwargs: None
        )
        switch = types.ModuleType("rtp_llm.utils.warmup")
        switch.model_warm_up_enabled = lambda: True
        mla = types.ModuleType("rtp_llm.models_py.modules.kimi_k3.mla_jit_warmup")
        mla.warmup_kimi_k3_mla_jit = lambda *args: None
        directed = types.ModuleType(
            "rtp_llm.models_py.modules.kimi_k3.bounded_projection_jit_warmup"
        )

        def fail(*args):
            raise RuntimeError("directed compile failed")

        directed.warmup_kimi_k3_projection_jit = fail
        startup_path = MODULE_PATH.parent / "kernel_jit_warmup.py"
        if not startup_path.exists():
            startup_path = HERE / "baseline_kernel_jit_warmup.py"
        with mock.patch.dict(
            sys.modules, {m.__name__: m for m in (torch, replay, switch, mla, directed)}
        ):
            spec = importlib.util.spec_from_file_location(
                "isolated_k3_startup", startup_path
            )
            startup = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(startup)
            startup._warmup_replay = lambda *args: None
            with self.assertRaisesRegex(RuntimeError, "directed compile failed"):
                startup.warmup_kimi_k3_kernel_jit(object(), object())

    def test_no_cuda_preparation_does_not_import_dense_or_fla_dependencies(self):
        dependency = types.ModuleType("rtp_llm.utils.warmup")
        dependency.model_warm_up_enabled = lambda: True
        torch = types.ModuleType("torch")
        torch.cuda = SimpleNamespace(is_available=lambda: False)
        with mock.patch.dict(
            sys.modules,
            {
                dependency.__name__: dependency,
                "torch": torch,
                "rtp_llm.models_py.modules.dsv41.dsv4_kernel_jit_warmup": None,
                "fla.modules.l2norm": None,
            },
        ):
            self.warmup.warmup_kimi_k3_projection_jit(object(), object())

    def test_bf16_collection_uses_raw_kv_up_weight_and_actual_head_splits(self):
        dependency = types.ModuleType(
            "rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear"
        )
        dependency.CudaF16Linear = lambda weight: SimpleNamespace(raw_weight=weight)
        weights = types.ModuleType("rtp_llm.utils.model_weight")
        weights.W = SimpleNamespace(mla_kv_b_w="kv_up")
        attention = type("KimiK3MLA", (), {})()
        raw = TensorMetadata((512, 3072), dtype="torch.bfloat16")
        model = SimpleNamespace(
            config=SimpleNamespace(
                attn_config=SimpleNamespace(
                    nope_head_dim=128, rope_head_dim=64, v_head_dim=128
                )
            ),
            layers=[SimpleNamespace(index=i, attention=attention) for i in (0, 2)],
            weight=SimpleNamespace(weights=[{"kv_up": raw}] * 3),
        )
        with mock.patch.dict(
            sys.modules, {dependency.__name__: dependency, weights.__name__: weights}
        ):
            specs = list(self.warmup._kv_up_projections(model))
        self.assertEqual(len(specs), 1)
        self.assertIs(specs[0][1].raw_weight, raw)
        self.assertEqual(specs[0][2], (128, 64, 128))

    def test_lazy_fp8_collection_preserves_loaded_scales_and_deduplicates_layout(self):
        dependency = types.ModuleType(
            "rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear"
        )
        dependency.CudaF16Linear = lambda _: self.fail("FP8 weight used BF16 wrapper")
        factory = types.ModuleType("rtp_llm.models_py.modules.factory.linear.factory")
        quant = object()

        def create(weights, weight_key, scale_key, bias, config):
            self.assertIsNone(bias)
            self.assertIs(config, quant)
            projection = CudaFp8GEMMLinear()
            projection.weight = weights[weight_key]
            projection.weight_scales = weights[scale_key]
            projection.N, projection.K = projection.weight.shape
            return projection

        factory.LinearFactory = SimpleNamespace(create_linear_from_weights=create)
        weights = types.ModuleType("rtp_llm.utils.model_weight")
        weights.W = SimpleNamespace(mla_kv_b_w="kv_up", mla_kv_b_s="kv_scale")
        raw = TensorMetadata((3072, 512))
        scales = TensorMetadata((3072, 1), strides=(1, 3072), dtype="torch.int32")
        other = TensorMetadata((3072, 1), strides=(1, 6144), dtype="torch.int32")
        model = SimpleNamespace(
            config=SimpleNamespace(
                attention_projection_quant_config=quant,
                attn_config=SimpleNamespace(
                    nope_head_dim=128, rope_head_dim=64, v_head_dim=128
                ),
            ),
            layers=[
                SimpleNamespace(index=i, attention=type("KimiK3MLA", (), {})())
                for i in range(3)
            ],
            weight=SimpleNamespace(
                weights=[
                    {"kv_up": raw, "kv_scale": scales},
                    {"kv_up": raw, "kv_scale": scales},
                    {"kv_up": raw, "kv_scale": other},
                ]
            ),
        )
        with mock.patch.dict(
            sys.modules, {m.__name__: m for m in (dependency, factory, weights)}
        ):
            specs = list(self.warmup._kv_up_projections(model))
        self.assertEqual(
            len(specs), 2, "lazy FP8 KV-up must not disappear from preparation"
        )
        self.assertIs(specs[0][1].weight, raw)
        self.assertIs(specs[0][1].weight_scales, scales)
        self.assertIs(specs[1][1].weight_scales, other)

    def test_fp8_kv_up_preparation_uses_skip_head_or_real_outer_fallback(self):
        torch = types.ModuleType("torch")
        torch.bfloat16 = "torch.bfloat16"
        torch.zeros = lambda shape, **_: SimpleNamespace(shape=shape)
        for supported in (True, False):
            with self.subTest(supported=supported):
                projection = CudaFp8GEMMLinear()
                packed = []
                projection.supports_skip_head_mid = lambda *_: supported
                projection.forward_skip_head_mid = lambda hidden, splits: packed.append(
                    (hidden.shape[0], splits)
                )
                with mock.patch.dict(sys.modules, {"torch": torch}):
                    try:
                        self.warmup._warmup_fp8(
                            projection, (1, 1003), head_splits=(128, 64, 128)
                        )
                    except TypeError as error:
                        self.fail(f"FP8 KV-up launch path is missing: {error}")
                self.assertEqual(
                    packed,
                    [(1, (128, 64, 128)), (1003, (128, 64, 128))] if supported else [],
                )
                self.assertEqual(projection.rows, [] if supported else [1, 1003])

    def test_unsupported_bf16_backend_does_not_launch_skip_head_projection(self):
        torch = types.ModuleType("torch")
        torch.bfloat16 = "torch.bfloat16"
        torch.zeros = lambda shape, **_: SimpleNamespace(shape=shape)
        projection = SimpleNamespace(
            weight=TensorMetadata((3072, 512), dtype="torch.bfloat16"),
            supports_skip_head_mid=lambda *_: False,
            forward_skip_head_mid=lambda *_: self.fail("unsupported backend launched"),
        )
        with mock.patch.dict(sys.modules, {"torch": torch}):
            self.assertFalse(
                self.warmup._warmup_bf16(projection, (128, 64, 128), (1, 65536))
            )

    def test_fla_uses_fixed_tiny_storage_with_installed_kernel_launch_metadata(self):
        calls = []

        class Kernel:
            def __getitem__(self, grid):
                def launch(**kwargs):
                    calls.append((grid, kwargs))

                return launch

        dependency = types.ModuleType("fla.modules.l2norm")
        dependency.l2norm_fwd_kernel = Kernel()
        torch = types.ModuleType("torch")
        torch.bfloat16, torch.float32 = "bf16", "fp32"
        torch.zeros = torch.empty = lambda shape, **_: SimpleNamespace(shape=shape)
        torch.empty_like = lambda tensor: SimpleNamespace(shape=tensor.shape)
        with mock.patch.dict(
            sys.modules, {dependency.__name__: dependency, "torch": torch}
        ):
            self.warmup._warmup_fla("cuda:0", (1, 12, 64))
        self.assertEqual([args["NB"] for _, args in calls], [1, 12, 64])
        for grid, args in calls:
            self.assertEqual(
                (grid, args["x"].shape, args["rstd"].shape), ((1,), (8, 128), (8,))
            )
            self.assertEqual(
                (args["T"], args["D"], args["BD"], args["BT"]), (8, 128, 128, 8)
            )
            self.assertEqual(
                (args["num_warps"], args["num_stages"], args["num_ctas"]), (1, 3, 1)
            )


if __name__ == "__main__":
    unittest.main()
