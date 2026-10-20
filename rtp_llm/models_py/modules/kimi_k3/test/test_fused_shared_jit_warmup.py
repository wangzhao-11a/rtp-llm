"""Exercise K3 shared-expert preparation without starting a distributed GPU job."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.kimi_k3 import moe


class _CpuPacker:
    name = "reference"

    def pack(self, x, routing, ids, buf, tokens):
        buf.x[:tokens].copy_(x)
        buf.topk_weights[:tokens].copy_(routing)
        buf.topk_idx[:tokens].copy_(ids)


class FusedSharedJitWarmupTest(unittest.TestCase):
    def _experts(self, *, capacity=32, shared_hidden=256):
        import deep_gemm

        # Keep production tensor preparation and forward real; only the external
        # GPU pack/DeepGEMM collective and CUDA synchronization are replaced.
        experts = moe._FusedSharedExperts.__new__(moe._FusedSharedExperts)
        torch.nn.Module.__init__(experts)
        experts.cfg = SimpleNamespace(
            ep_size=2,
            n_routed_experts=16,
            n_local_experts=8,
            n_activated_experts=2,
            local_expert_start=8,
            dim=128,
            moe_inter_dim=256,
            warmup_include_capacity=False,
            layer_id=3,
        )
        experts._max_tokens_per_rank = capacity
        experts._shared_intermediate = 256
        experts.buf = SimpleNamespace(
            num_max_tokens_per_rank=capacity,
            x=torch.empty(capacity, 128, dtype=torch.bfloat16),
            x_sf=torch.empty(capacity, 1),
            topk_weights=torch.empty(capacity, 2),
            topk_idx=torch.empty(capacity, 2, dtype=torch.long),
        )
        experts.routed_y = torch.empty(capacity, 128, dtype=torch.bfloat16)
        experts.shared_x = torch.empty(capacity, shared_hidden, dtype=torch.bfloat16)
        experts.shared_y = torch.empty_like(experts.shared_x)
        experts.routed_l1, experts.routed_l2 = object(), object()
        experts.shared_l1, experts.shared_l2 = object(), object()
        experts._mega_kernel = lambda *args, **kwargs: deep_gemm.fp8_fp4_mega_moe_k3(
            *args, **kwargs
        )
        experts.packer = _CpuPacker()
        experts.beta, experts.linear_beta, experts.fp4_block = 4.0, 25.0, 32
        return experts

    def test_prepares_real_shared_forward_once_and_separates_shared_geometry(self):
        import deep_gemm

        calls = []

        def launch(y, l1, l2, buf, **kwargs):
            tokens = y.shape[0]
            self.assertEqual(kwargs["activation"], "situ")
            self.assertEqual(kwargs["activation_beta"], 4.0)
            self.assertEqual(kwargs["activation_linear_beta"], 25.0)
            self.assertEqual(kwargs["recipe"], (1, 1, 32))
            self.assertEqual(tuple(buf.x[:tokens].shape), (tokens, 128))
            self.assertTrue(
                torch.equal(
                    buf.topk_idx[:tokens], torch.tensor([8, 9]).expand(tokens, -1)
                )
            )
            self.assertEqual(torch.count_nonzero(buf.topk_weights[:tokens]).item(), 0)
            y.zero_()
            kwargs["shared_y"][:tokens].copy_(kwargs["shared_x"][:tokens])
            calls.append((tokens, kwargs["shared_y"].shape[1]))

        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.dict(
                os.environ,
                {
                    "MODEL_WARM_UP": "1",
                    "MEGA_MOE_JIT_WARMUP_TOKENS": "1,7,32",
                    "MEGA_MOE_NVCC_TMPDIR": temporary,
                },
            ),
            mock.patch.object(deep_gemm, "get_num_sms", return_value=148),
            mock.patch.object(deep_gemm, "fp8_fp4_mega_moe_k3", side_effect=launch),
            mock.patch("torch.cuda.is_current_stream_capturing", return_value=False),
            mock.patch("torch.cuda.synchronize"),
            mock.patch("torch.distributed.barrier"),
            mock.patch("torch.distributed.is_initialized", return_value=False),
        ):
            moe._FUSED_SHARED_JIT_WARMED_KEYS.clear()
            self.addCleanup(moe._FUSED_SHARED_JIT_WARMED_KEYS.clear)
            self._experts()._maybe_warmup_jit_once()
            count = len(calls)
            self.assertGreater(count, 0)
            self._experts()._maybe_warmup_jit_once()
            self.assertEqual(len(calls), count)
            self._experts(shared_hidden=512)._maybe_warmup_jit_once()
            self.assertGreater(len(calls), count)
        self.assertEqual(
            set(calls), {(1, 256), (7, 256), (32, 256), (1, 512), (7, 512), (32, 512)}
        )

    def test_disabled_or_capturing_never_marks_prepared(self):
        experts = self._experts()
        with mock.patch.dict(os.environ, {"MODEL_WARM_UP": "0"}):
            experts._maybe_warmup_jit_once()
        with (
            mock.patch.dict(os.environ, {"MODEL_WARM_UP": "1"}),
            mock.patch("torch.cuda.is_current_stream_capturing", return_value=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "capture"):
                experts._maybe_warmup_jit_once()

    def test_fused_pack_prepares_both_valid_mask_layouts(self):
        import deep_gemm

        from rtp_llm.models_py.triton_kernels.moe import mega_moe_input_pack

        experts = self._experts()
        experts.packer = SimpleNamespace(name="fused")
        masks = []

        def pack(x, routing, ids, out, scales, out_ids, out_routing, **kwargs):
            masks.append(kwargs["valid_mask"] is not None)
            out.copy_(x)
            out_ids.copy_(ids)
            out_routing.copy_(routing)
            kwargs["shared_out"].copy_(kwargs["shared_input"])

        with (
            mock.patch.dict(os.environ, {"MODEL_WARM_UP": "1"}),
            mock.patch.object(
                mega_moe_input_pack, "fused_pack_mega_moe_inputs", side_effect=pack
            ),
            mock.patch.object(deep_gemm, "fp8_fp4_mega_moe_k3"),
            mock.patch("torch.distributed.barrier"),
            mock.patch("torch.cuda.synchronize"),
            mock.patch("torch.cuda.is_current_stream_capturing", return_value=False),
        ):
            experts.warmup_jit([1, 7])
        self.assertEqual(masks, [False, True, False, True])

    def test_automatic_plan_includes_shared_pack_block_boundary(self):
        experts = self._experts(capacity=2048)
        experts.packer = SimpleNamespace(name="fused")
        with mock.patch.dict(os.environ, clear=True):
            tokens = experts._resolve_jit_warmup_token_counts(148)
        self.assertIn(1, tokens)
        self.assertIn(2048, tokens)
        self.assertTrue(all(1 <= token <= 2048 for token in tokens))
        with mock.patch.dict(
            os.environ, {"MEGA_MOE_JIT_WARMUP_TOKENS": "7,9999"}, clear=True
        ):
            self.assertEqual(experts._resolve_jit_warmup_token_counts(148), [7, 2048])

    def test_failed_compile_can_retry_and_restores_tmpdir(self):
        import deep_gemm

        experts = self._experts()
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.dict(
                os.environ,
                {
                    "MODEL_WARM_UP": "1",
                    "MEGA_MOE_JIT_WARMUP_TOKENS": "1",
                    "MEGA_MOE_NVCC_TMPDIR": temporary,
                    "TMPDIR": "/before",
                },
            ),
            mock.patch.object(deep_gemm, "get_num_sms", return_value=148),
            mock.patch.object(
                deep_gemm,
                "fp8_fp4_mega_moe_k3",
                side_effect=RuntimeError("compiler failed"),
            ),
            mock.patch("torch.cuda.is_current_stream_capturing", return_value=False),
            mock.patch("torch.distributed.barrier"),
            mock.patch("torch.distributed.is_initialized", return_value=False),
        ):
            moe._FUSED_SHARED_JIT_WARMED_KEYS.clear()
            self.addCleanup(moe._FUSED_SHARED_JIT_WARMED_KEYS.clear)
            with self.assertRaisesRegex(RuntimeError, "compiler failed"):
                experts._maybe_warmup_jit_once()
            self.assertEqual(os.environ["TMPDIR"], "/before")
            self.assertFalse(moe._FUSED_SHARED_JIT_WARMED_KEYS)


if __name__ == "__main__":
    unittest.main()
