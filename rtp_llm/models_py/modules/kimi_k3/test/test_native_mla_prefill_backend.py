"""K3 backend setup and MLA JIT preparation preserve runtime contracts."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

SOURCE = Path(__file__).resolve().parents[1] / "tokenspeed_prefill.py"


class NativeMlaPrefillBackendTest(unittest.TestCase):
    def test_reuses_backend_setup_for_fp8_instances(self):
        backend = types.ModuleType("tokenspeed_mla")
        backend.__path__ = []
        prefill = types.ModuleType("tokenspeed_mla.mla_prefill")
        token_speed = lambda **_kwargs: None
        prefill.tokenspeed_mla_prefill = token_speed
        cutlass = types.ModuleType("rtp_llm.models_py.utils.cutlass")
        setup = unittest.mock.Mock()
        cutlass.setup_cutlass_import_path = setup
        fake_modules = {
            "tokenspeed_mla": backend,
            "tokenspeed_mla.mla_prefill": prefill,
            "rtp_llm.models_py.utils.cutlass": cutlass,
        }
        with (
            patch.dict(sys.modules, fake_modules),
            patch(
                "importlib.metadata.version", return_value="test-version"
            ) as get_version,
        ):
            spec = importlib.util.spec_from_file_location(
                "k3_prefill_backend_under_test", SOURCE
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            first = module.TokenspeedMlaPrefill(8, 0.125)
            second = module.TokenspeedMlaPrefill(16, 0.25)

        self.assertIs(first._run, token_speed)
        self.assertIs(second._run, token_speed)
        self.assertEqual(first.operand_dtype, torch.float8_e4m3fn)
        self.assertEqual(second.operand_dtype, torch.float8_e4m3fn)
        setup.assert_called_once_with()
        get_version.assert_called_once_with("tokenspeed-mla")


def load_warmup():
    from rtp_llm.models_py.modules.kimi_k3 import mla_jit_warmup

    spec = importlib.util.spec_from_file_location(
        "k3_mla_warmup_under_test", mla_jit_warmup.__file__
    )
    module = importlib.util.module_from_spec(spec)
    missing = object()
    previous = sys.modules.get(spec.name, missing)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is missing:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
    return module


def model_fixture(dtype=torch.bfloat16, page_rr=True):
    attention = type("KimiK3MLA", (), {})()
    attention.heads, attention.kv_rank = 3, 16
    attention.suffix_dim, attention.q_dim, attention.v_dim = 8, 192, 128
    storage = torch.ones((7, 64, 48), dtype=dtype)
    live = storage[:, :, :24]
    cache = SimpleNamespace(kv_cache_base=live, seq_size_per_block=64, tag="mla")
    model = SimpleNamespace(
        layers=[SimpleNamespace(index=i, attention=attention) for i in range(2)],
        kv_cache=SimpleNamespace(
            get_layer_cache_groups=lambda _index: [cache],
            get_seq_size_per_block=lambda _tag: 256,
        ),
        parallelism_config=SimpleNamespace(
            tp_size=2,
            tp_rank=1,
            prefill_cp_config=SimpleNamespace(kv_cache_sharded=page_rr),
        ),
        config=SimpleNamespace(attn_config=SimpleNamespace(softmax_extra_scale=1.25)),
    )
    return model, live


class MlaJitWarmupContractTest(unittest.TestCase):
    def test_geometry_preserves_cache_stride_and_deduplicates_layers(self):
        warmup = load_warmup()
        model, live = model_fixture()
        specs = list(warmup._mla_warmup_specs(model))
        self.assertEqual(len(specs), 1)
        spec = specs[0]
        self.assertEqual(spec.cache_stride, live.stride())
        self.assertEqual(spec.page_tokens, 64)
        self.assertEqual((spec.heads, spec.latent, spec.rope), (3, 16, 8))
        self.assertEqual(spec.softmax_scale, 1.25 / 192**0.5)

    def test_non_page_rr_models_do_no_prefix_preparation(self):
        warmup = load_warmup()
        model, _ = model_fixture(page_rr=False)
        self.assertEqual(list(warmup._mla_warmup_specs(model)), [])

    def test_page_rr_spec_uses_physical_pages_and_this_rank(self):
        warmup = load_warmup()
        model, _ = model_fixture(page_rr=True)
        spec = next(warmup._mla_warmup_specs(model))
        self.assertEqual(spec.physical_page_tokens, 256)
        self.assertEqual((spec.shards, spec.rank), (2, 1))

    def test_page_rr_preparation_uses_scratch_and_never_enters_collectives(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
            mla_page_rr_cache_gather as gather,
        )

        warmup = load_warmup()
        model, live = model_fixture()
        spec = next(warmup._mla_warmup_specs(model))
        before = live.clone()
        payload = torch.ones(
            (1, spec.physical_page_tokens, spec.latent + spec.rope),
            dtype=spec.cache_dtype,
        )
        with (
            patch.object(
                gather.MlaPageRRCacheGather,
                "_pack_prefix_chunk",
                autospec=True,
                return_value=payload,
            ) as pack,
            patch.object(gather, "_restore_mla_page_rr_prefix") as restore,
            patch.object(
                gather.collective_torch,
                "all_gather_into",
                side_effect=AssertionError("warmup must not enter a collective"),
            ),
        ):
            warmup._warmup_page_rr_prefix(spec)

        adapter, scratch, table, descriptor = pack.call_args.args
        self.assertEqual(scratch.stride(), live.stride())
        self.assertNotEqual(scratch.data_ptr(), live.data_ptr())
        self.assertEqual(adapter.table_layout, "compact")
        self.assertEqual(descriptor.prefix_starts, (spec.rank * 256,))
        self.assertEqual(descriptor.prefix_lens, (1,))
        self.assertEqual(table.tolist(), [[1, 2, 3, 4]])
        gathered, restored_descriptor = restore.call_args.args
        self.assertIs(restored_descriptor, descriptor)
        torch.testing.assert_close(gathered[spec.rank], payload, rtol=0, atol=0)
        torch.testing.assert_close(gathered[0], torch.zeros_like(payload), rtol=0, atol=0)
        torch.testing.assert_close(live, before, rtol=0, atol=0)

    def test_decode_and_unbound_models_do_no_prefix_preparation(self):
        warmup = load_warmup()
        model, _ = model_fixture()
        with patch.object(warmup, "_warmup_spec") as prepare:
            warmup.warmup_kimi_k3_mla_jit(model, SimpleNamespace(is_decode_role=True))
            model.kv_cache = None
            warmup.warmup_kimi_k3_mla_jit(model, SimpleNamespace(is_decode_role=False))
        prepare.assert_not_called()

    def test_preparation_is_once_per_geometry_and_keeps_live_cache_unchanged(self):
        warmup = load_warmup()
        model, live = model_fixture(torch.float8_e4m3fn)
        before = live.to(torch.float32).clone()
        with patch.object(warmup, "_warmup_spec") as prepare:
            warmup.warmup_kimi_k3_mla_jit(model, SimpleNamespace(is_decode_role=False))
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(prepare.call_args.args[0].cache_dtype, torch.float8_e4m3fn)
        torch.testing.assert_close(live.to(torch.float32), before, rtol=0, atol=0)

    def test_fp8_partial_attention_prepares_causal_and_noncausal_lse(self):
        from rtp_llm.models_py.modules.kimi_k3 import tokenspeed_prefill

        warmup = load_warmup()
        model, _ = model_fixture(torch.float8_e4m3fn)
        spec = next(warmup._mla_warmup_specs(model))
        launches = []

        def backend(**kwargs):
            launches.append(kwargs)

        with patch.object(
            tokenspeed_prefill, "_load_backend", return_value=(backend, "test")
        ):
            warmup._warmup_tokenspeed(spec)
        self.assertEqual([call["is_causal"] for call in launches], [True, False])
        for call in launches:
            self.assertTrue(call["return_lse"])
            self.assertEqual(call["query"].shape, (2, spec.heads, spec.q_dim))
            self.assertEqual(call["value"].shape, (2, spec.heads, spec.v_dim))
            self.assertEqual(call["query"].dtype, torch.float8_e4m3fn)
            self.assertEqual(call["softmax_scale"], 1.25 / spec.q_dim**0.5)


if __name__ == "__main__":
    unittest.main()
