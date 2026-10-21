import unittest

import torch
from rtp_llm.models_py.ascend.linear_attention import causal_conv1d, gated_delta_rule
from rtp_llm.models_py.utils.typed_storage_view import LinearCacheConverter
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_recurrent_gated_delta_rule,
)


class LinearAttentionReferenceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        torch.set_num_threads(1)
        # Deliberately unequal K/V widths: square model states hide transposes.
        self.tokens, self.h, self.hv, self.k, self.v = 9, 2, 4, 3, 5
        self.q = torch.randn(self.tokens, self.h, self.k)
        self.key = torch.randn_like(self.q)
        self.value = torch.randn(self.tokens, self.hv, self.v)
        self.g = -torch.rand(self.tokens, self.hv)
        self.beta = torch.rand_like(self.g)

    def reference(self, start, end, initial=None):
        result, state = torch_recurrent_gated_delta_rule(
            self.q[start:end].repeat_interleave(2, 1).unsqueeze(0),
            self.key[start:end].repeat_interleave(2, 1).unsqueeze(0),
            self.value[start:end].unsqueeze(0),
            self.g[start:end].unsqueeze(0),
            self.beta[start:end].unsqueeze(0),
            initial_state=initial,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        return result[0], state[0].transpose(-1, -2)

    def run_adapter(self, offsets, initial=None):
        return gated_delta_rule(
            self.q,
            self.key,
            self.value,
            self.g,
            self.beta,
            torch.tensor(offsets),
            initial,
        )

    def test_packed_sequences_do_not_share_state(self):
        initial = torch.randn(3, self.hv, self.v, self.k)
        saved = initial.clone()
        actual, final = self.run_adapter([0, 4, 4, 9], initial)
        for row, start, end in ((0, 0, 4), (2, 4, 9)):
            expected, expected_state = self.reference(
                start, end, initial[row : row + 1].transpose(-1, -2)
            )
            torch.testing.assert_close(
                actual[start:end], expected, atol=2e-6, rtol=2e-5
            )
            torch.testing.assert_close(final[row], expected_state, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(final[1], initial[1], atol=0, rtol=0)
        torch.testing.assert_close(initial, saved, atol=0, rtol=0)
        self.assertTrue(final.is_contiguous())

    def test_prefill_then_single_token_decode_matches_continuous(self):
        full, full_state = self.reference(0, self.tokens)
        for split in (1, 5, 8):
            prefill, state = gated_delta_rule(
                self.q[:split],
                self.key[:split],
                self.value[:split],
                self.g[:split],
                self.beta[:split],
                torch.tensor([0, split]),
            )
            outputs = [prefill]
            for token in range(split, self.tokens):
                result, state = gated_delta_rule(
                    self.q[token : token + 1],
                    self.key[token : token + 1],
                    self.value[token : token + 1],
                    self.g[token : token + 1],
                    self.beta[token : token + 1],
                    torch.tensor([0, 1]),
                    state,
                )
                outputs.append(result)
            torch.testing.assert_close(torch.cat(outputs), full, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(state[0], full_state, atol=2e-6, rtol=2e-5)

    def test_rejects_wrong_layout_and_invalid_packing(self):
        for offsets in ([1, 9], [0, 8], [0, 7, 4, 9]):
            with self.assertRaises(ValueError):
                self.run_adapter(offsets)
        with self.assertRaisesRegex(ValueError, "V-first"):
            self.run_adapter([0, 9], torch.zeros(1, self.hv, self.k, self.v))

    def test_conv_continuation_and_prefix_state_immutability(self):
        channels, width = 7, 4
        inputs = torch.randn(9, channels)
        weight = torch.randn(channels, width)
        bias = torch.randn(channels)
        whole, final = causal_conv1d(inputs, weight, torch.tensor([0, 9]), bias=bias)
        prefix, state = causal_conv1d(
            inputs[:5], weight, torch.tensor([0, 5]), bias=bias
        )
        saved = state.clone()
        suffix, last = causal_conv1d(
            inputs[5:], weight, torch.tensor([0, 4]), state, bias
        )
        torch.testing.assert_close(torch.cat([prefix, suffix]), whole)
        torch.testing.assert_close(last, final)
        torch.testing.assert_close(state, saved, atol=0, rtol=0)
        torch.testing.assert_close(final[0], inputs[-3:], atol=0, rtol=0)

    def test_cache_bytes_preserve_both_states_for_decode(self):
        # Use BF16 state/cache storage as required by RTP's Qwen3.5 config.
        channels, width = 2 * self.h * self.k + self.hv * self.v, 4
        converter = LinearCacheConverter(
            local_num_v_heads=self.hv,
            head_v_dim=self.v,
            head_k_dim=self.k,
            ssm_state_dtype=torch.bfloat16,
            linear_conv_kernel_dim=width,
            qkv_size=channels,
            conv_state_dtype=torch.bfloat16,
        )
        p_cache = torch.zeros(
            2, converter.block_size_bytes // 2 + 32, dtype=torch.bfloat16
        )
        p_ssm = converter.get_ssm_state_tensor(p_cache)
        p_conv = converter.get_conv_state_tensor(p_cache)
        inputs = torch.randn(9, channels).bfloat16()
        weight = (0.2 * torch.randn(channels, width)).bfloat16()
        _, conv_state = causal_conv1d(inputs[:8], weight, torch.tensor([0, 8]))
        _, ssm_state = gated_delta_rule(
            self.q[:8],
            self.key[:8],
            self.value[:8],
            self.g[:8],
            self.beta[:8],
            torch.tensor([0, 8]),
        )
        p_ssm[1].copy_(ssm_state[0])
        p_conv[1].copy_(conv_state[0])
        # A new backing allocation represents the D side, including block padding.
        wire = p_cache[1].view(torch.uint8).clone()
        d_cache = torch.zeros_like(p_cache)
        d_cache[1].view(torch.uint8).copy_(wire)
        d_ssm = converter.get_ssm_state_tensor(d_cache)
        d_conv = converter.get_conv_state_tensor(d_cache)
        self.assertNotEqual(p_cache.data_ptr(), d_cache.data_ptr())
        for p, d in ((p_ssm, d_ssm), (p_conv, d_conv)):
            torch.testing.assert_close(p[1], d[1], atol=0, rtol=0)
        for cache_ssm, cache_conv in ((p_ssm, p_conv), (d_ssm, d_conv)):
            conv, next_conv = causal_conv1d(
                inputs[8:], weight, torch.tensor([0, 1]), cache_conv[1:2]
            )
            query, key, value = torch.split(
                conv, [self.h * self.k, self.h * self.k, self.hv * self.v], dim=-1
            )
            output, next_ssm = gated_delta_rule(
                query.view(1, self.h, self.k),
                key.view(1, self.h, self.k),
                value.view(1, self.hv, self.v),
                self.g[8:],
                self.beta[8:],
                torch.tensor([0, 1]),
                cache_ssm[1:2],
            )
            if cache_ssm is p_ssm:
                expected = (output, next_ssm, next_conv)
            else:
                for actual, reference in zip((output, next_ssm, next_conv), expected):
                    torch.testing.assert_close(actual, reference, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
