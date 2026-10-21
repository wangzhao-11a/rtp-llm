import unittest

import torch

from rtp_llm.models_py.ascend.linear_attention import causal_conv1d, gated_delta_rule
from rtp_llm.models_py.ascend.linear_cache import linear_attention_with_cache
from rtp_llm.models_py.utils.typed_storage_view import LinearCacheConverter


class LinearCacheReferenceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        torch.set_num_threads(1)
        self.h, self.hv, self.k, self.v, self.width = 2, 4, 3, 5, 4
        self.channels = 2 * self.h * self.k + self.hv * self.v
        self.inputs = torch.randn(12, self.channels)
        self.weight = 0.2 * torch.randn(self.channels, self.width)
        self.g = -torch.rand(12, self.hv)
        self.beta = torch.rand_like(self.g)
        self.cache, self.ssm, self.conv = self.make_cache(torch.float32)

    def make_cache(self, dtype):
        converter = LinearCacheConverter(
            local_num_v_heads=self.hv,
            head_v_dim=self.v,
            head_k_dim=self.k,
            ssm_state_dtype=dtype,
            linear_conv_kernel_dim=self.width,
            qkv_size=self.channels,
            conv_state_dtype=dtype,
        )
        # Nontrivial block stride: adjacent states share a padded backing buffer.
        cache = torch.full(
            (12, converter.block_size_bytes // dtype.itemsize + 16),
            17.0,
            dtype=dtype,
        )
        return (
            cache,
            converter.get_ssm_state_tensor(cache),
            converter.get_conv_state_tensor(cache),
        )

    def reference(self, start, end, ssm=None, conv=None):
        lengths = torch.tensor([0, end - start])
        mixed, conv = causal_conv1d(self.inputs[start:end], self.weight, lengths, conv)
        q, k, v = mixed.split(
            [self.h * self.k, self.h * self.k, self.hv * self.v], dim=-1
        )
        result, ssm = gated_delta_rule(
            q.reshape(-1, self.h, self.k),
            k.reshape(-1, self.h, self.k),
            v.reshape(-1, self.hv, self.v),
            self.g[start:end],
            self.beta[start:end],
            lengths,
            ssm,
        )
        return result, ssm, conv

    def run_cached(self, start, end, offsets, prefixes, table, ssm=None, conv=None):
        return linear_attention_with_cache(
            self.inputs[start:end],
            self.weight,
            self.g[start:end],
            self.beta[start:end],
            torch.tensor(offsets),
            torch.tensor(prefixes),
            torch.tensor(table),
            self.ssm if ssm is None else ssm,
            self.conv if conv is None else conv,
            4,
        )

    def test_every_block_snapshot_matches_independent_prefix(self):
        saved = self.cache.clone()
        actual = self.run_cached(0, 11, [0, 11], [0], [[3, 7, 2]])
        expected, _, _ = self.reference(0, 11)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
        for length, block in ((4, 3), (8, 7), (11, 2)):
            _, ssm, conv = self.reference(0, length)
            torch.testing.assert_close(self.ssm[block], ssm[0], atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(self.conv[block], conv[0], atol=0, rtol=0)
        for block in (0, 1, 4, 5, 6, 8, 9, 10, 11):
            torch.testing.assert_close(self.cache[block], saved[block], atol=0, rtol=0)
        torch.testing.assert_close(self.cache[:, -16:], saved[:, -16:], atol=0, rtol=0)

    def test_shared_prefix_and_empty_sequence_are_not_overwritten(self):
        self.run_cached(0, 4, [0, 4], [0], [[3]])
        prefix = self.cache[3].clone()
        initial_ssm = self.ssm[3:4].clone()
        initial_conv = self.conv[3:4].clone()
        actual = self.run_cached(
            4, 11, [0, 3, 3, 7], [4, 0, 4], [[3, 7], [0, 0], [3, 2]]
        )
        for start, end, row_start, row_end, block in (
            (4, 7, 0, 3, 7),
            (7, 11, 3, 7, 2),
        ):
            expected, ssm, conv = self.reference(start, end, initial_ssm, initial_conv)
            torch.testing.assert_close(actual[row_start:row_end], expected)
            torch.testing.assert_close(self.ssm[block], ssm[0])
            torch.testing.assert_close(self.conv[block], conv[0])
        torch.testing.assert_close(self.cache[3], prefix, atol=0, rtol=0)

    def test_bf16_restored_partial_block_continues_identically(self):
        self.inputs = self.inputs.bfloat16()
        self.weight = self.weight.bfloat16()
        self.cache, self.ssm, self.conv = self.make_cache(torch.bfloat16)
        self.run_cached(0, 7, [0, 7], [0], [[3, 7, 2]])
        d_cache, d_ssm, d_conv = self.make_cache(torch.bfloat16)
        # Restore the last partial block at a different physical block ID.
        d_cache[5].view(torch.uint8).copy_(self.cache[7].view(torch.uint8))
        p_out = self.run_cached(7, 12, [0, 5], [7], [[3, 7, 2]])
        d_out = self.run_cached(7, 12, [0, 5], [7], [[0, 5, 9]], d_ssm, d_conv)
        torch.testing.assert_close(p_out, d_out, atol=0, rtol=0)
        for p, d in ((7, 5), (2, 9)):
            torch.testing.assert_close(self.ssm[p], d_ssm[d], atol=0, rtol=0)
            torch.testing.assert_close(self.conv[p], d_conv[d], atol=0, rtol=0)

    def test_invalid_later_row_rejected_before_any_cache_write(self):
        saved = self.cache.clone()
        cases = (
            ([0, 0], [[1], [12]]),  # invalid later row
            ([0, 0], [[1], [1]]),  # two writers
            ([1, 1], [[1], [1]]),  # shared partial prefix without copy-on-write
            ([0, 5], [[1, 2], [2, 1]]),  # another row reads the first writer's block
            ([0, -1], [[1], [2]]),
        )
        for prefixes, table in cases:
            with self.subTest(prefixes=prefixes, table=table):
                with self.assertRaises(ValueError):
                    self.run_cached(0, 2, [0, 1, 2], prefixes, table)
                torch.testing.assert_close(self.cache, saved, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
