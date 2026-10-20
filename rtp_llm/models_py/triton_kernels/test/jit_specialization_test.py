"""Request metadata must change outputs without creating new JIT variants."""

import contextlib
import unittest
from itertools import accumulate
from types import SimpleNamespace

import torch
import triton

from rtp_llm.models_py.triton_kernels.mla import page_rr
from rtp_llm.models_py.triton_kernels.moe.activation import (
    _situ_and_mul_kernel,
    situ_and_mul,
)


@contextlib.contextmanager
def compile_attempts(kernel):
    """Record every JIT compile attempt, including hits in the on-disk cache.

    Clearing only this function's in-process lookup makes a previous test/run's
    warm cache unable to hide specialization. Never filter already_compiled.
    """
    from triton import knobs

    kernel.device_caches.clear()
    attempts = []
    previous = knobs.runtime.jit_cache_hook

    def listen(**event):
        if event["fn"].jit_function is kernel:
            attempts.append(event["key"])
        if previous is not None:
            return previous(**event)
        return None

    knobs.runtime.jit_cache_hook = listen
    try:
        yield attempts
    finally:
        knobs.runtime.jit_cache_hook = previous


class JitSpecializationTest(unittest.TestCase):
    def setUp(self):
        # This target is a GPU contract; missing CUDA must fail, not claim PASS.
        self.assertTrue(torch.cuda.is_available(), "real CUDA execution required")
        torch.manual_seed(7362)

    def assert_one_variant(self, attempts):
        self.assertEqual(
            len(attempts), 1, f"unexpected JIT compile attempts: {len(attempts)}"
        )

    def table(self, requests, width, pad=0, step=1):
        host = torch.arange(
            1, requests * (width * step + pad) + 1, dtype=torch.int32
        ).reshape(requests, width * step + pad)
        return host.cuda()[:, : width * step : step]

    def slot(self, position, batch, table, rank, replicated):
        physical, offset = divmod(position, 8)
        if physical % 2 != rank:
            return -1
        owner_page = physical if replicated else physical // 2
        column = owner_page * 2 + offset // 4
        return int(table[batch, column]) * 4 + offset % 4

    def test_prefill_uniform_ragged_and_short_padding_compile_key(self):
        kernel = page_rr._prepare_prefill_metadata
        for replicated in (False, True):
            with compile_attempts(kernel) as attempts:
                for q_lens, width, pad in (
                    ([7], 17, 0),
                    ([3, 3, 3], 20, 3),
                    ([0, 2, 7], 23, 1),
                    ([2, 0, 1, 0, 4], 32, 5),
                    ([1, 1], 19, 0),
                ):
                    prefixes = [5 + i * 3 for i in range(len(q_lens))]
                    count, total, max_q = len(q_lens), sum(q_lens), max(q_lens)
                    fixed = q_lens[0] if len(set(q_lens)) == 1 else 0
                    # Uniform input really uses the short B+1 indptr as padding.
                    padding = (
                        list(accumulate([0] + q_lens))
                        if fixed
                        else [
                            b * max_q + j - row
                            for row, (b, j) in enumerate(
                                (b, j) for b, q in enumerate(q_lens) for j in range(q)
                            )
                        ]
                    )
                    table = self.table(count, width, pad)
                    host_table = table.cpu()
                    output = torch.empty(total, device="cuda", dtype=torch.int64)
                    logical = max(total - 1, 0)
                    kernel[(triton.cdiv(total, 128),)](
                        torch.tensor(padding, device="cuda", dtype=torch.int32),
                        torch.tensor(prefixes, device="cuda", dtype=torch.int32),
                        table,
                        output,
                        total,
                        logical,
                        width,
                        table.stride(0),
                        count,
                        fixed,
                        max_q,
                        8,
                        4,
                        2,
                        1,
                        128,
                        table_replicated=replicated,
                        debug=True,
                    )
                    expected = [
                        self.slot(prefixes[b] + j, b, host_table, 1, replicated)
                        for b, q in enumerate(q_lens)
                        for j in range(q)
                    ]
                    expected[logical:] = [-1] * (total - logical)
                    self.assertEqual(output.cpu().tolist(), expected)
            self.assert_one_variant(attempts)

    def test_decode_rectangular_and_mtp_compile_keys(self):
        kernel = page_rr._prepare_decode_metadata
        for replicated, mtp, graph in (
            (False, False, False),
            (True, False, True),
            (False, True, False),
            (True, True, False),
        ):
            with compile_attempts(kernel) as attempts:
                for requests, queries, width, pad in (
                    (1, 1, 17, 0),
                    (3, 3, 19, 3),
                    (2, 4, 33, 1),
                    (5, 5, 141, 5),
                ):
                    q_lens = (
                        ([0] + [queries] * (requests - 1))
                        if mtp
                        else [queries] * requests
                    )
                    offsets = list(accumulate([0] + q_lens))
                    rows = max(offsets[-1] + (2 if mtp else 0), 1)
                    base = [
                        0 if graph and b == 0 else 11 + 7 * b for b in range(requests)
                    ]
                    capture = [9 + b for b in range(requests)]
                    mask = [i % 3 != 1 for i in range(rows)]
                    table = self.table(requests, width, pad)
                    host_table = table.cpu()
                    query_width = triton.cdiv(width, 4) * 2 if replicated else width
                    out_table = torch.full(
                        (rows, query_width), -9, device="cuda", dtype=torch.int32
                    )
                    positions = torch.empty(rows, device="cuda", dtype=torch.int32)
                    slots = torch.empty(rows, device="cuda", dtype=torch.int64)
                    lengths = torch.empty_like(positions)

                    def ints(values):
                        return torch.tensor(values, device="cuda", dtype=torch.int32)

                    block = 128 if mtp else 32
                    kernel[
                        (
                            (rows, 1)
                            if mtp
                            else (
                                triton.cdiv(rows, block),
                                triton.cdiv(query_width, 128),
                            )
                        )
                    ](
                        ints(base),
                        ints(capture),
                        ints(offsets),
                        torch.tensor(mask, device="cuda", dtype=torch.bool),
                        table,
                        out_table,
                        positions,
                        slots,
                        lengths,
                        rows,
                        width,
                        table.stride(0),
                        query_width,
                        queries,
                        8,
                        4,
                        2,
                        1,
                        mtp,
                        graph,
                        mtp,
                        requests,
                        mtp,
                        block,
                        table_block=128,
                        table_replicated=replicated,
                    )
                    (
                        expected_positions,
                        expected_slots,
                        expected_lengths,
                        expected_table,
                    ) = ([], [], [], [])
                    for row in range(rows):
                        if mtp:
                            batch = min(
                                next(
                                    (
                                        b
                                        for b in range(requests)
                                        if row < offsets[b + 1]
                                    ),
                                    requests,
                                ),
                                requests - 1,
                            )
                            active = row < offsets[-1] and mask[row]
                            offset = row - offsets[batch]
                        else:
                            batch, offset = divmod(row, queries)
                            active = True
                        start = base[batch] or (capture[batch] + 1 if graph else 0)
                        position = (
                            (start + offset if mtp else start - 1) if active else 0
                        )
                        expected_positions.append(position)
                        expected_slots.append(
                            self.slot(position, batch, host_table, 1, replicated)
                            if active
                            else -1
                        )
                        expected_lengths.append(
                            sum(p // 8 % 2 == 1 for p in range(position + 1))
                            if active
                            else 0
                        )
                        columns = [
                            ((c // 2 * 2 + 1) * 2 + c % 2) if replicated else c
                            for c in range(query_width)
                        ]
                        expected_table.append(
                            [
                                int(host_table[batch, c]) if active and c < width else 0
                                for c in columns
                            ]
                        )
                    self.assertEqual(positions.cpu().tolist(), expected_positions)
                    self.assertEqual(slots.cpu().tolist(), expected_slots)
                    self.assertEqual(lengths.cpu().tolist(), expected_lengths)
                    self.assertEqual(out_table.cpu().tolist(), expected_table)
            self.assert_one_variant(attempts)

    def descriptor(self, lengths):
        page_offsets = list(
            accumulate([0] + [triton.cdiv(triton.cdiv(n, 8), 2) for n in lengths])
        )
        return SimpleNamespace(
            prefix_lens=tuple(lengths),
            request_indices=tuple(reversed(range(len(lengths)))),
            global_page_starts=tuple(i % 3 for i in range(len(lengths))),
            local_page_offsets=tuple(page_offsets),
            output_token_offsets=tuple(accumulate([0] + lengths)),
            feature_width=7,
            page_tokens=8,
            shard_size=2,
            batch_size=len(lengths),
            total_local_pages=page_offsets[-1],
            _cuda_metadata=None,
        )

    def test_pack_request_table_shapes_and_strides_compile_key(self):
        kernel = page_rr._pack_prefix_kernel
        cache = torch.arange(1024 * 4 * 7, dtype=torch.float32).reshape(1024, 4, 7)
        cuda_cache = cache.cuda()
        for replicated in (False, True):
            for rank in (0, 1):
                with compile_attempts(kernel) as attempts:
                    for lengths, width, pad, step in (
                        ([13], 17, 0, 1),
                        ([0, 0, 7, 25], 19, 3, 2),
                        ([41, 0, 3], 32, 1, 1),
                        ([1, 8], 21, 5, 2),
                    ):
                        descriptor = self.descriptor(lengths)
                        table = self.table(len(lengths), width, pad, step)
                        host_table = table.cpu()
                        output = torch.empty(
                            (descriptor.total_local_pages, 8, 7), device="cuda"
                        )
                        page_rr.pack_prefix_cuda(
                            cuda_cache, table, descriptor, rank, output, replicated
                        )
                        expected = torch.zeros_like(output, device="cpu")
                        for request, length in enumerate(lengths):
                            start = descriptor.global_page_starts[request]
                            for token in range(length):
                                relative, offset = divmod(token, 8)
                                global_page = start + relative
                                if global_page % 2 != rank:
                                    continue
                                local = global_page if replicated else global_page // 2
                                block_id = int(
                                    host_table[
                                        descriptor.request_indices[request],
                                        local * 2 + offset // 4,
                                    ]
                                )
                                packed = (
                                    descriptor.local_page_offsets[request]
                                    + relative // 2
                                )
                                expected[packed, offset] = cache[block_id, offset % 4]
                        torch.testing.assert_close(
                            output.cpu(), expected, rtol=0, atol=0
                        )
                self.assert_one_variant(attempts)

    def test_restore_requests_and_local_pages_compile_key(self):
        kernel = page_rr._restore_prefix_kernel
        with compile_attempts(kernel) as attempts:
            for lengths in ([13], [0, 0, 7, 25], [41, 0, 3], [1, 8]):
                descriptor = self.descriptor(lengths)
                gathered = torch.full((2, descriptor.total_local_pages, 8, 7), -999.0)
                expected = torch.arange(sum(lengths) * 7, dtype=torch.float32).reshape(
                    -1, 7
                )
                for request, length in enumerate(lengths):
                    start = descriptor.global_page_starts[request]
                    for token in range(length):
                        relative, offset = divmod(token, 8)
                        owner = (start + relative) % 2
                        packed = descriptor.local_page_offsets[request] + relative // 2
                        gathered[owner, packed, offset] = expected[
                            descriptor.output_token_offsets[request] + token
                        ]
                output = torch.empty((sum(lengths), 7), device="cuda")
                page_rr.restore_prefix_cuda(gathered.cuda(), descriptor, output)
                torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        self.assert_one_variant(attempts)

    def test_situ_actual_widths_split_and_inplace_compile_key(self):
        for width in (6144, 33792):
            for split, inplace, up_beta in (
                (True, False, 1.5),
                (True, False, None),
                (False, True, 1.5),
                (False, True, None),
            ):
                with compile_attempts(_situ_and_mul_kernel) as attempts:
                    for rows in (1, 3, 8, 17):
                        if split:
                            projected = torch.randn(
                                rows, 2 * width, device="cuda", dtype=torch.bfloat16
                            )
                            gate, up = projected.split(width, dim=-1)
                            self.assertEqual(gate.stride(0), 2 * width)
                        else:
                            gate = torch.randn(
                                rows, width, device="cuda", dtype=torch.bfloat16
                            )
                            up = torch.randn_like(gate)
                        g, u = gate.float(), up.float()
                        expected = 1.7 * torch.tanh(g / 1.7) * torch.sigmoid(g)
                        if up_beta is not None:
                            u = up_beta * torch.tanh(u / up_beta)
                        expected = (expected * u).to(gate.dtype)
                        output = situ_and_mul(gate, up, 1.7, up_beta, inplace=inplace)
                        if inplace:
                            self.assertEqual(output.data_ptr(), gate.data_ptr())
                        torch.testing.assert_close(
                            output, expected, rtol=0.016, atol=0.016
                        )
                self.assert_one_variant(attempts)

    def test_situ_int64_offsets_above_two_to_31(self):
        # Expanded zero-stride inputs make the >2**31 logical extent affordable:
        # only the BF16 output owns 4 GiB. Probe both sides of the boundary.
        free, _ = torch.cuda.mem_get_info()
        self.assertGreater(
            free, 12 * 1024**3, "large-index fixture requires 12 GiB idle GPU memory"
        )
        width = 6144
        rows = triton.cdiv(2**31, width) + 1
        n = rows * width
        gate = torch.tensor([[0.7]], dtype=torch.bfloat16, device="cuda").expand(
            rows, width
        )
        up = torch.tensor([[1.2]], dtype=torch.bfloat16, device="cuda").expand(
            rows, width
        )
        output = torch.empty((rows, width), dtype=torch.bfloat16, device="cuda")
        # Launch the actual kernel with N's boundary and a large grid, using cheap
        # zero-stride loads; every element and its last row must be written.
        _situ_and_mul_kernel[(triton.cdiv(n, 256),)](
            gate,
            up,
            output,
            n,
            width,
            0,
            0,
            0,
            0,
            1.7,
            None,
            256,
            enable_fp_fusion=False,
        )
        expected = (
            1.7
            * torch.tanh(gate[0, 0].float() / 1.7)
            * torch.sigmoid(gate[0, 0].float())
            * up[0, 0].float()
        ).to(torch.bfloat16)
        probes = torch.tensor([0, 2**31 - 1, 2**31, n - 1], device="cuda")
        torch.testing.assert_close(
            output.flatten()[probes], expected.expand(4), rtol=0.016, atol=0.016
        )


if __name__ == "__main__":
    unittest.main()
