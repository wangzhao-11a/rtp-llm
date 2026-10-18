"""Check Page-RR ownership against independently enumerated logical pages."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_page_rr_cache_gather import (
    MlaPageRRCacheGather,
    _restore_mla_page_rr_prefix,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_mla_metadata import (
    PageRRMlaMetadata,
)


PAGE, KERNEL = 4096, 128
POISON = -9999.0


def source_table(batch, pages, tp, rank, layout, seed=0):
    """Give every table column a distinct, shuffled physical kernel block."""
    owned = [p for p in range(pages) if p % tp == rank]
    columns = (pages if layout == "replicated" else len(owned)) * (PAGE // KERNEL)
    ids = (
        torch.randperm(batch * columns, generator=torch.Generator().manual_seed(seed))
        + 1
    )
    host = ids.reshape(batch, columns).to(torch.int32)
    storage = torch.zeros((batch, columns + 5), dtype=torch.int32, device="cuda")
    table = storage[:, :columns]
    table.copy_(host)
    # Build an oracle by enumerating physical pages and their token payloads.
    # Neither the production slot helper nor prefix packer defines this map.
    maps, stripes = [], []
    for request in range(batch):
        slots, stripe = {}, []
        table_pages = list(range(pages)) if layout == "replicated" else owned
        for table_page, logical_page in enumerate(table_pages):
            for subpage in range(PAGE // KERNEL):
                physical = int(host[request, table_page * (PAGE // KERNEL) + subpage])
                if logical_page in owned:
                    stripe.append(physical)
                    for offset in range(KERNEL):
                        position = logical_page * PAGE + subpage * KERNEL + offset
                        slots[position] = physical * KERNEL + offset
        maps.append(slots)
        stripes.append(stripe)
    return host, table, maps, stripes


def markers(request, positions):
    return torch.tensor([request * 1000000 + p for p in positions], dtype=torch.float32)


class PageRRTableLayoutValidationTest(unittest.TestCase):
    def test_invalid_layout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "table_layout"):
            PageRRMlaMetadata(PAGE, KERNEL, 4, 0, table_layout="unknown")
        with self.assertRaisesRegex(ValueError, "table_layout"):
            MlaPageRRCacheGather(PAGE, 4, 0, KERNEL, table_layout="unknown")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/Triton")
class PageRRTableLayoutTest(unittest.TestCase):

    def test_compact_prefill_payload_transfers_to_replicated_decode_across_tp(self):
        from rtp_llm.ops import compute_ops

        length, features, pages = 36865, 24, 16
        bits = ((torch.arange(length + 1)[:, None] + 17) >> torch.arange(features)) & 1
        values = bits.to(device="cuda", dtype=torch.bfloat16)
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            source = []
            scale = torch.ones((), device="cuda", dtype=torch.float32)
            for rank in range(8):
                host, table, maps, _ = source_table(
                    1, pages, 8, rank, "compact", seed=100 + rank
                )
                inputs = SimpleNamespace(
                    input_lengths=torch.tensor([length], dtype=torch.int32),
                    prefix_lengths=torch.zeros(1, dtype=torch.int32),
                    input_lengths_device=torch.tensor(
                        [length], dtype=torch.int32, device="cuda"
                    ),
                    prefix_lengths_device=torch.zeros(
                        1, dtype=torch.int32, device="cuda"
                    ),
                    cu_seqlens_device=torch.tensor(
                        [0, length], dtype=torch.int32, device="cuda"
                    ),
                    kv_cache_kernel_block_id_device=table,
                    logical_token_count=length,
                    physical_token_count=length,
                )
                metadata = PageRRMlaMetadata(
                    PAGE, KERNEL, 8, rank, table_layout="compact"
                )
                metadata.prepare_prefill(inputs)
                cache = torch.full(
                    (host.numel() + 1, KERNEL, features), 2.0, device="cuda"
                ).to(dtype)
                compute_ops.concat_and_cache_mla(
                    values[:length, :16],
                    values[:length, 16:],
                    cache,
                    metadata.slot_mapping,
                    "auto" if dtype == torch.bfloat16 else "fp8",
                    scale,
                )
                expected = torch.full(cache.shape, 2.0, dtype=torch.float32)
                positions = [p for p in range(length) if p in maps[0]]
                expected.view(-1, features)[
                    torch.tensor([maps[0][p] for p in positions])
                ] = bits[positions].float()
                torch.testing.assert_close(
                    cache.view(torch.uint8).cpu(),
                    expected.to(dtype).view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
                source.append((host, cache))

            for prefix in (4096, 32768, length):
                for rank in range(4):
                    with self.subTest(dtype=dtype, prefix=prefix, decode_rank=rank):
                        host, table, maps, stripe = source_table(
                            1, pages, 4, rank, "replicated", seed=200 + rank
                        )
                        cache = torch.full(
                            (host.numel() + 1, KERNEL, features), 2.0, device="cuda"
                        ).to(dtype)
                        # A logical page is copied whole from its TP8 owner into
                        # every TP4 receiver's global FULL table. These indices
                        # are enumerated from the transfer contract, not from
                        # either production metadata/slot mapping implementation.
                        for page in range((prefix + PAGE - 1) // PAGE):
                            src_table, src_cache = source[page % 8]
                            for subpage in range(PAGE // KERNEL):
                                src_id = int(
                                    src_table[
                                        0, (page // 8) * (PAGE // KERNEL) + subpage
                                    ]
                                )
                                dst_id = int(host[0, page * (PAGE // KERNEL) + subpage])
                                cache[dst_id].view(torch.uint8).copy_(
                                    src_cache[src_id].view(torch.uint8)
                                )
                        expected = torch.full(cache.shape, 2.0, dtype=torch.float32)
                        for page in range((prefix + PAGE - 1) // PAGE):
                            for subpage in range(PAGE // KERNEL):
                                begin = page * PAGE + subpage * KERNEL
                                count = min(KERNEL, max(0, prefix - begin))
                                dst_id = int(host[0, page * (PAGE // KERNEL) + subpage])
                                expected[dst_id, :count] = bits[
                                    begin : begin + count
                                ].float()
                        torch.testing.assert_close(
                            cache.view(torch.uint8).cpu(),
                            expected.to(dtype).view(torch.uint8),
                            rtol=0,
                            atol=0,
                        )
                        inputs = SimpleNamespace(
                            input_lengths=torch.ones(1, dtype=torch.int32),
                            physical_token_count=1,
                            is_target_verify=False,
                            is_mtp_draft_update=False,
                            is_cuda_graph=False,
                            sequence_lengths_plus_1_device=torch.tensor(
                                [prefix + 1], dtype=torch.int32, device="cuda"
                            ),
                            kv_cache_kernel_block_id_device=table,
                        )
                        metadata = PageRRMlaMetadata(
                            PAGE, KERNEL, 4, rank, table_layout="replicated"
                        )
                        metadata.prepare_decode(inputs)
                        self.assertEqual(
                            metadata.slot_mapping.cpu().tolist(),
                            [maps[0].get(prefix, -1)],
                        )
                        self.assertEqual(
                            metadata.query_block_tables.cpu().tolist(), stripe
                        )
                        compute_ops.concat_and_cache_mla(
                            values[prefix : prefix + 1, :16],
                            values[prefix : prefix + 1, 16:],
                            cache,
                            metadata.slot_mapping,
                            "auto" if dtype == torch.bfloat16 else "fp8",
                            scale,
                        )
                        if prefix in maps[0]:
                            expected.view(-1, features)[maps[0][prefix]] = bits[
                                prefix
                            ].float()
                        torch.testing.assert_close(
                            cache.view(torch.uint8).cpu(),
                            expected.to(dtype).view(torch.uint8),
                            rtol=0,
                            atol=0,
                        )
                        count = int(metadata.local_causal_lens.cpu().item())
                        ids = metadata.query_block_tables[
                            0, : (count + KERNEL - 1) // KERNEL
                        ].long()
                        actual = cache.index_select(0, ids).flatten(0, 1)[:count]
                        positions = [
                            p for p in range(prefix + 1) if (p // PAGE) % 4 == rank
                        ]
                        torch.testing.assert_close(
                            actual.view(torch.uint8).cpu(),
                            bits[positions].to(dtype).view(torch.uint8),
                            rtol=0,
                            atol=0,
                        )

    def test_bf16_fp8_cache_writer_and_prefix_payload_keep_exact_position_bits(self):
        from rtp_llm.ops import compute_ops

        length, features = 36865, 24
        bits = ((torch.arange(length)[:, None] + 17) >> torch.arange(features)) & 1
        values = bits.to(device="cuda", dtype=torch.bfloat16)
        for layout in ("compact", "replicated"):
            for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                payloads = []
                for rank in range(8):
                    with self.subTest(layout=layout, dtype=dtype, rank=rank):
                        host, table, maps, _ = source_table(1, 10, 8, rank, layout)
                        inputs = SimpleNamespace(
                            input_lengths=torch.tensor([length], dtype=torch.int32),
                            prefix_lengths=torch.zeros(1, dtype=torch.int32),
                            input_lengths_device=torch.tensor(
                                [length], dtype=torch.int32, device="cuda"
                            ),
                            prefix_lengths_device=torch.zeros(
                                1, dtype=torch.int32, device="cuda"
                            ),
                            cu_seqlens_device=torch.tensor(
                                [0, length], dtype=torch.int32, device="cuda"
                            ),
                            kv_cache_kernel_block_id_device=table,
                            logical_token_count=length,
                            physical_token_count=length,
                        )
                        metadata = PageRRMlaMetadata(
                            PAGE, KERNEL, 8, rank, table_layout=layout
                        )
                        metadata.prepare_prefill(inputs)
                        cache = torch.full(
                            (host.numel() + 1, KERNEL, features),
                            2.0,
                            dtype=torch.bfloat16,
                            device="cuda",
                        ).to(dtype)
                        scale = torch.ones((), device="cuda", dtype=torch.float32)
                        compute_ops.concat_and_cache_mla(
                            values[:, :16],
                            values[:, 16:],
                            cache,
                            metadata.slot_mapping,
                            "auto" if dtype == torch.bfloat16 else "fp8",
                            scale,
                        )
                        expected = torch.full(cache.shape, 2.0, dtype=torch.float32)
                        positions = [p for p in range(length) if p in maps[0]]
                        slots = torch.tensor([maps[0][p] for p in positions])
                        expected.view(-1, features)[slots] = bits[positions].float()
                        # 0/1 payload and poison=2 are exact in both formats.
                        torch.testing.assert_close(
                            cache.view(torch.uint8).cpu(),
                            expected.to(dtype).view(torch.uint8),
                            rtol=0,
                            atol=0,
                        )
                        adapter = MlaPageRRCacheGather(
                            PAGE, 8, rank, KERNEL, table_layout=layout
                        )
                        payloads.append(adapter._pack_prefix(cache, table, [length]))
                restored = _restore_mla_page_rr_prefix(
                    torch.stack(payloads), adapter._prefix_descriptor
                )
                torch.testing.assert_close(
                    restored.view(torch.uint8).cpu(),
                    bits.to(dtype).view(torch.uint8),
                    rtol=0,
                    atol=0,
                )

    def test_prefill_owner_writes_and_prefix_restore_use_same_physical_blocks(self):
        lengths = [36864, 40960]
        for layout in ("compact", "replicated"):
            payloads, chunk_payloads = [], []
            for rank in range(8):
                with self.subTest(layout=layout, rank=rank):
                    host, table, maps, _ = source_table(2, 12, 8, rank, layout)
                    cu = torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32)
                    max_q = max(lengths)
                    padding = torch.tensor(
                        [0] * lengths[0] + [max_q - lengths[0]] * lengths[1],
                        dtype=torch.int32,
                        device="cuda",
                    )
                    inputs = SimpleNamespace(
                        input_lengths=torch.tensor(lengths, dtype=torch.int32),
                        prefix_lengths=torch.zeros(2, dtype=torch.int32),
                        input_lengths_device=torch.tensor(
                            lengths, dtype=torch.int32, device="cuda"
                        ),
                        prefix_lengths_device=torch.zeros(
                            2, dtype=torch.int32, device="cuda"
                        ),
                        cu_seqlens_device=cu.cuda(),
                        kv_cache_kernel_block_id_device=table,
                        logical_token_count=sum(lengths),
                        physical_token_count=sum(lengths),
                        padding_offset=padding,
                    )
                    metadata = PageRRMlaMetadata(
                        PAGE, KERNEL, 8, rank, table_layout=layout
                    )
                    metadata.prepare_prefill(inputs)
                    expected_slots = [
                        maps[request].get(position, -1)
                        for request, length in enumerate(lengths)
                        for position in range(length)
                    ]
                    self.assertEqual(
                        metadata.slot_mapping.cpu().tolist(), expected_slots
                    )
                    cache = torch.full(
                        (host.numel() + 1, KERNEL, 1), POISON, device="cuda"
                    )
                    values = torch.cat(
                        [markers(r, range(n)) for r, n in enumerate(lengths)]
                    ).cuda()
                    owned = metadata.slot_mapping >= 0
                    cache.flatten()[metadata.slot_mapping[owned]] = values[owned]
                    expected_cache = torch.full(cache.shape, POISON)
                    expected_indices, expected_values = [], []
                    for request, length in enumerate(lengths):
                        for position in range(length):
                            if position in maps[request]:
                                expected_indices.append(maps[request][position])
                                expected_values.append(request * 1000000 + position)
                    expected_cache.flatten()[torch.tensor(expected_indices)] = (
                        torch.tensor(expected_values, dtype=torch.float32)
                    )
                    torch.testing.assert_close(
                        cache.cpu(), expected_cache, rtol=0, atol=0
                    )
                    adapter = MlaPageRRCacheGather(
                        PAGE, 8, rank, KERNEL, table_layout=layout
                    )
                    payloads.append(adapter._pack_prefix(cache, table, lengths))
                    chunk = adapter.build_prefix_chunk_descriptor(
                        (1, 0),
                        (PAGE, PAGE),
                        (lengths[1] - PAGE, lengths[0] - PAGE),
                        feature_width=1,
                    )
                    chunk_payloads.append(
                        adapter._pack_prefix_chunk(cache, table, chunk)
                    )
                    # A subsequent Prefill appends to two nonzero cached prefixes.
                    appended = SimpleNamespace(**vars(inputs))
                    appended.input_lengths = torch.tensor([1, 7], dtype=torch.int32)
                    appended.prefix_lengths = torch.tensor(lengths, dtype=torch.int32)
                    appended.input_lengths_device = appended.input_lengths.cuda()
                    appended.prefix_lengths_device = appended.prefix_lengths.cuda()
                    appended.cu_seqlens_device = torch.tensor(
                        [0, 1, 8], dtype=torch.int32, device="cuda"
                    )
                    appended.logical_token_count = appended.physical_token_count = 8
                    appended.padding_offset = torch.tensor(
                        [0] + [6] * 7, dtype=torch.int32, device="cuda"
                    )
                    metadata.prepare_prefill(appended)
                    self.assertEqual(
                        metadata.slot_mapping.cpu().tolist(),
                        [
                            maps[r].get(lengths[r] + offset, -1)
                            for r, count in enumerate((1, 7))
                            for offset in range(count)
                        ],
                    )
            restored = _restore_mla_page_rr_prefix(
                torch.stack(payloads), adapter._prefix_descriptor
            )
            expected = torch.cat([markers(r, range(n)) for r, n in enumerate(lengths)])[
                :, None
            ]
            torch.testing.assert_close(restored.cpu(), expected, rtol=0, atol=0)
            restored = _restore_mla_page_rr_prefix(torch.stack(chunk_payloads), chunk)
            expected = torch.cat([markers(r, range(PAGE, lengths[r])) for r in (1, 0)])[
                :, None
            ]
            torch.testing.assert_close(restored.cpu(), expected, rtol=0, atol=0)

    def test_decode_graph_projects_global_pages_and_writes_current_token(self):
        starts = [0, 4095, 4096, 4097, 32767, 32768, 32769, 36864, 45056]
        for layout in ("compact", "replicated"):
            for rank in range(4):
                for phase, query in (("decode", 1), ("verify", 4), ("update", 4)):
                    with self.subTest(layout=layout, rank=rank, phase=phase):
                        batch = len(starts)
                        host, table, _, _ = source_table(batch, 12, 4, rank, layout)
                        base = torch.tensor(starts, dtype=torch.int32, device="cuda")
                        if phase == "decode":
                            base.add_(1)
                        inputs = SimpleNamespace(
                            input_lengths=torch.full(
                                (batch,), query, dtype=torch.int32
                            ),
                            physical_token_count=batch * query,
                            is_target_verify=phase == "verify",
                            is_mtp_draft_update=phase == "update",
                            is_cuda_graph=True,
                            prefix_lengths_device=base,
                            sequence_lengths_plus_1_device=base,
                            kv_cache_kernel_block_id_device=table,
                            cu_seqlens_device=torch.arange(
                                batch + 1, dtype=torch.int32, device="cuda"
                            )
                            * query,
                            valid_token_mask=None,
                        )
                        metadata = PageRRMlaMetadata(
                            PAGE, KERNEL, 4, rank, table_layout=layout
                        )
                        metadata.prepare_decode(inputs)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            metadata.prepare_decode(inputs, forbid_realloc=True)
                        addresses = tuple(
                            t.data_ptr()
                            for t in (
                                metadata.positions_d,
                                metadata.slot_mapping,
                                metadata.query_block_tables,
                            )
                        )
                        for shift in (0, 1):
                            # Same source storage, new physical IDs and positions.
                            current = host.roll(shift, dims=1)
                            table.copy_(current)
                            base.copy_(
                                torch.tensor(
                                    [p + shift + (phase == "decode") for p in starts],
                                    device="cuda",
                                    dtype=torch.int32,
                                )
                            )
                            graph.replay()
                            expected_tables, expected_slots, expected_lengths = (
                                [],
                                [],
                                [],
                            )
                            for request, start in enumerate(starts):
                                owned_pages = [p for p in range(12) if p % 4 == rank]
                                pages = (
                                    list(range(12))
                                    if layout == "replicated"
                                    else owned_pages
                                )
                                positions_to_slots, stripe = {}, []
                                for table_page, logical_page in enumerate(pages):
                                    for subpage in range(PAGE // KERNEL):
                                        physical = int(
                                            current[
                                                request,
                                                table_page * (PAGE // KERNEL) + subpage,
                                            ]
                                        )
                                        if logical_page in owned_pages:
                                            stripe.append(physical)
                                            for offset in range(KERNEL):
                                                positions_to_slots[
                                                    logical_page * PAGE
                                                    + subpage * KERNEL
                                                    + offset
                                                ] = (physical * KERNEL + offset)
                                for offset in range(query):
                                    position = start + shift + offset
                                    expected_slots.append(
                                        positions_to_slots.get(position, -1)
                                    )
                                    expected_lengths.append(
                                        sum(
                                            min(PAGE, max(0, position + 1 - p * PAGE))
                                            for p in owned_pages
                                        )
                                    )
                                    expected_tables.append(stripe)
                            self.assertEqual(
                                metadata.slot_mapping.cpu().tolist(), expected_slots
                            )
                            # Consume the published cache slots and check every
                            # physical cell, including cells that must stay poison.
                            cache = torch.full(
                                (host.numel() + 1, KERNEL), POISON, device="cuda"
                            )
                            values = torch.arange(
                                len(expected_slots), dtype=torch.float32, device="cuda"
                            )
                            live = metadata.slot_mapping >= 0
                            cache.flatten()[metadata.slot_mapping[live]] = values[live]
                            expected_cache = torch.full(cache.shape, POISON)
                            live_rows = [
                                i for i, slot in enumerate(expected_slots) if slot >= 0
                            ]
                            if live_rows:
                                expected_cache.flatten()[
                                    torch.tensor([expected_slots[i] for i in live_rows])
                                ] = torch.tensor(live_rows, dtype=torch.float32)
                            torch.testing.assert_close(
                                cache.cpu(), expected_cache, rtol=0, atol=0
                            )
                            self.assertEqual(
                                metadata.local_causal_lens.flatten().cpu().tolist(),
                                expected_lengths,
                            )
                            torch.testing.assert_close(
                                metadata.query_block_tables.cpu(),
                                torch.tensor(expected_tables, dtype=torch.int32),
                                rtol=0,
                                atol=0,
                            )
                            self.assertEqual(
                                addresses,
                                tuple(
                                    t.data_ptr()
                                    for t in (
                                        metadata.positions_d,
                                        metadata.slot_mapping,
                                        metadata.query_block_tables,
                                    )
                                ),
                            )

    def test_ragged_update_masks_padding_and_accepts_empty_owner_table(self):
        for layout in ("compact", "replicated"):
            for empty in (False, True):
                for rank in range(1, 4) if empty else range(4):
                    with self.subTest(layout=layout, empty=empty, rank=rank):
                        _, table, maps, stripes = source_table(2, 12, 4, rank, layout)
                        if empty:
                            table = table[:, :0]
                        starts = [0, 1] if empty else [4096, 32768]
                        base = torch.tensor(starts, dtype=torch.int32, device="cuda")
                        inputs = SimpleNamespace(
                            input_lengths=torch.tensor([2, 1], dtype=torch.int32),
                            physical_token_count=8,
                            is_target_verify=False,
                            is_mtp_draft_update=True,
                            is_cuda_graph=False,
                            prefix_lengths_device=base,
                            sequence_lengths_plus_1_device=base,
                            kv_cache_kernel_block_id_device=table,
                            cu_seqlens_device=torch.tensor(
                                [0, 2, 3], dtype=torch.int32, device="cuda"
                            ),
                            valid_token_mask=torch.tensor(
                                [True, False, True] + [False] * 5, device="cuda"
                            ),
                        )
                        metadata = PageRRMlaMetadata(
                            PAGE, KERNEL, 4, rank, table_layout=layout
                        )
                        metadata.prepare_decode(inputs)
                        self.assertEqual(
                            metadata.positions_d.cpu().tolist(),
                            [starts[0], 0, starts[1]] + [0] * 5,
                        )
                        expected_slots = [-1] * 8
                        if not empty:
                            expected_slots[0] = maps[0].get(4096, -1)
                            expected_slots[2] = maps[1].get(32768, -1)
                        self.assertEqual(
                            metadata.slot_mapping.cpu().tolist(), expected_slots
                        )
                        tables = metadata.query_block_tables.cpu()
                        if empty:
                            self.assertEqual(tables.tolist(), [[0]] * 8)
                            self.assertEqual(
                                metadata.local_causal_lens.cpu().tolist(), [[0]] * 8
                            )
                        else:
                            expected = torch.zeros_like(tables)
                            expected[0] = torch.tensor(stripes[0], dtype=torch.int32)
                            expected[2] = torch.tensor(stripes[1], dtype=torch.int32)
                            torch.testing.assert_close(tables, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
