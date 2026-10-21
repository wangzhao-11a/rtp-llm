"""Eager linear-attention cache adapter for RTP scheduler block tables.

This correctness path writes both state types at every block boundary and at
the end of a request. Metadata is validated on the host before cache writes;
graph capture and speculative decoding are not supported.
"""

import torch

from rtp_llm.models_py.ascend.linear_attention import (
    _boundaries,
    causal_conv1d,
    gated_delta_rule,
)


def linear_attention_with_cache(
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    prefix_lengths: torch.Tensor,
    block_map: torch.Tensor,
    ssm_states: torch.Tensor,
    conv_states: torch.Tensor,
    seq_size_per_block: int,
) -> torch.Tensor:
    """Compute packed GDN output and update RTP's existing cache views.

    Inputs use [T, C], gates [T, HV], SSM [blocks, HV, V, K], and convolution
    [blocks, W-1, C]. Pass views obtained from LinearCacheConverter, so a block
    transfer includes both states. Prefix lengths count previously processed
    tokens, including for one-token decode. Block zero is reserved.

    The scheduler must supply writable destination blocks (copy-on-write for
    shared partial prefixes). Cross-request write/read aliases in this batch
    are rejected. Ownership outside this batch remains the scheduler's duty.
    FP32 SSM accumulation continues across block boundaries without a BF16
    round trip; only persisted snapshots are cast to the cache dtype.
    """
    if mixed_qkv.ndim != 2 or ssm_states.ndim != 4 or conv_states.ndim != 3:
        raise ValueError("expected packed QKV and rank-four/rank-three state views")
    tokens, channels = mixed_qkv.shape
    blocks, value_heads, value_dim, key_dim = ssm_states.shape
    if min(blocks, value_heads, value_dim, key_dim) <= 0:
        raise ValueError("cache dimensions must be positive")
    qk_channels = channels - value_heads * value_dim
    if qk_channels <= 0 or qk_channels % (2 * key_dim):
        raise ValueError("QKV channels do not match the SSM layout")
    key_heads = qk_channels // (2 * key_dim)
    if value_heads % key_heads:
        raise ValueError("value heads must be a multiple of key heads")
    if (
        conv_states.shape[0] != blocks
        or conv_states.shape[2] != channels
        or conv_states.shape[1] < 1
        or conv_weight.shape != (channels, conv_states.shape[1] + 1)
        or g.shape != (tokens, value_heads)
        or beta.shape != g.shape
    ):
        raise ValueError("incompatible convolution cache, weights, or gates")
    tensors = (mixed_qkv, conv_weight, g, beta, ssm_states, conv_states)
    if any(t.device != mixed_qkv.device or not t.is_floating_point() for t in tensors):
        raise ValueError("inputs and state views must be floating on the same device")
    if conv_weight.dtype != mixed_qkv.dtype or conv_states.dtype != mixed_qkv.dtype:
        raise ValueError("convolution inputs, weights, and cache must share dtype")
    if not isinstance(seq_size_per_block, int) or seq_size_per_block <= 0:
        raise ValueError("seq_size_per_block must be a positive integer")
    offsets = _boundaries(cu_seqlens, tokens)
    batch = len(offsets) - 1
    if (
        prefix_lengths.shape != (batch,)
        or prefix_lengths.dtype not in (torch.int32, torch.int64)
        or block_map.ndim != 2
        or block_map.shape[0] != batch
        or block_map.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("prefix lengths and block map must be batch integer metadata")
    prefixes = prefix_lengths.tolist()
    table = block_map.tolist()
    reads: dict[int, set[int]] = {}
    writes: dict[int, int] = {}
    plans = []

    def block_at(row: int, position: int) -> int:
        if position >= len(table[row]):
            raise ValueError("block map is too short for the request")
        block = table[row][position]
        if not 0 < block < blocks:
            raise ValueError("live cache block ID must be positive and in range")
        return block

    for row, (start, end) in enumerate(zip(offsets, offsets[1:])):
        prefix = prefixes[row]
        if prefix < 0:
            raise ValueError("prefix lengths must be nonnegative")
        initial = None
        segments = []
        if start != end:
            if prefix:
                initial = block_at(row, (prefix - 1) // seq_size_per_block)
                reads.setdefault(initial, set()).add(row)
            cursor = start
            position = prefix
            while cursor < end:
                length = min(
                    end - cursor, seq_size_per_block - position % seq_size_per_block
                )
                block = block_at(row, position // seq_size_per_block)
                if block in writes:
                    raise ValueError("destination cache blocks must be unique")
                writes[block] = row
                segments.append((cursor, cursor + length, block))
                cursor += length
                position += length
        plans.append((initial, segments))
    for block, row in writes.items():
        if reads.get(block, set()) - {row}:
            raise ValueError("destination aliases another request's prefix state")

    output = mixed_qkv.new_empty((tokens, value_heads, value_dim))
    split_sizes = [key_heads * key_dim, key_heads * key_dim, value_heads * value_dim]
    for initial, segments in plans:
        ssm = None if initial is None else ssm_states[initial : initial + 1]
        conv = None if initial is None else conv_states[initial : initial + 1]
        for start, end, block in segments:
            lengths = torch.tensor([0, end - start], dtype=torch.int64)
            convolved, conv = causal_conv1d(
                mixed_qkv[start:end], conv_weight, lengths, conv
            )
            query, key, value = convolved.split(split_sizes, dim=-1)
            result, ssm = gated_delta_rule(
                query.reshape(-1, key_heads, key_dim),
                key.reshape(-1, key_heads, key_dim),
                value.reshape(-1, value_heads, value_dim),
                g[start:end],
                beta[start:end],
                lengths,
                ssm,
            )
            output[start:end].copy_(result)
            ssm_states[block].copy_(ssm[0])
            conv_states[block].copy_(conv[0])
    return output
