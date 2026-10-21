"""Eager Qwen3.5 reference kernels with RTP's packed and V-first layouts.

The math comes from the merged Transformers 5.2 Qwen3.5 implementation, which
is already an RTP dependency. This adapter does not import CUDA/Triton kernels.
It is a correctness path: sequence metadata is read on the host, so graph
capture is unsupported. Returned states are FP32 until the caller stores them
in LinearCacheConverter's configured cache dtype.
"""

import torch


def _boundaries(cu_seqlens: torch.Tensor, tokens: int) -> list[int]:
    if cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("cu_seqlens must be a one-dimensional integer tensor")
    offsets = cu_seqlens.tolist()
    if (
        len(offsets) < 2
        or offsets[0] != 0
        or offsets[-1] != tokens
        or any(left > right for left, right in zip(offsets, offsets[1:]))
    ):
        raise ValueError("cu_seqlens must partition all packed tokens in order")
    return offsets


def gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    use_qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return packed [T, HV, V] output and [N, HV, V, K] final state.

    q/k use [T, H, K]; v uses [T, HV, V]; g/beta use [T, HV]. Each
    sequence receives its own state. Initial states are never modified, even
    when they alias cache storage shared by prefix-cache readers.
    """
    if q.ndim != 3 or k.shape != q.shape or v.ndim != 3:
        raise ValueError("q/k must share [T, H, K], and v must have rank three")
    tokens, heads, key_dim = q.shape
    value_heads, value_dim = v.shape[1:]
    if (
        heads == 0
        or key_dim == 0
        or value_heads == 0
        or value_dim == 0
        or value_heads % heads
        or v.shape[0] != tokens
        or g.shape != (tokens, value_heads)
        or beta.shape != g.shape
    ):
        raise ValueError("incompatible GDN token, head, or gate dimensions")
    tensors = (q, k, v, g, beta)
    if any(t.device != q.device or not t.is_floating_point() for t in tensors):
        raise ValueError("GDN inputs must be floating tensors on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q/k/v must have the same dtype")
    offsets = _boundaries(cu_seqlens, tokens)
    state_shape = (len(offsets) - 1, value_heads, value_dim, key_dim)
    if initial_state is not None and (
        tuple(initial_state.shape) != state_shape
        or initial_state.device != q.device
        or not initial_state.is_floating_point()
    ):
        raise ValueError(f"initial_state must use V-first layout {state_shape}")

    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule,
    )

    ratio = value_heads // heads
    if ratio != 1:
        q = q.repeat_interleave(ratio, dim=1)
        k = k.repeat_interleave(ratio, dim=1)
    output = torch.empty_like(v)
    final_state = torch.empty(state_shape, dtype=torch.float32, device=q.device)
    for index, (start, end) in enumerate(zip(offsets, offsets[1:])):
        state = (
            torch.zeros_like(final_state[index : index + 1])
            if initial_state is None
            else initial_state[index : index + 1].float()
        )
        if start == end:
            final_state[index].copy_(state[0])
            continue
        kernel = (
            torch_recurrent_gated_delta_rule
            if end - start == 1
            else torch_chunk_gated_delta_rule
        )
        result, state_kv = kernel(
            q[start:end].unsqueeze(0),
            k[start:end].unsqueeze(0),
            v[start:end].unsqueeze(0),
            g[start:end].unsqueeze(0),
            beta[start:end].unsqueeze(0),
            initial_state=state.transpose(-1, -2).contiguous(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm,
        )
        output[start:end].copy_(result[0])
        final_state[index].copy_(state_kv[0].transpose(-1, -2))
    return output, final_state


def causal_conv1d(
    mixed_qkv: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply SiLU depthwise convolution, using RTP's [N, W-1, C] state.

    Input/output are [T, C], weights are [C, W]. Unlike the reference helper,
    this function does not mutate initial_state (it may be a shared prefix).
    """
    if mixed_qkv.ndim != 2 or weight.ndim != 2:
        raise ValueError("mixed_qkv and convolution weight must have rank two")
    tokens, channels = mixed_qkv.shape
    width = weight.shape[1]
    if channels == 0 or weight.shape[0] != channels or width < 2:
        raise ValueError("expected [C, W] weights with W >= 2")
    if weight.device != mixed_qkv.device or weight.dtype != mixed_qkv.dtype:
        raise ValueError("convolution inputs and weights must share dtype/device")
    if not mixed_qkv.is_floating_point():
        raise ValueError("convolution inputs must be floating tensors")
    if bias is not None and (
        bias.shape != (channels,)
        or bias.device != weight.device
        or bias.dtype != weight.dtype
    ):
        raise ValueError("bias must match the convolution channels, dtype, device")
    offsets = _boundaries(cu_seqlens, tokens)
    state_shape = (len(offsets) - 1, width - 1, channels)
    if initial_state is not None and (
        tuple(initial_state.shape) != state_shape
        or initial_state.device != mixed_qkv.device
        or initial_state.dtype != mixed_qkv.dtype
    ):
        raise ValueError(f"convolution state must have shape {state_shape}")

    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_causal_conv1d_update,
    )

    final_state = (
        mixed_qkv.new_zeros(state_shape)
        if initial_state is None
        else initial_state.clone()
    )
    output = torch.empty_like(mixed_qkv)
    for index, (start, end) in enumerate(zip(offsets, offsets[1:])):
        if start == end:
            continue
        state = final_state[index : index + 1].transpose(1, 2).contiguous()
        result = torch_causal_conv1d_update(
            mixed_qkv[start:end].T.unsqueeze(0), state, weight, bias, "silu"
        )
        output[start:end].copy_(result[0].T)
        final_state[index].copy_(state[0].T)
    return output, final_state
