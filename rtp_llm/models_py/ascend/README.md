# Ascend bring-up components

This directory contains original adapters against merged RTP-LLM and the
published Transformers dependency. It does not enable an Ascend model backend
or provide a working PD service yet. No code from unmerged RTP-LLM PRs #1481 or
#1349 is included.

`linear_attention.py` adapts the published Qwen3.5 Torch reference functions to
RTP's packed sequences and cache layouts. GDN state uses `[N, HV, V, K]`, and
convolution state uses `[N, W-1, C]`. Initial states may alias shared prefix
storage and are never mutated. Accumulation returns FP32 GDN state; the caller
casts only when storing to `LinearCacheConverter`'s configured cache dtype.
Convolution produces SiLU-activated output, matching Qwen3.5.

The path reads sequence metadata on the host and cannot be captured in a graph.
The caller still needs to map scheduler block IDs to states, save intermediate
block-boundary snapshots for prefix caching, and connect full attention and the
native execution runtime. These adapters are not a replacement for that work.

Run the CPU regression tests from the repository root:

```sh
python -m unittest rtp_llm.models_py.ascend.test.linear_attention_test -v
```

Tests cover non-square K/V dimensions, grouped value heads, packed sequence
isolation, empty sequences, prefill-to-decode continuation, immutable prefix
state, and BF16 recurrent/convolution state restored from a separate allocation
using RTP's actual byte layout. The byte-copy test is a component test, not a
TCP/UB or two-device PD acceptance test.
