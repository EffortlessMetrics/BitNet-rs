# BITNET-SPEC-LLAMA3-8B-158-CUDA

Status: proposed
Owner: nvidia-5070ti
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0011](../proposals/BITNET-PROP-0011-llama3-8b-158-supported-model.md)
Linked plan: [Llama3 8B 1.58 implementation plan](../../plans/llama3-8b-158/implementation-plan.md)
Support-tier impact: no CUDA claim before CPU answer-ready
Policy impact: no policy exception

## Required sequence

```text
x86 I2_S CPU answer-ready
-> CUDA I2_S fixture parity
-> one-token CUDA
-> short-decode CUDA
-> warm-session CUDA
-> benchmark review
```

## Required CUDA receipt fields

CUDA receipts must record all-layer execution plan, QK256/I2_S route
classification, unsupported-op count, CUDA kernel invocation count, CPU fallback
count `0`, weights uploaded once, per-token weight upload `false`, CPU/CUDA
answer parity or first-divergence classification, selected backend, selected
kernel, and `speedup=false` until review.

## Hard rules

No CUDA proof before CPU answer-ready for the exact route. No TL2 CUDA before a
TL2 scalar oracle. No official Microsoft 2B CUDA proof inheritance. No dense
Llama3 CUDA proof inheritance.
