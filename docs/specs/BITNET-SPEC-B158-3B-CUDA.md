# BITNET-SPEC-B158-3B-CUDA

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0010](../proposals/BITNET-PROP-0010-bitnet-b158-3b-tl-model.md)
Linked specs: [3B CPU](BITNET-SPEC-B158-3B-CPU.md), [3B TL layout](BITNET-SPEC-B158-3B-TL1-TL2-LAYOUT.md), [3B performance](BITNET-SPEC-B158-3B-PERFORMANCE.md)
Linked ADRs: n/a
Linked plan: [3B TL implementation plan](../../plans/bitnet-b158-3b/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no CUDA support promotion until receipts pass
Policy impact: no policy exception

## Purpose

Define the CUDA proof boundary for the 3B TL2 lane. CUDA is valid only after the
same artifact and tokenizer/prompt profile pass x86 TL2 CPU proof.

## Required path

```text
x86 TL2 reference-good
→ Rust TL2 structural loader
→ scalar TL2 oracle
→ CPU answer-ready
→ TL2 CUDA fixture
→ one-token CUDA
→ short-decode CUDA
→ warm-session CUDA
→ benchmark review
```

## Required CUDA receipts

CUDA receipts must record:

- model id, source revision, artifact hash, and tokenizer/prompt hashes;
- route `tl2`;
- selected backend, for example `nvidia-rtx-5070-ti-cuda`;
- selected kernel, for example `tl2-cuda-reference-gemv`;
- fallback status with `fallback_used = false` for strict proof;
- prompt token IDs, generated token IDs, decoded text, and divergence taxonomy;
- quality result;
- `speedup_claim = false` unless exact-profile benchmark review accepts it.

## Hard rules

- No 3B CUDA proof before x86 TL2 CPU answer-ready.
- No QK256 CUDA kernel reuse unless proving diagnostic rejection.
- No dense regular-LLM CUDA proof inheritance.
- No speedup claim before exact-profile review.
- A one-token CUDA receipt does not prove short-decode, warm-session, server, or
  benchmark readiness.
