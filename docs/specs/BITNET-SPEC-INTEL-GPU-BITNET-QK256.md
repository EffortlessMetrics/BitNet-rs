# BITNET-SPEC-INTEL-GPU-BITNET-QK256

Status: proposed
Owner: intel-gpu/product
Created: 2026-05-18
Linked proposal: `docs/proposals/BITNET-PROP-0006-intel-gpu-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-INTEL-GPU-ROUTE-CONTRACT.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-QUALITY.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-RESIDENCY.md`
Linked ADRs: n/a
Linked plan: `plans/intel-gpu/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Defines native BitNet route requirements; no promotion.
Policy impact: No exception.

## Purpose

Define the native Intel GPU BitNet route for official I2_S/QK256 artifacts. The
first product lane is A770 OpenCL trusted partial acceleration for named
operations; Arc 140V native OpenCL remains a candidate until separately proven.

## Required kernels and operations

```text
qk256_i2s_gemv_opencl
embedding_lookup_opencl
lm_head_tied_logits_opencl
eventual qk256_i2s_prefill_gemm_opencl
```

A route may claim only the named operations it proves. QK256 linears alone are
trusted partial acceleration, not full model residency.

## Production semantics

Production BitNet QK256 proof must match:

```text
official Microsoft I2_S/QK256 GGUF
canonical packed layout
BitNet.cpp-aligned activation quantization
I2_S × I8_S scaled math
weight scale
activation scale/sum correction
tail-column behavior
row stride behavior
strict tokenizer/template authority
```

## Hard rule

A diagnostic four-values-per-byte toy I2_S kernel cannot satisfy official
QK256 proof. Toy kernels may remain smoke/parity evidence only when receipts and
status surfaces say they are not official BitNet QK256 execution.

## Required receipt evidence

A claim-grade native BitNet OpenCL receipt must include:

- selected backend and runtime API;
- official model contract and artifact hashes;
- tokenizer and prompt-template authority;
- QK256 kernel invocation counts greater than zero for claimed operations;
- CPU fallback count zero;
- `fallback_used=false`;
- quality gate result or explicit unpromoted status;
- not-claims for dense SLM, generic Intel GPU, full residency, and speedup.
