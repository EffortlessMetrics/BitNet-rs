# BITNET-SPEC-B158-LARGE-CUDA

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0009 bitnet_b1_58-large control model](../proposals/BITNET-PROP-0009-bitnet-b158-large-control-model.md)
Linked specs: [artifact contract](BITNET-SPEC-B158-LARGE-ARTIFACT-CONTRACT.md), [reference quality](BITNET-SPEC-B158-LARGE-REFERENCE-QUALITY.md), [CPU contract](BITNET-SPEC-B158-LARGE-CPU.md), [RTX 5070 Ti CUDA answer readiness](rtx5070ti-cuda-answer-readiness.md)
Linked ADRs: n/a
Linked plan: [bitnet_b1_58-large implementation plan](../../plans/bitnet-b158-large/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no CUDA support promotion until receipts pass
Policy impact: no policy exception

## Purpose

Define CUDA support for `1bitLLM/bitnet_b1_58-large` after CPU answer readiness
exists for the exact artifact.

## CUDA lanes

```text
b158_large_i2s_qk256_cuda_candidate
b158_large_tl2_cuda_candidate
b158_large_f16_reference_cuda_diagnostic
```

## Acceptance

CUDA promotion requires:

- CPU answer-ready receipt exists;
- execution plan exists;
- all BitNet linear layers classified;
- unsupported ops counted;
- one-token CUDA receipt;
- short-decode CUDA receipt;
- warm-session CUDA receipt;
- CPU/CUDA answer parity or first divergence;
- exact artifact SHA256 and tokenizer authority;
- selected route explicit;
- `fallback_used = false`;
- `speedup_claim = false` until benchmark review.

## Hard rules

- Do not reuse official Microsoft 2B CUDA proof.
- Do not label F16 reference conversion as packed `I2_S` CUDA.
- Do not claim `TL2` CUDA until the `TL2` layout/kernel route is specified and
  tested.
- Do not treat layer classification as execution proof.
- Do not claim speedup from one-token or short-decode quality receipts.
