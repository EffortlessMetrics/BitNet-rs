# BITNET-SPEC-B158-3B-CPU

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0010](../proposals/BITNET-PROP-0010-bitnet-b158-3b-tl-model.md)
Linked specs: [3B TL layout](BITNET-SPEC-B158-3B-TL1-TL2-LAYOUT.md), [3B reference quality](BITNET-SPEC-B158-3B-REFERENCE-QUALITY.md)
Linked ADRs: n/a
Linked plan: [3B TL implementation plan](../../plans/bitnet-b158-3b/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no CPU support promotion until receipts pass
Policy impact: no policy exception

## Purpose

Define CPU proof paths for the 3B TL lane. CPU proof is the first BitNet-rs
answer-ready target after artifact, conversion, tokenizer, reference-quality,
layout, loader, and scalar-oracle gates pass.

## Valid CPU routes

| Platform | Valid route |
| --- | --- |
| x86 | TL2 only |
| ARM | TL1 only |
| x86 `I2_S` | diagnostic rejection only |
| ARM `I2_S` | diagnostic rejection only |

## Required receipt shape

```json
{
  "model_id": "onebitllm_bitnet_b158_3b",
  "route": "tl2",
  "selected_backend": "cpu-rust",
  "selected_kernel": "tl2-scalar-reference-gemv",
  "fallback_used": false,
  "prompt_token_ids": [],
  "generated_token_ids": [],
  "quality_passed": true,
  "speedup_claim": false
}
```

## CPU acceptance

A CPU answer-ready claim requires:

- reference-good artifact exists for the same source revision and route family;
- Rust loader recognizes the TL route and classifies tensor roles;
- scalar TL oracle passes fixtures for the route family;
- strict CPU answer corpus passes;
- prompt and generated token IDs are recorded;
- `fallback_used = false`;
- `speedup_claim = false`.

## Hard rules

- No 3B CPU answer claim may use `I2_S`/QK256 kernels.
- x86 TL2 CPU proof does not prove ARM TL1 CPU proof.
- ARM TL1 CPU proof does not prove x86 TL2 CPU proof.
- AVX2, AVX-512, NEON, CUDA, Apple, OpenCL, or server claims require separate
  follow-on receipts.
