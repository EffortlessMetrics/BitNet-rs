# BITNET-SPEC-B158-3B-PERFORMANCE

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0010](../proposals/BITNET-PROP-0010-bitnet-b158-3b-tl-model.md)
Linked specs: [3B CPU](BITNET-SPEC-B158-3B-CPU.md), [3B CUDA](BITNET-SPEC-B158-3B-CUDA.md), [3B Apple](BITNET-SPEC-B158-3B-APPLE.md)
Linked ADRs: n/a
Linked plan: [3B TL implementation plan](../../plans/bitnet-b158-3b/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no performance support promotion until exact-profile review
Policy impact: no policy exception

## Purpose

Define performance profiles and promotion rules for the 3B TL lane. Performance
receipts are invalid unless answer quality already passed for the same artifact,
tokenizer, prompt profile, and route with fallback disabled.

## Profiles

- `cold_load`
- `warm_load`
- `one_token`
- `first_token`
- `prefill_128_decode_16`
- `prefill_512_decode_32`
- `decode_32`
- `decode_128`
- `warm_session_3_turns`
- `warm_session_10_turns`
- `server_nonstream_exact_profile`

## Metrics

Receipts should record:

- `model_load_ms`;
- `tokenizer_load_ms`;
- conversion/load metadata;
- `prompt_render_ms`;
- `tokenize_ms`;
- `prefill_ms`;
- `first_token_ms`;
- `decode_total_ms`;
- `steady_tok_per_s`;
- `kernel_time_ms`;
- launch count;
- memory high-water;
- VRAM or RSS high-water;
- thread count;
- selected backend;
- selected kernel;
- fallback status.

## Promotion rule

No performance or speedup claim is valid without:

- same artifact;
- same tokenizer;
- same prompt profile;
- `fallback_used = false`;
- answer quality passed;
- CPU comparator for accelerator claims;
- exact-profile benchmark review.

## Hard rules

- Benchmarks before reference-good and CPU answer-ready are diagnostic only.
- TL1 benchmark results do not prove TL2 performance, and TL2 benchmark results
  do not prove TL1 performance.
- Server performance requires exact-profile server receipts and does not inherit
  from CLI receipts.
