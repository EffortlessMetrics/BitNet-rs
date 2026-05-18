# BITNET-SPEC-B158-LARGE-PERFORMANCE

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0009 bitnet_b1_58-large control model](../proposals/BITNET-PROP-0009-bitnet-b158-large-control-model.md)
Linked specs: [artifact contract](BITNET-SPEC-B158-LARGE-ARTIFACT-CONTRACT.md), [reference quality](BITNET-SPEC-B158-LARGE-REFERENCE-QUALITY.md), [CPU](BITNET-SPEC-B158-LARGE-CPU.md), [CUDA](BITNET-SPEC-B158-LARGE-CUDA.md), [Apple](BITNET-SPEC-B158-LARGE-APPLE.md), [runtime performance contract](BITNET-SPEC-0014-runtime-performance-contract.md)
Linked ADRs: n/a
Linked plan: [bitnet_b1_58-large implementation plan](../../plans/bitnet-b158-large/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no benchmark promotion until exact-profile review passes
Policy impact: no policy exception

## Purpose

Define performance qualification for `1bitLLM/bitnet_b1_58-large` once artifact,
conversion, tokenizer, prompt, reference, CPU, and exact backend correctness
receipts exist.

## Profiles

```text
cold_load
warm_load
one_token
first_token
prefill_128_decode_16
prefill_512_decode_32
decode_32
decode_128
warm_session_3_turns
warm_session_10_turns
server_nonstream_exact_profile
```

## Required metrics

Benchmark receipts must record:

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
- H2D/D2H bytes and timing where relevant;
- VRAM/RSS high-water;
- thread count;
- backend selected;
- fallback used;
- exact artifact SHA256;
- tokenizer and prompt profile IDs;
- quality gate pointer.

## Promotion rule

No speedup claim is allowed until the benchmark uses the exact same artifact,
same tokenizer, same prompt profile, `fallback_used = false`, and benchmark
review accepts the profile. Smaller model size, structural loading, one-token
execution, or a single warm run is not a speedup claim.

## Product surfaces

Product CLI or server readiness may cite performance receipts only after the
underlying support tier is already proven. Benchmark-qualified does not by
itself imply `ask`, `chat`, or server readiness.
