# BITNET-SPEC-B158-LARGE-CPU

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0009 bitnet_b1_58-large control model](../proposals/BITNET-PROP-0009-bitnet-b158-large-control-model.md)
Linked specs: [artifact contract](BITNET-SPEC-B158-LARGE-ARTIFACT-CONTRACT.md), [reference quality](BITNET-SPEC-B158-LARGE-REFERENCE-QUALITY.md), [CPU scalar hotpath](BITNET-SPEC-CPU-SCALAR-HOTPATH.md), [CPU AVX2 hotpath](BITNET-SPEC-CPU-AVX2-HOTPATH.md), [CPU AVX-512 kernel contract](BITNET-SPEC-CPU-AVX512-KERNEL-CONTRACT.md)
Linked ADRs: n/a
Linked plan: [bitnet_b1_58-large implementation plan](../../plans/bitnet-b158-large/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no CPU support promotion until receipts pass
Policy impact: no policy exception

## Purpose

Define CPU support for `1bitLLM/bitnet_b1_58-large` after artifact, conversion,
tokenizer, prompt, and reference-good gates pass.

## CPU lanes

```text
scalar_reference
x86_avx2
x86_avx512
apple_cpu_neon
kaby_cpu_avx2_optional
```

## Required receipts

CPU receipts must record:

- strict model load;
- strict tokenizer and prompt authority;
- exact artifact SHA256;
- selected backend and kernel;
- `fallback_used = false`;
- prompt IDs;
- generated IDs;
- decoded text;
- quality result;
- phase timing;
- allocation/materialization counters where available;
- first divergence when comparing scalar and SIMD lanes;
- `speedup_claim = false` until benchmark review.

## Quantization rules

For `I2_S`:

- If the artifact is actually packed `I2_S`, use the existing QK256/scaled I8S
  path only if the layout matches the model/kernel compatibility ledger.
- If the artifact is F16 reference GGUF, do not claim packed BitNet
  acceleration.

For `TL1` and `TL2`:

- Add separate tensor/layout specs first.
- Do not route TL kernels through QK256 `I2_S` code.
- Do not infer TL correctness from an `I2_S` receipt.

## Promotion rule

`cpu_answer_ready` requires reference-good output for the exact artifact and a
strict CPU answer receipt with fallback false. Structural loading or a single
unscored generation is insufficient.
