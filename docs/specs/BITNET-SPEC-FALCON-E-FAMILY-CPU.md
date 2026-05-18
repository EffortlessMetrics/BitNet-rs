# BITNET-SPEC-FALCON-E-FAMILY-CPU

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: docs/proposals/BITNET-PROP-0013-falcon-e-family-supported-models.md
Linked specs: docs/specs/BITNET-SPEC-FALCON-E-FAMILY-I2S.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-TL1-TL2.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-REFERENCE-QUALITY.md
Linked ADRs: n/a
Linked plan: plans/falcon-e-family/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: CPU answer support only after strict receipts
Policy impact: no policy exception

## CPU paths

```text
x86 I2_S scalar
x86 I2_S AVX2
x86 I2_S AVX512
x86 TL2 scalar/AVX candidate
ARM I2_S scalar/NEON
ARM TL1 scalar/NEON candidate
```

## CPU acceptance

```text
artifact inventory exists
reference-good receipt exists
Rust loader recognizes Falcon-E route
I2_S scalar oracle passes fixtures
strict CPU answer corpus passes
fallback=false
prompt IDs recorded
generated IDs recorded
decoded text recorded
speedup=false
```

## Receipt requirements

CPU receipts must record artifact ID/SHA, model family, tokenizer/prompt receipt,
selected backend, selected kernel, route, host architecture, feature flags,
fallback status, fixture parity, prompt IDs, generated IDs, decoded text, answer
corpus result, memory envelope, and `speedup_claim=false` unless a later
performance review promotes an exact profile.

## Hard rules

CPU 1B proof does not prove CPU 3B. CPU scalar proof does not prove AVX2,
AVX512, Apple NEON, CUDA, A770, speed, server, or full-residency support.
