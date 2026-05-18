# BITNET-PROP-0013: Falcon-E family supported models

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: n/a
Linked specs: BITNET-SPEC-FALCON-E-FAMILY-ARTIFACT-CONTRACT, BITNET-SPEC-FALCON-E-FAMILY-ROUTE-COMPATIBILITY, BITNET-SPEC-FALCON-E-FAMILY-TOKENIZER-PROMPT, BITNET-SPEC-FALCON-E-FAMILY-REFERENCE-QUALITY, BITNET-SPEC-FALCON-E-FAMILY-I2S, BITNET-SPEC-FALCON-E-FAMILY-TL1-TL2, BITNET-SPEC-FALCON-E-FAMILY-CPU, BITNET-SPEC-FALCON-E-FAMILY-CUDA, BITNET-SPEC-FALCON-E-FAMILY-APPLE, BITNET-SPEC-FALCON-E-FAMILY-A770-OPENCL, BITNET-SPEC-FALCON-E-FAMILY-PERFORMANCE
Linked ADRs: n/a
Linked plan: plans/falcon-e-family/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: registers Falcon-E as a planning-only compact 1.58-bit family lane; no answer/backend/speed promotion
Policy impact: no policy exception

## Thesis

Falcon-E Family is BitNet-rs's compact 1.58-bit model-family lane. It gives the
repo direct `I2_S` GGUF artifacts at 1B and 3B scale, small memory footprints,
and a secondary BitNet-like family distinct from Microsoft BitNet, 1bitLLM,
Falcon3, and dense SLMs. It should be used to prove model-family breadth while
preserving artifact, tokenizer, prompt, route, backend, and performance claim
boundaries.

Upstream `microsoft/BitNet` support material lists Falcon-E Family as a
supported 1B-3B family with x86 `I2_S`/`TL2` and ARM `I2_S`/`TL1` routes. The
TII Falcon-E GGUF model cards expose direct `ggml-model-i2_s.gguf` artifacts for
both 1B and 3B instruct checkpoints and show BitNet/llama.cpp-style loading
examples. Those facts make Falcon-E a good candidate for fast artifact
onboarding, but they do not prove BitNet-rs runtime support.

## Why 1B starts first

Falcon-E-1B-Instruct-GGUF is the first target because it is the smallest direct
GGUF candidate in the Falcon-E lane, has the lowest expected storage pressure,
and should produce the fastest artifact inventory, tokenizer/prompt audit,
reference-runner check, and Rust CPU smoke loop. Its proof can establish the
family's receipt shape before larger models stress load time, RAM, VRAM, or
backend residency.

## Why 3B is second

Falcon-E-3B-Instruct-GGUF is the second target because it remains compact enough
for the BitNet-rs validation ladder while pressure-testing memory envelope,
loader classification, CUDA/A770/Apple transfer behavior, and longer decode
stability. The 3B target must receive its own artifact identity, tokenizer,
prompt, layout, answer, and backend receipts; 1B proof does not promote 3B.

## Why direct GGUF lowers artifact risk

A direct upstream GGUF avoids an immediate conversion campaign and lets the first
BitNet-rs work focus on artifact identity, metadata, route classification,
prompt authority, and reference behavior. It still requires exact revision,
file, size, SHA256, license, tokenizer metadata, cleanup status, and route
compatibility before any answer or backend claim.

## Family boundaries

Falcon-E is not Falcon3, not Microsoft BitNet 2B, not 1bitLLM, not dense Falcon,
and not generic Falcon support. Falcon-E proof cannot inherit prompt authority,
layout compatibility, answer quality, CUDA behavior, Apple behavior, A770
behavior, or speed envelopes from those families.

## Non-goals

- Broad Falcon support.
- Dense Falcon support.
- Falcon3 promotion.
- Microsoft BitNet 2B replacement.
- TL1/TL2 runtime claims before TL layout specs and scalar oracles.
- CPU, CUDA, Apple, A770, server, full-residency, or speed claims before exact
  receipts prove those claims for the exact artifact and route.
- Committing model binaries.

## First honest posture

```text
registered secondary BitNet-like family
direct I2_S GGUFs exist externally for 1B and 3B
not yet BitNet-rs answer-ready
not yet CPU/CUDA/Apple/A770-ready
not speed/server/full-residency qualified
```
