# BITNET-PROP-0012: Falcon3 Family Supported Models

Status: draft
Owner: model-artifacts
Created: 2026-05-18
Linked proposal: n/a
Linked specs: [Falcon3 artifact contract](../specs/BITNET-SPEC-FALCON3-FAMILY-ARTIFACT-CONTRACT.md), [Falcon3 route compatibility](../specs/BITNET-SPEC-FALCON3-FAMILY-ROUTE-COMPATIBILITY.md), [Falcon3 tokenizer/prompt](../specs/BITNET-SPEC-FALCON3-FAMILY-TOKENIZER-PROMPT.md), [Falcon3 reference quality](../specs/BITNET-SPEC-FALCON3-FAMILY-REFERENCE-QUALITY.md), [Falcon3 I2_S](../specs/BITNET-SPEC-FALCON3-FAMILY-I2S.md), [Falcon3 TL1/TL2](../specs/BITNET-SPEC-FALCON3-FAMILY-TL1-TL2.md), [Falcon3 CPU](../specs/BITNET-SPEC-FALCON3-FAMILY-CPU.md), [Falcon3 CUDA](../specs/BITNET-SPEC-FALCON3-FAMILY-CUDA.md), [Falcon3 Apple](../specs/BITNET-SPEC-FALCON3-FAMILY-APPLE.md), [Falcon3 A770/OpenCL](../specs/BITNET-SPEC-FALCON3-FAMILY-A770-OPENCL.md), [Falcon3 performance](../specs/BITNET-SPEC-FALCON3-FAMILY-PERFORMANCE.md)
Linked ADRs: [BITNET-ADR-0005](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Linked plan: [Falcon3 family implementation plan](../../plans/falcon3-family/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: registers a candidate family only; no answer/backend/speed promotion
Policy impact: no policy exception

## Thesis

Falcon3 Family should be BitNet-rs's first multi-size supported BitNet-family onboarding lane. It spans 1B through 10B, has upstream bitnet.cpp route support for `I2_S` on x86 and ARM, and can pressure-test artifact authority, tokenizer/prompt authority, QK256/I2_S compatibility, TL route handling, CPU, AVX, CUDA, Apple CPU/NEON, A770/OpenCL, and exact-profile performance across realistic model sizes.

The lane is a family-aware proof stack, not a single-model experiment. BitNet-rs must make claims per artifact, per route, per backend, and per profile.

## Why Falcon3 Starts With 1B

`tiiuae/Falcon3-1B-Instruct-1.58bit-GGUF` is the first target because it exposes the direct `ggml-model-i2_s.gguf` route and is small enough for fast artifact inventory, tokenizer/prompt audit, reference-runner checks, structural Rust load, and CPU answer proof.

The 1B target is allowed to prove only its own exact artifact/profile. It does not prove Falcon3 7B, 3B, 10B, TL routes, CUDA, Apple, A770, speedup, server readiness, or any non-Falcon3 family.

## Why 7B Is Second

`tiiuae/Falcon3-7B-Instruct-1.58bit-GGUF` is the second target because it also exposes `ggml-model-i2_s.gguf` and is large enough to pressure memory, load time, CUDA, AVX, Apple CPU/NEON, and A770/OpenCL route planning after the 1B direct-GGUF path is boring.

The 7B target still needs independent artifact inventory, tokenizer/prompt authority, reference-good output, structural load, I2_S layout proof, and CPU answer receipts.

## Why 3B and 10B Come Later

`tiiuae/Falcon3-3B-Instruct-1.58bit` and `tiiuae/Falcon3-10B-Instruct-1.58bit` are conversion/runner-authority candidates first. They must not enter backend proof until BitNet-rs records exact input revisions, conversion command, tool commit, output artifact hash if produced, reference runner command, tokenizer/prompt authority, and reference-good behavior.

Direct-GGUF 1B/7B routes may start sooner. Safetensors-only or non-GGUF routes must go through conversion/runner authority first.

## Falcon3 Is Not Falcon-E

The existing Apple candidate matrix mentions Falcon-E as a secondary BitNet-like family. Falcon3 is a different onboarding lane with its own family identity, artifact contracts, prompt authority, route support matrix, and support ladder. Falcon-E receipts cannot promote Falcon3 rows, and Falcon3 receipts cannot promote Falcon-E rows.

## Falcon3 Does Not Inherit Microsoft 2B Proof

Microsoft BitNet 2B I2_S/QK256 evidence is useful as a pattern, not as Falcon3 proof. Falcon3 artifacts need their own GGUF metadata audit, tensor-role classification, tokenizer/chat-template decision, I2_S/QK256 compatibility proof, reference-quality receipts, and strict CPU/backend receipts.

This proposal follows the proof-family separation recorded in BITNET-ADR-0005.

## Non-Goals

- Broad Falcon support.
- Dense Falcon3 or dense SLM support.
- Falcon-E promotion.
- Microsoft BitNet 2B promotion.
- Llama3-8B-1.58 promotion.
- Speedup claims.
- Server readiness claims.
- Full device residency claims.
- Runtime, kernel, loader, or tokenizer implementation changes in the documentation/spec PRs.
- Committing model binaries.

## Claim Boundary

```text
Falcon3 Family proof is not Microsoft BitNet 2B proof.
Falcon3 Family proof is not Falcon-E proof.
Falcon3 Family proof is not Llama3-8B-1.58 proof.
Falcon3 Family proof is not dense Falcon3 / dense SLM proof.
Falcon3 1B proof is not Falcon3 7B/10B proof.
Falcon3 I2_S proof is not TL1/TL2 proof.
Falcon3 TL1/TL2 proof is not I2_S/QK256 proof.
No third-party artifact substitution without artifact-authority decision.
No speedup claim before exact-profile benchmark review.
No server readiness before exact-profile server receipts.
No model binaries committed.
```
