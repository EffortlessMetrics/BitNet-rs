# BitNet b1.58 3B TL Candidate Campaign

Campaign ID: `bitnet-b158-3b`

Status: active

## Objective

Make `1bitLLM/bitnet_b1_58-3B` a first-class BitNet-rs TL-model candidate
without treating it as an `I2_S`/QK256 extension of the official Microsoft 2B
artifact.

## Why This Exists

Upstream bitnet.cpp lists the 3B model as a supported model with x86 TL2 and ARM
TL1 routes while marking `I2_S` unsupported for both x86 and ARM. The current
Hugging Face listing exposes safetensors shards and tokenizer/config assets, not
an official GGUF. BitNet-rs therefore needs artifact authority, conversion /
runner authority, TL1/TL2 layout contracts, tokenizer/prompt authority, and
reference-good output before any Rust CPU, CUDA, Apple, server, or performance
claim can be made.

## Scope Boundary

This campaign is a BitNet TL candidate lane. It does not promote model coverage,
does not commit model binaries, does not enable runtime inference, and does not
claim answer readiness. `I2_S`/QK256 use for this model is limited to diagnostic
rejection receipts.

## End State

- The proposal, source map, plan, and specs define the support ladder from
  `registered` through `server_exact_profile_ready`.
- Artifact inventory and conversion/runner receipts can be added without
  overclaiming.
- TL1 and TL2 route families are independent; proof does not cross-inherit.
- Backend and speed claims remain blocked until exact-profile receipts pass.

## Work Items

| Work item | Status | Notes |
| --- | --- | --- |
| B158-3B-001 | ready | Add docs rails and specs for guarded 3B TL candidate registration. |

## Review Policy

Docs/spec PRs must run the campaign check, campaign generated-status check, and
`git diff --check`. Runtime, model-binary, model-coverage, artifact-receipt, and
backend changes belong to later work items with narrower proof commands.
