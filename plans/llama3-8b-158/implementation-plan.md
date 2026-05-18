# Llama3 8B 1.58 Implementation Plan

Status: proposed
Owner: model-artifacts
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0011](../../docs/proposals/BITNET-PROP-0011-llama3-8b-158-supported-model.md)
Linked specs: [Llama3 8B 1.58 specs](../../docs/specs/INDEX.md#llama3-8b-158-supported-model-candidate)
Linked ADRs: n/a

## Claim boundary

This plan registers and sequences a candidate lane. It does not approve a GGUF,
claim answer readiness, claim I2_S/QK256 compatibility, claim TL support beyond
upstream listing, claim CUDA/Apple/CPU support, claim speedup, or claim server
readiness.

## Phase 0: source-of-truth docs

Add the proposal, source map, plan, active campaign, conservative model coverage
row, kernel compatibility row, Apple candidate entries, and generated tracker
updates. Acceptance requires documentation/configuration only, no model
binaries, no runtime/kernel changes, and no support promotion beyond
`registered`.

Proof commands:

```bash
cargo run --locked -p xtask --no-default-features -- campaign check llama3-8b-158
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

## Phase 1: specs

Add artifact, conversion, tokenizer/prompt, route compatibility, reference
quality, I2_S, TL1/TL2, CPU, CUDA, Apple, and performance contracts. These
specs are requirements only; they do not prove the model works.

## Phase 2: claim-control registration

Register a `registered_safetensors_no_approved_gguf` model coverage row with all
answer/backend/speed booleans false, `bitnet_packed_i2s_qk256_proof = false`,
and forbidden claims for official Microsoft 2B inheritance, dense Llama3
inheritance, CPU/CUDA/Apple readiness, speedup, and server readiness.

## Phase 3: artifact inventory

Probe the exact Hugging Face revision and record file names, byte sizes,
SHA256 values, tokenizer/config hashes, source metadata, identity discrepancy,
storage context, cleanup status, and whether an official GGUF exists. No model
binaries are committed.

## Phase 4: conversion and runner authority

Verify or block upstream-compatible `I2_S`, `TL1`, and `TL2` conversion paths,
plus reference-only safetensors/Transformers/vLLM/SGLang/F16 structural lanes.
Blocked conversion receipts remain useful and must stay diagnostic-only.

## Phase 5: tokenizer and reference quality

Audit tokenizer files, chat template, special tokens, prompt rendering, prompt
token IDs, stop policy, and reference runner policy. A reference-good claim
requires deterministic bounded output passing the required corpus.

## Phase 6: I2_S CPU path

Only after approved artifact/runner and reference-good evidence exist, recognize
an approved `I2_S` artifact structurally, prove layout fixtures, establish a
scalar oracle, and run strict CPU answer receipts with `fallback=false`.

## Phase 7: AVX/CUDA/Apple

Only after CPU answer-ready for the exact route, prove AVX parity, CUDA route
plans and one-token/short-decode/warm-session receipts, and Apple CPU/NEON proof
without M4/MacBook cross-claiming.

## Phase 8: TL1/TL2 routes

Add TL fixtures and scalar TL oracles before any TL answer proof. `TL1` and
`TL2` are not QK256 and must not use I2_S kernels.

## Phase 9: benchmark and product promotion

Exact-profile benchmark and product CLI/server promotion require same artifact,
same tokenizer, same prompt profile, answer quality passed, CPU comparator,
`fallback=false`, and explicit review.
