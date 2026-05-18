# BITNET-PROP-0011: Llama3 8B 1.58 Supported-Model Candidate

Status: proposed
Owner: model-artifacts
Type: proposal
Created: 2026-05-18

## Problem

`HF1BitLLM/Llama3-8B-1.58-100B-tokens` is listed by upstream BitNet as a
supported BitNet-family model, but BitNet-rs does not yet have artifact,
tokenizer, prompt, conversion, layout, reference-quality, CPU, accelerator, or
performance proof for this exact model. Treating it as simply “the official
Microsoft 2B model but bigger” would overclaim the current evidence and could
route a Llama3-derived artifact through the wrong prompt, tokenizer, or kernel
family.

The current public artifact lane is safetensors-first. The Hugging Face model
repository exposes `model.safetensors`, tokenizer files, and config files, not a
BitNet-rs-approved GGUF. Upstream support therefore makes the model promising,
but it is not answer authority and is not packed-kernel authority by itself.

## Proposal

Create a first-class large BitNet-family candidate lane for
`HF1BitLLM/Llama3-8B-1.58-100B-tokens` with conservative claim boundaries:

```text
artifact inventory
-> tokenizer / prompt authority
-> conversion or approved runner authority
-> I2_S / TL1 / TL2 layout authority
-> reference-good output
-> Rust structural load
-> scalar CPU oracle
-> CPU answer-ready
-> AVX2 / AVX-512 / CUDA / Apple backend proof
-> exact-profile performance
-> product CLI / server promotion
```

The thesis of the lane is:

```text
HF1BitLLM/Llama3-8B-1.58-100B-tokens is the largest currently listed
BitNet-family supported model that plausibly exercises both I2_S/QK256 and TL
routes across x86 and ARM. It is valuable as a large-model control beyond the
official Microsoft 2B artifact, but it must be onboarded through artifact,
tokenizer, conversion, reference-quality, CPU, accelerator, and benchmark gates
before any product claim.
```

## Source facts to preserve

- Upstream BitNet lists `Llama3-8B-1.58-100B-tokens` as supported with x86
  `I2_S`, x86 `TL2`, ARM `I2_S`, and ARM `TL1` routes.
- The model is a supported upstream BitNet-family model, not the official
  Microsoft 2B answer authority.
- The Hugging Face repository is safetensors/tokenizer/config visible; no
  BitNet-rs-approved GGUF is currently recorded.
- The model is Llama3-derived and prompt/tokenizer-sensitive.
- There is an identity ambiguity to record: upstream calls the model `8.0B`,
  while the Hugging Face display metadata has shown a smaller parameter-count
  label alongside a roughly 3.86 GB safetensors file.
- Existing `bitnet-st2gguf` F16 conversion is structural/reference-only and is
  not packed `I2_S`, `TL1`, or `TL2` proof.

## Goals

- Register the model as a large BitNet-family candidate without answer claims.
- Make artifact inventory, exact revision, file hashes, tokenizer files, config
  files, and storage context the first gate.
- Define conversion/runner lanes before backend work begins.
- Separate Llama3 tokenizer/prompt authority from the Microsoft 2B
  `bitnetcpp-answer` prompt authority.
- Keep `I2_S/QK256`, `TL1`, and `TL2` as separate route families until layout
  proof and scalar oracles exist.
- Enable future x86 CPU, AVX2/AVX-512, CUDA, ARM/NEON, Apple, and TL proof only
  after reference-good and route-compatible artifacts exist.

## Non-goals

- Do not download or commit model binaries in documentation/spec PRs.
- Do not claim CPU, CUDA, Apple, server, or speed readiness.
- Do not substitute third-party GGUF artifacts without explicit artifact
  authority.
- Do not inherit official Microsoft 2B proof, dense Llama proof, dense Qwen
  proof, or generic GGUF proof.
- Do not route `TL1` or `TL2` work through `I2_S/QK256` kernels.

## Claim boundary

Until receipts prove otherwise, the only allowed claim is:

```text
registered candidate: upstream route support is known, HF safetensors artifact
is visible, and BitNet-rs has a conservative onboarding plan.
```

The lane must not claim answer readiness, packed-layout compatibility, backend
support, performance, server readiness, or product CLI readiness from inventory
or upstream listing alone.

## Source-of-truth links

- Source map: [docs/bitnet/llama3-8b-158/README.md](../bitnet/llama3-8b-158/README.md)
- Plan: [plans/llama3-8b-158/implementation-plan.md](../../plans/llama3-8b-158/implementation-plan.md)
- Campaign: [docs/tracking/campaigns/llama3-8b-158/CAMPAIGN.md](../tracking/campaigns/llama3-8b-158/CAMPAIGN.md)
