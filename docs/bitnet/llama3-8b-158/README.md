# Llama3 8B 1.58 Candidate Source Map

This directory records the BitNet-rs source map for
`HF1BitLLM/Llama3-8B-1.58-100B-tokens`.

## Current honest state

```text
registered candidate
upstream route support known
HF safetensors artifact visible
no approved BitNet-rs GGUF
no conversion authority
no reference-good BitNet-rs receipt
no CPU/CUDA/Apple product claim
```

## Route support to verify

| Architecture | I2_S | TL1 | TL2 | BitNet-rs state |
| --- | ---: | ---: | ---: | --- |
| x86 | listed | unsupported upstream | listed | `listed_supported_verify_runner` |
| ARM | listed | listed | unsupported upstream | `listed_supported_verify_runner` for listed routes |

The table mirrors upstream support only as a starting point. BitNet-rs must
still prove exact artifact identity, conversion or runner authority, tokenizer
and prompt authority, layout compatibility, scalar oracle parity, and backend
receipts before any support claim.

## Artifact boundary

The Hugging Face source repository is treated as a safetensors candidate until
an approved GGUF/conversion/runner path is recorded. Required inventory fields
include source revision, file list, file sizes, SHA256 values, config hashes,
tokenizer hashes, displayed metadata, upstream parameter-count label, identity
ambiguity, storage context, and cleanup status.

No exact hashes means no artifact claim. No tokenizer authority means no answer
claim. No approved GGUF/conversion/runner route means no Rust backend proof.

## Prompt and tokenizer boundary

The model is Llama3-derived. The Microsoft 2B `bitnetcpp-answer` prompt
contract must not be reused unless a prompt-authority audit proves it is correct
for this exact model. Required prompt evidence includes rendered prompts, prompt
token IDs, chat template, special token IDs, generation config, stop-token
policy, and whether the model repository tokenizer or an external Meta Llama3
tokenizer is authoritative.

## Route-family boundary

- `I2_S` routes may use QK256 scalar/backend kernels only after layout proof.
- `TL1` and `TL2` need separate tensor-layout specs, fixtures, and scalar
  oracles.
- x86 `TL2` proof does not follow from x86 `I2_S` proof.
- ARM `TL1` proof does not follow from ARM `I2_S` proof.
- CUDA, Apple, server, and speed claims require exact-profile receipts.

## First proof ladder

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
