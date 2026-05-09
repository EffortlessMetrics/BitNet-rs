# MODEL-CAPS-001 Dense Verify Capability

## Summary

`MODEL-CAPS-001` adds a dense model capability summary to `bitnet model verify`
for the supported Qwen2.5 0.5B GGUF cache artifacts. This is metadata-only: it
does not change model loading, tokenizer behavior, prompt templates,
transformer math, CUDA, dense GGUF inference, server behavior, speed claims, or
full-residency claims.

## Behavior

`bitnet model verify qwen2.5-0.5b-instruct-q8_0 --json` now includes a
`model_capability` section that identifies the artifact as a dense Qwen SLM GGUF
artifact for the Apple M4 CPU/NEON SLM answer lane.

`bitnet model verify qwen2.5-0.5b-instruct-q4_k_m --json` also includes a
`model_capability` section, but its permitted claims are limited to artifact
inspection and storage/reference use because strict Rust execution remains
unsupported for that artifact.

BitNet contract artifacts keep using `model_contract` and do not emit the dense
`model_capability` summary.

## Claim Boundary

This PR may claim that Qwen dense SLM cache verification now exposes model
family, artifact class, tokenizer authority, prompt authority, route boundary,
permitted claims, and required receipt metadata.

It must not claim dense CUDA inference, BitNet packed QK256 proof, CPU/CUDA
speedup, server readiness, or full CUDA residency.
