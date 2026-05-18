# BITNET-PROP-0009: bitnet_b1_58-large control model

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: n/a
Linked specs: [artifact contract](../specs/BITNET-SPEC-B158-LARGE-ARTIFACT-CONTRACT.md), [conversion contract](../specs/BITNET-SPEC-B158-LARGE-CONVERSION.md), [tokenizer and prompt contract](../specs/BITNET-SPEC-B158-LARGE-TOKENIZER-PROMPT.md), [reference quality contract](../specs/BITNET-SPEC-B158-LARGE-REFERENCE-QUALITY.md), [CPU contract](../specs/BITNET-SPEC-B158-LARGE-CPU.md), [CUDA contract](../specs/BITNET-SPEC-B158-LARGE-CUDA.md), [Apple contract](../specs/BITNET-SPEC-B158-LARGE-APPLE.md), [performance contract](../specs/BITNET-SPEC-B158-LARGE-PERFORMANCE.md)
Linked ADRs: n/a
Linked plan: [bitnet_b1_58-large implementation plan](../../plans/bitnet-b158-large/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no support promotion; establishes source-of-truth rails only
Policy impact: no policy exception

## Thesis

`1bitLLM/bitnet_b1_58-large` is a smaller 0.7B BitNet b1.58 control model. It
can help BitNet-rs debug model-family assumptions, tokenizer and prompt
authority, safetensors-to-GGUF conversion, `I2_S`/`TL1`/`TL2` route behavior,
Apple storage constraints, and later CPU/CUDA performance without always relying
on the larger official Microsoft 2B artifact.

The first blocker is artifact and conversion authority, not backend speed.
BitNet-rs should treat the model as a supported-upstream, non-Microsoft control
candidate until exact artifacts, tokenizer authority, prompt rendering,
conversion reproducibility, reference output, and backend receipts prove a
narrower claim.

## Motivation

Upstream bitnet.cpp lists `bitnet_b1_58-large` as a supported 0.7B model with
x86 `I2_S`/`TL2` and ARM `I2_S`/`TL1` support. The upstream setup surface also
names `1bitLLM/bitnet_b1_58-large` as an `--hf-repo` option and exposes
`--quant-type {i2_s,tl1}`. That makes the model valuable as a smaller BitNet
control artifact.

The current Hugging Face repository for `1bitLLM/bitnet_b1_58-large` exposes a
`safetensors` model, tokenizer/config files, and Python model assets rather than
an official `.gguf` file in the inspected file listing. The Apple candidate
matrix already records the same blocker at revision
`85d047191dcb224f0e04f20d26110caaf8dc1a47`: the candidate is useful, but no
official GGUF or approved conversion/runner path exists yet.

## Product claim boundary

The first public claim must remain narrow:

```text
1bitLLM/bitnet_b1_58-large is a supported BitNet-rs control model only after
artifact authority, tokenizer authority, conversion/reproducibility,
reference-output quality, and backend receipts pass.
```

Do not claim:

- all `1bitLLM` BitNet models work;
- all BitNet b1.58 variants work;
- the official Microsoft 2B `I2_S`/QK256 proof applies to this 0.7B model;
- dense Qwen or other dense SLM proof applies to BitNet;
- performance is good because a smaller model loaded once.

## Source-of-truth links

- Source map: [docs/bitnet/bitnet-b158-large/README.md](../bitnet/bitnet-b158-large/README.md)
- Plan: [plans/bitnet-b158-large/implementation-plan.md](../../plans/bitnet-b158-large/implementation-plan.md)
- Apple candidate matrix: [docs/apple-silicon/bitnet-candidate-matrix.md](../apple-silicon/bitnet-candidate-matrix.md)
- Shared answer gate: [docs/model-artifacts/ANSWER_ARTIFACT_GATE.md](../model-artifacts/ANSWER_ARTIFACT_GATE.md)

## Goals

- Register `bitnet_b1_58-large` as a first-class control-model lane without
  promoting answer, backend, server, or speed claims.
- Define exact artifact inventory requirements for the HF safetensors source and
  any converted GGUF target.
- Define conversion lanes for F16 structural/reference GGUF and future
  upstream-compatible `I2_S`, `TL1`, and `TL2` artifacts.
- Require tokenizer, pre-tokenizer, prompt-template, stop-token, rendered prompt,
  and prompt-token-ID authority before answer claims.
- Gate CPU, CUDA, Apple, performance, CLI, and server promotion on reference-good
  output and strict fallback-explicit receipts.

## Non-goals

- Committing model binaries.
- Treating third-party GGUF files as authority without a separate decision.
- Treating `bitnet-st2gguf` F16 output as packed BitNet `I2_S`, `TL1`, or `TL2`
  proof.
- Promoting CPU, CUDA, Apple, Metal, OpenCL, NPU, CLI, server, or speed support
  in this proposal.

## Support ladder

| Level | Status | Meaning | Public claim |
| ---: | --- | --- | --- |
| 0 | `registered` | Repo knows the model exists and upstream lists it. | Planning only. |
| 1 | `artifact_discovered` | Exact files, revision, sizes, and hashes are recorded. | Artifact inventory. |
| 2 | `conversion_candidate` | Reproducible conversion path exists, or official GGUF is found. | Candidate only. |
| 3 | `structurally_valid` | BitNet-rs can parse/load the artifact and classify tensors. | Structural loading only. |
| 4 | `reference_good` | Reference runner produces coherent deterministic prompt-suite output. | Reference quality candidate. |
| 5 | `cpu_answer_ready` | Rust CPU path passes strict answer corpus with fallback false. | CPU answer support. |
| 6 | `accelerator_answer_ready` | CUDA/Apple/A770/etc. passes strict one-token, short, and warm receipts. | Exact backend support. |
| 7 | `benchmark_qualified` | Exact-profile benchmark review accepts or rejects speed. | Profile-specific performance only. |
| 8 | `product_cli_ready` | `model status`, `model verify`, `ask`, `chat`, `bench`, and `receipts explain` work. | Exact CLI support. |
| 9 | `server_exact_profile_ready` | Exact non-stream and stream server routes pass. | Exact server profile only. |

The truthful starting status is between `registered` and `artifact_discovered`:
BitNet-rs records the upstream-supported model and blocks answer claims until
artifact and conversion authority exists.

## Acceptance

This proposal is accepted when the source map, implementation plan, and linked
spec contracts make `bitnet_b1_58-large` a claim-safe control-model lane while
preserving these hard rules:

- no exact file hashes, no artifact claim;
- no tokenizer authority, no answer claim;
- no reference runner output, no backend answer work;
- no third-party GGUF without explicit artifact-authority approval;
- no model binaries committed;
- no inherited official Microsoft 2B or dense SLM proof;
- no speedup claim before exact-profile benchmark review.
