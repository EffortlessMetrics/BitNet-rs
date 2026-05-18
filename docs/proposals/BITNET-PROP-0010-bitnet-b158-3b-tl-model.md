# BITNET-PROP-0010: BitNet b1.58 3B TL model lane

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: n/a
Linked specs: [3B artifact contract](../specs/BITNET-SPEC-B158-3B-ARTIFACT-CONTRACT.md), [3B conversion contract](../specs/BITNET-SPEC-B158-3B-CONVERSION.md), [3B TL1/TL2 layout](../specs/BITNET-SPEC-B158-3B-TL1-TL2-LAYOUT.md), [3B tokenizer and prompt](../specs/BITNET-SPEC-B158-3B-TOKENIZER-PROMPT.md), [3B reference quality](../specs/BITNET-SPEC-B158-3B-REFERENCE-QUALITY.md), [3B CPU](../specs/BITNET-SPEC-B158-3B-CPU.md), [3B CUDA](../specs/BITNET-SPEC-B158-3B-CUDA.md), [3B Apple](../specs/BITNET-SPEC-B158-3B-APPLE.md), [3B performance](../specs/BITNET-SPEC-B158-3B-PERFORMANCE.md)
Linked ADRs: n/a
Linked plan: [3B TL implementation plan](../../plans/bitnet-b158-3b/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: registers a diagnostic TL-model lane only; no answer, backend, server, or speed promotion
Policy impact: no policy exception

## Thesis

`1bitLLM/bitnet_b1_58-3B` is a larger supported BitNet b1.58 model that can
validate BitNet-rs beyond the official Microsoft 2B `I2_S`/QK256 artifact.
Because upstream support is listed as TL2 on x86 and TL1 on ARM, and because no
official GGUF is present in the current Hugging Face file listing, this lane is
an artifact, conversion, and runner-authority project before it is a CPU, CUDA,
Apple, or performance project.

## Why this model is valuable

- It exercises a larger 3.3B BitNet b1.58 artifact instead of only the official
  Microsoft 2B artifact.
- It forces TL1/TL2 kernel and layout support to become first-class in
  BitNet-rs rather than treating QK256 as the only BitNet path.
- It is relevant to Apple demonstrations because upstream lists ARM TL1 support
  and shows 3B-class Apple usage, but BitNet-rs must still prove the exact route
  and artifact before making Apple claims.
- It gives x86, CUDA, and A770 lanes a future TL2 target after CPU TL2 reference
  proof exists.

## Why this lane is risky

- The current Hugging Face listing exposes sharded safetensors and tokenizer /
  config files, not an official `.gguf` artifact.
- Upstream lists x86 TL2 support for the model, but the shown setup helper only
  exposes `--quant-type {i2_s,tl1}`; x86 TL2 therefore needs a reproducible
  conversion and runner command before it can become proof authority.
- Upstream marks `I2_S` unsupported for this model on both x86 and ARM.
- TL1 and TL2 are route-specific proof families. TL1 proof does not prove TL2,
  and TL2 proof does not prove TL1.

## Non-inheritance rules

This proposal explicitly rejects proof inheritance from other lanes:

- Official Microsoft 2B `I2_S`/QK256 answer, CPU, CUDA, or Apple proof does not
  prove this 3B model.
- Dense Qwen or other dense SLM proof does not prove this 3B BitNet model.
- `I2_S`/QK256 rejection or success for another artifact does not prove TL1 or
  TL2 layout correctness for this artifact.
- A third-party GGUF may be diagnostic only until an artifact-authority decision
  approves its provenance, hashes, tokenizer policy, and route metadata.

## Product ladder

The only honest initial status is:

```text
registered / diagnostic candidate
blocked_no_official_gguf
TL1/TL2 runner path unverified
no answer-ready claim
no backend claim
```

Promotion must proceed through:

```text
artifact authority
→ conversion / runner authority
→ TL1/TL2 layout spec
→ reference-good output
→ Rust structural loader
→ scalar TL oracle
→ CPU answer proof
→ AVX/CUDA/Apple backend proof
→ benchmark qualification
→ product status
```

## Non-goals

- `I2_S` or QK256 enablement for the 3B artifact except unsupported-path or
  diagnostic rejection receipts.
- Dense SLM proof, dense regular-LLM CUDA proof, or dense prompt/template
  inheritance.
- Speedup, benchmark, server, or broad product-readiness claims.
- Full GPU residency or accelerator execution before reference-good and CPU TL
  answer proof exist.
- Committing model binaries or generated GGUF files to the repository.

## Acceptance

This proposal is accepted when the linked plan and specs make the 3B lane a
registered TL candidate with explicit artifact inventory, conversion, TL layout,
tokenizer/prompt, reference-quality, CPU, CUDA, Apple, and performance claim
boundaries, while keeping all answer, backend, server, and speed claims false
until receipt-backed gates pass.
