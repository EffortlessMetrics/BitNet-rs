# BitNet b1.58 3B TL candidate source map

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0010](../../proposals/BITNET-PROP-0010-bitnet-b158-3b-tl-model.md)
Linked specs: [artifact](../../specs/BITNET-SPEC-B158-3B-ARTIFACT-CONTRACT.md), [conversion](../../specs/BITNET-SPEC-B158-3B-CONVERSION.md), [TL layout](../../specs/BITNET-SPEC-B158-3B-TL1-TL2-LAYOUT.md), [tokenizer/prompt](../../specs/BITNET-SPEC-B158-3B-TOKENIZER-PROMPT.md), [quality](../../specs/BITNET-SPEC-B158-3B-REFERENCE-QUALITY.md), [CPU](../../specs/BITNET-SPEC-B158-3B-CPU.md), [CUDA](../../specs/BITNET-SPEC-B158-3B-CUDA.md), [Apple](../../specs/BITNET-SPEC-B158-3B-APPLE.md), [performance](../../specs/BITNET-SPEC-B158-3B-PERFORMANCE.md)
Linked ADRs: n/a
Linked plan: [3B TL implementation plan](../../../plans/bitnet-b158-3b/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: diagnostic candidate only
Policy impact: no policy exception

## Source map

| Field | Value |
| --- | --- |
| Model id | `onebitllm_bitnet_b158_3b` |
| Upstream repo | `1bitLLM/bitnet_b1_58-3B` |
| Current pinned revision for initial inventory | `af89e318d78a70802061246bf037199d2fb97020` |
| Source format | sharded safetensors plus tokenizer/config/Python model assets |
| Official GGUF status | `blocked_no_official_gguf` until inventory proves otherwise |
| x86 candidate route | TL2 listed upstream, runner/conversion unverified |
| ARM candidate route | TL1 listed upstream, runner/conversion unverified |
| Forbidden route | `I2_S`/QK256 except diagnostic rejection receipts |

## Current claim boundary

BitNet-rs may say only:

```text
1bitLLM/bitnet_b1_58-3B is a registered BitNet-rs TL candidate with guarded
artifact inventory and runner/conversion verification work pending.
```

BitNet-rs must not say:

- 3B `I2_S` is supported.
- 3B inherits the official Microsoft 2B `I2_S`/QK256 proof.
- 3B inherits dense Qwen or dense SLM proof.
- x86 TL2, ARM TL1, CUDA TL2, OpenCL TL2, Apple TL1, server, or speedup is
  answer-ready before route-specific receipts pass.

## First proof question

The first blocker is not a kernel optimization. The first blocker is:

```text
Do we have a verified TL1/TL2 artifact and reference runner that produces
coherent deterministic output for the exact 3B artifact, tokenizer, and prompt
policy?
```

Only after that question is answered may Rust structural loading, scalar TL
oracles, CPU answer proof, accelerator proof, benchmark qualification, and
product CLI/server promotion proceed.
