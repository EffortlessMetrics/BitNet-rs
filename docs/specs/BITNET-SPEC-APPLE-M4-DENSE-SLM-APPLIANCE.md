# BITNET-SPEC-APPLE-M4-DENSE-SLM-APPLIANCE

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0005 Apple Silicon productization](../proposals/BITNET-PROP-0005-apple-silicon-productization.md)
Linked specs: [Apple Silicon route contract](BITNET-SPEC-APPLE-SILICON-ROUTE-CONTRACT.md)
Linked ADRs: n/a
Linked plan: [Apple Silicon implementation plan](../../plans/apple-silicon/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no support promotion; codifies existing supported dense SLM states
Policy impact: no policy exception

## Purpose

Define the Apple M4 dense SLM appliance path. This spec inherits the current
M4 dense SLM support matrix and makes its promotion gates contractual for the
`apple_m4_cpu_neon_dense_slm` proof family.

## Supported states

| State | Meaning |
| --- | --- |
| `default` | The default dense M4 SLM for normal Mac appliance paths. |
| `supported` | Explicitly supported non-default model with all required gates. |
| `candidate` | Exact artifact may be evaluated but is not supported. |
| `diagnostic-only` | Useful for loader/tokenizer/debug work but not product support. |
| `rejected` | Outside the lane until a separate architecture/artifact campaign reopens it. |

## Current model states

| Model ID | State | Notes |
| --- | --- | --- |
| `qwen2.5-0.5b-instruct-q8_0` | `default` | Remains default unless a promotion review changes it. |
| `qwen2.5-0.5b-instruct-q4_k_m` | `supported` | Storage-conscious supported non-default row. |
| `qwen2.5-1.5b-instruct-q4_k_m` | `supported` | Larger supported non-default row. |
| Qwen3, SmolLM, Gemma, Phi candidates | `candidate` or `diagnostic-only` | Must pass exact gates before support. |
| Qwen3.5, hybrid, vision, MoE, state-space, random unpinned GGUFs | `rejected` unless reopened by a separate campaign | Not automatically supported. |

## Required model gates

A dense SLM may be `default` or `supported` only when receipts record:

- source repository and revision;
- file name, byte size, and SHA256;
- GGUF architecture;
- quantization;
- tokenizer model and pre-tokenizer authority;
- prompt-template authority;
- strict cache verification;
- reference-runner sanity;
- Rust M4 quality;
- generated text, prompt token IDs, and generated token IDs;
- timing;
- `fallback_used = false`;
- unsupported-backend failure behavior;
- 500-case deterministic corpus where promoted;
- matching-history receipts for operator envelope.

## Claim boundary

Dense SLM evidence is dense SLM evidence only. It does not prove BitNet,
QK256, full Apple Metal inference, Neural Engine execution, MPSGraph model
inference, MacBook behavior, CUDA, x86, or broad Apple Silicon performance.
