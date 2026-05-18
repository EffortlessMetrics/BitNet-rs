# Falcon-E Family source map

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: docs/proposals/BITNET-PROP-0013-falcon-e-family-supported-models.md
Linked specs: docs/specs/BITNET-SPEC-FALCON-E-FAMILY-ARTIFACT-CONTRACT.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-ROUTE-COMPATIBILITY.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-TOKENIZER-PROMPT.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-REFERENCE-QUALITY.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-I2S.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-TL1-TL2.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-CPU.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-CUDA.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-APPLE.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-A770-OPENCL.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-PERFORMANCE.md
Linked ADRs: n/a
Linked plan: plans/falcon-e-family/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: planning-only family registration
Policy impact: no policy exception

## Role

Falcon-E Family is the compact direct-GGUF 1.58-bit validation lane for
BitNet-rs. It is intentionally separate from Microsoft BitNet 2B, 1bitLLM,
Falcon3, dense Falcon, and dense SLM support.

## Source map

| Priority | Artifact | Source | File | Route | Initial claim |
|---:|---|---|---|---|---|
| 1 | Falcon-E-1B-Instruct-GGUF | `tiiuae/Falcon-E-1B-Instruct-GGUF` | `ggml-model-i2_s.gguf` | `I2_S` | registered candidate only |
| 2 | Falcon-E-3B-Instruct-GGUF | `tiiuae/Falcon-E-3B-Instruct-GGUF` | `ggml-model-i2_s.gguf` | `I2_S` | registered candidate only |

## Support ladder

| Level | State | Meaning | Allowed claim |
|---:|---|---|---|
| 0 | `registered` | Repo knows the family and candidate artifacts. | Planning only. |
| 1 | `artifact_inventory` | Exact repo, revision, file, size, SHA256 captured. | Artifact known. |
| 2 | `artifact_authorized` | Direct GGUF or approved conversion path exists. | Candidate only. |
| 3 | `structurally_valid` | Rust loader parses GGUF and classifies tensors. | Structural proof. |
| 4 | `runner_verified` | Reference runner loads artifact. | Runner path proof. |
| 5 | `reference_good` | Deterministic reference corpus passes. | Reference quality candidate. |
| 6 | `cpu_answer_ready` | Rust CPU path passes strict corpus with `fallback=false`. | CPU answer support. |
| 7 | `accelerator_answer_ready` | Exact backend receipts pass. | Exact backend support. |
| 8 | `benchmark_qualified` | Exact-profile benchmark accepted/rejected. | Profile-specific performance. |
| 9 | `product_cli_ready` | User CLI surfaces work with receipts. | User-facing support. |
| 10 | `server_exact_profile_ready` | Bounded server profile passes. | Exact server profile only. |

## Hard rails

```text
Falcon-E proof is not Falcon3 proof.
Falcon-E proof is not Microsoft BitNet 2B proof.
Falcon-E proof is not 1bitLLM proof.
Falcon-E proof is not dense Falcon proof.
Falcon-E-1B proof is not Falcon-E-3B proof.
I2_S proof is not TL1/TL2 proof.
TL1/TL2 proof is not I2_S/QK256 proof.
No third-party or alternate artifact substitution without artifact authority.
No speedup claim before exact-profile benchmark review.
No server readiness before exact-profile server receipts.
No model binaries committed.
```
