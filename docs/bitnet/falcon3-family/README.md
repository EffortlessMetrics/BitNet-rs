# Falcon3 Family Source Map

Status: draft
Owner: model-artifacts
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0012](../../proposals/BITNET-PROP-0012-falcon3-family-supported-models.md)
Linked specs: [Falcon3 specs](../../specs/INDEX.md#falcon3-family-onboarding)
Linked ADRs: [BITNET-ADR-0005](../../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Linked plan: [Falcon3 family implementation plan](../../../plans/falcon3-family/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: planning/registered only
Policy impact: no policy exception

## Current Honest Status

```text
registered candidate family
not answer-ready
not CPU answer-ready
not CUDA answer-ready
not Apple answer-ready
not A770/OpenCL answer-ready
not benchmark-qualified
not server-ready
```

Falcon3 is being registered as a multi-size BitNet-family onboarding lane. Registration records that candidate repositories exist and that direct I2_S GGUFs are the first intended proof path for 1B and 7B. It does not prove loader compatibility, prompt correctness, answer quality, backend execution, speed, or server readiness.

## Artifact Source Map

| Priority | Artifact ID | Source repo | Source format | File / route | Initial posture |
| ---: | --- | --- | --- | --- | --- |
| 1 | `falcon3_1b_instruct_158_i2s_gguf` | `tiiuae/Falcon3-1B-Instruct-1.58bit-GGUF` | GGUF | `ggml-model-i2_s.gguf` / `i2_s` | first direct I2_S target |
| 2 | `falcon3_7b_instruct_158_i2s_gguf` | `tiiuae/Falcon3-7B-Instruct-1.58bit-GGUF` | GGUF | `ggml-model-i2_s.gguf` / `i2_s` | second direct I2_S target |
| 3 | `falcon3_3b_instruct_158_safetensors` | `tiiuae/Falcon3-3B-Instruct-1.58bit` | safetensors / Transformers | conversion-required `i2_s` | conversion/runner target |
| 4 | `falcon3_10b_instruct_158_safetensors` | `tiiuae/Falcon3-10B-Instruct-1.58bit` | safetensors / Transformers | conversion-required `i2_s` | conversion/runner target after 1B/7B |

Artifact inventory receipts must record both nominal model size and Hugging Face displayed model-size metadata when a probe observes a discrepancy.

## Support Ladder

| Level | State | Meaning | Allowed claim |
| ---: | --- | --- | --- |
| 0 | `registered` | Falcon3 family and candidate repos are known. | Planning only. |
| 1 | `artifact_inventory` | Exact repo, revision, files, sizes, hashes captured. | Artifact known. |
| 2 | `artifact_authorized` | Official/approved GGUF or conversion route exists. | Candidate only. |
| 3 | `structurally_valid` | Rust loader parses artifact and classifies tensor roles. | Structural proof. |
| 4 | `runner_verified` | Reference runner loads the artifact. | Runner path proof. |
| 5 | `reference_good` | Deterministic reference corpus passes. | Reference quality candidate. |
| 6 | `cpu_answer_ready` | Rust CPU path passes strict corpus with fallback false. | CPU answer support. |
| 7 | `accelerator_answer_ready` | CUDA/AVX/Apple/A770 route passes strict receipts. | Exact backend support. |
| 8 | `benchmark_qualified` | Exact-profile benchmark accepted/rejected. | Profile-specific performance. |
| 9 | `product_cli_ready` | `verify`, `status`, `ask`, `chat`, `bench`, `receipts explain` work. | User-facing support. |
| 10 | `server_exact_profile_ready` | Bounded server profile passes. | Exact server profile only. |

## Claim Boundary

- Falcon3 proof is not Falcon-E proof.
- Falcon3 proof is not Microsoft BitNet 2B proof.
- Falcon3 proof is not Llama3-8B-1.58 proof.
- Falcon3 proof is not dense Falcon3 / dense SLM proof.
- Falcon3 1B proof is not 7B, 3B, or 10B proof.
- I2_S proof is not TL1/TL2 proof.
- TL1/TL2 proof is not I2_S/QK256 proof.
- x86 TL2 and ARM TL1 remain listed-supported-but-unpromoted until runner/conversion receipts exist.
- No model binaries may be committed.
