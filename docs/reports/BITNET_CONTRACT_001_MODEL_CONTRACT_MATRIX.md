# BITNET-CONTRACT-001 Model Contract Matrix

## Summary

This change makes BitNet-family artifact coverage explicit. The repo now records
which model family, artifact format, kernel family, architecture support,
tokenizer and prompt authority, CPU oracle, accelerator route, permitted claims,
and required receipts apply before a BitNet artifact can be used as proof.

The current reference lane remains the official Microsoft
`BitNet-b1.58-2B-4T` GGUF `I2_S` artifact with external `llama-bpe` tokenizer
authority and the `bitnetcpp-answer` prompt template.

## Contract Decisions

| Contract | State | Claim boundary |
|---|---|---|
| Official Microsoft 2B `I2_S` | `reference_ready` | May anchor x86 CPU and RTX 5070 Ti CUDA answer/parity/benchmark-baseline receipts. Speedup and full-residency claims still need profile-specific receipts. |
| Official Microsoft 2B `TL1` | `planned_proof_required` | ARM lane only until TL1 parser, fixtures, answer corpus, tokenizer/prompt authority, and backend receipts exist. |
| Official Microsoft 2B `TL2` | `planned_proof_required` | x86 alternate only until TL2 parser, AVX fixtures, answer corpus, tokenizer/prompt authority, and benchmark receipts exist. |
| 3B x86 `I2_S` | `upstream_unsupported` | Diagnostic-only. It cannot be answer, reference, backend-parity, or speed authority. |
| 3B x86 `TL2` | `listed_verify_runner` | Listed upstream, but needs runner-path verification before proof claims. |
| 3B ARM `TL1` | `listed_verify_runner` | Listed upstream, but needs runner-path verification before proof claims. |
| `tdh111` IQ2_BN_R4 | `alternate_control` | Useful control evidence, not an official Microsoft I2_S CUDA unblocker. |

## Files

- `ci/model-artifacts/bitnet-model-contracts.toml` is the machine-readable
  contract ledger.
- `crates/bitnet-models/src/model_contracts.rs` exposes a typed registry for
  future verifier/planner work.
- `docs/bitnet/BITNET_MODEL_CONTRACT.md` now describes the per-artifact matrix.

## Claim Boundary

This change does not modify model loading, tokenizer behavior, prompt rendering,
transformer math, QK256 kernels, CUDA behavior, dense GGUF inference, server
runtime, quality gates, or speed claims.
