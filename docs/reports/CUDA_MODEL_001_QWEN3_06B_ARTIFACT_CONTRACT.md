# CUDA-MODEL-001 Qwen3 0.6B Artifact Contract

## Scope

`CUDA-MODEL-001` records the exact Qwen3 0.6B Q8_0 artifact contract for the
RTX 5070 Ti productization lane. This is a source-of-truth contract for future
onboarding work, not a CPU answer, CUDA execution, benchmark, or server proof.

## Artifact

| Field | Value |
|---|---|
| Model id | `qwen3-0.6b-instruct-q8_0` |
| Contract id | `qwen3_0_6b_q8_0` |
| Source | Hugging Face |
| Repository | `Qwen/Qwen3-0.6B-GGUF` |
| Revision | `23749fefcc72300e3a2ad315e1317431b06b590a` |
| File | `Qwen3-0.6B-Q8_0.gguf` |
| SHA256 | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` |
| Bytes | `639446688` |
| Format | `gguf` |
| GGUF version | `GGUF V3` |
| Architecture | `qwen3` |
| Quantization | `Q8_0` |
| License | `apache-2.0` |

## Metadata Contract

| Field | Value |
|---|---|
| Context length | `40960` |
| Block count | `28` |
| Embedding length | `1024` |
| Attention heads | `16` |
| Attention KV heads | `8` |
| Tokenizer source | `gguf_metadata` |
| Tokenizer model | `gpt2` |
| Pre-tokenizer | `qwen2` |
| Chat template | present, Qwen3 ChatML policy required before answer claims |
| BOS token | `151643` |
| EOS token | `151645` |
| Padding token | `151643` |

## Storage And Runtime Envelope

The pinned artifact is 639,446,688 bytes, about 609.82 MiB. The contract records
`fits_local_small_slm_storage` as the storage envelope and uses a conservative
candidate-only CUDA memory envelope of at least the artifact size with 1.5 GiB
recommended headroom before route planning.

The memory envelope is not a CUDA residency claim. CUDA residency must be
measured by later strict receipts.

## Proof State

`ci/model-artifacts/dense-slm-model-contracts.toml` records the contract as
`structurally_valid` metadata only. The coverage row stays claim-safe:

- `cpu_answer_ready = false`
- `accelerator_answer_ready = false`
- `benchmark_qualified = false`
- `product_cli_ready = false`
- `server_ready = false`
- `speedup_claim = false`
- `full_residency_claim = false`
- `bitnet_packed_i2s_qk256_proof = false`
- `dense_regular_llm_cuda_proof = false`

## Known Boundary

Existing Qwen3 work has not promoted CPU answer sanity. The current evidence
includes reference/checkpoint divergence work, including an output-head audit
that found tied token embeddings rather than a missing dedicated output head.
The next step is CPU answer sanity or a precise blocker receipt, not CUDA proof.

## Source Artifacts

- `ci/model-artifacts/dense-slm-model-contracts.toml`
- `ci/model-artifacts/model-coverage-matrix.toml`
- `ci/quality/apple-m4-slm-model-breadth-reference-sanity.toml`
- `ci/quality/slm-answer-corpus.yaml`
- `ci/slm-cpu/model-candidates.toml`
- `ci/slm-cpu/intel-i5-8250u/2026-05-07/qwen3-post-008w-tied-head-audit.json`
- `ci/slm-cpu/intel-i5-8250u/2026-05-07/qwen3-qproj-drift-diagnosis.json`
- `docs/slm/SLM_CPU_8250U_RUNBOOK.md`
- `docs/slm/SLM_REFERENCE_DIVERGENCE.md`

## Claim Boundary

This report may claim that Qwen3 0.6B Q8_0 has a pinned artifact contract and
structural metadata in the RTX 5070 Ti onboarding lane. It must not claim Qwen3
CPU answer quality, strict CUDA execution, BitNet QK256 proof, speedup, server
readiness, or inheritance from Qwen2.5 receipts.
