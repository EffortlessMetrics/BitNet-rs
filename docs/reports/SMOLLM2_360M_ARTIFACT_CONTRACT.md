# SmolLM2 360M Artifact Contract

## Scope

`CUDA-MODEL-SMOLLM2-001` records the exact SmolLM2 360M Instruct Q8_0
artifact contract for the RTX 5070 Ti productization lane. This is an
artifact-identity and structural-metadata contract only. It is not a CPU answer,
CUDA execution, benchmark, server, full-residency, broad dense GGUF, or BitNet
QK256 proof.

## Artifact

| Field | Value |
|---|---|
| Model id | `smollm2-360m-instruct` |
| Artifact id | `smollm2-360m-instruct-q8_0` |
| Contract id | `smollm2_360m_instruct_q8_0` |
| Source | Hugging Face |
| Repository | `HuggingFaceTB/SmolLM2-360M-Instruct-GGUF` |
| Base model | `HuggingFaceTB/SmolLM2-360M-Instruct` |
| Revision | `593b5a2e04c8f3e4ee880263f93e0bd2901ad47f` |
| File | `smollm2-360m-instruct-q8_0.gguf` |
| SHA256 | `48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201` |
| Bytes | `386404992` |
| Format | `gguf` |
| GGUF version | `GGUF V3` |
| Architecture | `llama` |
| Quantization | `Q8_0` |
| License | `apache-2.0` |

## Metadata Contract

| Field | Value |
|---|---|
| Context length | `8192` |
| Block count | `32` |
| Embedding length | `960` |
| Intermediate length | `2560` |
| Attention heads | `15` |
| Attention KV heads | `5` |
| Vocab size | `49152` |
| Tied token embeddings | `true` |
| Tokenizer source | `gguf_metadata` |
| Tokenizer model | `gpt2` |
| Pre-tokenizer | `smollm` |
| Chat template | present, SmolLM2 ChatML with explicit system prompt |
| BOS token | `1` |
| EOS token | `2` |
| Padding token | `2` |

## Storage And Runtime Envelope

The pinned artifact is 386,404,992 bytes, about 368.50 MiB. The contract records
`fits_local_small_slm_storage` as the storage envelope and uses a conservative
candidate-only CUDA memory envelope of at least the artifact size with 1 GiB
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

Earlier Apple M4 model-breadth evidence accepted the SmolLM2 reference-runner
output, but the current Rust M4 support gate rejected the artifact before
generation. That evidence is useful source context, but it does not promote the
RTX 5070 Ti lane. The next NVIDIA work is same-box CPU sanity and a dense CUDA
route plan before any strict CUDA one-token proof.

## Source Artifacts

- `ci/model-artifacts/dense-slm-model-contracts.toml`
- `ci/model-artifacts/model-coverage-matrix.toml`
- `ci/quality/apple-m4-slm-model-breadth-reference-sanity.toml`
- `ci/quality/apple-m4-slm-model-breadth-rust-m4-quality.toml`
- `docs/slm/apple-m4-slm-model-breadth-candidates.md`
- `docs/slm/apple-m4-slm-model-breadth-reference-sanity.md`
- `docs/slm/apple-m4-slm-model-breadth-rust-m4-quality.md`

## Claim Boundary

This report may claim that SmolLM2 360M Instruct Q8_0 has a pinned artifact
contract and structural metadata in the RTX 5070 Ti onboarding lane. It must not
claim SmolLM2 CPU answer quality, strict CUDA execution, dense CUDA proof,
BitNet QK256 proof, speedup, server readiness, full CUDA residency, broad dense
GGUF support, or inheritance from Qwen2.5, Qwen3, or Apple M4 receipts.
