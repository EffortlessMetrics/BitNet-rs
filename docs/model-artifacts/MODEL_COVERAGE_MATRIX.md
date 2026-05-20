# Model Coverage Matrix

`ci/model-artifacts/model-coverage-matrix.toml` is the cross-family claim
surface for local inference coverage. It complements the BitNet-family contract
registry and the dense Qwen capability summaries by showing where each model
lane sits in the proof ladder.

## Coverage Tiers

| Tier | Meaning |
|---|---|
| `registered` | The repo knows the model family and artifact class. |
| `structurally_valid` | The artifact parses and tensor roles are classified. |
| `reference_good` | A reference runner or accepted external evidence produced bounded coherent output. |
| `cpu_answer_ready` | The Rust CPU path has strict answer receipts. |
| `accelerator_answer_ready` | A strict accelerator path has fallback-free one-token, short-decode, or warm-session receipts. |
| `benchmark_qualified` | Exact profiles have governed same-artifact benchmark qualification receipts. |
| `product_cli_ready` | Normal user CLI paths exist for verified ask/chat/bench receipt surfaces; server readiness is still separate. |

Higher tiers do not erase the underlying claim boundary. For example, a model
can be CLI-ready for a bounded CUDA ask/chat path while still having
`speedup_claim=false` and `full_residency_claim=false`. Server readiness is a
separate exact-profile claim and is true only where the row names the accepted
server receipt and scope.

## Required Boundaries

- BitNet packed I2_S/QK256 proof and dense regular-LLM CUDA proof are separate
  claims.
- I2_S/QK256, TL1, TL2, BF16-to-GPU-packed-int2, and MCU fixture rows are
  separate BitNet product lanes. TL1/TL2 or GPU-packed progress cannot satisfy
  the official GGUF I2_S/QK256 proof.
- Unsupported upstream routes can be registered, but they cannot claim
  structural validity, answer-readiness, backend parity, speedup, or server
  readiness.
- Dense SLM and small-LLM entries must not claim BitNet packed proof.
- Speedup claims require benchmark qualification receipts for exact profiles.
- Product CLI readiness does not imply server readiness.

## BitNet Family Rows

`MODEL-COVERAGE-002` expands the BitNet side of the matrix beyond the current
official I2_S answer lane:

| Entry | Artifact lane | Current tier | Boundary |
|---|---|---|---|
| `bitnet_official_2b_i2s_qk256` | GGUF I2_S / QK256 | `product_cli_ready` | Current official x86/CUDA answer lane with strict RTX 5070 Ti `bitnet_qk256_cuda` server-smoke evidence only; speedup, full residency, dense regular-LLM CUDA, and broad production server readiness remain unclaimed. |
| `bitnet_official_2b_tl1_arm_candidate` | GGUF TL1 | `registered` | ARM-oriented candidate; needs TL1 layout, scalar, NEON/Apple proofs. |
| `bitnet_official_2b_tl2_x86_candidate` | GGUF TL2 | `registered` | x86 LUT candidate; needs TL2 runner and scalar/AVX proofs. |
| `bitnet_official_2b_bf16_gpu_int2_candidate` | BF16 master to GPU packed int2/W2A8 | `registered` | Separate GPU-reference path; does not satisfy GGUF I2_S proof. |
| `bitnet_3b_x86_i2s_unsupported` | 3B GGUF I2_S on x86 | `registered` | Upstream-unsupported; diagnostic/unsupported-path receipts only. |
| `bitnet_3b_x86_tl2_candidate` | 3B GGUF TL2 on x86 | `registered` | Listed candidate; needs runner verification before answer claims. |
| `bitnet_onebit_large_diagnostic` | 1bitLLM large-family artifact | `registered` | Diagnostic until family-specific artifact, tokenizer, prompt, and route contracts land. |
| `bitnet_llama3_8b_158_diagnostic` | Llama3-family 1.58-bit variant | `registered` | Diagnostic contract, not official BitNet answer authority. |
| `bitnet_falcon3_falcon_e_158_diagnostic` | Falcon-family 1.58-bit variant | `registered` | Diagnostic contract, not official BitNet answer authority. |
| `bitnet_mcu_tiny_fixture` | MCU low-bit fixture | `registered` | Arithmetic/kernel regression testbed only, not LLM answer authority. |

## Dense SLM Family Rows

`MODEL-COVERAGE-003` expands the dense SLM side of the matrix beyond the
current Qwen2.5 Q8_0 CUDA answer lane. These rows are coverage contracts: each
model family must earn its own tier from receipts and must not inherit Qwen2.5
CUDA proof.

| Entry | Artifact lane | Current tier | Boundary |
|---|---|---|---|
| `dense_qwen25_05b_q8_cuda` | Qwen2.5 0.5B Q8_0 GGUF | `product_cli_ready` | Current dense CUDA SLM answer lane. `server_ready=true` is promoted only for the refreshed non-streaming RTX 5070 Ti shared-engine `/v1/chat/completions` receipt with artifact checksum identity, endpoint profile, and greedy two-token generation policy under BITNET-SPEC-0010. The post-OPS requalification review keeps benchmark-qualified speed, speedup, full residency, broad dense GGUF readiness, BitNet proof, concurrency, and deployment readiness false. |
| `dense_qwen3_06b_q8_candidate` | Qwen3 0.6B Q8_0 GGUF | `product_cli_ready` | Product CLI-ready dense SLM row for bounded RTX 5070 Ti `dense_regular_llm_cuda` ask/chat user paths and exact-profile server readiness for the committed non-streaming shared-engine `/v1/chat/completions` receipt at `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/server-strict-dense-qwen3-q8-smoke.json`. The repeated-comparator contract exists, but speedup, benchmark-qualified speed, full-residency, broad dense GGUF, Qwen2.5 inheritance, and BitNet QK256 claims remain false until a hardware aggregate receipt and separate review land. |
| `dense_smollm2_360m_candidate` | SmolLM2 360M GGUF candidate | `structurally_valid` | Exact artifact contract plus strict CPU preflight blocker, normalization-policy audit, exact metadata-scoped validation, strict CPU retry evidence, wrong-first-token diagnosis, and comparator contract. The retry reaches one-token generation but fails the math quality gate; a same-prompt reference-compatible first-token/top-k or checkpoint comparator capture is required before CPU answer readiness or CUDA planning. |
| `dense_smollm2_17b_candidate` | SmolLM2 1.7B GGUF candidate | `registered` | Larger low-footprint pressure row; no answer or CUDA claim. |
| `dense_llama32_1b_candidate` | Llama 3.2 1B GGUF candidate | `registered` | Llama-family tokenizer/model-shape control; does not inherit Qwen proof. |
| `dense_llama32_3b_candidate` | Llama 3.2 3B GGUF candidate | `registered` | Larger Llama-family small-LLM pressure row; needs its own proof ladder. |
| `dense_gemma_small_candidate` | Gemma small GGUF candidate | `registered` | Alternate architecture coverage row; activation and attention policy must be proven separately. |
| `dense_phi_small_candidate` | Phi small GGUF candidate | `registered` | Phi-family tokenizer/model-quirk coverage row; needs its own boundary fixtures and proof receipts. |

## Selected Small-LLM Rows

`MODEL-COVERAGE-004` expands the selected small-LLM side of the matrix. These
rows are larger than the first dense SLM answer lane and are intended to expose
memory envelope, longer layer-stack, KV, LM-head, and warm-session pressure
before any user-facing or speed claim is allowed.

| Entry | Artifact lane | Current tier | Boundary |
|---|---|---|---|
| `small_llm_qwen25_15b_q4km_candidate` | Qwen2.5 1.5B Q4_K_M GGUF candidate | `cpu_answer_ready` | Supported non-default Apple M4 CPU/NEON answer model after reference and Rust M4 quality gates; a 5070 Ti strict CUDA all-layer-plan attempt currently fails closed at `strict_cuda_ready=false`, so it still needs Q4_K_M CUDA route support or an explicit unsupported-route receipt before any CUDA claim. |
| `small_llm_qwen3_17b_q8_candidate` | Qwen3 1.7B-class Q8_0 GGUF candidate | `registered` | Future Qwen3 small-LLM row; does not inherit Qwen2.5 0.5B CUDA receipts. |
| `small_llm_llama32_3b_candidate` | Llama 3.2 3B-class GGUF candidate | `registered` | Llama-family selected small-LLM row; needs its own memory, CPU, KV, and CUDA plan receipts. |
| `small_llm_gemma_2b_candidate` | Gemma 2B-class GGUF candidate | `registered` | Alternate architecture selected small-LLM row; cannot inherit Qwen, Llama, or BitNet proof. |

## Modern LLM Placeholders

`MODEL-COVERAGE-005` expands docs-only modern LLM placeholders without naming a
specific public artifact or claiming current hardware support. These rows exist
to keep future proof ladders explicit while the current 5070 Ti/local CPU lanes
remain scoped to verified BitNet, dense SLM, and selected small-LLM work.

| Entry | Artifact lane | Current tier | Boundary |
|---|---|---|---|
| `modern_llm_dense_frontier_placeholder` | Dense frontier-scale placeholder | `registered` | Docs-only future contract; no local runtime, CUDA, answer, speedup, or server claim. |
| `modern_llm_moe_frontier_placeholder` | MoE frontier-scale placeholder | `registered` | Docs-only future contract; expert routing and memory envelope must be specified before runtime work. |
| `modern_llm_multimodal_placeholder` | Multimodal frontier-scale placeholder | `registered` | Docs-only future contract; modality boundaries and hardware envelope must be specified before runtime work. |
| `modern_llm_placeholder_contract` | Generic docs-only placeholder | `registered` | Existing generic placeholder for future concrete artifact contracts. |

## CLI Surface

`MODEL-COVERAGE-006` exposes the matrix through the model command surface:

```powershell
bitnet model coverage
bitnet model coverage dense_qwen25_05b_q8_cuda
bitnet model coverage bitnet_official_2b_i2s_qk256 --json
```

The command is read-only. It reports current proof tier, routes, required
receipts, next proof, and claim boundary from
`ci/model-artifacts/model-coverage-matrix.toml`. It does not verify artifacts,
run inference, select a backend, execute CUDA, or upgrade answer, speedup,
server, or residency claims. If the matrix is outside the working directory,
pass `--matrix <PATH>` or set `BITNET_MODEL_COVERAGE_MATRIX`.

`MODEL-COVERAGE-007` adds the matching CI guard in the model-cache tests: every
artifact exposed through `bitnet model fetch/list/verify` must be represented by
the coverage matrix through a `capability_id`, `contract_id`, or verifier
surface. Adding a new supported model without a coverage row is a test failure,
so user-facing artifacts cannot bypass the registry claim boundary.

## Validation

Run:

```powershell
cargo run --release --locked -p xtask --no-default-features -- check-model-coverage
```

The validator parses the matrix, checks tier ordering, requires core lane
coverage, and rejects common claim leaks such as dense entries claiming BitNet
packed proof, TL1/TL2 rows claiming I2_S/QK256 proof, unsupported entries
claiming answer readiness, or SLM candidates claiming dense CUDA proof without
a `dense_regular_llm_cuda` accelerator route. Selected small-LLM rows must also
require a `memory_envelope` receipt before later proof work can build on them.
Modern LLM docs-only placeholders must stay route-free and require an
`unsupported_on_current_hardware_receipt`.
