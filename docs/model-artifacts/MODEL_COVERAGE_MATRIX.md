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
`speedup_claim=false`, `full_residency_claim=false`, and `server_ready=false`.

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
| `bitnet_official_2b_i2s_qk256` | GGUF I2_S / QK256 | `product_cli_ready` | Current official x86/CUDA answer lane, not globally speed-qualified. |
| `bitnet_official_2b_tl1_arm_candidate` | GGUF TL1 | `registered` | ARM-oriented candidate; needs TL1 layout, scalar, NEON/Apple proofs. |
| `bitnet_official_2b_tl2_x86_candidate` | GGUF TL2 | `registered` | x86 LUT candidate; needs TL2 runner and scalar/AVX proofs. |
| `bitnet_official_2b_bf16_gpu_int2_candidate` | BF16 master to GPU packed int2/W2A8 | `registered` | Separate GPU-reference path; does not satisfy GGUF I2_S proof. |
| `bitnet_3b_x86_i2s_unsupported` | 3B GGUF I2_S on x86 | `registered` | Upstream-unsupported; diagnostic/unsupported-path receipts only. |
| `bitnet_3b_x86_tl2_candidate` | 3B GGUF TL2 on x86 | `registered` | Listed candidate; needs runner verification before answer claims. |
| `bitnet_onebit_large_diagnostic` | 1bitLLM large-family artifact | `registered` | Diagnostic until family-specific artifact, tokenizer, prompt, and route contracts land. |
| `bitnet_llama3_8b_158_diagnostic` | Llama3-family 1.58-bit variant | `registered` | Diagnostic contract, not official BitNet answer authority. |
| `bitnet_falcon3_falcon_e_158_diagnostic` | Falcon-family 1.58-bit variant | `registered` | Diagnostic contract, not official BitNet answer authority. |
| `bitnet_mcu_tiny_fixture` | MCU low-bit fixture | `registered` | Arithmetic/kernel regression testbed only, not LLM answer authority. |

## Validation

Run:

```powershell
cargo run --release --locked -p xtask --no-default-features -- check-model-coverage
```

The validator parses the matrix, checks tier ordering, requires core lane
coverage, and rejects common claim leaks such as dense entries claiming BitNet
packed proof, TL1/TL2 rows claiming I2_S/QK256 proof, or unsupported entries
claiming answer readiness.
