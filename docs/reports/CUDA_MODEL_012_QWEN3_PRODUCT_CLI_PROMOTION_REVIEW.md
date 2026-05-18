# CUDA-MODEL-012 Qwen3 Product CLI Promotion Review

Date: 2026-05-18
Campaign item: CUDA-MODEL-012
Platform: Windows 9950X3D + RTX 5070 Ti
Model: qwen3-0.6b-instruct-q8_0

## Decision

Accepted for `product_cli_ready`.

Qwen3 0.6B Q8_0 has now satisfied the bounded product CLI ladder for the
RTX 5070 Ti dense SLM route. The model coverage row is promoted from
`accelerator_answer_ready` to `product_cli_ready` because committed receipts now
cover the normal `bitnet ask` and `bitnet chat` user paths in addition to the
earlier artifact, CPU, all-layer, one-token, short-decode, warm-session, and
benchmark-review evidence.

This is not a server, speed, benchmark-qualified, full-residency, broad dense
GGUF, Qwen2.5 inheritance, or BitNet QK256 promotion.

## Evidence Reviewed

| Requirement | Evidence | Result |
| --- | --- | --- |
| Artifact contract | `ci/model-artifacts/model-coverage-matrix.toml` row `dense_qwen3_06b_q8_candidate` with contract `qwen3_0_6b_q8_0` | Passed |
| Tokenizer and prompt authority | Coverage row records `gguf_metadata_strict` and `qwen_no_think_chatml_cpu_sanity` | Passed |
| CPU answer sanity | `qwen3-0_6b-cpu-answer-corpus.json` in the Qwen3 ladder | Passed |
| All-layer route plan | `qwen3-0_6b-cuda-all-layer-plan.json` | Passed |
| Model-boundary fixtures | `qwen3-0_6b-model-boundary-fixtures.json` | Passed |
| KV and sampling policy | `qwen3-0_6b-kv-cache-policy.json` and `qwen3-0_6b-sampling-policy.json` | Passed |
| One-token strict CUDA | `qwen3-0_6b-one-token-cuda.json` | Passed |
| Short-decode strict CUDA | `qwen3-0_6b-short-decode-cuda.json` | Passed |
| Warm-session strict CUDA | `qwen3-0_6b-warm-session-cuda.json` | Passed |
| Benchmark review | `qwen3-0_6b-benchmark-qualification.json` rejects speedup for reviewed profiles | Passed as review, not speed |
| Normal ask user path | `qwen3-0_6b-ask-user-path-cuda.json` | Passed |
| Normal chat user path | `qwen3-0_6b-chat-user-path-cuda.json` | Passed |
| Fallback rejection | Ask and chat receipts record `fallback_used=false` | Passed |
| Route identity | Ask and chat receipts record `selected_backend=nvidia-rtx-5070-ti-cuda` and `selected_route=dense_regular_llm_cuda` | Passed |
| Receipt explain support | Ask and chat receipts explain against `dense_qwen3_06b_q8_candidate` | Passed |

## Promoted Row

```text
model_coverage_row = dense_qwen3_06b_q8_candidate
current_tier = product_cli_ready
product_cli_ready = true
selected_backend = nvidia-rtx-5070-ti-cuda
selected_route = dense_regular_llm_cuda
fallback_used = false
dense_regular_llm_cuda_proof = true
```

## Claims Still False

```text
server_ready = false
speedup_claim = false
benchmark_qualified = false
full_residency_claim = false
bitnet_packed_i2s_qk256_proof = false
qwen2_5_proof_inheritance = false
broad_dense_gguf_readiness = false
```

## Claim Boundary

This review may claim:

- Qwen3 0.6B Q8_0 is product CLI-ready for bounded RTX 5070 Ti CUDA
  `ask` and `chat` paths;
- the route is Qwen3-specific dense SLM CUDA proof on
  `dense_regular_llm_cuda`;
- fallback was rejected in the committed user-path receipts;
- benchmark review happened and did not accept speedup;
- server readiness, speedup, benchmark-qualified speed, and full residency
  remain future work.

It must not claim:

- Qwen3 server readiness;
- Qwen3 speedup;
- Qwen3 benchmark-qualified speed;
- Qwen3 full CUDA residency;
- broad dense GGUF CUDA readiness;
- Qwen3 proof inherited from Qwen2.5;
- dense regular LLM CUDA proof as BitNet packed I2_S/QK256 proof.

## Next Proof

The next Qwen3 proof should be one of these separate reviews:

- exact-profile server smoke or server readiness;
- repeated same-artifact CPU/CUDA performance comparator evidence;
- residency and transfer audit evidence.

No one of those reviews should promote the other claims by implication.

## Validation

```powershell
rtk cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli model_status_dashboard_lists_qwen3_as_product_cli_ready
rtk cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli receipts_explain_links_qwen3_dense_receipt_to_product_cli_coverage
rtk cargo run --locked -p xtask --no-default-features -- check-model-coverage
rtk git diff --check
```
