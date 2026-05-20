# CUDA-MODEL-014 Qwen3 Exact-Profile Server Readiness Review

Date: 2026-05-19
Campaign item: CUDA-MODEL-014
Platform: Windows 9950X3D + RTX 5070 Ti
Model: qwen3-0.6b-instruct-q8_0
Coverage row: `dense_qwen3_06b_q8_candidate`

## Decision

Rejected for `server_ready=true`.

Supersession note: this report records the original CUDA-MODEL-014 decision.
CUDA-MODEL-014B later accepted exact-profile Qwen3 server readiness after
CUDA-MODEL-013 committed the missing current-source server-smoke receipt.

Qwen3 0.6B Q8_0 is product CLI-ready for bounded normal `ask` and `chat`
paths on the RTX 5070 Ti `dense_regular_llm_cuda` route, but the repository
does not yet contain a current-source Qwen3 server-smoke receipt for the exact
non-streaming `/v1/chat/completions` profile. Without that receipt, server
readiness cannot be promoted.

This review does not change runtime behavior. It records that the validation
path can recognize exact Qwen3 dense server-smoke receipts, but validation
support is not itself product evidence.

Post-review note: CUDA-MODEL-013 later committed the missing server-smoke
receipt at
`ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/server-strict-dense-qwen3-q8-smoke.json`.
This report remains a rejected readiness review. A new exact-profile readiness
review is still required before `server_ready=true` can be promoted.

## Evidence Reviewed

| Requirement | Evidence | Result |
| --- | --- | --- |
| Product CLI tier | `ci/model-artifacts/model-coverage-matrix.toml` row `dense_qwen3_06b_q8_candidate` is `product_cli_ready` | Passed |
| Server receipt validator support | `server_shared_engine_chat_completion` validation accepts exact Qwen3 dense CUDA identity | Passed as validator support only |
| Exact model artifact | Qwen3 model ID and SHA are known by the server receipt path | Passed as identity contract only |
| Exact endpoint profile receipt | No committed Qwen3 `/v1/chat/completions` non-streaming server-smoke JSON exists under `ci/hardware/windows-9950x3d-rtx5070ti/` | Failed |
| Runtime backend and route evidence | No committed Qwen3 server receipt currently proves `selected_backend=nvidia-rtx-5070-ti-cuda`, `runtime_api=cuda`, and `selected_route=dense_regular_llm_cuda` for the server endpoint | Failed |
| Quality gate evidence | No committed Qwen3 server receipt currently proves the server non-empty UTF-8 quality gate | Failed |
| Claim boundary | Coverage row keeps `server_ready=false`, `speedup_claim=false`, `full_residency_claim=false`, and `bitnet_packed_i2s_qk256_proof=false` | Passed |

## Required Receipt Before Promotion

The next review needs a committed current-source receipt with these fields:

```text
receipt_kind = server_shared_engine_chat_completion
model_coverage_row = dense_qwen3_06b_q8_candidate
model_coverage_tier = product_cli_ready
model_id = qwen3-0.6b-instruct-q8_0
endpoint_profile.endpoint = /v1/chat/completions
endpoint_profile.request_profile = non_streaming_chat_completion
endpoint_profile.streaming = false
requested_backend = nvidia-rtx-5070-ti-cuda
selected_backend = nvidia-rtx-5070-ti-cuda
runtime_api = cuda
selected_route = dense_regular_llm_cuda
fallback_used = false
quality_gate.passed = true
server_smoke_response_claimed = true
server_ready_claimed = false
speedup_claim = false
full_cuda_residency_claimed = false
dense_regular_llm_cuda_inference_claimed = true
bitnet_packed_i2s_qk256_proof = false
```

The runtime receipt should still use `server_ready_claimed=false`. As with the
Qwen2.5 exact-profile review, the model coverage matrix owns any product
promotion after the receipt is reviewed.

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

## Non-Claims

This review must not be cited as:

- Qwen3 server readiness;
- Qwen3 speedup;
- Qwen3 benchmark-qualified speed;
- Qwen3 full CUDA residency;
- broad dense GGUF server readiness;
- Qwen3 proof inherited from Qwen2.5;
- dense regular LLM CUDA proof as BitNet packed I2_S/QK256 proof.

## Next Proof

Run a follow-up exact-profile server-readiness review against the committed
non-streaming `/v1/chat/completions` shared-engine smoke receipt. Only that
review can decide whether to set `server_ready=true` for this Qwen3 profile.

## Validation

```powershell
rtk cargo run --locked -p xtask --no-default-features -- check-model-coverage
rtk git diff --check
```
