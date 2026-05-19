# CUDA-MODEL-014B Qwen3 Exact-Profile Server Ready

Date: 2026-05-19
Campaign item: CUDA-MODEL-014B
Platform: Windows 9950X3D + RTX 5070 Ti
Model: qwen3-0.6b-instruct-q8_0
Coverage row: `dense_qwen3_06b_q8_candidate`

## Decision

Accepted for exact-profile `server_ready=true`.

CUDA-MODEL-014 originally rejected Qwen3 server readiness because no committed
current-source Qwen3 non-streaming `/v1/chat/completions` server-smoke receipt
existed at review time. CUDA-MODEL-013 later committed that missing receipt.
This follow-up review accepts only the exact bounded RTX 5070 Ti Qwen3 profile
proved by that receipt.

This review does not change runtime behavior. The runtime receipt still records
`server_ready_claimed=false`; as with Qwen2.5, the model coverage matrix owns the
product claim and scopes it to the exact profile below.

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/server-strict-dense-qwen3-q8-smoke.json
```

Key fields:

| Field | Value |
| --- | --- |
| `receipt_kind` | `server_shared_engine_chat_completion` |
| `model_coverage_row` | `dense_qwen3_06b_q8_candidate` |
| `model_coverage_tier` | `product_cli_ready` |
| `model_id` | `qwen3-0.6b-instruct-q8_0` |
| `model_sha256` | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` |
| `endpoint_profile.endpoint` | `/v1/chat/completions` |
| `endpoint_profile.request_profile` | `non_streaming_chat_completion` |
| `endpoint_profile.streaming` | `false` |
| `generation_policy.max_tokens` | `2` |
| `generation_policy.decoding` | `greedy` |
| `requested_backend` | `nvidia-rtx-5070-ti-cuda` |
| `selected_backend` | `nvidia-rtx-5070-ti-cuda` |
| `runtime_api` | `cuda` |
| `selected_route` | `dense_regular_llm_cuda` |
| `fallback_used` | `false` |
| `quality_gate.passed` | `true` |
| `server_smoke_response_claimed` | `true` |
| `server_ready_claimed` | `false` |
| `speedup_claim` | `false` |
| `full_cuda_residency_claimed` | `false` |
| `dense_regular_llm_cuda_inference_claimed` | `true` |
| `bitnet_packed_i2s_qk256_proof` | `false` |

## Promotion

The promoted model coverage state is:

```text
server_ready = true
speedup_claim = false
benchmark_qualified = false
full_residency_claim = false
bitnet_packed_i2s_qk256_proof = false
```

The promotion is scoped to:

- model: `qwen3-0.6b-instruct-q8_0`;
- artifact SHA-256:
  `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`;
- platform: Windows 9950X3D + RTX 5070 Ti;
- backend: `nvidia-rtx-5070-ti-cuda`;
- route: `dense_regular_llm_cuda`;
- endpoint: non-streaming `POST /v1/chat/completions`;
- generation policy: greedy, `max_tokens=2`, `temperature=0.0`, `top_p=1.0`.

## Non-Claims

This report does not claim:

- broad dense GGUF server readiness;
- Qwen2.5 proof inheritance;
- official BitNet I2_S/QK256 server readiness;
- dense Qwen proof as BitNet packed I2_S/QK256 proof;
- accepted CUDA speedup;
- benchmark-qualified speed;
- full CUDA residency;
- concurrency, uptime, deployment hardening, or production service readiness;
- broad chat quality beyond the non-empty UTF-8 smoke gate.

## Validation

```powershell
rtk cargo run --locked -p xtask --no-default-features -- check-model-coverage
rtk cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli model_status_dashboard_lists_qwen3_as_product_cli_ready
rtk cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli receipts_explain_links_qwen3_dense_receipt_to_product_cli_coverage
rtk git diff --check
```
