# CUDA-SERVER-006 BitNet QK256 Strict Server Smoke

Date: 2026-05-17
Campaign item: CUDA-SERVER-006
Platform: Windows 9950X3D + RTX 5070 Ti
Model: microsoft-bitnet-b1.58-2B-4T-i2s
Coverage row: `bitnet_official_2b_i2s_qk256`

## Summary

CUDA-SERVER-006 adds a strict RTX 5070 Ti shared-engine server smoke for the
official Microsoft BitNet I2_S/QK256 artifact. The receipt uses the separate
`bitnet_qk256_cuda` route, records QK256 CUDA invocation evidence, and preserves
zero BitNet linear CPU fallback.

This evidence is deliberately separate from the dense Qwen server-ready profile.
It does not create dense regular-LLM CUDA proof and does not promote speedup,
full CUDA residency, broad production server readiness, concurrency, deployment
readiness, or broad chat quality.

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-17/server-strict-bitnet-qk256-smoke.json
```

Key fields:

| Field | Value |
| --- | --- |
| `receipt_kind` | `server_shared_engine_chat_completion` |
| `model_sha256` | `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162` |
| `endpoint_profile.endpoint` | `/v1/chat/completions` |
| `endpoint_profile.request_profile` | `non_streaming_chat_completion` |
| `generation_policy.max_tokens` | `2` |
| `generation_policy.decoding` | `greedy` |
| `requested_backend` | `nvidia-rtx-5070-ti-cuda` |
| `selected_backend` | `nvidia-rtx-5070-ti-cuda` |
| `runtime_api` | `cuda` |
| `selected_route` | `bitnet_qk256_cuda` |
| `fallback_used` | `false` |
| `execution_plan.planner_version` | `cuda-planner-004` |
| `execution_plan.cuda_bitnet_qk256_ops` | `630` |
| `execution_plan.cuda_dense_regular_llm_ops` | `0` |
| `execution_plan.cpu_fallback_ops` | `0` |
| `execution_plan.unsupported_ops` | `0` |
| `execution_coverage.bitnet_linear_layers_total` | `630` |
| `execution_coverage.bitnet_linear_layers_on_cuda` | `630` |
| `execution_coverage.bitnet_linear_layers_cpu_fallback` | `0` |
| `kernel_stats[0].kernel_id` | `qk256_gemv_cuda` |
| `kernel_stats[0].invocations` | `630` |
| `kernel_stats[0].cpu_fallback_invocations` | `0` |
| `quality_gate.passed` | `true` |
| `server_smoke_response_claimed` | `true` |
| `server_ready_claimed` | `false` |
| `speedup_claim` | `false` |
| `full_cuda_residency_claimed` | `false` |
| `dense_regular_llm_cuda_inference_claimed` | `false` |
| `bitnet_packed_i2s_qk256_proof` | `true` |

The receipt's `server_ready_claimed=false` remains intentional. This slice adds
bounded exact-profile server-smoke evidence for the BitNet QK256 route; it does
not promote broad production server readiness.

## Coverage

The model coverage row now requires the `server_shared_engine_chat_completion`
receipt for the official BitNet I2_S/QK256 row and records this boundary:

```text
Official Microsoft 2B I2_S is the current BitNet QK256 x86/CUDA answer lane and
has strict RTX 5070 Ti server-smoke evidence for the bitnet_qk256_cuda route
only. It is not globally speed-qualified, does not prove dense regular-LLM CUDA,
and does not claim broad production server readiness.
```

## Non-Claims

This report does not claim:

- dense regular-LLM CUDA proof;
- dense Qwen proof as official BitNet QK256 proof;
- accepted CUDA speedup;
- full CUDA residency;
- global CUDA server readiness;
- concurrency, uptime, deployment hardening, or production service readiness;
- broad chat quality beyond the non-empty UTF-8 smoke gate.

## Commands Run

```powershell
rtk cargo test --locked -p bitnet-server --no-default-features --features cpu,cuda bitnet_qk256_server_smoke
rtk cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features bitnet_qk256
rtk python -m json.tool ci\hardware\windows-9950x3d-rtx5070ti\2026-05-17\server-strict-bitnet-qk256-smoke.json
rtk cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-17\server-strict-bitnet-qk256-smoke.json
```

The server test and live server launch required the Visual Studio developer
environment plus explicit CUDA, MSVC, and Windows SDK library paths on this
machine. An earlier link attempt failed before evidence generation because the
runtime path omitted required Windows/MSVC libraries; rerunning with the
corrected environment passed and produced the committed receipt.
