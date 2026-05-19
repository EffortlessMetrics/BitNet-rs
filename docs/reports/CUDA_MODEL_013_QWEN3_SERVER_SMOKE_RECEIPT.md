# CUDA-MODEL-013 Qwen3 Server-Smoke Receipt

Date: 2026-05-19
Campaign item: CUDA-MODEL-013
Platform: Windows 9950X3D + RTX 5070 Ti
Model: qwen3-0.6b-instruct-q8_0
Coverage row: `dense_qwen3_06b_q8_candidate`

## Summary

CUDA-MODEL-013 records a current-source Qwen3 non-streaming
`/v1/chat/completions` server-smoke receipt from the shared local inference
engine. The receipt proves a bounded server response for the exact Qwen3 0.6B
Q8_0 artifact on the strict RTX 5070 Ti dense CUDA route.

This is server-smoke evidence only. It does not promote Qwen3 server readiness,
speedup, benchmark-qualified speed, full CUDA residency, broad dense GGUF
readiness, Qwen2.5 proof inheritance, or BitNet packed I2_S/QK256 proof.

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-19/server-strict-dense-qwen3-q8-smoke.json
```

Key fields:

| Field | Value |
| --- | --- |
| `receipt_kind` | `server_shared_engine_chat_completion` |
| `model_identity.model_id` | `qwen3-0.6b-instruct-q8_0` |
| `model_sha256` | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` |
| `model_coverage_row` | `dense_qwen3_06b_q8_candidate` |
| `model_coverage_tier` | `product_cli_ready` |
| `endpoint_profile.endpoint` | `/v1/chat/completions` |
| `endpoint_profile.request_profile` | `non_streaming_chat_completion` |
| `endpoint_profile.streaming` | `false` |
| `generation_policy.max_tokens` | `2` |
| `generation_policy.temperature` | `0.0` |
| `generation_policy.top_p` | `1.0` |
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

`server_ready_claimed=false` is intentional. The runtime receipt records the
bounded server-smoke response. A separate exact-profile readiness review must
decide whether the model coverage row can promote `server_ready=true` for this
specific Qwen3 profile.

## Non-Claims

This report does not claim:

- exact-profile Qwen3 server readiness;
- broad dense GGUF server readiness;
- Qwen2.5 proof inheritance;
- official BitNet I2_S/QK256 proof;
- accepted CUDA speedup;
- benchmark-qualified speed;
- full CUDA residency;
- concurrency, uptime, deployment hardening, or production service readiness;
- broad chat quality beyond the non-empty UTF-8 smoke gate.

## Commands Run

```powershell
rtk cmd /v:on /c 'C:\PROGRA~1\MICROS~4\2022\COMMUN~1\VC\AUXILI~1\Build\vcvars64.bat && set RUSTFLAGS=-L native=C:\PROGRA~1\NVIDIA~2\CUDA\v12.9\lib\x64 && rtk cargo build --release --locked -p bitnet-server --no-default-features --features cpu,cuda'

.\target\release\server.exe `
  --host 127.0.0.1 `
  --port 18094 `
  --model C:\bntmp\cuda-model-010\Qwen3-0.6B-Q8_0.gguf `
  --device nvidia-rtx-5070-ti-cuda `
  --log-format compact

Invoke-RestMethod -Uri http://127.0.0.1:18094/readiness

Invoke-RestMethod `
  -Uri http://127.0.0.1:18094/v1/chat/completions `
  -Method Post `
  -ContentType application/json `
  -Body '{"model":"qwen3-0.6b-instruct-q8_0","messages":[{"role":"user","content":"Say OK."}],"max_tokens":2,"temperature":0.0,"top_p":1.0,"stream":false}'

Invoke-RestMethod -Uri http://127.0.0.1:18094/receipts/latest

rtk python -m json.tool ci\hardware\windows-9950x3d-rtx5070ti\2026-05-19\server-strict-dense-qwen3-q8-smoke.json
```

The CUDA server build required the Visual Studio developer environment plus an
explicit CUDA 12.9 library search path via `RUSTFLAGS=-L native=...`. A debug
server startup attempt was abandoned after slow model initialization; the
committed receipt was generated from the release server binary.
