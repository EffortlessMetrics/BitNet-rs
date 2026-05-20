# CUDA-SERVER-005 Dense Qwen Exact-Profile Server Readiness

Date: 2026-05-17
Campaign item: CUDA-SERVER-005
Platform: Windows 9950X3D + RTX 5070 Ti
Model: qwen2.5-0.5b-instruct-q8_0
Coverage row: `dense_qwen25_05b_q8_cuda`

## Summary

CUDA-SERVER-005 refreshes the dense Qwen shared-engine server smoke from the
hardened receipt path added by CUDA-SERVER-004. The refreshed receipt includes
the exact-profile fields required by
[BITNET-SPEC-0010](../specs/BITNET-SPEC-0010-server-readiness-proof-boundary.md):
artifact checksum identity, endpoint/request profile, generation policy, strict
backend, dense route, quality gate, and explicit claim booleans.

This promotes `server_ready=true` only for the exact non-streaming RTX 5070 Ti
`/v1/chat/completions` shared-engine profile for Qwen2.5 0.5B Q8_0. It does not
promote broad dense GGUF server readiness, official BitNet server readiness,
speedup, full CUDA residency, concurrency, deployment readiness, or broad chat
quality.

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-17/server-strict-dense-qwen25-q8-smoke.json
```

Key fields:

| Field | Value |
| --- | --- |
| `receipt_kind` | `server_shared_engine_chat_completion` |
| `model_sha256` | `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e` |
| `endpoint_profile.endpoint` | `/v1/chat/completions` |
| `endpoint_profile.request_profile` | `non_streaming_chat_completion` |
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

`server_ready_claimed=false` remains correct in the runtime receipt because the
receipt records what the server emitted at request time. The model coverage
matrix owns the product claim and promotes `server_ready=true` for this exact
profile after the promotion review checks the receipt.

## Promotion

The promoted model coverage state is:

```text
server_ready = true
speedup_claim = false
full_residency_claim = false
bitnet_packed_i2s_qk256_proof = false
```

The promotion is scoped to:

- model: `qwen2.5-0.5b-instruct-q8_0`;
- artifact SHA-256:
  `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`;
- platform: Windows 9950X3D + RTX 5070 Ti;
- backend: `nvidia-rtx-5070-ti-cuda`;
- route: `dense_regular_llm_cuda`;
- endpoint: non-streaming `POST /v1/chat/completions`;
- generation policy: greedy, `max_tokens=2`, `temperature=0.0`, `top_p=1.0`.

## Non-Claims

This report does not claim:

- global dense GGUF server readiness;
- official BitNet I2_S/QK256 server readiness;
- dense Qwen proof as BitNet packed I2_S/QK256 proof;
- accepted CUDA speedup;
- full CUDA residency;
- concurrency, uptime, deployment hardening, or production service readiness;
- broad chat quality beyond the non-empty UTF-8 smoke gate.

## Commands Run

```powershell
rtk cargo build --locked -p bitnet-server --no-default-features --features cpu,cuda
rtk python -m json.tool ci\hardware\windows-9950x3d-rtx5070ti\2026-05-17\server-strict-dense-qwen25-q8-smoke.json
rtk cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features server_shared_engine_chat_completion
```

The CUDA build required the Visual Studio developer environment plus explicit
CUDA and Windows SDK library paths on this machine. The first attempt without
those paths failed before evidence generation because `nvcc` could not find
`cl.exe`, then `link.exe` could not find CUDA/Windows SDK libraries. After the
environment was corrected, the build passed and the server generated the
receipt above from the current source tree.
