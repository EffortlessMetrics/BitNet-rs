# CUDA-SERVER-003 Dense Qwen Server Readiness Audit

Date: 2026-05-16
Campaign item: CUDA-SERVER-003
Platform: Windows 9950X3D + RTX 5070 Ti
Model: qwen2.5-0.5b-instruct-q8_0
Coverage row: `dense_qwen25_05b_q8_cuda`

## Summary

The committed dense Qwen2.5 server-smoke receipt is valid bounded evidence, but
it is not sufficient to promote `server_ready=true` under
[BITNET-SPEC-0010](../specs/BITNET-SPEC-0010-server-readiness-proof-boundary.md).

The receipt proves a strict RTX 5070 Ti CUDA server smoke with the dense route,
fallback disabled, a non-empty UTF-8 response, and no speed, residency, or
BitNet proof claim. It does not carry enough exact-profile identity for server
readiness promotion: the receipt is missing durable artifact checksum identity,
endpoint or request-profile scope, and generation-policy fields. It also
correctly records `server_ready_claimed=false`.

Therefore the correct model coverage state remains:

```text
server_ready = false
speedup_claim = false
full_residency_claim = false
bitnet_packed_i2s_qk256_proof = false
```

## Receipt Audited

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/server-strict-dense-qwen25-q8-smoke.json
```

## Field Audit

| BITNET-SPEC-0010 field | Receipt state | Promotion impact |
| --- | --- | --- |
| model coverage row | `dense_qwen25_05b_q8_cuda` | present |
| model identity | `requested_model = qwen2.5-0.5b-instruct-q8_0` | partial |
| artifact checksum | missing `sha256`, `model_sha256`, or `checksum` field | blocks exact-profile promotion |
| server endpoint/profile | no endpoint, request profile, or endpoint profile field | blocks exact-profile promotion |
| generation policy | no max tokens, temperature, top-p, seed, or policy object | blocks exact-profile promotion |
| requested backend | `nvidia-rtx-5070-ti-cuda` | present |
| selected backend | `nvidia-rtx-5070-ti-cuda` | present |
| runtime API | `cuda` | present |
| route | `dense_regular_llm_cuda` | present as `selected_route` |
| fallback status | `fallback_used = false` | present |
| tokenizer authority | `active_model_tokenizer` | present |
| prompt authority | `server_chat_template` | present |
| quality gate | `server_non_empty_utf8_response`, passed | present |
| speedup claim | `speedup_claim = false` | correct |
| full residency claim | `full_cuda_residency_claimed = false` | correct |
| dense proof | `dense_regular_llm_cuda_inference_claimed = true` | bounded by missing checksum |
| BitNet proof | `bitnet_packed_i2s_qk256_proof = false` | correct |
| server readiness | `server_ready_claimed = false` | correct for smoke, not promotion |

## Decision

Do not promote dense Qwen2.5 server readiness from the current smoke receipt.

The current receipt may support:

- bounded strict CUDA server-smoke evidence for Qwen2.5 0.5B Q8_0;
- selected backend `nvidia-rtx-5070-ti-cuda`;
- route `dense_regular_llm_cuda`;
- fallback-free server response smoke;
- non-empty UTF-8 response quality gate;
- explicit non-claims for speed, full residency, and BitNet QK256 proof.

The current receipt must not support:

- `server_ready=true`;
- broad dense GGUF server readiness;
- official BitNet server readiness;
- speedup;
- full CUDA residency;
- production service readiness;
- concurrency, uptime, or deployment hardening.

## Next Required Proof

A future promotion PR needs either a refreshed receipt or an additional
promotion receipt/report that includes:

- stable model id and artifact checksum;
- endpoint or internal server request profile;
- generation policy, including token limit and sampling settings;
- exact scope for the server-ready profile;
- `server_ready_claimed=true` only for that exact profile;
- unchanged `speedup_claim=false` unless separately benchmark-qualified;
- unchanged `full_residency_claim=false` unless separately proven;
- unchanged `bitnet_packed_i2s_qk256_proof=false`.

If those fields land, the promotion PR should update the model coverage row,
the human-readable model coverage page, the CUDA capability matrix, and the
model-status fixtures together.

## Commands Run

```powershell
rtk python -m json.tool ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\server-strict-dense-qwen25-q8-smoke.json
rtk powershell -NoProfile -Command '$j = Get-Content ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\server-strict-dense-qwen25-q8-smoke.json -Raw | ConvertFrom-Json; $j.PSObject.Properties.Name | Sort-Object'
```

Both commands completed successfully during the audit. The second command
showed that checksum, endpoint/profile, and generation-policy fields are absent
from the receipt.

## Claim Boundary

This audit records a blocker. It does not change runtime behavior, server
behavior, receipts, model artifacts, model coverage booleans, CUDA kernels,
speed claims, server claims, or BitNet proof claims.
