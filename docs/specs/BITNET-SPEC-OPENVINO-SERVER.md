# BITNET-SPEC-OPENVINO-SERVER: OpenVINO Exact-Profile Server Contract

Status: draft
Owner: intel/openvino
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md), [BITNET-SPEC-OPENVINO-DENSE-SLM](BITNET-SPEC-OPENVINO-DENSE-SLM.md), [BITNET-SPEC-OPENVINO-ROUTE-PROMOTION](BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md), [BITNET-SPEC-OPENVINO-RUST-BRIDGE](BITNET-SPEC-OPENVINO-RUST-BRIDGE.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no promotion; defines exact-profile server gates
Policy impact: no policy exception

## Purpose

Define when an OpenVINO Lunar Lake server endpoint can be described as ready for
an exact model, route, and workload profile. Server proof follows ask/chat route
readiness; it does not replace it.

This spec does not start a server, promote routes, claim broad server readiness,
claim streaming readiness, claim concurrency readiness, claim speedup, claim
power advantage, or prove BitNet QK256/I2_S behavior.

## Server Route Identity

Server receipts must use:

```text
route_id = openvino_server_exact_profile
runtime_api = openvino_model_server
```

and must also record the underlying execution route:

```json
{
  "server_route_id": "openvino_server_exact_profile",
  "underlying_route_id": "openvino_dense_slm_gpu_arc140v",
  "profile": "ask_short",
  "model_id": "qwen2_5_0_5b_instruct_openvino_int4_sym",
  "fallback_used": false
}
```

The server route is not a new model proof family. It wraps an already proven
ask/chat route for an exact profile.

## Prerequisites

Server proof requires:

1. The underlying route is promoted or explicitly approved for server testing on
   the exact profile.
2. The dense SLM model/export manifest is accepted.
3. The quality corpus passes for the exact profile.
4. Phase timing exists for the exact profile and cold/warm mode.
5. Route-promotion ledger and regression bundle are current.
6. Rust bridge stage is sufficient to produce equivalent server receipts.
7. Fallback is disabled and verified.

If the underlying route is only a candidate, the server can produce diagnostic
receipts but cannot claim readiness.

## Required Server Receipt

Minimum receipt shape:

```json
{
  "artifact_kind": "openvino_server_exact_profile",
  "server": {
    "host": "127.0.0.1",
    "port": 0,
    "endpoint": "/v1/chat/completions",
    "protocol": "http",
    "streaming": false,
    "concurrency": 1,
    "auth_mode": "local_only|none|configured"
  },
  "route": {
    "server_route_id": "openvino_server_exact_profile",
    "underlying_route_id": "openvino_dense_slm_gpu_arc140v",
    "requested_backend": "openvino-gpu",
    "selected_backend": "openvino-gpu",
    "runtime_api": "openvino_genai",
    "runtime_device": "GPU.0",
    "fallback_used": false
  },
  "model": {
    "model_id": "qwen2_5_0_5b_instruct_openvino_int4_sym",
    "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
    "export_format": "openvino_ir",
    "weight_format": "int4",
    "symmetric": true
  },
  "profile": "ask_short",
  "request": {
    "prompt_sha256": "<sha256>",
    "prompt_token_count": 64,
    "max_new_tokens": 32,
    "sampling": "greedy"
  },
  "response": {
    "http_status": 200,
    "answer_gate_passed": true,
    "stop_reason": "eos|stop_token|max_new_tokens|unknown",
    "generated_token_ids_source": "direct|retokenized|unavailable"
  },
  "phase_ms": {
    "server_start": null,
    "model_load_or_attach": null,
    "request_queue": null,
    "tokenize": null,
    "prefill": null,
    "first_token": null,
    "decode": null,
    "total_response": null
  },
  "does_not_prove": [
    "broad server readiness",
    "streaming readiness",
    "multi-client concurrency",
    "BitNet QK256/I2_S behavior"
  ]
}
```

Unknown fields must be explicit. Server receipts must not hide route fallback
behind HTTP success.

## Exact-Profile Scope

Server readiness is exact-profile only. A server receipt for `ask_short` does
not prove:

- `ask_normal`;
- `prefill_heavy`;
- `decode_heavy`;
- `structured`;
- `low_power`;
- warm/resident NPU service;
- streaming;
- concurrent users;
- BitNet service.

Each scope needs its own receipt and promotion review.

## Cold and Warm Server Timing

Server timing must say whether it includes:

- server process start;
- model load;
- OpenVINO compile;
- cache lookup;
- endpoint request handling;
- first token;
- decode;
- total response.

For NPU-backed server receipts, the cold/cache/warm/resident split from
`BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE` still applies.

## Streaming and Concurrency

Non-streaming, single-request proof is the default. Streaming or concurrency
requires separate receipts:

```text
streaming = true
concurrency = 2|4|8|...
```

Those receipts must record:

- per-client answer gates;
- per-client first-token and total latency;
- fallback status;
- queueing behavior;
- memory growth;
- timeout and cancellation behavior;
- route drift.

Until those receipts exist, server proof must state `streaming_ready=false` and
`concurrency_ready=false`.

## Security and Exposure

Server receipts must record exposure:

- bind address;
- port;
- auth mode;
- local-only or network exposure;
- whether model binaries are served from a committed or local-only path.

No PR may expose secrets, model binaries, or network service assumptions without
a separate policy decision.

## BitNet Server Boundary

Dense SLM server proof is not BitNet server proof. BitNet server readiness
requires:

- BitNet CPU reference or exact accelerator route proof;
- BitNet prompt/token/generated-token boundary;
- BitNet answer corpus for the server profile;
- QK256/I2_S claim fields when relevant;
- fallback rejection;
- exact server endpoint receipt.

OpenVINO dense SLM server success must not set `bitnet_qk256_proof=true`.

## Rejection Examples

| Evidence | Required decision |
| --- | --- |
| HTTP endpoint returns 200 but route fallback is unknown | Reject readiness |
| Server wraps GPU route that is only candidate | Diagnostic only |
| `ask_short` server proof exists | No `ask_normal` server claim |
| Non-streaming receipt exists | No streaming readiness claim |
| One-client receipt exists | No concurrency readiness claim |
| Dense Qwen server works | No BitNet server proof |

## Acceptance

This spec is complete when it defines:

1. Exact-profile server route identity and underlying-route linkage.
2. Prerequisites from ask/chat route readiness.
3. Required server receipt fields, timing phases, and fallback behavior.
4. Cold/warm, streaming, concurrency, security, and exposure boundaries.
5. Dense SLM versus BitNet server proof separation.
