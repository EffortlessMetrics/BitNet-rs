# Apple M4 Local Server

Campaign ID: `apple-m4-local-server`

Status: active

## Objective

Turn the M4 dense SLM CLI appliance into a local service surface that preserves
the same cache, tokenizer, backend, fallback, streaming, and receipt discipline.

## End State

- Server command/config is explicit.
- Health and ready endpoints expose cache/model/backend state.
- Streaming completions use the supported dense SLM path.
- Server requests can export strict receipts.
- Doctor/smoke/regression flows can verify server readiness.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| M4-SERVE-001 | merged | Defined command and configuration contract. |
| M4-SERVE-002 | merged | Added health and ready endpoint behavior. |
| M4-SERVE-003 | merged | Added streaming completion endpoint. |
| M4-SERVE-004 | merged | Added receipt export for server requests. |
| M4-SERVE-005 | in progress | Integrate doctor/smoke/regression readiness flow. |

## Current Contract

`M4-SERVE-001` defines the intended `bitnet mac serve` command/config contract
in `docs/slm/apple-m4-local-server-command-config.md`.

`M4-SERVE-002` added the initial `bitnet mac serve` endpoint slice: `/health`,
`/health/live`, `/ready`, and `/health/ready`, with startup cache verification,
strict `apple-m4-cpu-neon` routing, explicit no-hidden-fallback state, and no
generation by default. It remains intentionally short of completions, OpenAI
compatibility, receipt export, and production readiness.

`M4-SERVE-003` adds the first streaming local dense SLM completion endpoint:
`POST /v1/chat/completions`, using the verified supported model cache,
`apple-m4-cpu-neon`, strict no-hidden-fallback routing, resident startup
model/tokenizer load, per-request receipts, and no full OpenAI compatibility
claim.

`M4-SERVE-004` adds HTTP export for those per-request receipts through
`GET /receipts/{id}`. The endpoint is read-only, rejects unsafe receipt IDs, and
does not run generation.

`M4-SERVE-005` adds the operator check path for a running local server:
`bitnet mac serve-check` validates readiness without generation by default, and
can optionally run a tiny completion plus receipt export probe.

## Claim Boundary

Local server work does not prove production readiness, full OpenAI
compatibility, BitNet quality, full Apple Metal inference, QK256, Neural Engine
execution, MPSGraph model inference, or broad M4 performance.
