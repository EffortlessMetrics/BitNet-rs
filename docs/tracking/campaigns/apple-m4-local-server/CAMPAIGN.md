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
| M4-SERVE-003 | ready | Add streaming completion endpoint. |
| M4-SERVE-004 | pending | Add receipt export for server requests. |
| M4-SERVE-005 | pending | Integrate doctor/smoke/regression readiness flow. |

## Current Contract

`M4-SERVE-001` defines the intended `bitnet mac serve` command/config contract
in `docs/slm/apple-m4-local-server-command-config.md`.

`M4-SERVE-002` added the initial `bitnet mac serve` endpoint slice: `/health`,
`/health/live`, `/ready`, and `/health/ready`, with startup cache verification,
strict `apple-m4-cpu-neon` routing, explicit no-hidden-fallback state, and no
generation by default. It remains intentionally short of completions, OpenAI
compatibility, receipt export, and production readiness.

The next executable item is `M4-SERVE-003`: streaming local dense SLM completion
behavior through the same supported model-cache, tokenizer, backend, fallback,
resident-session, and receipt discipline.

## Claim Boundary

Local server work does not prove production readiness, full OpenAI
compatibility, BitNet quality, full Apple Metal inference, QK256, Neural Engine
execution, MPSGraph model inference, or broad M4 performance.
