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
| M4-SERVE-002 | in progress | Add health and ready endpoint behavior. |
| M4-SERVE-003 | pending | Add streaming completion endpoint. |
| M4-SERVE-004 | pending | Add receipt export for server requests. |
| M4-SERVE-005 | pending | Integrate doctor/smoke/regression readiness flow. |

## Current Contract

`M4-SERVE-001` defines the intended `bitnet mac serve` command/config contract
in `docs/slm/apple-m4-local-server-command-config.md`. The contract keeps the
first slice docs-only: no endpoint implementation, OpenAI compatibility,
production-readiness, BitNet, Metal, QK256, Neural Engine, MPSGraph, or broad
performance claim is introduced.

After `M4-SERVE-001`, the next executable item is `M4-SERVE-002`: health and
ready endpoint behavior for model-cache, tokenizer, backend/fallback, disk/cache,
and unsupported-backend state without expensive generation by default.

`M4-SERVE-002` adds the initial `bitnet mac serve` endpoint slice: `/health`,
`/health/live`, `/ready`, and `/health/ready`. It remains intentionally short
of completions, OpenAI compatibility, receipt export, and production readiness.

## Claim Boundary

Local server work does not prove production readiness, full OpenAI
compatibility, BitNet quality, full Apple Metal inference, QK256, Neural Engine
execution, MPSGraph model inference, or broad M4 performance.
