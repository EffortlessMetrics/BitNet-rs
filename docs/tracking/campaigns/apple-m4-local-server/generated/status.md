<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Apple M4 local server Campaign Status

- Campaign: `apple-m4-local-server`
- State: `active`
- Objective: Expose the working Apple M4 dense SLM appliance as a local service with command/config, health and ready endpoints, streaming completions, receipt export, model-cache verification, and no hidden fallback, while reusing the same Mac model/cache/tokenizer/backend discipline.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| M4-SERVE-001 | merged | #4359 | `codex/apple-m4-local-server/M4-SERVE-001-command-config` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Define the Apple M4 local server command and configuration contract, including model-id, cache-dir, device, host/port, streaming defaults, receipt path, cache verification, and unsupported-backend failure behavior without implementing serving endpoints. |
| M4-SERVE-002 | merged | #4363 | `codex/apple-m4-local-server/M4-SERVE-002-health-ready` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add health and ready endpoint behavior for model-cache, tokenizer, backend/fallback, disk/cache, and unsupported-backend state, with tests and no expensive generation by default. |
| M4-SERVE-003 | ready | TBD | `codex/apple-m4-local-server/M4-SERVE-003-streaming-completions` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add a streaming local completion endpoint for the supported dense SLM path, preserving model/tokenizer authority, resident session reuse, generated text and token IDs, backend identity, fallback_used=false, and timing receipts. |
| M4-SERVE-004 | blocked | TBD | `codex/apple-m4-local-server/M4-SERVE-004-receipt-export` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add receipt export for local server requests, including model, tokenizer, backend, fallback, generated text, token IDs, timing, request metadata, streaming status, and claim boundaries. |
| M4-SERVE-005 | blocked | TBD | `codex/apple-m4-local-server/M4-SERVE-005-doctor-smoke` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Integrate local server health with Mac doctor/smoke/regression workflows, documenting how operators verify server readiness and receipt export without making production uptime claims. |

## Hard Constraints

- This is an M4 Mac mini dense SLM service campaign.
- Do not claim production server readiness until endpoints and receipts are implemented.
- Do not claim full OpenAI API compatibility until request/response semantics are tested.
- Do not claim BitNet local-answer quality, full apple-m4-metal inference, Neural Engine execution, MPSGraph model inference, QK256 support, or broad M4 performance.
- Do not execute MacBook, x86, CUDA, A770, Lunar Lake, or NPU work.
- Never commit model binaries.
