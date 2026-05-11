<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Server real inference Campaign Status

- Campaign: `server-real-inference`
- State: `active`
- Objective: Replace server-side simulated inference surfaces with real engine execution or explicit unavailable responses without weakening strict proof boundaries.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| SERVER-001 | merged | #4429 | `codex/server-real-inference/SERVER-001-single-request-engine` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Wire single-request server inference to real engine execution or explicit 501/503, with no simulated response path in non-test builds. |
| SERVER-002 | merged | #4432 | `codex/server-real-inference/SERVER-002-no-placeholder-model-readiness` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Fail closed server model lifecycle scaffolds that previously reported placeholder HuggingFace/cache model readiness without real I/O or a real inference engine. |
| SERVER-003 | merged | #4477 | `codex/server-003-readiness-certification` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add a server readiness/certification endpoint that exposes active model, backend, inference, fallback, and claim-boundary state while failing closed until a real server inference engine is wired. |
| SERVER-004 | pr_open | #4479 | `codex/server-004-openai-chat-fail-closed` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add an OpenAI-compatible /v1/chat/completions endpoint that fails closed with readiness/certification details until a real server inference engine is wired. |

## Hard Constraints

- Do not reintroduce simulated inference.
- Do not bypass strict model loading, tokenizer authority, or fallback receipts.
- Do not mix server runtime work with hardware kernel claims.
