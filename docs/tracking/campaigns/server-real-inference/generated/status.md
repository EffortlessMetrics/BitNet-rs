<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Server real inference Campaign Status

- Campaign: `server-real-inference`
- State: `active`
- Objective: Replace server-side simulated inference surfaces with real engine execution or explicit unavailable responses without weakening strict proof boundaries.

## Work Items

| Item | State | PR | Branch | Acceptance |
|---|---|---:|---|---|
| SERVER-001 | proposed | TBD | `codex/server-real-inference/SERVER-001-single-request-engine` | Wire single-request server inference to real engine execution or explicit 501/503, with no simulated response path in non-test builds. |

## Hard Constraints

- Do not reintroduce simulated inference.
- Do not bypass strict model loading, tokenizer authority, or fallback receipts.
- Do not mix server runtime work with hardware kernel claims.
