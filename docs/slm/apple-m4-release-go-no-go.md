# Apple M4 Release Go/No-Go Matrix

`M4-RELEASE-001` defines the decision surface for changing the public M4 Mac
mini expectation envelope or marking a dense SLM or BitNet route excellent.
This matrix does not run a model, publish a new receipt, enable a route, or
create a runtime claim. It says which already-defined gates must be green before
a maintainer can make a release-envelope claim.

## Decision Rule

A release-envelope update is `go` only when every required row for the target
route, model family, model identity, and command surface is satisfied by
matching committed receipts or a cited release-gate hardware bundle. Any missing
receipt, stale identity, mismatched model/tokenizer/backend/template, failed
mechanical score, timeout failure, unsupported route state, failed claim lint,
or missing generated dashboard update is `no-go`.

The decision must record:

- target route and model family;
- exact model artifact, tokenizer authority, prompt template, backend, fallback
  state, corpus or profile id, and evidence date;
- report-refresh and regression-dashboard output paths;
- changed public wording or operator envelope row;
- release-gate workflow run or committed receipt bundle;
- claim-boundary result.

BitNet chat and BitNet serve are route-specific decisions. Passing BitNet ask,
warm-session, eval, or benchmark gates is not enough to mark chat or serve
excellent.

## Required Gates

| Gate | Required evidence | Passing condition | No-go triggers |
|---|---|---|---|
| Dense SLM accuracy | `M4-ACCURACY-007`, `M4-CANARY-001`, `M4-ROBUSTNESS-001`, `M4-DENSE-REF-001` | Supported dense identities have matching 500-case eval history, canary receipts, robustness results, and reference-vs-Rust comparison context for the exact artifact/tokenizer/template/backend identity. | Insufficient history, task-family regression, failed canary, unsupported robustness drift, reference comparison mismatch, missing token IDs where required, or identity mismatch. |
| Dense SLM benchmark | `M4-BENCH-003`, `M4-BENCH-007`, `M4-CONTEXT-002` | Benchmark summaries have calibrated harness metadata, matching-identity regression comparison, timing and memory drift fields, and context profile evidence for the route being claimed. | Uncalibrated harness, invalid comparison, timed-out required profile, missing cold/tokenizer/prefill/TTFT/decode/memory fields, stale profile set, or unsupported context request. |
| BitNet accuracy | `M4-BITNET-EX-015`, `M4-CANARY-001`, `M4-ROBUSTNESS-001` | Accepted Microsoft I2_S artifact and external tokenizer have repaired 250-case matching history, task-family trend context, canary evidence, robustness separation, fallback=false backend fields, and exact artifact/tokenizer/template identity. | Dense evidence used as BitNet evidence, insufficient BitNet history, failed repaired-corpus comparison, tokenizer mismatch, fallback ambiguity, timeout failure, missing generated token IDs, or broad quality wording. |
| BitNet benchmark and warm behavior | `M4-BITNET-REG-001`, `M4-BITNET-EX-005`, `M4-BITNET-EX-015` | Warm-session and benchmark receipts compare against matching accepted artifact/tokenizer/backend/template identity and preserve progress, timeout, partial-failure, aggregate timing, and memory fields. | Prompt-set mismatch, timeout-stage regression, missing progress/failure receipt, missing aggregate timing or memory, unmatched baseline, or chat/serve inference from warm evidence. |
| Dense service | `M4-SERVE-EX-001`, `M4-SERVE-EX-002`, `M4-SERVE-EX-003`, `M4-SERVE-EX-004`, `M4-ROUTE-MATRIX-001` | Dense server health, ready, one-shot, streaming, timeout, cancellation, safety-default, queue, backpressure, resident-state, and per-request receipts pass for the exact enabled local route. | Non-localhost default, missing ready or per-request receipt, streaming/failure mismatch, queue/backpressure failure, cache-path disclosure, production-hosting wording, or OpenAI-compatibility overclaim. |
| BitNet chat | `M4-BITNET-EX-006`, `M4-ROUTE-MATRIX-001`, `M4-OBS-001`, `M4-RELIABILITY-001` | Route-state matrix marks BitNet chat enabled and chat receipts cover variable prompts, repeated prompts, timeout/cancel, per-turn receipts, trace correlation, and failure receipts. | Current `disabled_without_ready_gate` state, missing chat gate receipt, no variable-prompt stability, missing per-turn receipt, timeout/cancel gap, or serve inference from chat evidence. |
| BitNet serve | `M4-BITNET-EX-007`, `M4-SERVE-EX-002`, `M4-SERVE-EX-004`, `M4-ROUTE-MATRIX-001`, `M4-OBS-001`, `M4-RELIABILITY-001` | Route-state matrix marks BitNet serve enabled and serve receipts cover health, ready, completion, streaming, timeout, cancellation, queue/backpressure, per-request receipts, trace correlation, and failure receipts. | Current `disabled_without_ready_gate` state, missing serve gate receipt, missing per-request receipt, streaming/failure mismatch, queue/backpressure gap, or production-hosting wording. |
| Setup, compatibility, and cache repair | `M4-SETUP-001`, `M4-COMPAT-001`, `M4-STABILITY-002`, `M4-MODEL-LIFECYCLE-001` | First-run, compatibility-refresh, model lifecycle, cache verification, disk-pressure, stale-symlink, bad-SHA, missing-tokenizer, and rollback guidance are current for the release target. | OS/toolchain/binary/manifest drift without compatibility refresh, low disk without repair guidance, stale cache, missing tokenizer, bad SHA, lifecycle mismatch, or unsupported default change. |
| Stability, trend, and retention | `M4-STABILITY-001`, `M4-STABILITY-003`, `M4-TREND-001`, `M4-EVIDENCE-REPLAY-001` | Mixed-model soak, trend-retention policy, seven-day trend history, skipped-day reasons, replay bundles, and dashboard status are current for the target route and family. | Missing retained receipt, stale identity, skipped day without reason, memory drift beyond threshold, replay bundle mismatch, or dashboard status that does not match the proposed envelope. |
| Operator UX and route state | `M4-OPS-SLO-001`, `M4-OPS-UX-003`, `M4-WORKLOAD-001`, `M4-ROUTE-MATRIX-001`, `M4-CONTEXT-001` | Operator envelope class, doctor/status readiness, workload suite, route matrix, context guardrails, unsupported routes, and recommended commands agree with the proposed public wording. | Route-state contradiction, context beyond recorded envelope, disabled route shown as ready, operator class mismatch, missing cache/disk warning, or no next command. |
| Observability and recovery | `M4-OBS-001`, `M4-RELIABILITY-001` | Trace IDs, progress/log correlation, cancellation, timeout, interrupted receipt write, missing/corrupt cache, low-disk, process restart, and retry guidance are present for the target command family. | Prompt or cache leakage beyond redaction policy, missing failure receipt, missing timeout/cancel stage, no retry guidance, or uncorrelated per-request/per-turn receipts. |
| CI and release lane | `M4-CI-001`, `M4-GATE-HYGIENE-001` | Generic PR checks stay model-free, release-gate hardware refresh uses explicit `run_class=release_gate`, and gate hygiene stays clean or intentionally documented. | Live model work in generic required PR CI, missing release-gate run when public envelope changes, stale generated dashboards, unaddressed gate hygiene failure, or missing retention. |
| Claim boundary | `M4-CLAIM-LINT-001` | `claim-lint --scope apple-m4 --check` passes for docs, generated tracker status, operator envelope text, and operator-facing command strings. | Unsupported Apple Silicon, MacBook, full Metal, Neural Engine, MPSGraph, QK256, dense-as-BitNet, broad quality/performance, or speedup wording without a matching accepted receipt gate. |

## Route Verdicts

| Target claim | Minimum verdict | Required extras | Still no-go when |
|---|---|---|---|
| Dense SLM `ask` excellent for one supported model identity | Dense accuracy, dense benchmark, setup/cache, operator UX, observability/recovery, CI, and claim-boundary gates are green for that exact identity. | Route-state row must be `enabled`; operator envelope must classify the route as interactive or advisory with evidence date and limits. | Any statement widens from one identity to all dense models, broad Apple Silicon, MacBook, full Metal, or speedup. |
| Dense SLM `chat` excellent for one supported model identity | Dense `ask` verdict plus `M4-DENSE-CHAT-001` chat receipts and workload receipts are green. | Multi-turn history, timeout/cancel, per-turn receipts, and token IDs/text must be present where required. | Chat evidence is stale, route state disagrees, or service readiness is inferred from CLI chat. |
| Dense local `serve` excellent for one supported model identity | Dense accuracy/benchmark gates plus all dense service gates are green. | localhost safety defaults, ready/health, streaming, failure semantics, queue/backpressure, and per-request receipts must pass. | The wording implies production hosting, broad OpenAI compatibility, BitNet serve, or non-local deployment readiness. |
| Dense long-context route | Context-specific go only for recorded prompt/context profiles. | `M4-CONTEXT-002` quality and timing receipts must match the route and profile; operator class remains batch when the envelope says batch. | The request exceeds recorded context, omits truncation/guardrail behavior, or claims general long-context quality. |
| BitNet `ask` excellent for the accepted artifact | BitNet accuracy, benchmark/warm, setup/cache, operator UX, observability/recovery, CI, and claim-boundary gates are green for Microsoft I2_S plus external tokenizer. | Route-state row must be `enabled`; claim must stay BitNet-only and one-shot or warm-session scoped. | Dense SLM evidence is used, tokenizer identity differs, fallback is ambiguous, or chat/serve readiness is implied. |
| BitNet warm-session route | BitNet `ask` verdict plus matching warm-session regression and progress/failure receipts. | Variable-prompt and repeated-prompt behavior, timeout, progress, per-turn or per-prompt receipts, aggregate timing, and memory must be present. | Warm evidence is used to enable chat or serve without separate ready gates. |
| BitNet chat | No-go until the BitNet chat row is `enabled` and chat receipts satisfy the chat gate. | Requires variable-prompt stability, repeated-prompt determinism, timeout/cancel, per-turn receipts, trace correlation, and failure receipts. | The route matrix remains `disabled_without_ready_gate`. |
| BitNet serve | No-go until the BitNet serve row is `enabled` and serve receipts satisfy the serve gate. | Requires health/ready/completion, streaming, timeout/cancel, queue/backpressure, per-request receipts, trace correlation, and failure receipts. | The route matrix remains `disabled_without_ready_gate`. |
| Full `apple-m4-metal`, Apple QK256, Neural Engine, MPSGraph, MacBook, or broad Apple Silicon route | No-go. | Requires separate future receipt families and tracker items, not this release matrix. | Any wording treats dense CPU/NEON, BitNet CPU/NEON, or phase-scoped Metal evidence as full backend/platform proof. |

## Release Bundle Checklist

Before an envelope update lands, the PR or release note must link the exact
evidence bundle and include these checks:

```bash
bitnet mac evidence --json-out target/apple-m4-inference-excellence/release/evidence-summary.json --json
bitnet mac report-refresh --json-out target/apple-m4-inference-excellence/release/report-refresh.json --explain --open-targets --json
bitnet mac regression-dashboard --json-out target/apple-m4-inference-excellence/release/regression-dashboard.json --markdown-out target/apple-m4-inference-excellence/release/regression-dashboard.md --explain --open-targets --json
bitnet mac receipts-check target/apple-m4-inference-excellence/release/evidence-summary.json --json
bitnet mac receipts-check target/apple-m4-inference-excellence/release/regression-dashboard.json --json
bitnet mac evidence replay --bundle <release-bundle>/manifest.json --dry-run --json
cargo run --locked -p xtask --no-default-features -- claim-lint --scope apple-m4 --check
cargo run --locked -p xtask --no-default-features -- campaign check apple-m4-inference-excellence
cargo run --locked -p xtask --no-default-features -- campaign generate --check
```

If the release changes public timing, quality, route readiness, default model,
supported model state, compatibility status, or operator envelope wording, cite
a release-gate hardware workflow run with `enable_run=true`,
`run_class=release_gate`, and retention of at least 90 days.

## Current Boundary

As of this matrix, the release process can evaluate future envelope updates, but
the matrix itself does not mark any new route excellent. BitNet chat and BitNet
serve remain no-go until their route-state rows are enabled by matching ready
gate receipts. Full `apple-m4-metal`, QK256, Neural Engine, MPSGraph, MacBook,
and broad Apple Silicon claims remain unsupported.
