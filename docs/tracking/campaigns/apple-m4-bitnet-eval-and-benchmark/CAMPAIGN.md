# Apple M4 BitNet Eval And Benchmark Campaign

Campaign ID: `apple-m4-bitnet-eval-and-benchmark`

Status: active

## Objective

Move Apple M4 BitNet from bounded one-shot and fixed-warm proof into repeatable
eval and benchmark reporting: accepted artifact/tokenizer identity, seeded
mechanical quality checks, one-shot and warm timing envelopes, and regression
dashboards without broad platform or product-surface claims.

## Why This Exists

`apple-m4-local-answer` proved that the accepted Microsoft I2_S GGUF can produce
strict Apple M4 CPU/NEON BitNet local answers when paired with the external
Microsoft tokenizer and `bitnetcpp-answer` prompt authority. It also added an
explicit one-shot `bitnet mac ask` route and a fixed-prompt `bitnet mac
bitnet-warm` proof route.

That is enough to show BitNet can answer through narrow, receipt-backed paths.
It is not enough to make BitNet operator-ready. The next layer needs a larger
mechanical corpus, task-family scoring, timeout/failure taxonomy, reference-vs-
Rust comparison fields, and timing envelopes that tell operators where the slow
parts are.

## Scope Boundary

This is a BitNet lane for the M4 Mac mini. It uses the accepted Microsoft I2_S
GGUF, the external Microsoft tokenizer authority, and the `bitnetcpp-answer`
prompt authority. It does not use dense Qwen evidence as BitNet evidence, and it
does not enable BitNet chat or serve.

The campaign does not prove full Apple Metal inference, QK256, Neural Engine
execution, MPSGraph inference, MacBook behavior, broad Apple Silicon performance,
or broad BitNet quality.

## End State

- A BitNet-specific seeded deterministic corpus has at least 100
  parser-validated mechanical cases.
- Eval reports record generated text, token IDs, model/tokenizer/backend
  identity, fallback status, task-family scoring, timeout/failure taxonomy, and
  reference-vs-Rust comparison fields.
- Benchmark reports cover one-shot and warm paths with model load, tokenizer
  load, prompt tokenize, prefill, TTFT, input/output/decode throughput, wall
  time, peak memory, timeout boundaries, and p50/p90/p99 summaries.
- Regression dashboards compare matching BitNet eval and benchmark reports
  without putting live M4 model runs in generic required PR CI.
- Productization receives a clear handoff for variable warm sessions, then chat,
  then serve, only after receipts prove those surfaces.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| M4-BITNET-EVAL-001 | merged | PR #4895 defined the campaign and added a 100-case deterministic BitNet corpus that dry-runs through parser/scoring only. |
| M4-BITNET-EVAL-002 | merged | PR #4899 added BitNet eval/report schema and reference-vs-Rust comparison fields; PR #4904 fixed the no-panic policy follow-up. |
| M4-BITNET-EVAL-003 | proposed | Run and publish M4 BitNet seeded eval reports for the accepted artifact. |
| M4-BITNET-EVAL-004 | proposed | Publish one-shot and fixed-warm M4 BitNet benchmark reports. |
| M4-BITNET-EVAL-005 | proposed | Wire BitNet eval and benchmark reports into advisory/nightly regression dashboards. |

## Review Policy

Each PR owns one item. Parser, schema, fixture, tracker, and synthetic report
checks belong in generic CI. Live model downloads, full BitNet evals, long warm
sessions, and hardware timing summaries belong in local, advisory, scheduled, or
release lanes.

Runtime eval PRs must preserve `apple-m4-cpu-neon`, `fallback_used=false`, exact
model SHA, tokenizer SHA/authority, `bitnetcpp-answer` prompt authority,
generated text, generated token IDs, timing fields, memory fields where
available, and explicit claim boundaries.
