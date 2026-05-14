# Apple M4 Dense SLM Eval V2 Campaign

Campaign ID: `apple-m4-slm-eval-v2`

Status: active

## Objective

Move the Apple M4 dense SLM lane from bounded 10-case proof into broader,
reproducible quality and benchmark reporting: a 100-500 case deterministic
corpus, task-family pass rates, full latency and throughput profiles, and
regression dashboards without broad model-quality or Apple Silicon benchmark
claims.

## Why This Exists

`apple-m4-slm-eval-and-proof` made the dense Qwen M4 path measurable. It added
the seeded corpus, deterministic scoring, report schema, supported-model
reports, CI tiers, and regression comparison. That was the first proof layer.

The v1 report set is intentionally small. Its 10-case seeded score exposed
useful failure modes such as stop-token tails and JSON formatting, but it is not
wide enough to describe model quality by task family or to support operator
benchmark expectations across prompt lengths and resident sessions.

This campaign adds the next proof layer while keeping old reports comparable.
The v2 corpus is separate from v1, and live M4 timing runs stay advisory,
nightly, or release-scoped.

## Scope Boundary

This is a dense Qwen SLM lane for the M4 Mac mini. It does not prove BitNet,
full Apple Metal inference, QK256, Neural Engine execution, MPSGraph inference,
MacBook behavior, or broad Apple Silicon performance.

## End State

- A v2 seeded deterministic dense SLM eval corpus has at least 100
  parser-validated mechanical cases.
- Stop-token, prompt-template, and scoring failure taxonomy separates format
  failures from answer-content failures.
- Per-model reports publish task-family pass rates for every supported dense
  M4 model ID.
- Benchmark profiles report cold load, tokenizer load, prompt tokenize,
  prefill, TTFT, input tok/s, output tok/s, decode tok/s, wall time, peak
  memory, memory drift, and p50/p90/p99 summaries.
- Regression dashboards compare matching v2 reports without putting live M4
  model runs in generic required PR CI.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| M4-SLM-EVAL2-001 | merged | PR #4777 defined the campaign and added a 120-case deterministic dense SLM corpus v2 that dry-runs through parser/scoring with no live model claim. |
| M4-SLM-EVAL2-002 | in_progress | Add stop-token/template/scoring failure taxonomy for v2 reports. |
| M4-SLM-EVAL2-003 | proposed | Run supported dense M4 models and publish task-family pass-rate reports. |
| M4-SLM-EVAL2-004 | proposed | Refresh dense M4 benchmark profiles with p50/p90/p99 timing and memory fields. |
| M4-SLM-EVAL2-005 | proposed | Wire v2 eval/benchmark reports into advisory or nightly regression dashboards. |

## Review Policy

Each PR owns one item. Runtime eval PRs must preserve `apple-m4-cpu-neon`,
`fallback_used=false`, model/tokenizer identity, prompt-template authority,
generated text and token-ID coverage, timing fields, and dense-SLM-only claim
boundaries.

Parser, schema, fixture, tracker, and synthetic report checks belong in generic
CI. Live model downloads, full supported-model evals, long resident soaks, and
hardware timing summaries belong in local, advisory, scheduled, or release
lanes.
