# Apple M4 Dense SLM Eval And Proof Campaign

Campaign ID: `apple-m4-slm-eval-and-proof`

Status: active

## Objective

Turn the usable Apple M4 dense SLM path into a structured, receipt-backed local
model runner proof: seeded quality eval, first-class speed and stability
metrics, per-model reports, and lightweight regression economics without broad
benchmark claims.

## Why This Exists

The M4 dense SLM appliance is already useful. The default
`qwen2.5-0.5b-instruct-q8_0` and the supported non-default Qwen models have
bounded receipts, cache registration, doctor/smoke/regression tooling, and an
operator expectation envelope.

That is enough to call the path usable, but not enough to call it broadly
proven. This campaign adds the missing bridge from "works in recorded receipts"
to "we can prove how good it is, how fast it is, and when it regresses" for the
supported dense SLM set.

## Scope Boundary

This is a dense Qwen SLM lane for the M4 Mac mini. It does not prove BitNet,
full Apple Metal inference, QK256, Neural Engine execution, MPSGraph inference,
MacBook behavior, or broad Apple Silicon performance.

## End State

- A seeded deterministic dense SLM eval corpus exists and is parser-validated.
- Accuracy scoring supports exact, normalized, schema-style, numeric tolerance,
  keyword, and forbidden-token checks.
- Per-model reports record model/tokenizer identity, prompt template, backend,
  fallback status, generated text/token coverage, speed, memory, stability, and
  claim boundaries.
- All supported dense M4 model IDs can run through the same report contract.
- Generic PR CI remains lightweight; live M4 model runs stay advisory, nightly,
  or release-scoped.
- Dense SLM eval reports can be compared over time through regression tooling.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| M4-SLM-EVAL-001 | merged | #4656 defined the campaign and seeded deterministic eval corpus spec; corpus shape only, no runtime accuracy claim. |
| M4-SLM-EVAL-002 | merged | #4660 added deterministic exact/normalized/schema/numeric/keyword scoring fixtures; no live runtime accuracy claim. |
| M4-SLM-EVAL-003 | merged | #4663 defined and validated the per-model M4 dense SLM eval summary report schema. |
| M4-SLM-EVAL-004 | merged | #4670 published 2026-05-14 per-model reports for the three supported dense M4 model IDs. |
| M4-SLM-EVAL-005 | in_progress | Wire lightweight generic CI checks and document advisory/nightly/release M4 tiers. |
| M4-SLM-EVAL-006 | ready | Compare matching dense SLM eval reports through regression tooling. |

## Review Policy

Each PR owns one item. Runtime eval PRs must preserve `apple-m4-cpu-neon`,
`fallback_used=false`, model/tokenizer identity, prompt-template authority,
generated text and token-ID coverage, timing fields, and dense-SLM-only claim
boundaries.

Parser, schema, fixture, tracker, and synthetic report checks belong in generic
CI. Live model downloads, full supported-model evals, long resident soaks, and
hardware timing summaries belong in local, advisory, scheduled, or release
lanes.
