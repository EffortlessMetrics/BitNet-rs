# Apple M4 Durable Inference Evidence Campaign

Campaign ID: `apple-m4-durable-inference-evidence`

Status: active

## Objective

Turn the completed Apple M4 dense SLM and BitNet proof surfaces into durable,
repeatable evidence: longer resident dense SLM benchmark profiles, refreshed
matching-identity report pairs, dashboard comparisons with real history, and
operator envelopes that describe measured drift without broad Apple Silicon or
model-quality claims.

## Scope Boundary

This campaign sits after dense SLM eval v2, BitNet eval/benchmark, BitNet
productization, and inference ops. It does not reopen those campaigns unless a
regression proves they were wrong. It may add evidence-refresh contracts and
new receipts, but it must keep dense SLM and BitNet evidence separate.

Live M4 model runs, hardware timing refreshes, and long resident soaks stay in
local, advisory, scheduled, or release lanes. Generic required PR CI remains
model-free.

## End State

- `bitnet mac benchmark` includes a bounded `resident_100` dense SLM profile.
- Dense SLM benchmark v2 reports have matching-history pairs for each supported
  model identity.
- BitNet eval, benchmark, and variable warm-session reports have matching
  history for the accepted artifact/tokenizer identity.
- `bitnet mac regression-dashboard` has comparable dense SLM and BitNet groups
  instead of only `insufficient_history`.
- The operator envelope records refresh cadence, thresholds, and boundaries
  from matching-identity comparisons.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| M4-DURABLE-001 | merged | PR #4973 added `resident_100` to the dense M4 SLM benchmark v2 profile contract, parser, validator, tests, and docs without claiming a fresh live run. |
| M4-DURABLE-002 | proposed | Refresh dense SLM eval/benchmark reports for every supported M4 model, including `resident_100`, and compare against previous matching reports. |
| M4-DURABLE-003 | proposed | Refresh BitNet eval, benchmark, and variable warm-session reports for the accepted artifact/tokenizer identity. |
| M4-DURABLE-004 | proposed | Regenerate report-refresh and regression-dashboard artifacts with real comparable history groups. |
| M4-DURABLE-005 | proposed | Publish an operator envelope refresh from the durable evidence run. |

## Review Policy

Each PR owns one item. Parser, schema, fixture, receipt-validation, and docs
changes belong in ordinary PR validation. Live model refreshes and timing
comparisons belong in local, advisory, scheduled, or release lanes and must
record exact model/tokenizer/backend/fallback identity before any drift claim is
made.
