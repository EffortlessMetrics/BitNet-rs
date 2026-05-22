# Apple M4 Inference Ops

This page tracks the operator layer above the completed Apple M4 dense SLM,
server, Metal phase, BitNet local-answer, BitNet eval/benchmark, and BitNet
productization campaigns. The goal is to make the Mac mini easy to operate and
easy to audit without turning generic PR CI into live hardware evaluation.

## Status Receipt

`M4-INF-OPS-001` adds a model-free status command:

```bash
bitnet mac status
```

The command writes an `apple_m4_inference_status` receipt to
`target/apple-m4-inference-ops/mac-status.json` by default. It summarizes:

- disk/cache recommendation from `bitnet mac models`;
- dense SLM default/support/cache readiness and ask/chat/serve availability;
- BitNet ask/warm readiness and explicit disabled chat/serve state;
- current local report inventory for dense eval v2, dense benchmark v2, BitNet
  eval, BitNet benchmark, and variable warm-session evidence;
- known operator commands for model fetch/verify, ask, chat, serve, doctor,
  smoke, regression, BitNet warm, and the BitNet chat gate.

The status receipt is not a live model run. It records
`requested_backend=apple-m4-cpu-neon`, `runtime_api=cpu`, `fallback_used=false`,
and an explicit claim boundary:

```text
status_only=true
no_live_model_run=true
dense_slm_and_bitnet_evidence_separated=true
bitnet_chat_enabled=false
bitnet_serve_enabled=false
full_metal_inference_claimed=false
qk256_apple_claimed=false
neural_engine_execution_claimed=false
mpsgraph_inference_claimed=false
macbook_evidence=false
broad_apple_silicon_claim=false
broad_performance_claim=false
speedup_claim=false
```

Validate the receipt with:

```bash
bitnet mac receipts-check target/apple-m4-inference-ops/mac-status.json --json
```

## Report Refresh Manifest

`M4-INF-OPS-002` adds a model-free manifest command:

```bash
bitnet mac report-refresh
```

The command writes an `apple_m4_report_refresh_manifest` receipt to
`target/apple-m4-inference-ops/report-refresh-manifest.json` by default. It
inventories committed M4 report families:

- dense SLM eval v2 summaries;
- dense SLM benchmark v2 summaries;
- BitNet seeded eval answer-corpus receipts;
- BitNet one-shot/fixed-warm benchmark summaries;
- BitNet variable-prompt warm-session receipts.

The manifest records each family separately with expected artifact kind, report
paths, latest report, model IDs when present, advisory/nightly/release refresh
tiers, and model-free validation commands. Generic PR CI may generate and
validate this manifest because it only reads committed receipts. It must not
download models, run live M4 inference, run long resident soaks, or mix dense
SLM evidence with BitNet evidence.

Validate the manifest with:

```bash
bitnet mac receipts-check \
  target/apple-m4-inference-ops/report-refresh-manifest.json \
  --json
```

## Regression Dashboard

`M4-INF-OPS-003` adds a model-free dashboard command:

```bash
bitnet mac regression-dashboard
```

The command writes:

```text
target/apple-m4-inference-ops/regression-dashboard.json
target/apple-m4-inference-ops/regression-dashboard.md
```

The JSON receipt uses artifact kind `apple_m4_regression_dashboard`. It groups
committed reports by evidence family, artifact kind, model ID, model SHA,
tokenizer authority, backend, and fallback state. A group is comparable only
when at least two matching reports exist; otherwise the dashboard records
`comparison_status=insufficient_history` instead of inventing a regression
claim.

The dashboard keeps dense SLM and BitNet families separate and records
`bitnet_chat_enabled=false`, `bitnet_serve_enabled=false`,
`full_metal_inference_claimed=false`, `qk256_apple_claimed=false`, and no broad
quality, performance, or speedup claim.

The 2026-05-15T1845Z dense SLM durable refresh gives the dashboard two
benchmark reports for each supported dense model identity. The dashboard can
group those reports by model SHA, tokenizer authority, backend, and fallback
state, but strict `bitnet mac regression` remains profile-set aware: direct
comparison from the 2026-05-15 baseline to the 2026-05-15T1845Z refresh stops
with `profiles_required mismatch` because the refresh adds `resident_100`.

## Operator Envelope V2

`M4-INF-OPS-004` publishes the command-to-receipt contract in
`docs/slm/apple-m4-operator-envelope-v2.md`. It maps each supported M4 operator
command to its receipt kind, gate, report family, and unsupported claim
boundary.

## Remaining Ops Work

The inference-ops lane is complete when the campaign tracker marks
`M4-INF-OPS-004` merged. Follow-on refresh work is tracked in
`apple-m4-durable-inference-evidence`; it keeps generic PR CI model-free and
uses matching report history before describing trends.

Live M4 model runs belong in local, advisory, scheduled, or release lanes.
The shared M4 CI lane and artifact-retention contract is recorded in
`docs/slm/apple-m4-evidence-ci-lanes.md`.

`M4-ROUTE-MATRIX-001` adds the route-state table embedded in
`bitnet mac status --json` and `bitnet mac evidence --json`. The documented
operator view is `docs/slm/apple-m4-route-state-matrix.md`.

`M4-WORKLOAD-001` adds the model-free workload-suite contract:
`bitnet mac workload --suite m4-operator`. The documented operator view is
`docs/slm/apple-m4-workload-suite.md`.
