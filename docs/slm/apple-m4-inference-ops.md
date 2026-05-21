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
- the M4 route-state matrix for dense SLM and BitNet ask, chat, warm-session,
  serve, streaming, disabled, batch-only, and unsupported surfaces;
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

The `route_state_matrix` field is model-free and descriptive. Enabled and
batch-only rows point at their required evidence item and receipt family;
disabled BitNet chat/serve/streaming rows point at the required gate receipt
families; unsupported backend rows remain non-routable until separate full-route
receipts exist.

Validate the receipt with:

```bash
bitnet mac receipts-check target/apple-m4-inference-ops/mac-status.json --json
```

## Evidence Replay Bundles

`M4-EVIDENCE-REPLAY-001` adds a model-free dry-run replay/audit command:

```bash
bitnet mac evidence replay \
  --bundle ci/hardware/apple-m4-mac-mini/2026-05-21T145609Z/evidence-replay/manifest.json \
  --dry-run \
  --json
```

The committed manifest uses artifact kind `apple_m4_evidence_replay_bundle`.
It lists exact audit commands, git and binary identity, dense SLM and BitNet
model/tokenizer identity, receipt inputs, dashboard outputs, the expected
advisory regression result, and the claim boundary. The dry-run command writes
an `apple_m4_evidence_replay_dry_run` receipt to
`target/apple-m4-inference-ops/evidence-replay-dry-run.json` by default.

The replay dry-run validates only the committed manifest and referenced receipt
paths. It does not execute replay commands, run live inference, download
models, validate uncommitted artifacts, enable BitNet chat or serve, or make
Metal, QK256, Neural Engine, MPSGraph, MacBook, broad quality, broad
performance, speedup, or broad Apple Silicon claims.

Validate the bundle and dry-run receipt with:

```bash
bitnet mac receipts-check \
  ci/hardware/apple-m4-mac-mini/2026-05-21T145609Z/evidence-replay/manifest.json \
  --json
bitnet mac receipts-check \
  target/apple-m4-inference-ops/evidence-replay-dry-run.json \
  --json
```

## Operator Workload Suite

`M4-WORKLOAD-001` adds a model-free workload manifest command:

```bash
bitnet mac workload \
  --suite m4-operator \
  --json-out ci/hardware/apple-m4-mac-mini/2026-05-21T171832Z/workload/summary.json \
  --json
```

The command writes an `apple_m4_operator_workload_suite` receipt. It does not
run live inference or download models. The committed manifest covers the
operator workflows `summarize`, `extract`, `classify`, `json`, `rewrite`, and
`table_qa` across dense SLM ask/chat/warm-session/serve surfaces and the enabled
BitNet ask/warm-session surfaces. BitNet chat and serve remain gate-disabled
route boundaries in the workload plan.

The receipt records:

- six workflow prompts with mechanical checks, not LLM judging;
- 48 route-plan entries covering the workflow, model family, and route-surface
  matrix;
- the current route-state matrix and report inventory used to explain each
  enabled, batch-only, disabled, or unsupported route;
- exact live commands an operator would run later for per-route receipts.

Validate the committed manifest with:

```bash
bitnet mac receipts-check \
  ci/hardware/apple-m4-mac-mini/2026-05-21T171832Z/workload/summary.json \
  --json
```

This is a workload plan and receipt-contract check. It is not broad assistant
quality proof, BitNet chat or serve enablement, production hosting proof, full
Metal inference, QK256, Neural Engine, MPSGraph, MacBook evidence, speedup, broad
performance, or broad Apple Silicon evidence.

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
