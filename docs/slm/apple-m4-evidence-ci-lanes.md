# Apple M4 Evidence CI Lanes

This contract keeps Apple M4 evidence repeatable without turning every PR into a
live hardware run. It applies to dense SLM and BitNet evidence used by the M4
operator envelope, report-refresh manifest, regression dashboard, and campaign
status.

The lane split is deliberately conservative:

| Lane | Trigger | Required checks | Live M4 model run | Blocking behavior |
|---|---|---|---|---|
| Generic PR Tier 0 | Pull requests that touch M4 docs, corpus files, committed summaries, workflows, or tracker state | Parser tests, scorer tests, dry-run corpus shape, committed receipt schema checks, self-baseline regression checks, generated dashboard checks, diff hygiene | no | Blocks only on model-free contract failures |
| Advisory local | After syncing `main`, changing model cache state, changing receipt schemas, or changing M4 CLI/operator code | `bitnet mac models`, `bitnet mac status`, `bitnet mac evidence`, `bitnet mac evidence replay --dry-run`, `bitnet mac report-refresh`, `bitnet mac regression-dashboard`, targeted `bitnet mac receipts-check` | no by default | Advisory; publish only when a tracker item asks for evidence |
| Scheduled M4 refresh | Nightly or weekly on the M4 Mac mini when disk/cache preflight passes | Dense SLM eval/benchmark, BitNet eval/benchmark, BitNet variable warm, report-refresh, regression-dashboard, receipt validation | yes | Non-blocking for generic PRs; failures open repair work or block later claims |
| Release gate | Before publishing a new M4 expectation envelope or promoting a route class | Full supported dense matrix, accepted BitNet artifact/tokenizer eval and benchmark, warm-session receipts, service conformance, dashboard refresh, operator docs | yes | Blocks the release claim when quality, timeout, fallback, identity, or required receipt validation fails |

Generic PR Tier 0 must never fetch model binaries, run live M4 inference, run
BitNet chat or serve, publish fresh hardware timing, or claim broad quality or
performance. It may validate committed receipts and dry-run corpora because
those operations are model-free.

## Current Workflows

| Workflow | Lane | Purpose | Hardware |
|---|---|---|---|
| `.github/workflows/apple-m4-slm-eval-tier0.yml` | Generic PR Tier 0 | Validates seeded dense SLM corpus shape, scorer/parser behavior, committed M4 SLM eval/benchmark summaries, and model-free self-baseline regression checks | Ubuntu only |
| `.github/workflows/apple-m4-inference-ops-tier0.yml` | Generic PR Tier 0 and scheduled model-free refresh | Generates and validates the model-free report-refresh manifest and regression dashboard from committed receipts | Ubuntu only |
| `.github/workflows/apple-m4-dense-slm-regression.yml` | Manual hardware lane | Staged dense SLM hardware regression on a provisioned M4 runner after an explicit `enable_run=true` dispatch | Self-hosted Apple M4 only |

The manual hardware workflow is intentionally staged. A dispatch with
`enable_run=false` only explains the runner requirement. A dispatch with
`enable_run=true` is an explicit hardware evidence run, not a generic PR gate.

## Artifact Retention

Retain source evidence separately from generated summaries:

| Artifact class | Example path | Retention rule |
|---|---|---|
| Committed source receipts | `ci/hardware/apple-m4-mac-mini/<date>/slm-eval-v2/<model-id>/summary.json` | Keep at least the current report and previous matching baseline for every current dense SLM and accepted BitNet identity |
| Required child receipts | `ci/hardware/apple-m4-mac-mini/<date>/bitnet-eval-250-repaired/answer-corpus-runs/*.json` | Commit only when needed for receipt validation, generated text or token-ID audit, or failure taxonomy |
| Generated dashboards | `target/apple-m4-inference-excellence/regression-dashboard.json` | Regenerate from committed receipts; do not treat as source evidence unless a tracker item explicitly asks to commit it |
| Replay bundle manifests | `ci/hardware/apple-m4-mac-mini/<date>/evidence-replay/manifest.json` | Commit when a tracker item asks for replay/audit coverage; validate with `bitnet mac evidence replay --dry-run` and `bitnet mac receipts-check` |
| Operator workload manifests | `ci/hardware/apple-m4-mac-mini/<date>/workload/summary.json` | Commit when a tracker item asks for workflow/route coverage; validate with `bitnet mac receipts-check`; the manifest is model-free and not a live evidence refresh |
| Generic PR Tier 0 artifacts | `target/apple-m4-slm-eval-tier0/**`, `target/apple-m4-inference-ops-tier0/**` | Upload as short-lived CI artifacts for debugging; they are not evidence refreshes |
| Scheduled or manual hardware artifacts | `target/apple-m4-dense-slm-regression/<run-id>/**` | Upload with the workflow retention setting and commit only the accepted receipt bundle requested by the tracker |

Model binaries, local cache copies, and intermediate `target/` build products
are not evidence-retention targets.

## Dashboard Generation

Dashboards are derived from retained receipts. The model-free refresh command
set is:

```bash
bitnet mac models
bitnet mac status
bitnet mac evidence \
  --json-out target/apple-m4-inference-excellence/evidence-summary.json \
  --json
bitnet mac evidence replay \
  --bundle ci/hardware/apple-m4-mac-mini/2026-05-21T145609Z/evidence-replay/manifest.json \
  --dry-run \
  --json
bitnet mac workload \
  --suite m4-operator \
  --json-out ci/hardware/apple-m4-mac-mini/2026-05-21T171832Z/workload/summary.json \
  --json
bitnet mac report-refresh \
  --json-out target/apple-m4-inference-excellence/report-refresh-manifest.json \
  --explain \
  --open-targets \
  --json
bitnet mac regression-dashboard \
  --json-out target/apple-m4-inference-excellence/regression-dashboard.json \
  --markdown-out target/apple-m4-inference-excellence/regression-dashboard.md \
  --explain \
  --open-targets \
  --json
bitnet mac receipts-check target/apple-m4-inference-excellence/evidence-summary.json --json
bitnet mac receipts-check ci/hardware/apple-m4-mac-mini/2026-05-21T171832Z/workload/summary.json --json
bitnet mac receipts-check target/apple-m4-inference-excellence/regression-dashboard.json --json
```

The refresh commands do not run live inference or download models by default.
They report missing history, stale identities, warnings, and failed groups from
the committed evidence inventory.

## Hardware Timing Policy

Hardware-only timing jobs are non-blocking for ordinary PRs. Scheduled M4 runs
may produce advisory timing warnings without failing generic PR CI. A release
gate may opt into `--fail-on-drift` only when the release claim says timing or
memory regression is a blocker for that claim.

Quality, timeout, fallback, identity, and receipt-validation failures are not
timing warnings. They block a route promotion or expectation-envelope update
until the receipt is repaired, the baseline is intentionally reset, or the claim
is narrowed.

Dense SLM evidence and BitNet evidence remain separate. Dense Qwen receipts do
not prove BitNet quality, BitNet chat, BitNet serve, Metal inference, QK256,
Neural Engine, MPSGraph, MacBook behavior, broad Apple Silicon behavior,
speedup, broad quality, or broad performance.
