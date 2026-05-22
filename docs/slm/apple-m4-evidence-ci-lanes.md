# Apple M4 Evidence CI Lanes

`M4-CI-001` defines the M4 evidence CI and retention contract. It is a workflow
and operations contract only: it does not add new runtime proof, fetch models by
default, publish fresh timing, or expand the public M4 claim envelope.

## Lane Matrix

| Lane | Workflow | Trigger | Runner | PR required | Evidence produced | Retention |
|---|---|---|---|---|---|---|
| Generic PR SLM Tier 0 | `.github/workflows/apple-m4-slm-eval-tier0.yml` | `pull_request`, `push`, `workflow_dispatch` | GitHub Ubuntu | Routed when changed paths match | Parser/scorer tests, corpus dry-run, committed summary receipt checks, self-baseline regression checks | 7 day workflow artifact |
| Generic PR ops Tier 0 | `.github/workflows/apple-m4-inference-ops-tier0.yml` | `pull_request`, `push`, daily `schedule`, `workflow_dispatch` | GitHub Ubuntu | Routed when changed paths match | Model-free report-refresh manifest and regression dashboard generation | 7 day workflow artifact |
| Advisory hardware refresh | `.github/workflows/apple-m4-dense-slm-regression.yml` | `workflow_dispatch` with `enable_run=true`, `run_class=advisory` | Self-hosted `apple-m4-dense-slm` | No | Dense Qwen cache verification, smoke quality receipt, release performance receipt, advisory baseline comparison | Minimum 30 days |
| Scheduled hardware refresh | `.github/workflows/apple-m4-dense-slm-regression.yml` | Weekly `schedule` status plus manual dispatch with `run_class=scheduled` when a runner is available | Ubuntu for staged status, self-hosted M4 for live dispatch | No | Staged status until runner availability is confirmed; live dispatch produces the same receipt bundle as advisory refresh | Minimum 45 days for live dispatch |
| Release-gate refresh | `.github/workflows/apple-m4-dense-slm-regression.yml` | Manual dispatch with `enable_run=true`, `run_class=release_gate` before release envelope updates | Self-hosted `apple-m4-dense-slm` | No | Dense Qwen release evidence bundle with branch, commit, optional PR or release reference, baseline comparison, and claim boundary summary | Minimum 90 days |

## Required Boundaries

Generic PR CI is model-free. It may validate parsers, corpus shape, receipt
schemas, report manifests, dashboards, and committed-report comparisons. It must
not fetch model artifacts, run live M4 inference, run long resident soaks,
create fresh hardware timing, or require Apple hardware to merge an ordinary PR.

Hardware refreshes are advisory unless a release process explicitly cites a
green `run_class=release_gate` bundle. Hardware-only timing jobs have no
`pull_request` trigger and are not branch-protection requirements. A failed
advisory or scheduled hardware run is evidence to inspect, not a generic PR
blocker.

Scheduled invocations of `apple-m4-dense-slm-regression.yml` are staged status
runs until a provisioned runner is confirmed. To produce live scheduled
evidence, dispatch the workflow with `enable_run=true`,
`run_class=scheduled`, and a retention value of at least 45 days.

Release-gate refreshes are explicit. Before updating a public M4 expectation
envelope from release evidence, dispatch the hardware workflow with
`run_class=release_gate`, `enable_run=true`, and retention of at least 90 days.
The release evidence may support only the recorded dense Qwen Apple CPU/NEON
context unless a separate receipt family proves more.

## Artifact Retention

Tier 0 artifacts are small diagnostic outputs from model-free CI and are kept
for 7 days. They are useful for PR debugging, but committed receipts remain the
authoritative evidence surface.

Hardware receipt bundles are retained as workflow artifacts and must not include
model binaries or raw cache directories. The minimum retention window is:

| Run class | Minimum retention | Purpose |
|---|---:|---|
| `advisory` | 30 days | Developer or maintainer inspection after manual hardware refresh |
| `scheduled` | 45 days | Trend review across routine refreshes |
| `release_gate` | 90 days | Release-envelope review and post-release audit |

If a bundle needs to become durable evidence, commit only the bounded receipt or
summary JSON under `ci/hardware/apple-m4-mac-mini/<date>/...`; never commit
model binaries or local cache contents.

## Dashboard Generation

The ops Tier 0 workflow regenerates the model-free report-refresh manifest and
regression dashboard from committed receipts. The dashboard can describe
matching-history status, stale identities, missing history, advisory drift, and
invalid comparisons. It cannot turn an uncommitted workflow artifact into a
runtime claim.

## Claim Boundaries

These lanes may claim only that the CI and retention surfaces are defined and
that a specific workflow run produced the recorded artifacts. They must not
claim BitNet quality from dense Qwen evidence, full `apple-m4-metal` inference,
QK256 support, Neural Engine execution, MPSGraph inference, MacBook behavior,
broad Apple Silicon performance, broad model quality, or speedup.
