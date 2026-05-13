# Routed CI Rollout Plan

This document is the implementation playbook for moving BitNet-rs to routed,
Linux-first PR verification. It converts the rollout north star into small,
reviewable PRs with explicit economics, proof boundaries, files, acceptance
criteria, and validation commands.

## North star

Default pull requests should receive cheap, deterministic, Linux-only proof
that is scoped to the crates and risk introduced by the diff. Expensive proof
still exists, but it is reserved for `main`, schedules, release gates,
hardware/campaign lanes, or explicit labels.

The target cost model is:

| PR class | Default target |
| --- | ---: |
| Ordinary Rust PR | 25-34 LEM |
| Docs / tracking PR | 3-8 LEM |
| Manifest / toolchain / global-risk PR | 35-50+ LEM |

`policy/ci-budget.toml` owns budget bands and runner multipliers. The relevant
operating vocabulary is:

- preferred ordinary-PR budget: 25 LEM,
- normal default limit: 35 LEM,
- macOS runner multiplier: 10x Linux,
- Windows runner multiplier: 2x Linux,
- GPU Docker multiplier: 6x Linux,
- override labels: `full-ci`, `ci-budget-override`, and `ci-budget-ack`.

## Agent operating rules

Each rollout PR must be one boundary change. Agents should avoid unrelated Rust
or runtime edits unless the work item explicitly scopes them in.

Required behavior for every rollout PR:

1. Start from a clean branch based on `origin/main` when a remote is available.
2. Make one routed-CI boundary change only.
3. Update workflow routing, policy lane data, and documentation together.
4. Run the validation commands listed for the work item.
5. Commit the change with the work-item title or an equivalent conventional
   commit subject.
6. Open the PR with the required body shape below.
7. Treat CI, bot, and reviewer feedback as implementation feedback for that
   same work item; fix only the first real failing cause.
8. Merge only when required checks are green and GitHub reports the PR
   mergeable.

## Required PR body

Every rollout PR must use this body shape so reviewers can compare economics
and proof boundaries without reading the workflow diff first:

```md
## Summary

## CI economics
- Default PR LEM before:
- Default PR LEM after:
- Lanes removed from default:
- Lanes still available by label/main/manual:
- Branch protection impact:

## Verification preserved
- What failure mode this still catches:
- What moved to main/label/nightly:
- Why that is acceptable:

## Boundaries
- No macOS default PR runner
- No Windows default PR runner
- No Docker/model/download default PR work
- No branch-protection change unless explicitly scoped
- No unrelated Rust/runtime changes

## Validation
- [ ] command
- [ ] command
```

## Routing principles

The rollout uses these invariants throughout the queue:

- Ordinary PRs should not start macOS, Windows, Docker, model-download, or live
  hardware jobs by default.
- Linux default proof should remain deterministic and Rust-native.
- Expensive lanes must be reachable by explicit label, `main`, schedule,
  release gate, or manual dispatch.
- Path filtering is only an optimization; `PR Gate Success` must eventually be
  the required-check authority so skipped unselected lanes do not become
  branch-protection traps.
- Advisory lanes can be noisy, flaky, or informational only if they are not
  selected as blocking gates.
- If a lane is selected as blocking, failures must fail honestly.

## Rollout phases

### Phase 1: immediate default-cost waste

These PRs remove expensive or duplicate lanes from ordinary PRs without
changing the deeper routing architecture.

| Order | PR title | Primary default-PR effect |
| ---: | --- | --- |
| 1 | `ci: remove macOS from ordinary PRs` | Removes 10x macOS broad PR work. |
| 2 | `ci: move performance smoke off default PRs` | Removes duplicate full-workspace performance smoke compile. |
| 3 | `ci: move test telemetry to main and label` | Removes advisory 18 LEM telemetry from ordinary PRs. |
| 4 | `ci: risk-gate MSRV compatibility` | Runs MSRV for dependency/toolchain/API risk, not leaf edits. |

### Phase 2: authoritative routing

These PRs make the planner and gate the source of truth rather than duplicating
routing shell logic in each workflow.

| Order | PR title | Primary effect |
| ---: | --- | --- |
| 5 | `ci(plan): emit stable routing schema` | Adds stable `ci-plan.json` for workflows and agents. |
| 6 | `ci(gate): make PR Gate consume ci plan` | Makes gate wait only for selected blocking lanes. |
| 7 | `ci: add soft budget guard` | Surfaces LEM warnings and override-label semantics. |

### Phase 3: narrow default Linux proof

These PRs keep Linux proof but scope it to changed crates and risk.

| Order | PR title | Primary effect |
| ---: | --- | --- |
| 8 | `ci-core: broaden no-Rust fast path` | Avoids cargo work for docs/campaign/receipt-only changes. |
| 9 | `ci-core: package-select build/test surface` | Tests changed packages, dependents, and canaries. |
| 10 | `feature-matrix: reduce ordinary PR feature smoke` | Makes `cpu+full-cli` risk-selected, not universal. |

### Phase 4: honest advisory lanes

| Order | PR title | Primary effect |
| ---: | --- | --- |
| 11 | `ci: stop selected gates from swallowing failures` | Blocking lanes fail honestly; advisory lanes are declared advisory. |
| 12 | `gpu-ci: remove duplicate CPU native check` | Removes CPU-only duplicate from GPU CI and narrows GPU triggers. |

### Phase 5: coverage as measured evidence

| Order | PR title | Primary effect |
| ---: | --- | --- |
| 13 | `coverage: keep Codecov lane quiet and policy-declared` | Keeps coverage label/main/manual only with quiet Codecov config. |
| 14 | `coverage: add coverage receipt and ignored-failure accounting` | Adds receipt and records ignored run failures. |

### Phase 6: cheap verification density

| Order | PR title | Primary effect |
| ---: | --- | --- |
| 15 | `policy(clippy): document strict agent lint baseline` | Documents lint tiers without activating new lint debt. |
| 16+ | lint activation slices | Activates one lint family at a time. |

### Phase 7: required-check consolidation

| Order | PR title | Primary effect |
| ---: | --- | --- |
| 20 | `ci: make PR Gate the only required check` | Documents and applies the branch-protection migration. |

## Work-item specs

### PR 1: `ci: remove macOS from ordinary PRs`

**Scope files**

- `.github/workflows/macos-arm64.yml`
- `policy/ci-lanes.toml`
- `policy/ci-lane-whitelist.toml`
- `docs/ci/cost-and-verification-policy.md`
- `docs/ci/labels.md`

**Required routing**

Run macOS only for push to `main`, manual dispatch, merge queue if required by
branch protection, Mac/Metal-specific paths, or labels: `macos`,
`apple-silicon`, `metal`, `full-ci`.

Remove ordinary PR triggers from broad Rust paths such as all `crates/**`,
`tests/**`, `xtask/**`, and generic `Cargo.toml` unless the PR is labeled.

**Acceptance**

- Normal Rust PRs do not launch `macos-14`.
- `macos`, `apple-silicon`, `metal`, and `full-ci` labels still opt in.
- Main/manual still preserves Apple proof.
- Policy lane costs show macOS is not default.

**Validation**

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
cargo run --locked -p xtask --no-default-features -- check-file-policy --report-dir target/bitnet/reports --fail-on-error
```

### PR 2: `ci: move performance smoke off default PRs`

**Scope files**

- `.github/workflows/performance-tracking.yml`
- `policy/ci-lanes.toml`
- `policy/ci-lane-whitelist.toml`
- `docs/ci/cost-and-verification-policy.md`

**Required routing**

Run performance tracking only on push to `main`, schedule, manual dispatch, or
labels: `performance`, `perf`, `full-ci`. Do not run the full-workspace smoke
check on ordinary PRs.

**Acceptance**

- Ordinary PRs do not run `Performance Baseline Tracking`.
- Main/schedule/manual still run.
- Labeled PRs still run.
- No branch-protection dependency is added.

**Validation**

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
```

### PR 3: `ci: move test telemetry to main and label`

**Scope files**

- `.github/workflows/test-telemetry.yml`
- `policy/ci-lanes.toml`
- `policy/ci-lane-whitelist.toml`
- `docs/ci/cost-and-verification-policy.md`

**Required routing**

Run telemetry only on push to `main`, manual dispatch, optional schedule, or
labels: `test-telemetry`, `slow-tests`, `full-ci`.

**Acceptance**

- Ordinary PRs do not run Test Telemetry.
- Main/manual/labeled runs still produce JUnit and slow-test summaries.
- Lane tables set `test-telemetry.default_pr = false`.

**Validation**

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
```

### PR 4: `ci: risk-gate MSRV compatibility`

**Scope files**

- `.github/workflows/compatibility.yml`
- `policy/ci-lanes.toml`
- `policy/ci-lane-whitelist.toml`
- `policy/ci-risk-packs.toml`
- `docs/ci/cost-and-verification-policy.md`

**Required routing**

Run MSRV for `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, `.cargo/**`,
public API surfaces, release/package surfaces, labels `msrv`, `compatibility`,
`full-ci`, push to `main`, and manual dispatch.

Keep the `manifest_release` risk pack selecting `compatibility-msrv` for global
dependency/toolchain risk.

**Acceptance**

- Leaf implementation PRs do not run MSRV by default.
- Manifest/toolchain/dependency PRs still run MSRV.
- Main still runs MSRV.
- `manifest_release` still selects MSRV.

**Validation**

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
cargo check --locked -p bitnet-common -p bitnet-models -p bitnet-tokenizers -p bitnet-quantization -p bitnet-kernels --tests --no-default-features --features cpu
```

### PR 5: `ci(plan): emit stable routing schema`

**Scope files**

- `xtask/src/ci/plan.rs`
- `xtask/src/ci/mod.rs`
- `policy/ci-budget.toml`
- `policy/ci-lanes.toml`
- `policy/ci-risk-packs.toml`
- `docs/ci/pr-plan.md`
- `tests/fixtures/ci-plan/**`

**Required schema**

`ci-plan.json` must include `schema_version`, `budget`, `classification`,
`selected_lanes`, `skipped_lanes`, `packages`, `risk_packs`, and `labels`.
The classification object must cover docs-only, tracker-only, Rust inputs,
manifest/toolchain, public API, GPU, macOS, model-validation, coverage request,
and full-CI request.

**Acceptance**

- Existing human summary still prints.
- New JSON schema is fixture-tested.
- Planner classifies docs-only, tracker-only, ordinary Rust,
  manifest/toolchain, GPU, macOS, `full-ci`, and coverage.
- Workflow behavior does not change in this PR.

**Validation**

```bash
cargo test -p xtask --no-default-features ci_plan --locked
cargo run --locked -p xtask --no-default-features -- ci plan --changed-file tests/fixtures/ci-plan/rust.txt --labels-json '[]' --json-out target/ci-plan.json --print
cargo run --locked -p xtask --no-default-features -- ci plan --changed-file tests/fixtures/ci-plan/docs.txt --labels-json '[]' --json-out target/ci-plan-docs.json --print
git diff --check
```

### PR 6: `ci(gate): make PR Gate consume ci plan`

**Scope files**

- `.github/workflows/pr-gate.yml`
- `.github/workflows/pr-plan.yml`
- `xtask/src/ci/plan.rs`
- `docs/ci/pr-gate-success.md`

**Required behavior**

`PR Gate Success` determines the PR head SHA, computes or downloads
`ci-plan.json`, waits only for selected blocking lanes, treats selected
blocking skipped lanes as failures, treats unselected skipped lanes as OK, and
prints selected lanes, skipped lanes, budget posture, and label overrides.

**Acceptance**

- PR Gate no longer has an independent path classifier.
- Path-filtered unselected workflows cannot create missing-required-check traps.
- Branch protection can later require only `PR Gate Success`.

**Validation**

```bash
cargo test -p xtask --no-default-features ci_plan --locked
git diff --check
```

Hosted validation should confirm docs-only PRs do not wait for Rust lanes,
ordinary Rust PRs wait for CI Core and feature smoke, and `full-ci` PRs wait
for selected expanded lanes.

### PR 7: `ci: add soft budget guard`

**Scope files**

- `xtask/src/ci/plan.rs`
- `.github/workflows/pr-plan.yml`
- `.github/workflows/pr-gate.yml`
- `docs/ci/cost-and-verification-policy.md`

**Required behavior**

Emit budget guard data for these bands:

| Estimated LEM | Behavior |
| ---: | --- |
| <=25 | preferred |
| 26-35 | normal default |
| 36-75 | warning |
| 76-100 | strong warning |
| 101-125 | require `ci-budget-ack` or `full-ci` |
| >125 | fail unless `ci-budget-override` or `full-ci` |

The first implementation may be advisory, but the schema must support later
enforcement.

**Acceptance**

- Budget warnings appear in the PR Plan summary.
- PR Gate understands budget override labels.
- No accidental hard failures occur before maintainers opt in.

**Validation**

```bash
cargo test -p xtask --no-default-features ci_plan_budget --locked
git diff --check
```

### PR 8: `ci-core: broaden no-Rust fast path`

**Scope files**

- `.github/workflows/ci-core.yml`
- `xtask/src/ci/plan.rs`
- `docs/ci/cost-and-verification-policy.md`

**Required behavior**

Replace the narrow tracker-only path with `no_rust_inputs`, `docs_only`,
`tracker_or_campaign_only`, and `hardware_receipt_only`. CI Core should skip
cargo build/test/clippy/doc work when `no_rust_inputs = true`, while preserving
campaign doctor/generate checks and receipt schema checks where relevant.

**Acceptance**

- Docs, campaign, and hardware receipt PRs do not compile Rust unless they
  touch Rust inputs.
- `CI Core Success` still emits for branch-protection compatibility.
- Rust PR behavior is unchanged.

**Validation**

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci plan --changed-file tests/fixtures/ci-plan/docs.txt --labels-json '[]' --json-out target/ci-plan-docs.json --print
```

### PR 9: `ci-core: package-select build/test surface`

**Scope files**

- `xtask/src/ci/plan.rs`
- `.github/workflows/ci-core.yml`
- `docs/ci/cost-and-verification-policy.md`

**Required behavior**

`xtask ci plan` computes changed packages, direct dependents, canary packages,
and whether a broad sweep is required. CI Core runs cargo test and clippy over
that package selection. Manifest/toolchain/shared foundational changes keep the
broad sweep.

**Acceptance**

- Foundational changes still get broad coverage.
- Leaf crate changes get scoped tests.
- CI summary prints selected packages and reasons.
- No macOS, Windows, Docker, or model download is introduced.

**Validation**

```bash
cargo test -p xtask --no-default-features ci_plan_packages --locked
cargo run --locked -p xtask --no-default-features -- ci plan --changed-file tests/fixtures/ci-plan/quantization.txt --labels-json '[]' --json-out target/ci-plan-quant.json --print
git diff --check
```

### PR 10: `feature-matrix: reduce ordinary PR feature smoke`

**Scope files**

- `.github/workflows/feature-matrix.yml`
- `policy/ci-risk-packs.toml`
- `policy/ci-lanes.toml`
- `docs/ci/cost-and-verification-policy.md`

**Required behavior**

Default ordinary PR feature checks are `no-features` and `cpu`. Run
`cpu+full-cli` for CLI/server/validation/model-cache/full-cli feature files,
manifest/toolchain changes, or labels `full-cli`, `feature-matrix`, `full-ci`.

**Acceptance**

- Ordinary PR feature matrix is cheaper.
- CLI/full-cli risk still gets checked.
- Main and `full-ci` still run the full matrix.

**Validation**

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
```

### PR 11: `ci: stop selected gates from swallowing failures`

**Scope files**

- `.github/workflows/validation.yml`
- `.github/workflows/test-framework.yml`
- `.github/workflows/compatibility.yml`
- `docs/ci/cost-and-verification-policy.md`
- `policy/ci-lanes.toml`

**Required behavior**

If a lane is selected as blocking, failures fail. If a lane is too flaky or too
expensive to fail honestly, move it to main, nightly, manual, or label-gated
execution and mark it advisory.

**Acceptance**

- No selected blocking lane uses `|| true` or equivalent swallowing.
- Advisory lanes are clearly named advisory and not selected by PR Gate as
  blocking.
- Policy tables reflect blocking versus advisory status.

**Validation**

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
```

### PR 12: `gpu-ci: remove duplicate CPU native check`

**Scope files**

- `.github/workflows/gpu-ci-matrix.yml`
- `policy/ci-lanes.toml`
- `policy/ci-risk-packs.toml`

**Required behavior**

Remove `native-check(cpu)` from GPU CI. Trigger GPU CI only on GPU paths or
labels `gpu-ci` and `full-ci`. Avoid generic manifest triggers unless GPU
dependencies changed or `full-ci` is present.

**Acceptance**

- GPU path PRs still run GPU feature compile checks.
- Ordinary manifest PRs do not fan out GPU CI automatically.
- Docker remains main/manual/label only.

**Validation**

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
```

### PR 13: `coverage: keep Codecov lane quiet and policy-declared`

**Scope files**

- `.github/workflows/coverage.yml`
- `codecov.yml`
- `README.md`
- `docs/ci/coverage.md`
- `policy/ci-lanes.toml`
- `policy/ci-lane-whitelist.toml`

**Required behavior**

Keep coverage out of CI Core. PR coverage runs only for `coverage` or
`full-ci`. Codecov comments and annotations stay disabled. The coverage flag is
`rust-cpu`, and the lane claim is CPU execution surface only.

**Acceptance**

- Ordinary PR cost is unchanged.
- Main gets coverage evidence.
- Labeled PRs can request coverage.
- Coverage makes no GPU, model, or hardware claim.

**Validation**

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
```

### PR 14: `coverage: add coverage receipt and ignored-failure accounting`

**Scope files**

- `.github/workflows/coverage.yml`
- `docs/ci/coverage.md`
- receipt-generation script embedded in the workflow or extracted under an
  appropriate tooling path

**Required receipt**

The coverage artifact must include a receipt with schema version, repo, lane,
flag, workflow, artifact booleans, claim boundary, and `ignored_run_failures`.
Main should fail if ignored run failures are nonzero unless explicitly allowed.

**Acceptance**

- Artifact includes the coverage receipt.
- Main fails when ignored failures are nonzero unless an explicit override is
  in scope.
- PR advisory coverage can upload the receipt even when non-blocking.

### PR 15: `policy(clippy): document strict agent lint baseline`

**Scope files**

- `docs/ci/clippy-policy.md`
- `policy/clippy-lints.toml`
- `clippy.toml`
- `Cargo.toml`
- `docs/ci/cost-and-verification-policy.md`

**Required behavior**

Document lint tiers without cleaning code or activating new lints in this PR:
Active, Staged, Planned 1.94/1.95, Debt, and Suppression. Suppressions must use
reasoned `expect`, not silent `allow`, once the relevant activation slice lands.

**Acceptance**

- No code cleanup is included.
- No new lint activation is included.
- Agents have a stable policy doc and TOML map.

**Validation**

```bash
cargo run --locked -p xtask --no-default-features -- check-lint-policy --report-dir target/bitnet/reports --fail-on-error
git diff --check
```

### PR 16+: lint activation slices

Activate one lint family per PR, in this order unless a maintainer scopes a
different slice:

1. suppression governance: `allow_attributes`,
   `allow_attributes_without_reason`,
2. panic family: no new `unwrap`, `expect`, `panic`, `unreachable`, unsafe
   slicing/indexing,
3. silent failure: `let_underscore_future`, `let_underscore_must_use`,
   `unused_result_ok`, `map_err_ignore`,
4. async/concurrency: `await_holding_lock`, `await_holding_refcell_ref`,
5. numeric: `cast_sign_loss`, `invalid_upcast_comparisons`, then staged
   truncation/precision lints.

### PR 20: `ci: make PR Gate the only required check`

**Scope files**

- `.github/workflows/pr-gate.yml`
- `docs/ci/pr-gate-success.md`
- `docs/ci/branch-protection.md`

**Required behavior**

Keep leaf checks as ordinary workflow checks, but branch protection requires
only `PR Gate Success`. PR Gate enforces selected blocking lanes from the plan.

**Acceptance**

- Docs explain branch-protection migration.
- Existing `CI Core Success` can remain for compatibility but is no longer
  required after the settings migration.
- Path-filtered skipped workflows no longer block.

This work item may require repository settings outside the codebase.

## Expected steady state

After phases 1-4, an ordinary Rust PR should spend roughly:

```text
PR Plan                         1 LEM
PR Gate                         1 LEM
CI Core scoped Linux proof      12-18 LEM
Feature smoke                   4-8 LEM
Policy                          3 LEM
ripr advisory                   4 LEM
--------------------------------------
Target                         25-34 LEM
```

A docs or tracking PR should spend roughly:

```text
PR Plan                         1 LEM
PR Gate                         1 LEM
Docs/campaign/receipt checks    2-6 LEM
--------------------------------------
Target                          3-8 LEM
```

A manifest or toolchain PR may exceed the ordinary target because it is a
global-risk PR and should receive broad Linux proof plus MSRV.
