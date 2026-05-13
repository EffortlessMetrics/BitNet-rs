# Routed Verification Rollout

This document is the implementation spec for the CI economics rollout. It turns
BitNet-rs CI into a routed verification system where ordinary PRs get cheap,
Linux-only, deterministic proof and expensive evidence moves to main,
schedules, release lanes, hardware/campaign lanes, or explicit labels.

## North star

> Default PRs get cheap, Linux-only, deterministic, crate/risk-scoped proof.
> Expensive proof still exists, but only on main, schedule, release,
> hardware/campaign lanes, or explicit labels.

The budget vocabulary is already present in the repository policy files:

| Concept | Value |
| --- | ---: |
| Preferred default PR budget | 25 LEM |
| Normal default PR limit | 35 LEM |
| macOS multiplier | 10x Linux |
| Windows multiplier | 2x Linux |
| GPU Docker multiplier | 6x Linux |
| Budget override labels | `full-ci`, `ci-budget-override`, `ci-budget-ack` |

The rollout must preserve verification quality by removing unrelated work from
ordinary PRs, not by deleting proof. Moved lanes must remain available through
main, scheduled, manual, release, hardware, campaign, or label-gated routes.

## Operating rules for each rollout PR

Each implementation PR must follow this sequence:

1. Start from fresh `origin/main`.
2. Make one boundary change per PR.
3. Avoid runtime code unless the work item explicitly scopes runtime code.
4. Run targeted validation listed for the work item.
5. Watch CI.
6. Fix only the first real failing cause.
7. Merge only when required checks are green.
8. After merge, fast-forward local `main` and re-check queue state.

Use the PR body template in `docs/ci/routed-verification-pr-template.md`.

## Global boundaries

Every rollout PR must preserve these boundaries unless the work item explicitly
says otherwise:

- No macOS default PR runner.
- No Windows default PR runner.
- No Docker, model download, or hardware validation in default PR work.
- No branch-protection change unless explicitly scoped.
- No unrelated Rust/runtime changes.
- No silent widening of ordinary PR triggers.

## Rollout phases

### Phase 1: remove immediate default-cost waste

| Order | Title | Primary outcome | Default PR budget effect |
| ---: | --- | --- | --- |
| 1 | `ci: remove macOS from ordinary PRs` | macOS/Metal proof moves to main/manual/labels/path-specific routing. | Removes 10x runner from ordinary Rust PRs. |
| 2 | `ci: move performance smoke off default PRs` | Performance tracking runs on main, schedule, manual, or perf labels. | Removes duplicate workspace CPU check. |
| 3 | `ci: move test telemetry to main and label` | Advisory nextest/JUnit telemetry moves off ordinary PRs. | Removes 18 LEM advisory duplicate lane. |
| 4 | `ci: risk-gate MSRV compatibility` | MSRV runs for global dependency/toolchain/API/release risk and labels. | Removes 12 LEM from leaf implementation PRs. |

### Phase 2: make routing authoritative

| Order | Title | Primary outcome |
| ---: | --- | --- |
| 5 | `ci(plan): emit stable routing schema` | `xtask ci plan` emits stable `ci-plan.json` for workflows and agents. |
| 6 | `ci(gate): make PR Gate consume ci plan` | PR Gate waits only for selected blocking lanes. |
| 7 | `ci: add soft budget guard` | Budget posture and override labels become visible in PR Plan and PR Gate. |

### Phase 3: narrow default Linux proof

| Order | Title | Primary outcome |
| ---: | --- | --- |
| 8 | `ci-core: broaden no-Rust fast path` | Docs/tracker/campaign/hardware-receipt PRs skip cargo work. |
| 9 | `ci-core: package-select build/test surface` | CI Core scopes cargo work to changed packages, direct dependents, and canaries. |
| 10 | `feature-matrix: reduce ordinary PR feature smoke` | Default feature smoke becomes `no-features` plus `cpu`; `cpu+full-cli` is risk-gated. |

### Phase 4: make advisory lanes honest

| Order | Title | Primary outcome |
| ---: | --- | --- |
| 11 | `ci: stop selected gates from swallowing failures` | Selected blocking lanes fail honestly; flaky lanes become advisory/off-default. |
| 12 | `gpu-ci: remove duplicate CPU native check` | GPU CI stops duplicating generic CPU compile proof. |

### Phase 5: keep coverage measured and explicit

| Order | Title | Primary outcome |
| ---: | --- | --- |
| 13 | `coverage: keep Codecov lane quiet and policy-declared` | Coverage remains main/manual/label-gated with quiet Codecov behavior. |
| 14 | `coverage: add coverage receipt and ignored-failure accounting` | Coverage artifacts include a receipt and ignored-failure count. |

### Phase 6: increase cheap proof density

| Order | Title | Primary outcome |
| ---: | --- | --- |
| 15 | `policy(clippy): document strict agent lint baseline` | Strict lint tiers are documented without activating new lints. |
| 16+ | lint activation slices | One lint family at a time, with small focused PRs. |

### Phase 7: consolidate branch protection

| Order | Title | Primary outcome |
| ---: | --- | --- |
| 20 | `ci: make PR Gate the only required check` | Branch protection requires only `PR Gate Success` after routing is proven. |

## Work item specs

The machine-readable queue is `policy/ci-routing-rollout.toml`. The TOML file
is the compact implementation backlog; this document explains the intent and
boundaries. If the two drift, update both in the same documentation-only PR.

### PR 1: remove macOS from ordinary PRs

Files:

- `.github/workflows/macos-arm64.yml`
- `policy/ci-lanes.toml`
- `policy/ci-lane-whitelist.toml`
- `docs/ci/cost-and-verification-policy.md`
- `docs/ci/labels.md`

Required routing:

- Run on push to `main`.
- Run on `workflow_dispatch`.
- Keep `merge_group` only if branch protection still needs it.
- Run on labels `macos`, `apple-silicon`, `metal`, and `full-ci`.
- Run on Mac/Metal-specific paths only.
- Remove broad ordinary PR triggers such as all `crates/**`, `tests/**`,
  `xtask/**`, and generic `Cargo.toml` unless selected by label/path risk.

Acceptance:

- No normal Rust PR launches `macos-14`.
- `macos`, `apple-silicon`, `metal`, and `full-ci` labels still work.
- Main/manual still preserves Apple proof.
- Policy lane costs reflect that macOS is not default.

Validation:

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
cargo run --locked -p xtask --no-default-features -- check-file-policy --report-dir target/bitnet/reports --fail-on-error
```

### PR 2: move performance smoke off default PRs

Files:

- `.github/workflows/performance-tracking.yml`
- `policy/ci-lanes.toml`
- `policy/ci-lane-whitelist.toml`
- `docs/ci/cost-and-verification-policy.md`

Required routing:

- Run on push to `main`.
- Run on schedule.
- Run on `workflow_dispatch`.
- Run on labels `performance`, `perf`, and `full-ci`.
- Do not run the workspace smoke check on ordinary PRs.

Acceptance:

- Ordinary PRs do not run `Performance Baseline Tracking`.
- Main/schedule/manual still run.
- Labeled PRs still run.
- No branch-protection dependency is introduced.

Validation:

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
```

### PR 3: move test telemetry to main and label

Files:

- `.github/workflows/test-telemetry.yml`
- `policy/ci-lanes.toml`
- `policy/ci-lane-whitelist.toml`
- `docs/ci/cost-and-verification-policy.md`

Required routing:

- Run on push to `main`.
- Run on `workflow_dispatch`.
- Schedule is optional.
- Run on labels `test-telemetry`, `slow-tests`, and `full-ci`.

Acceptance:

- No ordinary PR runs Test Telemetry.
- Main/manual/labeled runs still produce JUnit and slow-test summaries.
- Lane table sets `default_pr = false` for `test-telemetry`.

Validation:

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
```

### PR 4: risk-gate MSRV compatibility

Files:

- `.github/workflows/compatibility.yml`
- `policy/ci-lanes.toml`
- `policy/ci-lane-whitelist.toml`
- `policy/ci-risk-packs.toml`
- `docs/ci/cost-and-verification-policy.md`

Required routing:

- Run for `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, and `.cargo/**`.
- Run for public API surfaces.
- Run for release/package surfaces.
- Run on labels `msrv`, `compatibility`, and `full-ci`.
- Run on push to `main` and `workflow_dispatch`.

Acceptance:

- Leaf implementation PRs do not run MSRV by default.
- Manifest/toolchain/dependency PRs still run MSRV.
- Main still runs MSRV.
- The `manifest_release` risk pack still selects `compatibility-msrv` for global
  dependency/toolchain risk.

Validation:

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
cargo check --locked -p bitnet-common -p bitnet-models -p bitnet-tokenizers -p bitnet-quantization -p bitnet-kernels --tests --no-default-features --features cpu
```

### PR 5: emit stable routing schema

Files:

- `xtask/src/ci/plan.rs`
- `xtask/src/ci/mod.rs`
- `policy/ci-budget.toml`
- `policy/ci-lanes.toml`
- `policy/ci-risk-packs.toml`
- `docs/ci/pr-plan.md`
- `tests/fixtures/ci-plan/**`

Schema requirements are documented in `docs/ci/ci-plan-schema.md`.

Acceptance:

- Existing summary still prints.
- New JSON schema is fixture-tested.
- `xtask ci plan` classifies docs-only, tracker-only, ordinary Rust,
  manifest/toolchain, GPU, macOS, full-ci, and coverage changes.
- No workflow behavior changes yet.

Validation:

```bash
cargo test -p xtask --no-default-features ci_plan --locked
cargo run --locked -p xtask --no-default-features -- ci plan --changed-file tests/fixtures/ci-plan/rust.txt --labels-json '[]' --json-out target/ci-plan.json --print
cargo run --locked -p xtask --no-default-features -- ci plan --changed-file tests/fixtures/ci-plan/docs.txt --labels-json '[]' --json-out target/ci-plan-docs.json --print
git diff --check
```

### PR 6: make PR Gate consume ci plan

Files:

- `.github/workflows/pr-gate.yml`
- `.github/workflows/pr-plan.yml`
- `xtask/src/ci/plan.rs`
- `docs/ci/pr-gate-success.md`

Acceptance:

- PR Gate no longer has its own path classifier.
- PR Gate waits only for selected blocking lanes.
- Selected blocking lane `skipped` is failure.
- Unselected skipped lanes are OK.
- Summary prints selected lanes, skipped lanes, budget posture, and label
  overrides.
- Branch protection can later require only `PR Gate Success`.

Validation:

```bash
cargo test -p xtask --no-default-features ci_plan --locked
git diff --check
```

Hosted validation:

- Docs-only PR should not wait for Rust lanes.
- Ordinary Rust PR should wait for CI Core and feature smoke.
- `full-ci` PR should wait for selected expanded lanes.

### PR 7: add soft budget guard

Files:

- `xtask/src/ci/plan.rs`
- `.github/workflows/pr-plan.yml`
- `.github/workflows/pr-gate.yml`
- `docs/ci/cost-and-verification-policy.md`

Budget guard behavior:

| Estimated LEM | Behavior |
| ---: | --- |
| <= 25 | Preferred |
| 26-35 | Normal default |
| 36-75 | Warning |
| 76-100 | Strong warning |
| 101-125 | Require `ci-budget-ack` or `full-ci` |
| > 125 | Fail unless `ci-budget-override` or `full-ci` |

Acceptance:

- Budget warnings appear in PR Plan summary.
- PR Gate understands budget override labels.
- No accidental hard failures until maintainers opt in.

Validation:

```bash
cargo test -p xtask --no-default-features ci_plan_budget --locked
git diff --check
```

### PRs 8-20

The remaining work items are fully enumerated in
`policy/ci-routing-rollout.toml`. Keep each PR scoped to one item and copy the
TOML acceptance and validation fields into the PR body.

## Expected cost outcome

After phases 1-4, an ordinary Rust PR should be approximately:

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

Docs, tracking, campaign, and receipt-only PRs should target 3-8 LEM. Manifest
or toolchain PRs may exceed ordinary limits because they are global-risk PRs,
not ordinary implementation PRs.
