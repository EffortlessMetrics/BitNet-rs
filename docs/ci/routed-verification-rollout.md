# Routed Verification Rollout

This document is the implementation map for moving BitNet-rs CI to routed,
Linux-first verification. It is intentionally written as an agent-ready spec:
each work item is small, scoped, independently reviewable, and has explicit
validation commands.

## North star

Default pull requests should get cheap, deterministic, crate/risk-scoped proof.
Expensive proof still exists, but it runs only on `main`, schedules, release
lanes, hardware/campaign lanes, manual dispatch, or explicit labels.

Routing controls whether a lane runs. It must not control cost by starting an
expensive lane and then setting a cap just below healthy completion time. If a
lane is selected, its timeout should be sized as a hang guard with enough
cushion to produce the receipt or check result. If that is too expensive for
the event, route the lane elsewhere or select a smaller bounded profile before
expensive work begins.

The target shape is:

```text
ordinary PR = PR Plan + PR Gate + scoped Linux proof + feature smoke + policy + ripr
expensive PR = ordinary PR + explicit risk/label/main/scheduled/release lanes
```

The rollout preserves the existing budget vocabulary:

| Budget or multiplier | Value | Meaning |
| --- | ---: | --- |
| Preferred default budget | 25 LEM | Ordinary PRs should usually fit here. |
| Normal default limit | 35 LEM | Default PRs may use this, but should explain why. |
| macOS multiplier | 10x Linux | macOS is never part of ordinary PR CI. |
| Windows multiplier | 2x Linux | Windows is label/main/release only unless explicitly scoped. |
| GPU Docker multiplier | 6x Linux | Docker/GPU image work is label/main/manual only. |
| Override labels | `full-ci`, `ci-budget-override`, `ci-budget-ack` | Explicitly authorize elevated budget. |

LEM means Linux-equivalent minutes. Runner multipliers are declared in
`policy/ci-budget.toml`, `policy/ci-lanes.toml`, and
`policy/ci-lane-whitelist.toml`.

## Default PR boundaries

A default PR must not add any of the following without an explicit rollout item
that says so:

- macOS default PR runners,
- Windows default PR runners,
- Docker/model/download default PR work,
- branch-protection changes,
- unrelated Rust/runtime changes,
- broad workspace checks when the change has a smaller crate/risk surface.

## Agent operating rules

Every implementation PR in this rollout should:

1. start from fresh `origin/main`,
2. make one boundary change,
3. avoid runtime-code edits unless the work item explicitly names runtime code,
4. run the item validation commands,
5. fix only the first real failing cause,
6. keep expensive validation available through labels, `main`, schedule, or manual dispatch,
7. avoid changing branch protection until the dedicated PR Gate consolidation item.

## Required rollout PR body

Use this body shape for every rollout PR:

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

## Routing schema target

`xtask ci plan` should become the single routing authority. The stable JSON
artifact is `ci-plan.json` and should include this schema before workflows start
consuming it as an API:

```json
{
  "schema_version": 1,
  "budget": {
    "preferred_default_lem": 25,
    "default_limit_lem": 35,
    "estimated_lem": 0,
    "posture": "pennies|default|elevated|high|hard"
  },
  "classification": {
    "docs_only": false,
    "tracker_only": false,
    "rust_inputs_changed": false,
    "manifest_or_toolchain_changed": false,
    "public_api_changed": false,
    "gpu_changed": false,
    "macos_changed": false,
    "model_validation_changed": false,
    "coverage_requested": false,
    "full_ci_requested": false
  },
  "selected_lanes": [],
  "skipped_lanes": [],
  "packages": {
    "changed": [],
    "direct_dependents": [],
    "canaries": [],
    "selected": [],
    "broad_sweep_required": false,
    "selection_reason": "changed packages plus direct dependents and canaries"
  },
  "risk_packs": [],
  "labels": []
}
```

Until the schema lands, workflow changes must not assume fields that are not yet
emitted. After the schema lands, workflow routing should consume this artifact
instead of duplicating path logic in shell.

## Budget guard target

The planner and PR Gate should converge on these budget bands:

| Estimated LEM | Behavior |
| ---: | --- |
| `<=25` | Preferred. |
| `26-35` | Normal default. |
| `36-75` | Warning. |
| `76-100` | Strong warning. |
| `101-125` | Require `ci-budget-ack` or `full-ci` once enforcement is enabled. |
| `>125` | Fail unless `ci-budget-override` or `full-ci` once enforcement is enabled. |

The first implementation may be advisory, but the schema must carry enough
information for PR Gate to enforce this later.

## Agent-ready implementation queue

### PR 1 — `ci: remove macOS from ordinary PRs`

**Purpose:** Remove immediate 10x-runner waste from ordinary PRs while keeping
Apple proof available.

**Files:**

- `.github/workflows/macos-arm64.yml`
- `policy/ci-lanes.toml`
- `policy/ci-lane-whitelist.toml`
- `docs/ci/cost-and-verification-policy.md`
- `docs/ci/labels.md`

**Required behavior:**

- Run macOS on `push` to `main`, `workflow_dispatch`, and `merge_group` if the
  repository still needs merge-queue Apple proof.
- Run macOS PR jobs only for labels `macos`, `apple-silicon`, `metal`, or
  `full-ci`, or for Mac/Metal-specific paths.
- Remove broad ordinary-PR triggers such as all `crates/**`, `tests/**`,
  `xtask/**`, and generic `Cargo.toml` unless paired with label/manual/main
  routing.

**Acceptance:**

- Normal Rust PRs do not launch `macos-14`.
- `macos`, `apple-silicon`, `metal`, and `full-ci` labels still work.
- `main` and manual dispatch preserve Apple proof.
- Lane policy marks macOS as non-default.

**Validation:**

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
cargo run --locked -p xtask --no-default-features -- check-file-policy --report-dir target/bitnet/reports --fail-on-error
```

### PR 2 — `ci: move performance smoke off default PRs`

**Purpose:** Stop duplicating CI Core/Feature Matrix compile signal in the
Performance Baseline Tracking workflow.

**Files:**

- `.github/workflows/performance-tracking.yml`
- `policy/ci-lanes.toml`
- `policy/ci-lane-whitelist.toml`
- `docs/ci/cost-and-verification-policy.md`

**Required behavior:**

- Run performance tracking on `push` to `main`, schedule, manual dispatch, or
  labels `performance`, `perf`, and `full-ci`.
- Do not run the full workspace smoke `cargo check` on ordinary PRs.

**Acceptance:**

- Ordinary PRs do not run Performance Baseline Tracking.
- Main/schedule/manual still run.
- Labeled PRs still run.
- No branch-protection dependency is introduced.

**Validation:**

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
```

### PR 3 — `ci: move test telemetry to main and label`

**Purpose:** Move the advisory nextest/JUnit/slow-test lane out of default PRs.

**Files:**

- `.github/workflows/test-telemetry.yml`
- `policy/ci-lanes.toml`
- `policy/ci-lane-whitelist.toml`
- `docs/ci/cost-and-verification-policy.md`

**Required behavior:**

- Run test telemetry on `push` to `main`, manual dispatch, optional schedule, or
  labels `test-telemetry`, `slow-tests`, and `full-ci`.
- Set `test-telemetry` lane `default_pr = false`.

**Acceptance:**

- No ordinary PR runs Test Telemetry.
- Main/manual/labeled runs still upload JUnit and slow-test summaries.

**Validation:**

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
```

### PR 4 — `ci: risk-gate MSRV compatibility`

**Purpose:** Run MSRV for global-risk surfaces instead of every leaf Rust edit.

**Files:**

- `.github/workflows/compatibility.yml`
- `policy/ci-lanes.toml`
- `policy/ci-lane-whitelist.toml`
- `policy/ci-risk-packs.toml`
- `docs/ci/cost-and-verification-policy.md`

**Required behavior:**

- Run MSRV on manifest/toolchain/dependency changes, `.cargo/**`, public API
  surfaces, release/package surfaces, `push` to `main`, manual dispatch, or
  labels `msrv`, `compatibility`, and `full-ci`.
- Keep the existing `manifest_release` risk-pack model selecting
  `compatibility-msrv` for global dependency/toolchain risk.

**Acceptance:**

- Leaf implementation PRs do not run MSRV by default.
- Manifest/toolchain/dependency PRs still run MSRV.
- Main still runs MSRV.

**Validation:**

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
cargo check --locked -p bitnet-common -p bitnet-models -p bitnet-tokenizers -p bitnet-quantization -p bitnet-kernels --tests --no-default-features --features cpu
```

### PR 5 — `ci(plan): emit stable routing schema`

**Purpose:** Make `ci-plan.json` stable enough for workflows and agents to
consume.

**Files:**

- `xtask/src/ci/plan.rs`
- `xtask/src/ci/mod.rs`
- `policy/ci-budget.toml`
- `policy/ci-lanes.toml`
- `policy/ci-risk-packs.toml`
- `docs/ci/pr-plan.md`
- `tests/fixtures/ci-plan/**`

**Required behavior:**

- Emit the routing schema documented above.
- Preserve the existing human-readable summary.
- Fixture-test docs-only, tracker-only, ordinary Rust, manifest/toolchain, GPU,
  macOS, `full-ci`, and coverage classification.
- Do not change workflow behavior yet.

**Validation:**

```bash
cargo test -p xtask --no-default-features ci_plan --locked
cargo run --locked -p xtask --no-default-features -- ci plan --changed-file tests/fixtures/ci-plan/rust.txt --labels-json '[]' --json-out target/ci-plan.json --print
cargo run --locked -p xtask --no-default-features -- ci plan --changed-file tests/fixtures/ci-plan/docs.txt --labels-json '[]' --json-out target/ci-plan-docs.json --print
git diff --check
```

### PR 6 — `ci(gate): make PR Gate consume ci plan`

**Purpose:** Make PR Gate depend on the routing plan instead of duplicating path
classification.

**Files:**

- `.github/workflows/pr-gate.yml`
- `.github/workflows/pr-plan.yml`
- `xtask/src/ci/plan.rs`
- `docs/ci/pr-gate-success.md`

**Required behavior:**

- Determine the PR head SHA.
- Compute or download `ci-plan.json`.
- Wait only for selected blocking lanes.
- Treat a selected blocking lane that is `skipped` as a failure.
- Treat unselected skipped lanes as acceptable.
- Summarize selected lanes, skipped lanes, budget posture, and label overrides.

**Acceptance:**

- PR Gate no longer has its own path classifier.
- Path-filtered workflows no longer create missing-required-check traps.
- Branch protection can later require only `PR Gate Success`.

**Validation:**

```bash
cargo test -p xtask --no-default-features ci_plan --locked
git diff --check
```

Hosted validation should cover docs-only, ordinary Rust, and `full-ci` PRs.

### PR 7 — `ci: add soft budget guard`

**Purpose:** Turn the existing budget thresholds into visible PR feedback and a
future enforcement hook.

**Files:**

- `xtask/src/ci/plan.rs`
- `.github/workflows/pr-plan.yml`
- `.github/workflows/pr-gate.yml`
- `docs/ci/cost-and-verification-policy.md`

**Required behavior:**

- Emit budget warnings in PR Plan.
- Teach PR Gate the budget override labels.
- Keep hard failures disabled until maintainers explicitly opt in.

**Validation:**

```bash
cargo test -p xtask --no-default-features ci_plan_budget --locked
git diff --check
```

### PR 8 — `ci-core: broaden no-Rust fast path`

**Purpose:** Make docs/campaign/hardware-receipt changes avoid Rust compilation
when no Rust inputs changed.

**Files:**

- `.github/workflows/ci-core.yml`
- `xtask/src/ci/plan.rs`
- `docs/ci/cost-and-verification-policy.md`

**Required behavior:**

Replace the narrow tracker-only fast path with:

- `no_rust_inputs`,
- `docs_only`,
- `tracker_or_campaign_only`,
- `hardware_receipt_only`.

Routing should be:

| Classification | CI Core work |
| --- | --- |
| `no_rust_inputs=true` | No cargo build/test/clippy/doc. |
| `tracker_or_campaign_only=true` | Campaign doctor and generated-dashboard freshness. |
| `docs_only=true` | Docs/markdown/link checks elsewhere; CI Core emits success no-op. |
| `hardware_receipt_only=true` | Schema/receipt checks only. |

**Validation:**

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci plan --changed-file tests/fixtures/ci-plan/docs.txt --labels-json '[]' --json-out target/ci-plan-docs.json --print
```

### PR 9 — `ci-core: package-select build/test surface`

**Purpose:** Scope CI Core to changed packages, direct dependents, and canaries
unless a broad sweep is required.

**Files:**

- `xtask/src/ci/plan.rs`
- `.github/workflows/ci-core.yml`
- `docs/ci/cost-and-verification-policy.md`

**Required behavior:**

- Compute changed packages, direct dependents, canary packages, selected
  packages, and `broad_sweep_required`.
- Run CI Core build/test on the selected package set when a broad sweep is not
  required.
- Require broad sweep for manifest/toolchain/shared-foundation changes.
- Summarize selected packages and reason.

**Canaries:**

| Trigger | Canary |
| --- | --- |
| quantization / QK256 | scalar/oracle + AVX2 smoke |
| kernels | kernel CPU + AVX2 smoke |
| tokenizer | tokenizer fixtures |
| inference/model | CPU golden path |
| policy/BDD | BDD/policy unit tests |
| manifests/toolchain/common crate | broad sweep |

**Validation:**

```bash
cargo test -p xtask --no-default-features ci_plan_packages --locked
cargo run --locked -p xtask --no-default-features -- ci plan --changed-file tests/fixtures/ci-plan/quantization.txt --labels-json '[]' --json-out target/ci-plan-quant.json --print
git diff --check
```

### PR 10 — `feature-matrix: reduce ordinary PR feature smoke`

**Purpose:** Make `cpu+full-cli` risk-selected instead of universal.

**Files:**

- `.github/workflows/feature-matrix.yml`
- `.github/workflows/pr-gate.yml`
- `policy/ci-lane-whitelist.toml`
- `policy/ci-risk-packs.toml`
- `policy/ci-lanes.toml`
- `docs/ci/cost-and-verification-policy.md`

**Required behavior:**

- Default ordinary PR matrix: `no-features` and `cpu`.
- Run targeted `cpu+full-cli` for CLI/server/validation/model-cache/full-cli
  feature files, manifest/lock/toolchain changes, or label `full-cli`.
- Treat `feature-matrix` and `full-ci` as full-matrix opt-ins; that full
  matrix still includes `cpu+full-cli`.
- Keep full matrix available on `main` and `full-ci`.

**Validation:**

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
```

### PR 11 — `ci: stop selected gates from swallowing failures`

**Purpose:** Ensure selected blocking lanes fail honestly.

**Files:**

- `.github/workflows/validation.yml`
- `.github/workflows/test-framework.yml`
- `.github/workflows/compatibility.yml`
- `docs/ci/cost-and-verification-policy.md`
- `policy/ci-lanes.toml`

**Required behavior:**

- A lane selected as blocking must not use `continue-on-error` or `|| true` for
  its proof obligation.
- Flaky or expensive lanes that cannot fail honestly must move to
  main/nightly/manual/label and be marked advisory.
- Policy tables must distinguish blocking from advisory.

**Validation:**

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
```

### PR 12 — `gpu-ci: remove duplicate CPU native check`

**Purpose:** Stop GPU CI from duplicating the ordinary CPU feature compile lane.

**Files:**

- `.github/workflows/gpu-ci-matrix.yml`
- `policy/ci-lanes.toml`
- `policy/ci-risk-packs.toml`

**Required behavior:**

- Remove `native-check(cpu)` from GPU CI.
- Trigger GPU CI only on GPU paths or labels `gpu-ci` and `full-ci`.
- Avoid generic manifest triggers unless GPU dependencies changed or `full-ci`
  is present.
- Keep Docker main/manual/label only.

**Validation:**

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
```

### PR 13 — `coverage: keep Codecov lane quiet and policy-declared`

**Purpose:** Keep coverage as measured evidence, not default PR CI.

**Files:**

- `.github/workflows/coverage.yml`
- `codecov.yml`
- `README.md`
- `docs/ci/coverage.md`
- `policy/ci-lanes.toml`
- `policy/ci-lane-whitelist.toml`

**Required behavior:**

- Keep coverage out of CI Core.
- Run PR coverage only for labels `coverage` and `full-ci`.
- Add a README badge if missing.
- Set Codecov `comment: false`.
- Keep `github_checks.annotations: false`.
- Register the coverage lane cost and claim boundary.
- Keep flag `rust-cpu`.

**Validation:**

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
```

### PR 14 — `coverage: add coverage receipt and ignored-failure accounting`

**Purpose:** Make ignored test failures visible in coverage artifacts.

**Files:**

- `.github/workflows/coverage.yml`
- `docs/ci/coverage.md`
- target receipt generation script or inline workflow step

**Required behavior:**

- Upload a coverage receipt artifact with schema version, lane, flag, workflow,
  artifact presence, claim boundaries, and `ignored_run_failures`.
- Fail `main` if ignored failures are nonzero unless an explicit override is in
  scope.
- Allow advisory PR coverage to upload the receipt even when non-blocking.

**Receipt target:**

```json
{
  "schema_version": 1,
  "repo": "bitnet-rs",
  "lane": "coverage",
  "flag": "rust-cpu",
  "workflow": "Coverage",
  "artifacts": {
    "coverage_json": true,
    "coverage_text": true,
    "lcov": true
  },
  "claim_boundary": [
    "execution_surface_only",
    "cpu_path_only",
    "not_gpu_validation",
    "not_model_quality",
    "not_crossval",
    "not_mutation_adequacy"
  ],
  "ignored_run_failures": 0
}
```

### PR 15 — `policy(clippy): document strict agent lint baseline`

**Purpose:** Define strict lint/no-panic policy tiers without activating new
lint debt cleanup yet.

**Files:**

- `docs/ci/clippy-policy.md`
- `policy/clippy-lints.toml`
- `clippy.toml`
- `Cargo.toml`
- `docs/ci/cost-and-verification-policy.md`

**Required behavior:**

Document these policy tiers:

| Tier | Purpose |
| --- | --- |
| Active | Enforced now. |
| Staged | Measured but not enforced. |
| Planned 1.94/1.95 | Flip during toolchain upgrade. |
| Debt | Owned, reasoned, expiring. |
| Suppression | Must use reasoned `expect`, not silent `allow`. |

**Acceptance:**

- No code cleanup.
- No new lint activation.
- Stable policy doc and TOML map for agents.

**Validation:**

```bash
cargo run --locked -p xtask --no-default-features -- check-lint-policy --report-dir target/bitnet/reports --fail-on-error
git diff --check
```

### PR 16+ — lint activation slices

Activate one lint family per PR, in this order:

1. suppression governance: `allow_attributes`, `allow_attributes_without_reason`,
2. panic family: no new `unwrap`, `expect`, `panic`, `unreachable`, unsafe
   slicing/indexing,
3. silent failure: `let_underscore_future`, `let_underscore_must_use`,
   `unused_result_ok`, `map_err_ignore`,
4. async/concurrency: `await_holding_lock`, `await_holding_refcell_ref`,
5. numeric: `cast_sign_loss`, `invalid_upcast_comparisons`, staged
   truncation/precision lints.

Each activation PR should include only the policy/lint changes and the minimum
local cleanup needed to pass.

### PR 20 — `ci: make PR Gate the only required check`

**Purpose:** Move branch protection to the single routed aggregator after the
routing plan has proven stable.

**Files:**

- `.github/workflows/pr-gate.yml`
- `docs/ci/pr-gate-success.md`
- `docs/ci/branch-protection.md`

**Required behavior:**

- Keep leaf checks as normal workflow checks.
- Branch protection requires only `PR Gate Success`.
- PR Gate enforces selected blocking lanes.
- Document that this may require manual GitHub settings outside code.

**Acceptance:**

- Docs explain branch-protection migration.
- Existing `CI Core Success` may remain for compatibility but is not required.
- Path-filtered skipped workflows no longer block.

## Expected cost outcome

After the first four immediate cost wins and the routing work, expected default
costs are:

| PR type | Expected lanes | Target LEM |
| --- | --- | ---: |
| Ordinary Rust | PR Plan, PR Gate, scoped CI Core, feature smoke, policy, ripr | 25-34 |
| Docs/tracking | PR Plan, PR Gate, docs/campaign/receipt checks | 3-8 |
| Manifest/toolchain | CI Core broad, feature smoke, MSRV, policy, optional label lanes | 35-50+ |

Manifest/toolchain PRs are intentionally allowed to exceed ordinary PR cost
because they change global compatibility and dependency risk.

## Implementation notes

- Update `policy/ci-routed-rollout.toml` when this rollout map changes so
  agents and scripts have a compact machine-readable queue.
- Keep `policy/ci-lane-whitelist.toml` as the lane-level source of truth and
  mirror compact costs into `policy/ci-lanes.toml` when a lane changes.
- Keep labels documented in `docs/ci/labels.md` whenever a workflow introduces
  or removes a label route.
