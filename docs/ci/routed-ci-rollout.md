# Routed CI Rollout Specification

This document is the implementation map for making sub-`$0.50` ordinary PR CI
the default in BitNet-rs. It turns the existing CI budget vocabulary into an
agent-executable sequence of small PRs.

## North star

Default PRs get cheap, Linux-only, deterministic, crate/risk-scoped proof.
Expensive proof still exists, but only on `main`, schedule, release,
hardware/campaign lanes, or explicit labels.

The rollout uses Linux-equivalent minutes (LEM) as the shared planning unit:

| Budget term | Value |
| --- | ---: |
| Preferred default PR target | 25 LEM |
| Normal default PR limit | 35 LEM |
| Windows multiplier | 2x Linux |
| macOS multiplier | 10x Linux |
| GPU Docker multiplier | 6x Linux |

Budget override labels are `full-ci`, `ci-budget-override`, and
`ci-budget-ack`. They authorize extra spend; they do not make failures
optional.

## Operating rules for every rollout PR

Every rollout PR must be narrow and must include the following PR body sections:

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

Agent behavior for each item:

1. Start from fresh `origin/main`.
2. Make one boundary change per PR.
3. Avoid runtime-code edits unless the item explicitly scopes runtime code.
4. Run targeted local validation.
5. Fix only the first real failing cause.
6. Commit and open/update the PR with the sections above.
7. Merge only when required checks are green and GitHub reports the PR mergeable.
8. After merge, fast-forward local `main` and re-check the queue state.

## Rollout phases

| Phase | Objective | Items |
| --- | --- | --- |
| 1 | Remove immediate default-cost waste | 1-4 |
| 2 | Make routing authoritative | 5-7 |
| 3 | Narrow default Linux proof | 8-10 |
| 4 | Tighten advisory lanes into honest lanes | 11-12 |
| 5 | Treat coverage as measured evidence | 13-14 |
| 6 | Increase cheap verification density with policy | 15-16+ |
| 7 | Consolidate branch protection after routing is proven | 20 |

## Agent-ready queue

### 1. `ci: remove macOS from ordinary PRs`

**Scope:** `.github/workflows/macos-arm64.yml`, `policy/ci-lanes.toml`,
`policy/ci-lane-whitelist.toml`, `docs/ci/cost-and-verification-policy.md`,
`docs/ci/labels.md`.

**Change:** Remove broad ordinary-PR macOS triggers. macOS may run on `push` to
`main`, `workflow_dispatch`, `merge_group` if still required, labels `macos`,
`apple-silicon`, `metal`, `full-ci`, and Mac/Metal-specific paths.

**Acceptance:** Normal Rust PRs do not launch `macos-14`; Apple proof remains
available by label/main/manual; policy tables mark macOS as non-default.

**Validation:**

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
cargo run --locked -p xtask --no-default-features -- check-file-policy --report-dir target/bitnet/reports --fail-on-error
```

### 2. `ci: move performance smoke off default PRs`

**Scope:** `.github/workflows/performance-tracking.yml`,
`policy/ci-lanes.toml`, `policy/ci-lane-whitelist.toml`,
`docs/ci/cost-and-verification-policy.md`.

**Change:** Run performance tracking only on `push` to `main`, schedule,
`workflow_dispatch`, or labels `performance`, `perf`, `full-ci`. Do not run the
workspace smoke check on ordinary PRs.

**Acceptance:** Ordinary PRs do not run `Performance Baseline Tracking`; main,
schedule, manual, and labeled PR runs still work; branch protection is
unchanged.

**Validation:**

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
```

### 3. `ci: move test telemetry to main and label`

**Scope:** `.github/workflows/test-telemetry.yml`, `policy/ci-lanes.toml`,
`policy/ci-lane-whitelist.toml`, `docs/ci/cost-and-verification-policy.md`.

**Change:** Run telemetry only on `push` to `main`, `workflow_dispatch`,
optional schedule, or labels `test-telemetry`, `slow-tests`, `full-ci`.

**Acceptance:** Ordinary PRs do not run Test Telemetry; main/manual/labeled runs
still produce JUnit and slow-test summaries; lane tables set
`default_pr = false`.

**Validation:**

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
```

### 4. `ci: risk-gate MSRV compatibility`

**Scope:** `.github/workflows/compatibility.yml`, `policy/ci-lanes.toml`,
`policy/ci-lane-whitelist.toml`, `policy/ci-risk-packs.toml`,
`docs/ci/cost-and-verification-policy.md`.

**Change:** Run MSRV for manifest/toolchain/dependency risk, public API or
release/package surfaces, `push` to `main`, `workflow_dispatch`, or labels
`msrv`, `compatibility`, `full-ci`. Preserve the `manifest_release` risk pack as
the selector for global dependency/toolchain risk.

**Acceptance:** Leaf implementation PRs do not run MSRV by default; manifest,
toolchain, dependency, public API, release, and main changes still run MSRV.

**Validation:**

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
cargo check --locked -p bitnet-common -p bitnet-models -p bitnet-tokenizers -p bitnet-quantization -p bitnet-kernels --tests --no-default-features --features cpu
```

### 5. `ci(plan): emit stable routing schema`

**Scope:** `xtask/src/ci/plan.rs`, `xtask/src/ci/mod.rs`,
`policy/ci-budget.toml`, `policy/ci-lanes.toml`, `policy/ci-risk-packs.toml`,
`docs/ci/pr-plan.md`, `tests/fixtures/ci-plan/**`.

**Change:** Make `ci-plan.json` conform to the schema in
[`docs/ci/pr-plan.md`](./pr-plan.md) while preserving the existing human
summary. Add fixtures for docs-only, tracker-only, ordinary Rust,
manifest/toolchain, GPU, macOS, `full-ci`, and coverage classifications.

**Acceptance:** The schema is fixture-tested; no workflow behavior changes.

**Validation:**

```bash
cargo test -p xtask --no-default-features ci_plan --locked
cargo run --locked -p xtask --no-default-features -- ci plan --changed-file tests/fixtures/ci-plan/rust.txt --labels-json '[]' --json-out target/ci-plan.json --print
cargo run --locked -p xtask --no-default-features -- ci plan --changed-file tests/fixtures/ci-plan/docs.txt --labels-json '[]' --json-out target/ci-plan-docs.json --print
git diff --check
```

### 6. `ci(gate): make PR Gate consume ci plan`

**Scope:** `.github/workflows/pr-gate.yml`, `.github/workflows/pr-plan.yml`,
`xtask/src/ci/plan.rs`, `docs/ci/pr-gate-success.md`.

**Change:** PR Gate determines the PR head SHA, computes or downloads
`ci-plan.json`, waits only for selected blocking lanes, treats selected skipped
blocking lanes as failures, and treats unselected skipped lanes as OK.

**Acceptance:** PR Gate no longer has its own path classifier; path-filtered
workflows no longer create missing-required-check traps; branch protection can
later require only `PR Gate Success`.

**Validation:**

```bash
cargo test -p xtask --no-default-features ci_plan --locked
git diff --check
```

### 7. `ci: add soft budget guard`

**Scope:** `xtask/src/ci/plan.rs`, `.github/workflows/pr-plan.yml`,
`.github/workflows/pr-gate.yml`, `docs/ci/cost-and-verification-policy.md`.

**Change:** Emit budget posture and advisory guard state from the planner.
Support these behaviors in schema and summaries: `<=25` preferred, `26-35`
normal default, `36-75` warning, `76-100` strong warning, `101-125` requires
`ci-budget-ack` or `full-ci`, and `>125` fails unless `ci-budget-override` or
`full-ci` is present. Start advisory unless enforcement is explicitly enabled.

**Acceptance:** Budget warnings appear in PR Plan; PR Gate understands budget
override labels; maintainers do not get accidental new hard failures.

**Validation:**

```bash
cargo test -p xtask --no-default-features ci_plan_budget --locked
git diff --check
```

### 8. `ci-core: broaden no-Rust fast path`

**Scope:** `.github/workflows/ci-core.yml`, `xtask/src/ci/plan.rs`,
`docs/ci/cost-and-verification-policy.md`.

**Change:** Replace the narrow `tracker_only` fast path with
`no_rust_inputs`, `docs_only`, `tracker_or_campaign_only`, and
`hardware_receipt_only` classifications. No-Rust inputs should avoid cargo
build/test/clippy/doc while still emitting `CI Core Success`.

**Acceptance:** Docs, campaign, and hardware receipt PRs do not compile Rust
unless they touch Rust inputs; Rust PR behavior is unchanged.

**Validation:**

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci plan --changed-file tests/fixtures/ci-plan/docs.txt --labels-json '[]' --json-out target/ci-plan-docs.json --print
```

### 9. `ci-core: package-select build/test surface`

**Scope:** `xtask/src/ci/plan.rs`, `.github/workflows/ci-core.yml`,
`docs/ci/cost-and-verification-policy.md`.

**Change:** Have `xtask ci plan` compute changed packages, direct dependents,
canaries, and `broad_sweep_required`. CI Core should test/clippy selected
packages and canaries by default, while manifests/toolchain/foundational crates
still trigger a broad sweep.

**Acceptance:** Leaf crate PRs get scoped tests; global-risk PRs get broad
coverage; summaries explain selected packages and reasons; no macOS, Windows,
Docker, model, or download work is added.

**Validation:**

```bash
cargo test -p xtask --no-default-features ci_plan_packages --locked
cargo run --locked -p xtask --no-default-features -- ci plan --changed-file tests/fixtures/ci-plan/quantization.txt --labels-json '[]' --json-out target/ci-plan-quant.json --print
git diff --check
```

### 10. `feature-matrix: reduce ordinary PR feature smoke`

**Scope:** `.github/workflows/feature-matrix.yml`, `policy/ci-risk-packs.toml`,
`policy/ci-lanes.toml`, `docs/ci/cost-and-verification-policy.md`.

**Change:** Ordinary PRs run `no-features` and `cpu`. Run `cpu+full-cli` for
CLI/server/validation/model-cache/full-cli feature risk, manifest/toolchain
changes, or labels `full-cli`, `feature-matrix`, `full-ci`.

**Acceptance:** Ordinary feature smoke is cheaper; CLI/full-cli risk remains
checked; `main` and `full-ci` still run the full matrix.

**Validation:**

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
```

### 11. `ci: stop selected gates from swallowing failures`

**Scope:** `.github/workflows/validation.yml`, `.github/workflows/test-framework.yml`,
`.github/workflows/compatibility.yml`, `docs/ci/cost-and-verification-policy.md`,
`policy/ci-lanes.toml`.

**Change:** If a lane is selected as blocking, failures fail. Lanes too flaky or
expensive to fail must move to main/nightly/manual/label and be marked
advisory.

**Acceptance:** No selected blocking lane uses `|| true`; advisory lanes are
clearly named and not selected by PR Gate as blocking; policy reflects blocking
versus advisory status.

**Validation:**

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
```

### 12. `gpu-ci: remove duplicate CPU native check`

**Scope:** `.github/workflows/gpu-ci-matrix.yml`, `policy/ci-lanes.toml`,
`policy/ci-risk-packs.toml`.

**Change:** Remove `native-check(cpu)` from GPU CI. Trigger GPU CI only on GPU
paths or labels `gpu-ci` / `full-ci`; avoid generic manifest triggers unless
GPU dependencies changed or `full-ci` is present.

**Acceptance:** GPU path PRs still run GPU feature compile checks; ordinary
manifest PRs do not fan out GPU CI; Docker remains main/manual/label only.

**Validation:**

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
```

### 13. `coverage: keep Codecov lane quiet and policy-declared`

**Scope:** `.github/workflows/coverage.yml`, `codecov.yml`, `README.md`,
`docs/ci/coverage.md`, `policy/ci-lanes.toml`,
`policy/ci-lane-whitelist.toml`.

**Change:** Keep coverage out of CI Core. PR coverage runs only for `coverage`
or `full-ci`; main gets coverage evidence. Keep Codecov comments and annotation
spam disabled, keep flag `rust-cpu`, and document the CPU execution-surface
claim boundary.

**Acceptance:** Ordinary PR cost is unchanged; labeled PRs can request coverage;
coverage makes no GPU/model/hardware adequacy claim.

**Validation:**

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
```

### 14. `coverage: add coverage receipt and ignored-failure accounting`

**Scope:** `.github/workflows/coverage.yml`, `docs/ci/coverage.md`, receipt
generation in the workflow.

**Change:** Upload a coverage receipt with schema version, repo, lane, flag,
workflow, artifact booleans, claim boundary, and `ignored_run_failures`. If
`--ignore-run-fail` remains, main fails when ignored failures are nonzero unless
an explicit future allow mechanism is scoped in that PR.

**Acceptance:** Coverage artifacts include the receipt; nonzero ignored
failures are visible and do not masquerade as green proof.

### 15. `policy(clippy): document strict agent lint baseline`

**Scope:** `docs/ci/clippy-policy.md`, `policy/clippy-lints.toml`,
`clippy.toml`, `Cargo.toml`, `docs/ci/cost-and-verification-policy.md`.

**Change:** Document lint tiers: Active, Staged, Planned 1.94/1.95, Debt, and
Suppression. Do not activate new lints and do not perform code cleanup in this
PR.

**Acceptance:** Agents have a stable policy doc and TOML map; behavior is
unchanged.

**Validation:**

```bash
cargo run --locked -p xtask --no-default-features -- check-lint-policy --report-dir target/bitnet/reports --fail-on-error
git diff --check
```

### 16+. lint activation slices

Activate one lint family per PR, in this order: suppression governance, panic
family, silent failure, async/concurrency, then numeric correctness. Each slice
must include only the minimum code cleanup required for that lint family.

### 20. `ci: make PR Gate the only required check`

**Scope:** `.github/workflows/pr-gate.yml`, `docs/ci/pr-gate-success.md`,
`docs/ci/branch-protection.md`.

**Change:** After routing is proven, document and perform the branch-protection
migration so only `PR Gate Success` is required. Leaf checks continue to run as
normal workflow checks and PR Gate enforces selected blocking lanes.

**Acceptance:** Docs explain the migration; `CI Core Success` may remain for
compatibility but does not need to be required; path-filtered skipped workflows
no longer block.

## Expected cost outcomes

| PR type | Target LEM |
| --- | ---: |
| Ordinary Rust PR | 25-34 |
| Docs / tracking PR | 3-8 |
| Manifest / toolchain PR | 35-50+ |

Manifest and toolchain changes are allowed to exceed ordinary PR targets
because they create global dependency/toolchain risk.
