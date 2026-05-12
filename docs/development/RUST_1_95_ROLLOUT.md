# BitNet-rs Rust 1.95 / Next Minor CI Economics Continuation

This document is the control map for the Rust 1.95 / next minor quality wave.
It continues the Rust 1.93 CI economics control plane from #3866. It is not a
new rollout template and it does not reopen already-landed policy machinery.

Reading this document first prevents agents from:

- redoing #3866 control-plane work,
- mixing MSRV bump, lint activation, no-panic baseline, release bump, and API
  cleanup in one PR,
- broadening ordinary PR CI under pressure,
- treating `ripr` and mutation testing as unrelated proof families,
- hiding policy debt inside unrelated cleanup.

## Corrected Doctrine

Use this doctrine in docs, PR bodies, and Claude-facing instructions:

```text
ripr is static mutation-exposure analysis.

It catches much of the same signal mutation testing catches -- weak test/oracle
exposure -- but earlier and cheaper, because it runs statically and can run
per PR.

Mutation testing remains the runtime empirical backstop, especially for
nightly and release readiness. The CI design should use ripr to shift
mutation signal left, not to pretend mutation is unnecessary.
```

For BitNet-rs, that maps to this evidence stack:

```text
Default PR:
  ripr + normal gates + policy checks

Risk PR:
  ripr + targeted mutation for touched high-risk owner surfaces

Nightly:
  broader mutation matrix

Release:
  mutation/readiness clean enough to ship
```

The point is CI economics, not weaker proof. At industrialized PR volume,
verification cost can dominate LLM cost. Small per-PR lane choices compound, so
the architecture must make high-signal checks cheap and frequent while routing
expensive runtime confirmation only where it buys signal.

## Relationship To #3866

PR #3866 established the Rust 1.93 CI economics control plane:

- MSRV 1.93.0,
- CI lane whitelist and LEM-aware budget bands,
- policy workflow for lane, file-policy, lint-inheritance, Clippy exception,
  no-panic, and policy-report checks,
- PR Gate observation mode,
- `ripr` advisory workflow surface,
- lint inheritance checker,
- strict-policy ledgers for Clippy, no-panic, and non-Rust file allowlists.

The 1.93 rollout explicitly deferred:

- strict Clippy ratchets and receipt discipline,
- removal of the test carveouts that were still in `clippy.toml` at the time,
- no-panic identity hardening and no-new-debt mode,
- real `ripr` advisory execution when the binary is absent,
- LEM/risk-pack tightening to reduce default per-PR mutation and over-testing,
- branch-protection migration to the `PR Gate Success` aggregator,
- learned-estimate planner switch after enough actuals are collected.

The Rust 1.95 wave addresses those deferred items in a one-objective-per-PR
ladder.

## Current Vs Target State

| Layer | Current state on `main` | Target state | PR |
|---|---|---|---|
| Edition | Rust 2024 | Rust 2024 | current |
| Workspace MSRV | `1.95.0` in `Cargo.toml` | `1.95.0` | 3 |
| Toolchain | `1.95.0` in `rust-toolchain.toml` | `1.95.0` with rustfmt, clippy, rust-analyzer | 3 |
| Root version | `0.2.1-dev` | next minor dev line | 16 |
| Clippy MSRV | `msrv = "1.95.0"` | `msrv = "1.95.0"` | 3 |
| Clippy test carveouts | removed in PR 6 | no test carveouts | 6 |
| Clippy 1.94/1.95 lints | `policy/clippy-lints.toml` now records `msrv = "1.95"`; planned lints are explicitly staged at `allow` in `Cargo.toml` | active or explicitly deferred with debt | 5 |
| No-panic allowlist | present, empty/advisory, identity is `path + family + selector` | exact counted identity | 7 |
| No-panic baseline | absent | generated no-new-debt baseline | 8 |
| Non-Rust allowlist | present, broad | narrowed with explicit covered-by evidence | 10 |
| CI lane whitelist / LEM | present | calibrated for Rust 1.95 and risk-pack routing | 15 |
| Core CI toolchain | workflow toolchain pins use `1.95.0`; coverage image is `rust-1.95` | workflows and the Rust CI image stay on the declared floor | 3 |
| `ripr` | workflow exists, may skip when binary is absent | real advisory static mutation-exposure signal | 11 |
| Mutation testing | expensive runtime evidence outside default PR | targeted risk PR, broader nightly, release readiness | 11, 15, 17 |

## Version Note

The package line is `0.2.1-dev`, but the changelog has historical `0.3.0`
content. The release-prep PR must reconcile whether the next minor dev line is
`0.3.0-dev` or `0.4.0-dev`. The semver rule is fixed: MSRV increase ships as a
minor release.

## Rust 1.95 Value For BitNet-rs

| Rust 1.95 item | BitNet-rs use |
|---|---|
| `if let` guards | CUDA/CPU dispatch planner, backend route classification, GGUF metadata mapping, tokenizer compatibility, API route matching |
| `Vec::push_mut` / `insert_mut` | benchmark receipts, campaign dashboards, policy reports, generation events, OpenAI-compatible response builders |
| atomic `update` / `try_update` | runtime counters, once-warn counters, request/health/state counters |
| `cfg_select!` | GPU backend selection, SIMD path selection, WASM/native surfaces, Python/FFI conditionals, Windows/Linux path handling |
| `cold_path` | error/report/rejection paths in GGUF validation, download/auth, API routing, and policy failures |
| Clippy 1.95 | `manual_checked_ops`, `manual_take`, `manual_pop_if`, `duration_suboptimal_units`, `needless_type_cast`, `unnecessary_trailing_comma` |

## PR Ladder

Each PR below is a single objective. Start every PR from clean `origin/main`.

| PR | Branch | Title | Scope |
|---:|---|---|---|
| 1 | `docs/rust-1.95-rollout-refresh` | `docs(policy): refresh Rust 1.95 and next-minor rollout map` | Documentation refresh/correction only. No Cargo, workflow, toolchain, or Rust source changes. |
| 2 | `probe/rust-1.95-compat` | `chore(msrv): probe Rust 1.95 compatibility` | Run current `main` under Rust 1.95 before changing declared MSRV. Audit note preferred. |
| 3 | `chore/msrv-rust-1.95` | `chore(msrv): raise workspace toolchain to Rust 1.95` | Bump workspace and explicit member MSRV pins, `rust-toolchain.toml`, `clippy.toml`, workflow toolchain pins, the Rust CI image tag, `policy/clippy-lints.toml` MSRV, and explicit lint-ratchet deferrals. |
| 4 | `policy/rust-1.95-lints` | `policy(rust): enable Rust 1.95 compiler lint floor` | Move compiler lint floor forward. Do not use `unsafe_code = "forbid"` globally. |
| 5 | `policy/clippy-rust-1.95-ratchets` | `policy(clippy): activate Rust 1.95 lint ratchets` | Measure first, then activate clean or cheaply fixed 1.94/1.95 Clippy ratchets. Keep `disallowed_fields` planned until real fields are configured. |
| 6 | `policy/no-test-clippy-carveouts` | `policy(clippy): remove test unwrap and expect carveouts` | Remove Clippy test carveouts and convert one narrow helper slice only. |
| 7 | `policy/no-panic-exact-identity` | `policy(panic): harden no-panic allowlist identity` | Add exact counted identity: `path + family + selector_kind + selector_callee + snippet + count`. |
| 8 | `policy/no-panic-baseline` | `policy(panic): add no-panic baseline and no-new-debt gate` | Generate baseline, set no-new-debt mode, mark generated file in `.gitattributes`. |
| 9 | `policy/no-panic-diagnostics` | `policy(panic): improve no-panic report diagnostics` | Missing-baseline error, stale baseline entries, delta details, blocking-mode messaging. |
| 10 | `policy/file-allowlist-tightening` | `policy(files): tighten non-Rust allowlist coverage` | Remove stale entries, narrow broad globs, add review/expiry metadata where supported, keep GPU/FFI/Python/WASM explicit. |
| 11 | `ci/ripr-real-advisory` | `ci(ripr): provision real static mutation-exposure analysis` | Install and run `ripr`, emit JSON/SARIF/Markdown, keep advisory and not branch-protection blocking. |
| 12 | `refactor/rust-1.95-api-cleanups` | `refactor: use Rust 1.95 APIs in dispatch and receipt builders` | Targeted use of Rust 1.95 APIs after MSRV bump lands. |
| 13 | `policy/clippy-numeric-kernel-cleanup` | `policy(clippy): clean numeric and kernel lint debt` | Clean numeric/kernel lint debt without changing semantics. |
| 14 | `policy/no-panic-first-burndown` | `policy(panic): burn down first no-panic owner lane` | One narrow owner lane only. Baseline refresh may only drop disappeared entries. |
| 15 | `ci/bitnet-lem-lane-tightening` | `ci: tighten lane whitelist and LEM routing for Rust 1.95` | Reclassify `ripr-advisory` as real static mutation-exposure signal and route GPU/FFI/platform lanes by risk/label/main. |
| 16 | `release/next-minor-prep-rust-1.95` | `release: prepare next minor release for Rust 1.95` | Reconcile next minor version, update manifests/docs/changelog/release prep. |
| 17 | `release/next-minor-dry-run` | `release: validate next minor publish readiness` | Publish dry-run and readiness document for the chosen version. |

## Acceptance Gates

### PR 1: Documentation Refresh

```bash
cargo run --locked -p xtask --no-default-features -- check-file-policy --report-dir target/bitnet/reports
cargo run --locked -p xtask --no-default-features -- policy-report --report-dir target/bitnet/reports
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
cargo run --locked -p xtask --no-default-features -- check-lint-inheritance
cargo run --locked -p xtask --no-default-features -- check-clippy-exceptions
cargo fmt --all -- --check
git diff --check
```

### PR 2: Rust 1.95 Compatibility Spike

```bash
rustup toolchain install 1.95.0 --component rustfmt --component clippy --component rust-analyzer
cargo +1.95.0 fmt --all -- --check
cargo +1.95.0 check --locked --workspace --all-targets --no-default-features
cargo +1.95.0 check --locked --workspace --all-targets --features cpu
cargo +1.95.0 clippy --locked --workspace --all-targets --no-default-features -- -D warnings
cargo +1.95.0 clippy --locked --workspace --all-targets --features cpu -- -D warnings
cargo +1.95.0 run --locked -p xtask --no-default-features -- policy-report --report-dir target/bitnet/reports
git diff --check
```

Merge rule: no MSRV bump, no lint activation, no release bump, and no Rust
1.95 API cleanup.

### PR 3: MSRV/Toolchain Bump

```bash
cargo fmt --all -- --check
cargo check --locked --workspace --all-targets --features cpu
cargo run --locked -p xtask --no-default-features -- check-lint-inheritance
cargo run --locked -p xtask --no-default-features -- check-clippy-exceptions
cargo run --locked -p xtask --no-default-features -- policy-report --report-dir target/bitnet/reports
git diff --check
```

Do not remove test carveouts here. That belongs in PR 6.

### PR 4: Rust Compiler Lint Floor

```bash
cargo check --locked --workspace --all-targets --features cpu
cargo run --locked -p xtask --no-default-features -- check-lint-inheritance
cargo run --locked -p xtask --no-default-features -- check-clippy-exceptions
cargo run --locked -p xtask --no-default-features -- policy-report --report-dir target/bitnet/reports
```

### PR 5: Clippy 1.95 Ratchets

Measure first:

```bash
cargo clippy --locked --workspace --all-targets --features cpu -- \
  -W clippy::same_length_and_capacity \
  -W clippy::manual_ilog2 \
  -W clippy::decimal_bitwise_operands \
  -W clippy::needless_type_cast \
  -W clippy::manual_checked_ops \
  -W clippy::manual_take \
  -W clippy::manual_pop_if \
  -W clippy::duration_suboptimal_units \
  -W clippy::unnecessary_trailing_comma
```

Then:

```bash
cargo run --locked -p xtask --no-default-features -- check-lint-policy --report-dir target/bitnet/reports
cargo run --locked -p xtask --no-default-features -- check-clippy-exceptions --report-dir target/bitnet/reports
cargo clippy --locked --workspace --all-targets --features cpu
```

### PR 6: Remove Clippy Test Carveouts

Remove only:

```toml
allow-expect-in-tests = true
allow-unwrap-in-tests = true
```

Add or extend fallible helpers in the correct test-support crate and convert one
narrow helper slice only. Do not migrate the whole test suite here.

PR 6 uses the existing `bitnet-test-support::assertions` helpers and converts
that module's own unit tests to fallible `anyhow::Result<()>` tests. Broader
test-suite panic-family cleanup stays in the no-panic identity, baseline, and
owner-lane burndown PRs.

### PR 7: No-Panic Exact Identity

```bash
cargo test -p xtask no_panic --locked
cargo run --locked -p xtask --no-default-features -- check-no-panic-family --report-dir target/bitnet/reports
```

Required test names:

```text
allowlist_entry_requires_exact_snippet
allowlist_count_is_consumed_per_occurrence
allowlist_does_not_cover_same_file_same_callee_different_snippet
baseline_generation_subtracts_allowlisted_counts
blocking_mode_ignores_baseline_but_honors_counted_allowlist
duplicate_allowlist_keys_are_rejected
```

### PR 8: No-Panic Baseline

```bash
cargo run --locked -p xtask --no-default-features -- no-panic baseline --reset
cargo run --locked -p xtask --no-default-features -- check-no-panic-family --report-dir target/bitnet/reports
cargo test -p xtask no_panic --locked
git diff --check
```

Baseline refresh may drop disappeared entries. It must refuse to absorb new
debt unless `--reset` is explicit.

### PR 9: No-Panic Diagnostics

```bash
cargo test -p xtask no_panic --locked
cargo run --locked -p xtask --no-default-features -- check-no-panic-family --report-dir target/bitnet/reports
```

### PR 10: File Allowlist Tightening

```bash
cargo run --locked -p xtask --no-default-features -- check-file-policy --report-dir target/bitnet/reports
cargo run --locked -p xtask --no-default-features -- policy-report --report-dir target/bitnet/reports
```

### PR 11: Real Advisory `ripr`

Recommended workflow shape:

```yaml
- name: Install ripr
  run: cargo install ripr --locked

- name: Run ripr doctor
  run: ripr doctor || true

- name: Run ripr check
  run: |
    mkdir -p target/ripr
    ripr check \
      --base "origin/${{ github.base_ref }}" \
      --json target/ripr/ripr.json \
      --sarif target/ripr/ripr.sarif \
      --markdown target/ripr/ripr.md \
      --config ripr.toml || true
```

Use synchronize-only cancellation:

```yaml
cancel-in-progress: ${{ github.event_name == 'pull_request' && github.event.action == 'synchronize' }}
```

Keep advisory. Do not add branch protection.

### PR 12: Rust 1.95 API Cleanup

```bash
cargo test --locked -p bitnet-kernels --lib --no-default-features --features cpu
cargo test --locked -p bitnet-receipts --all-features
cargo test --locked -p xtask
cargo clippy --locked --workspace --all-targets --features cpu
```

### PR 13: Numeric And Kernel Lint Cleanup

```bash
cargo run --locked -p xtask --no-default-features -- check-clippy-exceptions --report-dir target/bitnet/reports
cargo run --locked -p xtask --no-default-features -- check-lint-policy --report-dir target/bitnet/reports
cargo clippy --locked -p bitnet-kernels --all-targets --no-default-features --features cpu
cargo test --locked -p bitnet-kernels --lib --no-default-features --features cpu
```

Do not mechanically satisfy Clippy in kernels if semantics change.

### PR 14: No-Panic First Owner-Lane Burndown

```bash
cargo test --locked -p <touched-crate>
cargo run --locked -p xtask --no-default-features -- check-no-panic-family --report-dir target/bitnet/reports
cargo run --locked -p xtask --no-default-features -- no-panic baseline
```

Good first lanes: `bitnet-atomic-file-core`, `bitnet-http-retry`,
`bitnet-api-key-auth-core`, `bitnet-client-ip-core`,
`bitnet-request-router-core`, `bitnet-server-health-types-core`, or narrow
`xtask` policy/report helpers.

### PR 15: LEM / Risk-Pack Tightening

Evidence split:

```text
Default PR:
  fmt
  check
  clippy
  tests
  no-panic
  file/lint/process/network policy
  ripr static mutation-exposure analysis

Targeted PR:
  mutation for changed high-risk owner surfaces
  coverage when label coverage/full-ci

Nightly:
  mutation matrix
  deeper coverage
  dogfood/report drift

Release:
  package list
  publish dry-run
  release-readiness
```

Skipped lanes must report `skipped-by-policy`. Do not hide skipped lanes as
passed.

### PR 16: Next Minor Release Prep

```bash
cargo check --locked --workspace --all-targets --features cpu
cargo run --locked -p xtask --no-default-features -- policy-report --report-dir target/bitnet/reports
cargo run --locked -p xtask --no-default-features -- campaign doctor || true
git diff --check
```

Required changelog text:

```markdown
### Changed
- Raised MSRV to Rust 1.95.0.
- Activated Rust 1.94/1.95 Clippy policy ratchets.
- Removed Clippy test unwrap/expect carveouts.
- Added exact no-panic baseline/no-new-debt enforcement.
- Tightened file-policy and CI lane whitelist handling.
- Applied targeted Rust 1.95 cleanup in dispatch, receipt, and policy paths.
```

### PR 17: Release Dry-Run Proof

```bash
cargo package --locked -p bitnet --allow-dirty
cargo publish --dry-run --locked -p bitnet

cargo check --locked --workspace --all-targets --features cpu
cargo test --locked -p bitnet-common
cargo test --locked -p bitnet-math
cargo test --locked -p bitnet-quantization --no-default-features --features cpu
cargo test --locked -p bitnet-kernels --lib --no-default-features --features cpu

cargo run --locked -p xtask --no-default-features -- policy-report --report-dir target/bitnet/reports
```

## Operating Rules

- Start every PR from clean `origin/main`.
- One PR per objective.
- Open PRs as draft first.
- Do not push `main`.
- Do not force-push except to your own PR branch after rebase.
- Address bot comments and CI failures before marking ready.
- Self-review every PR.
- Merge only when required checks are green.
- After each merge, fetch and fast-forward `main` before the next PR.
- Do not claim green until post-merge `main` checks are green.

## Do Not

- Do not combine MSRV bump, lint activation, no-panic baseline, release bump,
  and cleanup in one PR.
- Do not weaken schemas or policy to satisfy CI.
- Do not add test Clippy carveouts.
- Do not add bare `#[allow(clippy::...)]` suppressions.
- Do not reset no-panic baseline except in the dedicated baseline PR.
- Do not make `ripr` branch-protection blocking yet.
- Do not replace mutation testing with `ripr`.
- Do not put full mutation on ordinary PRs.
- Do not hide skipped lanes as passed.

## Bot And CI Loop

For each PR:

```bash
gh pr view <PR> --json statusCheckRollup,reviewDecision,mergeStateStatus
gh pr checks <PR> --watch
```

If CI fails:

```bash
gh run view <run-id> --log-failed
```

Then:

1. Identify the first real failing command.
2. Reproduce locally if possible.
3. Fix only that failure.
4. Rerun the matching local gate.
5. Push.
6. Check bot comments again.

If a bot comments:

- Real defect -> fix.
- False positive -> reply with evidence.
- Style-only but cheap and in scope -> fix.
- Out of scope -> document follow-up.
- Stale comment -> verify current HEAD and mark stale.

## Required Self-Review Comment

```markdown
## Self-review

- Scope matches PR title:
- Files touched are expected:
- No unrelated cleanup:
- Policy changes are intentional:
- No Clippy test carveouts added:
- No bare `#[allow(clippy::...)]` added:
- No-panic baseline handling is scoped:
- Non-Rust allowlist changes are narrow:
- CI economics: lanes are risk-pack appropriate:
- ripr mutation-exposure framing preserved:
- Local validation:
- CI status:
- Bot comments addressed:
- Follow-ups:
```

## Current Control Plane Summary

The repo already has the following policy machinery in place. The Rust 1.95
rollout builds on it without replacing it.

| Component | File | Status |
|---|---|---|
| CI lane whitelist | `policy/ci-lane-whitelist.toml` | present |
| CI whitelist exceptions | `policy/ci-whitelist-exceptions.toml` | present |
| CI budget | `policy/ci-budget.toml` | present |
| CI lanes | `policy/ci-lanes.toml` | present |
| CI risk packs | `policy/ci-risk-packs.toml` | present |
| Clippy lints ledger | `policy/clippy-lints.toml` | present, 1.94/1.95 staged |
| Clippy debt | `policy/clippy-debt.toml` | present, placeholder only |
| Clippy exceptions | `policy/clippy-exceptions.toml` | present, empty |
| No-panic allowlist | `policy/no-panic-allowlist.toml` | present, empty/advisory |
| No-panic baseline | `policy/no-panic-baseline.toml` | absent |
| Non-Rust allowlist | `policy/non-rust-allowlist.toml` | present, broad |
| `ripr` suppressions | `policy/ripr-suppressions.toml` | present |
| Policy workflow | `.github/workflows/policy.yml` | running policy checks |
| `ripr` workflow | `.github/workflows/ripr.yml` | exists, records no-op when binary absent |

## Focus Notes

### Clippy Test Carveout Mismatch

Before PR 6, `clippy.toml` had:

```toml
allow-expect-in-tests = true
allow-unwrap-in-tests = true
```

`policy/clippy-lints.toml` says:

```toml
panic_free_tests = true
allow_test_carveouts = false
```

PR 6 resolved this contradiction by removing the carveouts and converting one
narrow helper slice.

### No-Panic Identity Hardening

Current allowlist identity is `path + family + selector`, with line/column
advisory only. Before real allowlist entries or a generated baseline can be
trusted, identity must be exact and counted:

```text
path + family + selector_kind + selector_callee + snippet + count
```

Matching must be consumptive: exact allowlist counts first, baseline counts
second unless blocking mode ignores the baseline, and anything remaining is new
debt.

### `ripr` Advisory Status

`ripr.yml` currently checks whether the `ripr` binary exists and skips when it
does not. PR 11 replaces that with a real install and run. The job stays
advisory and does not become branch-protection blocking in this wave.

`ripr` is central to the CI economics plan because it shifts mutation signal
left. It catches much of the same weak-test/oracle-exposure signal mutation
testing catches, but earlier and cheaper. Mutation testing remains the runtime
backstop for targeted risk PRs, nightly, and release readiness.

### Candidate `disallowed_fields` Seams

`disallowed_fields` stays planned until protected fields are configured in
`clippy.toml`. Candidate seams:

```text
engine lifecycle state
KV-cache policy internals
request context / API auth internals
download/auth retry metadata
GGUF tensor metadata
benchmark receipt internals
campaign tracker state
backend dispatch route labels
```
