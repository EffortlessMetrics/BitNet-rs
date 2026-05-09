# BitNet-rs Rust 1.95 / Next Minor CI Economics Continuation

This document is the control map for the Rust 1.95 / next minor quality wave.
It is the direct continuation of #3866 (Rust 1.93 CI economics control plane),
not a speculative rollout.

Reading this document first prevents agents from:

- redoing already-landed #3866 work,
- mixing MSRV bump + lint activation + release bump in one PR,
- broadening default CI lanes under pressure,
- hiding policy debt inside unrelated cleanups.

## Control-plane intent

The Rust 1.95 wave is a CI economics continuation.
The goal is not to make every PR run more checks. The goal is to make each PR
run a better-shaped set of checks:

- cheaper default lane selection,
- clearer risk-pack routing,
- advisory static-exposure evidence from `ripr`,
- stricter Clippy receipts for intentional exceptions,
- less mutation-testing-style per-PR churn,
- no hidden broadening under CI pressure.

## Relationship to #3866

PR #3866 established the Rust 1.93 CI economics control plane:

- MSRV 1.93.0
- CI lane whitelist and LEM-aware budgets
- Policy workflow (CI lane, file-policy, lint-inheritance, Clippy exceptions, no-panic, policy-report)
- PR Gate observation mode
- `ripr` advisory surface
- Lint inheritance checker
- Strict-policy ledgers (Clippy, no-panic, non-Rust file allowlist)

The 1.93 rollout explicitly deferred:

- strict Clippy ratchets and receipt discipline,
- removal of test carveouts still in `clippy.toml`,
- no-panic identity hardening and no-new-debt mode,
- real `ripr` advisory execution (binary absent),
- LEM/risk-pack tightening to reduce default per-PR mutation,
- branch-protection migration to `PR Gate Success` aggregator,
- learned-estimate planner switch (needs one full sprint of data).

This document defines the Rust 1.95 / next minor ladder that addresses those
deferred items.

## Current vs. target state

| Layer | Current state | Target | Status |
|---|---|---|---|
| Edition | Rust 2024 | Rust 2024 | done |
| MSRV | 1.93.0 | 1.95.0 | planned |
| Release line | 0.2.1-dev | next minor dev line | planned |
| Clippy ledger | present, 1.94/1.95 lints staged | 1.95 lints active and ratcheted | partial |
| Clippy debt | present, placeholder entries only | owner/reason/expiry for real debt | partial |
| Clippy exceptions | present, empty | exact `#[expect]` receipts only | partial |
| `clippy.toml` carveouts | `allow-expect-in-tests = true` and `allow-unwrap-in-tests = true` still present | no test carveouts | todo |
| No-panic policy | allowlist exists, advisory/empty | no-new-debt baseline + exact allowlist | partial |
| Non-Rust policy | broad allowlist exists | tightened, reviewed, generated/executable companions | partial |
| CI lane whitelist | present | enforced and calibrated for 1.95 | partial |
| `ripr` | workflow exists, binary may be absent | real advisory analysis running | partial |

**Operating rule:** The first implementation PR after this doc PR is a
compatibility spike. No MSRV bump, lint activation, no-panic baseline reset,
release bump, or API cleanup happens in the same PR.

## Why a minor release?

Raising the MSRV breaks a user-visible contract: callers must now have Rust
1.95.0 to build the crate. Under semver this is a **minor** version change
(no existing API is removed, but the build requirement changes). The
development line therefore moves from `0.2.1-dev` to the next minor dev line.
The exact version and tag are finalized in the release-prep PR because
`CHANGELOG.md` already contains a historical `0.3.0` section.

## Rust 1.95 value for BitNet-rs

| Rust 1.95 item | BitNet-rs use |
|---|---|
| `if let` guards | dispatch planners, backend routing, tokenizer/model metadata classification, GGUF shape checks, CUDA/CPU route selection |
| `Vec::push_mut` / `insert_mut` | receipts, campaign dashboards, benchmark reports, API/generation events, diagnostic builders |
| Atomic `update` / `try_update` | runtime metrics, once-warn counters, request/health/state counters |
| `cfg_select!` | GPU/CPU/wasm/platform/backend routing without cfg sprawl |
| `cold_path` | error/report/rejection paths in parser, GGUF validation, download/auth, API routing, policy failures |
| Clippy 1.95 | `manual_checked_ops`, `manual_take`, `manual_pop_if`, `duration_suboptimal_units`, `needless_type_cast`, `unnecessary_trailing_comma` |

## Implementation PR ladder

Each PR below is a single objective. No PR combines MSRV bump, lint activation,
no-panic baseline changes, release bump, and code cleanup simultaneously.

| PR | Branch | Title | Scope | Keep? | Adjustment |
|---|---|---|---|---|---|
| 1 | `docs/rust-1.95-rollout` | `policy: map Rust 1.95 CI economics and strict Clippy continuation` | **This PR.** Documentation / control-plane map only. | yes | Rebuilt from current `main`. Replaces closed #4136. |
| 2 | `probe/rust-1.95-compat` | `chore(msrv): probe Rust 1.95 compatibility` | Run repo under 1.95.0 before changing declared MSRV. Audit note only; code changes only for true fallout. Must rerun on current `main`; do not trust old branch data. | yes | Must rerun on current `main`. |
| 3 | `chore/msrv-rust-1.95` | `chore(msrv): raise workspace toolchain to Rust 1.95` | `rust-toolchain.toml`, `Cargo.toml` `rust-version`, `clippy.toml` `msrv`, workflow toolchain keys. | yes | Only after probe/PR 2 is green. |
| 4 | `policy/rust-1.95-lints` | `policy(rust): enable Rust 1.95 compiler lint floor` | Activate `const_item_interior_mutations`, `function_casts_as_integer`, move `unexpected_cfgs` to `warn`. Keep narrow; no broad cleanup. | yes | Keep narrow; no broad cleanup. |
| 5 | `policy/clippy-rust-1.95-ratchets` | `policy(clippy): activate Rust 1.95 lint ratchets` | Promote `manual_checked_ops`, `manual_take`, `manual_pop_if`, `duration_suboptimal_units`, `needless_type_cast`, `unnecessary_trailing_comma` after measurement. Only activate lints with measured debt and receipts. | yes | Only activate with measured debt and receipts. |
| 6 | `policy/no-test-clippy-carveouts` | `policy(clippy): remove test unwrap and expect carveouts` | Delete `allow-expect-in-tests` / `allow-unwrap-in-tests` from `clippy.toml`. Add fallible test helpers. Current `clippy.toml` still has them. | yes | Current `clippy.toml` still has them. |
| 7 | `policy/no-panic-exact-identity` | `policy(panic): harden no-panic allowlist identity` | Require `path + family + selector_kind + selector_callee + snippet + count` identity. Counted-consumptive matching. Still aligns with #3866 deferred backlog. | yes | Still aligns with #3866 deferred backlog. |
| 8 | `policy/no-panic-baseline` | `policy(panic): add no-panic baseline and no-new-debt gate` | Generate baseline, set `no-new-debt` mode, add `.gitattributes` generated marker. | yes | Still aligns with #3866 deferred backlog. |
| 9 | `policy/no-panic-diagnostics` | `policy(panic): improve no-panic report diagnostics` | Missing-baseline error, stale entries in MD/JSON, blocking-mode messaging, delta details. | yes | Still aligns with #3866 deferred backlog. |
| 10 | `policy/file-allowlist-tightening` | `policy(files): tighten non-Rust allowlist coverage` | Remove stale entries, narrow broad globs, add `review_after`/`expires`. Tie to reduced mutation/noisy PR surface. | yes | Tie to reduced mutation/noisy PR surface. |
| 11 | `ci/ripr-real-advisory` | `ci(ripr): provision real advisory static exposure analysis` | Install `ripr`, run `ripr check`, emit JSON/SARIF/Markdown artefacts. Central CI economics item, not a side quest. | yes | Make this a central CI economics item. |
| 12 | `refactor/rust-1.95-api-cleanups` | `refactor: use Rust 1.95 APIs in dispatch and receipt builders` | Targeted `if let` guards, `Vec::push_mut`, atomic `update`, `cfg_select!`. Only after MSRV bump lands. | yes, but after PR 3 | Only after MSRV bump (PR 3) lands. |
| 13 | `policy/clippy-numeric-kernel-cleanup` | `policy(clippy): clean numeric and kernel lint debt` | `manual_checked_ops`, `cast_possible_truncation`, `indexing_slicing`. Risk-pack routed, not default broad lane. | yes | Risk-pack routed, not default broad lane. |
| 14 | `policy/no-panic-first-burndown` | `policy(panic): burn down first no-panic owner lane` | One narrow lane only (e.g. `bitnet-http-retry`, `bitnet-api-key-auth-core`, or `xtask` helpers). | yes | One owner lane only. |
| 15 | `ci/bitnet-lem-lane-tightening` | `ci: tighten lane whitelist and LEM routing for Rust 1.95` | Update `policy/ci-lane-whitelist.toml`, reclassify `ripr-advisory`, make GPU/FFI lanes label/risk-routed. Make explicit: reduce default PR mutation and over-testing. | yes | Make explicit: reduce default PR mutation. |
| 16 | `release/next-minor-prep-rust-1.95` | `release: prepare next minor release for Rust 1.95` | Move version from `0.2.1-dev` to the next minor dev line workspace-wide. Update CHANGELOG and reconcile the historical `0.3.0` section before choosing the exact version/tag. | yes | Minor bump because MSRV change is user-visible. |
| 17 | `release/next-minor-dry-run` | `release: validate next minor publish readiness` | `cargo package --dry-run`, full policy-report, release readiness doc for the chosen version. | yes | Minor bump because MSRV change is user-visible. |

### Acceptance gate per PR

#### PR 1 (this doc PR)

```bash
cargo run --locked -p xtask --no-default-features -- check-file-policy --report-dir target/bitnet/reports
cargo run --locked -p xtask --no-default-features -- policy-report --report-dir target/bitnet/reports
cargo run --locked -p xtask --no-default-features -- ci-lane-whitelist check
cargo run --locked -p xtask --no-default-features -- check-lint-inheritance
cargo run --locked -p xtask --no-default-features -- check-clippy-exceptions
cargo fmt --all -- --check
git diff --check
```

#### PR 2 (compatibility spike)

```bash
rustup toolchain install 1.95.0 --component rustfmt --component clippy --component rust-analyzer
cargo +1.95.0 fmt --all -- --check
cargo +1.95.0 check --locked --workspace --all-targets --no-default-features
cargo +1.95.0 check --locked --workspace --all-targets --features cpu
cargo +1.95.0 clippy --locked --workspace --all-targets --no-default-features -- -D warnings
cargo +1.95.0 clippy --locked --workspace --all-targets --features cpu -- -D warnings
cargo +1.95.0 run --locked -p xtask --no-default-features -- policy-report --report-dir target/bitnet/reports
```

#### PR 3 (MSRV bump)

```bash
cargo fmt --all -- --check
cargo check --locked --workspace --all-targets --features cpu
cargo run --locked -p xtask --no-default-features -- check-lint-inheritance
cargo run --locked -p xtask --no-default-features -- check-clippy-exceptions
cargo run --locked -p xtask --no-default-features -- policy-report --report-dir target/bitnet/reports
git diff --check
```

## Commit and PR operating rules

- One PR per objective.
- Start every PR from clean `origin/main`.
- Do not push `main`.
- Do not force-push except to your own PR branch after a rebase.
- Open PRs as draft first.
- Address bot comments and CI failures before marking ready.
- Self-review every PR before marking ready-for-review.
- Merge only when required checks are green.
- After each merge, fetch and fast-forward `main` before starting the next PR.
- Do not claim green until post-merge `main` checks are green.

## What not to do

- Do not combine MSRV bump, lint activation, no-panic baseline, release bump,
  and cleanup in one PR.
- Do not weaken schemas or policy to satisfy CI.
- Do not add test Clippy carveouts.
- Do not add bare `#[allow(clippy::...)]` suppressions.
- Do not reset no-panic baseline except in the dedicated baseline PR.
- Do not make `ripr` branch-protection blocking yet.
- Do not route expensive GPU/FFI/platform lanes into default PR without a
  risk-pack reason.
- Do not hide skipped lanes as passed.
- Do not broadening default CI lanes under CI pressure.

## CI economics operating frame

For each PR, the CI economics question is:

1. Which lanes actually run for this change?
2. Are those lanes risk-pack appropriate?
3. Is `ripr` evidence present?
4. Are any expensive lanes being pulled into the default path without justification?

Lane tightening (PR 15) makes this explicit. Until then, monitor
`policy/ci-lane-whitelist.toml` and do not add new lanes to the default set.

## Bot and CI loop

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
6. Re-check bot comments.

If CodeRabbit or another bot comments:

- Real defect → fix.
- False positive → reply with evidence.
- Style-only but cheap and in scope → fix.
- Out of scope → document as follow-up.
- Stale comment → verify current HEAD and mark stale.

## Required self-review checklist

Before marking any PR ready-for-review, add this comment:

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
- Local validation:
- CI status:
- Bot comments addressed:
- Follow-ups:
```

## Current control plane summary

The repo already has the following policy machinery in place (landed in #3866).
The 1.95 rollout builds on top of it without replacing it.

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
| No-panic allowlist | `policy/no-panic-allowlist.toml` | present, advisory |
| Non-Rust allowlist | `policy/non-rust-allowlist.toml` | present, broad |
| ripr suppressions | `policy/ripr-suppressions.toml` | present |
| Policy workflow | `.github/workflows/policy.yml` | running: CI lane, file-policy, lint-inheritance, Clippy exception, no-panic, policy-report |
| ripr workflow | `.github/workflows/ripr.yml` | exists, records no-op when binary absent |

## Clippy test carveout mismatch (PR 6 target)

`clippy.toml` currently has:

```toml
allow-expect-in-tests = true
allow-unwrap-in-tests = true
```

`policy/clippy-lints.toml` says:

```toml
panic_free_tests = true
allow_test_carveouts = false
```

This contradiction is intentional during the staging window (see original
rollout document). PR 6 resolves it by removing the carveouts after PR 5
activates the fallible test helpers.

## No-panic identity hardening (PR 7 target)

Current allowlist identity is `path + family + selector`. Before bulk baseline
work begins, the identity must expand to include `snippet` and `count` to
prevent one allow entry from accidentally covering unrelated calls in the same
file. PR 7 lands this before PR 8 generates the baseline.

## `ripr` advisory status

`ripr.yml` currently checks whether the `ripr` binary exists and skips when it
does not. PR 11 replaces this with a real install and run. The job stays
advisory; it does not become a branch-protection gate in this wave.
`ripr` is a central CI economics item: its static-exposure evidence reduces
the need for broad mutation-testing-style PR coverage.

## Candidate `disallowed_fields` seams (PR 5 prerequisite)

The following internal seams are candidates for `disallowed_fields` enforcement.
They must be defined in `clippy.toml` before the lint can be activated globally.

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

Activation of `disallowed_fields` globally is deferred until real seams are
defined. Use it only after defining protected field paths.
