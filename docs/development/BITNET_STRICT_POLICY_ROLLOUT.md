# BitNet-rs Strict Policy Rollout

This document describes the multi-PR rollout that brings BitNet-rs to:

* MSRV 1.93
* a governed CI lane whitelist with LEM-aware budgets
* a strict Rust policy stack (Clippy receipts, no-panic semantic
  allowlist, non-Rust file allowlist, workspace lint inheritance,
  unsafe islands)
* `ripr` advisory static-exposure analysis on production Rust diffs
* an `xtask`-driven CI control plane that emits `ci-plan.json` and
  later learns from observed actuals

## Stack overview

| PR | Purpose                                                                |
| -: | ---------------------------------------------------------------------- |
| 01 | CI lane whitelist policy files                                         |
| 02 | `xtask ci-lane-whitelist check` and policy module skeleton             |
| 03 | **this PR**: MSRV 1.93 bump + strict policy docs/ledgers               |
| 04 | non-Rust TOML allowlist + `xtask check-file-policy` enforcement        |
| 05 | semantic no-panic allowlist + checker                                  |
| 06 | Clippy exception checker (no bare `#[allow]`)                          |
| 07 | workspace lint inheritance enforcement                                 |
| 08 | Clippy Stage A staged profile                                          |
| 09 | fallible test support helpers (`ensure`, `ensure_eq`, etc.)            |
| 10 | panic debt cleanup, default members                                    |
| 11 | panic debt cleanup, optional surfaces (FFI/GPU/Python/WASM/fuzz)       |
| 12 | strict Clippy flip (panic-family promoted to deny)                     |
| 13 | `ripr` advisory analysis                                               |
| 14 | `xtask ci plan` (replaces inline Python in pr-plan.yml)                |
| 15 | policy-backed LEM routing files (budget, lanes, risk packs)            |
| 16 | nextest / JUnit / actuals telemetry                                    |
| 17 | risk-pack routing for optional lanes                                   |
| 18 | soft budget guard (warn at 35/75 LEM, fail >125 without override)      |
| 19 | required PR Gate Success aggregator                                    |
| 20 | learned LEM estimates from observed actuals                            |

## BitNet-specific decisions

### `unsafe_code = "deny"`, not `"forbid"`

BitNet-rs has FFI, GPU, language bindings, memory mapping, and SIMD
surfaces that require narrow unsafe islands. `forbid` would block
those legitimately; `deny` allows them but requires explicit
documentation. Each unsafe island is eventually receipted via
`policy/unsafe-allowlist.toml` (added later in the rollout).

### No Clippy test carveouts

BitNet-rs intentionally **does not** add:

```toml
allow-unwrap-in-tests   = true
allow-expect-in-tests   = true
allow-panic-in-tests    = true
allow-indexing-slicing-in-tests = true
allow-dbg-in-tests      = true
```

Tests are part of the contract. The current `clippy.toml` keeps
`allow-unwrap-in-tests` and `allow-expect-in-tests` only as a staging
window; PR 09 adds the fallible helpers and PR 12 deletes the
carveouts.

### No global `-D warnings` for the policy lane

Once staged numeric/readability lints arrive, every new toolchain
warning would otherwise become blocking. The strict profile uses
explicit `deny` for blocking lints and `warn` for staged debt.

### `ripr` is advisory first

`ripr` is the static oracle-gap layer ("does the changed behavior
appear gripped by a meaningful test discriminator?"). It is not a
substitute for mutation testing. PR 13 adds it as advisory JSON /
SARIF / step-summary output. Promotion is later, after baseline
behavior is understood.

## Status

* **PRs 01–09 + 13–20 (this rollout): landed together as one stacked
  PR**, one commit per rollout PR. See the table below for the
  per-PR shipped artefact set.
* **PRs 10, 11, 12 deferred** as documented in this file's
  *Follow-up backlog* section. Those are mass mechanical migrations
  that cannot land atomically.

### Per-PR shipped artefacts

| Rollout PR | Artefact set                                                                                  |
| ---------: | --------------------------------------------------------------------------------------------- |
| 01         | `policy/ci-lane-whitelist.toml`, `policy/ci-whitelist-exceptions.toml`, `docs/ci/`             |
| 02         | `xtask/src/policy/`, `xtask ci-lane-whitelist check`, `xtask check-{file-policy,no-panic-family,clippy-exceptions,lint-inheritance}`, `xtask policy-report` |
| 03         | MSRV 1.93.0 across workspace + 28 workflows; `policy/clippy-{lints,debt,exceptions}.toml`; `docs/{CLIPPY_POLICY,NO_PANIC_POLICY,POLICY_ALLOWLISTS,FILE_POLICY}.md` |
| 04         | `policy/non-rust-allowlist.toml` (8146 files / 112 entries / 0 findings); `.github/workflows/policy.yml` (frontdoor blocking, `check-file-policy --fail-on-error`) |
| 05         | `policy/no-panic-allowlist.toml` seed; checker stays advisory through PR 12                    |
| 06         | Tightened `#[allow]`/`#[expect]` attribute-line scanner                                       |
| 07         | `[lints] workspace = true` added to 99 crate manifests; checker reports 0 missing across 136  |
| 08         | `[workspace.lints.rust]` block + Stage A explicit Clippy profile alongside `clippy::all`      |
| 09         | `bitnet_test_support::assertions` (`ensure`, `ensure_eq`, `ensure_ne`, `require_some`, `require_ok`, `require_ok_display`) |
| 13         | `ripr.toml`, `policy/ripr-suppressions.toml`, `.github/workflows/ripr.yml`, `docs/RIPR_EVIDENCE_POLICY.md` (advisory only) |
| 14         | `xtask ci plan` Rust port of the inline Python in `pr-plan.yml`                                |
| 15         | `policy/ci-budget.toml`, `policy/ci-lanes.toml`, `policy/ci-risk-packs.toml`                  |
| 16         | `[profile.pr]`, `[profile.nightly]` in `.config/nextest.toml`; `xtask ci actuals`             |
| 17         | Plan emits `risk_packs: [String]` (qk256, kernels_cpu, gpu, ffi, tokenizer, bdd_policy, manifest_release, docs_tracking) |
| 18         | Plan emits `guard: String` + `override_labels_present`; `--enforce-budget` flag               |
| 19         | `.github/workflows/pr-gate.yml` (`PR Gate Success` aggregator, observation mode); `docs/ci/pr-gate-success.md` |
| 20         | `xtask ci estimate` (`p50 × 1.15`, p90 warning, p95 hard); `docs/ci/learned-budgets.md`        |

### Verification at landing time

```
RUSTFLAGS="-D warnings" cargo clippy --locked \
  -p bitnet-common -p bitnet-models -p bitnet-tokenizers \
  -p bitnet-quantization -p bitnet-kernels --lib \
  --no-default-features --features cpu                 -> clean
cargo test  -p xtask --bin xtask -- policy:: ci::      -> 36 passing
cargo test  -p bitnet-test-support                      -> 26 passing
cargo run   -p xtask -- ci-lane-whitelist check         -> 15 lanes / 2 exceptions / 0 errors
cargo run   -p xtask -- check-file-policy               -> 8146 files / 112 allow / 0 findings
cargo run   -p xtask -- check-lint-inheritance          -> 136 crates / 0 missing
cargo fmt --all -- --check                              -> clean
```

## Follow-up backlog (deferred)

These are the explicit follow-ups that the rollout depends on but
that this stacked PR does **not** land. Each is one or more dedicated
PRs against the artefacts above.

| PR  | Scope                                  | Why deferred                              |
| --- | -------------------------------------- | ----------------------------------------- |
| 10  | panic-debt cleanup, default members    | ~31 510 unallowlisted findings; needs per-crate-cluster stacked PRs to be reviewable |
| 11  | panic-debt cleanup, optional surfaces  | FFI / GPU / Python / WASM / fuzz / bench / tooling — same reason as PR 10 |
| 12  | strict Clippy flip                     | depends on PR 10 + 11; promotes panic-family lints from `warn`/`allow` to `deny`, removes broad-category warns and test carveouts, removes the 476 bare `#[allow(clippy::...)]` shapes |
| —   | `unsafe_code = "deny"` + receipt ledger | adds `policy/unsafe-allowlist.toml` and flips the workspace setting; needs the unsafe-island inventory first |
| —   | learned-estimate planner switch        | requires at least one full sprint of `.ci/metrics/ci-lane-history.jsonl` data; the planner-side change is reversible in one commit when the data is ready |
| —   | branch-protection migration            | runbook in `docs/ci/pr-gate-success.md`; flip happens after one observation cycle of the `PR Gate Success` aggregator |
| —   | strict-policy lane promotion           | flip `check-no-panic-family` and `check-clippy-exceptions` to `--fail-on-error` once PRs 10 / 11 / 12 land |
| —   | inline Python deletion in `pr-plan.yml` | the legacy block is kept behind `if: false` for one parity cycle after PR 14; delete after the parity comparison |
| —   | risk-pack TOML hot-loading             | PR 17 embeds the path-prefix table; the follow-up reads `policy/ci-risk-packs.toml` directly |
| —   | `xtask ci actuals` GitHub API source   | PR 16 accepts CLI arguments; the follow-up swaps in a GitHub API call |

## Receipts and expirations

Every policy file in `policy/` has an `expires` field on each
exception. Expired exceptions fail their check. Reviewing the policy
ledgers is part of the routine maintenance covered by the
`review_after` dates in each file.
