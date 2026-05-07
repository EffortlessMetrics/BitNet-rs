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

* PR 01: merged (policy files, advisory)
* PR 02: merged (xtask checker)
* PR 03: this PR — MSRV 1.93, strict policy docs and ledgers
* PR 04+ in flight on this branch

## Receipts and expirations

Every policy file in `policy/` has an `expires` field on each
exception. Expired exceptions fail their check. Reviewing the policy
ledgers is part of the routine maintenance covered by the
`review_after` dates in each file.
