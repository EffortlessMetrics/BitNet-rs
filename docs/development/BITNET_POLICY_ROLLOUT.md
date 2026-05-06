# BitNet Policy Rollout

This document tracks the staged adoption of the Effortless Metrics shared
strict Clippy + allowlist policy in BitNet-rs. The policy is designed to be
identical across the Rust estate — see
[`STRICT_CLIPPY_POLICY.md`](./STRICT_CLIPPY_POLICY.md) and
[`POLICY_ALLOWLISTS.md`](./POLICY_ALLOWLISTS.md) for the model.

## Stack overview

| PR | Branch / commit | Status | Purpose |
| -- | --- | --- | --- |
| 1 | `policy/msrv-1-93-rollout-plan` | **landed** | MSRV 1.93 + policy ledger + rollout docs |
| 2 | `policy/non-rust-toml-allowlist` | **landed** | TOML non-Rust allowlist + `xtask check-file-policy` |
| 3 | `policy/no-panic-semantic-allowlist` | **landed** | AST panic scanner + semantic TOML allowlist |
| 4 | `policy/workspace-lint-inheritance` | **landed** | Every workspace member opts in to workspace lints |
| 5 | `policy/clippy-ledger-stage-a` | **landed** | Explicit staged Clippy profile (panic still warn) |
| 6 | `testing/fallible-test-support` | pending | Fallible test helpers in `bitnet-test-support` |
| 7 | `panic-debt/default-members` | pending | Burn down panic-family debt in default-members |
| 8 | `panic-debt/optional-surfaces` | pending | Same for FFI/GPU/python/wasm/fuzz/xtask/crossval |
| 9 | `policy/strict-clippy-flip` | pending | Promote panic-family lints to deny + full profile |
| 10 | `ripr/evidence-reporting` | pending | Advisory `ripr` reports + CI artifacts |

PR 1–5 land the *infrastructure*: documentation, MSRV, the policy ledger,
allowlist schemas, scanners, lint inheritance, and the staged-A Clippy
profile that does not yet promote panic-family lints. They do not require
touching test bodies or refactoring panic-family call sites.

PR 6–10 are the *migration* phase: they remove or receipt existing debt,
flip panic-family lints to `deny`, and add `ripr` as the second-stage
evidence layer.

## Status snapshot

| Item | Current |
| --- | --- |
| Workspace MSRV | `1.93` |
| `rust-toolchain.toml` channel | `1.93.0` |
| Workspace metadata MSRV | `1.93` |
| `policy/clippy-lints.toml` schema | `1.0` |
| `policy/non-rust-allowlist.toml` schema | `1.0` |
| `policy/no-panic-allowlist.toml` schema | `0.3` |
| Non-Rust allowlist enforcement | advisory until PR 9 |
| No-panic allowlist enforcement | blocking after PR 5 |
| `[lints] workspace = true` inheritance | every member |
| Clippy profile stage | A (panic-family at warn) |
| Panic-family Clippy lints | `warn` (gated by `xtask check-no-panic-family`) |
| `unsafe_code` | `deny` (FFI/GPU islands use crate-local `#[expect]`) |
| Test carve-outs in `clippy.toml` | none |
| `ripr` evidence reports | not added yet (PR 10) |

## Operating principles

1. Strict baseline is the same across the Rust estate.
2. Repo-specific posture is encoded as overlays and structured exceptions,
   not as a weaker baseline.
3. Every exception is reviewable: owner, reason, optional expiry.
4. No bare `#[allow]`. Suppressions use `#[expect(..., reason = "...")]`.
5. Tests are workspace surface, not a carve-out.
6. Future toolchain flips are tracked before the bump, not after.
7. Reports under `target/bitnet/reports/` are the canonical artifact path.

## Updating this file

Each rollout PR updates the table above when it lands. New status fields
are added under "Status snapshot" rather than buried in PR descriptions.
