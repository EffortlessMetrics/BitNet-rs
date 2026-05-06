# Clippy and Lint Policy

> Status: **PR 1 of a stacked rollout — governance scaffolding only.**
> The strict baseline lint block has not yet been applied to `[workspace.lints]`.
> See "Rollout plan" below for the schedule.

## What this is

This document describes the lint governance model for the `bitnet-rs`
workspace. Lints are treated as **infrastructure**, not as personal taste:

- The list of lints we intend to enforce lives in
  [`policy/clippy-lints.toml`](../policy/clippy-lints.toml) — a
  machine-readable ledger covering both currently-active lints and lints
  planned to flip on a future Rust MSRV.
- Temporary exceptions live in
  [`policy/clippy-debt.toml`](../policy/clippy-debt.toml). Every entry has an
  owner, a reason, and an expiry; CI fails on expired or undeclared debt.
- Per-call-site panic exceptions live in
  [`policy/no-panic-allowlist.toml`](../policy/no-panic-allowlist.toml) using
  semantic identifiers (`path + family + selector`), not line numbers.
- Non-Rust source files live behind
  [`policy/non-rust-allowlist.toml`](../policy/non-rust-allowlist.toml),
  which migrates the prior pipe-delimited convention to TOML.
- `cargo run -p xtask -- check-lint-policy` verifies that the policy ledger,
  `Cargo.toml`, `clippy.toml`, and `rust-toolchain.toml` agree.

## Posture

Three rules drive the policy:

1. **Global deny by default. Local exception by structured receipt.**
   No silent `#[allow]`. Every suppression must use
   `#[expect(..., reason = "...")]`, or appear as a tracked entry in
   `policy/clippy-debt.toml` or `policy/no-panic-allowlist.toml`.
2. **Identity is semantic, not positional.**
   The no-panic allowlist keys on `path + family + selector`; line/column are
   advisory hints used only for diagnostics. This survives normal refactors.
3. **Future flips are tracked, not surprises.**
   Lints planned to flip when MSRV rises (Rust 1.94 / 1.95 cohorts) are
   declared in `policy/clippy-lints.toml` so we can adopt them on the same
   cycle as the toolchain bump.

## Suppression style

All suppressions must use `#[expect]` with a `reason` string:

```rust
#[expect(
    clippy::cast_possible_truncation,
    reason = "Tensor sizes are bounded by GGUF spec; truncation is impossible.",
)]
let len = vector.len() as u32;
```

Bare `#[allow(...)]` will be denied workspace-wide once PR 2 lands. Do not
add a `#[allow]` even temporarily — use `policy/clippy-debt.toml` instead so
the exception is reviewable, has an owner, and has an expiry.

## BitNet-rs overlay

bitnet-rs is a high-churn numeric / GPU / FFI workspace. The strict baseline
applies, but with an explicit overlay:

| Lint class                 | Workspace level | BitNet-rs overlay |
|----------------------------|-----------------|-------------------|
| Panic family               | `deny`          | `deny`            |
| Silent failure             | `deny`          | `deny`            |
| Suppression governance     | `deny`          | `deny`            |
| Numeric correctness        | `warn` → `deny` | `warn` (ratchet per crate) |
| `arithmetic_side_effects`  | `warn`          | `warn` (no estate-wide deny without kernel triage) |
| `unsafe_code`              | `forbid`        | `forbid` *except* in GPU backend crates (`bitnet-kernels`, `bitnet-rocm`, `bitnet-cuda*`, `bitnet-metal`, `bitnet-vulkan`, `bitnet-opencl`, `bitnet-wgpu`) where it is `deny` with documented `unsafe { }` blocks |
| Test carveouts             | none            | currently `allow-unwrap-in-tests = true` / `allow-expect-in-tests = true` in `clippy.toml`; **scheduled for removal in PR 2** |

The overlay is encoded in `policy/clippy-lints.toml` as `class` and
`overlay_exception` fields per planned lint.

## Rollout plan

This is a stacked rollout. Each PR is independently reviewable.

### PR 1 — Governance scaffolding (this PR)

- Add `policy/clippy-lints.toml`, `policy/clippy-debt.toml`,
  `policy/no-panic-allowlist.toml`, `policy/non-rust-allowlist.toml`.
- Add this document.
- Add `cargo xtask check-lint-policy` (advisory). No lint behavior changes.

### PR 2 — Strict baseline kickoff

- Promote `cargo xtask check-lint-policy --strict` to a CI gate (Guards
  workflow).
- Add `Cargo.toml` ↔ `policy/clippy-lints.toml` consistency check: every
  `[workspace.lints.<root>]` entry must appear as `[[active]]` in the ledger
  with the same level, and vice versa (category lints excepted until they are
  expanded into their explicit lint set).
- Promote `clippy::dbg_macro` to `deny` (zero-hit lint, safe to land).
- Pedantic overrides in `Cargo.toml` (`missing_errors_doc`, `missing_panics_doc`,
  `module_name_repetitions`, `must_use_candidate`) are recorded as
  `[[active]] level = "allow"` in the ledger.

### PR 3 — Suppression migration

- Convert existing `#[allow(clippy::*)]` attributes (~447 instances) to
  `#[expect(clippy::*, reason = "...")]` with reasons sourced from the
  surrounding context.
- Add `clippy::allow_attributes_without_reason = "deny"` and
  `clippy::blanket_clippy_restriction_lints = "deny"`.

### PR 4 — Panic family ratchet

- Per-crate ratchet of `clippy::unwrap_used`, `clippy::expect_used`,
  `clippy::panic`, `clippy::unimplemented`, `clippy::unreachable` from
  `allow` → `warn` → `deny`. Each crate gets a debt entry + `#![allow]` at
  the crate root until cleaned.
- Remove `allow-unwrap-in-tests` and `allow-expect-in-tests` from
  `clippy.toml` once test code is migrated to `Result`-returning helpers.

### PR 5 — MSRV ratchet 1.92 → 1.93

- Bump `rust-toolchain.toml` and `workspace.package.rust-version`.
- Bump `policy/clippy-lints.toml` `msrv = "1.93"`.
- Promote `warn`-level numeric lints to `deny` per crate as kernel debt is
  cleared.

### PR 6+ — Rust 1.94 / 1.95 flip cohorts

- When MSRV reaches 1.94, activate the lints declared with
  `activate_when_msrv = "1.94"` (same for 1.95).
- The xtask check verifies the cohort matches the planned ledger.

## Estate-wide context

This policy is part of a platform-wide rollout across the Effortless Metrics
Rust estate (`ripr`, `perl-lsp`, `perfgate`, `tokmd`, `shiplog`, `uselesskey`,
`hl7v2-rs`, `lintdiff`, `BitNet-rs`, …). The shared baseline is identical
across repos; per-repo overlays are restricted to *adding* strictness or
*tracking* domain-specific debt. The same `policy/` directory layout and
TOML schemas are used everywhere so that one tooling pass produces a
consistent estate report.

## Related tooling

- `ripr` — second-stage evidence layer. Clippy says "the code shape is
  acceptable"; ripr says "the behavior seams have test grip." The two
  together form the org-wide Rust quality floor.
- `cargo xtask check-lint-policy` (this repo) — verifies repo-local policy
  consistency.
- `cargo xtask check-no-panic-family` (planned) — AST-aware enforcement of
  `policy/no-panic-allowlist.toml`.
- `cargo xtask check-file-policy` (planned) — enforcement of
  `policy/non-rust-allowlist.toml`.
