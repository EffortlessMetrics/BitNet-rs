# Effortless Metrics Clippy policy

BitNet-rs uses the Effortless Metrics Rust lint policy as a governed engineering surface, not as an ad hoc list of local preferences. The policy is intentionally workspace-wide: production code, tests, examples, build helpers, and `xtask` all inherit the same baseline.

## Policy goals

The active baseline enforces four workspace promises. For this first BitNet-rs rollout, `unsafe_code` is deliberately staged as `warn` because the repository contains reviewed FFI/GPU boundaries that need a follow-up unsafe-boundary cleanup before a forbid-level ratchet:

1. **Panic-free Rust by default.** `unwrap`, `expect`, `panic!`, `todo!`, `unimplemented!`, `unreachable!`, unchecked indexing, and unchecked string slicing are denied for production and tests.
2. **No silent failure.** Futures, must-use values, locks, `Result::ok`, ignored `map_err` values, suspicious result assertions, and `lines().filter_map(Result::ok)` are denied.
3. **Explicit suppression governance.** New suppressions should use narrow `#[expect(..., reason = "...")]` receipts. Silent `#[allow]` attributes and blanket Clippy restriction enables are denied.
4. **Reviewable numeric/GPU code.** BitNet-rs keeps GPU/numeric footguns visible while staging churn-heavy arithmetic and numeric-cast checks as warnings where global denies would obscure product work.

## Source of truth

The policy has three layers:

- `Cargo.toml` contains the active workspace lint levels under `[workspace.lints.rust]` and `[workspace.lints.clippy]`.
- `policy/clippy-lints.toml` is the machine-readable ledger for active lints, policy posture, and planned Rust 1.94/1.95 flips.
- `policy/clippy-debt.toml` records temporary, reviewed exceptions with owner, reason, affected path, lint, and expiry.

`clippy.toml` is reserved for repo-local Clippy knobs such as thresholds and future `disallowed-methods` or `disallowed-types` policy. It must not contain test carveouts such as `allow-unwrap-in-tests`, `allow-expect-in-tests`, `allow-panic-in-tests`, `allow-indexing-slicing-in-tests`, or `allow-dbg-in-tests`.

## Suppression style

Prefer fixing the lint. When a local exception is truly needed, use an expectation with a reason:

```rust
#[expect(
    clippy::arithmetic_side_effects,
    reason = "Kernel loop uses reviewed wrapping arithmetic; tracked in policy/clippy-debt.toml."
)]
fn kernel_lane_index(base: usize, lane: usize) -> usize {
    base + lane
}
```

Do not add broad module-level `#[allow(...)]` blocks or test-only carveouts. If a suppression represents known debt rather than a permanent invariant, add an expiring entry to `policy/clippy-debt.toml`.

## Panic-free tests

Tests should return `Result` and use `?` for setup/fixture failures rather than `unwrap`, `expect`, or panic-driven setup:

```rust
#[test]
fn parses_fixture() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = std::fs::read_to_string("tests/fixtures/input.gguf")?;
    let parsed = parse_fixture(&fixture)?;

    ensure_fixture_shape(&parsed)?;
    Ok(())
}
```

This keeps test failures contextual without weakening the workspace panic policy.

## Upgrade tracking

Rust 1.94 and 1.95 lints are tracked in `policy/clippy-lints.toml` before the MSRV bump. `cargo xtask check-lint-policy` fails if planned lints become active early or if the ledger and workspace manifest drift.

## Required check

Run this before opening policy or linting changes:

```sh
cargo xtask check-lint-policy
```

The check validates MSRV alignment, workspace lint inheritance, active lint consistency, planned-lint staging, no test carveouts, and debt-entry shape.
