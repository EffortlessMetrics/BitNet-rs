# Effortless Metrics Clippy Policy

BitNet-rs uses the shared Effortless Metrics Rust lint policy as an engineering
surface, not as local taste embedded in `Cargo.toml`. The policy has three
checked artifacts:

- `Cargo.toml` carries the active workspace lint levels.
- `policy/clippy-lints.toml` is the machine-readable ledger for active lints and
  planned Rust 1.94/1.95 flips.
- `cargo xtask check-lint-policy` verifies that the manifest, Clippy config, and
  policy ledgers still agree.

## Baseline

The active baseline is workspace-wide and applies to production code and tests.
It surfaces unsafe code as reviewed rollout debt, denies panic-family lints,
prevents silent failure patterns, rejects broad suppression habits, and keeps
GPU/numeric review lints staged as warnings where a blanket deny would create
churn before policy.

There are no test carveouts. Do not add Clippy configuration such as:

```toml
allow-unwrap-in-tests = true
allow-expect-in-tests = true
allow-panic-in-tests = true
allow-indexing-slicing-in-tests = true
allow-dbg-in-tests = true
```

Tests should return `Result` and propagate setup failures instead of using
`unwrap`, `expect`, or panic-driven fixture setup:

```rust
#[test]
fn parses_fixture() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = std::fs::read_to_string("tests/fixtures/input.txt")?;
    let parsed = parse_fixture(&fixture)?;

    ensure_eq!(parsed.items.len(), 3, "fixture should expose three items")?;

    Ok(())
}
```

## Suppressions

The suppression style is **structured receipt, not silent carveout**:

```rust
#[expect(
    clippy::indexing_slicing,
    reason = "Reviewed fixed-size QK256 tile indexing; bounds proven by tile constructor."
)]
let value = tile.values[index];
```

Use `#[expect(..., reason = "...")]` for narrow, local exceptions. Do not use
plain `#[allow]` for Clippy policy suppressions. If an exception represents
rollout debt rather than a permanent invariant, add it to `policy/clippy-debt.toml`
with an owner, reason, path, lint, and expiry.

## BitNet-rs overlay

BitNet-rs is a GPU/numeric/performance workspace. Numeric-correctness lints are
therefore split between immediate denies for clearly dangerous casts/comparisons
and staged warnings for broad arithmetic/cast review:

- `clippy::arithmetic_side_effects = "warn"`
- `clippy::cast_possible_wrap = "warn"`
- `clippy::cast_possible_truncation = "warn"`
- `clippy::cast_precision_loss = "warn"`

Backend-specific exceptions should stay narrow and documented at the call site.
Do not weaken the workspace lint block or add test carveouts to `clippy.toml`.

## Planned Rust 1.94 / 1.95 flips

The ledger tracks planned lints before the MSRV bump. `cargo xtask
check-lint-policy` rejects planned lints that become active before their recorded
MSRV gate. When the workspace moves to a newer MSRV, update the ledger and root
lint block in the same PR.

## Required local checks

Run these before policy changes are reviewed:

```shell
cargo fmt --all --check
cargo xtask check-lint-policy
cargo xtask policy-report
```

Full Clippy enforcement is the desired CI gate once current debt is triaged:

```shell
cargo clippy --workspace --all-targets --all-features -- -D warnings
```
