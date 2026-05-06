# Clippy policy

BitNet-rs follows the Effortless Metrics Rust platform policy: MSRV 1.93,
panic-free production and test code, AST/string/indexing safety by default,
explicit suppression receipts, and machine-readable debt for temporary rollout
exceptions.

## Operating rules

- The root `Cargo.toml` owns the active workspace lint block.
- Every workspace member inherits the root lint policy with `[lints] workspace = true`.
- `clippy.toml` is only for repo-specific disallowed methods, types, macros, and similar additions.
- Do not add test carveouts such as `allow-unwrap-in-tests`, `allow-expect-in-tests`,
  `allow-panic-in-tests`, `allow-indexing-slicing-in-tests`, or `allow-dbg-in-tests`.
- Prefer panic-free tests that return `Result<(), Box<dyn std::error::Error>>` and use `?`.
- Use `#[expect(..., reason = "...")]` for narrow suppressions. Do not use silent `#[allow]`.
- Track temporary rollout debt in `policy/clippy-debt.toml`; debt must have owner, reason, path,
  lint, and expiry.

## Policy ledgers

- `policy/clippy-lints.toml` records the MSRV, policy flags, and planned Rust 1.94/1.95 flips.
- `policy/clippy-debt.toml` records temporary lint exceptions.
- `policy/no-panic-allowlist.toml` documents the semantic allowlist schema for panic-family checks.
- `policy/non-rust-allowlist.toml` documents the structured allowlist schema for non-Rust surfaces.

## Suppression example

```rust
#[expect(
    clippy::indexing_slicing,
    reason = "kernel tile bounds are proven by dispatch shape validation"
)]
fn load_tile(xs: &[f32], lane: usize) -> f32 {
    xs[lane]
}
```

If the suppression represents migration debt rather than a permanent invariant, add a matching
entry to `policy/clippy-debt.toml`.

## Validation

Run:

```sh
cargo xtask check-lint-policy
```

The gate verifies that the root Cargo lint block, policy ledger, `clippy.toml`, workspace-member
lint inheritance, planned upgrade flips, and debt metadata remain coherent.
