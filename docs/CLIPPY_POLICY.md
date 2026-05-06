# Clippy policy

BitNet-rs uses an Effortless Metrics lint policy: one workspace-level lint surface,
explicit repo-local debt, and machine-readable upgrade tracking. The goal is to make
panic-free and silent-failure-free Rust the default for production code and tests
without hiding GPU/numeric cleanup behind broad carveouts.

## Active baseline

The root `Cargo.toml` owns the active `[workspace.lints.rust]` and
`[workspace.lints.clippy]` policy. Workspace crates inherit it with:

```toml
[lints]
workspace = true
```

The active policy warns on existing unsafe-code debt while tracking the intended forbid ratchet in `policy/clippy-lints.toml`; it denies the panic family (`unwrap`, `expect`, `panic!`, `todo!`,
`unimplemented!`, `unreachable!`), silent-failure patterns (`let _ =` on futures,
locks, and must-use values), AST/string/indexing hazards, suppression footguns, and
file/process/path mistakes. Numeric lints are staged for BitNet's kernel-heavy
surface: reviewed correctness lints are denied, while high-churn cast and arithmetic
lints start as warnings.

## No test carveouts

`clippy.toml` must not contain test carveouts such as:

```toml
allow-unwrap-in-tests = true
allow-expect-in-tests = true
allow-panic-in-tests = true
allow-indexing-slicing-in-tests = true
allow-dbg-in-tests = true
```

Tests should return `Result` or use assertion helpers with useful diagnostics instead
of relying on panic-driven setup.

## Suppression style

Prefer fixing the lint. When a local exception is unavoidable, use a narrow
`#[expect(..., reason = "...")]` with a human-readable reason. Do not add broad
`#[allow(...)]` attributes or config-level weakenings. Existing broad suppressions are
legacy debt and should be migrated in follow-up cleanup PRs.

## Policy ledgers

- `policy/clippy-lints.toml` records the active policy, MSRV, and planned Rust 1.94 /
  1.95 flips.
- `policy/clippy-debt.toml` records temporary repo-local debt with owner, reason,
  path, lint, and expiry.
- `policy/no-panic-allowlist.toml` is reserved for semantic panic-family receipts
  keyed by path + family + selector, with advisory `last_seen` locations.
- `policy/non-rust-allowlist.toml` records why non-Rust programming/config surfaces
  exist and what command covers them.

## Gate

Run:

```sh
cargo xtask check-lint-policy
```

The gate verifies that Cargo, Clippy, and policy ledger metadata remain coherent. It
also fails expired debt. As this rollout proceeds through stacked PRs, the gate is the
place to ratchet additional checks from advisory reporting to blocking enforcement.
