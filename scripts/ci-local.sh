#!/usr/bin/env bash
set -euo pipefail

echo "== Clean =="
cargo clean

echo "== Build & Test (strict code lints) =="
RUSTFLAGS="-D warnings" \
  cargo build --locked --workspace --no-default-features --features cpu
RUSTFLAGS="-D warnings" \
  cargo test  --locked --workspace --no-default-features --features cpu --lib

echo "== Clippy (strict) =="
cargo clippy --workspace --all-targets --no-default-features --features cpu \
  -- -D warnings

echo "== Format check =="
cargo fmt --all -- --check

echo "== Docs (relaxed rustdoc) =="
# Important: only relax rustdoc; keep code builds strict.
RUSTDOCFLAGS="-A warnings" \
  cargo doc --locked --no-deps --workspace --no-default-features --features cpu

echo "== MSRV check (1.89.0) =="
rustup toolchain install 1.89.0 -q || true
cargo +1.89.0 check --workspace --all-targets --locked \
  --no-default-features --features cpu

echo "✅ All local checks passed."
