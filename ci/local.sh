#!/usr/bin/env bash
# ci/local.sh - Local CI reproduction script
# Run this locally to replicate what CI checks do before pushing
#
# Usage: ./ci/local.sh

set -euo pipefail

export RUSTFLAGS="-Dwarnings"
export CARGO_TERM_COLOR=always

echo "=== BitNet.rs CI Local Checks ==="
echo

# 0) Toolchain sanity check
echo "→ Checking Rust toolchain..."
rustup show active-toolchain
cargo --version
echo

# 1) Format check
echo "→ Running cargo fmt..."
cargo fmt --all -- --check
echo "✓ Format OK"
echo

# 2) Clippy (CPU features, all targets)
echo "→ Running clippy..."
cargo clippy --workspace --all-targets --no-default-features --features cpu -- -D warnings
echo "✓ Clippy OK"
echo

# 3) Build & test (CPU lane)
echo "→ Building workspace..."
cargo build --locked --workspace --no-default-features --features cpu
echo "✓ Build OK"
echo

echo "→ Running tests..."
cargo test --locked --workspace --no-default-features --features cpu -- --nocapture
echo "✓ Tests OK"
echo

# 4) Docs
echo "→ Building documentation..."
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --workspace
echo "✓ Docs OK"
echo

echo "=== ✓ All CI checks passed locally ==="
