#!/usr/bin/env bash
# One-command quality gate - runs all local checks before committing
set -euo pipefail

echo "🔍 Running BitNet-rs quality gate..."
echo ""

echo "📝 Formatting code..."
cargo fmt --all

echo ""
echo "🔎 Running clippy with all features..."
RUSTFLAGS="-Dwarnings" cargo clippy --workspace --all-features --all-targets -- -D warnings -D clippy::ptr_arg

echo ""
echo "✓ Checking tests compile (CPU only)..."
RUSTFLAGS="-Dwarnings" cargo check --workspace --tests --no-default-features --features cpu

echo ""
echo "🔒 Running dependency security audit..."
cargo deny check --hide-inclusion-graph

echo ""
echo "🚫 Checking for banned patterns..."
bash scripts/hooks/banned-patterns.sh

echo ""
echo "✅ All quality checks passed!"