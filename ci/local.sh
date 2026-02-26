#!/usr/bin/env bash
# ci/local.sh - Local CI reproduction script
# Run this locally to replicate what CI core checks do before pushing
#
# Usage: ./ci/local.sh
#
# This script matches the 4 required CI gates:
#   - Build & Test (ubuntu-latest)
#   - Clippy
#   - Documentation
#   - CI Core Success

set -euo pipefail

export RUSTFLAGS="-Dwarnings"
export CARGO_TERM_COLOR=always

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=== BitNet-rs CI Core Checks (Local) ==="
echo ""
echo "This replicates the 4 required GitHub CI gates:"
echo "  1. Format check"
echo "  2. Clippy (core packages)"
echo "  3. Build (core packages)"
echo "  4. Tests (core packages, lib only)"
echo ""

# Core packages that must pass (matches CI core checks)
CORE_PKGS=(
    "bitnet-common"
    "bitnet-models"
    "bitnet-tokenizers"
    "bitnet-quantization"
    "bitnet-kernels"
)

PKG_FLAGS=$(printf " -p %s" "${CORE_PKGS[@]}")

echo "Core packages: ${CORE_PKGS[*]}"
echo ""

FAILED=0

run_check() {
    local name="$1"
    shift
    echo -e "${BLUE}▶ ${name}${NC}"
    if "$@"; then
        echo -e "${GREEN}✓ ${name} passed${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}✗ ${name} failed${NC}"
        echo ""
        FAILED=1
        return 1
    fi
}

# 0) Toolchain sanity check
echo -e "${BLUE}▶ Checking Rust toolchain${NC}"
rustup show active-toolchain
cargo --version
echo ""

# 1) Format check
run_check "Format check" \
    cargo fmt --all -- --check

# 2) Clippy (core packages only, matches CI)
run_check "Clippy (core packages)" \
    cargo clippy --locked \
        $PKG_FLAGS \
        --all-targets --no-default-features --features cpu -- -D warnings

# 3) Build (core packages only, matches CI)
run_check "Build (core packages)" \
    cargo build --locked \
        $PKG_FLAGS \
        --no-default-features --features cpu

# 4) Tests (core packages, lib only - matches CI)
run_check "Tests (core packages, lib only)" \
    cargo test --locked \
        $PKG_FLAGS \
        --lib --no-default-features --features cpu -- --nocapture

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All CI core checks passed!${NC}"
    echo ""
    echo "Your changes should pass the 4 required GitHub CI gates."
    echo ""
    echo "Note: This script only runs core required checks (fast ~5-7 min)."
    echo "Full CI also includes optional/label-gated lanes:"
    echo "  - Coverage (label: coverage)"
    echo "  - Integration tests (label: framework)"
    echo "  - Cross-validation (label: crossval)"
    echo "  - TL LUT stress (label: lut)"
    echo "  - GPU tests (label: gpu)"
    echo "  - Quantization matrix (label: quant)"
    echo ""
    exit 0
else
    echo -e "${RED}❌ Some checks failed${NC}"
    echo ""
    echo "Fix the issues above before pushing to avoid CI failures."
    echo ""
    exit 1
fi
