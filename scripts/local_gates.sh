#!/usr/bin/env bash
# Local quality gates script for BitNet.rs CPU MVP
#
# Runs a comprehensive suite of checks before committing code:
# - Format check
# - Clippy lints
# - CPU test suite
# - Tiny benchmark (writes ci/inference.json receipt)
# - Receipt verification
#
# This script ensures local changes meet the same standards as CI.

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     BitNet.rs Local Quality Gates         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo ""

# Track overall status
FAILED=0

# 1. Format check
echo -e "${YELLOW}[1/5]${NC} Running format check..."
if cargo fmt --all -- --check; then
    echo -e "${GREEN}✓${NC} Format check passed"
else
    echo -e "${RED}✗${NC} Format check failed"
    FAILED=1
fi
echo ""

# 2. Clippy
echo -e "${YELLOW}[2/5]${NC} Running clippy..."
if cargo clippy --workspace --all-targets --no-default-features --features cpu -- -D warnings; then
    echo -e "${GREEN}✓${NC} Clippy passed"
else
    echo -e "${RED}✗${NC} Clippy failed"
    FAILED=1
fi
echo ""

# 3. CPU tests
echo -e "${YELLOW}[3/5]${NC} Running CPU test suite..."
if cargo test --workspace --no-default-features --features cpu; then
    echo -e "${GREEN}✓${NC} CPU tests passed"
else
    echo -e "${RED}✗${NC} CPU tests failed"
    FAILED=1
fi
echo ""

# 4. Tiny benchmark (writes receipt)
# Note: This is a placeholder - the actual benchmark command needs to be
# implemented to write ci/inference.json with proper receipt data
echo -e "${YELLOW}[4/5]${NC} Running tiny benchmark..."
echo -e "${BLUE}ℹ${NC}  Skipping benchmark (not yet implemented)"
echo -e "${BLUE}ℹ${NC}  TODO: Implement 'cargo run -p xtask -- benchmark --model tests/models/tiny.gguf --tokens 128 --deterministic'"
# Uncomment when benchmark is ready:
# if cargo run -p xtask -- benchmark --model tests/models/tiny.gguf --tokens 128 --deterministic; then
#     echo -e "${GREEN}✓${NC} Benchmark passed"
# else
#     echo -e "${RED}✗${NC} Benchmark failed"
#     FAILED=1
# fi
echo ""

# 5. Verify receipt
echo -e "${YELLOW}[5/5]${NC} Verifying receipt..."
# Skip if receipt doesn't exist (benchmark not yet implemented)
if [ -f "ci/inference.json" ]; then
    if cargo run -p xtask -- verify-receipt; then
        echo -e "${GREEN}✓${NC} Receipt verification passed"
    else
        echo -e "${RED}✗${NC} Receipt verification failed"
        FAILED=1
    fi
else
    echo -e "${BLUE}ℹ${NC}  Skipping receipt verification (ci/inference.json not found)"
    echo -e "${BLUE}ℹ${NC}  This will be required once benchmark is implemented"
fi
echo ""

# Summary
echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
if [ $FAILED -eq 0 ]; then
    echo -e "${BLUE}║${NC}  ${GREEN}✓ All local quality gates passed!${NC}       ${BLUE}║${NC}"
else
    echo -e "${BLUE}║${NC}  ${RED}✗ Some quality gates failed${NC}            ${BLUE}║${NC}"
fi
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"

exit $FAILED
