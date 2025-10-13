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

printf "%b\n" "${BLUE}╔════════════════════════════════════════════╗${NC}"
printf "%b\n" "${BLUE}║     BitNet.rs Local Quality Gates         ║${NC}"
printf "%b\n" "${BLUE}╚════════════════════════════════════════════╝${NC}"
printf "\n"

# Track overall status
FAILED=0

# 1. Format check
printf "%b\n" "${YELLOW}[1/5]${NC} Running format check..."
if cargo fmt --all -- --check; then
    printf "%b\n" "${GREEN}✓${NC} Format check passed"
else
    printf "%b\n" "${RED}✗${NC} Format check failed"
    FAILED=1
fi
printf "\n"

# 2. Clippy
printf "%b\n" "${YELLOW}[2/5]${NC} Running clippy..."
if cargo clippy --workspace --all-targets --no-default-features --features cpu -- -D warnings; then
    printf "%b\n" "${GREEN}✓${NC} Clippy passed"
else
    printf "%b\n" "${RED}✗${NC} Clippy failed"
    FAILED=1
fi
printf "\n"

# 3. CPU tests
printf "%b\n" "${YELLOW}[3/5]${NC} Running CPU test suite..."
if cargo test --workspace --no-default-features --features cpu; then
    printf "%b\n" "${GREEN}✓${NC} CPU tests passed"
else
    printf "%b\n" "${RED}✗${NC} CPU tests failed"
    FAILED=1
fi
printf "\n"

# 4. Tiny benchmark (writes receipt)
# Note: This is a placeholder - the actual benchmark command needs to be
# implemented to write ci/inference.json with proper receipt data
printf "%b\n" "${YELLOW}[4/5]${NC} Running tiny benchmark..."
printf "%b\n" "${BLUE}ℹ${NC}  Skipping benchmark (not yet implemented)"
printf "%b\n" "${BLUE}ℹ${NC}  TODO: Implement 'cargo run -p xtask -- benchmark --model tests/models/tiny.gguf --tokens 128 --deterministic'"
# Uncomment when benchmark is ready:
# if cargo run -p xtask -- benchmark --model tests/models/tiny.gguf --tokens 128 --deterministic; then
#     printf "%b\n" "${GREEN}✓${NC} Benchmark passed"
# else
#     printf "%b\n" "${RED}✗${NC} Benchmark failed"
#     FAILED=1
# fi
printf "\n"

# 5. Verify receipt
printf "%b\n" "${YELLOW}[5/5]${NC} Verifying receipt..."
# Skip if receipt doesn't exist (benchmark not yet implemented)
if [ -f "ci/inference.json" ]; then
    if cargo run -p xtask -- verify-receipt; then
        printf "%b\n" "${GREEN}✓${NC} Receipt verification passed"
    else
        printf "%b\n" "${RED}✗${NC} Receipt verification failed"
        FAILED=1
    fi
else
    printf "%b\n" "${BLUE}ℹ${NC}  Skipping receipt verification (ci/inference.json not found)"
    printf "%b\n" "${BLUE}ℹ${NC}  This will be required once benchmark is implemented"
fi
printf "\n"

# Summary
printf "%b\n" "${BLUE}╔════════════════════════════════════════════╗${NC}"
if [ "$FAILED" -eq 0 ]; then
    printf "%b\n" "${BLUE}║${NC}  ${GREEN}✓ All local quality gates passed!${NC}       ${BLUE}║${NC}"
else
    printf "%b\n" "${BLUE}║${NC}  ${RED}✗ Some quality gates failed${NC}            ${BLUE}║${NC}"
fi
printf "%b\n" "${BLUE}╚════════════════════════════════════════════╝${NC}"

exit "$FAILED"
