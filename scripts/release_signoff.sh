#!/usr/bin/env bash
# Production release sign-off script
# This script performs all integrity checks before cutting a release
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo -e "${BLUE}       BitNet.rs Release Sign-Off v1.0              ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo

# Track failures
FAILURES=()
WARNINGS=()

# Function to check a condition
check() {
    local name="$1"
    local cmd="$2"
    echo -n "  Checking: $name... "
    if eval "$cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        return 0
    else
        echo -e "${RED}✗${NC}"
        FAILURES+=("$name")
        return 1
    fi
}

# Function to warn
warn() {
    local name="$1"
    local cmd="$2"
    echo -n "  Warning: $name... "
    if eval "$cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠${NC}"
        WARNINGS+=("$name")
        return 1
    fi
}

# Setup deterministic environment
echo -e "${BLUE}[1/6] Environment Setup${NC}"
setup_deterministic_env
print_platform_banner

# Check for mock features in critical files
echo -e "\n${BLUE}[2/6] Integrity Checks${NC}"
echo "[Check] No mock feature usage in build/test/run commands"
MOCK_HITS=$(
  git grep -n -E \
    '(^|[[:space:]])cargo[[:space:]]+(build|test|run)([[:space:][:graph:]]*?)--features([[:space:]=]+)[^#\n]*\bmocks\b' \
    -- scripts/ crates/ Cargo.* Makefile* 2>/dev/null \
  | grep -v -E '\.github/|grep[[:space:]].*--features[[:space:]=]+mocks' \
  || true
)
if [ -n "$MOCK_HITS" ]; then
  echo "ERROR: Found real invocations using '--features mocks':"
  echo "$MOCK_HITS"
  exit 1
fi
echo "OK: no mock feature usage in cargo commands."

check "No debug prints in code" \
    "! git grep -q 'dbg!' -- crates --include='*.rs'"

check "No TODO comments in critical paths" \
    "! git grep -q 'TODO\|FIXME\|XXX' -- crates/bitnet-inference crates/bitnet-models"

check "All fast/smoke tests pass" \
    "cargo test --workspace --no-default-features --features cpu --quiet -- --ignored"

# Performance artifacts exist
echo -e "\n${BLUE}[3/6] Performance Artifacts${NC}"
check "SafeTensors benchmark exists" \
    "ls bench/results/*-safetensors.json 2>/dev/null | head -1"

check "GGUF benchmark exists" \
    "ls bench/results/*-gguf.json 2>/dev/null | head -1"

check "Performance comparison rendered" \
    "test -f docs/PERF_COMPARISON.md"

# Verify JSON metadata
echo -e "\n${BLUE}[4/6] Metadata Validation${NC}"
if ls bench/results/*.json > /dev/null 2>&1; then
    for json in bench/results/*.json; do
        name=$(basename "$json")
        check "Valid JSON: $name" \
            "jq empty < '$json'"
        
        check "Has platform stamp: $name" \
            "jq -e '.platform' '$json' > /dev/null"
        
        check "Has WSL2 flag: $name" \
            "jq -e '.wsl2 != null' '$json' > /dev/null"
    done
fi

# Documentation consistency
echo -e "\n${BLUE}[5/6] Documentation${NC}"
check "README exists" \
    "test -f README.md"

check "CLAUDE.md exists" \
    "test -f CLAUDE.md"

check "COMPATIBILITY.md exists" \
    "test -f COMPATIBILITY.md"

check "PRODUCTION_READINESS.md exists" \
    "test -f docs/PRODUCTION_READINESS.md"

warn "Changelog updated" \
    "git diff HEAD^ -- CHANGELOG.md | grep -q '^+'"

# Version consistency
echo -e "\n${BLUE}[6/6] Version Checks${NC}"
ROOT_VERSION=$(grep '^version' Cargo.toml | head -1 | cut -d'"' -f2)
echo "  Root package version: $ROOT_VERSION"

for crate in crates/*/Cargo.toml; do
    crate_name=$(basename $(dirname "$crate"))
    version=$(grep '^version' "$crate" | head -1 | cut -d'"' -f2)
    if [ "$version" != "$ROOT_VERSION" ]; then
        warn "Version mismatch: $crate_name ($version != $ROOT_VERSION)" "false"
    fi
done

# Git status
echo -e "\n${BLUE}Git Status${NC}"
if [ -n "$(git status --porcelain)" ]; then
    echo -e "  ${YELLOW}⚠ Working directory has uncommitted changes${NC}"
    WARNINGS+=("Uncommitted changes")
fi

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "  Current branch: $CURRENT_BRANCH"
echo "  Latest commit: $(git rev-parse --short HEAD) - $(git log -1 --pretty=%s)"

# Summary
echo
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo -e "${BLUE}                    SUMMARY                         ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"

if [ ${#FAILURES[@]} -eq 0 ] && [ ${#WARNINGS[@]} -eq 0 ]; then
    echo -e "${GREEN}✅ ALL CHECKS PASSED!${NC}"
    echo -e "${GREEN}Ready for release v$ROOT_VERSION${NC}"
    echo
    echo "Next steps:"
    echo "  1. Run: ./scripts/acceptance_test.sh"
    echo "  2. Run: ./scripts/preserve_ci_artifacts.sh"
    echo "  3. Tag: git tag -a v$ROOT_VERSION -m 'Release v$ROOT_VERSION'"
    echo "  4. Push: git push --tags"
    exit 0
else
    if [ ${#FAILURES[@]} -gt 0 ]; then
        echo -e "${RED}❌ FAILURES (${#FAILURES[@]}):${NC}"
        for failure in "${FAILURES[@]}"; do
            echo "  - $failure"
        done
    fi
    
    if [ ${#WARNINGS[@]} -gt 0 ]; then
        echo -e "${YELLOW}⚠️  WARNINGS (${#WARNINGS[@]}):${NC}"
        for warning in "${WARNINGS[@]}"; do
            echo "  - $warning"
        done
    fi
    
    echo
    echo -e "${RED}Release blocked. Fix failures before proceeding.${NC}"
    exit 1
fi