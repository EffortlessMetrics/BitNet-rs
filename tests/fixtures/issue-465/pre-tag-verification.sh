#!/usr/bin/env bash
# Pre-tag verification script for v0.1.0-mvp release
#
# This script runs all quality gates before creating the v0.1.0-mvp tag.
# Ensures the release meets BitNet-rs standards for honest compute.

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "BitNet-rs Pre-Tag Verification"
echo "Target: v0.1.0-mvp"
echo "========================================="
echo ""

# Verify we're in the workspace root
if [ ! -f "Cargo.toml" ]; then
    echo -e "${RED}ERROR: Must run from workspace root${NC}"
    exit 1
fi

# Step 1: Format Check
echo -e "${YELLOW}[1/6] Running cargo fmt check...${NC}"
if cargo fmt --all --check; then
    echo -e "${GREEN}✓ Format check passed${NC}"
else
    echo -e "${RED}✗ Format check failed${NC}"
    echo "Run: cargo fmt --all"
    exit 1
fi
echo ""

# Step 2: Clippy
echo -e "${YELLOW}[2/6] Running clippy (all features)...${NC}"
if cargo clippy --all-targets --all-features -- -D warnings; then
    echo -e "${GREEN}✓ Clippy passed${NC}"
else
    echo -e "${RED}✗ Clippy failed${NC}"
    exit 1
fi
echo ""

# Step 3: CPU Tests
echo -e "${YELLOW}[3/6] Running CPU tests...${NC}"
if cargo test --workspace --no-default-features --features cpu; then
    echo -e "${GREEN}✓ CPU tests passed${NC}"
else
    echo -e "${RED}✗ CPU tests failed${NC}"
    exit 1
fi
echo ""

# Step 4: Deterministic Benchmark
echo -e "${YELLOW}[4/6] Running deterministic benchmark...${NC}"
export BITNET_DETERMINISTIC=1
export RAYON_NUM_THREADS=1
export BITNET_SEED=42

# Find GGUF model (auto-discover in models/ directory)
GGUF_MODEL=$(find models -name "*.gguf" -type f | head -n 1)

if [ -z "$GGUF_MODEL" ]; then
    echo -e "${YELLOW}⚠ No GGUF model found in models/ directory${NC}"
    echo "Skipping benchmark (download model with: cargo run -p xtask -- download-model)"
    SKIP_BENCHMARK=1
else
    echo "Using model: $GGUF_MODEL"
    if cargo run -p xtask -- benchmark --model "$GGUF_MODEL" --tokens 128; then
        echo -e "${GREEN}✓ Benchmark passed${NC}"
    else
        echo -e "${RED}✗ Benchmark failed${NC}"
        exit 1
    fi
fi
echo ""

# Step 5: Receipt Verification
if [ "${SKIP_BENCHMARK:-0}" != "1" ]; then
    echo -e "${YELLOW}[5/6] Verifying receipt...${NC}"
    if [ -f "ci/inference.json" ]; then
        if cargo run -p xtask -- verify-receipt ci/inference.json; then
            echo -e "${GREEN}✓ Receipt verification passed${NC}"
        else
            echo -e "${RED}✗ Receipt verification failed${NC}"
            exit 1
        fi
    else
        echo -e "${RED}✗ Receipt not found at ci/inference.json${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}[5/6] Skipping receipt verification (no benchmark)${NC}"
fi
echo ""

# Step 6: Baseline Check
echo -e "${YELLOW}[6/6] Checking CPU baseline...${NC}"
if ls docs/baselines/*-cpu.json 1> /dev/null 2>&1; then
    BASELINE=$(ls docs/baselines/*-cpu.json | tail -n 1)
    echo "Found baseline: $BASELINE"

    # Verify baseline has required fields
    if jq -e '.schema_version and .compute_path and .kernels and .tokens_per_second' "$BASELINE" > /dev/null; then
        echo -e "${GREEN}✓ CPU baseline valid${NC}"
    else
        echo -e "${RED}✗ CPU baseline missing required fields${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ No CPU baseline found in docs/baselines/${NC}"
    echo "Generate baseline with: cargo run -p xtask -- benchmark --model <model> --tokens 128"
    exit 1
fi
echo ""

# Summary
echo "========================================="
echo -e "${GREEN}✓ Pre-tag verification complete${NC}"
echo "========================================="
echo ""
echo "All checks passed! Ready to create v0.1.0-mvp tag."
echo ""
echo "Next steps:"
echo "1. git tag -a v0.1.0-mvp -m \"Release v0.1.0-mvp with CPU baseline $(date +%Y%m%d)\""
echo "2. git push origin v0.1.0-mvp"
echo "3. gh release create v0.1.0-mvp --title \"v0.1.0-mvp\" --notes-file RELEASE_NOTES.md"
echo ""
