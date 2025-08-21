#!/usr/bin/env bash
# Comprehensive validation suite for BitNet.rs vs bitnet.cpp
# This script runs exhaustive tests across accuracy, performance, and compatibility

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}     BitNet.rs Comprehensive Validation Suite${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
echo

# Configuration
MODELS_DIR="${MODELS_DIR:-models}"
RESULTS_DIR="${RESULTS_DIR:-validation_results}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="${RESULTS_DIR}/${TIMESTAMP}"

# Create results directory
mkdir -p "${REPORT_DIR}"

# Test matrix
declare -A MODELS=(
    ["tinyllama"]="models/tinyllama-q2.gguf"
    ["microsoft_bitnet"]="models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf"
    ["synthetic_v3"]="target/test_v3.gguf"
    ["synthetic_v2"]="target/test_v2.gguf"
)

# Validation stages
STAGES=(
    "model_loading"
    "accuracy"
    "performance"
    "memory"
    "determinism"
    "edge_cases"
    "compatibility"
)

# Function to run validation for a single model
validate_model() {
    local model_name="$1"
    local model_path="$2"
    
    echo -e "\n${BLUE}▶ Testing: ${model_name}${NC}"
    echo -e "  Path: ${model_path}"
    
    if [ ! -f "${model_path}" ]; then
        echo -e "  ${YELLOW}⚠ Model not found, skipping${NC}"
        return 1
    fi
    
    local report_file="${REPORT_DIR}/${model_name}_report.json"
    local log_file="${REPORT_DIR}/${model_name}.log"
    
    # Set deterministic environment
    export RAYON_NUM_THREADS=1
    export BITNET_DETERMINISTIC=1
    export BITNET_SEED=42
    export OMP_NUM_THREADS=1
    
    # Run validation with xtask
    echo -e "  ${CYAN}Running validation...${NC}"
    
    if cargo run -p xtask --release -- validate \
        --model "${model_path}" \
        --output "${report_file}" \
        --stages "${STAGES[@]}" \
        > "${log_file}" 2>&1; then
        
        echo -e "  ${GREEN}✓ Validation completed${NC}"
        
        # Extract key metrics from report
        if [ -f "${report_file}" ]; then
            local status=$(jq -r '.status' "${report_file}")
            local rust_ok=$(jq -r '.rust_load.success' "${report_file}")
            local cpp_ok=$(jq -r '.cpp_load.success' "${report_file}")
            local accuracy=$(jq -r '.accuracy.token_match_rate' "${report_file}")
            local speedup=$(jq -r '.performance.speedup_factor' "${report_file}")
            
            echo -e "  ${CYAN}Results:${NC}"
            echo -e "    Status: $(format_status ${status})"
            echo -e "    Rust Load: $(format_bool ${rust_ok})"
            echo -e "    C++ Load: $(format_bool ${cpp_ok})"
            
            if [ "${accuracy}" != "null" ]; then
                echo -e "    Accuracy: $(printf "%.2f%%" $(echo "${accuracy} * 100" | bc -l))"
            fi
            
            if [ "${speedup}" != "null" ]; then
                echo -e "    Speedup: $(printf "%.2fx" ${speedup})"
            fi
        fi
    else
        echo -e "  ${RED}✗ Validation failed${NC}"
        echo -e "  Check log: ${log_file}"
    fi
}

# Helper functions
format_status() {
    case "$1" in
        "Pass") echo -e "${GREEN}✅ Pass${NC}" ;;
        "PartialPass") echo -e "${YELLOW}⚠️ Partial Pass${NC}" ;;
        "XFail") echo -e "${BLUE}🔧 XFail${NC}" ;;
        "Fail") echo -e "${RED}❌ Fail${NC}" ;;
        *) echo "$1" ;;
    esac
}

format_bool() {
    if [ "$1" = "true" ]; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi
}

# Main validation loop
echo -e "${CYAN}Starting validation suite...${NC}"
echo -e "Results will be saved to: ${REPORT_DIR}"

# Phase 1: Build everything
echo -e "\n${BLUE}═══ Phase 1: Building Components ═══${NC}"
cargo build --release --no-default-features --features cpu
cargo build -p xtask --release

# Phase 2: Generate synthetic test models
echo -e "\n${BLUE}═══ Phase 2: Generating Test Models ═══${NC}"
cargo run -p xtask --release -- generate-test-models

# Phase 3: Validate each model
echo -e "\n${BLUE}═══ Phase 3: Model Validation ═══${NC}"

for model_name in "${!MODELS[@]}"; do
    validate_model "${model_name}" "${MODELS[${model_name}]}"
done

# Phase 4: Generate consolidated report
echo -e "\n${BLUE}═══ Phase 4: Generating Consolidated Report ═══${NC}"

CONSOLIDATED_REPORT="${REPORT_DIR}/consolidated_report.json"
MARKDOWN_REPORT="${REPORT_DIR}/validation_report.md"

# Merge all individual reports
jq -s '.' "${REPORT_DIR}"/*_report.json > "${CONSOLIDATED_REPORT}" 2>/dev/null || true

# Generate markdown summary
cat > "${MARKDOWN_REPORT}" << EOF
# BitNet.rs Validation Report
**Date**: $(date)
**Commit**: $(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

## Summary

| Model | Status | Rust | C++ | Accuracy | Speedup |
|-------|--------|------|-----|----------|---------|
EOF

for model_name in "${!MODELS[@]}"; do
    report_file="${REPORT_DIR}/${model_name}_report.json"
    if [ -f "${report_file}" ]; then
        status=$(jq -r '.status' "${report_file}")
        rust_ok=$(jq -r '.rust_load.success' "${report_file}")
        cpp_ok=$(jq -r '.cpp_load.success' "${report_file}")
        accuracy=$(jq -r '.accuracy.token_match_rate // "N/A"' "${report_file}")
        speedup=$(jq -r '.performance.speedup_factor // "N/A"' "${report_file}")
        
        if [ "${accuracy}" != "N/A" ] && [ "${accuracy}" != "null" ]; then
            accuracy=$(printf "%.2f%%" $(echo "${accuracy} * 100" | bc -l))
        fi
        
        if [ "${speedup}" != "N/A" ] && [ "${speedup}" != "null" ]; then
            speedup=$(printf "%.2fx" ${speedup})
        fi
        
        echo "| ${model_name} | ${status} | ${rust_ok} | ${cpp_ok} | ${accuracy} | ${speedup} |" >> "${MARKDOWN_REPORT}"
    fi
done

echo "" >> "${MARKDOWN_REPORT}"
echo "## Detailed Reports" >> "${MARKDOWN_REPORT}"
echo "Individual reports available in: \`${REPORT_DIR}\`" >> "${MARKDOWN_REPORT}"

# Phase 5: CI Integration Check
echo -e "\n${BLUE}═══ Phase 5: CI Integration ═══${NC}"

# Check if we should upload to CI
if [ -n "${CI:-}" ]; then
    echo -e "${CYAN}CI environment detected, preparing artifacts...${NC}"
    
    # Create artifact bundle
    tar -czf "validation_results_${TIMESTAMP}.tar.gz" -C "${RESULTS_DIR}" "${TIMESTAMP}"
    
    echo -e "${GREEN}✓ Artifacts prepared for upload${NC}"
fi

# Final summary
echo -e "\n${CYAN}═══════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}                    Validation Complete${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"

# Count results
TOTAL_MODELS=${#MODELS[@]}
PASSED_MODELS=$(jq -s '[.[] | select(.status == "Pass")] | length' "${REPORT_DIR}"/*_report.json 2>/dev/null || echo 0)
PARTIAL_MODELS=$(jq -s '[.[] | select(.status == "PartialPass")] | length' "${REPORT_DIR}"/*_report.json 2>/dev/null || echo 0)
XFAIL_MODELS=$(jq -s '[.[] | select(.status == "XFail")] | length' "${REPORT_DIR}"/*_report.json 2>/dev/null || echo 0)

echo -e "\nResults Summary:"
echo -e "  Total Models: ${TOTAL_MODELS}"
echo -e "  ${GREEN}Passed: ${PASSED_MODELS}${NC}"
echo -e "  ${YELLOW}Partial: ${PARTIAL_MODELS}${NC}"
echo -e "  ${BLUE}XFail: ${XFAIL_MODELS}${NC}"
echo -e "\nFull report: ${MARKDOWN_REPORT}"

# Exit with appropriate code
if [ "${PASSED_MODELS}" -ge 1 ]; then
    exit 0
else
    exit 1
fi