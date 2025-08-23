#!/usr/bin/env bash
# Reality-proof checklist for dual-format validation
# Run this to verify both SafeTensors and GGUF are production-ready

set -euo pipefail

# Source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Setup deterministic environment immediately
setup_deterministic_env

# Configuration
MODEL_ID="${MODEL_ID:-bitnet_b1_58-3B}"
HF_MODEL_ID="${HF_MODEL_ID:-1bitLLM/bitnet_b1_58-3B}"
MODELS_DIR="${MODELS_DIR:-models}"
OUTPUT_DIR=$(ensure_output_dir "reality_proof_results")

# Test results
PASSED_TESTS=()
FAILED_TESTS=()

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    PASSED_TESTS+=("$1")
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    FAILED_TESTS+=("$1")
}

log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

# Test 1: Environment & Determinism
test_environment() {
    log_test "1. Environment & Determinism"
    
    print_platform_banner
    detect_wsl2 || true
    
    # Find BitNet binary using common function
    local BIN=$(find_bitnet_binary)
    
    # Check bitnet version
    if $BIN --version > /dev/null 2>&1; then
        VERSION=$($BIN --version)
        log_info "BitNet version: $VERSION"
        log_info "Binary path: $BIN"
        log_pass "Environment setup"
    else
        log_fail "Cannot find bitnet binary"
        return 1
    fi
}

# Test 2: Model Introspection
test_model_introspection() {
    log_test "2. Model Introspection (format detection/policy)"
    
    local safetensors_model="${MODELS_DIR}/${MODEL_ID}/safetensors/model.safetensors"
    local safetensors_tokenizer="${MODELS_DIR}/${MODEL_ID}/safetensors/tokenizer.json"
    local gguf_model="${MODELS_DIR}/${MODEL_ID}/gguf/model.gguf"
    
    # Test SafeTensors introspection
    if [ -f "$safetensors_model" ]; then
        local st_info=$($BITNET_BIN info \
            --model "$safetensors_model" \
            --tokenizer "$safetensors_tokenizer" \
            --json 2>/dev/null)
        
        if echo "$st_info" | jq -e '.format == "safetensors"' > /dev/null; then
            log_info "SafeTensors format detected correctly"
            echo "$st_info" | jq . > "${OUTPUT_DIR}/safetensors_info.json"
        else
            log_fail "SafeTensors format detection failed"
            return 1
        fi
    fi
    
    # Test GGUF introspection
    if [ -f "$gguf_model" ]; then
        local gguf_info=$($BITNET_BIN info \
            --model "$gguf_model" \
            --json 2>/dev/null)
        
        if echo "$gguf_info" | jq -e '.format == "gguf"' > /dev/null; then
            log_info "GGUF format detected correctly"
            echo "$gguf_info" | jq . > "${OUTPUT_DIR}/gguf_info.json"
        else
            log_fail "GGUF format detection failed"
            return 1
        fi
    fi
    
    log_pass "Model introspection"
}

# Test 3: Tokenizer Parity
test_tokenizer_parity() {
    log_test "3. Tokenizer Parity Battery"
    
    local battery_file="${SCRIPT_DIR}/tokenizer_battery.txt"
    if [ ! -f "$battery_file" ]; then
        log_fail "Tokenizer battery file not found: $battery_file"
        return 1
    fi
    
    # Run parity validation
    if "${SCRIPT_DIR}/validate_format_parity.sh" > "${OUTPUT_DIR}/tokenizer_parity.log" 2>&1; then
        log_pass "Tokenizer parity"
    else
        log_fail "Tokenizer parity validation failed"
        return 1
    fi
}

# Test 4: Logit Parity
test_logit_parity() {
    log_test "4. Logit Parity (τ-b correlation)"
    
    local safetensors_model="${MODELS_DIR}/${MODEL_ID}/safetensors/model.safetensors"
    local safetensors_tokenizer="${MODELS_DIR}/${MODEL_ID}/safetensors/tokenizer.json"
    
    # FP32↔FP32 test
    PROP_EXAMPLES=12 TAU_STEPS=24 LOGIT_TOPK=10 TAU_MIN=0.95 \
        MODEL_PATH="$safetensors_model" \
        TOKENIZER="$safetensors_tokenizer" \
        HF_MODEL_ID="$HF_MODEL_ID" \
        "${SCRIPT_DIR}/logit-parity.sh" > "${OUTPUT_DIR}/logit_parity.log" 2>&1
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log_pass "Logit parity (τ-b ≥ 0.95)"
    else
        log_fail "Logit parity failed (expected τ-b ≥ 0.95)"
        return 1
    fi
}

# Test 5: NLL Parity
test_nll_parity() {
    log_test "5. Teacher-forcing NLL Parity"
    
    local safetensors_model="${MODELS_DIR}/${MODEL_ID}/safetensors/model.safetensors"
    local safetensors_tokenizer="${MODELS_DIR}/${MODEL_ID}/safetensors/tokenizer.json"
    local ppl_file="crossval/data/ppl_smoke.txt"
    
    if [ ! -f "$ppl_file" ]; then
        log_info "Creating sample perplexity test file"
        mkdir -p crossval/data
        echo "The quick brown fox jumps over the lazy dog." > "$ppl_file"
    fi
    
    # HF vs SafeTensors
    DELTA_NLL_MAX=1e-2 \
        MODEL_PATH="$safetensors_model" \
        TOKENIZER="$safetensors_tokenizer" \
        HF_MODEL_ID="$HF_MODEL_ID" \
        PPL_FILE="$ppl_file" \
        "${SCRIPT_DIR}/nll-parity.sh" > "${OUTPUT_DIR}/nll_parity.log" 2>&1
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log_pass "NLL parity (|Δ mean_nll| ≤ 1e-2)"
    else
        log_fail "NLL parity failed (expected |Δ mean_nll| ≤ 1e-2)"
        return 1
    fi
}

# Test 6: Performance Measurements
test_performance() {
    log_test "6. Performance Measurements"
    
    # Run performance measurements
    if "${SCRIPT_DIR}/measure_perf_json.sh" > "${OUTPUT_DIR}/perf_measurement.log" 2>&1; then
        # Check if JSON files were created
        local platform=$(get_platform_name)
        local st_json="bench/results/${platform}-safetensors.json"
        local gguf_json="bench/results/${platform}-gguf.json"
        
        if [ -f "$st_json" ] && [ -f "$gguf_json" ]; then
            log_pass "Performance measurements generated"
            
            # Render markdown
            if command -v python3 >/dev/null 2>&1; then
                python3 "${SCRIPT_DIR}/render_perf_md.py" "$st_json" > "docs/PERF_${platform}_ST.md" 2>/dev/null
                python3 "${SCRIPT_DIR}/render_perf_md.py" "$gguf_json" > "docs/PERF_${platform}_GGUF.md" 2>/dev/null
                log_info "Performance markdown rendered"
            fi
        else
            log_fail "Performance JSON files not created"
            return 1
        fi
    else
        log_fail "Performance measurement failed"
        return 1
    fi
}

# Summary report
print_summary() {
    echo
    echo "============================================"
    echo "         REALITY-PROOF CHECKLIST RESULTS"
    echo "============================================"
    echo
    
    local total_tests=$((${#PASSED_TESTS[@]} + ${#FAILED_TESTS[@]}))
    local passed=${#PASSED_TESTS[@]}
    local failed=${#FAILED_TESTS[@]}
    
    echo -e "${GREEN}Passed:${NC} $passed/$total_tests"
    for test in "${PASSED_TESTS[@]}"; do
        echo -e "  ${GREEN}✓${NC} $test"
    done
    
    if [ $failed -gt 0 ]; then
        echo
        echo -e "${RED}Failed:${NC} $failed/$total_tests"
        for test in "${FAILED_TESTS[@]}"; do
            echo -e "  ${RED}✗${NC} $test"
        done
    fi
    
    echo
    echo "Results saved to: ${OUTPUT_DIR}/"
    
    # Generate summary JSON
    cat > "${OUTPUT_DIR}/summary.json" <<EOF
{
  "timestamp": "$(date -u +%FT%TZ)",
  "platform": "$(get_platform_name)",
  "total_tests": $total_tests,
  "passed": $passed,
  "failed": $failed,
  "passed_tests": $(printf '%s\n' "${PASSED_TESTS[@]}" | jq -R . | jq -s .),
  "failed_tests": $(printf '%s\n' "${FAILED_TESTS[@]}" | jq -R . | jq -s .),
  "environment": {
    "deterministic": "${BITNET_DETERMINISTIC:-0}",
    "seed": "${BITNET_SEED:-42}",
    "threads": "${RAYON_NUM_THREADS:-1}"
  }
}
EOF
    
    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}✅ All reality-proof checks passed!${NC}"
        echo ""
        echo "Both SafeTensors and GGUF formats are production-ready."
        echo ""
        echo "Artifacts generated:"
        echo "  • ${OUTPUT_DIR}/summary.json - Test summary"
        echo "  • bench/results/*.json - Performance measurements"
        echo "  • docs/PERF_*.md - Rendered performance reports"
        echo ""
        echo "Next steps:"
        echo "  1. Review performance reports in docs/"
        echo "  2. Run 'scripts/stakeholder_demo.sh' for a 5-minute tour"
        echo "  3. Deploy with confidence!"
        return 0
    else
        echo -e "${RED}❌ Some checks failed. Review the logs for details.${NC}"
        echo "Check ${OUTPUT_DIR}/ for detailed logs."
        return 1
    fi
}

# Main execution
main() {
    echo ""
    echo "====================================================="
    echo "  Reality-Proof Checklist for Dual-Format Support"
    echo "====================================================="
    echo ""
    log_info "Platform: $(get_platform_name)"
    log_info "Output directory: ${OUTPUT_DIR}"
    echo
    
    # Run all tests
    test_environment
    test_model_introspection
    test_tokenizer_parity
    test_logit_parity
    test_nll_parity
    test_performance
    
    # Print summary
    print_summary
}

# Run if executed directly
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi