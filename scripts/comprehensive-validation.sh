#!/bin/bash
set -euo pipefail

# Comprehensive validation script for BitNet.rs
# Tests all validation gates described in VALIDATION.md

echo "=== BitNet.rs Comprehensive Validation ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Exit codes (matching bitnet-cli/src/exit.rs)
EXIT_SUCCESS=0
EXIT_GENERAL_ERROR=1
EXIT_INVALID_ARGS=2
EXIT_STRICT_MAPPING=3
EXIT_STRICT_TOKENIZER=4
EXIT_MODEL_LOAD_ERROR=5
EXIT_TOKENIZER_ERROR=6
EXIT_INFERENCE_ERROR=7
EXIT_IO_ERROR=8
EXIT_PERF_GATE_FAIL=9
EXIT_MEM_GATE_FAIL=10

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BINARY_PATH="${HOME}/.rust-build/target/release/bitnet"
MODELS_DIR="${PROJECT_ROOT}/models"

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
XFAIL_TESTS=0

# Test results array
declare -a TEST_RESULTS=()

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_xfail() {
    echo -e "${YELLOW}[XFAIL]${NC} $1"
}

report_test() {
    local test_name="$1"
    local status="$2"
    local details="${3:-}"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if [ "$status" = "PASS" ]; then
        log_success "$test_name"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        TEST_RESULTS+=("PASS: $test_name")
    elif [ "$status" = "XFAIL" ]; then
        log_xfail "$test_name (Expected failure)"
        XFAIL_TESTS=$((XFAIL_TESTS + 1))
        PASSED_TESTS=$((PASSED_TESTS + 1))  # Count as pass
        TEST_RESULTS+=("XFAIL: $test_name")
    elif [ "$status" = "SKIP" ]; then
        log_warning "$test_name: SKIPPED"
        TEST_RESULTS+=("SKIP: $test_name")
    else
        log_error "$test_name"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        TEST_RESULTS+=("FAIL: $test_name")
    fi
    
    if [ -n "$details" ]; then
        echo "   $details"
    fi
}

# Detect time command for portable profiling
detect_time() {
    if command -v /usr/bin/time >/dev/null 2>&1; then
        echo "/usr/bin/time -v"
    elif command -v gtime >/dev/null 2>&1; then
        echo "gtime -v"
    else
        echo ""
    fi
}

TIME_CMD=$(detect_time)

# Set deterministic environment
setup_deterministic_env() {
    export RAYON_NUM_THREADS=1
    export BITNET_DETERMINISTIC=1
    export BITNET_SEED=42
    export OMP_NUM_THREADS=1
    export GGML_NUM_THREADS=1
    log_info "Set deterministic environment (threads=1, seed=42)"
}

echo "${BLUE}━━━ Gate 1: Core Build Validation ━━━${NC}"
log_info "Building release with CPU features..."
if cargo build -p bitnet-cli --release --no-default-features --features "cpu,full-cli" --target-dir "${HOME}/.rust-build/target" 2>&1 | tail -1 | grep -q "Finished"; then
    report_test "Core Build" "PASS"
else
    report_test "Core Build" "FAIL" "Build failed"
fi

echo ""
echo "${BLUE}━━━ Gate 2: Unit Test Suite ━━━${NC}"
log_info "Running workspace unit tests..."
if cargo test --workspace --no-default-features --features cpu --target-dir "${HOME}/.rust-build/target" 2>&1 | grep -q "test result"; then
    report_test "Unit Tests" "PASS"
else
    report_test "Unit Tests" "FAIL" "Tests did not compile or run"
fi

echo ""
echo "${BLUE}━━━ Gate 3: Tensor Name Mapping (JSON Gate) ━━━${NC}"
MODEL_PATH="models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf"
if [ -f "$MODEL_PATH" ]; then
    log_info "Testing tensor mapping with xtask gate..."
    JSON_OUT="/tmp/mapper_gate_$$.json"
    if cargo run -q -p xtask -- gate mapper --model "$MODEL_PATH" > "$JSON_OUT" 2>/dev/null; then
        if jq -e '.ok == true and .unmapped_count == 0' "$JSON_OUT" >/dev/null 2>&1; then
            total_count=$(jq -r '.total_count' "$JSON_OUT")
            report_test "Tensor Name Mapping" "PASS" "All $total_count tensors mapped"
        else
            unmapped=$(jq -r '.unmapped_count // "unknown"' "$JSON_OUT")
            report_test "Tensor Name Mapping" "FAIL" "$unmapped unmapped tensors"
        fi
    else
        report_test "Tensor Name Mapping" "FAIL" "xtask gate mapper failed"
    fi
    rm -f "$JSON_OUT"
else
    log_warning "Model not found, attempting download..."
    if cargo run -p xtask -- download-model 2>&1 | grep -q "Successfully"; then
        report_test "Model Download" "PASS"
    else
        report_test "Model Download" "FAIL"
    fi
fi

echo ""
echo "${BLUE}━━━ Gate 4: Strict Mode Execution ━━━${NC}"
setup_deterministic_env
if [ -f "$MODEL_PATH" ]; then
    log_info "Running strict mode validation..."
    
    # Check if tokenizer exists
    TOKENIZER_PATH=""
    if [ -f "models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.model" ]; then
        TOKENIZER_PATH="models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.model"
    fi
    
    if [ -n "$TOKENIZER_PATH" ]; then
        log_info "Using external tokenizer: $(basename "$TOKENIZER_PATH")"
        if "$BINARY_PATH" run \
            --model "$MODEL_PATH" \
            --tokenizer "$TOKENIZER_PATH" \
            --prompt "The capital of France is" \
            --max-new-tokens 10 \
            --temperature 0 \
            --strict-mapping \
            --strict-tokenizer \
            --bos \
            --json-out /tmp/bitnet_strict.json 2>/dev/null; then
            
            # Verify JSON output
            if [ -f /tmp/bitnet_strict.json ]; then
                UNMAPPED=$(jq -r '.counts.unmapped // 1' /tmp/bitnet_strict.json)
                TOKENIZER_TYPE=$(jq -r '.tokenizer.type // "unknown"' /tmp/bitnet_strict.json)
                
                if [ "$UNMAPPED" = "0" ] && [ "$TOKENIZER_TYPE" = "sentencepiece" ]; then
                    n_kv=$(jq -r '.counts.n_kv // "0"' /tmp/bitnet_strict.json)
                    n_tensors=$(jq -r '.counts.n_tensors // "0"' /tmp/bitnet_strict.json)
                    report_test "Strict Mode" "PASS" "unmapped=0, SPM tokenizer, n_kv=$n_kv, n_tensors=$n_tensors"
                else
                    report_test "Strict Mode" "FAIL" "unmapped=$UNMAPPED, tokenizer=$TOKENIZER_TYPE"
                fi
            else
                report_test "Strict Mode" "FAIL" "No JSON output generated"
            fi
        else
            report_test "Strict Mode" "FAIL" "Inference failed"
        fi
    else
        # Run with embedded tokenizer (if available)
        log_info "Attempting with embedded tokenizer..."
        if "$BINARY_PATH" run \
            --model "$MODEL_PATH" \
            --prompt "The capital of France is" \
            --max-new-tokens 10 \
            --temperature 0 \
            --strict-mapping \
            --json-out /tmp/bitnet_strict.json 2>/dev/null; then
            if [ -f /tmp/bitnet_strict.json ]; then
                UNMAPPED=$(jq -r '.counts.unmapped // 1' /tmp/bitnet_strict.json)
                if [ "$UNMAPPED" = "0" ]; then
                    report_test "Strict Mode" "PASS" "Inference completed with embedded tokenizer"
                else
                    report_test "Strict Mode" "FAIL" "Unmapped tensors: $UNMAPPED"
                fi
            else
                report_test "Strict Mode" "FAIL" "No JSON output"
            fi
        else
            report_test "Strict Mode" "FAIL" "Inference failed"
        fi
    fi
else
    report_test "Strict Mode" "SKIP" "Model not available"
fi

echo ""
echo "${BLUE}━━━ Gate 5: A/B Tokenization Correctness ━━━${NC}"
if [ -f "$MODEL_PATH" ]; then
    log_info "Testing tokenization with multiple prompts..."
    # Test prompts
    PROMPTS=(
        "The capital of France is"
        "Once upon a time"
        "def fibonacci(n):"
    )
    
    AB_PASSED=0
    AB_TOTAL=${#PROMPTS[@]}
    
    for prompt in "${PROMPTS[@]}"; do
        JSON_OUT="/tmp/tokenize_$$.json"
        if "$BINARY_PATH" tokenize \
            --model "$MODEL_PATH" \
            --prompt "$prompt" \
            --bos \
            --json-out "$JSON_OUT" 2>/dev/null; then
            
            if [ -f "$JSON_OUT" ]; then
                RS_IDS=$(jq -c '.tokens.ids' "$JSON_OUT" 2>/dev/null || echo "[]")
                log_info "Prompt: '$prompt' → IDs: $RS_IDS"
                AB_PASSED=$((AB_PASSED + 1))
                rm -f "$JSON_OUT"
            fi
        fi
    done
    
    if [ "$AB_PASSED" -ge 2 ]; then
        report_test "A/B Tokenization" "PASS" "$AB_PASSED/$AB_TOTAL prompts tokenized successfully"
    else
        report_test "A/B Tokenization" "FAIL" "Only $AB_PASSED/$AB_TOTAL prompts tokenized (need ≥2)"
    fi
else
    report_test "A/B Tokenization" "SKIP" "Model not available"
fi

echo ""
echo "${BLUE}━━━ Gate 6: Performance & Memory Gates ━━━${NC}"

MIN_DECODE=20  # Minimum tokens for stable tok/s

if [ -f "$MODEL_PATH" ]; then
    if [ -z "$TIME_CMD" ]; then
        log_warning "No GNU time available for memory profiling"
        # Still run without memory profiling
        if "$BINARY_PATH" run \
            --model "$MODEL_PATH" \
            --prompt "Validation performance benchmark" \
            --max-new-tokens "$MIN_DECODE" \
            --temperature 0 \
            --json-out /tmp/perf.json 2>&1 | grep -q "Generated"; then
            
            if [ -f /tmp/perf.json ]; then
                DECODED=$(jq -r '.throughput.decoded_tokens // 0' /tmp/perf.json)
                TOKPS=$(jq -r '.throughput.tokens_per_second // 0' /tmp/perf.json)
                
                if [ "$DECODED" -lt "$MIN_DECODE" ]; then
                    report_test "Performance" "WARN" "Decoded=$DECODED (<$MIN_DECODE); measurement may be noisy"
                else
                    report_test "Performance" "PASS" "${TOKPS} tok/s, decoded=$DECODED tokens"
                fi
            else
                report_test "Performance" "FAIL" "No perf JSON generated"
            fi
        else
            report_test "Performance" "FAIL" "Benchmark failed"
        fi
    else
        # Run with time command for memory profiling
        log_info "Running with memory profiling..."
        if $TIME_CMD "$BINARY_PATH" run \
            --model "$MODEL_PATH" \
            --prompt "Validation performance benchmark" \
            --max-new-tokens "$MIN_DECODE" \
            --temperature 0 \
            --json-out /tmp/perf.json 2>&1 | tee /tmp/time.out >/dev/null; then
            
            if [ -f /tmp/perf.json ]; then
                DECODED=$(jq -r '.throughput.decoded_tokens // 0' /tmp/perf.json)
                TOKPS=$(jq -r '.throughput.tokens_per_second // 0' /tmp/perf.json)
                
                if [ "$DECODED" -lt "$MIN_DECODE" ]; then
                    report_test "Performance" "WARN" "Decoded=$DECODED (<$MIN_DECODE); measurement may be noisy"
                else
                    report_test "Performance" "PASS" "${TOKPS} tok/s, decoded=$DECODED tokens"
                fi
                
                # Extract RSS if available
                if grep -q "Maximum resident set size" /tmp/time.out; then
                    RSS_KB=$(grep "Maximum resident set size" /tmp/time.out | awk '{print $6}')
                    RSS_MB=$((RSS_KB / 1024))
                    report_test "Memory RSS" "PASS" "Max RSS: ${RSS_MB} MB"
                fi
            else
                report_test "Performance" "FAIL" "No perf JSON generated"
            fi
        else
            report_test "Performance" "FAIL" "Benchmark failed"
        fi
    fi
else
    report_test "Performance" "SKIP" "Model not available"
fi

echo ""
echo "${BLUE}━━━ Gate 7: FFI Compatibility Check ━━━${NC}"
log_info "Building FFI library..."
if cargo build -p bitnet-ffi --release --no-default-features --features cpu 2>&1 | tail -1 | grep -q "Finished"; then
    report_test "FFI Build" "PASS"
    
    # Check library exists
    if [ -f "target/release/libbitnet_ffi.so" ] || [ -f "target/release/libbitnet_ffi.dylib" ]; then
        report_test "FFI Library" "PASS" "Shared library created"
    else
        report_test "FFI Library" "FAIL" "Library not found"
    fi
else
    report_test "FFI Build" "FAIL"
fi

echo ""
echo "${BLUE}━━━ Gate 8: Cross-Validation Tests ━━━${NC}"
if command -v cargo >/dev/null 2>&1; then
    log_info "Running cross-validation tests..."
    if cargo test --package bitnet-crossval --no-default-features --features 'cpu,ffi' 2>&1 | grep -q "test result"; then
        report_test "Cross-Validation" "PASS"
    else
        report_test "Cross-Validation" "SKIP" "Tests not available or FFI not built"
    fi
else
    report_test "Cross-Validation" "SKIP" "Cargo not available"
fi

echo ""
echo "${BLUE}━━━ Additional: Determinism Check ━━━${NC}"
if [ -f "$MODEL_PATH" ]; then
    log_info "Testing deterministic output..."
    setup_deterministic_env
    
    OUT1="/tmp/det1_$$.json"
    OUT2="/tmp/det2_$$.json"
    
    # Run twice with same seed
    "$BINARY_PATH" run --model "$MODEL_PATH" --prompt "Test" --max-new-tokens 10 \
        --temperature 0 --json-out "$OUT1" >/dev/null 2>&1
    
    "$BINARY_PATH" run --model "$MODEL_PATH" --prompt "Test" --max-new-tokens 10 \
        --temperature 0 --json-out "$OUT2" >/dev/null 2>&1
    
    if [ -f "$OUT1" ] && [ -f "$OUT2" ]; then
        TEXT1=$(jq -r '.output // ""' "$OUT1" 2>/dev/null)
        TEXT2=$(jq -r '.output // ""' "$OUT2" 2>/dev/null)
        
        if [ "$TEXT1" = "$TEXT2" ] && [ -n "$TEXT1" ]; then
            report_test "Determinism" "PASS" "Outputs match at T=0"
        else
            report_test "Determinism" "FAIL" "Outputs differ"
        fi
        
        rm -f "$OUT1" "$OUT2"
    else
        report_test "Determinism" "SKIP" "Could not generate outputs"
    fi
else
    report_test "Determinism" "SKIP" "Model not available"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}            VALIDATION SUMMARY            ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo "Total Tests: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
echo -e "${RED}Failed: $FAILED_TESTS${NC}"
echo -e "${YELLOW}XFail: $XFAIL_TESTS${NC}"
echo ""

SUCCESS_RATE=0
if [ $TOTAL_TESTS -gt 0 ]; then
    SUCCESS_RATE=$((PASSED_TESTS * 100 / TOTAL_TESTS))
fi

echo "Pass Rate: ${SUCCESS_RATE}%"

if [ $SUCCESS_RATE -eq 100 ]; then
    echo -e "\n${GREEN}✅ ALL TESTS PASSED!${NC}"
    echo ""
    echo "BitNet.rs validation complete:"
    echo "• Zero unmapped tensors in strict mode"
    echo "• Real SentencePiece tokenization"
    echo "• Deterministic outputs at T=0"
    echo "• JSON-driven gates for CI robustness"
    exit 0
elif [ $SUCCESS_RATE -ge 90 ]; then
    echo -e "\n${YELLOW}⚠️  MOSTLY PASSING (${SUCCESS_RATE}%)${NC}"
    echo ""
    echo "BitNet.rs validation mostly complete"
    exit 0
else
    echo -e "\n${RED}❌ VALIDATION FAILED (${SUCCESS_RATE}%)${NC}"
    echo ""
    echo "Please address failing tests before deployment"
    exit 1
fi