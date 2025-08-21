#!/bin/bash
# Comprehensive validation suite for BitNet.rs
set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPORT_FILE="${PROJECT_ROOT}/validation_report.json"

# Initialize report
echo "{" > "$REPORT_FILE"
echo '  "timestamp": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'",' >> "$REPORT_FILE"
echo '  "tests": {' >> "$REPORT_FILE"

echo "=== BitNet.rs Comprehensive Validation ==="
echo ""

# Test 1: Build with CPU features
echo "1. Building release with CPU features..."
if cargo build --release --no-default-features --features cpu >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Build successful"
    echo '    "build": "pass",' >> "$REPORT_FILE"
else
    echo -e "${RED}✗${NC} Build failed"
    echo '    "build": "fail",' >> "$REPORT_FILE"
    exit 1
fi

# Test 2: Mapper dry-run test
echo ""
echo "2. Testing tensor name mapping..."
if [ -f "models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf" ]; then
    if cargo test --package crossval ms_bitnet_names_map_clean -- --nocapture 2>&1 | grep -q "test result: ok"; then
        echo -e "${GREEN}✓${NC} All tensor names mapped successfully"
        echo '    "mapper": "pass",' >> "$REPORT_FILE"
    else
        echo -e "${RED}✗${NC} Tensor mapping failed"
        echo '    "mapper": "fail",' >> "$REPORT_FILE"
    fi
else
    echo -e "${YELLOW}⚠${NC} MS BitNet model not found, skipping mapper test"
    echo '    "mapper": "skip",' >> "$REPORT_FILE"
fi

# Test 3: SentencePiece roundtrip
echo ""
echo "3. Testing SentencePiece tokenizer..."
if [ ! -z "${SPM:-}" ]; then
    if cargo test --package bitnet-tokenizers sp_roundtrip -- --ignored --nocapture 2>&1 | grep -q "test result: ok"; then
        echo -e "${GREEN}✓${NC} SentencePiece roundtrip successful"
        echo '    "sentencepiece": "pass",' >> "$REPORT_FILE"
    else
        echo -e "${RED}✗${NC} SentencePiece test failed"
        echo '    "sentencepiece": "fail",' >> "$REPORT_FILE"
    fi
else
    echo -e "${YELLOW}⚠${NC} SPM env var not set, skipping tokenizer test"
    echo '    "sentencepiece": "skip",' >> "$REPORT_FILE"
fi

# Test 4: Strict mode execution
echo ""
echo "4. Testing strict mode execution..."
if [ ! -z "${BITNET_GGUF:-}" ] && [ -f "${BITNET_GGUF}" ]; then
    # Set deterministic mode
    export RAYON_NUM_THREADS=1
    export BITNET_DETERMINISTIC=1
    export BITNET_SEED=42
    
    JSON_OUT=$(mktemp)
    if ./target/release/bitnet run \
        --model "${BITNET_GGUF}" \
        --prompt "The capital of France is" \
        --max-new-tokens 16 \
        --temperature 0 \
        --strict-mapping \
        --strict-tokenizer \
        --json-out "$JSON_OUT" >/dev/null 2>&1; then
        
        # Check unmapped count
        UNMAPPED=$(jq -r '.counts.unmapped' "$JSON_OUT" 2>/dev/null || echo "-1")
        if [ "$UNMAPPED" = "0" ]; then
            echo -e "${GREEN}✓${NC} Strict mode: 0 unmapped tensors"
            echo '    "strict_mode": "pass",' >> "$REPORT_FILE"
        else
            echo -e "${RED}✗${NC} Strict mode: $UNMAPPED unmapped tensors"
            echo '    "strict_mode": "fail",' >> "$REPORT_FILE"
        fi
        
        # Extract performance metrics
        FIRST_TOKEN=$(jq -r '.performance.first_token_ms' "$JSON_OUT" 2>/dev/null || echo "null")
        TOK_PER_SEC=$(jq -r '.performance.tok_per_sec' "$JSON_OUT" 2>/dev/null || echo "null")
        echo "  Performance: first_token=${FIRST_TOKEN}ms, throughput=${TOK_PER_SEC} tok/s"
    else
        echo -e "${RED}✗${NC} Strict mode execution failed"
        echo '    "strict_mode": "fail",' >> "$REPORT_FILE"
    fi
    rm -f "$JSON_OUT"
else
    echo -e "${YELLOW}⚠${NC} BITNET_GGUF not set or file missing"
    echo '    "strict_mode": "skip",' >> "$REPORT_FILE"
fi

# Test 5: A/B token ID comparison
echo ""
echo "5. Testing A/B token ID comparison..."
if [ -f "${BITNET_GGUF:-}" ] && [ -f "${TOKENIZER_MODEL:-}" ]; then
    if bash "${SCRIPT_DIR}/ab-smoke.sh" "$BITNET_GGUF" "$TOKENIZER_MODEL" 2>&1 | grep -q "PASS"; then
        echo -e "${GREEN}✓${NC} Token IDs match between Rust and C++"
        echo '    "ab_tokens": "pass"' >> "$REPORT_FILE"
    else
        echo -e "${RED}✗${NC} Token ID mismatch detected"
        echo '    "ab_tokens": "fail"' >> "$REPORT_FILE"
    fi
else
    echo -e "${YELLOW}⚠${NC} Model or tokenizer not available for A/B test"
    echo '    "ab_tokens": "skip"' >> "$REPORT_FILE"
fi

# Close JSON
echo '  }' >> "$REPORT_FILE"
echo '}' >> "$REPORT_FILE"

# Summary
echo ""
echo "=== Validation Summary ==="
echo ""
if [ -f "$REPORT_FILE" ]; then
    echo "Report saved to: $REPORT_FILE"
    echo ""
    # Count results
    PASS=$(grep -c '"pass"' "$REPORT_FILE" || true)
    FAIL=$(grep -c '"fail"' "$REPORT_FILE" || true)
    SKIP=$(grep -c '"skip"' "$REPORT_FILE" || true)
    
    echo "Results: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}, ${YELLOW}$SKIP skipped${NC}"
    
    if [ "$FAIL" -gt 0 ]; then
        echo ""
        echo -e "${RED}Validation FAILED${NC}"
        exit 1
    elif [ "$SKIP" -gt 2 ]; then
        echo ""
        echo -e "${YELLOW}Validation INCOMPLETE${NC} (too many skipped tests)"
        echo "To run full validation:"
        echo "  1. Set BITNET_GGUF=/path/to/model.gguf"
        echo "  2. Set TOKENIZER_MODEL=/path/to/tokenizer.model (for external tokenizer)"
        echo "  3. Set SPM=/path/to/tokenizer.model (for SP tests)"
        exit 0
    else
        echo ""
        echo -e "${GREEN}Validation PASSED${NC}"
    fi
else
    echo -e "${RED}Failed to create report${NC}"
    exit 1
fi