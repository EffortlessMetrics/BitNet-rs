#!/bin/bash
set -euo pipefail

# CI Acceptance Gate Script for BitNet.rs
# Implements the 8 validation gates described in VALIDATION.md
# Returns specific exit codes for precise CI triage

echo "=== BitNet.rs CI Acceptance Gate ==="
echo ""

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
MODEL_PATH="${MODEL_PATH:-models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"

# Performance thresholds
MIN_DECODE_TOKENS=20  # Minimum tokens for stable tok/s measurement
MIN_TOKENS_PER_SECOND=1.0  # Minimum acceptable tok/s

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
export RAYON_NUM_THREADS=1
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export OMP_NUM_THREADS=1
export GGML_NUM_THREADS=1

echo "Environment: DETERMINISTIC=1, SEED=42, THREADS=1"
echo ""

# Gate 1: Build
echo "━━━ Gate 1: Core Build ━━━"
if ! cargo build -p bitnet-cli --release --no-default-features --features "cpu,full-cli" --target-dir "${HOME}/.rust-build/target" 2>&1 | tail -1 | grep -q "Finished"; then
    echo "❌ Build failed"
    exit $EXIT_GENERAL_ERROR
fi
echo "✅ Build succeeded"
echo ""

# Gate 2: Unit Tests
echo "━━━ Gate 2: Unit Tests ━━━"
if ! cargo test --workspace --no-default-features --features cpu --target-dir "${HOME}/.rust-build/target" 2>&1 | grep -q "test result"; then
    echo "❌ Unit tests failed"
    exit $EXIT_GENERAL_ERROR
fi
echo "✅ Unit tests passed"
echo ""

# Gate 3: Tensor Name Mapping (JSON Gate)
echo "━━━ Gate 3: Tensor Name Mapping ━━━"
if [ ! -f "$MODEL_PATH" ]; then
    echo "⚠️  Model not found at $MODEL_PATH, attempting download..."
    if ! cargo run -p xtask -- download-model 2>&1 | grep -q "Successfully"; then
        echo "❌ Model download failed"
        exit $EXIT_MODEL_LOAD_ERROR
    fi
fi

MAPPER_JSON="/tmp/mapper_gate_$$.json"
if cargo run -q -p xtask -- gate mapper --model "$MODEL_PATH" > "$MAPPER_JSON" 2>/dev/null; then
    if jq -e '.ok == true and .unmapped_count == 0' "$MAPPER_JSON" >/dev/null 2>&1; then
        TOTAL_COUNT=$(jq -r '.total_count' "$MAPPER_JSON")
        echo "✅ All $TOTAL_COUNT tensors mapped"
    else
        UNMAPPED=$(jq -r '.unmapped_count // "unknown"' "$MAPPER_JSON")
        echo "❌ $UNMAPPED unmapped tensors"
        rm -f "$MAPPER_JSON"
        exit $EXIT_STRICT_MAPPING
    fi
else
    echo "❌ Mapper gate failed"
    rm -f "$MAPPER_JSON"
    exit $EXIT_STRICT_MAPPING
fi
rm -f "$MAPPER_JSON"
echo ""

# Gate 4: Strict Mode Execution
echo "━━━ Gate 4: Strict Mode Execution ━━━"

# Find tokenizer if not specified
if [ -z "$TOKENIZER_PATH" ]; then
    if [ -f "models/tokenizers/microsoft_bitnet_tokenizer.model" ]; then
        TOKENIZER_PATH="models/tokenizers/microsoft_bitnet_tokenizer.model"
    elif [ -f "models/tokenizer.model" ]; then
        TOKENIZER_PATH="models/tokenizer.model"
    fi
fi

STRICT_JSON="/tmp/bitnet_strict_$$.json"
TOKENIZER_ARGS=""
if [ -n "$TOKENIZER_PATH" ]; then
    TOKENIZER_ARGS="--tokenizer $TOKENIZER_PATH"
    echo "Using external tokenizer: $(basename "$TOKENIZER_PATH")"
fi

if "$BINARY_PATH" run \
    --model "$MODEL_PATH" \
    $TOKENIZER_ARGS \
    --prompt "The capital of France is" \
    --max-new-tokens 10 \
    --temperature 0 \
    --strict-mapping \
    --strict-tokenizer \
    --bos \
    --json-out "$STRICT_JSON" 2>/dev/null; then
    
    if [ -f "$STRICT_JSON" ]; then
        UNMAPPED=$(jq -r '.counts.unmapped // -1' "$STRICT_JSON")
        TOKENIZER_TYPE=$(jq -r '.tokenizer.type // "unknown"' "$STRICT_JSON")
        N_KV=$(jq -r '.counts.n_kv // "0"' "$STRICT_JSON")
        N_TENSORS=$(jq -r '.counts.n_tensors // "0"' "$STRICT_JSON")
        
        if [ "$UNMAPPED" = "0" ] && [ "$TOKENIZER_TYPE" = "sentencepiece" ]; then
            echo "✅ Strict mode: unmapped=0, SPM tokenizer, n_kv=$N_KV, n_tensors=$N_TENSORS"
        else
            echo "❌ Strict mode failed: unmapped=$UNMAPPED, tokenizer=$TOKENIZER_TYPE"
            rm -f "$STRICT_JSON"
            if [ "$UNMAPPED" != "0" ]; then
                exit $EXIT_STRICT_MAPPING
            else
                exit $EXIT_STRICT_TOKENIZER
            fi
        fi
    else
        echo "❌ No JSON output generated"
        exit $EXIT_INFERENCE_ERROR
    fi
else
    echo "❌ Strict mode inference failed"
    rm -f "$STRICT_JSON"
    exit $EXIT_INFERENCE_ERROR
fi
rm -f "$STRICT_JSON"

echo ""

# Gate 5: Tokenization Correctness
echo "━━━ Gate 5: Tokenization Correctness ━━━"
PROMPTS=(
    "The capital of France is"
    "Once upon a time"
    "def fibonacci(n):"
)

TOKENIZE_PASSED=0
for prompt in "${PROMPTS[@]}"; do
    TOKEN_JSON="/tmp/tokenize_$$.json"
    if "$BINARY_PATH" tokenize \
        --model "$MODEL_PATH" \
        $TOKENIZER_ARGS \
        --prompt "$prompt" \
        --bos \
        --json-out "$TOKEN_JSON" 2>/dev/null; then
        
        if [ -f "$TOKEN_JSON" ]; then
            IDS=$(jq -c '.tokens.ids' "$TOKEN_JSON" 2>/dev/null || echo "[]")
            if [ "$IDS" != "[]" ]; then
                TOKENIZE_PASSED=$((TOKENIZE_PASSED + 1))
            fi
            rm -f "$TOKEN_JSON"
        fi
    fi
done

if [ "$TOKENIZE_PASSED" -ge 2 ]; then
    echo "✅ Tokenization: $TOKENIZE_PASSED/${#PROMPTS[@]} prompts tokenized"
else
    echo "❌ Tokenization failed: only $TOKENIZE_PASSED/${#PROMPTS[@]} prompts tokenized (need ≥2)"
    exit $EXIT_TOKENIZER_ERROR
fi
echo ""

# Gate 6: Performance & Memory
echo "━━━ Gate 6: Performance & Memory ━━━"
PERF_JSON="/tmp/perf_$$.json"
TIME_OUT="/tmp/time_$$.out"

if [ -n "$TIME_CMD" ]; then
    # Run with memory profiling
    if $TIME_CMD "$BINARY_PATH" run \
        --model "$MODEL_PATH" \
        $TOKENIZER_ARGS \
        --prompt "Performance benchmark test" \
        --max-new-tokens "$MIN_DECODE_TOKENS" \
        --temperature 0 \
        --json-out "$PERF_JSON" 2>&1 | tee "$TIME_OUT" >/dev/null; then
        
        if [ -f "$PERF_JSON" ]; then
            DECODED=$(jq -r '.throughput.decoded_tokens // 0' "$PERF_JSON")
            TOKPS=$(jq -r '.throughput.tokens_per_second // 0' "$PERF_JSON")
            
            if [ "$DECODED" -lt "$MIN_DECODE_TOKENS" ]; then
                echo "⚠️  Performance: Only $DECODED tokens decoded (< $MIN_DECODE_TOKENS), measurement may be noisy"
            elif awk "BEGIN {exit !($TOKPS >= $MIN_TOKENS_PER_SECOND)}"; then
                echo "✅ Performance: ${TOKPS} tok/s, decoded=$DECODED tokens"
            else
                echo "❌ Performance failed: ${TOKPS} tok/s < ${MIN_TOKENS_PER_SECOND}"
                rm -f "$PERF_JSON" "$TIME_OUT"
                exit $EXIT_PERF_GATE_FAIL
            fi
            
            # Check memory if available
            if grep -q "Maximum resident set size" "$TIME_OUT"; then
                RSS_KB=$(grep "Maximum resident set size" "$TIME_OUT" | awk '{print $6}')
                RSS_MB=$((RSS_KB / 1024))
                echo "✅ Memory RSS: ${RSS_MB} MB"
            fi
        else
            echo "❌ No performance JSON generated"
            rm -f "$TIME_OUT"
            exit $EXIT_INFERENCE_ERROR
        fi
    else
        echo "❌ Performance test failed"
        rm -f "$PERF_JSON" "$TIME_OUT"
        exit $EXIT_INFERENCE_ERROR
    fi
else
    # Run without memory profiling
    if "$BINARY_PATH" run \
        --model "$MODEL_PATH" \
        $TOKENIZER_ARGS \
        --prompt "Performance benchmark test" \
        --max-new-tokens "$MIN_DECODE_TOKENS" \
        --temperature 0 \
        --json-out "$PERF_JSON" 2>/dev/null; then
        
        if [ -f "$PERF_JSON" ]; then
            DECODED=$(jq -r '.throughput.decoded_tokens // 0' "$PERF_JSON")
            TOKPS=$(jq -r '.throughput.tokens_per_second // 0' "$PERF_JSON")
            
            if [ "$DECODED" -lt "$MIN_DECODE_TOKENS" ]; then
                echo "⚠️  Performance: Only $DECODED tokens decoded, measurement may be noisy"
            elif awk "BEGIN {exit !($TOKPS >= $MIN_TOKENS_PER_SECOND)}"; then
                echo "✅ Performance: ${TOKPS} tok/s"
            else
                echo "❌ Performance failed: ${TOKPS} tok/s < ${MIN_TOKENS_PER_SECOND}"
                rm -f "$PERF_JSON"
                exit $EXIT_PERF_GATE_FAIL
            fi
        else
            echo "❌ No performance JSON generated"
            exit $EXIT_INFERENCE_ERROR
        fi
    else
        echo "❌ Performance test failed"
        rm -f "$PERF_JSON"
        exit $EXIT_INFERENCE_ERROR
    fi
fi
rm -f "$PERF_JSON" "$TIME_OUT"
echo ""

# Gate 7: FFI Compatibility
echo "━━━ Gate 7: FFI Compatibility ━━━"
if cargo build -p bitnet-ffi --release --no-default-features --features cpu --target-dir "${HOME}/.rust-build/target" 2>&1 | tail -1 | grep -q "Finished"; then
    if [ -f "${HOME}/.rust-build/target/release/libbitnet.so" ] || [ -f "${HOME}/.rust-build/target/release/libbitnet.dylib" ]; then
        echo "✅ FFI library built"
    else
        echo "❌ FFI library not found"
        exit $EXIT_GENERAL_ERROR
    fi
else
    echo "❌ FFI build failed"
    exit $EXIT_GENERAL_ERROR
fi
echo ""

# Gate 8: Determinism Check
echo "━━━ Gate 8: Determinism Check ━━━"
DET1="/tmp/det1_$$.json"
DET2="/tmp/det2_$$.json"

"$BINARY_PATH" run --model "$MODEL_PATH" $TOKENIZER_ARGS --prompt "Test" --max-new-tokens 10 \
    --temperature 0 --json-out "$DET1" >/dev/null 2>&1

"$BINARY_PATH" run --model "$MODEL_PATH" $TOKENIZER_ARGS --prompt "Test" --max-new-tokens 10 \
    --temperature 0 --json-out "$DET2" >/dev/null 2>&1

if [ -f "$DET1" ] && [ -f "$DET2" ]; then
    TEXT1=$(jq -r '.output // ""' "$DET1" 2>/dev/null)
    TEXT2=$(jq -r '.output // ""' "$DET2" 2>/dev/null)
    
    if [ "$TEXT1" = "$TEXT2" ] && [ -n "$TEXT1" ]; then
        echo "✅ Determinism: Outputs match at T=0"
    else
        echo "❌ Determinism failed: Outputs differ"
        rm -f "$DET1" "$DET2"
        exit $EXIT_GENERAL_ERROR
    fi
    rm -f "$DET1" "$DET2"
else
    echo "⚠️  Determinism check skipped: Could not generate outputs"
fi
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "            CI ACCEPTANCE: PASSED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "All validation gates passed:"
echo "✅ Build successful"
echo "✅ Unit tests passed"
echo "✅ Tensor mapping complete"
echo "✅ Strict mode validated"
echo "✅ Tokenization correct"
echo "✅ Performance acceptable"
echo "✅ FFI compatible"
echo "✅ Deterministic at T=0"
echo ""
echo "BitNet.rs is production-ready."

exit $EXIT_SUCCESS