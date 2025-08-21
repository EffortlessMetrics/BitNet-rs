#!/usr/bin/env bash
# A/B comparison script for BitNet.rs vs bitnet.cpp
# Tests token ID equality for validation
set -euo pipefail

# Configuration
MODEL="${1:?Usage: $0 <model.gguf>}"
CPP_BIN="${LLAMA_BIN:-$HOME/.cache/bitnet_cpp/build/bin/llama-cli}"
RS_BIN="/home/steven/.rust-build/target/release/bitnet"

# Test prompts - short, deterministic
PROMPTS=(
    "2+2="
    "The capital of France is"
    "Hello, world!"
)
N_TOKENS=24

# Build if needed
if [ ! -f "$RS_BIN" ]; then
    echo "Building BitNet.rs..."
    cargo build --release --no-default-features --features cpu
fi

# Check C++ binary
if [ ! -f "$CPP_BIN" ]; then
    echo "Error: C++ binary not found at $CPP_BIN"
    echo "Set LLAMA_BIN or run: cargo xtask fetch-cpp"
    exit 1
fi

# Enforce determinism
export RAYON_NUM_THREADS=1 BITNET_DETERMINISTIC=1 BITNET_SEED=42
export OMP_NUM_THREADS=1 GGML_NUM_THREADS=1

echo "========================================"
echo "BitNet.rs vs bitnet.cpp Token ID A/B Test"
echo "========================================"
echo "Model: $MODEL"
echo "Tokens to generate: $N_TOKENS"
echo "Deterministic mode: SEED=42"
echo ""

# Results tracking
PASSED=0
FAILED=0

for prompt in "${PROMPTS[@]}"; do
    echo "Testing prompt: \"$prompt\""
    echo "----------------------------------------"
    
    # Run Rust implementation
    echo -n "  Running BitNet.rs... "
    if $RS_BIN run \
        --model "$MODEL" \
        --prompt "$prompt" \
        --max-new-tokens $N_TOKENS \
        --temperature 0.0 \
        --json-out /tmp/rs_output.json \
        --seed 42 \
        >/dev/null 2>&1; then
        echo "✓"
    else
        echo "✗ (failed to run)"
        ((FAILED++))
        continue
    fi
    
    # Run C++ implementation  
    echo -n "  Running bitnet.cpp... "
    if $CPP_BIN \
        -m "$MODEL" \
        -ngl 0 \
        -p "$prompt" \
        -n $N_TOKENS \
        -temp 0.0 \
        -seed 42 \
        --no-display-prompt \
        2>/dev/null | tail -n +2 > /tmp/cpp_output.txt; then
        echo "✓"
    else
        echo "✗ (failed to run)"
        ((FAILED++))
        continue
    fi
    
    # Extract token IDs from Rust output
    if [ -f /tmp/rs_output.json ]; then
        RS_IDS=$(jq -c '.ids' /tmp/rs_output.json 2>/dev/null || echo "[]")
        RS_TEXT=$(jq -r '.text' /tmp/rs_output.json 2>/dev/null || echo "")
    else
        RS_IDS="[]"
        RS_TEXT=""
    fi
    
    # Get C++ text output
    CPP_TEXT=$(cat /tmp/cpp_output.txt | tr -d '\n')
    
    # Compare outputs
    echo "  Results:"
    echo "    Rust text:  \"$RS_TEXT\""
    echo "    C++ text:   \"$CPP_TEXT\""
    echo "    Rust IDs:   $RS_IDS"
    
    # Simple text comparison for now (since we can't easily get IDs from C++)
    if [ "$RS_TEXT" = "$CPP_TEXT" ]; then
        echo "  ✅ PASS: Outputs match"
        ((PASSED++))
    else
        echo "  ❌ FAIL: Outputs differ"
        ((FAILED++))
    fi
    echo ""
done

# Summary
echo "========================================"
echo "Test Summary"
echo "========================================"
echo "Passed: $PASSED"
echo "Failed: $FAILED"

if [ $FAILED -eq 0 ]; then
    echo "✅ All tests passed!"
    exit 0
else
    echo "❌ Some tests failed"
    exit 1
fi