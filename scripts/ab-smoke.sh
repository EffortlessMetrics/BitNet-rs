#!/usr/bin/env bash
# A/B comparison script for bitnet-rs vs bitnet.cpp
# Tests token ID equality for validation
set -euo pipefail

# Configuration
MODEL="${1:?Usage: $0 <model.gguf> [tokenizer.model]}"
TOK="${2:-}"  # Optional tokenizer, will try to extract from GGUF if not provided
CPP_BIN="${LLAMA_BIN:-$HOME/.cache/bitnet_cpp/build/bin/llama-cli}"

# Locate Rust exe deterministically
RS_BIN=$(cargo build -p bitnet-cli --release --no-default-features --features cpu --message-format=json \
  | jq -r 'select(.executable != null) | .executable' | tail -1)

# Test prompts - short, deterministic
PROMPTS=(
    "2+2="
    "The capital of France is"
    "Hello, world!"
)
N_TOKENS=24

# Check Rust binary
if [ ! -f "$RS_BIN" ]; then
    echo "Error: Could not build bitnet-cli"
    exit 1
fi

# Build tokenizer args
TOK_ARGS=""
if [ -n "$TOK" ]; then
    TOK_ARGS="--tokenizer $TOK"
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
echo "bitnet-rs vs bitnet.cpp Token ID A/B Test"
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

    # Run Rust implementation with strict mode
    echo -n "  Running bitnet-rs... "
    if $RS_BIN run \
        --model "$MODEL" $TOK_ARGS \
        --prompt "$prompt" \
        --max-new-tokens $N_TOKENS \
        --temperature 0.0 \
        --strict-mapping \
        --json-out /tmp/rs_output.json \
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

    # Tokenize C++ output to get IDs
    if [ -n "$TOK" ]; then
        echo -n "  Tokenizing C++ output... "
        if $RS_BIN tokenize $TOK_ARGS --input-file /tmp/cpp_output.txt --json-out /tmp/cpp_ids.json >/dev/null 2>&1; then
            echo "✓"
            CPP_IDS=$(jq -c '.ids' /tmp/cpp_ids.json 2>/dev/null || echo "[]")
        else
            echo "✗ (tokenization failed)"
            CPP_IDS="[]"
        fi
    else
        CPP_IDS="[]"
    fi

    # Compare outputs
    echo "  Results:"
    echo "    Rust text:  \"${RS_TEXT:0:80}...\""
    echo "    C++ text:   \"${CPP_TEXT:0:80}...\""

    # Compare token IDs if available
    if [ -n "$TOK" ] && [ "$CPP_IDS" != "[]" ]; then
        echo "    Rust IDs:   $RS_IDS"
        echo "    C++ IDs:    $CPP_IDS"

        if [ "$RS_IDS" = "$CPP_IDS" ]; then
            echo "  ✅ PASS: Token IDs match exactly"
            ((PASSED++))
        else
            echo "  ❌ FAIL: Token IDs differ"
            ((FAILED++))
        fi
    else
        # Fallback to text comparison
        if [ "$RS_TEXT" = "$CPP_TEXT" ]; then
            echo "  ✅ PASS: Text outputs match"
            ((PASSED++))
        else
            echo "  ❌ FAIL: Text outputs differ"
            ((FAILED++))
        fi
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
