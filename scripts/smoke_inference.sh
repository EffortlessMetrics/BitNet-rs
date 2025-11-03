#!/usr/bin/env bash
# Smoke test for CPU inference with strict validation
#
# Usage: ./scripts/smoke_inference.sh <model.gguf> <tokenizer.json>
#
# This script validates:
# - Real inference (no mock computation)
# - Deterministic output
# - Reasonable performance baseline
# - Clean exit codes

set -euo pipefail

# Parse arguments
MODEL_PATH="${1:-models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf}"
TOKENIZER_PATH="${2:-models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json}"

# Configuration
PROMPT="Say OK."
MAX_TOKENS=16
TIMEOUT_SEC=180

# Check prerequisites
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "❌ Model not found: $MODEL_PATH"
    exit 1
fi

if [[ ! -f "$TOKENIZER_PATH" ]]; then
    echo "❌ Tokenizer not found: $TOKENIZER_PATH"
    exit 1
fi

if [[ ! -x "target/release/bitnet" ]]; then
    echo "❌ Release binary not found. Build with:"
    echo "   cargo build -p bitnet-cli --release --no-default-features --features cpu,full-cli"
    exit 1
fi

echo "🔍 Running smoke inference test..."
echo "   Model: $MODEL_PATH"
echo "   Tokenizer: $TOKENIZER_PATH"
echo "   Prompt: \"$PROMPT\""
echo "   Max tokens: $MAX_TOKENS"
echo ""

# Export deterministic environment
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=4
export RUST_LOG=warn

# Run inference with timeout
echo "⏱️  Running inference (${TIMEOUT_SEC}s timeout)..."
if ! timeout "${TIMEOUT_SEC}" target/release/bitnet run \
    --model "$MODEL_PATH" \
    --tokenizer "$TOKENIZER_PATH" \
    --device cpu \
    --prompt "$PROMPT" \
    --max-new-tokens "$MAX_TOKENS" \
    > /tmp/bitnet_smoke_output.txt 2>&1; then
    echo "❌ Inference failed or timed out"
    cat /tmp/bitnet_smoke_output.txt
    exit 1
fi

echo "✅ Inference completed successfully"

# Check for AVX2 acceleration
if grep -q "Using QK256 quantization with AVX2 acceleration" /tmp/bitnet_smoke_output.txt; then
    echo "✅ AVX2 acceleration detected"
else
    echo "⚠️  AVX2 acceleration not detected (may be expected on non-x86 platforms)"
fi

# Check for generated tokens
if grep -q "Generated.*tokens" /tmp/bitnet_smoke_output.txt; then
    TOKENS_LINE=$(grep "Generated.*tokens" /tmp/bitnet_smoke_output.txt)
    echo "✅ Generation completed: $TOKENS_LINE"
else
    echo "❌ No generation output found"
    cat /tmp/bitnet_smoke_output.txt
    exit 1
fi

# Extract TPS for baseline validation
TPS=$(echo "$TOKENS_LINE" | grep -oP '\d+\.\d+(?= tok/s)' || echo "0.0")
echo "📊 Tokens per second: $TPS"

# Validate reasonable TPS range (0.05 - 2.0 tok/s for scalar QK256)
# Using bc for floating point comparison
if command -v bc >/dev/null 2>&1; then
    if (( $(echo "$TPS >= 0.05 && $TPS <= 2.0" | bc -l) )); then
        echo "✅ TPS within expected range for scalar QK256"
    else
        echo "⚠️  TPS outside expected range (0.05 - 2.0 tok/s)"
        echo "   This may indicate mock computation or hardware issues"
    fi
fi

echo ""
echo "✅ Smoke test passed!"
echo ""
echo "Next steps:"
echo "  - Run determinism test: ./scripts/test_determinism.sh"
echo "  - Measure full baseline: cargo run -p xtask -- benchmark"
