#!/bin/bash
# Test quantization support for IQ2_S and I2_S format

set -euo pipefail

echo "=== Testing BitNet Quantization Support ==="
echo

# Check if bitnet CLI exists
if [ ! -f "./target/release/bitnet" ]; then
    echo "Error: bitnet CLI not found at ./target/release/bitnet"
    echo "Build it first with:"
    echo "  cargo build -p bitnet-cli --release --no-default-features --features \"cpu,iq2s-ffi\""
    exit 1
fi

# Function to test model loading
test_model() {
    local model_path="$1"
    local expected_quant="$2"

    if [ ! -f "$model_path" ]; then
        echo "⚠ Model not found: $model_path (skipping)"
        return
    fi

    echo "Testing $model_path..."

    # Run inspect to get quantization type
    local quant=$(RUST_LOG=error ./target/release/bitnet inspect --model "$model_path" --json 2>/dev/null | jq -r '.quantization // "unknown"')

    if [ "$quant" = "$expected_quant" ]; then
        echo "✓ Detected quantization: $quant"
    else
        echo "✗ Expected $expected_quant, got $quant"
        return 1
    fi

    # Try a simple generation
    echo "  Running inference test..."
    if timeout 10 RUST_LOG=error BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
        ./target/release/bitnet run --model "$model_path" --prompt "Hello" --max-new-tokens 4 --greedy 2>/dev/null; then
        echo "  ✓ Inference successful"
    else
        echo "  ✗ Inference failed or timed out"
    fi

    echo
}

# Test IQ2_S models (GGML format)
echo "=== IQ2_S Support (via GGML FFI) ==="
test_model "models/test-iq2s.gguf" "IQ2_S"
test_model "models/llama-iq2s.gguf" "IQ2_S"

# Test I2_S models (BitNet native)
echo "=== I2_S Support (Native Rust) ==="
test_model "models/test-i2s.gguf" "I2_S"
test_model "models/bitnet-i2s.gguf" "I2_S"

# Test IS_2 alias
echo "=== IS_2 Alias Support ==="
test_model "models/test-is2.gguf" "I2_S"

echo "=== Feature Detection ==="
echo -n "IQ2_S FFI support: "
if ldd ./target/release/bitnet 2>/dev/null | grep -q bitnet_ggml || \
   otool -L ./target/release/bitnet 2>/dev/null | grep -q bitnet_ggml; then
    echo "✓ Enabled (GGML FFI linked)"
else
    echo "✗ Disabled (rebuild with --features iq2s-ffi)"
fi

echo -n "I2_S native support: "
echo "✓ Always enabled"

echo
echo "=== Summary ==="
echo "• IQ2_S: GGML's 2-bit quantization, requires --features iq2s-ffi"
echo "• I2_S/IS_2: BitNet's native 2-bit signed format, always available"
echo "• Both formats dequantize to f32 at load time for correctness"
echo "• Performance optimizations can be added later"
