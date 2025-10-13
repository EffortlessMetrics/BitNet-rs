#!/bin/bash

# Diagnostic script to understand C++ model loading issues

MODEL="${1:-models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf}"
CPP_DIR="${HOME}/.cache/bitnet_cpp"

echo "=== BitNet C++ Diagnostic ==="
echo "Model: $MODEL"
echo ""

# Check if model exists
if [ ! -f "$MODEL" ]; then
    echo "❌ Model file not found: $MODEL"
    exit 1
fi

echo "✓ Model file exists"
echo "  Size: $(du -h "$MODEL" | cut -f1)"
echo ""

# Check GGUF header
echo "=== GGUF Header Analysis ==="
hexdump -C "$MODEL" | head -n 5
echo ""

# Extract version
VERSION=$(hexdump -s 4 -n 4 -e '1/4 "%u"' "$MODEL")
echo "GGUF Version: $VERSION"

if [ "$VERSION" -eq 3 ]; then
    echo "  ⚠️  GGUF v3 detected - may have compatibility issues with older C++"

    # Check tensor count (v3 uses 64-bit)
    TENSOR_COUNT=$(hexdump -s 8 -n 8 -e '1/8 "%llu"' "$MODEL")
    echo "  Tensor count: $TENSOR_COUNT"

    # Check metadata count
    METADATA_COUNT=$(hexdump -s 16 -n 8 -e '1/8 "%llu"' "$MODEL")
    echo "  Metadata KV count: $METADATA_COUNT"
else
    echo "  GGUF v$VERSION"

    # v2 uses 32-bit counts
    TENSOR_COUNT=$(hexdump -s 8 -n 4 -e '1/4 "%u"' "$MODEL")
    echo "  Tensor count: $TENSOR_COUNT"

    METADATA_COUNT=$(hexdump -s 12 -n 4 -e '1/4 "%u"' "$MODEL")
    echo "  Metadata KV count: $METADATA_COUNT"
fi

echo ""
echo "=== C++ Implementation Test ==="

# Find the C++ binary
CPP_BIN=""
for bin in "$CPP_DIR/build/bin/llama-cli" "$CPP_DIR/build/bin/main" "$CPP_DIR/build/3rdparty/llama.cpp/bin/llama-cli"; do
    if [ -f "$bin" ]; then
        CPP_BIN="$bin"
        echo "Found C++ binary: $bin"
        break
    fi
done

if [ -z "$CPP_BIN" ]; then
    echo "❌ No C++ binary found. Run: cargo xtask fetch-cpp"
    exit 1
fi

# Test loading with verbose output
echo ""
echo "Testing model load..."
echo "Command: $CPP_BIN -m \"$MODEL\" -n 1 -p \"test\" --log-disable 2>&1"
echo ""

# Capture both stdout and stderr
OUTPUT=$("$CPP_BIN" -m "$MODEL" -n 1 -p "test" --log-disable 2>&1)
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Model loaded successfully!"
    echo "Output sample:"
    echo "$OUTPUT" | head -10
else
    echo "❌ Model load failed (exit code: $EXIT_CODE)"
    echo "Error output:"
    echo "$OUTPUT" | grep -E "error|fail|Error|FAIL|llama_" | head -20

    echo ""
    echo "=== Potential Issues ==="

    # Check for common issues
    if echo "$OUTPUT" | grep -q "unsupported tensor"; then
        echo "• Unsupported tensor format or quantization"
    fi

    if echo "$OUTPUT" | grep -q "invalid magic"; then
        echo "• Invalid GGUF magic or version"
    fi

    if echo "$OUTPUT" | grep -q "failed to open"; then
        echo "• File access issue"
    fi

    if [ "$VERSION" -eq 3 ]; then
        echo "• GGUF v3 may require newer C++ implementation"
        echo "  Current C++ may only support v2"
    fi
fi

echo ""
echo "=== Recommendations ==="

if [ "$VERSION" -eq 3 ] && [ $EXIT_CODE -ne 0 ]; then
    echo "1. Update C++ implementation to latest version:"
    echo "   cargo xtask fetch-cpp --tag main --force"
    echo ""
    echo "2. Or convert model to GGUF v2:"
    echo "   cargo run -p bitnet-compat -- export-gguf-v2 \"$MODEL\" \"${MODEL%.gguf}_v2.gguf\""
    echo ""
    echo "3. Or use soft-fail in CI:"
    echo "   export CROSSVAL_ALLOW_CPP_FAIL=1"
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo "Model is compatible with C++ implementation ✅"
fi
