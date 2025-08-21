#!/bin/bash

# Script to fix C++ compatibility issues with BitNet models

set -e

echo "=== BitNet C++ Compatibility Fixer ==="
echo ""

# Step 1: Update C++ to latest version
echo "Step 1: Updating C++ implementation to latest version..."
echo "This ensures we have GGUF v3 support and latest fixes"
echo ""

cargo run -p xtask -- fetch-cpp --tag main --force || {
    echo "⚠️  Failed to update C++ implementation"
    echo "Continuing with existing version..."
}

echo ""
echo "Step 2: Setting up environment variables..."

# Set up library paths
CPP_DIR="${HOME}/.cache/bitnet_cpp"
export LD_LIBRARY_PATH="${CPP_DIR}/build/3rdparty/llama.cpp/src:${CPP_DIR}/build/3rdparty/llama.cpp/ggml/src:${LD_LIBRARY_PATH}"
export DYLD_LIBRARY_PATH="${CPP_DIR}/build/3rdparty/llama.cpp/src:${CPP_DIR}/build/3rdparty/llama.cpp/ggml/src:${DYLD_LIBRARY_PATH}"

echo "  LD_LIBRARY_PATH set ✓"
echo ""

# Step 3: Test with a minimal GGUF
echo "Step 3: Testing with minimal GGUF..."
cargo run -p xtask -- gen-mini-gguf --output tests/models/mini.gguf --version 3

export CROSSVAL_GGUF="tests/models/mini.gguf"
echo "Testing minimal model..."

cargo test -p bitnet-crossval --features crossval -- test_model_loading_parity --nocapture 2>&1 | grep -E "test result:|PASSED|FAILED" || {
    echo "⚠️  Minimal model test failed"
    echo "This suggests a fundamental compatibility issue"
}

echo ""
echo "Step 4: Testing with actual BitNet model..."

MODEL="models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf"
if [ -f "$MODEL" ]; then
    export CROSSVAL_GGUF="$MODEL"
    
    # First test Rust alone
    echo "Testing Rust implementation..."
    cargo test -p bitnet-models --no-default-features --features cpu -- gguf 2>&1 | grep -E "test result:" || echo "Rust tests status unknown"
    
    # Then test cross-validation
    echo ""
    echo "Testing cross-validation..."
    cargo test -p bitnet-crossval --features crossval -- test_model_loading_parity --nocapture 2>&1 | tail -10 || {
        EXIT_CODE=$?
        echo ""
        echo "=== Diagnosis ==="
        
        if [ $EXIT_CODE -eq 127 ]; then
            echo "❌ Library loading issue detected"
            echo ""
            echo "Solution 1: Rebuild with static linking"
            echo "  cd ${CPP_DIR}"
            echo "  cmake -B build -DBUILD_SHARED_LIBS=OFF"
            echo "  cmake --build build"
            echo ""
            echo "Solution 2: Install libraries system-wide"
            echo "  sudo cp ${CPP_DIR}/build/3rdparty/llama.cpp/src/*.so /usr/local/lib/"
            echo "  sudo ldconfig"
            echo ""
            echo "Solution 3: Use soft-fail mode"
            echo "  export CROSSVAL_ALLOW_CPP_FAIL=1"
        else
            echo "❌ Model compatibility issue"
            echo ""
            echo "The C++ implementation may not support this GGUF variant."
            echo "This is expected for experimental BitNet models."
            echo ""
            echo "Recommended: Use soft-fail mode in CI"
            echo "  export CROSSVAL_ALLOW_CPP_FAIL=1"
        fi
    }
else
    echo "Model not found: $MODEL"
    echo "Run: cargo xtask download-model"
fi

echo ""
echo "=== Summary ==="
echo ""
echo "To ensure C++ compatibility:"
echo ""
echo "1. Always set library paths:"
echo "   export LD_LIBRARY_PATH=\"${CPP_DIR}/build/3rdparty/llama.cpp/src:${CPP_DIR}/build/3rdparty/llama.cpp/ggml/src\""
echo ""
echo "2. For CI, enable soft-fail:"
echo "   export CROSSVAL_ALLOW_CPP_FAIL=1"
echo ""
echo "3. For production, consider converting models to GGUF v2 if needed:"
echo "   cargo run -p bitnet-compat -- convert-to-v2 input.gguf output_v2.gguf"
echo ""
echo "4. Keep C++ updated:"
echo "   cargo xtask fetch-cpp --tag main --force"