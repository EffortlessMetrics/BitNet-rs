#!/bin/bash
# Quick test to verify the C++ integration works

set -e

export BITNET_CPP_DIR=$HOME/.cache/bitnet_cpp
export OMP_NUM_THREADS=1
export GGML_NUM_THREADS=1

echo "Testing BitNet C++ integration..."
echo "BITNET_CPP_DIR: $BITNET_CPP_DIR"

# Build the project
echo "Building with crossval feature..."
cargo build -p bitnet-sys --features crossval

echo "✅ Build successful!"

# If a model path is provided, run tests
if [ -n "$1" ]; then
    export CROSSVAL_GGUF="$1"
    echo "Running parity tests with model: $CROSSVAL_GGUF"
    cargo test -p bitnet-crossval --features crossval -- --nocapture test_model_loading_parity
else
    echo "No model path provided. Skipping tests."
    echo "Usage: $0 /path/to/model.gguf"
fi