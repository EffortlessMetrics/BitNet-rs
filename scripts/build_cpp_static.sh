#!/bin/bash
# Build C++ with static linking to avoid library path issues
set -e

echo "=== Building BitNet C++ with Static Linking ==="
echo ""

CPP_DIR="${HOME}/.cache/bitnet_cpp"

if [ ! -d "$CPP_DIR" ]; then
    echo "Error: C++ not found. Run: cargo xtask fetch-cpp"
    exit 1
fi

cd "$CPP_DIR"

echo "Building with static libraries..."
cmake -B build -S . \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_STATIC=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DGGML_NATIVE=ON \
    -DLLAMA_CURL=OFF

cmake --build build -j

echo ""
echo "✅ Static build complete!"
echo ""
echo "Binaries available at:"
ls -la build/bin/llama* 2>/dev/null || echo "No llama binaries found"

echo ""
echo "Test with:"
echo "  $CPP_DIR/build/bin/llama-cli -m <model.gguf> -p \"test\" -n 1 -ngl 0"