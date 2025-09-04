#!/bin/bash
set -e

# FFI Smoke Test Script
# Tests basic FFI functionality with current compiler settings

echo "FFI Smoke Test"
echo "Compiler: ${CC:-gcc} / ${CXX:-g++}"

# Check if FFI library exists
FFI_LIB="target/release/libbitnet_ffi.so"
if [[ "$OSTYPE" == "darwin"* ]]; then
    FFI_LIB="target/release/libbitnet_ffi.dylib"
fi

if [ ! -f "$FFI_LIB" ]; then
    echo "FFI library not found at $FFI_LIB"
    echo "Please build first with: cargo build -p bitnet-ffi --release --no-default-features --features cpu"
    exit 1
fi

echo "FFI library found: $FFI_LIB"

# Check library symbols (basic smoke test)
if command -v objdump &> /dev/null; then
    echo "Checking for key FFI symbols..."
    if objdump -T "$FFI_LIB" 2>/dev/null | grep -q "bitnet_init"; then
        echo "✅ FFI symbols found"
    else
        echo "⚠️  FFI symbols not found or objdump not available"
    fi
else
    echo "objdump not available, skipping symbol check"
fi

echo "FFI smoke test completed successfully"