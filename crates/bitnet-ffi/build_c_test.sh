#!/bin/bash

# Build script for C integration test
# This script compiles the C integration test and links it with the Rust library

set -e

echo "Building BitNet C API integration test..."

# Build the Rust library first
echo "Building Rust library..."
cargo build --release -p bitnet-ffi

# Get the target directory
TARGET_DIR="../../target/release"
LIB_NAME="bitnet_ffi"

# Determine the library extension based on the platform
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    LIB_EXT="so"
    LIB_PREFIX="lib"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    LIB_EXT="dylib"
    LIB_PREFIX="lib"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    LIB_EXT="dll"
    LIB_PREFIX=""
else
    echo "Unsupported platform: $OSTYPE"
    exit 1
fi

# Compile the C test
echo "Compiling C integration test..."
gcc -std=c99 -Wall -Wextra -I./include \
    -L${TARGET_DIR} \
    -l${LIB_NAME} \
    tests/c_integration_test.c \
    -o c_integration_test

echo "C integration test compiled successfully!"

# Set library path for running the test
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export LD_LIBRARY_PATH="${TARGET_DIR}:$LD_LIBRARY_PATH"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH="${TARGET_DIR}:$DYLD_LIBRARY_PATH"
fi

echo "Running C integration test..."
./c_integration_test

echo "C integration test completed successfully!"

# Clean up
rm -f c_integration_test

echo "Build and test completed!"
