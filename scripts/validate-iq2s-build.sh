#!/bin/bash
# Quick validation script for IQ2_S build

set -e

echo "=== IQ2_S Build Validation ==="
echo

# Build with IQ2_S support
echo "1. Building with IQ2_S support..."
cargo build --package bitnet-cli --bin bitnet \
    --no-default-features --features "cpu,iq2s-ffi" \
    --release 2>&1 | tail -3

BITNET_BIN="/home/steven/.rust-build/target/release/bitnet"

# Check version shows features
echo
echo "2. Checking version output..."
BITNET_QUIET_BACKEND=1 $BITNET_BIN --version

# Check that features include iq2s-ffi
echo
echo "3. Verifying features..."
if $BITNET_BIN --version | grep -q "iq2s-ffi"; then
    echo "✓ IQ2_S feature enabled"
else
    echo "✗ IQ2_S feature not found"
    exit 1
fi

# Check GGML commit is shown (when not 'unknown')
echo
echo "4. Checking GGML commit..."
if $BITNET_BIN --version | grep -q "ggml:"; then
    echo "✓ GGML commit line present"
else
    echo "✗ GGML commit not shown"
    exit 1
fi

echo
echo "=== Validation Complete ==="
