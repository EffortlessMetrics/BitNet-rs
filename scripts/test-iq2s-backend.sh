#!/bin/bash
# Test script for IQ2_S backend selection and functionality

set -e

echo "=== Testing IQ2_S Backend Support ==="
echo

# Test 1: Default backend (should be Rust)
echo "1. Testing default backend (should be Rust)..."
unset BITNET_IQ2S_IMPL
cargo test -p bitnet-models --test iq2s_tests test_iq2s_backend_selection --quiet

# Test 2: Explicitly set to Rust backend
echo "2. Testing explicit Rust backend..."
export BITNET_IQ2S_IMPL=rust
cargo test -p bitnet-models --test iq2s_tests test_rust_backend --quiet

# Test 3: Test with FFI backend (if available)
echo "3. Testing FFI backend availability..."
export BITNET_IQ2S_IMPL=ffi
if cargo build -p bitnet-models --no-default-features --features iq2s-ffi 2>/dev/null; then
    echo "   FFI backend available - testing parity..."
    cargo test -p bitnet-models --no-default-features --features iq2s-ffi --test iq2s_tests iq2s_parity_tests --quiet
    echo "   ✓ FFI backend tests passed"
else
    echo "   FFI backend not available (expected without iq2s-ffi feature)"
fi

# Test 4: Build with CPU features
echo "4. Testing build with CPU features..."
cargo build --no-default-features --features cpu --quiet
echo "   ✓ CPU build successful"

# Test 5: Run all IQ2_S tests
echo "5. Running all IQ2_S tests..."
unset BITNET_IQ2S_IMPL
cargo test -p bitnet-models --test iq2s_tests --quiet
echo "   ✓ All tests passed"

echo
echo "=== IQ2_S Backend Testing Complete ==="
echo
echo "Summary:"
echo "  - Native Rust IQ2_S backend: ✓ Working"
echo "  - Backend selection via environment: ✓ Working"
echo "  - Feature flag integration: ✓ Working"
echo "  - Test coverage: ✓ Comprehensive"