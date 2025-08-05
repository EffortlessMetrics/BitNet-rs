#!/bin/bash
# Script to run Miri tests for undefined behavior detection

set -e

echo "Setting up Miri for undefined behavior detection..."

# Install Rust nightly if not present
if ! rustup toolchain list | grep -q nightly; then
    echo "Installing Rust nightly..."
    rustup toolchain install nightly
fi

# Install Miri component
echo "Installing Miri..."
rustup +nightly component add miri

# Setup Miri
echo "Setting up Miri..."
cargo +nightly miri setup

# List of crates to test with Miri (focusing on those with unsafe code)
CRATES=(
    "bitnet-common"
    "bitnet-kernels" 
    "bitnet-quantization"
    "bitnet-models"
)

echo "Running Miri tests..."

failed_crates=()

for crate in "${CRATES[@]}"; do
    echo "Testing $crate with Miri..."
    
    if [ -d "crates/$crate" ]; then
        cd "crates/$crate"
        
        # Run Miri tests
        if cargo +nightly miri test; then
            echo "✅ Miri tests passed for $crate"
        else
            echo "❌ Miri tests failed for $crate"
            failed_crates+=("$crate")
        fi
        
        cd ../..
    else
        echo "⚠️  Crate directory not found: crates/$crate"
    fi
    
    echo ""
done

# Run Miri on specific test files that exercise unsafe code
echo "Running Miri on integration tests..."
if cargo +nightly miri test --test integration_security; then
    echo "✅ Integration security tests passed with Miri"
else
    echo "❌ Integration security tests failed with Miri"
    failed_crates+=("integration_tests")
fi

# Summary
echo "=== Miri Testing Summary ==="
if [ ${#failed_crates[@]} -eq 0 ]; then
    echo "✅ All Miri tests passed!"
    echo "No undefined behavior detected."
else
    echo "❌ Failed crates/tests: ${failed_crates[*]}"
    echo "Undefined behavior may be present in the failed components."
    exit 1
fi

echo "Miri testing completed!"