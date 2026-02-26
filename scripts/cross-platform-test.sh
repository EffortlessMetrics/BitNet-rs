#!/bin/bash
# Cross-platform compatibility testing for BitNet-rs
# Tests compilation and basic functionality across different targets

set -euo pipefail

echo "🌐 Cross-Platform Compatibility Testing"
echo "======================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

# Test targets - common targets for BitNet deployment
TARGETS=(
    "x86_64-unknown-linux-gnu"      # Linux x86_64
    "x86_64-unknown-linux-musl"     # Linux x86_64 (musl)
    "aarch64-unknown-linux-gnu"     # Linux ARM64
    "x86_64-pc-windows-msvc"        # Windows x86_64
    "x86_64-apple-darwin"           # macOS Intel
    "aarch64-apple-darwin"          # macOS Apple Silicon
    "wasm32-unknown-unknown"        # WebAssembly
)

# Features to test
FEATURE_SETS=(
    "minimal"
    "cpu"
    "cpu,avx2"
    "cpu,neon"
)

# Check if cross is installed
if ! command -v cross &> /dev/null; then
    print_warning "cross not found, installing..."
    cargo install cross
fi

# Function to test a target with specific features
test_target_features() {
    local target=$1
    local features=$2

    print_status "Testing $target with features: $features"

    # Skip GPU features for cross-compilation
    if [[ "$features" == *"gpu"* ]]; then
        print_warning "Skipping GPU features for cross-compilation"
        return 0
    fi

    # Skip NEON on non-ARM targets
    if [[ "$features" == *"neon"* && "$target" != *"aarch64"* ]]; then
        print_warning "Skipping NEON features on non-ARM target"
        return 0
    fi

    # Skip AVX on non-x86 targets
    if [[ "$features" == *"avx"* && "$target" != *"x86_64"* ]]; then
        print_warning "Skipping AVX features on non-x86 target"
        return 0
    fi

    # Use cross for cross-compilation, cargo for native
    local build_cmd="cross"
    if [[ "$target" == "$(rustc -vV | grep host | cut -d' ' -f2)" ]]; then
        build_cmd="cargo"
    fi

    # Test compilation
    if $build_cmd build --target "$target" --features "$features" --workspace; then
        print_success "✓ $target ($features) - Build successful"
        return 0
    else
        print_error "✗ $target ($features) - Build failed"
        return 1
    fi
}

# Main testing loop
failed_tests=0
total_tests=0

print_status "Starting cross-platform compatibility tests..."

for target in "${TARGETS[@]}"; do
    print_status "Testing target: $target"

    # Check if target is installed
    if ! rustup target list --installed | grep -q "$target"; then
        print_status "Installing target: $target"
        rustup target add "$target" || {
            print_warning "Failed to install target $target, skipping..."
            continue
        }
    fi

    for features in "${FEATURE_SETS[@]}"; do
        total_tests=$((total_tests + 1))

        if ! test_target_features "$target" "$features"; then
            failed_tests=$((failed_tests + 1))
        fi
    done

    echo ""
done

# Test WebAssembly specifically
print_status "Testing WebAssembly compilation..."
if command -v wasm-pack &> /dev/null; then
    cd crates/bitnet-wasm
    if wasm-pack build --target web --features browser; then
        print_success "✓ WebAssembly build successful"
    else
        print_error "✗ WebAssembly build failed"
        failed_tests=$((failed_tests + 1))
    fi
    cd - > /dev/null
    total_tests=$((total_tests + 1))
else
    print_warning "wasm-pack not found, skipping WebAssembly test"
fi

# Summary
echo ""
echo "📊 Cross-Platform Test Summary"
echo "=============================="
echo "Total tests: $total_tests"
echo "Passed: $((total_tests - failed_tests))"
echo "Failed: $failed_tests"

if [[ $failed_tests -eq 0 ]]; then
    print_success "🎉 All cross-platform tests passed!"
    exit 0
else
    print_error "❌ $failed_tests tests failed"
    exit 1
fi
