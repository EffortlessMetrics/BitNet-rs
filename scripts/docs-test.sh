#!/bin/bash
# Test docs.rs compatibility for BitNet.rs
# This script simulates the docs.rs build environment

set -euo pipefail

echo "📚 Testing docs.rs Compatibility"
echo "================================"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

# Clean previous builds
print_status "Cleaning previous builds..."
cargo clean

# Test main crate documentation with all features
print_status "Testing main crate documentation with all features..."
RUSTDOCFLAGS="--cfg docsrs" cargo doc --all-features --no-deps

# Test individual crate documentation
print_status "Testing individual crate documentation..."
for crate_dir in crates/*/; do
    if [[ -f "$crate_dir/Cargo.toml" ]]; then
        crate_name=$(basename "$crate_dir")
        print_status "Testing documentation for $crate_name..."

        cd "$crate_dir"
        RUSTDOCFLAGS="--cfg docsrs" cargo doc --all-features --no-deps
        cd - > /dev/null
    fi
done

# Test documentation with minimal features
print_status "Testing documentation with minimal features..."
RUSTDOCFLAGS="--cfg docsrs" cargo doc --features minimal --no-deps

# Test that all public items are documented
print_status "Checking for missing documentation..."
RUSTDOCFLAGS="--cfg docsrs -D missing_docs" cargo doc --all-features --no-deps

# Test cross-references work
print_status "Testing cross-references..."
if command -v linkchecker &> /dev/null; then
    linkchecker target/doc/bitnet/index.html
else
    print_status "linkchecker not available, skipping link validation"
fi

print_success "All docs.rs compatibility tests passed!"

echo ""
echo "📖 Documentation is ready for docs.rs!"
echo ""
echo "View locally with: cargo doc --open --all-features"
