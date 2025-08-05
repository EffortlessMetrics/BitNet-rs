#!/bin/bash
# Comprehensive code quality validation for BitNet.rs
# This script runs all quality checks required for crates.io publication

set -euo pipefail

echo "🔍 BitNet.rs Code Quality Validation"
echo "===================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print status
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

# Check if we're in the right directory
if [[ ! -f "Cargo.toml" ]] || [[ ! -d "crates" ]]; then
    print_error "This script must be run from the BitNet.rs root directory"
    exit 1
fi

# Check Rust version
print_status "Checking Rust version..."
RUST_VERSION=$(rustc --version | cut -d' ' -f2)
REQUIRED_VERSION="1.70.0"
if ! printf '%s\n%s\n' "$REQUIRED_VERSION" "$RUST_VERSION" | sort -V -C; then
    print_error "Rust version $RUST_VERSION is below required $REQUIRED_VERSION"
    exit 1
fi
print_success "Rust version $RUST_VERSION meets requirements"

# Check for required tools
print_status "Checking required tools..."
REQUIRED_TOOLS=("cargo-fmt" "cargo-clippy" "cargo-audit" "cargo-deny")
for tool in "${REQUIRED_TOOLS[@]}"; do
    if ! command -v "$tool" &> /dev/null; then
        print_error "$tool is not installed"
        echo "Install with: cargo install $tool"
        exit 1
    fi
done
print_success "All required tools are available"

# Format check
print_status "Checking code formatting..."
if ! cargo fmt --all -- --check; then
    print_error "Code is not properly formatted"
    echo "Run: cargo fmt --all"
    exit 1
fi
print_success "Code formatting is correct"

# Clippy check with pedantic lints
print_status "Running Clippy with pedantic lints..."
if ! cargo clippy --all-targets --all-features -- -D warnings -D clippy::pedantic; then
    print_error "Clippy found issues"
    exit 1
fi
print_success "Clippy checks passed"

# Build check
print_status "Building all crates..."
if ! cargo build --workspace --all-features; then
    print_error "Build failed"
    exit 1
fi
print_success "Build successful"

# Test check
print_status "Running tests..."
if ! cargo test --workspace --all-features; then
    print_error "Tests failed"
    exit 1
fi
print_success "All tests passed"

# Documentation check
print_status "Checking documentation..."
if ! cargo doc --workspace --all-features --no-deps; then
    print_error "Documentation generation failed"
    exit 1
fi
print_success "Documentation generated successfully"

# Security audit
print_status "Running security audit..."
if ! cargo audit; then
    print_error "Security audit found vulnerabilities"
    exit 1
fi
print_success "Security audit passed"

# License compliance check
print_status "Checking license compliance..."
if ! cargo deny check; then
    print_error "License compliance check failed"
    exit 1
fi
print_success "License compliance verified"

# Check for TODO/FIXME comments in release-critical code
print_status "Checking for TODO/FIXME comments..."
TODO_COUNT=$(find src crates -name "*.rs" -exec grep -l "TODO\|FIXME" {} \; | wc -l)
if [[ $TODO_COUNT -gt 0 ]]; then
    print_warning "Found $TODO_COUNT files with TODO/FIXME comments"
    find src crates -name "*.rs" -exec grep -Hn "TODO\|FIXME" {} \;
    echo "Consider resolving these before release"
fi

# Check crate metadata completeness
print_status "Validating crate metadata..."
REQUIRED_FIELDS=("description" "license" "repository" "homepage" "keywords" "categories")
for crate_dir in . crates/*/; do
    if [[ -f "$crate_dir/Cargo.toml" ]]; then
        crate_name=$(basename "$crate_dir")
        print_status "Checking metadata for $crate_name..."
        
        for field in "${REQUIRED_FIELDS[@]}"; do
            if ! grep -q "^$field\s*=" "$crate_dir/Cargo.toml" && ! grep -q "^$field\.workspace\s*=" "$crate_dir/Cargo.toml"; then
                print_error "Missing $field in $crate_dir/Cargo.toml"
                exit 1
            fi
        done
    fi
done
print_success "All crate metadata is complete"

# Check for proper feature flags
print_status "Validating feature flags..."
if ! grep -q '\[features\]' Cargo.toml; then
    print_error "Main crate is missing [features] section"
    exit 1
fi
print_success "Feature flags are properly configured"

# Check README exists and is not empty
print_status "Checking README..."
if [[ ! -f "README.md" ]] || [[ ! -s "README.md" ]]; then
    print_error "README.md is missing or empty"
    exit 1
fi
print_success "README.md exists and is not empty"

# Check CHANGELOG exists
print_status "Checking CHANGELOG..."
if [[ ! -f "CHANGELOG.md" ]]; then
    print_error "CHANGELOG.md is missing"
    exit 1
fi
print_success "CHANGELOG.md exists"

# Check examples compile
print_status "Checking examples..."
if ! cargo check --examples --all-features; then
    print_error "Examples failed to compile"
    exit 1
fi
print_success "All examples compile successfully"

# Performance regression check (if benchmarks exist)
if [[ -d "benches" ]]; then
    print_status "Running benchmark compilation check..."
    if ! cargo bench --no-run --workspace; then
        print_error "Benchmarks failed to compile"
        exit 1
    fi
    print_success "Benchmarks compile successfully"
fi

# Final summary
echo ""
echo "🎉 All quality checks passed!"
echo "================================"
print_success "Code formatting: ✓"
print_success "Clippy lints: ✓"
print_success "Build: ✓"
print_success "Tests: ✓"
print_success "Documentation: ✓"
print_success "Security audit: ✓"
print_success "License compliance: ✓"
print_success "Crate metadata: ✓"
print_success "Feature flags: ✓"
print_success "README/CHANGELOG: ✓"
print_success "Examples: ✓"

echo ""
echo "🚀 BitNet.rs is ready for crates.io publication!"
echo ""
echo "Next steps:"
echo "1. Update version numbers if needed"
echo "2. Update CHANGELOG.md with release notes"
echo "3. Create a git tag for the release"
echo "4. Run 'cargo publish --dry-run' to verify"
echo "5. Run 'cargo publish' to publish to crates.io"