# BitNet.rs Development Workflows
# Run 'just' to see available commands

# Default recipe - show available commands
default:
    @just --list

# Build with CPU support (default)
cpu:
    cargo build --workspace --no-default-features --features cpu
    cargo test --workspace --no-default-features --features cpu --exclude bitnet-py

# Build with GPU/CUDA support
gpu:
    cargo build --workspace --no-default-features --features cuda
    cargo test -p bitnet-kernels --no-default-features --features cuda

# Build with FFI support
ffi:
    cargo run -p xtask -- fetch-cpp
    cargo build --workspace --no-default-features --features "cpu,ffi"
    cargo test -p bitnet-crossval --features "crossval,ffi"

# Run full cross-validation suite
crossval:
    cargo run -p xtask -- full-crossval

# Format all code
fmt:
    cargo fmt --all

# Run clippy lints
lint:
    cargo clippy --workspace --no-default-features --features cpu --exclude bitnet-py

# Run all quality checks
quality: fmt lint
    cargo test --workspace --no-default-features --features cpu --exclude bitnet-py

# Clean all build artifacts and caches
clean:
    cargo run -p xtask -- clean-cache
    cargo clean

# Download model for testing
download-model:
    cargo run -p xtask -- download-model

# Quick test of critical functionality
test-quick:
    cargo test -p bitnet-kernels --no-default-features --features cpu --lib
    cargo test -p bitnet-common --lib
    cargo test -p bitnet-quantization --lib

# Full test suite
test-all:
    cargo test --workspace --no-default-features --features cpu --exclude bitnet-py
    cargo test --workspace --no-default-features --features "cpu,ffi" --exclude bitnet-py

# Build release binaries
release:
    cargo build --release --no-default-features --features cpu

# Generate code coverage
coverage:
    cargo llvm-cov --workspace --features cpu --html

# Run benchmarks
bench:
    cargo bench --workspace --no-default-features --features cpu

# Check for security vulnerabilities
audit:
    cargo audit

# Check for outdated dependencies
outdated:
    cargo outdated

# CI simulation - run all checks locally
ci: quality test-quick
    @echo "✅ All CI checks passed!"