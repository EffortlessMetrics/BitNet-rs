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
cuda:
    cargo build --workspace --no-default-features --features cuda
    cargo test -p bitnet-kernels --no-default-features --features cuda
    cargo run -p bitnet-kernels --example cuda_smoke --no-default-features --features cuda

# Alias for backward compatibility
gpu: cuda

# Build with FFI support (CPU)
ffi TAG="main":
    cargo run -p xtask -- fetch-cpp --tag {{TAG}}
    cargo build --workspace --no-default-features --features "cpu,ffi"
    cargo test -p bitnet-crossval --features "crossval,ffi"

# Build with FFI support (CUDA)
ffi-cuda TAG="main" ARCHS="80;86":
    cargo run -p xtask -- fetch-cpp --tag {{TAG}} --backend cuda --cmake-flags "-DCMAKE_CUDA_ARCHITECTURES={{ARCHS}}"
    cargo build --workspace --no-default-features --features "cuda,ffi"
    cargo test -p bitnet-crossval --features "crossval,ffi"

# Run full cross-validation suite (CPU)
crossval TAG="main":
    cargo run -p xtask -- full-crossval --tag {{TAG}}

# Run full cross-validation suite (CUDA)
crossval-cuda TAG="main" ARCHS="80;86":
    cargo run -p xtask -- full-crossval --tag {{TAG}} --backend cuda --cmake-flags "-DCMAKE_CUDA_ARCHITECTURES={{ARCHS}}"

# Compare metrics for regression detection
compare-metrics BASELINE="baselines/cpu-main.json" CURRENT="crossval/results/last_run.json":
    cargo run -p xtask -- compare-metrics --baseline {{BASELINE}} --current {{CURRENT}}

# Check for API breaking changes
check-breaking BASELINE="origin/main":
    cargo run -p xtask -- detect-breaking --baseline {{BASELINE}} --current .

# Run API snapshot tests
test-api-snapshots:
    cargo test --test api_snapshots
    @echo "✅ API snapshots verified"

# Update API snapshots (use when intentionally changing API)
update-api-snapshots:
    cargo test --test api_snapshots -- --accept
    @echo "📸 API snapshots updated - remember to commit the changes!"

# Update API baselines (when making intentional API changes)
update-api-baselines:
    @echo "Updating Rust API baselines..."
    cargo public-api -p bitnet-common > api/rust/bitnet-common.public-api.txt || true
    cargo public-api -p bitnet-kernels > api/rust/bitnet-kernels.public-api.txt || true
    cargo public-api -p bitnet-inference > api/rust/bitnet-inference.public-api.txt || true
    cargo public-api -p bitnet-ffi > api/rust/bitnet-ffi.public-api.txt || true
    cargo public-api -p bitnet-cli > api/rust/bitnet-cli.public-api.txt || true
    @echo "Updating FFI header..."
    cbindgen crates/bitnet-ffi --config api/ffi/cbindgen.toml -o api/ffi/bitnet_ffi.h || true
    @echo "Updating FFI symbols..."
    cargo build -p bitnet-ffi --release --no-default-features --features ffi
    nm -D --defined-only target/release/libbitnet_ffi.so | awk '{print $3}' | sort > api/ffi/ffi.symbols.txt || true
    @echo "Updating CLI help..."
    cargo run -p bitnet-cli -- --help > api/cli/help.txt || true
    @echo "📝 API baselines updated - review changes and update API_CHANGES.md!"

# Full API compatibility check
api-check: check-breaking test-api-snapshots
    @echo "✅ API compatibility verified"

# Verify API baselines match current code
verify-api-baselines:
    @echo "Verifying API baselines..."
    @for crate in bitnet-common bitnet-kernels bitnet-inference bitnet-ffi bitnet-cli; do \
        echo "Checking $$crate..."; \
        cargo public-api -p $$crate > /tmp/$$crate.current.txt || true; \
        diff -q api/rust/$$crate.public-api.txt /tmp/$$crate.current.txt || echo "⚠️  $$crate baseline differs"; \
    done
    @echo "✅ API baseline verification complete"

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

# CI simulation - run all checks locally (CPU)
ci-cpu:
    cargo fmt --all -- --check
    cargo clippy --workspace --no-default-features --features cpu --exclude bitnet-py -- -D warnings
    cargo build --workspace --no-default-features --features cpu
    cargo test --workspace --no-default-features --features cpu --lib --exclude bitnet-py
    @echo "✅ All CPU CI checks passed!"

# CI simulation - run all checks locally (CUDA)
ci-cuda TAG="main" ARCHS="80;86":
    cargo run -p xtask -- fetch-cpp --tag {{TAG}} --backend cuda --cmake-flags "-DCMAKE_CUDA_ARCHITECTURES={{ARCHS}}"
    cargo build --workspace --no-default-features --features "cuda,ffi"
    cargo test -p bitnet-kernels --features cuda --tests
    cargo run -p xtask -- full-crossval --tag {{TAG}} --backend cuda --cmake-flags "-DCMAKE_CUDA_ARCHITECTURES={{ARCHS}}"
    @echo "✅ All CUDA CI checks passed!"

# Quick CI check (default)
ci: ci-cpu