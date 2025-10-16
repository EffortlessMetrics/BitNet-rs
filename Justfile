# BitNet.rs development commands

# Run all pre-tag MVP validation checks
mvp-pretag:
    cargo clippy --workspace --all-targets --no-default-features --features cpu -D warnings
    cargo test --workspace --no-default-features --features cpu
    BITNET_STRICT_MODE=1 BITNET_DETERMINISTIC=1 RAYON_NUM_THREADS=1 \
    cargo run -p xtask --no-default-features --features inference -- \
      verify-receipt --path docs/baselines/20251015-cpu.json

# Build release binaries and generate checksums
mvp-build:
    cargo build --release -p bitnet-cli --no-default-features --features cpu,full-cli
    cargo build --release -p bitnet-st2gguf
    cd target/release && shasum -a 256 bitnet st2gguf > SHA256SUMS && cat SHA256SUMS

# Smoke test gate validation (expect failures on bad receipts)
gate-smoke:
    @echo "Testing mocked compute path (should fail)..."
    jq '.compute_path = "mock"' docs/baselines/20251015-cpu.json > /tmp/bad-compute.json
    cargo run -p xtask --no-default-features --features inference -- \
      verify-receipt --path /tmp/bad-compute.json && echo "❌ UNEXPECTED PASS" || echo "✓ EXPECTED FAIL"
    @echo ""
    @echo "Testing empty kernels (should fail)..."
    jq '.kernels = []' docs/baselines/20251015-cpu.json > /tmp/bad-kernels.json
    cargo run -p xtask --no-default-features --features inference -- \
      verify-receipt --path /tmp/bad-kernels.json && echo "❌ UNEXPECTED PASS" || echo "✓ EXPECTED FAIL"
    @echo ""
    @echo "Testing invalid schema version (should fail)..."
    jq '.schema_version = "0.9.0"' docs/baselines/20251015-cpu.json > /tmp/bad-schema.json
    cargo run -p xtask --no-default-features --features inference -- \
      verify-receipt --path /tmp/bad-schema.json && echo "❌ UNEXPECTED PASS" || echo "✓ EXPECTED FAIL"

# Run CPU test suite with coverage
test-cpu:
    cargo test --workspace --no-default-features --features cpu

# Run TL stress tests
test-tl-stress:
    cargo test -p bitnet-kernels --no-default-features --features cpu \
      --test fuzz_tl_lut_stress -- --test-threads=1

# Build and verify documentation
docs:
    cargo doc --workspace --no-default-features --features cpu --no-deps
    cargo test --doc --workspace --no-default-features --features cpu

# Format and lint code
fmt:
    cargo fmt --all
    cargo clippy --workspace --all-targets --all-features -- -D warnings

# Generate a new CPU receipt baseline
receipt-baseline MODEL_PATH='models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf':
    BITNET_STRICT_MODE=1 BITNET_DETERMINISTIC=1 RAYON_NUM_THREADS=1 \
    cargo run -p xtask --no-default-features --features inference -- \
      benchmark --model {{MODEL_PATH}} --tokens 128 --deterministic
    cp ci/inference.json docs/baselines/$(date +%Y%m%d)-cpu.json
    @echo "Baseline saved to docs/baselines/$(date +%Y%m%d)-cpu.json"

# Clean build artifacts
clean:
    cargo clean
    rm -rf target/
