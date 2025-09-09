# BitNet.rs just commands for nextest

# CPU lane tests
perf-cpu:
    cargo nextest run --workspace --no-default-features --features cpu

# GPU lane tests (strict mode)
perf-gpu:
    BITNET_STRICT_NO_FAKE_GPU=1 \
    cargo nextest run -p bitnet-kernels --no-default-features --features gpu

# Strict tokenizer tests (no mock fallbacks)
check-strict:
    BITNET_STRICT_TOKENIZERS=1 cargo nextest run -p bitnet-tokenizers
    BITNET_STRICT_NO_FAKE_GPU=1 cargo nextest run -p bitnet-kernels --no-default-features --features gpu

# List all tests
list:
    cargo nextest list --workspace

# Run the verification script
verify:
    ./scripts/verify-tests.sh