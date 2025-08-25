#!/bin/bash
set -euo pipefail

echo "=== BitNet-rs Verification Tests ==="

# 1) Base build (no extra features)
echo "Testing base build (no extra features)..."
cargo check -p bitnet-inference --no-default-features

# 2) Build with async runtime features
echo "Testing build with rt-tokio features..."
cargo check -p bitnet-inference --no-default-features --features rt-tokio

# 3) Pure parser test
echo "Running pure header parser tests..."
cargo test -p bitnet-inference --test gguf_header

# 4) Async smoke (skip if no env)
echo "Running async smoke test (will skip if BITNET_GGUF not set)..."
cargo test -p bitnet-inference --no-default-features --features rt-tokio --test smoke -- --nocapture || true

# 5) Async smoke with stub
echo "Creating tiny GGUF stub and running smoke test..."
printf "GGUF\x02\x00\x00\x00" > /tmp/t.gguf
printf "\x00\x00\x00\x00\x00\x00\x00\x00" >> /tmp/t.gguf
printf "\x00\x00\x00\x00\x00\x00\x00\x00" >> /tmp/t.gguf
BITNET_GGUF=/tmp/t.gguf cargo test -p bitnet-inference --no-default-features --features rt-tokio --test smoke

echo "=== All verification tests completed successfully ==="