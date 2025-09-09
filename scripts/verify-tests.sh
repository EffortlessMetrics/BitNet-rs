#!/bin/bash
set -euo pipefail

# Color output helpers
red()  { printf "\033[31m%s\033[0m\n" "$*"; }
green(){ printf "\033[32m%s\033[0m\n" "$*"; }

# Ensure we actually execute tests (not just filter them out)
require_tests() {
  local pkg="$1"; shift
  local feats="$1"; shift
  local list
  echo "== Listing tests for ${pkg} (${feats:-no features}) =="
  if ! list=$(cargo test -p "${pkg}" ${feats:+--no-default-features --features ${feats}} -- --list 2>/dev/null); then
    red "failed to list tests for ${pkg}"
    exit 1
  fi
  local count
  count=$(printf "%s\n" "$list" | grep -E ': test$' | wc -l | xargs)
  if [ "$count" -eq 0 ]; then
    red "no tests discovered for ${pkg} (${feats:-no features})"
    exit 1
  fi
  green "${pkg}: discovered ${count} tests"
}

# Source concurrency caps and preflight checks
if [[ -f "$(dirname "$0")/preflight.sh" ]]; then
    source "$(dirname "$0")/preflight.sh"
fi

echo "=== BitNet-rs Verification Tests ==="

# Pre-flight: ensure key test suites exist
echo "== Pre-flight test discovery =="
require_tests bitnet-quantization ""
require_tests bitnet-kernels ""
require_tests bitnet-models ""
require_tests bitnet-inference ""

# 1) Base build (no extra features)
echo "Testing base build (no extra features)..."
cargo check -p bitnet-inference --no-default-features

# 2) Build with async runtime features
echo "Testing build with rt-tokio features..."
cargo check -p bitnet-inference --no-default-features --features rt-tokio

# 3) Pure parser test
echo "Running pure header parser tests..."
cargo test -p bitnet-inference --test gguf_header -- --test-threads="${RUST_TEST_THREADS:-2}"

# 4) Async smoke (skip if no env)
echo "Running async smoke test (will skip if BITNET_GGUF not set)..."
cargo test -p bitnet-inference --no-default-features --features rt-tokio --test smoke -- --test-threads="${RUST_TEST_THREADS:-2}" --nocapture || true

# 5) Async smoke with stub
echo "Creating tiny GGUF stub and running smoke test..."
printf "GGUF\x02\x00\x00\x00" > /tmp/t.gguf
printf "\x00\x00\x00\x00\x00\x00\x00\x00" >> /tmp/t.gguf
printf "\x00\x00\x00\x00\x00\x00\x00\x00" >> /tmp/t.gguf
BITNET_GGUF=/tmp/t.gguf cargo test -p bitnet-inference --no-default-features --features rt-tokio --test smoke -- --test-threads="${RUST_TEST_THREADS:-2}"

echo "=== All verification tests completed successfully ==="