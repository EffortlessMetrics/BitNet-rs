#!/bin/bash
set -euo pipefail

# Color output helpers
red()  { printf "\033[31m%s\033[0m\n" "$*"; }
green(){ printf "\033[32m%s\033[0m\n" "$*"; }

# Ensure we actually execute tests (not just filter them out)
require_tests() {
  local expr="$1"
  # Robust counter: counts lines that represent test entries.
  # If you have `jq`, prefer the JSON path (commented below).
  local n
  n=$(cargo nextest list -E "$expr" --workspace | awk '/^ *[^\s].*::/ {n++} END{print n+0}')
  # jq alternative (uncomment if jq available):
  # n=$(cargo nextest list -E "$expr" --workspace --message-format json | jq '[.tests[]] | length')
  if [ "$n" -eq 0 ]; then red "no tests discovered for filter: $expr"; exit 1; fi
  green "discovered $n tests for: $expr"
}

# Source concurrency caps and preflight checks
if [[ -f "$(dirname "$0")/preflight.sh" ]]; then
    source "$(dirname "$0")/preflight.sh"
fi

echo "=== BitNet-rs Verification Tests ==="

# Pre-flight: ensure key test suites exist
echo "== Pre-flight test discovery =="
require_tests 'package("bitnet-quantization")'
require_tests 'package("bitnet-models")'
require_tests 'package("bitnet-inference")'

# GPU test discovery with CPU-only bypass
if [ "${CI_NO_GPU:-}" != "1" ]; then
  echo "Discovering GPU tests..."
  require_tests 'package("bitnet-kernels") and test(~gpu)'
else
  green "Skipping GPU discovery (CI_NO_GPU=1)"
fi

# 1) Base build (no extra features)
echo "Testing base build (no extra features)..."
cargo check -p bitnet-inference --no-default-features

# 2) Build with async runtime features
echo "Testing build with rt-tokio features..."
cargo check -p bitnet-inference --no-default-features --features rt-tokio

echo "== Run CPU lane =="
cargo nextest run --workspace --no-default-features --features cpu

if [ "${CI_NO_GPU:-}" != "1" ]; then
  echo "== Run GPU lane =="
  BITNET_STRICT_NO_FAKE_GPU=1 \
  cargo nextest run -p bitnet-kernels --no-default-features --features gpu
fi

echo "== Strict edges =="
BITNET_STRICT_TOKENIZERS=1 cargo nextest run -p bitnet-tokenizers

# Pure parser test
echo "== Running GGUF header parser tests =="
cargo nextest run -p bitnet-inference --test gguf_header

# Async smoke with stub
echo "== Creating tiny GGUF stub and running smoke test =="
printf "GGUF\x02\x00\x00\x00" > /tmp/t.gguf
printf "\x00\x00\x00\x00\x00\x00\x00\x00" >> /tmp/t.gguf
printf "\x00\x00\x00\x00\x00\x00\x00\x00" >> /tmp/t.gguf
BITNET_GGUF=/tmp/t.gguf cargo nextest run -p bitnet-inference --no-default-features --features rt-tokio --test smoke

echo "=== All verification tests completed successfully ==="