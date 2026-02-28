#!/bin/bash
# Run mutation testing on OpenCL-related modules.
#
# Requires: cargo-mutants (cargo install cargo-mutants)
#
# This script focuses mutations on the Rust orchestration logic around
# OpenCL / oneAPI — not on the embedded .cl kernel source strings which
# can only be validated on real GPU hardware.
#
# Usage:
#   ./scripts/mutant_opencl.sh                # default: bitnet-kernels
#   ./scripts/mutant_opencl.sh --all          # also mutate bitnet-device-probe
#
# Exit code 0: all mutants caught or timed out.
# Exit code 1: some mutants survived (potential missing test coverage).

set -euo pipefail

FEATURES="oneapi,cpu"

echo "=== OpenCL mutation testing ==="
echo "Features: ${FEATURES}"
echo ""

# Core: bitnet-kernels OpenCL provider and device features
echo "▸ Mutating bitnet-kernels (opencl + device_features)..."
cargo mutants \
  --no-default-features --features "${FEATURES}" \
  -p bitnet-kernels \
  --file "src/gpu/opencl.rs" \
  --file "src/device_features.rs" \
  -- --test-threads=1

# If --all passed, also mutate the device-probe crate
if [[ "${1:-}" == "--all" ]]; then
  echo ""
  echo "▸ Mutating bitnet-device-probe (oneapi detection)..."
  cargo mutants \
    --no-default-features --features "${FEATURES}" \
    -p bitnet-device-probe \
    -- --test-threads=1
fi

echo ""
echo "=== OpenCL mutation testing complete ==="
