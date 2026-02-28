#!/usr/bin/env bash
# run-gpu-tests.sh — Local GPU test runner for developers.
# Runs feature-matrix builds, stress tests, and benchmarks.
# Detects available GPU hardware and adjusts tests accordingly.
#
# Usage:
#   ./scripts/run-gpu-tests.sh              # Run all tests
#   ./scripts/run-gpu-tests.sh --quick      # Skip benchmarks and stress
#   ./scripts/run-gpu-tests.sh --bench-only # Run only benchmarks

set -euo pipefail

# ---------- colour helpers ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
pass()  { echo -e "${GREEN}[PASS]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }

# ---------- parse arguments ----------
QUICK=false
BENCH_ONLY=false

for arg in "$@"; do
  case "$arg" in
    --quick)      QUICK=true ;;
    --bench-only) BENCH_ONLY=true ;;
    --help|-h)
      echo "Usage: $0 [--quick] [--bench-only]"
      echo ""
      echo "Options:"
      echo "  --quick       Skip benchmarks and stress tests"
      echo "  --bench-only  Run only benchmarks"
      echo "  --help        Show this help"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg"
      exit 1
      ;;
  esac
done

# ---------- detect hardware ----------
HAS_CUDA=false
HAS_AVX2=false
FEATURES="cpu"

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
  HAS_CUDA=true
  info "CUDA GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
fi

if grep -q avx2 /proc/cpuinfo 2>/dev/null; then
  HAS_AVX2=true
  info "AVX2 support detected"
elif sysctl -n machdep.cpu.features 2>/dev/null | grep -qi avx2; then
  HAS_AVX2=true
  info "AVX2 support detected"
fi

# ---------- summary state ----------
TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0
RESULTS=()

run_step() {
  local name="$1"
  shift
  TOTAL=$((TOTAL + 1))
  info "Running: $name"
  if "$@" 2>&1; then
    pass "$name"
    PASSED=$((PASSED + 1))
    RESULTS+=("PASS  $name")
  else
    fail "$name"
    FAILED=$((FAILED + 1))
    RESULTS+=("FAIL  $name")
  fi
}

skip_step() {
  local name="$1"
  local reason="$2"
  TOTAL=$((TOTAL + 1))
  SKIPPED=$((SKIPPED + 1))
  warn "Skipping: $name ($reason)"
  RESULTS+=("SKIP  $name — $reason")
}

# ---------- feature-matrix builds ----------
if [ "$BENCH_ONLY" = false ]; then
  info "=== Feature Matrix Builds ==="

  run_step "Build (cpu)" \
    cargo build --workspace --no-default-features --features cpu

  if [ "$HAS_AVX2" = true ]; then
    run_step "Build (cpu+avx2)" \
      env RUSTFLAGS="-C target-feature=+avx2" \
        cargo build --workspace --no-default-features --features cpu,avx2
  else
    skip_step "Build (cpu+avx2)" "AVX2 not available"
  fi

  if [ "$HAS_CUDA" = true ]; then
    run_step "Build (gpu)" \
      cargo build --workspace --no-default-features --features gpu

    run_step "Build (cpu+gpu)" \
      cargo build --workspace --no-default-features --features cpu,gpu
  else
    skip_step "Build (gpu)" "No CUDA GPU"
    skip_step "Build (cpu+gpu)" "No CUDA GPU"
  fi

  # ---------- test suite ----------
  info "=== Test Suite ==="

  run_step "Test (cpu)" \
    cargo test --workspace --no-default-features --features cpu

  if [ "$HAS_CUDA" = true ]; then
    run_step "Test (gpu)" \
      cargo test --workspace --no-default-features --features gpu
  else
    skip_step "Test (gpu)" "No CUDA GPU"
  fi

  # ---------- stress tests ----------
  if [ "$QUICK" = false ]; then
    info "=== Stress Tests ==="

    run_step "Stress: kernels" \
      cargo test -p bitnet-kernels --no-default-features --features cpu \
        -- --ignored --nocapture

    run_step "Stress: quantization" \
      cargo test -p bitnet-quantization --no-default-features --features cpu \
        -- --ignored --nocapture
  else
    skip_step "Stress: kernels" "--quick mode"
    skip_step "Stress: quantization" "--quick mode"
  fi
fi

# ---------- benchmarks ----------
if [ "$QUICK" = false ]; then
  info "=== Benchmarks ==="

  run_step "Benchmarks (cpu)" \
    cargo bench --no-default-features --features cpu -- --output-format bencher
else
  if [ "$BENCH_ONLY" = false ]; then
    skip_step "Benchmarks (cpu)" "--quick mode"
  fi
fi

if [ "$BENCH_ONLY" = true ]; then
  run_step "Benchmarks (cpu)" \
    cargo bench --no-default-features --features cpu -- --output-format bencher
fi

# ---------- summary report ----------
echo ""
echo "============================================"
echo "  GPU Test Summary"
echo "============================================"
echo ""
for r in "${RESULTS[@]}"; do
  case "$r" in
    PASS*) echo -e "  ${GREEN}$r${NC}" ;;
    FAIL*) echo -e "  ${RED}$r${NC}" ;;
    SKIP*) echo -e "  ${YELLOW}$r${NC}" ;;
  esac
done
echo ""
echo "  Total: $TOTAL  Passed: $PASSED  Failed: $FAILED  Skipped: $SKIPPED"
echo "============================================"

if [ "$FAILED" -gt 0 ]; then
  exit 1
fi
