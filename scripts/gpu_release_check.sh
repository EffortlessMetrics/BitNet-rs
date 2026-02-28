#!/usr/bin/env bash
# gpu_release_check.sh — Automated GPU pre-release verification
#
# Runs all GPU validation steps for release readiness:
#   1. Feature flag compilation matrix
#   2. Kernel test suite
#   3. Documentation currency
#   4. CHANGELOG audit
#   5. Clippy & formatting
#
# Usage:
#   ./scripts/gpu_release_check.sh           # Full check
#   ./scripts/gpu_release_check.sh --quick   # Skip slow compilation matrix
#   ./scripts/gpu_release_check.sh --verbose # Show all command output
#
# Exit codes: 0=pass, 1=compile, 2=tests, 3=docs, 4=changelog, 5=lint
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Flags ──────────────────────────────────────────────────────────────────────
QUICK=false
VERBOSE=false
for arg in "$@"; do
  case "$arg" in
    --quick)   QUICK=true ;;
    --verbose) VERBOSE=true ;;
    --help|-h) echo "Usage: $0 [--quick] [--verbose]"; exit 0 ;;
    *) echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

PASS=0
FAIL=0
SKIP=0

# ── Helpers ────────────────────────────────────────────────────────────────────
info()  { echo "  [INFO]  $*"; }
ok()    { echo "  [PASS]  $*"; PASS=$((PASS + 1)); }
fail()  { echo "  [FAIL]  $*"; FAIL=$((FAIL + 1)); }
skip()  { echo "  [SKIP]  $*"; SKIP=$((SKIP + 1)); }

run_cmd() {
  if $VERBOSE; then
    "$@"
  else
    "$@" >/dev/null 2>&1
  fi
}

# ── 1. Feature flag compilation matrix ─────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  GPU Release Check — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "── Step 1: Feature flag compilation matrix ──"

FEATURE_SETS=(
  "--no-default-features --features cpu"
  "--no-default-features --features gpu"
  "--no-default-features --features cuda"
  "--no-default-features --features cpu,gpu"
  "--no-default-features --features cpu,cuda"
)

if $QUICK; then
  skip "Feature compilation matrix (--quick mode)"
else
  cd "$REPO_ROOT"
  compile_fail=false
  for features in "${FEATURE_SETS[@]}"; do
    info "Checking: cargo check $features"
    if run_cmd cargo check $features; then
      ok "cargo check $features"
    else
      fail "cargo check $features"
      compile_fail=true
    fi
  done
  if $compile_fail; then
    echo ""
    echo "FATAL: Feature compilation matrix failed"
    exit 1
  fi
fi

# ── 2. Kernel test suite ──────────────────────────────────────────────────────
echo ""
echo "── Step 2: Kernel test suite ──"

cd "$REPO_ROOT"

# Run kernel tests with CPU features (should always work)
info "Running kernel tests (CPU)"
if run_cmd cargo test -p bitnet-kernels --no-default-features --features cpu -- --test-threads=1; then
  ok "bitnet-kernels CPU tests"
else
  fail "bitnet-kernels CPU tests"
fi

# Run device-probe tests
info "Running device-probe tests"
if run_cmd cargo test -p bitnet-device-probe --no-default-features --features cpu; then
  ok "bitnet-device-probe tests"
else
  fail "bitnet-device-probe tests"
fi

# GPU-fake environment test (validates GPU code paths without hardware)
info "Running GPU-fake fallback test"
if BITNET_GPU_FAKE=none run_cmd cargo test -p bitnet-device-probe --no-default-features --features cpu -- --test-threads=1; then
  ok "GPU-fake=none fallback test"
else
  fail "GPU-fake=none fallback test"
fi

# ── 3. Documentation currency ────────────────────────────────────────────────
echo ""
echo "── Step 3: Documentation currency ──"

docs_missing=false

GPU_DOCS=(
  "docs/GPU_SETUP.md"
  "docs/INTEL_GPU_SETUP.md"
  "docs/gpu-kernel-architecture.md"
  "docs/GPU_RELEASE_CHECKLIST.md"
)

for doc in "${GPU_DOCS[@]}"; do
  if [ -f "$REPO_ROOT/$doc" ]; then
    ok "$doc exists"
  else
    fail "$doc missing"
    docs_missing=true
  fi
done

# Check that GPU_SETUP.md mentions CUDA 12
if grep -q "CUDA.*12" "$REPO_ROOT/docs/GPU_SETUP.md" 2>/dev/null; then
  ok "GPU_SETUP.md references CUDA 12.x"
else
  fail "GPU_SETUP.md does not reference CUDA 12.x"
  docs_missing=true
fi

# Check that INTEL_GPU_SETUP.md mentions OpenCL
if grep -q "OpenCL" "$REPO_ROOT/docs/INTEL_GPU_SETUP.md" 2>/dev/null; then
  ok "INTEL_GPU_SETUP.md references OpenCL"
else
  fail "INTEL_GPU_SETUP.md does not reference OpenCL"
  docs_missing=true
fi

if $docs_missing; then
  echo ""
  echo "WARNING: Documentation checks had failures (exit 3)"
fi

# ── 4. CHANGELOG audit ──────────────────────────────────────────────────────
echo ""
echo "── Step 4: CHANGELOG audit ──"

if [ -f "$REPO_ROOT/CHANGELOG.md" ]; then
  # Check for GPU-related entries under [Unreleased]
  # Extract the Unreleased section and search for GPU keywords
  unreleased=$(sed -n '/^## \[Unreleased\]/,/^## \[/p' "$REPO_ROOT/CHANGELOG.md" | head -50)

  if echo "$unreleased" | grep -qi "gpu\|cuda\|kernel\|opencl\|oneapi\|metal\|vulkan"; then
    ok "CHANGELOG has GPU-related entries under [Unreleased]"
  else
    fail "CHANGELOG has no GPU-related entries under [Unreleased]"
  fi
else
  fail "CHANGELOG.md not found"
fi

# ── 5. Clippy & formatting ──────────────────────────────────────────────────
echo ""
echo "── Step 5: Clippy & formatting ──"

cd "$REPO_ROOT"

info "Checking formatting"
if run_cmd cargo fmt --all -- --check; then
  ok "cargo fmt --check"
else
  fail "cargo fmt --check"
fi

info "Running clippy (CPU features)"
if run_cmd cargo clippy --all-targets --no-default-features --features cpu -- -D warnings; then
  ok "clippy --features cpu"
else
  fail "clippy --features cpu"
fi

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Summary: $PASS passed, $FAIL failed, $SKIP skipped"
echo "═══════════════════════════════════════════════════════════════"
echo ""

if [ "$FAIL" -gt 0 ]; then
  echo "RESULT: FAIL — $FAIL check(s) did not pass"
  # Return most specific exit code
  exit 1
fi

echo "RESULT: PASS — All GPU release checks passed"
exit 0
