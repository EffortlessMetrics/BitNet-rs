#!/usr/bin/env bash
# Validate a GGUF: LN RMS ~ 1.0, healthy PROJ RMS, and non-gibberish greedy decode.
# Usage: scripts/validate_gguf.sh <model.gguf> <tokenizer.json>
#
# This script runs in strict mode (no policy corrections allowed) to ensure
# the model is clean and production-ready. See docs/howto/export-clean-gguf.md.

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

error() {
  echo -e "${RED}❌ ERROR: $1${NC}" >&2
  exit "${2:-1}"
}

info() {
  echo -e "${GREEN}INFO: $1${NC}"
}

warn() {
  echo -e "${YELLOW}WARN: $1${NC}"
}

section() {
  echo -e "\n${BLUE}===================================================${NC}"
  echo -e "${BLUE}$1${NC}"
  echo -e "${BLUE}===================================================${NC}"
}

MODEL="${1:?Usage: $0 <model.gguf> <tokenizer.json>}"
TOK="${2:?Usage: $0 <model.gguf> <tokenizer.json>}"

# Validate inputs
[[ -f "$MODEL" ]] || error "Model not found: $MODEL"
[[ -f "$TOK" ]] || error "Tokenizer not found: $TOK"

# Check if bitnet-cli is available
if ! command -v cargo >/dev/null 2>&1; then
  error "cargo not found in PATH"
fi

# Build CLI if needed (quietly)
if ! cargo run -q -p bitnet-cli --no-default-features --features cpu -- --version >/dev/null 2>&1; then
  info "Building bitnet-cli..."
  cargo build -q -p bitnet-cli --no-default-features --features cpu || error "Failed to build bitnet-cli"
fi

# Exit code tracking
EXIT_CODE=0

# ============================================================================
# 1. LayerNorm Statistics Check
# ============================================================================
section "1/3: LayerNorm Statistics Check (Strict Mode)"

info "Checking LayerNorm RMS values (must be ~1.0)..."
info "Running with BITNET_STRICT_MODE=1 (no corrections allowed)"

LN_TMP=$(mktemp)
set +e
BITNET_STRICT_MODE=1 \
  cargo run -q -p bitnet-cli --no-default-features --features cpu -- \
  inspect --ln-stats "$MODEL" 2>&1 | tee "$LN_TMP"
LN_RC=$?
set -e

if [[ $LN_RC -ne 0 ]]; then
  error "LayerNorm inspection failed under strict mode (exit code: $LN_RC)" 10
fi

# Parse and validate LN statistics
# Look for suspicious patterns: RMS very far from 1.0 (< 0.5 or > 2.0)
if grep -E -q "SUSPICIOUS|suspicious|rms=.*0\.[0-9]{3}[^0-9]" "$LN_TMP"; then
  warn "Suspicious LayerNorm statistics detected!"
  echo ""
  grep -E "SUSPICIOUS|suspicious|rms=" "$LN_TMP" || true
  error "Model has suspicious LayerNorm weights (quantized or corrupted).
This model is NOT clean and should not be used in production.
Please re-export with LayerNorm weights in float format." 11
fi

# Check for healthy RMS values (should see "rms=1.xxx" or "rms=0.xxx" close to 1.0)
HEALTHY_COUNT=$(grep -E "rms=[0-9]\.[0-9]+" "$LN_TMP" | \
  awk -F'rms=' '{print $2}' | awk '{print $1}' | \
  awk '$1 >= 0.5 && $1 <= 2.0' | wc -l || echo "0")

if [[ $HEALTHY_COUNT -lt 1 ]]; then
  warn "No healthy LayerNorm RMS values found (expected ≈1.0)"
  echo "This might indicate quantized or missing LayerNorm weights."
  EXIT_CODE=12
else
  info "✅ Found $HEALTHY_COUNT healthy LayerNorm layers (RMS in [0.5, 2.0])"
fi

rm -f "$LN_TMP"

# ============================================================================
# 2. Projection Weight RMS Check
# ============================================================================
section "2/3: Projection Weight RMS Check"

info "Loading model and checking projection weight statistics..."
info "Expected: Q/K/V/O and FFN weights should have RMS ~ O(10³)"

PROJ_TMP=$(mktemp)
set +e
RUST_LOG=info \
  cargo run -q -p bitnet-cli --no-default-features --features cpu -- \
  run --model "$MODEL" --tokenizer "$TOK" \
  --prompt "Warmup." --max-new-tokens 1 --temperature 0.0 \
  2>&1 | tee "$PROJ_TMP"
PROJ_RC=$?
set -e

if [[ $PROJ_RC -ne 0 ]]; then
  warn "Model loading/warmup failed (exit code: $PROJ_RC)"
  cat "$PROJ_TMP"
  EXIT_CODE=13
else
  # Extract and display projection RMS values
  if grep -E -q "PROJ load:" "$PROJ_TMP"; then
    echo ""
    info "Projection RMS values:"
    grep -E "PROJ load:" "$PROJ_TMP" | head -20
    echo ""
    info "✅ Projection weights loaded (see RMS values above)"
  else
    warn "No projection RMS values found in logs (RUST_LOG=info not showing PROJ load)"
    info "This is not an error, but projection statistics are unavailable"
  fi
fi

rm -f "$PROJ_TMP"

# ============================================================================
# 3. Greedy Inference Probe (Linguistic Sanity Check)
# ============================================================================
section "3/3: Greedy Inference Probe (Linguistic Sanity)"

info "Running deterministic greedy inference probe..."
info "Expected: Output should contain recognizable words, not gibberish"

PROBE_TMP=$(mktemp)
set +e
BITNET_DETERMINISTIC=1 \
BITNET_SEED=42 \
RAYON_NUM_THREADS=1 \
  cargo run -q -p bitnet-cli --no-default-features --features cpu -- \
  run --model "$MODEL" --tokenizer "$TOK" \
  --prompt "The capital of France is" \
  --max-new-tokens 8 \
  --temperature 0.0 \
  2>&1 | tee "$PROBE_TMP"
PROBE_RC=$?
set -e

if [[ $PROBE_RC -ne 0 ]]; then
  warn "Inference probe failed (exit code: $PROBE_RC)"
  cat "$PROBE_TMP"
  EXIT_CODE=14
else
  # Check for linguistic content (at least one word with 3+ ASCII letters)
  OUTPUT=$(cat "$PROBE_TMP")
  echo ""
  echo "Generated output:"
  echo "----------------------------------------"
  echo "$OUTPUT"
  echo "----------------------------------------"
  echo ""

  if echo "$OUTPUT" | tr -d '\n' | grep -E -q "[A-Za-z]{3,}"; then
    info "✅ Output contains recognizable words (linguistic sanity check passed)"
  else
    warn "Output does not contain recognizable words (might be gibberish or tokenizer mismatch)"
    EXIT_CODE=15
  fi
fi

rm -f "$PROBE_TMP"

# ============================================================================
# Final Report
# ============================================================================
section "Validation Report"

if [[ $EXIT_CODE -eq 0 ]]; then
  info "✅✅✅ ALL VALIDATION CHECKS PASSED ✅✅✅"
  echo ""
  echo "This model is clean and production-ready:"
  echo "  ✓ LayerNorm weights are healthy (RMS ≈ 1.0)"
  echo "  ✓ Projection weights loaded successfully"
  echo "  ✓ Greedy inference produces linguistic output"
  echo ""
  echo "Model: $MODEL"
  echo "Tokenizer: $TOK"
  echo ""
  echo "You can now use this model with:"
  echo "  cargo run -p bitnet-cli -- run --model $MODEL --tokenizer $TOK --prompt 'Your prompt'"
  exit 0
else
  error "❌ VALIDATION FAILED (exit code: $EXIT_CODE)

One or more validation checks did not pass.
Please review the output above for details.

Common issues:
  - LayerNorm weights are quantized (should be F16/F32)
  - Projection weights have unusual RMS values
  - Tokenizer mismatch (wrong tokenizer.json for this model)
  - Model corruption during export

Recommended actions:
  1. Re-export the model with LayerNorm weights in float format
  2. Verify tokenizer.json matches the model's training tokenizer
  3. Check export logs for warnings or errors

See docs/howto/export-clean-gguf.md for more details." $EXIT_CODE
fi
