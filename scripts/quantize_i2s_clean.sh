#!/usr/bin/env bash
# Quantize clean-f16.gguf to I2_S while EXCLUDING LayerNorm weights.
# Usage: scripts/quantize_i2s_clean.sh <clean-f16.gguf> <out_dir>
#
# IMPORTANT: This script requires a quantizer that supports pattern-based
# exclusion of LayerNorm weights. LayerNorm weights MUST remain in float
# format to avoid quantization-induced corruption.
#
# Required quantizer capabilities:
#   - I2_S quantization format
#   - Pattern-based include/exclude for tensor names
#   - Preserve metadata from source GGUF
#
# Status: STUB - Quantizer implementation pending
# See: docs/explanation/quantization-support.md for I2_S specification

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

error() {
  echo -e "${RED}ERROR: $1${NC}" >&2
  exit "${2:-1}"
}

info() {
  echo -e "${GREEN}INFO: $1${NC}"
}

warn() {
  echo -e "${YELLOW}WARN: $1${NC}"
}

IN="${1:?Usage: $0 <clean-f16.gguf> <out_dir>}"
OUTDIR="${2:?Usage: $0 <clean-f16.gguf> <out_dir>}"

# Validate inputs
[[ -f "$IN" ]] || error "Input model not found: $IN"
mkdir -p "$OUTDIR"

OUT="$OUTDIR/clean-i2s.gguf"

# ============================================================================
# Quantizer Detection
# ============================================================================

# Try to find a quantizer binary
# Priority: 1) QUANTIZER env var, 2) bitnet-quantize in target/release, 3) xtask quantize
QUANT=""
if [[ -n "${QUANTIZER:-}" ]] && [[ -x "$QUANTIZER" ]]; then
  QUANT="$QUANTIZER"
  info "Using explicit quantizer: $QUANT"
elif [[ -x "target/release/bitnet-quantize" ]]; then
  QUANT="target/release/bitnet-quantize"
  info "Using bitnet-quantize binary: $QUANT"
elif cargo run -p xtask -- help 2>/dev/null | grep -q "quantize"; then
  QUANT="cargo run -p xtask -- quantize"
  info "Using xtask quantize command"
else
  error "No quantizer found.

This script requires a quantizer binary with the following capabilities:
  - I2_S quantization format
  - Pattern-based tensor inclusion/exclusion
  - LayerNorm weight preservation (keep in F16/F32)

Options to resolve:
  1. Set QUANTIZER=/path/to/quantizer env variable
  2. Build bitnet-quantize: cargo build -p bitnet-quantize --release
  3. Implement quantize subcommand in xtask

For now, you can use the F16 model directly:
  cargo run -p bitnet-cli -- run --model $IN --tokenizer <tokenizer.json>

See docs/explanation/quantization-support.md for I2_S specification." 2
fi

# ============================================================================
# Quantization with LayerNorm Exclusion
# ============================================================================

warn "I2_S quantization is currently a STUB implementation."
warn "The quantizer must exclude LayerNorm weights from quantization."

info "Starting I2_S quantization..."
echo ""
echo "Source: $IN"
echo "Target: $OUT"
echo "Format: I2_S (2-bit signed)"
echo ""

# Tensor patterns to quantize (attention and FFN projections)
INCLUDE_PATTERNS=(
  "*.attn_q.weight"
  "*.attn_k.weight"
  "*.attn_v.weight"
  "*.attn_output.weight"
  "*.ffn_gate.weight"
  "*.ffn_up.weight"
  "*.ffn_down.weight"
)

# Tensor patterns to EXCLUDE (keep in float)
EXCLUDE_PATTERNS=(
  "*.attn_norm.weight"
  "*.ffn_norm.weight"
  "token_embd.weight"
  "output.weight"
  "*.norm.weight"
)

info "Quantization policy:"
echo "  Include: ${INCLUDE_PATTERNS[*]}"
echo "  Exclude (keep F16): ${EXCLUDE_PATTERNS[*]}"
echo ""

# Build quantizer command based on detected tool
# This is a TEMPLATE - actual CLI args depend on quantizer implementation

if [[ "$QUANT" == *"xtask"* ]]; then
  # xtask quantize API (example - adjust to actual implementation)
  warn "xtask quantize not yet implemented - this is a stub"

  # Example command structure:
  # $QUANT \
  #   --input "$IN" \
  #   --output "$OUT" \
  #   --format i2s \
  #   --threads "${RAYON_NUM_THREADS:-$(nproc)}" \
  #   $(printf -- '--include %s ' "${INCLUDE_PATTERNS[@]}") \
  #   $(printf -- '--exclude %s ' "${EXCLUDE_PATTERNS[@]}")

  error "xtask quantize subcommand not implemented. Please build a quantizer binary or use F16 model." 3

else
  # Binary quantizer API (example - adjust to actual implementation)
  warn "Assuming quantizer binary supports --include/--exclude flags"

  # Example command structure:
  set +e
  "$QUANT" \
    --input "$IN" \
    --output "$OUT" \
    --format I2_S \
    --threads "${RAYON_NUM_THREADS:-$(nproc)}" \
    $(printf -- '--include %s ' "${INCLUDE_PATTERNS[@]}") \
    $(printf -- '--exclude %s ' "${EXCLUDE_PATTERNS[@]}")
  QUANT_RC=$?
  set -e

  if [[ $QUANT_RC -ne 0 ]]; then
    error "Quantization failed (exit code: $QUANT_RC)
Check quantizer logs above for details." 4
  fi
fi

# ============================================================================
# Post-Quantization Validation
# ============================================================================

if [[ ! -f "$OUT" ]]; then
  error "Quantization completed but output file not found: $OUT" 5
fi

info "Computing fingerprint..."
if command -v sha256sum >/dev/null 2>&1; then
  FP="$(sha256sum "$OUT" | awk '{print "sha256-"$1}')"
elif command -v shasum >/dev/null 2>&1; then
  FP="$(shasum -a 256 "$OUT" | awk '{print "sha256-"$1}')"
else
  error "sha256sum/shasum not found" 6
fi

printf "%s\n" "$FP" > "$OUTDIR/clean-i2s.fingerprint"

# Record quantization metadata
cat > "$OUTDIR/clean-i2s.meta.json" <<EOF
{
  "source": "$IN",
  "output": "$OUT",
  "format": "i2s",
  "quantizer": "$QUANT",
  "fingerprint": "$FP",
  "quantize_date": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "layernorm_excluded": true,
  "include_patterns": $(printf '%s\n' "${INCLUDE_PATTERNS[@]}" | jq -R . | jq -s .),
  "exclude_patterns": $(printf '%s\n' "${EXCLUDE_PATTERNS[@]}" | jq -R . | jq -s .)
}
EOF

info "✅ Quantization complete!"
echo "  Output: $OUT"
echo "  Fingerprint: $FP"
echo "  Metadata: $OUTDIR/clean-i2s.meta.json"
echo ""
echo "Next steps:"
echo "  1) Validate: scripts/validate_gguf.sh $OUT <tokenizer.json>"
echo "  2) Compare with F16: Run inference on both and check outputs match"
echo ""
warn "Remember: I2_S quantization is lossy. Validate quality before production use."
