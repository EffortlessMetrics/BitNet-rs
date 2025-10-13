#!/usr/bin/env bash
# Export a clean (no-LN-quant) GGUF in F16 from SafeTensors or HF checkpoint.
# Usage: scripts/export_clean_gguf.sh <model_dir> <tokenizer.json> <out_dir>
#
# This script ensures LayerNorm weights remain in float format (F16) to avoid
# quantization-induced corruption. See docs/howto/export-clean-gguf.md for details.

set -euo pipefail

# Color output for better visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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

MODEL_DIR="${1:?Usage: $0 <model_dir> <tokenizer.json> <out_dir>}"
TOKENIZER="${2:?Usage: $0 <model_dir> <tokenizer.json> <out_dir>}"
OUTDIR="${3:?Usage: $0 <model_dir> <tokenizer.json> <out_dir>}"

# Validate inputs
[[ -d "$MODEL_DIR" ]] || error "Model directory not found: $MODEL_DIR"
[[ -f "$TOKENIZER" ]] || error "Tokenizer not found: $TOKENIZER"

mkdir -p "$OUTDIR"

# Determine the converter to use
# Priority: 1) explicit CONVERTER env, 2) Rust st2gguf, 3) Python converter, 4) llama.cpp
USE_RUST_CONVERTER=false
if [[ -n "${CONVERTER:-}" ]]; then
  if [[ "$CONVERTER" == "rust" ]] || [[ "$CONVERTER" == "st2gguf" ]]; then
    USE_RUST_CONVERTER=true
    info "Using Rust st2gguf converter (explicit)"
  elif [[ -f "$CONVERTER" ]]; then
    info "Using explicit converter: $CONVERTER"
  else
    error "Explicit converter not found: $CONVERTER"
  fi
elif [[ -f "target/release/st2gguf" ]]; then
  USE_RUST_CONVERTER=true
  CONVERTER="target/release/st2gguf"
  info "Using Rust st2gguf converter (preferred)"
elif command -v st2gguf >/dev/null 2>&1; then
  USE_RUST_CONVERTER=true
  CONVERTER="st2gguf"
  info "Using Rust st2gguf converter (installed)"
elif [[ -f "scripts/convert_safetensors_to_gguf.py" ]]; then
  CONVERTER="scripts/convert_safetensors_to_gguf.py"
  info "Using Python SafeTensors converter (fallback)"
else
  # Fall back to llama.cpp converter if available
  CONVERTER="${CONVERTER:-third_party/llama.cpp/convert-hf-to-gguf.py}"
  if [[ ! -f "$CONVERTER" ]]; then
    error "No converter found. Please:
  1) Build st2gguf: cargo build --release -p bitnet-st2gguf, or
  2) Set CONVERTER=/path/to/convert script, or
  3) Ensure scripts/convert_safetensors_to_gguf.py exists, or
  4) Vendor llama.cpp in third_party/" 2
  fi
fi

# Check for Python (only needed for Python converters)
if [[ "$USE_RUST_CONVERTER" == "false" ]]; then
  command -v python3 >/dev/null 2>&1 || error "python3 not found in PATH"
fi

# Detect model format
ST_FILES=($(find "$MODEL_DIR" -maxdepth 1 -name "*.safetensors" 2>/dev/null || true))
if [[ ${#ST_FILES[@]} -gt 0 ]]; then
  info "Found ${#ST_FILES[@]} SafeTensors file(s)"
  MODEL_INPUT="${ST_FILES[0]}"

  # Use either Rust or Python converter
  info "Converting SafeTensors to GGUF (F16 output, LayerNorm preserved)..."

  # Check if config.json exists
  CONFIG_ARG=""
  if [[ -f "$MODEL_DIR/config.json" ]]; then
    CONFIG_ARG="--config $MODEL_DIR/config.json"
  fi

  if [[ "$USE_RUST_CONVERTER" == "true" ]]; then
    # Rust st2gguf converter (preferred)
    # Automatically detects and preserves LayerNorm tensors as F16
    "$CONVERTER" \
      --input "$MODEL_INPUT" \
      --output "$OUTDIR/clean-f16.gguf" \
      $CONFIG_ARG \
      ${STRICT:+--strict}
  else
    # Python converter (fallback)
    python3 "$CONVERTER" \
      "$MODEL_INPUT" \
      "$OUTDIR/clean-f16.gguf" \
      --tokenizer "$TOKENIZER" \
      $CONFIG_ARG
  fi

elif [[ -f "$MODEL_DIR/pytorch_model.bin" ]] || [[ -f "$MODEL_DIR/model.safetensors" ]]; then
  info "Found HF checkpoint format"

  if [[ "$USE_RUST_CONVERTER" == "true" ]]; then
    # Rust st2gguf converter supports both single files and directories
    "$CONVERTER" \
      --input "$MODEL_DIR" \
      --output "$OUTDIR/clean-f16.gguf" \
      ${CONFIG_ARG:+$CONFIG_ARG} \
      ${STRICT:+--strict}
  else
    # Python/llama.cpp converter
    python3 "$CONVERTER" \
      --model "$MODEL_DIR" \
      --tokenizer "$TOKENIZER" \
      --outtype f16 \
      --outfile "$OUTDIR/clean-f16.gguf"
  fi
else
  error "No recognized model files found in $MODEL_DIR
Expected: *.safetensors, pytorch_model.bin, or model.safetensors"
fi

# Verify the output exists
[[ -f "$OUTDIR/clean-f16.gguf" ]] || error "Conversion failed - output file not created"

# Generate fingerprint for traceability
info "Computing fingerprint..."
if command -v sha256sum >/dev/null 2>&1; then
  FP="$(sha256sum "$OUTDIR/clean-f16.gguf" | awk '{print "sha256-"$1}')"
elif command -v shasum >/dev/null 2>&1; then
  FP="$(shasum -a 256 "$OUTDIR/clean-f16.gguf" | awk '{print "sha256-"$1}')"
else
  error "sha256sum/shasum not found - cannot generate fingerprint" 3
fi

printf "%s\n" "$FP" > "$OUTDIR/clean-f16.fingerprint"

# Record export metadata
CONVERTER_TYPE="python"
if [[ "$USE_RUST_CONVERTER" == "true" ]]; then
  CONVERTER_TYPE="rust_st2gguf"
fi

cat > "$OUTDIR/clean-f16.meta.json" <<EOF
{
  "source": "$MODEL_DIR",
  "tokenizer": "$TOKENIZER",
  "converter": "$CONVERTER",
  "converter_type": "$CONVERTER_TYPE",
  "output": "$OUTDIR/clean-f16.gguf",
  "fingerprint": "$FP",
  "export_date": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "format": "gguf_v3",
  "precision": "f16",
  "layernorm_format": "float_preserved",
  "layernorm_enforcement": "automatic"
}
EOF

info "✅ Export complete!"
echo "  Output: $OUTDIR/clean-f16.gguf"
echo "  Fingerprint: $FP"
echo "  Metadata: $OUTDIR/clean-f16.meta.json"
echo ""
echo "Next steps:"
echo "  1) Validate: scripts/validate_gguf.sh $OUTDIR/clean-f16.gguf $TOKENIZER"
echo "  2) (Optional) Quantize to I2_S: scripts/quantize_i2s_clean.sh $OUTDIR/clean-f16.gguf $OUTDIR"
