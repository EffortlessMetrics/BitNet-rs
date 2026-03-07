#!/usr/bin/env bash
# Quick validation runner for local development
set -euo pipefail

echo "==> BitNet-rs Quick Validation"
echo "==============================="

# Check for required tools
echo "→ Checking dependencies..."
command -v cargo >/dev/null || { echo "Error: cargo not found"; exit 1; }
command -v python3 >/dev/null || { echo "Error: python3 not found"; exit 1; }
command -v jq >/dev/null || { echo "Warning: jq not found (needed for perf tests)"; }
command -v bc >/dev/null || { echo "Warning: bc not found (needed for perf tests)"; }

# Build if needed
BITNET_BIN="$(command -v bitnet || echo "")"
if [[ -z "$BITNET_BIN" ]] || [[ "${1:-}" == "--rebuild" ]]; then
    echo "→ Building BitNet CLI..."
    cargo build -p bitnet-cli --release --no-default-features --features cpu
    BITNET_BIN="target/release/bitnet"
fi

# Check for model
if [[ -z "${MODEL_PATH:-}" ]]; then
    echo ""
    echo "No MODEL_PATH set. To run full validation:"
    echo "  MODEL_PATH=path/to/model.gguf \\"
    echo "  TOKENIZER=path/to/tokenizer.json \\"
    echo "  HF_MODEL_ID=compatible-hf-id \\"
    echo "  $0"
    echo ""
    echo "→ Running unit tests only..."
    cargo test --workspace --no-default-features --features cpu --lib
else
    echo "→ Running full validation suite..."
    echo "  Model: $MODEL_PATH"
    echo "  Tokenizer: ${TOKENIZER:-<will use embedded>}"
    echo "  HF Model: ${HF_MODEL_ID:-<not set>}"

    # Create artifacts directory
    mkdir -p artifacts

    # Run validation with reasonable defaults for quick testing
    PROP_EXAMPLES="${PROP_EXAMPLES:-3}" \
    TAU_STEPS="${TAU_STEPS:-8}" \
    TAU_MIN="${TAU_MIN:-0.50}" \
    DELTA_NLL_MAX="${DELTA_NLL_MAX:-2e-2}" \
    scripts/validate_all.sh
fi

echo ""
echo "✓ Validation complete!"
