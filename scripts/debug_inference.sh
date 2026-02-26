#!/usr/bin/env bash
# Deterministic inference validation script with comprehensive debug logging
#
# This script runs BitNet-rs inference with all diagnostic flags enabled
# to validate the fixes for attention scaling, RMSNorm, GQA, and tied embeddings.
#
# Usage:
#   ./scripts/debug_inference.sh [model_path] [tokenizer_path] [prompt]
#
# Example:
#   ./scripts/debug_inference.sh \
#     models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
#     models/llama3-tokenizer/tokenizer.json \
#     "Answer in one short sentence: Why is the sky blue?"

set -euo pipefail

# Default values
MODEL="${1:-models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf}"
TOKENIZER="${2:-models/llama3-tokenizer/tokenizer.json}"
PROMPT="${3:-Answer in one short sentence: Why is the sky blue?}"
MAX_TOKENS="${4:-32}"

echo "============================================"
echo "BitNet-rs Deterministic Debug Inference"
echo "============================================"
echo "Model: $MODEL"
echo "Tokenizer: $TOKENIZER"
echo "Prompt: $PROMPT"
echo "Max tokens: $MAX_TOKENS"
echo "============================================"
echo ""

# Enable all diagnostic flags
export BITNET_STRICT_MODE=1
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1

# Attention diagnostics
export BITNET_DEBUG_ATTN_SCALE=1  # Log scale computation, scores range, max-subtraction
export DEBUG_ATTN=1               # Log tensor stats for Q/K/V/scores/weights

# RMSNorm diagnostics
export BITNET_DEBUG_RMSNORM=1     # Log mean(x^2), rms before/after norm

# GQA diagnostics
export BITNET_DEBUG_GQA=1         # Log Q/K/V shapes and means

# Tied embeddings diagnostics
export BITNET_DEBUG_LOGITS=1      # Log tied logits sanity check

# MLP diagnostics
export BITNET_DEBUG_MLP=1         # Log MLP gate/up/down norms

# ROPE diagnostics
export BITNET_DEBUG_ROPE=1        # Log ROPE application details

echo "=== Diagnostic Flags Enabled ==="
echo "BITNET_DEBUG_ATTN_SCALE: ${BITNET_DEBUG_ATTN_SCALE}"
echo "DEBUG_ATTN: ${DEBUG_ATTN}"
echo "BITNET_DEBUG_RMSNORM: ${BITNET_DEBUG_RMSNORM}"
echo "BITNET_DEBUG_GQA: ${BITNET_DEBUG_GQA}"
echo "BITNET_DEBUG_LOGITS: ${BITNET_DEBUG_LOGITS}"
echo "BITNET_DEBUG_MLP: ${BITNET_DEBUG_MLP}"
echo "BITNET_DEBUG_ROPE: ${BITNET_DEBUG_ROPE}"
echo "BITNET_DETERMINISTIC: ${BITNET_DETERMINISTIC}"
echo "BITNET_SEED: ${BITNET_SEED}"
echo "RAYON_NUM_THREADS: ${RAYON_NUM_THREADS}"
echo "================================"
echo ""

# Run inference with full debug output
echo "=== Starting Inference ==="
cargo run --release -p bitnet-cli --no-default-features --features cpu -- run \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --prompt "$PROMPT" \
  --max-new-tokens "$MAX_TOKENS" \
  --temperature 0.0

echo ""
echo "=== Inference Complete ==="
echo ""
echo "To capture full diagnostic output, run:"
echo "  RUST_LOG=debug ./scripts/debug_inference.sh 2>&1 | tee inference_debug.log"
