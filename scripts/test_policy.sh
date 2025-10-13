#!/usr/bin/env bash
# Test policy engagement

set -euo pipefail

export BITNET_CORRECTION_POLICY=./correction-policy.yml
# BITNET_FIX_LN_SCALE is deprecated - policy-driven corrections are the only supported path
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export RAYON_NUM_THREADS=1
export RUST_LOG=info,bitnet_models=debug

cargo run --release -p bitnet-cli --no-default-features --features cpu -- run \
  --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
  --tokenizer models/llama3-tokenizer/tokenizer.json \
  --prompt "Test" \
  --max-new-tokens 5 \
  --temperature 0.0
