#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf}"
TOKENIZER="${2:-models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json}"

echo "=== Quantization Dispatch Probe ==="
echo "Model: $MODEL"
echo ""

# Build release
cargo build --release --no-default-features --features cpu,full-cli

# Run with quant tracing
BITNET_TRACE_QUANT=1 RUST_LOG=warn \
  target/release/bitnet run \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --prompt "test" \
  --max-tokens 1 \
  --greedy \
  2>&1 | grep "quant_dispatch" > docs/tdd/receipts/phase1_quant_probe.txt

echo "Results written to: docs/tdd/receipts/phase1_quant_probe.txt"
cat docs/tdd/receipts/phase1_quant_probe.txt
