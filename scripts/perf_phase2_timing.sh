#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf}"
TOKENIZER="${2:-models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json}"
RECEIPT="docs/baselines/perf/phase2_timing_i2s.md"

# Enable determinism for reproducible receipts
export BITNET_DETERMINISTIC=1
export RAYON_NUM_THREADS=1

echo "=== Timing Probe (1 token) ==="
echo "Model: $MODEL"
echo ""

# Build release with native ISA
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=thin" \
  cargo build --release --no-default-features --features cpu,full-cli

# Run 3 times, take median
echo "Running 3 iterations..."
mkdir -p docs/baselines/perf

for i in {1..3}; do
  echo "Iteration $i..."
  BITNET_TRACE_TIMING=1 RUST_LOG=warn \
    target/release/bitnet run \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --prompt "2+2=" \
    --max-tokens 1 \
    --greedy \
    2>&1 | grep "timing:" | tee -a "$RECEIPT.tmp"
done

echo ""
echo "=== Timing Summary ===" | tee "$RECEIPT"
echo "Model: $MODEL" | tee -a "$RECEIPT"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RECEIPT"
echo "" | tee -a "$RECEIPT"

# Host fingerprint (enhanced format per PR3 exploration plan)
echo "## Host Fingerprint" | tee -a "$RECEIPT"
echo "**Host:** $(uname -a)" | tee -a "$RECEIPT"
echo "**rustc:** $(rustc -V)" | tee -a "$RECEIPT"
echo "**cpu:** $(lscpu | sed -n '1,6p')" | tee -a "$RECEIPT"
echo "**Git commit:** $(git rev-parse --short HEAD)" | tee -a "$RECEIPT"
echo "**Determinism:** BITNET_DETERMINISTIC=1, RAYON_NUM_THREADS=1" | tee -a "$RECEIPT"
echo "" | tee -a "$RECEIPT"

echo "## Timing Results" | tee -a "$RECEIPT"
cat "$RECEIPT.tmp" | tee -a "$RECEIPT"
rm "$RECEIPT.tmp"

echo ""
echo "Receipt written to: $RECEIPT"
