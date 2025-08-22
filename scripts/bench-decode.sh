#!/usr/bin/env bash
set -euo pipefail

# Configuration
MODEL="${MODEL:-models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf}"
TOKENIZER="${TOKENIZER:?Set TOKENIZER=path/to/tokenizer.json or .model}"
BIN="${BIN:-target/release/bitnet}"
FEATURES="${FEATURES:-cpu}"

# Build in release mode if needed
if [ ! -f "$BIN" ]; then
    echo "Building bitnet-cli in release mode..."
    cargo build -p bitnet-cli --release --no-default-features --features "$FEATURES"
fi

# Test prompts
PROMPTS=(
    "The quick brown fox jumps over the lazy dog."
    "In a shocking discovery, researchers found that"
    "Write a short function in Rust that"
    "Explain the concept of quantum computing to a five year old"
    "What are the main differences between TCP and UDP protocols?"
)

OUT="bench-results.jsonl"
: > "$OUT"

echo "Running benchmark with:"
echo "  Model: $MODEL"
echo "  Tokenizer: $TOKENIZER"
echo "  Features: $FEATURES"
echo ""

for i in "${!PROMPTS[@]}"; do
    p="${PROMPTS[$i]}"
    echo "[$((i+1))/${#PROMPTS[@]}] Processing prompt: ${p:0:40}..."
    
    "$BIN" inference \
        --model "$MODEL" \
        --tokenizer "$TOKENIZER" \
        --prompt "$p" \
        --max-tokens 128 \
        --temperature 0 \
        --format json \
        2>/dev/null \
    | jq -c '. + {prompt: "'"$p"'"}' >> "$OUT" || {
        echo "Warning: Failed to process prompt $((i+1))"
    }
done

echo ""
echo "=== Benchmark Summary ==="
if [ -s "$OUT" ]; then
    jq -s '
      def median(f): map(f) | sort | if length > 0 then .[length/2|floor] else 0 end;
      def mean(f): map(f) | if length > 0 then add/length else 0 end;
      {
        n_samples: length,
        decode_tps: {
          median: median(.throughput_tps.decode),
          mean: mean(.throughput_tps.decode),
          min: [.[].throughput_tps.decode] | min,
          max: [.[].throughput_tps.decode] | max
        },
        prefill_tps: {
          median: median(.throughput_tps.prefill),
          mean: mean(.throughput_tps.prefill)
        },
        e2e_tps: {
          median: median(.throughput_tps.e2e),
          mean: mean(.throughput_tps.e2e)
        },
        tokens_generated: {
          median: median(.counts.generated_tokens),
          total: [.[].counts.generated_tokens] | add
        },
        timing_ms: {
          tokenize_median: median(.timing_ms.tokenize),
          prefill_median: median(.timing_ms.prefill),
          decode_median: median(.timing_ms.decode),
          total_median: median(.timing_ms.total)
        }
      }
    ' "$OUT" | jq .
else
    echo "No results collected. Check for errors above."
fi

echo ""
echo "Raw results saved to: $OUT"