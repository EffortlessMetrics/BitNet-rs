#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:-models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf}"
TOK="${2:-models/llama3-tokenizer/tokenizer.json}"

echo "== verify (human) =="
cargo run -p xtask -- verify --model "$MODEL" --tokenizer "$TOK" --format human

echo "== infer (auto template, 1 step, DEBUG_ATTN=1) =="
DEBUG_ATTN=1 cargo run -p xtask --features inference -- infer \
  --model "$MODEL" --tokenizer "$TOK" \
  --prompt "The capital of France is" --max-new-tokens 1 --deterministic --template auto

echo "== benchmark (prefill vs decode) =="
cargo run -p xtask --features inference -- benchmark \
  --model "$MODEL" --tokenizer "$TOK" \
  --prompt "Write two short lines." --tokens 64 --warmup-tokens 16 --no-output \
  --json /tmp/bench.json 2>/dev/null | tee /dev/stderr
echo "tokens_per_sec:"
jq '.performance.tokens_per_sec' /tmp/bench.json
echo "prefill/decode ms:"
jq '.timing.prefill_ms, .timing.decode_ms' /tmp/bench.json