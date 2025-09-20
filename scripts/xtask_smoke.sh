#!/usr/bin/env bash
set -euo pipefail

# Config
MODEL="${MODEL:-models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf}"
TOKENIZER="${TOKENIZER:-}"   # optional; use --allow-mock if missing
JSON=/tmp/xtask_smoke.json

echo "== verify: JSON cleanliness =="
out="$(cargo run -p xtask -- verify --model "$MODEL" --format json 2>/dev/null)"
echo "$out" | jq -e . >/dev/null

echo "== verify: strict failure exit=15 on bad path =="
set +e
cargo run -p xtask -- verify --model /nope/bad.gguf --strict >/dev/null 2>&1
code=$?; set -e
[[ $code -eq 15 ]] || { echo "verify strict exit code=$code (expected 15)"; exit 1; }

echo "== infer: deterministic mock json =="
out="$(cargo run -p xtask -- infer --model "$MODEL" --prompt 'hi' \
  --max-new-tokens 4 --allow-mock --deterministic --format json 2>/dev/null)"
echo "$out" | jq -e '.config.temperature==0 and .config.seed==42' >/dev/null

echo "== benchmark: 0-token short-circuit + json =="
cargo run -p xtask -- benchmark --model "$MODEL" --allow-mock \
  --tokens 0 --json "$JSON" >/dev/null 2>&1
jq -e '.success==true and .timing.total_ms==0' "$JSON" >/dev/null

echo "== benchmark: one-liner present even with json =="
cargo run -p xtask -- benchmark --model "$MODEL" --allow-mock \
  --tokens 8 --warmup-tokens 2 --no-output --json "$JSON" 2>/dev/null | \
  grep -E 'tokens in .*s .* tok/s ' >/dev/null

jq -e '.version and .performance.tokens_per_sec >= 0' "$JSON" >/dev/null
echo "✅ smoke OK"