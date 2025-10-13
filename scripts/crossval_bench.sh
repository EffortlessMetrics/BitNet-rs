#!/usr/bin/env bash
set -euo pipefail

# --- config (override via env) ---
MODEL="${MODEL:-models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf}"
PROMPT="${PROMPT:-The sky is}"
TOKENS="${TOKENS:-64}"
SEED="${SEED:-42}"
LLAMA_BIN="${LLAMA_BIN:-$HOME/.cache/bitnet_cpp/build/bin/llama-cli}"
BITNET_BIN="${BITNET_BIN:-${CARGO_TARGET_DIR:-$HOME/.rust-build}/target/release/bitnet}"
CSV="${CSV:-crossval_results.csv}"
# Comma-separated list of thread counts to try for C++ (default: 1 and nproc)
THREADS="${THREADS:-}"
# If STRICT=1, fail the script when the Rust path is MOCK/ENGINE_DISABLED/FAIL
STRICT="${STRICT:-0}"
# Warm-up tokens (ignored in metrics) to stabilize timings
WARMUP_TOKENS="${WARMUP_TOKENS:-16}"

# ensure paths
mkdir -p "$(dirname "$CSV")"
if [[ ! -x "$LLAMA_BIN" ]]; then
  echo "error: llama-cli not found at $LLAMA_BIN (set LLAMA_BIN or build C++ backend)" >&2
  exit 1
fi

# write header once
if [[ ! -f "$CSV" ]]; then
  echo "impl,prompt,tokens,threads,tokens_per_sec,status" > "$CSV"
fi

run_cpp() {
  local threads="$1"
  # warm-up (ignored)
  "$LLAMA_BIN" -m "$MODEL" -p "$PROMPT" -n "$WARMUP_TOKENS" -s "$SEED" --temp 0 --mirostat 0 -t "$threads" >/dev/null 2>&1 || true
  echo "===> C++ (threads=$threads)"
  set +e
  OUT="$("$LLAMA_BIN" \
    -m "$MODEL" \
    -p "$PROMPT" \
    -n "$TOKENS" \
    -s "$SEED" \
    --temp 0 \
    --top-k 40 \
    --top-p 0.95 \
    --mirostat 0 \
    -t "$threads" 2>&1)"
  RC=$?
  set -e

  if [[ $RC -ne 0 ]]; then
    echo "$OUT" >&2
    echo "cpp,\"$PROMPT\",$TOKENS,$threads,0,FAIL" >> "$CSV"
    return
  fi

  # robust parse: float just before 'tokens per second'
  TPS="$(printf '%s\n' "$OUT" | grep -Eio '[0-9]+(\.[0-9]+)?[[:space:]]+tokens per second' | awk '{print $1}' | tail -1)"
  [[ -z "$TPS" ]] && TPS=0
  echo "cpp,\"$PROMPT\",$TOKENS,$threads,$TPS,OK" >> "$CSV"
}

run_rust() {
  # Build if needed
  if [[ ! -x "$BITNET_BIN" ]]; then
    echo "===> building bitnet (release)"
    cargo build --locked -p bitnet-cli --release --features cpu
  fi

  echo "===> Rust (threads=auto)"
  # Time wall-clock; the CLI does not print t/s today. We compute rough TPS.
  local start_ns end_ns dur_ms tps status
  start_ns=$(date +%s%N || true)
  set +e
  OUT="$("$BITNET_BIN" run \
      --model "$MODEL" \
      --prompt "$PROMPT" \
      --max-new-tokens "$TOKENS" \
      2>&1)"
  RC=$?
  set -e
  end_ns=$(date +%s%N || true)

  if echo "$OUT" | grep -qi 'built without `inference`'; then
    status="ENGINE_DISABLED"
    tps=0
  elif echo "$OUT" | grep -qi 'mock tensor'; then
    status="MOCK_PATH"
    tps=0
  elif [[ $RC -ne 0 ]]; then
    status="FAIL"
    tps=0
  else
    dur_ms=$(( (end_ns - start_ns) / 1000000 ))
    if [[ "$dur_ms" -gt 0 ]]; then
      tps=$(awk -v n="$TOKENS" -v ms="$dur_ms" 'BEGIN { printf("%.2f", (n*1000.0)/ms) }')
    else
      tps=0
    fi
    status="OK"
  fi

  echo "rust,\"$PROMPT\",$TOKENS,auto,$tps,$status" >> "$CSV"
  if [[ "$STRICT" = "1" && "$status" != "OK" ]]; then
    echo "strict mode: rust status = $status → failing" >&2
    exit 1
  fi
}

# --- run ---
echo "Model: $MODEL"
echo "Prompt: $PROMPT"
echo

# Decide thread matrix
if [[ -z "$THREADS" ]]; then
  THREADS="1"
  if command -v nproc >/dev/null 2>&1; then
    THREADS="$THREADS,$(nproc)"
  fi
fi
IFS=, read -r -a THREAD_ARR <<< "$THREADS"
for th in "${THREAD_ARR[@]}"; do
  run_cpp "$th"
done

# Rust path: harmless if engine not ready; marks status accordingly.
run_rust || true

echo
echo "Wrote results to: $CSV"
column -s, -t "$CSV" || cat "$CSV"
