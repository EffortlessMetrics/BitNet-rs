#!/usr/bin/env bash
set -euo pipefail

need(){ command -v "$1" >/dev/null 2>&1 || { echo "missing: $1" >&2; exit 2; }; }
need jq
need diff

: "${BITNET_BIN:?set BITNET_BIN to your bitnet CLI}"
: "${LLAMA_BIN:?set LLAMA_BIN to your llama.cpp CLI}"
: "${MODEL_PATH:?set MODEL_PATH to a GGUF model}"
PROMPTS_YAML="${1:-crossval/prompts.yaml}"
MAX_NEW="${MAX_NEW_TOKENS:-64}"
THRESH="${THRESHOLD:-0.95}"

[[ -f "$PROMPTS_YAML" ]] || { echo "prompts file not found: $PROMPTS_YAML" >&2; exit 2; }

# Extract prompts (very light YAML parsing for lines starting with "- ")
mapfile -t PROMPTS < <(grep -E '^[[:space:]]*-\s' "$PROMPTS_YAML" | sed -E 's/^[[:space:]]*-\s//')

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
pass=0; total=${#PROMPTS[@]}; idx=0

echo "A/B token-id parity over $total prompts (threshold: $(awk -v t=$THRESH 'BEGIN{printf "%.1f%%", 100*t}'))"

for p in "${PROMPTS[@]}"; do
  idx=$((idx+1))
  rs_ids="$tmp/rs_${idx}.ids"
  cpp_ids="$tmp/cpp_${idx}.ids"
  rs_json="$tmp/rs_${idx}.json"
  cpp_json="$tmp/cpp_${idx}.json"

  # bitnet-rs run
  if [[ -n "${TOKENIZER_PATH:-}" ]]; then
    "$BITNET_BIN" run --model "$MODEL_PATH" --tokenizer "$TOKENIZER_PATH" \
      --prompt "$p" --bos --max-new-tokens "$MAX_NEW" --temperature 0 \
      --dump-token-ids "$rs_ids" --json-out "$rs_json" >/dev/null
  else
    "$BITNET_BIN" run --model "$MODEL_PATH" \
      --prompt "$p" --bos --max-new-tokens "$MAX_NEW" --temperature 0 \
      --dump-token-ids "$rs_ids" --json-out "$rs_json" >/dev/null
  fi

  # llama.cpp run
  if [[ -n "${TOKENIZER_PATH:-}" ]]; then
    "$LLAMA_BIN" --model "$MODEL_PATH" --tokenizer "$TOKENIZER_PATH" \
      --prompt "$p" --bos --max-new-tokens "$MAX_NEW" --temperature 0 \
      --dump-token-ids "$cpp_ids" --json-out "$cpp_json" >/dev/null
  else
    "$LLAMA_BIN" --model "$MODEL_PATH" \
      --prompt "$p" --bos --max-new-tokens "$MAX_NEW" --temperature 0 \
      --dump-token-ids "$cpp_ids" --json-out "$cpp_json" >/dev/null
  fi

  # Compare token id sequences (tolerate spacing/newlines)
  # Use awk instead of grep so pipefail doesn't trip when there are no matches.
  mapfile -t A < <(tr -s '[:space:]' ' ' < "$rs_ids"  | tr ' ' '\n' | awk '/^[0-9]+$/ {print}')
  mapfile -t B < <(tr -s '[:space:]' ' ' < "$cpp_ids" | tr ' ' '\n' | awk '/^[0-9]+$/ {print}')

  # Exact comparison
  exact=1
  min_len=${#A[@]}; [[ ${#B[@]} -lt $min_len ]] && min_len=${#B[@]}
  first_diff=-1
  for ((i=0;i<min_len;i++)); do
    if [[ "${A[$i]}" != "${B[$i]}" ]]; then first_diff=$i; exact=0; break; fi
  done
  # If all shared positions match, lengths must also match for "exact"
  if [[ $exact -eq 1 && ${#A[@]} -ne ${#B[@]} ]]; then
    first_diff=$min_len; exact=0
  fi

  if [[ $exact -eq 1 ]]; then
    echo "  [$idx/$total] ✓ EXACT"
    pass=$((pass+1))
  else
    echo "  [$idx/$total] ✗ DIFF at index $first_diff"
  fi
done

rate=$(awk -v p="$pass" -v t="$total" 'BEGIN{if(t==0)print 0; else print p/t}')
if awk -v r="$rate" -v th="$THRESH" 'BEGIN{exit !(r>=th)}'; then
  printf "✓ Token-ID parity: %.1f%% (>= %.1f%%)\n" "$(awk -v r="$rate" 'BEGIN{print r*100}')" "$(awk -v th="$THRESH" 'BEGIN{print th*100}')"
  exit 0
else
  printf "✗ Token-ID parity: %.1f%% (< %.1f%%)\n" "$(awk -v r="$rate" 'BEGIN{print r*100}')" "$(awk -v th="$THRESH" 'BEGIN{print th*100}')"
  exit 1
fi
