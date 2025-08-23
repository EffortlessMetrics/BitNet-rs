#!/usr/bin/env bash
set -euo pipefail

# ---- Config (override via env) ----------------------------------------------
# Auto-detect bitnet binary location
: "${BITNET_BIN:=$(command -v bitnet || echo "target/release/bitnet")}"
if [[ ! -x "$BITNET_BIN" ]]; then
    echo "Error: bitnet binary not found. Please build with:"
    echo "  cargo build -p bitnet-cli --release --no-default-features --features cpu"
    echo "Or install with:"
    echo "  cargo install --path crates/bitnet-cli --no-default-features --features cpu"
    exit 1
fi
: "${MODEL_PATH:?Set MODEL_PATH=path/to/model.gguf}"
: "${TOKENIZER:?Set TOKENIZER=path/to/tokenizer.json}"
: "${HF_MODEL_ID:?Set HF_MODEL_ID=compatible HF model id}"

# Parity thresholds (PR lane defaults)
: "${PROP_EXAMPLES:=12}"
: "${TAU_STEPS:=24}"
: "${LOGIT_TOPK:=10}"
: "${TAU_MIN:=0.60}"
: "${DELTA_NLL_MAX:=1e-2}"          # 2e-2 for quant vs FP32
: "${PPL_FILE:=crossval/data/ppl_smoke.txt}"

# Determinism knobs
export BITNET_DETERMINISTIC=1 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
ARTIFACT="${PARITY_ARTIFACT:-artifacts/parity_failures.jsonl}"

# ---- Build ------------------------------------------------------------------
echo "==> Building CLI (release, cpu)"
cargo build -p bitnet-cli --release --no-default-features --features cpu 1>/dev/null

# ---- Unit tests (fast sanity) -----------------------------------------------
echo "==> Running Rust unit tests (workspace)"
cargo test --workspace --exclude bitnet-py --no-default-features --features cpu -q

# ---- Tokenizer parity (smoke) -----------------------------------------------
echo "==> Tokenizer parity (smoke)"
BITNET_BIN="$BITNET_BIN" MODEL_PATH="$MODEL_PATH" TOKENIZER="$TOKENIZER" HF_MODEL_ID="$HF_MODEL_ID" \
  scripts/test-tokenizer-parity.py --smoke

# ---- Greedy argmax invariant (BitNet-only) ----------------------------------
echo "==> Greedy argmax invariant"
TMP_JSON="$(mktemp)"
"$BITNET_BIN" run --model "$MODEL_PATH" --tokenizer "$TOKENIZER" \
  --prompt "Greedy invariant smoke." --max-new-tokens 16 \
  --greedy --deterministic --threads 1 \
  --dump-logit-steps 8 --logits-topk 10 --json-out "$TMP_JSON" >/dev/null
python3 scripts/check_greedy_argmax.py "$TMP_JSON" || { echo "Greedy invariant failed"; exit 7; }
rm -f "$TMP_JSON"

# ---- Logit parity (teacher-forced shared path; τ-b) -------------------------
echo "==> Logit parity (TF shared path; median τ-b ≥ $TAU_MIN)"
BITNET_BIN="$BITNET_BIN" MODEL_PATH="$MODEL_PATH" TOKENIZER="$TOKENIZER" HF_MODEL_ID="$HF_MODEL_ID" \
  PROP_EXAMPLES="$PROP_EXAMPLES" TAU_STEPS="$TAU_STEPS" LOGIT_TOPK="$LOGIT_TOPK" TAU_MIN="$TAU_MIN" \
  PARITY_ARTIFACT="$ARTIFACT" \
  scripts/logit-parity.sh

# ---- NLL parity (teacher-forcing; token-weighted) ---------------------------
echo "==> NLL parity (|Δ mean_nll| ≤ $DELTA_NLL_MAX)"
BITNET_BIN="$BITNET_BIN" MODEL_PATH="$MODEL_PATH" TOKENIZER="$TOKENIZER" HF_MODEL_ID="$HF_MODEL_ID" \
  PPL_FILE="$PPL_FILE" DELTA_NLL_MAX="$DELTA_NLL_MAX" PARITY_ARTIFACT="$ARTIFACT" \
  scripts/nll-parity.sh

# ---- Optional: decode throughput bench (summary) ----------------------------
if [[ "${RUN_BENCH:-0}" == "1" ]]; then
  echo "==> Running decode throughput bench (optional)"
  scripts/bench-decode.sh || true
fi

echo "==> ALL GREEN ✅"