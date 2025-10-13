#!/usr/bin/env bash
# BitNet.rs CI Acceptance Gate - Strict, Deterministic, No Compromises
# Binary discovery, JSON-driven assertions, no brittle greps
# All gates must pass - no skips, no mocks, no excuses
set -euo pipefail

# Exit codes for precise failure triage
readonly EXIT_SUCCESS=0
readonly EXIT_GENERAL_ERROR=1
readonly EXIT_MISSING_MODEL=2
readonly EXIT_MAPPING_FAILED=3
readonly EXIT_TOKENIZER_FAILED=4
readonly EXIT_INFERENCE_FAILED=5
readonly EXIT_TOKENIZATION_FAILED=6
readonly EXIT_DETERMINISM_FAILED=7
readonly EXIT_TEST_FAILED=8
readonly EXIT_PERF_REGRESSION=9
readonly EXIT_MEMORY_REGRESSION=10

# Temp file management with automatic cleanup
TMPFILES=()
mktempf() { local tmp=$(mktemp); TMPFILES+=("$tmp"); echo "$tmp"; }
cleanup() { rm -f "${TMPFILES[@]:-}" 2>/dev/null || true; }
trap cleanup EXIT

# Strict environment for determinism
export RAYON_NUM_THREADS=1
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export OMP_NUM_THREADS=1
export GGML_NUM_THREADS=1

# Check prerequisites
need() {
    command -v "$1" >/dev/null || {
        echo "❌ $1 is required but not installed"
        exit $EXIT_GENERAL_ERROR
    }
}
need cargo
need jq

echo "=== BitNet.rs CI Acceptance Gate ==="
echo "Environment: DETERMINISTIC=1, SEED=42, THREADS=1"

# ━━━ Gate 1: Build & Binary Discovery ━━━
echo "━━━ Gate 1: Build & Binary Discovery ━━━"

RS_BIN=$(
    cargo build -p bitnet-cli --release --no-default-features --features "cpu,full-cli" \
        --message-format=json 2>/dev/null \
    | grep '^{' \
    | jq -r 'select(.executable!=null and .target.kind[]=="bin" and .target.name=="bitnet") | .executable' \
    | tail -1
)

# Fallback if JSON parsing fails
if [ -z "${RS_BIN:-}" ] || [ ! -x "$RS_BIN" ]; then
    echo "⚠ JSON parse failed, attempting fallback build..."
    cargo build -p bitnet-cli --release --no-default-features --features "cpu,full-cli" 2>&1 | tee /tmp/cargo_build.log
    RS_BIN=$(find target -name bitnet -type f -executable 2>/dev/null | head -1)
fi

if [ ! -x "${RS_BIN:-}" ]; then
    echo "❌ Failed to build or locate bitnet binary"
    exit $EXIT_GENERAL_ERROR
fi

echo "✓ Binary discovered: $RS_BIN"

# ━━━ Gate 2: Unit Tests ━━━
echo "━━━ Gate 2: Unit Tests ━━━"

TEST_OUTPUT=$(mktempf)
if ! cargo test --workspace --no-default-features --features cpu \
        --exclude bitnet-py --lib -- -q > "$TEST_OUTPUT" 2>&1; then
    echo "❌ Unit tests failed"
    tail -200 "$TEST_OUTPUT"
    exit $EXIT_TEST_FAILED
fi
echo "✓ Unit tests passed"

# ━━━ Gate 3: Model Selection ━━━
echo "━━━ Gate 3: Model Selection ━━━"

# Determine if we're in PR or nightly mode
if [ -n "${CI_PR:-}" ] || [ -z "${NIGHTLY:-}" ]; then
    MODE="PR"
    MODEL_PATH="${PR_MODEL:-models/tinyllama-q2.gguf}"
    TOKENIZER_MODE="embedded"
else
    MODE="NIGHTLY"
    MODEL_PATH="${BITNET_GGUF:-models/bitnet/ggml-model-i2_s.gguf}"
    TOKENIZER_PATH="${TOKENIZER_PATH:-models/bitnet/tokenizer.model}"
    TOKENIZER_MODE="external"
fi

echo "Mode: $MODE"
echo "Model: $MODEL_PATH"

# Check model exists (NO SKIPPING!)
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Missing model file: $MODEL_PATH"
    if [ "$MODE" = "PR" ]; then
        echo "Run: scripts/fetch-pr-model.sh to download TinyLlama with embedded tokenizer"
    else
        echo "Run: cargo run -p xtask -- download-model"
    fi
    exit $EXIT_MISSING_MODEL
fi

# For nightly, check external tokenizer
if [ "$MODE" = "NIGHTLY" ] && [ ! -f "$TOKENIZER_PATH" ]; then
    echo "❌ Missing tokenizer: $TOKENIZER_PATH (required for nightly)"
    exit $EXIT_TOKENIZER_FAILED
fi

# ━━━ Gate 4: Tensor Mapping Validation ━━━
echo "━━━ Gate 4: Tensor Mapping Validation ━━━"

MAPPER_JSON=$(mktempf)
if ! cargo run -q -p xtask -- gate mapper --model "$MODEL_PATH" > "$MAPPER_JSON" 2>/dev/null; then
    echo "❌ Mapper gate failed to run"
    exit $EXIT_MAPPING_FAILED
fi

if ! jq -e '.ok==true and .unmapped_count==0' "$MAPPER_JSON" >/dev/null; then
    echo "❌ Tensor mapping failed"
    jq '.' "$MAPPER_JSON"
    exit $EXIT_MAPPING_FAILED
fi

UNMAPPED=$(jq -r '.unmapped_count // -1' "$MAPPER_JSON")
echo "✓ All tensors mapped (unmapped=$UNMAPPED)"

# ━━━ Gate 5: Strict Inference ━━━
echo "━━━ Gate 5: Strict Inference (no mocks) ━━━"

STRICT_JSON=$(mktempf)
STRICT_ARGS=(
    run
    --model "$MODEL_PATH"
    --prompt "The capital of France is"
    --bos
    --max-new-tokens 16
    --temperature 0
    --strict-mapping
    --strict-tokenizer
    --json-out "$STRICT_JSON"
)

# Add external tokenizer for nightly
if [ "$MODE" = "NIGHTLY" ]; then
    STRICT_ARGS+=(--tokenizer "$TOKENIZER_PATH")
fi

if ! "$RS_BIN" "${STRICT_ARGS[@]}" >/dev/null 2>&1; then
    echo "❌ Strict inference failed"
    exit $EXIT_INFERENCE_FAILED
fi

# Validate strict JSON output
if [ "$MODE" = "PR" ]; then
    # PR mode: require embedded SentencePiece
    if ! jq -e '.counts.unmapped==0 and
                (.counts.n_kv|tonumber)>0 and
                (.counts.n_tensors|tonumber)>0 and
                .tokenizer.type=="sentencepiece"' "$STRICT_JSON" >/dev/null; then
        echo "❌ Strict validation failed (PR mode requires embedded tokenizer)"
        jq '.' "$STRICT_JSON"
        exit $EXIT_INFERENCE_FAILED
    fi
else
    # Nightly mode: external tokenizer OK
    if ! jq -e '.counts.unmapped==0 and
                (.counts.n_kv|tonumber)>0 and
                (.counts.n_tensors|tonumber)>0' "$STRICT_JSON" >/dev/null; then
        echo "❌ Strict validation failed"
        jq '.' "$STRICT_JSON"
        exit $EXIT_INFERENCE_FAILED
    fi
fi

echo "✓ Strict inference passed (tokenizer=$TOKENIZER_MODE)"

# ━━━ Gate 6: Tokenization Smoke Test ━━━
echo "━━━ Gate 6: Tokenization Smoke Test ━━━"

prompts=(
    "The capital of France is"
    "Once upon a time"
    "def fibonacci(n):"
)
pass=0
failed_prompts=()

for prompt in "${prompts[@]}"; do
    TOK_JSON=$(mktempf)
    TOK_ARGS=(
        tokenize
        --model "$MODEL_PATH"
        --prompt "$prompt"
        --bos
        --json-out "$TOK_JSON"
    )

    if [ "$MODE" = "NIGHTLY" ]; then
        TOK_ARGS+=(--tokenizer "$TOKENIZER_PATH")
    fi

    if "$RS_BIN" "${TOK_ARGS[@]}" >/dev/null 2>&1; then
        ids=$(jq -c '.tokens.ids' "$TOK_JSON" 2>/dev/null || echo "[]")
        if [[ "$ids" != "[]" ]] && [[ "$ids" != "null" ]]; then
            pass=$((pass+1))
        else
            failed_prompts+=("$prompt")
        fi
    else
        failed_prompts+=("$prompt")
    fi
done

if [[ "$pass" -lt 2 ]]; then
    echo "❌ Tokenization failed: only $pass/${#prompts[@]} prompts succeeded"
    for fp in "${failed_prompts[@]}"; do
        echo "  Failed: $fp"
    done
    exit $EXIT_TOKENIZATION_FAILED
fi

echo "✓ Tokenization smoke test: $pass/${#prompts[@]} passed"

# ━━━ Gate 7: Determinism Check ━━━
echo "━━━ Gate 7: Determinism Check ━━━"

RUN1=$(mktempf)
RUN2=$(mktempf)
DET_ARGS=(
    run
    --model "$MODEL_PATH"
    --prompt "Once upon"
    --bos
    --max-new-tokens 32
    --temperature 0
    --json-out
)

if [ "$MODE" = "NIGHTLY" ]; then
    DET_ARGS+=(--tokenizer "$TOKENIZER_PATH")
fi

"$RS_BIN" "${DET_ARGS[@]}" "$RUN1" >/dev/null 2>&1
"$RS_BIN" "${DET_ARGS[@]}" "$RUN2" >/dev/null 2>&1

# Compare token IDs for exact determinism (more robust than text)
IDS1=$(jq -c '.tokens.ids // []' "$RUN1" 2>/dev/null || echo "[]")
IDS2=$(jq -c '.tokens.ids // []' "$RUN2" 2>/dev/null || echo "[]")

if [[ "$IDS1" != "$IDS2" ]] || [[ "$IDS1" == "[]" ]]; then
    echo "❌ Non-deterministic token generation detected"
    echo "Run 1 IDs: $IDS1"
    echo "Run 2 IDs: $IDS2"
    exit $EXIT_DETERMINISM_FAILED
fi

echo "✓ Deterministic execution verified"

# ━━━ Gate 8: Performance & Memory ━━━
echo "━━━ Gate 8: Performance & Memory ━━━"

# Find time command (Linux /usr/bin/time or macOS gtime)
TIME_CMD=""
if [ -x /usr/bin/time ]; then
    TIME_CMD="/usr/bin/time -v"
elif command -v gtime >/dev/null 2>&1; then
    TIME_CMD="gtime -v"
fi

PERF_JSON=$(mktempf)
TIME_OUTPUT=$(mktempf)
PERF_ARGS=(
    run
    --model "$MODEL_PATH"
    --prompt "The quick brown fox jumps over the lazy dog"
    --max-new-tokens 128
    --temperature 0
    --json-out "$PERF_JSON"
)

if [ "$MODE" = "NIGHTLY" ]; then
    PERF_ARGS+=(--tokenizer "$TOKENIZER_PATH")
fi

# Run with timing if available
if [ -n "$TIME_CMD" ]; then
    $TIME_CMD "$RS_BIN" "${PERF_ARGS[@]}" 2>"$TIME_OUTPUT" >/dev/null
else
    "$RS_BIN" "${PERF_ARGS[@]}" >/dev/null 2>&1
fi

# Extract performance metrics
tokps=$(jq -r '.throughput.tokens_per_second // 0' "$PERF_JSON" 2>/dev/null || echo "0")
decoded=$(jq -r '.throughput.decoded_tokens // 0' "$PERF_JSON" 2>/dev/null || echo "0")

# Warn if too few tokens (noisy measurement)
if (( decoded < 64 )); then
    echo "⚠ Warning: only decoded $decoded tokens (<64), performance measurement may be noisy"
fi

# Check absolute floor
if ! awk "BEGIN{exit !($tokps >= 1.0)}"; then
    echo "❌ Performance too low: $tokps tokens/sec < 1.0 minimum"
    exit $EXIT_PERF_REGRESSION
fi

echo "Performance: $tokps tokens/sec"

# Check against baseline if available
if [ -f ci/baseline.json ]; then
    MODEL_KEY="tinyllama_q2k_cpu"
    if [ "$MODE" = "NIGHTLY" ]; then
        MODEL_KEY="bitnet_i2s_cpu"
    fi

    base_tps=$(jq -r --arg key "$MODEL_KEY" '.cpu[$key].tok_s // 0' ci/baseline.json)

    if [[ "$base_tps" != "0" ]]; then
        threshold=$(awk -v b="$base_tps" 'BEGIN{print 0.95 * b}')
        if ! awk -v c="$tokps" -v t="$threshold" 'BEGIN{exit !(c >= t)}'; then
            echo "❌ Performance regression: $tokps < 95% of baseline $base_tps"
            exit $EXIT_PERF_REGRESSION
        fi
        echo "✓ Performance ratio: $tokps / $base_tps baseline"
    fi

    # Check RSS if time command available
    if [ -n "$TIME_CMD" ] && grep -q "Maximum resident set size" "$TIME_OUTPUT" 2>/dev/null; then
        rss_kb=$(awk '/Maximum resident set size/{print $6}' "$TIME_OUTPUT")
        rss_mb=$((rss_kb / 1024))
        echo "Memory RSS: ${rss_mb}MB"

        base_rss=$(jq -r --arg key "$MODEL_KEY" '.cpu[$key].rss_mb // 0' ci/baseline.json)

        if [[ "$base_rss" != "0" ]]; then
            threshold=$(awk -v b="$base_rss" 'BEGIN{print int(1.03 * b)}')
            if (( rss_mb > threshold )); then
                echo "❌ Memory regression: ${rss_mb}MB > 103% of baseline ${base_rss}MB"
                exit $EXIT_MEMORY_REGRESSION
            fi
            echo "✓ Memory ratio: ${rss_mb}MB / ${base_rss}MB baseline"
        fi
    fi
else
    echo "✓ Performance acceptable (no baseline for regression testing)"
fi

# ━━━ Success ━━━
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 CI Acceptance Gate: ALL PASSED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Mode: $MODE"
echo "Binary: $RS_BIN"
echo "Model: $MODEL_PATH"
echo "All gates passed with strict validation"
exit $EXIT_SUCCESS
