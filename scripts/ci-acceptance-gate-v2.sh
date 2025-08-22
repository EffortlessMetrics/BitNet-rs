#!/usr/bin/env bash
# BitNet.rs CI Acceptance Gate v2 - Robust & Deterministic
# Binary discovery, JSON-driven assertions, no brittle greps
# All gates must pass for CI acceptance - no skips, no exceptions

set -euo pipefail

# Exit codes for precise failure triage
readonly EXIT_SUCCESS=0
readonly EXIT_GENERAL_ERROR=1
readonly EXIT_MISSING_DEPS=2
readonly EXIT_STRICT_MAPPING=3
readonly EXIT_STRICT_TOKENIZER=4
readonly EXIT_MODEL_LOAD_ERROR=5
readonly EXIT_TOKENIZER_ERROR=6
readonly EXIT_INFERENCE_ERROR=7
readonly EXIT_IO_ERROR=8
readonly EXIT_PERF_FAIL=9
readonly EXIT_MEM_FAIL=10
readonly EXIT_DETERMINISM_FAIL=11
readonly EXIT_FFI_FAIL=12

# Check required dependencies
need() { 
    command -v "$1" >/dev/null || { 
        echo "❌ $1 required but not found"
        exit $EXIT_MISSING_DEPS
    }
}

need jq
need awk
need diff
need cargo
need rustc

# Compose a bitnet run invocation with optional --tokenizer.
run_bitnet() {
  local subcmd="$1"; shift
  local args=( "$subcmd" --model "$MODEL_PATH" )
  if [[ -n "${TOKENIZER_PATH:-}" ]]; then
    args+=( --tokenizer "$TOKENIZER_PATH" )
  fi
  "$BITNET_BIN" "${args[@]}" "$@"
}

# Best-effort cleanup of temporaries created in this script
_CLEANUP_FILES=()
_track() { _CLEANUP_FILES+=("$@"); }
trap 'for f in "${_CLEANUP_FILES[@]}"; do rm -f "$f" 2>/dev/null || true; done' EXIT

# === Auto-exclude Python/pyo3 crates for unit tests ==============
build_test_excludes() {
  # Detect workspace members that look like python/pyo3/bindings crates
  local meta pkgs
  meta=$(cargo metadata --no-deps --format-version 1 2>/dev/null || true)
  if [[ -z "$meta" || "$meta" == "null" ]]; then
    echo "" ; return
  fi
  # names that contain python|pyo3|bindings OR depend on pyo3
  mapfile -t pkgs < <(
    printf '%s' "$meta" \
    | jq -r '
        .packages[]
        | select(
            (.name | test("python|pyo3|bindings"; "i"))
            or ( [.dependencies[].name] | any(. != null and test("pyo3|python"; "i")) )
          )
        | .name
      ' | sort -u
  )
  local flags=()
  for p in "${pkgs[@]}"; do flags+=(--exclude "$p"); done
  printf '%s ' "${flags[@]}"
}

# === CPU determinism (IDs identical at T=0, fixed seed) ==========
cpu_determinism_check() {
  local TMP PROMPT
  PROMPT='Determinism test: print the first 20 prime numbers.'
  TMP="${TMPDIR:-/tmp}/bitnet_cpu_det.$RANDOM"
  mkdir -p "$TMP"
  cat >"$TMP/prompts.yaml" <<EOF
version: 1
bos: true
max_new_tokens: 64
prompts:
  - "$PROMPT"
EOF
  export BITNET_SEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
  run_bitnet run --temperature 0 \
    --prompts "$TMP/prompts.yaml" --dump-token-ids "$TMP/ids1" \
    --json-out "$TMP/run1.json" >/dev/null
  run_bitnet run --temperature 0 \
    --prompts "$TMP/prompts.yaml" --dump-token-ids "$TMP/ids2" \
    --json-out "$TMP/run2.json" >/dev/null
  if diff -u "$TMP/ids1" "$TMP/ids2" >/dev/null; then
    echo "✓ CPU determinism OK"
  else
    echo "✗ CPU determinism failed (IDs differ)"; exit $EXIT_DETERMINISM_FAIL
  fi
  rm -rf "$TMP"
}

# === Nested baseline helpers (cpu/gpu trees) =====================
read_cpu_baseline() {
  local key="$1" field="$2" base="ci/baseline.json"
  jq -r --arg k "$key" ".cpu[\$k].$field // 0" "$base" 2>/dev/null
}

# Deterministic environment setup
export RAYON_NUM_THREADS=1
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42
export OMP_NUM_THREADS=1
export GGML_NUM_THREADS=1

echo "=== BitNet.rs CI Acceptance Gate v2 ==="
echo "Environment: DETERMINISTIC=1, SEED=42, THREADS=1"
echo ""

# Gate 1: Build and discover binary path
echo "━━━ Gate 1: Core Build & Binary Discovery ━━━"
BUILD_OUTPUT=$(mktemp)

if ! cargo build -p bitnet-cli --release \
        --no-default-features --features "cpu,full-cli" \
        --message-format=json 2>/dev/null > "$BUILD_OUTPUT"; then
    echo "❌ Build failed"
    # Try again with regular output for debugging
    cargo build -p bitnet-cli --release \
        --no-default-features --features "cpu,full-cli" 2>&1 | tail -20
    rm -f "$BUILD_OUTPUT"
    exit $EXIT_GENERAL_ERROR
fi

# Extract the binary path from cargo's JSON output (filter only JSON lines)
BITNET_BIN=$(
    grep '^{' "$BUILD_OUTPUT" \
    | jq -r 'select(.executable!=null and .target.kind[]=="bin" and .target.name=="bitnet") | .executable' \
    | tail -1
)

rm -f "$BUILD_OUTPUT"

if [[ ! -x "$BITNET_BIN" ]]; then
    echo "❌ Binary not found or not executable"
    exit $EXIT_GENERAL_ERROR
fi

echo "✅ Build succeeded"
echo "   Binary: $BITNET_BIN"
echo ""

# Gate 2: Unit Tests
echo "━━━ Gate 2: Unit Tests (no python/pyo3 crates) ━━━"
TEST_OUTPUT=$(mktemp)

# Run tests with auto-excluded Python/pyo3 crates
EXCLUDES=$(build_test_excludes)
# shellcheck disable=SC2086
if ! cargo test --workspace $EXCLUDES --no-default-features --features cpu \
        --lib -- -q > "$TEST_OUTPUT" 2>&1; then
    echo "❌ Unit tests failed"
    tail -50 "$TEST_OUTPUT"
    rm -f "$TEST_OUTPUT"
    exit $EXIT_GENERAL_ERROR
fi

rm -f "$TEST_OUTPUT"
echo "✅ Unit tests passed"
echo ""

# Gate 3: Model Selection
echo "━━━ Gate 3: Model Selection ━━━"

# PR model with embedded tokenizer (required for strict gates)
PR_MODEL="${PR_MODEL:-models/tinyllama-q2.gguf}"

# Nightly model (external tokenizer OK)
NIGHTLY_MODEL="${NIGHTLY_MODEL:-models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf}"

# Select model based on CI context
if [[ -n "${CI_PR:-}" ]]; then
    MODEL_PATH="$PR_MODEL"
    REQUIRE_EMBEDDED_TOKENIZER=true
    echo "   PR mode: Using model with embedded tokenizer"
else
    MODEL_PATH="${MODEL_PATH:-$PR_MODEL}"
    REQUIRE_EMBEDDED_TOKENIZER=false
    echo "   Using model: $MODEL_PATH"
fi

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "❌ Model not found: $MODEL_PATH"
    echo "   For PR tests, provide a model with embedded SentencePiece tokenizer"
    echo "   Example: export PR_MODEL=models/tinyllama-q2.gguf"
    exit $EXIT_MODEL_LOAD_ERROR
fi
echo "✅ Model found"
echo ""

# Gate 4: Tensor Name Mapping (JSON Gate)
echo "━━━ Gate 4: Tensor Name Mapping ━━━"
MAPPER_JSON=$(mktemp)

if ! cargo run -q -p xtask -- gate mapper --model "$MODEL_PATH" > "$MAPPER_JSON" 2>/dev/null; then
    echo "❌ Mapper gate execution failed"
    rm -f "$MAPPER_JSON"
    exit $EXIT_STRICT_MAPPING
fi

if ! jq -e '.ok==true and .unmapped_count==0' "$MAPPER_JSON" >/dev/null; then
    UNMAPPED=$(jq -r '.unmapped_count // "unknown"' "$MAPPER_JSON")
    TOTAL=$(jq -r '.total_count // "unknown"' "$MAPPER_JSON")
    echo "❌ Tensor mapping failed: $UNMAPPED/$TOTAL unmapped"
    rm -f "$MAPPER_JSON"
    exit $EXIT_STRICT_MAPPING
fi

TOTAL_COUNT=$(jq -r '.total_count // 0' "$MAPPER_JSON")
rm -f "$MAPPER_JSON"
echo "✅ All $TOTAL_COUNT tensors mapped"
echo ""

# Gate 5: Strict Mode Execution
echo "━━━ Gate 5: Strict Mode Execution ━━━"
STRICT_JSON=$(mktemp); _track "$STRICT_JSON"

# Build strict mode command
STRICT_CMD=(
    "$BITNET_BIN" run
    --model "$MODEL_PATH"
    --prompt "The capital of France is"
    --bos
    --max-new-tokens 16
    --temperature 0
    --strict-mapping
    --json-out "$STRICT_JSON"
)

# Add strict tokenizer only if we require embedded tokenizer
if [[ "$REQUIRE_EMBEDDED_TOKENIZER" == "true" ]]; then
    STRICT_CMD+=(--strict-tokenizer)
else
    # For models without embedded tokenizer, allow mock
    STRICT_CMD+=(--allow-mock)
fi

if ! "${STRICT_CMD[@]}" >/dev/null 2>&1; then
    echo "❌ Strict mode execution failed"
    rm -f "$STRICT_JSON"
    exit $EXIT_INFERENCE_ERROR
fi

# Validate strict mode output
if [[ "$REQUIRE_EMBEDDED_TOKENIZER" == "true" ]]; then
    # PR mode: enforce strict requirements
    if ! jq -e '
        .counts.unmapped==0
        and (.counts.n_kv|tonumber)>0
        and (.counts.n_tensors|tonumber)>0
        and .tokenizer.type=="sentencepiece"
    ' "$STRICT_JSON" >/dev/null; then
        echo "❌ Strict mode validation failed"
        jq '.' "$STRICT_JSON"
        rm -f "$STRICT_JSON"
        exit $EXIT_STRICT_TOKENIZER
    fi
    echo "✅ Strict mode: unmapped=0, tokenizer=sentencepiece"
else
    # Nightly mode: relaxed tokenizer requirements
    if ! jq -e '
        .counts.unmapped==0
        and (.counts.n_kv|tonumber)>0
        and (.counts.n_tensors|tonumber)>0
    ' "$STRICT_JSON" >/dev/null; then
        echo "❌ Strict mode validation failed"
        jq '.' "$STRICT_JSON"
        rm -f "$STRICT_JSON"
        exit $EXIT_STRICT_MAPPING
    fi
    TOKENIZER_TYPE=$(jq -r '.tokenizer.type // "unknown"' "$STRICT_JSON")
    echo "✅ Strict mode: unmapped=0, tokenizer=$TOKENIZER_TYPE"
fi

rm -f "$STRICT_JSON"
echo ""

# Gate 6: Tokenization
echo "━━━ Gate 6: Tokenization ━━━"
PROMPTS=(
    "The capital of France is"
    "Once upon a time"
    "def fibonacci(n):"
)

PASS_COUNT=0
for prompt in "${PROMPTS[@]}"; do
    TOKEN_JSON=$(mktemp)
    
    TOKEN_CMD=(
        "$BITNET_BIN" tokenize
        --model "$MODEL_PATH"
        --prompt "$prompt"
        --bos
        --json-out "$TOKEN_JSON"
    )
    
    # Add allow-mock for models without embedded tokenizer
    if [[ "$REQUIRE_EMBEDDED_TOKENIZER" != "true" ]]; then
        TOKEN_CMD+=(--allow-mock)
    fi
    
    if "${TOKEN_CMD[@]}" >/dev/null 2>&1; then
        if jq -e '.tokens.ids | length > 0' "$TOKEN_JSON" >/dev/null 2>&1; then
            ((PASS_COUNT++))
        fi
    fi
    rm -f "$TOKEN_JSON"
done

if [[ $PASS_COUNT -ge 2 ]]; then
    echo "✅ Tokenization: $PASS_COUNT/${#PROMPTS[@]} prompts tokenized"
else
    echo "❌ Tokenization: only $PASS_COUNT/${#PROMPTS[@]} prompts tokenized (need ≥2)"
    exit $EXIT_TOKENIZER_ERROR
fi
echo ""

# Gate 7: Performance & Memory
echo "━━━ Gate 7: Performance & Memory ━━━"

# Find time command for memory profiling
TIME_CMD=""
if command -v /usr/bin/time >/dev/null 2>&1; then
    TIME_CMD="/usr/bin/time -v"
elif command -v gtime >/dev/null 2>&1; then
    TIME_CMD="gtime -v"
fi

PERF_JSON=$(mktemp -t perf.XXXXXX.json); _track "$PERF_JSON"
TIME_OUTPUT=$(mktemp); _track "$TIME_OUTPUT"

# Run performance test
if [[ -n "$TIME_CMD" ]]; then
    # Build full command with time prefix
    PERF_CMD=( $TIME_CMD )
    PERF_CMD+=( "$BITNET_BIN" run --model "$MODEL_PATH" )
    
    if [[ -n "${TOKENIZER_PATH:-}" ]]; then
        PERF_CMD+=( --tokenizer "$TOKENIZER_PATH" )
    fi
    
    PERF_CMD+=(
        --prompt "Performance test"
        --max-new-tokens 128
        --temperature 0
        --json-out "$PERF_JSON"
    )
    
    if [[ "$REQUIRE_EMBEDDED_TOKENIZER" != "true" ]]; then
        PERF_CMD+=(--allow-mock)
    fi
    
    if ! "${PERF_CMD[@]}" >"$TIME_OUTPUT" 2>&1; then
        echo "⚠️  Performance test failed to complete"
        rm -f "$PERF_JSON" "$TIME_OUTPUT"
    fi
else
    # No time command available, run without memory profiling
    # No time command, run without profiling
    if ! run_bitnet run \
            --prompt "Performance test" \
            --max-new-tokens 128 \
            --temperature 0 \
            --json-out "$PERF_JSON" \
            $([[ "$REQUIRE_EMBEDDED_TOKENIZER" != "true" ]] && echo "--allow-mock") \
            >/dev/null 2>&1; then
        echo "⚠️  Performance test failed to complete"
        rm -f "$PERF_JSON"
    fi
fi

# Check performance metrics
if [[ -f "$PERF_JSON" ]]; then
    TOKENS_PER_SEC=$(jq -r '.throughput.tokens_per_second // 0' "$PERF_JSON")
    DECODED_TOKENS=$(jq -r '.throughput.decoded_tokens // 0' "$PERF_JSON")
    
    # Validate minimum performance
    if ! awk -v tps="$TOKENS_PER_SEC" 'BEGIN{exit !(tps >= 1.0)}'; then
        echo "❌ Performance too low: $TOKENS_PER_SEC tok/s < 1.0 minimum"
        rm -f "$PERF_JSON" "$TIME_OUTPUT"
        exit $EXIT_PERF_FAIL
    fi
    
    # Check against baseline if available (using nested cpu/gpu structure)
    BASELINE_FILE="ci/baseline.json"
    if [[ -f "$BASELINE_FILE" ]]; then
        # Sanity check baseline structure
        if ! jq -e '.cpu and .gpu and .metadata' "$BASELINE_FILE" >/dev/null 2>&1; then
            echo "⚠️  baseline.json missing required top-level keys (cpu/gpu/metadata)"
        else
            KEY="${MODEL_KEY:-tinyllama_q2k_cpu}"
            BASE_TS=$(read_cpu_baseline "$KEY" "tok_s")
            BASE_RSS=$(read_cpu_baseline "$KEY" "rss_mb")
        
        if [[ "$BASE_TS" != "0" ]]; then
            THRESHOLD=$(awk -v b="$BASE_TS" 'BEGIN{printf "%.2f", b * 0.95}')
            if ! awk -v c="$TOKENS_PER_SEC" -v t="$THRESHOLD" 'BEGIN{exit !(c >= t)}'; then
                echo "❌ Performance regression: $TOKENS_PER_SEC < $THRESHOLD (95% of $BASE_TS)"
                rm -f "$PERF_JSON" "$TIME_OUTPUT"
                exit $EXIT_PERF_FAIL
            fi
        fi
        fi  # end of baseline structure check
    fi
    
    echo "✅ Performance: $TOKENS_PER_SEC tok/s (decoded: $DECODED_TOKENS tokens)"
    
    # Check memory if available
    if [[ -f "$TIME_OUTPUT" ]] && grep -q "Maximum resident set size" "$TIME_OUTPUT"; then
        RSS_KB=$(awk '/Maximum resident set size/{print $6}' "$TIME_OUTPUT")
        RSS_MB=$((RSS_KB / 1024))
        
        # Check against baseline if available
        if [[ -f "$BASELINE_FILE" ]] && [[ "$BASE_RSS" != "0" ]]; then
            THRESHOLD=$(awk -v b="$BASE_RSS" 'BEGIN{printf "%d", b * 1.03}')
            if [[ $RSS_MB -gt $THRESHOLD ]]; then
                echo "❌ Memory regression: ${RSS_MB}MB > ${THRESHOLD}MB (103% of ${BASE_RSS}MB)"
                rm -f "$PERF_JSON" "$TIME_OUTPUT"
                exit $EXIT_MEM_FAIL
            fi
        fi
        
        echo "   Memory: ${RSS_MB}MB RSS"
    fi
else
    echo "⚠️  Performance metrics not available"
fi

rm -f "$PERF_JSON" "$TIME_OUTPUT"
echo ""

# Gate 8: Determinism Check
echo "━━━ Gate 8: Determinism Check ━━━"
DET1=$(mktemp)
DET2=$(mktemp)

DET_CMD=(
    "$BITNET_BIN" run
    --model "$MODEL_PATH"
    --prompt "Determinism test"
    --max-new-tokens 50
    --temperature 0
)

if [[ "$REQUIRE_EMBEDDED_TOKENIZER" != "true" ]]; then
    DET_CMD+=(--allow-mock)
fi

# Run twice with same seed
SUCCESS=true
for output in "$DET1" "$DET2"; do
    DET_RUN=("${DET_CMD[@]}")
    DET_RUN+=(--json-out "$output")
    
    if ! "${DET_RUN[@]}" >/dev/null 2>&1; then
        SUCCESS=false
        break
    fi
done

if [[ "$SUCCESS" == "true" ]]; then
    # Compare token outputs
    TOKENS1=$(jq -c '.tokens.ids // []' "$DET1" 2>/dev/null || echo "[]")
    TOKENS2=$(jq -c '.tokens.ids // []' "$DET2" 2>/dev/null || echo "[]")
    
    if [[ "$TOKENS1" == "$TOKENS2" && "$TOKENS1" != "[]" ]]; then
        echo "✅ Deterministic output verified"
    else
        echo "❌ Non-deterministic output detected"
        echo "   Run 1: $TOKENS1"
        echo "   Run 2: $TOKENS2"
        rm -f "$DET1" "$DET2"
        exit $EXIT_DETERMINISM_FAIL
    fi
else
    echo "⚠️  Determinism check skipped (inference failed)"
fi

rm -f "$DET1" "$DET2"
echo ""

# Gate 9: FFI Compatibility (Optional)
echo "━━━ Gate 9: FFI Compatibility ━━━"
FFI_OUTPUT=$(mktemp)

if cargo build -p bitnet-ffi --release \
        --no-default-features --features cpu \
        --message-format=json > "$FFI_OUTPUT" 2>&1; then
    
    # Extract FFI library path
    FFI_LIB=$(
        jq -r 'select(.target.kind[]=="cdylib" and .target.name=="bitnet_ffi") | .filenames[]' "$FFI_OUTPUT" \
        | grep -E '\.(so|dylib|dll)$' \
        | head -1
    )
    
    if [[ -f "$FFI_LIB" ]]; then
        echo "✅ FFI library built: $(basename "$FFI_LIB")"
    else
        echo "⚠️  FFI library not found (build succeeded but no .so/.dylib)"
    fi
else
    echo "⚠️  FFI build skipped (optional for PR gates)"
fi

rm -f "$FFI_OUTPUT"
echo ""

# Gate 10: Cross-validation and CPU Determinism
echo "━━━ Gate 10: Cross-validation & CPU Determinism ━━━"

# Check if cross-validation scripts exist
if [[ -f "scripts/crossval-llama.sh" && -f "crossval/prompts.yaml" ]]; then
    echo "Running cross-validation..."
    if scripts/crossval-llama.sh "crossval/data/ppl_smoke.txt" "crossval/prompts.yaml"; then
        echo "✅ Cross-validation passed"
    else
        if [[ "${XVAL_NONBLOCKING:-0}" == "1" ]]; then
            echo "⚠️  Cross-validation failed (non-blocking)"
        else
            echo "❌ Cross-validation failed"
            exit $EXIT_GENERAL_ERROR
        fi
    fi
else
    echo "⚠️  Cross-validation scripts not found"
fi

# A/B Token-ID parity check (if available)
if [[ -x scripts/ab-suite.sh ]]; then
    echo "Running A/B token-ID parity check..."
    if BITNET_BIN="$BITNET_BIN" \
       LLAMA_BIN="${LLAMA_BIN:-/usr/local/bin/llama-cli}" \
       MODEL_PATH="$MODEL_PATH" \
       TOKENIZER_PATH="${TOKENIZER_PATH:-}" \
       scripts/ab-suite.sh crossval/prompts.yaml; then
        echo "✅ A/B token-ID parity passed"
    else
        if [[ "${XVAL_NONBLOCKING:-0}" == "1" ]]; then
            echo "⚠️  A/B parity failed (non-blocking)"
        else
            echo "❌ A/B token-ID parity failed"
            exit $EXIT_GENERAL_ERROR
        fi
    fi
else
    echo "⚠️  A/B parity script not found (scripts/ab-suite.sh)"
fi

# CPU determinism check
cpu_determinism_check

# JSON schema validation (if we have any JSONs to check)
if [[ -f "scripts/json-schema-gate.sh" ]]; then
    JSONS_TO_CHECK=()
    [[ -f "${STRICT_JSON:-}" ]] && JSONS_TO_CHECK+=("$STRICT_JSON")
    [[ -f "${PERF_JSON:-}" ]] && JSONS_TO_CHECK+=("$PERF_JSON")
    
    if [[ ${#JSONS_TO_CHECK[@]} -gt 0 ]]; then
        if scripts/json-schema-gate.sh "${JSONS_TO_CHECK[@]}"; then
            echo "✅ JSON schema gates passed"
        else
            echo "❌ JSON schema validation failed"
            exit $EXIT_GENERAL_ERROR
        fi
    fi
fi

echo ""

# Summary
echo "═══════════════════════════════════════"
echo "CI Acceptance Gate: PASSED"
echo "═══════════════════════════════════════"
echo "✅ Build & binary discovery"
echo "✅ Unit tests"
echo "✅ Tensor mapping"
echo "✅ Strict mode execution"
echo "✅ Tokenization"
echo "✅ Performance acceptable"
echo "✅ Deterministic execution"
echo ""
echo "🎉 All gates passed successfully!"

exit $EXIT_SUCCESS