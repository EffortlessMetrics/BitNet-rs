#!/usr/bin/env bash
# Update performance baseline measurements for CI regression testing
# Run this on a known reference machine to establish baselines
set -euo pipefail

# Exit codes
readonly EXIT_SUCCESS=0
readonly EXIT_GENERAL_ERROR=1
readonly EXIT_MISSING_MODEL=2
readonly EXIT_BUILD_FAILED=3

# Strict deterministic environment
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

echo "=== BitNet.rs Baseline Update Tool ==="
echo "Environment: DETERMINISTIC=1, SEED=42, THREADS=1"
echo ""

# Parse arguments
UPDATE_MODEL=""
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            UPDATE_MODEL="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--model tinyllama|bitnet|all] [--force]"
            echo ""
            echo "Options:"
            echo "  --model MODEL  Update baseline for specific model (tinyllama, bitnet, or all)"
            echo "  --force        Overwrite existing baselines without confirmation"
            echo ""
            echo "Examples:"
            echo "  $0 --model tinyllama   # Update TinyLlama baseline only"
            echo "  $0 --model all --force # Update all baselines without prompting"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit $EXIT_GENERAL_ERROR
            ;;
    esac
done

# Default to all if not specified
if [ -z "$UPDATE_MODEL" ]; then
    UPDATE_MODEL="all"
fi

# Build the CLI
echo "Building bitnet-cli..."
RS_BIN=$(
    cargo build -p bitnet-cli --release --no-default-features --features "cpu,full-cli" \
        --message-format=json 2>/dev/null \
    | grep '^{' \
    | jq -r 'select(.executable!=null and .target.kind[]=="bin" and .target.name=="bitnet") | .executable' \
    | tail -1
)

if [ ! -x "${RS_BIN:-}" ]; then
    echo "❌ Failed to build bitnet-cli"
    exit $EXIT_BUILD_FAILED
fi
echo "✓ Binary: $RS_BIN"

# Find time command for memory measurement
TIME_CMD=""
if [ -x /usr/bin/time ]; then
    TIME_CMD="/usr/bin/time -v"
elif command -v gtime >/dev/null 2>&1; then
    TIME_CMD="gtime -v"
fi

# Function to measure a model's performance
measure_model() {
    local model_key="$1"
    local model_path="$2"
    local tokenizer_path="${3:-}"
    local description="$4"

    echo ""
    echo "━━━ Measuring: $description ━━━"

    # Check model exists
    if [ ! -f "$model_path" ]; then
        echo "⚠ Model not found: $model_path"
        echo "  Skipping $model_key baseline"
        return 1
    fi

    # Build command args
    local -a RUN_ARGS=(
        run
        --model "$model_path"
        --prompt "The quick brown fox jumps over the lazy dog. Once upon a time in a land far away, there lived a wise old owl who knew all the secrets of the forest."
        --max-new-tokens 256
        --temperature 0
        --json-out /tmp/perf_measure.json
    )

    if [ -n "$tokenizer_path" ] && [ -f "$tokenizer_path" ]; then
        RUN_ARGS+=(--tokenizer "$tokenizer_path")
    fi

    # Warm-up run
    echo "Warm-up run..."
    "$RS_BIN" "${RUN_ARGS[@]}" >/dev/null 2>&1

    # Measurement runs (take median of 3)
    local tokens_per_second_values=()
    local rss_mb=0

    for run in 1 2 3; do
        echo "Measurement run $run/3..."

        if [ -n "$TIME_CMD" ]; then
            # Run with memory profiling
            $TIME_CMD "$RS_BIN" "${RUN_ARGS[@]}" 2>/tmp/time_output.txt >/dev/null

            # Extract RSS if available
            if grep -q "Maximum resident set size" /tmp/time_output.txt; then
                local rss_kb=$(awk '/Maximum resident set size/{print $6}' /tmp/time_output.txt)
                rss_mb=$((rss_kb / 1024))
            fi
        else
            # Run without memory profiling
            "$RS_BIN" "${RUN_ARGS[@]}" >/dev/null 2>&1
        fi

        # Extract performance metrics
        if [ -f /tmp/perf_measure.json ]; then
            local tps=$(jq -r '.throughput.tokens_per_second // 0' /tmp/perf_measure.json)
            tokens_per_second_values+=("$tps")
        fi
    done

    # Calculate median tokens/sec (sort and take middle value)
    IFS=$'\n' sorted=($(sort -n <<<"${tokens_per_second_values[*]}"))
    local median_tps="${sorted[1]}"

    # Get model size
    local model_size_mb=$(du -m "$model_path" | cut -f1)

    # Output JSON fragment
    cat <<EOF
    "$model_key": {
        "tokens_per_second": $median_tps,
        "rss_mb": $rss_mb,
        "model_size_mb": $model_size_mb,
        "model": "$model_path",
        "description": "$description",
        "cpu": "$(uname -m)",
        "threads": 1,
        "deterministic": true
    }
EOF

    echo "✓ Measured: ${median_tps} tokens/sec, RSS: ${rss_mb}MB"

    return 0
}

# Start building baseline JSON
BASELINE_FILE="ci/baseline.json"
TEMP_BASELINE="/tmp/baseline_new.json"

echo "{" > "$TEMP_BASELINE"

# Track if we've added any entries
ADDED_ENTRIES=false

# Measure TinyLlama if requested
if [[ "$UPDATE_MODEL" == "tinyllama" ]] || [[ "$UPDATE_MODEL" == "all" ]]; then
    if measure_model \
        "tinyllama_q2k" \
        "models/tinyllama-q2.gguf" \
        "" \
        "TinyLlama 1.1B Q2_K with embedded SentencePiece tokenizer"; then

        ADDED_ENTRIES=true
    fi
fi

# Add comma if we have entries and more to come
if [[ "$ADDED_ENTRIES" == "true" ]] && [[ "$UPDATE_MODEL" == "all" ]]; then
    echo "," >> "$TEMP_BASELINE"
fi

# Measure MS BitNet if requested
if [[ "$UPDATE_MODEL" == "bitnet" ]] || [[ "$UPDATE_MODEL" == "all" ]]; then
    if measure_model \
        "ms_bitnet_i2s" \
        "models/bitnet/ggml-model-i2_s.gguf" \
        "models/bitnet/tokenizer.model" \
        "Microsoft BitNet 1.58b 2B I2_S with external tokenizer"; then

        ADDED_ENTRIES=true
    fi
fi

# Only add metadata if we have entries
if [[ "$ADDED_ENTRIES" == "true" ]]; then
    echo "," >> "$TEMP_BASELINE"
fi

# Add metadata
cat >> "$TEMP_BASELINE" <<EOF
    "metadata": {
        "version": "1.0.0",
        "created": "$(date -u +%Y-%m-%d)",
        "updated": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "machine": "$(hostname)",
        "os": "$(uname -s)",
        "arch": "$(uname -m)",
        "notes": "Baseline performance metrics for CI regression testing. Values should be updated periodically based on representative hardware.",
        "environment": {
            "RAYON_NUM_THREADS": 1,
            "BITNET_DETERMINISTIC": 1,
            "BITNET_SEED": 42,
            "OMP_NUM_THREADS": 1,
            "GGML_NUM_THREADS": 1
        }
    }
}
EOF

# Clean up temp files
rm -f /tmp/perf_measure.json /tmp/time_output.txt

# Format the JSON nicely
jq '.' "$TEMP_BASELINE" > "${TEMP_BASELINE}.formatted"
mv "${TEMP_BASELINE}.formatted" "$TEMP_BASELINE"

# Show the new baseline
echo ""
echo "━━━ New Baseline ━━━"
cat "$TEMP_BASELINE"
echo ""

# Confirm before overwriting
if [ -f "$BASELINE_FILE" ] && [ "$FORCE" != "true" ]; then
    echo "⚠ Baseline file already exists: $BASELINE_FILE"
    echo ""
    echo "Current baseline:"
    jq '.' "$BASELINE_FILE" 2>/dev/null || echo "(invalid JSON)"
    echo ""
    read -p "Overwrite with new measurements? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Baseline update cancelled"
        rm -f "$TEMP_BASELINE"
        exit $EXIT_SUCCESS
    fi
fi

# Create directory if needed
mkdir -p "$(dirname "$BASELINE_FILE")"

# Save the new baseline
cp "$TEMP_BASELINE" "$BASELINE_FILE"
rm -f "$TEMP_BASELINE"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Baseline Updated"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "File: $BASELINE_FILE"
echo ""
echo "The CI will now use these values for regression testing:"
jq -r 'to_entries | map(select(.key != "metadata")) | .[] | "  \(.key): \(.value.tokens_per_second) tok/s, \(.value.rss_mb) MB RSS"' "$BASELINE_FILE"
echo ""
echo "Remember to commit this file to the repository."

exit $EXIT_SUCCESS
