#!/usr/bin/env bash
set -euo pipefail

# Check required dependencies
command -v jq >/dev/null || { echo "Error: jq not found. Please install jq."; exit 1; }
command -v bc >/dev/null || { echo "Error: bc not found. Please install bc."; exit 1; }

# ---- Config (override via env) ----------------------------------------------
: "${BITNET_BIN:=$(command -v bitnet || echo "target/release/bitnet")}"
: "${MODEL_PATH:?Set MODEL_PATH=path/to/model.gguf}"
: "${TOKENIZER:?Set TOKENIZER=path/to/tokenizer.json}"

# Performance gate thresholds
: "${PERF_REGRESSION_THRESHOLD:=10}"  # Fail if >10% regression
: "${PERF_BASELINE_FILE:=perf_baseline.json}"
: "${PERF_RESULTS_FILE:=perf_results.json}"

# Benchmark config
: "${BENCH_PROMPTS:=5}"
: "${MAX_NEW_TOKENS:=128}"

# ---- Main -------------------------------------------------------------------
echo "==> Performance Gate Check"
echo "    Regression threshold: ${PERF_REGRESSION_THRESHOLD}%"
echo

# Run benchmark
echo "Running performance benchmark..."
BENCH_JSON="$PERF_RESULTS_FILE" \
BENCH_PROMPTS="$BENCH_PROMPTS" \
MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    scripts/bench-decode.sh

# Check if baseline exists
if [[ ! -f "$PERF_BASELINE_FILE" ]]; then
    echo
    echo "No baseline found at $PERF_BASELINE_FILE"
    echo "Creating baseline from current run..."
    cp "$PERF_RESULTS_FILE" "$PERF_BASELINE_FILE"
    echo "✅ Baseline created. Future runs will compare against this."
    exit 0
fi

# Compare with baseline
echo
echo "==> Comparing with baseline..."

# Extract metrics
current_tps=$(jq -r '.results.throughput.median_tps' "$PERF_RESULTS_FILE")
baseline_tps=$(jq -r '.results.throughput.median_tps' "$PERF_BASELINE_FILE")

current_first=$(jq -r '.results.first_token_ms.median' "$PERF_RESULTS_FILE")
baseline_first=$(jq -r '.results.first_token_ms.median' "$PERF_BASELINE_FILE")

# Calculate percentage changes
tps_change=$(echo "scale=2; (($current_tps - $baseline_tps) / $baseline_tps) * 100" | bc -l)
first_change=$(echo "scale=2; (($current_first - $baseline_first) / $baseline_first) * 100" | bc -l)

# Display results
echo "Throughput:"
echo "  Baseline:  ${baseline_tps} tok/s"
echo "  Current:   ${current_tps} tok/s"
echo "  Change:    ${tps_change}%"
echo

echo "First token latency:"
echo "  Baseline:  ${baseline_first}ms"
echo "  Current:   ${current_first}ms"
echo "  Change:    ${first_change}%"
echo

# Check for regressions
failed=0

# Check throughput regression
if (( $(echo "$tps_change < -$PERF_REGRESSION_THRESHOLD" | bc -l) )); then
    echo "❌ FAIL: Throughput regression > ${PERF_REGRESSION_THRESHOLD}%"
    failed=1
fi

# Check first token latency regression (inverse - higher is worse)
if (( $(echo "$first_change > $PERF_REGRESSION_THRESHOLD" | bc -l) )); then
    echo "❌ FAIL: First token latency regression > ${PERF_REGRESSION_THRESHOLD}%"
    failed=1
fi

if [[ $failed -eq 0 ]]; then
    echo "✅ PASS: Performance within acceptable bounds"
    
    # Optionally update baseline if significant improvement
    if (( $(echo "$tps_change > 20" | bc -l) )); then
        echo
        echo "🎉 Significant improvement detected (>20%)"
        read -p "Update baseline? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cp "$PERF_RESULTS_FILE" "$PERF_BASELINE_FILE"
            echo "Baseline updated."
        fi
    fi
else
    echo
    echo "Performance gate FAILED"
    echo "To investigate:"
    echo "  1. Review changes that might impact performance"
    echo "  2. Profile with: cargo bench --workspace"
    echo "  3. If regression is expected, update baseline:"
    echo "     cp $PERF_RESULTS_FILE $PERF_BASELINE_FILE"
    exit 9
fi