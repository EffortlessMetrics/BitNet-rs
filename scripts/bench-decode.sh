#!/usr/bin/env bash
set -euo pipefail

# Check required dependencies
command -v jq >/dev/null || { echo "Error: jq not found. Please install jq."; exit 1; }
command -v bc >/dev/null || { echo "Error: bc not found. Please install bc."; exit 1; }

# ---- Config (override via env) ----------------------------------------------
: "${BITNET_BIN:=$(command -v bitnet || echo "target/release/bitnet")}"
: "${MODEL_PATH:?Set MODEL_PATH=path/to/model.gguf}"
: "${TOKENIZER:?Set TOKENIZER=path/to/tokenizer.json}"

# Benchmark parameters
: "${BENCH_PROMPTS:=3}"
: "${MAX_NEW_TOKENS:=128}"
: "${WARMUP_TOKENS:=16}"
: "${THREADS:=0}"  # 0 = use all cores

# Output files
: "${BENCH_JSON:=bench_results.json}"
: "${BENCH_BASELINE:=}"  # Optional baseline JSON for comparison

# Test prompts for benchmarking
PROMPTS=(
    "The capital of France is"
    "Explain quantum computing in simple terms:"
    "Write a function to calculate the factorial of a number:"
)

# ---- Helper functions -------------------------------------------------------
run_bench() {
    local prompt="$1"
    local max_tokens="$2"
    local output_json="$(mktemp)"
    
    "$BITNET_BIN" run \
        --model "$MODEL_PATH" \
        --tokenizer "$TOKENIZER" \
        --prompt "$prompt" \
        --max-new-tokens "$max_tokens" \
        --greedy \
        --deterministic \
        --threads "$THREADS" \
        --json-out "$output_json" \
        >/dev/null 2>&1
    
    # Extract tokens per second
    local tps=$(jq -r '.throughput.tokens_per_second' "$output_json")
    local first_ms=$(jq -r '.latency.cmd_to_first_ms' "$output_json")
    
    rm -f "$output_json"
    echo "$tps $first_ms"
}

# ---- Main -------------------------------------------------------------------
echo "==> Running decode throughput benchmark"
echo "    Model: $MODEL_PATH"
echo "    Prompts: $BENCH_PROMPTS"
echo "    Max tokens: $MAX_NEW_TOKENS"
echo "    Threads: ${THREADS:-auto}"
echo

# Warmup run
echo "Warming up..."
run_bench "${PROMPTS[0]}" "$WARMUP_TOKENS" >/dev/null

# Collect results
TPS_VALUES=()
FIRST_TOKEN_MS=()

for ((i=0; i<BENCH_PROMPTS && i<${#PROMPTS[@]}; i++)); do
    prompt="${PROMPTS[$i]}"
    echo -n "  Prompt $((i+1))/$BENCH_PROMPTS: "
    
    # Run benchmark
    read tps first_ms <<< $(run_bench "$prompt" "$MAX_NEW_TOKENS")
    
    TPS_VALUES+=("$tps")
    FIRST_TOKEN_MS+=("$first_ms")
    
    echo "${tps} tok/s (first token: ${first_ms}ms)"
done

# Calculate statistics
calculate_stats() {
    local -a values=("$@")
    local sum=0
    local count=${#values[@]}
    
    # Sort for median
    IFS=$'\n' sorted=($(sort -g <<<"${values[*]}"))
    unset IFS
    
    # Sum for mean
    for v in "${values[@]}"; do
        sum=$(echo "$sum + $v" | bc -l)
    done
    
    # Mean
    local mean=$(echo "scale=2; $sum / $count" | bc -l)
    
    # Median
    local median
    if (( count % 2 == 0 )); then
        local mid1=${sorted[$((count/2 - 1))]}
        local mid2=${sorted[$((count/2))]}
        median=$(echo "scale=2; ($mid1 + $mid2) / 2" | bc -l)
    else
        median=${sorted[$((count/2))]}
    fi
    
    echo "$mean $median"
}

# Calculate TPS stats
read mean_tps median_tps <<< $(calculate_stats "${TPS_VALUES[@]}")
read mean_first median_first <<< $(calculate_stats "${FIRST_TOKEN_MS[@]}")

echo
echo "==> Results:"
echo "    Throughput: mean=${mean_tps} tok/s, median=${median_tps} tok/s"
echo "    First token: mean=${mean_first}ms, median=${median_first}ms"

# Write JSON results
cat > "$BENCH_JSON" <<EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "model": "$MODEL_PATH",
    "config": {
        "prompts": $BENCH_PROMPTS,
        "max_new_tokens": $MAX_NEW_TOKENS,
        "threads": $THREADS
    },
    "results": {
        "throughput": {
            "mean_tps": $mean_tps,
            "median_tps": $median_tps,
            "samples": [$(IFS=,; echo "${TPS_VALUES[*]}")]
        },
        "first_token_ms": {
            "mean": $mean_first,
            "median": $median_first,
            "samples": [$(IFS=,; echo "${FIRST_TOKEN_MS[*]}")]
        }
    }
}
EOF

echo "    Results written to: $BENCH_JSON"

# Compare with baseline if provided
if [[ -n "$BENCH_BASELINE" && -f "$BENCH_BASELINE" ]]; then
    echo
    echo "==> Comparison with baseline:"
    
    baseline_tps=$(jq -r '.results.throughput.median_tps' "$BENCH_BASELINE")
    
    # Calculate percentage change
    pct_change=$(echo "scale=1; (($median_tps - $baseline_tps) / $baseline_tps) * 100" | bc -l)
    
    echo "    Baseline: ${baseline_tps} tok/s"
    echo "    Current:  ${median_tps} tok/s"
    echo "    Change:   ${pct_change}%"
    
    # Fail if regression > 10%
    if (( $(echo "$pct_change < -10" | bc -l) )); then
        echo "    ❌ Performance regression detected (>${10}% slower)"
        exit 9
    elif (( $(echo "$pct_change > 10" | bc -l) )); then
        echo "    ✅ Performance improvement (>${10}% faster)"
    else
        echo "    ✅ Performance stable (within ±10%)"
    fi
fi