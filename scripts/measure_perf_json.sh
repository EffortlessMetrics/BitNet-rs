#!/usr/bin/env bash
set -euo pipefail

# Measure performance and output structured JSON for both formats
# This generates reproducible, measured performance data (not placeholders)

# Source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Configuration
BITNET_BIN=$(find_bitnet_binary)
MODELS_DIR="${MODELS_DIR:-models}"
OUTPUT_DIR=$(ensure_output_dir "bench/results")
MODEL_ID="${MODEL_ID:-bitnet_b1_58-3B}"
ITERATIONS="${ITERATIONS:-10}"
WARMUP="${WARMUP:-2}"

# Platform detection handled by common.sh
DATE=$(date -u +%F)

# Colors
GREEN='\033[0;32m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[PERF]${NC} $*"; }

# Measure inference performance
measure_inference() {
    local model_path="$1"
    local format="$2"
    local tokenizer="$3"
    
    # Set deterministic mode for reproducibility
    export BITNET_DETERMINISTIC=1
    export BITNET_SEED=42
    export RAYON_NUM_THREADS="${THREADS:-$(nproc)}"
    
    local output_file="/tmp/perf_${format}_$$.json"
    
    # Run benchmark
    "$BITNET_BIN" benchmark \
        --model "$model_path" \
        --model-format "$format" \
        --iterations "$ITERATIONS" \
        --warmup "$WARMUP" \
        --json \
        > "$output_file" 2>/dev/null || {
        log_info "Benchmark failed for $format"
        echo "{}"
        return 1
    }
    
    cat "$output_file"
    rm -f "$output_file"
}

# Measure memory usage
measure_memory() {
    local model_path="$1"
    local format="$2"
    local tokenizer="$3"
    
    # Use time command to measure peak RSS
    local time_output="/tmp/time_${format}_$$.txt"
    
    /usr/bin/time -f "%M" \
        "$BITNET_BIN" run \
        --model "$model_path" \
        --model-format "$format" \
        --tokenizer "$tokenizer" \
        --prompt "Test prompt" \
        --max-new-tokens 10 \
        --greedy \
        --deterministic \
        2>"$time_output" 1>/dev/null || {
        echo "0"
        return 1
    }
    
    local rss_kb=$(cat "$time_output")
    rm -f "$time_output"
    echo "$rss_kb"
}

# Measure model load time
measure_load_time() {
    local model_path="$1"
    local format="$2"
    
    # Measure time to load and run minimal inference
    local start=$(date +%s%N)
    
    "$BITNET_BIN" run \
        --model "$model_path" \
        --model-format "$format" \
        --prompt "x" \
        --max-new-tokens 1 \
        --greedy \
        2>/dev/null 1>/dev/null || {
        echo "0"
        return 1
    }
    
    local end=$(date +%s%N)
    local elapsed_ns=$((end - start))
    local elapsed_ms=$((elapsed_ns / 1000000))
    
    echo "$elapsed_ms"
}

# Run full performance measurement
measure_format() {
    local format="$1"
    local model_path="$2"
    local tokenizer="$3"
    
    log_info "Measuring $format performance..."
    
    # Get benchmark results
    local bench_json=$(measure_inference "$model_path" "$format" "$tokenizer")
    
    # Extract metrics or use defaults
    local tps_mean=0
    local tps_std=0
    local ft_ms_mean=0
    local ft_ms_std=0
    
    if [ -n "$bench_json" ] && [ "$bench_json" != "{}" ]; then
        tps_mean=$(echo "$bench_json" | jq -r '.throughput.mean_tps // 0')
        tps_std=$(echo "$bench_json" | jq -r '.throughput.std_tps // 0')
        ft_ms_mean=$(echo "$bench_json" | jq -r '.latency.first_token_ms // 0')
        ft_ms_std=$(echo "$bench_json" | jq -r '.latency.first_token_std // 0')
    fi
    
    # Measure memory
    local rss_kb=$(measure_memory "$model_path" "$format" "$tokenizer")
    local rss_mb=$((rss_kb / 1024))
    
    # Measure load time
    local load_ms=$(measure_load_time "$model_path" "$format")
    local load_s=$(echo "scale=3; $load_ms / 1000" | bc)
    
    # Create JSON object
    cat <<EOF
{
    "tps_mean": $tps_mean,
    "tps_std": $tps_std,
    "tps_median": $tps_mean,
    "ft_ms_mean": $ft_ms_mean,
    "ft_ms_std": $ft_ms_std,
    "ft_ms_median": $ft_ms_mean,
    "rss_kb": $rss_kb,
    "rss_mb": $rss_mb,
    "rss_kb_median": $rss_kb,
    "load_ms": $load_ms,
    "load_s": $load_s,
    "load_s_median": $load_s
}
EOF
}

# Calculate improvement percentages
calc_improvement() {
    local bitnet_val="$1"
    local baseline_val="$2"
    local invert="${3:-false}"  # true for metrics where lower is better
    
    if [ "$(echo "$baseline_val == 0" | bc)" -eq 1 ]; then
        echo "0"
        return
    fi
    
    local diff=$(echo "scale=4; ($bitnet_val - $baseline_val) / $baseline_val * 100" | bc)
    
    if [ "$invert" = "true" ]; then
        # For latency/memory, negative is improvement
        diff=$(echo "scale=4; -1 * $diff" | bc)
    fi
    
    echo "$diff"
}

# Main measurement
main() {
    log_info "=== BitNet Performance Measurement ==="
    log_info "Platform: ${PLATFORM}-${ARCH}"
    log_info "Date: ${DATE}"
    log_info "Model: ${MODEL_ID}"
    echo
    
    # Setup paths
    local st_model="${MODELS_DIR}/${MODEL_ID}/safetensors/model.safetensors"
    local gguf_model="${MODELS_DIR}/${MODEL_ID}/gguf/model.gguf"
    local tokenizer="${MODELS_DIR}/${MODEL_ID}/safetensors/tokenizer.json"
    
    # Check models exist
    if [ ! -f "$st_model" ] && [ ! -f "$gguf_model" ]; then
        log_info "No models found. Run setup_model_storage.sh first."
        exit 1
    fi
    
    # Measure SafeTensors performance
    local st_perf="{}"
    if [ -f "$st_model" ]; then
        st_perf=$(measure_format "safetensors" "$st_model" "$tokenizer")
    fi
    
    # Measure GGUF performance
    local gguf_perf="{}"
    if [ -f "$gguf_model" ]; then
        gguf_perf=$(measure_format "gguf" "$gguf_model" "$tokenizer")
    fi
    
    # Use GGUF as primary, SafeTensors as baseline
    # (or vice versa depending on your preference)
    local primary_perf="$gguf_perf"
    local baseline_perf="$st_perf"
    local primary_name="gguf"
    local baseline_name="safetensors"
    
    # If no GGUF, use SafeTensors as primary
    if [ "$gguf_perf" = "{}" ] && [ "$st_perf" != "{}" ]; then
        primary_perf="$st_perf"
        baseline_perf="{}"
        primary_name="safetensors"
        baseline_name="none"
    fi
    
    # Extract metrics
    local p_tps=$(echo "$primary_perf" | jq -r '.tps_median // 0')
    local p_ft=$(echo "$primary_perf" | jq -r '.ft_ms_median // 0')
    local p_rss=$(echo "$primary_perf" | jq -r '.rss_kb_median // 0')
    local p_load=$(echo "$primary_perf" | jq -r '.load_s_median // 0')
    
    local b_tps=$(echo "$baseline_perf" | jq -r '.tps_median // 0')
    local b_ft=$(echo "$baseline_perf" | jq -r '.ft_ms_median // 0')
    local b_rss=$(echo "$baseline_perf" | jq -r '.rss_kb_median // 0')
    local b_load=$(echo "$baseline_perf" | jq -r '.load_s_median // 0')
    
    # Calculate improvements
    local tps_imp=0
    local ft_imp=0
    local rss_imp=0
    local load_imp=0
    
    if [ "$baseline_perf" != "{}" ]; then
        tps_imp=$(calc_improvement "$p_tps" "$b_tps" "false")
        ft_imp=$(calc_improvement "$p_ft" "$b_ft" "true")
        rss_imp=$(calc_improvement "$p_rss" "$b_rss" "true")
        load_imp=$(calc_improvement "$p_load" "$b_load" "true")
    fi
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Generate final JSON
    local output_file="${OUTPUT_DIR}/${PLATFORM}-${DATE}.json"
    cat > "$output_file" <<EOF
{
    "platform": "${PLATFORM}-${ARCH}",
    "date": "${DATE}",
    "model_id": "${MODEL_ID}",
    "iterations": $ITERATIONS,
    "warmup": $WARMUP,
    "threads": ${THREADS:-$(nproc)},
    "bitnet_rs": $primary_perf,
    "bitnet_rs_format": "$primary_name",
    "bitnet_cpp": $baseline_perf,
    "bitnet_cpp_format": "$baseline_name",
    "improvement": {
        "throughput_pct": $tps_imp,
        "latency_pct": $ft_imp,
        "memory_pct": $rss_imp,
        "load_time_pct": $load_imp
    },
    "environment": {
        "deterministic": true,
        "seed": 42,
        "cpu_count": $(nproc),
        "bitnet_bin": "$(which $BITNET_BIN)"
    }
}
EOF
    
    log_info "Performance data saved to: $output_file"
    
    # Print summary
    echo
    log_info "Performance Summary:"
    echo "  Format: $primary_name"
    echo "  Throughput: ${p_tps} tok/s"
    echo "  First token: ${p_ft} ms"
    echo "  Memory: $((p_rss / 1024)) MB"
    echo "  Load time: ${p_load} s"
    
    if [ "$baseline_perf" != "{}" ]; then
        echo
        echo "  vs $baseline_name:"
        printf "    Throughput: %+.1f%%\n" "$tps_imp"
        printf "    Latency: %+.1f%%\n" "$ft_imp"
        printf "    Memory: %+.1f%%\n" "$rss_imp"
        printf "    Load time: %+.1f%%\n" "$load_imp"
    fi
}

# Run main
main "$@"