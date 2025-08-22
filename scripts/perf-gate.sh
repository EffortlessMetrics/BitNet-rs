#!/usr/bin/env bash
set -euo pipefail

# Performance regression gate
BASELINE="${1:-baseline.json}"
RESULTS="bench-results.jsonl"

# Run benchmark
echo "Running performance benchmark..."
scripts/bench-decode.sh >/dev/null 2>&1

if [ ! -f "$RESULTS" ]; then
    echo "Error: No benchmark results found"
    exit 1
fi

# Extract current metrics
CURR_DECODE_TPS=$(jq -s '[.[].throughput_tps.decode] | sort | .[length/2|floor]' "$RESULTS" 2>/dev/null || echo "0")

if [ -f "$BASELINE" ]; then
    # Compare with baseline
    BASE_DECODE_TPS=$(jq '.decode_tps_median' "$BASELINE" 2>/dev/null || echo "0")
    
    # Calculate regression percentage
    if [ "$(echo "$BASE_DECODE_TPS > 0" | bc -l)" -eq 1 ]; then
        REGRESSION=$(echo "scale=2; (($BASE_DECODE_TPS - $CURR_DECODE_TPS) / $BASE_DECODE_TPS) * 100" | bc -l)
        
        echo "Performance Comparison:"
        echo "  Baseline decode TPS: $BASE_DECODE_TPS"
        echo "  Current decode TPS:  $CURR_DECODE_TPS"
        echo "  Regression: ${REGRESSION}%"
        
        # Fail if regression > 10%
        if [ "$(echo "$REGRESSION > 10" | bc -l)" -eq 1 ]; then
            echo "ERROR: Performance regression exceeds 10% threshold"
            exit 1
        elif [ "$(echo "$REGRESSION > 5" | bc -l)" -eq 1 ]; then
            echo "WARNING: Performance regression between 5-10%"
        else
            echo "✓ Performance within acceptable range"
        fi
    else
        echo "Warning: Invalid baseline TPS value"
    fi
else
    echo "No baseline found. Creating new baseline..."
    jq -s '{
        decode_tps_median: [.[].throughput_tps.decode] | sort | .[length/2|floor],
        prefill_tps_median: [.[].throughput_tps.prefill] | sort | .[length/2|floor],
        e2e_tps_median: [.[].throughput_tps.e2e] | sort | .[length/2|floor],
        timestamp: now | strftime("%Y-%m-%d %H:%M:%S")
    }' "$RESULTS" > "$BASELINE"
    echo "Baseline created at: $BASELINE"
fi

# Save current metrics for reporting
jq -s '{
    decode_tps_median: [.[].throughput_tps.decode] | sort | .[length/2|floor],
    prefill_tps_median: [.[].throughput_tps.prefill] | sort | .[length/2|floor],
    e2e_tps_median: [.[].throughput_tps.e2e] | sort | .[length/2|floor],
    samples: length
}' "$RESULTS" > current-perf.json

echo ""
echo "Current performance saved to: current-perf.json"