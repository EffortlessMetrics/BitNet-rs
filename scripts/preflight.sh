#!/usr/bin/env bash
set -euo pipefail

# BitNet-rs preflight script - Resource monitoring and concurrency caps
# This script sets conservative defaults to prevent fork/PID/file-descriptor storms
# and ensures stable test execution across different machines

# Show current system pressure
pids_used=$(ps -e --no-headers | wc -l | xargs)
pid_max=$(cat /proc/sys/kernel/pid_max)
files_used=$(awk '{print $2}' /proc/sys/fs/file-nr)
files_max=$(cat /proc/sys/fs/file-max)
load_avg=$(uptime | awk -F'load average:' '{ print $2 }' | awk '{ print $1 }' | sed 's/,//')

echo "=== BitNet-rs System Resource Check ==="
echo "PIDs: $pids_used / $pid_max ($(echo "$pids_used * 100 / $pid_max" | bc)%)"
echo "Open files: $files_used / $files_max ($(echo "$files_used * 100 / $files_max" | bc)%)"
echo "Load average: $load_avg"

# Conservative defaults (overridable via env in CI/local)
export RUST_TEST_THREADS="${RUST_TEST_THREADS:-2}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-2}"
export CROSSVAL_WORKERS="${CROSSVAL_WORKERS:-2}"

# BLAS/NumPy thread limits (for Python validation scripts)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# BitNet-specific environment
export BITNET_DETERMINISTIC="${BITNET_DETERMINISTIC:-1}"
export BITNET_SEED="${BITNET_SEED:-42}"

# If near PID limit, drop to ultra-safe mode
pid_usage_pct=$((pids_used * 100 / pid_max))
if [ "$pid_usage_pct" -gt 85 ]; then
    export RUST_TEST_THREADS=1
    export RAYON_NUM_THREADS=1
    export CROSSVAL_WORKERS=1
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    echo "⚠️  System hot (${pid_usage_pct}% PID usage) → auto-degraded to single-threaded mode"
else
    echo "✅ System resources OK → using capped concurrency (RUST_TEST_THREADS=$RUST_TEST_THREADS, RAYON=$RAYON_NUM_THREADS)"
fi

# Show final configuration
echo "=== BitNet-rs Concurrency Configuration ==="
echo "RUST_TEST_THREADS=$RUST_TEST_THREADS"
echo "RAYON_NUM_THREADS=$RAYON_NUM_THREADS"
echo "CROSSVAL_WORKERS=$CROSSVAL_WORKERS"
echo "BITNET_DETERMINISTIC=$BITNET_DETERMINISTIC"
echo "BITNET_SEED=$BITNET_SEED"
echo "BLAS threads: OMP=$OMP_NUM_THREADS, OPENBLAS=$OPENBLAS_NUM_THREADS, MKL=$MKL_NUM_THREADS"
echo "========================================"