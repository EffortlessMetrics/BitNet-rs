#!/bin/bash
# Enhanced performance benchmarking script that integrates with CI/CD
# This script runs comprehensive benchmarks and generates structured output

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default configuration
RESULTS_DIR="benchmark-results"
BENCHMARK_FEATURES="${BENCHMARK_FEATURES:-cpu}"
BENCHMARK_TARGET="${BENCHMARK_TARGET:-}"
USE_CROSS="${USE_CROSS:-false}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"
ITERATIONS="${ITERATIONS:-3}"
TOKENS="${TOKENS:-32}"
SKIP_CPP="${SKIP_CPP:-true}"

# Create timestamp for this run
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
RUN_ID="benchmark_${TIMESTAMP}"

# Setup deterministic environment
setup_environment() {
    log_info "Setting up deterministic benchmark environment..."

    # Set deterministic flags
    export BITNET_DETERMINISTIC=1
    export BITNET_SEED=42
    export RAYON_NUM_THREADS=1
    export OMP_NUM_THREADS=1

    # Create results directory structure
    mkdir -p "$RESULTS_DIR"

    log_success "Environment configured for deterministic benchmarking"
}

# Run Criterion benchmarks
run_criterion_benchmarks() {
    log_info "Running Criterion performance benchmarks..."

    local benchmark_cmd=("cargo")
    if [[ "$USE_CROSS" == "true" && -n "$BENCHMARK_TARGET" ]]; then
        benchmark_cmd=("cross")
    fi

    benchmark_cmd+=("bench" "--workspace" "--no-default-features" "--features" "$BENCHMARK_FEATURES")
    if [[ -n "$BENCHMARK_TARGET" ]]; then
        benchmark_cmd+=("--target" "$BENCHMARK_TARGET")
    fi
    benchmark_cmd+=("--" "--output-format" "json")

    log_info "Running: ${benchmark_cmd[*]}"

    if timeout "$TIMEOUT_SECONDS" "${benchmark_cmd[@]}" > "$RESULTS_DIR/criterion-raw.json" 2>&1; then
        log_success "Criterion benchmarks completed successfully"

        # Parse Criterion output to standard format
        python3 << 'EOF'
import json
import sys
from pathlib import Path

def parse_criterion_output(raw_file, output_file):
    """Parse Criterion JSON output to our standard format"""
    try:
        with open(raw_file, 'r') as f:
            lines = f.readlines()

        benchmarks = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get('type') == 'benchmark_complete':
                    benchmark_id = data.get('id', {})
                    name = benchmark_id.get('function_name', 'unknown')

                    # Extract timing information
                    typical = data.get('typical', {})
                    estimate = typical.get('estimate', 0)  # nanoseconds

                    benchmarks.append({
                        'name': name,
                        'mean': {
                            'estimate': estimate
                        },
                        'throughput': data.get('throughput'),
                        'measurement_type': 'criterion'
                    })
            except json.JSONDecodeError:
                continue

        # Create our standard format
        result = {
            'benchmarks': benchmarks,
            'timestamp': '$(date -u +%Y-%m-%dT%H:%M:%SZ)',
            'source': 'criterion'
        }

        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"Parsed {len(benchmarks)} benchmarks from Criterion output")

    except Exception as e:
        print(f"Error parsing Criterion output: {e}")
        # Create fallback result
        with open(output_file, 'w') as f:
            json.dump({
                'benchmarks': [{
                    'name': 'criterion_parse_error',
                    'mean': {'estimate': 0},
                    'error': str(e)
                }],
                'timestamp': '$(date -u +%Y-%m-%dT%H:%M:%SZ)',
                'source': 'criterion_fallback'
            }, f, indent=2)

parse_criterion_output('$RESULTS_DIR/criterion-raw.json', '$RESULTS_DIR/rust-results.json')
EOF

    else
        log_warning "Criterion benchmarks timed out or failed, creating fallback results"

        # Create minimal fallback results
        cat > "$RESULTS_DIR/rust-results.json" << EOF
{
  "benchmarks": [
    {
      "name": "benchmark_timeout",
      "mean": {
        "estimate": 1000000
      },
      "error": "Benchmark suite timed out after ${TIMEOUT_SECONDS} seconds"
    }
  ],
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "source": "timeout_fallback"
}
EOF
    fi
}

# Run comparison benchmarks using benchmark_comparison.py
run_comparison_benchmarks() {
    log_info "Running comparison benchmarks..."

    # Check if we have the comparison script
    if [[ ! -f "benchmark_comparison.py" ]]; then
        log_warning "benchmark_comparison.py not found, skipping comparison benchmarks"
        return 0
    fi

    # Check if we have a test model
    local test_model=""
    if [[ -f "crossval/fixtures/test_model_small_metadata.json" ]]; then
        test_model="crossval/fixtures/test_model_small_metadata.json"
    elif [[ -n "${BITNET_GGUF:-}" ]] && [[ -f "${BITNET_GGUF}" ]]; then
        test_model="$BITNET_GGUF"
    else
        log_warning "No test model available for comparison benchmarks"
        return 0
    fi

    log_info "Using test model: $test_model"

    # Run the comparison script
    local comparison_cmd=(
        "python3" "benchmark_comparison.py"
        "--model" "$test_model"
        "--iterations" "$ITERATIONS"
        "--tokens" "$TOKENS"
        "--timeout" "60"
    )

    if [[ "$SKIP_CPP" == "true" ]]; then
        comparison_cmd+=("--skip-cpp")
    fi

    if [[ "$BENCHMARK_FEATURES" == "gpu" ]]; then
        comparison_cmd+=("--gpu")
    fi

    log_info "Running: ${comparison_cmd[*]}"

    if timeout "$TIMEOUT_SECONDS" "${comparison_cmd[@]}" > "$RESULTS_DIR/comparison-output.log" 2>&1; then
        log_success "Comparison benchmarks completed"

        # Find the results file created by the comparison script
        local results_file=$(ls benchmark_results_*.json 2>/dev/null | head -1 || echo "")
        if [[ -n "$results_file" ]]; then
            mv "$results_file" "$RESULTS_DIR/comparison-results.json"
            log_success "Comparison results saved to $RESULTS_DIR/comparison-results.json"
        else
            log_warning "No comparison results file found"
        fi
    else
        log_warning "Comparison benchmarks failed or timed out"
    fi
}

# Generate comprehensive performance report
generate_performance_report() {
    log_info "Generating comprehensive performance report..."

    cat > "$RESULTS_DIR/performance-report.json" << EOF
{
  "run_id": "$RUN_ID",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "configuration": {
    "features": "$BENCHMARK_FEATURES",
    "target": "$BENCHMARK_TARGET",
    "use_cross": $USE_CROSS,
    "iterations": $ITERATIONS,
    "tokens": $TOKENS,
    "timeout_seconds": $TIMEOUT_SECONDS,
    "skip_cpp": $SKIP_CPP
  },
  "system_info": {
    "platform": "$(uname -m)",
    "os": "$(uname -s)",
    "kernel": "$(uname -r)",
    "cpu_count": $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1),
    "rust_version": "$(rustc --version)",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "git_branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')"
  },
  "files": {
    "rust_results": "rust-results.json",
    "comparison_results": "comparison-results.json",
    "system_info": "system-info.json"
  }
}
EOF

    # Create detailed markdown report
    cat > "$RESULTS_DIR/performance-report.md" << EOF
# 📊 BitNet-rs Performance Report

**Run ID**: $RUN_ID
**Timestamp**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
**Platform**: $(uname -m) / $(uname -s)
**Git Commit**: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')

## Configuration

- **Features**: $BENCHMARK_FEATURES
- **Target**: ${BENCHMARK_TARGET:-native}
- **Cross-compilation**: $USE_CROSS
- **Iterations**: $ITERATIONS
- **Tokens per test**: $TOKENS
- **Timeout**: ${TIMEOUT_SECONDS}s

## Results Summary

EOF

    # Add Rust benchmark results
    if [[ -f "$RESULTS_DIR/rust-results.json" ]]; then
        echo "### Rust Implementation" >> "$RESULTS_DIR/performance-report.md"
        echo "" >> "$RESULTS_DIR/performance-report.md"

        python3 << 'EOF'
import json
try:
    with open('$RESULTS_DIR/rust-results.json', 'r') as f:
        data = json.load(f)

    benchmarks = data.get('benchmarks', [])
    if benchmarks:
        print(f"- **Total benchmarks**: {len(benchmarks)}")
        for bench in benchmarks[:5]:  # Show first 5
            name = bench.get('name', 'unknown')
            estimate = bench.get('mean', {}).get('estimate', 0)
            if estimate > 0:
                ms = estimate / 1_000_000  # ns to ms
                print(f"- **{name}**: {ms:.2f}ms")
    else:
        print("- No benchmark data available")
except Exception as e:
    print(f"- Error reading results: {e}")
EOF
        echo "" >> "$RESULTS_DIR/performance-report.md"
    fi

    # Add comparison results
    if [[ -f "$RESULTS_DIR/comparison-results.json" ]]; then
        echo "### Performance Comparison" >> "$RESULTS_DIR/performance-report.md"
        echo "" >> "$RESULTS_DIR/performance-report.md"

        python3 << 'EOF'
import json
try:
    with open('$RESULTS_DIR/comparison-results.json', 'r') as f:
        data = json.load(f)

    rust_data = data.get('rust', {})
    cpp_data = data.get('cpp', {})
    comparison = data.get('comparison', {})

    if rust_data:
        rust_mean = rust_data.get('mean', 0)
        print(f"- **Rust mean time**: {rust_mean:.3f}s")

        correctness = rust_data.get('response_correctness', 'unknown')
        print(f"- **Response correctness**: {correctness}")

    if cpp_data:
        cpp_mean = cpp_data.get('mean', 0)
        print(f"- **C++ mean time**: {cpp_mean:.3f}s")

    if comparison.get('speedup'):
        speedup = comparison['speedup']
        improvement = comparison.get('improvement_percent', 0)
        print(f"- **Speedup**: {speedup:.2f}x")
        if improvement > 0:
            print(f"- **Performance improvement**: {improvement:.1f}% faster")
        else:
            print(f"- **Performance difference**: {abs(improvement):.1f}% slower")

except Exception as e:
    print(f"- Error reading comparison results: {e}")
EOF
        echo "" >> "$RESULTS_DIR/performance-report.md"
    fi

    echo "## Files Generated" >> "$RESULTS_DIR/performance-report.md"
    echo "" >> "$RESULTS_DIR/performance-report.md"
    echo "- \`performance-report.json\`: Machine-readable summary" >> "$RESULTS_DIR/performance-report.md"
    echo "- \`rust-results.json\`: Criterion benchmark results" >> "$RESULTS_DIR/performance-report.md"
    echo "- \`comparison-results.json\`: Rust vs C++ comparison (if available)" >> "$RESULTS_DIR/performance-report.md"
    echo "- \`system-info.json\`: System and environment information" >> "$RESULTS_DIR/performance-report.md"
    echo "" >> "$RESULTS_DIR/performance-report.md"
    echo "---" >> "$RESULTS_DIR/performance-report.md"
    echo "*Report generated by \`run-performance-benchmarks.sh\`*" >> "$RESULTS_DIR/performance-report.md"

    log_success "Performance report generated: $RESULTS_DIR/performance-report.md"
}

# Create system information file
create_system_info() {
    log_info "Collecting system information..."

    cat > "$RESULTS_DIR/system-info.json" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "run_id": "$RUN_ID",
  "platform": {
    "architecture": "$(uname -m)",
    "os": "$(uname -s)",
    "kernel": "$(uname -r)",
    "hostname": "$(hostname)"
  },
  "hardware": {
    "cpu_count": $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1),
    "memory_gb": $(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo "unknown")
  },
  "software": {
    "rust_version": "$(rustc --version)",
    "cargo_version": "$(cargo --version)",
    "python_version": "$(python3 --version 2>&1)"
  },
  "git": {
    "commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "short_commit": "$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')",
    "branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
    "clean": $(git diff --quiet 2>/dev/null && echo "true" || echo "false")
  },
  "environment": {
    "bitnet_deterministic": "${BITNET_DETERMINISTIC:-}",
    "bitnet_seed": "${BITNET_SEED:-}",
    "rayon_num_threads": "${RAYON_NUM_THREADS:-}",
    "omp_num_threads": "${OMP_NUM_THREADS:-}",
    "rustflags": "${RUSTFLAGS:-}",
    "bitnet_cpp_root": "${BITNET_CPP_ROOT:-}"
  }
}
EOF

    log_success "System information saved to $RESULTS_DIR/system-info.json"
}

# Print usage information
usage() {
    cat << EOF
BitNet-rs Performance Benchmarking Script

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    --features FEATURES     Benchmark features (default: cpu)
    --target TARGET         Cross-compilation target
    --use-cross             Use 'cross' instead of 'cargo'
    --timeout SECONDS       Timeout for benchmarks (default: 300)
    --iterations N          Number of comparison iterations (default: 3)
    --tokens N              Tokens to generate in comparisons (default: 32)
    --include-cpp           Include C++ cross-validation benchmarks
    --results-dir DIR       Results directory (default: benchmark-results)

EXAMPLES:
    # Basic CPU benchmarks
    $0

    # GPU benchmarks with longer timeout
    $0 --features gpu --timeout 600

    # Cross-compilation for ARM64
    $0 --target aarch64-unknown-linux-gnu --use-cross

    # Include C++ comparison
    $0 --include-cpp
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            --features)
                BENCHMARK_FEATURES="$2"
                shift 2
                ;;
            --target)
                BENCHMARK_TARGET="$2"
                shift 2
                ;;
            --use-cross)
                USE_CROSS=true
                shift
                ;;
            --timeout)
                TIMEOUT_SECONDS="$2"
                shift 2
                ;;
            --iterations)
                ITERATIONS="$2"
                shift 2
                ;;
            --tokens)
                TOKENS="$2"
                shift 2
                ;;
            --include-cpp)
                SKIP_CPP=false
                shift
                ;;
            --results-dir)
                RESULTS_DIR="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Main function
main() {
    log_info "BitNet-rs Performance Benchmarking"
    log_info "==================================="

    parse_args "$@"

    log_info "Configuration:"
    log_info "  Features: $BENCHMARK_FEATURES"
    log_info "  Target: ${BENCHMARK_TARGET:-native}"
    log_info "  Cross-compilation: $USE_CROSS"
    log_info "  Timeout: ${TIMEOUT_SECONDS}s"
    log_info "  Iterations: $ITERATIONS"
    log_info "  Tokens: $TOKENS"
    log_info "  Results directory: $RESULTS_DIR"
    echo

    setup_environment
    create_system_info
    run_criterion_benchmarks
    run_comparison_benchmarks
    generate_performance_report

    log_success "Performance benchmarking complete!"
    echo
    log_info "Results available in: $RESULTS_DIR/"
    echo "  📊 performance-report.md  - Human-readable report"
    echo "  📋 performance-report.json - Machine-readable summary"
    echo "  🔬 rust-results.json      - Criterion benchmark data"
    echo "  ⚖️  comparison-results.json - Rust vs C++ comparison"
    echo "  💻 system-info.json       - System and environment info"
}

# Run main function with all arguments
main "$@"
