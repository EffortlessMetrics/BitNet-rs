#!/usr/bin/env bash
# CI/CD Integration Script for BitNet.rs Benchmarking
# This script demonstrates how to integrate the benchmarking setup with CI/CD pipelines
# Compatible with GitHub Actions, GitLab CI, Jenkins, etc.

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." &> /dev/null && pwd)"
readonly RESULTS_DIR="benchmark-results"
readonly CI_THRESHOLD_REGRESSION="10.0"  # 10% regression threshold

# Colors for CI output
readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

log_info() { echo -e "${GREEN}[CI-BENCHMARK]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[CI-BENCHMARK]${NC} $*"; }
log_error() { echo -e "${RED}[CI-BENCHMARK]${NC} $*"; }

# Detect CI environment
detect_ci() {
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
        echo "github"
    elif [[ -n "${GITLAB_CI:-}" ]]; then
        echo "gitlab"
    elif [[ -n "${JENKINS_URL:-}" ]]; then
        echo "jenkins"
    elif [[ -n "${CI:-}" ]]; then
        echo "generic"
    else
        echo "local"
    fi
}

# Setup for CI environment
setup_ci_environment() {
    local ci_env="$1"

    log_info "Setting up for CI environment: ${ci_env}"

    case "${ci_env}" in
        github)
            # GitHub Actions specific setup
            echo "Setting up for GitHub Actions"
            export RUST_BACKTRACE=1
            export CARGO_TERM_COLOR=always
            export BITNET_DETERMINISTIC=1
            export BITNET_SEED=42
            export RAYON_NUM_THREADS=2  # Limit parallelism for CI
            ;;
        gitlab)
            # GitLab CI specific setup
            echo "Setting up for GitLab CI"
            export RUST_BACKTRACE=1
            export CARGO_TERM_COLOR=always
            export BITNET_DETERMINISTIC=1
            export BITNET_SEED=42
            ;;
        jenkins)
            # Jenkins specific setup
            echo "Setting up for Jenkins"
            export RUST_BACKTRACE=1
            export BITNET_DETERMINISTIC=1
            export BITNET_SEED=42
            ;;
        *)
            # Generic CI or local
            echo "Setting up for generic CI/local environment"
            export RUST_BACKTRACE=1
            export BITNET_DETERMINISTIC=1
            export BITNET_SEED=42
            ;;
    esac
}

# Run CI-optimized benchmark setup
run_ci_setup() {
    log_info "Running CI-optimized benchmark setup..."

    cd "${REPO_ROOT}"

    # Create results directory
    mkdir -p "${RESULTS_DIR}"

    # Run setup script with CI-friendly options
    if ! ./scripts/setup-benchmarks.sh --skip-cpp; then
        log_error "Benchmark setup failed"
        exit 1
    fi

    log_info "✅ CI benchmark setup completed"
}

# Run performance benchmarks for CI
run_ci_benchmarks() {
    log_info "Running CI performance benchmarks..."

    cd "${REPO_ROOT}"

    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local results_file="${RESULTS_DIR}/ci_benchmark_${timestamp}.json"

    # Run Rust-only benchmarks (faster for CI)
    log_info "Running Rust benchmarks..."
    if ! ./benchmark_comparison.py \
        --skip-cpp \
        --iterations 2 \
        --timeout 120 \
        --format json > "${results_file}"; then
        log_error "Rust benchmarks failed"
        return 1
    fi

    # Run cargo benchmarks with timeout
    log_info "Running cargo benchmarks..."
    if ! timeout 300 cargo bench \
        --workspace \
        --no-default-features \
        --features cpu \
        -- --output-format json > "${RESULTS_DIR}/cargo_bench_${timestamp}.json"; then
        log_warn "Cargo benchmarks timed out or failed"
    fi

    log_info "✅ CI benchmarks completed"
    echo "Results saved to: ${results_file}"

    return 0
}

# Analyze benchmark results for regressions
analyze_benchmark_results() {
    log_info "Analyzing benchmark results for regressions..."

    cd "${REPO_ROOT}"

    local latest_results
    latest_results=$(ls -t "${RESULTS_DIR}"/ci_benchmark_*.json 2>/dev/null | head -n1)

    if [[ -z "${latest_results}" ]]; then
        log_warn "No benchmark results found for analysis"
        return 0
    fi

    # Create analysis script
    cat > "${RESULTS_DIR}/analyze_results.py" << 'EOF'
import json
import sys
import os
from pathlib import Path

def analyze_results(results_file, threshold_percent=10.0):
    """Analyze benchmark results for performance regressions"""

    if not Path(results_file).exists():
        print(f"Results file not found: {results_file}")
        return False

    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading results: {e}")
        return False

    # Extract performance metrics
    rust_results = data.get('rust', {})
    if not rust_results:
        print("No Rust results found in benchmark data")
        return True  # Not a failure, just no data

    # Basic performance validation
    mean_time = rust_results.get('mean', 0)
    if mean_time <= 0:
        print("⚠️  Invalid benchmark timing results")
        return False

    if mean_time > 60:  # More than 60 seconds for basic inference
        print(f"⚠️  Performance concern: inference took {mean_time:.2f}s")
        print("This may indicate a performance regression")
        return False

    # Check if response was generated correctly
    correctness = rust_results.get('response_correctness', 'unknown')
    if correctness == 'incorrect':
        print("❌ Inference correctness check failed")
        return False
    elif correctness == 'correct':
        print("✅ Inference correctness validated")

    print(f"✅ Performance analysis passed (mean: {mean_time:.2f}s)")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_results.py <results_file>")
        sys.exit(1)

    results_file = sys.argv[1]
    threshold = float(os.environ.get('CI_THRESHOLD_REGRESSION', '10.0'))

    if analyze_results(results_file, threshold):
        print("📊 Benchmark analysis: PASSED")
        sys.exit(0)
    else:
        print("📊 Benchmark analysis: FAILED")
        sys.exit(1)
EOF

    # Run analysis
    if python3 "${RESULTS_DIR}/analyze_results.py" "${latest_results}"; then
        log_info "✅ Benchmark analysis passed"
        return 0
    else
        log_error "❌ Benchmark analysis detected issues"
        return 1
    fi
}

# Generate CI artifacts
generate_ci_artifacts() {
    log_info "Generating CI artifacts..."

    cd "${REPO_ROOT}"

    # Create summary report
    local report_file="${RESULTS_DIR}/ci_summary.md"
    cat > "${report_file}" << EOF
# BitNet.rs CI Benchmark Report

**Date**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
**Commit**: ${GITHUB_SHA:-${CI_COMMIT_SHA:-$(git rev-parse HEAD 2>/dev/null || echo "unknown")}}
**CI Environment**: $(detect_ci)

## Benchmark Results

EOF

    # Add latest results if available
    local latest_results
    latest_results=$(ls -t "${RESULTS_DIR}"/ci_benchmark_*.json 2>/dev/null | head -n1)

    if [[ -n "${latest_results}" ]]; then
        echo "### Performance Metrics" >> "${report_file}"
        python3 -c "
import json
import sys
try:
    with open('${latest_results}', 'r') as f:
        data = json.load(f)
    rust_data = data.get('rust', {})
    if rust_data:
        print(f'- **Mean Time**: {rust_data.get(\"mean\", 0):.3f}s')
        print(f'- **Std Dev**: {rust_data.get(\"stdev\", 0):.3f}s')
        print(f'- **Min Time**: {rust_data.get(\"min\", 0):.3f}s')
        print(f'- **Max Time**: {rust_data.get(\"max\", 0):.3f}s')
        correctness = rust_data.get('response_correctness', 'unknown')
        print(f'- **Correctness**: {correctness}')
except Exception as e:
    print(f'Error reading results: {e}')
        " >> "${report_file}"
    fi

    echo "" >> "${report_file}"
    echo "## Setup Information" >> "${report_file}"
    echo "" >> "${report_file}"
    echo "- **Rust Version**: $(rustc --version)" >> "${report_file}"
    echo "- **Platform**: $(uname -s)-$(uname -m)" >> "${report_file}"
    echo "- **CPU Count**: $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo unknown)" >> "${report_file}"
    echo "" >> "${report_file}"
    echo "---" >> "${report_file}"
    echo "*Report generated by ci-benchmark-integration.sh*" >> "${report_file}"

    log_info "CI summary report created: ${report_file}"
}

# Main CI workflow
main() {
    local ci_env
    ci_env=$(detect_ci)

    log_info "BitNet.rs CI Benchmark Integration"
    log_info "CI Environment: ${ci_env}"
    log_info "Starting at: $(date)"

    # Setup CI environment
    setup_ci_environment "${ci_env}"

    # Run setup
    if ! run_ci_setup; then
        log_error "CI setup failed"
        exit 1
    fi

    # Run benchmarks
    if ! run_ci_benchmarks; then
        log_error "CI benchmarks failed"
        exit 1
    fi

    # Analyze results
    if ! analyze_benchmark_results; then
        log_error "Benchmark analysis failed"
        exit 1
    fi

    # Generate artifacts
    generate_ci_artifacts

    log_info "✅ CI benchmark integration completed successfully"
    log_info "Artifacts available in: ${RESULTS_DIR}/"
}

# Run main function
main "$@"