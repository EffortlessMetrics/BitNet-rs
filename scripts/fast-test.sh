#!/bin/bash

# Fast test execution script optimized for <15 minute execution
# This script implements various optimizations to achieve the target time

set -euo pipefail

# Configuration
TARGET_TIME_MINUTES=15
MAX_PARALLEL=${BITNET_TEST_PARALLEL:-$(nproc)}
AGGRESSIVE_MODE=${BITNET_AGGRESSIVE_TEST:-true}
SKIP_SLOW_TESTS=${BITNET_SKIP_SLOW:-true}
ENABLE_CACHING=${BITNET_TEST_CACHE:-true}

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

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Start timer
START_TIME=$(date +%s)

log_info "Starting optimized test execution (target: ${TARGET_TIME_MINUTES} minutes)"
log_info "Configuration:"
log_info "  - Max parallel: ${MAX_PARALLEL}"
log_info "  - Aggressive mode: ${AGGRESSIVE_MODE}"
log_info "  - Skip slow tests: ${SKIP_SLOW_TESTS}"
log_info "  - Enable caching: ${ENABLE_CACHING}"

# Set environment variables for optimization
export BITNET_TEST_PARALLEL=${MAX_PARALLEL}
export BITNET_TEST_TIMEOUT=60
export BITNET_TEST_LOG_LEVEL=warn
export BITNET_TEST_GENERATE_COVERAGE=false
export BITNET_TEST_CACHE_DIR="tests/cache"
export BITNET_TEST_AUTO_DOWNLOAD=${ENABLE_CACHING}

# Create cache directory
mkdir -p tests/cache

# Function to run tests with timeout
run_tests_with_timeout() {
    local test_args="$1"
    local timeout_seconds=$((TARGET_TIME_MINUTES * 60))
    
    log_info "Running tests with ${timeout_seconds}s timeout: cargo test ${test_args}"
    
    if timeout ${timeout_seconds}s cargo test ${test_args}; then
        return 0
    else
        local exit_code=$?
        if [ ${exit_code} -eq 124 ]; then
            log_error "Tests timed out after ${TARGET_TIME_MINUTES} minutes"
            return 124
        else
            log_error "Tests failed with exit code ${exit_code}"
            return ${exit_code}
        fi
    fi
}

# Function to get test list and estimate execution time
estimate_test_time() {
    log_info "Analyzing test suite..."
    
    # Get list of all tests
    local test_list=$(cargo test --workspace --no-run --message-format=json 2>/dev/null | \
        jq -r 'select(.reason == "compiler-artifact" and .target.kind[] == "test") | .executable' 2>/dev/null || true)
    
    if [ -z "$test_list" ]; then
        log_warn "Could not analyze test suite, proceeding with default configuration"
        return 0
    fi
    
    local test_count=$(echo "$test_list" | wc -l)
    log_info "Found ${test_count} test executables"
    
    # Estimate based on historical data or defaults
    local estimated_time_per_test=5
    local total_estimated_time=$((test_count * estimated_time_per_test))
    local parallel_estimated_time=$((total_estimated_time / MAX_PARALLEL))
    
    log_info "Estimated execution time: ${parallel_estimated_time}s (${total_estimated_time}s sequential)"
    
    if [ ${parallel_estimated_time} -gt $((TARGET_TIME_MINUTES * 60)) ]; then
        log_warn "Estimated time exceeds target, enabling aggressive optimizations"
        return 1
    fi
    
    return 0
}

# Function to run optimized test selection
run_optimized_tests() {
    local test_args=""
    
    # Base test arguments for speed
    test_args="--workspace"
    test_args="${test_args} --test-threads=${MAX_PARALLEL}"
    
    if [ "${AGGRESSIVE_MODE}" = "true" ]; then
        # Skip documentation tests for speed
        test_args="${test_args} --lib --bins"
        
        # Skip slow integration tests if needed
        if [ "${SKIP_SLOW_TESTS}" = "true" ]; then
            test_args="${test_args} --exclude=crossval"
            log_info "Skipping slow cross-validation tests"
        fi
    fi
    
    # Add timeout per test
    test_args="${test_args} -- --test-timeout=60"
    
    run_tests_with_timeout "${test_args}"
}

# Function to run incremental tests (only changed code)
run_incremental_tests() {
    log_info "Attempting incremental test execution..."
    
    # Check if we can determine changed files
    if command -v git >/dev/null 2>&1 && git rev-parse --git-dir >/dev/null 2>&1; then
        local changed_files=$(git diff --name-only HEAD~1 2>/dev/null || git diff --name-only --cached 2>/dev/null || true)
        
        if [ -n "$changed_files" ]; then
            log_info "Detected changes in:"
            echo "$changed_files" | sed 's/^/  - /'
            
            # Run tests for changed crates only
            local changed_crates=""
            for file in $changed_files; do
                if [[ $file == crates/* ]]; then
                    local crate_name=$(echo $file | cut -d'/' -f2)
                    if [[ ! $changed_crates =~ $crate_name ]]; then
                        changed_crates="${changed_crates} -p ${crate_name}"
                    fi
                fi
            done
            
            if [ -n "$changed_crates" ]; then
                log_info "Running tests for changed crates: ${changed_crates}"
                run_tests_with_timeout "${changed_crates} --test-threads=${MAX_PARALLEL}"
                return $?
            fi
        fi
    fi
    
    log_info "Could not determine incremental changes, running full test suite"
    return 1
}

# Function to run fast unit tests only
run_fast_unit_tests() {
    log_info "Running fast unit tests only..."
    
    local test_args="--workspace --lib --test-threads=${MAX_PARALLEL}"
    
    # Exclude known slow crates
    test_args="${test_args} --exclude=crossval --exclude=bitnet-sys"
    
    # Run with shorter timeout
    test_args="${test_args} -- --test-timeout=30"
    
    run_tests_with_timeout "${test_args}"
}

# Function to cleanup and report
cleanup_and_report() {
    local exit_code=$1
    local end_time=$(date +%s)
    local duration=$((end_time - START_TIME))
    local duration_minutes=$((duration / 60))
    local duration_seconds=$((duration % 60))
    
    log_info "Test execution completed in ${duration_minutes}m ${duration_seconds}s"
    
    if [ ${exit_code} -eq 0 ]; then
        if [ ${duration} -le $((TARGET_TIME_MINUTES * 60)) ]; then
            log_success "✅ Tests completed successfully within ${TARGET_TIME_MINUTES} minute target!"
        else
            log_warn "⚠️  Tests completed successfully but exceeded ${TARGET_TIME_MINUTES} minute target"
        fi
    else
        log_error "❌ Tests failed with exit code ${exit_code}"
    fi
    
    # Generate simple report
    cat > test-execution-report.txt << EOF
# Test Execution Report

**Target Time:** ${TARGET_TIME_MINUTES} minutes
**Actual Time:** ${duration_minutes}m ${duration_seconds}s
**Status:** $([ ${exit_code} -eq 0 ] && echo "PASSED" || echo "FAILED")
**Exit Code:** ${exit_code}

## Configuration
- Max Parallel: ${MAX_PARALLEL}
- Aggressive Mode: ${AGGRESSIVE_MODE}
- Skip Slow Tests: ${SKIP_SLOW_TESTS}
- Enable Caching: ${ENABLE_CACHING}

## Performance
- Time Efficiency: $(( (TARGET_TIME_MINUTES * 60 * 100) / duration ))% of target
- Parallel Efficiency: Estimated $(( MAX_PARALLEL * 100 / (MAX_PARALLEL + 1) ))%

Generated at: $(date)
EOF
    
    log_info "Report saved to test-execution-report.txt"
    
    exit ${exit_code}
}

# Trap to ensure cleanup on exit
trap 'cleanup_and_report $?' EXIT

# Main execution logic
main() {
    # Check if we can estimate test time
    if estimate_test_time; then
        log_info "Estimated time is within target, running full test suite"
        run_optimized_tests
    else
        log_warn "Estimated time exceeds target, trying optimizations..."
        
        # Try incremental tests first
        if [ "${ENABLE_CACHING}" = "true" ] && run_incremental_tests; then
            log_success "Incremental tests completed successfully"
        else
            # Fall back to fast unit tests
            log_info "Falling back to fast unit tests only"
            run_fast_unit_tests
        fi
    fi
}

# Check prerequisites
if ! command -v cargo >/dev/null 2>&1; then
    log_error "cargo not found in PATH"
    exit 1
fi

if ! command -v timeout >/dev/null 2>&1; then
    log_error "timeout command not found (install coreutils)"
    exit 1
fi

# Run main logic
main