#!/bin/bash
# Example test scripts for BitNet.rs testing framework
# This demonstrates various testing scenarios and automation scripts

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TEST_RESULTS_DIR="${PROJECT_ROOT}/test-results"
TEST_CACHE_DIR="${PROJECT_ROOT}/test-cache"
CONFIG_FILE="${PROJECT_ROOT}/test-config.toml"

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

# Setup test environment
setup_test_environment() {
    log_info "Setting up test environment..."

    # Create directories
    mkdir -p "$TEST_RESULTS_DIR" "$TEST_CACHE_DIR"

    # Set environment variables
    export BITNET_TEST_CACHE="$TEST_CACHE_DIR"
    export BITNET_LOG_LEVEL="${BITNET_LOG_LEVEL:-info}"
    export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"
    export CARGO_TERM_COLOR="${CARGO_TERM_COLOR:-always}"

    # Check if config file exists
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_warning "Config file not found at $CONFIG_FILE, using defaults"
        CONFIG_FILE=""
    fi

    log_success "Test environment setup complete"
}

# Clean test environment
cleanup_test_environment() {
    log_info "Cleaning up test environment..."

    # Clean old test results (keep last 5 runs)
    if [[ -d "$TEST_RESULTS_DIR" ]]; then
        find "$TEST_RESULTS_DIR" -type d -name "run-*" | sort -r | tail -n +6 | xargs rm -rf
    fi

    # Clean old cache files (older than 7 days)
    if [[ -d "$TEST_CACHE_DIR" ]]; then
        find "$TEST_CACHE_DIR" -type f -mtime +7 -delete
    fi

    log_success "Cleanup complete"
}

# Run unit tests
run_unit_tests() {
    log_info "Running unit tests..."

    local test_args=("--lib" "--all-features" "--workspace")
    local output_file="$TEST_RESULTS_DIR/unit-tests.json"

    if [[ -n "$CONFIG_FILE" ]]; then
        test_args+=("--config" "$CONFIG_FILE")
    fi

    # Run tests with coverage if tarpaulin is available
    if command -v cargo-tarpaulin >/dev/null 2>&1; then
        log_info "Running unit tests with coverage..."
        cargo tarpaulin \
            --verbose \
            --all-features \
            --workspace \
            --timeout 120 \
            --exclude-files "examples/*" "tests/*" \
            --out json \
            --output-dir "$TEST_RESULTS_DIR" \
            || { log_error "Unit tests with coverage failed"; return 1; }
    else
        log_info "Running unit tests without coverage (tarpaulin not available)..."
        cargo test "${test_args[@]}" \
            --format json > "$output_file" \
            || { log_error "Unit tests failed"; return 1; }
    fi

    log_success "Unit tests completed"
}

# Run integration tests
run_integration_tests() {
    log_info "Running integration tests..."

    local output_file="$TEST_RESULTS_DIR/integration-tests.json"

    cargo test \
        --test integration_tests \
        --test workflow_integration_tests \
        --test component_interaction_tests \
        --features integration-tests \
        --format json > "$output_file" \
        || { log_error "Integration tests failed"; return 1; }

    log_success "Integration tests completed"
}

# Run cross-validation tests
run_cross_validation_tests() {
    log_info "Running cross-validation tests..."

    # Check if C++ BitNet binary is available
    local cpp_binary="${BITNET_CPP_BINARY:-bitnet}"
    if ! command -v "$cpp_binary" >/dev/null 2>&1; then
        log_warning "C++ BitNet binary not found, skipping cross-validation tests"
        return 0
    fi

    local output_file="$TEST_RESULTS_DIR/cross-validation-tests.json"

    BITNET_CPP_BINARY="$cpp_binary" cargo test \
        --test cross_validation_tests \
        --features cross-validation \
        --format json > "$output_file" \
        || { log_error "Cross-validation tests failed"; return 1; }

    log_success "Cross-validation tests completed"
}

# Run performance benchmarks
run_performance_benchmarks() {
    log_info "Running performance benchmarks..."

    if ! command -v cargo-criterion >/dev/null 2>&1; then
        log_warning "cargo-criterion not available, skipping performance benchmarks"
        return 0
    fi

    local output_file="$TEST_RESULTS_DIR/benchmarks.json"

    cargo criterion \
        --bench inference \
        --bench kernels \
        --message-format json > "$output_file" \
        || { log_error "Performance benchmarks failed"; return 1; }

    log_success "Performance benchmarks completed"
}

# Generate test report
generate_test_report() {
    log_info "Generating test report..."

    local report_script="$PROJECT_ROOT/scripts/generate_test_report.py"
    if [[ ! -f "$report_script" ]]; then
        log_warning "Test report script not found, skipping report generation"
        return 0
    fi

    python3 "$report_script" \
        --input-dir "$TEST_RESULTS_DIR" \
        --output-dir "$TEST_RESULTS_DIR/report" \
        --format html \
        || { log_error "Test report generation failed"; return 1; }

    log_success "Test report generated at $TEST_RESULTS_DIR/report/index.html"
}

# Run specific test category
run_test_category() {
    local category="$1"

    case "$category" in
        "unit")
            run_unit_tests
            ;;
        "integration")
            run_integration_tests
            ;;
        "crossval"|"cross-validation")
            run_cross_validation_tests
            ;;
        "performance"|"bench")
            run_performance_benchmarks
            ;;
        "all")
            run_unit_tests
            run_integration_tests
            run_cross_validation_tests
            run_performance_benchmarks
            ;;
        *)
            log_error "Unknown test category: $category"
            log_info "Available categories: unit, integration, crossval, performance, all"
            return 1
            ;;
    esac
}

# Run comprehensive test suite
run_comprehensive_tests() {
    log_info "Running comprehensive test suite..."

    local start_time=$(date +%s)
    local run_id="run-$(date +%Y%m%d-%H%M%S)"
    local run_dir="$TEST_RESULTS_DIR/$run_id"

    # Create run directory
    mkdir -p "$run_dir"
    export TEST_RESULTS_DIR="$run_dir"

    # Run all test categories
    local failed_tests=()

    if ! run_unit_tests; then
        failed_tests+=("unit")
    fi

    if ! run_integration_tests; then
        failed_tests+=("integration")
    fi

    if ! run_cross_validation_tests; then
        failed_tests+=("cross-validation")
    fi

    if ! run_performance_benchmarks; then
        failed_tests+=("performance")
    fi

    # Generate report
    generate_test_report

    # Calculate duration
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # Summary
    log_info "Test suite completed in ${duration}s"

    if [[ ${#failed_tests[@]} -eq 0 ]]; then
        log_success "All tests passed!"
        return 0
    else
        log_error "Failed test categories: ${failed_tests[*]}"
        return 1
    fi
}

# Check test dependencies
check_dependencies() {
    log_info "Checking test dependencies..."

    local missing_deps=()

    # Check Rust toolchain
    if ! command -v cargo >/dev/null 2>&1; then
        missing_deps+=("cargo")
    fi

    # Check optional tools
    local optional_tools=("cargo-tarpaulin" "cargo-nextest" "cargo-criterion")
    for tool in "${optional_tools[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            log_warning "Optional tool not found: $tool"
        fi
    done

    # Check Python for reporting
    if ! command -v python3 >/dev/null 2>&1; then
        log_warning "Python3 not found, test reporting may not work"
    fi

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing required dependencies: ${missing_deps[*]}"
        return 1
    fi

    log_success "All required dependencies found"
}

# Install test dependencies
install_dependencies() {
    log_info "Installing test dependencies..."

    # Install Rust testing tools
    cargo install cargo-tarpaulin cargo-nextest cargo-criterion --locked

    # Install Python dependencies
    if command -v pip3 >/dev/null 2>&1; then
        pip3 install jinja2 matplotlib seaborn pandas
    fi

    log_success "Dependencies installed"
}

# Watch mode for continuous testing
watch_tests() {
    log_info "Starting watch mode for continuous testing..."

    if ! command -v cargo-watch >/dev/null 2>&1; then
        log_error "cargo-watch not found. Install with: cargo install cargo-watch"
        return 1
    fi

    cargo watch -x "test --lib" -x "test --test integration_tests"
}

# Docker-based testing
run_docker_tests() {
    log_info "Running tests in Docker container..."

    local dockerfile="$PROJECT_ROOT/examples/testing/ci_cd/docker_testing_example.dockerfile"
    if [[ ! -f "$dockerfile" ]]; then
        log_error "Docker file not found: $dockerfile"
        return 1
    fi

    # Build test image
    docker build -f "$dockerfile" -t bitnet-rs-test "$PROJECT_ROOT"

    # Run tests in container
    docker run --rm \
        -v "$TEST_RESULTS_DIR:/app/test-results" \
        -v "$TEST_CACHE_DIR:/app/test-cache" \
        bitnet-rs-test

    log_success "Docker tests completed"
}

# Memory profiling
run_memory_profiling() {
    log_info "Running memory profiling tests..."

    if ! command -v valgrind >/dev/null 2>&1; then
        log_warning "Valgrind not found, skipping memory profiling"
        return 0
    fi

    # Run tests under valgrind
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="valgrind --tool=memcheck --leak-check=full" \
        cargo test --target x86_64-unknown-linux-gnu

    log_success "Memory profiling completed"
}

# Security audit
run_security_audit() {
    log_info "Running security audit..."

    if ! command -v cargo-audit >/dev/null 2>&1; then
        log_warning "cargo-audit not found, installing..."
        cargo install cargo-audit --locked
    fi

    cargo audit

    log_success "Security audit completed"
}

# Usage information
show_usage() {
    cat << EOF
BitNet.rs Testing Framework Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    setup                   Setup test environment
    cleanup                 Clean test environment
    check-deps             Check test dependencies
    install-deps           Install test dependencies

    unit                   Run unit tests
    integration            Run integration tests
    crossval               Run cross-validation tests
    performance            Run performance benchmarks
    all                    Run all test categories

    comprehensive          Run comprehensive test suite
    report                 Generate test report
    watch                  Run tests in watch mode
    docker                 Run tests in Docker

    memory-profile         Run memory profiling tests
    security-audit         Run security audit

Options:
    --config FILE          Use specific config file
    --cache-dir DIR        Use specific cache directory
    --results-dir DIR      Use specific results directory
    --verbose              Enable verbose output
    --help                 Show this help message

Examples:
    $0 setup                           # Setup test environment
    $0 unit                           # Run unit tests only
    $0 comprehensive                  # Run all tests
    $0 --config dev.toml unit         # Run unit tests with dev config
    $0 docker                         # Run tests in Docker

Environment Variables:
    BITNET_TEST_CACHE      Test cache directory
    BITNET_LOG_LEVEL       Log level (debug, info, warn, error)
    BITNET_CPP_BINARY      Path to C++ BitNet binary
    RUST_BACKTRACE         Enable Rust backtraces
EOF
}

# Main function
main() {
    local command="${1:-help}"

    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --cache-dir)
                TEST_CACHE_DIR="$2"
                shift 2
                ;;
            --results-dir)
                TEST_RESULTS_DIR="$2"
                shift 2
                ;;
            --verbose)
                set -x
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            -*)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
            *)
                command="$1"
                shift
                ;;
        esac
    done

    # Execute command
    case "$command" in
        "setup")
            setup_test_environment
            ;;
        "cleanup")
            cleanup_test_environment
            ;;
        "check-deps")
            check_dependencies
            ;;
        "install-deps")
            install_dependencies
            ;;
        "unit"|"integration"|"crossval"|"cross-validation"|"performance"|"bench"|"all")
            setup_test_environment
            run_test_category "$command"
            ;;
        "comprehensive")
            setup_test_environment
            run_comprehensive_tests
            ;;
        "report")
            generate_test_report
            ;;
        "watch")
            watch_tests
            ;;
        "docker")
            run_docker_tests
            ;;
        "memory-profile")
            run_memory_profiling
            ;;
        "security-audit")
            run_security_audit
            ;;
        "help"|"--help"|"-h")
            show_usage
            ;;
        *)
            log_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
