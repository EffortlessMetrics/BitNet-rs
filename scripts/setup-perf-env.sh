#!/bin/bash
# Setup script for performance tracking environment
# This script ensures all dependencies and fixtures are available for benchmarking

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

# Check if we're in the BitNet.rs root directory
check_project_root() {
    if [[ ! -f "Cargo.toml" ]] || ! grep -q "bitnet" Cargo.toml; then
        log_error "This script must be run from the BitNet.rs project root directory"
        exit 1
    fi
    log_info "Running from BitNet.rs project root: $(pwd)"
}

# Setup Rust toolchain and targets
setup_rust() {
    log_info "Setting up Rust toolchain..."

    # Ensure we have the correct Rust version (MSRV 1.90.0)
    if ! rustc --version | grep -q "1.9[0-9]"; then
        log_warning "Rust version may not match MSRV 1.90.0. Current version:"
        rustc --version
    fi

    # Add required targets for cross-compilation
    if [[ "${SETUP_CROSS:-false}" == "true" ]]; then
        log_info "Adding cross-compilation targets..."
        rustup target add aarch64-unknown-linux-gnu || true
        rustup target add x86_64-apple-darwin || true
        rustup target add aarch64-apple-darwin || true
    fi

    log_success "Rust toolchain ready"
}

# Generate test fixtures using xtask
generate_fixtures() {
    log_info "Generating test fixtures for benchmarking..."

    # Clean existing fixtures
    rm -rf crossval/fixtures/

    # Generate fixtures using xtask
    log_info "Running: cargo run -p xtask -- gen-fixtures --size small --output crossval/fixtures/"
    cargo run -p xtask -- gen-fixtures --size small --output crossval/fixtures/

    # Verify fixtures were created
    if [[ ! -d "crossval/fixtures" ]] || [[ -z "$(ls -A crossval/fixtures/)" ]]; then
        log_error "Failed to generate test fixtures"
        return 1
    fi

    log_success "Test fixtures generated in crossval/fixtures/"
    ls -la crossval/fixtures/
}

# Setup BitNet.cpp for cross-validation (optional)
setup_cpp_crossval() {
    if [[ "${SKIP_CPP:-false}" == "true" ]]; then
        log_info "Skipping C++ setup (SKIP_CPP=true)"
        return 0
    fi

    log_info "Setting up BitNet.cpp for cross-validation..."

    # Check if we already have BitNet.cpp
    local cpp_dir="${BITNET_CPP_DIR:-$HOME/.cache/bitnet_cpp}"
    if [[ -f "$cpp_dir/build/bin/llama-cli" ]]; then
        log_success "BitNet.cpp already available at $cpp_dir"
        export BITNET_CPP_ROOT="$cpp_dir"
        return 0
    fi

    # Use cache script if available
    if [[ -f "ci/use-bitnet-cpp-cache.sh" ]]; then
        log_info "Using BitNet.cpp cache script..."
        chmod +x ci/use-bitnet-cpp-cache.sh
        ./ci/use-bitnet-cpp-cache.sh || log_warning "Cache script failed, continuing without C++ cross-validation"
    else
        log_info "Downloading and building BitNet.cpp..."
        cargo run -p xtask -- fetch-cpp || log_warning "Failed to fetch C++, continuing without cross-validation"
    fi

    # Set environment variable if successful
    if [[ -f "$cpp_dir/build/bin/llama-cli" ]]; then
        export BITNET_CPP_ROOT="$cpp_dir"
        log_success "BitNet.cpp ready for cross-validation"
    else
        log_warning "BitNet.cpp not available, cross-validation will be limited to Rust-only"
    fi
}

# Build optimized binaries for benchmarking
build_optimized() {
    log_info "Building optimized binaries for benchmarking..."

    # Features to use for building
    local features="${BENCHMARK_FEATURES:-cpu}"
    local target="${BENCHMARK_TARGET:-}"
    local use_cross="${USE_CROSS:-false}"

    log_info "Building with features: $features"
    if [[ -n "$target" ]]; then
        log_info "Target: $target"
    fi

    # Set optimization flags
    export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native -C opt-level=3"

    # Build the project
    local build_cmd=("cargo")
    if [[ "$use_cross" == "true" && -n "$target" ]]; then
        build_cmd=("cross")
    fi

    build_cmd+=("build" "--release" "--no-default-features" "--features" "$features")
    if [[ -n "$target" ]]; then
        build_cmd+=("--target" "$target")
    fi

    log_info "Running: ${build_cmd[*]}"
    "${build_cmd[@]}"

    # Also build the CLI tools we'll need
    build_cmd=("cargo")
    if [[ "$use_cross" == "true" && -n "$target" ]]; then
        build_cmd=("cross")
    fi

    build_cmd+=("build" "--release" "-p" "bitnet-cli" "--no-default-features" "--features" "$features")
    if [[ -n "$target" ]]; then
        build_cmd+=("--target" "$target")
    fi

    log_info "Building CLI: ${build_cmd[*]}"
    "${build_cmd[@]}"

    log_success "Optimized binaries built successfully"
}

# Create benchmark results directory
setup_results_dir() {
    log_info "Setting up benchmark results directory..."

    local results_dir="benchmark-results"
    rm -rf "$results_dir"
    mkdir -p "$results_dir"

    # Create system info
    cat > "$results_dir/system-info.json" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "platform": "$(uname -m)",
  "os": "$(uname -s)",
  "kernel": "$(uname -r)",
  "cpu_count": $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "unknown"),
  "memory_gb": $(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo "unknown"),
  "rust_version": "$(rustc --version)",
  "cargo_version": "$(cargo --version)",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo "unknown")",
  "git_branch": "$(git branch --show-current 2>/dev/null || echo "unknown")",
  "environment": {
    "RUSTFLAGS": "${RUSTFLAGS:-}",
    "BITNET_CPP_ROOT": "${BITNET_CPP_ROOT:-}",
    "RAYON_NUM_THREADS": "${RAYON_NUM_THREADS:-}",
    "OMP_NUM_THREADS": "${OMP_NUM_THREADS:-}"
  }
}
EOF

    log_success "Results directory created: $results_dir"
}

# Verify the environment is ready for benchmarking
verify_environment() {
    log_info "Verifying benchmark environment..."

    local errors=0

    # Check that fixtures exist
    if [[ ! -d "crossval/fixtures" ]] || [[ -z "$(ls -A crossval/fixtures/)" ]]; then
        log_error "Test fixtures not found"
        ((errors++))
    else
        log_success "Test fixtures: $(ls crossval/fixtures/ | wc -l) files"
    fi

    # Check that binaries exist
    local target_dir="target"
    if [[ -n "${BENCHMARK_TARGET:-}" ]]; then
        target_dir="target/${BENCHMARK_TARGET}"
    fi

    if [[ ! -f "$target_dir/release/bitnet-cli" ]]; then
        log_error "bitnet-cli binary not found at $target_dir/release/"
        ((errors++))
    else
        log_success "bitnet-cli binary available"
    fi

    # Check cross-validation setup
    if [[ -n "${BITNET_CPP_ROOT:-}" ]]; then
        if [[ -f "$BITNET_CPP_ROOT/build/bin/llama-cli" ]]; then
            log_success "C++ cross-validation available"
        else
            log_warning "BITNET_CPP_ROOT set but binary not found"
        fi
    else
        log_info "C++ cross-validation not configured (Rust-only benchmarks)"
    fi

    # Check benchmark comparison script
    if [[ -f "benchmark_comparison.py" ]]; then
        log_success "Benchmark comparison script available"
    else
        log_warning "benchmark_comparison.py not found"
    fi

    if [[ $errors -gt 0 ]]; then
        log_error "Environment verification failed with $errors errors"
        return 1
    fi

    log_success "Environment verification complete - ready for benchmarking!"
}

# Print usage information
usage() {
    cat << EOF
BitNet.rs Performance Environment Setup

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    --skip-cpp              Skip C++ setup for cross-validation
    --cross                 Setup cross-compilation targets
    --features FEATURES     Specify features for building (default: cpu)
    --target TARGET         Specify target for cross-compilation
    --use-cross             Use 'cross' instead of 'cargo' for building

ENVIRONMENT VARIABLES:
    SKIP_CPP=true          Skip C++ setup
    SETUP_CROSS=true       Add cross-compilation targets
    BENCHMARK_FEATURES     Features to use for building (default: cpu)
    BENCHMARK_TARGET       Target for cross-compilation
    USE_CROSS=true         Use 'cross' instead of 'cargo'
    BITNET_CPP_DIR         Directory for BitNet.cpp (default: ~/.cache/bitnet_cpp)

EXAMPLES:
    # Basic setup for CPU benchmarking
    $0

    # Setup with GPU features
    $0 --features gpu

    # Cross-compilation setup for Linux ARM64
    $0 --cross --target aarch64-unknown-linux-gnu --use-cross

    # Skip C++ cross-validation
    $0 --skip-cpp
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
            --skip-cpp)
                export SKIP_CPP=true
                shift
                ;;
            --cross)
                export SETUP_CROSS=true
                shift
                ;;
            --features)
                export BENCHMARK_FEATURES="$2"
                shift 2
                ;;
            --target)
                export BENCHMARK_TARGET="$2"
                shift 2
                ;;
            --use-cross)
                export USE_CROSS=true
                shift
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
    log_info "BitNet.rs Performance Environment Setup"
    log_info "======================================="

    parse_args "$@"

    check_project_root
    setup_rust
    generate_fixtures
    setup_cpp_crossval
    build_optimized
    setup_results_dir
    verify_environment

    log_success "Performance environment setup complete!"
    echo
    log_info "Next steps:"
    echo "  1. Run benchmarks: cargo bench --workspace --no-default-features --features ${BENCHMARK_FEATURES:-cpu}"
    echo "  2. Run comparison: python3 benchmark_comparison.py"
    echo "  3. Review results in benchmark-results/"
}

# Run main function with all arguments
main "$@"
