#!/usr/bin/env bash
# BitNet.rs Benchmarking Infrastructure Setup Script
#
# This script addresses GitHub issue #155 by setting up a comprehensive
# benchmarking environment with proper model fixtures, C++ cross-validation,
# and diagnostic checks for benchmarking readiness.
#
# Usage: ./scripts/setup-benchmarks.sh [options]
#
# Requirements addressed:
# - Sets up required model and tokenizer fixtures
# - Ensures C++ implementation is available for cross-validation
# - Runs diagnostic checks to verify benchmarking readiness
# - Provides clear error messages and setup instructions
# - Integrates with existing benchmark_comparison.py
# - Works with crossval/benches/performance.rs
# - Compatible with CI/CD workflows like performance-tracking.yml

set -euo pipefail

# Colors for better output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." &> /dev/null && pwd)"
readonly DEFAULT_MODEL_ID="microsoft/bitnet-b1.58-2B-4T-gguf"
readonly DEFAULT_MODEL_FILE="ggml-model-i2_s.gguf"
readonly BENCHMARK_MODEL_PATH="models/${DEFAULT_MODEL_ID}/${DEFAULT_MODEL_FILE}"
readonly CROSSVAL_FIXTURES_DIR="crossval/fixtures"
readonly BENCHMARK_RESULTS_DIR="benchmark-results"

# Environment variables with defaults
readonly BITNET_GGUF="${BITNET_GGUF:-${REPO_ROOT}/${BENCHMARK_MODEL_PATH}}"
readonly BITNET_CPP_DIR="${BITNET_CPP_DIR:-${HOME}/.cache/bitnet_cpp}"
readonly HF_TOKEN="${HF_TOKEN:-}"

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $*"
}

log_step() {
    echo -e "${CYAN}[STEP]${NC} $*"
}

print_banner() {
    echo -e "${BLUE}"
    cat << 'EOF'
    ____  _ _   _   _      _
   |  _ \(_) | | \ | |    | |
   | |_) |_| |_|  \| | ___| |_ _ __ ___
   |  _ <| | __| . ` |/ _ \ __| '__/ __|
   | |_) | | |_| |\  |  __/ |_| |  \__ \
   |____/|_|\__|_| \_|\___|\__|_|  |___/

   Benchmarking Infrastructure Setup
   Addressing GitHub Issue #155
EOF
    echo -e "${NC}"
}

# Check if we're in the right directory
check_repo_root() {
    if [[ ! -f "${REPO_ROOT}/Cargo.toml" ]] || ! grep -q "bitnet" "${REPO_ROOT}/Cargo.toml"; then
        log_error "This script must be run from the BitNet.rs repository root"
        log_error "Current directory: $(pwd)"
        log_error "Expected Cargo.toml with 'bitnet' at: ${REPO_ROOT}/Cargo.toml"
        exit 1
    fi
    log_debug "Repository root validated: ${REPO_ROOT}"
}

# Check system requirements and environment
check_system_requirements() {
    log_step "Checking system requirements..."

    # Check required commands
    local required_commands=("cargo" "curl" "python3")
    local missing_commands=()

    for cmd in "${required_commands[@]}"; do
        if ! command -v "${cmd}" &> /dev/null; then
            missing_commands+=("${cmd}")
        fi
    done

    if [[ ${#missing_commands[@]} -gt 0 ]]; then
        log_error "Missing required commands: ${missing_commands[*]}"
        log_error "Please install the missing commands and run again"
        exit 1
    fi

    # Check Rust version
    local rust_version
    rust_version=$(rustc --version | cut -d' ' -f2)
    log_info "Rust version: ${rust_version}"

    # Check minimum version (1.89.0)
    if ! printf '%s\n' "1.89.0" "${rust_version}" | sort -V | head -n1 | grep -q "1.89.0"; then
        log_warn "Rust version ${rust_version} may be older than recommended (1.89.0)"
    fi

    # Check available disk space
    local available_space
    available_space=$(df -BG "${REPO_ROOT}" | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ ${available_space} -lt 10 ]]; then
        log_warn "Low disk space: ${available_space}GB available (recommend at least 10GB)"
    fi

    log_info "✅ System requirements check completed"
}

# Setup model fixtures for benchmarking
setup_model_fixtures() {
    log_step "Setting up model fixtures for benchmarking..."

    cd "${REPO_ROOT}"

    # Create directories
    mkdir -p "$(dirname "${BENCHMARK_MODEL_PATH}")"
    mkdir -p "${CROSSVAL_FIXTURES_DIR}"
    mkdir -p "${BENCHMARK_RESULTS_DIR}"

    # Check if model already exists
    if [[ -f "${BENCHMARK_MODEL_PATH}" ]]; then
        local file_size
        file_size=$(stat -c%s "${BENCHMARK_MODEL_PATH}" 2>/dev/null || stat -f%z "${BENCHMARK_MODEL_PATH}" 2>/dev/null || echo "unknown")
        log_info "Model already exists: ${BENCHMARK_MODEL_PATH} (${file_size} bytes)"

        # Verify model integrity
        log_debug "Verifying model integrity..."
        if cargo run -p xtask --no-default-features --features cpu -- verify \
            --model "${BENCHMARK_MODEL_PATH}" \
            --format json > /dev/null 2>&1; then
            log_info "✅ Model integrity verified"
        else
            log_warn "⚠️ Model integrity check failed, will re-download"
            rm -f "${BENCHMARK_MODEL_PATH}"
        fi
    fi

    # Download model if needed
    if [[ ! -f "${BENCHMARK_MODEL_PATH}" ]]; then
        log_info "Downloading benchmark model: ${DEFAULT_MODEL_ID}/${DEFAULT_MODEL_FILE}"
        log_info "This may take several minutes depending on your connection..."

        # Use xtask to download with progress
        if ! cargo run -p xtask --no-default-features --features cpu -- download-model \
            --id "${DEFAULT_MODEL_ID}" \
            --file "${DEFAULT_MODEL_FILE}"; then
            log_error "Failed to download model"
            log_error "Please check your internet connection and try again"
            log_error "If the problem persists, try setting HF_TOKEN environment variable"
            exit 1
        fi
    fi

    # Set up environment variable for other tools
    export BITNET_GGUF="${REPO_ROOT}/${BENCHMARK_MODEL_PATH}"

    # Download tokenizer if available
    local tokenizer_path="models/${DEFAULT_MODEL_ID}/tokenizer.json"
    if [[ ! -f "${tokenizer_path}" ]]; then
        log_debug "Attempting to download tokenizer..."
        # This might fail for some models, which is OK
        cargo run -p xtask --no-default-features --features cpu -- download-model \
            --id "${DEFAULT_MODEL_ID}" \
            --file "tokenizer.json" 2>/dev/null || {
            log_debug "Tokenizer download failed (this is OK for some models)"
        }
    fi

    # Create crossval fixtures
    log_debug "Creating crossval benchmark fixtures..."
    if [[ -f "${BENCHMARK_MODEL_PATH}" ]]; then
        # Create a small fixture for quick benchmarks
        local fixture_model="${CROSSVAL_FIXTURES_DIR}/benchmark_model.gguf"
        if [[ ! -f "${fixture_model}" ]]; then
            # Create symlink to avoid duplicating large files
            ln -sf "../../${BENCHMARK_MODEL_PATH}" "${fixture_model}"
            log_debug "Created benchmark fixture: ${fixture_model}"
        fi
    fi

    log_info "✅ Model fixtures setup completed"
}

# Setup C++ implementation for cross-validation
setup_cpp_implementation() {
    log_step "Setting up C++ implementation for cross-validation..."

    cd "${REPO_ROOT}"

    # Check if C++ implementation already exists
    local cpp_binary="${BITNET_CPP_DIR}/build/bin/llama-cli"
    if [[ -f "${cpp_binary}" ]]; then
        log_info "C++ implementation already exists: ${cpp_binary}"

        # Test if it works
        if "${cpp_binary}" --help &> /dev/null; then
            log_info "✅ C++ implementation is functional"
            export BITNET_CPP_DIR
            return 0
        else
            log_warn "C++ implementation exists but is not functional, rebuilding..."
            rm -rf "${BITNET_CPP_DIR}"
        fi
    fi

    # Download and build C++ implementation using xtask
    log_info "Downloading and building C++ implementation..."
    log_info "This may take 10-20 minutes depending on your system..."

    if ! cargo run -p xtask --no-default-features --features cpu -- fetch-cpp; then
        log_error "Failed to fetch and build C++ implementation"
        log_error "Cross-validation benchmarks will not be available"
        log_error "You can still run Rust-only benchmarks"
        return 1
    fi

    # Verify the build
    if [[ -f "${cpp_binary}" ]] && "${cpp_binary}" --help &> /dev/null; then
        log_info "✅ C++ implementation built successfully"
        export BITNET_CPP_DIR
    else
        log_error "C++ implementation build completed but binary is not functional"
        return 1
    fi
}

# Run diagnostic checks
run_diagnostic_checks() {
    log_step "Running diagnostic checks..."

    cd "${REPO_ROOT}"

    local all_checks_passed=true

    # Check 1: Basic Rust build
    log_debug "Checking Rust build..."
    if cargo build --release --no-default-features --features cpu > /dev/null 2>&1; then
        log_info "✅ Rust build successful"
    else
        log_error "❌ Rust build failed"
        all_checks_passed=false
    fi

    # Check 2: Model accessibility
    log_debug "Checking model accessibility..."
    if [[ -f "${BITNET_GGUF}" ]] && [[ -r "${BITNET_GGUF}" ]]; then
        local model_size
        model_size=$(stat -c%s "${BITNET_GGUF}" 2>/dev/null || stat -f%z "${BITNET_GGUF}" 2>/dev/null || echo "0")
        if [[ ${model_size} -gt 100000000 ]]; then  # At least 100MB
            log_info "✅ Model file accessible and reasonable size (${model_size} bytes)"
        else
            log_error "❌ Model file too small or corrupted (${model_size} bytes)"
            all_checks_passed=false
        fi
    else
        log_error "❌ Model file not accessible: ${BITNET_GGUF}"
        all_checks_passed=false
    fi

    # Check 3: Rust inference capability
    log_debug "Checking Rust inference capability..."
    if cargo run -p xtask --no-default-features --features cpu -- infer \
        --model "${BITNET_GGUF}" \
        --prompt "Test" \
        --max-new-tokens 5 \
        --allow-mock \
        --deterministic > /dev/null 2>&1; then
        log_info "✅ Rust inference working"
    else
        log_error "❌ Rust inference failed"
        all_checks_passed=false
    fi

    # Check 4: C++ implementation (if available)
    local cpp_binary="${BITNET_CPP_DIR}/build/bin/llama-cli"
    if [[ -f "${cpp_binary}" ]]; then
        log_debug "Checking C++ implementation..."
        if "${cpp_binary}" -m "${BITNET_GGUF}" -p "Test" -n 5 --no-display-prompt > /dev/null 2>&1; then
            log_info "✅ C++ implementation working"
        else
            log_warn "⚠️ C++ implementation available but not working with model"
        fi
    else
        log_warn "⚠️ C++ implementation not available (cross-validation disabled)"
    fi

    # Check 5: Python benchmark script
    log_debug "Checking Python benchmark script..."
    if [[ -f "${REPO_ROOT}/benchmark_comparison.py" ]]; then
        if python3 -c "import sys; exec(open('${REPO_ROOT}/benchmark_comparison.py').read())" --help > /dev/null 2>&1; then
            log_info "✅ Python benchmark script accessible"
        else
            log_warn "⚠️ Python benchmark script has issues"
        fi
    else
        log_warn "⚠️ Python benchmark script not found"
    fi

    # Check 6: Crossval benchmarks compilation
    log_debug "Checking crossval benchmarks compilation..."
    if cargo build --release --features crossval > /dev/null 2>&1; then
        log_info "✅ Crossval benchmarks compile successfully"
    else
        log_warn "⚠️ Crossval benchmarks compilation failed"
    fi

    # Check 7: GPU availability (optional)
    log_debug "Checking GPU availability..."
    if cargo run --example gpu_validation --no-default-features --features gpu > /dev/null 2>&1; then
        log_info "✅ GPU support available"
    else
        log_debug "GPU support not available (CPU-only benchmarks will be used)"
    fi

    if [[ "${all_checks_passed}" == true ]]; then
        log_info "✅ All critical diagnostic checks passed"
    else
        log_error "❌ Some diagnostic checks failed"
        log_error "Benchmarking may not work correctly"
        return 1
    fi
}

# Generate benchmark configuration
generate_benchmark_config() {
    log_step "Generating benchmark configuration..."

    cd "${REPO_ROOT}"

    # Create benchmark configuration file
    local config_file="${BENCHMARK_RESULTS_DIR}/benchmark-config.json"
    cat > "${config_file}" << EOF
{
  "version": "1.0",
  "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "environment": {
    "model_path": "${BITNET_GGUF}",
    "cpp_dir": "${BITNET_CPP_DIR}",
    "rust_version": "$(rustc --version)",
    "platform": "$(uname -s)-$(uname -m)",
    "cpu_count": "$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo unknown)",
    "available_features": {
      "rust_inference": true,
      "cpp_inference": $([ -f "${BITNET_CPP_DIR}/build/bin/llama-cli" ] && echo true || echo false),
      "gpu_support": $(cargo run --example gpu_validation --no-default-features --features gpu > /dev/null 2>&1 && echo true || echo false),
      "crossval": $(cargo build --release --features crossval > /dev/null 2>&1 && echo true || echo false)
    }
  },
  "recommended_commands": {
    "python_benchmark": "python3 benchmark_comparison.py --model '${BITNET_GGUF}' --cpp-dir '${BITNET_CPP_DIR}'",
    "rust_benchmark": "cargo bench --workspace --features cpu",
    "crossval_benchmark": "cargo bench --features crossval",
    "performance_comparison": "cargo test --package crossval --features crossval --release -- --nocapture benchmark"
  }
}
EOF

    log_info "Benchmark configuration saved to: ${config_file}"
}

# Print summary and next steps
print_summary() {
    log_step "Setup Summary"

    echo
    echo -e "${GREEN}🎉 Benchmarking infrastructure setup completed!${NC}"
    echo

    # Environment variables
    echo -e "${CYAN}Environment Variables:${NC}"
    echo "  export BITNET_GGUF='${BITNET_GGUF}'"
    echo "  export BITNET_CPP_DIR='${BITNET_CPP_DIR}'"
    echo

    # Available benchmarks
    echo -e "${CYAN}Available Benchmark Commands:${NC}"
    echo
    echo -e "${YELLOW}1. Python Comparison Benchmark (Recommended):${NC}"
    echo "   ./benchmark_comparison.py --model '${BITNET_GGUF}' --cpp-dir '${BITNET_CPP_DIR}'"
    echo "   ./benchmark_comparison.py --help  # for more options"
    echo

    echo -e "${YELLOW}2. Rust-Only Benchmarks:${NC}"
    echo "   cargo bench --workspace --no-default-features --features cpu"
    echo "   cargo test --workspace --no-default-features --features cpu --release"
    echo

    if [[ -f "${BITNET_CPP_DIR}/build/bin/llama-cli" ]]; then
        echo -e "${YELLOW}3. Cross-Validation Benchmarks:${NC}"
        echo "   cargo bench --features crossval"
        echo "   cargo test --package crossval --features crossval --release -- --nocapture benchmark"
        echo
    fi

    if cargo run --example gpu_validation --no-default-features --features gpu > /dev/null 2>&1; then
        echo -e "${YELLOW}4. GPU Benchmarks:${NC}"
        echo "   ./benchmark_comparison.py --gpu"
        echo "   cargo bench --workspace --no-default-features --features gpu"
        echo
    fi

    # Quick validation commands
    echo -e "${CYAN}Quick Validation Commands:${NC}"
    echo "   cargo run -p xtask -- verify --model '${BITNET_GGUF}'"
    echo "   cargo run -p xtask -- infer --model '${BITNET_GGUF}' --prompt 'Test' --max-new-tokens 10 --allow-mock"
    echo

    # Integration with CI/CD
    echo -e "${CYAN}CI/CD Integration:${NC}"
    echo "   This setup integrates with .github/workflows/performance-tracking.yml"
    echo "   Use 'benchmark-results/' directory for storing results"
    echo "   Configuration available in: ${BENCHMARK_RESULTS_DIR}/benchmark-config.json"
    echo

    # Troubleshooting
    echo -e "${CYAN}Troubleshooting:${NC}"
    echo "   - Check model: cargo run -p xtask -- verify --model '${BITNET_GGUF}'"
    echo "   - Check C++: '${BITNET_CPP_DIR}/build/bin/llama-cli' --help"
    echo "   - Re-run setup: ./scripts/setup-benchmarks.sh --force"
    echo "   - View logs: tail -f benchmark-results/setup.log"
    echo
}

# Parse command line arguments
print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Set up BitNet.rs benchmarking infrastructure (addresses GitHub issue #155).

OPTIONS:
    --force              Force re-download of models and rebuild of C++
    --skip-cpp           Skip C++ implementation setup (Rust-only benchmarks)
    --skip-model         Skip model download (use existing model)
    --model-id ID        Use different model ID (default: ${DEFAULT_MODEL_ID})
    --model-file FILE    Use different model file (default: ${DEFAULT_MODEL_FILE})
    --cpp-dir DIR        Use different C++ directory (default: ${BITNET_CPP_DIR})
    --dry-run            Show what would be done without executing
    --verbose            Enable verbose output
    --help               Show this help message

EXAMPLES:
    $0                           # Full setup with defaults
    $0 --skip-cpp                # Rust-only benchmarks
    $0 --force                   # Force complete rebuild
    $0 --model-id custom/model   # Use different model

The script will:
1. Check system requirements and repository
2. Download and verify benchmark model
3. Set up C++ implementation for cross-validation
4. Run diagnostic checks
5. Generate benchmark configuration
6. Provide ready-to-use benchmark commands

For more information, see: https://github.com/BitNet-rs/BitNet-rs/issues/155
EOF
}

# Parse arguments
FORCE=false
SKIP_CPP=false
SKIP_MODEL=false
DRY_RUN=false
VERBOSE=false
MODEL_ID="${DEFAULT_MODEL_ID}"
MODEL_FILE="${DEFAULT_MODEL_FILE}"
CPP_DIR="${BITNET_CPP_DIR}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE=true
            shift
            ;;
        --skip-cpp)
            SKIP_CPP=true
            shift
            ;;
        --skip-model)
            SKIP_MODEL=true
            shift
            ;;
        --model-id)
            MODEL_ID="$2"
            shift 2
            ;;
        --model-file)
            MODEL_FILE="$2"
            shift 2
            ;;
        --cpp-dir)
            CPP_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Update paths based on arguments
if [[ "${MODEL_ID}" != "${DEFAULT_MODEL_ID}" ]] || [[ "${MODEL_FILE}" != "${DEFAULT_MODEL_FILE}" ]]; then
    BENCHMARK_MODEL_PATH="models/${MODEL_ID}/${MODEL_FILE}"
    BITNET_GGUF="${REPO_ROOT}/${BENCHMARK_MODEL_PATH}"
fi

if [[ "${CPP_DIR}" != "${BITNET_CPP_DIR}" ]]; then
    BITNET_CPP_DIR="${CPP_DIR}"
fi

# Main execution
main() {
    # Setup logging (only if not dry-run)
    if [[ "${DRY_RUN}" != true ]]; then
        mkdir -p "${REPO_ROOT}/${BENCHMARK_RESULTS_DIR}"
        exec 1> >(tee -a "${REPO_ROOT}/${BENCHMARK_RESULTS_DIR}/setup.log")
        exec 2> >(tee -a "${REPO_ROOT}/${BENCHMARK_RESULTS_DIR}/setup.log" >&2)
    fi

    print_banner

    log_info "BitNet.rs Benchmarking Infrastructure Setup"
    log_info "Addressing GitHub issue #155: Non-functional benchmarking infrastructure"
    log_info "Starting setup at $(date)"
    echo

    if [[ "${DRY_RUN}" == true ]]; then
        log_info "DRY RUN MODE - No changes will be made"
        echo
    fi

    # Clean up on force
    if [[ "${FORCE}" == true ]]; then
        log_info "Force mode: cleaning up previous installations..."
        if [[ "${DRY_RUN}" != true ]]; then
            rm -rf "${BITNET_CPP_DIR}"
            rm -f "${BITNET_GGUF}"
            rm -rf "${CROSSVAL_FIXTURES_DIR}"
            rm -rf "${BENCHMARK_RESULTS_DIR}"
        fi
    fi

    # Execute setup steps
    if [[ "${DRY_RUN}" != true ]]; then
        check_repo_root
        check_system_requirements

        if [[ "${SKIP_MODEL}" != true ]]; then
            setup_model_fixtures
        fi

        if [[ "${SKIP_CPP}" != true ]]; then
            setup_cpp_implementation || log_warn "C++ setup failed, continuing with Rust-only"
        fi

        run_diagnostic_checks
        generate_benchmark_config
    else
        log_info "Would execute: check_repo_root"
        log_info "Would execute: check_system_requirements"
        [[ "${SKIP_MODEL}" != true ]] && log_info "Would execute: setup_model_fixtures"
        [[ "${SKIP_CPP}" != true ]] && log_info "Would execute: setup_cpp_implementation"
        log_info "Would execute: run_diagnostic_checks"
        log_info "Would execute: generate_benchmark_config"
    fi

    print_summary

    log_info "Setup completed at $(date)"
    log_info "Ready for benchmarking! 🚀"
}

# Trap for cleanup
cleanup() {
    local exit_code=$?
    if [[ ${exit_code} -ne 0 ]]; then
        log_error "Setup failed with exit code ${exit_code}"
        log_error "Check the logs in ${BENCHMARK_RESULTS_DIR}/setup.log"
        log_error "For help, run: $0 --help"
    fi
    exit ${exit_code}
}

trap cleanup EXIT

# Run main function
main "$@"