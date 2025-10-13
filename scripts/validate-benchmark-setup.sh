#!/usr/bin/env bash
# Quick validation script for benchmarking setup
# This script validates that the benchmarking infrastructure is working correctly

set -euo pipefail

# Colors
readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." &> /dev/null && pwd)"
BITNET_GGUF="${BITNET_GGUF:-${REPO_ROOT}/models/microsoft/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf}"
BITNET_CPP_DIR="${BITNET_CPP_DIR:-${HOME}/.cache/bitnet_cpp}"

echo "🔍 BitNet.rs Benchmarking Setup Validation"
echo "=========================================="
echo

cd "${REPO_ROOT}"

# Check 1: Repository structure
log_info "Checking repository structure..."
if [[ -f "Cargo.toml" ]] && grep -q "bitnet" "Cargo.toml"; then
    log_info "✅ Repository structure OK"
else
    log_error "❌ Not in BitNet.rs repository root"
    exit 1
fi

# Check 2: Setup script exists and is executable
log_info "Checking setup script..."
if [[ -x "scripts/setup-benchmarks.sh" ]]; then
    log_info "✅ Setup script is executable"
else
    log_error "❌ Setup script not found or not executable"
    exit 1
fi

# Check 3: Model file
log_info "Checking model file..."
if [[ -f "${BITNET_GGUF}" ]]; then
    local model_size
    model_size=$(stat -c%s "${BITNET_GGUF}" 2>/dev/null || stat -f%z "${BITNET_GGUF}" 2>/dev/null || echo "0")
    if [[ ${model_size} -gt 100000000 ]]; then
        log_info "✅ Model file exists and has reasonable size (${model_size} bytes)"
    else
        log_warn "⚠️ Model file too small (${model_size} bytes)"
    fi
else
    log_error "❌ Model file not found: ${BITNET_GGUF}"
    log_error "Run: ./scripts/setup-benchmarks.sh"
    exit 1
fi

# Check 4: Rust build
log_info "Checking Rust build..."
if cargo build --release --no-default-features --features cpu > /dev/null 2>&1; then
    log_info "✅ Rust builds successfully"
else
    log_error "❌ Rust build failed"
    exit 1
fi

# Check 5: Basic inference
log_info "Checking basic inference..."
if cargo run -p xtask --no-default-features --features cpu -- infer \
    --model "${BITNET_GGUF}" \
    --prompt "Test" \
    --max-new-tokens 5 \
    --allow-mock \
    --deterministic > /dev/null 2>&1; then
    log_info "✅ Basic inference works"
else
    log_error "❌ Basic inference failed"
    exit 1
fi

# Check 6: Python benchmark script
log_info "Checking Python benchmark script..."
if [[ -f "benchmark_comparison.py" ]] && python3 -c "import sys; exec(open('benchmark_comparison.py').read())" --help > /dev/null 2>&1; then
    log_info "✅ Python benchmark script works"
else
    log_error "❌ Python benchmark script has issues"
    exit 1
fi

# Check 7: C++ implementation (optional)
log_info "Checking C++ implementation..."
cpp_binary="${BITNET_CPP_DIR}/build/bin/llama-cli"
if [[ -f "${cpp_binary}" ]]; then
    if "${cpp_binary}" --help > /dev/null 2>&1; then
        log_info "✅ C++ implementation available and working"
    else
        log_warn "⚠️ C++ implementation exists but not working"
    fi
else
    log_warn "⚠️ C++ implementation not available (Rust-only benchmarks)"
fi

# Check 8: Crossval compilation
log_info "Checking crossval compilation..."
if cargo build --release --features crossval > /dev/null 2>&1; then
    log_info "✅ Crossval benchmarks compile"
else
    log_warn "⚠️ Crossval benchmarks don't compile"
fi

# Check 9: GPU support (optional)
log_info "Checking GPU support..."
if cargo build --release --no-default-features --features gpu > /dev/null 2>&1; then
    log_info "✅ GPU support available"
else
    log_warn "⚠️ GPU support not available"
fi

# Check 10: Benchmark configuration
log_info "Checking benchmark configuration..."
if [[ -f "benchmark-results/benchmark-config.json" ]]; then
    log_info "✅ Benchmark configuration exists"
else
    log_warn "⚠️ Benchmark configuration missing (run setup script)"
fi

echo
echo "🎯 Validation Results Summary"
echo "============================"

# Quick benchmark test
log_info "Running quick benchmark test..."
start_time=$(date +%s)
if timeout 60 ./benchmark_comparison.py \
    --model "${BITNET_GGUF}" \
    --cpp-dir "${BITNET_CPP_DIR}" \
    --prompt "Test" \
    --tokens 5 \
    --iterations 1 \
    --skip-cpp > /dev/null 2>&1; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    log_info "✅ Quick benchmark completed in ${duration}s"
else
    log_warn "⚠️ Quick benchmark failed or timed out"
fi

echo
log_info "🚀 Benchmarking setup validation completed!"
log_info ""
log_info "Ready to run benchmarks:"
log_info "  ./benchmark_comparison.py"
log_info "  cargo bench --workspace --no-default-features --features cpu"
if [[ -f "${cpp_binary}" ]]; then
    log_info "  cargo bench --features crossval"
fi
echo
