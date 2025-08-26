#!/usr/bin/env bash
# Convenience script to run cross-validation tests against Microsoft BitNet C++
#
# Usage: ./scripts/crossval.sh [model.gguf]
#
# This script:
# 1. Fetches and builds the Microsoft BitNet C++ implementation
# 2. Sets up the environment for deterministic execution
# 3. Runs parity tests between Rust and C++ implementations

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
MODEL_PATH="${1:-}"
if [[ -n "$MODEL_PATH" ]] && [[ ! -f "$MODEL_PATH" ]]; then
    log_error "Model file not found: $MODEL_PATH"
    exit 1
fi

# Default cache directory
export BITNET_CPP_DIR="${BITNET_CPP_DIR:-$HOME/.cache/bitnet_cpp}"

log_info "================================================================"
log_info "BitNet Cross-Validation Test Runner"
log_info "================================================================"

# Step 1: Fetch and build Microsoft BitNet C++ if needed
if [[ ! -d "$BITNET_CPP_DIR/build" ]]; then
    log_info "Microsoft BitNet C++ not found, fetching and building..."
    "$REPO_ROOT/ci/fetch_bitnet_cpp.sh"
else
    log_info "Using existing BitNet C++ at: $BITNET_CPP_DIR"
    log_info "To rebuild, run: rm -rf $BITNET_CPP_DIR && $0"
fi

# Verify the build exists
if [[ ! -d "$BITNET_CPP_DIR/build" ]]; then
    log_error "BitNet C++ build directory not found!"
    log_error "Please run: ./ci/fetch_bitnet_cpp.sh"
    exit 1
fi

# Step 2: Set up library paths
log_info "Setting up library paths..."

# Find the actual library locations
LLAMA_LIB=""
GGML_LIB=""

for search_dir in \
    "$BITNET_CPP_DIR/build/3rdparty/llama.cpp/src" \
    "$BITNET_CPP_DIR/build/3rdparty/llama.cpp" \
    "$BITNET_CPP_DIR/build/lib"; do
    
    if [[ -f "$search_dir/libllama.so" ]]; then
        LLAMA_LIB="$search_dir"
        break
    elif [[ -f "$search_dir/libllama.dylib" ]]; then
        LLAMA_LIB="$search_dir"
        break
    fi
done

for search_dir in \
    "$BITNET_CPP_DIR/build/3rdparty/llama.cpp/ggml/src" \
    "$BITNET_CPP_DIR/build/3rdparty/llama.cpp/ggml" \
    "$BITNET_CPP_DIR/build/lib"; do
    
    if [[ -f "$search_dir/libggml.so" ]] || [[ -f "$search_dir/libggml.dylib" ]]; then
        GGML_LIB="$search_dir"
        break
    fi
done

if [[ -z "$LLAMA_LIB" ]]; then
    log_error "libllama not found in BitNet C++ build!"
    log_error "Build may have failed. Try: rm -rf $BITNET_CPP_DIR/build && ./ci/fetch_bitnet_cpp.sh"
    exit 1
fi

# Set library paths
if [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH="${LLAMA_LIB}:${GGML_LIB}:${DYLD_LIBRARY_PATH:-}"
else
    export LD_LIBRARY_PATH="${LLAMA_LIB}:${GGML_LIB}:${LD_LIBRARY_PATH:-}"
fi

log_info "Library paths configured:"
log_info "  LLAMA: $LLAMA_LIB"
if [[ -n "$GGML_LIB" ]]; then
    log_info "  GGML: $GGML_LIB"
fi

# Step 3: Set up deterministic execution environment
log_info "Configuring deterministic execution..."

# Force single-threaded execution for determinism
export OMP_NUM_THREADS=1
export GGML_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Disable GPU acceleration for deterministic CPU-only tests
export CUDA_VISIBLE_DEVICES=""
export LLAMA_CUDA_NO_DEVICES=1

# Set model path if provided
if [[ -n "$MODEL_PATH" ]]; then
    export CROSSVAL_GGUF="$MODEL_PATH"
    log_info "Using model: $MODEL_PATH"
else
    log_warn "No model specified. Set CROSSVAL_GGUF or pass model path as argument."
    log_warn "Example: ./scripts/crossval.sh /path/to/model.gguf"
fi

# Step 4: Build Rust with crossval feature
log_info "Building Rust with cross-validation support..."
cd "$REPO_ROOT"
cargo build --features crossval --release -p bitnet-crossval

# Step 5: Run cross-validation tests
log_info "================================================================"
log_info "Running cross-validation tests..."
log_info "================================================================"

# Run the tests with verbose output and single thread for determinism
cargo test --features crossval --release -p bitnet-crossval -- --nocapture --test-threads=1

# Store exit code
TEST_RESULT=$?

if [[ $TEST_RESULT -eq 0 ]]; then
    log_info "================================================================"
    log_info "✅ Cross-validation tests PASSED!"
    log_info "================================================================"
    log_info "The Rust implementation matches the C++ implementation!"
else
    log_error "================================================================"
    log_error "❌ Cross-validation tests FAILED!"
    log_error "================================================================"
    log_error "Check the test output above for divergence details."
    exit 1
fi

# Print summary
echo ""
log_info "Test environment:"
log_info "  BITNET_CPP_DIR: $BITNET_CPP_DIR"
log_info "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
log_info "  GGML_NUM_THREADS: $GGML_NUM_THREADS"
if [[ -n "${CROSSVAL_GGUF:-}" ]]; then
    log_info "  Model: $CROSSVAL_GGUF"
fi

echo ""
log_info "To run specific tests:"
log_info "  cargo test --features crossval -p bitnet-crossval test_tokenization_parity -- --nocapture"
log_info "  cargo test --features crossval -p bitnet-crossval test_single_step_logits -- --nocapture"
log_info "  cargo test --features crossval -p bitnet-crossval test_multi_step_generation -- --nocapture"