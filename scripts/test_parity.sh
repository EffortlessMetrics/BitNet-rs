#!/bin/bash
# Script to run full parity tests with the BitNet GGUF model

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Step 1: Check if model exists, provide download instructions if not
MODEL_DIR="models/bitnet-b1.58-2B-4T-gguf"
MODEL_FILE="$MODEL_DIR/ggml-model-i2_s.gguf"

if [ ! -f "$MODEL_FILE" ]; then
    log_error "Model not found at $MODEL_FILE"
    echo ""
    log_info "Please download the model using one of these methods:"
    echo ""
    echo "Option 1: Using huggingface-cli (recommended):"
    echo "  pip install -U 'huggingface_hub[cli]'"
    echo "  huggingface-cli download microsoft/bitnet-b1.58-2B-4T-gguf \\"
    echo "    --include 'ggml-model-i2_s.gguf' \\"
    echo "    --local-dir ./models/bitnet-b1.58-2B-4T-gguf"
    echo ""
    echo "Option 2: Using git-lfs:"
    echo "  git lfs install"
    echo "  git clone https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf models/bitnet-b1.58-2B-4T-gguf"
    echo ""
    echo "Option 3: Direct download with curl/wget:"
    echo "  mkdir -p $MODEL_DIR"
    echo "  curl -L -o $MODEL_FILE \\"
    echo "    https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf"
    echo ""
    exit 1
fi

log_info "Model found at $MODEL_FILE"

# Step 2: Build C++ implementation if not already built
log_info "Checking C++ implementation..."
if [ ! -d "$HOME/.cache/bitnet_cpp" ]; then
    log_info "Building C++ implementation..."
    ./ci/fetch_bitnet_cpp.sh --tag b1-65-ggml --clean
else
    log_info "C++ implementation already built"
fi

# Step 3: Set environment variables for deterministic execution
export BITNET_CPP_DIR=$HOME/.cache/bitnet_cpp
export OMP_NUM_THREADS=1
export GGML_NUM_THREADS=1
export CROSSVAL_GGUF="$(realpath "$MODEL_FILE")"

log_info "Environment configured:"
log_info "  BITNET_CPP_DIR: $BITNET_CPP_DIR"
log_info "  CROSSVAL_GGUF: $CROSSVAL_GGUF"
log_info "  Threads: 1 (deterministic)"

# Step 4: Build Rust with crossval feature
log_info "Building Rust with cross-validation support..."
cargo build --features crossval --release -p bitnet-crossval

# Step 5: Run parity tests
log_info "================================================================"
log_info "Running Cross-Validation Parity Tests"
log_info "================================================================"

# Run all tests with verbose output and single thread for determinism
cargo test --features crossval --release -p bitnet-crossval -- --nocapture --test-threads=1

TEST_RESULT=$?

# Step 6: Report results
echo ""
log_info "================================================================"
if [ $TEST_RESULT -eq 0 ]; then
    log_info "✅ ALL PARITY TESTS PASSED!"
else
    log_error "❌ Some tests failed. Check output above for details."
fi
log_info "================================================================"

# Show how to run individual tests
echo ""
log_info "To run individual tests:"
log_info "  cargo test --features crossval -p bitnet-crossval test_tokenization_parity -- --nocapture"
log_info "  cargo test --features crossval -p bitnet-crossval test_single_step_logits -- --nocapture"
log_info "  cargo test --features crossval -p bitnet-crossval test_multi_step_generation -- --nocapture"

exit $TEST_RESULT
