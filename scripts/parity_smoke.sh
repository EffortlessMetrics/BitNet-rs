#!/usr/bin/env bash
set -euo pipefail

# parity_smoke.sh - One-command parity validation demo for BitNet.rs QK256 MVP
# Usage: ./scripts/parity_smoke.sh <model.gguf> [tokenizer.json]

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Usage
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model.gguf> [tokenizer.json]"
    echo ""
    echo "Runs parity validation and prints receipt summary"
    echo ""
    echo "Examples:"
    echo "  $0 models/model.gguf"
    echo "  $0 models/model.gguf models/tokenizer.json"
    echo ""
    echo "Environment:"
    echo "  BITNET_CPP_DIR - Path to C++ reference implementation (optional)"
    echo "                   Default: \$HOME/.cache/bitnet_cpp"
    exit 1
fi

MODEL_PATH="$1"
TOKENIZER_PATH="${2:-}"

# Check model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model not found: $MODEL_PATH${NC}"
    exit 1
fi

# Check jq is installed
if ! command -v jq &> /dev/null; then
    echo -e "${YELLOW}Warning: jq not found, receipt summary will be raw JSON${NC}"
    echo "Install jq for pretty output: apt-get install jq (Ubuntu) or brew install jq (macOS)"
    JQ_AVAILABLE=false
else
    JQ_AVAILABLE=true
fi

# Set up environment variables for cross-validation
export CROSSVAL_GGUF="$(realpath "$MODEL_PATH")"
export BITNET_GGUF="$CROSSVAL_GGUF"
export RAYON_NUM_THREADS=1
export BITNET_DETERMINISTIC=1
export BITNET_SEED=42

if [ -n "$TOKENIZER_PATH" ]; then
    export BITNET_TOKENIZER="$(realpath "$TOKENIZER_PATH")"
fi

# Set up C++ reference path if available
: "${BITNET_CPP_DIR:=$HOME/.cache/bitnet_cpp}"

if [ -d "$BITNET_CPP_DIR" ]; then
    echo -e "${BLUE}C++ reference: $BITNET_CPP_DIR${NC}"
    # Set RPATH or LD_LIBRARY_PATH for C++ libraries
    if [ -d "$BITNET_CPP_DIR/build/3rdparty/llama.cpp/src" ]; then
        export LD_LIBRARY_PATH="$BITNET_CPP_DIR/build/3rdparty/llama.cpp/src:$BITNET_CPP_DIR/build/3rdparty/llama.cpp/ggml/src:${LD_LIBRARY_PATH:-}"
    fi
else
    echo -e "${YELLOW}C++ reference: not available (rust-only mode)${NC}"
    echo "Set BITNET_CPP_DIR to enable full parity validation"
fi

# Run parity test
echo ""
echo "Running parity validation..."
echo "Model: $MODEL_PATH"
if [ -n "$TOKENIZER_PATH" ]; then
    echo "Tokenizer: $TOKENIZER_PATH"
fi
echo ""

# Create temp log file
TEMP_LOG=$(mktemp)
trap 'rm -f "$TEMP_LOG"' EXIT

# Link check (if binary exists)
if [ -f target/release/deps/parity_bitnetcpp-* ]; then
    echo -e "${BLUE}== Link Check ==${NC}"
    ldd target/release/deps/parity_bitnetcpp-* 2>/dev/null | grep -E 'llama|ggml' || echo "No C++ libraries linked"
    echo ""
fi

# Run parity test
echo -e "${BLUE}== Running Parity Test (release) ==${NC}"
set +e
cargo test -p bitnet-crossval --release \
    --features crossval,integration-tests \
    --test parity_bitnetcpp \
    -- --nocapture 2>&1 | tee "$TEMP_LOG"
TEST_EXIT_CODE=$?
set -e

echo ""

# Find receipt (most recent in docs/baselines/)
RECEIPT=$(find docs/baselines -name "parity-bitnetcpp.json" -type f 2>/dev/null | sort -r | head -n1)

if [ -z "$RECEIPT" ]; then
    echo -e "${RED}Error: No receipt found in docs/baselines/${NC}"
    echo "Check test output above for errors"
    exit 1
fi

echo -e "${BLUE}=== Parity Receipt Summary ===${NC}"
echo ""

# Pretty-print key fields
if [ "$JQ_AVAILABLE" = true ]; then
    jq '{
        validation: .validation,
        tokenizer: (.tokenizer | {kind, vocab_size, source}),
        quant: .quant,
        parity: .parity
    }' "$RECEIPT"
else
    # Fallback: show raw JSON
    cat "$RECEIPT"
fi

echo ""
echo "Full receipt: $RECEIPT"
echo ""

# Check status
if [ "$JQ_AVAILABLE" = true ]; then
    STATUS=$(jq -r '.parity.status' "$RECEIPT")
    CPP_AVAILABLE=$(jq -r '.parity.cpp_available' "$RECEIPT")
else
    # Fallback: grep-based extraction
    STATUS=$(grep -o '"status"[[:space:]]*:[[:space:]]*"[^"]*"' "$RECEIPT" | grep -o '"[^"]*"$' | tr -d '"' | tail -n1)
    CPP_AVAILABLE=$(grep -o '"cpp_available"[[:space:]]*:[[:space:]]*[^,}]*' "$RECEIPT" | grep -o '[^:]*$' | tr -d ' ' | tail -n1)
fi

# Determine exit status based on parity result
if [ "$STATUS" = "ok" ]; then
    echo -e "${GREEN}✅ Parity validation PASSED${NC}"
    echo ""
    if [ "$CPP_AVAILABLE" = "true" ]; then
        echo "Full C++ parity validation successful"
    else
        echo "Note: Rust-only mode (C++ reference not available)"
    fi
    exit 0
elif [ "$STATUS" = "rust_only" ]; then
    echo -e "${GREEN}✅ Parity validation PASSED (rust-only mode)${NC}"
    echo ""
    echo "Note: C++ reference not available for comparison"
    echo "For full parity validation, set BITNET_CPP_DIR to C++ reference path"
    exit 0
else
    echo -e "${RED}❌ Parity validation FAILED${NC}"
    echo ""
    echo "Status: $STATUS"
    echo "Check full receipt for details: $RECEIPT"
    exit 1
fi
