#!/usr/bin/env bash
# Property-based testing for greedy decoding parity
# Tests BitNet.rs against reference implementations with adversarial prompts

set -euo pipefail

# Required environment variables
: "${MODEL_PATH:?Set MODEL_PATH to your .gguf model}"
: "${TOKENIZER:?Set TOKENIZER to tokenizer.json or .model}"
: "${BITNET_BIN:=target/release/bitnet}"

# Optional reference systems
LLAMA_BIN="${LLAMA_BIN:-}"
LLAMA_MODEL="${LLAMA_MODEL:-$MODEL_PATH}"
HF_MODEL_ID="${HF_MODEL_ID:-}"

# Test configuration
export PYTHONUNBUFFERED=1
export PROP_EXAMPLES="${PROP_EXAMPLES:-40}"
export PROP_MAX_NEW_TOKENS="${PROP_MAX_NEW_TOKENS:-128}"
export PROP_TIMEOUT="${PROP_TIMEOUT:-180}"

# Thresholds for approximate matching
export PROP_PREFIX_MIN="${PROP_PREFIX_MIN:-10}"
export PROP_BIGRAM_F1_MIN="${PROP_BIGRAM_F1_MIN:-0.55}"
export PROP_LEV_MAX="${PROP_LEV_MAX:-60}"
export PROP_COMBINED_MIN="${PROP_COMBINED_MIN:-0.65}"

# Artifact saving
export PROP_SAVE_ARTIFACTS="${PROP_SAVE_ARTIFACTS:-1}"
export PROP_ARTIFACTS_DIR="${PROP_ARTIFACTS_DIR:-test-artifacts}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Property-Based Greedy Parity Testing${NC}"
echo "================================================"
echo "Model:      $MODEL_PATH"
echo "Tokenizer:  $TOKENIZER"
echo "BitNet:     $BITNET_BIN"
echo "Examples:   $PROP_EXAMPLES"
echo "Max tokens: $PROP_MAX_NEW_TOKENS"
echo ""

# Check BitNet binary exists
if [ ! -f "$BITNET_BIN" ]; then
    echo -e "${RED}Error: BitNet binary not found at $BITNET_BIN${NC}"
    echo "Build with: cargo build -p bitnet-cli --release --no-default-features --features cpu"
    exit 1
fi

# Check model and tokenizer exist
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model not found at $MODEL_PATH${NC}"
    exit 1
fi

if [ ! -f "$TOKENIZER" ]; then
    echo -e "${RED}Error: Tokenizer not found at $TOKENIZER${NC}"
    exit 1
fi

# Report reference systems
echo "Reference systems:"
if [ -n "$LLAMA_BIN" ] && [ -f "$LLAMA_BIN" ]; then
    echo "  - llama.cpp: $LLAMA_BIN"
    export LLAMA_BIN LLAMA_MODEL
elif [ -n "$HF_MODEL_ID" ]; then
    echo "  - HuggingFace: $HF_MODEL_ID"
    export HF_MODEL_ID
else
    echo -e "  ${YELLOW}None (testing determinism only)${NC}"
fi
echo ""

# Create artifacts directory
mkdir -p "$PROP_ARTIFACTS_DIR"

# Install Python dependencies if needed
if ! python3 -c "import hypothesis" 2>/dev/null; then
    echo "Installing Python dependencies..."
    pip3 install -q hypothesis pytest numpy scipy || {
        echo -e "${YELLOW}Warning: Could not install Python dependencies${NC}"
        echo "Install manually with: pip3 install hypothesis pytest numpy scipy"
    }
fi

# Run the property tests
echo "Running property tests..."
echo "------------------------"

python3 -m pytest \
    crossval/props/test_greedy_parity.py \
    -v \
    --tb=short \
    --color=yes \
    -k "not test_very_long_prompt" \
    || TEST_RESULT=$?

# Report results
echo ""
echo "================================================"
if [ "${TEST_RESULT:-0}" -eq 0 ]; then
    echo -e "${GREEN}✓ All property tests passed!${NC}"
    
    # Show summary of artifacts if any
    if [ -d "$PROP_ARTIFACTS_DIR" ] && [ "$(ls -A "$PROP_ARTIFACTS_DIR")" ]; then
        echo ""
        echo "Test artifacts saved in: $PROP_ARTIFACTS_DIR"
        echo "Recent files:"
        ls -lt "$PROP_ARTIFACTS_DIR" | head -5
    fi
else
    echo -e "${RED}✗ Some property tests failed${NC}"
    echo ""
    echo "Debug artifacts saved in: $PROP_ARTIFACTS_DIR"
    
    # Show most recent failure
    if [ -d "$PROP_ARTIFACTS_DIR" ] && [ "$(ls -A "$PROP_ARTIFACTS_DIR")" ]; then
        LATEST=$(ls -t "$PROP_ARTIFACTS_DIR"/*.json 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            echo ""
            echo "Most recent failure: $LATEST"
            echo "Key info:"
            python3 -c "
import json
with open('$LATEST') as f:
    data = json.load(f)
    print(f\"  Prompt: {data.get('prompt', 'N/A')}\")
    print(f\"  Seed: {data.get('seed', 'N/A')}\")
    if 'metrics' in data:
        m = data['metrics']
        print(f\"  Metrics: prefix={m.get('prefix_match', 'N/A')}, \")
        print(f\"           bigram_f1={m.get('bigram_f1', 'N/A'):.3f}, \")
        print(f\"           levenshtein={m.get('levenshtein', 'N/A')}\")
"
        fi
    fi
    
    exit 1
fi