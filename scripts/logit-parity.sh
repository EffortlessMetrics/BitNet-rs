#!/usr/bin/env bash
# Logit parity test using Kendall's tau
# Tests that BitNet's top-k token rankings correlate with reference

set -euo pipefail

# Required environment variables
: "${MODEL_PATH:?Set MODEL_PATH to the BitNet model path}"
: "${TOKENIZER:?Set TOKENIZER to the tokenizer path}"
: "${BITNET_BIN:=target/release/bitnet}"

# Optional: HF model for cross-system comparison
: "${HF_MODEL_ID:=}"

# Test configuration
export PROP_EXAMPLES="${PROP_EXAMPLES:-20}"
export TAU_STEPS="${TAU_STEPS:-32}"
export LOGIT_TOPK="${LOGIT_TOPK:-10}"
export TAU_MIN="${TAU_MIN:-0.60}"
export PROP_MAX_NEW_TOKENS="${PROP_MAX_NEW_TOKENS:-128}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Running logit parity tests...${NC}"
echo "Configuration:"
echo "  MODEL_PATH: $MODEL_PATH"
echo "  TOKENIZER: $TOKENIZER"
echo "  BITNET_BIN: $BITNET_BIN"
echo "  HF_MODEL_ID: ${HF_MODEL_ID:-<not set>}"
echo "  TAU_STEPS: $TAU_STEPS"
echo "  LOGIT_TOPK: $LOGIT_TOPK"
echo "  TAU_MIN: $TAU_MIN"
echo "  PROP_EXAMPLES: $PROP_EXAMPLES"
echo ""

# Check if BitNet binary exists
if [ ! -f "$BITNET_BIN" ]; then
    echo -e "${RED}Error: BitNet binary not found at $BITNET_BIN${NC}"
    echo "Build with: cargo build -p bitnet-cli --release --no-default-features --features cpu"
    exit 1
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model not found at $MODEL_PATH${NC}"
    exit 1
fi

# Install Python dependencies if needed
if ! python3 -c "import hypothesis" 2>/dev/null; then
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    pip3 install -q hypothesis pytest numpy || {
        echo -e "${RED}Failed to install Python dependencies${NC}"
        echo "Install manually with: pip3 install hypothesis pytest numpy"
        exit 1
    }
fi

# If HF_MODEL_ID is set, also check for transformers
if [ -n "$HF_MODEL_ID" ]; then
    if ! python3 -c "import transformers" 2>/dev/null; then
        echo -e "${YELLOW}Installing transformers for HF comparison...${NC}"
        pip3 install -q transformers torch || {
            echo -e "${YELLOW}Warning: Could not install transformers${NC}"
            echo "HF comparison will be skipped"
            unset HF_MODEL_ID
        }
    fi
fi

# Run the test
echo -e "${GREEN}Starting logit parity test...${NC}"

if [ -n "$HF_MODEL_ID" ]; then
    echo "Testing against HuggingFace model: $HF_MODEL_ID"
else
    echo "Testing BitNet self-consistency (no HF model configured)"
fi

python3 -m pytest -xvs crossval/props/test_logit_parity.py::test_logit_parity_tau \
    --tb=short \
    --hypothesis-show-statistics \
    || {
        echo -e "${RED}Logit parity test failed!${NC}"
        exit 1
    }

echo -e "${GREEN}✓ Logit parity test passed!${NC}"