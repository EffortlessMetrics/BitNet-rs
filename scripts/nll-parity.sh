#!/usr/bin/env bash
# Teacher-forcing NLL parity test
# Tests that BitNet's mean NLL matches reference implementation

set -euo pipefail

# Required environment variables
: "${MODEL_PATH:?Set MODEL_PATH to the BitNet model path}"
: "${TOKENIZER:?Set TOKENIZER to the tokenizer path}"
: "${BITNET_BIN:=target/release/bitnet}"
: "${HF_MODEL_ID:?Set HF_MODEL_ID to a compatible HuggingFace model}"

# Test configuration
: "${PPL_FILE:=crossval/data/ppl_smoke.txt}"
export DELTA_NLL_MAX="${DELTA_NLL_MAX:-1e-2}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Running NLL parity test...${NC}"
echo "Configuration:"
echo "  MODEL_PATH: $MODEL_PATH"
echo "  TOKENIZER: $TOKENIZER"
echo "  BITNET_BIN: $BITNET_BIN"
echo "  HF_MODEL_ID: $HF_MODEL_ID"
echo "  PPL_FILE: $PPL_FILE"
echo "  DELTA_NLL_MAX: $DELTA_NLL_MAX"
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

# Check if test corpus exists
if [ ! -f "$PPL_FILE" ]; then
    echo -e "${YELLOW}Creating test corpus at $PPL_FILE${NC}"
    mkdir -p "$(dirname "$PPL_FILE")"
    cat > "$PPL_FILE" << 'EOF'
The quick brown fox jumps over the lazy dog.
BitNet is a 1-bit transformer architecture designed for efficient inference.
Machine learning models can be quantized to reduce memory and computation requirements.
Rust provides memory safety without garbage collection through its ownership system.
Property-based testing helps find edge cases that unit tests might miss.
Large language models have revolutionized natural language processing tasks.
Quantization techniques enable running large models on resource-constrained devices.
The attention mechanism is a key component of transformer architectures.
Cross-entropy loss is commonly used for training language models.
Teacher forcing is a training technique where the model uses ground truth tokens.
EOF
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

# Check for transformers (required for HF comparison)
if ! python3 -c "import transformers" 2>/dev/null; then
    echo -e "${YELLOW}Installing transformers for HF comparison...${NC}"
    pip3 install -q transformers torch || {
        echo -e "${RED}Failed to install transformers${NC}"
        echo "Install manually with: pip3 install transformers torch"
        exit 1
    }
fi

# Test that BitNet eval command works
echo -e "${GREEN}Testing BitNet eval command...${NC}"
"$BITNET_BIN" eval \
    --model "$MODEL_PATH" \
    --tokenizer "$TOKENIZER" \
    --text-file "$PPL_FILE" \
    --deterministic \
    --threads 1 \
    --json-out /tmp/bitnet_eval_test.json \
    >/dev/null 2>&1 || {
        echo -e "${RED}BitNet eval command failed!${NC}"
        echo "Try running manually to see the error:"
        echo "  $BITNET_BIN eval --model $MODEL_PATH --tokenizer $TOKENIZER --text-file $PPL_FILE"
        exit 1
    }

if [ -f /tmp/bitnet_eval_test.json ]; then
    echo -e "${GREEN}BitNet eval output:${NC}"
    python3 -c "import json; d=json.load(open('/tmp/bitnet_eval_test.json')); print(f'  Mean NLL: {d.get(\"mean_nll\", \"N/A\")}')"
    rm -f /tmp/bitnet_eval_test.json
fi

# Run the parity test
echo -e "${GREEN}Starting NLL parity test...${NC}"

python3 -m pytest -xvs crossval/props/test_nll_parity.py::test_mean_nll_parity \
    --tb=short \
    || {
        echo -e "${RED}NLL parity test failed!${NC}"
        echo ""
        echo "This could mean:"
        echo "1. The models are not compatible (different architectures or tokenizers)"
        echo "2. The NLL computation has numerical differences"
        echo "3. The threshold DELTA_NLL_MAX=$DELTA_NLL_MAX is too strict"
        echo ""
        echo "Try increasing DELTA_NLL_MAX if the models are known to be slightly different."
        exit 1
    }

echo -e "${GREEN}✓ NLL parity test passed!${NC}"
