#!/usr/bin/env bash
# Prepare test models for validation
# This ensures we have both SafeTensors and GGUF models for parity testing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}BitNet.rs Test Model Preparation${NC}"
echo "========================================="

# Check if we already have models
GGUF="$(find models -type f -name '*.gguf' | head -n1 || true)"
SAFETENSORS="$(find models -type f -name '*.safetensors' | head -n1 || true)"

if [[ -n "$GGUF" ]]; then
    echo -e "${GREEN}✓${NC} Found GGUF: $GGUF"
else
    echo -e "${YELLOW}⚠${NC} No GGUF model found"
fi

if [[ -n "$SAFETENSORS" ]]; then
    echo -e "${GREEN}✓${NC} Found SafeTensors: $SAFETENSORS"
else
    echo -e "${YELLOW}⚠${NC} No SafeTensors model found"
fi

# If we have at least one model, that's enough to start
if [[ -n "$GGUF" || -n "$SAFETENSORS" ]]; then
    if [[ -n "$GGUF" && -n "$SAFETENSORS" ]]; then
        echo -e "\n${GREEN}✓${NC} Both formats available for full parity testing"
        exit 0
    elif [[ -n "$GGUF" ]]; then
        echo -e "\n${YELLOW}ℹ${NC} GGUF available - can run basic tests"
        echo -e "${YELLOW}ℹ${NC} For full parity testing, SafeTensors model needed"
        exit 0
    elif [[ -n "$SAFETENSORS" ]]; then
        echo -e "\n${YELLOW}ℹ${NC} SafeTensors available - can run basic tests"
        echo -e "${YELLOW}ℹ${NC} For full parity testing, GGUF model needed"
    fi
fi

# Otherwise, check HF cache or environment
: "${HF_MODEL_ID:=1bitLLM/bitnet_b1_58-3B}"
: "${HF_CACHE:=$HOME/.cache/bitnet-rs/models/$HF_MODEL_ID}"
: "${MODEL_DIR:=models/test}"

echo -e "\nChecking for cached models in: $HF_CACHE"

# Try to locate models from cache
if [[ -f "$HF_CACHE/model.safetensors" && -f "$HF_CACHE/tokenizer.json" ]]; then
    echo -e "${GREEN}✓${NC} Found cached SafeTensors model"

    mkdir -p "$MODEL_DIR"

    # Copy tokenizer
    if [[ ! -f "$MODEL_DIR/tokenizer.json" ]]; then
        cp "$HF_CACHE/tokenizer.json" "$MODEL_DIR/tokenizer.json"
        echo -e "${GREEN}✓${NC} Copied tokenizer to $MODEL_DIR/"
    fi

    # Copy or link SafeTensors
    if [[ ! -f "$MODEL_DIR/model.safetensors" ]]; then
        ln -sf "$HF_CACHE/model.safetensors" "$MODEL_DIR/model.safetensors"
        echo -e "${GREEN}✓${NC} Linked SafeTensors model"
    fi

    # Convert to GGUF if converter exists and GGUF is missing
    if [[ -z "$GGUF" ]]; then
        if [[ -x scripts/convert_safetensors_to_gguf_validated.py ]]; then
            echo -e "\n${YELLOW}Converting SafeTensors to GGUF...${NC}"
            python3 scripts/convert_safetensors_to_gguf_validated.py \
                --input "$MODEL_DIR/model.safetensors" \
                --tokenizer "$MODEL_DIR/tokenizer.json" \
                --output "$MODEL_DIR/model.gguf"
            echo -e "${GREEN}✓${NC} Created GGUF: $MODEL_DIR/model.gguf"
        else
            echo -e "${YELLOW}⚠${NC} GGUF converter not found"
            echo "  To enable conversion, ensure scripts/convert_safetensors_to_gguf_validated.py exists"
        fi
    fi

    echo -e "\n${GREEN}✓${NC} Test models prepared in: $MODEL_DIR"
    ls -la "$MODEL_DIR"

elif [[ -n "${BITNET_GGUF:-}" && -f "$BITNET_GGUF" ]]; then
    # Use environment-provided GGUF
    echo -e "${GREEN}✓${NC} Using BITNET_GGUF: $BITNET_GGUF"
    mkdir -p "$MODEL_DIR"
    ln -sf "$BITNET_GGUF" "$MODEL_DIR/model.gguf"

else
    echo -e "\n${RED}No models found!${NC}"
    echo ""
    echo "Options to get models:"
    echo "1. Download from Hugging Face:"
    echo "   cargo run -p xtask -- download-model"
    echo ""
    echo "2. Set environment variable:"
    echo "   export BITNET_GGUF=/path/to/model.gguf"
    echo ""
    echo "3. Place models in the models/ directory"
    echo ""
    exit 2
fi

# Final check
GGUF="$(find models -type f -name '*.gguf' | head -n1 || true)"
SAFETENSORS="$(find models -type f -name '*.safetensors' | head -n1 || true)"

echo -e "\n${GREEN}Final Model Status:${NC}"
echo "-------------------"
if [[ -n "$GGUF" ]]; then
    echo -e "GGUF:        ${GREEN}✓${NC} $GGUF"
else
    echo -e "GGUF:        ${RED}✗${NC} Missing"
fi

if [[ -n "$SAFETENSORS" ]]; then
    echo -e "SafeTensors: ${GREEN}✓${NC} $SAFETENSORS"
else
    echo -e "SafeTensors: ${YELLOW}⚠${NC} Missing (parity tests will be limited)"
fi

echo ""
if [[ -n "$GGUF" ]]; then
    echo -e "${GREEN}Ready for testing!${NC}"
    exit 0
else
    echo -e "${RED}At least one model format is required${NC}"
    exit 1
fi
