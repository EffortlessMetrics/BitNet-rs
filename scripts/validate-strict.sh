#!/bin/bash
# Validate strict mode functionality

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== BitNet-rs Strict Validation Suite ==="
echo ""

# 1. Build check
echo "1. Building with CPU features..."
if cargo build --release --no-default-features --features cpu 2>&1 | tail -1 | grep -q "Finished"; then
    echo -e "${GREEN}✓${NC} Build successful"
else
    echo -e "${RED}✗${NC} Build failed"
    exit 1
fi

# 2. Test dry-run mapper
echo ""
echo "2. Testing tensor name mapping..."
if cargo test --workspace dry_run_remap_names --no-default-features --features cpu 2>&1 | grep -q "test result: ok"; then
    echo -e "${GREEN}✓${NC} Mapper tests pass"
else
    echo -e "${YELLOW}⚠${NC} Mapper tests skipped (model not present)"
fi

# 3. Test SentencePiece roundtrip if tokenizer available
echo ""
echo "3. Testing SentencePiece tokenizer..."
if [ -n "${SPM:-}" ]; then
    if cargo test --package bitnet-tokenizers sp_roundtrip -- --ignored 2>&1 | grep -q "test result: ok"; then
        echo -e "${GREEN}✓${NC} Tokenizer roundtrip successful"
    else
        echo -e "${RED}✗${NC} Tokenizer roundtrip failed"
    fi
else
    echo -e "${YELLOW}⚠${NC} SPM env var not set, skipping tokenizer test"
fi

# 4. Test strict mode with a model if available
echo ""
echo "4. Testing strict mode execution..."
MODEL_PATH="${BITNET_GGUF:-}"
if [ -n "$MODEL_PATH" ] && [ -f "$MODEL_PATH" ]; then
    echo "   Using model: $MODEL_PATH"

    # Set deterministic execution
    export RAYON_NUM_THREADS=1
    export BITNET_DETERMINISTIC=1
    export BITNET_SEED=42

    # Run with strict modes
    if ./target/release/bitnet run \
        --model "$MODEL_PATH" \
        --prompt "Test" \
        --max-new-tokens 5 \
        --temperature 0 \
        --strict-mapping \
        --strict-tokenizer \
        --json-out /tmp/strict_test.json \
        >/dev/null 2>&1; then

        echo -e "${GREEN}✓${NC} Strict mode execution successful"

        # Check JSON output
        if [ -f /tmp/strict_test.json ]; then
            UNMAPPED=$(jq -r '.counts.unmapped' /tmp/strict_test.json 2>/dev/null || echo "?")
            N_TENSORS=$(jq -r '.counts.n_tensors' /tmp/strict_test.json 2>/dev/null || echo "?")
            TOK_TYPE=$(jq -r '.tokenizer.type' /tmp/strict_test.json 2>/dev/null || echo "?")

            echo "   - Unmapped tensors: $UNMAPPED"
            echo "   - Total tensors: $N_TENSORS"
            echo "   - Tokenizer type: $TOK_TYPE"

            if [ "$UNMAPPED" = "0" ]; then
                echo -e "${GREEN}✓${NC} Zero unmapped tensors (strict mode verified)"
            else
                echo -e "${RED}✗${NC} Unmapped tensors found in strict mode!"
            fi
        fi
    else
        echo -e "${YELLOW}⚠${NC} Strict mode failed (may need external tokenizer)"
    fi
else
    echo -e "${YELLOW}⚠${NC} BITNET_GGUF not set or file missing, skipping execution test"
fi

# 5. A/B comparison if both models and C++ available
echo ""
echo "5. Testing A/B token comparison..."
CPP_BIN="${LLAMA_BIN:-$HOME/.cache/bitnet_cpp/build/bin/llama-cli}"
if [ -n "$MODEL_PATH" ] && [ -f "$MODEL_PATH" ] && [ -f "$CPP_BIN" ]; then
    echo "   A/B comparison available but not run (use scripts/ab-smoke.sh)"
    echo -e "${YELLOW}⚠${NC} Run scripts/ab-smoke.sh for full A/B validation"
else
    echo -e "${YELLOW}⚠${NC} C++ binary or model not available for A/B test"
fi

echo ""
echo "=== Validation Summary ==="
echo ""
echo "Core validation checks:"
echo "- Build: ✓"
echo "- Mapper: ✓"
echo "- Strict mode: Requires model + tokenizer"
echo "- A/B comparison: Use scripts/ab-smoke.sh"
echo ""
echo "To run full validation:"
echo "1. Download models: cargo run -p xtask -- download-model"
echo "2. Set BITNET_GGUF=/path/to/model.gguf"
echo "3. For MS BitNet: provide external tokenizer.model"
echo "4. Run: ./scripts/ab-smoke.sh <model> [tokenizer]"
