#!/usr/bin/env bash
# Quick test to verify deterministic greedy decoding works correctly

set -euo pipefail

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "Testing BitNet-rs Deterministic Greedy Decoding"
echo "==============================================="

# Build if needed
if [ ! -f "target/release/bitnet" ]; then
    echo "Building BitNet CLI..."
    cargo build -p bitnet-cli --release --no-default-features --features cpu
fi

# Test prompts
PROMPTS=(
    "What is 2+2?"
    "Complete this: The quick brown"
    "def fibonacci(n):"
    '{"name": "test", "value":'
    "List three colors:"
)

# Test configuration
MODEL="${MODEL_PATH:-models/test.gguf}"
TOKENIZER="${TOKENIZER:-models/tokenizer.json}"
SEED=12345
MAX_TOKENS=32

echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Tokenizer: $TOKENIZER"
echo "  Seed: $SEED"
echo "  Max tokens: $MAX_TOKENS"
echo ""

# Check files exist
if [ ! -f "$MODEL" ]; then
    echo -e "${RED}Error: Model not found at $MODEL${NC}"
    echo "Set MODEL_PATH environment variable"
    exit 1
fi

if [ ! -f "$TOKENIZER" ]; then
    echo -e "${RED}Warning: Tokenizer not found at $TOKENIZER${NC}"
    echo "Will try to proceed without explicit tokenizer"
fi

# Test each prompt twice for determinism
echo "Testing determinism..."
echo "----------------------"

FAILED=0

for i in "${!PROMPTS[@]}"; do
    PROMPT="${PROMPTS[$i]}"
    echo -n "Test $((i+1)): "

    # Run twice with same seed
    if [ -f "$TOKENIZER" ]; then
        CMD="target/release/bitnet run \
            --model '$MODEL' \
            --tokenizer '$TOKENIZER' \
            --prompt '$PROMPT' \
            --max-tokens $MAX_TOKENS \
            --seed $SEED \
            --greedy \
            --deterministic \
            --json-out /tmp/run1.json 2>/dev/null"
    else
        CMD="target/release/bitnet run \
            --model '$MODEL' \
            --prompt '$PROMPT' \
            --max-tokens $MAX_TOKENS \
            --seed $SEED \
            --greedy \
            --deterministic \
            --json-out /tmp/run1.json 2>/dev/null"
    fi

    # First run
    eval $CMD
    cp /tmp/run1.json /tmp/run1_backup.json

    # Second run
    eval ${CMD/run1.json/run2.json}

    # Compare outputs
    if python3 -c "
import json
with open('/tmp/run1.json') as f1, open('/tmp/run2.json') as f2:
    d1 = json.load(f1)
    d2 = json.load(f2)
    if d1.get('text') != d2.get('text'):
        print(f\"Output 1: {d1.get('text', 'N/A')}\")
        print(f\"Output 2: {d2.get('text', 'N/A')}\")
        exit(1)
" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Deterministic"
    else
        echo -e "${RED}✗${NC} Not deterministic!"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "Testing greedy mode..."
echo "----------------------"

# Test that greedy flag actually forces deterministic sampling
PROMPT="Generate a random story:"

# With greedy flag
target/release/bitnet run \
    --model "$MODEL" \
    ${TOKENIZER:+--tokenizer "$TOKENIZER"} \
    --prompt "$PROMPT" \
    --max-tokens 20 \
    --seed $SEED \
    --greedy \
    --json-out /tmp/greedy.json \
    >/dev/null 2>&1

# Without greedy (but temp=0 should be same)
target/release/bitnet run \
    --model "$MODEL" \
    ${TOKENIZER:+--tokenizer "$TOKENIZER"} \
    --prompt "$PROMPT" \
    --max-tokens 20 \
    --seed $SEED \
    --temperature 0 \
    --top-p 1 \
    --top-k 0 \
    --json-out /tmp/manual_greedy.json \
    >/dev/null 2>&1

if python3 -c "
import json
with open('/tmp/greedy.json') as f1, open('/tmp/manual_greedy.json') as f2:
    d1 = json.load(f1)
    d2 = json.load(f2)
    if d1.get('text') == d2.get('text'):
        print('✓ --greedy matches manual temp=0 settings')
    else:
        print('✗ --greedy produces different output than manual settings')
        exit(1)
" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Greedy flag works correctly"
else
    echo -e "${RED}✗${NC} Greedy flag issue detected"
    FAILED=$((FAILED + 1))
fi

# Summary
echo ""
echo "==============================================="
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All determinism tests passed!${NC}"

    # Show sample performance metrics
    echo ""
    echo "Sample metrics from last run:"
    python3 -c "
import json
with open('/tmp/greedy.json') as f:
    data = json.load(f)
    if 'timing_ms' in data:
        t = data['timing_ms']
        print(f\"  Tokenize: {t.get('tokenize', 0):.1f}ms\")
        print(f\"  Prefill:  {t.get('prefill', 0):.1f}ms\")
        print(f\"  Decode:   {t.get('decode', 0):.1f}ms\")
        print(f\"  Total:    {t.get('total', 0):.1f}ms\")
    if 'throughput_tps' in data:
        tps = data['throughput_tps']
        print(f\"  Decode TPS: {tps.get('decode', 0):.1f}\")
" 2>/dev/null || true
else
    echo -e "${RED}✗ $FAILED test(s) failed${NC}"
    exit 1
fi
