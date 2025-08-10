#!/bin/bash
# Test script for simple generation

echo "Testing simple generation..."

# Create dummy model file
touch dummy.gguf

# Build required crates
cargo build -p bitnet-models -q 2>/dev/null || true
cargo build -p bitnet-tokenizers -q 2>/dev/null || true
cargo build -p bitnet-cli -q 2>/dev/null || true

# Run the test
echo "Running generation test..."
cargo run -p bitnet-cli -q -- run \
    --model dummy.gguf \
    --prompt "Hello world" \
    --max-new-tokens 8 \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.9 \
    --repetition-penalty 1.1 \
    --seed 42

echo "Done!"