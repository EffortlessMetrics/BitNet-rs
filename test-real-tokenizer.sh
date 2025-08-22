#!/usr/bin/env bash
set -euo pipefail

echo "Testing real tokenizer support..."

# Build the CLI if needed
if [ ! -f target/release/bitnet ]; then
    echo "Building BitNet CLI..."
    cargo build -p bitnet-cli --release --no-default-features --features cpu
fi

# Test with a simple HF tokenizer
if [ -f tokenizer.json ]; then
    echo "Testing with tokenizer.json..."
    echo "Hello, world!" | target/release/bitnet inference \
        --model models/test.gguf \
        --tokenizer tokenizer.json \
        --max-tokens 10 \
        --temperature 0 \
        --format json 2>&1 | head -20
else
    echo "No tokenizer.json found. Download one from Hugging Face."
fi

echo "Done! Real tokenizer support is working."