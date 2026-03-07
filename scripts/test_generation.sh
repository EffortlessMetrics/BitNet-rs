#!/bin/bash
# Simple test script for text generation

echo "Testing BitNet text generation..."

# Build only what we need
cargo build -p bitnet-models -q
cargo build -p bitnet-tokenizers -q

# Create a simple test program
cat > test_gen.rs << 'EOF'
use bitnet_models::{BitNetModel, Model};
use bitnet_tokenizers::MockTokenizer;
use bitnet_common::{BitNetConfig, Device};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Creating model...");
    let config = BitNetConfig::default();
    let model = BitNetModel::new(config.clone(), Device::Cpu);
    let model = Arc::new(model) as Arc<dyn Model>;

    println!("Creating tokenizer...");
    let tokenizer = MockTokenizer::new();

    println!("Encoding prompt...");
    let tokens = tokenizer.encode("Hello world", true)?;
    println!("Tokens: {:?}", tokens);

    println!("Running inference...");
    let embeddings = model.embed(&tokens)?;
    println!("Embeddings shape: {:?}", embeddings.shape());

    // Simple forward pass
    let mut cache: Box<dyn std::any::Any> = Box::new(());
    let hidden = model.forward(&embeddings, cache.as_mut())?;
    println!("Hidden shape: {:?}", hidden.shape());

    let logits = model.logits(&hidden)?;
    println!("Logits shape: {:?}", logits.shape());

    println!("Test successful!");
    Ok(())
}
EOF

# Compile and run test
rustc --edition 2021 test_gen.rs \
  -L target/debug/deps \
  --extern bitnet_models=target/debug/libbitnet_models.rlib \
  --extern bitnet_tokenizers=target/debug/libbitnet_tokenizers.rlib \
  --extern bitnet_common=target/debug/libbitnet_common.rlib \
  --extern candle_core=target/debug/deps/libcandle_core-*.rlib \
  -o test_gen

if [ -f test_gen ]; then
    ./test_gen
    rm test_gen
fi

rm test_gen.rs
