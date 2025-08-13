//! Minimal inference example for BitNet models
//!
//! This example demonstrates how to load and run inference on a BitNet model.
//! Set BITNET_GGUF=/path/to/model.gguf to run the example.

use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Skeleton; keep dependencies minimal and compile-fast.
    // Load path from env so CI can skip if not set.
    let Some(path) = env::var_os("BITNET_GGUF") else {
        eprintln!("Set BITNET_GGUF=/path/to/model.gguf to run the example");
        eprintln!();
        eprintln!("You can download a model using:");
        eprintln!("  cargo xtask download-model");
        eprintln!();
        eprintln!("Then run:");
        eprintln!("  export BITNET_GGUF=models/ggml-model-i2_s.gguf");
        eprintln!("  cargo run --example infer");
        return Ok(());
    };

    // TODO: replace with your real API once wired:
    // let model = bitnet::load_gguf(&path)?;
    // let input_tokens = vec![1, 2, 3, 4, 5]; // Example token IDs
    // let logits = model.forward(&input_tokens)?;
    // println!("logits[0..5] = {:?}", &logits[..5]);

    println!("(placeholder) would load {:?}", path);
    println!();
    println!("Note: Full inference API is still being implemented.");
    println!("This example will be updated once the API is complete.");
    
    Ok(())
}