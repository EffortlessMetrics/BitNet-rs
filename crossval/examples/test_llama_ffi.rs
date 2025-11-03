use bitnet_crossval::cpp_bindings::*;
use std::path::Path;

fn main() {
    println!("Testing llama.cpp FFI bindings...");

    let model_path = Path::new("models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf");

    if !model_path.exists() {
        println!("✗ Model not found: {}", model_path.display());
        std::process::exit(1);
    }

    println!("✓ Model found: {}", model_path.display());

    match tokenize_bitnet(model_path, "Hello, world!", true, false) {
        Ok(tokens) => {
            println!("✓ Tokenization succeeded!");
            println!("  Tokens ({} total): {:?}", tokens.len(), &tokens[..tokens.len().min(10)]);
        }
        Err(e) => {
            println!("✗ Tokenization failed: {}", e);
            std::process::exit(1);
        }
    }

    println!("\n✓ All tests passed!");
}
