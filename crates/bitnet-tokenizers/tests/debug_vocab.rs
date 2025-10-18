//! Debug test to inspect GGUF vocabulary at specific positions

use std::fs;

#[test]
fn debug_gguf_vocab_positions() {
    let model_path = std::env::var("BITNET_GGUF")
        .unwrap_or_else(|_| "/home/steven/code/Rust/BitNet-rs/models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf".to_string());

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("⚠️  Skipping: Model not found");
        return;
    }

    let data = fs::read(&model_path).expect("Failed to read model");
    use bitnet_models::formats::gguf::GgufReader;
    let reader = GgufReader::new(&data).expect("Failed to parse GGUF");

    let tokens =
        reader.get_string_array_metadata("tokenizer.ggml.tokens").expect("No tokens found");

    println!("📊 GGUF Vocabulary Debug");
    println!("Total vocab size: {}", tokens.len());
    println!();

    // Check specific positions
    let positions = [3639, 3923, 374, 318, 220, 17, 10, 30];

    for &pos in &positions {
        if let Some(token) = tokens.get(pos) {
            println!("  Position {}: {:?}", pos, token);
        }
    }

    println!();
    println!("🔍 Searching for 'ĠWhat' and related tokens:");

    // Search for specific tokens
    let search_tokens = ["ĠWhat", " What", "What", "Ġis", " is"];
    for search in &search_tokens {
        if let Some((idx, _)) = tokens.iter().enumerate().find(|(_, t)| t.as_str() == *search) {
            println!("  Found {:?} at position {}", search, idx);
        } else {
            println!("  {:?} not found in vocab", search);
        }
    }
}
