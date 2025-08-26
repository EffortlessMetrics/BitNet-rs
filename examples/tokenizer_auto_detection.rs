#[cfg(all(feature = "examples", feature = "tokenizers"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use bitnet_tokenizers::UniversalTokenizer;
    use std::path::Path;

    // Load tokenizer configuration directly from a GGUF model file
    let tokenizer = UniversalTokenizer::from_gguf(Path::new("model.gguf"))?;
    println!("Loaded tokenizer with vocab size {}", tokenizer.vocab_size());
    Ok(())
}

#[cfg(not(all(feature = "examples", feature = "tokenizers")))]
fn main() {}
