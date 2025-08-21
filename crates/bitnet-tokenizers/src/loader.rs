/// Tokenizer loading utilities
use crate::Tokenizer;
use anyhow::Result;
use std::path::Path;

/// Load a tokenizer from a file path
pub fn load_tokenizer(path: &Path) -> Result<Box<dyn Tokenizer>> {
    // Check file extension to determine tokenizer type
    let ext = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    
    match ext {
        "gguf" => {
            // Load tokenizer from GGUF file
            Ok(Box::new(crate::gguf_tokenizer::GgufTokenizer::from_gguf_file(path)?))
        }
        "json" => {
            // TODO: Load HuggingFace tokenizer.json format
            anyhow::bail!("JSON tokenizer loading not yet implemented")
        }
        "model" => {
            // TODO: Load SentencePiece model
            anyhow::bail!("SentencePiece tokenizer loading not yet implemented")
        }
        _ => {
            anyhow::bail!("Unknown tokenizer file format: {}", ext)
        }
    }
}
