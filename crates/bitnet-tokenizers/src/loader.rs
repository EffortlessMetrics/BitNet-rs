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
            // Load SentencePiece model directly
            match crate::sp_tokenizer::SpTokenizer::from_file(path) {
                Ok(t) => Ok(t),
                Err(e) => anyhow::bail!("Failed to load SentencePiece model: {}", e),
            }
        }
        _ => {
            anyhow::bail!("Unknown tokenizer file format: {}", ext)
        }
    }
}

/// Load tokenizer from GGUF metadata (using HashMap)
pub fn load_tokenizer_from_gguf(metadata: &std::collections::HashMap<String, serde_json::Value>) -> Result<Box<dyn Tokenizer>> {
    use base64::Engine;
    
    // Check if we have a SentencePiece model embedded
    if let Some(model_blob) = metadata.get("tokenizer.ggml.model") {
        if let Some(blob_str) = model_blob.as_str() {
            // It's base64 encoded
            let bytes = base64::engine::general_purpose::STANDARD.decode(blob_str)?;
            
            let bos = metadata.get("tokenizer.ggml.bos_token_id")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
            let eos = metadata.get("tokenizer.ggml.eos_token_id")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
                
            return Ok(crate::sp_tokenizer::SpTokenizer::from_gguf_blob(&bytes, bos, eos)?);
        } else if let Some(blob_array) = model_blob.as_array() {
            // It's a byte array
            let bytes: Vec<u8> = blob_array
                .iter()
                .filter_map(|v| v.as_u64().map(|b| b as u8))
                .collect();
                
            let bos = metadata.get("tokenizer.ggml.bos_token_id")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
            let eos = metadata.get("tokenizer.ggml.eos_token_id")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
                
            return Ok(crate::sp_tokenizer::SpTokenizer::from_gguf_blob(&bytes, bos, eos)?);
        }
    }
    
    // If no embedded tokenizer, fail
    anyhow::bail!("GGUF file does not contain an embedded tokenizer (tokenizer.ggml.model)")
}

/// Load tokenizer from GGUF reader
pub fn load_tokenizer_from_gguf_reader(reader: &bitnet_models::GgufReader) -> Result<Box<dyn Tokenizer>> {
    // Check if the GGUF contains an embedded tokenizer (try both binary and array formats)
    if let Some(bytes) = reader.get_bin_or_u8_array("tokenizer.ggml.model") {
        // Get BOS/EOS token IDs from metadata
        let bos = reader.get_u32_metadata("tokenizer.ggml.bos_token_id");
        let eos = reader.get_u32_metadata("tokenizer.ggml.eos_token_id");
        
        // Load the SentencePiece tokenizer from the embedded blob
        return Ok(crate::sp_tokenizer::SpTokenizer::from_gguf_blob(&bytes, bos, eos)?);
    }
    
    Err(anyhow::anyhow!("GGUF missing tokenizer.ggml.model").into())
}
