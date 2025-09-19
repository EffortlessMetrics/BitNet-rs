/// Tokenizer loading utilities
use crate::Tokenizer;
use anyhow::{Context, Result};
use serde_json::Value;
use std::{fs, path::Path};

/// Load a tokenizer from a file path
///
/// Supports multiple tokenizer formats:
/// - `.gguf` - GGUF model files with embedded tokenizers
/// - `.json` - Hugging Face tokenizer.json files (requires `model.type` field)
/// - `.model` - SentencePiece model files (requires `spm` feature)
///
/// # Arguments
/// * `path` - Path to the tokenizer file
///
/// # Returns
/// A boxed tokenizer instance implementing the `Tokenizer` trait
///
/// # Errors
/// Returns an error if:
/// - The file cannot be read
/// - The file format is unknown or invalid
/// - The JSON structure is missing required fields
/// - The tokenizer fails to load
pub fn load_tokenizer(path: &Path) -> Result<Box<dyn Tokenizer + Send + Sync>> {
    // Check file extension to determine tokenizer type
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "gguf" => {
            // Load tokenizer from GGUF file
            Ok(Box::new(crate::gguf_tokenizer::GgufTokenizer::from_gguf_file(path)?))
        }
        "json" => {
            // Validate JSON structure before loading
            let data = fs::read_to_string(path).context("Failed to read tokenizer JSON file")?;
            let value: Value =
                serde_json::from_str(&data).context("Invalid tokenizer JSON format")?;

            if value.get("model").and_then(|m| m.get("type")).and_then(|t| t.as_str()).is_none() {
                anyhow::bail!("Unsupported tokenizer JSON structure: missing 'model.type' field");
            }

            let tokenizer = crate::hf_tokenizer::HfTokenizer::from_file(path)
                .map_err(|e| anyhow::anyhow!("Failed to load HuggingFace tokenizer: {e}"))?;
            Ok(Box::new(tokenizer))
        }
        "model" => {
            // Load SentencePiece model directly
            #[cfg(feature = "spm")]
            {
                match crate::sp_tokenizer::SpTokenizer::from_file(path) {
                    Ok(t) => Ok(t),
                    Err(e) => anyhow::bail!("Failed to load SentencePiece model: {}", e),
                }
            }
            #[cfg(not(feature = "spm"))]
            {
                anyhow::bail!("SentencePiece support not compiled in. Enable the 'spm' feature.")
            }
        }
        _ => {
            anyhow::bail!("Unknown tokenizer file format: {}", ext)
        }
    }
}

/// Load tokenizer from GGUF metadata (using HashMap)
pub fn load_tokenizer_from_gguf(
    metadata: &std::collections::HashMap<String, serde_json::Value>,
) -> Result<Box<dyn Tokenizer + Send + Sync>> {
    use base64::Engine;

    // Check if we have a SentencePiece model embedded
    if let Some(model_blob) = metadata.get("tokenizer.ggml.model") {
        // Decode to raw bytes (either base64 string or u8 array)
        let bytes: Option<Vec<u8>> = if let Some(blob_str) = model_blob.as_str() {
            Some(base64::engine::general_purpose::STANDARD.decode(blob_str)?)
        } else {
            model_blob.as_array().map(|blob_array| {
                blob_array.iter().filter_map(|v| v.as_u64().map(|b| b as u8)).collect()
            })
        };
        if let Some(bytes) = bytes {
            #[cfg(feature = "spm")]
            {
                let bos = metadata
                    .get("tokenizer.ggml.bos_token_id")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32);
                let eos = metadata
                    .get("tokenizer.ggml.eos_token_id")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32);
                return Ok(crate::sp_tokenizer::SpTokenizer::from_gguf_blob(&bytes, bos, eos)?);
            }
            #[cfg(not(feature = "spm"))]
            {
                let _ = bytes;
                return Err(anyhow::anyhow!(
                    "SentencePiece support not compiled in. Enable the 'spm' feature."
                ));
            }
        }
    }

    // If no embedded tokenizer, fail
    anyhow::bail!("GGUF file does not contain an embedded tokenizer (tokenizer.ggml.model)")
}

/// Load tokenizer from GGUF reader
pub fn load_tokenizer_from_gguf_reader(
    reader: &bitnet_models::GgufReader,
) -> Result<Box<dyn Tokenizer + Send + Sync>> {
    // Check if the GGUF contains an embedded tokenizer (try both binary and array formats)
    if let Some(bytes) = reader.get_bin_or_u8_array("tokenizer.ggml.model") {
        #[cfg(feature = "spm")]
        {
            let bos = reader.get_u32_metadata("tokenizer.ggml.bos_token_id");
            let eos = reader.get_u32_metadata("tokenizer.ggml.eos_token_id");
            return Ok(crate::sp_tokenizer::SpTokenizer::from_gguf_blob(&bytes, bos, eos)?);
        }
        #[cfg(not(feature = "spm"))]
        {
            let _ = bytes;
            return Err(anyhow::anyhow!(
                "SentencePiece support not compiled in. Enable the 'spm' feature."
            ));
        }
    }

    Err(anyhow::anyhow!("GGUF missing tokenizer.ggml.model"))
}
