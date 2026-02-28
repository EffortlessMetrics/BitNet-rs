/// Tokenizer loading utilities
use crate::Tokenizer;
use anyhow::{Context, Result};
use serde_json::Value;
use std::{fs, path::Path, sync::Arc};

// ---- SentencePiece helpers (conditional on `spm` feature) ---------------

#[cfg(feature = "spm")]
fn try_load_spm_from_file(path: &Path) -> Result<Arc<dyn Tokenizer + Send + Sync>> {
    let tokenizer = crate::sp_tokenizer::SpTokenizer::from_file(path)
        .map_err(|e| anyhow::anyhow!("Failed to load SentencePiece model: {}", e))?;
    Ok(Arc::new(tokenizer))
}

#[cfg(not(feature = "spm"))]
fn try_load_spm_from_file(_path: &Path) -> Result<Arc<dyn Tokenizer + Send + Sync>> {
    anyhow::bail!(
        "SentencePiece (.model) tokenizer requested but not compiled in. \
         Enable the 'spm' feature: cargo build --features spm"
    )
}

#[cfg(feature = "spm")]
fn try_load_spm_from_blob(
    bytes: &[u8],
    bos: Option<u32>,
    eos: Option<u32>,
) -> Result<Arc<dyn Tokenizer + Send + Sync>> {
    let tokenizer = crate::sp_tokenizer::SpTokenizer::from_gguf_blob(bytes, bos, eos)?;
    Ok(Arc::new(tokenizer))
}

#[cfg(not(feature = "spm"))]
fn try_load_spm_from_blob(
    _bytes: &[u8],
    _bos: Option<u32>,
    _eos: Option<u32>,
) -> Result<Arc<dyn Tokenizer + Send + Sync>> {
    anyhow::bail!(
        "SentencePiece (embedded GGUF tokenizer) requested but not compiled in. \
         Enable the 'spm' feature: cargo build --features spm"
    )
}

// ---- Public API ----------------------------------------------------------

/// Load a tokenizer from a file path.
///
/// Supports multiple tokenizer formats:
/// - `.gguf` — GGUF model files with embedded tokenizers
/// - `.json` — Hugging Face tokenizer.json files (requires `model.type` field)
/// - `.model` — SentencePiece model files (requires `spm` feature)
///
/// # Errors
/// Returns an error if the file cannot be read, the format is unrecognised,
/// or the tokenizer fails to load.  When the `spm` feature is not compiled in
/// and a `.model` file is supplied, a clear actionable error is returned.
pub fn load_tokenizer(path: &Path) -> Result<Arc<dyn Tokenizer + Send + Sync>> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "gguf" => Ok(Arc::new(crate::gguf_tokenizer::GgufTokenizer::from_gguf_file(path)?)),
        "json" => {
            let data = fs::read_to_string(path).context("Failed to read tokenizer JSON file")?;
            let value: Value =
                serde_json::from_str(&data).context("Invalid tokenizer JSON format")?;

            if value.get("model").and_then(|m| m.get("type")).and_then(|t| t.as_str()).is_none() {
                anyhow::bail!("Unsupported tokenizer JSON structure: missing 'model.type' field");
            }

            let tokenizer = crate::hf_tokenizer::HfTokenizer::from_file(path)
                .map_err(|e| anyhow::anyhow!("Failed to load HuggingFace tokenizer: {e}"))?;
            Ok(Arc::new(tokenizer))
        }
        "model" => try_load_spm_from_file(path),
        _ => {
            anyhow::bail!(
                "Unknown tokenizer file format '{}' for path: {}. \
                 Supported formats: .json (HuggingFace), .gguf (GGUF embedded), .model (SentencePiece)",
                ext,
                path.display()
            )
        }
    }
}

/// Load a tokenizer from a GGUF metadata map.
///
/// Looks for an embedded SentencePiece model blob under the
/// `tokenizer.ggml.model` key.  If the `spm` feature is not compiled in a
/// clear actionable error is returned instead of a cryptic missing-symbol
/// message.
pub fn load_tokenizer_from_gguf(
    metadata: &std::collections::HashMap<String, serde_json::Value>,
) -> Result<Arc<dyn Tokenizer + Send + Sync>> {
    use base64::Engine;

    if let Some(model_blob) = metadata.get("tokenizer.ggml.model") {
        let bytes: Option<Vec<u8>> = if let Some(blob_str) = model_blob.as_str() {
            Some(base64::engine::general_purpose::STANDARD.decode(blob_str)?)
        } else {
            model_blob.as_array().map(|blob_array| {
                blob_array.iter().filter_map(|v| v.as_u64().map(|b| b as u8)).collect()
            })
        };

        if let Some(bytes) = bytes {
            let bos = metadata
                .get("tokenizer.ggml.bos_token_id")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
            let eos = metadata
                .get("tokenizer.ggml.eos_token_id")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
            return try_load_spm_from_blob(&bytes, bos, eos);
        }
    }

    anyhow::bail!("GGUF file does not contain an embedded tokenizer (tokenizer.ggml.model)")
}

/// Load a tokenizer from a `GgufReader`.
///
/// Reads the embedded SentencePiece blob from `tokenizer.ggml.model`.
/// Returns a clear error when the `spm` feature is not compiled in.
pub fn load_tokenizer_from_gguf_reader(
    reader: &bitnet_models::GgufReader,
) -> Result<Arc<dyn Tokenizer + Send + Sync>> {
    if let Some(bytes) = reader.get_bin_or_u8_array("tokenizer.ggml.model") {
        let bos = reader.get_u32_metadata("tokenizer.ggml.bos_token_id");
        let eos = reader.get_u32_metadata("tokenizer.ggml.eos_token_id");
        return try_load_spm_from_blob(&bytes, bos, eos);
    }

    Err(anyhow::anyhow!("GGUF missing tokenizer.ggml.model"))
}
