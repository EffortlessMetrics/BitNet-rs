use crate::Tokenizer;
use anyhow::{Result, bail};
use std::path::Path;
use std::sync::Arc;

pub fn load_auto(
    model_path: &Path,
    explicit: Option<&Path>,
) -> Result<Arc<dyn Tokenizer + Send + Sync>> {
    if let Some(p) = explicit {
        tracing::info!("Using tokenizer: {}", p.display());
        return crate::load_tokenizer(p);
    }

    // Try tokenizer embedded in GGUF (using proper BPE/SPM implementation)
    if model_path.extension().and_then(|s| s.to_str()) == Some("gguf") {
        match load_gguf_tokenizer(model_path) {
            Ok(tok) => {
                tracing::info!("Using tokenizer: embedded in GGUF (BPE/SPM)");
                return Ok(tok);
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to load tokenizer from GGUF: {}. Will try external tokenizer files.",
                    e
                );
            }
        }
    }

    // Try tokenizer.json / tokenizer.model in the model directory
    if let Some(dir) = model_path.parent() {
        let json = dir.join("tokenizer.json");
        if json.exists() {
            tracing::info!("Using tokenizer: {} (auto-detected)", json.display());
            return crate::load_tokenizer(&json);
        }
        let spm = dir.join("tokenizer.model");
        if spm.exists() {
            tracing::info!("Using tokenizer: {} (auto-detected)", spm.display());
            return crate::load_tokenizer(&spm);
        }
    }

    // Do not silently use BasicTokenizer; better to fail and instruct user
    bail!(
        "No tokenizer found. Provide --tokenizer or include tokenizer.json/.model next to the GGUF."
    );
}

/// Load tokenizer from GGUF file using pure-Rust BPE/SPM implementation
fn load_gguf_tokenizer(model_path: &Path) -> Result<Arc<dyn Tokenizer + Send + Sync>> {
    use bitnet_models::formats::gguf::GgufReader;
    use bitnet_models::loader::MmapFile;

    // Memory-map the GGUF file
    let mmap = MmapFile::open(model_path)?;

    // Create GGUF reader
    let reader = GgufReader::new(mmap.as_slice())?;

    // Load tokenizer from GGUF metadata (BPE or SPM)
    let tokenizer = crate::gguf_loader::RustTokenizer::from_gguf(&reader)?;

    Ok(Arc::new(tokenizer))
}
