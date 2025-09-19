use std::path::Path;
use anyhow::{bail, Result};
use crate::Tokenizer;

pub fn load_auto(model_path: &Path, explicit: Option<&Path>) -> Result<Box<dyn Tokenizer>> {
    if let Some(p) = explicit {
        return crate::load_tokenizer(p);
    }

    // Try tokenizer embedded in GGUF
    if model_path.extension().and_then(|s| s.to_str()) == Some("gguf") {
        if let Ok(tok) = crate::gguf_tokenizer::GgufTokenizer::from_gguf_file(model_path) {
            tracing::debug!("tokenizer: loaded from GGUF");
            return Ok(Box::new(tok));
        }
    }

    // Try tokenizer.json / tokenizer.model in the model directory
    if let Some(dir) = model_path.parent() {
        let json = dir.join("tokenizer.json");
        if json.exists() {
            return crate::load_tokenizer(&json);
        }
        let spm = dir.join("tokenizer.model");
        if spm.exists() {
            return crate::load_tokenizer(&spm);
        }
    }

    // Do not silently use BasicTokenizer; better to fail and instruct user
    bail!("No tokenizer found. Provide --tokenizer or include tokenizer.json/.model next to the GGUF.");
}