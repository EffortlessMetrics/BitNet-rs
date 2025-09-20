use crate::Tokenizer;
use anyhow::{Result, bail};
use std::path::Path;
use std::sync::Arc;

pub fn load_auto(
    model_path: &Path,
    explicit: Option<&Path>,
) -> Result<Arc<dyn Tokenizer + Send + Sync>> {
    if let Some(p) = explicit {
        let boxed_tok = crate::load_tokenizer(p)?;
        return Ok(Arc::from(boxed_tok) as Arc<dyn Tokenizer + Send + Sync>);
    }

    // Try tokenizer embedded in GGUF
    if model_path.extension().and_then(|s| s.to_str()) == Some("gguf")
        && let Ok(tok) = crate::gguf_tokenizer::GgufTokenizer::from_gguf_file(model_path)
    {
        tracing::debug!("tokenizer: loaded from GGUF");
        return Ok(Arc::new(tok));
    }

    // Try tokenizer.json / tokenizer.model in the model directory
    if let Some(dir) = model_path.parent() {
        let json = dir.join("tokenizer.json");
        if json.exists() {
            let boxed_tok = crate::load_tokenizer(&json)?;
            return Ok(Arc::from(boxed_tok) as Arc<dyn Tokenizer + Send + Sync>);
        }
        let spm = dir.join("tokenizer.model");
        if spm.exists() {
            let boxed_tok = crate::load_tokenizer(&spm)?;
            return Ok(Arc::from(boxed_tok) as Arc<dyn Tokenizer + Send + Sync>);
        }
    }

    // Do not silently use BasicTokenizer; better to fail and instruct user
    bail!(
        "No tokenizer found. Provide --tokenizer or include tokenizer.json/.model next to the GGUF."
    );
}
