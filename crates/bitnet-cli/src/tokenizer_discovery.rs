//! Tokenizer auto-discovery for BitNet-rs CLI.

use anyhow::Result;
use bitnet_tokenizer_discovery_core::resolve_tokenizer_with;
use std::path::{Path, PathBuf};

/// Resolve tokenizer path with auto-discovery.
///
/// Priority:
/// 1. Explicit path (if provided).
/// 2. Sibling tokenizer.json.
/// 3. Parent tokenizer.json.
pub fn resolve_tokenizer(model_path: &Path, explicit_path: Option<PathBuf>) -> Result<PathBuf> {
    resolve_tokenizer_with(model_path, explicit_path, |candidate| {
        Ok(bitnet_tokenizers::loader::load_tokenizer(candidate).is_ok())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_mock_tokenizer() -> String {
        r#"{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {},
    "merges": []
  }
}"#
        .to_string()
    }

    fn create_mock_gguf() -> Vec<u8> {
        b"GGUF\x03\x00\x00\x00".to_vec()
    }

    #[test]
    fn explicit_path_takes_precedence() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let model_path = temp_dir.path().join("model.gguf");
        let sibling_tokenizer = temp_dir.path().join("tokenizer.json");
        let explicit_tokenizer = temp_dir.path().join("explicit_tokenizer.json");

        fs::write(&model_path, create_mock_gguf())?;
        fs::write(&sibling_tokenizer, create_mock_tokenizer())?;
        fs::write(&explicit_tokenizer, create_mock_tokenizer())?;

        let result = resolve_tokenizer(&model_path, Some(explicit_tokenizer.clone()))?;
        assert_eq!(result.canonicalize()?, explicit_tokenizer.canonicalize()?);
        Ok(())
    }
}
