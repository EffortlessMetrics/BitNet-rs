//! Tokenizer auto-discovery for BitNet-rs CLI.

use anyhow::Result;
use bitnet_tokenizer_discovery_core::{ResolvedTokenizer, resolve_tokenizer_with_source};
use std::path::{Path, PathBuf};

/// Resolve tokenizer path with auto-discovery.
///
/// Priority:
/// 1. Explicit path (if provided).
/// 2. Sibling tokenizer.json.
/// 3. Parent tokenizer.json.
pub fn resolve_tokenizer(model_path: &Path, explicit_path: Option<PathBuf>) -> Result<PathBuf> {
    resolve_tokenizer_with_source(model_path, explicit_path, |candidate| {
        Ok(bitnet_tokenizers::loader::load_tokenizer(candidate).is_ok())
    })
    .map(|resolved| resolved.path)
}

/// Resolve tokenizer path with its strict receipt source.
pub fn resolve_tokenizer_with_receipt_source(
    model_path: &Path,
    explicit_path: Option<PathBuf>,
) -> Result<ResolvedTokenizer> {
    resolve_tokenizer_with_source(model_path, explicit_path, |candidate| {
        Ok(bitnet_tokenizers::loader::load_tokenizer(candidate).is_ok())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitnet_tokenizer_discovery_core::TokenizerSource;
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

    #[test]
    fn receipt_source_reports_sibling_discovery() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let model_path = temp_dir.path().join("model.gguf");
        let sibling_tokenizer = temp_dir.path().join("tokenizer.json");

        fs::write(&model_path, create_mock_gguf())?;
        fs::write(&sibling_tokenizer, create_mock_tokenizer())?;

        let result = resolve_tokenizer_with_receipt_source(&model_path, None)?;
        assert_eq!(result.path.canonicalize()?, sibling_tokenizer.canonicalize()?);
        assert_eq!(result.source, TokenizerSource::SiblingFile);
        Ok(())
    }
}
