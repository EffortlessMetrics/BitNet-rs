use crate::Tokenizer;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Stable tokenizer source names for proof and receipt surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerSource {
    Explicit,
    GgufMetadata,
    Sibling,
    CompatibilityFallback,
}

impl TokenizerSource {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Explicit => "explicit",
            Self::GgufMetadata => "gguf_metadata",
            Self::Sibling => "sibling",
            Self::CompatibilityFallback => "compatibility_fallback",
        }
    }
}

/// Result of deterministic tokenizer resolution.
#[derive(Clone)]
pub struct TokenizerResolution {
    pub tokenizer: Arc<dyn Tokenizer + Send + Sync>,
    pub source: TokenizerSource,
    pub strict: bool,
    pub path: Option<PathBuf>,
}

pub fn load_auto(
    model_path: &Path,
    explicit: Option<&Path>,
) -> Result<Arc<dyn Tokenizer + Send + Sync>> {
    Ok(resolve_tokenizer(model_path, explicit, false)?.tokenizer)
}

/// Resolve a tokenizer using the strict CPU-BITNET precedence.
///
/// Precedence:
/// 1. Explicit tokenizer path.
/// 2. Tokenizer embedded in GGUF metadata.
/// 3. Sibling tokenizer assets next to the model.
/// 4. Error. Compatibility fallbacks must be handled by callers explicitly.
pub fn resolve_tokenizer(
    model_path: &Path,
    explicit: Option<&Path>,
    strict: bool,
) -> Result<TokenizerResolution> {
    resolve_tokenizer_with(model_path, explicit, strict, crate::load_tokenizer, load_gguf_tokenizer)
}

fn resolve_tokenizer_with<LoadPath, LoadGguf>(
    model_path: &Path,
    explicit: Option<&Path>,
    strict: bool,
    mut load_path: LoadPath,
    mut load_gguf: LoadGguf,
) -> Result<TokenizerResolution>
where
    LoadPath: FnMut(&Path) -> Result<Arc<dyn Tokenizer + Send + Sync>>,
    LoadGguf: FnMut(&Path) -> Result<Arc<dyn Tokenizer + Send + Sync>>,
{
    if let Some(p) = explicit {
        tracing::info!("Using tokenizer: {}", p.display());
        let tokenizer = load_path(p)
            .with_context(|| format!("Failed to load explicit tokenizer {}", p.display()))?;
        return Ok(TokenizerResolution {
            tokenizer,
            source: TokenizerSource::Explicit,
            strict,
            path: Some(p.to_path_buf()),
        });
    }

    // Try tokenizer embedded in GGUF (using proper BPE/SPM implementation)
    if model_path.extension().and_then(|s| s.to_str()) == Some("gguf") {
        match load_gguf(model_path) {
            Ok(tok) => {
                tracing::info!("Using tokenizer: embedded in GGUF (BPE/SPM)");
                return Ok(TokenizerResolution {
                    tokenizer: tok,
                    source: TokenizerSource::GgufMetadata,
                    strict,
                    path: Some(model_path.to_path_buf()),
                });
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
    let tokenizer_dir = if model_path.is_dir() { Some(model_path) } else { model_path.parent() };

    if let Some(dir) = tokenizer_dir {
        for file_name in ["tokenizer.json", "tokenizer.model"] {
            let candidate = dir.join(file_name);
            if candidate.exists() {
                tracing::info!("Using tokenizer: {} (auto-detected)", candidate.display());
                let tokenizer = load_path(&candidate).with_context(|| {
                    format!("Failed to load sibling tokenizer {}", candidate.display())
                })?;
                return Ok(TokenizerResolution {
                    tokenizer,
                    source: TokenizerSource::Sibling,
                    strict,
                    path: Some(candidate),
                });
            }
        }
    }

    // Do not silently use BasicTokenizer; better to fail and instruct user
    let searched_dir = model_path
        .parent()
        .map(|d| d.display().to_string())
        .unwrap_or_else(|| "<unknown>".to_string());
    bail!(
        "No tokenizer found for model '{}'. Searched directory: {}. \
         Provide --tokenizer <path> or place tokenizer.json / tokenizer.model \
         next to the GGUF file.",
        model_path.display(),
        searched_dir
    );
}

/// Load tokenizer from GGUF file using pure-Rust BPE/SPM implementation
pub fn load_gguf_tokenizer(model_path: &Path) -> Result<Arc<dyn Tokenizer + Send + Sync>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MockTokenizer;
    use anyhow::anyhow;
    use std::fs;
    use tempfile::TempDir;

    fn mock_tokenizer() -> Arc<dyn Tokenizer + Send + Sync> {
        Arc::new(MockTokenizer::new())
    }

    #[test]
    fn explicit_path_takes_precedence() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let model_path = temp_dir.path().join("model.gguf");
        let explicit = temp_dir.path().join("explicit.json");
        fs::write(&model_path, b"GGUF")?;
        fs::write(&explicit, b"{}")?;

        let result = resolve_tokenizer_with(
            &model_path,
            Some(&explicit),
            true,
            |_| Ok(mock_tokenizer()),
            |_| Err(anyhow!("gguf should not be used")),
        )?;

        assert_eq!(result.source, TokenizerSource::Explicit);
        assert!(result.strict);
        assert_eq!(result.path.as_deref(), Some(explicit.as_path()));
        Ok(())
    }

    #[test]
    fn gguf_metadata_wins_before_sibling() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let model_path = temp_dir.path().join("model.gguf");
        let sibling = temp_dir.path().join("tokenizer.json");
        fs::write(&model_path, b"GGUF")?;
        fs::write(&sibling, b"{}")?;

        let result = resolve_tokenizer_with(
            &model_path,
            None,
            true,
            |_| Err(anyhow!("sibling should not be used")),
            |_| Ok(mock_tokenizer()),
        )?;

        assert_eq!(result.source, TokenizerSource::GgufMetadata);
        assert!(result.strict);
        assert_eq!(result.path.as_deref(), Some(model_path.as_path()));
        Ok(())
    }

    #[test]
    fn sibling_selected_after_missing_gguf_tokenizer() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let model_path = temp_dir.path().join("model.gguf");
        let sibling = temp_dir.path().join("tokenizer.json");
        fs::write(&model_path, b"GGUF")?;
        fs::write(&sibling, b"{}")?;

        let result = resolve_tokenizer_with(
            &model_path,
            None,
            false,
            |_| Ok(mock_tokenizer()),
            |_| Err(anyhow!("no embedded tokenizer")),
        )?;

        assert_eq!(result.source, TokenizerSource::Sibling);
        assert!(!result.strict);
        assert_eq!(result.path.as_deref(), Some(sibling.as_path()));
        Ok(())
    }

    #[test]
    fn missing_tokenizer_fails_in_strict_mode() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let model_path = temp_dir.path().join("model.gguf");
        fs::write(&model_path, b"GGUF")?;

        let result = resolve_tokenizer_with(
            &model_path,
            None,
            true,
            |_| Ok(mock_tokenizer()),
            |_| Err(anyhow!("no embedded tokenizer")),
        );

        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn source_strings_are_stable() {
        assert_eq!(TokenizerSource::Explicit.as_str(), "explicit");
        assert_eq!(TokenizerSource::GgufMetadata.as_str(), "gguf_metadata");
        assert_eq!(TokenizerSource::Sibling.as_str(), "sibling");
        assert_eq!(TokenizerSource::CompatibilityFallback.as_str(), "compatibility_fallback");
    }
}
