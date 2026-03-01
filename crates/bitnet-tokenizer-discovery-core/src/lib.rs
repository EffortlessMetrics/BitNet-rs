use anyhow::{Context, Result, anyhow};
use std::path::{Path, PathBuf};
use tracing::debug;

/// Resolve tokenizer path with deterministic fallback ordering.
///
/// Priority:
/// 1. Explicit path (if provided).
/// 2. Sibling `tokenizer.json` next to model.
/// 3. Parent-directory `tokenizer.json`.
pub fn resolve_tokenizer_with<F>(
    model_path: &Path,
    explicit_path: Option<PathBuf>,
    mut verifier: F,
) -> Result<PathBuf>
where
    F: FnMut(&Path) -> Result<bool>,
{
    if let Some(path) = explicit_path {
        debug!("Using explicit tokenizer path: {}", path.display());
        return validate_explicit_path(path);
    }

    let discovery = TokenizerDiscovery::new(model_path.to_path_buf());
    discovery.discover_with(&mut verifier)
}

#[derive(Debug, Clone)]
pub struct TokenizerDiscovery {
    model_path: PathBuf,
}

impl TokenizerDiscovery {
    #[must_use]
    pub fn new(model_path: PathBuf) -> Self {
        Self { model_path }
    }

    pub fn discover_with<F>(&self, verifier: &mut F) -> Result<PathBuf>
    where
        F: FnMut(&Path) -> Result<bool>,
    {
        debug!("Starting tokenizer auto-discovery for model: {}", self.model_path.display());

        if let Some(path) = self.check_sibling_tokenizer(verifier)? {
            debug!("Discovered sibling tokenizer: {}", path.display());
            return Ok(path);
        }

        if let Some(path) = self.check_parent_tokenizer(verifier)? {
            debug!("Discovered parent tokenizer: {}", path.display());
            return Ok(path);
        }

        Err(self.discovery_failed_error())
    }

    fn check_sibling_tokenizer<F>(&self, verifier: &mut F) -> Result<Option<PathBuf>>
    where
        F: FnMut(&Path) -> Result<bool>,
    {
        let model_dir = self.model_path.parent().unwrap_or_else(|| Path::new("."));
        let sibling_path = model_dir.join("tokenizer.json");

        debug!("Checking sibling tokenizer: {}", sibling_path.display());

        if sibling_path.exists() && sibling_path.is_file() && verifier(&sibling_path)? {
            return Ok(Some(sibling_path.canonicalize()?));
        }

        Ok(None)
    }

    fn check_parent_tokenizer<F>(&self, verifier: &mut F) -> Result<Option<PathBuf>>
    where
        F: FnMut(&Path) -> Result<bool>,
    {
        let model_dir = self.model_path.parent().unwrap_or_else(|| Path::new("."));

        if let Some(parent_dir) = model_dir.parent() {
            let parent_path = parent_dir.join("tokenizer.json");

            debug!("Checking parent tokenizer: {}", parent_path.display());

            if parent_path.exists() && parent_path.is_file() && verifier(&parent_path)? {
                return Ok(Some(parent_path.canonicalize()?));
            }
        }

        Ok(None)
    }

    fn discovery_failed_error(&self) -> anyhow::Error {
        let model_dir = self.model_path.parent().unwrap_or_else(|| Path::new("."));
        let sibling_path = model_dir.join("tokenizer.json");
        let parent_path = model_dir
            .parent()
            .map(|p| p.join("tokenizer.json"))
            .unwrap_or_else(|| PathBuf::from("N/A"));

        anyhow!(
            "Tokenizer not found for model: {}\n\
             \n\
             Tokenizer auto-discovery failed. Tried:\n\
             1. Sibling tokenizer.json: {} (not found/invalid)\n\
             2. Parent directory: {} (not found/invalid)\n\
             \n\
             Solution:\n\
             1. Download tokenizer:\n\
                cargo run -p xtask -- tokenizer --into {}\n\
             2. Provide explicit tokenizer path:\n\
                --tokenizer /path/to/tokenizer.json",
            self.model_path.display(),
            sibling_path.display(),
            parent_path.display(),
            model_dir.display(),
        )
    }
}

fn validate_explicit_path(path: PathBuf) -> Result<PathBuf> {
    if !path.exists() {
        anyhow::bail!(
            "Explicit tokenizer path does not exist: {}\n\
             \n\
             Please provide a valid tokenizer.json file path.",
            path.display()
        );
    }

    if !path.is_file() {
        anyhow::bail!("Explicit tokenizer path is not a file: {}", path.display());
    }

    path.canonicalize().context("Failed to canonicalize explicit tokenizer path")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn always_valid(_path: &Path) -> Result<bool> {
        Ok(true)
    }

    #[test]
    fn explicit_path_takes_precedence() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let model_path = temp_dir.path().join("model.gguf");
        let explicit_tokenizer = temp_dir.path().join("custom.json");
        fs::write(&model_path, b"GGUF")?;
        fs::write(&explicit_tokenizer, b"{}")?;

        let resolved =
            resolve_tokenizer_with(&model_path, Some(explicit_tokenizer.clone()), always_valid)?;
        assert_eq!(resolved, explicit_tokenizer.canonicalize()?);
        Ok(())
    }

    #[test]
    fn finds_sibling_first() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let model_dir = temp_dir.path().join("models");
        fs::create_dir(&model_dir)?;

        let model_path = model_dir.join("model.gguf");
        let sibling = model_dir.join("tokenizer.json");
        let parent = temp_dir.path().join("tokenizer.json");
        fs::write(&model_path, b"GGUF")?;
        fs::write(&sibling, b"{}")?;
        fs::write(&parent, b"{}")?;

        let resolved = resolve_tokenizer_with(&model_path, None, always_valid)?;
        assert_eq!(resolved, sibling.canonicalize()?);
        Ok(())
    }

    #[test]
    fn falls_back_to_parent() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let model_dir = temp_dir.path().join("models");
        fs::create_dir(&model_dir)?;

        let model_path = model_dir.join("model.gguf");
        let parent = temp_dir.path().join("tokenizer.json");
        fs::write(&model_path, b"GGUF")?;
        fs::write(&parent, b"{}")?;

        let resolved = resolve_tokenizer_with(&model_path, None, always_valid)?;
        assert_eq!(resolved, parent.canonicalize()?);
        Ok(())
    }

    #[test]
    fn skips_invalid_tokenizer_candidates() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let model_path = temp_dir.path().join("model.gguf");
        let sibling = temp_dir.path().join("tokenizer.json");
        fs::write(&model_path, b"GGUF")?;
        fs::write(&sibling, b"{}")?;

        let mut called = 0usize;
        let result = resolve_tokenizer_with(&model_path, None, |_path| {
            called += 1;
            Ok(false)
        });

        assert!(result.is_err());
        assert_eq!(called, 1);
        Ok(())
    }
}
