use anyhow::Result;
use std::path::{Path, PathBuf};
use tracing::debug;

mod discovery;
mod explicit_path;

pub use discovery::TokenizerDiscovery;
use explicit_path::validate_explicit_path;

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

    #[test]
    fn explicit_path_missing_returns_error() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let model_path = temp_dir.path().join("model.gguf");
        fs::write(&model_path, b"GGUF")?;
        let missing = temp_dir.path().join("does_not_exist.json");

        let result = resolve_tokenizer_with(&model_path, Some(missing.clone()), always_valid);
        let message = match result {
            Err(err) => format!("{err:#}"),
            Ok(path) => format!("unexpected success: {}", path.display()),
        };
        assert!(message.contains("does not exist"), "got: {message}");
        assert!(message.contains("does_not_exist.json"), "got: {message}");
        Ok(())
    }

    #[test]
    fn explicit_path_directory_returns_error() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let model_path = temp_dir.path().join("model.gguf");
        fs::write(&model_path, b"GGUF")?;
        let dir_as_tokenizer = temp_dir.path().join("subdir");
        fs::create_dir(&dir_as_tokenizer)?;

        let result = resolve_tokenizer_with(&model_path, Some(dir_as_tokenizer), always_valid);
        let message = match result {
            Err(err) => format!("{err:#}"),
            Ok(path) => format!("unexpected success: {}", path.display()),
        };
        assert!(message.contains("not a file"), "got: {message}");
        Ok(())
    }

    #[test]
    fn verifier_error_propagates_through_discovery() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let model_path = temp_dir.path().join("model.gguf");
        let sibling = temp_dir.path().join("tokenizer.json");
        fs::write(&model_path, b"GGUF")?;
        fs::write(&sibling, b"{}")?;

        let result =
            resolve_tokenizer_with(&model_path, None, |_| Err(anyhow::anyhow!("verifier blew up")));
        let message = match result {
            Err(err) => format!("{err:#}"),
            Ok(path) => format!("unexpected success: {}", path.display()),
        };
        assert!(message.contains("verifier blew up"));
        Ok(())
    }

    #[test]
    fn falls_back_to_parent_when_sibling_rejected() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let model_dir = temp_dir.path().join("models");
        fs::create_dir(&model_dir)?;

        let model_path = model_dir.join("model.gguf");
        let sibling = model_dir.join("tokenizer.json");
        let parent = temp_dir.path().join("tokenizer.json");
        fs::write(&model_path, b"GGUF")?;
        fs::write(&sibling, b"sibling")?;
        fs::write(&parent, b"parent")?;

        let mut seen: Vec<PathBuf> = Vec::new();
        let resolved = resolve_tokenizer_with(&model_path, None, |path| {
            seen.push(path.to_path_buf());
            // Reject the sibling; accept the parent.
            Ok(path != sibling)
        })?;

        assert_eq!(resolved, parent.canonicalize()?);
        assert_eq!(seen.len(), 2, "verifier should be called for sibling then parent");
        assert_eq!(seen[0], sibling);
        assert_eq!(seen[1], parent);
        Ok(())
    }

    #[test]
    fn discovery_error_message_includes_attempted_paths_and_solutions() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let model_dir = temp_dir.path().join("models");
        fs::create_dir(&model_dir)?;
        let model_path = model_dir.join("model.gguf");
        fs::write(&model_path, b"GGUF")?;

        let result = resolve_tokenizer_with(&model_path, None, always_valid);
        let message = match result {
            Err(err) => format!("{err:#}"),
            Ok(path) => format!("unexpected success: {}", path.display()),
        };
        assert!(message.contains("model.gguf"));
        assert!(message.contains("Sibling tokenizer.json"));
        assert!(message.contains("Parent directory"));
        assert!(message.contains("cargo run -p xtask -- tokenizer --into"));
        assert!(message.contains("--tokenizer /path/to/tokenizer.json"));
        Ok(())
    }

    #[test]
    fn discovery_struct_usable_directly() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let model_path = temp_dir.path().join("model.gguf");
        let sibling = temp_dir.path().join("tokenizer.json");
        fs::write(&model_path, b"GGUF")?;
        fs::write(&sibling, b"{}")?;

        let discovery = TokenizerDiscovery::new(model_path);
        let resolved = discovery.discover_with(&mut always_valid)?;
        assert_eq!(resolved, sibling.canonicalize()?);
        Ok(())
    }

    #[test]
    fn sibling_not_found_skips_to_parent() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let model_dir = temp_dir.path().join("models");
        fs::create_dir(&model_dir)?;
        let model_path = model_dir.join("model.gguf");
        let parent = temp_dir.path().join("tokenizer.json");
        fs::write(&model_path, b"GGUF")?;
        fs::write(&parent, b"{}")?;
        // Note: no sibling tokenizer.json file.

        let mut verifier_call_count = 0usize;
        let resolved = resolve_tokenizer_with(&model_path, None, |_| {
            verifier_call_count += 1;
            Ok(true)
        })?;

        assert_eq!(resolved, parent.canonicalize()?);
        // Sibling does not exist, so verifier is only called once (for parent).
        assert_eq!(verifier_call_count, 1);
        Ok(())
    }
}
