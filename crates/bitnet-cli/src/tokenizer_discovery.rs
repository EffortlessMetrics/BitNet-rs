//! Tokenizer auto-discovery for BitNet-rs CLI
//!
//! This module implements automatic tokenizer discovery with a fallback chain:
//! 1. Explicit --tokenizer flag (highest priority)
//! 2. GGUF embedded tokenizer metadata
//! 3. Sibling tokenizer.json (same directory as model)
//! 4. Parent directory tokenizer.json (one level up)
//!
//! AC:ID llama3-tokenizer-fetching-spec.md#ac2-cli-auto-discovery

use anyhow::{Context, Result, anyhow};
use std::path::{Path, PathBuf};
use tracing::debug;

/// Tokenizer discovery engine
///
/// AC:ID llama3-tokenizer-api-contracts.md#cli-auto-discovery-v1
pub struct TokenizerDiscovery {
    model_path: PathBuf,
}

impl TokenizerDiscovery {
    /// Create discovery engine from model path
    ///
    /// Preconditions:
    /// - model_path must exist and be readable
    /// - model_path must point to a file (not directory)
    pub fn new(model_path: PathBuf) -> Self {
        Self { model_path }
    }

    /// Discover tokenizer via fallback chain
    ///
    /// Discovery order:
    /// 1. GGUF embedded tokenizer (tokenizer.ggml.model metadata)
    /// 2. Sibling tokenizer.json (same directory as model)
    /// 3. Parent directory tokenizer.json (one level up)
    ///
    /// Returns: Absolute path to valid tokenizer
    /// Errors: If no tokenizer found in any location
    pub fn discover(&self) -> Result<PathBuf> {
        debug!("Starting tokenizer auto-discovery for model: {}", self.model_path.display());

        // 1. Check GGUF embedded tokenizer
        if let Some(path) = self.check_embedded_tokenizer()? {
            debug!("Discovered embedded GGUF tokenizer");
            return Ok(path);
        }

        // 2. Check sibling tokenizer.json
        if let Some(path) = self.check_sibling_tokenizer()? {
            debug!("Discovered sibling tokenizer: {}", path.display());
            return Ok(path);
        }

        // 3. Check parent directory
        if let Some(path) = self.check_parent_tokenizer()? {
            debug!("Discovered parent tokenizer: {}", path.display());
            return Ok(path);
        }

        // No tokenizer found - return actionable error
        Err(self.discovery_failed_error())
    }

    /// Check for GGUF embedded tokenizer
    ///
    /// Returns: Some(PathBuf) if embedded tokenizer found and valid, None otherwise
    fn check_embedded_tokenizer(&self) -> Result<Option<PathBuf>> {
        // TODO: Implement GGUF embedded tokenizer extraction
        // For now, return None as GGUF embedded tokenizers are not yet supported
        debug!("Checking GGUF embedded tokenizer: Not present");
        Ok(None)
    }

    /// Check for sibling tokenizer.json
    ///
    /// Returns: Some(PathBuf) if sibling exists and is valid, None otherwise
    fn check_sibling_tokenizer(&self) -> Result<Option<PathBuf>> {
        let model_dir = self.model_path.parent().unwrap_or_else(|| Path::new("."));
        let sibling_path = model_dir.join("tokenizer.json");

        debug!("Checking sibling tokenizer: {}", sibling_path.display());

        if sibling_path.exists() && sibling_path.is_file() {
            // Verify it's a valid tokenizer
            if self.verify_tokenizer(&sibling_path)? {
                // Return absolute path
                return Ok(Some(sibling_path.canonicalize()?));
            }
        }

        Ok(None)
    }

    /// Check for parent directory tokenizer.json
    ///
    /// Returns: Some(PathBuf) if parent tokenizer exists and is valid, None otherwise
    fn check_parent_tokenizer(&self) -> Result<Option<PathBuf>> {
        let model_dir = self.model_path.parent().unwrap_or_else(|| Path::new("."));

        if let Some(parent_dir) = model_dir.parent() {
            let parent_path = parent_dir.join("tokenizer.json");

            debug!("Checking parent tokenizer: {}", parent_path.display());

            if parent_path.exists() && parent_path.is_file() {
                // Verify it's a valid tokenizer
                if self.verify_tokenizer(&parent_path)? {
                    // Return absolute path
                    return Ok(Some(parent_path.canonicalize()?));
                }
            }
        }

        Ok(None)
    }

    /// Verify tokenizer file is valid
    ///
    /// Returns: true if valid, false otherwise
    fn verify_tokenizer(&self, path: &Path) -> Result<bool> {
        // Try to load tokenizer using bitnet-tokenizers
        match bitnet_tokenizers::loader::load_tokenizer(path) {
            Ok(_) => {
                debug!("Tokenizer validation passed: {}", path.display());
                Ok(true)
            }
            Err(e) => {
                debug!("Tokenizer validation failed for {}: {}", path.display(), e);
                Ok(false) // Return false instead of error for discovery chain
            }
        }
    }

    /// Generate actionable error when discovery fails
    ///
    /// AC:ID llama3-tokenizer-api-contracts.md#error-message-contract
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
             1. GGUF embedded tokenizer: Not present\n\
             2. Sibling tokenizer.json: {} (not found)\n\
             3. Parent directory: {} (not found)\n\
             \n\
             Solution:\n\
             1. Download LLaMA-3 tokenizer:\n\
                cargo run -p xtask -- tokenizer --into {}\n\
             2. Provide explicit tokenizer path:\n\
                --tokenizer /path/to/tokenizer.json\n\
             \n\
             For more help, see: docs/quickstart.md#tokenizer-setup",
            self.model_path.display(),
            sibling_path.display(),
            parent_path.display(),
            model_dir.display()
        )
    }
}

/// Resolve tokenizer path with auto-discovery
///
/// This is the main entry point for tokenizer resolution.
///
/// Priority:
/// 1. Explicit path (if provided) - highest priority
/// 2. Auto-discovery chain (GGUF → sibling → parent)
///
/// AC:ID llama3-tokenizer-api-contracts.md#cli-auto-discovery-v1
pub fn resolve_tokenizer(model_path: &Path, explicit_path: Option<PathBuf>) -> Result<PathBuf> {
    // Priority 1: Explicit path takes precedence
    if let Some(path) = explicit_path {
        debug!("Using explicit tokenizer path: {}", path.display());

        // Verify explicit path exists
        if !path.exists() {
            anyhow::bail!(
                "Explicit tokenizer path does not exist: {}\n\
                 \n\
                 Please provide a valid tokenizer.json file path.",
                path.display()
            );
        }

        // Verify it's a file
        if !path.is_file() {
            anyhow::bail!("Explicit tokenizer path is not a file: {}", path.display());
        }

        // Return absolute path
        return path.canonicalize().context("Failed to canonicalize explicit tokenizer path");
    }

    // Priority 2: Auto-discovery
    debug!("Starting tokenizer auto-discovery (no explicit path provided)");
    let discovery = TokenizerDiscovery::new(model_path.to_path_buf());
    discovery.discover()
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
        // Minimal GGUF header for testing
        b"GGUF\x03\x00\x00\x00".to_vec()
    }

    #[test]
    fn test_explicit_path_takes_precedence() -> Result<()> {
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
    fn test_discover_sibling_tokenizer() -> Result<()> {
        let temp_dir = TempDir::new()?;

        let model_path = temp_dir.path().join("model.gguf");
        let sibling_tokenizer = temp_dir.path().join("tokenizer.json");

        fs::write(&model_path, create_mock_gguf())?;
        fs::write(&sibling_tokenizer, create_mock_tokenizer())?;

        let result = resolve_tokenizer(&model_path, None)?;

        assert_eq!(result.canonicalize()?, sibling_tokenizer.canonicalize()?);

        Ok(())
    }

    #[test]
    fn test_discover_parent_tokenizer() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let model_dir = temp_dir.path().join("models");
        fs::create_dir(&model_dir)?;

        let model_path = model_dir.join("model.gguf");
        let parent_tokenizer = temp_dir.path().join("tokenizer.json");

        fs::write(&model_path, create_mock_gguf())?;
        fs::write(&parent_tokenizer, create_mock_tokenizer())?;

        let result = resolve_tokenizer(&model_path, None)?;

        assert_eq!(result.canonicalize()?, parent_tokenizer.canonicalize()?);

        Ok(())
    }

    #[test]
    fn test_fail_with_clear_error() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("model.gguf");
        fs::write(&model_path, create_mock_gguf()).unwrap();

        let result = resolve_tokenizer(&model_path, None);

        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();

        assert!(error_msg.contains("Tokenizer not found"));
        assert!(error_msg.contains("cargo run -p xtask -- tokenizer"));
    }

    #[test]
    fn test_discovery_chain_order() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let model_dir = temp_dir.path().join("models");
        fs::create_dir(&model_dir)?;

        let model_path = model_dir.join("model.gguf");
        fs::write(&model_path, create_mock_gguf())?;

        // Test 1: No tokenizer anywhere - should fail
        let result_none = resolve_tokenizer(&model_path, None);
        assert!(result_none.is_err());

        // Test 2: Parent tokenizer only
        let parent_tokenizer = temp_dir.path().join("tokenizer.json");
        fs::write(&parent_tokenizer, create_mock_tokenizer())?;

        let result_parent = resolve_tokenizer(&model_path, None)?;
        assert_eq!(result_parent.canonicalize()?, parent_tokenizer.canonicalize()?);

        // Test 3: Sibling tokenizer (should take precedence over parent)
        let sibling_tokenizer = model_dir.join("tokenizer.json");
        fs::write(&sibling_tokenizer, create_mock_tokenizer())?;

        let result_sibling = resolve_tokenizer(&model_path, None)?;
        assert_eq!(result_sibling.canonicalize()?, sibling_tokenizer.canonicalize()?);

        Ok(())
    }

    #[test]
    fn test_discovery_returns_absolute_paths() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let model_path = temp_dir.path().join("model.gguf");
        let tokenizer_path = temp_dir.path().join("tokenizer.json");

        fs::write(&model_path, create_mock_gguf())?;
        fs::write(&tokenizer_path, create_mock_tokenizer())?;

        let result = resolve_tokenizer(&model_path, None)?;

        assert!(result.is_absolute());
        assert_eq!(result, tokenizer_path.canonicalize()?);

        Ok(())
    }
}
