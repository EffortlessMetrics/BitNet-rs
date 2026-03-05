//! Reusable session configuration contracts and validation helpers.

use serde::{Deserialize, Serialize};

/// Top-level configuration for creating an inference session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Filesystem path to the GGUF model file.
    pub model_path: String,
    /// Filesystem path to the tokenizer JSON file.
    pub tokenizer_path: String,
    /// Backend identifier (e.g. `"cpu"`, `"cuda"`, `"ffi"`).
    pub backend: String,
    /// Maximum context window in tokens (prompt + generation).
    pub max_context: usize,
    /// Optional random seed for reproducible sessions.
    pub seed: Option<u64>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            tokenizer_path: String::new(),
            backend: "cpu".to_string(),
            max_context: 2048,
            seed: None,
        }
    }
}

/// Accepted backend identifiers for [`SessionConfig`].
pub const VALID_BACKENDS: &[&str] = &["cpu", "cuda", "gpu", "ffi"];

/// Error returned by [`SessionConfig::validate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// `model_path` field is empty.
    EmptyModelPath,
    /// `tokenizer_path` field is empty.
    EmptyTokenizerPath,
    /// `backend` is not one of the recognised identifiers.
    UnsupportedBackend(String),
    /// `max_context` is zero; at least one token of context is required.
    ZeroContextWindow,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyModelPath => write!(f, "model_path must not be empty"),
            Self::EmptyTokenizerPath => write!(f, "tokenizer_path must not be empty"),
            Self::UnsupportedBackend(b) => write!(f, "unsupported backend: {b:?}"),
            Self::ZeroContextWindow => write!(f, "max_context must be greater than zero"),
        }
    }
}

impl std::error::Error for ConfigError {}

impl SessionConfig {
    /// Validate the configuration, returning the first error found.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] describing the first invalid field.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.model_path.is_empty() {
            return Err(ConfigError::EmptyModelPath);
        }
        if self.tokenizer_path.is_empty() {
            return Err(ConfigError::EmptyTokenizerPath);
        }
        if !VALID_BACKENDS.contains(&self.backend.as_str()) {
            return Err(ConfigError::UnsupportedBackend(self.backend.clone()));
        }
        if self.max_context == 0 {
            return Err(ConfigError::ZeroContextWindow);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_engine_expectations() {
        let cfg = SessionConfig::default();
        assert_eq!(cfg.backend, "cpu");
        assert_eq!(cfg.max_context, 2048);
        assert!(cfg.model_path.is_empty());
        assert!(cfg.tokenizer_path.is_empty());
    }

    #[test]
    fn validate_rejects_invalid_fields() {
        let mut cfg = SessionConfig::default();
        assert_eq!(cfg.validate(), Err(ConfigError::EmptyModelPath));

        cfg.model_path = "m.gguf".into();
        assert_eq!(cfg.validate(), Err(ConfigError::EmptyTokenizerPath));

        cfg.tokenizer_path = "t.json".into();
        cfg.backend = "bad".into();
        assert_eq!(cfg.validate(), Err(ConfigError::UnsupportedBackend("bad".into())));

        cfg.backend = "cpu".into();
        cfg.max_context = 0;
        assert_eq!(cfg.validate(), Err(ConfigError::ZeroContextWindow));
    }
}
