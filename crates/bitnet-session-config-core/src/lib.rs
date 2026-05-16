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

    fn valid_cfg() -> SessionConfig {
        SessionConfig {
            model_path: "m.gguf".into(),
            tokenizer_path: "t.json".into(),
            backend: "cpu".into(),
            max_context: 2048,
            seed: None,
        }
    }

    #[test]
    fn defaults_match_engine_expectations() {
        let cfg = SessionConfig::default();
        assert_eq!(cfg.backend, "cpu");
        assert_eq!(cfg.max_context, 2048);
        assert!(cfg.model_path.is_empty());
        assert!(cfg.tokenizer_path.is_empty());
        assert_eq!(cfg.seed, None);
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

    #[test]
    fn validate_accepts_minimal_valid_config() {
        assert_eq!(valid_cfg().validate(), Ok(()));
    }

    #[test]
    fn validate_accepts_all_known_backends() {
        for backend in VALID_BACKENDS {
            let mut cfg = valid_cfg();
            cfg.backend = (*backend).to_string();
            assert_eq!(cfg.validate(), Ok(()), "backend {backend} should validate");
        }
    }

    #[test]
    fn validate_accepts_minimum_context_window() {
        let mut cfg = valid_cfg();
        cfg.max_context = 1;
        assert_eq!(cfg.validate(), Ok(()));
    }

    #[test]
    fn validate_reports_model_path_before_other_errors() {
        // All fields invalid: model_path is reported first.
        let cfg = SessionConfig {
            model_path: String::new(),
            tokenizer_path: String::new(),
            backend: "bogus".into(),
            max_context: 0,
            seed: None,
        };
        assert_eq!(cfg.validate(), Err(ConfigError::EmptyModelPath));
    }

    #[test]
    fn validate_reports_tokenizer_path_before_backend_or_context() {
        let cfg = SessionConfig {
            model_path: "m.gguf".into(),
            tokenizer_path: String::new(),
            backend: "bogus".into(),
            max_context: 0,
            seed: None,
        };
        assert_eq!(cfg.validate(), Err(ConfigError::EmptyTokenizerPath));
    }

    #[test]
    fn validate_reports_backend_before_context() {
        let cfg = SessionConfig {
            model_path: "m.gguf".into(),
            tokenizer_path: "t.json".into(),
            backend: "bogus".into(),
            max_context: 0,
            seed: None,
        };
        assert_eq!(cfg.validate(), Err(ConfigError::UnsupportedBackend("bogus".into())));
    }

    #[test]
    fn valid_backends_constant_matches_documented_set() {
        assert_eq!(VALID_BACKENDS, &["cpu", "cuda", "gpu", "ffi"]);
    }

    #[test]
    fn config_error_display_messages() {
        assert_eq!(
            ConfigError::EmptyModelPath.to_string(),
            "model_path must not be empty"
        );
        assert_eq!(
            ConfigError::EmptyTokenizerPath.to_string(),
            "tokenizer_path must not be empty"
        );
        assert_eq!(
            ConfigError::UnsupportedBackend("foo".into()).to_string(),
            "unsupported backend: \"foo\""
        );
        assert_eq!(
            ConfigError::ZeroContextWindow.to_string(),
            "max_context must be greater than zero"
        );
    }

    #[test]
    fn config_error_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(ConfigError::EmptyModelPath);
        assert!(err.source().is_none());
    }

    #[test]
    fn config_error_eq_and_clone() {
        let a = ConfigError::UnsupportedBackend("x".into());
        let b = a.clone();
        assert_eq!(a, b);
        assert_ne!(a, ConfigError::UnsupportedBackend("y".into()));
        assert_ne!(ConfigError::EmptyModelPath, ConfigError::EmptyTokenizerPath);
    }
}
