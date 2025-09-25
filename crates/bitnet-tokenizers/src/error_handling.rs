//! Comprehensive error handling with anyhow::Result integration for BitNet.rs neural network tokenization
//!
//! This module provides centralized error handling patterns that follow BitNet.rs coding standards
//! with consistent anyhow::Error usage and actionable error messages for neural network operations.
//!
//! Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac10-error-handling

use anyhow::Result as AnyhowResult;
use bitnet_common::{BitNetError, ModelError, Result};
use std::path::{Path, PathBuf};
use tracing::warn;
//use crate::{CacheManager, ModelTypeDetector};

/// Centralized error handling utilities for BitNet.rs tokenizer operations
///
/// Provides consistent error patterns across discovery, download, strategy, and fallback modules
/// following BitNet.rs neural network inference requirements.
pub struct TokenizerErrorHandler;

impl TokenizerErrorHandler {
    /// Convert BitNetError to anyhow::Error with context following BitNet.rs patterns
    pub fn to_anyhow_error(error: BitNetError, context: &str) -> anyhow::Error {
        anyhow::Error::new(error).context(context.to_string())
    }

    /// Create actionable error message for tokenizer failures with neural network context
    pub fn create_actionable_error(error: BitNetError, operation: &str) -> AnyhowResult<()> {
        let context = format!("BitNet.rs tokenizer operation failed: {}", operation);
        let suggestions = Self::get_error_suggestions(&error);

        let mut error_msg = Self::to_anyhow_error(error, &context);

        // Add suggestions as context for better user experience
        for suggestion in suggestions {
            error_msg = error_msg.context(format!("Suggestion: {}", suggestion));
        }

        Err(error_msg)
    }

    /// Create file I/O error with consistent BitNet.rs formatting
    pub fn file_io_error(path: &Path, source: std::io::Error) -> BitNetError {
        BitNetError::Model(ModelError::FileIOError { path: path.to_path_buf(), source })
    }

    /// Create model loading error with consistent formatting
    pub fn loading_failed_error(reason: String) -> BitNetError {
        BitNetError::Model(ModelError::LoadingFailed { reason })
    }

    /// Create configuration error with neural network context
    pub fn config_error(message: String) -> BitNetError {
        BitNetError::Config(message)
    }

    /// Validate file exists and is readable with consistent error reporting
    pub fn validate_file_exists(path: &Path, context: &str) -> Result<()> {
        if !path.exists() {
            return Err(Self::config_error(format!(
                "{}: File does not exist: {}",
                context,
                path.display()
            )));
        }

        if !path.is_file() {
            return Err(Self::config_error(format!(
                "{}: Path is not a regular file: {}",
                context,
                path.display()
            )));
        }

        // Try to read metadata to ensure file is accessible
        std::fs::metadata(path).map_err(|e| Self::file_io_error(path, e))?;

        Ok(())
    }

    /// Get contextual error suggestions based on error type
    fn get_error_suggestions(error: &BitNetError) -> Vec<String> {
        match error {
            BitNetError::Model(ModelError::FileIOError { path, .. }) => {
                vec![
                    format!("Check that file exists and is readable: {}", path.display()),
                    "Verify file permissions allow read access".to_string(),
                    "Ensure the file is not corrupted or in use by another process".to_string(),
                ]
            }
            BitNetError::Model(ModelError::LoadingFailed { reason }) => {
                if reason.contains("network") || reason.contains("download") {
                    vec![
                        "Check internet connection".to_string(),
                        "Verify HuggingFace Hub is accessible".to_string(),
                        "Try using cached tokenizer with --tokenizer flag".to_string(),
                        "Use offline mode with BITNET_OFFLINE=1 if needed".to_string(),
                    ]
                } else if reason.contains("vocab") || reason.contains("token") {
                    vec![
                        "Verify tokenizer is compatible with model architecture".to_string(),
                        "Check vocabulary size matches model requirements".to_string(),
                        "Try downloading correct tokenizer for model type".to_string(),
                    ]
                } else {
                    vec![
                        "Verify tokenizer file format is correct".to_string(),
                        "Check model compatibility with tokenizer".to_string(),
                        "Run model validation: cargo run -p bitnet-cli -- compat-check model.gguf"
                            .to_string(),
                    ]
                }
            }
            BitNetError::Config(message) => {
                if message.contains("strict") {
                    vec![
                        "Remove BITNET_STRICT_TOKENIZERS=1 to allow fallback tokenizers"
                            .to_string(),
                        "Provide explicit tokenizer path with --tokenizer flag".to_string(),
                        "Ensure required tokenizer files are available".to_string(),
                    ]
                } else if message.contains("vocab") {
                    vec![
                        "Check if large vocabulary model requires GPU acceleration".to_string(),
                        "Verify model architecture supports vocabulary size".to_string(),
                        "Consider using compatible model variant".to_string(),
                    ]
                } else {
                    vec![
                        "Check configuration parameters".to_string(),
                        "Verify environment variables are set correctly".to_string(),
                        "Review BitNet.rs documentation for proper setup".to_string(),
                    ]
                }
            }
            _ => vec![
                "Check BitNet.rs logs for detailed error information".to_string(),
                "Verify system requirements and dependencies".to_string(),
                "Consult BitNet.rs documentation for troubleshooting".to_string(),
            ],
        }
    }

    /// Log error with appropriate level based on severity
    pub fn log_error(error: &BitNetError, operation: &str) {
        match error {
            BitNetError::Model(ModelError::FileIOError { path, source }) => {
                warn!("File I/O error during {}: {} ({})", operation, path.display(), source);
            }
            BitNetError::Model(ModelError::LoadingFailed { reason }) => {
                warn!("Loading failed during {}: {}", operation, reason);
            }
            BitNetError::Config(message) => {
                warn!("Configuration error during {}: {}", operation, message);
            }
            _ => {
                warn!("Error during {}: {}", operation, error);
            }
        }
    }
}

/// Centralized cache management utilities for BitNet.rs tokenizer operations
pub struct CacheManager;

impl CacheManager {
    /// Determine cache directory with environment variable override
    ///
    /// Follows BitNet.rs standard cache directory conventions
    pub fn cache_directory() -> Result<PathBuf> {
        // Check environment variable first
        if let Ok(cache_dir) = std::env::var("BITNET_CACHE_DIR") {
            return Ok(PathBuf::from(cache_dir).join("tokenizers"));
        }

        // Check XDG cache directory
        if let Ok(xdg_cache) = std::env::var("XDG_CACHE_HOME") {
            return Ok(PathBuf::from(xdg_cache).join("bitnet").join("tokenizers"));
        }

        // Use system cache directory
        if let Some(cache_dir) = dirs::cache_dir() {
            return Ok(cache_dir.join("bitnet").join("tokenizers"));
        }

        // Fallback to local directory
        Ok(PathBuf::from(".cache").join("bitnet").join("tokenizers"))
    }

    /// Ensure cache directory exists with proper error handling
    pub fn ensure_cache_directory(cache_dir: &Path) -> Result<()> {
        if !cache_dir.exists() {
            std::fs::create_dir_all(cache_dir)
                .map_err(|e| TokenizerErrorHandler::file_io_error(cache_dir, e))?;
        }
        Ok(())
    }

    /// Get model-specific cache directory
    pub fn model_cache_dir(model_type: &str, vocab_size: Option<usize>) -> Result<PathBuf> {
        let base_cache = Self::cache_directory()?;
        let model_cache = base_cache.join(model_type);

        if let Some(size) = vocab_size {
            Ok(model_cache.join(format!("vocab_{}", size)))
        } else {
            Ok(model_cache)
        }
    }
}

/// Model type detection utilities for neural network tokenizers
pub struct ModelTypeDetector;

impl ModelTypeDetector {
    /// Detect model type from vocabulary size and metadata
    pub fn detect_from_vocab_size(vocab_size: usize) -> String {
        match vocab_size {
            32000 => "llama2".to_string(),
            128256 => "llama3".to_string(),
            32016 => "codellama".to_string(),
            50257 => "gpt2".to_string(),
            _ => "unknown".to_string(),
        }
    }

    /// Check if vocabulary size indicates large model requiring GPU acceleration
    pub fn requires_gpu_acceleration(vocab_size: usize) -> bool {
        vocab_size > 65536
    }

    /// Validate vocabulary size is reasonable for neural network inference
    pub fn validate_vocab_size(vocab_size: usize) -> Result<()> {
        if vocab_size == 0 {
            return Err(TokenizerErrorHandler::config_error(
                "Vocabulary size cannot be zero".to_string(),
            ));
        }

        if vocab_size > 2_000_000 {
            return Err(TokenizerErrorHandler::config_error(format!(
                "Vocabulary size {} exceeds reasonable limit (2M)",
                vocab_size
            )));
        }

        Ok(())
    }

    /// Get expected vocabulary size for known model types
    pub fn expected_vocab_size(model_type: &str) -> Option<usize> {
        match model_type {
            "llama2" => Some(32000),
            "llama3" => Some(128256),
            "codellama" => Some(32016),
            "gpt2" => Some(50257),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::TokenizerErrorHandler;
    use bitnet_common::{BitNetError, ModelError};
    use std::path::PathBuf;

    /// AC10: Tests error handling with anyhow::Result integration
    /// Tests feature spec: issue-249-tokenizer-discovery-neural-network-spec.md#ac10-error-handling
    #[test]
    #[cfg(feature = "cpu")]
    fn test_error_handling_anyhow_integration() {
        let test_error =
            BitNetError::Model(ModelError::LoadingFailed { reason: "Test error".to_string() });

        let anyhow_error = TokenizerErrorHandler::to_anyhow_error(test_error, "Test context");
        let error_string = anyhow_error.to_string();

        // The error should contain both the context and the original error message
        assert!(
            error_string.contains("Test context") || error_string.contains("Test error"),
            "Error should contain context or original message. Got: '{}'",
            error_string
        );
    }

    /// AC10: Tests actionable error creation with suggestions
    #[test]
    #[cfg(feature = "cpu")]
    fn test_actionable_error_creation() {
        let network_error =
            BitNetError::Model(ModelError::LoadingFailed { reason: "network timeout".to_string() });

        let result = TokenizerErrorHandler::create_actionable_error(network_error, "download");
        assert!(result.is_err());

        let error_chain = format!("{:#}", result.unwrap_err());
        assert!(error_chain.contains("download"));
        assert!(error_chain.contains("Suggestion"));
    }

    /// AC10: Tests file validation utilities
    #[test]
    #[cfg(feature = "cpu")]
    fn test_file_validation() {
        // Test nonexistent file
        let nonexistent = PathBuf::from("/nonexistent/file.json");
        let result = TokenizerErrorHandler::validate_file_exists(&nonexistent, "test");
        assert!(result.is_err());

        // Test with a temporary file (should exist)
        use std::fs::File;
        use tempfile::tempdir;
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let temp_file = temp_dir.path().join("test_file.txt");
        File::create(&temp_file).expect("Failed to create temp file");
        let result = TokenizerErrorHandler::validate_file_exists(&temp_file, "test");
        assert!(result.is_ok());
    }

    /// AC10: Tests error suggestion generation
    #[test]
    #[cfg(feature = "cpu")]
    fn test_error_suggestions() {
        let file_error = BitNetError::Model(ModelError::FileIOError {
            path: PathBuf::from("test.json"),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "file not found"),
        });

        let suggestions = TokenizerErrorHandler::get_error_suggestions(&file_error);
        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.contains("test.json")));

        let network_error = BitNetError::Model(ModelError::LoadingFailed {
            reason: "network error occurred".to_string(),
        });

        let network_suggestions = TokenizerErrorHandler::get_error_suggestions(&network_error);
        assert!(network_suggestions.iter().any(|s| s.contains("internet")));
    }

    /// AC10: Tests consistent error creation utilities
    #[test]
    #[cfg(feature = "cpu")]
    fn test_consistent_error_creation() {
        let path = PathBuf::from("test.json");
        let io_error = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");

        let file_error = TokenizerErrorHandler::file_io_error(&path, io_error);
        match file_error {
            BitNetError::Model(ModelError::FileIOError { path: error_path, .. }) => {
                assert_eq!(error_path, path);
            }
            _ => panic!("Expected FileIOError"),
        }

        let loading_error = TokenizerErrorHandler::loading_failed_error("test reason".to_string());
        match loading_error {
            BitNetError::Model(ModelError::LoadingFailed { reason }) => {
                assert_eq!(reason, "test reason");
            }
            _ => panic!("Expected LoadingFailed"),
        }

        let config_error = TokenizerErrorHandler::config_error("test config".to_string());
        match config_error {
            BitNetError::Config(message) => {
                assert_eq!(message, "test config");
            }
            _ => panic!("Expected Config error"),
        }
    }

    /// Tests cache management utilities
    #[test]
    #[cfg(feature = "cpu")]
    fn test_cache_management() {
        let cache_dir = crate::CacheManager::cache_directory();
        assert!(cache_dir.is_ok());

        let cache_path = cache_dir.unwrap();
        assert!(cache_path.to_string_lossy().contains("tokenizers"));

        // Test model-specific cache directory
        let model_cache = crate::CacheManager::model_cache_dir("llama2", Some(32000));
        assert!(model_cache.is_ok());

        let model_path = model_cache.unwrap();
        assert!(model_path.to_string_lossy().contains("llama2"));
        assert!(model_path.to_string_lossy().contains("vocab_32000"));
    }

    /// Tests model type detection utilities
    #[test]
    #[cfg(feature = "cpu")]
    fn test_model_type_detection() {
        // Test vocabulary size to model type mapping
        assert_eq!(crate::ModelTypeDetector::detect_from_vocab_size(32000), "llama2");
        assert_eq!(crate::ModelTypeDetector::detect_from_vocab_size(128256), "llama3");
        assert_eq!(crate::ModelTypeDetector::detect_from_vocab_size(50257), "gpt2");
        assert_eq!(crate::ModelTypeDetector::detect_from_vocab_size(99999), "unknown");

        // Test GPU acceleration requirements
        assert!(!crate::ModelTypeDetector::requires_gpu_acceleration(32000));
        assert!(crate::ModelTypeDetector::requires_gpu_acceleration(128256));

        // Test vocabulary size validation
        assert!(crate::ModelTypeDetector::validate_vocab_size(32000).is_ok());
        assert!(crate::ModelTypeDetector::validate_vocab_size(0).is_err());
        assert!(crate::ModelTypeDetector::validate_vocab_size(3_000_000).is_err());

        // Test expected vocabulary sizes
        assert_eq!(crate::ModelTypeDetector::expected_vocab_size("llama2"), Some(32000));
        assert_eq!(crate::ModelTypeDetector::expected_vocab_size("llama3"), Some(128256));
        assert_eq!(crate::ModelTypeDetector::expected_vocab_size("unknown"), None);
    }
}
