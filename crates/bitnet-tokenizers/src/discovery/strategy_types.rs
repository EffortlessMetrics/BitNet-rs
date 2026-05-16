//! Tokenizer resolution strategy variants.

use crate::Tokenizer;
use std::path::PathBuf;
use std::sync::Arc;

use super::TokenizerDownloadInfo;

/// Comprehensive tokenizer resolution strategy for neural network models
#[derive(Clone)]
pub enum TokenizerStrategy {
    /// User explicitly specified tokenizer path
    Exact(PathBuf),
    /// Auto-discovered compatible tokenizer in model directory
    Discovered(PathBuf),
    /// Smart download required from HuggingFace Hub
    NeedsDownload(TokenizerDownloadInfo),
    /// GGUF file contains embedded tokenizer data
    EmbeddedGguf(Arc<dyn Tokenizer>),
    /// Mock tokenizer for testing (non-strict mode only)
    Mock,
}

impl TokenizerStrategy {
    /// Check if strategy requires network access
    pub fn requires_network(&self) -> bool {
        matches!(self, TokenizerStrategy::NeedsDownload(_))
    }

    /// Check if strategy uses cached resources
    pub fn uses_cache(&self) -> bool {
        matches!(self, TokenizerStrategy::Discovered(_) | TokenizerStrategy::NeedsDownload(_))
    }

    /// Get description for logging and error messages
    pub fn description(&self) -> &'static str {
        match self {
            TokenizerStrategy::Exact(_) => "user-specified tokenizer",
            TokenizerStrategy::Discovered(_) => "auto-discovered tokenizer",
            TokenizerStrategy::NeedsDownload(_) => "smart download required",
            TokenizerStrategy::EmbeddedGguf(_) => "GGUF-embedded tokenizer",
            TokenizerStrategy::Mock => "mock tokenizer (testing only)",
        }
    }
}
