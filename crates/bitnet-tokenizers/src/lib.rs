//! Tokenization support for BitNet models

pub mod loader;
mod mock;

use bitnet_common::Result;
use std::path::Path;
use std::sync::Arc;

pub use loader::load_tokenizer;
pub use mock::MockTokenizer;

/// Tokenizer trait
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>>;
    fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String>;
    fn vocab_size(&self) -> usize;
    fn eos_token_id(&self) -> Option<u32>;
    fn pad_token_id(&self) -> Option<u32>;
}

/// Basic tokenizer implementation
pub struct BasicTokenizer {
    vocab_size: usize,
    eos_token_id: Option<u32>,
    pad_token_id: Option<u32>,
}

impl BasicTokenizer {
    pub fn new() -> Self {
        Self {
            vocab_size: 50257, // GPT-2 vocab size
            eos_token_id: Some(50256),
            pad_token_id: None,
        }
    }

    pub fn with_config(
        vocab_size: usize,
        eos_token_id: Option<u32>,
        pad_token_id: Option<u32>,
    ) -> Self {
        Self { vocab_size, eos_token_id, pad_token_id }
    }
}

impl Default for BasicTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for BasicTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        // Simple word-based tokenization for testing
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut tokens: Vec<u32> = words.iter().enumerate().map(|(i, _)| i as u32).collect();

        // Add special tokens if requested
        if add_special_tokens {
            if let Some(eos_id) = self.eos_token_id {
                tokens.push(eos_id);
            }
        }

        Ok(tokens)
    }

    fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String> {
        if tokens.is_empty() {
            return Ok(String::new());
        }

        let mut filtered_tokens = tokens.to_vec();

        // Filter special tokens if requested
        if skip_special_tokens {
            if let Some(eos_id) = self.eos_token_id {
                filtered_tokens.retain(|&token| token != eos_id);
            }
            if let Some(pad_id) = self.pad_token_id {
                filtered_tokens.retain(|&token| token != pad_id);
            }
        }

        // Handle case where all tokens were filtered out
        if filtered_tokens.is_empty() {
            return Ok(String::new());
        }

        // Simple placeholder implementation - in real tokenizer this would map back to text
        Ok(format!("Generated text from {} tokens", filtered_tokens.len()))
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    fn pad_token_id(&self) -> Option<u32> {
        self.pad_token_id
    }
}

/// Tokenizer builder for creating tokenizers
pub struct TokenizerBuilder;

impl TokenizerBuilder {
    /// Create tokenizer from file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Arc<dyn Tokenizer>> {
        // Placeholder implementation
        tracing::debug!("Loading tokenizer from: {}", path.as_ref().display());
        Ok(Arc::new(BasicTokenizer::new()))
    }

    /// Create tokenizer from pretrained model
    pub fn from_pretrained(name: &str) -> Result<Arc<dyn Tokenizer>> {
        // Placeholder implementation
        tracing::debug!("Loading pretrained tokenizer: {}", name);

        // Return different configurations based on model name for testing
        match name {
            "gpt2" => Ok(Arc::new(BasicTokenizer::with_config(50257, Some(50256), None))),
            "bert" => Ok(Arc::new(BasicTokenizer::with_config(30522, Some(102), Some(0)))),
            "tiny" => Ok(Arc::new(BasicTokenizer::with_config(1000, Some(999), Some(0)))),
            _ => Ok(Arc::new(BasicTokenizer::new())),
        }
    }
}
