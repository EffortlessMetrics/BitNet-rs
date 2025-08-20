//! Tokenization support for BitNet models

pub mod loader;
mod mock;
pub mod universal;

use bitnet_common::Result;
use std::path::Path;
use std::sync::Arc;

pub use loader::load_tokenizer;
pub use mock::MockTokenizer;
pub use universal::UniversalTokenizer;

/// Configuration for tokenizer initialization
#[derive(Debug, Clone, Default)]
pub struct TokenizerConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub pre_tokenizer: Option<String>,
    pub add_bos: bool,
    pub add_eos: bool,
    pub add_space_prefix: bool,
    pub byte_fallback: bool,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub unk_token_id: Option<u32>,
    pub vocabulary: Option<Vec<(String, f32)>>,
    pub bpe_merges: Option<Vec<String>>,
}

impl TokenizerConfig {
    /// Create a default config
    pub fn new() -> Self {
        Self::default()
    }
}

/// Tokenizer trait
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>>;
    fn decode(&self, tokens: &[u32]) -> Result<String>;
    fn vocab_size(&self) -> usize;
    fn token_to_piece(&self, token: u32) -> Option<String>;
    
    // Legacy shims for backward compatibility (default implementations)
    /// Legacy encode method - calls new encode with sensible defaults
    fn encode_legacy(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.encode(text, true, add_special_tokens)
    }
    
    /// Legacy decode method - ignores skip_special_tokens parameter
    fn decode_legacy(&self, tokens: &[u32], _skip_special_tokens: bool) -> Result<String> {
        self.decode(tokens)
    }
    
    /// Legacy EOS token ID getter - returns None by default
    fn eos_token_id(&self) -> Option<u32> {
        None
    }
    
    /// Legacy PAD token ID getter - returns None by default  
    fn pad_token_id(&self) -> Option<u32> {
        None
    }
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
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        // Simple word-based tokenization for testing
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut tokens: Vec<u32> = words.iter().enumerate().map(|(i, _)| i as u32).collect();

        // Add special tokens if requested
        if add_special {
            if let Some(eos_id) = self.eos_token_id {
                tokens.push(eos_id);
            }
        }

        Ok(tokens)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        if tokens.is_empty() {
            return Ok(String::new());
        }

        // Simple placeholder implementation - in real tokenizer this would map back to text
        Ok(format!("Generated text from {} tokens", tokens.len()))
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        // Simple implementation
        Some(format!("<token_{}>", token))
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
