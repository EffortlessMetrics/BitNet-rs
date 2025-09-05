//! Tokenization support for BitNet models

pub mod gguf_tokenizer;
pub mod hf_tokenizer;
pub mod loader;
mod mock;
pub mod sp_tokenizer;
pub mod spm_tokenizer;
pub mod universal;

use bitnet_common::{BitNetError, ModelError, Result};
use std::path::Path;
use std::sync::Arc;

pub use hf_tokenizer::HfTokenizer;
pub use loader::load_tokenizer;
pub use mock::MockTokenizer;
#[cfg(feature = "spm")]
pub use spm_tokenizer::SpmTokenizer;
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

    /// BOS token ID getter - returns None by default
    fn bos_token_id(&self) -> Option<u32> {
        None
    }

    /// EOS token ID getter - returns None by default
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
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    pad_token_id: Option<u32>,
}

impl BasicTokenizer {
    pub fn new() -> Self {
        Self {
            vocab_size: 50257, // GPT-2 vocab size
            bos_token_id: None,
            eos_token_id: Some(50256),
            pad_token_id: None,
        }
    }

    pub fn with_config(
        vocab_size: usize,
        bos_token_id: Option<u32>,
        eos_token_id: Option<u32>,
        pad_token_id: Option<u32>,
    ) -> Self {
        Self { vocab_size, bos_token_id, eos_token_id, pad_token_id }
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

        let words: Vec<&str> = text.split_whitespace().collect();
        let mut tokens: Vec<u32> = Vec::new();

        if add_bos && let Some(bos) = self.bos_token_id {
            tokens.push(bos);
        }

        for (i, _) in words.iter().enumerate() {
            let id = i as u32;
            if id >= self.vocab_size as u32 {
                return Err(BitNetError::Model(ModelError::LoadingFailed {
                    reason: "token id exceeds vocab size".to_string(),
                }));
            }
            tokens.push(id);
        }

        if add_special {
            if let Some(eos_id) = self.eos_token_id {
                tokens.push(eos_id);
            }
            if let Some(pad_id) = self.pad_token_id {
                tokens.push(pad_id);
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
        Some(format!("<token_{}>", token))
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    fn pad_token_id(&self) -> Option<u32> {
        self.pad_token_id
    }
}

/// Tokenizer file kind
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerFileKind {
    HfJson,
    #[cfg(feature = "spm")]
    #[allow(dead_code)]
    Spm,
}

/// Load tokenizer from path based on file extension
pub fn from_path(path: &Path) -> Result<(Arc<dyn Tokenizer>, TokenizerFileKind)> {
    use bitnet_common::{BitNetError, ModelError};

    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();

    match ext.as_str() {
        "json" => {
            let t = HfTokenizer::from_file(path).map_err(|e| {
                BitNetError::Model(ModelError::LoadingFailed {
                    reason: format!("Failed to load HF tokenizer: {}", e),
                })
            })?;
            Ok((Arc::new(t), TokenizerFileKind::HfJson))
        }
        "model" => {
            #[cfg(feature = "spm")]
            {
                let t = SpmTokenizer::from_file(path).map_err(|e| {
                    BitNetError::Model(ModelError::LoadingFailed {
                        reason: format!("Failed to load SPM tokenizer: {}", e),
                    })
                })?;
                Ok((Arc::new(t), TokenizerFileKind::Spm))
            }
            #[cfg(not(feature = "spm"))]
            {
                Err(BitNetError::Model(ModelError::LoadingFailed {
                    reason: "Build with `--features spm` to load SentencePiece .model files"
                        .to_string(),
                }))
            }
        }
        _ => Err(BitNetError::Model(ModelError::LoadingFailed {
            reason: format!(
                "Unsupported tokenizer file (expected *.json or *.model): {}",
                path.display()
            ),
        })),
    }
}

/// Try to construct tokenizer from GGUF metadata (placeholder)
pub fn try_from_gguf_metadata<F>(_build_from_arrays: F) -> Option<Arc<dyn Tokenizer>>
where
    F: FnOnce() -> Result<Arc<dyn Tokenizer>>,
{
    // Hook for future GGUF-embedded tokenizer support
    None
}

/// Tokenizer builder for creating tokenizers
pub struct TokenizerBuilder;

impl TokenizerBuilder {
    /// Create tokenizer from file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Arc<dyn Tokenizer>> {
        let (tokenizer, _kind) = from_path(path.as_ref())?;
        Ok(tokenizer)
    }

    /// Create tokenizer from pretrained model
    pub fn from_pretrained(name: &str) -> Result<Arc<dyn Tokenizer>> {
        // Placeholder implementation
        tracing::debug!("Loading pretrained tokenizer: {}", name);

        // Return different configurations based on model name for testing
        match name {
            "gpt2" => Ok(Arc::new(BasicTokenizer::with_config(50257, None, Some(50256), None))),
            "bert" => {
                Ok(Arc::new(BasicTokenizer::with_config(30522, Some(101), Some(102), Some(0))))
            }
            "tiny" => Ok(Arc::new(BasicTokenizer::with_config(1000, None, Some(999), Some(0)))),
            _ => Ok(Arc::new(BasicTokenizer::new())),
        }
    }
}
