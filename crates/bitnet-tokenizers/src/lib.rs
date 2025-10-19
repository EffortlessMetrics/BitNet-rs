//! Tokenization support for BitNet models

// Compile-time policy: forbid FFI tokenizer outside crossval
// This ensures production code uses pure-Rust tokenization for security and determinism
#[cfg(all(not(feature = "crossval"), feature = "ffi_tokenizer"))]
compile_error!(
    "`ffi_tokenizer` is disallowed outside crossval; pure-Rust tokenizer must be used. \
     Build with `--features crossval` if you need FFI tokenizer for cross-validation."
);

pub mod auto;
pub mod gguf_loader;
pub mod gguf_tokenizer;
pub mod hf_tokenizer;
pub mod loader;
mod mock;
pub mod sp_tokenizer;
pub mod spm_tokenizer;
pub mod universal;

// New tokenizer discovery and auto-download modules
pub mod deterministic;
pub mod discovery;
pub mod download;
pub mod error_handling;
pub mod fallback;
pub mod strategy;

use bitnet_common::{BitNetError, ModelError, Result};
use std::path::Path;
use std::sync::Arc;

pub use hf_tokenizer::HfTokenizer;
pub use loader::load_tokenizer;
pub use mock::MockTokenizer;
#[cfg(feature = "spm")]
pub use spm_tokenizer::SpmTokenizer;
pub use universal::UniversalTokenizer;

// Export the new pure-Rust GGUF tokenizer types
pub use gguf_loader::{GgufTokKind, RustTokenizer};

// Export BasicTokenizer for internal and external use
// BasicTokenizer is defined below in this module

// New tokenizer discovery and strategy exports
pub use discovery::{TokenizerDiscovery, TokenizerDownloadInfo, TokenizerStrategy};
pub use download::{DownloadProgress, SmartTokenizerDownload};
pub use error_handling::{CacheManager, ModelTypeDetector, TokenizerErrorHandler};
pub use fallback::TokenizerFallbackChain;
pub use strategy::{
    BitNetTokenizerWrapper, Gpt2TokenizerWrapper, LlamaTokenizerWrapper, TokenizerStrategyResolver,
};

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

    /// Convert a token string to its token ID
    ///
    /// This method is useful for resolving special tokens like `<|eot_id|>` or `<|end_of_text|>`
    /// to their corresponding token IDs for efficient stop sequence matching during generation.
    ///
    /// # Arguments
    /// * `token` - The token string to look up (e.g., "<|eot_id|>")
    ///
    /// # Returns
    /// * `Some(token_id)` if the token exists in the vocabulary
    /// * `None` if the token is not found
    ///
    /// # Example
    /// ```ignore
    /// let eot_id = tokenizer.token_to_id("<|eot_id|>")?;
    /// ```
    fn token_to_id(&self, _token: &str) -> Option<u32> {
        // Default implementation returns None
        // Specific tokenizer implementations should override this
        None
    }

    /// Real vocabulary size from tokenizer model (no padding)
    ///
    /// AC5: This is the actual number of tokens in the vocabulary,
    /// not the padded size used in GGUF for alignment (e.g., 32000 vs 32064).
    /// Use this for cross-validation parity assertions.
    ///
    /// Default implementation returns `vocab_size()` (assumes no padding).
    /// GGUF tokenizers should override this to return the real token count.
    fn real_vocab_size(&self) -> usize {
        self.vocab_size()
    }

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

/// Wrapper for pure-Rust GGUF tokenizer loaded from model metadata
///
/// This wrapper adapts the `gguf_loader::RustTokenizer` to the `Tokenizer` trait,
/// allowing it to be used interchangeably with other tokenizer implementations.
///
/// The wrapper supports both SentencePiece (SPM) and Byte-Pair Encoding (BPE)
/// tokenizers loaded directly from GGUF model files without external tokenizer files.
///
/// # Example
///
/// ```no_run
/// use bitnet_models::{GgufReader, loader::MmapFile};
/// use bitnet_tokenizers::RustGgufTokenizer;
///
/// # fn example(path: &std::path::Path) -> anyhow::Result<()> {
/// let mmap = MmapFile::open(path)
///     .map_err(|e| anyhow::anyhow!("Failed to open file: {}", e))?;
/// let reader = GgufReader::new(mmap.as_slice())
///     .map_err(|e| anyhow::anyhow!("Failed to parse GGUF: {}", e))?;
/// let tokenizer = RustGgufTokenizer::from_gguf(&reader)?;
/// # Ok(())
/// # }
/// ```
pub struct RustGgufTokenizer {
    inner: crate::gguf_loader::RustTokenizer,
}

impl RustGgufTokenizer {
    /// Create tokenizer from GGUF metadata
    ///
    /// This method loads the tokenizer directly from GGUF model metadata,
    /// detecting the tokenizer kind (SPM or BPE) and extracting special token IDs.
    ///
    /// # Arguments
    ///
    /// * `reader` - GGUF file reader with metadata and tensors
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Tokenizer kind cannot be detected from metadata
    /// - Required tokenizer data is missing (SPM protobuf or BPE vocab/merges)
    /// - Tokenizer construction fails (invalid data)
    /// - SPM feature is not enabled when loading SPM tokenizer
    pub fn from_gguf(reader: &bitnet_models::GgufReader) -> anyhow::Result<Self> {
        let inner = crate::gguf_loader::RustTokenizer::from_gguf(reader)?;
        Ok(Self { inner })
    }

    /// Get BOS, EOS, and EOT token IDs
    ///
    /// This is useful for prompt formatting and template detection.
    ///
    /// # Returns
    ///
    /// Tuple of (bos_id, eos_id, eot_id) where each is `Option<u32>`
    pub fn bos_eos_eot(&self) -> (Option<u32>, Option<u32>, Option<u32>) {
        (self.inner.bos_id(), self.inner.eos_id(), self.inner.eot_id())
    }

    /// Get hint for whether to add BOS by default
    ///
    /// This hint is extracted from GGUF metadata (`tokenizer.ggml.add_bos_token`)
    /// and can be used to determine default encoding behavior.
    pub fn add_bos_hint(&self) -> Option<bool> {
        self.inner.add_bos_hint()
    }

    /// Get tokenizer kind (SPM or BPE)
    pub fn kind(&self) -> crate::gguf_loader::GgufTokKind {
        self.inner.kind()
    }
}

impl Tokenizer for RustGgufTokenizer {
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        // Map the trait's add_special parameter to RustTokenizer's parse_special
        self.inner
            .encode(text, add_bos, add_special)
            .map_err(|e| BitNetError::Model(ModelError::LoadingFailed { reason: e.to_string() }))
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        // RustTokenizer doesn't implement decode yet - return placeholder
        // TODO: Implement decode in RustTokenizer when needed
        Ok(format!("Generated text from {} tokens", tokens.len()))
    }

    fn vocab_size(&self) -> usize {
        // Return a reasonable default - actual vocab size varies by model
        // TODO: Expose vocab_size from RustTokenizer when BPE/SPM provide it
        match self.inner.kind() {
            crate::gguf_loader::GgufTokKind::Spm => 32000, // Typical LLaMA vocab size
            crate::gguf_loader::GgufTokKind::Bpe => 50257, // GPT-2 vocab size
        }
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        // Return placeholder - actual piece conversion requires vocab lookup
        // TODO: Implement token_to_piece in RustTokenizer when needed
        Some(format!("<token_{}>", token))
    }

    fn bos_token_id(&self) -> Option<u32> {
        self.inner.bos_id()
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.inner.eos_id()
    }
}

/// Tokenizer file kind
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerFileKind {
    HfJson,
    #[cfg(feature = "spm")]
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
        let (tokenizer, kind) = from_path(path.as_ref())?;
        #[cfg(feature = "spm")]
        {
            if let TokenizerFileKind::Spm = kind {}
        }
        #[cfg(not(feature = "spm"))]
        {
            let _ = kind;
        }
        Ok(tokenizer)
    }

    /// Create tokenizer from GGUF model metadata
    ///
    /// This method loads a pure-Rust tokenizer directly from GGUF model metadata,
    /// supporting both SentencePiece (SPM) and Byte-Pair Encoding (BPE) tokenizers
    /// without requiring external tokenizer files.
    ///
    /// # Arguments
    ///
    /// * `reader` - GGUF file reader with metadata and tensors
    ///
    /// # Returns
    ///
    /// Arc-wrapped tokenizer that implements the Tokenizer trait
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Tokenizer kind cannot be detected from metadata
    /// - Required tokenizer data is missing (SPM protobuf or BPE vocab/merges)
    /// - Tokenizer construction fails (invalid data)
    /// - SPM feature is not enabled when loading SPM tokenizer
    ///
    /// # Example
    ///
    /// ```no_run
    /// use bitnet_models::{GgufReader, loader::MmapFile};
    /// use bitnet_tokenizers::TokenizerBuilder;
    /// # use bitnet_common::Result;
    ///
    /// # fn example(path: &std::path::Path) -> Result<()> {
    /// let mmap = MmapFile::open(path)?;
    /// let reader = GgufReader::new(mmap.as_slice())?;
    /// let tokenizer = TokenizerBuilder::from_gguf_reader(&reader)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_gguf_reader(reader: &bitnet_models::GgufReader) -> Result<Arc<dyn Tokenizer>> {
        let tokenizer = RustGgufTokenizer::from_gguf(reader)
            .map_err(|e| BitNetError::Model(ModelError::LoadingFailed { reason: e.to_string() }))?;
        Ok(Arc::new(tokenizer))
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
