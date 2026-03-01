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
pub mod utils;
pub mod vocabulary;

use bitnet_common::{BitNetError, ModelError, Result};
use std::path::Path;
use std::sync::Arc;

pub use hf_tokenizer::HfTokenizer;
pub use loader::load_tokenizer;
pub use mock::MockTokenizer;
#[cfg(feature = "spm")]
pub use spm_tokenizer::SpmTokenizer;
pub use universal::{TokenizerBackend, UniversalTokenizer};

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

pub use bitnet_tokenizers_core::{BasicTokenizer, Tokenizer, TokenizerConfig};

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
        self.inner.decode(tokens)
    }

    fn vocab_size(&self) -> usize {
        // Delegate to inner RustTokenizer
        self.inner.vocab_size()
    }

    fn real_vocab_size(&self) -> usize {
        // Delegate to inner RustTokenizer
        self.inner.real_vocab_size()
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        self.inner.token_to_piece(token)
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

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    // BasicTokenizer byte-level encode is invertible (round-trips through decode).
    // Only ASCII text fits in byte-level vocab (IDs 0–255) with default vocab_size ≥ 256.
    proptest! {
        #[test]
        fn basic_tokenizer_ascii_roundtrip(text in "[a-zA-Z0-9 .,!?:;-]{1,80}") {
            let tok = BasicTokenizer::new();
            let tokens = tok.encode(&text, false, false).unwrap();
            let decoded = tok.decode(&tokens).unwrap();
            prop_assert_eq!(decoded, text.clone(), "round-trip failed for {:?}", text);
        }
    }

    // encode without add_bos/add_special: length equals UTF-8 byte count.
    proptest! {
        #[test]
        fn basic_tokenizer_length_matches_bytes(text in "[a-z]{1,64}") {
            let tok = BasicTokenizer::new();
            let tokens = tok.encode(&text, false, false).unwrap();
            prop_assert_eq!(tokens.len(), text.len(), "token count != byte count for {:?}", text);
        }
    }

    // encode of empty string always returns empty vec regardless of flags.
    proptest! {
        #[test]
        fn basic_tokenizer_empty_is_always_empty(add_bos in any::<bool>(), add_special in any::<bool>()) {
            // BasicTokenizer::encode returns early for empty text, producing no tokens
            // even when add_bos or add_special are true (bos_token_id is None by default).
            let tok = BasicTokenizer::new();
            let tokens = tok.encode("", add_bos, add_special).unwrap();
            prop_assert_eq!(tokens.len(), 0);
        }
    }

    // decode of empty slice always returns empty string.
    proptest! {
        #[test]
        fn basic_tokenizer_decode_empty_slice_is_empty(_dummy in any::<bool>()) {
            let tok = BasicTokenizer::new();
            let result = tok.decode(&[]).unwrap();
            prop_assert_eq!(result, "");
        }
    }
}

#[cfg(test)]
mod edge_case_tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_encode_empty_string() {
        let tok = BasicTokenizer::new();
        let result = tok.encode("", false, false).unwrap();
        assert!(result.is_empty(), "encoding empty string should produce empty vec");

        let result_with_bos = tok.encode("", true, true).unwrap();
        assert!(result_with_bos.is_empty(), "empty string returns early before BOS/EOS");
    }

    #[test]
    fn test_encode_empty_string_with_bos_configured() {
        let tok = BasicTokenizer::with_config(50257, Some(1), Some(2), None);
        // BasicTokenizer returns early on empty text, so even with BOS configured
        // no tokens are produced.
        let result = tok.encode("", true, true).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_decode_empty_tokens() {
        let tok = BasicTokenizer::new();
        let result = tok.decode(&[]).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_decode_invalid_utf8_bytes() {
        let tok = BasicTokenizer::new();
        // Feed byte-level token IDs that form invalid UTF-8 (0xFF 0xFE)
        let result = tok.decode(&[0xFF, 0xFE]).unwrap();
        // from_utf8_lossy replaces invalid bytes with U+FFFD
        assert!(result.contains('\u{FFFD}'), "invalid UTF-8 should produce replacement chars");
    }

    #[test]
    fn test_decode_skips_special_tokens() {
        let tok = BasicTokenizer::with_config(50257, Some(1), Some(2), Some(3));
        // Mix of special token IDs and real byte-level IDs (ASCII 'A' = 65)
        let result = tok.decode(&[1, 65, 2, 3]).unwrap();
        assert_eq!(result, "A", "special tokens (BOS=1, EOS=2, PAD=3) should be skipped");
    }

    #[test]
    fn test_is_special_token() {
        let tok = BasicTokenizer::with_config(50257, Some(1), Some(2), Some(3));
        assert!(tok.is_special_token(1), "BOS should be special");
        assert!(tok.is_special_token(2), "EOS should be special");
        assert!(tok.is_special_token(3), "PAD should be special");
        assert!(!tok.is_special_token(65), "regular token should not be special");
    }

    #[test]
    fn test_is_special_token_none_configured() {
        let tok = BasicTokenizer::new();
        // Default BasicTokenizer has bos=None, eos=Some(50256), pad=None
        assert!(!tok.is_special_token(0), "no BOS configured");
        assert!(tok.is_special_token(50256), "EOS should be special");
        assert!(!tok.is_special_token(42), "arbitrary ID not special");
    }

    #[test]
    fn test_get_family_name_default() {
        let tok = BasicTokenizer::new();
        assert_eq!(tok.get_family_name(), "unknown");
    }

    #[test]
    fn test_mock_special_token_family_detection() {
        let tok = MockTokenizer::with_special_tokens(&[
            ("<|eot_id|>", 128009),
            ("<|start_header_id|>", 128006),
        ]);
        assert_eq!(tok.get_family_name(), "llama3");
    }

    #[test]
    fn test_mock_mistral_family_detection() {
        let tok = MockTokenizer::with_special_tokens(&[("[INST]", 3)]);
        assert_eq!(tok.get_family_name(), "mistral-instruct");
    }

    #[test]
    fn test_from_path_unsupported_extension() {
        let path = PathBuf::from("/tmp/model.bin");
        let result = from_path(&path);
        assert!(result.is_err());
        let err_msg = result.err().map(|e| format!("{}", e)).unwrap_or_default();
        assert!(
            err_msg.contains("model.bin"),
            "error should mention the file path, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_from_path_no_extension() {
        let path = PathBuf::from("/tmp/tokenizer");
        let result = from_path(&path);
        assert!(result.is_err());
        let err_msg = result.err().map(|e| format!("{}", e)).unwrap_or_default();
        assert!(
            err_msg.contains("tokenizer"),
            "error should mention the file path, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_token_to_piece_byte_range() {
        let tok = BasicTokenizer::new();
        // ASCII 'A' = 65
        assert_eq!(tok.token_to_piece(65), Some("A".to_string()));
        // High byte-level token
        assert_eq!(tok.token_to_piece(0), Some("\0".to_string()));
        // Beyond byte range gives formatted placeholder
        assert_eq!(tok.token_to_piece(1000), Some("<token_1000>".to_string()));
    }

    #[test]
    fn test_encode_single_byte() {
        let tok = BasicTokenizer::new();
        let tokens = tok.encode("A", false, false).unwrap();
        assert_eq!(tokens, vec![65]);
    }

    #[test]
    fn test_encode_multibyte_utf8() {
        let tok = BasicTokenizer::new();
        // '€' is 3 bytes in UTF-8: 0xE2, 0x82, 0xAC
        let tokens = tok.encode("€", false, false).unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens, vec![0xE2, 0x82, 0xAC]);
    }

    #[test]
    fn test_mock_tokenizer_encode_empty() {
        let tok = MockTokenizer::new();
        let tokens = tok.encode("", false, false).unwrap();
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_mock_tokenizer_decode_high_ids_skipped() {
        let tok = MockTokenizer::new();
        // IDs >= 256 are skipped (treated as special/OOV tokens)
        let result = tok.decode(&[65, 500, 66]).unwrap();
        assert_eq!(result, "AB", "high IDs should be skipped in mock decode");
    }
}
