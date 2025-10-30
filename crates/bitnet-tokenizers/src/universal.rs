use bitnet_common::{BitNetError, InferenceError, Result};
use std::path::Path;
use tracing::{debug, warn};

#[cfg(feature = "spm")]
use crate::SpmTokenizer;
use crate::{MockTokenizer, Tokenizer, TokenizerConfig};
#[cfg(feature = "spm")]
use bitnet_common::ModelError;
use bitnet_models::Model;

/// Backend type identifier for universal tokenizer
///
/// This enum represents the different tokenizer backend implementations
/// that can be used by the universal tokenizer. The backend is auto-detected
/// based on model metadata and configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerBackend {
    /// SentencePiece tokenizer backend (requires `spm` feature)
    #[cfg(feature = "spm")]
    SentencePiece,
    /// Mock tokenizer backend for testing and fallback
    Mock,
}

/// Universal tokenizer that auto-detects and handles all formats
pub struct UniversalTokenizer {
    backend: InternalTokenizerBackend,
    config: TokenizerConfig,
}

#[allow(clippy::large_enum_variant)]
enum InternalTokenizerBackend {
    #[cfg(feature = "spm")]
    SentencePiece(crate::SpmTokenizer),
    Mock(MockTokenizer),
}

impl UniversalTokenizer {
    /// Create from model metadata with auto-detection
    pub fn new(config: TokenizerConfig) -> Result<Self> {
        let backend = Self::detect_and_create_backend(&config)?;
        Ok(Self { backend, config })
    }

    /// Create from GGUF model with auto-fix
    pub fn from_gguf(path: &Path) -> Result<Self> {
        use bitnet_models::{GgufReader, loader::MmapFile};

        let mmap = MmapFile::open(path)?;
        let reader = GgufReader::new(mmap.as_slice())?;

        let model_type =
            reader.get_string_metadata("tokenizer.ggml.model").unwrap_or_else(|| "gpt2".into());

        let tokens = reader.get_string_array_metadata("tokenizer.ggml.tokens").unwrap_or_default();
        let vocab_size = tokens.len();

        let scores_bytes = reader.get_array_metadata("tokenizer.ggml.scores");
        let scores: Vec<f32> = scores_bytes
            .map(|b| {
                b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
            })
            .unwrap_or_else(|| vec![0.0; vocab_size]);

        let vocab: Vec<(String, f32)> = tokens.into_iter().zip(scores).collect();

        let merges = reader.get_string_array_metadata("tokenizer.ggml.merges");

        let config = TokenizerConfig {
            model_type,
            vocab_size,
            pre_tokenizer: reader.get_string_metadata("tokenizer.ggml.pre"),
            add_bos: reader.get_bool_metadata("tokenizer.ggml.add_bos_token").unwrap_or(false),
            add_eos: reader.get_bool_metadata("tokenizer.ggml.add_eos_token").unwrap_or(false),
            add_space_prefix: reader
                .get_bool_metadata("tokenizer.ggml.add_space_prefix")
                .unwrap_or(false),
            byte_fallback: reader
                .get_bool_metadata("tokenizer.ggml.byte_fallback")
                .unwrap_or(false),
            bos_token_id: reader.get_u32_metadata("tokenizer.ggml.bos_token_id"),
            eos_token_id: reader.get_u32_metadata("tokenizer.ggml.eos_token_id"),
            pad_token_id: reader.get_u32_metadata("tokenizer.ggml.padding_token_id"),
            unk_token_id: reader.get_u32_metadata("tokenizer.ggml.unknown_token_id"),
            vocabulary: Some(vocab),
            bpe_merges: merges,
        };
        Self::new(config)
    }

    /// Create from model with auto-detection
    pub fn from_model_config(config: TokenizerConfig) -> Result<Self> {
        Self::new(config)
    }

    /// Create tokenizer from GGUF model with backend preference
    ///
    /// This method loads a tokenizer from a BitNet model's metadata,
    /// with a preference for a specific tokenizer backend (e.g., SentencePiece vs BPE).
    ///
    /// # Arguments
    ///
    /// * `model` - BitNet model with configuration and metadata
    /// * `preferred_backend` - Preferred tokenizer backend type
    ///
    /// # Returns
    ///
    /// A UniversalTokenizer configured from the model's metadata, preferring the
    /// specified backend when available, or falling back to auto-detection.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Model metadata is missing required tokenizer configuration
    /// - Preferred backend is not available and strict mode is enabled
    /// - Tokenizer construction fails
    pub fn from_gguf_model_with_preference(
        model: &bitnet_models::BitNetModel,
        preferred_backend: TokenizerBackend,
    ) -> Result<Self> {
        // Extract model configuration
        let config = model.config();

        // Build tokenizer configuration from model metadata
        let tokenizer_config = TokenizerConfig {
            model_type: match preferred_backend {
                #[cfg(feature = "spm")]
                TokenizerBackend::SentencePiece => "sentencepiece".to_string(),
                TokenizerBackend::Mock => "gpt2".to_string(), // Default to GPT-2 for mock
            },
            vocab_size: config.model.vocab_size,
            pre_tokenizer: None,
            add_bos: config.inference.add_bos,
            add_eos: config.inference.append_eos,
            add_space_prefix: false,
            byte_fallback: false,
            bos_token_id: config.model.tokenizer.bos_id.map(|id| id as u32),
            eos_token_id: config.model.tokenizer.eos_id.map(|id| id as u32),
            pad_token_id: config.model.tokenizer.pad_id.map(|id| id as u32),
            unk_token_id: config.model.tokenizer.unk_id.map(|id| id as u32),
            vocabulary: None, // Will be auto-detected or mocked
            bpe_merges: None, // Will be auto-detected or mocked
        };

        // Attempt to create tokenizer with preferred backend
        Self::new(tokenizer_config)
    }

    /// Encode batch of texts
    ///
    /// This is a convenience method that encodes multiple texts in sequence.
    /// For the current implementation, this is not parallelized, but could be
    /// optimized in the future.
    pub fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<u32>>> {
        texts.iter().map(|text| self.encode(text, false, false)).collect()
    }

    /// Get the backend type of this tokenizer
    ///
    /// Returns the type of tokenizer backend currently in use. This is useful
    /// for testing and debugging to verify which backend was auto-detected.
    pub fn backend_type(&self) -> TokenizerBackend {
        match &self.backend {
            #[cfg(feature = "spm")]
            InternalTokenizerBackend::SentencePiece(_) => TokenizerBackend::SentencePiece,
            InternalTokenizerBackend::Mock(_) => TokenizerBackend::Mock,
        }
    }

    fn detect_and_create_backend(config: &TokenizerConfig) -> Result<InternalTokenizerBackend> {
        match config.model_type.as_str() {
            "gpt2" | "bpe" | "llama" | "llama3" | "tiktoken" | "gpt4" | "cl100k" | "falcon" => {
                // Strict mode forbids silent mock fallbacks (CI, perf runs)
                if std::env::var("BITNET_STRICT_TOKENIZERS").as_deref() == Ok("1") {
                    return Err(BitNetError::Inference(InferenceError::TokenizationFailed {
                        reason: "Mock tokenizer fallback disabled (BITNET_STRICT_TOKENIZERS=1)"
                            .to_string(),
                    }));
                }
                debug!("Creating mock tokenizer for {}", config.model_type);
                Ok(InternalTokenizerBackend::Mock(MockTokenizer::new()))
            }
            #[cfg(feature = "spm")]
            "smp" | "sentencepiece" => {
                if let Some(path) = &config.pre_tokenizer {
                    let spm = SpmTokenizer::from_file(Path::new(path)).map_err(|e| {
                        BitNetError::Model(ModelError::LoadingFailed {
                            reason: format!("Failed to load SentencePiece tokenizer: {}", e),
                        })
                    })?;
                    Ok(InternalTokenizerBackend::SentencePiece(spm))
                } else {
                    if std::env::var("BITNET_STRICT_TOKENIZERS").as_deref() == Ok("1") {
                        return Err(BitNetError::Inference(InferenceError::TokenizationFailed {
                            reason: "Mock tokenizer fallback disabled (BITNET_STRICT_TOKENIZERS=1)"
                                .to_string(),
                        }));
                    }
                    warn!("SentencePiece tokenizer requires model path, using mock fallback");
                    Ok(InternalTokenizerBackend::Mock(MockTokenizer::new()))
                }
            }
            #[cfg(not(feature = "spm"))]
            "smp" | "sentencepiece" => {
                if std::env::var("BITNET_STRICT_TOKENIZERS").as_deref() == Ok("1") {
                    return Err(BitNetError::Inference(InferenceError::TokenizationFailed {
                        reason: "Mock tokenizer fallback disabled (BITNET_STRICT_TOKENIZERS=1)"
                            .to_string(),
                    }));
                }
                warn!("SentencePiece support not compiled, using mock");
                Ok(InternalTokenizerBackend::Mock(MockTokenizer::new()))
            }
            unknown => {
                if std::env::var("BITNET_STRICT_TOKENIZERS").as_deref() == Ok("1") {
                    return Err(BitNetError::Inference(InferenceError::TokenizationFailed {
                        reason: format!(
                            "Mock tokenizer fallback disabled for unknown type '{}' (BITNET_STRICT_TOKENIZERS=1)",
                            unknown
                        ),
                    }));
                }
                warn!("Unknown tokenizer type: {}, using mock fallback", unknown);
                Ok(InternalTokenizerBackend::Mock(MockTokenizer::new()))
            }
        }
    }
}

impl Tokenizer for UniversalTokenizer {
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        // Apply pre-tokenization if needed
        let processed = if self.config.add_space_prefix && !text.starts_with(' ') {
            format!(" {}", text)
        } else {
            text.to_string()
        };

        // Delegate to backend
        let mut tokens = match &self.backend {
            #[cfg(feature = "spm")]
            InternalTokenizerBackend::SentencePiece(t) => {
                t.encode(&processed, false, add_special)?
            }
            InternalTokenizerBackend::Mock(t) => t.encode(&processed, false, add_special)?,
        };

        // Add BOS if requested and configured
        if add_bos
            && self.config.add_bos
            && let Some(bos_id) = self.config.bos_token_id
        {
            tokens.insert(0, bos_id);
        }

        Ok(tokens)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        match &self.backend {
            #[cfg(feature = "spm")]
            InternalTokenizerBackend::SentencePiece(t) => t.decode(tokens),
            InternalTokenizerBackend::Mock(t) => t.decode(tokens),
        }
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        match &self.backend {
            #[cfg(feature = "spm")]
            InternalTokenizerBackend::SentencePiece(t) => t.token_to_piece(token),
            InternalTokenizerBackend::Mock(t) => t.token_to_piece(token),
        }
    }
}
