use bitnet_common::Result;
use std::path::Path;
use tracing::{debug, warn};

use crate::{HfTokenizer, MockTokenizer, Tokenizer, TokenizerConfig};

/// Universal tokenizer that auto-detects and handles all formats
pub struct UniversalTokenizer {
    backend: TokenizerBackend,
    config: TokenizerConfig,
}

#[allow(clippy::large_enum_variant)]
enum TokenizerBackend {
    #[cfg(feature = "spm")]
    SentencePiece(crate::SpmTokenizer),
    Hf(HfTokenizer),
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

    fn detect_and_create_backend(config: &TokenizerConfig) -> Result<TokenizerBackend> {
        match config.model_type.as_str() {
            "gpt2" | "bpe" | "llama" | "llama3" | "tiktoken" | "gpt4" | "cl100k" | "falcon" => {
                if let (Some(vocab), Some(merges)) =
                    (config.vocabulary.as_ref(), config.bpe_merges.as_ref())
                {
                    debug!("Creating HF tokenizer for {}", config.model_type);
                    let tok = HfTokenizer::from_vocab_and_merges(vocab, merges).map_err(|e| {
                        bitnet_common::BitNetError::Model(
                            bitnet_common::ModelError::LoadingFailed {
                                reason: format!("Tokenizer construction error: {}", e),
                            },
                        )
                    })?;
                    Ok(TokenizerBackend::Hf(tok))
                } else {
                    debug!(
                        "Missing vocab or merges for {}, using mock tokenizer",
                        config.model_type
                    );
                    Ok(TokenizerBackend::Mock(MockTokenizer::new()))
                }
            }
            #[cfg(feature = "spm")]
            "smp" | "sentencepiece" => {
                warn!("SentencePiece tokenizer requires file path, using mock fallback");
                Ok(TokenizerBackend::Mock(MockTokenizer::new()))
            }
            #[cfg(not(feature = "spm"))]
            "smp" | "sentencepiece" => {
                warn!("SentencePiece support not compiled, using mock");
                Ok(TokenizerBackend::Mock(MockTokenizer::new()))
            }
            unknown => {
                warn!("Unknown tokenizer type: {}, using mock fallback", unknown);
                Ok(TokenizerBackend::Mock(MockTokenizer::new()))
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
            TokenizerBackend::SentencePiece(t) => t.encode(&processed, false, add_special)?,
            TokenizerBackend::Hf(t) => t.encode(&processed, false, add_special)?,
            TokenizerBackend::Mock(t) => t.encode(&processed, false, add_special)?,
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
            TokenizerBackend::SentencePiece(t) => t.decode(tokens),
            TokenizerBackend::Hf(t) => t.decode(tokens),
            TokenizerBackend::Mock(t) => t.decode(tokens),
        }
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        match &self.backend {
            #[cfg(feature = "spm")]
            TokenizerBackend::SentencePiece(t) => t.token_to_piece(token),
            TokenizerBackend::Hf(t) => t.token_to_piece(token),
            TokenizerBackend::Mock(t) => t.token_to_piece(token),
        }
    }
}
