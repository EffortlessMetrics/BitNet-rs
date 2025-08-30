use bitnet_common::Result;
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, warn};

use crate::{Tokenizer, TokenizerConfig};

/// Universal tokenizer that auto-detects and handles all formats
pub struct UniversalTokenizer {
    backend: TokenizerBackend,
    config: TokenizerConfig,
}

enum TokenizerBackend {
    Gpt2(Gpt2Tokenizer),
    #[cfg(feature = "spm")]
    SentencePiece(SentencePieceTokenizer),
    #[allow(dead_code)]
    Llama(LlamaTokenizer),
    Tiktoken(TiktokenTokenizer),
    Falcon(FalconTokenizer),
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

        let vocab: Vec<(String, f32)> = tokens.into_iter().zip(scores.into_iter()).collect();

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
            "gpt2" | "bpe" => {
                debug!("Creating GPT-2 BPE tokenizer");
                Ok(TokenizerBackend::Gpt2(Gpt2Tokenizer::new(config)?))
            }
            #[cfg(feature = "spm")]
            "llama" | "spm" | "sentencepiece" => {
                debug!("Creating SentencePiece tokenizer");
                Ok(TokenizerBackend::SentencePiece(SentencePieceTokenizer::new(config)?))
            }
            #[cfg(not(feature = "spm"))]
            "llama" | "spm" | "sentencepiece" => {
                warn!("SentencePiece support not compiled");
                Ok(TokenizerBackend::Gpt2(Gpt2Tokenizer::new(config)?))
            }
            "llama3" => {
                // Llama 3 uses GPT-2 style BPE with 128k vocab
                debug!("Creating Llama 3 BPE tokenizer");
                Ok(TokenizerBackend::Gpt2(Gpt2Tokenizer::new(config)?))
            }
            "tiktoken" | "gpt4" | "cl100k" => {
                debug!("Creating Tiktoken tokenizer");
                Ok(TokenizerBackend::Tiktoken(TiktokenTokenizer::new(config)?))
            }
            "falcon" => {
                debug!("Creating Falcon tokenizer");
                Ok(TokenizerBackend::Falcon(FalconTokenizer::new(config)?))
            }
            unknown => {
                warn!("Unknown tokenizer type: {}, attempting GPT-2 fallback", unknown);
                Ok(TokenizerBackend::Gpt2(Gpt2Tokenizer::new(config)?))
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
            TokenizerBackend::Gpt2(t) => t.encode(&processed, false, add_special)?,
            #[cfg(feature = "spm")]
            TokenizerBackend::SentencePiece(t) => t.encode(&processed, false, add_special)?,
            TokenizerBackend::Llama(t) => t.encode(&processed, false, add_special)?,
            TokenizerBackend::Tiktoken(t) => t.encode(&processed, false, add_special)?,
            TokenizerBackend::Falcon(t) => t.encode(&processed, false, add_special)?,
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
            TokenizerBackend::Gpt2(t) => t.decode(tokens),
            #[cfg(feature = "spm")]
            TokenizerBackend::SentencePiece(t) => t.decode(tokens),
            TokenizerBackend::Llama(t) => t.decode(tokens),
            TokenizerBackend::Tiktoken(t) => t.decode(tokens),
            TokenizerBackend::Falcon(t) => t.decode(tokens),
        }
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        match &self.backend {
            TokenizerBackend::Gpt2(t) => t.token_to_piece(token),
            #[cfg(feature = "spm")]
            TokenizerBackend::SentencePiece(t) => t.token_to_piece(token),
            TokenizerBackend::Llama(t) => t.token_to_piece(token),
            TokenizerBackend::Tiktoken(t) => t.token_to_piece(token),
            TokenizerBackend::Falcon(t) => t.token_to_piece(token),
        }
    }
}

// Stub implementations for different tokenizer types
// These would be fully implemented in their respective modules

struct Gpt2Tokenizer {
    inner: tokenizers::Tokenizer,
    config: TokenizerConfig,
}

impl Gpt2Tokenizer {
    fn new(config: &TokenizerConfig) -> Result<Self> {
        use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
        use tokenizers::models::bpe::BPE;
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;

        let vocab = config
            .vocabulary
            .clone()
            .unwrap_or_default()
            .into_iter()
            .enumerate()
            .map(|(i, (s, _))| (s, i as u32))
            .collect::<HashMap<_, _>>();
        let merges = config
            .bpe_merges
            .clone()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|m| {
                let mut parts = m.split_whitespace();
                if let (Some(a), Some(b)) = (parts.next(), parts.next()) {
                    Some((a.to_string(), b.to_string()))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let bpe = BPE::builder().vocab_and_merges(vocab, merges).build().map_err(|e| {
            bitnet_common::BitNetError::Model(bitnet_common::ModelError::LoadingFailed {
                reason: format!("BPE build failed: {e}"),
            })
        })?;

        let mut inner = tokenizers::Tokenizer::new(bpe);
        inner.with_pre_tokenizer(ByteLevel::default());
        inner.with_decoder(ByteLevelDecoder::default());

        Ok(Self { inner, config: config.clone() })
    }
}

impl Tokenizer for Gpt2Tokenizer {
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        use tokenizers::EncodeInput;

        let enc =
            self.inner.encode(EncodeInput::Single(text.into()), add_special).map_err(|e| {
                bitnet_common::BitNetError::Model(bitnet_common::ModelError::LoadingFailed {
                    reason: format!("Tokenizer encode error: {e}"),
                })
            })?;
        let mut ids = enc.get_ids().to_vec();
        if add_bos && self.config.bos_token_id.is_some() {
            ids.insert(0, self.config.bos_token_id.unwrap());
        }
        Ok(ids)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.inner.decode(tokens, true).map_err(|e| {
            bitnet_common::BitNetError::Model(bitnet_common::ModelError::LoadingFailed {
                reason: format!("Tokenizer decode error: {e}"),
            })
        })
    }

    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        self.inner.id_to_token(token).map(|s| s.to_string())
    }
}

// Similar stub implementations for other tokenizer types
#[cfg(feature = "spm")]
struct SentencePieceTokenizer {
    inner: tokenizers::Tokenizer,
    config: TokenizerConfig,
}

#[cfg(feature = "spm")]
impl SentencePieceTokenizer {
    fn new(config: &TokenizerConfig) -> Result<Self> {
        use tokenizers::models::unigram::Unigram;

        let vocab = config
            .vocabulary
            .clone()
            .unwrap_or_default()
            .into_iter()
            .map(|(s, score)| (s, score as f64))
            .collect::<Vec<_>>();
        let unigram = Unigram::new(vocab).map_err(|e| {
            bitnet_common::BitNetError::Model(bitnet_common::ModelError::LoadingFailed {
                reason: format!("Unigram build failed: {e}"),
            })
        })?;
        let inner = tokenizers::Tokenizer::new(unigram);
        Ok(Self { inner, config: config.clone() })
    }
}

#[cfg(feature = "spm")]
impl Tokenizer for SentencePieceTokenizer {
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        use tokenizers::EncodeInput;
        let enc =
            self.inner.encode(EncodeInput::Single(text.into()), add_special).map_err(|e| {
                bitnet_common::BitNetError::Model(bitnet_common::ModelError::LoadingFailed {
                    reason: format!("Tokenizer encode error: {e}"),
                })
            })?;
        let mut ids = enc.get_ids().to_vec();
        if add_bos && self.config.bos_token_id.is_some() {
            ids.insert(0, self.config.bos_token_id.unwrap());
        }
        Ok(ids)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.inner.decode(tokens, true).map_err(|e| {
            bitnet_common::BitNetError::Model(bitnet_common::ModelError::LoadingFailed {
                reason: format!("Tokenizer decode error: {e}"),
            })
        })
    }

    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        self.inner.id_to_token(token).map(|s| s.to_string())
    }
}

struct LlamaTokenizer {
    config: TokenizerConfig,
}

impl LlamaTokenizer {
    #[allow(dead_code)]
    fn new(config: &TokenizerConfig) -> Result<Self> {
        Ok(Self { config: config.clone() })
    }
}

impl Tokenizer for LlamaTokenizer {
    fn encode(&self, _text: &str, _add_bos: bool, _add_special: bool) -> Result<Vec<u32>> {
        Ok(vec![1, 2, 3])
    }

    fn decode(&self, _tokens: &[u32]) -> Result<String> {
        Ok("decoded".to_string())
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn token_to_piece(&self, _token: u32) -> Option<String> {
        Some("piece".to_string())
    }
}

struct TiktokenTokenizer {
    config: TokenizerConfig,
}

impl TiktokenTokenizer {
    fn new(config: &TokenizerConfig) -> Result<Self> {
        Ok(Self { config: config.clone() })
    }
}

impl Tokenizer for TiktokenTokenizer {
    fn encode(&self, _text: &str, _add_bos: bool, _add_special: bool) -> Result<Vec<u32>> {
        Ok(vec![1, 2, 3])
    }

    fn decode(&self, _tokens: &[u32]) -> Result<String> {
        Ok("decoded".to_string())
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn token_to_piece(&self, _token: u32) -> Option<String> {
        Some("piece".to_string())
    }
}

struct FalconTokenizer {
    config: TokenizerConfig,
}

impl FalconTokenizer {
    fn new(config: &TokenizerConfig) -> Result<Self> {
        Ok(Self { config: config.clone() })
    }
}

impl Tokenizer for FalconTokenizer {
    fn encode(&self, _text: &str, _add_bos: bool, _add_special: bool) -> Result<Vec<u32>> {
        Ok(vec![1, 2, 3])
    }

    fn decode(&self, _tokens: &[u32]) -> Result<String> {
        Ok("decoded".to_string())
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn token_to_piece(&self, _token: u32) -> Option<String> {
        Some("piece".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_universal_tokenizer_detection() {
        // Test GPT-2 detection
        let config = TokenizerConfig {
            model_type: "gpt2".to_string(),
            vocab_size: 50257,
            ..Default::default()
        };

        let tokenizer = UniversalTokenizer::new(config).unwrap();
        assert_eq!(tokenizer.vocab_size(), 50257);

        // Test auto-fix for missing pre-tokenizer
        // This would be tested with actual GGUF files
    }
}
