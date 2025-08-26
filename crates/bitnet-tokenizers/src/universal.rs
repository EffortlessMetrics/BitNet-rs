use bitnet_common::{BitNetError, ModelError, Result};
use std::path::Path;
use tracing::{debug, warn};

use tokenizers::{models::unigram::Unigram, EncodeInput, Tokenizer as HfTokenizer};
use tiktoken_rs::{cl100k_base, p50k_base, r50k_base, CoreBPE};
use bitnet_models::GgufReader;

use crate::{Tokenizer, TokenizerConfig};

/// Universal tokenizer that auto-detects and handles all formats
pub struct UniversalTokenizer {
    backend: TokenizerBackend,
    config: TokenizerConfig,
}

enum TokenizerBackend {
    Gpt2(Gpt2Tokenizer),
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

    /// Create from GGUF model by reading tokenizer metadata
    pub fn from_gguf(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)?;
        let reader = GgufReader::new(&data)?;

        // Detect tokenizer model type
        let model_type = reader
            .get_string_metadata("tokenizer.ggml.model")
            .unwrap_or_else(|| "gpt2".to_string());

        let vocab_size = reader
            .get_u32_metadata("tokenizer.ggml.vocab_size")
            .or_else(|| reader.get_u32_metadata("llama.vocab_size"))
            .unwrap_or(0) as usize;

        let mut config = TokenizerConfig::default();
        config.model_type = model_type;
        config.vocab_size = vocab_size;
        config.bos_token_id = reader.get_u32_metadata("tokenizer.ggml.bos_token_id");
        config.eos_token_id = reader.get_u32_metadata("tokenizer.ggml.eos_token_id");
        config.pad_token_id = reader.get_u32_metadata("tokenizer.ggml.padding_token_id");
        config.unk_token_id = reader.get_u32_metadata("tokenizer.ggml.unknown_token_id");
        config.add_bos = reader.get_bool_metadata("tokenizer.ggml.add_bos_token").unwrap_or(false);
        config.add_eos = reader.get_bool_metadata("tokenizer.ggml.add_eos_token").unwrap_or(false);
        config.byte_fallback = reader.get_bool_metadata("tokenizer.ggml.byte_fallback").unwrap_or(false);

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
            "llama" | "spm" | "sentencepiece" => {
                debug!("Creating SentencePiece tokenizer");
                Ok(TokenizerBackend::SentencePiece(SentencePieceTokenizer::new(config)?))
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
            TokenizerBackend::SentencePiece(t) => t.token_to_piece(token),
            TokenizerBackend::Llama(t) => t.token_to_piece(token),
            TokenizerBackend::Tiktoken(t) => t.token_to_piece(token),
            TokenizerBackend::Falcon(t) => t.token_to_piece(token),
        }
    }
}

// Real implementations for different tokenizer backends

struct Gpt2Tokenizer {
    bpe: CoreBPE,
    config: TokenizerConfig,
}

impl Gpt2Tokenizer {
    fn new(config: &TokenizerConfig) -> Result<Self> {
        let bpe = r50k_base().map_err(|e| {
            BitNetError::Model(ModelError::LoadingFailed {
                reason: format!("tiktoken init error: {e}"),
            })
        })?;
        Ok(Self { bpe, config: config.clone() })
    }
}

impl Tokenizer for Gpt2Tokenizer {
    fn encode(&self, text: &str, _add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        let tokens = if add_special {
            self.bpe.encode_with_special_tokens(text)
        } else {
            self.bpe.encode_ordinary(text)
        };
        Ok(tokens)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.bpe.decode(tokens.to_vec()).map_err(|e| {
            BitNetError::Model(ModelError::LoadingFailed {
                reason: format!("decode error: {e}"),
            })
        })
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        self.bpe.decode(vec![token]).ok()
    }
}

struct SentencePieceTokenizer {
    inner: HfTokenizer,
    config: TokenizerConfig,
}

impl SentencePieceTokenizer {
    fn new(config: &TokenizerConfig) -> Result<Self> {
        let vocab = config
            .vocabulary
            .clone()
            .unwrap_or_default()
            .into_iter()
            .map(|(t, s)| (t, s as f64))
            .collect();
        let model = Unigram::from(vocab, config.unk_token_id.map(|i| i as usize), config.byte_fallback)
            .map_err(|e| BitNetError::Model(ModelError::LoadingFailed {
                reason: format!("Unigram build failed: {e}"),
            }))?;
        let tokenizer = HfTokenizer::new(model);
        Ok(Self { inner: tokenizer, config: config.clone() })
    }
}

impl Tokenizer for SentencePieceTokenizer {
    fn encode(&self, text: &str, _add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(EncodeInput::Single(text.into()), add_special)
            .map_err(|e| BitNetError::Model(ModelError::LoadingFailed {
                reason: format!("Tokenizer encode error: {e}"),
            }))?;
        Ok(enc.get_ids().to_vec())
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.inner.decode(tokens, true).map_err(|e| {
            BitNetError::Model(ModelError::LoadingFailed {
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
    inner: SentencePieceTokenizer,
}

impl LlamaTokenizer {
    fn new(config: &TokenizerConfig) -> Result<Self> {
        Ok(Self { inner: SentencePieceTokenizer::new(config)? })
    }
}

impl Tokenizer for LlamaTokenizer {
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        self.inner.encode(text, add_bos, add_special)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.inner.decode(tokens)
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        self.inner.token_to_piece(token)
    }
}

struct TiktokenTokenizer {
    bpe: CoreBPE,
    config: TokenizerConfig,
}

impl TiktokenTokenizer {
    fn new(config: &TokenizerConfig) -> Result<Self> {
        let bpe = cl100k_base().map_err(|e| {
            BitNetError::Model(ModelError::LoadingFailed {
                reason: format!("tiktoken init error: {e}"),
            })
        })?;
        Ok(Self { bpe, config: config.clone() })
    }
}

impl Tokenizer for TiktokenTokenizer {
    fn encode(&self, text: &str, _add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        let tokens = if add_special {
            self.bpe.encode_with_special_tokens(text)
        } else {
            self.bpe.encode_ordinary(text)
        };
        Ok(tokens)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.bpe.decode(tokens.to_vec()).map_err(|e| {
            BitNetError::Model(ModelError::LoadingFailed {
                reason: format!("decode error: {e}"),
            })
        })
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        self.bpe.decode(vec![token]).ok()
    }
}

struct FalconTokenizer {
    bpe: CoreBPE,
    config: TokenizerConfig,
}

impl FalconTokenizer {
    fn new(config: &TokenizerConfig) -> Result<Self> {
        let bpe = p50k_base().map_err(|e| {
            BitNetError::Model(ModelError::LoadingFailed {
                reason: format!("tiktoken init error: {e}"),
            })
        })?;
        Ok(Self { bpe, config: config.clone() })
    }
}

impl Tokenizer for FalconTokenizer {
    fn encode(&self, text: &str, _add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        let tokens = if add_special {
            self.bpe.encode_with_special_tokens(text)
        } else {
            self.bpe.encode_ordinary(text)
        };
        Ok(tokens)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.bpe.decode(tokens.to_vec()).map_err(|e| {
            BitNetError::Model(ModelError::LoadingFailed {
                reason: format!("decode error: {e}"),
            })
        })
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        self.bpe.decode(vec![token]).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_universal_tokenizer_detection() {
        // Test GPT-2 detection
        let mut config = TokenizerConfig::default();
        config.model_type = "gpt2".to_string();
        config.vocab_size = 50257;

        let tokenizer = UniversalTokenizer::new(config).unwrap();
        assert_eq!(tokenizer.vocab_size(), 50257);

        // Test auto-fix for missing pre-tokenizer
        // This would be tested with actual GGUF files
    }
}
