use bitnet_common::{BitNetError, ModelError, Result};
use bitnet_models::{GgufReader, loader::MmapFile};
use std::collections::HashMap;
#[cfg(feature = "spm")]
use std::io::Write;
use std::path::Path;
use tracing::{debug, warn};

#[cfg(feature = "spm")]
use sentencepiece;

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

    /// Create a tokenizer directly from a GGUF file.
    ///
    /// This reads the tokenizer metadata via [`GgufReader`] and constructs
    /// the appropriate tokenizer backend. Both BPE based tokenizers (GPT‑2,
    /// Tiktoken, Falcon, ... ) and SentencePiece models are supported.
    pub fn from_gguf(path: &Path) -> Result<Self> {
        // Memory map the file and create a reader
        let mmap = MmapFile::open(path)?;
        let reader = GgufReader::new(mmap.as_slice())?;

        // If the GGUF contains a binary `tokenizer.ggml.model` blob we treat
        // it as a SentencePiece model. Otherwise we fall back to the BPE style
        // metadata used by GPT‑2 style tokenizers.
        if let Some(bytes) = reader.get_bin_or_u8_array("tokenizer.ggml.model") {
            // SentencePiece model
            let bos = reader.get_u32_metadata("tokenizer.ggml.bos_token_id");
            let eos = reader.get_u32_metadata("tokenizer.ggml.eos_token_id");
            let config = TokenizerConfig {
                model_type: "sentencepiece".to_string(),
                vocab_size: 0,
                bos_token_id: bos,
                eos_token_id: eos,
                ..Default::default()
            };
            let backend = TokenizerBackend::SentencePiece(SentencePieceTokenizer::from_bytes(
                &bytes,
                config.clone(),
            )?);
            return Ok(Self { backend, config });
        }

        // BPE/GPT‑2 style tokenizer metadata
        let tokens = reader.get_string_array_metadata("tokenizer.ggml.tokens").ok_or(
            BitNetError::Model(ModelError::LoadingFailed {
                reason: "GGUF missing tokenizer.ggml.tokens".to_string(),
            }),
        )?;

        let merges = reader.get_string_array_metadata("tokenizer.ggml.merges");
        let model_type = reader
            .get_string_metadata("tokenizer.ggml.model")
            .unwrap_or_else(|| "gpt2".to_string());
        let bos = reader.get_u32_metadata("tokenizer.ggml.bos_token_id");
        let eos = reader.get_u32_metadata("tokenizer.ggml.eos_token_id");
        let add_space_prefix =
            reader.get_bool_metadata("tokenizer.ggml.add_space_prefix").unwrap_or(false);
        let byte_fallback =
            reader.get_bool_metadata("tokenizer.ggml.byte_fallback").unwrap_or(false);

        let vocab: Vec<(String, f32)> = tokens.into_iter().map(|s| (s, 0.0)).collect();

        let config = TokenizerConfig {
            model_type,
            vocab_size: vocab.len(),
            vocabulary: Some(vocab),
            bpe_merges: merges,
            add_bos: bos.is_some(),
            add_eos: eos.is_some(),
            add_space_prefix,
            byte_fallback,
            bos_token_id: bos,
            eos_token_id: eos,
            ..Default::default()
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

// Actual implementations for different tokenizer backends.

/// GPT‑2 style BPE tokenizer (also used for Tiktoken and Falcon models).
struct Gpt2Tokenizer {
    inner: Option<tokenizers::Tokenizer>,
    config: TokenizerConfig,
}

impl Gpt2Tokenizer {
    fn new(config: &TokenizerConfig) -> Result<Self> {
        use tokenizers::Tokenizer as HfTokenizer;
        use tokenizers::models::bpe::BPE;
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;

        // If we have vocab/merges build a proper BPE tokenizer, otherwise we
        // fall back to simple byte level encoding.
        let inner = if let (Some(vocab), Some(merges)) = (&config.vocabulary, &config.bpe_merges) {
            let vocab_map: HashMap<String, u32> =
                vocab.iter().enumerate().map(|(i, (s, _))| (s.clone(), i as u32)).collect();
            let merge_pairs: Vec<(String, String)> = merges
                .iter()
                .filter_map(|m| {
                    let mut parts = m.split_whitespace();
                    Some((parts.next()?.to_string(), parts.next()?.to_string()))
                })
                .collect();
            let bpe =
                BPE::builder().vocab_and_merges(vocab_map, merge_pairs).build().map_err(|e| {
                    BitNetError::Model(ModelError::LoadingFailed {
                        reason: format!("BPE build error: {e}"),
                    })
                })?;
            let mut tokenizer = HfTokenizer::new(bpe);
            tokenizer
                .with_pre_tokenizer(ByteLevel::default().add_prefix_space(config.add_space_prefix));
            tokenizer.with_decoder(tokenizers::decoders::byte_level::ByteLevel::default());
            Some(tokenizer)
        } else {
            None
        };

        Ok(Self { inner, config: config.clone() })
    }
}

impl Tokenizer for Gpt2Tokenizer {
    fn encode(&self, text: &str, _add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        use tokenizers::EncodeInput;
        if let Some(inner) = &self.inner {
            let enc = inner.encode(EncodeInput::Single(text.into()), add_special).map_err(|e| {
                BitNetError::Model(ModelError::LoadingFailed {
                    reason: format!("Tokenizer encode error: {e}"),
                })
            })?;
            Ok(enc.get_ids().to_vec())
        } else {
            // simple byte level fallback
            Ok(text.bytes().map(|b| b as u32).collect())
        }
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        if let Some(inner) = &self.inner {
            inner.decode(tokens, true).map_err(|e| {
                BitNetError::Model(ModelError::LoadingFailed {
                    reason: format!("Tokenizer decode error: {e}"),
                })
            })
        } else {
            Ok(tokens.iter().map(|&t| t as u8 as char).collect())
        }
    }

    fn vocab_size(&self) -> usize {
        if let Some(inner) = &self.inner {
            inner.get_vocab_size(false)
        } else {
            self.config.vocab_size.max(256)
        }
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        if let Some(inner) = &self.inner {
            inner.id_to_token(token).map(|s| s.to_string())
        } else {
            Some((token as u8 as char).to_string())
        }
    }
}

/// SentencePiece tokenizer backend.
struct SentencePieceTokenizer {
    #[cfg(feature = "spm")]
    processor: Option<sentencepiece::SentencePieceProcessor>,
    config: TokenizerConfig,
}

impl SentencePieceTokenizer {
    fn new(config: &TokenizerConfig) -> Result<Self> {
        Ok(Self {
            #[cfg(feature = "spm")]
            processor: None,
            config: config.clone(),
        })
    }

    #[cfg(feature = "spm")]
    #[allow(dead_code)]
    fn from_bytes(bytes: &[u8], config: TokenizerConfig) -> Result<Self> {
        let mut tmp = tempfile::NamedTempFile::new()?;
        tmp.write_all(bytes)?;
        let processor = sentencepiece::SentencePieceProcessor::open(tmp.path()).map_err(|e| {
            BitNetError::Model(ModelError::LoadingFailed {
                reason: format!("Failed to load SentencePiece model: {e}"),
            })
        })?;
        Ok(Self { processor: Some(processor), config })
    }

    #[cfg(not(feature = "spm"))]
    fn from_bytes(_bytes: &[u8], _config: TokenizerConfig) -> Result<Self> {
        Err(BitNetError::Model(ModelError::LoadingFailed {
            reason: "SentencePiece support not compiled in".to_string(),
        }))
    }
}

impl Tokenizer for SentencePieceTokenizer {
    fn encode(&self, text: &str, _add_bos: bool, _add_special: bool) -> Result<Vec<u32>> {
        #[cfg(feature = "spm")]
        {
            if let Some(proc) = &self.processor {
                let pieces = proc.encode(text).map_err(|e| {
                    BitNetError::Model(ModelError::LoadingFailed {
                        reason: format!("Tokenizer encode error: {e}"),
                    })
                })?;
                let mut ids: Vec<u32> = pieces.into_iter().map(|p| p.id).collect();
                if _add_bos {
                    if let Some(b) = self.config.bos_token_id {
                        ids.insert(0, b);
                    }
                }
                return Ok(ids);
            }
        }
        Ok(text.bytes().map(|b| b as u32).collect())
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        #[cfg(feature = "spm")]
        {
            if let Some(proc) = &self.processor {
                return proc.decode_piece_ids(tokens).map_err(|e| {
                    BitNetError::Model(ModelError::LoadingFailed {
                        reason: format!("Tokenizer decode error: {e}"),
                    })
                });
            }
        }
        Ok(tokens.iter().map(|&t| t as u8 as char).collect())
    }

    fn vocab_size(&self) -> usize {
        #[cfg(feature = "spm")]
        {
            if let Some(proc) = &self.processor {
                return proc.len();
            }
        }
        self.config.vocab_size.max(256)
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        #[cfg(feature = "spm")]
        {
            if let Some(proc) = &self.processor {
                return proc.decode_piece_ids(&[token]).ok();
            }
        }
        Some((token as u8 as char).to_string())
    }
}

/// Llama tokenizer – essentially a SentencePiece tokenizer.
struct LlamaTokenizer {
    inner: SentencePieceTokenizer,
}

impl LlamaTokenizer {
    #[allow(dead_code)]
    fn new(config: &TokenizerConfig) -> Result<Self> {
        Ok(Self { inner: SentencePieceTokenizer::new(config)? })
    }

    #[allow(dead_code)]
    fn from_bytes(bytes: &[u8], config: TokenizerConfig) -> Result<Self> {
        Ok(Self { inner: SentencePieceTokenizer::from_bytes(bytes, config)? })
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

/// Tiktoken tokenizer – wrapper around GPT‑2 BPE implementation.
struct TiktokenTokenizer {
    inner: Gpt2Tokenizer,
}

impl TiktokenTokenizer {
    fn new(config: &TokenizerConfig) -> Result<Self> {
        Ok(Self { inner: Gpt2Tokenizer::new(config)? })
    }
}

impl Tokenizer for TiktokenTokenizer {
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

/// Falcon tokenizer – also GPT‑2 style BPE.
struct FalconTokenizer {
    inner: Gpt2Tokenizer,
}

impl FalconTokenizer {
    fn new(config: &TokenizerConfig) -> Result<Self> {
        Ok(Self { inner: Gpt2Tokenizer::new(config)? })
    }
}

impl Tokenizer for FalconTokenizer {
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
