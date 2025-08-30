use bitnet_common::{BitNetError, ModelError, Result};
use bitnet_models::{loader::MmapFile, GgufReader};
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
    #[cfg(not(feature = "spm"))]
    SentencePiece(()),
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

    /// Create from GGUF model with auto-detection
    pub fn from_gguf(path: &Path) -> Result<Self> {
        let mmap = MmapFile::open(path)?;
        let reader = GgufReader::new(mmap.as_slice())?;

        // If the GGUF contains an embedded SentencePiece model, prefer that
        if let Some(bytes) = reader.get_bin_or_u8_array("tokenizer.ggml.model") {
            #[cfg(feature = "spm")]
            {
                let tokens =
                    reader.get_string_array_metadata("tokenizer.ggml.tokens").unwrap_or_default();
                let bos = reader.get_u32_metadata("tokenizer.ggml.bos_token_id");
                let eos = reader.get_u32_metadata("tokenizer.ggml.eos_token_id");
                let config = TokenizerConfig {
                    model_type: reader
                        .get_string_metadata("tokenizer.ggml.model")
                        .unwrap_or_else(|| "sentencepiece".to_string()),
                    vocab_size: tokens.len(),
                    add_bos: bos.is_some(),
                    add_eos: eos.is_some(),
                    bos_token_id: bos,
                    eos_token_id: eos,
                    vocabulary: Some(tokens.into_iter().map(|t| (t, 0.0)).collect()),
                    sp_model: Some(bytes),
                    ..Default::default()
                };
                return Self::new(config);
            }
            #[cfg(not(feature = "spm"))]
            {
                let _ = bytes;
                return Err(BitNetError::Model(ModelError::LoadingFailed {
                    reason: "SentencePiece support not compiled in".to_string(),
                }));
            }
        }

        // Otherwise, assume GPT-2 style BPE tokenizer
        let tokens = reader
            .get_string_array_metadata("tokenizer.ggml.tokens")
            .ok_or(BitNetError::Model(ModelError::LoadingFailed {
                reason: "GGUF missing tokenizer.ggml.tokens".to_string(),
            }))?;
        let merges = reader.get_string_array_metadata("tokenizer.ggml.merges");
        let bos = reader.get_u32_metadata("tokenizer.ggml.bos_token_id");
        let eos = reader.get_u32_metadata("tokenizer.ggml.eos_token_id");
        let pad = reader.get_u32_metadata("tokenizer.ggml.pad_token_id");
        let unk = reader.get_u32_metadata("tokenizer.ggml.unk_token_id");
        let config = TokenizerConfig {
            model_type: reader
                .get_string_metadata("tokenizer.ggml.model")
                .unwrap_or_else(|| "gpt2".to_string()),
            vocab_size: tokens.len(),
            add_bos: bos.is_some(),
            add_eos: eos.is_some(),
            add_space_prefix: reader
                .get_bool_metadata("tokenizer.ggml.add_space_prefix")
                .unwrap_or(false),
            byte_fallback: reader
                .get_bool_metadata("tokenizer.ggml.byte_fallback")
                .unwrap_or(false),
            bos_token_id: bos,
            eos_token_id: eos,
            pad_token_id: pad,
            unk_token_id: unk,
            vocabulary: Some(tokens.into_iter().map(|t| (t, 0.0)).collect()),
            bpe_merges: merges,
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
                #[cfg(feature = "spm")]
                {
                    Ok(TokenizerBackend::SentencePiece(SentencePieceTokenizer::new(config)?))
                }
                #[cfg(not(feature = "spm"))]
                {
                    Err(BitNetError::Model(ModelError::LoadingFailed {
                        reason: "SentencePiece support not compiled in".to_string(),
                    }))
                }
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
            #[cfg(not(feature = "spm"))]
            TokenizerBackend::SentencePiece(_) => unreachable!(),
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
            #[cfg(not(feature = "spm"))]
            TokenizerBackend::SentencePiece(_) => unreachable!(),
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
            #[cfg(not(feature = "spm"))]
            TokenizerBackend::SentencePiece(_) => unreachable!(),
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
        let vocab_list = config.vocabulary.clone().ok_or(BitNetError::Model(
            ModelError::LoadingFailed { reason: "missing vocabulary".to_string() },
        ))?;
        let mut vocab = HashMap::new();
        for (i, (tok, _)) in vocab_list.iter().enumerate() {
            vocab.insert(tok.clone(), i as u32);
        }

        let merges = config
            .bpe_merges
            .clone()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|m| {
                let mut parts = m.split_whitespace();
                Some((parts.next()?.to_string(), parts.next()?.to_string()))
            })
            .collect::<Vec<_>>();

        let bpe = tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .map_err(|e| BitNetError::Model(ModelError::LoadingFailed {
                reason: format!("BPE build error: {e}"),
            }))?;

        let inner = tokenizers::Tokenizer::new(bpe);
        Ok(Self { inner, config: config.clone() })
    }
}

impl Tokenizer for Gpt2Tokenizer {
    fn encode(&self, text: &str, _add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        use tokenizers::EncodeInput;
        let enc = self
            .inner
            .encode(EncodeInput::Single(text.into()), add_special)
            .map_err(|e| BitNetError::Model(ModelError::LoadingFailed {
                reason: format!("Tokenizer encode error: {e}"),
            }))?;
        Ok(enc.get_ids().to_vec())
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.inner
            .decode(tokens, true)
            .map_err(|e| BitNetError::Model(ModelError::LoadingFailed {
                reason: format!("Tokenizer decode error: {e}"),
            }))
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        self.inner.id_to_token(token).map(|s| s.to_string())
    }
}

// Similar stub implementations for other tokenizer types
#[cfg(feature = "spm")]
struct SentencePieceTokenizer {
    sp: sentencepiece::SentencePieceProcessor,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
}

#[cfg(feature = "spm")]
impl SentencePieceTokenizer {
    fn new(config: &TokenizerConfig) -> Result<Self> {
        let bytes = config.sp_model.as_ref().ok_or(BitNetError::Model(
            ModelError::LoadingFailed { reason: "missing SentencePiece model".to_string() },
        ))?;
        use std::io::Write;
        let mut tmp = tempfile::NamedTempFile::new().map_err(BitNetError::Io)?;
        tmp.write_all(bytes).map_err(BitNetError::Io)?;
        let sp = sentencepiece::SentencePieceProcessor::open(tmp.path()).map_err(|e| {
            BitNetError::Io(std::io::Error::other(format!(
                "Failed to load SentencePiece model: {e}",
            )))
        })?;
        Ok(Self { sp, bos_token_id: config.bos_token_id, eos_token_id: config.eos_token_id })
    }
}

#[cfg(feature = "spm")]
impl Tokenizer for SentencePieceTokenizer {
    fn encode(&self, text: &str, add_bos: bool, _add_special: bool) -> Result<Vec<u32>> {
        let pieces = self
            .sp
            .encode(text)
            .map_err(|e| BitNetError::Io(std::io::Error::other(format!("encode failed: {e}"))))?;
        let mut ids: Vec<u32> = pieces.into_iter().map(|p| p.id).collect();
        if add_bos && let Some(b) = self.bos_token_id {
            ids.insert(0, b);
        }
        Ok(ids)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.sp
            .decode_piece_ids(tokens)
            .map_err(|e| BitNetError::Io(std::io::Error::other(format!(
                "decode_piece_ids failed: {e}",
            ))))
    }

    fn vocab_size(&self) -> usize {
        self.sp.len()
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        self.sp.decode_piece_ids(&[token]).ok()
    }

    fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
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
        // Test GPT-2 detection with minimal vocabulary
        let config = TokenizerConfig {
            model_type: "gpt2".to_string(),
            vocab_size: 4,
            vocabulary: Some(vec![
                ("[UNK]".to_string(), 0.0),
                ("a".to_string(), 0.0),
                ("b".to_string(), 0.0),
                ("ab".to_string(), 0.0),
            ]),
            bpe_merges: Some(vec!["a b".to_string()]),
            unk_token_id: Some(0),
            ..Default::default()
        };

        let tokenizer = UniversalTokenizer::new(config).unwrap();
        assert_eq!(tokenizer.vocab_size(), 4);
    }
}
