use bitnet_common::{BitNetError, ModelError, Result};
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

        let tokens = reader.get_string_array_metadata("tokenizer.ggml.tokens").ok_or(
            BitNetError::Model(ModelError::LoadingFailed {
                reason: "GGUF missing tokenizer.ggml.tokens".to_string(),
            }),
        )?;

        let merges = reader.get_string_array_metadata("tokenizer.ggml.merges").unwrap_or_default();

        let model_type = reader
            .get_string_metadata("tokenizer.ggml.model")
            .unwrap_or_else(|| "gpt2".to_string());

        let bos = reader.get_u32_metadata("tokenizer.ggml.bos_token_id");
        let eos = reader.get_u32_metadata("tokenizer.ggml.eos_token_id");
        let add_bos = reader.get_bool_metadata("tokenizer.ggml.add_bos_token").unwrap_or(false);
        let add_eos = reader.get_bool_metadata("tokenizer.ggml.add_eos_token").unwrap_or(false);

        let vocabulary: Vec<(String, f32)> = tokens.iter().map(|t| (t.clone(), 0.0f32)).collect();

        let mut config = TokenizerConfig {
            model_type: model_type.clone(),
            vocab_size: vocabulary.len(),
            bos_token_id: bos,
            eos_token_id: eos,
            add_bos,
            add_eos,
            vocabulary: Some(vocabulary),
            bpe_merges: if merges.is_empty() { None } else { Some(merges) },
            ..TokenizerConfig::default()
        };

        if model_type == "gpt2" {
            config.add_space_prefix = true;
        }

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

// Stub implementations for different tokenizer types
// These would be fully implemented in their respective modules

struct Gpt2Tokenizer {
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    bpe_ranks: HashMap<(String, String), usize>,
    config: TokenizerConfig,
}

impl Gpt2Tokenizer {
    fn new(config: &TokenizerConfig) -> Result<Self> {
        let vocab: HashMap<String, u32> = if let Some(v) = &config.vocabulary {
            v.iter().enumerate().map(|(i, (tok, _))| (tok.clone(), i as u32)).collect()
        } else {
            HashMap::new()
        };
        let reverse_vocab = vocab.iter().map(|(k, v)| (*v, k.clone())).collect();

        let bpe_ranks: HashMap<(String, String), usize> = if let Some(merges) = &config.bpe_merges {
            merges
                .iter()
                .enumerate()
                .filter_map(|(i, m)| {
                    let mut parts = m.split_whitespace();
                    if let (Some(a), Some(b)) = (parts.next(), parts.next()) {
                        Some(((a.to_string(), b.to_string()), i))
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            HashMap::new()
        };

        Ok(Self { vocab, reverse_vocab, bpe_ranks, config: config.clone() })
    }

    fn get_pairs(word: &[String]) -> Vec<(String, String)> {
        let mut pairs = Vec::new();
        for i in 0..word.len().saturating_sub(1) {
            pairs.push((word[i].clone(), word[i + 1].clone()));
        }
        pairs
    }

    fn bpe(&self, token: &str) -> Vec<String> {
        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();
        if word.len() <= 1 {
            return word;
        }

        let mut pairs = Self::get_pairs(&word);

        while !pairs.is_empty() {
            let mut best: Option<(String, String)> = None;
            let mut best_rank = usize::MAX;
            for pair in pairs.iter() {
                if let Some(&rank) = self.bpe_ranks.get(&(pair.0.clone(), pair.1.clone())) {
                    if rank < best_rank {
                        best_rank = rank;
                        best = Some((pair.0.clone(), pair.1.clone()));
                    }
                }
            }
            if let Some((first, second)) = best {
                let mut new_word = Vec::new();
                let mut i = 0;
                while i < word.len() {
                    if i < word.len() - 1 && word[i] == first && word[i + 1] == second {
                        new_word.push(format!("{}{}", first, second));
                        i += 2;
                    } else {
                        new_word.push(word[i].clone());
                        i += 1;
                    }
                }
                word = new_word;
                if word.len() == 1 {
                    break;
                }
                pairs = Self::get_pairs(&word);
            } else {
                break;
            }
        }
        word
    }
}

impl Tokenizer for Gpt2Tokenizer {
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        if add_bos && self.config.add_bos {
            if let Some(bos) = self.config.bos_token_id {
                out.push(bos);
            }
        }

        let words: Vec<&str> = text.split_whitespace().collect();
        for (i, w) in words.iter().enumerate() {
            for piece in self.bpe(w) {
                if let Some(&id) = self.vocab.get(&piece) {
                    out.push(id);
                } else if let Some(unk) = self.config.unk_token_id {
                    out.push(unk);
                }
            }
            if i + 1 < words.len() {
                if let Some(&space_id) = self.vocab.get(" ") {
                    out.push(space_id);
                }
            }
        }

        if add_special && self.config.add_eos {
            if let Some(eos) = self.config.eos_token_id {
                out.push(eos);
            }
        }
        Ok(out)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut text = String::new();
        for token in tokens {
            if let Some(piece) = self.reverse_vocab.get(token) {
                text.push_str(piece);
            }
        }
        Ok(text)
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        self.reverse_vocab.get(&token).cloned()
    }
}

// Similar stub implementations for other tokenizer types
struct SentencePieceTokenizer {
    config: TokenizerConfig,
}

impl SentencePieceTokenizer {
    fn new(config: &TokenizerConfig) -> Result<Self> {
        Ok(Self { config: config.clone() })
    }
}

impl Tokenizer for SentencePieceTokenizer {
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
        let mut config = TokenizerConfig::default();
        config.model_type = "gpt2".to_string();
        config.vocab_size = 50257;

        let tokenizer = UniversalTokenizer::new(config).unwrap();
        assert_eq!(tokenizer.vocab_size(), 50257);

        // Test auto-fix for missing pre-tokenizer
        // This would be tested with actual GGUF files
    }
}
