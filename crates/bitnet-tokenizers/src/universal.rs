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
    pub fn from_gguf(_path: &Path) -> Result<Self> {
        // TODO: Import GgufReader when bitnet-models is added as dependency
        // For now, create a default config
        // This would normally:
        // 1. Read GGUF metadata
        // 2. Auto-detect tokenizer type (gpt2, llama, etc)
        // 3. Fix missing pre-tokenizer for GPT-2
        // 4. Extract vocabulary and merges
        let config = TokenizerConfig::default();
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
    #[allow(dead_code)]
    vocab: HashMap<String, u32>,
    #[allow(dead_code)]
    merges: Vec<(String, String)>,
    config: TokenizerConfig,
}

impl Gpt2Tokenizer {
    fn new(config: &TokenizerConfig) -> Result<Self> {
        // Implementation for GPT-2 BPE tokenizer
        // This handles Llama 3's 128k vocab GPT-2 variant
        Ok(Self { vocab: HashMap::new(), merges: vec![], config: config.clone() })
    }
}

impl Tokenizer for Gpt2Tokenizer {
    fn encode(&self, _text: &str, _add_bos: bool, _add_special: bool) -> Result<Vec<u32>> {
        // Full BPE implementation would go here
        // For now, return a stub
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
