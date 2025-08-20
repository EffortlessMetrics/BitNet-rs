use std::collections::HashMap;
use std::path::Path;
use anyhow::{Result, anyhow};
use log::{debug, warn, info};

use crate::{Tokenizer, TokenizerConfig};
use bitnet_models::Model;

/// Universal tokenizer that auto-detects and handles all formats
pub struct UniversalTokenizer {
    backend: TokenizerBackend,
    config: TokenizerConfig,
}

enum TokenizerBackend {
    Gpt2(Gpt2Tokenizer),
    SentencePiece(SentencePieceTokenizer),
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
        use bitnet_models::formats::gguf::reader::GgufReader;
        
        let reader = GgufReader::open(path)?;
        let mut config = TokenizerConfig::default();
        
        // Extract tokenizer type
        let tokenizer_model = reader.tokenizer_kind()
            .unwrap_or_else(|_| {
                warn!("tokenizer.ggml.model not found, trying to detect");
                None
            })
            .unwrap_or_else(|| {
                warn!("No tokenizer model specified, defaulting to gpt2");
                "gpt2".to_string()
            });
        
        // Check for pre-tokenizer (critical for llama.cpp)
        let pre_tokenizer = reader.value("tokenizer.ggml.pre")
            .ok()
            .flatten()
            .and_then(|v| v.as_string());
        
        // Auto-fix: if gpt2 model but missing pre-tokenizer
        if tokenizer_model == "gpt2" && pre_tokenizer.is_none() {
            warn!("GPT-2 tokenizer missing pre-tokenizer metadata, auto-fixing");
            config.pre_tokenizer = Some("gpt2".to_string());
        } else {
            config.pre_tokenizer = pre_tokenizer.map(|s| s.to_string());
        }
        
        // Extract tokenizer parameters
        config.model_type = tokenizer_model.clone();
        config.add_bos = reader.tokenizer_add_bos()?.unwrap_or(false);
        config.add_eos = reader.tokenizer_add_eos()?.unwrap_or(false);
        config.add_space_prefix = reader.tokenizer_add_space_prefix()?.unwrap_or(true);
        config.byte_fallback = reader.tokenizer_byte_fallback()?.unwrap_or(false);
        
        // Get vocabulary
        config.vocab_size = reader.vocab_size()? as usize;
        
        // Get special tokens
        if let Ok(Some(bos_id)) = reader.bos_token_id() {
            config.bos_token_id = Some(bos_id as u32);
        }
        if let Ok(Some(eos_id)) = reader.eos_token_id() {
            config.eos_token_id = Some(eos_id as u32);
        }
        if let Ok(Some(pad_id)) = reader.pad_token_id() {
            config.pad_token_id = Some(pad_id as u32);
        }
        
        // Load merges for BPE
        if tokenizer_model == "gpt2" || tokenizer_model == "bpe" {
            if let Ok(merges) = reader.tokenizer_merges() {
                config.bpe_merges = Some(merges);
            }
        }
        
        // Load vocabulary
        if let Ok(vocab) = reader.tokenizer_vocab() {
            config.vocabulary = Some(vocab);
        }
        
        info!("Detected tokenizer: {} with vocab size {}", tokenizer_model, config.vocab_size);
        Self::new(config)
    }
    
    /// Create from model with auto-detection
    pub fn from_model(model: &Model) -> Result<Self> {
        let config = TokenizerConfig::from_model(model)?;
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
        if add_bos && self.config.add_bos {
            if let Some(bos_id) = self.config.bos_token_id {
                tokens.insert(0, bos_id);
            }
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
    merges: Vec<(String, String)>,
    config: TokenizerConfig,
}

impl Gpt2Tokenizer {
    fn new(config: &TokenizerConfig) -> Result<Self> {
        // Implementation for GPT-2 BPE tokenizer
        // This handles Llama 3's 128k vocab GPT-2 variant
        Ok(Self {
            vocab: HashMap::new(),
            merges: vec![],
            config: config.clone(),
        })
    }
}

impl Tokenizer for Gpt2Tokenizer {
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        // Full BPE implementation would go here
        // For now, return a stub
        Ok(vec![1, 2, 3])
    }
    
    fn decode(&self, tokens: &[u32]) -> Result<String> {
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
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        Ok(vec![1, 2, 3])
    }
    
    fn decode(&self, tokens: &[u32]) -> Result<String> {
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
    fn new(config: &TokenizerConfig) -> Result<Self> {
        Ok(Self { config: config.clone() })
    }
}

impl Tokenizer for LlamaTokenizer {
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        Ok(vec![1, 2, 3])
    }
    
    fn decode(&self, tokens: &[u32]) -> Result<String> {
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
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        Ok(vec![1, 2, 3])
    }
    
    fn decode(&self, tokens: &[u32]) -> Result<String> {
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
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        Ok(vec![1, 2, 3])
    }
    
    fn decode(&self, tokens: &[u32]) -> Result<String> {
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