//! Core tokenizer contracts and baseline fallback tokenizer implementation.

use bitnet_common::{BitNetError, Result};

/// Configuration for tokenizer initialization
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TokenizerConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub pre_tokenizer: Option<String>,
    pub add_bos: bool,
    pub add_eos: bool,
    pub add_space_prefix: bool,
    pub byte_fallback: bool,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub unk_token_id: Option<u32>,
    pub vocabulary: Option<Vec<(String, f32)>>,
    pub bpe_merges: Option<Vec<String>>,
}

impl TokenizerConfig {
    /// Create a default config
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Tokenizer trait
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>>;
    fn decode(&self, tokens: &[u32]) -> Result<String>;
    fn vocab_size(&self) -> usize;
    fn token_to_piece(&self, token: u32) -> Option<String>;

    /// Convert a token string to its token ID
    fn token_to_id(&self, _token: &str) -> Option<u32> {
        None
    }

    /// Real vocabulary size from tokenizer model (no padding)
    fn real_vocab_size(&self) -> usize {
        self.vocab_size()
    }

    /// Legacy encode method - calls new encode with sensible defaults
    fn encode_legacy(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.encode(text, true, add_special_tokens)
    }

    /// Legacy decode method - ignores skip_special_tokens parameter
    fn decode_legacy(&self, tokens: &[u32], _skip_special_tokens: bool) -> Result<String> {
        self.decode(tokens)
    }

    /// BOS token ID getter - returns None by default
    fn bos_token_id(&self) -> Option<u32> {
        None
    }

    /// EOS token ID getter - returns None by default
    fn eos_token_id(&self) -> Option<u32> {
        None
    }

    /// Legacy PAD token ID getter - returns None by default
    fn pad_token_id(&self) -> Option<u32> {
        None
    }

    /// Returns true if the given token ID is a known special token (BOS, EOS, or PAD).
    fn is_special_token(&self, id: u32) -> bool {
        self.bos_token_id() == Some(id)
            || self.eos_token_id() == Some(id)
            || self.pad_token_id() == Some(id)
    }

    /// Returns the tokenizer family name based on known special tokens.
    fn get_family_name(&self) -> &'static str {
        if self.token_to_id("<|eot_id|>").is_some()
            || self.token_to_id("<|start_header_id|>").is_some()
        {
            "llama3"
        } else if self.token_to_id("[INST]").is_some() {
            "mistral-instruct"
        } else {
            "unknown"
        }
    }
}

/// Basic tokenizer implementation
pub struct BasicTokenizer {
    vocab_size: usize,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    pad_token_id: Option<u32>,
}

impl BasicTokenizer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            vocab_size: 50257,
            bos_token_id: None,
            eos_token_id: Some(50256),
            pad_token_id: None,
        }
    }

    #[must_use]
    pub fn with_config(
        vocab_size: usize,
        bos_token_id: Option<u32>,
        eos_token_id: Option<u32>,
        pad_token_id: Option<u32>,
    ) -> Self {
        Self { vocab_size, bos_token_id, eos_token_id, pad_token_id }
    }
}

impl Default for BasicTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for BasicTokenizer {
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let mut tokens: Vec<u32> = Vec::with_capacity(text.len() + 2);

        if add_bos && let Some(bos) = self.bos_token_id {
            tokens.push(bos);
        }

        for byte in text.bytes() {
            let id = byte as u32;
            if id >= self.vocab_size as u32 {
                return Err(BitNetError::Config(format!(
                    "byte value {id} exceeds vocab_size {}",
                    self.vocab_size
                )));
            }
            tokens.push(id);
        }

        if add_special {
            if let Some(eos_id) = self.eos_token_id {
                tokens.push(eos_id);
            }
            if let Some(pad_id) = self.pad_token_id {
                tokens.push(pad_id);
            }
        }

        Ok(tokens)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        if tokens.is_empty() {
            return Ok(String::new());
        }

        let mut byte_buf: Vec<u8> = Vec::with_capacity(tokens.len());
        for &id in tokens {
            let is_special = matches!(
                Some(id),
                Some(x) if self.bos_token_id == Some(x)
                    || self.eos_token_id == Some(x)
                    || self.pad_token_id == Some(x)
            );
            if is_special {
                continue;
            }
            if id < 256 {
                byte_buf.push(id as u8);
            }
        }

        Ok(String::from_utf8_lossy(&byte_buf).into_owned())
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        if token < 256 {
            let byte = token as u8;
            Some(String::from_utf8_lossy(&[byte]).into_owned())
        } else {
            Some(format!("<token_{}>", token))
        }
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    fn pad_token_id(&self) -> Option<u32> {
        self.pad_token_id
    }
}
