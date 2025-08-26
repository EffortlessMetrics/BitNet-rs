use crate::Tokenizer;
use bitnet_common::{BitNetError, ModelError, Result};
use bitnet_models::{GgufReader, loader::MmapFile};
use std::collections::HashMap;
use std::path::Path;

/// Tokenizer that reads vocab from GGUF files
pub struct GgufTokenizer {
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
}

impl GgufTokenizer {
    pub fn from_gguf_file(path: &Path) -> Result<Self> {
        // Parse metadata and vocabulary from GGUF file
        let (tokens, bos_token_id, eos_token_id) = read_gguf_metadata(path)?;

        // Extract vocabulary
        let vocab = extract_vocab(&tokens);
        let reverse_vocab: HashMap<u32, String> =
            vocab.iter().map(|(k, v)| (*v, k.clone())).collect();

        Ok(Self { vocab, reverse_vocab, bos_token_id, eos_token_id })
    }
}

impl Tokenizer for GgufTokenizer {
    fn encode(&self, text: &str, add_bos: bool, _add_special: bool) -> Result<Vec<u32>> {
        // Simple byte-level tokenization (like GPT-2)
        let mut tokens = Vec::new();

        if add_bos && let Some(bos) = self.bos_token_id {
            tokens.push(bos);
        }

        // Convert text to bytes and lookup in vocab
        for byte in text.bytes() {
            let byte_str = format!("<0x{:02X}>", byte);
            if let Some(&token_id) = self.vocab.get(&byte_str) {
                tokens.push(token_id);
            } else {
                // Fallback to direct byte value if not in vocab
                tokens.push(byte as u32);
            }
        }

        Ok(tokens)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut text = String::new();

        for &token in tokens {
            if let Some(token_str) = self.reverse_vocab.get(&token) {
                // Handle byte tokens
                if token_str.starts_with("<0x") && token_str.ends_with(">") {
                    if let Ok(byte_val) = u8::from_str_radix(&token_str[3..5], 16) {
                        text.push(byte_val as char);
                    }
                } else {
                    text.push_str(token_str);
                }
            } else if token < 256 {
                // Direct byte value
                text.push(token as u8 as char);
            }
        }

        Ok(text)
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        self.reverse_vocab.get(&token).cloned()
    }

    fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }
}

fn read_gguf_metadata(path: &Path) -> Result<(Vec<String>, Option<u32>, Option<u32>)> {
    let mmap = MmapFile::open(path)?;
    let reader = GgufReader::new(mmap.as_slice())?;

    let tokens =
        reader.get_string_array_metadata("tokenizer.ggml.tokens").ok_or(BitNetError::Model(
            ModelError::LoadingFailed { reason: "GGUF missing tokenizer.ggml.tokens".to_string() },
        ))?;

    let bos = reader.get_u32_metadata("tokenizer.ggml.bos_token_id");
    let eos = reader.get_u32_metadata("tokenizer.ggml.eos_token_id");

    Ok((tokens, bos, eos))
}

fn extract_vocab(tokens: &[String]) -> HashMap<String, u32> {
    let mut vocab = HashMap::with_capacity(tokens.len());
    for (i, token) in tokens.iter().enumerate() {
        vocab.insert(token.clone(), i as u32);
    }
    vocab
}
