use crate::Tokenizer;
use bitnet_common::{BitNetError, ModelError, Result};
use bitnet_models::{GgufReader, loader::MmapFile};
use bitnet_tokenizer_byte_core::ByteVocabulary;
use std::collections::HashMap;
use std::path::Path;

/// Tokenizer that reads vocab from GGUF files
pub struct GgufTokenizer {
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    byte_to_id: [Option<u32>; 256],
    id_to_byte: HashMap<u32, u8>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
}

impl GgufTokenizer {
    pub fn from_gguf_file(path: &Path) -> Result<Self> {
        // Parse metadata and vocabulary from GGUF file
        let (tokens, bos_token_id, eos_token_id) = read_gguf_metadata(path)?;

        // Extract vocabulary and byte-level mappings
        let byte_vocab = ByteVocabulary::from_tokens(&tokens);

        Ok(Self {
            vocab: byte_vocab.vocab().clone(),
            reverse_vocab: byte_vocab.reverse_vocab().clone(),
            byte_to_id: *byte_vocab.byte_to_id(),
            id_to_byte: byte_vocab.id_to_byte().clone(),
            bos_token_id,
            eos_token_id,
        })
    }
}

impl Tokenizer for GgufTokenizer {
    fn encode(&self, text: &str, add_bos: bool, _add_special: bool) -> Result<Vec<u32>> {
        // Simple byte-level tokenization (like GPT-2)
        let mut tokens = Vec::new();

        if add_bos && let Some(bos) = self.bos_token_id {
            tokens.push(bos);
        }

        // Convert text to bytes and lookup in byte mapping
        for byte in text.bytes() {
            if let Some(id) = self.byte_to_id[byte as usize] {
                tokens.push(id);
            } else {
                // Fallback to direct byte value if not in vocab
                tokens.push(byte as u32);
            }
        }

        Ok(tokens)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut text = String::new();
        let mut byte_buf: Vec<u8> = Vec::new();

        for &token in tokens {
            if let Some(&byte) = self.id_to_byte.get(&token) {
                byte_buf.push(byte);
            } else if token < 256 {
                // Direct byte value
                byte_buf.push(token as u8);
            } else if let Some(token_str) = self.reverse_vocab.get(&token) {
                if !byte_buf.is_empty() {
                    text.push_str(&String::from_utf8_lossy(&byte_buf));
                    byte_buf.clear();
                }
                text.push_str(token_str);
            }
        }

        if !byte_buf.is_empty() {
            text.push_str(&String::from_utf8_lossy(&byte_buf));
        }

        Ok(text)
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        if let Some(&byte) = self.id_to_byte.get(&token) {
            Some(String::from_utf8_lossy(&[byte]).to_string())
        } else if let Some(piece) = self.reverse_vocab.get(&token) {
            Some(piece.clone())
        } else if token < 256 {
            Some(String::from_utf8_lossy(&[token as u8]).to_string())
        } else {
            None
        }
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
