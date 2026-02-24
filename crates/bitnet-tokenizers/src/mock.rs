/// Mock tokenizer for testing
use crate::Tokenizer;
use bitnet_common::Result;
use std::collections::HashMap;

pub struct MockTokenizer {
    vocab_size: usize,
    token_to_id_map: HashMap<String, u32>,
}

impl MockTokenizer {
    pub fn new() -> Self {
        Self { vocab_size: 50257, token_to_id_map: HashMap::new() }
    }

    /// Create a mock tokenizer with predefined token-to-ID mappings
    pub fn with_special_tokens(mappings: &[(&str, u32)]) -> Self {
        let mut token_to_id_map = HashMap::new();
        for (token, id) in mappings {
            token_to_id_map.insert(token.to_string(), *id);
        }
        Self { vocab_size: 50257, token_to_id_map }
    }
}

impl Default for MockTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for MockTokenizer {
    fn encode(&self, text: &str, _add_bos: bool, _add_special: bool) -> Result<Vec<u32>> {
        // Byte-level encoding for realistic mock behaviour: each UTF-8 byte → token ID 0–255.
        // This produces a proper round-trip through decode(), making test assertions meaningful.
        Ok(text.bytes().map(|b| b as u32).collect())
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        // Reconstruct UTF-8 text from byte-level token IDs (0–255); skip higher IDs (special
        // tokens or out-of-range values) so the round-trip is lossless for ASCII/UTF-8 input.
        let bytes: Vec<u8> =
            tokens.iter().filter_map(|&t| if t < 256 { Some(t as u8) } else { None }).collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        if token < 256 {
            let b = [token as u8];
            Some(String::from_utf8_lossy(&b).into_owned())
        } else {
            Some(format!("<token_{}>", token))
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id_map.get(token).copied()
    }
}
