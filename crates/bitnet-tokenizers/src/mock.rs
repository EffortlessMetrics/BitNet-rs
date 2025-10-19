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
        // Simple character-based encoding for testing
        Ok(text.chars().map(|c| c as u32 % self.vocab_size as u32).collect())
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        // Simple placeholder
        Ok(tokens.iter().map(|&t| ((t % 128) as u8) as char).collect())
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn token_to_piece(&self, _token: u32) -> Option<String> {
        Some("<token>".to_string())
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id_map.get(token).copied()
    }
}
