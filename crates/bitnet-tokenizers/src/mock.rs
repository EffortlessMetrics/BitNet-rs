/// Mock tokenizer for testing
use crate::Tokenizer;
use bitnet_common::Result;

pub struct MockTokenizer {
    vocab_size: usize,
}

impl MockTokenizer {
    pub fn new() -> Self {
        Self { vocab_size: 50257 }
    }
}

impl Tokenizer for MockTokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> Result<Vec<u32>> {
        // Simple character-based encoding for testing
        Ok(text
            .chars()
            .map(|c| c as u32 % self.vocab_size as u32)
            .collect())
    }

    fn decode(&self, tokens: &[u32], _skip_special_tokens: bool) -> Result<String> {
        // Simple placeholder
        Ok(tokens.iter().map(|&t| ((t % 128) as u8) as char).collect())
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn eos_token_id(&self) -> Option<u32> {
        Some(50256)
    }

    fn pad_token_id(&self) -> Option<u32> {
        None
    }
}
