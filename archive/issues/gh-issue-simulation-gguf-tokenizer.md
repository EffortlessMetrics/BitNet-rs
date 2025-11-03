# [Tokenization] Enhance GGUF tokenizer with comprehensive token-to-piece mapping

## Problem Description

The `GgufTokenizer::token_to_piece` function uses simplified byte-level conversion that doesn't handle complex tokenization schemes, subword boundaries, or multi-byte tokens properly.

## Root Cause Analysis

```rust
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
```

## Proposed Solution

```rust
impl GgufTokenizer {
    fn token_to_piece(&self, token: u32) -> Option<String> {
        // Priority order for token resolution:
        // 1. Special tokens (BOS, EOS, UNK, etc.)
        // 2. Vocabulary lookup
        // 3. Byte fallback for byte-level tokens
        // 4. Handle unknown tokens gracefully

        // Check special tokens first
        if let Some(special_piece) = self.special_tokens.get(&token) {
            return Some(special_piece.clone());
        }

        // Primary vocabulary lookup
        if let Some(piece) = self.reverse_vocab.get(&token) {
            return Some(self.decode_piece(piece));
        }

        // Byte-level fallback for BPE tokenizers
        if token < self.byte_fallback_threshold {
            return self.handle_byte_token(token);
        }

        // Handle unknown tokens
        self.handle_unknown_token(token)
    }

    fn decode_piece(&self, piece: &str) -> String {
        // Handle special encoding (e.g., Ġ for spaces in GPT-2)
        if piece.starts_with('Ġ') {
            format!(" {}", &piece[1..])
        } else if piece.starts_with("##") {
            // BERT-style subword continuation
            piece[2..].to_string()
        } else {
            piece.to_string()
        }
    }

    fn handle_byte_token(&self, token: u32) -> Option<String> {
        if let Some(&byte) = self.id_to_byte.get(&token) {
            Some(String::from_utf8_lossy(&[byte]).to_string())
        } else if token < 256 {
            Some(String::from_utf8_lossy(&[token as u8]).to_string())
        } else {
            None
        }
    }

    fn handle_unknown_token(&self, token: u32) -> Option<String> {
        match self.unk_token_strategy {
            UnkTokenStrategy::ReturnUNK => Some(self.unk_token.clone()),
            UnkTokenStrategy::ReturnNone => None,
            UnkTokenStrategy::ReturnPlaceholder => Some(format!("<UNK_{}>", token)),
        }
    }
}

#[derive(Debug, Clone)]
enum UnkTokenStrategy {
    ReturnUNK,         // Return the UNK token string
    ReturnNone,        // Return None for unknown tokens
    ReturnPlaceholder, // Return a placeholder with token ID
}
```

## Acceptance Criteria

- [ ] Proper special token handling (BOS, EOS, UNK)
- [ ] Support for different tokenization schemes (BPE, BERT, etc.)
- [ ] Graceful unknown token handling
- [ ] Multi-byte Unicode support
- [ ] Configurable fallback strategies

## Priority: Medium
