# [OPTIMIZATION] Enhance GGUF tokenizer token-to-piece mapping with comprehensive vocabulary handling

## Problem Description
The `GgufTokenizer::token_to_piece` function in `crates/bitnet-tokenizers/src/gguf_tokenizer.rs` uses fallback logic and simplified byte-level conversion instead of comprehensive vocabulary mapping.

## Environment
- **File**: `crates/bitnet-tokenizers/src/gguf_tokenizer.rs`
- **Function**: `GgufTokenizer::token_to_piece`
- **Current State**: Multi-fallback approach with potential inaccuracies

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

**Issues:**
1. Multiple fallback strategies create unpredictable behavior
2. Byte-level fallback may not match original tokenizer
3. Assumption that tokens < 256 are bytes may be incorrect
4. Inefficient string cloning for common operations

## Proposed Solution
```rust
impl GgufTokenizer {
    fn token_to_piece(&self, token: u32) -> Option<String> {
        // Primary lookup in vocabulary
        if let Some(piece) = self.reverse_vocab.get(&token) {
            return Some(piece.clone());
        }

        // Handle special tokens
        if let Some(special_piece) = self.handle_special_token(token) {
            return Some(special_piece);
        }

        // Handle byte-level tokens for byte-pair encoders
        if self.is_byte_level_token(token) {
            return self.decode_byte_level_token(token);
        }

        // Unknown token
        None
    }

    fn handle_special_token(&self, token: u32) -> Option<String> {
        match token {
            t if t == self.bos_token_id => Some("<s>".to_string()),
            t if t == self.eos_token_id => Some("</s>".to_string()),
            t if t == self.unk_token_id => Some("<unk>".to_string()),
            t if t == self.pad_token_id => Some("<pad>".to_string()),
            _ => None,
        }
    }

    fn is_byte_level_token(&self, token: u32) -> bool {
        // Check if this tokenizer uses byte-level encoding
        self.tokenizer_type == TokenizerType::ByteLevel && token < 256
    }

    fn decode_byte_level_token(&self, token: u32) -> Option<String> {
        if token < 256 {
            // Handle byte-level decoding with proper UTF-8 handling
            let byte = token as u8;

            // Check if it's a valid UTF-8 start byte or ASCII
            if byte < 128 {
                Some(String::from_utf8_lossy(&[byte]).to_string())
            } else {
                // Handle multi-byte UTF-8 sequences properly
                self.decode_utf8_sequence(byte)
            }
        } else {
            None
        }
    }

    fn decode_utf8_sequence(&self, start_byte: u8) -> Option<String> {
        // This would need to be implemented based on the specific
        // byte-level encoding scheme used by the tokenizer
        // For now, return the byte as a lossy string
        Some(String::from_utf8_lossy(&[start_byte]).to_string())
    }

    // Optimized version using Arc<str> to avoid cloning
    fn token_to_piece_optimized(&self, token: u32) -> Option<Arc<str>> {
        // Use Arc<str> in reverse_vocab for zero-copy returns
        self.reverse_vocab_arc.get(&token).cloned()
    }
}

#[derive(Debug, Clone, PartialEq)]
enum TokenizerType {
    WordPiece,
    ByteLevel,
    SentencePiece,
    CharLevel,
}
```

## Implementation Plan
### Phase 1: Vocabulary Enhancement (1 day)
- [ ] Implement comprehensive vocabulary loading from GGUF
- [ ] Add special token handling with proper IDs
- [ ] Create tokenizer type detection

### Phase 2: Optimization (1 day)
- [ ] Implement Arc<str> for zero-copy piece access
- [ ] Add proper UTF-8 sequence handling
- [ ] Optimize for common token access patterns

## Acceptance Criteria
- [ ] Accurate token-to-piece mapping using full vocabulary
- [ ] Proper special token handling
- [ ] Optimized memory usage with Arc<str>
- [ ] Comprehensive test coverage with real GGUF files

**Labels**: `optimization`, `tokenization`, `memory-efficiency`, `P2-medium`
**Effort**: 2 days