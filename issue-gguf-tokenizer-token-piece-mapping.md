# [Tokenization] GGUF Tokenizer Token-to-Piece Mapping Enhancement

## Problem Description

The `GgufTokenizer::token_to_piece` function uses a simplified byte-level conversion that falls back to basic character mapping, rather than utilizing the full vocabulary mapping from GGUF files.

## Environment

- **File**: `crates/bitnet-tokenizers/src/gguf_tokenizer.rs`
- **Function**: `GgufTokenizer::token_to_piece`
- **Component**: GGUF Tokenizer Token Mapping

## Root Cause Analysis

### **Current Implementation:**
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

### **Issues:**
1. **Redundant Mappings**: Multiple fallback paths with overlapping logic
2. **Incomplete Vocabulary Usage**: Not fully utilizing GGUF vocabulary data
3. **Inefficient Lookups**: Multiple HashMap lookups for single operation

## Proposed Solution

Simplify and optimize using the complete vocabulary mapping:

```rust
fn token_to_piece(&self, token: u32) -> Option<String> {
    self.reverse_vocab.get(&token).cloned()
}
```

## Implementation Plan

- Ensure `reverse_vocab` contains complete token-to-piece mapping from GGUF
- Remove redundant `id_to_byte` mapping logic
- Add comprehensive vocabulary validation during tokenizer construction
- Implement proper error handling for missing tokens

## Success Metrics

- [ ] Complete vocabulary coverage from GGUF files
- [ ] Simplified and more efficient token-to-piece conversion
- [ ] Improved tokenizer initialization with full vocabulary validation

## Labels

- `tokenization`
- `gguf-parsing`
- `vocabulary-mapping`