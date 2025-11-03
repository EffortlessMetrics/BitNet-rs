# Simulation: `GgufTokenizer::token_to_piece` in `gguf_tokenizer.rs` is a simplified implementation

The `GgufTokenizer::token_to_piece` function in `crates/bitnet-tokenizers/src/gguf_tokenizer.rs` performs a simple byte-level conversion. It doesn't handle more complex token-to-piece mappings. This is a form of simulation.

**File:** `crates/bitnet-tokenizers/src/gguf_tokenizer.rs`

**Function:** `GgufTokenizer::token_to_piece`

**Code:**
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

## Proposed Fix

The `GgufTokenizer::token_to_piece` function should be implemented to handle more complex token-to-piece mappings. This would involve using the full vocabulary from the GGUF file to map token IDs to pieces.

### Example Implementation

```rust
    fn token_to_piece(&self, token: u32) -> Option<String> {
        self.reverse_vocab.get(&token).cloned()
    }
```
