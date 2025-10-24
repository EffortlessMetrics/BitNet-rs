# Simulation: `BasicTokenizer::token_to_piece` in `lib.rs` is a simplified implementation

The `BasicTokenizer::token_to_piece` function in `crates/bitnet-tokenizers/src/lib.rs` performs a simple placeholder. It doesn't actually convert a token ID to a piece. This is a form of simulation.

**File:** `crates/bitnet-tokenizers/src/lib.rs`

**Function:** `BasicTokenizer::token_to_piece`

**Code:**
```rust
    fn token_to_piece(&self, token: u32) -> Option<String> {
        Some(format!("<token_{}>", token))
    }
```

## Proposed Fix

The `BasicTokenizer::token_to_piece` function should be implemented to actually convert a token ID to a piece. This would involve using a proper vocabulary to map token IDs to pieces.

### Example Implementation

```rust
    fn token_to_piece(&self, token: u32) -> Option<String> {
        self.reverse_vocab.get(&token).cloned()
    }
```
