# Stub code: `MockTokenizer::token_to_piece` in `mock.rs` is a placeholder

The `MockTokenizer::token_to_piece` function in `crates/bitnet-tokenizers/src/mock.rs` always returns `"<token>"`. It doesn't actually convert a token ID to a piece. This is a form of stubbing.

**File:** `crates/bitnet-tokenizers/src/mock.rs`

**Function:** `MockTokenizer::token_to_piece`

**Code:**
```rust
    fn token_to_piece(&self, _token: u32) -> Option<String> {
        Some("<token>".to_string())
    }
```

## Proposed Fix

The `MockTokenizer::token_to_piece` function should be implemented to simulate a more realistic tokenizer. This would involve using a small, predefined vocabulary to map token IDs to pieces.

### Example Implementation

```rust
    fn token_to_piece(&self, token: u32) -> Option<String> {
        match token {
            1 => Some("<BOS>".to_string()),
            2 => Some("<SPECIAL>".to_string()),
            _ => Some(format!("char_{}", token)),
        }
    }
```
