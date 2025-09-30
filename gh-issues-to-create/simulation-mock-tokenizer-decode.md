# Simulation: `MockTokenizer::decode` in `mock.rs` is a simplified implementation

The `MockTokenizer::decode` function in `crates/bitnet-tokenizers/src/mock.rs` performs a simple character-based decoding. It doesn't use a proper tokenizer. This is a form of simulation.

**File:** `crates/bitnet-tokenizers/src/mock.rs`

**Function:** `MockTokenizer::decode`

**Code:**
```rust
    fn decode(&self, tokens: &[u32]) -> Result<String> {
        // Simple placeholder
        Ok(tokens.iter().map(|&t| ((t % 128) as u8) as char).collect())
    }
```

## Proposed Fix

The `MockTokenizer::decode` function should be implemented to simulate a more realistic tokenizer. This would involve:

1.  **Using a predefined vocabulary:** Use a small, predefined vocabulary to map token IDs to characters.
2.  **Handling special tokens:** Simulate the handling of special tokens.

### Example Implementation

```rust
    fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut text = String::new();
        for &token in tokens {
            if token == 1 { text.push_str("<BOS>"); } // Simulate BOS token
            else if token == 2 { text.push_str("<SPECIAL>"); } // Simulate special token
            else { text.push(((token % 128) as u8) as char); }
        }
        Ok(text)
    }
```
