# Simulation: `BasicTokenizer::decode` in `lib.rs` is a simplified implementation

The `BasicTokenizer::decode` function in `crates/bitnet-tokenizers/src/lib.rs` performs a simple placeholder. It doesn't actually decode tokens back to text. This is a form of simulation.

**File:** `crates/bitnet-tokenizers/src/lib.rs`

**Function:** `BasicTokenizer::decode`

**Code:**
```rust
    fn decode(&self, tokens: &[u32]) -> Result<String> {
        if tokens.is_empty() {
            return Ok(String::new());
        }

        // Simple placeholder implementation - in real tokenizer this would map back to text
        Ok(format!("Generated text from {} tokens", tokens.len()))
    }
```

## Proposed Fix

The `BasicTokenizer::decode` function should be implemented to actually decode tokens back to text. This would involve:

1.  **Using a proper vocabulary:** Use a proper vocabulary to map token IDs to words.
2.  **Handling special tokens:** Handle special tokens (e.g., BOS, EOS, UNK) according to the tokenizer's configuration.

### Example Implementation

```rust
    fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut text = String::new();
        for &token in tokens {
            if let Some(word) = self.reverse_vocab.get(&token) {
                text.push_str(word);
                text.push(' '); // Add space between words
            } else if token == self.bos_token_id.unwrap_or(0) { text.push_str("<BOS>"); }
            else if token == self.eos_token_id.unwrap_or(0) { text.push_str("<EOS>"); }
            else if token == self.pad_token_id.unwrap_or(0) { text.push_str("<PAD>"); }
            else { text.push_str("<UNK>"); }
        }
        Ok(text.trim().to_string())
    }
```
