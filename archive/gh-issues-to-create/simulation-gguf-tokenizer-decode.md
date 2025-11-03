# Simulation: `GgufTokenizer::decode` in `gguf_tokenizer.rs` is a simplified implementation

The `GgufTokenizer::decode` function in `crates/bitnet-tokenizers/src/gguf_tokenizer.rs` performs a simple byte-level decoding. It doesn't handle more complex detokenization schemes. This is a form of simulation.

**File:** `crates/bitnet-tokenizers/src/gguf_tokenizer.rs`

**Function:** `GgufTokenizer::decode`

**Code:**
```rust
    fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut text = String::new();
        let mut byte_buf: Vec<u8> = Vec::new();

        for &token in tokens {
            if let Some(&byte) = self.id_to_byte.get(&token) {
                byte_buf.push(byte);
            } else if token < 256 {
                // Direct byte value
                byte_buf.push(token as u8);
            } else if let Some(token_str) = self.reverse_vocab.get(&token) {
                if !byte_buf.is_empty() {
                    text.push_str(&String::from_utf8_lossy(&byte_buf));
                    byte_buf.clear();
                }
                text.push_str(token_str);
            }
        }

        if !byte_buf.is_empty() {
            text.push_str(&String::from_utf8_lossy(&byte_buf));
        }

        Ok(text)
    }
```

## Proposed Fix

The `GgufTokenizer::decode` function should be implemented to handle more complex detokenization schemes. This would involve:

1.  **Implementing BPE or SentencePiece detokenization:** Implement a BPE or SentencePiece detokenization algorithm.
2.  **Using a proper vocabulary:** Use the full vocabulary from the GGUF file to map token IDs to pieces.
3.  **Handling special tokens:** Handle special tokens (e.g., BOS, EOS, UNK) according to the tokenizer's configuration.

### Example Implementation

```rust
    fn decode(&self, tokens: &[u32]) -> Result<String> {
        // Example: Use a BPE detokenizer
        let text = self.bpe_tokenizer.decode(tokens)?;
        Ok(text)
    }
```
