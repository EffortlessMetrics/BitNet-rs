# Simulation: `BasicTokenizer::encode` in `lib.rs` is a simplified implementation

The `BasicTokenizer::encode` function in `crates/bitnet-tokenizers/src/lib.rs` performs a simple whitespace-based tokenization. It doesn't handle more complex tokenization schemes like BPE or SentencePiece. This is a form of simulation.

**File:** `crates/bitnet-tokenizers/src/lib.rs`

**Function:** `BasicTokenizer::encode`

**Code:**
```rust
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let words: Vec<&str> = text.split_whitespace().collect();
        let mut tokens: Vec<u32> = Vec::new();

        if add_bos && let Some(bos) = self.bos_token_id {
            tokens.push(bos);
        }

        for (i, _) in words.iter().enumerate() {
            let id = i as u32;
            if id >= self.vocab_size as u32 {
                return Err(BitNetError::Model(ModelError::LoadingFailed {
                    reason: "token id exceeds vocab size".to_string(),
                }));
            }
            tokens.push(id);
        }

        if add_special {
            if let Some(eos_id) = self.eos_token_id {
                tokens.push(eos_id);
            }
            if let Some(pad_id) = self.pad_token_id {
                tokens.push(pad_id);
            }
        }

        Ok(tokens)
    }
```

## Proposed Fix

The `BasicTokenizer::encode` function should be implemented to handle more complex tokenization schemes. This would involve:

1.  **Implementing BPE or SentencePiece:** Implement a BPE or SentencePiece tokenization algorithm.
2.  **Using a proper vocabulary:** Use a proper vocabulary to map words to token IDs.
3.  **Handling special tokens:** Handle special tokens (e.g., BOS, EOS, UNK) according to the tokenizer's configuration.

### Example Implementation

```rust
    fn encode(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();

        if add_bos && let Some(bos) = self.bos_token_id {
            tokens.push(bos);
        }

        // Example: Use a BPE tokenizer
        let bpe_tokens = self.bpe_tokenizer.encode(text)?;
        tokens.extend(bpe_tokens);

        if add_special {
            if let Some(eos_id) = self.eos_token_id {
                tokens.push(eos_id);
            }
            if let Some(pad_id) = self.pad_token_id {
                tokens.push(pad_id);
            }
        }

        Ok(tokens)
    }
```
