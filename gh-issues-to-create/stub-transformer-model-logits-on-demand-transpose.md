# Stub code: `TransformerModel::logits` in `transformer.rs` has a fallback for on-demand transpose

The `logits` function in `crates/bitnet-models/src/transformer.rs` has a fallback for on-demand transpose. This suggests that the transpose logic is not fully optimized. This is a form of stubbing.

**File:** `crates/bitnet-models/src/transformer.rs`

**Function:** `TransformerModel::logits`

**Code:**
```rust
    pub fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let vocab_size = self.config.model.vocab_size;

        match hidden.rank() {
            2 => {
                // ...
                } else if let Some(ref cached_weight) = self.embed_tied_weight {
                    // Use pre-transposed cached weight [H, V] - avoids per-step transpose!
                    hidden.matmul(cached_weight)? // [B, V]
                } else {
                    // Fallback: transpose on-demand (should be rare after optimization)
                    let embeddings = self.embed_tokens.embeddings();
                    let w = embeddings.transpose(0, 1)?; // [H, V]
                    hidden.matmul(&w)? // [B, V]
                };

                // ...
            }
            // ...
        }
    }
```

## Proposed Fix

The `TransformerModel::logits` function should be implemented to use the pre-transposed cached weight when available, and to always pre-transpose the embedding weight during model loading if tied weights are used. This will ensure that the transpose logic is fully optimized.

### Example Implementation

```rust
    pub fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let vocab_size = self.config.model.vocab_size;

        match hidden.rank() {
            2 => {
                let (b, _h) = (hidden.dims()[0], hidden.dims()[1]);

                let logits = if let Some(ref lm_head) = self.lm_head {
                    lm_head.forward(hidden)?.reshape(&[b, vocab_size])?
                } else if let Some(ref cached_weight) = self.embed_tied_weight {
                    hidden.matmul(cached_weight)?.reshape(&[b, vocab_size])?
                } else {
                    // This case should ideally not be reached if tied weights are handled at load time
                    return Err(BitNetError::Validation("Tied weights not properly pre-transposed".into()));
                };

                Ok(logits)
            }
            // ...
        }
    }
```