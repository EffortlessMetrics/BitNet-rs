# Stub code: `TransformerModel::logits` in `transformer.rs` has a `LOGGED` once for tied weights

The `logits` function in `crates/bitnet-models/src/transformer.rs` has a `LOGGED` once for tied weights. This suggests that the tied weights logic is not fully optimized. This is a form of stubbing.

**File:** `crates/bitnet-models/src/transformer.rs`

**Function:** `TransformerModel::logits`

**Code:**
```rust
    pub fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let vocab_size = self.config.model.vocab_size;

        match hidden.rank() {
            2 => {
                // [B, H] - last token only
                let (b, _h) = (hidden.dims()[0], hidden.dims()[1]);

                let logits = if let Some(ref lm_head) = self.lm_head {
                    // Use dedicated LM head if available
                    let logits = lm_head.forward(hidden)?; // [B, V]
                    logits.reshape(&[b, vocab_size])?
                } else {
                    // Tied weights: use embedding matrix
                    static LOGGED: std::sync::Once = std::sync::Once::new();
                    LOGGED.call_once(|| {
                        tracing::info!("LM head tied to input embeddings");
                    });

                    if self.embed_transposed {
                        // Embeddings are [hidden, vocab]
                        let embeddings = self.embed_tokens.embeddings();
                        hidden.matmul(embeddings)? // [B, V]
                    } else if let Some(ref cached_weight) = self.embed_tied_weight {
                        // Use pre-transposed cached weight [H, V] - avoids per-step transpose!
                        hidden.matmul(cached_weight)? // [B, V]
                    } else {
                        // Fallback: transpose on-demand (should be rare after optimization)
                        let embeddings = self.embed_tokens.embeddings();
                        let w = embeddings.transpose(0, 1)?; // [H, V]
                        hidden.matmul(&w)? // [B, V]
                    }
                };

                // ...

                Ok(logits)
            }
            // ...
        }
    }
```

## Proposed Fix

The `TransformerModel::logits` function should be implemented to use the tied weights logic without relying on a `LOGGED` once. This would involve ensuring that the tied weights are properly handled during model loading and that the `lm_head` is always available.

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
                    let embeddings = self.embed_tokens.embeddings();
                    let w = embeddings.transpose(0, 1)?; // [H, V]
                    hidden.matmul(&w)?.reshape(&[b, vocab_size])?
                };

                Ok(logits)
            }
            // ...
        }
    }
```
