# Stub code: `TransformerModel::embed_tied_weight` in `transformer.rs` is a placeholder

The `embed_tied_weight` field in `crates/bitnet-models/src/transformer.rs` is a placeholder for a cached transposed embedding weight. It's not clear if it's always used. This is a form of stubbing.

**File:** `crates/bitnet-models/src/transformer.rs`

**Field:** `TransformerModel::embed_tied_weight`

**Code:**
```rust
pub struct TransformerModel {
    // ...
    pub embed_tied_weight: Option<Tensor>, // Cached transposed embedding weight for tied models [H, V]
    // ...
}

impl TransformerModel {
    pub fn new(config: BitNetConfig, vb: VarBuilder) -> Result<Self> {
        // ...

        // PATCH 2: Optimize tied weights by pre-transposing embeddings once at load
        let (embed_transposed, embed_tied_weight) = if embed_transposed {
            // Already transposed from flag
            (true, None)
        } else if lm_head.is_none() {
            // No dedicated lm_head, we'll use tied weights - pre-transpose for efficiency
            let embed_weight = embed_tokens.embeddings();
            if embed_weight.dims() == [vocab_size, hidden_size] {
                // Embeddings are [V, H], transpose to [H, V] to avoid per-step transpose
                tracing::info!(
                    "Pre-transposing tied embeddings [V,H] -> [H,V] to avoid per-step transpose"
                );
                let transposed_weight = embed_weight.transpose(0, 1)?; // [H, V]
                (false, Some(transposed_weight)) // Keep original flag, but cache transposed weight
            } else {
                // Embeddings already in [H, V] format or unexpected shape
                tracing::warn!(
                    "Embeddings have unexpected shape: {:?}, expected [vocab={}, hidden={}]",
                    embed_weight.dims(),
                    vocab_size,
                    hidden_size
                );
                (embed_transposed, None)
            }
        } else {
            // Dedicated lm_head exists, no need to optimize embeddings
            (embed_transposed, None)
        };

        // ...
    }
}
```

## Proposed Fix

The `embed_tied_weight` field should be properly used to cache the transposed embedding weight for tied models. This would involve ensuring that the cached weight is always used when available, and that it is correctly updated when the model is loaded or reconfigured.

### Example Implementation

```rust
impl TransformerModel {
    pub fn new(config: BitNetConfig, vb: VarBuilder) -> Result<Self> {
        // ...

        let (embed_transposed, embed_tied_weight) = if embed_transposed {
            (true, None)
        } else if lm_head.is_none() {
            let embed_weight = embed_tokens.embeddings();
            if embed_weight.dims() == [vocab_size, hidden_size] {
                let transposed_weight = embed_weight.transpose(0, 1)?; // [H, V]
                (false, Some(transposed_weight)) // Cache transposed weight
            } else {
                (embed_transposed, None)
            }
        } else {
            (embed_transposed, None)
        };

        // ...
    }
}
```