# Stub code: `TransformerModel::embed_transposed` and `lm_head_transposed` in `transformer.rs` are determined by flags

The `embed_transposed` and `lm_head_transposed` fields in `crates/bitnet-models/src/transformer.rs` are determined by flags. This might be a form of stubbing if the flags are not properly set.

**File:** `crates/bitnet-models/src/transformer.rs`

**Fields:**
* `TransformerModel::embed_transposed`
* `TransformerModel::lm_head_transposed`

**Code:**
```rust
pub struct TransformerModel {
    // ...
    pub embed_transposed: bool, // True if embeddings are stored as [hidden, vocab]
    // ...
    pub lm_head_transposed: bool,       // True if lm_head is stored as [hidden, vocab]
    // ...
}

impl TransformerModel {
    pub fn new(config: BitNetConfig, vb: VarBuilder) -> Result<Self> {
        // ...

        // Read transpose flag for embeddings (1-element tensor)
        let embed_transposed = match vb.get((1,), "embed_tokens.transposed") {
            Ok(t) => {
                let vals = t.to_vec1::<f32>()?;
                vals.first().copied().unwrap_or(0.0) > 0.5
            }
            Err(_) => false, // If flag doesn't exist, assume not transposed
        };

        // ...

        // Read transpose flag for lm_head
        let transposed = match vb.get((1,), "lm_head.transposed") {
            Ok(t) => {
                let vals = t.to_vec1::<f32>()?;
                vals.first().copied().unwrap_or(0.0) > 0.5
            }
            Err(_) => false, // If flag doesn't exist, assume not transposed
        };

        // ...
    }
}
```

## Proposed Fix

The `embed_transposed` and `lm_head_transposed` fields should be determined by the actual shape of the embedding and LM head weights, rather than by flags. This would involve inspecting the shape of the loaded tensors and setting the flags accordingly.

### Example Implementation

```rust
impl TransformerModel {
    pub fn new(config: BitNetConfig, vb: VarBuilder) -> Result<Self> {
        // ...

        let embed_tokens = candle_nn::embedding(vocab_size, hidden_size, vb.pp("embed_tokens"))?;
        let embed_weight = embed_tokens.embeddings();
        let embed_transposed = embed_weight.dims() == &[hidden_size, vocab_size];

        // ...

        let (lm_head, lm_head_weight, lm_head_transposed) = match linear_with_optional_bias(
            hidden_size,
            vocab_size,
            vb.pp("lm_head"),
        ) {
            Ok(layer) => {
                let weight = vb
                    .get((vocab_size, hidden_size), "lm_head.weight")
                    .or_else(|_| vb.get((hidden_size, vocab_size), "lm_head.weight"))
                    .ok();

                let transposed = weight.map_or(false, |w| w.dims() == &[hidden_size, vocab_size]);

                (Some(layer), weight, transposed)
            }
            // ...
        };

        // ...
    }
}
```
