# Simulation: `TransformerModel::forward_full` in `transformer.rs` uses step-by-step processing

The `TransformerModel::forward_full` function in `crates/bitnet-models/src/transformer.rs` processes tokens step-by-step with a KV cache. This might be inefficient for full sequence processing. This is a form of simulation.

**File:** `crates/bitnet-models/src/transformer.rs`

**Function:** `TransformerModel::forward_full`

**Code:**
```rust
    pub fn forward_full(&self, token_ids: &Tensor) -> Result<Tensor> {
        // Token ids expected shape: [B,T]
        let (batch_size, seq_len) = token_ids.dims2()?;

        // Embed the entire sequence once.
        let flat_ids = token_ids.flatten_all()?;
        let ids_vec: Vec<u32> = flat_ids.to_vec1()?;
        let hidden = self.embed(&ids_vec)?;
        let hidden_size = self.config.model.hidden_size;
        let hidden = hidden.reshape(&[batch_size, seq_len, hidden_size])?;

        // Create per-layer KV cache so that rotary/absolute positional
        // encodings use the proper positions during iterative decoding.
        let mut kv_cache = KVCache::new(&self.config, batch_size, &self.device)?;

        // Collect logits for each position.
        let mut logits_steps = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            // Select the current token's embedding: [B,1,H]
            let step_hidden = hidden.narrow(1, t, 1)?;

            // Run through all layers using the incremental path which applies
            // positional encoding per layer and causal masking internally.
            let step_hidden = self.forward(&step_hidden, Some(&mut kv_cache))?;

            // Project to vocabulary logits for this step.
            let step_logits = self.logits(&step_hidden)?;
            logits_steps.push(step_logits);
        }

        // Concatenate logits from all steps: [B,T,V]
        Ok(Tensor::cat(&logits_steps, 1)?)
    }
```

## Proposed Fix

The `TransformerModel::forward_full` function should be implemented to process the full sequence in a single forward pass, rather than step-by-step. This would involve:

1.  **Processing the full sequence:** Process the entire sequence in a single forward pass.
2.  **Applying causal mask:** Apply a causal mask to prevent attending to future tokens.
3.  **Using optimized tensor operations:** Use optimized tensor operations for better performance.

### Example Implementation

```rust
    pub fn forward_full(&self, token_ids: &Tensor) -> Result<Tensor> {
        // Token ids expected shape: [B,T]
        let (batch_size, seq_len) = token_ids.dims2()?;

        // Embed the entire sequence once.
        let flat_ids = token_ids.flatten_all()?;
        let ids_vec: Vec<u32> = flat_ids.to_vec1()?;
        let mut hidden = self.embed(&ids_vec)?;
        let hidden_size = self.config.model.hidden_size;
        hidden = hidden.reshape(&[batch_size, seq_len, hidden_size])?;

        // Create per-layer KV cache
        let mut kv_cache = KVCache::new(&self.config, batch_size, &self.device)?;

        // Run through all layers
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_cache = kv_cache.layer_mut(i);
            hidden = layer.forward(&hidden, layer_cache)?;
        }

        // Project to vocabulary logits
        let logits = self.logits(&hidden)?;

        Ok(logits)
    }
```