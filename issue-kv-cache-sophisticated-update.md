# [Inference] Implement Sophisticated KV Cache Update Mechanism

## Problem Description

The `KVCache::update` method in `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/layers/attention.rs` uses a naive implementation that replaces entire cache tensors instead of efficiently appending new tokens, missing critical optimizations for autoregressive generation performance.

## Current Implementation
```rust
pub fn update(&mut self, layer_idx: usize, k: BitNetTensor, v: BitNetTensor, seq_len: usize) -> Result<()> {
    // For now, simple implementation - in production this would be more sophisticated
    self.k_cache[layer_idx] = k;
    self.v_cache[layer_idx] = v;
    self.current_len = seq_len;
    Ok(())
}
```

## Proposed Solution
Implement efficient incremental cache updates:

```rust
pub fn update(&mut self, layer_idx: usize, k: BitNetTensor, v: BitNetTensor, seq_len: usize) -> Result<()> {
    if seq_len > self.current_len {
        // Incremental update: append new tokens
        let new_tokens = seq_len - self.current_len;
        self.append_to_cache(layer_idx, k, v, new_tokens)?;
    } else if seq_len < self.current_len {
        // Truncate cache for shorter sequences
        self.truncate_cache(layer_idx, seq_len)?;
    }
    // For same length, cache remains unchanged
    self.current_len = seq_len;
    Ok(())
}

fn append_to_cache(&mut self, layer_idx: usize, k: BitNetTensor, v: BitNetTensor, new_tokens: usize) -> Result<()> {
    // Efficient tensor concatenation along sequence dimension
    let existing_k = &self.k_cache[layer_idx];
    let existing_v = &self.v_cache[layer_idx];

    self.k_cache[layer_idx] = BitNetTensor::cat(&[existing_k, &k], 1)?; // Concat on seq dim
    self.v_cache[layer_idx] = BitNetTensor::cat(&[existing_v, &v], 1)?;

    Ok(())
}
```

## Acceptance Criteria
- [ ] Efficient incremental cache updates for autoregressive generation
- [ ] Memory-efficient tensor concatenation operations
- [ ] Support for cache truncation and circular buffer management
- [ ] Performance improvement of 2-3x for sequential token generation
- [ ] Thread-safe cache operations for concurrent inference
