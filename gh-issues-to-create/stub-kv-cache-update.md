# Stub code: `KVCache::update` in `attention.rs` is a simplified implementation

The `KVCache::update` function in `crates/bitnet-inference/src/layers/attention.rs` has a comment "For now, simple implementation - in production this would be more sophisticated". It just replaces the existing key and value tensors. It doesn't handle more sophisticated cache updates like appending new tokens or managing a circular buffer. This is a form of stubbing.

**File:** `crates/bitnet-inference/src/layers/attention.rs`

**Function:** `KVCache::update`

**Code:**
```rust
    pub fn update(
        &mut self,
        layer_idx: usize,
        k: BitNetTensor,
        v: BitNetTensor,
        seq_len: usize,
    ) -> Result<()> {
        if layer_idx >= self.k_cache.len() {
            return Err(anyhow::anyhow!("Layer index {} out of bounds", layer_idx));
        }

        // For now, simple implementation - in production this would be more sophisticated
        self.k_cache[layer_idx] = k;
        self.v_cache[layer_idx] = v;
        self.current_len = seq_len;

        Ok(())
    }
```

## Proposed Fix

The `KVCache::update` function should be implemented to handle more sophisticated cache updates. This would involve:

1.  **Appending new tokens:** If the `seq_len` is greater than the current length, new tokens should be appended to the cache.
2.  **Managing a circular buffer:** If the `max_seq_len` is reached, older entries should be evicted using a circular buffer or other eviction policy.

### Example Implementation

```rust
    pub fn update(
        &mut self,
        layer_idx: usize,
        k: BitNetTensor,
        v: BitNetTensor,
        seq_len: usize,
    ) -> Result<()> {
        if layer_idx >= self.k_cache.len() {
            return Err(anyhow::anyhow!("Layer index {} out of bounds", layer_idx));
        }

        // Append new tokens if seq_len is greater than current_len
        if seq_len > self.current_len {
            // Append k and v to existing cache tensors
            // This would involve slicing the new k and v tensors and concatenating them
        } else {
            // Replace existing k and v tensors
            self.k_cache[layer_idx] = k;
            self.v_cache[layer_idx] = v;
        }

        self.current_len = seq_len;

        Ok(())
    }
```
