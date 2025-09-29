# [IMPLEMENTATION] Implement sophisticated KV cache update with token appending and circular buffer management

## Problem Description
The `KVCache::update` function in `crates/bitnet-inference/src/layers/attention.rs` uses a simple replacement strategy instead of sophisticated cache management needed for production inference.

## Environment
- **File**: `crates/bitnet-inference/src/layers/attention.rs`
- **Function**: `KVCache::update`
- **Current State**: Simple tensor replacement

## Root Cause Analysis
```rust
pub fn update(&mut self, layer_idx: usize, k: BitNetTensor, v: BitNetTensor, seq_len: usize) -> Result<()> {
    // For now, simple implementation - in production this would be more sophisticated
    self.k_cache[layer_idx] = k;
    self.v_cache[layer_idx] = v;
    self.current_len = seq_len;
    Ok(())
}
```

**Issues:**
1. Always replaces entire cache instead of appending
2. No support for incremental token generation
3. No circular buffer for long sequences
4. Inefficient memory usage for streaming inference

## Proposed Solution
```rust
impl KVCache {
    pub fn update(&mut self, layer_idx: usize, k: BitNetTensor, v: BitNetTensor, seq_len: usize) -> Result<()> {
        if layer_idx >= self.k_cache.len() {
            return Err(anyhow::anyhow!("Layer index {} out of bounds", layer_idx));
        }

        match self.update_strategy {
            UpdateStrategy::Append => self.append_tokens(layer_idx, k, v, seq_len),
            UpdateStrategy::CircularBuffer => self.circular_buffer_update(layer_idx, k, v, seq_len),
            UpdateStrategy::Replace => self.replace_cache(layer_idx, k, v, seq_len),
        }
    }

    fn append_tokens(&mut self, layer_idx: usize, k: BitNetTensor, v: BitNetTensor, seq_len: usize) -> Result<()> {
        if seq_len > self.current_len {
            // Extract new tokens from input tensors
            let new_k_slice = k.slice_range(self.current_len..seq_len)?;
            let new_v_slice = v.slice_range(self.current_len..seq_len)?;

            // Concatenate with existing cache
            self.k_cache[layer_idx] = self.k_cache[layer_idx].concat(&new_k_slice, 1)?;
            self.v_cache[layer_idx] = self.v_cache[layer_idx].concat(&new_v_slice, 1)?;

            // Check if we exceed max sequence length
            if self.k_cache[layer_idx].dim(1)? > self.max_seq_len {
                self.apply_eviction_policy(layer_idx)?;
            }
        }

        self.current_len = seq_len;
        Ok(())
    }

    fn circular_buffer_update(&mut self, layer_idx: usize, k: BitNetTensor, v: BitNetTensor, seq_len: usize) -> Result<()> {
        let cache_seq_len = self.k_cache[layer_idx].dim(1)?;

        if seq_len > cache_seq_len {
            // Need to shift and append
            let shift_amount = seq_len - cache_seq_len;
            let new_tokens = seq_len - self.current_len;

            // Shift existing cache left
            if shift_amount < cache_seq_len {
                let keep_start = shift_amount;
                let keep_k = self.k_cache[layer_idx].slice_range(keep_start..cache_seq_len)?;
                let keep_v = self.v_cache[layer_idx].slice_range(keep_start..cache_seq_len)?;

                // Append new tokens
                let new_k = k.slice_range((seq_len - new_tokens)..seq_len)?;
                let new_v = v.slice_range((seq_len - new_tokens)..seq_len)?;

                self.k_cache[layer_idx] = keep_k.concat(&new_k, 1)?;
                self.v_cache[layer_idx] = keep_v.concat(&new_v, 1)?;
            } else {
                // Replace entire cache with latest tokens
                let start_idx = seq_len.saturating_sub(cache_seq_len);
                self.k_cache[layer_idx] = k.slice_range(start_idx..seq_len)?;
                self.v_cache[layer_idx] = v.slice_range(start_idx..seq_len)?;
            }
        }

        self.current_len = seq_len.min(self.max_seq_len);
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum UpdateStrategy {
    Replace,        // Simple replacement (current behavior)
    Append,         // Append new tokens, evict old when full
    CircularBuffer, // Maintain fixed-size sliding window
}
```

## Implementation Plan
### Phase 1: Core Update Strategies (2 days)
- [ ] Implement token appending with concat operations
- [ ] Add circular buffer management for fixed-size windows
- [ ] Create configurable update strategies

### Phase 2: Memory Management (1 day)
- [ ] Implement eviction policies (FIFO, LRU)
- [ ] Add memory pressure handling
- [ ] Optimize tensor slicing and concatenation

## Acceptance Criteria
- [ ] Support for incremental token appending
- [ ] Circular buffer for long sequence handling
- [ ] Configurable cache management strategies
- [ ] Memory-efficient operations for streaming inference

**Labels**: `implementation`, `attention`, `memory-management`, `P2-medium`
**Effort**: 3 days