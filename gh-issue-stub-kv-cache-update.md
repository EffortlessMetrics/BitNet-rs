# [Attention] Implement sophisticated KV cache management with incremental updates

## Problem Description

The `KVCache::update` function in `crates/bitnet-inference/src/layers/attention.rs` uses a simplified implementation that replaces entire key and value tensors instead of supporting incremental updates, circular buffering, and efficient memory management required for production inference.

## Root Cause Analysis

### Current Implementation
```rust
pub fn update(&mut self, layer_idx: usize, k: BitNetTensor, v: BitNetTensor, seq_len: usize) -> Result<()> {
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

### Issues Identified
1. **No Incremental Updates**: Replaces entire tensors instead of appending new tokens
2. **Memory Inefficient**: Doesn't reuse existing cache memory
3. **No Eviction Policy**: No handling when cache exceeds maximum sequence length
4. **Missing Optimization**: No support for sliding window or circular buffer
5. **Poor Performance**: Full tensor replacement is computationally expensive

## Proposed Solution

### Production KV Cache Implementation

```rust
impl KVCache {
    /// Sophisticated cache update with incremental token support
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

        // Determine update strategy based on sequence length change
        match seq_len.cmp(&self.current_len) {
            std::cmp::Ordering::Greater => {
                // Appending new tokens
                self.append_tokens(layer_idx, k, v, seq_len)?;
            },
            std::cmp::Ordering::Equal => {
                // Updating existing position (rare, but possible)
                self.update_existing_tokens(layer_idx, k, v, seq_len)?;
            },
            std::cmp::Ordering::Less => {
                // Sequence got shorter (reset or truncation)
                self.truncate_sequence(layer_idx, k, v, seq_len)?;
            }
        }

        self.current_len = seq_len;
        Ok(())
    }

    fn append_tokens(
        &mut self,
        layer_idx: usize,
        k: BitNetTensor,
        v: BitNetTensor,
        seq_len: usize,
    ) -> Result<()> {
        let new_tokens = seq_len - self.current_len;
        debug!("Appending {} new tokens to KV cache layer {}", new_tokens, layer_idx);

        // Check if we need to handle cache overflow
        if seq_len > self.max_seq_len {
            return self.handle_cache_overflow(layer_idx, k, v, seq_len);
        }

        // Extract only the new tokens from input tensors
        let k_new = self.extract_new_tokens(&k, new_tokens)?;
        let v_new = self.extract_new_tokens(&v, new_tokens)?;

        // Concatenate with existing cache
        self.k_cache[layer_idx] = self.concatenate_tensors(&self.k_cache[layer_idx], &k_new)?;
        self.v_cache[layer_idx] = self.concatenate_tensors(&self.v_cache[layer_idx], &v_new)?;

        Ok(())
    }

    fn handle_cache_overflow(
        &mut self,
        layer_idx: usize,
        k: BitNetTensor,
        v: BitNetTensor,
        seq_len: usize,
    ) -> Result<()> {
        match self.cache_strategy {
            CacheStrategy::CircularBuffer => {
                self.apply_circular_buffer(layer_idx, k, v, seq_len)
            },
            CacheStrategy::SlidingWindow => {
                self.apply_sliding_window(layer_idx, k, v, seq_len)
            },
            CacheStrategy::Truncate => {
                self.apply_truncation(layer_idx, k, v, seq_len)
            },
            CacheStrategy::Error => {
                Err(anyhow::anyhow!(
                    "Sequence length {} exceeds maximum cache size {}",
                    seq_len, self.max_seq_len
                ))
            }
        }
    }

    fn apply_sliding_window(
        &mut self,
        layer_idx: usize,
        k: BitNetTensor,
        v: BitNetTensor,
        seq_len: usize,
    ) -> Result<()> {
        debug!("Applying sliding window cache management");

        let window_size = self.max_seq_len;
        let excess_tokens = seq_len - window_size;

        // Calculate how many tokens to keep from existing cache
        let keep_from_existing = window_size.saturating_sub(k.size(1));
        let start_idx = excess_tokens;

        // Slide existing cache and append new tokens
        if keep_from_existing > 0 {
            let k_kept = self.slice_tensor(&self.k_cache[layer_idx], start_idx, start_idx + keep_from_existing)?;
            let v_kept = self.slice_tensor(&self.v_cache[layer_idx], start_idx, start_idx + keep_from_existing)?;

            self.k_cache[layer_idx] = self.concatenate_tensors(&k_kept, &k)?;
            self.v_cache[layer_idx] = self.concatenate_tensors(&v_kept, &v)?;
        } else {
            // New tokens completely fill the window
            self.k_cache[layer_idx] = k;
            self.v_cache[layer_idx] = v;
        }

        Ok(())
    }

    fn apply_circular_buffer(
        &mut self,
        layer_idx: usize,
        k: BitNetTensor,
        v: BitNetTensor,
        seq_len: usize,
    ) -> Result<()> {
        debug!("Applying circular buffer cache management");

        let buffer_pos = self.current_len % self.max_seq_len;
        let new_tokens = seq_len - self.current_len;

        // Handle wrap-around in circular buffer
        if buffer_pos + new_tokens <= self.max_seq_len {
            // No wrap-around needed
            self.insert_at_position(layer_idx, &k, &v, buffer_pos)?;
        } else {
            // Handle wrap-around
            let before_wrap = self.max_seq_len - buffer_pos;
            let after_wrap = new_tokens - before_wrap;

            let (k_before, k_after) = self.split_tensor(&k, before_wrap)?;
            let (v_before, v_after) = self.split_tensor(&v, before_wrap)?;

            self.insert_at_position(layer_idx, &k_before, &v_before, buffer_pos)?;
            self.insert_at_position(layer_idx, &k_after, &v_after, 0)?;
        }

        // Update circular buffer metadata
        self.buffer_positions[layer_idx] = (buffer_pos + new_tokens) % self.max_seq_len;

        Ok(())
    }

    /// Compress old cache entries to save memory while preserving recent context
    pub fn compress_old_entries(&mut self, layer_idx: usize, compression_ratio: f32) -> Result<()> {
        if !self.compression_enabled {
            return Ok(());
        }

        let cache_len = self.k_cache[layer_idx].size(1);
        let compression_boundary = (cache_len as f32 * (1.0 - compression_ratio)) as usize;

        if compression_boundary > 0 {
            debug!("Compressing {} old tokens in layer {} cache", compression_boundary, layer_idx);

            // Compress older tokens (reduce precision, downsample, etc.)
            let k_compressed = self.compress_tensor_segment(
                &self.k_cache[layer_idx],
                0,
                compression_boundary,
                CompressionMethod::Quantization
            )?;

            let v_compressed = self.compress_tensor_segment(
                &self.v_cache[layer_idx],
                0,
                compression_boundary,
                CompressionMethod::Quantization
            )?;

            // Keep recent tokens uncompressed
            let k_recent = self.slice_tensor(&self.k_cache[layer_idx], compression_boundary, cache_len)?;
            let v_recent = self.slice_tensor(&self.v_cache[layer_idx], compression_boundary, cache_len)?;

            // Reconstruct cache with compressed old + uncompressed recent
            self.k_cache[layer_idx] = self.concatenate_tensors(&k_compressed, &k_recent)?;
            self.v_cache[layer_idx] = self.concatenate_tensors(&v_compressed, &v_recent)?;
        }

        Ok(())
    }

    /// Efficient cache retrieval for attention computation
    pub fn get_cache_for_attention(
        &self,
        layer_idx: usize,
        query_len: usize,
    ) -> Result<(BitNetTensor, BitNetTensor)> {
        if layer_idx >= self.k_cache.len() {
            return Err(anyhow::anyhow!("Layer index {} out of bounds", layer_idx));
        }

        let effective_cache_len = match self.cache_strategy {
            CacheStrategy::CircularBuffer => {
                // Handle circular buffer retrieval
                self.get_circular_cache(layer_idx, query_len)?
            },
            _ => {
                // Standard sequential cache
                (self.k_cache[layer_idx].clone(), self.v_cache[layer_idx].clone())
            }
        };

        Ok(effective_cache_len)
    }

    /// Get cache statistics for monitoring and optimization
    pub fn get_cache_stats(&self) -> CacheStatistics {
        CacheStatistics {
            current_length: self.current_len,
            max_length: self.max_seq_len,
            utilization: self.current_len as f32 / self.max_seq_len as f32,
            memory_usage_mb: self.estimate_memory_usage(),
            hits: self.cache_hits,
            misses: self.cache_misses,
            compressions_applied: self.compression_count,
            evictions: self.eviction_count,
        }
    }
}

#[derive(Debug, Clone)]
pub enum CacheStrategy {
    CircularBuffer,  // Overwrite oldest entries
    SlidingWindow,   // Keep most recent N tokens
    Truncate,        // Error when exceeding capacity
    Error,           // Fail hard on overflow
}

#[derive(Debug, Clone)]
pub struct CacheStatistics {
    pub current_length: usize,
    pub max_length: usize,
    pub utilization: f32,
    pub memory_usage_mb: f64,
    pub hits: u64,
    pub misses: u64,
    pub compressions_applied: u64,
    pub evictions: u64,
}
```

## Testing Strategy

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incremental_cache_updates() {
        let mut cache = KVCache::new(&test_config(), 100).unwrap();

        // Initial cache population
        let k1 = create_test_tensor([1, 5, 64]);
        let v1 = create_test_tensor([1, 5, 64]);
        cache.update(0, k1, v1, 5).unwrap();

        // Incremental update
        let k2 = create_test_tensor([1, 3, 64]);
        let v2 = create_test_tensor([1, 3, 64]);
        cache.update(0, k2, v2, 8).unwrap();

        assert_eq!(cache.current_len, 8);
        assert_eq!(cache.k_cache[0].size(1), 8);
    }

    #[test]
    fn test_sliding_window_overflow() {
        let mut cache = KVCache::with_strategy(CacheStrategy::SlidingWindow, 10);

        // Fill beyond capacity
        let k = create_test_tensor([1, 15, 64]);
        let v = create_test_tensor([1, 15, 64]);
        cache.update(0, k, v, 15).unwrap();

        // Should maintain window size
        assert_eq!(cache.k_cache[0].size(1), 10);
    }
}
```

## Acceptance Criteria

- [ ] Incremental token appending without full tensor replacement
- [ ] Circular buffer and sliding window cache strategies
- [ ] Memory-efficient cache compression for old entries
- [ ] Performance monitoring and statistics
- [ ] Comprehensive error handling for edge cases

## Priority: High

Essential for production inference performance and memory efficiency.
