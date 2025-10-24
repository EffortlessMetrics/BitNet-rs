# Stub code: `tensor_cache` in `autoregressive.rs` is a simplified cache

The `tensor_cache` in `AutoregressiveGenerator` in `crates/bitnet-inference/src/generation/autoregressive.rs` is a simplified cache that only stores the last tensor. The `try_get_cached_tensor` function always returns `None`, indicating no cache hit. This is a form of stubbing and should be replaced with a real implementation.

**File:** `crates/bitnet-inference/src/generation/autoregressive.rs`

**Field:** `AutoregressiveGenerator::tensor_cache`

**Code:**
```rust
    // Memory management
    tensor_cache: Option<BitNetTensor>,
    cache_hit_count: usize,
    cache_miss_count: usize,

// In `try_get_cached_tensor`:
    fn try_get_cached_tensor(&self, tokens: &[usize]) -> Result<Option<BitNetTensor>> {
        // Simple cache implementation - in practice would use more sophisticated caching
        if tokens.len() <= 1 {
            return Ok(None);
        }

        // For now, return None to indicate no cache hit
        // A full implementation would maintain a tensor cache
        Ok(None)
    }
```

## Proposed Fix

The `tensor_cache` should be implemented as a proper cache that stores and retrieves tensors based on the input tokens. This would involve:

1.  **Using a hash map:** Use a hash map to store the tensors, with the input tokens as the key.
2.  **Implementing a cache eviction policy:** Implement a cache eviction policy (e.g., LRU) to manage the cache size.
3.  **Updating `try_get_cached_tensor`:** Update the `try_get_cached_tensor` function to retrieve tensors from the cache.
4.  **Updating `update_tensor_cache`:** Update the `update_tensor_cache` function to store tensors in the cache.

### Example Implementation

```rust
// In `AutoregressiveGenerator` struct:
pub struct AutoregressiveGenerator {
    // ...
    tensor_cache: HashMap<Vec<usize>, BitNetTensor>,
    // ...
}

// In `AutoregressiveGenerator::new`:
            tensor_cache: HashMap::new(),

// In `try_get_cached_tensor`:
    fn try_get_cached_tensor(&self, tokens: &[usize]) -> Result<Option<BitNetTensor>> {
        Ok(self.tensor_cache.get(tokens).cloned())
    }

// In `update_tensor_cache`:
    fn update_tensor_cache(&mut self, tokens: &[usize], tensor: &BitNetTensor) -> Result<()> {
        self.tensor_cache.insert(tokens.to_vec(), tensor.clone());
        Ok(())
    }
```
