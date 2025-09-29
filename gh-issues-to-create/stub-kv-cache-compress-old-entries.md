# Stub code: `KVCache::compress_old_entries` in `cache.rs` is a mock implementation

The `KVCache::compress_old_entries` function in `crates/bitnet-inference/src/cache.rs` has a comment "Simple compression: reduce precision (this is a mock implementation)". It doesn't perform actual compression. This is a form of stubbing and should be replaced with a real implementation.

**File:** `crates/bitnet-inference/src/cache.rs`

**Function:** `KVCache::compress_old_entries`

**Code:**
```rust
    pub fn compress_old_entries(&mut self, age_threshold: std::time::Duration) -> Result<()> {
        if !self.config.enable_compression {
            return Ok(());
        }

        let now = std::time::Instant::now();
        let mut compressed_count = 0;

        for entry in self.cache.values_mut() {
            if !entry.compressed && now.duration_since(entry.last_accessed) > age_threshold {
                // Simple compression: reduce precision (this is a mock implementation)
                // In practice, you'd use a proper compression algorithm
                entry.compressed = true;
                compressed_count += 1;
            }
        }

        if compressed_count > 0 {
            debug!("Compressed {} cache entries", compressed_count);
        }

        Ok(())
    }
```

## Proposed Fix

The `KVCache::compress_old_entries` function should be implemented to perform actual compression of old cache entries. This would involve using a proper compression algorithm (e.g., quantization to a lower precision, or a lossless compression algorithm) to reduce the memory footprint of old cache entries.

### Example Implementation

```rust
    pub fn compress_old_entries(&mut self, age_threshold: std::time::Duration) -> Result<()> {
        if !self.config.enable_compression {
            return Ok(());
        }

        let now = std::time::Instant::now();
        let mut compressed_count = 0;

        for entry in self.cache.values_mut() {
            if !entry.compressed && now.duration_since(entry.last_accessed) > age_threshold {
                // Example: Quantize to a lower precision (e.g., FP16)
                entry.key = quantize_to_fp16(&entry.key);
                entry.value = quantize_to_fp16(&entry.value);
                entry.compressed = true;
                compressed_count += 1;
            }
        }

        if compressed_count > 0 {
            debug!("Compressed {} cache entries", compressed_count);
        }

        Ok(())
    }

    fn quantize_to_fp16(data: &[f32]) -> Vec<f32> {
        // ... implementation to quantize f32 to f16 ...
        data.to_vec() // Placeholder
    }
```
