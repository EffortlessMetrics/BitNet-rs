# [Performance] Implement Actual KV Cache Compression Algorithm

## Problem Description

The `KVCache::compress_old_entries` function in `crates/bitnet-inference/src/cache.rs` currently contains only a mock implementation that sets a flag without performing actual compression. This prevents the system from reclaiming memory from aged cache entries and can lead to excessive memory usage during long inference sessions.

## Environment

- **Component**: `crates/bitnet-inference/src/cache.rs`
- **Function**: `KVCache::compress_old_entries`
- **Feature Context**: Both `cpu` and `gpu` features for memory optimization
- **Impact**: Memory usage during long-running inference sessions

## Current Implementation Analysis

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

**Issues Identified:**
1. **Mock implementation**: Only sets a flag without actual compression
2. **No memory savings**: Doesn't reduce memory footprint
3. **Missing compression algorithms**: No quantization or lossless compression
4. **No decompression logic**: Compressed entries can't be used
5. **No performance metrics**: No measurement of compression effectiveness

## Impact Assessment

**Severity**: Medium
**Affected Users**: Users running long inference sessions, especially on memory-constrained devices
**Performance Impact**:
- Unbounded memory growth during long sessions
- No memory reclamation from aged cache entries
- Potential out-of-memory conditions

## Root Cause Analysis

The current implementation is a placeholder that doesn't provide the memory optimization benefits expected from KV cache compression. Effective KV cache compression requires:

1. **Precision reduction**: Quantizing FP32 values to FP16 or INT8
2. **Entropy-based compression**: Using compression algorithms like zstd or lz4
3. **Decompression on access**: Transparent decompression when entries are accessed
4. **Memory tracking**: Monitoring compression effectiveness

## Proposed Solution

### 1. Multi-Level Compression Strategy

Implement a tiered compression approach that balances memory savings with access performance:

```rust
use half::f16;
use lz4_flex::{compress_prepend_size, decompress_size_prepended};

impl KVCache {
    pub fn compress_old_entries(&mut self, age_threshold: std::time::Duration) -> Result<()> {
        if !self.config.enable_compression {
            return Ok(());
        }

        let now = std::time::Instant::now();
        let mut compression_stats = CompressionStats::new();

        for entry in self.cache.values_mut() {
            if !entry.is_compressed() && self.should_compress_entry(entry, age_threshold, now) {
                let original_size = entry.memory_footprint();

                match self.compress_entry(entry) {
                    Ok(compressed_size) => {
                        compression_stats.add_compressed(original_size, compressed_size);
                    }
                    Err(e) => {
                        warn!("Failed to compress cache entry: {}", e);
                        compression_stats.add_failed();
                    }
                }
            }
        }

        self.update_compression_metrics(&compression_stats);
        Ok(())
    }

    fn should_compress_entry(
        &self,
        entry: &CacheEntry,
        age_threshold: std::time::Duration,
        now: std::time::Instant,
    ) -> bool {
        // Check age threshold
        if now.duration_since(entry.last_accessed) <= age_threshold {
            return false;
        }

        // Check minimum size threshold
        if entry.memory_footprint() < self.config.min_compression_size {
            return false;
        }

        // Check access frequency
        if entry.access_count > self.config.high_frequency_threshold {
            return false;
        }

        true
    }

    fn compress_entry(&self, entry: &mut CacheEntry) -> Result<usize> {
        let compression_level = self.determine_compression_level(entry);

        match compression_level {
            CompressionLevel::Precision => self.compress_precision(entry),
            CompressionLevel::Lossless => self.compress_lossless(entry),
            CompressionLevel::Hybrid => self.compress_hybrid(entry),
        }
    }

    fn determine_compression_level(&self, entry: &CacheEntry) -> CompressionLevel {
        let age = std::time::Instant::now().duration_since(entry.last_accessed);
        let size = entry.memory_footprint();

        if age > self.config.deep_compression_threshold {
            CompressionLevel::Hybrid
        } else if size > self.config.large_entry_threshold {
            CompressionLevel::Lossless
        } else {
            CompressionLevel::Precision
        }
    }

    fn compress_precision(&self, entry: &mut CacheEntry) -> Result<usize> {
        let original_size = entry.memory_footprint();

        // Convert FP32 to FP16 for both keys and values
        let compressed_keys = self.quantize_to_fp16(&entry.keys)?;
        let compressed_values = self.quantize_to_fp16(&entry.values)?;

        entry.data = CacheData::CompressedPrecision {
            keys: compressed_keys,
            values: compressed_values,
            original_dtype: entry.keys.dtype(),
        };

        let compressed_size = entry.memory_footprint();
        debug!("Precision compression: {} -> {} bytes ({:.1}% reduction)",
               original_size, compressed_size,
               (1.0 - compressed_size as f64 / original_size as f64) * 100.0);

        Ok(compressed_size)
    }

    fn compress_lossless(&self, entry: &mut CacheEntry) -> Result<usize> {
        let original_size = entry.memory_footprint();

        // Serialize tensor data
        let keys_bytes = self.serialize_tensor(&entry.keys)?;
        let values_bytes = self.serialize_tensor(&entry.values)?;

        // Apply lossless compression
        let compressed_keys = compress_prepend_size(&keys_bytes);
        let compressed_values = compress_prepend_size(&values_bytes);

        entry.data = CacheData::CompressedLossless {
            keys: compressed_keys,
            values: compressed_values,
            keys_shape: entry.keys.dims().to_vec(),
            values_shape: entry.values.dims().to_vec(),
            dtype: entry.keys.dtype(),
        };

        let compressed_size = entry.memory_footprint();
        debug!("Lossless compression: {} -> {} bytes ({:.1}% reduction)",
               original_size, compressed_size,
               (1.0 - compressed_size as f64 / original_size as f64) * 100.0);

        Ok(compressed_size)
    }

    fn compress_hybrid(&self, entry: &mut CacheEntry) -> Result<usize> {
        // First apply precision reduction, then lossless compression
        let mut temp_entry = entry.clone();
        self.compress_precision(&mut temp_entry)?;

        // Serialize the precision-reduced data
        let serialized = self.serialize_compressed_precision(&temp_entry)?;

        // Apply lossless compression to the precision-reduced data
        let compressed_data = compress_prepend_size(&serialized);

        entry.data = CacheData::CompressedHybrid {
            data: compressed_data,
            keys_shape: entry.keys.dims().to_vec(),
            values_shape: entry.values.dims().to_vec(),
            original_dtype: entry.keys.dtype(),
        };

        Ok(entry.memory_footprint())
    }

    fn quantize_to_fp16(&self, tensor: &BitNetTensor) -> Result<Vec<f16>> {
        let f32_data = tensor.to_vec1::<f32>()?;
        let f16_data: Vec<f16> = f32_data.iter()
            .map(|&x| f16::from_f32(x))
            .collect();

        Ok(f16_data)
    }

    fn serialize_tensor(&self, tensor: &BitNetTensor) -> Result<Vec<u8>> {
        // Serialize tensor to bytes using a compact format
        let data = tensor.to_vec1::<f32>()?;
        let bytes = data.iter()
            .flat_map(|&x| x.to_le_bytes())
            .collect();

        Ok(bytes)
    }

    fn serialize_compressed_precision(&self, entry: &CacheEntry) -> Result<Vec<u8>> {
        match &entry.data {
            CacheData::CompressedPrecision { keys, values, .. } => {
                let mut serialized = Vec::new();

                // Serialize FP16 data
                for &key_f16 in keys {
                    serialized.extend_from_slice(&key_f16.to_le_bytes());
                }
                for &value_f16 in values {
                    serialized.extend_from_slice(&value_f16.to_le_bytes());
                }

                Ok(serialized)
            }
            _ => bail!("Entry is not precision compressed"),
        }
    }

    pub fn decompress_entry(&mut self, entry_id: &CacheEntryId) -> Result<()> {
        let entry = self.cache.get_mut(entry_id)
            .ok_or_else(|| anyhow!("Cache entry not found: {:?}", entry_id))?;

        if !entry.is_compressed() {
            return Ok(()); // Already decompressed
        }

        let decompressed_data = match &entry.data {
            CacheData::CompressedPrecision { keys, values, original_dtype } => {
                self.decompress_precision(keys, values, *original_dtype, entry)?
            }
            CacheData::CompressedLossless { keys, values, keys_shape, values_shape, dtype } => {
                self.decompress_lossless(keys, values, keys_shape, values_shape, *dtype)?
            }
            CacheData::CompressedHybrid { data, keys_shape, values_shape, original_dtype } => {
                self.decompress_hybrid(data, keys_shape, values_shape, *original_dtype)?
            }
            _ => return Ok(()), // Not compressed
        };

        entry.data = CacheData::Uncompressed {
            keys: decompressed_data.0,
            values: decompressed_data.1,
        };

        entry.last_accessed = std::time::Instant::now();
        Ok(())
    }

    fn decompress_precision(
        &self,
        keys: &[f16],
        values: &[f16],
        original_dtype: DType,
        entry: &CacheEntry,
    ) -> Result<(BitNetTensor, BitNetTensor)> {
        // Convert FP16 back to FP32
        let keys_f32: Vec<f32> = keys.iter().map(|&x| x.to_f32()).collect();
        let values_f32: Vec<f32> = values.iter().map(|&x| x.to_f32()).collect();

        // Reconstruct tensors with original shapes
        let keys_tensor = BitNetTensor::from_vec(
            keys_f32,
            entry.keys_shape.as_slice(),
            &entry.device,
        )?;

        let values_tensor = BitNetTensor::from_vec(
            values_f32,
            entry.values_shape.as_slice(),
            &entry.device,
        )?;

        Ok((keys_tensor, values_tensor))
    }

    fn decompress_lossless(
        &self,
        compressed_keys: &[u8],
        compressed_values: &[u8],
        keys_shape: &[usize],
        values_shape: &[usize],
        dtype: DType,
    ) -> Result<(BitNetTensor, BitNetTensor)> {
        // Decompress the byte data
        let keys_bytes = decompress_size_prepended(compressed_keys)?;
        let values_bytes = decompress_size_prepended(compressed_values)?;

        // Deserialize back to f32 arrays
        let keys_data = self.deserialize_f32_data(&keys_bytes)?;
        let values_data = self.deserialize_f32_data(&values_bytes)?;

        // Reconstruct tensors
        let keys_tensor = BitNetTensor::from_vec(
            keys_data,
            keys_shape,
            &Device::Cpu, // TODO: Restore original device
        )?;

        let values_tensor = BitNetTensor::from_vec(
            values_data,
            values_shape,
            &Device::Cpu,
        )?;

        Ok((keys_tensor, values_tensor))
    }

    fn deserialize_f32_data(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        if bytes.len() % 4 != 0 {
            bail!("Invalid byte array length for f32 data: {}", bytes.len());
        }

        let data = bytes.chunks_exact(4)
            .map(|chunk| {
                let array: [u8; 4] = chunk.try_into().unwrap();
                f32::from_le_bytes(array)
            })
            .collect();

        Ok(data)
    }

    fn update_compression_metrics(&mut self, stats: &CompressionStats) {
        if stats.compressed_count > 0 {
            info!("Compressed {} entries, saved {:.1} MB ({:.1}% reduction)",
                  stats.compressed_count,
                  stats.bytes_saved as f64 / 1024.0 / 1024.0,
                  stats.compression_ratio() * 100.0);

            self.metrics.total_compressions += stats.compressed_count;
            self.metrics.total_bytes_saved += stats.bytes_saved;
        }

        if stats.failed_count > 0 {
            warn!("Failed to compress {} entries", stats.failed_count);
            self.metrics.compression_failures += stats.failed_count;
        }
    }
}

#[derive(Debug, Clone)]
enum CacheData {
    Uncompressed {
        keys: BitNetTensor,
        values: BitNetTensor,
    },
    CompressedPrecision {
        keys: Vec<f16>,
        values: Vec<f16>,
        original_dtype: DType,
    },
    CompressedLossless {
        keys: Vec<u8>,
        values: Vec<u8>,
        keys_shape: Vec<usize>,
        values_shape: Vec<usize>,
        dtype: DType,
    },
    CompressedHybrid {
        data: Vec<u8>,
        keys_shape: Vec<usize>,
        values_shape: Vec<usize>,
        original_dtype: DType,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompressionLevel {
    Precision, // FP32 -> FP16
    Lossless,  // LZ4 compression
    Hybrid,    // Precision + Lossless
}

#[derive(Debug, Default)]
struct CompressionStats {
    compressed_count: usize,
    failed_count: usize,
    original_bytes: usize,
    compressed_bytes: usize,
    bytes_saved: usize,
}

impl CompressionStats {
    fn new() -> Self {
        Self::default()
    }

    fn add_compressed(&mut self, original_size: usize, compressed_size: usize) {
        self.compressed_count += 1;
        self.original_bytes += original_size;
        self.compressed_bytes += compressed_size;
        self.bytes_saved += original_size.saturating_sub(compressed_size);
    }

    fn add_failed(&mut self) {
        self.failed_count += 1;
    }

    fn compression_ratio(&self) -> f64 {
        if self.original_bytes == 0 {
            0.0
        } else {
            1.0 - (self.compressed_bytes as f64 / self.original_bytes as f64)
        }
    }
}

impl CacheEntry {
    fn is_compressed(&self) -> bool {
        !matches!(self.data, CacheData::Uncompressed { .. })
    }

    fn memory_footprint(&self) -> usize {
        match &self.data {
            CacheData::Uncompressed { keys, values } => {
                keys.numel() * std::mem::size_of::<f32>() +
                values.numel() * std::mem::size_of::<f32>()
            }
            CacheData::CompressedPrecision { keys, values, .. } => {
                keys.len() * std::mem::size_of::<f16>() +
                values.len() * std::mem::size_of::<f16>()
            }
            CacheData::CompressedLossless { keys, values, .. } => {
                keys.len() + values.len()
            }
            CacheData::CompressedHybrid { data, .. } => {
                data.len()
            }
        }
    }
}
```

## Implementation Breakdown

### Phase 1: Basic Compression Infrastructure
- [ ] Implement `CompressionLevel` enum and strategy selection
- [ ] Add precision-based compression (FP32 -> FP16)
- [ ] Implement decompression logic
- [ ] Add basic unit tests

### Phase 2: Advanced Compression Algorithms
- [ ] Implement lossless compression using LZ4
- [ ] Add hybrid compression strategy
- [ ] Implement compression metrics collection
- [ ] Add performance benchmarking

### Phase 3: Optimization and Tuning
- [ ] Add compression threshold heuristics
- [ ] Implement background compression threads
- [ ] Add compression effectiveness monitoring
- [ ] Optimize decompression performance

### Phase 4: Integration and Testing
- [ ] Integrate with existing cache management
- [ ] Add comprehensive test coverage
- [ ] Implement memory usage monitoring
- [ ] Add configuration options

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_compression_roundtrip() {
        let mut cache = create_test_cache();
        let entry_id = add_test_entry(&mut cache);

        // Compress using precision reduction
        cache.compress_old_entries(Duration::from_secs(0)).unwrap();

        // Verify compression occurred
        let entry = cache.cache.get(&entry_id).unwrap();
        assert!(entry.is_compressed());

        // Decompress and verify data integrity
        cache.decompress_entry(&entry_id).unwrap();
        let decompressed_entry = cache.cache.get(&entry_id).unwrap();
        assert!(!decompressed_entry.is_compressed());

        // Check that values are approximately equal (within FP16 precision)
        verify_tensors_approximately_equal(&original_data, &decompressed_data, 1e-3);
    }

    #[test]
    fn test_lossless_compression_exact_roundtrip() {
        let mut cache = create_test_cache();
        cache.config.compression_level = CompressionLevel::Lossless;
        let entry_id = add_test_entry(&mut cache);

        cache.compress_old_entries(Duration::from_secs(0)).unwrap();
        cache.decompress_entry(&entry_id).unwrap();

        // Lossless compression should preserve exact values
        verify_tensors_exactly_equal(&original_data, &decompressed_data);
    }

    #[test]
    fn test_compression_memory_savings() {
        let mut cache = create_test_cache();
        let original_memory = cache.total_memory_usage();

        // Add entries and age them
        add_multiple_test_entries(&mut cache, 100);
        age_all_entries(&mut cache, Duration::from_secs(60));

        // Compress and verify memory reduction
        cache.compress_old_entries(Duration::from_secs(30)).unwrap();
        let compressed_memory = cache.total_memory_usage();

        assert!(compressed_memory < original_memory);
        let savings_ratio = 1.0 - (compressed_memory as f64 / original_memory as f64);
        assert!(savings_ratio > 0.3); // At least 30% memory savings
    }
}
```

## Performance Considerations

1. **Compression Overhead**: Balance compression time vs memory savings
2. **Decompression Speed**: Optimize for fast access to compressed entries
3. **Memory Allocation**: Minimize allocations during compression/decompression
4. **Background Processing**: Consider async compression for better responsiveness

## Risk Assessment

**Low Risk Changes:**
- Adding compression infrastructure and metrics
- Implementing precision-based compression

**Medium Risk Changes:**
- Modifying cache data structures
- Adding decompression logic

**High Risk Changes:**
- Changing cache access patterns
- Modifying existing cache entry lifecycle

**Mitigation Strategies:**
- Comprehensive round-trip testing
- Performance regression testing
- Gradual rollout with feature flags
- Fallback to uncompressed mode on errors

## Acceptance Criteria

- [ ] Achieves minimum 30% memory reduction for aged entries
- [ ] Compression/decompression operations complete within 10ms for typical entries
- [ ] Round-trip accuracy within FP16 precision for precision compression
- [ ] Exact round-trip accuracy for lossless compression
- [ ] No performance regression for cache hit operations
- [ ] Comprehensive test coverage (>95% line coverage)
- [ ] Memory usage monitoring and reporting

## Related Issues/PRs

- **Related to**: Memory optimization framework
- **Depends on**: KV cache infrastructure improvements
- **Blocks**: Long-running inference session support
- **References**: Performance monitoring enhancements

## Additional Context

This implementation is crucial for supporting long-running inference sessions without excessive memory usage. The multi-level compression strategy provides flexibility to balance memory savings with access performance based on usage patterns and system constraints.