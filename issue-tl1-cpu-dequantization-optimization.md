# [Performance] Optimize TL1 CPU Dequantization with Lookup Tables and SIMD

## Problem Description

The `CPUQuantizer::dequantize_tl1` function in `crates/bitnet-quantization/src/device_aware_quantizer.rs` uses a simplified implementation that performs basic arithmetic operations instead of efficient lookup table-based dequantization. This results in suboptimal performance for TL1 (Table Lookup 1) quantization, which is designed to use precomputed lookup tables for fast dequantization.

## Environment

- **Component**: `crates/bitnet-quantization/src/device_aware_quantizer.rs`
- **Function**: `CPUQuantizer::dequantize_tl1`
- **Quantization Type**: TL1 (Table Lookup 1) - 4-bit quantization with lookup tables
- **Performance Critical**: Yes - affects inference latency

## Current Implementation Analysis

```rust
pub fn dequantize_tl1(&self, tensor: &QuantizedTensor) -> Result<Vec<f32>> {
    // ... validation ...

    for block_idx in 0..num_blocks {
        let scale = tensor.scales[block_idx];
        let start = block_idx * block_size;
        let end = (start + block_size).min(tensor.data.len());

        for i in start..end {
            let quantized = tensor.data[i] as f32;
            let normalized = (quantized / 7.5) - 1.0;  // Simplified arithmetic
            let dequantized_val = normalized * scale;
            dequantized.push(dequantized_val);
        }
    }
}
```

**Issues Identified:**
1. **Missing lookup table**: Uses arithmetic instead of precomputed lookup
2. **Inefficient bit unpacking**: Treats bytes as individual values instead of packed 4-bit values
3. **No SIMD optimization**: Sequential processing without vectorization
4. **Suboptimal memory access**: Non-cache-friendly access patterns

## Impact Assessment

**Severity**: Medium-High
**Affected Users**: Users utilizing TL1 quantization for inference
**Performance Impact**:
- Significantly slower TL1 dequantization compared to optimal implementation
- Missing performance benefits of lookup table approach
- Suboptimal CPU utilization

## Root Cause Analysis

TL1 quantization is specifically designed to use lookup tables for fast dequantization, but the current implementation:
1. Uses simplified arithmetic instead of lookup tables
2. Doesn't properly handle 4-bit packed data format
3. Lacks SIMD optimization opportunities
4. Doesn't utilize the performance characteristics that make TL1 attractive

## Proposed Solution

### 1. Lookup Table-Based TL1 Implementation

```rust
use rayon::prelude::*;
use std::arch::x86_64::*;

pub struct TL1LookupTable {
    values: [f32; 16], // 4-bit = 16 possible values
}

impl TL1LookupTable {
    pub fn new() -> Self {
        // Initialize TL1 lookup table with optimal 4-bit quantization values
        let mut values = [0.0f32; 16];

        // TL1 uses symmetric quantization with 16 levels
        for i in 0..16 {
            // Map 4-bit values to symmetric range [-1, 1]
            values[i] = (i as f32 - 7.5) / 7.5;
        }

        Self { values }
    }

    #[inline]
    pub fn dequantize(&self, quantized_4bit: u8) -> f32 {
        debug_assert!(quantized_4bit < 16);
        unsafe { *self.values.get_unchecked(quantized_4bit as usize) }
    }

    pub fn dequantize_simd_avx2(&self, quantized_values: &[u8]) -> Vec<f32> {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            self.dequantize_avx2_impl(quantized_values)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            // Fallback for non-x86_64
            quantized_values.iter()
                .map(|&q| self.dequantize(q))
                .collect()
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn dequantize_avx2_impl(&self, quantized_values: &[u8]) -> Vec<f32> {
        let mut result = Vec::with_capacity(quantized_values.len());

        // Load lookup table into AVX2 registers for efficient lookup
        let lookup_lo = _mm256_loadu_ps(&self.values[0]);
        let lookup_hi = _mm256_loadu_ps(&self.values[8]);

        // Process 8 values at a time
        for chunk in quantized_values.chunks_exact(8) {
            let indices = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);

            // Split into low and high 4-bit indices
            let mask = _mm256_set1_epi32(0x0F);
            let indices_lo = _mm256_and_si256(indices, mask);
            let indices_hi = _mm256_and_si256(_mm256_srli_epi32(indices, 4), mask);

            // Perform vectorized lookup
            let values_lo = _mm256_permutevar8x32_ps(lookup_lo, indices_lo);
            let values_hi = _mm256_permutevar8x32_ps(lookup_hi, indices_hi);

            // Store results
            let mut temp = [0.0f32; 8];
            _mm256_storeu_ps(temp.as_mut_ptr(), values_lo);
            result.extend_from_slice(&temp);
        }

        // Handle remaining values
        for &q in quantized_values.chunks_exact(8).remainder() {
            result.push(self.dequantize(q));
        }

        result
    }
}

impl CPUQuantizer {
    pub fn dequantize_tl1(&self, tensor: &QuantizedTensor) -> Result<Vec<f32>> {
        debug!("Performing optimized TL1 dequantization on CPU");

        if tensor.qtype != QuantizationType::TL1 {
            return Err(bitnet_common::BitNetError::Quantization(
                QuantizationError::UnsupportedType { qtype: tensor.qtype.to_string() },
            ));
        }

        // Validate tensor format
        self.validate_tl1_tensor_format(tensor)?;

        // Unpack 4-bit values from packed bytes
        let unpacked_values = self.unpack_4bit_values(&tensor.data, tensor.numel())?;

        // Perform optimized dequantization
        match self.cpu_features.simd_level {
            SimdLevel::Avx2 => self.dequantize_tl1_avx2(tensor, &unpacked_values),
            SimdLevel::Sse2 => self.dequantize_tl1_sse2(tensor, &unpacked_values),
            SimdLevel::None => self.dequantize_tl1_scalar(tensor, &unpacked_values),
        }
    }

    fn validate_tl1_tensor_format(&self, tensor: &QuantizedTensor) -> Result<()> {
        // Verify that tensor data length matches expected packed format
        let expected_packed_length = (tensor.numel() + 1) / 2; // 2 values per byte
        if tensor.data.len() != expected_packed_length {
            return Err(QuantizationError::InvalidFormat {
                reason: format!(
                    "TL1 tensor data length {} doesn't match expected packed length {}",
                    tensor.data.len(), expected_packed_length
                ),
            }.into());
        }

        // Verify block size is reasonable for TL1
        if tensor.block_size == 0 || tensor.block_size > 1024 {
            return Err(QuantizationError::InvalidFormat {
                reason: format!("Invalid TL1 block size: {}", tensor.block_size),
            }.into());
        }

        Ok(())
    }

    fn unpack_4bit_values(&self, packed_data: &[u8], num_elements: usize) -> Result<Vec<u8>> {
        let mut unpacked = Vec::with_capacity(num_elements);

        for (i, &packed_byte) in packed_data.iter().enumerate() {
            // Extract low 4 bits
            let low_nibble = packed_byte & 0x0F;
            unpacked.push(low_nibble);

            // Extract high 4 bits if we haven't reached the end
            if unpacked.len() < num_elements {
                let high_nibble = (packed_byte >> 4) & 0x0F;
                unpacked.push(high_nibble);
            }
        }

        // Ensure we have exactly the expected number of elements
        unpacked.truncate(num_elements);
        Ok(unpacked)
    }

    fn dequantize_tl1_avx2(
        &self,
        tensor: &QuantizedTensor,
        unpacked_values: &[u8],
    ) -> Result<Vec<f32>> {
        let mut result = Vec::with_capacity(tensor.numel());
        let block_size = tensor.block_size;

        // Process blocks in parallel
        let blocks: Vec<_> = (0..tensor.scales.len())
            .into_par_iter()
            .map(|block_idx| {
                let scale = tensor.scales[block_idx];
                let start = block_idx * block_size;
                let end = (start + block_size).min(unpacked_values.len());

                if start >= unpacked_values.len() {
                    return Vec::new();
                }

                let block_values = &unpacked_values[start..end];

                // Use SIMD lookup table for this block
                let lookup_values = self.lookup_table.dequantize_simd_avx2(block_values);

                // Apply scaling
                lookup_values.into_iter()
                    .map(|v| v * scale)
                    .collect::<Vec<f32>>()
            })
            .collect();

        // Combine results
        for block in blocks {
            result.extend(block);
        }

        Ok(result)
    }

    fn dequantize_tl1_scalar(
        &self,
        tensor: &QuantizedTensor,
        unpacked_values: &[u8],
    ) -> Result<Vec<f32>> {
        let mut result = Vec::with_capacity(tensor.numel());
        let block_size = tensor.block_size;

        for (block_idx, &scale) in tensor.scales.iter().enumerate() {
            let start = block_idx * block_size;
            let end = (start + block_size).min(unpacked_values.len());

            if start >= unpacked_values.len() {
                break;
            }

            for &quantized_4bit in &unpacked_values[start..end] {
                let normalized_value = self.lookup_table.dequantize(quantized_4bit);
                result.push(normalized_value * scale);
            }
        }

        Ok(result)
    }

    fn get_or_create_lookup_table(&mut self) -> &TL1LookupTable {
        if self.tl1_lookup_table.is_none() {
            self.tl1_lookup_table = Some(TL1LookupTable::new());
        }
        self.tl1_lookup_table.as_ref().unwrap()
    }
}

#[derive(Debug, Clone)]
struct CPUQuantizer {
    cpu_features: CpuFeatures,
    tl1_lookup_table: Option<TL1LookupTable>,
    // ... other fields
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SimdLevel {
    None,
    Sse2,
    Avx2,
    Avx512,
}
```

### 2. Advanced SIMD Optimizations

```rust
impl TL1LookupTable {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn dequantize_block_avx2(
        &self,
        block_values: &[u8],
        scale: f32,
        output: &mut [f32],
    ) {
        let scale_vec = _mm256_set1_ps(scale);
        let lookup_table = &self.values;

        // Process 8 4-bit values at a time
        for (chunk_idx, values_chunk) in block_values.chunks_exact(8).enumerate() {
            let output_start = chunk_idx * 8;
            if output_start + 8 <= output.len() {
                let output_slice = &mut output[output_start..output_start + 8];

                // Load 8 4-bit indices
                let indices = _mm256_loadu_si256(values_chunk.as_ptr() as *const __m256i);

                // Perform vectorized lookup
                let mut lookup_results = [0.0f32; 8];
                for i in 0..8 {
                    let idx = values_chunk[i] as usize;
                    lookup_results[i] = lookup_table[idx];
                }

                // Load lookup results and apply scaling
                let lookup_vec = _mm256_loadu_ps(lookup_results.as_ptr());
                let scaled_vec = _mm256_mul_ps(lookup_vec, scale_vec);

                // Store results
                _mm256_storeu_ps(output_slice.as_mut_ptr(), scaled_vec);
            }
        }

        // Handle remaining values with scalar code
        let remaining_start = (block_values.len() / 8) * 8;
        for (i, &quantized_val) in block_values[remaining_start..].iter().enumerate() {
            let output_idx = remaining_start + i;
            if output_idx < output.len() {
                output[output_idx] = lookup_table[quantized_val as usize] * scale;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn dequantize_block_neon(
        &self,
        block_values: &[u8],
        scale: f32,
        output: &mut [f32],
    ) {
        use std::arch::aarch64::*;

        let scale_vec = vdupq_n_f32(scale);
        let lookup_table = &self.values;

        // Process 4 values at a time with NEON
        for (chunk_idx, values_chunk) in block_values.chunks_exact(4).enumerate() {
            let output_start = chunk_idx * 4;
            if output_start + 4 <= output.len() {
                let output_slice = &mut output[output_start..output_start + 4];

                // Perform lookup for 4 values
                let mut lookup_results = [0.0f32; 4];
                for i in 0..4 {
                    lookup_results[i] = lookup_table[values_chunk[i] as usize];
                }

                // Load and scale with NEON
                let lookup_vec = vld1q_f32(lookup_results.as_ptr());
                let scaled_vec = vmulq_f32(lookup_vec, scale_vec);

                // Store results
                vst1q_f32(output_slice.as_mut_ptr(), scaled_vec);
            }
        }

        // Handle remaining values
        let remaining_start = (block_values.len() / 4) * 4;
        for (i, &quantized_val) in block_values[remaining_start..].iter().enumerate() {
            let output_idx = remaining_start + i;
            if output_idx < output.len() {
                output[output_idx] = lookup_table[quantized_val as usize] * scale;
            }
        }
    }
}
```

### 3. Memory Access Optimization

```rust
impl CPUQuantizer {
    fn dequantize_tl1_cache_optimized(
        &self,
        tensor: &QuantizedTensor,
        unpacked_values: &[u8],
    ) -> Result<Vec<f32>> {
        let mut result = vec![0.0f32; tensor.numel()];
        let block_size = tensor.block_size;

        // Process multiple blocks in cache-friendly chunks
        let cache_friendly_blocks = 8; // Process 8 blocks at a time

        for block_chunk_start in (0..tensor.scales.len()).step_by(cache_friendly_blocks) {
            let block_chunk_end = (block_chunk_start + cache_friendly_blocks)
                .min(tensor.scales.len());

            // Process this chunk of blocks
            for block_idx in block_chunk_start..block_chunk_end {
                let scale = tensor.scales[block_idx];
                let start = block_idx * block_size;
                let end = (start + block_size).min(unpacked_values.len());

                if start >= unpacked_values.len() {
                    break;
                }

                let block_values = &unpacked_values[start..end];
                let output_slice = &mut result[start..end];

                // Use optimized SIMD processing for this block
                self.process_block_simd_optimized(block_values, scale, output_slice);
            }
        }

        Ok(result)
    }

    fn process_block_simd_optimized(
        &self,
        block_values: &[u8],
        scale: f32,
        output: &mut [f32],
    ) {
        match self.cpu_features.simd_level {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 => unsafe {
                self.lookup_table.dequantize_block_avx2(block_values, scale, output);
            },
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon => unsafe {
                self.lookup_table.dequantize_block_neon(block_values, scale, output);
            },
            _ => {
                // Scalar fallback
                for (i, &quantized_val) in block_values.iter().enumerate() {
                    if i < output.len() {
                        output[i] = self.lookup_table.dequantize(quantized_val) * scale;
                    }
                }
            }
        }
    }
}
```

## Implementation Breakdown

### Phase 1: Lookup Table Infrastructure
- [ ] Implement TL1LookupTable with precomputed values
- [ ] Add 4-bit unpacking functionality
- [ ] Create basic lookup-based dequantization
- [ ] Add unit tests for lookup table accuracy

### Phase 2: SIMD Optimizations
- [ ] Implement AVX2 vectorized lookup
- [ ] Add NEON implementation for ARM64
- [ ] Implement SSE2 fallback
- [ ] Add SIMD capability detection

### Phase 3: Memory Access Optimization
- [ ] Implement cache-friendly block processing
- [ ] Add parallel block processing with Rayon
- [ ] Optimize memory layout for SIMD operations
- [ ] Add prefetching hints

### Phase 4: Integration and Testing
- [ ] Integrate with existing quantization pipeline
- [ ] Add comprehensive accuracy tests
- [ ] Implement performance benchmarking
- [ ] Add cross-platform validation

## Testing Strategy

### Performance Tests
```rust
#[cfg(test)]
mod performance_tests {
    #[test]
    fn benchmark_tl1_dequantization() {
        let tensor = create_large_tl1_tensor();
        let quantizer = CPUQuantizer::new();

        let old_time = time_execution(|| {
            quantizer.dequantize_tl1_old(&tensor).unwrap()
        });

        let new_time = time_execution(|| {
            quantizer.dequantize_tl1(&tensor).unwrap()
        });

        let speedup = old_time.as_secs_f64() / new_time.as_secs_f64();
        assert!(speedup >= 3.0, "Expected at least 3x speedup, got {}", speedup);
    }
}
```

### Accuracy Tests
```rust
#[cfg(test)]
mod accuracy_tests {
    #[test]
    fn test_lookup_table_accuracy() {
        let lookup_table = TL1LookupTable::new();

        for i in 0..16 {
            let expected = (i as f32 - 7.5) / 7.5;
            let actual = lookup_table.dequantize(i);
            assert!((expected - actual).abs() < 1e-6);
        }
    }

    #[test]
    fn test_simd_vs_scalar_consistency() {
        let test_values = create_test_4bit_values();
        let lookup_table = TL1LookupTable::new();

        let scalar_results: Vec<f32> = test_values.iter()
            .map(|&v| lookup_table.dequantize(v))
            .collect();

        let simd_results = lookup_table.dequantize_simd_avx2(&test_values);

        for (scalar, simd) in scalar_results.iter().zip(simd_results.iter()) {
            assert!((scalar - simd).abs() < 1e-6);
        }
    }
}
```

## Performance Targets

- **Lookup table**: 5-10x speedup over arithmetic calculation
- **SIMD optimization**: Additional 2-4x speedup on compatible hardware
- **Memory efficiency**: <10% memory overhead for lookup tables
- **Cache performance**: >90% cache hit rate for sequential access

## Risk Assessment

**Low Risk**: Adding lookup table infrastructure
**Medium Risk**: Implementing SIMD optimizations
**High Risk**: Changing core dequantization algorithm

**Mitigation**: Extensive testing, gradual rollout, fallback mechanisms

## Acceptance Criteria

- [ ] Uses proper lookup table approach for TL1 dequantization
- [ ] Achieves minimum 3x performance improvement
- [ ] Maintains bit-identical accuracy with reference implementation
- [ ] SIMD optimizations work on x86_64 and ARM64
- [ ] Proper 4-bit unpacking from packed byte format
- [ ] Comprehensive test coverage including edge cases

## Related Issues/PRs

- **Related to**: Quantization performance optimization
- **Depends on**: CPU feature detection infrastructure
- **Blocks**: High-performance TL1 inference
- **References**: SIMD optimization framework

## Additional Context

TL1 quantization is specifically designed for lookup table-based dequantization to achieve optimal performance. This implementation should leverage the algorithmic advantages of TL1 while providing platform-specific optimizations for maximum throughput.