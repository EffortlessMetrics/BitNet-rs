# [PERF] Vectorized Lookup Table SIMD Optimization for TL1/TL2 Quantization

## Problem Description

The vectorized lookup table implementation in BitNet.rs lacks SIMD optimization, preventing the full utilization of modern CPU vector instructions for table lookup operations. This architectural gap significantly limits quantization performance for TL1 and TL2 methods that rely heavily on efficient table lookups.

## Environment

- **Component**: `bitnet-quantization` crate
- **Affected Methods**: TL1, TL2 quantization table lookups
- **Target Architectures**: x86_64 (AVX2/AVX-512), ARM64 (NEON)
- **Performance Impact**: 4-8x potential speedup with proper SIMD implementation

## Root Cause Analysis

1. **Scalar Table Lookups**: Current implementation uses sequential table access
2. **Missing SIMD Intrinsics**: No utilization of vector gather/scatter operations
3. **Cache Inefficiency**: Suboptimal memory access patterns for SIMD
4. **Architecture Gaps**: No ARM64 NEON optimization paths

## Proposed Solution

### SIMD-Optimized Vectorized Lookup Tables

```rust
use std::arch::x86_64::*;
use std::arch::aarch64::*;

/// SIMD-optimized vectorized lookup table for quantization
pub struct VectorizedLookupTable {
    /// Lookup table data aligned for SIMD access
    table: Vec<f32>,
    /// Table size (power of 2)
    size: usize,
    /// SIMD capability
    simd_capability: SimdCapability,
}

impl VectorizedLookupTable {
    pub fn new(size: usize, values: Vec<f32>, simd_capability: SimdCapability) -> Self {
        assert!(size.is_power_of_two() && size >= 64);

        // Ensure proper SIMD alignment
        let alignment = simd_capability.cache_alignment();
        let mut aligned_table = Vec::with_capacity(size + alignment / std::mem::size_of::<f32>());

        // Align table to SIMD boundary
        while (aligned_table.as_ptr() as usize) % alignment != 0 {
            aligned_table.push(0.0);
        }

        aligned_table.extend_from_slice(&values);
        aligned_table.resize(size, 0.0);

        Self {
            table: aligned_table,
            size,
            simd_capability,
        }
    }

    /// SIMD-optimized batch lookup operation
    pub fn lookup_batch(&self, indices: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
        match self.simd_capability {
            SimdCapability::Avx512 => self.lookup_batch_avx512(indices, output),
            SimdCapability::Avx2 => self.lookup_batch_avx2(indices, output),
            SimdCapability::Neon => self.lookup_batch_neon(indices, output),
            _ => self.lookup_batch_scalar(indices, output),
        }
    }

    /// AVX-512 implementation with gather operations
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn lookup_batch_avx512(&self, indices: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
        let table_ptr = self.table.as_ptr();
        let mask = (self.size - 1) as u32;

        let mut i = 0;

        // Process 16 elements at a time with AVX-512
        while i + 16 <= indices.len() {
            // Load 16 u8 indices and convert to u32
            let indices_u8 = _mm_loadu_si128(indices.as_ptr().add(i) as *const __m128i);
            let indices_u16 = _mm256_cvtepu8_epi16(indices_u8);
            let indices_lo = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(indices_u16, 0));
            let indices_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(indices_u16, 1));

            // Convert to full 512-bit register
            let indices_512 = _mm512_inserti64x4(_mm512_castsi256_si512(indices_lo), indices_hi, 1);

            // Apply mask to ensure valid table indices
            let masked_indices = _mm512_and_epi32(indices_512, _mm512_set1_epi32(mask as i32));

            // Gather from lookup table
            let gathered = _mm512_i32gather_ps(masked_indices, table_ptr, 4);

            // Store results
            _mm512_storeu_ps(output.as_mut_ptr().add(i), gathered);

            i += 16;
        }

        // Handle remaining elements
        self.lookup_batch_scalar(&indices[i..], &mut output[i..])?;

        Ok(())
    }

    /// AVX2 implementation with optimized gather
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn lookup_batch_avx2(&self, indices: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
        let table_ptr = self.table.as_ptr();
        let mask = (self.size - 1) as u32;

        let mut i = 0;

        // Process 8 elements at a time with AVX2
        while i + 8 <= indices.len() {
            // Load 8 u8 indices
            let indices_64 = std::ptr::read_unaligned(indices.as_ptr().add(i) as *const u64);
            let indices_u8 = _mm_set_epi64x(0, indices_64 as i64);

            // Convert u8 to u32
            let indices_u32 = _mm256_cvtepu8_epi32(indices_u8);

            // Apply mask
            let masked_indices = _mm256_and_si256(indices_u32, _mm256_set1_epi32(mask as i32));

            // Gather from table (AVX2 gather is slower but still better than scalar)
            let gathered = _mm256_i32gather_ps(table_ptr, masked_indices, 4);

            // Store results
            _mm256_storeu_ps(output.as_mut_ptr().add(i), gathered);

            i += 8;
        }

        // Handle remaining elements with manual lookup (faster than gather for small counts)
        for j in i..indices.len() {
            let idx = (indices[j] as usize) & (self.size - 1);
            output[j] = self.table[idx];
        }

        Ok(())
    }

    /// ARM64 NEON implementation
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn lookup_batch_neon(&self, indices: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
        let table_ptr = self.table.as_ptr();
        let mask = (self.size - 1) as u8;

        let mut i = 0;

        // Process 4 elements at a time with NEON
        while i + 4 <= indices.len() {
            // Load 4 u8 indices
            let indices_u8 = vld1_u8(indices.as_ptr().add(i));

            // Apply mask
            let mask_vec = vdup_n_u8(mask);
            let masked_indices = vand_u8(indices_u8, mask_vec);

            // Manual lookup (NEON doesn't have efficient gather)
            let mut results = [0.0f32; 4];
            for j in 0..4 {
                let idx = vget_lane_u8(masked_indices, j) as usize;
                results[j] = *table_ptr.add(idx);
            }

            // Store results
            let result_vec = vld1q_f32(results.as_ptr());
            vst1q_f32(output.as_mut_ptr().add(i), result_vec);

            i += 4;
        }

        // Handle remaining elements
        for j in i..indices.len() {
            let idx = (indices[j] as usize) & (self.size - 1);
            output[j] = self.table[idx];
        }

        Ok(())
    }

    /// Scalar fallback implementation
    fn lookup_batch_scalar(&self, indices: &[u8], output: &mut [f32]) -> Result<(), QuantizationError> {
        let mask = self.size - 1;

        for (i, &idx) in indices.iter().enumerate() {
            let table_idx = (idx as usize) & mask;
            output[i] = self.table[table_idx];
        }

        Ok(())
    }

    /// Optimized lookup for TL2 quantization patterns
    pub fn lookup_tl2_block(&self, input: &[f32], output: &mut [u8], scale: f32, offset: f32) -> Result<(), QuantizationError> {
        match self.simd_capability {
            SimdCapability::Avx512 => self.lookup_tl2_block_avx512(input, output, scale, offset),
            SimdCapability::Avx2 => self.lookup_tl2_block_avx2(input, output, scale, offset),
            _ => self.lookup_tl2_block_scalar(input, output, scale, offset),
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn lookup_tl2_block_avx512(&self, input: &[f32], output: &mut [u8], scale: f32, offset: f32) -> Result<(), QuantizationError> {
        let scale_vec = _mm512_set1_ps(scale);
        let offset_vec = _mm512_set1_ps(offset);
        let size_mask = _mm512_set1_epi32((self.size - 1) as i32);

        let mut i = 0;
        while i + 16 <= input.len() {
            // Load input values
            let values = _mm512_loadu_ps(input.as_ptr().add(i));

            // Apply scale and offset
            let scaled = _mm512_fmadd_ps(values, scale_vec, offset_vec);

            // Convert to indices (clamped to table size)
            let indices = _mm512_cvtps_epi32(scaled);
            let clamped_indices = _mm512_and_epi32(indices, size_mask);

            // Use table lookup to get quantized values
            let quantized_f32 = _mm512_i32gather_ps(clamped_indices, self.table.as_ptr(), 4);

            // Convert to u8 and store
            let quantized_i32 = _mm512_cvtps_epi32(quantized_f32);
            let quantized_i16 = _mm512_packs_epi32(quantized_i32, quantized_i32);
            let quantized_u8 = _mm512_packus_epi16(quantized_i16, quantized_i16);

            // Extract and store (only lower 128 bits contain our 16 values as bytes)
            let result_128 = _mm512_extracti64x2_epi64(quantized_u8, 0);
            _mm_storeu_si128(output.as_mut_ptr().add(i) as *mut __m128i, result_128);

            i += 16;
        }

        // Handle remaining elements
        self.lookup_tl2_block_scalar(&input[i..], &mut output[i..], scale, offset)?;

        Ok(())
    }

    fn lookup_tl2_block_scalar(&self, input: &[f32], output: &mut [u8], scale: f32, offset: f32) -> Result<(), QuantizationError> {
        for (i, &val) in input.iter().enumerate() {
            let scaled = val * scale + offset;
            let idx = (scaled as usize).min(self.size - 1);
            output[i] = self.table[idx] as u8;
        }
        Ok(())
    }
}

/// Factory for creating optimized lookup tables
pub struct LookupTableFactory {
    simd_capability: SimdCapability,
}

impl LookupTableFactory {
    pub fn new() -> Self {
        Self {
            simd_capability: get_simd_capability(),
        }
    }

    /// Create optimized TL1 lookup table
    pub fn create_tl1_table(&self, config: &TL1Config) -> VectorizedLookupTable {
        let values = self.generate_tl1_values(config);
        VectorizedLookupTable::new(
            config.lookup_table_size,
            values,
            self.simd_capability,
        )
    }

    /// Create optimized TL2 lookup table
    pub fn create_tl2_table(&self, config: &TL2Config) -> VectorizedLookupTable {
        let values = self.generate_tl2_values(config);
        VectorizedLookupTable::new(
            config.lookup_table_size,
            values,
            self.simd_capability,
        )
    }

    fn generate_tl1_values(&self, config: &TL1Config) -> Vec<f32> {
        // Generate TL1 quantization lookup values
        (0..config.lookup_table_size)
            .map(|i| {
                // TL1 quantization mapping
                let normalized = i as f32 / (config.lookup_table_size - 1) as f32;
                (normalized * 2.0 - 1.0) // Map to [-1, 1] range
            })
            .collect()
    }

    fn generate_tl2_values(&self, config: &TL2Config) -> Vec<f32> {
        // Generate TL2 quantization lookup values
        (0..config.lookup_table_size)
            .map(|i| {
                // TL2 quantization mapping with 2-bit precision
                let level = i & 0b11; // 2-bit quantization
                match level {
                    0 => -1.0,
                    1 => -0.33,
                    2 => 0.33,
                    3 => 1.0,
                    _ => unreachable!(),
                }
            })
            .collect()
    }
}
```

## Implementation Plan

### Phase 1: SIMD Infrastructure (Week 1)
- [ ] Implement vectorized lookup table with SIMD alignment
- [ ] Add AVX-512 gather operations for maximum performance
- [ ] Create AVX2 optimized path with manual optimizations
- [ ] Add ARM64 NEON implementation

### Phase 2: TL1/TL2 Integration (Week 2)
- [ ] Integrate SIMD lookup tables with TL1 quantization
- [ ] Optimize TL2 quantization with vectorized tables
- [ ] Add batch processing optimizations
- [ ] Create lookup table factory with capability detection

### Phase 3: Performance Optimization (Week 3)
- [ ] Add cache-friendly memory layouts
- [ ] Optimize for different table sizes
- [ ] Implement prefetching strategies
- [ ] Add memory alignment optimizations

### Phase 4: Testing & Validation (Week 4)
- [ ] Comprehensive benchmarking across architectures
- [ ] Validate numerical accuracy
- [ ] Add performance regression testing
- [ ] Create optimization guidelines

## Success Criteria

- [ ] **Performance**: 4-8x speedup on SIMD-capable hardware
- [ ] **Accuracy**: Identical results to scalar implementation
- [ ] **Architecture Support**: x86_64 AVX2/AVX-512 and ARM64 NEON
- [ ] **Memory Efficiency**: Optimal cache utilization
- [ ] **Scalability**: Performance scales with table size and batch size

## Related Issues

- #XXX: TL1/TL2 quantization performance optimization
- #XXX: SIMD capability detection and dispatch
- #XXX: Memory alignment optimization
- #XXX: Cross-platform SIMD testing

## Implementation Notes

This SIMD-optimized vectorized lookup table implementation provides significant performance improvements for TL1/TL2 quantization methods while maintaining numerical accuracy and supporting diverse hardware architectures.
