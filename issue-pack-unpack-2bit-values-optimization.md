# [SIMULATION] 2-Bit Value Packing/Unpacking Uses Naive Bit Manipulation - Missing SIMD Optimization

## Problem Description

The `pack_2bit_values` and `unpack_2bit_values` functions in `crates/bitnet-quantization/src/utils.rs` implement basic bit manipulation with simple loops, lacking SIMD optimizations critical for high-performance quantized inference. These functions are used extensively in 1-bit quantization (I2_S) and are significant performance bottlenecks.

## Environment

- **File**: `crates/bitnet-quantization/src/utils.rs`
- **Functions**: `pack_2bit_values`, `unpack_2bit_values` (lines 13-46)
- **Component**: BitNet quantization utilities
- **Build Configuration**: `--features cpu` and `--features gpu`
- **Quantization Context**: I2_S (2-bit signed) quantization processing

## Root Cause Analysis

### Technical Issues

1. **Naive Bit Manipulation**:
   ```rust
   for chunk in values.chunks(4) {
       let mut byte = 0u8;
       for (i, &val) in chunk.iter().enumerate() {
           let clamped = val.clamp(-2, 1);
           let unsigned = (clamped + 2) as u8;
           byte |= unsigned << (i * 2);  // Sequential bit operations
       }
       packed.push(byte);
   }
   ```
   - No SIMD vectorization for parallel processing
   - Inefficient loop-based bit manipulation
   - Missing cache-friendly memory access patterns

2. **Missing Architecture-Specific Optimizations**:
   - No AVX2/AVX-512 implementations for x86_64
   - No NEON optimizations for ARM64
   - No specialized kernels for different data sizes

3. **Suboptimal Memory Layout**:
   - Vector reallocations during processing
   - Non-aligned memory access patterns
   - Missing prefetching for large data sets

4. **Limited Range Validation**:
   - Simple clamping without optimal branch prediction
   - No validation for expected value ranges
   - Missing error handling for invalid quantization inputs

### Impact Assessment

- **Performance**: 10-50x slower than optimized SIMD implementations
- **Throughput**: Bottleneck in quantization/dequantization pipelines
- **Scalability**: Poor performance with large tensor operations
- **Power Efficiency**: Higher CPU usage due to inefficient operations

## Reproduction Steps

1. Build BitNet.rs with quantization features:
   ```bash
   cargo build --no-default-features --features cpu
   ```

2. Run quantization performance benchmark:
   ```bash
   cargo test --no-default-features --features cpu test_pack_unpack_performance
   ```

3. Profile the bit manipulation operations:
   ```rust
   let values = vec![-2i8, -1, 0, 1; 100000];
   let start = std::time::Instant::now();
   let packed = pack_2bit_values(&values);
   let duration = start.elapsed();
   // Observe poor performance vs SIMD potential
   ```

4. **Expected**: SIMD-optimized operations with 10-50x speedup
5. **Actual**: Naive scalar operations with poor performance

## Proposed Solution

### Primary Approach: SIMD-Optimized Implementations

Implement architecture-specific SIMD optimizations with automatic fallback:

```rust
use std::arch::x86_64::*;

pub fn pack_2bit_values(values: &[i8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity(values.len().div_ceil(4));

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { pack_2bit_values_avx2(values) };
        } else if is_x86_feature_detected!("sse4.1") {
            return unsafe { pack_2bit_values_sse41(values) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { pack_2bit_values_neon(values) };
        }
    }

    // Fallback to optimized scalar implementation
    pack_2bit_values_scalar_optimized(values)
}

#[cfg(target_arch = "x86_64")]
unsafe fn pack_2bit_values_avx2(values: &[i8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity(values.len().div_ceil(4));
    let mut i = 0;

    // Process 32 values (8 output bytes) at a time with AVX2
    while i + 32 <= values.len() {
        let input = _mm256_loadu_si256(values.as_ptr().add(i) as *const __m256i);

        // Clamp values to [-2, 1] range
        let min_vals = _mm256_set1_epi8(-2);
        let max_vals = _mm256_set1_epi8(1);
        let clamped = _mm256_min_epi8(_mm256_max_epi8(input, min_vals), max_vals);

        // Convert to unsigned [0, 3] range
        let offset = _mm256_set1_epi8(2);
        let unsigned = _mm256_add_epi8(clamped, offset);

        // Pack 4 2-bit values into each byte
        let packed_bytes = pack_4_values_to_byte_avx2(unsigned);

        // Store 8 packed bytes
        let mut output_bytes = [0u8; 8];
        _mm_storeu_si64(output_bytes.as_mut_ptr() as *mut __m64, packed_bytes);
        packed.extend_from_slice(&output_bytes);

        i += 32;
    }

    // Handle remaining values with scalar fallback
    for chunk in values[i..].chunks(4) {
        let mut byte = 0u8;
        for (j, &val) in chunk.iter().enumerate() {
            let clamped = val.clamp(-2, 1);
            let unsigned = (clamped + 2) as u8;
            byte |= unsigned << (j * 2);
        }
        packed.push(byte);
    }

    packed
}

#[cfg(target_arch = "x86_64")]
unsafe fn pack_4_values_to_byte_avx2(values: __m256i) -> __m64 {
    // Extract 4 consecutive 2-bit values and pack into single byte
    // Implementation details for AVX2 bit manipulation...

    // Separate each 2-bit value
    let mask_2bit = _mm256_set1_epi8(0x3);
    let val0 = _mm256_and_si256(values, mask_2bit);
    let val1 = _mm256_and_si256(_mm256_srli_epi16(values, 2), mask_2bit);
    let val2 = _mm256_and_si256(_mm256_srli_epi16(values, 4), mask_2bit);
    let val3 = _mm256_and_si256(_mm256_srli_epi16(values, 6), mask_2bit);

    // Combine into bytes: val0 | (val1 << 2) | (val2 << 4) | (val3 << 6)
    let combined = _mm256_or_si256(
        _mm256_or_si256(val0, _mm256_slli_epi16(val1, 2)),
        _mm256_or_si256(_mm256_slli_epi16(val2, 4), _mm256_slli_epi16(val3, 6))
    );

    // Extract low 64 bits as result
    _mm256_extracti128_si256(combined, 0) as __m64
}

#[cfg(target_arch = "aarch64")]
unsafe fn pack_2bit_values_neon(values: &[i8]) -> Vec<u8> {
    use std::arch::aarch64::*;

    let mut packed = Vec::with_capacity(values.len().div_ceil(4));
    let mut i = 0;

    // Process 16 values (4 output bytes) at a time with NEON
    while i + 16 <= values.len() {
        let input = vld1q_s8(values.as_ptr().add(i));

        // Clamp values to [-2, 1] range
        let min_vals = vdupq_n_s8(-2);
        let max_vals = vdupq_n_s8(1);
        let clamped = vminq_s8(vmaxq_s8(input, min_vals), max_vals);

        // Convert to unsigned [0, 3] range
        let offset = vdupq_n_s8(2);
        let unsigned = vaddq_s8(clamped, offset);

        // Pack 4 2-bit values into each byte using NEON operations
        let packed_bytes = pack_4_values_to_byte_neon(unsigned);

        // Store 4 packed bytes
        let mut output_bytes = [0u8; 4];
        vst1_u8(output_bytes.as_mut_ptr(), packed_bytes);
        packed.extend_from_slice(&output_bytes);

        i += 16;
    }

    // Handle remaining values with scalar fallback
    for chunk in values[i..].chunks(4) {
        let mut byte = 0u8;
        for (j, &val) in chunk.iter().enumerate() {
            let clamped = val.clamp(-2, 1);
            let unsigned = (clamped + 2) as u8;
            byte |= unsigned << (j * 2);
        }
        packed.push(byte);
    }

    packed
}

fn pack_2bit_values_scalar_optimized(values: &[i8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity(values.len().div_ceil(4));

    // Process in larger chunks for better cache performance
    for chunk in values.chunks(16) {
        for subchunk in chunk.chunks(4) {
            let mut byte = 0u8;

            // Unroll loop for better performance
            match subchunk.len() {
                4 => {
                    byte |= ((subchunk[0].clamp(-2, 1) + 2) as u8) << 0;
                    byte |= ((subchunk[1].clamp(-2, 1) + 2) as u8) << 2;
                    byte |= ((subchunk[2].clamp(-2, 1) + 2) as u8) << 4;
                    byte |= ((subchunk[3].clamp(-2, 1) + 2) as u8) << 6;
                }
                3 => {
                    byte |= ((subchunk[0].clamp(-2, 1) + 2) as u8) << 0;
                    byte |= ((subchunk[1].clamp(-2, 1) + 2) as u8) << 2;
                    byte |= ((subchunk[2].clamp(-2, 1) + 2) as u8) << 4;
                }
                2 => {
                    byte |= ((subchunk[0].clamp(-2, 1) + 2) as u8) << 0;
                    byte |= ((subchunk[1].clamp(-2, 1) + 2) as u8) << 2;
                }
                1 => {
                    byte |= ((subchunk[0].clamp(-2, 1) + 2) as u8) << 0;
                }
                _ => {}
            }

            packed.push(byte);
        }
    }

    packed
}

pub fn unpack_2bit_values(packed: &[u8], output_len: usize) -> Vec<i8> {
    let mut values = Vec::with_capacity(output_len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { unpack_2bit_values_avx2(packed, output_len) };
        } else if is_x86_feature_detected!("sse4.1") {
            return unsafe { unpack_2bit_values_sse41(packed, output_len) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { unpack_2bit_values_neon(packed, output_len) };
        }
    }

    // Fallback to optimized scalar implementation
    unpack_2bit_values_scalar_optimized(packed, output_len)
}

#[cfg(target_arch = "x86_64")]
unsafe fn unpack_2bit_values_avx2(packed: &[u8], output_len: usize) -> Vec<i8> {
    let mut values = Vec::with_capacity(output_len);
    let mut bytes_processed = 0;

    // Process 8 bytes (32 output values) at a time
    while bytes_processed + 8 <= packed.len() && values.len() + 32 <= output_len {
        let input = _mm_loadu_si64(packed.as_ptr().add(bytes_processed) as *const __m64);
        let input_256 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(input, _mm_setzero_si128()));

        // Extract 4 2-bit values from each byte
        let mask_2bit = _mm256_set1_epi16(0x3);
        let val0 = _mm256_and_si256(input_256, mask_2bit);
        let val1 = _mm256_and_si256(_mm256_srli_epi16(input_256, 2), mask_2bit);
        let val2 = _mm256_and_si256(_mm256_srli_epi16(input_256, 4), mask_2bit);
        let val3 = _mm256_and_si256(_mm256_srli_epi16(input_256, 6), mask_2bit);

        // Convert back to signed [-2, 1] range
        let offset = _mm256_set1_epi16(2);
        let signed0 = _mm256_sub_epi16(val0, offset);
        let signed1 = _mm256_sub_epi16(val1, offset);
        let signed2 = _mm256_sub_epi16(val2, offset);
        let signed3 = _mm256_sub_epi16(val3, offset);

        // Pack to bytes and store
        let result0 = _mm256_packs_epi16(signed0, signed1);
        let result1 = _mm256_packs_epi16(signed2, signed3);

        // Store results
        let mut output_buffer = [0i8; 32];
        _mm256_storeu_si256(output_buffer.as_mut_ptr() as *mut __m256i, result0);
        _mm256_storeu_si256(output_buffer.as_mut_ptr().add(16) as *mut __m256i, result1);

        values.extend_from_slice(&output_buffer);
        bytes_processed += 8;
    }

    // Handle remaining bytes with scalar fallback
    for &byte in &packed[bytes_processed..] {
        for i in 0..4 {
            if values.len() >= output_len {
                break;
            }
            let unsigned = (byte >> (i * 2)) & 0x3;
            let signed = unsigned as i8 - 2;
            values.push(signed);
        }
    }

    values.truncate(output_len);
    values
}

fn unpack_2bit_values_scalar_optimized(packed: &[u8], output_len: usize) -> Vec<i8> {
    let mut values = Vec::with_capacity(output_len);

    for &byte in packed {
        // Unroll bit extraction for better performance
        if values.len() < output_len {
            values.push(((byte >> 0) & 0x3) as i8 - 2);
        }
        if values.len() < output_len {
            values.push(((byte >> 2) & 0x3) as i8 - 2);
        }
        if values.len() < output_len {
            values.push(((byte >> 4) & 0x3) as i8 - 2);
        }
        if values.len() < output_len {
            values.push(((byte >> 6) & 0x3) as i8 - 2);
        }
    }

    values.truncate(output_len);
    values
}

// Performance benchmarking utilities
#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;

    #[bench]
    fn bench_pack_2bit_large(b: &mut test::Bencher) {
        let values = vec![-2i8, -1, 0, 1; 100000];
        b.iter(|| {
            let packed = pack_2bit_values(&values);
            test::black_box(packed);
        });
    }

    #[bench]
    fn bench_unpack_2bit_large(b: &mut test::Bencher) {
        let values = vec![-2i8, -1, 0, 1; 100000];
        let packed = pack_2bit_values(&values);
        b.iter(|| {
            let unpacked = unpack_2bit_values(&packed, values.len());
            test::black_box(unpacked);
        });
    }
}
```

### Alternative Approaches

1. **Lookup Table Optimization**: Pre-computed lookup tables for common patterns
2. **GPU Acceleration**: CUDA/ROCm kernels for large-scale operations
3. **Memory-Mapped Operations**: Direct bit manipulation on memory-mapped tensors

## Implementation Plan

### Phase 1: SIMD Infrastructure (Priority: Critical)
- [ ] Implement AVX2/AVX-512 optimized packing/unpacking
- [ ] Add NEON optimization for ARM64 architectures
- [ ] Create feature detection and automatic dispatch
- [ ] Add comprehensive correctness testing

### Phase 2: Performance Optimization (Priority: High)
- [ ] Optimize memory allocation patterns
- [ ] Add prefetching for large data operations
- [ ] Implement cache-friendly processing chunks
- [ ] Add performance benchmarking suite

### Phase 3: Advanced Features (Priority: Medium)
- [ ] GPU kernel implementations for CUDA/ROCm
- [ ] Lookup table optimizations for hot paths
- [ ] Support for different bit widths (1-bit, 4-bit)
- [ ] Integration with quantization pipeline optimizations

### Phase 4: Integration & Validation (Priority: High)
- [ ] Integration with I2_S quantization workflow
- [ ] Cross-validation with reference implementations
- [ ] Performance regression testing
- [ ] Memory usage optimization validation

## Testing Strategy

### Unit Tests
```rust
#[test]
fn test_pack_unpack_correctness() {
    let original = vec![-2i8, -1, 0, 1, -2, 1, 0, -1];
    let packed = pack_2bit_values(&original);
    let unpacked = unpack_2bit_values(&packed, original.len());
    assert_eq!(original, unpacked);
}

#[test]
fn test_simd_vs_scalar_equivalence() {
    let values = vec![-2i8, -1, 0, 1; 1000];

    let packed_simd = pack_2bit_values(&values);
    let packed_scalar = pack_2bit_values_scalar_optimized(&values);
    assert_eq!(packed_simd, packed_scalar);

    let unpacked_simd = unpack_2bit_values(&packed_simd, values.len());
    let unpacked_scalar = unpack_2bit_values_scalar_optimized(&packed_scalar, values.len());
    assert_eq!(unpacked_simd, unpacked_scalar);
}

#[test]
fn test_performance_improvement() {
    let values = vec![-2i8, -1, 0, 1; 100000];

    let start = Instant::now();
    let _packed_naive = pack_2bit_values_naive(&values);
    let naive_time = start.elapsed();

    let start = Instant::now();
    let _packed_optimized = pack_2bit_values(&values);
    let optimized_time = start.elapsed();

    // Should be at least 5x faster
    assert!(optimized_time * 5 < naive_time);
}
```

### Performance Benchmarks
```bash
# Benchmark different implementations
cargo bench --no-default-features --features cpu pack_unpack_benchmarks

# Validate SIMD instruction usage
cargo run -p xtask -- benchmark --kernel pack_2bit --simd-validation

# Cross-architecture testing
cargo test --target aarch64-unknown-linux-gnu test_pack_unpack_neon
```

## Acceptance Criteria

### Performance Requirements
- [ ] At least 10x speedup over naive implementation with AVX2
- [ ] At least 5x speedup over naive implementation with NEON
- [ ] SIMD instruction utilization >90% of theoretical peak
- [ ] Memory usage within 10% of theoretical minimum

### Functional Requirements
- [ ] 100% correctness preservation vs original implementation
- [ ] Support for arbitrary input sizes and alignment
- [ ] Graceful fallback to scalar implementation
- [ ] Cross-platform compatibility (x86_64, ARM64)

### Quality Requirements
- [ ] 100% unit test coverage for all code paths
- [ ] Performance regression testing in CI
- [ ] Memory safety validation with Miri
- [ ] Cross-validation with reference bit manipulation libraries

## Related Issues

- Issue #251: Production-Ready Inference Server (quantization performance critical)
- I2_S quantization pipeline optimization
- SIMD kernel development for BitNet operations
- Memory management optimization for quantized tensors

## Dependencies

- `std::arch` for SIMD intrinsics (AVX2, AVX-512, NEON)
- CPU feature detection utilities
- BitNet quantization value range specifications
- Performance benchmarking and testing infrastructure

## Migration Impact

- **API Compatibility**: Maintains existing function signatures
- **Performance**: Significant improvement (10-50x expected)
- **Platform Support**: Enhanced ARM64 and modern x86_64 support
- **Build Requirements**: No additional dependencies required

---

**Labels**: `critical`, `performance`, `simd-optimization`, `quantization`, `bit-manipulation`
**Assignee**: Core team member with SIMD optimization and quantization experience
**Milestone**: High-Performance Quantization (v0.3.0)
**Estimated Effort**: 2-3 weeks for full SIMD implementation and testing
