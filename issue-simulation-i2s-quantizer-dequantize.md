# [OPTIMIZATION] Enhance I2S dequantization with optimized SIMD kernels and validation

## Problem Description
The `I2SQuantizer::dequantize` function in `crates/bitnet-quantization/src/i2s.rs` may use simplified implementations of `unpack_2bit_values` and `dequantize_simd` that could be further optimized for production performance.

## Environment
- **File**: `crates/bitnet-quantization/src/i2s.rs`
- **Function**: `I2SQuantizer::dequantize`
- **Current State**: Potentially unoptimized SIMD operations

## Root Cause Analysis
Current implementation relies on:
```rust
let quantized_data = unpack_2bit_values(&tensor.data, tensor_numel);
let dequantized_data = self.kernels.dequantize_simd(&quantized_data, &tensor.scales, self.block_size)?;
```

**Potential Issues:**
1. `unpack_2bit_values` may not use optimal bit manipulation
2. `dequantize_simd` might not leverage latest SIMD intrinsics
3. No platform-specific optimizations (AVX-512, NEON)
4. Potential memory alignment issues affecting SIMD performance

## Proposed Solution
```rust
impl I2SQuantizer {
    pub fn dequantize(&self, tensor: &QuantizedTensor, device: &Device) -> Result<BitNetTensor> {
        // ... existing validation ...

        // Optimized 2-bit unpacking with SIMD
        let quantized_data = self.unpack_2bit_values_optimized(&tensor.data, tensor_numel)?;

        // Platform-specific SIMD dequantization
        let dequantized_data = match self.simd_level {
            SimdLevel::Avx512 => self.kernels.dequantize_avx512(&quantized_data, &tensor.scales, self.block_size)?,
            SimdLevel::Avx2 => self.kernels.dequantize_avx2(&quantized_data, &tensor.scales, self.block_size)?,
            SimdLevel::Neon => self.kernels.dequantize_neon(&quantized_data, &tensor.scales, self.block_size)?,
            SimdLevel::Scalar => self.kernels.dequantize_scalar(&quantized_data, &tensor.scales, self.block_size)?,
        };

        create_tensor_from_f32(dequantized_data, &tensor.shape, device)
    }

    fn unpack_2bit_values_optimized(&self, data: &[u8], numel: usize) -> Result<Vec<i8>> {
        let mut result = vec![0i8; numel];

        // Vectorized unpacking using platform intrinsics
        #[cfg(target_feature = "avx2")]
        if self.simd_level >= SimdLevel::Avx2 {
            return self.unpack_2bit_avx2(data, &mut result);
        }

        // Fallback scalar implementation
        self.unpack_2bit_scalar(data, &mut result)
    }
}
```

## Implementation Plan
### Phase 1: SIMD Detection & Optimization (2 days)
- [ ] Implement runtime SIMD capability detection
- [ ] Create optimized AVX2/AVX-512 unpacking kernels
- [ ] Add ARM NEON support for ARM platforms

### Phase 2: Performance Validation (1 day)
- [ ] Benchmark against current implementation
- [ ] Validate accuracy with extensive test suite
- [ ] Profile memory access patterns and alignment

## Acceptance Criteria
- [ ] ≥30% performance improvement on AVX2+ platforms
- [ ] Maintain bit-perfect accuracy vs reference
- [ ] Support for multiple SIMD instruction sets
- [ ] Runtime feature detection and fallback

**Labels**: `optimization`, `quantization`, `simd`, `P2-medium`
**Effort**: 3 days