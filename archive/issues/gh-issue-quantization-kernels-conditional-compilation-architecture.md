# [PERF] SIMD Quantization Kernels Conditional Compilation Architecture

## Problem Description

The current quantization kernel implementation uses conditional compilation (`#[target_feature(enable = "avx2")]`) for SIMD operations, which creates an inconsistent API surface and prevents proper runtime optimization selection. This approach limits the ability to dynamically select optimal kernels based on actual hardware capabilities and doesn't provide clear error handling when specific instructions are unavailable.

## Environment

- **Component**: `bitnet-quantization` crate
- **File**: `crates/bitnet-quantization/src/simd_ops.rs`
- **Rust Version**: 1.90.0+ (2024 edition)
- **Target Architectures**: x86_64 (AVX2, AVX-512), ARM64 (NEON)
- **SIMD Requirements**: AVX2 minimum, AVX-512 preferred on supported hardware

## Current Implementation Analysis

### Problematic Conditional Compilation Pattern
```rust
impl QuantizationKernels {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]  // Compile-time only - no runtime selection
    unsafe fn quantize_avx2_block(&self, data: &[f32], output: &mut [i8], scale: f32, bits: u8) {
        // Implementation tied to compile-time feature detection
        // Cannot adapt to different hardware at runtime
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]  // Same limitation
    unsafe fn dequantize_avx2_block(&self, quantized: &[i8], output: &mut [f32], scale: f32) {
        // Rigid compile-time dispatch
    }
}
```

### Architecture Limitations
1. **No Runtime Adaptation**: Cannot select optimal kernel based on actual CPU
2. **API Inconsistency**: Different function signatures based on compilation target
3. **Error Handling Gap**: No graceful degradation when SIMD unavailable
4. **Performance Suboptimality**: Cannot use best available instruction set

## Root Cause Analysis

1. **Compile-Time vs Runtime**: Feature detection happens at compile time, not runtime
2. **Rigid Architecture**: Single kernel per compilation rather than adaptive selection
3. **Missing Abstraction**: No unified interface for different SIMD implementations
4. **Limited Scalability**: Cannot support multiple instruction set levels efficiently

## Impact Assessment

**Severity**: High - Core quantization performance and adaptability

**Affected Systems**:
- Heterogeneous deployment environments
- Cloud instances with varying CPU capabilities
- Edge devices with different SIMD support
- Performance optimization opportunities

**Performance Impact**:
- Suboptimal kernel selection in production
- Missed opportunities for AVX-512 acceleration
- Inconsistent performance across deployments

## Proposed Solution

### Dynamic SIMD Kernel Architecture

Replace conditional compilation with runtime dispatch system:

```rust
use std::sync::OnceLock;

/// SIMD capability detection and kernel selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimdCapability {
    Scalar,
    Sse2,
    Avx2,
    Avx512,
    Neon,       // ARM64
    NeonDotprod, // ARM64 with dot product
}

impl SimdCapability {
    /// Detect best available SIMD capability at runtime
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                return SimdCapability::Avx512;
            }
            if is_x86_feature_detected!("avx2") {
                return SimdCapability::Avx2;
            }
            if is_x86_feature_detected!("sse2") {
                return SimdCapability::Sse2;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if is_aarch64_feature_detected!("dotprod") {
                return SimdCapability::NeonDotprod;
            }
            if is_aarch64_feature_detected!("neon") {
                return SimdCapability::Neon;
            }
        }

        SimdCapability::Scalar
    }

    /// Check if capability supports quantization optimization
    pub fn supports_i2s_quantization(self) -> bool {
        match self {
            SimdCapability::Avx512 | SimdCapability::Avx2 => true,
            SimdCapability::NeonDotprod | SimdCapability::Neon => true,
            _ => false,
        }
    }
}

/// Global SIMD capability - initialized once at program start
static SIMD_CAPABILITY: OnceLock<SimdCapability> = OnceLock::new();

pub fn get_simd_capability() -> SimdCapability {
    *SIMD_CAPABILITY.get_or_init(SimdCapability::detect)
}

/// Quantization kernel trait for different SIMD implementations
pub trait QuantizationKernel: Send + Sync {
    fn name(&self) -> &'static str;
    fn capability(&self) -> SimdCapability;

    /// Quantize f32 values to I2S format with SIMD optimization
    fn quantize_i2s(
        &self,
        input: &[f32],
        output: &mut [i8],
        scales: &mut [f32],
        block_size: usize,
    ) -> Result<(), QuantizationError>;

    /// Dequantize I2S values to f32 with SIMD optimization
    fn dequantize_i2s(
        &self,
        input: &[i8],
        output: &mut [f32],
        scales: &[f32],
        block_size: usize,
    ) -> Result<(), QuantizationError>;

    /// Matrix multiplication with quantized weights
    fn matmul_i2s(
        &self,
        a: &[f32],      // Activations
        b: &[i8],       // Quantized weights
        c: &mut [f32],  // Output
        m: usize, n: usize, k: usize,
        scales: &[f32],
    ) -> Result<(), QuantizationError>;
}

/// AVX2 optimized quantization kernel
#[derive(Debug)]
pub struct Avx2QuantizationKernel;

impl QuantizationKernel for Avx2QuantizationKernel {
    fn name(&self) -> &'static str { "AVX2" }
    fn capability(&self) -> SimdCapability { SimdCapability::Avx2 }

    fn quantize_i2s(
        &self,
        input: &[f32],
        output: &mut [i8],
        scales: &mut [f32],
        block_size: usize,
    ) -> Result<(), QuantizationError> {
        if !is_x86_feature_detected!("avx2") {
            return Err(QuantizationError::UnsupportedHardware {
                required: "AVX2".to_string(),
                available: get_simd_capability(),
            });
        }

        unsafe { self.quantize_i2s_avx2_impl(input, output, scales, block_size) }
    }

    fn dequantize_i2s(
        &self,
        input: &[i8],
        output: &mut [f32],
        scales: &[f32],
        block_size: usize,
    ) -> Result<(), QuantizationError> {
        if !is_x86_feature_detected!("avx2") {
            return Err(QuantizationError::UnsupportedHardware {
                required: "AVX2".to_string(),
                available: get_simd_capability(),
            });
        }

        unsafe { self.dequantize_i2s_avx2_impl(input, output, scales, block_size) }
    }

    fn matmul_i2s(
        &self,
        a: &[f32],
        b: &[i8],
        c: &mut [f32],
        m: usize, n: usize, k: usize,
        scales: &[f32],
    ) -> Result<(), QuantizationError> {
        if !is_x86_feature_detected!("avx2") {
            return Err(QuantizationError::UnsupportedHardware {
                required: "AVX2".to_string(),
                available: get_simd_capability(),
            });
        }

        unsafe { self.matmul_i2s_avx2_impl(a, b, c, m, n, k, scales) }
    }
}

impl Avx2QuantizationKernel {
    #[target_feature(enable = "avx2")]
    unsafe fn quantize_i2s_avx2_impl(
        &self,
        input: &[f32],
        output: &mut [i8],
        scales: &mut [f32],
        block_size: usize,
    ) -> Result<(), QuantizationError> {
        use std::arch::x86_64::*;

        let num_blocks = (input.len() + block_size - 1) / block_size;

        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = std::cmp::min(start + block_size, input.len());
            let block = &input[start..end];

            // Compute scale for this block using AVX2
            let mut max_val = 0.0f32;
            let mut i = 0;

            // Process 8 floats at a time with AVX2
            while i + 8 <= block.len() {
                let data = _mm256_loadu_ps(block.as_ptr().add(i));
                let abs_data = _mm256_and_ps(data, _mm256_set1_ps(f32::from_bits(0x7FFFFFFF)));

                // Find max using horizontal max operations
                let max_vec = _mm256_max_ps(abs_data, _mm256_permute2f128_ps(abs_data, abs_data, 1));
                let max_vec = _mm256_max_ps(max_vec, _mm256_permute_ps(max_vec, 0b01001110));
                let max_vec = _mm256_max_ps(max_vec, _mm256_permute_ps(max_vec, 0b10110001));

                let block_max = _mm256_cvtss_f32(max_vec);
                max_val = max_val.max(block_max);
                i += 8;
            }

            // Handle remaining elements
            for &val in &block[i..] {
                max_val = max_val.max(val.abs());
            }

            // Compute scale (I2S uses {-1, 0, 1})
            let scale = if max_val > 0.0 { max_val } else { 1.0 };
            scales[block_idx] = scale;
            let inv_scale = 1.0 / scale;

            // Quantize block using AVX2
            i = 0;
            while i + 8 <= block.len() {
                let data = _mm256_loadu_ps(block.as_ptr().add(i));
                let scaled = _mm256_mul_ps(data, _mm256_set1_ps(inv_scale));

                // Quantize to {-1, 0, 1}
                let neg_ones = _mm256_cmp_ps(scaled, _mm256_set1_ps(-0.5), _CMP_LT_OQ);
                let pos_ones = _mm256_cmp_ps(scaled, _mm256_set1_ps(0.5), _CMP_GT_OQ);

                let result = _mm256_blendv_ps(
                    _mm256_blendv_ps(_mm256_setzero_ps(), _mm256_set1_ps(1.0), pos_ones),
                    _mm256_set1_ps(-1.0),
                    neg_ones
                );

                // Convert to i8 and store
                let result_i32 = _mm256_cvtps_epi32(result);
                let result_i16 = _mm256_packs_epi32(result_i32, _mm256_setzero_si256());
                let result_i8 = _mm256_packs_epi16(result_i16, _mm256_setzero_si256());

                // Extract and store 8 bytes
                let result_64 = _mm256_extract_epi64(result_i8, 0) as u64;
                let output_ptr = output.as_mut_ptr().add(start + i) as *mut u64;
                std::ptr::write_unaligned(output_ptr, result_64);

                i += 8;
            }

            // Handle remaining elements
            for j in i..block.len() {
                let val = block[j] * inv_scale;
                output[start + j] = if val < -0.5 {
                    -1
                } else if val > 0.5 {
                    1
                } else {
                    0
                };
            }
        }

        Ok(())
    }

    #[target_feature(enable = "avx2")]
    unsafe fn dequantize_i2s_avx2_impl(
        &self,
        input: &[i8],
        output: &mut [f32],
        scales: &[f32],
        block_size: usize,
    ) -> Result<(), QuantizationError> {
        use std::arch::x86_64::*;

        let num_blocks = (input.len() + block_size - 1) / block_size;

        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = std::cmp::min(start + block_size, input.len());
            let scale = scales[block_idx];

            let mut i = 0;
            while i + 8 <= (end - start) {
                // Load 8 quantized values
                let quantized_64 = std::ptr::read_unaligned(
                    input.as_ptr().add(start + i) as *const u64
                );

                // Convert to AVX2 register
                let quantized_vec = _mm256_set_epi64x(0, 0, 0, quantized_64 as i64);
                let quantized_i32 = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(quantized_vec));
                let quantized_f32 = _mm256_cvtepi32_ps(quantized_i32);

                // Scale to original range
                let result = _mm256_mul_ps(quantized_f32, _mm256_set1_ps(scale));

                // Store result
                _mm256_storeu_ps(output.as_mut_ptr().add(start + i), result);
                i += 8;
            }

            // Handle remaining elements
            for j in i..(end - start) {
                output[start + j] = input[start + j] as f32 * scale;
            }
        }

        Ok(())
    }

    #[target_feature(enable = "avx2")]
    unsafe fn matmul_i2s_avx2_impl(
        &self,
        a: &[f32],
        b: &[i8],
        c: &mut [f32],
        m: usize, n: usize, k: usize,
        scales: &[f32],
    ) -> Result<(), QuantizationError> {
        use std::arch::x86_64::*;

        // Optimized matrix multiplication for I2S quantized weights
        for i in 0..m {
            for j in 0..n {
                let mut sum = _mm256_setzero_ps();
                let mut l = 0;

                // Process 8 elements at a time
                while l + 8 <= k {
                    let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * k + l));

                    // Load quantized weights and convert to f32
                    let b_i8 = _mm256_set_epi64x(
                        0, 0, 0,
                        std::ptr::read_unaligned(b.as_ptr().add(l * n + j) as *const u64) as i64
                    );
                    let b_i32 = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(b_i8));
                    let b_f32 = _mm256_cvtepi32_ps(b_i32);

                    // Multiply and accumulate
                    sum = _mm256_fmadd_ps(a_vec, b_f32, sum);
                    l += 8;
                }

                // Horizontal sum of AVX2 vector
                let sum_hi = _mm256_extractf128_ps(sum, 1);
                let sum_lo = _mm256_extractf128_ps(sum, 0);
                let sum_128 = _mm_add_ps(sum_hi, sum_lo);
                let sum_64 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
                let sum_32 = _mm_add_ss(sum_64, _mm_movehdup_ps(sum_64));
                let mut result = _mm_cvtss_f32(sum_32);

                // Handle remaining elements
                for idx in l..k {
                    result += a[i * k + idx] * (b[idx * n + j] as f32);
                }

                // Apply scale and store
                c[i * n + j] = result * scales[j / block_size];
            }
        }

        Ok(())
    }
}

/// AVX-512 optimized kernel
#[derive(Debug)]
pub struct Avx512QuantizationKernel;

impl QuantizationKernel for Avx512QuantizationKernel {
    fn name(&self) -> &'static str { "AVX-512" }
    fn capability(&self) -> SimdCapability { SimdCapability::Avx512 }

    // Similar implementation but with 16-wide SIMD operations
    fn quantize_i2s(
        &self,
        input: &[f32],
        output: &mut [i8],
        scales: &mut [f32],
        block_size: usize,
    ) -> Result<(), QuantizationError> {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512bw") {
            return Err(QuantizationError::UnsupportedHardware {
                required: "AVX-512".to_string(),
                available: get_simd_capability(),
            });
        }

        // AVX-512 implementation processing 16 elements at a time
        // Implementation would be similar to AVX2 but with wider vectors
        todo!("AVX-512 implementation")
    }

    // ... other methods
}

/// Scalar fallback kernel
#[derive(Debug)]
pub struct ScalarQuantizationKernel;

impl QuantizationKernel for ScalarQuantizationKernel {
    fn name(&self) -> &'static str { "Scalar" }
    fn capability(&self) -> SimdCapability { SimdCapability::Scalar }

    fn quantize_i2s(
        &self,
        input: &[f32],
        output: &mut [i8],
        scales: &mut [f32],
        block_size: usize,
    ) -> Result<(), QuantizationError> {
        // Simple scalar implementation
        let num_blocks = (input.len() + block_size - 1) / block_size;

        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = std::cmp::min(start + block_size, input.len());
            let block = &input[start..end];

            // Find max absolute value for scale
            let max_val = block.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let scale = if max_val > 0.0 { max_val } else { 1.0 };
            scales[block_idx] = scale;

            // Quantize to {-1, 0, 1}
            let inv_scale = 1.0 / scale;
            for (i, &val) in block.iter().enumerate() {
                let scaled = val * inv_scale;
                output[start + i] = if scaled < -0.5 {
                    -1
                } else if scaled > 0.5 {
                    1
                } else {
                    0
                };
            }
        }

        Ok(())
    }

    // ... other methods with scalar implementations
}

/// Kernel factory and selection
pub struct QuantizationKernelSelector {
    available_kernels: Vec<Box<dyn QuantizationKernel>>,
    optimal_kernel: Box<dyn QuantizationKernel>,
}

impl QuantizationKernelSelector {
    pub fn new() -> Self {
        let mut available_kernels: Vec<Box<dyn QuantizationKernel>> = vec![
            Box::new(ScalarQuantizationKernel),
        ];

        // Add SIMD kernels based on runtime detection
        let capability = get_simd_capability();

        #[cfg(target_arch = "x86_64")]
        {
            if capability >= SimdCapability::Avx2 {
                available_kernels.push(Box::new(Avx2QuantizationKernel));
            }
            if capability >= SimdCapability::Avx512 {
                available_kernels.push(Box::new(Avx512QuantizationKernel));
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if capability >= SimdCapability::Neon {
                available_kernels.push(Box::new(NeonQuantizationKernel));
            }
        }

        // Select optimal kernel
        let optimal_kernel = available_kernels
            .iter()
            .max_by_key(|k| k.capability() as u8)
            .unwrap();

        Self {
            available_kernels,
            optimal_kernel: optimal_kernel.clone(), // Would need Clone implementation
        }
    }

    pub fn optimal(&self) -> &dyn QuantizationKernel {
        self.optimal_kernel.as_ref()
    }

    pub fn get_kernel(&self, capability: SimdCapability) -> Option<&dyn QuantizationKernel> {
        self.available_kernels
            .iter()
            .find(|k| k.capability() == capability)
            .map(|k| k.as_ref())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum QuantizationError {
    #[error("Unsupported hardware: requires {required}, available: {available:?}")]
    UnsupportedHardware {
        required: String,
        available: SimdCapability,
    },

    #[error("Invalid dimensions: input={input}, output={output}")]
    DimensionMismatch { input: usize, output: usize },

    #[error("Block size must be positive and power of 2, got {0}")]
    InvalidBlockSize(usize),
}
```

## Implementation Plan

### Phase 1: Architecture Foundation (Week 1)
- [ ] Define SIMD capability detection and kernel trait
- [ ] Implement runtime dispatch system
- [ ] Create scalar fallback implementations
- [ ] Establish benchmarking framework

### Phase 2: SIMD Implementations (Week 2-3)
- [ ] Implement AVX2 quantization kernels
- [ ] Implement AVX-512 quantization kernels (where available)
- [ ] Implement ARM NEON quantization kernels
- [ ] Add comprehensive SIMD intrinsics usage

### Phase 3: Integration & Optimization (Week 4)
- [ ] Integrate with existing quantization pipeline
- [ ] Performance optimization and micro-benchmarking
- [ ] Error handling and edge case coverage
- [ ] Memory alignment and cache optimization

### Phase 4: Production Readiness (Week 5)
- [ ] Comprehensive testing across hardware platforms
- [ ] Documentation and examples
- [ ] Feature flags for gradual rollout
- [ ] Performance regression monitoring

## Testing Strategy

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_capability_detection() {
        let capability = SimdCapability::detect();

        #[cfg(target_arch = "x86_64")]
        {
            // Should detect at least SSE2 on modern x86_64
            assert!(capability >= SimdCapability::Sse2);
        }

        #[cfg(target_arch = "aarch64")]
        {
            // Should detect NEON on ARM64
            assert!(capability >= SimdCapability::Neon);
        }
    }

    #[test]
    fn test_kernel_selection() {
        let selector = QuantizationKernelSelector::new();
        let optimal = selector.optimal();

        // Should select best available kernel
        assert!(optimal.capability().supports_i2s_quantization());
    }

    #[test]
    fn test_quantization_accuracy() {
        let selector = QuantizationKernelSelector::new();

        for kernel in &selector.available_kernels {
            let input = vec![0.5f32, -0.3, 0.8, -0.1, 0.0];
            let mut output = vec![0i8; input.len()];
            let mut scales = vec![0.0f32; 1];

            let result = kernel.quantize_i2s(&input, &mut output, &mut scales, input.len());
            assert!(result.is_ok());

            // Validate quantization to {-1, 0, 1}
            for &val in &output {
                assert!(val >= -1 && val <= 1);
            }
        }
    }

    #[test]
    fn test_kernel_parity() {
        // Ensure all kernels produce equivalent results
        let selector = QuantizationKernelSelector::new();
        let scalar_kernel = selector.get_kernel(SimdCapability::Scalar).unwrap();

        let input = generate_test_data(1024);
        let mut scalar_output = vec![0i8; input.len()];
        let mut scalar_scales = vec![0.0f32; 16];

        scalar_kernel.quantize_i2s(&input, &mut scalar_output, &mut scalar_scales, 64).unwrap();

        for kernel in &selector.available_kernels {
            if kernel.capability() == SimdCapability::Scalar {
                continue;
            }

            let mut simd_output = vec![0i8; input.len()];
            let mut simd_scales = vec![0.0f32; 16];

            if kernel.quantize_i2s(&input, &mut simd_output, &mut simd_scales, 64).is_ok() {
                // Results should be identical (or within epsilon for floating point)
                assert_eq!(scalar_output, simd_output);
                for (s1, s2) in scalar_scales.iter().zip(simd_scales.iter()) {
                    assert!((s1 - s2).abs() < 1e-6);
                }
            }
        }
    }
}

#[cfg(test)]
mod benchmarks {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion};

    fn bench_quantization_kernels(c: &mut Criterion) {
        let selector = QuantizationKernelSelector::new();
        let input = generate_test_data(8192);

        let mut group = c.benchmark_group("quantization_kernels");

        for kernel in &selector.available_kernels {
            group.bench_with_input(
                BenchmarkId::new("quantize_i2s", kernel.name()),
                &input,
                |b, input| {
                    let mut output = vec![0i8; input.len()];
                    let mut scales = vec![0.0f32; 128];

                    b.iter(|| {
                        kernel.quantize_i2s(
                            black_box(input),
                            black_box(&mut output),
                            black_box(&mut scales),
                            64
                        )
                    });
                }
            );
        }

        group.finish();
    }

    criterion_group!(benches, bench_quantization_kernels);
    criterion_main!(benches);
}
```

## Success Criteria

- [ ] **Runtime Adaptability**: Optimal kernel selected based on actual hardware
- [ ] **Performance Parity**: SIMD implementations >= 3x scalar performance
- [ ] **Numerical Accuracy**: All kernels produce identical quantization results
- [ ] **Error Handling**: Clear errors when hardware requirements not met
- [ ] **Cross-Platform**: Consistent API across x86_64 and ARM64
- [ ] **Maintainability**: Clean architecture supporting future SIMD extensions

## Related Issues

- #XXX: Matrix multiplication SIMD optimization
- #XXX: GPU kernel integration with CPU fallbacks
- #XXX: Performance benchmarking automation
- #XXX: Memory alignment optimization for SIMD operations

## Implementation Notes

This architecture enables BitNet-rs to automatically adapt to different hardware capabilities while maintaining a consistent API. The runtime dispatch system ensures optimal performance on each deployment target while providing clear error handling and fallback strategies.
