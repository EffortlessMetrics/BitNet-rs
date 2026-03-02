//! SIMD optimization utilities for quantization operations
//!
//! This module provides common SIMD patterns used across different quantization types
//! to eliminate code duplication and ensure consistent performance optimizations.

use bitnet_common::Result;
use rayon::prelude::*;

/// Architecture-specific SIMD capabilities
#[derive(Debug, Clone, Copy)]
pub struct SimdCapabilities {
    pub has_avx512: bool,
    pub has_avx2: bool,
    pub has_neon: bool,
    pub has_sse4_1: bool,
}

impl SimdCapabilities {
    /// Detect SIMD capabilities for the current architecture
    pub fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            Self {
                has_avx512: is_x86_feature_detected!("avx512f"),
                has_avx2: is_x86_feature_detected!("avx2"),
                has_neon: false,
                has_sse4_1: is_x86_feature_detected!("sse4.1"),
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                has_avx512: false,
                has_avx2: false,
                has_neon: std::arch::is_aarch64_feature_detected!("neon"),
                has_sse4_1: false,
            }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self { has_avx512: false, has_avx2: false, has_neon: false, has_sse4_1: false }
        }
    }

    /// Determine the best quantization strategy based on capabilities
    pub fn best_quantization_strategy(&self) -> QuantizationStrategy {
        if self.has_avx512 {
            QuantizationStrategy::AVX512
        } else if self.has_avx2 {
            QuantizationStrategy::AVX2
        } else if self.has_neon {
            QuantizationStrategy::NEON
        } else if self.has_sse4_1 {
            QuantizationStrategy::SSE4_1
        } else {
            QuantizationStrategy::Scalar
        }
    }

    /// Get optimal block size for the current architecture
    pub fn optimal_block_size(&self) -> usize {
        match self.best_quantization_strategy() {
            QuantizationStrategy::AVX512 => 256,
            QuantizationStrategy::AVX2 => 128,
            QuantizationStrategy::NEON => 64,
            QuantizationStrategy::SSE4_1 => 64,
            QuantizationStrategy::Scalar => 32,
        }
    }
}

/// Quantization strategy based on available SIMD instructions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationStrategy {
    Scalar,
    SSE4_1,
    AVX2,
    AVX512,
    NEON,
}

/// Common quantization kernels with architecture-specific optimizations
pub struct QuantizationKernels {
    capabilities: SimdCapabilities,
}

impl QuantizationKernels {
    /// Create new quantization kernels with detected SIMD capabilities
    pub fn new() -> Self {
        Self { capabilities: SimdCapabilities::detect() }
    }

    /// Create with specific capabilities (for testing)
    pub fn with_capabilities(capabilities: SimdCapabilities) -> Self {
        Self { capabilities }
    }

    /// Get the SIMD capabilities
    pub fn capabilities(&self) -> SimdCapabilities {
        self.capabilities
    }

    /// Scalar quantization implementation with safety checks
    pub fn quantize_scalar(
        &self,
        data: &[f32],
        scales: &[f32],
        block_size: usize,
        bits: u8,
    ) -> Result<Vec<i8>> {
        let mut quantized = vec![0i8; data.len()];

        quantized
            .par_chunks_mut(block_size)
            .zip(data.par_chunks(block_size))
            .zip(scales.par_iter())
            .for_each(|((quant_block, data_block), &scale)| {
                for (i, &value) in data_block.iter().enumerate() {
                    quant_block[i] = quantize_value_scalar(value, scale, bits);
                }
            });

        Ok(quantized)
    }

    /// Scalar dequantization implementation with safety checks
    pub fn dequantize_scalar(
        &self,
        quantized: &[i8],
        scales: &[f32],
        block_size: usize,
    ) -> Result<Vec<f32>> {
        let mut dequantized = vec![0.0f32; quantized.len()];

        dequantized
            .par_chunks_mut(block_size)
            .zip(quantized.par_chunks(block_size))
            .zip(scales.par_iter())
            .for_each(|((dequant_block, quant_block), &scale)| {
                for (i, &value) in quant_block.iter().enumerate() {
                    dequant_block[i] = dequantize_value_scalar(value, scale);
                }
            });

        Ok(dequantized)
    }

    /// SIMD-optimized quantization with automatic fallback
    pub fn quantize_simd(
        &self,
        data: &[f32],
        scales: &[f32],
        block_size: usize,
        bits: u8,
    ) -> Result<Vec<i8>> {
        match self.capabilities.best_quantization_strategy() {
            #[cfg(target_arch = "x86_64")]
            QuantizationStrategy::AVX2 if self.capabilities.has_avx2 => {
                self.quantize_avx2(data, scales, block_size, bits)
            }
            #[cfg(target_arch = "aarch64")]
            QuantizationStrategy::NEON if self.capabilities.has_neon => {
                self.quantize_neon(data, scales, block_size, bits)
            }
            _ => self.quantize_scalar(data, scales, block_size, bits),
        }
    }

    /// SIMD-optimized dequantization with automatic fallback
    pub fn dequantize_simd(
        &self,
        quantized: &[i8],
        scales: &[f32],
        block_size: usize,
    ) -> Result<Vec<f32>> {
        match self.capabilities.best_quantization_strategy() {
            #[cfg(target_arch = "x86_64")]
            QuantizationStrategy::AVX2 if self.capabilities.has_avx2 => {
                self.dequantize_avx2(quantized, scales, block_size)
            }
            #[cfg(target_arch = "aarch64")]
            QuantizationStrategy::NEON if self.capabilities.has_neon => {
                self.dequantize_neon(quantized, scales, block_size)
            }
            _ => self.dequantize_scalar(quantized, scales, block_size),
        }
    }

    /// AVX2-optimized quantization for x86_64
    #[cfg(target_arch = "x86_64")]
    fn quantize_avx2(
        &self,
        data: &[f32],
        scales: &[f32],
        block_size: usize,
        bits: u8,
    ) -> Result<Vec<i8>> {
        if !is_x86_feature_detected!("avx2") {
            return self.quantize_scalar(data, scales, block_size, bits);
        }

        let mut quantized = vec![0i8; data.len()];

        quantized
            .par_chunks_mut(block_size)
            .zip(data.par_chunks(block_size))
            .zip(scales.par_iter())
            .for_each(|((quant_block, data_block), &scale)| unsafe {
                self.quantize_avx2_block(data_block, quant_block, scale, bits);
            });

        Ok(quantized)
    }

    /// AVX2-optimized dequantization for x86_64
    #[cfg(target_arch = "x86_64")]
    fn dequantize_avx2(
        &self,
        quantized: &[i8],
        scales: &[f32],
        block_size: usize,
    ) -> Result<Vec<f32>> {
        if !is_x86_feature_detected!("avx2") {
            return self.dequantize_scalar(quantized, scales, block_size);
        }

        let mut dequantized = vec![0.0f32; quantized.len()];

        dequantized
            .par_chunks_mut(block_size)
            .zip(quantized.par_chunks(block_size))
            .zip(scales.par_iter())
            .for_each(|((dequant_block, quant_block), &scale)| unsafe {
                self.dequantize_avx2_block(quant_block, dequant_block, scale);
            });

        Ok(dequantized)
    }

    /// NEON-optimized quantization for ARM64
    #[cfg(target_arch = "aarch64")]
    fn quantize_neon(
        &self,
        data: &[f32],
        scales: &[f32],
        block_size: usize,
        bits: u8,
    ) -> Result<Vec<i8>> {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return self.quantize_scalar(data, scales, block_size, bits);
        }

        let mut quantized = vec![0i8; data.len()];

        quantized
            .par_chunks_mut(block_size)
            .zip(data.par_chunks(block_size))
            .zip(scales.par_iter())
            .for_each(|((quant_block, data_block), &scale)| unsafe {
                self.quantize_neon_block(data_block, quant_block, scale, bits);
            });

        Ok(quantized)
    }

    /// NEON-optimized dequantization for ARM64
    #[cfg(target_arch = "aarch64")]
    fn dequantize_neon(
        &self,
        quantized: &[i8],
        scales: &[f32],
        block_size: usize,
    ) -> Result<Vec<f32>> {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return self.dequantize_scalar(quantized, scales, block_size);
        }

        let mut dequantized = vec![0.0f32; quantized.len()];

        dequantized
            .par_chunks_mut(block_size)
            .zip(quantized.par_chunks(block_size))
            .zip(scales.par_iter())
            .for_each(|((dequant_block, quant_block), &scale)| unsafe {
                self.dequantize_neon_block(quant_block, dequant_block, scale);
            });

        Ok(dequantized)
    }

    /// AVX2 kernel for quantizing a single block
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn quantize_avx2_block(&self, data: &[f32], output: &mut [i8], scale: f32, bits: u8) {
        #[allow(clippy::wildcard_imports)]
        use std::arch::x86_64::*;

        let max_val = ((1 << (bits - 1)) - 1) as f32;
        let min_val = -(1 << (bits - 1)) as f32;

        let inv_scale = if scale.is_finite() && scale != 0.0 { 1.0 / scale } else { 0.0 };
        let inv_scale_vec = _mm256_set1_ps(inv_scale);
        let min_val_vec = _mm256_set1_ps(min_val);
        let max_val_vec = _mm256_set1_ps(max_val);

        let chunks = data.chunks_exact(8);
        let remainder = chunks.remainder();

        for (i, chunk) in chunks.enumerate() {
            unsafe {
                let data_vec = _mm256_loadu_ps(chunk.as_ptr());
                let scaled = _mm256_mul_ps(data_vec, inv_scale_vec);
                let rounded = _mm256_round_ps::<_MM_FROUND_TO_NEAREST_INT>(scaled);
                let clamped = _mm256_max_ps(_mm256_min_ps(rounded, max_val_vec), min_val_vec);

                // Convert to i32 and then to i8
                let i32_vec = _mm256_cvtps_epi32(clamped);
                let i16_vec = _mm256_packs_epi32(i32_vec, i32_vec);
                let i8_vec = _mm256_packs_epi16(i16_vec, i16_vec);

                // Store 8 bytes efficiently
                let result = _mm256_extract_epi64::<0>(i8_vec);
                std::ptr::write_unaligned(output.as_mut_ptr().add(i * 8) as *mut i64, result);
            }
        }

        // Handle remainder with scalar code
        for (i, &value) in remainder.iter().enumerate() {
            let idx = data.len() - remainder.len() + i;
            output[idx] = quantize_value_scalar(value, scale, bits);
        }
    }

    /// AVX2 kernel for dequantizing a single block
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn dequantize_avx2_block(&self, quantized: &[i8], output: &mut [f32], scale: f32) {
        #[allow(clippy::wildcard_imports)]
        use std::arch::x86_64::*;

        let scale_vec = _mm256_set1_ps(scale);

        let chunks = quantized.chunks_exact(8);
        let remainder = chunks.remainder();

        for (i, chunk) in chunks.enumerate() {
            unsafe {
                // Load 8 i8 values
                let i8_vec = _mm_loadu_si64(chunk.as_ptr() as *const u8);

                // Convert to i32 and then to f32
                let i32_vec = _mm256_cvtepi8_epi32(i8_vec);
                let f32_vec = _mm256_cvtepi32_ps(i32_vec);
                let result = _mm256_mul_ps(f32_vec, scale_vec);

                _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), result);
            }
        }

        // Handle remainder with scalar code
        for (i, &value) in remainder.iter().enumerate() {
            let idx = quantized.len() - remainder.len() + i;
            output[idx] = dequantize_value_scalar(value, scale);
        }
    }

    /// NEON kernel for quantizing a single block
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn quantize_neon_block(&self, data: &[f32], output: &mut [i8], scale: f32, bits: u8) {
        use std::arch::aarch64::*;

        let max_val = ((1 << (bits - 1)) - 1) as f32;
        let min_val = -(1 << (bits - 1)) as f32;

        let inv_scale = if scale.is_finite() && scale != 0.0 { 1.0 / scale } else { 0.0 };
        let inv_scale_vec = vdupq_n_f32(inv_scale);
        let min_val_vec = vdupq_n_f32(min_val);
        let max_val_vec = vdupq_n_f32(max_val);

        let chunks = data.chunks_exact(4);
        let remainder = chunks.remainder();

        for (i, chunk) in chunks.enumerate() {
            let data_vec = vld1q_f32(chunk.as_ptr());
            let scaled = vmulq_f32(data_vec, inv_scale_vec);
            let rounded = vrndnq_f32(scaled);
            let clamped = vmaxq_f32(vminq_f32(rounded, max_val_vec), min_val_vec);

            // Convert to i32 and then to i8
            let i32_vec = vcvtq_s32_f32(clamped);
            let i16_vec = vqmovn_s32(i32_vec);
            let i8_vec = vqmovn_s16(vcombine_s16(i16_vec, i16_vec));

            // Store 4 bytes
            let result = vget_lane_u32::<0>(vreinterpret_u32_s8(i8_vec));
            std::ptr::copy_nonoverlapping(
                &result as *const u32 as *const i8,
                output.as_mut_ptr().add(i * 4),
                4,
            );
        }

        // Handle remainder with scalar code
        for (i, &value) in remainder.iter().enumerate() {
            let idx = data.len() - remainder.len() + i;
            output[idx] = quantize_value_scalar(value, scale, bits);
        }
    }

    /// NEON kernel for dequantizing a single block
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn dequantize_neon_block(&self, quantized: &[i8], output: &mut [f32], scale: f32) {
        use std::arch::aarch64::*;

        let scale_vec = vdupq_n_f32(scale);

        let chunks = quantized.chunks_exact(4);
        let remainder = chunks.remainder();

        for (i, chunk) in chunks.enumerate() {
            // Load 4 i8 values
            let i8_vec = vreinterpret_s8_u32(vld1_dup_u32(chunk.as_ptr() as *const u32));

            // Convert to i32 and then to f32
            let i16_vec = vmovl_s8(i8_vec);
            let i32_vec = vmovl_s16(vget_low_s16(i16_vec));
            let f32_vec = vcvtq_f32_s32(i32_vec);
            let result = vmulq_f32(f32_vec, scale_vec);

            vst1q_f32(output.as_mut_ptr().add(i * 4), result);
        }

        // Handle remainder with scalar code
        for (i, &value) in remainder.iter().enumerate() {
            let idx = quantized.len() - remainder.len() + i;
            output[idx] = dequantize_value_scalar(value, scale);
        }
    }
}

impl Default for QuantizationKernels {
    fn default() -> Self {
        Self::new()
    }
}

/// Scalar quantization of a single value with numerical stability
#[inline]
fn quantize_value_scalar(value: f32, scale: f32, bits: u8) -> i8 {
    let max_val = (1 << (bits - 1)) - 1;
    let min_val = -(1 << (bits - 1));

    // Fast path for typical values
    if value.is_finite() && scale != 0.0 && scale.is_finite() {
        let normalized = value / scale;
        let quantized = normalized.round() as i32;
        return quantized.clamp(min_val, max_val) as i8;
    }

    // Fallback for edge cases
    0i8
}

/// Scalar dequantization of a single value with numerical stability
#[inline]
fn dequantize_value_scalar(quantized: i8, scale: f32) -> f32 {
    // Fast path for typical values
    if scale.is_finite() {
        quantized as f32 * scale
    } else {
        0.0 // Safe fallback for invalid scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_capabilities_detection() {
        let caps = SimdCapabilities::detect();
        let strategy = caps.best_quantization_strategy();

        // Should always have a valid strategy
        assert!(matches!(
            strategy,
            QuantizationStrategy::Scalar
                | QuantizationStrategy::SSE4_1
                | QuantizationStrategy::AVX2
                | QuantizationStrategy::AVX512
                | QuantizationStrategy::NEON
        ));
    }

    #[test]
    fn test_optimal_block_size() {
        let caps = SimdCapabilities::detect();
        let block_size = caps.optimal_block_size();

        // Should be a reasonable block size
        assert!(block_size >= 32);
        assert!(block_size <= 256);
        assert!(block_size.is_power_of_two());
    }

    #[test]
    fn test_quantization_kernels() {
        let kernels = QuantizationKernels::new();
        let data = vec![1.0, -2.0, 0.5, -0.5, 3.0, -1.5];
        let scales = vec![2.0, 2.0];
        let block_size = 3;
        let bits = 2;

        let quantized = kernels.quantize_scalar(&data, &scales, block_size, bits).unwrap();
        assert_eq!(quantized.len(), data.len());

        let dequantized = kernels.dequantize_scalar(&quantized, &scales, block_size).unwrap();
        assert_eq!(dequantized.len(), data.len());
    }

    #[test]
    fn test_simd_fallback() {
        let kernels = QuantizationKernels::new();
        let data = vec![1.0, -2.0, 0.5, -0.5];
        let scales = vec![2.0];
        let block_size = 4;
        let bits = 2;

        // Should work regardless of SIMD availability
        let quantized = kernels.quantize_simd(&data, &scales, block_size, bits).unwrap();
        assert_eq!(quantized.len(), data.len());

        let dequantized = kernels.dequantize_simd(&quantized, &scales, block_size).unwrap();
        assert_eq!(dequantized.len(), data.len());
    }

    #[test]
    fn test_scalar_quantize_dequantize_value() {
        let value = 1.0f32;
        let scale = 1.0f32;
        let bits = 2;

        let quantized = quantize_value_scalar(value, scale, bits);
        let dequantized = dequantize_value_scalar(quantized, scale);

        // 2-bit quantization has limited precision
        assert!((-2..=1).contains(&quantized)); // 2-bit signed range
        assert!(dequantized.abs() <= 2.0); // Should be in reasonable range
    }
}
