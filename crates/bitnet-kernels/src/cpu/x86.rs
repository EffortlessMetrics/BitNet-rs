//! x86 AVX2/AVX-512 optimized kernel implementations
//!
//! This module provides high-performance kernel implementations optimized for
//! x86_64 architectures using AVX2 and AVX-512 SIMD instructions. These kernels
//! are specifically tuned for TL2 quantization which is optimized for x86 platforms.
#![allow(clippy::needless_range_loop)] // Performance-critical loops with explicit indexing

use crate::KernelProvider;
use bitnet_common::{BitNetError, KernelError, QuantizationType, Result};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2 optimized kernel for x86_64 architectures
///
/// This kernel leverages AVX2 SIMD instructions for vectorized operations,
/// providing significant performance improvements over the fallback implementation.
/// It's specifically optimized for TL2 quantization patterns.
///
/// Performance characteristics:
/// - Matrix multiplication: Vectorized using AVX2 with 8x float32 operations per instruction
/// - TL2 quantization: Optimized lookup table generation and vectorized processing
/// - Memory access: Optimized for x86 cache hierarchy and memory bandwidth
///
/// Requirements:
/// - x86_64 architecture with AVX2 support
/// - Target feature "avx2" must be available at runtime
#[cfg(target_arch = "x86_64")]
pub struct Avx2Kernel;

#[cfg(target_arch = "x86_64")]
impl KernelProvider for Avx2Kernel {
    fn name(&self) -> &'static str {
        "avx2"
    }

    fn is_available(&self) -> bool {
        is_x86_feature_detected!("avx2")
    }

    fn matmul_i2s(
        &self,
        a: &[i8],
        b: &[u8],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        // Validate input dimensions
        if a.len() != m * k {
            return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: format!("Matrix A dimension mismatch: expected {}, got {}", m * k, a.len()),
            }));
        }
        if b.len() != k * n {
            return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: format!("Matrix B dimension mismatch: expected {}, got {}", k * n, b.len()),
            }));
        }
        if c.len() != m * n {
            return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: format!("Matrix C dimension mismatch: expected {}, got {}", m * n, c.len()),
            }));
        }

        // Use AVX2 optimized implementation
        unsafe { self.matmul_i2s_avx2(a, b, c, m, n, k) }
    }

    fn quantize(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
        qtype: QuantizationType,
    ) -> Result<()> {
        match qtype {
            QuantizationType::TL2 => unsafe { self.quantize_tl2_avx2(input, output, scales) },
            QuantizationType::I2S => unsafe { self.quantize_i2s_avx2(input, output, scales) },
            QuantizationType::TL1 => {
                // TL1 is optimized for ARM, fall back to basic implementation
                self.quantize_tl1_fallback(input, output, scales)
            }
        }
    }
}

// AVX-512 kernel is disabled due to unstable Rust features
// TODO: Re-enable when AVX-512 intrinsics are stabilized

#[cfg(target_arch = "x86_64")]
impl Avx2Kernel {
    /// AVX2 optimized matrix multiplication for i8 x u8 -> f32
    #[target_feature(enable = "avx2")]
    unsafe fn matmul_i2s_avx2(
        &self,
        a: &[i8],
        b: &[u8],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        // Initialize output matrix
        c.fill(0.0);

        for i in 0..m {
            for j in 0..n {
                let mut sum_vec = _mm256_setzero_si256();
                let mut kk = 0;

                while kk + 32 <= k {
                    let a_ptr = unsafe { a.as_ptr().add(i * k + kk) };
                    let a_lo =
                        _mm256_cvtepi8_epi16(unsafe { _mm_loadu_si128(a_ptr as *const __m128i) });
                    let a_hi = _mm256_cvtepi8_epi16(unsafe {
                        _mm_loadu_si128(a_ptr.add(16) as *const __m128i)
                    });

                    let mut b_col = [0u8; 32];
                    for t in 0..32 {
                        b_col[t] = b[(kk + t) * n + j];
                    }
                    let b_ptr = b_col.as_ptr();
                    let b_lo =
                        _mm256_cvtepu8_epi16(unsafe { _mm_loadu_si128(b_ptr as *const __m128i) });
                    let b_hi = _mm256_cvtepu8_epi16(unsafe {
                        _mm_loadu_si128(b_ptr.add(16) as *const __m128i)
                    });

                    let prod_lo = _mm256_madd_epi16(a_lo, b_lo);
                    let prod_hi = _mm256_madd_epi16(a_hi, b_hi);
                    sum_vec = _mm256_add_epi32(sum_vec, prod_lo);
                    sum_vec = _mm256_add_epi32(sum_vec, prod_hi);
                    kk += 32;
                }

                let mut total = {
                    let sum128 = _mm_add_epi32(
                        _mm256_castsi256_si128(sum_vec),
                        _mm256_extracti128_si256(sum_vec, 1),
                    );
                    let sum64 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
                    let sum32 = _mm_add_epi32(sum64, _mm_srli_si128(sum64, 4));
                    _mm_cvtsi128_si32(sum32)
                };

                for t in kk..k {
                    total += (a[i * k + t] as i32) * (b[t * n + j] as i32);
                }

                c[i * n + j] = total as f32;
            }
        }

        Ok(())
    }

    /// AVX2 optimized TL2 quantization
    #[target_feature(enable = "avx2")]
    unsafe fn quantize_tl2_avx2(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
    ) -> Result<()> {
        const BLOCK_SIZE: usize = 128;
        let num_blocks = input.len().div_ceil(BLOCK_SIZE);

        if output.len() < input.len() / 4 {
            return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: format!(
                    "Output buffer too small for TL2: expected {}, got {}",
                    input.len() / 4,
                    output.len()
                ),
            }));
        }

        if scales.len() < num_blocks {
            return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: format!(
                    "Scales buffer too small: expected {}, got {}",
                    num_blocks,
                    scales.len()
                ),
            }));
        }

        // TL2 lookup table optimized for x86
        let lut = [-1.2f32, -0.4, 0.4, 1.2];
        let _lut_vec = unsafe { _mm256_loadu_ps(lut.as_ptr()) };

        for block_idx in 0..num_blocks {
            let start = block_idx * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(input.len());
            let block = &input[start..end];

            // Find maximum absolute value using AVX2
            let mut max_vec = _mm256_setzero_ps();
            let mut i = 0;

            // Process 8 elements at a time
            while i + 8 <= block.len() {
                let vals = unsafe { _mm256_loadu_ps(block.as_ptr().add(i)) };
                let abs_vals = _mm256_andnot_ps(_mm256_set1_ps(-0.0), vals); // abs using bit mask
                max_vec = _mm256_max_ps(max_vec, abs_vals);
                i += 8;
            }

            // Horizontal maximum
            let mut final_max = {
                let max_hi = _mm256_extractf128_ps(max_vec, 1);
                let max_lo = _mm256_castps256_ps128(max_vec);
                let max_quad = _mm_max_ps(max_hi, max_lo);
                let max_dual = _mm_max_ps(max_quad, _mm_movehl_ps(max_quad, max_quad));
                let max_single = _mm_max_ss(max_dual, _mm_shuffle_ps(max_dual, max_dual, 0x55));
                _mm_cvtss_f32(max_single)
            };

            // Handle remaining elements
            for &val in &block[i..] {
                final_max = final_max.max(val.abs());
            }

            let scale = if final_max > 1e-8 { final_max / 1.5 } else { 1.0 };
            scales[block_idx] = scale;
            let scale_vec = _mm256_set1_ps(scale);

            // Quantize block using vectorized lookup
            i = 0;
            while i + 8 <= block.len() {
                let vals = unsafe { _mm256_loadu_ps(block.as_ptr().add(i)) };
                let normalized = _mm256_div_ps(vals, scale_vec);

                // Find closest values in lookup table for each element
                let mut quantized = [0u8; 8];

                // Extract and quantize each element (could be further optimized)
                for j in 0..8 {
                    let val = if j < 4 {
                        _mm_cvtss_f32(_mm256_extractf128_ps(normalized, 0))
                    } else {
                        _mm_cvtss_f32(_mm256_extractf128_ps(normalized, 1))
                    };

                    let mut best_idx = 0;
                    let mut best_dist = (val - lut[0]).abs();

                    for (idx, &lut_val) in lut.iter().enumerate().skip(1) {
                        let dist = (val - lut_val).abs();
                        if dist < best_dist {
                            best_dist = dist;
                            best_idx = idx;
                        }
                    }
                    quantized[j] = best_idx as u8;
                }

                // Pack values into bytes (2 bits each, 4 values per byte)
                for chunk in quantized.chunks(4) {
                    let byte_idx = (start + i) / 4;
                    if byte_idx < output.len() {
                        let mut packed = 0u8;
                        for (bit_idx, &val) in chunk.iter().enumerate() {
                            packed |= val << (bit_idx * 2);
                        }
                        output[byte_idx] = packed;
                    }
                    i += 4;
                }
            }

            // Handle remaining elements
            for (j, &val) in block[i..].iter().enumerate() {
                let normalized = val / scale;
                let mut best_idx = 0;
                let mut best_dist = (normalized - lut[0]).abs();

                for (idx, &lut_val) in lut.iter().enumerate().skip(1) {
                    let dist = (normalized - lut_val).abs();
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = idx;
                    }
                }

                let byte_idx = (start + i + j) / 4;
                let bit_offset = ((start + i + j) % 4) * 2;

                if byte_idx < output.len() {
                    output[byte_idx] |= (best_idx as u8) << bit_offset;
                }
            }
        }

        Ok(())
    }

    /// AVX2 optimized I2_S quantization
    #[target_feature(enable = "avx2")]
    unsafe fn quantize_i2s_avx2(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
    ) -> Result<()> {
        const BLOCK_SIZE: usize = 32;
        let num_blocks = input.len().div_ceil(BLOCK_SIZE);

        if output.len() < input.len() / 4 {
            return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: format!(
                    "Output buffer too small for I2_S: expected {}, got {}",
                    input.len() / 4,
                    output.len()
                ),
            }));
        }

        if scales.len() < num_blocks {
            return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: format!(
                    "Scales buffer too small: expected {}, got {}",
                    num_blocks,
                    scales.len()
                ),
            }));
        }

        let threshold_pos = _mm256_set1_ps(0.5);
        let threshold_neg = _mm256_set1_ps(-0.5);

        for block_idx in 0..num_blocks {
            let start = block_idx * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(input.len());
            let block = &input[start..end];

            // Find maximum absolute value using AVX2
            let mut max_vec = _mm256_setzero_ps();
            let mut i = 0;

            while i + 8 <= block.len() {
                let vals = unsafe { _mm256_loadu_ps(block.as_ptr().add(i)) };
                let abs_vals = _mm256_andnot_ps(_mm256_set1_ps(-0.0), vals);
                max_vec = _mm256_max_ps(max_vec, abs_vals);
                i += 8;
            }

            // Horizontal maximum
            let mut final_max = {
                let max_hi = _mm256_extractf128_ps(max_vec, 1);
                let max_lo = _mm256_castps256_ps128(max_vec);
                let max_quad = _mm_max_ps(max_hi, max_lo);
                let max_dual = _mm_max_ps(max_quad, _mm_movehl_ps(max_quad, max_quad));
                let max_single = _mm_max_ss(max_dual, _mm_shuffle_ps(max_dual, max_dual, 0x55));
                _mm_cvtss_f32(max_single)
            };

            // Handle remaining elements
            for &val in &block[i..] {
                final_max = final_max.max(val.abs());
            }

            let scale = if final_max > 1e-8 { final_max / 1.5 } else { 1.0 };
            scales[block_idx] = scale;
            let scale_vec = _mm256_set1_ps(scale);

            // Quantize block
            i = 0;
            while i + 8 <= block.len() {
                let vals = unsafe { _mm256_loadu_ps(block.as_ptr().add(i)) };
                let normalized = _mm256_div_ps(vals, scale_vec);

                // Vectorized quantization to {-1, 0, 1}
                let _gt_pos = _mm256_cmp_ps(normalized, threshold_pos, _CMP_GT_OQ);
                let _lt_neg = _mm256_cmp_ps(normalized, threshold_neg, _CMP_LT_OQ);

                let mut quantized = [0u8; 8];
                for j in 0..8 {
                    let val = if j < 4 {
                        _mm_cvtss_f32(_mm256_extractf128_ps(normalized, 0))
                    } else {
                        _mm_cvtss_f32(_mm256_extractf128_ps(normalized, 1))
                    };

                    quantized[j] = if val > 0.5 {
                        1u8 // +1
                    } else if val < -0.5 {
                        3u8 // -1 (represented as 3 in 2-bit)
                    } else {
                        0u8 // 0
                    };
                }

                // Pack into output
                for chunk in quantized.chunks(4) {
                    let byte_idx = (start + i) / 4;
                    if byte_idx < output.len() {
                        let mut packed = 0u8;
                        for (bit_idx, &val) in chunk.iter().enumerate() {
                            packed |= val << (bit_idx * 2);
                        }
                        output[byte_idx] = packed;
                    }
                    i += 4;
                }
            }

            // Handle remaining elements
            for (j, &val) in block[i..].iter().enumerate() {
                let normalized = val / scale;
                let quantized = if normalized > 0.5 {
                    1u8
                } else if normalized < -0.5 {
                    3u8
                } else {
                    0u8
                };

                let byte_idx = (start + i + j) / 4;
                let bit_offset = ((start + i + j) % 4) * 2;

                if byte_idx < output.len() {
                    output[byte_idx] |= quantized << bit_offset;
                }
            }
        }

        Ok(())
    }

    /// Fallback implementation for TL1 (not optimized for x86)
    fn quantize_tl1_fallback(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
    ) -> Result<()> {
        // Use the same implementation as fallback kernel for TL1
        const BLOCK_SIZE: usize = 64;
        let num_blocks = input.len().div_ceil(BLOCK_SIZE);

        if output.len() < input.len() / 4 {
            return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: format!(
                    "Output buffer too small for TL1: expected {}, got {}",
                    input.len() / 4,
                    output.len()
                ),
            }));
        }

        if scales.len() < num_blocks {
            return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: format!(
                    "Scales buffer too small: expected {}, got {}",
                    num_blocks,
                    scales.len()
                ),
            }));
        }

        let lut = [-1.0f32, -0.33, 0.33, 1.0];

        for block_idx in 0..num_blocks {
            let start = block_idx * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(input.len());
            let block = &input[start..end];

            let max_val = block.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let scale = if max_val > 1e-8 { max_val / 1.5 } else { 1.0 };
            scales[block_idx] = scale;

            for (i, &val) in block.iter().enumerate() {
                let normalized = val / scale;
                let mut best_idx = 0;
                let mut best_dist = (normalized - lut[0]).abs();

                for (idx, &lut_val) in lut.iter().enumerate().skip(1) {
                    let dist = (normalized - lut_val).abs();
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = idx;
                    }
                }

                let byte_idx = (start + i) / 4;
                let bit_offset = ((start + i) % 4) * 2;

                if byte_idx < output.len() {
                    output[byte_idx] |= (best_idx as u8) << bit_offset;
                }
            }
        }

        Ok(())
    }
}

// AVX-512 implementation removed due to unstable Rust features

// Stub implementations for non-x86_64 architectures
#[cfg(not(target_arch = "x86_64"))]
pub struct Avx2Kernel;

#[cfg(not(target_arch = "x86_64"))]
impl KernelProvider for Avx2Kernel {
    fn name(&self) -> &'static str {
        "avx2"
    }

    fn is_available(&self) -> bool {
        false
    }

    fn matmul_i2s(
        &self,
        _a: &[i8],
        _b: &[u8],
        _c: &mut [f32],
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<()> {
        Err(BitNetError::Kernel(KernelError::UnsupportedArchitecture {
            arch: "AVX2 kernel not available on non-x86_64 architectures".to_string(),
        }))
    }

    fn quantize(
        &self,
        _input: &[f32],
        _output: &mut [u8],
        _scales: &mut [f32],
        _qtype: QuantizationType,
    ) -> Result<()> {
        Err(BitNetError::Kernel(KernelError::UnsupportedArchitecture {
            arch: "AVX2 kernel not available on non-x86_64 architectures".to_string(),
        }))
    }
}

// AVX-512 kernel removed due to unstable Rust features

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avx2_kernel_availability() {
        let kernel = Avx2Kernel;

        #[cfg(target_arch = "x86_64")]
        {
            // On x86_64, availability depends on runtime detection
            println!("AVX2 available: {}", kernel.is_available());
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            // On non-x86_64, should not be available
            assert!(!kernel.is_available());
        }

        assert_eq!(kernel.name(), "avx2");
    }

    // AVX-512 tests removed due to unstable Rust features

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_matmul_matches_fallback() {
        let kernel = Avx2Kernel;
        assert!(kernel.is_available(), "AVX2 not detected on this CPU");

        let fallback = crate::cpu::fallback::FallbackKernel;

        let a = vec![1i8, 2, 3, 4];
        let b = vec![5u8, 6, 7, 8];
        let mut c_avx = vec![0.0f32; 4];
        let mut c_ref = vec![0.0f32; 4];

        kernel.matmul_i2s(&a, &b, &mut c_avx, 2, 2, 2).unwrap();
        fallback.matmul_i2s(&a, &b, &mut c_ref, 2, 2, 2).unwrap();

        assert_eq!(c_avx, c_ref);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_quantize_tl2() {
        let kernel = Avx2Kernel;
        assert!(kernel.is_available(), "AVX2 not detected on this CPU");

        let input = [1.5, -1.0, 0.5, -0.5, 0.0, 2.0, -2.0, 0.1].repeat(16); // 128 elements
        let mut output = vec![0u8; 32]; // 128 values / 4 per byte = 32 bytes
        let mut scales = vec![0.0f32; 1]; // 128 values / 128 per block = 1 block

        kernel.quantize(&input, &mut output, &mut scales, QuantizationType::TL2).unwrap();

        assert!(scales[0] > 0.0);
        assert!(output.iter().any(|&x| x != 0));
    }
}
