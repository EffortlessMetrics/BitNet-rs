//! x86/x86_64 CPU kernels with AVX2/AVX-512 optimizations
#![allow(unsafe_op_in_unsafe_fn)]

use crate::{KernelProvider, cpu::fallback::FallbackKernel};
use bitnet_common::{BitNetError, KernelError, QuantizationType, Result};
use std::arch::x86_64::*;

/// AVX2 optimized CPU kernel for x86_64
///
/// Provides high-performance implementations using AVX2 SIMD instructions
/// for 256-bit vector operations.
pub struct Avx2Kernel;

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
        if !self.is_available() {
            return Err(BitNetError::Kernel(KernelError::UnsupportedHardware {
                required: "AVX2".to_string(),
                available: "none".to_string(),
            }));
        }

        // Safety: We checked AVX2 is available
        unsafe { self.matmul_i2s_avx2(a, b, c, m, n, k) }
    }

    fn quantize(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
        qtype: QuantizationType,
    ) -> Result<()> {
        if !self.is_available() {
            // Fall back to non-SIMD implementation
            return FallbackKernel.quantize(input, output, scales, qtype);
        }

        match qtype {
            QuantizationType::I2S => {
                // Use fallback for now - I2S quantization doesn't benefit much from SIMD
                FallbackKernel.quantize(input, output, scales, qtype)
            }
            QuantizationType::TL1 => {
                // Use fallback for now
                FallbackKernel.quantize(input, output, scales, qtype)
            }
            QuantizationType::TL2 => {
                // Safety: We checked AVX2 is available
                unsafe { self.quantize_tl2_avx2(input, output, scales) }
            }
        }
    }
}

// AVX-512 kernel removed for now due to unstable intrinsics
// pub struct Avx512Kernel;
//
// impl KernelProvider for Avx512Kernel {
//     fn name(&self) -> &'static str {
//         "avx512"
//     }
//
//     fn is_available(&self) -> bool {
//         is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl")
//     }
//
//     fn matmul_i2s(
//         &self,
//         _a: &[i8],
//         _b: &[u8],
//         _c: &mut [f32],
//         _m: usize,
//         _n: usize,
//         _k: usize,
//     ) -> Result<()> {
//         // TODO: Implement when AVX-512 intrinsics are stabilized
//         Err(BitNetError::Kernel(KernelError::NotImplemented))
//     }
//
//     fn quantize(
//         &self,
//         _input: &[f32],
//         _output: &mut [u8],
//         _scales: &mut [f32],
//         _qtype: QuantizationType,
//     ) -> Result<()> {
//         // TODO: Implement when AVX-512 intrinsics are stabilized
//         Err(BitNetError::Kernel(KernelError::NotImplemented))
//     }
// }
// TODO: Re-enable when AVX-512 intrinsics are stabilized

#[cfg(target_arch = "x86_64")]
impl Avx2Kernel {
    /// AVX2 optimized matrix multiplication for i8 x u8 -> f32
    ///
    /// # Algorithm
    /// Uses blocked matrix multiplication with AVX2 SIMD instructions:
    /// - Processes 8x8 blocks for optimal cache and register usage
    /// - Sign-extends i8 values and zero-extends u8 values to i16
    /// - Uses `_mm256_madd_epi16` for efficient multiply-accumulate
    /// - Maintains per-block floating point accumulators for accuracy
    ///
    /// # Correctness
    /// This implementation has been validated against the fallback kernel
    /// for various matrix sizes including edge cases. The key fix from the
    /// original implementation is proper sign extension of i8 values using
    /// `_mm256_cvtepi8_epi16` instead of incorrect unpacking operations.
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
        // Initialize output to zero
        c.fill(0.0);

        // Process in blocks optimized for AVX2
        const BLOCK_M: usize = 8;
        const BLOCK_N: usize = 8;
        const BLOCK_K: usize = 32;

        for i in (0..m).step_by(BLOCK_M) {
            for j in (0..n).step_by(BLOCK_N) {
                // Accumulator for 8x8 block (rows x cols)
                let mut acc = [[_mm256_setzero_ps(); BLOCK_N]; BLOCK_M];

                for l in (0..k).step_by(BLOCK_K) {
                    let k_end = (l + BLOCK_K).min(k);
                    let k_len = k_end - l;

                    // Process A matrix rows
                    for ii in 0..(BLOCK_M.min(m - i)) {
                        // Load A row (i8 values) - 32 bytes = 32 i8 values
                        let a_row = &a[(i + ii) * k + l..];
                        let a_vec = if k_len >= 32 {
                            unsafe { _mm256_loadu_si256(a_row.as_ptr() as *const __m256i) }
                        } else {
                            // Handle partial loads
                            let mut temp = [0i8; 32];
                            temp[..k_len].copy_from_slice(&a_row[..k_len]);
                            unsafe { _mm256_loadu_si256(temp.as_ptr() as *const __m256i) }
                        };

                        // Process B matrix columns
                        for jj in 0..(BLOCK_N.min(n - j)) {
                            // Load B column (u8 values)
                            let mut b_col = [0u8; 32];
                            for kk in 0..k_len {
                                if l + kk < k {
                                    b_col[kk] = b[(l + kk) * n + (j + jj)];
                                }
                            }
                            let b_vec =
                                unsafe { _mm256_loadu_si256(b_col.as_ptr() as *const __m256i) };

                            // Convert to i16 for multiplication
                            // For signed i8, we need sign extension - use cvtepi8_epi16
                            // Split into low and high 128-bit lanes first
                            let a_128_lo = _mm256_castsi256_si128(a_vec);
                            let a_128_hi = _mm256_extracti128_si256(a_vec, 1);
                            let b_128_lo = _mm256_castsi256_si128(b_vec);
                            let b_128_hi = _mm256_extracti128_si256(b_vec, 1);

                            // Sign-extend i8 to i16 for A (signed)
                            let a_lo = _mm256_cvtepi8_epi16(a_128_lo);
                            let a_hi = _mm256_cvtepi8_epi16(a_128_hi);

                            // Zero-extend u8 to i16 for B (unsigned)
                            let b_lo = _mm256_cvtepu8_epi16(b_128_lo);
                            let b_hi = _mm256_cvtepu8_epi16(b_128_hi);

                            // Multiply and accumulate
                            let prod_lo = _mm256_madd_epi16(a_lo, b_lo);
                            let prod_hi = _mm256_madd_epi16(a_hi, b_hi);

                            // Sum products
                            let sum = _mm256_add_epi32(prod_lo, prod_hi);

                            // Convert to float and add to accumulator
                            let sum_f32 = _mm256_cvtepi32_ps(sum);
                            acc[ii][jj] = _mm256_add_ps(acc[ii][jj], sum_f32);
                        }
                    }
                }

                // Store results
                for ii in 0..(BLOCK_M.min(m - i)) {
                    for jj in 0..(BLOCK_N.min(n - j)) {
                        // Horizontal sum of the vector
                        let sum_vec = acc[ii][jj];
                        let sum_hi = _mm256_extractf128_ps(sum_vec, 1);
                        let sum_lo = _mm256_castps256_ps128(sum_vec);
                        let sum_quad = _mm_add_ps(sum_hi, sum_lo);
                        let sum_dual = _mm_add_ps(sum_quad, _mm_movehl_ps(sum_quad, sum_quad));
                        let sum_single =
                            _mm_add_ss(sum_dual, _mm_shuffle_ps(sum_dual, sum_dual, 0x55));

                        c[(i + ii) * n + (j + jj)] += _mm_cvtss_f32(sum_single);
                    }
                }
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

        if input.len() % BLOCK_SIZE != 0 {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: format!(
                    "Input length {} must be divisible by block size {}",
                    input.len(),
                    BLOCK_SIZE
                ),
            }));
        }

        let n_blocks = input.len() / BLOCK_SIZE;
        if scales.len() != n_blocks {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: format!(
                    "Scales length {} must match number of blocks {}",
                    scales.len(),
                    n_blocks
                ),
            }));
        }

        if output.len() != input.len() / 4 {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: format!(
                    "Output length {} must be input length {} / 4",
                    output.len(),
                    input.len()
                ),
            }));
        }

        for (block_idx, scale_slot) in scales.iter_mut().enumerate().take(n_blocks) {
            let block_start = block_idx * BLOCK_SIZE;
            let block = &input[block_start..block_start + BLOCK_SIZE];

            // Find min and max using AVX2
            let mut min_vec = _mm256_set1_ps(f32::INFINITY);
            let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);

            for i in (0..BLOCK_SIZE).step_by(8) {
                let vals = _mm256_loadu_ps(&block[i]);
                min_vec = _mm256_min_ps(min_vec, vals);
                max_vec = _mm256_max_ps(max_vec, vals);
            }

            // Horizontal min/max
            let min = horizontal_min_f32(min_vec);
            let max = horizontal_max_f32(max_vec);

            let scale = (max - min) / 3.0;
            *scale_slot = scale;

            // Quantize the block
            let scale_recip = if scale != 0.0 { 1.0 / scale } else { 0.0 };
            let min_vec = _mm256_set1_ps(min);
            let scale_recip_vec = _mm256_set1_ps(scale_recip);

            let out_start = block_idx * (BLOCK_SIZE / 4);
            for i in (0..BLOCK_SIZE).step_by(32) {
                // Process 32 values at a time (8 output bytes)
                let mut packed = [0u8; 8];

                for j in 0..4 {
                    let vals = _mm256_loadu_ps(&block[i + j * 8]);
                    let normalized = _mm256_mul_ps(_mm256_sub_ps(vals, min_vec), scale_recip_vec);

                    // Convert to integer [0, 3] with clamping
                    let three = _mm256_set1_ps(3.0);
                    let zero = _mm256_setzero_ps();
                    let clamped = _mm256_min_ps(_mm256_max_ps(normalized, zero), three);

                    // Convert to integers
                    let quantized = _mm256_cvtps_epi32(clamped);

                    // Pack into 2-bit values
                    // We need to extract 8 integers and pack them into 2 bytes
                    let mut temp = [0u32; 8];
                    _mm256_storeu_si256(temp.as_mut_ptr() as *mut __m256i, quantized);

                    // Pack 4 values into 1 byte
                    packed[j * 2] = (temp[0] & 0x3) as u8
                        | ((temp[1] & 0x3) << 2) as u8
                        | ((temp[2] & 0x3) << 4) as u8
                        | ((temp[3] & 0x3) << 6) as u8;

                    packed[j * 2 + 1] = (temp[4] & 0x3) as u8
                        | ((temp[5] & 0x3) << 2) as u8
                        | ((temp[6] & 0x3) << 4) as u8
                        | ((temp[7] & 0x3) << 6) as u8;
                }

                // Copy packed bytes to output
                output[out_start + i / 4..out_start + i / 4 + 8].copy_from_slice(&packed);
            }
        }

        Ok(())
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn horizontal_min_f32(v: __m256) -> f32 {
    // Reduce to 128-bit
    let v128 = _mm_min_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    // Reduce to 64-bit
    let v64 = _mm_min_ps(v128, _mm_movehl_ps(v128, v128));
    // Reduce to 32-bit
    let v32 = _mm_min_ss(v64, _mm_shuffle_ps(v64, v64, 0x55));
    _mm_cvtss_f32(v32)
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn horizontal_max_f32(v: __m256) -> f32 {
    // Reduce to 128-bit
    let v128 = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    // Reduce to 64-bit
    let v64 = _mm_max_ps(v128, _mm_movehl_ps(v128, v128));
    // Reduce to 32-bit
    let v32 = _mm_max_ss(v64, _mm_shuffle_ps(v64, v64, 0x55));
    _mm_cvtss_f32(v32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_kernel_availability() {
        let kernel = Avx2Kernel;

        // This test will pass or fail depending on the CPU
        if is_x86_feature_detected!("avx2") {
            assert!(kernel.is_available());
        } else {
            assert!(!kernel.is_available());
        }

        assert_eq!(kernel.name(), "avx2");
    }

    // AVX-512 tests removed due to unstable Rust features

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_matmul_basic() {
        let kernel = Avx2Kernel;

        if !kernel.is_available() {
            return; // Skip test if AVX2 not available
        }

        // Test 2x2 * 2x2 matrix multiplication
        let a = vec![1i8, 2, 3, 4];
        let b = vec![1u8, 0, 0, 1];
        let mut c = vec![0.0f32; 4];

        kernel.matmul_i2s(&a, &b, &mut c, 2, 2, 2).unwrap();

        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_matmul_matches_fallback() {
        let avx2_kernel = Avx2Kernel;

        if !avx2_kernel.is_available() {
            return; // Skip test if AVX2 not available
        }

        let fallback_kernel = crate::cpu::fallback::FallbackKernel;

        // Test various matrix sizes to ensure correctness
        let test_cases = vec![
            (2, 2, 2),    // Small matrices
            (8, 8, 8),    // Block-aligned
            (16, 16, 16), // Multiple blocks
            (7, 9, 11),   // Non-aligned sizes
            (32, 32, 32), // Exact block size
            (33, 33, 33), // Just over block size
        ];

        for (m, n, k) in test_cases {
            // Generate test data with predictable values
            let mut a = vec![0i8; m * k];
            let mut b = vec![0u8; k * n];

            for (i, a_slot) in a.iter_mut().enumerate().take(m * k) {
                *a_slot = ((i % 5) as i8) - 2; // Values from -2 to 2
            }
            for (i, b_slot) in b.iter_mut().enumerate().take(k * n) {
                *b_slot = (i % 3) as u8; // Values from 0 to 2  
            }

            let mut c_avx2 = vec![0.0f32; m * n];
            let mut c_fallback = vec![0.0f32; m * n];

            // Compute with AVX2
            avx2_kernel.matmul_i2s(&a, &b, &mut c_avx2, m, n, k).unwrap();

            // Compute with fallback
            fallback_kernel.matmul_i2s(&a, &b, &mut c_fallback, m, n, k).unwrap();

            // Compare results with tolerance for floating point
            for i in 0..m * n {
                assert!(
                    (c_avx2[i] - c_fallback[i]).abs() < 1e-6,
                    "Mismatch at position {} for {}x{}x{} matrix: AVX2={}, Fallback={}",
                    i,
                    m,
                    n,
                    k,
                    c_avx2[i],
                    c_fallback[i]
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_quantize_tl2() {
        let kernel = Avx2Kernel;

        if !kernel.is_available() {
            return;
        }

        let input = [1.5, -1.0, 0.5, -0.5, 0.0, 2.0, -2.0, 0.1].repeat(16); // 128 elements
        let mut output = vec![0u8; 32]; // 128 values / 4 per byte = 32 bytes
        let mut scales = vec![0.0f32; 1]; // 128 values / 128 per block = 1 block

        kernel.quantize(&input, &mut output, &mut scales, QuantizationType::TL2).unwrap();

        assert!(scales[0] > 0.0);
        assert!(output.iter().any(|&x| x != 0));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_tl2_validation() {
        let kernel = Avx2Kernel;

        if !kernel.is_available() {
            return;
        }

        let fallback = FallbackKernel;

        // Create test input with 256 elements (2 blocks)
        let mut input = vec![0.0f32; 256];
        for (i, input_slot) in input.iter_mut().enumerate().take(256) {
            *input_slot = ((i as f32) / 10.0).sin() * 5.0;
        }

        let mut output_avx = vec![0u8; 64];
        let mut output_fb = vec![0u8; 64];
        let mut scales_avx = vec![0.0f32; 2];
        let mut scales_fb = vec![0.0f32; 2];

        kernel.quantize(&input, &mut output_avx, &mut scales_avx, QuantizationType::TL2).unwrap();
        fallback.quantize(&input, &mut output_fb, &mut scales_fb, QuantizationType::TL2).unwrap();

        // Scales should be very similar
        for i in 0..2 {
            assert!(
                (scales_avx[i] - scales_fb[i]).abs() < 0.01,
                "Scale mismatch at block {}: AVX2={}, Fallback={}",
                i,
                scales_avx[i],
                scales_fb[i]
            );
        }

        // Output might differ slightly due to rounding, but should be mostly the same
        let mut diff_count = 0;
        for i in 0..64 {
            if output_avx[i] != output_fb[i] {
                diff_count += 1;
            }
        }
        assert!(diff_count < 10, "Too many differences in quantized output: {}/64", diff_count);
    }
}
