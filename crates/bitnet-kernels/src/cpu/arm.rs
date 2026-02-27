//! ARM NEON optimized kernel implementations
//!
//! This module provides high-performance kernel implementations optimized for
//! ARM64 architectures using NEON SIMD instructions. These kernels are specifically
//! tuned for TL1 quantization which is optimized for ARM platforms.

use crate::KernelProvider;
use bitnet_common::{BitNetError, KernelError, QuantizationType, Result};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[inline]
fn packed_2bit_bytes(len: usize) -> usize {
    len.div_ceil(4)
}

/// NEON optimized kernel for ARM64 architectures
///
/// This kernel leverages ARM NEON SIMD instructions for vectorized operations,
/// providing significant performance improvements over the fallback implementation.
/// It's specifically optimized for TL1 quantization patterns.
///
/// Performance characteristics:
/// - Matrix multiplication: Vectorized using NEON with 4x float32 operations per instruction
/// - TL1 quantization: Optimized lookup table generation and vectorized processing
/// - Memory access: Optimized for ARM cache hierarchy and memory bandwidth
///
/// Requirements:
/// - ARM64 architecture with NEON support
/// - Target feature "neon" must be available at runtime
#[cfg(target_arch = "aarch64")]
pub struct NeonKernel;

#[cfg(target_arch = "aarch64")]
impl KernelProvider for NeonKernel {
    fn name(&self) -> &'static str {
        "neon"
    }

    fn is_available(&self) -> bool {
        // NEON is mandatory on ARM64, but check for safety
        std::arch::is_aarch64_feature_detected!("neon")
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

        // Use NEON optimized implementation
        unsafe { self.matmul_i2s_neon(a, b, c, m, n, k) }
    }

    fn quantize(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
        qtype: QuantizationType,
    ) -> Result<()> {
        match qtype {
            QuantizationType::TL1 => unsafe { self.quantize_tl1_neon(input, output, scales) },
            QuantizationType::I2S => unsafe { self.quantize_i2s_neon(input, output, scales) },
            QuantizationType::TL2 => {
                // TL2 is optimized for x86, fall back to basic implementation
                self.quantize_tl2_fallback(input, output, scales)
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl NeonKernel {
    /// NEON optimized matrix multiplication for i8 x u8 -> f32
    #[target_feature(enable = "neon")]
    unsafe fn matmul_i2s_neon(
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

        // Process in blocks optimized for NEON
        const BLOCK_M: usize = 4;
        const BLOCK_N: usize = 4;
        const BLOCK_K: usize = 16;

        for i in (0..m).step_by(BLOCK_M) {
            for j in (0..n).step_by(BLOCK_N) {
                // Accumulator for 4x4 block
                let mut acc = [vdupq_n_f32(0.0); 4];

                for l in (0..k).step_by(BLOCK_K) {
                    let k_end = (l + BLOCK_K).min(k);
                    let k_len = k_end - l;

                    // Load and process A matrix block
                    for ii in 0..(BLOCK_M.min(m - i)) {
                        if i + ii >= m {
                            break;
                        }

                        // Load A row (i8 values)
                        let a_row = &a[(i + ii) * k + l..];
                        let a_vec = if k_len >= 16 {
                            vld1q_s8(a_row.as_ptr())
                        } else {
                            // Handle partial loads
                            let mut temp = [0i8; 16];
                            temp[..k_len].copy_from_slice(&a_row[..k_len]);
                            vld1q_s8(temp.as_ptr())
                        };

                        // Process B matrix columns
                        for jj in 0..(BLOCK_N.min(n - j)) {
                            if j + jj >= n {
                                break;
                            }

                            // Load B column (u8 values)
                            let mut b_col = [0u8; 16];
                            for kk in 0..k_len {
                                if l + kk < k {
                                    b_col[kk] = b[(l + kk) * n + (j + jj)];
                                }
                            }
                            let b_vec = vld1q_u8(b_col.as_ptr());

                            // Convert to i16 for multiplication
                            let a_lo = vmovl_s8(vget_low_s8(a_vec));
                            let a_hi = vmovl_s8(vget_high_s8(a_vec));
                            let b_lo = vmovl_u8(vget_low_u8(b_vec));
                            let b_hi = vmovl_u8(vget_high_u8(b_vec));

                            // Multiply and accumulate (low part)
                            let prod_lo = vmull_s16(
                                vget_low_s16(a_lo),
                                vget_low_s16(vreinterpretq_s16_u16(b_lo)),
                            );
                            let prod_hi_lo = vmull_high_s16(a_lo, vreinterpretq_s16_u16(b_lo));

                            // Multiply and accumulate (high part)
                            let prod_lo_hi = vmull_s16(
                                vget_low_s16(a_hi),
                                vget_low_s16(vreinterpretq_s16_u16(b_hi)),
                            );
                            let prod_hi_hi = vmull_high_s16(a_hi, vreinterpretq_s16_u16(b_hi));

                            // Sum all products
                            let sum1 = vaddq_s32(prod_lo, prod_hi_lo);
                            let sum2 = vaddq_s32(prod_lo_hi, prod_hi_hi);
                            let total_sum = vaddq_s32(sum1, sum2);

                            // Convert to float and add to accumulator
                            let sum_f32 = vcvtq_f32_s32(total_sum);
                            acc[jj] = vaddq_f32(acc[jj], sum_f32);
                        }
                    }
                }

                // Store results
                for ii in 0..(BLOCK_M.min(m - i)) {
                    for jj in 0..(BLOCK_N.min(n - j)) {
                        if i + ii < m && j + jj < n {
                            // Sum the vector elements
                            let sum = vaddvq_f32(acc[jj]);
                            c[(i + ii) * n + (j + jj)] += sum;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// NEON optimized TL1 quantization
    #[target_feature(enable = "neon")]
    unsafe fn quantize_tl1_neon(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
    ) -> Result<()> {
        const BLOCK_SIZE: usize = 64;
        let num_blocks = (input.len() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let required_output_bytes = packed_2bit_bytes(input.len());

        if output.len() < required_output_bytes {
            return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: format!(
                    "Output buffer too small for TL1: expected {}, got {}",
                    required_output_bytes,
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

        output[..required_output_bytes].fill(0);

        // TL1 lookup table optimized for ARM
        let lut = [-1.0f32, -0.33, 0.33, 1.0];

        for block_idx in 0..num_blocks {
            let start = block_idx * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(input.len());
            let block = &input[start..end];

            // Find maximum absolute value using NEON
            let mut max_vec = vdupq_n_f32(0.0);
            let mut i = 0;

            // Process 4 elements at a time
            while i + 4 <= block.len() {
                let vals = vld1q_f32(block.as_ptr().add(i));
                let abs_vals = vabsq_f32(vals);
                max_vec = vmaxq_f32(max_vec, abs_vals);
                i += 4;
            }

            // Find maximum in the vector
            let max_val = vmaxvq_f32(max_vec);

            // Handle remaining elements
            let mut final_max = max_val;
            for &val in &block[i..] {
                final_max = final_max.max(val.abs());
            }

            let scale = if final_max > 1e-8 { final_max / 1.5 } else { 1.0 };
            scales[block_idx] = scale;
            let scale_vec = vdupq_n_f32(scale);

            // Quantize block using vectorized lookup
            i = 0;

            while i + 4 <= block.len() {
                let vals = vld1q_f32(block.as_ptr().add(i));
                let normalized = vdivq_f32(vals, scale_vec);

                // Find closest values in lookup table for each element
                let mut quantized = [0u8; 4];

                for j in 0..4 {
                    let val = vgetq_lane_f32(normalized, j);
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

                // Pack 4 values into one byte (2 bits each)
                let byte_idx = (start + i) / 4;
                if byte_idx < output.len() {
                    output[byte_idx] = quantized[0]
                        | (quantized[1] << 2)
                        | (quantized[2] << 4)
                        | (quantized[3] << 6);
                }

                i += 4;
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

    /// NEON optimized I2_S quantization
    #[target_feature(enable = "neon")]
    unsafe fn quantize_i2s_neon(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
    ) -> Result<()> {
        const BLOCK_SIZE: usize = 32;
        let num_blocks = (input.len() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let required_output_bytes = packed_2bit_bytes(input.len());

        if output.len() < required_output_bytes {
            return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: format!(
                    "Output buffer too small for I2_S: expected {}, got {}",
                    required_output_bytes,
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

        let threshold_pos = vdupq_n_f32(0.5);
        let threshold_neg = vdupq_n_f32(-0.5);
        output[..required_output_bytes].fill(0);

        for block_idx in 0..num_blocks {
            let start = block_idx * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(input.len());
            let block = &input[start..end];

            // Find maximum absolute value using NEON
            let mut max_vec = vdupq_n_f32(0.0);
            let mut i = 0;

            while i + 4 <= block.len() {
                let vals = vld1q_f32(block.as_ptr().add(i));
                let abs_vals = vabsq_f32(vals);
                max_vec = vmaxq_f32(max_vec, abs_vals);
                i += 4;
            }

            let max_val = vmaxvq_f32(max_vec);
            let mut final_max = max_val;

            // Handle remaining elements
            for &val in &block[i..] {
                final_max = final_max.max(val.abs());
            }

            let scale = if final_max > 1e-8 { final_max / 1.5 } else { 1.0 };
            scales[block_idx] = scale;
            let scale_vec = vdupq_n_f32(scale);

            // Quantize block
            i = 0;
            while i + 4 <= block.len() {
                let vals = vld1q_f32(block.as_ptr().add(i));
                let normalized = vdivq_f32(vals, scale_vec);

                // Vectorized quantization to {-1, 0, 1}
                let mut quantized = [0u8; 4];
                for j in 0..4 {
                    let val = vgetq_lane_f32(normalized, j);
                    quantized[j] = if val > 0.5 {
                        1u8 // +1
                    } else if val < -0.5 {
                        3u8 // -1 (represented as 3 in 2-bit)
                    } else {
                        0u8 // 0
                    };
                }

                // Pack into output
                let byte_idx = (start + i) / 4;
                if byte_idx < output.len() {
                    output[byte_idx] = quantized[0]
                        | (quantized[1] << 2)
                        | (quantized[2] << 4)
                        | (quantized[3] << 6);
                }

                i += 4;
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

    /// Fallback implementation for TL2 (not optimized for ARM)
    fn quantize_tl2_fallback(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
    ) -> Result<()> {
        // Use the same implementation as fallback kernel for TL2
        const BLOCK_SIZE: usize = 128;
        let num_blocks = (input.len() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let required_output_bytes = packed_2bit_bytes(input.len());

        if output.len() < required_output_bytes {
            return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
                reason: format!(
                    "Output buffer too small for TL2: expected {}, got {}",
                    required_output_bytes,
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

        let lut = [-1.2f32, -0.4, 0.4, 1.2];
        output[..required_output_bytes].fill(0);

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

// Stub implementation for non-ARM architectures
#[cfg(not(target_arch = "aarch64"))]
pub struct NeonKernel;

#[cfg(not(target_arch = "aarch64"))]
impl KernelProvider for NeonKernel {
    fn name(&self) -> &'static str {
        "neon"
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
            arch: "NEON kernel not available on non-ARM64 architectures".to_string(),
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
            arch: "NEON kernel not available on non-ARM64 architectures".to_string(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neon_kernel_availability() {
        let kernel = NeonKernel;

        #[cfg(target_arch = "aarch64")]
        {
            // On ARM64, availability depends on runtime detection
            println!("NEON available: {}", kernel.is_available());
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            // On non-ARM64, should not be available
            assert!(!kernel.is_available());
        }

        assert_eq!(kernel.name(), "neon");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_matmul_basic() {
        let kernel = NeonKernel;

        if !kernel.is_available() {
            return; // Skip test if NEON not available
        }

        // Test 2x2 * 2x2 matrix multiplication
        let a = vec![1i8, 2, 3, 4];
        let b = vec![1u8, 0, 0, 1];
        let mut c = vec![0.0f32; 4];

        kernel.matmul_i2s(&a, &b, &mut c, 2, 2, 2).unwrap();

        // Expected result: A * I = A (approximately, due to quantization)
        assert!((c[0] - 1.0).abs() < 0.1);
        assert!((c[1] - 2.0).abs() < 0.1);
        assert!((c[2] - 3.0).abs() < 0.1);
        assert!((c[3] - 4.0).abs() < 0.1);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_quantize_tl1() {
        let kernel = NeonKernel;

        if !kernel.is_available() {
            return;
        }

        let input = vec![1.5, -1.0, 0.5, -0.5, 0.0, 2.0, -2.0, 0.1; 64];
        let mut output = vec![0u8; 16]; // 64 values / 4 per byte = 16 bytes
        let mut scales = vec![0.0f32; 1]; // 64 values / 64 per block = 1 block

        kernel.quantize(&input, &mut output, &mut scales, QuantizationType::TL1).unwrap();

        assert!(scales[0] > 0.0);
        assert!(output.iter().any(|&x| x != 0));
    }
}
