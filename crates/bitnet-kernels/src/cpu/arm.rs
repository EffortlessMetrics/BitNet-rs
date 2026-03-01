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
        // Quantization writes bit-packed values using `|=` in several paths.
        // Always clear the destination first so callers do not need to.
        output.fill(0);

        match qtype {
            QuantizationType::TL1 => unsafe { self.quantize_tl1_neon(input, output, scales) },
            QuantizationType::I2S => unsafe { self.quantize_i2s_neon(input, output, scales) },
            QuantizationType::TL2 => unsafe { self.quantize_tl2_neon(input, output, scales) },
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
        c.fill(0.0);

        for i in 0..m {
            for j in 0..n {
                let mut l = 0usize;
                let mut acc0 = vdupq_n_s32(0);
                let mut acc1 = vdupq_n_s32(0);
                let mut acc2 = vdupq_n_s32(0);
                let mut acc3 = vdupq_n_s32(0);

                while l + 16 <= k {
                    let a_vec = vld1q_s8(a.as_ptr().add(i * k + l));

                    let mut b_col = [0u8; 16];
                    for kk in 0..16 {
                        b_col[kk] = b[(l + kk) * n + j];
                    }
                    let b_vec = vld1q_u8(b_col.as_ptr());

                    let a_lo = vmovl_s8(vget_low_s8(a_vec));
                    let a_hi = vmovl_s8(vget_high_s8(a_vec));
                    let b_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b_vec)));
                    let b_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(b_vec)));

                    acc0 = vaddq_s32(acc0, vmull_s16(vget_low_s16(a_lo), vget_low_s16(b_lo)));
                    acc1 = vaddq_s32(acc1, vmull_s16(vget_high_s16(a_lo), vget_high_s16(b_lo)));
                    acc2 = vaddq_s32(acc2, vmull_s16(vget_low_s16(a_hi), vget_low_s16(b_hi)));
                    acc3 = vaddq_s32(acc3, vmull_s16(vget_high_s16(a_hi), vget_high_s16(b_hi)));

                    l += 16;
                }

                let mut sum =
                    (vaddvq_s32(acc0) + vaddvq_s32(acc1) + vaddvq_s32(acc2) + vaddvq_s32(acc3))
                        as f32;

                while l < k {
                    sum += (a[i * k + l] as f32) * (b[l * n + j] as f32);
                    l += 1;
                }

                c[i * n + j] = sum;
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

                // Find closest values in lookup table for each element.
                // vgetq_lane_f32 requires a compile-time constant lane index,
                // so we extract all 4 lanes via vst1q_f32 instead.
                let mut lane_vals = [0.0f32; 4];
                vst1q_f32(lane_vals.as_mut_ptr(), normalized);
                let mut quantized = [0u8; 4];

                for j in 0..4 {
                    let val = lane_vals[j];
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
    ///
    /// Uses vectorized comparisons (`vcgtq_f32`/`vcltq_f32`) to classify all four
    /// lanes in a single pass instead of extracting each lane individually.
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
        // Codes: +1 → 1, -1 → 3, 0 → 0
        let code_pos = vdupq_n_u32(1);
        let code_neg = vdupq_n_u32(3);
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

            for &val in &block[i..] {
                final_max = final_max.max(val.abs());
            }

            let scale = if final_max > 1e-8 { final_max / 1.5 } else { 1.0 };
            scales[block_idx] = scale;
            let scale_vec = vdupq_n_f32(scale);

            // Vectorized quantization using NEON comparisons
            i = 0;
            while i + 4 <= block.len() {
                let vals = vld1q_f32(block.as_ptr().add(i));
                let normalized = vdivq_f32(vals, scale_vec);

                // Vectorized classification: compare all 4 lanes simultaneously
                let gt_pos = vcgtq_f32(normalized, threshold_pos); // mask: val > 0.5
                let lt_neg = vcltq_f32(normalized, threshold_neg); // mask: val < -0.5

                // Select codes: positive → 1, negative → 3, else → 0
                let codes = vorrq_u32(vandq_u32(gt_pos, code_pos), vandq_u32(lt_neg, code_neg));

                // Pack 4 codes into one byte (2 bits each)
                let byte_idx = (start + i) / 4;
                if byte_idx < output.len() {
                    let c0 = vgetq_lane_u32(codes, 0) as u8;
                    let c1 = vgetq_lane_u32(codes, 1) as u8;
                    let c2 = vgetq_lane_u32(codes, 2) as u8;
                    let c3 = vgetq_lane_u32(codes, 3) as u8;
                    output[byte_idx] = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6);
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

    /// NEON optimized TL2 quantization
    ///
    /// Uses NEON vectorized max-abs and division for scale computation,
    /// with boundary-based bucket classification instead of distance search.
    #[target_feature(enable = "neon")]
    unsafe fn quantize_tl2_neon(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
    ) -> Result<()> {
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

        // TL2 LUT: [-1.2, -0.4, 0.4, 1.2]
        // Bucket boundaries: < -0.8 → 0, [-0.8, 0) → 1, [0, 0.8) → 2, >= 0.8 → 3
        let bound_neg = vdupq_n_f32(-0.8);
        let bound_zero = vdupq_n_f32(0.0);
        let bound_pos = vdupq_n_f32(0.8);
        let one = vdupq_n_u32(1);
        output[..required_output_bytes].fill(0);

        for block_idx in 0..num_blocks {
            let start = block_idx * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(input.len());
            let block = &input[start..end];

            // Vectorized max-abs
            let mut max_vec = vdupq_n_f32(0.0);
            let mut i = 0;
            while i + 4 <= block.len() {
                let vals = vld1q_f32(block.as_ptr().add(i));
                max_vec = vmaxq_f32(max_vec, vabsq_f32(vals));
                i += 4;
            }
            let mut final_max = vmaxvq_f32(max_vec);
            for &val in &block[i..] {
                final_max = final_max.max(val.abs());
            }

            let scale = if final_max > 1e-8 { final_max / 1.5 } else { 1.0 };
            scales[block_idx] = scale;
            let scale_vec = vdupq_n_f32(scale);

            i = 0;
            while i + 4 <= block.len() {
                let vals = vld1q_f32(block.as_ptr().add(i));
                let normalized = vdivq_f32(vals, scale_vec);

                // Vectorized bucket classification using boundary comparisons:
                // code = (normalized >= -0.8) + (normalized >= 0.0) + (normalized >= 0.8)
                let ge_neg = vcgeq_f32(normalized, bound_neg);
                let ge_zero = vcgeq_f32(normalized, bound_zero);
                let ge_pos = vcgeq_f32(normalized, bound_pos);

                // Each mask is all-1s (0xFFFFFFFF) or 0; AND with 1 gives 0 or 1
                let codes = vaddq_u32(
                    vaddq_u32(vandq_u32(ge_neg, one), vandq_u32(ge_zero, one)),
                    vandq_u32(ge_pos, one),
                );

                let byte_idx = (start + i) / 4;
                if byte_idx < output.len() {
                    let c0 = vgetq_lane_u32(codes, 0) as u8;
                    let c1 = vgetq_lane_u32(codes, 1) as u8;
                    let c2 = vgetq_lane_u32(codes, 2) as u8;
                    let c3 = vgetq_lane_u32(codes, 3) as u8;
                    output[byte_idx] = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6);
                }

                i += 4;
            }

            // Scalar tail
            let lut = [-1.2f32, -0.4, 0.4, 1.2];
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

    /// Dequantize QK256 format data to f32 with NEON acceleration
    ///
    /// Provides the same interface as `Avx2Kernel::dequantize_qk256`, with a
    /// NEON fast path and scalar fallback for non-NEON builds.
    ///
    /// # Arguments
    /// * `quantized` - Packed 2-bit data (i8 slice, length ≈ total_elements / 4)
    /// * `scales` - Per-block scales (length = total_elements / 256)
    /// * `block_size` - Must be 256 for QK256 format
    pub fn dequantize_qk256(
        &self,
        quantized: &[i8],
        scales: &[f32],
        block_size: usize,
    ) -> Result<Vec<f32>> {
        if block_size != 256 {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: format!("QK256 requires block_size=256, got {}", block_size),
            }));
        }

        if !self.is_available() {
            return self.dequantize_qk256_scalar(quantized, scales, block_size);
        }

        // Safety: NEON availability checked above
        unsafe { self.dequantize_qk256_neon(quantized, scales, block_size) }
    }

    /// Scalar fallback for QK256 dequantization (exposed for testing)
    pub fn dequantize_qk256_scalar(
        &self,
        quantized: &[i8],
        scales: &[f32],
        block_size: usize,
    ) -> Result<Vec<f32>> {
        const QK256: usize = 256;
        const QK256_PACKED_BYTES: usize = 64;

        if block_size != QK256 {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: format!("QK256 requires block_size=256, got {}", block_size),
            }));
        }

        let total_elements = scales.len() * QK256;
        let expected_bytes = scales.len() * QK256_PACKED_BYTES;

        if quantized.len().abs_diff(expected_bytes) > 128 {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: format!(
                    "QK256 size mismatch: got {} bytes, expected {} for {} blocks",
                    quantized.len(),
                    expected_bytes,
                    scales.len()
                ),
            }));
        }

        let mut output = vec![0.0f32; total_elements];
        const LUT: [f32; 4] = [-2.0, -1.0, 1.0, 2.0];

        for (block_idx, &scale) in scales.iter().enumerate() {
            let block_start = block_idx * QK256;
            let packed_start = block_idx * QK256_PACKED_BYTES;
            let packed_end = (packed_start + QK256_PACKED_BYTES).min(quantized.len());

            for (i, &byte_val) in quantized[packed_start..packed_end].iter().enumerate() {
                let byte = byte_val as u8;
                let base = block_start + i * 4;

                if base < total_elements {
                    output[base] = LUT[(byte & 0x03) as usize] * scale;
                }
                if base + 1 < total_elements {
                    output[base + 1] = LUT[((byte >> 2) & 0x03) as usize] * scale;
                }
                if base + 2 < total_elements {
                    output[base + 2] = LUT[((byte >> 4) & 0x03) as usize] * scale;
                }
                if base + 3 < total_elements {
                    output[base + 3] = LUT[((byte >> 6) & 0x03) as usize] * scale;
                }
            }
        }

        Ok(output)
    }

    /// NEON-accelerated QK256 dequantization (2-bit → f32 with LUT)
    ///
    /// Unpacks 64 packed bytes → 256 2-bit codes per block, converts to f32
    /// via LUT, and applies per-block scale using NEON `vmulq_f32`.
    #[target_feature(enable = "neon")]
    unsafe fn dequantize_qk256_neon(
        &self,
        quantized: &[i8],
        scales: &[f32],
        block_size: usize,
    ) -> Result<Vec<f32>> {
        const QK256: usize = 256;
        const QK256_PACKED_BYTES: usize = 64;

        if block_size != QK256 {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: format!("QK256 dequantize requires block_size=256, got {}", block_size),
            }));
        }

        let total_elements = scales.len() * QK256;
        let expected_bytes = scales.len() * QK256_PACKED_BYTES;

        if quantized.len().abs_diff(expected_bytes) > 128 {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: format!(
                    "QK256 size mismatch: got {} bytes, expected {} for {} blocks",
                    quantized.len(),
                    expected_bytes,
                    scales.len()
                ),
            }));
        }

        let mut output = vec![0.0f32; total_elements];
        const LUT: [f32; 4] = [-2.0, -1.0, 1.0, 2.0];

        let mut codes = [0u8; QK256];

        for (block_idx, &scale) in scales.iter().enumerate() {
            let block_start = block_idx * QK256;
            let packed_start = block_idx * QK256_PACKED_BYTES;
            let packed_end = (packed_start + QK256_PACKED_BYTES).min(quantized.len());
            let packed_slice = &quantized[packed_start..packed_end];

            // Unpack 64 bytes → 256 2-bit codes
            for (i, &byte_val) in packed_slice.iter().enumerate() {
                let byte = byte_val as u8;
                let base = i * 4;
                codes[base] = byte & 0x03;
                codes[base + 1] = (byte >> 2) & 0x03;
                codes[base + 2] = (byte >> 4) & 0x03;
                codes[base + 3] = (byte >> 6) & 0x03;
            }

            let scale_vec = vdupq_n_f32(scale);

            // Process 4 elements at a time with NEON
            let mut elem_idx = 0;
            while elem_idx + 4 <= QK256 {
                let weights = vld1q_f32(
                    [
                        LUT[codes[elem_idx] as usize],
                        LUT[codes[elem_idx + 1] as usize],
                        LUT[codes[elem_idx + 2] as usize],
                        LUT[codes[elem_idx + 3] as usize],
                    ]
                    .as_ptr(),
                );

                let scaled = vmulq_f32(weights, scale_vec);
                vst1q_f32(output.as_mut_ptr().add(block_start + elem_idx), scaled);

                elem_idx += 4;
            }

            // Scalar tail
            while elem_idx < QK256 && block_start + elem_idx < total_elements {
                output[block_start + elem_idx] = LUT[codes[elem_idx] as usize] * scale;
                elem_idx += 1;
            }
        }

        Ok(output)
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

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_dequantize_qk256_basic() {
        let kernel = NeonKernel;
        if !kernel.is_available() {
            return;
        }

        // 1 block of 256 elements = 64 packed bytes
        // All code 0b10 (code 2) → LUT[2] = 1.0, scaled by 2.0 → 2.0
        let quantized = vec![0xAAu8 as i8; 64]; // 0b10101010 → all code 2
        let scales = vec![2.0f32; 1];

        let result =
            kernel.dequantize_qk256(&quantized, &scales, 256).expect("dequantize should succeed");

        assert_eq!(result.len(), 256);
        for &val in &result {
            assert!((val - 2.0).abs() < 1e-6, "Expected 2.0 (LUT[2]=1.0 * scale=2.0), got {}", val);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_dequantize_qk256_matches_scalar() {
        let kernel = NeonKernel;
        if !kernel.is_available() {
            return;
        }

        // Mixed codes across 1 block
        let mut quantized = vec![0i8; 64];
        for (i, byte) in quantized.iter_mut().enumerate() {
            // Cycle through codes: 0,1,2,3,0,1,2,3,...
            *byte = (0x00 | (0x01 << 2) | (0x02 << 4) | (0x03 << 6)) as i8;
            let _ = i; // use i to suppress unused warning
        }
        let scales = vec![1.5f32; 1];

        let result_neon = kernel
            .dequantize_qk256(&quantized, &scales, 256)
            .expect("NEON dequantize should succeed");
        let result_scalar = kernel
            .dequantize_qk256_scalar(&quantized, &scales, 256)
            .expect("Scalar dequantize should succeed");

        assert_eq!(result_neon.len(), result_scalar.len());
        for (i, (n, s)) in result_neon.iter().zip(result_scalar.iter()).enumerate() {
            assert!((n - s).abs() < 1e-6, "Mismatch at index {}: neon={}, scalar={}", i, n, s);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_dequantize_qk256_invalid_block_size() {
        let kernel = NeonKernel;
        let quantized = vec![0i8; 64];
        let scales = vec![1.0f32; 1];
        let result = kernel.dequantize_qk256(&quantized, &scales, 128);
        assert!(result.is_err());
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_quantize_tl2() {
        let kernel = NeonKernel;
        if !kernel.is_available() {
            return;
        }

        let input: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 64.0).collect();
        let mut output = vec![0u8; 32]; // 128 values / 4 per byte
        let mut scales = vec![0.0f32; 1]; // 128 / 128 = 1 block

        kernel.quantize(&input, &mut output, &mut scales, QuantizationType::TL2).unwrap();

        assert!(scales[0] > 0.0);
        assert!(output.iter().any(|&x| x != 0));
    }
}
