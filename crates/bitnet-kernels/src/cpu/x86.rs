//! x86/x86_64 CPU kernels with AVX2/AVX-512 optimizations
#![allow(unsafe_op_in_unsafe_fn)]

use crate::{KernelProvider, cpu::fallback::FallbackKernel};
use bitnet_common::{BitNetError, KernelError, QuantizationType, Result};
#[allow(clippy::wildcard_imports)]
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

impl Avx2Kernel {
    /// Dequantize QK256 format data to f32 with AVX2 acceleration
    ///
    /// This is a public utility method for QK256 dequantization with runtime dispatch.
    /// Falls back to scalar if AVX2 is not available.
    ///
    /// # Arguments
    /// * `quantized` - Packed 2-bit data (i8 slice, length = total_elements.div_ceil(4))
    /// * `scales` - Per-block scales (length = total_elements.div_ceil(256))
    /// * `block_size` - Must be 256 for QK256 format
    ///
    /// # Returns
    /// Vector of dequantized f32 values (length = scales.len() * 256)
    ///
    /// # Errors
    /// Returns error if block_size != 256 or dimension mismatches
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
            // Fall back to scalar implementation
            return self.dequantize_qk256_scalar(quantized, scales, block_size);
        }

        // Safety: We checked AVX2 is available
        unsafe { self.dequantize_qk256_avx2(quantized, scales, block_size) }
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
}

/// AVX-512 optimized CPU kernel for x86_64
///
/// Uses AVX-512BW and AVX-512F instructions for wide vector operations. This
/// backend provides higher throughput than the AVX2 implementation by
/// processing 64 bytes of data per iteration.
pub struct Avx512Kernel;

impl KernelProvider for Avx512Kernel {
    fn name(&self) -> &'static str {
        "avx512"
    }

    fn is_available(&self) -> bool {
        // Requires full width AVX-512 with byte/word support
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
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
                required: "AVX-512F+BW".to_string(),
                available: "none".to_string(),
            }));
        }

        // Safety: feature availability checked above
        unsafe { self.matmul_i2s_avx512(a, b, c, m, n, k) }
    }

    fn quantize(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
        qtype: QuantizationType,
    ) -> Result<()> {
        if !self.is_available() {
            return FallbackKernel.quantize(input, output, scales, qtype);
        }

        match qtype {
            QuantizationType::TL2 => unsafe { self.quantize_tl2_avx512(input, output, scales) },
            _ => FallbackKernel.quantize(input, output, scales, qtype),
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl Avx512Kernel {
    /// AVX-512 optimized matrix multiplication for i8 x u8 -> f32
    ///
    /// Processes 16x16 blocks and operates on 64 elements of the K dimension
    /// at a time using 512-bit SIMD instructions.
    #[target_feature(enable = "avx512f,avx512bw")]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn matmul_i2s_avx512(
        &self,
        a: &[i8],
        b: &[u8],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        c.fill(0.0);

        const BLOCK_M: usize = 16;
        const BLOCK_N: usize = 16;
        const BLOCK_K: usize = 64;

        for i in (0..m).step_by(BLOCK_M) {
            for j in (0..n).step_by(BLOCK_N) {
                let mut acc = [[_mm512_setzero_ps(); BLOCK_N]; BLOCK_M];

                for l in (0..k).step_by(BLOCK_K) {
                    let k_end = (l + BLOCK_K).min(k);
                    let k_len = k_end - l;

                    for ii in 0..(BLOCK_M.min(m - i)) {
                        let a_row = &a[(i + ii) * k + l..];
                        let a_vec = if k_len >= 64 {
                            _mm512_loadu_si512(a_row.as_ptr() as *const __m512i)
                        } else {
                            let mut temp = [0i8; 64];
                            temp[..k_len].copy_from_slice(&a_row[..k_len]);
                            _mm512_loadu_si512(temp.as_ptr() as *const __m512i)
                        };

                        for jj in 0..(BLOCK_N.min(n - j)) {
                            let mut b_col = [0u8; 64];
                            for kk in 0..k_len {
                                if l + kk < k {
                                    b_col[kk] = b[(l + kk) * n + (j + jj)];
                                }
                            }
                            let b_vec = _mm512_loadu_si512(b_col.as_ptr() as *const __m512i);

                            // Split into 256-bit lanes for extension
                            let a_lo256 = _mm512_castsi512_si256(a_vec);
                            let a_hi256 = _mm512_extracti64x4_epi64(a_vec, 1);
                            let b_lo256 = _mm512_castsi512_si256(b_vec);
                            let b_hi256 = _mm512_extracti64x4_epi64(b_vec, 1);

                            let a_lo = _mm512_cvtepi8_epi16(a_lo256);
                            let a_hi = _mm512_cvtepi8_epi16(a_hi256);
                            let b_lo = _mm512_cvtepu8_epi16(b_lo256);
                            let b_hi = _mm512_cvtepu8_epi16(b_hi256);

                            let prod_lo = _mm512_madd_epi16(a_lo, b_lo);
                            let prod_hi = _mm512_madd_epi16(a_hi, b_hi);
                            let sum = _mm512_add_epi32(prod_lo, prod_hi);

                            let sum_f32 = _mm512_cvtepi32_ps(sum);
                            acc[ii][jj] = _mm512_add_ps(acc[ii][jj], sum_f32);
                        }
                    }
                }

                for ii in 0..(BLOCK_M.min(m - i)) {
                    for jj in 0..(BLOCK_N.min(n - j)) {
                        let sum_vec = acc[ii][jj];
                        let total = _mm512_reduce_add_ps(sum_vec);
                        c[(i + ii) * n + (j + jj)] += total;
                    }
                }
            }
        }

        Ok(())
    }

    /// AVX-512 optimized TL2 quantization
    #[target_feature(enable = "avx512f")]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn quantize_tl2_avx512(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
    ) -> Result<()> {
        const BLOCK_SIZE: usize = 128;
        let num_blocks = input.len().div_ceil(BLOCK_SIZE);

        if output.len() < input.len() / 4 {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: format!(
                    "Output buffer too small for TL2: expected {}, got {}",
                    input.len() / 4,
                    output.len()
                ),
            }));
        }

        if scales.len() < num_blocks {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: format!(
                    "Scales buffer too small: expected {}, got {}",
                    num_blocks,
                    scales.len()
                ),
            }));
        }

        let lut = [-1.2f32, -0.4, 0.4, 1.2];

        for (block_idx, scale_slot) in scales.iter_mut().enumerate().take(num_blocks) {
            let start = block_idx * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(input.len());
            let block = &input[start..end];

            let mut max_abs_vec = _mm512_setzero_ps();
            for i in (0..block.len()).step_by(16) {
                let vals = if i + 16 <= block.len() {
                    _mm512_loadu_ps(block[i..].as_ptr())
                } else {
                    let mut temp = [0.0f32; 16];
                    temp[..block.len() - i].copy_from_slice(&block[i..]);
                    _mm512_loadu_ps(temp.as_ptr())
                };
                let abs_vals = _mm512_max_ps(vals, _mm512_sub_ps(_mm512_setzero_ps(), vals));
                max_abs_vec = _mm512_max_ps(max_abs_vec, abs_vals);
            }

            let max_val = _mm512_reduce_max_ps(max_abs_vec);
            *scale_slot = if max_val > 1e-8 { max_val / 1.5 } else { 1.0 };

            for (i, &val) in block.iter().enumerate() {
                let normalized = val / *scale_slot;
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
    #[allow(unsafe_op_in_unsafe_fn)]
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

    /// AVX2-optimized QK256 dequantization (2-bit → f32 with LUT)
    ///
    /// Dequantizes QK256 quantized data using AVX2 SIMD instructions.
    /// Processes 256-element blocks using LUT-based unpacking and widening.
    ///
    /// # Algorithm
    /// - Unpack 64 bytes → 256 2-bit codes (scalar for correctness)
    /// - Convert codes → f32 using SIMD LUT indexing
    /// - Apply scales with AVX2 FMA
    ///
    /// # Arguments
    /// * `quantized` - Packed 2-bit data (length = total_elements.div_ceil(4))
    /// * `scales` - Per-block scales (length = total_elements.div_ceil(256))
    /// * `block_size` - Must be 256 for QK256 format
    ///
    /// # Returns
    /// Vector of dequantized f32 values
    ///
    /// # Errors
    /// Returns error if block_size != 256 or dimension mismatches
    #[target_feature(enable = "avx2")]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn dequantize_qk256_avx2(
        &self,
        quantized: &[i8],
        scales: &[f32],
        block_size: usize,
    ) -> Result<Vec<f32>> {
        const QK256: usize = 256;
        const QK256_PACKED_BYTES: usize = 64;

        // Validate block size
        if block_size != QK256 {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: format!("QK256 dequantize requires block_size=256, got {}", block_size),
            }));
        }

        // Calculate dimensions
        let total_elements = scales.len() * QK256;
        let expected_bytes = scales.len() * QK256_PACKED_BYTES;

        // Validate input size (allow some tolerance for alignment)
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

        // Allocate output buffer
        let mut output = vec![0.0f32; total_elements];

        // Code-to-float LUT (verified against GGML reference)
        const LUT: [f32; 4] = [-2.0, -1.0, 1.0, 2.0];

        // Scratch buffer for unpacking (stack-allocated per block)
        let mut codes = [0u8; QK256];

        // Process each block
        for (block_idx, scale) in scales.iter().enumerate() {
            let block_start = block_idx * QK256;
            let packed_start = block_idx * QK256_PACKED_BYTES;

            // Get packed bytes for this block (handle partial last block)
            let packed_end = (packed_start + QK256_PACKED_BYTES).min(quantized.len());
            let packed_slice = &quantized[packed_start..packed_end];

            // Convert i8 to u8 for unpacking (reinterpret as unsigned)
            let mut packed_bytes = [0u8; QK256_PACKED_BYTES];
            for (i, &val) in packed_slice.iter().enumerate() {
                packed_bytes[i] = val as u8;
            }

            // Unpack 64 bytes → 256 2-bit codes (scalar for correctness)
            for (i, &byte) in packed_bytes.iter().enumerate() {
                let base = i * 4;
                codes[base] = byte & 0x03;
                codes[base + 1] = (byte >> 2) & 0x03;
                codes[base + 2] = (byte >> 4) & 0x03;
                codes[base + 3] = (byte >> 6) & 0x03;
            }

            // SIMD conversion: codes → f32 using LUT, then scale
            let scale_vec = _mm256_set1_ps(*scale);

            let mut elem_idx = 0;
            // Process 8 elements at a time with AVX2
            while elem_idx + 8 <= QK256 {
                // Convert 8 codes to weights using LUT
                let weights = [
                    LUT[codes[elem_idx] as usize],
                    LUT[codes[elem_idx + 1] as usize],
                    LUT[codes[elem_idx + 2] as usize],
                    LUT[codes[elem_idx + 3] as usize],
                    LUT[codes[elem_idx + 4] as usize],
                    LUT[codes[elem_idx + 5] as usize],
                    LUT[codes[elem_idx + 6] as usize],
                    LUT[codes[elem_idx + 7] as usize],
                ];

                // Load weights as AVX2 vector
                let w_vec = _mm256_loadu_ps(weights.as_ptr());

                // Apply scale: output = weights * scale
                let scaled = _mm256_mul_ps(w_vec, scale_vec);

                // Store result
                let out_ptr = output.as_mut_ptr().add(block_start + elem_idx);
                _mm256_storeu_ps(out_ptr, scaled);

                elem_idx += 8;
            }

            // Handle tail elements (scalar path)
            while elem_idx < QK256 && block_start + elem_idx < total_elements {
                let w = LUT[codes[elem_idx] as usize];
                output[block_start + elem_idx] = w * scale;
                elem_idx += 1;
            }
        }

        Ok(output)
    }

    /// AVX2 optimized TL2 quantization
    ///
    /// This implementation matches the fallback TL2 algorithm exactly to ensure
    /// cross-validation compatibility. Uses lookup table approach with max absolute
    /// value scaling like the fallback implementation.
    #[target_feature(enable = "avx2")]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn quantize_tl2_avx2(
        &self,
        input: &[f32],
        output: &mut [u8],
        scales: &mut [f32],
    ) -> Result<()> {
        const BLOCK_SIZE: usize = 128;
        let num_blocks = input.len().div_ceil(BLOCK_SIZE);

        if output.len() < input.len() / 4 {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: format!(
                    "Output buffer too small for TL2: expected {}, got {}",
                    input.len() / 4,
                    output.len()
                ),
            }));
        }

        if scales.len() < num_blocks {
            return Err(BitNetError::Kernel(KernelError::InvalidArguments {
                reason: format!(
                    "Scales buffer too small: expected {}, got {}",
                    num_blocks,
                    scales.len()
                ),
            }));
        }

        // Lookup table for x86 TL2 (same as fallback)
        let lut = [-1.2f32, -0.4, 0.4, 1.2];

        for (block_idx, scale_slot) in scales.iter_mut().enumerate().take(num_blocks) {
            let start = block_idx * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(input.len());
            let block = &input[start..end];

            // Find max absolute value using AVX2
            let mut max_abs_vec = _mm256_setzero_ps();

            for i in (0..block.len()).step_by(8) {
                let vals = if i + 8 <= block.len() {
                    _mm256_loadu_ps(&block[i])
                } else {
                    // Handle remainder with partial load
                    let mut temp = [0.0f32; 8];
                    temp[..block.len() - i].copy_from_slice(&block[i..]);
                    _mm256_loadu_ps(temp.as_ptr())
                };

                // Get absolute values
                let abs_vals = _mm256_max_ps(vals, _mm256_sub_ps(_mm256_setzero_ps(), vals));
                max_abs_vec = _mm256_max_ps(max_abs_vec, abs_vals);
            }

            // Horizontal max
            let max_val = unsafe { horizontal_max_f32(max_abs_vec) };

            // Compute scale (same as fallback)
            *scale_slot = if max_val > 1e-8 { max_val / 1.5 } else { 1.0 };

            // Quantize using lookup table approach
            for (i, &val) in block.iter().enumerate() {
                let normalized = val / *scale_slot;

                // Find closest value in lookup table (same as fallback)
                let mut best_idx = 0;
                let mut best_dist = (normalized - lut[0]).abs();

                for (idx, &lut_val) in lut.iter().enumerate().skip(1) {
                    let dist = (normalized - lut_val).abs();
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = idx;
                    }
                }

                // Pack into output (2 bits per value, same as fallback)
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse,avx2")]
#[allow(unsafe_op_in_unsafe_fn, dead_code)]
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
#[target_feature(enable = "sse,avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
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

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx512_kernel_availability() {
        let kernel = Avx512Kernel;

        // This test will pass or fail depending on the CPU
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            assert!(kernel.is_available());
        } else {
            assert!(!kernel.is_available());
        }

        assert_eq!(kernel.name(), "avx512");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx512_matmul_basic() {
        let kernel = Avx512Kernel;

        if !kernel.is_available() {
            return; // Skip test if AVX-512 not available
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
    fn test_avx512_vs_avx2_correctness() {
        let avx512_kernel = Avx512Kernel;
        let avx2_kernel = Avx2Kernel;

        if !avx512_kernel.is_available() || !avx2_kernel.is_available() {
            return; // Skip test if either kernel not available
        }

        // Test 8x8 * 8x8 matrix multiplication for better coverage
        let a = (0..64i32).map(|i| (i % 127) as i8).collect::<Vec<_>>();
        let b = (0..64i32).map(|i| (i % 255) as u8).collect::<Vec<_>>();
        let mut c_avx512 = vec![0.0f32; 64];
        let mut c_avx2 = vec![0.0f32; 64];

        avx512_kernel.matmul_i2s(&a, &b, &mut c_avx512, 8, 8, 8).unwrap();
        avx2_kernel.matmul_i2s(&a, &b, &mut c_avx2, 8, 8, 8).unwrap();

        // Results should be identical
        for i in 0..64 {
            assert_eq!(
                c_avx512[i], c_avx2[i],
                "Mismatch at position {}: AVX-512={}, AVX2={}",
                i, c_avx512[i], c_avx2[i]
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx512_matmul_matches_fallback_non_aligned_shapes() {
        let avx512_kernel = Avx512Kernel;
        if !avx512_kernel.is_available() {
            return;
        }

        let fallback_kernel = crate::cpu::fallback::FallbackKernel;

        let test_cases = vec![
            (3, 5, 7),
            (9, 13, 15),
            (17, 19, 33),
            (31, 27, 65),
        ];

        for (m, n, k) in test_cases {
            let mut a = vec![0i8; m * k];
            let mut b = vec![0u8; k * n];

            for (idx, item) in a.iter_mut().enumerate() {
                *item = ((idx % 11) as i8) - 5;
            }
            for (idx, item) in b.iter_mut().enumerate() {
                *item = ((idx * 3 + 7) % 251) as u8;
            }

            let mut c_avx512 = vec![0.0f32; m * n];
            let mut c_fallback = vec![0.0f32; m * n];

            avx512_kernel.matmul_i2s(&a, &b, &mut c_avx512, m, n, k).unwrap();
            fallback_kernel.matmul_i2s(&a, &b, &mut c_fallback, m, n, k).unwrap();

            for idx in 0..(m * n) {
                assert_eq!(
                    c_avx512[idx], c_fallback[idx],
                    "Mismatch for (m={m}, n={n}, k={k}) at index {idx}: avx512={}, fallback={}",
                    c_avx512[idx], c_fallback[idx]
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx512_quantize_tl2() {
        let kernel = Avx512Kernel;

        if !kernel.is_available() {
            return; // Skip test if AVX-512 not available
        }

        // Create test input with 128 elements (1 block)
        let mut input = vec![0.0f32; 128];
        for (i, item) in input.iter_mut().enumerate() {
            *item = ((i as f32) / 20.0).sin() * 3.0;
        }

        let mut output = vec![0u8; 32];
        let mut scales = vec![0.0f32; 1];

        kernel.quantize(&input, &mut output, &mut scales, QuantizationType::TL2).unwrap();

        // Check that quantization produced meaningful results
        assert!(scales[0] > 0.0);
        assert!(output.iter().any(|&x| x != 0));
    }

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

            for (i, item) in a.iter_mut().enumerate() {
                *item = ((i % 5) as i8) - 2; // Values from -2 to 2
            }
            for (i, item) in b.iter_mut().enumerate() {
                *item = (i % 3) as u8; // Values from 0 to 2
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
        for (i, item) in input.iter_mut().enumerate() {
            *item = ((i as f32) / 10.0).sin() * 5.0;
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

    /// Test AVX2 QK256 dequantization basic correctness
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_dequantize_qk256_basic() {
        let kernel = Avx2Kernel;

        if !kernel.is_available() {
            eprintln!("Skipping QK256 dequantize test: AVX2 not available");
            return;
        }

        // Test with 2 blocks (512 elements total)
        const QK256: usize = 256;
        const QK256_PACKED_BYTES: usize = 64;
        let num_blocks = 2;

        // Create packed data: all codes = 2 (→ +1.0 with LUT)
        // Pattern: 0b_10_10_10_10 = 0xAA
        let quantized = vec![0xAAu8 as i8; num_blocks * QK256_PACKED_BYTES];

        // Scales for each block
        let scales = vec![2.0f32, 3.0f32];

        // Dequantize
        let result =
            kernel.dequantize_qk256(&quantized, &scales, 256).expect("dequantize should succeed");

        // Verify length
        assert_eq!(
            result.len(),
            num_blocks * QK256,
            "Output length should be {} elements",
            num_blocks * QK256
        );

        // Verify values in first block (scale=2.0, code=2 → weight=+1.0)
        for (i, &val) in result.iter().take(QK256).enumerate() {
            let expected = 1.0 * 2.0; // LUT[2] * scale[0]
            assert!(
                (val - expected).abs() < 1e-5,
                "Block 0 element {}: expected {}, got {}",
                i,
                expected,
                val
            );
        }

        // Verify values in second block (scale=3.0, code=2 → weight=+1.0)
        for (i, &val) in result.iter().skip(QK256).take(QK256).enumerate() {
            let expected = 1.0 * 3.0; // LUT[2] * scale[1]
            assert!(
                (val - expected).abs() < 1e-5,
                "Block 1 element {}: expected {}, got {}",
                i,
                expected,
                val
            );
        }
    }

    /// Test AVX2 QK256 dequantization matches scalar reference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_dequantize_qk256_matches_scalar() {
        let kernel = Avx2Kernel;

        if !kernel.is_available() {
            eprintln!("Skipping QK256 dequantize correctness test: AVX2 not available");
            return;
        }

        // Test with random data (4 blocks = 1024 elements)
        const QK256_PACKED_BYTES: usize = 64;
        let num_blocks = 4;

        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Generate random quantized data
        let mut quantized = vec![0i8; num_blocks * QK256_PACKED_BYTES];
        for byte in quantized.iter_mut() {
            *byte = rng.random();
        }

        // Generate random scales
        let scales: Vec<f32> = (0..num_blocks).map(|_| rng.random_range(0.5..5.0)).collect();

        // Compute AVX2 result
        let result_avx2 = kernel
            .dequantize_qk256(&quantized, &scales, 256)
            .expect("AVX2 dequantize should succeed");

        // Compute scalar reference
        let result_scalar = kernel
            .dequantize_qk256_scalar(&quantized, &scales, 256)
            .expect("Scalar dequantize should succeed");

        // Compare results
        assert_eq!(result_avx2.len(), result_scalar.len(), "Output lengths should match");

        for (i, (&avx2_val, &scalar_val)) in
            result_avx2.iter().zip(result_scalar.iter()).enumerate()
        {
            let abs_diff = (avx2_val - scalar_val).abs();
            assert!(
                abs_diff < 1e-5,
                "Mismatch at element {}: AVX2={}, Scalar={}, diff={}",
                i,
                avx2_val,
                scalar_val,
                abs_diff
            );
        }

        println!("✅ AVX2 QK256 dequantize matches scalar for {} elements", result_avx2.len());
    }

    /// Test QK256 dequantization with all LUT codes
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_dequantize_qk256_all_codes() {
        let kernel = Avx2Kernel;

        if !kernel.is_available() {
            eprintln!("Skipping QK256 all-codes test: AVX2 not available");
            return;
        }

        const QK256_PACKED_BYTES: usize = 64;
        const LUT: [f32; 4] = [-2.0, -1.0, 1.0, 2.0];

        // Create packed data cycling through all codes (0,1,2,3,0,1,2,3,...)
        let mut quantized = vec![0i8; QK256_PACKED_BYTES];
        for (i, byte) in quantized.iter_mut().enumerate() {
            let base = i * 4;
            let code0 = (base % 4) as u8;
            let code1 = ((base + 1) % 4) as u8;
            let code2 = ((base + 2) % 4) as u8;
            let code3 = ((base + 3) % 4) as u8;
            *byte = (code0 | (code1 << 2) | (code2 << 4) | (code3 << 6)) as i8;
        }

        let scales = vec![1.5f32];

        // Dequantize
        let result =
            kernel.dequantize_qk256(&quantized, &scales, 256).expect("dequantize should succeed");

        // Verify each element matches expected LUT value * scale
        for (i, &val) in result.iter().enumerate() {
            let expected_code = i % 4;
            let expected = LUT[expected_code] * 1.5;
            assert!(
                (val - expected).abs() < 1e-5,
                "Element {}: expected {} (code {} → LUT[{}]={} * 1.5), got {}",
                i,
                expected,
                expected_code,
                expected_code,
                LUT[expected_code],
                val
            );
        }

        println!("✅ AVX2 QK256 dequantize correctly handles all LUT codes");
    }

    /// Test QK256 dequantization error handling
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_dequantize_qk256_errors() {
        let kernel = Avx2Kernel;

        // Test 1: Wrong block size
        {
            let quantized = vec![0i8; 64];
            let scales = vec![1.0f32];
            let result = kernel.dequantize_qk256(&quantized, &scales, 128);
            assert!(result.is_err(), "Should fail with wrong block size");
            assert!(
                result.unwrap_err().to_string().contains("block_size=256"),
                "Error should mention required block size"
            );
        }

        // Test 2: Size mismatch (both AVX2 and scalar paths validate)
        // For 2 blocks: expected = 2 * 64 = 128 bytes
        // Tolerance = 128 bytes, so anything outside [0, 256] fails
        {
            let quantized = vec![0i8; 300]; // Way too large: 300 > 128 + 128 = 256
            let scales = vec![1.0f32, 1.0f32]; // 2 blocks
            let result = kernel.dequantize_qk256(&quantized, &scales, 256);
            // Both AVX2 and scalar paths should reject sizes outside tolerance
            assert!(result.is_err(), "Should fail with size mismatch");
            assert!(
                result.unwrap_err().to_string().contains("size mismatch"),
                "Error should mention size mismatch"
            );
        }
    }
}
