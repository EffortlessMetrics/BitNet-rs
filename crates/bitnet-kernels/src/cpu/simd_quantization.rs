//! SIMD-optimized quantization kernels for ternary (I2S) and 8-bit absmax formats.
//!
//! Provides scalar fallback, AVX2 (x86_64), and NEON (aarch64) implementations
//! with runtime dispatch that selects the best available backend.

use core::fmt;

// ── Error type ──────────────────────────────────────────────────────────────

/// Errors returned by SIMD quantization operations.
#[derive(Debug, Clone, PartialEq)]
pub enum SimdQuantError {
    /// Input length is not a multiple of the block size.
    InvalidBlockSize { input_len: usize, block_size: usize },
    /// Output buffer is too small.
    OutputBufferTooSmall { need: usize, have: usize },
    /// Scale buffer is too small.
    ScaleBufferTooSmall { need: usize, have: usize },
    /// Input contains NaN values.
    NanInput,
    /// Input contains infinite values.
    InfInput,
}

impl fmt::Display for SimdQuantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBlockSize { input_len, block_size } => {
                write!(f, "input length {input_len} is not a multiple of block size {block_size}")
            }
            Self::OutputBufferTooSmall { need, have } => {
                write!(f, "output buffer too small: need {need}, have {have}")
            }
            Self::ScaleBufferTooSmall { need, have } => {
                write!(f, "scale buffer too small: need {need}, have {have}")
            }
            Self::NanInput => write!(f, "input contains NaN values"),
            Self::InfInput => write!(f, "input contains infinite values"),
        }
    }
}

impl std::error::Error for SimdQuantError {}

// ── Configuration ───────────────────────────────────────────────────────────

/// Scale computation method.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScaleType {
    /// Use the absolute maximum value in the block.
    AbsMax,
    /// Use the root-mean-square of the block.
    Rms,
}

/// Rounding mode for quantization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RoundingMode {
    /// Round to nearest, ties to even.
    Nearest,
    /// Stochastic rounding (not yet implemented).
    Stochastic,
}

/// Configuration for SIMD quantization operations.
#[derive(Debug, Clone)]
pub struct SimdQuantConfig {
    /// Number of elements per quantization block.
    pub block_size: usize,
    /// How to compute the per-block scale factor.
    pub scale_type: ScaleType,
    /// Rounding mode.
    pub rounding_mode: RoundingMode,
    /// Threshold for ternary quantization (values with |x| < threshold*scale → 0).
    pub ternary_threshold: f32,
}

impl Default for SimdQuantConfig {
    fn default() -> Self {
        Self {
            block_size: 64,
            scale_type: ScaleType::AbsMax,
            rounding_mode: RoundingMode::Nearest,
            ternary_threshold: 0.5,
        }
    }
}

// ── Validation helpers ──────────────────────────────────────────────────────

fn validate_input(input: &[f32]) -> Result<(), SimdQuantError> {
    for &v in input {
        if v.is_nan() {
            return Err(SimdQuantError::NanInput);
        }
        if v.is_infinite() {
            return Err(SimdQuantError::InfInput);
        }
    }
    Ok(())
}

fn validate_blocks(
    input_len: usize,
    output_len: usize,
    scale_len: usize,
    block_size: usize,
) -> Result<usize, SimdQuantError> {
    if block_size == 0 || !input_len.is_multiple_of(block_size) {
        return Err(SimdQuantError::InvalidBlockSize { input_len, block_size });
    }
    let n_blocks = input_len / block_size;
    if output_len < input_len {
        return Err(SimdQuantError::OutputBufferTooSmall { need: input_len, have: output_len });
    }
    if scale_len < n_blocks {
        return Err(SimdQuantError::ScaleBufferTooSmall { need: n_blocks, have: scale_len });
    }
    Ok(n_blocks)
}

fn validate_dequant_blocks(
    quant_len: usize,
    output_len: usize,
    scale_len: usize,
    block_size: usize,
) -> Result<usize, SimdQuantError> {
    if block_size == 0 || !quant_len.is_multiple_of(block_size) {
        return Err(SimdQuantError::InvalidBlockSize { input_len: quant_len, block_size });
    }
    let n_blocks = quant_len / block_size;
    if output_len < quant_len {
        return Err(SimdQuantError::OutputBufferTooSmall { need: quant_len, have: output_len });
    }
    if scale_len < n_blocks {
        return Err(SimdQuantError::ScaleBufferTooSmall { need: n_blocks, have: scale_len });
    }
    Ok(n_blocks)
}

fn block_scale(block: &[f32], scale_type: ScaleType) -> f32 {
    match scale_type {
        ScaleType::AbsMax => {
            let mut mx = 0.0f32;
            for &v in block {
                let a = v.abs();
                if a > mx {
                    mx = a;
                }
            }
            mx
        }
        ScaleType::Rms => {
            let mut sum_sq = 0.0f64;
            for &v in block {
                sum_sq += (v as f64) * (v as f64);
            }
            (sum_sq / block.len() as f64).sqrt() as f32
        }
    }
}

// ── Scalar implementations ──────────────────────────────────────────────────

/// Ternary (I2S) quantization — scalar fallback.
///
/// Maps each element to {-1, 0, +1} based on `ternary_threshold * scale`.
pub fn quantize_i2s_scalar(
    input: &[f32],
    output: &mut [i8],
    scales: &mut [f32],
    config: &SimdQuantConfig,
) -> Result<(), SimdQuantError> {
    validate_input(input)?;
    let n_blocks = validate_blocks(input.len(), output.len(), scales.len(), config.block_size)?;
    let bs = config.block_size;

    for (blk, scale_slot) in scales.iter_mut().enumerate().take(n_blocks) {
        let start = blk * bs;
        let block = &input[start..start + bs];
        let s = block_scale(block, config.scale_type);
        *scale_slot = s;
        let thresh = config.ternary_threshold * s;
        for (j, &v) in block.iter().enumerate() {
            output[start + j] = if v > thresh {
                1
            } else if v < -thresh {
                -1
            } else {
                0
            };
        }
    }
    Ok(())
}

/// Ternary (I2S) dequantization — scalar fallback.
///
/// Reconstructs `output[i] = quantized[i] as f32 * scale`.
pub fn dequantize_i2s_scalar(
    quantized: &[i8],
    output: &mut [f32],
    scales: &[f32],
    config: &SimdQuantConfig,
) -> Result<(), SimdQuantError> {
    let n_blocks =
        validate_dequant_blocks(quantized.len(), output.len(), scales.len(), config.block_size)?;
    let bs = config.block_size;

    for (blk, &scale_ref) in scales.iter().enumerate().take(n_blocks) {
        let start = blk * bs;
        for j in 0..bs {
            output[start + j] = quantized[start + j] as f32 * scale_ref;
        }
    }
    Ok(())
}

/// 8-bit absmax quantization — scalar fallback.
///
/// Quantizes to `[-127, 127]` with per-block scale = absmax / 127.
pub fn quantize_absmax_scalar(
    input: &[f32],
    output: &mut [i8],
    scales: &mut [f32],
    config: &SimdQuantConfig,
) -> Result<(), SimdQuantError> {
    validate_input(input)?;
    let n_blocks = validate_blocks(input.len(), output.len(), scales.len(), config.block_size)?;
    let bs = config.block_size;

    for (blk, scale_slot) in scales.iter_mut().enumerate().take(n_blocks) {
        let start = blk * bs;
        let block = &input[start..start + bs];
        let abs_max = block_scale(block, ScaleType::AbsMax);
        let s = if abs_max == 0.0 { 1.0 } else { abs_max / 127.0 };
        *scale_slot = s;
        let inv = 1.0 / s;
        for (j, &v) in block.iter().enumerate() {
            let q = (v * inv).round().clamp(-127.0, 127.0) as i8;
            output[start + j] = q;
        }
    }
    Ok(())
}

/// 8-bit absmax dequantization — scalar fallback.
///
/// Reconstructs `output[i] = quantized[i] as f32 * scale`.
pub fn dequantize_absmax_scalar(
    quantized: &[i8],
    output: &mut [f32],
    scales: &[f32],
    config: &SimdQuantConfig,
) -> Result<(), SimdQuantError> {
    let n_blocks =
        validate_dequant_blocks(quantized.len(), output.len(), scales.len(), config.block_size)?;
    let bs = config.block_size;

    for (blk, &scale_ref) in scales.iter().enumerate().take(n_blocks) {
        let start = blk * bs;
        for j in 0..bs {
            output[start + j] = quantized[start + j] as f32 * scale_ref;
        }
    }
    Ok(())
}

// ── AVX2 implementations ───────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
mod avx2 {
    use super::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// SAFETY: Caller must ensure AVX2 is available at runtime.
    #[target_feature(enable = "avx2")]
    pub unsafe fn quantize_i2s_avx2(
        input: &[f32],
        output: &mut [i8],
        scales: &mut [f32],
        config: &SimdQuantConfig,
    ) -> Result<(), SimdQuantError> {
        validate_input(input)?;
        let n_blocks = validate_blocks(input.len(), output.len(), scales.len(), config.block_size)?;
        let bs = config.block_size;

        for (blk, scale_slot) in scales.iter_mut().enumerate().take(n_blocks) {
            let start = blk * bs;
            let block = &input[start..start + bs];
            let s = block_scale(block, config.scale_type);
            *scale_slot = s;
            let thresh = config.ternary_threshold * s;

            let pos_thresh = _mm256_set1_ps(thresh);
            let neg_thresh = _mm256_set1_ps(-thresh);
            let ones = _mm256_set1_ps(1.0);
            let neg_ones = _mm256_set1_ps(-1.0);
            let zero = _mm256_setzero_ps();

            // Process 8 floats at a time with AVX2
            let mut j = 0;
            while j + 8 <= bs {
                let v = _mm256_loadu_ps(block.as_ptr().add(j));
                // mask_pos: v > thresh
                let mask_pos = _mm256_cmp_ps(v, pos_thresh, _CMP_GT_OQ);
                // mask_neg: v < -thresh
                let mask_neg = _mm256_cmp_ps(v, neg_thresh, _CMP_LT_OQ);
                // result = blend(0, 1, mask_pos)
                let r = _mm256_blendv_ps(zero, ones, mask_pos);
                // result = blend(result, -1, mask_neg)
                let r = _mm256_blendv_ps(r, neg_ones, mask_neg);
                // Convert to i32
                let ri = _mm256_cvtps_epi32(r);
                // Pack i32 → i16 → i8 (we only need low bytes)
                let lo = _mm256_castsi256_si128(ri);
                let hi = _mm256_extracti128_si256(ri, 1);
                let packed16 = _mm_packs_epi32(lo, hi);
                let packed8 = _mm_packs_epi16(packed16, packed16);

                // SAFETY: _mm_packs_epi32 interleaves [lo0..lo3, hi0..hi3],
                // so the 8 values we need are in the low 8 bytes of packed8.
                let mut tmp = [0i8; 16];
                _mm_storeu_si128(tmp.as_mut_ptr().cast(), packed8);
                output[start + j..start + j + 8].copy_from_slice(&tmp[..8]);

                j += 8;
            }
            // Scalar tail
            while j < bs {
                let v = block[j];
                output[start + j] = if v > thresh {
                    1
                } else if v < -thresh {
                    -1
                } else {
                    0
                };
                j += 1;
            }
        }
        Ok(())
    }

    /// SAFETY: Caller must ensure AVX2 is available at runtime.
    #[target_feature(enable = "avx2")]
    pub unsafe fn dequantize_i2s_avx2(
        quantized: &[i8],
        output: &mut [f32],
        scales: &[f32],
        config: &SimdQuantConfig,
    ) -> Result<(), SimdQuantError> {
        let n_blocks = validate_dequant_blocks(
            quantized.len(),
            output.len(),
            scales.len(),
            config.block_size,
        )?;
        let bs = config.block_size;

        for (blk, &scale_ref) in scales.iter().enumerate().take(n_blocks) {
            let start = blk * bs;
            let scale_v = _mm256_set1_ps(scale_ref);

            let mut j = 0;
            while j + 8 <= bs {
                // Load 8 bytes, sign-extend to i32, convert to f32, multiply
                let raw = std::ptr::read_unaligned(quantized.as_ptr().add(start + j).cast::<i64>());
                let bytes = _mm_set_epi64x(0, raw);
                let lo_16 = _mm_cvtepi8_epi16(bytes);
                let lo_32 = _mm_cvtepi16_epi32(lo_16);
                let hi_16 = _mm_shuffle_epi32(lo_16, 0b01_00_11_10);
                let hi_32 = _mm_cvtepi16_epi32(hi_16);

                let lo_256 = _mm256_insertf128_si256(_mm256_castsi128_si256(lo_32), hi_32, 1);
                let fvals = _mm256_cvtepi32_ps(lo_256);
                let result = _mm256_mul_ps(fvals, scale_v);
                _mm256_storeu_ps(output.as_mut_ptr().add(start + j), result);

                j += 8;
            }
            while j < bs {
                output[start + j] = quantized[start + j] as f32 * scale_ref;
                j += 1;
            }
        }
        Ok(())
    }

    /// SAFETY: Caller must ensure AVX2 is available at runtime.
    #[target_feature(enable = "avx2")]
    pub unsafe fn quantize_absmax_avx2(
        input: &[f32],
        output: &mut [i8],
        scales: &mut [f32],
        config: &SimdQuantConfig,
    ) -> Result<(), SimdQuantError> {
        validate_input(input)?;
        let n_blocks = validate_blocks(input.len(), output.len(), scales.len(), config.block_size)?;
        let bs = config.block_size;

        for (blk, scale_slot) in scales.iter_mut().enumerate().take(n_blocks) {
            let start = blk * bs;
            let block = &input[start..start + bs];
            let abs_max = block_scale(block, ScaleType::AbsMax);
            let s = if abs_max == 0.0 { 1.0 } else { abs_max / 127.0 };
            *scale_slot = s;
            let inv_s = _mm256_set1_ps(1.0 / s);
            let clamp_lo = _mm256_set1_ps(-127.0);
            let clamp_hi = _mm256_set1_ps(127.0);

            let mut j = 0;
            while j + 8 <= bs {
                let v = _mm256_loadu_ps(block.as_ptr().add(j));
                let scaled = _mm256_mul_ps(v, inv_s);
                let rounded =
                    _mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                let clamped = _mm256_max_ps(_mm256_min_ps(rounded, clamp_hi), clamp_lo);
                let ints = _mm256_cvtps_epi32(clamped);

                let lo = _mm256_castsi256_si128(ints);
                let hi = _mm256_extracti128_si256(ints, 1);
                let packed16 = _mm_packs_epi32(lo, hi);
                let packed8 = _mm_packs_epi16(packed16, packed16);

                // SAFETY: see quantize_i2s_avx2 — low 8 bytes are the 8 packed values
                let mut tmp = [0i8; 16];
                _mm_storeu_si128(tmp.as_mut_ptr().cast(), packed8);
                output[start + j..start + j + 8].copy_from_slice(&tmp[..8]);

                j += 8;
            }
            while j < bs {
                let q = (block[j] / s).round().clamp(-127.0, 127.0) as i8;
                output[start + j] = q;
                j += 1;
            }
        }
        Ok(())
    }

    /// SAFETY: Caller must ensure AVX2 is available at runtime.
    #[target_feature(enable = "avx2")]
    pub unsafe fn dequantize_absmax_avx2(
        quantized: &[i8],
        output: &mut [f32],
        scales: &[f32],
        config: &SimdQuantConfig,
    ) -> Result<(), SimdQuantError> {
        let n_blocks = validate_dequant_blocks(
            quantized.len(),
            output.len(),
            scales.len(),
            config.block_size,
        )?;
        let bs = config.block_size;

        for (blk, &scale_ref) in scales.iter().enumerate().take(n_blocks) {
            let start = blk * bs;
            let scale_v = _mm256_set1_ps(scale_ref);

            let mut j = 0;
            while j + 8 <= bs {
                let raw = std::ptr::read_unaligned(quantized.as_ptr().add(start + j).cast::<i64>());
                let bytes = _mm_set_epi64x(0, raw);
                let lo_16 = _mm_cvtepi8_epi16(bytes);
                let lo_32 = _mm_cvtepi16_epi32(lo_16);
                let hi_16 = _mm_shuffle_epi32(lo_16, 0b01_00_11_10);
                let hi_32 = _mm_cvtepi16_epi32(hi_16);

                let lo_256 = _mm256_insertf128_si256(_mm256_castsi128_si256(lo_32), hi_32, 1);
                let fvals = _mm256_cvtepi32_ps(lo_256);
                let result = _mm256_mul_ps(fvals, scale_v);
                _mm256_storeu_ps(output.as_mut_ptr().add(start + j), result);

                j += 8;
            }
            while j < bs {
                output[start + j] = quantized[start + j] as f32 * scale_ref;
                j += 1;
            }
        }
        Ok(())
    }
}

// ── NEON implementations ────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;

    pub fn quantize_i2s_neon(
        input: &[f32],
        output: &mut [i8],
        scales: &mut [f32],
        config: &SimdQuantConfig,
    ) -> Result<(), SimdQuantError> {
        validate_input(input)?;
        let n_blocks = validate_blocks(input.len(), output.len(), scales.len(), config.block_size)?;
        let bs = config.block_size;

        for (blk, scale_slot) in scales.iter_mut().enumerate().take(n_blocks) {
            let start = blk * bs;
            let block = &input[start..start + bs];
            let s = block_scale(block, config.scale_type);
            *scale_slot = s;
            let thresh = config.ternary_threshold * s;

            // SAFETY: NEON is always available on aarch64.
            unsafe {
                let pos_thresh = vdupq_n_f32(thresh);
                let neg_thresh = vdupq_n_f32(-thresh);

                let mut j = 0;
                while j + 4 <= bs {
                    let v = vld1q_f32(block.as_ptr().add(j));
                    let mask_pos = vcgtq_f32(v, pos_thresh);
                    let mask_neg = vcltq_f32(v, neg_thresh);
                    let ones = vdupq_n_s32(1);
                    let neg_ones = vdupq_n_s32(-1);
                    let r = vbslq_s32(
                        vreinterpretq_u32_f32(vreinterpretq_f32_u32(mask_pos)),
                        ones,
                        vdupq_n_s32(0),
                    );
                    let r = vbslq_s32(
                        vreinterpretq_u32_f32(vreinterpretq_f32_u32(mask_neg)),
                        neg_ones,
                        r,
                    );
                    let n16 = vmovn_s32(r);
                    let n8 = vmovn_s16(vcombine_s16(n16, n16));
                    let mut tmp = [0i8; 8];
                    vst1_s8(tmp.as_mut_ptr(), n8);
                    output[start + j..start + j + 4].copy_from_slice(&tmp[..4]);

                    j += 4;
                }
                while j < bs {
                    let v = block[j];
                    output[start + j] = if v > thresh {
                        1
                    } else if v < -thresh {
                        -1
                    } else {
                        0
                    };
                    j += 1;
                }
            }
        }
        Ok(())
    }

    pub fn dequantize_i2s_neon(
        quantized: &[i8],
        output: &mut [f32],
        scales: &[f32],
        config: &SimdQuantConfig,
    ) -> Result<(), SimdQuantError> {
        let n_blocks = validate_dequant_blocks(
            quantized.len(),
            output.len(),
            scales.len(),
            config.block_size,
        )?;
        let bs = config.block_size;

        for (blk, &scale_ref) in scales.iter().enumerate().take(n_blocks) {
            let start = blk * bs;

            // SAFETY: NEON is always available on aarch64.
            unsafe {
                let scale_v = vdupq_n_f32(scale_ref);
                let mut j = 0;
                while j + 4 <= bs {
                    let b0 = quantized[start + j] as i32;
                    let b1 = quantized[start + j + 1] as i32;
                    let b2 = quantized[start + j + 2] as i32;
                    let b3 = quantized[start + j + 3] as i32;
                    let iv = vcombine_s32(
                        vcreate_s32(b0 as i64 | ((b1 as i64) << 32)),
                        vcreate_s32(b2 as i64 | ((b3 as i64) << 32)),
                    );
                    let fv = vcvtq_f32_s32(iv);
                    let result = vmulq_f32(fv, scale_v);
                    vst1q_f32(output.as_mut_ptr().add(start + j), result);
                    j += 4;
                }
                while j < bs {
                    output[start + j] = quantized[start + j] as f32 * scale_ref;
                    j += 1;
                }
            }
        }
        Ok(())
    }

    pub fn quantize_absmax_neon(
        input: &[f32],
        output: &mut [i8],
        scales: &mut [f32],
        config: &SimdQuantConfig,
    ) -> Result<(), SimdQuantError> {
        validate_input(input)?;
        let n_blocks = validate_blocks(input.len(), output.len(), scales.len(), config.block_size)?;
        let bs = config.block_size;

        for (blk, scale_slot) in scales.iter_mut().enumerate().take(n_blocks) {
            let start = blk * bs;
            let block = &input[start..start + bs];
            let abs_max = block_scale(block, ScaleType::AbsMax);
            let s = if abs_max == 0.0 { 1.0 } else { abs_max / 127.0 };
            *scale_slot = s;

            // SAFETY: NEON is always available on aarch64.
            unsafe {
                let inv_s = vdupq_n_f32(1.0 / s);
                let clamp_lo = vdupq_n_f32(-127.0);
                let clamp_hi = vdupq_n_f32(127.0);

                let mut j = 0;
                while j + 4 <= bs {
                    let v = vld1q_f32(block.as_ptr().add(j));
                    let scaled = vmulq_f32(v, inv_s);
                    let rounded = vrndnq_f32(scaled);
                    let clamped = vminq_f32(vmaxq_f32(rounded, clamp_lo), clamp_hi);
                    let ints = vcvtq_s32_f32(clamped);
                    let n16 = vmovn_s32(ints);
                    let n8 = vmovn_s16(vcombine_s16(n16, n16));
                    let mut tmp = [0i8; 8];
                    vst1_s8(tmp.as_mut_ptr(), n8);
                    output[start + j..start + j + 4].copy_from_slice(&tmp[..4]);
                    j += 4;
                }
                while j < bs {
                    let q = (block[j] / s).round().clamp(-127.0, 127.0) as i8;
                    output[start + j] = q;
                    j += 1;
                }
            }
        }
        Ok(())
    }

    pub fn dequantize_absmax_neon(
        quantized: &[i8],
        output: &mut [f32],
        scales: &[f32],
        config: &SimdQuantConfig,
    ) -> Result<(), SimdQuantError> {
        let n_blocks = validate_dequant_blocks(
            quantized.len(),
            output.len(),
            scales.len(),
            config.block_size,
        )?;
        let bs = config.block_size;

        for (blk, &scale_ref) in scales.iter().enumerate().take(n_blocks) {
            let start = blk * bs;

            // SAFETY: NEON is always available on aarch64.
            unsafe {
                let scale_v = vdupq_n_f32(scale_ref);
                let mut j = 0;
                while j + 4 <= bs {
                    let b0 = quantized[start + j] as i32;
                    let b1 = quantized[start + j + 1] as i32;
                    let b2 = quantized[start + j + 2] as i32;
                    let b3 = quantized[start + j + 3] as i32;
                    let iv = vcombine_s32(
                        vcreate_s32(b0 as i64 | ((b1 as i64) << 32)),
                        vcreate_s32(b2 as i64 | ((b3 as i64) << 32)),
                    );
                    let fv = vcvtq_f32_s32(iv);
                    let result = vmulq_f32(fv, scale_v);
                    vst1q_f32(output.as_mut_ptr().add(start + j), result);
                    j += 4;
                }
                while j < bs {
                    output[start + j] = quantized[start + j] as f32 * scale_ref;
                    j += 1;
                }
            }
        }
        Ok(())
    }
}

// ── Runtime dispatch ────────────────────────────────────────────────────────

/// Ternary (I2S) quantization with runtime SIMD dispatch.
pub fn quantize_i2s(
    input: &[f32],
    output: &mut [i8],
    scales: &mut [f32],
    config: &SimdQuantConfig,
) -> Result<(), SimdQuantError> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 availability checked above.
            return unsafe { avx2::quantize_i2s_avx2(input, output, scales, config) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return neon::quantize_i2s_neon(input, output, scales, config);
        }
    }
    quantize_i2s_scalar(input, output, scales, config)
}

/// Ternary (I2S) dequantization with runtime SIMD dispatch.
pub fn dequantize_i2s(
    quantized: &[i8],
    output: &mut [f32],
    scales: &[f32],
    config: &SimdQuantConfig,
) -> Result<(), SimdQuantError> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 availability checked above.
            return unsafe { avx2::dequantize_i2s_avx2(quantized, output, scales, config) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return neon::dequantize_i2s_neon(quantized, output, scales, config);
        }
    }
    dequantize_i2s_scalar(quantized, output, scales, config)
}

/// 8-bit absmax quantization with runtime SIMD dispatch.
pub fn quantize_absmax(
    input: &[f32],
    output: &mut [i8],
    scales: &mut [f32],
    config: &SimdQuantConfig,
) -> Result<(), SimdQuantError> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 availability checked above.
            return unsafe { avx2::quantize_absmax_avx2(input, output, scales, config) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return neon::quantize_absmax_neon(input, output, scales, config);
        }
    }
    quantize_absmax_scalar(input, output, scales, config)
}

/// 8-bit absmax dequantization with runtime SIMD dispatch.
pub fn dequantize_absmax(
    quantized: &[i8],
    output: &mut [f32],
    scales: &[f32],
    config: &SimdQuantConfig,
) -> Result<(), SimdQuantError> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 availability checked above.
            return unsafe { avx2::dequantize_absmax_avx2(quantized, output, scales, config) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return neon::dequantize_absmax_neon(quantized, output, scales, config);
        }
    }
    dequantize_absmax_scalar(quantized, output, scales, config)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::too_many_lines)]
mod tests {
    use super::*;

    fn cfg_with_block(block_size: usize) -> SimdQuantConfig {
        SimdQuantConfig { block_size, ..Default::default() }
    }

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn assert_approx_slice(a: &[f32], b: &[f32], eps: f32, label: &str) {
        assert_eq!(a.len(), b.len(), "{label}: length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(approx_eq(x, y, eps), "{label}[{i}]: {x} vs {y} (eps={eps})");
        }
    }

    // ── Error path tests ────────────────────────────────────────────────

    #[test]
    fn invalid_block_size_zero() {
        let cfg = cfg_with_block(0);
        let input = vec![1.0; 8];
        let mut out = vec![0i8; 8];
        let mut scales = vec![0.0; 1];
        let err = quantize_i2s_scalar(&input, &mut out, &mut scales, &cfg).unwrap_err();
        assert!(matches!(err, SimdQuantError::InvalidBlockSize { .. }));
    }

    #[test]
    fn invalid_block_size_not_multiple_of_8() {
        let cfg = cfg_with_block(3);
        let input = vec![1.0; 8];
        let mut out = vec![0i8; 8];
        let mut scales = vec![0.0; 3];
        let err = quantize_i2s_scalar(&input, &mut out, &mut scales, &cfg).unwrap_err();
        assert!(matches!(err, SimdQuantError::InvalidBlockSize { .. }));
    }

    #[test]
    fn output_buffer_too_small() {
        let cfg = cfg_with_block(8);
        let input = vec![1.0; 8];
        let mut out = vec![0i8; 4];
        let mut scales = vec![0.0; 1];
        let err = quantize_i2s_scalar(&input, &mut out, &mut scales, &cfg).unwrap_err();
        assert!(matches!(err, SimdQuantError::OutputBufferTooSmall { .. }));
    }

    #[test]
    fn scale_buffer_too_small() {
        let cfg = cfg_with_block(8);
        let input = vec![1.0; 16];
        let mut out = vec![0i8; 16];
        let mut scales = vec![0.0; 1]; // need 2
        let err = quantize_i2s_scalar(&input, &mut out, &mut scales, &cfg).unwrap_err();
        assert!(matches!(err, SimdQuantError::ScaleBufferTooSmall { .. }));
    }

    #[test]
    fn nan_input_rejected() {
        let cfg = cfg_with_block(8);
        let mut input = vec![1.0; 8];
        input[3] = f32::NAN;
        let mut out = vec![0i8; 8];
        let mut scales = vec![0.0; 1];
        let err = quantize_i2s_scalar(&input, &mut out, &mut scales, &cfg).unwrap_err();
        assert!(matches!(err, SimdQuantError::NanInput));
    }

    #[test]
    fn inf_input_rejected() {
        let cfg = cfg_with_block(8);
        let mut input = vec![1.0; 8];
        input[0] = f32::INFINITY;
        let mut out = vec![0i8; 8];
        let mut scales = vec![0.0; 1];
        let err = quantize_i2s_scalar(&input, &mut out, &mut scales, &cfg).unwrap_err();
        assert!(matches!(err, SimdQuantError::InfInput));
    }

    #[test]
    fn neg_inf_input_rejected() {
        let cfg = cfg_with_block(8);
        let mut input = vec![1.0; 8];
        input[0] = f32::NEG_INFINITY;
        let mut out = vec![0i8; 8];
        let mut scales = vec![0.0; 1];
        let err = quantize_i2s_scalar(&input, &mut out, &mut scales, &cfg).unwrap_err();
        assert!(matches!(err, SimdQuantError::InfInput));
    }

    #[test]
    fn nan_absmax_rejected() {
        let cfg = cfg_with_block(8);
        let mut input = vec![1.0; 8];
        input[5] = f32::NAN;
        let mut out = vec![0i8; 8];
        let mut scales = vec![0.0; 1];
        let err = quantize_absmax_scalar(&input, &mut out, &mut scales, &cfg).unwrap_err();
        assert!(matches!(err, SimdQuantError::NanInput));
    }

    #[test]
    fn error_display_coverage() {
        let e1 = SimdQuantError::InvalidBlockSize { input_len: 10, block_size: 3 };
        assert!(e1.to_string().contains("10"));
        let e2 = SimdQuantError::OutputBufferTooSmall { need: 8, have: 4 };
        assert!(e2.to_string().contains("8"));
        let e3 = SimdQuantError::ScaleBufferTooSmall { need: 2, have: 1 };
        assert!(e3.to_string().contains("2"));
        assert!(SimdQuantError::NanInput.to_string().contains("NaN"));
        assert!(SimdQuantError::InfInput.to_string().contains("infinite"));
    }

    // ── I2S scalar tests ────────────────────────────────────────────────

    #[test]
    fn i2s_scalar_basic_roundtrip() {
        let cfg = cfg_with_block(8);
        let input = vec![2.0, -3.0, 0.1, 0.0, -0.05, 5.0, -5.0, 0.2];
        let mut q = vec![0i8; 8];
        let mut scales = vec![0.0f32; 1];
        quantize_i2s_scalar(&input, &mut q, &mut scales, &cfg).unwrap();

        // All values should be in {-1, 0, 1}
        for &v in &q {
            assert!(v >= -1 && v <= 1, "ternary range: {v}");
        }

        let mut out = vec![0.0f32; 8];
        dequantize_i2s_scalar(&q, &mut out, &scales, &cfg).unwrap();
        // Dequantized values should be in {-scale, 0, scale}
        let s = scales[0];
        for &v in &out {
            assert!(
                approx_eq(v, 0.0, 1e-6) || approx_eq(v.abs(), s, 1e-6),
                "dequant value {v} not in {{-{s}, 0, {s}}}"
            );
        }
    }

    #[test]
    fn i2s_scalar_output_only_ternary() {
        let cfg = cfg_with_block(8);
        let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let mut q = vec![0i8; 64];
        let mut scales = vec![0.0f32; 8];
        quantize_i2s_scalar(&input, &mut q, &mut scales, &cfg).unwrap();
        for &v in &q {
            assert!(v >= -1 && v <= 1);
        }
    }

    #[test]
    fn i2s_all_zeros() {
        let cfg = cfg_with_block(8);
        let input = vec![0.0f32; 8];
        let mut q = vec![0i8; 8];
        let mut scales = vec![0.0f32; 1];
        quantize_i2s_scalar(&input, &mut q, &mut scales, &cfg).unwrap();
        assert!(q.iter().all(|&v| v == 0));
        assert_eq!(scales[0], 0.0);
    }

    #[test]
    fn i2s_scale_is_absmax() {
        let cfg = cfg_with_block(8);
        let input = vec![1.0, -3.0, 2.0, 0.0, 0.5, -0.5, 3.0, -1.0];
        let mut q = vec![0i8; 8];
        let mut scales = vec![0.0f32; 1];
        quantize_i2s_scalar(&input, &mut q, &mut scales, &cfg).unwrap();
        assert!(approx_eq(scales[0], 3.0, 1e-6), "scale should be absmax=3.0");
    }

    #[test]
    fn i2s_scale_rms_mode() {
        let cfg =
            SimdQuantConfig { block_size: 8, scale_type: ScaleType::Rms, ..Default::default() };
        let input = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let mut q = vec![0i8; 8];
        let mut scales = vec![0.0f32; 1];
        quantize_i2s_scalar(&input, &mut q, &mut scales, &cfg).unwrap();
        // RMS of all ±1 = 1.0
        assert!(approx_eq(scales[0], 1.0, 1e-6), "RMS scale should be 1.0");
    }

    #[test]
    fn zero_threshold_means_sign_only() {
        let cfg = SimdQuantConfig { block_size: 8, ternary_threshold: 0.0, ..Default::default() };
        let input = vec![0.01, -0.01, 100.0, -100.0, 0.0, 0.001, -0.001, 0.0];
        let mut q = vec![0i8; 8];
        let mut scales = vec![0.0f32; 1];
        quantize_i2s_scalar(&input, &mut q, &mut scales, &cfg).unwrap();
        assert_eq!(q[0], 1);
        assert_eq!(q[1], -1);
        assert_eq!(q[2], 1);
        assert_eq!(q[3], -1);
        assert_eq!(q[4], 0); // exactly 0 stays 0
    }

    #[test]
    fn i2s_multi_block() {
        let cfg = cfg_with_block(8);
        let input: Vec<f32> = (0..24).map(|i| (i as f32) - 12.0).collect();
        let mut q = vec![0i8; 24];
        let mut scales = vec![0.0f32; 3];
        quantize_i2s_scalar(&input, &mut q, &mut scales, &cfg).unwrap();
        assert_eq!(scales.len(), 3);
        // Each block should have its own scale
        let mut out = vec![0.0f32; 24];
        dequantize_i2s_scalar(&q, &mut out, &scales, &cfg).unwrap();
        for &v in &out {
            assert!(!v.is_nan());
        }
    }

    // ── Absmax scalar tests ─────────────────────────────────────────────

    #[test]
    fn absmax_scalar_basic_roundtrip() {
        let cfg = cfg_with_block(8);
        let input = vec![1.0, -2.0, 3.0, -4.0, 0.5, -0.5, 4.0, -3.0];
        let mut q = vec![0i8; 8];
        let mut scales = vec![0.0f32; 1];
        quantize_absmax_scalar(&input, &mut q, &mut scales, &cfg).unwrap();

        let mut out = vec![0.0f32; 8];
        dequantize_absmax_scalar(&q, &mut out, &scales, &cfg).unwrap();

        for (i, (&orig, &deq)) in input.iter().zip(out.iter()).enumerate() {
            let tol = scales[0] / 2.0;
            assert!(
                approx_eq(orig, deq, tol),
                "absmax roundtrip[{i}]: {orig} vs {deq} (tol={tol})"
            );
        }
    }

    #[test]
    fn absmax_all_zeros() {
        let cfg = cfg_with_block(8);
        let input = vec![0.0f32; 8];
        let mut q = vec![0i8; 8];
        let mut scales = vec![0.0f32; 1];
        quantize_absmax_scalar(&input, &mut q, &mut scales, &cfg).unwrap();
        assert!(q.iter().all(|&v| v == 0));
    }

    #[test]
    fn absmax_clamps_to_127() {
        let cfg = cfg_with_block(8);
        let input = vec![10.0; 8];
        let mut q = vec![0i8; 8];
        let mut scales = vec![0.0f32; 1];
        quantize_absmax_scalar(&input, &mut q, &mut scales, &cfg).unwrap();
        assert!(q.iter().all(|&v| v == 127));
    }

    #[test]
    fn absmax_negative_clamps() {
        let cfg = cfg_with_block(8);
        let input = vec![-10.0; 8];
        let mut q = vec![0i8; 8];
        let mut scales = vec![0.0f32; 1];
        quantize_absmax_scalar(&input, &mut q, &mut scales, &cfg).unwrap();
        assert!(q.iter().all(|&v| v == -127));
    }

    #[test]
    fn absmax_multi_block_roundtrip() {
        let cfg = cfg_with_block(8);
        let input: Vec<f32> = (0..16).map(|i| ((i as f32) - 8.0) * 0.25).collect();
        let mut q = vec![0i8; 16];
        let mut scales = vec![0.0f32; 2];
        quantize_absmax_scalar(&input, &mut q, &mut scales, &cfg).unwrap();

        let mut out = vec![0.0f32; 16];
        dequantize_absmax_scalar(&q, &mut out, &scales, &cfg).unwrap();

        for (blk, i) in (0..16).enumerate() {
            let blk_idx = blk / 8;
            let tol = scales[blk_idx] / 2.0;
            assert!(
                approx_eq(input[i], out[i], tol),
                "multi-block roundtrip[{i}]: {} vs {} (tol={tol})",
                input[i],
                out[i]
            );
        }
    }

    #[test]
    fn absmax_small_values_precision() {
        let cfg = cfg_with_block(8);
        let input = vec![0.001, -0.002, 0.003, -0.004, 0.005, -0.006, 0.007, -0.008];
        let mut q = vec![0i8; 8];
        let mut scales = vec![0.0f32; 1];
        quantize_absmax_scalar(&input, &mut q, &mut scales, &cfg).unwrap();

        let mut out = vec![0.0f32; 8];
        dequantize_absmax_scalar(&q, &mut out, &scales, &cfg).unwrap();

        for (i, (&orig, &deq)) in input.iter().zip(out.iter()).enumerate() {
            let tol = scales[0] / 2.0;
            assert!(approx_eq(orig, deq, tol), "small values[{i}]: {orig} vs {deq} (tol={tol})");
        }
    }

    // ── Dispatch tests ──────────────────────────────────────────────────

    #[test]
    fn i2s_dispatch_roundtrip() {
        let cfg = cfg_with_block(8);
        let input = vec![2.0, -3.0, 0.1, 0.0, -0.05, 5.0, -5.0, 0.2];
        let mut q = vec![0i8; 8];
        let mut scales = vec![0.0f32; 1];
        quantize_i2s(&input, &mut q, &mut scales, &cfg).unwrap();
        for &v in &q {
            assert!(v >= -1 && v <= 1);
        }
        let mut out = vec![0.0f32; 8];
        dequantize_i2s(&q, &mut out, &scales, &cfg).unwrap();
        let s = scales[0];
        for &v in &out {
            assert!(approx_eq(v, 0.0, 1e-6) || approx_eq(v.abs(), s, 1e-6), "dispatch dequant {v}");
        }
    }

    #[test]
    fn absmax_dispatch_roundtrip() {
        let cfg = cfg_with_block(8);
        let input = vec![1.0, -2.0, 3.0, -4.0, 0.5, -0.5, 4.0, -3.0];
        let mut q = vec![0i8; 8];
        let mut scales = vec![0.0f32; 1];
        quantize_absmax(&input, &mut q, &mut scales, &cfg).unwrap();

        let mut out = vec![0.0f32; 8];
        dequantize_absmax(&q, &mut out, &scales, &cfg).unwrap();

        for (i, (&orig, &deq)) in input.iter().zip(out.iter()).enumerate() {
            let tol = scales[0] / 2.0;
            assert!(approx_eq(orig, deq, tol), "dispatch roundtrip[{i}]: {orig} vs {deq}");
        }
    }

    #[test]
    fn i2s_dispatch_error_propagation() {
        let cfg = cfg_with_block(0);
        let input = vec![1.0; 8];
        let mut out = vec![0i8; 8];
        let mut scales = vec![0.0; 1];
        assert!(quantize_i2s(&input, &mut out, &mut scales, &cfg).is_err());
    }

    #[test]
    fn absmax_dispatch_error_propagation() {
        let cfg = cfg_with_block(0);
        let input = vec![1.0; 8];
        let mut out = vec![0i8; 8];
        let mut scales = vec![0.0; 1];
        assert!(quantize_absmax(&input, &mut out, &mut scales, &cfg).is_err());
    }

    // ── SIMD-scalar parity tests ────────────────────────────────────────

    #[test]
    fn i2s_simd_matches_scalar() {
        let cfg = cfg_with_block(32);
        let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.15).collect();

        let mut q_scalar = vec![0i8; 64];
        let mut s_scalar = vec![0.0f32; 2];
        quantize_i2s_scalar(&input, &mut q_scalar, &mut s_scalar, &cfg).unwrap();

        let mut q_dispatch = vec![0i8; 64];
        let mut s_dispatch = vec![0.0f32; 2];
        quantize_i2s(&input, &mut q_dispatch, &mut s_dispatch, &cfg).unwrap();

        assert_eq!(q_scalar, q_dispatch, "i2s quant mismatch");
        assert_approx_slice(&s_scalar, &s_dispatch, 1e-6, "i2s scales");

        let mut out_scalar = vec![0.0f32; 64];
        let mut out_dispatch = vec![0.0f32; 64];
        dequantize_i2s_scalar(&q_scalar, &mut out_scalar, &s_scalar, &cfg).unwrap();
        dequantize_i2s(&q_dispatch, &mut out_dispatch, &s_dispatch, &cfg).unwrap();
        assert_approx_slice(&out_scalar, &out_dispatch, 1e-6, "i2s dequant");
    }

    #[test]
    fn absmax_simd_matches_scalar() {
        let cfg = cfg_with_block(32);
        let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.15).collect();

        let mut q_scalar = vec![0i8; 64];
        let mut s_scalar = vec![0.0f32; 2];
        quantize_absmax_scalar(&input, &mut q_scalar, &mut s_scalar, &cfg).unwrap();

        let mut q_dispatch = vec![0i8; 64];
        let mut s_dispatch = vec![0.0f32; 2];
        quantize_absmax(&input, &mut q_dispatch, &mut s_dispatch, &cfg).unwrap();

        assert_eq!(q_scalar, q_dispatch, "absmax quant mismatch");
        assert_approx_slice(&s_scalar, &s_dispatch, 1e-6, "absmax scales");

        let mut out_scalar = vec![0.0f32; 64];
        let mut out_dispatch = vec![0.0f32; 64];
        dequantize_absmax_scalar(&q_scalar, &mut out_scalar, &s_scalar, &cfg).unwrap();
        dequantize_absmax(&q_dispatch, &mut out_dispatch, &s_dispatch, &cfg).unwrap();
        assert_approx_slice(&out_scalar, &out_dispatch, 1e-6, "absmax dequant");
    }

    #[test]
    fn i2s_output_only_ternary_values() {
        let cfg = cfg_with_block(8);
        let input: Vec<f32> = (0..256).map(|i| ((i * 17 + 3) % 200) as f32 - 100.0).collect();
        let mut q = vec![0i8; 256];
        let mut scales = vec![0.0f32; 32];
        quantize_i2s(&input, &mut q, &mut scales, &cfg).unwrap();
        for (i, &v) in q.iter().enumerate() {
            assert!(v >= -1 && v <= 1, "dispatch ternary range[{i}]: {v}");
        }
    }

    // ── Edge case tests ─────────────────────────────────────────────────

    #[test]
    fn single_block_exact() {
        let cfg = cfg_with_block(8);
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.0, 0.0, 0.75, -0.75];
        let mut q = vec![0i8; 8];
        let mut scales = vec![0.0f32; 1];
        quantize_i2s(&input, &mut q, &mut scales, &cfg).unwrap();
        assert!(approx_eq(scales[0], 1.0, 1e-6));
    }

    #[test]
    fn large_block_size_256() {
        let cfg = cfg_with_block(256);
        let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let mut q = vec![0i8; 256];
        let mut scales = vec![0.0f32; 1];
        quantize_i2s(&input, &mut q, &mut scales, &cfg).unwrap();
        let mut out = vec![0.0f32; 256];
        dequantize_i2s(&q, &mut out, &scales, &cfg).unwrap();
        for &v in &out {
            assert!(!v.is_nan());
        }
    }

    #[test]
    fn single_element_block_boundary() {
        // block_size=8 but only 8 elements — should work as exactly 1 block
        let cfg = cfg_with_block(8);
        let input = vec![42.0; 8];
        let mut q = vec![0i8; 8];
        let mut scales = vec![0.0f32; 1];
        quantize_absmax(&input, &mut q, &mut scales, &cfg).unwrap();
        assert!(q.iter().all(|&v| v == 127));
    }

    #[test]
    fn absmax_roundtrip_large() {
        let cfg = cfg_with_block(64);
        let input: Vec<f32> = (0..256).map(|i| ((i as f32) - 128.0) * 0.1).collect();
        let mut q = vec![0i8; 256];
        let mut scales = vec![0.0f32; 4];
        quantize_absmax(&input, &mut q, &mut scales, &cfg).unwrap();

        let mut out = vec![0.0f32; 256];
        dequantize_absmax(&q, &mut out, &scales, &cfg).unwrap();

        for i in 0..256 {
            let blk_idx = i / 64;
            let tol = scales[blk_idx] / 2.0;
            assert!(
                approx_eq(input[i], out[i], tol),
                "large roundtrip[{i}]: {} vs {} (tol={tol})",
                input[i],
                out[i]
            );
        }
    }

    #[test]
    fn i2s_dispatch_large() {
        let cfg = cfg_with_block(64);
        let input: Vec<f32> = (0..256).map(|i| ((i as f32) - 128.0) * 0.1).collect();
        let mut q = vec![0i8; 256];
        let mut scales = vec![0.0f32; 4];
        quantize_i2s(&input, &mut q, &mut scales, &cfg).unwrap();

        let mut out = vec![0.0f32; 256];
        dequantize_i2s(&q, &mut out, &scales, &cfg).unwrap();

        let s_max = scales.iter().cloned().fold(0.0f32, f32::max);
        for &v in &out {
            assert!(v.abs() <= s_max + 1e-6);
        }
    }

    #[test]
    fn absmax_roundtrip_mixed_magnitudes() {
        let cfg = cfg_with_block(8);
        let input = vec![100.0, -0.001, 50.0, -50.0, 0.0, 0.01, -100.0, 25.0];
        let mut q = vec![0i8; 8];
        let mut scales = vec![0.0f32; 1];
        quantize_absmax(&input, &mut q, &mut scales, &cfg).unwrap();

        let mut out = vec![0.0f32; 8];
        dequantize_absmax(&q, &mut out, &scales, &cfg).unwrap();

        for (i, (&orig, &deq)) in input.iter().zip(out.iter()).enumerate() {
            let tol = scales[0] / 2.0;
            assert!(
                approx_eq(orig, deq, tol),
                "mixed magnitudes[{i}]: {orig} vs {deq} (tol={tol})"
            );
        }
    }

    // ── Config tests ────────────────────────────────────────────────────

    #[test]
    fn default_config_values() {
        let cfg = SimdQuantConfig::default();
        assert_eq!(cfg.block_size, 64);
        assert_eq!(cfg.scale_type, ScaleType::AbsMax);
        assert_eq!(cfg.rounding_mode, RoundingMode::Nearest);
        assert!(approx_eq(cfg.ternary_threshold, 0.5, 1e-6));
    }

    // ── Dequant-only error path tests ───────────────────────────────────

    #[test]
    fn dequant_i2s_output_too_small() {
        let cfg = cfg_with_block(8);
        let q = vec![1i8; 8];
        let scales = vec![1.0f32; 1];
        let mut out = vec![0.0f32; 4]; // too small
        let err = dequantize_i2s_scalar(&q, &mut out, &scales, &cfg).unwrap_err();
        assert!(matches!(err, SimdQuantError::OutputBufferTooSmall { .. }));
    }

    #[test]
    fn dequant_absmax_scale_too_small() {
        let cfg = cfg_with_block(8);
        let q = vec![1i8; 16];
        let scales = vec![1.0f32; 1]; // need 2
        let mut out = vec![0.0f32; 16];
        let err = dequantize_absmax_scalar(&q, &mut out, &scales, &cfg).unwrap_err();
        assert!(matches!(err, SimdQuantError::ScaleBufferTooSmall { .. }));
    }

    // ── Known-value tests ───────────────────────────────────────────────

    #[test]
    fn dequant_i2s_known_values() {
        let cfg = cfg_with_block(8);
        let q = vec![1, -1, 0, 1, -1, 0, 1, -1];
        let scales = vec![2.5f32];
        let mut out = vec![0.0f32; 8];
        dequantize_i2s(&q, &mut out, &scales, &cfg).unwrap();
        let expected = [2.5, -2.5, 0.0, 2.5, -2.5, 0.0, 2.5, -2.5];
        assert_approx_slice(&out, &expected, 1e-6, "known i2s dequant");
    }

    #[test]
    fn dequant_absmax_known_values() {
        let cfg = cfg_with_block(8);
        let q = vec![127, -127, 0, 64, -64, 1, -1, 0];
        let scale = 2.0 / 127.0;
        let scales = vec![scale];
        let mut out = vec![0.0f32; 8];
        dequantize_absmax(&q, &mut out, &scales, &cfg).unwrap();
        for (i, &v) in q.iter().enumerate() {
            let expected = v as f32 * scale;
            assert!(approx_eq(out[i], expected, 1e-6), "known absmax dequant at {i}");
        }
    }

    #[test]
    fn absmax_symmetry_positive_negative() {
        let cfg = cfg_with_block(8);
        let input = vec![3.0, -3.0, 2.0, -2.0, 1.0, -1.0, 0.5, -0.5];
        let mut q = vec![0i8; 8];
        let mut scales = vec![0.0f32; 1];
        quantize_absmax(&input, &mut q, &mut scales, &cfg).unwrap();
        assert_eq!(q[0], -q[1], "symmetry: 3.0 vs -3.0");
        assert_eq!(q[2], -q[3], "symmetry: 2.0 vs -2.0");
        assert_eq!(q[4], -q[5], "symmetry: 1.0 vs -1.0");
        assert_eq!(q[6], -q[7], "symmetry: 0.5 vs -0.5");
    }

    // ── Additional coverage tests ───────────────────────────────────────

    #[test]
    fn i2s_threshold_boundary() {
        // Values exactly at threshold*scale should map to 0
        let cfg = SimdQuantConfig { block_size: 8, ternary_threshold: 0.5, ..Default::default() };
        let input = vec![10.0, 5.0, 4.99, -4.99, -5.0, -10.0, 0.0, 0.0];
        // absmax = 10, threshold = 0.5*10 = 5.0
        let mut q = vec![0i8; 8];
        let mut scales = vec![0.0f32; 1];
        quantize_i2s_scalar(&input, &mut q, &mut scales, &cfg).unwrap();
        assert_eq!(q[0], 1, "10.0 > 5.0 → 1");
        assert_eq!(q[1], 0, "5.0 is not > 5.0 → 0");
        assert_eq!(q[2], 0, "4.99 < 5.0 → 0");
        assert_eq!(q[4], 0, "-5.0 is not < -5.0 → 0");
        assert_eq!(q[5], -1, "-10.0 < -5.0 → -1");
    }

    #[test]
    fn absmax_scale_computed_correctly() {
        let cfg = cfg_with_block(8);
        let input = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0];
        let mut q = vec![0i8; 8];
        let mut scales = vec![0.0f32; 1];
        quantize_absmax_scalar(&input, &mut q, &mut scales, &cfg).unwrap();
        let expected_scale = 5.0 / 127.0;
        assert!(
            approx_eq(scales[0], expected_scale, 1e-7),
            "scale: {} vs expected {}",
            scales[0],
            expected_scale
        );
        assert_eq!(q[7], 127, "max value should quantize to 127");
    }

    #[test]
    fn i2s_dequant_dispatch_error_propagation() {
        let cfg = cfg_with_block(0);
        let q = vec![1i8; 8];
        let scales = vec![1.0f32; 1];
        let mut out = vec![0.0f32; 8];
        assert!(dequantize_i2s(&q, &mut out, &scales, &cfg).is_err());
    }

    #[test]
    fn absmax_dequant_dispatch_error_propagation() {
        let cfg = cfg_with_block(0);
        let q = vec![1i8; 8];
        let scales = vec![1.0f32; 1];
        let mut out = vec![0.0f32; 8];
        assert!(dequantize_absmax(&q, &mut out, &scales, &cfg).is_err());
    }

    #[test]
    fn i2s_dispatch_multi_block_64() {
        let cfg = cfg_with_block(8);
        let input: Vec<f32> = (0..512).map(|i| ((i % 37) as f32 - 18.0) * 0.3).collect();
        let mut q = vec![0i8; 512];
        let mut scales = vec![0.0f32; 64];
        quantize_i2s(&input, &mut q, &mut scales, &cfg).unwrap();
        let mut out = vec![0.0f32; 512];
        dequantize_i2s(&q, &mut out, &scales, &cfg).unwrap();
        for &v in &q {
            assert!(v >= -1 && v <= 1);
        }
    }

    #[test]
    fn absmax_dispatch_multi_block_64() {
        let cfg = cfg_with_block(8);
        let input: Vec<f32> = (0..512).map(|i| ((i % 37) as f32 - 18.0) * 0.3).collect();
        let mut q = vec![0i8; 512];
        let mut scales = vec![0.0f32; 64];
        quantize_absmax(&input, &mut q, &mut scales, &cfg).unwrap();
        let mut out = vec![0.0f32; 512];
        dequantize_absmax(&q, &mut out, &scales, &cfg).unwrap();
        for i in 0..512 {
            let blk_idx = i / 8;
            // Max rounding error is one quantization step = scale
            let tol = scales[blk_idx];
            assert!(
                approx_eq(input[i], out[i], tol),
                "64-block roundtrip[{i}]: {} vs {}",
                input[i],
                out[i]
            );
        }
    }

    #[test]
    fn i2s_uniform_positive_all_ones() {
        let cfg = SimdQuantConfig { block_size: 8, ternary_threshold: 0.5, ..Default::default() };
        let input = vec![10.0; 8];
        let mut q = vec![0i8; 8];
        let mut scales = vec![0.0f32; 1];
        quantize_i2s(&input, &mut q, &mut scales, &cfg).unwrap();
        // All identical positive values: absmax=10, threshold=5; 10>5 → all +1
        assert!(q.iter().all(|&v| v == 1), "uniform positive → all +1");
    }

    #[test]
    fn i2s_uniform_negative_all_neg_ones() {
        let cfg = SimdQuantConfig { block_size: 8, ternary_threshold: 0.5, ..Default::default() };
        let input = vec![-10.0; 8];
        let mut q = vec![0i8; 8];
        let mut scales = vec![0.0f32; 1];
        quantize_i2s(&input, &mut q, &mut scales, &cfg).unwrap();
        assert!(q.iter().all(|&v| v == -1), "uniform negative → all -1");
    }

    #[test]
    fn error_clone_and_eq() {
        let e1 = SimdQuantError::NanInput;
        let e2 = e1.clone();
        assert_eq!(e1, e2);
        let e3 = SimdQuantError::InvalidBlockSize { input_len: 7, block_size: 4 };
        let e4 = e3.clone();
        assert_eq!(e3, e4);
    }
}
