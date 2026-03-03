//! NEON-optimized I2_S quantization and dequantization kernels.
//!
//! Provides aarch64 NEON SIMD acceleration with automatic scalar fallback
//! for platforms without NEON support.

use std::fmt;

use thiserror::Error;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Configuration for quantization operations.
#[derive(Debug, Clone)]
pub struct QuantizeConfig {
    /// Number of elements per quantization block.
    pub block_size: usize,
    /// Number of bits per quantized element (always 2 for I2_S).
    pub bits: u8,
    /// Use symmetric quantization (no zero-point offset).
    pub symmetric: bool,
}

impl Default for QuantizeConfig {
    fn default() -> Self {
        Self { block_size: 32, bits: 2, symmetric: true }
    }
}

/// Errors specific to the NEON quantization kernels.
#[derive(Error, Debug, PartialEq)]
pub enum QuantizeError {
    #[error("empty input")]
    EmptyInput,
    #[error("input length {len} is not a multiple of block_size {block_size}")]
    BlockAlignment { len: usize, block_size: usize },
    #[error("invalid block size {0} (must be > 0 and a multiple of 4)")]
    InvalidBlockSize(usize),
    #[error("output buffer too small: need {need}, got {got}")]
    OutputTooSmall { need: usize, got: usize },
    #[error("scale buffer too small: need {need}, got {got}")]
    ScaleTooSmall { need: usize, got: usize },
    #[error("all-zero block at index {0} (scale is zero)")]
    ZeroScale(usize),
}

/// Result of a dequantization operation.
#[derive(Debug, Clone, PartialEq)]
pub struct DequantizeResult {
    /// Reconstructed floating-point values.
    pub values: Vec<f32>,
    /// Number of blocks that were processed.
    pub blocks_processed: usize,
    /// Maximum absolute reconstruction error observed (best-effort).
    pub max_abs_error: f32,
}

impl fmt::Display for DequantizeResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DequantizeResult {{ blocks: {}, max_abs_error: {:.6} }}",
            self.blocks_processed, self.max_abs_error
        )
    }
}

// ---------------------------------------------------------------------------
// I2_S encoding helpers (2-bit signed: 0b00→0, 0b01→+1, 0b11→−1)
// ---------------------------------------------------------------------------

const I2S_LUT: [f32; 4] = [0.0, 1.0, 0.0, -1.0];

/// Map a float to its I2_S 2-bit code.
#[inline(always)]
fn encode_i2s(v: f32) -> u8 {
    if v > 0.0 {
        0b01
    } else if v < 0.0 {
        0b11
    } else {
        0b00
    }
}

/// Decode a 2-bit I2_S code to float.
#[inline(always)]
fn decode_i2s(code: u8) -> f32 {
    I2S_LUT[(code & 0x03) as usize]
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate_config(cfg: &QuantizeConfig) -> Result<(), QuantizeError> {
    if cfg.block_size == 0 || !cfg.block_size.is_multiple_of(4) {
        return Err(QuantizeError::InvalidBlockSize(cfg.block_size));
    }
    Ok(())
}

fn validate_quantize_buffers(
    input: &[f32],
    output: &[u8],
    scales: &[f32],
    cfg: &QuantizeConfig,
) -> Result<usize, QuantizeError> {
    if input.is_empty() {
        return Err(QuantizeError::EmptyInput);
    }
    if !input.len().is_multiple_of(cfg.block_size) {
        return Err(QuantizeError::BlockAlignment { len: input.len(), block_size: cfg.block_size });
    }
    let n_blocks = input.len() / cfg.block_size;
    let need_bytes = packed_bytes(input.len());
    if output.len() < need_bytes {
        return Err(QuantizeError::OutputTooSmall { need: need_bytes, got: output.len() });
    }
    if scales.len() < n_blocks {
        return Err(QuantizeError::ScaleTooSmall { need: n_blocks, got: scales.len() });
    }
    Ok(n_blocks)
}

fn validate_dequantize_buffers(
    packed: &[u8],
    scales: &[f32],
    cfg: &QuantizeConfig,
) -> Result<usize, QuantizeError> {
    if scales.is_empty() {
        return Err(QuantizeError::EmptyInput);
    }
    let n_blocks = scales.len();
    let need_bytes = packed_bytes(n_blocks * cfg.block_size);
    if packed.len() < need_bytes {
        return Err(QuantizeError::OutputTooSmall { need: need_bytes, got: packed.len() });
    }
    Ok(n_blocks)
}

/// Number of bytes needed to store `n_elements` at 2 bits each.
#[inline]
fn packed_bytes(n_elements: usize) -> usize {
    (n_elements + 3) / 4
}

// ---------------------------------------------------------------------------
// Public API – full tensor quantize / dequantize
// ---------------------------------------------------------------------------

/// Quantize an f32 slice to I2_S packed bytes using NEON when available.
///
/// `output` must have at least `ceil(input.len() / 4)` bytes.
/// `scales` must have at least `input.len() / block_size` entries.
pub fn quantize_i2s_neon(
    input: &[f32],
    output: &mut [u8],
    scales: &mut [f32],
    cfg: &QuantizeConfig,
) -> Result<(), QuantizeError> {
    validate_config(cfg)?;
    let n_blocks = validate_quantize_buffers(input, output, scales, cfg)?;

    for bi in 0..n_blocks {
        let start = bi * cfg.block_size;
        let block = &input[start..start + cfg.block_size];
        let out_start = start / 4;
        let out_end = out_start + cfg.block_size / 4;
        let out_slice = &mut output[out_start..out_end];
        quantize_block_neon(block, out_slice, &mut scales[bi..bi + 1])?;
    }
    Ok(())
}

/// Dequantize I2_S packed bytes back to f32 using NEON when available.
pub fn dequantize_i2s_neon(
    packed: &[u8],
    scales: &[f32],
    cfg: &QuantizeConfig,
) -> Result<DequantizeResult, QuantizeError> {
    validate_config(cfg)?;
    let n_blocks = validate_dequantize_buffers(packed, scales, cfg)?;
    let total = n_blocks * cfg.block_size;
    let mut values = vec![0.0f32; total];
    let mut max_err: f32 = 0.0;

    for (bi, &block_scale) in scales.iter().enumerate().take(n_blocks) {
        let start = bi * cfg.block_size;
        let p_start = start / 4;
        let p_end = p_start + cfg.block_size / 4;
        let res = dequantize_block_neon(&packed[p_start..p_end], block_scale, cfg.block_size)?;
        values[start..start + cfg.block_size].copy_from_slice(&res.values);
        if res.max_abs_error > max_err {
            max_err = res.max_abs_error;
        }
    }

    Ok(DequantizeResult { values, blocks_processed: n_blocks, max_abs_error: max_err })
}

// ---------------------------------------------------------------------------
// Public API – single-block operations
// ---------------------------------------------------------------------------

/// Quantize a single block (length must be a multiple of 4).
///
/// Writes the packed bytes into `output` and the block scale into `scale[0]`.
pub fn quantize_block_neon(
    block: &[f32],
    output: &mut [u8],
    scale: &mut [f32],
) -> Result<(), QuantizeError> {
    if block.is_empty() {
        return Err(QuantizeError::EmptyInput);
    }
    if !block.len().is_multiple_of(4) {
        return Err(QuantizeError::InvalidBlockSize(block.len()));
    }
    let need = block.len() / 4;
    if output.len() < need {
        return Err(QuantizeError::OutputTooSmall { need, got: output.len() });
    }
    if scale.is_empty() {
        return Err(QuantizeError::ScaleTooSmall { need: 1, got: 0 });
    }

    let s = compute_scale_neon(block);
    scale[0] = s;

    if s == 0.0 {
        output[..need].fill(0);
        return Ok(());
    }

    let inv_s = 1.0 / s;

    // Pack 4 elements per byte (2 bits each, LSB-first)
    for (i, chunk) in block.chunks_exact(4).enumerate() {
        let mut byte: u8 = 0;
        for (j, &v) in chunk.iter().enumerate() {
            let q = (v * inv_s).round();
            byte |= encode_i2s(q) << (j * 2);
        }
        output[i] = byte;
    }
    Ok(())
}

/// Dequantize a single block of packed I2_S bytes.
pub fn dequantize_block_neon(
    packed: &[u8],
    scale: f32,
    block_size: usize,
) -> Result<DequantizeResult, QuantizeError> {
    if block_size == 0 || !block_size.is_multiple_of(4) {
        return Err(QuantizeError::InvalidBlockSize(block_size));
    }
    let need = block_size / 4;
    if packed.len() < need {
        return Err(QuantizeError::OutputTooSmall { need, got: packed.len() });
    }

    let mut values = vec![0.0f32; block_size];

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: feature detection above guarantees NEON.
            unsafe {
                dequant_block_neon_inner(packed, scale, &mut values);
            }
            return Ok(DequantizeResult {
                values,
                blocks_processed: 1,
                max_abs_error: scale.abs(),
            });
        }
    }

    // Scalar fallback
    for (i, &byte) in packed.iter().enumerate().take(need) {
        for j in 0..4 {
            let code = (byte >> (j * 2)) & 0x03;
            values[i * 4 + j] = decode_i2s(code) * scale;
        }
    }
    Ok(DequantizeResult { values, blocks_processed: 1, max_abs_error: scale.abs() })
}

/// Compute the I2_S scale factor for a block: `absmax(block)`.
pub fn compute_scale_neon(block: &[f32]) -> f32 {
    absmax_neon(block)
}

/// Return the maximum absolute value of a slice, using NEON when available.
pub fn absmax_neon(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: feature detection above guarantees NEON.
            return unsafe { absmax_neon_inner(data) };
        }
    }

    // Scalar fallback
    absmax_scalar(data)
}

// ---------------------------------------------------------------------------
// Scalar helpers (always compiled)
// ---------------------------------------------------------------------------

fn absmax_scalar(data: &[f32]) -> f32 {
    data.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()))
}

// ---------------------------------------------------------------------------
// NEON intrinsics (aarch64 only)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon_impl {
    use std::arch::aarch64::*;

    /// NEON-accelerated absmax over an f32 slice.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn absmax_neon_inner(data: &[f32]) -> f32 {
        let len = data.len();
        let ptr = data.as_ptr();

        let sign_mask = vdupq_n_u32(0x7FFF_FFFF);
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0usize;

        // Process 16 elements per iteration (4 × float32x4)
        while i + 16 <= len {
            let a = vreinterpretq_f32_u32(vandq_u32(
                vreinterpretq_u32_f32(vld1q_f32(ptr.add(i))),
                sign_mask,
            ));
            let b = vreinterpretq_f32_u32(vandq_u32(
                vreinterpretq_u32_f32(vld1q_f32(ptr.add(i + 4))),
                sign_mask,
            ));
            let c = vreinterpretq_f32_u32(vandq_u32(
                vreinterpretq_u32_f32(vld1q_f32(ptr.add(i + 8))),
                sign_mask,
            ));
            let d = vreinterpretq_f32_u32(vandq_u32(
                vreinterpretq_u32_f32(vld1q_f32(ptr.add(i + 12))),
                sign_mask,
            ));
            acc = vmaxq_f32(acc, vmaxq_f32(vmaxq_f32(a, b), vmaxq_f32(c, d)));
            i += 16;
        }

        // Process 4 elements
        while i + 4 <= len {
            let v = vreinterpretq_f32_u32(vandq_u32(
                vreinterpretq_u32_f32(vld1q_f32(ptr.add(i))),
                sign_mask,
            ));
            acc = vmaxq_f32(acc, v);
            i += 4;
        }

        // Horizontal max
        let mut result = vmaxvq_f32(acc);

        // Scalar tail
        while i < len {
            let a = (*ptr.add(i)).abs();
            if a > result {
                result = a;
            }
            i += 1;
        }
        result
    }

    /// NEON-accelerated dequantization of a packed I2_S block.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn dequant_block_neon_inner(packed: &[u8], scale: f32, out: &mut [f32]) {
        let scale_v = vdupq_n_f32(scale);
        let lut: [f32; 4] = [0.0, 1.0, 0.0, -1.0];
        let mut oi = 0usize;

        for &byte in packed {
            // Decode 4 elements from one byte
            let c0 = (byte & 0x03) as usize;
            let c1 = ((byte >> 2) & 0x03) as usize;
            let c2 = ((byte >> 4) & 0x03) as usize;
            let c3 = ((byte >> 6) & 0x03) as usize;

            let raw = [lut[c0], lut[c1], lut[c2], lut[c3]];
            let raw_v = vld1q_f32(raw.as_ptr());
            let scaled = vmulq_f32(raw_v, scale_v);
            vst1q_f32(out.as_mut_ptr().add(oi), scaled);
            oi += 4;
            if oi >= out.len() {
                break;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
use neon_impl::{absmax_neon_inner, dequant_block_neon_inner};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ===================================================================
    // QuantizeConfig tests
    // ===================================================================

    #[test]
    fn config_default_values() {
        let cfg = QuantizeConfig::default();
        assert_eq!(cfg.block_size, 32);
        assert_eq!(cfg.bits, 2);
        assert!(cfg.symmetric);
    }

    #[test]
    fn config_custom_block_size() {
        let cfg = QuantizeConfig { block_size: 64, ..Default::default() };
        assert_eq!(cfg.block_size, 64);
    }

    #[test]
    fn config_clone_eq() {
        let a = QuantizeConfig::default();
        let b = a.clone();
        assert_eq!(a.block_size, b.block_size);
        assert_eq!(a.bits, b.bits);
        assert_eq!(a.symmetric, b.symmetric);
    }

    // ===================================================================
    // QuantizeError tests
    // ===================================================================

    #[test]
    fn error_display_empty_input() {
        assert_eq!(QuantizeError::EmptyInput.to_string(), "empty input");
    }

    #[test]
    fn error_display_block_alignment() {
        let e = QuantizeError::BlockAlignment { len: 10, block_size: 8 };
        assert!(e.to_string().contains("10"));
        assert!(e.to_string().contains("8"));
    }

    #[test]
    fn error_display_invalid_block_size() {
        let e = QuantizeError::InvalidBlockSize(3);
        assert!(e.to_string().contains("3"));
    }

    #[test]
    fn error_display_output_too_small() {
        let e = QuantizeError::OutputTooSmall { need: 16, got: 4 };
        assert!(e.to_string().contains("16"));
        assert!(e.to_string().contains("4"));
    }

    #[test]
    fn error_display_scale_too_small() {
        let e = QuantizeError::ScaleTooSmall { need: 2, got: 0 };
        assert!(e.to_string().contains("2"));
    }

    #[test]
    fn error_display_zero_scale() {
        let e = QuantizeError::ZeroScale(5);
        assert!(e.to_string().contains("5"));
    }

    #[test]
    fn error_eq() {
        assert_eq!(QuantizeError::EmptyInput, QuantizeError::EmptyInput);
        assert_ne!(QuantizeError::EmptyInput, QuantizeError::ZeroScale(0));
    }

    // ===================================================================
    // DequantizeResult tests
    // ===================================================================

    #[test]
    fn dequantize_result_display() {
        let r = DequantizeResult { values: vec![1.0], blocks_processed: 1, max_abs_error: 0.5 };
        let s = r.to_string();
        assert!(s.contains("blocks: 1"));
        assert!(s.contains("0.500000"));
    }

    #[test]
    fn dequantize_result_clone() {
        let r =
            DequantizeResult { values: vec![1.0, -1.0], blocks_processed: 2, max_abs_error: 0.1 };
        let c = r.clone();
        assert_eq!(r.values, c.values);
        assert_eq!(r.blocks_processed, c.blocks_processed);
    }

    // ===================================================================
    // encode / decode helpers
    // ===================================================================

    #[test]
    fn encode_positive() {
        assert_eq!(encode_i2s(1.0), 0b01);
        assert_eq!(encode_i2s(0.5), 0b01);
    }

    #[test]
    fn encode_negative() {
        assert_eq!(encode_i2s(-1.0), 0b11);
        assert_eq!(encode_i2s(-0.3), 0b11);
    }

    #[test]
    fn encode_zero() {
        assert_eq!(encode_i2s(0.0), 0b00);
    }

    #[test]
    fn decode_lut_values() {
        assert_eq!(decode_i2s(0b00), 0.0);
        assert_eq!(decode_i2s(0b01), 1.0);
        assert_eq!(decode_i2s(0b10), 0.0);
        assert_eq!(decode_i2s(0b11), -1.0);
    }

    #[test]
    fn encode_decode_roundtrip() {
        for &v in &[1.0f32, -1.0, 0.0] {
            let code = encode_i2s(v);
            assert_eq!(decode_i2s(code), v);
        }
    }

    // ===================================================================
    // absmax_neon / compute_scale_neon
    // ===================================================================

    #[test]
    fn absmax_empty() {
        assert_eq!(absmax_neon(&[]), 0.0);
    }

    #[test]
    fn absmax_single() {
        assert_eq!(absmax_neon(&[-3.5]), 3.5);
    }

    #[test]
    fn absmax_positive() {
        assert_eq!(absmax_neon(&[1.0, 2.0, 3.0, 4.0]), 4.0);
    }

    #[test]
    fn absmax_negative_dominates() {
        assert_eq!(absmax_neon(&[1.0, -5.0, 3.0, 2.0]), 5.0);
    }

    #[test]
    fn absmax_all_zeros() {
        assert_eq!(absmax_neon(&[0.0; 16]), 0.0);
    }

    #[test]
    fn absmax_large_input() {
        let v: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.1).collect();
        let expected = absmax_scalar(&v);
        assert!((absmax_neon(&v) - expected).abs() < 1e-6);
    }

    #[test]
    fn absmax_tail_element_is_max() {
        // Ensure scalar tail is handled (length not multiple of 4)
        let v = [1.0, 2.0, 3.0, 4.0, 10.0];
        assert_eq!(absmax_neon(&v), 10.0);
    }

    #[test]
    fn compute_scale_matches_absmax() {
        let block = [1.5, -2.0, 0.0, 0.5];
        assert_eq!(compute_scale_neon(&block), absmax_neon(&block));
    }

    // ===================================================================
    // quantize_block_neon
    // ===================================================================

    #[test]
    fn block_quantize_simple() {
        let block = [1.0f32, -1.0, 0.0, 1.0];
        let mut out = [0u8; 1];
        let mut scale = [0.0f32; 1];
        quantize_block_neon(&block, &mut out, &mut scale).unwrap();
        assert_eq!(scale[0], 1.0);
    }

    #[test]
    fn block_quantize_all_zeros() {
        let block = [0.0f32; 8];
        let mut out = [0u8; 2];
        let mut scale = [0.0f32; 1];
        quantize_block_neon(&block, &mut out, &mut scale).unwrap();
        assert_eq!(scale[0], 0.0);
        assert_eq!(out, [0, 0]);
    }

    #[test]
    fn block_quantize_all_positive() {
        let block = [0.5f32, 0.7, 0.3, 0.9];
        let mut out = [0u8; 1];
        let mut scale = [0.0f32; 1];
        quantize_block_neon(&block, &mut out, &mut scale).unwrap();
        assert_eq!(scale[0], 0.9);
        // 0.5/0.9→round→1→01, 0.7/0.9→round→1→01, 0.3/0.9→round→0→00, 0.9/0.9→1→01
        assert_eq!(out[0], 0b01_00_01_01);
    }

    #[test]
    fn block_quantize_all_negative() {
        let block = [-0.5f32, -0.7, -0.3, -0.9];
        let mut out = [0u8; 1];
        let mut scale = [0.0f32; 1];
        quantize_block_neon(&block, &mut out, &mut scale).unwrap();
        // -0.5/0.9→round→-1→11, -0.7/0.9→round→-1→11, -0.3/0.9→round→0→00, -0.9/0.9→-1→11
        assert_eq!(out[0], 0b11_00_11_11);
    }

    #[test]
    fn block_quantize_empty_err() {
        let e = quantize_block_neon(&[], &mut [0u8; 1], &mut [0.0f32; 1]);
        assert_eq!(e, Err(QuantizeError::EmptyInput));
    }

    #[test]
    fn block_quantize_bad_alignment_err() {
        let e = quantize_block_neon(&[1.0, 2.0, 3.0], &mut [0u8; 1], &mut [0.0f32; 1]);
        assert_eq!(e, Err(QuantizeError::InvalidBlockSize(3)));
    }

    #[test]
    fn block_quantize_output_too_small_err() {
        let e = quantize_block_neon(&[1.0; 8], &mut [0u8; 1], &mut [0.0f32; 1]);
        assert_eq!(e, Err(QuantizeError::OutputTooSmall { need: 2, got: 1 }));
    }

    #[test]
    fn block_quantize_scale_too_small_err() {
        let e = quantize_block_neon(&[1.0; 4], &mut [0u8; 1], &mut []);
        assert_eq!(e, Err(QuantizeError::ScaleTooSmall { need: 1, got: 0 }));
    }

    // ===================================================================
    // dequantize_block_neon
    // ===================================================================

    #[test]
    fn block_dequantize_simple() {
        // Packed: code for [+1, -1, 0, +1] → 0b01_00_11_01
        let packed = [0b01_00_11_01u8];
        let res = dequantize_block_neon(&packed, 2.0, 4).unwrap();
        assert_eq!(res.values, &[2.0, -2.0, 0.0, 2.0]);
        assert_eq!(res.blocks_processed, 1);
    }

    #[test]
    fn block_dequantize_zero_scale() {
        let packed = [0u8; 2];
        let res = dequantize_block_neon(&packed, 0.0, 8).unwrap();
        assert!(res.values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn block_dequantize_bad_block_size() {
        let e = dequantize_block_neon(&[0u8; 1], 1.0, 3);
        assert_eq!(e, Err(QuantizeError::InvalidBlockSize(3)));
    }

    #[test]
    fn block_dequantize_packed_too_small() {
        let e = dequantize_block_neon(&[0u8; 1], 1.0, 8);
        assert_eq!(e, Err(QuantizeError::OutputTooSmall { need: 2, got: 1 }));
    }

    // ===================================================================
    // Round-trip: quantize_block → dequantize_block
    // ===================================================================

    #[test]
    fn block_roundtrip_ternary() {
        let block = [1.0f32, -1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0];
        let mut packed = [0u8; 2];
        let mut scale = [0.0f32; 1];
        quantize_block_neon(&block, &mut packed, &mut scale).unwrap();

        let res = dequantize_block_neon(&packed, scale[0], 8).unwrap();
        assert_eq!(res.values, block);
    }

    #[test]
    fn block_roundtrip_scaled() {
        let block = [3.0f32, -3.0, 0.0, 3.0];
        let mut packed = [0u8; 1];
        let mut scale = [0.0f32; 1];
        quantize_block_neon(&block, &mut packed, &mut scale).unwrap();
        assert_eq!(scale[0], 3.0);

        let res = dequantize_block_neon(&packed, scale[0], 4).unwrap();
        assert_eq!(res.values, block);
    }

    // ===================================================================
    // Full tensor quantize_i2s_neon / dequantize_i2s_neon
    // ===================================================================

    #[test]
    fn full_quantize_basic() {
        let cfg = QuantizeConfig { block_size: 4, ..Default::default() };
        let input = [1.0f32, -1.0, 0.0, 1.0, -0.5, 0.5, -0.5, 0.5];
        let mut packed = vec![0u8; 2];
        let mut scales = vec![0.0f32; 2];
        quantize_i2s_neon(&input, &mut packed, &mut scales, &cfg).unwrap();
        assert_eq!(scales[0], 1.0);
        assert_eq!(scales[1], 0.5);
    }

    #[test]
    fn full_dequantize_basic() {
        let cfg = QuantizeConfig { block_size: 4, ..Default::default() };
        let input = [1.0f32, -1.0, 0.0, 1.0];
        let mut packed = vec![0u8; 1];
        let mut scales = vec![0.0f32; 1];
        quantize_i2s_neon(&input, &mut packed, &mut scales, &cfg).unwrap();

        let res = dequantize_i2s_neon(&packed, &scales, &cfg).unwrap();
        assert_eq!(res.values, input);
    }

    #[test]
    fn full_roundtrip_multi_block() {
        let cfg = QuantizeConfig { block_size: 8, ..Default::default() };
        let input: Vec<f32> = (0..32)
            .map(|i| match i % 3 {
                0 => 1.0,
                1 => -1.0,
                _ => 0.0,
            })
            .collect();
        let n_blocks = input.len() / cfg.block_size;
        let mut packed = vec![0u8; input.len() / 4];
        let mut scales = vec![0.0f32; n_blocks];

        quantize_i2s_neon(&input, &mut packed, &mut scales, &cfg).unwrap();
        let res = dequantize_i2s_neon(&packed, &scales, &cfg).unwrap();

        assert_eq!(res.blocks_processed, n_blocks);
        assert_eq!(res.values, input);
    }

    #[test]
    fn full_quantize_empty_err() {
        let cfg = QuantizeConfig::default();
        let e = quantize_i2s_neon(&[], &mut [], &mut [], &cfg);
        assert_eq!(e, Err(QuantizeError::EmptyInput));
    }

    #[test]
    fn full_quantize_alignment_err() {
        let cfg = QuantizeConfig { block_size: 8, ..Default::default() };
        let input = [1.0f32; 10]; // not multiple of 8
        let e = quantize_i2s_neon(&input, &mut [0u8; 3], &mut [0.0f32; 2], &cfg);
        assert_eq!(e, Err(QuantizeError::BlockAlignment { len: 10, block_size: 8 }));
    }

    #[test]
    fn full_quantize_invalid_block_size_zero() {
        let cfg = QuantizeConfig { block_size: 0, ..Default::default() };
        let e = quantize_i2s_neon(&[1.0; 4], &mut [0u8; 1], &mut [0.0; 1], &cfg);
        assert_eq!(e, Err(QuantizeError::InvalidBlockSize(0)));
    }

    #[test]
    fn full_quantize_invalid_block_size_odd() {
        let cfg = QuantizeConfig { block_size: 5, ..Default::default() };
        let e = quantize_i2s_neon(&[1.0; 5], &mut [0u8; 2], &mut [0.0; 1], &cfg);
        assert_eq!(e, Err(QuantizeError::InvalidBlockSize(5)));
    }

    #[test]
    fn full_dequantize_empty_err() {
        let cfg = QuantizeConfig::default();
        let e = dequantize_i2s_neon(&[], &[], &cfg);
        assert_eq!(e, Err(QuantizeError::EmptyInput));
    }

    // ===================================================================
    // Validation helpers
    // ===================================================================

    #[test]
    fn validate_config_valid() {
        assert!(validate_config(&QuantizeConfig::default()).is_ok());
        assert!(validate_config(&QuantizeConfig { block_size: 256, ..Default::default() }).is_ok());
    }

    #[test]
    fn validate_config_invalid_zero() {
        let e = validate_config(&QuantizeConfig { block_size: 0, ..Default::default() });
        assert_eq!(e, Err(QuantizeError::InvalidBlockSize(0)));
    }

    #[test]
    fn validate_config_invalid_not_mult4() {
        let e = validate_config(&QuantizeConfig { block_size: 7, ..Default::default() });
        assert_eq!(e, Err(QuantizeError::InvalidBlockSize(7)));
    }

    #[test]
    fn packed_bytes_calc() {
        assert_eq!(packed_bytes(0), 0);
        assert_eq!(packed_bytes(1), 1);
        assert_eq!(packed_bytes(4), 1);
        assert_eq!(packed_bytes(5), 2);
        assert_eq!(packed_bytes(8), 2);
        assert_eq!(packed_bytes(32), 8);
    }

    // ===================================================================
    // Edge cases
    // ===================================================================

    #[test]
    fn absmax_scalar_agrees_with_neon() {
        let data = [0.1, -3.7, 2.5, 0.0, -1.0, 4.2, 0.0, -0.1];
        let scalar = absmax_scalar(&data);
        let neon_val = absmax_neon(&data);
        assert!((scalar - neon_val).abs() < 1e-6);
    }

    #[test]
    fn absmax_single_negative() {
        assert_eq!(absmax_neon(&[-42.0]), 42.0);
    }

    #[test]
    fn large_tensor_roundtrip() {
        let cfg = QuantizeConfig { block_size: 32, ..Default::default() };
        // Use only ternary values with uniform magnitude per block for exact round-trip
        let input: Vec<f32> = (0..256)
            .map(|i| match i % 3 {
                0 => 1.0,
                1 => -1.0,
                _ => 0.0,
            })
            .collect();
        let n_blocks = input.len() / cfg.block_size;
        let mut packed = vec![0u8; input.len() / 4];
        let mut scales = vec![0.0f32; n_blocks];

        quantize_i2s_neon(&input, &mut packed, &mut scales, &cfg).unwrap();
        let res = dequantize_i2s_neon(&packed, &scales, &cfg).unwrap();

        // All values are exactly representable in I2_S ternary, so round-trip is exact.
        assert_eq!(res.values, input);
        assert_eq!(res.blocks_processed, n_blocks);
    }

    #[test]
    fn quantize_non_ternary_values() {
        // Values that aren't exactly {-1,0,+1}*scale still produce valid ternary
        let block = [0.3f32, -0.8, 0.01, -0.01];
        let mut packed = [0u8; 1];
        let mut scale = [0.0f32; 1];
        quantize_block_neon(&block, &mut packed, &mut scale).unwrap();
        // 0.3/0.8 ≈ 0.375 → rounds to 0 → code 0b00
        // -0.8/0.8 = -1.0 → code 0b11
        // 0.01/0.8 ≈ 0.0125 → rounds to 0 → code 0b00
        // -0.01/0.8 ≈ -0.0125 → rounds to 0 → code 0b00
        // Expected packed byte: 0b00_00_11_00
        assert_eq!(packed[0], 0b00_00_11_00);
    }

    #[test]
    fn dequantize_negative_scale() {
        // Negative scales should still work correctly
        let packed = [0b01u8]; // code 01 → +1
        let res = dequantize_block_neon(&packed, -2.0, 4).unwrap();
        assert_eq!(res.values[0], -2.0);
    }

    #[test]
    fn block_size_four_minimum() {
        let cfg = QuantizeConfig { block_size: 4, ..Default::default() };
        let input = [1.0f32, -1.0, 0.0, 1.0];
        let mut packed = vec![0u8; 1];
        let mut scales = vec![0.0f32; 1];
        quantize_i2s_neon(&input, &mut packed, &mut scales, &cfg).unwrap();
        let res = dequantize_i2s_neon(&packed, &scales, &cfg).unwrap();
        assert_eq!(res.values, input);
    }

    #[test]
    fn block_size_256() {
        let cfg = QuantizeConfig { block_size: 256, ..Default::default() };
        let input: Vec<f32> = (0..256).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let mut packed = vec![0u8; 64];
        let mut scales = vec![0.0f32; 1];
        quantize_i2s_neon(&input, &mut packed, &mut scales, &cfg).unwrap();
        let res = dequantize_i2s_neon(&packed, &scales, &cfg).unwrap();
        assert_eq!(res.values, input);
    }

    // ===================================================================
    // proptest properties
    // ===================================================================

    mod proptests {
        use super::super::*;
        use proptest::prelude::*;

        /// Generate a block of ternary values (exactly representable in I2_S).
        fn ternary_block(block_size: usize) -> impl Strategy<Value = Vec<f32>> {
            proptest::collection::vec(prop_oneof![Just(0.0f32), Just(1.0), Just(-1.0)], block_size)
        }

        proptest! {
            /// Ternary values survive a full round-trip with no error.
            #[test]
            fn roundtrip_exact_ternary(block in ternary_block(32)) {
                let cfg = QuantizeConfig { block_size: 32, ..Default::default() };
                let mut packed = vec![0u8; 8];
                let mut scales = vec![0.0f32; 1];

                quantize_i2s_neon(&block, &mut packed, &mut scales, &cfg).unwrap();
                let res = dequantize_i2s_neon(&packed, &scales, &cfg).unwrap();

                // Scale is either 0 (all-zero block) or 1 (at least one ±1)
                prop_assert!(scales[0] == 0.0 || scales[0] == 1.0);
                prop_assert_eq!(res.values, block);
            }

            /// absmax is non-negative and >= every |element|.
            #[test]
            fn absmax_is_upper_bound(data in proptest::collection::vec(-100.0f32..100.0, 1..256)) {
                let m = absmax_neon(&data);
                prop_assert!(m >= 0.0);
                for &v in &data {
                    prop_assert!(m >= v.abs() - 1e-6, "absmax {} < |{}|", m, v);
                }
            }

            /// Dequantized values are always in {-scale, 0, +scale}.
            #[test]
            fn dequant_values_are_ternary(
                block in ternary_block(32),
            ) {
                let cfg = QuantizeConfig { block_size: 32, ..Default::default() };
                let mut packed = vec![0u8; 8];
                let mut scales = vec![0.0f32; 1];
                quantize_i2s_neon(&block, &mut packed, &mut scales, &cfg).unwrap();
                let res = dequantize_i2s_neon(&packed, &scales, &cfg).unwrap();
                let s = scales[0];
                for &v in &res.values {
                    prop_assert!(
                        (v - s).abs() < 1e-6 || (v + s).abs() < 1e-6 || v.abs() < 1e-6,
                        "value {} not in {{-{}, 0, {}}}", v, s, s,
                    );
                }
            }

            /// Scale equals absmax of input block.
            #[test]
            fn scale_equals_absmax(data in proptest::collection::vec(-50.0f32..50.0, 32..=32)) {
                let expected = data.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
                let actual = compute_scale_neon(&data);
                prop_assert!((expected - actual).abs() < 1e-6,
                    "expected {} got {}", expected, actual);
            }

            /// Output length invariants hold for any valid block count.
            #[test]
            fn output_length_invariant(n_blocks in 1usize..16) {
                let cfg = QuantizeConfig { block_size: 8, ..Default::default() };
                let input: Vec<f32> = vec![1.0; n_blocks * 8];
                let mut packed = vec![0u8; n_blocks * 2];
                let mut scales = vec![0.0f32; n_blocks];
                quantize_i2s_neon(&input, &mut packed, &mut scales, &cfg).unwrap();
                let res = dequantize_i2s_neon(&packed, &scales, &cfg).unwrap();
                prop_assert_eq!(res.values.len(), input.len());
                prop_assert_eq!(res.blocks_processed, n_blocks);
            }
        }
    }
}
