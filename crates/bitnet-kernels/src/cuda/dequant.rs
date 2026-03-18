//! CUDA-accelerated dequantization kernels for INT2/INT4/INT8 to FP16/FP32.
//!
//! Provides dequantization from packed low-bit integer representations to
//! floating-point output, with support for per-block and per-channel scale
//! factors. The primary use case is converting quantized model weights to
//! working precision for inference.
//!
//! # Supported conversions
//!
//! - **INT2 → FP16/FP32**: 2-bit ternary ({-1, 0, +1}) or unsigned (0..3)
//! - **INT4 → FP16/FP32**: 4-bit signed (−8..7) with two values packed per byte
//! - **INT8 → FP16/FP32**: 8-bit signed (−128..127)
//! - **QK256 block dequantization**: 256-element blocks with separate scale factors
//!
//! # Kernel strategy
//!
//! CPU fallback implementations are always available. CUDA kernel source strings
//! are gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]` and define
//! grid-stride loop kernels for GPU dispatch.

use bitnet_common::{KernelError, Result};

// ── Types ─────────────────────────────────────────────────────────────

/// Output precision for dequantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DequantPrecision {
    /// 32-bit IEEE 754 float.
    F32,
    /// 16-bit IEEE 754 half-precision float (stored as `u16`).
    F16,
}

/// Quantized element width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantBitWidth {
    /// 2-bit ternary ({-1, 0, +1}), 4 values packed per byte.
    Int2,
    /// 4-bit signed (−8..7), 2 values packed per byte.
    Int4,
    /// 8-bit signed (−128..127), 1 value per byte.
    Int8,
}

/// Scale factor application strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleMode {
    /// A single scale factor applied uniformly to all elements.
    Uniform,
    /// One scale factor per block of `block_size` elements.
    PerBlock,
    /// One scale factor per output channel (row).
    PerChannel,
}

/// Configuration for a dequantization pass.
#[derive(Debug, Clone)]
pub struct DequantConfig {
    /// Source bit width.
    pub bit_width: QuantBitWidth,
    /// Output precision.
    pub precision: DequantPrecision,
    /// Number of elements per quantization block (used with `ScaleMode::PerBlock`).
    pub block_size: usize,
    /// How scale factors are applied.
    pub scale_mode: ScaleMode,
}

impl Default for DequantConfig {
    fn default() -> Self {
        Self {
            bit_width: QuantBitWidth::Int2,
            precision: DequantPrecision::F32,
            block_size: 256,
            scale_mode: ScaleMode::PerBlock,
        }
    }
}

// ── f16 conversion helpers ────────────────────────────────────────────

/// Convert an f32 to IEEE 754 half-precision float stored as `u16`.
#[inline(always)]
fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7F_FFFF;

    if exponent == 0 {
        return sign << 15;
    }
    if exponent == 0xFF {
        let f16_mantissa = (mantissa >> 13) as u16;
        return (sign << 15) | (0x1F << 10) | f16_mantissa;
    }

    let new_exp = exponent - 112; // 127 - 15
    if new_exp >= 31 {
        return (sign << 15) | (0x1F << 10);
    }
    if new_exp <= 0 {
        return sign << 15;
    }
    let f16_mantissa = (mantissa >> 13) as u16;
    (sign << 15) | ((new_exp as u16) << 10) | f16_mantissa
}

/// Convert an IEEE 754 half-precision float (u16) to f32.
#[cfg(test)]
#[inline(always)]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut m = mantissa;
        let mut e: i32 = -14;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f32_exp = ((e + 127) as u32) & 0xFF;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }
    if exponent == 31 {
        let f32_mantissa = mantissa << 13;
        return f32::from_bits((sign << 31) | (0xFF << 23) | f32_mantissa);
    }

    let f32_exp = exponent + 112;
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
}

// ── INT2 decoding ─────────────────────────────────────────────────────

/// Decode a 2-bit I2_S code to its signed integer value.
/// Encoding: 0b01 → +1, 0b11 → −1, 0b00/0b10 → 0.
#[inline(always)]
fn decode_int2(bits: u8) -> i8 {
    match bits & 0x03 {
        0b01 => 1,
        0b11 => -1,
        _ => 0,
    }
}

/// Encode a ternary value to 2-bit I2_S code.
#[inline(always)]
fn encode_int2(v: i8) -> u8 {
    match v {
        1 => 0b01,
        -1 => 0b11,
        _ => 0b00,
    }
}

// ── INT4 decoding ─────────────────────────────────────────────────────

/// Decode a 4-bit signed nibble (two's complement in the low 4 bits).
/// Range: −8..7.
#[inline(always)]
fn decode_int4(nibble: u8) -> i8 {
    let val = (nibble & 0x0F) as i8;
    if val >= 8 { val - 16 } else { val }
}

/// Encode a signed i8 value (clamped to −8..7) into a 4-bit nibble.
#[inline(always)]
fn encode_int4(v: i8) -> u8 {
    let clamped = v.clamp(-8, 7);
    (clamped as u8) & 0x0F
}

// ── INT2 dequantization ───────────────────────────────────────────────

/// Dequantize INT2 packed bytes to f32 with per-block scales.
///
/// Four 2-bit values are packed per byte (LSB-first). Each block of
/// `block_size` elements shares one scale factor from `scales`.
///
/// # Errors
///
/// Returns an error if `block_size` is zero, the packed buffer is too
/// small, or there are insufficient scale factors.
pub fn dequantize_int2_to_f32(
    packed: &[u8],
    scales: &[f32],
    block_size: usize,
    output_len: usize,
) -> Result<Vec<f32>> {
    validate_int2_args(packed, scales, block_size, output_len)?;
    if output_len == 0 {
        return Ok(Vec::new());
    }

    let mut output = Vec::with_capacity(output_len);
    for i in 0..output_len {
        let byte_idx = i / 4;
        let bit_off = (i % 4) * 2;
        let bits = (packed[byte_idx] >> bit_off) & 0x03;
        let val = decode_int2(bits) as f32;
        let scale = resolve_scale(scales, i, block_size);
        output.push(val * scale);
    }
    Ok(output)
}

/// Dequantize INT2 packed bytes to f16 (returned as `Vec<u16>`) with per-block scales.
pub fn dequantize_int2_to_f16(
    packed: &[u8],
    scales: &[f32],
    block_size: usize,
    output_len: usize,
) -> Result<Vec<u16>> {
    validate_int2_args(packed, scales, block_size, output_len)?;
    if output_len == 0 {
        return Ok(Vec::new());
    }

    let mut output = Vec::with_capacity(output_len);
    for i in 0..output_len {
        let byte_idx = i / 4;
        let bit_off = (i % 4) * 2;
        let bits = (packed[byte_idx] >> bit_off) & 0x03;
        let val = decode_int2(bits) as f32;
        let scale = resolve_scale(scales, i, block_size);
        output.push(f32_to_f16(val * scale));
    }
    Ok(output)
}

fn validate_int2_args(
    packed: &[u8],
    scales: &[f32],
    block_size: usize,
    output_len: usize,
) -> Result<()> {
    if block_size == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "block_size must be > 0".into() }.into()
        );
    }
    if output_len == 0 {
        return Ok(());
    }
    let required_bytes = output_len.div_ceil(4);
    if packed.len() < required_bytes {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "packed buffer too small: need {required_bytes} bytes for {output_len} elements, got {}",
                packed.len()
            ),
        }
        .into());
    }
    let num_blocks = output_len.div_ceil(block_size);
    if scales.len() < num_blocks {
        return Err(KernelError::InvalidArguments {
            reason: format!("scales too short: need {num_blocks} blocks, got {}", scales.len()),
        }
        .into());
    }
    Ok(())
}

// ── INT4 dequantization ───────────────────────────────────────────────

/// Dequantize INT4 packed bytes to f32 with per-block scales.
///
/// Two 4-bit signed values are packed per byte: low nibble first, high
/// nibble second. Each block of `block_size` elements shares one scale.
pub fn dequantize_int4_to_f32(
    packed: &[u8],
    scales: &[f32],
    block_size: usize,
    output_len: usize,
) -> Result<Vec<f32>> {
    validate_int4_args(packed, scales, block_size, output_len)?;
    if output_len == 0 {
        return Ok(Vec::new());
    }

    let mut output = Vec::with_capacity(output_len);
    for i in 0..output_len {
        let byte_idx = i / 2;
        let nibble = if i % 2 == 0 { packed[byte_idx] & 0x0F } else { packed[byte_idx] >> 4 };
        let val = decode_int4(nibble) as f32;
        let scale = resolve_scale(scales, i, block_size);
        output.push(val * scale);
    }
    Ok(output)
}

/// Dequantize INT4 packed bytes to f16 (returned as `Vec<u16>`) with per-block scales.
pub fn dequantize_int4_to_f16(
    packed: &[u8],
    scales: &[f32],
    block_size: usize,
    output_len: usize,
) -> Result<Vec<u16>> {
    validate_int4_args(packed, scales, block_size, output_len)?;
    if output_len == 0 {
        return Ok(Vec::new());
    }

    let mut output = Vec::with_capacity(output_len);
    for i in 0..output_len {
        let byte_idx = i / 2;
        let nibble = if i % 2 == 0 { packed[byte_idx] & 0x0F } else { packed[byte_idx] >> 4 };
        let val = decode_int4(nibble) as f32;
        let scale = resolve_scale(scales, i, block_size);
        output.push(f32_to_f16(val * scale));
    }
    Ok(output)
}

fn validate_int4_args(
    packed: &[u8],
    scales: &[f32],
    block_size: usize,
    output_len: usize,
) -> Result<()> {
    if block_size == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "block_size must be > 0".into() }.into()
        );
    }
    if output_len == 0 {
        return Ok(());
    }
    let required_bytes = output_len.div_ceil(2);
    if packed.len() < required_bytes {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "packed buffer too small: need {required_bytes} bytes for {output_len} elements, got {}",
                packed.len()
            ),
        }
        .into());
    }
    let num_blocks = output_len.div_ceil(block_size);
    if scales.len() < num_blocks {
        return Err(KernelError::InvalidArguments {
            reason: format!("scales too short: need {num_blocks} blocks, got {}", scales.len()),
        }
        .into());
    }
    Ok(())
}

// ── INT8 dequantization ───────────────────────────────────────────────

/// Dequantize INT8 values to f32 with per-block scales.
///
/// Each `i8` element is multiplied by its block's scale factor.
pub fn dequantize_int8_to_f32(
    data: &[i8],
    scales: &[f32],
    block_size: usize,
    output_len: usize,
) -> Result<Vec<f32>> {
    validate_int8_args(data, scales, block_size, output_len)?;
    if output_len == 0 {
        return Ok(Vec::new());
    }

    let mut output = Vec::with_capacity(output_len);
    for (i, &val_i8) in data.iter().take(output_len).enumerate() {
        let val = val_i8 as f32;
        let scale = resolve_scale(scales, i, block_size);
        output.push(val * scale);
    }
    Ok(output)
}

/// Dequantize INT8 values to f16 (returned as `Vec<u16>`) with per-block scales.
pub fn dequantize_int8_to_f16(
    data: &[i8],
    scales: &[f32],
    block_size: usize,
    output_len: usize,
) -> Result<Vec<u16>> {
    validate_int8_args(data, scales, block_size, output_len)?;
    if output_len == 0 {
        return Ok(Vec::new());
    }

    let mut output = Vec::with_capacity(output_len);
    for (i, &val_i8) in data.iter().take(output_len).enumerate() {
        let val = val_i8 as f32;
        let scale = resolve_scale(scales, i, block_size);
        output.push(f32_to_f16(val * scale));
    }
    Ok(output)
}

fn validate_int8_args(
    data: &[i8],
    scales: &[f32],
    block_size: usize,
    output_len: usize,
) -> Result<()> {
    if block_size == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "block_size must be > 0".into() }.into()
        );
    }
    if output_len == 0 {
        return Ok(());
    }
    if data.len() < output_len {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "data buffer too small: need {output_len} elements, got {}",
                data.len()
            ),
        }
        .into());
    }
    let num_blocks = output_len.div_ceil(block_size);
    if scales.len() < num_blocks {
        return Err(KernelError::InvalidArguments {
            reason: format!("scales too short: need {num_blocks} blocks, got {}", scales.len()),
        }
        .into());
    }
    Ok(())
}

// ── QK256 block dequantization ────────────────────────────────────────

/// QK256 block size constant.
pub const QK256_BLOCK_SIZE: usize = 256;

/// Dequantize QK256-format packed INT2 blocks to f32.
///
/// QK256 packs 256 ternary values into 64 bytes (2 bits each) with one
/// `f32` scale factor per block. This is the canonical format for
/// BitNet I2_S models with 256-element quantization groups.
///
/// # Errors
///
/// Returns an error if the input length is not a multiple of 64 bytes,
/// or if the number of scales does not match the number of blocks.
pub fn dequantize_qk256_to_f32(packed: &[u8], scales: &[f32]) -> Result<Vec<f32>> {
    validate_qk256_args(packed, scales)?;
    if packed.is_empty() {
        return Ok(Vec::new());
    }

    let num_blocks = packed.len() / 64;
    let mut output = Vec::with_capacity(num_blocks * QK256_BLOCK_SIZE);

    for blk in 0..num_blocks {
        let blk_data = &packed[blk * 64..(blk + 1) * 64];
        let scale = scales[blk];
        for &byte in blk_data {
            for sub in 0..4 {
                let bits = (byte >> (sub * 2)) & 0x03;
                let val = decode_int2(bits) as f32;
                output.push(val * scale);
            }
        }
    }
    Ok(output)
}

/// Dequantize QK256-format packed INT2 blocks to f16 (returned as `Vec<u16>`).
pub fn dequantize_qk256_to_f16(packed: &[u8], scales: &[f32]) -> Result<Vec<u16>> {
    validate_qk256_args(packed, scales)?;
    if packed.is_empty() {
        return Ok(Vec::new());
    }

    let num_blocks = packed.len() / 64;
    let mut output = Vec::with_capacity(num_blocks * QK256_BLOCK_SIZE);

    for blk in 0..num_blocks {
        let blk_data = &packed[blk * 64..(blk + 1) * 64];
        let scale = scales[blk];
        for &byte in blk_data {
            for sub in 0..4 {
                let bits = (byte >> (sub * 2)) & 0x03;
                let val = decode_int2(bits) as f32;
                output.push(f32_to_f16(val * scale));
            }
        }
    }
    Ok(output)
}

fn validate_qk256_args(packed: &[u8], scales: &[f32]) -> Result<()> {
    if packed.is_empty() {
        return Ok(());
    }
    if !packed.len().is_multiple_of(64) {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "QK256 packed data must be a multiple of 64 bytes, got {}",
                packed.len()
            ),
        }
        .into());
    }
    let num_blocks = packed.len() / 64;
    if scales.len() < num_blocks {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "QK256 scales too short: need {num_blocks} blocks, got {}",
                scales.len()
            ),
        }
        .into());
    }
    Ok(())
}

// ── Batch dequantization ──────────────────────────────────────────────

/// Dequantize a batch of INT2 tensors to f32.
///
/// Each entry in `batch_packed` is a packed buffer with corresponding
/// scales. All tensors use the same `block_size` and `output_len`.
pub fn batch_dequantize_int2_to_f32(
    batch_packed: &[&[u8]],
    batch_scales: &[&[f32]],
    block_size: usize,
    output_len: usize,
) -> Result<Vec<Vec<f32>>> {
    if batch_packed.len() != batch_scales.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "batch size mismatch: {} packed vs {} scales",
                batch_packed.len(),
                batch_scales.len()
            ),
        }
        .into());
    }
    batch_packed
        .iter()
        .zip(batch_scales.iter())
        .map(|(packed, scales)| dequantize_int2_to_f32(packed, scales, block_size, output_len))
        .collect()
}

/// Dequantize with per-channel scales (one scale per row of `row_len` elements).
pub fn dequantize_int2_per_channel_f32(
    packed: &[u8],
    channel_scales: &[f32],
    row_len: usize,
    num_rows: usize,
) -> Result<Vec<f32>> {
    let output_len = num_rows * row_len;
    if row_len == 0 {
        return Err(KernelError::InvalidArguments { reason: "row_len must be > 0".into() }.into());
    }
    validate_int2_args(packed, &vec![0.0; output_len.div_ceil(row_len)], row_len, output_len)
        .map_err(|_| KernelError::InvalidArguments {
            reason: format!("packed buffer too small for {num_rows} rows × {row_len} elements"),
        })?;
    if channel_scales.len() < num_rows {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "channel_scales too short: need {num_rows}, got {}",
                channel_scales.len()
            ),
        }
        .into());
    }

    let mut output = Vec::with_capacity(output_len);
    for i in 0..output_len {
        let byte_idx = i / 4;
        let bit_off = (i % 4) * 2;
        let bits = (packed[byte_idx] >> bit_off) & 0x03;
        let val = decode_int2(bits) as f32;
        let row = i / row_len;
        output.push(val * channel_scales[row]);
    }
    Ok(output)
}

// ── Uniform-scale helpers ─────────────────────────────────────────────

/// Dequantize INT2 packed bytes to f32 with a single uniform scale.
pub fn dequantize_int2_uniform_f32(
    packed: &[u8],
    scale: f32,
    output_len: usize,
) -> Result<Vec<f32>> {
    dequantize_int2_to_f32(packed, &vec![scale; output_len.div_ceil(1)], 1, output_len)
}

/// Dequantize INT8 values to f32 with a single uniform scale.
pub fn dequantize_int8_uniform_f32(data: &[i8], scale: f32, output_len: usize) -> Result<Vec<f32>> {
    dequantize_int8_to_f32(data, &vec![scale; output_len], 1, output_len)
}

// ── Helper ────────────────────────────────────────────────────────────

/// Resolve the scale factor for element `i` given per-block scales.
#[inline(always)]
fn resolve_scale(scales: &[f32], i: usize, block_size: usize) -> f32 {
    let blk = i / block_size;
    scales[blk]
}

// ── Quantize helpers (for round-trip tests) ───────────────────────────

/// Quantize f32 values to INT2 packed bytes with per-block AbsMax scales.
/// Returns `(packed, scales)`.
pub fn quantize_to_int2(input: &[f32], block_size: usize) -> Result<(Vec<u8>, Vec<f32>)> {
    if block_size == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "block_size must be > 0".into() }.into()
        );
    }
    if input.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let num_blocks = input.len().div_ceil(block_size);
    let packed_len = input.len().div_ceil(4);
    let mut packed = vec![0u8; packed_len];
    let mut scales = Vec::with_capacity(num_blocks);

    for blk in 0..num_blocks {
        let start = blk * block_size;
        let end = (start + block_size).min(input.len());
        let block = &input[start..end];
        let abs_max = block.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
        scales.push(abs_max);
        let threshold = abs_max * 0.5;
        for (j, &v) in block.iter().enumerate() {
            let ternary = if abs_max == 0.0 {
                0_i8
            } else if v > threshold {
                1_i8
            } else if v < -threshold {
                -1_i8
            } else {
                0_i8
            };
            let global_idx = start + j;
            let byte_idx = global_idx / 4;
            let bit_off = (global_idx % 4) * 2;
            packed[byte_idx] |= encode_int2(ternary) << bit_off;
        }
    }
    Ok((packed, scales))
}

/// Quantize f32 values to INT4 packed bytes with per-block AbsMax scales.
/// Returns `(packed, scales)`.
pub fn quantize_to_int4(input: &[f32], block_size: usize) -> Result<(Vec<u8>, Vec<f32>)> {
    if block_size == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "block_size must be > 0".into() }.into()
        );
    }
    if input.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let num_blocks = input.len().div_ceil(block_size);
    let packed_len = input.len().div_ceil(2);
    let mut packed = vec![0u8; packed_len];
    let mut scales = Vec::with_capacity(num_blocks);

    for blk in 0..num_blocks {
        let start = blk * block_size;
        let end = (start + block_size).min(input.len());
        let block = &input[start..end];
        let abs_max = block.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
        let scale = if abs_max == 0.0 { 1.0 } else { abs_max / 7.0 };
        scales.push(scale);
        for (j, &v) in block.iter().enumerate() {
            let quantized =
                if scale == 0.0 { 0_i8 } else { (v / scale).round().clamp(-8.0, 7.0) as i8 };
            let global_idx = start + j;
            let byte_idx = global_idx / 2;
            let nibble = encode_int4(quantized);
            if global_idx.is_multiple_of(2) {
                packed[byte_idx] |= nibble;
            } else {
                packed[byte_idx] |= nibble << 4;
            }
        }
    }
    Ok((packed, scales))
}

/// Quantize f32 values to INT8 with per-block AbsMax scales.
/// Returns `(quantized, scales)`.
pub fn quantize_to_int8(input: &[f32], block_size: usize) -> Result<(Vec<i8>, Vec<f32>)> {
    if block_size == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "block_size must be > 0".into() }.into()
        );
    }
    if input.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let num_blocks = input.len().div_ceil(block_size);
    let mut quantized = Vec::with_capacity(input.len());
    let mut scales = Vec::with_capacity(num_blocks);

    for blk in 0..num_blocks {
        let start = blk * block_size;
        let end = (start + block_size).min(input.len());
        let block = &input[start..end];
        let abs_max = block.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
        let scale = if abs_max == 0.0 { 1.0 } else { abs_max / 127.0 };
        scales.push(scale);
        for &v in block {
            let q = if scale == 0.0 { 0 } else { (v / scale).round().clamp(-128.0, 127.0) as i8 };
            quantized.push(q);
        }
    }
    Ok((quantized, scales))
}

// ── CUDA kernel source ────────────────────────────────────────────────

/// CUDA C source for INT2 → f32 dequantization kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const DEQUANT_INT2_F32_KERNEL_SRC: &str = r#"
__device__ signed char decode_int2(unsigned char bits) {
    bits &= 0x03;
    if (bits == 0x01) return 1;
    if (bits == 0x03) return -1;
    return 0;
}

extern "C" __global__ void dequant_int2_f32(
    const unsigned char* __restrict__ packed,
    const float* __restrict__ scales,
    float* __restrict__ output,
    int n,
    int block_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        int byte_idx = i / 4;
        int bit_off  = (i % 4) * 2;
        unsigned char bits = (packed[byte_idx] >> bit_off) & 0x03;
        signed char val = decode_int2(bits);
        int blk = i / block_size;
        output[i] = (float)val * scales[blk];
    }
}
"#;

/// CUDA C source for INT4 → f32 dequantization kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const DEQUANT_INT4_F32_KERNEL_SRC: &str = r#"
__device__ signed char decode_int4(unsigned char nibble) {
    signed char val = (signed char)(nibble & 0x0F);
    return val >= 8 ? val - 16 : val;
}

extern "C" __global__ void dequant_int4_f32(
    const unsigned char* __restrict__ packed,
    const float* __restrict__ scales,
    float* __restrict__ output,
    int n,
    int block_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        int byte_idx = i / 2;
        unsigned char nibble = (i % 2 == 0)
            ? (packed[byte_idx] & 0x0F)
            : (packed[byte_idx] >> 4);
        signed char val = decode_int4(nibble);
        int blk = i / block_size;
        output[i] = (float)val * scales[blk];
    }
}
"#;

/// CUDA C source for INT8 → f32 dequantization kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const DEQUANT_INT8_F32_KERNEL_SRC: &str = r#"
extern "C" __global__ void dequant_int8_f32(
    const signed char* __restrict__ data,
    const float* __restrict__ scales,
    float* __restrict__ output,
    int n,
    int block_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        int blk = i / block_size;
        output[i] = (float)data[i] * scales[blk];
    }
}
"#;

/// CUDA C source for QK256 block → f32 dequantization kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const DEQUANT_QK256_F32_KERNEL_SRC: &str = r#"
__device__ signed char decode_int2_qk(unsigned char bits) {
    bits &= 0x03;
    if (bits == 0x01) return 1;
    if (bits == 0x03) return -1;
    return 0;
}

extern "C" __global__ void dequant_qk256_f32(
    const unsigned char* __restrict__ packed,
    const float* __restrict__ scales,
    float* __restrict__ output,
    int num_blocks)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_blocks * 256;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int blk = i / 256;
        int within = i % 256;
        int byte_off = blk * 64 + within / 4;
        int bit_off = (within % 4) * 2;
        unsigned char bits = (packed[byte_off] >> bit_off) & 0x03;
        signed char val = decode_int2_qk(bits);
        output[i] = (float)val * scales[blk];
    }
}
"#;

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Assert two f32 slices are element-wise close within tolerance.
    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at index {i}: {x} vs {y} (tol {tol})");
        }
    }

    /// Helper: pack ternary values into INT2 bytes (LSB-first).
    fn pack_int2(values: &[i8]) -> Vec<u8> {
        let len = values.len().div_ceil(4);
        let mut packed = vec![0u8; len];
        for (i, &v) in values.iter().enumerate() {
            let code = encode_int2(v);
            packed[i / 4] |= code << ((i % 4) * 2);
        }
        packed
    }

    /// Helper: pack signed nibbles into INT4 bytes (low nibble first).
    fn pack_int4(values: &[i8]) -> Vec<u8> {
        let len = values.len().div_ceil(2);
        let mut packed = vec![0u8; len];
        for (i, &v) in values.iter().enumerate() {
            let nibble = encode_int4(v);
            if i % 2 == 0 {
                packed[i / 2] |= nibble;
            } else {
                packed[i / 2] |= nibble << 4;
            }
        }
        packed
    }

    // ── DequantConfig defaults ────────────────────────────────────

    #[test]
    fn config_default_values() {
        let cfg = DequantConfig::default();
        assert_eq!(cfg.bit_width, QuantBitWidth::Int2);
        assert_eq!(cfg.precision, DequantPrecision::F32);
        assert_eq!(cfg.block_size, 256);
        assert_eq!(cfg.scale_mode, ScaleMode::PerBlock);
    }

    #[test]
    fn config_custom_construction() {
        let cfg = DequantConfig {
            bit_width: QuantBitWidth::Int8,
            precision: DequantPrecision::F16,
            block_size: 32,
            scale_mode: ScaleMode::PerChannel,
        };
        assert_eq!(cfg.bit_width, QuantBitWidth::Int8);
        assert_eq!(cfg.precision, DequantPrecision::F16);
    }

    // ── INT2 → F32 ───────────────────────────────────────────────

    #[test]
    fn int2_f32_empty() {
        let result = dequantize_int2_to_f32(&[], &[], 32, 0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn int2_f32_single_element() {
        // +1 encoded as 0b01
        let packed = vec![0b01u8];
        let scales = [2.0];
        let result = dequantize_int2_to_f32(&packed, &scales, 1, 1).unwrap();
        assert_close(&result, &[2.0], 1e-6);
    }

    #[test]
    fn int2_f32_all_ternary_values() {
        // +1, -1, 0, +1 → bits: 01 11 00 01 = 0b01_00_11_01 = 0x4D
        let packed = pack_int2(&[1, -1, 0, 1]);
        let scales = [3.0];
        let result = dequantize_int2_to_f32(&packed, &scales, 4, 4).unwrap();
        assert_close(&result, &[3.0, -3.0, 0.0, 3.0], 1e-6);
    }

    #[test]
    fn int2_f32_block_size_32() {
        let values: Vec<i8> = (0..32)
            .map(|i| {
                if i % 3 == 0 {
                    1
                } else if i % 3 == 1 {
                    -1
                } else {
                    0
                }
            })
            .collect();
        let packed = pack_int2(&values);
        let scales = [1.5];
        let result = dequantize_int2_to_f32(&packed, &scales, 32, 32).unwrap();
        assert_eq!(result.len(), 32);
        assert_close(&result[0..1], &[1.5], 1e-6); // i=0: +1
        assert_close(&result[1..2], &[-1.5], 1e-6); // i=1: -1
        assert_close(&result[2..3], &[0.0], 1e-6); // i=2: 0
    }

    #[test]
    fn int2_f32_multiple_blocks() {
        let values = [1_i8; 8];
        let packed = pack_int2(&values);
        let scales = vec![1.0, 2.0];
        let result = dequantize_int2_to_f32(&packed, &scales, 4, 8).unwrap();
        assert_close(&result[0..4], &[1.0, 1.0, 1.0, 1.0], 1e-6);
        assert_close(&result[4..8], &[2.0, 2.0, 2.0, 2.0], 1e-6);
    }

    #[test]
    fn int2_f32_large_256() {
        let values = [1_i8; 256];
        let packed = pack_int2(&values);
        let scales = [0.5];
        let result = dequantize_int2_to_f32(&packed, &scales, 256, 256).unwrap();
        assert_eq!(result.len(), 256);
        for &v in &result {
            assert!((v - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn int2_f32_large_1024() {
        let values: Vec<i8> = (0..1024).map(|i| [1, -1, 0, 1][i % 4]).collect();
        let packed = pack_int2(&values);
        let scales = [1.0; 4];
        let result = dequantize_int2_to_f32(&packed, &scales, 256, 1024).unwrap();
        assert_eq!(result.len(), 1024);
    }

    #[test]
    fn int2_f32_large_65536() {
        let values = [1_i8; 65536];
        let packed = pack_int2(&values);
        let scales = [1.0; 256];
        let result = dequantize_int2_to_f32(&packed, &scales, 256, 65536).unwrap();
        assert_eq!(result.len(), 65536);
    }

    #[test]
    fn int2_f32_non_aligned_size() {
        let values = vec![1_i8, -1, 0];
        let packed = pack_int2(&values);
        let scales = [2.0];
        let result = dequantize_int2_to_f32(&packed, &scales, 4, 3).unwrap();
        assert_close(&result, &[2.0, -2.0, 0.0], 1e-6);
    }

    #[test]
    fn int2_f32_zero_scale() {
        let packed = pack_int2(&[1, -1, 0, 1]);
        let scales = [0.0];
        let result = dequantize_int2_to_f32(&packed, &scales, 4, 4).unwrap();
        assert_close(&result, &[0.0, 0.0, 0.0, 0.0], 1e-6);
    }

    #[test]
    fn int2_f32_rejects_zero_block_size() {
        assert!(dequantize_int2_to_f32(&[0], &[1.0], 0, 1).is_err());
    }

    #[test]
    fn int2_f32_rejects_short_packed() {
        assert!(dequantize_int2_to_f32(&[], &[1.0], 4, 4).is_err());
    }

    #[test]
    fn int2_f32_rejects_short_scales() {
        let packed = pack_int2(&[1; 8]);
        assert!(dequantize_int2_to_f32(&packed, &[1.0], 4, 8).is_err());
    }

    // ── INT2 → F16 ───────────────────────────────────────────────

    #[test]
    fn int2_f16_empty() {
        let result = dequantize_int2_to_f16(&[], &[], 32, 0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn int2_f16_single_element() {
        let packed = pack_int2(&[1]);
        let scales = [1.0];
        let result = dequantize_int2_to_f16(&packed, &scales, 1, 1).unwrap();
        let f32_val = f16_to_f32(result[0]);
        assert!((f32_val - 1.0).abs() < 0.01);
    }

    #[test]
    fn int2_f16_ternary_values() {
        let packed = pack_int2(&[1, -1, 0, 1]);
        let scales = [2.0];
        let result = dequantize_int2_to_f16(&packed, &scales, 4, 4).unwrap();
        let f32_vals: Vec<f32> = result.iter().map(|&b| f16_to_f32(b)).collect();
        assert_close(&f32_vals, &[2.0, -2.0, 0.0, 2.0], 0.01);
    }

    #[test]
    fn int2_f16_large_256() {
        let packed = pack_int2(&vec![1_i8; 256]);
        let scales = [0.5];
        let result = dequantize_int2_to_f16(&packed, &scales, 256, 256).unwrap();
        assert_eq!(result.len(), 256);
        for &bits in &result {
            let v = f16_to_f32(bits);
            assert!((v - 0.5).abs() < 0.01);
        }
    }

    #[test]
    fn int2_f16_accuracy_vs_f32() {
        let packed = pack_int2(&[1, -1, 1, -1, 0, 1, -1, 0]);
        let scales = vec![0.75, 1.25];
        let f32_result = dequantize_int2_to_f32(&packed, &scales, 4, 8).unwrap();
        let f16_result = dequantize_int2_to_f16(&packed, &scales, 4, 8).unwrap();
        for (i, (&f32v, &f16bits)) in f32_result.iter().zip(f16_result.iter()).enumerate() {
            let f16v = f16_to_f32(f16bits);
            assert!((f32v - f16v).abs() < 0.01, "f16/f32 mismatch at {i}: f32={f32v}, f16={f16v}");
        }
    }

    // ── INT4 → F32 ───────────────────────────────────────────────

    #[test]
    fn int4_f32_empty() {
        let result = dequantize_int4_to_f32(&[], &[], 32, 0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn int4_f32_single_element() {
        // Value 3 in low nibble
        let packed = vec![0x03u8];
        let scales = [1.0];
        let result = dequantize_int4_to_f32(&packed, &scales, 1, 1).unwrap();
        assert_close(&result, &[3.0], 1e-6);
    }

    #[test]
    fn int4_f32_two_values_per_byte() {
        // Low nibble=5, high nibble=0xD (13 → -3 in signed 4-bit)
        let packed = vec![0xD5u8];
        let scales = [1.0];
        let result = dequantize_int4_to_f32(&packed, &scales, 2, 2).unwrap();
        assert_close(&result, &[5.0, -3.0], 1e-6);
    }

    #[test]
    fn int4_f32_negative_values() {
        // 0xF = 15 → signed = -1, 0x8 = 8 → signed = -8
        let packed = vec![0x8Fu8]; // low=0xF(-1), high=0x8(-8)
        let scales = [2.0];
        let result = dequantize_int4_to_f32(&packed, &scales, 2, 2).unwrap();
        assert_close(&result, &[-2.0, -16.0], 1e-6);
    }

    #[test]
    fn int4_f32_full_range() {
        // Test values -8 through 7
        let values: Vec<i8> = (-8..=7).collect();
        let packed = pack_int4(&values);
        let scales = [1.0];
        let result = dequantize_int4_to_f32(&packed, &scales, 16, 16).unwrap();
        let expected: Vec<f32> = values.iter().map(|&v| v as f32).collect();
        assert_close(&result, &expected, 1e-6);
    }

    #[test]
    fn int4_f32_block_size_32() {
        let values: Vec<i8> = (0..32).map(|i| (i % 8) as i8 - 4).collect();
        let packed = pack_int4(&values);
        let scales = [1.5];
        let result = dequantize_int4_to_f32(&packed, &scales, 32, 32).unwrap();
        assert_eq!(result.len(), 32);
    }

    #[test]
    fn int4_f32_multiple_blocks() {
        let values: Vec<i8> = vec![3; 8];
        let packed = pack_int4(&values);
        let scales = vec![1.0, 2.0];
        let result = dequantize_int4_to_f32(&packed, &scales, 4, 8).unwrap();
        assert_close(&result[0..4], &[3.0; 4], 1e-6);
        assert_close(&result[4..8], &[6.0; 4], 1e-6);
    }

    #[test]
    fn int4_f32_large_1024() {
        let values: Vec<i8> = (0..1024).map(|i| (i % 16) as i8 - 8).collect();
        let packed = pack_int4(&values);
        let scales = [1.0; 4];
        let result = dequantize_int4_to_f32(&packed, &scales, 256, 1024).unwrap();
        assert_eq!(result.len(), 1024);
    }

    #[test]
    fn int4_f32_non_aligned_size() {
        let values: Vec<i8> = vec![2, -3, 5];
        let packed = pack_int4(&values);
        let scales = [1.0];
        let result = dequantize_int4_to_f32(&packed, &scales, 4, 3).unwrap();
        assert_close(&result, &[2.0, -3.0, 5.0], 1e-6);
    }

    #[test]
    fn int4_f32_rejects_zero_block_size() {
        assert!(dequantize_int4_to_f32(&[0], &[1.0], 0, 1).is_err());
    }

    #[test]
    fn int4_f32_rejects_short_packed() {
        assert!(dequantize_int4_to_f32(&[], &[1.0], 4, 2).is_err());
    }

    // ── INT4 → F16 ───────────────────────────────────────────────

    #[test]
    fn int4_f16_empty() {
        let result = dequantize_int4_to_f16(&[], &[], 32, 0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn int4_f16_basic() {
        let packed = pack_int4(&[3, -2]);
        let scales = [1.0];
        let result = dequantize_int4_to_f16(&packed, &scales, 2, 2).unwrap();
        let f32_vals: Vec<f32> = result.iter().map(|&b| f16_to_f32(b)).collect();
        assert_close(&f32_vals, &[3.0, -2.0], 0.01);
    }

    #[test]
    fn int4_f16_accuracy_vs_f32() {
        let values: Vec<i8> = vec![7, -8, 0, 3, -1, 5, -4, 2];
        let packed = pack_int4(&values);
        let scales = vec![0.5, 1.5];
        let f32_result = dequantize_int4_to_f32(&packed, &scales, 4, 8).unwrap();
        let f16_result = dequantize_int4_to_f16(&packed, &scales, 4, 8).unwrap();
        for (i, (&f32v, &f16bits)) in f32_result.iter().zip(f16_result.iter()).enumerate() {
            let f16v = f16_to_f32(f16bits);
            assert!((f32v - f16v).abs() < 0.1, "f16/f32 mismatch at {i}: f32={f32v}, f16={f16v}");
        }
    }

    // ── INT8 → F32 ───────────────────────────────────────────────

    #[test]
    fn int8_f32_empty() {
        let result = dequantize_int8_to_f32(&[], &[], 32, 0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn int8_f32_single_element() {
        let data = vec![42_i8];
        let scales = [0.1];
        let result = dequantize_int8_to_f32(&data, &scales, 1, 1).unwrap();
        assert_close(&result, &[4.2], 1e-6);
    }

    #[test]
    fn int8_f32_positive_and_negative() {
        let data = vec![127_i8, -128, 0, 1, -1];
        let scales = [1.0];
        let result = dequantize_int8_to_f32(&data, &scales, 8, 5).unwrap();
        assert_close(&result, &[127.0, -128.0, 0.0, 1.0, -1.0], 1e-6);
    }

    #[test]
    fn int8_f32_with_scale() {
        let data = vec![100_i8, -50, 25];
        let scales = [0.01];
        let result = dequantize_int8_to_f32(&data, &scales, 4, 3).unwrap();
        assert_close(&result, &[1.0, -0.5, 0.25], 1e-6);
    }

    #[test]
    fn int8_f32_multiple_blocks() {
        let data = [10_i8; 8];
        let scales = vec![1.0, 0.5];
        let result = dequantize_int8_to_f32(&data, &scales, 4, 8).unwrap();
        assert_close(&result[0..4], &[10.0; 4], 1e-6);
        assert_close(&result[4..8], &[5.0; 4], 1e-6);
    }

    #[test]
    fn int8_f32_block_size_256() {
        let data: Vec<i8> = (0..256).map(|i| (i % 256) as i8).collect();
        let scales = [1.0];
        let result = dequantize_int8_to_f32(&data, &scales, 256, 256).unwrap();
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn int8_f32_large_1024() {
        let data: Vec<i8> = (0..1024).map(|i| (i % 256) as i8).collect();
        let scales = [0.1; 4];
        let result = dequantize_int8_to_f32(&data, &scales, 256, 1024).unwrap();
        assert_eq!(result.len(), 1024);
    }

    #[test]
    fn int8_f32_large_65536() {
        let data = [1_i8; 65536];
        let scales = [1.0; 256];
        let result = dequantize_int8_to_f32(&data, &scales, 256, 65536).unwrap();
        assert_eq!(result.len(), 65536);
        for &v in &result {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn int8_f32_zero_scale() {
        let data = vec![100_i8, -50];
        let scales = [0.0];
        let result = dequantize_int8_to_f32(&data, &scales, 4, 2).unwrap();
        assert_close(&result, &[0.0, 0.0], 1e-6);
    }

    #[test]
    fn int8_f32_rejects_zero_block_size() {
        assert!(dequantize_int8_to_f32(&[0], &[1.0], 0, 1).is_err());
    }

    #[test]
    fn int8_f32_rejects_short_data() {
        assert!(dequantize_int8_to_f32(&[0], &[1.0, 1.0], 1, 2).is_err());
    }

    #[test]
    fn int8_f32_rejects_short_scales() {
        assert!(dequantize_int8_to_f32(&[0; 8], &[1.0], 4, 8).is_err());
    }

    // ── INT8 → F16 ───────────────────────────────────────────────

    #[test]
    fn int8_f16_empty() {
        let result = dequantize_int8_to_f16(&[], &[], 32, 0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn int8_f16_basic() {
        let data = vec![10_i8, -20, 0];
        let scales = [0.5];
        let result = dequantize_int8_to_f16(&data, &scales, 4, 3).unwrap();
        let f32_vals: Vec<f32> = result.iter().map(|&b| f16_to_f32(b)).collect();
        assert_close(&f32_vals, &[5.0, -10.0, 0.0], 0.1);
    }

    #[test]
    fn int8_f16_accuracy_vs_f32() {
        let data: Vec<i8> = vec![127, -128, 64, -32, 0, 1, -1, 100];
        let scales = vec![0.01, 0.02];
        let f32_result = dequantize_int8_to_f32(&data, &scales, 4, 8).unwrap();
        let f16_result = dequantize_int8_to_f16(&data, &scales, 4, 8).unwrap();
        for (i, (&f32v, &f16bits)) in f32_result.iter().zip(f16_result.iter()).enumerate() {
            let f16v = f16_to_f32(f16bits);
            assert!((f32v - f16v).abs() < 0.1, "f16/f32 mismatch at {i}: f32={f32v}, f16={f16v}");
        }
    }

    // ── QK256 block dequantization ───────────────────────────────

    #[test]
    fn qk256_f32_empty() {
        let result = dequantize_qk256_to_f32(&[], &[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn qk256_f32_single_block() {
        // One QK256 block: 64 bytes = 256 ternary values, all +1
        let mut packed = [0u8; 64];
        for byte in &mut packed {
            // 01 01 01 01 = 0x55 (four +1 values)
            *byte = 0x55;
        }
        let scales = [3.0];
        let result = dequantize_qk256_to_f32(&packed, &scales).unwrap();
        assert_eq!(result.len(), 256);
        for &v in &result {
            assert!((v - 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn qk256_f32_two_blocks() {
        let mut packed = [0x55u8; 128]; // Two blocks, all +1
        // Second block: all -1 (0b11 11 11 11 = 0xFF)
        for byte in &mut packed[64..128] {
            *byte = 0xFF;
        }
        let scales = vec![1.0, 2.0];
        let result = dequantize_qk256_to_f32(&packed, &scales).unwrap();
        assert_eq!(result.len(), 512);
        // Block 0: all +1 × 1.0
        for &v in &result[0..256] {
            assert!((v - 1.0).abs() < 1e-6);
        }
        // Block 1: all -1 × 2.0
        for &v in &result[256..512] {
            assert!((v - (-2.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn qk256_f32_mixed_values() {
        let mut packed = [0u8; 64];
        // First byte: +1, -1, 0, +1 → 01 11 00 01 = 0b01_00_11_01 = 0x4D
        packed[0] = 0x4D;
        let scales = [5.0];
        let result = dequantize_qk256_to_f32(&packed, &scales).unwrap();
        assert_close(&result[0..4], &[5.0, -5.0, 0.0, 5.0], 1e-6);
    }

    #[test]
    fn qk256_f32_large_4_blocks() {
        let packed = [0x55u8; 256]; // 4 blocks
        let scales = vec![1.0, 2.0, 3.0, 4.0];
        let result = dequantize_qk256_to_f32(&packed, &scales).unwrap();
        assert_eq!(result.len(), 1024);
    }

    #[test]
    fn qk256_f32_rejects_non_aligned() {
        assert!(dequantize_qk256_to_f32(&[0; 63], &[1.0]).is_err());
    }

    #[test]
    fn qk256_f32_rejects_short_scales() {
        assert!(dequantize_qk256_to_f32(&[0; 128], &[1.0]).is_err());
    }

    #[test]
    fn qk256_f16_single_block() {
        let packed = [0x55u8; 64]; // all +1
        let scales = [1.0];
        let result = dequantize_qk256_to_f16(&packed, &scales).unwrap();
        assert_eq!(result.len(), 256);
        for &bits in &result {
            let v = f16_to_f32(bits);
            assert!((v - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn qk256_f16_accuracy_vs_f32() {
        let mut packed = [0u8; 64];
        packed[0] = 0x4D; // +1, -1, 0, +1
        let scales = [2.5];
        let f32_result = dequantize_qk256_to_f32(&packed, &scales).unwrap();
        let f16_result = dequantize_qk256_to_f16(&packed, &scales).unwrap();
        for i in 0..4 {
            let f16v = f16_to_f32(f16_result[i]);
            assert!((f32_result[i] - f16v).abs() < 0.01, "QK256 f16/f32 mismatch at {i}");
        }
    }

    // ── Scale factor handling ────────────────────────────────────

    #[test]
    fn uniform_scale_int2() {
        let packed = pack_int2(&[1, -1, 1, -1]);
        let result = dequantize_int2_uniform_f32(&packed, 3.0, 4).unwrap();
        assert_close(&result, &[3.0, -3.0, 3.0, -3.0], 1e-6);
    }

    #[test]
    fn uniform_scale_int8() {
        let data = vec![10_i8, -10, 5];
        let result = dequantize_int8_uniform_f32(&data, 0.1, 3).unwrap();
        assert_close(&result, &[1.0, -1.0, 0.5], 1e-6);
    }

    #[test]
    fn per_channel_int2_basic() {
        let values = [1_i8; 8];
        let packed = pack_int2(&values);
        let channel_scales = vec![1.0, 2.0];
        let result = dequantize_int2_per_channel_f32(&packed, &channel_scales, 4, 2).unwrap();
        assert_close(&result[0..4], &[1.0; 4], 1e-6);
        assert_close(&result[4..8], &[2.0; 4], 1e-6);
    }

    #[test]
    fn per_channel_rejects_zero_row_len() {
        assert!(dequantize_int2_per_channel_f32(&[], &[1.0], 0, 1).is_err());
    }

    // ── Numerical accuracy tests ─────────────────────────────────

    #[test]
    fn max_absolute_error_int2_f32() {
        // INT2 is ternary, so dequantized values are exactly {-s, 0, s}.
        // Max absolute error should be 0 for exact ternary encoding.
        let packed = pack_int2(&[1, -1, 0, 1, -1, 0, 1, -1]);
        let scales = vec![1.0, 1.0];
        let result = dequantize_int2_to_f32(&packed, &scales, 4, 8).unwrap();
        let expected = [1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0];
        let max_err =
            result.iter().zip(expected.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_err < 1e-7, "max absolute error too large: {max_err}");
    }

    #[test]
    fn max_absolute_error_int4_f32() {
        let values: Vec<i8> = (-8..=7).collect();
        let packed = pack_int4(&values);
        let scales = [1.0];
        let result = dequantize_int4_to_f32(&packed, &scales, 16, 16).unwrap();
        let expected: Vec<f32> = values.iter().map(|&v| v as f32).collect();
        let max_err =
            result.iter().zip(expected.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_err < 1e-7, "max absolute error too large: {max_err}");
    }

    #[test]
    fn max_absolute_error_int8_f32() {
        let data: Vec<i8> = (-128..=127).map(|v| v as i8).collect();
        let scales = [0.01];
        let result = dequantize_int8_to_f32(&data, &scales, 256, 256).unwrap();
        let expected: Vec<f32> = data.iter().map(|&v| v as f32 * 0.01).collect();
        let max_err =
            result.iter().zip(expected.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_err < 1e-6, "max absolute error too large: {max_err}");
    }

    #[test]
    fn relative_error_int8_f32() {
        let data: Vec<i8> = (1..=127).map(|v| v as i8).collect();
        let scales = [0.1];
        let result = dequantize_int8_to_f32(&data, &scales, 128, 127).unwrap();
        for (i, (&actual, &q)) in result.iter().zip(data.iter()).enumerate() {
            let expected = q as f32 * 0.1;
            if expected.abs() > 1e-10 {
                let rel = (actual - expected).abs() / expected.abs();
                assert!(rel < 1e-5, "relative error too large at {i}: {rel}");
            }
        }
    }

    // ── Round-trip tests ─────────────────────────────────────────

    #[test]
    fn round_trip_int2_identity_ternary() {
        // Ternary values round-trip exactly
        let original = vec![1.0_f32, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0];
        let (packed, scales) = quantize_to_int2(&original, 4).unwrap();
        let recovered = dequantize_int2_to_f32(&packed, &scales, 4, original.len()).unwrap();
        // The quantized values are ternary × scale, so they won't exactly match original
        // but the ternary pattern should be preserved
        for (i, &v) in recovered.iter().enumerate() {
            if original[i] > 0.0 {
                assert!(v > 0.0, "sign mismatch at {i}");
            } else if original[i] < 0.0 {
                assert!(v < 0.0, "sign mismatch at {i}");
            } else {
                assert!((v).abs() < 1e-6, "zero mismatch at {i}");
            }
        }
    }

    #[test]
    fn round_trip_int4_preserves_sign() {
        let original = vec![7.0_f32, -8.0, 0.0, 3.5, -2.1, 1.0, -0.5, 6.0];
        let (packed, scales) = quantize_to_int4(&original, 4).unwrap();
        let recovered = dequantize_int4_to_f32(&packed, &scales, 4, original.len()).unwrap();
        for (i, (&orig, &rec)) in original.iter().zip(recovered.iter()).enumerate() {
            if orig > 0.0 {
                assert!(rec >= 0.0, "sign mismatch at {i}: orig={orig}, rec={rec}");
            } else if orig < 0.0 {
                assert!(rec <= 0.0, "sign mismatch at {i}: orig={orig}, rec={rec}");
            }
        }
    }

    #[test]
    fn round_trip_int8_accuracy() {
        let original: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let (quantized, scales) = quantize_to_int8(&original, 256).unwrap();
        let recovered = dequantize_int8_to_f32(&quantized, &scales, 256, original.len()).unwrap();
        let max_err = original
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        // INT8 with 256 levels, max quantization error is ~scale/2
        assert!(max_err < 0.02, "round-trip max error too large: {max_err}");
    }

    #[test]
    fn round_trip_int2_empty() {
        let (packed, scales) = quantize_to_int2(&[], 32).unwrap();
        let recovered = dequantize_int2_to_f32(&packed, &scales, 32, 0).unwrap();
        assert!(recovered.is_empty());
    }

    #[test]
    fn round_trip_int4_empty() {
        let (packed, scales) = quantize_to_int4(&[], 32).unwrap();
        let recovered = dequantize_int4_to_f32(&packed, &scales, 32, 0).unwrap();
        assert!(recovered.is_empty());
    }

    #[test]
    fn round_trip_int8_empty() {
        let (quantized, scales) = quantize_to_int8(&[], 32).unwrap();
        let recovered = dequantize_int8_to_f32(&quantized, &scales, 32, 0).unwrap();
        assert!(recovered.is_empty());
    }

    // ── Batch dequantization ─────────────────────────────────────

    #[test]
    fn batch_int2_basic() {
        let p1 = pack_int2(&[1, -1, 0, 1]);
        let p2 = pack_int2(&[-1, 1, 1, 0]);
        let s1 = [2.0];
        let s2 = [3.0];
        let result = batch_dequantize_int2_to_f32(
            &[p1.as_slice(), p2.as_slice()],
            &[s1.as_slice(), s2.as_slice()],
            4,
            4,
        )
        .unwrap();
        assert_eq!(result.len(), 2);
        assert_close(&result[0], &[2.0, -2.0, 0.0, 2.0], 1e-6);
        assert_close(&result[1], &[-3.0, 3.0, 3.0, 0.0], 1e-6);
    }

    #[test]
    fn batch_mismatched_lengths() {
        let p1 = pack_int2(&[1]);
        assert!(batch_dequantize_int2_to_f32(&[p1.as_slice()], &[], 1, 1,).is_err());
    }

    // ── Quantize helper validation ───────────────────────────────

    #[test]
    fn quantize_int2_rejects_zero_block() {
        assert!(quantize_to_int2(&[1.0], 0).is_err());
    }

    #[test]
    fn quantize_int4_rejects_zero_block() {
        assert!(quantize_to_int4(&[1.0], 0).is_err());
    }

    #[test]
    fn quantize_int8_rejects_zero_block() {
        assert!(quantize_to_int8(&[1.0], 0).is_err());
    }

    // ── f16 conversion correctness ───────────────────────────────

    #[test]
    fn f16_roundtrip_special_values() {
        for val in [0.0_f32, 1.0, -1.0, 0.5, -0.5, 65504.0] {
            let bits = f32_to_f16(val);
            let back = f16_to_f32(bits);
            assert!((val - back).abs() < 0.01, "f16 roundtrip failed for {val}: got {back}");
        }
    }

    #[test]
    fn f16_zero_preserved() {
        let bits = f32_to_f16(0.0);
        let back = f16_to_f32(bits);
        assert_eq!(back, 0.0);
    }

    #[test]
    fn f16_negative_zero() {
        let bits = f32_to_f16(-0.0);
        let back = f16_to_f32(bits);
        assert!(back == 0.0 || back == -0.0);
    }

    // ── INT4 codec correctness ───────────────────────────────────

    #[test]
    fn int4_decode_encode_roundtrip() {
        for v in -8..=7_i8 {
            let nibble = encode_int4(v);
            let decoded = decode_int4(nibble);
            assert_eq!(v, decoded, "INT4 roundtrip failed for {v}");
        }
    }

    #[test]
    fn int4_decode_boundary_values() {
        assert_eq!(decode_int4(0x00), 0);
        assert_eq!(decode_int4(0x07), 7);
        assert_eq!(decode_int4(0x08), -8);
        assert_eq!(decode_int4(0x0F), -1);
    }

    // ── INT2 codec correctness ───────────────────────────────────

    #[test]
    fn int2_decode_encode_roundtrip() {
        for &v in &[-1_i8, 0, 1] {
            let bits = encode_int2(v);
            let decoded = decode_int2(bits);
            assert_eq!(v, decoded, "INT2 roundtrip failed for {v}");
        }
    }
}
