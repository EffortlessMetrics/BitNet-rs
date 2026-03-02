//! CUDA-style quantization kernels with CPU reference implementations.
//!
//! # Overview
//!
//! This module provides quantization and dequantization routines used by the
//! BitNet inference pipeline.  The implementations here are CPU reference paths
//! that mirror the behaviour of their CUDA counterparts so that round-trip
//! fidelity can be validated without GPU hardware.
//!
//! ## Supported schemes
//!
//! * **Symmetric quantization** – uniform grid centred on zero.
//! * **Absmax quantization** – per-group scaling by absolute maximum.
//! * **Ternary quantization** – BitNet-style {-1, 0, 1} encoding.
//!
//! ## GPU path (feature `gpu` or `cuda`)
//!
//! A future CUDA kernel will implement the same logic using one thread-block
//! per quantization group with shared-memory reduction for scale computation.
//!
//! ## CPU fallback
//!
//! All public functions in this module are available on every platform and
//! serve as the golden reference for correctness testing.

use std::fmt;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configures quantization behaviour.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizeConfig {
    /// Number of quantization bits (e.g. 2, 4, 8).
    pub bits: u8,
    /// Whether the quantization grid is symmetric around zero.
    pub symmetric: bool,
    /// Number of elements per quantization group.
    pub group_size: usize,
    /// When `true`, each output channel gets its own scale factor.
    pub per_channel: bool,
}

impl QuantizeConfig {
    /// Create a new symmetric quantization config.
    pub fn symmetric(bits: u8, group_size: usize) -> Self {
        Self { bits, symmetric: true, group_size, per_channel: false }
    }

    /// Create a per-channel config.
    pub fn per_channel(bits: u8, group_size: usize) -> Self {
        Self { bits, symmetric: true, group_size, per_channel: true }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by quantization routines.
#[derive(Debug, Clone, PartialEq)]
pub enum QuantizeError {
    /// Input length is not a multiple of the group size.
    InvalidGroupSize { input_len: usize, group_size: usize },
    /// Bit width is unsupported.
    UnsupportedBitWidth(u8),
    /// Empty input.
    EmptyInput,
    /// Scale/data length mismatch during dequantization.
    LengthMismatch { data_len: usize, scale_count: usize, group_size: usize },
}

impl fmt::Display for QuantizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGroupSize { input_len, group_size } => {
                write!(f, "input length {input_len} is not a multiple of group size {group_size}")
            }
            Self::UnsupportedBitWidth(b) => write!(f, "unsupported bit width: {b}"),
            Self::EmptyInput => write!(f, "input is empty"),
            Self::LengthMismatch { data_len, scale_count, group_size } => write!(
                f,
                "length mismatch: data_len={data_len}, \
                 scale_count={scale_count}, group_size={group_size}"
            ),
        }
    }
}

impl std::error::Error for QuantizeError {}

// ---------------------------------------------------------------------------
// Quantized block
// ---------------------------------------------------------------------------

/// A single quantized group.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedBlock {
    /// Packed quantized values.
    pub data: Vec<u8>,
    /// Scale factor used to reconstruct floating-point values.
    pub scale: f32,
    /// Zero-point offset (0.0 for symmetric schemes).
    pub zero_point: f32,
    /// Number of original elements this block represents.
    pub block_size: usize,
}

// ---------------------------------------------------------------------------
// Quality metrics
// ---------------------------------------------------------------------------

/// Quantization fidelity metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationQuality {
    /// Mean squared error.
    pub mse: f64,
    /// Maximum absolute error.
    pub max_error: f64,
    /// Signal-to-quantization-noise ratio in dB.
    pub sqnr: f64,
    /// Number of elements compared.
    pub num_elements: usize,
}

// ---------------------------------------------------------------------------
// Symmetric quantization
// ---------------------------------------------------------------------------

/// Quantize `input` using symmetric uniform quantization.
///
/// Values are mapped to the integer range `[-(2^(bits-1)-1), 2^(bits-1)-1]`
/// with a per-group scale derived from the absolute maximum.
pub fn quantize_symmetric(
    input: &[f32],
    bits: u8,
    group_size: usize,
) -> Result<Vec<QuantizedBlock>, QuantizeError> {
    if input.is_empty() {
        return Err(QuantizeError::EmptyInput);
    }
    if !(2..=8).contains(&bits) {
        return Err(QuantizeError::UnsupportedBitWidth(bits));
    }
    if group_size == 0 || !input.len().is_multiple_of(group_size) {
        return Err(QuantizeError::InvalidGroupSize { input_len: input.len(), group_size });
    }

    let qmax = ((1i32 << (bits - 1)) - 1) as f32;
    let mut blocks = Vec::with_capacity(input.len() / group_size);

    for chunk in input.chunks_exact(group_size) {
        let absmax = chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

        let scale = if absmax == 0.0 { 1.0 } else { absmax / qmax };

        let data: Vec<u8> = chunk
            .iter()
            .map(|&v| {
                let q = (v / scale).round().clamp(-qmax, qmax);
                // Store as biased u8: value + 128
                (q as i16 + 128) as u8
            })
            .collect();

        blocks.push(QuantizedBlock { data, scale, zero_point: 0.0, block_size: group_size });
    }

    Ok(blocks)
}

/// Dequantize blocks produced by [`quantize_symmetric`].
pub fn dequantize_symmetric(blocks: &[QuantizedBlock]) -> Vec<f32> {
    let mut output = Vec::with_capacity(blocks.iter().map(|b| b.block_size).sum());

    for block in blocks {
        for &byte in &block.data {
            let q = byte as i16 - 128;
            output.push(q as f32 * block.scale);
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Absmax quantization
// ---------------------------------------------------------------------------

/// Quantize `input` to `i8` using per-group absolute-maximum scaling.
///
/// Each group is scaled so that `absmax` maps to `±127`.
pub fn quantize_absmax(
    input: &[f32],
    group_size: usize,
) -> Result<(Vec<i8>, Vec<f32>), QuantizeError> {
    if input.is_empty() {
        return Err(QuantizeError::EmptyInput);
    }
    if group_size == 0 || !input.len().is_multiple_of(group_size) {
        return Err(QuantizeError::InvalidGroupSize { input_len: input.len(), group_size });
    }

    let num_groups = input.len() / group_size;
    let mut data = Vec::with_capacity(input.len());
    let mut scales = Vec::with_capacity(num_groups);

    for chunk in input.chunks_exact(group_size) {
        let absmax = chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

        let scale = if absmax == 0.0 { 1.0 } else { absmax / 127.0 };
        scales.push(scale);

        for &v in chunk {
            let q = (v / scale).round().clamp(-127.0, 127.0) as i8;
            data.push(q);
        }
    }

    Ok((data, scales))
}

/// Dequantize data produced by [`quantize_absmax`].
pub fn dequantize_absmax(
    data: &[i8],
    scales: &[f32],
    group_size: usize,
) -> Result<Vec<f32>, QuantizeError> {
    if scales.is_empty() && data.is_empty() {
        return Ok(Vec::new());
    }
    if data.len() != scales.len() * group_size {
        return Err(QuantizeError::LengthMismatch {
            data_len: data.len(),
            scale_count: scales.len(),
            group_size,
        });
    }

    let mut output = Vec::with_capacity(data.len());
    for (chunk, &scale) in data.chunks_exact(group_size).zip(scales.iter()) {
        for &q in chunk {
            output.push(q as f32 * scale);
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Ternary quantization (BitNet-style)
// ---------------------------------------------------------------------------

/// Quantize `input` to ternary values {-1, 0, 1} using the mean absolute
/// value as threshold (BitNet RTE strategy).
///
/// Returns the ternary codes and the scale factor (mean of absolute values).
pub fn quantize_ternary(input: &[f32]) -> Result<(Vec<i8>, f32), QuantizeError> {
    if input.is_empty() {
        return Err(QuantizeError::EmptyInput);
    }

    let mean_abs: f32 = input.iter().map(|v| v.abs()).sum::<f32>() / input.len() as f32;

    let scale = if mean_abs == 0.0 { 1.0 } else { mean_abs };

    let codes: Vec<i8> = input
        .iter()
        .map(|&v| {
            if v > scale * 0.5 {
                1
            } else if v < -scale * 0.5 {
                -1
            } else {
                0
            }
        })
        .collect();

    Ok((codes, scale))
}

/// Dequantize ternary codes back to floating-point.
pub fn dequantize_ternary(codes: &[i8], scale: f32) -> Vec<f32> {
    codes.iter().map(|&c| c as f32 * scale).collect()
}

// ---------------------------------------------------------------------------
// Scale factor computation
// ---------------------------------------------------------------------------

/// Compute per-group absolute-maximum scale factors.
pub fn compute_scale_factors(input: &[f32], group_size: usize) -> Result<Vec<f32>, QuantizeError> {
    if input.is_empty() {
        return Err(QuantizeError::EmptyInput);
    }
    if group_size == 0 || !input.len().is_multiple_of(group_size) {
        return Err(QuantizeError::InvalidGroupSize { input_len: input.len(), group_size });
    }

    Ok(input
        .chunks_exact(group_size)
        .map(|chunk| chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max))
        .collect())
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Compare `original` and `reconstructed` signals and return quality metrics.
pub fn validate_quantization_error(
    original: &[f32],
    reconstructed: &[f32],
) -> Result<QuantizationQuality, QuantizeError> {
    if original.is_empty() || reconstructed.is_empty() {
        return Err(QuantizeError::EmptyInput);
    }
    if original.len() != reconstructed.len() {
        return Err(QuantizeError::LengthMismatch {
            data_len: original.len(),
            scale_count: reconstructed.len(),
            group_size: 1,
        });
    }

    let n = original.len() as f64;
    let mut sum_sq_err = 0.0f64;
    let mut max_err = 0.0f64;
    let mut sum_sq_signal = 0.0f64;

    for (&o, &r) in original.iter().zip(reconstructed.iter()) {
        let err = (o as f64) - (r as f64);
        sum_sq_err += err * err;
        let ae = err.abs();
        if ae > max_err {
            max_err = ae;
        }
        sum_sq_signal += (o as f64) * (o as f64);
    }

    let mse = sum_sq_err / n;
    let sqnr =
        if sum_sq_err == 0.0 { f64::INFINITY } else { 10.0 * (sum_sq_signal / sum_sq_err).log10() };

    Ok(QuantizationQuality { mse, max_error: max_err, sqnr, num_elements: original.len() })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // QuantizeConfig
    // -----------------------------------------------------------------------

    #[test]
    fn config_symmetric_defaults() {
        let c = QuantizeConfig::symmetric(8, 32);
        assert_eq!(c.bits, 8);
        assert!(c.symmetric);
        assert_eq!(c.group_size, 32);
        assert!(!c.per_channel);
    }

    #[test]
    fn config_per_channel() {
        let c = QuantizeConfig::per_channel(4, 64);
        assert!(c.per_channel);
        assert!(c.symmetric);
    }

    #[test]
    fn config_clone_eq() {
        let a = QuantizeConfig::symmetric(2, 16);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn config_debug_format() {
        let c = QuantizeConfig::symmetric(8, 32);
        let dbg = format!("{c:?}");
        assert!(dbg.contains("QuantizeConfig"));
    }

    // -----------------------------------------------------------------------
    // QuantizeError
    // -----------------------------------------------------------------------

    #[test]
    fn error_display_invalid_group() {
        let e = QuantizeError::InvalidGroupSize { input_len: 10, group_size: 3 };
        let msg = format!("{e}");
        assert!(msg.contains("10"));
        assert!(msg.contains("3"));
    }

    #[test]
    fn error_display_unsupported_bits() {
        let e = QuantizeError::UnsupportedBitWidth(0);
        assert!(format!("{e}").contains("0"));
    }

    #[test]
    fn error_display_empty() {
        let e = QuantizeError::EmptyInput;
        assert!(format!("{e}").contains("empty"));
    }

    #[test]
    fn error_display_length_mismatch() {
        let e = QuantizeError::LengthMismatch { data_len: 10, scale_count: 3, group_size: 4 };
        assert!(format!("{e}").contains("mismatch"));
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(QuantizeError::EmptyInput);
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn error_clone_eq() {
        let a = QuantizeError::EmptyInput;
        let b = a.clone();
        assert_eq!(a, b);
    }

    // -----------------------------------------------------------------------
    // Symmetric quantization
    // -----------------------------------------------------------------------

    #[test]
    fn symmetric_basic_round_trip() {
        let input = vec![1.0, -1.0, 0.5, -0.5];
        let blocks = quantize_symmetric(&input, 8, 4).unwrap();
        let output = dequantize_symmetric(&blocks);
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 0.02, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn symmetric_zero_input() {
        let input = vec![0.0; 8];
        let blocks = quantize_symmetric(&input, 8, 4).unwrap();
        let output = dequantize_symmetric(&blocks);
        assert!(output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn symmetric_single_group() {
        let input = vec![3.0, -3.0, 1.5, 0.0];
        let blocks = quantize_symmetric(&input, 8, 4).unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].block_size, 4);
        assert_eq!(blocks[0].zero_point, 0.0);
    }

    #[test]
    fn symmetric_multiple_groups() {
        let input: Vec<f32> = (0..16).map(|i| i as f32 - 8.0).collect();
        let blocks = quantize_symmetric(&input, 8, 4).unwrap();
        assert_eq!(blocks.len(), 4);
    }

    #[test]
    fn symmetric_2bit() {
        let input = vec![1.0, -1.0, 0.0, 0.5];
        let blocks = quantize_symmetric(&input, 2, 4).unwrap();
        let output = dequantize_symmetric(&blocks);
        // 2-bit symmetric: range is [-1, 1], so limited precision
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn symmetric_4bit() {
        let input = vec![1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.0, 0.75];
        let blocks = quantize_symmetric(&input, 4, 8).unwrap();
        let output = dequantize_symmetric(&blocks);
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 0.15, "4-bit mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn symmetric_err_empty() {
        assert_eq!(quantize_symmetric(&[], 8, 4), Err(QuantizeError::EmptyInput));
    }

    #[test]
    fn symmetric_err_bad_bits_low() {
        assert_eq!(quantize_symmetric(&[1.0], 1, 1), Err(QuantizeError::UnsupportedBitWidth(1)));
    }

    #[test]
    fn symmetric_err_bad_bits_high() {
        assert_eq!(quantize_symmetric(&[1.0], 9, 1), Err(QuantizeError::UnsupportedBitWidth(9)));
    }

    #[test]
    fn symmetric_err_group_not_divisible() {
        assert_eq!(
            quantize_symmetric(&[1.0, 2.0, 3.0], 8, 2),
            Err(QuantizeError::InvalidGroupSize { input_len: 3, group_size: 2 })
        );
    }

    #[test]
    fn symmetric_err_group_zero() {
        assert_eq!(
            quantize_symmetric(&[1.0], 8, 0),
            Err(QuantizeError::InvalidGroupSize { input_len: 1, group_size: 0 })
        );
    }

    #[test]
    fn symmetric_large_values() {
        let input = vec![1e6, -1e6, 5e5, -5e5];
        let blocks = quantize_symmetric(&input, 8, 4).unwrap();
        let output = dequantize_symmetric(&blocks);
        for (a, b) in input.iter().zip(output.iter()) {
            let rel = (a - b).abs() / a.abs().max(1.0);
            assert!(rel < 0.01, "large value mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn symmetric_dequant_empty() {
        let output = dequantize_symmetric(&[]);
        assert!(output.is_empty());
    }

    // -----------------------------------------------------------------------
    // Absmax quantization
    // -----------------------------------------------------------------------

    #[test]
    fn absmax_basic_round_trip() {
        let input = vec![1.0, -1.0, 0.5, -0.5];
        let (data, scales) = quantize_absmax(&input, 4).unwrap();
        let output = dequantize_absmax(&data, &scales, 4).unwrap();
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 0.02, "absmax mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn absmax_zero_input() {
        let input = vec![0.0; 8];
        let (data, scales) = quantize_absmax(&input, 4).unwrap();
        let output = dequantize_absmax(&data, &scales, 4).unwrap();
        assert!(output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn absmax_multiple_groups() {
        let input: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.1).collect();
        let (data, scales) = quantize_absmax(&input, 4).unwrap();
        assert_eq!(scales.len(), 4);
        assert_eq!(data.len(), 16);
    }

    #[test]
    fn absmax_err_empty() {
        assert_eq!(quantize_absmax(&[], 4), Err(QuantizeError::EmptyInput));
    }

    #[test]
    fn absmax_err_group_mismatch() {
        assert_eq!(
            quantize_absmax(&[1.0, 2.0, 3.0], 2),
            Err(QuantizeError::InvalidGroupSize { input_len: 3, group_size: 2 })
        );
    }

    #[test]
    fn dequant_absmax_err_length() {
        let data = vec![1i8; 8];
        let scales = vec![1.0; 3];
        assert!(dequantize_absmax(&data, &scales, 4).is_err());
    }

    #[test]
    fn dequant_absmax_empty() {
        let out = dequantize_absmax(&[], &[], 1).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn absmax_clamp_to_127() {
        // Verify values are clamped within i8 range.
        let input = vec![127.0 * 0.5; 4];
        let (data, _) = quantize_absmax(&input, 4).unwrap();
        assert!(data.iter().all(|&v| v <= 127 && v >= -127));
    }

    // -----------------------------------------------------------------------
    // Ternary quantization
    // -----------------------------------------------------------------------

    #[test]
    fn ternary_basic() {
        let input = vec![1.0, -1.0, 0.0, 0.01];
        let (codes, scale) = quantize_ternary(&input).unwrap();
        assert!(scale > 0.0);
        assert!(codes.iter().all(|&c| c == -1 || c == 0 || c == 1));
    }

    #[test]
    fn ternary_all_positive() {
        let input = vec![2.0, 3.0, 4.0, 5.0];
        let (codes, _) = quantize_ternary(&input).unwrap();
        assert!(codes.iter().all(|&c| c == 1));
    }

    #[test]
    fn ternary_all_negative() {
        let input = vec![-2.0, -3.0, -4.0, -5.0];
        let (codes, _) = quantize_ternary(&input).unwrap();
        assert!(codes.iter().all(|&c| c == -1));
    }

    #[test]
    fn ternary_all_zero() {
        let input = vec![0.0; 4];
        let (codes, _) = quantize_ternary(&input).unwrap();
        assert!(codes.iter().all(|&c| c == 0));
    }

    #[test]
    fn ternary_err_empty() {
        assert_eq!(quantize_ternary(&[]), Err(QuantizeError::EmptyInput));
    }

    #[test]
    fn ternary_round_trip_sign_preservation() {
        let input = vec![5.0, -5.0, 0.01, -0.01, 10.0];
        let (codes, scale) = quantize_ternary(&input).unwrap();
        let output = dequantize_ternary(&codes, scale);
        // Signs of large values must be preserved.
        assert!(output[0] > 0.0);
        assert!(output[1] < 0.0);
        assert!(output[4] > 0.0);
    }

    #[test]
    fn ternary_dequant_matches_codes() {
        let codes = vec![1i8, -1, 0, 1];
        let scale = 2.5;
        let out = dequantize_ternary(&codes, scale);
        assert_eq!(out, vec![2.5, -2.5, 0.0, 2.5]);
    }

    // -----------------------------------------------------------------------
    // Scale factors
    // -----------------------------------------------------------------------

    #[test]
    fn scale_factors_basic() {
        let input = vec![1.0, -2.0, 3.0, -4.0, 0.5, 0.5, 0.5, 0.5];
        let scales = compute_scale_factors(&input, 4).unwrap();
        assert_eq!(scales.len(), 2);
        assert!((scales[0] - 4.0).abs() < 1e-6);
        assert!((scales[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn scale_factors_err_empty() {
        assert_eq!(compute_scale_factors(&[], 4), Err(QuantizeError::EmptyInput));
    }

    #[test]
    fn scale_factors_err_group() {
        assert_eq!(
            compute_scale_factors(&[1.0, 2.0, 3.0], 2),
            Err(QuantizeError::InvalidGroupSize { input_len: 3, group_size: 2 })
        );
    }

    #[test]
    fn scale_factors_all_zero() {
        let scales = compute_scale_factors(&[0.0; 8], 4).unwrap();
        assert!(scales.iter().all(|&s| s == 0.0));
    }

    // -----------------------------------------------------------------------
    // Validation / quality
    // -----------------------------------------------------------------------

    #[test]
    fn quality_perfect_reconstruction() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let q = validate_quantization_error(&a, &a).unwrap();
        assert_eq!(q.mse, 0.0);
        assert_eq!(q.max_error, 0.0);
        assert!(q.sqnr.is_infinite());
        assert_eq!(q.num_elements, 4);
    }

    #[test]
    fn quality_known_error() {
        let orig = vec![1.0, 0.0];
        let recon = vec![0.9, 0.1];
        let q = validate_quantization_error(&orig, &recon).unwrap();
        // MSE = (0.01 + 0.01) / 2 = 0.01
        assert!((q.mse - 0.01).abs() < 1e-6);
        assert!((q.max_error - 0.1).abs() < 1e-6);
    }

    #[test]
    fn quality_err_empty_orig() {
        assert!(validate_quantization_error(&[], &[1.0]).is_err());
    }

    #[test]
    fn quality_err_empty_recon() {
        assert!(validate_quantization_error(&[1.0], &[]).is_err());
    }

    #[test]
    fn quality_err_length_mismatch() {
        assert!(validate_quantization_error(&[1.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn quality_sqnr_positive_for_noisy() {
        let orig: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let recon: Vec<f32> = orig.iter().map(|v| v + 0.1).collect();
        let q = validate_quantization_error(&orig, &recon).unwrap();
        assert!(q.sqnr > 0.0);
        assert!(q.sqnr.is_finite());
    }

    // -----------------------------------------------------------------------
    // Round-trip fidelity (end-to-end)
    // -----------------------------------------------------------------------

    #[test]
    fn round_trip_symmetric_8bit_fidelity() {
        let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let blocks = quantize_symmetric(&input, 8, 32).unwrap();
        let output = dequantize_symmetric(&blocks);
        let q = validate_quantization_error(&input, &output).unwrap();
        assert!(q.mse < 1e-4, "8-bit MSE too high: {}", q.mse);
        assert!(q.sqnr > 30.0, "8-bit SQNR too low: {}", q.sqnr);
    }

    #[test]
    fn round_trip_absmax_fidelity() {
        let input: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 64.0).collect();
        let (data, scales) = quantize_absmax(&input, 32).unwrap();
        let output = dequantize_absmax(&data, &scales, 32).unwrap();
        let q = validate_quantization_error(&input, &output).unwrap();
        assert!(q.mse < 1e-4, "absmax MSE too high: {}", q.mse);
    }

    #[test]
    fn round_trip_ternary_bounded_error() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();
        let (codes, scale) = quantize_ternary(&input).unwrap();
        let output = dequantize_ternary(&codes, scale);
        let q = validate_quantization_error(&input, &output).unwrap();
        // Ternary is very coarse; just verify it doesn't blow up.
        assert!(q.mse.is_finite());
        assert!(q.max_error.is_finite());
    }

    #[test]
    fn round_trip_symmetric_preserves_length() {
        let input = vec![0.1; 64];
        let blocks = quantize_symmetric(&input, 4, 16).unwrap();
        let output = dequantize_symmetric(&blocks);
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn round_trip_absmax_preserves_length() {
        let input = vec![0.1; 64];
        let (data, scales) = quantize_absmax(&input, 16).unwrap();
        let output = dequantize_absmax(&data, &scales, 16).unwrap();
        assert_eq!(output.len(), input.len());
    }
}

// ===========================================================================
// Property-based tests
// ===========================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Generate a vector whose length is a multiple of `group_size`.
    fn aligned_vec(group_size: usize) -> impl Strategy<Value = Vec<f32>> {
        (1..=8usize).prop_flat_map(move |groups| {
            let len = groups * group_size;
            proptest::collection::vec(-100.0f32..100.0, len)
        })
    }

    proptest! {
        /// Symmetric round-trip never changes the vector length.
        #[test]
        fn prop_symmetric_length(input in aligned_vec(32)) {
            let blocks = quantize_symmetric(&input, 8, 32).unwrap();
            let output = dequantize_symmetric(&blocks);
            prop_assert_eq!(output.len(), input.len());
        }

        /// Absmax round-trip never changes the vector length.
        #[test]
        fn prop_absmax_length(input in aligned_vec(32)) {
            let (data, scales) = quantize_absmax(&input, 32).unwrap();
            let output = dequantize_absmax(&data, &scales, 32).unwrap();
            prop_assert_eq!(output.len(), input.len());
        }

        /// 8-bit symmetric MSE stays below a sensible bound.
        #[test]
        fn prop_symmetric_8bit_mse(input in aligned_vec(32)) {
            let blocks = quantize_symmetric(&input, 8, 32).unwrap();
            let output = dequantize_symmetric(&blocks);
            let q = validate_quantization_error(&input, &output).unwrap();
            // Relative bound: MSE should be small relative to signal energy.
            let energy: f64 = input.iter().map(|v| (*v as f64).powi(2)).sum::<f64>()
                / input.len() as f64;
            // For all-zero, MSE is 0; otherwise MSE < 1% of signal energy.
            prop_assert!(
                q.mse <= energy * 0.01 + 1e-6,
                "MSE {} exceeded 1% of energy {}", q.mse, energy
            );
        }

        /// Ternary codes are always in {-1, 0, 1}.
        #[test]
        fn prop_ternary_codes(
            input in proptest::collection::vec(-10.0f32..10.0, 1..256)
        ) {
            let (codes, scale) = quantize_ternary(&input).unwrap();
            prop_assert!(scale > 0.0);
            for &c in &codes {
                prop_assert!(c == -1 || c == 0 || c == 1);
            }
        }

        /// Scale factors are non-negative and ≤ absmax of input.
        #[test]
        fn prop_scale_factors_bounded(input in aligned_vec(16)) {
            let scales = compute_scale_factors(&input, 16).unwrap();
            let global_max = input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            for &s in &scales {
                prop_assert!(s >= 0.0);
                prop_assert!(s <= global_max + 1e-6);
            }
        }
    }
}
