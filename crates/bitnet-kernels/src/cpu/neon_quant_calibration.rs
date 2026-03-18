//! NEON-optimized quantization calibration kernels for Apple Silicon.
//!
//! Pure Rust implementation using scalar operations that auto-vectorize
//! on AArch64 targets. Provides calibration primitives for symmetric/asymmetric
//! quantization, BitNet I2_S encoding, histogram-based KL calibration,
//! SmoothQuant, and MSE loss measurement.

// ── Absmax ──────────────────────────────────────────────────────────

/// Find the absolute maximum value in `data` for symmetric quantization scale.
///
/// Returns `0.0` for empty slices.
#[inline]
pub fn compute_absmax(data: &[f32]) -> f32 {
    let mut max = 0.0_f32;
    for &v in data {
        let a = v.abs();
        if a > max {
            max = a;
        }
    }
    max
}

// ── Asymmetric scale + zero-point ───────────────────────────────────

/// Compute scale and zero-point for asymmetric quantization.
///
/// `num_bits` determines the quantized range `[0, 2^num_bits - 1]`.
/// Returns `(scale, zero_point)` where `scale > 0` unless the data range
/// is zero (in which case `scale = 1.0` and `zero_point = 0`).
pub fn compute_scale_zeropoint(data: &[f32], num_bits: u8) -> (f32, i32) {
    if data.is_empty() {
        return (1.0, 0);
    }

    let mut min_val = data[0];
    let mut max_val = data[0];
    for &v in &data[1..] {
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    }

    let qmax = ((1u64 << num_bits) - 1) as f32;
    let range = max_val - min_val;

    if range == 0.0 {
        return (1.0, 0);
    }

    let scale = range / qmax;
    let zero_point = (-min_val / scale).round() as i32;

    (scale, zero_point)
}

// ── Per-channel scales ──────────────────────────────────────────────

/// Compute per-channel absmax scales.
///
/// `data` is interpreted as `channels` contiguous blocks each of length
/// `channel_size`. Returns one scale per channel.
///
/// # Panics
///
/// Panics if `data.len() != channels * channel_size`.
pub fn compute_per_channel_scales(data: &[f32], channels: usize, channel_size: usize) -> Vec<f32> {
    assert_eq!(
        data.len(),
        channels * channel_size,
        "data length must equal channels * channel_size"
    );

    let mut scales = Vec::with_capacity(channels);
    for ch in 0..channels {
        let start = ch * channel_size;
        let end = start + channel_size;
        scales.push(compute_absmax(&data[start..end]));
    }
    scales
}

// ── Per-group scales ────────────────────────────────────────────────

/// Compute group-wise quantization scales (GPTQ-style).
///
/// Splits `data` into groups of `group_size` and computes absmax for each.
/// The last group may be smaller if `data.len()` is not a multiple of
/// `group_size`.
///
/// # Panics
///
/// Panics if `group_size == 0`.
pub fn compute_per_group_scales(data: &[f32], group_size: usize) -> Vec<f32> {
    assert!(group_size > 0, "group_size must be > 0");

    data.chunks(group_size).map(compute_absmax).collect()
}

// ── Symmetric int8 quantization ─────────────────────────────────────

/// Quantize `data` symmetrically to int8.
///
/// `output[i] = clamp(round(data[i] / scale), -128, 127)`
///
/// # Panics
///
/// Panics if `data.len() != output.len()`.
pub fn quantize_symmetric_i8(data: &[f32], scale: f32, output: &mut [i8]) {
    assert_eq!(data.len(), output.len(), "data and output must have equal length");

    let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };
    for (o, &v) in output.iter_mut().zip(data.iter()) {
        let q = (v * inv_scale).round();
        *o = q.clamp(-128.0, 127.0) as i8;
    }
}

// ── Symmetric int2 quantization (BitNet I2_S) ───────────────────────

/// Quantize `data` symmetrically to 2-bit using BitNet I2_S encoding.
///
/// Encoding per 2-bit pair: `0b00` = 0, `0b01` = +1, `0b11` = −1.
///
/// Each output byte packs 4 values (LSB-first): bits \[1:0\] = element 0,
/// bits \[3:2\] = element 1, bits \[5:4\] = element 2, bits \[7:6\] = element 3.
///
/// # Panics
///
/// Panics if `output.len() < ceil(data.len() / 4)`.
pub fn quantize_symmetric_i2(data: &[f32], scale: f32, output: &mut [u8]) {
    let needed = data.len().div_ceil(4);
    assert!(
        output.len() >= needed,
        "output must have at least {} bytes, got {}",
        needed,
        output.len()
    );

    let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

    for (byte_idx, chunk) in data.chunks(4).enumerate() {
        let mut byte: u8 = 0;
        for (j, &v) in chunk.iter().enumerate() {
            let q = (v * inv_scale).round() as i32;
            let bits: u8 = match q.signum() {
                1 => 0b01,  // +1
                -1 => 0b11, // −1
                _ => 0b00,  // 0
            };
            byte |= bits << (j * 2);
        }
        output[byte_idx] = byte;
    }
}

// ── Int8 dequantization ─────────────────────────────────────────────

/// Dequantize int8 values: `output[i] = data[i] as f32 * scale`.
///
/// # Panics
///
/// Panics if `data.len() != output.len()`.
pub fn dequantize_i8(data: &[i8], scale: f32, output: &mut [f32]) {
    assert_eq!(data.len(), output.len(), "data and output must have equal length");

    for (o, &v) in output.iter_mut().zip(data.iter()) {
        *o = v as f32 * scale;
    }
}

// ── Int2 (BitNet I2_S) dequantization ───────────────────────────────

/// Dequantize 2-bit BitNet I2_S encoded values.
///
/// Each input byte contains 4 packed 2-bit values (LSB-first).
/// Encoding: `0b00` = 0, `0b01` = +1, `0b11` = −1.
///
/// Writes exactly `output.len()` elements.
///
/// # Panics
///
/// Panics if `data.len() < ceil(output.len() / 4)`.
pub fn dequantize_i2(data: &[u8], scale: f32, output: &mut [f32]) {
    let needed = output.len().div_ceil(4);
    assert!(
        data.len() >= needed,
        "data must have at least {} bytes for {} outputs",
        needed,
        output.len()
    );

    let mut out_idx = 0;
    for &byte in data {
        for j in 0..4 {
            if out_idx >= output.len() {
                return;
            }
            let bits = (byte >> (j * 2)) & 0b11;
            let val: f32 = match bits {
                0b01 => 1.0,
                0b11 => -1.0,
                _ => 0.0, // 0b00 and 0b10 both map to 0
            };
            output[out_idx] = val * scale;
            out_idx += 1;
        }
    }
}

// ── Histogram for KL-divergence calibration ─────────────────────────

/// Build a histogram of absolute values in `data` for KL-divergence calibration.
///
/// Values are linearly mapped into `num_bins` bins spanning `[0, absmax]`.
/// Returns a vector of length `num_bins`. Out-of-range values are clamped
/// to the last bin.
///
/// Returns an empty vector if `num_bins == 0` or `data` is empty.
pub fn calibrate_histogram(data: &[f32], num_bins: usize) -> Vec<u64> {
    if num_bins == 0 || data.is_empty() {
        return vec![0u64; num_bins];
    }

    let absmax = compute_absmax(data);
    let mut hist = vec![0u64; num_bins];

    if absmax == 0.0 {
        // All values are zero — everything falls in bin 0.
        hist[0] = data.len() as u64;
        return hist;
    }

    let inv_range = num_bins as f64 / absmax as f64;
    for &v in data {
        let idx = (v.abs() as f64 * inv_range) as usize;
        let idx = idx.min(num_bins - 1);
        hist[idx] += 1;
    }
    hist
}

// ── KL divergence ───────────────────────────────────────────────────

/// Compute KL divergence D_KL(P || Q) = Σ p_i * ln(p_i / q_i).
///
/// Entries where `p_i == 0` contribute 0. Entries where `p_i > 0` and
/// `q_i == 0` contribute `f64::INFINITY`.
///
/// # Panics
///
/// Panics if `p.len() != q.len()`.
pub fn compute_kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "p and q must have equal length");

    let mut kl = 0.0_f64;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi > 0.0 {
            if qi == 0.0 {
                return f64::INFINITY;
            }
            kl += pi * (pi / qi).ln();
        }
    }
    kl
}

// ── SmoothQuant ─────────────────────────────────────────────────────

/// Compute SmoothQuant migration scales.
///
/// `output[i] = act_scales[i]^alpha / weight_scales[i]^(1 - alpha)`
///
/// This balances the quantization difficulty between activations and weights.
///
/// # Panics
///
/// Panics if `act_scales`, `weight_scales`, and `output` don't all have the
/// same length.
pub fn smooth_quant_scales(
    act_scales: &[f32],
    weight_scales: &[f32],
    alpha: f32,
    output: &mut [f32],
) {
    assert_eq!(
        act_scales.len(),
        weight_scales.len(),
        "act_scales and weight_scales must have equal length"
    );
    assert_eq!(act_scales.len(), output.len(), "scales and output must have equal length");

    let beta = 1.0 - alpha;
    for ((o, &a), &w) in output.iter_mut().zip(act_scales.iter()).zip(weight_scales.iter()) {
        *o = a.powf(alpha) / w.powf(beta);
    }
}

// ── MSE loss ────────────────────────────────────────────────────────

/// Compute mean squared error between original and quantized tensors.
///
/// Returns `0.0` for empty slices.
///
/// # Panics
///
/// Panics if `original.len() != quantized.len()`.
pub fn compute_mse_loss(original: &[f32], quantized: &[f32]) -> f32 {
    assert_eq!(original.len(), quantized.len(), "original and quantized must have equal length");

    if original.is_empty() {
        return 0.0;
    }

    let mut sum = 0.0_f64;
    for (&o, &q) in original.iter().zip(quantized.iter()) {
        let d = (o - q) as f64;
        sum += d * d;
    }
    (sum / original.len() as f64) as f32
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── compute_absmax ──────────────────────────────────────────────

    #[test]
    fn test_absmax_positive() {
        assert_eq!(compute_absmax(&[1.0, 2.0, 3.0]), 3.0);
    }

    #[test]
    fn test_absmax_negative() {
        assert_eq!(compute_absmax(&[-5.0, -1.0, 2.0]), 5.0);
    }

    #[test]
    fn test_absmax_empty() {
        assert_eq!(compute_absmax(&[]), 0.0);
    }

    #[test]
    fn test_absmax_single() {
        assert_eq!(compute_absmax(&[-7.5]), 7.5);
    }

    #[test]
    fn test_absmax_all_zeros() {
        assert_eq!(compute_absmax(&[0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn test_absmax_mixed() {
        assert_eq!(compute_absmax(&[-3.0, 0.0, 2.5, -1.0]), 3.0);
    }

    #[test]
    fn test_absmax_large_values() {
        assert_eq!(compute_absmax(&[1e30, -1e31, 1e20]), 1e31);
    }

    #[test]
    fn test_absmax_small_values() {
        let v = compute_absmax(&[1e-30, -1e-31, 1e-32]);
        assert!((v - 1e-30).abs() < 1e-40);
    }

    // ── compute_scale_zeropoint ─────────────────────────────────────

    #[test]
    fn test_scale_zp_basic_8bit() {
        let data = [0.0, 1.0, 2.0, 3.0];
        let (scale, zp) = compute_scale_zeropoint(&data, 8);
        // range = 3.0, qmax = 255, scale = 3/255
        assert!((scale - 3.0 / 255.0).abs() < 1e-6);
        assert_eq!(zp, 0); // min=0 → zp = -0/scale = 0
    }

    #[test]
    fn test_scale_zp_negative_range() {
        let data = [-2.0, -1.0, 0.0, 1.0];
        let (scale, _zp) = compute_scale_zeropoint(&data, 8);
        assert!((scale - 3.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_zp_constant() {
        let data = [5.0, 5.0, 5.0];
        let (scale, zp) = compute_scale_zeropoint(&data, 8);
        assert_eq!(scale, 1.0);
        assert_eq!(zp, 0);
    }

    #[test]
    fn test_scale_zp_empty() {
        let (scale, zp) = compute_scale_zeropoint(&[], 8);
        assert_eq!(scale, 1.0);
        assert_eq!(zp, 0);
    }

    #[test]
    fn test_scale_zp_single() {
        let (scale, zp) = compute_scale_zeropoint(&[3.0], 8);
        // single value → range=0
        assert_eq!(scale, 1.0);
        assert_eq!(zp, 0);
    }

    #[test]
    fn test_scale_positive() {
        // Scale should always be positive when range > 0
        let data = [-10.0, 5.0];
        let (scale, _) = compute_scale_zeropoint(&data, 8);
        assert!(scale > 0.0);
    }

    #[test]
    fn test_zeropoint_within_range() {
        let data = [-3.0, 5.0];
        let (_, zp) = compute_scale_zeropoint(&data, 8);
        // zp should be in [0, 255] for 8-bit
        assert!(zp >= 0 && zp <= 255);
    }

    // ── compute_per_channel_scales ──────────────────────────────────

    #[test]
    fn test_per_channel_basic() {
        let data = [1.0, -2.0, 3.0, -4.0, 0.5, -0.5];
        let scales = compute_per_channel_scales(&data, 2, 3);
        assert_eq!(scales.len(), 2);
        assert_eq!(scales[0], 3.0);
        assert_eq!(scales[1], 4.0);
    }

    #[test]
    fn test_per_channel_single_channel() {
        let data = [1.0, -5.0, 3.0];
        let scales = compute_per_channel_scales(&data, 1, 3);
        assert_eq!(scales, vec![5.0]);
    }

    #[test]
    #[should_panic(expected = "data length must equal channels * channel_size")]
    fn test_per_channel_mismatch() {
        compute_per_channel_scales(&[1.0, 2.0], 2, 3);
    }

    #[test]
    fn test_per_channel_zeros() {
        let data = [0.0, 0.0, 0.0, 0.0];
        let scales = compute_per_channel_scales(&data, 2, 2);
        assert_eq!(scales, vec![0.0, 0.0]);
    }

    // ── compute_per_group_scales ────────────────────────────────────

    #[test]
    fn test_per_group_basic() {
        let data = [1.0, -3.0, 2.0, -4.0, 5.0, 0.0];
        let scales = compute_per_group_scales(&data, 2);
        assert_eq!(scales, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_per_group_remainder() {
        let data = [1.0, -2.0, 3.0];
        let scales = compute_per_group_scales(&data, 2);
        assert_eq!(scales, vec![2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "group_size must be > 0")]
    fn test_per_group_zero_size() {
        compute_per_group_scales(&[1.0], 0);
    }

    #[test]
    fn test_per_group_single_element_groups() {
        let data = [1.0, -2.0, 3.0];
        let scales = compute_per_group_scales(&data, 1);
        assert_eq!(scales, vec![1.0, 2.0, 3.0]);
    }

    // ── quantize_symmetric_i8 ───────────────────────────────────────

    #[test]
    fn test_quant_i8_basic() {
        let data = [0.0, 1.0, -1.0, 0.5];
        let scale = 1.0 / 127.0;
        let mut out = [0i8; 4];
        quantize_symmetric_i8(&data, scale, &mut out);
        assert_eq!(out[0], 0);
        assert_eq!(out[1], 127);
        assert_eq!(out[2], -127);
        assert_eq!(out[3], 64); // round(0.5 * 127) = 64
    }

    #[test]
    fn test_quant_i8_clamp() {
        let data = [200.0, -200.0];
        let scale = 1.0;
        let mut out = [0i8; 2];
        quantize_symmetric_i8(&data, scale, &mut out);
        assert_eq!(out[0], 127);
        assert_eq!(out[1], -128);
    }

    #[test]
    fn test_quant_i8_zeros() {
        let data = [0.0, 0.0, 0.0];
        let mut out = [0i8; 3];
        quantize_symmetric_i8(&data, 1.0, &mut out);
        assert_eq!(out.to_vec(), vec![0, 0, 0]);
    }

    #[test]
    fn test_quant_i8_zero_scale() {
        let data = [1.0, 2.0];
        let mut out = [0i8; 2];
        quantize_symmetric_i8(&data, 0.0, &mut out);
        assert_eq!(out.to_vec(), vec![0, 0]);
    }

    #[test]
    #[should_panic(expected = "data and output must have equal length")]
    fn test_quant_i8_len_mismatch() {
        quantize_symmetric_i8(&[1.0], 1.0, &mut [0i8; 2]);
    }

    // ── quantize_symmetric_i2 (BitNet I2_S) ─────────────────────────

    #[test]
    fn test_quant_i2_encoding() {
        // 4 values → 1 byte
        let data = [0.0, 1.0, 0.0, -1.0];
        let scale = 1.0;
        let mut out = [0u8; 1];
        quantize_symmetric_i2(&data, scale, &mut out);
        // elem0=0→0b00, elem1=+1→0b01, elem2=0→0b00, elem3=-1→0b11
        // byte = 0b11_00_01_00 = 0xC4
        assert_eq!(out[0], 0b11_00_01_00);
    }

    #[test]
    fn test_quant_i2_all_positive() {
        let data = [1.0, 1.0, 1.0, 1.0];
        let mut out = [0u8; 1];
        quantize_symmetric_i2(&data, 1.0, &mut out);
        // all +1 → 0b01_01_01_01 = 0x55
        assert_eq!(out[0], 0x55);
    }

    #[test]
    fn test_quant_i2_all_negative() {
        let data = [-1.0, -1.0, -1.0, -1.0];
        let mut out = [0u8; 1];
        quantize_symmetric_i2(&data, 1.0, &mut out);
        // all -1 → 0b11_11_11_11 = 0xFF
        assert_eq!(out[0], 0xFF);
    }

    #[test]
    fn test_quant_i2_all_zeros() {
        let data = [0.0, 0.0, 0.0, 0.0];
        let mut out = [0u8; 1];
        quantize_symmetric_i2(&data, 1.0, &mut out);
        assert_eq!(out[0], 0x00);
    }

    #[test]
    fn test_quant_i2_partial_byte() {
        // 3 values → still needs 1 byte, 4th slot should be 0
        let data = [1.0, -1.0, 0.0];
        let mut out = [0u8; 1];
        quantize_symmetric_i2(&data, 1.0, &mut out);
        // elem0=+1→0b01, elem1=-1→0b11, elem2=0→0b00, pad→0b00
        assert_eq!(out[0], 0b00_00_11_01);
    }

    #[test]
    fn test_quant_i2_two_bytes() {
        let data = [1.0, 0.0, -1.0, 1.0, -1.0, -1.0, 0.0, 0.0];
        let mut out = [0u8; 2];
        quantize_symmetric_i2(&data, 1.0, &mut out);
        // byte0: +1=01, 0=00, -1=11, +1=01 → 0b01_11_00_01
        assert_eq!(out[0], 0b01_11_00_01);
        // byte1: -1=11, -1=11, 0=00, 0=00 → 0b00_00_11_11
        assert_eq!(out[1], 0b00_00_11_11);
    }

    #[test]
    fn test_quant_i2_with_scale() {
        // scale=0.5 → values mapped: 0.6/0.5=1.2→+1, -0.3/0.5=-0.6→-1
        let data = [0.6, -0.3, 0.0, 0.0];
        let scale = 0.5;
        let mut out = [0u8; 1];
        quantize_symmetric_i2(&data, scale, &mut out);
        // round(0.6/0.5)=1→+1=01, round(-0.3/0.5)=-1→-1=11, 0, 0
        assert_eq!(out[0], 0b00_00_11_01);
    }

    // ── dequantize_i8 ───────────────────────────────────────────────

    #[test]
    fn test_dequant_i8_basic() {
        let data = [0i8, 127, -127, 64];
        let scale = 1.0 / 127.0;
        let mut out = [0.0f32; 4];
        dequantize_i8(&data, scale, &mut out);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
        assert!((out[2] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_dequant_i8_zeros() {
        let data = [0i8; 4];
        let mut out = [0.0f32; 4];
        dequantize_i8(&data, 0.5, &mut out);
        assert_eq!(out.to_vec(), vec![0.0; 4]);
    }

    #[test]
    #[should_panic(expected = "data and output must have equal length")]
    fn test_dequant_i8_len_mismatch() {
        dequantize_i8(&[1i8], 1.0, &mut [0.0f32; 2]);
    }

    // ── dequantize_i2 (BitNet I2_S) ─────────────────────────────────

    #[test]
    fn test_dequant_i2_basic() {
        // byte: elem0=+1(01), elem1=0(00), elem2=-1(11), elem3=+1(01)
        let data = [0b01_11_00_01u8];
        let mut out = [0.0f32; 4];
        dequantize_i2(&data, 2.0, &mut out);
        assert_eq!(out.to_vec(), vec![2.0, 0.0, -2.0, 2.0]);
    }

    #[test]
    fn test_dequant_i2_all_zeros() {
        let data = [0x00u8];
        let mut out = [0.0f32; 4];
        dequantize_i2(&data, 1.0, &mut out);
        assert_eq!(out.to_vec(), vec![0.0; 4]);
    }

    #[test]
    fn test_dequant_i2_partial() {
        // only 3 output elements from 1 byte
        let data = [0b00_11_01_00u8]; // elem0=0, elem1=+1, elem2=-1
        let mut out = [0.0f32; 3];
        dequantize_i2(&data, 1.0, &mut out);
        assert_eq!(out.to_vec(), vec![0.0, 1.0, -1.0]);
    }

    #[test]
    fn test_dequant_i2_encoding_0b10() {
        // 0b10 maps to 0 (unspecified encoding treated as zero)
        let data = [0b10u8]; // elem0 = 0b10
        let mut out = [0.0f32; 1];
        dequantize_i2(&data, 1.0, &mut out);
        assert_eq!(out[0], 0.0);
    }

    // ── Round-trip i8 ───────────────────────────────────────────────

    #[test]
    fn test_roundtrip_i8() {
        let original = [0.0, 0.5, -0.5, 1.0, -1.0];
        let absmax = compute_absmax(&original);
        let scale = absmax / 127.0;
        let mut quantized = vec![0i8; original.len()];
        quantize_symmetric_i8(&original, scale, &mut quantized);
        let mut restored = vec![0.0f32; original.len()];
        dequantize_i8(&quantized, scale, &mut restored);
        for (o, r) in original.iter().zip(restored.iter()) {
            assert!((o - r).abs() < scale + 1e-6, "orig={o} restored={r}");
        }
    }

    // ── Round-trip i2 ───────────────────────────────────────────────

    #[test]
    fn test_roundtrip_i2() {
        // Values that map exactly to {-1, 0, +1}
        let original = [1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0];
        let scale = 1.0;
        let mut quantized = [0u8; 2];
        quantize_symmetric_i2(&original, scale, &mut quantized);
        let mut restored = vec![0.0f32; original.len()];
        dequantize_i2(&quantized, scale, &mut restored);
        assert_eq!(restored, original);
    }

    #[test]
    fn test_roundtrip_i2_with_scale() {
        let scale = 3.0;
        let original = [3.0, -3.0, 0.0, 3.0];
        let mut quantized = [0u8; 1];
        quantize_symmetric_i2(&original, scale, &mut quantized);
        let mut restored = [0.0f32; 4];
        dequantize_i2(&quantized, scale, &mut restored);
        assert_eq!(restored, original);
    }

    // ── calibrate_histogram ─────────────────────────────────────────

    #[test]
    fn test_histogram_basic() {
        let data = [0.0, 0.5, 1.0, 1.5, 2.0];
        let hist = calibrate_histogram(&data, 4);
        assert_eq!(hist.len(), 4);
        let total: u64 = hist.iter().sum();
        assert_eq!(total, data.len() as u64);
    }

    #[test]
    fn test_histogram_empty() {
        let hist = calibrate_histogram(&[], 10);
        assert_eq!(hist.to_vec(), vec[0u64; 10]);
    }

    #[test]
    fn test_histogram_zero_bins() {
        let hist = calibrate_histogram(&[1.0], 0);
        assert!(hist.is_empty());
    }

    #[test]
    fn test_histogram_all_same() {
        let data = [3.0, 3.0, 3.0];
        let hist = calibrate_histogram(&data, 4);
        let total: u64 = hist.iter().sum();
        assert_eq!(total, 3);
        // All values have abs=3.0 which is absmax → should go to last bin
        assert_eq!(hist[3], 3);
    }

    #[test]
    fn test_histogram_all_zeros() {
        let data = [0.0, 0.0, 0.0];
        let hist = calibrate_histogram(&data, 5);
        assert_eq!(hist[0], 3);
        assert_eq!(hist.iter().sum::<u64>(), 3);
    }

    #[test]
    fn test_histogram_negative_values() {
        // Histogram uses absolute values
        let data = [-1.0, -2.0, 1.0, 2.0];
        let hist = calibrate_histogram(&data, 2);
        assert_eq!(hist.iter().sum::<u64>(), 4);
    }

    // ── compute_kl_divergence ───────────────────────────────────────

    #[test]
    fn test_kl_identical() {
        let p = [0.25, 0.25, 0.25, 0.25];
        let q = [0.25, 0.25, 0.25, 0.25];
        assert!((compute_kl_divergence(&p, &q) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_kl_known_answer() {
        // P = [1/2, 1/2], Q = [1/4, 3/4]
        // KL = 0.5*ln(0.5/0.25) + 0.5*ln(0.5/0.75)
        //    = 0.5*ln(2) + 0.5*ln(2/3)
        let p = [0.5, 0.5];
        let q = [0.25, 0.75];
        let expected = 0.5 * (2.0_f64).ln() + 0.5 * (2.0_f64 / 3.0).ln();
        assert!((compute_kl_divergence(&p, &q) - expected).abs() < 1e-12);
    }

    #[test]
    fn test_kl_zero_p() {
        // Entries where p=0 contribute 0
        let p = [0.0, 1.0];
        let q = [0.5, 0.5];
        let expected = 1.0 * (1.0_f64 / 0.5).ln();
        assert!((compute_kl_divergence(&p, &q) - expected).abs() < 1e-12);
    }

    #[test]
    fn test_kl_zero_q_gives_infinity() {
        let p = [0.5, 0.5];
        let q = [0.0, 1.0];
        assert!(compute_kl_divergence(&p, &q).is_infinite());
    }

    #[test]
    fn test_kl_empty() {
        assert_eq!(compute_kl_divergence(&[], &[]), 0.0);
    }

    #[test]
    #[should_panic(expected = "p and q must have equal length")]
    fn test_kl_len_mismatch() {
        compute_kl_divergence(&[1.0], &[0.5, 0.5]);
    }

    // ── smooth_quant_scales ─────────────────────────────────────────

    #[test]
    fn test_smooth_quant_alpha_zero() {
        // alpha=0 → output = 1 / weight_scales
        let act = [2.0, 4.0];
        let wgt = [0.5, 2.0];
        let mut out = [0.0f32; 2];
        smooth_quant_scales(&act, &wgt, 0.0, &mut out);
        assert!((out[0] - 1.0 / 0.5).abs() < 1e-5);
        assert!((out[1] - 1.0 / 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_smooth_quant_alpha_one() {
        // alpha=1 → output = act_scales
        let act = [2.0, 4.0];
        let wgt = [0.5, 2.0];
        let mut out = [0.0f32; 2];
        smooth_quant_scales(&act, &wgt, 1.0, &mut out);
        assert!((out[0] - 2.0).abs() < 1e-5);
        assert!((out[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_smooth_quant_alpha_half() {
        // alpha=0.5 → output = sqrt(act) / sqrt(wgt)
        let act = [4.0, 9.0];
        let wgt = [1.0, 4.0];
        let mut out = [0.0f32; 2];
        smooth_quant_scales(&act, &wgt, 0.5, &mut out);
        assert!((out[0] - 2.0).abs() < 1e-5); // sqrt(4)/sqrt(1)=2
        assert!((out[1] - 1.5).abs() < 1e-5); // sqrt(9)/sqrt(4)=1.5
    }

    #[test]
    #[should_panic(expected = "act_scales and weight_scales must have equal length")]
    fn test_smooth_quant_mismatch() {
        smooth_quant_scales(&[1.0], &[1.0, 2.0], 0.5, &mut [0.0; 1]);
    }

    // ── compute_mse_loss ────────────────────────────────────────────

    #[test]
    fn test_mse_identical() {
        let a = [1.0, 2.0, 3.0];
        assert_eq!(compute_mse_loss(&a, &a), 0.0);
    }

    #[test]
    fn test_mse_known_answer() {
        let orig = [1.0, 2.0, 3.0];
        let quant = [1.0, 2.5, 2.5];
        // diffs = [0, 0.5, -0.5], sq = [0, 0.25, 0.25], mean = 0.5/3
        let expected = 0.5 / 3.0;
        assert!((compute_mse_loss(&orig, &quant) - expected as f32).abs() < 1e-6);
    }

    #[test]
    fn test_mse_empty() {
        assert_eq!(compute_mse_loss(&[], &[]), 0.0);
    }

    #[test]
    fn test_mse_single_element() {
        assert!((compute_mse_loss(&[3.0], &[1.0]) - 4.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "original and quantized must have equal length")]
    fn test_mse_len_mismatch() {
        compute_mse_loss(&[1.0], &[1.0, 2.0]);
    }

    // ── Property-style tests ────────────────────────────────────────

    #[test]
    fn test_absmax_is_nonnegative() {
        for v in [-100.0, -1.0, 0.0, 1.0, 100.0] {
            assert!(compute_absmax(&[v]) >= 0.0);
        }
    }

    #[test]
    fn test_scale_always_positive_when_range_nonzero() {
        let cases: &[&[f32]] = &[&[-1.0, 1.0], &[0.0, 100.0], &[-50.0, 50.0]];
        for data in cases {
            let (scale, _) = compute_scale_zeropoint(data, 8);
            assert!(scale > 0.0, "scale must be positive for data {:?}", data);
        }
    }

    #[test]
    fn test_zeropoint_bounded_for_various_data() {
        let cases: &[&[f32]] = &[&[-1.0, 1.0], &[0.0, 100.0], &[-50.0, 0.0]];
        for data in cases {
            let (_, zp) = compute_scale_zeropoint(data, 8);
            assert!(zp >= -256 && zp <= 256, "zp={zp} out of expected range");
        }
    }

    #[test]
    fn test_per_channel_count_matches() {
        let n = 5;
        let ch_size = 8;
        let data = vec![1.0; n * ch_size];
        let scales = compute_per_channel_scales(&data, n, ch_size);
        assert_eq!(scales.len(), n);
    }

    #[test]
    fn test_per_group_count_matches() {
        for len in [7, 8, 9, 16, 17] {
            let data = vec![1.0; len];
            let gs = 4;
            let expected = (len + gs - 1) / gs;
            let scales = compute_per_group_scales(&data, gs);
            assert_eq!(scales.len(), expected, "len={len} gs={gs}");
        }
    }

    #[test]
    fn test_kl_nonnegative() {
        // KL divergence is always >= 0
        let p = [0.3, 0.7];
        let q = [0.5, 0.5];
        assert!(compute_kl_divergence(&p, &q) >= 0.0);
    }

    #[test]
    fn test_mse_nonnegative() {
        let a = [1.0, 2.0, 3.0];
        let b = [3.0, 2.0, 1.0];
        assert!(compute_mse_loss(&a, &b) >= 0.0);
    }

    // ── Numerical edge cases ────────────────────────────────────────

    #[test]
    fn test_quant_i8_very_large_values() {
        let data = [1e10, -1e10];
        let scale = compute_absmax(&data) / 127.0;
        let mut out = [0i8; 2];
        quantize_symmetric_i8(&data, scale, &mut out);
        assert_eq!(out[0], 127);
        assert_eq!(out[1], -127);
    }

    #[test]
    fn test_quant_i8_very_small_values() {
        let data = [1e-10, -1e-10, 0.0];
        let scale = compute_absmax(&data) / 127.0;
        let mut out = [0i8; 3];
        quantize_symmetric_i8(&data, scale, &mut out);
        assert_eq!(out[0], 127);
        assert_eq!(out[1], -127);
        assert_eq!(out[2], 0);
    }

    #[test]
    fn test_mse_large_values() {
        let orig = [1e6, -1e6];
        let quant = [1e6 + 1.0, -1e6 - 1.0];
        let mse = compute_mse_loss(&orig, &quant);
        assert!((mse - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_histogram_single_value() {
        let data = [5.0];
        let hist = calibrate_histogram(&data, 3);
        assert_eq!(hist.iter().sum::<u64>(), 1);
    }

    #[test]
    fn test_smooth_quant_ones() {
        let ones = [1.0, 1.0, 1.0];
        let mut out = [0.0f32; 3];
        smooth_quant_scales(&ones, &ones, 0.5, &mut out);
        for &o in &out {
            assert!((o - 1.0).abs() < 1e-6);
        }
    }
}
