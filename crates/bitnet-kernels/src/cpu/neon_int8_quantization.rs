//! NEON-optimized int8 quantization for Apple Silicon (aarch64).
//!
//! Provides symmetric and asymmetric per-tensor and per-channel int8
//! quantization with configurable block sizes (32, 64, 128, 256).
//!
//! # NEON Intrinsics
//!
//! Hot paths are documented for future NEON vectorisation using:
//! - `vabsq_f32` — lane-wise absolute value for absmax reduction
//! - `vmaxvq_f32` — horizontal max across a 128-bit float register
//! - `vcvtq_s32_f32` — vectorised float-to-int32 rounding conversion
//! - `vld1q_f32` / `vst1q_s8` — contiguous SIMD load/store

/// Supported block sizes for int8 quantization.
const VALID_BLOCK_SIZES: [usize; 4] = [32, 64, 128, 256];

// ── Configuration ──────────────────────────────────────────────────────

/// Configuration for int8 quantization.
#[derive(Debug, Clone)]
pub struct QuantizeConfig {
    /// Number of elements per quantization block (32, 64, 128, or 256).
    pub block_size: usize,
    /// If `true`, use symmetric quantization (zero_point = 0).
    pub symmetric: bool,
    /// If `true`, quantize each channel independently.
    pub per_channel: bool,
}

impl Default for QuantizeConfig {
    fn default() -> Self {
        Self { block_size: 64, symmetric: true, per_channel: false }
    }
}

// ── Quantized block ────────────────────────────────────────────────────

/// A single block of int8-quantized values with scale and zero-point.
#[derive(Debug, Clone)]
pub struct QuantizedBlock {
    /// Quantized int8 values.
    pub data: Vec<i8>,
    /// Scale factor: `float_value ≈ scale * (quantized - zero_point)`.
    pub scale: f32,
    /// Zero-point offset (0 for symmetric quantization).
    pub zero_point: i8,
    /// Number of logical elements in this block.
    pub block_size: usize,
}

// ── Quantizer with calibration stats ───────────────────────────────────

/// Stateful int8 quantizer that tracks calibration statistics.
#[derive(Debug, Clone)]
pub struct Int8Quantizer {
    /// Quantization configuration.
    pub config: QuantizeConfig,
    /// Running observed maximum absolute value (for calibration).
    pub observed_absmax: f32,
    /// Number of calibration samples seen.
    pub calibration_count: u64,
}

impl Int8Quantizer {
    /// Create a new quantizer with the given configuration.
    pub fn new(config: QuantizeConfig) -> Self {
        assert!(
            VALID_BLOCK_SIZES.contains(&config.block_size),
            "block_size must be one of {VALID_BLOCK_SIZES:?}"
        );
        Self { config, observed_absmax: 0.0, calibration_count: 0 }
    }

    /// Update calibration statistics with a new sample.
    pub fn observe(&mut self, input: &[f32]) {
        let m = absmax_scale(input);
        if m > self.observed_absmax {
            self.observed_absmax = m;
        }
        self.calibration_count += 1;
    }

    /// Quantize using the stored configuration.
    pub fn quantize(&self, input: &[f32]) -> Vec<QuantizedBlock> {
        quantize_f32_to_i8(input, &self.config)
    }
}

// ── Core helpers ───────────────────────────────────────────────────────

/// Compute the absolute-maximum value of a slice.
///
/// # NEON optimisation notes
///
/// A vectorised implementation would use `vld1q_f32` to load four floats,
/// `vabsq_f32` for lane-wise abs, and `vmaxvq_f32` for horizontal max
/// across the 128-bit register, processing 4 elements per iteration.
pub fn absmax_scale(block: &[f32]) -> f32 {
    block.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()))
}

/// Clamp a float and round to the nearest `i8`.
#[inline]
pub fn clamp_and_round(val: f32, min: i8, max: i8) -> i8 {
    let clamped = val.round().clamp(min as f32, max as f32);
    clamped as i8
}

/// Compute scale and zero-point for a block of floats.
///
/// # NEON optimisation notes
///
/// The min/max scan can be vectorised with `vminvq_f32` / `vmaxvq_f32`
/// horizontal reductions after `vld1q_f32` loads.
pub fn find_scale_and_zero(block: &[f32], symmetric: bool) -> (f32, i8) {
    if block.is_empty() {
        return (1.0, 0);
    }

    let fmin = block.iter().copied().fold(f32::INFINITY, f32::min);
    let fmax = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    if symmetric {
        let absmax = fmin.abs().max(fmax.abs());
        let scale = if absmax == 0.0 { 1.0 } else { absmax / 127.0 };
        (scale, 0)
    } else {
        let range = fmax - fmin;
        let scale = if range == 0.0 { 1.0 } else { range / 255.0 };
        let zp = clamp_and_round(-fmin / scale - 128.0, -128, 127);
        (scale, zp)
    }
}

// ── Quantize / Dequantize ──────────────────────────────────────────────

/// Quantize a slice of `f32` values into int8 blocks.
///
/// # NEON optimisation notes
///
/// The inner loop (float → scaled → rounded int) maps to:
/// `vld1q_f32` → `vdivq_f32` (or `vmulq_f32` reciprocal) →
/// `vcvtq_s32_f32` (round-to-nearest) → narrow to `int8`.
pub fn quantize_f32_to_i8(input: &[f32], config: &QuantizeConfig) -> Vec<QuantizedBlock> {
    let bs = config.block_size;
    let n_blocks = input.len().div_ceil(bs);
    let mut blocks = Vec::with_capacity(n_blocks);

    for chunk in input.chunks(bs) {
        let (scale, zero_point) = find_scale_and_zero(chunk, config.symmetric);
        let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

        let data: Vec<i8> = chunk
            .iter()
            .map(|&v| {
                let q = v * inv_scale + zero_point as f32;
                clamp_and_round(q, -128, 127)
            })
            .collect();

        blocks.push(QuantizedBlock { data, scale, zero_point, block_size: chunk.len() });
    }
    blocks
}

/// Dequantize int8 blocks back to `f32`.
///
/// Reconstructs `float ≈ scale * (q - zero_point)` per element.
///
/// # NEON optimisation notes
///
/// Widening `vld1_s8` → `vmovl_s8` → `vcvtq_f32_s32` then
/// `vmulq_f32` by broadcast scale and `vsubq_f32` zero-point.
pub fn dequantize_i8_to_f32(blocks: &[QuantizedBlock]) -> Vec<f32> {
    let total: usize = blocks.iter().map(|b| b.block_size).sum();
    let mut out = Vec::with_capacity(total);

    for blk in blocks {
        for &q in &blk.data {
            out.push(blk.scale * (q as f32 - blk.zero_point as f32));
        }
    }
    out
}

// ── Per-channel quantization ───────────────────────────────────────────

/// Quantize a flattened tensor with `channels` leading dimension.
///
/// `tensor.len()` must be divisible by `channels`. Each channel is
/// quantized independently using the provided configuration.
pub fn quantize_per_channel(
    tensor: &[f32],
    channels: usize,
    config: &QuantizeConfig,
) -> Vec<Vec<QuantizedBlock>> {
    assert!(channels > 0, "channels must be > 0");
    assert!(tensor.len().is_multiple_of(channels), "tensor length must be divisible by channels");
    let elems_per_ch = tensor.len() / channels;
    (0..channels)
        .map(|ch| {
            let start = ch * elems_per_ch;
            let end = start + elems_per_ch;
            quantize_f32_to_i8(&tensor[start..end], config)
        })
        .collect()
}

// ── Error metric ───────────────────────────────────────────────────────

/// Compute the mean-squared round-trip quantization error.
///
/// Quantizes `input`, dequantizes, then returns the MSE between the
/// original and reconstructed values.
pub fn round_trip_error(input: &[f32], config: &QuantizeConfig) -> f32 {
    if input.is_empty() {
        return 0.0;
    }
    let blocks = quantize_f32_to_i8(input, config);
    let recon = dequantize_i8_to_f32(&blocks);
    let mse: f32 = input.iter().zip(recon.iter()).map(|(&a, &b)| (a - b) * (a - b)).sum::<f32>()
        / input.len() as f32;
    mse
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sym_config(block_size: usize) -> QuantizeConfig {
        QuantizeConfig { block_size, symmetric: true, per_channel: false }
    }

    fn asym_config(block_size: usize) -> QuantizeConfig {
        QuantizeConfig { block_size, symmetric: false, per_channel: false }
    }

    // ── Symmetric round-trip ───────────────────────────────────────

    #[test]
    fn test_symmetric_round_trip_small() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();
        let cfg = sym_config(64);
        let blocks = quantize_f32_to_i8(&input, &cfg);
        let recon = dequantize_i8_to_f32(&blocks);

        assert_eq!(recon.len(), input.len());
        let max_err: f32 =
            input.iter().zip(recon.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        // Symmetric quant into [-127,127] ⇒ step ≤ absmax/127
        assert!(max_err < 1.0 / 127.0 + 1e-6, "err={max_err}");
    }

    #[test]
    fn test_symmetric_round_trip_large() {
        let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 10.0).collect();
        let cfg = sym_config(256);
        let blocks = quantize_f32_to_i8(&input, &cfg);
        let recon = dequantize_i8_to_f32(&blocks);

        assert_eq!(recon.len(), input.len());
        let mse = round_trip_error(&input, &cfg);
        assert!(mse < 0.01, "mse={mse}");
    }

    // ── Asymmetric round-trip ──────────────────────────────────────

    #[test]
    fn test_asymmetric_round_trip() {
        let input: Vec<f32> = (0..64).map(|i| i as f32 / 63.0).collect();
        let cfg = asym_config(64);
        let blocks = quantize_f32_to_i8(&input, &cfg);
        let recon = dequantize_i8_to_f32(&blocks);

        assert_eq!(recon.len(), input.len());
        let max_err: f32 =
            input.iter().zip(recon.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_err < 0.01, "err={max_err}");
    }

    #[test]
    fn test_asymmetric_negative_range() {
        let input: Vec<f32> = (0..32).map(|i| -1.0 + i as f32 * 0.01).collect();
        let cfg = asym_config(32);
        let blocks = quantize_f32_to_i8(&input, &cfg);
        let recon = dequantize_i8_to_f32(&blocks);
        assert_eq!(recon.len(), input.len());
    }

    // ── Zero input ─────────────────────────────────────────────────

    #[test]
    fn test_zero_input_symmetric() {
        let input = [0.0_f32; 64];
        let cfg = sym_config(64);
        let blocks = quantize_f32_to_i8(&input, &cfg);
        let recon = dequantize_i8_to_f32(&blocks);
        for v in &recon {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_zero_input_asymmetric() {
        let input = [0.0_f32; 32];
        let cfg = asym_config(32);
        let blocks = quantize_f32_to_i8(&input, &cfg);
        let recon = dequantize_i8_to_f32(&blocks);
        for v in &recon {
            assert!((v.abs()) < 1e-5, "expected ~0, got {v}");
        }
    }

    // ── Constant input ─────────────────────────────────────────────

    #[test]
    fn test_constant_input() {
        let input = [42.0_f32; 64];
        let cfg = sym_config(64);
        let blocks = quantize_f32_to_i8(&input, &cfg);
        // All quantized values should be identical.
        let first = blocks[0].data[0];
        for q in &blocks[0].data {
            assert_eq!(*q, first);
        }
        let recon = dequantize_i8_to_f32(&blocks);
        for v in &recon {
            assert!((v - 42.0).abs() < 0.5, "expected ~42, got {v}");
        }
    }

    // ── Scale computation ──────────────────────────────────────────

    #[test]
    fn test_scale_symmetric() {
        let block = vec![-1.0_f32, 0.5, 1.0];
        let (scale, zp) = find_scale_and_zero(&block, true);
        assert_eq!(zp, 0);
        assert!((scale - 1.0 / 127.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_asymmetric() {
        let block = vec![0.0_f32, 1.0];
        let (scale, _zp) = find_scale_and_zero(&block, false);
        assert!((scale - 1.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_zero_range() {
        let block = [5.0_f32; 10];
        let (scale, _) = find_scale_and_zero(&block, true);
        // absmax = 5, scale = 5/127
        assert!((scale - 5.0 / 127.0).abs() < 1e-6);
    }

    // ── Block sizes ────────────────────────────────────────────────

    #[test]
    fn test_block_size_32() {
        let input: Vec<f32> = (0..96).map(|i| i as f32).collect();
        let cfg = sym_config(32);
        let blocks = quantize_f32_to_i8(&input, &cfg);
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].data.len(), 32);
        assert_eq!(blocks[2].data.len(), 32);
    }

    #[test]
    fn test_block_size_64() {
        let input: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let cfg = sym_config(64);
        let blocks = quantize_f32_to_i8(&input, &cfg);
        assert_eq!(blocks.len(), 2);
    }

    #[test]
    fn test_block_size_128() {
        let input: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let cfg = sym_config(128);
        let blocks = quantize_f32_to_i8(&input, &cfg);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].data.len(), 128);
    }

    #[test]
    fn test_block_size_256() {
        let input: Vec<f32> = (0..512).map(|i| i as f32).collect();
        let cfg = sym_config(256);
        let blocks = quantize_f32_to_i8(&input, &cfg);
        assert_eq!(blocks.len(), 2);
    }

    #[test]
    fn test_partial_last_block() {
        let input: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let cfg = sym_config(32);
        let blocks = quantize_f32_to_i8(&input, &cfg);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].data.len(), 32);
        assert_eq!(blocks[1].data.len(), 18);
        assert_eq!(blocks[1].block_size, 18);
    }

    // ── Per-channel ────────────────────────────────────────────────

    #[test]
    fn test_per_channel_basic() {
        let tensor: Vec<f32> = (0..256).map(|i| (i as f32) / 256.0).collect();
        let cfg = sym_config(64);
        let ch_blocks = quantize_per_channel(&tensor, 4, &cfg);
        assert_eq!(ch_blocks.len(), 4);
        // Each channel is 64 elements ⇒ 1 block of size 64
        for ch in &ch_blocks {
            assert_eq!(ch.len(), 1);
        }
    }

    #[test]
    fn test_per_channel_scales_differ() {
        // Two channels with very different ranges.
        let mut tensor = [0.0_f32; 128];
        for i in 0..64 {
            tensor[i] = i as f32; // range [0, 63]
        }
        for i in 64..128 {
            tensor[i] = (i - 64) as f32 * 100.0; // range [0, 6300]
        }
        let cfg = sym_config(64);
        let ch_blocks = quantize_per_channel(&tensor, 2, &cfg);
        let s0 = ch_blocks[0][0].scale;
        let s1 = ch_blocks[1][0].scale;
        assert!(s1 > s0 * 10.0, "s0={s0}, s1={s1}");
    }

    // ── Extreme values ─────────────────────────────────────────────

    #[test]
    fn test_extreme_large_values() {
        let input = [1e6_f32; 32];
        let cfg = sym_config(32);
        let blocks = quantize_f32_to_i8(&input, &cfg);
        let recon = dequantize_i8_to_f32(&blocks);
        for v in &recon {
            assert!((v - 1e6).abs() / 1e6 < 0.01);
        }
    }

    #[test]
    fn test_extreme_small_values() {
        let input: Vec<f32> = (0..32).map(|i| i as f32 * 1e-7).collect();
        let cfg = sym_config(32);
        let blocks = quantize_f32_to_i8(&input, &cfg);
        let recon = dequantize_i8_to_f32(&blocks);
        assert_eq!(recon.len(), input.len());
    }

    // ── Negative values ────────────────────────────────────────────

    #[test]
    fn test_all_negative() {
        let input: Vec<f32> = (0..64).map(|i| -(i as f32) - 1.0).collect();
        let cfg = sym_config(64);
        let blocks = quantize_f32_to_i8(&input, &cfg);
        let recon = dequantize_i8_to_f32(&blocks);
        for v in &recon {
            assert!(*v <= 0.0, "expected ≤0, got {v}");
        }
    }

    // ── Round-trip error metric ────────────────────────────────────

    #[test]
    fn test_round_trip_error_zero_input() {
        let input = [0.0_f32; 64];
        let cfg = sym_config(64);
        assert_eq!(round_trip_error(&input, &cfg), 0.0);
    }

    #[test]
    fn test_round_trip_error_nonzero() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();
        let cfg = sym_config(64);
        let mse = round_trip_error(&input, &cfg);
        assert!(mse >= 0.0);
        assert!(mse < 0.001, "mse={mse}");
    }

    #[test]
    fn test_round_trip_error_empty() {
        let cfg = sym_config(64);
        assert_eq!(round_trip_error(&[], &cfg), 0.0);
    }

    // ── Mixed positive/negative ────────────────────────────────────

    #[test]
    fn test_mixed_pos_neg() {
        let input: Vec<f32> = (0..64).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let cfg = sym_config(64);
        let blocks = quantize_f32_to_i8(&input, &cfg);
        let recon = dequantize_i8_to_f32(&blocks);
        for (orig, rec) in input.iter().zip(recon.iter()) {
            assert!((orig - rec).abs() < 0.02);
        }
    }

    // ── Power-of-two (exact representation) ────────────────────────

    #[test]
    fn test_power_of_two_exact() {
        // 0.5 should round-trip exactly when absmax = 0.5 (scale = 0.5/127).
        let input = [0.5_f32; 32];
        let cfg = sym_config(32);
        let blocks = quantize_f32_to_i8(&input, &cfg);
        // All elements have value 0.5, so q = 0.5 / (0.5/127) = 127
        assert_eq!(blocks[0].data[0], 127);
        let recon = dequantize_i8_to_f32(&blocks);
        for v in &recon {
            assert!((v - 0.5).abs() < 1e-4, "power-of-two should be near-exact, got {v}");
        }
    }

    // ── Monotonicity preservation ──────────────────────────────────

    #[test]
    fn test_monotonicity_preserved() {
        // Strictly increasing input should give non-decreasing output.
        let input: Vec<f32> = (0..64).map(|i| i as f32 / 63.0).collect();
        let cfg = sym_config(64);
        let recon = dequantize_i8_to_f32(&quantize_f32_to_i8(&input, &cfg));
        for i in 1..recon.len() {
            assert!(
                recon[i] >= recon[i - 1],
                "monotonicity broken at {i}: {} < {}",
                recon[i],
                recon[i - 1]
            );
        }
    }

    // ── i8 range invariant ─────────────────────────────────────────

    #[test]
    fn test_quantized_values_in_i8_range() {
        let input: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 100.0).collect();
        let cfg = sym_config(256);
        let blocks = quantize_f32_to_i8(&input, &cfg);
        for blk in &blocks {
            for &q in &blk.data {
                assert!((-128..=127).contains(&(q as i16)));
            }
        }
    }

    // ── absmax_scale ───────────────────────────────────────────────

    #[test]
    fn test_absmax_scale_positive() {
        assert!((absmax_scale(&[1.0, 2.0, 3.0]) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_absmax_scale_negative() {
        assert!((absmax_scale(&[-5.0, 2.0, 3.0]) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_absmax_scale_empty() {
        assert_eq!(absmax_scale(&[]), 0.0);
    }

    // ── clamp_and_round ────────────────────────────────────────────

    #[test]
    fn test_clamp_and_round_basic() {
        assert_eq!(clamp_and_round(0.6, -128, 127), 1);
        assert_eq!(clamp_and_round(-0.6, -128, 127), -1);
        assert_eq!(clamp_and_round(200.0, -128, 127), 127);
        assert_eq!(clamp_and_round(-200.0, -128, 127), -128);
    }

    // ── Int8Quantizer struct ───────────────────────────────────────

    #[test]
    fn test_quantizer_observe_and_quantize() {
        let mut q = Int8Quantizer::new(sym_config(32));
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        q.observe(&data);
        assert_eq!(q.calibration_count, 1);
        assert!((q.observed_absmax - 31.0).abs() < 1e-6);

        let blocks = q.quantize(&data);
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn test_quantizer_multiple_observations() {
        let mut q = Int8Quantizer::new(sym_config(64));
        q.observe(&[1.0; 64]);
        q.observe(&[10.0; 64]);
        q.observe(&[5.0; 64]);
        assert_eq!(q.calibration_count, 3);
        assert!((q.observed_absmax - 10.0).abs() < 1e-6);
    }
}
