//! CUDA quantization/dequantization kernels with CPU fallback.
//!
//! Provides ternary ({-1, 0, 1}) and I2_S 2-bit quantization with
//! configurable per-block scale computation strategies. Multiple
//! calibration methods are supported: absolute-maximum, min-max,
//! symmetric, and percentile-based.
//!
//! # Kernel strategy
//!
//! Quantization partitions the input into fixed-size blocks, computes
//! a per-block scale factor using the chosen [`QuantMethod`], and maps
//! each element to a low-bit representation.  I2_S packs four 2-bit
//! ternary values per byte (LSB-first), matching the encoding used in
//! [`super::quantized_matmul`].
//!
//! # CPU fallback
//!
//! All public functions have pure-Rust implementations that work on
//! any platform.  GPU-specific CUDA kernel source strings are gated
//! behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.

use bitnet_common::{KernelError, Result};

// ── Configuration ─────────────────────────────────────────────────────

/// Scale calibration method for quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMethod {
    /// Scale = max(|x|) in the block.
    AbsMax,
    /// Scale = (max(x) - min(x)) / 2, centred on the block midpoint.
    MinMax,
    /// Symmetric around zero: scale = max(|x|), identical to AbsMax
    /// but explicitly documents intent.
    Symmetric,
    /// Scale from the p-th percentile of |x| (clamped to [0, 100]).
    Percentile(u8),
}

/// Configuration for a quantization pass.
#[derive(Debug, Clone)]
pub struct QuantizeConfig {
    /// Number of elements per quantization block.
    pub block_size: usize,
    /// Calibration method for per-block scale computation.
    pub method: QuantMethod,
}

impl Default for QuantizeConfig {
    fn default() -> Self {
        Self { block_size: 32, method: QuantMethod::AbsMax }
    }
}

// ── Scale calibration ─────────────────────────────────────────────────

/// Compute per-block scale factors for the given input.
///
/// Returns one scale per block of `config.block_size` elements.  The
/// last block may be shorter if `input.len()` is not a multiple of the
/// block size.
///
/// # Errors
///
/// Returns an error if `config.block_size` is zero.
pub fn calibrate_scales(input: &[f32], config: &QuantizeConfig) -> Result<Vec<f32>> {
    if config.block_size == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "block_size must be > 0".into(),
        }
        .into());
    }
    let num_blocks = input.len().div_ceil(config.block_size);
    let mut scales = Vec::with_capacity(num_blocks);

    for blk in 0..num_blocks {
        let start = blk * config.block_size;
        let end = (start + config.block_size).min(input.len());
        let block = &input[start..end];
        let scale = block_scale(block, config.method);
        scales.push(scale);
    }
    Ok(scales)
}

/// Compute a single block's scale factor.
fn block_scale(block: &[f32], method: QuantMethod) -> f32 {
    if block.is_empty() {
        return 0.0;
    }
    match method {
        QuantMethod::AbsMax | QuantMethod::Symmetric => {
            block.iter().fold(0.0_f32, |m, &v| m.max(v.abs()))
        }
        QuantMethod::MinMax => {
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for &v in block {
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }
            (max - min) / 2.0
        }
        QuantMethod::Percentile(p) => {
            let p = p.min(100) as f32 / 100.0;
            let mut abs_vals: Vec<f32> = block.iter().map(|v| v.abs()).collect();
            abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = ((abs_vals.len() as f32 - 1.0) * p).round() as usize;
            let idx = idx.min(abs_vals.len() - 1);
            abs_vals[idx]
        }
    }
}

// ── Ternary quantization ──────────────────────────────────────────────

/// Quantize f32 values to ternary {-1, 0, +1} with a per-block scale.
///
/// The returned scale is the global maximum of all per-block scales.
/// Within each block, values are thresholded at `0.5 * block_scale`:
/// above → +1, below → −1, otherwise → 0.
///
/// # Errors
///
/// Returns an error if `config.block_size` is zero.
pub fn quantize_ternary_cpu(input: &[f32], config: &QuantizeConfig) -> Result<(Vec<i8>, f32)> {
    if config.block_size == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "block_size must be > 0".into(),
        }
        .into());
    }
    if input.is_empty() {
        return Ok((Vec::new(), 0.0));
    }

    let scales = calibrate_scales(input, config)?;
    let global_scale = scales.iter().copied().fold(0.0_f32, f32::max);

    let mut quantized = Vec::with_capacity(input.len());
    for (blk_idx, blk_scale) in scales.iter().enumerate() {
        let start = blk_idx * config.block_size;
        let end = (start + config.block_size).min(input.len());
        let threshold = blk_scale * 0.5;

        for &v in &input[start..end] {
            if v > threshold {
                quantized.push(1_i8);
            } else if v < -threshold {
                quantized.push(-1_i8);
            } else {
                quantized.push(0_i8);
            }
        }
    }
    Ok((quantized, global_scale))
}

/// Dequantize ternary values back to f32.
///
/// Each element is simply `quantized[i] as f32 * scale`.
pub fn dequantize_ternary_cpu(quantized: &[i8], scale: f32) -> Vec<f32> {
    quantized.iter().map(|&v| v as f32 * scale).collect()
}

// ── I2_S packing / unpacking ──────────────────────────────────────────

/// Encode a 2-bit ternary code: +1 → 0b01, −1 → 0b11, 0 → 0b00.
#[inline(always)]
fn encode_i2s(v: i8) -> u8 {
    match v {
        1 => 0b01,
        -1 => 0b11,
        _ => 0b00,
    }
}

/// Decode a 2-bit I2_S code to its signed integer value.
#[inline(always)]
fn decode_i2s(bits: u8) -> i8 {
    match bits & 0x03 {
        0b01 => 1,
        0b11 => -1,
        _ => 0,
    }
}

/// Quantize f32 input into I2_S packed bytes with per-block scales.
///
/// Four ternary values are packed into each byte (LSB-first).  Returns
/// `(packed_bytes, per_block_scales)`.
///
/// # Errors
///
/// Returns an error if `block_size` is zero.
pub fn quantize_i2s_cpu(input: &[f32], block_size: usize) -> Result<(Vec<u8>, Vec<f32>)> {
    if block_size == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "block_size must be > 0".into(),
        }
        .into());
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

        // AbsMax scale for this block
        let abs_max = block.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
        scales.push(abs_max);

        let threshold = abs_max * 0.5;
        for (i, &v) in block.iter().enumerate() {
            let global_idx = start + i;
            let ternary = if abs_max == 0.0 {
                0_i8
            } else if v > threshold {
                1_i8
            } else if v < -threshold {
                -1_i8
            } else {
                0_i8
            };
            let byte_idx = global_idx / 4;
            let bit_off = (global_idx % 4) * 2;
            packed[byte_idx] |= encode_i2s(ternary) << bit_off;
        }
    }
    Ok((packed, scales))
}

/// Dequantize I2_S packed bytes back to f32 values.
///
/// # Errors
///
/// Returns an error if `block_size` is zero or `output_len` would read
/// past the packed buffer.
pub fn dequantize_i2s_cpu(
    packed: &[u8],
    scales: &[f32],
    block_size: usize,
    output_len: usize,
) -> Result<Vec<f32>> {
    if block_size == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "block_size must be > 0".into(),
        }
        .into());
    }
    let required_bytes = output_len.div_ceil(4);
    if packed.len() < required_bytes {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "packed buffer too small: need {} bytes for {} elements, got {}",
                required_bytes, output_len, packed.len()
            ),
        }
        .into());
    }
    let num_blocks = output_len.div_ceil(block_size);
    if scales.len() < num_blocks {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "scales too short: need {} blocks, got {}",
                num_blocks,
                scales.len()
            ),
        }
        .into());
    }

    let mut output = Vec::with_capacity(output_len);
    for i in 0..output_len {
        let byte_idx = i / 4;
        let bit_off = (i % 4) * 2;
        let bits = (packed[byte_idx] >> bit_off) & 0x03;
        let val = decode_i2s(bits);
        let blk = i / block_size;
        output.push(val as f32 * scales[blk]);
    }
    Ok(output)
}

// ── CUDA kernel source ────────────────────────────────────────────────

/// CUDA C source for ternary quantization kernel.
///
/// Kernel `quantize_ternary_f32` maps each float to {-1, 0, +1} using
/// a per-block threshold (0.5 × block_scale).  Grid-stride loop over
/// `n` elements.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const QUANTIZE_TERNARY_KERNEL_SRC: &str = r#"
extern "C" __global__ void quantize_ternary_f32(
    const float* __restrict__ input,
    signed char* __restrict__ output,
    const float* __restrict__ block_scales,
    int n,
    int block_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        int blk = i / block_size;
        float threshold = block_scales[blk] * 0.5f;
        float v = input[i];
        signed char q;
        if (v > threshold) q = 1;
        else if (v < -threshold) q = -1;
        else q = 0;
        output[i] = q;
    }
}
"#;

/// CUDA C source for ternary dequantization kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const DEQUANTIZE_TERNARY_KERNEL_SRC: &str = r#"
extern "C" __global__ void dequantize_ternary_f32(
    const signed char* __restrict__ input,
    float* __restrict__ output,
    float scale,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        output[i] = (float)input[i] * scale;
    }
}
"#;

/// CUDA C source for I2_S pack kernel (4 ternary values → 1 byte).
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const QUANTIZE_I2S_KERNEL_SRC: &str = r#"
__device__ unsigned char encode_i2s(signed char v) {
    if (v == 1)  return 0x01;
    if (v == -1) return 0x03;
    return 0x00;
}

extern "C" __global__ void quantize_i2s_f32(
    const float* __restrict__ input,
    unsigned char* __restrict__ packed,
    const float* __restrict__ block_scales,
    int n,
    int block_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        int blk = i / block_size;
        float abs_max = block_scales[blk];
        float threshold = abs_max * 0.5f;
        float v = input[i];
        signed char q;
        if (abs_max == 0.0f) q = 0;
        else if (v > threshold) q = 1;
        else if (v < -threshold) q = -1;
        else q = 0;
        unsigned char code = encode_i2s(q);
        int byte_idx = i / 4;
        int bit_off  = (i % 4) * 2;
        atomicOr(&packed[byte_idx], code << bit_off);
    }
}
"#;

/// CUDA C source for I2_S unpack/dequantize kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const DEQUANTIZE_I2S_KERNEL_SRC: &str = r#"
__device__ signed char decode_i2s(unsigned char bits) {
    bits &= 0x03;
    if (bits == 0x01) return 1;
    if (bits == 0x03) return -1;
    return 0;
}

extern "C" __global__ void dequantize_i2s_f32(
    const unsigned char* __restrict__ packed,
    const float* __restrict__ block_scales,
    float* __restrict__ output,
    int n,
    int block_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        int byte_idx = i / 4;
        int bit_off  = (i % 4) * 2;
        unsigned char bits = (packed[byte_idx] >> bit_off) & 0x03;
        signed char val = decode_i2s(bits);
        int blk = i / block_size;
        output[i] = (float)val * block_scales[blk];
    }
}
"#;

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_all_ternary(vals: &[i8]) {
        for (i, &v) in vals.iter().enumerate() {
            assert!(
                v == -1 || v == 0 || v == 1,
                "non-ternary value {v} at index {i}"
            );
        }
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "mismatch at {i}: {x} vs {y} (tol {tol})"
            );
        }
    }

    // ── QuantizeConfig defaults ───────────────────────────────────

    #[test]
    fn config_default_block_size_32() {
        let cfg = QuantizeConfig::default();
        assert_eq!(cfg.block_size, 32);
        assert_eq!(cfg.method, QuantMethod::AbsMax);
    }

    // ── calibrate_scales ──────────────────────────────────────────

    #[test]
    fn calibrate_absmax_single_block() {
        let input = vec![-3.0, 1.0, 2.0, -0.5];
        let cfg = QuantizeConfig { block_size: 8, method: QuantMethod::AbsMax };
        let scales = calibrate_scales(&input, &cfg).unwrap();
        assert_eq!(scales.len(), 1);
        assert!((scales[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn calibrate_absmax_multiple_blocks() {
        let input = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
        let cfg = QuantizeConfig { block_size: 4, method: QuantMethod::AbsMax };
        let scales = calibrate_scales(&input, &cfg).unwrap();
        assert_eq!(scales.len(), 2);
        assert!((scales[0] - 4.0).abs() < 1e-6);
        assert!((scales[1] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn calibrate_minmax_method() {
        let input = vec![-2.0, 6.0];
        let cfg = QuantizeConfig { block_size: 4, method: QuantMethod::MinMax };
        let scales = calibrate_scales(&input, &cfg).unwrap();
        // (6 - (-2)) / 2 = 4.0
        assert!((scales[0] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn calibrate_symmetric_equals_absmax() {
        let input = vec![-5.0, 3.0, 1.0];
        let cfg_abs = QuantizeConfig { block_size: 8, method: QuantMethod::AbsMax };
        let cfg_sym = QuantizeConfig { block_size: 8, method: QuantMethod::Symmetric };
        let s1 = calibrate_scales(&input, &cfg_abs).unwrap();
        let s2 = calibrate_scales(&input, &cfg_sym).unwrap();
        assert_close(&s1, &s2, 1e-7);
    }

    #[test]
    fn calibrate_percentile_100_equals_absmax() {
        let input = vec![-5.0, 3.0, 1.0, -1.0];
        let cfg_abs = QuantizeConfig { block_size: 8, method: QuantMethod::AbsMax };
        let cfg_pct = QuantizeConfig { block_size: 8, method: QuantMethod::Percentile(100) };
        let s1 = calibrate_scales(&input, &cfg_abs).unwrap();
        let s2 = calibrate_scales(&input, &cfg_pct).unwrap();
        assert_close(&s1, &s2, 1e-6);
    }

    #[test]
    fn calibrate_percentile_0_returns_min_abs() {
        let input = vec![-5.0, 3.0, 1.0, -2.0];
        let cfg = QuantizeConfig { block_size: 8, method: QuantMethod::Percentile(0) };
        let scales = calibrate_scales(&input, &cfg).unwrap();
        assert!((scales[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn calibrate_rejects_zero_block_size() {
        let cfg = QuantizeConfig { block_size: 0, method: QuantMethod::AbsMax };
        assert!(calibrate_scales(&[1.0], &cfg).is_err());
    }

    #[test]
    fn calibrate_empty_input() {
        let cfg = QuantizeConfig::default();
        let scales = calibrate_scales(&[], &cfg).unwrap();
        assert!(scales.is_empty());
    }

    // ── quantize_ternary_cpu / dequantize_ternary_cpu ─────────────

    #[test]
    fn ternary_roundtrip_basic() {
        let input = vec![1.0, -1.0, 0.1, -0.1, 0.0];
        let cfg = QuantizeConfig { block_size: 8, method: QuantMethod::AbsMax };
        let (q, scale) = quantize_ternary_cpu(&input, &cfg).unwrap();
        assert_all_ternary(&q);
        assert!(scale > 0.0);
        let deq = dequantize_ternary_cpu(&q, scale);
        assert_eq!(deq.len(), input.len());
    }

    #[test]
    fn ternary_values_correctness() {
        let input = vec![10.0, -10.0, 10.0, -10.0];
        let cfg = QuantizeConfig { block_size: 8, method: QuantMethod::AbsMax };
        let (q, _) = quantize_ternary_cpu(&input, &cfg).unwrap();
        assert_eq!(q, vec![1, -1, 1, -1]);
    }

    #[test]
    fn ternary_near_zero_becomes_zero() {
        let input = vec![0.01, -0.01, 0.02, -0.02, 100.0];
        let cfg = QuantizeConfig { block_size: 8, method: QuantMethod::AbsMax };
        let (q, _) = quantize_ternary_cpu(&input, &cfg).unwrap();
        assert_eq!(q[0], 0);
        assert_eq!(q[1], 0);
        assert_eq!(q[2], 0);
        assert_eq!(q[3], 0);
        assert_eq!(q[4], 1);
    }

    #[test]
    fn ternary_empty_input() {
        let cfg = QuantizeConfig::default();
        let (q, scale) = quantize_ternary_cpu(&[], &cfg).unwrap();
        assert!(q.is_empty());
        assert_eq!(scale, 0.0);
    }

    #[test]
    fn ternary_all_zeros() {
        let input = vec![0.0; 16];
        let cfg = QuantizeConfig { block_size: 8, method: QuantMethod::AbsMax };
        let (q, scale) = quantize_ternary_cpu(&input, &cfg).unwrap();
        assert!(q.iter().all(|&v| v == 0));
        assert_eq!(scale, 0.0);
    }

    #[test]
    fn ternary_rejects_zero_block_size() {
        let cfg = QuantizeConfig { block_size: 0, method: QuantMethod::AbsMax };
        assert!(quantize_ternary_cpu(&[1.0], &cfg).is_err());
    }

    #[test]
    fn ternary_dequant_scale_applied() {
        let q = vec![1_i8, -1, 0, 1];
        let deq = dequantize_ternary_cpu(&q, 2.5);
        assert_close(&deq, &[2.5, -2.5, 0.0, 2.5], 1e-7);
    }

    #[test]
    fn ternary_multiple_blocks() {
        let mut input = vec![10.0, -10.0, 0.1, -0.1]; // block 1
        input.extend_from_slice(&[1.0, -1.0, 0.01, -0.01]); // block 2
        let cfg = QuantizeConfig { block_size: 4, method: QuantMethod::AbsMax };
        let (q, _) = quantize_ternary_cpu(&input, &cfg).unwrap();
        assert_all_ternary(&q);
        assert_eq!(q.len(), 8);
    }

    // ── I2_S quantize / dequantize ────────────────────────────────

    #[test]
    fn i2s_roundtrip_basic() {
        let input = vec![1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0];
        let (packed, scales) = quantize_i2s_cpu(&input, 8).unwrap();
        let output = dequantize_i2s_cpu(&packed, &scales, 8, input.len()).unwrap();
        assert_eq!(output.len(), input.len());
        for (i, (&orig, &deq)) in input.iter().zip(output.iter()).enumerate() {
            if orig > 0.0 {
                assert!(deq > 0.0, "pos mismatch at {i}: orig={orig} deq={deq}");
            } else if orig < 0.0 {
                assert!(deq < 0.0, "neg mismatch at {i}: orig={orig} deq={deq}");
            }
        }
    }

    #[test]
    fn i2s_packing_four_values() {
        let input = vec![1.0, -1.0, 0.0, 1.0];
        let (packed, _) = quantize_i2s_cpu(&input, 4).unwrap();
        assert_eq!(packed.len(), 1); // 4 values → 1 byte
        let byte = packed[0];
        assert_eq!(decode_i2s((byte >> 0) & 0x03), 1);
        assert_eq!(decode_i2s((byte >> 2) & 0x03), -1);
        assert_eq!(decode_i2s((byte >> 4) & 0x03), 0);
        assert_eq!(decode_i2s((byte >> 6) & 0x03), 1);
    }

    #[test]
    fn i2s_non_multiple_of_four_length() {
        let input = vec![1.0, -1.0, 0.5]; // 3 elements
        let (packed, scales) = quantize_i2s_cpu(&input, 4).unwrap();
        assert_eq!(packed.len(), 1); // ceil(3/4) = 1 byte
        let output = dequantize_i2s_cpu(&packed, &scales, 4, 3).unwrap();
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn i2s_block_size_32() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32 / 32.0) - 1.0).collect();
        let (packed, scales) = quantize_i2s_cpu(&input, 32).unwrap();
        assert_eq!(scales.len(), 2);
        let output = dequantize_i2s_cpu(&packed, &scales, 32, 64).unwrap();
        assert_eq!(output.len(), 64);
    }

    #[test]
    fn i2s_block_size_256() {
        let input: Vec<f32> = (0..512).map(|i| ((i as f32) * 0.01).sin()).collect();
        let (packed, scales) = quantize_i2s_cpu(&input, 256).unwrap();
        assert_eq!(scales.len(), 2);
        let output = dequantize_i2s_cpu(&packed, &scales, 256, 512).unwrap();
        assert_eq!(output.len(), 512);
    }

    #[test]
    fn i2s_empty_input() {
        let (packed, scales) = quantize_i2s_cpu(&[], 32).unwrap();
        assert!(packed.is_empty());
        assert!(scales.is_empty());
    }

    #[test]
    fn i2s_rejects_zero_block_size() {
        assert!(quantize_i2s_cpu(&[1.0], 0).is_err());
    }

    #[test]
    fn i2s_dequant_rejects_zero_block_size() {
        assert!(dequantize_i2s_cpu(&[0], &[1.0], 0, 1).is_err());
    }

    #[test]
    fn i2s_dequant_rejects_short_packed() {
        assert!(dequantize_i2s_cpu(&[0], &[1.0, 1.0], 4, 5).is_err());
    }

    #[test]
    fn i2s_dequant_rejects_short_scales() {
        assert!(dequantize_i2s_cpu(&[0, 0], &[1.0], 4, 5).is_err());
    }

    #[test]
    fn i2s_all_zeros() {
        let input = vec![0.0; 16];
        let (packed, scales) = quantize_i2s_cpu(&input, 8).unwrap();
        assert!(packed.iter().all(|&b| b == 0));
        assert!(scales.iter().all(|&s| s == 0.0));
        let output = dequantize_i2s_cpu(&packed, &scales, 8, 16).unwrap();
        assert!(output.iter().all(|&v| v == 0.0));
    }

    // ── Scale computation ─────────────────────────────────────────

    #[test]
    fn scale_absmax_matches_manual() {
        let input = vec![-7.0, 3.0, 5.0, -2.0];
        let cfg = QuantizeConfig { block_size: 4, method: QuantMethod::AbsMax };
        let scales = calibrate_scales(&input, &cfg).unwrap();
        assert!((scales[0] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn scale_minmax_all_positive() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let cfg = QuantizeConfig { block_size: 8, method: QuantMethod::MinMax };
        let scales = calibrate_scales(&input, &cfg).unwrap();
        // (8 - 2) / 2 = 3.0
        assert!((scales[0] - 3.0).abs() < 1e-6);
    }

    // ── Batch / larger operations ─────────────────────────────────

    #[test]
    fn i2s_large_batch() {
        let input: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.1).sin()).collect();
        let (packed, scales) = quantize_i2s_cpu(&input, 32).unwrap();
        assert_eq!(scales.len(), 32);
        let output = dequantize_i2s_cpu(&packed, &scales, 32, 1024).unwrap();
        assert_eq!(output.len(), 1024);
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn ternary_large_batch() {
        let input: Vec<f32> = (0..2048).map(|i| ((i as f32) * 0.05).cos()).collect();
        let cfg = QuantizeConfig { block_size: 64, method: QuantMethod::AbsMax };
        let (q, scale) = quantize_ternary_cpu(&input, &cfg).unwrap();
        assert_eq!(q.len(), 2048);
        assert_all_ternary(&q);
        assert!(scale > 0.0);
    }

    // ── encode / decode roundtrip ─────────────────────────────────

    #[test]
    fn encode_decode_i2s_roundtrip() {
        for val in [-1_i8, 0, 1] {
            let code = encode_i2s(val);
            let decoded = decode_i2s(code);
            assert_eq!(val, decoded, "roundtrip failed for {val}");
        }
    }

    #[test]
    fn decode_i2s_unused_code() {
        // 0b10 is unused in I2_S spec → maps to 0
        assert_eq!(decode_i2s(0b10), 0);
    }
}
