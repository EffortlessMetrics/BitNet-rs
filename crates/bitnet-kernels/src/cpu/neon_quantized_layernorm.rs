//! ARM NEON quantized layer normalization kernels for Apple Silicon.
//!
//! Provides NEON-accelerated layer normalization primitives that operate on
//! quantized (int8) data or fuse normalization with quantization in a single
//! pass. Designed for BitNet 1-bit inference where weights and activations
//! are quantized to low bit-widths.
//!
//! Key functions:
//! - [`quantized_rms_norm_neon`] — RMS normalization on quantized int8 data
//! - [`quantized_layer_norm_neon`] — full LayerNorm with scale/bias on int8
//! - [`fused_layernorm_quantize_neon`] — fuse f32 LayerNorm + int8 quantize
//! - [`online_variance_neon`] — Welford's online variance with NEON
//! - [`apply_scale_bias_neon`] — vectorized scale+bias application

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Welford's online variance ──────────────────────────────────────

/// Welford's online variance state.
#[cfg(target_arch = "aarch64")]
#[derive(Debug, Clone, Copy)]
pub struct WelfordState {
    /// Running mean.
    pub mean: f32,
    /// Running M2 accumulator (sum of squared deviations).
    pub m2: f32,
    /// Number of samples seen.
    pub count: usize,
}

#[cfg(target_arch = "aarch64")]
impl WelfordState {
    /// Create a new empty Welford state.
    #[inline]
    pub fn new() -> Self {
        Self { mean: 0.0, m2: 0.0, count: 0 }
    }

    /// Return the population variance (or 0 if fewer than 1 sample).
    #[inline]
    pub fn variance(&self) -> f32 {
        if self.count == 0 { 0.0 } else { self.m2 / self.count as f32 }
    }
}

#[cfg(target_arch = "aarch64")]
impl Default for WelfordState {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute mean and variance using Welford's online algorithm with NEON
/// accumulation.
///
/// Processes 4 elements at a time in NEON lanes, then merges partial
/// accumulators with a scalar tail. Returns `(mean, variance)`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn online_variance_neon(data: &[f32]) -> WelfordState {
    let n = data.len();
    if n == 0 {
        return WelfordState::new();
    }

    // Two-pass is more numerically stable for SIMD — first pass computes mean,
    // second pass computes variance around that mean. We use NEON for both.
    unsafe {
        let mean = neon_sum(data) / n as f32;

        let chunks = n / 4;
        let remainder = n % 4;
        let ptr = data.as_ptr();
        let mean_vec = vdupq_n_f32(mean);
        let mut acc = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * 4));
            let diff = vsubq_f32(v, mean_vec);
            acc = vfmaq_f32(acc, diff, diff);
        }

        let mut m2: f32 = vaddvq_f32(acc);
        let tail_start = chunks * 4;
        for i in 0..remainder {
            let d = data[tail_start + i] - mean;
            m2 += d * d;
        }

        WelfordState { mean, m2, count: n }
    }
}

// ── NEON sum helper ────────────────────────────────────────────────

/// Compute sum of all elements using NEON horizontal adds.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_sum(data: &[f32]) -> f32 {
    let n = data.len();
    let chunks = n / 4;
    let remainder = n % 4;

    unsafe {
        let mut sum_vec = vdupq_n_f32(0.0);
        let ptr = data.as_ptr();

        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * 4));
            sum_vec = vaddq_f32(sum_vec, v);
        }

        let mut sum: f32 = vaddvq_f32(sum_vec);
        let tail_start = chunks * 4;
        for i in 0..remainder {
            sum += data[tail_start + i];
        }
        sum
    }
}

// ── apply_scale_bias_neon ──────────────────────────────────────────

/// Apply `output[i] = input[i] * scale[i] + bias[i]` using NEON.
///
/// Processes 4 lanes at a time with `vfmaq_f32` for fused multiply-add.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `output`, `scale`, or `bias` length differs from `input`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn apply_scale_bias_neon(
    input: &[f32],
    output: &mut [f32],
    scale: &[f32],
    bias: &[f32],
) {
    let n = input.len();
    assert_eq!(output.len(), n, "output length mismatch");
    assert_eq!(scale.len(), n, "scale length mismatch");
    assert_eq!(bias.len(), n, "bias length mismatch");

    if n == 0 {
        return;
    }

    let chunks = n / 4;
    let remainder = n % 4;

    unsafe {
        let in_ptr = input.as_ptr();
        let sc_ptr = scale.as_ptr();
        let bi_ptr = bias.as_ptr();
        let out_ptr = output.as_mut_ptr();

        for i in 0..chunks {
            let off = i * 4;
            let v = vld1q_f32(in_ptr.add(off));
            let s = vld1q_f32(sc_ptr.add(off));
            let b = vld1q_f32(bi_ptr.add(off));
            let result = vfmaq_f32(b, v, s); // b + v * s
            vst1q_f32(out_ptr.add(off), result);
        }
    }

    // Scalar tail.
    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        output[idx] = input[idx] * scale[idx] + bias[idx];
    }
}

// ── quantized_rms_norm_neon ────────────────────────────────────────

/// RMS normalization on int8 quantized data with NEON.
///
/// Dequantizes input using `input_scale`, applies RMS normalization with
/// `gamma` weights, then re-quantizes to int8 with `output_scale`.
///
/// Formula per element:
/// ```text
/// x_f32 = input[i] as f32 * input_scale
/// rms   = sqrt(mean(x_f32²) + eps)
/// norm  = gamma[i] * x_f32 / rms
/// out   = clamp(round(norm / output_scale), -128, 127) as i8
/// ```
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `output` or `gamma` length differs from `input`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn quantized_rms_norm_neon(
    input: &[i8],
    output: &mut [i8],
    gamma: &[f32],
    input_scale: f32,
    output_scale: f32,
    eps: f32,
) {
    let n = input.len();
    assert_eq!(output.len(), n, "output length mismatch");
    assert_eq!(gamma.len(), n, "gamma length mismatch");

    if n == 0 {
        return;
    }

    // Dequantize to f32.
    let dequant: Vec<f32> = input.iter().map(|&x| x as f32 * input_scale).collect();

    // Compute mean(x²) using NEON.
    let mean_sq = unsafe { neon_mean_of_squares(&dequant) };
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    let inv_out = 1.0 / output_scale;

    // Normalize, scale by gamma, re-quantize.
    let chunks = n / 4;
    let remainder = n % 4;

    unsafe {
        let inv_rms_vec = vdupq_n_f32(inv_rms);
        let inv_out_vec = vdupq_n_f32(inv_out);
        let dq_ptr = dequant.as_ptr();
        let gam_ptr = gamma.as_ptr();

        for i in 0..chunks {
            let off = i * 4;
            let v = vld1q_f32(dq_ptr.add(off));
            let g = vld1q_f32(gam_ptr.add(off));

            let normed = vmulq_f32(v, inv_rms_vec);
            let scaled = vmulq_f32(g, normed);
            let quantized = vmulq_f32(scaled, inv_out_vec);

            for lane in 0..4usize {
                let val = extract_lane_f32(quantized, lane);
                output[off + lane] = val.round().clamp(-128.0, 127.0) as i8;
            }
        }
    }

    // Scalar tail.
    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        let normed = dequant[idx] * inv_rms;
        let scaled = gamma[idx] * normed;
        output[idx] = (scaled * inv_out).round().clamp(-128.0, 127.0) as i8;
    }
}

// ── quantized_layer_norm_neon ──────────────────────────────────────

/// Full layer normalization on int8 quantized data with scale and bias.
///
/// Dequantizes, applies LayerNorm `gamma * (x - mean) / std + beta`,
/// then re-quantizes to int8.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `output`, `gamma`, or `beta` length differs from `input`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn quantized_layer_norm_neon(
    input: &[i8],
    output: &mut [i8],
    gamma: &[f32],
    beta: &[f32],
    input_scale: f32,
    output_scale: f32,
    eps: f32,
) {
    let n = input.len();
    assert_eq!(output.len(), n, "output length mismatch");
    assert_eq!(gamma.len(), n, "gamma length mismatch");
    assert_eq!(beta.len(), n, "beta length mismatch");

    if n == 0 {
        return;
    }

    // Dequantize to f32.
    let dequant: Vec<f32> = input.iter().map(|&x| x as f32 * input_scale).collect();

    // Compute mean and variance using NEON.
    let state = unsafe { online_variance_neon(&dequant) };
    let inv_std = 1.0 / (state.variance() + eps).sqrt();
    let mean = state.mean;
    let inv_out = 1.0 / output_scale;

    // Normalize with affine params, re-quantize.
    let chunks = n / 4;
    let remainder = n % 4;

    unsafe {
        let mean_vec = vdupq_n_f32(mean);
        let inv_std_vec = vdupq_n_f32(inv_std);
        let inv_out_vec = vdupq_n_f32(inv_out);
        let dq_ptr = dequant.as_ptr();
        let gam_ptr = gamma.as_ptr();
        let bet_ptr = beta.as_ptr();

        for i in 0..chunks {
            let off = i * 4;
            let v = vld1q_f32(dq_ptr.add(off));
            let g = vld1q_f32(gam_ptr.add(off));
            let b = vld1q_f32(bet_ptr.add(off));

            let centered = vsubq_f32(v, mean_vec);
            let normed = vmulq_f32(centered, inv_std_vec);
            let scaled = vfmaq_f32(b, g, normed); // b + g * normed
            let quantized = vmulq_f32(scaled, inv_out_vec);

            for lane in 0..4usize {
                let val = extract_lane_f32(quantized, lane);
                output[off + lane] = val.round().clamp(-128.0, 127.0) as i8;
            }
        }
    }

    // Scalar tail.
    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        let normed = (dequant[idx] - mean) * inv_std;
        let scaled = gamma[idx] * normed + beta[idx];
        output[idx] = (scaled * inv_out).round().clamp(-128.0, 127.0) as i8;
    }
}

// ── fused_layernorm_quantize_neon ──────────────────────────────────

/// Fused layer normalization and int8 quantization in a single pass.
///
/// Takes f32 input, applies LayerNorm with `gamma`/`beta`, then quantizes
/// to int8 in-flight without a separate normalization buffer.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `output`, `gamma`, or `beta` length differs from `input`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn fused_layernorm_quantize_neon(
    input: &[f32],
    output: &mut [i8],
    gamma: &[f32],
    beta: &[f32],
    output_scale: f32,
    eps: f32,
) {
    let n = input.len();
    assert_eq!(output.len(), n, "output length mismatch");
    assert_eq!(gamma.len(), n, "gamma length mismatch");
    assert_eq!(beta.len(), n, "beta length mismatch");

    if n == 0 {
        return;
    }

    // Pass 1: mean + variance via NEON.
    let state = unsafe { online_variance_neon(input) };
    let inv_std = 1.0 / (state.variance() + eps).sqrt();
    let mean = state.mean;
    let inv_out = 1.0 / output_scale;

    // Pass 2: normalize + quantize.
    let chunks = n / 4;
    let remainder = n % 4;

    unsafe {
        let mean_vec = vdupq_n_f32(mean);
        let inv_std_vec = vdupq_n_f32(inv_std);
        let inv_out_vec = vdupq_n_f32(inv_out);
        let in_ptr = input.as_ptr();
        let gam_ptr = gamma.as_ptr();
        let bet_ptr = beta.as_ptr();

        for i in 0..chunks {
            let off = i * 4;
            let v = vld1q_f32(in_ptr.add(off));
            let g = vld1q_f32(gam_ptr.add(off));
            let b = vld1q_f32(bet_ptr.add(off));

            let centered = vsubq_f32(v, mean_vec);
            let normed = vmulq_f32(centered, inv_std_vec);
            let scaled = vfmaq_f32(b, g, normed);
            let quantized = vmulq_f32(scaled, inv_out_vec);

            for lane in 0..4usize {
                let val = extract_lane_f32(quantized, lane);
                output[off + lane] = val.round().clamp(-128.0, 127.0) as i8;
            }
        }
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        let normed = (input[idx] - mean) * inv_std;
        let scaled = gamma[idx] * normed + beta[idx];
        output[idx] = (scaled * inv_out).round().clamp(-128.0, 127.0) as i8;
    }
}

// ── NEON helpers ───────────────────────────────────────────────────

/// Compute `mean(x²)` using NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_mean_of_squares(data: &[f32]) -> f32 {
    let n = data.len();
    let chunks = n / 4;
    let remainder = n % 4;

    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let ptr = data.as_ptr();

        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * 4));
            acc = vfmaq_f32(acc, v, v);
        }

        let mut sq_sum: f32 = vaddvq_f32(acc);
        let tail_start = chunks * 4;
        for i in 0..remainder {
            let x = data[tail_start + i];
            sq_sum += x * x;
        }

        sq_sum / n as f32
    }
}

/// Extract a single f32 lane from a NEON vector by index.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn extract_lane_f32(v: float32x4_t, lane: usize) -> f32 {
    // SAFETY: we transmute to an array which is always valid for float32x4_t.
    let arr: [f32; 4] = unsafe { std::mem::transmute(v) };
    arr[lane]
}

// ── Scalar references (test-only) ──────────────────────────────────

#[cfg(test)]
fn scalar_rms_norm_quantized(
    input: &[i8],
    gamma: &[f32],
    input_scale: f32,
    output_scale: f32,
    eps: f32,
) -> Vec<i8> {
    let dequant: Vec<f32> = input.iter().map(|&x| x as f32 * input_scale).collect();
    let n = dequant.len();
    let mean_sq: f32 = dequant.iter().map(|x| x * x).sum::<f32>() / n as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    let inv_out = 1.0 / output_scale;
    dequant
        .iter()
        .enumerate()
        .map(|(i, &x)| (gamma[i] * x * inv_rms * inv_out).round().clamp(-128.0, 127.0) as i8)
        .collect()
}

#[cfg(test)]
fn scalar_layer_norm_quantized(
    input: &[i8],
    gamma: &[f32],
    beta: &[f32],
    input_scale: f32,
    output_scale: f32,
    eps: f32,
) -> Vec<i8> {
    let dequant: Vec<f32> = input.iter().map(|&x| x as f32 * input_scale).collect();
    let n = dequant.len();
    let mean: f32 = dequant.iter().sum::<f32>() / n as f32;
    let var: f32 = dequant.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    let inv_out = 1.0 / output_scale;
    dequant
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let normed = (x - mean) * inv_std;
            let scaled = gamma[i] * normed + beta[i];
            (scaled * inv_out).round().clamp(-128.0, 127.0) as i8
        })
        .collect()
}

#[cfg(test)]
fn scalar_fused_layernorm_quantize(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output_scale: f32,
    eps: f32,
) -> Vec<i8> {
    let n = input.len();
    let mean: f32 = input.iter().sum::<f32>() / n as f32;
    let var: f32 = input.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    let inv_out = 1.0 / output_scale;
    input
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let normed = (x - mean) * inv_std;
            let scaled = gamma[i] * normed + beta[i];
            (scaled * inv_out).round().clamp(-128.0, 127.0) as i8
        })
        .collect()
}

#[cfg(test)]
fn scalar_apply_scale_bias(input: &[f32], scale: &[f32], bias: &[f32]) -> Vec<f32> {
    input.iter().enumerate().map(|(i, &x)| x * scale[i] + bias[i]).collect()
}

#[cfg(test)]
fn scalar_online_variance(data: &[f32]) -> (f32, f32) {
    let n = data.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mean = data.iter().sum::<f32>() / n as f32;
    let var = data.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
    (mean, var)
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;
    const TOL: f32 = 1e-4;

    fn assert_approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "mismatch at index {i}: {x} vs {y} (diff {})",
                (x - y).abs()
            );
        }
    }

    fn assert_i8_eq(a: &[i8], b: &[i8]) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x, y, "mismatch at index {i}: {x} vs {y}");
        }
    }

    /// Allow ±1 quantization tolerance for int8 outputs.
    fn assert_i8_approx(a: &[i8], b: &[i8], tol: i8) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x as i16 - y as i16).abs() <= tol as i16,
                "mismatch at index {i}: {x} vs {y} (diff {})",
                (x as i16 - y as i16).abs()
            );
        }
    }

    // ── online_variance_neon tests ────────────────────────────────

    #[test]
    fn test_online_variance_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let state = unsafe { online_variance_neon(&data) };
        let (exp_mean, exp_var) = scalar_online_variance(&data);
        assert!((state.mean - exp_mean).abs() < TOL, "mean: {} vs {}", state.mean, exp_mean);
        assert!(
            (state.variance() - exp_var).abs() < TOL,
            "var: {} vs {}",
            state.variance(),
            exp_var
        );
        assert_eq!(state.count, 8);
    }

    #[test]
    fn test_online_variance_empty() {
        let data: Vec<f32> = vec![];
        let state = unsafe { online_variance_neon(&data) };
        assert_eq!(state.count, 0);
        assert_eq!(state.variance(), 0.0);
    }

    #[test]
    fn test_online_variance_single() {
        let data = vec![42.0];
        let state = unsafe { online_variance_neon(&data) };
        assert!((state.mean - 42.0).abs() < TOL);
        assert!(state.variance().abs() < TOL);
        assert_eq!(state.count, 1);
    }

    #[test]
    fn test_online_variance_constant() {
        let data = vec![5.0; 16];
        let state = unsafe { online_variance_neon(&data) };
        assert!((state.mean - 5.0).abs() < TOL);
        assert!(state.variance().abs() < TOL);
    }

    #[test]
    fn test_online_variance_non_aligned() {
        let data: Vec<f32> = (0..13).map(|i| i as f32).collect();
        let state = unsafe { online_variance_neon(&data) };
        let (exp_mean, exp_var) = scalar_online_variance(&data);
        assert!((state.mean - exp_mean).abs() < TOL);
        assert!((state.variance() - exp_var).abs() < TOL);
    }

    #[test]
    fn test_online_variance_negative_values() {
        let data = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
        let state = unsafe { online_variance_neon(&data) };
        let (exp_mean, exp_var) = scalar_online_variance(&data);
        assert!((state.mean - exp_mean).abs() < TOL);
        assert!((state.variance() - exp_var).abs() < TOL);
    }

    #[test]
    fn test_online_variance_large() {
        let data: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let state = unsafe { online_variance_neon(&data) };
        let (exp_mean, exp_var) = scalar_online_variance(&data);
        assert!((state.mean - exp_mean).abs() < 1e-3);
        assert!((state.variance() - exp_var).abs() < 1e-2);
    }

    #[test]
    fn test_online_variance_two_elements() {
        let data = vec![10.0, 20.0];
        let state = unsafe { online_variance_neon(&data) };
        assert!((state.mean - 15.0).abs() < TOL);
        assert!((state.variance() - 25.0).abs() < TOL);
    }

    #[test]
    fn test_online_variance_three_elements() {
        let data = vec![1.0, 2.0, 3.0];
        let state = unsafe { online_variance_neon(&data) };
        let (exp_mean, exp_var) = scalar_online_variance(&data);
        assert!((state.mean - exp_mean).abs() < TOL);
        assert!((state.variance() - exp_var).abs() < TOL);
    }

    // ── apply_scale_bias_neon tests ───────────────────────────────

    #[test]
    fn test_apply_scale_bias_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let scale = vec![2.0; 8];
        let bias = vec![0.5; 8];
        let expected = scalar_apply_scale_bias(&input, &scale, &bias);
        let mut output = vec![0.0; 8];
        unsafe { apply_scale_bias_neon(&input, &mut output, &scale, &bias) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_apply_scale_bias_identity() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let scale = vec![1.0; 4];
        let bias = vec![0.0; 4];
        let mut output = vec![0.0; 4];
        unsafe { apply_scale_bias_neon(&input, &mut output, &scale, &bias) };
        assert_approx_eq(&output, &input, TOL);
    }

    #[test]
    fn test_apply_scale_bias_zero_scale() {
        let input = vec![100.0, 200.0, 300.0, 400.0];
        let scale = vec![0.0; 4];
        let bias = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        unsafe { apply_scale_bias_neon(&input, &mut output, &scale, &bias) };
        assert_approx_eq(&output, &bias, TOL);
    }

    #[test]
    fn test_apply_scale_bias_non_aligned() {
        let input: Vec<f32> = (0..11).map(|i| i as f32).collect();
        let scale: Vec<f32> = (0..11).map(|i| 0.5 + i as f32 * 0.1).collect();
        let bias: Vec<f32> = (0..11).map(|i| -1.0 + i as f32 * 0.2).collect();
        let expected = scalar_apply_scale_bias(&input, &scale, &bias);
        let mut output = vec![0.0; 11];
        unsafe { apply_scale_bias_neon(&input, &mut output, &scale, &bias) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_apply_scale_bias_empty() {
        let input: Vec<f32> = vec![];
        let scale: Vec<f32> = vec![];
        let bias: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];
        unsafe { apply_scale_bias_neon(&input, &mut output, &scale, &bias) };
        assert!(output.is_empty());
    }

    #[test]
    fn test_apply_scale_bias_single() {
        let input = vec![5.0];
        let scale = vec![3.0];
        let bias = vec![1.0];
        let mut output = vec![0.0; 1];
        unsafe { apply_scale_bias_neon(&input, &mut output, &scale, &bias) };
        assert!((output[0] - 16.0).abs() < TOL);
    }

    #[test]
    fn test_apply_scale_bias_negative() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let scale = vec![2.0; 5];
        let bias = vec![10.0; 5];
        let expected = scalar_apply_scale_bias(&input, &scale, &bias);
        let mut output = vec![0.0; 5];
        unsafe { apply_scale_bias_neon(&input, &mut output, &scale, &bias) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_apply_scale_bias_large() {
        let n = 512;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 2.5).collect();
        let scale: Vec<f32> = (0..n).map(|i| 0.8 + (i % 5) as f32 * 0.1).collect();
        let bias: Vec<f32> = (0..n).map(|i| -0.5 + (i % 3) as f32 * 0.3).collect();
        let expected = scalar_apply_scale_bias(&input, &scale, &bias);
        let mut output = vec![0.0; n];
        unsafe { apply_scale_bias_neon(&input, &mut output, &scale, &bias) };
        assert_approx_eq(&output, &expected, TOL);
    }

    // ── quantized_rms_norm_neon tests ─────────────────────────────

    #[test]
    fn test_quantized_rms_norm_basic() {
        let input: Vec<i8> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let gamma = vec![1.0; 8];
        let input_scale = 0.1;
        let output_scale = 0.1;
        let expected = scalar_rms_norm_quantized(&input, &gamma, input_scale, output_scale, EPS);
        let mut output = vec![0i8; 8];
        unsafe {
            quantized_rms_norm_neon(&input, &mut output, &gamma, input_scale, output_scale, EPS)
        };
        assert_i8_approx(&output, &expected, 1);
    }

    #[test]
    fn test_quantized_rms_norm_with_gamma() {
        let input: Vec<i8> = vec![10, -20, 30, -40, 50];
        let gamma = vec![0.5, 1.0, 1.5, 2.0, 0.1];
        let input_scale = 0.05;
        let output_scale = 0.05;
        let expected = scalar_rms_norm_quantized(&input, &gamma, input_scale, output_scale, EPS);
        let mut output = vec![0i8; 5];
        unsafe {
            quantized_rms_norm_neon(&input, &mut output, &gamma, input_scale, output_scale, EPS)
        };
        assert_i8_approx(&output, &expected, 1);
    }

    #[test]
    fn test_quantized_rms_norm_empty() {
        let input: Vec<i8> = vec![];
        let gamma: Vec<f32> = vec![];
        let mut output: Vec<i8> = vec![];
        unsafe { quantized_rms_norm_neon(&input, &mut output, &gamma, 0.1, 0.1, EPS) };
        assert!(output.is_empty());
    }

    #[test]
    fn test_quantized_rms_norm_single() {
        let input: Vec<i8> = vec![100];
        let gamma = vec![1.0];
        let input_scale = 0.01;
        let output_scale = 0.01;
        let expected = scalar_rms_norm_quantized(&input, &gamma, input_scale, output_scale, EPS);
        let mut output = vec![0i8; 1];
        unsafe {
            quantized_rms_norm_neon(&input, &mut output, &gamma, input_scale, output_scale, EPS)
        };
        assert_i8_approx(&output, &expected, 1);
    }

    #[test]
    fn test_quantized_rms_norm_saturation() {
        // Large gamma should cause clamping at i8 boundaries.
        let input: Vec<i8> = vec![100, 100, 100, 100];
        let gamma = vec![10.0; 4];
        let input_scale = 1.0;
        let output_scale = 0.01;
        let mut output = vec![0i8; 4];
        unsafe {
            quantized_rms_norm_neon(&input, &mut output, &gamma, input_scale, output_scale, EPS)
        };
        for &v in &output {
            assert!(v == 127 || v == -128, "expected saturation, got {v}");
        }
    }

    #[test]
    fn test_quantized_rms_norm_non_aligned() {
        let input: Vec<i8> = (0..13).map(|i| (i * 5 - 30) as i8).collect();
        let gamma = vec![1.0; 13];
        let input_scale = 0.1;
        let output_scale = 0.1;
        let expected = scalar_rms_norm_quantized(&input, &gamma, input_scale, output_scale, EPS);
        let mut output = vec![0i8; 13];
        unsafe {
            quantized_rms_norm_neon(&input, &mut output, &gamma, input_scale, output_scale, EPS)
        };
        assert_i8_approx(&output, &expected, 1);
    }

    #[test]
    fn test_quantized_rms_norm_large_scale() {
        let input: Vec<i8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let gamma = vec![1.0; 8];
        let input_scale = 10.0;
        let output_scale = 10.0;
        let expected = scalar_rms_norm_quantized(&input, &gamma, input_scale, output_scale, EPS);
        let mut output = vec![0i8; 8];
        unsafe {
            quantized_rms_norm_neon(&input, &mut output, &gamma, input_scale, output_scale, EPS)
        };
        assert_i8_approx(&output, &expected, 1);
    }

    #[test]
    fn test_quantized_rms_norm_all_zeros() {
        let input: Vec<i8> = vec![0; 8];
        let gamma = vec![1.0; 8];
        let mut output = vec![0i8; 8];
        unsafe { quantized_rms_norm_neon(&input, &mut output, &gamma, 0.1, 0.1, EPS) };
        // All zero input → output should be all zeros (0 / rms ≈ 0).
        for &v in &output {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn test_quantized_rms_norm_negative_inputs() {
        let input: Vec<i8> = vec![-50, -40, -30, -20, -10, 10, 20, 30];
        let gamma = vec![1.0; 8];
        let input_scale = 0.1;
        let output_scale = 0.1;
        let expected = scalar_rms_norm_quantized(&input, &gamma, input_scale, output_scale, EPS);
        let mut output = vec![0i8; 8];
        unsafe {
            quantized_rms_norm_neon(&input, &mut output, &gamma, input_scale, output_scale, EPS)
        };
        assert_i8_approx(&output, &expected, 1);
    }

    // ── quantized_layer_norm_neon tests ───────────────────────────

    #[test]
    fn test_quantized_layer_norm_basic() {
        let input: Vec<i8> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let input_scale = 0.1;
        let output_scale = 0.1;
        let expected =
            scalar_layer_norm_quantized(&input, &gamma, &beta, input_scale, output_scale, EPS);
        let mut output = vec![0i8; 8];
        unsafe {
            quantized_layer_norm_neon(
                &input,
                &mut output,
                &gamma,
                &beta,
                input_scale,
                output_scale,
                EPS,
            )
        };
        assert_i8_approx(&output, &expected, 1);
    }

    #[test]
    fn test_quantized_layer_norm_with_affine() {
        let input: Vec<i8> = vec![10, -20, 30, -40, 50];
        let gamma = vec![0.5, 1.0, 1.5, 2.0, 0.1];
        let beta = vec![0.1, -0.1, 0.0, 0.5, -0.5];
        let input_scale = 0.05;
        let output_scale = 0.1;
        let expected =
            scalar_layer_norm_quantized(&input, &gamma, &beta, input_scale, output_scale, EPS);
        let mut output = vec![0i8; 5];
        unsafe {
            quantized_layer_norm_neon(
                &input,
                &mut output,
                &gamma,
                &beta,
                input_scale,
                output_scale,
                EPS,
            )
        };
        assert_i8_approx(&output, &expected, 1);
    }

    #[test]
    fn test_quantized_layer_norm_zero_variance() {
        let input: Vec<i8> = vec![42; 8];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let mut output = vec![0i8; 8];
        unsafe { quantized_layer_norm_neon(&input, &mut output, &gamma, &beta, 0.1, 0.1, EPS) };
        // All identical → zero variance → output ≈ beta/output_scale ≈ 0.
        for &v in &output {
            assert!(v.abs() <= 1, "expected ~0 with zero variance, got {v}");
        }
    }

    #[test]
    fn test_quantized_layer_norm_empty() {
        let input: Vec<i8> = vec![];
        let gamma: Vec<f32> = vec![];
        let beta: Vec<f32> = vec![];
        let mut output: Vec<i8> = vec![];
        unsafe { quantized_layer_norm_neon(&input, &mut output, &gamma, &beta, 0.1, 0.1, EPS) };
        assert!(output.is_empty());
    }

    #[test]
    fn test_quantized_layer_norm_single() {
        let input: Vec<i8> = vec![50];
        let gamma = vec![2.0];
        let beta = vec![1.0];
        let input_scale = 0.1;
        let output_scale = 0.1;
        let expected =
            scalar_layer_norm_quantized(&input, &gamma, &beta, input_scale, output_scale, EPS);
        let mut output = vec![0i8; 1];
        unsafe {
            quantized_layer_norm_neon(
                &input,
                &mut output,
                &gamma,
                &beta,
                input_scale,
                output_scale,
                EPS,
            )
        };
        assert_i8_approx(&output, &expected, 1);
    }

    #[test]
    fn test_quantized_layer_norm_non_aligned() {
        let input: Vec<i8> = (0..15).map(|i| (i * 7 - 50) as i8).collect();
        let gamma = vec![1.0; 15];
        let beta = vec![0.0; 15];
        let input_scale = 0.1;
        let output_scale = 0.1;
        let expected =
            scalar_layer_norm_quantized(&input, &gamma, &beta, input_scale, output_scale, EPS);
        let mut output = vec![0i8; 15];
        unsafe {
            quantized_layer_norm_neon(
                &input,
                &mut output,
                &gamma,
                &beta,
                input_scale,
                output_scale,
                EPS,
            )
        };
        assert_i8_approx(&output, &expected, 1);
    }

    #[test]
    fn test_quantized_layer_norm_saturation() {
        let input: Vec<i8> = vec![127, 127, 127, 127];
        let gamma = vec![10.0; 4];
        let beta = vec![100.0; 4];
        let mut output = vec![0i8; 4];
        unsafe { quantized_layer_norm_neon(&input, &mut output, &gamma, &beta, 1.0, 0.01, EPS) };
        // Large beta/output_scale → should clamp at boundaries.
        for &v in &output {
            assert!(v == 127 || v == -128, "expected saturation, got {v}");
        }
    }

    #[test]
    fn test_quantized_layer_norm_large() {
        let n = 256;
        let input: Vec<i8> = (0..n).map(|i| ((i * 3 + 7) % 256) as i8).collect();
        let gamma = vec![1.0; n];
        let beta = vec![0.0; n];
        let input_scale = 0.05;
        let output_scale = 0.05;
        let expected =
            scalar_layer_norm_quantized(&input, &gamma, &beta, input_scale, output_scale, EPS);
        let mut output = vec![0i8; n];
        unsafe {
            quantized_layer_norm_neon(
                &input,
                &mut output,
                &gamma,
                &beta,
                input_scale,
                output_scale,
                EPS,
            )
        };
        assert_i8_approx(&output, &expected, 1);
    }

    #[test]
    fn test_quantized_layer_norm_negative_inputs() {
        let input: Vec<i8> = vec![-100, -80, -60, -40, -20, 0, 20, 40];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let input_scale = 0.1;
        let output_scale = 0.1;
        let expected =
            scalar_layer_norm_quantized(&input, &gamma, &beta, input_scale, output_scale, EPS);
        let mut output = vec![0i8; 8];
        unsafe {
            quantized_layer_norm_neon(
                &input,
                &mut output,
                &gamma,
                &beta,
                input_scale,
                output_scale,
                EPS,
            )
        };
        assert_i8_approx(&output, &expected, 1);
    }

    // ── fused_layernorm_quantize_neon tests ───────────────────────

    #[test]
    fn test_fused_layernorm_quantize_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let output_scale = 0.1;
        let expected = scalar_fused_layernorm_quantize(&input, &gamma, &beta, output_scale, EPS);
        let mut output = vec![0i8; 8];
        unsafe {
            fused_layernorm_quantize_neon(&input, &mut output, &gamma, &beta, output_scale, EPS)
        };
        assert_i8_approx(&output, &expected, 1);
    }

    #[test]
    fn test_fused_layernorm_quantize_with_affine() {
        let input = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let gamma = vec![0.5, 1.0, 1.5, 2.0, 0.1];
        let beta = vec![0.1, -0.1, 0.0, 0.5, -0.5];
        let output_scale = 0.1;
        let expected = scalar_fused_layernorm_quantize(&input, &gamma, &beta, output_scale, EPS);
        let mut output = vec![0i8; 5];
        unsafe {
            fused_layernorm_quantize_neon(&input, &mut output, &gamma, &beta, output_scale, EPS)
        };
        assert_i8_approx(&output, &expected, 1);
    }

    #[test]
    fn test_fused_layernorm_quantize_empty() {
        let input: Vec<f32> = vec![];
        let gamma: Vec<f32> = vec![];
        let beta: Vec<f32> = vec![];
        let mut output: Vec<i8> = vec![];
        unsafe { fused_layernorm_quantize_neon(&input, &mut output, &gamma, &beta, 0.1, EPS) };
        assert!(output.is_empty());
    }

    #[test]
    fn test_fused_layernorm_quantize_single() {
        let input = vec![42.0];
        let gamma = vec![2.0];
        let beta = vec![1.0];
        let output_scale = 0.1;
        let expected = scalar_fused_layernorm_quantize(&input, &gamma, &beta, output_scale, EPS);
        let mut output = vec![0i8; 1];
        unsafe {
            fused_layernorm_quantize_neon(&input, &mut output, &gamma, &beta, output_scale, EPS)
        };
        assert_i8_approx(&output, &expected, 1);
    }

    #[test]
    fn test_fused_layernorm_quantize_zero_variance() {
        let input = vec![5.0; 8];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let mut output = vec![0i8; 8];
        unsafe { fused_layernorm_quantize_neon(&input, &mut output, &gamma, &beta, 0.1, EPS) };
        for &v in &output {
            assert!(v.abs() <= 1, "expected ~0, got {v}");
        }
    }

    #[test]
    fn test_fused_layernorm_quantize_non_aligned() {
        let input: Vec<f32> = (0..17).map(|i| i as f32 * 0.3 - 2.5).collect();
        let gamma = vec![1.0; 17];
        let beta = vec![0.0; 17];
        let output_scale = 0.1;
        let expected = scalar_fused_layernorm_quantize(&input, &gamma, &beta, output_scale, EPS);
        let mut output = vec![0i8; 17];
        unsafe {
            fused_layernorm_quantize_neon(&input, &mut output, &gamma, &beta, output_scale, EPS)
        };
        assert_i8_approx(&output, &expected, 1);
    }

    #[test]
    fn test_fused_layernorm_quantize_saturation() {
        let input = vec![100.0, 100.0, -100.0, -100.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let output_scale = 0.001; // Very small scale → large quantized values → clamp.
        let mut output = vec![0i8; 4];
        unsafe {
            fused_layernorm_quantize_neon(&input, &mut output, &gamma, &beta, output_scale, EPS)
        };
        // Normalized values are ±1, divided by 0.001 → ±1000 → clamp.
        assert!(output[0] == 127);
        assert!(output[2] == -128);
    }

    #[test]
    fn test_fused_layernorm_quantize_large() {
        let n = 512;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 2.56).collect();
        let gamma = vec![1.0; n];
        let beta = vec![0.0; n];
        let output_scale = 0.05;
        let expected = scalar_fused_layernorm_quantize(&input, &gamma, &beta, output_scale, EPS);
        let mut output = vec![0i8; n];
        unsafe {
            fused_layernorm_quantize_neon(&input, &mut output, &gamma, &beta, output_scale, EPS)
        };
        assert_i8_approx(&output, &expected, 1);
    }

    #[test]
    fn test_fused_layernorm_quantize_negative_input() {
        let input = vec![-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let output_scale = 0.1;
        let expected = scalar_fused_layernorm_quantize(&input, &gamma, &beta, output_scale, EPS);
        let mut output = vec![0i8; 8];
        unsafe {
            fused_layernorm_quantize_neon(&input, &mut output, &gamma, &beta, output_scale, EPS)
        };
        assert_i8_approx(&output, &expected, 1);
    }

    // ── Parity / cross-check tests ────────────────────────────────

    #[test]
    fn test_rms_norm_parity_varying_scales() {
        for &(in_s, out_s) in &[(0.01, 0.01), (0.1, 0.05), (0.5, 0.1), (1.0, 1.0)] {
            let input: Vec<i8> = (0..16).map(|i| (i * 8 - 60) as i8).collect();
            let gamma = vec![1.0; 16];
            let expected = scalar_rms_norm_quantized(&input, &gamma, in_s, out_s, EPS);
            let mut output = vec![0i8; 16];
            unsafe { quantized_rms_norm_neon(&input, &mut output, &gamma, in_s, out_s, EPS) };
            assert_i8_approx(&output, &expected, 1);
        }
    }

    #[test]
    fn test_layer_norm_parity_varying_scales() {
        for &(in_s, out_s) in &[(0.01, 0.01), (0.1, 0.05), (0.5, 0.1), (1.0, 1.0)] {
            let input: Vec<i8> = (0..16).map(|i| (i * 8 - 60) as i8).collect();
            let gamma = vec![1.0; 16];
            let beta = vec![0.0; 16];
            let expected = scalar_layer_norm_quantized(&input, &gamma, &beta, in_s, out_s, EPS);
            let mut output = vec![0i8; 16];
            unsafe {
                quantized_layer_norm_neon(&input, &mut output, &gamma, &beta, in_s, out_s, EPS)
            };
            assert_i8_approx(&output, &expected, 1);
        }
    }

    #[test]
    fn test_fused_vs_separate_layernorm_quantize() {
        // Fused should match a two-step approach: layernorm → quantize.
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let output_scale = 0.1;

        // Fused path.
        let mut fused_out = vec![0i8; 8];
        unsafe {
            fused_layernorm_quantize_neon(&input, &mut fused_out, &gamma, &beta, output_scale, EPS)
        };

        // Separate path: scalar layernorm → quantize.
        let n = input.len();
        let mean: f32 = input.iter().sum::<f32>() / n as f32;
        let var: f32 = input.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
        let inv_std = 1.0 / (var + EPS).sqrt();
        let inv_out = 1.0 / output_scale;
        let separate_out: Vec<i8> = input
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let normed = (x - mean) * inv_std;
                let scaled = gamma[i] * normed + beta[i];
                (scaled * inv_out).round().clamp(-128.0, 127.0) as i8
            })
            .collect();

        assert_i8_eq(&fused_out, &separate_out);
    }

    #[test]
    fn test_online_variance_matches_two_pass() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32) * 0.07 - 3.5).collect();
        let state = unsafe { online_variance_neon(&data) };
        let (exp_mean, exp_var) = scalar_online_variance(&data);
        assert!(
            (state.mean - exp_mean).abs() < 1e-3,
            "mean mismatch: {} vs {}",
            state.mean,
            exp_mean
        );
        assert!(
            (state.variance() - exp_var).abs() < 1e-3,
            "var mismatch: {} vs {}",
            state.variance(),
            exp_var
        );
    }

    // ── Edge case tests ───────────────────────────────────────────

    #[test]
    fn test_quantized_rms_norm_min_max_i8() {
        let input: Vec<i8> = vec![-128, 127, -128, 127, -128, 127, -128, 127];
        let gamma = vec![1.0; 8];
        let expected = scalar_rms_norm_quantized(&input, &gamma, 0.01, 0.01, EPS);
        let mut output = vec![0i8; 8];
        unsafe { quantized_rms_norm_neon(&input, &mut output, &gamma, 0.01, 0.01, EPS) };
        assert_i8_approx(&output, &expected, 1);
    }

    #[test]
    fn test_quantized_layer_norm_min_max_i8() {
        let input: Vec<i8> = vec![-128, 127, -128, 127];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let expected = scalar_layer_norm_quantized(&input, &gamma, &beta, 0.01, 0.01, EPS);
        let mut output = vec![0i8; 4];
        unsafe { quantized_layer_norm_neon(&input, &mut output, &gamma, &beta, 0.01, 0.01, EPS) };
        assert_i8_approx(&output, &expected, 1);
    }

    #[test]
    fn test_fused_layernorm_quantize_tiny_eps() {
        let input = vec![1.0, 1.0, 1.0, 1.0001];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let output_scale = 0.1;
        let tiny_eps = 1e-12;
        let expected =
            scalar_fused_layernorm_quantize(&input, &gamma, &beta, output_scale, tiny_eps);
        let mut output = vec![0i8; 4];
        unsafe {
            fused_layernorm_quantize_neon(
                &input,
                &mut output,
                &gamma,
                &beta,
                output_scale,
                tiny_eps,
            )
        };
        assert_i8_approx(&output, &expected, 1);
    }

    #[test]
    fn test_fused_layernorm_quantize_large_eps() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let output_scale = 0.1;
        let large_eps = 1.0;
        let expected =
            scalar_fused_layernorm_quantize(&input, &gamma, &beta, output_scale, large_eps);
        let mut output = vec![0i8; 4];
        unsafe {
            fused_layernorm_quantize_neon(
                &input,
                &mut output,
                &gamma,
                &beta,
                output_scale,
                large_eps,
            )
        };
        assert_i8_approx(&output, &expected, 1);
    }

    #[test]
    fn test_apply_scale_bias_large_values() {
        let input = vec![1e6, -1e6, 1e6, -1e6];
        let scale = vec![1e-6; 4];
        let bias = vec![0.0; 4];
        let expected = scalar_apply_scale_bias(&input, &scale, &bias);
        let mut output = vec![0.0; 4];
        unsafe { apply_scale_bias_neon(&input, &mut output, &scale, &bias) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_welford_state_default() {
        let state = WelfordState::default();
        assert_eq!(state.count, 0);
        assert_eq!(state.mean, 0.0);
        assert_eq!(state.variance(), 0.0);
    }

    #[test]
    fn test_eps_typical_value() {
        // Verify our test EPS constant matches the typical default.
        assert!((EPS - 1e-5).abs() < f32::EPSILON);
    }

    // ── Dimension sweep tests ─────────────────────────────────────

    #[test]
    fn test_rms_norm_dimension_sweep() {
        for n in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65] {
            let input: Vec<i8> = (0..n).map(|i| ((i * 11 + 3) % 256) as i8).collect();
            let gamma = vec![1.0; n];
            let expected = scalar_rms_norm_quantized(&input, &gamma, 0.1, 0.1, EPS);
            let mut output = vec![0i8; n];
            unsafe { quantized_rms_norm_neon(&input, &mut output, &gamma, 0.1, 0.1, EPS) };
            assert_i8_approx(&output, &expected, 1);
        }
    }

    #[test]
    fn test_layer_norm_dimension_sweep() {
        for n in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65] {
            let input: Vec<i8> = (0..n).map(|i| ((i * 11 + 3) % 256) as i8).collect();
            let gamma = vec![1.0; n];
            let beta = vec![0.0; n];
            let expected = scalar_layer_norm_quantized(&input, &gamma, &beta, 0.1, 0.1, EPS);
            let mut output = vec![0i8; n];
            unsafe { quantized_layer_norm_neon(&input, &mut output, &gamma, &beta, 0.1, 0.1, EPS) };
            assert_i8_approx(&output, &expected, 1);
        }
    }

    #[test]
    fn test_fused_dimension_sweep() {
        for n in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65] {
            let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - (n as f32) * 0.05).collect();
            let gamma = vec![1.0; n];
            let beta = vec![0.0; n];
            let expected = scalar_fused_layernorm_quantize(&input, &gamma, &beta, 0.1, EPS);
            let mut output = vec![0i8; n];
            unsafe { fused_layernorm_quantize_neon(&input, &mut output, &gamma, &beta, 0.1, EPS) };
            assert_i8_approx(&output, &expected, 1);
        }
    }

    #[test]
    fn test_apply_scale_bias_dimension_sweep() {
        for n in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
            let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
            let scale: Vec<f32> = (0..n).map(|i| 0.8 + (i % 3) as f32 * 0.1).collect();
            let bias: Vec<f32> = (0..n).map(|i| -0.2 + (i % 4) as f32 * 0.1).collect();
            let expected = scalar_apply_scale_bias(&input, &scale, &bias);
            let mut output = vec![0.0; n];
            unsafe { apply_scale_bias_neon(&input, &mut output, &scale, &bias) };
            assert_approx_eq(&output, &expected, TOL);
        }
    }

    #[test]
    fn test_online_variance_dimension_sweep() {
        for n in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
            let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.3 - 1.5).collect();
            let state = unsafe { online_variance_neon(&data) };
            let (exp_mean, exp_var) = scalar_online_variance(&data);
            assert!(
                (state.mean - exp_mean).abs() < 1e-3,
                "n={n}: mean {:.6} vs {:.6}",
                state.mean,
                exp_mean
            );
            assert!(
                (state.variance() - exp_var).abs() < 1e-2,
                "n={n}: var {:.6} vs {:.6}",
                state.variance(),
                exp_var
            );
        }
    }

    // ── Round-trip tests ──────────────────────────────────────────

    #[test]
    fn test_quantize_dequantize_round_trip() {
        // Fused LN+quantize then dequantize should approximate the original LN.
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let output_scale = 0.05;

        let mut quantized = vec![0i8; 8];
        unsafe {
            fused_layernorm_quantize_neon(&input, &mut quantized, &gamma, &beta, output_scale, EPS)
        };

        // Dequantize.
        let dequant: Vec<f32> = quantized.iter().map(|&x| x as f32 * output_scale).collect();

        // Compute expected layernorm output.
        let n = input.len();
        let mean: f32 = input.iter().sum::<f32>() / n as f32;
        let var: f32 = input.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
        let inv_std = 1.0 / (var + EPS).sqrt();
        let expected: Vec<f32> = input
            .iter()
            .enumerate()
            .map(|(i, &x)| gamma[i] * (x - mean) * inv_std + beta[i])
            .collect();

        // Round-trip error should be bounded by output_scale / 2.
        for (i, (&d, &e)) in dequant.iter().zip(expected.iter()).enumerate() {
            assert!(
                (d - e).abs() <= output_scale * 0.6,
                "round-trip error at {i}: {d} vs {e} (diff {})",
                (d - e).abs()
            );
        }
    }

    // ── Additional coverage tests ─────────────────────────────────

    #[test]
    fn test_apply_scale_bias_exact_4() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let scale = vec![2.0, 3.0, 4.0, 5.0];
        let bias = vec![0.1, 0.2, 0.3, 0.4];
        let expected = scalar_apply_scale_bias(&input, &scale, &bias);
        let mut output = vec![0.0; 4];
        unsafe { apply_scale_bias_neon(&input, &mut output, &scale, &bias) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_quantized_rms_norm_symmetric_input() {
        // Symmetric input: [-4, -3, -2, -1, 1, 2, 3, 4].
        let input: Vec<i8> = vec![-40, -30, -20, -10, 10, 20, 30, 40];
        let gamma = vec![1.0; 8];
        let input_scale = 0.1;
        let output_scale = 0.1;
        let expected = scalar_rms_norm_quantized(&input, &gamma, input_scale, output_scale, EPS);
        let mut output = vec![0i8; 8];
        unsafe {
            quantized_rms_norm_neon(&input, &mut output, &gamma, input_scale, output_scale, EPS)
        };
        assert_i8_approx(&output, &expected, 1);
    }

    #[test]
    fn test_quantized_layer_norm_with_large_beta() {
        let input: Vec<i8> = vec![10, 20, 30, 40];
        let gamma = vec![1.0; 4];
        let beta = vec![5.0, 5.0, 5.0, 5.0];
        let input_scale = 0.1;
        let output_scale = 0.1;
        let expected =
            scalar_layer_norm_quantized(&input, &gamma, &beta, input_scale, output_scale, EPS);
        let mut output = vec![0i8; 4];
        unsafe {
            quantized_layer_norm_neon(
                &input,
                &mut output,
                &gamma,
                &beta,
                input_scale,
                output_scale,
                EPS,
            )
        };
        assert_i8_approx(&output, &expected, 1);
    }

    #[test]
    fn test_fused_layernorm_quantize_with_large_gamma() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = vec![5.0; 8];
        let beta = vec![0.0; 8];
        let output_scale = 0.5;
        let expected = scalar_fused_layernorm_quantize(&input, &gamma, &beta, output_scale, EPS);
        let mut output = vec![0i8; 8];
        unsafe {
            fused_layernorm_quantize_neon(&input, &mut output, &gamma, &beta, output_scale, EPS)
        };
        assert_i8_approx(&output, &expected, 1);
    }

    #[test]
    fn test_online_variance_high_magnitude() {
        let data = vec![1e4, 1e4 + 1.0, 1e4 + 2.0, 1e4 + 3.0];
        let state = unsafe { online_variance_neon(&data) };
        let (exp_mean, exp_var) = scalar_online_variance(&data);
        assert!((state.mean - exp_mean).abs() < 1.0);
        assert!((state.variance() - exp_var).abs() < 1.0);
    }

    #[test]
    fn test_quantized_rms_norm_unit_scales() {
        let input: Vec<i8> = vec![1, 2, 3, 4];
        let gamma = vec![1.0; 4];
        let expected = scalar_rms_norm_quantized(&input, &gamma, 1.0, 1.0, EPS);
        let mut output = vec![0i8; 4];
        unsafe { quantized_rms_norm_neon(&input, &mut output, &gamma, 1.0, 1.0, EPS) };
        assert_i8_approx(&output, &expected, 1);
    }

    #[test]
    fn test_quantized_layer_norm_unit_scales() {
        let input: Vec<i8> = vec![1, 2, 3, 4];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let expected = scalar_layer_norm_quantized(&input, &gamma, &beta, 1.0, 1.0, EPS);
        let mut output = vec![0i8; 4];
        unsafe { quantized_layer_norm_neon(&input, &mut output, &gamma, &beta, 1.0, 1.0, EPS) };
        assert_i8_approx(&output, &expected, 1);
    }

    #[test]
    fn test_fused_layernorm_quantize_unit_scale() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let expected = scalar_fused_layernorm_quantize(&input, &gamma, &beta, 1.0, EPS);
        let mut output = vec![0i8; 4];
        unsafe { fused_layernorm_quantize_neon(&input, &mut output, &gamma, &beta, 1.0, EPS) };
        assert_i8_approx(&output, &expected, 1);
    }

    #[test]
    fn test_apply_scale_bias_all_negative_bias() {
        let input = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let scale = vec![1.0; 5];
        let bias = vec![-100.0; 5];
        let expected = scalar_apply_scale_bias(&input, &scale, &bias);
        let mut output = vec![0.0; 5];
        unsafe { apply_scale_bias_neon(&input, &mut output, &scale, &bias) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_welford_state_clone() {
        let state = WelfordState { mean: 3.0, m2: 10.0, count: 5 };
        let cloned = state;
        assert_eq!(cloned.mean, 3.0);
        assert_eq!(cloned.m2, 10.0);
        assert_eq!(cloned.count, 5);
        assert!((cloned.variance() - 2.0).abs() < TOL);
    }
}
