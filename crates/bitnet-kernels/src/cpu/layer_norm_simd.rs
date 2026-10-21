//! SIMD-optimized layer normalization kernels for x86_64 (AVX2).
//!
//! Provides numerically stable LayerNorm and RMSNorm with AVX2
//! acceleration and automatic scalar fallback.  Uses Welford's
//! online algorithm for numerically stable variance computation.
//!
//! All public functions perform runtime feature detection and
//! dispatch to the fastest available path.
#![allow(unsafe_op_in_unsafe_fn)]

use bitnet_common::{BitNetError, KernelError, Result};

// ── Error helper ───────────────────────────────────────────────────

fn invalid_args(reason: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.to_string() })
}

// ── Configuration ──────────────────────────────────────────────────

/// Configuration for SIMD-optimized layer normalization.
#[derive(Debug, Clone)]
pub struct LayerNormSimdConfig {
    /// Shape of the normalized dimensions (product gives the
    /// normalization length per instance).
    pub normalized_shape: Vec<usize>,
    /// Small constant added to variance for numerical stability.
    pub eps: f32,
    /// Whether to apply learnable affine parameters (gamma/beta).
    pub elementwise_affine: bool,
    /// Whether to include bias (beta) when affine is enabled.
    pub bias: bool,
}

impl LayerNormSimdConfig {
    /// Convenience constructor with default eps (1e-5), affine and bias enabled.
    pub fn new(normalized_shape: Vec<usize>) -> Self {
        Self { normalized_shape, eps: 1e-5, elementwise_affine: true, bias: true }
    }

    /// Total number of elements in the normalized dimensions.
    fn norm_size(&self) -> usize {
        self.normalized_shape.iter().product()
    }
}

impl Default for LayerNormSimdConfig {
    fn default() -> Self {
        Self { normalized_shape: vec![1], eps: 1e-5, elementwise_affine: true, bias: true }
    }
}

/// Configuration for SIMD-optimized RMS normalization.
#[derive(Debug, Clone)]
pub struct RMSNormConfig {
    /// Shape of the normalized dimensions.
    pub normalized_shape: Vec<usize>,
    /// Small constant added for numerical stability.
    pub eps: f32,
}

impl RMSNormConfig {
    /// Convenience constructor with default eps (1e-5).
    pub fn new(normalized_shape: Vec<usize>) -> Self {
        Self { normalized_shape, eps: 1e-5 }
    }

    fn norm_size(&self) -> usize {
        self.normalized_shape.iter().product()
    }
}

impl Default for RMSNormConfig {
    fn default() -> Self {
        Self { normalized_shape: vec![1], eps: 1e-5 }
    }
}

// ── Welford's online algorithm ─────────────────────────────────────

/// Welford accumulator for numerically stable mean/variance.
struct WelfordAccumulator {
    count: u64,
    mean: f64,
    m2: f64,
}

impl WelfordAccumulator {
    fn new() -> Self {
        Self { count: 0, mean: 0.0, m2: 0.0 }
    }

    #[inline]
    fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    fn finalize(&self) -> (f32, f32) {
        let mean = self.mean as f32;
        let variance = if self.count > 0 { (self.m2 / self.count as f64) as f32 } else { 0.0 };
        (mean, variance)
    }
}

// ── Scalar implementations ─────────────────────────────────────────

/// Scalar layer normalization using Welford's algorithm.
pub fn layer_norm_f32(
    input: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    config: &LayerNormSimdConfig,
) -> Result<Vec<f32>> {
    let norm_size = validate_layer_norm_args(input, gamma, beta, config)?;
    let batch_size = input.len() / norm_size;
    let mut output = vec![0.0f32; input.len()];

    for b in 0..batch_size {
        let start = b * norm_size;
        let slice = &input[start..start + norm_size];
        let out = &mut output[start..start + norm_size];

        let mut acc = WelfordAccumulator::new();
        for &v in slice {
            acc.update(v as f64);
        }
        let (mean, variance) = acc.finalize();
        let inv_std = 1.0 / (variance + config.eps).sqrt();

        if config.elementwise_affine && config.bias {
            if let Some(beta) = beta {
                for i in 0..norm_size {
                    out[i] = (slice[i] - mean) * inv_std * gamma[i] + beta[i];
                }
            } else {
                for i in 0..norm_size {
                    out[i] = (slice[i] - mean) * inv_std * gamma[i];
                }
            }
        } else if config.elementwise_affine {
            for i in 0..norm_size {
                out[i] = (slice[i] - mean) * inv_std * gamma[i];
            }
        } else {
            for i in 0..norm_size {
                out[i] = (slice[i] - mean) * inv_std;
            }
        }
    }

    Ok(output)
}

/// Scalar RMS normalization using Welford-style accumulation.
pub fn rms_norm_f32(input: &[f32], gamma: &[f32], config: &RMSNormConfig) -> Result<Vec<f32>> {
    let norm_size = validate_rms_norm_args(input, gamma, config)?;
    let batch_size = input.len() / norm_size;
    let mut output = vec![0.0f32; input.len()];

    for b in 0..batch_size {
        let start = b * norm_size;
        let slice = &input[start..start + norm_size];
        let out = &mut output[start..start + norm_size];

        let mut sum_sq = 0.0f64;
        for &v in slice {
            let d = v as f64;
            sum_sq += d * d;
        }
        let rms = (sum_sq / norm_size as f64) as f32;
        let inv_rms = 1.0 / (rms + config.eps).sqrt();

        for i in 0..norm_size {
            out[i] = slice[i] * inv_rms * gamma[i];
        }
    }

    Ok(output)
}

// ── AVX2 implementations ───────────────────────────────────────────

/// AVX2-accelerated layer normalization with runtime detection.
///
/// Falls back to scalar if AVX2 is unavailable at runtime.
pub fn layer_norm_avx2(
    input: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    config: &LayerNormSimdConfig,
) -> Result<Vec<f32>> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // Safety: runtime feature detection passed.
            return unsafe { layer_norm_avx2_inner(input, gamma, beta, config) };
        }
    }
    layer_norm_f32(input, gamma, beta, config)
}

/// AVX2-accelerated RMS normalization with runtime detection.
///
/// Falls back to scalar if AVX2 is unavailable at runtime.
pub fn rms_norm_avx2(input: &[f32], gamma: &[f32], config: &RMSNormConfig) -> Result<Vec<f32>> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // Safety: runtime feature detection passed.
            return unsafe { rms_norm_avx2_inner(input, gamma, config) };
        }
    }
    rms_norm_f32(input, gamma, config)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn layer_norm_avx2_inner(
    input: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    config: &LayerNormSimdConfig,
) -> Result<Vec<f32>> {
    use std::arch::x86_64::*;

    let norm_size = validate_layer_norm_args(input, gamma, beta, config)?;
    let batch_size = input.len() / norm_size;
    let mut output = vec![0.0f32; input.len()];

    let chunks = norm_size / 8;

    for b in 0..batch_size {
        let start = b * norm_size;
        let slice = &input[start..start + norm_size];
        let out = &mut output[start..start + norm_size];

        // Pass 1: mean via SIMD sum + scalar finalize.
        let mut sum_vec = _mm256_setzero_ps();
        for c in 0..chunks {
            let v = _mm256_loadu_ps(slice.as_ptr().add(c * 8));
            sum_vec = _mm256_add_ps(sum_vec, v);
        }
        let mut sum = hsum_avx2(sum_vec) as f64;
        for &v in &slice[(chunks * 8)..norm_size] {
            sum += v as f64;
        }
        let mean = (sum / norm_size as f64) as f32;

        // Pass 2: variance via SIMD.
        let mean_vec = _mm256_set1_ps(mean);
        let mut var_vec = _mm256_setzero_ps();
        for c in 0..chunks {
            let v = _mm256_loadu_ps(slice.as_ptr().add(c * 8));
            let d = _mm256_sub_ps(v, mean_vec);
            var_vec = _mm256_fmadd_ps(d, d, var_vec);
        }
        let mut var_sum = hsum_avx2(var_vec) as f64;
        for &v in &slice[(chunks * 8)..norm_size] {
            let d = v as f64 - mean as f64;
            var_sum += d * d;
        }
        let variance = (var_sum / norm_size as f64) as f32;
        let inv_std = 1.0 / (variance + config.eps).sqrt();
        let inv_std_vec = _mm256_set1_ps(inv_std);

        // Pass 3: normalize + affine.
        if config.elementwise_affine {
            let has_beta = config.bias && beta.is_some();
            for c in 0..chunks {
                let offset = c * 8;
                let v = _mm256_loadu_ps(slice.as_ptr().add(offset));
                let d = _mm256_sub_ps(v, mean_vec);
                let normed = _mm256_mul_ps(d, inv_std_vec);
                let g = _mm256_loadu_ps(gamma.as_ptr().add(offset));
                let mut result = _mm256_mul_ps(normed, g);
                if has_beta {
                    let bt = _mm256_loadu_ps(beta.unwrap().as_ptr().add(offset));
                    result = _mm256_add_ps(result, bt);
                }
                _mm256_storeu_ps(out.as_mut_ptr().add(offset), result);
            }
            for i in (chunks * 8)..norm_size {
                let normed = (slice[i] - mean) * inv_std;
                out[i] =
                    if has_beta { normed * gamma[i] + beta.unwrap()[i] } else { normed * gamma[i] };
            }
        } else {
            for c in 0..chunks {
                let offset = c * 8;
                let v = _mm256_loadu_ps(slice.as_ptr().add(offset));
                let d = _mm256_sub_ps(v, mean_vec);
                let normed = _mm256_mul_ps(d, inv_std_vec);
                _mm256_storeu_ps(out.as_mut_ptr().add(offset), normed);
            }
            for i in (chunks * 8)..norm_size {
                out[i] = (slice[i] - mean) * inv_std;
            }
        }
    }

    Ok(output)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rms_norm_avx2_inner(
    input: &[f32],
    gamma: &[f32],
    config: &RMSNormConfig,
) -> Result<Vec<f32>> {
    use std::arch::x86_64::*;

    let norm_size = validate_rms_norm_args(input, gamma, config)?;
    let batch_size = input.len() / norm_size;
    let mut output = vec![0.0f32; input.len()];

    let chunks = norm_size / 8;

    for b in 0..batch_size {
        let start = b * norm_size;
        let slice = &input[start..start + norm_size];
        let out = &mut output[start..start + norm_size];

        // Sum of squares via SIMD FMA.
        let mut sq_vec = _mm256_setzero_ps();
        for c in 0..chunks {
            let v = _mm256_loadu_ps(slice.as_ptr().add(c * 8));
            sq_vec = _mm256_fmadd_ps(v, v, sq_vec);
        }
        let mut sum_sq = hsum_avx2(sq_vec) as f64;
        for &v in &slice[(chunks * 8)..norm_size] {
            let d = v as f64;
            sum_sq += d * d;
        }
        let rms = (sum_sq / norm_size as f64) as f32;
        let inv_rms = 1.0 / (rms + config.eps).sqrt();
        let inv_rms_vec = _mm256_set1_ps(inv_rms);

        // Scale by gamma.
        for c in 0..chunks {
            let offset = c * 8;
            let v = _mm256_loadu_ps(slice.as_ptr().add(offset));
            let g = _mm256_loadu_ps(gamma.as_ptr().add(offset));
            let scaled = _mm256_mul_ps(v, inv_rms_vec);
            let result = _mm256_mul_ps(scaled, g);
            _mm256_storeu_ps(out.as_mut_ptr().add(offset), result);
        }
        for i in (chunks * 8)..norm_size {
            out[i] = slice[i] * inv_rms * gamma[i];
        }
    }

    Ok(output)
}

/// Horizontal sum of 8 × f32 in an AVX2 register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let sums2 = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(sums2)
}

// ── Group normalization ────────────────────────────────────────────

/// SIMD-aware group normalization.
///
/// Input layout: `[batch, channels, spatial]` flattened.
/// Channels are divided into `num_groups` groups, each independently
/// normalized.
pub fn group_norm(
    input: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    num_groups: usize,
    num_channels: usize,
    spatial_size: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    validate_group_norm_args(input, gamma, beta, num_groups, num_channels, spatial_size, eps)?;

    let cpg = num_channels / num_groups;
    let batch_size = input.len() / (num_channels * spatial_size);
    let mut output = vec![0.0f32; input.len()];

    for b in 0..batch_size {
        for g in 0..num_groups {
            let mut acc = WelfordAccumulator::new();
            for c in (g * cpg)..((g + 1) * cpg) {
                let off = b * num_channels * spatial_size + c * spatial_size;
                for &v in &input[off..off + spatial_size] {
                    acc.update(v as f64);
                }
            }
            let (mean, variance) = acc.finalize();
            let inv_std = 1.0 / (variance + eps).sqrt();

            for c in (g * cpg)..((g + 1) * cpg) {
                let off = b * num_channels * spatial_size + c * spatial_size;
                if let Some(beta) = beta {
                    for s in 0..spatial_size {
                        output[off + s] = (input[off + s] - mean) * inv_std * gamma[c] + beta[c];
                    }
                } else {
                    for s in 0..spatial_size {
                        output[off + s] = (input[off + s] - mean) * inv_std * gamma[c];
                    }
                }
            }
        }
    }

    Ok(output)
}

// ── Instance normalization ─────────────────────────────────────────

/// Instance normalization (group norm with `num_groups == num_channels`).
pub fn instance_norm(
    input: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    num_channels: usize,
    spatial_size: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    group_norm(input, gamma, beta, num_channels, num_channels, spatial_size, eps)
}

// ── Backward pass ──────────────────────────────────────────────────

/// Compute gradients for layer normalization backward pass.
///
/// Returns `(d_input, d_gamma, d_beta)`.
pub fn layer_norm_backward(
    grad_output: &[f32],
    input: &[f32],
    gamma: &[f32],
    config: &LayerNormSimdConfig,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let norm_size = config.norm_size();
    if norm_size == 0 {
        return Err(invalid_args("normalized_shape must have non-zero product"));
    }
    if input.is_empty() || grad_output.is_empty() {
        return Err(invalid_args("input and grad_output must be non-empty"));
    }
    if input.len() != grad_output.len() {
        return Err(invalid_args("input and grad_output must have the same length"));
    }
    if !input.len().is_multiple_of(norm_size) {
        return Err(invalid_args("input length must be a multiple of normalized_shape product"));
    }
    if gamma.len() != norm_size {
        return Err(invalid_args("gamma length must match normalized_shape product"));
    }

    let batch_size = input.len() / norm_size;
    let mut d_input = vec![0.0f32; input.len()];
    let mut d_gamma = vec![0.0f64; norm_size];
    let mut d_beta = vec![0.0f64; norm_size];

    for b in 0..batch_size {
        let start = b * norm_size;
        let x = &input[start..start + norm_size];
        let dy = &grad_output[start..start + norm_size];
        let dx = &mut d_input[start..start + norm_size];

        // Forward stats via Welford.
        let mut acc = WelfordAccumulator::new();
        for &v in x {
            acc.update(v as f64);
        }
        let (mean, variance) = acc.finalize();
        let inv_std = 1.0 / (variance + config.eps).sqrt();
        let n = norm_size as f64;

        // Accumulate d_gamma and d_beta.
        for i in 0..norm_size {
            let x_hat = ((x[i] - mean) * inv_std) as f64;
            d_gamma[i] += dy[i] as f64 * x_hat;
            d_beta[i] += dy[i] as f64;
        }

        // Compute d_input.
        let mut sum_dy_xhat = 0.0f64;
        let mut sum_dy = 0.0f64;
        for i in 0..norm_size {
            let x_hat = ((x[i] - mean) * inv_std) as f64;
            sum_dy_xhat += dy[i] as f64 * gamma[i] as f64 * x_hat;
            sum_dy += dy[i] as f64 * gamma[i] as f64;
        }

        for i in 0..norm_size {
            let x_hat = ((x[i] - mean) * inv_std) as f64;
            let di = (dy[i] as f64 * gamma[i] as f64 - (sum_dy + x_hat * sum_dy_xhat) / n)
                * inv_std as f64;
            dx[i] = di as f32;
        }
    }

    let d_gamma_f32: Vec<f32> = d_gamma.iter().map(|&v| v as f32).collect();
    let d_beta_f32: Vec<f32> = d_beta.iter().map(|&v| v as f32).collect();
    Ok((d_input, d_gamma_f32, d_beta_f32))
}

// ── Fused layer norm + residual ────────────────────────────────────

/// Fused layer normalization with residual addition.
///
/// Computes `LayerNorm(input + residual)`, saving one memory pass.
pub fn fused_layer_norm_residual(
    input: &[f32],
    residual: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    config: &LayerNormSimdConfig,
) -> Result<Vec<f32>> {
    if input.len() != residual.len() {
        return Err(invalid_args("input and residual must have the same length"));
    }
    let norm_size = validate_layer_norm_args(input, gamma, beta, config)?;
    let batch_size = input.len() / norm_size;
    let mut output = vec![0.0f32; input.len()];

    for b in 0..batch_size {
        let start = b * norm_size;
        let out = &mut output[start..start + norm_size];

        // Fused add + Welford stats.
        let mut acc = WelfordAccumulator::new();
        let mut fused = vec![0.0f32; norm_size];
        for i in 0..norm_size {
            fused[i] = input[start + i] + residual[start + i];
            acc.update(fused[i] as f64);
        }
        let (mean, variance) = acc.finalize();
        let inv_std = 1.0 / (variance + config.eps).sqrt();

        if config.elementwise_affine {
            if let Some(beta) = beta {
                for i in 0..norm_size {
                    out[i] = (fused[i] - mean) * inv_std * gamma[i] + beta[i];
                }
            } else {
                for i in 0..norm_size {
                    out[i] = (fused[i] - mean) * inv_std * gamma[i];
                }
            }
        } else {
            for i in 0..norm_size {
                out[i] = (fused[i] - mean) * inv_std;
            }
        }
    }

    Ok(output)
}

// ── FP16 software emulation ────────────────────────────────────────

/// Layer normalization with software FP16 input/output.
///
/// Inputs and outputs are `u16` bit-patterns in IEEE 754 half-precision.
/// Internal computation uses f32 for accuracy.
pub fn layer_norm_fp16(
    input: &[u16],
    gamma: &[f32],
    beta: Option<&[f32]>,
    config: &LayerNormSimdConfig,
) -> Result<Vec<u16>> {
    let norm_size = config.norm_size();
    if norm_size == 0 {
        return Err(invalid_args("normalized_shape must have non-zero product"));
    }
    if input.is_empty() {
        return Err(invalid_args("input must be non-empty"));
    }
    if !input.len().is_multiple_of(norm_size) {
        return Err(invalid_args("input length must be a multiple of normalized_shape product"));
    }
    if config.eps <= 0.0 || !config.eps.is_finite() {
        return Err(invalid_args("eps must be positive and finite"));
    }
    if config.elementwise_affine && gamma.len() != norm_size {
        return Err(invalid_args("gamma length must match normalized_shape product"));
    }
    if let Some(beta) = beta
        && beta.len() != norm_size
    {
        return Err(invalid_args("beta length must match normalized_shape product"));
    }

    // Convert FP16 → f32.
    let input_f32: Vec<f32> = input.iter().map(|&h| fp16_to_f32(h)).collect();
    let result = layer_norm_f32(&input_f32, gamma, beta, config)?;
    // Convert f32 → FP16.
    Ok(result.iter().map(|&v| f32_to_fp16(v)).collect())
}

// ── Batch layer normalization ──────────────────────────────────────

/// Apply SIMD layer normalization to multiple independent inputs.
pub fn batch_layer_norm(
    inputs: &[&[f32]],
    gamma: &[f32],
    beta: Option<&[f32]>,
    config: &LayerNormSimdConfig,
) -> Result<Vec<Vec<f32>>> {
    inputs.iter().map(|inp| layer_norm_avx2(inp, gamma, beta, config)).collect()
}

// ── FP16 conversion helpers ────────────────────────────────────────

fn fp16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let frac = (h & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal: normalize.
        let mut e = 0i32;
        let mut f = frac;
        while (f & 0x400) == 0 {
            f <<= 1;
            e -= 1;
        }
        f &= 0x3FF;
        let exp32 = (127 - 15 + 1 + e) as u32;
        let bits = (sign << 31) | (exp32 << 23) | (f << 13);
        return f32::from_bits(bits);
    }
    if exp == 31 {
        let bits = (sign << 31) | (0xFF << 23) | (frac << 13);
        return f32::from_bits(bits);
    }
    let exp32 = exp + 127 - 15;
    let bits = (sign << 31) | (exp32 << 23) | (frac << 13);
    f32::from_bits(bits)
}

fn f32_to_fp16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7F_FFFF;

    if exp == 0xFF {
        // Inf or NaN.
        let h_frac = if frac != 0 { 0x200 } else { 0 };
        return (sign << 15) | 0x7C00 | h_frac;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        // Overflow → Inf.
        return (sign << 15) | 0x7C00;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign << 15;
        }
        // Subnormal.
        let mant = frac | 0x80_0000;
        let shift = (1 - new_exp) as u32 + 13;
        let h_frac = (mant >> shift) as u16;
        return (sign << 15) | h_frac;
    }
    let h_exp = (new_exp as u16) << 10;
    let h_frac = (frac >> 13) as u16;
    (sign << 15) | h_exp | h_frac
}

// ── Validation ─────────────────────────────────────────────────────

fn validate_common_config(norm_size: usize, input: &[f32], eps: f32) -> Result<()> {
    if norm_size == 0 {
        return Err(invalid_args("normalized_shape must have non-zero product"));
    }
    if input.is_empty() {
        return Err(invalid_args("input must be non-empty"));
    }
    if !input.len().is_multiple_of(norm_size) {
        return Err(invalid_args("input length must be a multiple of normalized_shape product"));
    }
    if eps <= 0.0 || !eps.is_finite() {
        return Err(invalid_args("eps must be positive and finite"));
    }
    Ok(())
}

fn validate_layer_norm_args(
    input: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    config: &LayerNormSimdConfig,
) -> Result<usize> {
    let norm_size = config.norm_size();
    validate_common_config(norm_size, input, config.eps)?;
    if config.elementwise_affine {
        if gamma.len() != norm_size {
            return Err(invalid_args(&format!(
                "gamma length {} != normalized_shape product {norm_size}",
                gamma.len(),
            )));
        }
        if let Some(beta) = beta
            && beta.len() != norm_size
        {
            return Err(invalid_args(&format!(
                "beta length {} != normalized_shape product {norm_size}",
                beta.len(),
            )));
        }
    }
    Ok(norm_size)
}

fn validate_rms_norm_args(input: &[f32], gamma: &[f32], config: &RMSNormConfig) -> Result<usize> {
    let norm_size = config.norm_size();
    if norm_size == 0 {
        return Err(invalid_args("normalized_shape must have non-zero product"));
    }
    if input.is_empty() {
        return Err(invalid_args("input must be non-empty"));
    }
    if !input.len().is_multiple_of(norm_size) {
        return Err(invalid_args("input length must be a multiple of normalized_shape product"));
    }
    if config.eps <= 0.0 || !config.eps.is_finite() {
        return Err(invalid_args("eps must be positive and finite"));
    }
    if gamma.len() != norm_size {
        return Err(invalid_args(&format!(
            "gamma length {} != normalized_shape product {norm_size}",
            gamma.len(),
        )));
    }
    Ok(norm_size)
}

fn validate_group_norm_args(
    input: &[f32],
    gamma: &[f32],
    beta: Option<&[f32]>,
    num_groups: usize,
    num_channels: usize,
    spatial_size: usize,
    eps: f32,
) -> Result<()> {
    if num_groups == 0 || num_channels == 0 || spatial_size == 0 {
        return Err(invalid_args("num_groups, num_channels, and spatial_size must be non-zero"));
    }
    if !num_channels.is_multiple_of(num_groups) {
        return Err(invalid_args("num_channels must be divisible by num_groups"));
    }
    if input.is_empty() {
        return Err(invalid_args("input must be non-empty"));
    }
    let frame = num_channels * spatial_size;
    if !input.len().is_multiple_of(frame) {
        return Err(invalid_args("input length must be a multiple of num_channels * spatial_size"));
    }
    if eps <= 0.0 || !eps.is_finite() {
        return Err(invalid_args("eps must be positive and finite"));
    }
    if gamma.len() != num_channels {
        return Err(invalid_args(&format!(
            "gamma length {} != num_channels {num_channels}",
            gamma.len(),
        )));
    }
    if let Some(beta) = beta
        && beta.len() != num_channels
    {
        return Err(invalid_args(&format!(
            "beta length {} != num_channels {num_channels}",
            beta.len(),
        )));
    }
    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;
    const LOOSE_TOL: f32 = 1e-3;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    /// Reference layer norm (pure f64 for verification).
    fn reference_layer_norm(
        input: &[f32],
        gamma: &[f32],
        beta: Option<&[f32]>,
        eps: f32,
    ) -> Vec<f32> {
        let n = gamma.len();
        let batch = input.len() / n;
        let mut out = vec![0.0f32; input.len()];
        for b in 0..batch {
            let s = &input[b * n..(b + 1) * n];
            let mean: f64 = s.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
            let var: f64 = s
                .iter()
                .map(|&x| {
                    let d = x as f64 - mean;
                    d * d
                })
                .sum::<f64>()
                / n as f64;
            let inv_std = 1.0 / (var + eps as f64).sqrt();
            for i in 0..n {
                let normed = (s[i] as f64 - mean) * inv_std;
                let val = normed * gamma[i] as f64 + beta.map_or(0.0, |b| b[i] as f64);
                out[b * n + i] = val as f32;
            }
        }
        out
    }

    /// Reference RMS norm (pure f64 for verification).
    fn reference_rms_norm(input: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        let n = gamma.len();
        let batch = input.len() / n;
        let mut out = vec![0.0f32; input.len()];
        for b in 0..batch {
            let s = &input[b * n..(b + 1) * n];
            let rms: f64 = s
                .iter()
                .map(|&x| {
                    let v = x as f64;
                    v * v
                })
                .sum::<f64>()
                / n as f64;
            let inv_rms = 1.0 / (rms + eps as f64).sqrt();
            for i in 0..n {
                out[b * n + i] = (s[i] as f64 * inv_rms * gamma[i] as f64) as f32;
            }
        }
        out
    }

    // ── LayerNormSimdConfig tests ──────────────────────────

    #[test]
    fn config_default() {
        let c = LayerNormSimdConfig::default();
        assert_eq!(c.normalized_shape, vec![1]);
        assert!((c.eps - 1e-5).abs() < 1e-10);
        assert!(c.elementwise_affine);
        assert!(c.bias);
    }

    #[test]
    fn config_new() {
        let c = LayerNormSimdConfig::new(vec![64]);
        assert_eq!(c.normalized_shape, vec![64]);
        assert!(c.elementwise_affine);
        assert!(c.bias);
    }

    #[test]
    fn rms_config_default() {
        let c = RMSNormConfig::default();
        assert_eq!(c.normalized_shape, vec![1]);
        assert!((c.eps - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn rms_config_new() {
        let c = RMSNormConfig::new(vec![128]);
        assert_eq!(c.normalized_shape, vec![128]);
    }

    // ── Scalar layer norm: basic ───────────────────────────

    #[test]
    fn scalar_ln_uniform_input() {
        let input = vec![2.0; 4];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let config = LayerNormSimdConfig::new(vec![4]);
        let out = layer_norm_f32(&input, &gamma, Some(&beta), &config).unwrap();
        for &v in &out {
            assert!(v.abs() < TOL, "expected ~0, got {v}");
        }
    }

    #[test]
    fn scalar_ln_known_values() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gamma = vec![1.0; 5];
        let beta = vec![0.0; 5];
        let config = LayerNormSimdConfig::new(vec![5]);
        let out = layer_norm_f32(&input, &gamma, Some(&beta), &config).unwrap();
        let expected = reference_layer_norm(&input, &gamma, Some(&beta), 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn scalar_ln_with_affine() {
        let input = vec![1.0, 2.0, 3.0];
        let gamma = vec![2.0, 0.5, 1.0];
        let beta = vec![1.0, -1.0, 0.0];
        let config = LayerNormSimdConfig::new(vec![3]);
        let out = layer_norm_f32(&input, &gamma, Some(&beta), &config).unwrap();
        let expected = reference_layer_norm(&input, &gamma, Some(&beta), 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn scalar_ln_no_beta() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let config = LayerNormSimdConfig::new(vec![4]);
        let out = layer_norm_f32(&input, &gamma, None, &config).unwrap();
        let expected = reference_layer_norm(&input, &gamma, None, 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn scalar_ln_affine_disabled() {
        let input = vec![1.0, 3.0, 5.0];
        let gamma = vec![999.0; 3];
        let mut config = LayerNormSimdConfig::new(vec![3]);
        config.elementwise_affine = false;
        let out = layer_norm_f32(&input, &gamma, None, &config).unwrap();
        let ones = vec![1.0; 3];
        let expected = reference_layer_norm(&input, &ones, None, 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn scalar_ln_bias_disabled() {
        let input = vec![1.0, 2.0, 3.0];
        let gamma = vec![2.0, 0.5, 1.0];
        let beta = vec![1.0, -1.0, 0.0];
        let mut config = LayerNormSimdConfig::new(vec![3]);
        config.bias = false;
        let out = layer_norm_f32(&input, &gamma, Some(&beta), &config).unwrap();
        // With bias=false, beta is ignored.
        let expected = reference_layer_norm(&input, &gamma, None, 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn scalar_ln_output_zero_mean() {
        let input = vec![10.0, 20.0, 30.0, 40.0];
        let gamma = vec![1.0; 4];
        let config = LayerNormSimdConfig::new(vec![4]);
        let out = layer_norm_f32(&input, &gamma, None, &config).unwrap();
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        assert!(mean.abs() < TOL, "mean should be ~0, got {mean}");
    }

    #[test]
    fn scalar_ln_output_unit_variance() {
        let input: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
        let gamma = vec![1.0; 128];
        let config = LayerNormSimdConfig::new(vec![128]);
        let out = layer_norm_f32(&input, &gamma, None, &config).unwrap();
        let mean = out.iter().sum::<f32>() / 128.0;
        let var = out.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / 128.0;
        assert!((var - 1.0).abs() < 0.01, "variance should be ~1, got {var}");
    }

    #[test]
    fn scalar_ln_negative_inputs() {
        let input = vec![-5.0, -3.0, -1.0, 1.0, 3.0, 5.0];
        let gamma = vec![1.0; 6];
        let beta = vec![0.0; 6];
        let config = LayerNormSimdConfig::new(vec![6]);
        let out = layer_norm_f32(&input, &gamma, Some(&beta), &config).unwrap();
        let expected = reference_layer_norm(&input, &gamma, Some(&beta), 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    // ── Numerical stability (Welford) ──────────────────────

    #[test]
    fn scalar_ln_large_values() {
        let input = vec![1e6, 1e6 + 1.0, 1e6 + 2.0];
        let gamma = vec![1.0; 3];
        let config = LayerNormSimdConfig::new(vec![3]);
        let out = layer_norm_f32(&input, &gamma, None, &config).unwrap();
        let expected = reference_layer_norm(&input, &gamma, None, 1e-5);
        assert!(
            approx_eq(&out, &expected, LOOSE_TOL),
            "large values: out={out:?}, expected={expected:?}",
        );
    }

    #[test]
    fn scalar_ln_tiny_variance() {
        let input = vec![1.0, 1.0 + 1e-7, 1.0 - 1e-7];
        let gamma = vec![1.0; 3];
        let config = LayerNormSimdConfig::new(vec![3]);
        let out = layer_norm_f32(&input, &gamma, None, &config).unwrap();
        for &v in &out {
            assert!(v.is_finite(), "output should be finite, got {v}");
        }
    }

    #[test]
    fn scalar_ln_no_nan_or_inf() {
        let input = vec![1e10, -1e10, 0.0, 1e-10];
        let gamma = vec![1.0; 4];
        let config = LayerNormSimdConfig::new(vec![4]);
        let out = layer_norm_f32(&input, &gamma, None, &config).unwrap();
        for &v in &out {
            assert!(v.is_finite(), "output must be finite, got {v}");
        }
    }

    #[test]
    fn welford_accumulator_basic() {
        let mut acc = WelfordAccumulator::new();
        for &v in &[1.0, 2.0, 3.0, 4.0, 5.0f64] {
            acc.update(v);
        }
        let (mean, var) = acc.finalize();
        assert!((mean - 3.0).abs() < TOL, "mean={mean}");
        assert!((var - 2.0).abs() < TOL, "var={var}");
    }

    #[test]
    fn welford_accumulator_single() {
        let mut acc = WelfordAccumulator::new();
        acc.update(42.0);
        let (mean, var) = acc.finalize();
        assert!((mean - 42.0).abs() < TOL);
        assert!(var.abs() < TOL);
    }

    #[test]
    fn welford_accumulator_empty() {
        let acc = WelfordAccumulator::new();
        let (_, var) = acc.finalize();
        assert_eq!(var, 0.0);
    }

    // ── Eps variations ─────────────────────────────────────

    #[test]
    fn scalar_ln_custom_eps() {
        let input = vec![1.0, 2.0, 3.0];
        let gamma = vec![1.0; 3];
        let mut config = LayerNormSimdConfig::new(vec![3]);
        config.eps = 1e-3;
        let out = layer_norm_f32(&input, &gamma, None, &config).unwrap();
        let expected = reference_layer_norm(&input, &gamma, None, 1e-3);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn scalar_ln_large_eps() {
        let input = vec![1.0, 2.0, 3.0];
        let gamma = vec![1.0; 3];
        let mut config = LayerNormSimdConfig::new(vec![3]);
        config.eps = 1.0;
        let out = layer_norm_f32(&input, &gamma, None, &config).unwrap();
        let expected = reference_layer_norm(&input, &gamma, None, 1.0);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn scalar_ln_tiny_eps() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let mut config = LayerNormSimdConfig::new(vec![4]);
        config.eps = 1e-12;
        let out = layer_norm_f32(&input, &gamma, None, &config).unwrap();
        let expected = reference_layer_norm(&input, &gamma, None, 1e-12);
        assert!(approx_eq(&out, &expected, TOL));
    }

    // ── Batched inputs ─────────────────────────────────────

    #[test]
    fn scalar_ln_batch_two() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gamma = vec![1.0; 3];
        let beta = vec![0.0; 3];
        let config = LayerNormSimdConfig::new(vec![3]);
        let out = layer_norm_f32(&input, &gamma, Some(&beta), &config).unwrap();
        let expected = reference_layer_norm(&input, &gamma, Some(&beta), 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn scalar_ln_batch_independence() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![10.0, 20.0, 30.0];
        let gamma = vec![1.0; 3];
        let config = LayerNormSimdConfig::new(vec![3]);

        let out_a = layer_norm_f32(&a, &gamma, None, &config).unwrap();
        let out_b = layer_norm_f32(&b, &gamma, None, &config).unwrap();

        let combined: Vec<f32> = a.iter().chain(b.iter()).copied().collect();
        let out_combined = layer_norm_f32(&combined, &gamma, None, &config).unwrap();

        assert!(approx_eq(&out_combined[..3], &out_a, TOL));
        assert!(approx_eq(&out_combined[3..], &out_b, TOL));
    }

    #[test]
    fn scalar_ln_batch_four() {
        let input: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let gamma = vec![1.0; 5];
        let beta = vec![0.5; 5];
        let config = LayerNormSimdConfig::new(vec![5]);
        let out = layer_norm_f32(&input, &gamma, Some(&beta), &config).unwrap();
        let expected = reference_layer_norm(&input, &gamma, Some(&beta), 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn scalar_ln_2d_normalized_shape() {
        let input: Vec<f32> = (1..=12).map(|i| i as f32).collect();
        let gamma = vec![1.0; 6];
        let beta = vec![0.0; 6];
        let config = LayerNormSimdConfig::new(vec![2, 3]);
        let out = layer_norm_f32(&input, &gamma, Some(&beta), &config).unwrap();
        let expected = reference_layer_norm(&input, &gamma, Some(&beta), 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    // ── Scalar RMS norm ────────────────────────────────────

    #[test]
    fn scalar_rms_known_values() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let config = RMSNormConfig::new(vec![4]);
        let out = rms_norm_f32(&input, &gamma, &config).unwrap();
        let expected = reference_rms_norm(&input, &gamma, 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn scalar_rms_with_gamma() {
        let input = vec![1.0, 2.0, 3.0];
        let gamma = vec![2.0, 0.5, 1.0];
        let config = RMSNormConfig::new(vec![3]);
        let out = rms_norm_f32(&input, &gamma, &config).unwrap();
        let expected = reference_rms_norm(&input, &gamma, 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn scalar_rms_uniform() {
        let input = vec![3.0; 4];
        let gamma = vec![1.0; 4];
        let config = RMSNormConfig::new(vec![4]);
        let out = rms_norm_f32(&input, &gamma, &config).unwrap();
        for &v in &out {
            assert!((v - 1.0).abs() < 0.01, "expected ~1.0, got {v}");
        }
    }

    #[test]
    fn scalar_rms_batch() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gamma = vec![1.0; 3];
        let config = RMSNormConfig::new(vec![3]);
        let out = rms_norm_f32(&input, &gamma, &config).unwrap();
        let expected = reference_rms_norm(&input, &gamma, 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn scalar_rms_batch_independence() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![10.0, 20.0, 30.0];
        let gamma = vec![1.0; 3];
        let config = RMSNormConfig::new(vec![3]);

        let out_a = rms_norm_f32(&a, &gamma, &config).unwrap();
        let out_b = rms_norm_f32(&b, &gamma, &config).unwrap();

        let combined: Vec<f32> = a.iter().chain(b.iter()).copied().collect();
        let out_combined = rms_norm_f32(&combined, &gamma, &config).unwrap();

        assert!(approx_eq(&out_combined[..3], &out_a, TOL));
        assert!(approx_eq(&out_combined[3..], &out_b, TOL));
    }

    #[test]
    fn scalar_rms_no_nan_or_inf() {
        let input = vec![1e10, -1e10, 0.0];
        let gamma = vec![1.0; 3];
        let config = RMSNormConfig::new(vec![3]);
        let out = rms_norm_f32(&input, &gamma, &config).unwrap();
        for &v in &out {
            assert!(v.is_finite(), "output must be finite, got {v}");
        }
    }

    #[test]
    fn scalar_rms_custom_eps() {
        let input = vec![1.0, 2.0, 3.0];
        let gamma = vec![1.0; 3];
        let config = RMSNormConfig { normalized_shape: vec![3], eps: 0.1 };
        let out = rms_norm_f32(&input, &gamma, &config).unwrap();
        let expected = reference_rms_norm(&input, &gamma, 0.1);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn scalar_ln_and_rms_differ() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let ln_config = LayerNormSimdConfig::new(vec![4]);
        let rms_config = RMSNormConfig::new(vec![4]);
        let ln = layer_norm_f32(&input, &gamma, None, &ln_config).unwrap();
        let rms = rms_norm_f32(&input, &gamma, &rms_config).unwrap();
        assert!(!approx_eq(&ln, &rms, TOL), "LN and RMS norm should differ");
    }

    // ── AVX2 vs scalar parity ──────────────────────────────

    #[test]
    fn avx2_ln_matches_scalar() {
        let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.1 - 3.2).collect();
        let gamma = vec![1.0; 64];
        let beta = vec![0.0; 64];
        let config = LayerNormSimdConfig::new(vec![64]);

        let scalar = layer_norm_f32(&input, &gamma, Some(&beta), &config).unwrap();
        let avx = layer_norm_avx2(&input, &gamma, Some(&beta), &config).unwrap();
        assert!(approx_eq(&scalar, &avx, TOL), "AVX2 layer norm diverged from scalar");
    }

    #[test]
    fn avx2_ln_matches_scalar_with_affine() {
        let input: Vec<f32> = (0..32).map(|i| (i as f32).sin()).collect();
        let gamma: Vec<f32> = (0..32).map(|i| 0.5 + i as f32 * 0.1).collect();
        let beta: Vec<f32> = (0..32).map(|i| -1.0 + i as f32 * 0.05).collect();
        let config = LayerNormSimdConfig::new(vec![32]);

        let scalar = layer_norm_f32(&input, &gamma, Some(&beta), &config).unwrap();
        let avx = layer_norm_avx2(&input, &gamma, Some(&beta), &config).unwrap();
        assert!(approx_eq(&scalar, &avx, TOL), "AVX2 layer norm with affine diverged from scalar");
    }

    #[test]
    fn avx2_ln_matches_scalar_no_affine() {
        let input: Vec<f32> = (0..48).map(|i| (i as f32 * 0.3).cos()).collect();
        let gamma = vec![1.0; 48];
        let mut config = LayerNormSimdConfig::new(vec![48]);
        config.elementwise_affine = false;

        let scalar = layer_norm_f32(&input, &gamma, None, &config).unwrap();
        let avx = layer_norm_avx2(&input, &gamma, None, &config).unwrap();
        assert!(approx_eq(&scalar, &avx, TOL), "AVX2 LN no-affine diverged from scalar");
    }

    #[test]
    fn avx2_ln_remainder_elements() {
        // 13 elements: 1 chunk of 8 + 5 remainder
        let input: Vec<f32> = (0..13).map(|i| i as f32).collect();
        let gamma = vec![1.0; 13];
        let beta = vec![0.0; 13];
        let config = LayerNormSimdConfig::new(vec![13]);

        let scalar = layer_norm_f32(&input, &gamma, Some(&beta), &config).unwrap();
        let avx = layer_norm_avx2(&input, &gamma, Some(&beta), &config).unwrap();
        assert!(approx_eq(&scalar, &avx, TOL));
    }

    #[test]
    fn avx2_ln_small_input() {
        // Fewer than 8 elements: all scalar remainder.
        let input = vec![1.0, 2.0, 3.0];
        let gamma = vec![1.0; 3];
        let config = LayerNormSimdConfig::new(vec![3]);

        let scalar = layer_norm_f32(&input, &gamma, None, &config).unwrap();
        let avx = layer_norm_avx2(&input, &gamma, None, &config).unwrap();
        assert!(approx_eq(&scalar, &avx, TOL));
    }

    #[test]
    fn avx2_ln_batched() {
        let input: Vec<f32> = (0..96).map(|i| (i as f32 * 0.7).sin()).collect();
        let gamma = vec![1.0; 32];
        let beta = vec![0.5; 32];
        let config = LayerNormSimdConfig::new(vec![32]);

        let scalar = layer_norm_f32(&input, &gamma, Some(&beta), &config).unwrap();
        let avx = layer_norm_avx2(&input, &gamma, Some(&beta), &config).unwrap();
        assert!(approx_eq(&scalar, &avx, TOL));
    }

    #[test]
    fn avx2_rms_matches_scalar() {
        let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.1 - 3.2).collect();
        let gamma = vec![1.0; 64];
        let config = RMSNormConfig::new(vec![64]);

        let scalar = rms_norm_f32(&input, &gamma, &config).unwrap();
        let avx = rms_norm_avx2(&input, &gamma, &config).unwrap();
        assert!(approx_eq(&scalar, &avx, TOL), "AVX2 RMS norm diverged from scalar");
    }

    #[test]
    fn avx2_rms_matches_scalar_with_gamma() {
        let input: Vec<f32> = (0..32).map(|i| (i as f32).sin()).collect();
        let gamma: Vec<f32> = (0..32).map(|i| 0.5 + i as f32 * 0.1).collect();
        let config = RMSNormConfig::new(vec![32]);

        let scalar = rms_norm_f32(&input, &gamma, &config).unwrap();
        let avx = rms_norm_avx2(&input, &gamma, &config).unwrap();
        assert!(approx_eq(&scalar, &avx, TOL));
    }

    #[test]
    fn avx2_rms_remainder_elements() {
        let input: Vec<f32> = (0..17).map(|i| i as f32 * 0.2).collect();
        let gamma = vec![1.0; 17];
        let config = RMSNormConfig::new(vec![17]);

        let scalar = rms_norm_f32(&input, &gamma, &config).unwrap();
        let avx = rms_norm_avx2(&input, &gamma, &config).unwrap();
        assert!(approx_eq(&scalar, &avx, TOL));
    }

    #[test]
    fn avx2_rms_batched() {
        let input: Vec<f32> = (0..96).map(|i| (i as f32 * 0.5).cos()).collect();
        let gamma = vec![1.0; 32];
        let config = RMSNormConfig::new(vec![32]);

        let scalar = rms_norm_f32(&input, &gamma, &config).unwrap();
        let avx = rms_norm_avx2(&input, &gamma, &config).unwrap();
        assert!(approx_eq(&scalar, &avx, TOL));
    }

    #[test]
    fn avx2_ln_large_values() {
        let input =
            vec![1e6, 1e6 + 1.0, 1e6 + 2.0, 1e6 + 3.0, 1e6 + 4.0, 1e6 + 5.0, 1e6 + 6.0, 1e6 + 7.0];
        let gamma = vec![1.0; 8];
        let config = LayerNormSimdConfig::new(vec![8]);

        let scalar = layer_norm_f32(&input, &gamma, None, &config).unwrap();
        let avx = layer_norm_avx2(&input, &gamma, None, &config).unwrap();
        assert!(
            approx_eq(&scalar, &avx, LOOSE_TOL),
            "AVX2 large values: scalar={scalar:?}, avx={avx:?}",
        );
    }

    // ── Group normalization ────────────────────────────────

    #[test]
    fn group_norm_basic() {
        let input: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let out = group_norm(&input, &gamma, Some(&beta), 2, 4, 3, 1e-5).unwrap();
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn group_norm_with_affine() {
        let input: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
        let gamma = vec![2.0, 0.5, 1.0, 3.0];
        let beta = vec![1.0, -1.0, 0.5, 0.0];
        let out = group_norm(&input, &gamma, Some(&beta), 2, 4, 2, 1e-5).unwrap();
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn group_norm_no_beta() {
        let input: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let gamma = vec![1.0; 2];
        let out = group_norm(&input, &gamma, None, 1, 2, 3, 1e-5).unwrap();
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn group_norm_uniform_within_group() {
        let input = vec![5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0, 3.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let out = group_norm(&input, &gamma, Some(&beta), 2, 4, 2, 1e-5).unwrap();
        for &v in &out {
            assert!(v.abs() < TOL, "expected ~0, got {v}");
        }
    }

    #[test]
    fn group_norm_batch_two() {
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let gamma = vec![1.0; 4];
        let out = group_norm(&input, &gamma, None, 2, 4, 2, 1e-5).unwrap();
        assert_eq!(out.len(), 16);
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    // ── Group norm errors ──────────────────────────────────

    #[test]
    fn group_norm_empty_error() {
        assert!(group_norm(&[], &[1.0; 2], None, 1, 2, 3, 1e-5).is_err());
    }

    #[test]
    fn group_norm_channels_not_divisible_error() {
        let input = vec![1.0; 6];
        let gamma = vec![1.0; 3];
        assert!(group_norm(&input, &gamma, None, 2, 3, 2, 1e-5).is_err());
    }

    #[test]
    fn group_norm_gamma_mismatch_error() {
        let input = vec![1.0; 8];
        let gamma = vec![1.0; 3];
        assert!(group_norm(&input, &gamma, None, 2, 4, 2, 1e-5).is_err());
    }

    #[test]
    fn group_norm_beta_mismatch_error() {
        let input = vec![1.0; 8];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 3];
        assert!(group_norm(&input, &gamma, Some(&beta), 2, 4, 2, 1e-5).is_err());
    }

    #[test]
    fn group_norm_zero_eps_error() {
        let input = vec![1.0; 4];
        let gamma = vec![1.0; 2];
        assert!(group_norm(&input, &gamma, None, 1, 2, 2, 0.0).is_err());
    }

    // ── Instance normalization ─────────────────────────────

    #[test]
    fn instance_norm_basic() {
        let input: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let gamma = vec![1.0; 3];
        let out = instance_norm(&input, &gamma, None, 3, 4, 1e-5).unwrap();
        assert_eq!(out.len(), 12);
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn instance_norm_uniform_channel() {
        let input = vec![7.0, 7.0, 7.0, 3.0, 3.0, 3.0];
        let gamma = vec![1.0; 2];
        let beta = vec![0.0; 2];
        let out = instance_norm(&input, &gamma, Some(&beta), 2, 3, 1e-5).unwrap();
        for &v in &out {
            assert!(v.abs() < TOL, "expected ~0, got {v}");
        }
    }

    // ── Backward pass ──────────────────────────────────────

    #[test]
    fn backward_gradient_shapes() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let grad = vec![0.1, 0.2, 0.3, 0.4];
        let gamma = vec![1.0; 4];
        let config = LayerNormSimdConfig::new(vec![4]);

        let (dx, dg, db) = layer_norm_backward(&grad, &input, &gamma, &config).unwrap();
        assert_eq!(dx.len(), 4);
        assert_eq!(dg.len(), 4);
        assert_eq!(db.len(), 4);
    }

    #[test]
    fn backward_d_beta_is_sum_of_grads() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let grad = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let gamma = vec![1.0; 3];
        let config = LayerNormSimdConfig::new(vec![3]);

        let (_, _, db) = layer_norm_backward(&grad, &input, &gamma, &config).unwrap();
        // d_beta[i] = sum over batch of grad[b*n + i]
        assert!((db[0] - 0.5).abs() < TOL); // 0.1 + 0.4
        assert!((db[1] - 0.7).abs() < TOL); // 0.2 + 0.5
        assert!((db[2] - 0.9).abs() < TOL); // 0.3 + 0.6
    }

    #[test]
    fn backward_uniform_input_zero_d_input() {
        let input = vec![5.0; 4];
        let grad = vec![1.0; 4];
        let gamma = vec![1.0; 4];
        let config = LayerNormSimdConfig::new(vec![4]);

        let (dx, _, _) = layer_norm_backward(&grad, &input, &gamma, &config).unwrap();
        for &v in &dx {
            assert!(v.abs() < LOOSE_TOL, "expected ~0, got {v}");
        }
    }

    #[test]
    fn backward_length_mismatch_error() {
        let input = vec![1.0, 2.0, 3.0];
        let grad = vec![0.1, 0.2];
        let gamma = vec![1.0; 3];
        let config = LayerNormSimdConfig::new(vec![3]);
        assert!(layer_norm_backward(&grad, &input, &gamma, &config).is_err());
    }

    #[test]
    fn backward_empty_error() {
        let gamma = vec![1.0; 3];
        let config = LayerNormSimdConfig::new(vec![3]);
        assert!(layer_norm_backward(&[], &[], &gamma, &config).is_err());
    }

    #[test]
    fn backward_finite_outputs() {
        let input: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let grad: Vec<f32> = (0..16).map(|i| (i as f32 * 0.1).sin()).collect();
        let gamma = vec![1.0; 16];
        let config = LayerNormSimdConfig::new(vec![16]);

        let (dx, dg, db) = layer_norm_backward(&grad, &input, &gamma, &config).unwrap();
        for &v in dx.iter().chain(dg.iter()).chain(db.iter()) {
            assert!(v.is_finite(), "gradient must be finite, got {v}");
        }
    }

    // ── Fused layer norm + residual ────────────────────────

    #[test]
    fn fused_residual_matches_manual() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![0.5, -0.5, 1.0, -1.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let config = LayerNormSimdConfig::new(vec![4]);

        let fused =
            fused_layer_norm_residual(&input, &residual, &gamma, Some(&beta), &config).unwrap();

        let added: Vec<f32> = input.iter().zip(&residual).map(|(a, b)| a + b).collect();
        let manual = layer_norm_f32(&added, &gamma, Some(&beta), &config).unwrap();

        assert!(approx_eq(&fused, &manual, TOL));
    }

    #[test]
    fn fused_residual_batched() {
        let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let residual: Vec<f32> = (0..8).map(|i| -(i as f32) * 0.5).collect();
        let gamma = vec![1.0; 4];
        let config = LayerNormSimdConfig::new(vec![4]);

        let fused = fused_layer_norm_residual(&input, &residual, &gamma, None, &config).unwrap();
        assert_eq!(fused.len(), 8);
        for &v in &fused {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn fused_residual_length_mismatch_error() {
        let input = vec![1.0, 2.0, 3.0];
        let residual = vec![0.5, -0.5];
        let gamma = vec![1.0; 3];
        let config = LayerNormSimdConfig::new(vec![3]);
        assert!(fused_layer_norm_residual(&input, &residual, &gamma, None, &config).is_err());
    }

    #[test]
    fn fused_residual_with_affine() {
        let input = vec![1.0, 2.0, 3.0];
        let residual = vec![0.5, 0.5, 0.5];
        let gamma = vec![2.0, 0.5, 1.0];
        let beta = vec![1.0, -1.0, 0.0];
        let config = LayerNormSimdConfig::new(vec![3]);

        let fused =
            fused_layer_norm_residual(&input, &residual, &gamma, Some(&beta), &config).unwrap();

        let added: Vec<f32> = input.iter().zip(&residual).map(|(a, b)| a + b).collect();
        let manual = layer_norm_f32(&added, &gamma, Some(&beta), &config).unwrap();
        assert!(approx_eq(&fused, &manual, TOL));
    }

    // ── FP16 software layer norm ───────────────────────────

    #[test]
    fn fp16_roundtrip_normal() {
        let vals = [0.0f32, 1.0, -1.0, 0.5, 65504.0, -65504.0, 0.00006103515625];
        for &v in &vals {
            let h = f32_to_fp16(v);
            let back = fp16_to_f32(h);
            assert!(
                (back - v).abs() < v.abs() * 0.002 + 1e-7,
                "fp16 roundtrip failed for {v}: got {back}",
            );
        }
    }

    #[test]
    fn fp16_inf_nan() {
        let h_inf = f32_to_fp16(f32::INFINITY);
        assert!(fp16_to_f32(h_inf).is_infinite());

        let h_nan = f32_to_fp16(f32::NAN);
        assert!(fp16_to_f32(h_nan).is_nan());
    }

    #[test]
    fn fp16_layer_norm_basic() {
        let input_f32 = vec![1.0f32, 2.0, 3.0, 4.0];
        let input_fp16: Vec<u16> = input_f32.iter().map(|&v| f32_to_fp16(v)).collect();
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let config = LayerNormSimdConfig::new(vec![4]);

        let out = layer_norm_fp16(&input_fp16, &gamma, Some(&beta), &config).unwrap();
        assert_eq!(out.len(), 4);
        for &h in &out {
            let v = fp16_to_f32(h);
            assert!(v.is_finite(), "fp16 output should be finite, got {v}");
        }
    }

    #[test]
    fn fp16_layer_norm_matches_f32_approx() {
        let input_f32 = vec![1.0, 2.0, 3.0, 4.0];
        let input_fp16: Vec<u16> = input_f32.iter().map(|&v| f32_to_fp16(v)).collect();
        let gamma = vec![1.0; 4];
        let config = LayerNormSimdConfig::new(vec![4]);

        let fp16_result = layer_norm_fp16(&input_fp16, &gamma, None, &config).unwrap();
        let f32_result = layer_norm_f32(&input_f32, &gamma, None, &config).unwrap();

        let fp16_f32: Vec<f32> = fp16_result.iter().map(|&h| fp16_to_f32(h)).collect();
        assert!(
            approx_eq(&fp16_f32, &f32_result, 0.01),
            "fp16 vs f32: fp16={fp16_f32:?}, f32={f32_result:?}",
        );
    }

    #[test]
    fn fp16_empty_error() {
        let gamma = vec![1.0; 3];
        let config = LayerNormSimdConfig::new(vec![3]);
        assert!(layer_norm_fp16(&[], &gamma, None, &config).is_err());
    }

    // ── Batch layer norm ───────────────────────────────────

    #[test]
    fn batch_ln_matches_individual() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![10.0, 20.0, 30.0];
        let gamma = vec![1.0; 3];
        let config = LayerNormSimdConfig::new(vec![3]);

        let individual_a = layer_norm_avx2(&a, &gamma, None, &config).unwrap();
        let individual_b = layer_norm_avx2(&b, &gamma, None, &config).unwrap();
        let batched = batch_layer_norm(&[&a, &b], &gamma, None, &config).unwrap();

        assert_eq!(batched.len(), 2);
        assert!(approx_eq(&batched[0], &individual_a, TOL));
        assert!(approx_eq(&batched[1], &individual_b, TOL));
    }

    #[test]
    fn batch_ln_empty() {
        let gamma = vec![1.0; 3];
        let config = LayerNormSimdConfig::new(vec![3]);
        let inputs: Vec<&[f32]> = vec![];
        let batched = batch_layer_norm(&inputs, &gamma, None, &config).unwrap();
        assert!(batched.is_empty());
    }

    #[test]
    fn batch_ln_propagates_error() {
        let good = vec![1.0, 2.0, 3.0];
        let bad = vec![1.0, 2.0];
        let gamma = vec![1.0; 3];
        let config = LayerNormSimdConfig::new(vec![3]);
        assert!(batch_layer_norm(&[&good, &bad], &gamma, None, &config).is_err());
    }

    // ── Error cases ────────────────────────────────────────

    #[test]
    fn scalar_ln_empty_error() {
        let config = LayerNormSimdConfig::new(vec![3]);
        assert!(layer_norm_f32(&[], &[1.0; 3], None, &config).is_err());
    }

    #[test]
    fn scalar_rms_empty_error() {
        let config = RMSNormConfig::new(vec![3]);
        assert!(rms_norm_f32(&[], &[1.0; 3], &config).is_err());
    }

    #[test]
    fn scalar_ln_zero_eps_error() {
        let input = vec![1.0, 2.0, 3.0];
        let gamma = vec![1.0; 3];
        let mut config = LayerNormSimdConfig::new(vec![3]);
        config.eps = 0.0;
        assert!(layer_norm_f32(&input, &gamma, None, &config).is_err());
    }

    #[test]
    fn scalar_ln_negative_eps_error() {
        let input = vec![1.0, 2.0];
        let gamma = vec![1.0; 2];
        let mut config = LayerNormSimdConfig::new(vec![2]);
        config.eps = -1e-5;
        assert!(layer_norm_f32(&input, &gamma, None, &config).is_err());
    }

    #[test]
    fn scalar_ln_inf_eps_error() {
        let input = vec![1.0, 2.0];
        let gamma = vec![1.0; 2];
        let mut config = LayerNormSimdConfig::new(vec![2]);
        config.eps = f32::INFINITY;
        assert!(layer_norm_f32(&input, &gamma, None, &config).is_err());
    }

    #[test]
    fn scalar_ln_gamma_mismatch_error() {
        let input = vec![1.0, 2.0, 3.0];
        let gamma = vec![1.0; 2];
        let config = LayerNormSimdConfig::new(vec![3]);
        assert!(layer_norm_f32(&input, &gamma, None, &config).is_err());
    }

    #[test]
    fn scalar_ln_beta_mismatch_error() {
        let input = vec![1.0, 2.0, 3.0];
        let gamma = vec![1.0; 3];
        let beta = vec![0.0; 2];
        let config = LayerNormSimdConfig::new(vec![3]);
        assert!(layer_norm_f32(&input, &gamma, Some(&beta), &config).is_err());
    }

    #[test]
    fn scalar_ln_input_not_multiple_error() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gamma = vec![1.0; 3];
        let config = LayerNormSimdConfig::new(vec![3]);
        assert!(layer_norm_f32(&input, &gamma, None, &config).is_err());
    }

    #[test]
    fn scalar_rms_gamma_mismatch_error() {
        let input = vec![1.0, 2.0, 3.0];
        let gamma = vec![1.0; 4];
        let config = RMSNormConfig::new(vec![3]);
        assert!(rms_norm_f32(&input, &gamma, &config).is_err());
    }

    #[test]
    fn scalar_ln_zero_shape_error() {
        let input = vec![1.0, 2.0];
        let gamma: Vec<f32> = vec![];
        let config = LayerNormSimdConfig {
            normalized_shape: vec![0],
            eps: 1e-5,
            elementwise_affine: true,
            bias: true,
        };
        assert!(layer_norm_f32(&input, &gamma, None, &config).is_err());
    }

    // ── Edge cases ─────────────────────────────────────────

    #[test]
    fn scalar_ln_single_element() {
        let input = vec![42.0];
        let gamma = vec![2.0];
        let beta = vec![1.0];
        let config = LayerNormSimdConfig::new(vec![1]);
        let out = layer_norm_f32(&input, &gamma, Some(&beta), &config).unwrap();
        assert!((out[0] - 1.0).abs() < TOL, "expected 1.0, got {}", out[0]);
    }

    #[test]
    fn scalar_ln_two_elements() {
        let input = vec![0.0, 2.0];
        let gamma = vec![1.0; 2];
        let config = LayerNormSimdConfig::new(vec![2]);
        let out = layer_norm_f32(&input, &gamma, None, &config).unwrap();
        let expected = reference_layer_norm(&input, &gamma, None, 1e-5);
        assert!(approx_eq(&out, &expected, TOL));
    }

    #[test]
    fn avx2_ln_single_element() {
        let input = vec![42.0];
        let gamma = vec![2.0];
        let beta = vec![1.0];
        let config = LayerNormSimdConfig::new(vec![1]);
        let out = layer_norm_avx2(&input, &gamma, Some(&beta), &config).unwrap();
        assert!((out[0] - 1.0).abs() < TOL);
    }

    #[test]
    fn avx2_ln_exactly_8_elements() {
        let input: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let gamma = vec![1.0; 8];
        let config = LayerNormSimdConfig::new(vec![8]);

        let scalar = layer_norm_f32(&input, &gamma, None, &config).unwrap();
        let avx = layer_norm_avx2(&input, &gamma, None, &config).unwrap();
        assert!(approx_eq(&scalar, &avx, TOL));
    }

    #[test]
    fn avx2_ln_exactly_16_elements() {
        let input: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        let gamma = vec![1.0; 16];
        let beta = vec![0.5; 16];
        let config = LayerNormSimdConfig::new(vec![16]);

        let scalar = layer_norm_f32(&input, &gamma, Some(&beta), &config).unwrap();
        let avx = layer_norm_avx2(&input, &gamma, Some(&beta), &config).unwrap();
        assert!(approx_eq(&scalar, &avx, TOL));
    }

    #[test]
    fn avx2_rms_single_element() {
        let input = vec![42.0];
        let gamma = vec![1.0];
        let config = RMSNormConfig::new(vec![1]);

        let scalar = rms_norm_f32(&input, &gamma, &config).unwrap();
        let avx = rms_norm_avx2(&input, &gamma, &config).unwrap();
        assert!(approx_eq(&scalar, &avx, TOL));
    }

    // ── Large dimension stress test ────────────────────────

    #[test]
    fn avx2_ln_large_dim() {
        let n = 1024;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
        let gamma = vec![1.0; n];
        let beta = vec![0.0; n];
        let config = LayerNormSimdConfig::new(vec![n]);

        let scalar = layer_norm_f32(&input, &gamma, Some(&beta), &config).unwrap();
        let avx = layer_norm_avx2(&input, &gamma, Some(&beta), &config).unwrap();
        assert!(approx_eq(&scalar, &avx, TOL), "AVX2 large dim diverged from scalar");
    }

    #[test]
    fn avx2_rms_large_dim() {
        let n = 1024;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).cos()).collect();
        let gamma = vec![1.0; n];
        let config = RMSNormConfig::new(vec![n]);

        let scalar = rms_norm_f32(&input, &gamma, &config).unwrap();
        let avx = rms_norm_avx2(&input, &gamma, &config).unwrap();
        assert!(approx_eq(&scalar, &avx, TOL));
    }
}
