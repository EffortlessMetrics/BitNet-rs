//! SIMD-optimized LayerNorm and RMSNorm for CPU inference.
//!
//! Provides vectorised `f32` implementations of:
//! - **LayerNorm**: `(x − μ) / √(σ² + ε) · γ + β`
//! - **RMSNorm**: `x / √(mean(x²) + ε) · w`
//! - **Fused LayerNorm + residual add**
//!
//! On x86-64 with AVX2 the hot loops use 256-bit intrinsics; otherwise an
//! auto-vectorisable scalar fallback is used.  All public functions validate
//! shapes up-front and return [`NormError`] on mismatch.

use std::fmt;

// ── Configuration types ──────────────────────────────────────────

/// Configuration for standard Layer Normalization.
#[derive(Debug, Clone)]
pub struct LayerNormConfig {
    /// Small constant added to the variance for numerical stability.
    pub eps: f32,
    /// Whether to apply learnable affine parameters (gamma/beta).
    pub elementwise_affine: bool,
    /// The shape of the normalized dimension (number of elements).
    pub normalized_shape: usize,
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        Self { eps: 1e-5, elementwise_affine: true, normalized_shape: 0 }
    }
}

/// Configuration for Root Mean Square Normalization.
#[derive(Debug, Clone)]
pub struct RmsNormConfig {
    /// Small constant added to the mean-square for numerical stability.
    pub eps: f32,
    /// Optional scaling factor applied to the weight after normalisation.
    pub weight_scaling: f32,
}

impl Default for RmsNormConfig {
    fn default() -> Self {
        Self { eps: 1e-5, weight_scaling: 1.0 }
    }
}

// ── Error type ───────────────────────────────────────────────────

/// Errors returned by normalization functions.
#[derive(Debug, Clone, PartialEq)]
pub enum NormError {
    /// Input length is not divisible by `normalized_shape`.
    ShapeMismatch { input_len: usize, expected: usize },
    /// Epsilon must be positive and finite.
    InvalidEps(f32),
    /// Input slice is empty.
    EmptyInput,
    /// Weight/bias length does not match the normalized dimension.
    WeightMismatch { weight_len: usize, expected: usize },
}

impl fmt::Display for NormError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { input_len, expected } => {
                write!(f, "shape mismatch: input length {input_len} not divisible by {expected}")
            }
            Self::InvalidEps(e) => write!(f, "invalid eps: {e}"),
            Self::EmptyInput => write!(f, "empty input"),
            Self::WeightMismatch { weight_len, expected } => {
                write!(f, "weight length {weight_len} != expected {expected}")
            }
        }
    }
}

impl std::error::Error for NormError {}

// ── Statistics ───────────────────────────────────────────────────

/// Summary statistics computed over an input slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormStats {
    pub mean: f32,
    pub variance: f32,
    pub rms: f32,
}

// ── Validation helpers ───────────────────────────────────────────

fn validate_eps(eps: f32) -> Result<(), NormError> {
    if !eps.is_finite() || eps <= 0.0 {
        return Err(NormError::InvalidEps(eps));
    }
    Ok(())
}

fn validate_layer_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    config: &LayerNormConfig,
) -> Result<usize, NormError> {
    validate_eps(config.eps)?;
    if input.is_empty() {
        return Err(NormError::EmptyInput);
    }
    let n = config.normalized_shape;
    if n == 0 || !input.len().is_multiple_of(n) {
        return Err(NormError::ShapeMismatch { input_len: input.len(), expected: n });
    }
    if config.elementwise_affine {
        if gamma.len() != n {
            return Err(NormError::WeightMismatch { weight_len: gamma.len(), expected: n });
        }
        if beta.len() != n {
            return Err(NormError::WeightMismatch { weight_len: beta.len(), expected: n });
        }
    }
    Ok(n)
}

fn validate_rms_norm(
    input: &[f32],
    weight: &[f32],
    config: &RmsNormConfig,
) -> Result<(), NormError> {
    validate_eps(config.eps)?;
    if input.is_empty() {
        return Err(NormError::EmptyInput);
    }
    if weight.len() != input.len() {
        return Err(NormError::WeightMismatch { weight_len: weight.len(), expected: input.len() });
    }
    Ok(())
}

// ── SIMD primitives (AVX2) ───────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod avx2 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// Horizontal sum of all 8 lanes in a `__m256`.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(super) unsafe fn hsum_ps(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(hi, lo);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let sums2 = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(sums2)
    }

    /// Compute (sum, sum_sq) over `data` using AVX2+FMA.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(super) unsafe fn sum_and_sumsq(data: &[f32]) -> (f32, f32) {
        unsafe {
            let n = data.len();
            let ptr = data.as_ptr();
            let mut vsum = _mm256_setzero_ps();
            let mut vsumsq = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 8 <= n {
                let v = _mm256_loadu_ps(ptr.add(i));
                vsum = _mm256_add_ps(vsum, v);
                vsumsq = _mm256_fmadd_ps(v, v, vsumsq);
                i += 8;
            }
            let mut s = hsum_ps(vsum);
            let mut sq = hsum_ps(vsumsq);
            while i < n {
                let x = *ptr.add(i);
                s += x;
                sq += x * x;
                i += 1;
            }
            (s, sq)
        }
    }

    /// Compute sum-of-squares over `data` using AVX2+FMA.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(super) unsafe fn sum_of_squares(data: &[f32]) -> f32 {
        unsafe {
            let n = data.len();
            let ptr = data.as_ptr();
            let mut vsumsq = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 8 <= n {
                let v = _mm256_loadu_ps(ptr.add(i));
                vsumsq = _mm256_fmadd_ps(v, v, vsumsq);
                i += 8;
            }
            let mut sq = hsum_ps(vsumsq);
            while i < n {
                let x = *ptr.add(i);
                sq += x * x;
                i += 1;
            }
            sq
        }
    }

    /// out[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(super) unsafe fn normalize_affine(
        x: &[f32],
        out: &mut [f32],
        mean: f32,
        inv_std: f32,
        gamma: &[f32],
        beta: &[f32],
    ) {
        unsafe {
            let n = x.len();
            let vmean = _mm256_set1_ps(mean);
            let vinv = _mm256_set1_ps(inv_std);
            let mut i = 0usize;
            while i + 8 <= n {
                let vx = _mm256_loadu_ps(x.as_ptr().add(i));
                let vg = _mm256_loadu_ps(gamma.as_ptr().add(i));
                let vb = _mm256_loadu_ps(beta.as_ptr().add(i));
                let centered = _mm256_sub_ps(vx, vmean);
                let normed = _mm256_mul_ps(centered, vinv);
                let scaled = _mm256_fmadd_ps(normed, vg, vb);
                _mm256_storeu_ps(out.as_mut_ptr().add(i), scaled);
                i += 8;
            }
            while i < n {
                let normed = (x[i] - mean) * inv_std;
                out[i] = normed * gamma[i] + beta[i];
                i += 1;
            }
        }
    }

    /// out[i] = (x[i] - mean) * inv_std  (no affine)
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(super) unsafe fn normalize_no_affine(x: &[f32], out: &mut [f32], mean: f32, inv_std: f32) {
        unsafe {
            let n = x.len();
            let vmean = _mm256_set1_ps(mean);
            let vinv = _mm256_set1_ps(inv_std);
            let mut i = 0usize;
            while i + 8 <= n {
                let vx = _mm256_loadu_ps(x.as_ptr().add(i));
                let centered = _mm256_sub_ps(vx, vmean);
                let normed = _mm256_mul_ps(centered, vinv);
                _mm256_storeu_ps(out.as_mut_ptr().add(i), normed);
                i += 8;
            }
            while i < n {
                out[i] = (x[i] - mean) * inv_std;
                i += 1;
            }
        }
    }

    /// out[i] = x[i] * inv_rms * weight[i] * weight_scaling
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(super) unsafe fn rms_normalize(
        x: &[f32],
        out: &mut [f32],
        inv_rms: f32,
        weight: &[f32],
        weight_scaling: f32,
    ) {
        unsafe {
            let n = x.len();
            let vinv = _mm256_set1_ps(inv_rms * weight_scaling);
            let mut i = 0usize;
            while i + 8 <= n {
                let vx = _mm256_loadu_ps(x.as_ptr().add(i));
                let vw = _mm256_loadu_ps(weight.as_ptr().add(i));
                let scaled = _mm256_mul_ps(vx, vinv);
                let result = _mm256_mul_ps(scaled, vw);
                _mm256_storeu_ps(out.as_mut_ptr().add(i), result);
                i += 8;
            }
            let factor = inv_rms * weight_scaling;
            while i < n {
                out[i] = x[i] * factor * weight[i];
                i += 1;
            }
        }
    }

    /// out[i] = (x[i] + residual[i] - mean) * inv_std * gamma[i] + beta[i]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(super) unsafe fn fused_residual_normalize(
        x: &[f32],
        residual: &[f32],
        out: &mut [f32],
        mean: f32,
        inv_std: f32,
        gamma: &[f32],
        beta: &[f32],
    ) {
        unsafe {
            let n = x.len();
            let vmean = _mm256_set1_ps(mean);
            let vinv = _mm256_set1_ps(inv_std);
            let mut i = 0usize;
            while i + 8 <= n {
                let vx = _mm256_loadu_ps(x.as_ptr().add(i));
                let vr = _mm256_loadu_ps(residual.as_ptr().add(i));
                let vg = _mm256_loadu_ps(gamma.as_ptr().add(i));
                let vb = _mm256_loadu_ps(beta.as_ptr().add(i));
                let added = _mm256_add_ps(vx, vr);
                let centered = _mm256_sub_ps(added, vmean);
                let normed = _mm256_mul_ps(centered, vinv);
                let scaled = _mm256_fmadd_ps(normed, vg, vb);
                _mm256_storeu_ps(out.as_mut_ptr().add(i), scaled);
                i += 8;
            }
            while i < n {
                let sum = x[i] + residual[i];
                let normed = (sum - mean) * inv_std;
                out[i] = normed * gamma[i] + beta[i];
                i += 1;
            }
        }
    }
}

// ── Scalar fallback primitives ───────────────────────────────────

mod scalar {
    pub(super) fn sum_and_sumsq(data: &[f32]) -> (f32, f32) {
        let mut s: f64 = 0.0;
        let mut sq: f64 = 0.0;
        for &x in data {
            let x64 = x as f64;
            s += x64;
            sq += x64 * x64;
        }
        (s as f32, sq as f32)
    }

    pub(super) fn sum_of_squares(data: &[f32]) -> f32 {
        let mut sq: f64 = 0.0;
        for &x in data {
            let x64 = x as f64;
            sq += x64 * x64;
        }
        sq as f32
    }
}

// ── Runtime dispatch helpers ─────────────────────────────────────

fn sum_and_sumsq(data: &[f32]) -> (f32, f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { avx2::sum_and_sumsq(data) };
        }
    }
    scalar::sum_and_sumsq(data)
}

fn sum_of_squares(data: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { avx2::sum_of_squares(data) };
        }
    }
    scalar::sum_of_squares(data)
}

/// Normalize a single row with affine parameters.
fn normalize_row_affine(
    row: &[f32],
    out: &mut [f32],
    mean: f32,
    inv_std: f32,
    gamma: &[f32],
    beta: &[f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { avx2::normalize_affine(row, out, mean, inv_std, gamma, beta) };
            return;
        }
    }
    for i in 0..row.len() {
        let normed = (row[i] - mean) * inv_std;
        out[i] = normed * gamma[i] + beta[i];
    }
}

/// Normalize a single row without affine parameters.
fn normalize_row_no_affine(row: &[f32], out: &mut [f32], mean: f32, inv_std: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { avx2::normalize_no_affine(row, out, mean, inv_std) };
            return;
        }
    }
    for i in 0..row.len() {
        out[i] = (row[i] - mean) * inv_std;
    }
}

/// Mean and inverse standard-deviation of a row.
fn mean_inv_std(row: &[f32], eps: f32) -> (f32, f32) {
    let n = row.len() as f32;
    let (s, sq) = sum_and_sumsq(row);
    let mean = s / n;
    let var = (sq / n) - mean * mean;
    let inv_std = 1.0 / (var + eps).sqrt();
    (mean, inv_std)
}

// ── Public API ───────────────────────────────────────────────────

/// Compute summary statistics (mean, variance, RMS) over a slice.
pub fn compute_norm_stats(input: &[f32]) -> NormStats {
    if input.is_empty() {
        return NormStats { mean: 0.0, variance: 0.0, rms: 0.0 };
    }
    let n = input.len() as f32;
    let (s, sq) = sum_and_sumsq(input);
    let mean = s / n;
    let variance = (sq / n) - mean * mean;
    let rms = (sq / n).sqrt();
    NormStats { mean, variance, rms }
}

/// 1-D Layer Normalization.
///
/// Normalises `input` (length must equal `config.normalized_shape`) and applies
/// the affine transformation `gamma * x_hat + beta` when
/// `config.elementwise_affine` is `true`.
pub fn layer_norm_f32(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    config: &LayerNormConfig,
) -> Result<Vec<f32>, NormError> {
    let n = validate_layer_norm(input, gamma, beta, config)?;
    debug_assert_eq!(input.len(), n);
    let mut out = vec![0.0f32; n];
    let (mean, inv_std) = mean_inv_std(input, config.eps);
    if config.elementwise_affine {
        normalize_row_affine(input, &mut out, mean, inv_std, gamma, beta);
    } else {
        normalize_row_no_affine(input, &mut out, mean, inv_std);
    }
    Ok(out)
}

/// RMS Normalization.
///
/// `out[i] = x[i] / rms(x) * weight[i] * weight_scaling`
pub fn rms_norm_f32(
    input: &[f32],
    weight: &[f32],
    config: &RmsNormConfig,
) -> Result<Vec<f32>, NormError> {
    validate_rms_norm(input, weight, config)?;
    let n = input.len();
    let sq = sum_of_squares(input);
    let rms = (sq / n as f32 + config.eps).sqrt();
    let inv_rms = 1.0 / rms;
    let mut out = vec![0.0f32; n];
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { avx2::rms_normalize(input, &mut out, inv_rms, weight, config.weight_scaling) };
            return Ok(out);
        }
    }
    let factor = inv_rms * config.weight_scaling;
    for i in 0..n {
        out[i] = input[i] * factor * weight[i];
    }
    Ok(out)
}

/// 2-D Layer Normalization (batch of rows).
///
/// `input` is a flat `[rows × cols]` buffer.  Each row of length `cols` is
/// normalised independently.  `gamma` and `beta` have length `cols`.
pub fn layer_norm_2d(
    input: &[f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
    beta: &[f32],
    config: &LayerNormConfig,
) -> Result<Vec<f32>, NormError> {
    validate_eps(config.eps)?;
    if input.is_empty() {
        return Err(NormError::EmptyInput);
    }
    if cols == 0 || input.len() != rows * cols {
        return Err(NormError::ShapeMismatch { input_len: input.len(), expected: rows * cols });
    }
    if config.elementwise_affine {
        if gamma.len() != cols {
            return Err(NormError::WeightMismatch { weight_len: gamma.len(), expected: cols });
        }
        if beta.len() != cols {
            return Err(NormError::WeightMismatch { weight_len: beta.len(), expected: cols });
        }
    }

    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let start = r * cols;
        let end = start + cols;
        let row = &input[start..end];
        let (mean, inv_std) = mean_inv_std(row, config.eps);
        if config.elementwise_affine {
            normalize_row_affine(row, &mut out[start..end], mean, inv_std, gamma, beta);
        } else {
            normalize_row_no_affine(row, &mut out[start..end], mean, inv_std);
        }
    }
    Ok(out)
}

/// Fused residual-add + Layer Normalization.
///
/// Computes `LayerNorm(input + residual)` in a single pass, avoiding an
/// intermediate allocation for the element-wise sum.
pub fn fused_layer_norm_residual(
    input: &[f32],
    residual: &[f32],
    gamma: &[f32],
    beta: &[f32],
    config: &LayerNormConfig,
) -> Result<Vec<f32>, NormError> {
    let n = validate_layer_norm(input, gamma, beta, config)?;
    if residual.len() != n {
        return Err(NormError::ShapeMismatch { input_len: residual.len(), expected: n });
    }

    // Compute mean and variance of (input + residual) in one pass.
    let nf = n as f32;
    let mut s: f64 = 0.0;
    let mut sq: f64 = 0.0;
    for i in 0..n {
        let v = (input[i] + residual[i]) as f64;
        s += v;
        sq += v * v;
    }
    let mean = (s / nf as f64) as f32;
    let var = (sq / nf as f64) as f32 - mean * mean;
    let inv_std = 1.0 / (var + config.eps).sqrt();

    let mut out = vec![0.0f32; n];
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                avx2::fused_residual_normalize(
                    input, residual, &mut out, mean, inv_std, gamma, beta,
                )
            };
            return Ok(out);
        }
    }
    if config.elementwise_affine {
        for i in 0..n {
            let sum = input[i] + residual[i];
            let normed = (sum - mean) * inv_std;
            out[i] = normed * gamma[i] + beta[i];
        }
    } else {
        for i in 0..n {
            let sum = input[i] + residual[i];
            out[i] = (sum - mean) * inv_std;
        }
    }
    Ok(out)
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_ln_config(n: usize) -> LayerNormConfig {
        LayerNormConfig { eps: 1e-5, elementwise_affine: true, normalized_shape: n }
    }

    fn ones(n: usize) -> Vec<f32> {
        vec![1.0; n]
    }

    fn zeros(n: usize) -> Vec<f32> {
        vec![0.0; n]
    }

    // ── compute_norm_stats ───────────────────────────────────────

    #[test]
    fn stats_empty_input() {
        let s = compute_norm_stats(&[]);
        assert_eq!(s.mean, 0.0);
        assert_eq!(s.variance, 0.0);
        assert_eq!(s.rms, 0.0);
    }

    #[test]
    fn stats_single_element() {
        let s = compute_norm_stats(&[5.0]);
        assert!((s.mean - 5.0).abs() < 1e-6);
        assert!(s.variance.abs() < 1e-6);
        assert!((s.rms - 5.0).abs() < 1e-6);
    }

    #[test]
    fn stats_uniform() {
        let input = vec![3.0; 16];
        let s = compute_norm_stats(&input);
        assert!((s.mean - 3.0).abs() < 1e-5);
        assert!(s.variance.abs() < 1e-5);
    }

    #[test]
    fn stats_known_values() {
        // [1,2,3,4,5] → mean=3, var=2, rms=sqrt(11)
        let s = compute_norm_stats(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((s.mean - 3.0).abs() < 1e-5);
        assert!((s.variance - 2.0).abs() < 1e-4);
        assert!((s.rms - (11.0f32).sqrt()).abs() < 1e-4);
    }

    #[test]
    fn stats_negative_values() {
        let s = compute_norm_stats(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        assert!(s.mean.abs() < 1e-6);
        assert!((s.variance - 2.0).abs() < 1e-4);
    }

    #[test]
    fn stats_large_array() {
        let input: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let s = compute_norm_stats(&input);
        let expected_mean = 511.5;
        assert!((s.mean - expected_mean).abs() < 0.5);
    }

    // ── layer_norm_f32 ───────────────────────────────────────────

    #[test]
    fn ln_identity_transform() {
        // gamma=1, beta=0 → normalised output
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let cfg = default_ln_config(4);
        let out = layer_norm_f32(&input, &ones(4), &zeros(4), &cfg).unwrap();
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean should be ~0, got {mean}");
    }

    #[test]
    fn ln_output_variance() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let cfg = default_ln_config(4);
        let out = layer_norm_f32(&input, &ones(4), &zeros(4), &cfg).unwrap();
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        let var: f32 = out.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 0.01, "variance should be ~1, got {var}");
    }

    #[test]
    fn ln_with_affine() {
        let input = vec![0.0, 1.0, 2.0, 3.0];
        let gamma = vec![2.0; 4];
        let beta = vec![1.0; 4];
        let cfg = default_ln_config(4);
        let out = layer_norm_f32(&input, &gamma, &beta, &cfg).unwrap();
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        // mean of output should be ~1 (beta)
        assert!((mean - 1.0).abs() < 0.01);
    }

    #[test]
    fn ln_no_affine() {
        let input = vec![1.0, 3.0, 5.0, 7.0];
        let cfg = LayerNormConfig { eps: 1e-5, elementwise_affine: false, normalized_shape: 4 };
        let out = layer_norm_f32(&input, &[], &[], &cfg).unwrap();
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5);
    }

    #[test]
    fn ln_constant_input() {
        let input = vec![5.0; 8];
        let cfg = default_ln_config(8);
        let out = layer_norm_f32(&input, &ones(8), &zeros(8), &cfg).unwrap();
        // All the same → after norm all should be ~0
        for &v in &out {
            assert!(v.abs() < 0.01, "expected ~0 for constant input, got {v}");
        }
    }

    #[test]
    fn ln_two_element() {
        let input = vec![0.0, 2.0];
        let cfg = default_ln_config(2);
        let out = layer_norm_f32(&input, &ones(2), &zeros(2), &cfg).unwrap();
        // mean=1, std=1 → out ≈ [-1, 1]
        assert!((out[0] + 1.0).abs() < 0.01);
        assert!((out[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn ln_large_eps() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let cfg = LayerNormConfig { eps: 1.0, elementwise_affine: true, normalized_shape: 4 };
        let out = layer_norm_f32(&input, &ones(4), &zeros(4), &cfg).unwrap();
        // Should still produce finite output
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn ln_size_8() {
        let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let cfg = default_ln_config(8);
        let out = layer_norm_f32(&input, &ones(8), &zeros(8), &cfg).unwrap();
        assert_eq!(out.len(), 8);
        let mean: f32 = out.iter().sum::<f32>() / 8.0;
        assert!(mean.abs() < 1e-4);
    }

    #[test]
    fn ln_size_16() {
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let cfg = default_ln_config(16);
        let out = layer_norm_f32(&input, &ones(16), &zeros(16), &cfg).unwrap();
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn ln_size_17_odd() {
        // Non-SIMD-aligned size
        let input: Vec<f32> = (0..17).map(|i| i as f32).collect();
        let cfg = default_ln_config(17);
        let out = layer_norm_f32(&input, &ones(17), &zeros(17), &cfg).unwrap();
        assert_eq!(out.len(), 17);
        let mean: f32 = out.iter().sum::<f32>() / 17.0;
        assert!(mean.abs() < 1e-4);
    }

    #[test]
    fn ln_size_1() {
        let cfg = default_ln_config(1);
        let out = layer_norm_f32(&[42.0], &[1.0], &[0.0], &cfg).unwrap();
        // single element → variance 0 → output ~0
        assert!(out[0].abs() < 0.01);
    }

    #[test]
    fn ln_negative_input() {
        let input = vec![-4.0, -2.0, 0.0, 2.0, 4.0];
        let cfg = default_ln_config(5);
        let out = layer_norm_f32(&input, &ones(5), &zeros(5), &cfg).unwrap();
        let mean: f32 = out.iter().sum::<f32>() / 5.0;
        assert!(mean.abs() < 1e-4);
    }

    #[test]
    fn ln_preserves_length() {
        for n in [1, 2, 3, 7, 8, 9, 15, 16, 31, 32, 33, 64, 128, 255, 256] {
            let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let cfg = default_ln_config(n);
            let out = layer_norm_f32(&input, &ones(n), &zeros(n), &cfg).unwrap();
            assert_eq!(out.len(), n);
        }
    }

    #[test]
    fn ln_output_finite() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let cfg = default_ln_config(64);
        let out = layer_norm_f32(&input, &ones(64), &zeros(64), &cfg).unwrap();
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn ln_symmetry() {
        // Symmetric input → symmetric output
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let cfg = default_ln_config(5);
        let out = layer_norm_f32(&input, &ones(5), &zeros(5), &cfg).unwrap();
        assert!((out[0] + out[4]).abs() < 1e-5);
        assert!((out[1] + out[3]).abs() < 1e-5);
        assert!(out[2].abs() < 1e-5);
    }

    #[test]
    fn ln_gamma_scaling() {
        let input = vec![0.0, 1.0, 2.0, 3.0];
        let gamma1 = vec![1.0; 4];
        let gamma2 = vec![2.0; 4];
        let beta = zeros(4);
        let cfg = default_ln_config(4);
        let out1 = layer_norm_f32(&input, &gamma1, &beta, &cfg).unwrap();
        let out2 = layer_norm_f32(&input, &gamma2, &beta, &cfg).unwrap();
        for i in 0..4 {
            assert!((out2[i] - 2.0 * out1[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn ln_beta_shift() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = ones(4);
        let beta1 = zeros(4);
        let beta2 = vec![10.0; 4];
        let cfg = default_ln_config(4);
        let out1 = layer_norm_f32(&input, &gamma, &beta1, &cfg).unwrap();
        let out2 = layer_norm_f32(&input, &gamma, &beta2, &cfg).unwrap();
        for i in 0..4 {
            assert!((out2[i] - out1[i] - 10.0).abs() < 1e-5);
        }
    }

    // ── layer_norm_f32 error cases ──────────────────────────────

    #[test]
    fn ln_err_empty() {
        let cfg = default_ln_config(4);
        assert_eq!(layer_norm_f32(&[], &ones(4), &zeros(4), &cfg), Err(NormError::EmptyInput));
    }

    #[test]
    fn ln_err_shape_mismatch() {
        let cfg = default_ln_config(4);
        let err = layer_norm_f32(&[1.0, 2.0, 3.0], &ones(4), &zeros(4), &cfg);
        assert!(matches!(err, Err(NormError::ShapeMismatch { .. })));
    }

    #[test]
    fn ln_err_gamma_mismatch() {
        let cfg = default_ln_config(4);
        let err = layer_norm_f32(&[1.0; 4], &ones(3), &zeros(4), &cfg);
        assert!(matches!(err, Err(NormError::WeightMismatch { .. })));
    }

    #[test]
    fn ln_err_beta_mismatch() {
        let cfg = default_ln_config(4);
        let err = layer_norm_f32(&[1.0; 4], &ones(4), &zeros(3), &cfg);
        assert!(matches!(err, Err(NormError::WeightMismatch { .. })));
    }

    #[test]
    fn ln_err_invalid_eps_zero() {
        let cfg = LayerNormConfig { eps: 0.0, elementwise_affine: true, normalized_shape: 4 };
        let err = layer_norm_f32(&[1.0; 4], &ones(4), &zeros(4), &cfg);
        assert!(matches!(err, Err(NormError::InvalidEps(_))));
    }

    #[test]
    fn ln_err_invalid_eps_negative() {
        let cfg = LayerNormConfig { eps: -1.0, elementwise_affine: true, normalized_shape: 4 };
        let err = layer_norm_f32(&[1.0; 4], &ones(4), &zeros(4), &cfg);
        assert!(matches!(err, Err(NormError::InvalidEps(_))));
    }

    #[test]
    fn ln_err_invalid_eps_inf() {
        let cfg =
            LayerNormConfig { eps: f32::INFINITY, elementwise_affine: true, normalized_shape: 4 };
        let err = layer_norm_f32(&[1.0; 4], &ones(4), &zeros(4), &cfg);
        assert!(matches!(err, Err(NormError::InvalidEps(_))));
    }

    #[test]
    fn ln_err_invalid_eps_nan() {
        let cfg = LayerNormConfig { eps: f32::NAN, elementwise_affine: true, normalized_shape: 4 };
        let err = layer_norm_f32(&[1.0; 4], &ones(4), &zeros(4), &cfg);
        assert!(matches!(err, Err(NormError::InvalidEps(_))));
    }

    #[test]
    fn ln_err_zero_normalized_shape() {
        let cfg = LayerNormConfig { eps: 1e-5, elementwise_affine: true, normalized_shape: 0 };
        let err = layer_norm_f32(&[1.0; 4], &ones(4), &zeros(4), &cfg);
        assert!(matches!(err, Err(NormError::ShapeMismatch { .. })));
    }

    // ── rms_norm_f32 ─────────────────────────────────────────────

    #[test]
    fn rms_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let cfg = RmsNormConfig::default();
        let out = rms_norm_f32(&input, &weight, &cfg).unwrap();
        assert_eq!(out.len(), 4);
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn rms_unit_weight() {
        let input = vec![3.0; 8];
        let weight = vec![1.0; 8];
        let cfg = RmsNormConfig::default();
        let out = rms_norm_f32(&input, &weight, &cfg).unwrap();
        // All same → rms ≈ 3.0, out ≈ 1.0
        for &v in &out {
            assert!((v - 1.0).abs() < 0.01, "expected ~1, got {v}");
        }
    }

    #[test]
    fn rms_weight_scaling() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let cfg1 = RmsNormConfig { eps: 1e-5, weight_scaling: 1.0 };
        let cfg2 = RmsNormConfig { eps: 1e-5, weight_scaling: 2.0 };
        let out1 = rms_norm_f32(&input, &weight, &cfg1).unwrap();
        let out2 = rms_norm_f32(&input, &weight, &cfg2).unwrap();
        for i in 0..4 {
            assert!((out2[i] - 2.0 * out1[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn rms_different_weights() {
        let input = vec![1.0, 1.0, 1.0, 1.0];
        let weight = vec![1.0, 2.0, 3.0, 4.0];
        let cfg = RmsNormConfig::default();
        let out = rms_norm_f32(&input, &weight, &cfg).unwrap();
        // input is uniform, so out[i] ~ weight[i] * (1 / rms)
        // rms(1,1,1,1) = 1
        for i in 0..4 {
            assert!((out[i] - weight[i]).abs() < 0.01);
        }
    }

    #[test]
    fn rms_size_17() {
        let input: Vec<f32> = (1..=17).map(|i| i as f32).collect();
        let weight = vec![1.0; 17];
        let cfg = RmsNormConfig::default();
        let out = rms_norm_f32(&input, &weight, &cfg).unwrap();
        assert_eq!(out.len(), 17);
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn rms_single_element() {
        let cfg = RmsNormConfig::default();
        let out = rms_norm_f32(&[7.0], &[1.0], &cfg).unwrap();
        // rms = 7.0, out = 7.0 / 7.0 = 1.0
        assert!((out[0] - 1.0).abs() < 0.01);
    }

    // ── rms_norm_f32 error cases ─────────────────────────────────

    #[test]
    fn rms_err_empty() {
        let cfg = RmsNormConfig::default();
        assert_eq!(rms_norm_f32(&[], &[], &cfg), Err(NormError::EmptyInput));
    }

    #[test]
    fn rms_err_weight_mismatch() {
        let cfg = RmsNormConfig::default();
        let err = rms_norm_f32(&[1.0; 4], &[1.0; 3], &cfg);
        assert!(matches!(err, Err(NormError::WeightMismatch { .. })));
    }

    #[test]
    fn rms_err_invalid_eps() {
        let cfg = RmsNormConfig { eps: 0.0, weight_scaling: 1.0 };
        let err = rms_norm_f32(&[1.0; 4], &[1.0; 4], &cfg);
        assert!(matches!(err, Err(NormError::InvalidEps(_))));
    }

    // ── layer_norm_2d ────────────────────────────────────────────

    #[test]
    fn ln2d_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2×4
        let cfg = default_ln_config(4);
        let out = layer_norm_2d(&input, 2, 4, &ones(4), &zeros(4), &cfg).unwrap();
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn ln2d_each_row_normalized() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]; // 2×4
        let cfg = default_ln_config(4);
        let out = layer_norm_2d(&input, 2, 4, &ones(4), &zeros(4), &cfg).unwrap();
        // Each row should have mean ~0
        let row0_mean: f32 = out[..4].iter().sum::<f32>() / 4.0;
        let row1_mean: f32 = out[4..].iter().sum::<f32>() / 4.0;
        assert!(row0_mean.abs() < 1e-4);
        assert!(row1_mean.abs() < 1e-4);
    }

    #[test]
    fn ln2d_single_row() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let cfg = default_ln_config(4);
        let out_2d = layer_norm_2d(&input, 1, 4, &ones(4), &zeros(4), &cfg).unwrap();
        let out_1d = layer_norm_f32(&input, &ones(4), &zeros(4), &cfg).unwrap();
        for i in 0..4 {
            assert!((out_2d[i] - out_1d[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn ln2d_many_rows() {
        let rows = 32;
        let cols = 64;
        let input: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.01).collect();
        let cfg = default_ln_config(cols);
        let out = layer_norm_2d(&input, rows, cols, &ones(cols), &zeros(cols), &cfg).unwrap();
        assert_eq!(out.len(), rows * cols);
    }

    #[test]
    fn ln2d_no_affine() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
        let cfg = LayerNormConfig { eps: 1e-5, elementwise_affine: false, normalized_shape: 3 };
        let out = layer_norm_2d(&input, 2, 3, &[], &[], &cfg).unwrap();
        assert_eq!(out.len(), 6);
    }

    // ── layer_norm_2d error cases ────────────────────────────────

    #[test]
    fn ln2d_err_shape_mismatch() {
        let cfg = default_ln_config(4);
        let err = layer_norm_2d(&[1.0; 7], 2, 4, &ones(4), &zeros(4), &cfg);
        assert!(matches!(err, Err(NormError::ShapeMismatch { .. })));
    }

    #[test]
    fn ln2d_err_empty() {
        let cfg = default_ln_config(4);
        let err = layer_norm_2d(&[], 0, 4, &ones(4), &zeros(4), &cfg);
        assert!(matches!(err, Err(NormError::EmptyInput)));
    }

    #[test]
    fn ln2d_err_zero_cols() {
        let cfg = default_ln_config(0);
        let err = layer_norm_2d(&[1.0; 4], 4, 0, &[], &[], &cfg);
        assert!(matches!(err, Err(NormError::ShapeMismatch { .. })));
    }

    #[test]
    fn ln2d_err_gamma_mismatch() {
        let cfg = default_ln_config(4);
        let err = layer_norm_2d(&[1.0; 8], 2, 4, &ones(3), &zeros(4), &cfg);
        assert!(matches!(err, Err(NormError::WeightMismatch { .. })));
    }

    // ── fused_layer_norm_residual ────────────────────────────────

    #[test]
    fn fused_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![0.1, 0.2, 0.3, 0.4];
        let cfg = default_ln_config(4);
        let out = fused_layer_norm_residual(&input, &residual, &ones(4), &zeros(4), &cfg).unwrap();
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn fused_matches_separate() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![0.5, -0.5, 1.0, -1.0];
        let cfg = default_ln_config(4);
        let gamma = ones(4);
        let beta = zeros(4);

        let fused = fused_layer_norm_residual(&input, &residual, &gamma, &beta, &cfg).unwrap();

        // Manual: add then normalise
        let summed: Vec<f32> = input.iter().zip(&residual).map(|(a, b)| a + b).collect();
        let manual = layer_norm_f32(&summed, &gamma, &beta, &cfg).unwrap();

        for i in 0..4 {
            assert!(
                (fused[i] - manual[i]).abs() < 1e-4,
                "mismatch at {i}: fused={} manual={}",
                fused[i],
                manual[i],
            );
        }
    }

    #[test]
    fn fused_zero_residual() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![0.0; 4];
        let cfg = default_ln_config(4);
        let fused =
            fused_layer_norm_residual(&input, &residual, &ones(4), &zeros(4), &cfg).unwrap();
        let plain = layer_norm_f32(&input, &ones(4), &zeros(4), &cfg).unwrap();
        for i in 0..4 {
            assert!((fused[i] - plain[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn fused_size_17() {
        let n = 17;
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let residual: Vec<f32> = (0..n).map(|i| -(i as f32)).collect();
        let cfg = default_ln_config(n);
        let out = fused_layer_norm_residual(&input, &residual, &ones(n), &zeros(n), &cfg).unwrap();
        assert_eq!(out.len(), n);
    }

    // ── fused error cases ────────────────────────────────────────

    #[test]
    fn fused_err_residual_mismatch() {
        let cfg = default_ln_config(4);
        let err = fused_layer_norm_residual(&[1.0; 4], &[1.0; 3], &ones(4), &zeros(4), &cfg);
        assert!(matches!(err, Err(NormError::ShapeMismatch { .. })));
    }

    #[test]
    fn fused_err_empty() {
        let cfg = default_ln_config(4);
        let err = fused_layer_norm_residual(&[], &[], &ones(4), &zeros(4), &cfg);
        assert!(matches!(err, Err(NormError::EmptyInput)));
    }

    // ── NormError Display ────────────────────────────────────────

    #[test]
    fn error_display_shape() {
        let e = NormError::ShapeMismatch { input_len: 3, expected: 4 };
        assert!(e.to_string().contains("3"));
    }

    #[test]
    fn error_display_eps() {
        let e = NormError::InvalidEps(0.0);
        assert!(e.to_string().contains("0"));
    }

    #[test]
    fn error_display_empty() {
        assert!(NormError::EmptyInput.to_string().contains("empty"));
    }

    #[test]
    fn error_display_weight() {
        let e = NormError::WeightMismatch { weight_len: 3, expected: 4 };
        assert!(e.to_string().contains("3"));
    }

    // ── Default configs ──────────────────────────────────────────

    #[test]
    fn default_ln_config_values() {
        let cfg = LayerNormConfig::default();
        assert!((cfg.eps - 1e-5).abs() < 1e-10);
        assert!(cfg.elementwise_affine);
        assert_eq!(cfg.normalized_shape, 0);
    }

    #[test]
    fn default_rms_config_values() {
        let cfg = RmsNormConfig::default();
        assert!((cfg.eps - 1e-5).abs() < 1e-10);
        assert!((cfg.weight_scaling - 1.0).abs() < 1e-10);
    }

    // ── Additional edge cases ────────────────────────────────────

    #[test]
    fn ln_large_values() {
        let input = vec![1e6, 2e6, 3e6, 4e6];
        let cfg = default_ln_config(4);
        let out = layer_norm_f32(&input, &ones(4), &zeros(4), &cfg).unwrap();
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-2);
    }

    #[test]
    fn ln_small_values() {
        let input = vec![1e-6, 2e-6, 3e-6, 4e-6];
        let cfg = default_ln_config(4);
        let out = layer_norm_f32(&input, &ones(4), &zeros(4), &cfg).unwrap();
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn rms_large_array() {
        let n = 256;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let weight = vec![1.0; n];
        let cfg = RmsNormConfig::default();
        let out = rms_norm_f32(&input, &weight, &cfg).unwrap();
        assert_eq!(out.len(), n);
    }

    #[test]
    fn ln_size_256() {
        let n = 256;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) / n as f32).collect();
        let cfg = default_ln_config(n);
        let out = layer_norm_f32(&input, &ones(n), &zeros(n), &cfg).unwrap();
        let mean: f32 = out.iter().sum::<f32>() / n as f32;
        assert!(mean.abs() < 1e-3);
    }

    #[test]
    fn rms_preserves_sign() {
        let input = vec![-3.0, -1.0, 1.0, 3.0];
        let weight = vec![1.0; 4];
        let cfg = RmsNormConfig::default();
        let out = rms_norm_f32(&input, &weight, &cfg).unwrap();
        assert!(out[0] < 0.0);
        assert!(out[1] < 0.0);
        assert!(out[2] > 0.0);
        assert!(out[3] > 0.0);
    }

    #[test]
    fn ln2d_row_independence() {
        // Changing one row should not affect another
        let mut input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2×4
        let cfg = default_ln_config(4);
        let out1 = layer_norm_2d(&input, 2, 4, &ones(4), &zeros(4), &cfg).unwrap();
        input[0] = 100.0; // change row 0
        let out2 = layer_norm_2d(&input, 2, 4, &ones(4), &zeros(4), &cfg).unwrap();
        // Row 1 should be unchanged
        for i in 4..8 {
            assert!((out1[i] - out2[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn fused_with_affine() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![1.0, 1.0, 1.0, 1.0];
        let gamma = vec![2.0; 4];
        let beta = vec![0.5; 4];
        let cfg = default_ln_config(4);
        let out = fused_layer_norm_residual(&input, &residual, &gamma, &beta, &cfg).unwrap();
        assert_eq!(out.len(), 4);
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        assert!((mean - 0.5).abs() < 0.01);
    }

    #[test]
    fn stats_rms_non_negative() {
        let input = vec![-5.0, -3.0, -1.0, 0.0];
        let s = compute_norm_stats(&input);
        assert!(s.rms >= 0.0);
    }

    // ── proptest ─────────────────────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn arb_vec(min: usize, max: usize) -> impl Strategy<Value = Vec<f32>> {
            proptest::collection::vec(-100.0f32..100.0, min..=max)
        }

        proptest! {
            /// LayerNorm output should have mean ≈ 0 (with gamma=1, beta=0).
            #[test]
            fn prop_ln_output_mean_near_zero(input in arb_vec(2, 128)) {
                let n = input.len();
                let cfg = default_ln_config(n);
                let out = layer_norm_f32(&input, &ones(n), &zeros(n), &cfg).unwrap();
                let mean: f32 = out.iter().sum::<f32>() / n as f32;
                prop_assert!(mean.abs() < 0.01, "mean = {mean}");
            }

            /// LayerNorm output should have variance ≈ 1 (with gamma=1, beta=0).
            #[test]
            fn prop_ln_output_variance_near_one(input in arb_vec(4, 128)) {
                let n = input.len();
                let cfg = default_ln_config(n);
                let out = layer_norm_f32(&input, &ones(n), &zeros(n), &cfg).unwrap();
                let mean: f32 = out.iter().sum::<f32>() / n as f32;
                let var: f32 = out.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n as f32;
                prop_assert!((var - 1.0).abs() < 0.05, "variance = {var}");
            }

            /// RMSNorm with unit weights should produce values with RMS ≈ 1.
            #[test]
            fn prop_rms_output_rms_near_one(input in arb_vec(2, 128)) {
                let n = input.len();
                let weight = ones(n);
                let cfg = RmsNormConfig::default();
                let out = rms_norm_f32(&input, &weight, &cfg).unwrap();
                let rms = (out.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
                prop_assert!((rms - 1.0).abs() < 0.05, "rms = {rms}");
            }

            /// Fused residual+norm matches separate add-then-norm.
            #[test]
            fn prop_fused_matches_separate(
                input in arb_vec(4, 64),
            ) {
                let n = input.len();
                let residual: Vec<f32> = input.iter().map(|x| x * 0.1).collect();
                let cfg = default_ln_config(n);
                let gamma = ones(n);
                let beta = zeros(n);

                let fused = fused_layer_norm_residual(
                    &input, &residual, &gamma, &beta, &cfg,
                ).unwrap();

                let summed: Vec<f32> = input.iter().zip(&residual).map(|(a, b)| a + b).collect();
                let manual = layer_norm_f32(&summed, &gamma, &beta, &cfg).unwrap();

                for i in 0..n {
                    prop_assert!(
                        (fused[i] - manual[i]).abs() < 1e-3,
                        "mismatch at {i}: fused={} manual={}", fused[i], manual[i],
                    );
                }
            }

            /// compute_norm_stats RMS should always be non-negative.
            #[test]
            fn prop_stats_rms_non_negative(input in arb_vec(1, 128)) {
                let s = compute_norm_stats(&input);
                prop_assert!(s.rms >= 0.0);
                prop_assert!(s.variance >= -1e-5, "variance = {}", s.variance);
            }
        }
    }
}
