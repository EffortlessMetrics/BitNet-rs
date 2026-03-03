//! AVX-512 optimized softmax kernels with scalar fallback.
//!
//! Provides numerically stable softmax, log-softmax, in-place, 2-D row-wise,
//! online (single-pass), and masked variants.  When the runtime CPU supports
//! AVX-512F the hot loops are vectorized 16-wide; otherwise an equivalent
//! scalar path executes.
#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86_64")]
#[allow(clippy::wildcard_imports)]
use std::arch::x86_64::*;
use std::fmt;

// ── Configuration & error types ─────────────────────────────────────────

/// Configuration for softmax operations.
#[derive(Debug, Clone)]
pub struct SoftmaxConfig {
    /// Temperature scaling factor (must be > 0).
    pub temperature: f32,
    /// Expected input dimension (0 = unchecked).
    pub dim: usize,
    /// Whether to use numerically stable (max-subtraction) path.
    pub stable: bool,
}

impl Default for SoftmaxConfig {
    fn default() -> Self {
        Self { temperature: 1.0, dim: 0, stable: true }
    }
}

/// Errors that can occur during softmax computation.
#[derive(Debug, Clone, PartialEq)]
pub enum SoftmaxError {
    /// Input slice is empty.
    EmptyInput,
    /// Temperature is non-positive or NaN.
    InvalidTemperature(f32),
    /// Input length does not match the configured `dim`.
    DimensionMismatch { expected: usize, got: usize },
    /// Computation produced NaN or Inf.
    NumericalInstability,
}

impl fmt::Display for SoftmaxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "softmax input is empty"),
            Self::InvalidTemperature(t) => {
                write!(f, "invalid temperature: {t} (must be > 0)")
            }
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::NumericalInstability => {
                write!(f, "numerical instability detected (NaN/Inf)")
            }
        }
    }
}

impl std::error::Error for SoftmaxError {}

// ── Validation helpers ──────────────────────────────────────────────────

fn validate_config(len: usize, config: &SoftmaxConfig) -> Result<(), SoftmaxError> {
    if len == 0 {
        return Err(SoftmaxError::EmptyInput);
    }
    if config.temperature <= 0.0 || config.temperature.is_nan() {
        return Err(SoftmaxError::InvalidTemperature(config.temperature));
    }
    if config.dim != 0 && config.dim != len {
        return Err(SoftmaxError::DimensionMismatch { expected: config.dim, got: len });
    }
    Ok(())
}

fn check_numerical(data: &[f32]) -> Result<(), SoftmaxError> {
    if data.iter().any(|&v| v.is_nan() || v.is_infinite()) {
        return Err(SoftmaxError::NumericalInstability);
    }
    Ok(())
}

// ── Fast exp with clamping ──────────────────────────────────────────────

#[inline(always)]
fn fast_exp(x: f32) -> f32 {
    x.clamp(-88.0, 88.0).exp()
}

// ── AVX-512 helpers ─────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn hmax_avx512(v: __m512) -> f32 {
    // Reduce 16 → 8 → 4 → 2 → 1 using shuffle + max.
    let hi = _mm512_shuffle_f32x4(v, v, 0b_01_00_11_10); // swap 256-bit halves
    let m = _mm512_max_ps(v, hi);
    let hi2 = _mm512_shuffle_f32x4(m, m, 0b_00_00_01_01);
    let m2 = _mm512_max_ps(m, hi2);
    // Now 4 copies; extract low 128 and reduce.
    let lo128 = _mm512_castps512_ps128(m2);
    let hi64 = _mm_movehl_ps(lo128, lo128);
    let m3 = _mm_max_ps(lo128, hi64);
    let m4 = _mm_max_ss(m3, _mm_shuffle_ps(m3, m3, 1));
    _mm_cvtss_f32(m4)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn hsum_avx512(v: __m512) -> f32 {
    let hi = _mm512_shuffle_f32x4(v, v, 0b_01_00_11_10);
    let s = _mm512_add_ps(v, hi);
    let hi2 = _mm512_shuffle_f32x4(s, s, 0b_00_00_01_01);
    let s2 = _mm512_add_ps(s, hi2);
    let lo128 = _mm512_castps512_ps128(s2);
    let hi64 = _mm_movehl_ps(lo128, lo128);
    let s3 = _mm_add_ps(lo128, hi64);
    let s4 = _mm_add_ss(s3, _mm_shuffle_ps(s3, s3, 1));
    _mm_cvtss_f32(s4)
}

// ── Scalar core ─────────────────────────────────────────────────────────

fn scalar_max(data: &[f32]) -> f32 {
    data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

fn softmax_scalar_stable(input: &[f32], output: &mut [f32]) {
    let max_val = scalar_max(input);
    let mut sum = 0.0f32;
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        let e = fast_exp(x - max_val);
        *o = e;
        sum += e;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for o in output.iter_mut() {
            *o *= inv;
        }
    }
}

fn softmax_scalar_unstable(input: &[f32], output: &mut [f32]) {
    let mut sum = 0.0f32;
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        let e = fast_exp(x);
        *o = e;
        sum += e;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for o in output.iter_mut() {
            *o *= inv;
        }
    }
}

// ── AVX-512 softmax core ────────────────────────────────────────────────

/// # Safety
/// Caller must ensure AVX-512F is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn softmax_avx512_inner(input: &[f32], output: &mut [f32]) {
    let n = input.len();
    let chunks = n / 16;
    let inp = input.as_ptr();
    let outp = output.as_mut_ptr();

    // Pass 1: find max
    let mut vmax = _mm512_set1_ps(f32::NEG_INFINITY);
    for i in 0..chunks {
        let v = _mm512_loadu_ps(inp.add(i * 16));
        vmax = _mm512_max_ps(vmax, v);
    }
    let mut max_val = hmax_avx512(vmax);
    for i in (chunks * 16)..n {
        max_val = max_val.max(*inp.add(i));
    }

    // Pass 2: exp(x - max) and accumulate sum
    let vmax_bc = _mm512_set1_ps(max_val);
    let mut vsum = _mm512_setzero_ps();

    for i in 0..chunks {
        let v = _mm512_loadu_ps(inp.add(i * 16));
        let shifted = _mm512_sub_ps(v, vmax_bc);
        // Per-lane exp via scalar (compiler may auto-vectorise).
        let mut buf = [0.0f32; 16];
        _mm512_storeu_ps(buf.as_mut_ptr(), shifted);
        for b in &mut buf {
            *b = fast_exp(*b);
        }
        let exp_v = _mm512_loadu_ps(buf.as_ptr());
        _mm512_storeu_ps(outp.add(i * 16), exp_v);
        vsum = _mm512_add_ps(vsum, exp_v);
    }
    let mut sum_exp = hsum_avx512(vsum);
    for i in (chunks * 16)..n {
        let e = fast_exp(*inp.add(i) - max_val);
        *outp.add(i) = e;
        sum_exp += e;
    }

    // Pass 3: normalise
    if sum_exp > 0.0 {
        let inv = _mm512_set1_ps(1.0 / sum_exp);
        for i in 0..chunks {
            let v = _mm512_loadu_ps(outp.add(i * 16));
            _mm512_storeu_ps(outp.add(i * 16), _mm512_mul_ps(v, inv));
        }
        let inv_s = 1.0 / sum_exp;
        for i in (chunks * 16)..n {
            *outp.add(i) *= inv_s;
        }
    }
}

/// Runtime dispatch: prefer AVX-512 when available, else scalar.
fn softmax_dispatch(input: &[f32], output: &mut [f32], stable: bool) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && stable {
            // SAFETY: feature detection above.
            unsafe { softmax_avx512_inner(input, output) };
            return;
        }
    }
    if stable {
        softmax_scalar_stable(input, output);
    } else {
        softmax_scalar_unstable(input, output);
    }
}

// ── Public API ──────────────────────────────────────────────────────────

/// Numerically stable softmax with AVX-512 acceleration.
///
/// Returns a new vector containing the softmax of `input`.
pub fn softmax_f32(input: &[f32], config: &SoftmaxConfig) -> Result<Vec<f32>, SoftmaxError> {
    validate_config(input.len(), config)?;

    let scaled: Vec<f32>;
    let src = if (config.temperature - 1.0).abs() > f32::EPSILON {
        let inv_t = 1.0 / config.temperature;
        scaled = input.iter().map(|&x| x * inv_t).collect();
        &scaled
    } else {
        input
    };

    let mut output = vec![0.0f32; src.len()];
    softmax_dispatch(src, &mut output, config.stable);
    check_numerical(&output)?;
    Ok(output)
}

/// In-place numerically stable softmax with AVX-512 acceleration.
pub fn softmax_f32_inplace(data: &mut [f32], config: &SoftmaxConfig) -> Result<(), SoftmaxError> {
    validate_config(data.len(), config)?;

    if (config.temperature - 1.0).abs() > f32::EPSILON {
        let inv_t = 1.0 / config.temperature;
        for x in data.iter_mut() {
            *x *= inv_t;
        }
    }

    let input_copy = data.to_vec();
    softmax_dispatch(&input_copy, data, config.stable);
    check_numerical(data)?;
    Ok(())
}

/// Numerically stable log-softmax: `x_i - max - log(Σ exp(x_j - max))`.
pub fn log_softmax_f32(input: &[f32], config: &SoftmaxConfig) -> Result<Vec<f32>, SoftmaxError> {
    validate_config(input.len(), config)?;

    let scaled: Vec<f32>;
    let src = if (config.temperature - 1.0).abs() > f32::EPSILON {
        let inv_t = 1.0 / config.temperature;
        scaled = input.iter().map(|&x| x * inv_t).collect();
        &scaled
    } else {
        input
    };

    let max_val = scalar_max(src);
    let mut sum_exp = 0.0f32;
    for &x in src {
        sum_exp += fast_exp(x - max_val);
    }
    let log_sum_exp = max_val + sum_exp.ln();
    let output: Vec<f32> = src.iter().map(|&x| x - log_sum_exp).collect();
    check_numerical(&output)?;
    Ok(output)
}

/// Row-wise softmax on a 2-D matrix stored in row-major order.
pub fn softmax_2d(
    input: &[f32],
    rows: usize,
    cols: usize,
    config: &SoftmaxConfig,
) -> Result<Vec<f32>, SoftmaxError> {
    if rows == 0 || cols == 0 {
        return Err(SoftmaxError::EmptyInput);
    }
    if input.len() != rows * cols {
        return Err(SoftmaxError::DimensionMismatch { expected: rows * cols, got: input.len() });
    }
    if config.temperature <= 0.0 || config.temperature.is_nan() {
        return Err(SoftmaxError::InvalidTemperature(config.temperature));
    }

    let row_config =
        SoftmaxConfig { temperature: config.temperature, dim: 0, stable: config.stable };

    let mut output = vec![0.0f32; input.len()];
    for r in 0..rows {
        let start = r * cols;
        let end = start + cols;
        let row_out = softmax_f32(&input[start..end], &row_config)?;
        output[start..end].copy_from_slice(&row_out);
    }
    Ok(output)
}

/// Single-pass numerically stable (online) softmax.
///
/// Uses the online algorithm that tracks a running max and correction
/// factor so the output is produced in one scan without a separate
/// max-finding pass.
pub fn online_softmax_f32(input: &[f32]) -> Result<Vec<f32>, SoftmaxError> {
    if input.is_empty() {
        return Err(SoftmaxError::EmptyInput);
    }

    let mut running_max = f32::NEG_INFINITY;
    let mut running_sum = 0.0f32;

    for &x in input {
        if x > running_max {
            running_sum *= fast_exp(running_max - x);
            running_max = x;
        }
        running_sum += fast_exp(x - running_max);
    }

    let log_sum = running_max + running_sum.ln();
    let output: Vec<f32> = input.iter().map(|&x| fast_exp(x - log_sum)).collect();
    check_numerical(&output)?;
    Ok(output)
}

/// Masked softmax: positions where `mask[i]` is `false` are set to 0;
/// remaining positions receive a valid softmax distribution.
pub fn softmax_with_mask(
    input: &[f32],
    mask: &[bool],
    config: &SoftmaxConfig,
) -> Result<Vec<f32>, SoftmaxError> {
    validate_config(input.len(), config)?;
    if input.len() != mask.len() {
        return Err(SoftmaxError::DimensionMismatch { expected: input.len(), got: mask.len() });
    }

    // If no position is unmasked, every output is 0.
    if !mask.iter().any(|&m| m) {
        return Ok(vec![0.0; input.len()]);
    }

    let masked: Vec<f32> = input
        .iter()
        .zip(mask.iter())
        .map(|(&x, &m)| if m { x } else { f32::NEG_INFINITY })
        .collect();

    let mut result = softmax_f32(&masked, config)?;
    // Clamp masked positions to exact 0 (exp(-inf) may leave tiny residues).
    for (v, &m) in result.iter_mut().zip(mask.iter()) {
        if !m {
            *v = 0.0;
        }
    }
    Ok(result)
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> SoftmaxConfig {
        SoftmaxConfig::default()
    }

    fn cfg_with_temp(temperature: f32) -> SoftmaxConfig {
        SoftmaxConfig { temperature, ..Default::default() }
    }

    fn cfg_with_dim(dim: usize) -> SoftmaxConfig {
        SoftmaxConfig { dim, ..Default::default() }
    }

    fn cfg_unstable() -> SoftmaxConfig {
        SoftmaxConfig { stable: false, ..Default::default() }
    }

    // Helper: assert all elements roughly sum to 1.
    fn assert_sum_one(v: &[f32], tol: f32) {
        let s: f32 = v.iter().sum();
        assert!((s - 1.0).abs() < tol, "sum = {s}, expected ~1.0");
    }

    // Helper: assert element-wise close.
    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < tol, "index {i}: {x} vs {y}, diff={}", (x - y).abs());
        }
    }

    // ── softmax_f32 basic ───────────────────────────────────────────────

    #[test]
    fn test_softmax_basic() {
        let input = vec![1.0, 2.0, 3.0];
        let out = softmax_f32(&input, &default_cfg()).unwrap();
        assert_sum_one(&out, 1e-5);
        // Values should be monotonically increasing.
        assert!(out[0] < out[1]);
        assert!(out[1] < out[2]);
    }

    #[test]
    fn test_softmax_single_element() {
        let out = softmax_f32(&[42.0], &default_cfg()).unwrap();
        assert_eq!(out, vec![1.0]);
    }

    #[test]
    fn test_softmax_two_elements() {
        let out = softmax_f32(&[0.0, 0.0], &default_cfg()).unwrap();
        assert_close(&out, &[0.5, 0.5], 1e-6);
    }

    #[test]
    fn test_softmax_uniform_input() {
        let input = vec![5.0; 10];
        let out = softmax_f32(&input, &default_cfg()).unwrap();
        for &v in &out {
            assert!((v - 0.1).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_negative_inputs() {
        let input = vec![-1.0, -2.0, -3.0];
        let out = softmax_f32(&input, &default_cfg()).unwrap();
        assert_sum_one(&out, 1e-5);
        assert!(out[0] > out[1]);
    }

    #[test]
    fn test_softmax_large_values() {
        let input = vec![1000.0, 1001.0, 1002.0];
        let out = softmax_f32(&input, &default_cfg()).unwrap();
        assert_sum_one(&out, 1e-5);
    }

    #[test]
    fn test_softmax_very_negative_values() {
        let input = vec![-1000.0, -999.0, -998.0];
        let out = softmax_f32(&input, &default_cfg()).unwrap();
        assert_sum_one(&out, 1e-5);
    }

    #[test]
    fn test_softmax_mixed_extreme_values() {
        let input = vec![-100.0, 0.0, 100.0];
        let out = softmax_f32(&input, &default_cfg()).unwrap();
        assert_sum_one(&out, 1e-5);
        // The largest should dominate.
        assert!(out[2] > 0.99);
    }

    #[test]
    fn test_softmax_zeros() {
        let input = vec![0.0; 5];
        let out = softmax_f32(&input, &default_cfg()).unwrap();
        assert_sum_one(&out, 1e-6);
        for &v in &out {
            assert!((v - 0.2).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_length_16() {
        // Exactly one AVX-512 vector width.
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let out = softmax_f32(&input, &default_cfg()).unwrap();
        assert_sum_one(&out, 1e-5);
    }

    #[test]
    fn test_softmax_length_17() {
        // One full vector + 1 tail element.
        let input: Vec<f32> = (0..17).map(|i| i as f32).collect();
        let out = softmax_f32(&input, &default_cfg()).unwrap();
        assert_sum_one(&out, 1e-5);
    }

    #[test]
    fn test_softmax_length_32() {
        let input: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let out = softmax_f32(&input, &default_cfg()).unwrap();
        assert_sum_one(&out, 1e-5);
    }

    #[test]
    fn test_softmax_length_100() {
        let input: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.1).collect();
        let out = softmax_f32(&input, &default_cfg()).unwrap();
        assert_sum_one(&out, 1e-5);
    }

    #[test]
    fn test_softmax_all_negative_infinity() {
        // All -inf → all exp are 0 → sum is 0 → output remains 0.
        let input = vec![f32::NEG_INFINITY; 3];
        // With stable path, 0/0 = NaN → NumericalInstability.
        let result = softmax_f32(&input, &default_cfg());
        assert!(result.is_err());
    }

    // ── Error cases ─────────────────────────────────────────────────────

    #[test]
    fn test_softmax_empty_input() {
        let result = softmax_f32(&[], &default_cfg());
        assert_eq!(result, Err(SoftmaxError::EmptyInput));
    }

    #[test]
    fn test_softmax_invalid_temperature_zero() {
        let result = softmax_f32(&[1.0], &cfg_with_temp(0.0));
        assert!(matches!(result, Err(SoftmaxError::InvalidTemperature(_))));
    }

    #[test]
    fn test_softmax_invalid_temperature_negative() {
        let result = softmax_f32(&[1.0], &cfg_with_temp(-1.0));
        assert!(matches!(result, Err(SoftmaxError::InvalidTemperature(_))));
    }

    #[test]
    fn test_softmax_invalid_temperature_nan() {
        let result = softmax_f32(&[1.0], &cfg_with_temp(f32::NAN));
        assert!(matches!(result, Err(SoftmaxError::InvalidTemperature(_))));
    }

    #[test]
    fn test_softmax_dimension_mismatch() {
        let result = softmax_f32(&[1.0, 2.0], &cfg_with_dim(5));
        assert_eq!(result, Err(SoftmaxError::DimensionMismatch { expected: 5, got: 2 }));
    }

    #[test]
    fn test_softmax_dimension_zero_means_unchecked() {
        let result = softmax_f32(&[1.0, 2.0], &cfg_with_dim(0));
        assert!(result.is_ok());
    }

    // ── Temperature scaling ─────────────────────────────────────────────

    #[test]
    fn test_softmax_high_temperature() {
        // High temperature → near-uniform.
        let input = vec![1.0, 2.0, 3.0];
        let out = softmax_f32(&input, &cfg_with_temp(100.0)).unwrap();
        assert_sum_one(&out, 1e-5);
        let spread = out.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
            - out.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        assert!(spread < 0.01, "spread = {spread}");
    }

    #[test]
    fn test_softmax_low_temperature() {
        // Low temperature → peaky.
        let input = vec![1.0, 2.0, 3.0];
        let out = softmax_f32(&input, &cfg_with_temp(0.01)).unwrap();
        assert_sum_one(&out, 1e-5);
        assert!(out[2] > 0.99);
    }

    #[test]
    fn test_softmax_temperature_one_is_identity() {
        let input = vec![1.0, 2.0, 3.0];
        let a = softmax_f32(&input, &default_cfg()).unwrap();
        let b = softmax_f32(&input, &cfg_with_temp(1.0)).unwrap();
        assert_close(&a, &b, 1e-7);
    }

    // ── softmax_f32_inplace ─────────────────────────────────────────────

    #[test]
    fn test_inplace_basic() {
        let mut data = vec![1.0, 2.0, 3.0];
        softmax_f32_inplace(&mut data, &default_cfg()).unwrap();
        assert_sum_one(&data, 1e-5);
    }

    #[test]
    fn test_inplace_matches_out_of_place() {
        let input = vec![0.5, 1.5, -0.5, 2.0];
        let out = softmax_f32(&input, &default_cfg()).unwrap();
        let mut inplace = input.clone();
        softmax_f32_inplace(&mut inplace, &default_cfg()).unwrap();
        assert_close(&out, &inplace, 1e-6);
    }

    #[test]
    fn test_inplace_empty() {
        let result = softmax_f32_inplace(&mut [], &default_cfg());
        assert_eq!(result, Err(SoftmaxError::EmptyInput));
    }

    #[test]
    fn test_inplace_with_temperature() {
        let mut data = vec![1.0, 2.0, 3.0];
        softmax_f32_inplace(&mut data, &cfg_with_temp(0.5)).unwrap();
        assert_sum_one(&data, 1e-5);
        assert!(data[2] > data[1]);
    }

    // ── log_softmax_f32 ─────────────────────────────────────────────────

    #[test]
    fn test_log_softmax_basic() {
        let input = vec![1.0, 2.0, 3.0];
        let out = log_softmax_f32(&input, &default_cfg()).unwrap();
        // All values should be negative.
        for &v in &out {
            assert!(v <= 0.0, "log_softmax value {v} should be ≤ 0");
        }
        // exp(log_softmax) should sum to 1.
        let exp_sum: f32 = out.iter().map(|&v| v.exp()).sum();
        assert!((exp_sum - 1.0).abs() < 1e-5, "exp sum = {exp_sum}");
    }

    #[test]
    fn test_log_softmax_matches_log_of_softmax() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let sm = softmax_f32(&input, &default_cfg()).unwrap();
        let log_sm = log_softmax_f32(&input, &default_cfg()).unwrap();
        let log_of_sm: Vec<f32> = sm.iter().map(|v| v.ln()).collect();
        assert_close(&log_sm, &log_of_sm, 1e-5);
    }

    #[test]
    fn test_log_softmax_empty() {
        let result = log_softmax_f32(&[], &default_cfg());
        assert_eq!(result, Err(SoftmaxError::EmptyInput));
    }

    #[test]
    fn test_log_softmax_single() {
        let out = log_softmax_f32(&[5.0], &default_cfg()).unwrap();
        assert!((out[0]).abs() < 1e-6, "single element log softmax = {}", out[0]);
    }

    #[test]
    fn test_log_softmax_with_temperature() {
        let input = vec![1.0, 2.0, 3.0];
        let out = log_softmax_f32(&input, &cfg_with_temp(2.0)).unwrap();
        let exp_sum: f32 = out.iter().map(|&v| v.exp()).sum();
        assert!((exp_sum - 1.0).abs() < 1e-5);
    }

    // ── softmax_2d ──────────────────────────────────────────────────────

    #[test]
    fn test_softmax_2d_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = softmax_2d(&input, 2, 3, &default_cfg()).unwrap();
        // Each row should sum to 1.
        let row1: f32 = out[..3].iter().sum();
        let row2: f32 = out[3..].iter().sum();
        assert!((row1 - 1.0).abs() < 1e-5);
        assert!((row2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_2d_single_row() {
        let input = vec![1.0, 2.0, 3.0];
        let out = softmax_2d(&input, 1, 3, &default_cfg()).unwrap();
        let expected = softmax_f32(&input, &default_cfg()).unwrap();
        assert_close(&out, &expected, 1e-6);
    }

    #[test]
    fn test_softmax_2d_single_col() {
        let input = vec![1.0, 2.0, 3.0];
        let out = softmax_2d(&input, 3, 1, &default_cfg()).unwrap();
        // Each row has a single element → each must be 1.0.
        assert_close(&out, &[1.0, 1.0, 1.0], 1e-6);
    }

    #[test]
    fn test_softmax_2d_dimension_mismatch() {
        let result = softmax_2d(&[1.0, 2.0], 2, 2, &default_cfg());
        assert!(matches!(result, Err(SoftmaxError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_softmax_2d_empty_rows() {
        let result = softmax_2d(&[], 0, 5, &default_cfg());
        assert_eq!(result, Err(SoftmaxError::EmptyInput));
    }

    #[test]
    fn test_softmax_2d_with_temperature() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = softmax_2d(&input, 2, 3, &cfg_with_temp(0.5)).unwrap();
        let row1: f32 = out[..3].iter().sum();
        assert!((row1 - 1.0).abs() < 1e-5);
    }

    // ── online_softmax_f32 ──────────────────────────────────────────────

    #[test]
    fn test_online_softmax_basic() {
        let input = vec![1.0, 2.0, 3.0];
        let out = online_softmax_f32(&input).unwrap();
        assert_sum_one(&out, 1e-5);
    }

    #[test]
    fn test_online_softmax_matches_standard() {
        let input = vec![0.1, 0.5, -0.3, 1.2, 0.8];
        let standard = softmax_f32(&input, &default_cfg()).unwrap();
        let online = online_softmax_f32(&input).unwrap();
        assert_close(&standard, &online, 1e-5);
    }

    #[test]
    fn test_online_softmax_large_values() {
        let input = vec![1000.0, 1001.0, 1002.0];
        let out = online_softmax_f32(&input).unwrap();
        assert_sum_one(&out, 1e-3);
    }

    #[test]
    fn test_online_softmax_single() {
        let out = online_softmax_f32(&[7.0]).unwrap();
        assert_eq!(out, vec![1.0]);
    }

    #[test]
    fn test_online_softmax_empty() {
        let result = online_softmax_f32(&[]);
        assert_eq!(result, Err(SoftmaxError::EmptyInput));
    }

    // ── softmax_with_mask ───────────────────────────────────────────────

    #[test]
    fn test_mask_basic() {
        let input = vec![1.0, 2.0, 3.0];
        let mask = vec![true, true, true];
        let out = softmax_with_mask(&input, &mask, &default_cfg()).unwrap();
        let expected = softmax_f32(&input, &default_cfg()).unwrap();
        assert_close(&out, &expected, 1e-6);
    }

    #[test]
    fn test_mask_some_masked() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![true, false, true, false];
        let out = softmax_with_mask(&input, &mask, &default_cfg()).unwrap();
        assert_eq!(out[1], 0.0);
        assert_eq!(out[3], 0.0);
        let active_sum: f32 = out.iter().sum();
        assert!((active_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_mask_all_masked() {
        let input = vec![1.0, 2.0, 3.0];
        let mask = vec![false, false, false];
        let out = softmax_with_mask(&input, &mask, &default_cfg()).unwrap();
        assert_eq!(out, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_mask_single_unmasked() {
        let input = vec![1.0, 2.0, 3.0];
        let mask = vec![false, true, false];
        let out = softmax_with_mask(&input, &mask, &default_cfg()).unwrap();
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 1.0);
        assert_eq!(out[2], 0.0);
    }

    #[test]
    fn test_mask_length_mismatch() {
        let result = softmax_with_mask(&[1.0, 2.0], &[true], &default_cfg());
        assert!(matches!(result, Err(SoftmaxError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_mask_with_temperature() {
        let input = vec![1.0, 2.0, 3.0];
        let mask = vec![true, true, false];
        let out = softmax_with_mask(&input, &mask, &cfg_with_temp(0.5)).unwrap();
        assert_eq!(out[2], 0.0);
        let active_sum: f32 = out.iter().sum();
        assert!((active_sum - 1.0).abs() < 1e-5);
    }

    // ── Numerical stability ─────────────────────────────────────────────

    #[test]
    fn test_stability_large_positive_shift() {
        // Should not overflow despite large values.
        let input = vec![88.0, 89.0, 90.0];
        let out = softmax_f32(&input, &default_cfg()).unwrap();
        assert_sum_one(&out, 1e-5);
        assert!(!out.iter().any(|v| v.is_nan() || v.is_infinite()));
    }

    #[test]
    fn test_stability_large_negative() {
        let input = vec![-88.0, -89.0, -90.0];
        let out = softmax_f32(&input, &default_cfg()).unwrap();
        assert_sum_one(&out, 1e-5);
    }

    #[test]
    fn test_stability_wide_range() {
        let input = vec![-50.0, 0.0, 50.0];
        let out = softmax_f32(&input, &default_cfg()).unwrap();
        assert_sum_one(&out, 1e-5);
        assert!(out[2] > 0.999);
    }

    #[test]
    fn test_unstable_mode() {
        // Unstable mode should still work for moderate inputs.
        let input = vec![1.0, 2.0, 3.0];
        let out = softmax_f32(&input, &cfg_unstable()).unwrap();
        assert_sum_one(&out, 1e-5);
    }

    // ── SoftmaxConfig / SoftmaxError ────────────────────────────────────

    #[test]
    fn test_config_default() {
        let cfg = SoftmaxConfig::default();
        assert_eq!(cfg.temperature, 1.0);
        assert_eq!(cfg.dim, 0);
        assert!(cfg.stable);
    }

    #[test]
    fn test_error_display() {
        assert_eq!(SoftmaxError::EmptyInput.to_string(), "softmax input is empty");
        assert!(SoftmaxError::InvalidTemperature(-1.0).to_string().contains("-1"));
        assert!(SoftmaxError::DimensionMismatch { expected: 3, got: 5 }.to_string().contains("3"));
        assert!(SoftmaxError::NumericalInstability.to_string().contains("NaN"));
    }

    #[test]
    fn test_error_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(SoftmaxError::EmptyInput);
        let _ = err.to_string(); // Should compile.
    }

    #[test]
    fn test_config_clone_debug() {
        let cfg = SoftmaxConfig::default();
        let cfg2 = cfg.clone();
        assert_eq!(format!("{cfg:?}"), format!("{cfg2:?}"));
    }

    #[test]
    fn test_error_clone_eq() {
        let e1 = SoftmaxError::EmptyInput;
        let e2 = e1.clone();
        assert_eq!(e1, e2);
    }

    // ── proptest properties ─────────────────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        /// Generate a non-empty Vec<f32> of moderate values.
        fn arb_input() -> impl Strategy<Value = Vec<f32>> {
            proptest::collection::vec(-50.0f32..50.0, 1..=256)
        }

        proptest! {
            /// Softmax output always sums to 1 (within tolerance).
            #[test]
            fn prop_sum_to_one(input in arb_input()) {
                let out = softmax_f32(&input, &default_cfg()).unwrap();
                let s: f32 = out.iter().sum();
                prop_assert!((s - 1.0).abs() < 1e-4, "sum = {s}");
            }

            /// All output values are in [0, 1].
            #[test]
            fn prop_values_in_unit_interval(input in arb_input()) {
                let out = softmax_f32(&input, &default_cfg()).unwrap();
                for &v in &out {
                    prop_assert!(v >= 0.0 && v <= 1.0, "value {v} out of [0,1]");
                }
            }

            /// Softmax preserves the ordering of the input.
            #[test]
            fn prop_monotonicity(input in arb_input()) {
                let out = softmax_f32(&input, &default_cfg()).unwrap();
                for i in 0..input.len() {
                    for j in (i + 1)..input.len() {
                        if input[i] < input[j] {
                            prop_assert!(out[i] <= out[j] + 1e-6,
                                "monotonicity: out[{i}]={} > out[{j}]={}",
                                out[i], out[j]);
                        }
                    }
                }
            }

            /// online_softmax matches standard softmax.
            #[test]
            fn prop_online_matches_standard(input in arb_input()) {
                let standard = softmax_f32(&input, &default_cfg()).unwrap();
                let online = online_softmax_f32(&input).unwrap();
                for (i, (&a, &b)) in standard.iter().zip(online.iter()).enumerate() {
                    prop_assert!((a - b).abs() < 1e-4,
                        "mismatch at {i}: standard={a}, online={b}");
                }
            }

            /// Higher temperature → lower max probability (flatter).
            #[test]
            fn prop_temperature_flattening(
                input in proptest::collection::vec(-10.0f32..10.0, 2..=64),
                t_lo in 0.1f32..1.0,
                t_hi in 2.0f32..20.0,
            ) {
                let lo = softmax_f32(&input, &cfg_with_temp(t_lo)).unwrap();
                let hi = softmax_f32(&input, &cfg_with_temp(t_hi)).unwrap();
                let max_lo = lo.iter().copied().fold(0.0f32, f32::max);
                let max_hi = hi.iter().copied().fold(0.0f32, f32::max);
                prop_assert!(max_lo >= max_hi - 1e-5,
                    "low temp max {max_lo} < high temp max {max_hi}");
            }
        }
    }
}
