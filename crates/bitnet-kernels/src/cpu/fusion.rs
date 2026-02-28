//! CPU kernel fusion module
//!
//! Combines multiple small operations into fused kernels for better cache
//! utilization and throughput.  Each fused operation produces results
//! numerically equivalent (within floating-point tolerance) to executing
//! the constituent operations separately, but avoids intermediate
//! allocations and extra memory passes.

use std::fmt;

// ── Configuration ──────────────────────────────────────────────────

/// Controls which fused kernels are eligible for dispatch.
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Fuse RMSNorm + linear projection into a single pass.
    pub enable_rmsnorm_linear: bool,
    /// Fuse GELU activation + linear projection.
    pub enable_gelu_linear: bool,
    /// Fuse masking + scaling + softmax.
    pub enable_softmax_mask: bool,
    /// Minimum element count before fusion is attempted.
    /// Inputs smaller than this fall back to separate ops.
    pub min_fusion_size: usize,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            enable_rmsnorm_linear: true,
            enable_gelu_linear: true,
            enable_softmax_mask: true,
            min_fusion_size: 32,
        }
    }
}

impl FusionConfig {
    /// All fusions disabled – useful as a comparison baseline.
    pub fn disabled() -> Self {
        Self {
            enable_rmsnorm_linear: false,
            enable_gelu_linear: false,
            enable_softmax_mask: false,
            min_fusion_size: usize::MAX,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), FusionError> {
        if self.min_fusion_size == 0 {
            return Err(FusionError::InvalidConfig("min_fusion_size must be > 0".into()));
        }
        Ok(())
    }
}

// ── FusedOp enum ───────────────────────────────────────────────────

/// Identifies a specific fused operation for logging / profiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusedOp {
    RmsNormLinear,
    GeluLinear,
    SoftmaxMask,
    AddNormalize,
    ScaleAndAdd,
}

impl fmt::Display for FusedOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RmsNormLinear => write!(f, "RmsNormLinear"),
            Self::GeluLinear => write!(f, "GeluLinear"),
            Self::SoftmaxMask => write!(f, "SoftmaxMask"),
            Self::AddNormalize => write!(f, "AddNormalize"),
            Self::ScaleAndAdd => write!(f, "ScaleAndAdd"),
        }
    }
}

// ── Errors ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum FusionError {
    DimensionMismatch { expected: usize, got: usize },
    InvalidConfig(String),
    EmptyInput,
}

impl fmt::Display for FusionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::EmptyInput => write!(f, "empty input"),
        }
    }
}

impl std::error::Error for FusionError {}

// ── Fused kernels ──────────────────────────────────────────────────

/// Fused RMSNorm + linear projection.
///
/// Equivalent to:
/// ```text
/// normed  = input * gamma / rms(input, eps)
/// output  = normed · weight          (matrix-vector product)
/// ```
/// but computed in a single pass over `input`.
///
/// * `input`  – `[n]`  row vector
/// * `weight` – `[out_dim × n]`  weight matrix (row-major)
/// * `gamma`  – `[n]`  per-element scale
/// * `eps`    – RMSNorm epsilon
///
/// Returns `[out_dim]`.
pub fn fused_rmsnorm_linear(
    input: &[f32],
    weight: &[f32],
    gamma: &[f32],
    eps: f32,
) -> Result<Vec<f32>, FusionError> {
    let n = input.len();
    if n == 0 {
        return Err(FusionError::EmptyInput);
    }
    if gamma.len() != n {
        return Err(FusionError::DimensionMismatch { expected: n, got: gamma.len() });
    }
    if !weight.len().is_multiple_of(n) {
        return Err(FusionError::DimensionMismatch { expected: n, got: weight.len() % n });
    }

    let out_dim = weight.len() / n;

    // RMS = sqrt( mean(x²) + eps )
    let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    let mut output = vec![0.0f32; out_dim];
    for (o, row) in output.iter_mut().zip(weight.chunks_exact(n)) {
        let mut acc = 0.0f32;
        for ((&w, &x), &g) in row.iter().zip(input).zip(gamma) {
            acc += w * (x * g * inv_rms);
        }
        *o = acc;
    }
    Ok(output)
}

/// Fused GELU activation + linear projection.
///
/// Equivalent to:
/// ```text
/// activated = gelu(input)
/// output    = activated · weight  +  bias
/// ```
///
/// * `input`  – `[n]`
/// * `weight` – `[out_dim × n]`  (row-major)
/// * `bias`   – `[out_dim]` (or empty for no bias)
///
/// Returns `[out_dim]`.
pub fn fused_gelu_linear(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
) -> Result<Vec<f32>, FusionError> {
    let n = input.len();
    if n == 0 {
        return Err(FusionError::EmptyInput);
    }
    if !weight.len().is_multiple_of(n) {
        return Err(FusionError::DimensionMismatch { expected: n, got: weight.len() % n });
    }

    let out_dim = weight.len() / n;
    if !bias.is_empty() && bias.len() != out_dim {
        return Err(FusionError::DimensionMismatch { expected: out_dim, got: bias.len() });
    }

    // Pre-compute GELU(input) once, then reuse for every output row.
    let activated: Vec<f32> = input.iter().map(|&x| gelu(x)).collect();

    let mut output = vec![0.0f32; out_dim];
    for (i, row) in weight.chunks_exact(n).enumerate() {
        let mut acc = 0.0f32;
        for (&w, &a) in row.iter().zip(&activated) {
            acc += w * a;
        }
        if !bias.is_empty() {
            acc += bias[i];
        }
        output[i] = acc;
    }
    Ok(output)
}

/// Fused masking + scaling + softmax.
///
/// Equivalent to:
/// ```text
/// masked = scores + mask   (additive mask; -inf for masked positions)
/// scaled = masked * scale
/// output = softmax(scaled)
/// ```
///
/// * `scores` – `[n]`
/// * `mask`   – `[n]`  additive mask (0 = keep, large negative = mask out)
/// * `scale`  – scalar multiplier
///
/// Returns `[n]`.
pub fn fused_softmax_mask(
    scores: &[f32],
    mask: &[f32],
    scale: f32,
) -> Result<Vec<f32>, FusionError> {
    let n = scores.len();
    if n == 0 {
        return Err(FusionError::EmptyInput);
    }
    if mask.len() != n {
        return Err(FusionError::DimensionMismatch { expected: n, got: mask.len() });
    }

    // Fuse mask + scale + max-find in a single pass.
    let mut max_val = f32::NEG_INFINITY;
    for (&s, &m) in scores.iter().zip(mask) {
        let v = (s + m) * scale;
        if v > max_val {
            max_val = v;
        }
    }

    // Second pass: exp(x - max) and running sum.
    let mut output = vec![0.0f32; n];
    let mut sum = 0.0f32;
    for ((&s, &m), o) in scores.iter().zip(mask).zip(&mut output) {
        let v = ((s + m) * scale - max_val).exp();
        *o = v;
        sum += v;
    }

    // Normalise.
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for o in &mut output {
            *o *= inv;
        }
    }
    Ok(output)
}

/// Fused residual addition + RMSNorm (layer norm variant).
///
/// Equivalent to:
/// ```text
/// combined = a + b
/// output   = combined * gamma / rms(combined, eps)
/// ```
///
/// * `a`, `b` – `[n]`
/// * `gamma`  – `[n]`
/// * `eps`    – normalization epsilon
///
/// Returns `[n]`.
pub fn fused_add_normalize(
    a: &[f32],
    b: &[f32],
    gamma: &[f32],
    eps: f32,
) -> Result<Vec<f32>, FusionError> {
    let n = a.len();
    if n == 0 {
        return Err(FusionError::EmptyInput);
    }
    if b.len() != n {
        return Err(FusionError::DimensionMismatch { expected: n, got: b.len() });
    }
    if gamma.len() != n {
        return Err(FusionError::DimensionMismatch { expected: n, got: gamma.len() });
    }

    // Single pass: accumulate sum-of-squares while building combined.
    let mut sum_sq = 0.0f32;
    let combined: Vec<f32> = a
        .iter()
        .zip(b)
        .map(|(&ai, &bi)| {
            let c = ai + bi;
            sum_sq += c * c;
            c
        })
        .collect();

    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    let output: Vec<f32> = combined.iter().zip(gamma).map(|(&c, &g)| c * g * inv_rms).collect();
    Ok(output)
}

/// Fused scale-and-add: `output = a + scale * b`.
///
/// * `a`, `b` – `[n]`
/// * `scale`  – scalar multiplier for `b`
///
/// Returns `[n]`.
pub fn fused_scale_add(a: &[f32], b: &[f32], scale: f32) -> Result<Vec<f32>, FusionError> {
    let n = a.len();
    if n == 0 {
        return Err(FusionError::EmptyInput);
    }
    if b.len() != n {
        return Err(FusionError::DimensionMismatch { expected: n, got: b.len() });
    }

    let output: Vec<f32> = a.iter().zip(b).map(|(&ai, &bi)| ai + scale * bi).collect();
    Ok(output)
}

/// Performance comparison helper: returns elapsed time in seconds for a
/// closure.  Used by tests to compare fused vs separate implementations.
pub fn benchmark_fused_vs_separate<F: FnOnce()>(f: F) -> f64 {
    let start = std::time::Instant::now();
    f();
    start.elapsed().as_secs_f64()
}

// ── Internal helpers ───────────────────────────────────────────────

/// Scalar GELU approximation (tanh variant).
#[inline]
fn gelu(x: f32) -> f32 {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    const SQRT_2_OVER_PI: f32 = 0.797_884_6; // sqrt(2/π)
    const COEFF: f32 = 0.044_715;
    let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

// ── Reference (unfused) implementations for testing ────────────────

/// Separate RMSNorm then linear, for correctness comparison.
#[cfg(test)]
fn reference_rmsnorm_linear(input: &[f32], weight: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    let out_dim = weight.len() / n;

    // Step 1: RMSNorm
    let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let normed: Vec<f32> = input.iter().zip(gamma).map(|(&x, &g)| x * g / rms).collect();

    // Step 2: linear
    let mut output = vec![0.0f32; out_dim];
    for (o, row) in output.iter_mut().zip(weight.chunks_exact(n)) {
        *o = row.iter().zip(&normed).map(|(&w, &x)| w * x).sum();
    }
    output
}

/// Separate GELU then linear, for correctness comparison.
#[cfg(test)]
fn reference_gelu_linear(input: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let n = input.len();
    let out_dim = weight.len() / n;

    let activated: Vec<f32> = input.iter().map(|&x| gelu(x)).collect();

    let mut output = vec![0.0f32; out_dim];
    for (i, row) in weight.chunks_exact(n).enumerate() {
        let mut acc: f32 = row.iter().zip(&activated).map(|(&w, &a)| w * a).sum();
        if !bias.is_empty() {
            acc += bias[i];
        }
        output[i] = acc;
    }
    output
}

/// Separate mask + scale + softmax, for correctness comparison.
#[cfg(test)]
fn reference_softmax_mask(scores: &[f32], mask: &[f32], scale: f32) -> Vec<f32> {
    let scaled: Vec<f32> = scores.iter().zip(mask).map(|(&s, &m)| (s + m) * scale).collect();
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 { exps.iter().map(|&e| e / sum).collect() } else { exps }
}

/// Separate add + RMSNorm, for correctness comparison.
#[cfg(test)]
fn reference_add_normalize(a: &[f32], b: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let combined: Vec<f32> = a.iter().zip(b).map(|(&x, &y)| x + y).collect();
    let n = combined.len();
    let sum_sq: f32 = combined.iter().map(|&x| x * x).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt();
    combined.iter().zip(gamma).map(|(&c, &g)| c * g / rms).collect()
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;
    const TOL: f32 = 1e-5;

    fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    // ── FusionConfig ───────────────────────────────────────────────

    #[test]
    fn config_default_enables_all() {
        let cfg = FusionConfig::default();
        assert!(cfg.enable_rmsnorm_linear);
        assert!(cfg.enable_gelu_linear);
        assert!(cfg.enable_softmax_mask);
        assert_eq!(cfg.min_fusion_size, 32);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_disabled() {
        let cfg = FusionConfig::disabled();
        assert!(!cfg.enable_rmsnorm_linear);
        assert!(!cfg.enable_gelu_linear);
        assert!(!cfg.enable_softmax_mask);
        assert_eq!(cfg.min_fusion_size, usize::MAX);
    }

    #[test]
    fn config_zero_min_fusion_size_rejected() {
        let cfg = FusionConfig { min_fusion_size: 0, ..FusionConfig::default() };
        assert!(cfg.validate().is_err());
    }

    // ── FusedOp Display ────────────────────────────────────────────

    #[test]
    fn fused_op_display() {
        assert_eq!(FusedOp::RmsNormLinear.to_string(), "RmsNormLinear");
        assert_eq!(FusedOp::GeluLinear.to_string(), "GeluLinear");
        assert_eq!(FusedOp::SoftmaxMask.to_string(), "SoftmaxMask");
        assert_eq!(FusedOp::AddNormalize.to_string(), "AddNormalize");
        assert_eq!(FusedOp::ScaleAndAdd.to_string(), "ScaleAndAdd");
    }

    // ── fused_rmsnorm_linear ───────────────────────────────────────

    #[test]
    fn rmsnorm_linear_matches_reference() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        // 2×4 weight matrix → out_dim = 2
        let weight = vec![0.5, -0.5, 0.25, 0.1, 0.1, 0.2, 0.3, 0.4];
        let fused = fused_rmsnorm_linear(&input, &weight, &gamma, EPS).unwrap();
        let reference = reference_rmsnorm_linear(&input, &weight, &gamma, EPS);
        assert!(max_abs_error(&fused, &reference) < TOL, "fused {fused:?} vs ref {reference:?}");
    }

    #[test]
    fn rmsnorm_linear_single_element() {
        let input = vec![3.0];
        let gamma = vec![1.0];
        let weight = vec![2.0];
        let fused = fused_rmsnorm_linear(&input, &weight, &gamma, EPS).unwrap();
        let reference = reference_rmsnorm_linear(&input, &weight, &gamma, EPS);
        assert!(max_abs_error(&fused, &reference) < TOL);
    }

    #[test]
    fn rmsnorm_linear_zero_input() {
        let input = vec![0.0; 4];
        let gamma = vec![1.0; 4];
        let weight = vec![1.0; 8];
        let fused = fused_rmsnorm_linear(&input, &weight, &gamma, EPS).unwrap();
        // All zeros normed → all zeros output.
        for &v in &fused {
            assert!(v.abs() < TOL);
        }
    }

    #[test]
    fn rmsnorm_linear_empty_input() {
        let r = fused_rmsnorm_linear(&[], &[1.0], &[], 1e-5);
        assert_eq!(r.unwrap_err(), FusionError::EmptyInput);
    }

    #[test]
    fn rmsnorm_linear_gamma_mismatch() {
        let r = fused_rmsnorm_linear(&[1.0, 2.0], &[1.0; 4], &[1.0], EPS);
        assert!(matches!(r, Err(FusionError::DimensionMismatch { .. })));
    }

    #[test]
    fn rmsnorm_linear_large_input() {
        let n = 256;
        let out_dim = 64;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let gamma = vec![1.0f32; n];
        let weight: Vec<f32> = (0..out_dim * n).map(|i| (i as f32) * 0.001).collect();
        let fused = fused_rmsnorm_linear(&input, &weight, &gamma, EPS).unwrap();
        let reference = reference_rmsnorm_linear(&input, &weight, &gamma, EPS);
        assert!(max_abs_error(&fused, &reference) < 1e-3);
    }

    // ── fused_gelu_linear ──────────────────────────────────────────

    #[test]
    fn gelu_linear_matches_reference() {
        let input = vec![1.0, -1.0, 0.5, -0.5];
        let weight = vec![0.2, 0.3, 0.4, 0.5, -0.1, 0.6, 0.7, -0.2];
        let bias = vec![0.01, -0.01];
        let fused = fused_gelu_linear(&input, &weight, &bias).unwrap();
        let reference = reference_gelu_linear(&input, &weight, &bias);
        assert!(max_abs_error(&fused, &reference) < TOL, "fused {fused:?} vs ref {reference:?}");
    }

    #[test]
    fn gelu_linear_no_bias() {
        let input = vec![1.0, 2.0];
        let weight = vec![0.5, 0.5, -0.5, -0.5];
        let fused = fused_gelu_linear(&input, &weight, &[]).unwrap();
        let reference = reference_gelu_linear(&input, &weight, &[]);
        assert!(max_abs_error(&fused, &reference) < TOL);
    }

    #[test]
    fn gelu_linear_zero_input() {
        let input = vec![0.0; 4];
        let weight = vec![1.0; 8];
        let bias = vec![0.5, 0.5];
        let fused = fused_gelu_linear(&input, &weight, &bias).unwrap();
        // GELU(0) = 0 → output = bias
        for (v, b) in fused.iter().zip(&bias) {
            assert!((v - b).abs() < TOL, "{v} vs {b}");
        }
    }

    #[test]
    fn gelu_linear_single_element() {
        let input = vec![2.0];
        let weight = vec![1.0];
        let bias = vec![0.0];
        let fused = fused_gelu_linear(&input, &weight, &bias).unwrap();
        let expected = gelu(2.0);
        assert!((fused[0] - expected).abs() < TOL);
    }

    #[test]
    fn gelu_linear_empty_input() {
        assert_eq!(fused_gelu_linear(&[], &[], &[]).unwrap_err(), FusionError::EmptyInput,);
    }

    #[test]
    fn gelu_linear_bias_mismatch() {
        let r = fused_gelu_linear(&[1.0], &[1.0], &[1.0, 2.0]);
        assert!(matches!(r, Err(FusionError::DimensionMismatch { .. })));
    }

    // ── fused_softmax_mask ─────────────────────────────────────────

    #[test]
    fn softmax_mask_matches_reference() {
        let scores = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![0.0, 0.0, -1e9, 0.0];
        let scale = 0.5;
        let fused = fused_softmax_mask(&scores, &mask, scale).unwrap();
        let reference = reference_softmax_mask(&scores, &mask, scale);
        assert!(max_abs_error(&fused, &reference) < TOL, "fused {fused:?} vs ref {reference:?}");
    }

    #[test]
    fn softmax_mask_sums_to_one() {
        let scores = vec![2.0, 1.0, 0.5, 3.0];
        let mask = vec![0.0; 4];
        let result = fused_softmax_mask(&scores, &mask, 1.0).unwrap();
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < TOL, "softmax sum = {sum}");
    }

    #[test]
    fn softmax_mask_fully_masked() {
        let scores = vec![1.0, 2.0, 3.0];
        let mask = vec![-1e30; 3];
        let result = fused_softmax_mask(&scores, &mask, 1.0).unwrap();
        // All masked → uniform after softmax (all exp(-big) ≈ 0 → 0/0 guarded).
        for &v in &result {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn softmax_mask_single_element() {
        let result = fused_softmax_mask(&[5.0], &[0.0], 1.0).unwrap();
        assert!((result[0] - 1.0).abs() < TOL);
    }

    #[test]
    fn softmax_mask_empty_input() {
        assert_eq!(fused_softmax_mask(&[], &[], 1.0).unwrap_err(), FusionError::EmptyInput,);
    }

    #[test]
    fn softmax_mask_dimension_mismatch() {
        let r = fused_softmax_mask(&[1.0, 2.0], &[0.0], 1.0);
        assert!(matches!(r, Err(FusionError::DimensionMismatch { .. })));
    }

    #[test]
    fn softmax_mask_numerical_stability() {
        // Large values that would overflow without max-subtraction.
        let scores = vec![1000.0, 1001.0, 1002.0];
        let mask = vec![0.0; 3];
        let result = fused_softmax_mask(&scores, &mask, 1.0).unwrap();
        assert!(result.iter().all(|v| v.is_finite()), "must be finite: {result:?}");
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < TOL);
    }

    // ── fused_add_normalize ────────────────────────────────────────

    #[test]
    fn add_normalize_matches_reference() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.5, -0.5, 0.25, -0.25];
        let gamma = vec![1.0; 4];
        let fused = fused_add_normalize(&a, &b, &gamma, EPS).unwrap();
        let reference = reference_add_normalize(&a, &b, &gamma, EPS);
        assert!(max_abs_error(&fused, &reference) < TOL, "fused {fused:?} vs ref {reference:?}");
    }

    #[test]
    fn add_normalize_single_element() {
        let fused = fused_add_normalize(&[3.0], &[1.0], &[1.0], EPS).unwrap();
        let reference = reference_add_normalize(&[3.0], &[1.0], &[1.0], EPS);
        assert!(max_abs_error(&fused, &reference) < TOL);
    }

    #[test]
    fn add_normalize_zero_residual() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0; 3];
        let gamma = vec![1.0; 3];
        let fused = fused_add_normalize(&a, &b, &gamma, EPS).unwrap();
        let reference = reference_add_normalize(&a, &b, &gamma, EPS);
        assert!(max_abs_error(&fused, &reference) < TOL);
    }

    #[test]
    fn add_normalize_empty_input() {
        assert_eq!(fused_add_normalize(&[], &[], &[], EPS).unwrap_err(), FusionError::EmptyInput,);
    }

    #[test]
    fn add_normalize_b_mismatch() {
        let r = fused_add_normalize(&[1.0, 2.0], &[1.0], &[1.0, 2.0], EPS);
        assert!(matches!(r, Err(FusionError::DimensionMismatch { .. })));
    }

    #[test]
    fn add_normalize_gamma_mismatch() {
        let r = fused_add_normalize(&[1.0], &[1.0], &[1.0, 2.0], EPS);
        assert!(matches!(r, Err(FusionError::DimensionMismatch { .. })));
    }

    // ── fused_scale_add ────────────────────────────────────────────

    #[test]
    fn scale_add_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = fused_scale_add(&a, &b, 0.5).unwrap();
        let expected = vec![3.0, 4.5, 6.0]; // a + 0.5*b
        assert!(max_abs_error(&result, &expected) < TOL);
    }

    #[test]
    fn scale_add_zero_scale() {
        let a = vec![1.0, 2.0];
        let b = vec![100.0, 200.0];
        let result = fused_scale_add(&a, &b, 0.0).unwrap();
        assert!(max_abs_error(&result, &a) < TOL);
    }

    #[test]
    fn scale_add_negative_scale() {
        let a = vec![10.0, 20.0];
        let b = vec![10.0, 20.0];
        let result = fused_scale_add(&a, &b, -1.0).unwrap();
        assert!(max_abs_error(&result, &[0.0, 0.0]) < TOL);
    }

    #[test]
    fn scale_add_single_element() {
        let result = fused_scale_add(&[3.0], &[4.0], 2.0).unwrap();
        assert!((result[0] - 11.0).abs() < TOL);
    }

    #[test]
    fn scale_add_empty_input() {
        assert_eq!(fused_scale_add(&[], &[], 1.0).unwrap_err(), FusionError::EmptyInput,);
    }

    #[test]
    fn scale_add_dimension_mismatch() {
        let r = fused_scale_add(&[1.0, 2.0], &[1.0], 1.0);
        assert!(matches!(r, Err(FusionError::DimensionMismatch { .. })));
    }

    // ── Various input sizes ────────────────────────────────────────

    #[test]
    fn scale_add_various_sizes() {
        for &n in &[1, 7, 8, 15, 16, 31, 32, 64, 128, 256, 1024] {
            let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..n).map(|i| -(i as f32)).collect();
            let result = fused_scale_add(&a, &b, 1.0).unwrap();
            assert!(result.iter().all(|&v| v.abs() < TOL), "failed for n={n}");
        }
    }

    #[test]
    fn rmsnorm_linear_various_sizes() {
        for &n in &[1, 4, 16, 64] {
            let input: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();
            let gamma = vec![1.0f32; n];
            let weight: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
            let fused = fused_rmsnorm_linear(&input, &weight, &gamma, EPS).unwrap();
            let reference = reference_rmsnorm_linear(&input, &weight, &gamma, EPS);
            assert!(max_abs_error(&fused, &reference) < 1e-3, "failed for n={n}");
        }
    }

    // ── Benchmark helper ───────────────────────────────────────────

    #[test]
    fn benchmark_helper_returns_positive_duration() {
        let elapsed = benchmark_fused_vs_separate(|| {
            let _sum: f32 = (0..1000).map(|i| (i as f32).sqrt()).sum();
        });
        assert!(elapsed > 0.0);
    }

    #[test]
    fn benchmark_fused_vs_separate_rmsnorm() {
        let n = 512;
        let out_dim = 128;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let gamma = vec![1.0f32; n];
        let weight: Vec<f32> = (0..out_dim * n).map(|i| (i as f32) * 0.001).collect();

        let fused_time = benchmark_fused_vs_separate(|| {
            for _ in 0..100 {
                let _ = fused_rmsnorm_linear(&input, &weight, &gamma, EPS);
            }
        });

        let separate_time = benchmark_fused_vs_separate(|| {
            for _ in 0..100 {
                let _ = reference_rmsnorm_linear(&input, &weight, &gamma, EPS);
            }
        });

        // Just ensure both run without panicking; in debug mode fused may
        // not be faster, so we only check they produce valid times.
        assert!(fused_time > 0.0);
        assert!(separate_time > 0.0);
    }
}
