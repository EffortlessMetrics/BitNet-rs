//! ARM NEON-optimized RMSNorm kernels for Apple Silicon.
//!
//! Provides vectorized RMS normalization using NEON SIMD intrinsics on
//! AArch64. Includes fused RMSNorm+SiLU, fused RMSNorm+linear projection,
//! online/streaming computation for long sequences, and pre-norm/post-norm
//! transformer patterns. Processes 4 × f32 lanes at a time with scalar
//! fallback for remainder elements.
//!
//! Uses `vrsqrteq_f32` with Newton-Raphson refinement for fast reciprocal
//! square root, avoiding the latency of a full `sqrt` + division.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Lane count for `float32x4_t` NEON vectors.
const LANES: usize = 4;

/// Default epsilon for numerical stability.
pub const DEFAULT_EPS: f32 = 1e-5;

// ── Configuration ──────────────────────────────────────────────────

/// Configuration for RMSNorm operations.
#[derive(Debug, Clone, Copy)]
pub struct RmsNormConfig {
    /// Small constant added inside the square root for stability.
    pub eps: f32,
}

impl Default for RmsNormConfig {
    fn default() -> Self {
        Self { eps: DEFAULT_EPS }
    }
}

impl RmsNormConfig {
    /// Create a config with a custom epsilon.
    pub fn with_eps(eps: f32) -> Self {
        Self { eps }
    }
}

/// Accumulated statistics for online/streaming RMSNorm.
#[derive(Debug, Clone, Copy)]
pub struct RmsNormAccumulator {
    /// Running sum of squares.
    pub sum_sq: f64,
    /// Number of elements seen so far.
    pub count: usize,
}

impl Default for RmsNormAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl RmsNormAccumulator {
    /// Create a fresh accumulator.
    pub fn new() -> Self {
        Self { sum_sq: 0.0, count: 0 }
    }

    /// Current mean-of-squares estimate.
    pub fn mean_of_squares(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.sum_sq / self.count as f64 }
    }

    /// Current inverse-RMS value given an epsilon.
    pub fn inv_rms(&self, eps: f32) -> f32 {
        1.0 / ((self.mean_of_squares() as f32) + eps).sqrt()
    }
}

// ── Core: NEON fast inverse square root ────────────────────────────

/// Compute an approximate reciprocal square root of each lane using
/// `vrsqrteq_f32` followed by one Newton-Raphson refinement step.
///
/// Accuracy: ~22-bit mantissa (sufficient for f32 inference).
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn neon_fast_rsqrt(val: float32x4_t) -> float32x4_t {
    let est = vrsqrteq_f32(val);
    // Newton-Raphson: est' = est * (3 - val * est^2) / 2
    // NEON provides vrsqrtsq_f32(val, est*est) = (3 - val*est*est)/2
    let refine = vrsqrtsq_f32(vmulq_f32(val, est), est);
    vmulq_f32(est, refine)
}

// ── Core: compute sum-of-squares via NEON ──────────────────────────

/// Compute the sum of squares of `data` using NEON FMA.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_sum_of_squares(data: &[f32]) -> f32 {
    let n = data.len();
    let chunks = n / LANES;
    let remainder = n % LANES;

    // SAFETY: inside #[target_feature(enable = "neon")] function.
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let ptr = data.as_ptr();

        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * LANES));
            acc = vfmaq_f32(acc, v, v);
        }

        let mut sum: f32 = vaddvq_f32(acc);

        let tail = chunks * LANES;
        for i in 0..remainder {
            let x = data[tail + i];
            sum += x * x;
        }

        sum
    }
}

// ── 1. Core RMSNorm ────────────────────────────────────────────────

/// Compute RMS normalization with scale using NEON and fast reciprocal
/// square root via `vrsqrteq_f32` + Newton-Raphson.
///
/// `output[i] = gamma[i] * input[i] * rsqrt(mean(input²) + eps)`
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `output` or `gamma` length differs from `input`, or if
/// `input` is empty.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_rmsnorm(input: &[f32], output: &mut [f32], gamma: &[f32], eps: f32) {
    let n = input.len();
    assert!(n > 0, "input must not be empty");
    assert_eq!(output.len(), n, "output length mismatch");
    assert_eq!(gamma.len(), n, "gamma length mismatch");

    // SAFETY: inside #[target_feature(enable = "neon")] function.
    unsafe {
        let sq_sum = neon_sum_of_squares(input);
        let mean_sq = sq_sum / n as f32;

        // Use fast NEON rsqrt with Newton-Raphson refinement.
        let val = vdupq_n_f32(mean_sq + eps);
        let inv_rms_vec = neon_fast_rsqrt(val);
        // Extract lane 0 (all lanes are identical).
        let inv_rms = vgetq_lane_f32::<0>(inv_rms_vec);

        neon_scale_multiply(input, output, gamma, inv_rms);
    }
}

/// Compute RMSNorm with configurable epsilon.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_rmsnorm_with_config(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    config: &RmsNormConfig,
) {
    // SAFETY: forwarding to neon_rmsnorm with same preconditions.
    unsafe {
        neon_rmsnorm(input, output, gamma, config.eps);
    }
}

// ── 2. Fused RMSNorm + SiLU ───────────────────────────────────────

/// Scalar SiLU: x * sigmoid(x).
#[inline(always)]
fn scalar_silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Fused RMSNorm followed by SiLU activation.
///
/// `output[i] = silu(gamma[i] * input[i] * rsqrt(mean(input²) + eps))`
///
/// Fusing avoids an extra pass over the data and reduces memory traffic.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `output` or `gamma` length differs from `input`, or if
/// `input` is empty.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_rmsnorm_silu(input: &[f32], output: &mut [f32], gamma: &[f32], eps: f32) {
    let n = input.len();
    assert!(n > 0, "input must not be empty");
    assert_eq!(output.len(), n, "output length mismatch");
    assert_eq!(gamma.len(), n, "gamma length mismatch");

    // SAFETY: inside #[target_feature(enable = "neon")] function.
    unsafe {
        let sq_sum = neon_sum_of_squares(input);
        let mean_sq = sq_sum / n as f32;

        let val = vdupq_n_f32(mean_sq + eps);
        let inv_rms_vec = neon_fast_rsqrt(val);
        let inv_rms = vgetq_lane_f32::<0>(inv_rms_vec);

        // First pass: RMSNorm into output.
        neon_scale_multiply(input, output, gamma, inv_rms);
    }

    // Second pass: apply SiLU in-place (scalar; NEON lacks native exp).
    for v in output.iter_mut() {
        *v = scalar_silu(*v);
    }
}

// ── 3. Fused RMSNorm + linear projection ──────────────────────────

/// Fused RMSNorm followed by a linear projection (matrix-vector multiply).
///
/// Computes `proj_output = weight @ rmsnorm(input, gamma, eps)` where
/// `weight` has shape `[out_features, in_features]` in row-major order.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if dimensions are inconsistent:
/// - `gamma.len() != input.len()`
/// - `weight.len() != out_features * input.len()`
/// - `proj_output.len() != out_features`
/// - `input` is empty
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_rmsnorm_linear(
    input: &[f32],
    gamma: &[f32],
    weight: &[f32],
    out_features: usize,
    proj_output: &mut [f32],
    eps: f32,
) {
    let in_features = input.len();
    assert!(in_features > 0, "input must not be empty");
    assert_eq!(gamma.len(), in_features, "gamma length mismatch");
    assert_eq!(weight.len(), out_features * in_features, "weight length mismatch");
    assert_eq!(proj_output.len(), out_features, "proj_output length mismatch");

    // SAFETY: inside #[target_feature(enable = "neon")] function.
    unsafe {
        // Step 1: compute inv_rms.
        let sq_sum = neon_sum_of_squares(input);
        let mean_sq = sq_sum / in_features as f32;
        let val = vdupq_n_f32(mean_sq + eps);
        let inv_rms_vec = neon_fast_rsqrt(val);
        let inv_rms = vgetq_lane_f32::<0>(inv_rms_vec);

        // Step 2: for each output row, compute dot(weight_row, normed).
        let chunks = in_features / LANES;
        let remainder = in_features % LANES;
        let inv_rms_v = vdupq_n_f32(inv_rms);

        for row in 0..out_features {
            let w_ptr = weight.as_ptr().add(row * in_features);
            let in_ptr = input.as_ptr();
            let g_ptr = gamma.as_ptr();

            let mut dot_acc = vdupq_n_f32(0.0);

            for c in 0..chunks {
                let off = c * LANES;
                let x = vld1q_f32(in_ptr.add(off));
                let g = vld1q_f32(g_ptr.add(off));
                let w = vld1q_f32(w_ptr.add(off));

                // normed = x * inv_rms * g
                let normed = vmulq_f32(vmulq_f32(x, inv_rms_v), g);
                dot_acc = vfmaq_f32(dot_acc, w, normed);
            }

            let mut dot: f32 = vaddvq_f32(dot_acc);

            let tail = chunks * LANES;
            for i in 0..remainder {
                let idx = tail + i;
                let normed = input[idx] * inv_rms * gamma[idx];
                dot += weight[row * in_features + idx] * normed;
            }

            proj_output[row] = dot;
        }
    }
}

// ── 4. Online/streaming RMSNorm ────────────────────────────────────

/// Ingest a chunk of data into the streaming accumulator using NEON.
///
/// Call this repeatedly for each chunk of a long sequence, then use
/// [`neon_rmsnorm_online_finalize`] to normalize.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_rmsnorm_online_accumulate(acc: &mut RmsNormAccumulator, chunk: &[f32]) {
    // SAFETY: inside #[target_feature(enable = "neon")] function.
    let sq_sum = unsafe { neon_sum_of_squares(chunk) };
    acc.sum_sq += sq_sum as f64;
    acc.count += chunk.len();
}

/// Finalize streaming RMSNorm: normalize `input` using the accumulated
/// statistics.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `output` or `gamma` length differs from `input`, or if
/// the accumulator has seen zero elements.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_rmsnorm_online_finalize(
    acc: &RmsNormAccumulator,
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    eps: f32,
) {
    let n = input.len();
    assert!(acc.count > 0, "accumulator must have seen at least one element");
    assert_eq!(output.len(), n, "output length mismatch");
    assert_eq!(gamma.len(), n, "gamma length mismatch");

    let inv_rms = acc.inv_rms(eps);

    // SAFETY: forwarding to neon_scale_multiply with same preconditions.
    unsafe {
        neon_scale_multiply(input, output, gamma, inv_rms);
    }
}

// ── 5. Pre-norm and post-norm patterns ─────────────────────────────

/// Pre-norm pattern: RMSNorm **before** a sub-layer, with residual add.
///
/// `output[i] = sublayer_output[i] + input[i]`
///
/// where `normed[i] = gamma[i] * input[i] * inv_rms` is fed into the
/// sub-layer. This function computes only the normalization step and
/// writes `normed` into `output`; the caller applies the sub-layer and
/// adds the residual.
///
/// Used in LLaMA, Mistral, and most modern transformer architectures.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `output` or `gamma` length differs from `input`, or if
/// `input` is empty.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_rmsnorm_prenorm(input: &[f32], output: &mut [f32], gamma: &[f32], eps: f32) {
    // Pre-norm is just RMSNorm; the caller handles the residual.
    // SAFETY: forwarding to neon_rmsnorm with same preconditions.
    unsafe {
        neon_rmsnorm(input, output, gamma, eps);
    }
}

/// Post-norm pattern: residual add **then** RMSNorm.
///
/// `output[i] = gamma[i] * (input[i] + residual[i]) * inv_rms`
///
/// where `inv_rms = rsqrt(mean((input + residual)²) + eps)`.
///
/// Used in the original Transformer (Vaswani et al., 2017) and some
/// hybrid architectures.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `output`, `gamma`, or `residual` length differs from
/// `input`, or if `input` is empty.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_rmsnorm_postnorm(
    input: &[f32],
    residual: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    eps: f32,
) {
    let n = input.len();
    assert!(n > 0, "input must not be empty");
    assert_eq!(residual.len(), n, "residual length mismatch");
    assert_eq!(output.len(), n, "output length mismatch");
    assert_eq!(gamma.len(), n, "gamma length mismatch");

    // SAFETY: inside #[target_feature(enable = "neon")] function.
    unsafe {
        // Fused add + sum-of-squares in a single pass.
        let chunks = n / LANES;
        let remainder = n % LANES;
        let in_ptr = input.as_ptr();
        let res_ptr = residual.as_ptr();
        let out_ptr = output.as_mut_ptr();

        let mut sq_acc = vdupq_n_f32(0.0);

        // Pass 1: compute (input + residual), store in output, accumulate
        // sum of squares.
        for i in 0..chunks {
            let off = i * LANES;
            let a = vld1q_f32(in_ptr.add(off));
            let b = vld1q_f32(res_ptr.add(off));
            let sum = vaddq_f32(a, b);
            vst1q_f32(out_ptr.add(off), sum);
            sq_acc = vfmaq_f32(sq_acc, sum, sum);
        }

        let mut sq_sum: f32 = vaddvq_f32(sq_acc);
        let tail = chunks * LANES;
        for i in 0..remainder {
            let idx = tail + i;
            let s = input[idx] + residual[idx];
            output[idx] = s;
            sq_sum += s * s;
        }

        // Pass 2: scale by gamma * inv_rms.
        let mean_sq = sq_sum / n as f32;
        let val = vdupq_n_f32(mean_sq + eps);
        let inv_rms_vec = neon_fast_rsqrt(val);
        let inv_rms = vgetq_lane_f32::<0>(inv_rms_vec);

        // Re-use output as both source and destination.
        neon_scale_multiply_inplace(output, gamma, inv_rms);
    }
}

// ── NEON helpers ───────────────────────────────────────────────────

/// Apply `output = gamma * (input * inv_rms)` using NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_scale_multiply(input: &[f32], output: &mut [f32], gamma: &[f32], inv_rms: f32) {
    let n = input.len();
    let chunks = n / LANES;
    let remainder = n % LANES;

    // SAFETY: inside #[target_feature(enable = "neon")] function.
    unsafe {
        let inv_rms_vec = vdupq_n_f32(inv_rms);
        let in_ptr = input.as_ptr();
        let gam_ptr = gamma.as_ptr();
        let out_ptr = output.as_mut_ptr();

        for i in 0..chunks {
            let off = i * LANES;
            let v = vld1q_f32(in_ptr.add(off));
            let g = vld1q_f32(gam_ptr.add(off));

            let normed = vmulq_f32(v, inv_rms_vec);
            let scaled = vmulq_f32(g, normed);
            vst1q_f32(out_ptr.add(off), scaled);
        }
    }

    let tail = chunks * LANES;
    for i in 0..remainder {
        let idx = tail + i;
        output[idx] = gamma[idx] * (input[idx] * inv_rms);
    }
}

/// Apply `data[i] = gamma[i] * data[i] * inv_rms` in-place using NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_scale_multiply_inplace(data: &mut [f32], gamma: &[f32], inv_rms: f32) {
    let n = data.len();
    let chunks = n / LANES;
    let remainder = n % LANES;

    // SAFETY: inside #[target_feature(enable = "neon")] function.
    unsafe {
        let inv_rms_vec = vdupq_n_f32(inv_rms);
        let ptr = data.as_mut_ptr();
        let gam_ptr = gamma.as_ptr();

        for i in 0..chunks {
            let off = i * LANES;
            let v = vld1q_f32(ptr.add(off));
            let g = vld1q_f32(gam_ptr.add(off));

            let normed = vmulq_f32(v, inv_rms_vec);
            let scaled = vmulq_f32(g, normed);
            vst1q_f32(ptr.add(off), scaled);
        }
    }

    let tail = chunks * LANES;
    for i in 0..remainder {
        let idx = tail + i;
        data[idx] = gamma[idx] * (data[idx] * inv_rms);
    }
}

// ── Scalar references (test-only) ─────────────────────────────────

#[cfg(test)]
fn scalar_rmsnorm_ref(input: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    let mean_sq: f32 = input.iter().map(|x| x * x).sum::<f32>() / n as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    input.iter().enumerate().map(|(i, &x)| gamma[i] * x * inv_rms).collect()
}

#[cfg(test)]
fn scalar_rmsnorm_silu_ref(input: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let normed = scalar_rmsnorm_ref(input, gamma, eps);
    normed.iter().map(|&x| scalar_silu(x)).collect()
}

#[cfg(test)]
fn scalar_rmsnorm_linear_ref(
    input: &[f32],
    gamma: &[f32],
    weight: &[f32],
    out_features: usize,
    eps: f32,
) -> Vec<f32> {
    let in_features = input.len();
    let normed = scalar_rmsnorm_ref(input, gamma, eps);
    (0..out_features)
        .map(|row| {
            let row_start = row * in_features;
            normed.iter().enumerate().map(|(j, &n)| weight[row_start + j] * n).sum()
        })
        .collect()
}

#[cfg(test)]
fn scalar_rmsnorm_postnorm_ref(
    input: &[f32],
    residual: &[f32],
    gamma: &[f32],
    eps: f32,
) -> Vec<f32> {
    let combined: Vec<f32> = input.iter().zip(residual).map(|(&a, &b)| a + b).collect();
    scalar_rmsnorm_ref(&combined, gamma, eps)
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;
    // Fast rsqrt tolerance: Newton-Raphson gives ~22-bit accuracy.
    const TOL: f32 = 5e-4;
    const STRICT_TOL: f32 = 1e-5;

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

    // ── Core RMSNorm tests ─────────────────────────────────────────

    #[test]
    fn test_rmsnorm_basic_aligned() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = vec![1.0; 8];
        let expected = scalar_rmsnorm_ref(&input, &gamma, EPS);

        let mut output = vec![0.0; 8];
        unsafe { neon_rmsnorm(&input, &mut output, &gamma, EPS) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rmsnorm_with_gamma() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gamma = vec![0.5, 1.0, 1.5, 2.0, 0.1];
        let expected = scalar_rmsnorm_ref(&input, &gamma, EPS);

        let mut output = vec![0.0; 5];
        unsafe { neon_rmsnorm(&input, &mut output, &gamma, EPS) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rmsnorm_single_element() {
        let input = vec![42.0];
        let gamma = vec![2.0];
        let expected = scalar_rmsnorm_ref(&input, &gamma, EPS);

        let mut output = vec![0.0; 1];
        unsafe { neon_rmsnorm(&input, &mut output, &gamma, EPS) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rmsnorm_non_aligned_length() {
        // 13 elements: 3 NEON chunks + 1 scalar remainder.
        let input: Vec<f32> = (0..13).map(|i| (i as f32) + 1.0).collect();
        let gamma = vec![1.0; 13];
        let expected = scalar_rmsnorm_ref(&input, &gamma, EPS);

        let mut output = vec![0.0; 13];
        unsafe { neon_rmsnorm(&input, &mut output, &gamma, EPS) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rmsnorm_negative_values() {
        let input = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 8];
        let expected = scalar_rmsnorm_ref(&input, &gamma, EPS);

        let mut output = vec![0.0; 8];
        unsafe { neon_rmsnorm(&input, &mut output, &gamma, EPS) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rmsnorm_large_input() {
        let n = 1024;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let gamma = vec![1.0; n];
        let expected = scalar_rmsnorm_ref(&input, &gamma, EPS);

        let mut output = vec![0.0; n];
        unsafe { neon_rmsnorm(&input, &mut output, &gamma, EPS) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rmsnorm_uniform_input() {
        // All-same values: RMS = |val|, so output ≈ gamma * sign(val).
        let input = vec![3.0; 8];
        let gamma = vec![1.0; 8];
        let expected = scalar_rmsnorm_ref(&input, &gamma, EPS);

        let mut output = vec![0.0; 8];
        unsafe { neon_rmsnorm(&input, &mut output, &gamma, EPS) };
        assert_approx_eq(&output, &expected, TOL);
    }

    // ── Configurable epsilon ───────────────────────────────────────

    #[test]
    fn test_rmsnorm_custom_epsilon() {
        let input = vec![0.001, 0.002, 0.003, 0.004];
        let gamma = vec![1.0; 4];
        let large_eps = 1.0;
        let expected = scalar_rmsnorm_ref(&input, &gamma, large_eps);

        let config = RmsNormConfig::with_eps(large_eps);
        let mut output = vec![0.0; 4];
        unsafe { neon_rmsnorm_with_config(&input, &mut output, &gamma, &config) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rmsnorm_default_config() {
        let config = RmsNormConfig::default();
        assert_eq!(config.eps, DEFAULT_EPS);
    }

    // ── Fused RMSNorm + SiLU tests ────────────────────────────────

    #[test]
    fn test_rmsnorm_silu_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = vec![1.0; 8];
        let expected = scalar_rmsnorm_silu_ref(&input, &gamma, EPS);

        let mut output = vec![0.0; 8];
        unsafe { neon_rmsnorm_silu(&input, &mut output, &gamma, EPS) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rmsnorm_silu_non_aligned() {
        let input: Vec<f32> = (1..=7).map(|i| i as f32).collect();
        let gamma = vec![1.0; 7];
        let expected = scalar_rmsnorm_silu_ref(&input, &gamma, EPS);

        let mut output = vec![0.0; 7];
        unsafe { neon_rmsnorm_silu(&input, &mut output, &gamma, EPS) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rmsnorm_silu_negative_inputs() {
        let input = vec![-4.0, -2.0, 0.0, 2.0, 4.0];
        let gamma = vec![1.0; 5];
        let expected = scalar_rmsnorm_silu_ref(&input, &gamma, EPS);

        let mut output = vec![0.0; 5];
        unsafe { neon_rmsnorm_silu(&input, &mut output, &gamma, EPS) };
        assert_approx_eq(&output, &expected, TOL);
    }

    // ── Fused RMSNorm + linear tests ──────────────────────────────

    #[test]
    fn test_rmsnorm_linear_identity_weight() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        // Identity matrix.
        #[rustfmt::skip]
        let weight = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let expected = scalar_rmsnorm_ref(&input, &gamma, EPS);

        let mut proj_output = vec![0.0; 4];
        unsafe { neon_rmsnorm_linear(&input, &gamma, &weight, 4, &mut proj_output, EPS) };
        assert_approx_eq(&proj_output, &expected, TOL);
    }

    #[test]
    fn test_rmsnorm_linear_projection() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        // 2×4 weight projects from 4 → 2 features.
        let weight = vec![1.0, 0.5, 0.0, -0.5, 0.0, 1.0, -1.0, 0.0];
        let expected = scalar_rmsnorm_linear_ref(&input, &gamma, &weight, 2, EPS);

        let mut proj_output = vec![0.0; 2];
        unsafe { neon_rmsnorm_linear(&input, &gamma, &weight, 2, &mut proj_output, EPS) };
        assert_approx_eq(&proj_output, &expected, TOL);
    }

    #[test]
    fn test_rmsnorm_linear_large() {
        let in_features = 64;
        let out_features = 16;
        let input: Vec<f32> = (0..in_features).map(|i| (i as f32) * 0.1).collect();
        let gamma = vec![1.0; in_features];
        let weight: Vec<f32> =
            (0..out_features * in_features).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let expected = scalar_rmsnorm_linear_ref(&input, &gamma, &weight, out_features, EPS);

        let mut proj_output = vec![0.0; out_features];
        unsafe {
            neon_rmsnorm_linear(&input, &gamma, &weight, out_features, &mut proj_output, EPS)
        };
        assert_approx_eq(&proj_output, &expected, TOL);
    }

    // ── Online/streaming tests ─────────────────────────────────────

    #[test]
    fn test_online_rmsnorm_single_chunk() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = vec![1.0; 8];

        let mut acc = RmsNormAccumulator::new();
        unsafe { neon_rmsnorm_online_accumulate(&mut acc, &input) };
        assert_eq!(acc.count, 8);

        let expected = scalar_rmsnorm_ref(&input, &gamma, EPS);
        let mut output = vec![0.0; 8];
        unsafe { neon_rmsnorm_online_finalize(&acc, &input, &mut output, &gamma, EPS) };
        // Online uses f64 accumulation → higher precision than NEON
        // rsqrt, so compare against scalar reference with relaxed tol.
        assert_approx_eq(&output, &expected, STRICT_TOL);
    }

    #[test]
    fn test_online_rmsnorm_multi_chunk() {
        // Accumulate in 3 chunks, finalize on the full sequence.
        let full: Vec<f32> = (1..=24).map(|i| i as f32).collect();
        let gamma = vec![1.0; 24];

        let mut acc = RmsNormAccumulator::new();
        unsafe {
            neon_rmsnorm_online_accumulate(&mut acc, &full[0..8]);
            neon_rmsnorm_online_accumulate(&mut acc, &full[8..16]);
            neon_rmsnorm_online_accumulate(&mut acc, &full[16..24]);
        }
        assert_eq!(acc.count, 24);

        let expected = scalar_rmsnorm_ref(&full, &gamma, EPS);
        let mut output = vec![0.0; 24];
        unsafe { neon_rmsnorm_online_finalize(&acc, &full, &mut output, &gamma, EPS) };
        assert_approx_eq(&output, &expected, STRICT_TOL);
    }

    #[test]
    fn test_online_accumulator_mean_of_squares() {
        let data = vec![3.0, 4.0];
        let mut acc = RmsNormAccumulator::new();
        unsafe { neon_rmsnorm_online_accumulate(&mut acc, &data) };
        // mean_sq = (9 + 16) / 2 = 12.5
        let expected = 12.5_f64;
        assert!(
            (acc.mean_of_squares() - expected).abs() < 1e-10,
            "mean_of_squares: {} vs {expected}",
            acc.mean_of_squares()
        );
    }

    #[test]
    fn test_online_accumulator_empty() {
        let acc = RmsNormAccumulator::new();
        assert_eq!(acc.count, 0);
        assert_eq!(acc.mean_of_squares(), 0.0);
    }

    // ── Pre-norm / post-norm tests ─────────────────────────────────

    #[test]
    fn test_prenorm_matches_rmsnorm() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = vec![1.0; 8];

        let mut expected = vec![0.0; 8];
        unsafe { neon_rmsnorm(&input, &mut expected, &gamma, EPS) };

        let mut output = vec![0.0; 8];
        unsafe { neon_rmsnorm_prenorm(&input, &mut output, &gamma, EPS) };
        assert_approx_eq(&output, &expected, STRICT_TOL);
    }

    #[test]
    fn test_postnorm_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![0.5, -0.5, 1.0, -1.0];
        let gamma = vec![1.0; 4];
        let expected = scalar_rmsnorm_postnorm_ref(&input, &residual, &gamma, EPS);

        let mut output = vec![0.0; 4];
        unsafe { neon_rmsnorm_postnorm(&input, &residual, &mut output, &gamma, EPS) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_postnorm_large() {
        let n = 256;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02 - 2.0).collect();
        let residual: Vec<f32> = (0..n).map(|i| ((i * 3) as f32) * 0.01 - 1.0).collect();
        let gamma = vec![1.0; n];
        let expected = scalar_rmsnorm_postnorm_ref(&input, &residual, &gamma, EPS);

        let mut output = vec![0.0; n];
        unsafe { neon_rmsnorm_postnorm(&input, &residual, &mut output, &gamma, EPS) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_postnorm_zero_residual() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let residual = vec![0.0; 5];
        let gamma = vec![1.0; 5];
        let expected = scalar_rmsnorm_ref(&input, &gamma, EPS);

        let mut output = vec![0.0; 5];
        unsafe { neon_rmsnorm_postnorm(&input, &residual, &mut output, &gamma, EPS) };
        assert_approx_eq(&output, &expected, TOL);
    }

    // ── Fast rsqrt accuracy test ───────────────────────────────────

    #[test]
    fn test_fast_rsqrt_accuracy() {
        // Verify Newton-Raphson refined rsqrt is within tolerance.
        let vals = [0.25_f32, 1.0, 4.0, 16.0, 100.0, 0.01];
        for &v in &vals {
            let expected = 1.0 / v.sqrt();
            let result = unsafe {
                let vec = vdupq_n_f32(v);
                let r = neon_fast_rsqrt(vec);
                vgetq_lane_f32::<0>(r)
            };
            let rel_err = ((result - expected) / expected).abs();
            assert!(
                rel_err < 1e-3,
                "rsqrt({v}): got {result}, expected {expected}, \
                 rel_err {rel_err}"
            );
        }
    }

    // ── Parity test: NEON vs scalar ────────────────────────────────

    #[test]
    fn test_neon_vs_scalar_parity() {
        let n = 137;
        let input: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 100) as f32 * 0.1 - 5.0).collect();
        let gamma: Vec<f32> = (0..n).map(|i| 0.5 + (i % 5) as f32 * 0.2).collect();

        let expected = scalar_rmsnorm_ref(&input, &gamma, EPS);

        let mut output = vec![0.0; n];
        unsafe { neon_rmsnorm(&input, &mut output, &gamma, EPS) };
        assert_approx_eq(&output, &expected, TOL);
    }

    // ── Ignored tests with justification ───────────────────────────

    #[test]
    #[ignore = "Slow: benchmarks 4096-dim RMSNorm throughput; \
                run manually with --ignored"]
    fn bench_rmsnorm_4096_dim() {
        let n = 4096;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let gamma = vec![1.0; n];
        let mut output = vec![0.0; n];

        for _ in 0..10_000 {
            unsafe {
                neon_rmsnorm(&input, &mut output, &gamma, EPS);
            }
        }
    }

    #[test]
    #[ignore = "Slow: benchmarks fused RMSNorm+linear throughput; \
                run manually with --ignored"]
    fn bench_rmsnorm_linear_throughput() {
        let in_features = 4096;
        let out_features = 4096;
        let input: Vec<f32> = (0..in_features).map(|i| (i as f32) * 0.001).collect();
        let gamma = vec![1.0; in_features];
        let weight: Vec<f32> = vec![0.01; out_features * in_features];
        let mut proj_output = vec![0.0; out_features];

        for _ in 0..100 {
            unsafe {
                neon_rmsnorm_linear(&input, &gamma, &weight, out_features, &mut proj_output, EPS);
            }
        }
    }

    #[test]
    #[ignore = "TDD scaffold: requires f16 NEON (FHMA) support \
                for half-precision RMSNorm"]
    fn test_rmsnorm_f16() {
        panic!("not yet implemented: f16 RMSNorm via NEON FHMA");
    }
}
