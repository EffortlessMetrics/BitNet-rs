//! ARM NEON-optimized LayerNorm and RMSNorm kernels for Apple Silicon.
//!
//! Provides vectorized layer normalization and RMS normalization using
//! NEON SIMD intrinsics on AArch64. Processes 4 × f32 lanes at a time
//! with scalar fallback for remainder elements.

/// Compute layer normalization with affine parameters using NEON intrinsics.
///
/// Normalizes `input` to zero mean and unit variance, then applies
/// `output[i] = gamma[i] * normalized[i] + beta[i]`.
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
pub unsafe fn layernorm_neon(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) {
    let n = input.len();
    assert_eq!(output.len(), n, "output length mismatch");
    assert_eq!(gamma.len(), n, "gamma length mismatch");
    assert_eq!(beta.len(), n, "beta length mismatch");

    if n == 0 {
        return;
    }

    let mean = neon_mean(input);
    let variance = neon_variance(input, mean);
    let inv_std = 1.0 / (variance + eps).sqrt();

    neon_normalize_affine(input, output, gamma, beta, mean, inv_std);
}

/// Compute RMS normalization with scale using NEON intrinsics.
///
/// `output[i] = gamma[i] * input[i] / rms` where
/// `rms = sqrt(mean(input²) + eps)`.
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
pub unsafe fn rmsnorm_neon(input: &[f32], output: &mut [f32], gamma: &[f32], eps: f32) {
    let n = input.len();
    assert_eq!(output.len(), n, "output length mismatch");
    assert_eq!(gamma.len(), n, "gamma length mismatch");

    if n == 0 {
        return;
    }

    let mean_sq = neon_mean_of_squares(input);
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();

    neon_normalize_scale(input, output, gamma, inv_rms);
}

// ── NEON helpers ───────────────────────────────────────────────────

/// Compute the arithmetic mean of `data` using NEON horizontal adds.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_mean(data: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = data.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum_vec = unsafe { vdupq_n_f32(0.0) };
    let ptr = data.as_ptr();

    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * 4)) };
        sum_vec = unsafe { vaddq_f32(sum_vec, v) };
    }

    // Horizontal reduction: sum all 4 lanes.
    let mut sum: f32 = unsafe { vaddvq_f32(sum_vec) };

    // Scalar tail.
    let tail_start = chunks * 4;
    for i in 0..remainder {
        sum += data[tail_start + i];
    }

    sum / n as f32
}

/// Compute variance around `mean` using NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_variance(data: &[f32], mean: f32) -> f32 {
    use std::arch::aarch64::*;

    let n = data.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mean_vec = unsafe { vdupq_n_f32(mean) };
    let mut acc = unsafe { vdupq_n_f32(0.0) };
    let ptr = data.as_ptr();

    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * 4)) };
        let diff = unsafe { vsubq_f32(v, mean_vec) };
        acc = unsafe { vfmaq_f32(acc, diff, diff) };
    }

    let mut var_sum: f32 = unsafe { vaddvq_f32(acc) };

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let d = data[tail_start + i] - mean;
        var_sum += d * d;
    }

    var_sum / n as f32
}

/// Compute `mean(x²)` for RMSNorm using NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_mean_of_squares(data: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = data.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut acc = unsafe { vdupq_n_f32(0.0) };
    let ptr = data.as_ptr();

    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * 4)) };
        acc = unsafe { vfmaq_f32(acc, v, v) };
    }

    let mut sq_sum: f32 = unsafe { vaddvq_f32(acc) };

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let x = data[tail_start + i];
        sq_sum += x * x;
    }

    sq_sum / n as f32
}

/// Apply `output = gamma * ((input - mean) * inv_std) + beta` using NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_normalize_affine(
    input: &[f32],
    output: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    mean: f32,
    inv_std: f32,
) {
    use std::arch::aarch64::*;

    let n = input.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mean_vec = unsafe { vdupq_n_f32(mean) };
    let inv_std_vec = unsafe { vdupq_n_f32(inv_std) };
    let in_ptr = input.as_ptr();
    let gam_ptr = gamma.as_ptr();
    let bet_ptr = beta.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        let v = unsafe { vld1q_f32(in_ptr.add(off)) };
        let g = unsafe { vld1q_f32(gam_ptr.add(off)) };
        let b = unsafe { vld1q_f32(bet_ptr.add(off)) };

        let centered = unsafe { vsubq_f32(v, mean_vec) };
        let normed = unsafe { vmulq_f32(centered, inv_std_vec) };
        let scaled = unsafe { vfmaq_f32(b, g, normed) }; // b + g * normed
        unsafe { vst1q_f32(out_ptr.add(off), scaled) };
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        let normed = (input[idx] - mean) * inv_std;
        output[idx] = gamma[idx] * normed + beta[idx];
    }
}

/// Apply `output = gamma * (input * inv_rms)` using NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_normalize_scale(input: &[f32], output: &mut [f32], gamma: &[f32], inv_rms: f32) {
    use std::arch::aarch64::*;

    let n = input.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let inv_rms_vec = unsafe { vdupq_n_f32(inv_rms) };
    let in_ptr = input.as_ptr();
    let gam_ptr = gamma.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        let v = unsafe { vld1q_f32(in_ptr.add(off)) };
        let g = unsafe { vld1q_f32(gam_ptr.add(off)) };

        let normed = unsafe { vmulq_f32(v, inv_rms_vec) };
        let scaled = unsafe { vmulq_f32(g, normed) };
        unsafe { vst1q_f32(out_ptr.add(off), scaled) };
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        output[idx] = gamma[idx] * (input[idx] * inv_rms);
    }
}

// ── Scalar reference (test-only) ──────────────────────────────────

#[cfg(test)]
fn scalar_layernorm(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    let mean: f32 = input.iter().sum::<f32>() / n as f32;
    let var: f32 = input.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    input.iter().enumerate().map(|(i, &x)| gamma[i] * (x - mean) * inv_std + beta[i]).collect()
}

#[cfg(test)]
fn scalar_rmsnorm(input: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    let mean_sq: f32 = input.iter().map(|x| x * x).sum::<f32>() / n as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    input.iter().enumerate().map(|(i, &x)| gamma[i] * x * inv_rms).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;
    const TOL: f32 = 1e-5;

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

    // ── LayerNorm tests ────────────────────────────────────────────

    #[test]
    fn test_layernorm_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let expected = scalar_layernorm(&input, &gamma, &beta, EPS);

        let mut output = vec![0.0; 8];
        unsafe { layernorm_neon(&input, &mut output, &gamma, &beta, EPS) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_layernorm_with_affine() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gamma = vec![0.5, 1.0, 1.5, 2.0, 0.1];
        let beta = vec![0.1, -0.1, 0.0, 0.5, -0.5];
        let expected = scalar_layernorm(&input, &gamma, &beta, EPS);

        let mut output = vec![0.0; 5];
        unsafe { layernorm_neon(&input, &mut output, &gamma, &beta, EPS) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_layernorm_zero_variance() {
        let input = vec![3.0; 8];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];

        let mut output = vec![0.0; 8];
        unsafe { layernorm_neon(&input, &mut output, &gamma, &beta, EPS) };

        // All values identical → zero variance → output ≈ beta.
        for &v in &output {
            assert!(v.abs() < TOL, "expected ~0 with zero variance, got {v}");
        }
    }

    #[test]
    fn test_layernorm_single_element() {
        let input = vec![42.0];
        let gamma = vec![2.0];
        let beta = vec![1.0];
        let expected = scalar_layernorm(&input, &gamma, &beta, EPS);

        let mut output = vec![0.0; 1];
        unsafe { layernorm_neon(&input, &mut output, &gamma, &beta, EPS) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_layernorm_large_input() {
        let n = 1024;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let gamma = vec![1.0; n];
        let beta = vec![0.0; n];
        let expected = scalar_layernorm(&input, &gamma, &beta, EPS);

        let mut output = vec![0.0; n];
        unsafe { layernorm_neon(&input, &mut output, &gamma, &beta, EPS) };
        assert_approx_eq(&output, &expected, 1e-4);
    }

    #[test]
    fn test_layernorm_non_aligned_length() {
        // 13 elements → 3 NEON chunks + 1 scalar remainder.
        let input: Vec<f32> = (0..13).map(|i| i as f32).collect();
        let gamma = vec![1.0; 13];
        let beta = vec![0.0; 13];
        let expected = scalar_layernorm(&input, &gamma, &beta, EPS);

        let mut output = vec![0.0; 13];
        unsafe { layernorm_neon(&input, &mut output, &gamma, &beta, EPS) };
        assert_approx_eq(&output, &expected, TOL);
    }

    // ── RMSNorm tests ──────────────────────────────────────────────

    #[test]
    fn test_rmsnorm_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = vec![1.0; 8];
        let expected = scalar_rmsnorm(&input, &gamma, EPS);

        let mut output = vec![0.0; 8];
        unsafe { rmsnorm_neon(&input, &mut output, &gamma, EPS) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rmsnorm_with_scale() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gamma = vec![0.5, 1.0, 1.5, 2.0, 0.1];
        let expected = scalar_rmsnorm(&input, &gamma, EPS);

        let mut output = vec![0.0; 5];
        unsafe { rmsnorm_neon(&input, &mut output, &gamma, EPS) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rmsnorm_single_element() {
        let input = vec![42.0];
        let gamma = vec![2.0];
        let expected = scalar_rmsnorm(&input, &gamma, EPS);

        let mut output = vec![0.0; 1];
        unsafe { rmsnorm_neon(&input, &mut output, &gamma, EPS) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_rmsnorm_large_input() {
        let n = 1024;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let gamma = vec![1.0; n];
        let expected = scalar_rmsnorm(&input, &gamma, EPS);

        let mut output = vec![0.0; n];
        unsafe { rmsnorm_neon(&input, &mut output, &gamma, EPS) };
        assert_approx_eq(&output, &expected, 1e-4);
    }

    // ── Parity tests ──────────────────────────────────────────────

    #[test]
    fn test_neon_vs_scalar_layernorm_parity() {
        // Non-power-of-two length with non-trivial affine params.
        let n = 137;
        let input: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 100) as f32 * 0.1 - 5.0).collect();
        let gamma: Vec<f32> = (0..n).map(|i| 0.5 + (i % 5) as f32 * 0.2).collect();
        let beta: Vec<f32> = (0..n).map(|i| -0.3 + (i % 3) as f32 * 0.1).collect();

        let expected = scalar_layernorm(&input, &gamma, &beta, EPS);

        let mut output = vec![0.0; n];
        unsafe { layernorm_neon(&input, &mut output, &gamma, &beta, EPS) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_neon_vs_scalar_rmsnorm_parity() {
        let n = 137;
        let input: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 100) as f32 * 0.1 - 5.0).collect();
        let gamma: Vec<f32> = (0..n).map(|i| 0.5 + (i % 5) as f32 * 0.2).collect();

        let expected = scalar_rmsnorm(&input, &gamma, EPS);

        let mut output = vec![0.0; n];
        unsafe { rmsnorm_neon(&input, &mut output, &gamma, EPS) };
        assert_approx_eq(&output, &expected, TOL);
    }
}
