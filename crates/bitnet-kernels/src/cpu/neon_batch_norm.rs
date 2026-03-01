//! ARM NEON-optimized batch normalization kernels for Apple Silicon.
//!
//! Provides vectorized batch normalization and instance normalization using
//! NEON SIMD intrinsics on AArch64. Processes 4 × f32 lanes at a time
//! with scalar fallback for remainder elements.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Compute batch normalization using NEON intrinsics.
///
/// `output[i] = gamma[i] * (input[i] - mean[i]) / sqrt(variance[i] + epsilon) + beta[i]`
///
/// All parameter slices must have the same length as `input`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if any slice length differs from `input`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_batch_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    variance: &[f32],
    epsilon: f32,
    output: &mut [f32],
) {
    let n = input.len();
    assert_eq!(gamma.len(), n, "gamma length mismatch");
    assert_eq!(beta.len(), n, "beta length mismatch");
    assert_eq!(mean.len(), n, "mean length mismatch");
    assert_eq!(variance.len(), n, "variance length mismatch");
    assert_eq!(output.len(), n, "output length mismatch");

    if n == 0 {
        return;
    }

    // SAFETY: we are inside a #[target_feature(enable = "neon")] function.
    unsafe {
        neon_bn_core(input, gamma, beta, mean, variance, epsilon, output);
    }
}

/// In-place batch normalization using NEON intrinsics.
///
/// Equivalent to [`neon_batch_norm`] but writes results back into `data`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if any parameter slice length differs from `data`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_batch_norm_inplace(
    data: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    variance: &[f32],
    epsilon: f32,
) {
    let n = data.len();
    assert_eq!(gamma.len(), n, "gamma length mismatch");
    assert_eq!(beta.len(), n, "beta length mismatch");
    assert_eq!(mean.len(), n, "mean length mismatch");
    assert_eq!(variance.len(), n, "variance length mismatch");

    if n == 0 {
        return;
    }

    // Copy input so we can read original values while writing output.
    let input_copy = data.to_vec();

    // SAFETY: we are inside a #[target_feature(enable = "neon")] function.
    unsafe {
        neon_bn_core(&input_copy, gamma, beta, mean, variance, epsilon, data);
    }
}

/// Compute instance normalization using NEON intrinsics.
///
/// Input is a flat buffer of shape `[channels, height, width]`. For each
/// channel, the mean and variance are computed over the spatial dimensions
/// (`height × width`), then an affine transform is applied per-channel:
///
/// `output[c*H*W + i] = gamma[c] * (input[c*H*W + i] - μ_c) / sqrt(σ²_c + ε) + beta[c]`
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `input.len() != channels * height * width`, or if `gamma`/`beta`
/// length differs from `channels`, or if `height * width == 0`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_instance_norm(
    input: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    gamma: &[f32],
    beta: &[f32],
    epsilon: f32,
    output: &mut [f32],
) {
    let spatial = height * width;
    assert!(spatial > 0, "spatial dimensions must be non-zero");
    assert_eq!(input.len(), channels * spatial, "input length mismatch");
    assert_eq!(output.len(), channels * spatial, "output length mismatch");
    assert_eq!(gamma.len(), channels, "gamma length mismatch");
    assert_eq!(beta.len(), channels, "beta length mismatch");

    // SAFETY: we are inside a #[target_feature(enable = "neon")] function.
    unsafe {
        for c in 0..channels {
            let start = c * spatial;
            let ch_input = &input[start..start + spatial];
            let ch_output = &mut output[start..start + spatial];

            let ch_mean = neon_sum(ch_input) / spatial as f32;
            let ch_var = neon_sum_of_squared_diffs(ch_input, ch_mean) / spatial as f32;
            let inv_std = 1.0 / (ch_var + epsilon).sqrt();

            neon_normalize_affine_broadcast(
                ch_input, ch_output, gamma[c], beta[c], ch_mean, inv_std,
            );
        }
    }
}

// ── NEON helpers ───────────────────────────────────────────────────

/// Core batch normalization with per-element statistics using NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_bn_core(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    variance: &[f32],
    epsilon: f32,
    output: &mut [f32],
) {
    let n = input.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // SAFETY: all intrinsics require neon which this fn guarantees.
    unsafe {
        let eps_vec = vdupq_n_f32(epsilon);
        let half = vdupq_n_f32(0.5);
        let three = vdupq_n_f32(3.0);

        for i in 0..chunks {
            let off = i * 4;
            let x = vld1q_f32(input.as_ptr().add(off));
            let g = vld1q_f32(gamma.as_ptr().add(off));
            let b = vld1q_f32(beta.as_ptr().add(off));
            let m = vld1q_f32(mean.as_ptr().add(off));
            let v = vld1q_f32(variance.as_ptr().add(off));

            let v_eps = vaddq_f32(v, eps_vec);

            // vrsqrteq_f32 + one Newton-Raphson refinement step.
            let est = vrsqrteq_f32(v_eps);
            let est_sq = vmulq_f32(est, est);
            let muls = vmulq_f32(v_eps, est_sq);
            let sub = vsubq_f32(three, muls);
            let inv_std = vmulq_f32(vmulq_f32(half, est), sub);

            // gamma * (input - mean) * inv_std + beta
            let centered = vsubq_f32(x, m);
            let normed = vmulq_f32(centered, inv_std);
            let scaled = vaddq_f32(vmulq_f32(g, normed), b);
            vst1q_f32(output.as_mut_ptr().add(off), scaled);
        }
    }

    // Scalar fallback for remainder elements.
    let tail = chunks * 4;
    for i in 0..remainder {
        let idx = tail + i;
        let inv_std = 1.0 / (variance[idx] + epsilon).sqrt();
        output[idx] = gamma[idx] * (input[idx] - mean[idx]) * inv_std + beta[idx];
    }
}

/// Sum all elements using NEON horizontal adds.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_sum(data: &[f32]) -> f32 {
    let n = data.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // SAFETY: all intrinsics require neon which this fn guarantees.
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let ptr = data.as_ptr();

        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * 4));
            acc = vaddq_f32(acc, v);
        }

        let mut sum: f32 = vaddvq_f32(acc);

        let tail = chunks * 4;
        for i in 0..remainder {
            sum += data[tail + i];
        }

        sum
    }
}

/// Sum of squared differences from `center` using NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_sum_of_squared_diffs(data: &[f32], center: f32) -> f32 {
    let n = data.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // SAFETY: all intrinsics require neon which this fn guarantees.
    unsafe {
        let center_vec = vdupq_n_f32(center);
        let mut acc = vdupq_n_f32(0.0);
        let ptr = data.as_ptr();

        for i in 0..chunks {
            let v = vld1q_f32(ptr.add(i * 4));
            let diff = vsubq_f32(v, center_vec);
            acc = vaddq_f32(acc, vmulq_f32(diff, diff));
        }

        let mut var_sum: f32 = vaddvq_f32(acc);

        let tail = chunks * 4;
        for i in 0..remainder {
            let d = data[tail + i] - center;
            var_sum += d * d;
        }

        var_sum
    }
}

/// Apply `output[i] = gamma * (input[i] - mean) * inv_std + beta` using
/// NEON with broadcast scalar affine parameters.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_normalize_affine_broadcast(
    input: &[f32],
    output: &mut [f32],
    gamma: f32,
    beta: f32,
    mean: f32,
    inv_std: f32,
) {
    let n = input.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // SAFETY: all intrinsics require neon which this fn guarantees.
    unsafe {
        let mean_vec = vdupq_n_f32(mean);
        let inv_std_vec = vdupq_n_f32(inv_std);
        let g_vec = vdupq_n_f32(gamma);
        let b_vec = vdupq_n_f32(beta);

        for i in 0..chunks {
            let off = i * 4;
            let x = vld1q_f32(input.as_ptr().add(off));
            let centered = vsubq_f32(x, mean_vec);
            let normed = vmulq_f32(centered, inv_std_vec);
            let scaled = vaddq_f32(vmulq_f32(g_vec, normed), b_vec);
            vst1q_f32(output.as_mut_ptr().add(off), scaled);
        }
    }

    let tail = chunks * 4;
    for i in 0..remainder {
        let idx = tail + i;
        output[idx] = gamma * (input[idx] - mean) * inv_std + beta;
    }
}

// ── Scalar references (test-only) ─────────────────────────────────

#[cfg(test)]
fn scalar_batch_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    variance: &[f32],
    epsilon: f32,
) -> Vec<f32> {
    input
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let inv_std = 1.0 / (variance[i] + epsilon).sqrt();
            gamma[i] * (x - mean[i]) * inv_std + beta[i]
        })
        .collect()
}

#[cfg(test)]
fn scalar_instance_norm(
    input: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    gamma: &[f32],
    beta: &[f32],
    epsilon: f32,
) -> Vec<f32> {
    let spatial = height * width;
    let mut output = vec![0.0; input.len()];
    for c in 0..channels {
        let start = c * spatial;
        let ch = &input[start..start + spatial];
        let mean: f32 = ch.iter().sum::<f32>() / spatial as f32;
        let var: f32 = ch.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / spatial as f32;
        let inv_std = 1.0 / (var + epsilon).sqrt();
        for i in 0..spatial {
            output[start + i] = gamma[c] * (ch[i] - mean) * inv_std + beta[c];
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;
    // Tolerance accounts for vrsqrteq_f32 + Newton-Raphson approximation.
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

    // ── Batch normalization tests ─────────────────────────────────

    #[test]
    fn test_batch_norm_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let mean = vec![0.0; 8];
        let variance = vec![1.0; 8];
        let expected = scalar_batch_norm(&input, &gamma, &beta, &mean, &variance, EPS);

        let mut output = vec![0.0; 8];
        unsafe { neon_batch_norm(&input, &gamma, &beta, &mean, &variance, EPS, &mut output) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_batch_norm_with_affine() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = vec![0.5, 1.0, 1.5, 2.0, 0.1, 0.3, 0.7, 1.2];
        let beta = vec![0.1, -0.1, 0.0, 0.5, -0.5, 0.2, -0.3, 0.0];
        let mean = vec![1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];
        let variance = vec![0.5, 1.0, 2.0, 0.1, 4.0, 0.3, 1.5, 0.8];
        let expected = scalar_batch_norm(&input, &gamma, &beta, &mean, &variance, EPS);

        let mut output = vec![0.0; 8];
        unsafe { neon_batch_norm(&input, &gamma, &beta, &mean, &variance, EPS, &mut output) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_batch_norm_non_aligned() {
        // 13 elements → 3 NEON chunks + 1 scalar remainder.
        let input: Vec<f32> = (0..13).map(|i| i as f32 * 0.5).collect();
        let gamma = vec![1.0; 13];
        let beta = vec![0.0; 13];
        let mean: Vec<f32> = (0..13).map(|i| i as f32 * 0.3).collect();
        let variance = vec![2.0; 13];
        let expected = scalar_batch_norm(&input, &gamma, &beta, &mean, &variance, EPS);

        let mut output = vec![0.0; 13];
        unsafe { neon_batch_norm(&input, &gamma, &beta, &mean, &variance, EPS, &mut output) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_batch_norm_zero_variance() {
        let input = vec![5.0; 8];
        let gamma = vec![1.0; 8];
        let beta = vec![0.0; 8];
        let mean = vec![5.0; 8];
        let variance = vec![0.0; 8];

        let mut output = vec![0.0; 8];
        unsafe { neon_batch_norm(&input, &gamma, &beta, &mean, &variance, EPS, &mut output) };

        // (5 - 5) / sqrt(0 + eps) = 0 → output ≈ beta = 0.
        for &v in &output {
            assert!(v.abs() < TOL, "expected ~0 with zero variance, got {v}");
        }
    }

    // ── In-place batch normalization tests ─────────────────────────

    #[test]
    fn test_batch_norm_inplace_matches() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let gamma = vec![0.5, 1.0, 1.5, 2.0, 0.1, 0.3, 0.7, 1.2, 0.9];
        let beta = vec![0.1, -0.1, 0.0, 0.5, -0.5, 0.2, -0.3, 0.0, 0.4];
        let mean = vec![1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
        let variance = vec![0.5, 1.0, 2.0, 0.1, 4.0, 0.3, 1.5, 0.8, 1.0];

        let mut out_separate = vec![0.0; 9];
        unsafe { neon_batch_norm(&input, &gamma, &beta, &mean, &variance, EPS, &mut out_separate) };

        let mut data = input.clone();
        unsafe { neon_batch_norm_inplace(&mut data, &gamma, &beta, &mean, &variance, EPS) };

        assert_approx_eq(&data, &out_separate, TOL);
    }

    // ── Instance normalization tests ──────────────────────────────

    #[test]
    fn test_instance_norm_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let channels = 2;
        let (h, w) = (2, 2);
        let gamma = vec![1.0, 1.0];
        let beta = vec![0.0, 0.0];
        let expected = scalar_instance_norm(&input, channels, h, w, &gamma, &beta, EPS);

        let mut output = vec![0.0; 8];
        unsafe { neon_instance_norm(&input, channels, h, w, &gamma, &beta, EPS, &mut output) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_instance_norm_multi_channel() {
        // 3 channels, 3×5 spatial (non-aligned: 15 elements per channel).
        let channels = 3;
        let (h, w) = (3, 5);
        let spatial = h * w;
        let input: Vec<f32> =
            (0..channels * spatial).map(|i| ((i * 7 + 3) % 100) as f32 * 0.1 - 5.0).collect();
        let gamma: Vec<f32> = (0..channels).map(|c| 0.5 + c as f32 * 0.3).collect();
        let beta: Vec<f32> = (0..channels).map(|c| -0.2 + c as f32 * 0.1).collect();
        let expected = scalar_instance_norm(&input, channels, h, w, &gamma, &beta, EPS);

        let mut output = vec![0.0; channels * spatial];
        unsafe { neon_instance_norm(&input, channels, h, w, &gamma, &beta, EPS, &mut output) };
        assert_approx_eq(&output, &expected, TOL);
    }

    // ── Parity tests ──────────────────────────────────────────────

    #[test]
    fn test_neon_vs_scalar_batch_norm_parity() {
        let n = 137;
        let input: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 100) as f32 * 0.1 - 5.0).collect();
        let gamma: Vec<f32> = (0..n).map(|i| 0.5 + (i % 5) as f32 * 0.2).collect();
        let beta: Vec<f32> = (0..n).map(|i| -0.3 + (i % 3) as f32 * 0.1).collect();
        let mean: Vec<f32> = (0..n).map(|i| (i % 10) as f32 * 0.5 - 2.5).collect();
        let variance: Vec<f32> = (0..n).map(|i| 0.1 + (i % 7) as f32 * 0.5).collect();

        let expected = scalar_batch_norm(&input, &gamma, &beta, &mean, &variance, EPS);

        let mut output = vec![0.0; n];
        unsafe { neon_batch_norm(&input, &gamma, &beta, &mean, &variance, EPS, &mut output) };
        assert_approx_eq(&output, &expected, TOL);
    }

    #[test]
    fn test_batch_norm_large_input() {
        let n = 1024;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let gamma = vec![1.0; n];
        let beta = vec![0.0; n];
        let mean = vec![0.0; n];
        let variance = vec![1.0; n];
        let expected = scalar_batch_norm(&input, &gamma, &beta, &mean, &variance, EPS);

        let mut output = vec![0.0; n];
        unsafe { neon_batch_norm(&input, &gamma, &beta, &mean, &variance, EPS, &mut output) };
        assert_approx_eq(&output, &expected, TOL);
    }
}
