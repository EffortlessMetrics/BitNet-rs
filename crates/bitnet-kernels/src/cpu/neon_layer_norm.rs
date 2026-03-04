//! NEON-optimized layer normalization kernels (v2) for Apple Silicon.
//!
//! Provides `neon_layer_norm_f32`, `neon_rms_norm_f32`, `neon_layer_norm_inplace`,
//! and a `neon_compute_mean_var` helper. All functions process 4 × f32 lanes via
//! NEON intrinsics with scalar fallback for remainder elements.

#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Public kernels ────────────────────────────────────────────────

/// Full LayerNorm with affine parameters using NEON intrinsics.
///
/// `output[i] = gamma[i] * (input[i] - mean) / sqrt(var + eps) + beta[i]`
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `gamma`, `beta`, or `output` length differs from `input`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_layer_norm_f32(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    output: &mut [f32],
) {
    let n = input.len();
    assert_eq!(gamma.len(), n, "gamma length mismatch");
    assert_eq!(beta.len(), n, "beta length mismatch");
    assert_eq!(output.len(), n, "output length mismatch");

    if n == 0 {
        return;
    }

    let (mean, var) = neon_compute_mean_var(input);
    let inv_std = 1.0 / (var + eps).sqrt();

    apply_affine_neon(input, gamma, beta, mean, inv_std, output);
}

/// RMSNorm (no mean subtraction, no beta) using NEON intrinsics.
///
/// `output[i] = gamma[i] * input[i] / sqrt(mean(input²) + eps)`
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `gamma` or `output` length differs from `input`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_rms_norm_f32(input: &[f32], gamma: &[f32], eps: f32, output: &mut [f32]) {
    let n = input.len();
    assert_eq!(gamma.len(), n, "gamma length mismatch");
    assert_eq!(output.len(), n, "output length mismatch");

    if n == 0 {
        return;
    }

    let mean_sq = neon_mean_of_squares(input);
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();

    apply_scale_neon(input, gamma, inv_rms, output);
}

/// In-place LayerNorm with affine parameters using NEON intrinsics.
///
/// Equivalent to `neon_layer_norm_f32` but writes results back into `data`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `gamma` or `beta` length differs from `data`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_layer_norm_inplace(data: &mut [f32], gamma: &[f32], beta: &[f32], eps: f32) {
    let n = data.len();
    assert_eq!(gamma.len(), n, "gamma length mismatch");
    assert_eq!(beta.len(), n, "beta length mismatch");

    if n == 0 {
        return;
    }

    let (mean, var) = neon_compute_mean_var(data);
    let inv_std = 1.0 / (var + eps).sqrt();

    apply_affine_inplace_neon(data, gamma, beta, mean, inv_std);
}

/// Compute mean and variance of `data` using NEON intrinsics.
///
/// Returns `(mean, variance)` where variance is the population variance
/// (divided by N, not N-1).
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `data` is empty.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_compute_mean_var(data: &[f32]) -> (f32, f32) {
    let n = data.len();
    assert!(n > 0, "data must not be empty");

    let chunks = n / 4;
    let remainder = n % 4;
    let ptr = data.as_ptr();

    // ── Pass 1: compute mean ──
    let mut sum_vec = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let v = vld1q_f32(ptr.add(i * 4));
        sum_vec = vaddq_f32(sum_vec, v);
    }
    let mut sum = vaddvq_f32(sum_vec);
    let tail = chunks * 4;
    for i in 0..remainder {
        sum += *ptr.add(tail + i);
    }
    let mean = sum / n as f32;

    // ── Pass 2: compute variance ──
    let mean_vec = vdupq_n_f32(mean);
    let mut var_acc = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let v = vld1q_f32(ptr.add(i * 4));
        let diff = vsubq_f32(v, mean_vec);
        var_acc = vfmaq_f32(var_acc, diff, diff);
    }
    let mut var_sum = vaddvq_f32(var_acc);
    for i in 0..remainder {
        let d = *ptr.add(tail + i) - mean;
        var_sum += d * d;
    }
    let var = var_sum / n as f32;

    (mean, var)
}

// ── Private NEON helpers ──────────────────────────────────────────

/// Compute `mean(x²)` using NEON (for RMSNorm).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_mean_of_squares(data: &[f32]) -> f32 {
    let n = data.len();
    let chunks = n / 4;
    let remainder = n % 4;
    let ptr = data.as_ptr();

    let mut acc = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let v = vld1q_f32(ptr.add(i * 4));
        acc = vfmaq_f32(acc, v, v);
    }
    let mut sq_sum = vaddvq_f32(acc);
    let tail = chunks * 4;
    for i in 0..remainder {
        let x = *ptr.add(tail + i);
        sq_sum += x * x;
    }
    sq_sum / n as f32
}

/// Apply `output = gamma * ((input - mean) * inv_std) + beta` using NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn apply_affine_neon(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    mean: f32,
    inv_std: f32,
    output: &mut [f32],
) {
    let n = input.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mean_vec = vdupq_n_f32(mean);
    let inv_std_vec = vdupq_n_f32(inv_std);
    let in_ptr = input.as_ptr();
    let g_ptr = gamma.as_ptr();
    let b_ptr = beta.as_ptr();
    let o_ptr = output.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        let v = vld1q_f32(in_ptr.add(off));
        let g = vld1q_f32(g_ptr.add(off));
        let b = vld1q_f32(b_ptr.add(off));

        let centered = vsubq_f32(v, mean_vec);
        let normed = vmulq_f32(centered, inv_std_vec);
        let scaled = vfmaq_f32(b, g, normed); // b + g * normed
        vst1q_f32(o_ptr.add(off), scaled);
    }

    let tail = chunks * 4;
    for i in 0..remainder {
        let idx = tail + i;
        let normed = (input[idx] - mean) * inv_std;
        output[idx] = gamma[idx] * normed + beta[idx];
    }
}

/// Apply `output = gamma * (input * inv_rms)` using NEON (for RMSNorm).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn apply_scale_neon(input: &[f32], gamma: &[f32], inv_rms: f32, output: &mut [f32]) {
    let n = input.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let inv_rms_vec = vdupq_n_f32(inv_rms);
    let in_ptr = input.as_ptr();
    let g_ptr = gamma.as_ptr();
    let o_ptr = output.as_mut_ptr();

    for i in 0..chunks {
        let off = i * 4;
        let v = vld1q_f32(in_ptr.add(off));
        let g = vld1q_f32(g_ptr.add(off));

        let normed = vmulq_f32(v, inv_rms_vec);
        let scaled = vmulq_f32(g, normed);
        vst1q_f32(o_ptr.add(off), scaled);
    }

    let tail = chunks * 4;
    for i in 0..remainder {
        let idx = tail + i;
        output[idx] = gamma[idx] * (input[idx] * inv_rms);
    }
}

/// In-place affine normalization using NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn apply_affine_inplace_neon(
    data: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    mean: f32,
    inv_std: f32,
) {
    let n = data.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mean_vec = vdupq_n_f32(mean);
    let inv_std_vec = vdupq_n_f32(inv_std);
    let ptr = data.as_mut_ptr();
    let g_ptr = gamma.as_ptr();
    let b_ptr = beta.as_ptr();

    for i in 0..chunks {
        let off = i * 4;
        let v = vld1q_f32(ptr.add(off));
        let g = vld1q_f32(g_ptr.add(off));
        let b = vld1q_f32(b_ptr.add(off));

        let centered = vsubq_f32(v, mean_vec);
        let normed = vmulq_f32(centered, inv_std_vec);
        let scaled = vfmaq_f32(b, g, normed);
        vst1q_f32(ptr.add(off), scaled);
    }

    let tail = chunks * 4;
    for i in 0..remainder {
        let idx = tail + i;
        let normed = (data[idx] - mean) * inv_std;
        data[idx] = gamma[idx] * normed + beta[idx];
    }
}

// ── Scalar references (test-only) ─────────────────────────────────

#[cfg(test)]
fn scalar_layer_norm(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    if n == 0 {
        return vec![];
    }
    let mean = input.iter().sum::<f32>() / n as f32;
    let var = input.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    input.iter().enumerate().map(|(i, &x)| gamma[i] * (x - mean) * inv_std + beta[i]).collect()
}

#[cfg(test)]
fn scalar_rms_norm(input: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    if n == 0 {
        return vec![];
    }
    let mean_sq = input.iter().map(|x| x * x).sum::<f32>() / n as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    input.iter().enumerate().map(|(i, &x)| gamma[i] * x * inv_rms).collect()
}

#[cfg(test)]
fn scalar_mean_var(data: &[f32]) -> (f32, f32) {
    let n = data.len() as f32;
    let mean = data.iter().sum::<f32>() / n;
    let var = data.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
    (mean, var)
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;
    const TOL: f32 = 1e-5;

    fn assert_approx(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            let scale = x.abs().max(y.abs()).max(1.0);
            assert!(
                diff <= tol * scale,
                "mismatch at [{i}]: neon={x} scalar={y} (abs_diff={diff}, rel={:.2e})",
                diff / scale
            );
        }
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "mismatch at [{i}]: {x} vs {y} (diff {})", (x - y).abs());
        }
    }

    fn ones(n: usize) -> Vec<f32> {
        vec![1.0; n]
    }
    fn zeros(n: usize) -> Vec<f32> {
        vec![0.0; n]
    }
    fn linspace(n: usize, lo: f32, hi: f32) -> Vec<f32> {
        if n <= 1 {
            return vec![lo];
        }
        (0..n).map(|i| lo + (hi - lo) * i as f32 / (n - 1) as f32).collect()
    }
    fn pseudorandom(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i * 7 + 3) % 100) as f32 * 0.1 - 5.0).collect()
    }

    // ════════════════════════════════════════════════════════════════
    // 1. LayerNorm correctness — various sizes
    // ════════════════════════════════════════════════════════════════

    macro_rules! layernorm_size_test {
        ($name:ident, $n:expr) => {
            #[test]
            fn $name() {
                let input = pseudorandom($n);
                let gamma = ones($n);
                let beta = zeros($n);
                let expected = scalar_layer_norm(&input, &gamma, &beta, EPS);
                let mut output = vec![0.0; $n];
                unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut output) };
                assert_approx(&output, &expected, TOL);
            }
        };
    }

    layernorm_size_test!(layernorm_size_1, 1);
    layernorm_size_test!(layernorm_size_4, 4);
    layernorm_size_test!(layernorm_size_7, 7);
    layernorm_size_test!(layernorm_size_8, 8);
    layernorm_size_test!(layernorm_size_15, 15);
    layernorm_size_test!(layernorm_size_16, 16);
    layernorm_size_test!(layernorm_size_31, 31);
    layernorm_size_test!(layernorm_size_32, 32);
    layernorm_size_test!(layernorm_size_63, 63);
    layernorm_size_test!(layernorm_size_64, 64);
    layernorm_size_test!(layernorm_size_128, 128);
    layernorm_size_test!(layernorm_size_256, 256);
    layernorm_size_test!(layernorm_size_512, 512);
    layernorm_size_test!(layernorm_size_1024, 1024);

    // ════════════════════════════════════════════════════════════════
    // 2. Edge cases
    // ════════════════════════════════════════════════════════════════

    #[test]
    fn layernorm_empty() {
        let mut output: Vec<f32> = vec![];
        unsafe { neon_layer_norm_f32(&[], &[], &[], EPS, &mut output) };
        assert!(output.is_empty());
    }

    #[test]
    fn layernorm_single_element() {
        let input = [42.0];
        let gamma = [2.0];
        let beta = [1.0];
        let mut output = [0.0];
        let expected = scalar_layer_norm(&input, &gamma, &beta, EPS);
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut output) };
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn layernorm_all_zeros() {
        let n = 16;
        let input = zeros(n);
        let gamma = ones(n);
        let beta = zeros(n);
        let mut output = vec![f32::NAN; n];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut output) };
        for &v in &output {
            assert!(v.abs() < TOL, "expected ~0 for zero input, got {v}");
        }
    }

    #[test]
    fn layernorm_all_ones() {
        let n = 16;
        let input = ones(n);
        let gamma = ones(n);
        let beta = zeros(n);
        let mut output = vec![0.0; n];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut output) };
        // Zero variance → all output ≈ 0.
        for &v in &output {
            assert!(v.abs() < TOL, "expected ~0 for constant input, got {v}");
        }
    }

    #[test]
    fn layernorm_large_values() {
        let n = 8;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 1e6).collect();
        let gamma = ones(n);
        let beta = zeros(n);
        let expected = scalar_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; n];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut output) };
        assert_approx(&output, &expected, 1e-4);
    }

    #[test]
    fn layernorm_very_small_values() {
        let n = 8;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 1e-6).collect();
        let gamma = ones(n);
        let beta = zeros(n);
        let expected = scalar_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; n];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut output) };
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn layernorm_negative_values() {
        let n = 8;
        let input: Vec<f32> = (0..n).map(|i| -(i as f32 + 1.0)).collect();
        let gamma = ones(n);
        let beta = zeros(n);
        let expected = scalar_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; n];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut output) };
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn layernorm_mixed_pos_neg() {
        let input = vec![-3.0, -1.5, 0.0, 1.5, 3.0];
        let gamma = ones(5);
        let beta = zeros(5);
        let expected = scalar_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; 5];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut output) };
        assert_approx(&output, &expected, TOL);
    }

    // ════════════════════════════════════════════════════════════════
    // 3. Numerical stability
    // ════════════════════════════════════════════════════════════════

    #[test]
    fn layernorm_values_near_epsilon() {
        let eps = 1e-5;
        let input = vec![eps, eps * 2.0, eps * 3.0, eps * 4.0];
        let gamma = ones(4);
        let beta = zeros(4);
        let expected = scalar_layer_norm(&input, &gamma, &beta, eps);
        let mut output = vec![0.0; 4];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, eps, &mut output) };
        assert_approx(&output, &expected, 1e-3);
    }

    #[test]
    fn layernorm_large_epsilon() {
        let eps = 1.0;
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = ones(4);
        let beta = zeros(4);
        let expected = scalar_layer_norm(&input, &gamma, &beta, eps);
        let mut output = vec![0.0; 4];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, eps, &mut output) };
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn layernorm_denormals() {
        let tiny = f32::MIN_POSITIVE / 2.0; // denormal
        let input = vec![tiny, tiny * 2.0, tiny * 3.0, tiny * 4.0];
        let gamma = ones(4);
        let beta = zeros(4);
        let expected = scalar_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; 4];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut output) };
        assert_approx(&output, &expected, 1e-3);
    }

    #[test]
    fn layernorm_wide_range() {
        let input = vec![-1e4, -1.0, 0.0, 1.0, 1e4, -1e-4, 1e-4, 0.5];
        let gamma = ones(8);
        let beta = zeros(8);
        let expected = scalar_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; 8];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut output) };
        assert_approx(&output, &expected, 1e-3);
    }

    // ════════════════════════════════════════════════════════════════
    // 4. RMSNorm correctness
    // ════════════════════════════════════════════════════════════════

    macro_rules! rmsnorm_size_test {
        ($name:ident, $n:expr) => {
            #[test]
            fn $name() {
                let input = pseudorandom($n);
                let gamma = ones($n);
                let expected = scalar_rms_norm(&input, &gamma, EPS);
                let mut output = vec![0.0; $n];
                unsafe { neon_rms_norm_f32(&input, &gamma, EPS, &mut output) };
                assert_approx(&output, &expected, TOL);
            }
        };
    }

    rmsnorm_size_test!(rmsnorm_size_1, 1);
    rmsnorm_size_test!(rmsnorm_size_4, 4);
    rmsnorm_size_test!(rmsnorm_size_7, 7);
    rmsnorm_size_test!(rmsnorm_size_8, 8);
    rmsnorm_size_test!(rmsnorm_size_15, 15);
    rmsnorm_size_test!(rmsnorm_size_16, 16);
    rmsnorm_size_test!(rmsnorm_size_31, 31);
    rmsnorm_size_test!(rmsnorm_size_32, 32);
    rmsnorm_size_test!(rmsnorm_size_64, 64);
    rmsnorm_size_test!(rmsnorm_size_128, 128);
    rmsnorm_size_test!(rmsnorm_size_256, 256);
    rmsnorm_size_test!(rmsnorm_size_512, 512);
    rmsnorm_size_test!(rmsnorm_size_1024, 1024);

    #[test]
    fn rmsnorm_empty() {
        let mut output: Vec<f32> = vec![];
        unsafe { neon_rms_norm_f32(&[], &[], EPS, &mut output) };
        assert!(output.is_empty());
    }

    #[test]
    fn rmsnorm_no_mean_subtraction() {
        // RMSNorm should NOT subtract mean — verify by checking formula directly.
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let gamma = ones(4);
        let mean_sq = input.iter().map(|x| x * x).sum::<f32>() / 4.0;
        let rms = (mean_sq + EPS).sqrt();
        let expected: Vec<f32> = input.iter().map(|&x| x / rms).collect();
        let mut output = vec![0.0; 4];
        unsafe { neon_rms_norm_f32(&input, &gamma, EPS, &mut output) };
        assert_close(&output, &expected, TOL);
    }

    #[test]
    fn rmsnorm_vs_layernorm_differ() {
        // RMSNorm and LayerNorm should produce DIFFERENT results for non-zero-mean input.
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let gamma = ones(8);
        let beta = zeros(8);
        let mut ln_out = vec![0.0; 8];
        let mut rms_out = vec![0.0; 8];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut ln_out) };
        unsafe { neon_rms_norm_f32(&input, &gamma, EPS, &mut rms_out) };
        // They should not be identical.
        let any_diff = ln_out.iter().zip(rms_out.iter()).any(|(a, b)| (a - b).abs() > 1e-3);
        assert!(any_diff, "RMSNorm and LayerNorm should differ for non-zero-mean input");
    }

    #[test]
    fn rmsnorm_with_scale() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gamma = vec![0.5, 1.0, 1.5, 2.0, 0.1];
        let expected = scalar_rms_norm(&input, &gamma, EPS);
        let mut output = vec![0.0; 5];
        unsafe { neon_rms_norm_f32(&input, &gamma, EPS, &mut output) };
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn rmsnorm_negative_input() {
        let input = vec![-1.0, -2.0, -3.0, -4.0];
        let gamma = ones(4);
        let expected = scalar_rms_norm(&input, &gamma, EPS);
        let mut output = vec![0.0; 4];
        unsafe { neon_rms_norm_f32(&input, &gamma, EPS, &mut output) };
        assert_approx(&output, &expected, TOL);
    }

    // ════════════════════════════════════════════════════════════════
    // 5. In-place LayerNorm
    // ════════════════════════════════════════════════════════════════

    macro_rules! inplace_size_test {
        ($name:ident, $n:expr) => {
            #[test]
            fn $name() {
                let input = pseudorandom($n);
                let gamma = ones($n);
                let beta = zeros($n);
                let mut out_of_place = vec![0.0; $n];
                unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut out_of_place) };
                let mut inplace = input.clone();
                unsafe { neon_layer_norm_inplace(&mut inplace, &gamma, &beta, EPS) };
                assert_close(&inplace, &out_of_place, TOL);
            }
        };
    }

    inplace_size_test!(inplace_size_1, 1);
    inplace_size_test!(inplace_size_4, 4);
    inplace_size_test!(inplace_size_7, 7);
    inplace_size_test!(inplace_size_16, 16);
    inplace_size_test!(inplace_size_31, 31);
    inplace_size_test!(inplace_size_64, 64);
    inplace_size_test!(inplace_size_128, 128);
    inplace_size_test!(inplace_size_1024, 1024);

    #[test]
    fn inplace_empty() {
        let mut data: Vec<f32> = vec![];
        unsafe { neon_layer_norm_inplace(&mut data, &[], &[], EPS) };
        assert!(data.is_empty());
    }

    #[test]
    fn inplace_with_affine() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gamma = vec![0.5, 1.0, 1.5, 2.0, 0.1];
        let beta = vec![0.1, -0.1, 0.0, 0.5, -0.5];
        let mut out_of_place = vec![0.0; 5];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut out_of_place) };
        let mut inplace = input.clone();
        unsafe { neon_layer_norm_inplace(&mut inplace, &gamma, &beta, EPS) };
        assert_close(&inplace, &out_of_place, TOL);
    }

    // ════════════════════════════════════════════════════════════════
    // 6. Mean/variance helper
    // ════════════════════════════════════════════════════════════════

    #[test]
    fn mean_var_single() {
        let data = [42.0f32];
        let (mean, var) = unsafe { neon_compute_mean_var(&data) };
        assert!((mean - 42.0).abs() < TOL);
        assert!(var.abs() < TOL);
    }

    #[test]
    fn mean_var_uniform() {
        let data = vec![5.0; 16];
        let (mean, var) = unsafe { neon_compute_mean_var(&data) };
        assert!((mean - 5.0).abs() < TOL);
        assert!(var.abs() < TOL);
    }

    #[test]
    fn mean_var_known_distribution() {
        // [1, 2, 3, 4]: mean=2.5, var=1.25
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let (mean, var) = unsafe { neon_compute_mean_var(&data) };
        assert!((mean - 2.5).abs() < TOL);
        assert!((var - 1.25).abs() < TOL);
    }

    #[test]
    fn mean_var_negative() {
        let data = vec![-4.0, -2.0, 0.0, 2.0, 4.0];
        let (mean, var) = unsafe { neon_compute_mean_var(&data) };
        let (exp_mean, exp_var) = scalar_mean_var(&data);
        assert!((mean - exp_mean).abs() < TOL);
        assert!((var - exp_var).abs() < TOL);
    }

    #[test]
    fn mean_var_large_n() {
        let data = linspace(1024, -10.0, 10.0);
        let (mean, var) = unsafe { neon_compute_mean_var(&data) };
        let (exp_mean, exp_var) = scalar_mean_var(&data);
        assert!((mean - exp_mean).abs() < 1e-3);
        assert!((var - exp_var).abs() < 1e-2);
    }

    #[test]
    fn mean_var_non_aligned() {
        let data: Vec<f32> = (0..13).map(|i| i as f32).collect();
        let (mean, var) = unsafe { neon_compute_mean_var(&data) };
        let (exp_mean, exp_var) = scalar_mean_var(&data);
        assert!((mean - exp_mean).abs() < TOL);
        assert!((var - exp_var).abs() < TOL);
    }

    #[test]
    fn mean_var_symmetric() {
        // Symmetric around 0 → mean ≈ 0.
        let data = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let (mean, var) = unsafe { neon_compute_mean_var(&data) };
        assert!(mean.abs() < TOL, "mean should be ~0, got {mean}");
        let (_, exp_var) = scalar_mean_var(&data);
        assert!((var - exp_var).abs() < TOL);
    }

    // ════════════════════════════════════════════════════════════════
    // 7. Non-trivial gamma/beta
    // ════════════════════════════════════════════════════════════════

    #[test]
    fn layernorm_varying_gamma_beta() {
        let n = 16;
        let input = linspace(n, -5.0, 5.0);
        let gamma: Vec<f32> = (0..n).map(|i| 0.1 + (i as f32) * 0.2).collect();
        let beta: Vec<f32> = (0..n).map(|i| -1.0 + (i as f32) * 0.15).collect();
        let expected = scalar_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; n];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut output) };
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn layernorm_negative_gamma() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![-1.0; 4];
        let beta = zeros(4);
        let expected = scalar_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; 4];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut output) };
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn layernorm_large_beta() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = ones(4);
        let beta = vec![100.0; 4];
        let expected = scalar_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; 4];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut output) };
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn layernorm_zero_gamma() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = zeros(4);
        let beta = vec![5.0; 4];
        let mut output = vec![0.0; 4];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut output) };
        // gamma=0 → output = beta.
        assert_close(&output, &beta, TOL);
    }

    #[test]
    fn rmsnorm_varying_gamma() {
        let n = 16;
        let input = linspace(n, 1.0, 10.0);
        let gamma: Vec<f32> = (0..n).map(|i| 0.1 + (i as f32) * 0.2).collect();
        let expected = scalar_rms_norm(&input, &gamma, EPS);
        let mut output = vec![0.0; n];
        unsafe { neon_rms_norm_f32(&input, &gamma, EPS, &mut output) };
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn rmsnorm_zero_gamma() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = zeros(4);
        let mut output = vec![0.0; 4];
        unsafe { neon_rms_norm_f32(&input, &gamma, EPS, &mut output) };
        for &v in &output {
            assert!(v.abs() < TOL, "gamma=0 should yield 0, got {v}");
        }
    }

    // ════════════════════════════════════════════════════════════════
    // 8. Property tests
    // ════════════════════════════════════════════════════════════════

    #[test]
    fn property_layernorm_output_mean_approx_zero() {
        // With gamma=1, beta=0 the output mean should be ≈ 0.
        for &n in &[8, 16, 33, 64, 128, 256] {
            let input = pseudorandom(n);
            let gamma = ones(n);
            let beta = zeros(n);
            let mut output = vec![0.0; n];
            unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut output) };
            let mean = output.iter().sum::<f32>() / n as f32;
            assert!(mean.abs() < 1e-4, "n={n}: output mean should be ~0, got {mean}");
        }
    }

    #[test]
    fn property_layernorm_output_unit_variance() {
        // With gamma=1, beta=0 the output variance should be ≈ 1.
        for &n in &[16, 32, 64, 128, 256, 512] {
            let input = linspace(n, -10.0, 10.0);
            let gamma = ones(n);
            let beta = zeros(n);
            let mut output = vec![0.0; n];
            unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut output) };
            let mean = output.iter().sum::<f32>() / n as f32;
            let var = output.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
            assert!((var - 1.0).abs() < 1e-3, "n={n}: output variance should be ~1, got {var}");
        }
    }

    #[test]
    fn property_rmsnorm_preserves_sign() {
        // With gamma=1 the sign of output should match sign of input.
        let input = vec![-5.0, -1.0, 0.0, 1.0, 5.0, -3.0, 3.0, 0.0];
        let gamma = ones(8);
        let mut output = vec![0.0; 8];
        unsafe { neon_rms_norm_f32(&input, &gamma, EPS, &mut output) };
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            if inp.abs() > TOL {
                assert_eq!(
                    inp.is_sign_positive(),
                    out.is_sign_positive(),
                    "sign mismatch at [{i}]: inp={inp}, out={out}"
                );
            }
        }
    }

    #[test]
    fn property_layernorm_idempotent_on_normalized() {
        // Applying LayerNorm to already-normalized data (gamma=1, beta=0) is ~idempotent.
        let n = 64;
        let input = pseudorandom(n);
        let gamma = ones(n);
        let beta = zeros(n);
        let mut first = vec![0.0; n];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut first) };
        let mut second = vec![0.0; n];
        unsafe { neon_layer_norm_f32(&first, &gamma, &beta, EPS, &mut second) };
        assert_approx(&first, &second, 1e-4);
    }

    #[test]
    fn property_layernorm_shift_invariant() {
        // LayerNorm(x + c) == LayerNorm(x) when gamma=1, beta=0.
        let n = 32;
        let input = pseudorandom(n);
        let shifted: Vec<f32> = input.iter().map(|x| x + 1000.0).collect();
        let gamma = ones(n);
        let beta = zeros(n);
        let mut out_orig = vec![0.0; n];
        let mut out_shifted = vec![0.0; n];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut out_orig) };
        unsafe { neon_layer_norm_f32(&shifted, &gamma, &beta, EPS, &mut out_shifted) };
        assert_approx(&out_orig, &out_shifted, 1e-3);
    }

    #[test]
    fn property_layernorm_scale_invariant() {
        // LayerNorm(c * x) == LayerNorm(x) when gamma=1, beta=0, c > 0.
        let n = 32;
        let input = linspace(n, 1.0, 10.0);
        let scaled: Vec<f32> = input.iter().map(|x| x * 100.0).collect();
        let gamma = ones(n);
        let beta = zeros(n);
        let mut out_orig = vec![0.0; n];
        let mut out_scaled = vec![0.0; n];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut out_orig) };
        unsafe { neon_layer_norm_f32(&scaled, &gamma, &beta, EPS, &mut out_scaled) };
        assert_approx(&out_orig, &out_scaled, 1e-3);
    }

    // ════════════════════════════════════════════════════════════════
    // 9. Transformer hidden dims
    // ════════════════════════════════════════════════════════════════

    macro_rules! transformer_dim_test {
        ($ln_name:ident, $rms_name:ident, $n:expr) => {
            #[test]
            fn $ln_name() {
                let input = pseudorandom($n);
                let gamma = ones($n);
                let beta = zeros($n);
                let expected = scalar_layer_norm(&input, &gamma, &beta, EPS);
                let mut output = vec![0.0; $n];
                unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut output) };
                assert_approx(&output, &expected, 1e-4);
            }
            #[test]
            fn $rms_name() {
                let input = pseudorandom($n);
                let gamma = ones($n);
                let expected = scalar_rms_norm(&input, &gamma, EPS);
                let mut output = vec![0.0; $n];
                unsafe { neon_rms_norm_f32(&input, &gamma, EPS, &mut output) };
                assert_approx(&output, &expected, 1e-4);
            }
        };
    }

    transformer_dim_test!(layernorm_dim_768, rmsnorm_dim_768, 768);
    transformer_dim_test!(layernorm_dim_1024, rmsnorm_dim_1024, 1024);
    transformer_dim_test!(layernorm_dim_2048, rmsnorm_dim_2048, 2048);
    transformer_dim_test!(layernorm_dim_4096, rmsnorm_dim_4096, 4096);

    // ════════════════════════════════════════════════════════════════
    // 10. Tolerance (relative) tests
    // ════════════════════════════════════════════════════════════════

    #[test]
    fn tolerance_layernorm_rel_1e5() {
        let n = 137;
        let input = pseudorandom(n);
        let gamma: Vec<f32> = (0..n).map(|i| 0.5 + (i % 5) as f32 * 0.2).collect();
        let beta: Vec<f32> = (0..n).map(|i| -0.3 + (i % 3) as f32 * 0.1).collect();
        let expected = scalar_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; n];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut output) };
        // Relative tolerance: |neon - scalar| / max(|scalar|, 1) ≤ 1e-5
        assert_approx(&output, &expected, 1e-5);
    }

    #[test]
    fn tolerance_rmsnorm_rel_1e5() {
        let n = 137;
        let input = pseudorandom(n);
        let gamma: Vec<f32> = (0..n).map(|i| 0.5 + (i % 5) as f32 * 0.2).collect();
        let expected = scalar_rms_norm(&input, &gamma, EPS);
        let mut output = vec![0.0; n];
        unsafe { neon_rms_norm_f32(&input, &gamma, EPS, &mut output) };
        assert_approx(&output, &expected, 1e-5);
    }

    #[test]
    fn tolerance_layernorm_sweep_sizes() {
        for n in [3, 5, 9, 17, 33, 65, 129, 257, 513] {
            let input = pseudorandom(n);
            let gamma = ones(n);
            let beta = zeros(n);
            let expected = scalar_layer_norm(&input, &gamma, &beta, EPS);
            let mut output = vec![0.0; n];
            unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut output) };
            assert_approx(&output, &expected, 1e-4);
        }
    }

    #[test]
    fn tolerance_rmsnorm_sweep_sizes() {
        for n in [3, 5, 9, 17, 33, 65, 129, 257, 513] {
            let input = pseudorandom(n);
            let gamma = ones(n);
            let expected = scalar_rms_norm(&input, &gamma, EPS);
            let mut output = vec![0.0; n];
            unsafe { neon_rms_norm_f32(&input, &gamma, EPS, &mut output) };
            assert_approx(&output, &expected, 1e-4);
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Additional edge-case tests
    // ════════════════════════════════════════════════════════════════

    #[test]
    fn layernorm_alternating_sign() {
        let input: Vec<f32> = (0..16).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let gamma = ones(16);
        let beta = zeros(16);
        let expected = scalar_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; 16];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut output) };
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn rmsnorm_all_ones() {
        let n = 8;
        let input = ones(n);
        let gamma = ones(n);
        let expected = scalar_rms_norm(&input, &gamma, EPS);
        let mut output = vec![0.0; n];
        unsafe { neon_rms_norm_f32(&input, &gamma, EPS, &mut output) };
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn rmsnorm_all_zeros() {
        let n = 8;
        let input = zeros(n);
        let gamma = ones(n);
        let mut output = vec![f32::NAN; n];
        unsafe { neon_rms_norm_f32(&input, &gamma, EPS, &mut output) };
        // 0 / sqrt(eps) ≈ 0.
        for &v in &output {
            assert!(v.abs() < TOL, "expected ~0 for zero input, got {v}");
        }
    }

    #[test]
    fn inplace_with_varying_gamma_beta() {
        let n = 33;
        let input = pseudorandom(n);
        let gamma: Vec<f32> = (0..n).map(|i| 0.1 * (i as f32 + 1.0)).collect();
        let beta: Vec<f32> = (0..n).map(|i| -0.5 + 0.05 * i as f32).collect();
        let mut out_of_place = vec![0.0; n];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut out_of_place) };
        let mut inplace = input.clone();
        unsafe { neon_layer_norm_inplace(&mut inplace, &gamma, &beta, EPS) };
        assert_close(&inplace, &out_of_place, TOL);
    }

    #[test]
    fn mean_var_two_elements() {
        let data = [1.0f32, 3.0];
        let (mean, var) = unsafe { neon_compute_mean_var(&data) };
        assert!((mean - 2.0).abs() < TOL);
        assert!((var - 1.0).abs() < TOL);
    }

    #[test]
    fn mean_var_three_elements() {
        let data = [2.0f32, 4.0, 6.0];
        let (mean, var) = unsafe { neon_compute_mean_var(&data) };
        let (exp_m, exp_v) = scalar_mean_var(&data);
        assert!((mean - exp_m).abs() < TOL);
        assert!((var - exp_v).abs() < TOL);
    }

    #[test]
    fn layernorm_two_elements() {
        let input = vec![0.0, 1.0];
        let gamma = vec![1.0, 1.0];
        let beta = vec![0.0, 0.0];
        let expected = scalar_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; 2];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut output) };
        assert_approx(&output, &expected, TOL);
    }

    #[test]
    fn layernorm_three_elements() {
        let input = vec![-1.0, 0.0, 1.0];
        let gamma = vec![2.0, 2.0, 2.0];
        let beta = vec![0.5, 0.5, 0.5];
        let expected = scalar_layer_norm(&input, &gamma, &beta, EPS);
        let mut output = vec![0.0; 3];
        unsafe { neon_layer_norm_f32(&input, &gamma, &beta, EPS, &mut output) };
        assert_approx(&output, &expected, TOL);
    }
}
