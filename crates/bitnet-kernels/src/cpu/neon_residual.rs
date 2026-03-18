//! ARM NEON-optimized residual connection and element-wise operations for Apple Silicon.
//!
//! Provides vectorized residual add, scaled residual add, fused residual + layer norm,
//! Hadamard product, scalar multiplication, and bias addition using NEON intrinsics,
//! with scalar fallback for remainder elements.

#![allow(unsafe_op_in_unsafe_fn)]

use std::arch::aarch64::*;

// ── helpers ──────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
const LANES: usize = 4; // float32x4_t

// ── 1. residual_add ─────────────────────────────────────────────────

/// Element-wise residual addition: `output[i] = input[i] + residual[i]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
///
/// Panics if slice lengths differ.
#[target_feature(enable = "neon")]
pub unsafe fn neon_residual_add_f32(input: &[f32], residual: &[f32], output: &mut [f32]) {
    let n = input.len();
    assert_eq!(residual.len(), n, "residual length mismatch");
    assert_eq!(output.len(), n, "output length mismatch");

    let chunks = n / LANES;
    let inp = input.as_ptr();
    let res = residual.as_ptr();
    let out = output.as_mut_ptr();

    for i in 0..chunks {
        let off = i * LANES;
        let va = vld1q_f32(inp.add(off));
        let vb = vld1q_f32(res.add(off));
        vst1q_f32(out.add(off), vaddq_f32(va, vb));
    }
    for i in (chunks * LANES)..n {
        *out.add(i) = *inp.add(i) + *res.add(i);
    }
}

// ── 2. residual_add_inplace ─────────────────────────────────────────

/// In-place residual addition: `data[i] += residual[i]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
///
/// Panics if `residual` length differs from `data`.
#[target_feature(enable = "neon")]
pub unsafe fn neon_residual_add_inplace(data: &mut [f32], residual: &[f32]) {
    let n = data.len();
    assert_eq!(residual.len(), n, "residual length mismatch");

    let chunks = n / LANES;
    let d = data.as_mut_ptr();
    let r = residual.as_ptr();

    for i in 0..chunks {
        let off = i * LANES;
        let va = vld1q_f32(d.add(off));
        let vb = vld1q_f32(r.add(off));
        vst1q_f32(d.add(off), vaddq_f32(va, vb));
    }
    for i in (chunks * LANES)..n {
        *d.add(i) += *r.add(i);
    }
}

// ── 3. residual_add_scale ───────────────────────────────────────────

/// Scaled residual addition: `output[i] = input[i] + alpha * residual[i]`.
///
/// Uses `vfmaq_f32` (fused multiply-add) for better precision and throughput.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
///
/// Panics if slice lengths differ.
#[target_feature(enable = "neon")]
pub unsafe fn neon_residual_add_scale_f32(
    input: &[f32],
    residual: &[f32],
    alpha: f32,
    output: &mut [f32],
) {
    let n = input.len();
    assert_eq!(residual.len(), n, "residual length mismatch");
    assert_eq!(output.len(), n, "output length mismatch");

    let chunks = n / LANES;
    let inp = input.as_ptr();
    let res = residual.as_ptr();
    let out = output.as_mut_ptr();
    let valpha = vdupq_n_f32(alpha);

    for i in 0..chunks {
        let off = i * LANES;
        let va = vld1q_f32(inp.add(off));
        let vb = vld1q_f32(res.add(off));
        // fma: va + valpha * vb
        vst1q_f32(out.add(off), vfmaq_f32(va, vb, valpha));
    }
    for i in (chunks * LANES)..n {
        *out.add(i) = *inp.add(i) + alpha * *res.add(i);
    }
}

// ── 4. fused residual + layer norm ──────────────────────────────────

/// Fused residual addition and layer normalization in a single pass.
///
/// Computes `tmp = input + residual`, then layer-normalises `tmp` with
/// affine parameters `gamma` and `beta`:
///
/// ```text
/// output[i] = gamma[i] * (tmp[i] - mean) / sqrt(var + eps) + beta[i]
/// ```
///
/// Two-pass algorithm: first pass computes mean and variance of `tmp`,
/// second pass normalises with the affine transform.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
///
/// Panics if any slice length differs from `input`.
#[target_feature(enable = "neon")]
pub unsafe fn neon_fused_residual_layernorm_f32(
    input: &[f32],
    residual: &[f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    output: &mut [f32],
) {
    let n = input.len();
    assert_eq!(residual.len(), n, "residual length mismatch");
    assert_eq!(gamma.len(), n, "gamma length mismatch");
    assert_eq!(beta.len(), n, "beta length mismatch");
    assert_eq!(output.len(), n, "output length mismatch");

    if n == 0 {
        return;
    }

    let chunks = n / LANES;
    let inp = input.as_ptr();
    let res = residual.as_ptr();

    // ── pass 1: sum & sum-of-squares of (input + residual) ──────
    let mut vsum = vdupq_n_f32(0.0);
    let mut vsum_sq = vdupq_n_f32(0.0);
    let mut scalar_sum: f32 = 0.0;
    let mut scalar_sum_sq: f32 = 0.0;

    for i in 0..chunks {
        let off = i * LANES;
        let va = vld1q_f32(inp.add(off));
        let vb = vld1q_f32(res.add(off));
        let vt = vaddq_f32(va, vb);
        vsum = vaddq_f32(vsum, vt);
        vsum_sq = vfmaq_f32(vsum_sq, vt, vt);
    }
    for i in (chunks * LANES)..n {
        let t = *inp.add(i) + *res.add(i);
        scalar_sum += t;
        scalar_sum_sq += t * t;
    }

    // horizontal reduction
    let sum = vaddvq_f32(vsum) + scalar_sum;
    let sum_sq = vaddvq_f32(vsum_sq) + scalar_sum_sq;

    let inv_n = 1.0 / n as f32;
    let mean = sum * inv_n;
    let variance = sum_sq * inv_n - mean * mean;
    let inv_std = 1.0 / (variance + eps).sqrt();

    // ── pass 2: normalise with affine ───────────────────────────
    let vmean = vdupq_n_f32(mean);
    let vinv = vdupq_n_f32(inv_std);
    let gp = gamma.as_ptr();
    let bp = beta.as_ptr();
    let out = output.as_mut_ptr();

    for i in 0..chunks {
        let off = i * LANES;
        let va = vld1q_f32(inp.add(off));
        let vb = vld1q_f32(res.add(off));
        let vt = vaddq_f32(va, vb);
        let vn = vmulq_f32(vsubq_f32(vt, vmean), vinv);
        let vg = vld1q_f32(gp.add(off));
        let vbt = vld1q_f32(bp.add(off));
        // gamma * norm + beta
        vst1q_f32(out.add(off), vfmaq_f32(vbt, vn, vg));
    }
    for i in (chunks * LANES)..n {
        let t = *inp.add(i) + *res.add(i);
        let norm = (t - mean) * inv_std;
        *out.add(i) = gamma[i] * norm + beta[i];
    }
}

// ── 5. elementwise mul (Hadamard) ───────────────────────────────────

/// Hadamard (element-wise) product: `output[i] = a[i] * b[i]`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
///
/// Panics if slice lengths differ.
#[target_feature(enable = "neon")]
pub unsafe fn neon_elementwise_mul_f32(a: &[f32], b: &[f32], output: &mut [f32]) {
    let n = a.len();
    assert_eq!(b.len(), n, "b length mismatch");
    assert_eq!(output.len(), n, "output length mismatch");

    let chunks = n / LANES;
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let out = output.as_mut_ptr();

    for i in 0..chunks {
        let off = i * LANES;
        let va = vld1q_f32(ap.add(off));
        let vb = vld1q_f32(bp.add(off));
        vst1q_f32(out.add(off), vmulq_f32(va, vb));
    }
    for i in (chunks * LANES)..n {
        *out.add(i) = *ap.add(i) * *bp.add(i);
    }
}

// ── 6. scale ────────────────────────────────────────────────────────

/// Scalar multiplication: `output[i] = data[i] * scale`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
///
/// Panics if `output` length differs from `data`.
#[target_feature(enable = "neon")]
pub unsafe fn neon_scale_f32(data: &[f32], scale: f32, output: &mut [f32]) {
    let n = data.len();
    assert_eq!(output.len(), n, "output length mismatch");

    let chunks = n / LANES;
    let dp = data.as_ptr();
    let out = output.as_mut_ptr();
    let vs = vdupq_n_f32(scale);

    for i in 0..chunks {
        let off = i * LANES;
        let va = vld1q_f32(dp.add(off));
        vst1q_f32(out.add(off), vmulq_f32(va, vs));
    }
    for i in (chunks * LANES)..n {
        *out.add(i) = *dp.add(i) * scale;
    }
}

// ── 7. add bias ─────────────────────────────────────────────────────

/// Broadcast bias addition: `output[i] = data[i] + bias[i % bias.len()]`.
///
/// `data` length must be a multiple of `bias` length (broadcasts bias
/// across sequence/batch positions).
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
///
/// Panics if `data.len()` is not a multiple of `bias.len()`, or `output`
/// length differs from `data`.
#[target_feature(enable = "neon")]
pub unsafe fn neon_add_bias_f32(data: &[f32], bias: &[f32], output: &mut [f32]) {
    let n = data.len();
    let b = bias.len();
    assert!(b > 0, "bias must not be empty");
    assert_eq!(n % b, 0, "data length must be a multiple of bias length");
    assert_eq!(output.len(), n, "output length mismatch");

    let dp = data.as_ptr();
    let bp = bias.as_ptr();
    let out = output.as_mut_ptr();
    let reps = n / b;

    for r in 0..reps {
        let base = r * b;
        let chunks = b / LANES;

        for i in 0..chunks {
            let off = i * LANES;
            let vd = vld1q_f32(dp.add(base + off));
            let vb = vld1q_f32(bp.add(off));
            vst1q_f32(out.add(base + off), vaddq_f32(vd, vb));
        }
        for i in (chunks * LANES)..b {
            *out.add(base + i) = *dp.add(base + i) + *bp.add(i);
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    // ── scalar reference implementations ────────────────────────

    fn ref_residual_add(input: &[f32], residual: &[f32]) -> Vec<f32> {
        input.iter().zip(residual).map(|(a, b)| a + b).collect()
    }

    fn ref_residual_add_scale(input: &[f32], residual: &[f32], alpha: f32) -> Vec<f32> {
        input.iter().zip(residual).map(|(a, b)| a + alpha * b).collect()
    }

    fn ref_fused_residual_layernorm(
        input: &[f32],
        residual: &[f32],
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Vec<f32> {
        let tmp: Vec<f32> = input.iter().zip(residual).map(|(a, b)| a + b).collect();
        let n = tmp.len() as f32;
        let mean: f32 = tmp.iter().sum::<f32>() / n;
        let var: f32 = tmp.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
        let inv_std = 1.0 / (var + eps).sqrt();
        tmp.iter()
            .zip(gamma.iter().zip(beta))
            .map(|(t, (g, b))| g * (t - mean) * inv_std + b)
            .collect()
    }

    fn ref_elementwise_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b).map(|(x, y)| x * y).collect()
    }

    fn ref_scale(data: &[f32], scale: f32) -> Vec<f32> {
        data.iter().map(|x| x * scale).collect()
    }

    fn ref_add_bias(data: &[f32], bias: &[f32]) -> Vec<f32> {
        data.iter().enumerate().map(|(i, x)| x + bias[i % bias.len()]).collect()
    }

    // ── helpers ─────────────────────────────────────────────────

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            let diff = (x - y).abs();
            assert!(diff <= tol, "mismatch at index {i}: {x} vs {y} (diff {diff})");
        }
    }

    #[cfg(target_arch = "aarch64")]
    const SIZES: &[usize] = &[1, 3, 4, 7, 8, 15, 16, 31, 32, 64, 128, 256, 512, 1024];

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32 * 0.1).collect()
    }

    fn constant(n: usize, v: f32) -> Vec<f32> {
        vec![v; n]
    }

    fn alternating(n: usize) -> Vec<f32> {
        (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect()
    }

    // ==============================================================
    // 1. neon_residual_add_f32
    // ==============================================================

    #[test]
    fn residual_add_sizes() {
        for &n in SIZES {
            let a = ramp(n);
            let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.1).collect();
            let mut out = vec![0.0f32; n];
            unsafe { neon_residual_add_f32(&a, &b, &mut out) };
            assert_close(&out, &ref_residual_add(&a, &b), 1e-6);
        }
    }

    #[test]
    fn residual_add_zeros() {
        for &n in SIZES {
            let a = ramp(n);
            let z = constant(n, 0.0);
            let mut out = vec![0.0f32; n];
            unsafe { neon_residual_add_f32(&a, &z, &mut out) };
            assert_close(&out, &a, 1e-7);
        }
    }

    #[test]
    fn residual_add_ones() {
        for &n in SIZES {
            let a = constant(n, 1.0);
            let b = constant(n, 1.0);
            let mut out = vec![0.0f32; n];
            unsafe { neon_residual_add_f32(&a, &b, &mut out) };
            assert_close(&out, &constant(n, 2.0), 1e-7);
        }
    }

    #[test]
    fn residual_add_negative() {
        for &n in SIZES {
            let a: Vec<f32> = (0..n).map(|i| -(i as f32)).collect();
            let b: Vec<f32> = (0..n).map(|i| -(i as f32) * 0.5).collect();
            let mut out = vec![0.0f32; n];
            unsafe { neon_residual_add_f32(&a, &b, &mut out) };
            assert_close(&out, &ref_residual_add(&a, &b), 1e-6);
        }
    }

    #[test]
    fn residual_add_mixed_sign() {
        for &n in SIZES {
            let a = alternating(n);
            let b = ramp(n);
            let mut out = vec![0.0f32; n];
            unsafe { neon_residual_add_f32(&a, &b, &mut out) };
            assert_close(&out, &ref_residual_add(&a, &b), 1e-6);
        }
    }

    #[test]
    fn residual_add_commutative() {
        for &n in SIZES {
            let a = ramp(n);
            let b = alternating(n);
            let mut out1 = vec![0.0f32; n];
            let mut out2 = vec![0.0f32; n];
            unsafe {
                neon_residual_add_f32(&a, &b, &mut out1);
                neon_residual_add_f32(&b, &a, &mut out2);
            }
            assert_close(&out1, &out2, 1e-7);
        }
    }

    #[test]
    fn residual_add_large_values() {
        let n = 16;
        let a = constant(n, 1e30);
        let b = constant(n, 1e30);
        let mut out = vec![0.0f32; n];
        unsafe { neon_residual_add_f32(&a, &b, &mut out) };
        assert_close(&out, &constant(n, 2e30), 1e24);
    }

    // ==============================================================
    // 2. neon_residual_add_inplace
    // ==============================================================

    #[test]
    fn inplace_add_sizes() {
        for &n in SIZES {
            let orig = ramp(n);
            let r: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.1).collect();
            let mut data = orig.clone();
            unsafe { neon_residual_add_inplace(&mut data, &r) };
            assert_close(&data, &ref_residual_add(&orig, &r), 1e-6);
        }
    }

    #[test]
    fn inplace_matches_outofplace() {
        for &n in SIZES {
            let a = ramp(n);
            let r = alternating(n);
            let mut data = a.clone();
            let mut out = vec![0.0f32; n];
            unsafe {
                neon_residual_add_inplace(&mut data, &r);
                neon_residual_add_f32(&a, &r, &mut out);
            }
            assert_close(&data, &out, 1e-7);
        }
    }

    #[test]
    fn inplace_add_zeros() {
        for &n in SIZES {
            let orig = ramp(n);
            let mut data = orig.clone();
            let z = constant(n, 0.0);
            unsafe { neon_residual_add_inplace(&mut data, &z) };
            assert_close(&data, &orig, 1e-7);
        }
    }

    #[test]
    fn inplace_add_negative() {
        for &n in SIZES {
            let a = constant(n, 5.0);
            let b = constant(n, -5.0);
            let mut data = a;
            unsafe { neon_residual_add_inplace(&mut data, &b) };
            assert_close(&data, &constant(n, 0.0), 1e-7);
        }
    }

    // ==============================================================
    // 3. neon_residual_add_scale_f32
    // ==============================================================

    #[test]
    fn scaled_add_sizes() {
        for &n in SIZES {
            let a = ramp(n);
            let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.1).collect();
            let alpha = 0.5;
            let mut out = vec![0.0f32; n];
            unsafe { neon_residual_add_scale_f32(&a, &b, alpha, &mut out) };
            assert_close(&out, &ref_residual_add_scale(&a, &b, alpha), 1e-5);
        }
    }

    #[test]
    fn scaled_add_alpha_zero() {
        for &n in SIZES {
            let a = ramp(n);
            let b = constant(n, 999.0);
            let mut out = vec![0.0f32; n];
            unsafe { neon_residual_add_scale_f32(&a, &b, 0.0, &mut out) };
            assert_close(&out, &a, 1e-7);
        }
    }

    #[test]
    fn scaled_add_alpha_one() {
        for &n in SIZES {
            let a = ramp(n);
            let b = alternating(n);
            let mut out_scaled = vec![0.0f32; n];
            let mut out_plain = vec![0.0f32; n];
            unsafe {
                neon_residual_add_scale_f32(&a, &b, 1.0, &mut out_scaled);
                neon_residual_add_f32(&a, &b, &mut out_plain);
            }
            assert_close(&out_scaled, &out_plain, 1e-7);
        }
    }

    #[test]
    fn scaled_add_alpha_neg_one() {
        for &n in SIZES {
            let a = ramp(n);
            let b = ramp(n);
            let mut out = vec![0.0f32; n];
            unsafe { neon_residual_add_scale_f32(&a, &b, -1.0, &mut out) };
            assert_close(&out, &constant(n, 0.0), 1e-6);
        }
    }

    #[test]
    fn scaled_add_negative_alpha() {
        for &n in SIZES {
            let a = constant(n, 10.0);
            let b = constant(n, 2.0);
            let mut out = vec![0.0f32; n];
            unsafe { neon_residual_add_scale_f32(&a, &b, -3.0, &mut out) };
            assert_close(&out, &constant(n, 4.0), 1e-6);
        }
    }

    #[test]
    fn scaled_add_mixed_sign() {
        for &n in SIZES {
            let a = alternating(n);
            let b = ramp(n);
            let alpha = 0.25;
            let mut out = vec![0.0f32; n];
            unsafe { neon_residual_add_scale_f32(&a, &b, alpha, &mut out) };
            assert_close(&out, &ref_residual_add_scale(&a, &b, alpha), 1e-5);
        }
    }

    #[test]
    fn scaled_add_large_alpha() {
        let n = 32;
        let a = constant(n, 1.0);
        let b = constant(n, 1.0);
        let alpha = 1e10;
        let mut out = vec![0.0f32; n];
        unsafe { neon_residual_add_scale_f32(&a, &b, alpha, &mut out) };
        let expected = constant(n, 1.0 + alpha);
        assert_close(&out, &expected, 1e4);
    }

    // ==============================================================
    // 4. neon_fused_residual_layernorm_f32
    // ==============================================================

    #[test]
    fn fused_ln_sizes() {
        for &n in SIZES {
            let inp = ramp(n);
            let res: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.05).collect();
            let gamma = constant(n, 1.0);
            let beta = constant(n, 0.0);
            let eps = 1e-5;
            let mut out = vec![0.0f32; n];
            unsafe {
                neon_fused_residual_layernorm_f32(&inp, &res, &gamma, &beta, eps, &mut out);
            }
            let expected = ref_fused_residual_layernorm(&inp, &res, &gamma, &beta, eps);
            assert_close(&out, &expected, 1e-4);
        }
    }

    #[test]
    fn fused_ln_matches_sequential() {
        for &n in &[8, 16, 64, 256] {
            let inp = ramp(n);
            let res = alternating(n);
            let gamma: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.01).collect();
            let beta: Vec<f32> = (0..n).map(|i| -(i as f32) * 0.005).collect();
            let eps = 1e-5;

            let mut out = vec![0.0f32; n];
            unsafe {
                neon_fused_residual_layernorm_f32(&inp, &res, &gamma, &beta, eps, &mut out);
            }
            let expected = ref_fused_residual_layernorm(&inp, &res, &gamma, &beta, eps);
            assert_close(&out, &expected, 1e-4);
        }
    }

    #[test]
    fn fused_ln_identity_gamma_beta() {
        let n = 32;
        let inp = ramp(n);
        let res = constant(n, 0.0);
        let gamma = constant(n, 1.0);
        let beta = constant(n, 0.0);
        let eps = 1e-5;
        let mut out = vec![0.0f32; n];
        unsafe {
            neon_fused_residual_layernorm_f32(&inp, &res, &gamma, &beta, eps, &mut out);
        }
        // Result should be mean-0, unit-var normalised ramp
        let mean: f32 = out.iter().sum::<f32>() / n as f32;
        assert!(mean.abs() < 1e-4, "mean should be ~0, got {mean}");
    }

    #[test]
    fn fused_ln_constant_input() {
        // Constant input → all zeros after norm (with gamma=1, beta=0)
        let n = 64;
        let inp = constant(n, 3.0);
        let res = constant(n, 2.0);
        let gamma = constant(n, 1.0);
        let beta = constant(n, 0.0);
        let eps = 1e-5;
        let mut out = vec![0.0f32; n];
        unsafe {
            neon_fused_residual_layernorm_f32(&inp, &res, &gamma, &beta, eps, &mut out);
        }
        // All same value → variance ≈ 0 → normalised ≈ 0
        for &v in &out {
            assert!(v.abs() < 1e-2, "expected ~0, got {v}");
        }
    }

    #[test]
    fn fused_ln_beta_shift() {
        let n = 16;
        let inp = constant(n, 5.0);
        let res = constant(n, 0.0);
        let gamma = constant(n, 1.0);
        let beta = constant(n, 7.0);
        let eps = 1e-5;
        let mut out = vec![0.0f32; n];
        unsafe {
            neon_fused_residual_layernorm_f32(&inp, &res, &gamma, &beta, eps, &mut out);
        }
        // Constant → norm ≈ 0, so output ≈ beta
        for &v in &out {
            assert!((v - 7.0).abs() < 1e-2, "expected ~7.0, got {v}");
        }
    }

    #[test]
    fn fused_ln_zero_length() {
        let mut out: Vec<f32> = vec![];
        unsafe {
            neon_fused_residual_layernorm_f32(&[], &[], &[], &[], 1e-5, &mut out);
        }
        assert!(out.is_empty());
    }

    #[test]
    fn fused_ln_negative_values() {
        let n = 32;
        let inp: Vec<f32> = (0..n).map(|i| -(i as f32) * 0.3).collect();
        let res: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let gamma = constant(n, 2.0);
        let beta = constant(n, 1.0);
        let eps = 1e-5;
        let mut out = vec![0.0f32; n];
        unsafe {
            neon_fused_residual_layernorm_f32(&inp, &res, &gamma, &beta, eps, &mut out);
        }
        let expected = ref_fused_residual_layernorm(&inp, &res, &gamma, &beta, eps);
        assert_close(&out, &expected, 1e-4);
    }

    // ==============================================================
    // 5. neon_elementwise_mul_f32  (Hadamard)
    // ==============================================================

    #[test]
    fn hadamard_sizes() {
        for &n in SIZES {
            let a = ramp(n);
            let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.1).collect();
            let mut out = vec![0.0f32; n];
            unsafe { neon_elementwise_mul_f32(&a, &b, &mut out) };
            assert_close(&out, &ref_elementwise_mul(&a, &b), 1e-5);
        }
    }

    #[test]
    fn hadamard_ones() {
        for &n in SIZES {
            let a = ramp(n);
            let ones = constant(n, 1.0);
            let mut out = vec![0.0f32; n];
            unsafe { neon_elementwise_mul_f32(&a, &ones, &mut out) };
            assert_close(&out, &a, 1e-7);
        }
    }

    #[test]
    fn hadamard_zeros() {
        for &n in SIZES {
            let a = ramp(n);
            let z = constant(n, 0.0);
            let mut out = vec![0.0f32; n];
            unsafe { neon_elementwise_mul_f32(&a, &z, &mut out) };
            assert_close(&out, &z, 1e-7);
        }
    }

    #[test]
    fn hadamard_commutative() {
        for &n in SIZES {
            let a = ramp(n);
            let b = alternating(n);
            let mut out1 = vec![0.0f32; n];
            let mut out2 = vec![0.0f32; n];
            unsafe {
                neon_elementwise_mul_f32(&a, &b, &mut out1);
                neon_elementwise_mul_f32(&b, &a, &mut out2);
            }
            assert_close(&out1, &out2, 1e-7);
        }
    }

    #[test]
    fn hadamard_negative() {
        for &n in SIZES {
            let a: Vec<f32> = (0..n).map(|i| -(i as f32)).collect();
            let b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
            let mut out = vec![0.0f32; n];
            unsafe { neon_elementwise_mul_f32(&a, &b, &mut out) };
            assert_close(&out, &ref_elementwise_mul(&a, &b), 1e-5);
        }
    }

    #[test]
    fn hadamard_large_values() {
        let n = 16;
        let a = constant(n, 1e15);
        let b = constant(n, 1e15);
        let mut out = vec![0.0f32; n];
        unsafe { neon_elementwise_mul_f32(&a, &b, &mut out) };
        assert_close(&out, &constant(n, 1e30), 1e24);
    }

    #[test]
    fn hadamard_mixed_sign() {
        for &n in SIZES {
            let a = alternating(n);
            let b = ramp(n);
            let mut out = vec![0.0f32; n];
            unsafe { neon_elementwise_mul_f32(&a, &b, &mut out) };
            assert_close(&out, &ref_elementwise_mul(&a, &b), 1e-6);
        }
    }

    // ==============================================================
    // 6. neon_scale_f32
    // ==============================================================

    #[test]
    fn scale_sizes() {
        for &n in SIZES {
            let a = ramp(n);
            let s = 2.5;
            let mut out = vec![0.0f32; n];
            unsafe { neon_scale_f32(&a, s, &mut out) };
            assert_close(&out, &ref_scale(&a, s), 1e-5);
        }
    }

    #[test]
    fn scale_zero() {
        for &n in SIZES {
            let a = ramp(n);
            let mut out = vec![0.0f32; n];
            unsafe { neon_scale_f32(&a, 0.0, &mut out) };
            assert_close(&out, &constant(n, 0.0), 1e-7);
        }
    }

    #[test]
    fn scale_one_identity() {
        for &n in SIZES {
            let a = ramp(n);
            let mut out = vec![0.0f32; n];
            unsafe { neon_scale_f32(&a, 1.0, &mut out) };
            assert_close(&out, &a, 1e-7);
        }
    }

    #[test]
    fn scale_neg_one() {
        for &n in SIZES {
            let a = ramp(n);
            let mut out = vec![0.0f32; n];
            unsafe { neon_scale_f32(&a, -1.0, &mut out) };
            let expected: Vec<f32> = a.iter().map(|x| -x).collect();
            assert_close(&out, &expected, 1e-7);
        }
    }

    #[test]
    fn scale_half() {
        for &n in SIZES {
            let a = constant(n, 4.0);
            let mut out = vec![0.0f32; n];
            unsafe { neon_scale_f32(&a, 0.5, &mut out) };
            assert_close(&out, &constant(n, 2.0), 1e-7);
        }
    }

    #[test]
    fn scale_negative_data() {
        for &n in SIZES {
            let a: Vec<f32> = (0..n).map(|i| -(i as f32)).collect();
            let s = 3.0;
            let mut out = vec![0.0f32; n];
            unsafe { neon_scale_f32(&a, s, &mut out) };
            assert_close(&out, &ref_scale(&a, s), 1e-5);
        }
    }

    #[test]
    fn scale_large_value() {
        let n = 16;
        let a = constant(n, 1e20);
        let s = 1e10;
        let mut out = vec![0.0f32; n];
        unsafe { neon_scale_f32(&a, s, &mut out) };
        assert_close(&out, &constant(n, 1e30), 1e24);
    }

    // ==============================================================
    // 7. neon_add_bias_f32
    // ==============================================================

    #[test]
    fn bias_single_rep() {
        for &n in SIZES {
            let a = ramp(n);
            let bias: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
            let mut out = vec![0.0f32; n];
            unsafe { neon_add_bias_f32(&a, &bias, &mut out) };
            assert_close(&out, &ref_add_bias(&a, &bias), 1e-6);
        }
    }

    #[test]
    fn bias_broadcast() {
        let bias = vec![1.0, 2.0, 3.0, 4.0];
        let data = [0.0; 16]; // 4 reps of bias
        let mut out = [0.0f32; 16];
        unsafe { neon_add_bias_f32(&data, &bias, &mut out) };
        let expected = ref_add_bias(&data, &bias);
        assert_close(&out, &expected, 1e-7);
    }

    #[test]
    fn bias_broadcast_multi() {
        let bias = vec![10.0, 20.0];
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut out = [0.0f32; 8];
        unsafe { neon_add_bias_f32(&data, &bias, &mut out) };
        let expected = ref_add_bias(&data, &bias);
        assert_close(&out, &expected, 1e-7);
    }

    #[test]
    fn bias_zero_bias() {
        for &n in SIZES {
            let a = ramp(n);
            let bias = constant(n, 0.0);
            let mut out = vec![0.0f32; n];
            unsafe { neon_add_bias_f32(&a, &bias, &mut out) };
            assert_close(&out, &a, 1e-7);
        }
    }

    #[test]
    fn bias_negative() {
        let n = 16;
        let a = constant(n, 5.0);
        let bias = constant(n, -3.0);
        let mut out = vec![0.0f32; n];
        unsafe { neon_add_bias_f32(&a, &bias, &mut out) };
        assert_close(&out, &constant(n, 2.0), 1e-7);
    }

    #[test]
    fn bias_ones() {
        for &n in SIZES {
            let a = ramp(n);
            let bias = constant(n, 1.0);
            let mut out = vec![0.0f32; n];
            unsafe { neon_add_bias_f32(&a, &bias, &mut out) };
            let expected: Vec<f32> = a.iter().map(|x| x + 1.0).collect();
            assert_close(&out, &expected, 1e-7);
        }
    }

    #[test]
    fn bias_broadcast_sequence_positions() {
        // Simulates [batch=3, hidden=4] with bias of [hidden=4]
        let bias = vec![0.1, 0.2, 0.3, 0.4];
        let data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.5).collect(); // 3×4
        let mut out = [0.0f32; 12];
        unsafe { neon_add_bias_f32(&data, &bias, &mut out) };
        let expected = ref_add_bias(&data, &bias);
        assert_close(&out, &expected, 1e-6);
    }

    #[test]
    fn bias_large_values() {
        let n = 16;
        let a = constant(n, 1e30);
        let bias = constant(n, 1e30);
        let mut out = vec![0.0f32; n];
        unsafe { neon_add_bias_f32(&a, &bias, &mut out) };
        assert_close(&out, &constant(n, 2e30), 1e24);
    }

    // ==============================================================
    // Cross-function property tests
    // ==============================================================

    #[test]
    fn scale_then_add_equals_scaled_add() {
        for &n in &[8, 16, 64, 256] {
            let input = ramp(n);
            let residual = alternating(n);
            let alpha = 0.75;

            // Method 1: neon_residual_add_scale_f32
            let mut out1 = vec![0.0f32; n];
            unsafe {
                neon_residual_add_scale_f32(&input, &residual, alpha, &mut out1);
            }

            // Method 2: scale then add
            let mut scaled = vec![0.0f32; n];
            let mut out2 = vec![0.0f32; n];
            unsafe {
                neon_scale_f32(&residual, alpha, &mut scaled);
                neon_residual_add_f32(&input, &scaled, &mut out2);
            }
            assert_close(&out1, &out2, 1e-5);
        }
    }

    #[test]
    fn hadamard_ones_identity_property() {
        for &n in SIZES {
            let a = ramp(n);
            let ones = constant(n, 1.0);
            let mut out = vec![0.0f32; n];
            unsafe { neon_elementwise_mul_f32(&a, &ones, &mut out) };
            assert_close(&out, &a, 1e-7);
        }
    }

    #[test]
    fn scale_one_identity_property() {
        for &n in SIZES {
            let a = ramp(n);
            let mut out = vec![0.0f32; n];
            unsafe { neon_scale_f32(&a, 1.0, &mut out) };
            assert_close(&out, &a, 1e-7);
        }
    }

    #[test]
    fn residual_add_matches_scale_alpha1() {
        for &n in SIZES {
            let a = ramp(n);
            let b = alternating(n);
            let mut out_add = vec![0.0f32; n];
            let mut out_scaled = vec![0.0f32; n];
            unsafe {
                neon_residual_add_f32(&a, &b, &mut out_add);
                neon_residual_add_scale_f32(&a, &b, 1.0, &mut out_scaled);
            }
            assert_close(&out_add, &out_scaled, 1e-7);
        }
    }

    #[test]
    fn inplace_double_add_equals_scale2() {
        for &n in &[8, 32, 128] {
            let orig = ramp(n);
            let r = constant(n, 3.0);

            // Method 1: in-place add twice
            let mut data = orig.clone();
            unsafe {
                neon_residual_add_inplace(&mut data, &r);
                neon_residual_add_inplace(&mut data, &r);
            }

            // Method 2: scale residual by 2 then add once
            let mut out = vec![0.0f32; n];
            unsafe { neon_residual_add_scale_f32(&orig, &r, 2.0, &mut out) };

            assert_close(&data, &out, 1e-5);
        }
    }

    #[test]
    fn hadamard_scale_distributive() {
        // scale(a*b, s) == scale(a, s) * b
        for &n in &[8, 32, 64] {
            let a = ramp(n);
            let b = alternating(n);
            let s = 2.5;

            let mut prod = vec![0.0f32; n];
            let mut lhs = vec![0.0f32; n];
            unsafe {
                neon_elementwise_mul_f32(&a, &b, &mut prod);
                neon_scale_f32(&prod, s, &mut lhs);
            }

            let mut sa = vec![0.0f32; n];
            let mut rhs = vec![0.0f32; n];
            unsafe {
                neon_scale_f32(&a, s, &mut sa);
                neon_elementwise_mul_f32(&sa, &b, &mut rhs);
            }
            assert_close(&lhs, &rhs, 1e-4);
        }
    }

    #[test]
    fn bias_add_equivalent_residual_add() {
        // When bias.len() == data.len(), bias add == residual add
        for &n in SIZES {
            let a = ramp(n);
            let b = alternating(n);
            let mut out_bias = vec![0.0f32; n];
            let mut out_res = vec![0.0f32; n];
            unsafe {
                neon_add_bias_f32(&a, &b, &mut out_bias);
                neon_residual_add_f32(&a, &b, &mut out_res);
            }
            assert_close(&out_bias, &out_res, 1e-7);
        }
    }

    #[test]
    fn fused_ln_with_zero_residual_matches_plain() {
        for &n in &[8, 32, 128] {
            let inp = ramp(n);
            let zero_res = constant(n, 0.0);
            let gamma = constant(n, 1.0);
            let beta = constant(n, 0.0);
            let eps = 1e-5;

            let mut fused_out = vec![0.0f32; n];
            unsafe {
                neon_fused_residual_layernorm_f32(
                    &inp,
                    &zero_res,
                    &gamma,
                    &beta,
                    eps,
                    &mut fused_out,
                );
            }
            let expected = ref_fused_residual_layernorm(&inp, &zero_res, &gamma, &beta, eps);
            assert_close(&fused_out, &expected, 1e-4);
        }
    }

    #[test]
    fn fused_ln_gamma_scale_equivalence() {
        // gamma=2 should double the normalised values vs gamma=1
        let n = 64;
        let inp = ramp(n);
        let res = alternating(n);
        let gamma1 = constant(n, 1.0);
        let gamma2 = constant(n, 2.0);
        let beta = constant(n, 0.0);
        let eps = 1e-5;

        let mut out1 = vec![0.0f32; n];
        let mut out2 = vec![0.0f32; n];
        unsafe {
            neon_fused_residual_layernorm_f32(&inp, &res, &gamma1, &beta, eps, &mut out1);
            neon_fused_residual_layernorm_f32(&inp, &res, &gamma2, &beta, eps, &mut out2);
        }
        let doubled: Vec<f32> = out1.iter().map(|x| x * 2.0).collect();
        assert_close(&out2, &doubled, 1e-4);
    }

    // ==============================================================
    // Additional edge cases
    // ==============================================================

    #[test]
    fn residual_add_single_element() {
        let mut out = [0.0f32; 1];
        unsafe {
            neon_residual_add_f32(&[3.0], &[4.0], &mut out);
        }
        assert_close(&out, &[7.0], 1e-7);
    }

    #[test]
    fn scale_single_element() {
        let mut out = [0.0f32; 1];
        unsafe { neon_scale_f32(&[5.0], 3.0, &mut out) };
        assert_close(&out, &[15.0], 1e-7);
    }

    #[test]
    fn hadamard_single_element() {
        let mut out = [0.0f32; 1];
        unsafe { neon_elementwise_mul_f32(&[3.0], &[4.0], &mut out) };
        assert_close(&out, &[12.0], 1e-7);
    }

    #[test]
    fn bias_single_element() {
        let mut out = [0.0f32; 1];
        unsafe { neon_add_bias_f32(&[2.0], &[3.0], &mut out) };
        assert_close(&out, &[5.0], 1e-7);
    }

    #[test]
    fn inplace_single_element() {
        let mut data = [10.0f32];
        unsafe { neon_residual_add_inplace(&mut data, &[5.0]) };
        assert_close(&data, &[15.0], 1e-7);
    }

    #[test]
    fn scaled_add_single_element() {
        let mut out = [0.0f32; 1];
        unsafe {
            neon_residual_add_scale_f32(&[2.0], &[3.0], 0.5, &mut out);
        }
        assert_close(&out, &[3.5], 1e-7);
    }

    #[test]
    fn fused_ln_single_element() {
        let mut out = [0.0f32; 1];
        unsafe {
            neon_fused_residual_layernorm_f32(&[5.0], &[3.0], &[2.0], &[1.0], 1e-5, &mut out);
        }
        // Single element → variance ≈ 0, norm ≈ 0, output ≈ beta = 1.0
        assert!((out[0] - 1.0).abs() < 0.1, "got {}", out[0]);
    }

    #[test]
    fn residual_add_subnormal_values() {
        let n = 8;
        let a = constant(n, f32::MIN_POSITIVE);
        let b = constant(n, f32::MIN_POSITIVE);
        let mut out = vec![0.0f32; n];
        unsafe { neon_residual_add_f32(&a, &b, &mut out) };
        let expected = constant(n, 2.0 * f32::MIN_POSITIVE);
        assert_close(&out, &expected, 1e-40);
    }

    #[test]
    fn scale_negative_infinity() {
        let n = 4;
        let a = constant(n, 1.0);
        let mut out = vec![0.0f32; n];
        unsafe { neon_scale_f32(&a, f32::NEG_INFINITY, &mut out) };
        for &v in &out {
            assert!(v == f32::NEG_INFINITY);
        }
    }

    #[test]
    fn hadamard_self_is_square() {
        for &n in &[4, 8, 16, 64] {
            let a = ramp(n);
            let mut out = vec![0.0f32; n];
            unsafe { neon_elementwise_mul_f32(&a, &a, &mut out) };
            let expected: Vec<f32> = a.iter().map(|x| x * x).collect();
            assert_close(&out, &expected, 1e-5);
        }
    }
}
