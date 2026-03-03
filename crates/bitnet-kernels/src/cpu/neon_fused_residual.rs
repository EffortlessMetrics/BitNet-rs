//! NEON-accelerated fused residual connection kernels for ARM aarch64.
//!
//! Provides 8 variants of residual connections:
//! 1. Simple add
//! 2. Scaled residual
//! 3. Pre-norm residual
//! 4. Post-norm residual
//! 5. Gated residual
//! 6. Weighted residual
//! 7. Residual with dropout mask
//! 8. Stochastic depth

// ---------------------------------------------------------------------------
// Variant 1: Simple residual add — output = input + residual
// ---------------------------------------------------------------------------

/// Simple fused residual: `output[i] = input[i] + residual[i]`.
#[cfg(target_arch = "aarch64")]
pub fn neon_fused_residual_add(input: &[f32], residual: &[f32], output: &mut [f32]) {
    use std::arch::aarch64::*;

    let len = input.len().min(residual.len()).min(output.len());
    let chunks = len / 4;
    let tail = chunks * 4;

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let a = vld1q_f32(input.as_ptr().add(off));
            let b = vld1q_f32(residual.as_ptr().add(off));
            let c = vaddq_f32(a, b);
            vst1q_f32(output.as_mut_ptr().add(off), c);
        }
    }
    for i in tail..len {
        output[i] = input[i] + residual[i];
    }
}

/// Simple fused residual: `output[i] = input[i] + residual[i]` (scalar
/// fallback).
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_fused_residual_add(input: &[f32], residual: &[f32], output: &mut [f32]) {
    let len = input.len().min(residual.len()).min(output.len());
    for i in 0..len {
        output[i] = input[i] + residual[i];
    }
}

// ---------------------------------------------------------------------------
// Variant 2: Scaled residual — output = input + alpha * residual
// ---------------------------------------------------------------------------

/// Scaled residual: `output[i] = input[i] + alpha * residual[i]`.
#[cfg(target_arch = "aarch64")]
pub fn neon_fused_residual_scale(input: &[f32], residual: &[f32], alpha: f32, output: &mut [f32]) {
    use std::arch::aarch64::*;

    let len = input.len().min(residual.len()).min(output.len());
    let chunks = len / 4;
    let tail = chunks * 4;

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let a = vld1q_f32(input.as_ptr().add(off));
            let b = vld1q_f32(residual.as_ptr().add(off));
            let alpha_v = vdupq_n_f32(alpha);
            // input + alpha * residual  →  fma(a, b, alpha_v)
            let c = vfmaq_f32(a, b, alpha_v);
            vst1q_f32(output.as_mut_ptr().add(off), c);
        }
    }
    for i in tail..len {
        output[i] = input[i] + alpha * residual[i];
    }
}

/// Scaled residual (scalar fallback).
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_fused_residual_scale(input: &[f32], residual: &[f32], alpha: f32, output: &mut [f32]) {
    let len = input.len().min(residual.len()).min(output.len());
    for i in 0..len {
        output[i] = input[i] + alpha * residual[i];
    }
}

// ---------------------------------------------------------------------------
// Helpers: lightweight RMS-norm used by pre-norm / post-norm variants
// ---------------------------------------------------------------------------

/// Compute RMS norm of `src` into `dst`: `dst[i] = src[i] / rms(src)` where
/// `rms = sqrt(mean(src²) + eps)`.
#[cfg(target_arch = "aarch64")]
fn rms_norm_into(src: &[f32], dst: &mut [f32], eps: f32) {
    use std::arch::aarch64::*;

    let len = src.len().min(dst.len());
    if len == 0 {
        return;
    }

    // --- accumulate sum-of-squares via NEON ---
    let chunks = len / 4;
    let tail = chunks * 4;
    let mut sum_sq: f32;

    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let off = i * 4;
            let v = vld1q_f32(src.as_ptr().add(off));
            acc = vfmaq_f32(acc, v, v);
        }
        // horizontal add of 4 lanes
        sum_sq = vaddvq_f32(acc);
    }
    for i in tail..len {
        sum_sq += src[i] * src[i];
    }

    let rms = (sum_sq / len as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    // --- normalise ---
    unsafe {
        let inv_v = vdupq_n_f32(inv_rms);
        for i in 0..chunks {
            let off = i * 4;
            let v = vld1q_f32(src.as_ptr().add(off));
            let n = vmulq_f32(v, inv_v);
            vst1q_f32(dst.as_mut_ptr().add(off), n);
        }
    }
    for i in tail..len {
        dst[i] = src[i] * inv_rms;
    }
}

/// Compute RMS norm (scalar fallback).
#[cfg(not(target_arch = "aarch64"))]
fn rms_norm_into(src: &[f32], dst: &mut [f32], eps: f32) {
    let len = src.len().min(dst.len());
    if len == 0 {
        return;
    }
    let sum_sq: f32 = src[..len].iter().map(|&x| x * x).sum();
    let rms = (sum_sq / len as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    for i in 0..len {
        dst[i] = src[i] * inv_rms;
    }
}

// ---------------------------------------------------------------------------
// Variant 3: Pre-norm residual — output = norm(input) + residual
// ---------------------------------------------------------------------------

/// Pre-norm residual: `output = rms_norm(input) + residual`.
///
/// `eps` is the epsilon added inside the RMS denominator for numerical
/// stability (commonly 1e-5 or 1e-6).
#[cfg(target_arch = "aarch64")]
pub fn neon_fused_prenorm_residual(input: &[f32], residual: &[f32], eps: f32, output: &mut [f32]) {
    use std::arch::aarch64::*;

    let len = input.len().min(residual.len()).min(output.len());

    // normalise input into output (in-place buffer reuse)
    rms_norm_into(&input[..len], &mut output[..len], eps);

    // output += residual (NEON)
    let chunks = len / 4;
    let tail = chunks * 4;

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let a = vld1q_f32(output.as_ptr().add(off));
            let b = vld1q_f32(residual.as_ptr().add(off));
            let c = vaddq_f32(a, b);
            vst1q_f32(output.as_mut_ptr().add(off), c);
        }
    }
    for i in tail..len {
        output[i] += residual[i];
    }
}

/// Pre-norm residual (scalar fallback).
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_fused_prenorm_residual(input: &[f32], residual: &[f32], eps: f32, output: &mut [f32]) {
    let len = input.len().min(residual.len()).min(output.len());
    rms_norm_into(&input[..len], &mut output[..len], eps);
    for i in 0..len {
        output[i] += residual[i];
    }
}

// ---------------------------------------------------------------------------
// Variant 4: Post-norm residual — output = norm(input + residual)
// ---------------------------------------------------------------------------

/// Post-norm residual: `output = rms_norm(input + residual)`.
#[cfg(target_arch = "aarch64")]
pub fn neon_fused_postnorm_residual(input: &[f32], residual: &[f32], eps: f32, output: &mut [f32]) {
    use std::arch::aarch64::*;

    let len = input.len().min(residual.len()).min(output.len());
    let chunks = len / 4;
    let tail = chunks * 4;

    // Step 1: output = input + residual
    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let a = vld1q_f32(input.as_ptr().add(off));
            let b = vld1q_f32(residual.as_ptr().add(off));
            vst1q_f32(output.as_mut_ptr().add(off), vaddq_f32(a, b));
        }
    }
    for i in tail..len {
        output[i] = input[i] + residual[i];
    }

    // Step 2: in-place normalise output
    let mut tmp = vec![0.0f32; len];
    tmp.copy_from_slice(&output[..len]);
    rms_norm_into(&tmp, &mut output[..len], eps);
}

/// Post-norm residual (scalar fallback).
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_fused_postnorm_residual(input: &[f32], residual: &[f32], eps: f32, output: &mut [f32]) {
    let len = input.len().min(residual.len()).min(output.len());
    let mut tmp = vec![0.0f32; len];
    for i in 0..len {
        tmp[i] = input[i] + residual[i];
    }
    rms_norm_into(&tmp, &mut output[..len], eps);
}

// ---------------------------------------------------------------------------
// Variant 5: Gated residual — output = input + gate * residual
// ---------------------------------------------------------------------------

/// Gated residual: `output[i] = input[i] + gate[i] * residual[i]`.
///
/// `gate` is a per-element gating vector (e.g. sigmoid output).
#[cfg(target_arch = "aarch64")]
pub fn neon_fused_gated_residual(
    input: &[f32],
    residual: &[f32],
    gate: &[f32],
    output: &mut [f32],
) {
    use std::arch::aarch64::*;

    let len = input.len().min(residual.len()).min(gate.len()).min(output.len());
    let chunks = len / 4;
    let tail = chunks * 4;

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let a = vld1q_f32(input.as_ptr().add(off));
            let r = vld1q_f32(residual.as_ptr().add(off));
            let g = vld1q_f32(gate.as_ptr().add(off));
            // a + g * r
            let c = vfmaq_f32(a, g, r);
            vst1q_f32(output.as_mut_ptr().add(off), c);
        }
    }
    for i in tail..len {
        output[i] = input[i] + gate[i] * residual[i];
    }
}

/// Gated residual (scalar fallback).
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_fused_gated_residual(
    input: &[f32],
    residual: &[f32],
    gate: &[f32],
    output: &mut [f32],
) {
    let len = input.len().min(residual.len()).min(gate.len()).min(output.len());
    for i in 0..len {
        output[i] = input[i] + gate[i] * residual[i];
    }
}

// ---------------------------------------------------------------------------
// Variant 6: Weighted residual — output = w1*input + w2*residual
// ---------------------------------------------------------------------------

/// Weighted residual: `output[i] = w1 * input[i] + w2 * residual[i]`.
#[cfg(target_arch = "aarch64")]
pub fn neon_fused_weighted_residual(
    input: &[f32],
    residual: &[f32],
    w1: f32,
    w2: f32,
    output: &mut [f32],
) {
    use std::arch::aarch64::*;

    let len = input.len().min(residual.len()).min(output.len());
    let chunks = len / 4;
    let tail = chunks * 4;

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let a = vld1q_f32(input.as_ptr().add(off));
            let b = vld1q_f32(residual.as_ptr().add(off));
            let w1v = vdupq_n_f32(w1);
            let w2v = vdupq_n_f32(w2);
            // w1*a + w2*b  →  fma(w1*a, b, w2v)
            let wa = vmulq_f32(w1v, a);
            let c = vfmaq_f32(wa, w2v, b);
            vst1q_f32(output.as_mut_ptr().add(off), c);
        }
    }
    for i in tail..len {
        output[i] = w1 * input[i] + w2 * residual[i];
    }
}

/// Weighted residual (scalar fallback).
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_fused_weighted_residual(
    input: &[f32],
    residual: &[f32],
    w1: f32,
    w2: f32,
    output: &mut [f32],
) {
    let len = input.len().min(residual.len()).min(output.len());
    for i in 0..len {
        output[i] = w1 * input[i] + w2 * residual[i];
    }
}

// ---------------------------------------------------------------------------
// Variant 7: Residual with dropout mask
// ---------------------------------------------------------------------------

/// Residual with dropout: `output[i] = input[i] + mask[i] * residual[i]`
/// where `mask` contains `0.0` (dropped) or `1.0 / (1.0 - drop_prob)`
/// (kept & scaled).
///
/// The caller is responsible for generating the mask (e.g. from a PRNG).
#[cfg(target_arch = "aarch64")]
pub fn neon_fused_residual_dropout(
    input: &[f32],
    residual: &[f32],
    mask: &[f32],
    output: &mut [f32],
) {
    use std::arch::aarch64::*;

    let len = input.len().min(residual.len()).min(mask.len()).min(output.len());
    let chunks = len / 4;
    let tail = chunks * 4;

    for i in 0..chunks {
        let off = i * 4;
        unsafe {
            let a = vld1q_f32(input.as_ptr().add(off));
            let r = vld1q_f32(residual.as_ptr().add(off));
            let m = vld1q_f32(mask.as_ptr().add(off));
            // a + m * r
            let c = vfmaq_f32(a, m, r);
            vst1q_f32(output.as_mut_ptr().add(off), c);
        }
    }
    for i in tail..len {
        output[i] = input[i] + mask[i] * residual[i];
    }
}

/// Residual with dropout (scalar fallback).
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_fused_residual_dropout(
    input: &[f32],
    residual: &[f32],
    mask: &[f32],
    output: &mut [f32],
) {
    let len = input.len().min(residual.len()).min(mask.len()).min(output.len());
    for i in 0..len {
        output[i] = input[i] + mask[i] * residual[i];
    }
}

// ---------------------------------------------------------------------------
// Variant 8: Stochastic depth — skip residual entirely with probability p
// ---------------------------------------------------------------------------

/// Stochastic depth: if `keep` is `true` the residual is applied
/// (`output = input + residual`), otherwise the layer is skipped
/// (`output = input`).
///
/// During training the caller draws `keep = rand() >= drop_prob`.  At
/// inference time pass `keep = true` unconditionally.
#[cfg(target_arch = "aarch64")]
pub fn neon_fused_stochastic_depth(
    input: &[f32],
    residual: &[f32],
    keep: bool,
    output: &mut [f32],
) {
    use std::arch::aarch64::*;

    let len = input.len().min(residual.len()).min(output.len());
    let chunks = len / 4;
    let tail = chunks * 4;

    if keep {
        for i in 0..chunks {
            let off = i * 4;
            unsafe {
                let a = vld1q_f32(input.as_ptr().add(off));
                let b = vld1q_f32(residual.as_ptr().add(off));
                vst1q_f32(output.as_mut_ptr().add(off), vaddq_f32(a, b));
            }
        }
        for i in tail..len {
            output[i] = input[i] + residual[i];
        }
    } else {
        for i in 0..chunks {
            let off = i * 4;
            unsafe {
                let a = vld1q_f32(input.as_ptr().add(off));
                vst1q_f32(output.as_mut_ptr().add(off), a);
            }
        }
        for i in tail..len {
            output[i] = input[i];
        }
    }
}

/// Stochastic depth (scalar fallback).
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_fused_stochastic_depth(
    input: &[f32],
    residual: &[f32],
    keep: bool,
    output: &mut [f32],
) {
    let len = input.len().min(residual.len()).min(output.len());
    if keep {
        for i in 0..len {
            output[i] = input[i] + residual[i];
        }
    } else {
        output[..len].copy_from_slice(&input[..len]);
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Tolerance for floating-point comparisons
    const EPS: f32 = 1e-5;

    fn assert_approx(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "mismatch at index {i}: {x} vs {y} (diff={})",
                (x - y).abs()
            );
        }
    }

    // Helper: reference RMS norm for verification
    fn ref_rms_norm(src: &[f32], eps: f32) -> Vec<f32> {
        let n = src.len();
        if n == 0 {
            return vec![];
        }
        let sum_sq: f32 = src.iter().map(|&x| x * x).sum();
        let rms = (sum_sq / n as f32 + eps).sqrt();
        src.iter().map(|&x| x / rms).collect()
    }

    // =================================================================
    // Variant 1: neon_fused_residual_add
    // =================================================================

    #[test]
    fn test_add_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.5, 0.5, 0.5, 0.5];
        let mut out = vec![0.0; 4];
        neon_fused_residual_add(&a, &b, &mut out);
        assert_approx(&out, &[1.5, 2.5, 3.5, 4.5], EPS);
    }

    #[test]
    fn test_add_zeros() {
        let a = vec![0.0; 8];
        let b = vec![0.0; 8];
        let mut out = vec![0.0; 8];
        neon_fused_residual_add(&a, &b, &mut out);
        assert_approx(&out, &[0.0; 8], EPS);
    }

    #[test]
    fn test_add_negatives() {
        let a = vec![-1.0, -2.0, -3.0, -4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 4];
        neon_fused_residual_add(&a, &b, &mut out);
        assert_approx(&out, &[0.0; 4], EPS);
    }

    #[test]
    fn test_add_tail_elements() {
        // 5 elements: 4 via NEON + 1 tail
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let mut out = vec![0.0; 5];
        neon_fused_residual_add(&a, &b, &mut out);
        assert_approx(&out, &[11.0, 22.0, 33.0, 44.0, 55.0], EPS);
    }

    #[test]
    fn test_add_single_element() {
        let a = vec![3.14];
        let b = vec![2.71];
        let mut out = vec![0.0];
        neon_fused_residual_add(&a, &b, &mut out);
        assert_approx(&out, &[3.14 + 2.71], EPS);
    }

    #[test]
    fn test_add_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let mut out: Vec<f32> = vec![];
        neon_fused_residual_add(&a, &b, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_add_large() {
        let n = 1024;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
        let mut out = vec![0.0; n];
        neon_fused_residual_add(&a, &b, &mut out);
        for v in &out {
            assert_approx(&[*v], &[n as f32], EPS);
        }
    }

    #[test]
    fn test_add_identity_residual_zero() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![0.0; 8];
        let mut out = vec![0.0; 8];
        neon_fused_residual_add(&a, &b, &mut out);
        assert_approx(&out, &a, EPS);
    }

    #[test]
    fn test_add_mismatched_shorter_output() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0; 3]; // shorter
        neon_fused_residual_add(&a, &b, &mut out);
        assert_approx(&out, &[2.0, 3.0, 4.0], EPS);
    }

    #[test]
    fn test_add_two_elements() {
        let a = vec![10.0, 20.0];
        let b = vec![5.0, 15.0];
        let mut out = vec![0.0; 2];
        neon_fused_residual_add(&a, &b, &mut out);
        assert_approx(&out, &[15.0, 35.0], EPS);
    }

    #[test]
    fn test_add_three_elements() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let mut out = vec![0.0; 3];
        neon_fused_residual_add(&a, &b, &mut out);
        assert_approx(&out, &[5.0, 7.0, 9.0], EPS);
    }

    // =================================================================
    // Variant 2: neon_fused_residual_scale
    // =================================================================

    #[test]
    fn test_scale_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![10.0, 20.0, 30.0, 40.0];
        let mut out = vec![0.0; 4];
        neon_fused_residual_scale(&a, &b, 0.5, &mut out);
        assert_approx(&out, &[6.0, 12.0, 18.0, 24.0], EPS);
    }

    #[test]
    fn test_scale_zero_alpha() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![100.0; 4];
        let mut out = vec![0.0; 4];
        neon_fused_residual_scale(&a, &b, 0.0, &mut out);
        assert_approx(&out, &a, EPS);
    }

    #[test]
    fn test_scale_one_alpha() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut out = vec![0.0; 4];
        neon_fused_residual_scale(&a, &b, 1.0, &mut out);
        assert_approx(&out, &[6.0, 8.0, 10.0, 12.0], EPS);
    }

    #[test]
    fn test_scale_negative_alpha() {
        let a = vec![10.0, 20.0, 30.0, 40.0];
        let b = vec![10.0, 20.0, 30.0, 40.0];
        let mut out = vec![0.0; 4];
        neon_fused_residual_scale(&a, &b, -1.0, &mut out);
        assert_approx(&out, &[0.0, 0.0, 0.0, 0.0], EPS);
    }

    #[test]
    fn test_scale_tail() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = vec![1.0; 7];
        let mut out = vec![0.0; 7];
        neon_fused_residual_scale(&a, &b, 2.0, &mut out);
        assert_approx(&out, &[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], EPS);
    }

    #[test]
    fn test_scale_empty() {
        let mut out: Vec<f32> = vec![];
        neon_fused_residual_scale(&[], &[], 1.0, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_scale_large_alpha() {
        let a = vec![0.0; 4];
        let b = vec![1.0; 4];
        let mut out = vec![0.0; 4];
        neon_fused_residual_scale(&a, &b, 1000.0, &mut out);
        assert_approx(&out, &[1000.0; 4], EPS);
    }

    #[test]
    fn test_scale_fractional_alpha() {
        let a = vec![0.0; 8];
        let b = vec![4.0; 8];
        let mut out = vec![0.0; 8];
        neon_fused_residual_scale(&a, &b, 0.25, &mut out);
        assert_approx(&out, &[1.0; 8], EPS);
    }

    #[test]
    fn test_scale_single_element() {
        let mut out = vec![0.0];
        neon_fused_residual_scale(&[2.0], &[3.0], 0.5, &mut out);
        assert_approx(&out, &[3.5], EPS);
    }

    #[test]
    fn test_scale_identity_when_residual_zero() {
        let a = vec![7.0, 8.0, 9.0, 10.0];
        let b = vec![0.0; 4];
        let mut out = vec![0.0; 4];
        neon_fused_residual_scale(&a, &b, 42.0, &mut out);
        assert_approx(&out, &a, EPS);
    }

    // =================================================================
    // Variant 3: neon_fused_prenorm_residual
    // =================================================================

    #[test]
    fn test_prenorm_basic() {
        let input = vec![3.0, 4.0, 0.0, 0.0];
        let residual = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0; 4];
        let eps = 1e-5;
        neon_fused_prenorm_residual(&input, &residual, eps, &mut out);
        let normed = ref_rms_norm(&input, eps);
        let expected: Vec<f32> = normed.iter().zip(residual.iter()).map(|(n, r)| n + r).collect();
        assert_approx(&out, &expected, 1e-4);
    }

    #[test]
    fn test_prenorm_zero_input() {
        let input = vec![0.0; 4];
        let residual = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 4];
        neon_fused_prenorm_residual(&input, &residual, 1e-5, &mut out);
        // norm(0) ≈ 0, so output ≈ residual
        assert_approx(&out, &residual, 1e-2);
    }

    #[test]
    fn test_prenorm_uniform_input() {
        let input = vec![2.0; 8];
        let residual = vec![0.0; 8];
        let mut out = vec![0.0; 8];
        let eps = 1e-6;
        neon_fused_prenorm_residual(&input, &residual, eps, &mut out);
        // norm of uniform vector: each element → x / rms(x) ≈ 1.0
        let normed = ref_rms_norm(&input, eps);
        assert_approx(&out, &normed, 1e-4);
    }

    #[test]
    fn test_prenorm_tail() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let residual = vec![0.1; 5];
        let mut out = vec![0.0; 5];
        let eps = 1e-5;
        neon_fused_prenorm_residual(&input, &residual, eps, &mut out);
        let normed = ref_rms_norm(&input, eps);
        let expected: Vec<f32> = normed.iter().zip(residual.iter()).map(|(n, r)| n + r).collect();
        assert_approx(&out, &expected, 1e-4);
    }

    #[test]
    fn test_prenorm_empty() {
        let mut out: Vec<f32> = vec![];
        neon_fused_prenorm_residual(&[], &[], 1e-5, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_prenorm_single() {
        let input = vec![5.0];
        let residual = vec![1.0];
        let mut out = vec![0.0];
        let eps = 1e-5;
        neon_fused_prenorm_residual(&input, &residual, eps, &mut out);
        let normed = ref_rms_norm(&input, eps);
        assert_approx(&out, &[normed[0] + 1.0], 1e-4);
    }

    #[test]
    fn test_prenorm_negative_input() {
        let input = vec![-3.0, -4.0, -5.0, -6.0];
        let residual = vec![0.0; 4];
        let mut out = vec![0.0; 4];
        let eps = 1e-5;
        neon_fused_prenorm_residual(&input, &residual, eps, &mut out);
        let expected = ref_rms_norm(&input, eps);
        assert_approx(&out, &expected, 1e-4);
    }

    #[test]
    fn test_prenorm_large_eps() {
        let input = vec![1.0; 4];
        let residual = vec![0.0; 4];
        let mut out = vec![0.0; 4];
        neon_fused_prenorm_residual(&input, &residual, 100.0, &mut out);
        let expected = ref_rms_norm(&input, 100.0);
        assert_approx(&out, &expected, 1e-4);
    }

    #[test]
    fn test_prenorm_residual_dominates() {
        let input = vec![0.001; 4];
        let residual = vec![100.0; 4];
        let mut out = vec![0.0; 4];
        neon_fused_prenorm_residual(&input, &residual, 1e-5, &mut out);
        // output ≈ 100 since norm(tiny) ≈ 1.0 (uniform → 1.0)
        for v in &out {
            assert!(*v > 99.0 && *v < 102.0);
        }
    }

    // =================================================================
    // Variant 4: neon_fused_postnorm_residual
    // =================================================================

    #[test]
    fn test_postnorm_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, -1.0, 0.0];
        let mut out = vec![0.0; 4];
        let eps = 1e-5;
        neon_fused_postnorm_residual(&a, &b, eps, &mut out);
        let sum: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        let expected = ref_rms_norm(&sum, eps);
        assert_approx(&out, &expected, 1e-4);
    }

    #[test]
    fn test_postnorm_zero_sum() {
        let a = vec![1.0, -1.0, 1.0, -1.0];
        let b = vec![-1.0, 1.0, -1.0, 1.0];
        let mut out = vec![0.0; 4];
        neon_fused_postnorm_residual(&a, &b, 1e-5, &mut out);
        // sum is zero → norm(0) ≈ 0
        for v in &out {
            assert!(v.abs() < 1e-2);
        }
    }

    #[test]
    fn test_postnorm_uniform_sum() {
        let a = vec![1.0; 8];
        let b = vec![1.0; 8];
        let mut out = vec![0.0; 8];
        let eps = 1e-6;
        neon_fused_postnorm_residual(&a, &b, eps, &mut out);
        // sum=[2..] → norm(uniform) ≈ 1.0 each
        let expected = ref_rms_norm(&vec![2.0; 8], eps);
        assert_approx(&out, &expected, 1e-4);
    }

    #[test]
    fn test_postnorm_tail() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = vec![0.0; 7];
        let mut out = vec![0.0; 7];
        let eps = 1e-5;
        neon_fused_postnorm_residual(&a, &b, eps, &mut out);
        let expected = ref_rms_norm(&a, eps);
        assert_approx(&out, &expected, 1e-4);
    }

    #[test]
    fn test_postnorm_empty() {
        let mut out: Vec<f32> = vec![];
        neon_fused_postnorm_residual(&[], &[], 1e-5, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_postnorm_single() {
        let a = vec![3.0];
        let b = vec![2.0];
        let mut out = vec![0.0];
        let eps = 1e-5;
        neon_fused_postnorm_residual(&a, &b, eps, &mut out);
        let expected = ref_rms_norm(&[5.0], eps);
        assert_approx(&out, &expected, 1e-4);
    }

    #[test]
    fn test_postnorm_negative_values() {
        let a = vec![-5.0, -3.0, -1.0, 0.0];
        let b = vec![-1.0, -2.0, -3.0, -4.0];
        let mut out = vec![0.0; 4];
        let eps = 1e-5;
        neon_fused_postnorm_residual(&a, &b, eps, &mut out);
        let sum: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        let expected = ref_rms_norm(&sum, eps);
        assert_approx(&out, &expected, 1e-4);
    }

    #[test]
    fn test_postnorm_preserves_direction() {
        let a = vec![2.0, 4.0, 6.0, 8.0];
        let b = vec![0.0; 4];
        let mut out = vec![0.0; 4];
        neon_fused_postnorm_residual(&a, &b, 1e-6, &mut out);
        // All positive input → all positive output
        for v in &out {
            assert!(*v > 0.0);
        }
        // Ratios preserved: out[1]/out[0] ≈ 2.0
        assert!((out[1] / out[0] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_postnorm_large_values() {
        let a = vec![1e6; 4];
        let b = vec![1e6; 4];
        let mut out = vec![0.0; 4];
        neon_fused_postnorm_residual(&a, &b, 1e-5, &mut out);
        // uniform → all ≈ 1.0
        assert_approx(&out, &[1.0; 4], 1e-4);
    }

    // =================================================================
    // Variant 5: neon_fused_gated_residual
    // =================================================================

    #[test]
    fn test_gated_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![10.0, 20.0, 30.0, 40.0];
        let gate = vec![0.5, 0.5, 0.5, 0.5];
        let mut out = vec![0.0; 4];
        neon_fused_gated_residual(&input, &residual, &gate, &mut out);
        assert_approx(&out, &[6.0, 12.0, 18.0, 24.0], EPS);
    }

    #[test]
    fn test_gated_zero_gate() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![100.0; 4];
        let gate = vec![0.0; 4];
        let mut out = vec![0.0; 4];
        neon_fused_gated_residual(&input, &residual, &gate, &mut out);
        assert_approx(&out, &input, EPS);
    }

    #[test]
    fn test_gated_one_gate() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![5.0, 6.0, 7.0, 8.0];
        let gate = vec![1.0; 4];
        let mut out = vec![0.0; 4];
        neon_fused_gated_residual(&input, &residual, &gate, &mut out);
        assert_approx(&out, &[6.0, 8.0, 10.0, 12.0], EPS);
    }

    #[test]
    fn test_gated_mixed_gate() {
        let input = vec![0.0; 4];
        let residual = vec![10.0; 4];
        let gate = vec![0.0, 0.25, 0.5, 1.0];
        let mut out = vec![0.0; 4];
        neon_fused_gated_residual(&input, &residual, &gate, &mut out);
        assert_approx(&out, &[0.0, 2.5, 5.0, 10.0], EPS);
    }

    #[test]
    fn test_gated_tail() {
        let input = vec![1.0; 6];
        let residual = vec![2.0; 6];
        let gate = vec![0.5; 6];
        let mut out = vec![0.0; 6];
        neon_fused_gated_residual(&input, &residual, &gate, &mut out);
        assert_approx(&out, &[2.0; 6], EPS);
    }

    #[test]
    fn test_gated_empty() {
        let mut out: Vec<f32> = vec![];
        neon_fused_gated_residual(&[], &[], &[], &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_gated_single() {
        let mut out = vec![0.0];
        neon_fused_gated_residual(&[3.0], &[4.0], &[0.5], &mut out);
        assert_approx(&out, &[5.0], EPS);
    }

    #[test]
    fn test_gated_negative_gate() {
        let input = vec![10.0; 4];
        let residual = vec![10.0; 4];
        let gate = vec![-1.0; 4];
        let mut out = vec![0.0; 4];
        neon_fused_gated_residual(&input, &residual, &gate, &mut out);
        assert_approx(&out, &[0.0; 4], EPS);
    }

    #[test]
    fn test_gated_large_vector() {
        let n = 513;
        let input = vec![1.0; n];
        let residual = vec![2.0; n];
        let gate = vec![0.5; n];
        let mut out = vec![0.0; n];
        neon_fused_gated_residual(&input, &residual, &gate, &mut out);
        assert_approx(&out, &vec![2.0; n], EPS);
    }

    #[test]
    fn test_gated_per_element() {
        let input = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let residual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gate = vec![1.0, 0.5, 0.0, 0.5, 1.0];
        let mut out = vec![0.0; 5];
        neon_fused_gated_residual(&input, &residual, &gate, &mut out);
        assert_approx(&out, &[1.0, 1.0, 0.0, 2.0, 5.0], EPS);
    }

    // =================================================================
    // Variant 6: neon_fused_weighted_residual
    // =================================================================

    #[test]
    fn test_weighted_basic() {
        let a = vec![2.0; 4];
        let b = vec![3.0; 4];
        let mut out = vec![0.0; 4];
        neon_fused_weighted_residual(&a, &b, 0.5, 0.5, &mut out);
        assert_approx(&out, &[2.5; 4], EPS);
    }

    #[test]
    fn test_weighted_input_only() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![10.0; 4];
        let mut out = vec![0.0; 4];
        neon_fused_weighted_residual(&a, &b, 1.0, 0.0, &mut out);
        assert_approx(&out, &a, EPS);
    }

    #[test]
    fn test_weighted_residual_only() {
        let a = vec![10.0; 4];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 4];
        neon_fused_weighted_residual(&a, &b, 0.0, 1.0, &mut out);
        assert_approx(&out, &b, EPS);
    }

    #[test]
    fn test_weighted_equal_weights() {
        let a = vec![10.0; 4];
        let b = vec![20.0; 4];
        let mut out = vec![0.0; 4];
        neon_fused_weighted_residual(&a, &b, 1.0, 1.0, &mut out);
        assert_approx(&out, &[30.0; 4], EPS);
    }

    #[test]
    fn test_weighted_negative_weights() {
        let a = vec![5.0; 4];
        let b = vec![5.0; 4];
        let mut out = vec![0.0; 4];
        neon_fused_weighted_residual(&a, &b, -1.0, -1.0, &mut out);
        assert_approx(&out, &[-10.0; 4], EPS);
    }

    #[test]
    fn test_weighted_tail() {
        let a = vec![1.0; 9];
        let b = vec![2.0; 9];
        let mut out = vec![0.0; 9];
        neon_fused_weighted_residual(&a, &b, 3.0, 4.0, &mut out);
        assert_approx(&out, &[11.0; 9], EPS);
    }

    #[test]
    fn test_weighted_empty() {
        let mut out: Vec<f32> = vec![];
        neon_fused_weighted_residual(&[], &[], 1.0, 1.0, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_weighted_single() {
        let mut out = vec![0.0];
        neon_fused_weighted_residual(&[3.0], &[7.0], 0.5, 0.5, &mut out);
        assert_approx(&out, &[5.0], EPS);
    }

    #[test]
    fn test_weighted_zero_weights() {
        let a = vec![100.0; 4];
        let b = vec![200.0; 4];
        let mut out = vec![0.0; 4];
        neon_fused_weighted_residual(&a, &b, 0.0, 0.0, &mut out);
        assert_approx(&out, &[0.0; 4], EPS);
    }

    #[test]
    fn test_weighted_complementary() {
        // w1 + w2 = 1 (convex combination)
        let a = vec![0.0; 4];
        let b = vec![10.0; 4];
        let mut out = vec![0.0; 4];
        neon_fused_weighted_residual(&a, &b, 0.3, 0.7, &mut out);
        assert_approx(&out, &[7.0; 4], EPS);
    }

    #[test]
    fn test_weighted_large() {
        let n = 1025;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (2 * i) as f32).collect();
        let mut out = vec![0.0; n];
        neon_fused_weighted_residual(&a, &b, 1.0, 1.0, &mut out);
        for i in 0..n {
            assert_approx(&[out[i]], &[3.0 * i as f32], EPS);
        }
    }

    // =================================================================
    // Variant 7: neon_fused_residual_dropout
    // =================================================================

    #[test]
    fn test_dropout_all_kept() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![5.0, 6.0, 7.0, 8.0];
        let mask = vec![1.0; 4]; // all kept (no scaling for simplicity)
        let mut out = vec![0.0; 4];
        neon_fused_residual_dropout(&input, &residual, &mask, &mut out);
        assert_approx(&out, &[6.0, 8.0, 10.0, 12.0], EPS);
    }

    #[test]
    fn test_dropout_all_dropped() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![100.0; 4];
        let mask = vec![0.0; 4];
        let mut out = vec![0.0; 4];
        neon_fused_residual_dropout(&input, &residual, &mask, &mut out);
        assert_approx(&out, &input, EPS);
    }

    #[test]
    fn test_dropout_alternating() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![10.0, 10.0, 10.0, 10.0];
        let mask = vec![1.0, 0.0, 1.0, 0.0];
        let mut out = vec![0.0; 4];
        neon_fused_residual_dropout(&input, &residual, &mask, &mut out);
        assert_approx(&out, &[11.0, 2.0, 13.0, 4.0], EPS);
    }

    #[test]
    fn test_dropout_scaled_mask() {
        // Inverted dropout: mask = 1/(1-p) for kept, 0 for dropped
        let p = 0.5_f32;
        let scale = 1.0 / (1.0 - p);
        let input = vec![0.0; 4];
        let residual = vec![2.0; 4];
        let mask = vec![scale, 0.0, scale, 0.0];
        let mut out = vec![0.0; 4];
        neon_fused_residual_dropout(&input, &residual, &mask, &mut out);
        assert_approx(&out, &[4.0, 0.0, 4.0, 0.0], EPS);
    }

    #[test]
    fn test_dropout_tail() {
        let input = vec![1.0; 5];
        let residual = vec![2.0; 5];
        let mask = vec![1.0, 0.0, 1.0, 0.0, 1.0];
        let mut out = vec![0.0; 5];
        neon_fused_residual_dropout(&input, &residual, &mask, &mut out);
        assert_approx(&out, &[3.0, 1.0, 3.0, 1.0, 3.0], EPS);
    }

    #[test]
    fn test_dropout_empty() {
        let mut out: Vec<f32> = vec![];
        neon_fused_residual_dropout(&[], &[], &[], &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_dropout_single_kept() {
        let mut out = vec![0.0];
        neon_fused_residual_dropout(&[1.0], &[9.0], &[1.0], &mut out);
        assert_approx(&out, &[10.0], EPS);
    }

    #[test]
    fn test_dropout_single_dropped() {
        let mut out = vec![0.0];
        neon_fused_residual_dropout(&[1.0], &[9.0], &[0.0], &mut out);
        assert_approx(&out, &[1.0], EPS);
    }

    #[test]
    fn test_dropout_large() {
        let n = 512;
        let input = vec![1.0; n];
        let residual = vec![2.0; n];
        let mask: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let mut out = vec![0.0; n];
        neon_fused_residual_dropout(&input, &residual, &mask, &mut out);
        for i in 0..n {
            let expected = if i % 2 == 0 { 3.0 } else { 1.0 };
            assert_approx(&[out[i]], &[expected], EPS);
        }
    }

    #[test]
    fn test_dropout_fractional_mask() {
        let input = vec![0.0; 4];
        let residual = vec![10.0; 4];
        let mask = vec![0.1, 0.5, 0.9, 1.0];
        let mut out = vec![0.0; 4];
        neon_fused_residual_dropout(&input, &residual, &mask, &mut out);
        assert_approx(&out, &[1.0, 5.0, 9.0, 10.0], EPS);
    }

    // =================================================================
    // Variant 8: neon_fused_stochastic_depth
    // =================================================================

    #[test]
    fn test_stochastic_keep() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut out = vec![0.0; 4];
        neon_fused_stochastic_depth(&a, &b, true, &mut out);
        assert_approx(&out, &[6.0, 8.0, 10.0, 12.0], EPS);
    }

    #[test]
    fn test_stochastic_drop() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![100.0; 4];
        let mut out = vec![0.0; 4];
        neon_fused_stochastic_depth(&a, &b, false, &mut out);
        assert_approx(&out, &a, EPS);
    }

    #[test]
    fn test_stochastic_keep_tail() {
        let a = vec![1.0; 7];
        let b = vec![2.0; 7];
        let mut out = vec![0.0; 7];
        neon_fused_stochastic_depth(&a, &b, true, &mut out);
        assert_approx(&out, &[3.0; 7], EPS);
    }

    #[test]
    fn test_stochastic_drop_tail() {
        let a = vec![5.0; 7];
        let b = vec![99.0; 7];
        let mut out = vec![0.0; 7];
        neon_fused_stochastic_depth(&a, &b, false, &mut out);
        assert_approx(&out, &[5.0; 7], EPS);
    }

    #[test]
    fn test_stochastic_keep_zeros() {
        let a = vec![0.0; 8];
        let b = vec![0.0; 8];
        let mut out = vec![0.0; 8];
        neon_fused_stochastic_depth(&a, &b, true, &mut out);
        assert_approx(&out, &[0.0; 8], EPS);
    }

    #[test]
    fn test_stochastic_drop_zeros() {
        let a = vec![0.0; 8];
        let b = vec![1.0; 8];
        let mut out = vec![0.0; 8];
        neon_fused_stochastic_depth(&a, &b, false, &mut out);
        assert_approx(&out, &[0.0; 8], EPS);
    }

    #[test]
    fn test_stochastic_empty() {
        let mut out: Vec<f32> = vec![];
        neon_fused_stochastic_depth(&[], &[], true, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_stochastic_single_keep() {
        let mut out = vec![0.0];
        neon_fused_stochastic_depth(&[3.0], &[7.0], true, &mut out);
        assert_approx(&out, &[10.0], EPS);
    }

    #[test]
    fn test_stochastic_single_drop() {
        let mut out = vec![0.0];
        neon_fused_stochastic_depth(&[3.0], &[7.0], false, &mut out);
        assert_approx(&out, &[3.0], EPS);
    }

    #[test]
    fn test_stochastic_large_keep() {
        let n = 1024;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b = vec![1.0; n];
        let mut out = vec![0.0; n];
        neon_fused_stochastic_depth(&a, &b, true, &mut out);
        for i in 0..n {
            assert_approx(&[out[i]], &[i as f32 + 1.0], EPS);
        }
    }

    #[test]
    fn test_stochastic_large_drop() {
        let n = 1024;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b = vec![999.0; n];
        let mut out = vec![0.0; n];
        neon_fused_stochastic_depth(&a, &b, false, &mut out);
        for i in 0..n {
            assert_approx(&[out[i]], &[i as f32], EPS);
        }
    }

    #[test]
    fn test_stochastic_drop_preserves_input_exactly() {
        let a = vec![1.5, 2.5, 3.5, 4.5, 5.5];
        let b = vec![-1.0; 5];
        let mut out = vec![0.0; 5];
        neon_fused_stochastic_depth(&a, &b, false, &mut out);
        // Exact equality for drop path (no arithmetic)
        assert_eq!(out, a);
    }

    #[test]
    fn test_stochastic_keep_negative() {
        let a = vec![-1.0, -2.0, -3.0, -4.0];
        let b = vec![-5.0, -6.0, -7.0, -8.0];
        let mut out = vec![0.0; 4];
        neon_fused_stochastic_depth(&a, &b, true, &mut out);
        assert_approx(&out, &[-6.0, -8.0, -10.0, -12.0], EPS);
    }

    // =================================================================
    // Cross-variant consistency tests
    // =================================================================

    #[test]
    fn test_scale_one_equals_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let mut out_add = vec![0.0; 5];
        let mut out_scale = vec![0.0; 5];
        neon_fused_residual_add(&a, &b, &mut out_add);
        neon_fused_residual_scale(&a, &b, 1.0, &mut out_scale);
        assert_approx(&out_add, &out_scale, EPS);
    }

    #[test]
    fn test_gate_ones_equals_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let gate = vec![1.0; 4];
        let mut out_add = vec![0.0; 4];
        let mut out_gated = vec![0.0; 4];
        neon_fused_residual_add(&a, &b, &mut out_add);
        neon_fused_gated_residual(&a, &b, &gate, &mut out_gated);
        assert_approx(&out_add, &out_gated, EPS);
    }

    #[test]
    fn test_weighted_11_equals_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut out_add = vec![0.0; 6];
        let mut out_w = vec![0.0; 6];
        neon_fused_residual_add(&a, &b, &mut out_add);
        neon_fused_weighted_residual(&a, &b, 1.0, 1.0, &mut out_w);
        assert_approx(&out_add, &out_w, EPS);
    }

    #[test]
    fn test_dropout_ones_equals_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mask = vec![1.0; 4];
        let mut out_add = vec![0.0; 4];
        let mut out_drop = vec![0.0; 4];
        neon_fused_residual_add(&a, &b, &mut out_add);
        neon_fused_residual_dropout(&a, &b, &mask, &mut out_drop);
        assert_approx(&out_add, &out_drop, EPS);
    }

    #[test]
    fn test_stochastic_keep_equals_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        let mut out_add = vec![0.0; 5];
        let mut out_sd = vec![0.0; 5];
        neon_fused_residual_add(&a, &b, &mut out_add);
        neon_fused_stochastic_depth(&a, &b, true, &mut out_sd);
        assert_approx(&out_add, &out_sd, EPS);
    }

    #[test]
    fn test_gated_zero_equals_identity() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![999.0; 4];
        let gate = vec![0.0; 4];
        let mut out = vec![0.0; 4];
        neon_fused_gated_residual(&a, &b, &gate, &mut out);
        assert_approx(&out, &a, EPS);
    }

    #[test]
    fn test_scale_zero_equals_identity() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![999.0; 4];
        let mut out = vec![0.0; 4];
        neon_fused_residual_scale(&a, &b, 0.0, &mut out);
        assert_approx(&out, &a, EPS);
    }

    #[test]
    fn test_stochastic_drop_equals_identity() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![999.0; 6];
        let mut out = vec![0.0; 6];
        neon_fused_stochastic_depth(&a, &b, false, &mut out);
        assert_approx(&out, &a, EPS);
    }

    #[test]
    fn test_dropout_zeros_equals_identity() {
        let a = vec![7.0, 8.0, 9.0, 10.0, 11.0];
        let b = vec![999.0; 5];
        let mask = vec![0.0; 5];
        let mut out = vec![0.0; 5];
        neon_fused_residual_dropout(&a, &b, &mask, &mut out);
        assert_approx(&out, &a, EPS);
    }

    // =================================================================
    // Norm-specific edge cases
    // =================================================================

    #[test]
    fn test_prenorm_postnorm_agree_on_zero_residual() {
        let input = vec![3.0, 4.0, 5.0, 6.0];
        let zero_res = vec![0.0; 4];
        let eps = 1e-5;
        let mut out_pre = vec![0.0; 4];
        let mut out_post = vec![0.0; 4];
        neon_fused_prenorm_residual(&input, &zero_res, eps, &mut out_pre);
        neon_fused_postnorm_residual(&input, &zero_res, eps, &mut out_post);
        // With zero residual both should produce rms_norm(input)
        assert_approx(&out_pre, &out_post, 1e-4);
    }

    #[test]
    fn test_prenorm_with_two_elements() {
        let input = vec![3.0, 4.0];
        let residual = vec![0.0, 0.0];
        let mut out = vec![0.0; 2];
        neon_fused_prenorm_residual(&input, &residual, 1e-6, &mut out);
        let expected = ref_rms_norm(&input, 1e-6);
        assert_approx(&out, &expected, 1e-4);
    }

    #[test]
    fn test_postnorm_with_two_elements() {
        let input = vec![3.0, 4.0];
        let residual = vec![1.0, 2.0];
        let mut out = vec![0.0; 2];
        neon_fused_postnorm_residual(&input, &residual, 1e-6, &mut out);
        let expected = ref_rms_norm(&[4.0, 6.0], 1e-6);
        assert_approx(&out, &expected, 1e-4);
    }

    // =================================================================
    // Mismatched-length safety tests
    // =================================================================

    #[test]
    fn test_add_input_shorter() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0, 5.0, 6.0];
        let mut out = vec![0.0; 4];
        neon_fused_residual_add(&a, &b, &mut out);
        // min(2,4,4)=2 processed
        assert_approx(&out[..2], &[4.0, 6.0], EPS);
    }

    #[test]
    fn test_add_residual_shorter() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![10.0, 20.0];
        let mut out = vec![0.0; 4];
        neon_fused_residual_add(&a, &b, &mut out);
        assert_approx(&out[..2], &[11.0, 22.0], EPS);
    }

    #[test]
    fn test_gated_mismatched_lengths() {
        let a = vec![1.0; 6];
        let b = vec![2.0; 4];
        let g = vec![0.5; 5];
        let mut out = vec![0.0; 8];
        neon_fused_gated_residual(&a, &b, &g, &mut out);
        // min(6,4,5,8)=4
        assert_approx(&out[..4], &[2.0; 4], EPS);
    }

    #[test]
    fn test_weighted_mismatched() {
        let a = vec![1.0; 3];
        let b = vec![2.0; 5];
        let mut out = vec![0.0; 4];
        neon_fused_weighted_residual(&a, &b, 1.0, 1.0, &mut out);
        // min(3,5,4)=3
        assert_approx(&out[..3], &[3.0; 3], EPS);
    }

    #[test]
    fn test_dropout_mismatched() {
        let a = vec![1.0; 5];
        let b = vec![2.0; 3];
        let m = vec![1.0; 4];
        let mut out = vec![0.0; 6];
        neon_fused_residual_dropout(&a, &b, &m, &mut out);
        // min(5,3,4,6)=3
        assert_approx(&out[..3], &[3.0; 3], EPS);
    }

    // =================================================================
    // Exact chunk-boundary tests (multiples of 4)
    // =================================================================

    #[test]
    fn test_add_exact_8() {
        let a = vec![1.0; 8];
        let b = vec![2.0; 8];
        let mut out = vec![0.0; 8];
        neon_fused_residual_add(&a, &b, &mut out);
        assert_approx(&out, &[3.0; 8], EPS);
    }

    #[test]
    fn test_scale_exact_12() {
        let a = vec![0.0; 12];
        let b = vec![4.0; 12];
        let mut out = vec![0.0; 12];
        neon_fused_residual_scale(&a, &b, 0.25, &mut out);
        assert_approx(&out, &[1.0; 12], EPS);
    }

    #[test]
    fn test_gated_exact_16() {
        let a = vec![1.0; 16];
        let b = vec![2.0; 16];
        let g = vec![0.5; 16];
        let mut out = vec![0.0; 16];
        neon_fused_gated_residual(&a, &b, &g, &mut out);
        assert_approx(&out, &[2.0; 16], EPS);
    }

    #[test]
    fn test_dropout_exact_8() {
        let a = vec![0.0; 8];
        let b = vec![10.0; 8];
        let m = vec![0.5; 8];
        let mut out = vec![0.0; 8];
        neon_fused_residual_dropout(&a, &b, &m, &mut out);
        assert_approx(&out, &[5.0; 8], EPS);
    }

    #[test]
    fn test_stochastic_exact_12_keep() {
        let a = vec![1.0; 12];
        let b = vec![2.0; 12];
        let mut out = vec![0.0; 12];
        neon_fused_stochastic_depth(&a, &b, true, &mut out);
        assert_approx(&out, &[3.0; 12], EPS);
    }

    #[test]
    fn test_stochastic_exact_12_drop() {
        let a = vec![5.0; 12];
        let b = vec![99.0; 12];
        let mut out = vec![0.0; 12];
        neon_fused_stochastic_depth(&a, &b, false, &mut out);
        assert_approx(&out, &[5.0; 12], EPS);
    }
}
