#![allow(unsafe_op_in_unsafe_fn, unused_unsafe, dead_code, unused_variables, unused_assignments)]
//! ARM NEON-optimized residual connection and normalization operations
//! for Apple Silicon (AArch64).
//!
//! Provides vectorized residual add, RMS normalization, pre-norm residual,
//! and affine scale-and-shift using NEON intrinsics. Processes 4 × f32 lanes
//! at a time with scalar fallback for remainder elements.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// In-place residual connection: `output[i] += residual[i]`.
///
/// # Panics
///
/// Panics if `output` and `residual` have different lengths.
#[cfg(target_arch = "aarch64")]
pub fn neon_residual_add(output: &mut [f32], residual: &[f32]) {
    assert_eq!(output.len(), residual.len(), "length mismatch");
    let n = output.len();
    let chunks = n / 4;
    let o_ptr = output.as_mut_ptr();
    let r_ptr = residual.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        unsafe {
            let vo = vld1q_f32(o_ptr.add(offset));
            let vr = vld1q_f32(r_ptr.add(offset));
            let vs = vaddq_f32(vo, vr);
            vst1q_f32(o_ptr.add(offset), vs);
        }
    }

    for i in (chunks * 4)..n {
        output[i] += residual[i];
    }
}

/// RMS normalization (LLaMA-style): `out[i] = (input[i] / rms) * weight[i]`
/// where `rms = sqrt(mean(input²) + eps)`.
///
/// Uses `vrsqrteq_f32` for fast inverse square root estimation with one
/// Newton–Raphson refinement step.
///
/// # Panics
///
/// Panics if `input` and `weight` have different lengths.
#[cfg(target_arch = "aarch64")]
pub fn neon_rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(input.len(), weight.len(), "length mismatch");
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    // Compute sum of squares
    let chunks = n / 4;
    let i_ptr = input.as_ptr();
    let mut sum_sq = 0.0_f32;

    unsafe {
        let mut vacc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let offset = i * 4;
            let vi = vld1q_f32(i_ptr.add(offset));
            vacc = vfmaq_f32(vacc, vi, vi);
        }
        // Horizontal sum of vacc
        sum_sq = vaddvq_f32(vacc);
    }

    for &val in input.iter().take(n).skip(chunks * 4) {
        sum_sq += val * val;
    }

    let mean_sq = sum_sq / n as f32;

    // Fast inverse sqrt via NEON estimate + Newton–Raphson refinement
    let inv_rms = unsafe {
        let val = vdupq_n_f32(mean_sq + eps);
        let est = vrsqrteq_f32(val);
        // One Newton–Raphson step: est * (3 - val * est * est) / 2
        let refined = vmulq_f32(vrsqrtsq_f32(vmulq_f32(val, est), est), est);
        vgetq_lane_f32(refined, 0)
    };

    // Normalize and apply weight
    let w_ptr = weight.as_ptr();
    let mut out = vec![0.0_f32; n];
    let o_ptr = out.as_mut_ptr();

    unsafe {
        let v_inv_rms = vdupq_n_f32(inv_rms);
        for i in 0..chunks {
            let offset = i * 4;
            let vi = vld1q_f32(i_ptr.add(offset));
            let vw = vld1q_f32(w_ptr.add(offset));
            let vnorm = vmulq_f32(vi, v_inv_rms);
            let vout = vmulq_f32(vnorm, vw);
            vst1q_f32(o_ptr.add(offset), vout);
        }
    }

    for i in (chunks * 4)..n {
        out[i] = input[i] * inv_rms * weight[i];
    }

    out
}

/// Pre-norm residual: computes `combined = input + residual`, then
/// `normalized = rms_norm(combined, weight, eps)`.
///
/// Returns `(normalized, combined)` so callers can reuse the residual sum.
///
/// # Panics
///
/// Panics if `input`, `residual`, and `weight` have different lengths.
#[cfg(target_arch = "aarch64")]
pub fn neon_pre_norm_residual(
    input: &[f32],
    residual: &[f32],
    weight: &[f32],
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(input.len(), residual.len(), "input/residual length mismatch");
    assert_eq!(input.len(), weight.len(), "input/weight length mismatch");
    let n = input.len();

    // Compute combined = input + residual
    let mut combined = vec![0.0_f32; n];
    let chunks = n / 4;
    let i_ptr = input.as_ptr();
    let r_ptr = residual.as_ptr();
    let c_ptr = combined.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        unsafe {
            let vi = vld1q_f32(i_ptr.add(offset));
            let vr = vld1q_f32(r_ptr.add(offset));
            let vs = vaddq_f32(vi, vr);
            vst1q_f32(c_ptr.add(offset), vs);
        }
    }

    for i in (chunks * 4)..n {
        combined[i] = input[i] + residual[i];
    }

    let normalized = neon_rms_norm(&combined, weight, eps);
    (normalized, combined)
}

/// Affine transform in-place: `input[i] = input[i] * scale[i] + shift[i]`.
///
/// # Panics
///
/// Panics if `input`, `scale`, and `shift` have different lengths.
#[cfg(target_arch = "aarch64")]
pub fn neon_scale_and_shift(input: &mut [f32], scale: &[f32], shift: &[f32]) {
    assert_eq!(input.len(), scale.len(), "input/scale length mismatch");
    assert_eq!(input.len(), shift.len(), "input/shift length mismatch");
    let n = input.len();
    let chunks = n / 4;
    let i_ptr = input.as_mut_ptr();
    let s_ptr = scale.as_ptr();
    let h_ptr = shift.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        unsafe {
            let vi = vld1q_f32(i_ptr.add(offset));
            let vs = vld1q_f32(s_ptr.add(offset));
            let vh = vld1q_f32(h_ptr.add(offset));
            let vr = vfmaq_f32(vh, vi, vs);
            vst1q_f32(i_ptr.add(offset), vr);
        }
    }

    for i in (chunks * 4)..n {
        input[i] = input[i] * scale[i] + shift[i];
    }
}

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    #[test]
    fn test_residual_add() {
        let mut output = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let residual = vec![0.5, 1.0, 1.5, 2.0, 2.5];
        neon_residual_add(&mut output, &residual);
        let expected = vec![1.5, 3.0, 4.5, 6.0, 7.5];
        for (a, b) in output.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "got {a}, expected {b}");
        }
    }

    #[test]
    fn test_rms_norm_unit_weights() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = [1.0; 4];
        let eps = 1e-5;
        let out = neon_rms_norm(&input, &weight, eps);

        // RMS = sqrt(mean([1,4,9,16]) + eps) = sqrt(7.5 + eps)
        let rms = (7.5_f32 + eps).sqrt();
        for (i, &v) in out.iter().enumerate() {
            let expected = input[i] / rms;
            assert!((v - expected).abs() < 1e-3, "index {i}: got {v}, expected {expected}");
        }
    }

    #[test]
    fn test_pre_norm_residual() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let residual = vec![0.5, 0.5, 0.5, 0.5];
        let weight = [1.0; 4];
        let eps = 1e-5;

        let (normalized, combined) = neon_pre_norm_residual(&input, &residual, &weight, eps);

        // combined = input + residual
        let expected_combined = vec![1.5, 2.5, 3.5, 4.5];
        for (a, b) in combined.iter().zip(expected_combined.iter()) {
            assert!((a - b).abs() < 1e-6, "combined: got {a}, expected {b}");
        }

        // normalized = rms_norm(combined, weight, eps)
        let expected_norm = neon_rms_norm(&expected_combined, &weight, eps);
        for (a, b) in normalized.iter().zip(expected_norm.iter()) {
            assert!((a - b).abs() < 1e-6, "normalized: got {a}, expected {b}");
        }
    }

    #[test]
    fn test_scale_and_shift() {
        let mut input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let scale = vec![2.0, 3.0, 0.5, 1.0, 2.0];
        let shift = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        neon_scale_and_shift(&mut input, &scale, &shift);
        let expected = vec![2.1, 6.2, 1.8, 4.4, 10.5];
        for (a, b) in input.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "got {a}, expected {b}");
        }
    }
}
