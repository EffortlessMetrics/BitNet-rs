//! NEON-optimized fused softmax kernels for Apple Silicon.
//!
//! Provides a comprehensive suite of softmax variants accelerated with ARM NEON
//! `float32x4` intrinsics for 4-wide parallel computation. Each kernel uses a
//! fast polynomial exp approximation, numerical stability via max subtraction,
//! and scalar fallback for tail elements whose count is not a multiple of 4.
//!
//! # Kernels
//!
//! - [`neon_softmax_1d`] — standard 1-D softmax
//! - [`neon_softmax_2d`] — row-wise softmax over a flattened 2-D matrix
//! - [`neon_online_softmax`] — single-pass online/streaming softmax
//! - [`neon_fused_softmax_scale`] — fused scale + softmax
//! - [`neon_masked_softmax`] — masked softmax for attention layers
//! - [`neon_softmax_with_temperature`] — temperature-scaled softmax
//! - [`neon_log_softmax`] — numerically stable log-softmax
//! - [`neon_softmax_backward`] — backward pass (Jacobian–vector product)
//!
//! # Safety
//!
//! Every function that touches NEON load/store intrinsics is marked `unsafe`.

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(
    clippy::missing_safety_doc,
    clippy::float_cmp,
    clippy::manual_div_ceil,
    clippy::unnecessary_cast,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::let_and_return,
    clippy::derivable_impls,
    clippy::excessive_precision,
    clippy::manual_is_multiple_of
)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Lane count for `float32x4_t` NEON vectors.
const LANES: usize = 4;

// ── Fast exp approximation ─────────────────────────────────────────────

/// Scalar fast exp approximation (degree-4 Cody–Waite polynomial).
/// Maximum relative error ≈ 2 × 10⁻⁴ for |x| ≤ 20.
#[inline(always)]
fn fast_exp_scalar(x: f32) -> f32 {
    let x = x.clamp(-88.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E).round();
    let r = x - n * std::f32::consts::LN_2;
    let poly = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6.0 + r * (1.0 / 24.0))));
    poly * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// NEON vectorised fast exp for four lanes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn fast_exp_neon(x: float32x4_t) -> float32x4_t {
    let min_val = vdupq_n_f32(-88.0);
    let max_val = vdupq_n_f32(88.0);
    let x = vmaxq_f32(vminq_f32(x, max_val), min_val);

    let log2e = vdupq_n_f32(std::f32::consts::LOG2_E);
    let ln2 = vdupq_n_f32(std::f32::consts::LN_2);
    let n = vrndnq_f32(vmulq_f32(x, log2e));
    let r = vsubq_f32(x, vmulq_f32(n, ln2));

    let c1 = vdupq_n_f32(1.0 / 24.0);
    let c2 = vdupq_n_f32(1.0 / 6.0);
    let c3 = vdupq_n_f32(0.5);
    let one = vdupq_n_f32(1.0);

    let p = vfmaq_f32(c2, r, c1);
    let p = vfmaq_f32(c3, r, p);
    let p = vfmaq_f32(one, r, p);
    let poly = vfmaq_f32(one, r, p);

    let bias = vdupq_n_s32(127);
    let ni = vcvtq_s32_f32(n);
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(ni, bias), 23));

    vmulq_f32(poly, pow2n)
}

// ── Internal helpers ───────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn find_max_neon(data: &[f32]) -> f32 {
    if data.is_empty() {
        return f32::NEG_INFINITY;
    }
    let len = data.len();
    let chunks = len / LANES;
    let remainder = len % LANES;

    let mut max_vec = vdupq_n_f32(f32::NEG_INFINITY);
    let ptr = data.as_ptr();
    for i in 0..chunks {
        let v = vld1q_f32(ptr.add(i * LANES));
        max_vec = vmaxq_f32(max_vec, v);
    }

    let mut max_val = vmaxvq_f32(max_vec);
    for i in 0..remainder {
        let val = data[chunks * LANES + i];
        if val > max_val {
            max_val = val;
        }
    }
    max_val
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn exp_sum_neon(data: &[f32], max_val: f32) -> (Vec<f32>, f32) {
    let len = data.len();
    let mut exps = vec![0.0f32; len];
    if len == 0 {
        return (exps, 0.0);
    }

    let chunks = len / LANES;
    let remainder = len % LANES;
    let max_vec = vdupq_n_f32(max_val);
    let mut sum_vec = vdupq_n_f32(0.0);
    let in_ptr = data.as_ptr();
    let out_ptr = exps.as_mut_ptr();

    for i in 0..chunks {
        let v = vld1q_f32(in_ptr.add(i * LANES));
        let shifted = vsubq_f32(v, max_vec);
        let e = fast_exp_neon(shifted);
        sum_vec = vaddq_f32(sum_vec, e);
        vst1q_f32(out_ptr.add(i * LANES), e);
    }

    let mut sum_val = vaddvq_f32(sum_vec);
    let tail_start = chunks * LANES;
    for i in 0..remainder {
        let e = fast_exp_scalar(data[tail_start + i] - max_val);
        exps[tail_start + i] = e;
        sum_val += e;
    }
    (exps, sum_val)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn normalize_neon(exps: &[f32], sum: f32) -> Vec<f32> {
    let len = exps.len();
    let mut out = vec![0.0f32; len];
    if len == 0 {
        return out;
    }
    let chunks = len / LANES;
    let remainder = len % LANES;
    let inv_sum = 1.0 / sum;
    let inv_sum_vec = vdupq_n_f32(inv_sum);
    let exp_ptr = exps.as_ptr();
    let out_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let e = vld1q_f32(exp_ptr.add(i * LANES));
        let r = vmulq_f32(e, inv_sum_vec);
        vst1q_f32(out_ptr.add(i * LANES), r);
    }
    let tail = chunks * LANES;
    for i in 0..remainder {
        out[tail + i] = exps[tail + i] * inv_sum;
    }
    out
}

// ── Scalar reference (for tests / non-aarch64 fallback) ────────────────

/// Plain scalar softmax used as the ground-truth reference in tests.
pub fn scalar_softmax(input: &[f32]) -> Vec<f32> {
    if input.is_empty() {
        return vec![];
    }
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| fast_exp_scalar(x - max_val)).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Plain scalar log-softmax reference.
pub fn scalar_log_softmax(input: &[f32]) -> Vec<f32> {
    if input.is_empty() {
        return vec![];
    }
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| fast_exp_scalar(x - max_val)).collect();
    let sum: f32 = exps.iter().sum();
    let log_sum = sum.ln();
    input.iter().map(|&x| (x - max_val) - log_sum).collect()
}

// ── Public kernels ─────────────────────────────────────────────────────

/// Standard 1-D softmax: `output[i] = exp(input[i] - max) / Σ exp(…)`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_softmax_1d(input: &[f32]) -> Vec<f32> {
    if input.is_empty() {
        return vec![];
    }
    let max_val = find_max_neon(input);
    let (exps, sum) = exp_sum_neon(input, max_val);
    normalize_neon(&exps, sum)
}

/// Row-wise softmax over a flattened `[rows × cols]` matrix.
///
/// Each row of length `cols` gets an independent softmax.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_softmax_2d(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(
        input.len(),
        rows * cols,
        "input length {} != rows({}) * cols({})",
        input.len(),
        rows,
        cols
    );
    if input.is_empty() {
        return vec![];
    }
    let mut output = vec![0.0f32; input.len()];
    for r in 0..rows {
        let start = r * cols;
        let row = &input[start..start + cols];
        let softmaxed = neon_softmax_1d(row);
        output[start..start + cols].copy_from_slice(&softmaxed);
    }
    output
}

/// Single-pass online/streaming softmax (numerically stable without a
/// separate max-finding pass).
///
/// Maintains a running `max` and `sum` and rescales on the fly.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_online_softmax(input: &[f32]) -> Vec<f32> {
    let len = input.len();
    if len == 0 {
        return vec![];
    }

    let chunks = len / LANES;
    let remainder = len % LANES;
    let ptr = input.as_ptr();

    // Phase 1: compute running (max, sum_of_exp) in one pass.
    let mut max_vec = vdupq_n_f32(f32::NEG_INFINITY);
    let mut sum_vec = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let v = vld1q_f32(ptr.add(i * LANES));
        let old_max = max_vec;
        max_vec = vmaxq_f32(max_vec, v);
        // Rescale running sum: sum *= exp(old_max - new_max)
        let correction = fast_exp_neon(vsubq_f32(old_max, max_vec));
        sum_vec = vfmaq_f32(
            vmulq_f32(sum_vec, correction),
            vdupq_n_f32(1.0),
            fast_exp_neon(vsubq_f32(v, max_vec)),
        );
    }

    // Reduce 4 lanes to scalar.
    let mut lane_max = [0.0f32; LANES];
    let mut lane_sum = [0.0f32; LANES];
    vst1q_f32(lane_max.as_mut_ptr(), max_vec);
    vst1q_f32(lane_sum.as_mut_ptr(), sum_vec);

    let mut global_max = lane_max[0];
    let mut global_sum = lane_sum[0];
    for i in 1..LANES {
        if lane_max[i] > global_max {
            global_sum *= fast_exp_scalar(global_max - lane_max[i]);
            global_sum += lane_sum[i];
            global_max = lane_max[i];
        } else {
            global_sum += lane_sum[i] * fast_exp_scalar(lane_max[i] - global_max);
        }
    }

    // Scalar tail.
    let tail_start = chunks * LANES;
    for i in 0..remainder {
        let val = input[tail_start + i];
        if val > global_max {
            global_sum *= fast_exp_scalar(global_max - val);
            global_sum += 1.0;
            global_max = val;
        } else {
            global_sum += fast_exp_scalar(val - global_max);
        }
    }

    // Phase 2: normalise.
    let mut output = vec![0.0f32; len];
    let max_vec2 = vdupq_n_f32(global_max);
    let inv_sum = 1.0 / global_sum;
    let inv_sum_vec = vdupq_n_f32(inv_sum);
    let out_ptr = output.as_mut_ptr();

    for i in 0..chunks {
        let v = vld1q_f32(ptr.add(i * LANES));
        let e = fast_exp_neon(vsubq_f32(v, max_vec2));
        let r = vmulq_f32(e, inv_sum_vec);
        vst1q_f32(out_ptr.add(i * LANES), r);
    }
    for i in 0..remainder {
        let e = fast_exp_scalar(input[tail_start + i] - global_max);
        output[tail_start + i] = e * inv_sum;
    }

    output
}

/// Fused scale + softmax: `softmax(input * scale)`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_fused_softmax_scale(input: &[f32], scale: f32) -> Vec<f32> {
    if input.is_empty() {
        return vec![];
    }
    let len = input.len();
    let chunks = len / LANES;
    let remainder = len % LANES;

    // Scale the input.
    let mut scaled = vec![0.0f32; len];
    let scale_vec = vdupq_n_f32(scale);
    let in_ptr = input.as_ptr();
    let sc_ptr = scaled.as_mut_ptr();

    for i in 0..chunks {
        let v = vld1q_f32(in_ptr.add(i * LANES));
        vst1q_f32(sc_ptr.add(i * LANES), vmulq_f32(v, scale_vec));
    }
    let tail = chunks * LANES;
    for i in 0..remainder {
        scaled[tail + i] = input[tail + i] * scale;
    }

    neon_softmax_1d(&scaled)
}

/// Masked softmax for attention: positions where `mask[i]` is `true` are
/// replaced with `mask_value` (typically `f32::NEG_INFINITY`) before softmax.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_masked_softmax(input: &[f32], mask: &[bool], mask_value: f32) -> Vec<f32> {
    assert_eq!(input.len(), mask.len(), "input and mask must have same length");
    if input.is_empty() {
        return vec![];
    }
    let masked: Vec<f32> =
        input.iter().zip(mask.iter()).map(|(&v, &m)| if m { mask_value } else { v }).collect();
    neon_softmax_1d(&masked)
}

/// Temperature-scaled softmax: `softmax(input / temperature)`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_softmax_with_temperature(input: &[f32], temperature: f32) -> Vec<f32> {
    assert!(temperature > 0.0, "temperature must be positive");
    neon_fused_softmax_scale(input, 1.0 / temperature)
}

/// Numerically stable log-softmax: `log(softmax(input))`.
///
/// Computed as `(x - max) - log(Σ exp(x - max))` to avoid log-of-small-number.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_log_softmax(input: &[f32]) -> Vec<f32> {
    if input.is_empty() {
        return vec![];
    }
    let len = input.len();
    let max_val = find_max_neon(input);
    let (_exps, sum) = exp_sum_neon(input, max_val);
    let log_sum = sum.ln();

    let chunks = len / LANES;
    let remainder = len % LANES;

    let mut output = vec![0.0f32; len];
    let max_vec = vdupq_n_f32(max_val);
    let log_sum_vec = vdupq_n_f32(log_sum);
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for i in 0..chunks {
        let v = vld1q_f32(in_ptr.add(i * LANES));
        let shifted = vsubq_f32(v, max_vec);
        let result = vsubq_f32(shifted, log_sum_vec);
        vst1q_f32(out_ptr.add(i * LANES), result);
    }

    let tail = chunks * LANES;
    for i in 0..remainder {
        output[tail + i] = (input[tail + i] - max_val) - log_sum;
    }

    output
}

/// Backward pass for softmax (Jacobian–vector product).
///
/// Given softmax output `y` and upstream gradient `dy`:
///   `dx[i] = y[i] * (dy[i] - Σ_j y[j] * dy[j])`
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_softmax_backward(output: &[f32], grad_output: &[f32]) -> Vec<f32> {
    assert_eq!(output.len(), grad_output.len(), "output and grad_output must have same length");
    let len = output.len();
    if len == 0 {
        return vec![];
    }

    let chunks = len / LANES;
    let remainder = len % LANES;

    // Compute dot = Σ y[j] * dy[j] using NEON.
    let mut dot_vec = vdupq_n_f32(0.0);
    let y_ptr = output.as_ptr();
    let dy_ptr = grad_output.as_ptr();

    for i in 0..chunks {
        let y = vld1q_f32(y_ptr.add(i * LANES));
        let dy = vld1q_f32(dy_ptr.add(i * LANES));
        dot_vec = vfmaq_f32(dot_vec, y, dy);
    }
    let mut dot = vaddvq_f32(dot_vec);
    let tail = chunks * LANES;
    for i in 0..remainder {
        dot += output[tail + i] * grad_output[tail + i];
    }

    // dx[i] = y[i] * (dy[i] - dot)
    let mut grad_input = vec![0.0f32; len];
    let dot_vec = vdupq_n_f32(dot);
    let out_ptr = grad_input.as_mut_ptr();

    for i in 0..chunks {
        let y = vld1q_f32(y_ptr.add(i * LANES));
        let dy = vld1q_f32(dy_ptr.add(i * LANES));
        let diff = vsubq_f32(dy, dot_vec);
        let dx = vmulq_f32(y, diff);
        vst1q_f32(out_ptr.add(i * LANES), dx);
    }
    for i in 0..remainder {
        grad_input[tail + i] = output[tail + i] * (grad_output[tail + i] - dot);
    }

    grad_input
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for comparing fast-exp-based NEON results to scalar reference.
    const TOL: f32 = 1e-3;

    // ── Scalar helpers for test oracles ─────────────────────────────────

    fn reference_softmax(input: &[f32]) -> Vec<f32> {
        scalar_softmax(input)
    }

    fn reference_log_softmax(input: &[f32]) -> Vec<f32> {
        scalar_log_softmax(input)
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "mismatch at index {}: {} vs {} (diff={})",
                i,
                x,
                y,
                (x - y).abs()
            );
        }
    }

    fn assert_valid_distribution(v: &[f32]) {
        for &x in v {
            assert!(x >= 0.0, "negative probability: {}", x);
            assert!(x <= 1.0 + 1e-5, "probability > 1: {}", x);
        }
        if !v.is_empty() {
            let sum: f32 = v.iter().sum();
            assert!((sum - 1.0).abs() < 1e-2, "probabilities sum to {} (expected ~1.0)", sum);
        }
    }

    // ==================================================================
    // neon_softmax_1d tests
    // ==================================================================

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_1d_empty() {
        let result = unsafe { neon_softmax_1d(&[]) };
        assert!(result.is_empty());
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_1d_single() {
        let result = unsafe { neon_softmax_1d(&[5.0]) };
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_1d_two_elements() {
        let result = unsafe { neon_softmax_1d(&[1.0, 2.0]) };
        assert_valid_distribution(&result);
        assert!(result[1] > result[0]);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_1d_three_elements() {
        let result = unsafe { neon_softmax_1d(&[1.0, 2.0, 3.0]) };
        assert_valid_distribution(&result);
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_1d_exact_neon_width() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let result = unsafe { neon_softmax_1d(&input) };
        let expected = reference_softmax(&input);
        assert_close(&result, &expected, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_1d_five_elements() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = unsafe { neon_softmax_1d(&input) };
        let expected = reference_softmax(&input);
        assert_close(&result, &expected, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_1d_eight_elements() {
        let input = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let result = unsafe { neon_softmax_1d(&input) };
        let expected = reference_softmax(&input);
        assert_close(&result, &expected, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_1d_nine_elements() {
        let input: Vec<f32> = (0..9).map(|i| i as f32 * 0.5).collect();
        let result = unsafe { neon_softmax_1d(&input) };
        let expected = reference_softmax(&input);
        assert_close(&result, &expected, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_1d_all_zeros() {
        let input = [0.0; 8];
        let result = unsafe { neon_softmax_1d(&input) };
        assert_valid_distribution(&result);
        for &v in &result {
            assert!((v - 0.125).abs() < TOL);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_1d_all_same() {
        let input = [3.14; 5];
        let result = unsafe { neon_softmax_1d(&input) };
        assert_valid_distribution(&result);
        for &v in &result {
            assert!((v - 0.2).abs() < TOL);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_1d_negative_values() {
        let input = [-3.0, -2.0, -1.0, -0.5];
        let result = unsafe { neon_softmax_1d(&input) };
        let expected = reference_softmax(&input);
        assert_close(&result, &expected, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_1d_mixed_pos_neg() {
        let input = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let result = unsafe { neon_softmax_1d(&input) };
        let expected = reference_softmax(&input);
        assert_close(&result, &expected, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_1d_large_values() {
        let input = [80.0, 81.0, 82.0, 83.0];
        let result = unsafe { neon_softmax_1d(&input) };
        assert_valid_distribution(&result);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_1d_very_large_values() {
        let input = [500.0, 501.0, 502.0, 503.0, 504.0];
        let result = unsafe { neon_softmax_1d(&input) };
        assert_valid_distribution(&result);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_1d_very_small_values() {
        let input = [-500.0, -501.0, -502.0, -503.0];
        let result = unsafe { neon_softmax_1d(&input) };
        assert_valid_distribution(&result);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_1d_large_spread() {
        let input = [-100.0, 0.0, 100.0];
        let result = unsafe { neon_softmax_1d(&input) };
        assert_valid_distribution(&result);
        assert!(result[2] > 0.99);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_1d_1024_elements() {
        let input: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let result = unsafe { neon_softmax_1d(&input) };
        let expected = reference_softmax(&input);
        assert_close(&result, &expected, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_1d_4096_elements() {
        let input: Vec<f32> = (0..4096).map(|i| (i as f32) * 0.001 - 2.0).collect();
        let result = unsafe { neon_softmax_1d(&input) };
        assert_valid_distribution(&result);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_1d_monotonic_output() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = unsafe { neon_softmax_1d(&input) };
        for i in 1..result.len() {
            assert!(result[i] >= result[i - 1]);
        }
    }

    // ==================================================================
    // neon_softmax_2d tests
    // ==================================================================

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_2d_single_row() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let result = unsafe { neon_softmax_2d(&input, 1, 4) };
        let expected = reference_softmax(&input);
        assert_close(&result, &expected, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_2d_single_col() {
        let input = [1.0, 2.0, 3.0];
        let result = unsafe { neon_softmax_2d(&input, 3, 1) };
        // Each row with single element → softmax = 1.0.
        for &v in &result {
            assert!((v - 1.0).abs() < TOL);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_2d_two_rows() {
        let input = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let result = unsafe { neon_softmax_2d(&input, 2, 4) };
        let row0 = reference_softmax(&input[0..4]);
        let row1 = reference_softmax(&input[4..8]);
        assert_close(&result[0..4], &row0, TOL);
        assert_close(&result[4..8], &row1, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_2d_square() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let result = unsafe { neon_softmax_2d(&input, 3, 3) };
        for r in 0..3 {
            let row = &result[r * 3..(r + 1) * 3];
            assert_valid_distribution(row);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_2d_empty() {
        let result = unsafe { neon_softmax_2d(&[], 0, 0) };
        assert!(result.is_empty());
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_2d_wide_rows() {
        let input: Vec<f32> = (0..40).map(|i| i as f32 * 0.1).collect();
        let result = unsafe { neon_softmax_2d(&input, 4, 10) };
        for r in 0..4 {
            let row = &result[r * 10..(r + 1) * 10];
            assert_valid_distribution(row);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_2d_many_rows() {
        let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.02 - 2.0).collect();
        let result = unsafe { neon_softmax_2d(&input, 64, 4) };
        for r in 0..64 {
            let row = &result[r * 4..(r + 1) * 4];
            assert_valid_distribution(row);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_2d_row_independence() {
        let input1 = [1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0];
        let result = unsafe { neon_softmax_2d(&input1, 2, 4) };
        let lone = reference_softmax(&[1.0, 2.0, 3.0, 4.0]);
        assert_close(&result[0..4], &lone, TOL);
    }

    // ==================================================================
    // neon_online_softmax tests
    // ==================================================================

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_online_softmax_empty() {
        let result = unsafe { neon_online_softmax(&[]) };
        assert!(result.is_empty());
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_online_softmax_single() {
        let result = unsafe { neon_online_softmax(&[42.0]) };
        assert!((result[0] - 1.0).abs() < TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_online_softmax_matches_1d() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let standard = unsafe { neon_softmax_1d(&input) };
        let online = unsafe { neon_online_softmax(&input) };
        assert_close(&online, &standard, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_online_softmax_exact_width() {
        let input = [0.5, 1.5, 2.5, 3.5];
        let standard = unsafe { neon_softmax_1d(&input) };
        let online = unsafe { neon_online_softmax(&input) };
        assert_close(&online, &standard, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_online_softmax_large() {
        let input: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let standard = unsafe { neon_softmax_1d(&input) };
        let online = unsafe { neon_online_softmax(&input) };
        assert_close(&online, &standard, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_online_softmax_stability_large_values() {
        let input = [500.0, 501.0, 502.0, 503.0, 504.0];
        let result = unsafe { neon_online_softmax(&input) };
        assert_valid_distribution(&result);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_online_softmax_all_same() {
        let input = [7.7; 8];
        let result = unsafe { neon_online_softmax(&input) };
        for &v in &result {
            assert!((v - 0.125).abs() < TOL);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_online_softmax_negatives() {
        let input = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0];
        let standard = unsafe { neon_softmax_1d(&input) };
        let online = unsafe { neon_online_softmax(&input) };
        assert_close(&online, &standard, TOL);
    }

    // ==================================================================
    // neon_fused_softmax_scale tests
    // ==================================================================

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_fused_scale_identity() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let result = unsafe { neon_fused_softmax_scale(&input, 1.0) };
        let expected = reference_softmax(&input);
        assert_close(&result, &expected, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_fused_scale_two() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let scaled: Vec<f32> = input.iter().map(|&x| x * 2.0).collect();
        let result = unsafe { neon_fused_softmax_scale(&input, 2.0) };
        let expected = reference_softmax(&scaled);
        assert_close(&result, &expected, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_fused_scale_small() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = unsafe { neon_fused_softmax_scale(&input, 0.5) };
        assert_valid_distribution(&result);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_fused_scale_empty() {
        let result = unsafe { neon_fused_softmax_scale(&[], 2.0) };
        assert!(result.is_empty());
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_fused_scale_large_scale_sharpens() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let sharp = unsafe { neon_fused_softmax_scale(&input, 10.0) };
        let normal = unsafe { neon_fused_softmax_scale(&input, 1.0) };
        // Larger scale → distribution more peaked at max.
        assert!(sharp[3] > normal[3]);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_fused_scale_small_scale_flattens() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let flat = unsafe { neon_fused_softmax_scale(&input, 0.01) };
        assert_valid_distribution(&flat);
        // Very small scale → nearly uniform.
        let max_diff = flat.iter().copied().fold(0.0f32, |a, b| a.max(b))
            - flat.iter().copied().fold(f32::MAX, f32::min);
        assert!(max_diff < 0.05);
    }

    // ==================================================================
    // neon_masked_softmax tests
    // ==================================================================

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_masked_none_masked() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mask = [false, false, false, false];
        let result = unsafe { neon_masked_softmax(&input, &mask, f32::NEG_INFINITY) };
        let expected = reference_softmax(&input);
        assert_close(&result, &expected, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_masked_all_but_one() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mask = [true, true, true, false];
        let result = unsafe { neon_masked_softmax(&input, &mask, f32::NEG_INFINITY) };
        assert!((result[3] - 1.0).abs() < TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_masked_alternating() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mask = [true, false, true, false, true, false];
        let result = unsafe { neon_masked_softmax(&input, &mask, f32::NEG_INFINITY) };
        // Masked positions should be near zero.
        assert!(result[0] < TOL);
        assert!(result[2] < TOL);
        assert!(result[4] < TOL);
        // Unmasked should form a valid distribution (approximately).
        let unmasked_sum: f32 = [result[1], result[3], result[5]].iter().sum();
        assert!((unmasked_sum - 1.0).abs() < 0.02);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_masked_with_finite_mask_value() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mask = [true, false, false, false];
        let result = unsafe { neon_masked_softmax(&input, &mask, -1000.0) };
        assert_valid_distribution(&result);
        assert!(result[0] < TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_masked_empty() {
        let result = unsafe { neon_masked_softmax(&[], &[], f32::NEG_INFINITY) };
        assert!(result.is_empty());
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_masked_single_unmasked() {
        let input = [5.0];
        let mask = [false];
        let result = unsafe { neon_masked_softmax(&input, &mask, f32::NEG_INFINITY) };
        assert!((result[0] - 1.0).abs() < TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_masked_causal_pattern() {
        // Simulating a 4-token causal mask for position 2 (can see 0,1,2 but not 3).
        let input = [1.0, 2.0, 3.0, 4.0];
        let mask = [false, false, false, true];
        let result = unsafe { neon_masked_softmax(&input, &mask, f32::NEG_INFINITY) };
        assert!(result[3] < TOL);
        let visible_sum: f32 = result[0..3].iter().sum();
        assert!((visible_sum - 1.0).abs() < 0.02);
    }

    // ==================================================================
    // neon_softmax_with_temperature tests
    // ==================================================================

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_temperature_1_is_identity() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let result = unsafe { neon_softmax_with_temperature(&input, 1.0) };
        let expected = reference_softmax(&input);
        assert_close(&result, &expected, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_temperature_low_sharpens() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let sharp = unsafe { neon_softmax_with_temperature(&input, 0.1) };
        let normal = reference_softmax(&input);
        assert!(sharp[3] > normal[3]);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_temperature_high_flattens() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let flat = unsafe { neon_softmax_with_temperature(&input, 10.0) };
        let normal = reference_softmax(&input);
        // Higher T → more uniform → max element has lower probability.
        assert!(flat[3] < normal[3]);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_temperature_very_low() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = unsafe { neon_softmax_with_temperature(&input, 0.01) };
        assert_valid_distribution(&result);
        assert!(result[4] > 0.99);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_temperature_very_high() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let result = unsafe { neon_softmax_with_temperature(&input, 100.0) };
        assert_valid_distribution(&result);
        let max_diff = result.iter().copied().fold(0.0f32, |a, b| a.max(b))
            - result.iter().copied().fold(f32::MAX, f32::min);
        assert!(max_diff < 0.05);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_temperature_empty() {
        let result = unsafe { neon_softmax_with_temperature(&[], 1.0) };
        assert!(result.is_empty());
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_temperature_single_element() {
        let result = unsafe { neon_softmax_with_temperature(&[42.0], 0.5) };
        assert!((result[0] - 1.0).abs() < TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_temperature_with_negatives() {
        let input = [-3.0, -1.0, 0.0, 2.0];
        let result = unsafe { neon_softmax_with_temperature(&input, 2.0) };
        assert_valid_distribution(&result);
    }

    // ==================================================================
    // neon_log_softmax tests
    // ==================================================================

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_log_softmax_empty() {
        let result = unsafe { neon_log_softmax(&[]) };
        assert!(result.is_empty());
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_log_softmax_single() {
        let result = unsafe { neon_log_softmax(&[5.0]) };
        assert!((result[0] - 0.0).abs() < TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_log_softmax_basic() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let result = unsafe { neon_log_softmax(&input) };
        let expected = reference_log_softmax(&input);
        assert_close(&result, &expected, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_log_softmax_exp_matches_softmax() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let log_sm = unsafe { neon_log_softmax(&input) };
        let sm = unsafe { neon_softmax_1d(&input) };
        for (i, (&ls, &s)) in log_sm.iter().zip(sm.iter()).enumerate() {
            assert!(
                (ls.exp() - s).abs() < TOL,
                "exp(log_softmax) != softmax at {}: {} vs {}",
                i,
                ls.exp(),
                s
            );
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_log_softmax_all_negative() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let result = unsafe { neon_log_softmax(&input) };
        for &v in &result {
            assert!(v <= 0.0, "log-softmax should be <= 0, got {}", v);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_log_softmax_large_values() {
        let input = [500.0, 501.0, 502.0, 503.0];
        let result = unsafe { neon_log_softmax(&input) };
        for &v in &result {
            assert!(v.is_finite(), "log-softmax not finite: {}", v);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_log_softmax_1024_elements() {
        let input: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let result = unsafe { neon_log_softmax(&input) };
        let expected = reference_log_softmax(&input);
        assert_close(&result, &expected, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_log_softmax_five_elements() {
        let input = [0.1, 0.2, 0.3, 0.4, 0.5];
        let result = unsafe { neon_log_softmax(&input) };
        let expected = reference_log_softmax(&input);
        assert_close(&result, &expected, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_log_softmax_all_same() {
        let input = [2.0; 4];
        let result = unsafe { neon_log_softmax(&input) };
        let expected = (-4.0f32).ln();
        // All equal inputs → all log probs equal → ln(1/4).
        for &v in &result {
            assert!((v - expected).abs() < TOL);
        }
    }

    // ==================================================================
    // neon_softmax_backward tests
    // ==================================================================

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_backward_empty() {
        let result = unsafe { neon_softmax_backward(&[], &[]) };
        assert!(result.is_empty());
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_backward_single() {
        // For single element softmax output = [1.0], any grad → dx = 0.
        let result = unsafe { neon_softmax_backward(&[1.0], &[5.0]) };
        assert!((result[0]).abs() < TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_backward_shape_preserved() {
        let output = unsafe { neon_softmax_1d(&[1.0, 2.0, 3.0, 4.0]) };
        let grad = vec![1.0; 4];
        let result = unsafe { neon_softmax_backward(&output, &grad) };
        assert_eq!(result.len(), 4);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_backward_uniform_grad_is_zero() {
        // If grad_output is constant, dx should be ~0 everywhere.
        let output = unsafe { neon_softmax_1d(&[1.0, 2.0, 3.0, 4.0]) };
        let grad = vec![3.0; 4];
        let result = unsafe { neon_softmax_backward(&output, &grad) };
        for &v in &result {
            assert!(v.abs() < TOL, "expected ~0, got {}", v);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_backward_sum_is_zero() {
        // The Jacobian of softmax is: J = diag(y) - y*y^T.
        // For any grad, Σ dx_i = Σ y_i*(dy_i - dot) = dot - dot = 0.
        let output = unsafe { neon_softmax_1d(&[0.5, 1.5, 2.5, 3.5, 4.5]) };
        let grad = vec![0.1, -0.2, 0.3, -0.4, 0.5];
        let result = unsafe { neon_softmax_backward(&output, &grad) };
        let sum: f32 = result.iter().sum();
        assert!(sum.abs() < TOL, "backward sum should be ~0, got {}", sum);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_backward_five_elements() {
        let output = unsafe { neon_softmax_1d(&[1.0, 2.0, 3.0, 4.0, 5.0]) };
        let grad = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let result = unsafe { neon_softmax_backward(&output, &grad) };
        assert_eq!(result.len(), 5);
        let sum: f32 = result.iter().sum();
        assert!(sum.abs() < TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_backward_large() {
        let input: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
        let output = unsafe { neon_softmax_1d(&input) };
        let grad: Vec<f32> = (0..128).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let result = unsafe { neon_softmax_backward(&output, &grad) };
        assert_eq!(result.len(), 128);
        let sum: f32 = result.iter().sum();
        assert!(sum.abs() < 0.01);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_backward_zero_grad() {
        let output = unsafe { neon_softmax_1d(&[1.0, 2.0, 3.0, 4.0]) };
        let grad = vec![0.0; 4];
        let result = unsafe { neon_softmax_backward(&output, &grad) };
        for &v in &result {
            assert!(v.abs() < TOL);
        }
    }

    // ==================================================================
    // Cross-kernel consistency tests
    // ==================================================================

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_1d_matches_2d_single_row() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let r1 = unsafe { neon_softmax_1d(&input) };
        let r2 = unsafe { neon_softmax_2d(&input, 1, 5) };
        assert_close(&r1, &r2, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_scale_1_matches_1d() {
        let input = [0.5, 1.5, 2.5, 3.5];
        let r1 = unsafe { neon_softmax_1d(&input) };
        let r2 = unsafe { neon_fused_softmax_scale(&input, 1.0) };
        assert_close(&r1, &r2, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_temp_1_matches_1d() {
        let input = [0.5, 1.5, 2.5, 3.5, 4.5];
        let r1 = unsafe { neon_softmax_1d(&input) };
        let r2 = unsafe { neon_softmax_with_temperature(&input, 1.0) };
        assert_close(&r1, &r2, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_online_matches_standard_large() {
        let input: Vec<f32> = (0..256).map(|i| (i as f32).sin()).collect();
        let standard = unsafe { neon_softmax_1d(&input) };
        let online = unsafe { neon_online_softmax(&input) };
        assert_close(&online, &standard, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_no_mask_matches_1d() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mask = vec![false; 6];
        let r1 = unsafe { neon_softmax_1d(&input) };
        let r2 = unsafe { neon_masked_softmax(&input, &mask, f32::NEG_INFINITY) };
        assert_close(&r1, &r2, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_output_sums_to_one_various_sizes() {
        for size in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65] {
            let input: Vec<f32> = (0..size).map(|i| (i as f32) * 0.3 - 1.0).collect();
            let result = unsafe { neon_softmax_1d(&input) };
            assert_valid_distribution(&result);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_log_softmax_logsumexp_identity() {
        // log_softmax(x)_i = x_i - logsumexp(x)
        let input = [2.0, 3.0, 4.0, 5.0];
        let ls = unsafe { neon_log_softmax(&input) };
        // logsumexp = max + ln(Σ exp(x - max))
        let max_val = 5.0f32;
        let lse = max_val + input.iter().map(|&x| fast_exp_scalar(x - max_val)).sum::<f32>().ln();
        for (i, &x) in input.iter().enumerate() {
            assert!((ls[i] - (x - lse)).abs() < TOL);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_fused_scale_with_temperature_equivalence() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let temp = 0.5f32;
        let r_temp = unsafe { neon_softmax_with_temperature(&input, temp) };
        let r_scale = unsafe { neon_fused_softmax_scale(&input, 1.0 / temp) };
        assert_close(&r_temp, &r_scale, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_backward_gradient_direction() {
        // One-hot grad at index 2. dx[2] should be positive (pushed toward).
        let output = unsafe { neon_softmax_1d(&[1.0, 2.0, 3.0, 4.0]) };
        let grad = vec![0.0, 0.0, 1.0, 0.0];
        let result = unsafe { neon_softmax_backward(&output, &grad) };
        assert!(result[2] > 0.0, "gradient at target should be positive");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_backward_non_target_direction() {
        let output = unsafe { neon_softmax_1d(&[1.0, 2.0, 3.0, 4.0]) };
        let grad = vec![0.0, 0.0, 1.0, 0.0];
        let result = unsafe { neon_softmax_backward(&output, &grad) };
        // Non-target elements should have negative gradient.
        assert!(result[0] < 0.0);
        assert!(result[1] < 0.0);
        assert!(result[3] < 0.0);
    }

    // ==================================================================
    // Additional edge-case and stress tests
    // ==================================================================

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_1d_size_15() {
        let input: Vec<f32> = (0..15).map(|i| i as f32).collect();
        let result = unsafe { neon_softmax_1d(&input) };
        assert_valid_distribution(&result);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_1d_size_16() {
        let input: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let result = unsafe { neon_softmax_1d(&input) };
        let expected = reference_softmax(&input);
        assert_close(&result, &expected, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_1d_size_17() {
        let input: Vec<f32> = (0..17).map(|i| i as f32 * 0.3).collect();
        let result = unsafe { neon_softmax_1d(&input) };
        let expected = reference_softmax(&input);
        assert_close(&result, &expected, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_online_softmax_size_7() {
        let input: Vec<f32> = (0..7).map(|i| i as f32).collect();
        let standard = unsafe { neon_softmax_1d(&input) };
        let online = unsafe { neon_online_softmax(&input) };
        assert_close(&online, &standard, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_masked_softmax_large() {
        let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let mask: Vec<bool> = (0..64).map(|i| i % 3 == 0).collect();
        let result = unsafe { neon_masked_softmax(&input, &mask, f32::NEG_INFINITY) };
        assert_eq!(result.len(), 64);
        for (i, &m) in mask.iter().enumerate() {
            if m {
                assert!(result[i] < TOL);
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_2d_non_aligned_cols() {
        let input: Vec<f32> = (0..15).map(|i| i as f32 * 0.2).collect();
        let result = unsafe { neon_softmax_2d(&input, 3, 5) };
        for r in 0..3 {
            let row = &result[r * 5..(r + 1) * 5];
            assert_valid_distribution(row);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_log_softmax_negative_inputs() {
        let input = [-5.0, -4.0, -3.0, -2.0, -1.0];
        let result = unsafe { neon_log_softmax(&input) };
        let expected = reference_log_softmax(&input);
        assert_close(&result, &expected, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_backward_exact_neon_width() {
        let output = unsafe { neon_softmax_1d(&[1.0, 2.0, 3.0, 4.0]) };
        let grad = vec![0.1, 0.2, 0.3, 0.4];
        let result = unsafe { neon_softmax_backward(&output, &grad) };
        let sum: f32 = result.iter().sum();
        assert!(sum.abs() < TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_fused_scale_negative_scale() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let result = unsafe { neon_fused_softmax_scale(&input, -1.0) };
        assert_valid_distribution(&result);
        // Negative scale reverses ordering.
        assert!(result[0] > result[3]);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_1d_descending_order() {
        let input = [5.0, 4.0, 3.0, 2.0, 1.0];
        let result = unsafe { neon_softmax_1d(&input) };
        for i in 1..result.len() {
            assert!(result[i] <= result[i - 1]);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_online_softmax_4096() {
        let input: Vec<f32> = (0..4096).map(|i| ((i as f32) * 0.01).sin()).collect();
        let standard = unsafe { neon_softmax_1d(&input) };
        let online = unsafe { neon_online_softmax(&input) };
        assert_close(&online, &standard, TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_backward_8_elements() {
        let output = unsafe { neon_softmax_1d(&[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]) };
        let grad: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let result = unsafe { neon_softmax_backward(&output, &grad) };
        let sum: f32 = result.iter().sum();
        assert!(sum.abs() < TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_2d_1x1() {
        let result = unsafe { neon_softmax_2d(&[42.0], 1, 1) };
        assert!((result[0] - 1.0).abs() < TOL);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_softmax_2d_4x4() {
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let result = unsafe { neon_softmax_2d(&input, 4, 4) };
        for r in 0..4 {
            let row = &result[r * 4..(r + 1) * 4];
            assert_valid_distribution(row);
        }
    }

    // ── scalar reference tests ──────────────────────────────────────────

    #[test]
    fn test_scalar_softmax_basic() {
        let result = scalar_softmax(&[1.0, 2.0, 3.0]);
        assert_valid_distribution(&result);
    }

    #[test]
    fn test_scalar_softmax_empty() {
        let result = scalar_softmax(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_scalar_log_softmax_basic() {
        let result = scalar_log_softmax(&[1.0, 2.0, 3.0]);
        for &v in &result {
            assert!(v <= 0.0);
        }
    }

    #[test]
    fn test_scalar_log_softmax_empty() {
        let result = scalar_log_softmax(&[]);
        assert!(result.is_empty());
    }
}
