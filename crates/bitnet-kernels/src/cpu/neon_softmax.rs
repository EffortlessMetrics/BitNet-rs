//! ARM NEON optimized softmax kernel for Apple Silicon
//!
//! Provides SIMD-accelerated softmax using `float32x4` NEON intrinsics for
//! 4-wide parallel computation. Includes a fast polynomial exp approximation,
//! numerical stability via max subtraction, and scalar fallback for tail
//! elements whose count is not a multiple of 4.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Lane count for `float32x4_t` NEON vectors.
const LANES: usize = 4;

// ── Fast exp approximation ──────────────────────────────────────────────

/// Scalar fast exp approximation (degree-4 polynomial in the range produced
/// by `x - max`).  Maximum relative error ≈ 2 × 10⁻⁴ for |x| ≤ 20, which
/// is more than adequate for softmax normalisation.
#[inline(always)]
fn fast_exp_scalar(x: f32) -> f32 {
    // Clamp to avoid overflow / underflow in the polynomial.
    let x = x.clamp(-88.0, 88.0);
    // Cody-Waite style: reduce x = n·ln2 + r, then exp(r) via polynomial.
    let n = (x * std::f32::consts::LOG2_E).round();
    let r = x - n * std::f32::consts::LN_2;
    // Degree-4 minimax polynomial for exp(r) on [-ln2/2, ln2/2].
    let poly = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6.0 + r * (1.0 / 24.0))));
    poly * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// NEON vectorised fast exp for four lanes.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn fast_exp_neon(x: float32x4_t) -> float32x4_t {
    // Clamp
    let min_val = vdupq_n_f32(-88.0);
    let max_val = vdupq_n_f32(88.0);
    let x = vmaxq_f32(vminq_f32(x, max_val), min_val);

    // n = round(x * log2(e))
    let log2e = vdupq_n_f32(std::f32::consts::LOG2_E);
    let ln2 = vdupq_n_f32(std::f32::consts::LN_2);
    let n = vrndnq_f32(vmulq_f32(x, log2e));

    // r = x - n * ln2
    let r = vsubq_f32(x, vmulq_f32(n, ln2));

    // Polynomial: 1 + r*(1 + r*(0.5 + r*(1/6 + r/24)))
    let c1 = vdupq_n_f32(1.0 / 24.0);
    let c2 = vdupq_n_f32(1.0 / 6.0);
    let c3 = vdupq_n_f32(0.5);
    let one = vdupq_n_f32(1.0);

    let p = vfmaq_f32(c2, r, c1); // 1/6 + r/24
    let p = vfmaq_f32(c3, r, p); // 0.5 + r*(…)
    let p = vfmaq_f32(one, r, p); // 1 + r*(…)
    let poly = vfmaq_f32(one, r, p); // 1 + r*(…)

    // 2^n via integer bit manipulation: reinterpret (n+127)<<23 as f32
    let bias = vdupq_n_s32(127);
    let ni = vcvtq_s32_f32(n);
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(ni, bias), 23));

    vmulq_f32(poly, pow2n)
}

// ── Internal NEON helpers ───────────────────────────────────────────────

/// Find the maximum value in `data` using NEON horizontal max.
///
/// # Safety
/// Requires `aarch64` target with NEON.
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
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        max_vec = vmaxq_f32(max_vec, v);
    }

    // Horizontal max of the four lanes.
    let mut max_val = vmaxvq_f32(max_vec);

    // Scalar tail.
    for i in 0..remainder {
        let val = data[chunks * LANES + i];
        if val > max_val {
            max_val = val;
        }
    }

    max_val
}

/// Compute `exp(data[i] - max_val)` for every element and return (exps, sum).
///
/// # Safety
/// Requires `aarch64` target with NEON.
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
        let v = unsafe { vld1q_f32(in_ptr.add(i * LANES)) };
        let shifted = vsubq_f32(v, max_vec);
        let e = unsafe { fast_exp_neon(shifted) };
        sum_vec = vaddq_f32(sum_vec, e);
        unsafe { vst1q_f32(out_ptr.add(i * LANES), e) };
    }

    let mut sum_val = vaddvq_f32(sum_vec);

    // Scalar tail.
    let tail_start = chunks * LANES;
    for i in 0..remainder {
        let e = fast_exp_scalar(data[tail_start + i] - max_val);
        exps[tail_start + i] = e;
        sum_val += e;
    }

    (exps, sum_val)
}

// ── Public API ──────────────────────────────────────────────────────────

/// NEON-accelerated softmax: `output[i] = exp(input[i] - max) / Σ exp(…)`.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn softmax_neon(input: &[f32], output: &mut [f32]) {
    assert!(
        output.len() >= input.len(),
        "output buffer too small: {} < {}",
        output.len(),
        input.len()
    );

    let len = input.len();
    if len == 0 {
        return;
    }

    let max_val = unsafe { find_max_neon(input) };
    let (exps, sum) = unsafe { exp_sum_neon(input, max_val) };

    // Divide every exp value by the sum.
    let chunks = len / LANES;
    let remainder = len % LANES;
    let inv_sum = 1.0 / sum;
    let inv_sum_vec = vdupq_n_f32(inv_sum);

    let exp_ptr = exps.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for i in 0..chunks {
        let e = unsafe { vld1q_f32(exp_ptr.add(i * LANES)) };
        let r = vmulq_f32(e, inv_sum_vec);
        unsafe { vst1q_f32(out_ptr.add(i * LANES), r) };
    }

    let tail_start = chunks * LANES;
    for i in 0..remainder {
        output[tail_start + i] = exps[tail_start + i] * inv_sum;
    }
}

/// In-place NEON-accelerated softmax.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn softmax_neon_inplace(data: &mut [f32]) {
    let len = data.len();
    if len == 0 {
        return;
    }

    let max_val = unsafe { find_max_neon(data) };
    let (_exps, sum) = unsafe { exp_sum_neon(data, max_val) };

    // Write normalised values back in-place.
    let chunks = len / LANES;
    let remainder = len % LANES;
    let inv_sum = 1.0 / sum;
    let inv_sum_vec = vdupq_n_f32(inv_sum);

    // Re-compute exp and normalise in a fused pass to avoid the temporary.
    let max_vec = vdupq_n_f32(max_val);
    let ptr = data.as_mut_ptr();

    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        let shifted = vsubq_f32(v, max_vec);
        let e = unsafe { fast_exp_neon(shifted) };
        let r = vmulq_f32(e, inv_sum_vec);
        unsafe { vst1q_f32(ptr.add(i * LANES), r) };
    }

    let tail_start = chunks * LANES;
    for i in 0..remainder {
        let e = fast_exp_scalar(data[tail_start + i] - max_val);
        data[tail_start + i] = e * inv_sum;
    }
}

// ── Scalar reference (used in tests & as fallback documentation) ────────

/// Plain scalar softmax for parity testing.
pub fn softmax_scalar(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len());
    let len = input.len();
    if len == 0 {
        return;
    }
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    for i in 0..len {
        output[i] = exps[i] / sum;
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    /// Helper: call softmax_neon through a safe wrapper.
    fn run_softmax(input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0f32; input.len()];
        // SAFETY: we are on aarch64 in test configuration.
        unsafe { softmax_neon(input, &mut output) };
        output
    }

    fn assert_close(a: f32, b: f32, tol: f32, ctx: &str) {
        assert!((a - b).abs() < tol, "{ctx}: expected {b}, got {a} (diff {})", (a - b).abs());
    }

    #[test]
    fn test_softmax_basic() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let out = run_softmax(&input);
        let sum: f32 = out.iter().sum();
        assert_close(sum, 1.0, 1e-3, "sum");
        // Output should be monotonically increasing.
        for w in out.windows(2) {
            assert!(w[0] < w[1], "expected monotonic increase");
        }
    }

    #[test]
    fn test_softmax_single_element() {
        let out = run_softmax(&[42.0]);
        assert_close(out[0], 1.0, 1e-5, "single element");
    }

    #[test]
    fn test_softmax_all_equal() {
        let input = [3.0; 8];
        let out = run_softmax(&input);
        let expected = 1.0 / 8.0;
        for (i, &v) in out.iter().enumerate() {
            assert_close(v, expected, 1e-3, &format!("uniform[{i}]"));
        }
    }

    #[test]
    fn test_softmax_large_values() {
        let input = [1000.0, 1001.0, 1002.0, 1003.0];
        let out = run_softmax(&input);
        let sum: f32 = out.iter().sum();
        assert_close(sum, 1.0, 1e-3, "large values sum");
        // Should not produce NaN/Inf thanks to max subtraction.
        for &v in &out {
            assert!(v.is_finite(), "expected finite, got {v}");
        }
    }

    #[test]
    fn test_softmax_negative_values() {
        let input = [-5.0, -3.0, -1.0, 0.0];
        let out = run_softmax(&input);
        let sum: f32 = out.iter().sum();
        assert_close(sum, 1.0, 1e-3, "negative values sum");
        for w in out.windows(2) {
            assert!(w[0] < w[1], "expected monotonic increase");
        }
    }

    #[test]
    fn test_softmax_inplace() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        unsafe { softmax_neon_inplace(&mut data) };
        let sum: f32 = data.iter().sum();
        assert_close(sum, 1.0, 1e-3, "inplace sum");
    }

    #[test]
    fn test_softmax_empty() {
        let mut output: Vec<f32> = vec![];
        unsafe { softmax_neon(&[], &mut output) };
        assert!(output.is_empty());
    }

    #[test]
    fn test_softmax_non_aligned() {
        for &len in &[5, 7, 13] {
            let input: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
            let out = run_softmax(&input);
            let sum: f32 = out.iter().sum();
            assert_close(sum, 1.0, 1e-3, &format!("non-aligned len={len}"));
            for &v in &out {
                assert!(v.is_finite(), "len={len}: non-finite {v}");
            }
        }
    }

    #[test]
    fn test_softmax_parity_with_scalar() {
        let input: Vec<f32> = (0..17).map(|i| (i as f32) * 0.3 - 2.5).collect();
        let neon_out = run_softmax(&input);
        let mut scalar_out = vec![0.0f32; input.len()];
        softmax_scalar(&input, &mut scalar_out);

        for (i, (&n, &s)) in neon_out.iter().zip(scalar_out.iter()).enumerate() {
            assert!(
                (n - s).abs() < 1e-3,
                "parity[{i}]: neon={n}, scalar={s}, diff={}",
                (n - s).abs()
            );
        }
    }
}
