//! NEON-optimized quantized softmax operations for Apple Silicon / ARM64.
//!
//! Provides numerically stable softmax variants designed for quantized
//! inference pipelines: standard, in-place, log-softmax, temperature-scaled,
//! masked, top-k, and backward pass. All hot paths use `float32x4` NEON
//! intrinsics for 4-wide SIMD parallelism with scalar tails.

use std::arch::aarch64::*;

/// Lane count for `float32x4_t` NEON vectors.
#[cfg(target_arch = "aarch64")]
const LANES: usize = 4;

// ── Fast exp approximation ──────────────────────────────────────────────

/// Scalar fast exp (degree-4 Cody–Waite polynomial).
/// Max relative error ≈ 2 × 10⁻⁴ for |x| ≤ 20.
#[inline(always)]
fn fast_exp_scalar(x: f32) -> f32 {
    let x = x.clamp(-88.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E).round();
    let r = x - n * std::f32::consts::LN_2;
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
    let min_val = unsafe { vdupq_n_f32(-88.0) };
    let max_val = unsafe { vdupq_n_f32(88.0) };
    let x = unsafe { vmaxq_f32(vminq_f32(x, max_val), min_val) };

    let log2e = unsafe { vdupq_n_f32(std::f32::consts::LOG2_E) };
    let ln2 = unsafe { vdupq_n_f32(std::f32::consts::LN_2) };
    let n = unsafe { vrndnq_f32(vmulq_f32(x, log2e)) };
    let r = unsafe { vsubq_f32(x, vmulq_f32(n, ln2)) };

    let c1 = unsafe { vdupq_n_f32(1.0 / 24.0) };
    let c2 = unsafe { vdupq_n_f32(1.0 / 6.0) };
    let c3 = unsafe { vdupq_n_f32(0.5) };
    let one = unsafe { vdupq_n_f32(1.0) };

    let p = unsafe { vfmaq_f32(c2, r, c1) };
    let p = unsafe { vfmaq_f32(c3, r, p) };
    let p = unsafe { vfmaq_f32(one, r, p) };
    let poly = unsafe { vfmaq_f32(one, r, p) };

    let bias = unsafe { vdupq_n_s32(127) };
    let ni = unsafe { vcvtq_s32_f32(n) };
    let pow2n = unsafe { vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(ni, bias), 23)) };

    unsafe { vmulq_f32(poly, pow2n) }
}

// ── Internal NEON helpers ───────────────────────────────────────────────

/// Find the maximum value in `data` using NEON horizontal max.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn find_max_neon(data: &[f32]) -> f32 {
    if data.is_empty() {
        return f32::NEG_INFINITY;
    }

    let len = data.len();
    let chunks = len / LANES;
    let remainder = len % LANES;

    let mut max_vec = unsafe { vdupq_n_f32(f32::NEG_INFINITY) };
    let ptr = data.as_ptr();
    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        max_vec = unsafe { vmaxq_f32(max_vec, v) };
    }

    let mut max_val = unsafe { vmaxvq_f32(max_vec) };
    for i in 0..remainder {
        let val = data[chunks * LANES + i];
        if val > max_val {
            max_val = val;
        }
    }
    max_val
}

/// Compute `exp(data[i] - max_val)` for every element; returns (exps, sum).
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

    let max_vec = unsafe { vdupq_n_f32(max_val) };
    let mut sum_vec = unsafe { vdupq_n_f32(0.0) };

    let in_ptr = data.as_ptr();
    let out_ptr = exps.as_mut_ptr();

    for i in 0..chunks {
        let v = unsafe { vld1q_f32(in_ptr.add(i * LANES)) };
        let shifted = unsafe { vsubq_f32(v, max_vec) };
        let e = unsafe { fast_exp_neon(shifted) };
        sum_vec = unsafe { vaddq_f32(sum_vec, e) };
        unsafe { vst1q_f32(out_ptr.add(i * LANES), e) };
    }

    let mut sum_val = unsafe { vaddvq_f32(sum_vec) };
    let tail_start = chunks * LANES;
    for i in 0..remainder {
        let e = fast_exp_scalar(data[tail_start + i] - max_val);
        exps[tail_start + i] = e;
        sum_val += e;
    }

    (exps, sum_val)
}

/// Divide every element of `exps` by `sum`, writing into `output`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn normalize_neon(exps: &[f32], sum: f32, output: &mut [f32]) {
    let len = exps.len();
    let chunks = len / LANES;
    let remainder = len % LANES;
    let inv_sum = 1.0 / sum;
    let inv_sum_vec = unsafe { vdupq_n_f32(inv_sum) };

    let exp_ptr = exps.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for i in 0..chunks {
        let e = unsafe { vld1q_f32(exp_ptr.add(i * LANES)) };
        let r = unsafe { vmulq_f32(e, inv_sum_vec) };
        unsafe { vst1q_f32(out_ptr.add(i * LANES), r) };
    }

    let tail_start = chunks * LANES;
    for i in 0..remainder {
        output[tail_start + i] = exps[tail_start + i] * inv_sum;
    }
}

// ── Scalar reference helpers ────────────────────────────────────────────

/// Plain scalar softmax for parity testing.
fn scalar_softmax(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    if len == 0 {
        return;
    }
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for i in 0..len {
        let e = fast_exp_scalar(input[i] - max_val);
        output[i] = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for v in output[..len].iter_mut() {
        *v *= inv;
    }
}

// ── Public API ──────────────────────────────────────────────────────────

/// NEON-accelerated numerically stable softmax.
///
/// `output[i] = exp(input[i] - max) / Σ exp(input[j] - max)`
///
/// # Panics
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
pub fn neon_softmax_f32(input: &[f32], output: &mut [f32]) {
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
    // SAFETY: we are on aarch64 where NEON is always available.
    unsafe {
        let max_val = find_max_neon(input);
        let (exps, sum) = exp_sum_neon(input, max_val);
        normalize_neon(&exps, sum, output);
    }
}

/// In-place NEON-accelerated softmax.
#[cfg(target_arch = "aarch64")]
pub fn neon_softmax_inplace_f32(data: &mut [f32]) {
    let len = data.len();
    if len == 0 {
        return;
    }
    // SAFETY: aarch64 always has NEON.
    unsafe {
        let max_val = find_max_neon(data);
        let (exps, sum) = exp_sum_neon(data, max_val);
        normalize_neon(&exps, sum, data);
    }
}

/// NEON-accelerated log-softmax for cross-entropy loss.
///
/// `output[i] = (input[i] - max) - ln(Σ exp(input[j] - max))`
///
/// # Panics
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
pub fn neon_log_softmax_f32(input: &[f32], output: &mut [f32]) {
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
    // SAFETY: aarch64 always has NEON.
    unsafe {
        let max_val = find_max_neon(input);
        let (_exps, sum) = exp_sum_neon(input, max_val);
        let log_sum = sum.ln();

        let chunks = len / LANES;
        let remainder = len % LANES;

        let max_vec = vdupq_n_f32(max_val);
        let log_sum_vec = vdupq_n_f32(log_sum);
        let in_ptr = input.as_ptr();
        let out_ptr = output.as_mut_ptr();

        for i in 0..chunks {
            let v = vld1q_f32(in_ptr.add(i * LANES));
            // (x - max) - log_sum
            let shifted = vsubq_f32(v, max_vec);
            let r = vsubq_f32(shifted, log_sum_vec);
            vst1q_f32(out_ptr.add(i * LANES), r);
        }

        let tail_start = chunks * LANES;
        for i in 0..remainder {
            output[tail_start + i] = (input[tail_start + i] - max_val) - log_sum;
        }
    }
}

/// NEON-accelerated temperature-scaled softmax.
///
/// `output[i] = softmax(input / temperature)`
///
/// When `temperature` is very close to zero (< 1e-7), falls back to
/// argmax-like behavior (1.0 at the maximum, 0.0 elsewhere).
///
/// # Panics
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
pub fn neon_softmax_with_temperature_f32(input: &[f32], output: &mut [f32], temperature: f32) {
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

    // Near-zero temperature → argmax behavior.
    if temperature.abs() < 1e-7 {
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0;
        for (i, &v) in input.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        for v in output[..len].iter_mut() {
            *v = 0.0;
        }
        output[max_idx] = 1.0;
        return;
    }

    // Scale inputs by 1/temperature and delegate.
    let inv_temp = 1.0 / temperature;
    let mut scaled = vec![0.0f32; len];
    // SAFETY: aarch64 always has NEON.
    unsafe {
        let chunks = len / LANES;
        let remainder = len % LANES;
        let inv_t_vec = vdupq_n_f32(inv_temp);
        let in_ptr = input.as_ptr();
        let sc_ptr = scaled.as_mut_ptr();

        for i in 0..chunks {
            let v = vld1q_f32(in_ptr.add(i * LANES));
            let r = vmulq_f32(v, inv_t_vec);
            vst1q_f32(sc_ptr.add(i * LANES), r);
        }

        let tail_start = chunks * LANES;
        for i in 0..remainder {
            scaled[tail_start + i] = input[tail_start + i] * inv_temp;
        }
    }

    neon_softmax_f32(&scaled, output);
}

/// NEON-accelerated masked softmax for attention.
///
/// Positions where `mask[i] == true` are set to `-inf` before softmax so
/// they receive probability ≈ 0. If all positions are masked, output is
/// filled with zeros.
///
/// # Panics
/// Panics if `mask.len() < input.len()` or `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
pub fn neon_masked_softmax_f32(input: &[f32], mask: &[bool], output: &mut [f32]) {
    let len = input.len();
    assert!(mask.len() >= len, "mask buffer too small: {} < {}", mask.len(), len);
    assert!(output.len() >= len, "output buffer too small: {} < {}", output.len(), len);
    if len == 0 {
        return;
    }

    // If all masked, output zeros.
    if mask[..len].iter().all(|&m| m) {
        for v in output[..len].iter_mut() {
            *v = 0.0;
        }
        return;
    }

    // Apply mask: masked positions → -inf.
    let mut masked_input = vec![0.0f32; len];
    for i in 0..len {
        masked_input[i] = if mask[i] { f32::NEG_INFINITY } else { input[i] };
    }

    neon_softmax_f32(&masked_input, output);
}

/// NEON-accelerated top-k softmax.
///
/// Computes softmax only over the top `k` values; remaining positions are
/// set to 0. When `k >= input.len()`, this is equivalent to standard softmax.
///
/// # Panics
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
pub fn neon_top_k_softmax_f32(input: &[f32], output: &mut [f32], k: usize) {
    let len = input.len();
    assert!(output.len() >= len, "output buffer too small: {} < {}", output.len(), len);
    if len == 0 {
        return;
    }

    if k == 0 {
        for v in output[..len].iter_mut() {
            *v = 0.0;
        }
        return;
    }

    if k >= len {
        neon_softmax_f32(input, output);
        return;
    }

    // Find the k-th largest value as the threshold.
    let mut indices: Vec<usize> = (0..len).collect();
    indices.sort_unstable_by(|&a, &b| {
        input[b].partial_cmp(&input[a]).unwrap_or(std::cmp::Ordering::Equal)
    });

    let threshold = input[indices[k - 1]];

    // Collect top-k entries; apply softmax over them only.
    let mut top_k_vals = Vec::with_capacity(k);
    let mut top_k_indices = Vec::with_capacity(k);
    for &idx in &indices[..k] {
        top_k_vals.push(input[idx]);
        top_k_indices.push(idx);
    }

    // Handle ties: if the threshold ties with values outside top-k,
    // we still only take exactly k entries (first k from sorted order).
    let _ = threshold; // used implicitly via sorted indices

    let mut top_k_output = vec![0.0f32; k];
    neon_softmax_f32(&top_k_vals, &mut top_k_output);

    // Zero the output, then scatter top-k probabilities.
    for v in output[..len].iter_mut() {
        *v = 0.0;
    }
    for (j, &idx) in top_k_indices.iter().enumerate() {
        output[idx] = top_k_output[j];
    }
}

/// NEON-accelerated softmax backward pass for training.
///
/// Given forward output `y = softmax(x)` and upstream gradient `dL/dy`,
/// computes `dL/dx`:
///
///   `grad_input[i] = y[i] * (grad_output[i] - Σ_j(y[j] * grad_output[j]))`
///
/// # Panics
/// Panics if buffer lengths are inconsistent.
#[cfg(target_arch = "aarch64")]
pub fn neon_softmax_backward_f32(output: &[f32], grad_output: &[f32], grad_input: &mut [f32]) {
    let len = output.len();
    assert!(grad_output.len() >= len, "grad_output too small: {} < {}", grad_output.len(), len);
    assert!(grad_input.len() >= len, "grad_input too small: {} < {}", grad_input.len(), len);
    if len == 0 {
        return;
    }

    // dot = Σ y[i] * grad_output[i]
    let dot: f32;
    // SAFETY: aarch64 always has NEON.
    unsafe {
        let chunks = len / LANES;
        let remainder = len % LANES;
        let mut dot_vec = vdupq_n_f32(0.0);
        let y_ptr = output.as_ptr();
        let g_ptr = grad_output.as_ptr();

        for i in 0..chunks {
            let y = vld1q_f32(y_ptr.add(i * LANES));
            let g = vld1q_f32(g_ptr.add(i * LANES));
            dot_vec = vfmaq_f32(dot_vec, y, g);
        }

        let mut d = vaddvq_f32(dot_vec);
        let tail = chunks * LANES;
        for i in 0..remainder {
            d += output[tail + i] * grad_output[tail + i];
        }
        dot = d;
    }

    // grad_input[i] = y[i] * (grad_output[i] - dot)
    unsafe {
        let chunks = len / LANES;
        let remainder = len % LANES;
        let dot_vec = vdupq_n_f32(dot);
        let y_ptr = output.as_ptr();
        let g_ptr = grad_output.as_ptr();
        let gi_ptr = grad_input.as_mut_ptr();

        for i in 0..chunks {
            let y = vld1q_f32(y_ptr.add(i * LANES));
            let g = vld1q_f32(g_ptr.add(i * LANES));
            let diff = vsubq_f32(g, dot_vec);
            let r = vmulq_f32(y, diff);
            vst1q_f32(gi_ptr.add(i * LANES), r);
        }

        let tail = chunks * LANES;
        for i in 0..remainder {
            grad_input[tail + i] = output[tail + i] * (grad_output[tail + i] - dot);
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    /// Scalar reference softmax using stdlib exp for correctness checking.
    fn reference_softmax(input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return vec![];
        }
        let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    fn reference_log_softmax(input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return vec![];
        }
        let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp: f32 = input.iter().map(|&x| (x - max_val).exp()).sum::<f32>().ln();
        input.iter().map(|&x| (x - max_val) - log_sum_exp).collect()
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, ctx: &str) {
        assert_eq!(a.len(), b.len(), "{ctx}: length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < tol, "{ctx}[{i}]: {x} vs {y} (diff {})", (x - y).abs());
        }
    }

    // ── neon_softmax_f32 ────────────────────────────────────────────

    #[test]
    fn softmax_empty() {
        let input: &[f32] = &[];
        let mut output: Vec<f32> = vec![];
        neon_softmax_f32(input, &mut output);
        assert!(output.is_empty());
    }

    #[test]
    fn softmax_single() {
        let input = [42.0f32];
        let mut output = vec![0.0f32; 1];
        neon_softmax_f32(&input, &mut output);
        assert!((output[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_two_elements() {
        let input = [1.0f32, 2.0];
        let mut output = vec![0.0f32; 2];
        neon_softmax_f32(&input, &mut output);
        let expected = reference_softmax(&input);
        assert_close(&output, &expected, 1e-4, "softmax_two");
    }

    #[test]
    fn softmax_four_elements_exact_lane() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];
        neon_softmax_f32(&input, &mut output);
        let expected = reference_softmax(&input);
        assert_close(&output, &expected, 1e-4, "softmax_4");
    }

    #[test]
    fn softmax_eight_elements() {
        let input = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let mut output = vec![0.0f32; 8];
        neon_softmax_f32(&input, &mut output);
        let expected = reference_softmax(&input);
        assert_close(&output, &expected, 1e-4, "softmax_8");
    }

    #[test]
    fn softmax_non_lane_aligned() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0f32; 5];
        neon_softmax_f32(&input, &mut output);
        let expected = reference_softmax(&input);
        assert_close(&output, &expected, 1e-4, "softmax_5");
    }

    #[test]
    fn softmax_sixteen_elements() {
        let input: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let mut output = vec![0.0f32; 16];
        neon_softmax_f32(&input, &mut output);
        let expected = reference_softmax(&input);
        assert_close(&output, &expected, 1e-4, "softmax_16");
    }

    #[test]
    fn softmax_100_elements() {
        let input: Vec<f32> = (0..100).map(|i| (i as f32) * 0.01).collect();
        let mut output = vec![0.0f32; 100];
        neon_softmax_f32(&input, &mut output);
        let expected = reference_softmax(&input);
        assert_close(&output, &expected, 1e-3, "softmax_100");
    }

    #[test]
    fn softmax_1000_elements() {
        let input: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.001).collect();
        let mut output = vec![0.0f32; 1000];
        neon_softmax_f32(&input, &mut output);
        let expected = reference_softmax(&input);
        assert_close(&output, &expected, 1e-3, "softmax_1000");
    }

    #[test]
    fn softmax_sums_to_one() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut output = vec![0.0f32; 7];
        neon_softmax_f32(&input, &mut output);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum = {sum}");
    }

    #[test]
    fn softmax_all_same_values() {
        let input = [5.0f32; 8];
        let mut output = vec![0.0f32; 8];
        neon_softmax_f32(&input, &mut output);
        for &v in &output {
            assert!((v - 0.125).abs() < 1e-5, "expected uniform 1/8, got {v}");
        }
    }

    #[test]
    fn softmax_all_zeros() {
        let input = [0.0f32; 4];
        let mut output = vec![0.0f32; 4];
        neon_softmax_f32(&input, &mut output);
        for &v in &output {
            assert!((v - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    fn softmax_negative_values() {
        let input = [-1.0, -2.0, -3.0, -4.0];
        let mut output = vec![0.0f32; 4];
        neon_softmax_f32(&input, &mut output);
        let expected = reference_softmax(&input);
        assert_close(&output, &expected, 1e-4, "softmax_neg");
    }

    #[test]
    fn softmax_mixed_signs() {
        let input = [-5.0, 0.0, 5.0, 10.0];
        let mut output = vec![0.0f32; 4];
        neon_softmax_f32(&input, &mut output);
        let expected = reference_softmax(&input);
        assert_close(&output, &expected, 1e-4, "softmax_mixed");
    }

    // ── Numerical stability ─────────────────────────────────────────

    #[test]
    fn softmax_large_values_stability() {
        let input = [1000.0, 1001.0, 1002.0, 1003.0];
        let mut output = vec![0.0f32; 4];
        neon_softmax_f32(&input, &mut output);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum = {sum}");
        // Largest value should get highest probability.
        assert!(output[3] > output[2]);
        assert!(output[2] > output[1]);
    }

    #[test]
    fn softmax_very_large_values() {
        let input = [1e30, 1e30 + 1.0, 1e30 - 1.0, 1e30];
        let mut output = vec![0.0f32; 4];
        neon_softmax_f32(&input, &mut output);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "sum = {sum}");
        assert!(output.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn softmax_very_small_values() {
        let input = [-1000.0, -999.0, -998.0, -997.0];
        let mut output = vec![0.0f32; 4];
        neon_softmax_f32(&input, &mut output);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "sum = {sum}");
        assert!(output[3] > output[0]);
    }

    #[test]
    fn softmax_extreme_range() {
        let input = [-100.0, 0.0, 100.0, -100.0];
        let mut output = vec![0.0f32; 4];
        neon_softmax_f32(&input, &mut output);
        // The 100.0 element should dominate.
        assert!(output[2] > 0.99, "max element prob = {}", output[2]);
        assert!(output.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn softmax_all_negative_inf() {
        // All -inf: degenerate case. With max-subtract, all become 0/0.
        // We check it doesn't panic or produce NaN in a harmful way;
        // output may be NaN by IEEE rules (0/0) which is mathematically correct.
        let input = [f32::NEG_INFINITY; 4];
        let mut output = vec![0.0f32; 4];
        neon_softmax_f32(&input, &mut output);
        // No panic is the primary assertion.
    }

    #[test]
    fn softmax_preserves_ordering() {
        let input = [1.0, 5.0, 3.0, 2.0, 4.0, 6.0, 0.0, 7.0];
        let mut output = vec![0.0f32; 8];
        neon_softmax_f32(&input, &mut output);
        // Softmax preserves strict ordering.
        assert!(output[7] > output[5]); // 7 > 6
        assert!(output[5] > output[1]); // 6 > 5
        assert!(output[1] > output[4]); // 5 > 4
    }

    #[test]
    fn softmax_all_positive() {
        let input: Vec<f32> = (1..=10).map(|x| x as f32).collect();
        let mut output = vec![0.0f32; 10];
        neon_softmax_f32(&input, &mut output);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
        assert!(output.iter().all(|&v| v >= 0.0));
    }

    // ── neon_softmax_inplace_f32 ────────────────────────────────────

    #[test]
    fn inplace_empty() {
        let mut data: Vec<f32> = vec![];
        neon_softmax_inplace_f32(&mut data);
        assert!(data.is_empty());
    }

    #[test]
    fn inplace_single() {
        let mut data = vec![99.0f32];
        neon_softmax_inplace_f32(&mut data);
        assert!((data[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn inplace_matches_out_of_place() {
        let input = [2.0, 4.0, 6.0, 8.0, 1.0];
        let mut output = vec![0.0f32; 5];
        neon_softmax_f32(&input, &mut output);

        let mut inplace = input.to_vec();
        neon_softmax_inplace_f32(&mut inplace);
        assert_close(&inplace, &output, 1e-6, "inplace_vs_outofplace");
    }

    #[test]
    fn inplace_four_elements() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        neon_softmax_inplace_f32(&mut data);
        let expected = reference_softmax(&[1.0, 2.0, 3.0, 4.0]);
        assert_close(&data, &expected, 1e-4, "inplace_4");
    }

    #[test]
    fn inplace_eight_elements() {
        let orig = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];
        let mut data = orig.to_vec();
        neon_softmax_inplace_f32(&mut data);
        let expected = reference_softmax(&orig);
        assert_close(&data, &expected, 1e-4, "inplace_8");
    }

    #[test]
    fn inplace_sums_to_one() {
        let mut data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        neon_softmax_inplace_f32(&mut data);
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn inplace_large_values() {
        let mut data = vec![500.0, 501.0, 502.0, 503.0];
        neon_softmax_inplace_f32(&mut data);
        assert!(data.iter().all(|&v| v.is_finite()));
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    // ── neon_log_softmax_f32 ────────────────────────────────────────

    #[test]
    fn log_softmax_empty() {
        let input: &[f32] = &[];
        let mut output: Vec<f32> = vec![];
        neon_log_softmax_f32(input, &mut output);
        assert!(output.is_empty());
    }

    #[test]
    fn log_softmax_single() {
        let input = [0.0f32];
        let mut output = vec![0.0f32; 1];
        neon_log_softmax_f32(&input, &mut output);
        assert!(output[0].abs() < 1e-5, "log(1.0) should be ~0");
    }

    #[test]
    fn log_softmax_four() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];
        neon_log_softmax_f32(&input, &mut output);
        let expected = reference_log_softmax(&input);
        assert_close(&output, &expected, 1e-3, "log_softmax_4");
    }

    #[test]
    fn log_softmax_eight() {
        let input = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let mut output = vec![0.0f32; 8];
        neon_log_softmax_f32(&input, &mut output);
        let expected = reference_log_softmax(&input);
        assert_close(&output, &expected, 1e-3, "log_softmax_8");
    }

    #[test]
    fn log_softmax_non_aligned() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut output = vec![0.0f32; 7];
        neon_log_softmax_f32(&input, &mut output);
        let expected = reference_log_softmax(&input);
        assert_close(&output, &expected, 1e-3, "log_softmax_7");
    }

    #[test]
    fn log_softmax_all_values_nonpositive() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];
        neon_log_softmax_f32(&input, &mut output);
        // log-softmax values must be ≤ 0.
        for &v in &output {
            assert!(v <= 1e-6, "log-softmax should be ≤ 0, got {v}");
        }
    }

    #[test]
    fn log_softmax_exp_matches_softmax() {
        let input = [2.0, 4.0, 6.0, 8.0];
        let mut log_out = vec![0.0f32; 4];
        let mut sm_out = vec![0.0f32; 4];
        neon_log_softmax_f32(&input, &mut log_out);
        neon_softmax_f32(&input, &mut sm_out);
        // exp(log_softmax) ≈ softmax
        for (i, (&ls, &s)) in log_out.iter().zip(sm_out.iter()).enumerate() {
            let diff = (ls.exp() - s).abs();
            assert!(diff < 1e-3, "[{i}] exp(log_sm)={}, sm={s}", ls.exp());
        }
    }

    #[test]
    fn log_softmax_large_values() {
        let input = [1000.0, 1001.0, 1002.0, 1003.0];
        let mut output = vec![0.0f32; 4];
        neon_log_softmax_f32(&input, &mut output);
        assert!(output.iter().all(|&v| v.is_finite()));
        // Max element should have least negative log-softmax.
        assert!(output[3] > output[0]);
    }

    #[test]
    fn log_softmax_100_elements() {
        let input: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let mut output = vec![0.0f32; 100];
        neon_log_softmax_f32(&input, &mut output);
        let expected = reference_log_softmax(&input);
        assert_close(&output, &expected, 1e-2, "log_softmax_100");
    }

    #[test]
    fn log_softmax_negative_inputs() {
        let input = [-10.0, -5.0, -1.0, 0.0];
        let mut output = vec![0.0f32; 4];
        neon_log_softmax_f32(&input, &mut output);
        let expected = reference_log_softmax(&input);
        assert_close(&output, &expected, 1e-3, "log_softmax_neg");
    }

    // ── neon_softmax_with_temperature_f32 ───────────────────────────

    #[test]
    fn temperature_empty() {
        let input: &[f32] = &[];
        let mut output: Vec<f32> = vec![];
        neon_softmax_with_temperature_f32(input, &mut output, 1.0);
        assert!(output.is_empty());
    }

    #[test]
    fn temperature_single() {
        let input = [7.0f32];
        let mut output = vec![0.0f32; 1];
        neon_softmax_with_temperature_f32(&input, &mut output, 2.0);
        assert!((output[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn temperature_one_is_standard() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut t1_out = vec![0.0f32; 4];
        let mut sm_out = vec![0.0f32; 4];
        neon_softmax_with_temperature_f32(&input, &mut t1_out, 1.0);
        neon_softmax_f32(&input, &mut sm_out);
        assert_close(&t1_out, &sm_out, 1e-5, "temp_1_vs_standard");
    }

    #[test]
    fn temperature_near_zero_argmax() {
        let input = [1.0, 5.0, 3.0, 2.0];
        let mut output = vec![0.0f32; 4];
        neon_softmax_with_temperature_f32(&input, &mut output, 1e-10);
        assert!((output[1] - 1.0).abs() < 1e-5, "argmax at idx 1");
        assert!(output[0].abs() < 1e-5);
        assert!(output[2].abs() < 1e-5);
        assert!(output[3].abs() < 1e-5);
    }

    #[test]
    fn temperature_zero_exact() {
        let input = [3.0, 1.0, 4.0, 1.0, 5.0];
        let mut output = vec![0.0f32; 5];
        neon_softmax_with_temperature_f32(&input, &mut output, 0.0);
        assert!((output[4] - 1.0).abs() < 1e-5, "argmax at idx 4");
        assert!(output[..4].iter().all(|&v| v.abs() < 1e-5));
    }

    #[test]
    fn temperature_high_approaches_uniform() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];
        neon_softmax_with_temperature_f32(&input, &mut output, 1000.0);
        for &v in &output {
            assert!((v - 0.25).abs() < 0.01, "high temp should be ~uniform, got {v}");
        }
    }

    #[test]
    fn temperature_low_sharpens() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut low_out = vec![0.0f32; 4];
        let mut std_out = vec![0.0f32; 4];
        neon_softmax_with_temperature_f32(&input, &mut low_out, 0.1);
        neon_softmax_f32(&input, &mut std_out);
        // Low temperature should make the max-element probability higher.
        assert!(low_out[3] > std_out[3]);
    }

    #[test]
    fn temperature_sums_to_one() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0f32; 8];
        neon_softmax_with_temperature_f32(&input, &mut output, 0.5);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum = {sum}");
    }

    #[test]
    fn temperature_negative_temp_handled() {
        // Negative temperature inverts ordering; still should produce valid probs.
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];
        neon_softmax_with_temperature_f32(&input, &mut output, -1.0);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
        // With negative temp, smallest input gets highest prob.
        assert!(output[0] > output[3]);
    }

    #[test]
    fn temperature_eight_elements() {
        let input = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let mut output = vec![0.0f32; 8];
        neon_softmax_with_temperature_f32(&input, &mut output, 2.0);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    // ── neon_masked_softmax_f32 ─────────────────────────────────────

    #[test]
    fn masked_softmax_empty() {
        let input: &[f32] = &[];
        let mask: &[bool] = &[];
        let mut output: Vec<f32> = vec![];
        neon_masked_softmax_f32(input, mask, &mut output);
        assert!(output.is_empty());
    }

    #[test]
    fn masked_softmax_none_masked() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mask = [false, false, false, false];
        let mut output = vec![0.0f32; 4];
        neon_masked_softmax_f32(&input, &mask, &mut output);
        let mut expected = vec![0.0f32; 4];
        neon_softmax_f32(&input, &mut expected);
        assert_close(&output, &expected, 1e-5, "mask_none");
    }

    #[test]
    fn masked_softmax_all_masked() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mask = [true, true, true, true];
        let mut output = vec![0.0f32; 4];
        neon_masked_softmax_f32(&input, &mask, &mut output);
        for &v in &output {
            assert!(v.abs() < 1e-10, "all masked → all zeros");
        }
    }

    #[test]
    fn masked_softmax_partial_mask() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mask = [true, false, true, false];
        let mut output = vec![0.0f32; 4];
        neon_masked_softmax_f32(&input, &mask, &mut output);
        // Masked positions should be ≈ 0.
        assert!(output[0] < 1e-5);
        assert!(output[2] < 1e-5);
        // Unmasked should sum to ≈ 1.
        let unmasked_sum = output[1] + output[3];
        assert!((unmasked_sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn masked_softmax_single_unmasked() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mask = [true, true, false, true];
        let mut output = vec![0.0f32; 4];
        neon_masked_softmax_f32(&input, &mask, &mut output);
        assert!((output[2] - 1.0).abs() < 1e-4);
        assert!(output[0] < 1e-5);
        assert!(output[1] < 1e-5);
        assert!(output[3] < 1e-5);
    }

    #[test]
    fn masked_softmax_eight_elements() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mask = [true, false, true, false, true, false, true, false];
        let mut output = vec![0.0f32; 8];
        neon_masked_softmax_f32(&input, &mask, &mut output);
        // Masked positions should be ≈ 0.
        assert!(output[0] < 1e-5);
        assert!(output[2] < 1e-5);
        assert!(output[4] < 1e-5);
        assert!(output[6] < 1e-5);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "sum = {sum}");
    }

    #[test]
    fn masked_softmax_preserves_unmasked_order() {
        let input = [3.0, 1.0, 4.0, 1.0, 5.0];
        let mask = [false, true, false, true, false];
        let mut output = vec![0.0f32; 5];
        neon_masked_softmax_f32(&input, &mask, &mut output);
        assert!(output[4] > output[2]); // 5 > 4
        assert!(output[2] > output[0]); // 4 > 3
    }

    #[test]
    fn masked_softmax_non_aligned() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mask = [false, false, true, false, false];
        let mut output = vec![0.0f32; 5];
        neon_masked_softmax_f32(&input, &mask, &mut output);
        assert!(output[2] < 1e-5);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3);
    }

    // ── neon_top_k_softmax_f32 ──────────────────────────────────────

    #[test]
    fn top_k_empty() {
        let input: &[f32] = &[];
        let mut output: Vec<f32> = vec![];
        neon_top_k_softmax_f32(input, &mut output, 3);
        assert!(output.is_empty());
    }

    #[test]
    fn top_k_zero() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];
        neon_top_k_softmax_f32(&input, &mut output, 0);
        for &v in &output {
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn top_k_one() {
        let input = [1.0, 5.0, 3.0, 2.0];
        let mut output = vec![0.0f32; 4];
        neon_top_k_softmax_f32(&input, &mut output, 1);
        assert!((output[1] - 1.0).abs() < 1e-5, "top-1 at idx 1");
        assert!(output[0].abs() < 1e-10);
        assert!(output[2].abs() < 1e-10);
        assert!(output[3].abs() < 1e-10);
    }

    #[test]
    fn top_k_equals_len() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut top_out = vec![0.0f32; 4];
        let mut full_out = vec![0.0f32; 4];
        neon_top_k_softmax_f32(&input, &mut top_out, 4);
        neon_softmax_f32(&input, &mut full_out);
        assert_close(&top_out, &full_out, 1e-5, "top_k_full");
    }

    #[test]
    fn top_k_exceeds_len() {
        let input = [1.0, 2.0, 3.0];
        let mut top_out = vec![0.0f32; 3];
        let mut full_out = vec![0.0f32; 3];
        neon_top_k_softmax_f32(&input, &mut top_out, 100);
        neon_softmax_f32(&input, &mut full_out);
        assert_close(&top_out, &full_out, 1e-5, "top_k_exceed");
    }

    #[test]
    fn top_k_two_of_four() {
        let input = [1.0, 4.0, 2.0, 3.0];
        let mut output = vec![0.0f32; 4];
        neon_top_k_softmax_f32(&input, &mut output, 2);
        // Top-2 are indices 1 (val=4) and 3 (val=3).
        assert!(output[0].abs() < 1e-10);
        assert!(output[2].abs() < 1e-10);
        assert!(output[1] > 0.0);
        assert!(output[3] > 0.0);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn top_k_three_of_eight() {
        let input = [1.0, 8.0, 3.0, 6.0, 2.0, 7.0, 4.0, 5.0];
        let mut output = vec![0.0f32; 8];
        neon_top_k_softmax_f32(&input, &mut output, 3);
        // Top-3 values: 8 (idx1), 7 (idx5), 6 (idx3).
        assert!(output[1] > 0.0);
        assert!(output[5] > 0.0);
        assert!(output[3] > 0.0);
        // Others should be 0.
        assert!(output[0].abs() < 1e-10);
        assert!(output[2].abs() < 1e-10);
        assert!(output[4].abs() < 1e-10);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn top_k_single_element() {
        let input = [42.0f32];
        let mut output = vec![0.0f32; 1];
        neon_top_k_softmax_f32(&input, &mut output, 1);
        assert!((output[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn top_k_preserves_relative_probs() {
        let input = [1.0, 10.0, 5.0, 2.0];
        let mut output = vec![0.0f32; 4];
        neon_top_k_softmax_f32(&input, &mut output, 2);
        // Top-2 are idx1 (10) and idx2 (5); idx1 should have higher prob.
        assert!(output[1] > output[2]);
    }

    // ── neon_softmax_backward_f32 ───────────────────────────────────

    #[test]
    fn backward_empty() {
        let output: &[f32] = &[];
        let grad_output: &[f32] = &[];
        let mut grad_input: Vec<f32> = vec![];
        neon_softmax_backward_f32(output, grad_output, &mut grad_input);
    }

    #[test]
    fn backward_single() {
        let output = [1.0f32];
        let grad_output = [1.0f32];
        let mut grad_input = vec![0.0f32; 1];
        neon_softmax_backward_f32(&output, &grad_output, &mut grad_input);
        // y*(g - y*g) = 1*(1 - 1) = 0
        assert!(grad_input[0].abs() < 1e-6);
    }

    #[test]
    fn backward_four_elements() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0f32; 4];
        neon_softmax_f32(&input, &mut y);
        let grad_out = [1.0, 0.0, 0.0, 0.0];
        let mut grad_in = vec![0.0f32; 4];
        neon_softmax_backward_f32(&y, &grad_out, &mut grad_in);

        // Reference: dot = y[0]*1 = y[0]
        // grad_in[i] = y[i] * (grad_out[i] - dot)
        let dot: f32 = y[0];
        for i in 0..4 {
            let expected = y[i] * (grad_out[i] - dot);
            assert!(
                (grad_in[i] - expected).abs() < 1e-5,
                "backward[{i}]: {} vs {expected}",
                grad_in[i]
            );
        }
    }

    #[test]
    fn backward_gradient_sums_to_zero() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut y = vec![0.0f32; 5];
        neon_softmax_f32(&input, &mut y);
        let grad_out = [0.1, 0.2, 0.3, 0.4, 0.5];
        let mut grad_in = vec![0.0f32; 5];
        neon_softmax_backward_f32(&y, &grad_out, &mut grad_in);

        // The Jacobian of softmax has the property that
        // Σ grad_input[i] = 0.
        let sum: f32 = grad_in.iter().sum();
        assert!(sum.abs() < 1e-4, "grad sum = {sum}");
    }

    #[test]
    fn backward_eight_elements() {
        let input = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let mut y = vec![0.0f32; 8];
        neon_softmax_f32(&input, &mut y);
        let grad_out = [1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5];
        let mut grad_in = vec![0.0f32; 8];
        neon_softmax_backward_f32(&y, &grad_out, &mut grad_in);

        let dot: f32 = y.iter().zip(grad_out.iter()).map(|(a, b)| a * b).sum();
        for i in 0..8 {
            let expected = y[i] * (grad_out[i] - dot);
            assert!((grad_in[i] - expected).abs() < 1e-5, "backward_8[{i}]");
        }
    }

    #[test]
    fn backward_non_aligned() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut y = vec![0.0f32; 7];
        neon_softmax_f32(&input, &mut y);
        let grad_out = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let mut grad_in = vec![0.0f32; 7];
        neon_softmax_backward_f32(&y, &grad_out, &mut grad_in);
        let sum: f32 = grad_in.iter().sum();
        assert!(sum.abs() < 1e-4, "grad sum = {sum}");
    }

    #[test]
    fn backward_uniform_grad_gives_zero() {
        // If grad_output is uniform, backward should be all zeros.
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0f32; 4];
        neon_softmax_f32(&input, &mut y);
        let grad_out = [1.0, 1.0, 1.0, 1.0];
        let mut grad_in = vec![0.0f32; 4];
        neon_softmax_backward_f32(&y, &grad_out, &mut grad_in);
        // dot = Σ y[i]*1 = 1.0; grad_in[i] = y[i]*(1 - 1) = 0
        for (i, &g) in grad_in.iter().enumerate() {
            assert!(g.abs() < 1e-5, "backward_uniform[{i}] = {g}");
        }
    }

    #[test]
    fn backward_100_elements() {
        let input: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let mut y = vec![0.0f32; 100];
        neon_softmax_f32(&input, &mut y);
        let grad_out: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.01).collect();
        let mut grad_in = vec![0.0f32; 100];
        neon_softmax_backward_f32(&y, &grad_out, &mut grad_in);
        let sum: f32 = grad_in.iter().sum();
        assert!(sum.abs() < 1e-3, "grad sum = {sum}");
    }

    // ── Cross-function consistency ──────────────────────────────────

    #[test]
    fn log_softmax_consistency_with_softmax() {
        let input = [1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0];
        let mut sm = vec![0.0f32; 8];
        let mut lsm = vec![0.0f32; 8];
        neon_softmax_f32(&input, &mut sm);
        neon_log_softmax_f32(&input, &mut lsm);
        for i in 0..8 {
            let diff = (sm[i].ln() - lsm[i]).abs();
            assert!(diff < 1e-3, "ln(softmax) vs log_softmax [{i}]");
        }
    }

    #[test]
    fn temp_scaled_matches_manual_scale() {
        let input = [2.0, 4.0, 6.0, 8.0];
        let temp = 2.0f32;
        let mut temp_out = vec![0.0f32; 4];
        neon_softmax_with_temperature_f32(&input, &mut temp_out, temp);

        let scaled: Vec<f32> = input.iter().map(|&x| x / temp).collect();
        let mut manual_out = vec![0.0f32; 4];
        neon_softmax_f32(&scaled, &mut manual_out);
        assert_close(&temp_out, &manual_out, 1e-5, "temp_manual");
    }

    #[test]
    fn top_k_subset_sums_correctly() {
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; 16];
        neon_top_k_softmax_f32(&input, &mut output, 5);
        let nonzero: Vec<f32> = output.iter().copied().filter(|&v| v > 0.0).collect();
        assert_eq!(nonzero.len(), 5, "should have exactly 5 nonzero");
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn softmax_monotonically_increasing_input() {
        let input: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; 12];
        neon_softmax_f32(&input, &mut output);
        for i in 1..12 {
            assert!(
                output[i] >= output[i - 1],
                "monotonicity at {i}: {} < {}",
                output[i],
                output[i - 1]
            );
        }
    }

    #[test]
    fn softmax_symmetry() {
        let input = [1.0, 2.0, 1.0, 2.0];
        let mut output = vec![0.0f32; 4];
        neon_softmax_f32(&input, &mut output);
        assert!((output[0] - output[2]).abs() < 1e-6, "symmetric 0 vs 2");
        assert!((output[1] - output[3]).abs() < 1e-6, "symmetric 1 vs 3");
    }

    #[test]
    fn softmax_three_elements() {
        let input = [1.0, 2.0, 3.0];
        let mut output = vec![0.0f32; 3];
        neon_softmax_f32(&input, &mut output);
        let expected = reference_softmax(&input);
        assert_close(&output, &expected, 1e-4, "softmax_3");
    }

    #[test]
    fn inplace_sixteen_elements() {
        let orig: Vec<f32> = (0..16).map(|i| i as f32 * 0.3).collect();
        let mut data = orig.clone();
        neon_softmax_inplace_f32(&mut data);
        let expected = reference_softmax(&orig);
        assert_close(&data, &expected, 1e-4, "inplace_16");
    }

    #[test]
    fn log_softmax_sixteen() {
        let input: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let mut output = vec![0.0f32; 16];
        neon_log_softmax_f32(&input, &mut output);
        let expected = reference_log_softmax(&input);
        assert_close(&output, &expected, 1e-2, "log_softmax_16");
    }

    #[test]
    fn temperature_1000_elements() {
        let input: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.001).collect();
        let mut output = vec![0.0f32; 1000];
        neon_softmax_with_temperature_f32(&input, &mut output, 0.5);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "sum = {sum}");
    }

    #[test]
    fn masked_softmax_1000_elements() {
        let input: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.001).collect();
        let mask: Vec<bool> = (0..1000).map(|i| i % 3 == 0).collect();
        let mut output = vec![0.0f32; 1000];
        neon_masked_softmax_f32(&input, &mask, &mut output);
        // Masked positions (every 3rd) should be ≈ 0.
        for (i, &v) in output.iter().enumerate() {
            if mask[i] {
                assert!(v < 1e-5, "masked[{i}] = {v}");
            }
        }
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-2, "sum = {sum}");
    }

    #[test]
    fn backward_1000_elements_sum_zero() {
        let input: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.001).collect();
        let mut y = vec![0.0f32; 1000];
        neon_softmax_f32(&input, &mut y);
        let grad_out: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) * 0.001).collect();
        let mut grad_in = vec![0.0f32; 1000];
        neon_softmax_backward_f32(&y, &grad_out, &mut grad_in);
        let sum: f32 = grad_in.iter().sum();
        assert!(sum.abs() < 0.1, "grad sum = {sum}");
    }

    #[test]
    fn top_k_1000_elements() {
        let input: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.01).collect();
        let mut output = vec![0.0f32; 1000];
        neon_top_k_softmax_f32(&input, &mut output, 10);
        let nonzero_count = output.iter().filter(|&&v| v > 0.0).count();
        assert_eq!(nonzero_count, 10);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3);
    }

    #[test]
    fn softmax_all_nonnegative() {
        let input = [-5.0, -3.0, 0.0, 3.0, 5.0, 10.0, -10.0, 1.0];
        let mut output = vec![0.0f32; 8];
        neon_softmax_f32(&input, &mut output);
        for &v in &output {
            assert!(v >= 0.0, "negative probability: {v}");
        }
    }

    #[test]
    fn log_softmax_1000_elements() {
        let input: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.001).collect();
        let mut output = vec![0.0f32; 1000];
        neon_log_softmax_f32(&input, &mut output);
        // All log-softmax values should be ≤ 0.
        for (i, &v) in output.iter().enumerate() {
            assert!(v <= 1e-6, "log_softmax[{i}] = {v} > 0");
        }
    }

    #[test]
    fn inplace_1000_elements() {
        let orig: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.001).collect();
        let mut data = orig.clone();
        neon_softmax_inplace_f32(&mut data);
        let expected = reference_softmax(&orig);
        assert_close(&data, &expected, 1e-3, "inplace_1000");
    }

    #[test]
    fn backward_sign_correctness() {
        // For the argmax position with grad=1, others grad=0:
        // grad_input[argmax] > 0, grad_input[others] < 0
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0f32; 4];
        neon_softmax_f32(&input, &mut y);
        let grad_out = [0.0, 0.0, 0.0, 1.0]; // one-hot at argmax
        let mut grad_in = vec![0.0f32; 4];
        neon_softmax_backward_f32(&y, &grad_out, &mut grad_in);
        assert!(grad_in[3] > 0.0, "argmax grad should be positive");
        for i in 0..3 {
            assert!(grad_in[i] < 0.0, "non-argmax grad[{i}] should be negative");
        }
    }

    #[test]
    fn softmax_output_buffer_larger_than_input() {
        let input = [1.0, 2.0, 3.0];
        let mut output = vec![0.0f32; 10]; // larger buffer
        neon_softmax_f32(&input, &mut output);
        let expected = reference_softmax(&input);
        assert_close(&output[..3], &expected, 1e-4, "larger_buffer");
    }

    #[test]
    fn masked_softmax_last_unmasked() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mask = [true, true, true, false];
        let mut output = vec![0.0f32; 4];
        neon_masked_softmax_f32(&input, &mask, &mut output);
        assert!((output[3] - 1.0).abs() < 1e-4, "only unmasked");
    }

    #[test]
    fn temperature_fractional() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0f32; 8];
        neon_softmax_with_temperature_f32(&input, &mut output, 0.3);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
        // Very low temp → strongly peaked at max.
        assert!(output[7] > 0.9);
    }
}
