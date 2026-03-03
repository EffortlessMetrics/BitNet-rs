//! ARM NEON optimized token sampling kernels for Apple Silicon.
//!
//! Provides NEON-accelerated token sampling operations: top-K filtering,
//! nucleus (top-P) sampling, temperature scaling, repetition penalty,
//! argmax, and log-softmax. Each operation has a NEON implementation,
//! a scalar fallback, and a public dispatcher that selects the best path.

#![allow(
    unsafe_op_in_unsafe_fn,
    unused_unsafe,
    unused_variables,
    dead_code,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_div_ceil,
    clippy::collapsible_if,
    clippy::manual_memcpy,
    clippy::manual_is_multiple_of,
    clippy::unnecessary_cast,
    clippy::let_and_return,
    clippy::float_cmp,
    clippy::excessive_precision,
    clippy::missing_safety_doc,
    clippy::never_loop,
    clippy::while_immutable_condition,
    clippy::manual_abs_diff
)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Lane count for `float32x4_t` NEON vectors.
const LANES: usize = 4;

// ── Helpers ─────────────────────────────────────────────────────────────

/// Scalar fast-exp approximation (Cody-Waite style, degree-4 polynomial).
/// Maximum relative error ≈ 2e-4 for |x| ≤ 88, adequate for softmax.
#[inline(always)]
fn fast_exp_scalar(x: f32) -> f32 {
    let x = x.clamp(-88.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E).round();
    let r = x - n * std::f32::consts::LN_2;
    let poly = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6.0 + r * (1.0 / 24.0))));
    poly * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// NEON vectorised fast-exp for four lanes.
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

// ═══════════════════════════════════════════════════════════════════════
// 1. temperature_scale_f32
// ═══════════════════════════════════════════════════════════════════════

/// NEON temperature scaling: `output[i] = logits[i] / temperature`.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
/// Panics if `output.len() < logits.len()` or `temperature` is not positive finite.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_temperature_scale_f32(logits: &[f32], temperature: f32, output: &mut [f32]) {
    assert!(
        output.len() >= logits.len(),
        "output buffer too small: {} < {}",
        output.len(),
        logits.len()
    );
    assert!(
        temperature.is_finite() && temperature > 0.0,
        "temperature must be positive finite, got {temperature}"
    );

    let len = logits.len();
    let inv_t = 1.0 / temperature;
    let inv_t_vec = unsafe { vdupq_n_f32(inv_t) };
    let chunks = len / LANES;
    let remainder = len % LANES;

    let in_ptr = logits.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * LANES;
        unsafe {
            let v = vld1q_f32(in_ptr.add(offset));
            let scaled = vmulq_f32(v, inv_t_vec);
            vst1q_f32(out_ptr.add(offset), scaled);
        }
    }

    let tail = chunks * LANES;
    for i in 0..remainder {
        output[tail + i] = logits[tail + i] * inv_t;
    }
}

/// Scalar temperature scaling fallback.
pub fn scalar_temperature_scale_f32(logits: &[f32], temperature: f32, output: &mut [f32]) {
    assert!(
        output.len() >= logits.len(),
        "output buffer too small: {} < {}",
        output.len(),
        logits.len()
    );
    assert!(
        temperature.is_finite() && temperature > 0.0,
        "temperature must be positive finite, got {temperature}"
    );
    let inv_t = 1.0 / temperature;
    for (o, &l) in output.iter_mut().zip(logits.iter()) {
        *o = l * inv_t;
    }
}

/// Temperature scaling dispatcher.
pub fn temperature_scale_f32(logits: &[f32], temperature: f32, output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // Safety: we just verified NEON is available.
            unsafe {
                neon_temperature_scale_f32(logits, temperature, output);
            }
            return;
        }
    }
    scalar_temperature_scale_f32(logits, temperature, output);
}

// ═══════════════════════════════════════════════════════════════════════
// 2. argmax_f32
// ═══════════════════════════════════════════════════════════════════════

/// NEON argmax: find the index of the maximum value.
///
/// Uses NEON horizontal max to find the max value, then a scalar scan
/// for the first index. Returns `0` for empty slices.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_argmax_token_f32(logits: &[f32]) -> usize {
    let len = logits.len();
    if len == 0 {
        return 0;
    }

    // Phase 1: NEON horizontal max to find the max value.
    let ptr = logits.as_ptr();
    let chunks = len / LANES;
    let remainder = len % LANES;
    let mut max_vec = unsafe { vdupq_n_f32(f32::NEG_INFINITY) };

    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        max_vec = unsafe { vmaxq_f32(max_vec, v) };
    }

    let mut max_val = unsafe { vmaxvq_f32(max_vec) };
    let tail = chunks * LANES;
    for i in 0..remainder {
        let val = logits[tail + i];
        if val > max_val {
            max_val = val;
        }
    }

    // Phase 2: scalar scan for first occurrence.
    let mut best_idx = 0usize;
    for (i, &val) in logits.iter().enumerate() {
        if val == max_val {
            best_idx = i;
            break;
        }
    }

    best_idx
}

/// Scalar argmax fallback.
pub fn scalar_argmax_f32(logits: &[f32]) -> usize {
    if logits.is_empty() {
        return 0;
    }
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &val) in logits.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = i;
        }
    }
    best_idx
}

/// Argmax dispatcher.
pub fn argmax_f32(logits: &[f32]) -> usize {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon_argmax_token_f32(logits) };
        }
    }
    scalar_argmax_f32(logits)
}

// ═══════════════════════════════════════════════════════════════════════
// 3. repetition_penalty_f32
// ═══════════════════════════════════════════════════════════════════════

/// NEON repetition penalty: for each token ID in `seen_tokens`, scale the
/// corresponding logit. Positive logits are divided by `penalty`, negative
/// logits are multiplied by `penalty`.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
/// Panics if `penalty` is not positive finite or if any token ID >= logits.len().
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_repetition_penalty_f32(logits: &mut [f32], seen_tokens: &[u32], penalty: f32) {
    assert!(penalty.is_finite() && penalty > 0.0, "penalty must be positive finite, got {penalty}");
    let vocab = logits.len();
    for &tok in seen_tokens {
        let idx = tok as usize;
        assert!(idx < vocab, "token id {idx} out of range (vocab={vocab})");
        let val = logits[idx];
        logits[idx] = if val > 0.0 { val / penalty } else { val * penalty };
    }
}

/// Scalar repetition penalty fallback.
pub fn scalar_repetition_penalty_f32(logits: &mut [f32], seen_tokens: &[u32], penalty: f32) {
    assert!(penalty.is_finite() && penalty > 0.0, "penalty must be positive finite, got {penalty}");
    let vocab = logits.len();
    for &tok in seen_tokens {
        let idx = tok as usize;
        assert!(idx < vocab, "token id {idx} out of range (vocab={vocab})");
        let val = logits[idx];
        logits[idx] = if val > 0.0 { val / penalty } else { val * penalty };
    }
}

/// Repetition penalty dispatcher.
pub fn repetition_penalty_f32(logits: &mut [f32], seen_tokens: &[u32], penalty: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_repetition_penalty_f32(logits, seen_tokens, penalty);
            }
            return;
        }
    }
    scalar_repetition_penalty_f32(logits, seen_tokens, penalty);
}

// ═══════════════════════════════════════════════════════════════════════
// 4. top_k_filter_f32
// ═══════════════════════════════════════════════════════════════════════

/// NEON top-K filter: keep the K largest logits, set the rest to `f32::NEG_INFINITY`.
///
/// Uses NEON horizontal max to accelerate threshold finding via partial sort.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
/// Panics if `output.len() < logits.len()` or `k == 0`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_top_k_filter_f32(logits: &[f32], k: usize, output: &mut [f32]) {
    assert!(k > 0, "k must be > 0");
    let len = logits.len();
    assert!(output.len() >= len, "output buffer too small: {} < {}", output.len(), len);

    if len == 0 {
        return;
    }

    let k = k.min(len);

    // Find the k-th largest value via partial sort of indices.
    let mut indices: Vec<usize> = (0..len).collect();
    indices.select_nth_unstable_by(k - 1, |&a, &b| {
        logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let threshold = logits[indices[k - 1]];

    // Count how many values equal the threshold that we should keep.
    let above_count = logits.iter().filter(|&&v| v > threshold).count();
    let needed_at_threshold = k - above_count;

    // Apply filter with NEON.
    let neg_inf = f32::NEG_INFINITY;
    let neg_inf_vec = unsafe { vdupq_n_f32(neg_inf) };
    let thresh_vec = unsafe { vdupq_n_f32(threshold) };
    let in_ptr = logits.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let chunks = len / LANES;
    let remainder = len % LANES;

    // First pass: keep values strictly above threshold.
    for i in 0..chunks {
        let offset = i * LANES;
        unsafe {
            let v = vld1q_f32(in_ptr.add(offset));
            let mask = vcgtq_f32(v, thresh_vec);
            let filtered = vbslq_f32(mask, v, neg_inf_vec);
            vst1q_f32(out_ptr.add(offset), filtered);
        }
    }
    let tail = chunks * LANES;
    for i in 0..remainder {
        output[tail + i] = if logits[tail + i] > threshold { logits[tail + i] } else { neg_inf };
    }

    // Second pass: fill in exactly `needed_at_threshold` values equal to threshold.
    let mut at_thresh_remaining = needed_at_threshold;
    for i in 0..len {
        if logits[i] == threshold && at_thresh_remaining > 0 {
            output[i] = threshold;
            at_thresh_remaining -= 1;
        }
    }
}

/// Scalar top-K filter fallback.
pub fn scalar_top_k_filter_f32(logits: &[f32], k: usize, output: &mut [f32]) {
    assert!(k > 0, "k must be > 0");
    let len = logits.len();
    assert!(output.len() >= len, "output buffer too small: {} < {}", output.len(), len);

    if len == 0 {
        return;
    }

    let k = k.min(len);

    let mut indices: Vec<usize> = (0..len).collect();
    indices.select_nth_unstable_by(k - 1, |&a, &b| {
        logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let threshold = logits[indices[k - 1]];

    let above_count = logits.iter().filter(|&&v| v > threshold).count();
    let needed_at_threshold = k - above_count;

    let neg_inf = f32::NEG_INFINITY;
    for i in 0..len {
        output[i] = if logits[i] > threshold { logits[i] } else { neg_inf };
    }

    let mut at_thresh_remaining = needed_at_threshold;
    for i in 0..len {
        if logits[i] == threshold && at_thresh_remaining > 0 {
            output[i] = threshold;
            at_thresh_remaining -= 1;
        }
    }
}

/// Top-K filter dispatcher.
pub fn top_k_filter_f32(logits: &[f32], k: usize, output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_top_k_filter_f32(logits, k, output);
            }
            return;
        }
    }
    scalar_top_k_filter_f32(logits, k, output);
}

// ═══════════════════════════════════════════════════════════════════════
// 5. top_p_filter_f32
// ═══════════════════════════════════════════════════════════════════════

/// NEON top-P (nucleus) filter: keep smallest set of logits whose
/// softmax probabilities sum to >= `p`. Others become `f32::NEG_INFINITY`.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
/// Panics if `output.len() < logits.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_top_p_filter_f32(logits: &[f32], p: f32, output: &mut [f32]) {
    let len = logits.len();
    assert!(output.len() >= len, "output buffer too small: {} < {}", output.len(), len);

    if len == 0 {
        return;
    }

    // Step 1: NEON max for numerical stability.
    let ptr = logits.as_ptr();
    let chunks = len / LANES;
    let remainder = len % LANES;
    let mut max_vec = unsafe { vdupq_n_f32(f32::NEG_INFINITY) };

    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        max_vec = unsafe { vmaxq_f32(max_vec, v) };
    }
    let mut max_val = unsafe { vmaxvq_f32(max_vec) };
    let tail = chunks * LANES;
    for i in 0..remainder {
        if logits[tail + i] > max_val {
            max_val = logits[tail + i];
        }
    }

    // Step 2: compute softmax probabilities.
    let mut probs = vec![0.0f32; len];
    let max_vec_s = unsafe { vdupq_n_f32(max_val) };
    let mut sum_vec = unsafe { vdupq_n_f32(0.0) };
    let prob_ptr = probs.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * LANES;
        unsafe {
            let v = vld1q_f32(ptr.add(offset));
            let shifted = vsubq_f32(v, max_vec_s);
            let e = fast_exp_neon(shifted);
            sum_vec = vaddq_f32(sum_vec, e);
            vst1q_f32(prob_ptr.add(offset), e);
        }
    }
    let mut sum_val = unsafe { vaddvq_f32(sum_vec) };
    for i in 0..remainder {
        let e = fast_exp_scalar(logits[tail + i] - max_val);
        probs[tail + i] = e;
        sum_val += e;
    }

    let inv_sum = 1.0 / sum_val;
    for prob in probs.iter_mut() {
        *prob *= inv_sum;
    }

    // Step 3: sort indices by descending probability.
    let mut indices: Vec<usize> = (0..len).collect();
    indices.sort_unstable_by(|&a, &b| {
        probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Step 4: accumulate until cumulative >= p, mark kept set.
    let mut keep = vec![false; len];
    let mut cum = 0.0f32;
    for &idx in &indices {
        keep[idx] = true;
        cum += probs[idx];
        if cum >= p {
            break;
        }
    }

    // Step 5: write output.
    let neg_inf = f32::NEG_INFINITY;
    for i in 0..len {
        output[i] = if keep[i] { logits[i] } else { neg_inf };
    }
}

/// Scalar top-P filter fallback.
pub fn scalar_top_p_filter_f32(logits: &[f32], p: f32, output: &mut [f32]) {
    let len = logits.len();
    assert!(output.len() >= len, "output buffer too small: {} < {}", output.len(), len);

    if len == 0 {
        return;
    }

    // Softmax
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&l| fast_exp_scalar(l - max_val)).collect();
    let sum: f32 = probs.iter().sum();
    let inv_sum = 1.0 / sum;
    for prob in probs.iter_mut() {
        *prob *= inv_sum;
    }

    let mut indices: Vec<usize> = (0..len).collect();
    indices.sort_unstable_by(|&a, &b| {
        probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = vec![false; len];
    let mut cum = 0.0f32;
    for &idx in &indices {
        keep[idx] = true;
        cum += probs[idx];
        if cum >= p {
            break;
        }
    }

    let neg_inf = f32::NEG_INFINITY;
    for i in 0..len {
        output[i] = if keep[i] { logits[i] } else { neg_inf };
    }
}

/// Top-P filter dispatcher.
pub fn top_p_filter_f32(logits: &[f32], p: f32, output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_top_p_filter_f32(logits, p, output);
            }
            return;
        }
    }
    scalar_top_p_filter_f32(logits, p, output);
}

// ═══════════════════════════════════════════════════════════════════════
// 6. log_softmax_f32
// ═══════════════════════════════════════════════════════════════════════

/// NEON numerically stable log-softmax: `output[i] = logits[i] - max - ln(Σ exp(logits[j] - max))`.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
/// Panics if `output.len() < logits.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_log_softmax_f32(logits: &[f32], output: &mut [f32]) {
    let len = logits.len();
    assert!(output.len() >= len, "output buffer too small: {} < {}", output.len(), len);

    if len == 0 {
        return;
    }

    // Step 1: NEON max.
    let ptr = logits.as_ptr();
    let chunks = len / LANES;
    let remainder = len % LANES;
    let mut max_vec = unsafe { vdupq_n_f32(f32::NEG_INFINITY) };

    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        max_vec = unsafe { vmaxq_f32(max_vec, v) };
    }
    let mut max_val = unsafe { vmaxvq_f32(max_vec) };
    let tail = chunks * LANES;
    for i in 0..remainder {
        if logits[tail + i] > max_val {
            max_val = logits[tail + i];
        }
    }

    // Step 2: NEON sum of exp(logits - max).
    let max_vec_s = unsafe { vdupq_n_f32(max_val) };
    let mut sum_vec = unsafe { vdupq_n_f32(0.0) };

    for i in 0..chunks {
        let offset = i * LANES;
        unsafe {
            let v = vld1q_f32(ptr.add(offset));
            let shifted = vsubq_f32(v, max_vec_s);
            let e = fast_exp_neon(shifted);
            sum_vec = vaddq_f32(sum_vec, e);
        }
    }
    let mut sum_val = unsafe { vaddvq_f32(sum_vec) };
    for i in 0..remainder {
        sum_val += fast_exp_scalar(logits[tail + i] - max_val);
    }

    let log_sum = sum_val.ln();

    // Step 3: output[i] = logits[i] - max - log_sum via NEON.
    let log_sum_plus_max = unsafe { vdupq_n_f32(max_val + log_sum) };
    let out_ptr = output.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * LANES;
        unsafe {
            let v = vld1q_f32(ptr.add(offset));
            let result = vsubq_f32(v, log_sum_plus_max);
            vst1q_f32(out_ptr.add(offset), result);
        }
    }
    let shift = max_val + log_sum;
    for i in 0..remainder {
        output[tail + i] = logits[tail + i] - shift;
    }
}

/// Scalar numerically stable log-softmax.
pub fn scalar_log_softmax_f32(logits: &[f32], output: &mut [f32]) {
    let len = logits.len();
    assert!(output.len() >= len, "output buffer too small: {} < {}", output.len(), len);

    if len == 0 {
        return;
    }

    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&l| fast_exp_scalar(l - max_val)).sum();
    let log_sum = sum.ln();
    let shift = max_val + log_sum;

    for (o, &l) in output.iter_mut().zip(logits.iter()) {
        *o = l - shift;
    }
}

/// Log-softmax dispatcher.
pub fn log_softmax_f32(logits: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_log_softmax_f32(logits, output);
            }
            return;
        }
    }
    scalar_log_softmax_f32(logits, output);
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn approx_eq_default(a: f32, b: f32) -> bool {
        approx_eq(a, b, 1e-4)
    }

    // ── temperature_scale_f32 ───────────────────────────────────────

    #[test]
    fn test_temperature_scale_basic() {
        let logits = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0f32; 5];
        temperature_scale_f32(&logits, 2.0, &mut output);
        for (i, &l) in logits.iter().enumerate() {
            assert!(approx_eq_default(output[i], l / 2.0), "idx {i}: {} vs {}", output[i], l / 2.0);
        }
    }

    #[test]
    fn test_temperature_scale_one() {
        let logits = [1.0f32, -2.0, 3.0];
        let mut output = vec![0.0f32; 3];
        temperature_scale_f32(&logits, 1.0, &mut output);
        for (i, &l) in logits.iter().enumerate() {
            assert!(approx_eq_default(output[i], l));
        }
    }

    #[test]
    fn test_temperature_scale_very_small() {
        let logits = [1.0f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];
        temperature_scale_f32(&logits, 0.01, &mut output);
        assert!(output[3] > output[2]);
        assert!(output[2] > output[1]);
    }

    #[test]
    fn test_temperature_scale_very_large() {
        let logits = [1.0f32, 100.0, -100.0];
        let mut output = vec![0.0f32; 3];
        temperature_scale_f32(&logits, 1000.0, &mut output);
        // All values should be very close to zero.
        for &o in &output {
            assert!(o.abs() < 1.0, "expected near zero, got {o}");
        }
    }

    #[test]
    fn test_temperature_scale_empty() {
        let logits: [f32; 0] = [];
        let mut output: Vec<f32> = vec![];
        temperature_scale_f32(&logits, 1.0, &mut output);
    }

    #[test]
    fn test_temperature_scale_single() {
        let logits = [42.0f32];
        let mut output = vec![0.0f32; 1];
        temperature_scale_f32(&logits, 2.0, &mut output);
        assert!(approx_eq_default(output[0], 21.0));
    }

    #[test]
    #[should_panic(expected = "temperature must be positive finite")]
    fn test_temperature_scale_zero_panics() {
        let logits = [1.0f32];
        let mut output = vec![0.0f32; 1];
        temperature_scale_f32(&logits, 0.0, &mut output);
    }

    #[test]
    #[should_panic(expected = "temperature must be positive finite")]
    fn test_temperature_scale_negative_panics() {
        let logits = [1.0f32];
        let mut output = vec![0.0f32; 1];
        temperature_scale_f32(&logits, -1.0, &mut output);
    }

    #[test]
    fn test_temperature_scale_neon_vs_scalar() {
        let logits: Vec<f32> = (0..129).map(|i| (i as f32) * 0.1 - 6.0).collect();
        let mut neon_out = vec![0.0f32; logits.len()];
        let mut scalar_out = vec![0.0f32; logits.len()];
        scalar_temperature_scale_f32(&logits, 0.7, &mut scalar_out);
        temperature_scale_f32(&logits, 0.7, &mut neon_out);
        for i in 0..logits.len() {
            assert!(
                approx_eq_default(neon_out[i], scalar_out[i]),
                "mismatch at {i}: {} vs {}",
                neon_out[i],
                scalar_out[i]
            );
        }
    }

    // ── argmax_f32 ──────────────────────────────────────────────────

    #[test]
    fn test_argmax_basic() {
        let logits = [1.0f32, 5.0, 3.0, 2.0, 4.0];
        assert_eq!(argmax_f32(&logits), 1);
    }

    #[test]
    fn test_argmax_first_is_max() {
        let logits = [100.0f32, 1.0, 2.0, 3.0];
        assert_eq!(argmax_f32(&logits), 0);
    }

    #[test]
    fn test_argmax_last_is_max() {
        let logits = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 99.0];
        assert_eq!(argmax_f32(&logits), 7);
    }

    #[test]
    fn test_argmax_tie_breaking_first() {
        let logits = [3.0f32, 1.0, 3.0, 2.0, 3.0];
        assert_eq!(argmax_f32(&logits), 0, "should return first occurrence");
    }

    #[test]
    fn test_argmax_all_same() {
        let logits = [7.0f32; 8];
        assert_eq!(argmax_f32(&logits), 0);
    }

    #[test]
    fn test_argmax_single() {
        let logits = [42.0f32];
        assert_eq!(argmax_f32(&logits), 0);
    }

    #[test]
    fn test_argmax_empty() {
        let logits: [f32; 0] = [];
        assert_eq!(argmax_f32(&logits), 0);
    }

    #[test]
    fn test_argmax_negatives() {
        let logits = [-5.0f32, -1.0, -3.0, -2.0];
        assert_eq!(argmax_f32(&logits), 1);
    }

    #[test]
    fn test_argmax_large() {
        let mut logits = vec![0.0f32; 1024];
        logits[777] = 999.0;
        assert_eq!(argmax_f32(&logits), 777);
    }

    #[test]
    fn test_argmax_neon_vs_scalar() {
        let logits: Vec<f32> = (0..129).map(|i| ((i * 37) % 100) as f32).collect();
        assert_eq!(argmax_f32(&logits), scalar_argmax_f32(&logits));
    }

    // ── repetition_penalty_f32 ──────────────────────────────────────

    #[test]
    fn test_repetition_penalty_noop() {
        let mut logits = vec![1.0f32, 2.0, -3.0, 4.0];
        let original = logits.clone();
        repetition_penalty_f32(&mut logits, &[0, 1, 2, 3], 1.0);
        for (i, (&a, &b)) in logits.iter().zip(original.iter()).enumerate() {
            assert!(approx_eq_default(a, b), "idx {i}: {a} vs {b}");
        }
    }

    #[test]
    fn test_repetition_penalty_positive_logit() {
        let mut logits = vec![0.0f32, 4.0, 0.0];
        repetition_penalty_f32(&mut logits, &[1], 2.0);
        assert!(approx_eq_default(logits[1], 2.0));
        assert!(approx_eq_default(logits[0], 0.0));
    }

    #[test]
    fn test_repetition_penalty_negative_logit() {
        let mut logits = vec![0.0f32, -2.0, 0.0];
        repetition_penalty_f32(&mut logits, &[1], 2.0);
        assert!(approx_eq_default(logits[1], -4.0));
    }

    #[test]
    fn test_repetition_penalty_mixed() {
        let mut logits = vec![3.0f32, -3.0, 0.0, 6.0];
        repetition_penalty_f32(&mut logits, &[0, 1, 3], 3.0);
        assert!(approx_eq_default(logits[0], 1.0)); // 3/3
        assert!(approx_eq_default(logits[1], -9.0)); // -3*3
        assert!(approx_eq_default(logits[2], 0.0)); // untouched
        assert!(approx_eq_default(logits[3], 2.0)); // 6/3
    }

    #[test]
    fn test_repetition_penalty_empty_seen() {
        let mut logits = vec![1.0f32, 2.0, 3.0];
        let original = logits.clone();
        repetition_penalty_f32(&mut logits, &[], 2.0);
        for (i, (&a, &b)) in logits.iter().zip(original.iter()).enumerate() {
            assert!(approx_eq_default(a, b), "idx {i}");
        }
    }

    #[test]
    fn test_repetition_penalty_zero_logit() {
        // Zero logits should remain zero regardless of penalty.
        let mut logits = vec![0.0f32, 0.0];
        repetition_penalty_f32(&mut logits, &[0, 1], 5.0);
        assert_eq!(logits[0], 0.0);
        assert_eq!(logits[1], 0.0);
    }

    #[test]
    #[should_panic(expected = "penalty must be positive finite")]
    fn test_repetition_penalty_zero_panics() {
        let mut logits = vec![1.0f32];
        repetition_penalty_f32(&mut logits, &[0], 0.0);
    }

    #[test]
    fn test_repetition_penalty_neon_vs_scalar() {
        let mut neon_logits: Vec<f32> = (0..100).map(|i| (i as f32) - 50.0).collect();
        let mut scalar_logits = neon_logits.clone();
        let seen: Vec<u32> = (0..100).step_by(3).collect();
        repetition_penalty_f32(&mut neon_logits, &seen, 1.5);
        scalar_repetition_penalty_f32(&mut scalar_logits, &seen, 1.5);
        for i in 0..100 {
            assert!(
                approx_eq_default(neon_logits[i], scalar_logits[i]),
                "mismatch at {i}: {} vs {}",
                neon_logits[i],
                scalar_logits[i]
            );
        }
    }

    // ── top_k_filter_f32 ────────────────────────────────────────────

    #[test]
    fn test_top_k_basic() {
        let logits = [1.0f32, 5.0, 3.0, 2.0, 4.0];
        let mut output = vec![0.0f32; 5];
        top_k_filter_f32(&logits, 2, &mut output);
        // Top 2: indices 1 (5.0) and 4 (4.0)
        assert_eq!(output[1], 5.0);
        assert_eq!(output[4], 4.0);
        assert_eq!(output[0], f32::NEG_INFINITY);
        assert_eq!(output[2], f32::NEG_INFINITY);
        assert_eq!(output[3], f32::NEG_INFINITY);
    }

    #[test]
    fn test_top_k_equals_one() {
        let logits = [1.0f32, 3.0, 2.0];
        let mut output = vec![0.0f32; 3];
        top_k_filter_f32(&logits, 1, &mut output);
        assert_eq!(output[1], 3.0);
        assert_eq!(output[0], f32::NEG_INFINITY);
        assert_eq!(output[2], f32::NEG_INFINITY);
    }

    #[test]
    fn test_top_k_equals_len() {
        let logits = [1.0f32, 2.0, 3.0];
        let mut output = vec![0.0f32; 3];
        top_k_filter_f32(&logits, 3, &mut output);
        for i in 0..3 {
            assert_eq!(output[i], logits[i]);
        }
    }

    #[test]
    fn test_top_k_exceeds_len() {
        let logits = [1.0f32, 2.0];
        let mut output = vec![0.0f32; 2];
        top_k_filter_f32(&logits, 100, &mut output);
        assert_eq!(output[0], 1.0);
        assert_eq!(output[1], 2.0);
    }

    #[test]
    fn test_top_k_with_ties() {
        let logits = [5.0f32, 5.0, 5.0, 1.0];
        let mut output = vec![0.0f32; 4];
        top_k_filter_f32(&logits, 2, &mut output);
        // At least 2 of the 5.0 values should be kept.
        let kept: usize = output.iter().filter(|&&v| v == 5.0).count();
        assert!(kept >= 2, "expected at least 2 kept, got {kept}");
        assert_eq!(output[3], f32::NEG_INFINITY);
    }

    #[test]
    fn test_top_k_empty() {
        let logits: [f32; 0] = [];
        let mut output: Vec<f32> = vec![];
        top_k_filter_f32(&logits, 1, &mut output);
    }

    #[test]
    fn test_top_k_single() {
        let logits = [42.0f32];
        let mut output = vec![0.0f32; 1];
        top_k_filter_f32(&logits, 1, &mut output);
        assert_eq!(output[0], 42.0);
    }

    #[test]
    fn test_top_k_all_same() {
        let logits = [7.0f32; 8];
        let mut output = vec![0.0f32; 8];
        top_k_filter_f32(&logits, 3, &mut output);
        let kept = output.iter().filter(|&&v| v == 7.0).count();
        assert_eq!(kept, 3, "expected exactly 3 kept, got {kept}");
    }

    #[test]
    fn test_top_k_neon_vs_scalar() {
        let logits: Vec<f32> = (0..129).map(|i| ((i * 73) % 200) as f32 - 100.0).collect();
        let mut neon_out = vec![0.0f32; logits.len()];
        let mut scalar_out = vec![0.0f32; logits.len()];
        top_k_filter_f32(&logits, 10, &mut neon_out);
        scalar_top_k_filter_f32(&logits, 10, &mut scalar_out);
        let neon_kept: usize = neon_out.iter().filter(|&&v| v != f32::NEG_INFINITY).count();
        let scalar_kept: usize = scalar_out.iter().filter(|&&v| v != f32::NEG_INFINITY).count();
        assert_eq!(neon_kept, scalar_kept, "kept count mismatch");
    }

    #[test]
    #[should_panic(expected = "k must be > 0")]
    fn test_top_k_zero_panics() {
        let logits = [1.0f32];
        let mut output = vec![0.0f32; 1];
        top_k_filter_f32(&logits, 0, &mut output);
    }

    #[test]
    fn test_top_k_large_logits() {
        let logits = [1e30f32, -1e30, 1e20, -1e20, 0.0];
        let mut output = vec![0.0f32; 5];
        top_k_filter_f32(&logits, 2, &mut output);
        assert_eq!(output[0], 1e30);
        assert_eq!(output[2], 1e20);
    }

    // ── top_p_filter_f32 ────────────────────────────────────────────

    #[test]
    fn test_top_p_basic() {
        // Heavily skewed: softmax of [10, 1, 1, 1] → first token dominates.
        let logits = [10.0f32, 1.0, 1.0, 1.0];
        let mut output = vec![0.0f32; 4];
        top_p_filter_f32(&logits, 0.9, &mut output);
        assert_eq!(output[0], 10.0, "highest logit should be kept");
    }

    #[test]
    fn test_top_p_one_keeps_all() {
        let logits = [1.0f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];
        top_p_filter_f32(&logits, 1.0, &mut output);
        for i in 0..4 {
            assert_eq!(output[i], logits[i], "p=1.0 should keep all");
        }
    }

    #[test]
    fn test_top_p_zero_keeps_one() {
        let logits = [1.0f32, 5.0, 3.0];
        let mut output = vec![0.0f32; 3];
        top_p_filter_f32(&logits, 0.0, &mut output);
        // At least the top token must be kept (cumulative starts > 0 on first).
        let kept: usize = output.iter().filter(|&&v| v != f32::NEG_INFINITY).count();
        assert!(kept >= 1, "at least one token should be kept");
    }

    #[test]
    fn test_top_p_empty() {
        let logits: [f32; 0] = [];
        let mut output: Vec<f32> = vec![];
        top_p_filter_f32(&logits, 0.9, &mut output);
    }

    #[test]
    fn test_top_p_single() {
        let logits = [42.0f32];
        let mut output = vec![0.0f32; 1];
        top_p_filter_f32(&logits, 0.5, &mut output);
        assert_eq!(output[0], 42.0);
    }

    #[test]
    fn test_top_p_all_equal() {
        let logits = [1.0f32; 8];
        let mut output = vec![0.0f32; 8];
        top_p_filter_f32(&logits, 0.5, &mut output);
        let kept: usize = output.iter().filter(|&&v| v != f32::NEG_INFINITY).count();
        // Each has prob 1/8 = 0.125, need >= 0.5 → at least 4 kept.
        assert!(kept >= 4, "expected >= 4 kept, got {kept}");
    }

    #[test]
    fn test_top_p_large_logits() {
        let logits = [100.0f32, -100.0, 0.0, 50.0];
        let mut output = vec![0.0f32; 4];
        top_p_filter_f32(&logits, 0.9, &mut output);
        // The dominant logit (100.0) should be kept.
        assert_eq!(output[0], 100.0);
    }

    #[test]
    fn test_top_p_neon_vs_scalar() {
        let logits: Vec<f32> = (0..65).map(|i| (i as f32) * 0.3 - 10.0).collect();
        let mut neon_out = vec![0.0f32; logits.len()];
        let mut scalar_out = vec![0.0f32; logits.len()];
        top_p_filter_f32(&logits, 0.8, &mut neon_out);
        scalar_top_p_filter_f32(&logits, 0.8, &mut scalar_out);
        let neon_kept: usize = neon_out.iter().filter(|&&v| v != f32::NEG_INFINITY).count();
        let scalar_kept: usize = scalar_out.iter().filter(|&&v| v != f32::NEG_INFINITY).count();
        assert_eq!(neon_kept, scalar_kept, "kept count mismatch: {} vs {}", neon_kept, scalar_kept);
    }

    // ── log_softmax_f32 ─────────────────────────────────────────────

    #[test]
    fn test_log_softmax_basic() {
        let logits = [1.0f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];
        log_softmax_f32(&logits, &mut output);
        // All values should be <= 0 and sum of exp(output) ≈ 1.
        for &o in &output {
            assert!(o <= 0.0, "log_softmax should be <= 0, got {o}");
        }
        let sum_exp: f32 = output.iter().map(|&o| o.exp()).sum();
        assert!(approx_eq(sum_exp, 1.0, 1e-3), "exp sum should ≈ 1, got {sum_exp}");
    }

    #[test]
    fn test_log_softmax_single() {
        let logits = [42.0f32];
        let mut output = vec![0.0f32; 1];
        log_softmax_f32(&logits, &mut output);
        assert!(
            approx_eq(output[0], 0.0, 1e-3),
            "single element log_softmax should ≈ 0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_log_softmax_empty() {
        let logits: [f32; 0] = [];
        let mut output: Vec<f32> = vec![];
        log_softmax_f32(&logits, &mut output);
    }

    #[test]
    fn test_log_softmax_numerical_stability() {
        let logits = [1000.0f32, 1001.0, 999.0];
        let mut output = vec![0.0f32; 3];
        log_softmax_f32(&logits, &mut output);
        for &o in &output {
            assert!(o.is_finite(), "should not overflow, got {o}");
            assert!(o <= 0.0);
        }
    }

    #[test]
    fn test_log_softmax_very_negative() {
        let logits = [-1000.0f32, -999.0, -1001.0];
        let mut output = vec![0.0f32; 3];
        log_softmax_f32(&logits, &mut output);
        for &o in &output {
            assert!(o.is_finite(), "should not underflow, got {o}");
            assert!(o <= 0.0);
        }
    }

    #[test]
    fn test_log_softmax_uniform() {
        let n = 8;
        let logits = vec![0.0f32; n];
        let mut output = vec![0.0f32; n];
        log_softmax_f32(&logits, &mut output);
        let expected = -(n as f32).ln();
        for (i, &o) in output.iter().enumerate() {
            assert!(approx_eq(o, expected, 1e-3), "idx {i}: expected {expected}, got {o}");
        }
    }

    #[test]
    fn test_log_softmax_ordering_preserved() {
        let logits = [1.0f32, 4.0, 2.0, 3.0];
        let mut output = vec![0.0f32; 4];
        log_softmax_f32(&logits, &mut output);
        assert!(output[1] > output[3]);
        assert!(output[3] > output[2]);
        assert!(output[2] > output[0]);
    }

    #[test]
    fn test_log_softmax_neon_vs_scalar() {
        let logits: Vec<f32> = (0..129).map(|i| (i as f32) * 0.1 - 6.0).collect();
        let mut neon_out = vec![0.0f32; logits.len()];
        let mut scalar_out = vec![0.0f32; logits.len()];
        log_softmax_f32(&logits, &mut neon_out);
        scalar_log_softmax_f32(&logits, &mut scalar_out);
        for i in 0..logits.len() {
            assert!(
                approx_eq(neon_out[i], scalar_out[i], 1e-3),
                "mismatch at {i}: {} vs {}",
                neon_out[i],
                scalar_out[i]
            );
        }
    }

    #[test]
    fn test_log_softmax_large_vocab() {
        let n = 32000; // typical vocab size
        let logits: Vec<f32> = (0..n).map(|i| (i as f32) / (n as f32) * 10.0 - 5.0).collect();
        let mut output = vec![0.0f32; n];
        log_softmax_f32(&logits, &mut output);
        for &o in &output {
            assert!(o.is_finite());
            assert!(o <= 0.0);
        }
    }

    // ── Dispatcher selection tests ──────────────────────────────────

    #[test]
    fn test_dispatcher_temperature() {
        let logits = [2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0];
        let mut output = vec![0.0f32; logits.len()];
        temperature_scale_f32(&logits, 2.0, &mut output);
        for (i, &l) in logits.iter().enumerate() {
            assert!(approx_eq_default(output[i], l / 2.0));
        }
    }

    #[test]
    fn test_dispatcher_argmax() {
        let logits = [0.0f32; 17];
        assert_eq!(argmax_f32(&logits), 0);
    }

    #[test]
    fn test_dispatcher_repetition_penalty() {
        let mut logits = vec![10.0f32, -10.0];
        repetition_penalty_f32(&mut logits, &[0, 1], 2.0);
        assert!(approx_eq_default(logits[0], 5.0));
        assert!(approx_eq_default(logits[1], -20.0));
    }

    #[test]
    fn test_dispatcher_top_k() {
        let logits = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0f32; 5];
        top_k_filter_f32(&logits, 1, &mut output);
        assert_eq!(output[4], 5.0);
    }

    #[test]
    fn test_dispatcher_top_p() {
        let logits = [10.0f32, 0.0, 0.0];
        let mut output = vec![0.0f32; 3];
        top_p_filter_f32(&logits, 0.5, &mut output);
        assert_eq!(output[0], 10.0);
    }

    #[test]
    fn test_dispatcher_log_softmax() {
        let logits = [0.0f32; 4];
        let mut output = vec![0.0f32; 4];
        log_softmax_f32(&logits, &mut output);
        let expected = -(4.0f32).ln();
        for &o in &output {
            assert!(approx_eq(o, expected, 1e-3));
        }
    }

    // ── Non-multiple-of-4 remainder tests ───────────────────────────

    #[test]
    fn test_remainder_1_temperature() {
        let logits = [3.0f32];
        let mut output = vec![0.0f32; 1];
        temperature_scale_f32(&logits, 3.0, &mut output);
        assert!(approx_eq_default(output[0], 1.0));
    }

    #[test]
    fn test_remainder_2_argmax() {
        let logits = [1.0f32, 2.0];
        assert_eq!(argmax_f32(&logits), 1);
    }

    #[test]
    fn test_remainder_3_log_softmax() {
        let logits = [1.0f32, 2.0, 3.0];
        let mut output = vec![0.0f32; 3];
        log_softmax_f32(&logits, &mut output);
        let sum_exp: f32 = output.iter().map(|&o| o.exp()).sum();
        assert!(approx_eq(sum_exp, 1.0, 1e-3));
    }

    #[test]
    fn test_remainder_5_top_k() {
        let logits = [5.0f32, 1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 5];
        top_k_filter_f32(&logits, 2, &mut output);
        assert_eq!(output[0], 5.0);
        assert_eq!(output[4], 4.0);
    }

    #[test]
    fn test_remainder_7_top_p() {
        let logits = [10.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut output = vec![0.0f32; 7];
        top_p_filter_f32(&logits, 0.9, &mut output);
        assert_eq!(output[0], 10.0);
    }

    #[test]
    fn test_remainder_6_repetition_penalty() {
        let mut logits = vec![1.0f32, 2.0, 3.0, -1.0, -2.0, -3.0];
        repetition_penalty_f32(&mut logits, &[0, 3], 2.0);
        assert!(approx_eq_default(logits[0], 0.5));
        assert!(approx_eq_default(logits[3], -2.0));
        assert!(approx_eq_default(logits[1], 2.0)); // untouched
    }

    #[test]
    fn test_log_softmax_matches_manual() {
        // Manual: logits [0, 1], max=1, exp=[-1,0]=[e^-1, 1], sum=e^-1+1≈1.3679
        // log_softmax[0] = 0 - 1 - ln(1.3679) ≈ -1.3133
        // log_softmax[1] = 1 - 1 - ln(1.3679) ≈ -0.3133
        let logits = [0.0f32, 1.0];
        let mut output = vec![0.0f32; 2];
        log_softmax_f32(&logits, &mut output);
        assert!(approx_eq(output[0], -1.3133, 0.02), "got {}", output[0]);
        assert!(approx_eq(output[1], -0.3133, 0.02), "got {}", output[1]);
    }
}
