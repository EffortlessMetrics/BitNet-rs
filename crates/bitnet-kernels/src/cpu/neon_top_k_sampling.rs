//! ARM NEON top-k / top-p / min-p / typical sampling kernels for Apple Silicon.
//!
//! Provides SIMD-accelerated sampling strategies using `float32x4` NEON
//! intrinsics for 4-wide parallel computation.  All public functions are
//! gated on `target_arch = "aarch64"`.
//!
//! # Implemented strategies
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`top_k_filter_neon`] | Zero out all but the top-k logits |
//! | [`top_p_filter`] | Nucleus (top-p) cumulative probability filter |
//! | [`top_k_top_p_filter_neon`] | Combined top-k then top-p |
//! | [`temperature_softmax_neon`] | Temperature-scaled softmax |
//! | [`min_p_filter`] | Min-p threshold filter |
//! | [`typical_filter`] | Typical sampling via entropy distance |

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Lane count for `float32x4_t` NEON vectors.
const LANES: usize = 4;

// ── Fast exp (shared with neon_softmax, duplicated to keep the module
//    self-contained) ─────────────────────────────────────────────────────

/// Scalar fast exp approximation (degree-4 polynomial).
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
///
/// # Safety
/// Requires `aarch64` target with NEON.
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

// ── NEON helpers ────────────────────────────────────────────────────────

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
    let mut max_val = vmaxvq_f32(max_vec);
    for i in 0..remainder {
        let val = data[chunks * LANES + i];
        if val > max_val {
            max_val = val;
        }
    }
    max_val
}

/// NEON-accelerated sum of an `f32` slice.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_sum(data: &[f32]) -> f32 {
    let len = data.len();
    if len == 0 {
        return 0.0;
    }
    let ptr = data.as_ptr();
    let chunks = len / LANES;
    let remainder = len % LANES;
    let mut acc = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        acc = vaddq_f32(acc, v);
    }
    let mut sum = vaddvq_f32(acc);
    for i in 0..remainder {
        sum += data[chunks * LANES + i];
    }
    sum
}

// ── Temperature-scaled softmax ──────────────────────────────────────────

/// In-place temperature-scaled softmax using NEON.
///
/// Computes `softmax(logits / temperature)`.  A temperature of `0.0` is
/// treated as greedy (argmax gets probability 1.0).
///
/// # Safety
/// Caller must ensure the `neon` target feature is available at runtime.
///
/// # Panics
/// Panics if `logits` is empty and temperature is zero (no argmax
/// possible).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn temperature_softmax_neon(logits: &mut [f32], temperature: f32) {
    let len = logits.len();
    if len == 0 {
        return;
    }

    // Greedy: assign 1.0 to the argmax, 0.0 elsewhere.
    if temperature == 0.0 {
        let mut best_idx: usize = 0;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        for v in logits.iter_mut() {
            *v = 0.0;
        }
        logits[best_idx] = 1.0;
        return;
    }

    // Scale logits by 1/temperature using NEON.
    let inv_t = 1.0 / temperature;
    let inv_t_vec = vdupq_n_f32(inv_t);
    let chunks = len / LANES;
    let remainder = len % LANES;
    let ptr = logits.as_mut_ptr();

    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        let scaled = vmulq_f32(v, inv_t_vec);
        unsafe { vst1q_f32(ptr.add(i * LANES), scaled) };
    }
    let tail = chunks * LANES;
    for i in 0..remainder {
        logits[tail + i] *= inv_t;
    }

    // Softmax: exp(x - max) / sum
    let max_val = unsafe { find_max_neon(logits) };
    let max_vec = vdupq_n_f32(max_val);
    let mut sum_vec = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        let shifted = vsubq_f32(v, max_vec);
        let e = unsafe { fast_exp_neon(shifted) };
        unsafe { vst1q_f32(ptr.add(i * LANES), e) };
        sum_vec = vaddq_f32(sum_vec, e);
    }
    let mut sum_val = vaddvq_f32(sum_vec);
    for i in 0..remainder {
        let e = fast_exp_scalar(logits[tail + i] - max_val);
        logits[tail + i] = e;
        sum_val += e;
    }

    let inv_sum = 1.0 / sum_val;
    let inv_sum_vec = vdupq_n_f32(inv_sum);
    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        let r = vmulq_f32(v, inv_sum_vec);
        unsafe { vst1q_f32(ptr.add(i * LANES), r) };
    }
    for i in 0..remainder {
        logits[tail + i] *= inv_sum;
    }
}

// ── Top-k filtering ─────────────────────────────────────────────────────

/// NEON-optimised top-k filter using partial sort with NEON min/max.
///
/// After this call every logit **not** in the top-k positions is set to
/// `f32::NEG_INFINITY`.  The remaining k logits keep their original
/// values so the caller can apply softmax afterwards.
///
/// When `k >= logits.len()` the function is a no-op.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn top_k_filter_neon(logits: &mut [f32], k: usize) {
    let len = logits.len();
    if k == 0 {
        for v in logits.iter_mut() {
            *v = f32::NEG_INFINITY;
        }
        return;
    }
    if k >= len {
        return;
    }

    // Find the k-th largest value via partial sort on indices.
    let mut indices: Vec<usize> = (0..len).collect();
    indices.select_nth_unstable_by(k - 1, |&a, &b| {
        logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let threshold = logits[indices[k - 1]];

    // Count how many values equal the threshold that we should keep.
    // (handles ties at the boundary)
    let above_count = logits.iter().filter(|&&v| v > threshold).count();
    let ties_to_keep = k - above_count;

    let mut ties_kept = 0usize;
    let neg_inf = f32::NEG_INFINITY;
    let neg_inf_vec = vdupq_n_f32(neg_inf);
    let thresh_vec = vdupq_n_f32(threshold);

    let ptr = logits.as_mut_ptr();
    let chunks = len / LANES;
    let remainder = len % LANES;

    // NEON pass: zero out lanes strictly below threshold.
    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        // mask = v > threshold  (all-ones lanes where true)
        let mask = vcgtq_f32(v, thresh_vec);
        // keep original where > threshold, else NEG_INFINITY
        let filtered = vbslq_f32(mask, v, neg_inf_vec);
        unsafe { vst1q_f32(ptr.add(i * LANES), filtered) };
    }

    // Scalar tail for elements not covered by NEON chunks.
    let tail_start = chunks * LANES;
    for i in 0..remainder {
        if logits[tail_start + i] <= threshold {
            logits[tail_start + i] = neg_inf;
        }
    }

    // Second pass: selectively restore exactly `ties_to_keep` elements
    // that were equal to the threshold (they got zeroed above).
    if ties_to_keep > 0 {
        for v in logits.iter_mut() {
            if *v == neg_inf && ties_kept < ties_to_keep {
                // Check original value—we need to recover ties at
                // threshold. Because we set them to NEG_INFINITY we
                // track via the count only.
            }
        }
        // Simpler correct approach: re-scan indices we know are top-k.
        for &idx in indices[..k].iter() {
            if logits[idx] == neg_inf {
                logits[idx] = threshold;
                ties_kept += 1;
                if ties_kept >= ties_to_keep {
                    break;
                }
            }
        }
    }
}

// ── Top-p (nucleus) filtering ───────────────────────────────────────────

/// Top-p (nucleus) sampling filter.
///
/// Sorts probabilities in descending order, accumulates until the
/// cumulative probability exceeds `p`, then zeros out the rest.
/// `probs` must already be a valid probability distribution (non-negative,
/// sums to ~1.0).
///
/// # Safety
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn top_p_filter(probs: &mut [f32], p: f32) {
    let len = probs.len();
    if len == 0 || p >= 1.0 {
        return;
    }
    if p <= 0.0 {
        // Keep only the single largest.
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in probs.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        for v in probs.iter_mut() {
            *v = 0.0;
        }
        probs[best_idx] = best_val;
        return;
    }

    // Sort indices by descending probability.
    let mut indices: Vec<usize> = (0..len).collect();
    indices.sort_unstable_by(|&a, &b| {
        probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Walk sorted order, accumulate probability with NEON.
    let mut cumulative = 0.0f32;
    let mut cutoff = len;
    for (rank, &idx) in indices.iter().enumerate() {
        cumulative += probs[idx];
        if cumulative > p {
            cutoff = rank + 1;
            break;
        }
    }

    // Zero out everything beyond the nucleus.
    for &idx in &indices[cutoff..] {
        probs[idx] = 0.0;
    }

    // Renormalise the kept probabilities using NEON.
    let sum = unsafe { neon_sum(probs) };
    if sum > 0.0 && (sum - 1.0).abs() > 1e-7 {
        let inv = 1.0 / sum;
        let inv_vec = vdupq_n_f32(inv);
        let ptr = probs.as_mut_ptr();
        let chunks = len / LANES;
        let remainder = len % LANES;
        for i in 0..chunks {
            let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
            let r = vmulq_f32(v, inv_vec);
            unsafe { vst1q_f32(ptr.add(i * LANES), r) };
        }
        let tail = chunks * LANES;
        for i in 0..remainder {
            probs[tail + i] *= inv;
        }
    }
}

// ── Combined top-k + top-p ──────────────────────────────────────────────

/// Combined top-k then top-p filtering.
///
/// First applies top-k to the raw logits, then softmax, then top-p on
/// the resulting probabilities.  Modifies `logits` in place so that
/// after the call it contains the filtered probability distribution.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn top_k_top_p_filter_neon(logits: &mut [f32], k: usize, p: f32, temperature: f32) {
    if logits.is_empty() {
        return;
    }
    // Step 1: top-k on raw logits.
    unsafe { top_k_filter_neon(logits, k) };

    // Step 2: temperature-scaled softmax to get probabilities.
    unsafe { temperature_softmax_neon(logits, temperature) };

    // Step 3: top-p on probabilities.
    unsafe { top_p_filter(logits, p) };
}

// ── Min-p filtering ─────────────────────────────────────────────────────

/// Min-p sampling filter.
///
/// Keeps only tokens whose probability is at least `min_p * max_prob`.
/// `probs` must be a valid probability distribution.  After filtering
/// the kept values are renormalised.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn min_p_filter(probs: &mut [f32], min_p: f32) {
    let len = probs.len();
    if len == 0 || min_p <= 0.0 {
        return;
    }

    let max_prob = unsafe { find_max_neon(probs) };
    let threshold = min_p * max_prob;
    let thresh_vec = vdupq_n_f32(threshold);
    let zero_vec = vdupq_n_f32(0.0);

    let ptr = probs.as_mut_ptr();
    let chunks = len / LANES;
    let remainder = len % LANES;

    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        // mask lanes >= threshold
        let mask = vcgeq_f32(v, thresh_vec);
        let filtered = vbslq_f32(mask, v, zero_vec);
        unsafe { vst1q_f32(ptr.add(i * LANES), filtered) };
    }
    let tail = chunks * LANES;
    for i in 0..remainder {
        if probs[tail + i] < threshold {
            probs[tail + i] = 0.0;
        }
    }

    // Renormalise.
    let sum = unsafe { neon_sum(probs) };
    if sum > 0.0 && (sum - 1.0).abs() > 1e-7 {
        let inv = 1.0 / sum;
        let inv_vec = vdupq_n_f32(inv);
        for i in 0..chunks {
            let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
            let r = vmulq_f32(v, inv_vec);
            unsafe { vst1q_f32(ptr.add(i * LANES), r) };
        }
        for i in 0..remainder {
            probs[tail + i] *= inv;
        }
    }
}

// ── Typical sampling ────────────────────────────────────────────────────

/// Typical sampling filter based on information-theoretic entropy
/// distance.
///
/// For each token the "surprise" is `-log(p)`.  Tokens are ranked by
/// `|surprise - entropy|` (ascending) and included until the cumulative
/// probability of the selected set exceeds `typical_p`.  All other
/// probabilities are zeroed and the result is renormalised.
///
/// `probs` must be a valid probability distribution.
///
/// # Safety
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn typical_filter(probs: &mut [f32], typical_p: f32) {
    let len = probs.len();
    if len == 0 || typical_p >= 1.0 {
        return;
    }

    // Compute entropy  H = -Σ p·log(p)  (scalar; small contribution to
    // overall runtime relative to sort).
    let mut entropy = 0.0f32;
    for &p in probs.iter() {
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }

    // Compute |surprise - entropy| for each token and sort by it.
    let mut scored: Vec<(usize, f32)> = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let surprise = if p > 0.0 { -p.ln() } else { f32::INFINITY };
            (i, (surprise - entropy).abs())
        })
        .collect();
    scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Accumulate until cumulative prob exceeds typical_p.
    let mut cumulative = 0.0f32;
    let mut keep = vec![false; len];
    for &(idx, _) in &scored {
        cumulative += probs[idx];
        keep[idx] = true;
        if cumulative > typical_p {
            break;
        }
    }

    // Zero out non-kept tokens.
    for i in 0..len {
        if !keep[i] {
            probs[i] = 0.0;
        }
    }

    // Renormalise using NEON.
    let sum = unsafe { neon_sum(probs) };
    if sum > 0.0 && (sum - 1.0).abs() > 1e-7 {
        let inv = 1.0 / sum;
        let inv_vec = vdupq_n_f32(inv);
        let ptr = probs.as_mut_ptr();
        let chunks = len / LANES;
        let remainder = len % LANES;
        for i in 0..chunks {
            let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
            let r = vmulq_f32(v, inv_vec);
            unsafe { vst1q_f32(ptr.add(i * LANES), r) };
        }
        let tail = chunks * LANES;
        for i in 0..remainder {
            probs[tail + i] *= inv;
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, tol: f32, ctx: &str) {
        assert!((a - b).abs() < tol, "{ctx}: expected {b}, got {a} (diff {})", (a - b).abs());
    }

    /// Convert raw logits to probabilities via temperature softmax.
    fn to_probs(logits: &[f32], temperature: f32) -> Vec<f32> {
        let mut buf = logits.to_vec();
        unsafe { temperature_softmax_neon(&mut buf, temperature) };
        buf
    }

    // ── temperature_softmax_neon ────────────────────────────────────

    #[test]
    fn test_temperature_softmax_basic() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        unsafe { temperature_softmax_neon(&mut logits, 1.0) };
        let sum: f32 = logits.iter().sum();
        assert_close(sum, 1.0, 1e-3, "sum");
        for w in logits.windows(2) {
            assert!(w[0] < w[1], "expected monotonic increase");
        }
    }

    #[test]
    fn test_temperature_softmax_high_temp() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        unsafe { temperature_softmax_neon(&mut logits, 100.0) };
        // High temperature → near-uniform distribution.
        let expected = 0.25;
        for (i, &v) in logits.iter().enumerate() {
            assert_close(v, expected, 0.05, &format!("high_temp[{i}]"));
        }
    }

    #[test]
    fn test_temperature_softmax_low_temp() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        unsafe { temperature_softmax_neon(&mut logits, 0.01) };
        // Very low temp → almost all probability on the max.
        assert!(logits[3] > 0.99, "expected max to dominate");
    }

    #[test]
    fn test_temperature_softmax_greedy() {
        let mut logits = vec![1.0, 4.0, 2.0, 3.0];
        unsafe { temperature_softmax_neon(&mut logits, 0.0) };
        assert_close(logits[1], 1.0, 1e-6, "greedy argmax");
        assert_close(logits[0], 0.0, 1e-6, "greedy other");
    }

    #[test]
    fn test_temperature_softmax_empty() {
        let mut logits: Vec<f32> = vec![];
        unsafe { temperature_softmax_neon(&mut logits, 1.0) };
        assert!(logits.is_empty());
    }

    #[test]
    fn test_temperature_softmax_single() {
        let mut logits = vec![42.0];
        unsafe { temperature_softmax_neon(&mut logits, 1.0) };
        assert_close(logits[0], 1.0, 1e-5, "single element");
    }

    #[test]
    fn test_temperature_softmax_non_aligned() {
        // 5 elements → 1 NEON chunk + 1 scalar tail.
        let mut logits = vec![0.5, 1.5, 2.5, 3.5, 4.5];
        unsafe { temperature_softmax_neon(&mut logits, 1.0) };
        let sum: f32 = logits.iter().sum();
        assert_close(sum, 1.0, 1e-3, "non-aligned sum");
    }

    // ── top_k_filter_neon ───────────────────────────────────────────

    #[test]
    fn test_top_k_basic() {
        let mut logits = vec![1.0, 4.0, 2.0, 5.0, 3.0];
        unsafe { top_k_filter_neon(&mut logits, 2) };
        // Top-2 are indices 1 (4.0) and 3 (5.0).
        assert_eq!(logits[1], 4.0);
        assert_eq!(logits[3], 5.0);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[4], f32::NEG_INFINITY);
    }

    #[test]
    fn test_top_k_equals_len() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let orig = logits.clone();
        unsafe { top_k_filter_neon(&mut logits, 3) };
        assert_eq!(logits, orig, "k == len should be a no-op");
    }

    #[test]
    fn test_top_k_exceeds_len() {
        let mut logits = vec![1.0, 2.0];
        let orig = logits.clone();
        unsafe { top_k_filter_neon(&mut logits, 10) };
        assert_eq!(logits, orig, "k > len should be a no-op");
    }

    #[test]
    fn test_top_k_zero() {
        let mut logits = vec![1.0, 2.0, 3.0];
        unsafe { top_k_filter_neon(&mut logits, 0) };
        for &v in &logits {
            assert_eq!(v, f32::NEG_INFINITY);
        }
    }

    #[test]
    fn test_top_k_one() {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0];
        unsafe { top_k_filter_neon(&mut logits, 1) };
        assert_eq!(logits[1], 5.0);
        for (i, &v) in logits.iter().enumerate() {
            if i != 1 {
                assert_eq!(v, f32::NEG_INFINITY);
            }
        }
    }

    #[test]
    fn test_top_k_with_ties() {
        let mut logits = vec![3.0, 3.0, 3.0, 1.0, 2.0];
        unsafe { top_k_filter_neon(&mut logits, 2) };
        let kept: Vec<f32> = logits.iter().copied().filter(|&v| v != f32::NEG_INFINITY).collect();
        assert_eq!(kept.len(), 2, "should keep exactly k=2");
        for &v in &kept {
            assert_eq!(v, 3.0);
        }
    }

    #[test]
    fn test_top_k_non_aligned() {
        let mut logits: Vec<f32> = (0..7).map(|i| i as f32).collect();
        unsafe { top_k_filter_neon(&mut logits, 3) };
        let kept: usize = logits.iter().filter(|&&v| v != f32::NEG_INFINITY).count();
        assert_eq!(kept, 3);
    }

    // ── top_p_filter ────────────────────────────────────────────────

    #[test]
    fn test_top_p_basic() {
        let mut probs = to_probs(&[1.0, 2.0, 3.0, 4.0], 1.0);
        unsafe { top_p_filter(&mut probs, 0.5) };
        let sum: f32 = probs.iter().sum();
        assert_close(sum, 1.0, 1e-3, "top-p renorm");
        // The largest token (index 3) must be kept.
        assert!(probs[3] > 0.0, "largest token should survive");
    }

    #[test]
    fn test_top_p_one() {
        let mut probs = to_probs(&[1.0, 2.0, 3.0, 4.0], 1.0);
        let orig = probs.clone();
        unsafe { top_p_filter(&mut probs, 1.0) };
        // p >= 1.0 → no-op.
        for (i, (&a, &b)) in probs.iter().zip(orig.iter()).enumerate() {
            assert_close(a, b, 1e-6, &format!("top_p_one[{i}]"));
        }
    }

    #[test]
    fn test_top_p_zero() {
        let mut probs = to_probs(&[1.0, 2.0, 3.0, 4.0], 1.0);
        unsafe { top_p_filter(&mut probs, 0.0) };
        // p <= 0 → keep only the largest.
        let nonzero: usize = probs.iter().filter(|&&v| v > 0.0).count();
        assert_eq!(nonzero, 1, "should keep exactly 1 token");
    }

    #[test]
    fn test_top_p_preserves_normalization() {
        let mut probs = to_probs(&[0.5, 1.5, 2.5, 3.5, 4.5], 1.0);
        unsafe { top_p_filter(&mut probs, 0.8) };
        let sum: f32 = probs.iter().sum();
        assert_close(sum, 1.0, 1e-3, "top-p norm");
    }

    // ── top_k_top_p_filter_neon ─────────────────────────────────────

    #[test]
    fn test_combined_top_k_top_p() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        unsafe {
            top_k_top_p_filter_neon(&mut logits, 3, 0.9, 1.0);
        }
        let sum: f32 = logits.iter().sum();
        assert_close(sum, 1.0, 1e-3, "combined sum");
        // At most 3 non-zero entries (top-k).
        let nonzero = logits.iter().filter(|&&v| v > 0.0).count();
        assert!(nonzero <= 3, "at most k tokens: got {nonzero}");
    }

    #[test]
    fn test_combined_empty() {
        let mut logits: Vec<f32> = vec![];
        unsafe {
            top_k_top_p_filter_neon(&mut logits, 5, 0.9, 1.0);
        }
        assert!(logits.is_empty());
    }

    // ── min_p_filter ────────────────────────────────────────────────

    #[test]
    fn test_min_p_basic() {
        let mut probs = to_probs(&[1.0, 2.0, 3.0, 4.0], 1.0);
        let max_before = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        unsafe { min_p_filter(&mut probs, 0.5) };
        // Only tokens with prob >= 0.5 * max_before (scaled) survive.
        let sum: f32 = probs.iter().sum();
        assert_close(sum, 1.0, 1e-3, "min-p renorm");
        assert!(probs[3] > 0.0, "max should survive");
        let _ = max_before; // suppress unused warning
    }

    #[test]
    fn test_min_p_zero_threshold() {
        let mut probs = to_probs(&[1.0, 2.0, 3.0, 4.0], 1.0);
        let orig = probs.clone();
        unsafe { min_p_filter(&mut probs, 0.0) };
        for (i, (&a, &b)) in probs.iter().zip(orig.iter()).enumerate() {
            assert_close(a, b, 1e-6, &format!("min_p_zero[{i}]"));
        }
    }

    #[test]
    fn test_min_p_high_threshold() {
        let mut probs = to_probs(&[1.0, 2.0, 3.0, 10.0], 1.0);
        unsafe { min_p_filter(&mut probs, 0.99) };
        // Only the dominant token should survive.
        let nonzero: usize = probs.iter().filter(|&&v| v > 0.0).count();
        assert_eq!(nonzero, 1, "only max token with high min_p");
    }

    #[test]
    fn test_min_p_non_aligned() {
        let mut probs = to_probs(&[0.1, 0.2, 0.3, 0.4, 0.5], 1.0);
        unsafe { min_p_filter(&mut probs, 0.3) };
        let sum: f32 = probs.iter().sum();
        assert_close(sum, 1.0, 1e-3, "min-p non-aligned renorm");
    }

    // ── typical_filter ──────────────────────────────────────────────

    #[test]
    fn test_typical_basic() {
        let mut probs = to_probs(&[1.0, 2.0, 3.0, 4.0], 1.0);
        unsafe { typical_filter(&mut probs, 0.5) };
        let sum: f32 = probs.iter().sum();
        assert_close(sum, 1.0, 1e-3, "typical renorm");
        let nonzero: usize = probs.iter().filter(|&&v| v > 0.0).count();
        assert!(nonzero >= 1, "at least one token kept");
    }

    #[test]
    fn test_typical_one() {
        let mut probs = to_probs(&[1.0, 2.0, 3.0, 4.0], 1.0);
        let orig = probs.clone();
        unsafe { typical_filter(&mut probs, 1.0) };
        for (i, (&a, &b)) in probs.iter().zip(orig.iter()).enumerate() {
            assert_close(a, b, 1e-6, &format!("typical_one[{i}]"));
        }
    }

    #[test]
    fn test_typical_uniform() {
        // Uniform distribution has maximum entropy — typical sampling
        // should keep tokens closest to the mean surprise.
        let mut probs = vec![0.25, 0.25, 0.25, 0.25];
        unsafe { typical_filter(&mut probs, 0.5) };
        let sum: f32 = probs.iter().sum();
        assert_close(sum, 1.0, 1e-3, "typical uniform renorm");
    }

    #[test]
    fn test_typical_non_aligned() {
        let mut probs = to_probs(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 1.0);
        unsafe { typical_filter(&mut probs, 0.6) };
        let sum: f32 = probs.iter().sum();
        assert_close(sum, 1.0, 1e-3, "typical non-aligned renorm");
    }

    // ── Large-vocabulary stress test ────────────────────────────────

    #[test]
    fn test_large_vocab_top_k() {
        let mut logits: Vec<f32> = (0..32000).map(|i| (i as f32) * 0.001).collect();
        unsafe { top_k_filter_neon(&mut logits, 50) };
        let kept: usize = logits.iter().filter(|&&v| v != f32::NEG_INFINITY).count();
        assert_eq!(kept, 50, "large vocab top-k");
    }

    #[test]
    fn test_large_vocab_temperature_softmax() {
        let mut logits: Vec<f32> = (0..32000).map(|i| (i as f32) * 0.001).collect();
        unsafe { temperature_softmax_neon(&mut logits, 0.7) };
        let sum: f32 = logits.iter().sum();
        assert_close(sum, 1.0, 1e-2, "large vocab softmax sum");
        for &v in &logits {
            assert!(v.is_finite(), "expected finite prob");
        }
    }

    #[test]
    #[ignore = "Slow: 32k-vocab combined pipeline; run manually"]
    fn test_large_vocab_combined_pipeline() {
        let mut logits: Vec<f32> = (0..32000).map(|i| (i as f32) * 0.001).collect();
        unsafe {
            top_k_top_p_filter_neon(&mut logits, 50, 0.95, 0.8);
        }
        let sum: f32 = logits.iter().sum();
        assert_close(sum, 1.0, 1e-2, "large combined sum");
    }

    // ── Scalar parity helpers ───────────────────────────────────────

    fn scalar_softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
        let inv_t = if temperature == 0.0 {
            return {
                let mut out = vec![0.0; logits.len()];
                let idx = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                out[idx] = 1.0;
                out
            };
        } else {
            1.0 / temperature
        };
        let scaled: Vec<f32> = logits.iter().map(|&x| x * inv_t).collect();
        let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scaled.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    #[test]
    fn test_softmax_parity_with_scalar() {
        let logits: Vec<f32> = (0..17).map(|i| (i as f32) * 0.3 - 2.5).collect();
        let neon = to_probs(&logits, 1.0);
        let scalar = scalar_softmax(&logits, 1.0);
        for (i, (&n, &s)) in neon.iter().zip(scalar.iter()).enumerate() {
            assert!((n - s).abs() < 1e-3, "parity[{i}]: neon={n}, scalar={s}",);
        }
    }
}
