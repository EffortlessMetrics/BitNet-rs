//! ARM NEON token sampling operations for Apple Silicon.
//!
//! Provides NEON-accelerated kernels for LLM token sampling:
//! argmax, top-k selection, temperature-scaled softmax, repetition
//! penalty, and nucleus (top-p) threshold computation.

use std::arch::aarch64::*;

/// Lane count for `float32x4_t` NEON vectors.
const LANES: usize = 4;

// ── Fast exp approximation ──────────────────────────────────────────────

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
#[target_feature(enable = "neon")]
#[inline]
unsafe fn fast_exp_neon(x: float32x4_t) -> float32x4_t {
    unsafe {
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
}

// ── argmax ──────────────────────────────────────────────────────────────

/// Find the index of the maximum value in `logits` using NEON parallel
/// max reduction. Returns `0` for empty slices. When multiple elements
/// share the maximum value the index of the first occurrence is returned.
///
/// Uses NEON `vmaxq_f32` for 4-wide chunk reduction, then a scalar pass
/// to identify the winning lane.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
#[target_feature(enable = "neon")]
pub unsafe fn neon_argmax_f32(logits: &[f32]) -> usize {
    let len = logits.len();
    if len == 0 {
        return 0;
    }

    let ptr = logits.as_ptr();
    let chunks = len / LANES;
    let remainder = len % LANES;

    // Phase 1: find the global max value using NEON.
    let mut acc = unsafe { vdupq_n_f32(f32::NEG_INFINITY) };
    for i in 0..chunks {
        unsafe {
            let v = vld1q_f32(ptr.add(i * LANES));
            acc = vmaxq_f32(acc, v);
        }
    }
    let mut max_val = unsafe { vmaxvq_f32(acc) };
    for i in 0..remainder {
        let val = unsafe { *ptr.add(chunks * LANES + i) };
        if val > max_val {
            max_val = val;
        }
    }

    // Phase 2: find first index matching max_val.
    let max_vec = unsafe { vdupq_n_f32(max_val) };
    for i in 0..chunks {
        unsafe {
            let v = vld1q_f32(ptr.add(i * LANES));
            let cmp = vceqq_f32(v, max_vec);
            let mask: [u32; 4] = std::mem::transmute(cmp);
            for (lane, &m) in mask.iter().enumerate() {
                if m != 0 {
                    return i * LANES + lane;
                }
            }
        }
    }

    // Check remainder.
    for i in 0..remainder {
        if unsafe { *ptr.add(chunks * LANES + i) } == max_val {
            return chunks * LANES + i;
        }
    }

    0
}

// ── top-k selection ─────────────────────────────────────────────────────

/// Entry in the top-k result: a (value, original_index) pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopKEntry {
    pub value: f32,
    pub index: usize,
}

/// Fast top-k selection using a min-heap with NEON-accelerated initial scan.
///
/// Returns the `k` largest elements from `logits` in descending order by
/// value. If `k >= logits.len()` all elements are returned (sorted
/// descending). Empty input yields an empty result.
///
/// The algorithm:
/// 1. NEON scan to find global max and quickly seed the heap.
/// 2. Maintain a size-k min-heap; for each element, compare against
///    the heap minimum and swap if larger.
/// 3. Final sort of the k-element heap to produce descending output.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
#[target_feature(enable = "neon")]
pub unsafe fn neon_top_k_f32(logits: &[f32], k: usize) -> Vec<TopKEntry> {
    let len = logits.len();
    if len == 0 || k == 0 {
        return Vec::new();
    }

    let k = k.min(len);

    // Build initial min-heap from first k elements.
    let mut heap: Vec<TopKEntry> =
        logits[..k].iter().enumerate().map(|(i, &v)| TopKEntry { value: v, index: i }).collect();
    build_min_heap(&mut heap);

    // Scan remaining elements with NEON comparisons.
    let ptr = logits.as_ptr();
    let remaining_start = k;
    let remaining = len - remaining_start;
    let chunks = remaining / LANES;
    let tail = remaining % LANES;

    for c in 0..chunks {
        let base = remaining_start + c * LANES;
        unsafe {
            let v = vld1q_f32(ptr.add(base));
            let min_val = vdupq_n_f32(heap[0].value);
            let cmp = vcgtq_f32(v, min_val);
            let mask: [u32; 4] = std::mem::transmute(cmp);

            for (lane, &m) in mask.iter().enumerate() {
                if m != 0 {
                    let idx = base + lane;
                    let val = logits[idx];
                    if val > heap[0].value {
                        heap[0] = TopKEntry { value: val, index: idx };
                        sift_down(&mut heap, 0);
                    }
                }
            }
        }
    }

    // Scalar tail.
    let tail_start = remaining_start + chunks * LANES;
    for i in 0..tail {
        let idx = tail_start + i;
        let val = logits[idx];
        if val > heap[0].value {
            heap[0] = TopKEntry { value: val, index: idx };
            sift_down(&mut heap, 0);
        }
    }

    // Sort descending by value.
    heap.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap_or(std::cmp::Ordering::Equal));
    heap
}

/// Build a min-heap in-place (standard Floyd's algorithm).
fn build_min_heap(heap: &mut [TopKEntry]) {
    let n = heap.len();
    if n <= 1 {
        return;
    }
    for i in (0..n / 2).rev() {
        sift_down(heap, i);
    }
}

/// Sift element at `pos` down to restore min-heap property.
fn sift_down(heap: &mut [TopKEntry], mut pos: usize) {
    let n = heap.len();
    loop {
        let left = 2 * pos + 1;
        let right = 2 * pos + 2;
        let mut smallest = pos;

        if left < n && heap[left].value < heap[smallest].value {
            smallest = left;
        }
        if right < n && heap[right].value < heap[smallest].value {
            smallest = right;
        }
        if smallest == pos {
            break;
        }
        heap.swap(pos, smallest);
        pos = smallest;
    }
}

// ── temperature-scaled softmax ──────────────────────────────────────────

/// NEON-accelerated temperature-scaled softmax.
///
/// Computes `softmax(logits / temperature)` into `output`. Temperature
/// must be positive; a temperature of 1.0 reduces to standard softmax.
/// Lower temperatures sharpen the distribution; higher temperatures
/// flatten it.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// * `output.len() < logits.len()`
/// * `temperature <= 0.0`
#[target_feature(enable = "neon")]
pub unsafe fn neon_softmax_temperature(logits: &[f32], temperature: f32, output: &mut [f32]) {
    assert!(temperature > 0.0, "temperature must be positive, got {temperature}");
    let len = logits.len();
    if len == 0 {
        return;
    }
    assert!(output.len() >= len, "output too short");

    let ptr = logits.as_ptr();
    let chunks = len / LANES;
    let remainder = len % LANES;
    let inv_temp = 1.0 / temperature;

    // Pass 1: find max(logits / temperature) for numerical stability.
    let (mut max_acc, inv_temp_vec) =
        unsafe { (vdupq_n_f32(f32::NEG_INFINITY), vdupq_n_f32(inv_temp)) };
    for i in 0..chunks {
        unsafe {
            let v = vld1q_f32(ptr.add(i * LANES));
            let scaled = vmulq_f32(v, inv_temp_vec);
            max_acc = vmaxq_f32(max_acc, scaled);
        }
    }
    let mut max_val = unsafe { vmaxvq_f32(max_acc) };
    for i in 0..remainder {
        let val = unsafe { *ptr.add(chunks * LANES + i) } * inv_temp;
        if val > max_val {
            max_val = val;
        }
    }

    // Pass 2: compute exp(logits/temp - max) and accumulate sum.
    let (max_vec, mut sum_acc) = unsafe { (vdupq_n_f32(max_val), vdupq_n_f32(0.0)) };
    let out_ptr = output.as_mut_ptr();

    for i in 0..chunks {
        unsafe {
            let v = vld1q_f32(ptr.add(i * LANES));
            let scaled = vmulq_f32(v, inv_temp_vec);
            let shifted = vsubq_f32(scaled, max_vec);
            let e = fast_exp_neon(shifted);
            vst1q_f32(out_ptr.add(i * LANES), e);
            sum_acc = vaddq_f32(sum_acc, e);
        }
    }
    let mut sum = unsafe { vaddvq_f32(sum_acc) };
    for i in 0..remainder {
        let idx = chunks * LANES + i;
        let val = unsafe { *ptr.add(idx) } * inv_temp - max_val;
        let e = fast_exp_scalar(val);
        output[idx] = e;
        sum += e;
    }

    // Pass 3: normalise.
    let inv_sum = 1.0 / sum;
    let inv_sum_vec = unsafe { vdupq_n_f32(inv_sum) };
    for i in 0..chunks {
        unsafe {
            let v = vld1q_f32(out_ptr.add(i * LANES));
            let r = vmulq_f32(v, inv_sum_vec);
            vst1q_f32(out_ptr.add(i * LANES), r);
        }
    }
    for i in 0..remainder {
        let idx = chunks * LANES + i;
        output[idx] *= inv_sum;
    }
}

// ── repetition penalty ──────────────────────────────────────────────────

/// Apply repetition penalty to `logits` for the given `token_ids`.
///
/// For each token ID in `token_ids`, the corresponding logit is divided
/// by `penalty` if positive, or multiplied by `penalty` if negative.
/// This follows the convention from Keskar et al. (2019).
///
/// Uses NEON to batch-process groups of 4 token IDs with vectorised
/// compare-and-select.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// * `penalty <= 0.0`
/// * Any token ID in `token_ids` ≥ `logits.len()`
#[target_feature(enable = "neon")]
pub unsafe fn neon_repetition_penalty(logits: &mut [f32], token_ids: &[u32], penalty: f32) {
    assert!(penalty > 0.0, "penalty must be positive, got {penalty}");
    let vocab_size = logits.len();

    let n = token_ids.len();
    let chunks = n / LANES;
    let remainder = n % LANES;
    let ptr = logits.as_mut_ptr();

    let (penalty_vec, inv_penalty_vec, zero_vec) =
        unsafe { (vdupq_n_f32(penalty), vdupq_n_f32(1.0 / penalty), vdupq_n_f32(0.0)) };

    for c in 0..chunks {
        let base = c * LANES;
        let i0 = token_ids[base] as usize;
        let i1 = token_ids[base + 1] as usize;
        let i2 = token_ids[base + 2] as usize;
        let i3 = token_ids[base + 3] as usize;
        assert!(i0 < vocab_size && i1 < vocab_size && i2 < vocab_size && i3 < vocab_size);

        unsafe {
            // Gather the 4 logit values at token_ids[base..base+4].
            let vals = vsetq_lane_f32::<3>(
                *ptr.add(i3),
                vsetq_lane_f32::<2>(
                    *ptr.add(i2),
                    vsetq_lane_f32::<1>(*ptr.add(i1), vdupq_n_f32(*ptr.add(i0))),
                ),
            );

            // positive mask: logit >= 0 → divide by penalty; else multiply.
            let pos_mask = vcgeq_f32(vals, zero_vec);
            let divided = vmulq_f32(vals, inv_penalty_vec);
            let multiplied = vmulq_f32(vals, penalty_vec);
            let result = vbslq_f32(pos_mask, divided, multiplied);

            // Scatter back.
            let out: [f32; 4] = std::mem::transmute(result);
            *ptr.add(i0) = out[0];
            *ptr.add(i1) = out[1];
            *ptr.add(i2) = out[2];
            *ptr.add(i3) = out[3];
        }
    }

    // Scalar tail.
    for i in 0..remainder {
        let idx = token_ids[chunks * LANES + i] as usize;
        assert!(idx < vocab_size, "token_id {idx} out of bounds for vocab size {vocab_size}");
        let val = logits[idx];
        logits[idx] = if val >= 0.0 { val / penalty } else { val * penalty };
    }
}

// ── nucleus (top-p) threshold ───────────────────────────────────────────

/// Compute the nucleus sampling cutoff index from sorted (descending)
/// probabilities.
///
/// Given `sorted_probs` in descending order, returns the smallest index
/// `k` such that `sum(sorted_probs[0..=k]) >= top_p`. This index can
/// then be used to truncate the distribution for nucleus sampling.
///
/// Uses NEON for 4-wide prefix-sum accumulation.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// * `top_p` not in `(0.0, 1.0]`
#[target_feature(enable = "neon")]
pub unsafe fn neon_nucleus_sampling_threshold(sorted_probs: &[f32], top_p: f32) -> usize {
    assert!(top_p > 0.0 && top_p <= 1.0, "top_p must be in (0.0, 1.0], got {top_p}");

    let len = sorted_probs.len();
    if len == 0 {
        return 0;
    }

    let ptr = sorted_probs.as_ptr();
    let chunks = len / LANES;
    let remainder = len % LANES;
    let mut cumulative = 0.0f32;

    // NEON-accelerated chunk scan: sum each 4-element group and check.
    for c in 0..chunks {
        let (v, chunk_sum) = unsafe {
            let v = vld1q_f32(ptr.add(c * LANES));
            (v, vaddvq_f32(v))
        };

        // Check if threshold is reached within this chunk.
        if cumulative + chunk_sum >= top_p {
            let vals: [f32; 4] = unsafe { std::mem::transmute(v) };
            for (lane, &val) in vals.iter().enumerate() {
                cumulative += val;
                if cumulative >= top_p {
                    return c * LANES + lane;
                }
            }
        }
        cumulative += chunk_sum;
    }

    // Scalar tail.
    for i in 0..remainder {
        cumulative += unsafe { *ptr.add(chunks * LANES + i) };
        if cumulative >= top_p {
            return chunks * LANES + i;
        }
    }

    // If rounding prevents reaching top_p, return last index.
    len.saturating_sub(1)
}

// ── Scalar reference implementations (test parity) ──────────────────────

/// Scalar argmax for parity testing.
pub fn scalar_argmax_f32(logits: &[f32]) -> usize {
    if logits.is_empty() {
        return 0;
    }
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best
}

/// Scalar temperature-scaled softmax for parity testing.
pub fn scalar_softmax_temperature(logits: &[f32], temperature: f32, output: &mut [f32]) {
    assert!(temperature > 0.0);
    let len = logits.len();
    if len == 0 {
        return;
    }
    let inv_temp = 1.0 / temperature;
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> =
        logits.iter().map(|&x| (x * inv_temp - max_val * inv_temp).exp()).collect();
    let sum: f32 = exps.iter().sum();
    for i in 0..len {
        output[i] = exps[i] / sum;
    }
}

/// Scalar repetition penalty for parity testing.
pub fn scalar_repetition_penalty(logits: &mut [f32], token_ids: &[u32], penalty: f32) {
    for &tid in token_ids {
        let idx = tid as usize;
        let val = logits[idx];
        logits[idx] = if val >= 0.0 { val / penalty } else { val * penalty };
    }
}

/// Scalar nucleus threshold for parity testing.
pub fn scalar_nucleus_threshold(sorted_probs: &[f32], top_p: f32) -> usize {
    let mut cum = 0.0f32;
    for (i, &p) in sorted_probs.iter().enumerate() {
        cum += p;
        if cum >= top_p {
            return i;
        }
    }
    sorted_probs.len().saturating_sub(1)
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, tol: f32, ctx: &str) {
        assert!((a - b).abs() < tol, "{ctx}: expected {b}, got {a} (diff {})", (a - b).abs());
    }

    // ── argmax tests ────────────────────────────────────────────────

    #[test]
    fn test_argmax_basic() {
        let logits = [1.0, 3.0, 2.0, 4.0, 0.5];
        let idx = unsafe { neon_argmax_f32(&logits) };
        assert_eq!(idx, 3);
    }

    #[test]
    fn test_argmax_first_element() {
        let logits = [10.0, 1.0, 2.0, 3.0];
        let idx = unsafe { neon_argmax_f32(&logits) };
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_argmax_duplicate_max_returns_first() {
        let logits = [1.0, 5.0, 5.0, 2.0];
        let idx = unsafe { neon_argmax_f32(&logits) };
        assert_eq!(idx, 1, "should return first occurrence of max");
    }

    #[test]
    fn test_argmax_single_element() {
        let logits = [42.0];
        let idx = unsafe { neon_argmax_f32(&logits) };
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_argmax_empty() {
        let idx = unsafe { neon_argmax_f32(&[]) };
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_argmax_negative_values() {
        let logits = [-5.0, -3.0, -1.0, -4.0, -2.0];
        let idx = unsafe { neon_argmax_f32(&logits) };
        assert_eq!(idx, 2);
    }

    #[test]
    fn test_argmax_non_aligned_length() {
        // 13 elements — not a multiple of 4.
        let logits: Vec<f32> = (0..13).map(|i| (i as f32) * 0.7 - 4.0).collect();
        let neon_idx = unsafe { neon_argmax_f32(&logits) };
        let scalar_idx = scalar_argmax_f32(&logits);
        assert_eq!(neon_idx, scalar_idx);
    }

    #[test]
    fn test_argmax_parity_large() {
        let logits: Vec<f32> = (0..257).map(|i| ((i * 97 + 13) % 500) as f32).collect();
        let neon_idx = unsafe { neon_argmax_f32(&logits) };
        let scalar_idx = scalar_argmax_f32(&logits);
        assert_eq!(neon_idx, scalar_idx);
    }

    // ── top-k tests ─────────────────────────────────────────────────

    #[test]
    fn test_top_k_basic() {
        let logits = [1.0, 5.0, 3.0, 4.0, 2.0];
        let result = unsafe { neon_top_k_f32(&logits, 3) };
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].index, 1); // value 5.0
        assert_eq!(result[1].index, 3); // value 4.0
        assert_eq!(result[2].index, 2); // value 3.0
    }

    #[test]
    fn test_top_k_returns_k_elements() {
        let logits: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let result = unsafe { neon_top_k_f32(&logits, 5) };
        assert_eq!(result.len(), 5);
        // Descending order.
        for w in result.windows(2) {
            assert!(w[0].value >= w[1].value, "expected descending order");
        }
    }

    #[test]
    fn test_top_k_k_larger_than_len() {
        let logits = [1.0, 2.0, 3.0];
        let result = unsafe { neon_top_k_f32(&logits, 10) };
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_top_k_k_zero() {
        let logits = [1.0, 2.0];
        let result = unsafe { neon_top_k_f32(&logits, 0) };
        assert!(result.is_empty());
    }

    #[test]
    fn test_top_k_empty_input() {
        let result = unsafe { neon_top_k_f32(&[], 5) };
        assert!(result.is_empty());
    }

    // ── temperature softmax tests ───────────────────────────────────

    #[test]
    fn test_softmax_temp_unit() {
        let logits = [1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 4];
        unsafe { neon_softmax_temperature(&logits, 1.0, &mut out) };
        let sum: f32 = out.iter().sum();
        assert_close(sum, 1.0, 1e-3, "sum@temp=1");
        for w in out.windows(2) {
            assert!(w[0] < w[1], "expected monotonic increase");
        }
    }

    #[test]
    fn test_softmax_temp_low_sharpens() {
        let logits = [1.0, 2.0, 3.0, 4.0];
        let mut out_low = vec![0.0; 4];
        let mut out_high = vec![0.0; 4];
        unsafe {
            neon_softmax_temperature(&logits, 0.1, &mut out_low);
            neon_softmax_temperature(&logits, 10.0, &mut out_high);
        }
        // Low temperature should concentrate more mass on the max.
        assert!(
            out_low[3] > out_high[3],
            "low temp should sharpen: {} vs {}",
            out_low[3],
            out_high[3]
        );
    }

    #[test]
    fn test_softmax_temp_high_flattens() {
        let logits = [1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 4];
        unsafe { neon_softmax_temperature(&logits, 100.0, &mut out) };
        // Very high temperature → nearly uniform.
        let expected = 0.25;
        for (i, &v) in out.iter().enumerate() {
            assert_close(v, expected, 0.05, &format!("uniform[{i}]"));
        }
    }

    #[test]
    fn test_softmax_temp_parity() {
        let logits: Vec<f32> = (0..17).map(|i| (i as f32) * 0.3 - 2.5).collect();
        let temp = 0.7;
        let mut neon_out = vec![0.0f32; logits.len()];
        let mut scalar_out = vec![0.0f32; logits.len()];
        unsafe { neon_softmax_temperature(&logits, temp, &mut neon_out) };
        scalar_softmax_temperature(&logits, temp, &mut scalar_out);

        for (i, (&n, &s)) in neon_out.iter().zip(scalar_out.iter()).enumerate() {
            assert!(
                (n - s).abs() < 1e-3,
                "parity[{i}]: neon={n}, scalar={s}, diff={}",
                (n - s).abs()
            );
        }
    }

    #[test]
    #[should_panic(expected = "temperature must be positive")]
    fn test_softmax_temp_zero_panics() {
        let logits = [1.0, 2.0];
        let mut out = vec![0.0; 2];
        unsafe { neon_softmax_temperature(&logits, 0.0, &mut out) };
    }

    #[test]
    fn test_softmax_temp_empty() {
        let mut out: Vec<f32> = vec![];
        unsafe { neon_softmax_temperature(&[], 1.0, &mut out) };
        assert!(out.is_empty());
    }

    #[test]
    fn test_softmax_temp_large_values() {
        let logits = [1000.0, 1001.0, 1002.0, 1003.0];
        let mut out = vec![0.0; 4];
        unsafe { neon_softmax_temperature(&logits, 1.0, &mut out) };
        let sum: f32 = out.iter().sum();
        assert_close(sum, 1.0, 1e-3, "large values sum");
        for &v in &out {
            assert!(v.is_finite(), "expected finite, got {v}");
        }
    }

    // ── repetition penalty tests ────────────────────────────────────

    #[test]
    fn test_repetition_penalty_positive_logits() {
        let mut logits = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let tokens = [1, 3]; // penalise indices 1 and 3
        let penalty = 2.0;
        unsafe { neon_repetition_penalty(&mut logits, &tokens, penalty) };
        assert_close(logits[1], 2.0, 1e-6, "4.0/2.0"); // 4/2
        assert_close(logits[3], 4.0, 1e-6, "8.0/2.0"); // 8/2
        // Unaffected logits.
        assert_close(logits[0], 2.0, 1e-6, "unchanged[0]");
        assert_close(logits[2], 6.0, 1e-6, "unchanged[2]");
    }

    #[test]
    fn test_repetition_penalty_negative_logits() {
        let mut logits = vec![-2.0, -4.0, 6.0, 8.0];
        let tokens = [0, 1];
        let penalty = 2.0;
        unsafe { neon_repetition_penalty(&mut logits, &tokens, penalty) };
        // Negative logits are multiplied by penalty (making them more negative).
        assert_close(logits[0], -4.0, 1e-6, "-2.0*2.0");
        assert_close(logits[1], -8.0, 1e-6, "-4.0*2.0");
    }

    #[test]
    fn test_repetition_penalty_parity() {
        let mut neon_logits = vec![1.5, -2.3, 0.0, 4.1, -0.5, 3.2, -1.1, 2.8, 0.9];
        let mut scalar_logits = neon_logits.clone();
        let tokens = [0, 2, 4, 6, 8];
        let penalty = 1.5;
        unsafe { neon_repetition_penalty(&mut neon_logits, &tokens, penalty) };
        scalar_repetition_penalty(&mut scalar_logits, &tokens, penalty);
        for (i, (&n, &s)) in neon_logits.iter().zip(scalar_logits.iter()).enumerate() {
            assert_close(n, s, 1e-5, &format!("rep_penalty parity[{i}]"));
        }
    }

    #[test]
    #[should_panic(expected = "penalty must be positive")]
    fn test_repetition_penalty_zero_panics() {
        let mut logits = vec![1.0];
        unsafe { neon_repetition_penalty(&mut logits, &[0], 0.0) };
    }

    // ── nucleus sampling tests ──────────────────────────────────────

    #[test]
    fn test_nucleus_basic() {
        let probs = [0.5, 0.3, 0.15, 0.05];
        let idx = unsafe { neon_nucleus_sampling_threshold(&probs, 0.9) };
        // cumulative: 0.5, 0.8, 0.95 >= 0.9 → index 2
        assert_eq!(idx, 2);
    }

    #[test]
    fn test_nucleus_top_p_one() {
        let probs = [0.4, 0.3, 0.2, 0.1];
        let idx = unsafe { neon_nucleus_sampling_threshold(&probs, 1.0) };
        assert_eq!(idx, 3, "top_p=1.0 should include all");
    }

    #[test]
    fn test_nucleus_single_element() {
        let probs = [1.0];
        let idx = unsafe { neon_nucleus_sampling_threshold(&probs, 0.5) };
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_nucleus_parity() {
        let probs: Vec<f32> = {
            let mut p: Vec<f32> = (0..20).map(|i| (20 - i) as f32).collect();
            let sum: f32 = p.iter().sum();
            for v in &mut p {
                *v /= sum;
            }
            p
        };
        for &top_p in &[0.5, 0.8, 0.9, 0.95, 1.0] {
            let neon_idx = unsafe { neon_nucleus_sampling_threshold(&probs, top_p) };
            let scalar_idx = scalar_nucleus_threshold(&probs, top_p);
            assert_eq!(
                neon_idx, scalar_idx,
                "nucleus parity at top_p={top_p}: neon={neon_idx}, scalar={scalar_idx}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "top_p must be in")]
    fn test_nucleus_zero_panics() {
        unsafe { neon_nucleus_sampling_threshold(&[0.5, 0.5], 0.0) };
    }

    #[test]
    fn test_nucleus_empty() {
        let idx = unsafe { neon_nucleus_sampling_threshold(&[], 0.9) };
        assert_eq!(idx, 0);
    }

    // ── property-style tests ────────────────────────────────────────

    #[test]
    fn test_softmax_temp_sum_is_one_various_lengths() {
        for len in [1, 3, 4, 5, 8, 15, 16, 33, 64, 100] {
            let logits: Vec<f32> = (0..len).map(|i| (i as f32) * 0.1 - 3.0).collect();
            let mut out = vec![0.0f32; len];
            unsafe { neon_softmax_temperature(&logits, 0.5, &mut out) };
            let sum: f32 = out.iter().sum();
            assert_close(sum, 1.0, 1e-3, &format!("sum@len={len}"));
            for &v in &out {
                assert!(v >= 0.0 && v.is_finite(), "non-negative finite at len={len}");
            }
        }
    }

    #[test]
    fn test_top_k_values_are_largest() {
        let logits: Vec<f32> = (0..50).map(|i| ((i * 37 + 7) % 100) as f32).collect();
        let k = 5;
        let result = unsafe { neon_top_k_f32(&logits, k) };
        let mut sorted_all = logits.clone();
        sorted_all.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let top_5_values: Vec<f32> = sorted_all[..k].to_vec();
        let result_values: Vec<f32> = result.iter().map(|e| e.value).collect();
        assert_eq!(result_values, top_5_values, "top-k values must match sorted top-k");
    }

    #[test]
    fn test_argmax_agrees_with_top_1() {
        let logits = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let argmax_idx = unsafe { neon_argmax_f32(&logits) };
        let top1 = unsafe { neon_top_k_f32(&logits, 1) };
        assert_eq!(argmax_idx, top1[0].index, "argmax and top-1 must agree");
    }
}
