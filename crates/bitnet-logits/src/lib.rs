//! Pure logits transform functions for LLM text generation.
//!
//! All functions operate in-place on `f32` slices and have no external
//! dependencies – they are pure mathematical transforms suitable for use
//! in `no_std` environments (barring `std::cmp`).
//!
//! ## Typical pipeline
//!
//! ```
//! use bitnet_logits::*;
//!
//! let mut logits = vec![1.0f32, 2.0, 3.0, 0.5];
//! let token_history: Vec<u32> = vec![2];
//!
//! apply_repetition_penalty(&mut logits, &token_history, 1.3);
//! apply_temperature(&mut logits, 0.8);
//! softmax_in_place(&mut logits);
//! apply_top_p(&mut logits, 0.9);
//! let best = argmax(&logits);
//! ```

use std::cmp::Ordering;

/// Scale logits by `1 / temperature`.
///
/// * `temperature == 0.0` → no-op (handled externally via greedy path).
/// * `temperature == 1.0` → no-op (identity scaling).
/// * Values in `(0, 1)` sharpen the distribution (lower entropy).
/// * Values `> 1` flatten it (higher entropy / more randomness).
///
/// # Examples
///
/// ```
/// use bitnet_logits::apply_temperature;
///
/// let mut logits = vec![2.0f32, 4.0, 6.0];
/// apply_temperature(&mut logits, 2.0);
/// // Each logit is multiplied by 1/temperature = 0.5
/// assert!((logits[0] - 1.0).abs() < 1e-6);
/// assert!((logits[1] - 2.0).abs() < 1e-6);
/// assert!((logits[2] - 3.0).abs() < 1e-6);
/// ```
///
/// Temperature `1.0` is a no-op:
///
/// ```
/// use bitnet_logits::apply_temperature;
///
/// let original = vec![1.0f32, 2.0, 3.0];
/// let mut logits = original.clone();
/// apply_temperature(&mut logits, 1.0);
/// assert_eq!(logits, original);
/// ```
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if logits.is_empty() {
        return;
    }
    #[allow(clippy::float_cmp)]
    if temperature == 0.0 || temperature == 1.0 {
        return;
    }
    let inv = 1.0 / temperature;
    for l in logits.iter_mut() {
        *l *= inv;
    }
}

/// Zero out all but the top-`top_k` logits (by value).
///
/// Entries outside the top-k are set to `f32::NEG_INFINITY` so that a
/// subsequent [`softmax_in_place`] maps them to probability `0.0`.
///
/// Returns the number of non-`NEG_INFINITY` entries remaining.
/// If `top_k == 0` or `top_k >= logits.len()`, the slice is unchanged.
///
/// # Examples
///
/// ```
/// use bitnet_logits::{apply_top_k, softmax_in_place};
///
/// let mut logits = vec![1.0f32, 5.0, 3.0, 2.0, 4.0];
/// let kept = apply_top_k(&mut logits, 2);
/// assert_eq!(kept, 2);
/// // Only the two highest values (5.0 at idx 1, 4.0 at idx 4) survive.
/// assert!(logits[1].is_finite());
/// assert!(logits[4].is_finite());
/// assert!(logits[0].is_infinite());
///
/// // After softmax, NEG_INFINITY entries become probability 0.
/// softmax_in_place(&mut logits);
/// assert_eq!(logits[0], 0.0);
/// ```
pub fn apply_top_k(logits: &mut [f32], top_k: usize) -> usize {
    if top_k == 0 || top_k >= logits.len() {
        return logits.len();
    }
    // Use O(N) selection to find the k-th largest threshold.
    let mut vals: Vec<f32> = logits.to_vec();
    // select_nth_unstable_by puts the (len - top_k)-th smallest at index (len-top_k),
    // with all smaller values before it and larger values after it.
    let partition_idx = vals.len() - top_k;
    vals.select_nth_unstable_by(partition_idx, |a, b| f32_ascending(*a, *b));
    let threshold = vals[partition_idx];
    let mut kept = 0usize;
    for l in logits.iter_mut() {
        if *l >= threshold && kept < top_k {
            kept += 1;
        } else {
            *l = f32::NEG_INFINITY;
        }
    }
    kept
}

/// Nucleus (top-p) filtering on a **probability** slice (post-softmax).
///
/// Tokens are ranked by probability (descending). The smallest set whose
/// cumulative probability ≥ `top_p` is kept; all others are zeroed.
///
/// Call [`softmax_in_place`] before this function; call [`apply_top_k`] before
/// softmax if both filters are desired.
///
/// # Examples
///
/// ```
/// use bitnet_logits::apply_top_p;
///
/// // Probs already sum to 1.0 (post-softmax).
/// let mut probs = vec![0.5f32, 0.3, 0.2];
/// apply_top_p(&mut probs, 0.8);
/// // 0.5 + 0.3 = 0.8 ≥ top_p, so only the third token is zeroed.
/// assert!(probs[0] > 0.0);
/// assert!(probs[1] > 0.0);
/// assert_eq!(probs[2], 0.0);
/// ```
pub fn apply_top_p(probs: &mut [f32], top_p: f32) {
    if top_p >= 1.0 || probs.is_empty() {
        return;
    }
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| f32_descending(a.1, b.1));

    let mut cumsum = 0.0f32;
    let mut cutoff = indexed.len();
    for (rank, (_, p)) in indexed.iter().enumerate() {
        cumsum += p;
        if cumsum >= top_p {
            cutoff = rank + 1;
            break;
        }
    }
    for (_, (idx, _)) in indexed.iter().enumerate().skip(cutoff) {
        probs[*idx] = 0.0;
    }
}

/// Convert raw logits to a probability distribution in-place via softmax.
///
/// Uses the numerically-stable "subtract max" form.  `f32::NEG_INFINITY`
/// entries (from [`apply_top_k`]) become `0.0` after exponentiation.
///
/// Falls back to a uniform distribution when all exponentiated values underflow
/// to zero (rare with finite logits).
///
/// # Examples
///
/// ```
/// use bitnet_logits::softmax_in_place;
///
/// let mut logits = vec![1.0f32, 2.0, 3.0];
/// softmax_in_place(&mut logits);
/// let sum: f32 = logits.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-5);
/// // Higher logit → higher probability.
/// assert!(logits[2] > logits[1] && logits[1] > logits[0]);
/// ```
pub fn softmax_in_place(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for l in logits.iter_mut() {
        *l = (*l - max).exp();
        sum += *l;
    }
    if sum > 0.0 {
        for l in logits.iter_mut() {
            *l /= sum;
        }
    } else {
        // Degenerate case: all exponentiated values underflowed to 0.
        // Fall back to a uniform distribution so downstream sampling receives
        // a valid probability distribution.
        #[allow(clippy::cast_precision_loss)]
        let uniform = 1.0_f32 / logits.len() as f32;
        for l in logits.iter_mut() {
            *l = uniform;
        }
    }
}

/// Apply a multiplicative repetition penalty to previously-seen tokens.
///
/// * Positive logits are divided by `penalty` (reduced).
/// * Negative logits are multiplied by `penalty` (made more negative).
/// * `penalty == 1.0` → no-op.
///
/// This applies the same penalty regardless of how many times a token has
/// appeared. For count-proportional penalties use
/// `SamplingStrategy::sample()` from `bitnet-sampling`.
///
/// # Examples
///
/// ```
/// use bitnet_logits::apply_repetition_penalty;
///
/// let mut logits = vec![0.0f32, 2.0, -1.0];
/// apply_repetition_penalty(&mut logits, &[1, 2], 2.0);
/// // Positive logit divided: 2.0 / 2.0 = 1.0
/// assert!((logits[1] - 1.0).abs() < 1e-6);
/// // Negative logit multiplied: -1.0 * 2.0 = -2.0
/// assert!((logits[2] - (-2.0)).abs() < 1e-6);
/// // Unseen token unchanged
/// assert_eq!(logits[0], 0.0);
/// ```
pub fn apply_repetition_penalty(logits: &mut [f32], token_ids: &[u32], penalty: f32) {
    #[allow(clippy::float_cmp)]
    if penalty <= 0.0 || !penalty.is_finite() || penalty == 1.0 || token_ids.is_empty() {
        return;
    }
    for &id in token_ids {
        let idx = id as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Return the index of the maximum value (argmax).
///
/// Ties are broken by returning the **last** maximum found (standard
/// [`Iterator::max_by`] semantics).
/// Returns `0` on an empty slice.
///
/// # Examples
///
/// ```
/// use bitnet_logits::argmax;
///
/// let logits = vec![0.1f32, 0.5, 0.9, 0.2];
/// assert_eq!(argmax(&logits), 2); // 0.9 is at index 2
///
/// // Empty slice returns 0.
/// assert_eq!(argmax(&[]), 0);
/// ```
pub fn argmax(logits: &[f32]) -> usize {
    logits.iter().enumerate().max_by(|(_, a), (_, b)| f32_ascending(**a, **b)).map_or(0, |(i, _)| i)
}

// --- helpers ---------------------------------------------------------------

#[inline]
fn f32_descending(a: f32, b: f32) -> Ordering {
    b.partial_cmp(&a).unwrap_or(Ordering::Equal)
}

#[inline]
fn f32_ascending(a: f32, b: f32) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_scales_logits() {
        let mut logits = vec![2.0f32, 4.0, 6.0];
        apply_temperature(&mut logits, 2.0);
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn temperature_one_is_noop() {
        let original = vec![1.0f32, 2.0, 3.0];
        let mut logits = original.clone();
        apply_temperature(&mut logits, 1.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut logits = vec![1.0f32, 2.0, 3.0, 4.0];
        softmax_in_place(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_preserves_order() {
        let mut logits = vec![1.0f32, 3.0, 2.0];
        softmax_in_place(&mut logits);
        assert!(logits[1] > logits[2]);
        assert!(logits[2] > logits[0]);
    }

    #[test]
    fn argmax_finds_maximum() {
        let logits = vec![0.1f32, 0.5, 0.9, 0.2];
        assert_eq!(argmax(&logits), 2);
    }

    #[test]
    fn argmax_empty_returns_zero() {
        assert_eq!(argmax(&[]), 0);
    }

    #[test]
    fn top_k_keeps_k_largest() {
        let mut logits = vec![1.0f32, 5.0, 3.0, 2.0, 4.0];
        let kept = apply_top_k(&mut logits, 2);
        assert_eq!(kept, 2);
        // Only indices 1 (5.0) and 4 (4.0) should remain finite.
        assert!(logits[1].is_finite());
        assert!(logits[4].is_finite());
        assert!(logits[0].is_infinite());
        assert!(logits[2].is_infinite());
        assert!(logits[3].is_infinite());
    }

    #[test]
    fn top_k_zero_is_noop() {
        let original = vec![1.0f32, 2.0, 3.0];
        let mut logits = original.clone();
        apply_top_k(&mut logits, 0);
        assert_eq!(logits, original);
    }

    #[test]
    fn top_p_removes_low_prob_tokens() {
        // Uniform probs: [0.5, 0.3, 0.2]. top_p=0.8 → keep 0.5+0.3=0.8.
        let mut probs = vec![0.5f32, 0.3, 0.2];
        apply_top_p(&mut probs, 0.8);
        assert!(probs[0] > 0.0);
        assert!(probs[1] > 0.0);
        // apply_top_p explicitly sets excluded tokens to exactly 0.0
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(probs[2], 0.0);
        }
    }

    #[test]
    fn top_p_one_is_noop() {
        let original = vec![0.5f32, 0.3, 0.2];
        let mut probs = original.clone();
        apply_top_p(&mut probs, 1.0);
        assert_eq!(probs, original);
    }

    #[test]
    fn repetition_penalty_reduces_positive_logit() {
        let mut logits = vec![0.0f32, 2.0, -1.0];
        apply_repetition_penalty(&mut logits, &[1], 2.0);
        assert!((logits[1] - 1.0).abs() < 1e-6); // 2.0 / 2.0 = 1.0
    }

    #[test]
    fn repetition_penalty_increases_negative_logit() {
        let mut logits = vec![0.0f32, 2.0, -1.0];
        apply_repetition_penalty(&mut logits, &[2], 2.0);
        assert!((logits[2] - (-2.0)).abs() < 1e-6); // -1.0 * 2.0 = -2.0
    }

    #[test]
    fn repetition_penalty_one_is_noop() {
        let original = vec![1.0f32, 2.0, 3.0];
        let mut logits = original.clone();
        apply_repetition_penalty(&mut logits, &[0, 1, 2], 1.0);
        assert_eq!(logits, original);
    }

    // --- proptest -----------------------------------------------------------

    proptest::proptest! {
        #[test]
        fn softmax_always_sums_to_one(vals in proptest::collection::vec(-100.0f32..100.0f32, 1..50)) {
            let mut logits = vals;
            softmax_in_place(&mut logits);
            let sum: f32 = logits.iter().sum();
            proptest::prop_assert!((sum - 1.0).abs() < 1e-4,
                "softmax sum = {sum}");
        }

        #[test]
        fn temperature_preserves_argmax(
            vals in proptest::collection::vec(0.1f32..10.0f32, 2..20),
            temp in 0.1f32..3.0f32,
        ) {
            let best_before = argmax(&vals);
            let mut logits = vals;
            apply_temperature(&mut logits, temp);
            let best_after = argmax(&logits);
            proptest::prop_assert_eq!(best_before, best_after);
        }
    }
}
