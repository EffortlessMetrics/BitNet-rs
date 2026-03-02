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
    // Optimization: Filter out zero probabilities (e.g. from prior top-k)
    // to avoid sorting the entire vocabulary.
    let mut indexed: Vec<(usize, f32)> =
        probs.iter().copied().enumerate().filter(|&(_, p)| p > 0.0).collect();
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
        let v = *l;
        // Optimization: skip exp() for NEG_INFINITY which always yields 0.0.
        // This is extremely common after top_k filtering sets out-of-bounds to NEG_INFINITY.
        if v == f32::NEG_INFINITY {
            *l = 0.0;
        } else {
            let exp = (v - max).exp();
            *l = exp;
            sum += exp;
        }
    }
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for l in logits.iter_mut() {
            *l *= inv_sum;
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

/// Min-p filtering on a **probability** slice (post-softmax).
///
/// Zeroes out all tokens whose probability is below `min_p * max_probability`.
/// This adapts the threshold dynamically based on the most likely token,
/// keeping more tokens when the model is uncertain and fewer when confident.
///
/// `min_p` should be in `[0.0, 1.0]`. Values ≤ 0.0 are no-ops.
///
/// # Examples
///
/// ```
/// use bitnet_logits::apply_min_p;
///
/// let mut probs = vec![0.5f32, 0.3, 0.1, 0.05, 0.05];
/// apply_min_p(&mut probs, 0.2);
/// // Threshold = 0.2 * 0.5 = 0.1. Tokens with prob < 0.1 are zeroed.
/// assert!(probs[0] > 0.0); // 0.5 >= 0.1
/// assert!(probs[1] > 0.0); // 0.3 >= 0.1
/// assert!(probs[2] > 0.0); // 0.1 >= 0.1
/// assert_eq!(probs[3], 0.0); // 0.05 < 0.1
/// assert_eq!(probs[4], 0.0); // 0.05 < 0.1
/// ```
pub fn apply_min_p(probs: &mut [f32], min_p: f32) {
    if min_p <= 0.0 || probs.is_empty() {
        return;
    }
    let max_prob = probs.iter().copied().fold(0.0f32, f32::max);
    let threshold = min_p * max_prob;
    for p in probs.iter_mut() {
        if *p < threshold {
            *p = 0.0;
        }
    }
}

/// Locally typical sampling filter on a **probability** slice (post-softmax).
///
/// Keeps tokens whose "surprise" (negative log probability) is closest to
/// the expected surprise (entropy), until the cumulative probability of
/// kept tokens reaches `typical_p`. This prefers tokens that are
/// information-theoretically "typical" of the distribution.
///
/// `typical_p` should be in `(0.0, 1.0]`. Values ≥ 1.0 are no-ops.
///
/// # Examples
///
/// ```
/// use bitnet_logits::{softmax_in_place, apply_typical};
///
/// let mut probs = vec![0.0f32; 5];
/// probs[0] = 0.5;
/// probs[1] = 0.25;
/// probs[2] = 0.15;
/// probs[3] = 0.07;
/// probs[4] = 0.03;
/// apply_typical(&mut probs, 0.8);
/// // Tokens closest to the entropy are kept first.
/// let non_zero = probs.iter().filter(|&&p| p > 0.0).count();
/// assert!(non_zero >= 1);
/// ```
pub fn apply_typical(probs: &mut [f32], typical_p: f32) {
    if typical_p >= 1.0 || probs.is_empty() {
        return;
    }

    // Compute entropy H = -Σ p * ln(p)
    let entropy: f32 = probs.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();

    // For each token, compute |surprise - entropy| = |(-ln(p)) - H|
    let mut indexed: Vec<(usize, f32, f32)> = probs
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, p)| p > 0.0)
        .map(|(i, p)| {
            let surprise = -p.ln();
            let deviation = (surprise - entropy).abs();
            (i, p, deviation)
        })
        .collect();

    // Sort by deviation ascending (most typical first)
    indexed.sort_unstable_by(|a, b| f32_ascending(a.2, b.2));

    // Keep tokens until cumulative probability reaches typical_p
    let mut cumsum = 0.0f32;
    let mut keep = std::collections::HashSet::new();
    for &(idx, p, _) in &indexed {
        keep.insert(idx);
        cumsum += p;
        if cumsum >= typical_p {
            break;
        }
    }

    // Zero out tokens not in the keep set
    for (i, p) in probs.iter_mut().enumerate() {
        if !keep.contains(&i) {
            *p = 0.0;
        }
    }
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

    #[test]
    fn min_p_filters_below_threshold() {
        let mut probs = vec![0.5f32, 0.3, 0.1, 0.05, 0.05];
        apply_min_p(&mut probs, 0.2);
        // Threshold = 0.2 * 0.5 = 0.1
        assert!(probs[0] > 0.0);
        assert!(probs[1] > 0.0);
        assert!(probs[2] > 0.0); // 0.1 >= 0.1
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(probs[3], 0.0);
            assert_eq!(probs[4], 0.0);
        }
    }

    #[test]
    fn min_p_zero_is_noop() {
        let original = vec![0.5f32, 0.3, 0.2];
        let mut probs = original.clone();
        apply_min_p(&mut probs, 0.0);
        assert_eq!(probs, original);
    }

    #[test]
    fn min_p_one_keeps_only_max() {
        let mut probs = vec![0.5f32, 0.3, 0.2];
        apply_min_p(&mut probs, 1.0);
        // Threshold = 1.0 * 0.5 = 0.5. Only token with prob >= 0.5 survives.
        assert!(probs[0] > 0.0);
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(probs[1], 0.0);
            assert_eq!(probs[2], 0.0);
        }
    }

    #[test]
    fn typical_filters_atypical_tokens() {
        let mut probs = vec![0.5f32, 0.25, 0.15, 0.07, 0.03];
        apply_typical(&mut probs, 0.5);
        // At least one token must survive
        let non_zero = probs.iter().filter(|&&p| p > 0.0).count();
        assert!(non_zero >= 1);
        // Not all tokens should survive with typical_p = 0.5
        assert!(non_zero < 5);
    }

    #[test]
    fn typical_one_is_noop() {
        let original = vec![0.5f32, 0.3, 0.2];
        let mut probs = original.clone();
        apply_typical(&mut probs, 1.0);
        assert_eq!(probs, original);
    }

    #[test]
    fn typical_preserves_sum_bound() {
        let mut probs = vec![0.4f32, 0.3, 0.2, 0.1];
        apply_typical(&mut probs, 0.8);
        let sum: f32 = probs.iter().sum();
        // Remaining sum must be > 0
        assert!(sum > 0.0);
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

        #[test]
        fn min_p_never_removes_max_token(
            probs in proptest::collection::vec(0.01f32..1.0f32, 2..32),
            min_p in 0.0f32..1.0f32,
        ) {
            let max_idx = probs.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i).unwrap();
            let mut filtered = probs;
            apply_min_p(&mut filtered, min_p);
            proptest::prop_assert!(filtered[max_idx] > 0.0,
                "min-p should never remove the highest-probability token");
        }

        #[test]
        fn typical_keeps_at_least_one_token(
            vals in proptest::collection::vec(0.01f32..1.0f32, 2..32),
            typical_p in 0.01f32..0.99f32,
        ) {
            // Normalize to valid distribution
            let sum: f32 = vals.iter().sum();
            let mut probs: Vec<f32> = vals.iter().map(|&v| v / sum).collect();
            apply_typical(&mut probs, typical_p);
            let non_zero = probs.iter().filter(|&&p| p > 0.0).count();
            proptest::prop_assert!(non_zero >= 1, "typical sampling must keep at least one token");
        }
    }
}
