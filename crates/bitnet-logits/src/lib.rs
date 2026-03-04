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

pub use bitnet_logits_filters::{apply_min_p, apply_top_k, apply_top_p, apply_typical};

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
    // Optimization: Pre-calculate 1.0 / penalty outside the loop to replace division with multiplication,
    // as strict IEEE-754 compliance prevents the Rust compiler from doing this automatically.
    let inv_penalty = 1.0 / penalty;
    for &id in token_ids {
        let idx = id as usize;
        // Optimization: Use get_mut to eliminate redundant bounds checking
        if let Some(logit) = logits.get_mut(idx) {
            if *logit > 0.0 {
                *logit *= inv_penalty;
            } else {
                *logit *= penalty;
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
