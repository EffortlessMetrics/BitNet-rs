//! Property-based tests for `bitnet-logits`.
//!
//! Key invariants tested:
//! - `softmax_in_place`: output sums to ≈1.0, all values ≥ 0
//! - `apply_temperature`: scaling is monotone-preserving (order unchanged)
//! - `apply_top_k`: exactly `k` non-NEG_INFINITY elements remain
//! - `argmax`: always returns the index of the maximum element
//! - `apply_repetition_penalty`: penalised tokens are not larger than unpenalised

use bitnet_logits::{apply_repetition_penalty, apply_temperature, apply_top_k, argmax, softmax_in_place};
use proptest::prelude::*;

// ── helpers ───────────────────────────────────────────────────────────────

fn finite_logits(min: f32, max: f32, len_range: std::ops::Range<usize>) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(min..max, len_range)
        .prop_filter("must have at least one finite value", |v| v.iter().any(|x| x.is_finite()))
}

// ── softmax ───────────────────────────────────────────────────────────────

proptest! {
    /// Softmax output sums to approximately 1.0 for any finite input.
    #[test]
    fn softmax_sums_to_one(logits in finite_logits(-20.0, 20.0, 1..200)) {
        let mut probs = logits.clone();
        softmax_in_place(&mut probs);
        let total: f32 = probs.iter().sum();
        prop_assert!((total - 1.0).abs() < 1e-4, "Softmax sum = {total}, expected ≈1.0");
    }

    /// All softmax outputs are non-negative.
    #[test]
    fn softmax_all_non_negative(logits in finite_logits(-20.0, 20.0, 1..200)) {
        let mut probs = logits.clone();
        softmax_in_place(&mut probs);
        for &p in &probs {
            prop_assert!(p >= 0.0, "Softmax produced negative probability {p}");
        }
    }

    /// Softmax preserves the relative order: the argmax of logits equals the argmax of probs.
    #[test]
    fn softmax_preserves_argmax(logits in finite_logits(-10.0, 10.0, 2..100)) {
        // Only valid when there is a unique maximum.
        let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let max_count = logits.iter().filter(|&&x| x == max_val).count();
        prop_assume!(max_count == 1);

        let pre_argmax = argmax(&logits);
        let mut probs = logits.clone();
        softmax_in_place(&mut probs);
        let post_argmax = argmax(&probs);
        prop_assert_eq!(pre_argmax, post_argmax, "Argmax changed after softmax");
    }
}

// ── apply_temperature ─────────────────────────────────────────────────────

proptest! {
    /// Temperature=1.0 is a strict no-op.
    #[test]
    fn temperature_one_is_noop(logits in finite_logits(-10.0, 10.0, 1..50)) {
        let original = logits.clone();
        let mut l = logits;
        apply_temperature(&mut l, 1.0);
        prop_assert_eq!(l, original, "Temperature=1.0 mutated the slice");
    }

    /// Temperature scaling preserves ordinal order (argmax is unchanged).
    #[test]
    fn temperature_preserves_argmax(
        logits in finite_logits(-5.0, 5.0, 2..50),
        temperature in 0.01f32..5.0f32
    ) {
        let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let max_count = logits.iter().filter(|&&x| x == max_val).count();
        prop_assume!(max_count == 1);

        let expected = argmax(&logits);
        let mut l = logits;
        apply_temperature(&mut l, temperature);
        let actual = argmax(&l);
        prop_assert_eq!(expected, actual, "Temperature changed argmax");
    }
}

// ── apply_top_k ───────────────────────────────────────────────────────────

proptest! {
    /// After top-k filtering, at most `k` elements are non-NEG_INFINITY.
    #[test]
    fn top_k_at_most_k_remain(
        logits in finite_logits(-5.0, 5.0, 2..100),
        k in 1usize..50
    ) {
        let k_capped = k.min(logits.len());
        let mut l = logits;
        apply_top_k(&mut l, k_capped);
        let kept = l.iter().filter(|&&x| x != f32::NEG_INFINITY).count();
        prop_assert!(kept <= k_capped, "kept={kept} > top_k={k_capped}");
    }

    /// Top-k with `k=0` (disabled) leaves the slice unchanged.
    #[test]
    fn top_k_zero_is_noop(logits in finite_logits(-5.0, 5.0, 2..50)) {
        let original = logits.clone();
        let mut l = logits;
        apply_top_k(&mut l, 0);
        prop_assert_eq!(l, original, "top_k=0 mutated the slice");
    }

    /// Top-k with `k >= len` leaves the slice unchanged.
    #[test]
    fn top_k_ge_len_is_noop(logits in finite_logits(-5.0, 5.0, 2..50)) {
        let original = logits.clone();
        let mut l = logits;
        let len = l.len();
        apply_top_k(&mut l, len + 10);
        prop_assert_eq!(l, original, "top_k >= len mutated the slice");
    }
}

// ── argmax ────────────────────────────────────────────────────────────────

proptest! {
    /// argmax returns the index of the maximum value.
    #[test]
    fn argmax_returns_max_index(logits in finite_logits(-10.0, 10.0, 1..100)) {
        let idx = argmax(&logits);
        let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        prop_assert_eq!(logits[idx], max_val, "argmax index does not point to maximum");
    }

    /// argmax result is always a valid index.
    #[test]
    fn argmax_index_in_range(logits in finite_logits(-10.0, 10.0, 1..200)) {
        let idx = argmax(&logits);
        prop_assert!(idx < logits.len(), "argmax={idx} >= len={}", logits.len());
    }
}

// ── apply_repetition_penalty ──────────────────────────────────────────────

proptest! {
    /// Repetition penalty with factor=1.0 is a strict no-op.
    #[test]
    fn repetition_penalty_one_is_noop(
        logits in finite_logits(-5.0, 5.0, 2..50),
        token_ids in prop::collection::vec(0u32..50u32, 1..5)
    ) {
        let original = logits.clone();
        let mut l = logits;
        apply_repetition_penalty(&mut l, &token_ids, 1.0);
        prop_assert_eq!(l, original, "Penalty=1.0 mutated the slice");
    }

    /// A penalised positive logit must be ≤ the original positive logit.
    #[test]
    fn repetition_penalty_reduces_positive_logits(
        base_logit in 0.1f32..10.0f32,
        penalty in 1.01f32..3.0f32,
        vocab_size in 2usize..50usize
    ) {
        prop_assume!(vocab_size >= 1);
        let mut logits = vec![0.0f32; vocab_size];
        logits[0] = base_logit;
        let original = logits[0];
        apply_repetition_penalty(&mut logits, &[0u32], penalty);
        prop_assert!(
            logits[0] <= original,
            "Penalty={penalty} increased positive logit: {original} → {}",
            logits[0]
        );
    }

    /// A penalised negative logit must be ≤ (more negative than) the original.
    #[test]
    fn repetition_penalty_worsens_negative_logits(
        base_logit in -10.0f32..-0.1f32,
        penalty in 1.01f32..3.0f32,
        vocab_size in 2usize..50usize
    ) {
        prop_assume!(vocab_size >= 1);
        let mut logits = vec![0.0f32; vocab_size];
        logits[0] = base_logit;
        let original = logits[0];
        apply_repetition_penalty(&mut logits, &[0u32], penalty);
        prop_assert!(
            logits[0] <= original,
            "Penalty={penalty} improved negative logit: {original} → {}",
            logits[0]
        );
    }
}
