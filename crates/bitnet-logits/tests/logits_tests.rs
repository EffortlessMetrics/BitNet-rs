//! Task-spec property tests for `bitnet-logits`.
//!
//! These tests verify the core invariants required by the Phase 6 SRP
//! extraction spec using the actual public API:
//! - [`bitnet_logits::softmax_in_place`] (spec: `softmax`)
//! - [`bitnet_logits::apply_top_k`]      (spec: `top_k_filter`)
//! - [`bitnet_logits::apply_temperature`]
//! - [`bitnet_logits::argmax`]

use bitnet_logits::{
    apply_repetition_penalty, apply_temperature, apply_top_k, apply_top_p, argmax, softmax_in_place,
};
use proptest::prelude::*;

// ── Spec-required named unit tests ────────────────────────────────────────

/// Output probabilities sum to 1.0 after softmax.
#[test]
fn softmax_output_sums_to_one() {
    let mut logits = vec![1.0f32, 2.0, 3.0];
    softmax_in_place(&mut logits);
    let sum: f32 = logits.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "expected sum≈1.0, got {sum}");
}

/// The highest logit stays the highest after temperature scaling.
#[test]
fn temperature_scaling_preserves_argmax() {
    let logits = vec![0.5f32, 3.0, 1.0];
    let best_before = argmax(&logits);
    let mut scaled = logits.clone();
    apply_temperature(&mut scaled, 0.5);
    assert_eq!(argmax(&scaled), best_before);
}

/// Exactly `n - k` entries become `NEG_INFINITY` after apply_top_k.
#[test]
fn apply_top_k_zeros_correct_count() {
    let mut logits = vec![1.0f32, 4.0, 3.0, 2.0, 5.0];
    let n = logits.len();
    let k = 2;
    apply_top_k(&mut logits, k);
    let inf_count = logits.iter().filter(|&&x| x == f32::NEG_INFINITY).count();
    assert_eq!(
        inf_count,
        n - k,
        "expected {n}-{k}={} NEG_INFINITY entries, got {inf_count}",
        n - k
    );
}

/// All-equal inputs produce a uniform distribution after softmax.
#[test]
fn softmax_handles_all_equal_inputs() {
    let mut logits = vec![1.0f32; 4];
    softmax_in_place(&mut logits);
    for &p in &logits {
        assert!((p - 0.25).abs() < 1e-6, "expected 0.25, got {p}");
    }
}

/// temperature=1.0 is an identity operation.
#[test]
fn apply_temperature_one_is_identity() {
    let original = vec![1.5f32, -0.5, 2.0];
    let mut logits = original.clone();
    apply_temperature(&mut logits, 1.0);
    assert_eq!(logits, original);
}

proptest! {
    /// softmax output must sum to ≈1.0 for any finite input.
    #[test]
    fn test_softmax_sums_to_one(
        logits in prop::collection::vec(-100.0f32..100.0f32, 1..100)
    ) {
        let mut v = logits;
        softmax_in_place(&mut v);
        let sum: f32 = v.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
    }

    /// `apply_temperature(t=1.0)` must be a no-op.
    #[test]
    fn test_temperature_no_change_at_one(
        logits in prop::collection::vec(-100.0f32..100.0f32, 1..100)
    ) {
        let original = logits.clone();
        let mut v = logits;
        apply_temperature(&mut v, 1.0);
        prop_assert_eq!(v, original);
    }

    /// `argmax` must return a valid index and that index must hold the maximum value.
    #[test]
    fn test_argmax_within_bounds(
        logits in prop::collection::vec(-100.0f32..100.0f32, 1..100)
    ) {
        let idx = argmax(&logits);
        prop_assert!(idx < logits.len());
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        prop_assert_eq!(logits[idx], max_val);
    }

/// After `apply_top_k(k)`, at most `k` entries must be non-`NEG_INFINITY`.
    #[test]
    fn test_top_k_leaves_k_nonzero(
        logits in prop::collection::vec(-10.0f32..10.0f32, 5..50),
        k in 1usize..5,
    ) {
        let mut v = logits;
        apply_top_k(&mut v, k);
        let nonzero = v.iter().filter(|&&x| x != f32::NEG_INFINITY).count();
        prop_assert!(nonzero <= k, "nonzero={nonzero} k={k}");
    }
}

// ── apply_top_p tests ────────────────────────────────────────────────────────

/// top_p >= 1.0 is a no-op (keep all tokens).
#[test]
fn apply_top_p_one_is_noop() {
    let original = vec![0.5f32, 0.3, 0.2];
    let mut probs = original.clone();
    apply_top_p(&mut probs, 1.0);
    assert_eq!(probs, original);
}

/// top_p cuts off tokens below the nucleus threshold.
#[test]
fn apply_top_p_removes_low_probability_tokens() {
    // Sorted desc: 0.5, 0.3, 0.2. Cumsum: 0.5 → 0.8 >= 0.8, so token at 0.2 is zeroed.
    let mut probs = vec![0.5f32, 0.3, 0.2];
    apply_top_p(&mut probs, 0.8);
    assert!(probs[0] > 0.0, "first token must survive");
    assert!(probs[1] > 0.0, "second token must survive");
    assert_eq!(probs[2], 0.0, "third token must be zeroed");
}

/// top_p on a single-token distribution keeps that token.
#[test]
fn apply_top_p_single_element_keeps_token() {
    let mut probs = vec![1.0f32];
    apply_top_p(&mut probs, 0.5);
    assert!(probs[0] > 0.0, "sole token must survive");
}

/// top_p with very small threshold keeps at least the highest-probability token.
#[test]
fn apply_top_p_very_small_threshold_keeps_at_least_one() {
    let mut probs = vec![0.6f32, 0.3, 0.1];
    apply_top_p(&mut probs, 0.01);
    let surviving = probs.iter().filter(|&&p| p > 0.0).count();
    assert!(surviving >= 1, "at least one token must survive even with tiny top_p");
}

/// Zeroed entries from prior top_k are correctly handled (excluded from sorting).
#[test]
fn apply_top_p_handles_zeros_from_top_k() {
    // Simulate a post-top_k probability vector where two tokens are non-zero.
    // 0.8 alone exceeds top_p=0.75, so the 0.2 token should be zeroed.
    let mut probs = vec![0.0f32, 0.8, 0.0, 0.2];
    apply_top_p(&mut probs, 0.75);
    assert!(probs[1] > 0.0, "dominant token must survive");
    assert_eq!(probs[3], 0.0, "smaller token must be zeroed");
}

// ── apply_repetition_penalty tests ───────────────────────────────────────────

/// penalty=1.0 is a no-op.
#[test]
fn apply_repetition_penalty_one_is_noop() {
    let original = vec![2.0f32, -1.0, 0.5];
    let mut logits = original.clone();
    apply_repetition_penalty(&mut logits, &[0, 1, 2], 1.0);
    assert_eq!(logits, original);
}

/// Positive logits are divided by the penalty.
#[test]
fn apply_repetition_penalty_divides_positive_logits() {
    let mut logits = vec![0.0f32, 4.0, 0.0];
    apply_repetition_penalty(&mut logits, &[1], 2.0);
    assert!((logits[1] - 2.0).abs() < 1e-6, "4.0 / 2.0 should equal 2.0");
}

/// Negative logits are multiplied by the penalty (pushed further negative).
#[test]
fn apply_repetition_penalty_multiplies_negative_logits() {
    let mut logits = vec![0.0f32, 0.0, -3.0];
    apply_repetition_penalty(&mut logits, &[2], 2.0);
    assert!((logits[2] - (-6.0)).abs() < 1e-6, "-3.0 * 2.0 should equal -6.0");
}

/// Unseen tokens are unaffected.
#[test]
fn apply_repetition_penalty_leaves_unseen_tokens_unchanged() {
    let mut logits = vec![1.0f32, 2.0, 3.0];
    apply_repetition_penalty(&mut logits, &[0], 2.0);
    // Index 1 and 2 are unseen and must not change.
    assert!((logits[1] - 2.0).abs() < 1e-6);
    assert!((logits[2] - 3.0).abs() < 1e-6);
}

/// Out-of-bounds token IDs are silently ignored.
#[test]
fn apply_repetition_penalty_ignores_out_of_bounds_ids() {
    let mut logits = vec![1.0f32, 2.0];
    let original = logits.clone();
    apply_repetition_penalty(&mut logits, &[100, 999], 3.0);
    assert_eq!(logits, original, "out-of-bounds IDs must not mutate logits");
}

/// Empty token history is a no-op.
#[test]
fn apply_repetition_penalty_empty_history_is_noop() {
    let original = vec![1.5f32, -0.5, 0.0];
    let mut logits = original.clone();
    apply_repetition_penalty(&mut logits, &[], 2.0);
    assert_eq!(logits, original);
}

// ── argmax edge cases ────────────────────────────────────────────────────────

/// argmax on a single-element slice returns index 0.
#[test]
fn argmax_single_element() {
    assert_eq!(argmax(&[42.0f32]), 0);
}

/// argmax on two elements returns the index of the larger value.
#[test]
fn argmax_two_elements() {
    assert_eq!(argmax(&[1.0f32, 2.0]), 1);
    assert_eq!(argmax(&[3.0f32, -1.0]), 0);
}

/// argmax with all-equal values returns a valid index.
#[test]
fn argmax_all_equal_returns_valid_index() {
    let logits = vec![5.0f32; 10];
    let idx = argmax(&logits);
    assert!(idx < logits.len(), "argmax must return a valid index for equal-value input");
}

// ── temperature edge cases ────────────────────────────────────────────────────

/// Temperature < 1 sharpens the distribution (max logit becomes relatively larger).
#[test]
fn apply_temperature_below_one_sharpens() {
    let mut logits = vec![1.0f32, 2.0, 3.0];
    let before_spread = logits[2] - logits[0];
    apply_temperature(&mut logits, 0.5);
    let after_spread = logits[2] - logits[0];
    assert!(after_spread > before_spread, "temperature<1 must increase relative spread");
}

/// Temperature > 1 flattens the distribution (spread between logits shrinks).
#[test]
fn apply_temperature_above_one_flattens() {
    let mut logits = vec![1.0f32, 2.0, 3.0];
    let before_spread = logits[2] - logits[0];
    apply_temperature(&mut logits, 2.0);
    let after_spread = logits[2] - logits[0];
    assert!(after_spread < before_spread, "temperature>1 must decrease relative spread");
}

/// Empty slice is a safe no-op for apply_temperature.
#[test]
fn apply_temperature_empty_slice_is_safe() {
    let mut logits: Vec<f32> = vec![];
    apply_temperature(&mut logits, 0.5); // must not panic
}

// ── softmax edge cases ────────────────────────────────────────────────────────

/// Single-element softmax produces [1.0].
#[test]
fn softmax_single_element_is_one() {
    let mut logits = vec![42.0f32];
    softmax_in_place(&mut logits);
    assert!((logits[0] - 1.0).abs() < 1e-6, "single-element softmax must be 1.0");
}

/// Softmax output values are all in [0, 1].
#[test]
fn softmax_output_values_in_unit_interval() {
    let mut logits = vec![-10.0f32, 0.0, 5.0, 1000.0, -1000.0];
    softmax_in_place(&mut logits);
    for &p in &logits {
        assert!(p >= 0.0 && p <= 1.0, "softmax output must be in [0,1], got {p}");
    }
}

/// Softmax with NEG_INFINITY entries produces 0.0 for those positions.
#[test]
fn softmax_neg_infinity_becomes_zero() {
    let mut logits = vec![f32::NEG_INFINITY, 1.0f32, 2.0];
    softmax_in_place(&mut logits);
    assert_eq!(logits[0], 0.0, "NEG_INFINITY must become 0.0 after softmax");
    let sum: f32 = logits.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "sum must still be 1.0");
}

// ── apply_top_k edge cases ────────────────────────────────────────────────────

/// apply_top_k with k=1 retains only the maximum.
#[test]
fn apply_top_k_k_one_retains_max_only() {
    let mut logits = vec![1.0f32, 5.0, 3.0, 2.0];
    apply_top_k(&mut logits, 1);
    let finite_count = logits.iter().filter(|&&x| x.is_finite()).count();
    assert_eq!(finite_count, 1, "k=1 must leave exactly one finite entry");
    // The finite entry must be the original maximum.
    let finite_val = logits.iter().find(|&&x| x.is_finite()).copied().unwrap();
    assert!((finite_val - 5.0).abs() < 1e-6, "the surviving entry must be the maximum");
}

/// apply_top_k with k >= len is a no-op (all entries survive).
#[test]
fn apply_top_k_k_ge_len_is_noop() {
    let original = vec![1.0f32, 2.0, 3.0];
    let mut logits = original.clone();
    apply_top_k(&mut logits, original.len());
    assert_eq!(logits, original, "k>=len must not change the slice");

    let mut logits2 = original.clone();
    apply_top_k(&mut logits2, original.len() + 10);
    assert_eq!(logits2, original, "k>len must not change the slice");
}

proptest! {
    /// top_p >= 1.0 is always a no-op.
    #[test]
    fn test_top_p_one_is_noop(
        probs in prop::collection::vec(0.0f32..1.0f32, 1..20)
    ) {
        let original = probs.clone();
        let mut v = probs;
        apply_top_p(&mut v, 1.0);
        prop_assert_eq!(v, original);
    }

    /// apply_repetition_penalty with penalty=1.0 never changes logits.
    #[test]
    fn test_repetition_penalty_one_is_noop(
        logits in prop::collection::vec(-50.0f32..50.0f32, 1..30),
        ids in prop::collection::vec(0u32..30, 0..10),
    ) {
        let original = logits.clone();
        let mut v = logits;
        apply_repetition_penalty(&mut v, &ids, 1.0);
        prop_assert_eq!(v, original);
    }

    /// softmax output values are always in [0, 1].
    #[test]
    fn test_softmax_output_in_unit_interval(
        logits in prop::collection::vec(-100.0f32..100.0f32, 1..50)
    ) {
        let mut v = logits;
        softmax_in_place(&mut v);
        for &p in &v {
            prop_assert!(p >= 0.0 && p <= 1.0 + 1e-5, "softmax output out of range: {p}");
        }
    }

    /// apply_temperature preserves argmax for any temperature in (0, 1) ∪ (1, ∞).
    #[test]
    fn test_temperature_preserves_argmax_any_temp(
        logits in prop::collection::vec(0.1f32..10.0f32, 2..20),
        temp in prop::sample::select(vec![0.1f32, 0.5, 0.8, 1.5, 2.0, 5.0]),
    ) {
        let best_before = argmax(&logits);
        let mut v = logits;
        apply_temperature(&mut v, temp);
        let best_after = argmax(&v);
        prop_assert_eq!(best_before, best_after, "argmax must be stable across temperature scaling");
    }
}
