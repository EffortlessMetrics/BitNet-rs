//! Task-spec property tests for `bitnet-logits`.
//!
//! These tests verify the core invariants required by the Phase 6 SRP
//! extraction spec using the actual public API:
//! - [`bitnet_logits::softmax_in_place`] (spec: `softmax`)
//! - [`bitnet_logits::apply_top_k`]      (spec: `top_k_filter`)
//! - [`bitnet_logits::apply_temperature`]
//! - [`bitnet_logits::argmax`]

use bitnet_logits::{
    apply_repetition_penalty, apply_temperature, apply_top_k, argmax, softmax_in_place,
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
