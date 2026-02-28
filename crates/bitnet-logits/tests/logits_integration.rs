//! Integration tests for `bitnet-logits` public API.
//!
//! These tests exercise the complete logits-processing pipeline end-to-end,
//! composing multiple transforms in sequence as they would be used during
//! LLM token sampling.

use bitnet_logits::{
    apply_repetition_penalty, apply_temperature, apply_top_k, apply_top_p, argmax, softmax_in_place,
};

// ── apply_temperature ─────────────────────────────────────────────────────────

/// Temperature below 1.0 sharpens the distribution; above 1.0 flattens it.
/// Verify exact scaling for a variety of temperature values.
#[test]
fn temperature_variety_of_inputs() {
    let input = vec![1.0f32, 2.0, 4.0];

    // temp = 0.5 → multiply each logit by 2.0
    let mut logits = input.to_vec();
    apply_temperature(&mut logits, 0.5);
    assert!((logits[0] - 2.0).abs() < 1e-5, "temp 0.5: 1.0 → 2.0, got {}", logits[0]);
    assert!((logits[1] - 4.0).abs() < 1e-5, "temp 0.5: 2.0 → 4.0, got {}", logits[1]);
    assert!((logits[2] - 8.0).abs() < 1e-5, "temp 0.5: 4.0 → 8.0, got {}", logits[2]);

    // temp = 2.0 → multiply each logit by 0.5
    let mut logits = input.to_vec();
    apply_temperature(&mut logits, 2.0);
    assert!((logits[0] - 0.5).abs() < 1e-5, "temp 2.0: 1.0 → 0.5, got {}", logits[0]);
    assert!((logits[1] - 1.0).abs() < 1e-5, "temp 2.0: 2.0 → 1.0, got {}", logits[1]);
    assert!((logits[2] - 2.0).abs() < 1e-5, "temp 2.0: 4.0 → 2.0, got {}", logits[2]);

    // temp = 10.0 → multiply each logit by 0.1
    let mut logits = vec![10.0f32, 20.0, 30.0];
    apply_temperature(&mut logits, 10.0);
    assert!((logits[0] - 1.0).abs() < 1e-5, "temp 10.0: 10.0 → 1.0, got {}", logits[0]);
    assert!((logits[1] - 2.0).abs() < 1e-5, "temp 10.0: 20.0 → 2.0, got {}", logits[1]);
    assert!((logits[2] - 3.0).abs() < 1e-5, "temp 10.0: 30.0 → 3.0, got {}", logits[2]);
}

/// Temperature 0.0 is a no-op (greedy decoding is handled externally).
#[test]
fn temperature_zero_is_noop() {
    let original = vec![1.0f32, 2.0, 3.0];
    let mut logits = original.clone();
    apply_temperature(&mut logits, 0.0);
    assert_eq!(logits, original);
}

/// Temperature scaling preserves the argmax for a range of temperature values.
#[test]
fn temperature_preserves_argmax_integration() {
    let base = vec![0.1f32, 5.0, 2.0, 0.5, 3.0];
    let best = argmax(&base);
    for &temp in &[0.1f32, 0.5, 0.8, 1.5, 3.0, 10.0] {
        let mut logits = base.clone();
        apply_temperature(&mut logits, temp);
        assert_eq!(argmax(&logits), best, "argmax must be stable after temperature {temp}");
    }
}

// ── softmax ───────────────────────────────────────────────────────────────────

/// Softmax output sums to 1.0 for a variety of input shapes and magnitudes.
#[test]
fn softmax_sums_to_one_variety() {
    let cases: &[&[f32]] = &[
        &[0.0],
        &[1.0, 2.0, 3.0],
        &[100.0, -100.0, 50.0],
        &[1.0f32; 16],
        &[-10.0, -5.0, 0.0, 5.0, 10.0],
    ];
    for &case in cases {
        let mut v = case.to_vec();
        softmax_in_place(&mut v);
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum={sum} for input {case:?}");
    }
}

/// Softmax preserves the relative ordering of logit values.
#[test]
fn softmax_preserves_relative_order() {
    let mut logits = vec![1.0f32, 3.0, 2.0, 0.5];
    softmax_in_place(&mut logits);
    // Original order: logits[1] > logits[2] > logits[0] > logits[3]
    assert!(logits[1] > logits[2], "highest logit must have highest probability");
    assert!(logits[2] > logits[0]);
    assert!(logits[0] > logits[3]);
}

// ── top_k_filter ──────────────────────────────────────────────────────────────

/// After `apply_top_k(k)`, at most k entries must be non-`NEG_INFINITY`.
#[test]
fn top_k_reduces_to_at_most_k_elements() {
    let cases = [(10usize, 3usize), (5, 1), (8, 8), (4, 2)];
    for (n, k) in cases {
        let logits: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut v = logits;
        apply_top_k(&mut v, k);
        let kept = v.iter().filter(|&&x| x != f32::NEG_INFINITY).count();
        assert!(kept <= k, "n={n} k={k}: {kept} elements kept but must be <= {k}");
    }
}

/// The top-k entries retained must be the k largest values from the input.
#[test]
fn top_k_retains_highest_values() {
    // Input: [1, 2, 3, 4, 5] — top 2 should keep 4.0 and 5.0 (indices 3 and 4).
    let mut logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    apply_top_k(&mut logits, 2);
    assert!(logits[4].is_finite(), "5.0 (highest) must be kept");
    assert!(logits[3].is_finite(), "4.0 (second highest) must be kept");
    assert!(logits[0].is_infinite(), "1.0 must be masked to NEG_INFINITY");
    assert!(logits[1].is_infinite(), "2.0 must be masked to NEG_INFINITY");
    assert!(logits[2].is_infinite(), "3.0 must be masked to NEG_INFINITY");
}

/// k ≥ len leaves the slice unchanged.
#[test]
fn top_k_noop_when_k_ge_len() {
    let original = vec![3.0f32, 1.0, 2.0];
    let mut logits = original.clone();
    apply_top_k(&mut logits, 10);
    assert_eq!(logits, original, "k >= len must be a no-op");
}

// ── repetition_penalty ───────────────────────────────────────────────────────

/// After applying repetition penalty + softmax, previously-seen tokens must
/// have lower probability than unseen tokens that started with equal logits.
#[test]
fn repetition_penalty_reduces_seen_token_probability() {
    // All tokens start with equal logits.
    let mut logits = vec![1.0f32; 5];
    let seen_tokens: &[u32] = &[0, 2, 4];
    apply_repetition_penalty(&mut logits, seen_tokens, 2.0);
    softmax_in_place(&mut logits);
    // Unseen tokens (1, 3) must have strictly higher probability than seen (0, 2, 4).
    let unseen_prob = logits[1];
    let seen_prob = logits[0];
    assert!(
        unseen_prob > seen_prob,
        "unseen token prob ({unseen_prob}) must exceed seen token prob ({seen_prob})"
    );
}

/// Penalty 1.0 is a no-op regardless of the token list.
#[test]
fn repetition_penalty_one_never_changes_logits() {
    let original = vec![1.5f32, -0.5, 2.0, 0.0, -3.0];
    let mut logits = original.clone();
    apply_repetition_penalty(&mut logits, &[0, 1, 2, 3, 4], 1.0);
    assert_eq!(logits, original, "penalty=1.0 must be identity");
}

/// Out-of-bounds token IDs are silently ignored without panicking.
#[test]
fn repetition_penalty_ignores_out_of_bounds_ids() {
    let original = vec![1.0f32, 2.0, 3.0];
    let mut logits = original.clone();
    apply_repetition_penalty(&mut logits, &[10, 100, u32::MAX], 2.0);
    assert_eq!(logits, original, "out-of-bounds token ids must leave logits unchanged");
}

// ── full pipeline ─────────────────────────────────────────────────────────────

/// Exercise the full sampling pipeline in sequence:
/// `repetition_penalty` → `temperature` → `top_k` → `softmax` → `top_p`.
///
/// The output must be a valid probability distribution (all values ≥ 0, sum ≈ 1.0).
#[test]
fn full_pipeline_produces_valid_distribution() {
    let mut logits = vec![0.5f32, 3.0, 1.0, -1.0, 2.5, 0.0, -0.5, 1.8];
    let token_history: Vec<u32> = vec![2, 5];

    apply_repetition_penalty(&mut logits, &token_history, 1.3);
    apply_temperature(&mut logits, 0.8);
    apply_top_k(&mut logits, 4);
    softmax_in_place(&mut logits);
    apply_top_p(&mut logits, 0.9);

    for &p in &logits {
        assert!(p >= 0.0, "probability {p} must be non-negative");
    }
    // apply_top_p zeroes low-probability tokens without renormalising, so the sum
    // may be slightly less than 1.0. It must remain in (0, 1].
    let sum: f32 = logits.iter().sum();
    assert!(sum > 0.0 && sum <= 1.0 + 1e-4, "pipeline output sum={sum} must be in (0, 1]");
}
