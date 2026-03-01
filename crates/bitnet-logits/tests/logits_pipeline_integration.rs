//! Pipeline integration tests for bitnet-logits transforms.
//!
//! These tests exercise the full logits → sampling pipeline as used in
//! multi-SLM inference, covering realistic vocab sizes (32K–256K), edge
//! cases in each transform, and end-to-end pipeline composition.

use bitnet_logits::*;

// ---------------------------------------------------------------------------
// Pipeline: temperature → top_k → softmax → top_p → argmax
// ---------------------------------------------------------------------------

#[test]
fn full_pipeline_deterministic_greedy() {
    let mut logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
    // Greedy: temperature=0 is a no-op, argmax picks highest
    apply_temperature(&mut logits, 0.0);
    assert_eq!(argmax(&logits), 1); // 5.0 at index 1
}

#[test]
fn full_pipeline_with_temperature_and_top_k() {
    let mut logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
    apply_temperature(&mut logits, 0.5); // sharpen
    apply_top_k(&mut logits, 2);
    softmax_in_place(&mut logits);
    // Only top 2 should have probability > 0
    let non_zero: usize = logits.iter().filter(|&&p| p > 0.0).count();
    assert_eq!(non_zero, 2);
    // Sum should be ~1.0
    let sum: f32 = logits.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn full_pipeline_temperature_top_k_softmax_top_p() {
    let mut logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
    apply_temperature(&mut logits, 0.8);
    apply_top_k(&mut logits, 3);
    softmax_in_place(&mut logits);
    apply_top_p(&mut logits, 0.9);
    // After top_p, some additional tokens may be zeroed
    let non_zero: usize = logits.iter().filter(|&&p| p > 0.0).count();
    assert!(non_zero >= 1);
}

#[test]
fn full_pipeline_with_repetition_penalty() {
    let mut logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
    let history = vec![1u32]; // token at index 1 was used
    apply_repetition_penalty(&mut logits, &history, 2.0);
    // logits[1] should be penalized: 5.0 / 2.0 = 2.5
    assert!((logits[1] - 2.5).abs() < 1e-6);
    apply_top_k(&mut logits, 2);
    softmax_in_place(&mut logits);
    // After penalty, index 4 (4.0) should now be highest
    let best = argmax(&logits);
    assert_eq!(best, 4);
}

// ---------------------------------------------------------------------------
// apply_temperature edge cases
// ---------------------------------------------------------------------------

#[test]
fn temperature_zero_is_noop() {
    let original = vec![1.0, 2.0, 3.0];
    let mut logits = original.clone();
    apply_temperature(&mut logits, 0.0);
    assert_eq!(logits, original);
}

#[test]
fn temperature_one_is_noop() {
    let original = vec![1.0, 2.0, 3.0];
    let mut logits = original.clone();
    apply_temperature(&mut logits, 1.0);
    assert_eq!(logits, original);
}

#[test]
fn temperature_low_sharpens() {
    let mut logits = vec![1.0, 2.0, 3.0];
    apply_temperature(&mut logits, 0.5);
    // Each logit *= 1/0.5 = 2
    assert!((logits[0] - 2.0).abs() < 1e-6);
    assert!((logits[1] - 4.0).abs() < 1e-6);
    assert!((logits[2] - 6.0).abs() < 1e-6);
}

#[test]
fn temperature_high_flattens() {
    let mut logits = vec![2.0, 4.0, 6.0];
    apply_temperature(&mut logits, 2.0);
    assert!((logits[0] - 1.0).abs() < 1e-6);
    assert!((logits[1] - 2.0).abs() < 1e-6);
    assert!((logits[2] - 3.0).abs() < 1e-6);
}

#[test]
fn temperature_empty_slice() {
    let mut logits: Vec<f32> = vec![];
    apply_temperature(&mut logits, 0.5);
    assert!(logits.is_empty());
}

// ---------------------------------------------------------------------------
// apply_top_k edge cases
// ---------------------------------------------------------------------------

#[test]
fn top_k_zero_is_noop() {
    let mut logits = vec![1.0, 2.0, 3.0];
    let kept = apply_top_k(&mut logits, 0);
    assert_eq!(kept, 3);
    assert!(logits.iter().all(|l| l.is_finite()));
}

#[test]
fn top_k_exceeds_len_is_noop() {
    let mut logits = vec![1.0, 2.0, 3.0];
    let kept = apply_top_k(&mut logits, 100);
    assert_eq!(kept, 3);
}

#[test]
fn top_k_one_keeps_best() {
    let mut logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
    let kept = apply_top_k(&mut logits, 1);
    assert_eq!(kept, 1);
    assert!(logits[1].is_finite()); // 5.0 at index 1
    // Others should be NEG_INFINITY
    assert!(logits[0].is_infinite());
    assert!(logits[2].is_infinite());
}

#[test]
fn top_k_empty_slice() {
    let mut logits: Vec<f32> = vec![];
    let kept = apply_top_k(&mut logits, 5);
    assert_eq!(kept, 0);
}

// ---------------------------------------------------------------------------
// apply_top_p edge cases
// ---------------------------------------------------------------------------

#[test]
fn top_p_one_is_noop() {
    let mut probs = vec![0.5, 0.3, 0.2];
    let original = probs.clone();
    apply_top_p(&mut probs, 1.0);
    assert_eq!(probs, original);
}

#[test]
fn top_p_above_one_is_noop() {
    let mut probs = vec![0.5, 0.3, 0.2];
    let original = probs.clone();
    apply_top_p(&mut probs, 1.5);
    assert_eq!(probs, original);
}

#[test]
fn top_p_very_low_keeps_top_only() {
    let mut probs = vec![0.5, 0.3, 0.2];
    apply_top_p(&mut probs, 0.4);
    // Only the top token (0.5) should survive
    assert!(probs[0] > 0.0);
    // At least one other zeroed
    let zeroed = probs.iter().filter(|&&p| p == 0.0).count();
    assert!(zeroed >= 1);
}

#[test]
fn top_p_empty_slice() {
    let mut probs: Vec<f32> = vec![];
    apply_top_p(&mut probs, 0.9);
    assert!(probs.is_empty());
}

// ---------------------------------------------------------------------------
// softmax_in_place edge cases
// ---------------------------------------------------------------------------

#[test]
fn softmax_sums_to_one() {
    let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    softmax_in_place(&mut logits);
    let sum: f32 = logits.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn softmax_preserves_order() {
    let mut logits = vec![1.0, 3.0, 2.0];
    softmax_in_place(&mut logits);
    assert!(logits[1] > logits[2]);
    assert!(logits[2] > logits[0]);
}

#[test]
fn softmax_handles_neg_infinity() {
    let mut logits = vec![1.0, f32::NEG_INFINITY, 3.0];
    softmax_in_place(&mut logits);
    assert_eq!(logits[1], 0.0);
    assert!(logits[0] > 0.0);
    assert!(logits[2] > 0.0);
    let sum: f32 = logits.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn softmax_all_neg_infinity_uniform() {
    let mut logits = vec![f32::NEG_INFINITY; 5];
    softmax_in_place(&mut logits);
    // Should fall back to uniform
    for &p in &logits {
        assert!((p - 0.2).abs() < 1e-5);
    }
}

#[test]
fn softmax_single_element() {
    let mut logits = vec![42.0];
    softmax_in_place(&mut logits);
    assert!((logits[0] - 1.0).abs() < 1e-5);
}

#[test]
fn softmax_empty_slice() {
    let mut logits: Vec<f32> = vec![];
    softmax_in_place(&mut logits);
    assert!(logits.is_empty());
}

#[test]
fn softmax_identical_logits_uniform() {
    let mut logits = vec![3.0; 4];
    softmax_in_place(&mut logits);
    for &p in &logits {
        assert!((p - 0.25).abs() < 1e-5);
    }
}

// ---------------------------------------------------------------------------
// apply_repetition_penalty edge cases
// ---------------------------------------------------------------------------

#[test]
fn repetition_penalty_one_is_noop() {
    let mut logits = vec![1.0, 2.0, 3.0];
    let original = logits.clone();
    apply_repetition_penalty(&mut logits, &[0, 1, 2], 1.0);
    assert_eq!(logits, original);
}

#[test]
fn repetition_penalty_empty_history_is_noop() {
    let mut logits = vec![1.0, 2.0, 3.0];
    let original = logits.clone();
    apply_repetition_penalty(&mut logits, &[], 2.0);
    assert_eq!(logits, original);
}

#[test]
fn repetition_penalty_positive_logit_divided() {
    let mut logits = vec![0.0, 4.0, 0.0];
    apply_repetition_penalty(&mut logits, &[1], 2.0);
    assert!((logits[1] - 2.0).abs() < 1e-6);
}

#[test]
fn repetition_penalty_negative_logit_multiplied() {
    let mut logits = vec![0.0, -2.0, 0.0];
    apply_repetition_penalty(&mut logits, &[1], 2.0);
    assert!((logits[1] - (-4.0)).abs() < 1e-6);
}

#[test]
fn repetition_penalty_out_of_bounds_ignored() {
    let mut logits = vec![1.0, 2.0, 3.0];
    let original = logits.clone();
    apply_repetition_penalty(&mut logits, &[100], 2.0);
    assert_eq!(logits, original);
}

#[test]
fn repetition_penalty_zero_logit_unchanged() {
    let mut logits = vec![0.0, 2.0, 3.0];
    apply_repetition_penalty(&mut logits, &[0], 2.0);
    assert_eq!(logits[0], 0.0); // 0 * anything = 0
}

// ---------------------------------------------------------------------------
// argmax edge cases
// ---------------------------------------------------------------------------

#[test]
fn argmax_picks_highest() {
    assert_eq!(argmax(&[1.0, 5.0, 3.0]), 1);
}

#[test]
fn argmax_empty_returns_zero() {
    assert_eq!(argmax(&[]), 0);
}

#[test]
fn argmax_single_element() {
    assert_eq!(argmax(&[42.0]), 0);
}

#[test]
fn argmax_all_equal_returns_last() {
    // max_by returns last maximum for equal elements
    let result = argmax(&[1.0, 1.0, 1.0]);
    assert_eq!(result, 2);
}

// ---------------------------------------------------------------------------
// apply_min_p edge cases
// ---------------------------------------------------------------------------

#[test]
fn min_p_zero_is_noop() {
    let mut probs = vec![0.5, 0.3, 0.1, 0.05, 0.05];
    let original = probs.clone();
    apply_min_p(&mut probs, 0.0);
    assert_eq!(probs, original);
}

#[test]
fn min_p_negative_is_noop() {
    let mut probs = vec![0.5, 0.3, 0.2];
    let original = probs.clone();
    apply_min_p(&mut probs, -1.0);
    assert_eq!(probs, original);
}

#[test]
fn min_p_filters_low_probability() {
    let mut probs = vec![0.5, 0.3, 0.1, 0.05, 0.05];
    apply_min_p(&mut probs, 0.2);
    // threshold = 0.2 * 0.5 = 0.1
    assert!(probs[0] > 0.0); // 0.5
    assert!(probs[1] > 0.0); // 0.3
    assert!(probs[2] > 0.0); // 0.1 == 0.1 (not strictly less)
    assert_eq!(probs[3], 0.0); // 0.05 < 0.1
    assert_eq!(probs[4], 0.0); // 0.05 < 0.1
}

#[test]
fn min_p_empty_slice() {
    let mut probs: Vec<f32> = vec![];
    apply_min_p(&mut probs, 0.5);
    assert!(probs.is_empty());
}

// ---------------------------------------------------------------------------
// apply_typical edge cases
// ---------------------------------------------------------------------------

#[test]
fn typical_one_is_noop() {
    let mut probs = vec![0.5, 0.3, 0.2];
    let original = probs.clone();
    apply_typical(&mut probs, 1.0);
    assert_eq!(probs, original);
}

#[test]
fn typical_above_one_is_noop() {
    let mut probs = vec![0.5, 0.3, 0.2];
    let original = probs.clone();
    apply_typical(&mut probs, 1.5);
    assert_eq!(probs, original);
}

#[test]
fn typical_filters_atypical_tokens() {
    let mut probs = vec![0.5, 0.25, 0.15, 0.07, 0.03];
    apply_typical(&mut probs, 0.8);
    let non_zero = probs.iter().filter(|&&p| p > 0.0).count();
    assert!(non_zero >= 1);
    assert!(non_zero < 5);
}

#[test]
fn typical_empty_slice() {
    let mut probs: Vec<f32> = vec![];
    apply_typical(&mut probs, 0.5);
    assert!(probs.is_empty());
}

#[test]
fn typical_all_zeros_is_noop() {
    let mut probs = vec![0.0; 5];
    apply_typical(&mut probs, 0.5);
    assert_eq!(probs, vec![0.0; 5]);
}

// ---------------------------------------------------------------------------
// Large vocab stress tests (multi-SLM realistic sizes)
// ---------------------------------------------------------------------------

#[test]
fn pipeline_32k_vocab() {
    let mut logits = vec![0.0f32; 32000];
    logits[15000] = 10.0; // spike at one token
    apply_temperature(&mut logits, 0.7);
    softmax_in_place(&mut logits);
    let best = argmax(&logits);
    assert_eq!(best, 15000);
}

#[test]
fn pipeline_100k_vocab() {
    let mut logits = vec![0.0f32; 100352]; // Phi-4 vocab size
    logits[50000] = 10.0;
    apply_temperature(&mut logits, 0.8);
    softmax_in_place(&mut logits);
    assert_eq!(argmax(&logits), 50000);
}

#[test]
fn pipeline_128k_vocab() {
    let mut logits = vec![0.0f32; 128256]; // LLaMA-3 vocab size
    logits[64000] = 10.0;
    softmax_in_place(&mut logits);
    assert_eq!(argmax(&logits), 64000);
}

#[test]
fn softmax_numerical_stability_large_values() {
    let mut logits = vec![1000.0, 1001.0, 999.0];
    softmax_in_place(&mut logits);
    let sum: f32 = logits.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    assert!(logits[1] > logits[0]); // 1001 > 1000
}

#[test]
fn softmax_numerical_stability_very_negative() {
    let mut logits = vec![-1000.0, -999.0, -1001.0];
    softmax_in_place(&mut logits);
    let sum: f32 = logits.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    assert!(logits[1] > logits[0]); // -999 > -1000
}
