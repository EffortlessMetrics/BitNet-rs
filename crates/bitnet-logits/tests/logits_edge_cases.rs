//! Edge case and boundary tests for logits transformations.
//!
//! Tests exercise unusual inputs, boundary conditions, and numerical stability
//! for all logits pipeline operations.

use bitnet_logits::{
    apply_min_p, apply_repetition_penalty, apply_temperature, apply_top_k, apply_top_p,
    apply_typical, argmax, softmax_in_place,
};

// --- apply_temperature edge cases ---

#[test]
fn temperature_very_low_makes_distribution_peaked() {
    let mut logits = vec![1.0, 2.0, 3.0];
    apply_temperature(&mut logits, 0.01);
    // The largest logit should be significantly larger relative to others
    assert!(logits[2] > logits[0], "Max logit should be larger after low temperature");
    assert!(logits[2] > logits[1], "Max logit should exceed second largest");
}

#[test]
fn temperature_one_is_identity() {
    let original = vec![1.0, 2.0, 3.0, -1.0];
    let mut logits = original.clone();
    apply_temperature(&mut logits, 1.0);
    for (a, b) in logits.iter().zip(original.iter()) {
        assert!((a - b).abs() < 1e-6, "Temperature 1.0 should be identity");
    }
}

#[test]
fn temperature_high_flattens_distribution() {
    let mut logits = vec![1.0, 100.0, -50.0];
    apply_temperature(&mut logits, 100.0);
    // After high temperature, values should be closer together
    let range = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - logits.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(range < 2.0, "High temperature should flatten distribution");
}

#[test]
fn temperature_single_element() {
    let mut logits = vec![5.0];
    apply_temperature(&mut logits, 0.5);
    assert!((logits[0] - 10.0).abs() < f32::EPSILON); // 5.0 / 0.5
}

#[test]
fn temperature_empty_array() {
    let mut logits: Vec<f32> = vec![];
    apply_temperature(&mut logits, 1.0);
    assert!(logits.is_empty());
}

#[test]
fn temperature_all_zeros() {
    let mut logits = vec![0.0, 0.0, 0.0];
    apply_temperature(&mut logits, 0.5);
    assert!(logits.iter().all(|&x| x == 0.0));
}

// --- apply_top_k edge cases ---

#[test]
fn top_k_zero_returns_all() {
    let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let kept = apply_top_k(&mut logits, 0);
    // k=0 should keep all
    assert_eq!(kept, 5);
}

#[test]
fn top_k_one_keeps_max() {
    let mut logits = vec![1.0, 5.0, 3.0, 2.0];
    apply_top_k(&mut logits, 1);
    assert!((logits[1] - 5.0).abs() < f32::EPSILON);
    assert!(logits[0] == f32::NEG_INFINITY);
    assert!(logits[2] == f32::NEG_INFINITY);
    assert!(logits[3] == f32::NEG_INFINITY);
}

#[test]
fn top_k_equal_to_length() {
    let mut logits = vec![1.0, 2.0, 3.0];
    let kept = apply_top_k(&mut logits, 3);
    assert_eq!(kept, 3);
    assert!(logits.iter().all(|&x| x != f32::NEG_INFINITY));
}

#[test]
fn top_k_larger_than_length() {
    let mut logits = vec![1.0, 2.0];
    let kept = apply_top_k(&mut logits, 100);
    assert_eq!(kept, 2);
}

#[test]
fn top_k_single_element() {
    let mut logits = vec![42.0];
    let kept = apply_top_k(&mut logits, 1);
    assert_eq!(kept, 1);
    assert!((logits[0] - 42.0).abs() < f32::EPSILON);
}

#[test]
fn top_k_empty_array() {
    let mut logits: Vec<f32> = vec![];
    let kept = apply_top_k(&mut logits, 5);
    assert_eq!(kept, 0);
}

#[test]
fn top_k_with_duplicates() {
    let mut logits = vec![3.0, 3.0, 3.0, 1.0, 2.0];
    let kept = apply_top_k(&mut logits, 3);
    assert!(kept >= 3);
}

// --- softmax_in_place edge cases ---

#[test]
fn softmax_sums_to_one() {
    let mut logits = vec![1.0, 2.0, 3.0, 4.0];
    softmax_in_place(&mut logits);
    let sum: f32 = logits.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Softmax should sum to ~1.0, got {sum}");
}

#[test]
fn softmax_all_equal_gives_uniform() {
    let mut logits = vec![5.0, 5.0, 5.0, 5.0];
    softmax_in_place(&mut logits);
    for &p in &logits {
        assert!((p - 0.25).abs() < 1e-5, "Equal logits should give uniform distribution");
    }
}

#[test]
fn softmax_single_element() {
    let mut logits = vec![42.0];
    softmax_in_place(&mut logits);
    assert!((logits[0] - 1.0).abs() < 1e-6);
}

#[test]
fn softmax_large_values_no_overflow() {
    let mut logits = vec![1000.0, 1001.0, 999.0];
    softmax_in_place(&mut logits);
    let sum: f32 = logits.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "Large values should not overflow");
    assert!(logits.iter().all(|&x| x.is_finite()));
}

#[test]
fn softmax_negative_values() {
    let mut logits = vec![-100.0, -200.0, -50.0];
    softmax_in_place(&mut logits);
    let sum: f32 = logits.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4);
    assert!(logits[2] > logits[0]); // -50 > -100, so prob should be higher
}

#[test]
fn softmax_empty_array() {
    let mut logits: Vec<f32> = vec![];
    softmax_in_place(&mut logits);
    assert!(logits.is_empty());
}

// --- apply_top_p edge cases ---

#[test]
fn top_p_one_keeps_all() {
    let mut probs = vec![0.1, 0.2, 0.3, 0.4];
    apply_top_p(&mut probs, 1.0);
    assert!(probs.iter().all(|&p| p > 0.0));
}

#[test]
fn top_p_very_small_keeps_top() {
    let mut probs = vec![0.1, 0.6, 0.2, 0.1];
    apply_top_p(&mut probs, 0.01);
    // The top probability (0.6) should remain
    assert!(probs[1] > 0.0);
}

#[test]
fn top_p_single_element() {
    let mut probs = vec![1.0];
    apply_top_p(&mut probs, 0.5);
    assert!(probs[0] > 0.0);
}

#[test]
fn top_p_empty_array() {
    let mut probs: Vec<f32> = vec![];
    apply_top_p(&mut probs, 0.9);
    assert!(probs.is_empty());
}

// --- apply_repetition_penalty edge cases ---

#[test]
fn repetition_penalty_one_is_identity() {
    let original = vec![1.0, 2.0, -1.0, 3.0];
    let mut logits = original.clone();
    apply_repetition_penalty(&mut logits, &[0, 1, 2, 3], 1.0);
    for (a, b) in logits.iter().zip(original.iter()) {
        assert!((a - b).abs() < 1e-6, "Penalty 1.0 should be identity");
    }
}

#[test]
fn repetition_penalty_reduces_positive_logits() {
    let mut logits = vec![1.0, 5.0, 3.0];
    apply_repetition_penalty(&mut logits, &[1], 2.0);
    assert!(logits[1] < 5.0, "Positive logit should be reduced by penalty");
    assert!((logits[0] - 1.0).abs() < f32::EPSILON, "Unpenalized logit should be unchanged");
}

#[test]
fn repetition_penalty_amplifies_negative_logits() {
    let mut logits = vec![-1.0, 2.0, -3.0];
    apply_repetition_penalty(&mut logits, &[0, 2], 2.0);
    assert!(logits[0] < -1.0, "Negative logit should be more negative");
    assert!(logits[2] < -3.0, "Negative logit should be more negative");
}

#[test]
fn repetition_penalty_empty_token_ids() {
    let original = vec![1.0, 2.0, 3.0];
    let mut logits = original.clone();
    apply_repetition_penalty(&mut logits, &[], 2.0);
    for (a, b) in logits.iter().zip(original.iter()) {
        assert!((a - b).abs() < 1e-6, "No tokens should leave logits unchanged");
    }
}

#[test]
fn repetition_penalty_out_of_bounds_token_ids_ignored() {
    let mut logits = vec![1.0, 2.0, 3.0];
    // Token ID 100 is out of bounds for a 3-element array
    apply_repetition_penalty(&mut logits, &[100], 2.0);
    // Should not panic, logits unchanged
    assert_eq!(logits, vec![1.0, 2.0, 3.0]);
}

// --- argmax edge cases ---

#[test]
fn argmax_single_element() {
    assert_eq!(argmax(&[42.0]), 0);
}

#[test]
fn argmax_first_is_max() {
    assert_eq!(argmax(&[10.0, 1.0, 2.0]), 0);
}

#[test]
fn argmax_last_is_max() {
    assert_eq!(argmax(&[1.0, 2.0, 10.0]), 2);
}

#[test]
fn argmax_all_equal() {
    let result = argmax(&[5.0, 5.0, 5.0]);
    assert!(result < 3);
}

#[test]
fn argmax_negative_values() {
    assert_eq!(argmax(&[-10.0, -5.0, -1.0]), 2);
}

// --- apply_min_p edge cases ---

#[test]
fn min_p_zero_keeps_all() {
    let mut probs = vec![0.001, 0.5, 0.3, 0.199];
    apply_min_p(&mut probs, 0.0);
    assert!(probs.iter().all(|&p| p > 0.0));
}

#[test]
fn min_p_one_keeps_only_max() {
    let mut probs = vec![0.1, 0.5, 0.3, 0.1];
    apply_min_p(&mut probs, 1.0);
    // Only the max (0.5) should remain non-zero
    assert!(probs[1] > 0.0);
    // Others below 1.0 * 0.5 = 0.5 should be zeroed
    assert!((probs[0] - 0.0).abs() < f32::EPSILON);
    assert!((probs[3] - 0.0).abs() < f32::EPSILON);
}

#[test]
fn min_p_single_element() {
    let mut probs = vec![1.0];
    apply_min_p(&mut probs, 0.5);
    assert!(probs[0] > 0.0);
}

#[test]
fn min_p_empty_array() {
    let mut probs: Vec<f32> = vec![];
    apply_min_p(&mut probs, 0.5);
    assert!(probs.is_empty());
}

// --- apply_typical edge cases ---

#[test]
fn typical_one_keeps_all() {
    let mut probs = vec![0.25, 0.25, 0.25, 0.25];
    apply_typical(&mut probs, 1.0);
    assert!(probs.iter().all(|&p| p > 0.0));
}

#[test]
fn typical_small_value_filters() {
    let mut probs = vec![0.5, 0.3, 0.15, 0.05];
    apply_typical(&mut probs, 0.1);
    // Should keep at least one probability
    let non_zero = probs.iter().filter(|&&p| p > 0.0).count();
    assert!(non_zero >= 1);
}

#[test]
fn typical_single_element() {
    let mut probs = vec![1.0];
    apply_typical(&mut probs, 0.5);
    assert!(probs[0] > 0.0);
}

#[test]
fn typical_empty_array() {
    let mut probs: Vec<f32> = vec![];
    apply_typical(&mut probs, 0.5);
    assert!(probs.is_empty());
}

// --- Pipeline composition tests ---

#[test]
fn full_pipeline_temperature_topk_softmax_argmax() {
    let mut logits = vec![1.0, 3.0, 2.0, 0.5, 4.0];
    apply_temperature(&mut logits, 0.5);
    apply_top_k(&mut logits, 3);
    softmax_in_place(&mut logits);
    let idx = argmax(&logits);
    // Token 4 (value 4.0) should be picked
    assert_eq!(idx, 4);
}

#[test]
fn pipeline_with_repetition_penalty() {
    let mut logits = vec![1.0, 5.0, 3.0, 2.0];
    // Penalize token 1 (previously generated)
    apply_repetition_penalty(&mut logits, &[1], 3.0);
    apply_temperature(&mut logits, 1.0);
    softmax_in_place(&mut logits);
    // Token 1 should no longer be the argmax due to penalty
    let idx = argmax(&logits);
    assert_ne!(idx, 1, "Penalized token should not be argmax");
}

#[test]
fn pipeline_topk_then_topp() {
    let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    apply_top_k(&mut logits, 3);
    softmax_in_place(&mut logits);
    apply_top_p(&mut logits, 0.9);
    // Should have at most 3 non-zero probabilities
    let non_zero = logits.iter().filter(|&&p| p > 0.0).count();
    assert!(non_zero <= 3);
}
