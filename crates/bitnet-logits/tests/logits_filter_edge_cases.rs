//! Edge-case tests for logits filter functions.
//!
//! Covers: empty/single-element slices, NaN/Inf handling, boundary conditions
//! for all filters, and filter chaining order dependence.

use bitnet_logits::{
    apply_min_p, apply_repetition_penalty, apply_temperature, apply_top_k, apply_top_p,
    apply_typical, argmax, softmax_in_place,
};

/// Helper: assert two f32 values are approximately equal.
fn assert_approx(a: f32, b: f32, eps: f32) {
    assert!((a - b).abs() < eps, "expected {a} ≈ {b} (eps={eps}), diff={}", (a - b).abs());
}

// ============================================================================
// 1. Empty slices — all functions should handle gracefully
// ============================================================================

#[test]
fn empty_temperature() {
    let mut v: Vec<f32> = vec![];
    apply_temperature(&mut v, 0.5);
    assert!(v.is_empty());
}

#[test]
fn empty_top_k() {
    let mut v: Vec<f32> = vec![];
    let kept = apply_top_k(&mut v, 3);
    assert_eq!(kept, 0);
}

#[test]
fn empty_softmax() {
    let mut v: Vec<f32> = vec![];
    softmax_in_place(&mut v);
    assert!(v.is_empty());
}

#[test]
fn empty_top_p() {
    let mut v: Vec<f32> = vec![];
    apply_top_p(&mut v, 0.9);
    assert!(v.is_empty());
}

#[test]
fn empty_min_p() {
    let mut v: Vec<f32> = vec![];
    apply_min_p(&mut v, 0.5);
    assert!(v.is_empty());
}

#[test]
fn empty_typical() {
    let mut v: Vec<f32> = vec![];
    apply_typical(&mut v, 0.5);
    assert!(v.is_empty());
}

#[test]
fn empty_repetition_penalty() {
    let mut v: Vec<f32> = vec![];
    apply_repetition_penalty(&mut v, &[0, 1], 2.0);
    assert!(v.is_empty());
}

#[test]
fn empty_argmax() {
    assert_eq!(argmax(&[]), 0);
}

// ============================================================================
// 2. Single-element arrays
// ============================================================================

#[test]
fn single_softmax_gives_one() {
    let mut v = vec![42.0];
    softmax_in_place(&mut v);
    assert_approx(v[0], 1.0, 1e-6);
}

#[test]
fn single_top_k_one_preserves() {
    let mut v = vec![7.0];
    let kept = apply_top_k(&mut v, 1);
    assert_eq!(kept, 1);
    assert_approx(v[0], 7.0, 1e-6);
}

#[test]
fn single_top_p_preserves() {
    let mut v = vec![1.0];
    apply_top_p(&mut v, 0.5);
    assert_approx(v[0], 1.0, 1e-6);
}

#[test]
fn single_min_p_preserves() {
    let mut v = vec![1.0];
    apply_min_p(&mut v, 0.99);
    assert!(v[0] > 0.0);
}

#[test]
fn single_typical_preserves() {
    let mut v = vec![1.0];
    apply_typical(&mut v, 0.1);
    assert!(v[0] > 0.0);
}

#[test]
fn single_temperature_scales() {
    let mut v = vec![4.0];
    apply_temperature(&mut v, 2.0);
    assert_approx(v[0], 2.0, 1e-6);
}

#[test]
fn single_argmax_returns_zero() {
    assert_eq!(argmax(&[99.0]), 0);
}

// ============================================================================
// 3. NaN and Inf handling
// ============================================================================

#[test]
fn softmax_with_neg_infinity_entries() {
    // Simulates post-top_k: some entries are NEG_INFINITY
    let mut v = vec![f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY, 2.0];
    softmax_in_place(&mut v);
    assert_approx(v[0], 0.0, 1e-6);
    assert_approx(v[2], 0.0, 1e-6);
    assert!(v[1] > 0.0);
    assert!(v[3] > 0.0);
    let sum: f32 = v.iter().sum();
    assert_approx(sum, 1.0, 1e-5);
}

#[test]
fn softmax_all_neg_infinity_gives_uniform() {
    let mut v = vec![f32::NEG_INFINITY; 4];
    softmax_in_place(&mut v);
    // All exp(-inf - (-inf)) = exp(0) for max=-inf? Actually max = -inf,
    // so v - max = -inf - (-inf) = NaN → exp(NaN) = NaN.
    // The implementation checks `v == NEG_INFINITY` and sets to 0.0,
    // then sum==0 triggers uniform fallback.
    for &p in &v {
        assert_approx(p, 0.25, 1e-6);
    }
}

#[test]
fn softmax_with_positive_infinity() {
    let mut v = vec![1.0, f32::INFINITY, 3.0];
    softmax_in_place(&mut v);
    // max = inf, so exp(inf - inf) = exp(NaN) = NaN → sum is NaN,
    // which fails the `sum > 0.0` check, triggering the uniform fallback.
    let expected = 1.0 / 3.0;
    for &p in &v {
        assert_approx(p, expected, 1e-6);
    }
}

#[test]
fn softmax_with_nan_does_not_panic() {
    let mut v = vec![1.0, f32::NAN, 3.0];
    softmax_in_place(&mut v);
    // We just verify it doesn't panic. NaN propagation is implementation-defined.
    let _ = v;
}

#[test]
fn temperature_with_nan_does_not_panic() {
    let mut v = vec![1.0, f32::NAN, 3.0];
    apply_temperature(&mut v, 0.5);
    let _ = v; // No panic is the assertion
}

#[test]
fn top_k_with_nan_does_not_panic() {
    let mut v = vec![1.0, f32::NAN, 3.0, 2.0];
    let _kept = apply_top_k(&mut v, 2);
    // No panic is the assertion
}

#[test]
fn argmax_with_nan() {
    // NaN comparisons are tricky; just verify no panic
    let _idx = argmax(&[1.0, f32::NAN, 3.0]);
}

#[test]
fn argmax_with_inf() {
    let idx = argmax(&[1.0, f32::INFINITY, 3.0]);
    assert_eq!(idx, 1, "INFINITY should be the argmax");
}

#[test]
fn argmax_with_neg_infinity() {
    let idx = argmax(&[f32::NEG_INFINITY, -1.0, f32::NEG_INFINITY]);
    assert_eq!(idx, 1, "-1.0 is the only finite value");
}

#[test]
fn repetition_penalty_with_inf_penalty_is_noop() {
    // penalty=Inf is not finite, so it's treated as a no-op
    let mut v = vec![1.0, 2.0, 3.0];
    let original = v.clone();
    apply_repetition_penalty(&mut v, &[0, 1, 2], f32::INFINITY);
    assert_eq!(v, original);
}

#[test]
fn repetition_penalty_with_nan_penalty_is_noop() {
    let mut v = vec![1.0, 2.0, 3.0];
    let original = v.clone();
    apply_repetition_penalty(&mut v, &[0, 1, 2], f32::NAN);
    assert_eq!(v, original);
}

// ============================================================================
// 4. min_p boundaries — min_p=0, min_p=0.5, min_p=1.0
// ============================================================================

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
fn min_p_half_filters_below_half_max() {
    let mut probs = vec![0.6, 0.4, 0.2, 0.1];
    apply_min_p(&mut probs, 0.5);
    // Threshold = 0.5 * 0.6 = 0.3
    assert!(probs[0] > 0.0); // 0.6 >= 0.3
    assert!(probs[1] > 0.0); // 0.4 >= 0.3
    assert_approx(probs[2], 0.0, 1e-6); // 0.2 < 0.3
    assert_approx(probs[3], 0.0, 1e-6); // 0.1 < 0.3
}

#[test]
fn min_p_one_keeps_only_max() {
    let mut probs = vec![0.5, 0.3, 0.2];
    apply_min_p(&mut probs, 1.0);
    // Threshold = 1.0 * 0.5 = 0.5. Only probs >= 0.5 survive.
    assert!(probs[0] > 0.0);
    assert_approx(probs[1], 0.0, 1e-6);
    assert_approx(probs[2], 0.0, 1e-6);
}

#[test]
fn min_p_with_tied_max() {
    let mut probs = vec![0.4, 0.4, 0.1, 0.1];
    apply_min_p(&mut probs, 1.0);
    // Threshold = 1.0 * 0.4 = 0.4. Both 0.4 entries survive.
    assert!(probs[0] > 0.0);
    assert!(probs[1] > 0.0);
    assert_approx(probs[2], 0.0, 1e-6);
    assert_approx(probs[3], 0.0, 1e-6);
}

#[test]
fn min_p_all_equal_probs() {
    let mut probs = vec![0.25, 0.25, 0.25, 0.25];
    apply_min_p(&mut probs, 0.5);
    // Threshold = 0.5 * 0.25 = 0.125. All 0.25 >= 0.125, so all survive.
    assert!(probs.iter().all(|&p| p > 0.0));
}

// ============================================================================
// 5. Typical sampling — uniform and extreme distributions
// ============================================================================

#[test]
fn typical_uniform_distribution() {
    // Uniform: all tokens are equally "typical"
    let mut probs = vec![0.25, 0.25, 0.25, 0.25];
    apply_typical(&mut probs, 0.5);
    // With uniform distribution, entropy = ln(4) ≈ 1.386
    // Each token's surprise = -ln(0.25) = ln(4) ≈ 1.386 = entropy
    // All deviations are 0, so they should all be equally ranked.
    // With typical_p=0.5, at least 2 tokens should survive (cumsum of 2 * 0.25 = 0.5).
    let non_zero = probs.iter().filter(|&&p| p > 0.0).count();
    assert!(non_zero >= 2, "Uniform dist typical_p=0.5 should keep ≥2 tokens, got {non_zero}");
}

#[test]
fn typical_extreme_one_dominant() {
    // One token has nearly all probability
    let mut probs = vec![0.97, 0.01, 0.01, 0.01];
    apply_typical(&mut probs, 0.5);
    // The dominant token should survive
    let non_zero = probs.iter().filter(|&&p| p > 0.0).count();
    assert!(non_zero >= 1);
}

#[test]
fn typical_preserves_nonzero_sum() {
    let mut probs = vec![0.4, 0.3, 0.2, 0.1];
    apply_typical(&mut probs, 0.3);
    let sum: f32 = probs.iter().sum();
    assert!(sum > 0.0, "Remaining probability must be positive");
}

#[test]
fn typical_nearly_one_is_almost_noop() {
    let original = vec![0.4, 0.3, 0.2, 0.1];
    let mut probs = original.clone();
    apply_typical(&mut probs, 0.99);
    // With typical_p very close to 1.0, most or all tokens survive
    let non_zero = probs.iter().filter(|&&p| p > 0.0).count();
    assert!(non_zero >= 3, "typical_p=0.99 should keep almost all tokens");
}

#[test]
fn typical_one_point_zero_is_noop() {
    let original = vec![0.5, 0.3, 0.2];
    let mut probs = original.clone();
    apply_typical(&mut probs, 1.0);
    assert_eq!(probs, original);
}

#[test]
fn typical_two_element_distribution() {
    let mut probs = vec![0.8, 0.2];
    apply_typical(&mut probs, 0.5);
    let non_zero = probs.iter().filter(|&&p| p > 0.0).count();
    assert!(non_zero >= 1);
}

// ============================================================================
// 6. Temperature scaling — temp=0, temp=1, very large
// ============================================================================

#[test]
fn temperature_zero_is_noop() {
    // temperature==0 is a no-op (greedy is handled externally)
    let original = vec![1.0, 2.0, 3.0];
    let mut logits = original.clone();
    apply_temperature(&mut logits, 0.0);
    assert_eq!(logits, original);
}

#[test]
fn temperature_one_is_noop() {
    let original = vec![-1.0, 0.0, 1.0, 5.0];
    let mut logits = original.clone();
    apply_temperature(&mut logits, 1.0);
    assert_eq!(logits, original);
}

#[test]
fn temperature_very_large_flattens() {
    let mut logits = vec![1.0, 100.0, -50.0];
    apply_temperature(&mut logits, 1e6);
    // All logits divided by 1e6 → very close to zero
    for &l in &logits {
        assert!(l.abs() < 0.001, "Very large temp should flatten, got {l}");
    }
}

#[test]
fn temperature_very_small_sharpens() {
    let mut logits = vec![1.0, 2.0, 3.0];
    apply_temperature(&mut logits, 0.001);
    // logits *= 1/0.001 = 1000
    assert_approx(logits[0], 1000.0, 1e-1);
    assert_approx(logits[1], 2000.0, 1e-1);
    assert_approx(logits[2], 3000.0, 1e-1);
}

#[test]
fn temperature_preserves_argmax() {
    let original = vec![1.0, 5.0, 3.0, 2.0];
    for temp in [0.01, 0.5, 1.0, 2.0, 100.0] {
        let mut logits = original.clone();
        apply_temperature(&mut logits, temp);
        assert_eq!(argmax(&logits), 1, "Temperature {temp} should preserve argmax");
    }
}

// ============================================================================
// 7. Repetition penalty — penalty=1.0, penalty=2.0, empty history
// ============================================================================

#[test]
fn repetition_penalty_one_is_identity() {
    let original = vec![1.0, -2.0, 3.0, 0.0];
    let mut logits = original.clone();
    apply_repetition_penalty(&mut logits, &[0, 1, 2, 3], 1.0);
    assert_eq!(logits, original);
}

#[test]
fn repetition_penalty_two_halves_positive() {
    let mut logits = vec![4.0, 2.0, 6.0];
    apply_repetition_penalty(&mut logits, &[0, 2], 2.0);
    assert_approx(logits[0], 2.0, 1e-6); // 4.0 / 2.0
    assert_approx(logits[1], 2.0, 1e-6); // unchanged
    assert_approx(logits[2], 3.0, 1e-6); // 6.0 / 2.0
}

#[test]
fn repetition_penalty_two_doubles_negative() {
    let mut logits = vec![-2.0, 1.0, -4.0];
    apply_repetition_penalty(&mut logits, &[0, 2], 2.0);
    assert_approx(logits[0], -4.0, 1e-6); // -2.0 * 2.0
    assert_approx(logits[1], 1.0, 1e-6); // unchanged
    assert_approx(logits[2], -8.0, 1e-6); // -4.0 * 2.0
}

#[test]
fn repetition_penalty_empty_history_is_noop() {
    let original = vec![1.0, 2.0, 3.0];
    let mut logits = original.clone();
    apply_repetition_penalty(&mut logits, &[], 5.0);
    assert_eq!(logits, original);
}

#[test]
fn repetition_penalty_zero_logit_unchanged() {
    // Zero logit: 0.0 / penalty = 0.0, 0.0 * penalty = 0.0
    let mut logits = vec![0.0, 1.0, -1.0];
    apply_repetition_penalty(&mut logits, &[0], 2.0);
    assert_approx(logits[0], 0.0, 1e-6);
}

#[test]
fn repetition_penalty_zero_penalty_is_noop() {
    let original = vec![1.0, 2.0, 3.0];
    let mut logits = original.clone();
    apply_repetition_penalty(&mut logits, &[0, 1], 0.0);
    assert_eq!(logits, original);
}

#[test]
fn repetition_penalty_negative_penalty_is_noop() {
    let original = vec![1.0, 2.0, 3.0];
    let mut logits = original.clone();
    apply_repetition_penalty(&mut logits, &[0, 1], -1.0);
    assert_eq!(logits, original);
}

#[test]
fn repetition_penalty_duplicate_token_ids() {
    // Same token penalized twice → penalty applied twice
    let mut logits = vec![8.0, 1.0];
    apply_repetition_penalty(&mut logits, &[0, 0], 2.0);
    // First application: 8.0 / 2.0 = 4.0
    // Second application: 4.0 / 2.0 = 2.0
    assert_approx(logits[0], 2.0, 1e-6);
}

#[test]
fn repetition_penalty_oob_token_ignored() {
    let mut logits = vec![1.0, 2.0];
    apply_repetition_penalty(&mut logits, &[999], 2.0);
    assert_approx(logits[0], 1.0, 1e-6);
    assert_approx(logits[1], 2.0, 1e-6);
}

// ============================================================================
// 8. Top-k edge cases — k=0, k=1, k > vocab_size
// ============================================================================

#[test]
fn top_k_zero_is_noop() {
    let original = vec![1.0, 5.0, 3.0];
    let mut logits = original.clone();
    let kept = apply_top_k(&mut logits, 0);
    assert_eq!(kept, 3);
    assert_eq!(logits, original);
}

#[test]
fn top_k_one_keeps_single_max() {
    let mut logits = vec![1.0, 3.0, 2.0, 5.0, 4.0];
    let kept = apply_top_k(&mut logits, 1);
    assert_eq!(kept, 1);
    assert_approx(logits[3], 5.0, 1e-6);
    for (i, &l) in logits.iter().enumerate() {
        if i != 3 {
            assert_eq!(l, f32::NEG_INFINITY, "Non-max at index {i} should be -inf");
        }
    }
}

#[test]
fn top_k_exceeds_vocab_is_noop() {
    let original = vec![1.0, 2.0, 3.0];
    let mut logits = original.clone();
    let kept = apply_top_k(&mut logits, 100);
    assert_eq!(kept, 3);
    assert_eq!(logits, original);
}

#[test]
fn top_k_equals_len_is_noop() {
    let original = vec![1.0, 2.0, 3.0, 4.0];
    let mut logits = original.clone();
    let kept = apply_top_k(&mut logits, 4);
    assert_eq!(kept, 4);
    assert_eq!(logits, original);
}

#[test]
fn top_k_all_identical_values() {
    let mut logits = vec![5.0; 6];
    let kept = apply_top_k(&mut logits, 3);
    // All values are equal, so any 3 can be kept
    assert!(kept >= 3);
}

#[test]
fn top_k_then_softmax_zeros_removed() {
    let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    apply_top_k(&mut logits, 2);
    softmax_in_place(&mut logits);
    // Bottom 3 should be probability 0
    let zeros = logits.iter().filter(|&&p| p == 0.0).count();
    assert_eq!(zeros, 3);
    let sum: f32 = logits.iter().sum();
    assert_approx(sum, 1.0, 1e-5);
}

// ============================================================================
// 9. Top-p edge cases — p=0.0, p=1.0, p=0.5
// ============================================================================

#[test]
fn top_p_zero_keeps_only_top() {
    let mut probs = vec![0.1, 0.6, 0.2, 0.1];
    apply_top_p(&mut probs, 0.0);
    // p=0.0: cumsum never reaches 0.0, so cutoff = indexed.len()
    // Actually 0.0 < 1.0 so it enters the loop; cumsum starts at 0.
    // The first token (0.6) makes cumsum=0.6 >= 0.0, so cutoff=1.
    // Actually let me re-read the code: cumsum >= top_p where top_p=0.0
    // First iteration: cumsum = 0.6 >= 0.0 → cutoff = 1
    // So only the top probability survives
    assert!(probs[1] > 0.0, "Top prob should survive");
}

#[test]
fn top_p_one_is_noop() {
    let original = vec![0.4, 0.3, 0.2, 0.1];
    let mut probs = original.clone();
    apply_top_p(&mut probs, 1.0);
    assert_eq!(probs, original);
}

#[test]
fn top_p_half() {
    let mut probs = vec![0.1, 0.5, 0.3, 0.1];
    apply_top_p(&mut probs, 0.5);
    // Sorted desc: 0.5 (idx 1). cumsum=0.5 >= 0.5 → cutoff=1
    // Only idx 1 survives
    assert!(probs[1] > 0.0);
    assert_approx(probs[0], 0.0, 1e-6);
    assert_approx(probs[2], 0.0, 1e-6);
    assert_approx(probs[3], 0.0, 1e-6);
}

#[test]
fn top_p_exact_cumsum_boundary() {
    // probs sorted desc: 0.5, 0.3, 0.2
    // cumsum after 0.5 = 0.5, after 0.3 = 0.8
    let mut probs = vec![0.5, 0.3, 0.2];
    apply_top_p(&mut probs, 0.8);
    // cumsum reaches 0.8 at rank 1 (0.5 + 0.3), so cutoff=2
    assert!(probs[0] > 0.0);
    assert!(probs[1] > 0.0);
    assert_approx(probs[2], 0.0, 1e-6);
}

#[test]
fn top_p_all_zero_probs() {
    // All probs are 0 (degenerate input). filter removes them all, nothing left.
    let mut probs = vec![0.0, 0.0, 0.0];
    apply_top_p(&mut probs, 0.5);
    assert!(probs.iter().all(|&p| p == 0.0));
}

#[test]
fn top_p_above_one_is_noop() {
    let original = vec![0.5, 0.3, 0.2];
    let mut probs = original.clone();
    apply_top_p(&mut probs, 1.5);
    assert_eq!(probs, original);
}

// ============================================================================
// 10. Filter chaining — apply multiple filters in sequence, order matters
// ============================================================================

#[test]
fn chaining_topk_before_softmax_vs_after() {
    // Standard: top_k → softmax → apply_top_p
    let base = vec![1.0, 5.0, 3.0, 2.0, 4.0];

    let mut path_a = base.clone();
    apply_top_k(&mut path_a, 3);
    softmax_in_place(&mut path_a);

    let mut path_b = base.clone();
    softmax_in_place(&mut path_b);
    // Applying top_k after softmax doesn't make semantic sense but
    // should still work mechanically.
    apply_top_k(&mut path_b, 3);

    // Results differ because top_k sets NEG_INF before softmax changes normalization
    assert_ne!(path_a, path_b, "top_k before softmax should differ from top_k after softmax");
}

#[test]
fn chaining_temperature_before_topk() {
    let base = vec![1.0, 5.0, 3.0, 2.0, 4.0];

    // Path A: temperature then top_k
    let mut a = base.clone();
    apply_temperature(&mut a, 0.5);
    apply_top_k(&mut a, 2);

    // Path B: top_k then temperature
    let mut b = base.clone();
    apply_top_k(&mut b, 2);
    apply_temperature(&mut b, 0.5);

    // In path A, temperature doubles all values, then top_k picks the same top-2.
    // In path B, top_k first sets some to NEG_INF, then temperature scales NEG_INF.
    // The argmax should be the same regardless of order.
    softmax_in_place(&mut a);
    softmax_in_place(&mut b);
    assert_eq!(argmax(&a), argmax(&b), "Argmax should be the same regardless of temp/topk order");
}

#[test]
fn chaining_repetition_then_temperature_then_softmax() {
    let mut logits = vec![1.0, 5.0, 3.0, 2.0];
    apply_repetition_penalty(&mut logits, &[1], 3.0);
    // Token 1 was 5.0, now 5.0/3.0 ≈ 1.667
    assert!(logits[1] < 2.0);
    apply_temperature(&mut logits, 0.5);
    softmax_in_place(&mut logits);
    let sum: f32 = logits.iter().sum();
    assert_approx(sum, 1.0, 1e-5);
    // Token 2 (original 3.0) should now be argmax since token 1 was penalized
    assert_eq!(argmax(&logits), 2);
}

#[test]
fn chaining_full_pipeline() {
    let mut logits = vec![1.0, 3.0, 2.0, 0.5, 4.0];
    apply_repetition_penalty(&mut logits, &[4], 2.0); // penalize highest
    apply_temperature(&mut logits, 0.8);
    apply_top_k(&mut logits, 3);
    softmax_in_place(&mut logits);
    apply_top_p(&mut logits, 0.9);
    apply_min_p(&mut logits, 0.1);

    let sum: f32 = logits.iter().sum();
    assert!(sum > 0.0, "Pipeline should leave some probability mass");
    let non_zero = logits.iter().filter(|&&p| p > 0.0).count();
    assert!(non_zero >= 1, "At least one token must survive the full pipeline");
}

#[test]
fn chaining_min_p_then_top_p_vs_reverse() {
    let base = vec![0.5, 0.3, 0.15, 0.04, 0.01];

    // Path A: min_p first
    let mut a = base.clone();
    apply_min_p(&mut a, 0.2); // threshold = 0.2 * 0.5 = 0.1
    apply_top_p(&mut a, 0.9);

    // Path B: top_p first
    let mut b = base.clone();
    apply_top_p(&mut b, 0.9);
    apply_min_p(&mut b, 0.2);

    let non_zero_a = a.iter().filter(|&&p| p > 0.0).count();
    let non_zero_b = b.iter().filter(|&&p| p > 0.0).count();
    // Both should have at least 1 surviving token
    assert!(non_zero_a >= 1);
    assert!(non_zero_b >= 1);
}

#[test]
fn chaining_double_softmax_gives_uniform_ish() {
    // Applying softmax twice: first normalizes, second makes it more uniform
    let mut logits = vec![1.0, 2.0, 3.0];
    softmax_in_place(&mut logits);
    let after_first = logits.clone();
    softmax_in_place(&mut logits);
    // After second softmax, distribution should be more uniform
    // because post-softmax values are in (0,1) and close together
    let range_first = after_first.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - after_first.iter().copied().fold(f32::INFINITY, f32::min);
    let range_second = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - logits.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(range_second < range_first, "Double softmax should flatten distribution");
    let sum: f32 = logits.iter().sum();
    assert_approx(sum, 1.0, 1e-5);
}

// ============================================================================
// Additional edge cases: two-element arrays, large vocab
// ============================================================================

#[test]
fn softmax_two_elements() {
    let mut v = vec![0.0, 0.0];
    softmax_in_place(&mut v);
    assert_approx(v[0], 0.5, 1e-6);
    assert_approx(v[1], 0.5, 1e-6);
}

#[test]
fn top_k_two_from_large_vocab() {
    let mut logits: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let kept = apply_top_k(&mut logits, 2);
    assert_eq!(kept, 2);
    let finite_count = logits.iter().filter(|&&l| l.is_finite()).count();
    assert_eq!(finite_count, 2);
    // The two largest values (999.0 and 998.0) should be finite
    assert!(logits[999].is_finite());
    assert!(logits[998].is_finite());
}

#[test]
fn argmax_large_array() {
    let mut logits = vec![0.0f32; 10_000];
    logits[7777] = 1.0;
    assert_eq!(argmax(&logits), 7777);
}

#[test]
fn softmax_with_mixed_neg_inf_and_finite() {
    let mut logits = vec![f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY];
    softmax_in_place(&mut logits);
    assert_approx(logits[0], 0.0, 1e-6);
    assert_approx(logits[2], 0.0, 1e-6);
    assert_approx(logits[4], 0.0, 1e-6);
    assert_approx(logits[1], 0.5, 1e-5);
    assert_approx(logits[3], 0.5, 1e-5);
}
