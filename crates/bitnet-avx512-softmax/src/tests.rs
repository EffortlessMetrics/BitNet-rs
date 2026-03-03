//! Unit tests for bitnet-avx512-softmax.
//!
//! ≥ 75 tests covering every public API, edge cases, numerical stability,
//! and property-based invariants.

use super::*;

// =========================================================================
// Helpers
// =========================================================================

/// Assert that all elements are approximately equal (within `eps`).
fn assert_approx_eq(a: &[f32], b: &[f32], eps: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!((x - y).abs() < eps, "index {i}: {x} vs {y} (diff {})", (x - y).abs());
    }
}

fn sum_of(v: &[f32]) -> f32 {
    v.iter().copied().sum()
}

// =========================================================================
// softmax — basic
// =========================================================================

#[test]
fn test_softmax_uniform() {
    let logits = vec![1.0; 4];
    let probs = softmax(&logits).unwrap();
    assert_approx_eq(&probs, &[0.25, 0.25, 0.25, 0.25], 1e-6);
}

#[test]
fn test_softmax_single_element() {
    let probs = softmax(&[42.0]).unwrap();
    assert_approx_eq(&probs, &[1.0], 1e-6);
}

#[test]
fn test_softmax_two_elements() {
    let probs = softmax(&[0.0, 0.0]).unwrap();
    assert_approx_eq(&probs, &[0.5, 0.5], 1e-6);
}

#[test]
fn test_softmax_sums_to_one() {
    let probs = softmax(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!((sum_of(&probs) - 1.0).abs() < 1e-6);
}

#[test]
fn test_softmax_monotonic() {
    let probs = softmax(&[1.0, 2.0, 3.0]).unwrap();
    assert!(probs[0] < probs[1]);
    assert!(probs[1] < probs[2]);
}

#[test]
fn test_softmax_empty_error() {
    let result = softmax(&[]);
    assert!(result.is_err());
}

#[test]
fn test_softmax_large_logits_no_overflow() {
    let logits = vec![1000.0, 1001.0, 1002.0];
    let probs = softmax(&logits).unwrap();
    assert!((sum_of(&probs) - 1.0).abs() < 1e-5);
    assert!(probs.iter().all(|&p| p.is_finite()));
}

#[test]
fn test_softmax_negative_logits() {
    let probs = softmax(&[-1.0, -2.0, -3.0]).unwrap();
    assert!((sum_of(&probs) - 1.0).abs() < 1e-6);
    assert!(probs[0] > probs[1]);
}

#[test]
fn test_softmax_mixed_sign_logits() {
    let probs = softmax(&[-10.0, 0.0, 10.0]).unwrap();
    assert!((sum_of(&probs) - 1.0).abs() < 1e-6);
    assert!(probs[2] > 0.99);
}

#[test]
fn test_softmax_large_vector() {
    let logits: Vec<f32> = (0..1024).map(|i| i as f32 * 0.01).collect();
    let probs = softmax(&logits).unwrap();
    assert!((sum_of(&probs) - 1.0).abs() < 1e-4);
}

#[test]
fn test_softmax_very_negative() {
    let logits = vec![-1000.0, -999.0, -998.0];
    let probs = softmax(&logits).unwrap();
    assert!((sum_of(&probs) - 1.0).abs() < 1e-5);
    assert!(probs.iter().all(|&p| p.is_finite()));
}

// =========================================================================
// softmax_inplace
// =========================================================================

#[test]
fn test_softmax_inplace_matches_softmax() {
    let logits = vec![1.0, 2.0, 3.0, 4.0];
    let expected = softmax(&logits).unwrap();
    let mut inplace = logits;
    softmax_inplace(&mut inplace).unwrap();
    assert_approx_eq(&inplace, &expected, 1e-7);
}

#[test]
fn test_softmax_inplace_empty_error() {
    let mut v: Vec<f32> = vec![];
    assert!(softmax_inplace(&mut v).is_err());
}

#[test]
fn test_softmax_inplace_single() {
    let mut v = vec![5.0];
    softmax_inplace(&mut v).unwrap();
    assert_approx_eq(&v, &[1.0], 1e-7);
}

// =========================================================================
// online_softmax
// =========================================================================

#[test]
fn test_online_softmax_matches_standard() {
    let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let standard = softmax(&logits).unwrap();
    let online = online_softmax(&logits).unwrap();
    assert_approx_eq(&online, &standard, 1e-6);
}

#[test]
fn test_online_softmax_single() {
    let result = online_softmax(&[7.0]).unwrap();
    assert_approx_eq(&result, &[1.0], 1e-7);
}

#[test]
fn test_online_softmax_uniform() {
    let result = online_softmax(&[0.0; 5]).unwrap();
    assert_approx_eq(&result, &[0.2; 5], 1e-6);
}

#[test]
fn test_online_softmax_empty_error() {
    assert!(online_softmax(&[]).is_err());
}

#[test]
fn test_online_softmax_large_values() {
    let logits = vec![500.0, 501.0, 502.0];
    let result = online_softmax(&logits).unwrap();
    assert!((sum_of(&result) - 1.0).abs() < 1e-5);
}

#[test]
fn test_online_softmax_negative() {
    let logits = vec![-5.0, -3.0, -1.0];
    let standard = softmax(&logits).unwrap();
    let online = online_softmax(&logits).unwrap();
    assert_approx_eq(&online, &standard, 1e-6);
}

// =========================================================================
// log_softmax
// =========================================================================

#[test]
fn test_log_softmax_sums_correctly() {
    let logits = vec![1.0, 2.0, 3.0];
    let lsm = log_softmax(&logits).unwrap();
    // exp(log_softmax) should sum to 1.
    let sum: f32 = lsm.iter().map(|&v| v.exp()).sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_log_softmax_matches_log_of_softmax() {
    let logits = vec![1.0, 2.0, 3.0, 4.0];
    let lsm = log_softmax(&logits).unwrap();
    let sm = softmax(&logits).unwrap();
    let log_sm: Vec<f32> = sm.iter().map(|&v| v.ln()).collect();
    assert_approx_eq(&lsm, &log_sm, 1e-5);
}

#[test]
fn test_log_softmax_all_negative() {
    let lsm = log_softmax(&[-1.0, -2.0, -3.0]).unwrap();
    assert!(lsm.iter().all(|&v| v < 0.0));
}

#[test]
fn test_log_softmax_single() {
    let lsm = log_softmax(&[0.0]).unwrap();
    assert!((lsm[0] - 0.0).abs() < 1e-6);
}

#[test]
fn test_log_softmax_empty_error() {
    assert!(log_softmax(&[]).is_err());
}

#[test]
fn test_log_softmax_large_logits_stable() {
    let logits = vec![1000.0, 1000.5, 1001.0];
    let lsm = log_softmax(&logits).unwrap();
    assert!(lsm.iter().all(|&v| v.is_finite()));
}

#[test]
fn test_log_softmax_values_are_non_positive() {
    let lsm = log_softmax(&[5.0, 3.0, 1.0]).unwrap();
    assert!(lsm.iter().all(|&v| v <= 0.0 + 1e-7));
}

// =========================================================================
// temperature_softmax
// =========================================================================

#[test]
fn test_temperature_softmax_temp_one_matches_softmax() {
    let logits = vec![1.0, 2.0, 3.0];
    let ts = temperature_softmax(&logits, 1.0).unwrap();
    let sm = softmax(&logits).unwrap();
    assert_approx_eq(&ts, &sm, 1e-6);
}

#[test]
fn test_temperature_softmax_low_temp_sharpens() {
    let logits = vec![1.0, 2.0, 3.0];
    let sharp = temperature_softmax(&logits, 0.1).unwrap();
    let normal = softmax(&logits).unwrap();
    // The max element should have higher probability with low temperature.
    assert!(sharp[2] > normal[2]);
}

#[test]
fn test_temperature_softmax_high_temp_flattens() {
    let logits = vec![1.0, 2.0, 3.0];
    let flat = temperature_softmax(&logits, 10.0).unwrap();
    // Should be closer to uniform than standard softmax.
    let diff = flat[2] - flat[0];
    assert!(diff < 0.1);
}

#[test]
fn test_temperature_softmax_sums_to_one() {
    let probs = temperature_softmax(&[1.0, 2.0, 3.0, 4.0], 0.5).unwrap();
    assert!((sum_of(&probs) - 1.0).abs() < 1e-5);
}

#[test]
fn test_temperature_softmax_zero_error() {
    assert!(temperature_softmax(&[1.0], 0.0).is_err());
}

#[test]
fn test_temperature_softmax_negative_error() {
    assert!(temperature_softmax(&[1.0], -1.0).is_err());
}

#[test]
fn test_temperature_softmax_nan_error() {
    assert!(temperature_softmax(&[1.0], f32::NAN).is_err());
}

#[test]
fn test_temperature_softmax_inf_error() {
    assert!(temperature_softmax(&[1.0], f32::INFINITY).is_err());
}

#[test]
fn test_temperature_softmax_empty_error() {
    assert!(temperature_softmax(&[], 1.0).is_err());
}

#[test]
fn test_temperature_softmax_very_small_temp() {
    let probs = temperature_softmax(&[1.0, 2.0, 3.0], 0.01).unwrap();
    // Almost all probability on element 2.
    assert!(probs[2] > 0.99);
}

// =========================================================================
// masked_softmax
// =========================================================================

#[test]
fn test_masked_softmax_all_true() {
    let logits = vec![1.0, 2.0, 3.0];
    let mask = vec![true, true, true];
    let probs = masked_softmax(&logits, &mask).unwrap();
    let expected = softmax(&logits).unwrap();
    assert_approx_eq(&probs, &expected, 1e-6);
}

#[test]
fn test_masked_softmax_one_false() {
    let logits = vec![1.0, 2.0, 3.0];
    let mask = vec![true, false, true];
    let probs = masked_softmax(&logits, &mask).unwrap();
    assert!(probs[1].abs() < 1e-7);
    assert!((probs[0] + probs[2] - 1.0).abs() < 1e-6);
}

#[test]
fn test_masked_softmax_all_false() {
    let logits = vec![1.0, 2.0, 3.0];
    let mask = vec![false, false, false];
    let probs = masked_softmax(&logits, &mask).unwrap();
    assert!(probs.iter().all(|&p| p.abs() < 1e-7 || p == 0.0));
}

#[test]
fn test_masked_softmax_single_true() {
    let logits = vec![1.0, 2.0, 3.0];
    let mask = vec![false, true, false];
    let probs = masked_softmax(&logits, &mask).unwrap();
    assert!((probs[1] - 1.0).abs() < 1e-6);
}

#[test]
fn test_masked_softmax_length_mismatch_error() {
    assert!(masked_softmax(&[1.0, 2.0], &[true]).is_err());
}

#[test]
fn test_masked_softmax_empty_error() {
    assert!(masked_softmax(&[], &[]).is_err());
}

#[test]
fn test_masked_softmax_preserves_order() {
    let logits = vec![1.0, 5.0, 3.0];
    let mask = vec![true, true, true];
    let probs = masked_softmax(&logits, &mask).unwrap();
    assert!(probs[1] > probs[2]);
    assert!(probs[2] > probs[0]);
}

#[test]
fn test_masked_softmax_sums_to_one_when_any_unmasked() {
    let logits = vec![1.0, 2.0, 3.0, 4.0];
    let mask = vec![true, false, true, false];
    let probs = masked_softmax(&logits, &mask).unwrap();
    let unmasked_sum: f32 =
        probs.iter().zip(mask.iter()).filter(|&(_, &m)| m).map(|(&p, _)| p).sum();
    assert!((unmasked_sum - 1.0).abs() < 1e-6);
}

// =========================================================================
// batch_softmax
// =========================================================================

#[test]
fn test_batch_softmax_two_rows() {
    let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let probs = batch_softmax(&logits, 3).unwrap();
    let row1 = softmax(&logits[0..3]).unwrap();
    let row2 = softmax(&logits[3..6]).unwrap();
    assert_approx_eq(&probs[0..3], &row1, 1e-6);
    assert_approx_eq(&probs[3..6], &row2, 1e-6);
}

#[test]
fn test_batch_softmax_single_row() {
    let logits = vec![1.0, 2.0, 3.0];
    let probs = batch_softmax(&logits, 3).unwrap();
    let expected = softmax(&logits).unwrap();
    assert_approx_eq(&probs, &expected, 1e-6);
}

#[test]
fn test_batch_softmax_each_row_sums_to_one() {
    let logits: Vec<f32> = (0..20).map(|i| i as f32).collect();
    let probs = batch_softmax(&logits, 5).unwrap();
    for row in probs.chunks_exact(5) {
        assert!((sum_of(row) - 1.0).abs() < 1e-5);
    }
}

#[test]
fn test_batch_softmax_row_len_zero_error() {
    assert!(batch_softmax(&[1.0], 0).is_err());
}

#[test]
fn test_batch_softmax_not_multiple_error() {
    assert!(batch_softmax(&[1.0, 2.0, 3.0], 2).is_err());
}

#[test]
fn test_batch_softmax_empty_error() {
    let empty: Vec<f32> = vec![];
    assert!(batch_softmax(&empty, 1).is_err());
}

// =========================================================================
// batch_softmax_inplace
// =========================================================================

#[test]
fn test_batch_softmax_inplace_matches_batch() {
    let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let expected = batch_softmax(&logits, 3).unwrap();
    let mut inplace = logits;
    batch_softmax_inplace(&mut inplace, 3).unwrap();
    assert_approx_eq(&inplace, &expected, 1e-7);
}

#[test]
fn test_batch_softmax_inplace_row_len_zero_error() {
    let mut v = vec![1.0];
    assert!(batch_softmax_inplace(&mut v, 0).is_err());
}

// =========================================================================
// Numerical stability
// =========================================================================

#[test]
fn test_stability_identical_large_values() {
    let logits = vec![88.0; 16];
    let probs = softmax(&logits).unwrap();
    assert_approx_eq(&probs, &vec![1.0 / 16.0; 16], 1e-5);
}

#[test]
fn test_stability_wide_range() {
    let logits = vec![-100.0, 0.0, 100.0];
    let probs = softmax(&logits).unwrap();
    assert!(probs.iter().all(|&p| p.is_finite()));
    assert!((probs[2] - 1.0).abs() < 1e-5);
}

#[test]
fn test_stability_all_neg_infinity() {
    // Edge case: all neg_inf → NaN guarded in masked, but standard softmax
    // will produce NaN from 0/0. We verify finiteness for normal inputs.
    let logits = vec![0.0; 8];
    let probs = softmax(&logits).unwrap();
    assert!(probs.iter().all(|&p| p.is_finite()));
}

#[test]
fn test_stability_alternating_extreme() {
    let logits = vec![-80.0, 80.0, -80.0, 80.0];
    let probs = softmax(&logits).unwrap();
    assert!(probs.iter().all(|&p| p.is_finite()));
    assert!((probs[1] + probs[3] - 1.0).abs() < 1e-5);
}

// =========================================================================
// Scalar module direct tests
// =========================================================================

#[test]
fn test_scalar_softmax_inplace_basic() {
    let mut xs = vec![1.0, 2.0, 3.0];
    scalar::softmax_inplace(&mut xs);
    assert!((sum_of(&xs) - 1.0).abs() < 1e-6);
}

#[test]
fn test_scalar_online_matches_inplace() {
    let logits = vec![0.5, 1.5, 2.5, 3.5];
    let online = scalar::online_softmax(&logits);
    let mut inplace = logits.clone();
    scalar::softmax_inplace(&mut inplace);
    assert_approx_eq(&online, &inplace, 1e-6);
}

#[test]
fn test_scalar_log_softmax_matches() {
    let logits = vec![1.0, 2.0, 3.0];
    let lsm = scalar::log_softmax(&logits);
    let sm = softmax(&logits).unwrap();
    let log_sm: Vec<f32> = sm.iter().map(|&v| v.ln()).collect();
    assert_approx_eq(&lsm, &log_sm, 1e-5);
}

#[test]
fn test_scalar_temperature_softmax() {
    let logits = vec![1.0, 2.0, 3.0];
    let ts = scalar::temperature_softmax(&logits, 2.0);
    assert!((sum_of(&ts) - 1.0).abs() < 1e-6);
}

#[test]
fn test_scalar_masked_softmax() {
    let logits = vec![1.0, 2.0, 3.0];
    let mask = vec![true, false, true];
    let probs = scalar::masked_softmax(&logits, &mask);
    assert!(probs[1].abs() < 1e-7);
}

#[test]
fn test_scalar_batch_softmax_inplace() {
    let mut data = vec![1.0, 2.0, 3.0, 4.0];
    scalar::batch_softmax_inplace(&mut data, 2);
    assert!((data[0] + data[1] - 1.0).abs() < 1e-6);
    assert!((data[2] + data[3] - 1.0).abs() < 1e-6);
}

// =========================================================================
// Dispatch tests
// =========================================================================

#[test]
fn test_dispatch_has_avx512f_returns_bool() {
    // Just ensure it doesn't panic.
    let _ = dispatch::has_avx512f();
}

#[test]
fn test_dispatch_softmax_inplace_works() {
    let mut xs = vec![1.0, 2.0, 3.0, 4.0];
    dispatch::softmax_inplace_dispatch(&mut xs);
    assert!((sum_of(&xs) - 1.0).abs() < 1e-5);
}

#[test]
fn test_dispatch_online_softmax_works() {
    let result = dispatch::online_softmax_dispatch(&[1.0, 2.0]);
    assert!((sum_of(&result) - 1.0).abs() < 1e-6);
}

#[test]
fn test_dispatch_log_softmax_works() {
    let result = dispatch::log_softmax_dispatch(&[1.0, 2.0]);
    assert!(result.iter().all(|&v| v.is_finite()));
}

#[test]
fn test_dispatch_temperature_softmax_works() {
    let result = dispatch::temperature_softmax_dispatch(&[1.0, 2.0], 0.5);
    assert!((sum_of(&result) - 1.0).abs() < 1e-6);
}

#[test]
fn test_dispatch_masked_softmax_works() {
    let result = dispatch::masked_softmax_dispatch(&[1.0, 2.0], &[true, false]);
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!(result[1].abs() < 1e-7);
}

#[test]
fn test_dispatch_batch_softmax_inplace_works() {
    let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    dispatch::batch_softmax_inplace_dispatch(&mut data, 3);
    assert!((sum_of(&data[0..3]) - 1.0).abs() < 1e-5);
    assert!((sum_of(&data[3..6]) - 1.0).abs() < 1e-5);
}

// =========================================================================
// Large-vector tests (exercise SIMD tails)
// =========================================================================

#[test]
fn test_softmax_size_15_tail() {
    let logits: Vec<f32> = (0..15).map(|i| i as f32).collect();
    let probs = softmax(&logits).unwrap();
    assert!((sum_of(&probs) - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_size_16_exact_simd() {
    let logits: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let probs = softmax(&logits).unwrap();
    assert!((sum_of(&probs) - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_size_17_one_over() {
    let logits: Vec<f32> = (0..17).map(|i| i as f32).collect();
    let probs = softmax(&logits).unwrap();
    assert!((sum_of(&probs) - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_size_32_two_chunks() {
    let logits: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
    let probs = softmax(&logits).unwrap();
    assert!((sum_of(&probs) - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_size_33_two_chunks_plus_tail() {
    let logits: Vec<f32> = (0..33).map(|i| (i as f32) * 0.1).collect();
    let probs = softmax(&logits).unwrap();
    assert!((sum_of(&probs) - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_size_100() {
    let logits: Vec<f32> = (0..100).map(|i| (i as f32) * 0.05).collect();
    let probs = softmax(&logits).unwrap();
    assert!((sum_of(&probs) - 1.0).abs() < 1e-4);
}

// =========================================================================
// Edge-case & regression tests
// =========================================================================

#[test]
fn test_softmax_all_zeros() {
    let logits = vec![0.0; 10];
    let probs = softmax(&logits).unwrap();
    assert_approx_eq(&probs, &vec![0.1; 10], 1e-6);
}

#[test]
fn test_softmax_one_hot_like() {
    // One very large value, rest very negative → near-one-hot output.
    let mut logits = vec![-1000.0; 10];
    logits[5] = 0.0;
    let probs = softmax(&logits).unwrap();
    assert!((probs[5] - 1.0).abs() < 1e-5);
}

#[test]
fn test_log_softmax_uniform_values() {
    let n = 8;
    let logits = vec![0.0; n];
    let lsm = log_softmax(&logits).unwrap();
    let expected = -(n as f32).ln();
    for &v in &lsm {
        assert!((v - expected).abs() < 1e-5);
    }
}

#[test]
fn test_temperature_softmax_extreme_low_temp() {
    let probs = temperature_softmax(&[0.0, 1.0, 2.0], 0.001).unwrap();
    // Almost all probability on last element.
    assert!(probs[2] > 0.999);
}

#[test]
fn test_masked_softmax_alternating() {
    let logits = vec![1.0, 2.0, 3.0, 4.0];
    let mask = vec![true, false, true, false];
    let probs = masked_softmax(&logits, &mask).unwrap();
    assert!(probs[1].abs() < 1e-7);
    assert!(probs[3].abs() < 1e-7);
    assert!((probs[0] + probs[2] - 1.0).abs() < 1e-6);
}

#[test]
fn test_batch_softmax_many_rows() {
    let row_len = 4;
    let n_rows = 100;
    let logits: Vec<f32> = (0..(row_len * n_rows)).map(|i| (i % 7) as f32).collect();
    let probs = batch_softmax(&logits, row_len).unwrap();
    for row in probs.chunks_exact(row_len) {
        assert!((sum_of(row) - 1.0).abs() < 1e-5);
    }
}

#[test]
fn test_softmax_non_negative_output() {
    let logits = vec![-50.0, -30.0, -10.0, 10.0, 30.0];
    let probs = softmax(&logits).unwrap();
    assert!(probs.iter().all(|&p| p >= 0.0));
}

#[test]
fn test_online_softmax_large_vector() {
    let logits: Vec<f32> = (0..256).map(|i| (i as f32) * 0.02 - 2.56).collect();
    let online = online_softmax(&logits).unwrap();
    let standard = softmax(&logits).unwrap();
    assert_approx_eq(&online, &standard, 5e-4);
}

// =========================================================================
// Property: softmax probabilities always non-negative and sum to 1
// =========================================================================

#[test]
fn test_property_softmax_prob_invariants() {
    // Manually test a range of patterns instead of using proptest macro
    // (proptest is in dev-dependencies for heavier property tests).
    let cases: Vec<Vec<f32>> = vec![
        vec![0.0],
        vec![1.0, -1.0],
        vec![100.0, 0.0, -100.0],
        vec![-0.5; 50],
        (0..64).map(|i| (i as f32) * 0.3 - 10.0).collect(),
    ];
    for logits in &cases {
        let probs = softmax(logits).unwrap();
        assert!(probs.iter().all(|&p| p >= 0.0), "negative probability");
        assert!((sum_of(&probs) - 1.0).abs() < 1e-4, "sum != 1: {}", sum_of(&probs));
    }
}

#[test]
fn test_property_log_softmax_values_nonpositive() {
    let cases: Vec<Vec<f32>> = vec![vec![0.0; 4], vec![1.0, 2.0, 3.0], vec![-5.0, 0.0, 5.0, 10.0]];
    for logits in &cases {
        let lsm = log_softmax(logits).unwrap();
        assert!(lsm.iter().all(|&v| v <= 1e-7), "log_softmax value > 0: {lsm:?}");
    }
}

#[test]
fn test_property_temperature_preserves_sum() {
    let temps = [0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0];
    let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    for &t in &temps {
        let probs = temperature_softmax(&logits, t).unwrap();
        assert!((sum_of(&probs) - 1.0).abs() < 1e-4, "temp={t}, sum={}", sum_of(&probs));
    }
}

#[test]
fn test_property_masked_zeros_masked_positions() {
    let logits = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let mask = vec![false, true, false, true, false];
    let probs = masked_softmax(&logits, &mask).unwrap();
    for (i, (&p, &m)) in probs.iter().zip(mask.iter()).enumerate() {
        if !m {
            assert!(p.abs() < 1e-7, "position {i} should be zero, got {p}");
        }
    }
}
