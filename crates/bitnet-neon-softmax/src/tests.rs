//! Comprehensive tests for `bitnet-neon-softmax`.

use crate::*;

// ── Test helpers ───────────────────────────────────────────────────────────

fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!((x - y).abs() < tol, "element {i}: {x} vs {y} (diff={})", (x - y).abs());
    }
}

fn sums_to_one(v: &[f32], tol: f32) {
    let s: f32 = v.iter().sum();
    assert!((s - 1.0).abs() < tol, "sum = {s}, expected 1.0");
}

fn all_non_negative(v: &[f32]) {
    for (i, &x) in v.iter().enumerate() {
        assert!(x >= 0.0, "element {i} is negative: {x}");
    }
}

fn reference_softmax(input: &[f32]) -> Vec<f32> {
    let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&v| v / sum).collect()
}

fn reference_log_softmax(input: &[f32]) -> Vec<f32> {
    let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = input.iter().map(|&v| (v - max).exp()).sum::<f32>().ln() + max;
    input.iter().map(|&v| v - lse).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// softmax – basic correctness
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn softmax_empty() {
    assert!(softmax(&[]).is_empty());
}

#[test]
fn softmax_single_element() {
    let out = softmax(&[42.0]);
    approx_eq(&out, &[1.0], 1e-6);
}

#[test]
fn softmax_two_elements() {
    let out = softmax(&[0.0, 0.0]);
    approx_eq(&out, &[0.5, 0.5], 1e-6);
}

#[test]
fn softmax_known_values() {
    let input = [1.0, 2.0, 3.0];
    let expected = reference_softmax(&input);
    let out = softmax(&input);
    approx_eq(&out, &expected, 1e-6);
}

#[test]
fn softmax_sums_to_one() {
    let out = softmax(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    sums_to_one(&out, 1e-6);
}

#[test]
fn softmax_all_non_negative() {
    let out = softmax(&[-10.0, -5.0, 0.0, 5.0, 10.0]);
    all_non_negative(&out);
}

#[test]
fn softmax_uniform_input() {
    let input = vec![3.15; 8];
    let out = softmax(&input);
    let expected = 1.0 / 8.0;
    for &v in &out {
        assert!((v - expected).abs() < 1e-6);
    }
}

#[test]
fn softmax_large_positive_values() {
    let input = [1000.0, 1001.0, 1002.0];
    let out = softmax(&input);
    sums_to_one(&out, 1e-5);
    all_non_negative(&out);
}

#[test]
fn softmax_large_negative_values() {
    let input = [-1000.0, -999.0, -998.0];
    let out = softmax(&input);
    sums_to_one(&out, 1e-5);
    all_non_negative(&out);
}

#[test]
fn softmax_mixed_extreme_values() {
    let input = [-1000.0, 0.0, 1000.0];
    let out = softmax(&input);
    sums_to_one(&out, 1e-5);
    // The largest element should dominate.
    assert!(out[2] > 0.99);
}

#[test]
fn softmax_preserves_ordering() {
    let input = [1.0, 3.0, 2.0, 5.0, 4.0];
    let out = softmax(&input);
    assert!(out[3] > out[4]);
    assert!(out[4] > out[1]);
    assert!(out[1] > out[2]);
    assert!(out[2] > out[0]);
}

#[test]
fn softmax_length_not_multiple_of_4() {
    for n in 1..=17 {
        let input: Vec<f32> = (0..n)
            .map(|i| {
                #[expect(clippy::cast_precision_loss)]
                {
                    i as f32
                }
            })
            .collect();
        let out = softmax(&input);
        assert_eq!(out.len(), n);
        sums_to_one(&out, 1e-5);
    }
}

#[test]
fn softmax_length_exact_multiple_of_4() {
    for n in [4, 8, 12, 16, 64] {
        let input: Vec<f32> = (0..n)
            .map(|i| {
                #[expect(clippy::cast_precision_loss)]
                {
                    i as f32 * 0.1
                }
            })
            .collect();
        let out = softmax(&input);
        assert_eq!(out.len(), n);
        sums_to_one(&out, 1e-5);
    }
}

#[test]
fn softmax_negative_infinity_elements() {
    let input = [f32::NEG_INFINITY, 0.0, 1.0];
    let out = softmax(&input);
    assert!((out[0] - 0.0).abs() < 1e-6);
    sums_to_one(&out, 1e-5);
}

// ═══════════════════════════════════════════════════════════════════════════
// softmax_inplace
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn softmax_inplace_matches_softmax() {
    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let expected = softmax(&input);
    let mut buf = input.to_vec();
    softmax_inplace(&mut buf);
    approx_eq(&buf, &expected, 1e-7);
}

#[test]
fn softmax_inplace_empty() {
    let mut buf: Vec<f32> = vec![];
    softmax_inplace(&mut buf);
    assert!(buf.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════
// log_softmax
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn log_softmax_empty() {
    assert!(log_softmax(&[]).is_empty());
}

#[test]
fn log_softmax_single() {
    let out = log_softmax(&[5.0]);
    approx_eq(&out, &[0.0], 1e-6);
}

#[test]
fn log_softmax_known_values() {
    let input = [1.0, 2.0, 3.0];
    let expected = reference_log_softmax(&input);
    let out = log_softmax(&input);
    approx_eq(&out, &expected, 1e-5);
}

#[test]
fn log_softmax_all_negative() {
    let out = log_softmax(&[-10.0, -5.0, 0.0, 5.0, 10.0]);
    // All values should be ≤ 0.
    for &v in &out {
        assert!(v <= 1e-6, "log_softmax element should be ≤ 0: {v}");
    }
}

#[test]
fn log_softmax_exp_matches_softmax() {
    let input = [1.0, 2.0, 3.0, 4.0, 5.0];
    let ls = log_softmax(&input);
    let s = softmax(&input);
    let exp_ls: Vec<f32> = ls.iter().map(|&v| v.exp()).collect();
    approx_eq(&exp_ls, &s, 1e-5);
}

#[test]
fn log_softmax_large_values() {
    let input = [500.0, 501.0, 502.0];
    let out = log_softmax(&input);
    let expected = reference_log_softmax(&input);
    approx_eq(&out, &expected, 1e-4);
}

#[test]
fn log_softmax_sum_exp_is_one() {
    let input = [0.1, 0.2, 0.3, 0.4];
    let out = log_softmax(&input);
    let sum: f32 = out.iter().map(|&v| v.exp()).sum();
    assert!((sum - 1.0).abs() < 1e-5, "exp(log_softmax) sum = {sum}");
}

#[test]
fn log_softmax_various_lengths() {
    for n in 1..=20 {
        let input: Vec<f32> = (0..n)
            .map(|i| {
                #[expect(clippy::cast_precision_loss)]
                {
                    i as f32
                }
            })
            .collect();
        let out = log_softmax(&input);
        assert_eq!(out.len(), n);
        for &v in &out {
            assert!(v <= 1e-6);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// temperature_softmax
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn temperature_softmax_empty() {
    assert!(temperature_softmax(&[], 1.0).is_empty());
}

#[test]
fn temperature_1_equals_softmax() {
    let input = [1.0, 2.0, 3.0, 4.0];
    let s = softmax(&input);
    let ts = temperature_softmax(&input, 1.0);
    approx_eq(&ts, &s, 1e-6);
}

#[test]
fn high_temperature_flattens() {
    let input = [0.0, 10.0];
    let ts = temperature_softmax(&input, 100.0);
    // High temp → more uniform.
    assert!((ts[0] - ts[1]).abs() < 0.2);
}

#[test]
fn low_temperature_sharpens() {
    let input = [0.0, 1.0, 2.0];
    let ts = temperature_softmax(&input, 0.01);
    // Low temp → winner-take-all.
    assert!(ts[2] > 0.99);
}

#[test]
fn temperature_softmax_sums_to_one() {
    let input = [1.0, 2.0, 3.0, 4.0, 5.0];
    for temp in [0.1, 0.5, 1.0, 2.0, 10.0] {
        let ts = temperature_softmax(&input, temp);
        sums_to_one(&ts, 1e-5);
    }
}

#[test]
#[should_panic(expected = "temperature must be positive")]
fn temperature_softmax_zero_panics() {
    let _ = temperature_softmax(&[1.0], 0.0);
}

#[test]
#[should_panic(expected = "temperature must be positive")]
fn temperature_softmax_negative_panics() {
    let _ = temperature_softmax(&[1.0], -1.0);
}

#[test]
#[should_panic(expected = "temperature must be positive")]
fn temperature_softmax_nan_panics() {
    let _ = temperature_softmax(&[1.0], f32::NAN);
}

#[test]
#[should_panic(expected = "temperature must be positive")]
fn temperature_softmax_inf_panics() {
    let _ = temperature_softmax(&[1.0], f32::INFINITY);
}

// ═══════════════════════════════════════════════════════════════════════════
// online_softmax
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn online_softmax_empty() {
    assert!(online_softmax(&[]).is_empty());
}

#[test]
fn online_softmax_single() {
    let out = online_softmax(&[42.0]);
    approx_eq(&out, &[1.0], 1e-6);
}

#[test]
fn online_softmax_matches_standard() {
    let input = [1.0, 2.0, 3.0, 4.0, 5.0];
    let expected = softmax(&input);
    let out = online_softmax(&input);
    approx_eq(&out, &expected, 1e-5);
}

#[test]
fn online_softmax_large_values() {
    let input = [500.0, 501.0, 502.0];
    let expected = softmax(&input);
    let out = online_softmax(&input);
    approx_eq(&out, &expected, 1e-5);
}

#[test]
fn online_softmax_negative_values() {
    let input = [-100.0, -50.0, 0.0];
    let expected = softmax(&input);
    let out = online_softmax(&input);
    approx_eq(&out, &expected, 1e-5);
}

#[test]
fn online_softmax_sums_to_one() {
    let input = [0.5, 1.5, 2.5, 3.5];
    let out = online_softmax(&input);
    sums_to_one(&out, 1e-5);
}

#[test]
fn online_softmax_various_lengths() {
    for n in 1..=20 {
        let input: Vec<f32> = (0..n)
            .map(|i| {
                #[expect(clippy::cast_precision_loss)]
                {
                    i as f32 * 0.5
                }
            })
            .collect();
        let expected = softmax(&input);
        let out = online_softmax(&input);
        approx_eq(&out, &expected, 1e-5);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// batch_softmax
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn batch_softmax_single_row() {
    let input = [1.0, 2.0, 3.0];
    let out = batch_softmax(&input, 3);
    approx_eq(&out, &softmax(&input), 1e-6);
}

#[test]
fn batch_softmax_two_rows() {
    let row0 = [1.0, 2.0, 3.0];
    let row1 = [4.0, 5.0, 6.0];
    let mut data = Vec::new();
    data.extend_from_slice(&row0);
    data.extend_from_slice(&row1);

    let out = batch_softmax(&data, 3);
    let expected0 = softmax(&row0);
    let expected1 = softmax(&row1);

    approx_eq(&out[0..3], &expected0, 1e-6);
    approx_eq(&out[3..6], &expected1, 1e-6);
}

#[test]
fn batch_softmax_inplace_matches() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let expected = batch_softmax(&data, 3);
    let mut buf = data;
    batch_softmax_inplace(&mut buf, 3);
    approx_eq(&buf, &expected, 1e-7);
}

#[test]
#[should_panic(expected = "cols must be positive")]
fn batch_softmax_zero_cols_panics() {
    batch_softmax_inplace(&mut [1.0, 2.0], 0);
}

#[test]
#[should_panic(expected = "not divisible")]
fn batch_softmax_misaligned_panics() {
    let _ = batch_softmax(&[1.0, 2.0, 3.0], 2);
}

#[test]
fn batch_softmax_many_rows() {
    let cols = 5;
    let rows = 10;
    let data: Vec<f32> = (0..(rows * cols))
        .map(|i| {
            #[expect(clippy::cast_precision_loss)]
            {
                i as f32 * 0.1
            }
        })
        .collect();
    let out = batch_softmax(&data, cols);
    for r in 0..rows {
        let row = &out[r * cols..(r + 1) * cols];
        sums_to_one(row, 1e-5);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// batch_log_softmax
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn batch_log_softmax_single_row() {
    let input = [1.0, 2.0, 3.0, 4.0];
    let out = batch_log_softmax(&input, 4);
    approx_eq(&out, &log_softmax(&input), 1e-5);
}

#[test]
fn batch_log_softmax_two_rows() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let out = batch_log_softmax(&data, 3);
    let e0 = log_softmax(&data[0..3]);
    let e1 = log_softmax(&data[3..6]);
    approx_eq(&out[0..3], &e0, 1e-5);
    approx_eq(&out[3..6], &e1, 1e-5);
}

#[test]
#[should_panic(expected = "cols must be positive")]
fn batch_log_softmax_zero_cols_panics() {
    let _ = batch_log_softmax(&[1.0], 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// batch_temperature_softmax
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn batch_temperature_softmax_single_row() {
    let input = [1.0, 2.0, 3.0];
    let out = batch_temperature_softmax(&input, 3, 0.5);
    approx_eq(&out, &temperature_softmax(&input, 0.5), 1e-6);
}

#[test]
fn batch_temperature_softmax_two_rows() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let out = batch_temperature_softmax(&data, 3, 2.0);
    let e0 = temperature_softmax(&data[0..3], 2.0);
    let e1 = temperature_softmax(&data[3..6], 2.0);
    approx_eq(&out[0..3], &e0, 1e-6);
    approx_eq(&out[3..6], &e1, 1e-6);
}

#[test]
#[should_panic(expected = "temperature must be positive")]
fn batch_temperature_softmax_zero_temp_panics() {
    let _ = batch_temperature_softmax(&[1.0, 2.0], 2, 0.0);
}

#[test]
#[should_panic(expected = "not divisible")]
fn batch_temperature_softmax_misaligned_panics() {
    let _ = batch_temperature_softmax(&[1.0, 2.0, 3.0], 2, 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Numerical stability
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn numerical_stability_identical_large() {
    let input = vec![1e30; 16];
    let out = softmax(&input);
    let expected = 1.0 / 16.0;
    for &v in &out {
        assert!((v - expected).abs() < 1e-6, "got {v}");
    }
}

#[test]
fn numerical_stability_wide_range() {
    let input = [-1e6, 0.0, 1e6];
    let out = softmax(&input);
    sums_to_one(&out, 1e-5);
    assert!(out[2] > 0.99);
}

#[test]
fn numerical_stability_all_negative_infinity() {
    // When all inputs are -inf, exp(-inf - (-inf)) = exp(NaN) which is NaN.
    // This is a degenerate case; we just check it doesn't panic.
    let input = [f32::NEG_INFINITY; 4];
    let _out = softmax(&input);
}

#[test]
fn numerical_stability_log_softmax_large() {
    let input = [1000.0, 1001.0, 1002.0];
    let out = log_softmax(&input);
    // Should not produce NaN or Inf.
    for &v in &out {
        assert!(v.is_finite(), "log_softmax produced non-finite: {v}");
    }
}

#[test]
fn numerical_stability_subnormal_inputs() {
    let input = [f32::MIN_POSITIVE, f32::MIN_POSITIVE * 2.0, f32::MIN_POSITIVE * 3.0];
    let out = softmax(&input);
    sums_to_one(&out, 1e-5);
}

// ═══════════════════════════════════════════════════════════════════════════
// NaN handling
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn nan_in_softmax_propagates() {
    let input = [1.0, f32::NAN, 3.0];
    let out = softmax(&input);
    // NaN propagates through max / exp / sum → at least one NaN expected.
    assert!(out.iter().any(|v| v.is_nan()), "expected NaN propagation");
}

#[test]
fn nan_in_log_softmax_propagates() {
    let input = [1.0, f32::NAN, 3.0];
    let out = log_softmax(&input);
    assert!(out.iter().any(|v| v.is_nan()));
}

#[test]
fn nan_in_online_softmax_propagates() {
    let input = [1.0, f32::NAN, 3.0];
    let out = online_softmax(&input);
    assert!(out.iter().any(|v| v.is_nan()));
}

// ═══════════════════════════════════════════════════════════════════════════
// Scalar backend direct tests (always available)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn scalar_softmax_matches_reference() {
    let input = [1.0, 2.0, 3.0, 4.0, 5.0];
    let expected = reference_softmax(&input);
    let mut buf = input.to_vec();
    crate::scalar::softmax_inplace(&mut buf);
    approx_eq(&buf, &expected, 1e-6);
}

#[test]
fn scalar_log_softmax_matches_reference() {
    let input = [1.0, 2.0, 3.0, 4.0, 5.0];
    let expected = reference_log_softmax(&input);
    let out = crate::scalar::log_softmax(&input);
    approx_eq(&out, &expected, 1e-5);
}

#[test]
fn scalar_temperature_softmax_correctness() {
    let input = [1.0, 2.0, 3.0];
    let out = crate::scalar::temperature_softmax(&input, 0.5);
    sums_to_one(&out, 1e-5);
}

#[test]
fn scalar_online_softmax_matches_reference() {
    let input = [1.0, 2.0, 3.0, 4.0, 5.0];
    let expected = reference_softmax(&input);
    let out = crate::scalar::online_softmax(&input);
    approx_eq(&out, &expected, 1e-5);
}

// ═══════════════════════════════════════════════════════════════════════════
// Edge cases & regression guards
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn softmax_length_one_always_one() {
    for v in [-100.0, 0.0, 100.0, f32::MIN_POSITIVE] {
        let out = softmax(&[v]);
        approx_eq(&out, &[1.0], 1e-6);
    }
}

#[test]
fn softmax_identical_elements_uniform() {
    for n in [2, 5, 8, 13, 16] {
        let input = vec![7.0; n];
        let out = softmax(&input);
        let expected = {
            #[expect(clippy::cast_precision_loss)]
            {
                1.0 / n as f32
            }
        };
        for &v in &out {
            assert!((v - expected).abs() < 1e-6);
        }
    }
}

#[test]
fn log_softmax_identical_elements() {
    let n = 10;
    let input = vec![0.0; n];
    let out = log_softmax(&input);
    let expected = {
        #[expect(clippy::cast_precision_loss)]
        {
            -(n as f32).ln()
        }
    };
    for &v in &out {
        assert!((v - expected).abs() < 1e-5, "got {v}, expected {expected}");
    }
}

#[test]
fn online_softmax_descending_order() {
    let input = [5.0, 4.0, 3.0, 2.0, 1.0];
    let expected = softmax(&input);
    let out = online_softmax(&input);
    approx_eq(&out, &expected, 1e-5);
}

#[test]
fn online_softmax_ascending_order() {
    let input = [1.0, 2.0, 3.0, 4.0, 5.0];
    let expected = softmax(&input);
    let out = online_softmax(&input);
    approx_eq(&out, &expected, 1e-5);
}

#[test]
fn temperature_softmax_preserves_ordering() {
    let input = [1.0, 3.0, 2.0];
    let out = temperature_softmax(&input, 0.5);
    assert!(out[1] > out[2]);
    assert!(out[2] > out[0]);
}

#[test]
fn batch_softmax_empty_data() {
    let out = batch_softmax(&[], 1);
    assert!(out.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════
// proptest – property-based tests
// ═══════════════════════════════════════════════════════════════════════════

mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn finite_f32() -> impl Strategy<Value = f32> {
        -1e6_f32..1e6_f32
    }

    fn finite_vec(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(finite_f32(), min_len..=max_len)
    }

    proptest! {
        #[test]
        fn prop_softmax_sums_to_one(input in finite_vec(1, 128)) {
            let out = softmax(&input);
            let sum: f32 = out.iter().sum();
            prop_assert!((sum - 1.0).abs() < 1e-4, "sum = {sum}");
        }

        #[test]
        fn prop_softmax_non_negative(input in finite_vec(1, 128)) {
            let out = softmax(&input);
            for &v in &out {
                prop_assert!(v >= 0.0, "negative value: {v}");
            }
        }

        #[test]
        fn prop_softmax_preserves_length(input in finite_vec(0, 128)) {
            let out = softmax(&input);
            prop_assert_eq!(out.len(), input.len());
        }

        #[test]
        fn prop_log_softmax_all_non_positive(input in finite_vec(1, 128)) {
            let out = log_softmax(&input);
            for &v in &out {
                prop_assert!(v <= 1e-5, "positive log_softmax: {v}");
            }
        }

        #[test]
        fn prop_log_softmax_exp_sums_to_one(input in finite_vec(1, 64)) {
            let out = log_softmax(&input);
            let sum: f32 = out.iter().map(|&v| v.exp()).sum();
            prop_assert!((sum - 1.0).abs() < 1e-3, "sum = {sum}");
        }

        #[test]
        fn prop_online_matches_standard(input in finite_vec(1, 128)) {
            let standard = softmax(&input);
            let online = online_softmax(&input);
            for (i, (&a, &b)) in standard.iter().zip(online.iter()).enumerate() {
                prop_assert!(
                    (a - b).abs() < 1e-4,
                    "mismatch at {i}: standard={a} online={b}"
                );
            }
        }

        #[test]
        fn prop_temperature_1_equals_softmax(input in finite_vec(1, 64)) {
            let s = softmax(&input);
            let ts = temperature_softmax(&input, 1.0);
            for (i, (&a, &b)) in s.iter().zip(ts.iter()).enumerate() {
                prop_assert!(
                    (a - b).abs() < 1e-5,
                    "mismatch at {i}: softmax={a} temp_softmax={b}"
                );
            }
        }

        #[test]
        fn prop_batch_rows_independent(
            row_a in finite_vec(4, 4),
            row_b in finite_vec(4, 4),
        ) {
            let mut data = Vec::new();
            data.extend_from_slice(&row_a);
            data.extend_from_slice(&row_b);
            let out = batch_softmax(&data, 4);

            let sa = softmax(&row_a);
            let sb = softmax(&row_b);
            for i in 0..4 {
                prop_assert!((out[i] - sa[i]).abs() < 1e-5);
                prop_assert!((out[4 + i] - sb[i]).abs() < 1e-5);
            }
        }

        #[test]
        fn prop_softmax_max_element_has_max_probability(input in finite_vec(2, 64)) {
            let out = softmax(&input);
            let max_idx = input
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
            let max_prob = out[max_idx];
            for (i, &p) in out.iter().enumerate() {
                if i != max_idx {
                    // Due to ties, use >=.
                    prop_assert!(max_prob >= p - 1e-6,
                        "max element at {max_idx} (prob={max_prob}) < element {i} (prob={p})");
                }
            }
        }
    }
}
