//! Unit tests for numerical accuracy comparison
//!
//! These tests verify the `compare_numerical_accuracy` function implementation
//! for cross-validation between Rust and C++ inference implementations.
#[cfg(feature = "crossval")]
struct NumericalComparison {
    within_tolerance: bool,
    max_difference: f32,
    rmse: f32,
}
#[cfg(feature = "crossval")]
fn compare_numerical_accuracy(
    rust_logits: &[f32],
    cpp_logits: &[f32],
    tolerance: f32,
) -> NumericalComparison {
    assert_eq!(
        rust_logits.len(),
        cpp_logits.len(),
        "Logit arrays must have the same length for comparison"
    );
    if rust_logits.is_empty() {
        return NumericalComparison { within_tolerance: true, max_difference: 0.0, rmse: 0.0 };
    }
    let mut max_diff = 0.0_f32;
    let mut sum_squared_error = 0.0_f32;
    for (rust_val, cpp_val) in rust_logits.iter().zip(cpp_logits.iter()) {
        let abs_diff = (rust_val - cpp_val).abs();
        max_diff = max_diff.max(abs_diff);
        sum_squared_error += abs_diff * abs_diff;
    }
    let rmse = (sum_squared_error / rust_logits.len() as f32).sqrt();
    let within_tolerance = max_diff <= tolerance;
    NumericalComparison { within_tolerance, max_difference: max_diff, rmse }
}
#[cfg(all(test, feature = "crossval"))]
mod tests {
    use super::*;
    #[test]
    fn test_compare_numerical_accuracy_exact_match() {
        let rust_logits = vec![1.0, 2.0, 3.0, 4.0];
        let cpp_logits = vec![1.0, 2.0, 3.0, 4.0];
        let result = compare_numerical_accuracy(&rust_logits, &cpp_logits, 1e-6);
        assert!(result.within_tolerance, "Exact match should be within tolerance");
        assert_eq!(result.max_difference, 0.0, "Max difference should be 0.0");
        assert_eq!(result.rmse, 0.0, "RMSE should be 0.0");
    }
    #[test]
    fn test_compare_numerical_accuracy_small_difference_within_tolerance() {
        let rust_logits = vec![1.0, 2.0, 3.0, 4.0];
        let cpp_logits = vec![1.00001, 2.00001, 3.00001, 4.00001];
        let result = compare_numerical_accuracy(&rust_logits, &cpp_logits, 1e-4);
        assert!(result.within_tolerance, "Small differences should be within tolerance");
        assert!(result.max_difference < 1e-4, "Max difference should be < 1e-4");
        assert!(result.rmse < 1e-4, "RMSE should be < 1e-4");
    }
    #[test]
    fn test_compare_numerical_accuracy_large_difference_exceeds_tolerance() {
        let rust_logits = vec![1.0, 2.0, 3.0, 4.0];
        let cpp_logits = vec![1.1, 2.1, 3.1, 4.1];
        let result = compare_numerical_accuracy(&rust_logits, &cpp_logits, 1e-4);
        assert!(!result.within_tolerance, "Large differences should exceed tolerance");
        assert!((result.max_difference - 0.1).abs() < 1e-6, "Max difference should be ~0.1");
        assert!(result.rmse > 0.0, "RMSE should be > 0.0");
    }
    #[test]
    fn test_compare_numerical_accuracy_empty_arrays() {
        let rust_logits: Vec<f32> = vec![];
        let cpp_logits: Vec<f32> = vec![];
        let result = compare_numerical_accuracy(&rust_logits, &cpp_logits, 1e-4);
        assert!(result.within_tolerance, "Empty arrays should be within tolerance");
        assert_eq!(result.max_difference, 0.0, "Max difference should be 0.0");
        assert_eq!(result.rmse, 0.0, "RMSE should be 0.0");
    }
    #[test]
    fn test_compare_numerical_accuracy_rmse_calculation() {
        let rust_logits = vec![0.0, 0.0, 0.0, 0.0];
        let cpp_logits = vec![1.0, 1.0, 1.0, 1.0];
        let result = compare_numerical_accuracy(&rust_logits, &cpp_logits, 2.0);
        assert!((result.rmse - 1.0).abs() < 1e-6, "RMSE should be 1.0");
        assert!(result.within_tolerance, "Should be within tolerance of 2.0");
        assert_eq!(result.max_difference, 1.0, "Max difference should be 1.0");
    }
    #[test]
    #[should_panic(expected = "Logit arrays must have the same length for comparison")]
    fn test_compare_numerical_accuracy_mismatched_lengths() {
        let rust_logits = vec![1.0, 2.0, 3.0];
        let cpp_logits = vec![1.0, 2.0, 3.0, 4.0];
        compare_numerical_accuracy(&rust_logits, &cpp_logits, 1e-4);
    }
    #[test]
    fn test_compare_numerical_accuracy_mixed_positive_negative() {
        let rust_logits = vec![-1.0, 2.0, -3.0, 4.0];
        let cpp_logits = vec![-1.1, 2.1, -3.1, 4.1];
        let result = compare_numerical_accuracy(&rust_logits, &cpp_logits, 0.15);
        assert!(result.within_tolerance, "Should be within tolerance of 0.15");
        assert!((result.max_difference - 0.1).abs() < 1e-6, "Max difference should be ~0.1");
    }
    #[test]
    fn test_compare_numerical_accuracy_typical_inference_values() {
        let rust_logits = vec![-5.2, 3.1, 8.4, -2.7, 0.9, 12.3, -0.5, 4.8, 6.1, -8.9];
        let cpp_logits = vec![
            -5.20001, 3.10002, 8.39998, -2.70001, 0.90001, 12.29999, -0.50002, 4.79999, 6.10001,
            -8.90001,
        ];
        let result = compare_numerical_accuracy(&rust_logits, &cpp_logits, 1e-3);
        assert!(result.within_tolerance, "Should be within typical numerical precision");
        assert!(result.max_difference < 1e-3, "Max difference should be < 1e-3");
        assert!(result.rmse < 1e-4, "RMSE should be very small");
    }
    #[test]
    fn test_compare_numerical_accuracy_single_element() {
        let rust_logits = vec![42.0];
        let cpp_logits = vec![42.0001];
        let result = compare_numerical_accuracy(&rust_logits, &cpp_logits, 1e-3);
        assert!(result.within_tolerance, "Should be within tolerance");
        assert!((result.max_difference - 0.0001).abs() < 1e-6, "Max difference should be 0.0001");
        assert!(
            (result.rmse - 0.0001).abs() < 1e-6,
            "RMSE should equal max_diff for single element"
        );
    }
}
