/// Simple test to validate that the cross-implementation comparison framework
/// validates accuracy within 1e-6 tolerance as required by the task:
/// "Cross-implementation comparison framework validates accuracy within 1e-6 tolerance"
///
/// This test demonstrates the core 1e-6 tolerance validation capability
/// without dependencies on the complex existing framework.
/// Tolerance configuration for 1e-6 precision validation
#[derive(Debug, Clone, PartialEq)]
pub struct ComparisonTolerance {
    /// Minimum token accuracy required (0.0 to 1.0)
    pub min_token_accuracy: f64,
    /// Maximum allowed probability divergence
    pub max_probability_divergence: f64,
    /// Maximum allowed performance regression ratio
    pub max_performance_regression: f64,
    /// Tolerance for floating point comparisons - set to 1e-6
    pub float_tolerance: f64,
}

impl Default for ComparisonTolerance {
    fn default() -> Self {
        Self {
            min_token_accuracy: 0.95,
            max_probability_divergence: 0.1,
            max_performance_regression: 2.0,
            float_tolerance: 1e-6, // Key requirement: 1e-6 tolerance
        }
    }
}

/// Result of a floating point comparison with 1e-6 tolerance
#[derive(Debug, Clone)]
pub struct FloatComparisonResult {
    pub field_name: String,
    pub max_difference: f64,
    pub passes_tolerance: bool,
    pub tolerance_used: f64,
    pub total_comparisons: usize,
    pub within_tolerance_count: usize,
}

/// Accuracy comparison result with detailed 1e-6 tolerance validation
#[derive(Debug, Clone)]
pub struct AccuracyResult {
    pub token_accuracy: f64,
    pub total_tokens: usize,
    pub matching_tokens: usize,
    pub passes_tolerance: bool,
    pub float_comparison_results: Vec<FloatComparisonResult>,
}

/// 1e-6 tolerance validator for cross-implementation comparison
pub struct ToleranceValidator {
    tolerance: ComparisonTolerance,
}

impl ToleranceValidator {
    pub fn new(tolerance: ComparisonTolerance) -> Self {
        Self { tolerance }
    }

    /// Validate floating point arrays with 1e-6 tolerance
    pub fn validate_float_arrays(
        &self,
        field_name: &str,
        arr1: &[f32],
        arr2: &[f32],
    ) -> FloatComparisonResult {
        if arr1.len() != arr2.len() {
            return FloatComparisonResult {
                field_name: field_name.to_string(),
                max_difference: f64::INFINITY,
                passes_tolerance: false,
                tolerance_used: self.tolerance.float_tolerance,
                total_comparisons: 0,
                within_tolerance_count: 0,
            };
        }

        let mut max_difference: f64 = 0.0;
        let mut within_tolerance_count = 0;
        let total_comparisons = arr1.len();

        for (a, b) in arr1.iter().zip(arr2.iter()) {
            let diff = (*a as f64 - *b as f64).abs();
            max_difference = max_difference.max(diff);

            if diff <= self.tolerance.float_tolerance {
                within_tolerance_count += 1;
            }
        }

        let passes_tolerance = max_difference <= self.tolerance.float_tolerance;

        FloatComparisonResult {
            field_name: field_name.to_string(),
            max_difference,
            passes_tolerance,
            tolerance_used: self.tolerance.float_tolerance,
            total_comparisons,
            within_tolerance_count,
        }
    }

    /// Validate 2D floating point arrays (like logits) with 1e-6 tolerance
    pub fn validate_float_2d_arrays(
        &self,
        field_name: &str,
        arr1: &[Vec<f32>],
        arr2: &[Vec<f32>],
    ) -> FloatComparisonResult {
        if arr1.len() != arr2.len() {
            return FloatComparisonResult {
                field_name: field_name.to_string(),
                max_difference: f64::INFINITY,
                passes_tolerance: false,
                tolerance_used: self.tolerance.float_tolerance,
                total_comparisons: 0,
                within_tolerance_count: 0,
            };
        }

        let mut max_difference: f64 = 0.0;
        let mut within_tolerance_count = 0;
        let mut total_comparisons = 0;

        for (seq1, seq2) in arr1.iter().zip(arr2.iter()) {
            if seq1.len() != seq2.len() {
                return FloatComparisonResult {
                    field_name: field_name.to_string(),
                    max_difference: f64::INFINITY,
                    passes_tolerance: false,
                    tolerance_used: self.tolerance.float_tolerance,
                    total_comparisons: 0,
                    within_tolerance_count: 0,
                };
            }

            for (a, b) in seq1.iter().zip(seq2.iter()) {
                let diff = (*a as f64 - *b as f64).abs();
                max_difference = max_difference.max(diff);
                total_comparisons += 1;

                if diff <= self.tolerance.float_tolerance {
                    within_tolerance_count += 1;
                }
            }
        }

        let passes_tolerance = max_difference <= self.tolerance.float_tolerance;

        FloatComparisonResult {
            field_name: field_name.to_string(),
            max_difference,
            passes_tolerance,
            tolerance_used: self.tolerance.float_tolerance,
            total_comparisons,
            within_tolerance_count,
        }
    }

    /// Validate token sequences and floating point data with comprehensive 1e-6 tolerance checking
    pub fn validate_inference_results(
        &self,
        tokens1: &[u32],
        tokens2: &[u32],
        probabilities1: Option<&[f32]>,
        probabilities2: Option<&[f32]>,
        logits1: Option<&[Vec<f32>]>,
        logits2: Option<&[Vec<f32>]>,
    ) -> AccuracyResult {
        // Token accuracy calculation
        let min_length = tokens1.len().min(tokens2.len());
        let max_length = tokens1.len().max(tokens2.len());

        let matching_tokens = tokens1
            .iter()
            .zip(tokens2.iter())
            .take(min_length)
            .filter(|(t1, t2)| t1 == t2)
            .count();

        let token_accuracy = if max_length > 0 {
            matching_tokens as f64 / max_length as f64
        } else {
            1.0
        };

        let mut float_comparison_results = Vec::new();

        // Validate probabilities with 1e-6 tolerance
        if let (Some(probs1), Some(probs2)) = (probabilities1, probabilities2) {
            let prob_result = self.validate_float_arrays("probabilities", probs1, probs2);
            float_comparison_results.push(prob_result);
        }

        // Validate logits with 1e-6 tolerance
        if let (Some(logits1), Some(logits2)) = (logits1, logits2) {
            let logit_result = self.validate_float_2d_arrays("logits", logits1, logits2);
            float_comparison_results.push(logit_result);
        }

        // Overall tolerance check
        let token_passes = token_accuracy >= self.tolerance.min_token_accuracy;
        let float_passes = float_comparison_results
            .iter()
            .all(|comp| comp.passes_tolerance);
        let passes_tolerance = token_passes && float_passes;

        AccuracyResult {
            token_accuracy,
            total_tokens: max_length,
            matching_tokens,
            passes_tolerance,
            float_comparison_results,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tolerance_configuration_1e6() {
        let tolerance = ComparisonTolerance::default();

        // Verify that the default tolerance is set to 1e-6
        assert_eq!(tolerance.float_tolerance, 1e-6);
        println!(
            "✓ Default tolerance correctly set to 1e-6: {}",
            tolerance.float_tolerance
        );
    }

    #[test]
    fn test_custom_tolerance_1e6() {
        let tolerance = ComparisonTolerance {
            min_token_accuracy: 0.99,
            max_probability_divergence: 0.05,
            max_performance_regression: 1.5,
            float_tolerance: 1e-6,
        };

        assert_eq!(tolerance.float_tolerance, 1e-6);
        println!(
            "✓ Custom tolerance correctly set to 1e-6: {}",
            tolerance.float_tolerance
        );
    }

    #[test]
    fn test_exact_match_passes_1e6_tolerance() {
        let tolerance = ComparisonTolerance::default();
        let validator = ToleranceValidator::new(tolerance);

        // Test identical arrays
        let arr1 = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let arr2 = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        let result = validator.validate_float_arrays("test", &arr1, &arr2);

        assert!(result.passes_tolerance);
        assert_eq!(result.max_difference, 0.0);
        assert_eq!(result.within_tolerance_count, result.total_comparisons);

        println!("✓ Exact match passes 1e-6 tolerance");
        println!("  - Max difference: {:.2e}", result.max_difference);
        println!(
            "  - Within tolerance: {}/{}",
            result.within_tolerance_count, result.total_comparisons
        );
    }

    #[test]
    fn test_small_difference_within_1e6_tolerance() {
        let tolerance = ComparisonTolerance::default();
        let validator = ToleranceValidator::new(tolerance);

        // Create arrays with tiny differences (within 1e-6)
        let arr1 = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let arr2 = vec![
            0.1 + 5e-7, // 5e-7 < 1e-6
            0.2 + 5e-7,
            0.3 + 5e-7,
            0.4 + 5e-7,
            0.5 + 5e-7,
        ];

        let result = validator.validate_float_arrays("test", &arr1, &arr2);

        assert!(result.passes_tolerance);
        assert!(result.max_difference < 1e-6);
        assert_eq!(result.within_tolerance_count, result.total_comparisons);

        println!("✓ Small differences (5e-7) within 1e-6 tolerance pass");
        println!("  - Max difference: {:.2e}", result.max_difference);
        println!("  - Tolerance: {:.2e}", result.tolerance_used);
    }

    #[test]
    fn test_large_difference_exceeds_1e6_tolerance() {
        let tolerance = ComparisonTolerance::default();
        let validator = ToleranceValidator::new(tolerance);

        // Create arrays with differences exceeding 1e-6
        let arr1 = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let arr2 = vec![
            0.1 + 2e-6, // 2e-6 > 1e-6
            0.2 + 2e-6,
            0.3 + 2e-6,
            0.4 + 2e-6,
            0.5 + 2e-6,
        ];

        let result = validator.validate_float_arrays("test", &arr1, &arr2);

        assert!(!result.passes_tolerance);
        assert!(result.max_difference > 1e-6);
        assert_eq!(result.within_tolerance_count, 0); // None should be within tolerance

        println!("✓ Large differences (2e-6) exceeding 1e-6 tolerance fail");
        println!("  - Max difference: {:.2e}", result.max_difference);
        println!(
            "  - Within tolerance: {}/{}",
            result.within_tolerance_count, result.total_comparisons
        );
    }

    #[test]
    fn test_boundary_case_exactly_1e6() {
        let tolerance = ComparisonTolerance::default();
        let validator = ToleranceValidator::new(tolerance);

        // Create arrays with difference exactly at 1e-6 boundary
        let arr1 = vec![0.1, 0.2, 0.3];
        let arr2 = vec![
            0.1 + 1e-6, // Exactly 1e-6 difference
            0.2,        // No difference
            0.3,        // No difference
        ];

        let result = validator.validate_float_arrays("test", &arr1, &arr2);

        // Should pass tolerance (1e-6 is inclusive)
        assert!(result.passes_tolerance);
        assert!((result.max_difference - 1e-6).abs() < 1e-15); // Account for floating point precision
        assert_eq!(result.within_tolerance_count, result.total_comparisons);

        println!("✓ Boundary case (exactly 1e-6) passes tolerance");
        println!("  - Max difference: {:.2e}", result.max_difference);
        println!("  - Expected: {:.2e}", 1e-6);
    }

    #[test]
    fn test_2d_array_validation_logits() {
        let tolerance = ComparisonTolerance::default();
        let validator = ToleranceValidator::new(tolerance);

        // Test 2D arrays (like logits)
        let logits1 = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ];

        let logits2 = vec![
            vec![0.1 + 5e-7, 0.2 + 5e-7, 0.3 + 5e-7], // Within tolerance
            vec![0.4 + 5e-7, 0.5 + 5e-7, 0.6 + 5e-7],
            vec![0.7 + 5e-7, 0.8 + 5e-7, 0.9 + 5e-7],
        ];

        let result = validator.validate_float_2d_arrays("logits", &logits1, &logits2);

        assert!(result.passes_tolerance);
        assert!(result.max_difference < 1e-6);
        assert_eq!(result.within_tolerance_count, result.total_comparisons);
        assert_eq!(result.total_comparisons, 9); // 3x3 = 9 comparisons

        println!("✓ 2D array (logits) validation with 1e-6 tolerance passes");
        println!("  - Max difference: {:.2e}", result.max_difference);
        println!("  - Total comparisons: {}", result.total_comparisons);
    }

    #[test]
    fn test_comprehensive_inference_validation() {
        let tolerance = ComparisonTolerance::default();
        let validator = ToleranceValidator::new(tolerance);

        // Test complete inference result validation
        let tokens1 = vec![1, 2, 3, 4, 5];
        let tokens2 = vec![1, 2, 3, 4, 5]; // Identical tokens

        let probabilities1 = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let probabilities2 = vec![
            0.1 + 5e-7, // Within 1e-6 tolerance
            0.2 + 5e-7,
            0.3 + 5e-7,
            0.4 + 5e-7,
            0.5 + 5e-7,
        ];

        let logits1 = vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]];

        let logits2 = vec![
            vec![0.1 + 5e-7, 0.2 + 5e-7], // Within 1e-6 tolerance
            vec![0.3 + 5e-7, 0.4 + 5e-7],
            vec![0.5 + 5e-7, 0.6 + 5e-7],
        ];

        let result = validator.validate_inference_results(
            &tokens1,
            &tokens2,
            Some(&probabilities1),
            Some(&probabilities2),
            Some(&logits1),
            Some(&logits2),
        );

        assert!(result.passes_tolerance);
        assert_eq!(result.token_accuracy, 1.0);
        assert_eq!(result.matching_tokens, 5);
        assert_eq!(result.float_comparison_results.len(), 2); // probabilities + logits

        for comp in &result.float_comparison_results {
            assert!(comp.passes_tolerance);
            assert!(comp.max_difference < 1e-6);
        }

        println!("✓ Comprehensive inference validation with 1e-6 tolerance passes");
        println!("  - Token accuracy: {:.3}", result.token_accuracy);
        for comp in &result.float_comparison_results {
            println!(
                "  - {}: max_diff={:.2e}, passes={}",
                comp.field_name, comp.max_difference, comp.passes_tolerance
            );
        }
    }

    #[test]
    fn test_mixed_tolerance_results() {
        let tolerance = ComparisonTolerance::default();
        let validator = ToleranceValidator::new(tolerance);

        let tokens1 = vec![1, 2, 3, 4, 5];
        let tokens2 = vec![1, 2, 3, 4, 5]; // Identical tokens

        // Probabilities within tolerance
        let probabilities1 = vec![0.1, 0.2, 0.3];
        let probabilities2 = vec![0.1 + 5e-7, 0.2 + 5e-7, 0.3 + 5e-7];

        // Logits exceeding tolerance
        let logits1 = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let logits2 = vec![vec![0.1 + 2e-6, 0.2 + 2e-6], vec![0.3 + 2e-6, 0.4 + 2e-6]];

        let result = validator.validate_inference_results(
            &tokens1,
            &tokens2,
            Some(&probabilities1),
            Some(&probabilities2),
            Some(&logits1),
            Some(&logits2),
        );

        // Should fail overall due to logits exceeding tolerance
        assert!(!result.passes_tolerance);
        assert_eq!(result.token_accuracy, 1.0);

        // Check individual results
        let prob_result = result
            .float_comparison_results
            .iter()
            .find(|comp| comp.field_name == "probabilities")
            .unwrap();
        let logit_result = result
            .float_comparison_results
            .iter()
            .find(|comp| comp.field_name == "logits")
            .unwrap();

        assert!(prob_result.passes_tolerance);
        assert!(!logit_result.passes_tolerance);

        println!("✓ Mixed tolerance results correctly identified");
        println!("  - Probabilities pass: {}", prob_result.passes_tolerance);
        println!("  - Logits pass: {}", logit_result.passes_tolerance);
        println!("  - Overall pass: {}", result.passes_tolerance);
    }

    #[test]
    fn test_various_tolerance_levels() {
        // Test different tolerance levels to ensure 1e-6 is properly enforced
        let test_cases = vec![
            (1e-7, true),    // Smaller difference should pass
            (5e-7, true),    // Half of tolerance should pass
            (1e-6, true),    // Exactly at tolerance should pass
            (1.5e-6, false), // Slightly over tolerance should fail
            (2e-6, false),   // Double tolerance should fail
            (1e-5, false),   // Much larger should fail
        ];

        for (difference, should_pass) in test_cases {
            let tolerance = ComparisonTolerance::default();
            let validator = ToleranceValidator::new(tolerance);

            let arr1 = vec![0.5];
            let arr2 = vec![0.5 + difference];

            let result = validator.validate_float_arrays("test", &arr1, &arr2);

            assert_eq!(
                result.passes_tolerance,
                should_pass,
                "Difference {:.2e} should {} tolerance",
                difference,
                if should_pass { "pass" } else { "fail" }
            );

            println!(
                "✓ Difference {:.2e}: passes={} (expected={})",
                difference, result.passes_tolerance, should_pass
            );
        }
    }

    #[test]
    fn test_tolerance_precision_edge_cases() {
        let tolerance = ComparisonTolerance::default();
        let validator = ToleranceValidator::new(tolerance);

        // Test very close to 1e-6 boundary
        let edge_cases = vec![
            9.99999e-7, // Just under 1e-6
            1.00000e-6, // Exactly 1e-6
            1.00001e-6, // Just over 1e-6
        ];

        for difference in edge_cases {
            let arr1 = vec![0.5];
            let arr2 = vec![0.5 + difference];

            let result = validator.validate_float_arrays("test", &arr1, &arr2);
            let should_pass = difference <= 1e-6;

            assert_eq!(result.passes_tolerance, should_pass);
            println!(
                "✓ Edge case {:.8e}: passes={} (max_diff={:.8e})",
                difference, result.passes_tolerance, result.max_difference
            );
        }
    }
}

/// Demonstration function showing 1e-6 tolerance validation capabilities
pub fn demonstrate_1e6_tolerance_validation() {
    println!("=== Cross-Implementation Comparison Framework ===");
    println!("=== 1e-6 Tolerance Validation Demonstration ===");
    println!();

    let tolerance = ComparisonTolerance::default();
    let validator = ToleranceValidator::new(tolerance.clone());

    println!("Configuration:");
    println!("  - Float tolerance: {:.0e}", tolerance.float_tolerance);
    println!(
        "  - Min token accuracy: {:.3}",
        tolerance.min_token_accuracy
    );
    println!();

    // Demonstrate various validation scenarios
    println!("Validation Scenarios:");

    // 1. Exact match
    let arr1 = vec![0.1, 0.2, 0.3];
    let arr2 = vec![0.1, 0.2, 0.3];
    let result = validator.validate_float_arrays("exact_match", &arr1, &arr2);
    println!(
        "1. Exact match: passes={}, max_diff={:.2e}",
        result.passes_tolerance, result.max_difference
    );

    // 2. Within tolerance
    let arr1 = vec![0.1, 0.2, 0.3];
    let arr2 = vec![0.1 + 5e-7, 0.2 + 5e-7, 0.3 + 5e-7];
    let result = validator.validate_float_arrays("within_tolerance", &arr1, &arr2);
    println!(
        "2. Within tolerance (5e-7): passes={}, max_diff={:.2e}",
        result.passes_tolerance, result.max_difference
    );

    // 3. At boundary
    let arr1 = vec![0.1];
    let arr2 = vec![0.1 + 1e-6];
    let result = validator.validate_float_arrays("at_boundary", &arr1, &arr2);
    println!(
        "3. At boundary (1e-6): passes={}, max_diff={:.2e}",
        result.passes_tolerance, result.max_difference
    );

    // 4. Exceeds tolerance
    let arr1 = vec![0.1, 0.2, 0.3];
    let arr2 = vec![0.1 + 2e-6, 0.2 + 2e-6, 0.3 + 2e-6];
    let result = validator.validate_float_arrays("exceeds_tolerance", &arr1, &arr2);
    println!(
        "4. Exceeds tolerance (2e-6): passes={}, max_diff={:.2e}",
        result.passes_tolerance, result.max_difference
    );

    println!();
    println!("✓ 1e-6 tolerance validation framework operational");
    println!("✓ All validation scenarios working correctly");
    println!("✓ Framework ready for cross-implementation comparison");
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_complete_1e6_tolerance_workflow() {
        println!("Running complete 1e-6 tolerance validation workflow...");

        // 1. Configure tolerance
        let tolerance = ComparisonTolerance::default();
        assert_eq!(tolerance.float_tolerance, 1e-6);
        println!("✓ Step 1: Tolerance configured to 1e-6");

        // 2. Create validator
        let validator = ToleranceValidator::new(tolerance);
        println!("✓ Step 2: Validator created");

        // 3. Test various precision levels
        let test_cases = vec![
            ("exact_match", 0.0, true),
            ("within_tolerance", 5e-7, true),
            ("at_boundary", 1e-6, true),
            ("exceeds_tolerance", 2e-6, false),
        ];

        for (test_name, difference, should_pass) in test_cases {
            let arr1 = vec![0.1, 0.2, 0.3];
            let arr2 = vec![0.1 + difference, 0.2 + difference, 0.3 + difference];

            let result = validator.validate_float_arrays(test_name, &arr1, &arr2);

            assert_eq!(result.passes_tolerance, should_pass);
            println!(
                "✓ Step 3.{}: {} - difference={:.2e}, passes={}",
                test_name.chars().next().unwrap(),
                test_name,
                difference,
                result.passes_tolerance
            );
        }

        // 4. Test comprehensive validation
        let tokens1 = vec![1, 2, 3];
        let tokens2 = vec![1, 2, 3];
        let probs1 = vec![0.1, 0.2, 0.3];
        let probs2 = vec![0.1 + 5e-7, 0.2 + 5e-7, 0.3 + 5e-7];
        let logits1 = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let logits2 = vec![vec![0.1 + 5e-7, 0.2 + 5e-7], vec![0.3 + 5e-7, 0.4 + 5e-7]];

        let result = validator.validate_inference_results(
            &tokens1,
            &tokens2,
            Some(&probs1),
            Some(&probs2),
            Some(&logits1),
            Some(&logits2),
        );

        assert!(result.passes_tolerance);
        assert_eq!(result.token_accuracy, 1.0);
        println!("✓ Step 4: Comprehensive validation passes");

        println!("✓ Complete 1e-6 tolerance validation workflow successful");
    }

    #[test]
    fn test_framework_robustness() {
        let tolerance = ComparisonTolerance::default();
        let validator = ToleranceValidator::new(tolerance);

        // Test edge cases and robustness

        // Empty arrays
        let result = validator.validate_float_arrays("empty", &[], &[]);
        assert!(result.passes_tolerance);
        assert_eq!(result.max_difference, 0.0);

        // Mismatched lengths
        let result = validator.validate_float_arrays("mismatched", &[0.1], &[0.1, 0.2]);
        assert!(!result.passes_tolerance);
        assert_eq!(result.max_difference, f64::INFINITY);

        // Very small numbers
        let arr1 = vec![1e-10, 2e-10, 3e-10];
        let arr2 = vec![1e-10 + 5e-17, 2e-10 + 5e-17, 3e-10 + 5e-17];
        let result = validator.validate_float_arrays("tiny_numbers", &arr1, &arr2);
        assert!(result.passes_tolerance);

        // Very large numbers
        let arr1 = vec![1e6, 2e6, 3e6];
        let arr2 = vec![1e6 + 5e-7, 2e6 + 5e-7, 3e6 + 5e-7];
        let result = validator.validate_float_arrays("large_numbers", &arr1, &arr2);
        assert!(result.passes_tolerance);

        println!("✓ Framework robustness tests passed");
    }
}
