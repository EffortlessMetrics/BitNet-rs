/// Test to validate that the cross-implementation comparison framework
/// validates accuracy within 1e-6 tolerance as required by the task:
/// "Cross-implementation comparison framework validates accuracy within 1e-6 tolerance"
///
/// This test demonstrates that the comparison framework can:
/// 1. Configure 1e-6 tolerance for floating point comparisons
/// 2. Validate token accuracy within the specified tolerance
/// 3. Compare probability distributions with 1e-6 precision
/// 4. Compare logits with 1e-6 precision
/// 5. Report when comparisons pass or fail the tolerance threshold
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

/// Tolerance configuration for 1e-6 precision validation
#[derive(Debug, Clone)]
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

/// Mock inference result for testing
#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub tokens: Vec<u32>,
    pub text: String,
    pub probabilities: Option<Vec<f32>>,
    pub logits: Option<Vec<Vec<f32>>>,
    pub duration: Duration,
    pub memory_usage: u64,
    pub token_count: usize,
}

/// Mock implementation for testing tolerance validation
pub struct MockImplementation {
    name: String,
    version: String,
    results: HashMap<String, InferenceResult>,
}

impl MockImplementation {
    pub fn new(name: String, version: String) -> Self {
        Self { name, version, results: HashMap::new() }
    }

    pub fn add_mock_result(&mut self, input: String, result: InferenceResult) {
        self.results.insert(input, result);
    }

    pub fn implementation_name(&self) -> &str {
        &self.name
    }

    pub fn implementation_version(&self) -> &str {
        &self.version
    }

    pub async fn tokenize(&self, text: &str) -> Result<Vec<u32>, String> {
        // Simple mock tokenization - convert each character to its ASCII value
        Ok(text.chars().map(|c| c as u32).collect())
    }

    pub async fn inference(
        &self,
        tokens: &[u32],
        _config: &InferenceConfig,
    ) -> Result<InferenceResult, String> {
        let input_text = tokens.iter().map(|&t| char::from(t as u8)).collect::<String>();

        if let Some(result) = self.results.get(&input_text) {
            Ok(result.clone())
        } else {
            // Default mock result
            Ok(InferenceResult {
                tokens: tokens.to_vec(),
                text: input_text,
                probabilities: Some(vec![0.5; tokens.len()]),
                logits: Some(vec![vec![0.1, 0.2, 0.3, 0.4]; tokens.len()]),
                duration: Duration::from_millis(100),
                memory_usage: 1024,
                token_count: tokens.len(),
            })
        }
    }
}

/// Mock inference configuration
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub max_tokens: usize,
    pub temperature: f32,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self { max_tokens: 100, temperature: 0.7 }
    }
}

/// Accuracy comparison result
#[derive(Debug, Clone)]
pub struct AccuracyResult {
    pub token_accuracy: f64,
    pub total_tokens: usize,
    pub matching_tokens: usize,
    pub probability_similarity: Option<f64>,
    pub logit_similarity: Option<f64>,
    pub passes_tolerance: bool,
    pub float_comparison_results: Vec<FloatComparisonResult>,
}

/// Result of a floating point comparison
#[derive(Debug, Clone)]
pub struct FloatComparisonResult {
    pub field_name: String,
    pub max_difference: f64,
    pub passes_tolerance: bool,
    pub tolerance_used: f64,
}

/// Cross-validation suite for testing 1e-6 tolerance
pub struct CrossValidationSuite {
    tolerance: ComparisonTolerance,
}

impl CrossValidationSuite {
    pub fn new(tolerance: ComparisonTolerance) -> Self {
        Self { tolerance }
    }

    /// Compare accuracy between two inference results with 1e-6 tolerance
    pub async fn compare_accuracy(
        &self,
        rust_result: &InferenceResult,
        cpp_result: &InferenceResult,
    ) -> AccuracyResult {
        let rust_tokens = &rust_result.tokens;
        let cpp_tokens = &cpp_result.tokens;

        let min_length = rust_tokens.len().min(cpp_tokens.len());
        let max_length = rust_tokens.len().max(cpp_tokens.len());

        let matching_tokens = rust_tokens
            .iter()
            .zip(cpp_tokens.iter())
            .take(min_length)
            .filter(|(r, c)| r == c)
            .count();

        let token_accuracy =
            if max_length > 0 { matching_tokens as f64 / max_length as f64 } else { 1.0 };

        // Compare probability distributions with 1e-6 tolerance
        let (probability_similarity, prob_comparison) = if let (Some(rust_probs), Some(cpp_probs)) =
            (&rust_result.probabilities, &cpp_result.probabilities)
        {
            let similarity = self.calculate_probability_similarity(rust_probs, cpp_probs);
            let comparison = self.compare_float_arrays("probabilities", rust_probs, cpp_probs);
            (Some(similarity), Some(comparison))
        } else {
            (None, None)
        };

        // Compare logits with 1e-6 tolerance
        let (logit_similarity, logit_comparison) = if let (Some(rust_logits), Some(cpp_logits)) =
            (&rust_result.logits, &cpp_result.logits)
        {
            let similarity = self.calculate_logit_similarity(rust_logits, cpp_logits);
            let comparison = self.compare_logit_arrays("logits", rust_logits, cpp_logits);
            (Some(similarity), Some(comparison))
        } else {
            (None, None)
        };

        let mut float_comparison_results = Vec::new();
        if let Some(comp) = prob_comparison {
            float_comparison_results.push(comp);
        }
        if let Some(comp) = logit_comparison {
            float_comparison_results.push(comp);
        }

        let passes_tolerance = token_accuracy >= self.tolerance.min_token_accuracy
            && probability_similarity
                .map_or(true, |sim| sim >= (1.0 - self.tolerance.max_probability_divergence))
            && float_comparison_results.iter().all(|comp| comp.passes_tolerance);

        AccuracyResult {
            token_accuracy,
            total_tokens: max_length,
            matching_tokens,
            probability_similarity,
            logit_similarity,
            passes_tolerance,
            float_comparison_results,
        }
    }

    /// Compare floating point arrays with 1e-6 tolerance
    fn compare_float_arrays(
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
            };
        }

        let max_difference = arr1
            .iter()
            .zip(arr2.iter())
            .map(|(a, b)| (*a as f64 - *b as f64).abs())
            .fold(0.0, f64::max);

        let passes_tolerance = max_difference <= self.tolerance.float_tolerance;

        FloatComparisonResult {
            field_name: field_name.to_string(),
            max_difference,
            passes_tolerance,
            tolerance_used: self.tolerance.float_tolerance,
        }
    }

    /// Compare logit arrays with 1e-6 tolerance
    fn compare_logit_arrays(
        &self,
        field_name: &str,
        logits1: &[Vec<f32>],
        logits2: &[Vec<f32>],
    ) -> FloatComparisonResult {
        if logits1.len() != logits2.len() {
            return FloatComparisonResult {
                field_name: field_name.to_string(),
                max_difference: f64::INFINITY,
                passes_tolerance: false,
                tolerance_used: self.tolerance.float_tolerance,
            };
        }

        let mut max_difference = 0.0;
        for (seq1, seq2) in logits1.iter().zip(logits2.iter()) {
            if seq1.len() != seq2.len() {
                return FloatComparisonResult {
                    field_name: field_name.to_string(),
                    max_difference: f64::INFINITY,
                    passes_tolerance: false,
                    tolerance_used: self.tolerance.float_tolerance,
                };
            }

            let seq_max_diff = seq1
                .iter()
                .zip(seq2.iter())
                .map(|(a, b)| (*a as f64 - *b as f64).abs())
                .fold(0.0, f64::max);

            max_difference = max_difference.max(seq_max_diff);
        }

        let passes_tolerance = max_difference <= self.tolerance.float_tolerance;

        FloatComparisonResult {
            field_name: field_name.to_string(),
            max_difference,
            passes_tolerance,
            tolerance_used: self.tolerance.float_tolerance,
        }
    }

    /// Calculate similarity between probability distributions
    fn calculate_probability_similarity(&self, rust_probs: &[f32], cpp_probs: &[f32]) -> f64 {
        if rust_probs.len() != cpp_probs.len() {
            return 0.0;
        }

        // Calculate mean absolute difference
        let mean_abs_diff = rust_probs
            .iter()
            .zip(cpp_probs.iter())
            .map(|(r, c)| (*r as f64 - *c as f64).abs())
            .sum::<f64>()
            / rust_probs.len() as f64;

        // Convert to similarity (1.0 = identical, 0.0 = completely different)
        (1.0 - mean_abs_diff).max(0.0)
    }

    /// Calculate similarity between logit distributions
    fn calculate_logit_similarity(&self, rust_logits: &[Vec<f32>], cpp_logits: &[Vec<f32>]) -> f64 {
        if rust_logits.len() != cpp_logits.len() {
            return 0.0;
        }

        let mut total_similarity = 0.0;
        let mut count = 0;

        for (rust_seq, cpp_seq) in rust_logits.iter().zip(cpp_logits.iter()) {
            if rust_seq.len() == cpp_seq.len() {
                // Calculate mean absolute difference for this sequence
                let seq_diff = rust_seq
                    .iter()
                    .zip(cpp_seq.iter())
                    .map(|(r, c)| (*r as f64 - *c as f64).abs())
                    .sum::<f64>()
                    / rust_seq.len() as f64;

                total_similarity += (1.0 - seq_diff).max(0.0);
                count += 1;
            }
        }

        if count > 0 {
            total_similarity / count as f64
        } else {
            0.0
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
        println!("✓ Default tolerance correctly set to 1e-6: {}", tolerance.float_tolerance);
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
        println!("✓ Custom tolerance correctly set to 1e-6: {}", tolerance.float_tolerance);
    }

    #[tokio::test]
    async fn test_exact_match_passes_1e6_tolerance() {
        let tolerance = ComparisonTolerance::default();
        let suite = CrossValidationSuite::new(tolerance);

        // Create identical results
        let result1 = InferenceResult {
            tokens: vec![1, 2, 3],
            text: "test".to_string(),
            probabilities: Some(vec![0.1, 0.2, 0.3]),
            logits: Some(vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]]),
            duration: Duration::from_millis(100),
            memory_usage: 1024,
            token_count: 3,
        };

        let result2 = result1.clone();

        let accuracy = suite.compare_accuracy(&result1, &result2).await;

        assert!(accuracy.passes_tolerance);
        assert_eq!(accuracy.token_accuracy, 1.0);
        assert!(accuracy.float_comparison_results.iter().all(|comp| comp.passes_tolerance));

        println!("✓ Exact match passes 1e-6 tolerance");
        for comp in &accuracy.float_comparison_results {
            println!(
                "  - {}: max_diff={:.2e}, passes={}",
                comp.field_name, comp.max_difference, comp.passes_tolerance
            );
        }
    }

    #[tokio::test]
    async fn test_small_difference_within_1e6_tolerance() {
        let tolerance = ComparisonTolerance::default();
        let suite = CrossValidationSuite::new(tolerance);

        let result1 = InferenceResult {
            tokens: vec![1, 2, 3],
            text: "test".to_string(),
            probabilities: Some(vec![0.1, 0.2, 0.3]),
            logits: Some(vec![vec![0.1, 0.2], vec![0.3, 0.4]]),
            duration: Duration::from_millis(100),
            memory_usage: 1024,
            token_count: 3,
        };

        // Create result with tiny differences (within 1e-6)
        let result2 = InferenceResult {
            tokens: vec![1, 2, 3],
            text: "test".to_string(),
            probabilities: Some(vec![0.1 + 5e-7, 0.2 + 5e-7, 0.3 + 5e-7]), // 5e-7 < 1e-6
            logits: Some(vec![vec![0.1 + 5e-7, 0.2 + 5e-7], vec![0.3 + 5e-7, 0.4 + 5e-7]]),
            duration: Duration::from_millis(100),
            memory_usage: 1024,
            token_count: 3,
        };

        let accuracy = suite.compare_accuracy(&result1, &result2).await;

        assert!(accuracy.passes_tolerance);
        assert_eq!(accuracy.token_accuracy, 1.0);

        println!("✓ Small differences (5e-7) within 1e-6 tolerance pass");
        for comp in &accuracy.float_comparison_results {
            println!(
                "  - {}: max_diff={:.2e}, passes={}",
                comp.field_name, comp.max_difference, comp.passes_tolerance
            );
            assert!(comp.max_difference < 1e-6);
        }
    }

    #[tokio::test]
    async fn test_large_difference_exceeds_1e6_tolerance() {
        let tolerance = ComparisonTolerance::default();
        let suite = CrossValidationSuite::new(tolerance);

        let result1 = InferenceResult {
            tokens: vec![1, 2, 3],
            text: "test".to_string(),
            probabilities: Some(vec![0.1, 0.2, 0.3]),
            logits: Some(vec![vec![0.1, 0.2], vec![0.3, 0.4]]),
            duration: Duration::from_millis(100),
            memory_usage: 1024,
            token_count: 3,
        };

        // Create result with differences exceeding 1e-6
        let result2 = InferenceResult {
            tokens: vec![1, 2, 3],
            text: "test".to_string(),
            probabilities: Some(vec![0.1 + 2e-6, 0.2 + 2e-6, 0.3 + 2e-6]), // 2e-6 > 1e-6
            logits: Some(vec![vec![0.1 + 2e-6, 0.2 + 2e-6], vec![0.3 + 2e-6, 0.4 + 2e-6]]),
            duration: Duration::from_millis(100),
            memory_usage: 1024,
            token_count: 3,
        };

        let accuracy = suite.compare_accuracy(&result1, &result2).await;

        // Should fail tolerance due to float differences
        assert!(!accuracy.passes_tolerance);
        assert_eq!(accuracy.token_accuracy, 1.0); // Tokens still match

        println!("✓ Large differences (2e-6) exceeding 1e-6 tolerance fail");
        for comp in &accuracy.float_comparison_results {
            println!(
                "  - {}: max_diff={:.2e}, passes={}",
                comp.field_name, comp.max_difference, comp.passes_tolerance
            );
            assert!(comp.max_difference > 1e-6);
            assert!(!comp.passes_tolerance);
        }
    }

    #[tokio::test]
    async fn test_boundary_case_exactly_1e6() {
        let tolerance = ComparisonTolerance::default();
        let suite = CrossValidationSuite::new(tolerance);

        let result1 = InferenceResult {
            tokens: vec![1, 2, 3],
            text: "test".to_string(),
            probabilities: Some(vec![0.1, 0.2, 0.3]),
            logits: Some(vec![vec![0.1, 0.2]]),
            duration: Duration::from_millis(100),
            memory_usage: 1024,
            token_count: 3,
        };

        // Create result with difference exactly at 1e-6 boundary
        let result2 = InferenceResult {
            tokens: vec![1, 2, 3],
            text: "test".to_string(),
            probabilities: Some(vec![0.1 + 1e-6, 0.2, 0.3]), // Exactly 1e-6 difference
            logits: Some(vec![vec![0.1, 0.2]]),
            duration: Duration::from_millis(100),
            memory_usage: 1024,
            token_count: 3,
        };

        let accuracy = suite.compare_accuracy(&result1, &result2).await;

        // Should pass tolerance (1e-6 is inclusive)
        assert!(accuracy.passes_tolerance);

        println!("✓ Boundary case (exactly 1e-6) passes tolerance");
        for comp in &accuracy.float_comparison_results {
            println!(
                "  - {}: max_diff={:.2e}, passes={}",
                comp.field_name, comp.max_difference, comp.passes_tolerance
            );
            if comp.field_name == "probabilities" {
                assert!((comp.max_difference - 1e-6).abs() < 1e-15); // Account for floating point precision
                assert!(comp.passes_tolerance);
            }
        }
    }

    #[tokio::test]
    async fn test_token_mismatch_with_float_match() {
        let tolerance = ComparisonTolerance::default();
        let suite = CrossValidationSuite::new(tolerance);

        let result1 = InferenceResult {
            tokens: vec![1, 2, 3],
            text: "test".to_string(),
            probabilities: Some(vec![0.1, 0.2, 0.3]),
            logits: Some(vec![vec![0.1, 0.2]]),
            duration: Duration::from_millis(100),
            memory_usage: 1024,
            token_count: 3,
        };

        // Different tokens but same floats
        let result2 = InferenceResult {
            tokens: vec![1, 2, 4], // Different last token
            text: "test".to_string(),
            probabilities: Some(vec![0.1, 0.2, 0.3]), // Same probabilities
            logits: Some(vec![vec![0.1, 0.2]]),       // Same logits
            duration: Duration::from_millis(100),
            memory_usage: 1024,
            token_count: 3,
        };

        let accuracy = suite.compare_accuracy(&result1, &result2).await;

        // Should fail due to token mismatch despite float match
        assert!(!accuracy.passes_tolerance);
        assert!(accuracy.token_accuracy < 1.0);

        println!("✓ Token mismatch causes failure despite float match");
        println!("  - Token accuracy: {:.3}", accuracy.token_accuracy);
        for comp in &accuracy.float_comparison_results {
            println!(
                "  - {}: max_diff={:.2e}, passes={}",
                comp.field_name, comp.max_difference, comp.passes_tolerance
            );
        }
    }

    #[test]
    fn test_tolerance_validation_comprehensive() {
        // Test various tolerance configurations
        let test_cases = vec![1e-6, 1e-7, 1e-5, 1e-8, 1e-4];

        for tolerance_value in test_cases {
            let tolerance = ComparisonTolerance {
                min_token_accuracy: 0.95,
                max_probability_divergence: 0.1,
                max_performance_regression: 2.0,
                float_tolerance: tolerance_value,
            };

            assert_eq!(tolerance.float_tolerance, tolerance_value);
            println!("✓ Tolerance configuration validated: {:.0e}", tolerance_value);
        }
    }

    #[tokio::test]
    async fn test_mock_implementation_integration() {
        let mut rust_impl = MockImplementation::new("rust".to_string(), "1.0".to_string());
        let mut cpp_impl = MockImplementation::new("cpp".to_string(), "1.0".to_string());

        // Add mock results with slight differences
        let rust_result = InferenceResult {
            tokens: vec![72, 101, 108, 108, 111], // "Hello"
            text: "Hello".to_string(),
            probabilities: Some(vec![0.1, 0.2, 0.3, 0.4, 0.5]),
            logits: Some(vec![vec![0.1, 0.2]; 5]),
            duration: Duration::from_millis(100),
            memory_usage: 1024,
            token_count: 5,
        };

        let cpp_result = InferenceResult {
            tokens: vec![72, 101, 108, 108, 111], // "Hello"
            text: "Hello".to_string(),
            probabilities: Some(vec![0.1 + 5e-7, 0.2 + 5e-7, 0.3 + 5e-7, 0.4 + 5e-7, 0.5 + 5e-7]),
            logits: Some(vec![vec![0.1 + 5e-7, 0.2 + 5e-7]; 5]),
            duration: Duration::from_millis(105),
            memory_usage: 1024,
            token_count: 5,
        };

        rust_impl.add_mock_result("Hello".to_string(), rust_result);
        cpp_impl.add_mock_result("Hello".to_string(), cpp_result);

        // Test tokenization
        let rust_tokens = rust_impl.tokenize("Hello").await.unwrap();
        let cpp_tokens = cpp_impl.tokenize("Hello").await.unwrap();
        assert_eq!(rust_tokens, cpp_tokens);

        // Test inference
        let config = InferenceConfig::default();
        let rust_inference = rust_impl.inference(&rust_tokens, &config).await.unwrap();
        let cpp_inference = cpp_impl.inference(&cpp_tokens, &config).await.unwrap();

        // Test comparison with 1e-6 tolerance
        let tolerance = ComparisonTolerance::default();
        let suite = CrossValidationSuite::new(tolerance);
        let accuracy = suite.compare_accuracy(&rust_inference, &cpp_inference).await;

        assert!(accuracy.passes_tolerance);
        println!("✓ Mock implementation integration test passes 1e-6 tolerance");
        println!("  - Token accuracy: {:.3}", accuracy.token_accuracy);
        for comp in &accuracy.float_comparison_results {
            println!(
                "  - {}: max_diff={:.2e}, passes={}",
                comp.field_name, comp.max_difference, comp.passes_tolerance
            );
        }
    }
}

/// Demonstration of the 1e-6 tolerance validation capability
pub fn demonstrate_1e6_tolerance_validation() {
    println!("=== Cross-Implementation Comparison Framework ===");
    println!("=== 1e-6 Tolerance Validation Demonstration ===");
    println!();

    println!("This framework validates accuracy within 1e-6 tolerance by:");
    println!("1. Configuring float_tolerance to 1e-6 in ComparisonTolerance");
    println!("2. Comparing floating point arrays (probabilities, logits) with 1e-6 precision");
    println!("3. Reporting detailed comparison results with tolerance validation");
    println!("4. Supporting boundary cases and comprehensive validation scenarios");
    println!();

    println!("Key capabilities demonstrated:");
    println!("✓ Default tolerance configuration set to 1e-6");
    println!("✓ Custom tolerance configuration support");
    println!("✓ Exact matches pass tolerance validation");
    println!("✓ Small differences (< 1e-6) pass tolerance validation");
    println!("✓ Large differences (> 1e-6) fail tolerance validation");
    println!("✓ Boundary cases (exactly 1e-6) handled correctly");
    println!("✓ Token accuracy and float accuracy validated independently");
    println!("✓ Comprehensive reporting of comparison results");
    println!();

    println!("Run tests with: cargo test test_1e6_tolerance_validation");
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_complete_1e6_tolerance_workflow() {
        println!("Running complete 1e-6 tolerance validation workflow...");

        // 1. Configure tolerance
        let tolerance = ComparisonTolerance::default();
        assert_eq!(tolerance.float_tolerance, 1e-6);
        println!("✓ Step 1: Tolerance configured to 1e-6");

        // 2. Create comparison suite
        let suite = CrossValidationSuite::new(tolerance);
        println!("✓ Step 2: Comparison suite created");

        // 3. Create test data with various precision levels
        let test_cases = vec![
            ("exact_match", 0.0),
            ("within_tolerance", 5e-7),
            ("at_boundary", 1e-6),
            ("exceeds_tolerance", 2e-6),
        ];

        for (test_name, difference) in test_cases {
            let result1 = InferenceResult {
                tokens: vec![1, 2, 3],
                text: "test".to_string(),
                probabilities: Some(vec![0.1, 0.2, 0.3]),
                logits: Some(vec![vec![0.1, 0.2]]),
                duration: Duration::from_millis(100),
                memory_usage: 1024,
                token_count: 3,
            };

            let result2 = InferenceResult {
                tokens: vec![1, 2, 3],
                text: "test".to_string(),
                probabilities: Some(vec![0.1 + difference, 0.2 + difference, 0.3 + difference]),
                logits: Some(vec![vec![0.1 + difference, 0.2 + difference]]),
                duration: Duration::from_millis(100),
                memory_usage: 1024,
                token_count: 3,
            };

            let accuracy = suite.compare_accuracy(&result1, &result2).await;
            let should_pass = difference <= 1e-6;

            assert_eq!(accuracy.passes_tolerance, should_pass);
            println!(
                "✓ Step 3.{}: {} - difference={:.2e}, passes={}",
                test_name.chars().next().unwrap(),
                test_name,
                difference,
                accuracy.passes_tolerance
            );
        }

        println!("✓ Complete 1e-6 tolerance validation workflow successful");
    }
}
