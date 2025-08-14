use super::implementation::{
    BitNetImplementation, InferenceConfig, InferenceResult, PerformanceMetrics,
};
use crate::errors::{ComparisonError, ComparisonResult};
use serde::{Deserialize, Serialize};

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Configuration for comparison tolerance thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonTolerance {
    /// Minimum token accuracy required (0.0 to 1.0)
    pub min_token_accuracy: f64,
    /// Maximum allowed probability divergence
    pub max_probability_divergence: f64,
    /// Maximum allowed performance regression ratio
    pub max_performance_regression: f64,
    /// Tolerance for floating point comparisons
    pub float_tolerance: f64,
}

impl Default for ComparisonTolerance {
    fn default() -> Self {
        Self {
            min_token_accuracy: 0.95,
            max_probability_divergence: 0.1,
            max_performance_regression: 2.0, // Allow 2x slower
            float_tolerance: 1e-6,
        }
    }
}

/// A single test case for cross-implementation comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonTestCase {
    pub name: String,
    pub input: String,
    pub config: InferenceConfig,
    pub expected_min_tokens: Option<usize>,
    pub expected_max_tokens: Option<usize>,
    pub description: String,
}

impl ComparisonTestCase {
    pub fn new<S: Into<String>>(name: S, input: S, config: InferenceConfig) -> Self {
        Self {
            name: name.into(),
            input: input.into(),
            config,
            expected_min_tokens: None,
            expected_max_tokens: None,
            description: String::new(),
        }
    }

    pub fn with_token_range(mut self, min: usize, max: usize) -> Self {
        self.expected_min_tokens = Some(min);
        self.expected_max_tokens = Some(max);
        self
    }

    pub fn with_description<S: Into<String>>(mut self, description: S) -> Self {
        self.description = description.into();
        self
    }
}

/// Information about a token mismatch between implementations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenMismatch {
    pub position: usize,
    pub rust_token: u32,
    pub cpp_token: u32,
    pub rust_text: Option<String>,
    pub cpp_text: Option<String>,
    pub context_before: Vec<u32>,
    pub context_after: Vec<u32>,
}

/// Result of accuracy comparison between implementations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyResult {
    pub token_accuracy: f64,
    pub total_tokens: usize,
    pub matching_tokens: usize,
    pub first_mismatch: Option<TokenMismatch>,
    pub probability_similarity: Option<f64>,
    pub logit_similarity: Option<f64>,
    pub passes_tolerance: bool,
    pub detailed_mismatches: Vec<TokenMismatch>,
}

/// Performance comparison between implementations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceComparison {
    pub rust_duration: Duration,
    pub cpp_duration: Duration,
    pub throughput_ratio: f64, // rust_time / cpp_time (< 1.0 means Rust is faster)
    pub rust_memory: u64,
    pub cpp_memory: u64,
    pub memory_ratio: f64, // rust_memory / cpp_memory (< 1.0 means Rust uses less)
    pub rust_tokens_per_second: f64,
    pub cpp_tokens_per_second: f64,
    pub performance_regression: bool,
}

/// Result of a single comparison test case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingleComparisonResult {
    pub test_case: ComparisonTestCase,
    pub tokenization_match: bool,
    pub rust_tokens: Vec<u32>,
    pub cpp_tokens: Vec<u32>,
    pub accuracy_result: AccuracyResult,
    pub performance_comparison: PerformanceComparison,
    pub rust_result: InferenceResult,
    pub cpp_result: InferenceResult,
    pub success: bool,
    pub error: Option<String>,
}

/// Summary statistics for a comparison run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonSummary {
    pub total_tests: usize,
    pub successful_tests: usize,
    pub failed_tests: usize,
    pub average_token_accuracy: f64,
    pub average_throughput_ratio: f64,
    pub average_memory_ratio: f64,
    pub tests_passing_tolerance: usize,
    pub first_failure: Option<String>,
}

/// Complete result of a cross-validation comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationResult {
    pub model_path: PathBuf,
    pub model_name: String,
    pub tolerance: ComparisonTolerance,
    pub test_results: Vec<SingleComparisonResult>,
    pub summary: ComparisonSummary,
    pub rust_metrics: PerformanceMetrics,
    pub cpp_metrics: PerformanceMetrics,
    pub total_duration: Duration,
    pub timestamp: String,
}

/// Main cross-validation suite for comparing implementations
pub struct CrossValidationSuite {
    rust_impl: Box<dyn BitNetImplementation>,
    cpp_impl: Box<dyn BitNetImplementation>,
    tolerance: ComparisonTolerance,
    test_cases: Vec<ComparisonTestCase>,
    context_window: usize,
}

impl CrossValidationSuite {
    /// Create a new cross-validation suite with the given implementations and tolerance
    pub fn new(
        rust_impl: Box<dyn BitNetImplementation>,
        cpp_impl: Box<dyn BitNetImplementation>,
        tolerance: ComparisonTolerance,
    ) -> Self {
        Self {
            rust_impl,
            cpp_impl,
            tolerance,
            test_cases: Vec::new(),
            context_window: 5, // Show 5 tokens before/after mismatch
        }
    }

    /// Add a test case to the suite
    pub fn add_test_case(&mut self, test_case: ComparisonTestCase) {
        self.test_cases.push(test_case);
    }

    /// Add multiple test cases to the suite
    pub fn add_test_cases(&mut self, test_cases: Vec<ComparisonTestCase>) {
        self.test_cases.extend(test_cases);
    }

    /// Clear all test cases from the suite
    pub fn clear_test_cases(&mut self) {
        self.test_cases.clear();
    }

    /// Set the context window size for mismatch reporting
    pub fn set_context_window(&mut self, size: usize) {
        self.context_window = size;
    }

    /// Run the complete comparison suite
    pub async fn run_comparison(
        &mut self,
        model_path: &Path,
    ) -> ComparisonResult<CrossValidationResult> {
        let start_time = Instant::now();
        let model_name =
            model_path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown").to_string();

        // Initialize both implementations
        self.rust_impl
            .initialize(None)
            .await
            .map_err(|e| ComparisonError::ImplementationError(e))?;
        self.cpp_impl
            .initialize(None)
            .await
            .map_err(|e| ComparisonError::ImplementationError(e))?;

        // Load model in both implementations
        self.rust_impl
            .load_model(model_path)
            .await
            .map_err(|e| ComparisonError::ImplementationError(e))?;
        self.cpp_impl
            .load_model(model_path)
            .await
            .map_err(|e| ComparisonError::ImplementationError(e))?;

        let mut test_results = Vec::new();

        // Run each test case
        let test_cases = self.test_cases.clone();
        for test_case in &test_cases {
            let result = self.run_single_comparison(test_case).await;
            test_results.push(result);
        }

        let total_duration = start_time.elapsed();
        let summary = self.calculate_summary(&test_results);

        Ok(CrossValidationResult {
            model_path: model_path.to_path_buf(),
            model_name,
            tolerance: self.tolerance.clone(),
            test_results,
            summary,
            rust_metrics: self.rust_impl.get_metrics(),
            cpp_metrics: self.cpp_impl.get_metrics(),
            total_duration,
            timestamp: chrono::Utc::now().to_rfc3339(),
        })
    }

    /// Run a single comparison test case
    async fn run_single_comparison(
        &mut self,
        test_case: &ComparisonTestCase,
    ) -> SingleComparisonResult {
        let mut result = SingleComparisonResult {
            test_case: test_case.clone(),
            tokenization_match: false,
            rust_tokens: Vec::new(),
            cpp_tokens: Vec::new(),
            accuracy_result: AccuracyResult {
                token_accuracy: 0.0,
                total_tokens: 0,
                matching_tokens: 0,
                first_mismatch: None,
                probability_similarity: None,
                logit_similarity: None,
                passes_tolerance: false,
                detailed_mismatches: Vec::new(),
            },
            performance_comparison: PerformanceComparison {
                rust_duration: Duration::ZERO,
                cpp_duration: Duration::ZERO,
                throughput_ratio: 0.0,
                rust_memory: 0,
                cpp_memory: 0,
                memory_ratio: 0.0,
                rust_tokens_per_second: 0.0,
                cpp_tokens_per_second: 0.0,
                performance_regression: false,
            },
            rust_result: InferenceResult {
                tokens: Vec::new(),
                text: String::new(),
                probabilities: None,
                logits: None,
                duration: Duration::ZERO,
                memory_usage: 0,
                token_count: 0,
            },
            cpp_result: InferenceResult {
                tokens: Vec::new(),
                text: String::new(),
                probabilities: None,
                logits: None,
                duration: Duration::ZERO,
                memory_usage: 0,
                token_count: 0,
            },
            success: false,
            error: None,
        };

        // Tokenize input in both implementations
        let rust_tokens_result = self.rust_impl.tokenize(&test_case.input).await;
        let cpp_tokens_result = self.cpp_impl.tokenize(&test_case.input).await;

        let (rust_tokens, cpp_tokens) = match (rust_tokens_result, cpp_tokens_result) {
            (Ok(rust), Ok(cpp)) => (rust, cpp),
            (Err(e), _) => {
                result.error = Some(format!("Rust tokenization failed: {}", e));
                return result;
            }
            (_, Err(e)) => {
                result.error = Some(format!("C++ tokenization failed: {}", e));
                return result;
            }
        };

        result.rust_tokens = rust_tokens.clone();
        result.cpp_tokens = cpp_tokens.clone();
        result.tokenization_match = rust_tokens == cpp_tokens;

        // Run inference in both implementations
        let rust_inference_result = self.rust_impl.inference(&rust_tokens, &test_case.config).await;
        let cpp_inference_result = self.cpp_impl.inference(&cpp_tokens, &test_case.config).await;

        let (rust_inference, cpp_inference) = match (rust_inference_result, cpp_inference_result) {
            (Ok(rust), Ok(cpp)) => (rust, cpp),
            (Err(e), _) => {
                result.error = Some(format!("Rust inference failed: {}", e));
                return result;
            }
            (_, Err(e)) => {
                result.error = Some(format!("C++ inference failed: {}", e));
                return result;
            }
        };

        result.rust_result = rust_inference.clone();
        result.cpp_result = cpp_inference.clone();

        // Compare accuracy
        result.accuracy_result = self.compare_accuracy(&rust_inference, &cpp_inference).await;

        // Compare performance
        result.performance_comparison = self.compare_performance(&rust_inference, &cpp_inference);

        // Determine overall success
        result.success = result.accuracy_result.passes_tolerance
            && !result.performance_comparison.performance_regression
            && result.error.is_none();

        result
    }

    /// Compare accuracy between two inference results
    async fn compare_accuracy(
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

        // Find first mismatch
        let first_mismatch = self.find_first_mismatch(rust_tokens, cpp_tokens).await;

        // Find all mismatches for detailed analysis
        let detailed_mismatches = self.find_all_mismatches(rust_tokens, cpp_tokens).await;

        // Compare probability distributions if available
        let probability_similarity = if let (Some(rust_probs), Some(cpp_probs)) =
            (&rust_result.probabilities, &cpp_result.probabilities)
        {
            Some(self.calculate_probability_similarity(rust_probs, cpp_probs))
        } else {
            None
        };

        // Compare logits if available
        let logit_similarity = if let (Some(rust_logits), Some(cpp_logits)) =
            (&rust_result.logits, &cpp_result.logits)
        {
            Some(self.calculate_logit_similarity(rust_logits, cpp_logits))
        } else {
            None
        };

        let passes_tolerance = token_accuracy >= self.tolerance.min_token_accuracy
            && probability_similarity
                .map_or(true, |sim| sim >= (1.0 - self.tolerance.max_probability_divergence));

        AccuracyResult {
            token_accuracy,
            total_tokens: max_length,
            matching_tokens,
            first_mismatch,
            probability_similarity,
            logit_similarity,
            passes_tolerance,
            detailed_mismatches,
        }
    }

    /// Find the first token mismatch between implementations
    async fn find_first_mismatch(
        &self,
        rust_tokens: &[u32],
        cpp_tokens: &[u32],
    ) -> Option<TokenMismatch> {
        for (idx, (rust_token, cpp_token)) in rust_tokens.iter().zip(cpp_tokens.iter()).enumerate()
        {
            if rust_token != cpp_token {
                let context_start = idx.saturating_sub(self.context_window);
                let context_end_rust = (idx + self.context_window + 1).min(rust_tokens.len());
                let _context_end_cpp = (idx + self.context_window + 1).min(cpp_tokens.len());

                let context_before = rust_tokens[context_start..idx].to_vec();
                let context_after = rust_tokens[(idx + 1)..context_end_rust].to_vec();

                // Try to detokenize for better readability
                let rust_text = self.rust_impl.detokenize(&[*rust_token]).await.ok();
                let cpp_text = self.cpp_impl.detokenize(&[*cpp_token]).await.ok();

                return Some(TokenMismatch {
                    position: idx,
                    rust_token: *rust_token,
                    cpp_token: *cpp_token,
                    rust_text,
                    cpp_text,
                    context_before,
                    context_after,
                });
            }
        }

        // Check for length mismatches
        if rust_tokens.len() != cpp_tokens.len() {
            let min_len = rust_tokens.len().min(cpp_tokens.len());
            let context_start = min_len.saturating_sub(self.context_window);
            let context_before =
                if min_len > 0 { rust_tokens[context_start..min_len].to_vec() } else { Vec::new() };

            return Some(TokenMismatch {
                position: min_len,
                rust_token: rust_tokens.get(min_len).copied().unwrap_or(0),
                cpp_token: cpp_tokens.get(min_len).copied().unwrap_or(0),
                rust_text: None,
                cpp_text: None,
                context_before,
                context_after: Vec::new(),
            });
        }

        None
    }

    /// Find all token mismatches for detailed analysis
    async fn find_all_mismatches(
        &self,
        rust_tokens: &[u32],
        cpp_tokens: &[u32],
    ) -> Vec<TokenMismatch> {
        let mut mismatches = Vec::new();
        let max_mismatches = 10; // Limit to prevent excessive memory usage

        for (idx, (rust_token, cpp_token)) in rust_tokens.iter().zip(cpp_tokens.iter()).enumerate()
        {
            if rust_token != cpp_token && mismatches.len() < max_mismatches {
                let context_start = idx.saturating_sub(self.context_window);
                let context_end = (idx + self.context_window + 1).min(rust_tokens.len());

                let context_before = rust_tokens[context_start..idx].to_vec();
                let context_after = rust_tokens[(idx + 1)..context_end].to_vec();

                let rust_text = self.rust_impl.detokenize(&[*rust_token]).await.ok();
                let cpp_text = self.cpp_impl.detokenize(&[*cpp_token]).await.ok();

                mismatches.push(TokenMismatch {
                    position: idx,
                    rust_token: *rust_token,
                    cpp_token: *cpp_token,
                    rust_text,
                    cpp_text,
                    context_before,
                    context_after,
                });
            }
        }

        mismatches
    }

    /// Calculate similarity between probability distributions
    fn calculate_probability_similarity(&self, rust_probs: &[f32], cpp_probs: &[f32]) -> f64 {
        if rust_probs.len() != cpp_probs.len() {
            return 0.0;
        }

        // Calculate Jensen-Shannon divergence
        let mut js_divergence = 0.0;
        let len = rust_probs.len();

        for i in 0..len {
            let p = rust_probs[i] as f64;
            let q = cpp_probs[i] as f64;
            let m = (p + q) / 2.0;

            if p > 0.0 && m > 0.0 {
                js_divergence += p * (p / m).ln();
            }
            if q > 0.0 && m > 0.0 {
                js_divergence += q * (q / m).ln();
            }
        }

        js_divergence /= 2.0;

        // Convert to similarity (0 = identical, higher = more different)
        (-js_divergence).exp()
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
                // Calculate cosine similarity
                let dot_product: f64 = rust_seq
                    .iter()
                    .zip(cpp_seq.iter())
                    .map(|(r, c)| (*r as f64) * (*c as f64))
                    .sum();

                let rust_norm: f64 =
                    rust_seq.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
                let cpp_norm: f64 = cpp_seq.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();

                if rust_norm > 0.0 && cpp_norm > 0.0 {
                    total_similarity += dot_product / (rust_norm * cpp_norm);
                    count += 1;
                }
            }
        }

        if count > 0 {
            total_similarity / count as f64
        } else {
            0.0
        }
    }

    /// Compare performance between implementations
    fn compare_performance(
        &self,
        rust_result: &InferenceResult,
        cpp_result: &InferenceResult,
    ) -> PerformanceComparison {
        let throughput_ratio = if cpp_result.duration.as_secs_f64() > 0.0 {
            rust_result.duration.as_secs_f64() / cpp_result.duration.as_secs_f64()
        } else {
            1.0
        };

        let memory_ratio = if cpp_result.memory_usage > 0 {
            rust_result.memory_usage as f64 / cpp_result.memory_usage as f64
        } else {
            1.0
        };

        let rust_tokens_per_second = if rust_result.duration.as_secs_f64() > 0.0 {
            rust_result.token_count as f64 / rust_result.duration.as_secs_f64()
        } else {
            0.0
        };

        let cpp_tokens_per_second = if cpp_result.duration.as_secs_f64() > 0.0 {
            cpp_result.token_count as f64 / cpp_result.duration.as_secs_f64()
        } else {
            0.0
        };

        let performance_regression = throughput_ratio > self.tolerance.max_performance_regression;

        PerformanceComparison {
            rust_duration: rust_result.duration,
            cpp_duration: cpp_result.duration,
            throughput_ratio,
            rust_memory: rust_result.memory_usage,
            cpp_memory: cpp_result.memory_usage,
            memory_ratio,
            rust_tokens_per_second,
            cpp_tokens_per_second,
            performance_regression,
        }
    }

    /// Calculate summary statistics for all test results
    fn calculate_summary(&self, results: &[SingleComparisonResult]) -> ComparisonSummary {
        let total_tests = results.len();
        let successful_tests = results.iter().filter(|r| r.success).count();
        let failed_tests = total_tests - successful_tests;

        let average_token_accuracy = if total_tests > 0 {
            results.iter().map(|r| r.accuracy_result.token_accuracy).sum::<f64>()
                / total_tests as f64
        } else {
            0.0
        };

        let average_throughput_ratio = if total_tests > 0 {
            results.iter().map(|r| r.performance_comparison.throughput_ratio).sum::<f64>()
                / total_tests as f64
        } else {
            0.0
        };

        let average_memory_ratio = if total_tests > 0 {
            results.iter().map(|r| r.performance_comparison.memory_ratio).sum::<f64>()
                / total_tests as f64
        } else {
            0.0
        };

        let tests_passing_tolerance =
            results.iter().filter(|r| r.accuracy_result.passes_tolerance).count();

        let first_failure = results.iter().find(|r| !r.success).map(|r| r.test_case.name.clone());

        ComparisonSummary {
            total_tests,
            successful_tests,
            failed_tests,
            average_token_accuracy,
            average_throughput_ratio,
            average_memory_ratio,
            tests_passing_tolerance,
            first_failure,
        }
    }
}

/// Utility functions for creating common test cases
pub mod test_cases {
    use super::*;

    /// Create basic inference test cases
    pub fn create_basic_test_cases() -> Vec<ComparisonTestCase> {
        vec![
            ComparisonTestCase::new(
                "simple_greeting",
                "Hello, how are you?",
                InferenceConfig::default(),
            )
            .with_description("Simple greeting test"),
            ComparisonTestCase::new(
                "short_completion",
                "The capital of France is",
                InferenceConfig {
                    max_tokens: 10,
                    temperature: 0.0, // Deterministic
                    ..Default::default()
                },
            )
            .with_description("Short factual completion"),
            ComparisonTestCase::new(
                "creative_writing",
                "Once upon a time in a distant galaxy",
                InferenceConfig { max_tokens: 50, temperature: 0.7, ..Default::default() },
            )
            .with_description("Creative writing prompt"),
            ComparisonTestCase::new(
                "code_completion",
                "def fibonacci(n):",
                InferenceConfig { max_tokens: 30, temperature: 0.1, ..Default::default() },
            )
            .with_description("Code completion test"),
            ComparisonTestCase::new(
                "empty_input",
                "",
                InferenceConfig { max_tokens: 5, ..Default::default() },
            )
            .with_description("Empty input edge case"),
        ]
    }

    /// Create performance benchmark test cases
    pub fn create_performance_test_cases() -> Vec<ComparisonTestCase> {
        vec![
            ComparisonTestCase::new(
                "long_generation",
                "Write a detailed explanation of quantum computing:",
                InferenceConfig { max_tokens: 200, temperature: 0.3, ..Default::default() },
            )
            .with_description("Long text generation for performance testing"),
            ComparisonTestCase::new(
                "batch_processing",
                "Translate the following to French: Hello world",
                InferenceConfig { max_tokens: 20, temperature: 0.0, ..Default::default() },
            )
            .with_description("Batch processing simulation"),
        ]
    }

    /// Create edge case test cases
    pub fn create_edge_case_test_cases() -> Vec<ComparisonTestCase> {
        vec![
            ComparisonTestCase::new(
                "special_characters",
                "Test with émojis 🚀 and spëcial chars: @#$%",
                InferenceConfig { max_tokens: 15, ..Default::default() },
            )
            .with_description("Special characters and emojis"),
            ComparisonTestCase::new(
                "very_long_input",
                &"This is a very long input sentence that repeats itself. ".repeat(20),
                InferenceConfig { max_tokens: 10, ..Default::default() },
            )
            .with_description("Very long input text"),
            ComparisonTestCase::new(
                "single_token",
                "A",
                InferenceConfig { max_tokens: 1, ..Default::default() },
            )
            .with_description("Single token input"),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparison_tolerance_default() {
        let tolerance = ComparisonTolerance::default();
        assert_eq!(tolerance.min_token_accuracy, 0.95);
        assert_eq!(tolerance.max_probability_divergence, 0.1);
        assert_eq!(tolerance.max_performance_regression, 2.0);
        assert_eq!(tolerance.float_tolerance, 1e-6);
    }

    #[test]
    fn test_comparison_test_case_creation() {
        let test_case = ComparisonTestCase::new("test", "Hello world", InferenceConfig::default())
            .with_token_range(5, 15)
            .with_description("Test case");

        assert_eq!(test_case.name, "test");
        assert_eq!(test_case.input, "Hello world");
        assert_eq!(test_case.expected_min_tokens, Some(5));
        assert_eq!(test_case.expected_max_tokens, Some(15));
        assert_eq!(test_case.description, "Test case");
    }

    #[test]
    fn test_basic_test_cases_creation() {
        let test_cases = test_cases::create_basic_test_cases();
        assert!(!test_cases.is_empty());
        assert!(test_cases.iter().any(|tc| tc.name == "simple_greeting"));
        assert!(test_cases.iter().any(|tc| tc.name == "short_completion"));
    }

    #[test]
    fn test_probability_similarity_calculation() {
        use crate::cross_validation::test_implementation::MockImplementation;

        let suite = CrossValidationSuite {
            rust_impl: Box::new(MockImplementation::new("rust".to_string(), "1.0".to_string())),
            cpp_impl: Box::new(MockImplementation::new("cpp".to_string(), "1.0".to_string())),
            tolerance: ComparisonTolerance::default(),
            test_cases: Vec::new(),
            context_window: 5,
        };

        // Test identical distributions
        let probs1 = vec![0.5, 0.3, 0.2];
        let probs2 = vec![0.5, 0.3, 0.2];
        let similarity = suite.calculate_probability_similarity(&probs1, &probs2);
        assert!(similarity > 0.99); // Should be very close to 1.0

        // Test different distributions
        let probs3 = vec![0.8, 0.1, 0.1];
        let probs4 = vec![0.2, 0.4, 0.4];
        let similarity2 = suite.calculate_probability_similarity(&probs3, &probs4);
        assert!(similarity2 < 0.9); // Should be noticeably different
    }
}
