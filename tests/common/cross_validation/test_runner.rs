use super::test_cases::ModelSize;
#[cfg(feature = "cpp")]
use crate::cross_validation::CppImplementation;
use crate::cross_validation::test_cases::{
    ComparisonTestCaseRegistry, TestCaseCategory, test_suites,
};
use crate::cross_validation::{
    ComparisonTestCase, ComparisonTolerance, CrossValidationResult, CrossValidationSuite,
    RustImplementation,
};
use crate::errors::{TestError, TestOpResult as TestResultCompat};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Registry that discovers available GGUF models on disk
#[derive(Debug, Default)]
pub struct TestModelRegistry {
    models: Vec<(ModelSize, PathBuf)>,
}

impl TestModelRegistry {
    /// Discover models in the directory specified by the `TEST_MODEL_DIR` env var
    /// or the current working directory if the variable is not set.
    pub async fn new() -> Result<Self, TestError> {
        let base = std::env::var("TEST_MODEL_DIR").unwrap_or_else(|_| ".".to_string());
        Self::from_directory(Path::new(&base)).await
    }

    /// Create a registry from a specific directory
    pub async fn from_directory(dir: &Path) -> Result<Self, TestError> {
        let mut models = Vec::new();
        Self::scan_dir(dir, &mut models).map_err(TestError::IoError)?;
        Ok(Self { models })
    }

    /// Recursively scan a directory for `.gguf` files
    fn scan_dir(dir: &Path, models: &mut Vec<(ModelSize, PathBuf)>) -> Result<(), std::io::Error> {
        if !dir.exists() {
            return Ok(());
        }

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                Self::scan_dir(&path, models)?;
            } else if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
                let metadata = entry.metadata()?;
                let size = Self::classify_size(metadata.len());
                models.push((size, path));
            }
        }
        Ok(())
    }

    fn classify_size(bytes: u64) -> ModelSize {
        const MB: u64 = 1024 * 1024;
        let mb = bytes / MB;
        match mb {
            0..=100 => ModelSize::Tiny,
            101..=1024 => ModelSize::Small,
            1025..=10240 => ModelSize::Medium,
            _ => ModelSize::Large,
        }
    }

    /// Return all model paths for a given size
    pub fn by_size(&self, size: ModelSize) -> Vec<PathBuf> {
        self.models
            .iter()
            .filter_map(|(s, p)| if *s == size { Some(p.clone()) } else { None })
            .collect()
    }

    /// Return all discovered model paths
    pub fn all_models(&self) -> Vec<PathBuf> {
        self.models.iter().map(|(_, p)| p.clone()).collect()
    }
}

/// Comprehensive test runner for cross-implementation comparison
pub struct ComparisonTestRunner {
    suite: CrossValidationSuite,
    test_registry: ComparisonTestCaseRegistry,
    model_registry: TestModelRegistry,
    results: Vec<CrossValidationResult>,
}

impl ComparisonTestRunner {
    /// Create a new test runner with default configuration
    pub async fn new() -> TestResultCompat<Self> {
        let tolerance = ComparisonTolerance::default();
        let rust_impl = Box::new(RustImplementation::new());
        #[cfg(feature = "cpp")]
        let cpp_impl = Box::new(CppImplementation::new());
        #[cfg(not(feature = "cpp"))]
        let cpp_impl = Box::new(RustImplementation::new());
        let suite = CrossValidationSuite::new(rust_impl, cpp_impl, tolerance);
        let test_registry = ComparisonTestCaseRegistry::new();
        let model_registry = TestModelRegistry::new().await?;

        Ok(Self { suite, test_registry, model_registry, results: Vec::new() })
    }

    /// Create a new test runner with custom tolerance
    pub async fn with_tolerance(tolerance: ComparisonTolerance) -> TestResultCompat<Self> {
        let rust_impl = Box::new(RustImplementation::new());
        #[cfg(feature = "cpp")]
        let cpp_impl = Box::new(CppImplementation::new());
        #[cfg(not(feature = "cpp"))]
        let cpp_impl = Box::new(RustImplementation::new());
        let suite = CrossValidationSuite::new(rust_impl, cpp_impl, tolerance);
        let test_registry = ComparisonTestCaseRegistry::new();
        let model_registry = TestModelRegistry::new().await?;

        Ok(Self { suite, test_registry, model_registry, results: Vec::new() })
    }

    /// Run basic functionality tests
    pub async fn run_basic_tests(
        &mut self,
        model_path: &Path,
    ) -> TestResultCompat<CrossValidationResult> {
        let test_cases = test_suites::create_basic_suite();
        self.run_test_suite("Basic Functionality", test_cases, model_path).await
    }

    /// Run edge case tests
    pub async fn run_edge_case_tests(
        &mut self,
        model_path: &Path,
    ) -> TestResultCompat<CrossValidationResult> {
        let test_cases = test_suites::create_edge_case_suite();
        self.run_test_suite("Edge Cases", test_cases, model_path).await
    }

    /// Run performance benchmark tests
    pub async fn run_performance_tests(
        &mut self,
        model_path: &Path,
    ) -> TestResultCompat<CrossValidationResult> {
        let test_cases = test_suites::create_performance_suite();
        self.run_test_suite("Performance Benchmarks", test_cases, model_path).await
    }

    /// Run regression tests
    pub async fn run_regression_tests(
        &mut self,
        model_path: &Path,
    ) -> TestResultCompat<CrossValidationResult> {
        let test_cases = test_suites::create_regression_suite();
        self.run_test_suite("Regression Tests", test_cases, model_path).await
    }

    /// Run format compatibility tests
    pub async fn run_format_compatibility_tests(
        &mut self,
        model_path: &Path,
    ) -> TestResultCompat<CrossValidationResult> {
        let test_cases = test_suites::create_format_compatibility_suite();
        self.run_test_suite("Format Compatibility", test_cases, model_path).await
    }

    /// Run model size variation tests
    pub async fn run_model_size_tests(
        &mut self,
        model_path: &Path,
    ) -> TestResultCompat<CrossValidationResult> {
        let test_cases = test_suites::create_model_size_suite();
        self.run_test_suite("Model Size Variations", test_cases, model_path).await
    }

    /// Run smoke tests (quick validation)
    pub async fn run_smoke_tests(
        &mut self,
        model_path: &Path,
    ) -> TestResultCompat<CrossValidationResult> {
        let test_cases = test_suites::create_smoke_test_suite();
        self.run_test_suite("Smoke Tests", test_cases, model_path).await
    }

    /// Run comprehensive test suite (all categories)
    pub async fn run_comprehensive_tests(
        &mut self,
        model_path: &Path,
    ) -> TestResultCompat<CrossValidationResult> {
        let test_cases = test_suites::create_comprehensive_suite();
        self.run_test_suite("Comprehensive Tests", test_cases, model_path).await
    }

    /// Run tests for a specific model size
    pub async fn run_tests_for_model_size(
        &mut self,
        model_size: ModelSize,
    ) -> TestResultCompat<CrossValidationResult> {
        let model_path = self.model_registry.by_size(model_size).into_iter().next().ok_or(
            TestError::SetupError { message: format!("No model found for size {:?}", model_size) },
        )?;
        let test_cases = test_suites::create_suite_for_model_size(model_size);
        let suite_name = format!("Tests for {:?} Models", model_size);
        self.run_test_suite(&suite_name, test_cases, &model_path).await
    }

    /// Run tests by category
    pub async fn run_tests_by_category(
        &mut self,
        category: TestCaseCategory,
        model_path: &Path,
    ) -> TestResultCompat<CrossValidationResult> {
        let test_cases = self.test_registry.by_category(category);
        let suite_name = format!("{:?} Tests", category);
        self.run_test_suite(&suite_name, test_cases.into_iter().cloned().collect(), model_path)
            .await
    }

    /// Run a custom test suite
    pub async fn run_custom_test_suite(
        &mut self,
        suite_name: &str,
        test_cases: Vec<ComparisonTestCase>,
        model_path: &Path,
    ) -> TestResultCompat<CrossValidationResult> {
        self.run_test_suite(suite_name, test_cases, model_path).await
    }

    /// Internal method to run a test suite
    async fn run_test_suite(
        &mut self,
        suite_name: &str,
        test_cases: Vec<ComparisonTestCase>,
        model_path: &Path,
    ) -> TestResultCompat<CrossValidationResult> {
        println!("Running {} with {} test cases...", suite_name, test_cases.len());

        let start_time = Instant::now();

        // Clear previous test cases and add new ones
        self.suite.clear_test_cases();
        self.suite.add_test_cases(test_cases);

        // Run the comparison
        let result = self.suite.run_comparison(model_path).await.map_err(|e| {
            TestError::ExecutionError { message: format!("Comparison failed: {}", e) }
        })?;

        let duration = start_time.elapsed();
        println!("Completed {} in {:?}", suite_name, duration);

        // Store result
        self.results.push(result.clone());

        Ok(result)
    }

    /// Get all test results
    pub fn get_results(&self) -> &[CrossValidationResult] {
        &self.results
    }

    /// Clear all stored results
    pub fn clear_results(&mut self) {
        self.results.clear();
    }

    /// Get summary statistics across all results
    pub fn get_summary_statistics(&self) -> TestSummaryStatistics {
        if self.results.is_empty() {
            return TestSummaryStatistics::default();
        }

        let total_tests = self.results.iter().map(|r| r.test_results.len()).sum();
        let total_passed = self
            .results
            .iter()
            .flat_map(|r| &r.test_results)
            .filter(|tr| tr.accuracy_result.passes_tolerance)
            .count();

        let total_duration = self.results.iter().map(|r| r.total_duration).sum();

        let avg_token_accuracy = if total_tests > 0 {
            self.results
                .iter()
                .flat_map(|r| &r.test_results)
                .map(|tr| tr.accuracy_result.token_accuracy)
                .sum::<f64>()
                / total_tests as f64
        } else {
            0.0
        };

        let avg_throughput_ratio = if total_tests > 0 {
            self.results
                .iter()
                .flat_map(|r| &r.test_results)
                .map(|tr| tr.performance_comparison.throughput_ratio)
                .sum::<f64>()
                / total_tests as f64
        } else {
            1.0
        };

        TestSummaryStatistics {
            total_test_suites: self.results.len(),
            total_test_cases: total_tests,
            total_passed,
            total_failed: total_tests - total_passed,
            success_rate: if total_tests > 0 {
                total_passed as f64 / total_tests as f64
            } else {
                0.0
            },
            total_duration,
            average_token_accuracy: avg_token_accuracy,
            average_throughput_ratio: avg_throughput_ratio,
        }
    }

    /// Run a complete validation workflow for a model
    pub async fn run_complete_validation(
        &mut self,
        model_path: &Path,
    ) -> TestResultCompat<CompleteValidationResult> {
        println!("Starting complete validation workflow for model: {:?}", model_path);

        let start_time = Instant::now();
        let mut validation_results = Vec::new();

        // 1. Run smoke tests first (quick validation)
        println!("Step 1/6: Running smoke tests...");
        match self.run_smoke_tests(model_path).await {
            Ok(result) => {
                validation_results.push(("Smoke Tests".to_string(), result));
                println!("✓ Smoke tests completed successfully");
            }
            Err(e) => {
                println!("✗ Smoke tests failed: {}", e);
                return Err(e);
            }
        }

        // 2. Run basic functionality tests
        println!("Step 2/6: Running basic functionality tests...");
        match self.run_basic_tests(model_path).await {
            Ok(result) => {
                validation_results.push(("Basic Functionality".to_string(), result));
                println!("✓ Basic functionality tests completed");
            }
            Err(e) => {
                println!("✗ Basic functionality tests failed: {}", e);
                // Continue with other tests even if basic tests fail
            }
        }

        // 3. Run edge case tests
        println!("Step 3/6: Running edge case tests...");
        match self.run_edge_case_tests(model_path).await {
            Ok(result) => {
                validation_results.push(("Edge Cases".to_string(), result));
                println!("✓ Edge case tests completed");
            }
            Err(e) => {
                println!("✗ Edge case tests failed: {}", e);
            }
        }

        // 4. Run regression tests
        println!("Step 4/6: Running regression tests...");
        match self.run_regression_tests(model_path).await {
            Ok(result) => {
                validation_results.push(("Regression Tests".to_string(), result));
                println!("✓ Regression tests completed");
            }
            Err(e) => {
                println!("✗ Regression tests failed: {}", e);
            }
        }

        // 5. Run format compatibility tests
        println!("Step 5/6: Running format compatibility tests...");
        match self.run_format_compatibility_tests(model_path).await {
            Ok(result) => {
                validation_results.push(("Format Compatibility".to_string(), result));
                println!("✓ Format compatibility tests completed");
            }
            Err(e) => {
                println!("✗ Format compatibility tests failed: {}", e);
            }
        }

        // 6. Run performance tests (last as they take longest)
        println!("Step 6/6: Running performance tests...");
        match self.run_performance_tests(model_path).await {
            Ok(result) => {
                validation_results.push(("Performance Benchmarks".to_string(), result));
                println!("✓ Performance tests completed");
            }
            Err(e) => {
                println!("✗ Performance tests failed: {}", e);
            }
        }

        let total_duration = start_time.elapsed();
        let summary = self.get_summary_statistics();

        println!("Complete validation finished in {:?}", total_duration);
        println!("Overall success rate: {:.1}%", summary.success_rate * 100.0);

        Ok(CompleteValidationResult {
            model_path: model_path.to_path_buf(),
            validation_results,
            summary_statistics: summary,
            total_duration,
        })
    }
}

/// Summary statistics across multiple test runs
#[derive(Debug, Clone)]
pub struct TestSummaryStatistics {
    pub total_test_suites: usize,
    pub total_test_cases: usize,
    pub total_passed: usize,
    pub total_failed: usize,
    pub success_rate: f64,
    pub total_duration: std::time::Duration,
    pub average_token_accuracy: f64,
    pub average_throughput_ratio: f64,
}

impl Default for TestSummaryStatistics {
    fn default() -> Self {
        Self {
            total_test_suites: 0,
            total_test_cases: 0,
            total_passed: 0,
            total_failed: 0,
            success_rate: 0.0,
            total_duration: std::time::Duration::from_secs(0),
            average_token_accuracy: 0.0,
            average_throughput_ratio: 1.0,
        }
    }
}

/// Result of a complete validation workflow
#[derive(Debug, Clone)]
pub struct CompleteValidationResult {
    pub model_path: std::path::PathBuf,
    pub validation_results: Vec<(String, CrossValidationResult)>,
    pub summary_statistics: TestSummaryStatistics,
    pub total_duration: std::time::Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_test_runner_creation() {
        let runner = ComparisonTestRunner::new().await;
        assert!(runner.is_ok());

        let runner = runner.unwrap();
        assert_eq!(runner.results.len(), 0);
    }

    #[tokio::test]
    async fn test_custom_tolerance() {
        let tolerance = ComparisonTolerance {
            min_token_accuracy: 0.9,
            max_probability_divergence: 0.2,
            max_performance_regression: 3.0,
            float_tolerance: 1e-5,
        };

        let runner = ComparisonTestRunner::with_tolerance(tolerance).await;
        assert!(runner.is_ok());
    }

    #[test]
    fn test_summary_statistics_default() {
        let stats = TestSummaryStatistics::default();
        assert_eq!(stats.total_test_suites, 0);
        assert_eq!(stats.total_test_cases, 0);
        assert_eq!(stats.success_rate, 0.0);
        assert_eq!(stats.average_throughput_ratio, 1.0);
    }

    #[tokio::test]
    async fn test_test_registry_integration() {
        let runner = ComparisonTestRunner::new().await.unwrap();

        // Test that we can access different test categories
        let basic_tests = runner.test_registry.by_category(TestCaseCategory::Basic);
        let edge_tests = runner.test_registry.by_category(TestCaseCategory::EdgeCase);
        let perf_tests = runner.test_registry.by_category(TestCaseCategory::Performance);

        assert!(!basic_tests.is_empty());
        assert!(!edge_tests.is_empty());
        assert!(!perf_tests.is_empty());
    }

    #[tokio::test]
    async fn test_model_registry_discovery() {
        // create temporary models
        let dir = tempdir().unwrap();
        let tiny = dir.path().join("tiny-model.gguf");
        let tiny_file = File::create(&tiny).unwrap();
        tiny_file.set_len(1 * 1024 * 1024).unwrap();
        let small = dir.path().join("small-model.gguf");
        let small_file = File::create(&small).unwrap();
        small_file.set_len(110 * 1024 * 1024).unwrap();

        unsafe {
            std::env::set_var("TEST_MODEL_DIR", dir.path());
        }
        let runner = ComparisonTestRunner::new().await.unwrap();

        let tiny_models = runner.model_registry.by_size(ModelSize::Tiny);
        let small_models = runner.model_registry.by_size(ModelSize::Small);

        assert_eq!(tiny_models.len(), 1);
        assert_eq!(small_models.len(), 1);
        unsafe {
            std::env::remove_var("TEST_MODEL_DIR");
        }
    }

    #[tokio::test]
    async fn test_model_registry_missing_models() {
        let dir = tempdir().unwrap();
        unsafe {
            std::env::set_var("TEST_MODEL_DIR", dir.path());
        }
        let mut runner = ComparisonTestRunner::new().await.unwrap();
        assert!(runner.model_registry.by_size(ModelSize::Tiny).is_empty());
        assert!(runner.run_tests_for_model_size(ModelSize::Tiny).await.is_err());
        unsafe {
            std::env::remove_var("TEST_MODEL_DIR");
        }
    }
}
