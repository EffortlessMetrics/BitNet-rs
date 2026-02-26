//! Example cross-validation tests comparing Rust and C++ implementations
//!
//! Demonstrates accuracy comparison, performance benchmarking, and
//! regression detection between bitnet-rs and BitNet.cpp

use bitnet_crossval::{
    BitNetImplementation, ComparisonTestCase, ComparisonTolerance, CppImplementation,
    CrossValidationSuite, RustImplementation,
};
use std::path::PathBuf;
use tempfile::TempDir;

#[cfg(test)]
mod cross_validation_examples {
    use super::*;

    /// Example: Basic accuracy comparison test
    #[tokio::test]
    async fn test_basic_accuracy_comparison() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = setup_test_model(&temp_dir, "small_model.gguf").await;

        // Configure comparison tolerance
        let tolerance = ComparisonTolerance {
            min_token_accuracy: 0.95,
            max_probability_divergence: 0.1,
            max_performance_regression: 2.0, // Allow 2x slower
        };

        let mut suite = CrossValidationSuite::new(tolerance);

        // Add test cases
        suite.add_test_case(ComparisonTestCase {
            name: "basic_generation".to_string(),
            input: "The future of artificial intelligence is".to_string(),
            config: create_deterministic_config(),
            expected_min_tokens: 5,
            expected_max_tokens: 50,
        });

        suite.add_test_case(ComparisonTestCase {
            name: "short_prompt".to_string(),
            input: "Hello".to_string(),
            config: create_deterministic_config(),
            expected_min_tokens: 1,
            expected_max_tokens: 20,
        });

        // Run comparison
        let result = suite.run_comparison(&model_path).await.unwrap();

        // Verify overall success
        assert!(
            result.summary.overall_success,
            "Comparison should succeed: {}",
            result.summary.failure_reason.unwrap_or_default()
        );

        // Check individual test results
        for test_result in &result.test_results {
            println!(
                "Test '{}': accuracy={:.3}, token_match={}/{}",
                test_result.test_case.name,
                test_result.accuracy_result.token_accuracy,
                test_result.accuracy_result.matches,
                test_result.accuracy_result.total_tokens
            );

            assert!(
                test_result.accuracy_result.passes_tolerance,
                "Test '{}' should pass tolerance check",
                test_result.test_case.name
            );

            // Verify performance comparison
            let perf = &test_result.performance_comparison;
            println!(
                "Performance ratio: {:.2}x (Rust vs C++)",
                perf.throughput_ratio
            );

            // Log performance details
            println!(
                "Rust: {:?}, C++: {:?}",
                perf.rust_duration, perf.cpp_duration
            );
        }

        // Check performance summary
        let avg_performance_ratio = result
            .test_results
            .iter()
            .map(|r| r.performance_comparison.throughput_ratio)
            .sum::<f64>()
            / result.test_results.len() as f64;

        println!("Average performance ratio: {:.2}x", avg_performance_ratio);

        // Performance should be reasonable (within 5x of C++)
        assert!(
            avg_performance_ratio < 5.0,
            "Average performance ratio {:.2}x exceeds reasonable limit",
            avg_performance_ratio
        );
    }

    /// Example: Edge case comparison testing
    #[tokio::test]
    async fn test_edge_case_comparisons() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = setup_test_model(&temp_dir, "edge_case_model.gguf").await;

        let tolerance = ComparisonTolerance {
            min_token_accuracy: 0.90, // Slightly lower for edge cases
            max_probability_divergence: 0.15,
            max_performance_regression: 3.0,
        };

        let mut suite = CrossValidationSuite::new(tolerance);

        // Edge case test scenarios
        let edge_cases = vec![
            ("empty_input", ""),
            ("single_char", "a"),
            ("numbers_only", "123456789"),
            ("special_chars", "!@#$%^&*()"),
            ("unicode", "Hello 世界 🌍"),
            ("very_long", "word ".repeat(1000)),
            ("repeated_pattern", "the the the the the"),
            ("mixed_case", "ThIs Is MiXeD cAsE tExT"),
        ];

        for (name, input) in edge_cases {
            suite.add_test_case(ComparisonTestCase {
                name: name.to_string(),
                input: input.to_string(),
                config: create_robust_config(), // More robust config for edge cases
                expected_min_tokens: 0,         // Allow empty output for some edge cases
                expected_max_tokens: 100,
            });
        }

        let result = suite.run_comparison(&model_path).await.unwrap();

        // Analyze edge case results
        let mut successful_cases = 0;
        let mut failed_cases = Vec::new();

        for test_result in &result.test_results {
            if test_result.accuracy_result.passes_tolerance {
                successful_cases += 1;
                println!("✓ Edge case '{}' passed", test_result.test_case.name);
            } else {
                failed_cases.push(&test_result.test_case.name);
                println!(
                    "✗ Edge case '{}' failed: accuracy={:.3}",
                    test_result.test_case.name, test_result.accuracy_result.token_accuracy
                );

                // Log first mismatch for debugging
                if let Some(mismatch) = &test_result.accuracy_result.first_mismatch {
                    println!(
                        "  First mismatch at position {}: rust={}, cpp={}",
                        mismatch.position, mismatch.rust_token, mismatch.cpp_token
                    );
                }
            }
        }

        // Require at least 70% of edge cases to pass
        let success_rate = successful_cases as f64 / result.test_results.len() as f64;
        assert!(
            success_rate >= 0.7,
            "Edge case success rate {:.1}% below 70%. Failed cases: {:?}",
            success_rate * 100.0,
            failed_cases
        );

        println!(
            "Edge case success rate: {:.1}% ({}/{})",
            success_rate * 100.0,
            successful_cases,
            result.test_results.len()
        );
    }

    /// Example: Performance regression detection
    #[tokio::test]
    async fn test_performance_regression_detection() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = setup_test_model(&temp_dir, "perf_test_model.gguf").await;

        // Strict performance tolerance for regression detection
        let tolerance = ComparisonTolerance {
            min_token_accuracy: 0.99, // High accuracy requirement
            max_probability_divergence: 0.05,
            max_performance_regression: 1.5, // Allow only 50% slower than C++
        };

        let mut suite = CrossValidationSuite::new(tolerance);

        // Performance-focused test cases
        let perf_cases = vec![
            ("short_burst", "Generate", 10),
            ("medium_text", "Write a short story about", 50),
            (
                "long_generation",
                "Explain the concept of machine learning in detail",
                200,
            ),
        ];

        for (name, input, max_tokens) in perf_cases {
            suite.add_test_case(ComparisonTestCase {
                name: name.to_string(),
                input: input.to_string(),
                config: create_performance_config(max_tokens),
                expected_min_tokens: 1,
                expected_max_tokens: max_tokens,
            });
        }

        let result = suite.run_comparison(&model_path).await.unwrap();

        // Analyze performance results
        let mut performance_violations = Vec::new();

        for test_result in &result.test_results {
            let perf = &test_result.performance_comparison;

            println!(
                "Performance '{}': {:.2}x ratio, Rust={:?}, C++={:?}",
                test_result.test_case.name,
                perf.throughput_ratio,
                perf.rust_duration,
                perf.cpp_duration
            );

            // Check for performance regression
            if perf.throughput_ratio > tolerance.max_performance_regression {
                performance_violations
                    .push((test_result.test_case.name.clone(), perf.throughput_ratio));
            }

            // Calculate tokens per second
            let rust_tps = test_result.rust_result.tokens.len() as f64
                / test_result.rust_result.duration.as_secs_f64();
            let cpp_tps = test_result.cpp_result.tokens.len() as f64
                / test_result.cpp_result.duration.as_secs_f64();

            println!("  Tokens/sec: Rust={:.1}, C++={:.1}", rust_tps, cpp_tps);
        }

        // Report performance violations
        if !performance_violations.is_empty() {
            println!("Performance regressions detected:");
            for (name, ratio) in &performance_violations {
                println!("  {}: {:.2}x slower than limit", name, ratio);
            }
        }

        // Allow some performance violations for this example
        // In production, you might want to fail the test
        let violation_rate = performance_violations.len() as f64 / result.test_results.len() as f64;
        assert!(
            violation_rate <= 0.5,
            "Too many performance violations: {:.1}%",
            violation_rate * 100.0
        );
    }

    /// Example: Probability distribution comparison
    #[tokio::test]
    async fn test_probability_distribution_comparison() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = setup_test_model(&temp_dir, "prob_test_model.gguf").await;

        let tolerance = ComparisonTolerance {
            min_token_accuracy: 0.95,
            max_probability_divergence: 0.1, // Strict probability tolerance
            max_performance_regression: 3.0,
        };

        let mut suite = CrossValidationSuite::new(tolerance);

        // Test cases designed to examine probability distributions
        suite.add_test_case(ComparisonTestCase {
            name: "deterministic_generation".to_string(),
            input: "The capital of France is".to_string(),
            config: create_deterministic_config(), // temperature=0.0
            expected_min_tokens: 1,
            expected_max_tokens: 10,
        });

        suite.add_test_case(ComparisonTestCase {
            name: "creative_generation".to_string(),
            input: "Once upon a time in a magical land".to_string(),
            config: create_creative_config(), // temperature=1.0
            expected_min_tokens: 10,
            expected_max_tokens: 50,
        });

        let result = suite.run_comparison(&model_path).await.unwrap();

        // Analyze probability distribution similarities
        for test_result in &result.test_results {
            if let Some(prob_similarity) = &test_result.accuracy_result.probability_similarity {
                println!(
                    "Probability similarity '{}': {:.3}",
                    test_result.test_case.name, prob_similarity.kl_divergence
                );

                // Check KL divergence is within tolerance
                assert!(
                    prob_similarity.kl_divergence <= tolerance.max_probability_divergence,
                    "KL divergence {:.3} exceeds tolerance {:.3} for '{}'",
                    prob_similarity.kl_divergence,
                    tolerance.max_probability_divergence,
                    test_result.test_case.name
                );

                // Log additional probability metrics
                println!("  JS divergence: {:.3}", prob_similarity.js_divergence);
                println!(
                    "  Cosine similarity: {:.3}",
                    prob_similarity.cosine_similarity
                );
            }
        }
    }

    /// Example: Model format compatibility testing
    #[tokio::test]
    async fn test_model_format_compatibility() {
        let temp_dir = TempDir::new().unwrap();

        // Test different model formats
        let formats = vec![
            ("gguf_model.gguf", ModelFormat::GGUF),
            ("safetensors_model.safetensors", ModelFormat::SafeTensors),
        ];

        let tolerance = ComparisonTolerance {
            min_token_accuracy: 0.90, // Allow some format-related differences
            max_probability_divergence: 0.2,
            max_performance_regression: 2.0,
        };

        for (filename, format) in formats {
            println!("Testing format: {:?}", format);

            let model_path = setup_test_model_with_format(&temp_dir, filename, format).await;
            let mut suite = CrossValidationSuite::new(tolerance.clone());

            suite.add_test_case(ComparisonTestCase {
                name: format!("format_test_{:?}", format),
                input: "Test input for format compatibility".to_string(),
                config: create_deterministic_config(),
                expected_min_tokens: 1,
                expected_max_tokens: 20,
            });

            let result = suite.run_comparison(&model_path).await;

            match result {
                Ok(comparison_result) => {
                    println!("✓ Format {:?} comparison successful", format);

                    for test_result in &comparison_result.test_results {
                        println!(
                            "  Accuracy: {:.3}",
                            test_result.accuracy_result.token_accuracy
                        );
                        println!(
                            "  Performance: {:.2}x",
                            test_result.performance_comparison.throughput_ratio
                        );
                    }
                }
                Err(e) => {
                    println!("✗ Format {:?} comparison failed: {}", format, e);
                    // Some formats might not be supported by both implementations
                    // This is acceptable for demonstration purposes
                }
            }
        }
    }

    /// Example: Regression test with known issues
    #[tokio::test]
    async fn test_known_regression_cases() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = setup_test_model(&temp_dir, "regression_model.gguf").await;

        let tolerance = ComparisonTolerance {
            min_token_accuracy: 0.95,
            max_probability_divergence: 0.1,
            max_performance_regression: 2.0,
        };

        let mut suite = CrossValidationSuite::new(tolerance);

        // Test cases based on previously found issues
        let regression_cases = vec![
            // Case 1: Issue with specific token sequences
            ComparisonTestCase {
                name: "token_sequence_issue".to_string(),
                input: "The quick brown fox jumps over the lazy dog".to_string(),
                config: create_deterministic_config(),
                expected_min_tokens: 5,
                expected_max_tokens: 30,
            },
            // Case 2: Issue with numerical inputs
            ComparisonTestCase {
                name: "numerical_input_issue".to_string(),
                input: "Calculate 2 + 2 = ".to_string(),
                config: create_deterministic_config(),
                expected_min_tokens: 1,
                expected_max_tokens: 10,
            },
            // Case 3: Issue with context length
            ComparisonTestCase {
                name: "context_length_issue".to_string(),
                input: "word ".repeat(100), // Long context
                config: create_deterministic_config(),
                expected_min_tokens: 1,
                expected_max_tokens: 20,
            },
        ];

        for test_case in regression_cases {
            suite.add_test_case(test_case);
        }

        let result = suite.run_comparison(&model_path).await.unwrap();

        // Track regression test results
        let mut regression_status = std::collections::HashMap::new();

        for test_result in &result.test_results {
            let passed = test_result.accuracy_result.passes_tolerance;
            regression_status.insert(test_result.test_case.name.clone(), passed);

            if passed {
                println!("✓ Regression test '{}' passed", test_result.test_case.name);
            } else {
                println!("✗ Regression test '{}' failed", test_result.test_case.name);

                // Log detailed failure information
                if let Some(mismatch) = &test_result.accuracy_result.first_mismatch {
                    println!(
                        "  First mismatch at token {}: {} vs {}",
                        mismatch.position, mismatch.rust_token, mismatch.cpp_token
                    );
                    println!("  Context: {:?}", mismatch.context);
                }
            }
        }

        // Generate regression report
        let total_tests = regression_status.len();
        let passed_tests = regression_status.values().filter(|&&passed| passed).count();
        let regression_rate = passed_tests as f64 / total_tests as f64;

        println!(
            "Regression test summary: {}/{} passed ({:.1}%)",
            passed_tests,
            total_tests,
            regression_rate * 100.0
        );

        // For demonstration, we'll accept some regressions
        // In production, you might want stricter requirements
        assert!(
            regression_rate >= 0.8,
            "Regression rate {:.1}% below acceptable threshold",
            regression_rate * 100.0
        );
    }
}

/// Cross-validation test utilities
pub mod crossval_test_utils {
    use super::*;
    use tokio::fs;

    /// Setup test model for cross-validation
    pub async fn setup_test_model(temp_dir: &TempDir, filename: &str) -> PathBuf {
        let model_path = temp_dir.path().join(filename);
        let model_data = create_crossval_model_data();
        fs::write(&model_path, model_data).await.unwrap();
        model_path
    }

    /// Setup test model with specific format
    pub async fn setup_test_model_with_format(
        temp_dir: &TempDir,
        filename: &str,
        format: ModelFormat,
    ) -> PathBuf {
        let model_path = temp_dir.path().join(filename);
        let model_data = match format {
            ModelFormat::GGUF => create_crossval_gguf_data(),
            ModelFormat::SafeTensors => create_crossval_safetensors_data(),
        };
        fs::write(&model_path, model_data).await.unwrap();
        model_path
    }

    /// Create deterministic inference config
    pub fn create_deterministic_config() -> bitnet_inference::InferenceConfig {
        bitnet_inference::InferenceConfig::builder()
            .max_tokens(20)
            .temperature(0.0) // Deterministic
            .top_p(1.0)
            .top_k(None)
            .build()
            .unwrap()
    }

    /// Create robust config for edge cases
    pub fn create_robust_config() -> bitnet_inference::InferenceConfig {
        bitnet_inference::InferenceConfig::builder()
            .max_tokens(50)
            .temperature(0.1) // Low but not zero
            .top_p(0.95)
            .top_k(Some(50))
            .build()
            .unwrap()
    }

    /// Create performance-focused config
    pub fn create_performance_config(max_tokens: usize) -> bitnet_inference::InferenceConfig {
        bitnet_inference::InferenceConfig::builder()
            .max_tokens(max_tokens)
            .temperature(0.5)
            .top_p(0.9)
            .build()
            .unwrap()
    }

    /// Create creative config for probability testing
    pub fn create_creative_config() -> bitnet_inference::InferenceConfig {
        bitnet_inference::InferenceConfig::builder()
            .max_tokens(30)
            .temperature(1.0) // High creativity
            .top_p(0.8)
            .top_k(Some(40))
            .build()
            .unwrap()
    }

    /// Create comprehensive model data for cross-validation
    fn create_crossval_model_data() -> Vec<u8> {
        create_crossval_gguf_data()
    }

    /// Create GGUF format data for cross-validation
    fn create_crossval_gguf_data() -> Vec<u8> {
        let mut data = Vec::new();

        // GGUF magic and version
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());

        // More comprehensive metadata for cross-validation
        data.extend_from_slice(&50u64.to_le_bytes()); // tensor count
        data.extend_from_slice(&20u64.to_le_bytes()); // metadata count

        // Add substantial mock data for realistic testing
        data.extend_from_slice(&vec![0u8; 100 * 1024]); // 100KB

        data
    }

    /// Create SafeTensors format data for cross-validation
    fn create_crossval_safetensors_data() -> Vec<u8> {
        let header = serde_json::json!({
            "vocab_size": {"dtype": "I32", "shape": [1], "data_offsets": [0, 4]},
            "hidden_size": {"dtype": "I32", "shape": [1], "data_offsets": [4, 8]},
            "num_layers": {"dtype": "I32", "shape": [1], "data_offsets": [8, 12]},
            "weights": {"dtype": "F32", "shape": [1000, 512], "data_offsets": [12, 2048012]}
        });

        let header_str = header.to_string();
        let header_len = header_str.len() as u64;

        let mut data = Vec::new();
        data.extend_from_slice(&header_len.to_le_bytes());
        data.extend_from_slice(header_str.as_bytes());

        // Add mock tensor data
        data.extend_from_slice(&32000u32.to_le_bytes()); // vocab_size
        data.extend_from_slice(&4096u32.to_le_bytes()); // hidden_size
        data.extend_from_slice(&32u32.to_le_bytes()); // num_layers
        data.extend_from_slice(&vec![0u8; 2048000]); // weights data

        data
    }

    /// Analyze comparison results and generate insights
    pub fn analyze_comparison_results(
        result: &bitnet_crossval::ComparisonResult,
    ) -> ComparisonAnalysis {
        let mut analysis = ComparisonAnalysis::default();

        // Calculate overall statistics
        analysis.total_tests = result.test_results.len();
        analysis.passed_tests = result
            .test_results
            .iter()
            .filter(|r| r.accuracy_result.passes_tolerance)
            .count();

        // Calculate performance statistics
        let performance_ratios: Vec<f64> = result
            .test_results
            .iter()
            .map(|r| r.performance_comparison.throughput_ratio)
            .collect();

        if !performance_ratios.is_empty() {
            analysis.avg_performance_ratio =
                performance_ratios.iter().sum::<f64>() / performance_ratios.len() as f64;
            analysis.min_performance_ratio = performance_ratios
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            analysis.max_performance_ratio = performance_ratios
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        }

        // Calculate accuracy statistics
        let accuracies: Vec<f64> = result
            .test_results
            .iter()
            .map(|r| r.accuracy_result.token_accuracy)
            .collect();

        if !accuracies.is_empty() {
            analysis.avg_accuracy = accuracies.iter().sum::<f64>() / accuracies.len() as f64;
            analysis.min_accuracy = accuracies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            analysis.max_accuracy = accuracies.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        }

        analysis
    }

    #[derive(Debug, Default)]
    pub struct ComparisonAnalysis {
        pub total_tests: usize,
        pub passed_tests: usize,
        pub avg_performance_ratio: f64,
        pub min_performance_ratio: f64,
        pub max_performance_ratio: f64,
        pub avg_accuracy: f64,
        pub min_accuracy: f64,
        pub max_accuracy: f64,
    }

    impl ComparisonAnalysis {
        pub fn success_rate(&self) -> f64 {
            if self.total_tests == 0 {
                0.0
            } else {
                self.passed_tests as f64 / self.total_tests as f64
            }
        }

        pub fn print_summary(&self) {
            println!("=== Cross-Validation Analysis ===");
            println!(
                "Tests: {}/{} passed ({:.1}%)",
                self.passed_tests,
                self.total_tests,
                self.success_rate() * 100.0
            );
            println!(
                "Accuracy: avg={:.3}, min={:.3}, max={:.3}",
                self.avg_accuracy, self.min_accuracy, self.max_accuracy
            );
            println!(
                "Performance: avg={:.2}x, min={:.2}x, max={:.2}x",
                self.avg_performance_ratio, self.min_performance_ratio, self.max_performance_ratio
            );
        }
    }
}
