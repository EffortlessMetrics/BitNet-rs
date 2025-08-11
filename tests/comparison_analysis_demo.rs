//! Demonstration of comparison analysis reporting functionality
//!
//! This test demonstrates how to use the comparison analysis reporter
//! to generate comprehensive reports from cross-validation results.

use bitnet_tests::common::cross_validation::{
    implementation::{InferenceConfig, InferenceResult, PerformanceMetrics},
    AccuracyResult, ComparisonSummary, ComparisonTestCase, CrossValidationResult,
    PerformanceComparison, SingleComparisonResult, TokenMismatch,
};
use bitnet_tests::common::reporting::comparison_analysis::{
    ComparisonAnalysisConfig, ComparisonAnalysisReporter, StatusLevel,
};
use std::path::PathBuf;
use std::time::Duration;

#[tokio::test]
async fn test_comparison_analysis_report_generation() {
    // Create a sample cross-validation result
    let sample_result = create_sample_cross_validation_result();

    // Configure the analysis reporter
    let config = ComparisonAnalysisConfig {
        output_dir: PathBuf::from("test-reports/demo-analysis"),
        include_detailed_mismatches: true,
        include_performance_charts: true,
        include_regression_analysis: true,
        max_mismatch_details: 10,
        executive_summary: true,
        trend_analysis_days: 7,
    };

    let reporter = ComparisonAnalysisReporter::new(config);

    // Generate the analysis report
    let result = reporter.generate_analysis_report(&sample_result).await;

    match result {
        Ok(report) => {
            println!("✅ Analysis report generated successfully!");
            println!("Overall accuracy: {:.2}%", report.overall_accuracy * 100.0);
            println!("Test cases analyzed: {}", report.test_case_accuracies.len());
            println!(
                "Total mismatches: {}",
                report.mismatch_analysis.total_mismatches
            );
            println!("Recommendations: {}", report.recommendations.len());

            // Verify report structure
            assert!(!report.test_case_accuracies.is_empty());
            assert!(!report.recommendations.is_empty());
        }
        Err(e) => {
            eprintln!("❌ Failed to generate analysis report: {}", e);
            panic!("Analysis report generation failed");
        }
    }
}

#[tokio::test]
async fn test_accuracy_analysis_with_mismatches() {
    let mut sample_result = create_sample_cross_validation_result();

    // Add some mismatches to test the analysis
    for test_result in &mut sample_result.test_results {
        test_result.accuracy_result.first_mismatch = Some(TokenMismatch {
            position: 5,
            rust_token: 123,
            cpp_token: 456,
            rust_text: Some("hello".to_string()),
            cpp_text: Some("world".to_string()),
            context_before: vec![1, 2, 3, 4],
            context_after: vec![6, 7, 8, 9],
        });
        test_result.accuracy_result.token_accuracy = 0.85; // Reduced accuracy
        test_result.accuracy_result.matching_tokens = 85;
        test_result.accuracy_result.total_tokens = 100;
    }

    let config = ComparisonAnalysisConfig::default();
    let reporter = ComparisonAnalysisReporter::new(config);

    let result = reporter.generate_analysis_report(&sample_result).await;

    match result {
        Ok(report) => {
            println!("✅ Mismatch analysis completed!");
            println!(
                "Mismatch patterns found: {}",
                report.mismatch_analysis.mismatch_patterns.len()
            );
            println!(
                "Common positions: {:?}",
                report.mismatch_analysis.common_mismatch_positions
            );

            // Verify mismatch analysis
            assert!(report.mismatch_analysis.total_mismatches > 0);
            assert!(!report.mismatch_analysis.mismatch_patterns.is_empty());
        }
        Err(e) => {
            eprintln!("❌ Mismatch analysis failed: {}", e);
            panic!("Mismatch analysis failed");
        }
    }
}

#[tokio::test]
async fn test_performance_regression_analysis() {
    let mut sample_result = create_sample_cross_validation_result();

    // Add performance regressions
    for test_result in &mut sample_result.test_results {
        test_result.performance_comparison.throughput_ratio = 2.5; // 2.5x slower
        test_result.performance_comparison.performance_regression = true;
        test_result.performance_comparison.rust_tokens_per_second = 100.0;
        test_result.performance_comparison.cpp_tokens_per_second = 250.0;
    }

    let config = ComparisonAnalysisConfig::default();
    let mut reporter = ComparisonAnalysisReporter::new(config);

    // Add some historical data for regression analysis
    let historical_result = create_sample_cross_validation_result();
    reporter.load_historical_data(vec![historical_result]);

    let result = reporter.generate_analysis_report(&sample_result).await;

    match result {
        Ok(report) => {
            println!("✅ Performance regression analysis completed!");

            // Check that performance issues are detected
            let performance_issues = report
                .recommendations
                .iter()
                .filter(|r| r.contains("performance") || r.contains("regression"))
                .count();

            assert!(performance_issues > 0, "Should detect performance issues");
            println!(
                "Performance-related recommendations: {}",
                performance_issues
            );
        }
        Err(e) => {
            eprintln!("❌ Performance analysis failed: {}", e);
            panic!("Performance analysis failed");
        }
    }
}

#[test]
fn test_executive_summary_status_levels() {
    let config = ComparisonAnalysisConfig::default();
    let reporter = ComparisonAnalysisReporter::new(config);

    // Test different status level determinations
    let excellent = reporter.determine_overall_status(0.98, 1.1);
    let good = reporter.determine_overall_status(0.92, 1.3);
    let acceptable = reporter.determine_overall_status(0.85, 1.8);
    let concerning = reporter.determine_overall_status(0.75, 2.5);
    let critical = reporter.determine_overall_status(0.60, 4.0);

    println!("Status level classifications:");
    println!("  Excellent: {:?}", excellent);
    println!("  Good: {:?}", good);
    println!("  Acceptable: {:?}", acceptable);
    println!("  Concerning: {:?}", concerning);
    println!("  Critical: {:?}", critical);

    // Verify classifications
    assert!(matches!(excellent, StatusLevel::Excellent));
    assert!(matches!(good, StatusLevel::Good));
    assert!(matches!(acceptable, StatusLevel::Acceptable));
    assert!(matches!(concerning, StatusLevel::Concerning));
    assert!(matches!(critical, StatusLevel::Critical));
}

#[test]
fn test_accuracy_distribution_analysis() {
    let config = ComparisonAnalysisConfig::default();
    let reporter = ComparisonAnalysisReporter::new(config);

    // Test accuracy distribution calculation
    let scores = vec![0.95, 0.92, 0.98, 0.89, 0.96, 0.91, 0.94, 0.97, 0.93, 0.90];
    let distribution = reporter.calculate_accuracy_distribution(&scores);

    println!("Accuracy distribution analysis:");
    println!("  Mean: {:.3}", distribution.mean);
    println!("  Median: {:.3}", distribution.median);
    println!("  Std Dev: {:.3}", distribution.std_dev);
    println!("  Histogram buckets: {}", distribution.histogram.len());

    // Verify distribution calculation
    assert!(distribution.mean > 0.9);
    assert!(distribution.median > 0.9);
    assert!(distribution.std_dev > 0.0);
    assert_eq!(distribution.histogram.len(), 10);

    // Check percentiles
    assert!(distribution.percentiles.contains_key(&50));
    assert!(distribution.percentiles.contains_key(&95));
}

/// Create a sample cross-validation result for testing
fn create_sample_cross_validation_result() -> CrossValidationResult {
    let test_cases = vec![
        create_sample_test_result("basic_test", 0.95, 1.2),
        create_sample_test_result("performance_test", 0.92, 1.8),
        create_sample_test_result("edge_case_test", 0.88, 1.5),
        create_sample_test_result("simple_greeting", 0.98, 1.1),
        create_sample_test_result("code_completion", 0.91, 1.6),
    ];

    let summary = ComparisonSummary {
        total_tests: test_cases.len(),
        successful_tests: test_cases.iter().filter(|t| t.success).count(),
        failed_tests: test_cases.iter().filter(|t| !t.success).count(),
        average_token_accuracy: test_cases
            .iter()
            .map(|t| t.accuracy_result.token_accuracy)
            .sum::<f64>()
            / test_cases.len() as f64,
        average_throughput_ratio: test_cases
            .iter()
            .map(|t| t.performance_comparison.throughput_ratio)
            .sum::<f64>()
            / test_cases.len() as f64,
        average_memory_ratio: 1.1,
        tests_passing_tolerance: test_cases
            .iter()
            .filter(|t| t.accuracy_result.passes_tolerance)
            .count(),
        first_failure: test_cases
            .iter()
            .find(|t| !t.success)
            .map(|t| t.test_case.name.clone()),
    };

    CrossValidationResult {
        model_path: PathBuf::from("test_model.gguf"),
        model_name: "test_model".to_string(),
        tolerance: bitnet_tests::common::cross_validation::ComparisonTolerance::default(),
        test_results: test_cases,
        summary,
        rust_metrics: PerformanceMetrics::default(),
        cpp_metrics: PerformanceMetrics::default(),
        total_duration: Duration::from_secs(120),
        timestamp: "2024-01-15T10:30:00Z".to_string(),
    }
}

/// Create a sample test result
fn create_sample_test_result(
    name: &str,
    accuracy: f64,
    throughput_ratio: f64,
) -> SingleComparisonResult {
    SingleComparisonResult {
        test_case: ComparisonTestCase::new(name, "Sample input", InferenceConfig::default()),
        tokenization_match: true,
        rust_tokens: vec![1, 2, 3, 4, 5],
        cpp_tokens: vec![1, 2, 3, 4, 5],
        accuracy_result: AccuracyResult {
            token_accuracy: accuracy,
            total_tokens: 100,
            matching_tokens: (accuracy * 100.0) as usize,
            first_mismatch: None,
            probability_similarity: Some(0.95),
            logit_similarity: Some(0.92),
            passes_tolerance: accuracy >= 0.95,
            detailed_mismatches: Vec::new(),
        },
        performance_comparison: PerformanceComparison {
            rust_duration: Duration::from_millis((throughput_ratio * 100.0) as u64),
            cpp_duration: Duration::from_millis(100),
            throughput_ratio,
            rust_memory: 1024 * 1024,
            cpp_memory: 1024 * 1024,
            memory_ratio: 1.0,
            rust_tokens_per_second: 1000.0 / throughput_ratio,
            cpp_tokens_per_second: 1000.0,
            performance_regression: throughput_ratio > 1.5,
        },
        rust_result: InferenceResult {
            tokens: vec![1, 2, 3, 4, 5],
            text: "Sample output".to_string(),
            probabilities: Some(vec![0.8, 0.9, 0.7, 0.85, 0.92]),
            logits: None,
            duration: Duration::from_millis((throughput_ratio * 100.0) as u64),
            memory_usage: 1024 * 1024,
            token_count: 5,
        },
        cpp_result: InferenceResult {
            tokens: vec![1, 2, 3, 4, 5],
            text: "Sample output".to_string(),
            probabilities: Some(vec![0.8, 0.9, 0.7, 0.85, 0.92]),
            logits: None,
            duration: Duration::from_millis(100),
            memory_usage: 1024 * 1024,
            token_count: 5,
        },
        success: accuracy >= 0.9 && throughput_ratio <= 2.0,
        error: None,
    }
}
