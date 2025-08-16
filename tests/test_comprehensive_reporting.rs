//! Comprehensive test for the reporting system
//!
//! This test validates that all report formats (HTML, JSON, JUnit XML, Markdown)
//! can be generated successfully with comprehensive test data.

use bitnet_tests::prelude::*;
use bitnet_tests::reporting::{
    formats::{HtmlReporter, JsonReporter, JunitReporter, MarkdownReporter},
    ReportConfig, ReportFormat, ReportingManager, TestReporter,
};
use bitnet_tests::results::{TestMetrics, TestResult, TestStatus, TestSuiteResult, TestSummary};
use std::collections::HashMap;
use std::time::Duration;
use tempfile::TempDir;
use tokio::fs;
use bitnet_tests::units::{BYTES_PER_KB, BYTES_PER_MB, BYTES_PER_GB};

/// Create comprehensive test data for reporting
fn create_comprehensive_test_data() -> Vec<TestSuiteResult> {
    vec![
        // First test suite - mostly passing
        TestSuiteResult {
            suite_name: "bitnet_core_tests".to_string(),
            total_duration: Duration::from_secs(15),
            test_results: vec![
                TestResult {
                    test_name: "test_model_loading".to_string(),
                    status: TestStatus::Passed,
                    duration: Duration::from_secs(3),
                    metrics: TestMetrics {
                        memory_peak: Some(512 * BYTES_PER_MB), // 512 MB
                        memory_average: Some(256 * BYTES_PER_MB), // 256 MB
                        cpu_time: Some(Duration::from_secs(2)),
                        wall_time: Duration::from_secs(3),
                        custom_metrics: {
                            let mut metrics = HashMap::new();
                            metrics.insert("model_size_mb".to_string(), 128.5);
                            metrics.insert("load_throughput_mbps".to_string(), 42.8);
                            metrics
                        },
                        assertions: 8,
                        operations: 15,
                    },
                    error: None,
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(3),
                    end_time: std::time::SystemTime::now(),
                    metadata: {
                        let mut metadata = HashMap::new();
                        metadata.insert("model_type".to_string(), "bitnet_1b5".to_string());
                        metadata.insert("quantization".to_string(), "1bit".to_string());
                        metadata
                    },
                },
                TestResult {
                    test_name: "test_inference_accuracy".to_string(),
                    status: TestStatus::Passed,
                    duration: Duration::from_secs(8),
                    metrics: TestMetrics {
                        memory_peak: Some(BYTES_PER_MB * 1024), // 1 GB
                        memory_average: Some(768 * BYTES_PER_MB), // 768 MB
                        cpu_time: Some(Duration::from_secs(7)),
                        wall_time: Duration::from_secs(8),
                        custom_metrics: {
                            let mut metrics = HashMap::new();
                            metrics.insert("accuracy_score".to_string(), 0.987);
                            metrics.insert("tokens_per_second".to_string(), 125.3);
                            metrics.insert("perplexity".to_string(), 12.4);
                            metrics
                        },
                        assertions: 25,
                        operations: 50,
                    },
                    error: None,
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(8),
                    end_time: std::time::SystemTime::now(),
                    metadata: {
                        let mut metadata = HashMap::new();
                        metadata.insert("test_dataset".to_string(), "wikitext-103".to_string());
                        metadata.insert("sequence_length".to_string(), "2048".to_string());
                        metadata
                    },
                },
                TestResult {
                    test_name: "test_tokenization".to_string(),
                    status: TestStatus::Passed,
                    duration: Duration::from_secs(2),
                    metrics: TestMetrics {
                        memory_peak: Some(64 * BYTES_PER_MB), // 64 MB
                        memory_average: Some(32 * BYTES_PER_MB), // 32 MB
                        cpu_time: Some(Duration::from_millis(1500)),
                        wall_time: Duration::from_secs(2),
                        custom_metrics: {
                            let mut metrics = HashMap::new();
                            metrics.insert("tokens_processed".to_string(), 10000.0);
                            metrics.insert("tokenization_speed_tps".to_string(), 5000.0);
                            metrics
                        },
                        assertions: 12,
                        operations: 20,
                    },
                    error: None,
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(2),
                    end_time: std::time::SystemTime::now(),
                    metadata: HashMap::new(),
                },
                TestResult {
                    test_name: "test_memory_management".to_string(),
                    status: TestStatus::Failed,
                    duration: Duration::from_secs(2),
                    metrics: TestMetrics {
                        memory_peak: Some(2048 * BYTES_PER_MB), // 2 GB
                        memory_average: Some(1536 * BYTES_PER_MB), // 1.5 GB
                        cpu_time: Some(Duration::from_millis(1800)),
                        wall_time: Duration::from_secs(2),
                        custom_metrics: {
                            let mut metrics = HashMap::new();
                            metrics.insert("memory_leak_mb".to_string(), 128.0);
                            metrics.insert("gc_pressure".to_string(), 0.85);
                            metrics
                        },
                        assertions: 5,
                        operations: 8,
                    },
                    error: Some("Memory leak detected: 128MB not freed after test completion. Expected memory usage to return to baseline within 5% tolerance.".to_string()),
                    stack_trace: Some("at test_memory_management:45\n  at memory_allocator::check_leaks:123\n  at test_runner::cleanup:67".to_string()),
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(2),
                    end_time: std::time::SystemTime::now(),
                    metadata: {
                        let mut metadata = HashMap::new();
                        metadata.insert("memory_profiler".to_string(), "valgrind".to_string());
                        metadata.insert("leak_threshold_mb".to_string(), "10".to_string());
                        metadata
                    },
                },
            ],
            summary: TestSummary {
                total_tests: 4,
                passed: 3,
                failed: 1,
                skipped: 0,
                timeout: 0,
                success_rate: 75.0,
                total_duration: Duration::from_secs(15),
                average_duration: Duration::from_millis(3750),
                peak_memory: Some(2048 * BYTES_PER_MB),
                total_assertions: 50,
            },
            environment: {
                let mut env = HashMap::new();
                env.insert("rust_version".to_string(), "1.75.0".to_string());
                env.insert("target_triple".to_string(), "x86_64-unknown-linux-gnu".to_string());
                env.insert("cpu_model".to_string(), "Intel Core i7-12700K".to_string());
                env.insert("memory_total_gb".to_string(), "32".to_string());
                env
            },
            configuration: {
                let mut config = HashMap::new();
                config.insert("max_parallel_tests".to_string(), "4".to_string());
                config.insert("test_timeout_secs".to_string(), "300".to_string());
                config.insert("memory_limit_gb".to_string(), "8".to_string());
                config
            },
            start_time: std::time::SystemTime::now() - Duration::from_secs(15),
            end_time: std::time::SystemTime::now(),
        },
        // Second test suite - integration tests with mixed results
        TestSuiteResult {
            suite_name: "bitnet_integration_tests".to_string(),
            total_duration: Duration::from_secs(25),
            test_results: vec![
                TestResult {
                    test_name: "test_end_to_end_inference".to_string(),
                    status: TestStatus::Passed,
                    duration: Duration::from_secs(12),
                    metrics: TestMetrics {
                        memory_peak: Some(1536 * BYTES_PER_MB), // 1.5 GB
                        memory_average: Some(BYTES_PER_MB * 1024), // 1 GB
                        cpu_time: Some(Duration::from_secs(10)),
                        wall_time: Duration::from_secs(12),
                        custom_metrics: {
                            let mut metrics = HashMap::new();
                            metrics.insert("inference_latency_ms".to_string(), 45.2);
                            metrics.insert("throughput_tokens_per_sec".to_string(), 89.7);
                            metrics.insert("model_accuracy".to_string(), 0.943);
                            metrics
                        },
                        assertions: 35,
                        operations: 100,
                    },
                    error: None,
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(12),
                    end_time: std::time::SystemTime::now(),
                    metadata: {
                        let mut metadata = HashMap::new();
                        metadata.insert("test_type".to_string(), "e2e".to_string());
                        metadata.insert("model_variant".to_string(), "bitnet_3b".to_string());
                        metadata
                    },
                },
                TestResult {
                    test_name: "test_cross_platform_compatibility".to_string(),
                    status: TestStatus::Skipped,
                    duration: Duration::from_secs(0),
                    metrics: TestMetrics::default(),
                    error: Some("Test skipped: CUDA not available on this system".to_string()),
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now(),
                    end_time: std::time::SystemTime::now(),
                    metadata: {
                        let mut metadata = HashMap::new();
                        metadata.insert("skip_reason".to_string(), "missing_cuda".to_string());
                        metadata.insert("required_capability".to_string(), "gpu_compute".to_string());
                        metadata
                    },
                },
                TestResult {
                    test_name: "test_performance_regression".to_string(),
                    status: TestStatus::Timeout,
                    duration: Duration::from_secs(300), // 5 minutes timeout
                    metrics: TestMetrics {
                        memory_peak: Some(4096 * BYTES_PER_MB), // 4 GB
                        memory_average: Some(3072 * BYTES_PER_MB), // 3 GB
                        cpu_time: Some(Duration::from_secs(295)),
                        wall_time: Duration::from_secs(300),
                        custom_metrics: {
                            let mut metrics = HashMap::new();
                            metrics.insert("partial_progress".to_string(), 0.73);
                            metrics.insert("last_checkpoint_sec".to_string(), 287.0);
                            metrics
                        },
                        assertions: 15,
                        operations: 45,
                    },
                    error: Some("Test exceeded maximum execution time of 300 seconds. Performance regression detected: current implementation is 3.2x slower than baseline.".to_string()),
                    stack_trace: Some("at test_performance_regression:89\n  at benchmark_runner::execute:234\n  at timeout_handler::abort:12".to_string()),
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(300),
                    end_time: std::time::SystemTime::now(),
                    metadata: {
                        let mut metadata = HashMap::new();
                        metadata.insert("baseline_duration_sec".to_string(), "93".to_string());
                        metadata.insert("regression_factor".to_string(), "3.2".to_string());
                        metadata.insert("timeout_reason".to_string(), "performance_regression".to_string());
                        metadata
                    },
                },
            ],
            summary: TestSummary {
                total_tests: 3,
                passed: 1,
                failed: 0,
                skipped: 1,
                timeout: 1,
                success_rate: 33.33,
                total_duration: Duration::from_secs(25),
                average_duration: Duration::from_millis(8333),
                peak_memory: Some(4096 * BYTES_PER_MB),
                total_assertions: 50,
            },
            environment: {
                let mut env = HashMap::new();
                env.insert("rust_version".to_string(), "1.75.0".to_string());
                env.insert("target_triple".to_string(), "x86_64-unknown-linux-gnu".to_string());
                env.insert("cuda_available".to_string(), "false".to_string());
                env.insert("test_environment".to_string(), "ci".to_string());
                env
            },
            configuration: {
                let mut config = HashMap::new();
                config.insert("max_parallel_tests".to_string(), "2".to_string());
                config.insert("test_timeout_secs".to_string(), "300".to_string());
                config.insert("performance_baseline_enabled".to_string(), "true".to_string());
                config
            },
            start_time: std::time::SystemTime::now() - Duration::from_secs(25),
            end_time: std::time::SystemTime::now(),
        },
    ]
}

#[tokio::test]
async fn test_html_report_generation() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("comprehensive_report.html");

    let reporter = HtmlReporter::new(true); // Interactive mode
    let test_data = create_comprehensive_test_data();

    let result = reporter.generate_report(&test_data, &output_path).await.unwrap();

    // Verify report was generated
    assert_eq!(result.format, ReportFormat::Html);
    assert!(output_path.exists());
    assert!(result.size_bytes > 1000); // Should be substantial

    // Verify HTML content
    let content = fs::read_to_string(&output_path).await.unwrap();

    // Check HTML structure
    assert!(content.contains("<!DOCTYPE html>"));
    assert!(content.contains("<title>BitNet.rs Test Report</title>"));
    assert!(content.contains("BitNet.rs Test Report"));

    // Check test suite names
    assert!(content.contains("bitnet_core_tests"));
    assert!(content.contains("bitnet_integration_tests"));

    // Check test names
    assert!(content.contains("test_model_loading"));
    assert!(content.contains("test_inference_accuracy"));
    assert!(content.contains("test_memory_management"));
    assert!(content.contains("test_end_to_end_inference"));
    assert!(content.contains("test_cross_platform_compatibility"));
    assert!(content.contains("test_performance_regression"));

    // Check status indicators
    assert!(content.contains("status-badge passed"));
    assert!(content.contains("status-badge failed"));
    assert!(content.contains("status-badge skipped"));

    // Check interactive features
    assert!(content.contains("addEventListener"));
    assert!(content.contains("toggle"));

    println!("✅ HTML report generated successfully: {} bytes", result.size_bytes);
}

#[tokio::test]
async fn test_json_report_generation() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("comprehensive_report.json");

    let reporter = JsonReporter::new();
    let test_data = create_comprehensive_test_data();

    let result = reporter.generate_report(&test_data, &output_path).await.unwrap();

    // Verify report was generated
    assert_eq!(result.format, ReportFormat::Json);
    assert!(output_path.exists());
    assert!(result.size_bytes > 500);

    // Verify JSON content
    let content = fs::read_to_string(&output_path).await.unwrap();
    let json_value: serde_json::Value = serde_json::from_str(&content).unwrap();

    // Check JSON structure
    assert!(json_value["metadata"].is_object());
    assert!(json_value["test_suites"].is_array());
    assert!(json_value["summary"].is_object());

    // Check metadata
    assert_eq!(json_value["metadata"]["total_suites"], 2);
    assert!(json_value["metadata"]["generated_at"].is_string());

    // Check summary
    assert_eq!(json_value["summary"]["total_tests"], 7);
    assert_eq!(json_value["summary"]["total_passed"], 4);
    assert_eq!(json_value["summary"]["total_failed"], 1);
    assert_eq!(json_value["summary"]["total_skipped"], 1);

    // Check test suites
    let test_suites = json_value["test_suites"].as_array().unwrap();
    assert_eq!(test_suites.len(), 2);
    assert_eq!(test_suites[0]["suite_name"], "bitnet_core_tests");
    assert_eq!(test_suites[1]["suite_name"], "bitnet_integration_tests");

    println!("✅ JSON report generated successfully: {} bytes", result.size_bytes);
}

#[tokio::test]
async fn test_junit_report_generation() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("comprehensive_report.xml");

    let reporter = JunitReporter::new();
    let test_data = create_comprehensive_test_data();

    let result = reporter.generate_report(&test_data, &output_path).await.unwrap();

    // Verify report was generated
    assert_eq!(result.format, ReportFormat::Junit);
    assert!(output_path.exists());
    assert!(result.size_bytes > 500);

    // Verify XML content
    let content = fs::read_to_string(&output_path).await.unwrap();

    // Check XML structure
    assert!(content.contains("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"));
    assert!(content.contains("<testsuites"));
    assert!(content.contains("<testsuite"));
    assert!(content.contains("<testcase"));

    // Check test suite names
    assert!(content.contains("name=\"bitnet_core_tests\""));
    assert!(content.contains("name=\"bitnet_integration_tests\""));

    // Check test case names and statuses
    assert!(content.contains("name=\"test_model_loading\""));
    assert!(content.contains("name=\"test_memory_management\""));
    assert!(content.contains("<failure"));
    assert!(content.contains("<skipped"));
    assert!(content.contains("<error")); // For timeout

    // Check attributes
    assert!(content.contains("tests=\"7\""));
    assert!(content.contains("failures=\"1\""));

    println!("✅ JUnit XML report generated successfully: {} bytes", result.size_bytes);
}

#[tokio::test]
async fn test_markdown_report_generation() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("comprehensive_report.md");

    let reporter = MarkdownReporter::new();
    let test_data = create_comprehensive_test_data();

    let result = reporter.generate_report(&test_data, &output_path).await.unwrap();

    // Verify report was generated
    assert_eq!(result.format, ReportFormat::Markdown);
    assert!(output_path.exists());
    assert!(result.size_bytes > 500);

    // Verify Markdown content
    let content = fs::read_to_string(&output_path).await.unwrap();

    // Check Markdown structure
    assert!(content.contains("# BitNet.rs Test Report"));
    assert!(content.contains("## Summary"));
    assert!(content.contains("## Test Suites"));

    // Check summary table
    assert!(content.contains("| Metric | Value |"));
    assert!(content.contains("| Total Tests | 7 |"));
    assert!(content.contains("| Passed | 4 |"));
    assert!(content.contains("| Failed | 1 |"));

    // Check test suite sections
    assert!(content.contains("### bitnet_core_tests"));
    assert!(content.contains("### bitnet_integration_tests"));

    // Check status emojis
    assert!(content.contains("✅"));
    assert!(content.contains("❌"));
    assert!(content.contains("⏭️"));
    assert!(content.contains("⏰"));

    // Check test case table
    assert!(content.contains("| Test | Status | Duration | Details |"));
    assert!(content.contains("test_model_loading"));
    assert!(content.contains("test_memory_management"));

    println!("✅ Markdown report generated successfully: {} bytes", result.size_bytes);
}

#[tokio::test]
async fn test_reporting_manager_all_formats() {
    let temp_dir = TempDir::new().unwrap();

    let config = ReportConfig {
        output_dir: temp_dir.path().to_path_buf(),
        formats: vec![
            ReportFormat::Html,
            ReportFormat::Json,
            ReportFormat::Junit,
            ReportFormat::Markdown,
        ],
        include_artifacts: true,
        generate_coverage: false,
        interactive_html: true,
    };

    let manager = ReportingManager::new(config);
    let test_data = create_comprehensive_test_data();

    let results = manager.generate_all_reports(&test_data).await.unwrap();

    // Verify all formats were generated
    assert_eq!(results.len(), 4);

    let formats: Vec<ReportFormat> = results.iter().map(|r| r.format.clone()).collect();
    assert!(formats.contains(&ReportFormat::Html));
    assert!(formats.contains(&ReportFormat::Json));
    assert!(formats.contains(&ReportFormat::Junit));
    assert!(formats.contains(&ReportFormat::Markdown));

    // Verify all files exist
    for result in &results {
        assert!(result.output_path.exists());
        assert!(result.size_bytes > 0);
        println!(
            "✅ Generated {:?} report: {} bytes at {:?}",
            result.format, result.size_bytes, result.output_path
        );
    }

    // Verify summary report was created
    let summary_path = temp_dir.path().join("report_summary.md");
    assert!(summary_path.exists());

    let summary_content = fs::read_to_string(&summary_path).await.unwrap();
    assert!(summary_content.contains("# Test Report Summary"));
    assert!(summary_content.contains("## Generated Reports"));
    assert!(summary_content.contains("HTML"));
    assert!(summary_content.contains("JSON"));
    assert!(summary_content.contains("Junit"));
    assert!(summary_content.contains("Markdown"));

    println!("✅ All report formats generated successfully via ReportingManager");
}

#[tokio::test]
async fn test_report_content_consistency() {
    let temp_dir = TempDir::new().unwrap();
    let test_data = create_comprehensive_test_data();

    // Generate all formats
    let html_path = temp_dir.path().join("test.html");
    let json_path = temp_dir.path().join("test.json");
    let junit_path = temp_dir.path().join("test.xml");
    let markdown_path = temp_dir.path().join("test.md");

    let html_reporter = HtmlReporter::new(false);
    let json_reporter = JsonReporter::new();
    let junit_reporter = JunitReporter::new();
    let markdown_reporter = MarkdownReporter::new();

    html_reporter.generate_report(&test_data, &html_path).await.unwrap();
    json_reporter.generate_report(&test_data, &json_path).await.unwrap();
    junit_reporter.generate_report(&test_data, &junit_path).await.unwrap();
    markdown_reporter.generate_report(&test_data, &markdown_path).await.unwrap();

    // Read all contents
    let html_content = fs::read_to_string(&html_path).await.unwrap();
    let json_content = fs::read_to_string(&json_path).await.unwrap();
    let junit_content = fs::read_to_string(&junit_path).await.unwrap();
    let markdown_content = fs::read_to_string(&markdown_path).await.unwrap();

    // Verify consistent test names across all formats
    let test_names = [
        "test_model_loading",
        "test_inference_accuracy",
        "test_memory_management",
        "test_end_to_end_inference",
        "test_cross_platform_compatibility",
        "test_performance_regression",
    ];

    for test_name in &test_names {
        assert!(html_content.contains(test_name), "HTML missing {}", test_name);
        assert!(json_content.contains(test_name), "JSON missing {}", test_name);
        assert!(junit_content.contains(test_name), "JUnit missing {}", test_name);
        assert!(markdown_content.contains(test_name), "Markdown missing {}", test_name);
    }

    // Verify consistent suite names
    let suite_names = ["bitnet_core_tests", "bitnet_integration_tests"];
    for suite_name in &suite_names {
        assert!(html_content.contains(suite_name), "HTML missing suite {}", suite_name);
        assert!(json_content.contains(suite_name), "JSON missing suite {}", suite_name);
        assert!(junit_content.contains(suite_name), "JUnit missing suite {}", suite_name);
        assert!(markdown_content.contains(suite_name), "Markdown missing suite {}", suite_name);
    }

    println!("✅ All report formats contain consistent test and suite names");
}
