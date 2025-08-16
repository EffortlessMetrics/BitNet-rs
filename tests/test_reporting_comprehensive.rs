//! Comprehensive test for the reporting system
//!
//! This test demonstrates the complete reporting system functionality
//! including HTML, JSON, JUnit XML, and Markdown report generation.

use bitnet_tests::reporting::{
    formats::{HtmlReporter, JsonReporter, JunitReporter, MarkdownReporter},
    ReportConfig, ReportFormat, TestReporter,
};
use bitnet_tests::results::{TestMetrics, TestResult, TestStatus, TestSuiteResult, TestSummary};
use std::collections::HashMap;
use std::time::Duration;
use tempfile::TempDir;
use tokio::fs;
use bitnet_tests::common::units::{BYTES_PER_KB, BYTES_PER_MB, BYTES_PER_GB};

/// Create comprehensive test data for reporting
fn create_comprehensive_test_data() -> Vec<TestSuiteResult> {
    vec![
        // First test suite - mostly passing
        TestSuiteResult {
            suite_name: "bitnet_core_tests".to_string(),
            total_duration: Duration::from_secs(15),
            test_results: vec![
                TestResult {
                    test_name: "core::test_model_loading".to_string(),
                    status: TestStatus::Passed,
                    duration: Duration::from_secs(3),
                    metrics: TestMetrics {
                        memory_peak: Some(BYTES_PER_MB), // 1MB
                        memory_average: Some(512 * BYTES_PER_KB), // 512KB
                        cpu_time: Some(Duration::from_secs(2)),
                        wall_time: Duration::from_secs(3),
                        custom_metrics: {
                            let mut metrics = HashMap::new();
                            metrics.insert("model_size_mb".to_string(), 50.0);
                            metrics.insert("load_speed_mbps".to_string(), 16.67);
                            metrics
                        },
                        assertions: 8,
                        operations: 12,
                    },
                    error: None,
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(15),
                    end_time: std::time::SystemTime::now() - Duration::from_secs(12),
                    metadata: {
                        let mut meta = HashMap::new();
                        meta.insert("test_category".to_string(), "core".to_string());
                        meta.insert("priority".to_string(), "high".to_string());
                        meta
                    },
                },
                TestResult {
                    test_name: "core::test_tokenization".to_string(),
                    status: TestStatus::Passed,
                    duration: Duration::from_secs(2),
                    metrics: TestMetrics {
                        memory_peak: Some(256 * BYTES_PER_KB), // 256KB
                        memory_average: Some(128 * BYTES_PER_KB), // 128KB
                        cpu_time: Some(Duration::from_millis(1500)),
                        wall_time: Duration::from_secs(2),
                        custom_metrics: {
                            let mut metrics = HashMap::new();
                            metrics.insert("tokens_per_second".to_string(), 1000.0);
                            metrics.insert("vocab_size".to_string(), 32000.0);
                            metrics
                        },
                        assertions: 5,
                        operations: 8,
                    },
                    error: None,
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(12),
                    end_time: std::time::SystemTime::now() - Duration::from_secs(10),
                    metadata: HashMap::new(),
                },
                TestResult {
                    test_name: "core::test_inference_basic".to_string(),
                    status: TestStatus::Failed,
                    duration: Duration::from_secs(8),
                    metrics: TestMetrics {
                        memory_peak: Some(2048 * BYTES_PER_KB), // 2MB
                        memory_average: Some(BYTES_PER_MB), // 1MB
                        cpu_time: Some(Duration::from_secs(6)),
                        wall_time: Duration::from_secs(8),
                        custom_metrics: {
                            let mut metrics = HashMap::new();
                            metrics.insert("inference_time_ms".to_string(), 8000.0);
                            metrics.insert("throughput_tokens_per_sec".to_string(), 12.5);
                            metrics
                        },
                        assertions: 3,
                        operations: 5,
                    },
                    error: Some("Inference accuracy below threshold: expected 0.95, got 0.87".to_string()),
                    stack_trace: Some("at core::inference::run_inference (inference.rs:142)\nat core::test_inference_basic (test.rs:89)".to_string()),
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(10),
                    end_time: std::time::SystemTime::now() - Duration::from_secs(2),
                    metadata: HashMap::new(),
                },
                TestResult {
                    test_name: "core::test_memory_management".to_string(),
                    status: TestStatus::Skipped,
                    duration: Duration::ZERO,
                    metrics: TestMetrics::default(),
                    error: Some("Skipped: requires GPU support".to_string()),
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(2),
                    end_time: std::time::SystemTime::now() - Duration::from_secs(2),
                    metadata: HashMap::new(),
                },
            ],
            summary: TestSummary {
                total_tests: 4,
                passed: 2,
                failed: 1,
                skipped: 1,
                timeout: 0,
                success_rate: 50.0,
                total_duration: Duration::from_secs(15),
                average_duration: Duration::from_millis(3750),
                peak_memory: Some(2048 * BYTES_PER_KB),
                total_assertions: 16,
            },
            environment: {
                let mut env = HashMap::new();
                env.insert("rust_version".to_string(), "1.75.0".to_string());
                env.insert("target_triple".to_string(), "x86_64-unknown-linux-gnu".to_string());
                env.insert("cpu_count".to_string(), "8".to_string());
                env
            },
            configuration: {
                let mut config = HashMap::new();
                config.insert("optimization_level".to_string(), "release".to_string());
                config.insert("features".to_string(), "default".to_string());
                config
            },
            start_time: std::time::SystemTime::now() - Duration::from_secs(15),
            end_time: std::time::SystemTime::now(),
        },
        // Second test suite - integration tests
        TestSuiteResult {
            suite_name: "bitnet_integration_tests".to_string(),
            total_duration: Duration::from_secs(25),
            test_results: vec![
                TestResult {
                    test_name: "integration::test_end_to_end_workflow".to_string(),
                    status: TestStatus::Passed,
                    duration: Duration::from_secs(12),
                    metrics: TestMetrics {
                        memory_peak: Some(4096 * BYTES_PER_KB), // 4MB
                        memory_average: Some(2048 * BYTES_PER_KB), // 2MB
                        cpu_time: Some(Duration::from_secs(10)),
                        wall_time: Duration::from_secs(12),
                        custom_metrics: {
                            let mut metrics = HashMap::new();
                            metrics.insert("total_tokens_processed".to_string(), 1000.0);
                            metrics.insert("end_to_end_latency_ms".to_string(), 12000.0);
                            metrics
                        },
                        assertions: 15,
                        operations: 25,
                    },
                    error: None,
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(25),
                    end_time: std::time::SystemTime::now() - Duration::from_secs(13),
                    metadata: HashMap::new(),
                },
                TestResult {
                    test_name: "integration::test_concurrent_inference".to_string(),
                    status: TestStatus::Timeout,
                    duration: Duration::from_secs(10),
                    metrics: TestMetrics {
                        memory_peak: Some(8192 * BYTES_PER_KB), // 8MB
                        memory_average: Some(4096 * BYTES_PER_KB), // 4MB
                        cpu_time: Some(Duration::from_secs(8)),
                        wall_time: Duration::from_secs(10),
                        custom_metrics: {
                            let mut metrics = HashMap::new();
                            metrics.insert("concurrent_requests".to_string(), 10.0);
                            metrics.insert("completed_requests".to_string(), 6.0);
                            metrics
                        },
                        assertions: 8,
                        operations: 15,
                    },
                    error: Some("Test exceeded timeout of 10s".to_string()),
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(13),
                    end_time: std::time::SystemTime::now() - Duration::from_secs(3),
                    metadata: HashMap::new(),
                },
                TestResult {
                    test_name: "integration::test_model_compatibility".to_string(),
                    status: TestStatus::Passed,
                    duration: Duration::from_secs(3),
                    metrics: TestMetrics {
                        memory_peak: Some(BYTES_PER_MB), // 1MB
                        memory_average: Some(512 * BYTES_PER_KB), // 512KB
                        cpu_time: Some(Duration::from_secs(2)),
                        wall_time: Duration::from_secs(3),
                        custom_metrics: {
                            let mut metrics = HashMap::new();
                            metrics.insert("models_tested".to_string(), 5.0);
                            metrics.insert("compatibility_rate".to_string(), 100.0);
                            metrics
                        },
                        assertions: 10,
                        operations: 20,
                    },
                    error: None,
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(3),
                    end_time: std::time::SystemTime::now(),
                    metadata: HashMap::new(),
                },
            ],
            summary: TestSummary {
                total_tests: 3,
                passed: 2,
                failed: 0,
                skipped: 0,
                timeout: 1,
                success_rate: 66.67,
                total_duration: Duration::from_secs(25),
                average_duration: Duration::from_millis(8333),
                peak_memory: Some(8192 * BYTES_PER_KB),
                total_assertions: 33,
            },
            environment: {
                let mut env = HashMap::new();
                env.insert("rust_version".to_string(), "1.75.0".to_string());
                env.insert("target_triple".to_string(), "x86_64-unknown-linux-gnu".to_string());
                env.insert("cpu_count".to_string(), "8".to_string());
                env
            },
            configuration: {
                let mut config = HashMap::new();
                config.insert("optimization_level".to_string(), "release".to_string());
                config.insert("features".to_string(), "integration".to_string());
                config
            },
            start_time: std::time::SystemTime::now() - Duration::from_secs(25),
            end_time: std::time::SystemTime::now(),
        },
    ]
}

#[tokio::test]
async fn test_comprehensive_html_report() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("comprehensive_report.html");

    let reporter = HtmlReporter::new(true); // Interactive mode
    let test_data = create_comprehensive_test_data();

    let result = reporter.generate_report(&test_data, &output_path).await.unwrap();

    assert_eq!(result.format, ReportFormat::Html);
    assert!(output_path.exists());
    assert!(result.size_bytes > 1000); // Should be substantial

    let content = fs::read_to_string(&output_path).await.unwrap();

    // Verify HTML structure
    assert!(content.contains("<!DOCTYPE html>"));
    assert!(content.contains("BitNet.rs Test Report"));

    // Verify test suite content
    assert!(content.contains("bitnet_core_tests"));
    assert!(content.contains("bitnet_integration_tests"));

    // Verify test cases
    assert!(content.contains("test_model_loading"));
    assert!(content.contains("test_inference_basic"));
    assert!(content.contains("test_end_to_end_workflow"));

    // Verify status indicators
    assert!(content.contains("status-badge passed"));
    assert!(content.contains("status-badge failed"));
    assert!(content.contains("status-badge skipped"));

    // Verify interactive features
    assert!(content.contains("addEventListener"));
    assert!(content.contains("toggle"));

    println!("✅ HTML report generated successfully: {} bytes", result.size_bytes);
}

#[tokio::test]
async fn test_comprehensive_json_report() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("comprehensive_report.json");

    let reporter = JsonReporter::new();
    let test_data = create_comprehensive_test_data();

    let result = reporter.generate_report(&test_data, &output_path).await.unwrap();

    assert_eq!(result.format, ReportFormat::Json);
    assert!(output_path.exists());
    assert!(result.size_bytes > 500);

    let content = fs::read_to_string(&output_path).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

    // Verify JSON structure
    assert!(parsed["metadata"].is_object());
    assert!(parsed["test_suites"].is_array());
    assert!(parsed["summary"].is_object());

    // Verify metadata
    assert_eq!(parsed["metadata"]["total_suites"], 2);
    assert_eq!(parsed["metadata"]["generator"], "BitNet.rs Test Framework");

    // Verify summary
    assert_eq!(parsed["summary"]["total_tests"], 7);
    assert_eq!(parsed["summary"]["total_passed"], 4);
    assert_eq!(parsed["summary"]["total_failed"], 1);
    assert_eq!(parsed["summary"]["total_skipped"], 1);

    // Verify test suites
    let test_suites = parsed["test_suites"].as_array().unwrap();
    assert_eq!(test_suites.len(), 2);
    assert_eq!(test_suites[0]["suite_name"], "bitnet_core_tests");
    assert_eq!(test_suites[1]["suite_name"], "bitnet_integration_tests");

    println!("✅ JSON report generated successfully: {} bytes", result.size_bytes);
}

#[tokio::test]
async fn test_comprehensive_junit_report() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("comprehensive_report.xml");

    let reporter = JunitReporter::new();
    let test_data = create_comprehensive_test_data();

    let result = reporter.generate_report(&test_data, &output_path).await.unwrap();

    assert_eq!(result.format, ReportFormat::Junit);
    assert!(output_path.exists());
    assert!(result.size_bytes > 500);

    let content = fs::read_to_string(&output_path).await.unwrap();

    // Verify XML structure
    assert!(content.contains("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"));
    assert!(content.contains("<testsuites"));
    assert!(content.contains("<testsuite"));
    assert!(content.contains("<testcase"));

    // Verify test suite attributes
    assert!(content.contains("name=\"bitnet_core_tests\""));
    assert!(content.contains("name=\"bitnet_integration_tests\""));

    // Verify test case structure
    assert!(content.contains("classname=\"core\""));
    assert!(content.contains("name=\"test_model_loading\""));
    assert!(content.contains("name=\"test_inference_basic\""));

    // Verify failure and skip elements
    assert!(content.contains("<failure"));
    assert!(content.contains("<skipped"));
    assert!(content.contains("message=\"Test was skipped\""));

    // Verify properties
    assert!(content.contains("<properties>"));
    assert!(content.contains("name=\"success_rate\""));

    println!("✅ JUnit XML report generated successfully: {} bytes", result.size_bytes);
}

#[tokio::test]
async fn test_comprehensive_markdown_report() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("comprehensive_report.md");

    let reporter = MarkdownReporter::new();
    let test_data = create_comprehensive_test_data();

    let result = reporter.generate_report(&test_data, &output_path).await.unwrap();

    assert_eq!(result.format, ReportFormat::Markdown);
    assert!(output_path.exists());
    assert!(result.size_bytes > 500);

    let content = fs::read_to_string(&output_path).await.unwrap();

    // Verify Markdown structure
    assert!(content.contains("# BitNet.rs Test Report"));
    assert!(content.contains("## Summary"));
    assert!(content.contains("## Test Suites"));

    // Verify summary table
    assert!(content.contains("| Metric | Value |"));
    assert!(content.contains("| Total Tests | 7 |"));
    assert!(content.contains("| Passed | 4 |"));
    assert!(content.contains("| Failed | 1 |"));

    // Verify test suites
    assert!(content.contains("### bitnet_core_tests"));
    assert!(content.contains("### bitnet_integration_tests"));

    // Verify status emojis
    assert!(content.contains("✅"));
    assert!(content.contains("❌"));
    assert!(content.contains("⏭️"));
    assert!(content.contains("⏰"));

    // Verify test case table
    assert!(content.contains("| Test | Status | Duration | Details |"));
    assert!(content.contains("| `core::test_model_loading` | ✅ Passed |"));
    assert!(content.contains("| `core::test_inference_basic` | ❌ Failed |"));

    // Verify metrics section
    assert!(content.contains("#### Metrics"));
    assert!(content.contains("| Peak Memory |"));
    assert!(content.contains("| Total Assertions |"));

    println!("✅ Markdown report generated successfully: {} bytes", result.size_bytes);
}

#[tokio::test]
async fn test_all_formats_together() {
    let temp_dir = TempDir::new().unwrap();
    let test_data = create_comprehensive_test_data();

    // Test all formats
    let formats = vec![
        ("html", Box::new(HtmlReporter::new(true)) as Box<dyn TestReporter>),
        ("json", Box::new(JsonReporter::new()) as Box<dyn TestReporter>),
        ("xml", Box::new(JunitReporter::new()) as Box<dyn TestReporter>),
        ("md", Box::new(MarkdownReporter::new()) as Box<dyn TestReporter>),
    ];

    let mut total_size = 0u64;
    let mut generation_times = Vec::new();

    for (extension, reporter) in formats {
        let output_path = temp_dir.path().join(format!("test_report.{}", extension));
        let result = reporter.generate_report(&test_data, &output_path).await.unwrap();

        assert!(output_path.exists());
        assert!(result.size_bytes > 0);

        total_size += result.size_bytes;
        generation_times.push(result.generation_time);

        println!(
            "✅ {} report: {} bytes in {:?}",
            extension.to_uppercase(),
            result.size_bytes,
            result.generation_time
        );
    }

    let total_time: Duration = generation_times.iter().sum();
    println!("📊 Total: {} bytes across all formats in {:?}", total_size, total_time);

    // Verify all files exist
    assert!(temp_dir.path().join("test_report.html").exists());
    assert!(temp_dir.path().join("test_report.json").exists());
    assert!(temp_dir.path().join("test_report.xml").exists());
    assert!(temp_dir.path().join("test_report.md").exists());
}

#[tokio::test]
async fn test_report_performance() {
    use std::time::Instant;

    let temp_dir = TempDir::new().unwrap();
    let test_data = create_comprehensive_test_data();

    // Test performance with larger dataset
    let mut large_test_data = Vec::new();
    for i in 0..10 {
        let mut suite = test_data[0].clone();
        suite.suite_name = format!("performance_test_suite_{}", i);
        large_test_data.push(suite);
    }

    let start = Instant::now();

    let reporter = JsonReporter::new();
    let output_path = temp_dir.path().join("performance_test.json");
    let result = reporter.generate_report(&large_test_data, &output_path).await.unwrap();

    let total_time = start.elapsed();

    println!(
        "📈 Performance test: {} test suites, {} bytes in {:?}",
        large_test_data.len(),
        result.size_bytes,
        total_time
    );

    // Should complete reasonably quickly
    assert!(total_time < Duration::from_secs(1));
    assert!(result.size_bytes > 10000); // Should be substantial
}
