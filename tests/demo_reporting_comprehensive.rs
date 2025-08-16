//! Comprehensive demonstration of the reporting system
//!
//! This demo shows all four report formats (HTML, JSON, JUnit XML, Markdown)
//! working together to generate comprehensive test reports.

use bitnet_tests::reporting::{
    formats::{HtmlReporter, JsonReporter, JunitReporter, MarkdownReporter},
    ReportConfig, ReportFormat, ReportingManager, TestReporter,
};
use bitnet_tests::results::{TestMetrics, TestResult, TestStatus, TestSuiteResult, TestSummary};
use std::collections::HashMap;
use std::time::Duration;
use tempfile::TempDir;
use tokio::fs;

// Import the canonical KB and MB constants from the test harness crate
use bitnet_tests::common::{BYTES_PER_KB, BYTES_PER_MB};
use bitnet_tests::common::units::{BYTES_PER_KB, BYTES_PER_MB, BYTES_PER_GB};

/// Create comprehensive test data for demonstration
fn create_demo_test_data() -> Vec<TestSuiteResult> {
    vec![
        // Core functionality tests
        TestSuiteResult {
            suite_name: "bitnet_core_functionality".to_string(),
            total_duration: Duration::from_secs(12),
            test_results: vec![
                TestResult {
                    test_name: "core::test_model_loading_gguf".to_string(),
                    status: TestStatus::Passed,
                    duration: Duration::from_secs(4),
                    metrics: TestMetrics {
                        memory_peak: Some(2 * BYTES_PER_MB), // 2MB
                        memory_average: Some(BYTES_PER_MB), // 1MB
                        cpu_time: Some(Duration::from_secs(3)),
                        wall_time: Duration::from_secs(4),
                        custom_metrics: {
                            let mut metrics = HashMap::new();
                            metrics.insert("model_size_mb".to_string(), 125.5);
                            metrics.insert("load_speed_mbps".to_string(), 31.375);
                            metrics
                        },
                        assertions: 12,
                        operations: 18,
                    },
                    error: None,
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(12),
                    end_time: std::time::SystemTime::now() - Duration::from_secs(8),
                    metadata: {
                        let mut meta = HashMap::new();
                        meta.insert("model_format".to_string(), "GGUF".to_string());
                        meta.insert("test_priority".to_string(), "high".to_string());
                        meta
                    },
                },
                TestResult {
                    test_name: "core::test_tokenization_performance".to_string(),
                    status: TestStatus::Passed,
                    duration: Duration::from_secs(3),
                    metrics: TestMetrics {
                        memory_peak: Some(512 * BYTES_PER_KB), // 512KB
                        memory_average: Some(256 * BYTES_PER_KB), // 256KB
                        cpu_time: Some(Duration::from_millis(2500)),
                        wall_time: Duration::from_secs(3),
                        custom_metrics: {
                            let mut metrics = HashMap::new();
                            metrics.insert("tokens_per_second".to_string(), 2500.0);
                            metrics.insert("vocab_coverage".to_string(), 99.8);
                            metrics
                        },
                        assertions: 8,
                        operations: 15,
                    },
                    error: None,
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(8),
                    end_time: std::time::SystemTime::now() - Duration::from_secs(5),
                    metadata: HashMap::new(),
                },
                TestResult {
                    test_name: "core::test_inference_accuracy".to_string(),
                    status: TestStatus::Failed,
                    duration: Duration::from_secs(5),
                    metrics: TestMetrics {
                        memory_peak: Some(4096 * BYTES_PER_KB), // 4MB
                        memory_average: Some(2048 * BYTES_PER_KB), // 2MB
                        cpu_time: Some(Duration::from_secs(4)),
                        wall_time: Duration::from_secs(5),
                        custom_metrics: {
                            let mut metrics = HashMap::new();
                            metrics.insert("accuracy_score".to_string(), 0.847);
                            metrics.insert("expected_accuracy".to_string(), 0.95);
                            metrics.insert("inference_time_ms".to_string(), 5000.0);
                            metrics
                        },
                        assertions: 5,
                        operations: 8,
                    },
                    error: Some("Accuracy below threshold: expected ≥0.95, got 0.847 (difference: -0.103)".to_string()),
                    stack_trace: Some("at core::inference::validate_accuracy (accuracy.rs:89)\nat core::test_inference_accuracy (test_core.rs:156)".to_string()),
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(5),
                    end_time: std::time::SystemTime::now(),
                    metadata: HashMap::new(),
                },
            ],
            summary: TestSummary {
                total_tests: 3,
                passed: 2,
                failed: 1,
                skipped: 0,
                timeout: 0,
                success_rate: 66.67,
                total_duration: Duration::from_secs(12),
                average_duration: Duration::from_secs(4),
                peak_memory: Some(4096 * BYTES_PER_KB),
                total_assertions: 25,
            },
            environment: {
                let mut env = HashMap::new();
                env.insert("rust_version".to_string(), "1.75.0".to_string());
                env.insert("target_os".to_string(), "windows".to_string());
                env.insert("cpu_arch".to_string(), "x86_64".to_string());
                env.insert("available_memory_gb".to_string(), "16".to_string());
                env
            },
            configuration: {
                let mut config = HashMap::new();
                config.insert("optimization_level".to_string(), "release".to_string());
                config.insert("features".to_string(), "default,gpu".to_string());
                config.insert("test_mode".to_string(), "comprehensive".to_string());
                config
            },
            start_time: std::time::SystemTime::now() - Duration::from_secs(12),
            end_time: std::time::SystemTime::now(),
        },
        // Performance benchmarks
        TestSuiteResult {
            suite_name: "bitnet_performance_benchmarks".to_string(),
            total_duration: Duration::from_secs(20),
            test_results: vec![
                TestResult {
                    test_name: "perf::benchmark_inference_throughput".to_string(),
                    status: TestStatus::Passed,
                    duration: Duration::from_secs(8),
                    metrics: TestMetrics {
                        memory_peak: Some(8192 * BYTES_PER_KB), // 8MB
                        memory_average: Some(4096 * BYTES_PER_KB), // 4MB
                        cpu_time: Some(Duration::from_secs(7)),
                        wall_time: Duration::from_secs(8),
                        custom_metrics: {
                            let mut metrics = HashMap::new();
                            metrics.insert("throughput_tokens_per_sec".to_string(), 125.0);
                            metrics.insert("latency_p50_ms".to_string(), 45.2);
                            metrics.insert("latency_p95_ms".to_string(), 89.7);
                            metrics.insert("latency_p99_ms".to_string(), 156.3);
                            metrics
                        },
                        assertions: 20,
                        operations: 50,
                    },
                    error: None,
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(20),
                    end_time: std::time::SystemTime::now() - Duration::from_secs(12),
                    metadata: HashMap::new(),
                },
                TestResult {
                    test_name: "perf::benchmark_memory_efficiency".to_string(),
                    status: TestStatus::Passed,
                    duration: Duration::from_secs(6),
                    metrics: TestMetrics {
                        memory_peak: Some(BYTES_PER_MB), // 1MB
                        memory_average: Some(768 * BYTES_PER_KB), // 768KB
                        cpu_time: Some(Duration::from_secs(5)),
                        wall_time: Duration::from_secs(6),
                        custom_metrics: {
                            let mut metrics = HashMap::new();
                            metrics.insert("memory_efficiency_score".to_string(), 0.92);
                            metrics.insert("peak_memory_mb".to_string(), 1.0);
                            metrics.insert("memory_fragmentation".to_string(), 0.08);
                            metrics
                        },
                        assertions: 15,
                        operations: 25,
                    },
                    error: None,
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(12),
                    end_time: std::time::SystemTime::now() - Duration::from_secs(6),
                    metadata: HashMap::new(),
                },
                TestResult {
                    test_name: "perf::benchmark_concurrent_requests".to_string(),
                    status: TestStatus::Timeout,
                    duration: Duration::from_secs(6),
                    metrics: TestMetrics {
                        memory_peak: Some(16384 * BYTES_PER_KB), // 16MB
                        memory_average: Some(8192 * BYTES_PER_KB), // 8MB
                        cpu_time: Some(Duration::from_secs(5)),
                        wall_time: Duration::from_secs(6),
                        custom_metrics: {
                            let mut metrics = HashMap::new();
                            metrics.insert("concurrent_requests".to_string(), 20.0);
                            metrics.insert("completed_requests".to_string(), 12.0);
                            metrics.insert("success_rate".to_string(), 0.60);
                            metrics
                        },
                        assertions: 10,
                        operations: 20,
                    },
                    error: Some("Test exceeded timeout of 6s while processing concurrent requests".to_string()),
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(6),
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
                total_duration: Duration::from_secs(20),
                average_duration: Duration::from_millis(6667),
                peak_memory: Some(16384 * BYTES_PER_KB),
                total_assertions: 45,
            },
            environment: {
                let mut env = HashMap::new();
                env.insert("rust_version".to_string(), "1.75.0".to_string());
                env.insert("target_os".to_string(), "windows".to_string());
                env.insert("cpu_arch".to_string(), "x86_64".to_string());
                env.insert("available_memory_gb".to_string(), "16".to_string());
                env
            },
            configuration: {
                let mut config = HashMap::new();
                config.insert("optimization_level".to_string(), "release".to_string());
                config.insert("features".to_string(), "performance,benchmarks".to_string());
                config.insert("test_mode".to_string(), "benchmark".to_string());
                config
            },
            start_time: std::time::SystemTime::now() - Duration::from_secs(20),
            end_time: std::time::SystemTime::now(),
        },
    ]
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 BitNet.rs Reporting System Comprehensive Demo");
    println!("================================================\n");

    let temp_dir = TempDir::new()?;
    let test_data = create_demo_test_data();

    println!("📊 Test Data Summary:");
    println!("  • {} test suites", test_data.len());
    let total_tests: usize = test_data.iter().map(|s| s.summary.total_tests).sum();
    let total_passed: usize = test_data.iter().map(|s| s.summary.passed).sum();
    let total_failed: usize = test_data.iter().map(|s| s.summary.failed).sum();
    let total_timeout: usize = test_data.iter().map(|s| s.summary.timeout).sum();
    println!(
        "  • {} total tests ({} passed, {} failed, {} timeout)",
        total_tests, total_passed, total_failed, total_timeout
    );
    println!();

    // Test individual reporters
    println!("🔧 Testing Individual Reporters:");
    println!("--------------------------------");

    // JSON Reporter
    let json_reporter = JsonReporter::new();
    let json_path = temp_dir.path().join("comprehensive_report.json");
    let json_result = json_reporter.generate_report(&test_data, &json_path).await?;
    println!(
        "✅ JSON Report: {} bytes in {:?}",
        json_result.size_bytes, json_result.generation_time
    );

    // HTML Reporter
    let html_reporter = HtmlReporter::new(true); // Interactive mode
    let html_path = temp_dir.path().join("comprehensive_report.html");
    let html_result = html_reporter.generate_report(&test_data, &html_path).await?;
    println!(
        "✅ HTML Report: {} bytes in {:?}",
        html_result.size_bytes, html_result.generation_time
    );

    // JUnit XML Reporter
    let junit_reporter = JunitReporter::new();
    let junit_path = temp_dir.path().join("comprehensive_report.xml");
    let junit_result = junit_reporter.generate_report(&test_data, &junit_path).await?;
    println!(
        "✅ JUnit XML Report: {} bytes in {:?}",
        junit_result.size_bytes, junit_result.generation_time
    );

    // Markdown Reporter
    let markdown_reporter = MarkdownReporter::new();
    let markdown_path = temp_dir.path().join("comprehensive_report.md");
    let markdown_result = markdown_reporter.generate_report(&test_data, &markdown_path).await?;
    println!(
        "✅ Markdown Report: {} bytes in {:?}",
        markdown_result.size_bytes, markdown_result.generation_time
    );

    println!();

    // Test ReportingManager
    println!("🎯 Testing ReportingManager:");
    println!("----------------------------");

    let manager_dir = temp_dir.path().join("manager_reports");
    let config = ReportConfig {
        output_dir: manager_dir.clone(),
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
    let manager_results = manager.generate_all_reports(&test_data).await?;

    println!("✅ ReportingManager generated {} reports:", manager_results.len());
    for result in &manager_results {
        println!(
            "   • {:?}: {} bytes in {:?}",
            result.format, result.size_bytes, result.generation_time
        );
    }

    // Verify all files exist
    assert!(manager_dir.join("test_report.html").exists());
    assert!(manager_dir.join("test_report.json").exists());
    assert!(manager_dir.join("test_report.xml").exists());
    assert!(manager_dir.join("test_report.md").exists());
    assert!(manager_dir.join("report_summary.md").exists());

    println!();

    // Display content samples
    println!("📄 Content Samples:");
    println!("-------------------");

    // JSON sample
    let json_content = fs::read_to_string(&json_path).await?;
    let json_parsed: serde_json::Value = serde_json::from_str(&json_content)?;
    println!("JSON Structure:");
    println!("  • Metadata: {}", json_parsed["metadata"]["generator"]);
    println!("  • Test Suites: {}", json_parsed["test_suites"].as_array().unwrap().len());
    println!("  • Summary - Total Tests: {}", json_parsed["summary"]["total_tests"]);
    println!(
        "  • Summary - Success Rate: {:.1}%",
        json_parsed["summary"]["overall_success_rate"].as_f64().unwrap() * 100.0
    );

    // HTML sample
    let html_content = fs::read_to_string(&html_path).await?;
    println!("\nHTML Features:");
    println!("  • Interactive: {}", html_content.contains("addEventListener"));
    println!("  • CSS Styling: {}", html_content.contains("background: linear-gradient"));
    println!("  • Test Suites: {}", html_content.matches("test-suite").count());
    println!("  • Status Badges: {}", html_content.matches("status-badge").count());

    // Markdown sample
    let markdown_content = fs::read_to_string(&markdown_path).await?;
    println!("\nMarkdown Elements:");
    println!("  • Headers: {}", markdown_content.matches("##").count());
    println!("  • Tables: {}", markdown_content.matches("|").count() / 10); // Rough estimate
    println!(
        "  • Emojis: {}",
        markdown_content.matches("✅").count() + markdown_content.matches("❌").count()
    );

    // JUnit XML sample
    let junit_content = fs::read_to_string(&junit_path).await?;
    println!("\nJUnit XML Structure:");
    println!("  • Test Suites: {}", junit_content.matches("<testsuite").count());
    println!("  • Test Cases: {}", junit_content.matches("<testcase").count());
    println!("  • Failures: {}", junit_content.matches("<failure").count());
    println!("  • Properties: {}", junit_content.matches("<property").count());

    println!();

    // Performance summary
    let total_generation_time: Duration = [
        json_result.generation_time,
        html_result.generation_time,
        junit_result.generation_time,
        markdown_result.generation_time,
    ]
    .iter()
    .sum();

    let total_size = json_result.size_bytes
        + html_result.size_bytes
        + junit_result.size_bytes
        + markdown_result.size_bytes;

    println!("📈 Performance Summary:");
    println!("----------------------");
    println!("  • Total Generation Time: {:?}", total_generation_time);
    println!("  • Total Report Size: {} bytes ({:.1} KB)", total_size, total_size as f64 / 1024.0);
    println!(
        "  • Average Generation Speed: {:.1} KB/s",
        (total_size as f64 / 1024.0) / total_generation_time.as_secs_f64()
    );

    println!();
    println!("🎉 All reports generated successfully!");
    println!("📁 Reports saved to: {}", temp_dir.path().display());
    println!("   (Note: Temporary directory will be cleaned up automatically)");

    Ok(())
}
