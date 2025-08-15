//! JSON report format implementation

use super::super::{ReportError, ReportFormat, ReportResult, TestReporter};
use crate::results::TestSuiteResult;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;
use tokio::fs;

/// JSON format reporter for machine processing
pub struct JsonReporter {
    pretty_print: bool,
}

impl JsonReporter {
    /// Create a new JSON reporter
    pub fn new() -> Self {
        Self { pretty_print: true }
    }

    /// Create a new JSON reporter with compact output
    pub fn new_compact() -> Self {
        Self { pretty_print: false }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonReport {
    metadata: ReportMetadata,
    test_suites: Vec<TestSuiteResult>,
    summary: GlobalSummary,
}

#[derive(Debug, Serialize, Deserialize)]
struct ReportMetadata {
    generated_at: String,
    generator: String,
    version: String,
    total_suites: usize,
    total_duration_ms: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct GlobalSummary {
    total_tests: usize,
    total_passed: usize,
    total_failed: usize,
    total_skipped: usize,
    overall_success_rate: f64,
    total_duration_ms: u64,
}

#[async_trait]
impl TestReporter for JsonReporter {
    async fn generate_report(
        &self,
        results: &[TestSuiteResult],
        output_path: &Path,
    ) -> Result<ReportResult, ReportError> {
        let start_time = Instant::now();
        super::super::reporter::prepare_output_path(output_path).await?;

        // Calculate global summary
        let total_tests: usize = results.iter().map(|r| r.summary.total_tests).sum();
        let total_passed: usize = results.iter().map(|r| r.summary.passed).sum();
        let total_failed: usize = results.iter().map(|r| r.summary.failed).sum();
        let total_skipped: usize = results.iter().map(|r| r.summary.skipped).sum();
        let total_duration_ms: u64 =
            results.iter().map(|r| r.total_duration.as_millis() as u64).sum();

        let overall_success_rate =
            if total_tests > 0 { total_passed as f64 / total_tests as f64 } else { 0.0 };

        let report = JsonReport {
            metadata: ReportMetadata {
                generated_at: chrono::Utc::now().to_rfc3339(),
                generator: "BitNet.rs Test Framework".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                total_suites: results.len(),
                total_duration_ms,
            },
            test_suites: results.to_vec(),
            summary: GlobalSummary {
                total_tests,
                total_passed,
                total_failed,
                total_skipped,
                overall_success_rate,
                total_duration_ms,
            },
        };

        let json_content = if self.pretty_print {
            serde_json::to_string_pretty(&report)?
        } else {
            serde_json::to_string(&report)?
        };

        fs::write(output_path, &json_content).await?;

        let generation_time = start_time.elapsed();
        let size_bytes = json_content.len() as u64;

        Ok(ReportResult {
            format: ReportFormat::Json,
            output_path: output_path.to_path_buf(),
            size_bytes,
            generation_time,
        })
    }

    fn format(&self) -> ReportFormat {
        ReportFormat::Json
    }

    fn file_extension(&self) -> &'static str {
        "json"
    }
}

impl Default for JsonReporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::results::{TestMetrics, TestResult, TestStatus, TestSummary};
    use std::collections::HashMap;
    use std::time::Duration;
    use tempfile::TempDir;

    fn create_test_suite_result() -> TestSuiteResult {
        TestSuiteResult {
            suite_name: "json_test_suite".to_string(),
            total_duration: Duration::from_secs(5),
            test_results: vec![
                TestResult {
                    test_name: "test_json_pass".to_string(),
                    status: TestStatus::Passed,
                    duration: Duration::from_secs(2),
                    metrics: TestMetrics {
                        memory_peak: Some(2048),
                        memory_average: Some(1024),
                        cpu_time: Some(Duration::from_secs(1)),
                        wall_time: Duration::from_secs(2),
                        custom_metrics: HashMap::new(),
                        assertions: 5,
                        operations: 10,
                    },
                    error: None,
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(2),
                    end_time: std::time::SystemTime::now(),
                    metadata: HashMap::new(),
                },
                TestResult {
                    test_name: "test_json_fail".to_string(),
                    status: TestStatus::Failed,
                    duration: Duration::from_secs(3),
                    metrics: TestMetrics {
                        memory_peak: Some(1024),
                        memory_average: Some(512),
                        cpu_time: Some(Duration::from_secs(2)),
                        wall_time: Duration::from_secs(3),
                        custom_metrics: HashMap::new(),
                        assertions: 3,
                        operations: 5,
                    },
                    error: Some("Expected true, got false".to_string()),
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(3),
                    end_time: std::time::SystemTime::now(),
                    metadata: HashMap::new(),
                },
            ],
            summary: TestSummary {
                total_tests: 2,
                passed: 1,
                failed: 1,
                skipped: 0,
                timeout: 0,
                success_rate: 50.0,
                total_duration: Duration::from_secs(5),
                average_duration: Duration::from_millis(2500),
                peak_memory: Some(2048),
                total_assertions: 8,
            },
            environment: HashMap::new(),
            configuration: HashMap::new(),
            start_time: std::time::SystemTime::now() - Duration::from_secs(5),
            end_time: std::time::SystemTime::now(),
        }
    }

    #[tokio::test]
    async fn test_json_report_generation() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_report.json");

        let reporter = JsonReporter::new();
        let results = vec![create_test_suite_result()];

        let report_result = reporter.generate_report(&results, &output_path).await.unwrap();

        assert_eq!(report_result.format, ReportFormat::Json);
        assert!(output_path.exists());
        assert!(report_result.size_bytes > 0);

        // Verify JSON content
        let content = fs::read_to_string(&output_path).await.unwrap();
        let parsed: JsonReport = serde_json::from_str(&content).unwrap();

        assert_eq!(parsed.test_suites.len(), 1);
        assert_eq!(parsed.summary.total_tests, 2);
        assert_eq!(parsed.summary.total_passed, 1);
        assert_eq!(parsed.summary.total_failed, 1);
        assert_eq!(parsed.summary.overall_success_rate, 0.5);
    }

    #[tokio::test]
    async fn test_compact_json_output() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("compact_report.json");

        let reporter = JsonReporter::new_compact();
        let results = vec![create_test_suite_result()];

        reporter.generate_report(&results, &output_path).await.unwrap();

        let content = fs::read_to_string(&output_path).await.unwrap();
        // Compact JSON should not have pretty formatting
        assert!(!content.contains("  "));
    }
}
