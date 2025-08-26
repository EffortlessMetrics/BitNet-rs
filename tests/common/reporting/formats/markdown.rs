//! Markdown report format implementation

use super::super::{ReportError, ReportFormat, ReportResult, TestReporter};
use crate::results::{TestResult, TestStatus, TestSuiteResult};
use async_trait::async_trait;
use std::path::Path;
use std::time::Instant;
use tokio::fs;

/// Markdown format reporter for documentation
pub struct MarkdownReporter {
    include_details: bool,
    include_metrics: bool,
}

impl MarkdownReporter {
    /// Create a new Markdown reporter
    pub fn new() -> Self {
        Self { include_details: true, include_metrics: true }
    }

    /// Create a new Markdown reporter with minimal output
    pub fn new_minimal() -> Self {
        Self { include_details: false, include_metrics: false }
    }

    /// Generate the complete Markdown report
    fn generate_markdown_content(
        &self,
        results: &[TestSuiteResult],
    ) -> Result<String, ReportError> {
        let mut markdown = String::new();

        // Header
        markdown.push_str("# BitNet.rs Test Report\n\n");
        markdown.push_str(&format!(
            "Generated on: {}\n\n",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ));

        // Summary
        markdown.push_str(&self.generate_summary(results));

        // Test suites
        markdown.push_str(&self.generate_test_suites(results));

        // Footer
        markdown.push_str(&self.generate_footer());

        Ok(markdown)
    }

    /// Generate summary section
    fn generate_summary(&self, results: &[TestSuiteResult]) -> String {
        let total_tests: usize = results.iter().map(|r| r.summary.total_tests).sum();
        let total_passed: usize = results.iter().map(|r| r.summary.passed).sum();
        let total_failed: usize = results.iter().map(|r| r.summary.failed).sum();
        let total_skipped: usize = results.iter().map(|r| r.summary.skipped).sum();
        let total_duration: f64 = results.iter().map(|r| r.total_duration.as_secs_f64()).sum();

        let success_rate =
            if total_tests > 0 { (total_passed as f64 / total_tests as f64) * 100.0 } else { 0.0 };

        let mut summary = String::new();
        summary.push_str("## Summary\n\n");
        summary.push_str("| Metric | Value |\n");
        summary.push_str("|--------|-------|\n");
        summary.push_str(&format!("| Total Tests | {} |\n", total_tests));
        summary.push_str(&format!("| Passed | {} |\n", total_passed));
        summary.push_str(&format!("| Failed | {} |\n", total_failed));
        summary.push_str(&format!("| Skipped | {} |\n", total_skipped));
        summary.push_str(&format!("| Success Rate | {:.1}% |\n", success_rate));
        summary.push_str(&format!("| Total Duration | {:.2}s |\n", total_duration));
        summary.push_str(&format!("| Test Suites | {} |\n", results.len()));
        summary.push_str("\n");

        summary
    }

    /// Generate test suites section
    fn generate_test_suites(&self, results: &[TestSuiteResult]) -> String {
        let mut markdown = String::new();
        markdown.push_str("## Test Suites\n\n");

        for suite in results {
            markdown.push_str(&self.generate_test_suite(suite));
        }

        markdown
    }

    /// Generate a single test suite
    fn generate_test_suite(&self, suite: &TestSuiteResult) -> String {
        let mut markdown = String::new();

        // Suite header
        markdown.push_str(&format!("### {}\n\n", suite.suite_name));

        // Suite summary
        let status_emoji =
            if suite.summary.failed == 0 && suite.summary.timeout == 0 { "✅" } else { "❌" };

        markdown.push_str(&format!(
            "{} **{}** tests: {} passed, {} failed, {} skipped ({:.1}% success rate) in {:.2}s\n\n",
            status_emoji,
            suite.summary.total_tests,
            suite.summary.passed,
            suite.summary.failed,
            suite.summary.skipped,
            suite.summary.success_rate,
            suite.total_duration.as_secs_f64()
        ));

        if self.include_details {
            // Test cases table
            markdown.push_str("| Test | Status | Duration | Details |\n");
            markdown.push_str("|------|--------|----------|----------|\n");

            for test in &suite.test_results {
                markdown.push_str(&self.generate_test_case_row(test));
            }
            markdown.push_str("\n");

            for test in &suite.test_results {
                if let Some(trace) = &test.stack_trace {
                    markdown.push_str(&format!(
                        "#### Stack trace for `{}`\n\n```\n{}\n```\n\n",
                        test.test_name, trace
                    ));
                }
            }
        }

        if self.include_metrics && suite.summary.peak_memory.is_some() {
            markdown.push_str("#### Metrics\n\n");
            markdown.push_str("| Metric | Value |\n");
            markdown.push_str("|--------|-------|\n");

            if let Some(peak_memory) = suite.summary.peak_memory {
                markdown.push_str(&format!("| Peak Memory | {} MB |\n", peak_memory / 1024 / 1024));
            }

            markdown
                .push_str(&format!("| Total Assertions | {} |\n", suite.summary.total_assertions));
            markdown.push_str(&format!(
                "| Average Duration | {:.3}s |\n",
                suite.summary.average_duration.as_secs_f64()
            ));
            markdown.push_str("\n");
        }

        markdown
    }

    /// Generate a single test case row
    fn generate_test_case_row(&self, test: &TestResult) -> String {
        let status_emoji = match test.status {
            TestStatus::Passed => "✅",
            TestStatus::Failed => "❌",
            TestStatus::Skipped => "⏭️",
            TestStatus::Timeout => "⏰",
            TestStatus::Running => "🔄",
        };

        let details = if let Some(error) = &test.error {
            format!("`{}`", error.chars().take(50).collect::<String>())
        } else {
            "-".to_string()
        };

        format!(
            "| `{}` | {} {:?} | {:.3}s | {} |\n",
            test.test_name,
            status_emoji,
            test.status,
            test.duration.as_secs_f64(),
            details
        )
    }

    /// Generate footer section
    fn generate_footer(&self) -> String {
        format!(
            "---\n\n*Generated by BitNet.rs Test Framework v{} on {}*\n",
            env!("CARGO_PKG_VERSION"),
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        )
    }
}

#[async_trait]
impl TestReporter for MarkdownReporter {
    async fn generate_report(
        &self,
        results: &[TestSuiteResult],
        output_path: &Path,
    ) -> Result<ReportResult, ReportError> {
        let start_time = Instant::now();
        super::super::reporter::prepare_output_path(output_path).await?;

        let markdown_content = self.generate_markdown_content(results)?;
        fs::write(output_path, &markdown_content).await?;

        let generation_time = start_time.elapsed();
        let size_bytes = markdown_content.len() as u64;

        Ok(ReportResult {
            format: ReportFormat::Markdown,
            output_path: output_path.to_path_buf(),
            size_bytes,
            generation_time,
        })
    }

    fn format(&self) -> ReportFormat {
        ReportFormat::Markdown
    }

    fn file_extension(&self) -> &'static str {
        "md"
    }
}

impl Default for MarkdownReporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::results::{TestMetrics, TestSummary};
    use std::collections::HashMap;
    use std::time::Duration;
    use tempfile::TempDir;

    fn create_test_suite_result() -> TestSuiteResult {
        TestSuiteResult {
            suite_name: "markdown_test_suite".to_string(),
            total_duration: Duration::from_secs(7),
            test_results: vec![
                TestResult {
                    test_name: "test_markdown_pass".to_string(),
                    status: TestStatus::Passed,
                    duration: Duration::from_secs(2),
                    metrics: TestMetrics {
                        memory_peak: Some(1024),
                        memory_average: Some(512),
                        cpu_time: Some(Duration::from_secs(1)),
                        wall_time: Duration::from_secs(2),
                        custom_metrics: HashMap::new(),
                        assertions: 4,
                        operations: 6,
                    },
                    error: None,
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(2),
                    end_time: std::time::SystemTime::now(),
                    metadata: HashMap::new(),
                },
                TestResult {
                    test_name: "test_markdown_fail".to_string(),
                    status: TestStatus::Failed,
                    duration: Duration::from_secs(3),
                    metrics: TestMetrics {
                        memory_peak: Some(2048),
                        memory_average: Some(1024),
                        cpu_time: Some(Duration::from_secs(2)),
                        wall_time: Duration::from_secs(3),
                        custom_metrics: HashMap::new(),
                        assertions: 2,
                        operations: 3,
                    },
                    error: Some(
                        "Markdown test assertion failed with detailed error message".to_string(),
                    ),
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(3),
                    end_time: std::time::SystemTime::now(),
                    metadata: HashMap::new(),
                },
                TestResult {
                    test_name: "test_markdown_skip".to_string(),
                    status: TestStatus::Skipped,
                    duration: Duration::from_secs(0),
                    metrics: TestMetrics::default(),
                    error: Some("Test skipped due to missing dependency".to_string()),
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now(),
                    end_time: std::time::SystemTime::now(),
                    metadata: HashMap::new(),
                },
            ],
            summary: TestSummary {
                total_tests: 3,
                passed: 1,
                failed: 1,
                skipped: 1,
                timeout: 0,
                success_rate: 33.33,
                total_duration: Duration::from_secs(7),
                average_duration: Duration::from_millis(2333),
                peak_memory: Some(2048),
                total_assertions: 6,
            },
            environment: HashMap::new(),
            configuration: HashMap::new(),
            start_time: std::time::SystemTime::now() - Duration::from_secs(7),
            end_time: std::time::SystemTime::now(),
        }
    }

    #[tokio::test]
    async fn test_markdown_report_generation() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_report.md");

        let reporter = MarkdownReporter::new();
        let results = vec![create_test_suite_result()];

        let report_result = reporter.generate_report(&results, &output_path).await.unwrap();

        assert_eq!(report_result.format, ReportFormat::Markdown);
        assert!(output_path.exists());
        assert!(report_result.size_bytes > 0);

        // Verify Markdown content structure
        let content = fs::read_to_string(&output_path).await.unwrap();
        assert!(content.contains("# BitNet.rs Test Report"));
        assert!(content.contains("## Summary"));
        assert!(content.contains("## Test Suites"));
        assert!(content.contains("### markdown_test_suite"));
        assert!(content.contains("| Test | Status | Duration | Details |"));
        assert!(content.contains("✅ Passed"));
        assert!(content.contains("❌ Failed"));
        assert!(content.contains("⏭️ Skipped"));
        assert!(content.contains("test_markdown_pass"));
        assert!(content.contains("test_markdown_fail"));
        assert!(content.contains("test_markdown_skip"));
    }

    #[tokio::test]
    async fn test_minimal_markdown_output() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("minimal_report.md");

        let reporter = MarkdownReporter::new_minimal();
        let results = vec![create_test_suite_result()];

        reporter.generate_report(&results, &output_path).await.unwrap();

        let content = fs::read_to_string(&output_path).await.unwrap();
        // Minimal output should not include detailed test case table
        assert!(!content.contains("| Test | Status | Duration | Details |"));
        // Should not include metrics section
        assert!(!content.contains("#### Metrics"));
    }

    #[tokio::test]
    async fn test_markdown_content_structure() {
        let reporter = MarkdownReporter::new();
        let results = vec![create_test_suite_result()];

        let content = reporter.generate_markdown_content(&results).unwrap();

        // Check for proper Markdown structure
        assert!(content.contains("# BitNet.rs Test Report"));
        assert!(content.contains("## Summary"));
        assert!(content.contains("| Metric | Value |"));
        assert!(content.contains("|--------|-------|"));
        assert!(content.contains("## Test Suites"));
        assert!(content.contains("### markdown_test_suite"));

        // Check for emojis and status indicators
        assert!(content.contains("✅"));
        assert!(content.contains("❌"));
        assert!(content.contains("⏭️"));

        // Check for proper table formatting
        assert!(content.contains("| `test_markdown_pass` | ✅ Passed |"));
        assert!(content.contains("| `test_markdown_fail` | ❌ Failed |"));
        assert!(content.contains("| `test_markdown_skip` | ⏭️ Skipped |"));
    }
}
