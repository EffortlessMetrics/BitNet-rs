//! Markdown report format implementation for documentation

use crate::reporting::{ReportError, ReportFormat, ReportResult, TestReporter};
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
    /// Create a new Markdown reporter with full details
    pub fn new() -> Self {
        Self {
            include_details: true,
            include_metrics: true,
        }
    }

    /// Create a new Markdown reporter with summary only
    pub fn new_summary_only() -> Self {
        Self {
            include_details: false,
            include_metrics: false,
        }
    }

    /// Generate the complete Markdown report
    fn generate_markdown_content(
        &self,
        results: &[TestSuiteResult],
    ) -> Result<String, ReportError> {
        let mut markdown = String::new();

        // Title and metadata
        markdown.push_str("# BitNet.rs Test Report\n\n");
        markdown.push_str(&format!(
            "**Generated:** {}\n\n",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ));

        // Executive summary
        markdown.push_str(&self.generate_summary(results));

        // Test suites overview
        markdown.push_str(&self.generate_suites_overview(results));

        // Detailed results (if enabled)
        if self.include_details {
            markdown.push_str(&self.generate_detailed_results(results));
        }

        // Metrics section (if enabled)
        if self.include_metrics {
            markdown.push_str(&self.generate_metrics_section(results));
        }

        // Footer
        markdown.push_str(&self.generate_footer());

        Ok(markdown)
    }

    /// Generate executive summary section
    fn generate_summary(&self, results: &[TestSuiteResult]) -> String {
        let total_tests: usize = results.iter().map(|r| r.summary.total_tests).sum();
        let total_passed: usize = results.iter().map(|r| r.summary.passed).sum();
        let total_failed: usize = results.iter().map(|r| r.summary.failed).sum();
        let total_skipped: usize = results.iter().map(|r| r.summary.skipped).sum();
        let total_duration: f64 = results.iter().map(|r| r.total_duration.as_secs_f64()).sum();

        let success_rate = if total_tests > 0 {
            (total_passed as f64 / total_tests as f64) * 100.0
        } else {
            0.0
        };

        let status_emoji = if success_rate >= 95.0 {
            "✅"
        } else if success_rate >= 80.0 {
            "⚠️"
        } else {
            "❌"
        };

        format!(
            r#"## Executive Summary {status_emoji}

| Metric | Value |
|--------|-------|
| **Total Tests** | {total_tests} |
| **Passed** | {total_passed} |
| **Failed** | {total_failed} |
| **Skipped** | {total_skipped} |
| **Success Rate** | {success_rate:.1}% |
| **Total Duration** | {total_duration:.2}s |
| **Test Suites** | {suite_count} |

"#,
            status_emoji = status_emoji,
            total_tests = total_tests,
            total_passed = total_passed,
            total_failed = total_failed,
            total_skipped = total_skipped,
            success_rate = success_rate,
            total_duration = total_duration,
            suite_count = results.len()
        )
    }

    /// Generate test suites overview
    fn generate_suites_overview(&self, results: &[TestSuiteResult]) -> String {
        let mut markdown = String::new();

        markdown.push_str("## Test Suites Overview\n\n");
        markdown.push_str(
            "| Suite Name | Tests | Passed | Failed | Skipped | Success Rate | Duration |\n",
        );
        markdown.push_str(
            "|------------|-------|--------|--------|---------|--------------|----------|\n",
        );

        for suite in results {
            let status_emoji = if suite.summary.success_rate >= 0.95 {
                "✅"
            } else if suite.summary.success_rate >= 0.80 {
                "⚠️"
            } else {
                "❌"
            };

            markdown.push_str(&format!(
                "| {} {} | {} | {} | {} | {} | {:.1}% | {:.2}s |\n",
                status_emoji,
                suite.suite_name,
                suite.summary.total_tests,
                suite.summary.passed,
                suite.summary.failed,
                suite.summary.skipped,
                suite.summary.success_rate * 100.0,
                suite.total_duration.as_secs_f64()
            ));
        }

        markdown.push_str("\n");
        markdown
    }

    /// Generate detailed results section
    fn generate_detailed_results(&self, results: &[TestSuiteResult]) -> String {
        let mut markdown = String::new();

        markdown.push_str("## Detailed Test Results\n\n");

        for suite in results {
            markdown.push_str(&self.generate_suite_details(suite));
        }

        markdown
    }

    /// Generate details for a single test suite
    fn generate_suite_details(&self, suite: &TestSuiteResult) -> String {
        let mut markdown = String::new();

        let status_emoji = if suite.summary.success_rate >= 0.95 {
            "✅"
        } else if suite.summary.success_rate >= 0.80 {
            "⚠️"
        } else {
            "❌"
        };

        markdown.push_str(&format!("### {} {}\n\n", status_emoji, suite.suite_name));

        // Suite summary
        markdown.push_str(&format!(
            "**Duration:** {:.2}s | **Success Rate:** {:.1}% | **Tests:** {} passed, {} failed, {} skipped\n\n",
            suite.total_duration.as_secs_f64(),
            suite.summary.success_rate * 100.0,
            suite.summary.passed,
            suite.summary.failed,
            suite.summary.skipped
        ));

        // Failed tests (if any)
        let failed_tests: Vec<_> = suite
            .test_results
            .iter()
            .filter(|t| matches!(t.status, TestStatus::Failed | TestStatus::Timeout))
            .collect();

        if !failed_tests.is_empty() {
            markdown.push_str("#### ❌ Failed Tests\n\n");
            for test in failed_tests {
                markdown.push_str(&format!(
                    "- **{}** ({:.3}s)\n",
                    test.test_name,
                    test.duration.as_secs_f64()
                ));

                if let Some(error) = &test.error {
                    markdown.push_str(&format!(
                        "  ```\n  {}\n  ```\n",
                        error.to_string().replace('\n', "\n  ")
                    ));
                }
            }
            markdown.push_str("\n");
        }

        // Skipped tests (if any)
        let skipped_tests: Vec<_> = suite
            .test_results
            .iter()
            .filter(|t| matches!(t.status, TestStatus::Skipped))
            .collect();

        if !skipped_tests.is_empty() {
            markdown.push_str("#### ⚠️ Skipped Tests\n\n");
            for test in skipped_tests {
                markdown.push_str(&format!("- **{}**\n", test.test_name));
            }
            markdown.push_str("\n");
        }

        // All test details (collapsible)
        markdown.push_str("<details>\n");
        markdown.push_str("<summary>All Test Cases</summary>\n\n");
        markdown.push_str("| Test Name | Status | Duration | Memory Peak |\n");
        markdown.push_str("|-----------|--------|----------|-------------|\n");

        for test in &suite.test_results {
            let status_symbol = match test.status {
                TestStatus::Passed => "✅",
                TestStatus::Failed => "❌",
                TestStatus::Skipped => "⚠️",
                TestStatus::Timeout => "⏰",
                TestStatus::Running => "🔄",
            };

            let memory_info = test
                .metrics
                .memory_peak
                .map(|m| format!("{} KB", m / 1024))
                .unwrap_or_else(|| "N/A".to_string());

            markdown.push_str(&format!(
                "| `{}` | {} {:?} | {:.3}s | {} |\n",
                test.test_name,
                status_symbol,
                test.status,
                test.duration.as_secs_f64(),
                memory_info
            ));
        }

        markdown.push_str("\n</details>\n\n");
        markdown
    }

    /// Generate metrics section
    fn generate_metrics_section(&self, results: &[TestSuiteResult]) -> String {
        let mut markdown = String::new();

        markdown.push_str("## Performance Metrics\n\n");

        // Calculate aggregate metrics
        let total_tests: usize = results.iter().map(|r| r.summary.total_tests).sum();
        let total_duration: f64 = results.iter().map(|r| r.total_duration.as_secs_f64()).sum();

        let avg_test_duration = if total_tests > 0 {
            total_duration / total_tests as f64
        } else {
            0.0
        };

        // Memory usage statistics
        let memory_peaks: Vec<u64> = results
            .iter()
            .flat_map(|suite| &suite.test_results)
            .filter_map(|test| test.metrics.memory_peak)
            .collect();

        let (avg_memory, max_memory) = if !memory_peaks.is_empty() {
            let avg = memory_peaks.iter().sum::<u64>() as f64 / memory_peaks.len() as f64;
            let max = *memory_peaks.iter().max().unwrap();
            (avg, max)
        } else {
            (0.0, 0)
        };

        markdown.push_str("### Timing Metrics\n\n");
        markdown.push_str(&format!(
            "- **Total Execution Time:** {:.2}s\n",
            total_duration
        ));
        markdown.push_str(&format!(
            "- **Average Test Duration:** {:.3}s\n",
            avg_test_duration
        ));

        // Find slowest tests
        let mut all_tests: Vec<_> = results
            .iter()
            .flat_map(|suite| &suite.test_results)
            .collect();
        all_tests.sort_by(|a, b| b.duration.cmp(&a.duration));

        if !all_tests.is_empty() {
            markdown.push_str("- **Slowest Tests:**\n");
            for test in all_tests.iter().take(5) {
                markdown.push_str(&format!(
                    "  - `{}`: {:.3}s\n",
                    test.test_name,
                    test.duration.as_secs_f64()
                ));
            }
        }

        markdown.push_str("\n### Memory Metrics\n\n");
        if !memory_peaks.is_empty() {
            markdown.push_str(&format!(
                "- **Average Peak Memory:** {:.1} KB\n",
                avg_memory / 1024.0
            ));
            markdown.push_str(&format!(
                "- **Maximum Peak Memory:** {:.1} KB\n",
                max_memory as f64 / 1024.0
            ));
        } else {
            markdown.push_str("- No memory metrics available\n");
        }

        markdown.push_str("\n");
        markdown
    }

    /// Generate footer section
    fn generate_footer(&self) -> String {
        format!(
            r#"---

*Report generated by BitNet.rs Test Framework v{} on {}*
"#,
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
        crate::reporting::reporter::prepare_output_path(output_path).await?;

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
            total_duration: Duration::from_secs(12),
            test_results: vec![
                TestResult {
                    test_name: "test_markdown_pass".to_string(),
                    status: TestStatus::Passed,
                    duration: Duration::from_secs(4),
                    metrics: TestMetrics {
                        memory_peak: Some(4096),
                        memory_average: Some(2048),
                        cpu_time: Some(Duration::from_secs(3)),
                        wall_time: Duration::from_secs(4),
                        custom_metrics: HashMap::new(),
                        assertions: 8,
                        operations: 12,
                    },
                    error: None,
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(4),
                    end_time: std::time::SystemTime::now(),
                    metadata: HashMap::new(),
                },
                TestResult {
                    test_name: "test_markdown_fail".to_string(),
                    status: TestStatus::Failed,
                    duration: Duration::from_secs(6),
                    metrics: TestMetrics {
                        memory_peak: Some(8192),
                        memory_average: Some(4096),
                        cpu_time: Some(Duration::from_secs(5)),
                        wall_time: Duration::from_secs(6),
                        custom_metrics: HashMap::new(),
                        assertions: 4,
                        operations: 6,
                    },
                    error: Some("Markdown test failed".to_string()),
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(6),
                    end_time: std::time::SystemTime::now(),
                    metadata: HashMap::new(),
                },
                TestResult {
                    test_name: "test_markdown_skip".to_string(),
                    status: TestStatus::Skipped,
                    duration: Duration::from_secs(0),
                    metrics: TestMetrics {
                        memory_peak: None,
                        memory_average: None,
                        cpu_time: None,
                        wall_time: Duration::from_secs(0),
                        custom_metrics: HashMap::new(),
                        assertions: 0,
                        operations: 0,
                    },
                    error: None,
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
                total_duration: Duration::from_secs(12),
                average_duration: Duration::from_secs(4),
                peak_memory: Some(8192),
                total_assertions: 12,
            },
            environment: HashMap::new(),
            configuration: HashMap::new(),
            start_time: std::time::SystemTime::now() - Duration::from_secs(12),
            end_time: std::time::SystemTime::now(),
        }
    }

    #[tokio::test]
    async fn test_markdown_report_generation() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_report.md");

        let reporter = MarkdownReporter::new();
        let results = vec![create_test_suite_result()];

        let report_result = reporter
            .generate_report(&results, &output_path)
            .await
            .unwrap();

        assert_eq!(report_result.format, ReportFormat::Markdown);
        assert!(output_path.exists());
        assert!(report_result.size_bytes > 0);

        // Verify Markdown content structure
        let content = fs::read_to_string(&output_path).await.unwrap();
        assert!(content.contains("# BitNet.rs Test Report"));
        assert!(content.contains("## Executive Summary"));
        assert!(content.contains("## Test Suites Overview"));
        assert!(content.contains("## Detailed Test Results"));
        assert!(content.contains("## Performance Metrics"));
        assert!(content.contains("markdown_test_suite"));
        assert!(content.contains("test_markdown_pass"));
        assert!(content.contains("test_markdown_fail"));
        assert!(content.contains("test_markdown_skip"));
        assert!(content.contains("✅")); // Success emoji
        assert!(content.contains("❌")); // Failure emoji
        assert!(content.contains("⚠️")); // Warning emoji
    }

    #[tokio::test]
    async fn test_summary_only_markdown() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("summary_report.md");

        let reporter = MarkdownReporter::new_summary_only();
        let results = vec![create_test_suite_result()];

        reporter
            .generate_report(&results, &output_path)
            .await
            .unwrap();

        let content = fs::read_to_string(&output_path).await.unwrap();
        // Summary only should not include detailed results or metrics
        assert!(!content.contains("## Detailed Test Results"));
        assert!(!content.contains("## Performance Metrics"));
        assert!(content.contains("## Executive Summary"));
        assert!(content.contains("## Test Suites Overview"));
    }
}
