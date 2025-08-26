//! Core test reporter trait and management

use super::{ReportConfig, ReportError, ReportFormat, ReportResult};
use crate::results::TestSuiteResult;
use async_trait::async_trait;
use std::path::Path;
use tokio::fs;

/// Core trait for test reporters
#[async_trait]
pub trait TestReporter: Send + Sync {
    /// Generate a report for the given test results
    async fn generate_report(
        &self,
        results: &[TestSuiteResult],
        output_path: &Path,
    ) -> Result<ReportResult, ReportError>;

    /// Get the format this reporter handles
    fn format(&self) -> ReportFormat;

    /// Get the file extension for this format
    fn file_extension(&self) -> &'static str;
}

/// Default implementation for preparing output path
pub async fn prepare_output_path(output_path: &Path) -> Result<(), ReportError> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).await?;
    }
    Ok(())
}

/// Enum wrapper for different reporter types to make them object-safe
pub enum ReporterType {
    Html(super::formats::HtmlReporter),
    Json(super::formats::JsonReporter),
    Junit(super::formats::JunitReporter),
    Markdown(super::formats::MarkdownReporter),
}

#[async_trait]
impl TestReporter for ReporterType {
    async fn generate_report(
        &self,
        results: &[TestSuiteResult],
        output_path: &Path,
    ) -> Result<ReportResult, ReportError> {
        match self {
            ReporterType::Html(reporter) => reporter.generate_report(results, output_path).await,
            ReporterType::Json(reporter) => reporter.generate_report(results, output_path).await,
            ReporterType::Junit(reporter) => reporter.generate_report(results, output_path).await,
            ReporterType::Markdown(reporter) => {
                reporter.generate_report(results, output_path).await
            }
        }
    }

    fn format(&self) -> ReportFormat {
        match self {
            ReporterType::Html(reporter) => reporter.format(),
            ReporterType::Json(reporter) => reporter.format(),
            ReporterType::Junit(reporter) => reporter.format(),
            ReporterType::Markdown(reporter) => reporter.format(),
        }
    }

    fn file_extension(&self) -> &'static str {
        match self {
            ReporterType::Html(reporter) => reporter.file_extension(),
            ReporterType::Json(reporter) => reporter.file_extension(),
            ReporterType::Junit(reporter) => reporter.file_extension(),
            ReporterType::Markdown(reporter) => reporter.file_extension(),
        }
    }
}

/// Manages multiple test reporters and coordinates report generation
pub struct ReportingManager {
    config: ReportConfig,
    reporters: Vec<ReporterType>,
}

impl ReportingManager {
    /// Create a new reporting manager with the given configuration
    pub fn new(config: ReportConfig) -> Self {
        let mut reporters: Vec<ReporterType> = Vec::new();

        // Register reporters based on configuration
        for format in &config.formats {
            match format {
                ReportFormat::Html => {
                    reporters.push(ReporterType::Html(super::formats::HtmlReporter::new(
                        config.interactive_html,
                    )));
                }
                ReportFormat::Json => {
                    reporters.push(ReporterType::Json(super::formats::JsonReporter::new()));
                }
                ReportFormat::Junit => {
                    reporters.push(ReporterType::Junit(super::formats::JunitReporter::new()));
                }
                ReportFormat::Markdown => {
                    reporters.push(ReporterType::Markdown(super::formats::MarkdownReporter::new()));
                }
            }
        }

        Self { config, reporters }
    }

    /// Generate all configured reports
    pub async fn generate_all_reports(
        &self,
        results: &[TestSuiteResult],
    ) -> Result<Vec<ReportResult>, ReportError> {
        // Ensure output directory exists
        fs::create_dir_all(&self.config.output_dir).await?;

        let mut report_results = Vec::new();

        for reporter in &self.reporters {
            let filename = format!("test_report.{}", reporter.file_extension());
            let output_path = self.config.output_dir.join(filename);

            let result = reporter.generate_report(results, &output_path).await?;
            report_results.push(result);
        }

        // Generate summary report
        self.generate_summary_report(&report_results).await?;

        Ok(report_results)
    }

    /// Generate a summary of all generated reports
    async fn generate_summary_report(
        &self,
        report_results: &[ReportResult],
    ) -> Result<(), ReportError> {
        let summary_path = self.config.output_dir.join("report_summary.md");

        let mut content = String::new();
        content.push_str("# Test Report Summary\n\n");
        content.push_str(&format!(
            "Generated on: {}\n\n",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ));

        content.push_str("## Generated Reports\n\n");
        for result in report_results {
            content.push_str(&format!(
                "- **{}**: `{}` ({} bytes, generated in {:?})\n",
                result.format,
                result.output_path.file_name().unwrap().to_string_lossy(),
                result.size_bytes,
                result.generation_time
            ));
        }

        fs::write(summary_path, content).await?;
        Ok(())
    }

    /// Get the configured output directory
    pub fn output_dir(&self) -> &Path {
        &self.config.output_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::results::{TestMetrics, TestResult, TestStatus, TestSummary};
    use std::collections::HashMap;
    use std::time::Duration;
    use tempfile::TempDir;

    fn create_test_results() -> Vec<TestSuiteResult> {
        vec![TestSuiteResult {
            suite_name: "test_suite".to_string(),
            total_duration: Duration::from_secs(10),
            test_results: vec![TestResult {
                test_name: "test_1".to_string(),
                status: TestStatus::Passed,
                duration: Duration::from_secs(5),
                metrics: TestMetrics {
                    memory_peak: Some(1024),
                    memory_average: Some(512),
                    cpu_time: Some(Duration::from_secs(4)),
                    wall_time: Duration::from_secs(5),
                    custom_metrics: HashMap::new(),
                    assertions: 3,
                    operations: 5,
                },
                error: None,
                stack_trace: None,
                artifacts: Vec::new(),
                start_time: std::time::SystemTime::now() - Duration::from_secs(5),
                end_time: std::time::SystemTime::now(),
                metadata: HashMap::new(),
            }],
            summary: TestSummary {
                total_tests: 1,
                passed: 1,
                failed: 0,
                skipped: 0,
                timeout: 0,
                success_rate: 100.0,
                total_duration: Duration::from_secs(10),
                average_duration: Duration::from_secs(5),
                peak_memory: Some(1024),
                total_assertions: 3,
            },
            environment: HashMap::new(),
            configuration: HashMap::new(),
            start_time: std::time::SystemTime::now() - Duration::from_secs(10),
            end_time: std::time::SystemTime::now(),
        }]
    }

    #[tokio::test]
    async fn test_reporting_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = ReportConfig {
            output_dir: temp_dir.path().to_path_buf(),
            formats: vec![ReportFormat::Json],
            include_artifacts: false,
            generate_coverage: false,
            interactive_html: false,
        };

        let manager = ReportingManager::new(config);
        assert_eq!(manager.reporters.len(), 1);
    }

    #[tokio::test]
    async fn test_generate_summary_report() {
        let temp_dir = TempDir::new().unwrap();
        let config = ReportConfig {
            output_dir: temp_dir.path().to_path_buf(),
            formats: vec![],
            include_artifacts: false,
            generate_coverage: false,
            interactive_html: false,
        };

        let manager = ReportingManager::new(config);
        let report_results = vec![ReportResult {
            format: ReportFormat::Json,
            output_path: temp_dir.path().join("test.json"),
            size_bytes: 1024,
            generation_time: Duration::from_millis(100),
        }];

        manager.generate_summary_report(&report_results).await.unwrap();

        let summary_path = temp_dir.path().join("report_summary.md");
        assert!(summary_path.exists());

        let content = fs::read_to_string(summary_path).await.unwrap();
        assert!(content.contains("Test Report Summary"));
        assert!(content.contains("JSON"));
    }
}
