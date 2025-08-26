//! JUnit XML report format implementation for CI integration

use super::super::{ReportError, ReportFormat, ReportResult, TestReporter};
use crate::results::{TestResult, TestStatus, TestSuiteResult};
use async_trait::async_trait;
use std::path::Path;
use std::time::Instant;
use tokio::fs;
use xml::writer::{EmitterConfig, XmlEvent};

/// JUnit XML format reporter for CI integration
pub struct JunitReporter {
    include_system_out: bool,
    include_system_err: bool,
}

impl JunitReporter {
    /// Create a new JUnit reporter
    pub fn new() -> Self {
        Self { include_system_out: true, include_system_err: true }
    }

    /// Create a new JUnit reporter with minimal output
    pub fn new_minimal() -> Self {
        Self { include_system_out: false, include_system_err: false }
    }

    /// Generate XML content for test results
    fn generate_xml_content(&self, results: &[TestSuiteResult]) -> Result<String, ReportError> {
        let mut output = Vec::new();
        let mut writer = EmitterConfig::new().perform_indent(true).create_writer(&mut output);

        // Write XML declaration
        writer
            .write(XmlEvent::StartDocument {
                version: xml::common::XmlVersion::Version10,
                encoding: Some("UTF-8"),
                standalone: None,
            })
            .map_err(|e| ReportError::XmlError(e.to_string()))?;

        // Calculate totals
        let total_tests: usize = results.iter().map(|r| r.summary.total_tests).sum();
        let total_failures: usize = results.iter().map(|r| r.summary.failed).sum();
        let total_errors = 0; // We don't distinguish between errors and failures
        let total_skipped: usize = results.iter().map(|r| r.summary.skipped).sum();
        let total_time: f64 = results.iter().map(|r| r.total_duration.as_secs_f64()).sum();

        // Start testsuites element
        writer
            .write(
                XmlEvent::start_element("testsuites")
                    .attr("name", "BitNet.rs Test Suite")
                    .attr("tests", &total_tests.to_string())
                    .attr("failures", &total_failures.to_string())
                    .attr("errors", &total_errors.to_string())
                    .attr("skipped", &total_skipped.to_string())
                    .attr("time", &format!("{:.3}", total_time))
                    .attr("timestamp", &chrono::Utc::now().to_rfc3339()),
            )
            .map_err(|e| ReportError::XmlError(e.to_string()))?;

        // Write each test suite
        for suite in results {
            self.write_test_suite(&mut writer, suite)?;
        }

        // End testsuites element
        writer.write(XmlEvent::end_element()).map_err(|e| ReportError::XmlError(e.to_string()))?;

        String::from_utf8(output).map_err(|e| ReportError::XmlError(e.to_string()))
    }

    /// Write a single test suite to XML
    fn write_test_suite(
        &self,
        writer: &mut xml::writer::EventWriter<&mut Vec<u8>>,
        suite: &TestSuiteResult,
    ) -> Result<(), ReportError> {
        // Start testsuite element
        writer
            .write(
                XmlEvent::start_element("testsuite")
                    .attr("name", &suite.suite_name)
                    .attr("tests", &suite.summary.total_tests.to_string())
                    .attr("failures", &suite.summary.failed.to_string())
                    .attr("errors", "0")
                    .attr("skipped", &suite.summary.skipped.to_string())
                    .attr("time", &format!("{:.3}", suite.total_duration.as_secs_f64()))
                    .attr("timestamp", &chrono::Utc::now().to_rfc3339()),
            )
            .map_err(|e| ReportError::XmlError(e.to_string()))?;

        // Write properties if needed
        writer
            .write(XmlEvent::start_element("properties"))
            .map_err(|e| ReportError::XmlError(e.to_string()))?;

        writer
            .write(
                XmlEvent::start_element("property")
                    .attr("name", "success_rate")
                    .attr("value", &format!("{:.2}", suite.summary.success_rate)),
            )
            .map_err(|e| ReportError::XmlError(e.to_string()))?;
        writer.write(XmlEvent::end_element()).map_err(|e| ReportError::XmlError(e.to_string()))?;

        writer
            .write(XmlEvent::end_element()) // properties
            .map_err(|e| ReportError::XmlError(e.to_string()))?;

        // Write each test case
        for test in &suite.test_results {
            self.write_test_case(writer, test)?;
        }

        // Add system-out and system-err if enabled
        if self.include_system_out {
            writer
                .write(XmlEvent::start_element("system-out"))
                .map_err(|e| ReportError::XmlError(e.to_string()))?;
            writer
                .write(XmlEvent::cdata(&format!(
                    "Test suite: {}\nTotal duration: {:?}\nSuccess rate: {:.2}%",
                    suite.suite_name,
                    suite.total_duration,
                    suite.summary.success_rate * 100.0
                )))
                .map_err(|e| ReportError::XmlError(e.to_string()))?;
            writer
                .write(XmlEvent::end_element())
                .map_err(|e| ReportError::XmlError(e.to_string()))?;
        }

        if self.include_system_err {
            writer
                .write(XmlEvent::start_element("system-err"))
                .map_err(|e| ReportError::XmlError(e.to_string()))?;
            writer.write(XmlEvent::cdata("")).map_err(|e| ReportError::XmlError(e.to_string()))?;
            writer
                .write(XmlEvent::end_element())
                .map_err(|e| ReportError::XmlError(e.to_string()))?;
        }

        // End testsuite element
        writer.write(XmlEvent::end_element()).map_err(|e| ReportError::XmlError(e.to_string()))?;

        Ok(())
    }

    /// Write a single test case to XML
    fn write_test_case(
        &self,
        writer: &mut xml::writer::EventWriter<&mut Vec<u8>>,
        test: &TestResult,
    ) -> Result<(), ReportError> {
        let classname = test.test_name.split("::").next().unwrap_or("unknown");
        let name = test.test_name.split("::").last().unwrap_or(&test.test_name);

        // Start testcase element
        writer
            .write(
                XmlEvent::start_element("testcase")
                    .attr("classname", classname)
                    .attr("name", name)
                    .attr("time", &format!("{:.3}", test.duration.as_secs_f64())),
            )
            .map_err(|e| ReportError::XmlError(e.to_string()))?;

        // Handle different test statuses
        match test.status {
            TestStatus::Failed => {
                writer
                    .write(
                        XmlEvent::start_element("failure")
                            .attr(
                                "message",
                                &test
                                    .error
                                    .as_ref()
                                    .map(|e| e.to_string())
                                    .unwrap_or_else(|| "Test failed".to_string()),
                            )
                            .attr("type", "AssertionError"),
                    )
                    .map_err(|e| ReportError::XmlError(e.to_string()))?;

                let mut details = String::new();
                if let Some(error) = &test.error {
                    details.push_str(&format!("{:#?}\n", error));
                }
                if let Some(trace) = &test.stack_trace {
                    details.push_str(trace);
                }
                if !details.is_empty() {
                    writer
                        .write(XmlEvent::cdata(&details))
                        .map_err(|e| ReportError::XmlError(e.to_string()))?;
                }

                writer
                    .write(XmlEvent::end_element()) // failure
                    .map_err(|e| ReportError::XmlError(e.to_string()))?;
            }
            TestStatus::Skipped => {
                writer
                    .write(XmlEvent::start_element("skipped").attr("message", "Test was skipped"))
                    .map_err(|e| ReportError::XmlError(e.to_string()))?;
                writer
                    .write(XmlEvent::end_element()) // skipped
                    .map_err(|e| ReportError::XmlError(e.to_string()))?;
            }
            TestStatus::Timeout => {
                writer
                    .write(
                        XmlEvent::start_element("error")
                            .attr("message", "Test timed out")
                            .attr("type", "TimeoutError"),
                    )
                    .map_err(|e| ReportError::XmlError(e.to_string()))?;
                writer
                    .write(XmlEvent::end_element()) // error
                    .map_err(|e| ReportError::XmlError(e.to_string()))?;
            }
            TestStatus::Passed => {
                // No additional elements needed for passed tests
            }
            TestStatus::Running => {
                writer
                    .write(
                        XmlEvent::start_element("error")
                            .attr("message", "Test is still running")
                            .attr("type", "RunningError"),
                    )
                    .map_err(|e| ReportError::XmlError(e.to_string()))?;
                writer
                    .write(XmlEvent::end_element()) // error
                    .map_err(|e| ReportError::XmlError(e.to_string()))?;
            }
        }

        // End testcase element
        writer.write(XmlEvent::end_element()).map_err(|e| ReportError::XmlError(e.to_string()))?;

        Ok(())
    }
}

#[async_trait]
impl TestReporter for JunitReporter {
    async fn generate_report(
        &self,
        results: &[TestSuiteResult],
        output_path: &Path,
    ) -> Result<ReportResult, ReportError> {
        let start_time = Instant::now();
        super::super::reporter::prepare_output_path(output_path).await?;

        let xml_content = self.generate_xml_content(results)?;
        fs::write(output_path, &xml_content).await?;

        let generation_time = start_time.elapsed();
        let size_bytes = xml_content.len() as u64;

        Ok(ReportResult {
            format: ReportFormat::Junit,
            output_path: output_path.to_path_buf(),
            size_bytes,
            generation_time,
        })
    }

    fn format(&self) -> ReportFormat {
        ReportFormat::Junit
    }

    fn file_extension(&self) -> &'static str {
        "xml"
    }
}

impl Default for JunitReporter {
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
            suite_name: "junit_test_suite".to_string(),
            total_duration: Duration::from_secs(10),
            test_results: vec![
                TestResult {
                    test_name: "module::test_pass".to_string(),
                    status: TestStatus::Passed,
                    duration: Duration::from_secs(3),
                    metrics: TestMetrics {
                        memory_peak: Some(1024),
                        memory_average: Some(512),
                        cpu_time: Some(Duration::from_secs(2)),
                        wall_time: Duration::from_secs(3),
                        custom_metrics: HashMap::new(),
                        assertions: 5,
                        operations: 8,
                    },
                    error: None,
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(3),
                    end_time: std::time::SystemTime::now(),
                    metadata: HashMap::new(),
                },
                TestResult {
                    test_name: "module::test_fail".to_string(),
                    status: TestStatus::Failed,
                    duration: Duration::from_secs(5),
                    metrics: TestMetrics {
                        memory_peak: Some(2048),
                        memory_average: Some(1024),
                        cpu_time: Some(Duration::from_secs(4)),
                        wall_time: Duration::from_secs(5),
                        custom_metrics: HashMap::new(),
                        assertions: 3,
                        operations: 4,
                    },
                    error: Some("Assertion failed".to_string()),
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(5),
                    end_time: std::time::SystemTime::now(),
                    metadata: HashMap::new(),
                },
                TestResult {
                    test_name: "module::test_skip".to_string(),
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
                total_duration: Duration::from_secs(10),
                average_duration: Duration::from_millis(3333),
                peak_memory: Some(2048),
                total_assertions: 8,
            },
            environment: HashMap::new(),
            configuration: HashMap::new(),
            start_time: std::time::SystemTime::now() - Duration::from_secs(10),
            end_time: std::time::SystemTime::now(),
        }
    }

    #[tokio::test]
    async fn test_junit_report_generation() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("junit_report.xml");

        let reporter = JunitReporter::new();
        let results = vec![create_test_suite_result()];

        let report_result = reporter.generate_report(&results, &output_path).await.unwrap();

        assert_eq!(report_result.format, ReportFormat::Junit);
        assert!(output_path.exists());
        assert!(report_result.size_bytes > 0);

        // Verify XML content structure
        let content = fs::read_to_string(&output_path).await.unwrap();
        assert!(content.contains("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"));
        assert!(content.contains("<testsuites"));
        assert!(content.contains("<testsuite"));
        assert!(content.contains("<testcase"));
        assert!(content.contains("classname=\"module\""));
        assert!(content.contains("name=\"test_pass\""));
        assert!(content.contains("<failure"));
        assert!(content.contains("<skipped"));
    }

    #[tokio::test]
    async fn test_minimal_junit_output() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("minimal_junit.xml");

        let reporter = JunitReporter::new_minimal();
        let results = vec![create_test_suite_result()];

        reporter.generate_report(&results, &output_path).await.unwrap();

        let content = fs::read_to_string(&output_path).await.unwrap();
        // Minimal output should not include system-out/system-err with content
        assert!(!content.contains("Test suite:"));
    }
}
