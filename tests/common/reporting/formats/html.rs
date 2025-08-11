//! HTML report format implementation with interactive features

use crate::reporting::{ReportError, ReportFormat, ReportResult, TestReporter};
use crate::results::{TestResult, TestStatus, TestSuiteResult};
use async_trait::async_trait;
use std::path::Path;
use std::time::Instant;
use tokio::fs;

/// HTML format reporter with interactive features
pub struct HtmlReporter {
    interactive: bool,
    include_charts: bool,
}

impl HtmlReporter {
    /// Create a new HTML reporter with interactive features
    pub fn new(interactive: bool) -> Self {
        Self {
            interactive,
            include_charts: true,
        }
    }

    /// Create a new HTML reporter with minimal features
    pub fn new_minimal() -> Self {
        Self {
            interactive: false,
            include_charts: false,
        }
    }

    /// Generate the complete HTML report
    fn generate_html_content(&self, results: &[TestSuiteResult]) -> Result<String, ReportError> {
        let mut html = String::new();

        // HTML document structure
        html.push_str("<!DOCTYPE html>\n");
        html.push_str("<html lang=\"en\">\n");
        html.push_str("<head>\n");
        html.push_str("    <meta charset=\"UTF-8\">\n");
        html.push_str(
            "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n",
        );
        html.push_str("    <title>BitNet.rs Test Report</title>\n");
        html.push_str(&self.generate_css());
        if self.interactive {
            html.push_str(&self.generate_javascript());
        }
        html.push_str("</head>\n");
        html.push_str("<body>\n");

        // Header
        html.push_str(&self.generate_header(results));

        // Summary section
        html.push_str(&self.generate_summary(results));

        // Charts section (if enabled)
        if self.include_charts {
            html.push_str(&self.generate_charts(results));
        }

        // Test suites section
        html.push_str(&self.generate_test_suites(results));

        // Footer
        html.push_str(&self.generate_footer());

        html.push_str("</body>\n");
        html.push_str("</html>\n");

        Ok(html)
    }

    /// Generate CSS styles
    fn generate_css(&self) -> String {
        r#"
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .header .subtitle {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .summary-card {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .summary-card h3 {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        
        .summary-card.passed h3 { color: #28a745; }
        .summary-card.failed h3 { color: #dc3545; }
        .summary-card.skipped h3 { color: #ffc107; }
        .summary-card.total h3 { color: #6c757d; }
        
        .test-suite {
            background: white;
            margin-bottom: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .suite-header {
            background: #f8f9fa;
            padding: 1rem 1.5rem;
            border-bottom: 1px solid #dee2e6;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .suite-header:hover {
            background: #e9ecef;
        }
        
        .suite-title {
            font-size: 1.2rem;
            font-weight: 600;
        }
        
        .suite-stats {
            display: flex;
            gap: 1rem;
            font-size: 0.9rem;
        }
        
        .stat {
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-weight: 500;
        }
        
        .stat.passed { background: #d4edda; color: #155724; }
        .stat.failed { background: #f8d7da; color: #721c24; }
        .stat.skipped { background: #fff3cd; color: #856404; }
        
        .test-cases {
            display: none;
        }
        
        .test-cases.expanded {
            display: block;
        }
        
        .test-case {
            padding: 1rem 1.5rem;
            border-bottom: 1px solid #f1f3f4;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .test-case:last-child {
            border-bottom: none;
        }
        
        .test-name {
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.9rem;
        }
        
        .test-status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .status-badge {
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 500;
            text-transform: uppercase;
        }
        
        .status-badge.passed {
            background: #d4edda;
            color: #155724;
        }
        
        .status-badge.failed {
            background: #f8d7da;
            color: #721c24;
        }
        
        .status-badge.skipped {
            background: #fff3cd;
            color: #856404;
        }
        
        .duration {
            font-size: 0.8rem;
            color: #6c757d;
        }
        
        .error-details {
            background: #f8f9fa;
            padding: 1rem;
            margin-top: 0.5rem;
            border-left: 4px solid #dc3545;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.8rem;
            white-space: pre-wrap;
            display: none;
        }
        
        .error-details.expanded {
            display: block;
        }
        
        .charts {
            background: white;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        
        .chart-container {
            height: 300px;
            margin: 1rem 0;
        }
        
        .footer {
            text-align: center;
            padding: 2rem;
            color: #6c757d;
            font-size: 0.9rem;
        }
        
        .toggle-icon {
            transition: transform 0.2s;
        }
        
        .toggle-icon.expanded {
            transform: rotate(90deg);
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .summary {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .suite-stats {
                flex-direction: column;
                gap: 0.5rem;
            }
        }
    </style>
"#
        .to_string()
    }

    /// Generate JavaScript for interactive features
    fn generate_javascript(&self) -> String {
        r#"
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Toggle test suite expansion
            document.querySelectorAll('.suite-header').forEach(header => {
                header.addEventListener('click', function() {
                    const testCases = this.nextElementSibling;
                    const icon = this.querySelector('.toggle-icon');
                    
                    testCases.classList.toggle('expanded');
                    icon.classList.toggle('expanded');
                });
            });
            
            // Toggle error details
            document.querySelectorAll('.test-case').forEach(testCase => {
                testCase.addEventListener('click', function() {
                    const errorDetails = this.querySelector('.error-details');
                    if (errorDetails) {
                        errorDetails.classList.toggle('expanded');
                    }
                });
            });
            
            // Filter functionality
            const filterButtons = document.querySelectorAll('.filter-btn');
            filterButtons.forEach(btn => {
                btn.addEventListener('click', function() {
                    const filter = this.dataset.filter;
                    
                    // Update active button
                    filterButtons.forEach(b => b.classList.remove('active'));
                    this.classList.add('active');
                    
                    // Filter test cases
                    document.querySelectorAll('.test-case').forEach(testCase => {
                        const status = testCase.querySelector('.status-badge').textContent.toLowerCase();
                        
                        if (filter === 'all' || status === filter) {
                            testCase.style.display = 'flex';
                        } else {
                            testCase.style.display = 'none';
                        }
                    });
                });
            });
        });
    </script>
"#.to_string()
    }

    /// Generate header section
    fn generate_header(&self, results: &[TestSuiteResult]) -> String {
        let total_duration: f64 = results.iter().map(|r| r.total_duration.as_secs_f64()).sum();

        format!(
            r#"
    <div class="container">
        <div class="header">
            <h1>BitNet.rs Test Report</h1>
            <div class="subtitle">
                Generated on {} | {} test suites | {:.2}s total duration
            </div>
        </div>
"#,
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
            results.len(),
            total_duration
        )
    }

    /// Generate summary section
    fn generate_summary(&self, results: &[TestSuiteResult]) -> String {
        let total_tests: usize = results.iter().map(|r| r.summary.total_tests).sum();
        let total_passed: usize = results.iter().map(|r| r.summary.passed).sum();
        let total_failed: usize = results.iter().map(|r| r.summary.failed).sum();
        let total_skipped: usize = results.iter().map(|r| r.summary.skipped).sum();

        let success_rate = if total_tests > 0 {
            (total_passed as f64 / total_tests as f64) * 100.0
        } else {
            0.0
        };

        format!(
            r#"
        <div class="summary">
            <div class="summary-card total">
                <h3>{}</h3>
                <p>Total Tests</p>
            </div>
            <div class="summary-card passed">
                <h3>{}</h3>
                <p>Passed</p>
            </div>
            <div class="summary-card failed">
                <h3>{}</h3>
                <p>Failed</p>
            </div>
            <div class="summary-card skipped">
                <h3>{}</h3>
                <p>Skipped</p>
            </div>
            <div class="summary-card">
                <h3>{:.1}%</h3>
                <p>Success Rate</p>
            </div>
        </div>
"#,
            total_tests, total_passed, total_failed, total_skipped, success_rate
        )
    }

    /// Generate charts section
    fn generate_charts(&self, _results: &[TestSuiteResult]) -> String {
        // For now, return a placeholder for charts
        // In a full implementation, this would include Chart.js or similar
        r#"
        <div class="charts">
            <h2>Test Results Overview</h2>
            <div class="chart-container">
                <p style="text-align: center; color: #6c757d; margin-top: 100px;">
                    Charts would be rendered here with Chart.js or similar library
                </p>
            </div>
        </div>
"#
        .to_string()
    }

    /// Generate test suites section
    fn generate_test_suites(&self, results: &[TestSuiteResult]) -> String {
        let mut html = String::new();

        html.push_str("        <div class=\"test-suites\">\n");
        html.push_str("            <h2>Test Suites</h2>\n");

        if self.interactive {
            html.push_str(&self.generate_filter_buttons());
        }

        for suite in results {
            html.push_str(&self.generate_test_suite(suite));
        }

        html.push_str("        </div>\n");
        html
    }

    /// Generate filter buttons for interactive mode
    fn generate_filter_buttons(&self) -> String {
        r#"
            <div style="margin: 1rem 0; text-align: center;">
                <button class="filter-btn active" data-filter="all" style="margin: 0 0.5rem; padding: 0.5rem 1rem; border: none; border-radius: 4px; background: #007bff; color: white; cursor: pointer;">All</button>
                <button class="filter-btn" data-filter="passed" style="margin: 0 0.5rem; padding: 0.5rem 1rem; border: none; border-radius: 4px; background: #28a745; color: white; cursor: pointer;">Passed</button>
                <button class="filter-btn" data-filter="failed" style="margin: 0 0.5rem; padding: 0.5rem 1rem; border: none; border-radius: 4px; background: #dc3545; color: white; cursor: pointer;">Failed</button>
                <button class="filter-btn" data-filter="skipped" style="margin: 0 0.5rem; padding: 0.5rem 1rem; border: none; border-radius: 4px; background: #ffc107; color: white; cursor: pointer;">Skipped</button>
            </div>
"#.to_string()
    }

    /// Generate a single test suite
    fn generate_test_suite(&self, suite: &TestSuiteResult) -> String {
        let mut html = String::new();

        html.push_str("            <div class=\"test-suite\">\n");
        html.push_str(&format!(
            r#"                <div class="suite-header">
                    <div class="suite-title">
                        <span class="toggle-icon">▶</span> {}
                    </div>
                    <div class="suite-stats">
                        <span class="stat passed">{} passed</span>
                        <span class="stat failed">{} failed</span>
                        <span class="stat skipped">{} skipped</span>
                        <span class="duration">{:.2}s</span>
                    </div>
                </div>
"#,
            suite.suite_name,
            suite.summary.passed,
            suite.summary.failed,
            suite.summary.skipped,
            suite.total_duration.as_secs_f64()
        ));

        html.push_str("                <div class=\"test-cases\">\n");
        for test in &suite.test_results {
            html.push_str(&self.generate_test_case(test));
        }
        html.push_str("                </div>\n");
        html.push_str("            </div>\n");

        html
    }

    /// Generate a single test case
    fn generate_test_case(&self, test: &TestResult) -> String {
        let status_class = match test.status {
            TestStatus::Passed => "passed",
            TestStatus::Failed => "failed",
            TestStatus::Skipped => "skipped",
            TestStatus::Timeout => "failed",
            TestStatus::Running => "running",
        };

        let mut html = format!(
            r#"                    <div class="test-case">
                        <div class="test-name">{}</div>
                        <div class="test-status">
                            <span class="status-badge {}">{:?}</span>
                            <span class="duration">{:.3}s</span>
                        </div>
"#,
            test.test_name,
            status_class,
            test.status,
            test.duration.as_secs_f64()
        );

        if let Some(error) = &test.error {
            html.push_str(&format!(
                r#"                        <div class="error-details">{}</div>
"#,
                html_escape::encode_text(&format!("{:#?}", error))
            ));
        }

        html.push_str("                    </div>\n");
        html
    }

    /// Generate footer section
    fn generate_footer(&self) -> String {
        format!(
            r#"        <div class="footer">
            Generated by BitNet.rs Test Framework v{} | {}
        </div>
    </div>
"#,
            env!("CARGO_PKG_VERSION"),
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        )
    }
}

#[async_trait]
impl TestReporter for HtmlReporter {
    async fn generate_report(
        &self,
        results: &[TestSuiteResult],
        output_path: &Path,
    ) -> Result<ReportResult, ReportError> {
        let start_time = Instant::now();
        crate::reporting::reporter::prepare_output_path(output_path).await?;

        let html_content = self.generate_html_content(results)?;
        fs::write(output_path, &html_content).await?;

        let generation_time = start_time.elapsed();
        let size_bytes = html_content.len() as u64;

        Ok(ReportResult {
            format: ReportFormat::Html,
            output_path: output_path.to_path_buf(),
            size_bytes,
            generation_time,
        })
    }

    fn format(&self) -> ReportFormat {
        ReportFormat::Html
    }

    fn file_extension(&self) -> &'static str {
        "html"
    }
}

impl Default for HtmlReporter {
    fn default() -> Self {
        Self::new(true)
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
            suite_name: "html_test_suite".to_string(),
            total_duration: Duration::from_secs(8),
            test_results: vec![
                TestResult {
                    test_name: "test_html_pass".to_string(),
                    status: TestStatus::Passed,
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
                    error: None,
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(3),
                    end_time: std::time::SystemTime::now(),
                    metadata: HashMap::new(),
                },
                TestResult {
                    test_name: "test_html_fail".to_string(),
                    status: TestStatus::Failed,
                    duration: Duration::from_secs(5),
                    metrics: TestMetrics {
                        memory_peak: Some(2048),
                        memory_average: Some(1024),
                        cpu_time: Some(Duration::from_secs(4)),
                        wall_time: Duration::from_secs(5),
                        custom_metrics: HashMap::new(),
                        assertions: 2,
                        operations: 3,
                    },
                    error: Some("HTML test assertion failed".to_string()),
                    stack_trace: None,
                    artifacts: Vec::new(),
                    start_time: std::time::SystemTime::now() - Duration::from_secs(5),
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
                total_duration: Duration::from_secs(8),
                average_duration: Duration::from_secs(4),
                peak_memory: Some(2048),
                total_assertions: 5,
            },
            environment: HashMap::new(),
            configuration: HashMap::new(),
            start_time: std::time::SystemTime::now() - Duration::from_secs(8),
            end_time: std::time::SystemTime::now(),
        }
    }

    #[tokio::test]
    async fn test_html_report_generation() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test_report.html");

        let reporter = HtmlReporter::new(true);
        let results = vec![create_test_suite_result()];

        let report_result = reporter
            .generate_report(&results, &output_path)
            .await
            .unwrap();

        assert_eq!(report_result.format, ReportFormat::Html);
        assert!(output_path.exists());
        assert!(report_result.size_bytes > 0);

        // Verify HTML content structure
        let content = fs::read_to_string(&output_path).await.unwrap();
        assert!(content.contains("<!DOCTYPE html>"));
        assert!(content.contains("BitNet.rs Test Report"));
        assert!(content.contains("html_test_suite"));
        assert!(content.contains("test_html_pass"));
        assert!(content.contains("test_html_fail"));
        assert!(content.contains("status-badge passed"));
        assert!(content.contains("status-badge failed"));
    }

    #[tokio::test]
    async fn test_minimal_html_output() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("minimal_report.html");

        let reporter = HtmlReporter::new_minimal();
        let results = vec![create_test_suite_result()];

        reporter
            .generate_report(&results, &output_path)
            .await
            .unwrap();

        let content = fs::read_to_string(&output_path).await.unwrap();
        // Minimal output should not include JavaScript
        assert!(!content.contains("addEventListener"));
        // Should not include charts
        assert!(!content.contains("Charts would be rendered"));
    }
}
