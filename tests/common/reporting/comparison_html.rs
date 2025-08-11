//! HTML report generation for comparison analysis
//!
//! This module provides HTML report generation capabilities for comparison
//! analysis results, including interactive charts and detailed visualizations.

use super::comparison_analysis::{
    AccuracyAnalysisReport, AccuracySeverity, ExecutiveSummaryReport, PerformanceAnalysisReport,
    PerformanceCategory, RegressionAnalysisReport, StatusLevel, TestCaseAccuracy,
    TestCasePerformance,
};
use std::path::{Path, PathBuf};

/// HTML report generator for comparison analysis
pub struct ComparisonHtmlReporter {
    output_dir: PathBuf,
}

impl ComparisonHtmlReporter {
    /// Create a new HTML reporter
    pub fn new(output_dir: PathBuf) -> Self {
        Self { output_dir }
    }

    /// Generate comprehensive HTML report
    pub async fn generate_html_report(
        &self,
        accuracy_report: &AccuracyAnalysisReport,
        performance_report: &PerformanceAnalysisReport,
        regression_report: &RegressionAnalysisReport,
        executive_report: &ExecutiveSummaryReport,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        // Create output directory
        std::fs::create_dir_all(&self.output_dir)?;

        // Generate main HTML report
        let html_content = self.generate_main_html(
            accuracy_report,
            performance_report,
            regression_report,
            executive_report,
        );

        let report_path = self.output_dir.join("comparison_analysis_report.html");
        std::fs::write(&report_path, html_content)?;

        // Generate supporting files
        self.generate_css_file().await?;
        self.generate_js_file().await?;

        Ok(report_path)
    }

    /// Generate the main HTML content
    fn generate_main_html(
        &self,
        accuracy_report: &AccuracyAnalysisReport,
        performance_report: &PerformanceAnalysisReport,
        regression_report: &RegressionAnalysisReport,
        executive_report: &ExecutiveSummaryReport,
    ) -> String {
        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BitNet.rs Cross-Implementation Comparison Analysis</title>
    <link rel="stylesheet" href="styles.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>🔍 BitNet.rs Cross-Implementation Comparison Analysis</h1>
            <div class="status-badge status-{status_class}">
                {overall_status}
            </div>
            <p class="timestamp">Generated: {timestamp}</p>
        </header>

        <nav class="nav-tabs">
            <button class="tab-button active" onclick="showTab('executive')">📊 Executive Summary</button>
            <button class="tab-button" onclick="showTab('accuracy')">🎯 Accuracy Analysis</button>
            <button class="tab-button" onclick="showTab('performance')">⚡ Performance Analysis</button>
            <button class="tab-button" onclick="showTab('regression')">📈 Regression Analysis</button>
        </nav>

        <main>
            {executive_section}
            {accuracy_section}
            {performance_section}
            {regression_section}
        </main>
    </div>

    <script src="report.js"></script>
</body>
</html>"#,
            status_class = self.status_to_css_class(&executive_report.overall_status.status),
            overall_status = format!("{:?}", executive_report.overall_status.status),
            timestamp = executive_report.overall_status.last_updated,
            executive_section = self.generate_executive_section(executive_report),
            accuracy_section = self.generate_accuracy_section(accuracy_report),
            performance_section = self.generate_performance_section(performance_report),
            regression_section = self.generate_regression_section(regression_report),
        )
    }

    /// Generate executive summary section
    fn generate_executive_section(&self, report: &ExecutiveSummaryReport) -> String {
        let key_metrics_cards = self.generate_key_metrics_cards(&report.key_metrics);
        let critical_issues = self.generate_critical_issues_list(&report.critical_issues);
        let achievements = self.generate_achievements_list(&report.achievements);
        let recommendations = self.generate_recommendations_list(&report.recommendations);

        format!(
            r#"<section id="executive" class="tab-content active">
                <h2>📊 Executive Summary</h2>
                
                <div class="summary-overview">
                    <div class="status-card">
                        <h3>Overall Status</h3>
                        <div class="status-indicator status-{status_class}">
                            {status}
                        </div>
                        <p class="confidence">Confidence: {confidence:.1}%</p>
                        <p class="summary-text">{summary}</p>
                    </div>
                </div>

                <div class="metrics-grid">
                    {key_metrics_cards}
                </div>

                <div class="section-grid">
                    <div class="section-card">
                        <h3>🚨 Critical Issues</h3>
                        {critical_issues}
                    </div>
                    
                    <div class="section-card">
                        <h3>🏆 Achievements</h3>
                        {achievements}
                    </div>
                </div>

                <div class="recommendations-section">
                    <h3>💡 Executive Recommendations</h3>
                    {recommendations}
                </div>
            </section>"#,
            status_class = self.status_to_css_class(&report.overall_status.status),
            status = format!("{:?}", report.overall_status.status),
            confidence = report.overall_status.confidence * 100.0,
            summary = report.overall_status.summary,
            key_metrics_cards = key_metrics_cards,
            critical_issues = critical_issues,
            achievements = achievements,
            recommendations = recommendations,
        )
    }

    /// Generate accuracy analysis section
    fn generate_accuracy_section(&self, report: &AccuracyAnalysisReport) -> String {
        let test_cases_table =
            self.generate_accuracy_test_cases_table(&report.test_case_accuracies);
        let mismatch_analysis = self.generate_mismatch_analysis(&report.mismatch_analysis);
        let distribution_chart =
            self.generate_accuracy_distribution_chart(&report.accuracy_distribution);

        format!(
            r#"<section id="accuracy" class="tab-content">
                <h2>🎯 Accuracy Analysis</h2>
                
                <div class="overview-stats">
                    <div class="stat-card">
                        <h3>Overall Accuracy</h3>
                        <div class="stat-value">{overall_accuracy:.2}%</div>
                    </div>
                    <div class="stat-card">
                        <h3>Test Cases</h3>
                        <div class="stat-value">{test_count}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Total Mismatches</h3>
                        <div class="stat-value">{total_mismatches}</div>
                    </div>
                </div>

                <div class="chart-container">
                    <h3>Accuracy Distribution</h3>
                    {distribution_chart}
                </div>

                <div class="test-cases-section">
                    <h3>Test Case Results</h3>
                    {test_cases_table}
                </div>

                <div class="mismatch-section">
                    <h3>Mismatch Analysis</h3>
                    {mismatch_analysis}
                </div>

                <div class="recommendations-section">
                    <h3>Recommendations</h3>
                    <ul class="recommendations-list">
                        {recommendations}
                    </ul>
                </div>
            </section>"#,
            overall_accuracy = report.overall_accuracy * 100.0,
            test_count = report.test_case_accuracies.len(),
            total_mismatches = report.mismatch_analysis.total_mismatches,
            distribution_chart = distribution_chart,
            test_cases_table = test_cases_table,
            mismatch_analysis = mismatch_analysis,
            recommendations = report
                .recommendations
                .iter()
                .map(|r| format!("<li>{}</li>", r))
                .collect::<Vec<_>>()
                .join("\n"),
        )
    }

    /// Generate performance analysis section
    fn generate_performance_section(&self, report: &PerformanceAnalysisReport) -> String {
        let performance_overview = self.generate_performance_overview(&report.overall_performance);
        let test_cases_table =
            self.generate_performance_test_cases_table(&report.test_case_performance);
        let bottleneck_analysis = self.generate_bottleneck_analysis(&report.bottleneck_analysis);

        format!(
            r#"<section id="performance" class="tab-content">
                <h2>⚡ Performance Analysis</h2>
                
                {performance_overview}

                <div class="test-cases-section">
                    <h3>Performance by Test Case</h3>
                    {test_cases_table}
                </div>

                <div class="bottleneck-section">
                    <h3>Bottleneck Analysis</h3>
                    {bottleneck_analysis}
                </div>

                <div class="recommendations-section">
                    <h3>Performance Recommendations</h3>
                    <ul class="recommendations-list">
                        {recommendations}
                    </ul>
                </div>
            </section>"#,
            performance_overview = performance_overview,
            test_cases_table = test_cases_table,
            bottleneck_analysis = bottleneck_analysis,
            recommendations = report
                .recommendations
                .iter()
                .map(|r| format!("<li>{}</li>", r))
                .collect::<Vec<_>>()
                .join("\n"),
        )
    }

    /// Generate regression analysis section
    fn generate_regression_section(&self, report: &RegressionAnalysisReport) -> String {
        let regression_summary = self.generate_regression_summary(&report.regression_summary);
        let accuracy_regressions =
            self.generate_accuracy_regressions_table(&report.accuracy_regressions);
        let performance_regressions =
            self.generate_performance_regressions_table(&report.performance_regressions);

        format!(
            r#"<section id="regression" class="tab-content">
                <h2>📈 Regression Analysis</h2>
                
                {regression_summary}

                <div class="regressions-section">
                    <div class="regression-category">
                        <h3>Accuracy Regressions</h3>
                        {accuracy_regressions}
                    </div>
                    
                    <div class="regression-category">
                        <h3>Performance Regressions</h3>
                        {performance_regressions}
                    </div>
                </div>
            </section>"#,
            regression_summary = regression_summary,
            accuracy_regressions = accuracy_regressions,
            performance_regressions = performance_regressions,
        )
    }

    // Helper methods for generating HTML components

    fn status_to_css_class(&self, status: &StatusLevel) -> &'static str {
        match status {
            StatusLevel::Excellent => "excellent",
            StatusLevel::Good => "good",
            StatusLevel::Acceptable => "acceptable",
            StatusLevel::Concerning => "concerning",
            StatusLevel::Critical => "critical",
        }
    }

    fn generate_key_metrics_cards(
        &self,
        metrics: &super::comparison_analysis::KeyMetrics,
    ) -> String {
        format!(
            r#"<div class="metric-card">
                <h4>Accuracy Score</h4>
                <div class="metric-value">{:.1}%</div>
            </div>
            <div class="metric-card">
                <h4>Performance Score</h4>
                <div class="metric-value">{:.1}%</div>
            </div>
            <div class="metric-card">
                <h4>Reliability Score</h4>
                <div class="metric-value">{:.1}%</div>
            </div>
            <div class="metric-card">
                <h4>Regression Rate</h4>
                <div class="metric-value">{:.1}%</div>
            </div>"#,
            metrics.accuracy_score,
            metrics.performance_score,
            metrics.reliability_score,
            metrics.regression_rate,
        )
    }

    fn generate_critical_issues_list(
        &self,
        issues: &[super::comparison_analysis::CriticalIssue],
    ) -> String {
        if issues.is_empty() {
            return "<p class=\"no-issues\">✅ No critical issues identified</p>".to_string();
        }

        issues
            .iter()
            .map(|issue| {
                format!(
                    r#"<div class="issue-card impact-{impact}">
                    <h4>{title}</h4>
                    <p>{description}</p>
                    <div class="issue-meta">
                        <span class="urgency">Urgency: {urgency:?}</span>
                        <span class="action">Action: {action}</span>
                    </div>
                </div>"#,
                    impact = format!("{:?}", issue.impact).to_lowercase(),
                    title = issue.title,
                    description = issue.description,
                    urgency = issue.urgency,
                    action = issue.recommended_action,
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn generate_achievements_list(
        &self,
        achievements: &[super::comparison_analysis::Achievement],
    ) -> String {
        if achievements.is_empty() {
            return "<p class=\"no-achievements\">No significant achievements to highlight</p>"
                .to_string();
        }

        achievements
            .iter()
            .map(|achievement| {
                format!(
                    r#"<div class="achievement-card">
                    <h4>🏆 {title}</h4>
                    <p>{description}</p>
                    {improvement}
                    <p class="business-value"><strong>Business Value:</strong> {value}</p>
                </div>"#,
                    title = achievement.title,
                    description = achievement.description,
                    improvement = achievement
                        .metric_improvement
                        .map(|i| format!("<p class=\"improvement\">Improvement: {:.1}%</p>", i))
                        .unwrap_or_default(),
                    value = achievement.business_value,
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn generate_recommendations_list(
        &self,
        recommendations: &[super::comparison_analysis::ExecutiveRecommendation],
    ) -> String {
        recommendations
            .iter()
            .map(|rec| {
                format!(
                    r#"<div class="recommendation-card priority-{priority}">
                    <h4>{title}</h4>
                    <p>{description}</p>
                    <div class="recommendation-meta">
                        <span class="effort">Effort: {effort}</span>
                        <span class="timeline">Timeline: {timeline}</span>
                    </div>
                    <p class="benefit"><strong>Expected Benefit:</strong> {benefit}</p>
                </div>"#,
                    priority = format!("{:?}", rec.priority).to_lowercase(),
                    title = rec.title,
                    description = rec.description,
                    effort = rec.effort_estimate,
                    timeline = rec.timeline,
                    benefit = rec.expected_benefit,
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn generate_accuracy_test_cases_table(&self, test_cases: &[TestCaseAccuracy]) -> String {
        let rows = test_cases
            .iter()
            .map(|tc| {
                format!(
                    r#"<tr class="severity-{severity}">
                    <td>{name}</td>
                    <td>{category}</td>
                    <td>{accuracy:.2}%</td>
                    <td>{matching}/{total}</td>
                    <td>{mismatch_pos}</td>
                    <td><span class="severity-badge severity-{severity}">{severity:?}</span></td>
                </tr>"#,
                    severity = format!("{:?}", tc.severity).to_lowercase(),
                    name = tc.test_name,
                    category = tc.category,
                    accuracy = tc.token_accuracy * 100.0,
                    matching = tc.matching_tokens,
                    total = tc.total_tokens,
                    mismatch_pos = tc
                        .first_mismatch_position
                        .map(|p| p.to_string())
                        .unwrap_or_else(|| "None".to_string()),
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"<table class="test-cases-table">
                <thead>
                    <tr>
                        <th>Test Case</th>
                        <th>Category</th>
                        <th>Accuracy</th>
                        <th>Tokens (Match/Total)</th>
                        <th>First Mismatch</th>
                        <th>Severity</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>"#,
            rows = rows
        )
    }

    fn generate_mismatch_analysis(
        &self,
        analysis: &super::comparison_analysis::MismatchAnalysis,
    ) -> String {
        let patterns = analysis
            .mismatch_patterns
            .iter()
            .map(|pattern| {
                format!(
                    r#"<div class="pattern-card">
                    <h4>{pattern_type}</h4>
                    <p>Frequency: {frequency} ({impact:.1}% impact)</p>
                    <div class="examples">
                        <strong>Examples:</strong>
                        <ul>
                            {examples}
                        </ul>
                    </div>
                </div>"#,
                    pattern_type = pattern.pattern_type,
                    frequency = pattern.frequency,
                    impact = pattern.impact_score * 100.0,
                    examples = pattern
                        .examples
                        .iter()
                        .map(|ex| format!("<li><code>{}</code></li>", ex))
                        .collect::<Vec<_>>()
                        .join("\n"),
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"<div class="mismatch-patterns">
                {patterns}
            </div>
            <div class="common-positions">
                <h4>Common Mismatch Positions</h4>
                <p>{positions}</p>
            </div>"#,
            patterns = patterns,
            positions = analysis
                .common_mismatch_positions
                .iter()
                .map(|p| p.to_string())
                .collect::<Vec<_>>()
                .join(", "),
        )
    }

    fn generate_accuracy_distribution_chart(
        &self,
        _distribution: &super::comparison_analysis::AccuracyDistribution,
    ) -> String {
        // Placeholder for chart - would integrate with Chart.js
        r#"<canvas id="accuracyDistributionChart" width="400" height="200"></canvas>
        <script>
            // Chart.js code would go here
            console.log('Accuracy distribution chart placeholder');
        </script>"#
            .to_string()
    }

    fn generate_performance_overview(
        &self,
        overview: &super::comparison_analysis::PerformanceOverview,
    ) -> String {
        format!(
            r#"<div class="performance-overview">
                <div class="perf-stat-card">
                    <h4>Average Throughput Ratio</h4>
                    <div class="stat-value">{:.2}x</div>
                </div>
                <div class="perf-stat-card">
                    <h4>Performance Improvement</h4>
                    <div class="stat-value {improvement_class}">{improvement:+.1}%</div>
                </div>
                <div class="perf-stat-card">
                    <h4>Memory Efficiency</h4>
                    <div class="stat-value {memory_class}">{memory:+.1}%</div>
                </div>
                <div class="perf-stat-card">
                    <h4>Regressions</h4>
                    <div class="stat-value regression">{regressions}</div>
                </div>
            </div>"#,
            improvement = overview.performance_improvement,
            improvement_class = if overview.performance_improvement > 0.0 {
                "positive"
            } else {
                "negative"
            },
            memory = overview.memory_efficiency,
            memory_class = if overview.memory_efficiency > 0.0 {
                "positive"
            } else {
                "negative"
            },
            regressions = overview.regression_count,
        )
    }

    fn generate_performance_test_cases_table(&self, test_cases: &[TestCasePerformance]) -> String {
        let rows = test_cases
            .iter()
            .map(|tc| {
                format!(
                    r#"<tr class="category-{category}">
                    <td>{name}</td>
                    <td>{throughput:.2}x</td>
                    <td>{memory:.2}x</td>
                    <td>{rust_tps:.1}</td>
                    <td>{cpp_tps:.1}</td>
                    <td><span class="category-badge category-{category}">{category:?}</span></td>
                    <td>{regression}</td>
                </tr>"#,
                    category = format!("{:?}", tc.performance_category).to_lowercase(),
                    name = tc.test_name,
                    throughput = tc.throughput_ratio,
                    memory = tc.memory_ratio,
                    rust_tps = tc.rust_tokens_per_second,
                    cpp_tps = tc.cpp_tokens_per_second,
                    regression = if tc.regression {
                        "⚠️ Yes"
                    } else {
                        "✅ No"
                    },
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"<table class="performance-table">
                <thead>
                    <tr>
                        <th>Test Case</th>
                        <th>Throughput Ratio</th>
                        <th>Memory Ratio</th>
                        <th>Rust TPS</th>
                        <th>C++ TPS</th>
                        <th>Category</th>
                        <th>Regression</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>"#,
            rows = rows
        )
    }

    fn generate_bottleneck_analysis(
        &self,
        analysis: &super::comparison_analysis::BottleneckAnalysis,
    ) -> String {
        let bottlenecks = analysis
            .primary_bottlenecks
            .iter()
            .map(|bottleneck| {
                format!(
                    r#"<div class="bottleneck-card">
                    <h4>{component}</h4>
                    <p>{description}</p>
                    <p><strong>Impact Score:</strong> {impact:.2}</p>
                    <p><strong>Affected Tests:</strong> {affected_count}</p>
                </div>"#,
                    component = bottleneck.component,
                    description = bottleneck.description,
                    impact = bottleneck.impact_score,
                    affected_count = bottleneck.affected_tests.len(),
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        let opportunities = analysis
            .optimization_opportunities
            .iter()
            .map(|opp| {
                format!(
                    r#"<div class="opportunity-card priority-{priority}">
                    <h4>{area}</h4>
                    <p><strong>Potential Improvement:</strong> {improvement:.1}%</p>
                    <p><strong>Effort:</strong> {effort}</p>
                    <p><strong>Priority:</strong> {priority:?}</p>
                </div>"#,
                    priority = format!("{:?}", opp.priority).to_lowercase(),
                    area = opp.area,
                    improvement = opp.potential_improvement,
                    effort = opp.effort_estimate,
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"<div class="bottlenecks-section">
                <h4>Primary Bottlenecks</h4>
                {bottlenecks}
            </div>
            <div class="opportunities-section">
                <h4>Optimization Opportunities</h4>
                {opportunities}
            </div>"#,
            bottlenecks = bottlenecks,
            opportunities = opportunities,
        )
    }

    fn generate_regression_summary(
        &self,
        summary: &super::comparison_analysis::RegressionSummary,
    ) -> String {
        format!(
            r#"<div class="regression-summary">
                <div class="regression-stat">
                    <h4>Total Regressions</h4>
                    <div class="stat-value">{total}</div>
                </div>
                <div class="regression-stat">
                    <h4>Accuracy Regressions</h4>
                    <div class="stat-value">{accuracy}</div>
                </div>
                <div class="regression-stat">
                    <h4>Performance Regressions</h4>
                    <div class="stat-value">{performance}</div>
                </div>
                <div class="regression-stat">
                    <h4>Critical Regressions</h4>
                    <div class="stat-value critical">{critical}</div>
                </div>
                <div class="regression-stat">
                    <h4>Regression Rate</h4>
                    <div class="stat-value">{rate:.1}%</div>
                </div>
            </div>"#,
            total = summary.total_regressions,
            accuracy = summary.accuracy_regressions,
            performance = summary.performance_regressions,
            critical = summary.critical_regressions,
            rate = summary.regression_rate,
        )
    }

    fn generate_accuracy_regressions_table(
        &self,
        regressions: &[super::comparison_analysis::AccuracyRegression],
    ) -> String {
        if regressions.is_empty() {
            return "<p class=\"no-regressions\">✅ No accuracy regressions detected</p>"
                .to_string();
        }

        let rows = regressions
            .iter()
            .map(|reg| {
                format!(
                    r#"<tr class="severity-{severity}">
                    <td>{test_case}</td>
                    <td>{previous:.2}%</td>
                    <td>{current:.2}%</td>
                    <td>{magnitude:.2}%</td>
                    <td><span class="severity-badge severity-{severity}">{severity:?}</span></td>
                </tr>"#,
                    severity = format!("{:?}", reg.severity).to_lowercase(),
                    test_case = reg.test_case,
                    previous = reg.previous_accuracy * 100.0,
                    current = reg.current_accuracy * 100.0,
                    magnitude = reg.regression_magnitude * 100.0,
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"<table class="regressions-table">
                <thead>
                    <tr>
                        <th>Test Case</th>
                        <th>Previous Accuracy</th>
                        <th>Current Accuracy</th>
                        <th>Regression</th>
                        <th>Severity</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>"#,
            rows = rows
        )
    }

    fn generate_performance_regressions_table(
        &self,
        regressions: &[super::comparison_analysis::PerformanceRegression],
    ) -> String {
        if regressions.is_empty() {
            return "<p class=\"no-regressions\">✅ No performance regressions detected</p>"
                .to_string();
        }

        let rows = regressions
            .iter()
            .map(|reg| {
                format!(
                    r#"<tr class="severity-{severity}">
                    <td>{test_case}</td>
                    <td>{metric}</td>
                    <td>{previous:.2}</td>
                    <td>{current:.2}</td>
                    <td>{percentage:.1}%</td>
                    <td><span class="severity-badge severity-{severity}">{severity:?}</span></td>
                </tr>"#,
                    severity = format!("{:?}", reg.severity).to_lowercase(),
                    test_case = reg.test_case,
                    metric = reg.metric,
                    previous = reg.previous_value,
                    current = reg.current_value,
                    percentage = reg.regression_percentage,
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            r#"<table class="regressions-table">
                <thead>
                    <tr>
                        <th>Test Case</th>
                        <th>Metric</th>
                        <th>Previous Value</th>
                        <th>Current Value</th>
                        <th>Regression %</th>
                        <th>Severity</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>"#,
            rows = rows
        )
    }

    /// Generate CSS file
    async fn generate_css_file(&self) -> Result<(), Box<dyn std::error::Error>> {
        let css_content = include_str!("../../assets/comparison_report.css");
        let css_path = self.output_dir.join("styles.css");
        std::fs::write(css_path, css_content)?;
        Ok(())
    }

    /// Generate JavaScript file
    async fn generate_js_file(&self) -> Result<(), Box<dyn std::error::Error>> {
        let js_content = r#"
// Tab switching functionality
function showTab(tabName) {
    // Hide all tab contents
    const contents = document.querySelectorAll('.tab-content');
    contents.forEach(content => content.classList.remove('active'));
    
    // Remove active class from all buttons
    const buttons = document.querySelectorAll('.tab-button');
    buttons.forEach(button => button.classList.remove('active'));
    
    // Show selected tab content
    document.getElementById(tabName).classList.add('active');
    
    // Add active class to clicked button
    event.target.classList.add('active');
}

// Initialize charts when page loads
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
});

function initializeCharts() {
    // Initialize accuracy distribution chart
    const accuracyCtx = document.getElementById('accuracyDistributionChart');
    if (accuracyCtx) {
        new Chart(accuracyCtx, {
            type: 'histogram',
            data: {
                // Chart data would be populated from the analysis results
                labels: ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'],
                datasets: [{
                    label: 'Test Cases',
                    data: [0, 0, 0, 0, 0, 0, 1, 2, 5, 12], // Example data
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Test Cases'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Accuracy Range'
                        }
                    }
                }
            }
        });
    }
}

// Add interactive features
document.addEventListener('DOMContentLoaded', function() {
    // Add click handlers for expandable sections
    const expandableHeaders = document.querySelectorAll('.expandable-header');
    expandableHeaders.forEach(header => {
        header.addEventListener('click', function() {
            const content = this.nextElementSibling;
            content.classList.toggle('expanded');
            this.classList.toggle('expanded');
        });
    });
    
    // Add tooltips for technical terms
    const tooltipElements = document.querySelectorAll('[data-tooltip]');
    tooltipElements.forEach(element => {
        element.addEventListener('mouseenter', showTooltip);
        element.addEventListener('mouseleave', hideTooltip);
    });
});

function showTooltip(event) {
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    tooltip.textContent = event.target.getAttribute('data-tooltip');
    document.body.appendChild(tooltip);
    
    const rect = event.target.getBoundingClientRect();
    tooltip.style.left = rect.left + 'px';
    tooltip.style.top = (rect.top - tooltip.offsetHeight - 5) + 'px';
}

function hideTooltip() {
    const tooltip = document.querySelector('.tooltip');
    if (tooltip) {
        tooltip.remove();
    }
}
"#;
        let js_path = self.output_dir.join("report.js");
        std::fs::write(js_path, js_content)?;
        Ok(())
    }
}
