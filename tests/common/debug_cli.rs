use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::debugging::DebugReport;
use super::errors::{TestError, TestOpResult};

/// Command-line interface for debugging test issues
pub struct DebugCli {
    debug_dir: PathBuf,
    verbose: bool,
}

impl DebugCli {
    /// Create a new debug CLI instance
    pub fn new(debug_dir: PathBuf, verbose: bool) -> Self {
        Self { debug_dir, verbose }
    }

    /// Analyze a debug report and provide insights
    pub async fn analyze_report(&self, report_path: &Path) -> TestOpResult<AnalysisResult> {
        if self.verbose {
            println!("🔍 Analyzing debug report: {}", report_path.display());
        }

        let report_content = tokio::fs::read_to_string(report_path).await?;
        let report: DebugReport = serde_json::from_str(&report_content)
            .map_err(|e| TestError::execution(format!("Failed to parse debug report: {}", e)))?;

        let analysis = AnalysisResult {
            report_summary: self.generate_report_summary(&report),
            critical_issues: self.identify_critical_issues(&report),
            performance_issues: self.identify_performance_issues(&report),
            stability_issues: self.identify_stability_issues(&report),
            recommendations: self.generate_recommendations(&report),
            quick_fixes: self.suggest_quick_fixes(&report),
        };

        if self.verbose {
            println!("✅ Analysis complete. Found {} critical issues, {} performance issues, {} stability issues",
                analysis.critical_issues.len(),
                analysis.performance_issues.len(),
                analysis.stability_issues.len()
            );
        }

        Ok(analysis)
    }

    /// Find similar issues across multiple debug reports
    pub async fn find_patterns(&self) -> TestOpResult<Vec<IssuePattern>> {
        if self.verbose {
            println!("🔍 Searching for patterns across debug reports...");
        }

        let mut reports = Vec::new();
        let mut entries = tokio::fs::read_dir(&self.debug_dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_file() && path.extension().map_or(false, |ext| ext == "json") {
                if let Ok(content) = tokio::fs::read_to_string(&path).await {
                    if let Ok(report) = serde_json::from_str::<DebugReport>(&content) {
                        reports.push((path, report));
                    }
                }
            }
        }

        let patterns = self.analyze_patterns(&reports);

        if self.verbose {
            println!("✅ Found {} patterns across {} reports", patterns.len(), reports.len());
        }

        Ok(patterns)
    }

    /// Generate a troubleshooting guide for specific test failures
    pub async fn generate_troubleshooting_guide(&self, test_name: &str) -> TestOpResult<String> {
        if self.verbose {
            println!("🔍 Generating troubleshooting guide for test: {}", test_name);
        }

        let mut guide = String::new();
        guide.push_str(&format!("# Troubleshooting Guide for Test: {}\n\n", test_name));

        // Find relevant debug reports
        let relevant_reports = self.find_reports_for_test(test_name).await?;

        if relevant_reports.is_empty() {
            guide.push_str("No debug reports found for this test.\n");
            guide.push_str("To generate debug information, run the test with debugging enabled:\n");
            guide.push_str("```bash\n");
            guide.push_str("BITNET_DEBUG_ENABLED=true cargo test ");
            guide.push_str(test_name);
            guide.push_str("\n```\n");
            return Ok(guide);
        }

        // Analyze common failure patterns
        let failure_patterns = self.analyze_test_failures(&relevant_reports, test_name);

        guide.push_str("## Common Issues\n\n");
        for (pattern, count) in &failure_patterns {
            guide.push_str(&format!("### {} (occurred {} times)\n\n", pattern.title, count));
            guide.push_str(&format!("{}\n\n", pattern.description));

            guide.push_str("**Troubleshooting Steps:**\n");
            for (i, step) in pattern.steps.iter().enumerate() {
                guide.push_str(&format!("{}. {}\n", i + 1, step));
            }
            guide.push_str("\n");
        }

        // Add general debugging tips
        guide.push_str("## General Debugging Tips\n\n");
        guide.push_str(
            "1. **Run in isolation**: Execute the test alone to eliminate interference\n",
        );
        guide.push_str("2. **Check environment**: Verify environment variables and system state\n");
        guide.push_str("3. **Enable verbose logging**: Set `RUST_LOG=debug` for detailed logs\n");
        guide.push_str(
            "4. **Monitor resources**: Check memory and CPU usage during test execution\n",
        );
        guide.push_str("5. **Review recent changes**: Check if recent code changes might have caused the issue\n\n");

        // Add specific commands
        guide.push_str("## Useful Commands\n\n");
        guide.push_str("```bash\n");
        guide.push_str("# Run test with full debugging\n");
        guide.push_str(&format!(
            "BITNET_DEBUG_ENABLED=true BITNET_DEBUG_VERBOSE=true cargo test {}\n\n",
            test_name
        ));
        guide.push_str("# Run test in isolation with logging\n");
        guide.push_str(&format!("RUST_LOG=debug cargo test {} -- --nocapture\n\n", test_name));
        guide.push_str("# Check test with memory monitoring\n");
        guide.push_str(&format!("valgrind --tool=memcheck cargo test {}\n", test_name));
        guide.push_str("```\n");

        Ok(guide)
    }

    /// Interactive debugging session
    pub async fn interactive_debug(&self) -> TestOpResult<()> {
        println!("🔍 BitNet.rs Test Debugger - Interactive Mode");
        println!("Type 'help' for available commands, 'quit' to exit\n");

        loop {
            print!("debug> ");
            use std::io::{self, Write};
            io::stdout().flush().unwrap();

            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            let input = input.trim();

            if input.is_empty() {
                continue;
            }

            match input {
                "quit" | "exit" => break,
                "help" => self.print_help(),
                cmd if cmd.starts_with("analyze ") => {
                    let path = cmd.strip_prefix("analyze ").unwrap();
                    match self.analyze_report(Path::new(path)).await {
                        Ok(analysis) => self.print_analysis(&analysis),
                        Err(e) => println!("❌ Error: {}", e),
                    }
                }
                cmd if cmd.starts_with("guide ") => {
                    let test_name = cmd.strip_prefix("guide ").unwrap();
                    match self.generate_troubleshooting_guide(test_name).await {
                        Ok(guide) => println!("{}", guide),
                        Err(e) => println!("❌ Error: {}", e),
                    }
                }
                "patterns" => match self.find_patterns().await {
                    Ok(patterns) => self.print_patterns(&patterns),
                    Err(e) => println!("❌ Error: {}", e),
                },
                "list" => match self.list_debug_reports().await {
                    Ok(reports) => self.print_report_list(&reports),
                    Err(e) => println!("❌ Error: {}", e),
                },
                _ => println!("❌ Unknown command: {}. Type 'help' for available commands.", input),
            }
        }

        println!("👋 Goodbye!");
        Ok(())
    }

    // Private helper methods

    fn generate_report_summary(&self, report: &DebugReport) -> ReportSummary {
        let duration = report.end_time.duration_since(report.start_time).unwrap_or_default();
        let success_rate = if report.total_tests > 0 {
            ((report.total_tests - report.failed_tests) as f64 / report.total_tests as f64) * 100.0
        } else {
            0.0
        };

        ReportSummary {
            session_id: report.session_id.clone(),
            duration,
            total_tests: report.total_tests,
            failed_tests: report.failed_tests,
            success_rate,
            error_count: report.error_count,
            peak_memory: report.performance_summary.peak_memory,
        }
    }

    fn identify_critical_issues(&self, report: &DebugReport) -> Vec<CriticalIssue> {
        let mut issues = Vec::new();

        // High failure rate
        if report.total_tests > 0 {
            let failure_rate = (report.failed_tests as f64 / report.total_tests as f64) * 100.0;
            if failure_rate > 50.0 {
                issues.push(CriticalIssue {
                    title: "High Test Failure Rate".to_string(),
                    description: format!("{:.1}% of tests failed", failure_rate),
                    severity: IssueSeverity::Critical,
                    affected_tests: report
                        .test_summaries
                        .iter()
                        .filter(|t| t.failed_phases > 0)
                        .map(|t| t.test_name.clone())
                        .collect(),
                });
            }
        }

        // Memory issues
        if report.performance_summary.peak_memory > 2 * 1024 * 1024 * 1024 {
            // > 2GB
            issues.push(CriticalIssue {
                title: "Excessive Memory Usage".to_string(),
                description: format!(
                    "Peak memory usage: {} MB",
                    report.performance_summary.peak_memory / (1024 * 1024)
                ),
                severity: IssueSeverity::High,
                affected_tests: report.test_summaries.iter()
                    .filter(|t| t.peak_memory > 1024 * 1024 * 1024) // > 1GB
                    .map(|t| t.test_name.clone())
                    .collect(),
            });
        }

        // High error count
        if report.error_count > 20 {
            issues.push(CriticalIssue {
                title: "High Error Count".to_string(),
                description: format!("{} errors reported", report.error_count),
                severity: IssueSeverity::High,
                affected_tests: Vec::new(),
            });
        }

        issues
    }

    fn identify_performance_issues(&self, report: &DebugReport) -> Vec<PerformanceIssue> {
        let mut issues = Vec::new();

        // Slow tests
        for test in &report.test_summaries {
            if test.duration > Duration::from_secs(300) {
                // > 5 minutes
                issues.push(PerformanceIssue {
                    test_name: test.test_name.clone(),
                    issue_type: PerformanceIssueType::SlowExecution,
                    value: test.duration.as_secs_f64(),
                    threshold: 300.0,
                    description: format!(
                        "Test took {:.1}s to complete",
                        test.duration.as_secs_f64()
                    ),
                });
            }
        }

        // High memory usage per test
        for test in &report.test_summaries {
            if test.peak_memory > 512 * 1024 * 1024 {
                // > 512MB
                issues.push(PerformanceIssue {
                    test_name: test.test_name.clone(),
                    issue_type: PerformanceIssueType::HighMemoryUsage,
                    value: test.peak_memory as f64,
                    threshold: 512.0 * 1024.0 * 1024.0,
                    description: format!(
                        "Test used {} MB of memory",
                        test.peak_memory / (1024 * 1024)
                    ),
                });
            }
        }

        issues
    }

    fn identify_stability_issues(&self, report: &DebugReport) -> Vec<StabilityIssue> {
        let mut issues = Vec::new();

        // Tests with multiple failed phases
        for test in &report.test_summaries {
            if test.failed_phases > 1 {
                issues.push(StabilityIssue {
                    test_name: test.test_name.clone(),
                    issue_type: StabilityIssueType::MultiplePhaseFailures,
                    description: format!(
                        "Test had {} failed phases out of {}",
                        test.failed_phases, test.phases
                    ),
                });
            }
        }

        // Tests with many debug messages (might indicate instability)
        for test in &report.test_summaries {
            if test.debug_messages > 50 {
                issues.push(StabilityIssue {
                    test_name: test.test_name.clone(),
                    issue_type: StabilityIssueType::ExcessiveLogging,
                    description: format!("Test generated {} debug messages", test.debug_messages),
                });
            }
        }

        issues
    }

    fn generate_recommendations(&self, report: &DebugReport) -> Vec<String> {
        let mut recommendations = Vec::new();

        if report.failed_tests > 0 {
            recommendations.push("Review failed tests for common error patterns".to_string());
        }

        if report.performance_summary.peak_memory > 1024 * 1024 * 1024 {
            recommendations.push("Consider optimizing memory usage in tests".to_string());
        }

        if report.performance_summary.average_test_duration > Duration::from_secs(60) {
            recommendations.push("Investigate slow tests and consider optimization".to_string());
        }

        if report.error_count > 10 {
            recommendations.push("Review error handling and test stability".to_string());
        }

        recommendations
    }

    fn suggest_quick_fixes(&self, report: &DebugReport) -> Vec<QuickFix> {
        let mut fixes = Vec::new();

        // Suggest timeout increases for slow tests
        for test in &report.test_summaries {
            if test.duration > Duration::from_secs(180) {
                fixes.push(QuickFix {
                    title: format!("Increase timeout for {}", test.test_name),
                    description: "Test appears to be slow, consider increasing timeout".to_string(),
                    command: Some(format!(
                        "# Add to test configuration:\ntimeout = \"{}s\"",
                        (test.duration.as_secs() * 2).max(300)
                    )),
                });
            }
        }

        // Suggest memory optimization
        if report.performance_summary.peak_memory > 1024 * 1024 * 1024 {
            fixes.push(QuickFix {
                title: "Optimize memory usage".to_string(),
                description: "High memory usage detected, consider reducing test data size"
                    .to_string(),
                command: Some(
                    "# Consider using smaller test fixtures or streaming data".to_string(),
                ),
            });
        }

        fixes
    }

    fn analyze_patterns(&self, reports: &[(PathBuf, DebugReport)]) -> Vec<IssuePattern> {
        let mut patterns = Vec::new();
        let mut error_counts: HashMap<String, usize> = HashMap::new();
        let mut slow_tests: HashMap<String, usize> = HashMap::new();

        // Analyze error patterns
        for (_, report) in reports {
            for test in &report.test_summaries {
                for issue in &test.issues {
                    *error_counts.entry(issue.clone()).or_insert(0) += 1;
                }

                if test.duration > Duration::from_secs(120) {
                    *slow_tests.entry(test.test_name.clone()).or_insert(0) += 1;
                }
            }
        }

        // Create patterns for frequent errors
        for (error, count) in error_counts {
            if count > 2 {
                patterns.push(IssuePattern {
                    title: format!("Frequent Error: {}", error),
                    description: format!(
                        "This error occurred {} times across different test runs",
                        count
                    ),
                    frequency: count,
                    steps: vec![
                        "Review the error message and context".to_string(),
                        "Check if this is a known issue in the project".to_string(),
                        "Consider adding error handling or fixing the root cause".to_string(),
                    ],
                });
            }
        }

        // Create patterns for consistently slow tests
        for (test_name, count) in slow_tests {
            if count > 1 {
                patterns.push(IssuePattern {
                    title: format!("Consistently Slow Test: {}", test_name),
                    description: format!("This test was slow in {} different runs", count),
                    frequency: count,
                    steps: vec![
                        "Profile the test to identify bottlenecks".to_string(),
                        "Consider optimizing the test logic or data".to_string(),
                        "Check if the test can be parallelized".to_string(),
                    ],
                });
            }
        }

        patterns
    }

    async fn find_reports_for_test(&self, test_name: &str) -> TestOpResult<Vec<DebugReport>> {
        let mut reports = Vec::new();
        let mut entries = tokio::fs::read_dir(&self.debug_dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_file() && path.extension().map_or(false, |ext| ext == "json") {
                if let Ok(content) = tokio::fs::read_to_string(&path).await {
                    if let Ok(report) = serde_json::from_str::<DebugReport>(&content) {
                        if report.test_summaries.iter().any(|t| t.test_name == test_name) {
                            reports.push(report);
                        }
                    }
                }
            }
        }

        Ok(reports)
    }

    fn analyze_test_failures(
        &self,
        reports: &[DebugReport],
        test_name: &str,
    ) -> HashMap<IssuePattern, usize> {
        let mut patterns = HashMap::new();

        for report in reports {
            if let Some(test_summary) =
                report.test_summaries.iter().find(|t| t.test_name == test_name)
            {
                for issue in &test_summary.issues {
                    let pattern = IssuePattern {
                        title: issue.clone(),
                        description: format!("Issue in test {}", test_name),
                        frequency: 1,
                        steps: vec![
                            "Check the test implementation".to_string(),
                            "Review test data and environment".to_string(),
                            "Consider adding debug logging".to_string(),
                        ],
                    };
                    *patterns.entry(pattern).or_insert(0) += 1;
                }
            }
        }

        patterns
    }

    async fn list_debug_reports(&self) -> TestOpResult<Vec<(PathBuf, ReportSummary)>> {
        let mut reports = Vec::new();
        let mut entries = tokio::fs::read_dir(&self.debug_dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_file() && path.extension().map_or(false, |ext| ext == "json") {
                if let Ok(content) = tokio::fs::read_to_string(&path).await {
                    if let Ok(report) = serde_json::from_str::<DebugReport>(&content) {
                        let summary = self.generate_report_summary(&report);
                        reports.push((path, summary));
                    }
                }
            }
        }

        reports.sort_by(|a, b| b.1.duration.cmp(&a.1.duration));
        Ok(reports)
    }

    fn print_help(&self) {
        println!("Available commands:");
        println!("  analyze <path>  - Analyze a debug report file");
        println!("  guide <test>    - Generate troubleshooting guide for a test");
        println!("  patterns        - Find common patterns across all reports");
        println!("  list           - List all available debug reports");
        println!("  help           - Show this help message");
        println!("  quit/exit      - Exit the debugger");
    }

    fn print_analysis(&self, analysis: &AnalysisResult) {
        println!("\n📊 Debug Report Analysis");
        println!("========================");

        println!("\n📋 Summary:");
        println!("  Session: {}", analysis.report_summary.session_id);
        println!("  Duration: {:?}", analysis.report_summary.duration);
        println!(
            "  Tests: {} total, {} failed ({:.1}% success rate)",
            analysis.report_summary.total_tests,
            analysis.report_summary.failed_tests,
            analysis.report_summary.success_rate
        );
        println!("  Peak Memory: {} MB", analysis.report_summary.peak_memory / (1024 * 1024));

        if !analysis.critical_issues.is_empty() {
            println!("\n🚨 Critical Issues:");
            for issue in &analysis.critical_issues {
                println!("  • {} ({})", issue.title, issue.severity);
                println!("    {}", issue.description);
            }
        }

        if !analysis.performance_issues.is_empty() {
            println!("\n⚡ Performance Issues:");
            for issue in &analysis.performance_issues {
                println!("  • {}: {}", issue.test_name, issue.description);
            }
        }

        if !analysis.recommendations.is_empty() {
            println!("\n💡 Recommendations:");
            for (i, rec) in analysis.recommendations.iter().enumerate() {
                println!("  {}. {}", i + 1, rec);
            }
        }

        if !analysis.quick_fixes.is_empty() {
            println!("\n🔧 Quick Fixes:");
            for fix in &analysis.quick_fixes {
                println!("  • {}", fix.title);
                println!("    {}", fix.description);
                if let Some(cmd) = &fix.command {
                    println!("    {}", cmd);
                }
            }
        }
    }

    fn print_patterns(&self, patterns: &[IssuePattern]) {
        println!("\n🔍 Issue Patterns");
        println!("=================");

        for pattern in patterns {
            println!("\n📌 {} (frequency: {})", pattern.title, pattern.frequency);
            println!("   {}", pattern.description);
            println!("   Steps to resolve:");
            for (i, step) in pattern.steps.iter().enumerate() {
                println!("   {}. {}", i + 1, step);
            }
        }
    }

    fn print_report_list(&self, reports: &[(PathBuf, ReportSummary)]) {
        println!("\n📁 Available Debug Reports");
        println!("==========================");

        for (path, summary) in reports {
            println!("\n📄 {}", path.file_name().unwrap().to_string_lossy());
            println!("   Session: {}", summary.session_id);
            println!("   Tests: {} total, {} failed", summary.total_tests, summary.failed_tests);
            println!("   Duration: {:?}", summary.duration);
            println!("   Success Rate: {:.1}%", summary.success_rate);
        }
    }
}

// Data structures for analysis results

#[derive(Debug)]
pub struct AnalysisResult {
    pub report_summary: ReportSummary,
    pub critical_issues: Vec<CriticalIssue>,
    pub performance_issues: Vec<PerformanceIssue>,
    pub stability_issues: Vec<StabilityIssue>,
    pub recommendations: Vec<String>,
    pub quick_fixes: Vec<QuickFix>,
}

#[derive(Debug)]
pub struct ReportSummary {
    pub session_id: String,
    pub duration: Duration,
    pub total_tests: usize,
    pub failed_tests: usize,
    pub success_rate: f64,
    pub error_count: usize,
    pub peak_memory: u64,
}

#[derive(Debug)]
pub struct CriticalIssue {
    pub title: String,
    pub description: String,
    pub severity: IssueSeverity,
    pub affected_tests: Vec<String>,
}

#[derive(Debug)]
pub enum IssueSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl std::fmt::Display for IssueSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "LOW"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::High => write!(f, "HIGH"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

#[derive(Debug)]
pub struct PerformanceIssue {
    pub test_name: String,
    pub issue_type: PerformanceIssueType,
    pub value: f64,
    pub threshold: f64,
    pub description: String,
}

#[derive(Debug)]
pub enum PerformanceIssueType {
    SlowExecution,
    HighMemoryUsage,
    HighCpuUsage,
}

#[derive(Debug)]
pub struct StabilityIssue {
    pub test_name: String,
    pub issue_type: StabilityIssueType,
    pub description: String,
}

#[derive(Debug)]
pub enum StabilityIssueType {
    MultiplePhaseFailures,
    ExcessiveLogging,
    InconsistentResults,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IssuePattern {
    pub title: String,
    pub description: String,
    pub frequency: usize,
    pub steps: Vec<String>,
}

#[derive(Debug)]
pub struct QuickFix {
    pub title: String,
    pub description: String,
    pub command: Option<String>,
}

/// Create a debug CLI instance from environment
pub fn create_debug_cli() -> DebugCli {
    let debug_dir = std::env::var("BITNET_DEBUG_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("tests/debug"));

    let verbose =
        std::env::var("BITNET_DEBUG_VERBOSE").map(|v| v.parse().unwrap_or(false)).unwrap_or(false);

    DebugCli::new(debug_dir, verbose)
}
