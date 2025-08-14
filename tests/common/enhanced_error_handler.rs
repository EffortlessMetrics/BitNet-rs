/// Enhanced error handler that provides comprehensive debugging support
///
/// This module integrates error analysis, debugging context, and actionable
/// recommendations into the test execution flow.
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{error, info, warn};

use super::{
    error_analysis::{ErrorAnalysis, ErrorAnalyzer, ErrorContext, SystemResourceSnapshot},
    errors::{ErrorReport, ErrorSeverity, TestError},
    logging::{DebugContext, LoggingManager},
};

/// Enhanced error handler that provides actionable debugging information
pub struct EnhancedErrorHandler {
    analyzer: Arc<RwLock<ErrorAnalyzer>>,
    logging_manager: Arc<LoggingManager>,
    config: ErrorHandlerConfig,
    error_reports: Arc<RwLock<Vec<ErrorReport>>>,
}

impl EnhancedErrorHandler {
    /// Create a new enhanced error handler
    pub fn new(logging_manager: Arc<LoggingManager>, config: ErrorHandlerConfig) -> Self {
        Self {
            analyzer: Arc::new(RwLock::new(ErrorAnalyzer::new())),
            logging_manager,
            config,
            error_reports: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Handle a test error with comprehensive analysis and debugging support
    pub async fn handle_test_error(
        &self,
        test_name: &str,
        error: &TestError,
        debug_context: &DebugContext,
        execution_context: TestExecutionContext,
    ) -> Result<ErrorHandlingResult, TestError> {
        // Create error context for analysis
        let error_context = self.create_error_context(test_name, &execution_context).await;

        // Perform comprehensive error analysis
        let mut analyzer = self.analyzer.write().await;
        let analysis = analyzer.analyze_error(error, error_context).await;
        drop(analyzer);

        // Generate detailed error report
        let error_report = error.create_error_report();

        // Store error report
        let mut reports = self.error_reports.write().await;
        reports.push(error_report.clone());
        drop(reports);

        // Log structured error information
        self.log_error_details(test_name, error, &analysis).await;

        // Handle failure artifacts collection
        self.logging_manager.handle_test_failure(test_name, error, debug_context).await?;

        // Save detailed analysis if configured
        if self.config.save_analysis_reports {
            self.save_analysis_report(test_name, &analysis).await?;
        }

        // Generate debugging guide
        let debugging_guide = analysis.generate_debugging_guide();

        // Send notifications if configured (before moving analysis)
        if self.config.send_notifications {
            self.send_error_notification(test_name, error, &analysis).await?;
        }

        // Create actionable result
        let result = ErrorHandlingResult {
            error_report,
            analysis,
            debugging_guide,
            recovery_suggestions: error.recovery_suggestions(),
            troubleshooting_steps: error.troubleshooting_steps(),
            should_retry: self.should_retry_test(error, &execution_context),
            retry_delay: self.calculate_retry_delay(error),
        };

        // Log debugging guide if configured
        if self.config.log_debugging_guides {
            info!("Debugging guide for test '{}':\n{}", test_name, result.debugging_guide);
        }

        Ok(result)
    }

    /// Create error context from test execution information
    async fn create_error_context(
        &self,
        test_name: &str,
        execution_context: &TestExecutionContext,
    ) -> ErrorContext {
        ErrorContext {
            test_name: test_name.to_string(),
            test_suite: execution_context.test_suite.clone(),
            execution_time: execution_context.execution_time,
            concurrent_tests: execution_context.concurrent_tests,
            is_ci_environment: self.is_ci_environment(),
            system_resources: self.capture_system_resources().await,
            environment_variables: self.collect_relevant_env_vars(),
        }
    }

    /// Log detailed error information with structured data
    async fn log_error_details(
        &self,
        test_name: &str,
        error: &TestError,
        analysis: &ErrorAnalysis,
    ) {
        error!(
            test_name = test_name,
            error_category = error.category(),
            error_severity = %error.severity(),
            confidence_score = analysis.confidence_score,
            debugging_priority = %analysis.debugging_priority,
            "Test failed with detailed analysis"
        );

        // Log primary cause if identified
        if let Some(primary_cause) = &analysis.root_cause_analysis.primary_cause {
            warn!(
                test_name = test_name,
                cause = primary_cause.cause,
                likelihood = primary_cause.likelihood,
                "Primary cause identified"
            );
        }

        // Log top recommendations
        for (i, recommendation) in analysis.recommendations.iter().take(3).enumerate() {
            info!(
                test_name = test_name,
                recommendation_priority = %recommendation.priority,
                recommendation_title = recommendation.title,
                success_probability = recommendation.success_probability,
                "Recommendation #{}", i + 1
            );
        }
    }

    /// Save analysis report to file
    async fn save_analysis_report(
        &self,
        test_name: &str,
        analysis: &ErrorAnalysis,
    ) -> Result<(), TestError> {
        let output_dir = &self.config.analysis_output_dir;
        tokio::fs::create_dir_all(output_dir).await?;

        let timestamp = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs();

        let filename = format!("error_analysis_{}_{}.json", test_name, timestamp);
        let file_path = output_dir.join(filename);

        analysis
            .save_to_file(&file_path)
            .await
            .map_err(|e| TestError::execution(format!("Failed to save analysis report: {}", e)))?;

        info!("Saved error analysis report to: {:?}", file_path);
        Ok(())
    }

    /// Determine if a test should be retried based on error characteristics
    fn should_retry_test(&self, error: &TestError, context: &TestExecutionContext) -> bool {
        // Don't retry if retries are disabled
        if !self.config.enable_retries {
            return false;
        }

        // Don't retry if already at max attempts
        if context.retry_count >= self.config.max_retry_attempts {
            return false;
        }

        // Only retry recoverable errors
        if !error.is_recoverable() {
            return false;
        }

        // Retry based on error type
        match error {
            TestError::TimeoutError { .. } => true,
            TestError::HttpError(_) => true,
            TestError::FixtureError(_) => true,
            TestError::IoError(_) => true,
            _ => false,
        }
    }

    /// Calculate retry delay based on error type and attempt count
    fn calculate_retry_delay(&self, error: &TestError) -> Option<Duration> {
        if !error.is_recoverable() {
            return None;
        }

        let base_delay = match error {
            TestError::HttpError(_) => Duration::from_secs(5),
            TestError::FixtureError(_) => Duration::from_secs(10),
            TestError::TimeoutError { .. } => Duration::from_secs(2),
            _ => Duration::from_secs(1),
        };

        Some(base_delay)
    }

    /// Check if running in CI environment
    fn is_ci_environment(&self) -> bool {
        std::env::var("CI").is_ok()
            || std::env::var("GITHUB_ACTIONS").is_ok()
            || std::env::var("GITLAB_CI").is_ok()
            || std::env::var("JENKINS_URL").is_ok()
    }

    /// Capture current system resources
    async fn capture_system_resources(&self) -> SystemResourceSnapshot {
        SystemResourceSnapshot {
            memory_usage_mb: self.get_memory_usage_mb(),
            cpu_usage_percent: self.get_cpu_usage_percent(),
            disk_usage_percent: self.get_disk_usage_percent(),
            network_active: self.is_network_active(),
        }
    }

    /// Get memory usage in MB
    fn get_memory_usage_mb(&self) -> u64 {
        // Simplified implementation - in production use proper system monitoring
        use super::utils::get_memory_usage;
        get_memory_usage() / (1024 * 1024) // Convert bytes to MB
    }

    /// Get CPU usage percentage
    fn get_cpu_usage_percent(&self) -> f64 {
        // Simplified implementation - in production use proper system monitoring
        50.0 // Placeholder
    }

    /// Get disk usage percentage
    fn get_disk_usage_percent(&self) -> f64 {
        // Simplified implementation - in production use proper system monitoring
        30.0 // Placeholder
    }

    /// Check if network is active
    fn is_network_active(&self) -> bool {
        // Simplified implementation - in production use proper network monitoring
        true // Placeholder
    }

    /// Collect relevant environment variables
    fn collect_relevant_env_vars(&self) -> HashMap<String, String> {
        let mut vars = HashMap::new();

        let relevant_vars = [
            "RUST_LOG",
            "RUST_BACKTRACE",
            "CI",
            "GITHUB_ACTIONS",
            "BITNET_TEST_LOG_LEVEL",
            "BITNET_TEST_CACHE",
            "PATH",
        ];

        for var in &relevant_vars {
            if let Ok(value) = std::env::var(var) {
                vars.insert(var.to_string(), value);
            }
        }

        vars
    }

    /// Send error notification (placeholder implementation)
    async fn send_error_notification(
        &self,
        test_name: &str,
        error: &TestError,
        analysis: &ErrorAnalysis,
    ) -> Result<(), TestError> {
        // In a real implementation, this would send notifications via:
        // - Slack/Discord webhooks
        // - Email
        // - GitHub issue creation
        // - PagerDuty alerts

        info!(
            "Error notification for test '{}': {} ({})",
            test_name, error, analysis.debugging_priority
        );

        Ok(())
    }

    /// Get all error reports
    pub async fn get_error_reports(&self) -> Vec<ErrorReport> {
        self.error_reports.read().await.clone()
    }

    /// Clear error reports
    pub async fn clear_error_reports(&self) {
        let mut reports = self.error_reports.write().await;
        reports.clear();
    }

    /// Generate error summary report
    pub async fn generate_error_summary(&self) -> ErrorSummaryReport {
        let reports = self.error_reports.read().await;

        let total_errors = reports.len();
        let mut by_category = HashMap::new();
        let mut by_severity = HashMap::new();

        for report in reports.iter() {
            *by_category.entry(report.error_category.clone()).or_insert(0) += 1;
            *by_severity.entry(report.severity).or_insert(0) += 1;
        }

        let critical_errors = by_severity.get(&ErrorSeverity::Critical).unwrap_or(&0);
        let high_errors = by_severity.get(&ErrorSeverity::High).unwrap_or(&0);

        ErrorSummaryReport {
            total_errors,
            critical_errors: *critical_errors,
            high_severity_errors: *high_errors,
            errors_by_category: by_category,
            errors_by_severity: by_severity,
            most_common_category: reports
                .iter()
                .map(|r| &r.error_category)
                .max_by_key(|category| {
                    reports.iter().filter(|r| &r.error_category == *category).count()
                })
                .cloned()
                .unwrap_or_default(),
        }
    }
}

/// Configuration for the enhanced error handler
#[derive(Debug, Clone)]
pub struct ErrorHandlerConfig {
    pub save_analysis_reports: bool,
    pub analysis_output_dir: PathBuf,
    pub log_debugging_guides: bool,
    pub send_notifications: bool,
    pub enable_retries: bool,
    pub max_retry_attempts: usize,
}

impl Default for ErrorHandlerConfig {
    fn default() -> Self {
        Self {
            save_analysis_reports: true,
            analysis_output_dir: PathBuf::from("test-output/error-analysis"),
            log_debugging_guides: true,
            send_notifications: false,
            enable_retries: true,
            max_retry_attempts: 3,
        }
    }
}

/// Context information about test execution
#[derive(Debug, Clone)]
pub struct TestExecutionContext {
    pub test_suite: String,
    pub execution_time: Duration,
    pub concurrent_tests: usize,
    pub retry_count: usize,
    pub test_metadata: HashMap<String, String>,
}

/// Result of error handling with actionable information
#[derive(Debug, Clone)]
pub struct ErrorHandlingResult {
    pub error_report: ErrorReport,
    pub analysis: ErrorAnalysis,
    pub debugging_guide: String,
    pub recovery_suggestions: Vec<String>,
    pub troubleshooting_steps: Vec<super::errors::TroubleshootingStep>,
    pub should_retry: bool,
    pub retry_delay: Option<Duration>,
}

impl ErrorHandlingResult {
    /// Check if this error should block the test suite
    pub fn should_block_suite(&self) -> bool {
        matches!(self.analysis.severity, ErrorSeverity::Critical | ErrorSeverity::High)
            && matches!(
                self.analysis.debugging_priority,
                super::error_analysis::DebuggingPriority::Critical
                    | super::error_analysis::DebuggingPriority::High
            )
    }

    /// Get the most important recommendation
    pub fn primary_recommendation(
        &self,
    ) -> Option<&super::error_analysis::ActionableRecommendation> {
        self.analysis.recommendations.first()
    }

    /// Generate a concise error summary for logging
    pub fn generate_summary(&self) -> String {
        format!(
            "{} error in {} - {} ({}% confidence)",
            self.analysis.severity,
            self.analysis.error_category,
            self.analysis.error_summary,
            (self.analysis.confidence_score * 100.0) as u32
        )
    }
}

/// Summary report of all errors
#[derive(Debug, Clone)]
pub struct ErrorSummaryReport {
    pub total_errors: usize,
    pub critical_errors: usize,
    pub high_severity_errors: usize,
    pub errors_by_category: HashMap<String, usize>,
    pub errors_by_severity: HashMap<ErrorSeverity, usize>,
    pub most_common_category: String,
}

impl ErrorSummaryReport {
    /// Generate a human-readable summary
    pub fn generate_summary(&self) -> String {
        format!(
            "Error Summary: {} total errors, {} critical, {} high severity. Most common: {}",
            self.total_errors,
            self.critical_errors,
            self.high_severity_errors,
            self.most_common_category
        )
    }

    /// Check if error levels are concerning
    pub fn is_concerning(&self) -> bool {
        self.critical_errors > 0 || self.high_severity_errors > 5
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn create_test_execution_context() -> TestExecutionContext {
        TestExecutionContext {
            test_suite: "unit_tests".to_string(),
            execution_time: Duration::from_secs(30),
            concurrent_tests: 4,
            retry_count: 0,
            test_metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_should_retry_logic() {
        let config = ErrorHandlerConfig::default();
        let logging_manager = Arc::new(
            super::super::logging::LoggingManager::new(super::super::config::TestConfig::default())
                .unwrap(),
        );
        let handler = EnhancedErrorHandler::new(logging_manager, config);

        let timeout_error = TestError::timeout(Duration::from_secs(30));
        let assertion_error = TestError::assertion("Test failed");
        let context = create_test_execution_context();

        assert!(handler.should_retry_test(&timeout_error, &context));
        assert!(!handler.should_retry_test(&assertion_error, &context));
    }

    #[test]
    fn test_retry_delay_calculation() {
        let config = ErrorHandlerConfig::default();
        let logging_manager = Arc::new(
            super::super::logging::LoggingManager::new(super::super::config::TestConfig::default())
                .unwrap(),
        );
        let handler = EnhancedErrorHandler::new(logging_manager, config);

        let timeout_error = TestError::timeout(Duration::from_secs(30));
        let assertion_error = TestError::assertion("Test failed");

        assert!(handler.calculate_retry_delay(&timeout_error).is_some());
        assert!(handler.calculate_retry_delay(&assertion_error).is_none());
    }

    #[test]
    fn test_error_handling_result() {
        let error_report = super::super::errors::ErrorReport {
            timestamp: SystemTime::now(),
            error_message: "Test error".to_string(),
            error_category: "test".to_string(),
            severity: ErrorSeverity::Medium,
            debug_info: super::super::errors::ErrorDebugInfo {
                error_type: "TestError".to_string(),
                category: "test".to_string(),
                severity: ErrorSeverity::Medium,
                recoverable: true,
                recovery_suggestions: Vec::new(),
                related_components: Vec::new(),
                troubleshooting_steps: Vec::new(),
            },
            stack_trace: None,
            environment_info: super::super::errors::collect_environment_info(),
        };

        let analysis = ErrorAnalysis {
            timestamp: SystemTime::now(),
            error_summary: "Test error".to_string(),
            error_category: "test".to_string(),
            severity: ErrorSeverity::Medium,
            context: super::super::error_analysis::ErrorContext {
                test_name: "test".to_string(),
                test_suite: "suite".to_string(),
                execution_time: Duration::from_secs(1),
                concurrent_tests: 1,
                is_ci_environment: false,
                system_resources: SystemResourceSnapshot {
                    memory_usage_mb: 100,
                    cpu_usage_percent: 50.0,
                    disk_usage_percent: 30.0,
                    network_active: true,
                },
                environment_variables: HashMap::new(),
            },
            detected_patterns: Vec::new(),
            root_cause_analysis: super::super::error_analysis::RootCauseAnalysis {
                primary_cause: None,
                alternative_causes: Vec::new(),
                analysis_confidence: 0.5,
            },
            recommendations: Vec::new(),
            similar_errors: Vec::new(),
            confidence_score: 0.5,
            debugging_priority: super::super::error_analysis::DebuggingPriority::Medium,
        };

        let result = ErrorHandlingResult {
            error_report,
            analysis,
            debugging_guide: "Test guide".to_string(),
            recovery_suggestions: Vec::new(),
            troubleshooting_steps: Vec::new(),
            should_retry: false,
            retry_delay: None,
        };

        assert!(!result.should_block_suite());
        assert!(result.primary_recommendation().is_none());
        assert!(result.generate_summary().contains("MEDIUM"));
    }
}
