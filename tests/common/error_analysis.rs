use serde::{Deserialize, Serialize};
/// Error analysis utilities for actionable debugging information
///
/// This module provides comprehensive error analysis capabilities including:
/// - Error pattern detection and classification
/// - Root cause analysis suggestions
/// - Error correlation and trend analysis
/// - Automated debugging recommendations
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use super::errors::{ErrorReport, ErrorSeverity, TestError, TroubleshootingStep};

/// Error analyzer that provides actionable debugging information
pub struct ErrorAnalyzer {
    error_patterns: Vec<ErrorPattern>,
    error_history: Vec<ErrorHistoryEntry>,
    analysis_config: AnalysisConfig,
}

impl ErrorAnalyzer {
    /// Create a new error analyzer
    pub fn new() -> Self {
        Self {
            error_patterns: Self::load_default_patterns(),
            error_history: Vec::new(),
            analysis_config: AnalysisConfig::default(),
        }
    }

    /// Analyze an error and provide actionable debugging information
    pub async fn analyze_error(
        &mut self,
        error: &TestError,
        context: ErrorContext,
    ) -> ErrorAnalysis {
        // Record error in history
        self.record_error(error, &context).await;

        // Detect patterns
        let detected_patterns = self.detect_patterns(error, &context);

        // Perform root cause analysis
        let root_cause_analysis = self.analyze_root_cause(error, &context, &detected_patterns);

        // Generate recommendations
        let recommendations = self.generate_recommendations(error, &context, &detected_patterns);

        // Check for similar errors in history
        let similar_errors = self.find_similar_errors(error, &context);

        // Calculate confidence score
        let confidence_score = self.calculate_confidence_score(&detected_patterns, &similar_errors);

        ErrorAnalysis {
            timestamp: SystemTime::now(),
            error_summary: error.to_string(),
            error_category: error.category().to_string(),
            severity: error.severity(),
            context: context.clone(),
            detected_patterns,
            root_cause_analysis,
            recommendations,
            similar_errors,
            confidence_score,
            debugging_priority: self.calculate_debugging_priority(error, &context),
        }
    }

    /// Record an error in the history for pattern analysis
    async fn record_error(&mut self, error: &TestError, context: &ErrorContext) {
        let entry = ErrorHistoryEntry {
            timestamp: SystemTime::now(),
            error_type: std::any::type_name::<TestError>().to_string(),
            error_message: error.to_string(),
            error_category: error.category().to_string(),
            severity: error.severity(),
            context: context.clone(),
            resolved: false,
            resolution_time: None,
            resolution_method: None,
        };

        self.error_history.push(entry);

        // Keep history size manageable
        if self.error_history.len() > self.analysis_config.max_history_size {
            self.error_history.remove(0);
        }
    }

    /// Detect error patterns based on known patterns and history
    fn detect_patterns(&self, error: &TestError, context: &ErrorContext) -> Vec<DetectedPattern> {
        let mut detected = Vec::new();

        for pattern in &self.error_patterns {
            if pattern.matches(error, context) {
                let confidence = pattern.calculate_confidence(error, context);
                detected.push(DetectedPattern {
                    pattern: pattern.clone(),
                    confidence,
                    evidence: pattern.collect_evidence(error, context),
                });
            }
        }

        // Sort by confidence
        detected.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        detected
    }

    /// Perform root cause analysis
    fn analyze_root_cause(
        &self,
        error: &TestError,
        context: &ErrorContext,
        patterns: &[DetectedPattern],
    ) -> RootCauseAnalysis {
        let mut potential_causes = Vec::new();

        // Analyze based on error type
        match error {
            TestError::TimeoutError { timeout } => {
                potential_causes.push(PotentialCause {
                    cause: "Resource contention".to_string(),
                    likelihood: 0.7,
                    evidence: vec![
                        format!("Test timed out after {:?}", timeout),
                        "Multiple tests may be competing for resources".to_string(),
                    ],
                    investigation_steps: vec![
                        "Check system resource usage during test execution".to_string(),
                        "Run test in isolation to verify timing".to_string(),
                        "Monitor CPU and memory usage patterns".to_string(),
                    ],
                });

                if context.test_name.contains("parallel") || context.concurrent_tests > 1 {
                    potential_causes.push(PotentialCause {
                        cause: "Parallel execution interference".to_string(),
                        likelihood: 0.8,
                        evidence: vec![
                            format!(
                                "Test running with {} concurrent tests",
                                context.concurrent_tests
                            ),
                            "Parallel tests may be interfering with each other".to_string(),
                        ],
                        investigation_steps: vec![
                            "Run test with reduced parallelism".to_string(),
                            "Check for shared resource conflicts".to_string(),
                        ],
                    });
                }
            }
            TestError::FixtureError(fixture_err) => {
                potential_causes.push(PotentialCause {
                    cause: "Network connectivity issues".to_string(),
                    likelihood: 0.6,
                    evidence: vec![
                        "Fixture operation failed".to_string(),
                        fixture_err.to_string(),
                    ],
                    investigation_steps: vec![
                        "Test network connectivity".to_string(),
                        "Check proxy and firewall settings".to_string(),
                        "Verify fixture URLs are accessible".to_string(),
                    ],
                });
            }
            TestError::AssertionError { message } => {
                potential_causes.push(PotentialCause {
                    cause: "Logic error or changed behavior".to_string(),
                    likelihood: 0.8,
                    evidence: vec![
                        format!("Assertion failed: {}", message),
                        "Expected behavior doesn't match actual behavior".to_string(),
                    ],
                    investigation_steps: vec![
                        "Review recent code changes".to_string(),
                        "Compare expected vs actual values".to_string(),
                        "Check if test expectations need updating".to_string(),
                    ],
                });
            }
            _ => {}
        }

        // Add causes from detected patterns
        for pattern in patterns {
            if let Some(cause) = &pattern.pattern.root_cause {
                potential_causes.push(cause.clone());
            }
        }

        // Sort by likelihood
        potential_causes.sort_by(|a, b| b.likelihood.partial_cmp(&a.likelihood).unwrap());

        let analysis_confidence = self.calculate_analysis_confidence(&potential_causes);
        let primary_cause = potential_causes.first().cloned();
        let alternative_causes = potential_causes.into_iter().skip(1).take(3).collect();

        RootCauseAnalysis {
            primary_cause,
            alternative_causes,
            analysis_confidence,
        }
    }

    /// Generate actionable recommendations
    fn generate_recommendations(
        &self,
        error: &TestError,
        context: &ErrorContext,
        patterns: &[DetectedPattern],
    ) -> Vec<ActionableRecommendation> {
        let mut recommendations = Vec::new();

        // Add basic error-specific recommendations
        let basic_suggestions = error.recovery_suggestions();
        for (i, suggestion) in basic_suggestions.into_iter().enumerate() {
            recommendations.push(ActionableRecommendation {
                priority: RecommendationPriority::Medium,
                category: "Basic Recovery".to_string(),
                title: format!("Recovery Step {}", i + 1),
                description: suggestion,
                estimated_effort: EstimatedEffort::Low,
                success_probability: 0.6,
                prerequisites: Vec::new(),
                commands: Vec::new(),
                verification_steps: Vec::new(),
            });
        }

        // Add pattern-based recommendations
        for pattern in patterns {
            if let Some(rec) = &pattern.pattern.recommendation {
                recommendations.push(rec.clone());
            }
        }

        // Add context-specific recommendations
        if context.is_ci_environment {
            recommendations.push(ActionableRecommendation {
                priority: RecommendationPriority::High,
                category: "CI Environment".to_string(),
                title: "Check CI-specific issues".to_string(),
                description:
                    "This error occurred in CI environment, check for CI-specific problems"
                        .to_string(),
                estimated_effort: EstimatedEffort::Medium,
                success_probability: 0.7,
                prerequisites: vec!["Access to CI logs".to_string()],
                commands: vec![
                    "Review CI environment variables".to_string(),
                    "Check CI resource limits".to_string(),
                ],
                verification_steps: vec!["Run test locally to compare behavior".to_string()],
            });
        }

        // Sort by priority and success probability
        recommendations.sort_by(|a, b| {
            b.priority.cmp(&a.priority).then_with(|| {
                b.success_probability
                    .partial_cmp(&a.success_probability)
                    .unwrap()
            })
        });

        recommendations
    }

    /// Find similar errors in history
    fn find_similar_errors(&self, error: &TestError, context: &ErrorContext) -> Vec<SimilarError> {
        let mut similar = Vec::new();

        for entry in &self.error_history {
            let similarity = self.calculate_similarity(error, context, entry);
            if similarity > 0.5 {
                similar.push(SimilarError {
                    error_entry: entry.clone(),
                    similarity_score: similarity,
                    key_similarities: self.identify_similarities(error, context, entry),
                });
            }
        }

        // Sort by similarity
        similar.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap());
        similar.into_iter().take(5).collect() // Return top 5 similar errors
    }

    /// Calculate similarity between current error and historical error
    fn calculate_similarity(
        &self,
        error: &TestError,
        context: &ErrorContext,
        entry: &ErrorHistoryEntry,
    ) -> f64 {
        let mut score = 0.0;
        let mut factors = 0;

        // Category similarity
        if error.category() == entry.error_category {
            score += 0.3;
        }
        factors += 1;

        // Severity similarity
        if error.severity() == entry.severity {
            score += 0.2;
        }
        factors += 1;

        // Test name similarity
        if context.test_name == entry.context.test_name {
            score += 0.4;
        } else if context.test_name.contains(&entry.context.test_name)
            || entry.context.test_name.contains(&context.test_name)
        {
            score += 0.2;
        }
        factors += 1;

        // Environment similarity
        if context.is_ci_environment == entry.context.is_ci_environment {
            score += 0.1;
        }
        factors += 1;

        score / factors as f64
    }

    /// Identify key similarities between errors
    fn identify_similarities(
        &self,
        error: &TestError,
        context: &ErrorContext,
        entry: &ErrorHistoryEntry,
    ) -> Vec<String> {
        let mut similarities = Vec::new();

        if error.category() == entry.error_category {
            similarities.push(format!("Same error category: {}", error.category()));
        }

        if error.severity() == entry.severity {
            similarities.push(format!("Same severity level: {}", error.severity()));
        }

        if context.test_name == entry.context.test_name {
            similarities.push(format!("Same test: {}", context.test_name));
        }

        if context.is_ci_environment == entry.context.is_ci_environment {
            let env = if context.is_ci_environment {
                "CI"
            } else {
                "local"
            };
            similarities.push(format!("Same environment: {}", env));
        }

        similarities
    }

    /// Calculate confidence score for the analysis
    fn calculate_confidence_score(
        &self,
        patterns: &[DetectedPattern],
        similar_errors: &[SimilarError],
    ) -> f64 {
        let mut confidence = 0.5; // Base confidence

        // Increase confidence based on detected patterns
        if !patterns.is_empty() {
            let avg_pattern_confidence: f64 =
                patterns.iter().map(|p| p.confidence).sum::<f64>() / patterns.len() as f64;
            confidence += avg_pattern_confidence * 0.3;
        }

        // Increase confidence based on similar errors
        if !similar_errors.is_empty() {
            let resolved_similar = similar_errors
                .iter()
                .filter(|e| e.error_entry.resolved)
                .count();
            if resolved_similar > 0 {
                confidence += (resolved_similar as f64 / similar_errors.len() as f64) * 0.2;
            }
        }

        confidence.min(1.0)
    }

    /// Calculate debugging priority
    fn calculate_debugging_priority(
        &self,
        error: &TestError,
        context: &ErrorContext,
    ) -> DebuggingPriority {
        let mut score = 0;

        // Severity contributes to priority
        score += match error.severity() {
            ErrorSeverity::Critical => 4,
            ErrorSeverity::High => 3,
            ErrorSeverity::Medium => 2,
            ErrorSeverity::Low => 1,
        };

        // CI environment increases priority
        if context.is_ci_environment {
            score += 2;
        }

        // Frequent failures increase priority
        let similar_count = self
            .error_history
            .iter()
            .filter(|e| e.error_category == error.category())
            .count();
        if similar_count > 3 {
            score += 2;
        }

        match score {
            0..=2 => DebuggingPriority::Low,
            3..=4 => DebuggingPriority::Medium,
            5..=6 => DebuggingPriority::High,
            _ => DebuggingPriority::Critical,
        }
    }

    /// Calculate analysis confidence
    fn calculate_analysis_confidence(&self, causes: &[PotentialCause]) -> f64 {
        if causes.is_empty() {
            return 0.0;
        }

        let avg_likelihood: f64 =
            causes.iter().map(|c| c.likelihood).sum::<f64>() / causes.len() as f64;
        avg_likelihood
    }

    /// Load default error patterns
    fn load_default_patterns() -> Vec<ErrorPattern> {
        vec![
            ErrorPattern {
                name: "Timeout in CI".to_string(),
                description: "Test timeouts occurring specifically in CI environment".to_string(),
                conditions: vec![
                    PatternCondition::ErrorCategory("timeout".to_string()),
                    PatternCondition::CIEnvironment(true),
                ],
                root_cause: Some(PotentialCause {
                    cause: "CI resource limitations".to_string(),
                    likelihood: 0.8,
                    evidence: vec![
                        "Timeout occurs in CI but not locally".to_string(),
                        "CI environments often have resource constraints".to_string(),
                    ],
                    investigation_steps: vec![
                        "Compare CI and local resource availability".to_string(),
                        "Check CI runner specifications".to_string(),
                        "Monitor CI resource usage during test execution".to_string(),
                    ],
                }),
                recommendation: Some(ActionableRecommendation {
                    priority: RecommendationPriority::High,
                    category: "CI Optimization".to_string(),
                    title: "Optimize for CI environment".to_string(),
                    description: "Adjust test configuration for CI resource constraints"
                        .to_string(),
                    estimated_effort: EstimatedEffort::Medium,
                    success_probability: 0.8,
                    prerequisites: vec!["CI configuration access".to_string()],
                    commands: vec![
                        "Increase test timeout for CI".to_string(),
                        "Reduce parallel test execution in CI".to_string(),
                    ],
                    verification_steps: vec!["Run tests in CI with new configuration".to_string()],
                }),
            },
            ErrorPattern {
                name: "Fixture Download Failure".to_string(),
                description: "Repeated failures downloading test fixtures".to_string(),
                conditions: vec![
                    PatternCondition::ErrorCategory("fixture".to_string()),
                    PatternCondition::ErrorMessageContains("download".to_string()),
                ],
                root_cause: Some(PotentialCause {
                    cause: "Network connectivity or fixture server issues".to_string(),
                    likelihood: 0.7,
                    evidence: vec![
                        "Fixture download consistently fails".to_string(),
                        "Network-dependent operation".to_string(),
                    ],
                    investigation_steps: vec![
                        "Test fixture URL accessibility".to_string(),
                        "Check network connectivity".to_string(),
                        "Verify fixture server status".to_string(),
                    ],
                }),
                recommendation: Some(ActionableRecommendation {
                    priority: RecommendationPriority::Medium,
                    category: "Fixture Management".to_string(),
                    title: "Implement fixture fallback".to_string(),
                    description: "Use cached fixtures or alternative sources when download fails"
                        .to_string(),
                    estimated_effort: EstimatedEffort::Low,
                    success_probability: 0.9,
                    prerequisites: vec!["Fixture cache available".to_string()],
                    commands: vec![
                        "Enable fixture cache fallback".to_string(),
                        "Configure alternative fixture sources".to_string(),
                    ],
                    verification_steps: vec!["Test with network disconnected".to_string()],
                }),
            },
        ]
    }
}

impl Default for ErrorAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for error analysis
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    pub max_history_size: usize,
    pub pattern_confidence_threshold: f64,
    pub similarity_threshold: f64,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            max_history_size: 1000,
            pattern_confidence_threshold: 0.7,
            similarity_threshold: 0.5,
        }
    }
}

/// Context information for error analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    pub test_name: String,
    pub test_suite: String,
    pub execution_time: Duration,
    pub concurrent_tests: usize,
    pub is_ci_environment: bool,
    pub system_resources: SystemResourceSnapshot,
    pub environment_variables: HashMap<String, String>,
}

/// Snapshot of system resources at error time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemResourceSnapshot {
    pub memory_usage_mb: u64,
    pub cpu_usage_percent: f64,
    pub disk_usage_percent: f64,
    pub network_active: bool,
}

/// Complete error analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorAnalysis {
    pub timestamp: SystemTime,
    pub error_summary: String,
    pub error_category: String,
    pub severity: ErrorSeverity,
    pub context: ErrorContext,
    pub detected_patterns: Vec<DetectedPattern>,
    pub root_cause_analysis: RootCauseAnalysis,
    pub recommendations: Vec<ActionableRecommendation>,
    pub similar_errors: Vec<SimilarError>,
    pub confidence_score: f64,
    pub debugging_priority: DebuggingPriority,
}

impl ErrorAnalysis {
    /// Generate a human-readable debugging guide
    pub fn generate_debugging_guide(&self) -> String {
        let mut guide = String::new();

        guide.push_str(&format!(
            "DEBUGGING GUIDE - {} Priority\n",
            self.debugging_priority
        ));
        guide.push_str("=".repeat(50).as_str());
        guide.push('\n');

        guide.push_str(&format!("Error: {}\n", self.error_summary));
        guide.push_str(&format!("Category: {}\n", self.error_category));
        guide.push_str(&format!("Severity: {}\n", self.severity));
        guide.push_str(&format!(
            "Confidence: {:.1}%\n\n",
            self.confidence_score * 100.0
        ));

        if let Some(primary_cause) = &self.root_cause_analysis.primary_cause {
            guide.push_str("PRIMARY CAUSE:\n");
            guide.push_str(&format!(
                "- {} ({}% likelihood)\n",
                primary_cause.cause,
                (primary_cause.likelihood * 100.0) as u32
            ));
            guide.push_str("  Evidence:\n");
            for evidence in &primary_cause.evidence {
                guide.push_str(&format!("  • {}\n", evidence));
            }
            guide.push('\n');
        }

        if !self.recommendations.is_empty() {
            guide.push_str("RECOMMENDED ACTIONS:\n");
            for (i, rec) in self.recommendations.iter().take(3).enumerate() {
                guide.push_str(&format!("{}. {} ({})\n", i + 1, rec.title, rec.priority));
                guide.push_str(&format!("   {}\n", rec.description));
                guide.push_str(&format!(
                    "   Effort: {} | Success Rate: {}%\n",
                    rec.estimated_effort,
                    (rec.success_probability * 100.0) as u32
                ));
                guide.push('\n');
            }
        }

        if !self.similar_errors.is_empty() {
            guide.push_str("SIMILAR PAST ERRORS:\n");
            for similar in self.similar_errors.iter().take(2) {
                guide.push_str(&format!(
                    "- {} ({}% similar)\n",
                    similar.error_entry.error_message,
                    (similar.similarity_score * 100.0) as u32
                ));
                if similar.error_entry.resolved {
                    guide.push_str("  ✓ Previously resolved\n");
                }
            }
        }

        guide
    }

    /// Save analysis to file
    pub async fn save_to_file(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        tokio::fs::write(path, json).await
    }
}

/// Error pattern for detection
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ErrorPattern {
    pub name: String,
    pub description: String,
    pub conditions: Vec<PatternCondition>,
    pub root_cause: Option<PotentialCause>,
    pub recommendation: Option<ActionableRecommendation>,
}

impl ErrorPattern {
    /// Check if this pattern matches the given error and context
    pub fn matches(&self, error: &TestError, context: &ErrorContext) -> bool {
        self.conditions
            .iter()
            .all(|condition| condition.matches(error, context))
    }

    /// Calculate confidence for this pattern match
    pub fn calculate_confidence(&self, _error: &TestError, _context: &ErrorContext) -> f64 {
        // Simple implementation - in practice this would be more sophisticated
        0.8
    }

    /// Collect evidence for this pattern
    pub fn collect_evidence(&self, error: &TestError, context: &ErrorContext) -> Vec<String> {
        let mut evidence = Vec::new();

        for condition in &self.conditions {
            if let Some(desc) = condition.describe_match(error, context) {
                evidence.push(desc);
            }
        }

        evidence
    }
}

/// Condition for pattern matching
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum PatternCondition {
    ErrorCategory(String),
    ErrorMessageContains(String),
    CIEnvironment(bool),
    ConcurrentTests(usize),
    TestNameContains(String),
}

impl PatternCondition {
    /// Check if this condition matches
    pub fn matches(&self, error: &TestError, context: &ErrorContext) -> bool {
        match self {
            Self::ErrorCategory(category) => error.category() == category,
            Self::ErrorMessageContains(text) => error.to_string().contains(text),
            Self::CIEnvironment(is_ci) => context.is_ci_environment == *is_ci,
            Self::ConcurrentTests(min_count) => context.concurrent_tests >= *min_count,
            Self::TestNameContains(text) => context.test_name.contains(text),
        }
    }

    /// Describe the match for evidence collection
    pub fn describe_match(&self, error: &TestError, context: &ErrorContext) -> Option<String> {
        if !self.matches(error, context) {
            return None;
        }

        match self {
            Self::ErrorCategory(category) => Some(format!("Error category is '{}'", category)),
            Self::ErrorMessageContains(text) => Some(format!("Error message contains '{}'", text)),
            Self::CIEnvironment(true) => Some("Running in CI environment".to_string()),
            Self::CIEnvironment(false) => Some("Running in local environment".to_string()),
            Self::ConcurrentTests(count) => Some(format!("Running {} concurrent tests", count)),
            Self::TestNameContains(text) => Some(format!("Test name contains '{}'", text)),
        }
    }
}

/// Detected pattern with confidence and evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedPattern {
    pub pattern: ErrorPattern,
    pub confidence: f64,
    pub evidence: Vec<String>,
}

/// Root cause analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootCauseAnalysis {
    pub primary_cause: Option<PotentialCause>,
    pub alternative_causes: Vec<PotentialCause>,
    pub analysis_confidence: f64,
}

/// Potential cause of an error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PotentialCause {
    pub cause: String,
    pub likelihood: f64,
    pub evidence: Vec<String>,
    pub investigation_steps: Vec<String>,
}

/// Actionable recommendation for fixing an error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionableRecommendation {
    pub priority: RecommendationPriority,
    pub category: String,
    pub title: String,
    pub description: String,
    pub estimated_effort: EstimatedEffort,
    pub success_probability: f64,
    pub prerequisites: Vec<String>,
    pub commands: Vec<String>,
    pub verification_steps: Vec<String>,
}

/// Priority level for recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

impl std::fmt::Display for RecommendationPriority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "Low Priority"),
            Self::Medium => write!(f, "Medium Priority"),
            Self::High => write!(f, "High Priority"),
            Self::Critical => write!(f, "Critical Priority"),
        }
    }
}

/// Estimated effort for implementing a recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EstimatedEffort {
    Low,    // < 30 minutes
    Medium, // 30 minutes - 2 hours
    High,   // 2+ hours
}

impl std::fmt::Display for EstimatedEffort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "Low Effort"),
            Self::Medium => write!(f, "Medium Effort"),
            Self::High => write!(f, "High Effort"),
        }
    }
}

/// Similar error from history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarError {
    pub error_entry: ErrorHistoryEntry,
    pub similarity_score: f64,
    pub key_similarities: Vec<String>,
}

/// Historical error entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorHistoryEntry {
    pub timestamp: SystemTime,
    pub error_type: String,
    pub error_message: String,
    pub error_category: String,
    pub severity: ErrorSeverity,
    pub context: ErrorContext,
    pub resolved: bool,
    pub resolution_time: Option<Duration>,
    pub resolution_method: Option<String>,
}

/// Debugging priority level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DebuggingPriority {
    Low,
    Medium,
    High,
    Critical,
}

impl std::fmt::Display for DebuggingPriority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "Low"),
            Self::Medium => write!(f, "Medium"),
            Self::High => write!(f, "High"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn create_test_context() -> ErrorContext {
        ErrorContext {
            test_name: "test_example".to_string(),
            test_suite: "unit_tests".to_string(),
            execution_time: Duration::from_secs(30),
            concurrent_tests: 4,
            is_ci_environment: false,
            system_resources: SystemResourceSnapshot {
                memory_usage_mb: 512,
                cpu_usage_percent: 75.0,
                disk_usage_percent: 45.0,
                network_active: true,
            },
            environment_variables: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_error_analysis() {
        let mut analyzer = ErrorAnalyzer::new();
        let error = TestError::timeout(Duration::from_secs(30));
        let context = create_test_context();

        let analysis = analyzer.analyze_error(&error, context).await;

        assert_eq!(analysis.error_category, "timeout");
        assert_eq!(analysis.severity, ErrorSeverity::Medium);
        assert!(!analysis.recommendations.is_empty());
    }

    #[test]
    fn test_pattern_matching() {
        let pattern = ErrorPattern {
            name: "Test Pattern".to_string(),
            description: "Test pattern for unit tests".to_string(),
            conditions: vec![
                PatternCondition::ErrorCategory("timeout".to_string()),
                PatternCondition::CIEnvironment(false),
            ],
            root_cause: None,
            recommendation: None,
        };

        let error = TestError::timeout(Duration::from_secs(30));
        let context = create_test_context();

        assert!(pattern.matches(&error, &context));
    }

    #[test]
    fn test_debugging_guide_generation() {
        let analysis = ErrorAnalysis {
            timestamp: SystemTime::now(),
            error_summary: "Test timeout error".to_string(),
            error_category: "timeout".to_string(),
            severity: ErrorSeverity::Medium,
            context: create_test_context(),
            detected_patterns: Vec::new(),
            root_cause_analysis: RootCauseAnalysis {
                primary_cause: Some(PotentialCause {
                    cause: "Resource contention".to_string(),
                    likelihood: 0.8,
                    evidence: vec!["High CPU usage detected".to_string()],
                    investigation_steps: vec!["Monitor system resources".to_string()],
                }),
                alternative_causes: Vec::new(),
                analysis_confidence: 0.8,
            },
            recommendations: vec![ActionableRecommendation {
                priority: RecommendationPriority::High,
                category: "Performance".to_string(),
                title: "Reduce resource usage".to_string(),
                description: "Optimize test to use fewer resources".to_string(),
                estimated_effort: EstimatedEffort::Medium,
                success_probability: 0.7,
                prerequisites: Vec::new(),
                commands: Vec::new(),
                verification_steps: Vec::new(),
            }],
            similar_errors: Vec::new(),
            confidence_score: 0.8,
            debugging_priority: DebuggingPriority::Medium,
        };

        let guide = analysis.generate_debugging_guide();
        assert!(guide.contains("DEBUGGING GUIDE"));
        assert!(guide.contains("Resource contention"));
        assert!(guide.contains("Reduce resource usage"));
    }
}
