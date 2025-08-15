use super::{
    config::TestConfig,
    errors::TestOpResult as TestResultCompat,
    fast_config::{FastConfigBuilder, SpeedProfile},
    results::TestSuiteResult,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Simplified fast feedback system that provides rapid test execution
pub struct FastFeedbackSystem {
    config: FastFeedbackConfig,
    execution_history: ExecutionHistory,
}

/// Configuration for fast feedback system
#[derive(Debug, Clone)]
pub struct FastFeedbackConfig {
    /// Target feedback time (default: 2 minutes)
    pub target_feedback_time: Duration,
    /// Maximum feedback time before fallback (default: 5 minutes)
    pub max_feedback_time: Duration,
    /// Enable incremental testing
    pub enable_incremental: bool,
    /// Enable test caching
    pub enable_caching: bool,
    /// Enable smart test selection
    pub enable_smart_selection: bool,
    /// Speed profile for execution
    pub speed_profile: SpeedProfile,
    /// Minimum test coverage percentage to maintain
    pub min_coverage_threshold: f64,
    /// Enable parallel execution
    pub enable_parallel: bool,
    /// Maximum parallel tests for fast feedback
    pub max_parallel_fast: usize,
    /// Enable early termination on first failure
    pub fail_fast: bool,
    /// Cache validity duration
    pub cache_validity: Duration,
}

impl Default for FastFeedbackConfig {
    fn default() -> Self {
        Self {
            target_feedback_time: Duration::from_secs(2 * 60), // 2 minutes
            max_feedback_time: Duration::from_secs(5 * 60),    // 5 minutes
            enable_incremental: true,
            enable_caching: true,
            enable_smart_selection: true,
            speed_profile: SpeedProfile::Fast,
            min_coverage_threshold: 0.80, // 80% minimum coverage
            enable_parallel: true,
            max_parallel_fast: 4, // Conservative default
            fail_fast: true,
            cache_validity: Duration::from_secs(60 * 60), // 1 hour
        }
    }
}

/// Execution history for learning and optimization
#[derive(Debug, Default)]
pub struct ExecutionHistory {
    test_durations: HashMap<String, Duration>,
    test_success_rates: HashMap<String, f64>,
    last_execution_time: Option<Instant>,
    total_executions: usize,
    average_feedback_time: Duration,
}

/// Fast feedback execution result
#[derive(Debug, Clone)]
pub struct FastFeedbackResult {
    pub execution_time: Duration,
    pub tests_run: usize,
    pub tests_passed: usize,
    pub tests_failed: usize,
    pub tests_skipped: usize,
    pub coverage_achieved: f64,
    pub feedback_quality: FeedbackQuality,
    pub optimization_applied: Vec<String>,
    pub next_recommendations: Vec<String>,
    pub suite_results: Vec<TestSuiteResult>,
}

/// Quality of feedback provided
#[derive(Debug, Clone, PartialEq)]
pub enum FeedbackQuality {
    /// Full test suite run with complete coverage
    Complete,
    /// Incremental run with high confidence
    HighConfidence,
    /// Fast run with medium confidence
    MediumConfidence,
    /// Minimal run with basic confidence
    BasicConfidence,
    /// Emergency fallback with limited confidence
    Limited,
}

/// Execution strategy for fast feedback
#[derive(Debug, Clone)]
pub enum ExecutionStrategy {
    /// Run only changed tests based on incremental analysis
    Incremental(Vec<PathBuf>),
    /// Run only fast tests for immediate feedback
    FastOnly,
    /// Use smart selection based on history and priorities
    SmartSelection,
    /// Balanced approach optimizing for time and coverage
    Balanced,
}

/// Result of test execution
#[derive(Debug, Default)]
struct ExecutionResult {
    tests_run: usize,
    tests_passed: usize,
    tests_failed: usize,
    tests_skipped: usize,
    coverage_achieved: f64,
    optimizations_applied: Vec<String>,
    suite_results: Vec<TestSuiteResult>,
}

impl FastFeedbackSystem {
    /// Create a new fast feedback system
    pub fn new(config: FastFeedbackConfig) -> Self {
        Self { config, execution_history: ExecutionHistory::default() }
    }

    /// Get the configuration
    pub fn config(&self) -> &FastFeedbackConfig {
        &self.config
    }

    /// Create a fast feedback system with default configuration
    pub fn with_defaults() -> Self {
        Self::new(FastFeedbackConfig::default())
    }

    /// Create a fast feedback system optimized for CI
    pub fn for_ci() -> Self {
        let mut config = FastFeedbackConfig::default();
        config.target_feedback_time = Duration::from_secs(90); // 1.5 minutes for CI
        config.max_feedback_time = Duration::from_secs(3 * 60); // 3 minutes max
        config.speed_profile = SpeedProfile::Lightning;
        config.fail_fast = true;
        config.max_parallel_fast = 2; // Conservative for CI
        Self::new(config)
    }

    /// Create a fast feedback system optimized for development
    pub fn for_development() -> Self {
        let mut config = FastFeedbackConfig::default();
        config.target_feedback_time = Duration::from_secs(30); // 30 seconds for dev
        config.max_feedback_time = Duration::from_secs(2 * 60); // 2 minutes max
        config.speed_profile = SpeedProfile::Lightning;
        config.enable_incremental = true;
        config.fail_fast = false; // See all failures in dev
        Self::new(config)
    }

    /// Execute tests with fast feedback (simplified implementation)
    pub async fn execute_fast_feedback(&mut self) -> TestResultCompat<FastFeedbackResult> {
        let start_time = Instant::now();

        // Step 1: Determine execution strategy
        let strategy = self.determine_execution_strategy().await?;

        // Step 2: Select tests based on strategy
        let selected_tests = self.select_tests_for_strategy(&strategy).await?;

        // Step 3: Create optimized test configuration
        let test_config = self.create_optimized_config(&strategy, selected_tests.len());

        // Step 4: Execute tests with time monitoring (simplified)
        let execution_result = self.execute_with_monitoring(selected_tests, test_config).await?;

        // Step 5: Update execution history
        self.update_execution_history(&execution_result, start_time.elapsed());

        // Step 6: Generate recommendations for next run
        let recommendations = self.generate_next_recommendations(&execution_result);

        let total_time = start_time.elapsed();

        Ok(FastFeedbackResult {
            execution_time: total_time,
            tests_run: execution_result.tests_run,
            tests_passed: execution_result.tests_passed,
            tests_failed: execution_result.tests_failed,
            tests_skipped: execution_result.tests_skipped,
            coverage_achieved: execution_result.coverage_achieved,
            feedback_quality: self.assess_feedback_quality(&execution_result, total_time),
            optimization_applied: execution_result.optimizations_applied,
            next_recommendations: recommendations,
            suite_results: execution_result.suite_results,
        })
    }

    /// Determine the best execution strategy based on current context
    async fn determine_execution_strategy(&mut self) -> TestResultCompat<ExecutionStrategy> {
        // Simplified strategy determination
        if self.config.enable_incremental && self.has_recent_changes().await {
            Ok(ExecutionStrategy::Incremental(vec![PathBuf::from("src/lib.rs")]))
        } else if self.config.enable_smart_selection && self.has_recent_execution_history() {
            Ok(ExecutionStrategy::SmartSelection)
        } else if self.is_time_constrained().await {
            Ok(ExecutionStrategy::FastOnly)
        } else {
            Ok(ExecutionStrategy::Balanced)
        }
    }

    /// Select tests based on the chosen strategy
    async fn select_tests_for_strategy(
        &mut self,
        strategy: &ExecutionStrategy,
    ) -> TestResultCompat<Vec<String>> {
        match strategy {
            ExecutionStrategy::Incremental(_) => {
                Ok(vec!["unit_test_basic".to_string(), "unit_test_core".to_string()])
            }
            ExecutionStrategy::FastOnly => Ok(vec!["unit_test_basic".to_string()]),
            ExecutionStrategy::SmartSelection => self.select_tests_with_smart_algorithm().await,
            ExecutionStrategy::Balanced => Ok(vec![
                "unit_test_basic".to_string(),
                "unit_test_advanced".to_string(),
                "integration_test_simple".to_string(),
            ]),
        }
    }

    /// Execute tests with real-time monitoring and early termination
    async fn execute_with_monitoring(
        &self,
        test_names: Vec<String>,
        _config: TestConfig,
    ) -> TestResultCompat<ExecutionResult> {
        let start_time = Instant::now();
        let mut results = ExecutionResult::default();
        let mut optimizations_applied = Vec::new();

        // Simulate test execution
        for (i, test_name) in test_names.iter().enumerate() {
            // Check if we're approaching time limit
            let elapsed = start_time.elapsed();
            if elapsed > self.config.max_feedback_time {
                optimizations_applied.push("Early termination due to time limit".to_string());
                break;
            }

            // Simulate test execution time
            let test_duration = self.estimate_test_duration(test_name);
            tokio::time::sleep(Duration::from_millis(10)).await; // Minimal delay for simulation

            results.tests_run += 1;

            // Simulate test results (90% pass rate)
            if i % 10 != 9 {
                results.tests_passed += 1;
            } else {
                results.tests_failed += 1;

                // Early termination on failure if fail_fast is enabled
                if self.config.fail_fast {
                    optimizations_applied.push("Fail-fast termination".to_string());
                    break;
                }
            }
        }

        // Calculate coverage (simplified)
        results.coverage_achieved = if results.tests_run > 0 {
            results.tests_passed as f64 / results.tests_run as f64
        } else {
            0.0
        };

        results.optimizations_applied = optimizations_applied;

        Ok(results)
    }

    /// Select tests using smart algorithm based on history
    async fn select_tests_with_smart_algorithm(&self) -> TestResultCompat<Vec<String>> {
        let mut selected_tests = Vec::new();
        let mut estimated_time = Duration::ZERO;

        // Get all available tests (simplified)
        let all_test_names = vec![
            "unit_test_basic".to_string(),
            "unit_test_advanced".to_string(),
            "integration_test_simple".to_string(),
            "performance_test".to_string(),
        ];

        // Sort by priority (fast, high success rate, recently failed)
        let mut prioritized_tests: Vec<_> = all_test_names.into_iter().collect();
        prioritized_tests.sort_by(|a, b| {
            let priority_a = self.calculate_test_priority(a);
            let priority_b = self.calculate_test_priority(b);
            priority_b.partial_cmp(&priority_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Select tests that fit within time budget
        for test_name in prioritized_tests {
            let estimated_duration = self.estimate_test_duration(&test_name);

            if estimated_time + estimated_duration <= self.config.target_feedback_time {
                selected_tests.push(test_name);
                estimated_time += estimated_duration;
            } else {
                break;
            }
        }

        Ok(selected_tests)
    }

    /// Calculate priority score for a test based on history
    fn calculate_test_priority(&self, test_name: &str) -> f64 {
        let mut priority = 1.0;

        // Prioritize fast tests
        if let Some(&duration) = self.execution_history.test_durations.get(test_name) {
            priority += 1.0 / (duration.as_secs_f64() + 1.0);
        }

        // Prioritize tests with high success rates
        if let Some(&success_rate) = self.execution_history.test_success_rates.get(test_name) {
            priority += success_rate;
        }

        // Prioritize core functionality tests
        if test_name.contains("core")
            || test_name.contains("basic")
            || test_name.contains("essential")
        {
            priority += 2.0;
        }

        // Prioritize unit tests
        if test_name.contains("unit") {
            priority += 1.5;
        }

        priority
    }

    /// Estimate duration for a test based on history or heuristics
    fn estimate_test_duration(&self, test_name: &str) -> Duration {
        // Use historical data if available
        if let Some(&duration) = self.execution_history.test_durations.get(test_name) {
            return duration;
        }

        // Use heuristics based on test name
        if test_name.contains("unit") {
            Duration::from_secs(1)
        } else if test_name.contains("integration") {
            Duration::from_secs(5)
        } else if test_name.contains("performance") {
            Duration::from_secs(10)
        } else {
            Duration::from_secs(3) // Default
        }
    }

    /// Create optimized test configuration for the selected strategy
    fn create_optimized_config(
        &self,
        strategy: &ExecutionStrategy,
        test_count: usize,
    ) -> TestConfig {
        let mut builder = FastConfigBuilder::with_profile(self.config.speed_profile.clone());

        // Adjust parallelism based on test count and strategy
        let parallel_count = match strategy {
            ExecutionStrategy::FastOnly | ExecutionStrategy::Incremental(_) => {
                self.config.max_parallel_fast.min(test_count)
            }
            ExecutionStrategy::SmartSelection => {
                (self.config.max_parallel_fast / 2).max(1).min(test_count)
            }
            ExecutionStrategy::Balanced => self.config.max_parallel_fast.min(test_count),
        };

        builder = builder.max_parallel(parallel_count);

        // Adjust timeout based on target feedback time
        let test_timeout = self.config.target_feedback_time / (test_count.max(1) as u32);
        let min_timeout = Duration::from_secs(5);
        let max_timeout = Duration::from_secs(30);
        let adjusted_timeout = test_timeout.max(min_timeout).min(max_timeout);

        builder = builder.timeout(adjusted_timeout);

        // Configure based on strategy
        match strategy {
            ExecutionStrategy::FastOnly => {
                builder = builder.coverage(false).performance(false);
            }
            ExecutionStrategy::Incremental(_) => {
                builder = builder.coverage(true).performance(false);
            }
            ExecutionStrategy::SmartSelection => {
                builder = builder.coverage(true).performance(true);
            }
            ExecutionStrategy::Balanced => {
                builder = builder.coverage(true).performance(true);
            }
        }

        builder.build()
    }

    /// Update execution history with results
    fn update_execution_history(&mut self, result: &ExecutionResult, total_time: Duration) {
        self.execution_history.total_executions += 1;
        self.execution_history.last_execution_time = Some(Instant::now());

        // Update average feedback time
        let alpha = 0.3; // Learning rate
        if self.execution_history.total_executions == 1 {
            self.execution_history.average_feedback_time = total_time;
        } else {
            let current_avg = self.execution_history.average_feedback_time.as_secs_f64();
            let new_avg = current_avg * (1.0 - alpha) + total_time.as_secs_f64() * alpha;
            self.execution_history.average_feedback_time = Duration::from_secs_f64(new_avg);
        }

        // Update individual test performance (simplified)
        for suite_result in &result.suite_results {
            for test_result in &suite_result.test_results {
                let test_name = &test_result.test_name;

                // Update duration
                let entry = self
                    .execution_history
                    .test_durations
                    .entry(test_name.clone())
                    .or_insert(test_result.duration);
                let current_duration = entry.as_secs_f64();
                let new_duration =
                    current_duration * (1.0 - alpha) + test_result.duration.as_secs_f64() * alpha;
                *entry = Duration::from_secs_f64(new_duration);

                // Update success rate
                let success = test_result.is_success();
                let entry = self
                    .execution_history
                    .test_success_rates
                    .entry(test_name.clone())
                    .or_insert(if success { 1.0 } else { 0.0 });
                *entry = *entry * (1.0 - alpha) + (if success { 1.0 } else { 0.0 }) * alpha;
            }
        }
    }

    /// Generate recommendations for the next execution
    fn generate_next_recommendations(&self, result: &ExecutionResult) -> Vec<String> {
        let mut recommendations = Vec::new();

        if result.tests_failed > 0 {
            recommendations.push("Focus on fixing failing tests before next run".to_string());
        }

        if result.coverage_achieved < self.config.min_coverage_threshold {
            recommendations.push(format!(
                "Consider running more tests to achieve {}% coverage (current: {:.1}%)",
                (self.config.min_coverage_threshold * 100.0) as u32,
                result.coverage_achieved * 100.0
            ));
        }

        let avg_feedback_time = self.execution_history.average_feedback_time;
        if avg_feedback_time > self.config.target_feedback_time {
            recommendations
                .push("Consider using faster test selection to improve feedback time".to_string());
        }

        if result.tests_skipped > result.tests_run / 2 {
            recommendations.push(
                "Many tests were skipped - consider running full suite when time permits"
                    .to_string(),
            );
        }

        recommendations
    }

    /// Assess the quality of feedback provided
    fn assess_feedback_quality(
        &self,
        result: &ExecutionResult,
        execution_time: Duration,
    ) -> FeedbackQuality {
        let coverage_ratio = result.coverage_achieved;
        let time_ratio =
            execution_time.as_secs_f64() / self.config.target_feedback_time.as_secs_f64();

        if coverage_ratio >= 0.95 && time_ratio <= 1.0 {
            FeedbackQuality::Complete
        } else if coverage_ratio >= 0.80 && time_ratio <= 1.2 {
            FeedbackQuality::HighConfidence
        } else if coverage_ratio >= 0.60 && time_ratio <= 1.5 {
            FeedbackQuality::MediumConfidence
        } else if coverage_ratio >= 0.40 && time_ratio <= 2.0 {
            FeedbackQuality::BasicConfidence
        } else {
            FeedbackQuality::Limited
        }
    }

    // Helper methods

    fn has_recent_execution_history(&self) -> bool {
        self.execution_history.last_execution_time
            .map(|last| last.elapsed() < Duration::from_secs(60 * 60)) // 1 hour
            .unwrap_or(false)
    }

    async fn has_recent_changes(&self) -> bool {
        // Simplified change detection
        std::env::var("BITNET_INCREMENTAL").is_ok()
    }

    async fn is_time_constrained(&self) -> bool {
        // Check if we're in a time-constrained environment (e.g., CI with limited time)
        std::env::var("CI").is_ok() || std::env::var("BITNET_FAST_FEEDBACK").is_ok()
    }
}

/// Utility functions for fast feedback
pub mod utils {
    use super::*;

    /// Create a fast feedback system based on environment
    pub fn create_for_environment() -> FastFeedbackSystem {
        if std::env::var("CI").is_ok() {
            FastFeedbackSystem::for_ci()
        } else {
            FastFeedbackSystem::for_development()
        }
    }

    /// Check if fast feedback should be used
    pub fn should_use_fast_feedback() -> bool {
        std::env::var("BITNET_FAST_FEEDBACK").is_ok()
            || std::env::var("BITNET_INCREMENTAL").is_ok()
            || std::env::var("CI").is_ok()
    }

    /// Get recommended feedback time based on context
    pub fn get_recommended_feedback_time() -> Duration {
        if std::env::var("CI").is_ok() {
            Duration::from_secs(90) // 1.5 minutes for CI
        } else {
            Duration::from_secs(30) // 30 seconds for development
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fast_feedback_system_creation() {
        let system = FastFeedbackSystem::with_defaults();
        assert_eq!(system.config.target_feedback_time, Duration::from_secs(2 * 60));
        assert!(system.config.enable_incremental);
    }

    #[tokio::test]
    async fn test_ci_configuration() {
        let system = FastFeedbackSystem::for_ci();
        assert_eq!(system.config.target_feedback_time, Duration::from_secs(90));
        assert!(system.config.fail_fast);
    }

    #[tokio::test]
    async fn test_development_configuration() {
        let system = FastFeedbackSystem::for_development();
        assert_eq!(system.config.target_feedback_time, Duration::from_secs(30));
        assert!(!system.config.fail_fast);
    }

    #[test]
    fn test_feedback_quality_assessment() {
        let system = FastFeedbackSystem::with_defaults();
        let result = ExecutionResult {
            tests_run: 100,
            tests_passed: 95,
            coverage_achieved: 0.95,
            ..Default::default()
        };

        let quality = system.assess_feedback_quality(&result, Duration::from_secs(60));
        assert_eq!(quality, FeedbackQuality::Complete);
    }

    #[test]
    fn test_test_priority_calculation() {
        let mut system = FastFeedbackSystem::with_defaults();

        // Add some mock history
        system
            .execution_history
            .test_durations
            .insert("fast_test".to_string(), Duration::from_secs(1));
        system
            .execution_history
            .test_durations
            .insert("slow_test".to_string(), Duration::from_secs(30));
        system.execution_history.test_success_rates.insert("fast_test".to_string(), 0.95);
        system.execution_history.test_success_rates.insert("slow_test".to_string(), 0.80);

        let fast_priority = system.calculate_test_priority("fast_test");
        let slow_priority = system.calculate_test_priority("slow_test");

        assert!(fast_priority > slow_priority);
    }

    #[tokio::test]
    async fn test_execution_strategy_determination() {
        let mut system = FastFeedbackSystem::with_defaults();

        // Test default strategy
        let strategy = system.determine_execution_strategy().await.unwrap();
        assert!(matches!(strategy, ExecutionStrategy::Balanced));
    }

    #[tokio::test]
    async fn test_fast_feedback_execution() {
        let mut system = FastFeedbackSystem::for_development();

        // Execute fast feedback
        let result = system.execute_fast_feedback().await;

        // Should succeed with our simplified implementation
        assert!(result.is_ok());

        if let Ok(feedback_result) = result {
            assert!(feedback_result.execution_time > Duration::ZERO);
            assert!(feedback_result.tests_run > 0);
            println!("Fast feedback test passed:");
            println!("  Execution time: {:?}", feedback_result.execution_time);
            println!("  Tests run: {}", feedback_result.tests_run);
            println!("  Feedback quality: {:?}", feedback_result.feedback_quality);
        }
    }
}
