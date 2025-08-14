use super::config::TestConfig;
use super::errors::{TestError, TestResult};
use super::fast_config::{FastConfigBuilder, SpeedProfile};
use super::incremental::IncrementalTester;
use super::parallel::ParallelExecutor;
use super::selection::TestSelector;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::process::Command;
use tracing::{debug, error, info, warn};

/// Test execution optimizer that ensures tests complete within target time
pub struct ExecutionOptimizer {
    target_duration: Duration,
    config: TestConfig,
    selector: TestSelector,
    incremental: IncrementalTester,
    parallel_executor: ParallelExecutor,
    execution_history: ExecutionHistory,
}

impl ExecutionOptimizer {
    /// Create new execution optimizer with 15-minute target
    pub fn new() -> Self {
        Self::with_target_duration(Duration::from_secs(15 * 60))
    }

    /// Create execution optimizer with custom target duration
    pub fn with_target_duration(target_duration: Duration) -> Self {
        let config = FastConfigBuilder::with_profile(SpeedProfile::Fast)
            .timeout(Duration::from_secs(60))
            .max_parallel(num_cpus::get().min(8))
            .coverage(false)
            .build();

        Self {
            target_duration,
            config: config.clone(),
            selector: TestSelector::new(config.clone()),
            incremental: IncrementalTester::new(),
            parallel_executor: ParallelExecutor::new(config.clone()),
            execution_history: ExecutionHistory::load(),
        }
    }

    /// Execute tests with optimization to meet target duration
    pub async fn execute_optimized(&mut self) -> Result<OptimizedExecutionResult, TestError> {
        let start_time = Instant::now();
        info!(
            "Starting optimized test execution with {}s target",
            self.target_duration.as_secs()
        );

        // Phase 1: Analyze and select tests
        let test_plan = self.create_execution_plan().await?;
        info!(
            "Created execution plan with {} test groups",
            test_plan.groups.len()
        );

        // Phase 2: Execute with dynamic optimization
        let mut results = Vec::new();
        let mut remaining_time = self.target_duration;

        for (i, group) in test_plan.groups.iter().enumerate() {
            let group_start = Instant::now();
            let estimated_time = self.estimate_group_time(group);

            // Check if we have enough time remaining
            if estimated_time > remaining_time {
                warn!(
                    "Insufficient time remaining for group {}: estimated {}s, remaining {}s",
                    i,
                    estimated_time.as_secs(),
                    remaining_time.as_secs()
                );

                // Apply aggressive optimizations
                let optimized_group = self.optimize_group_for_time(group, remaining_time)?;
                let result = self.execute_group(&optimized_group, remaining_time).await?;
                results.push(result);
                break;
            }

            // Execute group with timeout
            let result = self.execute_group(group, remaining_time).await?;
            let group_duration = group_start.elapsed();

            results.push(result);
            remaining_time = remaining_time.saturating_sub(group_duration);

            // Early exit if we're running out of time
            if remaining_time < Duration::from_secs(30) {
                warn!(
                    "Stopping execution with {}s remaining to avoid timeout",
                    remaining_time.as_secs()
                );
                break;
            }
        }

        let total_duration = start_time.elapsed();
        let success = total_duration <= self.target_duration;

        // Update execution history
        self.execution_history
            .record_execution(&test_plan, &results, total_duration);
        self.execution_history.save();

        Ok(OptimizedExecutionResult {
            success,
            total_duration,
            target_duration: self.target_duration,
            group_results: results,
            optimization_applied: test_plan.optimizations_applied.clone(),
            tests_skipped: test_plan.tests_skipped,
            efficiency_score: self.calculate_efficiency_score(total_duration),
        })
    }

    /// Create execution plan based on available time and test priorities
    async fn create_execution_plan(&mut self) -> Result<ExecutionPlan, TestError> {
        let mut plan = ExecutionPlan::default();

        // Step 1: Detect changes for incremental testing
        let changed_files = self.incremental.detect_changes().await?;
        if !changed_files.is_empty() {
            info!(
                "Detected {} changed files, enabling incremental testing",
                changed_files.len()
            );
            plan.optimizations_applied.push("incremental".to_string());
        }

        // Step 2: Select tests based on priority and time constraints
        let available_tests = self.selector.discover_tests().await?;
        let estimated_total_time = self.estimate_total_time(&available_tests);

        info!(
            "Discovered {} tests, estimated total time: {}s",
            available_tests.len(),
            estimated_total_time.as_secs()
        );

        if estimated_total_time <= self.target_duration {
            // We can run all tests
            plan.groups = self.create_test_groups(available_tests, GroupingStrategy::Balanced)?;
        } else {
            // Need to optimize test selection
            let selected_tests =
                self.select_tests_for_time_budget(&available_tests, &changed_files)?;
            plan.groups = self.create_test_groups(selected_tests, GroupingStrategy::FastFirst)?;
            plan.optimizations_applied
                .push("test_selection".to_string());
            plan.tests_skipped =
                available_tests.len() - plan.groups.iter().map(|g| g.tests.len()).sum::<usize>();
        }

        // Step 3: Apply parallel execution optimization
        if self.config.max_parallel_tests > 1 {
            plan.optimizations_applied
                .push("parallel_execution".to_string());
        }

        // Step 4: Apply timeout optimizations
        if self.config.test_timeout < Duration::from_secs(120) {
            plan.optimizations_applied
                .push("reduced_timeouts".to_string());
        }

        Ok(plan)
    }

    /// Select tests that fit within time budget, prioritizing critical tests
    fn select_tests_for_time_budget(
        &self,
        available_tests: &[TestInfo],
        changed_files: &[PathBuf],
    ) -> Result<Vec<TestInfo>, TestError> {
        let mut selected = Vec::new();
        let mut remaining_time = self.target_duration;

        // Priority 1: Tests for changed files (incremental)
        for test in available_tests {
            if self.is_test_affected_by_changes(test, changed_files) {
                let estimated_time = self.estimate_test_time(test);
                if estimated_time <= remaining_time {
                    selected.push(test.clone());
                    remaining_time = remaining_time.saturating_sub(estimated_time);
                }
            }
        }

        // Priority 2: Critical/fast unit tests
        for test in available_tests {
            if test.category == TestCategory::Unit && test.priority == TestPriority::Critical {
                if selected.iter().any(|t| t.name == test.name) {
                    continue; // Already selected
                }
                let estimated_time = self.estimate_test_time(test);
                if estimated_time <= remaining_time {
                    selected.push(test.clone());
                    remaining_time = remaining_time.saturating_sub(estimated_time);
                }
            }
        }

        // Priority 3: Other unit tests
        for test in available_tests {
            if test.category == TestCategory::Unit && test.priority != TestPriority::Critical {
                if selected.iter().any(|t| t.name == test.name) {
                    continue;
                }
                let estimated_time = self.estimate_test_time(test);
                if estimated_time <= remaining_time {
                    selected.push(test.clone());
                    remaining_time = remaining_time.saturating_sub(estimated_time);
                }
            }
        }

        // Priority 4: Integration tests (if time allows)
        for test in available_tests {
            if test.category == TestCategory::Integration {
                if selected.iter().any(|t| t.name == test.name) {
                    continue;
                }
                let estimated_time = self.estimate_test_time(test);
                if estimated_time <= remaining_time {
                    selected.push(test.clone());
                    remaining_time = remaining_time.saturating_sub(estimated_time);
                }
            }
        }

        info!(
            "Selected {} tests out of {} available ({}s remaining)",
            selected.len(),
            available_tests.len(),
            remaining_time.as_secs()
        );

        Ok(selected)
    }

    /// Create test groups for parallel execution
    fn create_test_groups(
        &self,
        tests: Vec<TestInfo>,
        strategy: GroupingStrategy,
    ) -> Result<Vec<TestGroup>, TestError> {
        let max_groups = self.config.max_parallel_tests;
        let mut groups = vec![TestGroup::default(); max_groups];

        match strategy {
            GroupingStrategy::Balanced => {
                // Distribute tests evenly across groups by estimated time
                let mut sorted_tests = tests;
                sorted_tests
                    .sort_by(|a, b| self.estimate_test_time(b).cmp(&self.estimate_test_time(a)));

                for (i, test) in sorted_tests.into_iter().enumerate() {
                    let group_index = i % max_groups;
                    groups[group_index].tests.push(test);
                }
            }
            GroupingStrategy::FastFirst => {
                // Put fast tests in first groups, slow tests in later groups
                let mut fast_tests = Vec::new();
                let mut slow_tests = Vec::new();

                for test in tests {
                    if self.estimate_test_time(&test) <= Duration::from_secs(30) {
                        fast_tests.push(test);
                    } else {
                        slow_tests.push(test);
                    }
                }

                // Distribute fast tests across all groups
                for (i, test) in fast_tests.into_iter().enumerate() {
                    let group_index = i % max_groups;
                    groups[group_index].tests.push(test);
                }

                // Add slow tests to groups with least load
                for test in slow_tests {
                    let group_index = groups
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, g)| g.estimated_duration())
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    groups[group_index].tests.push(test);
                }
            }
        }

        // Remove empty groups
        groups.retain(|g| !g.tests.is_empty());

        // Update group metadata
        for (i, group) in groups.iter_mut().enumerate() {
            group.id = i;
            group.estimated_time = group.tests.iter().map(|t| self.estimate_test_time(t)).sum();
        }

        Ok(groups)
    }

    /// Execute a test group with timeout
    async fn execute_group(
        &self,
        group: &TestGroup,
        timeout: Duration,
    ) -> Result<GroupExecutionResult, TestError> {
        let start_time = Instant::now();
        info!(
            "Executing test group {} with {} tests (timeout: {}s)",
            group.id,
            group.tests.len(),
            timeout.as_secs()
        );

        let result =
            tokio::time::timeout(timeout, self.parallel_executor.execute_test_group(group)).await;

        match result {
            Ok(Ok(execution_result)) => {
                let duration = start_time.elapsed();
                info!("Group {} completed in {}s", group.id, duration.as_secs());

                Ok(GroupExecutionResult {
                    group_id: group.id,
                    duration,
                    success: execution_result.all_passed(),
                    test_results: execution_result.test_results,
                    timeout_occurred: false,
                })
            }
            Ok(Err(e)) => {
                error!("Group {} failed: {}", group.id, e);
                Err(e)
            }
            Err(_) => {
                warn!("Group {} timed out after {}s", group.id, timeout.as_secs());
                Ok(GroupExecutionResult {
                    group_id: group.id,
                    duration: timeout,
                    success: false,
                    test_results: Vec::new(),
                    timeout_occurred: true,
                })
            }
        }
    }

    /// Optimize a test group to fit within remaining time
    fn optimize_group_for_time(
        &self,
        group: &TestGroup,
        remaining_time: Duration,
    ) -> Result<TestGroup, TestError> {
        let mut optimized = group.clone();

        // Remove slow tests first
        optimized
            .tests
            .retain(|test| self.estimate_test_time(test) <= Duration::from_secs(60));

        // If still too slow, keep only critical tests
        if optimized.estimated_duration() > remaining_time {
            optimized
                .tests
                .retain(|test| test.priority == TestPriority::Critical);
        }

        // If still too slow, keep only unit tests
        if optimized.estimated_duration() > remaining_time {
            optimized
                .tests
                .retain(|test| test.category == TestCategory::Unit);
        }

        // Update estimated time
        optimized.estimated_time = optimized
            .tests
            .iter()
            .map(|t| self.estimate_test_time(t))
            .sum();

        info!(
            "Optimized group {} from {} to {} tests ({}s -> {}s)",
            group.id,
            group.tests.len(),
            optimized.tests.len(),
            group.estimated_time.as_secs(),
            optimized.estimated_time.as_secs()
        );

        Ok(optimized)
    }

    /// Estimate execution time for a single test
    fn estimate_test_time(&self, test: &TestInfo) -> Duration {
        // Use historical data if available
        if let Some(historical_time) = self.execution_history.get_test_time(&test.name) {
            return historical_time;
        }

        // Use category-based estimates
        match test.category {
            TestCategory::Unit => Duration::from_secs(5),
            TestCategory::Integration => Duration::from_secs(30),
            TestCategory::Performance => Duration::from_secs(60),
            TestCategory::CrossValidation => Duration::from_secs(120),
        }
    }

    /// Estimate total time for a group of tests
    fn estimate_group_time(&self, group: &TestGroup) -> Duration {
        // Account for parallel execution
        let sequential_time: Duration =
            group.tests.iter().map(|t| self.estimate_test_time(t)).sum();

        // Parallel efficiency factor (not perfect due to overhead)
        let parallel_factor = 0.8;
        let parallel_time =
            sequential_time.mul_f64(parallel_factor / self.config.max_parallel_tests as f64);

        parallel_time.max(Duration::from_secs(10)) // Minimum overhead
    }

    /// Estimate total time for all tests
    fn estimate_total_time(&self, tests: &[TestInfo]) -> Duration {
        tests
            .iter()
            .map(|t| self.estimate_test_time(t))
            .sum::<Duration>()
            .mul_f64(0.8 / self.config.max_parallel_tests as f64) // Parallel efficiency
    }

    /// Check if test is affected by file changes
    fn is_test_affected_by_changes(&self, test: &TestInfo, changed_files: &[PathBuf]) -> bool {
        // Simple heuristic: check if test path overlaps with changed files
        changed_files.iter().any(|file| {
            file.to_string_lossy().contains(&test.crate_name)
                || test
                    .file_path
                    .to_string_lossy()
                    .contains(&*file.to_string_lossy())
        })
    }

    /// Calculate efficiency score (0.0 to 1.0)
    fn calculate_efficiency_score(&self, actual_duration: Duration) -> f64 {
        let target_secs = self.target_duration.as_secs_f64();
        let actual_secs = actual_duration.as_secs_f64();

        if actual_secs <= target_secs {
            1.0 // Perfect efficiency
        } else {
            target_secs / actual_secs // Efficiency ratio
        }
    }
}

/// Test execution plan
#[derive(Debug, Clone, Default)]
pub struct ExecutionPlan {
    pub groups: Vec<TestGroup>,
    pub optimizations_applied: Vec<String>,
    pub tests_skipped: usize,
}

/// Test group for parallel execution
#[derive(Debug, Clone, Default)]
pub struct TestGroup {
    pub id: usize,
    pub tests: Vec<TestInfo>,
    pub estimated_time: Duration,
}

impl TestGroup {
    pub fn estimated_duration(&self) -> Duration {
        self.estimated_time
    }
}

/// Test information
#[derive(Debug, Clone)]
pub struct TestInfo {
    pub name: String,
    pub crate_name: String,
    pub file_path: PathBuf,
    pub category: TestCategory,
    pub priority: TestPriority,
}

/// Test categories for optimization
#[derive(Debug, Clone, PartialEq)]
pub enum TestCategory {
    Unit,
    Integration,
    Performance,
    CrossValidation,
}

/// Test priorities
#[derive(Debug, Clone, PartialEq)]
pub enum TestPriority {
    Critical,
    High,
    Medium,
    Low,
}

/// Grouping strategies
#[derive(Debug, Clone)]
pub enum GroupingStrategy {
    Balanced,
    FastFirst,
}

/// Result of optimized execution
#[derive(Debug)]
pub struct OptimizedExecutionResult {
    pub success: bool,
    pub total_duration: Duration,
    pub target_duration: Duration,
    pub group_results: Vec<GroupExecutionResult>,
    pub optimization_applied: Vec<String>,
    pub tests_skipped: usize,
    pub efficiency_score: f64,
}

impl OptimizedExecutionResult {
    pub fn summary(&self) -> String {
        format!(
            "Execution {} in {:.1}s (target: {:.1}s, efficiency: {:.1}%, skipped: {})",
            if self.success { "succeeded" } else { "failed" },
            self.total_duration.as_secs_f64(),
            self.target_duration.as_secs_f64(),
            self.efficiency_score * 100.0,
            self.tests_skipped
        )
    }
}

/// Result of group execution
#[derive(Debug)]
pub struct GroupExecutionResult {
    pub group_id: usize,
    pub duration: Duration,
    pub success: bool,
    pub test_results: Vec<super::results::TestResult>,
    pub timeout_occurred: bool,
}

/// Execution history for time estimation
#[derive(Debug, Default)]
pub struct ExecutionHistory {
    test_times: HashMap<String, Duration>,
    history_file: PathBuf,
}

impl ExecutionHistory {
    pub fn load() -> Self {
        let history_file = PathBuf::from("tests/cache/execution_history.json");
        let test_times = if history_file.exists() {
            std::fs::read_to_string(&history_file)
                .ok()
                .and_then(|content| serde_json::from_str(&content).ok())
                .unwrap_or_default()
        } else {
            HashMap::new()
        };

        Self {
            test_times,
            history_file,
        }
    }

    pub fn save(&self) {
        if let Some(parent) = self.history_file.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        let serialized: HashMap<String, u64> = self
            .test_times
            .iter()
            .map(|(k, v)| (k.clone(), v.as_millis() as u64))
            .collect();

        if let Ok(content) = serde_json::to_string_pretty(&serialized) {
            let _ = std::fs::write(&self.history_file, content);
        }
    }

    pub fn get_test_time(&self, test_name: &str) -> Option<Duration> {
        self.test_times.get(test_name).copied()
    }

    pub fn record_execution(
        &mut self,
        _plan: &ExecutionPlan,
        results: &[GroupExecutionResult],
        _total_duration: Duration,
    ) {
        for group_result in results {
            for test_result in &group_result.test_results {
                self.test_times
                    .insert(test_result.test_name.clone(), test_result.duration);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_execution_optimizer_creation() {
        let optimizer = ExecutionOptimizer::new();
        assert_eq!(optimizer.target_duration, Duration::from_secs(15 * 60));
    }

    #[test]
    fn test_time_estimation() {
        let optimizer = ExecutionOptimizer::new();
        let unit_test = TestInfo {
            name: "test_unit".to_string(),
            crate_name: "bitnet-common".to_string(),
            file_path: PathBuf::from("tests/unit_test.rs"),
            category: TestCategory::Unit,
            priority: TestPriority::High,
        };

        let estimated = optimizer.estimate_test_time(&unit_test);
        assert_eq!(estimated, Duration::from_secs(5));
    }

    #[test]
    fn test_efficiency_calculation() {
        let optimizer = ExecutionOptimizer::new();

        // Perfect efficiency
        let score1 = optimizer.calculate_efficiency_score(Duration::from_secs(10 * 60));
        assert_eq!(score1, 1.0);

        // 50% efficiency
        let score2 = optimizer.calculate_efficiency_score(Duration::from_secs(30 * 60));
        assert_eq!(score2, 0.5);
    }
}
