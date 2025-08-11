use super::cache::{CacheKey, TestCache};
use super::errors::{TestError, TestResult};
use super::harness::{TestCase, TestSuite};
use super::incremental::{IncrementalAnalysis, IncrementalTestRunner};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, SystemTime};
use tracing::{debug, info};

/// Smart test selector that prioritizes tests based on various factors
pub struct SmartTestSelector {
    config: SelectionConfig,
    history: TestHistory,
    cache: TestCache,
}

/// Configuration for smart test selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionConfig {
    /// Enable smart test selection
    pub enabled: bool,
    /// Maximum number of tests to run in a single batch
    pub max_batch_size: usize,
    /// Prioritize recently failed tests
    pub prioritize_failures: bool,
    /// Prioritize slow tests to run them early
    pub prioritize_slow_tests: bool,
    /// Prioritize tests with low cache hit rates
    pub prioritize_cache_misses: bool,
    /// Weight for failure history in prioritization
    pub failure_weight: f64,
    /// Weight for execution time in prioritization
    pub time_weight: f64,
    /// Weight for cache miss rate in prioritization
    pub cache_weight: f64,
    /// Minimum time threshold for considering a test "slow"
    pub slow_test_threshold: Duration,
}

impl Default for SelectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_batch_size: 50,
            prioritize_failures: true,
            prioritize_slow_tests: true,
            prioritize_cache_misses: true,
            failure_weight: 2.0,
            time_weight: 1.5,
            cache_weight: 1.2,
            slow_test_threshold: Duration::from_secs(10),
        }
    }
}

/// Historical data about test execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestHistory {
    /// Test execution records
    pub records: HashMap<String, TestRecord>,
    /// Last update time
    pub last_updated: SystemTime,
}

/// Record of a test's execution history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestRecord {
    /// Test name
    pub name: String,
    /// Recent execution times
    pub execution_times: Vec<Duration>,
    /// Recent failure count
    pub recent_failures: usize,
    /// Total runs
    pub total_runs: usize,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Last execution time
    pub last_execution: SystemTime,
    /// Average execution time
    pub average_time: Duration,
    /// Failure rate (0.0 to 1.0)
    pub failure_rate: f64,
    /// Flakiness score (0.0 to 1.0)
    pub flakiness_score: f64,
}

impl Default for TestRecord {
    fn default() -> Self {
        Self {
            name: String::new(),
            execution_times: Vec::new(),
            recent_failures: 0,
            total_runs: 0,
            cache_hit_rate: 0.0,
            last_execution: SystemTime::UNIX_EPOCH,
            average_time: Duration::ZERO,
            failure_rate: 0.0,
            flakiness_score: 0.0,
        }
    }
}

/// Test selection result
#[derive(Debug, Clone)]
pub struct TestSelection {
    /// Tests selected to run, in priority order
    pub selected_tests: Vec<String>,
    /// Tests that can be skipped (cached)
    pub skipped_tests: Vec<String>,
    /// Reason for each test's selection/skipping
    pub reasons: HashMap<String, String>,
    /// Priority scores for selected tests
    pub priorities: HashMap<String, f64>,
}

/// Test execution plan with batching and optimization
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Batches of tests to run in parallel
    pub batches: Vec<TestBatch>,
    /// Total estimated execution time
    pub estimated_time: Duration,
    /// Optimization strategy used
    pub strategy: OptimizationStrategy,
}

/// A batch of tests to run together
#[derive(Debug, Clone)]
pub struct TestBatch {
    /// Tests in this batch
    pub tests: Vec<String>,
    /// Estimated batch execution time
    pub estimated_time: Duration,
    /// Batch priority (higher = run earlier)
    pub priority: f64,
}

/// Optimization strategy for test execution
#[derive(Debug, Clone)]
pub enum OptimizationStrategy {
    /// Run tests in priority order
    Priority,
    /// Balance load across parallel workers
    LoadBalancing,
    /// Minimize total execution time
    TimeOptimized,
    /// Maximize early feedback (fail fast)
    FailFast,
}

impl SmartTestSelector {
    /// Create a new smart test selector
    pub async fn new(config: SelectionConfig, cache: TestCache) -> TestResult<Self> {
        let history = TestHistory::load().await.unwrap_or_default();

        Ok(Self {
            config,
            history,
            cache,
        })
    }

    /// Select tests to run based on incremental analysis and smart prioritization
    pub async fn select_tests<T: TestSuite>(
        &mut self,
        suite: &T,
        incremental_analysis: &IncrementalAnalysis,
    ) -> TestResult<TestSelection> {
        if !self.config.enabled {
            // Return all tests if smart selection is disabled
            let all_tests: Vec<String> = suite
                .test_cases()
                .iter()
                .map(|t| t.name().to_string())
                .collect();
            return Ok(TestSelection {
                selected_tests: all_tests.clone(),
                skipped_tests: Vec::new(),
                reasons: all_tests
                    .iter()
                    .map(|t| (t.clone(), "Smart selection disabled".to_string()))
                    .collect(),
                priorities: HashMap::new(),
            });
        }

        info!("Selecting tests with smart prioritization");

        let mut selected_tests = Vec::new();
        let mut skipped_tests = Vec::new();
        let mut reasons = HashMap::new();
        let mut priorities = HashMap::new();

        // Start with affected tests from incremental analysis
        let mut candidate_tests: HashSet<String> = incremental_analysis.affected_tests.clone();

        // Add cached tests as skipped
        for test_name in &incremental_analysis.cached_tests {
            skipped_tests.push(test_name.clone());
            reasons.insert(
                test_name.clone(),
                "Valid cached result available".to_string(),
            );
        }

        // If we have too few affected tests, add some additional tests based on history
        if candidate_tests.len() < self.config.max_batch_size / 2 {
            let additional_tests = self
                .select_additional_tests(suite, &candidate_tests)
                .await?;
            candidate_tests.extend(additional_tests);
        }

        // Calculate priorities for all candidate tests
        for test_name in &candidate_tests {
            let priority = self.calculate_test_priority(test_name).await;
            priorities.insert(test_name.clone(), priority);
        }

        // Sort tests by priority (highest first)
        let mut prioritized_tests: Vec<_> = candidate_tests.into_iter().collect();
        prioritized_tests.sort_by(|a, b| {
            let priority_a = priorities.get(a).unwrap_or(&0.0);
            let priority_b = priorities.get(b).unwrap_or(&0.0);
            priority_b
                .partial_cmp(priority_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Select up to max_batch_size tests
        let selected_count = prioritized_tests.len().min(self.config.max_batch_size);
        selected_tests = prioritized_tests.into_iter().take(selected_count).collect();

        // Generate reasons for selection
        for test_name in &selected_tests {
            let priority = priorities.get(test_name).unwrap_or(&0.0);
            let reason = self.generate_selection_reason(test_name, *priority).await;
            reasons.insert(test_name.clone(), reason);
        }

        info!(
            "Selected {} tests to run, {} tests cached/skipped",
            selected_tests.len(),
            skipped_tests.len()
        );

        Ok(TestSelection {
            selected_tests,
            skipped_tests,
            reasons,
            priorities,
        })
    }

    /// Create an optimized execution plan
    pub async fn create_execution_plan(
        &self,
        selection: &TestSelection,
        max_parallel: usize,
    ) -> TestResult<ExecutionPlan> {
        let strategy = self.determine_optimization_strategy(selection).await;

        match strategy {
            OptimizationStrategy::Priority => {
                self.create_priority_plan(selection, max_parallel).await
            }
            OptimizationStrategy::LoadBalancing => {
                self.create_load_balanced_plan(selection, max_parallel)
                    .await
            }
            OptimizationStrategy::TimeOptimized => {
                self.create_time_optimized_plan(selection, max_parallel)
                    .await
            }
            OptimizationStrategy::FailFast => {
                self.create_fail_fast_plan(selection, max_parallel).await
            }
        }
    }

    /// Update test history with execution results
    pub async fn update_history(
        &mut self,
        test_name: &str,
        duration: Duration,
        success: bool,
    ) -> TestResult<()> {
        let record = self
            .history
            .records
            .entry(test_name.to_string())
            .or_default();

        record.name = test_name.to_string();
        record.execution_times.push(duration);
        record.total_runs += 1;
        record.last_execution = SystemTime::now();

        if !success {
            record.recent_failures += 1;
        }

        // Keep only recent execution times (last 10)
        if record.execution_times.len() > 10 {
            record.execution_times.remove(0);
        }

        // Update derived metrics
        record.average_time = Duration::from_nanos(
            record
                .execution_times
                .iter()
                .map(|d| d.as_nanos())
                .sum::<u128>() as u64
                / record.execution_times.len() as u64,
        );

        record.failure_rate = record.recent_failures as f64 / record.total_runs as f64;

        // Calculate flakiness score (variance in success/failure)
        record.flakiness_score = self.calculate_flakiness_score(record);

        self.history.last_updated = SystemTime::now();
        self.history.save().await?;

        Ok(())
    }

    /// Select additional tests to run when incremental analysis yields few tests
    async fn select_additional_tests<T: TestSuite>(
        &self,
        suite: &T,
        existing_tests: &HashSet<String>,
    ) -> TestResult<Vec<String>> {
        let mut additional_tests = Vec::new();
        let all_tests: Vec<_> = suite
            .test_cases()
            .iter()
            .map(|t| t.name().to_string())
            .collect();

        // Add tests that haven't been run recently
        for test_name in &all_tests {
            if existing_tests.contains(test_name) {
                continue;
            }

            if let Some(record) = self.history.records.get(test_name) {
                let age = SystemTime::now()
                    .duration_since(record.last_execution)
                    .unwrap_or(Duration::MAX);

                // Add tests that haven't been run in the last 24 hours
                if age > Duration::from_secs(24 * 60 * 60) {
                    additional_tests.push(test_name.clone());
                }
            } else {
                // Add tests with no history
                additional_tests.push(test_name.clone());
            }
        }

        // Limit additional tests
        additional_tests.truncate(self.config.max_batch_size / 4);

        Ok(additional_tests)
    }

    /// Calculate priority score for a test
    async fn calculate_test_priority(&self, test_name: &str) -> f64 {
        let mut priority = 1.0;

        if let Some(record) = self.history.records.get(test_name) {
            // Factor in failure rate
            if self.config.prioritize_failures {
                priority += record.failure_rate * self.config.failure_weight;
            }

            // Factor in execution time
            if self.config.prioritize_slow_tests
                && record.average_time > self.config.slow_test_threshold
            {
                let time_factor = record.average_time.as_secs_f64()
                    / self.config.slow_test_threshold.as_secs_f64();
                priority += time_factor * self.config.time_weight;
            }

            // Factor in cache miss rate
            if self.config.prioritize_cache_misses {
                let cache_miss_rate = 1.0 - record.cache_hit_rate;
                priority += cache_miss_rate * self.config.cache_weight;
            }

            // Factor in flakiness (flaky tests should run more often)
            priority += record.flakiness_score * 0.5;
        } else {
            // New tests get higher priority
            priority += 1.0;
        }

        priority
    }

    /// Generate a human-readable reason for test selection
    async fn generate_selection_reason(&self, test_name: &str, priority: f64) -> String {
        let mut reasons = Vec::new();

        if let Some(record) = self.history.records.get(test_name) {
            if record.failure_rate > 0.1 {
                reasons.push("recently failed");
            }
            if record.average_time > self.config.slow_test_threshold {
                reasons.push("slow test");
            }
            if record.cache_hit_rate < 0.5 {
                reasons.push("low cache hit rate");
            }
            if record.flakiness_score > 0.3 {
                reasons.push("flaky test");
            }
        } else {
            reasons.push("new test");
        }

        if reasons.is_empty() {
            format!("priority score: {:.2}", priority)
        } else {
            format!("{} (priority: {:.2})", reasons.join(", "), priority)
        }
    }

    /// Determine the best optimization strategy
    async fn determine_optimization_strategy(
        &self,
        selection: &TestSelection,
    ) -> OptimizationStrategy {
        let total_tests = selection.selected_tests.len();
        let high_priority_tests = selection.priorities.values().filter(|&&p| p > 2.0).count();

        if high_priority_tests > total_tests / 2 {
            OptimizationStrategy::FailFast
        } else if total_tests > 20 {
            OptimizationStrategy::LoadBalancing
        } else {
            OptimizationStrategy::Priority
        }
    }

    /// Create a priority-based execution plan
    async fn create_priority_plan(
        &self,
        selection: &TestSelection,
        max_parallel: usize,
    ) -> TestResult<ExecutionPlan> {
        let mut batches = Vec::new();
        let mut current_batch = Vec::new();
        let mut current_batch_time = Duration::ZERO;

        for test_name in &selection.selected_tests {
            let estimated_time = self.estimate_test_time(test_name).await;

            if current_batch.len() >= max_parallel {
                batches.push(TestBatch {
                    tests: current_batch.clone(),
                    estimated_time: current_batch_time,
                    priority: self.calculate_batch_priority(&current_batch, &selection.priorities),
                });
                current_batch.clear();
                current_batch_time = Duration::ZERO;
            }

            current_batch.push(test_name.clone());
            current_batch_time = current_batch_time.max(estimated_time);
        }

        if !current_batch.is_empty() {
            batches.push(TestBatch {
                tests: current_batch.clone(),
                estimated_time: current_batch_time,
                priority: self.calculate_batch_priority(&current_batch, &selection.priorities),
            });
        }

        let estimated_time = batches.iter().map(|b| b.estimated_time).sum();

        Ok(ExecutionPlan {
            batches,
            estimated_time,
            strategy: OptimizationStrategy::Priority,
        })
    }

    /// Create a load-balanced execution plan
    async fn create_load_balanced_plan(
        &self,
        selection: &TestSelection,
        max_parallel: usize,
    ) -> TestResult<ExecutionPlan> {
        // Estimate test times
        let mut test_times = HashMap::new();
        for test_name in &selection.selected_tests {
            let time = self.estimate_test_time(test_name).await;
            test_times.insert(test_name.clone(), time);
        }

        // Sort tests by estimated time (longest first)
        let mut sorted_tests = selection.selected_tests.clone();
        sorted_tests.sort_by(|a, b| {
            let time_a = test_times.get(a).unwrap_or(&Duration::ZERO);
            let time_b = test_times.get(b).unwrap_or(&Duration::ZERO);
            time_b.cmp(time_a)
        });

        // Distribute tests across batches using a greedy algorithm
        let mut batches: Vec<TestBatch> = Vec::new();
        let mut batch_loads: Vec<Duration> = vec![Duration::ZERO; max_parallel];

        for test_name in sorted_tests {
            let test_time = test_times.get(&test_name).unwrap_or(&Duration::ZERO);

            // Find the batch with the least load
            let min_batch_idx = batch_loads
                .iter()
                .enumerate()
                .min_by_key(|(_, &load)| load)
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            // Add test to the batch with least load
            if batches.len() <= min_batch_idx {
                batches.push(TestBatch {
                    tests: Vec::new(),
                    estimated_time: Duration::ZERO,
                    priority: 0.0,
                });
            }

            batches[min_batch_idx].tests.push(test_name.clone());
            batch_loads[min_batch_idx] += *test_time;
            batches[min_batch_idx].estimated_time = batch_loads[min_batch_idx];
            batches[min_batch_idx].priority =
                self.calculate_batch_priority(&batches[min_batch_idx].tests, &selection.priorities);
        }

        // Remove empty batches
        batches.retain(|batch| !batch.tests.is_empty());

        let estimated_time = batch_loads.into_iter().max().unwrap_or(Duration::ZERO);

        Ok(ExecutionPlan {
            batches,
            estimated_time,
            strategy: OptimizationStrategy::LoadBalancing,
        })
    }

    /// Create a time-optimized execution plan
    async fn create_time_optimized_plan(
        &self,
        selection: &TestSelection,
        max_parallel: usize,
    ) -> TestResult<ExecutionPlan> {
        // This is similar to load balancing but focuses more on minimizing total time
        self.create_load_balanced_plan(selection, max_parallel)
            .await
    }

    /// Create a fail-fast execution plan
    async fn create_fail_fast_plan(
        &self,
        selection: &TestSelection,
        max_parallel: usize,
    ) -> TestResult<ExecutionPlan> {
        // Prioritize tests that are likely to fail early
        let mut high_risk_tests = Vec::new();
        let mut normal_tests = Vec::new();

        for test_name in &selection.selected_tests {
            if let Some(record) = self.history.records.get(test_name) {
                if record.failure_rate > 0.2 || record.flakiness_score > 0.3 {
                    high_risk_tests.push(test_name.clone());
                } else {
                    normal_tests.push(test_name.clone());
                }
            } else {
                normal_tests.push(test_name.clone());
            }
        }

        // Create batches with high-risk tests first
        let mut all_tests = high_risk_tests;
        all_tests.extend(normal_tests);

        let selection_with_reordered_tests = TestSelection {
            selected_tests: all_tests,
            skipped_tests: selection.skipped_tests.clone(),
            reasons: selection.reasons.clone(),
            priorities: selection.priorities.clone(),
        };

        self.create_priority_plan(&selection_with_reordered_tests, max_parallel)
            .await
    }

    /// Estimate execution time for a test
    async fn estimate_test_time(&self, test_name: &str) -> Duration {
        if let Some(record) = self.history.records.get(test_name) {
            if record.average_time > Duration::ZERO {
                return record.average_time;
            }
        }

        // Default estimate for unknown tests
        Duration::from_secs(5)
    }

    /// Calculate priority for a batch of tests
    fn calculate_batch_priority(&self, tests: &[String], priorities: &HashMap<String, f64>) -> f64 {
        if tests.is_empty() {
            return 0.0;
        }

        let total_priority: f64 = tests
            .iter()
            .map(|test| priorities.get(test).unwrap_or(&1.0))
            .sum();

        total_priority / tests.len() as f64
    }

    /// Calculate flakiness score for a test record
    fn calculate_flakiness_score(&self, record: &TestRecord) -> f64 {
        if record.total_runs < 5 {
            return 0.0; // Not enough data
        }

        // Simple flakiness calculation based on failure rate variance
        let failure_rate = record.failure_rate;

        // Tests that fail sometimes but not always are considered flaky
        if failure_rate > 0.1 && failure_rate < 0.9 {
            // Peak flakiness at 50% failure rate
            let distance_from_50 = (failure_rate - 0.5).abs();
            1.0 - (distance_from_50 * 2.0)
        } else {
            0.0
        }
    }
}

impl TestHistory {
    /// Load test history from disk
    async fn load() -> TestResult<Self> {
        let history_path = std::env::var("BITNET_TEST_HISTORY")
            .unwrap_or_else(|_| "tests/cache/history.json".to_string());

        if let Ok(content) = tokio::fs::read_to_string(&history_path).await {
            serde_json::from_str(&content)
                .map_err(|e| TestError::cache(format!("Failed to parse test history: {}", e)))
        } else {
            Ok(Self::default())
        }
    }

    /// Save test history to disk
    async fn save(&self) -> TestResult<()> {
        let history_path = std::env::var("BITNET_TEST_HISTORY")
            .unwrap_or_else(|_| "tests/cache/history.json".to_string());

        // Ensure directory exists
        if let Some(parent) = std::path::Path::new(&history_path).parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let content = serde_json::to_string_pretty(self)
            .map_err(|e| TestError::cache(format!("Failed to serialize test history: {}", e)))?;

        tokio::fs::write(&history_path, content).await?;
        Ok(())
    }
}

impl Default for TestHistory {
    fn default() -> Self {
        Self {
            records: HashMap::new(),
            last_updated: SystemTime::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_smart_test_selection() {
        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().join("cache");
        let config = super::super::cache::CacheConfig::default();
        let cache = TestCache::new(cache_dir, config).await.unwrap();

        let selection_config = SelectionConfig::default();
        let mut selector = SmartTestSelector::new(selection_config, cache)
            .await
            .unwrap();

        // Create mock incremental analysis
        let mut affected_tests = HashSet::new();
        affected_tests.insert("test1".to_string());
        affected_tests.insert("test2".to_string());

        let incremental_analysis = IncrementalAnalysis {
            affected_tests,
            cached_tests: HashSet::new(),
            changes: Vec::new(),
            run_all: false,
            reason: "Test analysis".to_string(),
        };

        // Mock test suite would be needed here for a complete test
        // For now, we'll test the priority calculation
        let priority = selector.calculate_test_priority("test1").await;
        assert!(priority > 0.0);
    }

    #[test]
    fn test_flakiness_calculation() {
        let config = SelectionConfig::default();
        let cache = TestCache::new(
            std::path::PathBuf::from("/tmp"),
            super::super::cache::CacheConfig::default(),
        )
        .await
        .unwrap();
        let selector = SmartTestSelector::new(config, cache).await.unwrap();

        let mut record = TestRecord::default();
        record.total_runs = 10;
        record.failure_rate = 0.5; // 50% failure rate

        let flakiness = selector.calculate_flakiness_score(&record);
        assert!(flakiness > 0.5); // Should be considered flaky

        record.failure_rate = 0.0; // Never fails
        let flakiness = selector.calculate_flakiness_score(&record);
        assert_eq!(flakiness, 0.0); // Not flaky

        record.failure_rate = 1.0; // Always fails
        let flakiness = selector.calculate_flakiness_score(&record);
        assert_eq!(flakiness, 0.0); // Not flaky, just broken
    }
}
