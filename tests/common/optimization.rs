use super::config::TestConfig;
use super::errors::TestOpResult as TestResultCompat;
use super::units::BYTES_PER_MB;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Configuration for parallel test execution
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Maximum number of parallel workers
    pub max_workers: usize,
    /// Timeout for individual tests
    pub test_timeout: Duration,
    /// Maximum parallel tests (same as max_workers for compatibility)
    pub max_parallel: usize,
    /// Enable dynamic parallelism adjustment
    pub dynamic_parallelism: bool,
    /// Enable resource monitoring
    pub resource_monitoring: bool,
    /// Enable load balancing
    pub load_balancing: bool,
    /// CPU usage threshold (0.0 to 1.0)
    pub cpu_threshold: f64,
    /// Memory usage threshold (0.0 to 1.0)
    pub memory_threshold: f64,
    /// Minimum parallel tests
    pub min_parallel: usize,
    /// Resource check interval in seconds
    pub resource_check_interval: u64,
    /// Timeout multiplier for parallel execution
    pub timeout_multiplier: f64,
}

/// Test execution optimizer focused on achieving <15 minute execution time
pub struct TestExecutionOptimizer {
    config: OptimizationConfig,
    target_duration: Duration,
    historical_data: HashMap<String, TestPerformanceData>,
}

/// Configuration for test execution optimization
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Target total execution time
    pub target_duration: Duration,
    /// Maximum parallel tests
    pub max_parallel: usize,
    /// Enable aggressive optimizations
    pub aggressive_mode: bool,
    /// Skip slow tests if needed to meet target
    pub skip_slow_tests: bool,
    /// Slow test threshold
    pub slow_test_threshold: Duration,
    /// Enable test result caching
    pub enable_caching: bool,
    /// Enable incremental testing
    pub enable_incremental: bool,
    /// Timeout for individual tests
    pub test_timeout: Duration,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            target_duration: Duration::from_secs(15 * 60), // 15 minutes
            max_parallel: num_cpus::get().max(4),
            aggressive_mode: true,
            skip_slow_tests: true,
            slow_test_threshold: Duration::from_secs(30),
            enable_caching: true,
            enable_incremental: true,
            test_timeout: Duration::from_secs(60), // Reduced from default
        }
    }
}

/// Performance data for a test
#[derive(Debug, Clone)]
pub struct TestPerformanceData {
    pub name: String,
    pub average_duration: Duration,
    pub success_rate: f64,
    pub cache_hit_rate: f64,
    pub last_run: Option<Instant>,
    pub priority_score: f64,
}

/// Optimization strategy result
#[derive(Debug, Clone)]
pub struct OptimizationStrategy {
    pub estimated_duration: Duration,
    pub parallel_config: ParallelConfig,
    pub test_selection: Vec<String>,
    pub skipped_tests: Vec<String>,
    pub optimization_notes: Vec<String>,
}

impl TestExecutionOptimizer {
    /// Create a new test execution optimizer
    pub fn new(config: OptimizationConfig) -> Self {
        Self { target_duration: config.target_duration, config, historical_data: HashMap::new() }
    }

    /// Create an optimized test configuration for fast execution
    pub fn create_fast_config(&self) -> TestConfig {
        let mut config = TestConfig::default();

        // Optimize parallel execution
        config.max_parallel_tests = self.config.max_parallel;

        // Reduce timeouts for faster feedback
        config.test_timeout = self.config.test_timeout;

        // Enable caching for speed
        #[cfg(feature = "fixtures")]
        {
            config.fixtures.auto_download = self.config.enable_caching;
            config.fixtures.max_cache_size = 5 * BYTES_PER_MB * 1024; // 5GB cache
        }

        // Optimize reporting for speed
        config.reporting.generate_coverage = false; // Skip coverage for speed
        config.reporting.generate_performance = true;
        config.reporting.formats = vec![super::config::ReportFormat::Json]; // Fastest format

        // Disable cross-validation if not needed
        config.crossval.enabled = false;

        // Optimize logging
        config.log_level = "warn".to_string(); // Reduce logging overhead

        config
    }

    /// Create an optimized parallel configuration
    pub fn create_parallel_config(&self) -> ParallelConfig {
        ParallelConfig {
            max_workers: self.config.max_parallel,
            test_timeout: self.config.test_timeout,
            max_parallel: self.config.max_parallel,
            dynamic_parallelism: true,
            resource_monitoring: false, // Disable for speed
            load_balancing: true,
            cpu_threshold: 0.95,    // Allow higher CPU usage
            memory_threshold: 0.90, // Allow higher memory usage
            min_parallel: (self.config.max_parallel / 2).max(1),
            resource_check_interval: 10, // Less frequent checks
            timeout_multiplier: 0.8,     // Shorter timeouts in parallel
        }
    }

    /// Optimize test execution plan to meet time target
    pub async fn optimize_execution_plan(
        &mut self,
        all_tests: Vec<String>,
    ) -> TestResultCompat<OptimizationStrategy> {
        let _start_time = Instant::now();
        let mut optimization_notes = Vec::new();

        // Load historical performance data
        self.load_historical_data().await?;

        // Calculate estimated durations for all tests
        let mut test_estimates: HashMap<String, Duration> = HashMap::new();
        let mut total_estimated_time = Duration::ZERO;

        for test_name in &all_tests {
            let estimate = self.estimate_test_duration(test_name);
            test_estimates.insert(test_name.clone(), estimate);
            total_estimated_time += estimate;
        }

        optimization_notes.push(format!(
            "Initial estimate: {} tests in {:?}",
            all_tests.len(),
            total_estimated_time
        ));

        // If we're already under target, use all tests
        if total_estimated_time <= self.target_duration {
            let parallel_time = self.calculate_parallel_time(&test_estimates);

            if parallel_time <= self.target_duration {
                optimization_notes.push("All tests can run within target time".to_string());

                return Ok(OptimizationStrategy {
                    estimated_duration: parallel_time,
                    parallel_config: self.create_parallel_config(),
                    test_selection: all_tests,
                    skipped_tests: Vec::new(),
                    optimization_notes,
                });
            }
        }

        // Apply optimizations to meet target
        let (selected_tests, skipped_tests) =
            self.apply_optimizations(all_tests, &test_estimates, &mut optimization_notes).await?;

        let selected_estimates: HashMap<String, Duration> = selected_tests
            .iter()
            .filter_map(|name| test_estimates.get(name).map(|&dur| (name.clone(), dur)))
            .collect();

        let estimated_duration = self.calculate_parallel_time(&selected_estimates);

        optimization_notes.push(format!(
            "Final selection: {} tests (skipped {}) in estimated {:?}",
            selected_tests.len(),
            skipped_tests.len(),
            estimated_duration
        ));

        Ok(OptimizationStrategy {
            estimated_duration,
            parallel_config: self.create_parallel_config(),
            test_selection: selected_tests,
            skipped_tests,
            optimization_notes,
        })
    }

    /// Apply various optimizations to meet time target
    async fn apply_optimizations(
        &self,
        mut all_tests: Vec<String>,
        test_estimates: &HashMap<String, Duration>,
        notes: &mut Vec<String>,
    ) -> TestResultCompat<(Vec<String>, Vec<String>)> {
        let mut skipped_tests = Vec::new();

        // 1. Skip extremely slow tests if enabled
        if self.config.skip_slow_tests {
            let (remaining, slow_tests) = self.filter_slow_tests(all_tests, test_estimates);
            all_tests = remaining;
            skipped_tests.extend(slow_tests.clone());

            if !slow_tests.is_empty() {
                notes.push(format!(
                    "Skipped {} slow tests (>{:?})",
                    slow_tests.len(),
                    self.config.slow_test_threshold
                ));
            }
        }

        // 2. Prioritize tests by importance and speed
        all_tests = self.prioritize_tests(all_tests, test_estimates);

        // 3. Select tests that fit within time budget
        let selected_tests =
            self.select_tests_for_time_budget(all_tests.clone(), test_estimates, notes).await?;

        // 4. Add remaining tests to skipped list
        let selected_set: std::collections::HashSet<_> = selected_tests.iter().collect();
        for test in all_tests {
            if !selected_set.contains(&test) {
                skipped_tests.push(test);
            }
        }

        Ok((selected_tests, skipped_tests))
    }

    /// Filter out extremely slow tests
    fn filter_slow_tests(
        &self,
        tests: Vec<String>,
        estimates: &HashMap<String, Duration>,
    ) -> (Vec<String>, Vec<String>) {
        let mut remaining = Vec::new();
        let mut slow_tests = Vec::new();

        for test in tests {
            let duration = estimates.get(&test).unwrap_or(&Duration::ZERO);

            if *duration > self.config.slow_test_threshold {
                slow_tests.push(test);
            } else {
                remaining.push(test);
            }
        }

        (remaining, slow_tests)
    }

    /// Prioritize tests by importance and execution time
    fn prioritize_tests(
        &self,
        mut tests: Vec<String>,
        estimates: &HashMap<String, Duration>,
    ) -> Vec<String> {
        tests.sort_by(|a, b| {
            let priority_a = self.calculate_test_priority(a, estimates);
            let priority_b = self.calculate_test_priority(b, estimates);

            // Higher priority first
            priority_b.partial_cmp(&priority_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        tests
    }

    /// Calculate priority score for a test
    fn calculate_test_priority(
        &self,
        test_name: &str,
        estimates: &HashMap<String, Duration>,
    ) -> f64 {
        let mut priority = 1.0;

        // Prioritize faster tests
        if let Some(&duration) = estimates.get(test_name) {
            let speed_factor = 1.0 / (duration.as_secs_f64() + 1.0);
            priority += speed_factor * 2.0;
        }

        // Prioritize core functionality tests
        if test_name.contains("core") || test_name.contains("basic") {
            priority += 3.0;
        }

        // Prioritize unit tests over integration tests
        if test_name.contains("unit") {
            priority += 2.0;
        } else if test_name.contains("integration") {
            priority += 1.0;
        }

        // Use historical data if available
        if let Some(perf_data) = self.historical_data.get(test_name) {
            priority += perf_data.priority_score;

            // Prioritize tests with high success rates
            priority += perf_data.success_rate * 1.5;

            // Prioritize tests with good cache hit rates
            priority += perf_data.cache_hit_rate * 1.0;
        }

        priority
    }

    /// Select tests that fit within the time budget
    async fn select_tests_for_time_budget(
        &self,
        tests: Vec<String>,
        estimates: &HashMap<String, Duration>,
        notes: &mut Vec<String>,
    ) -> TestResultCompat<Vec<String>> {
        let mut selected = Vec::new();
        let mut current_batches: Vec<Vec<String>> = vec![Vec::new(); self.config.max_parallel];
        let mut batch_times: Vec<Duration> = vec![Duration::ZERO; self.config.max_parallel];

        for test in tests {
            let default_duration = Duration::from_secs(5);
            let test_duration = estimates.get(&test).unwrap_or(&default_duration);

            // Find the batch with the least current time
            let min_batch_idx = batch_times
                .iter()
                .enumerate()
                .min_by_key(|(_, &time)| time)
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            let new_batch_time = batch_times[min_batch_idx] + *test_duration;

            // Check if adding this test would exceed our target
            let max_batch_time = batch_times.iter().max().unwrap_or(&Duration::ZERO);
            let estimated_total_time = new_batch_time.max(*max_batch_time);

            if estimated_total_time <= self.target_duration {
                current_batches[min_batch_idx].push(test.clone());
                batch_times[min_batch_idx] = new_batch_time;
                selected.push(test);
            } else {
                // Stop adding tests if we would exceed the target
                notes.push(format!(
                    "Stopped at {} tests to stay within {:?} target",
                    selected.len(),
                    self.target_duration
                ));
                break;
            }
        }

        Ok(selected)
    }

    /// Calculate parallel execution time
    fn calculate_parallel_time(&self, test_estimates: &HashMap<String, Duration>) -> Duration {
        if test_estimates.is_empty() {
            return Duration::ZERO;
        }

        // Simulate load balancing across parallel workers
        let mut worker_loads: Vec<Duration> = vec![Duration::ZERO; self.config.max_parallel];

        // Sort tests by duration (longest first) for better load balancing
        let mut sorted_tests: Vec<_> = test_estimates.iter().collect();
        sorted_tests.sort_by(|a, b| b.1.cmp(a.1));

        for (_, &duration) in sorted_tests {
            // Assign to worker with least load
            let min_worker = worker_loads
                .iter()
                .enumerate()
                .min_by_key(|(_, &load)| load)
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            worker_loads[min_worker] += duration;
        }

        // Total time is the maximum worker load
        worker_loads.into_iter().max().unwrap_or(Duration::ZERO)
    }

    /// Estimate duration for a test
    fn estimate_test_duration(&self, test_name: &str) -> Duration {
        // Use historical data if available
        if let Some(perf_data) = self.historical_data.get(test_name) {
            return perf_data.average_duration;
        }

        // Estimate based on test name patterns
        if test_name.contains("integration") {
            Duration::from_secs(15)
        } else if test_name.contains("performance") || test_name.contains("benchmark") {
            Duration::from_secs(20)
        } else if test_name.contains("crossval") || test_name.contains("comparison") {
            Duration::from_secs(30)
        } else if test_name.contains("unit") {
            Duration::from_secs(2)
        } else {
            Duration::from_secs(5) // Default estimate
        }
    }

    /// Load historical performance data
    async fn load_historical_data(&mut self) -> TestResultCompat<()> {
        // In a real implementation, this would load from a persistent store
        // For now, we'll create some mock data

        let mock_data = vec![
            ("unit_test_basic", Duration::from_secs(1), 0.98, 0.9),
            ("unit_test_advanced", Duration::from_secs(3), 0.95, 0.8),
            ("integration_test_workflow", Duration::from_secs(12), 0.90, 0.7),
            ("performance_benchmark", Duration::from_secs(25), 0.85, 0.6),
            ("crossval_comparison", Duration::from_secs(45), 0.80, 0.5),
        ];

        for (name, duration, success_rate, cache_hit_rate) in mock_data {
            let priority_score = success_rate * 2.0
                + cache_hit_rate * 1.0
                + (1.0 / duration.as_secs_f64().max(1.0)) * 3.0;

            self.historical_data.insert(
                name.to_string(),
                TestPerformanceData {
                    name: name.to_string(),
                    average_duration: duration,
                    success_rate,
                    cache_hit_rate,
                    last_run: None,
                    priority_score,
                },
            );
        }

        Ok(())
    }

    /// Update performance data after test execution
    pub fn update_performance_data(
        &mut self,
        test_name: &str,
        duration: Duration,
        success: bool,
        cache_hit: bool,
    ) {
        let entry = self.historical_data.entry(test_name.to_string()).or_insert_with(|| {
            TestPerformanceData {
                name: test_name.to_string(),
                average_duration: duration,
                success_rate: if success { 1.0 } else { 0.0 },
                cache_hit_rate: if cache_hit { 1.0 } else { 0.0 },
                last_run: Some(Instant::now()),
                priority_score: 1.0,
            }
        });

        // Update with exponential moving average
        let alpha = 0.3; // Learning rate
        entry.average_duration = Duration::from_secs_f64(
            entry.average_duration.as_secs_f64() * (1.0 - alpha) + duration.as_secs_f64() * alpha,
        );

        entry.success_rate =
            entry.success_rate * (1.0 - alpha) + (if success { 1.0 } else { 0.0 }) * alpha;

        entry.cache_hit_rate =
            entry.cache_hit_rate * (1.0 - alpha) + (if cache_hit { 1.0 } else { 0.0 }) * alpha;

        entry.last_run = Some(Instant::now());

        // Recalculate priority score
        entry.priority_score = entry.success_rate * 2.0
            + entry.cache_hit_rate * 1.0
            + (1.0 / entry.average_duration.as_secs_f64().max(1.0)) * 3.0;
    }

    /// Generate optimization report
    pub fn generate_optimization_report(&self, strategy: &OptimizationStrategy) -> String {
        let mut report = String::new();

        report.push_str("# Test Execution Optimization Report\n\n");

        report.push_str(&format!("**Target Duration:** {:?}\n", self.target_duration));

        report.push_str(&format!("**Estimated Duration:** {:?}\n", strategy.estimated_duration));

        let efficiency = if self.target_duration > Duration::ZERO {
            strategy.estimated_duration.as_secs_f64() / self.target_duration.as_secs_f64() * 100.0
        } else {
            0.0
        };

        report.push_str(&format!("**Time Efficiency:** {:.1}% of target\n\n", efficiency));

        report.push_str(&format!("**Selected Tests:** {}\n", strategy.test_selection.len()));

        report.push_str(&format!("**Skipped Tests:** {}\n\n", strategy.skipped_tests.len()));

        report.push_str("## Optimization Notes\n\n");
        for note in &strategy.optimization_notes {
            report.push_str(&format!("- {}\n", note));
        }

        report.push_str("\n## Parallel Configuration\n\n");
        report
            .push_str(&format!("- **Max Parallel:** {}\n", strategy.parallel_config.max_parallel));
        report.push_str(&format!(
            "- **Dynamic Parallelism:** {}\n",
            strategy.parallel_config.dynamic_parallelism
        ));
        report.push_str(&format!(
            "- **Load Balancing:** {}\n",
            strategy.parallel_config.load_balancing
        ));

        if !strategy.skipped_tests.is_empty() {
            report.push_str("\n## Skipped Tests\n\n");
            for test in &strategy.skipped_tests {
                report.push_str(&format!("- {}\n", test));
            }
        }

        report
    }
}

/// Utility functions for test execution optimization
pub mod utils {
    use super::*;

    /// Get optimal number of parallel tests based on system resources
    pub fn get_optimal_parallel_tests() -> usize {
        let cpu_count = num_cpus::get();
        let available_memory_gb = get_available_memory_gb();

        // Base on CPU count but consider memory constraints
        let cpu_based = cpu_count;
        let memory_based = (available_memory_gb / 2).max(1) as usize; // 2GB per test

        cpu_based.min(memory_based).max(1)
    }

    /// Get available memory in GB (simplified)
    fn get_available_memory_gb() -> u64 {
        // In a real implementation, this would query system memory
        // For now, return a conservative estimate
        8 // 8 GB
    }

    /// Create a fast test configuration for CI environments
    pub fn create_ci_fast_config() -> OptimizationConfig {
        OptimizationConfig {
            target_duration: Duration::from_secs(10 * 60), // 10 minutes for CI
            max_parallel: get_optimal_parallel_tests().min(8), // Limit for CI stability
            aggressive_mode: true,
            skip_slow_tests: true,
            slow_test_threshold: Duration::from_secs(20), // Stricter for CI
            enable_caching: true,
            enable_incremental: true,
            test_timeout: Duration::from_secs(45), // Shorter timeout for CI
        }
    }

    /// Create a development-friendly configuration
    pub fn create_dev_config() -> OptimizationConfig {
        OptimizationConfig {
            target_duration: Duration::from_secs(5 * 60), // 5 minutes for dev
            max_parallel: get_optimal_parallel_tests(),
            aggressive_mode: false,
            skip_slow_tests: false, // Run all tests in dev
            slow_test_threshold: Duration::from_secs(60),
            enable_caching: true,
            enable_incremental: true,
            test_timeout: Duration::from_secs(120),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_optimization_config() {
        let config = OptimizationConfig::default();
        assert_eq!(config.target_duration, Duration::from_secs(15 * 60));
        assert!(config.max_parallel > 0);
    }

    #[tokio::test]
    async fn test_test_execution_optimizer() {
        let config = OptimizationConfig::default();
        let mut optimizer = TestExecutionOptimizer::new(config);

        let test_names = vec![
            "unit_test_1".to_string(),
            "unit_test_2".to_string(),
            "integration_test_1".to_string(),
        ];

        let strategy = optimizer.optimize_execution_plan(test_names).await.unwrap();

        assert!(!strategy.test_selection.is_empty());
        assert!(strategy.estimated_duration <= Duration::from_secs(15 * 60));
    }

    #[test]
    fn test_priority_calculation() {
        let config = OptimizationConfig::default();
        let optimizer = TestExecutionOptimizer::new(config);

        let mut estimates = HashMap::new();
        estimates.insert("fast_test".to_string(), Duration::from_secs(1));
        estimates.insert("slow_test".to_string(), Duration::from_secs(30));

        let fast_priority = optimizer.calculate_test_priority("fast_test", &estimates);
        let slow_priority = optimizer.calculate_test_priority("slow_test", &estimates);

        assert!(fast_priority > slow_priority);
    }

    #[test]
    fn test_parallel_time_calculation() {
        let config = OptimizationConfig { max_parallel: 2, ..OptimizationConfig::default() };
        let optimizer = TestExecutionOptimizer::new(config);

        let mut estimates = HashMap::new();
        estimates.insert("test1".to_string(), Duration::from_secs(10));
        estimates.insert("test2".to_string(), Duration::from_secs(5));
        estimates.insert("test3".to_string(), Duration::from_secs(3));

        let parallel_time = optimizer.calculate_parallel_time(&estimates);

        // With 2 parallel workers: worker1 gets 10s test, worker2 gets 5s+3s=8s
        // So total time should be 10s (max of the two workers)
        assert_eq!(parallel_time, Duration::from_secs(10));
    }
}
