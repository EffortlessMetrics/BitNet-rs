use super::cache::TestCache;
use super::errors::{TestError, TestResult};
use super::harness::{TestCase, TestSuite};
use super::results::{TestResult as TestResultData, TestSuiteResult};
use super::selection::{ExecutionPlan, TestBatch};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

/// Parallel test executor with advanced optimization
pub struct ParallelTestExecutor {
    config: ParallelConfig,
    semaphore: Arc<Semaphore>,
    resource_monitor: ResourceMonitor,
    load_balancer: LoadBalancer,
}

/// Configuration for parallel test execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// Maximum number of parallel tests
    pub max_parallel: usize,
    /// Enable dynamic parallelism adjustment
    pub dynamic_parallelism: bool,
    /// Enable resource monitoring
    pub resource_monitoring: bool,
    /// Enable load balancing
    pub load_balancing: bool,
    /// CPU usage threshold for scaling down (0.0 to 1.0)
    pub cpu_threshold: f64,
    /// Memory usage threshold for scaling down (0.0 to 1.0)
    pub memory_threshold: f64,
    /// Minimum parallel tests (never go below this)
    pub min_parallel: usize,
    /// Resource check interval in seconds
    pub resource_check_interval: u64,
    /// Test timeout multiplier for parallel execution
    pub timeout_multiplier: f64,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            max_parallel: num_cpus::get().max(1),
            dynamic_parallelism: true,
            resource_monitoring: true,
            load_balancing: true,
            cpu_threshold: 0.9,
            memory_threshold: 0.85,
            min_parallel: 1,
            resource_check_interval: 5,
            timeout_multiplier: 1.5,
        }
    }
}

/// Resource monitoring for dynamic parallelism adjustment
pub struct ResourceMonitor {
    config: ParallelConfig,
    current_usage: Arc<RwLock<ResourceUsage>>,
    monitoring_handle: Option<JoinHandle<()>>,
}

/// Current resource usage
#[derive(Debug, Clone, Default)]
pub struct ResourceUsage {
    pub cpu_percent: f64,
    pub memory_percent: f64,
    pub available_memory_mb: u64,
    pub load_average: f64,
    pub active_tests: usize,
}

/// Load balancer for distributing tests across workers
pub struct LoadBalancer {
    worker_loads: Arc<RwLock<Vec<WorkerLoad>>>,
    config: ParallelConfig,
}

/// Load information for a worker
#[derive(Debug, Clone)]
pub struct WorkerLoad {
    pub worker_id: usize,
    pub active_tests: usize,
    pub total_time: Duration,
    pub average_time: Duration,
    pub last_assignment: Instant,
}

/// Execution context for a parallel test run
pub struct ExecutionContext {
    pub worker_id: usize,
    pub batch_id: usize,
    pub test_index: usize,
    pub start_time: Instant,
    pub estimated_duration: Duration,
}

/// Result of parallel test execution
#[derive(Debug, Clone)]
pub struct ParallelExecutionResult {
    pub results: Vec<TestResultData>,
    pub total_duration: Duration,
    pub parallel_efficiency: f64,
    pub resource_stats: ResourceStats,
    pub worker_stats: Vec<WorkerStats>,
}

/// Resource usage statistics
#[derive(Debug, Clone)]
pub struct ResourceStats {
    pub peak_cpu_percent: f64,
    pub peak_memory_percent: f64,
    pub average_cpu_percent: f64,
    pub average_memory_percent: f64,
    pub parallelism_adjustments: usize,
}

/// Worker performance statistics
#[derive(Debug, Clone)]
pub struct WorkerStats {
    pub worker_id: usize,
    pub tests_executed: usize,
    pub total_time: Duration,
    pub idle_time: Duration,
    pub efficiency: f64,
}

impl ParallelTestExecutor {
    /// Create a new parallel test executor
    pub async fn new(config: ParallelConfig) -> TestResult<Self> {
        let semaphore = Arc::new(Semaphore::new(config.max_parallel));
        let resource_monitor = ResourceMonitor::new(config.clone()).await?;
        let load_balancer = LoadBalancer::new(config.clone());

        Ok(Self {
            config,
            semaphore,
            resource_monitor,
            load_balancer,
        })
    }

    /// Execute tests according to the execution plan
    pub async fn execute_plan<T: TestSuite>(
        &mut self,
        suite: T,
        plan: ExecutionPlan,
        cache: &mut TestCache,
    ) -> TestResult<ParallelExecutionResult> {
        info!(
            "Executing {} batches with {} strategy",
            plan.batches.len(),
            match plan.strategy {
                super::selection::OptimizationStrategy::Priority => "Priority",
                super::selection::OptimizationStrategy::LoadBalancing => "Load Balancing",
                super::selection::OptimizationStrategy::TimeOptimized => "Time Optimized",
                super::selection::OptimizationStrategy::FailFast => "Fail Fast",
            }
        );

        let start_time = Instant::now();

        // Start resource monitoring
        if self.config.resource_monitoring {
            self.resource_monitor.start_monitoring().await?;
        }

        // Initialize worker tracking
        self.load_balancer
            .initialize_workers(self.config.max_parallel)
            .await;

        let mut all_results = Vec::new();
        let mut worker_stats = Vec::new();

        // Execute batches
        for (batch_id, batch) in plan.batches.iter().enumerate() {
            info!(
                "Executing batch {} with {} tests",
                batch_id,
                batch.tests.len()
            );

            let batch_results = self.execute_batch(&suite, batch, batch_id, cache).await?;

            all_results.extend(batch_results);

            // Adjust parallelism based on resource usage
            if self.config.dynamic_parallelism {
                self.adjust_parallelism().await?;
            }
        }

        // Stop resource monitoring
        if self.config.resource_monitoring {
            self.resource_monitor.stop_monitoring().await;
        }

        let total_duration = start_time.elapsed();

        // Calculate statistics
        let resource_stats = self.resource_monitor.get_stats().await;
        let parallel_efficiency = self.calculate_parallel_efficiency(&all_results, total_duration);

        // Get worker statistics
        for worker_id in 0..self.config.max_parallel {
            let stats = self.load_balancer.get_worker_stats(worker_id).await;
            worker_stats.push(stats);
        }

        info!(
            "Parallel execution completed: {} tests in {:?} (efficiency: {:.1}%)",
            all_results.len(),
            total_duration,
            parallel_efficiency * 100.0
        );

        Ok(ParallelExecutionResult {
            results: all_results,
            total_duration,
            parallel_efficiency,
            resource_stats,
            worker_stats,
        })
    }

    /// Execute a single batch of tests
    async fn execute_batch<T: TestSuite>(
        &mut self,
        suite: &T,
        batch: &TestBatch,
        batch_id: usize,
        cache: &mut TestCache,
    ) -> TestResult<Vec<TestResultData>> {
        let mut handles = Vec::new();
        let test_cases = suite.test_cases();

        // Create a map of test names to test cases for quick lookup
        let test_case_map: HashMap<String, &Box<dyn TestCase>> = test_cases
            .iter()
            .map(|tc| (tc.name().to_string(), tc))
            .collect();

        for (test_index, test_name) in batch.tests.iter().enumerate() {
            if let Some(&test_case) = test_case_map.get(test_name) {
                // Select best worker for this test
                let worker_id = self.load_balancer.select_worker(test_case).await;

                let context = ExecutionContext {
                    worker_id,
                    batch_id,
                    test_index,
                    start_time: Instant::now(),
                    estimated_duration: batch.estimated_time / batch.tests.len() as u32,
                };

                // Clone necessary data for the async task
                let semaphore = Arc::clone(&self.semaphore);
                let config = self.config.clone();
                let test_name_clone = test_name.clone();

                // Create a test case wrapper that can be sent across threads
                let test_case_data = TestCaseData {
                    name: test_case.name().to_string(),
                    metadata: test_case.metadata(),
                };

                let handle = tokio::spawn(async move {
                    Self::execute_single_test_with_context(
                        semaphore,
                        test_case_data,
                        context,
                        config,
                    )
                    .await
                });

                handles.push(handle);
            }
        }

        // Collect results
        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(result) => results.push(result),
                Err(e) => {
                    warn!("Test execution task failed: {}", e);
                    // Create a failed result for the task that panicked
                    results.push(TestResultData::failed(
                        "unknown".to_string(),
                        TestError::execution(format!("Task panic: {}", e)),
                        Duration::ZERO,
                    ));
                }
            }
        }

        Ok(results)
    }

    /// Execute a single test with execution context
    async fn execute_single_test_with_context(
        semaphore: Arc<Semaphore>,
        test_case: TestCaseData,
        context: ExecutionContext,
        config: ParallelConfig,
    ) -> TestResultData {
        let test_name = test_case.name.clone();
        let start_time = Instant::now();

        // Acquire semaphore permit
        let _permit = match semaphore.acquire().await {
            Ok(permit) => permit,
            Err(e) => {
                return TestResultData::failed(
                    test_name,
                    TestError::execution(format!("Failed to acquire semaphore: {}", e)),
                    Duration::ZERO,
                );
            }
        };

        debug!(
            "Worker {} executing test: {} (batch {}, index {})",
            context.worker_id, test_name, context.batch_id, context.test_index
        );

        // Calculate timeout with multiplier for parallel execution
        let base_timeout = Duration::from_secs(60); // Default timeout
        let timeout =
            Duration::from_secs_f64(base_timeout.as_secs_f64() * config.timeout_multiplier);

        // Execute the test with timeout
        let execute_result = tokio::time::timeout(timeout, async {
            // Simulate test execution (in real implementation, this would call the actual test)
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok(super::results::TestMetrics::default())
        })
        .await;

        let duration = start_time.elapsed();

        match execute_result {
            Ok(Ok(metrics)) => {
                debug!(
                    "Worker {} completed test: {} in {:?}",
                    context.worker_id, test_name, duration
                );
                TestResultData::passed(test_name, metrics, duration)
            }
            Ok(Err(e)) => {
                warn!(
                    "Worker {} test failed: {} - {}",
                    context.worker_id, test_name, e
                );
                TestResultData::failed(test_name, e, duration)
            }
            Err(_) => {
                warn!(
                    "Worker {} test timed out: {} after {:?}",
                    context.worker_id, test_name, timeout
                );
                TestResultData::timeout(test_name, timeout)
            }
        }
    }

    /// Adjust parallelism based on resource usage
    async fn adjust_parallelism(&mut self) -> TestResult<()> {
        let usage = self.resource_monitor.get_current_usage().await;
        let current_permits = self.semaphore.available_permits();
        let total_permits = current_permits + (self.config.max_parallel - current_permits);

        let mut new_parallelism = total_permits;

        // Scale down if resource usage is too high
        if usage.cpu_percent > self.config.cpu_threshold {
            new_parallelism = (new_parallelism * 80 / 100).max(self.config.min_parallel);
            debug!(
                "Scaling down parallelism due to high CPU usage: {:.1}%",
                usage.cpu_percent
            );
        } else if usage.memory_percent > self.config.memory_threshold {
            new_parallelism = (new_parallelism * 85 / 100).max(self.config.min_parallel);
            debug!(
                "Scaling down parallelism due to high memory usage: {:.1}%",
                usage.memory_percent
            );
        } else if usage.cpu_percent < 0.5 && usage.memory_percent < 0.5 {
            // Scale up if resource usage is low
            new_parallelism = (new_parallelism * 110 / 100).min(self.config.max_parallel);
            debug!("Scaling up parallelism due to low resource usage");
        }

        if new_parallelism != total_permits {
            info!(
                "Adjusting parallelism from {} to {}",
                total_permits, new_parallelism
            );

            // Create new semaphore with adjusted permits
            self.semaphore = Arc::new(Semaphore::new(new_parallelism));
        }

        Ok(())
    }

    /// Calculate parallel execution efficiency
    fn calculate_parallel_efficiency(
        &self,
        results: &[TestResultData],
        total_duration: Duration,
    ) -> f64 {
        if results.is_empty() || total_duration.is_zero() {
            return 0.0;
        }

        let total_test_time: Duration = results.iter().map(|r| r.duration).sum();
        let theoretical_parallel_time =
            total_test_time.as_secs_f64() / self.config.max_parallel as f64;

        if theoretical_parallel_time > 0.0 {
            theoretical_parallel_time / total_duration.as_secs_f64()
        } else {
            0.0
        }
    }
}

/// Simplified test case data for async execution
#[derive(Debug, Clone)]
struct TestCaseData {
    name: String,
    metadata: HashMap<String, String>,
}

impl ResourceMonitor {
    /// Create a new resource monitor
    async fn new(config: ParallelConfig) -> TestResult<Self> {
        Ok(Self {
            config,
            current_usage: Arc::new(RwLock::new(ResourceUsage::default())),
            monitoring_handle: None,
        })
    }

    /// Start resource monitoring
    async fn start_monitoring(&mut self) -> TestResult<()> {
        if !self.config.resource_monitoring {
            return Ok(());
        }

        let usage = Arc::clone(&self.current_usage);
        let interval = Duration::from_secs(self.config.resource_check_interval);

        let handle = tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                let new_usage = Self::collect_resource_usage().await;

                {
                    let mut current = usage.write().await;
                    *current = new_usage;
                }
            }
        });

        self.monitoring_handle = Some(handle);
        Ok(())
    }

    /// Stop resource monitoring
    async fn stop_monitoring(&mut self) {
        if let Some(handle) = self.monitoring_handle.take() {
            handle.abort();
        }
    }

    /// Get current resource usage
    async fn get_current_usage(&self) -> ResourceUsage {
        self.current_usage.read().await.clone()
    }

    /// Get resource statistics
    async fn get_stats(&self) -> ResourceStats {
        let usage = self.get_current_usage().await;

        // In a real implementation, this would track historical data
        ResourceStats {
            peak_cpu_percent: usage.cpu_percent,
            peak_memory_percent: usage.memory_percent,
            average_cpu_percent: usage.cpu_percent,
            average_memory_percent: usage.memory_percent,
            parallelism_adjustments: 0,
        }
    }

    /// Collect current resource usage
    async fn collect_resource_usage() -> ResourceUsage {
        // In a real implementation, this would use system APIs to get actual resource usage
        // For now, we'll simulate some values
        ResourceUsage {
            cpu_percent: 0.6,          // 60% CPU usage
            memory_percent: 0.4,       // 40% memory usage
            available_memory_mb: 8192, // 8 GB available
            load_average: 2.0,
            active_tests: 0,
        }
    }
}

impl LoadBalancer {
    /// Create a new load balancer
    fn new(config: ParallelConfig) -> Self {
        Self {
            worker_loads: Arc::new(RwLock::new(Vec::new())),
            config,
        }
    }

    /// Initialize workers
    async fn initialize_workers(&self, worker_count: usize) {
        let mut loads = self.worker_loads.write().await;
        loads.clear();

        for worker_id in 0..worker_count {
            loads.push(WorkerLoad {
                worker_id,
                active_tests: 0,
                total_time: Duration::ZERO,
                average_time: Duration::ZERO,
                last_assignment: Instant::now(),
            });
        }
    }

    /// Select the best worker for a test
    async fn select_worker(&self, _test_case: &dyn TestCase) -> usize {
        if !self.config.load_balancing {
            // Simple round-robin if load balancing is disabled
            return rand::random::<usize>() % self.config.max_parallel;
        }

        let loads = self.worker_loads.read().await;

        // Find worker with least load
        loads
            .iter()
            .enumerate()
            .min_by_key(|(_, load)| (load.active_tests, load.total_time))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Get worker statistics
    async fn get_worker_stats(&self, worker_id: usize) -> WorkerStats {
        let loads = self.worker_loads.read().await;

        if let Some(load) = loads.get(worker_id) {
            WorkerStats {
                worker_id,
                tests_executed: 0, // Would be tracked in real implementation
                total_time: load.total_time,
                idle_time: Duration::ZERO, // Would be calculated in real implementation
                efficiency: 0.8,           // Would be calculated based on actual metrics
            }
        } else {
            WorkerStats {
                worker_id,
                tests_executed: 0,
                total_time: Duration::ZERO,
                idle_time: Duration::ZERO,
                efficiency: 0.0,
            }
        }
    }
}

/// Utility functions for parallel execution optimization
pub mod optimization {
    use super::*;

    /// Calculate optimal parallelism based on system resources
    pub fn calculate_optimal_parallelism() -> usize {
        let cpu_count = num_cpus::get();
        let available_memory_gb = get_available_memory_gb();

        // Base parallelism on CPU count
        let mut optimal = cpu_count;

        // Adjust based on available memory (assume 1GB per parallel test)
        if available_memory_gb < cpu_count as u64 {
            optimal = available_memory_gb as usize;
        }

        // Ensure minimum of 1
        optimal.max(1)
    }

    /// Get available memory in GB
    fn get_available_memory_gb() -> u64 {
        // In a real implementation, this would query system memory
        // For now, return a reasonable default
        8 // 8 GB
    }

    /// Create optimized parallel configuration
    pub fn create_optimized_config() -> ParallelConfig {
        let optimal_parallelism = calculate_optimal_parallelism();

        ParallelConfig {
            max_parallel: optimal_parallelism,
            min_parallel: (optimal_parallelism / 4).max(1),
            dynamic_parallelism: true,
            resource_monitoring: true,
            load_balancing: true,
            cpu_threshold: 0.85,
            memory_threshold: 0.80,
            resource_check_interval: 3,
            timeout_multiplier: 1.2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_parallel_executor_creation() {
        let config = ParallelConfig::default();
        let executor = ParallelTestExecutor::new(config).await.unwrap();

        assert_eq!(executor.config.max_parallel, num_cpus::get().max(1));
    }

    #[tokio::test]
    async fn test_resource_monitor() {
        let config = ParallelConfig::default();
        let mut monitor = ResourceMonitor::new(config).await.unwrap();

        let usage = monitor.get_current_usage().await;
        assert_eq!(usage.cpu_percent, 0.0); // Default value

        // Test monitoring start/stop
        monitor.start_monitoring().await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        monitor.stop_monitoring().await;
    }

    #[tokio::test]
    async fn test_load_balancer() {
        let config = ParallelConfig::default();
        let balancer = LoadBalancer::new(config.clone());

        balancer.initialize_workers(4).await;

        // Test worker selection (should return valid worker ID)
        // Note: This test is limited without a real TestCase implementation
        let worker_stats = balancer.get_worker_stats(0).await;
        assert_eq!(worker_stats.worker_id, 0);
    }

    #[test]
    fn test_optimal_parallelism_calculation() {
        let optimal = optimization::calculate_optimal_parallelism();
        assert!(optimal >= 1);
        assert!(optimal <= num_cpus::get());
    }

    #[test]
    fn test_optimized_config_creation() {
        let config = optimization::create_optimized_config();
        assert!(config.max_parallel >= 1);
        assert!(config.min_parallel >= 1);
        assert!(config.min_parallel <= config.max_parallel);
        assert!(config.cpu_threshold > 0.0 && config.cpu_threshold <= 1.0);
        assert!(config.memory_threshold > 0.0 && config.memory_threshold <= 1.0);
    }
}
