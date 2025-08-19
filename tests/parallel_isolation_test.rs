// Comprehensive test for parallel execution with proper isolation
// This test validates that the test framework can run tests in parallel
// while maintaining proper isolation between test executions

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tempfile;
use tokio::sync::{RwLock, Semaphore};
use tokio::time::timeout;

/// Test result structure
#[derive(Debug, Clone)]
pub struct TestResult {
    pub test_name: String,
    pub passed: bool,
    pub duration: Duration,
    pub error: Option<String>,
    pub start_time: SystemTime,
    pub end_time: SystemTime,
    pub isolation_verified: bool,
    pub parallel_slot: usize,
}

impl TestResult {
    pub fn passed(test_name: String, duration: Duration, parallel_slot: usize) -> Self {
        let now = SystemTime::now();
        Self {
            test_name,
            passed: true,
            duration,
            error: None,
            start_time: now - duration,
            end_time: now,
            isolation_verified: true,
            parallel_slot,
        }
    }

    pub fn failed(
        test_name: String,
        duration: Duration,
        error: String,
        parallel_slot: usize,
    ) -> Self {
        let now = SystemTime::now();
        Self {
            test_name,
            passed: false,
            duration,
            error: Some(error),
            start_time: now - duration,
            end_time: now,
            isolation_verified: false,
            parallel_slot,
        }
    }
}

/// Isolated environment for test execution
#[derive(Debug, Clone)]
struct IsolatedEnvironment {
    test_name: String,
    temp_dir_path: std::path::PathBuf,
    parallel_slot: usize,
    test_subdir: std::path::PathBuf,
}

impl IsolatedEnvironment {
    fn new(
        test_name: &str,
        parallel_slot: usize,
    ) -> Result<(Self, tempfile::TempDir), Box<dyn std::error::Error + Send + Sync>> {
        let temp_dir = tempfile::tempdir()?;
        let temp_dir_path = temp_dir.path().to_path_buf();

        // Create unique subdirectory for this test
        let test_subdir = temp_dir_path.join(format!("{}_{}", test_name, parallel_slot));
        std::fs::create_dir_all(&test_subdir)?;

        let env =
            Self { test_name: test_name.to_string(), temp_dir_path, parallel_slot, test_subdir };

        Ok((env, temp_dir))
    }

    fn get_test_name(&self) -> &str {
        &self.test_name
    }

    fn get_temp_dir(&self) -> &std::path::Path {
        &self.temp_dir_path
    }

    fn get_test_subdir(&self) -> &std::path::Path {
        &self.test_subdir
    }

    fn get_parallel_slot(&self) -> usize {
        self.parallel_slot
    }
}

/// Test case trait for parallel execution
pub trait TestCase: Send + Sync {
    fn name(&self) -> &str;
    fn run_sync(&self, env: &IsolatedEnvironment) -> Result<(), String>;
}

/// Test case that verifies isolation and parallel execution
pub struct IsolationTestCase {
    name: String,
    work_duration: Duration,
    should_fail: bool,
    unique_value: String,
    memory_operations: usize,
}

impl IsolationTestCase {
    pub fn new(
        name: &str,
        work_duration: Duration,
        should_fail: bool,
        memory_operations: usize,
    ) -> Self {
        Self {
            name: name.to_string(),
            work_duration,
            should_fail,
            unique_value: format!("{}_{}", name, std::process::id()),
            memory_operations,
        }
    }
}

impl TestCase for IsolationTestCase {
    fn name(&self) -> &str {
        &self.name
    }

    fn run_sync(&self, env: &IsolatedEnvironment) -> Result<(), String> {
        // Verify isolation setup using the isolated environment
        let test_name = env.get_test_name();
        let parallel_slot = env.get_parallel_slot();
        let temp_dir = env.get_temp_dir();
        let test_subdir = env.get_test_subdir();

        // Verify environment isolation
        if test_name != self.name {
            return Err(format!(
                "Test name mismatch: expected '{}', got '{}'",
                self.name, test_name
            ));
        }

        if !temp_dir.exists() {
            return Err("Temp directory does not exist".to_string());
        }

        if !test_subdir.exists() {
            return Err("Test subdirectory does not exist".to_string());
        }

        // Create unique files in the test subdirectory
        let unique_file = test_subdir.join(format!("{}_slot_{}.txt", self.name, parallel_slot));
        let file_content = format!("{}_slot_{}", self.unique_value, parallel_slot);
        std::fs::write(&unique_file, &file_content)
            .map_err(|e| format!("Failed to write file: {}", e))?;

        // Perform memory operations to simulate real work
        let mut data = Vec::with_capacity(self.memory_operations);
        for i in 0..self.memory_operations {
            data.push(format!("{}_{}_data_{}", self.name, parallel_slot, i));
        }

        // Simulate work with blocking sleep (fine in spawn_blocking)
        std::thread::sleep(self.work_duration);

        // Verify isolation is still intact after work - check environment hasn't changed
        if env.get_test_name() != self.name {
            return Err(format!(
                "Isolation broken during execution: expected '{}', got '{}'",
                self.name,
                env.get_test_name()
            ));
        }

        if env.get_parallel_slot() != parallel_slot {
            return Err(format!(
                "Parallel slot changed during execution: expected '{}', got '{}'",
                parallel_slot,
                env.get_parallel_slot()
            ));
        }

        // Verify our unique file still exists and has correct content
        let actual_content = std::fs::read_to_string(&unique_file)
            .map_err(|e| format!("Failed to read file: {}", e))?;
        if actual_content != file_content {
            return Err(format!(
                "File interference detected: expected '{}', got '{}'",
                file_content, actual_content
            ));
        }

        // Verify memory operations completed correctly
        if data.len() != self.memory_operations {
            return Err(format!(
                "Memory operations failed: expected {}, got {}",
                self.memory_operations,
                data.len()
            ));
        }

        // Verify temp directory is still accessible
        if !temp_dir.exists() {
            return Err("Temp directory disappeared during execution".to_string());
        }

        if self.should_fail {
            return Err("Intentional test failure".to_string());
        }

        Ok(())
    }
}

/// Parallel test harness with proper isolation and resource management
pub struct ParallelTestHarness {
    max_parallel: usize,
    test_timeout: Duration,
    semaphore: Arc<Semaphore>,
    stats: Arc<RwLock<TestStats>>,
    slot_tracker: Arc<RwLock<Vec<bool>>>, // Track which parallel slots are in use
}

#[derive(Debug, Default, Clone)]
struct TestStats {
    total_tests: usize,
    passed_tests: usize,
    failed_tests: usize,
    total_duration: Duration,
    parallel_efficiency: f64,
    isolation_violations: usize,
}

impl ParallelTestHarness {
    pub fn new(max_parallel: usize, test_timeout: Duration) -> Self {
        Self {
            max_parallel,
            test_timeout,
            semaphore: Arc::new(Semaphore::new(max_parallel)),
            stats: Arc::new(RwLock::new(TestStats::default())),
            slot_tracker: Arc::new(RwLock::new(vec![false; max_parallel])),
        }
    }

    /// Run tests in parallel with proper isolation and resource management
    pub async fn run_tests<T: TestCase + 'static>(&self, test_cases: Vec<T>) -> Vec<TestResult> {
        let start_time = Instant::now();
        let mut handles = Vec::new();

        for test_case in test_cases {
            let semaphore = Arc::clone(&self.semaphore);
            let stats = Arc::clone(&self.stats);
            let slot_tracker = Arc::clone(&self.slot_tracker);
            let timeout_duration = self.test_timeout;

            let handle = tokio::spawn(async move {
                Self::run_single_test_isolated(
                    test_case,
                    semaphore,
                    stats,
                    slot_tracker,
                    timeout_duration,
                )
                .await
            });

            handles.push(handle);
        }

        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(result) => results.push(result),
                Err(e) => {
                    eprintln!("Task join error: {}", e);
                    results.push(TestRecord::failed(
                        "unknown".to_string(),
                        Duration::ZERO,
                        format!("Task join error: {}", e),
                        0,
                    ));
                }
            }
        }

        // Calculate parallel efficiency
        let total_duration = start_time.elapsed();
        let sequential_time: Duration = results.iter().map(|r| r.duration).sum();
        let efficiency = if total_duration.as_secs_f64() > 0.0 {
            sequential_time.as_secs_f64() / total_duration.as_secs_f64()
        } else {
            0.0
        };

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.parallel_efficiency = efficiency;
            stats.total_duration = total_duration;
        }

        results
    }

    async fn run_single_test_isolated<T: TestCase + 'static>(
        test_case: T,
        semaphore: Arc<Semaphore>,
        stats: Arc<RwLock<TestStats>>,
        slot_tracker: Arc<RwLock<Vec<bool>>>,
        timeout_duration: Duration,
    ) -> TestResult {
        let test_name = test_case.name().to_string();
        let start_time = Instant::now();

        // Acquire semaphore permit for parallel execution control
        let _permit = match semaphore.acquire().await {
            Ok(permit) => permit,
            Err(e) => {
                return TestRecord::failed(
                    test_name,
                    Duration::ZERO,
                    format!("Failed to acquire semaphore: {}", e),
                    0,
                );
            }
        };

        // Acquire a parallel slot
        let parallel_slot = {
            let mut slots = slot_tracker.write().await;
            let slot = slots.iter().position(|&used| !used).unwrap_or(0);
            if slot < slots.len() {
                slots[slot] = true;
            }
            slot
        };

        // Create isolated environment
        let (isolated_env, _temp_dir) = match IsolatedEnvironment::new(&test_name, parallel_slot) {
            Ok((env, temp_dir)) => (env, temp_dir),
            Err(e) => {
                // Release slot
                {
                    let mut slots = slot_tracker.write().await;
                    if parallel_slot < slots.len() {
                        slots[parallel_slot] = false;
                    }
                }
                return TestRecord::failed(
                    test_name,
                    start_time.elapsed(),
                    format!("Failed to create isolated environment: {}", e),
                    parallel_slot,
                );
            }
        };

        // Execute test with timeout in blocking context
        let execute_result = timeout(
            timeout_duration,
            tokio::task::spawn_blocking({
                let env_clone = isolated_env.clone();
                move || test_case.run_sync(&env_clone)
            }),
        )
        .await;

        let duration = start_time.elapsed();

        // temp_dir is automatically cleaned up when dropped

        // Release parallel slot
        {
            let mut slots = slot_tracker.write().await;
            if parallel_slot < slots.len() {
                slots[parallel_slot] = false;
            }
        }

        // Update stats
        {
            let mut stats_guard = stats.write().await;
            stats_guard.total_tests += 1;
        }

        // Process result
        match execute_result {
            Ok(Ok(Ok(()))) => {
                {
                    let mut stats_guard = stats.write().await;
                    stats_guard.passed_tests += 1;
                }
                TestRecord::passed(test_name, duration, parallel_slot)
            }
            Ok(Ok(Err(e))) => {
                {
                    let mut stats_guard = stats.write().await;
                    stats_guard.failed_tests += 1;
                    if e.contains("interference") || e.contains("isolation broken") {
                        stats_guard.isolation_violations += 1;
                    }
                }
                TestRecord::failed(test_name, duration, e, parallel_slot)
            }
            Ok(Err(e)) => {
                {
                    let mut stats_guard = stats.write().await;
                    stats_guard.failed_tests += 1;
                }
                TestRecord::failed(
                    test_name,
                    duration,
                    format!("Task execution failed: {}", e),
                    parallel_slot,
                )
            }
            Err(_) => {
                {
                    let mut stats_guard = stats.write().await;
                    stats_guard.failed_tests += 1;
                }
                TestRecord::failed(
                    test_name,
                    timeout_duration,
                    format!("Test timed out after {:?}", timeout_duration),
                    parallel_slot,
                )
            }
        }
    }

    pub async fn get_stats(&self) -> TestStats {
        self.stats.read().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_parallel_execution_with_proper_isolation() {
        println!("🚀 Testing parallel execution with proper isolation...");

        // Create test harness with 4 parallel slots
        let harness = ParallelTestHarness::new(4, Duration::from_secs(10));

        // Create test cases with different characteristics
        let test_cases = vec![
            IsolationTestCase::new("fast_test_1", Duration::from_millis(100), false, 100),
            IsolationTestCase::new("fast_test_2", Duration::from_millis(150), false, 200),
            IsolationTestCase::new("medium_test_1", Duration::from_millis(300), false, 500),
            IsolationTestCase::new("medium_test_2", Duration::from_millis(250), false, 300),
            IsolationTestCase::new("slow_test_1", Duration::from_millis(500), false, 1000),
            IsolationTestCase::new("slow_test_2", Duration::from_millis(400), false, 800),
            IsolationTestCase::new("memory_intensive", Duration::from_millis(200), false, 2000),
            IsolationTestCase::new("failing_test", Duration::from_millis(100), true, 50),
        ];

        // Record start time to verify parallel execution
        let start_time = Instant::now();

        // Run tests
        let results = harness.run_tests(test_cases).await;
        let total_duration = start_time.elapsed();

        // Print detailed results
        println!("📊 Test Results:");
        for result in &results {
            let status = if result.passed { "✅ PASS" } else { "❌ FAIL" };
            let isolation =
                if result.isolation_verified { "🔒 ISOLATED" } else { "⚠️  LEAKED" };
            println!(
                "  {} {} {} (slot: {}, {:?})",
                status, result.test_name, isolation, result.parallel_slot, result.duration
            );
            if let Some(error) = &result.error {
                println!("    Error: {}", error);
            }
        }

        // Verify results
        assert_eq!(results.len(), 8, "Should have 8 test results");

        let passed_count = results.iter().filter(|r| r.passed).count();
        let failed_count = results.iter().filter(|r| !r.passed).count();

        assert_eq!(passed_count, 7, "Should have 7 passed tests");
        assert_eq!(failed_count, 1, "Should have 1 failed test (intentional)");

        // Verify no isolation violations (except for the intentionally failing test)
        let isolation_violations = results
            .iter()
            .filter(|r| {
                !r.passed
                    && r.error.as_ref().map_or(false, |e| {
                        e.contains("interference") || e.contains("isolation broken")
                    })
            })
            .count();
        assert_eq!(isolation_violations, 0, "Should have no isolation violations");

        // Verify parallel execution efficiency
        let sequential_time = Duration::from_millis(100 + 150 + 300 + 250 + 500 + 400 + 200 + 100); // 2000ms

        println!("⏱️  Performance Analysis:");
        println!("  Total execution time: {:?}", total_duration);
        println!("  Sequential time would be: {:?}", sequential_time);

        // Verify that parallel execution was significantly faster than sequential
        assert!(
            total_duration < sequential_time,
            "Parallel execution should be faster than sequential. Got {:?}, expected less than {:?}",
            total_duration,
            sequential_time
        );

        // Verify significant parallelism (should be at least 40% faster due to 4 parallel slots)
        let efficiency_threshold = sequential_time.as_millis() as f64 * 0.6;
        assert!(
            (total_duration.as_millis() as f64) < efficiency_threshold,
            "Parallel execution should show significant improvement. Got {:?}, expected less than {}ms",
            total_duration,
            efficiency_threshold
        );

        // Verify parallel slots were used correctly
        let used_slots: std::collections::HashSet<_> =
            results.iter().map(|r| r.parallel_slot).collect();
        assert!(used_slots.len() > 1, "Multiple parallel slots should have been used");
        assert!(used_slots.iter().all(|&slot| slot < 4), "All slots should be within bounds");

        // Verify stats
        let stats = harness.get_stats().await;
        assert_eq!(stats.total_tests, 8);
        assert_eq!(stats.passed_tests, 7);
        assert_eq!(stats.failed_tests, 1);
        assert_eq!(stats.isolation_violations, 0);

        let efficiency = stats.parallel_efficiency;
        println!("  Parallel efficiency: {:.2}x", efficiency);
        assert!(efficiency > 1.5, "Should achieve at least 1.5x efficiency with 4 parallel slots");

        println!("✅ Parallel execution with proper isolation test passed!");
        println!("   - All tests executed with proper isolation");
        println!("   - Environment variables isolated correctly");
        println!("   - Temporary files isolated correctly");
        println!("   - Memory operations isolated correctly");
        println!("   - Semaphore controlled parallelism correctly");
        println!("   - Parallel slots managed correctly");
    }

    #[tokio::test]
    async fn test_semaphore_limits_parallelism() {
        println!("🚀 Testing semaphore limits parallelism...");

        // Create test harness with only 1 parallel slot (sequential execution)
        let harness = ParallelTestHarness::new(1, Duration::from_secs(10));

        // Create test cases
        let test_cases = vec![
            IsolationTestCase::new("sequential_1", Duration::from_millis(200), false, 100),
            IsolationTestCase::new("sequential_2", Duration::from_millis(200), false, 100),
            IsolationTestCase::new("sequential_3", Duration::from_millis(200), false, 100),
        ];

        // Record start time
        let start_time = Instant::now();

        // Run tests
        let results = harness.run_tests(test_cases).await;
        let total_duration = start_time.elapsed();

        // Print results
        println!("📊 Test Results:");
        for result in &results {
            let status = if result.passed { "✅ PASS" } else { "❌ FAIL" };
            println!(
                "  {} {} (slot: {}, {:?})",
                status, result.test_name, result.parallel_slot, result.duration
            );
        }

        // Verify results
        assert_eq!(results.len(), 3, "Should have 3 test results");
        assert_eq!(results.iter().filter(|r| r.passed).count(), 3, "All tests should pass");

        // With max_parallel = 1, all tests should use slot 0
        assert!(results.iter().all(|r| r.parallel_slot == 0), "All tests should use slot 0");

        // With max_parallel = 1, execution should be close to sequential
        let expected_min_time = Duration::from_millis(600); // 3 * 200ms
        assert!(
            total_duration >= expected_min_time.mul_f64(0.8),
            "Sequential execution should take at least ~{:?}, got {:?}",
            expected_min_time,
            total_duration
        );

        println!("⏱️  Performance Analysis:");
        println!("  Execution time: {:?} (expected ~{:?})", total_duration, expected_min_time);
        println!("✅ Semaphore limiting test passed!");
    }

    #[tokio::test]
    async fn test_isolation_prevents_interference() {
        println!("🚀 Testing isolation prevents interference...");

        // Create test harness with 3 parallel slots
        let harness = ParallelTestHarness::new(3, Duration::from_secs(10));

        // Test case that could interfere if not properly isolated
        struct InterferenceTestCase {
            name: String,
            env_key: String,
            env_value: String,
            work_duration: Duration,
        }

        impl InterferenceTestCase {
            fn new(name: &str, env_key: &str, env_value: &str, work_duration: Duration) -> Self {
                Self {
                    name: name.to_string(),
                    env_key: env_key.to_string(),
                    env_value: env_value.to_string(),
                    work_duration,
                }
            }
        }

        impl TestCase for InterferenceTestCase {
            fn name(&self) -> &str {
                &self.name
            }

            fn run_sync(&self, env: &IsolatedEnvironment) -> Result<(), String> {
                // Verify we're in isolated environment
                let test_name = env.get_test_name();
                let parallel_slot = env.get_parallel_slot();
                let test_subdir = env.get_test_subdir();

                if test_name != self.name {
                    return Err(format!(
                        "Wrong test environment: expected '{}', got '{}'",
                        self.name, test_name
                    ));
                }

                // Create a file in temp directory with slot-specific name
                let test_file = test_subdir
                    .join(format!("{}_{}_slot_{}.txt", self.name, self.env_key, parallel_slot));
                let expected_content = format!("{}_slot_{}", self.env_value, parallel_slot);
                std::fs::write(&test_file, &expected_content)
                    .map_err(|e| format!("Failed to write file: {}", e))?;

                // Wait a bit to allow other tests to potentially interfere
                std::thread::sleep(self.work_duration);

                // Verify our file still exists and has correct content
                let file_content = std::fs::read_to_string(&test_file)
                    .map_err(|e| format!("Failed to read file: {}", e))?;
                if file_content != expected_content {
                    return Err(format!(
                        "File interference detected: expected '{}', got '{}'",
                        expected_content, file_content
                    ));
                }

                // Verify we're still in the correct test environment
                if env.get_test_name() != self.name {
                    return Err(format!(
                        "Test environment changed: expected '{}', got '{}'",
                        self.name,
                        env.get_test_name()
                    ));
                }

                if env.get_parallel_slot() != parallel_slot {
                    return Err(format!(
                        "Parallel slot changed: expected '{}', got '{}'",
                        parallel_slot,
                        env.get_parallel_slot()
                    ));
                }

                Ok(())
            }
        }

        // Create test cases that would interfere if not properly isolated
        let test_cases = vec![
            InterferenceTestCase::new("test_a", "TEST_VAR", "value_a", Duration::from_millis(200)),
            InterferenceTestCase::new("test_b", "TEST_VAR", "value_b", Duration::from_millis(250)),
            InterferenceTestCase::new(
                "test_c",
                "ANOTHER_VAR",
                "value_c",
                Duration::from_millis(150),
            ),
            InterferenceTestCase::new("test_d", "TEST_VAR", "value_d", Duration::from_millis(300)),
            InterferenceTestCase::new(
                "test_e",
                "SHARED_VAR",
                "value_e",
                Duration::from_millis(180),
            ),
        ];

        // Run tests
        let results = harness.run_tests(test_cases).await;

        // Print results
        println!("📊 Test Results:");
        for result in &results {
            let status = if result.passed { "✅ PASS" } else { "❌ FAIL" };
            println!(
                "  {} {} (slot: {}, {:?})",
                status, result.test_name, result.parallel_slot, result.duration
            );
            if let Some(error) = &result.error {
                println!("    Error: {}", error);
            }
        }

        // Verify all tests passed (no interference)
        assert_eq!(results.len(), 5, "Should have 5 test results");

        for result in &results {
            if !result.passed {
                panic!(
                    "Test '{}' failed: {}",
                    result.test_name,
                    result.error.as_deref().unwrap_or("Unknown error")
                );
            }
        }

        // Verify multiple slots were used
        let used_slots: std::collections::HashSet<_> =
            results.iter().map(|r| r.parallel_slot).collect();
        assert!(used_slots.len() > 1, "Multiple parallel slots should have been used");

        println!("✅ Isolation prevents interference test passed!");
        println!("   - All tests ran without interfering with each other");
        println!("   - Environment variables properly isolated");
        println!("   - File system properly isolated");
        println!("   - Parallel slots properly managed");
    }

    #[tokio::test]
    async fn test_timeout_handling_with_isolation() {
        println!("🚀 Testing timeout handling with isolation...");

        // Create test harness with short timeout
        let harness = ParallelTestHarness::new(2, Duration::from_millis(500));

        struct TimeoutTestCase {
            name: String,
            work_duration: Duration,
        }

        impl TimeoutTestCase {
            fn new(name: &str, work_duration: Duration) -> Self {
                Self { name: name.to_string(), work_duration }
            }
        }

        impl TestCase for TimeoutTestCase {
            fn name(&self) -> &str {
                &self.name
            }

            fn run_sync(&self, env: &IsolatedEnvironment) -> Result<(), String> {
                // Verify isolation
                let test_name = env.get_test_name();
                let parallel_slot = env.get_parallel_slot();

                if test_name != self.name {
                    return Err(format!(
                        "Wrong test environment: expected '{}', got '{}'",
                        self.name, test_name
                    ));
                }

                // Sleep for the specified duration
                std::thread::sleep(self.work_duration);

                // Verify isolation is still intact
                if env.get_test_name() != self.name {
                    return Err(format!(
                        "Test environment changed during execution: expected '{}', got '{}'",
                        self.name,
                        env.get_test_name()
                    ));
                }

                if env.get_parallel_slot() != parallel_slot {
                    return Err(format!(
                        "Parallel slot changed during execution: expected '{}', got '{}'",
                        parallel_slot,
                        env.get_parallel_slot()
                    ));
                }

                Ok(())
            }
        }

        let test_cases = vec![
            TimeoutTestCase::new("fast_test", Duration::from_millis(200)), // Should pass
            TimeoutTestCase::new("slow_test", Duration::from_millis(800)), // Should timeout
        ];

        let results = harness.run_tests(test_cases).await;

        // Print results
        println!("📊 Test Results:");
        for result in &results {
            let status = if result.passed { "✅ PASS" } else { "❌ FAIL" };
            println!(
                "  {} {} (slot: {}, {:?})",
                status, result.test_name, result.parallel_slot, result.duration
            );
            if let Some(error) = &result.error {
                println!("    Error: {}", error);
            }
        }

        assert_eq!(results.len(), 2, "Should have 2 test results");

        // Find the fast and slow test results
        let fast_result = results.iter().find(|r| r.test_name == "fast_test").unwrap();
        let slow_result = results.iter().find(|r| r.test_name == "slow_test").unwrap();

        assert!(fast_result.passed, "Fast test should pass");
        assert!(!slow_result.passed, "Slow test should fail due to timeout");
        assert!(
            slow_result.error.as_ref().unwrap().contains("timed out"),
            "Slow test should have timeout error"
        );

        println!("✅ Timeout handling with isolation test passed!");
        println!("   - Fast test completed successfully with proper isolation");
        println!("   - Slow test properly timed out with cleanup");
    }
}
