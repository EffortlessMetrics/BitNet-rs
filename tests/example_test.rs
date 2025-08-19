// Example test demonstrating the testing framework
use std::sync::Arc;
use std::time::Duration;

mod common;

use common::{
    ConsoleReporter, DebugContext, FixtureManager, TestCase, TestConfig, TestError, TestHarness,
    TestMetrics, TestResult, TestSuite, TestTracer, TraceEventType, dev_config, init_logging,
};

/// Example test case implementation
struct ExampleTestCase {
    name: String,
    should_pass: bool,
}

impl ExampleTestCase {
    fn new(name: &str, should_pass: bool) -> Self {
        Self { name: name.to_string(), should_pass }
    }
}

#[async_trait::async_trait]
impl TestCase for ExampleTestCase {
    fn name(&self) -> &str {
        &self.name
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        tracing::debug!("Setting up test: {}", self.name);

        // Simulate some setup work
        tokio::time::sleep(Duration::from_millis(10)).await;

        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        tracing::debug!("Executing test: {}", self.name);

        let start_time = std::time::Instant::now();

        // Simulate test work
        tokio::time::sleep(Duration::from_millis(50)).await;

        let duration = start_time.elapsed();
        let mut metrics = TestMetrics::with_duration(duration);

        // Add some custom metrics
        metrics.add_metric("operations_performed", 42.0);
        metrics.add_metric("data_processed_mb", 1.5);
        metrics.add_assertion();

        if self.should_pass {
            tracing::info!("Test '{}' completed successfully", self.name);
            Ok(metrics)
        } else {
            Err(TestError::assertion("This test was designed to fail"))
        }
    }

    async fn cleanup(&self) -> TestResult<()> {
        tracing::debug!("Cleaning up test: {}", self.name);

        // Simulate cleanup work
        tokio::time::sleep(Duration::from_millis(5)).await;

        Ok(())
    }
}

/// Example test suite implementation
struct ExampleTestSuite {
    name: String,
    test_cases: Vec<Box<dyn TestCase>>,
}

impl ExampleTestSuite {
    fn new() -> Self {
        let test_cases: Vec<Box<dyn TestCase>> = vec![
            Box::new(ExampleTestCase::new("test_basic_functionality", true)),
            Box::new(ExampleTestCase::new("test_edge_case_handling", true)),
            Box::new(ExampleTestCase::new("test_error_conditions", false)), // This one fails
            Box::new(ExampleTestCase::new("test_performance_benchmark", true)),
            Box::new(ExampleTestCase::new("test_memory_usage", true)),
        ];

        Self { name: "Example Test Suite".to_string(), test_cases }
    }
}

impl TestSuite for ExampleTestSuite {
    fn name(&self) -> &str {
        &self.name
    }

    fn test_cases(&self) -> Vec<Box<dyn TestCase>> {
        // Clone the test cases (this is a simplified approach)
        // In a real implementation, you might want to use Arc<dyn TestCase> instead
        vec![
            Box::new(ExampleTestCase::new("test_basic_functionality", true)),
            Box::new(ExampleTestCase::new("test_edge_case_handling", true)),
            Box::new(ExampleTestCase::new("test_error_conditions", false)),
            Box::new(ExampleTestCase::new("test_performance_benchmark", true)),
            Box::new(ExampleTestCase::new("test_memory_usage", true)),
        ]
    }
}

/// Example test with tracing and debugging
struct TracedTestCase {
    name: String,
    tracer: Arc<TestTracer>,
}

impl TracedTestCase {
    fn new(name: &str, tracer: Arc<TestTracer>) -> Self {
        Self { name: name.to_string(), tracer }
    }
}

#[async_trait::async_trait]
impl TestCase for TracedTestCase {
    fn name(&self) -> &str {
        &self.name
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestResult<()> {
        let debug_ctx = DebugContext::new(self.name.clone(), Arc::clone(&self.tracer));

        debug_ctx.trace(TraceEventType::Setup, "Starting test setup".to_string()).await;

        // Simulate setup with tracing
        let setup_scope = debug_ctx.scope("setup");
        setup_scope.trace(TraceEventType::Info, "Initializing test data".to_string()).await;

        tokio::time::sleep(Duration::from_millis(20)).await;

        setup_scope.trace(TraceEventType::Info, "Setup completed".to_string()).await;

        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let debug_ctx = DebugContext::new(self.name.clone(), Arc::clone(&self.tracer));

        debug_ctx.trace(TraceEventType::Execution, "Starting test execution".to_string()).await;

        let start_time = std::time::Instant::now();

        // Simulate some work with detailed tracing
        for i in 1..=3 {
            let step_scope = debug_ctx.scope(&format!("step_{}", i));
            step_scope.trace(TraceEventType::Info, format!("Executing step {}", i)).await;

            tokio::time::sleep(Duration::from_millis(30)).await;

            step_scope
                .trace(
                    TraceEventType::Performance,
                    format!("Step {} completed in {:?}", i, step_scope.elapsed()),
                )
                .await;
        }

        let duration = start_time.elapsed();
        debug_ctx
            .trace(TraceEventType::Performance, format!("Total execution time: {:?}", duration))
            .await;

        let mut metrics = TestMetrics::with_duration(duration);
        metrics.add_metric("traced_operations", 3.0);
        metrics.add_assertion();

        debug_ctx
            .trace(TraceEventType::TestEnd, "Test execution completed successfully".to_string())
            .await;

        Ok(metrics)
    }

    async fn cleanup(&self) -> TestResult<()> {
        let debug_ctx = DebugContext::new(self.name.clone(), Arc::clone(&self.tracer));
        debug_ctx.trace(TraceEventType::Cleanup, "Performing cleanup".to_string()).await;

        tokio::time::sleep(Duration::from_millis(10)).await;

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize configuration
    let config = dev_config();

    // Initialize logging
    init_logging(&config)?;

    tracing::info!("Starting BitNet.rs testing framework example");

    // Create test harness
    let mut harness = TestHarness::new(config).await?;

    // Add console reporter
    harness.add_reporter(Box::new(ConsoleReporter::new(true)));

    // Run basic example test suite
    println!("\n=== Running Basic Example Test Suite ===");
    let basic_suite = ExampleTestSuite::new();
    let basic_result = harness.run_test_suite(basic_suite).await?;

    println!("\nBasic suite results:");
    println!("  Total tests: {}", basic_result.summary.total_tests);
    println!("  Passed: {}", basic_result.summary.passed);
    println!("  Failed: {}", basic_result.summary.failed);
    println!("  Success rate: {:.1}%", basic_result.summary.success_rate);
    println!("  Total duration: {:?}", basic_result.total_duration);

    // Run traced test suite
    println!("\n=== Running Traced Test Suite ===");
    let tracer = Arc::new(TestTracer::new(true));

    struct TracedTestSuite {
        tracer: Arc<TestTracer>,
    }

    impl TestSuite for TracedTestSuite {
        fn name(&self) -> &str {
            "Traced Test Suite"
        }

        fn test_cases(&self) -> Vec<Box<dyn TestCase>> {
            vec![
                Box::new(TracedTestCase::new("traced_test_1", Arc::clone(&self.tracer))),
                Box::new(TracedTestCase::new("traced_test_2", Arc::clone(&self.tracer))),
            ]
        }
    }

    let traced_suite = TracedTestSuite { tracer: Arc::clone(&tracer) };
    let traced_result = harness.run_test_suite(traced_suite).await?;

    println!("\nTraced suite results:");
    println!("  Total tests: {}", traced_result.summary.total_tests);
    println!("  Passed: {}", traced_result.summary.passed);
    println!("  Success rate: {:.1}%", traced_result.summary.success_rate);

    // Display trace information
    println!("\n=== Trace Information ===");
    let all_traces = tracer.get_all_traces().await;
    for (test_name, traces) in all_traces {
        println!("\nTraces for {}:", test_name);
        for (i, trace) in traces.iter().enumerate() {
            println!("  {}: [{}] {}", i + 1, trace.event_type, trace.message);
        }
    }

    // Get overall execution stats
    let stats = harness.get_stats().await;
    println!("\n=== Overall Execution Stats ===");
    println!("  Total suites: {}", stats.total_suites);
    println!("  Total tests: {}", stats.total_tests);
    println!("  Total passed: {}", stats.total_passed);
    println!("  Total failed: {}", stats.total_failed);
    println!("  Overall success rate: {:.1}%", stats.success_rate());
    println!("  Total execution time: {:?}", stats.total_duration);
    println!("  Average per test: {:?}", stats.average_duration());

    tracing::info!("BitNet.rs testing framework example completed");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_example_test_case() {
        let test_case = ExampleTestCase::new("test_example", true);
        assert_eq!(test_case.name(), "test_example");

        // Create a minimal fixture manager for testing
        let config = dev_config();
        let fixtures = FixtureManager::new(&config.fixtures).await.unwrap();

        // Test setup
        assert!(test_case.setup(&fixtures).await.is_ok());

        // Test execution (should pass)
        let result = test_case.execute().await;
        assert!(result.is_ok());

        let metrics = result.unwrap();
        assert!(metrics.wall_time > Duration::ZERO);
        assert_eq!(metrics.assertions, 1);
        assert!(metrics.custom_metrics.contains_key("operations_performed"));

        // Test cleanup
        assert!(test_case.cleanup().await.is_ok());
    }

    #[tokio::test]
    async fn test_failing_test_case() {
        let test_case = ExampleTestCase::new("test_failing", false);
        let config = dev_config();
        let fixtures = FixtureManager::new(&config.fixtures).await.unwrap();

        assert!(test_case.setup(&fixtures).await.is_ok());

        // This should fail
        let result = test_case.execute().await;
        assert!(result.is_err());

        assert!(test_case.cleanup().await.is_ok());
    }

    #[tokio::test]
    async fn test_example_test_suite() {
        let suite = ExampleTestSuite::new();
        assert_eq!(suite.name(), "Example Test Suite");

        let test_cases = suite.test_cases();
        assert_eq!(test_cases.len(), 5);

        // Check that we have the expected test names
        let names: Vec<&str> = test_cases.iter().map(|tc| tc.name()).collect();
        assert!(names.contains(&"test_basic_functionality"));
        assert!(names.contains(&"test_edge_case_handling"));
        assert!(names.contains(&"test_error_conditions"));
        assert!(names.contains(&"test_performance_benchmark"));
        assert!(names.contains(&"test_memory_usage"));
    }
}
