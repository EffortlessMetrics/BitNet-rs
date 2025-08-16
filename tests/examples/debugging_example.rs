use std::collections::HashMap;
use std::time::Duration;
use tokio::time::sleep;

use tests::common::{
use bitnet_tests::units::{BYTES_PER_KB, BYTES_PER_MB, BYTES_PER_GB};
    config::TestConfig,
    debug_integration::{create_debug_harness, debug_config_from_env, DebugTestReporter},
    debugging::DebugConfig,
    errors::{TestError, TestOpResult},
    fixtures::FixtureManager,
    harness::{TestCase, TestSuite},
    results::TestMetrics,
};

/// Example test case that demonstrates debugging features
struct DebuggableTestCase {
    name: String,
    should_fail: bool,
    duration: Duration,
    memory_usage: u64,
}

impl DebuggableTestCase {
    fn new(name: &str, should_fail: bool, duration: Duration, memory_usage: u64) -> Self {
        Self { name: name.to_string(), should_fail, duration, memory_usage }
    }
}

#[async_trait::async_trait]
impl TestCase for DebuggableTestCase {
    fn name(&self) -> &str {
        &self.name
    }

    async fn setup(&self, _fixtures: &FixtureManager) -> TestOpResult<()> {
        println!("Setting up test: {}", self.name);

        // Simulate setup work
        sleep(Duration::from_millis(100)).await;

        if self.should_fail && self.name.contains("setup_fail") {
            return Err(TestError::setup("Simulated setup failure for debugging demo"));
        }

        Ok(())
    }

    async fn execute(&self) -> TestOpResult<TestMetrics> {
        println!("Executing test: {}", self.name);

        // Simulate test work
        sleep(self.duration).await;

        // Simulate memory allocation
        let _memory_simulation = vec![0u8; self.memory_usage as usize];

        if self.should_fail && !self.name.contains("setup_fail") {
            return Err(TestError::execution(format!(
                "Simulated execution failure for debugging demo: {}",
                self.name
            )));
        }

        Ok(TestMetrics {
            wall_time: self.duration,
            memory_peak: Some(self.memory_usage),
            memory_average: Some(self.memory_usage / 2),
            cpu_time: Some(self.duration / 2),
            custom_metrics: [
                ("simulated_metric".to_string(), 42.0),
                ("test_complexity".to_string(), 3.14),
            ]
            .into(),
        })
    }

    async fn cleanup(&self) -> TestOpResult<()> {
        println!("Cleaning up test: {}", self.name);

        // Simulate cleanup work
        sleep(Duration::from_millis(50)).await;

        Ok(())
    }

    fn metadata(&self) -> HashMap<String, String> {
        [
            ("category".to_string(), "debugging_example".to_string()),
            ("expected_duration".to_string(), format!("{:?}", self.duration)),
            ("expected_memory".to_string(), self.memory_usage.to_string()),
        ]
        .into()
    }
}

/// Example test suite for debugging demonstration
struct DebuggingExampleSuite {
    test_cases: Vec<Box<dyn TestCase>>,
}

impl DebuggingExampleSuite {
    fn new() -> Self {
        let test_cases: Vec<Box<dyn TestCase>> = vec![
            Box::new(DebuggableTestCase::new(
                "fast_passing_test",
                false,
                Duration::from_millis(100),
                BYTES_PER_MB, // 1MB
            )),
            Box::new(DebuggableTestCase::new(
                "slow_passing_test",
                false,
                Duration::from_secs(2),
                10 * BYTES_PER_MB, // 10MB
            )),
            Box::new(DebuggableTestCase::new(
                "memory_intensive_test",
                false,
                Duration::from_millis(500),
                100 * BYTES_PER_MB, // 100MB
            )),
            Box::new(DebuggableTestCase::new(
                "failing_execution_test",
                true,
                Duration::from_millis(200),
                5 * BYTES_PER_MB, // 5MB
            )),
            Box::new(DebuggableTestCase::new(
                "setup_fail_test",
                true,
                Duration::from_millis(100),
                BYTES_PER_MB, // 1MB
            )),
        ];

        Self { test_cases }
    }
}

impl TestSuite for DebuggingExampleSuite {
    fn name(&self) -> &str {
        "debugging_example_suite"
    }

    fn test_cases(&self) -> Vec<Box<dyn TestCase>> {
        self.test_cases
            .iter()
            .map(|tc| {
                // Create a new instance for each call
                let name = tc.name();
                if name.contains("fast_passing") {
                    Box::new(DebuggableTestCase::new(
                        name,
                        false,
                        Duration::from_millis(100),
                        BYTES_PER_MB,
                    )) as Box<dyn TestCase>
                } else if name.contains("slow_passing") {
                    Box::new(DebuggableTestCase::new(
                        name,
                        false,
                        Duration::from_secs(2),
                        10 * BYTES_PER_MB,
                    )) as Box<dyn TestCase>
                } else if name.contains("memory_intensive") {
                    Box::new(DebuggableTestCase::new(
                        name,
                        false,
                        Duration::from_millis(500),
                        100 * BYTES_PER_MB,
                    )) as Box<dyn TestCase>
                } else if name.contains("failing_execution") {
                    Box::new(DebuggableTestCase::new(
                        name,
                        true,
                        Duration::from_millis(200),
                        5 * BYTES_PER_MB,
                    )) as Box<dyn TestCase>
                } else if name.contains("setup_fail") {
                    Box::new(DebuggableTestCase::new(
                        name,
                        true,
                        Duration::from_millis(100),
                        BYTES_PER_MB,
                    )) as Box<dyn TestCase>
                } else {
                    Box::new(DebuggableTestCase::new(
                        name,
                        false,
                        Duration::from_millis(100),
                        BYTES_PER_MB,
                    )) as Box<dyn TestCase>
                }
            })
            .collect()
    }

    fn metadata(&self) -> HashMap<String, String> {
        [
            ("purpose".to_string(), "Demonstrate debugging capabilities".to_string()),
            ("test_count".to_string(), self.test_cases.len().to_string()),
        ]
        .into()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 BitNet.rs Debugging Example");
    println!("===============================\n");

    // Create test configuration
    let test_config = TestConfig {
        max_parallel_tests: 2,
        test_timeout: Duration::from_secs(30),
        ..Default::default()
    };

    // Create debug configuration
    let debug_config = DebugConfig {
        enabled: true,
        capture_stack_traces: true,
        capture_environment: true,
        capture_system_info: true,
        verbose_logging: true,
        save_debug_artifacts: true,
        max_debug_files: 50,
        debug_output_dir: std::path::PathBuf::from("tests/debug/example"),
    };

    println!("📋 Configuration:");
    println!("  Debug enabled: {}", debug_config.enabled);
    println!("  Verbose logging: {}", debug_config.verbose_logging);
    println!("  Output directory: {}", debug_config.debug_output_dir.display());
    println!("  Max parallel tests: {}", test_config.max_parallel_tests);
    println!();

    // Create debug-enabled test harness
    let debug_harness = create_debug_harness(test_config, Some(debug_config)).await?;
    let debugger = debug_harness.debugger();

    // Add debug reporter
    let mut harness = debug_harness;
    // Note: We can't easily add reporters to the wrapped harness in this design
    // In a real implementation, we'd need to modify the harness creation

    // Create test suite
    let suite = DebuggingExampleSuite::new();

    println!("🚀 Running test suite with debugging enabled...\n");

    // Run the test suite
    let result = harness.run_test_suite_with_debug(&suite).await?;

    println!("\n📊 Test Results:");
    println!("  Total tests: {}", result.summary.total_tests);
    println!("  Passed: {}", result.summary.passed);
    println!("  Failed: {}", result.summary.failed);
    println!("  Success rate: {:.1}%", result.summary.success_rate);
    println!("  Total duration: {:?}", result.total_duration);

    // Generate and display debug report
    println!("\n🔍 Generating debug report...");
    let debug_report = debugger.generate_debug_report().await?;
    let report_path = debugger.save_debug_report(&debug_report).await?;

    println!("✅ Debug report saved to: {}", report_path.display());

    // Generate troubleshooting guide for failed tests
    if result.summary.failed > 0 {
        println!("\n📖 Generating troubleshooting guide...");
        let guide = debugger.generate_troubleshooting_guide().await?;
        let guide_path = report_path.parent().unwrap().join("troubleshooting_guide.md");
        tokio::fs::write(&guide_path, guide).await?;
        println!("✅ Troubleshooting guide saved to: {}", guide_path.display());
    }

    // Display summary of debug information
    println!("\n📈 Debug Summary:");
    println!("  Session ID: {}", debug_report.session_id);
    println!("  Peak memory: {} MB", debug_report.performance_summary.peak_memory / (BYTES_PER_MB));
    println!(
        "  Average test duration: {:?}",
        debug_report.performance_summary.average_test_duration
    );
    println!("  Error reports: {}", debug_report.error_count);
    println!("  Debug artifacts: {}", debug_report.artifacts.len());

    if !debug_report.recommendations.is_empty() {
        println!("\n💡 Recommendations:");
        for (i, rec) in debug_report.recommendations.iter().enumerate() {
            println!("  {}. {}", i + 1, rec);
        }
    }

    // Show how to use the debug CLI
    println!("\n🛠️  Debug CLI Usage:");
    println!("  To analyze this report interactively:");
    println!("  cargo run --example debug_cli_example");
    println!("  Then use: analyze {}", report_path.display());

    println!("\n✨ Debugging example completed!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_debuggable_test_case() {
        let test_case =
            DebuggableTestCase::new("test_case", false, Duration::from_millis(100), 1024);

        let fixtures = FixtureManager::new(&Default::default()).await.unwrap();

        assert!(test_case.setup(&fixtures).await.is_ok());
        assert!(test_case.execute().await.is_ok());
        assert!(test_case.cleanup().await.is_ok());
    }

    #[tokio::test]
    async fn test_failing_test_case() {
        let test_case =
            DebuggableTestCase::new("failing_test", true, Duration::from_millis(100), 1024);

        let fixtures = FixtureManager::new(&Default::default()).await.unwrap();

        assert!(test_case.setup(&fixtures).await.is_ok());
        assert!(test_case.execute().await.is_err());
        assert!(test_case.cleanup().await.is_ok());
    }

    #[test]
    fn test_debugging_suite_creation() {
        let suite = DebuggingExampleSuite::new();
        assert_eq!(suite.name(), "debugging_example_suite");
        assert_eq!(suite.test_cases().len(), 5);
    }
}
