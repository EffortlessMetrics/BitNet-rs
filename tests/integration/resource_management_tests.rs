//! # Resource Management Integration Tests
//!
//! These tests verify that resources are allocated and released correctly and
//! that error conditions are surfaced through the [`TestResult`] harness. Each
//! case documents the expected behaviour and records simple metrics that can be
//! inspected by the reporting system.

use super::*;
use crate::common::harness::FixtureCtx;
use crate::{TestCase, TestError, TestMetrics, TestResult, TestSuite};
use async_trait::async_trait;
use std::time::Instant;
use tokio::fs::{self, File};
use tokio::io::AsyncWriteExt;
use tracing::{debug, info};

/// Test suite exercising basic resource management scenarios.
pub struct ResourceManagementTestSuite;

impl ResourceManagementTestSuite {
    pub fn new() -> Self {
        Self
    }
}

impl TestSuite for ResourceManagementTestSuite {
    fn name(&self) -> &str {
        "Resource Management Integration Tests"
    }

    fn test_cases(&self) -> Vec<Box<dyn TestCase>> {
        vec![
            Box::new(MemoryAllocationTest),
            Box::new(FileCleanupTest),
            Box::new(ErrorPropagationTest),
        ]
    }
}

/// Verifies that simple memory allocation succeeds and is tracked.
struct MemoryAllocationTest;

#[async_trait]
impl TestCase for MemoryAllocationTest {
    fn name(&self) -> &str {
        "memory_allocation"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        info!("Setting up memory allocation test");
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start = Instant::now();

        // Allocate 1MiB of memory and confirm the allocation size.
        let buffer = vec![0u8; 1024 * 1024];
        if buffer.len() != 1024 * 1024 {
            return Err(TestError::assertion("allocation size mismatch"));
        }

        let mut metrics = TestMetrics::with_duration(start.elapsed());
        metrics.add_metric("allocated_bytes", buffer.len() as f64);
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestResult<()> {
        // Buffer is dropped automatically, but include a log entry for clarity.
        debug!("memory allocation test cleanup complete");
        Ok(())
    }
}

/// Ensures temporary files are removed during cleanup.
struct FileCleanupTest;

#[async_trait]
impl TestCase for FileCleanupTest {
    fn name(&self) -> &str {
        "file_cleanup"
    }

    async fn setup(&self, _fixtures: FixtureCtx<'_>) -> TestResult<()> {
        fs::create_dir_all("tests/temp").await?;
        Ok(())
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start = Instant::now();
        let path = "tests/temp/temp_file.txt";
        let mut file = File::create(path)
            .await
            .map_err(|e| TestError::execution(format!("create file failed: {}", e)))?;
        file.write_all(b"resource cleanup test")
            .await
            .map_err(|e| TestError::execution(format!("write file failed: {}", e)))?;
        file.flush().await.map_err(|e| TestError::execution(format!("flush failed: {}", e)))?;

        // Record artifact information in metrics before cleanup.
        let meta = fs::metadata(path).await?;
        let mut metrics = TestMetrics::with_duration(start.elapsed());
        metrics.add_metric("file_size_bytes", meta.len() as f64);
        metrics.add_metric("file_exists_before_cleanup", 1.0);
        Ok(metrics)
    }

    async fn cleanup(&self) -> TestResult<()> {
        let path = "tests/temp/temp_file.txt";
        fs::remove_file(path)
            .await
            .map_err(|e| TestError::execution(format!("failed to remove temp file: {}", e)))?;
        // Verify the file was removed.
        if fs::metadata(path).await.is_ok() {
            return Err(TestError::execution("temp file still exists after cleanup"));
        }
        Ok(())
    }
}

/// Demonstrates error propagation when a resource operation fails.
struct ErrorPropagationTest;

#[async_trait]
impl TestCase for ErrorPropagationTest {
    fn name(&self) -> &str {
        "error_propagation"
    }

    async fn execute(&self) -> TestResult<TestMetrics> {
        let start = Instant::now();
        // Attempt to open a missing file and treat the resulting error as success.
        let missing = "tests/temp/does_not_exist.txt";
        match File::open(missing).await {
            Ok(_) => Err(TestError::assertion("opening missing file should fail")),
            Err(e) => {
                let mut metrics = TestMetrics::with_duration(start.elapsed());
                metrics.add_metric("error_kind_not_found", 1.0);
                metrics.add_metric("os_error_code", e.raw_os_error().unwrap_or(-1) as f64);
                Ok(metrics)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TestConfig, TestHarness};

    #[tokio::test]
    async fn test_resource_management_suite() {
        let config = TestConfig::default();
        let harness = TestHarness::new(config).await.unwrap();
        let suite = ResourceManagementTestSuite::new();
        let result = harness.run_test_suite(&suite).await;
        assert!(result.is_ok());
        let suite_result = result.unwrap();
        assert_eq!(suite_result.summary.total_tests, 3);
    }
}
