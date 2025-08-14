use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::fs;
use tokio::time::timeout;

use super::errors::{TestError, TestOpResult as TestResultCompat};

/// Utility functions for common test operations
pub struct TestUtilities;

impl TestUtilities {
    /// Create a temporary directory for test data
    pub async fn create_temp_dir(prefix: &str) -> Result<PathBuf, TestError> {
        let temp_dir = std::env::temp_dir();
        let test_dir = temp_dir.join(format!("bitnet_test_{}_{}", prefix, generate_unique_id()));

        fs::create_dir_all(&test_dir).await?;

        tracing::debug!("Created temporary test directory: {:?}", test_dir);
        Ok(test_dir)
    }

    /// Clean up a temporary directory
    pub async fn cleanup_temp_dir(dir: &Path) -> TestResultCompat<()> {
        if dir.exists() {
            fs::remove_dir_all(dir).await?;
            tracing::debug!("Cleaned up temporary directory: {:?}", dir);
        }
        Ok(())
    }

    /// Write test data to a file
    pub async fn write_test_file<P: AsRef<Path>>(path: P, content: &[u8]) -> TestResultCompat<()> {
        let path = path.as_ref();

        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }

        fs::write(path, content).await?;
        tracing::trace!("Wrote {} bytes to {:?}", content.len(), path);
        Ok(())
    }

    /// Read test data from a file
    pub async fn read_test_file<P: AsRef<Path>>(path: P) -> TestResultCompat<Vec<u8>> {
        let path = path.as_ref();
        let content = fs::read(path).await?;
        tracing::trace!("Read {} bytes from {:?}", content.len(), path);
        Ok(content)
    }

    /// Check if a file exists and has the expected size
    pub async fn verify_file<P: AsRef<Path>>(
        path: P,
        expected_size: Option<u64>,
    ) -> TestResultCompat<bool> {
        let path = path.as_ref();

        if !path.exists() {
            return Ok(false);
        }

        if let Some(expected) = expected_size {
            let metadata = fs::metadata(path).await?;
            if metadata.len() != expected {
                tracing::warn!(
                    "File {:?} has size {} but expected {}",
                    path,
                    metadata.len(),
                    expected
                );
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Wait for a condition to be true with timeout
    pub async fn wait_for_condition<F, Fut>(
        condition: F,
        timeout_duration: Duration,
        check_interval: Duration,
    ) -> TestResultCompat<()>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = bool>,
    {
        let start = Instant::now();

        while start.elapsed() < timeout_duration {
            if condition().await {
                return Ok(());
            }

            tokio::time::sleep(check_interval).await;
        }

        Err(TestError::timeout(timeout_duration))
    }

    /// Run a test with a timeout
    pub async fn run_with_timeout<F, Fut, T>(
        operation: F,
        timeout_duration: Duration,
    ) -> TestResultCompat<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = TestResultCompat<T>>,
    {
        match timeout(timeout_duration, operation()).await {
            Ok(result) => result,
            Err(_) => Err(TestError::timeout(timeout_duration)),
        }
    }

    /// Retry an operation with exponential backoff
    pub async fn retry_with_backoff<F, Fut, T>(
        mut operation: F,
        max_attempts: usize,
        initial_delay: Duration,
        max_delay: Duration,
    ) -> TestResultCompat<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = TestResultCompat<T>>,
    {
        let mut delay = initial_delay;
        let mut last_error = None;

        for attempt in 1..=max_attempts {
            match operation().await {
                Ok(result) => {
                    if attempt > 1 {
                        tracing::info!("Operation succeeded on attempt {}", attempt);
                    }
                    return Ok(result);
                }
                Err(error) => {
                    last_error = Some(error);

                    if attempt < max_attempts {
                        tracing::warn!(
                            "Operation failed on attempt {} of {}, retrying in {:?}",
                            attempt,
                            max_attempts,
                            delay
                        );

                        tokio::time::sleep(delay).await;
                        delay = (delay * 2).min(max_delay);
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| TestError::execution("All retry attempts failed")))
    }

    /// Compare two byte arrays with detailed diff information
    pub fn compare_bytes(expected: &[u8], actual: &[u8]) -> ByteComparison {
        if expected == actual {
            return ByteComparison {
                is_equal: true,
                first_difference: None,
                expected_length: expected.len(),
                actual_length: actual.len(),
                similarity_ratio: 1.0,
            };
        }

        let first_difference =
            expected.iter().zip(actual.iter()).enumerate().find(|(_, (e, a))| e != a).map(
                |(index, (expected_byte, actual_byte))| ByteDifference {
                    position: index,
                    expected: *expected_byte,
                    actual: *actual_byte,
                },
            );

        // Calculate similarity ratio
        let min_len = expected.len().min(actual.len());
        let matching_bytes =
            expected.iter().zip(actual.iter()).take(min_len).filter(|(e, a)| e == a).count();

        let max_len = expected.len().max(actual.len());
        let similarity_ratio =
            if max_len == 0 { 1.0 } else { matching_bytes as f64 / max_len as f64 };

        ByteComparison {
            is_equal: false,
            first_difference,
            expected_length: expected.len(),
            actual_length: actual.len(),
            similarity_ratio,
        }
    }

    /// Generate test data with specific patterns
    pub fn generate_pattern_data(pattern: DataPattern, size: usize) -> Vec<u8> {
        match pattern {
            DataPattern::Zeros => vec![0u8; size],
            DataPattern::Ones => vec![0xFFu8; size],
            DataPattern::Alternating => {
                (0..size).map(|i| if i % 2 == 0 { 0xAA } else { 0x55 }).collect()
            }
            DataPattern::Sequential => (0..size).map(|i| (i % 256) as u8).collect(),
            DataPattern::Random(seed) => {
                use rand::{Rng, SeedableRng};
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                (0..size).map(|_| rng.gen()).collect()
            }
        }
    }

    /// Calculate checksum of data
    pub fn calculate_checksum(data: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(data);
        format!("{:x}", hasher.finalize())
    }

    /// Validate checksum of data
    pub fn validate_checksum(data: &[u8], expected_checksum: &str) -> bool {
        let actual_checksum = Self::calculate_checksum(data);
        actual_checksum == expected_checksum
    }

    /// Get system information for test context
    pub fn get_system_info() -> SystemInfo {
        SystemInfo {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            cpu_count: num_cpus::get(),
            available_memory: get_available_memory(),
            is_ci: super::utils::is_ci(),
            rust_version: get_rust_version(),
        }
    }

    /// Create a test report summary
    pub fn create_test_summary(
        test_name: &str,
        results: &[TestResultCompat<()>],
        duration: Duration,
    ) -> TestSummary {
        let total = results.len();
        let passed = results.iter().filter(|r| r.is_ok()).count();
        let failed = total - passed;
        let success_rate = if total > 0 { passed as f64 / total as f64 } else { 0.0 };

        TestSummary {
            test_name: test_name.to_string(),
            total_tests: total,
            passed_tests: passed,
            failed_tests: failed,
            success_rate,
            total_duration: duration,
            system_info: Self::get_system_info(),
        }
    }
}

/// Patterns for generating test data
#[derive(Debug, Clone, Copy)]
pub enum DataPattern {
    /// All zeros
    Zeros,
    /// All ones (0xFF)
    Ones,
    /// Alternating pattern (0xAA, 0x55)
    Alternating,
    /// Sequential bytes (0, 1, 2, ..., 255, 0, 1, ...)
    Sequential,
    /// Random data with specified seed
    Random(u64),
}

/// Result of byte comparison
#[derive(Debug, Clone)]
pub struct ByteComparison {
    /// Whether the byte arrays are equal
    pub is_equal: bool,
    /// First difference found (if any)
    pub first_difference: Option<ByteDifference>,
    /// Length of expected data
    pub expected_length: usize,
    /// Length of actual data
    pub actual_length: usize,
    /// Similarity ratio (0.0 to 1.0)
    pub similarity_ratio: f64,
}

/// Information about a byte difference
#[derive(Debug, Clone)]
pub struct ByteDifference {
    /// Position of the difference
    pub position: usize,
    /// Expected byte value
    pub expected: u8,
    /// Actual byte value
    pub actual: u8,
}

/// System information for test context
#[derive(Debug, Clone)]
pub struct SystemInfo {
    /// Operating system
    pub os: String,
    /// CPU architecture
    pub arch: String,
    /// Number of CPU cores
    pub cpu_count: usize,
    /// Available memory in bytes
    pub available_memory: u64,
    /// Whether running in CI environment
    pub is_ci: bool,
    /// Rust version
    pub rust_version: String,
}

/// Test summary information
#[derive(Debug, Clone)]
pub struct TestSummary {
    /// Name of the test
    pub test_name: String,
    /// Total number of tests
    pub total_tests: usize,
    /// Number of passed tests
    pub passed_tests: usize,
    /// Number of failed tests
    pub failed_tests: usize,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Total duration of all tests
    pub total_duration: Duration,
    /// System information
    pub system_info: SystemInfo,
}

impl TestSummary {
    /// Format the summary as a human-readable string
    pub fn format(&self) -> String {
        format!(
            "Test Summary: {}\n\
             Total: {} | Passed: {} | Failed: {} | Success Rate: {:.1}%\n\
             Duration: {}\n\
             System: {} {} ({} cores, {} memory)\n\
             Rust: {}",
            self.test_name,
            self.total_tests,
            self.passed_tests,
            self.failed_tests,
            self.success_rate * 100.0,
            super::format_duration(self.total_duration),
            self.system_info.os,
            self.system_info.arch,
            self.system_info.cpu_count,
            super::format_bytes(self.system_info.available_memory),
            self.system_info.rust_version
        )
    }
}

// Helper functions

/// Generate a unique ID for test resources
fn generate_unique_id() -> String {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let timestamp =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    format!("{}_{}", timestamp, id)
}

/// Get available system memory
fn get_available_memory() -> u64 {
    // This is a simplified implementation
    // In a real implementation, you might use system-specific APIs
    #[cfg(target_os = "linux")]
    {
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemAvailable:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            return kb * 1024; // Convert KB to bytes
                        }
                    }
                }
            }
        }
    }

    // Fallback: assume 8GB available
    8 * 1024 * 1024 * 1024
}

/// Get Rust version
fn get_rust_version() -> String {
    option_env!("CARGO_PKG_RUST_VERSION").unwrap_or("unknown").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_temp_dir_operations() {
        let temp_dir = TestUtilities::create_temp_dir("test").await.unwrap();
        assert!(temp_dir.exists());

        TestUtilities::cleanup_temp_dir(&temp_dir).await.unwrap();
        assert!(!temp_dir.exists());
    }

    #[tokio::test]
    async fn test_file_operations() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test_file.txt");
        let test_data = b"Hello, test world!";

        // Write test file
        TestUtilities::write_test_file(&file_path, test_data).await.unwrap();
        assert!(file_path.exists());

        // Read test file
        let read_data = TestUtilities::read_test_file(&file_path).await.unwrap();
        assert_eq!(read_data, test_data);

        // Verify file
        let is_valid =
            TestUtilities::verify_file(&file_path, Some(test_data.len() as u64)).await.unwrap();
        assert!(is_valid);

        let is_invalid = TestUtilities::verify_file(&file_path, Some(999)).await.unwrap();
        assert!(!is_invalid);
    }

    #[tokio::test]
    async fn test_wait_for_condition() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let start = Instant::now();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        let result = TestUtilities::wait_for_condition(
            move || {
                let count = counter_clone.fetch_add(1, Ordering::Relaxed) + 1;
                async move { count >= 3 }
            },
            Duration::from_secs(1),
            Duration::from_millis(10),
        )
        .await;

        assert!(result.is_ok());
        assert!(start.elapsed() >= Duration::from_millis(20)); // At least 2 intervals
        assert_eq!(counter.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn test_run_with_timeout() {
        // Test successful operation
        let result =
            TestUtilities::run_with_timeout(|| async { Ok("success") }, Duration::from_secs(1))
                .await;
        assert_eq!(result.unwrap(), "success");

        // Test timeout
        let result = TestUtilities::run_with_timeout(
            || async {
                tokio::time::sleep(Duration::from_secs(2)).await;
                Ok("should not reach here")
            },
            Duration::from_millis(100),
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_retry_with_backoff() {
        let mut attempt_count = 0;

        let result = TestUtilities::retry_with_backoff(
            || {
                attempt_count += 1;
                async move {
                    if attempt_count < 3 {
                        Err(TestError::execution("Not ready yet"))
                    } else {
                        Ok("success")
                    }
                }
            },
            5,
            Duration::from_millis(10),
            Duration::from_millis(100),
        )
        .await;

        assert_eq!(result.unwrap(), "success");
        assert_eq!(attempt_count, 3);
    }

    #[test]
    fn test_compare_bytes() {
        let data1 = b"hello world";
        let data2 = b"hello world";
        let data3 = b"hello rust!";

        // Equal data
        let comparison = TestUtilities::compare_bytes(data1, data2);
        assert!(comparison.is_equal);
        assert_eq!(comparison.similarity_ratio, 1.0);
        assert!(comparison.first_difference.is_none());

        // Different data
        let comparison = TestUtilities::compare_bytes(data1, data3);
        assert!(!comparison.is_equal);
        assert!(comparison.similarity_ratio < 1.0);
        assert!(comparison.first_difference.is_some());

        let diff = comparison.first_difference.unwrap();
        assert_eq!(diff.position, 6); // First difference at 'w' vs 'r'
        assert_eq!(diff.expected, b'w');
        assert_eq!(diff.actual, b'r');
    }

    #[test]
    fn test_generate_pattern_data() {
        let zeros = TestUtilities::generate_pattern_data(DataPattern::Zeros, 10);
        assert_eq!(zeros, vec![0u8; 10]);

        let ones = TestUtilities::generate_pattern_data(DataPattern::Ones, 5);
        assert_eq!(ones, vec![0xFFu8; 5]);

        let sequential = TestUtilities::generate_pattern_data(DataPattern::Sequential, 5);
        assert_eq!(sequential, vec![0, 1, 2, 3, 4]);

        let random1 = TestUtilities::generate_pattern_data(DataPattern::Random(42), 10);
        let random2 = TestUtilities::generate_pattern_data(DataPattern::Random(42), 10);
        assert_eq!(random1, random2); // Same seed should produce same data

        let random3 = TestUtilities::generate_pattern_data(DataPattern::Random(43), 10);
        assert_ne!(random1, random3); // Different seed should produce different data
    }

    #[test]
    fn test_checksum_operations() {
        let data = b"test data for checksum";
        let checksum = TestUtilities::calculate_checksum(data);

        assert!(!checksum.is_empty());
        assert_eq!(checksum.len(), 64); // SHA256 produces 64 hex characters

        // Validate correct checksum
        assert!(TestUtilities::validate_checksum(data, &checksum));

        // Validate incorrect checksum
        assert!(!TestUtilities::validate_checksum(data, "invalid_checksum"));
    }

    #[test]
    fn test_system_info() {
        let info = TestUtilities::get_system_info();

        assert!(!info.os.is_empty());
        assert!(!info.arch.is_empty());
        assert!(info.cpu_count > 0);
        assert!(info.available_memory > 0);
        assert!(!info.rust_version.is_empty());
    }

    #[test]
    fn test_test_summary() {
        let results = vec![Ok(()), Ok(()), Err(TestError::execution("test error"))];

        let summary =
            TestUtilities::create_test_summary("test_suite", &results, Duration::from_secs(10));

        assert_eq!(summary.test_name, "test_suite");
        assert_eq!(summary.total_tests, 3);
        assert_eq!(summary.passed_tests, 2);
        assert_eq!(summary.failed_tests, 1);
        assert!((summary.success_rate - 2.0 / 3.0).abs() < f64::EPSILON);
        assert_eq!(summary.total_duration, Duration::from_secs(10));

        let formatted = summary.format();
        assert!(formatted.contains("test_suite"));
        assert!(formatted.contains("Total: 3"));
        assert!(formatted.contains("Passed: 2"));
        assert!(formatted.contains("Failed: 1"));
    }
}
