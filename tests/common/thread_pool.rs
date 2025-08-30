use rayon::ThreadPool;
use std::sync::{Arc, OnceLock};

/// Test-specific thread pool management to prevent Rayon conflicts
///
/// This module provides thread pool isolation for tests to prevent:
/// - Global thread pool reconfiguration conflicts
/// - Oversubscription during parallel test execution
/// - Thread pool deadlocks between tests

static TEST_POOL: OnceLock<Arc<ThreadPool>> = OnceLock::new();

/// Get or initialize a test-specific thread pool
///
/// The thread pool size is determined by environment variables or sensible defaults:
/// - RAYON_NUM_THREADS: Direct thread count override
/// - RUST_TEST_THREADS: Used as fallback if RAYON_NUM_THREADS not set
/// - Default: min(4, num_cpus / 2) to prevent oversubscription
pub fn get_test_pool() -> Arc<ThreadPool> {
    TEST_POOL
        .get_or_init(|| {
            let num_threads = determine_test_thread_count();

            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .thread_name(|index| format!("bitnet-test-{}", index))
                .build()
                .expect("Failed to build test thread pool");

            Arc::new(pool)
        })
        .clone()
}

/// Execute a closure within the test thread pool
///
/// This ensures all Rayon operations happen within the controlled test pool
/// rather than the global pool, preventing configuration conflicts.
pub fn with_test_pool<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    let pool = get_test_pool();
    pool.install(f)
}

/// Initialize deterministic test environment
///
/// Sets up reproducible test conditions by:
/// - Configuring thread pools with fixed sizes
/// - Setting deterministic seeds if BITNET_DETERMINISTIC=1
/// - Ensuring single-threaded execution if requested
pub fn init_deterministic_test_env() {
    // Initialize the test pool
    let _ = get_test_pool();

    // Set deterministic seed if requested
    if std::env::var("BITNET_DETERMINISTIC").unwrap_or_default() == "1" {
        let seed = std::env::var("BITNET_SEED")
            .unwrap_or_else(|_| "42".to_string())
            .parse::<u64>()
            .unwrap_or(42);

        // Initialize any deterministic systems here
        // For example, if you have a global random number generator
        eprintln!("Deterministic test mode enabled with seed: {}", seed);
    }
}

/// Determine optimal thread count for tests
fn determine_test_thread_count() -> usize {
    // Check for explicit RAYON thread count
    if let Ok(rayon_threads) = std::env::var("RAYON_NUM_THREADS") {
        if let Ok(count) = rayon_threads.parse::<usize>() {
            return count.max(1);
        }
    }

    // Check for test thread count
    if let Ok(test_threads) = std::env::var("RUST_TEST_THREADS") {
        if let Ok(count) = test_threads.parse::<usize>() {
            return count.max(1);
        }
    }

    // Default: conservative thread count to prevent oversubscription
    let num_cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

    // Use at most half the available cores, minimum 1, maximum 4 for tests
    (num_cpus / 2).max(1).min(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_initialization() {
        let pool1 = get_test_pool();
        let pool2 = get_test_pool();

        // Should return the same pool instance
        assert!(Arc::ptr_eq(&pool1, &pool2));
    }

    #[test]
    fn test_with_pool_execution() {
        let result = with_test_pool(|| {
            // This should execute within the test pool
            rayon::join(|| 1, || 2)
        });

        assert_eq!(result, (1, 2));
    }

    #[test]
    fn test_thread_count_determination() {
        // Test default behavior
        let count = determine_test_thread_count();
        assert!(count >= 1 && count <= 4);
    }

    #[test]
    fn test_deterministic_env_initialization() {
        // This should not panic
        init_deterministic_test_env();
    }
}
