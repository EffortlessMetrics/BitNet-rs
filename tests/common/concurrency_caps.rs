//! Concurrency caps for BitNet-rs tests
//!
//! This module provides utilities to initialize thread pool limits and prevent
//! resource exhaustion during parallel test execution.

use std::sync::Once;

static INIT_CONCURRENCY: Once = Once::new();

/// Initialize global concurrency limits for tests
///
/// This should be called once at the start of test suites to ensure:
/// - Rayon thread pool is capped to prevent fork bombs
/// - Test execution respects system resource limits
/// - Deterministic test behavior across machines
pub fn init_concurrency_caps() {
    INIT_CONCURRENCY.call_once(|| {
        // Get thread limits from environment or use conservative defaults
        let rust_test_threads = std::env::var("RUST_TEST_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(2);

        let rayon_threads = std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(2);

        // Initialize Rayon with capped thread pool
        if let Err(e) = rayon::ThreadPoolBuilder::new()
            .num_threads(rayon_threads)
            .thread_name(|i| format!("bitnet-rayon-{}", i))
            .build_global()
        {
            // If global pool is already initialized, that's okay
            eprintln!("Note: Rayon global pool already initialized: {}", e);
        }

        // Set environment hints for any spawned processes
        std::env::set_var("RAYON_NUM_THREADS", rayon_threads.to_string());
        
        tracing::info!(
            "Concurrency caps initialized: RUST_TEST_THREADS={}, RAYON_NUM_THREADS={}",
            rust_test_threads,
            rayon_threads
        );
    });
}

/// Initialize concurrency caps and return bounded parallel limit for async operations
pub fn init_and_get_async_limit() -> usize {
    init_concurrency_caps();
    
    // Use half of Rayon threads for async concurrency to avoid oversubscription
    std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|n| (n + 1) / 2) // Round up division
        .unwrap_or(1)
        .max(1)
}

/// Get the current thread limit for spawning parallel tasks
pub fn get_parallel_limit() -> usize {
    std::env::var("RUST_TEST_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(2)
        .max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_concurrency_caps() {
        // Should not panic on repeated calls
        init_concurrency_caps();
        init_concurrency_caps();
    }

    #[test]
    fn test_get_limits() {
        let async_limit = init_and_get_async_limit();
        let parallel_limit = get_parallel_limit();
        
        assert!(async_limit >= 1);
        assert!(parallel_limit >= 1);
    }
}