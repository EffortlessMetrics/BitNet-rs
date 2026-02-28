//! Demonstration of the warn_once utility for rate-limited logging.
//!
//! This example shows how to use the warn_once macro to avoid log spam
//! when the same warning condition occurs repeatedly.

use bitnet_warn_once::warn_once;

fn deprecated_function_v1() {
    warn_once!("deprecated_api_v1", "Using deprecated API v1, please migrate to v2");
    // Actual function logic would go here
    println!("Executing deprecated function v1...");
}

fn model_fallback_scenario(iteration: usize) {
    // Simulate a scenario where we need to fallback to CPU
    if iteration.is_multiple_of(3) {
        warn_once!("gpu_unavailable", "GPU not available, falling back to CPU for this operation");
    }
    println!("Processing iteration {}", iteration);
}

fn main() {
    // Initialize tracing subscriber to see the logs
    tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG).init();

    println!("=== Warn Once Demonstration ===\n");

    println!("1. Deprecated API warning (called 3 times):");
    for i in 1..=3 {
        println!("  Call {}", i);
        deprecated_function_v1();
    }

    println!("\n2. Model fallback scenario (10 iterations):");
    for i in 1..=10 {
        model_fallback_scenario(i);
    }

    println!("\n3. Multiple different warnings:");
    warn_once!("warning_a", "This is warning A");
    warn_once!("warning_b", "This is warning B");
    warn_once!("warning_a", "This is warning A again (rate-limited)");
    warn_once!("warning_c", "This is warning C");

    println!("\n4. Formatted warnings:");
    for value in [10, 20, 30] {
        warn_once!("threshold_exceeded", "Value {} exceeds threshold of 5", value);
    }

    println!("\n=== Demo Complete ===");
    println!("\nNotice that:");
    println!("- First occurrence of each warning is logged at WARN level");
    println!("- Subsequent occurrences are logged at DEBUG level (rate-limited)");
    println!("- Each unique key is tracked independently");
}
