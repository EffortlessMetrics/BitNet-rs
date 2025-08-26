#!/usr/bin/env rust-script
//! Test GPU memory validation functionality
//!
//! Run with: cargo run --example test_gpu_memory --no-default-features --features cuda

// Note: The GPU validation module is internal to bitnet-kernels
// This example demonstrates the intended API usage

fn main() {
    // Initialize logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("=== Testing GPU Memory Validation ===\n");

    // Create a validator with default configuration
    let validator = GpuValidator::new();

    println!("Running memory health check...");

    // Test the public API
    match validator.check_memory_health() {
        Ok(result) => {
            println!("✅ Memory health check completed successfully!");
            println!("   Peak GPU memory: {} MB", result.peak_gpu_memory / (1024 * 1024));
            println!("   Memory leaks detected: {}", result.leaks_detected);
            println!("   Memory efficiency: {:.2}%", result.efficiency_score * 100.0);

            if result.leaks_detected {
                println!("⚠️  Warning: Memory leaks were detected!");
            } else {
                println!("✅ No memory leaks detected");
            }
        }
        Err(e) => {
            println!("❌ Memory health check failed: {}", e);
            println!("   This is expected if CUDA is not available on this system.");
        }
    }

    println!("\n=== Testing with Custom Configuration ===\n");

    // Create a custom configuration
    let mut config = ValidationConfig::default();
    config.check_memory_leaks = true;
    config.benchmark_iterations = 10;

    let validator_custom = GpuValidator::with_config(config);

    println!("Running comprehensive validation...");
    match validator_custom.validate() {
        Ok(results) => {
            println!("✅ Comprehensive validation completed!");
            println!("   Success: {}", results.success);
            println!("   Accuracy tests: {}", results.accuracy_results.len());
            println!("   Performance tests: {}", results.performance_results.len());

            if let Some(memory) = &results.memory_results {
                println!("   Memory test results:");
                println!("     - Peak usage: {} MB", memory.peak_gpu_memory / (1024 * 1024));
                println!("     - Leaks detected: {}", memory.leaks_detected);
                println!("     - Efficiency: {:.2}%", memory.efficiency_score * 100.0);
            }
        }
        Err(e) => {
            println!("❌ Comprehensive validation failed: {}", e);
            println!("   This is expected if CUDA is not available.");
        }
    }

    println!("\n=== Test Complete ===");
}
