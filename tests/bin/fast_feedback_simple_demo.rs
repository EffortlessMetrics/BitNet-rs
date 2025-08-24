use std::env;
use std::time::Instant;

// Import the simplified fast feedback system
use bitnet_tests::fast_feedback_simple::{FastFeedbackSystem, utils};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("BitNet.rs Fast Feedback Demo (Simplified)");
    println!("==========================================");

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("auto");

    // Create fast feedback system based on mode
    let mut system = match mode {
        "ci" => {
            println!("Using CI-optimized configuration");
            FastFeedbackSystem::for_ci()
        }
        "dev" => {
            println!("Using development-optimized configuration");
            FastFeedbackSystem::for_development()
        }
        "auto" => {
            println!("Auto-detecting environment");
            utils::create_for_environment()
        }
        _ => {
            println!("Using default configuration");
            FastFeedbackSystem::with_defaults()
        }
    };

    // Display configuration
    let config = system.config();
    println!("Configuration:");
    println!("  Target feedback time: {:?}", config.target_feedback_time);
    println!("  Max feedback time: {:?}", config.max_feedback_time);
    println!("  Incremental testing: {}", config.enable_incremental);
    println!("  Smart selection: {}", config.enable_smart_selection);
    println!("  Parallel execution: {}", config.enable_parallel);
    println!("  Max parallel: {}", config.max_parallel_fast);
    println!("  Fail fast: {}", config.fail_fast);

    // Check if fast feedback should be used
    if utils::should_use_fast_feedback() {
        println!("Fast feedback is recommended for this environment");
    } else {
        println!("Fast feedback is optional for this environment");
    }

    // Execute fast feedback
    let start_time = Instant::now();
    println!("Starting fast feedback execution...");

    match system.execute_fast_feedback().await {
        Ok(result) => {
            let _total_time = start_time.elapsed();

            println!("Fast feedback completed successfully!");
            println!("Results:");
            println!("  Execution time: {:?}", result.execution_time);
            println!("  Tests run: {}", result.tests_run);
            println!("  Tests passed: {}", result.tests_passed);
            println!("  Tests failed: {}", result.tests_failed);
            println!("  Tests skipped: {}", result.tests_skipped);
            println!("  Coverage achieved: {:.1}%", result.coverage_achieved * 100.0);
            println!("  Feedback quality: {:?}", result.feedback_quality);

            if !result.optimization_applied.is_empty() {
                println!("Optimizations applied:");
                for optimization in &result.optimization_applied {
                    println!("  - {}", optimization);
                }
            }

            if !result.next_recommendations.is_empty() {
                println!("Recommendations for next run:");
                for recommendation in &result.next_recommendations {
                    println!("  - {}", recommendation);
                }
            }

            // Check if we met our target
            let config = system.config();
            if result.execution_time <= config.target_feedback_time {
                println!("✅ Target feedback time achieved!");
            } else {
                println!(
                    "⚠️  Target feedback time exceeded by {:?}",
                    result.execution_time - config.target_feedback_time
                );
            }

            // Display success rate
            let success_rate = if result.tests_run > 0 {
                result.tests_passed as f64 / result.tests_run as f64 * 100.0
            } else {
                0.0
            };

            if success_rate >= 95.0 {
                println!("✅ Excellent test success rate: {:.1}%", success_rate);
            } else if success_rate >= 80.0 {
                println!("⚠️  Good test success rate: {:.1}%", success_rate);
            } else {
                println!("❌ Low test success rate: {:.1}%", success_rate);
            }
        }
        Err(e) => {
            println!("Fast feedback execution failed: {:?}", e);
            return Err(e.into());
        }
    }

    println!("Demo completed in {:?}", start_time.elapsed());

    // Show usage instructions
    println!("\nUsage instructions:");
    println!("  1. For development: cargo run --bin fast_feedback_simple_demo -- dev");
    println!("  2. For CI: cargo run --bin fast_feedback_simple_demo -- ci");
    println!("  3. Auto-detect: cargo run --bin fast_feedback_simple_demo -- auto");

    println!("\nEnvironment integration:");
    println!("  - Set BITNET_FAST_FEEDBACK=1 to enable fast feedback");
    println!("  - Set BITNET_INCREMENTAL=1 to enable incremental testing");
    println!("  - CI environments automatically use optimized settings");

    println!("\nPerformance targets:");
    println!("  - Development: 30 seconds for immediate feedback");
    println!("  - CI: 90 seconds for balanced speed and coverage");
    println!("  - Full suite: 15 minutes maximum execution time");

    Ok(())
}
