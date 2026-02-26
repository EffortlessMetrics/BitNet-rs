use std::env;
use std::time::Instant;
use tracing::{info, warn, Level};
use tracing_subscriber;

// Import the fast feedback system
use tests::common::fast_feedback::{utils, FastFeedbackConfig, FastFeedbackSystem};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("BitNet-rs Fast Feedback Demo");
    info!("============================");

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("auto");

    // Create fast feedback system based on mode
    let mut system = match mode {
        "ci" => {
            info!("Using CI-optimized configuration");
            FastFeedbackSystem::for_ci()
        }
        "dev" => {
            info!("Using development-optimized configuration");
            FastFeedbackSystem::for_development()
        }
        "auto" => {
            info!("Auto-detecting environment");
            utils::create_for_environment()
        }
        _ => {
            info!("Using default configuration");
            FastFeedbackSystem::with_defaults()
        }
    };

    // Display configuration
    info!("Configuration:");
    info!("  Target feedback time: {:?}", system.config.target_feedback_time);
    info!("  Max feedback time: {:?}", system.config.max_feedback_time);
    info!("  Incremental testing: {}", system.config.enable_incremental);
    info!("  Smart selection: {}", system.config.enable_smart_selection);
    info!("  Parallel execution: {}", system.config.enable_parallel);
    info!("  Max parallel: {}", system.config.max_parallel_fast);
    info!("  Fail fast: {}", system.config.fail_fast);

    // Check if fast feedback should be used
    if utils::should_use_fast_feedback() {
        info!("Fast feedback is recommended for this environment");
    } else {
        info!("Fast feedback is optional for this environment");
    }

    // Execute fast feedback
    let start_time = Instant::now();
    info!("Starting fast feedback execution...");

    match system.execute_fast_feedback().await {
        Ok(result) => {
            let total_time = start_time.elapsed();

            info!("Fast feedback completed successfully!");
            info!("Results:");
            info!("  Execution time: {:?}", result.execution_time);
            info!("  Tests run: {}", result.tests_run);
            info!("  Tests passed: {}", result.tests_passed);
            info!("  Tests failed: {}", result.tests_failed);
            info!("  Tests skipped: {}", result.tests_skipped);
            info!("  Coverage achieved: {:.1}%", result.coverage_achieved * 100.0);
            info!("  Feedback quality: {:?}", result.feedback_quality);

            if !result.optimization_applied.is_empty() {
                info!("Optimizations applied:");
                for optimization in &result.optimization_applied {
                    info!("  - {}", optimization);
                }
            }

            if !result.next_recommendations.is_empty() {
                info!("Recommendations for next run:");
                for recommendation in &result.next_recommendations {
                    info!("  - {}", recommendation);
                }
            }

            // Check if we met our target
            if result.execution_time <= system.config.target_feedback_time {
                info!("✅ Target feedback time achieved!");
            } else {
                warn!(
                    "⚠️  Target feedback time exceeded by {:?}",
                    result.execution_time - system.config.target_feedback_time
                );
            }

            // Display success rate
            let success_rate = if result.tests_run > 0 {
                result.tests_passed as f64 / result.tests_run as f64 * 100.0
            } else {
                0.0
            };

            if success_rate >= 95.0 {
                info!("✅ Excellent test success rate: {:.1}%", success_rate);
            } else if success_rate >= 80.0 {
                info!("⚠️  Good test success rate: {:.1}%", success_rate);
            } else {
                warn!("❌ Low test success rate: {:.1}%", success_rate);
            }
        }
        Err(e) => {
            warn!("Fast feedback execution failed: {}", e);
            return Err(e.into());
        }
    }

    info!("Demo completed in {:?}", start_time.elapsed());
    Ok(())
}
