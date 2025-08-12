use std::env;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio;
use tracing::{error, info, warn, Level};
use tracing_subscriber;

// Import our test framework components
use bitnet_tests::common::{
    config::TestConfig,
    errors::TestError,
    execution_optimizer::{ExecutionOptimizer, OptimizedExecutionResult},
    fast_config::{FastConfigBuilder, SpeedProfile},
};

/// Fast test runner optimized for <15 minute execution
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    init_logging();

    info!("Starting BitNet fast test runner");
    let start_time = Instant::now();

    // Parse command line arguments
    let args = parse_args();

    // Create execution optimizer with target time
    let target_duration = Duration::from_secs(args.target_minutes * 60);
    let mut optimizer = ExecutionOptimizer::with_target_duration(target_duration);

    // Execute optimized test suite
    match optimizer.execute_optimized().await {
        Ok(result) => {
            print_results(&result);

            if result.success {
                info!("✅ All tests completed successfully!");
                std::process::exit(0);
            } else {
                error!("❌ Some tests failed");
                std::process::exit(1);
            }
        }
        Err(e) => {
            error!("Test execution failed: {}", e);
            std::process::exit(1);
        }
    }
}

/// Command line arguments
#[derive(Debug)]
struct Args {
    target_minutes: u64,
    profile: SpeedProfile,
    parallel: Option<usize>,
    verbose: bool,
    incremental: bool,
    categories: Vec<String>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            target_minutes: 15,
            profile: SpeedProfile::Fast,
            parallel: None,
            verbose: false,
            incremental: true,
            categories: Vec::new(),
        }
    }
}

/// Parse command line arguments
fn parse_args() -> Args {
    let mut args = Args::default();
    let env_args: Vec<String> = env::args().collect();

    let mut i = 1;
    while i < env_args.len() {
        match env_args[i].as_str() {
            "--target" | "-t" => {
                if i + 1 < env_args.len() {
                    args.target_minutes = env_args[i + 1].parse().unwrap_or(15);
                    i += 1;
                }
            }
            "--profile" | "-p" => {
                if i + 1 < env_args.len() {
                    args.profile = match env_args[i + 1].as_str() {
                        "lightning" => SpeedProfile::Lightning,
                        "fast" => SpeedProfile::Fast,
                        "balanced" => SpeedProfile::Balanced,
                        "thorough" => SpeedProfile::Thorough,
                        _ => SpeedProfile::Fast,
                    };
                    i += 1;
                }
            }
            "--parallel" | "-j" => {
                if i + 1 < env_args.len() {
                    args.parallel = env_args[i + 1].parse().ok();
                    i += 1;
                }
            }
            "--verbose" | "-v" => {
                args.verbose = true;
            }
            "--no-incremental" => {
                args.incremental = false;
            }
            "--categories" | "-c" => {
                if i + 1 < env_args.len() {
                    args.categories = env_args[i + 1].split(',').map(|s| s.to_string()).collect();
                    i += 1;
                }
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    // Override from environment variables
    if let Ok(target) = env::var("BITNET_TEST_TARGET_MINUTES") {
        args.target_minutes = target.parse().unwrap_or(args.target_minutes);
    }

    if let Ok(parallel) = env::var("BITNET_TEST_PARALLEL") {
        args.parallel = parallel.parse().ok();
    }

    if env::var("BITNET_TEST_VERBOSE").is_ok() {
        args.verbose = true;
    }

    args
}

/// Initialize logging based on verbosity
fn init_logging() {
    let level = if env::var("BITNET_TEST_VERBOSE").is_ok() {
        Level::DEBUG
    } else {
        Level::INFO
    };

    tracing_subscriber::fmt()
        .with_max_level(level)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .init();
}

/// Print help message
fn print_help() {
    println!("BitNet Fast Test Runner");
    println!();
    println!("USAGE:");
    println!("    fast_test_runner [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    -t, --target <MINUTES>     Target execution time in minutes [default: 15]");
    println!("    -p, --profile <PROFILE>    Speed profile: lightning, fast, balanced, thorough [default: fast]");
    println!("    -j, --parallel <COUNT>     Number of parallel test threads [default: auto]");
    println!("    -v, --verbose              Enable verbose output");
    println!("    --no-incremental           Disable incremental testing");
    println!("    -c, --categories <LIST>    Comma-separated list of test categories to run");
    println!("    -h, --help                 Print this help message");
    println!();
    println!("ENVIRONMENT VARIABLES:");
    println!("    BITNET_TEST_TARGET_MINUTES    Override target execution time");
    println!("    BITNET_TEST_PARALLEL          Override parallel thread count");
    println!("    BITNET_TEST_VERBOSE           Enable verbose output");
    println!();
    println!("EXAMPLES:");
    println!("    fast_test_runner --target 10 --profile lightning");
    println!("    fast_test_runner --parallel 4 --categories unit,integration");
    println!("    BITNET_TEST_VERBOSE=1 fast_test_runner --no-incremental");
}

/// Print execution results
fn print_results(result: &OptimizedExecutionResult) {
    println!();
    println!("=== Test Execution Results ===");
    println!();

    // Summary
    println!("📊 Summary:");
    println!(
        "   Status: {}",
        if result.success {
            "✅ PASSED"
        } else {
            "❌ FAILED"
        }
    );
    println!(
        "   Duration: {:.1}s (target: {:.1}s)",
        result.total_duration.as_secs_f64(),
        result.target_duration.as_secs_f64()
    );
    println!("   Efficiency: {:.1}%", result.efficiency_score * 100.0);
    println!("   Tests Skipped: {}", result.tests_skipped);
    println!();

    // Optimizations applied
    if !result.optimization_applied.is_empty() {
        println!("⚡ Optimizations Applied:");
        for optimization in &result.optimization_applied {
            println!("   • {}", optimization);
        }
        println!();
    }

    // Group results
    println!("📋 Test Group Results:");
    let mut total_tests = 0;
    let mut passed_tests = 0;

    for (i, group_result) in result.group_results.iter().enumerate() {
        let group_passed = group_result
            .test_results
            .iter()
            .filter(|r| r.passed())
            .count();
        let group_total = group_result.test_results.len();

        total_tests += group_total;
        passed_tests += group_passed;

        let status_icon = if group_result.success { "✅" } else { "❌" };
        let timeout_info = if group_result.timeout_occurred {
            " (TIMEOUT)"
        } else {
            ""
        };

        println!(
            "   Group {}: {} {}/{} tests in {:.1}s{}",
            i + 1,
            status_icon,
            group_passed,
            group_total,
            group_result.duration.as_secs_f64(),
            timeout_info
        );
    }

    println!();
    println!("📈 Overall Statistics:");
    println!("   Total Tests: {}", total_tests);
    println!(
        "   Passed: {} ({:.1}%)",
        passed_tests,
        if total_tests > 0 {
            passed_tests as f64 / total_tests as f64 * 100.0
        } else {
            0.0
        }
    );
    println!("   Failed: {}", total_tests - passed_tests);

    // Performance analysis
    let target_met = result.total_duration <= result.target_duration;
    println!();
    println!("🎯 Performance Analysis:");
    println!(
        "   Target Met: {}",
        if target_met { "✅ YES" } else { "❌ NO" }
    );

    if target_met {
        let time_saved = result.target_duration.as_secs_f64() - result.total_duration.as_secs_f64();
        println!("   Time Saved: {:.1}s", time_saved);
    } else {
        let time_over = result.total_duration.as_secs_f64() - result.target_duration.as_secs_f64();
        println!("   Time Over: {:.1}s", time_over);
    }

    // Recommendations
    println!();
    println!("💡 Recommendations:");

    if result.efficiency_score < 0.7 {
        println!("   • Consider reducing parallel threads to improve efficiency");
    }

    if result.tests_skipped > 0 {
        println!(
            "   • {} tests were skipped to meet time target",
            result.tests_skipped
        );
        println!("   • Consider running full test suite in CI/CD pipeline");
    }

    if !target_met {
        println!("   • Consider using 'lightning' profile for faster execution");
        println!("   • Enable incremental testing to reduce test scope");
    }

    if result.efficiency_score > 0.9 && target_met {
        println!("   • Excellent performance! Consider running more comprehensive tests");
    }

    println!();
}

/// Create test configuration from arguments
fn create_config_from_args(args: &Args) -> TestConfig {
    let mut builder = FastConfigBuilder::with_profile(args.profile.clone());

    if let Some(parallel) = args.parallel {
        builder = builder.max_parallel(parallel);
    }

    builder = builder
        .timeout(Duration::from_secs(60))
        .log_level(if args.verbose { "debug" } else { "warn" })
        .coverage(false)
        .performance(true);

    builder.build()
}

/// Validate environment and prerequisites
fn validate_environment() -> Result<(), TestError> {
    // Check if cargo is available
    if std::process::Command::new("cargo")
        .arg("--version")
        .output()
        .is_err()
    {
        return Err(TestError::SetupError("cargo not found in PATH".to_string()));
    }

    // Check if we're in a Rust workspace
    if !PathBuf::from("Cargo.toml").exists() {
        return Err(TestError::SetupError(
            "Cargo.toml not found - not in a Rust workspace".to_string(),
        ));
    }

    // Check if tests directory exists
    if !PathBuf::from("tests").exists() {
        warn!("tests directory not found - some tests may not be available");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_args_parsing() {
        let args = Args::default();
        assert_eq!(args.target_minutes, 15);
        assert!(matches!(args.profile, SpeedProfile::Fast));
    }

    #[test]
    fn test_config_creation() {
        let args = Args {
            target_minutes: 10,
            profile: SpeedProfile::Lightning,
            parallel: Some(4),
            verbose: true,
            incremental: true,
            categories: vec!["unit".to_string()],
        };

        let config = create_config_from_args(&args);
        assert_eq!(config.max_parallel_tests, 4);
        assert_eq!(config.log_level, "debug");
    }
}
