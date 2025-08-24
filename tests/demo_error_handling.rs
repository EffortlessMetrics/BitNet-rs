/// Demonstration of Enhanced Error Handling Capabilities
///
/// This demo shows the key features of the enhanced error handling system
/// without relying on complex dependencies that have compilation issues.

fn main() {
    #[cfg(feature = "integration-tests")]
    run_demo();

    #[cfg(not(feature = "integration-tests"))]
    {
        println!("Error handling demo requires 'integration-tests' feature.");
        println!("Run with: cargo run --bin demo_error_handling --features integration-tests");
    }
}

#[cfg(feature = "integration-tests")]
fn run_demo() {
    use bitnet_tests::errors::{FixtureError, TestError};
    use std::time::Duration;

    println!("=== BitNet.rs Enhanced Error Handling Demo ===\n");

    // Demo 1: Error Severity and Categorization
    println!("1. Error Severity and Categorization:");
    let errors = vec![
        TestError::TimeoutError { timeout: Duration::from_secs(30) },
        TestError::AssertionError { message: "Expected 42 but got 24".to_string() },
        TestError::ConfigError { message: "Invalid configuration parameter".to_string() },
        TestError::FixtureError(FixtureError::DownloadError {
            url: "https://example.com/model.gguf".to_string(),
            reason: "Network timeout".to_string(),
        }),
    ];

    for (i, error) in errors.iter().enumerate() {
        println!(
            "  Error {}: {} (Category: {}, Severity: {})",
            i + 1,
            error,
            error.category(),
            error.severity()
        );
    }

    // Demo 2: Error Context and Recovery
    println!("\n2. Error Context and Recovery:");
    let error_with_context =
        TestError::AssertionError { message: "Value mismatch in tensor operation".to_string() }
            .with_context("While processing layer 3 of the model")
            .with_suggestion("Check tensor dimensions match expected shape");

    println!("  Full error: {}", error_with_context);
    if let Some(suggestion) = error_with_context.suggestion() {
        println!("  💡 Suggestion: {}", suggestion);
    }

    // Demo 3: Error Chain Demonstration
    println!("\n3. Error Chaining:");
    let root_error = TestError::ExecutionError { message: "Failed to load model".to_string() };

    let chain_error = TestError::SetupError { message: "Test setup failed".to_string() }
        .with_source(Box::new(root_error))
        .with_context("During test initialization");

    println!("  Error: {}", chain_error);
    println!("  Full chain:");
    let mut current_error: Option<&dyn std::error::Error> = Some(&chain_error);
    let mut depth = 0;
    while let Some(err) = current_error {
        println!("    {} {}", "  ".repeat(depth), err);
        current_error = err.source();
        depth += 1;
    }

    // Demo 4: Fixture Error Handling
    println!("\n4. Fixture-Specific Errors:");
    let fixture_errors = vec![
        FixtureError::NotFound { name: "test-model.gguf".to_string() },
        FixtureError::ValidationFailed {
            name: "corrupted-model.gguf".to_string(),
            reason: "Checksum mismatch".to_string(),
        },
        FixtureError::CacheError { message: "Cache directory not writable".to_string() },
    ];

    for error in &fixture_errors {
        println!("  - {}", error);
        if error.is_retriable() {
            println!("    ↻ This error is retriable");
        }
    }

    // Demo 5: Error Recovery Strategies
    println!("\n5. Recovery Strategies:");
    let recoverable_error = TestError::TimeoutError { timeout: Duration::from_secs(10) }
        .with_recovery_strategy(
            "Increase timeout duration or run test with fewer parallel workers",
        );

    println!("  Error: {}", recoverable_error);
    if let Some(recovery) = recoverable_error.recovery_strategy() {
        println!("  Recovery: {}", recovery);
    }

    println!("\n=== Demo Complete ===");
}

#[cfg(test)]
mod tests {

    #[test]
    #[cfg(feature = "integration-tests")]
    fn test_demo_runs_without_panic() {
        // This test ensures the demo code runs without panicking
        run_demo();
    }
}
