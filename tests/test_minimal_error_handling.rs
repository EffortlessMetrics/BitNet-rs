/// Minimal test for core error handling functionality
///
/// This test verifies the enhanced error handling works without
/// complex dependencies that cause compilation issues.
use std::time::Duration;

#[test]
fn test_basic_error_functionality() {
    // Test that we can create different error types
    let timeout_error = bitnet_tests::TestError::timeout(Duration::from_secs(30));
    let assertion_error = bitnet_tests::TestError::assertion("Test failed");
    let config_error = bitnet_tests::TestError::config("Invalid config");

    // Test error categorization
    assert_eq!(timeout_error.category(), "timeout");
    assert_eq!(assertion_error.category(), "assertion");
    assert_eq!(config_error.category(), "config");

    // Test severity levels
    assert_eq!(timeout_error.severity(), bitnet_tests::ErrorSeverity::Medium);
    assert_eq!(assertion_error.severity(), bitnet_tests::ErrorSeverity::High);
    assert_eq!(config_error.severity(), bitnet_tests::ErrorSeverity::Low);

    // Test recoverability
    assert!(timeout_error.is_recoverable());
    assert!(!assertion_error.is_recoverable());
    assert!(!config_error.is_recoverable());

    println!("✓ Basic error functionality working correctly");
}

#[test]
fn test_error_recovery_suggestions() {
    let timeout_error = bitnet_tests::TestError::timeout(Duration::from_secs(45));
    let suggestions = timeout_error.recovery_suggestions();

    assert!(!suggestions.is_empty(), "Timeout errors should have recovery suggestions");

    // Check that suggestions are relevant to timeout errors
    let has_timeout_suggestion = suggestions.iter().any(|s| {
        s.to_lowercase().contains("timeout")
            || s.to_lowercase().contains("time")
            || s.to_lowercase().contains("resource")
    });
    assert!(has_timeout_suggestion, "Should have timeout-related suggestions");

    println!("✓ Error recovery suggestions working correctly");
    println!("Sample suggestions for timeout error:");
    for (i, suggestion) in suggestions.iter().take(3).enumerate() {
        println!("  {}. {}", i + 1, suggestion);
    }
}

#[test]
fn test_error_troubleshooting_steps() {
    let error = bitnet_tests::TestError::assertion("Expected 42 but got 24");
    let steps = error.troubleshooting_steps();

    assert!(!steps.is_empty(), "Should provide troubleshooting steps");
    assert!(steps.len() >= 2, "Should have multiple troubleshooting steps");

    // Verify steps are properly structured
    for (i, step) in steps.iter().enumerate() {
        assert_eq!(step.step_number, (i + 1) as u32, "Steps should be numbered sequentially");
        assert!(!step.title.is_empty(), "Each step should have a title");
        assert!(!step.description.is_empty(), "Each step should have a description");
    }

    println!("✓ Error troubleshooting steps working correctly");
    println!("Sample troubleshooting steps for assertion error:");
    for step in steps.iter().take(3) {
        println!("  {}. {} - {}", step.step_number, step.title, step.description);
    }
}

#[test]
fn test_error_debug_info() {
    let fixture_error = bitnet_tests::TestError::FixtureError(
        bitnet_tests::FixtureError::download("https://example.com/model.gguf", "Network error"),
    );

    let debug_info = fixture_error.debug_info();

    assert_eq!(debug_info.category, "fixture");
    assert_eq!(debug_info.severity, bitnet_tests::ErrorSeverity::Medium);
    assert!(debug_info.recoverable);
    assert!(!debug_info.recovery_suggestions.is_empty());
    assert!(!debug_info.related_components.is_empty());
    assert!(!debug_info.troubleshooting_steps.is_empty());

    // Check that related components make sense
    assert!(debug_info.related_components.contains(&"fixture_manager".to_string()));

    println!("✓ Error debug info working correctly");
    println!("Related components: {:?}", debug_info.related_components);
}

#[test]
fn test_error_report_generation() {
    let error = bitnet_tests::TestError::execution("Process failed with exit code 1");
    let report = error.create_error_report();

    assert_eq!(report.error_category, "execution");
    assert_eq!(report.severity, bitnet_tests::ErrorSeverity::High);
    assert!(!report.error_message.is_empty());

    // Test summary generation
    let summary = report.generate_summary();
    assert!(summary.contains("ERROR REPORT"));
    assert!(summary.contains("execution"));
    assert!(summary.contains("RECOVERY SUGGESTIONS"));
    assert!(summary.contains("TROUBLESHOOTING STEPS"));

    println!("✓ Error report generation working correctly");
}

#[test]
fn test_environment_info_collection() {
    let env_info = bitnet_tests::collect_environment_info();

    assert!(!env_info.platform.is_empty());
    assert!(!env_info.architecture.is_empty());
    assert!(!env_info.working_directory.is_empty());
    assert!(env_info.system_resources.cpu_cores > 0);

    println!("✓ Environment info collection working correctly");
    println!("Platform: {} ({})", env_info.platform, env_info.architecture);
    println!("CPU cores: {}", env_info.system_resources.cpu_cores);
}

#[test]
fn test_fixture_error_types() {
    // Test different fixture error types
    let download_error = bitnet_tests::FixtureError::download(
        "https://example.com/model.gguf",
        "Connection timeout",
    );
    let checksum_error =
        bitnet_tests::FixtureError::checksum_mismatch("model.gguf", "expected_hash", "actual_hash");
    let not_found_error = bitnet_tests::FixtureError::not_found("/path/to/model.gguf");

    // Test that they all convert to TestError properly
    let test_errors = vec![
        bitnet_tests::TestError::FixtureError(download_error),
        bitnet_tests::TestError::FixtureError(checksum_error),
        bitnet_tests::TestError::FixtureError(not_found_error),
    ];

    for error in test_errors {
        assert_eq!(error.category(), "fixture");
        assert_eq!(error.severity(), bitnet_tests::ErrorSeverity::Medium);
        assert!(!error.recovery_suggestions().is_empty());
        assert!(!error.related_components().is_empty());
    }

    println!("✓ Fixture error types working correctly");
}

#[test]
fn test_error_severity_ordering() {
    use bitnet_tests::ErrorSeverity;

    // Test that severity levels are properly ordered
    assert!(ErrorSeverity::Low < ErrorSeverity::Medium);
    assert!(ErrorSeverity::Medium < ErrorSeverity::High);
    assert!(ErrorSeverity::High < ErrorSeverity::Critical);

    // Test display
    assert_eq!(ErrorSeverity::Low.to_string(), "LOW");
    assert_eq!(ErrorSeverity::Medium.to_string(), "MEDIUM");
    assert_eq!(ErrorSeverity::High.to_string(), "HIGH");
    assert_eq!(ErrorSeverity::Critical.to_string(), "CRITICAL");

    println!("✓ Error severity ordering working correctly");
}

#[test]
fn test_comprehensive_error_analysis() {
    // Test that all error types provide comprehensive analysis
    let errors = vec![
        bitnet_tests::TestError::timeout(Duration::from_secs(30)),
        bitnet_tests::TestError::assertion("Test failed"),
        bitnet_tests::TestError::config("Invalid parameter"),
        bitnet_tests::TestError::setup("Setup failed"),
        bitnet_tests::TestError::execution("Execution failed"),
    ];

    for error in errors {
        // All errors should have these basic properties
        assert!(!error.category().is_empty());
        assert!(!error.recovery_suggestions().is_empty());
        assert!(!error.related_components().is_empty());
        assert!(!error.troubleshooting_steps().is_empty());

        // Debug info should be comprehensive
        let debug_info = error.debug_info();
        assert!(!debug_info.error_type.is_empty());
        assert_eq!(debug_info.category, error.category());
        assert_eq!(debug_info.severity, error.severity());
        assert_eq!(debug_info.recoverable, error.is_recoverable());
    }

    println!("✓ Comprehensive error analysis working correctly");
}

#[test]
fn test_error_display_formatting() {
    let timeout_error = bitnet_tests::TestError::timeout(Duration::from_secs(45));
    let error_string = timeout_error.to_string();

    assert!(error_string.contains("timeout"));
    assert!(error_string.contains("45s"));

    let assertion_error = bitnet_tests::TestError::assertion("Expected 'hello' but got 'world'");
    let assertion_string = assertion_error.to_string();

    assert!(assertion_string.contains("Assertion failed"));
    assert!(assertion_string.contains("Expected 'hello' but got 'world'"));

    println!("✓ Error display formatting working correctly");
    println!("Timeout error: {}", error_string);
    println!("Assertion error: {}", assertion_string);
}
