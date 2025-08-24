#![cfg(feature = "integration-tests")]
// Simple demonstration of enhanced error handling functionality
// This file can be run independently to show the enhanced error handling works

use std::time::Duration;

// Import the enhanced error handling components
mod common {
    pub mod enhanced_error_handler;
    pub mod error_analysis;
    pub mod errors;
}

use common::enhanced_error_handler::EnhancedErrorHandler;
use common::error_analysis::{ErrorAnalyzer, ErrorContext};
use common::errors::{ErrorSeverity, TestError};

fn main() {
    println!("=== Enhanced Error Handling Demonstration ===\n");

    // Create different types of errors to demonstrate enhanced handling
    let timeout_error = TestError::timeout(Duration::from_secs(30));
    let fixture_error = TestError::fixture("Failed to load test model".to_string());
    let assertion_error = TestError::assertion("Expected 42, got 24".to_string());

    // Demonstrate error severity classification
    println!("🔍 Error Severity Classification:");
    println!("Timeout Error: {:?}", timeout_error.severity());
    println!("Fixture Error: {:?}", fixture_error.severity());
    println!("Assertion Error: {:?}", assertion_error.severity());
    println!();

    // Demonstrate recovery suggestions
    println!("💡 Recovery Suggestions:");
    println!("For Timeout Error:");
    for suggestion in timeout_error.recovery_suggestions() {
        println!("  - {}", suggestion);
    }
    println!();

    println!("For Fixture Error:");
    for suggestion in fixture_error.recovery_suggestions() {
        println!("  - {}", suggestion);
    }
    println!();

    // Demonstrate troubleshooting steps
    println!("🔧 Troubleshooting Steps for Timeout Error:");
    for step in timeout_error.troubleshooting_steps() {
        println!("  {}. {} - {}", step.step_number, step.title, step.description);
        println!("     Estimated effort: {} minutes", step.estimated_effort_minutes);
    }
    println!();

    // Demonstrate comprehensive error report
    println!("📋 Comprehensive Error Report:");
    let report = timeout_error.create_error_report();
    println!("{}", report.generate_summary());
    println!();

    // Demonstrate error analysis
    println!("🧠 Error Analysis:");
    let analyzer = ErrorAnalyzer::new();
    let context = ErrorContext::current();
    let analysis = analyzer.analyze_error(&timeout_error, &context);

    println!("Root cause analysis:");
    for cause in &analysis.root_causes {
        println!("  - {} (confidence: {:.1}%)", cause.description, cause.confidence * 100.0);
    }
    println!();

    println!("Actionable recommendations:");
    for rec in &analysis.recommendations {
        println!(
            "  - {} (priority: {:?}, effort: {} min)",
            rec.description, rec.priority, rec.estimated_effort_minutes
        );
    }
    println!();

    // Demonstrate enhanced error handler
    println!("🚀 Enhanced Error Handler:");
    let mut handler = EnhancedErrorHandler::new();

    // Simulate handling the error
    match handler.handle_error(timeout_error.clone()) {
        Ok(_) => println!("✅ Error handled successfully with recovery"),
        Err(enhanced_error) => {
            println!("❌ Error could not be recovered:");
            println!("   {}", enhanced_error);
            println!(
                "   Debugging guide generated with {} steps",
                enhanced_error.troubleshooting_steps().len()
            );
        }
    }

    println!("\n✅ Enhanced Error Handling Demonstration Complete!");
    println!("The system provides:");
    println!("  • Severity-based error prioritization");
    println!("  • Context-aware recovery suggestions");
    println!("  • Step-by-step troubleshooting guides");
    println!("  • Comprehensive error analysis");
    println!("  • Automatic retry logic for recoverable errors");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_error_handling_works() {
        // Test that all enhanced error handling components work correctly
        let error = TestError::timeout(Duration::from_secs(30));

        // Test severity classification
        assert_eq!(error.severity(), ErrorSeverity::Medium);

        // Test recovery suggestions are provided
        let suggestions = error.recovery_suggestions();
        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.contains("timeout")));

        // Test troubleshooting steps are provided
        let steps = error.troubleshooting_steps();
        assert!(!steps.is_empty());
        assert!(steps[0].step_number == 1);

        // Test error report generation
        let report = error.create_error_report();
        let summary = report.generate_summary();
        assert!(summary.contains("Error Report"));
        assert!(summary.contains("Recovery Suggestions"));

        // Test error analysis
        let analyzer = ErrorAnalyzer::new();
        let context = ErrorContext::current();
        let analysis = analyzer.analyze_error(&error, &context);
        assert!(!analysis.root_causes.is_empty());
        assert!(!analysis.recommendations.is_empty());

        // Test enhanced error handler
        let mut handler = EnhancedErrorHandler::new();
        let result = handler.handle_error(error);
        // Should either succeed with recovery or fail with enhanced error info
        match result {
            Ok(_) => println!("Error was recovered"),
            Err(enhanced_error) => {
                assert!(!enhanced_error.troubleshooting_steps().is_empty());
                assert!(!enhanced_error.recovery_suggestions().is_empty());
            }
        }

        println!("✅ All enhanced error handling tests passed!");
    }
}
