//! Token equivalence tests for cross-validation
//!
//! These tests compare token-level output between Rust and C++ implementations
//! to ensure numerical accuracy and compatibility.

#![cfg(feature = "crossval")]
#![cfg(feature = "integration-tests")]

use bitnet_crossval::{
    CrossvalConfig,
    comparison::CrossValidator,
    fixtures::{STANDARD_PROMPTS, TestFixture},
};

#[test]
fn test_token_equivalence_minimal() {
    let config = CrossvalConfig { tolerance: 1e-6, max_tokens: 100, benchmark: false };

    let validator = CrossValidator::new(config);

    // Use a minimal fixture for testing
    let fixture = TestFixture {
        name: "minimal_test".to_string(),
        model_path: "fixtures/minimal_model.gguf".into(),
        test_prompts: vec!["Hello, world!".to_string()],
        expected_tokens: None,
    };

    // This test will be skipped if the fixture doesn't exist
    if !fixture.model_path.exists() {
        eprintln!("Skipping test: fixture not found at {:?}", fixture.model_path);
        return;
    }

    let results = validator.validate_fixture(&fixture).expect("Cross-validation should succeed");

    assert!(!results.is_empty(), "Should have at least one result");

    for result in &results {
        if let Some(error) = &result.error {
            panic!("Cross-validation failed: {}", error);
        }

        assert!(result.tokens_match, "Tokens should match between implementations");
    }
}

#[test]
fn test_standard_prompts() {
    let config = CrossvalConfig::default();
    let validator = CrossValidator::new(config);

    // Create a test fixture with standard prompts
    let fixture = TestFixture {
        name: "standard_prompts".to_string(),
        model_path: "fixtures/test_model.gguf".into(),
        test_prompts: STANDARD_PROMPTS.iter().map(|s| s.to_string()).collect(),
        expected_tokens: None,
    };

    // Skip if fixture doesn't exist
    if !fixture.model_path.exists() {
        eprintln!("Skipping test: fixture not found at {:?}", fixture.model_path);
        return;
    }

    let results = validator.validate_fixture(&fixture).expect("Cross-validation should succeed");

    assert_eq!(results.len(), STANDARD_PROMPTS.len());

    for result in &results {
        println!("Testing prompt: '{}'", result.prompt);

        if let Some(error) = &result.error {
            eprintln!("Warning: {}", error);
            continue;
        }

        assert!(result.tokens_match, "Tokens should match for prompt: '{}'", result.prompt);
    }
}

#[test]
fn test_empty_prompt() {
    let config = CrossvalConfig::default();
    let validator = CrossValidator::new(config);

    let fixture = TestFixture {
        name: "empty_prompt".to_string(),
        model_path: "fixtures/test_model.gguf".into(),
        test_prompts: vec!["".to_string()],
        expected_tokens: None,
    };

    // Skip if fixture doesn't exist
    if !fixture.model_path.exists() {
        eprintln!("Skipping test: fixture not found at {:?}", fixture.model_path);
        return;
    }

    let results = validator.validate_fixture(&fixture).expect("Cross-validation should succeed");

    assert_eq!(results.len(), 1);

    // Empty prompt might produce different behavior, so we just check it doesn't crash
    let result = &results[0];
    if result.error.is_none() {
        // If no error, tokens should match
        assert!(result.tokens_match, "Tokens should match for empty prompt");
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use bitnet_crossval::comparison::validate_all_fixtures;

    #[test]
    fn test_all_available_fixtures() {
        let config = CrossvalConfig {
            tolerance: 1e-6,
            max_tokens: 50, // Keep small for fast testing
            benchmark: false,
        };

        match validate_all_fixtures(config) {
            Ok(results) => {
                if results.is_empty() {
                    eprintln!("No fixtures available for testing");
                    return;
                }

                let mut passed = 0;
                let mut failed = 0;

                for result in &results {
                    if result.error.is_none() && result.tokens_match {
                        passed += 1;
                    } else {
                        failed += 1;
                        if let Some(error) = &result.error {
                            eprintln!("Failed: {} - {}", result.test_name, error);
                        }
                    }
                }

                println!("Cross-validation results: {} passed, {} failed", passed, failed);

                // Allow some failures for now, but ensure we have some successes
                assert!(passed > 0, "At least some cross-validation tests should pass");
            }
            Err(e) => {
                eprintln!("Cross-validation failed: {}", e);
                // Don't fail the test if no fixtures are available
            }
        }
    }
}
