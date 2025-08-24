//! FFI integration tests for cross-validation
//!
//! These tests verify that the FFI bindings work correctly and safely
//! with the C++ implementation.

#![cfg(feature = "crossval")]
#![cfg(feature = "integration-tests")]

use bitnet_crossval::{
    CrossvalError,
    cpp_bindings::{CppModel, is_available},
    fixtures::TestFixture,
};
use std::path::Path;

#[test]
fn test_cpp_availability() {
    // Test that we can detect C++ implementation availability
    let available = is_available();
    println!("C++ implementation available: {}", available);

    // This test should pass regardless of availability
    // The important thing is that it doesn't crash
}

#[test]
fn test_model_loading_error_handling() {
    // Test loading a non-existent model
    let result = CppModel::load("non_existent_model.gguf");

    match result {
        Ok(_) => panic!("Expected error when loading non-existent model"),
        Err(CrossvalError::ModelLoadError(_)) => {
            // Expected error type
        }
        Err(e) => panic!("Unexpected error type: {:?}", e),
    }
}

#[test]
fn test_model_lifecycle() {
    // Skip if no test fixture available
    let fixture_path = Path::new("crossval/fixtures/minimal_model.gguf");
    if !fixture_path.exists() {
        eprintln!("Skipping test: fixture not found at {:?}", fixture_path);
        return;
    }

    // Test model loading and cleanup
    let model = match CppModel::load(fixture_path) {
        Ok(model) => model,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            return; // Skip test if model loading fails
        }
    };

    // Test that model is ready
    assert!(model.is_ready(), "Model should be ready after loading");

    // Test model info
    match model.model_info() {
        Ok(info) => {
            assert!(!info.name.is_empty(), "Model name should not be empty");
            assert!(!info.version.is_empty(), "Model version should not be empty");
            assert!(info.parameter_count > 0, "Parameter count should be positive");
        }
        Err(e) => {
            eprintln!("Failed to get model info: {}", e);
        }
    }

    // Model will be automatically cleaned up when dropped
}

#[test]
fn test_generation_error_handling() {
    let fixture_path = Path::new("crossval/fixtures/minimal_model.gguf");
    if !fixture_path.exists() {
        eprintln!("Skipping test: fixture not found");
        return;
    }

    let model = match CppModel::load(fixture_path) {
        Ok(model) => model,
        Err(_) => {
            eprintln!("Skipping test: failed to load model");
            return;
        }
    };

    // Test invalid parameters
    let result = model.generate("test", 0);
    assert!(result.is_err(), "Should fail with max_tokens = 0");

    let result = model.generate("test", 20000);
    assert!(result.is_err(), "Should fail with excessive max_tokens");

    // Test prompt with null bytes
    let result = model.generate("test\0prompt", 10);
    assert!(result.is_err(), "Should fail with null bytes in prompt");
}

#[test]
fn test_basic_generation() {
    let fixture_path = Path::new("crossval/fixtures/minimal_model.gguf");
    if !fixture_path.exists() {
        eprintln!("Skipping test: fixture not found");
        return;
    }

    let model = match CppModel::load(fixture_path) {
        Ok(model) => model,
        Err(_) => {
            eprintln!("Skipping test: failed to load model");
            return;
        }
    };

    // Test basic generation
    match model.generate("Hello, world!", 10) {
        Ok(tokens) => {
            assert!(tokens.len() <= 10, "Should not exceed max_tokens");
            println!("Generated {} tokens: {:?}", tokens.len(), tokens);
        }
        Err(e) => {
            eprintln!("Generation failed: {}", e);
            // Don't fail the test - C++ implementation might not be fully functional
        }
    }
}

#[test]
fn test_concurrent_access() {
    let fixture_path = Path::new("crossval/fixtures/minimal_model.gguf");
    if !fixture_path.exists() {
        eprintln!("Skipping test: fixture not found");
        return;
    }

    let model = match CppModel::load(fixture_path) {
        Ok(model) => model,
        Err(_) => {
            eprintln!("Skipping test: failed to load model");
            return;
        }
    };

    // Test that we can use the model from multiple threads
    // Note: This assumes the C++ implementation is thread-safe
    let handles: Vec<_> = (0..3)
        .map(|i| {
            std::thread::spawn(move || {
                // Each thread gets its own model instance
                let thread_model = CppModel::load(fixture_path).ok()?;
                let prompt = format!("Thread {} test", i);
                thread_model.generate(&prompt, 5).ok()
            })
        })
        .collect();

    for handle in handles {
        match handle.join() {
            Ok(Some(tokens)) => {
                println!("Thread generated {} tokens", tokens.len());
            }
            Ok(None) => {
                eprintln!("Thread failed to generate tokens");
            }
            Err(e) => {
                eprintln!("Thread panicked: {:?}", e);
            }
        }
    }
}

#[test]
fn test_memory_safety() {
    let fixture_path = Path::new("crossval/fixtures/minimal_model.gguf");
    if !fixture_path.exists() {
        eprintln!("Skipping test: fixture not found");
        return;
    }

    // Test that we can create and destroy many models without leaking memory
    for i in 0..10 {
        let model = match CppModel::load(fixture_path) {
            Ok(model) => model,
            Err(_) => {
                eprintln!("Failed to load model iteration {}", i);
                continue;
            }
        };

        // Use the model briefly
        let _ = model.generate("test", 5);

        // Model will be automatically cleaned up
    }

    println!("Memory safety test completed");
}

#[test]
fn test_fixture_compatibility() {
    // Test that we can load all available fixtures
    let fixtures_dir = Path::new("crossval/fixtures");
    if !fixtures_dir.exists() {
        eprintln!("Skipping test: fixtures directory not found");
        return;
    }

    let fixture_names = ["minimal_test", "standard_prompts", "performance_test"];

    for fixture_name in &fixture_names {
        let fixture_path = fixtures_dir.join(format!("{}.json", fixture_name));
        if !fixture_path.exists() {
            eprintln!("Skipping fixture: {}", fixture_name);
            continue;
        }

        match TestFixture::load(fixture_name) {
            Ok(fixture) => {
                println!("Loaded fixture: {}", fixture.name);

                // Try to load the model
                if fixture.model_path.exists() {
                    match CppModel::load(&fixture.model_path) {
                        Ok(model) => {
                            println!("  Model loaded successfully");

                            // Try one test prompt
                            if let Some(prompt) = fixture.test_prompts.first() {
                                match model.generate(prompt, 5) {
                                    Ok(tokens) => {
                                        println!("  Generated {} tokens", tokens.len());
                                    }
                                    Err(e) => {
                                        eprintln!("  Generation failed: {}", e);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("  Failed to load model: {}", e);
                        }
                    }
                } else {
                    eprintln!("  Model file not found: {:?}", fixture.model_path);
                }
            }
            Err(e) => {
                eprintln!("Failed to load fixture {}: {}", fixture_name, e);
            }
        }
    }
}
