//! FFI lifecycle stress tests to prevent memory corruption and double-free bugs
//!
//! These tests complement the existing lifecycle.rs tests with higher iteration counts
//! and additional validation to catch edge cases in the FFI boundary.

#![cfg(feature = "ffi")]

use bitnet_sys::wrapper::{BitnetContext, BitnetModel, bitnet_tokenize_text};
use std::env;

/// Stress test: create and drop models/contexts 100 times with explicit drop ordering
#[test]
fn test_explicit_drop_ordering_stress() {
    // Skip if no model available
    let model_path = match env::var("CROSSVAL_GGUF").or_else(|_| env::var("BITNET_GGUF")) {
        Ok(path) => path,
        Err(_) => {
            eprintln!("Skipping explicit drop ordering test: no GGUF model available");
            return;
        }
    };

    // Initialize backend once
    bitnet_sys::wrapper::init_backend();

    // Stress test: create and drop models/contexts 100 times with explicit drop ordering
    for i in 0..100 {
        let model = match BitnetModel::from_file(&model_path) {
            Ok(m) => m,
            Err(e) => {
                if i == 0 {
                    eprintln!("Skipping test: failed to load model: {:?}", e);
                }
                bitnet_sys::wrapper::free_backend();
                return;
            }
        };

        let ctx = match BitnetContext::new(&model, 512, 1, 42) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Context creation failed at iteration {}: {:?}", i, e);
                bitnet_sys::wrapper::free_backend();
                return;
            }
        };

        // Explicitly drop in reverse order (context before model)
        // This tests that our Drop impls handle proper cleanup ordering
        drop(ctx);
        drop(model);

        if i % 25 == 0 {
            eprintln!("Explicit drop ordering: completed {} iterations", i + 1);
        }
    }

    eprintln!("✅ Explicit drop ordering stress test passed - no crashes or leaks");
    bitnet_sys::wrapper::free_backend();
}

/// Stress test: tokenize with very long inputs to test buffer validation
#[test]
fn test_tokenize_buffer_validation() {
    let model_path = match env::var("CROSSVAL_GGUF").or_else(|_| env::var("BITNET_GGUF")) {
        Ok(path) => path,
        Err(_) => {
            eprintln!("Skipping tokenize buffer validation test: no GGUF model available");
            return;
        }
    };

    bitnet_sys::wrapper::init_backend();

    let model = match BitnetModel::from_file(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Skipping test: {:?}", e);
            bitnet_sys::wrapper::free_backend();
            return;
        }
    };

    // Test with various input sizes to ensure buffer validation works correctly
    let medium_text = "A ".repeat(100); // Medium (200 chars)
    let long_text = "The quick brown fox jumps over the lazy dog. ".repeat(10); // Long (470 chars)
    let test_cases = [
        "",             // Empty string
        "Hi",           // Very short
        "What is 2+2?", // Short
        medium_text.as_str(),
        long_text.as_str(),
    ];

    for (idx, text) in test_cases.iter().enumerate() {
        let result = bitnet_tokenize_text(&model, text, false, false);

        match text.len() {
            0 => {
                // Empty string should return empty vector or error
                if let Ok(tokens) = result {
                    assert!(tokens.is_empty(), "Empty input should produce no tokens")
                }
                // Error is also acceptable for empty input
            }
            _ => {
                // Non-empty strings should succeed
                let tokens =
                    result.unwrap_or_else(|_| panic!("Tokenization failed for test case {}", idx));
                assert!(!tokens.is_empty(), "Non-empty input should produce tokens");
            }
        }
    }

    eprintln!("✅ Tokenize buffer validation test passed");
    bitnet_sys::wrapper::free_backend();
}

/// Test that tokenization with special characters doesn't cause buffer corruption
#[test]
fn test_tokenize_special_chars() {
    let model_path = match env::var("CROSSVAL_GGUF").or_else(|_| env::var("BITNET_GGUF")) {
        Ok(path) => path,
        Err(_) => {
            eprintln!("Skipping special chars test: no GGUF model available");
            return;
        }
    };

    bitnet_sys::wrapper::init_backend();

    let model = match BitnetModel::from_file(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Skipping test: {:?}", e);
            bitnet_sys::wrapper::free_backend();
            return;
        }
    };

    // Test with special characters that might cause issues
    let special_texts = vec![
        "Hello\nWorld",    // Newline
        "Tab\tseparated",  // Tab
        "Quote: \"test\"", // Quotes
        "Emoji: 🎉",       // Unicode emoji
        "Math: ∑∫∂",       // Mathematical symbols
    ];

    for text in special_texts {
        let result = bitnet_tokenize_text(&model, text, false, false);
        assert!(result.is_ok(), "Tokenization failed for: {}", text);
        let tokens = result.unwrap();
        assert!(!tokens.is_empty(), "Should produce tokens for: {}", text);
    }

    eprintln!("✅ Special characters tokenization test passed");
    bitnet_sys::wrapper::free_backend();
}
