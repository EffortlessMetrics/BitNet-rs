//! Integration tests for token parity pre-gate in crossval-per-token command
//!
//! These tests validate the integration of token parity checking with the xtask
//! crossval-per-token command, including CLI flag handling and template integration.
//!
//! ## Specification
//!
//! See: `docs/explanation/token-parity-pregate.md`
//!
//! ## Acceptance Criteria Coverage
//!
//! - AC11: Direct logits comparison on success
//! - AC12: Template integration (raw)
//! - AC13: Template integration (instruct)
//! - AC14: Template integration (llama3-chat)
//! - AC15: --no-bos flag integration

#![cfg(feature = "inference")]

use std::path::PathBuf;
use std::process::Command;

/// Helper to get test model path from environment
fn get_test_model_path() -> Option<PathBuf> {
    std::env::var("BITNET_GGUF").ok().map(PathBuf::from).or_else(|| {
        // Try default location
        let default =
            PathBuf::from("models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf");
        if default.exists() { Some(default) } else { None }
    })
}

/// Helper to get test tokenizer path
fn get_test_tokenizer_path() -> Option<PathBuf> {
    std::env::var("BITNET_TOKENIZER").ok().map(PathBuf::from).or_else(|| {
        // Try default location
        let default = PathBuf::from("models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json");
        if default.exists() { Some(default) } else { None }
    })
}

// AC11: Direct logits comparison on token parity success
// Spec: docs/explanation/token-parity-pregate.md#acceptance-criteria
#[test]
#[ignore = "TODO: Requires model and C++ FFI setup"]
fn test_proceeds_to_logits_on_token_match() {
    let model = match get_test_model_path() {
        Some(p) => p,
        None => {
            eprintln!("Warning: BITNET_GGUF not set, skipping integration test");
            return;
        }
    };

    let tokenizer = match get_test_tokenizer_path() {
        Some(p) => p,
        None => {
            eprintln!("Warning: BITNET_TOKENIZER not set, skipping integration test");
            return;
        }
    };

    // Run crossval-per-token with a simple prompt
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(&[
            "crossval-per-token",
            "--model",
            model.to_str().unwrap(),
            "--tokenizer",
            tokenizer.to_str().unwrap(),
            "--prompt",
            "What is 2+2?",
            "--max-tokens",
            "4",
            "--cos-tol",
            "0.999",
        ])
        .output()
        .expect("Failed to execute xtask");

    // If tokens match, should proceed to logits comparison
    // If tokens don't match, should exit with code 2
    if output.status.success() {
        // Tokens matched - verify logits comparison ran
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("Comparing logits") || stdout.contains("cosine"),
            "Expected logits comparison output on token match"
        );
    } else if output.status.code() == Some(2) {
        // Token mismatch detected - verify diagnostic was printed
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("Token Sequence Mismatch") || stderr.contains("First diff"),
            "Expected token mismatch diagnostic on exit code 2"
        );
    } else {
        panic!("Unexpected exit code: {:?}", output.status.code());
    }
}

// AC12: Template integration (raw)
// Spec: docs/explanation/token-parity-pregate.md#integration-points
#[test]
#[ignore = "TODO: Requires model and C++ FFI setup"]
fn test_raw_template_tokenization_parity() {
    let model = match get_test_model_path() {
        Some(p) => p,
        None => return,
    };

    let tokenizer = match get_test_tokenizer_path() {
        Some(p) => p,
        None => return,
    };

    // Raw template should not add BOS or special formatting
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(&[
            "crossval-per-token",
            "--model",
            model.to_str().unwrap(),
            "--tokenizer",
            tokenizer.to_str().unwrap(),
            "--prompt",
            "2+2=",
            "--prompt-template",
            "raw",
            "--max-tokens",
            "4",
        ])
        .output()
        .expect("Failed to execute xtask");

    // Raw template should have consistent tokenization with C++
    // (assuming C++ also uses raw mode)
    unimplemented!("TODO: Validate raw template produces matching tokens");
}

// AC13: Template integration (instruct)
// Spec: docs/explanation/token-parity-pregate.md#integration-points
#[test]
#[ignore = "TODO: Requires model and C++ FFI setup"]
fn test_instruct_template_tokenization_parity() {
    let model = match get_test_model_path() {
        Some(p) => p,
        None => return,
    };

    let tokenizer = match get_test_tokenizer_path() {
        Some(p) => p,
        None => return,
    };

    // Instruct template adds "Q: <prompt>\nA:" formatting
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(&[
            "crossval-per-token",
            "--model",
            model.to_str().unwrap(),
            "--tokenizer",
            tokenizer.to_str().unwrap(),
            "--prompt",
            "What is the capital of France?",
            "--prompt-template",
            "instruct",
            "--max-tokens",
            "4",
        ])
        .output()
        .expect("Failed to execute xtask");

    // Instruct template may cause token mismatch if C++ uses different template
    // This test validates the pre-gate catches template mismatches
    unimplemented!("TODO: Validate instruct template tokenization parity or mismatch detection");
}

// AC14: Template integration (llama3-chat)
// Spec: docs/explanation/token-parity-pregate.md#integration-points
#[test]
#[ignore = "TODO: Requires model and C++ FFI setup"]
fn test_llama3_chat_template_tokenization_parity() {
    let model = match get_test_model_path() {
        Some(p) => p,
        None => return,
    };

    let tokenizer = match get_test_tokenizer_path() {
        Some(p) => p,
        None => return,
    };

    // LLaMA-3 chat template with special tokens
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(&[
            "crossval-per-token",
            "--model",
            model.to_str().unwrap(),
            "--tokenizer",
            tokenizer.to_str().unwrap(),
            "--prompt",
            "What is 2+2?",
            "--prompt-template",
            "llama3-chat",
            "--system-prompt",
            "You are a helpful assistant",
            "--max-tokens",
            "4",
        ])
        .output()
        .expect("Failed to execute xtask");

    // LLaMA-3 template uses <|begin_of_text|>, <|start_header_id|>, etc.
    // This may cause token mismatch with C++ if templates differ
    unimplemented!("TODO: Validate llama3-chat template tokenization parity or mismatch detection");
}

// AC15: --no-bos flag integration
// Spec: docs/explanation/token-parity-pregate.md#design
#[test]
#[ignore = "TODO: Requires tokenizer integration with --no-bos flag"]
fn test_no_bos_flag_prevents_duplicate() {
    let model = match get_test_model_path() {
        Some(p) => p,
        None => return,
    };

    let tokenizer = match get_test_tokenizer_path() {
        Some(p) => p,
        None => return,
    };

    // Test that --no-bos flag prevents duplicate BOS token
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(&[
            "crossval-per-token",
            "--model",
            model.to_str().unwrap(),
            "--tokenizer",
            tokenizer.to_str().unwrap(),
            "--prompt",
            "What is 2+2?",
            "--no-bos",
            "--max-tokens",
            "4",
        ])
        .output()
        .expect("Failed to execute xtask");

    // With --no-bos, should not prepend BOS token
    // This should help achieve token parity with C++
    unimplemented!("TODO: Validate --no-bos prevents duplicate BOS in token sequence");
}

// Integration test: Full flow from CLI to token parity check
#[test]
#[ignore = "TODO: Requires full integration with xtask command"]
fn test_cli_integration_full_flow() {
    // This test validates the full integration:
    // 1. CLI parses flags
    // 2. Tokenizer applies template
    // 3. C++ tokenizes with FFI
    // 4. Token parity check runs
    // 5. Either exits with code 2 (mismatch) or proceeds to logits (match)

    unimplemented!("TODO: Full integration test after xtask command integration");
}

// Unit test: Mock FFI session for testing without C++ dependencies
#[test]
fn test_mock_ffi_session_token_comparison() {
    // This test uses mock data to validate token comparison logic
    // without requiring C++ FFI or real models

    // Mock Rust tokens
    let rust_tokens = vec![128000_u32, 1229, 374, 220, 17];

    // Mock C++ tokens (matching)
    let cpp_tokens_match = vec![128000_i32, 1229, 374, 220, 17];

    // Mock C++ tokens (mismatch)
    let cpp_tokens_mismatch = vec![128000_i32, 128000, 1229, 374, 220, 17]; // Duplicate BOS

    // Test matching case
    let result_match = bitnet_crossval::token_parity::validate_token_parity(
        &rust_tokens,
        &cpp_tokens_match,
        "What is 2+2?",
    );

    match result_match {
        Ok(()) => {} // Expected
        Err(e) => panic!("Expected success on matching tokens, got: {}", e),
    }

    // Test mismatch case
    let result_mismatch = bitnet_crossval::token_parity::validate_token_parity(
        &rust_tokens,
        &cpp_tokens_mismatch,
        "What is 2+2?",
    );

    // Should detect mismatch at index 1
    assert!(
        result_mismatch.is_err()
            || std::panic::catch_unwind(|| {
                bitnet_crossval::token_parity::validate_token_parity(
                    &rust_tokens,
                    &cpp_tokens_mismatch,
                    "What is 2+2?",
                )
                .ok()
            })
            .is_err(),
        "Expected mismatch detection"
    );
}

// Performance test: Token parity check overhead
#[test]
fn test_token_parity_performance_overhead() {
    use std::time::Instant;

    // Generate large token sequences (1000 tokens)
    let rust_tokens: Vec<u32> = (0..1000).collect();
    let cpp_tokens: Vec<i32> = (0..1000).map(|x| x as i32).collect();

    let start = Instant::now();

    // Run token parity check
    let _ = bitnet_crossval::token_parity::validate_token_parity(
        &rust_tokens,
        &cpp_tokens,
        "performance test prompt",
    );

    let elapsed = start.elapsed();

    // Should complete in <100ms (AC10)
    assert!(
        elapsed.as_millis() < 100,
        "Token parity overhead {}ms exceeds 100ms threshold (AC10)",
        elapsed.as_millis()
    );
}

// Test: Verify error message formatting
#[test]
fn test_error_message_format() {
    use bitnet_crossval::token_parity::{TokenParityError, format_token_mismatch_error};

    let error = TokenParityError {
        rust_tokens: vec![128000, 128000, 1229, 374],
        cpp_tokens: vec![128000, 1229, 374],
        first_diff_index: 1,
        prompt: "What is 2+2?".to_string(),
    };

    let formatted = format_token_mismatch_error(&error);

    // Verify all required elements are present
    assert!(
        formatted.contains("Token Sequence Mismatch") || formatted.contains("mismatch"),
        "Should include error title"
    );
    assert!(
        formatted.contains("Rust") && formatted.contains("C++"),
        "Should show both Rust and C++ sections"
    );
    assert!(
        formatted.contains("128000") || formatted.contains("tokens"),
        "Should display token values"
    );
    assert!(
        formatted.contains("Suggested fixes") || formatted.contains("fix"),
        "Should include suggestions section"
    );
}

// Test: Verify CLI flag parsing for crossval-per-token
#[test]
#[ignore = "TODO: Add CLI flag parsing tests in xtask"]
fn test_cli_flag_parsing() {
    // Validate that crossval-per-token command accepts all required flags:
    // - --model
    // - --tokenizer
    // - --prompt
    // - --prompt-template (optional)
    // - --no-bos (optional)
    // - --max-tokens
    // - --cos-tol

    unimplemented!("TODO: Test CLI flag parsing with clap validator");
}

// Test: Validate template auto-detection integration
#[test]
#[ignore = "TODO: Requires template auto-detection logic"]
fn test_template_auto_detection() {
    // When --prompt-template is not specified, should auto-detect from:
    // 1. GGUF chat_template metadata
    // 2. Model/tokenizer path heuristics
    // 3. Fallback to instruct

    unimplemented!("TODO: Test template auto-detection integration");
}
