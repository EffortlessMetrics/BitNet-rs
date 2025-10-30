//! Tests for --dump-ids and --dump-cpp-ids flags in crossval-per-token command
//!
//! These tests verify that the token ID dumping functionality works correctly
//! and integrates properly with the CLI interface.
//!
//! ## Implementation Details
//!
//! The flags are implemented in `xtask/src/main.rs`:
//! - `--dump-ids`: Prints Rust token IDs to stderr in format:
//!   ```
//!   🦀 Rust tokens (N total):
//!     [token1, token2, ...]
//!   ```
//! - `--dump-cpp-ids`: Prints C++ token IDs to stderr in format:
//!   ```
//!   🔧 C++ tokens (N total, backend: bitnet|llama):
//!     [token1, token2, ...]
//!   ```
//!
//! Both flags output to stderr to avoid polluting JSON output when using --format json.

#![cfg(feature = "inference")]

use assert_cmd::prelude::*;
use std::process::Command;

/// Helper to build xtask command
fn xtask_cmd() -> Command {
    cargo_bin_cmd!("xtask")
}

// ============================================================================
// CLI Parsing Tests
// ============================================================================

/// Test that --dump-ids flag is recognized and parsed correctly
#[test]
fn test_dump_ids_flag_parsing() {
    let mut cmd = xtask_cmd();
    cmd.args([
        "crossval-per-token",
        "--model",
        "/nonexistent/model.gguf",
        "--tokenizer",
        "/nonexistent/tokenizer.json",
        "--prompt",
        "test",
        "--dump-ids", // Flag under test
    ]);

    // Should fail due to missing model, but NOT due to unrecognized flag
    let output = cmd.output().expect("Failed to execute command");
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should NOT contain "unexpected argument" or "unrecognized"
    assert!(!stderr.contains("unexpected argument"), "Flag --dump-ids should be recognized");
    assert!(!stderr.contains("unrecognized"), "Flag --dump-ids should be recognized");
}

/// Test that --dump-cpp-ids flag is recognized and parsed correctly
#[test]
fn test_dump_cpp_ids_flag_parsing() {
    let mut cmd = xtask_cmd();
    cmd.args([
        "crossval-per-token",
        "--model",
        "/nonexistent/model.gguf",
        "--tokenizer",
        "/nonexistent/tokenizer.json",
        "--prompt",
        "test",
        "--dump-cpp-ids", // Flag under test
    ]);

    // Should fail due to missing model, but NOT due to unrecognized flag
    let output = cmd.output().expect("Failed to execute command");
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should NOT contain "unexpected argument" or "unrecognized"
    assert!(!stderr.contains("unexpected argument"), "Flag --dump-cpp-ids should be recognized");
    assert!(!stderr.contains("unrecognized"), "Flag --dump-cpp-ids should be recognized");
}

/// Test that both flags can be used together
#[test]
fn test_both_dump_flags_combined() {
    let mut cmd = xtask_cmd();
    cmd.args([
        "crossval-per-token",
        "--model",
        "/nonexistent/model.gguf",
        "--tokenizer",
        "/nonexistent/tokenizer.json",
        "--prompt",
        "test",
        "--dump-ids", // Both flags
        "--dump-cpp-ids",
    ]);

    // Should fail due to missing model, but NOT due to flag conflicts
    let output = cmd.output().expect("Failed to execute command");
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(!stderr.contains("unexpected argument"), "Both flags should be recognized together");
}

/// Test that dump flags work with other common flags
#[test]
fn test_dump_flags_with_other_options() {
    let mut cmd = xtask_cmd();
    cmd.args([
        "crossval-per-token",
        "--model",
        "/nonexistent/model.gguf",
        "--tokenizer",
        "/nonexistent/tokenizer.json",
        "--prompt",
        "What is 2+2?",
        "--max-tokens",
        "4",
        "--cos-tol",
        "0.999",
        "--format",
        "json",
        "--verbose",
        "--dump-ids",
        "--dump-cpp-ids",
    ]);

    let output = cmd.output().expect("Failed to execute command");
    let stderr = String::from_utf8_lossy(&output.stderr);

    // All flags should be recognized
    assert!(!stderr.contains("unexpected argument"), "All flags should be recognized together");
}

// ============================================================================
// Integration Tests (require model files)
// ============================================================================

/// Helper to get test model path from environment
fn get_test_model_path() -> Option<std::path::PathBuf> {
    std::env::var("BITNET_GGUF").ok().map(std::path::PathBuf::from).or_else(|| {
        let default = std::path::PathBuf::from(
            "models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf",
        );
        if default.exists() { Some(default) } else { None }
    })
}

/// Helper to get test tokenizer path
fn get_test_tokenizer_path() -> Option<std::path::PathBuf> {
    std::env::var("BITNET_TOKENIZER").ok().map(std::path::PathBuf::from).or_else(|| {
        let default =
            std::path::PathBuf::from("models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json");
        if default.exists() { Some(default) } else { None }
    })
}

/// Test that --dump-ids produces expected output format
#[test]
#[ignore = "Requires model file and may require C++ FFI setup"]
fn test_dump_ids_output_format() {
    let model = match get_test_model_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: BITNET_GGUF not set and default model not found");
            return;
        }
    };

    let tokenizer = match get_test_tokenizer_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: tokenizer.json not found");
            return;
        }
    };

    let mut cmd = xtask_cmd();
    cmd.args([
        "crossval-per-token",
        "--model",
        model.to_str().unwrap(),
        "--tokenizer",
        tokenizer.to_str().unwrap(),
        "--prompt",
        "test",
        "--max-tokens",
        "1",
        "--dump-ids",
    ]);

    let output = cmd.output().expect("Failed to execute command");
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Expected format:
    // 🦀 Rust tokens (N total):
    //   [token1, token2, ...]
    assert!(stderr.contains("🦀 Rust tokens"), "Should contain Rust tokens header");
    assert!(stderr.contains("total"), "Should show total count");

    // Should contain array format with brackets
    assert!(stderr.contains('[') && stderr.contains(']'), "Should display tokens in array format");
}

/// Test that --dump-cpp-ids produces expected output format
#[test]
#[ignore = "Requires model file and C++ FFI setup"]
fn test_dump_cpp_ids_output_format() {
    let model = match get_test_model_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: BITNET_GGUF not set and default model not found");
            return;
        }
    };

    let tokenizer = match get_test_tokenizer_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: tokenizer.json not found");
            return;
        }
    };

    let mut cmd = xtask_cmd();
    cmd.args([
        "crossval-per-token",
        "--model",
        model.to_str().unwrap(),
        "--tokenizer",
        tokenizer.to_str().unwrap(),
        "--prompt",
        "test",
        "--max-tokens",
        "1",
        "--dump-cpp-ids",
    ]);

    let output = cmd.output().expect("Failed to execute command");
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Expected format:
    // 🔧 C++ tokens (N total, backend: bitnet|llama):
    //   [token1, token2, ...]
    assert!(stderr.contains("🔧 C++ tokens"), "Should contain C++ tokens header");
    assert!(
        stderr.contains("total") && stderr.contains("backend:"),
        "Should show total count and backend name"
    );

    // Should contain array format with brackets
    assert!(stderr.contains('[') && stderr.contains(']'), "Should display tokens in array format");
}

/// Test that both dumps show tokens when used together
#[test]
#[ignore = "Requires model file and C++ FFI setup"]
fn test_both_dumps_show_tokens() {
    let model = match get_test_model_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: BITNET_GGUF not set and default model not found");
            return;
        }
    };

    let tokenizer = match get_test_tokenizer_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: tokenizer.json not found");
            return;
        }
    };

    let mut cmd = xtask_cmd();
    cmd.args([
        "crossval-per-token",
        "--model",
        model.to_str().unwrap(),
        "--tokenizer",
        tokenizer.to_str().unwrap(),
        "--prompt",
        "test",
        "--max-tokens",
        "1",
        "--dump-ids",
        "--dump-cpp-ids",
    ]);

    let output = cmd.output().expect("Failed to execute command");
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should contain both outputs
    assert!(stderr.contains("🦀 Rust tokens"), "Should contain Rust tokens output");
    assert!(stderr.contains("🔧 C++ tokens"), "Should contain C++ tokens output");
}

/// Test that dumps go to stderr, not stdout (preserves JSON output)
#[test]
#[ignore = "Requires model file and C++ FFI setup"]
fn test_dumps_to_stderr_not_stdout() {
    let model = match get_test_model_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: BITNET_GGUF not set and default model not found");
            return;
        }
    };

    let tokenizer = match get_test_tokenizer_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: tokenizer.json not found");
            return;
        }
    };

    let mut cmd = xtask_cmd();
    cmd.args([
        "crossval-per-token",
        "--model",
        model.to_str().unwrap(),
        "--tokenizer",
        tokenizer.to_str().unwrap(),
        "--prompt",
        "test",
        "--max-tokens",
        "1",
        "--format",
        "json",
        "--dump-ids",
        "--dump-cpp-ids",
    ]);

    let output = cmd.output().expect("Failed to execute command");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Stdout should only contain JSON (if successful) or error message
    // It should NOT contain token dump output
    assert!(!stdout.contains("🦀 Rust tokens"), "Rust tokens should go to stderr, not stdout");
    assert!(!stdout.contains("🔧 C++ tokens"), "C++ tokens should go to stderr, not stdout");

    // Stderr should contain the dumps
    assert!(
        stderr.contains("🦀 Rust tokens") || stderr.contains("🔧 C++ tokens"),
        "Token dumps should appear in stderr"
    );
}

// ============================================================================
// Documentation Tests
// ============================================================================

/// Verify that the help text mentions the dump flags
#[test]
#[ignore = "Requires shared libraries to be available - flags are defined in source"]
fn test_help_text_includes_dump_flags() {
    let mut cmd = xtask_cmd();
    cmd.args(["crossval-per-token", "--help"]);

    let output = cmd.output().expect("Failed to get help text");
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(stdout.contains("--dump-ids"), "Help text should mention --dump-ids flag");
    assert!(stdout.contains("--dump-cpp-ids"), "Help text should mention --dump-cpp-ids flag");
}
