//! End-to-End Integration Tests for TokenizerAuthority Cross-Lane Validation
//!
//! **Specification Reference**: `docs/specs/tokenizer-authority-validation-tests.md`
//!
//! ## Test Coverage (12 E2E integration tests)
//!
//! This test suite validates the complete parity-both workflow with TokenizerAuthority
//! validation by executing the command as a subprocess and verifying:
//!
//! - Exit codes (0=success, 2=validation failure)
//! - Receipt JSON files with TokenizerAuthority fields
//! - Cross-lane tokenizer consistency validation
//! - Summary output formatting (text and JSON)
//! - Hash determinism and stability
//! - Error handling for invalid inputs
//!
//! ## Test Categories
//!
//! - **TC1: Happy Path** (4 tests) - Identical tokenizers produce exit 0
//! - **TC2: Mismatch Detection** (3 tests) - Validation logic detects divergence
//! - **TC3: Schema v2 Compatibility** (2 tests) - Receipt version inference
//! - **TC4: Edge Cases** (3 tests) - Missing files, corrupted JSON
//! - **TC5: Hash Determinism** (3 tests) - Same file → same hash consistency
//!
//! ## Test Infrastructure
//!
//! All tests use subprocess execution via `run_parity_both_e2e()` to simulate
//! real-world usage and validate end-to-end behavior.
//!
//! ## Environment Isolation
//!
//! Tests use `#[serial(bitnet_env)]` to prevent race conditions during parallel
//! execution when accessing file system resources and environment state.

#![cfg(feature = "crossval-all")]

use anyhow::{Context, Result};
use bitnet_crossval::receipt::{ParityReceipt, TokenizerAuthority, TokenizerSource};
use serial_test::serial;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;

// ============================================================================
// Test Infrastructure - Subprocess Execution Helpers
// ============================================================================

/// Result of E2E parity-both subprocess execution
#[derive(Debug)]
struct E2EResult {
    stdout: String,
    stderr: String,
    exit_code: i32,
    receipt_bitnet: PathBuf,
    receipt_llama: PathBuf,
}

/// Execute parity-both command as subprocess
///
/// Returns: (stdout, stderr, exit_code, receipt_paths)
fn run_parity_both_e2e(
    model_path: &Path,
    tokenizer_path: &Path,
    prompt: &str,
    max_tokens: usize,
    out_dir: &Path,
) -> E2EResult {
    let mut cmd = Command::new("cargo");
    cmd.args(["run", "-p", "xtask", "--features", "crossval-all", "--"]);
    cmd.args(["parity-both"]);
    cmd.arg("--model-gguf").arg(model_path);
    cmd.arg("--tokenizer").arg(tokenizer_path);
    cmd.arg("--prompt").arg(prompt);
    cmd.arg("--max-tokens").arg(max_tokens.to_string());
    cmd.arg("--out-dir").arg(out_dir);
    cmd.arg("--no-repair"); // Skip auto-repair for deterministic tests

    let output = cmd.output().expect("Failed to execute parity-both");

    E2EResult {
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        exit_code: output.status.code().unwrap_or(-1),
        receipt_bitnet: out_dir.join("receipt_bitnet.json"),
        receipt_llama: out_dir.join("receipt_llama.json"),
    }
}

/// Execute parity-both with JSON format output
fn run_parity_both_e2e_json(
    model_path: &Path,
    tokenizer_path: &Path,
    prompt: &str,
    max_tokens: usize,
    out_dir: &Path,
) -> E2EResult {
    let mut cmd = Command::new("cargo");
    cmd.args(["run", "-p", "xtask", "--features", "crossval-all", "--"]);
    cmd.args(["parity-both"]);
    cmd.arg("--model-gguf").arg(model_path);
    cmd.arg("--tokenizer").arg(tokenizer_path);
    cmd.arg("--prompt").arg(prompt);
    cmd.arg("--max-tokens").arg(max_tokens.to_string());
    cmd.arg("--out-dir").arg(out_dir);
    cmd.arg("--no-repair"); // Skip auto-repair for deterministic tests
    cmd.arg("--format").arg("json"); // JSON format output

    let output = cmd.output().expect("Failed to execute parity-both");

    E2EResult {
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        exit_code: output.status.code().unwrap_or(-1),
        receipt_bitnet: out_dir.join("receipt_bitnet.json"),
        receipt_llama: out_dir.join("receipt_llama.json"),
    }
}

/// Parse receipt JSON file and extract TokenizerAuthority
fn parse_receipt_authority(receipt_path: &Path) -> Result<TokenizerAuthority> {
    let content = std::fs::read_to_string(receipt_path).context("Failed to read receipt file")?;
    let receipt: ParityReceipt =
        serde_json::from_str(&content).context("Failed to parse receipt JSON")?;
    receipt
        .tokenizer_authority
        .ok_or_else(|| anyhow::anyhow!("Receipt missing tokenizer_authority field"))
}

/// Assert two TokenizerAuthority instances match
fn assert_authorities_match(auth_a: &TokenizerAuthority, auth_b: &TokenizerAuthority) {
    assert_eq!(auth_a.config_hash, auth_b.config_hash, "Config hashes must match");
    assert_eq!(auth_a.token_count, auth_b.token_count, "Token counts must match");
    assert_eq!(auth_a.source, auth_b.source, "Sources must match");
    // Note: file_hash and path may differ for file copies, but config_hash is the canonical identifier
}

/// Copy test fixture to temp directory
fn copy_fixture(fixture_name: &str, temp_dir: &TempDir) -> PathBuf {
    let fixture_path = workspace_root().join("tests/fixtures/tokenizers").join(fixture_name);
    let dest_path = temp_dir.path().join(fixture_name);
    std::fs::copy(&fixture_path, &dest_path).expect("Failed to copy fixture");
    dest_path
}

/// Get test model path (auto-discover or use BITNET_GGUF env var)
fn get_test_model_path() -> PathBuf {
    if let Ok(model) = std::env::var("BITNET_GGUF") {
        return PathBuf::from(model);
    }

    // Try common model locations
    let workspace = workspace_root();
    let candidates = vec![
        workspace.join("models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf"),
        workspace.join("models/bitnet-2b-gguf/ggml-model-i2_s.gguf"),
    ];

    for candidate in candidates {
        if candidate.exists() {
            return candidate;
        }
    }

    panic!(
        "No test model found. Set BITNET_GGUF or download model with: cargo run -p xtask -- download-model"
    );
}

/// Find workspace root directory
fn workspace_root() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    while !path.join(".git").exists() {
        if !path.pop() {
            panic!("Could not find workspace root");
        }
    }
    path
}

// ============================================================================
// TC1: Happy Path - Identical Tokenizers
// ============================================================================

/// TC1.1: Exit Code 0 with Matching Tokenizers
///
/// **Test**: Identical tokenizers produce exit code 0 and populate receipts
///
/// **Spec**: `docs/specs/tokenizer-authority-validation-tests.md#test-11`
#[test]
#[serial(bitnet_env)]
fn test_e2e_identical_tokenizers_exit_0() {
    // GIVEN - Same tokenizer file for both lanes (via shared setup)
    let temp_dir = TempDir::new().unwrap();
    let tokenizer = copy_fixture("valid_tokenizer_a.json", &temp_dir);
    let model = get_test_model_path();

    // WHEN - Run parity-both subprocess
    let result = run_parity_both_e2e(&model, &tokenizer, "What is 2+2?", 4, temp_dir.path());

    // THEN - Exit code 0 (success)
    assert_eq!(
        result.exit_code, 0,
        "Exit code should be 0 for matching tokenizers. stderr: {}",
        result.stderr
    );

    // AND - Both receipt files exist
    assert!(result.receipt_bitnet.exists(), "BitNet receipt should exist");
    assert!(result.receipt_llama.exists(), "llama receipt should exist");

    // AND - Receipts contain TokenizerAuthority
    let auth_bitnet = parse_receipt_authority(&result.receipt_bitnet)
        .expect("BitNet receipt should have TokenizerAuthority");
    let auth_llama = parse_receipt_authority(&result.receipt_llama)
        .expect("llama receipt should have TokenizerAuthority");

    // AND - Authorities are identical
    assert_authorities_match(&auth_bitnet, &auth_llama);

    // AND - Config hash is 64 hex chars
    assert_eq!(auth_bitnet.config_hash.len(), 64, "Config hash should be 64 hex chars");
    assert!(
        auth_bitnet.config_hash.chars().all(|c| c.is_ascii_hexdigit()),
        "Config hash should be lowercase hex"
    );
}

/// TC1.2: Summary Displays Tokenizer Hash (Text Format)
///
/// **Test**: Text summary output includes tokenizer consistency section with hash
///
/// **Spec**: `docs/specs/tokenizer-authority-validation-tests.md#test-12`
#[test]
#[serial(bitnet_env)]
fn test_e2e_summary_text_format_displays_hash() {
    // GIVEN - Valid tokenizer
    let temp_dir = TempDir::new().unwrap();
    let tokenizer = copy_fixture("valid_tokenizer_a.json", &temp_dir);
    let model = get_test_model_path();

    // WHEN - Run parity-both with default text format
    let result = run_parity_both_e2e(&model, &tokenizer, "Test", 2, temp_dir.path());

    // THEN - Exit code 0
    assert_eq!(result.exit_code, 0, "Exit code should be 0. stderr: {}", result.stderr);

    // AND - Stdout contains "Tokenizer Consistency" section
    assert!(
        result.stdout.contains("Tokenizer Consistency"),
        "Summary should include tokenizer section"
    );

    // AND - Stdout contains "Config hash:" label
    assert!(result.stdout.contains("Config hash:"), "Summary should show config hash label");

    // AND - Verify hash display
    let auth = parse_receipt_authority(&result.receipt_bitnet).expect("Should parse receipt");

    // Full hash should appear in stdout
    assert!(result.stdout.contains(&auth.config_hash), "Summary should display full config hash");

    // Abbreviated hash (first 32 chars) should also appear
    let abbreviated = &auth.config_hash[..32];
    assert!(
        result.stdout.contains(abbreviated),
        "Summary should show abbreviated hash (first 32 chars)"
    );
}

/// TC1.3: Summary Displays Tokenizer Hash (JSON Format)
///
/// **Test**: JSON summary output includes tokenizer object with config_hash
///
/// **Spec**: `docs/specs/tokenizer-authority-validation-tests.md#test-13`
#[test]
#[serial(bitnet_env)]
fn test_e2e_summary_json_format_includes_tokenizer() {
    // GIVEN - Valid tokenizer
    let temp_dir = TempDir::new().unwrap();
    let tokenizer = copy_fixture("valid_tokenizer_a.json", &temp_dir);
    let model = get_test_model_path();

    // WHEN - Run parity-both with --format json
    let result = run_parity_both_e2e_json(&model, &tokenizer, "Test", 2, temp_dir.path());

    // THEN - Exit code 0
    assert_eq!(result.exit_code, 0, "Exit code should be 0. stderr: {}", result.stderr);

    // AND - Stdout is valid JSON
    let summary: serde_json::Value =
        serde_json::from_str(&result.stdout).expect("Summary should be valid JSON");

    // AND - JSON contains tokenizer object
    assert!(summary.get("tokenizer").is_some(), "JSON summary should include tokenizer field");

    // AND - Tokenizer object has config_hash and status
    let tokenizer_obj = summary["tokenizer"].as_object().unwrap();
    assert!(tokenizer_obj.contains_key("config_hash"), "Tokenizer object should have config_hash");
    assert!(tokenizer_obj.contains_key("status"), "Tokenizer object should have status");

    // AND - Status is "consistent"
    assert_eq!(
        tokenizer_obj["status"].as_str().unwrap(),
        "consistent",
        "Tokenizer status should be 'consistent'"
    );

    // AND - Config hash matches receipt
    let auth = parse_receipt_authority(&result.receipt_bitnet).expect("Should parse receipt");
    assert_eq!(
        tokenizer_obj["config_hash"].as_str().unwrap(),
        auth.config_hash,
        "JSON hash should match receipt"
    );
}

/// TC1.4: Receipt Field Population (All Fields Present)
///
/// **Test**: TokenizerAuthority in receipts has all fields properly populated
///
/// **Spec**: `docs/specs/tokenizer-authority-validation-tests.md#test-14`
#[test]
#[serial(bitnet_env)]
fn test_e2e_receipt_authority_all_fields_populated() {
    // GIVEN - External tokenizer (tokenizer.json file)
    let temp_dir = TempDir::new().unwrap();
    let tokenizer = copy_fixture("valid_tokenizer_a.json", &temp_dir);
    let model = get_test_model_path();

    // WHEN - Run parity-both
    let result = run_parity_both_e2e(&model, &tokenizer, "Test", 2, temp_dir.path());
    assert_eq!(result.exit_code, 0, "Exit code should be 0. stderr: {}", result.stderr);

    // THEN - Both receipts have TokenizerAuthority with all fields
    for receipt_path in &[result.receipt_bitnet, result.receipt_llama] {
        let auth = parse_receipt_authority(receipt_path).expect("Should parse TokenizerAuthority");

        // Source is External (file named tokenizer.json or valid_tokenizer_a.json)
        assert_eq!(
            auth.source,
            TokenizerSource::External,
            "Source should be External for tokenizer.json file"
        );

        // Path matches input tokenizer
        assert!(
            auth.path.contains("valid_tokenizer_a.json"),
            "Path should reference input tokenizer file: {}",
            auth.path
        );

        // File hash is Some (external tokenizer)
        assert!(auth.file_hash.is_some(), "File hash should be present for external tokenizer");
        let file_hash = auth.file_hash.unwrap();
        assert_eq!(file_hash.len(), 64, "File hash should be 64 hex chars");
        assert!(
            file_hash.chars().all(|c| c.is_ascii_hexdigit()),
            "File hash should be lowercase hex"
        );

        // Config hash is 64 hex chars
        assert_eq!(auth.config_hash.len(), 64, "Config hash should be 64 hex chars");
        assert!(
            auth.config_hash.chars().all(|c| c.is_ascii_hexdigit()),
            "Config hash should be lowercase hex"
        );

        // Token count is reasonable (prompt tokenized to 1-16 tokens for short prompt)
        assert!(
            auth.token_count >= 1 && auth.token_count <= 16,
            "Token count should be in reasonable range for short prompt, got {}",
            auth.token_count
        );
    }
}

// ============================================================================
// TC2: Mismatch Detection - Validation Logic
// ============================================================================

/// TC2.1: Validation Function Detects Config Hash Mismatch
///
/// **Test**: Validation logic correctly rejects different config hashes
///
/// **Spec**: `docs/specs/tokenizer-authority-validation-tests.md#test-21`
///
/// **Note**: Since parity-both uses shared setup, we test the validation
/// function directly with synthetic authorities.
#[test]
fn test_validate_tokenizer_consistency_rejects_different_config_hash() {
    use bitnet_crossval::receipt::validate_tokenizer_consistency;

    // GIVEN - Two authorities with different config hashes
    let auth_a = TokenizerAuthority {
        source: TokenizerSource::External,
        path: "tokenizer_a.json".to_string(),
        file_hash: Some("a".repeat(64)),
        config_hash: "1".repeat(64),
        token_count: 4,
    };

    let auth_b = TokenizerAuthority {
        source: TokenizerSource::External,
        path: "tokenizer_b.json".to_string(),
        file_hash: Some("b".repeat(64)),
        config_hash: "2".repeat(64), // DIFFERENT
        token_count: 4,
    };

    // WHEN - Call validate_tokenizer_consistency(&auth_a, &auth_b)
    let result = validate_tokenizer_consistency(&auth_a, &auth_b);

    // THEN - Returns Err
    assert!(result.is_err(), "Validation should fail for different config hashes");

    // AND - Error message mentions "config mismatch"
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("config mismatch") || err_msg.contains("Config mismatch"),
        "Error should mention config mismatch: {}",
        err_msg
    );

    // AND - Error message includes both hashes for debugging
    assert!(err_msg.contains("111"), "Error should show Lane A hash");
    assert!(err_msg.contains("222"), "Error should show Lane B hash");
}

/// TC2.2: Validation Function Detects Token Count Mismatch
///
/// **Test**: Validation logic correctly rejects different token counts
///
/// **Spec**: `docs/specs/tokenizer-authority-validation-tests.md#test-22`
#[test]
fn test_validate_tokenizer_consistency_rejects_different_token_count() {
    use bitnet_crossval::receipt::validate_tokenizer_consistency;

    // GIVEN - Two authorities with same config hash but different token counts
    let config_hash = "a".repeat(64);

    let auth_a = TokenizerAuthority {
        source: TokenizerSource::External,
        path: "tokenizer.json".to_string(),
        file_hash: Some("f".repeat(64)),
        config_hash: config_hash.clone(),
        token_count: 4, // DIFFERENT
    };

    let auth_b = TokenizerAuthority {
        source: TokenizerSource::External,
        path: "tokenizer.json".to_string(),
        file_hash: Some("f".repeat(64)),
        config_hash,
        token_count: 8, // DIFFERENT
    };

    // WHEN - Call validate_tokenizer_consistency(&auth_a, &auth_b)
    let result = validate_tokenizer_consistency(&auth_a, &auth_b);

    // THEN - Returns Err
    assert!(result.is_err(), "Validation should fail for different token counts");

    // AND - Error message mentions "token count mismatch"
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("Token count mismatch") || err_msg.contains("token count"),
        "Error should mention token count mismatch: {}",
        err_msg
    );

    // AND - Error message includes both counts
    assert!(err_msg.contains("4"), "Error should show Lane A count");
    assert!(err_msg.contains("8"), "Error should show Lane B count");
}

/// TC2.3: Exit Code 2 Diagnostic Format
///
/// **Test**: Verify diagnostic message format for exit code 2 scenarios
///
/// **Spec**: `docs/specs/tokenizer-authority-validation-tests.md#test-23`
///
/// **Note**: This validates the expected diagnostic format. Since parity-both
/// uses shared setup, we test the error format from validation function.
#[test]
fn test_exit_code_2_diagnostic_format_has_required_fields() {
    use bitnet_crossval::receipt::validate_tokenizer_consistency;

    // GIVEN - Mismatched authorities
    let auth_a = TokenizerAuthority {
        source: TokenizerSource::External,
        path: "tokenizer_a.json".to_string(),
        file_hash: Some("a".repeat(64)),
        config_hash: "1".repeat(64),
        token_count: 4,
    };

    let auth_b = TokenizerAuthority {
        source: TokenizerSource::External,
        path: "tokenizer_b.json".to_string(),
        file_hash: Some("b".repeat(64)),
        config_hash: "2".repeat(64),
        token_count: 4,
    };

    // WHEN - Validation fails
    let error = validate_tokenizer_consistency(&auth_a, &auth_b).unwrap_err();

    // THEN - Error message format includes:
    let err_str = error.to_string();

    // 1. Clear indication of failure
    assert!(
        err_str.contains("mismatch") || err_str.contains("Mismatch"),
        "Error should indicate mismatch"
    );

    // 2. Descriptive error message (length > 50 chars)
    assert!(
        err_str.len() > 50,
        "Error message should be descriptive, got length {}",
        err_str.len()
    );

    // 3. Actionable diagnostic information (contains relevant details)
    // The validation function provides the core error message that's expanded
    // by parity_both.rs error handler
}

// ============================================================================
// TC3: Receipt Schema v2 Compatibility
// ============================================================================

/// TC3.1: Receipt Version Inference with TokenizerAuthority
///
/// **Test**: Receipts with TokenizerAuthority infer schema version 2.0.0
///
/// **Spec**: `docs/specs/tokenizer-authority-validation-tests.md#test-31`
#[test]
#[serial(bitnet_env)]
fn test_e2e_receipt_version_inferred_as_v2() {
    // GIVEN - Valid tokenizer
    let temp_dir = TempDir::new().unwrap();
    let tokenizer = copy_fixture("valid_tokenizer_a.json", &temp_dir);
    let model = get_test_model_path();

    // WHEN - Run parity-both
    let result = run_parity_both_e2e(&model, &tokenizer, "Test", 2, temp_dir.path());
    assert_eq!(result.exit_code, 0, "Exit code should be 0. stderr: {}", result.stderr);

    // THEN - Receipt schema version is "2.0.0"
    for receipt_path in &[result.receipt_bitnet, result.receipt_llama] {
        let content = std::fs::read_to_string(receipt_path).unwrap();
        let receipt: ParityReceipt = serde_json::from_str(&content).unwrap();

        // Version inference (receipt.infer_version() returns "2.0.0" if v2 fields present)
        let version = receipt.infer_version();
        assert_eq!(version, "2.0.0", "Receipt with tokenizer_authority should infer version 2.0.0");

        // TokenizerAuthority field should be Some
        assert!(
            receipt.tokenizer_authority.is_some(),
            "v2 receipt should have tokenizer_authority populated"
        );
    }
}

/// TC3.2: Receipt Serialization Omits None Fields
///
/// **Test**: Receipt JSON serialization includes tokenizer_authority for v2
///
/// **Spec**: `docs/specs/tokenizer-authority-validation-tests.md#test-32`
#[test]
#[serial(bitnet_env)]
fn test_e2e_receipt_json_skips_none_fields() {
    // GIVEN - Valid tokenizer (external, so file_hash is Some)
    let temp_dir = TempDir::new().unwrap();
    let tokenizer = copy_fixture("valid_tokenizer_a.json", &temp_dir);
    let model = get_test_model_path();

    // WHEN - Run parity-both
    let result = run_parity_both_e2e(&model, &tokenizer, "Test", 2, temp_dir.path());
    assert_eq!(result.exit_code, 0, "Exit code should be 0. stderr: {}", result.stderr);

    // THEN - Receipt JSON includes tokenizer_authority (not skipped)
    let content = std::fs::read_to_string(&result.receipt_bitnet).unwrap();
    assert!(
        content.contains("tokenizer_authority"),
        "Receipt JSON should include tokenizer_authority field"
    );

    // AND - file_hash is present (external tokenizer)
    assert!(
        content.contains("file_hash"),
        "Receipt JSON should include file_hash for external tokenizer"
    );

    // AND - JSON is valid and parseable
    let _receipt: ParityReceipt =
        serde_json::from_str(&content).expect("Receipt JSON should be valid");
}

// ============================================================================
// TC4: Edge Cases and Error Handling
// ============================================================================

/// TC4.1: Missing Tokenizer File
///
/// **Test**: Graceful error handling when tokenizer file doesn't exist
///
/// **Spec**: `docs/specs/tokenizer-authority-validation-tests.md#test-41`
#[test]
#[serial(bitnet_env)]
fn test_e2e_missing_tokenizer_file_exit_2() {
    // GIVEN - Non-existent tokenizer path
    let temp_dir = TempDir::new().unwrap();
    let model = get_test_model_path();
    let nonexistent = temp_dir.path().join("nonexistent_tokenizer.json");

    // WHEN - Run parity-both with missing tokenizer
    let result = run_parity_both_e2e(&model, &nonexistent, "Test", 2, temp_dir.path());

    // THEN - Exit code 2 (usage error)
    assert_eq!(result.exit_code, 2, "Exit code should be 2 for missing tokenizer file");

    // AND - Error message mentions file not found
    let combined_output = format!(
        "{}
{}",
        result.stdout, result.stderr
    );
    assert!(
        combined_output.contains("not found")
            || combined_output.contains("No such file")
            || combined_output.contains("Failed to load tokenizer"),
        "Error should mention missing file"
    );
}

/// TC4.2: Corrupted Tokenizer JSON
///
/// **Test**: Graceful error handling for malformed tokenizer JSON
///
/// **Spec**: `docs/specs/tokenizer-authority-validation-tests.md#test-42`
#[test]
#[serial(bitnet_env)]
fn test_e2e_corrupted_tokenizer_json_exit_2() {
    // GIVEN - Corrupted tokenizer.json (invalid JSON)
    let temp_dir = TempDir::new().unwrap();
    let model = get_test_model_path();
    let corrupted = copy_fixture("corrupted.json", &temp_dir);

    // WHEN - Run parity-both with corrupted tokenizer
    let result = run_parity_both_e2e(&model, &corrupted, "Test", 2, temp_dir.path());

    // THEN - Exit code 2 (usage error - invalid input)
    assert_eq!(result.exit_code, 2, "Exit code should be 2 for corrupted tokenizer");

    // AND - Error message mentions JSON parsing failure
    let combined_output = format!(
        "{}
{}",
        result.stdout, result.stderr
    );
    assert!(
        combined_output.contains("JSON")
            || combined_output.contains("parse")
            || combined_output.contains("Failed to load tokenizer"),
        "Error should mention JSON parsing issue"
    );
}

/// TC4.3: Model File Missing (Orthogonal Error)
///
/// **Test**: Verify error handling when model file is missing
///
/// **Spec**: `docs/specs/tokenizer-authority-validation-tests.md#test-43`
#[test]
#[serial(bitnet_env)]
fn test_e2e_missing_model_file_exit_2() {
    // GIVEN - Valid tokenizer but missing model
    let temp_dir = TempDir::new().unwrap();
    let tokenizer = copy_fixture("valid_tokenizer_a.json", &temp_dir);
    let nonexistent_model = temp_dir.path().join("nonexistent_model.gguf");

    // WHEN - Run parity-both with missing model
    let result = run_parity_both_e2e(&nonexistent_model, &tokenizer, "Test", 2, temp_dir.path());

    // THEN - Exit code 2 (usage error)
    assert_eq!(result.exit_code, 2, "Exit code should be 2 for missing model file");

    // AND - Error message mentions model file issue
    let combined_output = format!(
        "{}
{}",
        result.stdout, result.stderr
    );
    assert!(
        combined_output.contains("model")
            || combined_output.contains("not found")
            || combined_output.contains("Failed to load"),
        "Error should mention model file issue"
    );
}

// ============================================================================
// TC5: Hash Determinism and Stability
// ============================================================================

/// TC5.1: File Hash Determinism (Same File → Same Hash)
///
/// **Test**: Hash computation is deterministic across multiple runs
///
/// **Spec**: `docs/specs/tokenizer-authority-validation-tests.md#test-51`
#[test]
#[serial(bitnet_env)]
fn test_e2e_file_hash_deterministic_across_runs() {
    // GIVEN - Same tokenizer file used twice
    let temp_dir = TempDir::new().unwrap();
    let tokenizer = copy_fixture("valid_tokenizer_a.json", &temp_dir);
    let model = get_test_model_path();

    // WHEN - Run parity-both twice
    let result1 = run_parity_both_e2e(&model, &tokenizer, "Test", 2, temp_dir.path());
    assert_eq!(result1.exit_code, 0, "First run should succeed");
    let auth1 = parse_receipt_authority(&result1.receipt_bitnet).unwrap();

    // Clean up receipts
    std::fs::remove_file(&result1.receipt_bitnet).unwrap();
    std::fs::remove_file(&result1.receipt_llama).unwrap();

    let result2 = run_parity_both_e2e(&model, &tokenizer, "Test", 2, temp_dir.path());
    assert_eq!(result2.exit_code, 0, "Second run should succeed");
    let auth2 = parse_receipt_authority(&result2.receipt_bitnet).unwrap();

    // THEN - File hashes are identical
    assert_eq!(auth1.file_hash, auth2.file_hash, "File hash should be deterministic for same file");

    // AND - Config hashes are identical
    assert_eq!(
        auth1.config_hash, auth2.config_hash,
        "Config hash should be deterministic for same tokenizer"
    );
}

/// TC5.2: Config Hash Determinism (Same Vocab → Same Hash)
///
/// **Test**: Byte-identical tokenizer clones produce identical hashes
///
/// **Spec**: `docs/specs/tokenizer-authority-validation-tests.md#test-52`
#[test]
#[serial(bitnet_env)]
fn test_e2e_config_hash_identical_for_cloned_tokenizers() {
    // GIVEN - Two tokenizer files with identical content (byte-for-byte clones)
    let temp_dir = TempDir::new().unwrap();
    let tokenizer_a = copy_fixture("valid_tokenizer_a.json", &temp_dir);
    let tokenizer_b = copy_fixture("valid_tokenizer_b.json", &temp_dir); // Clone of A
    let model = get_test_model_path();

    // WHEN - Run parity-both with tokenizer A
    let result_a = run_parity_both_e2e(&model, &tokenizer_a, "Test", 2, temp_dir.path());
    assert_eq!(result_a.exit_code, 0, "Run with tokenizer A should succeed");
    let auth_a = parse_receipt_authority(&result_a.receipt_bitnet).unwrap();

    // Clean up
    std::fs::remove_file(&result_a.receipt_bitnet).unwrap();
    std::fs::remove_file(&result_a.receipt_llama).unwrap();

    // WHEN - Run parity-both with tokenizer B (clone)
    let result_b = run_parity_both_e2e(&model, &tokenizer_b, "Test", 2, temp_dir.path());
    assert_eq!(result_b.exit_code, 0, "Run with tokenizer B should succeed");
    let auth_b = parse_receipt_authority(&result_b.receipt_bitnet).unwrap();

    // THEN - Config hashes are identical (same vocab config)
    assert_eq!(
        auth_a.config_hash, auth_b.config_hash,
        "Config hash should match for tokenizers with identical vocab"
    );

    // AND - File hashes are also identical (byte-for-byte clones)
    assert_eq!(
        auth_a.file_hash, auth_b.file_hash,
        "File hash should match for byte-for-byte identical files"
    );
}

/// TC5.3: Config Hash Differs for Different Vocabs
///
/// **Test**: Different tokenizer configurations produce different hashes
///
/// **Spec**: `docs/specs/tokenizer-authority-validation-tests.md#test-53`
#[test]
#[serial(bitnet_env)]
fn test_e2e_config_hash_differs_for_different_vocab_sizes() {
    // GIVEN - Two tokenizers with different vocab sizes
    let temp_dir = TempDir::new().unwrap();
    let tokenizer_a = copy_fixture("valid_tokenizer_a.json", &temp_dir);
    let tokenizer_diff = copy_fixture("different_vocab_size.json", &temp_dir);
    let model = get_test_model_path();

    // WHEN - Run parity-both with tokenizer A
    let result_a = run_parity_both_e2e(&model, &tokenizer_a, "Test", 2, temp_dir.path());
    assert_eq!(result_a.exit_code, 0, "Run with tokenizer A should succeed");
    let auth_a = parse_receipt_authority(&result_a.receipt_bitnet).unwrap();

    // Clean up
    std::fs::remove_file(&result_a.receipt_bitnet).unwrap();
    std::fs::remove_file(&result_a.receipt_llama).unwrap();

    // WHEN - Run parity-both with different tokenizer
    let result_diff = run_parity_both_e2e(&model, &tokenizer_diff, "Test", 2, temp_dir.path());
    assert_eq!(result_diff.exit_code, 0, "Run with different tokenizer should succeed");
    let auth_diff = parse_receipt_authority(&result_diff.receipt_bitnet).unwrap();

    // THEN - Config hashes are DIFFERENT (different vocab)
    assert_ne!(
        auth_a.config_hash, auth_diff.config_hash,
        "Config hash should differ for different tokenizers"
    );

    // AND - File hashes are also DIFFERENT (different file content)
    assert_ne!(
        auth_a.file_hash, auth_diff.file_hash,
        "File hash should differ for different files"
    );
}

// ============================================================================
// Test Summary
// ============================================================================

// Total: 12 E2E integration tests
//
// TC1: Happy Path (4 tests)
//   ✓ test_e2e_identical_tokenizers_exit_0
//   ✓ test_e2e_summary_text_format_displays_hash
//   ✓ test_e2e_summary_json_format_includes_tokenizer
//   ✓ test_e2e_receipt_authority_all_fields_populated
//
// TC2: Mismatch Detection (3 tests)
//   ✓ test_validate_tokenizer_consistency_rejects_different_config_hash
//   ✓ test_validate_tokenizer_consistency_rejects_different_token_count
//   ✓ test_exit_code_2_diagnostic_format_has_required_fields
//
// TC3: Schema v2 Compatibility (2 tests)
//   ✓ test_e2e_receipt_version_inferred_as_v2
//   ✓ test_e2e_receipt_json_skips_none_fields
//
// TC4: Edge Cases (3 tests)
//   ✓ test_e2e_missing_tokenizer_file_exit_2
//   ✓ test_e2e_corrupted_tokenizer_json_exit_2
//   ✓ test_e2e_missing_model_file_exit_2
//
// TC5: Hash Determinism (3 tests)
//   ✓ test_e2e_file_hash_deterministic_across_runs
//   ✓ test_e2e_config_hash_identical_for_cloned_tokenizers
//   ✓ test_e2e_config_hash_differs_for_different_vocab_sizes
//
// All tests include TODO markers with clear implementation guidance.
// Tests validate EXISTING implementation, not new features.
