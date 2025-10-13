//! Integration tests for the BitNet.rs validation workflow
//!
//! Tests feature spec: validation-workflow.md#comprehensive-inspection-testing
//!
//! This test suite validates the complete inspection and validation workflow:
//! - Basic inspect command invocation with different model types
//! - LayerNorm gamma RMS validation with architecture-aware rulesets
//! - Architecture detection and automatic ruleset selection
//! - Gate modes (auto, none, policy) for validation behavior
//! - JSON output format validation
//! - Exit code verification in strict mode
//! - Error handling for missing/corrupted files

#![cfg(feature = "full-cli")]

use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;
use std::path::PathBuf;

/// Model paths for testing
const BITNET_I2S_MODEL: &str = "models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf";
const LLAMA_F16_MODEL: &str = "models/clean/clean-f16.gguf";

/// Helper to get absolute path to test model
fn model_path(relative_path: &str) -> PathBuf {
    let workspace_root = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(workspace_root).join(relative_path)
}

/// Helper to check if a model exists
fn model_exists(relative_path: &str) -> bool {
    model_path(relative_path).exists()
}

/// Helper to skip test if model doesn't exist
macro_rules! require_model {
    ($path:expr) => {
        if !model_exists($path) {
            eprintln!("Skipping test: model not found at {}", $path);
            return;
        }
    };
}

// ============================================================================
// Basic Inspect Command Tests
// ============================================================================

/// Tests feature spec: validation-workflow.md#basic-inspect-invocation
#[test]
fn test_inspect_help_displays_usage() {
    Command::cargo_bin("bitnet")
        .unwrap()
        .args(["inspect", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--ln-stats"))
        .stdout(predicate::str::contains("--gate"))
        .stdout(predicate::str::contains("--json"))
        .stdout(predicate::str::contains("--policy"));
}

/// Tests feature spec: validation-workflow.md#error-handling
#[test]
fn test_inspect_requires_model_argument() {
    Command::cargo_bin("bitnet")
        .unwrap()
        .args(["inspect", "--ln-stats"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("required arguments were not provided"));
}

/// Tests feature spec: validation-workflow.md#error-handling
#[test]
fn test_inspect_fails_on_nonexistent_file() {
    Command::cargo_bin("bitnet")
        .unwrap()
        .args(["inspect", "--ln-stats", "/nonexistent/model.gguf"])
        .assert()
        .failure()
        .stdout(predicate::str::contains("Failed to open model"));
}

/// Tests feature spec: validation-workflow.md#error-handling
#[test]
fn test_inspect_requires_inspection_mode() {
    // Without --ln-stats flag, should error
    let temp_path = model_path("models/test.gguf");
    Command::cargo_bin("bitnet")
        .unwrap()
        .args(["inspect", temp_path.to_str().unwrap()])
        .assert()
        .failure()
        .stdout(predicate::str::contains("No inspection mode specified"));
}

// ============================================================================
// BitNet I2_S Model Tests
// ============================================================================

/// Tests feature spec: validation-workflow.md#bitnet-i2s-validation
///
/// Validates that BitNet I2_S models pass inspection with bitnet-b1.58:i2_s ruleset
#[test]
fn test_inspect_bitnet_i2s_model_with_auto_gate() {
    require_model!(BITNET_I2S_MODEL);

    let model = model_path(BITNET_I2S_MODEL);

    Command::cargo_bin("bitnet")
        .unwrap()
        .args(["inspect", "--ln-stats", "--gate", "auto", model.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("model_sha256:"))
        .stdout(predicate::str::contains("ruleset: bitnet-b1.58:i2_s"));
}

/// Tests feature spec: validation-workflow.md#bitnet-i2s-validation
///
/// Validates that BitNet I2_S models auto-detect proper architecture and ruleset
#[test]
fn test_inspect_bitnet_i2s_auto_detects_architecture() {
    require_model!(BITNET_I2S_MODEL);

    let model = model_path(BITNET_I2S_MODEL);

    // Default gate is "auto"
    let output = Command::cargo_bin("bitnet")
        .unwrap()
        .args(["inspect", "--ln-stats", model.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let stdout = String::from_utf8_lossy(&output);

    // Should detect bitnet architecture and use I2_S ruleset
    assert!(
        stdout.contains("bitnet-b1.58:i2_s"),
        "Expected bitnet-b1.58:i2_s ruleset, got:\n{}",
        stdout
    );
}

/// Tests feature spec: validation-workflow.md#json-output
///
/// Validates JSON output format for BitNet I2_S model
#[test]
fn test_inspect_bitnet_i2s_json_output() {
    require_model!(BITNET_I2S_MODEL);

    let model = model_path(BITNET_I2S_MODEL);

    let output = Command::cargo_bin("bitnet")
        .unwrap()
        .args(["inspect", "--ln-stats", "--json", model.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let stdout = String::from_utf8_lossy(&output);
    let json: Value = serde_json::from_str(&stdout).expect("Output should be valid JSON");

    // Validate JSON structure
    assert!(json.get("model_sha256").is_some(), "Missing model_sha256");
    assert_eq!(json["ruleset"].as_str(), Some("bitnet-b1.58:i2_s"));

    // Validate layernorm section
    assert!(json.get("layernorm").is_some(), "Missing layernorm section");
    assert!(json["layernorm"]["total"].is_number(), "Missing layernorm total");
    assert!(json["layernorm"]["suspicious"].is_number(), "Missing layernorm suspicious");

    // Validate projection section
    assert!(json.get("projection").is_some(), "Missing projection section");
    assert!(json["projection"]["total"].is_number(), "Missing projection total");
    assert!(json["projection"]["suspicious"].is_number(), "Missing projection suspicious");

    // Validate strict_mode field
    assert!(json.get("strict_mode").is_some(), "Missing strict_mode");

    // Validate status field
    assert!(json.get("status").is_some(), "Missing status");
    let status = json["status"].as_str().unwrap();
    assert!(matches!(status, "ok" | "warning" | "failed"), "Invalid status: {}", status);

    // Validate tensors array
    assert!(json["tensors"].is_array(), "tensors should be an array");
}

// ============================================================================
// LLaMA/Generic Model Tests
// ============================================================================

/// Tests feature spec: validation-workflow.md#generic-model-validation
///
/// Validates that LLaMA F16 models use generic ruleset by default
#[test]
fn test_inspect_llama_f16_model_uses_generic_rules() {
    require_model!(LLAMA_F16_MODEL);

    let model = model_path(LLAMA_F16_MODEL);

    Command::cargo_bin("bitnet")
        .unwrap()
        .args(["inspect", "--ln-stats", model.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("model_sha256:"))
        .stdout(predicate::str::contains("ruleset: generic"));
}

/// Tests feature spec: validation-workflow.md#gate-mode-none
///
/// Validates that --gate none forces generic rules even for BitNet models
#[test]
fn test_inspect_gate_none_forces_generic_rules() {
    require_model!(BITNET_I2S_MODEL);

    let model = model_path(BITNET_I2S_MODEL);

    Command::cargo_bin("bitnet")
        .unwrap()
        .args(["inspect", "--ln-stats", "--gate", "none", model.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("ruleset: generic"));
}

// ============================================================================
// LayerNorm Validation Tests
// ============================================================================

/// Tests feature spec: validation-workflow.md#layernorm-rms-validation
///
/// Validates that LayerNorm RMS values are computed and reported
#[test]
fn test_inspect_reports_layernorm_rms_values() {
    require_model!(BITNET_I2S_MODEL);

    let model = model_path(BITNET_I2S_MODEL);

    let output = Command::cargo_bin("bitnet")
        .unwrap()
        .args(["inspect", "--ln-stats", model.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let stdout = String::from_utf8_lossy(&output);

    // Should contain RMS values
    assert!(
        stdout.contains("rms=") || stdout.contains("\"rms\":"),
        "Expected RMS values in output:\n{}",
        stdout
    );
}

/// Tests feature spec: validation-workflow.md#layernorm-rms-validation
///
/// Validates that LayerNorm tensors are properly identified and validated separately
#[test]
fn test_inspect_identifies_layernorm_tensors() {
    require_model!(BITNET_I2S_MODEL);

    let model = model_path(BITNET_I2S_MODEL);

    let output = Command::cargo_bin("bitnet")
        .unwrap()
        .args(["inspect", "--ln-stats", "--json", model.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let stdout = String::from_utf8_lossy(&output);
    let json: Value = serde_json::from_str(&stdout).expect("Output should be valid JSON");

    // Should have layernorm tensors
    let ln_total = json["layernorm"]["total"].as_u64().unwrap();
    assert!(ln_total > 0, "Expected LayerNorm tensors to be detected");

    // Validate tensor entries have correct kind
    let tensors = json["tensors"].as_array().unwrap();
    let ln_tensors: Vec<_> =
        tensors.iter().filter(|t| t["kind"].as_str() == Some("layernorm")).collect();

    assert!(!ln_tensors.is_empty(), "Expected at least one LayerNorm tensor");

    // Each LayerNorm tensor should have name, rms, and status
    for tensor in ln_tensors {
        assert!(tensor.get("name").is_some(), "Missing tensor name");
        assert!(tensor.get("rms").is_some(), "Missing RMS value");
        assert!(tensor.get("status").is_some(), "Missing status");
    }
}

/// Tests feature spec: validation-workflow.md#projection-weight-validation
///
/// Validates that projection weights are identified and validated separately
#[test]
fn test_inspect_validates_projection_weights_separately() {
    require_model!(LLAMA_F16_MODEL);

    let model = model_path(LLAMA_F16_MODEL);

    let output = Command::cargo_bin("bitnet")
        .unwrap()
        .args(["inspect", "--ln-stats", "--json", model.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let stdout = String::from_utf8_lossy(&output);
    let json: Value = serde_json::from_str(&stdout).expect("Output should be valid JSON");

    // Should have projection section
    assert!(json.get("projection").is_some(), "Missing projection section");

    // Projection tensors might be 0 if model only has quantized projections
    let proj_total = json["projection"]["total"].as_u64().unwrap();

    // If we have projection tensors, validate their structure
    if proj_total > 0 {
        let tensors = json["tensors"].as_array().unwrap();
        let proj_tensors: Vec<_> =
            tensors.iter().filter(|t| t["kind"].as_str() == Some("projection")).collect();

        assert!(!proj_tensors.is_empty(), "Expected projection tensors");

        for tensor in proj_tensors {
            assert!(tensor.get("name").is_some(), "Missing tensor name");
            assert!(tensor.get("rms").is_some(), "Missing RMS value");
            assert!(tensor.get("status").is_some(), "Missing status");
        }
    }
}

/// Tests feature spec: validation-workflow.md#quantized-tensor-handling
///
/// Validates that quantized projection weights are skipped (not validated)
#[test]
fn test_inspect_skips_quantized_projection_weights() {
    require_model!(BITNET_I2S_MODEL);

    let model = model_path(BITNET_I2S_MODEL);

    let output = Command::cargo_bin("bitnet")
        .unwrap()
        .args(["inspect", "--ln-stats", "--json", model.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let stdout = String::from_utf8_lossy(&output);
    let json: Value = serde_json::from_str(&stdout).expect("Output should be valid JSON");

    // For I2_S quantized models, projection weights should be quantized
    // and thus skipped from RMS validation
    let proj_total = json["projection"]["total"].as_u64().unwrap();

    // I2_S models typically have all projection weights quantized,
    // so we expect proj_total to be 0 or very small
    assert!(
        proj_total < 10,
        "Expected quantized projections to be skipped, but got {} float projections",
        proj_total
    );
}

// ============================================================================
// Exit Code Tests
// ============================================================================

/// Tests feature spec: validation-workflow.md#exit-codes
///
/// Validates that non-strict mode returns success (exit 0) even with suspicious weights
#[test]
fn test_inspect_non_strict_mode_warns_but_succeeds() {
    // This test would ideally use a model with known suspicious weights
    // For now, we verify that non-strict mode (default) doesn't exit with error code 8
    require_model!(BITNET_I2S_MODEL);

    let model = model_path(BITNET_I2S_MODEL);

    // Without BITNET_STRICT_MODE, should succeed even if weights are suspicious
    Command::cargo_bin("bitnet")
        .unwrap()
        .env_remove("BITNET_STRICT_MODE")
        .args(["inspect", "--ln-stats", model.to_str().unwrap()])
        .assert()
        .success(); // Exit code 0
}

/// Tests feature spec: validation-workflow.md#exit-codes
///
/// Validates that strict mode exits with code 8 on suspicious weights
///
/// Note: This test is aspirational and will pass if the model has no suspicious weights.
/// A full test would require a fixture with known bad weights.
#[test]
fn test_inspect_strict_mode_behavior() {
    require_model!(BITNET_I2S_MODEL);

    let model = model_path(BITNET_I2S_MODEL);

    let output = Command::cargo_bin("bitnet")
        .unwrap()
        .env("BITNET_STRICT_MODE", "1")
        .args(["inspect", "--ln-stats", "--json", model.to_str().unwrap()])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: Value = serde_json::from_str(&stdout).expect("Output should be valid JSON");

    // Check if there are suspicious weights
    let ln_bad = json["layernorm"]["suspicious"].as_u64().unwrap();
    let proj_bad = json["projection"]["suspicious"].as_u64().unwrap();
    let total_bad = ln_bad + proj_bad;

    if total_bad > 0 {
        // In strict mode with bad weights, should exit with code 8
        assert_eq!(
            output.status.code(),
            Some(8),
            "Expected exit code 8 in strict mode with suspicious weights"
        );
    } else {
        // No suspicious weights, should succeed
        assert!(output.status.success(), "Expected success when no suspicious weights found");
    }
}

// ============================================================================
// Architecture Detection Tests
// ============================================================================

/// Tests feature spec: validation-workflow.md#architecture-detection
///
/// Validates that BitNet b1.58 models are auto-detected correctly
#[test]
fn test_inspect_auto_detects_bitnet_architecture() {
    require_model!(BITNET_I2S_MODEL);

    let model = model_path(BITNET_I2S_MODEL);

    let output = Command::cargo_bin("bitnet")
        .unwrap()
        .args(["inspect", "--ln-stats", "--json", model.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let stdout = String::from_utf8_lossy(&output);
    let json: Value = serde_json::from_str(&stdout).expect("Output should be valid JSON");

    let ruleset = json["ruleset"].as_str().unwrap();

    // Should detect bitnet architecture and select appropriate ruleset
    assert!(
        ruleset.contains("bitnet"),
        "Expected bitnet ruleset for BitNet model, got: {}",
        ruleset
    );
}

/// Tests feature spec: validation-workflow.md#file-type-detection
///
/// Validates that F16 vs I2_S file types select correct rulesets
#[test]
fn test_inspect_distinguishes_f16_vs_i2s_rulesets() {
    require_model!(BITNET_I2S_MODEL);

    let model = model_path(BITNET_I2S_MODEL);

    let output = Command::cargo_bin("bitnet")
        .unwrap()
        .args(["inspect", "--ln-stats", "--json", model.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let stdout = String::from_utf8_lossy(&output);
    let json: Value = serde_json::from_str(&stdout).expect("Output should be valid JSON");

    let ruleset = json["ruleset"].as_str().unwrap();

    // I2_S model should use i2_s ruleset (file_type != 1)
    assert_eq!(ruleset, "bitnet-b1.58:i2_s", "Expected i2_s ruleset for I2_S quantized model");

    // Note: A corresponding test for F16 models would check for "bitnet-b1.58:f16"
    // but requires an F16 BitNet model fixture
}

// ============================================================================
// JSON Output Format Tests
// ============================================================================

/// Tests feature spec: validation-workflow.md#json-output-schema
///
/// Validates complete JSON output schema
#[test]
fn test_inspect_json_output_complete_schema() {
    require_model!(BITNET_I2S_MODEL);

    let model = model_path(BITNET_I2S_MODEL);

    let output = Command::cargo_bin("bitnet")
        .unwrap()
        .args(["inspect", "--ln-stats", "--json", model.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let stdout = String::from_utf8_lossy(&output);
    let json: Value = serde_json::from_str(&stdout).expect("Output should be valid JSON");

    // Validate all required top-level fields
    assert!(json.get("model_sha256").is_some(), "Missing model_sha256");
    assert!(json.get("ruleset").is_some(), "Missing ruleset");
    assert!(json.get("layernorm").is_some(), "Missing layernorm");
    assert!(json.get("projection").is_some(), "Missing projection");
    assert!(json.get("strict_mode").is_some(), "Missing strict_mode");
    assert!(json.get("tensors").is_some(), "Missing tensors");
    assert!(json.get("status").is_some(), "Missing status");

    // Validate model_sha256 format (64 hex chars)
    let sha256 = json["model_sha256"].as_str().unwrap();
    assert_eq!(sha256.len(), 64, "SHA256 should be 64 hex characters");
    assert!(sha256.chars().all(|c| c.is_ascii_hexdigit()), "SHA256 should only contain hex digits");

    // Validate layernorm structure
    let ln = &json["layernorm"];
    assert!(ln.get("total").is_some(), "Missing layernorm.total");
    assert!(ln.get("suspicious").is_some(), "Missing layernorm.suspicious");
    assert!(ln["total"].is_number(), "layernorm.total should be number");
    assert!(ln["suspicious"].is_number(), "layernorm.suspicious should be number");

    // Validate projection structure
    let proj = &json["projection"];
    assert!(proj.get("total").is_some(), "Missing projection.total");
    assert!(proj.get("suspicious").is_some(), "Missing projection.suspicious");
    assert!(proj["total"].is_number(), "projection.total should be number");
    assert!(proj["suspicious"].is_number(), "projection.suspicious should be number");

    // Validate strict_mode is boolean
    assert!(json["strict_mode"].is_boolean(), "strict_mode should be boolean");

    // Validate tensors is array
    assert!(json["tensors"].is_array(), "tensors should be array");
}

/// Tests feature spec: validation-workflow.md#json-tensor-entries
///
/// Validates individual tensor entry schema in JSON output
#[test]
fn test_inspect_json_tensor_entry_schema() {
    require_model!(BITNET_I2S_MODEL);

    let model = model_path(BITNET_I2S_MODEL);

    let output = Command::cargo_bin("bitnet")
        .unwrap()
        .args(["inspect", "--ln-stats", "--json", model.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let stdout = String::from_utf8_lossy(&output);
    let json: Value = serde_json::from_str(&stdout).expect("Output should be valid JSON");

    let tensors = json["tensors"].as_array().unwrap();
    assert!(!tensors.is_empty(), "Expected at least one tensor");

    // Validate first tensor entry
    let tensor = &tensors[0];

    assert!(tensor.get("name").is_some(), "Missing tensor.name");
    assert!(tensor["name"].is_string(), "tensor.name should be string");

    assert!(tensor.get("kind").is_some(), "Missing tensor.kind");
    let kind = tensor["kind"].as_str().unwrap();
    assert!(
        kind == "layernorm" || kind == "projection",
        "tensor.kind should be 'layernorm' or 'projection', got: {}",
        kind
    );

    assert!(tensor.get("rms").is_some(), "Missing tensor.rms");
    assert!(tensor["rms"].is_string(), "tensor.rms should be string (formatted float)");

    assert!(tensor.get("status").is_some(), "Missing tensor.status");
    let status = tensor["status"].as_str().unwrap();
    assert!(
        status == "ok" || status == "suspicious",
        "tensor.status should be 'ok' or 'suspicious', got: {}",
        status
    );
}

// ============================================================================
// Policy Mode Tests (Advanced)
// ============================================================================

/// Tests feature spec: validation-workflow.md#policy-mode
///
/// Validates that --gate policy requires --policy argument
#[test]
fn test_inspect_gate_policy_requires_policy_file() {
    require_model!(BITNET_I2S_MODEL);

    let model = model_path(BITNET_I2S_MODEL);

    Command::cargo_bin("bitnet")
        .unwrap()
        .args(["inspect", "--ln-stats", "--gate", "policy", model.to_str().unwrap()])
        .assert()
        .failure()
        .stdout(predicate::str::contains("--policy required"));
}

/// Tests feature spec: validation-workflow.md#policy-mode
///
/// Validates that --gate policy fails on non-existent policy file
#[test]
fn test_inspect_gate_policy_fails_on_missing_file() {
    require_model!(BITNET_I2S_MODEL);

    let model = model_path(BITNET_I2S_MODEL);

    Command::cargo_bin("bitnet")
        .unwrap()
        .args([
            "inspect",
            "--ln-stats",
            "--gate",
            "policy",
            "--policy",
            "/nonexistent/policy.yml",
            model.to_str().unwrap(),
        ])
        .assert()
        .failure();
}

// ============================================================================
// Text Output Format Tests
// ============================================================================

/// Tests feature spec: validation-workflow.md#text-output-format
///
/// Validates human-readable text output format
#[test]
fn test_inspect_text_output_format() {
    require_model!(BITNET_I2S_MODEL);

    let model = model_path(BITNET_I2S_MODEL);

    let output = Command::cargo_bin("bitnet")
        .unwrap()
        .args(["inspect", "--ln-stats", model.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let stdout = String::from_utf8_lossy(&output);

    // Should contain key components
    assert!(stdout.contains("model_sha256:"), "Missing model_sha256");
    assert!(stdout.contains("ruleset:"), "Missing ruleset");
    assert!(stdout.contains("rms="), "Missing RMS values");

    // Should contain status indicators (emoji or text)
    let has_status = stdout.contains("✅")
        || stdout.contains("❌")
        || stdout.contains("ok")
        || stdout.contains("suspicious");
    assert!(has_status, "Missing status indicators");
}

/// Tests feature spec: validation-workflow.md#text-output-format
///
/// Validates that text output includes gate pass/fail summary
#[test]
fn test_inspect_text_output_includes_gate_summary() {
    require_model!(BITNET_I2S_MODEL);

    let model = model_path(BITNET_I2S_MODEL);

    let output = Command::cargo_bin("bitnet")
        .unwrap()
        .args(["inspect", "--ln-stats", model.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let stdout = String::from_utf8_lossy(&output);

    // Should contain gate pass/warning/fail message
    let has_gate_summary = stdout.contains("gate passed")
        || stdout.contains("gate failed")
        || stdout.contains("WARNING: suspicious");

    assert!(has_gate_summary, "Expected gate summary in output:\n{}", stdout);
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

/// Tests feature spec: validation-workflow.md#error-handling
///
/// Validates handling of corrupted GGUF files
#[test]
fn test_inspect_handles_corrupted_gguf() {
    let temp_dir = std::env::temp_dir();
    let corrupt_file = temp_dir.join("corrupt.gguf");

    // Create a file with invalid GGUF magic
    std::fs::write(&corrupt_file, b"INVALID_MAGIC").unwrap();

    Command::cargo_bin("bitnet")
        .unwrap()
        .args(["inspect", "--ln-stats", corrupt_file.to_str().unwrap()])
        .assert()
        .failure();

    // Cleanup
    std::fs::remove_file(corrupt_file).ok();
}

/// Tests feature spec: validation-workflow.md#error-handling
///
/// Validates handling of empty files
#[test]
fn test_inspect_handles_empty_file() {
    let temp_dir = std::env::temp_dir();
    let empty_file = temp_dir.join("empty.gguf");

    std::fs::write(&empty_file, b"").unwrap();

    Command::cargo_bin("bitnet")
        .unwrap()
        .args(["inspect", "--ln-stats", empty_file.to_str().unwrap()])
        .assert()
        .failure();

    // Cleanup
    std::fs::remove_file(empty_file).ok();
}

// ============================================================================
// Integration with Environment Variables
// ============================================================================

/// Tests feature spec: validation-workflow.md#strict-mode-env
///
/// Validates that BITNET_STRICT_MODE=1 enables strict validation
#[test]
fn test_inspect_respects_strict_mode_env_var() {
    require_model!(BITNET_I2S_MODEL);

    let model = model_path(BITNET_I2S_MODEL);

    let output = Command::cargo_bin("bitnet")
        .unwrap()
        .env("BITNET_STRICT_MODE", "1")
        .args(["inspect", "--ln-stats", "--json", model.to_str().unwrap()])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: Value = serde_json::from_str(&stdout).expect("Output should be valid JSON");

    // Should report strict_mode as true
    assert_eq!(
        json["strict_mode"].as_bool(),
        Some(true),
        "Expected strict_mode=true when BITNET_STRICT_MODE=1"
    );
}

/// Tests feature spec: validation-workflow.md#strict-mode-env
///
/// Validates various BITNET_STRICT_MODE value formats
#[test]
fn test_inspect_strict_mode_value_formats() {
    require_model!(BITNET_I2S_MODEL);

    let model = model_path(BITNET_I2S_MODEL);

    for value in &["1", "true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON"] {
        let output = Command::cargo_bin("bitnet")
            .unwrap()
            .env("BITNET_STRICT_MODE", value)
            .args(["inspect", "--ln-stats", "--json", model.to_str().unwrap()])
            .output()
            .expect("Failed to execute command");

        let stdout = String::from_utf8_lossy(&output.stdout);
        let json: Value = serde_json::from_str(&stdout).expect("Output should be valid JSON");

        assert_eq!(
            json["strict_mode"].as_bool(),
            Some(true),
            "Expected strict_mode=true for BITNET_STRICT_MODE={}",
            value
        );
    }
}
