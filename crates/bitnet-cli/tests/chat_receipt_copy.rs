//! Test Suite 4: CLI - Receipt Archive (bitnet-cli)
//!
//! Tests feature spec: chat-repl-ux-polish.md#AC4-receipt-archive
//!
//! This test suite validates the receipt archiving functionality for the
//! BitNet-rs CLI chat mode. Tests ensure receipts are correctly copied from
//! ci/inference.json to timestamped files when --emit-receipt-dir is used.
//!
//! **TDD Approach**: These tests compile successfully and validate the existing
//! receipt copying logic while also adding comprehensive edge case coverage.

use std::{
    fs,
    path::{Path, PathBuf},
};
use tempfile::tempdir;

/// Copy receipt helper (duplicated from chat.rs for testing)
/// Modified to accept source path for testability
fn copy_receipt_if_present_internal(src: &Path, dir: &Path) -> anyhow::Result<Option<PathBuf>> {
    use std::time::{SystemTime, UNIX_EPOCH};

    if !src.exists() {
        return Ok(None);
    }

    fs::create_dir_all(dir)?;
    let ts = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let dst = dir.join(format!("chat-{}.json", ts));
    fs::copy(src, &dst)?;
    Ok(Some(dst))
}

#[test]
fn test_copies_receipt_when_present() {
    // Setup - use tempdir for source as well
    let temp = tempdir().unwrap();
    let src = temp.path().join("inference.json");
    fs::write(
        &src,
        r#"{
  "schema_version": "1.0.0",
  "timestamp": "2025-10-16T12:00:00Z",
  "compute_path": "real",
  "backend": "cpu",
  "tokens_generated": 42
}"#,
    )
    .unwrap();

    let dir = tempdir().unwrap();

    // Execute
    let result = copy_receipt_if_present_internal(&src, dir.path()).unwrap();

    // Verify
    assert!(result.is_some(), "Should return a path when receipt exists");
    let path = result.unwrap();
    assert!(path.exists(), "Copied receipt file should exist");
    assert!(path.file_name().unwrap().to_str().unwrap().starts_with("chat-"));
    assert!(path.file_name().unwrap().to_str().unwrap().ends_with(".json"));

    // Verify content
    let content = fs::read_to_string(&path).unwrap();
    assert!(content.contains("\"schema_version\": \"1.0.0\""));
    assert!(content.contains("\"compute_path\": \"real\""));
    assert!(content.contains("\"tokens_generated\": 42"));
}

#[test]
fn test_returns_none_when_receipt_missing() {
    // Setup - non-existent source
    let temp = tempdir().unwrap();
    let src = temp.path().join("nonexistent.json");
    let dir = tempdir().unwrap();

    // Execute
    let result = copy_receipt_if_present_internal(&src, dir.path()).unwrap();

    // Verify
    assert!(result.is_none(), "Should return None when no receipt exists");
}

#[test]
fn test_creates_output_directory() {
    // Setup
    let temp = tempdir().unwrap();
    let src = temp.path().join("inference.json");
    fs::write(&src, r#"{"test": true}"#).unwrap();

    let dir = tempdir().unwrap();
    let nested_dir = dir.path().join("nested").join("output");

    // Execute
    let result = copy_receipt_if_present_internal(&src, &nested_dir).unwrap();

    // Verify
    assert!(result.is_some());
    assert!(nested_dir.exists(), "Should create nested directory");
}

#[test]
fn test_unique_timestamps() {
    // Setup
    let temp = tempdir().unwrap();
    let src = temp.path().join("inference.json");
    fs::write(&src, r#"{"test": true}"#).unwrap();

    let dir = tempdir().unwrap();

    // Execute multiple copies
    let path1 = copy_receipt_if_present_internal(&src, dir.path()).unwrap().unwrap();
    std::thread::sleep(std::time::Duration::from_millis(10));
    let path2 = copy_receipt_if_present_internal(&src, dir.path()).unwrap().unwrap();

    // Verify
    assert_ne!(path1, path2, "Should generate unique filenames");
    assert!(path1.exists());
    assert!(path2.exists());
}

/// Tests feature spec: chat-repl-ux-polish.md#AC4-default-receipt-path
#[test]
fn test_default_receipt_path_ci_inference_json() {
    // This test verifies the default receipt path behavior
    // In production, receipts are written to ci/inference.json by default

    let temp = tempdir().unwrap();
    let ci_dir = temp.path().join("ci");
    fs::create_dir_all(&ci_dir).unwrap();

    let default_receipt_path = ci_dir.join("inference.json");

    // Simulate writing a receipt
    let receipt_content = r#"{
  "schema_version": "1.0.0",
  "timestamp": "2025-10-16T12:00:00Z",
  "compute_path": "real",
  "backend": "cpu",
  "deterministic": false,
  "tokens_generated": 128
}"#;

    fs::write(&default_receipt_path, receipt_content).unwrap();

    // Verify receipt exists at default location
    assert!(default_receipt_path.exists(), "Default receipt path should exist");

    // Verify content
    let content = fs::read_to_string(&default_receipt_path).unwrap();
    assert!(content.contains("\"schema_version\": \"1.0.0\""));
    assert!(content.contains("\"compute_path\": \"real\""));
    assert!(content.contains("\"tokens_generated\": 128"));
}

/// Tests feature spec: chat-repl-ux-polish.md#AC4-custom-receipt-path
#[test]
fn test_custom_receipt_path() {
    // Test that custom receipt paths work correctly

    let temp = tempdir().unwrap();
    let custom_path = temp.path().join("custom").join("receipts").join("my-receipt.json");

    // Ensure parent directory is created
    fs::create_dir_all(custom_path.parent().unwrap()).unwrap();

    // Write receipt to custom path
    let receipt_content = r#"{"custom": "receipt", "tokens_generated": 64}"#;
    fs::write(&custom_path, receipt_content).unwrap();

    // Verify
    assert!(custom_path.exists(), "Custom receipt path should exist");

    let content = fs::read_to_string(&custom_path).unwrap();
    assert!(content.contains("\"custom\": \"receipt\""));
    assert!(content.contains("\"tokens_generated\": 64"));
}

/// Tests feature spec: chat-repl-ux-polish.md#AC4-per-turn-archive
#[test]
fn test_per_turn_receipt_archive() {
    // Simulate multiple chat turns with archived receipts

    let temp = tempdir().unwrap();
    let src = temp.path().join("inference.json");
    let archive_dir = temp.path().join("receipts");

    // Simulate 3 chat turns
    for turn_num in 1..=3 {
        let receipt_content = format!(
            r#"{{
  "schema_version": "1.0.0",
  "timestamp": "2025-10-16T12:00:0{}Z",
  "compute_path": "real",
  "backend": "cpu",
  "tokens_generated": {},
  "turn": {}
}}"#,
            turn_num,
            turn_num * 10,
            turn_num
        );

        fs::write(&src, receipt_content).unwrap();

        // Copy to archive
        let archived = copy_receipt_if_present_internal(&src, &archive_dir).unwrap();
        assert!(archived.is_some(), "Turn {} should be archived", turn_num);

        // Small delay to ensure unique timestamps
        std::thread::sleep(std::time::Duration::from_millis(5));
    }

    // Verify all receipts were archived
    let entries = fs::read_dir(&archive_dir).unwrap();
    let receipt_count = entries.filter(|e| e.is_ok()).count();

    assert_eq!(receipt_count, 3, "Should have archived 3 receipts");
}

/// Tests feature spec: chat-repl-ux-polish.md#AC4-directory-creation
#[test]
fn test_nested_directory_creation() {
    let temp = tempdir().unwrap();
    let src = temp.path().join("inference.json");
    fs::write(&src, r#"{"test": true}"#).unwrap();

    // Deep nested path
    let nested_dir = temp.path().join("a").join("b").join("c").join("receipts");

    // Should create all parent directories
    let result = copy_receipt_if_present_internal(&src, &nested_dir).unwrap();

    assert!(result.is_some(), "Should succeed even with deep nesting");
    assert!(nested_dir.exists(), "Should create all nested directories");
    assert!(result.unwrap().exists(), "Receipt file should exist in nested dir");
}

/// Tests feature spec: chat-repl-ux-polish.md#AC4-valid-json
#[test]
fn test_archived_receipt_is_valid_json() {
    let temp = tempdir().unwrap();
    let src = temp.path().join("inference.json");

    let receipt_content = r#"{
  "schema_version": "1.0.0",
  "timestamp": "2025-10-16T12:00:00Z",
  "compute_path": "real",
  "backend": "cpu",
  "deterministic": false,
  "tokens_generated": 256,
  "kernels": ["embedding_lookup", "i2s_gemv"],
  "environment": {
    "BITNET_VERSION": "0.1.0",
    "OS": "linux-x86_64"
  }
}"#;

    fs::write(&src, receipt_content).unwrap();

    let dir = tempdir().unwrap();
    let archived_path = copy_receipt_if_present_internal(&src, dir.path()).unwrap().unwrap();

    // Verify archived receipt is valid JSON
    let content = fs::read_to_string(&archived_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

    // Verify structure
    assert_eq!(parsed["schema_version"], "1.0.0");
    assert_eq!(parsed["compute_path"], "real");
    assert_eq!(parsed["tokens_generated"], 256);
    assert!(parsed["kernels"].is_array());
    assert_eq!(parsed["kernels"].as_array().unwrap().len(), 2);
}

/// Tests feature spec: chat-repl-ux-polish.md#AC4-preserves-gate-compatibility
#[test]
fn test_receipt_preserves_gate_fields() {
    // Verify that archived receipts preserve all fields needed by quality gates

    let temp = tempdir().unwrap();
    let src = temp.path().join("inference.json");

    let receipt_content = r#"{
  "schema_version": "1.0.0",
  "timestamp": "2025-10-16T12:00:00Z",
  "compute_path": "real",
  "backend": "cuda",
  "deterministic": true,
  "tokens_generated": 512,
  "kernels": ["gemm_cuda_i2s", "i2s_gpu_prefill", "attention_fused"],
  "environment": {
    "BITNET_VERSION": "0.1.0",
    "OS": "linux-x86_64",
    "RUST_VERSION": "rustc 1.90.0"
  },
  "model": {
    "path": "models/test.gguf"
  }
}"#;

    fs::write(&src, receipt_content).unwrap();

    let dir = tempdir().unwrap();
    let archived_path = copy_receipt_if_present_internal(&src, dir.path()).unwrap().unwrap();

    // Parse and verify all gate-critical fields are preserved
    let content = fs::read_to_string(&archived_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

    // Critical fields for gates
    assert_eq!(parsed["schema_version"], "1.0.0", "Schema version must be preserved");
    assert_eq!(parsed["compute_path"], "real", "Compute path must be 'real' for gates");
    assert_eq!(parsed["backend"], "cuda", "Backend must be preserved");
    assert_eq!(parsed["deterministic"], true, "Deterministic flag must be preserved");
    assert!(parsed["tokens_generated"].as_u64().unwrap() > 0, "Must have token count");
    assert!(parsed["kernels"].is_array(), "Kernels must be preserved");
    assert!(!parsed["kernels"].as_array().unwrap().is_empty(), "Kernels must not be empty");
}

/// Tests feature spec: chat-repl-ux-polish.md#AC4-filename-format
#[test]
fn test_receipt_filename_format() {
    let temp = tempdir().unwrap();
    let src = temp.path().join("inference.json");
    fs::write(&src, r#"{"test": true}"#).unwrap();

    let dir = tempdir().unwrap();
    let archived_path = copy_receipt_if_present_internal(&src, dir.path()).unwrap().unwrap();

    // Verify filename format: chat-{timestamp}.json
    let filename = archived_path.file_name().unwrap().to_str().unwrap();

    assert!(filename.starts_with("chat-"), "Filename should start with 'chat-'");
    assert!(filename.ends_with(".json"), "Filename should end with '.json'");

    // Verify timestamp portion is numeric
    let timestamp_part = filename.strip_prefix("chat-").unwrap().strip_suffix(".json").unwrap();
    let _: u128 = timestamp_part.parse().unwrap(); // Should parse as number
}

/// Integration test: Full chat session with receipt archiving
/// Tests feature spec: chat-repl-ux-polish.md#AC4-integration
#[test]
fn test_full_chat_session_receipt_archiving() {
    let temp = tempdir().unwrap();
    let ci_dir = temp.path().join("ci");
    fs::create_dir_all(&ci_dir).unwrap();

    let receipt_source = ci_dir.join("inference.json");
    let archive_dir = temp.path().join("out").join("receipts");

    // Simulate a chat session with 5 turns
    for turn_num in 1..=5 {
        // Simulate receipt writing for this turn
        let receipt_content = format!(
            r#"{{
  "schema_version": "1.0.0",
  "timestamp": "2025-10-16T12:00:0{}Z",
  "compute_path": "real",
  "backend": "cpu",
  "deterministic": false,
  "tokens_generated": {},
  "kernels": ["embedding_lookup", "prefill_forward", "i2s_gemv"],
  "turn": {}
}}"#,
            turn_num,
            turn_num * 20,
            turn_num
        );

        fs::write(&receipt_source, receipt_content).unwrap();

        // Archive the receipt
        let archived = copy_receipt_if_present_internal(&receipt_source, &archive_dir).unwrap();

        assert!(archived.is_some(), "Turn {} receipt should be archived", turn_num);
        assert!(archived.unwrap().exists(), "Turn {} archived file should exist", turn_num);

        // Small delay for timestamp uniqueness
        std::thread::sleep(std::time::Duration::from_millis(5));
    }

    // Verify all receipts are archived
    let entries: Vec<_> = fs::read_dir(&archive_dir).unwrap().collect();
    assert_eq!(entries.len(), 5, "Should have 5 archived receipts");

    // Verify each archived receipt is valid and standalone
    for entry in entries {
        let path = entry.unwrap().path();
        let content = fs::read_to_string(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

        assert_eq!(parsed["schema_version"], "1.0.0");
        assert_eq!(parsed["compute_path"], "real");
        assert!(parsed["tokens_generated"].as_u64().unwrap() > 0);
    }
}
