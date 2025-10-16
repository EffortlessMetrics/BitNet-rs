//! Test for chat receipt copying functionality
//!
//! Validates that receipts are correctly copied from ci/inference.json
//! to timestamped files when --emit-receipt-dir is used.

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
