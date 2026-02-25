//! Snapshot tests for bitnet-compat crate.
//!
//! These tests pin the diagnostic message strings from `GgufCompatibilityFixer::diagnose()`
//! on a minimal, well-known GGUF buffer so that any accidental wording changes are caught.

use bitnet_compat::GgufCompatibilityFixer;
use insta::assert_json_snapshot;
use std::fs;
use tempfile::TempDir;

/// Build a minimal valid GGUF v3 file with zero tensors and zero metadata entries.
fn minimal_gguf_bytes() -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF"); // Magic
    data.extend_from_slice(&3u32.to_le_bytes()); // Version 3
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count = 0
    data
}

#[test]
fn diagnose_minimal_gguf_issues() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("minimal.gguf");
    fs::write(&path, minimal_gguf_bytes()).unwrap();

    let issues = GgufCompatibilityFixer::diagnose(&path).unwrap();
    // Snapshot the list of diagnostics produced for a bare GGUF with no metadata.
    // This pins the exact issue strings; any wording change will fail this test.
    assert_json_snapshot!("diagnose_minimal_gguf_issues", issues);
}

#[test]
fn diagnose_returns_vec_type() {
    // Sanity-check that diagnose() always returns a Vec<String>, never panics
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.gguf");
    fs::write(&path, minimal_gguf_bytes()).unwrap();

    let result = GgufCompatibilityFixer::diagnose(&path);
    assert!(result.is_ok(), "diagnose should not error on minimal GGUF");
    let issues = result.unwrap();
    // All entries must be non-empty strings
    for issue in &issues {
        assert!(!issue.is_empty(), "Issue strings should be non-empty");
    }
    assert_json_snapshot!("diagnose_issue_count", issues.len());
}
