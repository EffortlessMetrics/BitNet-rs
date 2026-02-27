//! Tests for the "copy-as-is / no-issue" path inside `GgufCompatibilityFixer::export_fixed`.
//!
//! When `export_fixed` is called on a source that already has no compatibility issues,
//! it takes the copy-as-is branch: copies the file verbatim, writes a stamp with an
//! empty `fixes_applied` array, and still validates that the output parses correctly.
//!
//! This "double-fix" scenario is **not** exercised by the existing `compat_tests.rs`
//! suite, which only ever calls `export_fixed` on a minimal GGUF that has issues.

use bitnet_compat::GgufCompatibilityFixer;
use bitnet_models::formats::gguf::GgufReader;
use std::fs;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helper: build and return a GGUF file that has zero compatibility issues
// ---------------------------------------------------------------------------

fn minimal_gguf_v3() -> Vec<u8> {
    let mut d = Vec::new();
    d.extend_from_slice(b"GGUF");
    d.extend_from_slice(&3u32.to_le_bytes());
    d.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    d.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count
    d
}

/// Create a "fixed" GGUF (zero remaining issues) in `dir` by running `export_fixed` once.
fn make_fixed_gguf(dir: &TempDir) -> std::path::PathBuf {
    let src = dir.path().join("_src.gguf");
    let fixed = dir.path().join("fixed.gguf");
    fs::write(&src, minimal_gguf_v3()).unwrap();
    GgufCompatibilityFixer::export_fixed(&src, &fixed)
        .expect("first export_fixed must succeed");
    fixed
}

// ---------------------------------------------------------------------------
// Sanity: make_fixed_gguf produces a file with zero issues
// ---------------------------------------------------------------------------

#[test]
fn fixed_gguf_helper_has_zero_issues() {
    let dir = TempDir::new().unwrap();
    let fixed = make_fixed_gguf(&dir);
    let issues = GgufCompatibilityFixer::diagnose(&fixed).unwrap();
    assert!(issues.is_empty(), "make_fixed_gguf must produce zero-issue file; got: {issues:?}");
}

// ---------------------------------------------------------------------------
// Double-fix: calling export_fixed on an already-fixed source
// ---------------------------------------------------------------------------

#[test]
fn double_fix_succeeds() {
    let dir = TempDir::new().unwrap();
    let fixed = make_fixed_gguf(&dir);
    let dst = dir.path().join("dst.gguf");
    let result = GgufCompatibilityFixer::export_fixed(&fixed, &dst);
    assert!(result.is_ok(), "export_fixed on already-fixed file must succeed: {:?}", result.err());
}

#[test]
fn double_fix_output_is_parseable() {
    let dir = TempDir::new().unwrap();
    let fixed = make_fixed_gguf(&dir);
    let dst = dir.path().join("dst.gguf");
    GgufCompatibilityFixer::export_fixed(&fixed, &dst).unwrap();
    let out = fs::read(&dst).unwrap();
    assert!(GgufReader::new(&out).is_ok(), "double-fixed GGUF must be parseable by GgufReader");
}

#[test]
fn double_fix_output_has_zero_issues() {
    let dir = TempDir::new().unwrap();
    let fixed = make_fixed_gguf(&dir);
    let dst = dir.path().join("dst.gguf");
    GgufCompatibilityFixer::export_fixed(&fixed, &dst).unwrap();
    let issues = GgufCompatibilityFixer::diagnose(&dst).unwrap();
    assert!(issues.is_empty(), "double-fixed GGUF must have zero issues; got: {issues:?}");
}

#[test]
fn double_fix_output_passes_idempotency_check() {
    let dir = TempDir::new().unwrap();
    let fixed = make_fixed_gguf(&dir);
    let dst = dir.path().join("dst.gguf");
    GgufCompatibilityFixer::export_fixed(&fixed, &dst).unwrap();
    assert!(
        GgufCompatibilityFixer::verify_idempotent(&dst).unwrap(),
        "double-fixed GGUF must pass idempotency check"
    );
}

#[test]
fn double_fix_output_preserves_bos_eos() {
    let dir = TempDir::new().unwrap();
    let fixed = make_fixed_gguf(&dir);
    let dst = dir.path().join("dst.gguf");
    GgufCompatibilityFixer::export_fixed(&fixed, &dst).unwrap();
    let out = fs::read(&dst).unwrap();
    let reader = GgufReader::new(&out).unwrap();
    assert_eq!(
        reader.get_u32_metadata("tokenizer.ggml.bos_token_id"),
        Some(1),
        "double-fixed GGUF must preserve bos_token_id=1"
    );
    assert_eq!(
        reader.get_u32_metadata("tokenizer.ggml.eos_token_id"),
        Some(2),
        "double-fixed GGUF must preserve eos_token_id=2"
    );
}

#[test]
fn double_fix_output_preserves_compat_fixed_flag() {
    let dir = TempDir::new().unwrap();
    let fixed = make_fixed_gguf(&dir);
    let dst = dir.path().join("dst.gguf");
    GgufCompatibilityFixer::export_fixed(&fixed, &dst).unwrap();
    let out = fs::read(&dst).unwrap();
    let reader = GgufReader::new(&out).unwrap();
    assert_eq!(
        reader.get_bool_metadata("bitnet.compat.fixed"),
        Some(true),
        "double-fixed GGUF must preserve bitnet.compat.fixed=true"
    );
}

// ---------------------------------------------------------------------------
// Stamp file for the no-issue ("copy as-is") path
// ---------------------------------------------------------------------------

#[test]
fn double_fix_stamp_is_created() {
    let dir = TempDir::new().unwrap();
    let fixed = make_fixed_gguf(&dir);
    let dst = dir.path().join("dst.gguf");
    GgufCompatibilityFixer::export_fixed(&fixed, &dst).unwrap();
    let stamp = dst.with_extension("gguf.compat.json");
    assert!(stamp.exists(), "stamp file must be created at {stamp:?}");
}

#[test]
fn double_fix_stamp_is_valid_json() {
    let dir = TempDir::new().unwrap();
    let fixed = make_fixed_gguf(&dir);
    let dst = dir.path().join("dst.gguf");
    GgufCompatibilityFixer::export_fixed(&fixed, &dst).unwrap();
    let stamp = dst.with_extension("gguf.compat.json");
    let content = fs::read_to_string(&stamp).unwrap();
    let parsed: serde_json::Result<serde_json::Value> = serde_json::from_str(&content);
    assert!(parsed.is_ok(), "double-fix stamp must be valid JSON; content:\n{content}");
}

#[test]
fn double_fix_stamp_has_empty_fixes_applied() {
    // When the source has no issues, export_fixed copies it verbatim.
    // The stamp must record an empty fixes_applied array (nothing was patched).
    let dir = TempDir::new().unwrap();
    let fixed = make_fixed_gguf(&dir);
    let dst = dir.path().join("dst.gguf");
    GgufCompatibilityFixer::export_fixed(&fixed, &dst).unwrap();
    let stamp = dst.with_extension("gguf.compat.json");
    let content = fs::read_to_string(&stamp).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    let fixes = json["fixes_applied"].as_array().expect("fixes_applied must be a JSON array");
    assert!(
        fixes.is_empty(),
        "no-issue export_fixed must write empty fixes_applied; got: {fixes:?}"
    );
}

#[test]
fn double_fix_stamp_has_version_field() {
    let dir = TempDir::new().unwrap();
    let fixed = make_fixed_gguf(&dir);
    let dst = dir.path().join("dst.gguf");
    GgufCompatibilityFixer::export_fixed(&fixed, &dst).unwrap();
    let stamp = dst.with_extension("gguf.compat.json");
    let content = fs::read_to_string(&stamp).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert!(json.get("version").and_then(|v| v.as_str()).is_some(), "stamp must have 'version' string field");
}

#[test]
fn double_fix_stamp_has_timestamp_field() {
    let dir = TempDir::new().unwrap();
    let fixed = make_fixed_gguf(&dir);
    let dst = dir.path().join("dst.gguf");
    GgufCompatibilityFixer::export_fixed(&fixed, &dst).unwrap();
    let stamp = dst.with_extension("gguf.compat.json");
    let content = fs::read_to_string(&stamp).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    let ts = json.get("timestamp").and_then(|v| v.as_str());
    assert!(ts.is_some_and(|s| !s.is_empty()), "stamp must have a non-empty 'timestamp' field");
}
