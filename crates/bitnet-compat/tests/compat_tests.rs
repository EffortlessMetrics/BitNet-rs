//! Comprehensive tests for `bitnet-compat` crate.
//!
//! Exercises `GgufCompatibilityFixer`: diagnose, export_fixed, verify_idempotent,
//! print_report, and the companion stamp-file contract.

use bitnet_compat::GgufCompatibilityFixer;
use bitnet_models::formats::gguf::GgufReader;
use std::fs;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Minimal syntactically-valid GGUF v3 blob: magic + version + two zero counts.
fn minimal_gguf_v3() -> Vec<u8> {
    let mut data = Vec::with_capacity(24);
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count
    data
}

fn write_temp(dir: &TempDir, name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path = dir.path().join(name);
    fs::write(&path, bytes).expect("write temp file");
    path
}

// ---------------------------------------------------------------------------
// diagnose() — error handling
// ---------------------------------------------------------------------------

#[test]
fn diagnose_nonexistent_path_returns_err() {
    let result = GgufCompatibilityFixer::diagnose("/nonexistent/no/such/file.gguf");
    assert!(result.is_err(), "diagnose on missing path must return Err");
}

#[test]
fn diagnose_empty_path_returns_err() {
    let result = GgufCompatibilityFixer::diagnose("");
    assert!(result.is_err(), "diagnose on empty path must return Err");
}

#[test]
fn diagnose_invalid_magic_returns_err() {
    let dir = TempDir::new().unwrap();
    // 24 bytes that do NOT start with "GGUF"
    let bad: Vec<u8> = b"NOTG-not-gguf-magic-12345678".to_vec();
    let path = write_temp(&dir, "bad_magic.gguf", &bad);
    let result = GgufCompatibilityFixer::diagnose(&path);
    assert!(result.is_err(), "diagnose on non-GGUF magic must return Err");
}

#[test]
fn diagnose_truncated_file_returns_err() {
    let dir = TempDir::new().unwrap();
    // Only 3 bytes — too short for any valid GGUF
    let path = write_temp(&dir, "trunc.gguf", b"GGU");
    let result = GgufCompatibilityFixer::diagnose(&path);
    assert!(result.is_err(), "diagnose on truncated file must return Err");
}

// ---------------------------------------------------------------------------
// diagnose() — Ok path on minimal GGUF
// ---------------------------------------------------------------------------

#[test]
fn diagnose_minimal_gguf_returns_ok() {
    let dir = TempDir::new().unwrap();
    let path = write_temp(&dir, "minimal.gguf", &minimal_gguf_v3());
    let result = GgufCompatibilityFixer::diagnose(&path);
    assert!(result.is_ok(), "diagnose on minimal GGUF must succeed: {:?}", result.err());
}

#[test]
fn diagnose_minimal_gguf_reports_missing_bos() {
    let dir = TempDir::new().unwrap();
    let path = write_temp(&dir, "minimal.gguf", &minimal_gguf_v3());
    let issues = GgufCompatibilityFixer::diagnose(&path).unwrap();
    assert!(
        issues.iter().any(|i| i.contains("BOS")),
        "should report missing BOS token; got: {:?}",
        issues
    );
}

#[test]
fn diagnose_minimal_gguf_reports_missing_eos() {
    let dir = TempDir::new().unwrap();
    let path = write_temp(&dir, "minimal.gguf", &minimal_gguf_v3());
    let issues = GgufCompatibilityFixer::diagnose(&path).unwrap();
    assert!(
        issues.iter().any(|i| i.contains("EOS")),
        "should report missing EOS token; got: {:?}",
        issues
    );
}

#[test]
fn diagnose_minimal_gguf_reports_missing_vocabulary() {
    let dir = TempDir::new().unwrap();
    let path = write_temp(&dir, "minimal.gguf", &minimal_gguf_v3());
    let issues = GgufCompatibilityFixer::diagnose(&path).unwrap();
    assert!(
        issues.iter().any(|i| i.to_lowercase().contains("vocab")),
        "should report missing vocabulary; got: {:?}",
        issues
    );
}

#[test]
fn diagnose_minimal_gguf_has_at_least_three_issues() {
    let dir = TempDir::new().unwrap();
    let path = write_temp(&dir, "minimal.gguf", &minimal_gguf_v3());
    let issues = GgufCompatibilityFixer::diagnose(&path).unwrap();
    assert!(
        issues.len() >= 3,
        "minimal GGUF should have ≥3 issues (BOS, EOS, vocab); got: {:?}",
        issues
    );
}

#[test]
fn diagnose_all_issue_strings_are_nonempty() {
    let dir = TempDir::new().unwrap();
    let path = write_temp(&dir, "minimal.gguf", &minimal_gguf_v3());
    let issues = GgufCompatibilityFixer::diagnose(&path).unwrap();
    for issue in &issues {
        assert!(!issue.trim().is_empty(), "every issue string must be non-empty");
    }
}

#[test]
fn diagnose_all_issue_strings_are_single_line() {
    let dir = TempDir::new().unwrap();
    let path = write_temp(&dir, "minimal.gguf", &minimal_gguf_v3());
    let issues = GgufCompatibilityFixer::diagnose(&path).unwrap();
    for issue in &issues {
        assert!(
            !issue.contains('\n') && !issue.contains('\r'),
            "issue string must be single-line; got: {:?}",
            issue
        );
    }
}

#[test]
fn diagnose_is_deterministic_across_two_calls() {
    let dir = TempDir::new().unwrap();
    let path = write_temp(&dir, "det.gguf", &minimal_gguf_v3());
    let first = GgufCompatibilityFixer::diagnose(&path).unwrap();
    let second = GgufCompatibilityFixer::diagnose(&path).unwrap();
    assert_eq!(first, second, "diagnose must return identical issues on repeated calls");
}

#[test]
fn diagnose_issue_count_is_bounded() {
    let dir = TempDir::new().unwrap();
    let path = write_temp(&dir, "bounded.gguf", &minimal_gguf_v3());
    let issues = GgufCompatibilityFixer::diagnose(&path).unwrap();
    assert!(issues.len() <= 20, "issue count must be ≤ 20; got {}", issues.len());
}

// ---------------------------------------------------------------------------
// export_fixed() — error handling
// ---------------------------------------------------------------------------

#[test]
fn export_fixed_same_src_dst_returns_err() {
    let dir = TempDir::new().unwrap();
    let path = write_temp(&dir, "same.gguf", &minimal_gguf_v3());
    let result = GgufCompatibilityFixer::export_fixed(&path, &path);
    assert!(result.is_err(), "export_fixed with identical src and dst must return Err");
}

#[test]
fn export_fixed_nonexistent_src_returns_err() {
    let dir = TempDir::new().unwrap();
    let dst = dir.path().join("out.gguf");
    let src = std::path::Path::new("/no/such/file.gguf");
    let result = GgufCompatibilityFixer::export_fixed(src, &dst);
    assert!(result.is_err(), "export_fixed with nonexistent src must return Err");
}

// ---------------------------------------------------------------------------
// export_fixed() — output correctness
// ---------------------------------------------------------------------------

#[test]
fn export_fixed_creates_output_file() {
    let dir = TempDir::new().unwrap();
    let src = write_temp(&dir, "src.gguf", &minimal_gguf_v3());
    let dst = dir.path().join("dst.gguf");
    assert!(!dst.exists(), "dst must not exist before export_fixed");
    GgufCompatibilityFixer::export_fixed(&src, &dst).unwrap();
    assert!(dst.exists(), "export_fixed must create the output file");
}

#[test]
fn export_fixed_output_is_parseable_by_gguf_reader() {
    let dir = TempDir::new().unwrap();
    let src = write_temp(&dir, "src.gguf", &minimal_gguf_v3());
    let dst = dir.path().join("dst.gguf");
    GgufCompatibilityFixer::export_fixed(&src, &dst).unwrap();
    let out = fs::read(&dst).unwrap();
    let result = GgufReader::new(&out);
    assert!(result.is_ok(), "exported GGUF must be parseable: {:?}", result.err());
}

#[test]
fn export_fixed_sets_compat_fixed_flag() {
    let dir = TempDir::new().unwrap();
    let src = write_temp(&dir, "src.gguf", &minimal_gguf_v3());
    let dst = dir.path().join("dst.gguf");
    GgufCompatibilityFixer::export_fixed(&src, &dst).unwrap();
    let out = fs::read(&dst).unwrap();
    let reader = GgufReader::new(&out).unwrap();
    assert_eq!(
        reader.get_bool_metadata("bitnet.compat.fixed"),
        Some(true),
        "export_fixed must set bitnet.compat.fixed=true"
    );
}

#[test]
fn export_fixed_adds_bos_token_id_one() {
    let dir = TempDir::new().unwrap();
    let src = write_temp(&dir, "src.gguf", &minimal_gguf_v3());
    let dst = dir.path().join("dst.gguf");
    GgufCompatibilityFixer::export_fixed(&src, &dst).unwrap();
    let out = fs::read(&dst).unwrap();
    let reader = GgufReader::new(&out).unwrap();
    assert_eq!(
        reader.get_u32_metadata("tokenizer.ggml.bos_token_id"),
        Some(1),
        "export_fixed must set bos_token_id=1 for a minimal GGUF"
    );
}

#[test]
fn export_fixed_adds_eos_token_id_two() {
    let dir = TempDir::new().unwrap();
    let src = write_temp(&dir, "src.gguf", &minimal_gguf_v3());
    let dst = dir.path().join("dst.gguf");
    GgufCompatibilityFixer::export_fixed(&src, &dst).unwrap();
    let out = fs::read(&dst).unwrap();
    let reader = GgufReader::new(&out).unwrap();
    assert_eq!(
        reader.get_u32_metadata("tokenizer.ggml.eos_token_id"),
        Some(2),
        "export_fixed must set eos_token_id=2 for a minimal GGUF"
    );
}

#[test]
fn export_fixed_adds_vocab_size_metadata() {
    let dir = TempDir::new().unwrap();
    let src = write_temp(&dir, "src.gguf", &minimal_gguf_v3());
    let dst = dir.path().join("dst.gguf");
    GgufCompatibilityFixer::export_fixed(&src, &dst).unwrap();
    let out = fs::read(&dst).unwrap();
    let reader = GgufReader::new(&out).unwrap();
    assert!(
        reader.get_u32_metadata("tokenizer.ggml.vocab_size").is_some(),
        "export_fixed must add tokenizer.ggml.vocab_size metadata"
    );
}

#[test]
fn export_fixed_output_is_larger_than_source() {
    // A fixed GGUF must be larger than the bare minimal source because it adds metadata.
    let dir = TempDir::new().unwrap();
    let src = write_temp(&dir, "src.gguf", &minimal_gguf_v3());
    let dst = dir.path().join("dst.gguf");
    let src_len = fs::metadata(&src).unwrap().len();
    GgufCompatibilityFixer::export_fixed(&src, &dst).unwrap();
    let dst_len = fs::metadata(&dst).unwrap().len();
    assert!(
        dst_len > src_len,
        "fixed GGUF must be larger than source; src={src_len}, dst={dst_len}"
    );
}

#[test]
fn export_fixed_does_not_increase_issue_count() {
    let dir = TempDir::new().unwrap();
    let src = write_temp(&dir, "src.gguf", &minimal_gguf_v3());
    let dst = dir.path().join("dst.gguf");
    let before = GgufCompatibilityFixer::diagnose(&src).unwrap().len();
    GgufCompatibilityFixer::export_fixed(&src, &dst).unwrap();
    let after = GgufCompatibilityFixer::diagnose(&dst).unwrap().len();
    assert!(
        after <= before,
        "export_fixed must not increase issues: before={before}, after={after}"
    );
}

// ---------------------------------------------------------------------------
// verify_idempotent()
// ---------------------------------------------------------------------------

#[test]
fn verify_idempotent_false_for_unfixed_minimal() {
    let dir = TempDir::new().unwrap();
    let path = write_temp(&dir, "unfixed.gguf", &minimal_gguf_v3());
    let result = GgufCompatibilityFixer::verify_idempotent(&path);
    assert!(result.is_ok(), "verify_idempotent must not error on valid GGUF");
    assert!(!result.unwrap(), "unfixed minimal GGUF must not be considered idempotent");
}

#[test]
fn verify_idempotent_true_after_export_fixed() {
    let dir = TempDir::new().unwrap();
    let src = write_temp(&dir, "src.gguf", &minimal_gguf_v3());
    let dst = dir.path().join("fixed.gguf");
    GgufCompatibilityFixer::export_fixed(&src, &dst).unwrap();
    let result = GgufCompatibilityFixer::verify_idempotent(&dst);
    assert!(result.is_ok(), "verify_idempotent must not error on fixed GGUF");
    assert!(result.unwrap(), "export_fixed output must pass idempotency check");
}

#[test]
fn verify_idempotent_err_for_nonexistent_path() {
    let result = GgufCompatibilityFixer::verify_idempotent("/no/such/file.gguf");
    assert!(result.is_err(), "verify_idempotent must return Err for nonexistent path");
}

// ---------------------------------------------------------------------------
// print_report()
// ---------------------------------------------------------------------------

#[test]
fn print_report_ok_on_valid_minimal_gguf() {
    let dir = TempDir::new().unwrap();
    let path = write_temp(&dir, "report.gguf", &minimal_gguf_v3());
    let result = GgufCompatibilityFixer::print_report(&path);
    assert!(result.is_ok(), "print_report must succeed on valid GGUF: {:?}", result.err());
}

#[test]
fn print_report_err_on_nonexistent_path() {
    let result = GgufCompatibilityFixer::print_report("/no/such/file.gguf");
    assert!(result.is_err(), "print_report must return Err for nonexistent path");
}

#[test]
fn print_report_ok_on_fixed_gguf() {
    let dir = TempDir::new().unwrap();
    let src = write_temp(&dir, "src.gguf", &minimal_gguf_v3());
    let dst = dir.path().join("fixed.gguf");
    GgufCompatibilityFixer::export_fixed(&src, &dst).unwrap();
    let result = GgufCompatibilityFixer::print_report(&dst);
    assert!(result.is_ok(), "print_report must succeed on a fixed GGUF");
}

// ---------------------------------------------------------------------------
// Stamp file — content invariants
// ---------------------------------------------------------------------------

#[test]
fn stamp_file_exists_after_export_fixed() {
    let dir = TempDir::new().unwrap();
    let src = write_temp(&dir, "src.gguf", &minimal_gguf_v3());
    let dst = dir.path().join("dst.gguf");
    GgufCompatibilityFixer::export_fixed(&src, &dst).unwrap();
    let stamp = dst.with_extension("gguf.compat.json");
    assert!(stamp.exists(), "stamp file must be created at {:?}", stamp);
}

#[test]
fn stamp_file_is_valid_json() {
    let dir = TempDir::new().unwrap();
    let src = write_temp(&dir, "src.gguf", &minimal_gguf_v3());
    let dst = dir.path().join("dst.gguf");
    GgufCompatibilityFixer::export_fixed(&src, &dst).unwrap();
    let stamp = dst.with_extension("gguf.compat.json");
    let content = fs::read_to_string(&stamp).unwrap();
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(&content);
    assert!(parsed.is_ok(), "stamp file must be valid JSON; content:\n{content}");
}

#[test]
fn stamp_json_has_timestamp_field() {
    let dir = TempDir::new().unwrap();
    let src = write_temp(&dir, "src.gguf", &minimal_gguf_v3());
    let dst = dir.path().join("dst.gguf");
    GgufCompatibilityFixer::export_fixed(&src, &dst).unwrap();
    let content = fs::read_to_string(dst.with_extension("gguf.compat.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    let ts = json.get("timestamp").and_then(|v| v.as_str());
    assert!(ts.is_some(), "stamp must have a non-null 'timestamp' string field");
    assert!(!ts.unwrap().is_empty(), "timestamp must not be empty");
}

#[test]
fn stamp_json_has_version_field_with_semver() {
    let dir = TempDir::new().unwrap();
    let src = write_temp(&dir, "src.gguf", &minimal_gguf_v3());
    let dst = dir.path().join("dst.gguf");
    GgufCompatibilityFixer::export_fixed(&src, &dst).unwrap();
    let content = fs::read_to_string(dst.with_extension("gguf.compat.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    let ver = json.get("version").and_then(|v| v.as_str()).expect("stamp must have 'version'");
    let parts: Vec<&str> = ver.split('.').collect();
    assert!(parts.len() >= 3, "version must have ≥3 dot-separated parts; got: {ver}");
    for part in &parts[..3] {
        let numeric = part.split('-').next().unwrap_or(part);
        assert!(
            numeric.chars().all(|c| c.is_ascii_digit()),
            "semver component {numeric:?} is not numeric in {ver:?}"
        );
    }
}

#[test]
fn stamp_json_has_fixes_applied_array() {
    let dir = TempDir::new().unwrap();
    let src = write_temp(&dir, "src.gguf", &minimal_gguf_v3());
    let dst = dir.path().join("dst.gguf");
    GgufCompatibilityFixer::export_fixed(&src, &dst).unwrap();
    let content = fs::read_to_string(dst.with_extension("gguf.compat.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    let fixes = json.get("fixes_applied");
    assert!(fixes.is_some(), "stamp must have 'fixes_applied' field");
    assert!(fixes.unwrap().is_array(), "'fixes_applied' must be a JSON array");
}

#[test]
fn stamp_fixes_applied_nonempty_for_minimal_gguf() {
    let dir = TempDir::new().unwrap();
    let src = write_temp(&dir, "src.gguf", &minimal_gguf_v3());
    let dst = dir.path().join("dst.gguf");
    GgufCompatibilityFixer::export_fixed(&src, &dst).unwrap();
    let content = fs::read_to_string(dst.with_extension("gguf.compat.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    let fixes = json["fixes_applied"].as_array().unwrap();
    assert!(
        !fixes.is_empty(),
        "fixes_applied must be non-empty for a minimal GGUF that needed patches"
    );
}

#[test]
fn stamp_fixes_applied_entries_are_nonempty_strings() {
    let dir = TempDir::new().unwrap();
    let src = write_temp(&dir, "src.gguf", &minimal_gguf_v3());
    let dst = dir.path().join("dst.gguf");
    GgufCompatibilityFixer::export_fixed(&src, &dst).unwrap();
    let content = fs::read_to_string(dst.with_extension("gguf.compat.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    let fixes = json["fixes_applied"].as_array().unwrap();
    for fix in fixes {
        let s = fix.as_str().expect("each fix entry must be a string");
        assert!(!s.trim().is_empty(), "fix entry string must not be empty");
    }
}
