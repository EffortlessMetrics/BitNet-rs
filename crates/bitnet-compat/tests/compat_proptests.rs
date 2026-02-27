//! Additional property-based tests for `bitnet-compat`.
//!
//! These tests focus on invariants not already covered in `property_tests.rs`:
//! - Issue strings are single-line and contain human-readable keywords (BOS/EOS).
//! - The number of issues is bounded by a reasonable ceiling.
//! - `verify_idempotent()` returns `false` for files never processed by `export_fixed()`.
//! - `export_fixed()` creates a companion stamp JSON file on disk.
//! - Files with invalid GGUF magic (not "GGUF") are always rejected with `Err`.

use bitnet_compat::gguf_fixer::GgufCompatibilityFixer;
use proptest::prelude::*;
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

fn write_temp(dir: &TempDir, bytes: &[u8], name: &str) -> std::path::PathBuf {
    let path = dir.path().join(name);
    std::fs::write(&path, bytes).expect("write temp file");
    path
}

// ---------------------------------------------------------------------------
// Properties: issue string format
// ---------------------------------------------------------------------------

proptest! {
    /// Every issue string returned by `diagnose()` contains no newline characters.
    ///
    /// Issues are designed to be displayed as single-line bullet points; embedding
    /// a newline would break CLI output formatting.
    #[test]
    fn prop_issue_strings_have_no_newlines(_seed in 0u32..50u32) {
        let dir = TempDir::new().unwrap();
        let path = write_temp(&dir, &minimal_gguf_v3(), "no_newlines.gguf");

        let issues = GgufCompatibilityFixer::diagnose(&path).unwrap();
        for issue in &issues {
            prop_assert!(
                !issue.contains('\n') && !issue.contains('\r'),
                "issue string must be single-line, got: {:?}", issue
            );
        }
    }

    /// When BOS token metadata is absent the corresponding issue must mention "BOS".
    ///
    /// A minimal v3 GGUF has no metadata at all, so the BOS-missing check always fires.
    /// The resulting message must contain a human-readable "BOS" keyword so operators
    /// can understand the problem without consulting source code.
    #[test]
    fn prop_missing_bos_issue_mentions_bos(_seed in 0u32..50u32) {
        let dir = TempDir::new().unwrap();
        let path = write_temp(&dir, &minimal_gguf_v3(), "bos.gguf");

        let issues = GgufCompatibilityFixer::diagnose(&path).unwrap();
        let has_bos = issues.iter().any(|i| i.contains("BOS") || i.contains("bos"));
        prop_assert!(has_bos, "one issue must mention BOS; got: {:?}", issues);
    }

    /// When EOS token metadata is absent the corresponding issue must mention "EOS".
    #[test]
    fn prop_missing_eos_issue_mentions_eos(_seed in 0u32..50u32) {
        let dir = TempDir::new().unwrap();
        let path = write_temp(&dir, &minimal_gguf_v3(), "eos.gguf");

        let issues = GgufCompatibilityFixer::diagnose(&path).unwrap();
        let has_eos = issues.iter().any(|i| i.contains("EOS") || i.contains("eos"));
        prop_assert!(has_eos, "one issue must mention EOS; got: {:?}", issues);
    }

    /// The number of issues for any GGUF file is at most 20.
    ///
    /// An unbounded issue list would be a sign of runaway diagnostic logic.
    /// For a minimal v3 GGUF (missing BOS, EOS, vocab_size) the count should be 3.
    #[test]
    fn prop_issue_count_is_bounded(_seed in 0u32..50u32) {
        let dir = TempDir::new().unwrap();
        let path = write_temp(&dir, &minimal_gguf_v3(), "bounded.gguf");

        let issues = GgufCompatibilityFixer::diagnose(&path).unwrap();
        prop_assert!(
            issues.len() <= 20,
            "issue count must be ≤ 20, got {}", issues.len()
        );
    }
}

// ---------------------------------------------------------------------------
// Properties: verify_idempotent
// ---------------------------------------------------------------------------

proptest! {
    /// `verify_idempotent()` returns `Ok(false)` for a file that has never been
    /// passed through `export_fixed()`.
    ///
    /// The function returns `false` when the file still has diagnostic issues,
    /// which is true for any minimal GGUF (missing BOS/EOS/vocab).
    #[test]
    fn prop_verify_idempotent_false_on_unfixed_minimal(_seed in 0u32..50u32) {
        let dir = TempDir::new().unwrap();
        let path = write_temp(&dir, &minimal_gguf_v3(), "unfixed.gguf");

        let result = GgufCompatibilityFixer::verify_idempotent(&path);
        prop_assert!(result.is_ok(), "verify_idempotent must not Err on a valid GGUF file");
        prop_assert!(
            !result.unwrap(),
            "verify_idempotent must return false for an unfixed minimal GGUF"
        );
    }
}

// ---------------------------------------------------------------------------
// Properties: export_fixed side effects
// ---------------------------------------------------------------------------

proptest! {
    /// `export_fixed()` always creates a companion `.gguf.compat.json` stamp file.
    ///
    /// The stamp records the timestamp, crate version, and list of fixes applied.
    /// Downstream tooling (CI, audit logs) depends on this file being present.
    #[test]
    fn prop_export_fixed_creates_stamp_file(_seed in 0u32..30u32) {
        let dir = TempDir::new().unwrap();
        let src = write_temp(&dir, &minimal_gguf_v3(), "stamp_src.gguf");
        let dst = dir.path().join("stamp_dst.gguf");

        GgufCompatibilityFixer::export_fixed(&src, &dst).unwrap();

        // write_stamp uses Path::with_extension("gguf.compat.json") on the output path.
        let stamp_path = dst.with_extension("gguf.compat.json");
        prop_assert!(
            stamp_path.exists(),
            "stamp file must exist at {:?} after export_fixed()", stamp_path
        );
    }

    /// The stamp JSON produced by `export_fixed()` contains a `"version"` field.
    ///
    /// This field carries the crate version of the tool that produced the fix,
    /// enabling operators to track which version of bitnet-compat was used.
    #[test]
    fn prop_stamp_json_contains_version_field(_seed in 0u32..30u32) {
        let dir = TempDir::new().unwrap();
        let src = write_temp(&dir, &minimal_gguf_v3(), "vsrc.gguf");
        let dst = dir.path().join("vdst.gguf");

        GgufCompatibilityFixer::export_fixed(&src, &dst).unwrap();

        let stamp_path = dst.with_extension("gguf.compat.json");
        let content = std::fs::read_to_string(&stamp_path).expect("read stamp file");
        prop_assert!(
            content.contains("\"version\""),
            "stamp JSON must contain a \"version\" field; got: {:?}", content
        );
    }
}

// ---------------------------------------------------------------------------
// Properties: input validation — wrong magic bytes
// ---------------------------------------------------------------------------

proptest! {
    /// Files whose first 4 bytes differ from the "GGUF" magic are rejected with `Err`.
    ///
    /// The GGUF format is identified by its 4-byte magic at offset 0. Any file lacking
    /// this marker must be refused early rather than parsed incorrectly.
    #[test]
    fn prop_non_gguf_magic_rejected_with_error(
        bytes in proptest::collection::vec(any::<u8>(), 24usize)
            .prop_filter("skip accidental GGUF magic", |b| &b[..4] != b"GGUF")
    ) {
        let dir = TempDir::new().unwrap();
        let path = write_temp(&dir, &bytes, "badmagic.gguf");

        let result = GgufCompatibilityFixer::diagnose(&path);
        prop_assert!(
            result.is_err(),
            "bytes not starting with GGUF magic must be rejected with Err"
        );
    }
}
