//! Property-based tests for `bitnet-compat`.
//!
//! Key invariants tested:
//! - `diagnose()` is deterministic: same file bytes → same issue list
//! - All issue strings returned by `diagnose()` are non-empty
//! - A minimal GGUF (header only, no metadata) always produces issues
//! - `export_fixed()` produces a file that passes `verify_idempotent()`
//! - Non-existent `.gguf` paths are rejected with an error (not a panic)
//! - Empty and null-byte-containing paths are always rejected
//! - The crate version string follows semantic versioning (X.Y.Z)

use bitnet_compat::gguf_fixer::GgufCompatibilityFixer;
use proptest::prelude::*;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a minimal syntactically-valid GGUF blob: magic + version + zero counts.
fn make_minimal_gguf(version: u32) -> Vec<u8> {
    let mut data = Vec::with_capacity(24);
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&version.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count
    data
}

/// Write bytes to a temp file and return the dir (keeps it alive) + path.
fn write_temp_gguf(dir: &TempDir, bytes: &[u8], name: &str) -> std::path::PathBuf {
    let path = dir.path().join(name);
    std::fs::write(&path, bytes).expect("write temp gguf");
    path
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

proptest! {
    /// `diagnose()` is deterministic: calling it twice on the same file content
    /// always returns an identical issue list (same order, same strings).
    ///
    /// Note: `ggus` only supports GGUF version 3.
    #[test]
    fn prop_diagnose_is_deterministic(_seed in 0u32..100u32) {
        let bytes = make_minimal_gguf(3);
        let dir = TempDir::new().unwrap();
        let path = write_temp_gguf(&dir, &bytes, "det.gguf");

        let first = GgufCompatibilityFixer::diagnose(&path).unwrap();
        let second = GgufCompatibilityFixer::diagnose(&path).unwrap();

        prop_assert_eq!(first.len(), second.len(), "issue count must be stable across calls");
        for (a, b) in first.iter().zip(second.iter()) {
            prop_assert_eq!(a, b, "issue string must be identical across calls");
        }
    }

    /// All issue strings returned by `diagnose()` are non-empty and non-whitespace.
    #[test]
    fn prop_issues_are_nonempty_strings(_seed in 0u32..100u32) {
        let bytes = make_minimal_gguf(3);
        let dir = TempDir::new().unwrap();
        let path = write_temp_gguf(&dir, &bytes, "nonempty.gguf");

        let issues = GgufCompatibilityFixer::diagnose(&path).unwrap();
        for issue in &issues {
            prop_assert!(!issue.trim().is_empty(),
                "issue string must not be empty or whitespace-only: {:?}", issue);
        }
    }

    /// A minimal GGUF (header only, no metadata) must produce at least one issue
    /// because BOS/EOS/vocab_size metadata are missing.
    #[test]
    fn prop_minimal_gguf_always_has_issues(_seed in 0u32..100u32) {
        let bytes = make_minimal_gguf(3);
        let dir = TempDir::new().unwrap();
        let path = write_temp_gguf(&dir, &bytes, "minimal.gguf");

        let issues = GgufCompatibilityFixer::diagnose(&path).unwrap();
        prop_assert!(!issues.is_empty(),
            "minimal GGUF with no metadata must have at least one issue");
    }

    /// After `export_fixed()`, `verify_idempotent()` must return `Ok(true)`.
    #[test]
    fn prop_export_fixed_passes_idempotency_check(_seed in 0u32..100u32) {
        let bytes = make_minimal_gguf(3);
        let dir = TempDir::new().unwrap();
        let src = write_temp_gguf(&dir, &bytes, "src.gguf");
        let dst = dir.path().join("fixed.gguf");

        GgufCompatibilityFixer::export_fixed(&src, &dst).unwrap();
        let idempotent = GgufCompatibilityFixer::verify_idempotent(&dst).unwrap();
        prop_assert!(idempotent, "export_fixed output must pass idempotency check");
    }

    /// `export_fixed()` never increases the number of diagnostic issues.
    ///
    /// The fixed file must have the same or fewer issues than the source.
    #[test]
    fn prop_export_fixed_does_not_increase_issues(_seed in 0u32..100u32) {
        let bytes = make_minimal_gguf(3);
        let dir = TempDir::new().unwrap();
        let src = write_temp_gguf(&dir, &bytes, "src2.gguf");
        let dst = dir.path().join("fixed2.gguf");

        let before = GgufCompatibilityFixer::diagnose(&src).unwrap().len();
        GgufCompatibilityFixer::export_fixed(&src, &dst).unwrap();
        let after = GgufCompatibilityFixer::diagnose(&dst).unwrap().len();

        prop_assert!(
            after <= before,
            "export_fixed must not add issues: before={before}, after={after}"
        );
    }
}

proptest! {
    /// Non-existent paths with a `.gguf` extension return an `Err`, never panic.
    ///
    /// The fixer must not crash on valid-looking paths that don't exist on disk.
    #[test]
    fn prop_nonexistent_gguf_path_returns_err(name in "[a-z]{1,20}") {
        let path = format!("/nonexistent/tmp/bitnet_prop_test_{name}.gguf");
        let result = GgufCompatibilityFixer::diagnose(&path);
        prop_assert!(result.is_err(), "non-existent .gguf path must return Err, got Ok");
    }

    /// An empty path string is always rejected with an `Err`.
    #[test]
    fn prop_empty_path_is_rejected(_seed in 0u32..10u32) {
        let result = GgufCompatibilityFixer::diagnose("");
        prop_assert!(result.is_err(), "empty path must be rejected with Err");
    }

    /// Paths containing a null byte are always rejected (OS-level invalid path).
    #[test]
    fn prop_null_byte_path_is_rejected(prefix in "[a-z]{1,10}") {
        let path = format!("{prefix}\0.gguf");
        let result = GgufCompatibilityFixer::diagnose(&path);
        prop_assert!(result.is_err(), "path with null byte must be rejected");
    }

    /// The crate version string follows the semver X.Y.Z pattern.
    ///
    /// Each of the first three dot-separated components must be purely numeric.
    #[test]
    fn prop_crate_version_is_semver(_seed in 0u32..10u32) {
        let version = env!("CARGO_PKG_VERSION");
        let parts: Vec<&str> = version.split('.').collect();
        prop_assert!(
            parts.len() >= 3,
            "semver must have at least 3 dot-separated parts, got: {version:?}"
        );
        for part in &parts[..3] {
            // Strip any pre-release suffix (e.g. "0-alpha") before checking digits.
            let numeric = part.split('-').next().unwrap_or(part);
            prop_assert!(
                numeric.chars().all(|c| c.is_ascii_digit()),
                "semver component {numeric:?} is not numeric in version {version:?}"
            );
        }
    }
}
