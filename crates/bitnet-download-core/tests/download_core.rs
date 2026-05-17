//! Integration tests for `bitnet-download-core`.
//!
//! These tests broaden behavioural coverage for the four public entry points
//! exposed by the crate: `parse_content_range_total`, `offline_enabled`,
//! `atomic_write`, and the `DownloadValidationError` enum.
//!
//! Environment-mutating tests are serialised with `#[serial_test::serial]`
//! against the `bitnet_env` key to avoid races with each other and with the
//! crate's inline tests.

use std::error::Error;
use std::fs;
use std::path::Path;

use bitnet_download_core::{
    DownloadValidationError, atomic_write, offline_enabled, parse_content_range_total,
};
use serial_test::serial;

// ---------------------------------------------------------------------------
// parse_content_range_total
// ---------------------------------------------------------------------------

#[test]
fn parse_content_range_well_formed_simple() {
    assert_eq!(parse_content_range_total("bytes 0-99/100"), Some(100));
}

#[test]
fn parse_content_range_well_formed_with_large_window() {
    assert_eq!(parse_content_range_total("bytes 1024-2047/4096"), Some(4096));
}

#[test]
fn parse_content_range_leading_and_trailing_whitespace_trimmed() {
    // The implementation calls `.trim()` on the whole input, so leading and
    // trailing whitespace (including CRLF) is tolerated.
    assert_eq!(parse_content_range_total("  bytes 0-0/1234  "), Some(1234));
    assert_eq!(parse_content_range_total("\tbytes 0-0/42\n"), Some(42));
    assert_eq!(parse_content_range_total("bytes 0-0/1234\r\n"), Some(1234));
}

#[test]
fn parse_content_range_star_total_returns_none() {
    // RFC 7233: `*` for the complete length means the total is unknown, so we
    // cannot validate against it and must return `None`.
    assert_eq!(parse_content_range_total("bytes 0-99/*"), None);
    assert_eq!(parse_content_range_total("bytes */*"), None);
}

#[test]
fn parse_content_range_star_range_with_known_total() {
    // The range can be `*` even when the total is known.
    assert_eq!(parse_content_range_total("bytes */2048"), Some(2048));
}

#[test]
fn parse_content_range_missing_bytes_prefix_returns_none() {
    assert_eq!(parse_content_range_total("0-99/100"), None);
    assert_eq!(parse_content_range_total("items 0-99/100"), None);
}

#[test]
fn parse_content_range_no_slash_returns_none() {
    assert_eq!(parse_content_range_total("bytes 0-99"), None);
    assert_eq!(parse_content_range_total("bytes 0-99 100"), None);
}

#[test]
fn parse_content_range_empty_input_returns_none() {
    assert_eq!(parse_content_range_total(""), None);
    assert_eq!(parse_content_range_total("   "), None);
}

#[test]
fn parse_content_range_non_numeric_total_returns_none() {
    assert_eq!(parse_content_range_total("bytes 0-99/abc"), None);
    assert_eq!(parse_content_range_total("bytes 0-99/12x4"), None);
}

#[test]
fn parse_content_range_non_numeric_range_returns_none() {
    assert_eq!(parse_content_range_total("bytes a-b/1234"), None);
    assert_eq!(parse_content_range_total("bytes 0-x/1234"), None);
}

#[test]
fn parse_content_range_start_greater_than_end_returns_none() {
    assert_eq!(parse_content_range_total("bytes 5-1/1234"), None);
}

#[test]
fn parse_content_range_range_without_dash_returns_none() {
    // `range.split_once('-')` returns `None` when there is no dash, which
    // should propagate as `None`.
    assert_eq!(parse_content_range_total("bytes 100/1234"), None);
}

#[test]
fn parse_content_range_multiple_dashes_in_range_returns_none() {
    // `bytes 0-0-1/1234`: the `end` segment contains a `-`, which is rejected.
    assert_eq!(parse_content_range_total("bytes 0-0-1/1234"), None);
}

#[test]
fn parse_content_range_multiple_slashes_returns_none() {
    // Multiple `/` separators are explicitly rejected.
    assert_eq!(parse_content_range_total("bytes 0-0/1/2"), None);
    assert_eq!(parse_content_range_total("bytes 0-0/1234/extra"), None);
}

#[test]
fn parse_content_range_zero_total_returns_none() {
    // Zero is treated as invalid because a zero-length file cannot meaningfully
    // appear in a `Content-Range` response.
    assert_eq!(parse_content_range_total("bytes 0-0/0"), None);
}

#[test]
fn parse_content_range_very_large_total_near_u64_max() {
    let max = u64::MAX;
    let input = format!("bytes 0-0/{max}");
    assert_eq!(parse_content_range_total(&input), Some(max));

    // Overflow past u64::MAX should fail to parse.
    let overflow = format!("bytes 0-0/{}0", u64::MAX);
    assert_eq!(parse_content_range_total(&overflow), None);
}

#[test]
fn parse_content_range_case_sensitive_prefix() {
    // `strip_prefix("bytes ")` is byte-exact, so upper-case `BYTES` is rejected.
    assert_eq!(parse_content_range_total("BYTES 0-0/1234"), None);
    assert_eq!(parse_content_range_total("Bytes 0-0/1234"), None);
}

#[test]
fn parse_content_range_negative_numbers_rejected() {
    // `u64::parse` rejects negative numbers.
    assert_eq!(parse_content_range_total("bytes -1-5/1234"), None);
    assert_eq!(parse_content_range_total("bytes 0-0/-1"), None);
}

// ---------------------------------------------------------------------------
// offline_enabled
// ---------------------------------------------------------------------------

#[test]
#[serial(bitnet_env)]
fn offline_enabled_cli_flag_short_circuits_regardless_of_env() {
    // When the CLI flag is set we should not even look at the env var, so we
    // do not need env isolation for this case — but use the serial guard just
    // to keep this consistent with the rest of the env-var tests.
    let _saved = SavedEnvVar::capture("BITNET_OFFLINE");
    // Try with both `BITNET_OFFLINE` unset and set to a falsey value, and a
    // truthy value, to assert short-circuit behaviour holds.
    // SAFETY: env mutation is gated by the SavedEnvVar restore at scope exit.
    unsafe { std::env::remove_var("BITNET_OFFLINE") };
    assert!(offline_enabled(true));
    unsafe { std::env::set_var("BITNET_OFFLINE", "0") };
    assert!(offline_enabled(true));
    unsafe { std::env::set_var("BITNET_OFFLINE", "1") };
    assert!(offline_enabled(true));
}

/// Drives every documented scenario for the `BITNET_OFFLINE` environment
/// variable in a single, sequential test. This avoids any cross-test race on
/// the shared global env table.
#[test]
#[serial(bitnet_env)]
fn offline_enabled_env_scenarios_sequential() {
    let _saved = SavedEnvVar::capture("BITNET_OFFLINE");

    // Unset env -> offline disabled.
    // SAFETY: env mutation guarded by `_saved`.
    unsafe { std::env::remove_var("BITNET_OFFLINE") };
    assert!(!offline_enabled(false), "unset BITNET_OFFLINE should leave offline mode disabled");

    // Empty env -> offline disabled (empty string is not truthy).
    unsafe { std::env::set_var("BITNET_OFFLINE", "") };
    assert!(!offline_enabled(false), "empty BITNET_OFFLINE should not enable offline mode");

    // Truthy values.
    for truthy in ["1", "true", "TRUE", "True", "yes", "YES", "on", "ON", " 1 ", "\tyes\n"] {
        unsafe { std::env::set_var("BITNET_OFFLINE", truthy) };
        assert!(
            offline_enabled(false),
            "expected BITNET_OFFLINE={truthy:?} to enable offline mode"
        );
    }

    // Falsey / non-truthy values that should be ignored.
    for falsey in ["0", "false", "off", "no", "nope", "maybe", "2", "trueish"] {
        unsafe { std::env::set_var("BITNET_OFFLINE", falsey) };
        assert!(
            !offline_enabled(false),
            "expected BITNET_OFFLINE={falsey:?} to leave offline mode disabled"
        );
    }
}

// ---------------------------------------------------------------------------
// atomic_write
// ---------------------------------------------------------------------------

#[test]
fn atomic_write_writes_bytes_to_fresh_path() -> Result<(), Box<dyn Error>> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("fresh.bin");
    assert!(!path.exists(), "precondition: target should not exist");

    atomic_write(&path, b"hello-world")?;

    assert!(path.exists(), "file should exist after write");
    assert_eq!(fs::read(&path)?, b"hello-world");
    Ok(())
}

#[test]
fn atomic_write_overwrites_existing_file() -> Result<(), Box<dyn Error>> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("etag");

    atomic_write(&path, b"v1")?;
    assert_eq!(fs::read(&path)?, b"v1");

    atomic_write(&path, b"v2-longer-than-v1")?;
    assert_eq!(fs::read(&path)?, b"v2-longer-than-v1");

    atomic_write(&path, b"v3")?;
    assert_eq!(fs::read(&path)?, b"v3");
    Ok(())
}

#[test]
fn atomic_write_in_existing_parent_directory_succeeds() -> Result<(), Box<dyn Error>> {
    let dir = tempfile::tempdir()?;
    let nested = dir.path().join("nested");
    fs::create_dir(&nested)?;
    let path = nested.join("file");

    atomic_write(&path, b"payload")?;

    assert_eq!(fs::read(&path)?, b"payload");
    Ok(())
}

#[test]
fn atomic_write_fails_when_parent_directory_missing() -> Result<(), Box<dyn Error>> {
    let dir = tempfile::tempdir()?;
    let missing = dir.path().join("does-not-exist");
    let path = missing.join("file");

    let err = atomic_write(&path, b"payload");
    // The underlying `tempfile::NamedTempFile::new_in` fails with `NotFound`
    // (or in some kernels, a different IO error). We only assert that an error
    // is returned and that the target file was not created.
    assert!(!path.exists(), "target file must not exist after failure");
    assert!(err.is_err(), "atomic_write should fail when parent directory is absent");
    Ok(())
}

#[test]
fn atomic_write_round_trip_matches_input() -> Result<(), Box<dyn Error>> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("round-trip");

    let payload: Vec<u8> = (0u16..=255u16).map(|b| b as u8).collect();
    atomic_write(&path, &payload)?;

    let read_back = fs::read(&path)?;
    assert_eq!(read_back, payload);
    Ok(())
}

#[test]
fn atomic_write_handles_empty_payload() -> Result<(), Box<dyn Error>> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("empty");

    atomic_write(&path, &[])?;

    assert!(path.exists(), "empty file should still be created");
    assert_eq!(fs::read(&path)?, Vec::<u8>::new());
    Ok(())
}

#[test]
fn atomic_write_handles_binary_extremes() -> Result<(), Box<dyn Error>> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("binary");

    let payload = [0x00u8, 0xFFu8, 0x00u8, 0xFFu8, 0x7Fu8, 0x80u8];
    atomic_write(&path, &payload)?;

    assert_eq!(fs::read(&path)?, payload);
    Ok(())
}

#[test]
fn atomic_write_handles_large_payload() -> Result<(), Box<dyn Error>> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("large");

    let payload: Vec<u8> = (0..1024 * 32).map(|i| (i % 251) as u8).collect();
    atomic_write(&path, &payload)?;

    let read_back = fs::read(&path)?;
    assert_eq!(read_back.len(), payload.len());
    assert_eq!(read_back, payload);
    Ok(())
}

#[test]
#[serial(cwd)]
fn atomic_write_to_path_without_parent_uses_cwd() -> Result<(), Box<dyn Error>> {
    // A path with no directory component should fall back to "." for the
    // temp-file location. Use a serialised CWD swap to avoid clobbering other
    // tests that read the working directory.
    let _saved = SavedEnvVar::capture("BITNET_OFFLINE"); // unrelated, used only for shared scheduling
    let dir = tempfile::tempdir()?;
    let previous = std::env::current_dir()?;
    std::env::set_current_dir(dir.path())?;
    let result = atomic_write(Path::new("bare-name"), b"bare");
    let read_back = fs::read("bare-name");
    std::env::set_current_dir(&previous)?;

    result?;
    assert_eq!(read_back?, b"bare");
    Ok(())
}

// ---------------------------------------------------------------------------
// DownloadValidationError
// ---------------------------------------------------------------------------

#[test]
fn validation_error_display_message_format() {
    let err = DownloadValidationError::Truncated { downloaded: 7, expected: 11 };
    assert_eq!(err.to_string(), "download truncated: got 7 bytes, expected 11 bytes");
}

#[test]
fn validation_error_debug_includes_field_values() {
    let err = DownloadValidationError::Truncated { downloaded: 1, expected: 2 };
    let dbg = format!("{err:?}");
    assert!(dbg.contains("Truncated"), "debug output should name the variant: {dbg}");
    assert!(dbg.contains('1'), "debug output should contain downloaded value: {dbg}");
    assert!(dbg.contains('2'), "debug output should contain expected value: {dbg}");
}

#[test]
fn validation_error_equality_and_clone_semantics() {
    let a = DownloadValidationError::Truncated { downloaded: 4, expected: 8 };
    let b = DownloadValidationError::Truncated { downloaded: 4, expected: 8 };
    let c = DownloadValidationError::Truncated { downloaded: 5, expected: 8 };
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn validation_error_implements_std_error() {
    // Compile-time check that `DownloadValidationError: std::error::Error`.
    fn assert_error<T: Error>(_: &T) {}
    let err = DownloadValidationError::Truncated { downloaded: 0, expected: 1 };
    assert_error(&err);

    // The error has no underlying source.
    let dyn_err: &dyn Error = &err;
    assert!(dyn_err.source().is_none());
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// RAII helper that snapshots an environment variable on construction and
/// restores it on drop. Used inside `#[serial(...)]` tests so that any
/// mutation to a process-global env var does not bleed into other tests.
struct SavedEnvVar {
    key: &'static str,
    value: Option<std::ffi::OsString>,
}

impl SavedEnvVar {
    fn capture(key: &'static str) -> Self {
        Self { key, value: std::env::var_os(key) }
    }
}

impl Drop for SavedEnvVar {
    fn drop(&mut self) {
        // SAFETY: We are restoring the env var to its previously observed
        // value. Tests using this helper are gated by `#[serial(...)]` (or
        // are otherwise the only mutator of this variable in this test
        // binary), so the unsafe env-mutation is sound under the test
        // scheduler.
        unsafe {
            match self.value.take() {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }
}
