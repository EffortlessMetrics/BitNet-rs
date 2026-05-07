use std::{
    io::Write,
    path::{Path, PathBuf},
};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DownloadValidationError {
    #[error("download truncated: got {downloaded} bytes, expected {expected} bytes")]
    Truncated { downloaded: u64, expected: u64 },
}

/// Returns true when download logic should operate in offline mode.
#[must_use]
pub fn offline_enabled(cli_offline: bool) -> bool {
    cli_offline || std::env::var("BITNET_OFFLINE").ok().as_deref().is_some_and(is_truthy_env_value)
}

#[must_use]
fn is_truthy_env_value(value: &str) -> bool {
    let value = value.trim();
    value.eq_ignore_ascii_case("1")
        || value.eq_ignore_ascii_case("true")
        || value.eq_ignore_ascii_case("yes")
        || value.eq_ignore_ascii_case("on")
}

/// Parses `Content-Range` total bytes from values like `bytes 0-0/1234`.
#[must_use]
pub fn parse_content_range_total(content_range: &str) -> Option<u64> {
    let value = content_range.trim();
    let (range, total) = value.split_once('/')?;

    // Reject malformed inputs with multiple `/` separators (e.g. `bytes 0-0/1/2`).
    if total.contains('/') {
        return None;
    }

    // RFC 7233 uses `*` when the complete length is unknown.
    if total == "*" {
        return None;
    }

    let range = range.strip_prefix("bytes ")?;
    if range != "*" {
        let (start, end) = range.split_once('-')?;
        if end.contains('-') {
            return None;
        }

        let start = start.parse::<u64>().ok()?;
        let end = end.parse::<u64>().ok()?;
        if start > end {
            return None;
        }
    }

    let total = total.parse::<u64>().ok()?;
    if total == 0 {
        return None;
    }
    Some(total)
}

/// Ensure downloaded bytes match expected total when available.
pub const fn validate_downloaded_len(
    downloaded: u64,
    expected_total: Option<u64>,
) -> Result<(), DownloadValidationError> {
    if let Some(expected) = expected_total
        && downloaded != expected
    {
        return Err(DownloadValidationError::Truncated { downloaded, expected });
    }
    Ok(())
}

/// Atomic write helper for small metadata files (etag/last-modified).
///
/// Uses a randomly named NamedTempFile in the destination's parent directory
/// to avoid colliding with a pre-existing sibling at `<path>.tmp` (which the
/// previous implementation would have clobbered or mis-renamed across), and
/// to remain race-free across concurrent writers targeting the same path.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let parent = atomic_write_parent(path);
    let mut tmp = tempfile::NamedTempFile::new_in(parent)?;
    tmp.write_all(bytes)?;
    tmp.flush()?;

    #[cfg(unix)]
    {
        tmp.as_file().sync_all()?;
    }

    tmp.persist(path).map_err(|err| err.error)?;

    #[cfg(unix)]
    {
        if let Some(parent) = path.parent()
            && let Ok(dir) = std::fs::File::open(parent)
        {
            let _ = dir.sync_all();
        }
    }

    Ok(())
}

fn atomic_write_parent(path: &Path) -> PathBuf {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn parses_content_range_total() {
        assert_eq!(parse_content_range_total("bytes 0-0/1234"), Some(1234));
        assert_eq!(parse_content_range_total("invalid"), None);
        assert_eq!(parse_content_range_total("bytes */*"), None);
        assert_eq!(parse_content_range_total("bytes */2048"), Some(2048));
        assert_eq!(parse_content_range_total("bytes 0-0/1234\r\n"), Some(1234));
        assert_eq!(parse_content_range_total("bytes 0-0/1/2"), None);
        assert_eq!(parse_content_range_total("items 0-0/1234"), None);
        assert_eq!(parse_content_range_total("bytes nope/1234"), None);
        assert_eq!(parse_content_range_total("bytes 5-1/1234"), None);
        assert_eq!(parse_content_range_total("bytes 0-0/0"), None);
    }

    #[test]
    #[serial_test::serial]
    fn offline_enabled_accepts_common_truthy_env_values() {
        unsafe { std::env::remove_var("BITNET_OFFLINE") };
        assert!(!offline_enabled(false));
        assert!(offline_enabled(true));

        for truthy in ["1", "true", "TRUE", " yes ", "YES", "on", "ON"] {
            unsafe { std::env::set_var("BITNET_OFFLINE", truthy) };
            assert!(
                offline_enabled(false),
                "expected truthy value {truthy} to enable offline mode"
            );
        }

        for falsey in ["0", "false", "off", "no", ""] {
            unsafe { std::env::set_var("BITNET_OFFLINE", falsey) };
            assert!(
                !offline_enabled(false),
                "expected falsey value {falsey} to disable offline mode"
            );
        }

        unsafe { std::env::remove_var("BITNET_OFFLINE") };
    }

    #[test]
    fn validates_downloaded_len() {
        assert!(validate_downloaded_len(1024, Some(1024)).is_ok());
        assert!(validate_downloaded_len(1024, None).is_ok());
        assert!(matches!(
            validate_downloaded_len(1, Some(2)),
            Err(DownloadValidationError::Truncated { downloaded: 1, expected: 2 })
        ));
    }

    #[test]
    fn atomic_write_persists_contents() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let file_path = dir.path().join("etag");
        atomic_write(&file_path, b"etag-v1").expect("atomic write should succeed");
        assert_eq!(fs::read(&file_path).expect("read file"), b"etag-v1");

        atomic_write(&file_path, b"etag-v2").expect("atomic overwrite should succeed");
        assert_eq!(fs::read(&file_path).expect("read file"), b"etag-v2");
    }

    #[test]
    fn atomic_write_ignores_preexisting_tmp_extension_path() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let file_path = dir.path().join("etag");
        let conflicting_tmp_path = file_path.with_extension("tmp");
        fs::create_dir(&conflicting_tmp_path).expect("create conflicting tmp directory");

        atomic_write(&file_path, b"etag").expect("atomic write should not depend on .tmp path");

        assert_eq!(fs::read(&file_path).expect("read file"), b"etag");
        assert!(
            conflicting_tmp_path.is_dir(),
            "pre-existing sibling temp path should be left untouched"
        );
    }

    #[test]
    #[serial_test::serial]
    fn atomic_write_supports_relative_paths_without_parent() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let previous_dir = std::env::current_dir().expect("read current dir");
        std::env::set_current_dir(dir.path()).expect("switch to temp dir");

        let result = atomic_write(Path::new("etag"), b"relative");
        let read_result = fs::read("etag");
        std::env::set_current_dir(previous_dir).expect("restore previous dir");

        result.expect("atomic write should support relative paths");
        assert_eq!(read_result.expect("read relative file"), b"relative");
    }
}
