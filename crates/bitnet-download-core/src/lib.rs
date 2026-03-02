use thiserror::Error;
use std::{fs, path::Path};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DownloadValidationError {
    #[error("download truncated: got {downloaded} bytes, expected {expected} bytes")]
    Truncated { downloaded: u64, expected: u64 },
}

/// Returns true when download logic should operate in offline mode.
#[must_use]
pub fn offline_enabled(cli_offline: bool) -> bool {
    cli_offline || std::env::var("BITNET_OFFLINE").as_deref() == Ok("1")
}

/// Parses `Content-Range` total bytes from values like `bytes 0-0/1234`.
#[must_use]
pub fn parse_content_range_total(content_range: &str) -> Option<u64> {
    content_range.rsplit('/').next()?.parse::<u64>().ok()
}

/// Ensure downloaded bytes match expected total when available.
pub fn validate_downloaded_len(
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
pub fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, bytes)?;

    #[cfg(unix)]
    {
        if let Ok(f) = std::fs::File::open(&tmp) {
            f.sync_all()?;
        }
    }

    fs::rename(&tmp, path)?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn parses_content_range_total() {
        assert_eq!(parse_content_range_total("bytes 0-0/1234"), Some(1234));
        assert_eq!(parse_content_range_total("invalid"), None);
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
    fn atomic_write_creates_file() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("etag.txt");
        atomic_write(&path, b"abc").expect("atomic write should succeed");
        assert_eq!(std::fs::read(&path).expect("file should exist"), b"abc");
    }

    #[test]
    fn atomic_write_overwrites_file() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("etag.txt");
        atomic_write(&path, b"first").expect("first write should succeed");
        atomic_write(&path, b"second").expect("second write should succeed");
        assert_eq!(std::fs::read(&path).expect("file should exist"), b"second");
    }
}
