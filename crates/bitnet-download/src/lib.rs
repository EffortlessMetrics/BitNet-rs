use reqwest::header::{HeaderMap, RETRY_AFTER};
use std::{fs, path::Path, time::SystemTime};
use thiserror::Error;

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

/// Safe exponential backoff helper with deterministic jitter.
#[must_use]
pub fn exp_backoff_ms(attempt: u32) -> u64 {
    let shift = attempt.saturating_sub(1).min(20);
    let base = (200u64).saturating_mul(1u64 << shift).min(10_000);
    let jitter = (attempt as u64 * 37) % 200;
    base.saturating_add(jitter)
}

/// Parse Retry-After header (supports both seconds and HTTP-date), capping to 1 hour.
#[must_use]
pub fn retry_after_secs(headers: &HeaderMap) -> u64 {
    retry_after_secs_at(headers, SystemTime::now())
}

/// Same as [`retry_after_secs`] but allows injecting the current time for deterministic tests.
#[must_use]
pub fn retry_after_secs_at(headers: &HeaderMap, now: SystemTime) -> u64 {
    let raw = match headers.get(RETRY_AFTER).and_then(|v| v.to_str().ok()) {
        Some(s) => s,
        None => return 5,
    };

    if let Ok(s) = raw.parse::<u64>() {
        return s.min(3600);
    }

    httpdate::parse_http_date(raw)
        .ok()
        .and_then(|when| when.duration_since(now).ok())
        .map(|d| d.as_secs().clamp(1, 3600))
        .unwrap_or(5)
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
    use reqwest::header::RETRY_AFTER;
    use std::time::{Duration, SystemTime};

    #[test]
    fn retry_after_seconds() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, "10".parse().expect("valid retry-after seconds"));
        assert_eq!(retry_after_secs(&headers), 10);
    }

    #[test]
    fn retry_after_http_date() {
        let now = SystemTime::now();
        let future = now + Duration::from_secs(5);
        let mut headers = HeaderMap::new();
        headers.insert(
            RETRY_AFTER,
            httpdate::fmt_http_date(future).parse().expect("valid retry-after date"),
        );

        let wait = retry_after_secs_at(&headers, now);
        assert!((4..=6).contains(&wait));
    }

    #[test]
    fn retry_after_past_date_falls_back() {
        let now = SystemTime::now();
        let past = now - Duration::from_secs(10);
        let mut headers = HeaderMap::new();
        headers.insert(
            RETRY_AFTER,
            httpdate::fmt_http_date(past).parse().expect("valid retry-after date"),
        );

        assert_eq!(retry_after_secs_at(&headers, now), 5);
    }

    #[test]
    fn exp_backoff_matches_contract() {
        assert_eq!(exp_backoff_ms(1), 237);
        assert_eq!(exp_backoff_ms(2), 474);
        assert_eq!(exp_backoff_ms(3), 911);
        assert_eq!(exp_backoff_ms(10), 10_170);
    }

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
}
