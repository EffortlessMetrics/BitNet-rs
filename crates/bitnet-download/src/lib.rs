use reqwest::header::{HeaderMap, RETRY_AFTER};
use std::{fs, path::Path, time::SystemTime};

pub use bitnet_download_core::{
    DownloadValidationError, exp_backoff_ms, parse_content_range_total, validate_downloaded_len,
};

/// Returns true when download logic should operate in offline mode.
#[must_use]
pub fn offline_enabled(cli_offline: bool) -> bool {
    cli_offline || std::env::var("BITNET_OFFLINE").as_deref() == Ok("1")
}

/// Parse Retry-After header (supports both seconds and HTTP-date), capping to 1 hour.
#[must_use]
pub fn retry_after_secs(headers: &HeaderMap) -> u64 {
    retry_after_secs_at(headers, SystemTime::now())
}

/// Same as [`retry_after_secs`] but allows injecting the current time for deterministic tests.
#[must_use]
pub fn retry_after_secs_at(headers: &HeaderMap, now: SystemTime) -> u64 {
    let raw = headers.get(RETRY_AFTER).and_then(|v| v.to_str().ok());
    bitnet_download_core::retry_after_secs_value(raw, now)
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
            httpdate::fmt_http_date(future)
                .parse()
                .expect("valid retry-after date"),
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
            httpdate::fmt_http_date(past)
                .parse()
                .expect("valid retry-after date"),
        );

        assert_eq!(retry_after_secs_at(&headers, now), 5);
    }
}
