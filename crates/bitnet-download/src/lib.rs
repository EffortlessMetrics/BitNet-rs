pub use bitnet_download_core::{
    DownloadValidationError, offline_enabled, parse_content_range_total, validate_downloaded_len,
};
pub use bitnet_http_retry::exp_backoff_ms;
use bitnet_http_retry::retry_after_secs_at as parse_retry_after_secs_at;
use reqwest::header::{HeaderMap, RETRY_AFTER};
use std::time::SystemTime;
/// Parse Retry-After header (supports both seconds and HTTP-date), capping to 1 hour.
#[must_use]
pub fn retry_after_secs(headers: &HeaderMap) -> u64 {
    retry_after_secs_at(headers, SystemTime::now())
}

/// Same as [`retry_after_secs`] but allows injecting the current time for deterministic tests.
#[must_use]
pub fn retry_after_secs_at(headers: &HeaderMap, now: SystemTime) -> u64 {
    let retry_after = headers.get(RETRY_AFTER).and_then(|v| v.to_str().ok());
    parse_retry_after_secs_at(retry_after, now)
}

pub use bitnet_atomic_file_core::atomic_write;

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
}
