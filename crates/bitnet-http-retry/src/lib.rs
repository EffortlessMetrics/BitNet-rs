use std::time::SystemTime;

/// Safe exponential backoff helper with deterministic jitter.
#[must_use]
pub fn exp_backoff_ms(attempt: u32) -> u64 {
    let shift = attempt.saturating_sub(1).min(20);
    let base = (200u64).saturating_mul(1u64 << shift).min(10_000);
    let jitter = (u64::from(attempt) * 37) % 200;
    base.saturating_add(jitter)
}

/// Parse an HTTP `Retry-After` value (seconds or HTTP-date), capping to 1 hour.
///
/// If the value is missing or invalid, defaults to 5 seconds.
#[must_use]
pub fn retry_after_secs(value: Option<&str>) -> u64 {
    retry_after_secs_at(value, SystemTime::now())
}

/// Same as [`retry_after_secs`] but allows injecting the current time for deterministic tests.
#[must_use]
pub fn retry_after_secs_at(value: Option<&str>, now: SystemTime) -> u64 {
    let Some(raw) = value else {
        return 5;
    };

    if let Ok(s) = raw.parse::<u64>() {
        return s.min(3600);
    }

    httpdate::parse_http_date(raw)
        .ok()
        .and_then(|when| when.duration_since(now).ok())
        .map_or(5, |d| d.as_secs().clamp(1, 3600))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    #[test]
    fn retry_after_seconds() {
        assert_eq!(retry_after_secs(Some("10")), 10);
    }

    #[test]
    fn retry_after_http_date() {
        let now = SystemTime::now();
        let future = now + Duration::from_secs(5);
        let wait = retry_after_secs_at(Some(&httpdate::fmt_http_date(future)), now);
        assert!((4..=6).contains(&wait));
    }

    #[test]
    fn retry_after_past_date_falls_back() {
        let now = SystemTime::now();
        let past = now - Duration::from_secs(10);
        assert_eq!(retry_after_secs_at(Some(&httpdate::fmt_http_date(past)), now), 5);
    }

    #[test]
    fn retry_after_missing_or_invalid_falls_back() {
        assert_eq!(retry_after_secs(None), 5);
        assert_eq!(retry_after_secs(Some("not-a-date")), 5);
    }

    #[test]
    fn exp_backoff_matches_contract() {
        assert_eq!(exp_backoff_ms(1), 237);
        assert_eq!(exp_backoff_ms(2), 474);
        assert_eq!(exp_backoff_ms(3), 911);
        assert_eq!(exp_backoff_ms(10), 10_170);
    }
}
