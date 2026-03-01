//! Small, reusable helpers for deriving client IP addresses from HTTP headers.

use http::HeaderMap;
use std::net::IpAddr;

/// Parse the first valid IP from an `X-Forwarded-For` header value.
#[must_use]
pub fn parse_x_forwarded_for(value: &str) -> Option<IpAddr> {
    value.split(',').find_map(|candidate| candidate.trim().parse::<IpAddr>().ok())
}

/// Extract client IP from standard proxy headers.
///
/// Checks `X-Forwarded-For` first, then falls back to `X-Real-IP`.
#[must_use]
pub fn extract_client_ip_from_headers(headers: &HeaderMap) -> Option<IpAddr> {
    if let Some(forwarded) = headers.get("x-forwarded-for")
        && let Ok(forwarded_str) = forwarded.to_str()
        && let Some(ip) = parse_x_forwarded_for(forwarded_str)
    {
        return Some(ip);
    }

    if let Some(real_ip) = headers.get("x-real-ip")
        && let Ok(real_ip_str) = real_ip.to_str()
        && let Ok(ip) = real_ip_str.parse::<IpAddr>()
    {
        return Some(ip);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_first_valid_forwarded_ip() {
        let parsed = parse_x_forwarded_for("unknown, 192.168.1.8, 10.0.0.1");
        assert_eq!(parsed, Some("192.168.1.8".parse().unwrap()));
    }

    #[test]
    fn extracts_forwarded_ip_before_real_ip() {
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "203.0.113.10, 10.0.0.1".parse().unwrap());
        headers.insert("x-real-ip", "198.51.100.7".parse().unwrap());

        let parsed = extract_client_ip_from_headers(&headers);
        assert_eq!(parsed, Some("203.0.113.10".parse().unwrap()));
    }

    #[test]
    fn falls_back_to_real_ip() {
        let mut headers = HeaderMap::new();
        headers.insert("x-real-ip", "2001:db8::42".parse().unwrap());

        let parsed = extract_client_ip_from_headers(&headers);
        assert_eq!(parsed, Some("2001:db8::42".parse().unwrap()));
    }
}
