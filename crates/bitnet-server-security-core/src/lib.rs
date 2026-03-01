//! Shared security parsing primitives for server transports.

use http::HeaderMap;
use std::net::IpAddr;

/// Extract a client IP from common proxy headers.
///
/// Priority:
/// 1. `x-forwarded-for` (first comma-separated entry)
/// 2. `x-real-ip`
#[must_use]
pub fn extract_client_ip_from_headers(headers: &HeaderMap) -> Option<IpAddr> {
    // Try X-Forwarded-For header first (for reverse proxies)
    if let Some(forwarded) = headers.get("x-forwarded-for")
        && let Ok(forwarded_str) = forwarded.to_str()
        && let Some(first_ip) = forwarded_str.split(',').next()
        && let Ok(ip) = first_ip.trim().parse::<IpAddr>()
    {
        return Some(ip);
    }

    // Try X-Real-IP header
    if let Some(real_ip) = headers.get("x-real-ip")
        && let Ok(ip_str) = real_ip.to_str()
        && let Ok(ip) = ip_str.parse::<IpAddr>()
    {
        return Some(ip);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn takes_first_x_forwarded_for_entry() {
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "203.0.113.7, 198.51.100.2".parse().unwrap());
        headers.insert("x-real-ip", "198.51.100.2".parse().unwrap());

        let parsed = extract_client_ip_from_headers(&headers);
        assert_eq!(parsed, Some("203.0.113.7".parse().unwrap()));
    }

    #[test]
    fn falls_back_to_x_real_ip() {
        let mut headers = HeaderMap::new();
        headers.insert("x-real-ip", "203.0.113.10".parse().unwrap());

        let parsed = extract_client_ip_from_headers(&headers);
        assert_eq!(parsed, Some("203.0.113.10".parse().unwrap()));
    }

    #[test]
    fn invalid_values_return_none() {
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "invalid-ip".parse().unwrap());

        let parsed = extract_client_ip_from_headers(&headers);
        assert_eq!(parsed, None);
    }
}
