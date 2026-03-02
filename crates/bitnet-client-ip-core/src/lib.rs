use http::HeaderMap;
use std::net::IpAddr;

/// Extract the client IP from common proxy headers.
///
/// Resolution order:
/// 1. `x-forwarded-for` (first comma-separated IP)
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
    fn extracts_first_forwarded_ip() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-forwarded-for",
            "203.0.113.2, 198.51.100.10".parse().expect("valid header value"),
        );

        let extracted = extract_client_ip_from_headers(&headers);
        assert_eq!(extracted, Some("203.0.113.2".parse().expect("valid IP")));
    }

    #[test]
    fn falls_back_to_x_real_ip() {
        let mut headers = HeaderMap::new();
        headers.insert("x-real-ip", "198.51.100.7".parse().expect("valid header value"));

        let extracted = extract_client_ip_from_headers(&headers);
        assert_eq!(extracted, Some("198.51.100.7".parse().expect("valid IP")));
    }

    #[test]
    fn returns_none_for_invalid_headers() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-forwarded-for",
            "not-an-ip, still-not-an-ip".parse().expect("valid header value"),
        );

        let extracted = extract_client_ip_from_headers(&headers);
        assert_eq!(extracted, None);
    }
}
