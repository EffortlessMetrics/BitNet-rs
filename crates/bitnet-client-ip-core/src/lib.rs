//! SRP helpers for determining the client IP from HTTP headers.

use http::HeaderMap;
use std::net::IpAddr;

/// Extract the client IP from common proxy headers.
///
/// Priority:
/// 1. `x-forwarded-for` (first IP)
/// 2. `x-real-ip`
pub fn extract_client_ip_from_headers(headers: &HeaderMap) -> Option<IpAddr> {
    if let Some(forwarded) = headers.get("x-forwarded-for")
        && let Ok(forwarded_str) = forwarded.to_str()
        && let Some(first_ip) = forwarded_str.split(',').next()
        && let Ok(ip) = first_ip.trim().parse::<IpAddr>()
    {
        return Some(ip);
    }

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
    use super::extract_client_ip_from_headers;
    use http::HeaderMap;
    use std::net::IpAddr;

    #[test]
    fn prefers_x_forwarded_for_first_address() {
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "203.0.113.42, 10.0.0.1".parse().unwrap());
        headers.insert("x-real-ip", "198.51.100.7".parse().unwrap());

        let ip = extract_client_ip_from_headers(&headers);

        assert_eq!(ip, Some(IpAddr::from([203, 0, 113, 42])));
    }

    #[test]
    fn falls_back_to_x_real_ip() {
        let mut headers = HeaderMap::new();
        headers.insert("x-real-ip", "198.51.100.7".parse().unwrap());

        let ip = extract_client_ip_from_headers(&headers);

        assert_eq!(ip, Some(IpAddr::from([198, 51, 100, 7])));
    }

    #[test]
    fn returns_none_for_missing_or_invalid_headers() {
        let mut headers = HeaderMap::new();
        assert_eq!(extract_client_ip_from_headers(&headers), None);

        headers.insert("x-forwarded-for", "not-an-ip".parse().unwrap());
        headers.insert("x-real-ip", "also-not-an-ip".parse().unwrap());

        assert_eq!(extract_client_ip_from_headers(&headers), None);
    }
}
