//! Reusable client IP extraction helpers from common reverse-proxy headers.

use http::HeaderMap;
use std::net::IpAddr;

/// Parse the first client IP from an `X-Forwarded-For` header value.
///
/// Header values can contain a comma-separated chain of proxy hops. This
/// function returns the first parseable IP address in that chain.
#[must_use]
pub fn parse_x_forwarded_for(value: &str) -> Option<IpAddr> {
    value.split(',').find_map(|candidate| candidate.trim().parse::<IpAddr>().ok())
}

/// Extract client IP from HTTP headers.
///
/// Resolution order:
/// 1. `X-Forwarded-For` (first parseable hop)
/// 2. `X-Real-IP`
#[must_use]
pub fn extract_client_ip_from_headers(headers: &HeaderMap) -> Option<IpAddr> {
    if let Some(forwarded) = headers.get("x-forwarded-for")
        && let Ok(forwarded_str) = forwarded.to_str()
        && let Some(ip) = parse_x_forwarded_for(forwarded_str)
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
    use super::*;

    #[test]
    fn parses_first_hop_from_x_forwarded_for() {
        let ip = parse_x_forwarded_for("203.0.113.42, 10.0.0.1, 192.168.1.1");
        assert_eq!(ip, Some("203.0.113.42".parse().unwrap()));
    }

    #[test]
    fn skips_invalid_hops_and_finds_first_valid_ip() {
        let ip = parse_x_forwarded_for("unknown, 198.51.100.10");
        assert_eq!(ip, Some("198.51.100.10".parse().unwrap()));
    }

    #[test]
    fn returns_none_for_missing_headers() {
        let headers = HeaderMap::new();
        assert_eq!(extract_client_ip_from_headers(&headers), None);
    }

    #[test]
    fn falls_back_to_x_real_ip() {
        let mut headers = HeaderMap::new();
        headers.insert("x-real-ip", "198.51.100.7".parse().unwrap());

        let ip = extract_client_ip_from_headers(&headers);
        assert_eq!(ip, Some("198.51.100.7".parse().unwrap()));
    }
}
