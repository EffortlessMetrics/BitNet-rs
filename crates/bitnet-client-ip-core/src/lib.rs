//! Reusable client IP extraction helpers.

use std::net::IpAddr;

/// Extract the client IP address from canonical forwarding headers.
///
/// `x_forwarded_for` takes precedence over `x_real_ip`.
#[must_use]
pub fn extract_client_ip(x_forwarded_for: Option<&str>, x_real_ip: Option<&str>) -> Option<IpAddr> {
    x_forwarded_for.and_then(parse_x_forwarded_for).or_else(|| x_real_ip.and_then(parse_ip))
}

/// Parse an `X-Forwarded-For` header value and return the first valid IP if present.
#[must_use]
pub fn parse_x_forwarded_for(value: &str) -> Option<IpAddr> {
    value.split(',').next().and_then(parse_ip)
}

/// Parse a single IP value, trimming surrounding whitespace.
#[must_use]
pub fn parse_ip(value: &str) -> Option<IpAddr> {
    value.trim().parse::<IpAddr>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

    #[test]
    fn x_forwarded_for_takes_precedence() {
        let ip = extract_client_ip(Some("203.0.113.42, 10.0.0.1"), Some("192.0.2.15"));
        assert_eq!(ip, Some(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 42))));
    }

    #[test]
    fn fallback_to_x_real_ip() {
        let ip = extract_client_ip(None, Some("192.0.2.15"));
        assert_eq!(ip, Some(IpAddr::V4(Ipv4Addr::new(192, 0, 2, 15))));
    }

    #[test]
    fn parses_ipv6_and_whitespace() {
        let ip = extract_client_ip(Some(" 2001:db8::1 "), None);
        assert_eq!(ip, Some(IpAddr::V6(Ipv6Addr::from(0x20010db8000000000000000000000001_u128))));
    }

    #[test]
    fn invalid_values_return_none() {
        assert_eq!(extract_client_ip(Some("not-an-ip"), None), None);
        assert_eq!(extract_client_ip(None, Some("bad")), None);
    }
}
