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
    value.split(',').find_map(parse_ip)
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
    fn x_forwarded_for_skips_invalid_first_hop() {
        let ip = parse_x_forwarded_for("unknown, 203.0.113.7, 10.0.0.4");
        assert_eq!(ip, Some(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 7))));
    }

    #[test]
    fn x_forwarded_for_returns_none_when_all_hops_invalid() {
        assert_eq!(parse_x_forwarded_for("unknown, invalid, ???"), None);
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

    #[test]
    fn both_headers_none_returns_none() {
        assert_eq!(extract_client_ip(None, None), None);
    }

    #[test]
    fn empty_strings_return_none() {
        assert_eq!(extract_client_ip(Some(""), None), None);
        assert_eq!(extract_client_ip(None, Some("")), None);
        assert_eq!(extract_client_ip(Some(""), Some("")), None);
    }

    #[test]
    fn falls_through_to_x_real_ip_when_forwarded_is_all_invalid() {
        let ip = extract_client_ip(Some("unknown, ???"), Some("198.51.100.7"));
        assert_eq!(ip, Some(IpAddr::V4(Ipv4Addr::new(198, 51, 100, 7))));
    }

    #[test]
    fn parse_x_forwarded_for_single_value() {
        let ip = parse_x_forwarded_for("203.0.113.1");
        assert_eq!(ip, Some(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 1))));
    }

    #[test]
    fn parse_x_forwarded_for_empty_input_returns_none() {
        assert_eq!(parse_x_forwarded_for(""), None);
    }

    #[test]
    fn parse_x_forwarded_for_handles_commas_only() {
        // ",,," yields empty hops that all fail parsing.
        assert_eq!(parse_x_forwarded_for(",,,"), None);
    }

    #[test]
    fn parse_x_forwarded_for_ipv6_first_hop() {
        let ip = parse_x_forwarded_for("2001:db8::abcd, 10.0.0.1");
        assert_eq!(ip, Some(IpAddr::V6(Ipv6Addr::from(0x20010db800000000000000000000abcd_u128))));
    }

    #[test]
    fn parse_ip_handles_internal_whitespace_as_invalid() {
        // Only surrounding whitespace is trimmed; embedded whitespace is invalid.
        assert_eq!(parse_ip("203.0 .113.1"), None);
    }

    #[test]
    fn parse_ip_strips_leading_and_trailing_whitespace() {
        let ip = parse_ip("\t  203.0.113.5\n");
        assert_eq!(ip, Some(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 5))));
    }

    #[test]
    fn parse_ip_rejects_empty_and_whitespace_only() {
        assert_eq!(parse_ip(""), None);
        assert_eq!(parse_ip("   "), None);
    }
}
