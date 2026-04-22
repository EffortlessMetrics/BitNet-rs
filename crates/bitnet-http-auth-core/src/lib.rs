//! SRP helpers for HTTP Authorization Bearer-token parsing.

/// Returns the bearer token when the header is in the `Bearer <token>` format.
#[must_use]
pub fn bearer_token(header_value: &str) -> Option<&str> {
    let value = header_value.trim();
    let split_idx = value.find(|ch: char| ch.is_ascii_whitespace())?;
    let (scheme, remainder) = value.split_at(split_idx);
    if !scheme.eq_ignore_ascii_case("bearer") {
        return None;
    }

    let token = remainder.trim();
    if token.is_empty() || token.chars().any(|ch| ch.is_ascii_whitespace()) {
        return None;
    }

    Some(token)
}

/// Strips a `Bearer ` prefix when present, otherwise returns the original value.
#[must_use]
pub fn strip_bearer_prefix(value: &str) -> &str {
    bearer_token(value).unwrap_or(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bearer_token_parses_prefixed_value() {
        assert_eq!(bearer_token("Bearer abc123"), Some("abc123"));
    }

    #[test]
    fn bearer_token_rejects_other_schemes() {
        assert_eq!(bearer_token("Token abc123"), None);
    }

    #[test]
    fn bearer_token_rejects_empty_token() {
        assert_eq!(bearer_token("Bearer "), None);
    }

    #[test]
    fn bearer_token_accepts_case_insensitive_scheme() {
        assert_eq!(bearer_token("bearer abc123"), Some("abc123"));
        assert_eq!(bearer_token("BeArEr abc123"), Some("abc123"));
    }

    #[test]
    fn bearer_token_accepts_extra_whitespace_around_value() {
        assert_eq!(bearer_token("  Bearer\tabc123  "), Some("abc123"));
    }

    #[test]
    fn bearer_token_rejects_whitespace_in_token() {
        assert_eq!(bearer_token("Bearer abc 123"), None);
    }

    #[test]
    fn strip_bearer_prefix_leaves_unprefixed_value() {
        assert_eq!(strip_bearer_prefix("abc123"), "abc123");
    }
}
