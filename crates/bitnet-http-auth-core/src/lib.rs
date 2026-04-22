//! SRP helpers for HTTP Authorization Bearer-token parsing.

/// Returns the bearer token when the header is in the `Bearer <token>` format.
#[must_use]
pub fn bearer_token(header_value: &str) -> Option<&str> {
    let (scheme, token_part) = header_value.split_once(|c: char| c.is_ascii_whitespace())?;
    if !scheme.eq_ignore_ascii_case("bearer") {
        return None;
    }

    let token = token_part.trim_start_matches(|c: char| c.is_ascii_whitespace());
    if token.is_empty() || token.contains(|c: char| c.is_ascii_whitespace()) {
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
    fn bearer_token_is_case_insensitive() {
        assert_eq!(bearer_token("bearer abc123"), Some("abc123"));
        assert_eq!(bearer_token("BEARER abc123"), Some("abc123"));
    }

    #[test]
    fn bearer_token_accepts_extra_whitespace_between_scheme_and_token() {
        assert_eq!(bearer_token("Bearer    abc123"), Some("abc123"));
        assert_eq!(bearer_token("Bearer\tabc123"), Some("abc123"));
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
    fn bearer_token_rejects_tokens_with_whitespace() {
        assert_eq!(bearer_token("Bearer abc 123"), None);
    }

    #[test]
    fn strip_bearer_prefix_leaves_unprefixed_value() {
        assert_eq!(strip_bearer_prefix("abc123"), "abc123");
    }
}
