//! SRP helpers for HTTP Authorization Bearer-token parsing.

/// Returns the bearer token when the header is in the `Bearer <token>` format.
#[must_use]
pub fn bearer_token(header_value: &str) -> Option<&str> {
    header_value.strip_prefix("Bearer ").filter(|token| !token.is_empty())
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
    fn strip_bearer_prefix_leaves_unprefixed_value() {
        assert_eq!(strip_bearer_prefix("abc123"), "abc123");
    }
}
