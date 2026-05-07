//! SRP helpers for HTTP Authorization Bearer-token parsing.

/// Returns the bearer token when the header is in the `Bearer <token>` format.
#[must_use]
pub fn bearer_token(header_value: &str) -> Option<&str> {
    let header = trim_http_auth_whitespace(header_value);
    if header.len() <= "bearer".len() {
        return None;
    }

    let (scheme, rest) = header.split_at("bearer".len());
    if !scheme.eq_ignore_ascii_case("bearer") {
        return None;
    }

    let first_token_byte = rest.as_bytes().first()?;
    if !is_http_auth_whitespace(*first_token_byte) {
        return None;
    }

    let token = trim_http_auth_whitespace(rest);
    if token.is_empty() || token.bytes().any(|byte| byte.is_ascii_whitespace()) {
        return None;
    }

    Some(token)
}

/// Strips a `Bearer ` prefix when present, otherwise returns the original value.
#[must_use]
pub fn strip_bearer_prefix(value: &str) -> &str {
    bearer_token(value).unwrap_or(value)
}

fn trim_http_auth_whitespace(value: &str) -> &str {
    value.trim_matches(|c| matches!(c, ' ' | '\t'))
}

const fn is_http_auth_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t')
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
    fn bearer_token_accepts_case_insensitive_scheme() {
        assert_eq!(bearer_token("bearer abc123"), Some("abc123"));
        assert_eq!(bearer_token("BEARER abc123"), Some("abc123"));
    }

    #[test]
    fn bearer_token_accepts_surrounding_and_separator_tabs() {
        assert_eq!(bearer_token("   Bearer abc123   "), Some("abc123"));
        assert_eq!(bearer_token("\tBearer\tabc123\t"), Some("abc123"));
    }

    #[test]
    fn bearer_token_rejects_missing_separator() {
        assert_eq!(bearer_token("Bearerabc123"), None);
    }

    #[test]
    fn bearer_token_rejects_multipart_tokens() {
        assert_eq!(bearer_token("Bearer abc 123"), None);
        assert_eq!(bearer_token("Bearer abc\t123"), None);
    }

    #[test]
    fn bearer_token_rejects_empty_token() {
        assert_eq!(bearer_token("Bearer "), None);
    }

    #[test]
    fn bearer_token_rejects_line_wrapped_values() {
        assert_eq!(bearer_token("\nBearer abc123"), None);
        assert_eq!(bearer_token("Bearer abc123\n"), None);
    }

    #[test]
    fn strip_bearer_prefix_leaves_unprefixed_value() {
        assert_eq!(strip_bearer_prefix("abc123"), "abc123");
    }
}
