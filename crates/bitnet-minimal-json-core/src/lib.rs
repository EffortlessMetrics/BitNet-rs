//! Extremely small JSON field extractor for flat-object payloads.
//!
//! This parser is intentionally limited and dependency-free:
//! - Top-level JSON object only
//! - Field values are stored as strings
//! - Commas inside nested arrays/objects/strings are preserved

use std::collections::HashMap;

/// Extremely basic JSON field extractor. Handles flat objects only.
#[derive(Debug)]
pub struct MinimalJson {
    fields: HashMap<String, String>,
}

impl MinimalJson {
    /// Parse a JSON object string into a [`MinimalJson`] map.
    pub fn parse(text: &str) -> Result<Self, String> {
        let trimmed = text.trim();
        if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
            return Err("expected JSON object".to_string());
        }
        let inner = &trimmed[1..trimmed.len() - 1];
        let mut fields = HashMap::new();

        for part in Self::split_top_level(inner)? {
            let part = part.trim();
            if part.is_empty() {
                return Err("empty field in JSON object".to_string());
            }
            let colon = Self::find_top_level_colon(part)
                .ok_or_else(|| format!("expected ':' in JSON field: {part}"))?;
            let (k, v) = part.split_at(colon);
            let key = Self::parse_quoted_string(k.trim())?;
            let val = v[1..].trim();
            if val.is_empty() {
                return Err(format!("missing value for JSON field: {key}"));
            }
            let val = if val.starts_with('"') {
                Self::parse_quoted_string(val)?
            } else {
                val.to_string()
            };
            fields.insert(key, val);
        }
        Ok(Self { fields })
    }

    /// Split on commas that are not inside braces/brackets/quotes.
    fn split_top_level(s: &str) -> Result<Vec<String>, String> {
        let mut parts = Vec::new();
        let mut current = String::new();
        let mut depth = 0_i32;
        let mut in_string = false;
        let mut escaped = false;
        for ch in s.chars() {
            if in_string {
                if escaped {
                    escaped = false;
                } else if ch == '\\' {
                    escaped = true;
                } else if ch == '"' {
                    in_string = false;
                }
            } else {
                match ch {
                    '"' => in_string = true,
                    '{' | '[' => depth += 1,
                    '}' | ']' => {
                        depth -= 1;
                        if depth < 0 {
                            return Err("unbalanced closing delimiter in JSON object".to_string());
                        }
                    }
                    ',' if depth == 0 => {
                        parts.push(std::mem::take(&mut current));
                        continue;
                    }
                    _ => {}
                }
            }
            current.push(ch);
        }
        if in_string {
            return Err("unterminated string in JSON object".to_string());
        }
        if depth != 0 {
            return Err("unbalanced nested delimiter in JSON object".to_string());
        }
        if !current.trim().is_empty() {
            parts.push(current);
        } else if s.trim_end().ends_with(',') {
            return Err("trailing comma in JSON object".to_string());
        }
        Ok(parts)
    }

    fn find_top_level_colon(s: &str) -> Option<usize> {
        let mut depth = 0_i32;
        let mut in_string = false;
        let mut escaped = false;
        for (idx, ch) in s.char_indices() {
            if in_string {
                if escaped {
                    escaped = false;
                } else if ch == '\\' {
                    escaped = true;
                } else if ch == '"' {
                    in_string = false;
                }
            } else {
                match ch {
                    '"' => in_string = true,
                    '{' | '[' => depth += 1,
                    '}' | ']' => depth -= 1,
                    ':' if depth == 0 => return Some(idx),
                    _ => {}
                }
            }
        }
        None
    }

    fn parse_quoted_string(s: &str) -> Result<String, String> {
        if !s.starts_with('"') || !s.ends_with('"') || s.len() < 2 {
            return Err(format!("expected quoted JSON string: {s}"));
        }
        let inner = &s[1..s.len() - 1];
        let mut output = String::new();
        let mut chars = inner.chars();
        while let Some(ch) = chars.next() {
            if ch != '\\' {
                if ch == '"' {
                    return Err("unescaped quote in JSON string".to_string());
                }
                if ch.is_control() {
                    return Err("unescaped control character in JSON string".to_string());
                }
                output.push(ch);
                continue;
            }

            let escaped =
                chars.next().ok_or_else(|| "unterminated JSON string escape".to_string())?;
            match escaped {
                '"' => output.push('"'),
                '\\' => output.push('\\'),
                '/' => output.push('/'),
                'b' => output.push('\u{0008}'),
                'f' => output.push('\u{000c}'),
                'n' => output.push('\n'),
                'r' => output.push('\r'),
                't' => output.push('\t'),
                'u' => {
                    let mut value = 0_u32;
                    for _ in 0..4 {
                        let hex =
                            chars.next().ok_or_else(|| "short JSON unicode escape".to_string())?;
                        value = value
                            .checked_mul(16)
                            .and_then(|v| hex.to_digit(16).map(|digit| v + digit))
                            .ok_or_else(|| "invalid JSON unicode escape".to_string())?;
                    }
                    let decoded = char::from_u32(value)
                        .ok_or_else(|| "invalid JSON unicode scalar".to_string())?;
                    output.push(decoded);
                }
                _ => return Err(format!("unsupported JSON string escape: \\{escaped}")),
            }
        }
        Ok(output)
    }

    #[must_use]
    pub fn get_str(&self, key: &str) -> Option<String> {
        self.fields.get(key).cloned()
    }

    #[must_use]
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.fields.get(key)?.parse().ok()
    }

    #[must_use]
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.fields.get(key)?.parse().ok()
    }

    #[must_use]
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.fields.get(key)?.as_str() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_string_fields() {
        let j = MinimalJson::parse(r#"{"key":"value"}"#).unwrap();
        assert_eq!(j.get_str("key"), Some("value".to_string()));
    }

    #[test]
    fn parses_u32_fields() {
        let j = MinimalJson::parse(r#"{"n":42}"#).unwrap();
        assert_eq!(j.get_u32("n"), Some(42));
    }

    #[test]
    fn parses_f32_fields() {
        let j = MinimalJson::parse(r#"{"f":0.7}"#).unwrap();
        assert_eq!(j.get_f32("f"), Some(0.7));
    }

    #[test]
    fn parses_bool_fields() {
        let j = MinimalJson::parse(r#"{"b":true}"#).unwrap();
        assert_eq!(j.get_bool("b"), Some(true));
    }

    #[test]
    fn preserves_nested_values_as_raw_strings() {
        let j = MinimalJson::parse(r#"{"obj":{"a":1},"arr":[1,2,3]}"#).unwrap();
        assert_eq!(j.get_str("obj"), Some("{\"a\":1}".to_string()));
        assert_eq!(j.get_str("arr"), Some("[1,2,3]".to_string()));
    }

    #[test]
    fn handles_missing_keys() {
        let j = MinimalJson::parse("{}").unwrap();
        assert_eq!(j.get_str("missing"), None);
    }

    #[test]
    fn rejects_non_object_json() {
        assert!(MinimalJson::parse("not json").is_err());
        assert!(MinimalJson::parse("[1,2]").is_err());
    }

    #[test]
    fn handles_escaped_quotes_and_commas_inside_strings() {
        let j = MinimalJson::parse(r#"{"message":"a \"quoted, comma\" value","next":2}"#).unwrap();

        assert_eq!(j.get_str("message"), Some("a \"quoted, comma\" value".to_string()));
        assert_eq!(j.get_u32("next"), Some(2));
    }

    #[test]
    fn handles_even_backslashes_before_string_quote_boundaries() {
        let j = MinimalJson::parse(r#"{"path":"C:\\","next":true}"#).unwrap();

        assert_eq!(j.get_str("path"), Some("C:\\".to_string()));
        assert_eq!(j.get_bool("next"), Some(true));
    }

    #[test]
    fn supports_escaped_keys_and_unicode_string_values() {
        let j = MinimalJson::parse(r#"{"spaced\"key":"line\n\u263a"}"#).unwrap();

        assert_eq!(j.get_str("spaced\"key"), Some("line\n☺".to_string()));
    }

    #[test]
    fn rejects_malformed_fields_instead_of_silently_dropping_them() {
        assert!(MinimalJson::parse(r#"{"ok":1,broken}"#).is_err());
        assert!(MinimalJson::parse(r#"{"missing":}"#).is_err());
        assert!(MinimalJson::parse(r#"{"unclosed":"value}"#).is_err());
        assert!(MinimalJson::parse(r#"{"trailing":1,}"#).is_err());
        assert!(MinimalJson::parse(r#"{"nested":[1,2}"#).is_err());
        assert!(MinimalJson::parse(r#"{"quote":"bad " quote"}"#).is_err());
    }
}
