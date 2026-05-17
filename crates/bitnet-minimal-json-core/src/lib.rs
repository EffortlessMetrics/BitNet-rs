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
                continue;
            }
            let Some(separator) = Self::top_level_colon(part)? else {
                return Err("expected ':' after object key".to_string());
            };
            let key = Self::parse_key(&part[..separator])?;
            let val = Self::parse_value(&part[separator + 1..])?;
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
                current.push(ch);
                if escaped {
                    escaped = false;
                } else if ch == '\\' {
                    escaped = true;
                } else if ch == '"' {
                    in_string = false;
                }
                continue;
            }

            match ch {
                '"' => {
                    in_string = true;
                    current.push(ch);
                }
                '{' | '[' => {
                    depth += 1;
                    current.push(ch);
                }
                '}' | ']' => {
                    depth -= 1;
                    if depth < 0 {
                        return Err("unbalanced nested JSON value".to_string());
                    }
                    current.push(ch);
                }
                ',' if depth == 0 => {
                    parts.push(std::mem::take(&mut current));
                }
                _ => current.push(ch),
            }
        }
        if in_string {
            return Err("unterminated string".to_string());
        }
        if depth != 0 {
            return Err("unbalanced nested JSON value".to_string());
        }
        if !current.trim().is_empty() {
            parts.push(current);
        }
        Ok(parts)
    }

    fn top_level_colon(s: &str) -> Result<Option<usize>, String> {
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
                continue;
            }

            match ch {
                '"' => in_string = true,
                '{' | '[' => depth += 1,
                '}' | ']' => {
                    depth -= 1;
                    if depth < 0 {
                        return Err("unbalanced nested JSON value".to_string());
                    }
                }
                ':' if depth == 0 => return Ok(Some(idx)),
                _ => {}
            }
        }
        if in_string {
            return Err("unterminated string".to_string());
        }
        if depth != 0 {
            return Err("unbalanced nested JSON value".to_string());
        }
        Ok(None)
    }

    fn parse_key(raw: &str) -> Result<String, String> {
        let raw = raw.trim();
        if !raw.starts_with('"') || !raw.ends_with('"') || raw.len() < 2 {
            return Err("expected quoted object key".to_string());
        }
        Self::parse_json_string(raw)
    }

    fn parse_value(raw: &str) -> Result<String, String> {
        let raw = raw.trim();
        if raw.is_empty() {
            return Err("expected value after ':'".to_string());
        }
        if raw.starts_with('"') {
            if !raw.ends_with('"') || raw.len() < 2 {
                return Err("unterminated string".to_string());
            }
            Self::parse_json_string(raw)
        } else {
            Ok(raw.to_string())
        }
    }

    fn parse_json_string(raw: &str) -> Result<String, String> {
        let inner = raw
            .strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .ok_or_else(|| "expected JSON string".to_string())?;
        let mut decoded = String::new();
        let mut chars = inner.chars();
        while let Some(ch) = chars.next() {
            if ch != '\\' {
                decoded.push(ch);
                continue;
            }

            let escaped = chars.next().ok_or_else(|| "unterminated escape".to_string())?;
            match escaped {
                '"' => decoded.push('"'),
                '\\' => decoded.push('\\'),
                '/' => decoded.push('/'),
                'b' => decoded.push('\u{0008}'),
                'f' => decoded.push('\u{000c}'),
                'n' => decoded.push('\n'),
                'r' => decoded.push('\r'),
                't' => decoded.push('\t'),
                'u' => {
                    let mut digits = String::with_capacity(4);
                    for _ in 0..4 {
                        digits.push(
                            chars.next().ok_or_else(|| "incomplete unicode escape".to_string())?,
                        );
                    }
                    let code = u32::from_str_radix(&digits, 16)
                        .map_err(|_| "invalid unicode escape".to_string())?;
                    let ch = char::from_u32(code)
                        .ok_or_else(|| "invalid unicode scalar value".to_string())?;
                    decoded.push(ch);
                }
                _ => return Err("invalid escape".to_string()),
            }
        }
        Ok(decoded)
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
    fn parses_colons_and_commas_inside_strings() {
        let j = MinimalJson::parse(r#"{"message":"time: 12:30, ok","key:with:colon":"value"}"#)
            .unwrap();
        assert_eq!(j.get_str("message"), Some("time: 12:30, ok".to_string()));
        assert_eq!(j.get_str("key:with:colon"), Some("value".to_string()));
    }

    #[test]
    fn unescapes_quoted_string_keys_and_values() {
        let j = MinimalJson::parse(r##"{"quote\"key":"say \"hello\"","path":"C:\\tmp"}"##).unwrap();
        assert_eq!(j.get_str("quote\"key"), Some("say \"hello\"".to_string()));
        assert_eq!(j.get_str("path"), Some("C:\\tmp".to_string()));
    }

    #[test]
    fn rejects_malformed_object_fields() {
        assert!(MinimalJson::parse(r#"{"ok":1, "missing_colon"}"#).is_err());
        assert!(MinimalJson::parse(r#"{"unterminated":"value}"#).is_err());
        assert!(MinimalJson::parse(r#"{"arr":[1,2}"#).is_err());
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
}
