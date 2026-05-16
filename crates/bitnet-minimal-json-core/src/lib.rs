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

        for part in Self::split_top_level(inner) {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            if let Some((k, v)) = part.split_once(':') {
                let key = k.trim().trim_matches('"').to_string();
                let val = v.trim();
                let val = if val.starts_with('"') && val.ends_with('"') && val.len() >= 2 {
                    val[1..val.len() - 1].to_string()
                } else {
                    val.to_string()
                };
                fields.insert(key, val);
            }
        }
        Ok(Self { fields })
    }

    /// Split on commas that are not inside braces/brackets/quotes.
    fn split_top_level(s: &str) -> Vec<String> {
        let mut parts = Vec::new();
        let mut current = String::new();
        let mut depth = 0_i32;
        let mut in_string = false;
        let mut prev = '\0';
        for ch in s.chars() {
            if ch == '"' && prev != '\\' {
                in_string = !in_string;
            }
            if !in_string {
                match ch {
                    '{' | '[' => depth += 1,
                    '}' | ']' => depth -= 1,
                    ',' if depth == 0 => {
                        parts.push(std::mem::take(&mut current));
                        prev = ch;
                        continue;
                    }
                    _ => {}
                }
            }
            current.push(ch);
            prev = ch;
        }
        if !current.trim().is_empty() {
            parts.push(current);
        }
        parts
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
    fn parses_multiple_fields() {
        let j = MinimalJson::parse(r#"{"a":"x","b":1,"c":true}"#).unwrap();
        assert_eq!(j.get_str("a"), Some("x".to_string()));
        assert_eq!(j.get_u32("b"), Some(1));
        assert_eq!(j.get_bool("c"), Some(true));
    }

    #[test]
    fn parses_false_bool() {
        let j = MinimalJson::parse(r#"{"flag":false}"#).unwrap();
        assert_eq!(j.get_bool("flag"), Some(false));
    }

    #[test]
    fn bool_accessor_returns_none_for_non_bool_value() {
        let j = MinimalJson::parse(r#"{"flag":"yes"}"#).unwrap();
        assert_eq!(j.get_bool("flag"), None);
    }

    #[test]
    fn u32_accessor_returns_none_for_non_numeric() {
        let j = MinimalJson::parse(r#"{"n":"abc"}"#).unwrap();
        assert_eq!(j.get_u32("n"), None);
    }

    #[test]
    fn f32_accessor_returns_none_for_non_numeric() {
        let j = MinimalJson::parse(r#"{"f":"abc"}"#).unwrap();
        assert_eq!(j.get_f32("f"), None);
    }

    #[test]
    fn parses_empty_object() {
        let j = MinimalJson::parse("{}").unwrap();
        assert_eq!(j.get_str("missing"), None);
        assert_eq!(j.get_u32("missing"), None);
        assert_eq!(j.get_bool("missing"), None);
    }

    #[test]
    fn tolerates_outer_whitespace() {
        let j = MinimalJson::parse(r#"   { "a" : "b" }   "#).unwrap();
        assert_eq!(j.get_str("a"), Some("b".to_string()));
    }

    #[test]
    fn nested_object_value_is_preserved_with_commas() {
        let j = MinimalJson::parse(r#"{"obj":{"a":1,"b":2,"c":3}}"#).unwrap();
        // The whole brace-balanced value (commas included) is kept as a single
        // string — not split across keys.
        assert_eq!(j.get_str("obj"), Some(r#"{"a":1,"b":2,"c":3}"#.to_string()));
    }

    #[test]
    fn rejects_inputs_without_closing_brace() {
        assert!(MinimalJson::parse("{").is_err());
        assert!(MinimalJson::parse(r#"{"a":1"#).is_err());
    }

    #[test]
    fn string_value_with_comma_does_not_split_field() {
        let j = MinimalJson::parse(r#"{"msg":"a, b, c","n":1}"#).unwrap();
        assert_eq!(j.get_str("msg"), Some("a, b, c".to_string()));
        assert_eq!(j.get_u32("n"), Some(1));
    }
}
