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
}
