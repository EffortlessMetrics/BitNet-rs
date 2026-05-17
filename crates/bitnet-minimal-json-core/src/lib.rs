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
    fn parses_string_fields() -> Result<(), String> {
        let j = MinimalJson::parse(r#"{"key":"value"}"#)?;
        assert_eq!(j.get_str("key"), Some("value".to_string()));
        Ok(())
    }

    #[test]
    fn parses_u32_fields() -> Result<(), String> {
        let j = MinimalJson::parse(r#"{"n":42}"#)?;
        assert_eq!(j.get_u32("n"), Some(42));
        Ok(())
    }

    #[test]
    fn parses_f32_fields() -> Result<(), String> {
        let j = MinimalJson::parse(r#"{"f":0.7}"#)?;
        assert_eq!(j.get_f32("f"), Some(0.7));
        Ok(())
    }

    #[test]
    fn parses_bool_fields() -> Result<(), String> {
        let j = MinimalJson::parse(r#"{"b":true}"#)?;
        assert_eq!(j.get_bool("b"), Some(true));
        Ok(())
    }

    #[test]
    fn preserves_nested_values_as_raw_strings() -> Result<(), String> {
        let j = MinimalJson::parse(r#"{"obj":{"a":1},"arr":[1,2,3]}"#)?;
        assert_eq!(j.get_str("obj"), Some("{\"a\":1}".to_string()));
        assert_eq!(j.get_str("arr"), Some("[1,2,3]".to_string()));
        Ok(())
    }

    #[test]
    fn handles_missing_keys() -> Result<(), String> {
        let j = MinimalJson::parse("{}")?;
        assert_eq!(j.get_str("missing"), None);
        Ok(())
    }

    #[test]
    fn rejects_non_object_json() {
        assert!(MinimalJson::parse("not json").is_err());
        assert!(MinimalJson::parse("[1,2]").is_err());
    }

    #[test]
    fn rejects_empty_and_whitespace_only_input() {
        assert!(MinimalJson::parse("").is_err());
        assert!(MinimalJson::parse("   \n\t").is_err());
    }

    #[test]
    fn rejects_unbalanced_braces() {
        assert!(MinimalJson::parse("{\"k\":\"v\"").is_err());
        assert!(MinimalJson::parse("\"k\":\"v\"}").is_err());
    }

    #[test]
    fn accepts_surrounding_whitespace() -> Result<(), String> {
        let j = MinimalJson::parse("  \n {\"k\":\"v\"}\t ")?;
        assert_eq!(j.get_str("k"), Some("v".to_string()));
        Ok(())
    }

    #[test]
    fn parses_multiple_top_level_fields() -> Result<(), String> {
        let j = MinimalJson::parse(r#"{"a":"x","b":2,"c":true}"#)?;
        assert_eq!(j.get_str("a"), Some("x".to_string()));
        assert_eq!(j.get_u32("b"), Some(2));
        assert_eq!(j.get_bool("c"), Some(true));
        Ok(())
    }

    #[test]
    fn parses_false_bool() -> Result<(), String> {
        let j = MinimalJson::parse(r#"{"b":false}"#)?;
        assert_eq!(j.get_bool("b"), Some(false));
        Ok(())
    }

    #[test]
    fn typed_getters_return_none_for_missing_keys() -> Result<(), String> {
        let j = MinimalJson::parse("{}")?;
        assert_eq!(j.get_u32("none"), None);
        assert_eq!(j.get_f32("none"), None);
        assert_eq!(j.get_bool("none"), None);
        Ok(())
    }

    #[test]
    fn get_u32_returns_none_for_non_integer() -> Result<(), String> {
        let j = MinimalJson::parse(r#"{"k":"abc"}"#)?;
        assert_eq!(j.get_u32("k"), None);
        Ok(())
    }

    #[test]
    fn get_f32_returns_none_for_non_numeric() -> Result<(), String> {
        let j = MinimalJson::parse(r#"{"k":"not-a-number"}"#)?;
        assert_eq!(j.get_f32("k"), None);
        Ok(())
    }

    #[test]
    fn get_bool_returns_none_for_non_bool_value() -> Result<(), String> {
        let j = MinimalJson::parse(r#"{"flag":"yes"}"#)?;
        assert_eq!(j.get_bool("flag"), None);
        Ok(())
    }

    #[test]
    fn empty_object_yields_no_fields() -> Result<(), String> {
        let j = MinimalJson::parse("{}")?;
        assert_eq!(j.get_str("anything"), None);
        Ok(())
    }

    #[test]
    fn empty_object_with_internal_whitespace() -> Result<(), String> {
        let j = MinimalJson::parse("{   }")?;
        assert_eq!(j.get_str("anything"), None);
        Ok(())
    }

    #[test]
    fn preserves_commas_inside_strings() -> Result<(), String> {
        // A comma inside a quoted string value must not split the field.
        let j = MinimalJson::parse(r#"{"k":"a,b,c","n":3}"#)?;
        assert_eq!(j.get_str("k"), Some("a,b,c".to_string()));
        assert_eq!(j.get_u32("n"), Some(3));
        Ok(())
    }

    #[test]
    fn nested_array_values_keep_internal_commas() -> Result<(), String> {
        let j = MinimalJson::parse(r#"{"arr":[1,2,3,4],"tail":"end"}"#)?;
        assert_eq!(j.get_str("arr"), Some("[1,2,3,4]".to_string()));
        assert_eq!(j.get_str("tail"), Some("end".to_string()));
        Ok(())
    }
}
