//! Extremely small JSON field extractor for flat-object payloads.
//!
//! This parser is intentionally limited and dependency-free:
//! - Top-level JSON object only
//! - Field values are stored as strings
//! - Commas inside nested arrays/objects/strings are preserved

use std::collections::HashMap;

mod json_string;

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

        if inner.trim().is_empty() {
            return Ok(Self { fields });
        }

        for part in Self::split_top_level(inner)? {
            let part = part.trim();
            if part.is_empty() {
                return Err("empty field in JSON object".to_string());
            }

            let (k, v) = Self::split_key_value(part)?;
            let key = json_string::parse_json_string(k.trim())?;
            let value = Self::parse_value(v.trim())?;
            fields.insert(key, value);
        }
        Ok(Self { fields })
    }

    /// Split on commas that are not inside braces/brackets/quotes.
    fn split_top_level(s: &str) -> Result<Vec<&str>, String> {
        let mut parts = Vec::new();
        let mut stack = Vec::new();
        let mut in_string = false;
        let mut part_start = 0;

        for (index, ch) in s.char_indices() {
            if ch == '"' && !Self::quote_is_escaped_at(s, index) {
                in_string = !in_string;
            }

            if !in_string {
                match ch {
                    '{' => stack.push('}'),
                    '[' => stack.push(']'),
                    '}' | ']' => {
                        if stack.pop() != Some(ch) {
                            return Err("unbalanced nested JSON value".to_string());
                        }
                    }
                    ',' if stack.is_empty() => {
                        parts.push(&s[part_start..index]);
                        part_start = index + ch.len_utf8();
                    }
                    _ => {}
                }
            }
        }

        if in_string {
            return Err("unterminated string in JSON object".to_string());
        }
        if !stack.is_empty() {
            return Err("unbalanced nested JSON value".to_string());
        }
        if part_start < s.len() || !s.trim().is_empty() {
            parts.push(&s[part_start..]);
        }

        Ok(parts)
    }

    fn split_key_value(part: &str) -> Result<(&str, &str), String> {
        let mut stack = Vec::new();
        let mut in_string = false;

        for (index, ch) in part.char_indices() {
            if ch == '"' && !Self::quote_is_escaped_at(part, index) {
                in_string = !in_string;
            }

            if !in_string {
                match ch {
                    '{' => stack.push('}'),
                    '[' => stack.push(']'),
                    '}' | ']' => {
                        if stack.pop() != Some(ch) {
                            return Err("unbalanced nested JSON value".to_string());
                        }
                    }
                    ':' if stack.is_empty() => return Ok((&part[..index], &part[index + 1..])),
                    _ => {}
                }
            }
        }

        Err("expected ':' between JSON object key and value".to_string())
    }

    fn parse_value(value: &str) -> Result<String, String> {
        if value.is_empty() {
            return Err("expected JSON value".to_string());
        }

        if value.starts_with('"') {
            json_string::parse_json_string(value)
        } else {
            Ok(value.to_string())
        }
    }

    fn quote_is_escaped_at(s: &str, quote_index: usize) -> bool {
        s[..quote_index].bytes().rev().take_while(|byte| *byte == b'\\').count() % 2 == 1
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

    #[test]
    fn preserves_commas_and_colons_inside_strings() -> Result<(), String> {
        let j = MinimalJson::parse(r#"{"message":"ready: yes, continue","tail":"ok"}"#)?;
        assert_eq!(j.get_str("message"), Some("ready: yes, continue".to_string()));
        assert_eq!(j.get_str("tail"), Some("ok".to_string()));
        Ok(())
    }

    #[test]
    fn handles_escaped_quotes_and_even_backslashes_before_quotes() -> Result<(), String> {
        let j =
            MinimalJson::parse(r#"{"quote":"say \"hi\", then C:\\tools","after":"still parsed"}"#)?;
        assert_eq!(j.get_str("quote"), Some("say \"hi\", then C:\\tools".to_string()));
        assert_eq!(j.get_str("after"), Some("still parsed".to_string()));
        Ok(())
    }

    #[test]
    fn decodes_common_string_escapes_in_keys_and_values() -> Result<(), String> {
        let j = MinimalJson::parse(r#"{"line\nkey":"one\ntwo\tthree"}"#)?;
        assert_eq!(j.get_str("line\nkey"), Some("one\ntwo\tthree".to_string()));
        Ok(())
    }

    #[test]
    fn decodes_unicode_string_escapes() -> Result<(), String> {
        let j = MinimalJson::parse(r#"{"snowman":"\u2603","face":"\uD83D\uDE00"}"#)?;
        assert_eq!(j.get_str("snowman"), Some("☃".to_string()));
        assert_eq!(j.get_str("face"), Some("😀".to_string()));
        Ok(())
    }

    #[test]
    fn rejects_invalid_json_strings() {
        assert!(MinimalJson::parse("{\"key\":\"bad\nvalue\"}").is_err());
        assert!(MinimalJson::parse(r#"{"key":"bad "quote""}"#).is_err());
        assert!(MinimalJson::parse(r#"{"key":"\uD83D"}"#).is_err());
        assert!(MinimalJson::parse(r#"{"key":"\uDE00"}"#).is_err());
        assert!(MinimalJson::parse(r#"{"key":"\uZZZZ"}"#).is_err());
    }

    #[test]
    fn rejects_malformed_fields_instead_of_silently_dropping_them() {
        assert!(MinimalJson::parse(r#"{"ok":1,"missing_colon"}"#).is_err());
        assert!(MinimalJson::parse(r#"{"ok":1,}"#).is_err());
        assert!(MinimalJson::parse(r#"{"ok":}"#).is_err());
    }

    #[test]
    fn rejects_unbalanced_or_unterminated_nested_values() {
        assert!(MinimalJson::parse(r#"{"arr":[1,2}"#).is_err());
        assert!(MinimalJson::parse(r#"{"obj":{"a":1] }"#).is_err());
        assert!(MinimalJson::parse(r#"{"text":"unterminated}"#).is_err());
    }

    #[test]
    fn rejects_unquoted_keys_and_invalid_string_escapes() {
        assert!(MinimalJson::parse(r#"{key:"value"}"#).is_err());
        assert!(MinimalJson::parse(r#"{"key":"bad\xescape"}"#).is_err());
    }
}
