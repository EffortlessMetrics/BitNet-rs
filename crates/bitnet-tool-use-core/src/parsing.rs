//! Format-specific extraction and JSON parsing for model-produced tool calls.

use serde::{Deserialize, Serialize};

use crate::contracts::ToolCall;

/// Prompt format families that support tool / function calling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolUseFormat {
    /// `ChatML` with function-calling extensions (Qwen, Phi).
    ChatMLTools,
    /// `LLaMA` 3.1+ tool-calling format.
    Llama3Tools,
    /// Mistral tool-use format.
    MistralTools,
    /// Plain JSON function-calling envelope.
    GenericJson,
    /// Hermes / `NousResearch` tool-calling format.
    HermesTools,
}

/// Try to parse a tool call from raw model output.
///
/// Looks for the JSON payload in format-specific delimiters, then falls back
/// to bare `{"name": …, "arguments": …}` extraction.
pub fn parse_tool_call(text: &str, format: &ToolUseFormat) -> Option<ToolCall> {
    let payload = payload_extraction::extract_payload(text, format);
    json_parsing::parse_call_json(payload)
}

mod payload_extraction {
    use super::ToolUseFormat;

    pub(super) fn extract_payload<'a>(text: &'a str, format: &ToolUseFormat) -> &'a str {
        let extracted = match format {
            ToolUseFormat::ChatMLTools | ToolUseFormat::HermesTools => {
                extract_between(text, "<tool_call>", "</tool_call>")
            }
            ToolUseFormat::Llama3Tools => extract_between(text, "<|python_tag|>", "<|eot_id|>"),
            ToolUseFormat::MistralTools => extract_between(text, "[TOOL_CALLS]", "[/TOOL_CALLS]"),
            ToolUseFormat::GenericJson => None,
        };
        extracted.unwrap_or_else(|| text.trim())
    }

    fn extract_between<'a>(text: &'a str, start_tag: &str, end_tag: &str) -> Option<&'a str> {
        let start = text.find(start_tag).map(|i| i + start_tag.len())?;
        let end = text[start..].find(end_tag).map(|i| i + start)?;
        Some(text[start..end].trim())
    }
}

mod json_parsing {
    use crate::contracts::ToolCall;

    pub(super) fn parse_call_json(s: &str) -> Option<ToolCall> {
        let v: serde_json::Value = serde_json::from_str(s.trim()).ok()?;
        let name = v.get("name")?.as_str()?.to_string();
        let arguments = v.get("arguments").map_or_else(|| "{}".to_string(), ToString::to_string);
        Some(ToolCall { name, arguments })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_tool_call_valid() -> Result<(), Box<dyn std::error::Error>> {
        let text =
            r#"<tool_call>{"name":"get_weather","arguments":{"location":"London"}}</tool_call>"#;
        let Some(call) = parse_tool_call(text, &ToolUseFormat::ChatMLTools) else {
            return Err("expected valid ChatML tool call".into());
        };
        assert_eq!(call.name, "get_weather");
        assert!(call.arguments.contains("London"));
        Ok(())
    }

    #[test]
    fn parse_tool_call_malformed_returns_none() {
        assert!(
            parse_tool_call("<tool_call>{oops}</tool_call>", &ToolUseFormat::ChatMLTools).is_none()
        );
    }

    #[test]
    fn parse_tool_call_supported_payload_boundaries() -> Result<(), Box<dyn std::error::Error>> {
        let cases = [
            (
                ToolUseFormat::ChatMLTools,
                r#"prefix <tool_call> { "name":"chatml","arguments":{"value":1}} </tool_call> suffix"#,
                "chatml",
            ),
            (
                ToolUseFormat::HermesTools,
                r#"<tool_call>{"name":"hermes","arguments":{"value":2}}</tool_call>"#,
                "hermes",
            ),
            (
                ToolUseFormat::Llama3Tools,
                r#"prefix <|python_tag|> {"name":"llama","arguments":{"value":3}} <|eot_id|> suffix"#,
                "llama",
            ),
            (
                ToolUseFormat::MistralTools,
                r#"[TOOL_CALLS]{"name":"mistral","arguments":{"value":4}}[/TOOL_CALLS]"#,
                "mistral",
            ),
            (
                ToolUseFormat::GenericJson,
                r#"  {"name":"generic","arguments":{"value":5}}  "#,
                "generic",
            ),
            (
                ToolUseFormat::ChatMLTools,
                r#"  {"name":"fallback","arguments":{"value":6}}  "#,
                "fallback",
            ),
        ];

        for (format, text, expected_name) in cases {
            let Some(call) = parse_tool_call(text, &format) else {
                return Err(format!("expected {expected_name} tool call").into());
            };
            assert_eq!(call.name, expected_name);
            assert!(call.arguments.contains("value"));
        }

        Ok(())
    }

    #[test]
    fn parse_tool_call_preserves_argument_serialization() -> Result<(), Box<dyn std::error::Error>>
    {
        let cases = [
            (r#"{"name":"missing"}"#, "{}"),
            (r#"{"name":"object","arguments":{"count":2}}"#, r#"{"count":2}"#),
            (r#"{"name":"string","arguments":"literal"}"#, r#""literal""#),
            (r#"{"name":"array","arguments":[1,2]}"#, "[1,2]"),
        ];

        for (text, expected_arguments) in cases {
            let Some(call) = parse_tool_call(text, &ToolUseFormat::GenericJson) else {
                return Err(format!("expected tool call for {text}").into());
            };
            assert_eq!(call.arguments, expected_arguments);
        }

        Ok(())
    }
}
