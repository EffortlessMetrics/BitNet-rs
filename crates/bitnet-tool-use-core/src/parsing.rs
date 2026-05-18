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
    let json_str = match format {
        ToolUseFormat::ChatMLTools | ToolUseFormat::HermesTools => {
            extract_between(text, "<tool_call>", "</tool_call>")
        }
        ToolUseFormat::Llama3Tools => extract_between(text, "<|python_tag|>", "<|eot_id|>"),
        ToolUseFormat::MistralTools => extract_between(text, "[TOOL_CALLS]", "[/TOOL_CALLS]"),
        ToolUseFormat::GenericJson => Some(text.trim().to_string()),
    };
    let json_str = json_str.as_deref().unwrap_or_else(|| text.trim());
    parse_call_json(json_str)
}

fn extract_between(text: &str, start_tag: &str, end_tag: &str) -> Option<String> {
    let start = text.find(start_tag).map(|i| i + start_tag.len())?;
    let end = text[start..].find(end_tag).map(|i| i + start)?;
    Some(text[start..end].trim().to_string())
}

fn parse_call_json(s: &str) -> Option<ToolCall> {
    let v: serde_json::Value = serde_json::from_str(s.trim()).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    let arguments = v.get("arguments").map_or_else(|| "{}".to_string(), ToString::to_string);
    Some(ToolCall { name, arguments })
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
}
