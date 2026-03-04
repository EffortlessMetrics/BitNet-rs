//! Reusable tool-calling contracts and parsing/detection helpers.
//!
//! This crate intentionally contains only pure data contracts and lightweight
//! parsing/format detection logic so it can be shared by higher-level crates.

use serde::{Deserialize, Serialize};

/// Describes a single parameter accepted by a tool.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolParameter {
    pub name: String,
    pub param_type: String,
    pub description: String,
    pub required: bool,
    pub default_value: Option<String>,
}

/// A tool that can be offered to the model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Vec<ToolParameter>,
}

/// A structured tool invocation produced (or consumed) by the model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    /// JSON-encoded arguments string.
    pub arguments: String,
}

/// The result returned after executing a tool call.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub content: String,
    pub is_error: bool,
}

/// Prompt format families that support tool / function calling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolUseFormat {
    /// ChatML with function-calling extensions (Qwen, Phi).
    ChatMLTools,
    /// LLaMA 3.1+ tool-calling format.
    Llama3Tools,
    /// Mistral tool-use format.
    MistralTools,
    /// Plain JSON function-calling envelope.
    GenericJson,
    /// Hermes / NousResearch tool-calling format.
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
    let json_str = json_str.as_deref().unwrap_or(text.trim());
    parse_call_json(json_str)
}

/// Auto-detect the tool-use format from a model name / path.
#[must_use]
pub fn detect_tool_format(model_name: &str) -> ToolUseFormat {
    let lower = model_name.to_lowercase();
    if lower.contains("qwen") || lower.contains("phi") {
        ToolUseFormat::ChatMLTools
    } else if lower.contains("llama-3.1")
        || lower.contains("llama-3.2")
        || lower.contains("llama-3.3")
        || lower.contains("llama3.1")
    {
        ToolUseFormat::Llama3Tools
    } else if lower.contains("mistral") || lower.contains("mixtral") {
        ToolUseFormat::MistralTools
    } else if lower.contains("hermes") || lower.contains("nous") {
        ToolUseFormat::HermesTools
    } else {
        ToolUseFormat::GenericJson
    }
}

fn extract_between(text: &str, start_tag: &str, end_tag: &str) -> Option<String> {
    let start = text.find(start_tag).map(|i| i + start_tag.len())?;
    let end = text[start..].find(end_tag).map(|i| i + start)?;
    Some(text[start..end].trim().to_string())
}

fn parse_call_json(s: &str) -> Option<ToolCall> {
    let v: serde_json::Value = serde_json::from_str(s.trim()).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    let arguments = v.get("arguments").map_or_else(|| "{}".to_string(), |a| a.to_string());
    Some(ToolCall { name, arguments })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_tool_call_valid() {
        let text =
            r#"<tool_call>{"name":"get_weather","arguments":{"location":"London"}}</tool_call>"#;
        let call = parse_tool_call(text, &ToolUseFormat::ChatMLTools).expect("valid tool call");
        assert_eq!(call.name, "get_weather");
        assert!(call.arguments.contains("London"));
    }

    #[test]
    fn parse_tool_call_malformed_returns_none() {
        assert!(
            parse_tool_call("<tool_call>{oops}</tool_call>", &ToolUseFormat::ChatMLTools).is_none()
        );
    }

    #[test]
    fn detect_tool_format_works_for_known_families() {
        assert_eq!(detect_tool_format("qwen2-7b"), ToolUseFormat::ChatMLTools);
        assert_eq!(detect_tool_format("llama-3.1-8b"), ToolUseFormat::Llama3Tools);
        assert_eq!(detect_tool_format("Mistral-7B"), ToolUseFormat::MistralTools);
        assert_eq!(detect_tool_format("hermes-2-pro"), ToolUseFormat::HermesTools);
        assert_eq!(detect_tool_format("unknown-model"), ToolUseFormat::GenericJson);
    }
}
