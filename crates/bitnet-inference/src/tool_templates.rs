//! Tool-use / function-calling prompt templates for SLM models.
//!
//! Provides formatting infrastructure for models that support tool calling
//! (Qwen, LLaMA 3.1+, Mistral, Hermes/NousResearch, and generic JSON).

use bitnet_tool_use_core::{ToolCall, ToolDefinition, ToolResult, ToolUseFormat};
pub use bitnet_tool_use_core::{detect_tool_format, parse_tool_call};

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/// Render a JSON-schema-style parameter block shared by several formats.
fn params_json(tools: &[ToolDefinition]) -> String {
    let mut out = String::from("[\n");
    for (i, tool) in tools.iter().enumerate() {
        out.push_str("  {\n");
        out.push_str(&format!("    \"name\": \"{}\",\n", tool.name));
        out.push_str(&format!("    \"description\": \"{}\",\n", tool.description));
        out.push_str("    \"parameters\": {\n");
        out.push_str("      \"type\": \"object\",\n");
        out.push_str("      \"properties\": {\n");
        for (j, p) in tool.parameters.iter().enumerate() {
            out.push_str(&format!("        \"{}\": {{", p.name));
            out.push_str(&format!(
                "\"type\": \"{}\", \"description\": \"{}\"",
                p.param_type, p.description
            ));
            if let Some(ref dv) = p.default_value {
                out.push_str(&format!(", \"default\": \"{}\"", dv));
            }
            out.push('}');
            if j + 1 < tool.parameters.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("      },\n");
        let required: Vec<&str> =
            tool.parameters.iter().filter(|p| p.required).map(|p| p.name.as_str()).collect();
        out.push_str(&format!("      \"required\": {:?}\n", required));
        out.push_str("    }\n");
        out.push_str("  }");
        if i + 1 < tools.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push(']');
    out
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Generate the system prompt that describes available tools for the given format.
pub fn format_tools_system_prompt(tools: &[ToolDefinition], format: &ToolUseFormat) -> String {
    if tools.is_empty() {
        return String::new();
    }
    let tools_json = params_json(tools);
    match format {
        ToolUseFormat::ChatMLTools => {
            format!(
                "<|im_start|>system\nYou are a helpful assistant with access to the following functions. \
                 Use them if required:\n{tools_json}<|im_end|>\n"
            )
        }
        ToolUseFormat::Llama3Tools => {
            format!(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
                 You have access to the following tools:\n{tools_json}\n\
                 When you need to call a tool, use <|python_tag|> followed by the JSON call.<|eot_id|>\n"
            )
        }
        ToolUseFormat::MistralTools => {
            format!("[AVAILABLE_TOOLS]{tools_json}[/AVAILABLE_TOOLS]\n")
        }
        ToolUseFormat::GenericJson => {
            format!(
                "You have access to the following tools:\n{tools_json}\n\
                 To call a tool, respond with a JSON object: \
                 {{\"name\": \"<tool>\", \"arguments\": {{...}}}}\n"
            )
        }
        ToolUseFormat::HermesTools => {
            format!(
                "<|im_start|>system\nYou are a function calling AI model. You may call one or more functions. \
                 Available tools:\n{tools_json}\n\
                 Respond with <tool_call> JSON </tool_call> when calling a tool.<|im_end|>\n"
            )
        }
    }
}

/// Format a tool call in the model-specific format.
pub fn format_tool_call(call: &ToolCall, format: &ToolUseFormat) -> String {
    match format {
        ToolUseFormat::ChatMLTools => {
            format!(
                "<|im_start|>assistant\n<tool_call>\n{{\"name\": \"{}\", \"arguments\": {}}}\n</tool_call><|im_end|>\n",
                call.name, call.arguments
            )
        }
        ToolUseFormat::Llama3Tools => {
            format!(
                "<|start_header_id|>assistant<|end_header_id|>\n\n\
                 <|python_tag|>{{\"name\": \"{}\", \"arguments\": {}}}<|eot_id|>\n",
                call.name, call.arguments
            )
        }
        ToolUseFormat::MistralTools => {
            format!(
                "[TOOL_CALLS]{{\"name\": \"{}\", \"arguments\": {}}}[/TOOL_CALLS]\n",
                call.name, call.arguments
            )
        }
        ToolUseFormat::GenericJson => {
            format!("{{\"name\": \"{}\", \"arguments\": {}}}\n", call.name, call.arguments)
        }
        ToolUseFormat::HermesTools => {
            format!(
                "<tool_call>\n{{\"name\": \"{}\", \"arguments\": {}}}\n</tool_call>\n",
                call.name, call.arguments
            )
        }
    }
}

/// Format a tool result / response in the model-specific format.
pub fn format_tool_result(result: &ToolResult, format: &ToolUseFormat) -> String {
    let status = if result.is_error { "error" } else { "success" };
    match format {
        ToolUseFormat::ChatMLTools => {
            format!(
                "<|im_start|>tool\n{{\"id\": \"{}\", \"status\": \"{}\", \"content\": {}}}<|im_end|>\n",
                result.tool_call_id,
                status,
                serde_json::to_string(&result.content)
                    .unwrap_or_else(|_| format!("\"{}\"", result.content)),
            )
        }
        ToolUseFormat::Llama3Tools => {
            format!(
                "<|start_header_id|>tool<|end_header_id|>\n\n\
                 {{\"id\": \"{}\", \"status\": \"{}\", \"output\": {}}}<|eot_id|>\n",
                result.tool_call_id,
                status,
                serde_json::to_string(&result.content)
                    .unwrap_or_else(|_| format!("\"{}\"", result.content)),
            )
        }
        ToolUseFormat::MistralTools => {
            format!(
                "[TOOL_RESULTS]{{\"id\": \"{}\", \"content\": {}}}[/TOOL_RESULTS]\n",
                result.tool_call_id,
                serde_json::to_string(&result.content)
                    .unwrap_or_else(|_| format!("\"{}\"", result.content)),
            )
        }
        ToolUseFormat::GenericJson => {
            format!(
                "{{\"tool_call_id\": \"{}\", \"status\": \"{}\", \"content\": {}}}\n",
                result.tool_call_id,
                status,
                serde_json::to_string(&result.content)
                    .unwrap_or_else(|_| format!("\"{}\"", result.content)),
            )
        }
        ToolUseFormat::HermesTools => {
            format!(
                "<tool_response>\n{{\"id\": \"{}\", \"result\": {}}}\n</tool_response>\n",
                result.tool_call_id,
                serde_json::to_string(&result.content)
                    .unwrap_or_else(|_| format!("\"{}\"", result.content)),
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use bitnet_tool_use_core::ToolParameter;

    use super::*;

    // -- helpers --

    fn sample_tool() -> ToolDefinition {
        ToolDefinition {
            name: "get_weather".into(),
            description: "Get current weather for a location".into(),
            parameters: vec![
                ToolParameter {
                    name: "location".into(),
                    param_type: "string".into(),
                    description: "City name".into(),
                    required: true,
                    default_value: None,
                },
                ToolParameter {
                    name: "units".into(),
                    param_type: "string".into(),
                    description: "Temperature units".into(),
                    required: false,
                    default_value: Some("celsius".into()),
                },
            ],
        }
    }

    fn sample_call() -> ToolCall {
        ToolCall { name: "get_weather".into(), arguments: r#"{"location": "London"}"#.into() }
    }

    fn sample_result() -> ToolResult {
        ToolResult {
            tool_call_id: "call_1".into(), content: "72°F, sunny".into(), is_error: false
        }
    }

    // 1. ToolDefinition construction
    #[test]
    fn test_tool_definition_construction() {
        let t = sample_tool();
        assert_eq!(t.name, "get_weather");
        assert_eq!(t.parameters.len(), 2);
    }

    // 2. ToolParameter defaults and types
    #[test]
    fn test_tool_parameter_defaults() {
        let t = sample_tool();
        assert!(t.parameters[0].required);
        assert_eq!(t.parameters[0].default_value, None);
        assert!(!t.parameters[1].required);
        assert_eq!(t.parameters[1].default_value, Some("celsius".into()));
        assert_eq!(t.parameters[1].param_type, "string");
    }

    // 3. ToolParameter all types
    #[test]
    fn test_tool_parameter_types() {
        for ty in &["string", "integer", "boolean", "array", "object"] {
            let p = ToolParameter {
                name: "p".into(),
                param_type: (*ty).to_string(),
                description: "d".into(),
                required: false,
                default_value: None,
            };
            assert_eq!(p.param_type, *ty);
        }
    }

    // 4. ToolCall construction and JSON argument handling
    #[test]
    fn test_tool_call_construction() {
        let c = sample_call();
        assert_eq!(c.name, "get_weather");
        let v: serde_json::Value = serde_json::from_str(&c.arguments).unwrap();
        assert_eq!(v["location"], "London");
    }

    // 5. ToolResult success
    #[test]
    fn test_tool_result_success() {
        let r = sample_result();
        assert!(!r.is_error);
        assert_eq!(r.tool_call_id, "call_1");
    }

    // 6. ToolResult error variant
    #[test]
    fn test_tool_result_error() {
        let r = ToolResult { tool_call_id: "c2".into(), content: "timeout".into(), is_error: true };
        assert!(r.is_error);
        assert_eq!(r.content, "timeout");
    }

    // 7-11. format_tools_system_prompt for each format
    #[test]
    fn test_system_prompt_chatml() {
        let out = format_tools_system_prompt(&[sample_tool()], &ToolUseFormat::ChatMLTools);
        assert!(out.contains("<|im_start|>system"));
        assert!(out.contains("get_weather"));
        assert!(out.contains("<|im_end|>"));
    }

    #[test]
    fn test_system_prompt_llama3() {
        let out = format_tools_system_prompt(&[sample_tool()], &ToolUseFormat::Llama3Tools);
        assert!(out.contains("<|begin_of_text|>"));
        assert!(out.contains("get_weather"));
        assert!(out.contains("<|eot_id|>"));
    }

    #[test]
    fn test_system_prompt_mistral() {
        let out = format_tools_system_prompt(&[sample_tool()], &ToolUseFormat::MistralTools);
        assert!(out.contains("[AVAILABLE_TOOLS]"));
        assert!(out.contains("[/AVAILABLE_TOOLS]"));
        assert!(out.contains("get_weather"));
    }

    #[test]
    fn test_system_prompt_generic_json() {
        let out = format_tools_system_prompt(&[sample_tool()], &ToolUseFormat::GenericJson);
        assert!(out.contains("get_weather"));
        assert!(out.contains("\"name\""));
    }

    #[test]
    fn test_system_prompt_hermes() {
        let out = format_tools_system_prompt(&[sample_tool()], &ToolUseFormat::HermesTools);
        assert!(out.contains("<|im_start|>system"));
        assert!(out.contains("<tool_call>"));
        assert!(out.contains("get_weather"));
    }

    // 12-16. format_tool_call for each format
    #[test]
    fn test_tool_call_chatml() {
        let out = format_tool_call(&sample_call(), &ToolUseFormat::ChatMLTools);
        assert!(out.contains("<tool_call>"));
        assert!(out.contains("get_weather"));
        assert!(out.contains("</tool_call>"));
    }

    #[test]
    fn test_tool_call_llama3() {
        let out = format_tool_call(&sample_call(), &ToolUseFormat::Llama3Tools);
        assert!(out.contains("<|python_tag|>"));
        assert!(out.contains("get_weather"));
    }

    #[test]
    fn test_tool_call_mistral() {
        let out = format_tool_call(&sample_call(), &ToolUseFormat::MistralTools);
        assert!(out.contains("[TOOL_CALLS]"));
        assert!(out.contains("get_weather"));
    }

    #[test]
    fn test_tool_call_generic() {
        let out = format_tool_call(&sample_call(), &ToolUseFormat::GenericJson);
        assert!(out.contains("get_weather"));
        let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
        assert_eq!(v["name"], "get_weather");
    }

    #[test]
    fn test_tool_call_hermes() {
        let out = format_tool_call(&sample_call(), &ToolUseFormat::HermesTools);
        assert!(out.contains("<tool_call>"));
        assert!(out.contains("</tool_call>"));
    }

    // 17. format_tool_result round-trips
    #[test]
    fn test_tool_result_all_formats() {
        let r = sample_result();
        for fmt in &[
            ToolUseFormat::ChatMLTools,
            ToolUseFormat::Llama3Tools,
            ToolUseFormat::MistralTools,
            ToolUseFormat::GenericJson,
            ToolUseFormat::HermesTools,
        ] {
            let out = format_tool_result(&r, fmt);
            assert!(out.contains("call_1"), "format {:?} missing id", fmt);
        }
    }

    // 18. parse_tool_call: valid JSON
    #[test]
    fn test_parse_tool_call_valid() {
        let text =
            r#"<tool_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</tool_call>"#;
        let call = parse_tool_call(text, &ToolUseFormat::ChatMLTools).unwrap();
        assert_eq!(call.name, "get_weather");
        assert!(call.arguments.contains("Paris"));
    }

    // 19. parse_tool_call: malformed
    #[test]
    fn test_parse_tool_call_malformed() {
        let text = "<tool_call>not json</tool_call>";
        assert!(parse_tool_call(text, &ToolUseFormat::ChatMLTools).is_none());
    }

    // 20. parse_tool_call: empty
    #[test]
    fn test_parse_tool_call_empty() {
        assert!(parse_tool_call("", &ToolUseFormat::GenericJson).is_none());
    }

    // 21. detect_tool_format
    #[test]
    fn test_detect_tool_format() {
        assert_eq!(detect_tool_format("qwen2-7b"), ToolUseFormat::ChatMLTools);
        assert_eq!(detect_tool_format("Phi-4-mini"), ToolUseFormat::ChatMLTools);
        assert_eq!(detect_tool_format("llama-3.1-8b"), ToolUseFormat::Llama3Tools);
        assert_eq!(detect_tool_format("Mistral-7B"), ToolUseFormat::MistralTools);
        assert_eq!(detect_tool_format("mixtral-8x7b"), ToolUseFormat::MistralTools);
        assert_eq!(detect_tool_format("hermes-2-pro"), ToolUseFormat::HermesTools);
        assert_eq!(detect_tool_format("NousResearch-Llama"), ToolUseFormat::HermesTools);
        assert_eq!(detect_tool_format("unknown-model"), ToolUseFormat::GenericJson);
    }

    // 22. Edge case: no tools → empty prompt
    #[test]
    fn test_system_prompt_no_tools() {
        for fmt in &[
            ToolUseFormat::ChatMLTools,
            ToolUseFormat::Llama3Tools,
            ToolUseFormat::MistralTools,
            ToolUseFormat::GenericJson,
            ToolUseFormat::HermesTools,
        ] {
            assert!(format_tools_system_prompt(&[], fmt).is_empty());
        }
    }

    // 23. Edge case: empty arguments
    #[test]
    fn test_tool_call_empty_arguments() {
        let call = ToolCall { name: "ping".into(), arguments: "{}".into() };
        let out = format_tool_call(&call, &ToolUseFormat::GenericJson);
        let v: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
        assert_eq!(v["arguments"], serde_json::json!({}));
    }

    // 24. Edge case: special characters in tool names
    #[test]
    fn test_special_characters_in_name() {
        let call = ToolCall { name: "get-data_v2".into(), arguments: "{}".into() };
        let out = format_tool_call(&call, &ToolUseFormat::ChatMLTools);
        assert!(out.contains("get-data_v2"));
    }

    // 25. parse_tool_call Llama3 format
    #[test]
    fn test_parse_tool_call_llama3() {
        let text = r#"<|python_tag|>{"name": "search", "arguments": {"q": "rust"}}<|eot_id|>"#;
        let call = parse_tool_call(text, &ToolUseFormat::Llama3Tools).unwrap();
        assert_eq!(call.name, "search");
    }

    // 26. parse_tool_call Mistral format
    #[test]
    fn test_parse_tool_call_mistral() {
        let text = r#"[TOOL_CALLS]{"name": "calc", "arguments": {"x": 1}}[/TOOL_CALLS]"#;
        let call = parse_tool_call(text, &ToolUseFormat::MistralTools).unwrap();
        assert_eq!(call.name, "calc");
    }

    // 27. parse_tool_call GenericJson bare object
    #[test]
    fn test_parse_tool_call_generic_json() {
        let text = r#"{"name": "echo", "arguments": {"msg": "hi"}}"#;
        let call = parse_tool_call(text, &ToolUseFormat::GenericJson).unwrap();
        assert_eq!(call.name, "echo");
    }

    // 28. Serde round-trip for ToolDefinition
    #[test]
    fn test_serde_round_trip_tool_definition() {
        let t = sample_tool();
        let json = serde_json::to_string(&t).unwrap();
        let t2: ToolDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(t, t2);
    }
}
