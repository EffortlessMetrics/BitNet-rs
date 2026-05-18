//! Model-name heuristics for selecting a tool-use prompt format.

use crate::parsing::ToolUseFormat;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_tool_format_works_for_known_families() {
        assert_eq!(detect_tool_format("qwen2-7b"), ToolUseFormat::ChatMLTools);
        assert_eq!(detect_tool_format("llama-3.1-8b"), ToolUseFormat::Llama3Tools);
        assert_eq!(detect_tool_format("Mistral-7B"), ToolUseFormat::MistralTools);
        assert_eq!(detect_tool_format("hermes-2-pro"), ToolUseFormat::HermesTools);
        assert_eq!(detect_tool_format("unknown-model"), ToolUseFormat::GenericJson);
    }
}
