//! Fast Q&A Template Formatting Regression Tests
//!
//! Tests feature spec: docs/explanation/cli-ux-improvements-spec.md#template-formatting
//! Architecture: docs/reference/prompt-template-architecture.md
//!
//! This test suite provides fast, lightweight regression tests for Q&A template
//! formatting without requiring model loading. These tests catch regressions in
//! template behavior and ensure correct formatting for different question types.
//!
//! **Performance Target**: < 1 second total execution
//! **TDD Approach**: Tests compile successfully and validate existing implementation
//!
//! # Specification References
//! - Template formatting: cli-ux-improvements-spec.md#AC5-template-formatting
//! - Stop sequences: cli-ux-improvements-spec.md#AC6-stop-sequences
//! - Template behavior: prompt-template-architecture.md#template-types
use anyhow::Result;
use bitnet_inference::prompt_template::TemplateType;
#[cfg(test)]
mod qa_formatting_fast {
    use super::*;
    /// Tests feature spec: cli-ux-improvements-spec.md#AC5-raw-template
    /// Verify that Raw template preserves input exactly
    #[test]
    fn test_raw_template_preserves_math_question() -> Result<()> {
        let template = TemplateType::Raw;
        let input = "What is 2+2?";
        let output = template.apply(input, None);
        assert_eq!(
            output, input,
            "Raw template should preserve input exactly without modification"
        );
        Ok(())
    }
    /// Tests feature spec: cli-ux-improvements-spec.md#AC5-raw-template-system
    /// Verify that Raw template ignores system prompts
    #[test]
    fn test_raw_template_ignores_system_prompt() -> Result<()> {
        let template = TemplateType::Raw;
        let input = "2+2=";
        let output = template.apply(input, Some("You are a calculator"));
        assert_eq!(
            output, input,
            "Raw template should ignore system prompt and preserve input only"
        );
        Ok(())
    }
    /// Tests feature spec: cli-ux-improvements-spec.md#AC5-instruct-template
    /// Verify that Instruct template adds Q&A formatting
    #[test]
    fn test_instruct_template_adds_qa_formatting() -> Result<()> {
        let template = TemplateType::Instruct;
        let input = "What is 2+2?";
        let output = template.apply(input, None);
        assert!(output.starts_with("Q: "), "Instruct template should start with 'Q: '");
        assert!(output.contains("What is 2+2?"), "Should preserve input text");
        assert!(output.ends_with("\nA:"), "Instruct template should end with '\\nA:'");
        assert_eq!(output, "Q: What is 2+2?\nA:", "Instruct template should format as Q/A");
        Ok(())
    }
    /// Tests feature spec: cli-ux-improvements-spec.md#AC5-instruct-template-system
    /// Verify that Instruct template includes system prompt correctly
    #[test]
    fn test_instruct_template_includes_system_prompt() -> Result<()> {
        let template = TemplateType::Instruct;
        let input = "Calculate 2+2";
        let system = "You are a helpful math assistant";
        let output = template.apply(input, Some(system));
        assert!(
            output.starts_with("System: "),
            "Instruct template with system prompt should start with 'System: '"
        );
        assert!(
            output.contains("You are a helpful math assistant"),
            "System prompt content should be present"
        );
        assert!(output.contains("Q: Calculate 2+2"), "User input should follow system prompt");
        assert!(output.ends_with("\nA:"), "Should end with A: ready for response");
        let expected = "System: You are a helpful math assistant\n\nQ: Calculate 2+2\nA:";
        assert_eq!(output, expected, "Full instruct template format with system prompt");
        Ok(())
    }
    /// Tests feature spec: cli-ux-improvements-spec.md#AC5-llama3-template
    /// Verify that LLaMA-3 template adds special tokens correctly
    #[test]
    fn test_llama3_chat_template_special_tokens() -> Result<()> {
        let template = TemplateType::Llama3Chat;
        let input = "What is 2+2?";
        let output = template.apply(input, None);
        assert!(
            output.starts_with("<|begin_of_text|>"),
            "LLaMA-3 template should start with <|begin_of_text|>"
        );
        assert!(
            output.contains("<|start_header_id|>user<|end_header_id|>"),
            "Should contain user role header"
        );
        assert!(output.contains("What is 2+2?"), "Should preserve input text");
        assert!(
            output.contains("<|eot_id|>"),
            "Should contain <|eot_id|> token after user message"
        );
        assert!(
            output.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"),
            "Should end with assistant header ready for generation"
        );
        Ok(())
    }
    /// Tests feature spec: cli-ux-improvements-spec.md#AC5-llama3-template-system
    /// Verify that LLaMA-3 template includes system prompt with special tokens
    #[test]
    fn test_llama3_chat_template_with_system_prompt() -> Result<()> {
        let template = TemplateType::Llama3Chat;
        let input = "Calculate 2+2";
        let system = "You are a helpful math assistant";
        let output = template.apply(input, Some(system));
        assert!(output.starts_with("<|begin_of_text|>"), "Must start with begin_of_text");
        assert!(
            output.contains("<|start_header_id|>system<|end_header_id|>"),
            "Should contain system role header"
        );
        assert!(
            output.contains("You are a helpful math assistant"),
            "System prompt content should be present"
        );
        let system_section = output.split("<|start_header_id|>user<").next().unwrap();
        assert!(
            system_section.contains("<|eot_id|>"),
            "System message should be terminated with <|eot_id|>"
        );
        assert!(output.contains("Calculate 2+2"), "User message should be present");
        assert!(
            output.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"),
            "Should end with assistant header"
        );
        Ok(())
    }
    /// Tests feature spec: cli-ux-improvements-spec.md#AC6-stop-sequences
    /// Verify that Raw template has no stop sequences
    #[test]
    fn test_raw_template_no_stop_sequences() {
        let template = TemplateType::Raw;
        let stops = template.default_stop_sequences();
        assert!(stops.is_empty(), "Raw template should have no default stop sequences");
    }
    /// Tests feature spec: cli-ux-improvements-spec.md#AC6-stop-sequences
    /// Verify that Instruct template has Q&A stop sequences
    #[test]
    fn test_instruct_template_stop_sequences() {
        let template = TemplateType::Instruct;
        let stops = template.default_stop_sequences();
        assert!(!stops.is_empty(), "Instruct template should have stop sequences");
        assert!(
            stops.contains(&"\n\nQ:".to_string()),
            "Instruct should stop on '\\n\\nQ:' to prevent generating new questions"
        );
        assert!(
            stops.contains(&"\n\nHuman:".to_string()),
            "Instruct should stop on '\\n\\nHuman:' as alternative question marker"
        );
    }
    /// Tests feature spec: cli-ux-improvements-spec.md#AC6-stop-sequences
    /// Verify that LLaMA-3 template has EOT stop sequences
    #[test]
    fn test_llama3_chat_template_stop_sequences() {
        let template = TemplateType::Llama3Chat;
        let stops = template.default_stop_sequences();
        assert!(!stops.is_empty(), "LLaMA-3 template should have stop sequences");
        assert!(
            stops.contains(&"<|eot_id|>".to_string()),
            "LLaMA-3 should stop on <|eot_id|> token"
        );
        assert!(
            stops.contains(&"<|end_of_text|>".to_string()),
            "LLaMA-3 should stop on <|end_of_text|> token"
        );
    }
    /// Tests feature spec: cli-ux-improvements-spec.md#AC7-bos-control
    /// Verify BOS token control for each template type
    #[test]
    fn test_bos_token_control() {
        assert!(
            TemplateType::Raw.should_add_bos(),
            "Raw template should add BOS token during encoding"
        );
        assert!(
            TemplateType::Instruct.should_add_bos(),
            "Instruct template should add BOS token during encoding"
        );
        assert!(
            !TemplateType::Llama3Chat.should_add_bos(),
            "LLaMA-3 template should NOT add BOS (already embedded as <|begin_of_text|>)"
        );
    }
    /// Tests feature spec: cli-ux-improvements-spec.md#AC8-special-token-parsing
    /// Verify special token parsing control for each template type
    #[test]
    fn test_special_token_parsing_control() {
        assert!(!TemplateType::Raw.parse_special(), "Raw template should NOT parse special tokens");
        assert!(
            !TemplateType::Instruct.parse_special(),
            "Instruct template should NOT parse special tokens"
        );
        assert!(
            TemplateType::Llama3Chat.parse_special(),
            "LLaMA-3 template SHOULD parse special tokens for proper tokenization"
        );
    }
    /// Regression test: Verify math completion prompt formatting
    /// Tests feature spec: cli-ux-improvements-spec.md#math-completion
    #[test]
    fn test_math_completion_prompt_raw() -> Result<()> {
        let template = TemplateType::Raw;
        let input = "2+2=";
        let output = template.apply(input, None);
        assert_eq!(output, "2+2=", "Raw template should preserve completion prompt exactly");
        Ok(())
    }
    /// Regression test: Verify math completion with Instruct template
    /// Tests feature spec: cli-ux-improvements-spec.md#math-completion
    #[test]
    fn test_math_completion_prompt_instruct() -> Result<()> {
        let template = TemplateType::Instruct;
        let input = "2+2=";
        let output = template.apply(input, None);
        assert_eq!(output, "Q: 2+2=\nA:", "Instruct template should format completion as Q/A");
        Ok(())
    }
    /// Regression test: Verify complex question formatting
    /// Tests feature spec: cli-ux-improvements-spec.md#complex-questions
    #[test]
    fn test_complex_question_formatting() -> Result<()> {
        let template = TemplateType::Instruct;
        let input = "What is the capital of France? Please provide historical context.";
        let system = "You are a knowledgeable geography and history tutor.";
        let output = template.apply(input, Some(system));
        assert!(output.contains("System: You are a knowledgeable geography and history tutor."));
        assert!(
            output.contains("Q: What is the capital of France? Please provide historical context.")
        );
        assert!(output.ends_with("\nA:"));
        Ok(())
    }
    /// Regression test: Verify empty input handling
    /// Tests feature spec: cli-ux-improvements-spec.md#edge-cases
    #[test]
    fn test_empty_input_handling() -> Result<()> {
        let template_raw = TemplateType::Raw;
        let template_instruct = TemplateType::Instruct;
        let empty_input = "";
        let output_raw = template_raw.apply(empty_input, None);
        assert_eq!(output_raw, "", "Raw template should preserve empty input");
        let output_instruct = template_instruct.apply(empty_input, None);
        assert_eq!(output_instruct, "Q: \nA:", "Instruct template should format empty input");
        Ok(())
    }
    /// Regression test: Verify special characters in input
    /// Tests feature spec: cli-ux-improvements-spec.md#special-characters
    #[test]
    fn test_special_characters_in_input() -> Result<()> {
        let template = TemplateType::Instruct;
        let input = "What is \"2+2\" & <x>?";
        let output = template.apply(input, None);
        assert!(
            output.contains("What is \"2+2\" & <x>?"),
            "Special characters should be preserved in template"
        );
        assert!(output.starts_with("Q: "), "Template formatting should be intact");
        assert!(output.ends_with("\nA:"), "Template formatting should be intact");
        Ok(())
    }
    /// Regression test: Verify newlines in input
    /// Tests feature spec: cli-ux-improvements-spec.md#multiline-input
    #[test]
    fn test_multiline_input_handling() -> Result<()> {
        let template = TemplateType::Instruct;
        let input = "Question 1: What is 2+2?\nQuestion 2: What is 3+3?";
        let output = template.apply(input, None);
        assert!(output.contains("Question 1: What is 2+2?\nQuestion 2: What is 3+3?"));
        assert!(output.starts_with("Q: "));
        assert!(output.ends_with("\nA:"));
        Ok(())
    }
}
#[cfg(test)]
mod qa_template_parsing_fast {
    use super::*;
    use std::str::FromStr;
    /// Tests feature spec: cli-ux-improvements-spec.md#template-parsing
    /// Verify template type parsing from string
    #[test]
    fn test_parse_template_types() -> Result<()> {
        assert_eq!(TemplateType::from_str("raw")?, TemplateType::Raw);
        assert_eq!(TemplateType::from_str("instruct")?, TemplateType::Instruct);
        assert_eq!(TemplateType::from_str("llama3-chat")?, TemplateType::Llama3Chat);
        assert_eq!(TemplateType::from_str("llama3_chat")?, TemplateType::Llama3Chat);
        assert_eq!(TemplateType::from_str("RAW")?, TemplateType::Raw);
        assert_eq!(TemplateType::from_str("Instruct")?, TemplateType::Instruct);
        assert_eq!(TemplateType::from_str("LLAMA3-CHAT")?, TemplateType::Llama3Chat);
        Ok(())
    }
    /// Tests feature spec: cli-ux-improvements-spec.md#template-parsing
    /// Verify invalid template type parsing errors
    #[test]
    fn test_parse_invalid_template_types() {
        assert!(TemplateType::from_str("invalid").is_err());
        assert!(TemplateType::from_str("").is_err());
        assert!(TemplateType::from_str("chat").is_err());
        assert!(TemplateType::from_str("llama").is_err());
    }
    /// Tests feature spec: cli-ux-improvements-spec.md#template-display
    /// Verify template type display formatting
    #[test]
    fn test_template_display() {
        assert_eq!(TemplateType::Raw.to_string(), "raw");
        assert_eq!(TemplateType::Instruct.to_string(), "instruct");
        assert_eq!(TemplateType::Llama3Chat.to_string(), "llama3-chat");
    }
}
#[cfg(test)]
mod qa_template_detection_fast {
    use super::*;
    /// Tests feature spec: cli-ux-improvements-spec.md#AC4-detection-llama3
    /// Verify LLaMA-3 detection from GGUF metadata
    #[test]
    fn test_detect_llama3_from_gguf_metadata() {
        let jinja = "<|start_header_id|>user<|end_header_id|>\n\n{{ user }}<|eot_id|>";
        let detected = TemplateType::detect(None, Some(jinja));
        assert_eq!(
            detected,
            TemplateType::Llama3Chat,
            "Should detect LLaMA-3 from special tokens in GGUF metadata"
        );
    }
    /// Tests feature spec: cli-ux-improvements-spec.md#AC4-detection-instruct
    /// Verify Instruct detection from GGUF metadata
    #[test]
    fn test_detect_instruct_from_gguf_metadata() {
        let jinja = "{% for message in messages %}User: {{ message.content }}{% endfor %}";
        let detected = TemplateType::detect(None, Some(jinja));
        assert_eq!(
            detected,
            TemplateType::Instruct,
            "Should detect Instruct from Jinja loop in GGUF metadata"
        );
    }
    /// Tests feature spec: cli-ux-improvements-spec.md#AC4-detection-tokenizer
    /// Verify detection from tokenizer family name
    #[test]
    fn test_detect_from_tokenizer_family() {
        assert_eq!(
            TemplateType::detect(Some("llama3"), None),
            TemplateType::Llama3Chat,
            "Should detect LLaMA-3 from 'llama3' family name"
        );
        assert_eq!(
            TemplateType::detect(Some("llama-3-instruct"), None),
            TemplateType::Llama3Chat,
            "Should detect LLaMA-3 from 'llama-3' family name"
        );
        assert_eq!(
            TemplateType::detect(Some("mistral-instruct"), None),
            TemplateType::Instruct,
            "Should detect Instruct from 'mistral' family name"
        );
        assert_eq!(
            TemplateType::detect(Some("phi-2-instruct"), None),
            TemplateType::Instruct,
            "Should detect Instruct from 'instruct' family name"
        );
    }
    /// Tests feature spec: cli-ux-improvements-spec.md#AC4-detection-fallback
    /// Verify fallback to Raw when no hints available
    #[test]
    fn test_detect_fallback_to_raw() {
        let detected = TemplateType::detect(None, None);
        assert_eq!(detected, TemplateType::Raw, "Should fall back to Raw when no hints available");
        let detected_unknown = TemplateType::detect(Some("custom-tokenizer"), None);
        assert_eq!(
            detected_unknown,
            TemplateType::Raw,
            "Should fall back to Raw for unknown tokenizer family"
        );
        let detected_unknown_gguf = TemplateType::detect(None, Some("custom template"));
        assert_eq!(
            detected_unknown_gguf,
            TemplateType::Raw,
            "Should fall back to Raw for unknown GGUF template"
        );
    }
    /// Tests feature spec: cli-ux-improvements-spec.md#AC4-priority-order
    /// Verify GGUF metadata takes priority over tokenizer hints
    #[test]
    fn test_detection_priority_gguf_over_tokenizer() {
        let jinja = "<|start_header_id|>user<|end_header_id|><|eot_id|>";
        let tokenizer_name = "mistral-instruct";
        let detected = TemplateType::detect(Some(tokenizer_name), Some(jinja));
        assert_eq!(
            detected,
            TemplateType::Llama3Chat,
            "GGUF metadata should take priority over tokenizer hints"
        );
    }
}
