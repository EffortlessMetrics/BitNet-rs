//! # Prompt Template System
//!
//! Provides chat and instruct format templates for common model families.
//! Ensures proper prompt formatting for optimal model behavior.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

/// Supported prompt template types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TemplateType {
    /// Raw text (no formatting)
    Raw,
    /// Simple Q&A instruct format
    Instruct,
    /// LLaMA-3 chat format with special tokens
    Llama3Chat,
}

impl std::str::FromStr for TemplateType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "raw" => Ok(Self::Raw),
            "instruct" => Ok(Self::Instruct),
            "llama3-chat" | "llama3_chat" => Ok(Self::Llama3Chat),
            _ => bail!("Unknown template type: {}. Supported: raw, instruct, llama3-chat", s),
        }
    }
}

impl TemplateType {
    /// Apply the template to a user prompt
    pub fn apply(&self, user_text: &str, system_prompt: Option<&str>) -> String {
        match self {
            Self::Raw => user_text.to_string(),
            Self::Instruct => Self::apply_instruct(user_text, system_prompt),
            Self::Llama3Chat => Self::apply_llama3_chat(user_text, system_prompt),
        }
    }

    /// Apply simple instruct template
    fn apply_instruct(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::new();

        if let Some(system) = system_prompt {
            result.push_str("System: ");
            result.push_str(system);
            result.push_str("\n\n");
        }

        result.push_str("Q: ");
        result.push_str(user_text);
        result.push_str("\nA:");

        result
    }

    /// Apply LLaMA-3 chat template with proper special tokens
    ///
    /// Format:
    /// ```text
    /// <|begin_of_text|>
    /// [<|start_header_id|>system<|end_header_id|>
    /// {system_prompt}<|eot_id|>]
    /// <|start_header_id|>user<|end_header_id|>
    /// {user_text}<|eot_id|>
    /// <|start_header_id|>assistant<|end_header_id|>
    /// ```
    fn apply_llama3_chat(user_text: &str, system_prompt: Option<&str>) -> String {
        let mut result = String::from("<|begin_of_text|>");

        // Add system prompt if provided
        if let Some(system) = system_prompt {
            result.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
            result.push_str(system);
            result.push_str("<|eot_id|>");
        }

        // Add user message
        result.push_str("<|start_header_id|>user<|end_header_id|>\n\n");
        result.push_str(user_text);
        result.push_str("<|eot_id|>");

        // Start assistant response
        result.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");

        result
    }

    /// Get the default stop sequences for this template
    pub fn default_stop_sequences(&self) -> Vec<String> {
        match self {
            Self::Raw => vec![],
            Self::Instruct => vec!["\n\nQ:".to_string(), "\n\nHuman:".to_string()],
            Self::Llama3Chat => vec!["<|eot_id|>".to_string(), "<|end_of_text|>".to_string()],
        }
    }

    /// Check if BOS should be added for this template
    /// LLaMA-3 chat includes its own BOS token in the template
    pub fn should_add_bos(&self) -> bool {
        match self {
            Self::Raw | Self::Instruct => true,
            Self::Llama3Chat => false, // Template includes <|begin_of_text|>
        }
    }
}

/// Prompt template builder with history support
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    template_type: TemplateType,
    system_prompt: Option<String>,
    conversation_history: Vec<(String, String)>,
}

impl PromptTemplate {
    /// Create a new prompt template
    pub fn new(template_type: TemplateType) -> Self {
        Self { template_type, system_prompt: None, conversation_history: Vec::new() }
    }

    /// Set system prompt
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Add a turn to conversation history
    pub fn add_turn(&mut self, user: impl Into<String>, assistant: impl Into<String>) {
        self.conversation_history.push((user.into(), assistant.into()));
    }

    /// Clear conversation history
    pub fn clear_history(&mut self) {
        self.conversation_history.clear();
    }

    /// Format a user message with full context
    pub fn format(&self, user_text: &str) -> String {
        // For now, just apply the template to the current message
        // Multi-turn history can be added later
        self.template_type.apply(user_text, self.system_prompt.as_deref())
    }

    /// Get default stop sequences for this template
    pub fn stop_sequences(&self) -> Vec<String> {
        self.template_type.default_stop_sequences()
    }

    /// Check if BOS should be added
    pub fn should_add_bos(&self) -> bool {
        self.template_type.should_add_bos()
    }

    /// Get template type
    pub fn template_type(&self) -> TemplateType {
        self.template_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_template() {
        let template = TemplateType::Raw;
        let result = template.apply("Hello, world!", None);
        assert_eq!(result, "Hello, world!");

        let result_with_system = template.apply("Hello, world!", Some("You are helpful"));
        assert_eq!(result_with_system, "Hello, world!");
    }

    #[test]
    fn test_instruct_template() {
        let template = TemplateType::Instruct;

        // Without system prompt
        let result = template.apply("What is 2+2?", None);
        assert_eq!(result, "Q: What is 2+2?\nA:");

        // With system prompt
        let result = template.apply("What is 2+2?", Some("You are a math tutor"));
        assert!(result.contains("System: You are a math tutor"));
        assert!(result.contains("Q: What is 2+2?"));
        assert!(result.ends_with("\nA:"));
    }

    #[test]
    fn test_llama3_chat_template() {
        let template = TemplateType::Llama3Chat;

        // Without system prompt
        let result = template.apply("Hello!", None);
        assert!(result.starts_with("<|begin_of_text|>"));
        assert!(result.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(result.contains("Hello!"));
        assert!(result.contains("<|eot_id|>"));
        assert!(result.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));

        // With system prompt
        let result = template.apply("Hello!", Some("You are helpful"));
        assert!(result.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(result.contains("You are helpful"));
    }

    #[test]
    fn test_template_from_str() {
        assert_eq!("raw".parse::<TemplateType>().unwrap(), TemplateType::Raw);
        assert_eq!("instruct".parse::<TemplateType>().unwrap(), TemplateType::Instruct);
        assert_eq!("llama3-chat".parse::<TemplateType>().unwrap(), TemplateType::Llama3Chat);
        assert_eq!("llama3_chat".parse::<TemplateType>().unwrap(), TemplateType::Llama3Chat);

        assert!("invalid".parse::<TemplateType>().is_err());
    }

    #[test]
    fn test_stop_sequences() {
        assert_eq!(TemplateType::Raw.default_stop_sequences(), Vec::<String>::new());
        assert!(!TemplateType::Instruct.default_stop_sequences().is_empty());
        assert!(!TemplateType::Llama3Chat.default_stop_sequences().is_empty());

        // Check llama3-chat has the expected stop tokens
        let llama3_stops = TemplateType::Llama3Chat.default_stop_sequences();
        assert!(llama3_stops.contains(&"<|eot_id|>".to_string()));
    }

    #[test]
    fn test_bos_control() {
        assert!(TemplateType::Raw.should_add_bos());
        assert!(TemplateType::Instruct.should_add_bos());
        assert!(!TemplateType::Llama3Chat.should_add_bos()); // Has its own BOS
    }

    #[test]
    fn test_prompt_template_builder() {
        let template = PromptTemplate::new(TemplateType::Instruct)
            .with_system_prompt("You are a helpful assistant");

        let formatted = template.format("What is Rust?");
        assert!(formatted.contains("System: You are a helpful assistant"));
        assert!(formatted.contains("Q: What is Rust?"));

        assert!(!template.stop_sequences().is_empty());
        assert!(template.should_add_bos());
    }

    #[test]
    fn test_conversation_history() {
        let mut template = PromptTemplate::new(TemplateType::Raw);

        template.add_turn("Hello", "Hi there!");
        template.add_turn("How are you?", "I'm doing well!");

        assert_eq!(template.conversation_history.len(), 2);

        template.clear_history();
        assert_eq!(template.conversation_history.len(), 0);
    }
}
