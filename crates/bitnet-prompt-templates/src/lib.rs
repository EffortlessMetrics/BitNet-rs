//! # Prompt Template System
//!
//! Provides chat and instruct format templates for common model families.
//! Ensures proper prompt formatting for optimal model behavior.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

/// Role in a chat conversation
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl ChatRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        }
    }
}

/// A single turn in a chat conversation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatTurn {
    pub role: ChatRole,
    pub text: String,
}

impl ChatTurn {
    pub fn new(role: ChatRole, text: impl Into<String>) -> Self {
        Self { role, text: text.into() }
    }
}

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

impl std::fmt::Display for TemplateType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Raw => write!(f, "raw"),
            Self::Instruct => write!(f, "instruct"),
            Self::Llama3Chat => write!(f, "llama3-chat"),
        }
    }
}

impl TemplateType {
    /// Detect template type from GGUF metadata and tokenizer hints.
    ///
    /// Priority order:
    /// 1. GGUF chat_template metadata (if present)
    /// 2. Tokenizer family name heuristics
    /// 3. Fallback to Raw
    pub fn detect(tokenizer_name: Option<&str>, chat_template_jinja: Option<&str>) -> Self {
        // Priority 1: GGUF chat_template metadata
        if let Some(jinja) = chat_template_jinja {
            // LLaMA-3 signature
            if jinja.contains("<|start_header_id|>") && jinja.contains("<|eot_id|>") {
                return Self::Llama3Chat;
            }
            // Generic instruct template
            if jinja.contains("{% for message in messages %}") {
                return Self::Instruct;
            }
        }

        // Priority 2: Tokenizer family name heuristics
        if let Some(name) = tokenizer_name {
            let lower = name.to_ascii_lowercase();
            if lower.contains("llama3") || lower.contains("llama-3") {
                return Self::Llama3Chat;
            }
            if lower.contains("instruct") || lower.contains("mistral") {
                return Self::Instruct;
            }
        }

        // Priority 3: Fallback
        Self::Raw
    }

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

    /// Resolve stop sequences to token IDs using the provided tokenizer
    ///
    /// This method converts the template's default stop sequences (like "<|eot_id|>")
    /// to their corresponding token IDs for efficient stop detection during generation.
    ///
    /// Token ID-based stops are checked before string matching, making termination
    /// faster and more reliable for models with special stop tokens.
    ///
    /// # Arguments
    /// * `tokenizer` - The tokenizer to use for token ID resolution
    ///
    /// # Returns
    /// A vector of token IDs that should trigger generation stop.
    /// Returns empty if no stop sequences can be resolved or if the template has no stops.
    ///
    /// # Example
    /// ```ignore
    /// let template = TemplateType::Llama3Chat;
    /// let stop_ids = template.resolve_stop_token_ids(&tokenizer);
    /// // stop_ids might contain [128009] for <|eot_id|>
    /// ```
    pub fn resolve_stop_token_ids(&self, tokenizer: &dyn bitnet_tokenizers::Tokenizer) -> Vec<u32> {
        let stop_sequences = self.default_stop_sequences();
        let mut stop_ids = Vec::new();

        for seq in &stop_sequences {
            if let Some(id) = tokenizer.token_to_id(seq) {
                stop_ids.push(id);
            }
        }

        stop_ids
    }

    /// Check if BOS should be added for this template
    /// LLaMA-3 chat includes its own BOS token in the template
    pub fn should_add_bos(&self) -> bool {
        match self {
            Self::Raw | Self::Instruct => true,
            Self::Llama3Chat => false, // Template includes <|begin_of_text|>
        }
    }

    /// Check if special tokens should be parsed during encoding
    /// LLaMA-3 chat templates contain special tokens that need to be parsed
    pub fn parse_special(&self) -> bool {
        matches!(self, Self::Llama3Chat)
    }

    /// Render a chat history (system + turns) into a single prompt string.
    /// This method formats multi-turn conversations with proper role markers.
    pub fn render_chat(&self, history: &[ChatTurn], system: Option<&str>) -> Result<String> {
        use std::fmt::Write as _;
        let mut out = String::new();

        match self {
            TemplateType::Llama3Chat => {
                // LLaMA-3 chat format with special tokens
                out.push_str("<|begin_of_text|>");

                // System prompt if provided
                if let Some(sys) = system {
                    write!(out, "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>", sys)?;
                }

                // Render conversation history
                for turn in history {
                    let role = turn.role.as_str();
                    write!(
                        out,
                        "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                        role, turn.text
                    )?;
                }

                // Start assistant response
                write!(out, "<|start_header_id|>assistant<|end_header_id|>\n\n")?;
            }
            TemplateType::Instruct => {
                // Simple Q&A format
                if let Some(sys) = system {
                    writeln!(out, "System: {}\n", sys)?;
                }

                for turn in history {
                    match turn.role {
                        ChatRole::User => {
                            writeln!(out, "Q: {}", turn.text)?;
                        }
                        ChatRole::Assistant => {
                            writeln!(out, "A: {}", turn.text)?;
                        }
                        ChatRole::System => {
                            // System messages already emitted above
                        }
                    }
                }

                // Prompt for assistant response
                write!(out, "A: ")?;
            }
            TemplateType::Raw => {
                // Minimal: concatenate system prompt and full history
                if let Some(sys) = system {
                    writeln!(out, "{}\n", sys)?;
                }

                // Concatenate all turns with double newline separators
                for (i, turn) in history.iter().enumerate() {
                    if i > 0 {
                        write!(out, "\n\n")?;
                    }
                    write!(out, "{}", turn.text)?;
                }
            }
        }

        Ok(out)
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
    fn test_resolve_stop_token_ids() {
        // Create a mock tokenizer that can resolve special tokens
        use bitnet_tokenizers::MockTokenizer;
        let tokenizer = MockTokenizer::new();

        // Test that Raw template returns empty (no stops)
        let raw_ids = TemplateType::Raw.resolve_stop_token_ids(&tokenizer);
        assert_eq!(raw_ids, Vec::<u32>::new());

        // Test that Instruct template returns empty for mock tokenizer
        // (mock tokenizer doesn't resolve the instruct stop sequences)
        let instruct_ids = TemplateType::Instruct.resolve_stop_token_ids(&tokenizer);
        assert_eq!(instruct_ids, Vec::<u32>::new());

        // Test that LLaMA3Chat template also returns empty for mock tokenizer
        // In a real scenario with a real tokenizer that has <|eot_id|> in vocab,
        // this would return the resolved token IDs
        let llama3_ids = TemplateType::Llama3Chat.resolve_stop_token_ids(&tokenizer);
        assert_eq!(llama3_ids, Vec::<u32>::new());
    }

    #[test]
    fn test_template_glue_with_real_token_ids() {
        // This test proves the complete template glue: template → stops → token IDs
        // Given a mock tokenizer that maps <|eot_id|> → 128009 (LLaMA-3's actual EOT token ID)
        use bitnet_tokenizers::MockTokenizer;

        let tokenizer = MockTokenizer::with_special_tokens(&[
            ("<|eot_id|>", 128009),
            ("<|end_of_text|>", 128010),
        ]);

        // Test LLaMA3Chat template
        let template = TemplateType::Llama3Chat;

        // Assert: default_stop_sequences includes "<|eot_id|>"
        let stops = template.default_stop_sequences();
        assert!(stops.contains(&"<|eot_id|>".to_string()));
        assert!(stops.contains(&"<|end_of_text|>".to_string()));

        // Assert: resolve_stop_token_ids returns [128009, 128010]
        let stop_ids = template.resolve_stop_token_ids(&tokenizer);
        assert!(stop_ids.contains(&128009), "Expected 128009 for <|eot_id|>");
        assert!(stop_ids.contains(&128010), "Expected 128010 for <|end_of_text|>");

        // Assert: apply() wraps system_prompt + user in LLaMA-3 format
        let formatted = template.apply("What is 2+2?", Some("You are helpful"));
        assert!(formatted.contains("<|begin_of_text|>"));
        assert!(formatted.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(formatted.contains("You are helpful"));
        assert!(formatted.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(formatted.contains("What is 2+2?"));
        assert!(formatted.contains("<|eot_id|>"));
        assert!(formatted.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_bos_control() {
        assert!(TemplateType::Raw.should_add_bos());
        assert!(TemplateType::Instruct.should_add_bos());
        assert!(!TemplateType::Llama3Chat.should_add_bos()); // Has its own BOS
    }

    #[test]
    fn test_parse_special_control() {
        assert!(!TemplateType::Raw.parse_special());
        assert!(!TemplateType::Instruct.parse_special());
        assert!(TemplateType::Llama3Chat.parse_special()); // LLaMA-3 has special tokens
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

    #[test]
    fn test_render_chat_llama3() {
        let t = TemplateType::Llama3Chat;
        let hist = vec![
            ChatTurn::new(ChatRole::User, "Hello"),
            ChatTurn::new(ChatRole::Assistant, "Hi there!"),
            ChatTurn::new(ChatRole::User, "How are you?"),
        ];
        let s = t.render_chat(&hist, Some("You are helpful.")).unwrap();

        // Check for LLaMA-3 special tokens
        assert!(s.contains("<|begin_of_text|>"));
        assert!(s.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(s.contains("You are helpful."));
        assert!(s.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(s.contains("Hello"));
        assert!(s.contains("<|start_header_id|>assistant<|end_header_id|>"));
        assert!(s.contains("Hi there!"));
        assert!(s.contains("How are you?"));
        assert!(s.contains("<|eot_id|>"));

        // Should end with assistant header ready for generation
        assert!(s.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_render_chat_instruct() {
        let t = TemplateType::Instruct;
        let hist = vec![
            ChatTurn::new(ChatRole::User, "What is 2+2?"),
            ChatTurn::new(ChatRole::Assistant, "It's 4."),
            ChatTurn::new(ChatRole::User, "What about 3+3?"),
        ];
        let s = t.render_chat(&hist, None).unwrap();

        // Check Q&A format
        assert!(s.contains("Q: What is 2+2?"));
        assert!(s.contains("A: It's 4."));
        assert!(s.contains("Q: What about 3+3?"));

        // Should end with "A: " to prompt for response
        assert!(s.ends_with("A: "));
    }

    #[test]
    fn test_render_chat_instruct_with_system() {
        let t = TemplateType::Instruct;
        let hist = vec![ChatTurn::new(ChatRole::User, "Q1")];
        let s = t.render_chat(&hist, Some("You are a math tutor")).unwrap();

        assert!(s.contains("System: You are a math tutor"));
        assert!(s.contains("Q: Q1"));
        assert!(s.ends_with("A: "));
    }

    #[test]
    fn test_render_chat_raw() {
        let t = TemplateType::Raw;
        let hist = vec![
            ChatTurn::new(ChatRole::User, "First message"),
            ChatTurn::new(ChatRole::Assistant, "First response"),
            ChatTurn::new(ChatRole::User, "Second message"),
        ];
        let s = t.render_chat(&hist, None).unwrap();

        // Raw mode should concatenate full history with double newline separators
        assert!(s.contains("First message"));
        assert!(s.contains("First response"));
        assert!(s.contains("Second message"));
    }

    #[test]
    fn test_render_chat_raw_with_system() {
        let t = TemplateType::Raw;
        let hist = vec![ChatTurn::new(ChatRole::User, "Hello")];
        let s = t.render_chat(&hist, Some("System context")).unwrap();

        assert!(s.contains("System context"));
        assert!(s.contains("Hello"));
    }

    #[test]
    fn test_chat_role_as_str() {
        assert_eq!(ChatRole::System.as_str(), "system");
        assert_eq!(ChatRole::User.as_str(), "user");
        assert_eq!(ChatRole::Assistant.as_str(), "assistant");
    }

    #[test]
    fn test_chat_turn_new() {
        let turn = ChatTurn::new(ChatRole::User, "test message");
        assert_eq!(turn.role, ChatRole::User);
        assert_eq!(turn.text, "test message");
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    fn arb_template_type() -> impl Strategy<Value = TemplateType> {
        prop_oneof![
            Just(TemplateType::Raw),
            Just(TemplateType::Instruct),
            Just(TemplateType::Llama3Chat),
        ]
    }

    // apply always returns a non-empty string containing the user text.
    proptest! {
        #[test]
        fn apply_contains_user_text(
            template in arb_template_type(),
            user_text in "[a-zA-Z0-9 .,?!]{1,80}",
        ) {
            let result = template.apply(&user_text, None);
            prop_assert!(
                !result.is_empty(),
                "apply returned empty string for template={:?}",
                template
            );
            prop_assert!(
                result.contains(&user_text),
                "output {:?} should contain user_text {:?}",
                result,
                user_text
            );
        }
    }

    // Raw template passes user text through unchanged (no system prompt).
    proptest! {
        #[test]
        fn raw_template_is_identity(user_text in "[a-zA-Z0-9 .,?!]{1,80}") {
            let result = TemplateType::Raw.apply(&user_text, None);
            prop_assert_eq!(result, user_text);
        }
    }

    // Instruct template always ends with "\nA:".
    proptest! {
        #[test]
        fn instruct_ends_with_answer_prompt(
            user_text in "[a-zA-Z0-9 .,?!]{1,80}",
            system in proptest::option::of("[a-zA-Z0-9 ]{1,40}"),
        ) {
            let result = TemplateType::Instruct.apply(&user_text, system.as_deref());
            prop_assert!(
                result.ends_with("\nA:"),
                "instruct result {:?} should end with '\\nA:'",
                result
            );
        }
    }

    // default_stop_sequences returns at least one entry for non-Raw templates.
    proptest! {
        #[test]
        fn non_raw_templates_have_stop_sequences(
            template in prop_oneof![Just(TemplateType::Instruct), Just(TemplateType::Llama3Chat)],
        ) {
            let stops = template.default_stop_sequences();
            prop_assert!(
                !stops.is_empty(),
                "template={:?} should have default stop sequences",
                template
            );
        }
    }
}
