//! Prompt template engine for popular LLM chat formats.
//!
//! Supports `ChatML`, `Llama2`, `Llama3`, `Mistral`, `Phi3`, `Alpaca`,
//! `Vicuna`, `Zephyr`, and arbitrary custom templates.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// ChatRole
// ---------------------------------------------------------------------------

/// Role of a participant in a chat conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
    Function,
}

impl fmt::Display for ChatRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::System => write!(f, "system"),
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
            Self::Tool => write!(f, "tool"),
            Self::Function => write!(f, "function"),
        }
    }
}

impl ChatRole {
    /// Parse a role from its string representation.
    pub fn from_str_name(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "system" => Some(Self::System),
            "user" => Some(Self::User),
            "assistant" => Some(Self::Assistant),
            "tool" => Some(Self::Tool),
            "function" => Some(Self::Function),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// ChatMessage
// ---------------------------------------------------------------------------

/// A single message in a chat conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    pub name: Option<String>,
    pub tool_calls: Option<Vec<String>>,
}

impl ChatMessage {
    /// Create a new message with the given role and content.
    #[must_use]
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self { role, content: content.into(), name: None, tool_calls: None }
    }

    /// Attach an optional sender name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Attach optional tool calls (serialised as opaque strings).
    #[must_use]
    pub fn with_tool_calls(mut self, calls: Vec<String>) -> Self {
        self.tool_calls = Some(calls);
        self
    }
}

// ---------------------------------------------------------------------------
// TemplateFormat
// ---------------------------------------------------------------------------

/// Well-known prompt template formats.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TemplateFormat {
    ChatML,
    Llama2,
    Llama3,
    Mistral,
    Phi3,
    Alpaca,
    Vicuna,
    Zephyr,
    /// Arbitrary user-defined format identified by name.
    Custom(String),
}

impl fmt::Display for TemplateFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChatML => write!(f, "chatml"),
            Self::Llama2 => write!(f, "llama2"),
            Self::Llama3 => write!(f, "llama3"),
            Self::Mistral => write!(f, "mistral"),
            Self::Phi3 => write!(f, "phi3"),
            Self::Alpaca => write!(f, "alpaca"),
            Self::Vicuna => write!(f, "vicuna"),
            Self::Zephyr => write!(f, "zephyr"),
            Self::Custom(name) => write!(f, "custom:{name}"),
        }
    }
}

// ---------------------------------------------------------------------------
// PromptTemplate
// ---------------------------------------------------------------------------

/// Definition of how a prompt template renders messages.
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    pub format: TemplateFormat,
    pub bos_token: String,
    pub eos_token: String,
    pub system_prefix: String,
    pub system_suffix: String,
    pub user_prefix: String,
    pub user_suffix: String,
    pub assistant_prefix: String,
    pub assistant_suffix: String,
    pub separator: String,
}

impl PromptTemplate {
    /// Render a single message to a string fragment.
    #[must_use]
    pub fn render_message(&self, msg: &ChatMessage) -> String {
        match msg.role {
            ChatRole::System => {
                format!("{}{}{}", self.system_prefix, msg.content, self.system_suffix)
            }
            ChatRole::User => {
                format!("{}{}{}", self.user_prefix, msg.content, self.user_suffix)
            }
            ChatRole::Assistant => {
                format!("{}{}{}", self.assistant_prefix, msg.content, self.assistant_suffix)
            }
            // Tool / Function are rendered like user messages with a role tag.
            ChatRole::Tool | ChatRole::Function => {
                format!("{}[{}] {}{}", self.user_prefix, msg.role, msg.content, self.user_suffix)
            }
        }
    }

    /// Render a full conversation to a single prompt string.
    #[must_use]
    pub fn render(&self, messages: &[ChatMessage]) -> String {
        let mut out = String::with_capacity(256);
        out.push_str(&self.bos_token);
        for (i, msg) in messages.iter().enumerate() {
            if i > 0 {
                out.push_str(&self.separator);
            }
            out.push_str(&self.render_message(msg));
        }
        out.push_str(&self.eos_token);
        out
    }

    /// Render a conversation and append the assistant prefix so the model
    /// begins generating immediately.
    #[must_use]
    pub fn render_with_generation_prompt(&self, messages: &[ChatMessage]) -> String {
        let mut out = String::with_capacity(256);
        out.push_str(&self.bos_token);
        for (i, msg) in messages.iter().enumerate() {
            if i > 0 {
                out.push_str(&self.separator);
            }
            out.push_str(&self.render_message(msg));
        }
        out.push_str(&self.separator);
        out.push_str(&self.assistant_prefix);
        out
    }
}

// ---------------------------------------------------------------------------
// Built-in template constructors
// ---------------------------------------------------------------------------

/// Create the `ChatML` template.
#[must_use]
pub fn chatml_template() -> PromptTemplate {
    PromptTemplate {
        format: TemplateFormat::ChatML,
        bos_token: String::new(),
        eos_token: String::new(),
        system_prefix: "<|im_start|>system\n".into(),
        system_suffix: "<|im_end|>".into(),
        user_prefix: "<|im_start|>user\n".into(),
        user_suffix: "<|im_end|>".into(),
        assistant_prefix: "<|im_start|>assistant\n".into(),
        assistant_suffix: "<|im_end|>".into(),
        separator: "\n".into(),
    }
}

/// Create the Llama-2 template.
#[must_use]
pub fn llama2_template() -> PromptTemplate {
    PromptTemplate {
        format: TemplateFormat::Llama2,
        bos_token: "<s>".into(),
        eos_token: "</s>".into(),
        system_prefix: "[INST] <<SYS>>\n".into(),
        system_suffix: "\n<</SYS>>\n\n".into(),
        user_prefix: String::new(),
        user_suffix: " [/INST] ".into(),
        assistant_prefix: String::new(),
        assistant_suffix: " ".into(),
        separator: String::new(),
    }
}

/// Create the Llama-3 template.
#[must_use]
pub fn llama3_template() -> PromptTemplate {
    PromptTemplate {
        format: TemplateFormat::Llama3,
        bos_token: "<|begin_of_text|>".into(),
        eos_token: String::new(),
        system_prefix: "<|start_header_id|>system<|end_header_id|>\n\n".into(),
        system_suffix: "<|eot_id|>".into(),
        user_prefix: "<|start_header_id|>user<|end_header_id|>\n\n".into(),
        user_suffix: "<|eot_id|>".into(),
        assistant_prefix: "<|start_header_id|>assistant<|end_header_id|>\n\n".into(),
        assistant_suffix: "<|eot_id|>".into(),
        separator: String::new(),
    }
}

/// Create the Mistral template.
#[must_use]
pub fn mistral_template() -> PromptTemplate {
    PromptTemplate {
        format: TemplateFormat::Mistral,
        bos_token: "<s>".into(),
        eos_token: "</s>".into(),
        system_prefix: "[INST] ".into(),
        system_suffix: "\n".into(),
        user_prefix: String::new(),
        user_suffix: " [/INST]".into(),
        assistant_prefix: String::new(),
        assistant_suffix: " ".into(),
        separator: String::new(),
    }
}

/// Create the Phi-3 template.
#[must_use]
pub fn phi3_template() -> PromptTemplate {
    PromptTemplate {
        format: TemplateFormat::Phi3,
        bos_token: String::new(),
        eos_token: String::new(),
        system_prefix: "<|system|>\n".into(),
        system_suffix: "<|end|>".into(),
        user_prefix: "<|user|>\n".into(),
        user_suffix: "<|end|>".into(),
        assistant_prefix: "<|assistant|>\n".into(),
        assistant_suffix: "<|end|>".into(),
        separator: "\n".into(),
    }
}

/// Create the Alpaca template (instruction-following, no multi-turn chat).
#[must_use]
pub fn alpaca_template() -> PromptTemplate {
    PromptTemplate {
        format: TemplateFormat::Alpaca,
        bos_token: String::new(),
        eos_token: String::new(),
        system_prefix: "### System:\n".into(),
        system_suffix: "\n".into(),
        user_prefix: "### Instruction:\n".into(),
        user_suffix: "\n".into(),
        assistant_prefix: "### Response:\n".into(),
        assistant_suffix: "\n".into(),
        separator: "\n".into(),
    }
}

/// Create the Vicuna template.
#[must_use]
pub fn vicuna_template() -> PromptTemplate {
    PromptTemplate {
        format: TemplateFormat::Vicuna,
        bos_token: String::new(),
        eos_token: String::new(),
        system_prefix: "SYSTEM: ".into(),
        system_suffix: "\n".into(),
        user_prefix: "USER: ".into(),
        user_suffix: "\n".into(),
        assistant_prefix: "ASSISTANT: ".into(),
        assistant_suffix: "\n".into(),
        separator: String::new(),
    }
}

/// Create the Zephyr template.
#[must_use]
pub fn zephyr_template() -> PromptTemplate {
    PromptTemplate {
        format: TemplateFormat::Zephyr,
        bos_token: String::new(),
        eos_token: String::new(),
        system_prefix: "<|system|>\n".into(),
        system_suffix: "</s>".into(),
        user_prefix: "<|user|>\n".into(),
        user_suffix: "</s>".into(),
        assistant_prefix: "<|assistant|>\n".into(),
        assistant_suffix: "</s>".into(),
        separator: "\n".into(),
    }
}

// ---------------------------------------------------------------------------
// TemplateRegistry
// ---------------------------------------------------------------------------

/// Registry of named prompt templates.
#[derive(Debug, Clone)]
pub struct TemplateRegistry {
    templates: HashMap<String, PromptTemplate>,
}

impl Default for TemplateRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl TemplateRegistry {
    /// Create a registry pre-populated with all built-in formats.
    #[must_use]
    pub fn new() -> Self {
        let mut templates = HashMap::new();
        templates.insert("chatml".into(), chatml_template());
        templates.insert("llama2".into(), llama2_template());
        templates.insert("llama3".into(), llama3_template());
        templates.insert("mistral".into(), mistral_template());
        templates.insert("phi3".into(), phi3_template());
        templates.insert("alpaca".into(), alpaca_template());
        templates.insert("vicuna".into(), vicuna_template());
        templates.insert("zephyr".into(), zephyr_template());
        Self { templates }
    }

    /// Look up a template by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&PromptTemplate> {
        self.templates.get(name)
    }

    /// Register (or overwrite) a template under the given name.
    pub fn register(&mut self, name: impl Into<String>, template: PromptTemplate) {
        self.templates.insert(name.into(), template);
    }

    /// Remove a template by name. Returns the removed template if present.
    pub fn remove(&mut self, name: &str) -> Option<PromptTemplate> {
        self.templates.remove(name)
    }

    /// List all registered template names.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.templates.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// Number of registered templates.
    #[must_use]
    pub fn len(&self) -> usize {
        self.templates.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }
}

// ---------------------------------------------------------------------------
// PromptBuilder
// ---------------------------------------------------------------------------

/// Incrementally builds a prompt from chat messages using a template.
#[derive(Debug, Clone)]
pub struct PromptBuilder {
    template: PromptTemplate,
    messages: Vec<ChatMessage>,
}

impl PromptBuilder {
    /// Create a new builder backed by the given template.
    #[must_use]
    pub const fn new(template: PromptTemplate) -> Self {
        Self { template, messages: Vec::new() }
    }

    /// Append a message.
    pub fn add_message(&mut self, msg: ChatMessage) -> &mut Self {
        self.messages.push(msg);
        self
    }

    /// Append a system message.
    pub fn system(&mut self, content: impl Into<String>) -> &mut Self {
        self.add_message(ChatMessage::new(ChatRole::System, content))
    }

    /// Append a user message.
    pub fn user(&mut self, content: impl Into<String>) -> &mut Self {
        self.add_message(ChatMessage::new(ChatRole::User, content))
    }

    /// Append an assistant message.
    pub fn assistant(&mut self, content: impl Into<String>) -> &mut Self {
        self.add_message(ChatMessage::new(ChatRole::Assistant, content))
    }

    /// Build the prompt string (with trailing EOS).
    #[must_use]
    pub fn build(&self) -> String {
        self.template.render(&self.messages)
    }

    /// Build the prompt string with assistant generation prompt appended.
    #[must_use]
    pub fn build_with_generation_prompt(&self) -> String {
        self.template.render_with_generation_prompt(&self.messages)
    }

    /// Access the accumulated messages.
    #[must_use]
    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    /// Reset the builder, keeping the template.
    pub fn clear(&mut self) {
        self.messages.clear();
    }
}

// ---------------------------------------------------------------------------
// TokenCounter
// ---------------------------------------------------------------------------

/// Heuristic token-count estimator (character-based).
///
/// Uses a configurable chars-per-token ratio. The default of 4.0 is a
/// reasonable approximation for English BPE tokenizers.
#[derive(Debug, Clone, Copy)]
pub struct TokenCounter {
    chars_per_token: f64,
}

impl Default for TokenCounter {
    fn default() -> Self {
        Self { chars_per_token: 4.0 }
    }
}

impl TokenCounter {
    /// Create a counter with a custom chars-per-token ratio.
    #[must_use]
    pub const fn new(chars_per_token: f64) -> Self {
        Self { chars_per_token }
    }

    /// Estimate token count for the given text.
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn estimate(&self, text: &str) -> usize {
        let chars = text.chars().count() as f64;
        (chars / self.chars_per_token).ceil() as usize
    }

    /// Estimate token count for a slice of messages rendered with a template.
    #[must_use]
    pub fn estimate_messages(&self, template: &PromptTemplate, messages: &[ChatMessage]) -> usize {
        let rendered = template.render(messages);
        self.estimate(&rendered)
    }
}

// ---------------------------------------------------------------------------
// ContextTrimmer
// ---------------------------------------------------------------------------

/// Trims a conversation to fit within a token budget.
///
/// Strategy: always keep the system message (if any) and trim the *oldest*
/// user/assistant pairs first.
#[derive(Debug, Clone)]
pub struct ContextTrimmer {
    max_tokens: usize,
    counter: TokenCounter,
}

impl ContextTrimmer {
    /// Create a trimmer with the given token budget.
    #[must_use]
    pub fn new(max_tokens: usize) -> Self {
        Self { max_tokens, counter: TokenCounter::default() }
    }

    /// Create a trimmer with a custom `TokenCounter`.
    #[must_use]
    pub const fn with_counter(max_tokens: usize, counter: TokenCounter) -> Self {
        Self { max_tokens, counter }
    }

    /// Trim `messages` so the rendered prompt fits within the token budget.
    ///
    /// Returns the (possibly shortened) message list and a boolean indicating
    /// whether any messages were removed.
    #[must_use]
    pub fn trim(
        &self,
        template: &PromptTemplate,
        messages: &[ChatMessage],
    ) -> (Vec<ChatMessage>, bool) {
        // Fast path: already fits.
        if self.counter.estimate_messages(template, messages) <= self.max_tokens {
            return (messages.to_vec(), false);
        }

        // Separate system messages (kept) from the rest.
        let (system_msgs, other_msgs): (Vec<_>, Vec<_>) =
            messages.iter().cloned().partition(|m| m.role == ChatRole::System);

        // Progressively drop the oldest non-system messages.
        let mut kept = other_msgs;
        let mut trimmed = false;
        while !kept.is_empty() {
            let candidate: Vec<ChatMessage> =
                system_msgs.iter().cloned().chain(kept.iter().cloned()).collect();
            if self.counter.estimate_messages(template, &candidate) <= self.max_tokens {
                return (candidate, trimmed);
            }
            kept.remove(0);
            trimmed = true;
        }

        // Only system messages left (or empty).
        (system_msgs, trimmed)
    }
}

// ---------------------------------------------------------------------------
// PromptValidator
// ---------------------------------------------------------------------------

/// Validation errors for prompt construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    EmptyMessage { index: usize },
    EmptyConversation,
    ExceedsTokenLimit { estimated: usize, limit: usize },
    InvalidRole { index: usize, role: String },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMessage { index } => {
                write!(f, "message at index {index} has empty content")
            }
            Self::EmptyConversation => write!(f, "conversation has no messages"),
            Self::ExceedsTokenLimit { estimated, limit } => {
                write!(f, "estimated {estimated} tokens exceeds limit of {limit}")
            }
            Self::InvalidRole { index, role } => {
                write!(f, "invalid role '{role}' at index {index}")
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// Validates a sequence of chat messages.
#[derive(Debug, Clone)]
pub struct PromptValidator {
    max_tokens: Option<usize>,
    counter: TokenCounter,
}

impl Default for PromptValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl PromptValidator {
    /// Create a validator with no token limit.
    #[must_use]
    pub fn new() -> Self {
        Self { max_tokens: None, counter: TokenCounter::default() }
    }

    /// Set the maximum token limit.
    #[must_use]
    pub const fn with_token_limit(mut self, limit: usize) -> Self {
        self.max_tokens = Some(limit);
        self
    }

    /// Set a custom `TokenCounter`.
    #[must_use]
    pub const fn with_counter(mut self, counter: TokenCounter) -> Self {
        self.counter = counter;
        self
    }

    /// Validate messages, returning all errors found.
    pub fn validate(
        &self,
        template: &PromptTemplate,
        messages: &[ChatMessage],
    ) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();

        if messages.is_empty() {
            errors.push(ValidationError::EmptyConversation);
            return Err(errors);
        }

        for (i, msg) in messages.iter().enumerate() {
            if msg.content.trim().is_empty() {
                errors.push(ValidationError::EmptyMessage { index: i });
            }
        }

        if let Some(limit) = self.max_tokens {
            let est = self.counter.estimate_messages(template, messages);
            if est > limit {
                errors.push(ValidationError::ExceedsTokenLimit { estimated: est, limit });
            }
        }

        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }
}

// ---------------------------------------------------------------------------
// PromptTemplateMetrics
// ---------------------------------------------------------------------------

/// Lightweight metrics tracker for template usage.
#[derive(Debug, Clone, Default)]
pub struct PromptTemplateMetrics {
    pub renders: u64,
    pub total_prompt_chars: u64,
    pub trims: u64,
    pub validation_failures: u64,
}

impl PromptTemplateMetrics {
    /// Create a fresh metrics instance.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a render event.
    pub fn record_render(&mut self, prompt_chars: usize) {
        self.renders += 1;
        self.total_prompt_chars += u64::try_from(prompt_chars).unwrap_or(u64::MAX);
    }

    /// Record a trim event.
    pub const fn record_trim(&mut self) {
        self.trims += 1;
    }

    /// Record a validation failure.
    pub const fn record_validation_failure(&mut self) {
        self.validation_failures += 1;
    }

    /// Average prompt length in characters (0 if no renders).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_prompt_chars(&self) -> f64 {
        if self.renders == 0 { 0.0 } else { self.total_prompt_chars as f64 / self.renders as f64 }
    }

    /// Trim rate as a fraction of total renders.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn trim_rate(&self) -> f64 {
        if self.renders == 0 { 0.0 } else { self.trims as f64 / self.renders as f64 }
    }
}

// ---------------------------------------------------------------------------
// Convenience: validate a role string
// ---------------------------------------------------------------------------

/// Validate that a string is a known chat role.
pub fn validate_role(role: &str) -> Result<ChatRole, String> {
    ChatRole::from_str_name(role).ok_or_else(|| format!("unknown chat role: '{role}'"))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::similar_names, clippy::float_cmp)]
mod tests {
    use super::*;

    // --- helper -----------------------------------------------------------

    fn sys(content: &str) -> ChatMessage {
        ChatMessage::new(ChatRole::System, content)
    }
    fn user(content: &str) -> ChatMessage {
        ChatMessage::new(ChatRole::User, content)
    }
    fn asst(content: &str) -> ChatMessage {
        ChatMessage::new(ChatRole::Assistant, content)
    }

    // =====================================================================
    // ChatRole
    // =====================================================================

    #[test]
    fn role_display() {
        assert_eq!(ChatRole::System.to_string(), "system");
        assert_eq!(ChatRole::User.to_string(), "user");
        assert_eq!(ChatRole::Assistant.to_string(), "assistant");
        assert_eq!(ChatRole::Tool.to_string(), "tool");
        assert_eq!(ChatRole::Function.to_string(), "function");
    }

    #[test]
    fn role_from_str_valid() {
        assert_eq!(ChatRole::from_str_name("system"), Some(ChatRole::System));
        assert_eq!(ChatRole::from_str_name("USER"), Some(ChatRole::User));
        assert_eq!(ChatRole::from_str_name("Assistant"), Some(ChatRole::Assistant));
        assert_eq!(ChatRole::from_str_name("TOOL"), Some(ChatRole::Tool));
        assert_eq!(ChatRole::from_str_name("function"), Some(ChatRole::Function));
    }

    #[test]
    fn role_from_str_invalid() {
        assert_eq!(ChatRole::from_str_name("unknown"), None);
        assert_eq!(ChatRole::from_str_name(""), None);
    }

    #[test]
    fn validate_role_ok() {
        assert!(validate_role("system").is_ok());
    }

    #[test]
    fn validate_role_err() {
        assert!(validate_role("robot").is_err());
    }

    // =====================================================================
    // ChatMessage
    // =====================================================================

    #[test]
    fn message_new() {
        let m = ChatMessage::new(ChatRole::User, "hello");
        assert_eq!(m.role, ChatRole::User);
        assert_eq!(m.content, "hello");
        assert!(m.name.is_none());
        assert!(m.tool_calls.is_none());
    }

    #[test]
    fn message_with_name() {
        let m = ChatMessage::new(ChatRole::User, "hi").with_name("alice");
        assert_eq!(m.name.as_deref(), Some("alice"));
    }

    #[test]
    fn message_with_tool_calls() {
        let m = ChatMessage::new(ChatRole::Assistant, "ok").with_tool_calls(vec!["call_1".into()]);
        assert_eq!(m.tool_calls.as_ref().unwrap().len(), 1);
    }

    // =====================================================================
    // TemplateFormat display
    // =====================================================================

    #[test]
    fn template_format_display() {
        assert_eq!(TemplateFormat::ChatML.to_string(), "chatml");
        assert_eq!(TemplateFormat::Llama2.to_string(), "llama2");
        assert_eq!(TemplateFormat::Llama3.to_string(), "llama3");
        assert_eq!(TemplateFormat::Mistral.to_string(), "mistral");
        assert_eq!(TemplateFormat::Phi3.to_string(), "phi3");
        assert_eq!(TemplateFormat::Alpaca.to_string(), "alpaca");
        assert_eq!(TemplateFormat::Vicuna.to_string(), "vicuna");
        assert_eq!(TemplateFormat::Zephyr.to_string(), "zephyr");
        assert_eq!(TemplateFormat::Custom("foo".into()).to_string(), "custom:foo");
    }

    // =====================================================================
    // ChatML format
    // =====================================================================

    #[test]
    fn chatml_system_only() {
        let t = chatml_template();
        let prompt = t.render(&[sys("You are helpful.")]);
        assert!(prompt.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
    }

    #[test]
    fn chatml_user_assistant() {
        let t = chatml_template();
        let prompt = t.render(&[user("Hi"), asst("Hello!")]);
        assert!(prompt.contains("<|im_start|>user\nHi<|im_end|>"));
        assert!(prompt.contains("<|im_start|>assistant\nHello!<|im_end|>"));
    }

    #[test]
    fn chatml_multi_turn() {
        let t = chatml_template();
        let msgs = vec![sys("Be concise."), user("What is 2+2?"), asst("4"), user("And 3+3?")];
        let prompt = t.render(&msgs);
        assert!(prompt.contains("system\nBe concise."));
        assert!(prompt.contains("user\nWhat is 2+2?"));
        assert!(prompt.contains("assistant\n4"));
        assert!(prompt.contains("user\nAnd 3+3?"));
    }

    #[test]
    fn chatml_generation_prompt() {
        let t = chatml_template();
        let prompt = t.render_with_generation_prompt(&[user("Hello")]);
        assert!(
            prompt.ends_with("<|im_start|>assistant\n"),
            "Expected generation prompt suffix, got: {prompt}"
        );
    }

    // =====================================================================
    // Llama2 format
    // =====================================================================

    #[test]
    fn llama2_system_user() {
        let t = llama2_template();
        let prompt = t.render(&[sys("You help."), user("Hi")]);
        assert!(prompt.starts_with("<s>"));
        assert!(prompt.contains("<<SYS>>"));
        assert!(prompt.contains("You help."));
        assert!(prompt.contains("[/INST]"));
        assert!(prompt.ends_with("</s>"));
    }

    #[test]
    fn llama2_no_system() {
        let t = llama2_template();
        let prompt = t.render(&[user("Hi")]);
        assert!(prompt.contains("Hi"));
        assert!(prompt.contains("[/INST]"));
    }

    #[test]
    fn llama2_generation_prompt() {
        let t = llama2_template();
        let prompt = t.render_with_generation_prompt(&[user("Hi")]);
        assert!(!prompt.ends_with("</s>"));
    }

    // =====================================================================
    // Llama3 format
    // =====================================================================

    #[test]
    fn llama3_system_user() {
        let t = llama3_template();
        let prompt = t.render(&[sys("System."), user("Hello")]);
        assert!(prompt.starts_with("<|begin_of_text|>"));
        assert!(prompt.contains("start_header_id|>system"));
        assert!(prompt.contains("start_header_id|>user"));
        assert!(prompt.contains("<|eot_id|>"));
    }

    #[test]
    fn llama3_generation_prompt() {
        let t = llama3_template();
        let prompt = t.render_with_generation_prompt(&[user("Q")]);
        assert!(prompt.contains("start_header_id|>assistant"));
    }

    #[test]
    fn llama3_multi_turn() {
        let t = llama3_template();
        let msgs = vec![user("A"), asst("B"), user("C")];
        let prompt = t.render(&msgs);
        assert!(prompt.contains("A<|eot_id|>"));
        assert!(prompt.contains("B<|eot_id|>"));
        assert!(prompt.contains("C<|eot_id|>"));
    }

    // =====================================================================
    // Mistral format
    // =====================================================================

    #[test]
    fn mistral_basic() {
        let t = mistral_template();
        let prompt = t.render(&[sys("Be brief."), user("Hi")]);
        assert!(prompt.starts_with("<s>"));
        assert!(prompt.contains("[INST]"));
        assert!(prompt.contains("[/INST]"));
    }

    #[test]
    fn mistral_generation_prompt() {
        let t = mistral_template();
        let prompt = t.render_with_generation_prompt(&[user("Q")]);
        assert!(!prompt.ends_with("</s>"));
    }

    // =====================================================================
    // Phi3 format
    // =====================================================================

    #[test]
    fn phi3_system_user() {
        let t = phi3_template();
        let prompt = t.render(&[sys("Sys"), user("Hello")]);
        assert!(prompt.contains("<|system|>\nSys<|end|>"));
        assert!(prompt.contains("<|user|>\nHello<|end|>"));
    }

    #[test]
    fn phi3_generation_prompt() {
        let t = phi3_template();
        let prompt = t.render_with_generation_prompt(&[user("Q")]);
        assert!(prompt.contains("<|assistant|>"));
    }

    // =====================================================================
    // Alpaca format
    // =====================================================================

    #[test]
    fn alpaca_instruction() {
        let t = alpaca_template();
        let prompt = t.render(&[user("Summarise this.")]);
        assert!(prompt.contains("### Instruction:\nSummarise this."));
    }

    #[test]
    fn alpaca_with_system() {
        let t = alpaca_template();
        let prompt = t.render(&[sys("You are a bot."), user("Go")]);
        assert!(prompt.contains("### System:\nYou are a bot."));
        assert!(prompt.contains("### Instruction:\nGo"));
    }

    #[test]
    fn alpaca_generation_prompt() {
        let t = alpaca_template();
        let prompt = t.render_with_generation_prompt(&[user("Go")]);
        assert!(prompt.contains("### Response:\n"));
    }

    // =====================================================================
    // Vicuna format
    // =====================================================================

    #[test]
    fn vicuna_basic() {
        let t = vicuna_template();
        let prompt = t.render(&[user("Hi")]);
        assert!(prompt.contains("USER: Hi"));
    }

    #[test]
    fn vicuna_system() {
        let t = vicuna_template();
        let prompt = t.render(&[sys("Sys"), user("Hi"), asst("Hey")]);
        assert!(prompt.contains("SYSTEM: Sys"));
        assert!(prompt.contains("USER: Hi"));
        assert!(prompt.contains("ASSISTANT: Hey"));
    }

    #[test]
    fn vicuna_generation_prompt() {
        let t = vicuna_template();
        let prompt = t.render_with_generation_prompt(&[user("Hi")]);
        assert!(prompt.contains("ASSISTANT: "));
    }

    // =====================================================================
    // Zephyr format
    // =====================================================================

    #[test]
    fn zephyr_basic() {
        let t = zephyr_template();
        let prompt = t.render(&[user("Hi")]);
        assert!(prompt.contains("<|user|>\nHi</s>"));
    }

    #[test]
    fn zephyr_system_user() {
        let t = zephyr_template();
        let prompt = t.render(&[sys("Sys"), user("Q")]);
        assert!(prompt.contains("<|system|>\nSys</s>"));
        assert!(prompt.contains("<|user|>\nQ</s>"));
    }

    #[test]
    fn zephyr_generation_prompt() {
        let t = zephyr_template();
        let prompt = t.render_with_generation_prompt(&[user("Hi")]);
        assert!(prompt.contains("<|assistant|>"));
    }

    // =====================================================================
    // Tool / Function role rendering
    // =====================================================================

    #[test]
    fn tool_role_rendered() {
        let t = chatml_template();
        let m = ChatMessage::new(ChatRole::Tool, "result=42");
        let fragment = t.render_message(&m);
        assert!(fragment.contains("[tool]"));
        assert!(fragment.contains("result=42"));
    }

    #[test]
    fn function_role_rendered() {
        let t = chatml_template();
        let m = ChatMessage::new(ChatRole::Function, "fn_output");
        let fragment = t.render_message(&m);
        assert!(fragment.contains("[function]"));
        assert!(fragment.contains("fn_output"));
    }

    // =====================================================================
    // TemplateRegistry
    // =====================================================================

    #[test]
    fn registry_builtin_count() {
        let r = TemplateRegistry::new();
        assert_eq!(r.len(), 8);
        assert!(!r.is_empty());
    }

    #[test]
    fn registry_get_builtin() {
        let r = TemplateRegistry::new();
        assert!(r.get("chatml").is_some());
        assert!(r.get("llama3").is_some());
        assert!(r.get("nonexistent").is_none());
    }

    #[test]
    fn registry_register_custom() {
        let mut r = TemplateRegistry::new();
        let custom = PromptTemplate {
            format: TemplateFormat::Custom("mine".into()),
            bos_token: "<B>".into(),
            eos_token: "<E>".into(),
            system_prefix: "S:".into(),
            system_suffix: "\n".into(),
            user_prefix: "U:".into(),
            user_suffix: "\n".into(),
            assistant_prefix: "A:".into(),
            assistant_suffix: "\n".into(),
            separator: String::new(),
        };
        r.register("mine", custom);
        assert_eq!(r.len(), 9);
        assert!(r.get("mine").is_some());
    }

    #[test]
    fn registry_remove() {
        let mut r = TemplateRegistry::new();
        assert!(r.remove("chatml").is_some());
        assert_eq!(r.len(), 7);
        assert!(r.remove("chatml").is_none());
    }

    #[test]
    fn registry_names_sorted() {
        let r = TemplateRegistry::new();
        let names = r.names();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
    }

    #[test]
    fn registry_overwrite() {
        let mut r = TemplateRegistry::new();
        let original_bos = r.get("chatml").unwrap().bos_token.clone();
        let mut replacement = chatml_template();
        replacement.bos_token = "<CUSTOM_BOS>".into();
        r.register("chatml", replacement);
        assert_ne!(r.get("chatml").unwrap().bos_token, original_bos);
        assert_eq!(r.get("chatml").unwrap().bos_token, "<CUSTOM_BOS>");
        assert_eq!(r.len(), 8); // count unchanged
    }

    // =====================================================================
    // PromptBuilder
    // =====================================================================

    #[test]
    fn builder_basic() {
        let mut b = PromptBuilder::new(chatml_template());
        b.system("You are helpful.");
        b.user("Hi");
        let prompt = b.build();
        assert!(prompt.contains("system\nYou are helpful."));
        assert!(prompt.contains("user\nHi"));
    }

    #[test]
    fn builder_generation_prompt() {
        let mut b = PromptBuilder::new(chatml_template());
        b.user("Hi");
        let prompt = b.build_with_generation_prompt();
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn builder_messages_accessor() {
        let mut b = PromptBuilder::new(chatml_template());
        b.user("A");
        b.assistant("B");
        assert_eq!(b.messages().len(), 2);
    }

    #[test]
    fn builder_clear() {
        let mut b = PromptBuilder::new(chatml_template());
        b.user("A");
        b.clear();
        assert!(b.messages().is_empty());
    }

    #[test]
    fn builder_add_message_chaining() {
        let mut b = PromptBuilder::new(chatml_template());
        b.user("A").assistant("B").user("C");
        assert_eq!(b.messages().len(), 3);
    }

    // =====================================================================
    // TokenCounter
    // =====================================================================

    #[test]
    fn token_counter_default() {
        let c = TokenCounter::default();
        // "hello" = 5 chars, ceil(5/4) = 2
        assert_eq!(c.estimate("hello"), 2);
    }

    #[test]
    fn token_counter_empty() {
        let c = TokenCounter::default();
        assert_eq!(c.estimate(""), 0);
    }

    #[test]
    fn token_counter_custom_ratio() {
        let c = TokenCounter::new(2.0);
        // "hello" = 5 chars, ceil(5/2) = 3
        assert_eq!(c.estimate("hello"), 3);
    }

    #[test]
    fn token_counter_long_text() {
        let c = TokenCounter::default();
        let text = "a".repeat(400);
        assert_eq!(c.estimate(&text), 100);
    }

    #[test]
    fn token_counter_estimate_messages() {
        let c = TokenCounter::default();
        let t = chatml_template();
        let msgs = vec![user("Hello world")];
        let est = c.estimate_messages(&t, &msgs);
        assert!(est > 0);
    }

    // =====================================================================
    // ContextTrimmer
    // =====================================================================

    #[test]
    fn trimmer_no_trim_needed() {
        let trimmer = ContextTrimmer::new(10_000);
        let t = chatml_template();
        let msgs = vec![sys("System."), user("Hi")];
        let (result, trimmed) = trimmer.trim(&t, &msgs);
        assert!(!trimmed);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn trimmer_removes_oldest_first() {
        // Very tight budget that can't fit everything.
        let trimmer = ContextTrimmer::new(20);
        let t = chatml_template();
        let msgs = vec![
            sys("Sys"),
            user("Old question"),
            asst("Old answer"),
            user("New question"),
            asst("New answer"),
        ];
        let (result, trimmed) = trimmer.trim(&t, &msgs);
        assert!(trimmed);
        // System must survive.
        assert!(result.iter().any(|m| m.role == ChatRole::System));
        // Total messages should be fewer.
        assert!(result.len() < msgs.len());
    }

    #[test]
    fn trimmer_preserves_system() {
        let trimmer = ContextTrimmer::new(30);
        let t = chatml_template();
        let msgs = vec![
            sys("Important system prompt"),
            user("Q1"),
            asst("A1"),
            user("Q2"),
            asst("A2"),
            user("Q3"),
        ];
        let (result, _) = trimmer.trim(&t, &msgs);
        // System message must always be present.
        assert_eq!(result[0].role, ChatRole::System);
        assert_eq!(result[0].content, "Important system prompt");
    }

    #[test]
    fn trimmer_empty_input() {
        let trimmer = ContextTrimmer::new(100);
        let t = chatml_template();
        let (result, trimmed) = trimmer.trim(&t, &[]);
        assert!(!trimmed);
        assert!(result.is_empty());
    }

    #[test]
    fn trimmer_only_system_left() {
        // Budget so small only system survives.
        let trimmer = ContextTrimmer::new(15);
        let t = chatml_template();
        let msgs = vec![
            sys("Ok"),
            user("Very long question that exceeds the budget"),
            asst("Very long answer that also exceeds the budget"),
        ];
        let (result, trimmed) = trimmer.trim(&t, &msgs);
        assert!(trimmed);
        assert!(result.iter().all(|m| m.role == ChatRole::System));
    }

    #[test]
    fn trimmer_custom_counter() {
        let counter = TokenCounter::new(1.0); // 1 char = 1 token
        let trimmer = ContextTrimmer::with_counter(50, counter);
        let t = vicuna_template();
        let msgs = vec![user("a".repeat(100).as_str())];
        let (result, trimmed) = trimmer.trim(&t, &msgs);
        // Should trim because 100+ chars > 50 tokens with ratio 1.0.
        assert!(trimmed);
        assert!(result.is_empty());
    }

    // =====================================================================
    // PromptValidator
    // =====================================================================

    #[test]
    fn validator_ok() {
        let v = PromptValidator::new();
        let t = chatml_template();
        assert!(v.validate(&t, &[user("Hi")]).is_ok());
    }

    #[test]
    fn validator_empty_conversation() {
        let v = PromptValidator::new();
        let t = chatml_template();
        let errs = v.validate(&t, &[]).unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, ValidationError::EmptyConversation)));
    }

    #[test]
    fn validator_empty_message() {
        let v = PromptValidator::new();
        let t = chatml_template();
        let errs = v.validate(&t, &[user(""), user("ok")]).unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, ValidationError::EmptyMessage { index: 0 })));
    }

    #[test]
    fn validator_whitespace_only_message() {
        let v = PromptValidator::new();
        let t = chatml_template();
        let errs = v.validate(&t, &[user("   ")]).unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, ValidationError::EmptyMessage { .. })));
    }

    #[test]
    fn validator_token_limit_exceeded() {
        let v = PromptValidator::new().with_token_limit(5);
        let t = chatml_template();
        let errs = v.validate(&t, &[user("This is a fairly long message")]).unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, ValidationError::ExceedsTokenLimit { .. })));
    }

    #[test]
    fn validator_token_limit_ok() {
        let v = PromptValidator::new().with_token_limit(10_000);
        let t = chatml_template();
        assert!(v.validate(&t, &[user("Short")]).is_ok());
    }

    #[test]
    fn validator_multiple_errors() {
        let v = PromptValidator::new().with_token_limit(5);
        let t = chatml_template();
        let errs = v
            .validate(&t, &[user(""), user("A long enough message to exceed the limit")])
            .unwrap_err();
        // Should have both EmptyMessage and ExceedsTokenLimit.
        assert!(errs.len() >= 2);
    }

    #[test]
    fn validator_custom_counter() {
        let counter = TokenCounter::new(1.0);
        let v = PromptValidator::new().with_counter(counter).with_token_limit(10);
        let t = chatml_template();
        let errs = v.validate(&t, &[user("This exceeds ten tokens easily")]).unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, ValidationError::ExceedsTokenLimit { .. })));
    }

    #[test]
    fn validation_error_display() {
        let e = ValidationError::EmptyMessage { index: 3 };
        assert_eq!(e.to_string(), "message at index 3 has empty content");

        let e = ValidationError::EmptyConversation;
        assert_eq!(e.to_string(), "conversation has no messages");

        let e = ValidationError::ExceedsTokenLimit { estimated: 100, limit: 50 };
        assert!(e.to_string().contains("100"));
        assert!(e.to_string().contains("50"));

        let e = ValidationError::InvalidRole { index: 1, role: "robot".into() };
        assert!(e.to_string().contains("robot"));
    }

    // =====================================================================
    // PromptTemplateMetrics
    // =====================================================================

    #[test]
    fn metrics_default() {
        let m = PromptTemplateMetrics::new();
        assert_eq!(m.renders, 0);
        assert_eq!(m.total_prompt_chars, 0);
        assert_eq!(m.trims, 0);
        assert_eq!(m.validation_failures, 0);
    }

    #[test]
    fn metrics_avg_prompt_chars_no_renders() {
        let m = PromptTemplateMetrics::new();
        assert_eq!(m.avg_prompt_chars(), 0.0);
    }

    #[test]
    fn metrics_avg_prompt_chars() {
        let mut m = PromptTemplateMetrics::new();
        m.record_render(100);
        m.record_render(200);
        assert!((m.avg_prompt_chars() - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_trim_rate() {
        let mut m = PromptTemplateMetrics::new();
        m.record_render(50);
        m.record_render(50);
        m.record_trim();
        assert!((m.trim_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_trim_rate_no_renders() {
        let m = PromptTemplateMetrics::new();
        assert_eq!(m.trim_rate(), 0.0);
    }

    #[test]
    fn metrics_record_validation_failure() {
        let mut m = PromptTemplateMetrics::new();
        m.record_validation_failure();
        m.record_validation_failure();
        assert_eq!(m.validation_failures, 2);
    }

    // =====================================================================
    // Custom template
    // =====================================================================

    #[test]
    fn custom_template_render() {
        let t = PromptTemplate {
            format: TemplateFormat::Custom("simple".into()),
            bos_token: String::new(),
            eos_token: String::new(),
            system_prefix: "[SYS]".into(),
            system_suffix: "[/SYS]".into(),
            user_prefix: "[USR]".into(),
            user_suffix: "[/USR]".into(),
            assistant_prefix: "[AST]".into(),
            assistant_suffix: "[/AST]".into(),
            separator: "|".into(),
        };
        let prompt = t.render(&[sys("Be good"), user("Hi"), asst("Hey")]);
        assert_eq!(prompt, "[SYS]Be good[/SYS]|[USR]Hi[/USR]|[AST]Hey[/AST]");
    }

    #[test]
    fn custom_template_generation_prompt() {
        let t = PromptTemplate {
            format: TemplateFormat::Custom("test".into()),
            bos_token: String::new(),
            eos_token: String::new(),
            system_prefix: "S:".into(),
            system_suffix: "\n".into(),
            user_prefix: "U:".into(),
            user_suffix: "\n".into(),
            assistant_prefix: "A:".into(),
            assistant_suffix: "\n".into(),
            separator: String::new(),
        };
        let prompt = t.render_with_generation_prompt(&[user("Hello")]);
        assert!(prompt.ends_with("A:"));
    }

    // =====================================================================
    // Edge cases
    // =====================================================================

    #[test]
    fn single_message() {
        let t = chatml_template();
        let prompt = t.render(&[user("Solo")]);
        assert!(prompt.contains("Solo"));
    }

    #[test]
    fn no_system_message() {
        let t = llama3_template();
        let prompt = t.render(&[user("Q"), asst("A")]);
        assert!(!prompt.contains("system"));
    }

    #[test]
    fn very_long_system_prompt() {
        let long_sys = "x".repeat(10_000);
        let t = chatml_template();
        let prompt = t.render(&[sys(&long_sys), user("Q")]);
        assert!(prompt.contains(&long_sys));
    }

    #[test]
    fn empty_messages_slice() {
        let t = chatml_template();
        let prompt = t.render(&[]);
        // Should just be bos+eos (both empty for ChatML).
        assert!(prompt.is_empty());
    }

    #[test]
    fn generation_prompt_empty_messages() {
        let t = chatml_template();
        let prompt = t.render_with_generation_prompt(&[]);
        // Should end with assistant prefix.
        assert!(prompt.contains("<|im_start|>assistant\n"));
    }

    #[test]
    fn unicode_content() {
        let t = chatml_template();
        let prompt = t.render(&[user("こんにちは 🌍")]);
        assert!(prompt.contains("こんにちは 🌍"));
    }

    #[test]
    fn newlines_in_content() {
        let t = chatml_template();
        let prompt = t.render(&[user("line1\nline2\nline3")]);
        assert!(prompt.contains("line1\nline2\nline3"));
    }

    // =====================================================================
    // Round-trip: format → check invariants
    // =====================================================================

    #[test]
    fn roundtrip_all_builtins_contain_content() {
        let templates: Vec<PromptTemplate> = vec![
            chatml_template(),
            llama2_template(),
            llama3_template(),
            mistral_template(),
            phi3_template(),
            alpaca_template(),
            vicuna_template(),
            zephyr_template(),
        ];
        for t in &templates {
            let prompt = t.render(&[user("MARKER_CONTENT")]);
            assert!(prompt.contains("MARKER_CONTENT"), "Template {:?} lost user content", t.format);
        }
    }

    #[test]
    fn roundtrip_generation_prompt_all_builtins() {
        let templates: Vec<PromptTemplate> = vec![
            chatml_template(),
            llama2_template(),
            llama3_template(),
            mistral_template(),
            phi3_template(),
            alpaca_template(),
            vicuna_template(),
            zephyr_template(),
        ];
        for t in &templates {
            let gen_prompt = t.render_with_generation_prompt(&[user("Q")]);
            let full = t.render(&[user("Q")]);
            // Generation prompt should differ from full render (no eos, has
            // assistant prefix).
            assert_ne!(
                gen_prompt, full,
                "Template {:?}: generation prompt should differ from full render",
                t.format
            );
        }
    }

    #[test]
    fn roundtrip_system_preserved_all_builtins() {
        let templates: Vec<PromptTemplate> = vec![
            chatml_template(),
            llama2_template(),
            llama3_template(),
            mistral_template(),
            phi3_template(),
            alpaca_template(),
            vicuna_template(),
            zephyr_template(),
        ];
        for t in &templates {
            let prompt = t.render(&[sys("SYS_MARKER"), user("U")]);
            assert!(prompt.contains("SYS_MARKER"), "Template {:?} lost system content", t.format);
        }
    }

    // =====================================================================
    // Integration: builder + validator + trimmer + metrics
    // =====================================================================

    #[test]
    fn integration_build_validate_trim() {
        let t = chatml_template();
        let mut builder = PromptBuilder::new(t.clone());
        builder.system("System prompt.");
        builder.user("User message.");
        builder.assistant("Reply.");

        let validator = PromptValidator::new().with_token_limit(10_000);
        assert!(validator.validate(&t, builder.messages()).is_ok());

        let trimmer = ContextTrimmer::new(10_000);
        let (trimmed_msgs, was_trimmed) = trimmer.trim(&t, builder.messages());
        assert!(!was_trimmed);
        assert_eq!(trimmed_msgs.len(), 3);

        let mut metrics = PromptTemplateMetrics::new();
        let prompt = builder.build();
        metrics.record_render(prompt.len());
        assert_eq!(metrics.renders, 1);
        assert!(metrics.avg_prompt_chars() > 0.0);
    }

    #[test]
    fn integration_trim_then_validate() {
        let t = chatml_template();
        let msgs = vec![sys("System."), user("Q1"), asst("A1"), user("Q2"), asst("A2"), user("Q3")];
        let trimmer = ContextTrimmer::new(30);
        let (trimmed, _) = trimmer.trim(&t, &msgs);
        let validator = PromptValidator::new().with_token_limit(30);
        // After trimming, validation should pass.
        assert!(validator.validate(&t, &trimmed).is_ok());
    }
}
