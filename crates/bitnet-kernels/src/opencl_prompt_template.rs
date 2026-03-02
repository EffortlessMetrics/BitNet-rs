//! Prompt template engine for Intel Arc A770 (OpenCL backend).
//!
//! This module provides CPU reference implementations for formatting prompts
//! in various chat template formats (ChatML, Llama2, Alpaca, Vicuna, BitNet),
//! managing conversation context with token-budget truncation, and tracking
//! template usage statistics.

use std::fmt;

// ── Types ──────────────────────────────────────────────────────────────

/// Supported prompt template formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateType {
    ChatML,
    Llama2,
    Alpaca,
    Vicuna,
    Raw,
    BitNet,
}

impl fmt::Display for TemplateType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChatML => write!(f, "ChatML"),
            Self::Llama2 => write!(f, "Llama2"),
            Self::Alpaca => write!(f, "Alpaca"),
            Self::Vicuna => write!(f, "Vicuna"),
            Self::Raw => write!(f, "Raw"),
            Self::BitNet => write!(f, "BitNet"),
        }
    }
}

/// Role of a chat participant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::System => write!(f, "system"),
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
        }
    }
}

/// A single chat message with role and content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self { role, content: content.into() }
    }
}

/// Configuration for the template engine.
#[derive(Debug, Clone)]
pub struct PromptConfig {
    pub template_type: TemplateType,
    pub max_context_tokens: usize,
    pub system_prompt: Option<String>,
    pub add_generation_prompt: bool,
}

impl Default for PromptConfig {
    fn default() -> Self {
        Self {
            template_type: TemplateType::ChatML,
            max_context_tokens: 2048,
            system_prompt: None,
            add_generation_prompt: true,
        }
    }
}

/// Result of formatting a prompt.
#[derive(Debug, Clone)]
pub struct FormattedPrompt {
    pub text: String,
    pub token_estimate: usize,
    pub messages_included: usize,
    pub truncated: bool,
}

/// Manages a conversation with automatic context windowing.
#[derive(Debug, Clone)]
pub struct ConversationContext {
    pub messages: Vec<ChatMessage>,
    pub config: PromptConfig,
    pub total_tokens_estimate: usize,
}

/// Template engine with formatting and statistics tracking.
#[derive(Debug)]
pub struct TemplateEngine {
    pub config: PromptConfig,
    pub stats: TemplateStats,
}

/// Cumulative statistics for template engine usage.
#[derive(Debug, Clone, Default)]
pub struct TemplateStats {
    pub prompts_formatted: u64,
    pub messages_truncated: u64,
    pub total_tokens_processed: u64,
}

/// Errors produced by the template engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateError {
    EmptyPrompt,
    ContextOverflow { tokens: usize, max: usize },
    InvalidTemplate(String),
}

impl fmt::Display for TemplateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPrompt => write!(f, "empty prompt"),
            Self::ContextOverflow { tokens, max } => {
                write!(f, "context overflow: {tokens} tokens exceeds max {max}")
            }
            Self::InvalidTemplate(msg) => write!(f, "invalid template: {msg}"),
        }
    }
}

impl std::error::Error for TemplateError {}

// ── CPU reference implementations ──────────────────────────────────────

/// Create a new template engine with the given configuration.
pub fn create_template_engine(config: PromptConfig) -> TemplateEngine {
    TemplateEngine { config, stats: TemplateStats::default() }
}

/// Format messages using the ChatML template.
pub fn cpu_format_chatml(messages: &[ChatMessage], add_gen_prompt: bool) -> String {
    let mut out = String::new();
    for msg in messages {
        out.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
    }
    if add_gen_prompt {
        out.push_str("<|im_start|>assistant\n");
    }
    out
}

/// Format messages using the Llama-2 `[INST]` template.
pub fn cpu_format_llama2(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    let mut system_text: Option<&str> = None;

    for msg in messages {
        if msg.role == Role::System {
            system_text = Some(&msg.content);
            continue;
        }
    }

    let mut first_user = true;
    for msg in messages {
        match msg.role {
            Role::System => {}
            Role::User => {
                if first_user {
                    if let Some(sys) = system_text {
                        out.push_str(&format!(
                            "<s>[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{} [/INST]",
                            msg.content
                        ));
                    } else {
                        out.push_str(&format!("<s>[INST] {} [/INST]", msg.content));
                    }
                    first_user = false;
                } else {
                    out.push_str(&format!("<s>[INST] {} [/INST]", msg.content));
                }
            }
            Role::Assistant => {
                out.push_str(&format!(" {} </s>", msg.content));
            }
        }
    }
    out
}

/// Format messages using the Alpaca template.
pub fn cpu_format_alpaca(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for msg in messages {
        match msg.role {
            Role::System => {
                out.push_str(&format!("### System:\n{}\n\n", msg.content));
            }
            Role::User => {
                out.push_str(&format!("### Instruction:\n{}\n\n### Response:\n", msg.content));
            }
            Role::Assistant => {
                out.push_str(&format!("{}\n\n", msg.content));
            }
        }
    }
    out
}

/// Format messages using the Vicuna template.
pub fn cpu_format_vicuna(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for msg in messages {
        match msg.role {
            Role::System => {
                out.push_str(&format!("{}\n\n", msg.content));
            }
            Role::User => {
                out.push_str(&format!("USER: {}\n", msg.content));
            }
            Role::Assistant => {
                out.push_str(&format!("ASSISTANT: {}\n", msg.content));
            }
        }
    }
    out.push_str("ASSISTANT:");
    out
}

/// Format messages using the custom BitNet template.
pub fn cpu_format_bitnet(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for msg in messages {
        out.push_str(&format!(
            "<|bitnet_{role}|>{content}<|/bitnet_{role}|>",
            role = msg.role,
            content = msg.content
        ));
    }
    out.push_str("<|bitnet_assistant|>");
    out
}

/// Format a prompt through the engine, updating stats.
pub fn cpu_format_prompt(
    engine: &mut TemplateEngine,
    messages: &[ChatMessage],
) -> Result<FormattedPrompt, TemplateError> {
    if messages.is_empty() {
        return Err(TemplateError::EmptyPrompt);
    }

    let truncated_msgs = cpu_truncate_context(messages, engine.config.max_context_tokens);
    let truncated = truncated_msgs.len() < messages.len();

    let text = match engine.config.template_type {
        TemplateType::ChatML => {
            cpu_format_chatml(&truncated_msgs, engine.config.add_generation_prompt)
        }
        TemplateType::Llama2 => cpu_format_llama2(&truncated_msgs),
        TemplateType::Alpaca => cpu_format_alpaca(&truncated_msgs),
        TemplateType::Vicuna => cpu_format_vicuna(&truncated_msgs),
        TemplateType::Raw => {
            truncated_msgs.iter().map(|m| m.content.as_str()).collect::<Vec<_>>().join("\n")
        }
        TemplateType::BitNet => cpu_format_bitnet(&truncated_msgs),
    };

    let token_estimate = cpu_estimate_tokens(&text);
    let messages_included = truncated_msgs.len();

    engine.stats.prompts_formatted += 1;
    if truncated {
        engine.stats.messages_truncated += (messages.len() - truncated_msgs.len()) as u64;
    }
    engine.stats.total_tokens_processed += token_estimate as u64;

    Ok(FormattedPrompt { text, token_estimate, messages_included, truncated })
}

/// Rough token count estimate (~4 chars per token).
pub fn cpu_estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

/// Truncate conversation to fit within a token budget.
///
/// Keeps the system message (if present) plus the most recent messages.
pub fn cpu_truncate_context(messages: &[ChatMessage], max_tokens: usize) -> Vec<ChatMessage> {
    if messages.is_empty() {
        return Vec::new();
    }

    // Separate system message from the rest
    let (system_msgs, other_msgs): (Vec<_>, Vec<_>) =
        messages.iter().partition(|m| m.role == Role::System);

    let mut budget = max_tokens;
    let mut result: Vec<ChatMessage> = Vec::new();

    // Always include system messages first
    for sys in &system_msgs {
        let cost = cpu_estimate_tokens(&sys.content);
        if cost <= budget {
            result.push((*sys).clone());
            budget = budget.saturating_sub(cost);
        }
    }

    // Add non-system messages from most-recent first until budget exhausted
    let mut recent: Vec<&ChatMessage> = Vec::new();
    for msg in other_msgs.iter().rev() {
        let cost = cpu_estimate_tokens(&msg.content);
        if cost <= budget {
            recent.push(msg);
            budget = budget.saturating_sub(cost);
        } else {
            break;
        }
    }

    // Reverse back to chronological order
    recent.reverse();
    result.extend(recent.into_iter().cloned());

    result
}

/// Prepend a system prompt to a message list if not already present.
pub fn cpu_add_system_prompt(messages: &mut Vec<ChatMessage>, system: &str) {
    let has_system = messages.iter().any(|m| m.role == Role::System);
    if !has_system {
        messages.insert(0, ChatMessage::new(Role::System, system));
    }
}

/// Create a new conversation context with the given config.
pub fn cpu_create_conversation(config: PromptConfig) -> ConversationContext {
    let mut messages = Vec::new();
    if let Some(ref sys) = config.system_prompt {
        messages.push(ChatMessage::new(Role::System, sys.clone()));
    }
    let total_tokens_estimate = messages.iter().map(|m| cpu_estimate_tokens(&m.content)).sum();
    ConversationContext { messages, config, total_tokens_estimate }
}

/// Append a message to the conversation and update the token estimate.
pub fn cpu_add_message(ctx: &mut ConversationContext, msg: ChatMessage) {
    ctx.total_tokens_estimate += cpu_estimate_tokens(&msg.content);
    ctx.messages.push(msg);
}

/// Format the current conversation context into a prompt.
pub fn cpu_get_formatted(ctx: &ConversationContext) -> Result<FormattedPrompt, TemplateError> {
    if ctx.messages.is_empty() {
        return Err(TemplateError::EmptyPrompt);
    }

    let truncated_msgs = cpu_truncate_context(&ctx.messages, ctx.config.max_context_tokens);
    let truncated = truncated_msgs.len() < ctx.messages.len();

    let text = match ctx.config.template_type {
        TemplateType::ChatML => {
            cpu_format_chatml(&truncated_msgs, ctx.config.add_generation_prompt)
        }
        TemplateType::Llama2 => cpu_format_llama2(&truncated_msgs),
        TemplateType::Alpaca => cpu_format_alpaca(&truncated_msgs),
        TemplateType::Vicuna => cpu_format_vicuna(&truncated_msgs),
        TemplateType::Raw => {
            truncated_msgs.iter().map(|m| m.content.as_str()).collect::<Vec<_>>().join("\n")
        }
        TemplateType::BitNet => cpu_format_bitnet(&truncated_msgs),
    };

    let token_estimate = cpu_estimate_tokens(&text);
    let messages_included = truncated_msgs.len();

    Ok(FormattedPrompt { text, token_estimate, messages_included, truncated })
}

/// Format template stats as a human-readable summary.
pub fn format_template_stats(stats: &TemplateStats) -> String {
    format!(
        "prompts_formatted={}, messages_truncated={}, total_tokens_processed={}",
        stats.prompts_formatted, stats.messages_truncated, stats.total_tokens_processed,
    )
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn user(content: &str) -> ChatMessage {
        ChatMessage::new(Role::User, content)
    }
    fn assistant(content: &str) -> ChatMessage {
        ChatMessage::new(Role::Assistant, content)
    }
    fn system(content: &str) -> ChatMessage {
        ChatMessage::new(Role::System, content)
    }

    // ── ChatML ─────────────────────────────────────────────────────

    #[test]
    fn chatml_basic_user() {
        let msgs = vec![user("Hello")];
        let out = cpu_format_chatml(&msgs, false);
        assert!(out.contains("<|im_start|>user"));
        assert!(out.contains("Hello"));
        assert!(out.contains("<|im_end|>"));
    }

    #[test]
    fn chatml_system_and_user() {
        let msgs = vec![system("You are helpful"), user("Hi")];
        let out = cpu_format_chatml(&msgs, true);
        assert!(out.contains("<|im_start|>system"));
        assert!(out.contains("You are helpful"));
        assert!(out.contains("<|im_start|>user"));
        assert!(out.contains("<|im_start|>assistant\n"));
    }

    #[test]
    fn chatml_no_gen_prompt() {
        let msgs = vec![user("Hi")];
        let out = cpu_format_chatml(&msgs, false);
        assert!(!out.contains("<|im_start|>assistant\n"));
    }

    #[test]
    fn chatml_multi_turn() {
        let msgs = vec![user("Hi"), assistant("Hello!"), user("How?")];
        let out = cpu_format_chatml(&msgs, true);
        assert_eq!(out.matches("<|im_start|>").count(), 4); // 3 msgs + gen
        assert_eq!(out.matches("<|im_end|>").count(), 3);
    }

    // ── Llama2 ─────────────────────────────────────────────────────

    #[test]
    fn llama2_basic_user() {
        let msgs = vec![user("Hello")];
        let out = cpu_format_llama2(&msgs);
        assert!(out.contains("[INST]"));
        assert!(out.contains("[/INST]"));
        assert!(out.contains("Hello"));
    }

    #[test]
    fn llama2_with_system() {
        let msgs = vec![system("Be concise"), user("Hi")];
        let out = cpu_format_llama2(&msgs);
        assert!(out.contains("<<SYS>>"));
        assert!(out.contains("Be concise"));
        assert!(out.contains("<</SYS>>"));
    }

    #[test]
    fn llama2_multi_turn() {
        let msgs = vec![user("Hi"), assistant("Hello!"), user("More")];
        let out = cpu_format_llama2(&msgs);
        assert_eq!(out.matches("[INST]").count(), 2);
        assert!(out.contains("Hello!"));
    }

    // ── Alpaca ─────────────────────────────────────────────────────

    #[test]
    fn alpaca_instruction() {
        let msgs = vec![user("Summarize this")];
        let out = cpu_format_alpaca(&msgs);
        assert!(out.contains("### Instruction:"));
        assert!(out.contains("### Response:"));
        assert!(out.contains("Summarize this"));
    }

    #[test]
    fn alpaca_with_system() {
        let msgs = vec![system("You are a writer"), user("Write")];
        let out = cpu_format_alpaca(&msgs);
        assert!(out.contains("### System:"));
        assert!(out.contains("You are a writer"));
    }

    #[test]
    fn alpaca_multi_turn() {
        let msgs = vec![user("Q1"), assistant("A1"), user("Q2")];
        let out = cpu_format_alpaca(&msgs);
        assert_eq!(out.matches("### Instruction:").count(), 2);
    }

    // ── Vicuna ─────────────────────────────────────────────────────

    #[test]
    fn vicuna_basic() {
        let msgs = vec![user("Hello")];
        let out = cpu_format_vicuna(&msgs);
        assert!(out.contains("USER: Hello"));
        assert!(out.ends_with("ASSISTANT:"));
    }

    #[test]
    fn vicuna_with_system() {
        let msgs = vec![system("Sys prompt"), user("Hi")];
        let out = cpu_format_vicuna(&msgs);
        assert!(out.starts_with("Sys prompt"));
    }

    #[test]
    fn vicuna_multi_turn() {
        let msgs = vec![user("Hi"), assistant("Hey"), user("Bye")];
        let out = cpu_format_vicuna(&msgs);
        assert!(out.contains("USER: Hi"));
        assert!(out.contains("ASSISTANT: Hey"));
        assert!(out.contains("USER: Bye"));
    }

    // ── BitNet ─────────────────────────────────────────────────────

    #[test]
    fn bitnet_basic() {
        let msgs = vec![user("Hello")];
        let out = cpu_format_bitnet(&msgs);
        assert!(out.contains("<|bitnet_user|>Hello<|/bitnet_user|>"));
        assert!(out.ends_with("<|bitnet_assistant|>"));
    }

    #[test]
    fn bitnet_system_and_user() {
        let msgs = vec![system("Sys"), user("Hi")];
        let out = cpu_format_bitnet(&msgs);
        assert!(out.contains("<|bitnet_system|>Sys<|/bitnet_system|>"));
        assert!(out.contains("<|bitnet_user|>Hi<|/bitnet_user|>"));
    }

    #[test]
    fn bitnet_multi_turn() {
        let msgs = vec![user("A"), assistant("B"), user("C")];
        let out = cpu_format_bitnet(&msgs);
        assert!(out.contains("<|bitnet_assistant|>B<|/bitnet_assistant|>"));
    }

    // ── Token estimation ───────────────────────────────────────────

    #[test]
    fn token_estimate_empty() {
        assert_eq!(cpu_estimate_tokens(""), 0);
    }

    #[test]
    fn token_estimate_short() {
        // "Hello" = 5 chars → (5+3)/4 = 2
        assert_eq!(cpu_estimate_tokens("Hello"), 2);
    }

    #[test]
    fn token_estimate_reasonable() {
        let text = "The quick brown fox jumps over the lazy dog";
        let est = cpu_estimate_tokens(text);
        // 43 chars → (43+3)/4 = 11
        assert_eq!(est, 11);
        // Real tokenizer would give ~10, so this is reasonable
        assert!(est >= 8 && est <= 15);
    }

    // ── Context truncation ─────────────────────────────────────────

    #[test]
    fn truncate_keeps_system_and_recent() {
        let msgs = vec![system("Sys"), user("Old"), assistant("Old reply"), user("Recent")];
        // Budget that fits system + recent but not old messages
        let result = cpu_truncate_context(&msgs, 6);
        assert!(result.iter().any(|m| m.role == Role::System));
        assert!(result.last().unwrap().content == "Recent");
    }

    #[test]
    fn truncate_all_fit() {
        let msgs = vec![system("S"), user("U")];
        let result = cpu_truncate_context(&msgs, 10000);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn truncate_empty() {
        let result = cpu_truncate_context(&[], 100);
        assert!(result.is_empty());
    }

    #[test]
    fn truncate_drops_oldest_non_system() {
        let msgs = vec![system("S"), user("M1"), user("M2"), user("M3"), user("M4")];
        let result = cpu_truncate_context(&msgs, 3);
        // System "S" (1 tok) + most recent "M4" (1 tok) = 2, fits in 3
        assert!(result.iter().any(|m| m.content == "S"));
        assert!(result.last().unwrap().content == "M4");
        assert!(result.len() <= 3);
    }

    // ── System prompt ──────────────────────────────────────────────

    #[test]
    fn add_system_prompt_when_absent() {
        let mut msgs = vec![user("Hi")];
        cpu_add_system_prompt(&mut msgs, "Be helpful");
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[0].content, "Be helpful");
    }

    #[test]
    fn add_system_prompt_no_duplicate() {
        let mut msgs = vec![system("Existing"), user("Hi")];
        cpu_add_system_prompt(&mut msgs, "New");
        assert_eq!(msgs.iter().filter(|m| m.role == Role::System).count(), 1);
        assert_eq!(msgs[0].content, "Existing");
    }

    // ── Conversation ───────────────────────────────────────────────

    #[test]
    fn conversation_basic_flow() {
        let config = PromptConfig {
            template_type: TemplateType::ChatML,
            max_context_tokens: 2048,
            system_prompt: Some("You are helpful.".to_string()),
            add_generation_prompt: true,
        };
        let mut ctx = cpu_create_conversation(config);
        assert_eq!(ctx.messages.len(), 1); // system

        cpu_add_message(&mut ctx, user("Hello"));
        cpu_add_message(&mut ctx, assistant("Hi there!"));
        cpu_add_message(&mut ctx, user("How are you?"));

        let formatted = cpu_get_formatted(&ctx).unwrap();
        assert_eq!(formatted.messages_included, 4);
        assert!(!formatted.truncated);
        assert!(formatted.text.contains("Hello"));
        assert!(formatted.text.contains("Hi there!"));
        assert!(formatted.text.contains("How are you?"));
    }

    #[test]
    fn conversation_no_system() {
        let config = PromptConfig { system_prompt: None, ..PromptConfig::default() };
        let mut ctx = cpu_create_conversation(config);
        assert!(ctx.messages.is_empty());

        cpu_add_message(&mut ctx, user("Hi"));
        let formatted = cpu_get_formatted(&ctx).unwrap();
        assert_eq!(formatted.messages_included, 1);
    }

    #[test]
    fn conversation_token_tracking() {
        let config = PromptConfig::default();
        let mut ctx = cpu_create_conversation(config);
        cpu_add_message(&mut ctx, user("Hello world"));
        assert!(ctx.total_tokens_estimate > 0);
    }

    // ── Generation prompt ──────────────────────────────────────────

    #[test]
    fn gen_prompt_chatml_appended() {
        let msgs = vec![user("Test")];
        let out = cpu_format_chatml(&msgs, true);
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn gen_prompt_chatml_not_appended() {
        let msgs = vec![user("Test")];
        let out = cpu_format_chatml(&msgs, false);
        assert!(!out.ends_with("<|im_start|>assistant\n"));
    }

    // ── Edge cases ─────────────────────────────────────────────────

    #[test]
    fn edge_single_message() {
        let msgs = vec![user("Solo")];
        let mut engine = create_template_engine(PromptConfig::default());
        let result = cpu_format_prompt(&mut engine, &msgs).unwrap();
        assert_eq!(result.messages_included, 1);
        assert!(!result.truncated);
    }

    #[test]
    fn edge_empty_content() {
        let msgs = vec![user("")];
        let out = cpu_format_chatml(&msgs, false);
        assert!(out.contains("<|im_start|>user"));
    }

    #[test]
    fn edge_very_long_message_truncated() {
        let long_msg = "x".repeat(50_000);
        let msgs = vec![user(&long_msg)];
        let config = PromptConfig { max_context_tokens: 100, ..PromptConfig::default() };
        let mut engine = create_template_engine(config);
        let result = cpu_format_prompt(&mut engine, &msgs);
        // Single message can't be split, but truncation logic is at
        // message granularity — the message is still included.
        assert!(result.is_ok());
    }

    #[test]
    fn edge_no_system_prompt_in_llama2() {
        let msgs = vec![user("Hi")];
        let out = cpu_format_llama2(&msgs);
        assert!(!out.contains("<<SYS>>"));
    }

    #[test]
    fn edge_empty_messages_error() {
        let mut engine = create_template_engine(PromptConfig::default());
        let result = cpu_format_prompt(&mut engine, &[]);
        assert_eq!(result.unwrap_err(), TemplateError::EmptyPrompt);
    }

    // ── Property tests ─────────────────────────────────────────────

    #[test]
    fn property_formatted_contains_all_content() {
        let msgs = vec![system("Sys"), user("UserMsg"), assistant("AssistMsg")];
        let out = cpu_format_chatml(&msgs, false);
        for msg in &msgs {
            assert!(
                out.contains(&msg.content),
                "formatted output missing content: {}",
                msg.content
            );
        }
    }

    #[test]
    fn property_truncated_fits_budget() {
        let msgs: Vec<ChatMessage> =
            (0..100).map(|i| user(&format!("Message number {i}"))).collect();
        let max_tokens = 50;
        let truncated = cpu_truncate_context(&msgs, max_tokens);
        let total: usize = truncated.iter().map(|m| cpu_estimate_tokens(&m.content)).sum();
        assert!(total <= max_tokens, "truncated context {total} exceeds budget {max_tokens}");
    }

    // ── Stats ──────────────────────────────────────────────────────

    #[test]
    fn stats_tracking() {
        let mut engine = create_template_engine(PromptConfig::default());
        let msgs = vec![user("Hi")];
        cpu_format_prompt(&mut engine, &msgs).unwrap();
        cpu_format_prompt(&mut engine, &msgs).unwrap();
        assert_eq!(engine.stats.prompts_formatted, 2);
        assert!(engine.stats.total_tokens_processed > 0);
    }

    #[test]
    fn stats_format_display() {
        let stats = TemplateStats {
            prompts_formatted: 10,
            messages_truncated: 2,
            total_tokens_processed: 500,
        };
        let display = format_template_stats(&stats);
        assert!(display.contains("10"));
        assert!(display.contains("500"));
    }

    // ── Raw template ───────────────────────────────────────────────

    #[test]
    fn raw_format_joins_content() {
        let msgs = vec![user("Hello"), assistant("World")];
        let config = PromptConfig { template_type: TemplateType::Raw, ..PromptConfig::default() };
        let mut engine = create_template_engine(config);
        let result = cpu_format_prompt(&mut engine, &msgs).unwrap();
        assert_eq!(result.text, "Hello\nWorld");
    }

    // ── TemplateType Display ───────────────────────────────────────

    #[test]
    fn template_type_display() {
        assert_eq!(TemplateType::ChatML.to_string(), "ChatML");
        assert_eq!(TemplateType::BitNet.to_string(), "BitNet");
    }

    // ── TemplateError Display ──────────────────────────────────────

    #[test]
    fn template_error_display() {
        let err = TemplateError::ContextOverflow { tokens: 100, max: 50 };
        let msg = err.to_string();
        assert!(msg.contains("100"));
        assert!(msg.contains("50"));
    }

    #[test]
    fn template_error_empty_display() {
        assert_eq!(TemplateError::EmptyPrompt.to_string(), "empty prompt");
    }

    #[test]
    fn template_error_invalid() {
        let err = TemplateError::InvalidTemplate("bad".to_string());
        assert!(err.to_string().contains("bad"));
    }
}
