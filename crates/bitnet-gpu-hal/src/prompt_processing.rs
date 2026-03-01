//! Prompt processing pipeline for GPU inference.
//!
//! Provides template-aware tokenization, conversation history management,
//! intelligent truncation, stop-sequence detection, validation, and metrics
//! collection — all behind a unified [`PromptEngine`] façade.

#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PromptConfig
// ---------------------------------------------------------------------------

/// Top-level configuration for prompt processing.
#[derive(Debug, Clone)]
pub struct PromptConfig {
    /// Maximum tokens the model context window can hold.
    pub max_context_tokens: usize,
    /// Maximum tokens reserved for generation output.
    pub max_generation_tokens: usize,
    /// Chat template to apply when formatting prompts.
    pub template: PromptTemplate,
    /// Optional system prompt prepended to every conversation.
    pub system_prompt: Option<String>,
    /// Stop sequences that signal generation should halt.
    pub stop_sequences: Vec<String>,
    /// Maximum allowed prompt length in characters (0 = unlimited).
    pub max_prompt_chars: usize,
    /// Whether to enable content-policy validation.
    pub content_policy_enabled: bool,
}

impl Default for PromptConfig {
    fn default() -> Self {
        Self {
            max_context_tokens: 2048,
            max_generation_tokens: 256,
            template: PromptTemplate::ChatML,
            system_prompt: None,
            stop_sequences: Vec::new(),
            max_prompt_chars: 0,
            content_policy_enabled: false,
        }
    }
}

impl PromptConfig {
    /// Create a new config with the given context and generation limits.
    pub fn new(max_context_tokens: usize, max_generation_tokens: usize) -> Self {
        Self { max_context_tokens, max_generation_tokens, ..Default::default() }
    }

    /// Set the chat template.
    pub fn with_template(mut self, template: PromptTemplate) -> Self {
        self.template = template;
        self
    }

    /// Set the system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Add a stop sequence.
    pub fn with_stop_sequence(mut self, seq: impl Into<String>) -> Self {
        self.stop_sequences.push(seq.into());
        self
    }

    /// Tokens available for the prompt after reserving generation budget.
    pub fn prompt_token_budget(&self) -> usize {
        self.max_context_tokens.saturating_sub(self.max_generation_tokens)
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_context_tokens == 0 {
            return Err("max_context_tokens must be > 0".into());
        }
        if self.max_generation_tokens >= self.max_context_tokens {
            return Err("max_generation_tokens must be < max_context_tokens".into());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PromptTemplate
// ---------------------------------------------------------------------------

/// Chat template format used to wrap user/assistant/system turns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptTemplate {
    /// ChatML format (`<|im_start|>role\ncontent<|im_end|>`).
    ChatML,
    /// Alpaca instruction format.
    Alpaca,
    /// Generic instruct format (`### Instruction:\n…\n### Response:\n`).
    Instruct,
    /// Raw passthrough — no template wrapping.
    Raw,
    /// User-supplied custom template with `{role}` / `{content}` placeholders.
    Custom {
        /// Template for a message turn. Must contain `{role}` and `{content}`.
        turn_template: String,
        /// Separator inserted between turns.
        separator: String,
        /// String appended after all turns to prime the assistant reply.
        assistant_prefix: String,
    },
}

impl PromptTemplate {
    /// Format a single `(role, content)` turn using this template.
    pub fn format_turn(&self, role: &str, content: &str) -> String {
        match self {
            Self::ChatML => {
                format!("<|im_start|>{role}\n{content}<|im_end|>\n")
            }
            Self::Alpaca => match role {
                "system" => format!("{content}\n\n"),
                "user" => {
                    format!("### Instruction:\n{content}\n\n")
                }
                "assistant" => {
                    format!("### Response:\n{content}\n\n")
                }
                _ => format!("### {role}:\n{content}\n\n"),
            },
            Self::Instruct => match role {
                "system" => format!("{content}\n\n"),
                "user" => {
                    format!("### Instruction:\n{content}\n\n### Response:\n")
                }
                "assistant" => format!("{content}\n\n"),
                _ => format!("{content}\n\n"),
            },
            Self::Raw => content.to_string(),
            Self::Custom { turn_template, separator: _, assistant_prefix: _ } => {
                turn_template.replace("{role}", role).replace("{content}", content)
            }
        }
    }

    /// Separator inserted between formatted turns.
    pub fn separator(&self) -> &str {
        match self {
            Self::ChatML | Self::Alpaca | Self::Instruct | Self::Raw => "",
            Self::Custom { separator, .. } => separator,
        }
    }

    /// Prefix appended after the last turn to prime assistant generation.
    pub fn assistant_prefix(&self) -> &str {
        match self {
            Self::ChatML => "<|im_start|>assistant\n",
            Self::Alpaca => "### Response:\n",
            Self::Instruct => "",
            Self::Raw => "",
            Self::Custom { assistant_prefix, .. } => assistant_prefix,
        }
    }

    /// Format a full conversation into a single prompt string.
    pub fn format_conversation(&self, messages: &[ConversationMessage]) -> String {
        let mut out = String::new();
        let sep = self.separator();
        for (i, msg) in messages.iter().enumerate() {
            if i > 0 && !sep.is_empty() {
                out.push_str(sep);
            }
            out.push_str(&self.format_turn(&msg.role, &msg.content));
        }
        out.push_str(self.assistant_prefix());
        out
    }
}

// ---------------------------------------------------------------------------
// PromptTokenizer
// ---------------------------------------------------------------------------

/// Tokenizes prompts with template awareness.
///
/// Uses a simple whitespace-based tokenizer as a CPU reference
/// implementation. Production deployments should substitute a real
/// vocabulary-based tokenizer.
#[derive(Debug, Clone)]
pub struct PromptTokenizer {
    /// Average characters per token (used for estimation).
    chars_per_token: f64,
}

impl Default for PromptTokenizer {
    fn default() -> Self {
        Self { chars_per_token: 4.0 }
    }
}

impl PromptTokenizer {
    /// Create a tokenizer with a custom chars-per-token ratio.
    pub fn new(chars_per_token: f64) -> Self {
        Self { chars_per_token }
    }

    /// Estimate the token count for `text`.
    pub fn estimate_tokens(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        #[allow(clippy::cast_sign_loss)]
        let est = (text.len() as f64 / self.chars_per_token).ceil() as usize;
        est.max(1)
    }

    /// Tokenize `text` into whitespace-delimited pseudo-tokens.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace().map(String::from).collect()
    }

    /// Detokenize a slice of pseudo-tokens back into text.
    pub fn detokenize(&self, tokens: &[String]) -> String {
        tokens.join(" ")
    }

    /// Tokenize a formatted conversation, returning the tokens and their
    /// estimated count.
    pub fn tokenize_formatted(
        &self,
        template: &PromptTemplate,
        messages: &[ConversationMessage],
    ) -> (Vec<String>, usize) {
        let formatted = template.format_conversation(messages);
        let tokens = self.tokenize(&formatted);
        let count = tokens.len();
        (tokens, count)
    }
}

// ---------------------------------------------------------------------------
// SystemPromptManager
// ---------------------------------------------------------------------------

/// Manages system prompts with per-key caching.
#[derive(Debug, Clone)]
pub struct SystemPromptManager {
    prompts: HashMap<String, String>,
    default_key: String,
}

impl Default for SystemPromptManager {
    fn default() -> Self {
        Self { prompts: HashMap::new(), default_key: "default".into() }
    }
}

impl SystemPromptManager {
    /// Create a manager with a single default system prompt.
    pub fn new(default_prompt: impl Into<String>) -> Self {
        let mut mgr = Self::default();
        mgr.set("default", default_prompt);
        mgr
    }

    /// Store a system prompt under `key`.
    pub fn set(&mut self, key: impl Into<String>, prompt: impl Into<String>) {
        self.prompts.insert(key.into(), prompt.into());
    }

    /// Retrieve the prompt for `key`, falling back to the default.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.prompts.get(key).map(String::as_str)
    }

    /// Retrieve the default system prompt.
    pub fn default_prompt(&self) -> Option<&str> {
        self.get(&self.default_key)
    }

    /// Change which key is considered the default.
    pub fn set_default_key(&mut self, key: impl Into<String>) {
        self.default_key = key.into();
    }

    /// Number of cached prompts.
    pub fn len(&self) -> usize {
        self.prompts.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.prompts.is_empty()
    }

    /// Remove a prompt by key.
    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.prompts.remove(key)
    }

    /// Clear all cached prompts.
    pub fn clear(&mut self) {
        self.prompts.clear();
    }
}

// ---------------------------------------------------------------------------
// ConversationMessage / ConversationHistory
// ---------------------------------------------------------------------------

/// A single message in a conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversationMessage {
    /// Role of the speaker (`"system"`, `"user"`, `"assistant"`, …).
    pub role: String,
    /// Message content.
    pub content: String,
}

impl ConversationMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self { role: role.into(), content: content.into() }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

/// Manages multi-turn conversation history with context-window awareness.
#[derive(Debug, Clone)]
pub struct ConversationHistory {
    messages: Vec<ConversationMessage>,
    /// Maximum number of messages to retain (0 = unlimited).
    max_turns: usize,
}

impl Default for ConversationHistory {
    fn default() -> Self {
        Self { messages: Vec::new(), max_turns: 0 }
    }
}

impl ConversationHistory {
    /// Create a history with a maximum turn count.
    pub fn new(max_turns: usize) -> Self {
        Self { messages: Vec::new(), max_turns }
    }

    /// Append a message, evicting the oldest non-system messages if
    /// `max_turns` is exceeded.
    pub fn push(&mut self, msg: ConversationMessage) {
        self.messages.push(msg);
        self.enforce_limit();
    }

    /// Add a user message.
    pub fn add_user(&mut self, content: impl Into<String>) {
        self.push(ConversationMessage::user(content));
    }

    /// Add an assistant message.
    pub fn add_assistant(&mut self, content: impl Into<String>) {
        self.push(ConversationMessage::assistant(content));
    }

    /// Add a system message.
    pub fn add_system(&mut self, content: impl Into<String>) {
        self.push(ConversationMessage::system(content));
    }

    /// All messages in order.
    pub fn messages(&self) -> &[ConversationMessage] {
        &self.messages
    }

    /// Number of messages.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Whether the history is empty.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Clear all messages.
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Window the history to the most recent `n` messages, always preserving
    /// any leading system messages.
    pub fn window(&self, n: usize) -> Vec<ConversationMessage> {
        if self.messages.len() <= n {
            return self.messages.clone();
        }
        let mut result: Vec<ConversationMessage> = Vec::new();
        // Collect leading system messages.
        let mut first_non_system = 0;
        for msg in &self.messages {
            if msg.role == "system" {
                result.push(msg.clone());
                first_non_system += 1;
            } else {
                break;
            }
        }
        let remaining = &self.messages[first_non_system..];
        let keep = n.saturating_sub(result.len());
        if remaining.len() > keep {
            result.extend_from_slice(&remaining[remaining.len() - keep..]);
        } else {
            result.extend_from_slice(remaining);
        }
        result
    }

    /// Enforce `max_turns` by removing the oldest non-system messages.
    fn enforce_limit(&mut self) {
        if self.max_turns == 0 {
            return;
        }
        while self.messages.len() > self.max_turns {
            // Find the first non-system message and remove it.
            if let Some(idx) = self.messages.iter().position(|m| m.role != "system") {
                self.messages.remove(idx);
            } else {
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PromptTruncator
// ---------------------------------------------------------------------------

/// Strategy used when a prompt must be shortened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationStrategy {
    /// Remove tokens from the left (oldest context).
    Left,
    /// Remove tokens from the right (newest context).
    Right,
    /// Remove tokens from the middle, keeping start and end.
    Middle,
}

/// Intelligently truncates prompts that exceed the context window.
#[derive(Debug, Clone)]
pub struct PromptTruncator {
    strategy: TruncationStrategy,
    /// Number of tokens to reserve as headroom beyond the hard limit.
    headroom: usize,
}

impl Default for PromptTruncator {
    fn default() -> Self {
        Self { strategy: TruncationStrategy::Left, headroom: 0 }
    }
}

impl PromptTruncator {
    pub fn new(strategy: TruncationStrategy, headroom: usize) -> Self {
        Self { strategy, headroom }
    }

    /// Truncate `tokens` so that at most `budget` tokens remain (minus
    /// headroom). Returns the truncated tokens and the number removed.
    pub fn truncate(&self, tokens: &[String], budget: usize) -> (Vec<String>, usize) {
        let effective = budget.saturating_sub(self.headroom);
        if tokens.len() <= effective {
            return (tokens.to_vec(), 0);
        }
        let remove = tokens.len() - effective;
        let truncated = match self.strategy {
            TruncationStrategy::Left => tokens[remove..].to_vec(),
            TruncationStrategy::Right => tokens[..effective].to_vec(),
            TruncationStrategy::Middle => {
                let keep_start = effective / 2;
                let keep_end = effective - keep_start;
                let mut out = tokens[..keep_start].to_vec();
                out.extend_from_slice(&tokens[tokens.len() - keep_end..]);
                out
            }
        };
        (truncated, remove)
    }
}

// ---------------------------------------------------------------------------
// StopSequenceDetector
// ---------------------------------------------------------------------------

/// Detects stop sequences during incremental generation.
#[derive(Debug, Clone)]
pub struct StopSequenceDetector {
    sequences: Vec<String>,
    /// Accumulated text being scanned.
    buffer: String,
}

impl StopSequenceDetector {
    /// Create a detector with the given stop sequences.
    pub fn new(sequences: Vec<String>) -> Self {
        Self { sequences, buffer: String::new() }
    }

    /// Feed a new token/chunk of text.  Returns the first matched stop
    /// sequence if one is found.
    pub fn feed(&mut self, text: &str) -> Option<String> {
        self.buffer.push_str(text);
        for seq in &self.sequences {
            if self.buffer.contains(seq.as_str()) {
                return Some(seq.clone());
            }
        }
        None
    }

    /// Check whether `text` ends with a *partial* prefix of any stop
    /// sequence, indicating that more tokens might complete a match.
    pub fn has_partial_match(&self, text: &str) -> bool {
        for seq in &self.sequences {
            for prefix_len in 1..seq.len() {
                if text.ends_with(&seq[..prefix_len]) {
                    return true;
                }
            }
        }
        false
    }

    /// Reset the internal buffer.
    pub fn reset(&mut self) {
        self.buffer.clear();
    }

    /// The current buffer contents.
    pub fn buffer(&self) -> &str {
        &self.buffer
    }

    /// Text accumulated *before* the matched stop sequence. Returns `None`
    /// when no stop has been matched yet.
    pub fn text_before_stop(&self) -> Option<String> {
        for seq in &self.sequences {
            if let Some(idx) = self.buffer.find(seq.as_str()) {
                return Some(self.buffer[..idx].to_string());
            }
        }
        None
    }

    /// Number of registered stop sequences.
    pub fn num_sequences(&self) -> usize {
        self.sequences.len()
    }
}

// ---------------------------------------------------------------------------
// PromptValidator
// ---------------------------------------------------------------------------

/// Validation error kinds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// The prompt is empty.
    EmptyPrompt,
    /// The prompt exceeds the maximum character length.
    TooLong { len: usize, max: usize },
    /// The prompt exceeds the token budget.
    ExceedsTokenBudget { tokens: usize, budget: usize },
    /// The prompt contains content that violates the policy.
    ContentPolicy(String),
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyPrompt => write!(f, "prompt is empty"),
            Self::TooLong { len, max } => {
                write!(f, "prompt too long: {len} chars (max {max})")
            }
            Self::ExceedsTokenBudget { tokens, budget } => {
                write!(f, "prompt exceeds token budget: {tokens} tokens (budget {budget})")
            }
            Self::ContentPolicy(reason) => {
                write!(f, "content policy violation: {reason}")
            }
        }
    }
}

/// Validates prompts against length, token budget, and content policy.
#[derive(Debug, Clone)]
pub struct PromptValidator {
    max_chars: usize,
    token_budget: usize,
    content_policy_enabled: bool,
    /// Blocked substrings for the simple content policy.
    blocked_patterns: Vec<String>,
}

impl PromptValidator {
    /// Create a validator from a [`PromptConfig`].
    pub fn from_config(config: &PromptConfig) -> Self {
        Self {
            max_chars: config.max_prompt_chars,
            token_budget: config.prompt_token_budget(),
            content_policy_enabled: config.content_policy_enabled,
            blocked_patterns: Vec::new(),
        }
    }

    /// Create a validator with explicit limits.
    pub fn new(max_chars: usize, token_budget: usize, content_policy_enabled: bool) -> Self {
        Self { max_chars, token_budget, content_policy_enabled, blocked_patterns: Vec::new() }
    }

    /// Add a blocked pattern for content-policy checking.
    pub fn add_blocked_pattern(&mut self, pattern: impl Into<String>) {
        self.blocked_patterns.push(pattern.into());
    }

    /// Validate raw prompt text.
    pub fn validate(
        &self,
        text: &str,
        tokenizer: &PromptTokenizer,
    ) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();

        if text.is_empty() {
            errors.push(ValidationError::EmptyPrompt);
        }

        if self.max_chars > 0 && text.len() > self.max_chars {
            errors.push(ValidationError::TooLong { len: text.len(), max: self.max_chars });
        }

        let est = tokenizer.estimate_tokens(text);
        if self.token_budget > 0 && est > self.token_budget {
            errors.push(ValidationError::ExceedsTokenBudget {
                tokens: est,
                budget: self.token_budget,
            });
        }

        if self.content_policy_enabled {
            let lower = text.to_lowercase();
            for pat in &self.blocked_patterns {
                if lower.contains(&pat.to_lowercase()) {
                    errors.push(ValidationError::ContentPolicy(format!("blocked pattern: {pat}")));
                }
            }
        }

        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }
}

// ---------------------------------------------------------------------------
// PromptMetrics
// ---------------------------------------------------------------------------

/// Metrics collected during prompt processing.
#[derive(Debug, Clone, Default)]
pub struct PromptMetrics {
    /// Raw prompt length in characters.
    pub raw_char_count: usize,
    /// Estimated token count before truncation.
    pub estimated_tokens_before: usize,
    /// Estimated token count after truncation.
    pub estimated_tokens_after: usize,
    /// Number of tokens removed by truncation.
    pub tokens_truncated: usize,
    /// Number of conversation turns.
    pub turn_count: usize,
    /// Whether the system prompt was included.
    pub has_system_prompt: bool,
    /// Number of stop sequences configured.
    pub stop_sequence_count: usize,
    /// Template name used.
    pub template_name: String,
}

impl PromptMetrics {
    /// Truncation ratio (0.0 = none, 1.0 = fully truncated).
    pub fn truncation_ratio(&self) -> f64 {
        if self.estimated_tokens_before == 0 {
            return 0.0;
        }
        self.tokens_truncated as f64 / self.estimated_tokens_before as f64
    }

    /// Summarise the metrics as a human-readable string.
    pub fn summary(&self) -> String {
        format!(
            "chars={} tokens_before={} tokens_after={} truncated={} turns={} template={}",
            self.raw_char_count,
            self.estimated_tokens_before,
            self.estimated_tokens_after,
            self.tokens_truncated,
            self.turn_count,
            self.template_name,
        )
    }
}

// ---------------------------------------------------------------------------
// PromptEngine
// ---------------------------------------------------------------------------

/// Unified prompt processing pipeline.
///
/// Combines template formatting, tokenization, truncation, validation,
/// and metrics collection into a single entry-point.
#[derive(Debug, Clone)]
pub struct PromptEngine {
    config: PromptConfig,
    tokenizer: PromptTokenizer,
    truncator: PromptTruncator,
    system_prompts: SystemPromptManager,
}

/// Result of running the prompt engine.
#[derive(Debug, Clone)]
pub struct ProcessedPrompt {
    /// Final formatted and (possibly truncated) prompt text.
    pub text: String,
    /// Pseudo-tokens after processing.
    pub tokens: Vec<String>,
    /// Metrics collected during processing.
    pub metrics: PromptMetrics,
}

impl PromptEngine {
    /// Create an engine from a [`PromptConfig`].
    pub fn new(config: PromptConfig) -> Self {
        let system_prompts = if let Some(ref sp) = config.system_prompt {
            SystemPromptManager::new(sp.clone())
        } else {
            SystemPromptManager::default()
        };
        Self {
            config,
            tokenizer: PromptTokenizer::default(),
            truncator: PromptTruncator::default(),
            system_prompts,
        }
    }

    /// Replace the tokenizer.
    pub fn with_tokenizer(mut self, tokenizer: PromptTokenizer) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Replace the truncator.
    pub fn with_truncator(mut self, truncator: PromptTruncator) -> Self {
        self.truncator = truncator;
        self
    }

    /// Access the inner config.
    pub fn config(&self) -> &PromptConfig {
        &self.config
    }

    /// Access the system-prompt manager.
    pub fn system_prompts(&self) -> &SystemPromptManager {
        &self.system_prompts
    }

    /// Mutable access to the system-prompt manager.
    pub fn system_prompts_mut(&mut self) -> &mut SystemPromptManager {
        &mut self.system_prompts
    }

    /// Process a single user prompt string (no conversation history).
    pub fn process_single(
        &self,
        user_prompt: &str,
    ) -> Result<ProcessedPrompt, Vec<ValidationError>> {
        let mut messages = Vec::new();
        if let Some(sp) = self.system_prompts.default_prompt() {
            messages.push(ConversationMessage::system(sp));
        }
        messages.push(ConversationMessage::user(user_prompt));
        self.process_messages(&messages)
    }

    /// Process a full conversation.
    pub fn process_messages(
        &self,
        messages: &[ConversationMessage],
    ) -> Result<ProcessedPrompt, Vec<ValidationError>> {
        let formatted = self.config.template.format_conversation(messages);
        let raw_char_count = formatted.len();

        // Validate.
        let validator = PromptValidator::from_config(&self.config);
        // We validate the raw text but allow truncation to fix over-budget.
        // Only hard errors (empty, content policy) are propagated before
        // truncation.
        let pre_errors: Vec<ValidationError> = validator
            .validate(&formatted, &self.tokenizer)
            .err()
            .unwrap_or_default()
            .into_iter()
            .filter(|e| {
                matches!(e, ValidationError::EmptyPrompt | ValidationError::ContentPolicy(_))
            })
            .collect();
        if !pre_errors.is_empty() {
            return Err(pre_errors);
        }

        let tokens = self.tokenizer.tokenize(&formatted);
        let estimated_before = tokens.len();
        let budget = self.config.prompt_token_budget();

        let (final_tokens, truncated_count) = self.truncator.truncate(&tokens, budget);
        let estimated_after = final_tokens.len();
        let text = self.tokenizer.detokenize(&final_tokens);

        let template_name = match &self.config.template {
            PromptTemplate::ChatML => "ChatML",
            PromptTemplate::Alpaca => "Alpaca",
            PromptTemplate::Instruct => "Instruct",
            PromptTemplate::Raw => "Raw",
            PromptTemplate::Custom { .. } => "Custom",
        };

        let metrics = PromptMetrics {
            raw_char_count,
            estimated_tokens_before: estimated_before,
            estimated_tokens_after: estimated_after,
            tokens_truncated: truncated_count,
            turn_count: messages.len(),
            has_system_prompt: messages.iter().any(|m| m.role == "system"),
            stop_sequence_count: self.config.stop_sequences.len(),
            template_name: template_name.into(),
        };

        Ok(ProcessedPrompt { text, tokens: final_tokens, metrics })
    }

    /// Create a [`StopSequenceDetector`] pre-loaded with this engine's stop
    /// sequences.
    pub fn stop_detector(&self) -> StopSequenceDetector {
        StopSequenceDetector::new(self.config.stop_sequences.clone())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── PromptConfig ─────────────────────────────────────────────────────

    #[test]
    fn config_default_values() {
        let c = PromptConfig::default();
        assert_eq!(c.max_context_tokens, 2048);
        assert_eq!(c.max_generation_tokens, 256);
        assert!(c.system_prompt.is_none());
        assert!(c.stop_sequences.is_empty());
    }

    #[test]
    fn config_new_sets_limits() {
        let c = PromptConfig::new(4096, 512);
        assert_eq!(c.max_context_tokens, 4096);
        assert_eq!(c.max_generation_tokens, 512);
    }

    #[test]
    fn config_builder_methods() {
        let c = PromptConfig::new(4096, 512)
            .with_template(PromptTemplate::Alpaca)
            .with_system_prompt("Be helpful.")
            .with_stop_sequence("<|end|>");
        assert_eq!(c.template, PromptTemplate::Alpaca);
        assert_eq!(c.system_prompt.as_deref(), Some("Be helpful."));
        assert_eq!(c.stop_sequences, vec!["<|end|>"]);
    }

    #[test]
    fn config_prompt_token_budget() {
        let c = PromptConfig::new(2048, 256);
        assert_eq!(c.prompt_token_budget(), 1792);
    }

    #[test]
    fn config_budget_saturates() {
        let c = PromptConfig::new(100, 200);
        assert_eq!(c.prompt_token_budget(), 0);
    }

    #[test]
    fn config_validate_ok() {
        assert!(PromptConfig::new(2048, 256).validate().is_ok());
    }

    #[test]
    fn config_validate_zero_context() {
        let mut c = PromptConfig::default();
        c.max_context_tokens = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_validate_gen_ge_context() {
        let c = PromptConfig::new(256, 256);
        assert!(c.validate().is_err());
    }

    // ── PromptTemplate ───────────────────────────────────────────────────

    #[test]
    fn chatml_format_user() {
        let t = PromptTemplate::ChatML;
        let s = t.format_turn("user", "Hello");
        assert!(s.contains("<|im_start|>user"));
        assert!(s.contains("Hello"));
        assert!(s.contains("<|im_end|>"));
    }

    #[test]
    fn chatml_format_system() {
        let t = PromptTemplate::ChatML;
        let s = t.format_turn("system", "You are helpful.");
        assert!(s.contains("<|im_start|>system"));
    }

    #[test]
    fn chatml_assistant_prefix() {
        assert_eq!(PromptTemplate::ChatML.assistant_prefix(), "<|im_start|>assistant\n");
    }

    #[test]
    fn alpaca_format_user() {
        let s = PromptTemplate::Alpaca.format_turn("user", "Hi");
        assert!(s.contains("### Instruction:"));
        assert!(s.contains("Hi"));
    }

    #[test]
    fn alpaca_format_system() {
        let s = PromptTemplate::Alpaca.format_turn("system", "Be brief.");
        assert_eq!(s, "Be brief.\n\n");
    }

    #[test]
    fn alpaca_format_assistant() {
        let s = PromptTemplate::Alpaca.format_turn("assistant", "Ok");
        assert!(s.contains("### Response:"));
    }

    #[test]
    fn instruct_format_user() {
        let s = PromptTemplate::Instruct.format_turn("user", "Explain X.");
        assert!(s.contains("### Instruction:"));
        assert!(s.contains("### Response:"));
    }

    #[test]
    fn raw_passthrough() {
        let s = PromptTemplate::Raw.format_turn("user", "raw text");
        assert_eq!(s, "raw text");
    }

    #[test]
    fn custom_template() {
        let t = PromptTemplate::Custom {
            turn_template: "[{role}]: {content}".into(),
            separator: "\n".into(),
            assistant_prefix: "[assistant]: ".into(),
        };
        let s = t.format_turn("user", "hi");
        assert_eq!(s, "[user]: hi");
        assert_eq!(t.separator(), "\n");
        assert_eq!(t.assistant_prefix(), "[assistant]: ");
    }

    #[test]
    fn format_conversation_chatml() {
        let msgs =
            vec![ConversationMessage::system("Be kind."), ConversationMessage::user("Hello")];
        let text = PromptTemplate::ChatML.format_conversation(&msgs);
        assert!(text.contains("<|im_start|>system"));
        assert!(text.contains("<|im_start|>user"));
        assert!(text.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn format_conversation_empty() {
        let text = PromptTemplate::Raw.format_conversation(&[]);
        assert_eq!(text, "");
    }

    #[test]
    fn format_conversation_custom_separator() {
        let t = PromptTemplate::Custom {
            turn_template: "{role}:{content}".into(),
            separator: "|".into(),
            assistant_prefix: "".into(),
        };
        let msgs = vec![ConversationMessage::user("a"), ConversationMessage::user("b")];
        let text = t.format_conversation(&msgs);
        assert!(text.contains("|"));
    }

    // ── PromptTokenizer ─────────────────────────────────────────────────

    #[test]
    fn tokenizer_estimate_empty() {
        assert_eq!(PromptTokenizer::default().estimate_tokens(""), 0);
    }

    #[test]
    fn tokenizer_estimate_short() {
        let t = PromptTokenizer::default();
        // "hi" = 2 chars, 4.0 cpt → ceil(0.5) = 1
        assert_eq!(t.estimate_tokens("hi"), 1);
    }

    #[test]
    fn tokenizer_estimate_longer() {
        let t = PromptTokenizer::new(2.0);
        // 10 chars / 2.0 = 5
        assert_eq!(t.estimate_tokens("0123456789"), 5);
    }

    #[test]
    fn tokenizer_roundtrip() {
        let t = PromptTokenizer::default();
        let tokens = t.tokenize("hello world foo");
        assert_eq!(tokens, vec!["hello", "world", "foo"]);
        assert_eq!(t.detokenize(&tokens), "hello world foo");
    }

    #[test]
    fn tokenizer_tokenize_empty() {
        let t = PromptTokenizer::default();
        assert!(t.tokenize("").is_empty());
    }

    #[test]
    fn tokenizer_formatted() {
        let t = PromptTokenizer::default();
        let msgs = vec![ConversationMessage::user("hello world")];
        let (tokens, count) = t.tokenize_formatted(&PromptTemplate::Raw, &msgs);
        assert_eq!(count, tokens.len());
        assert!(count >= 2);
    }

    // ── SystemPromptManager ─────────────────────────────────────────────

    #[test]
    fn spm_new_with_default() {
        let m = SystemPromptManager::new("You are helpful.");
        assert_eq!(m.default_prompt(), Some("You are helpful."));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn spm_empty() {
        let m = SystemPromptManager::default();
        assert!(m.is_empty());
        assert!(m.default_prompt().is_none());
    }

    #[test]
    fn spm_set_and_get() {
        let mut m = SystemPromptManager::default();
        m.set("code", "You are a coding assistant.");
        assert_eq!(m.get("code"), Some("You are a coding assistant."));
        assert_eq!(m.get("missing"), None);
    }

    #[test]
    fn spm_remove() {
        let mut m = SystemPromptManager::new("test");
        assert_eq!(m.remove("default"), Some("test".into()));
        assert!(m.is_empty());
    }

    #[test]
    fn spm_clear() {
        let mut m = SystemPromptManager::new("test");
        m.set("other", "x");
        m.clear();
        assert!(m.is_empty());
    }

    #[test]
    fn spm_set_default_key() {
        let mut m = SystemPromptManager::default();
        m.set("custom", "custom prompt");
        m.set_default_key("custom");
        assert_eq!(m.default_prompt(), Some("custom prompt"));
    }

    // ── ConversationMessage ─────────────────────────────────────────────

    #[test]
    fn msg_constructors() {
        let s = ConversationMessage::system("sys");
        let u = ConversationMessage::user("usr");
        let a = ConversationMessage::assistant("asst");
        assert_eq!(s.role, "system");
        assert_eq!(u.role, "user");
        assert_eq!(a.role, "assistant");
    }

    #[test]
    fn msg_new() {
        let m = ConversationMessage::new("custom", "content");
        assert_eq!(m.role, "custom");
        assert_eq!(m.content, "content");
    }

    // ── ConversationHistory ─────────────────────────────────────────────

    #[test]
    fn history_push_and_len() {
        let mut h = ConversationHistory::default();
        h.add_user("hi");
        h.add_assistant("hello");
        assert_eq!(h.len(), 2);
    }

    #[test]
    fn history_empty() {
        let h = ConversationHistory::default();
        assert!(h.is_empty());
    }

    #[test]
    fn history_clear() {
        let mut h = ConversationHistory::default();
        h.add_user("a");
        h.clear();
        assert!(h.is_empty());
    }

    #[test]
    fn history_max_turns_eviction() {
        let mut h = ConversationHistory::new(3);
        h.add_system("sys");
        h.add_user("u1");
        h.add_user("u2");
        h.add_user("u3"); // should evict u1
        assert_eq!(h.len(), 3);
        // System message should survive.
        assert_eq!(h.messages()[0].role, "system");
    }

    #[test]
    fn history_window_preserves_system() {
        let mut h = ConversationHistory::default();
        h.add_system("sys");
        h.add_user("u1");
        h.add_user("u2");
        h.add_user("u3");
        let w = h.window(3);
        assert_eq!(w.len(), 3);
        assert_eq!(w[0].role, "system");
        assert_eq!(w[2].content, "u3");
    }

    #[test]
    fn history_window_smaller_than_history() {
        let mut h = ConversationHistory::default();
        h.add_user("a");
        h.add_user("b");
        let w = h.window(10);
        assert_eq!(w.len(), 2);
    }

    #[test]
    fn history_window_zero() {
        let mut h = ConversationHistory::default();
        h.add_user("a");
        let w = h.window(0);
        assert!(w.is_empty() || w.len() <= 1);
    }

    #[test]
    fn history_messages_slice() {
        let mut h = ConversationHistory::default();
        h.add_user("x");
        assert_eq!(h.messages().len(), 1);
        assert_eq!(h.messages()[0].content, "x");
    }

    // ── PromptTruncator ────────────────────────────────────────────────

    #[test]
    fn truncator_no_truncation_needed() {
        let t = PromptTruncator::default();
        let tokens: Vec<String> = vec!["a", "b", "c"].into_iter().map(Into::into).collect();
        let (out, removed) = t.truncate(&tokens, 10);
        assert_eq!(out.len(), 3);
        assert_eq!(removed, 0);
    }

    #[test]
    fn truncator_left() {
        let t = PromptTruncator::new(TruncationStrategy::Left, 0);
        let tokens: Vec<String> = vec!["a", "b", "c", "d"].into_iter().map(Into::into).collect();
        let (out, removed) = t.truncate(&tokens, 2);
        assert_eq!(out, vec!["c", "d"]);
        assert_eq!(removed, 2);
    }

    #[test]
    fn truncator_right() {
        let t = PromptTruncator::new(TruncationStrategy::Right, 0);
        let tokens: Vec<String> = vec!["a", "b", "c", "d"].into_iter().map(Into::into).collect();
        let (out, removed) = t.truncate(&tokens, 2);
        assert_eq!(out, vec!["a", "b"]);
        assert_eq!(removed, 2);
    }

    #[test]
    fn truncator_middle() {
        let t = PromptTruncator::new(TruncationStrategy::Middle, 0);
        let tokens: Vec<String> =
            vec!["a", "b", "c", "d", "e", "f"].into_iter().map(Into::into).collect();
        let (out, removed) = t.truncate(&tokens, 4);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0], "a");
        assert_eq!(out[1], "b");
        assert_eq!(out[3], "f");
        assert_eq!(removed, 2);
    }

    #[test]
    fn truncator_headroom() {
        let t = PromptTruncator::new(TruncationStrategy::Left, 2);
        let tokens: Vec<String> =
            vec!["a", "b", "c", "d", "e"].into_iter().map(Into::into).collect();
        // budget=5, headroom=2 → effective=3 → remove 2
        let (out, removed) = t.truncate(&tokens, 5);
        assert_eq!(out.len(), 3);
        assert_eq!(removed, 2);
    }

    #[test]
    fn truncator_empty_input() {
        let t = PromptTruncator::default();
        let (out, removed) = t.truncate(&[], 10);
        assert!(out.is_empty());
        assert_eq!(removed, 0);
    }

    #[test]
    fn truncator_zero_budget() {
        let t = PromptTruncator::default();
        let tokens: Vec<String> = vec!["a", "b"].into_iter().map(Into::into).collect();
        let (out, removed) = t.truncate(&tokens, 0);
        assert!(out.is_empty());
        assert_eq!(removed, 2);
    }

    // ── StopSequenceDetector ────────────────────────────────────────────

    #[test]
    fn stop_detector_no_match() {
        let mut d = StopSequenceDetector::new(vec!["<|end|>".into()]);
        assert!(d.feed("hello").is_none());
        assert!(d.feed(" world").is_none());
    }

    #[test]
    fn stop_detector_match() {
        let mut d = StopSequenceDetector::new(vec!["<|end|>".into()]);
        d.feed("some text");
        let result = d.feed("<|end|>");
        assert_eq!(result, Some("<|end|>".into()));
    }

    #[test]
    fn stop_detector_match_across_chunks() {
        let mut d = StopSequenceDetector::new(vec!["stop".into()]);
        assert!(d.feed("st").is_none());
        assert!(d.feed("op").is_some());
    }

    #[test]
    fn stop_detector_multiple_sequences() {
        let mut d = StopSequenceDetector::new(vec!["<|end|>".into(), "\n\n".into()]);
        assert!(d.feed("text\n").is_none());
        assert!(d.feed("\n").is_some());
    }

    #[test]
    fn stop_detector_reset() {
        let mut d = StopSequenceDetector::new(vec!["stop".into()]);
        d.feed("st");
        d.reset();
        assert_eq!(d.buffer(), "");
        assert!(d.feed("op").is_none());
    }

    #[test]
    fn stop_detector_text_before_stop() {
        let mut d = StopSequenceDetector::new(vec!["<END>".into()]);
        d.feed("Hello world<END>");
        assert_eq!(d.text_before_stop(), Some("Hello world".into()));
    }

    #[test]
    fn stop_detector_text_before_no_stop() {
        let mut d = StopSequenceDetector::new(vec!["<END>".into()]);
        d.feed("Hello world");
        assert!(d.text_before_stop().is_none());
    }

    #[test]
    fn stop_detector_partial_match() {
        let d = StopSequenceDetector::new(vec!["stop".into()]);
        assert!(d.has_partial_match("blah st"));
        assert!(!d.has_partial_match("blah x"));
    }

    #[test]
    fn stop_detector_num_sequences() {
        let d = StopSequenceDetector::new(vec!["a".into(), "b".into(), "c".into()]);
        assert_eq!(d.num_sequences(), 3);
    }

    #[test]
    fn stop_detector_buffer_accumulates() {
        let mut d = StopSequenceDetector::new(vec![]);
        d.feed("hello ");
        d.feed("world");
        assert_eq!(d.buffer(), "hello world");
    }

    // ── PromptValidator ─────────────────────────────────────────────────

    #[test]
    fn validator_ok() {
        let v = PromptValidator::new(100, 100, false);
        let t = PromptTokenizer::default();
        assert!(v.validate("Hello", &t).is_ok());
    }

    #[test]
    fn validator_empty() {
        let v = PromptValidator::new(100, 100, false);
        let t = PromptTokenizer::default();
        let errs = v.validate("", &t).unwrap_err();
        assert!(errs.contains(&ValidationError::EmptyPrompt));
    }

    #[test]
    fn validator_too_long() {
        let v = PromptValidator::new(5, 1000, false);
        let t = PromptTokenizer::default();
        let errs = v.validate("abcdef", &t).unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, ValidationError::TooLong { .. })));
    }

    #[test]
    fn validator_exceeds_budget() {
        let v = PromptValidator::new(0, 1, false);
        let t = PromptTokenizer::new(1.0); // 1 char = 1 token
        let errs = v.validate("ab", &t).unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, ValidationError::ExceedsTokenBudget { .. })));
    }

    #[test]
    fn validator_content_policy() {
        let mut v = PromptValidator::new(0, 0, true);
        v.add_blocked_pattern("BLOCKED");
        let t = PromptTokenizer::default();
        let errs = v.validate("this is BLOCKED text", &t).unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, ValidationError::ContentPolicy(_))));
    }

    #[test]
    fn validator_content_policy_case_insensitive() {
        let mut v = PromptValidator::new(0, 0, true);
        v.add_blocked_pattern("bad");
        let t = PromptTokenizer::default();
        let errs = v.validate("this is BAD text", &t).unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, ValidationError::ContentPolicy(_))));
    }

    #[test]
    fn validator_policy_disabled() {
        let mut v = PromptValidator::new(0, 0, false);
        v.add_blocked_pattern("bad");
        let t = PromptTokenizer::default();
        assert!(v.validate("this is bad text", &t).is_ok());
    }

    #[test]
    fn validator_from_config() {
        let config = PromptConfig::new(2048, 256);
        let v = PromptValidator::from_config(&config);
        let t = PromptTokenizer::default();
        assert!(v.validate("Hello", &t).is_ok());
    }

    #[test]
    fn validator_multiple_errors() {
        let mut v = PromptValidator::new(2, 1, true);
        v.add_blocked_pattern("x");
        let t = PromptTokenizer::new(1.0);
        let errs = v.validate("abcx", &t).unwrap_err();
        // Should have TooLong, ExceedsTokenBudget, ContentPolicy
        assert!(errs.len() >= 3);
    }

    // ── PromptMetrics ───────────────────────────────────────────────────

    #[test]
    fn metrics_default() {
        let m = PromptMetrics::default();
        assert_eq!(m.raw_char_count, 0);
        assert_eq!(m.truncation_ratio(), 0.0);
    }

    #[test]
    fn metrics_truncation_ratio() {
        let m = PromptMetrics {
            estimated_tokens_before: 100,
            tokens_truncated: 25,
            ..Default::default()
        };
        assert!((m.truncation_ratio() - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_summary_format() {
        let m = PromptMetrics {
            raw_char_count: 100,
            estimated_tokens_before: 25,
            estimated_tokens_after: 20,
            tokens_truncated: 5,
            turn_count: 3,
            template_name: "ChatML".into(),
            ..Default::default()
        };
        let s = m.summary();
        assert!(s.contains("chars=100"));
        assert!(s.contains("turns=3"));
        assert!(s.contains("template=ChatML"));
    }

    #[test]
    fn metrics_no_truncation_ratio_zero() {
        let m = PromptMetrics {
            estimated_tokens_before: 50,
            tokens_truncated: 0,
            ..Default::default()
        };
        assert_eq!(m.truncation_ratio(), 0.0);
    }

    // ── PromptEngine ────────────────────────────────────────────────────

    #[test]
    fn engine_process_single_raw() {
        let config = PromptConfig::new(2048, 256).with_template(PromptTemplate::Raw);
        let engine = PromptEngine::new(config);
        let result = engine.process_single("Hello world").unwrap();
        assert!(result.text.contains("Hello"));
        assert!(result.text.contains("world"));
        assert_eq!(result.metrics.turn_count, 1);
    }

    #[test]
    fn engine_process_single_with_system() {
        let config = PromptConfig::new(2048, 256)
            .with_template(PromptTemplate::Raw)
            .with_system_prompt("Be helpful.");
        let engine = PromptEngine::new(config);
        let result = engine.process_single("Hi").unwrap();
        assert!(result.metrics.has_system_prompt);
        assert_eq!(result.metrics.turn_count, 2);
    }

    #[test]
    fn engine_process_chatml() {
        let config = PromptConfig::new(2048, 256)
            .with_template(PromptTemplate::ChatML)
            .with_system_prompt("System.");
        let engine = PromptEngine::new(config);
        let result = engine.process_single("User input").unwrap();
        assert!(result.text.contains("<|im_start|>"));
        assert_eq!(result.metrics.template_name, "ChatML");
    }

    #[test]
    fn engine_process_alpaca() {
        let config = PromptConfig::new(2048, 256).with_template(PromptTemplate::Alpaca);
        let engine = PromptEngine::new(config);
        let result = engine.process_single("Do something").unwrap();
        assert!(result.text.contains("Instruction"));
    }

    #[test]
    fn engine_process_messages() {
        let config = PromptConfig::new(2048, 256).with_template(PromptTemplate::Raw);
        let engine = PromptEngine::new(config);
        let msgs = vec![
            ConversationMessage::user("one"),
            ConversationMessage::assistant("two"),
            ConversationMessage::user("three"),
        ];
        let result = engine.process_messages(&msgs).unwrap();
        assert_eq!(result.metrics.turn_count, 3);
    }

    #[test]
    fn engine_truncation_applied() {
        // Config with very small budget to force truncation.
        let config = PromptConfig::new(10, 5).with_template(PromptTemplate::Raw);
        let engine = PromptEngine::new(config);
        let long_input = "a b c d e f g h i j k l m n o p";
        let result = engine.process_single(long_input).unwrap();
        assert!(result.metrics.tokens_truncated > 0);
        assert!(result.tokens.len() <= 5);
    }

    #[test]
    fn engine_metrics_populated() {
        let config = PromptConfig::new(2048, 256)
            .with_template(PromptTemplate::ChatML)
            .with_stop_sequence("<|end|>");
        let engine = PromptEngine::new(config);
        let result = engine.process_single("test").unwrap();
        assert!(result.metrics.raw_char_count > 0);
        assert!(result.metrics.estimated_tokens_before > 0);
        assert_eq!(result.metrics.stop_sequence_count, 1);
    }

    #[test]
    fn engine_stop_detector() {
        let config =
            PromptConfig::new(2048, 256).with_stop_sequence("<|end|>").with_stop_sequence("\n\n");
        let engine = PromptEngine::new(config);
        let mut det = engine.stop_detector();
        assert_eq!(det.num_sequences(), 2);
        assert!(det.feed("text<|end|>").is_some());
    }

    #[test]
    fn engine_with_custom_tokenizer() {
        let config = PromptConfig::new(2048, 256).with_template(PromptTemplate::Raw);
        let engine = PromptEngine::new(config).with_tokenizer(PromptTokenizer::new(2.0));
        let result = engine.process_single("Hello world").unwrap();
        assert!(!result.tokens.is_empty());
    }

    #[test]
    fn engine_with_custom_truncator() {
        let config = PromptConfig::new(10, 5).with_template(PromptTemplate::Raw);
        let truncator = PromptTruncator::new(TruncationStrategy::Right, 0);
        let engine = PromptEngine::new(config).with_truncator(truncator);
        let result = engine.process_single("a b c d e f g h i j").unwrap();
        assert!(result.tokens.len() <= 5);
        // Right truncation keeps the beginning.
        assert_eq!(result.tokens[0], "a");
    }

    #[test]
    fn engine_config_accessor() {
        let config = PromptConfig::new(4096, 512);
        let engine = PromptEngine::new(config);
        assert_eq!(engine.config().max_context_tokens, 4096);
    }

    #[test]
    fn engine_system_prompts_accessor() {
        let config = PromptConfig::new(2048, 256).with_system_prompt("SP");
        let engine = PromptEngine::new(config);
        assert_eq!(engine.system_prompts().default_prompt(), Some("SP"));
    }

    #[test]
    fn engine_system_prompts_mut() {
        let config = PromptConfig::new(2048, 256);
        let mut engine = PromptEngine::new(config);
        engine.system_prompts_mut().set("alt", "alt prompt");
        assert_eq!(engine.system_prompts().get("alt"), Some("alt prompt"));
    }

    // ── Integration / round-trip tests ──────────────────────────────────

    #[test]
    fn roundtrip_conversation_chatml() {
        let mut history = ConversationHistory::default();
        history.add_system("You are a helpful assistant.");
        history.add_user("What is Rust?");
        history.add_assistant("Rust is a systems programming language.");
        history.add_user("Tell me more.");

        let config = PromptConfig::new(2048, 256)
            .with_template(PromptTemplate::ChatML)
            .with_system_prompt("You are a helpful assistant.");
        let engine = PromptEngine::new(config);
        let result = engine.process_messages(history.messages()).unwrap();
        assert_eq!(result.metrics.turn_count, 4);
        assert!(result.metrics.has_system_prompt);
        assert!(result.text.contains("Rust"));
    }

    #[test]
    fn roundtrip_window_then_process() {
        let mut history = ConversationHistory::default();
        history.add_system("sys");
        for i in 0..20 {
            history.add_user(format!("msg {i}"));
        }
        let windowed = history.window(5);
        assert_eq!(windowed[0].role, "system");

        let config = PromptConfig::new(2048, 256).with_template(PromptTemplate::Raw);
        let engine = PromptEngine::new(config);
        let result = engine.process_messages(&windowed).unwrap();
        assert_eq!(result.metrics.turn_count, 5);
    }

    #[test]
    fn stop_detector_with_engine_roundtrip() {
        let config = PromptConfig::new(2048, 256)
            .with_template(PromptTemplate::ChatML)
            .with_stop_sequence("<|im_end|>");
        let engine = PromptEngine::new(config);
        let mut det = engine.stop_detector();

        // Simulate token-by-token generation.
        let tokens = ["Hello", " ", "world", "<|im_end|>"];
        let mut stopped = false;
        for tok in &tokens {
            if det.feed(tok).is_some() {
                stopped = true;
                break;
            }
        }
        assert!(stopped);
        assert_eq!(det.text_before_stop(), Some("Hello world".into()));
    }

    #[test]
    fn truncation_metrics_consistent() {
        let config = PromptConfig::new(10, 5).with_template(PromptTemplate::Raw);
        let engine = PromptEngine::new(config);
        let result = engine.process_single("a b c d e f g h i j").unwrap();
        let m = &result.metrics;
        assert_eq!(m.estimated_tokens_before, m.estimated_tokens_after + m.tokens_truncated);
    }

    #[test]
    fn validation_error_display() {
        let e = ValidationError::EmptyPrompt;
        assert_eq!(format!("{e}"), "prompt is empty");

        let e = ValidationError::TooLong { len: 10, max: 5 };
        assert!(format!("{e}").contains("10"));

        let e = ValidationError::ExceedsTokenBudget { tokens: 100, budget: 50 };
        assert!(format!("{e}").contains("100"));

        let e = ValidationError::ContentPolicy("test".into());
        assert!(format!("{e}").contains("test"));
    }

    #[test]
    fn instruct_template_format() {
        let config = PromptConfig::new(2048, 256).with_template(PromptTemplate::Instruct);
        let engine = PromptEngine::new(config);
        let result = engine.process_single("Explain X.").unwrap();
        assert!(result.text.contains("Instruction"));
    }

    #[test]
    fn custom_template_end_to_end() {
        let template = PromptTemplate::Custom {
            turn_template: "<{role}>{content}</{role}>".into(),
            separator: "".into(),
            assistant_prefix: "<assistant>".into(),
        };
        let config = PromptConfig::new(2048, 256).with_template(template);
        let engine = PromptEngine::new(config);
        let result = engine.process_single("hi").unwrap();
        assert!(result.text.contains("<user>hi</user>"));
        assert!(result.text.contains("<assistant>"));
    }

    #[test]
    fn history_max_turns_preserves_system() {
        let mut h = ConversationHistory::new(2);
        h.add_system("sys");
        h.add_user("u1");
        h.add_user("u2"); // should evict u1
        assert_eq!(h.len(), 2);
        assert_eq!(h.messages()[0].role, "system");
        assert_eq!(h.messages()[1].content, "u2");
    }

    #[test]
    fn config_multiple_stop_sequences() {
        let c = PromptConfig::default()
            .with_stop_sequence("a")
            .with_stop_sequence("b")
            .with_stop_sequence("c");
        assert_eq!(c.stop_sequences.len(), 3);
    }

    #[test]
    fn truncator_middle_even_split() {
        let t = PromptTruncator::new(TruncationStrategy::Middle, 0);
        let tokens: Vec<String> = (0..10).map(|i| format!("t{i}")).collect();
        let (out, removed) = t.truncate(&tokens, 6);
        assert_eq!(out.len(), 6);
        assert_eq!(removed, 4);
        // First token preserved.
        assert_eq!(out[0], "t0");
        // Last token preserved.
        assert_eq!(out[5], "t9");
    }

    #[test]
    fn engine_empty_conversation_error() {
        let config = PromptConfig::new(2048, 256).with_template(PromptTemplate::Raw);
        let engine = PromptEngine::new(config);
        let result = engine.process_messages(&[]);
        // Empty conversation = empty prompt → validation error.
        assert!(result.is_err());
    }

    #[test]
    fn tokenizer_whitespace_normalisation() {
        let t = PromptTokenizer::default();
        let tokens = t.tokenize("  hello   world  ");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn spm_overwrite() {
        let mut m = SystemPromptManager::new("first");
        m.set("default", "second");
        assert_eq!(m.default_prompt(), Some("second"));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn alpaca_custom_role() {
        let s = PromptTemplate::Alpaca.format_turn("tool", "result");
        assert!(s.contains("### tool:"));
    }

    #[test]
    fn raw_assistant_prefix_empty() {
        assert_eq!(PromptTemplate::Raw.assistant_prefix(), "");
    }

    #[test]
    fn instruct_assistant_turn() {
        let s = PromptTemplate::Instruct.format_turn("assistant", "reply");
        assert_eq!(s, "reply\n\n");
    }

    #[test]
    fn history_add_system() {
        let mut h = ConversationHistory::default();
        h.add_system("sys");
        assert_eq!(h.messages()[0].role, "system");
    }

    #[test]
    fn engine_instruct_template_name() {
        let config = PromptConfig::new(2048, 256).with_template(PromptTemplate::Instruct);
        let engine = PromptEngine::new(config);
        let result = engine.process_single("test").unwrap();
        assert_eq!(result.metrics.template_name, "Instruct");
    }
}
