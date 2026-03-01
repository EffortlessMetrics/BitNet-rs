//! Module stub - implementation pending merge from feature branch
//! Tokenizer wrapper with chat templates and prompt formatting.
//!
//! Provides a unified interface for tokenization with support for multiple
//! tokenizer types (BPE, `WordPiece`, `SentencePiece`, Unigram, Character),
//! chat template formatting, prompt modes, and truncation strategies.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

// ── Tokenizer type ─────────────────────────────────────────────────────────

/// Supported tokenizer algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenizerType {
    /// Byte-Pair Encoding (GPT-2, `LLaMA`, etc.).
    Bpe,
    /// `WordPiece` (BERT-style).
    WordPiece,
    /// `SentencePiece` (T5, mBART, etc.).
    SentencePiece,
    /// Unigram (`XLNet`, ALBERT, etc.).
    Unigram,
    /// Character-level tokenization.
    Character,
}

impl fmt::Display for TokenizerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bpe => write!(f, "BPE"),
            Self::WordPiece => write!(f, "WordPiece"),
            Self::SentencePiece => write!(f, "SentencePiece"),
            Self::Unigram => write!(f, "Unigram"),
            Self::Character => write!(f, "Character"),
        }
    }
}

// ── Special tokens ─────────────────────────────────────────────────────────

/// Special token definitions for a tokenizer.
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    /// Beginning-of-sequence token string.
    pub bos_token: String,
    /// Beginning-of-sequence token ID.
    pub bos_id: u32,
    /// End-of-sequence token string.
    pub eos_token: String,
    /// End-of-sequence token ID.
    pub eos_id: u32,
    /// Padding token string.
    pub pad_token: String,
    /// Padding token ID.
    pub pad_id: u32,
    /// Unknown token string.
    pub unk_token: String,
    /// Unknown token ID.
    pub unk_id: u32,
    /// Mask token string (for masked-LM tasks).
    pub mask_token: String,
    /// Mask token ID.
    pub mask_id: u32,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos_token: "<s>".to_string(),
            bos_id: 1,
            eos_token: "</s>".to_string(),
            eos_id: 2,
            pad_token: "<pad>".to_string(),
            pad_id: 0,
            unk_token: "<unk>".to_string(),
            unk_id: 3,
            mask_token: "<mask>".to_string(),
            mask_id: 4,
        }
    }
}

impl SpecialTokens {
    /// Returns all special token IDs as a set for filtering.
    #[must_use]
    pub fn all_ids(&self) -> Vec<u32> {
        vec![self.bos_id, self.eos_id, self.pad_id, self.unk_id, self.mask_id]
    }

    /// Check whether a token ID is a special token.
    #[must_use]
    pub const fn is_special(&self, id: u32) -> bool {
        id == self.bos_id
            || id == self.eos_id
            || id == self.pad_id
            || id == self.unk_id
            || id == self.mask_id
    }
}

// ── Tokenizer config ───────────────────────────────────────────────────────

/// Configuration needed to initialise a tokenizer.
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    /// Path to the vocabulary file.
    pub vocab_path: PathBuf,
    /// Tokenizer algorithm type.
    pub model_type: TokenizerType,
    /// Special-token definitions.
    pub special_tokens: SpecialTokens,
    /// Whether to add BOS at the start of every encoding.
    pub add_bos: bool,
    /// Whether to add EOS at the end of every encoding.
    pub add_eos: bool,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            vocab_path: PathBuf::from("tokenizer.json"),
            model_type: TokenizerType::Bpe,
            special_tokens: SpecialTokens::default(),
            add_bos: true,
            add_eos: true,
        }
    }
}

// ── Vocabulary info ────────────────────────────────────────────────────────

/// Runtime vocabulary metadata and lookup tables.
#[derive(Debug, Clone)]
pub struct VocabularyInfo {
    /// Total vocabulary size (including special tokens).
    pub vocab_size: usize,
    /// Token string → ID.
    token_to_id: HashMap<String, u32>,
    /// ID → token string.
    id_to_token: HashMap<u32, String>,
}

impl VocabularyInfo {
    /// Build vocabulary from an iterator of (token, id) pairs.
    #[must_use]
    pub fn from_pairs(pairs: impl IntoIterator<Item = (String, u32)>) -> Self {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();
        for (tok, id) in pairs {
            id_to_token.insert(id, tok.clone());
            token_to_id.insert(tok, id);
        }
        let vocab_size = token_to_id.len();
        Self { vocab_size, token_to_id, id_to_token }
    }

    /// Look up the ID for a token string. Returns `None` if absent.
    #[must_use]
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Look up the token string for an ID. Returns `None` if absent.
    #[must_use]
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(String::as_str)
    }

    /// Check whether the vocabulary contains a token.
    #[must_use]
    pub fn contains_token(&self, token: &str) -> bool {
        self.token_to_id.contains_key(token)
    }

    /// Check whether the vocabulary contains an ID.
    #[must_use]
    pub fn contains_id(&self, id: u32) -> bool {
        self.id_to_token.contains_key(&id)
    }
}

// ── Truncation strategy ────────────────────────────────────────────────────

/// Strategy for truncating token sequences that exceed a maximum length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TruncationStrategy {
    /// Remove tokens from the left (keep the end).
    Left,
    /// Remove tokens from the right (keep the beginning).
    #[default]
    Right,
    /// When encoding a pair, truncate the longest sequence first.
    Longest,
    /// Do not truncate; return the full sequence.
    None,
}

impl TruncationStrategy {
    /// Apply this strategy to truncate `tokens` to at most `max_len`.
    #[must_use]
    pub fn apply(&self, tokens: &[u32], max_len: usize) -> Vec<u32> {
        if tokens.len() <= max_len || max_len == 0 {
            return tokens.to_vec();
        }
        match self {
            Self::Right | Self::Longest => tokens[..max_len].to_vec(),
            Self::Left => tokens[tokens.len() - max_len..].to_vec(),
            Self::None => tokens.to_vec(),
        }
    }
}

// ── Token encoder ──────────────────────────────────────────────────────────

/// Encodes text into token IDs using a vocabulary and optional pre-processing.
#[derive(Debug, Clone)]
pub struct TokenEncoder {
    vocab: VocabularyInfo,
    special_tokens: SpecialTokens,
    add_bos: bool,
    add_eos: bool,
    normalise_whitespace: bool,
}

impl TokenEncoder {
    /// Create a new encoder.
    #[must_use]
    #[allow(clippy::similar_names)]
    pub const fn new(
        vocab: VocabularyInfo,
        special_tokens: SpecialTokens,
        add_bos: bool,
        add_eos: bool,
    ) -> Self {
        Self { vocab, special_tokens, add_bos, add_eos, normalise_whitespace: true }
    }

    /// Enable or disable whitespace normalisation.
    #[must_use]
    pub const fn with_normalise_whitespace(mut self, normalise: bool) -> Self {
        self.normalise_whitespace = normalise;
        self
    }

    /// Normalise input text (collapse whitespace, trim).
    fn normalise(&self, text: &str) -> String {
        if !self.normalise_whitespace {
            return text.to_string();
        }
        let collapsed: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
        collapsed.trim().to_string()
    }

    /// Pre-tokenise text into word-level tokens.
    fn pre_tokenise(text: &str) -> Vec<String> {
        if text.is_empty() {
            return vec![];
        }
        text.split_whitespace().map(String::from).collect()
    }

    /// Encode a single word into token IDs via greedy longest-match.
    fn encode_word(&self, word: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let chars: Vec<char> = word.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let mut best_len = 0;
            let mut best_id = None;
            // Greedy longest match.
            for end in (i + 1..=chars.len()).rev() {
                let substr: String = chars[i..end].iter().collect();
                if let Some(id) = self.vocab.token_to_id(&substr) {
                    best_len = end - i;
                    best_id = Some(id);
                    break;
                }
            }
            if let Some(id) = best_id {
                ids.push(id);
                i += best_len;
            } else {
                // Fall back to UNK for unknown characters.
                ids.push(self.special_tokens.unk_id);
                i += 1;
            }
        }
        ids
    }

    /// Encode text into token IDs.
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let normalised = self.normalise(text);
        let words = Self::pre_tokenise(&normalised);
        let mut ids = Vec::new();
        if self.add_bos {
            ids.push(self.special_tokens.bos_id);
        }
        for word in &words {
            ids.extend(self.encode_word(word));
        }
        if self.add_eos {
            ids.push(self.special_tokens.eos_id);
        }
        ids
    }

    /// Encode text then apply truncation.
    #[must_use]
    pub fn encode_truncated(
        &self,
        text: &str,
        max_len: usize,
        strategy: TruncationStrategy,
    ) -> Vec<u32> {
        let ids = self.encode(text);
        strategy.apply(&ids, max_len)
    }

    /// Encode and pad to exactly `target_len` using the pad token.
    #[must_use]
    pub fn encode_padded(&self, text: &str, target_len: usize) -> Vec<u32> {
        let mut ids = self.encode(text);
        while ids.len() < target_len {
            ids.push(self.special_tokens.pad_id);
        }
        ids
    }

    /// Batch-encode multiple texts.
    #[must_use]
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        texts.iter().map(|t| self.encode(t)).collect()
    }

    /// Reference to the vocabulary.
    #[must_use]
    pub const fn vocab(&self) -> &VocabularyInfo {
        &self.vocab
    }
}

// ── Token decoder ──────────────────────────────────────────────────────────

/// Decodes token IDs back into text.
#[derive(Debug, Clone)]
pub struct TokenDecoder {
    vocab: VocabularyInfo,
    special_tokens: SpecialTokens,
    skip_special_tokens: bool,
}

impl TokenDecoder {
    /// Create a new decoder.
    #[must_use]
    pub const fn new(
        vocab: VocabularyInfo,
        special_tokens: SpecialTokens,
        skip_special_tokens: bool,
    ) -> Self {
        Self { vocab, special_tokens, skip_special_tokens }
    }

    /// Decode a sequence of token IDs into a string.
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut parts: Vec<String> = Vec::new();
        for &id in ids {
            if self.skip_special_tokens && self.special_tokens.is_special(id) {
                continue;
            }
            if let Some(tok) = self.vocab.id_to_token(id) {
                parts.push(tok.to_string());
            } else {
                // Byte-fallback: emit replacement character for unknown IDs.
                parts.push("\u{FFFD}".to_string());
            }
        }
        parts.join(" ")
    }

    /// Decode with special tokens included.
    #[must_use]
    pub fn decode_raw(&self, ids: &[u32]) -> String {
        let mut parts: Vec<String> = Vec::new();
        for &id in ids {
            if let Some(tok) = self.vocab.id_to_token(id) {
                parts.push(tok.to_string());
            } else {
                parts.push("\u{FFFD}".to_string());
            }
        }
        parts.join(" ")
    }

    /// Decode a batch of token ID sequences.
    #[must_use]
    pub fn decode_batch(&self, batch: &[Vec<u32>]) -> Vec<String> {
        batch.iter().map(|ids| self.decode(ids)).collect()
    }
}

// ── Chat template ──────────────────────────────────────────────────────────

/// Role in a chat conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChatRole {
    /// System prompt.
    System,
    /// User message.
    User,
    /// Assistant response.
    Assistant,
}

impl fmt::Display for ChatRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::System => write!(f, "system"),
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
        }
    }
}

/// A single message in a chat conversation.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// Role of the message author.
    pub role: ChatRole,
    /// Content of the message.
    pub content: String,
}

impl ChatMessage {
    /// Create a new chat message.
    #[must_use]
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self { role, content: content.into() }
    }
}

/// Applies chat template formatting to conversations.
#[derive(Debug, Clone)]
pub struct ChatTemplate {
    /// Per-role prefix strings.
    role_prefixes: HashMap<ChatRole, String>,
    /// Per-role suffix strings.
    role_suffixes: HashMap<ChatRole, String>,
    /// Separator between messages.
    message_separator: String,
    /// String appended at the very end to prompt the assistant.
    generation_prompt: String,
}

impl Default for ChatTemplate {
    fn default() -> Self {
        let mut prefixes = HashMap::new();
        prefixes.insert(ChatRole::System, "<|system|>\n".to_string());
        prefixes.insert(ChatRole::User, "<|user|>\n".to_string());
        prefixes.insert(ChatRole::Assistant, "<|assistant|>\n".to_string());

        let mut suffixes = HashMap::new();
        suffixes.insert(ChatRole::System, "<|end|>\n".to_string());
        suffixes.insert(ChatRole::User, "<|end|>\n".to_string());
        suffixes.insert(ChatRole::Assistant, "<|end|>\n".to_string());

        Self {
            role_prefixes: prefixes,
            role_suffixes: suffixes,
            message_separator: String::new(),
            generation_prompt: "<|assistant|>\n".to_string(),
        }
    }
}

impl ChatTemplate {
    /// Create a new chat template with custom role formatting.
    #[must_use]
    pub const fn new(
        role_prefixes: HashMap<ChatRole, String>,
        role_suffixes: HashMap<ChatRole, String>,
        message_separator: String,
        generation_prompt: String,
    ) -> Self {
        Self { role_prefixes, role_suffixes, message_separator, generation_prompt }
    }

    /// Create a LLaMA-2-style chat template.
    #[must_use]
    pub fn llama2() -> Self {
        let mut prefixes = HashMap::new();
        prefixes.insert(ChatRole::System, "[INST] <<SYS>>\n".to_string());
        prefixes.insert(ChatRole::User, "[INST] ".to_string());
        prefixes.insert(ChatRole::Assistant, String::new());

        let mut suffixes = HashMap::new();
        suffixes.insert(ChatRole::System, "\n<</SYS>>\n\n".to_string());
        suffixes.insert(ChatRole::User, " [/INST] ".to_string());
        suffixes.insert(ChatRole::Assistant, " </s>".to_string());

        Self {
            role_prefixes: prefixes,
            role_suffixes: suffixes,
            message_separator: String::new(),
            generation_prompt: String::new(),
        }
    }

    /// Create a ChatML-style template.
    #[must_use]
    pub fn chatml() -> Self {
        let mut prefixes = HashMap::new();
        prefixes.insert(ChatRole::System, "<|im_start|>system\n".to_string());
        prefixes.insert(ChatRole::User, "<|im_start|>user\n".to_string());
        prefixes.insert(ChatRole::Assistant, "<|im_start|>assistant\n".to_string());

        let mut suffixes = HashMap::new();
        let end = "<|im_end|>\n".to_string();
        suffixes.insert(ChatRole::System, end.clone());
        suffixes.insert(ChatRole::User, end.clone());
        suffixes.insert(ChatRole::Assistant, end);

        Self {
            role_prefixes: prefixes,
            role_suffixes: suffixes,
            message_separator: String::new(),
            generation_prompt: "<|im_start|>assistant\n".to_string(),
        }
    }

    /// Format a single message.
    fn format_message(&self, msg: &ChatMessage) -> String {
        let prefix = self.role_prefixes.get(&msg.role).cloned().unwrap_or_default();
        let suffix = self.role_suffixes.get(&msg.role).cloned().unwrap_or_default();
        format!("{prefix}{}{suffix}", msg.content)
    }

    /// Apply the template to a full conversation.
    ///
    /// If `add_generation_prompt` is `true`, appends the generation prompt at
    /// the end to cue the model for a response.
    #[must_use]
    pub fn apply(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> String {
        let formatted: Vec<String> = messages.iter().map(|m| self.format_message(m)).collect();
        let mut result = formatted.join(&self.message_separator);
        if add_generation_prompt {
            result.push_str(&self.generation_prompt);
        }
        result
    }
}

// ── Prompt formatter ───────────────────────────────────────────────────────

/// Prompt formatting mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PromptMode {
    /// Instruction-following format.
    Instruct,
    /// Multi-turn chat format.
    Chat,
    /// Raw completion (no formatting).
    Completion,
}

impl fmt::Display for PromptMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Instruct => write!(f, "instruct"),
            Self::Chat => write!(f, "chat"),
            Self::Completion => write!(f, "completion"),
        }
    }
}

/// Formats prompts for different generation modes.
#[derive(Debug, Clone)]
pub struct PromptFormatter {
    /// System prompt used in instruct and chat modes.
    system_prompt: String,
    /// Instruction prefix for instruct mode.
    instruct_prefix: String,
    /// Instruction suffix for instruct mode.
    instruct_suffix: String,
    /// Chat template for chat mode.
    chat_template: ChatTemplate,
}

impl Default for PromptFormatter {
    fn default() -> Self {
        Self {
            system_prompt: "You are a helpful assistant.".to_string(),
            instruct_prefix: "[INST] ".to_string(),
            instruct_suffix: " [/INST]".to_string(),
            chat_template: ChatTemplate::default(),
        }
    }
}

impl PromptFormatter {
    /// Create a formatter with a custom system prompt.
    #[must_use]
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Create a formatter with a custom chat template.
    #[must_use]
    pub fn with_chat_template(mut self, template: ChatTemplate) -> Self {
        self.chat_template = template;
        self
    }

    /// Create a formatter with custom instruct affixes.
    #[must_use]
    pub fn with_instruct_affixes(
        mut self,
        prefix: impl Into<String>,
        suffix: impl Into<String>,
    ) -> Self {
        self.instruct_prefix = prefix.into();
        self.instruct_suffix = suffix.into();
        self
    }

    /// Format a single user query in the given mode.
    #[must_use]
    pub fn format(&self, text: &str, mode: PromptMode) -> String {
        match mode {
            PromptMode::Completion => text.to_string(),
            PromptMode::Instruct => {
                format!("{}{text}{}", self.instruct_prefix, self.instruct_suffix)
            }
            PromptMode::Chat => {
                let messages = vec![
                    ChatMessage::new(ChatRole::System, &self.system_prompt),
                    ChatMessage::new(ChatRole::User, text),
                ];
                self.chat_template.apply(&messages, true)
            }
        }
    }

    /// Format a multi-turn conversation.
    #[must_use]
    pub fn format_conversation(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> String {
        self.chat_template.apply(messages, add_generation_prompt)
    }
}

// ── Tokenizer wrapper ──────────────────────────────────────────────────────

/// Top-level tokenizer wrapper: configuration → encoder/decoder → template.
///
/// Provides a single entry point for loading a tokenizer configuration,
/// building the vocabulary, and performing encode/decode with optional
/// chat template formatting.
#[derive(Debug, Clone)]
pub struct TokenizerWrapper {
    config: TokenizerConfig,
    encoder: TokenEncoder,
    decoder: TokenDecoder,
    prompt_formatter: PromptFormatter,
    truncation: TruncationStrategy,
    max_length: Option<usize>,
}

impl TokenizerWrapper {
    /// Build a tokenizer wrapper from config and vocabulary pairs.
    #[must_use]
    pub fn new(
        config: TokenizerConfig,
        vocab_pairs: impl IntoIterator<Item = (String, u32)>,
    ) -> Self {
        let vocab = VocabularyInfo::from_pairs(vocab_pairs);
        let encoder = TokenEncoder::new(
            vocab.clone(),
            config.special_tokens.clone(),
            config.add_bos,
            config.add_eos,
        );
        let decoder = TokenDecoder::new(vocab, config.special_tokens.clone(), true);
        Self {
            config,
            encoder,
            decoder,
            prompt_formatter: PromptFormatter::default(),
            truncation: TruncationStrategy::Right,
            max_length: None,
        }
    }

    /// Set the truncation strategy.
    #[must_use]
    pub const fn with_truncation(mut self, strategy: TruncationStrategy) -> Self {
        self.truncation = strategy;
        self
    }

    /// Set the maximum sequence length.
    #[must_use]
    pub const fn with_max_length(mut self, max_len: usize) -> Self {
        self.max_length = Some(max_len);
        self
    }

    /// Set the prompt formatter.
    #[must_use]
    pub fn with_prompt_formatter(mut self, formatter: PromptFormatter) -> Self {
        self.prompt_formatter = formatter;
        self
    }

    /// Encode text into token IDs, applying truncation if configured.
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let ids = self.encoder.encode(text);
        if let Some(max_len) = self.max_length { self.truncation.apply(&ids, max_len) } else { ids }
    }

    /// Encode with explicit truncation parameters.
    #[must_use]
    pub fn encode_with_truncation(
        &self,
        text: &str,
        max_len: usize,
        strategy: TruncationStrategy,
    ) -> Vec<u32> {
        self.encoder.encode_truncated(text, max_len, strategy)
    }

    /// Encode and pad to a target length.
    #[must_use]
    pub fn encode_padded(&self, text: &str, target_len: usize) -> Vec<u32> {
        self.encoder.encode_padded(text, target_len)
    }

    /// Decode token IDs to text.
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        self.decoder.decode(ids)
    }

    /// Decode without skipping special tokens.
    #[must_use]
    pub fn decode_raw(&self, ids: &[u32]) -> String {
        self.decoder.decode_raw(ids)
    }

    /// Format a prompt using the configured formatter.
    #[must_use]
    pub fn format_prompt(&self, text: &str, mode: PromptMode) -> String {
        self.prompt_formatter.format(text, mode)
    }

    /// Encode a formatted prompt.
    #[must_use]
    pub fn encode_prompt(&self, text: &str, mode: PromptMode) -> Vec<u32> {
        let formatted = self.format_prompt(text, mode);
        self.encode(&formatted)
    }

    /// Format and encode a multi-turn conversation.
    #[must_use]
    pub fn encode_conversation(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Vec<u32> {
        let formatted = self.prompt_formatter.format_conversation(messages, add_generation_prompt);
        self.encode(&formatted)
    }

    /// Batch-encode multiple texts.
    #[must_use]
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        texts.iter().map(|t| self.encode(t)).collect()
    }

    /// Reference to the configuration.
    #[must_use]
    pub const fn config(&self) -> &TokenizerConfig {
        &self.config
    }

    /// Reference to the encoder.
    #[must_use]
    pub const fn encoder(&self) -> &TokenEncoder {
        &self.encoder
    }

    /// Reference to the decoder.
    #[must_use]
    pub const fn decoder(&self) -> &TokenDecoder {
        &self.decoder
    }

    /// Reference to the prompt formatter.
    #[must_use]
    pub const fn prompt_formatter(&self) -> &PromptFormatter {
        &self.prompt_formatter
    }

    /// Vocabulary size.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.encoder.vocab().vocab_size
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test helpers ────────────────────────────────────────────────────

    /// Build a small vocabulary for tests. Includes special tokens plus
    /// a handful of real words and subwords.
    fn test_vocab() -> Vec<(String, u32)> {
        vec![
            ("<pad>".into(), 0),
            ("<s>".into(), 1),
            ("</s>".into(), 2),
            ("<unk>".into(), 3),
            ("<mask>".into(), 4),
            ("hello".into(), 10),
            ("world".into(), 11),
            ("the".into(), 12),
            ("cat".into(), 13),
            ("sat".into(), 14),
            ("on".into(), 15),
            ("mat".into(), 16),
            ("a".into(), 17),
            ("is".into(), 18),
            ("good".into(), 19),
            ("day".into(), 20),
            ("foo".into(), 21),
            ("bar".into(), 22),
            ("baz".into(), 23),
            ("qu".into(), 24),
            ("ick".into(), 25),
            ("brown".into(), 26),
            ("fox".into(), 27),
            ("jumps".into(), 28),
            ("over".into(), 29),
            ("lazy".into(), 30),
            ("dog".into(), 31),
            ("test".into(), 32),
            ("ing".into(), 33),
            ("ed".into(), 34),
            ("un".into(), 35),
            ("re".into(), 36),
            ("in".into(), 37),
            ("b".into(), 38),
            ("c".into(), 39),
            ("d".into(), 40),
            ("e".into(), 41),
            ("f".into(), 42),
            ("g".into(), 43),
            ("h".into(), 44),
            ("i".into(), 45),
            ("j".into(), 46),
            ("k".into(), 47),
            ("l".into(), 48),
            ("m".into(), 49),
            ("n".into(), 50),
            ("o".into(), 51),
            ("p".into(), 52),
            ("r".into(), 53),
            ("s".into(), 54),
            ("t".into(), 55),
            ("u".into(), 56),
            ("v".into(), 57),
            ("w".into(), 58),
            ("x".into(), 59),
            ("y".into(), 60),
            ("z".into(), 61),
        ]
    }

    fn test_encoder() -> TokenEncoder {
        TokenEncoder::new(
            VocabularyInfo::from_pairs(test_vocab()),
            SpecialTokens::default(),
            true,
            true,
        )
    }

    fn test_decoder() -> TokenDecoder {
        TokenDecoder::new(VocabularyInfo::from_pairs(test_vocab()), SpecialTokens::default(), true)
    }

    fn test_wrapper() -> TokenizerWrapper {
        TokenizerWrapper::new(TokenizerConfig::default(), test_vocab())
    }

    // ── TokenizerType tests ────────────────────────────────────────────

    #[test]
    fn tokenizer_type_display() {
        assert_eq!(TokenizerType::Bpe.to_string(), "BPE");
        assert_eq!(TokenizerType::WordPiece.to_string(), "WordPiece");
        assert_eq!(TokenizerType::SentencePiece.to_string(), "SentencePiece");
        assert_eq!(TokenizerType::Unigram.to_string(), "Unigram");
        assert_eq!(TokenizerType::Character.to_string(), "Character");
    }

    #[test]
    fn tokenizer_type_eq() {
        assert_eq!(TokenizerType::Bpe, TokenizerType::Bpe);
        assert_ne!(TokenizerType::Bpe, TokenizerType::WordPiece);
    }

    #[test]
    fn tokenizer_type_clone() {
        let t = TokenizerType::SentencePiece;
        let t2 = t;
        assert_eq!(t, t2);
    }

    #[test]
    fn tokenizer_type_hash() {
        let mut map = HashMap::new();
        map.insert(TokenizerType::Bpe, "bpe");
        map.insert(TokenizerType::Unigram, "uni");
        assert_eq!(map[&TokenizerType::Bpe], "bpe");
    }

    // ── SpecialTokens tests ────────────────────────────────────────────

    #[test]
    fn special_tokens_default() {
        let st = SpecialTokens::default();
        assert_eq!(st.bos_token, "<s>");
        assert_eq!(st.eos_token, "</s>");
        assert_eq!(st.pad_token, "<pad>");
        assert_eq!(st.unk_token, "<unk>");
        assert_eq!(st.mask_token, "<mask>");
    }

    #[test]
    fn special_tokens_ids() {
        let st = SpecialTokens::default();
        assert_eq!(st.bos_id, 1);
        assert_eq!(st.eos_id, 2);
        assert_eq!(st.pad_id, 0);
        assert_eq!(st.unk_id, 3);
        assert_eq!(st.mask_id, 4);
    }

    #[test]
    fn special_tokens_all_ids() {
        let st = SpecialTokens::default();
        let ids = st.all_ids();
        assert_eq!(ids.len(), 5);
        assert!(ids.contains(&0));
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));
        assert!(ids.contains(&4));
    }

    #[test]
    fn special_tokens_is_special() {
        let st = SpecialTokens::default();
        assert!(st.is_special(0)); // pad
        assert!(st.is_special(1)); // bos
        assert!(st.is_special(2)); // eos
        assert!(st.is_special(3)); // unk
        assert!(st.is_special(4)); // mask
        assert!(!st.is_special(10));
        assert!(!st.is_special(999));
    }

    #[test]
    fn special_tokens_custom() {
        let st = SpecialTokens {
            bos_token: "[CLS]".into(),
            bos_id: 101,
            eos_token: "[SEP]".into(),
            eos_id: 102,
            pad_token: "[PAD]".into(),
            pad_id: 0,
            unk_token: "[UNK]".into(),
            unk_id: 100,
            mask_token: "[MASK]".into(),
            mask_id: 103,
        };
        assert!(st.is_special(101));
        assert!(st.is_special(102));
        assert!(!st.is_special(1));
    }

    // ── TokenizerConfig tests ──────────────────────────────────────────

    #[test]
    fn config_default() {
        let cfg = TokenizerConfig::default();
        assert_eq!(cfg.model_type, TokenizerType::Bpe);
        assert!(cfg.add_bos);
        assert!(cfg.add_eos);
        assert_eq!(cfg.vocab_path, PathBuf::from("tokenizer.json"));
    }

    #[test]
    fn config_custom() {
        let cfg = TokenizerConfig {
            vocab_path: PathBuf::from("/tmp/vocab.json"),
            model_type: TokenizerType::WordPiece,
            special_tokens: SpecialTokens::default(),
            add_bos: false,
            add_eos: true,
        };
        assert_eq!(cfg.model_type, TokenizerType::WordPiece);
        assert!(!cfg.add_bos);
    }

    // ── VocabularyInfo tests ───────────────────────────────────────────

    #[test]
    fn vocab_from_pairs() {
        let vocab = VocabularyInfo::from_pairs(test_vocab());
        assert!(vocab.vocab_size > 0);
    }

    #[test]
    fn vocab_token_to_id() {
        let vocab = VocabularyInfo::from_pairs(test_vocab());
        assert_eq!(vocab.token_to_id("hello"), Some(10));
        assert_eq!(vocab.token_to_id("world"), Some(11));
        assert_eq!(vocab.token_to_id("<s>"), Some(1));
    }

    #[test]
    fn vocab_id_to_token() {
        let vocab = VocabularyInfo::from_pairs(test_vocab());
        assert_eq!(vocab.id_to_token(10), Some("hello"));
        assert_eq!(vocab.id_to_token(11), Some("world"));
        assert_eq!(vocab.id_to_token(1), Some("<s>"));
    }

    #[test]
    fn vocab_unknown_token() {
        let vocab = VocabularyInfo::from_pairs(test_vocab());
        assert_eq!(vocab.token_to_id("nonexistent"), None);
    }

    #[test]
    fn vocab_unknown_id() {
        let vocab = VocabularyInfo::from_pairs(test_vocab());
        assert_eq!(vocab.id_to_token(9999), None);
    }

    #[test]
    fn vocab_contains() {
        let vocab = VocabularyInfo::from_pairs(test_vocab());
        assert!(vocab.contains_token("hello"));
        assert!(!vocab.contains_token("nonexistent"));
        assert!(vocab.contains_id(10));
        assert!(!vocab.contains_id(9999));
    }

    #[test]
    fn vocab_empty() {
        let vocab = VocabularyInfo::from_pairs(std::iter::empty());
        assert_eq!(vocab.vocab_size, 0);
        assert_eq!(vocab.token_to_id("hello"), None);
        assert_eq!(vocab.id_to_token(0), None);
    }

    // ── TruncationStrategy tests ───────────────────────────────────────

    #[test]
    fn truncation_right() {
        let tokens = vec![1, 2, 3, 4, 5];
        let result = TruncationStrategy::Right.apply(&tokens, 3);
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn truncation_left() {
        let tokens = vec![1, 2, 3, 4, 5];
        let result = TruncationStrategy::Left.apply(&tokens, 3);
        assert_eq!(result, vec![3, 4, 5]);
    }

    #[test]
    fn truncation_none() {
        let tokens = vec![1, 2, 3, 4, 5];
        let result = TruncationStrategy::None.apply(&tokens, 3);
        assert_eq!(result, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn truncation_longest_acts_as_right() {
        let tokens = vec![1, 2, 3, 4, 5];
        let result = TruncationStrategy::Longest.apply(&tokens, 3);
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn truncation_no_op_when_shorter() {
        let tokens = vec![1, 2];
        let result = TruncationStrategy::Right.apply(&tokens, 5);
        assert_eq!(result, vec![1, 2]);
    }

    #[test]
    fn truncation_exact_length() {
        let tokens = vec![1, 2, 3];
        let result = TruncationStrategy::Right.apply(&tokens, 3);
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn truncation_to_one() {
        let tokens = vec![1, 2, 3, 4, 5];
        assert_eq!(TruncationStrategy::Right.apply(&tokens, 1), vec![1]);
        assert_eq!(TruncationStrategy::Left.apply(&tokens, 1), vec![5]);
    }

    #[test]
    fn truncation_empty_input() {
        let tokens: Vec<u32> = vec![];
        assert_eq!(TruncationStrategy::Right.apply(&tokens, 5), Vec::<u32>::new());
    }

    #[test]
    fn truncation_default_is_right() {
        assert_eq!(TruncationStrategy::default(), TruncationStrategy::Right);
    }

    #[test]
    fn truncation_max_len_zero() {
        let tokens = vec![1, 2, 3];
        // max_len == 0 is treated as "no truncation" (guard clause).
        let result = TruncationStrategy::Right.apply(&tokens, 0);
        assert_eq!(result, vec![1, 2, 3]);
    }

    // ── TokenEncoder tests ─────────────────────────────────────────────

    #[test]
    fn encode_known_words() {
        let enc = test_encoder();
        let ids = enc.encode("hello world");
        // BOS + hello + world + EOS
        assert_eq!(ids[0], 1); // BOS
        assert_eq!(ids[1], 10); // hello
        assert_eq!(ids[2], 11); // world
        assert_eq!(ids[3], 2); // EOS
    }

    #[test]
    fn encode_bpe_subword_split() {
        let enc = test_encoder();
        // "testing" should split: "test" (32) + "ing" (33)
        let ids = enc.encode("testing");
        assert_eq!(ids[0], 1); // BOS
        assert_eq!(ids[1], 32); // test
        assert_eq!(ids[2], 33); // ing
        assert_eq!(ids[3], 2); // EOS
    }

    #[test]
    fn encode_adds_bos_eos() {
        let enc = test_encoder();
        let ids = enc.encode("hello");
        assert_eq!(*ids.first().unwrap(), 1);
        assert_eq!(*ids.last().unwrap(), 2);
    }

    #[test]
    fn encode_no_bos_eos() {
        let enc = TokenEncoder::new(
            VocabularyInfo::from_pairs(test_vocab()),
            SpecialTokens::default(),
            false,
            false,
        );
        let ids = enc.encode("hello");
        assert_eq!(ids, vec![10]);
    }

    #[test]
    fn encode_only_bos() {
        let enc = TokenEncoder::new(
            VocabularyInfo::from_pairs(test_vocab()),
            SpecialTokens::default(),
            true,
            false,
        );
        let ids = enc.encode("hello");
        assert_eq!(ids, vec![1, 10]);
    }

    #[test]
    fn encode_only_eos() {
        let enc = TokenEncoder::new(
            VocabularyInfo::from_pairs(test_vocab()),
            SpecialTokens::default(),
            false,
            true,
        );
        let ids = enc.encode("hello");
        assert_eq!(ids, vec![10, 2]);
    }

    #[test]
    fn encode_unknown_falls_back_to_unk() {
        let enc = test_encoder();
        // Digits are not in our vocab, so each character → UNK.
        let ids = enc.encode("123");
        // BOS + 3× UNK + EOS
        assert_eq!(ids.len(), 5);
        assert_eq!(ids[1], 3); // UNK
        assert_eq!(ids[2], 3);
        assert_eq!(ids[3], 3);
    }

    #[test]
    fn encode_empty_string() {
        let enc = test_encoder();
        let ids = enc.encode("");
        // BOS + EOS only.
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn encode_whitespace_only() {
        let enc = test_encoder();
        let ids = enc.encode("   ");
        // After normalisation → empty → BOS + EOS.
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn encode_normalise_whitespace() {
        let enc = test_encoder();
        let ids1 = enc.encode("hello   world");
        let ids2 = enc.encode("hello world");
        assert_eq!(ids1, ids2);
    }

    #[test]
    fn encode_disable_normalisation() {
        let enc = TokenEncoder::new(
            VocabularyInfo::from_pairs(test_vocab()),
            SpecialTokens::default(),
            false,
            false,
        )
        .with_normalise_whitespace(false);
        // With normalisation off, leading space is preserved in pre-tokenise.
        let ids_norm = enc.encode("hello");
        let ids_space = enc.encode(" hello");
        // " hello" gets pre-tokenised by split_whitespace → ["hello"] anyway,
        // but the normalise path is different.
        assert_eq!(ids_norm, ids_space);
    }

    #[test]
    fn encode_single_character() {
        let enc = test_encoder();
        let ids = enc.encode("a");
        // BOS + "a" (17) + EOS
        assert_eq!(ids, vec![1, 17, 2]);
    }

    #[test]
    fn encode_multiple_sentences() {
        let enc = test_encoder();
        let ids = enc.encode("hello world the cat");
        assert_eq!(ids.len(), 6); // BOS + 4 words + EOS
    }

    #[test]
    fn encode_truncated_right() {
        let enc = test_encoder();
        let ids = enc.encode_truncated("hello world the cat", 4, TruncationStrategy::Right);
        assert_eq!(ids.len(), 4);
        assert_eq!(ids[0], 1); // BOS preserved
    }

    #[test]
    fn encode_truncated_left() {
        let enc = test_encoder();
        let ids = enc.encode_truncated("hello world the cat", 4, TruncationStrategy::Left);
        assert_eq!(ids.len(), 4);
        assert_eq!(*ids.last().unwrap(), 2); // EOS preserved
    }

    #[test]
    fn encode_padded() {
        let enc = test_encoder();
        let ids = enc.encode_padded("hello", 8);
        assert_eq!(ids.len(), 8);
        // Last entries are PAD (0).
        assert_eq!(ids[3], 0);
        assert_eq!(ids[7], 0);
    }

    #[test]
    fn encode_padded_no_extra_when_exact() {
        let enc = test_encoder();
        let ids = enc.encode_padded("hello", 3);
        // "hello" → BOS+hello+EOS = 3 tokens → no padding needed.
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn encode_padded_shorter_than_encoded() {
        let enc = test_encoder();
        // Target shorter than encoded length → no truncation, just no padding.
        let ids = enc.encode_padded("hello world", 2);
        // 4 tokens, target 2 → still 4 (padding only adds, never removes).
        assert_eq!(ids.len(), 4);
    }

    #[test]
    fn encode_batch_consistency() {
        let enc = test_encoder();
        let batch = enc.encode_batch(&["hello", "world", "hello"]);
        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0], batch[2]); // Same input → same output.
    }

    #[test]
    fn encode_batch_empty() {
        let enc = test_encoder();
        let batch = enc.encode_batch(&[]);
        assert!(batch.is_empty());
    }

    #[test]
    fn encode_batch_single() {
        let enc = test_encoder();
        let batch = enc.encode_batch(&["hello"]);
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0], enc.encode("hello"));
    }

    // ── TokenDecoder tests ─────────────────────────────────────────────

    #[test]
    fn decode_basic() {
        let dec = test_decoder();
        let text = dec.decode(&[10, 11]);
        assert_eq!(text, "hello world");
    }

    #[test]
    fn decode_skips_special_tokens() {
        let dec = test_decoder();
        let text = dec.decode(&[1, 10, 11, 2]);
        assert_eq!(text, "hello world");
    }

    #[test]
    fn decode_raw_includes_special() {
        let dec = test_decoder();
        let text = dec.decode_raw(&[1, 10, 11, 2]);
        assert_eq!(text, "<s> hello world </s>");
    }

    #[test]
    fn decode_unknown_id_replacement() {
        let dec = test_decoder();
        let text = dec.decode(&[10, 9999, 11]);
        assert!(text.contains('\u{FFFD}'));
    }

    #[test]
    fn decode_empty() {
        let dec = test_decoder();
        assert_eq!(dec.decode(&[]), "");
    }

    #[test]
    fn decode_only_special_tokens() {
        let dec = test_decoder();
        assert_eq!(dec.decode(&[1, 2, 0, 3, 4]), "");
    }

    #[test]
    fn decode_no_skip_special() {
        let dec = TokenDecoder::new(
            VocabularyInfo::from_pairs(test_vocab()),
            SpecialTokens::default(),
            false,
        );
        let text = dec.decode(&[1, 10, 2]);
        assert!(text.contains("<s>"));
        assert!(text.contains("</s>"));
    }

    #[test]
    fn decode_batch() {
        let dec = test_decoder();
        let texts = dec.decode_batch(&[vec![10], vec![11]]);
        assert_eq!(texts, vec!["hello", "world"]);
    }

    #[test]
    fn decode_batch_empty() {
        let dec = test_decoder();
        let texts = dec.decode_batch(&[]);
        assert!(texts.is_empty());
    }

    #[test]
    fn encode_decode_roundtrip() {
        let enc = test_encoder();
        let dec = TokenDecoder::new(
            VocabularyInfo::from_pairs(test_vocab()),
            SpecialTokens::default(),
            true,
        );
        let original = "hello world";
        let ids = enc.encode(original);
        let decoded = dec.decode(&ids);
        assert_eq!(decoded, original);
    }

    // ── ChatRole tests ─────────────────────────────────────────────────

    #[test]
    fn chat_role_display() {
        assert_eq!(ChatRole::System.to_string(), "system");
        assert_eq!(ChatRole::User.to_string(), "user");
        assert_eq!(ChatRole::Assistant.to_string(), "assistant");
    }

    #[test]
    fn chat_role_eq() {
        assert_eq!(ChatRole::System, ChatRole::System);
        assert_ne!(ChatRole::System, ChatRole::User);
    }

    // ── ChatMessage tests ──────────────────────────────────────────────

    #[test]
    fn chat_message_new() {
        let msg = ChatMessage::new(ChatRole::User, "hello");
        assert_eq!(msg.role, ChatRole::User);
        assert_eq!(msg.content, "hello");
    }

    #[test]
    fn chat_message_from_string() {
        let msg = ChatMessage::new(ChatRole::System, String::from("system prompt"));
        assert_eq!(msg.content, "system prompt");
    }

    // ── ChatTemplate tests ─────────────────────────────────────────────

    #[test]
    fn chat_template_default_system_user() {
        let tmpl = ChatTemplate::default();
        let messages = vec![
            ChatMessage::new(ChatRole::System, "You are helpful."),
            ChatMessage::new(ChatRole::User, "Hi!"),
        ];
        let result = tmpl.apply(&messages, true);
        assert!(result.contains("<|system|>"));
        assert!(result.contains("You are helpful."));
        assert!(result.contains("<|user|>"));
        assert!(result.contains("Hi!"));
        assert!(result.contains("<|assistant|>"));
    }

    #[test]
    fn chat_template_default_no_gen_prompt() {
        let tmpl = ChatTemplate::default();
        let messages = vec![ChatMessage::new(ChatRole::User, "Hi!")];
        let result = tmpl.apply(&messages, false);
        // Should NOT end with assistant prompt.
        assert!(result.ends_with("<|end|>\n"));
    }

    #[test]
    fn chat_template_system_user_assistant() {
        let tmpl = ChatTemplate::default();
        let messages = vec![
            ChatMessage::new(ChatRole::System, "System."),
            ChatMessage::new(ChatRole::User, "Question?"),
            ChatMessage::new(ChatRole::Assistant, "Answer."),
        ];
        let result = tmpl.apply(&messages, false);
        assert!(result.contains("System."));
        assert!(result.contains("Question?"));
        assert!(result.contains("Answer."));
    }

    #[test]
    fn chat_template_llama2() {
        let tmpl = ChatTemplate::llama2();
        let messages = vec![
            ChatMessage::new(ChatRole::System, "Be concise."),
            ChatMessage::new(ChatRole::User, "Hello"),
        ];
        let result = tmpl.apply(&messages, false);
        assert!(result.contains("<<SYS>>"));
        assert!(result.contains("Be concise."));
        assert!(result.contains("[INST]"));
        assert!(result.contains("[/INST]"));
    }

    #[test]
    fn chat_template_chatml() {
        let tmpl = ChatTemplate::chatml();
        let messages = vec![
            ChatMessage::new(ChatRole::System, "You are helpful."),
            ChatMessage::new(ChatRole::User, "Hi!"),
        ];
        let result = tmpl.apply(&messages, true);
        assert!(result.contains("<|im_start|>system"));
        assert!(result.contains("<|im_start|>user"));
        assert!(result.contains("<|im_end|>"));
        assert!(result.contains("<|im_start|>assistant"));
    }

    #[test]
    fn chat_template_empty_messages() {
        let tmpl = ChatTemplate::default();
        let result = tmpl.apply(&[], false);
        assert!(result.is_empty());
    }

    #[test]
    fn chat_template_empty_with_gen_prompt() {
        let tmpl = ChatTemplate::default();
        let result = tmpl.apply(&[], true);
        assert_eq!(result, "<|assistant|>\n");
    }

    #[test]
    fn chat_template_custom() {
        let mut prefixes = HashMap::new();
        prefixes.insert(ChatRole::User, "USER: ".to_string());
        let mut suffixes = HashMap::new();
        suffixes.insert(ChatRole::User, "\n".to_string());
        let tmpl = ChatTemplate::new(prefixes, suffixes, String::new(), "BOT: ".into());
        let messages = vec![ChatMessage::new(ChatRole::User, "Hello")];
        let result = tmpl.apply(&messages, true);
        assert!(result.starts_with("USER: Hello"));
        assert!(result.ends_with("BOT: "));
    }

    #[test]
    fn chat_template_multi_turn() {
        let tmpl = ChatTemplate::default();
        let messages = vec![
            ChatMessage::new(ChatRole::User, "Turn 1"),
            ChatMessage::new(ChatRole::Assistant, "Response 1"),
            ChatMessage::new(ChatRole::User, "Turn 2"),
        ];
        let result = tmpl.apply(&messages, true);
        assert!(result.contains("Turn 1"));
        assert!(result.contains("Response 1"));
        assert!(result.contains("Turn 2"));
    }

    // ── PromptMode tests ───────────────────────────────────────────────

    #[test]
    fn prompt_mode_display() {
        assert_eq!(PromptMode::Instruct.to_string(), "instruct");
        assert_eq!(PromptMode::Chat.to_string(), "chat");
        assert_eq!(PromptMode::Completion.to_string(), "completion");
    }

    // ── PromptFormatter tests ──────────────────────────────────────────

    #[test]
    fn prompt_formatter_completion() {
        let fmt = PromptFormatter::default();
        let result = fmt.format("hello world", PromptMode::Completion);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn prompt_formatter_instruct() {
        let fmt = PromptFormatter::default();
        let result = fmt.format("What is 2+2?", PromptMode::Instruct);
        assert!(result.starts_with("[INST] "));
        assert!(result.contains("What is 2+2?"));
        assert!(result.ends_with(" [/INST]"));
    }

    #[test]
    fn prompt_formatter_chat() {
        let fmt = PromptFormatter::default();
        let result = fmt.format("Hi!", PromptMode::Chat);
        assert!(result.contains("You are a helpful assistant."));
        assert!(result.contains("Hi!"));
    }

    #[test]
    fn prompt_formatter_custom_system() {
        let fmt = PromptFormatter::default().with_system_prompt("You are a pirate.");
        let result = fmt.format("Ahoy!", PromptMode::Chat);
        assert!(result.contains("You are a pirate."));
    }

    #[test]
    fn prompt_formatter_custom_instruct_affixes() {
        let fmt = PromptFormatter::default()
            .with_instruct_affixes("### Instruction:\n", "\n### Response:\n");
        let result = fmt.format("Test", PromptMode::Instruct);
        assert!(result.starts_with("### Instruction:\n"));
        assert!(result.ends_with("\n### Response:\n"));
    }

    #[test]
    fn prompt_formatter_conversation() {
        let fmt = PromptFormatter::default();
        let msgs = vec![
            ChatMessage::new(ChatRole::User, "Hello"),
            ChatMessage::new(ChatRole::Assistant, "Hi there!"),
            ChatMessage::new(ChatRole::User, "How are you?"),
        ];
        let result = fmt.format_conversation(&msgs, true);
        assert!(result.contains("Hello"));
        assert!(result.contains("Hi there!"));
        assert!(result.contains("How are you?"));
    }

    #[test]
    fn prompt_formatter_with_chatml() {
        let fmt = PromptFormatter::default().with_chat_template(ChatTemplate::chatml());
        let result = fmt.format("Hi!", PromptMode::Chat);
        assert!(result.contains("<|im_start|>"));
    }

    // ── TokenizerWrapper tests ─────────────────────────────────────────

    #[test]
    fn wrapper_encode_basic() {
        let w = test_wrapper();
        let ids = w.encode("hello world");
        assert_eq!(ids[0], 1);
        assert_eq!(ids[1], 10);
        assert_eq!(ids[2], 11);
        assert_eq!(ids[3], 2);
    }

    #[test]
    fn wrapper_decode_basic() {
        let w = test_wrapper();
        let text = w.decode(&[10, 11]);
        assert_eq!(text, "hello world");
    }

    #[test]
    fn wrapper_decode_raw() {
        let w = test_wrapper();
        let text = w.decode_raw(&[1, 10, 2]);
        assert!(text.contains("<s>"));
    }

    #[test]
    fn wrapper_encode_with_max_length() {
        let w = test_wrapper().with_max_length(3);
        let ids = w.encode("hello world the cat");
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn wrapper_encode_with_left_truncation() {
        let w = test_wrapper().with_truncation(TruncationStrategy::Left).with_max_length(3);
        let ids = w.encode("hello world the cat");
        assert_eq!(ids.len(), 3);
        assert_eq!(*ids.last().unwrap(), 2); // EOS
    }

    #[test]
    fn wrapper_encode_with_truncation_explicit() {
        let w = test_wrapper();
        let ids = w.encode_with_truncation("hello world", 3, TruncationStrategy::Right);
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn wrapper_encode_padded() {
        let w = test_wrapper();
        let ids = w.encode_padded("hello", 8);
        assert_eq!(ids.len(), 8);
    }

    #[test]
    fn wrapper_format_prompt_completion() {
        let w = test_wrapper();
        let result = w.format_prompt("test", PromptMode::Completion);
        assert_eq!(result, "test");
    }

    #[test]
    fn wrapper_format_prompt_instruct() {
        let w = test_wrapper();
        let result = w.format_prompt("test", PromptMode::Instruct);
        assert!(result.contains("[INST]"));
    }

    #[test]
    fn wrapper_format_prompt_chat() {
        let w = test_wrapper();
        let result = w.format_prompt("Hi!", PromptMode::Chat);
        assert!(result.contains("Hi!"));
        assert!(result.contains("<|system|>"));
    }

    #[test]
    fn wrapper_encode_prompt() {
        let w = test_wrapper();
        let ids = w.encode_prompt("hello", PromptMode::Completion);
        assert_eq!(ids, w.encode("hello"));
    }

    #[test]
    fn wrapper_encode_conversation() {
        let w = test_wrapper();
        let msgs = vec![
            ChatMessage::new(ChatRole::User, "hello"),
            ChatMessage::new(ChatRole::Assistant, "world"),
        ];
        let ids = w.encode_conversation(&msgs, true);
        assert!(!ids.is_empty());
    }

    #[test]
    fn wrapper_encode_batch() {
        let w = test_wrapper();
        let batch = w.encode_batch(&["hello", "world"]);
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn wrapper_vocab_size() {
        let w = test_wrapper();
        assert!(w.vocab_size() > 0);
    }

    #[test]
    fn wrapper_config_accessor() {
        let w = test_wrapper();
        assert_eq!(w.config().model_type, TokenizerType::Bpe);
    }

    #[test]
    fn wrapper_custom_prompt_formatter() {
        let fmt = PromptFormatter::default().with_system_prompt("Custom system.");
        let w = test_wrapper().with_prompt_formatter(fmt);
        let result = w.format_prompt("Hi!", PromptMode::Chat);
        assert!(result.contains("Custom system."));
    }

    #[test]
    fn wrapper_no_max_length() {
        let w = test_wrapper();
        let ids = w.encode("hello world the cat sat on the mat");
        // No truncation → full sequence.
        assert!(ids.len() > 5);
    }

    // ── Edge case tests ────────────────────────────────────────────────

    #[test]
    fn edge_unicode_encoding() {
        let enc = test_encoder();
        // Unicode chars not in vocab → UNK per character.
        let ids = enc.encode("日本語");
        assert!(ids.len() >= 2); // At least BOS + EOS.
        // All non-special IDs should be UNK.
        for &id in &ids[1..ids.len() - 1] {
            assert_eq!(id, 3);
        }
    }

    #[test]
    fn edge_emoji() {
        let enc = test_encoder();
        let ids = enc.encode("🎉");
        // Single emoji not in vocab → UNK.
        assert_eq!(ids, vec![1, 3, 2]);
    }

    #[test]
    fn edge_mixed_known_unknown() {
        let enc = test_encoder();
        let ids = enc.encode("hello 123 world");
        assert_eq!(ids[0], 1); // BOS
        assert_eq!(ids[1], 10); // hello
        assert_eq!(ids[2], 3); // 1 → UNK
        assert_eq!(ids[3], 3); // 2 → UNK
        assert_eq!(ids[4], 3); // 3 → UNK
        assert_eq!(ids[5], 11); // world
        assert_eq!(ids[6], 2); // EOS
    }

    #[test]
    fn edge_very_long_text() {
        let enc = test_encoder();
        let long_text = "hello ".repeat(1000);
        let ids = enc.encode(&long_text);
        // BOS + 1000× hello + EOS.
        assert_eq!(ids.len(), 1002);
    }

    #[test]
    fn edge_newlines_and_tabs() {
        let enc = test_encoder();
        // Whitespace normalisation collapses \n and \t.
        let ids = enc.encode("hello\n\tworld");
        let ids2 = enc.encode("hello world");
        assert_eq!(ids, ids2);
    }

    #[test]
    fn edge_repeated_spaces() {
        let enc = test_encoder();
        let ids = enc.encode("hello     world");
        let ids2 = enc.encode("hello world");
        assert_eq!(ids, ids2);
    }

    // ── proptest ───────────────────────────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn encode_never_panics(s in ".*") {
                let enc = test_encoder();
                let _ = enc.encode(&s);
            }

            #[test]
            fn decode_never_panics(ids in proptest::collection::vec(0u32..100, 0..50)) {
                let dec = test_decoder();
                let _ = dec.decode(&ids);
            }

            #[test]
            fn encode_always_starts_with_bos(s in ".{0,100}") {
                let enc = test_encoder();
                let ids = enc.encode(&s);
                prop_assert_eq!(ids[0], 1);
            }

            #[test]
            fn encode_always_ends_with_eos(s in ".{0,100}") {
                let enc = test_encoder();
                let ids = enc.encode(&s);
                prop_assert_eq!(*ids.last().unwrap(), 2);
            }

            #[test]
            fn truncation_right_respects_max_len(
                ids in proptest::collection::vec(0u32..100, 1..50),
                max_len in 1usize..50
            ) {
                let result = TruncationStrategy::Right.apply(&ids, max_len);
                prop_assert!(result.len() <= max_len.max(ids.len().min(max_len)));
            }

            #[test]
            fn truncation_left_respects_max_len(
                ids in proptest::collection::vec(0u32..100, 1..50),
                max_len in 1usize..50
            ) {
                let result = TruncationStrategy::Left.apply(&ids, max_len);
                prop_assert!(result.len() <= max_len.max(ids.len().min(max_len)));
            }

            #[test]
            fn padded_length_at_least_target(
                s in "[a-z ]{0,20}",
                target in 1usize..30
            ) {
                let enc = test_encoder();
                let ids = enc.encode_padded(&s, target);
                prop_assert!(ids.len() >= target);
            }

            #[test]
            fn batch_encode_length_matches(
                inputs in proptest::collection::vec("[a-z]{1,10}", 0..10)
            ) {
                let enc = test_encoder();
                let refs: Vec<&str> = inputs.iter().map(|s| s.as_str()).collect();
                let batch = enc.encode_batch(&refs);
                prop_assert_eq!(batch.len(), inputs.len());
            }

            #[test]
            fn wrapper_encode_never_panics(s in ".*") {
                let w = test_wrapper();
                let _ = w.encode(&s);
            }

            #[test]
            fn wrapper_format_prompt_never_panics(s in ".{0,50}") {
                let w = test_wrapper();
                let _ = w.format_prompt(&s, PromptMode::Completion);
                let _ = w.format_prompt(&s, PromptMode::Instruct);
                let _ = w.format_prompt(&s, PromptMode::Chat);
            }

            #[test]
            fn special_tokens_is_special_consistent(id in 0u32..1000) {
                let st = SpecialTokens::default();
                let expected = [0, 1, 2, 3, 4].contains(&id);
                prop_assert_eq!(st.is_special(id), expected);
            }
        }
    }
}
