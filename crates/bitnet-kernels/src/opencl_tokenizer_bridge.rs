//! Bridge between tokenizer and GPU inference for efficient token processing.
//!
//! Provides CPU reference implementations for all tokenization operations
//! needed by the OpenCL inference pipeline:
//!
//! - **`TokenBatch`** — batched token IDs with padding and attention masks
//! - **`TokenEncoder`** — encode text → token IDs with special tokens
//! - **`TokenDecoder`** — decode token IDs → text with cleanup
//! - **`PromptTemplate`** — chat / instruct template formatting (ChatML, Llama, etc.)
//! - **`SpecialTokens`** — BOS, EOS, PAD, UNK, SEP token management
//! - **`TokenizerConfig`** — vocab size, max sequence length, padding, truncation
//! - **`BatchPadder`** — pad sequences to uniform length for GPU dispatch
//! - **`TokenStats`** — tokens/sec, average sequence length, vocab coverage
//!
//! All operations compile and run without an OpenCL runtime.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// PaddingSide
// ---------------------------------------------------------------------------

/// Which end of a sequence receives padding tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingSide {
    /// Pad on the left (common for decoder-only models).
    Left,
    /// Pad on the right (common for encoder models).
    Right,
}

impl fmt::Display for PaddingSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Left => write!(f, "left"),
            Self::Right => write!(f, "right"),
        }
    }
}

// ---------------------------------------------------------------------------
// TruncationStrategy
// ---------------------------------------------------------------------------

/// How to truncate sequences that exceed `max_seq_len`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationStrategy {
    /// Drop tokens from the end.
    TruncateEnd,
    /// Drop tokens from the beginning.
    TruncateStart,
    /// Return an error instead of truncating.
    Error,
}

// ---------------------------------------------------------------------------
// SpecialTokens
// ---------------------------------------------------------------------------

/// Standard special tokens used by the tokenizer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecialTokens {
    /// Beginning-of-sequence token ID.
    pub bos_id: u32,
    /// End-of-sequence token ID.
    pub eos_id: u32,
    /// Padding token ID.
    pub pad_id: u32,
    /// Unknown-token ID.
    pub unk_id: u32,
    /// Optional separator token ID (used by some models).
    pub sep_id: Option<u32>,
}

impl SpecialTokens {
    /// Create a new set of special tokens.
    pub fn new(bos_id: u32, eos_id: u32, pad_id: u32, unk_id: u32) -> Self {
        Self { bos_id, eos_id, pad_id, unk_id, sep_id: None }
    }

    /// Set the separator token.
    #[must_use]
    pub fn with_sep(mut self, sep_id: u32) -> Self {
        self.sep_id = Some(sep_id);
        self
    }

    /// Return `true` if `id` is any of the special tokens.
    pub fn is_special(&self, id: u32) -> bool {
        id == self.bos_id
            || id == self.eos_id
            || id == self.pad_id
            || id == self.unk_id
            || self.sep_id == Some(id)
    }

    /// Typical defaults for a 32 000-vocab model.
    pub fn default_32k() -> Self {
        Self { bos_id: 1, eos_id: 2, pad_id: 0, unk_id: 3, sep_id: None }
    }
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self::default_32k()
    }
}

// ---------------------------------------------------------------------------
// TokenizerConfig
// ---------------------------------------------------------------------------

/// Configuration that governs encoding, decoding, padding and truncation.
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    /// Total number of tokens in the vocabulary.
    pub vocab_size: usize,
    /// Maximum sequence length the model supports.
    pub max_seq_len: usize,
    /// Side on which padding tokens are added.
    pub padding_side: PaddingSide,
    /// How to handle sequences that exceed `max_seq_len`.
    pub truncation: TruncationStrategy,
    /// Special tokens.
    pub special_tokens: SpecialTokens,
    /// Whether to automatically prepend BOS.
    pub add_bos: bool,
    /// Whether to automatically append EOS.
    pub add_eos: bool,
}

impl TokenizerConfig {
    /// Create a config with sensible defaults for a decoder-only LLM.
    pub fn new(vocab_size: usize, max_seq_len: usize) -> Self {
        Self {
            vocab_size,
            max_seq_len,
            padding_side: PaddingSide::Left,
            truncation: TruncationStrategy::TruncateEnd,
            special_tokens: SpecialTokens::default(),
            add_bos: true,
            add_eos: false,
        }
    }

    #[must_use]
    pub fn with_padding_side(mut self, side: PaddingSide) -> Self {
        self.padding_side = side;
        self
    }

    #[must_use]
    pub fn with_truncation(mut self, strategy: TruncationStrategy) -> Self {
        self.truncation = strategy;
        self
    }

    #[must_use]
    pub fn with_special_tokens(mut self, tokens: SpecialTokens) -> Self {
        self.special_tokens = tokens;
        self
    }

    #[must_use]
    pub fn with_add_bos(mut self, add: bool) -> Self {
        self.add_bos = add;
        self
    }

    #[must_use]
    pub fn with_add_eos(mut self, add: bool) -> Self {
        self.add_eos = add;
        self
    }

    /// Validate the configuration, returning a descriptive error.
    pub fn validate(&self) -> Result<(), String> {
        if self.vocab_size == 0 {
            return Err("vocab_size must be > 0".into());
        }
        if self.max_seq_len == 0 {
            return Err("max_seq_len must be > 0".into());
        }
        let st = &self.special_tokens;
        for &id in &[st.bos_id, st.eos_id, st.pad_id, st.unk_id] {
            if id as usize >= self.vocab_size {
                return Err(format!("special token id {id} >= vocab_size {}", self.vocab_size));
            }
        }
        if let Some(sep) = st.sep_id
            && sep as usize >= self.vocab_size
        {
            return Err(format!("sep token id {sep} >= vocab_size {}", self.vocab_size));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TokenBatch
// ---------------------------------------------------------------------------

/// A batch of padded token-ID sequences ready for GPU dispatch.
///
/// All sequences share the same length (`seq_len`).  An attention mask
/// distinguishes real tokens (1) from padding (0).
#[derive(Debug, Clone, PartialEq)]
pub struct TokenBatch {
    /// Flat token IDs in row-major layout: `[batch_size, seq_len]`.
    pub token_ids: Vec<u32>,
    /// Attention mask: 1 for real tokens, 0 for padding.
    pub attention_mask: Vec<u8>,
    /// Number of sequences in the batch.
    pub batch_size: usize,
    /// Uniform sequence length (after padding).
    pub seq_len: usize,
}

impl TokenBatch {
    /// Build from a list of variable-length sequences using a `BatchPadder`.
    pub fn from_sequences(
        sequences: &[Vec<u32>],
        config: &TokenizerConfig,
    ) -> Result<Self, String> {
        if sequences.is_empty() {
            return Ok(Self {
                token_ids: Vec::new(),
                attention_mask: Vec::new(),
                batch_size: 0,
                seq_len: 0,
            });
        }
        let padder = BatchPadder::new(config);
        padder.pad_batch(sequences)
    }

    /// Get the token IDs for a single sequence in the batch.
    pub fn sequence(&self, idx: usize) -> Option<&[u32]> {
        if idx >= self.batch_size {
            return None;
        }
        let start = idx * self.seq_len;
        let end = start + self.seq_len;
        Some(&self.token_ids[start..end])
    }

    /// Get the attention mask for a single sequence.
    pub fn mask(&self, idx: usize) -> Option<&[u8]> {
        if idx >= self.batch_size {
            return None;
        }
        let start = idx * self.seq_len;
        let end = start + self.seq_len;
        Some(&self.attention_mask[start..end])
    }

    /// Number of *real* (non-padding) tokens across the whole batch.
    pub fn total_real_tokens(&self) -> usize {
        self.attention_mask.iter().filter(|&&m| m == 1).count()
    }
}

// ---------------------------------------------------------------------------
// BatchPadder
// ---------------------------------------------------------------------------

/// Pads variable-length sequences to a uniform length for GPU processing.
pub struct BatchPadder<'a> {
    config: &'a TokenizerConfig,
}

impl<'a> BatchPadder<'a> {
    pub fn new(config: &'a TokenizerConfig) -> Self {
        Self { config }
    }

    /// Pad a batch of variable-length sequences.
    ///
    /// The target length is `min(max_in_batch, max_seq_len)`.
    pub fn pad_batch(&self, sequences: &[Vec<u32>]) -> Result<TokenBatch, String> {
        if sequences.is_empty() {
            return Ok(TokenBatch {
                token_ids: Vec::new(),
                attention_mask: Vec::new(),
                batch_size: 0,
                seq_len: 0,
            });
        }

        let max_len_in_batch = sequences.iter().map(|s| s.len()).max().unwrap_or(0);
        let target_len = max_len_in_batch.min(self.config.max_seq_len);

        // Truncate then pad each sequence.
        let batch_size = sequences.len();
        let mut token_ids = Vec::with_capacity(batch_size * target_len);
        let mut attention_mask = Vec::with_capacity(batch_size * target_len);

        for seq in sequences {
            let truncated = self.truncate(seq, target_len)?;
            let (padded_ids, padded_mask) = self.pad_single(&truncated, target_len);
            token_ids.extend_from_slice(&padded_ids);
            attention_mask.extend_from_slice(&padded_mask);
        }

        Ok(TokenBatch { token_ids, attention_mask, batch_size, seq_len: target_len })
    }

    /// Pad a single sequence to `target_len`.
    fn pad_single(&self, seq: &[u32], target_len: usize) -> (Vec<u32>, Vec<u8>) {
        let pad_id = self.config.special_tokens.pad_id;
        let pad_count = target_len.saturating_sub(seq.len());

        let mut ids = Vec::with_capacity(target_len);
        let mut mask = Vec::with_capacity(target_len);

        match self.config.padding_side {
            PaddingSide::Left => {
                ids.extend(std::iter::repeat_n(pad_id, pad_count));
                mask.extend(std::iter::repeat_n(0u8, pad_count));
                ids.extend_from_slice(seq);
                mask.extend(std::iter::repeat_n(1u8, seq.len()));
            }
            PaddingSide::Right => {
                ids.extend_from_slice(seq);
                mask.extend(std::iter::repeat_n(1u8, seq.len()));
                ids.extend(std::iter::repeat_n(pad_id, pad_count));
                mask.extend(std::iter::repeat_n(0u8, pad_count));
            }
        }

        (ids, mask)
    }

    /// Truncate a sequence if it exceeds `max_len`.
    fn truncate(&self, seq: &[u32], max_len: usize) -> Result<Vec<u32>, String> {
        if seq.len() <= max_len {
            return Ok(seq.to_vec());
        }
        match self.config.truncation {
            TruncationStrategy::TruncateEnd => Ok(seq[..max_len].to_vec()),
            TruncationStrategy::TruncateStart => Ok(seq[seq.len() - max_len..].to_vec()),
            TruncationStrategy::Error => {
                Err(format!("sequence length {} exceeds max_seq_len {}", seq.len(), max_len))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TokenEncoder
// ---------------------------------------------------------------------------

/// CPU reference encoder: text → token IDs.
///
/// Uses a simple word-level vocabulary for the reference implementation.
/// Real deployments would use BPE / SentencePiece via `bitnet-tokenizers`.
#[derive(Debug, Clone)]
pub struct TokenEncoder {
    /// Word → token ID mapping.
    vocab: HashMap<String, u32>,
    config: TokenizerConfig,
}

impl TokenEncoder {
    /// Create an encoder from a vocabulary mapping and config.
    pub fn new(vocab: HashMap<String, u32>, config: TokenizerConfig) -> Self {
        Self { vocab, config }
    }

    /// Build a tiny test encoder with a fixed vocabulary.
    pub fn tiny_test() -> Self {
        let mut vocab = HashMap::new();
        let words = [
            "hello", "world", "the", "a", "is", "of", "to", "and", "in", "that", "it", "for",
            "you", "was", "on", "are", "with", "this", "from", "or", "an", "be", "one", "had",
            "by", "not", "but", "what", "all", "were", "when", "we", "there", "can", "your",
            "which", "their", "will", "each", "about", "how", "up", "out", "them", "then", "she",
            "many", "some", "so", "these", "would", "other", "into", "has", "more", "two", "her",
            "like", "him",
        ];
        // IDs 0..3 are special tokens; vocab starts at 4
        for (i, word) in words.iter().enumerate() {
            vocab.insert(word.to_string(), (i as u32) + 4);
        }
        let config = TokenizerConfig::new(64 + words.len(), 128);
        Self { vocab, config }
    }

    /// Encode text into token IDs.
    ///
    /// Words not in the vocabulary are mapped to `unk_id`.  BOS / EOS are
    /// prepended / appended according to `config`.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let mut ids = Vec::new();

        if self.config.add_bos {
            ids.push(self.config.special_tokens.bos_id);
        }

        for word in tokenize_words(text) {
            let id = self.vocab.get(word).copied().unwrap_or(self.config.special_tokens.unk_id);
            ids.push(id);
        }

        if self.config.add_eos {
            ids.push(self.config.special_tokens.eos_id);
        }

        // Truncation
        if ids.len() > self.config.max_seq_len {
            match self.config.truncation {
                TruncationStrategy::TruncateEnd => {
                    ids.truncate(self.config.max_seq_len);
                }
                TruncationStrategy::TruncateStart => {
                    let start = ids.len() - self.config.max_seq_len;
                    ids = ids[start..].to_vec();
                }
                TruncationStrategy::Error => {
                    return Err(format!(
                        "encoded length {} exceeds max_seq_len {}",
                        ids.len(),
                        self.config.max_seq_len,
                    ));
                }
            }
        }

        Ok(ids)
    }

    /// Encode a batch of texts.
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>, String> {
        texts.iter().map(|t| self.encode(t)).collect()
    }

    /// Access the underlying config.
    pub fn config(&self) -> &TokenizerConfig {
        &self.config
    }

    /// Access the vocab (for the decoder).
    pub fn vocab(&self) -> &HashMap<String, u32> {
        &self.vocab
    }
}

// ---------------------------------------------------------------------------
// TokenDecoder
// ---------------------------------------------------------------------------

/// CPU reference decoder: token IDs → text.
#[derive(Debug, Clone)]
pub struct TokenDecoder {
    /// Token ID → word mapping (inverse of encoder vocab).
    id_to_word: HashMap<u32, String>,
    special_tokens: SpecialTokens,
}

impl TokenDecoder {
    /// Create a decoder from an encoder's vocabulary.
    pub fn from_encoder(encoder: &TokenEncoder) -> Self {
        let id_to_word: HashMap<u32, String> =
            encoder.vocab.iter().map(|(w, &id)| (id, w.clone())).collect();
        Self { id_to_word, special_tokens: encoder.config.special_tokens.clone() }
    }

    /// Create a decoder directly from an id→word map and special tokens.
    pub fn new(id_to_word: HashMap<u32, String>, special_tokens: SpecialTokens) -> Self {
        Self { id_to_word, special_tokens }
    }

    /// Decode a sequence of token IDs into text.
    ///
    /// Special tokens are omitted and words are joined with spaces.
    pub fn decode(&self, ids: &[u32]) -> String {
        let words: Vec<&str> = ids
            .iter()
            .filter(|&&id| !self.special_tokens.is_special(id))
            .map(|&id| self.id_to_word.get(&id).map(|s| s.as_str()).unwrap_or("<unk>"))
            .collect();
        words.join(" ")
    }

    /// Decode, preserving special tokens as `<BOS>`, `<EOS>`, etc.
    pub fn decode_with_special(&self, ids: &[u32]) -> String {
        let words: Vec<String> = ids
            .iter()
            .map(|&id| {
                if id == self.special_tokens.bos_id {
                    "<BOS>".to_string()
                } else if id == self.special_tokens.eos_id {
                    "<EOS>".to_string()
                } else if id == self.special_tokens.pad_id {
                    "<PAD>".to_string()
                } else if id == self.special_tokens.unk_id {
                    "<UNK>".to_string()
                } else if self.special_tokens.sep_id == Some(id) {
                    "<SEP>".to_string()
                } else {
                    self.id_to_word.get(&id).cloned().unwrap_or_else(|| format!("<{id}>"))
                }
            })
            .collect();
        words.join(" ")
    }

    /// Decode a batch of sequences.
    pub fn decode_batch(&self, batch: &[Vec<u32>]) -> Vec<String> {
        batch.iter().map(|ids| self.decode(ids)).collect()
    }
}

// ---------------------------------------------------------------------------
// PromptTemplate
// ---------------------------------------------------------------------------

/// Supported chat / instruct template formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateFormat {
    /// ChatML: `<|im_start|>role\ncontent<|im_end|>`
    ChatML,
    /// Llama-2 instruct: `[INST] ... [/INST]`
    Llama2,
    /// Simple: `### Instruction:\n...\n### Response:\n`
    Simple,
    /// Raw: no template wrapping.
    Raw,
}

impl fmt::Display for TemplateFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChatML => write!(f, "chatml"),
            Self::Llama2 => write!(f, "llama2"),
            Self::Simple => write!(f, "simple"),
            Self::Raw => write!(f, "raw"),
        }
    }
}

/// A single message in a conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".into(), content: content.into() }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: content.into() }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".into(), content: content.into() }
    }
}

/// Applies a chat / instruct template to messages.
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    format: TemplateFormat,
    /// Optional system preamble injected before user messages.
    system_prompt: Option<String>,
}

impl PromptTemplate {
    pub fn new(format: TemplateFormat) -> Self {
        Self { format, system_prompt: None }
    }

    #[must_use]
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub fn format(&self) -> TemplateFormat {
        self.format
    }

    /// Format a single user instruction (convenience wrapper).
    pub fn format_instruction(&self, instruction: &str) -> String {
        let mut messages = Vec::new();
        if let Some(sys) = &self.system_prompt {
            messages.push(ChatMessage::system(sys.clone()));
        }
        messages.push(ChatMessage::user(instruction));
        self.apply(&messages)
    }

    /// Apply the template to a sequence of chat messages.
    pub fn apply(&self, messages: &[ChatMessage]) -> String {
        match self.format {
            TemplateFormat::ChatML => self.apply_chatml(messages),
            TemplateFormat::Llama2 => self.apply_llama2(messages),
            TemplateFormat::Simple => self.apply_simple(messages),
            TemplateFormat::Raw => self.apply_raw(messages),
        }
    }

    fn apply_chatml(&self, messages: &[ChatMessage]) -> String {
        let mut out = String::new();
        for msg in messages {
            out.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
        }
        out.push_str("<|im_start|>assistant\n");
        out
    }

    fn apply_llama2(&self, messages: &[ChatMessage]) -> String {
        let mut out = String::new();
        let mut system_text = String::new();

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    system_text = msg.content.clone();
                }
                "user" => {
                    out.push_str("<s>[INST] ");
                    if !system_text.is_empty() {
                        out.push_str(&format!("<<SYS>>\n{system_text}\n<</SYS>>\n\n"));
                        system_text.clear();
                    }
                    out.push_str(&msg.content);
                    out.push_str(" [/INST]");
                }
                "assistant" => {
                    out.push(' ');
                    out.push_str(&msg.content);
                    out.push_str(" </s>");
                }
                _ => {}
            }
        }
        out
    }

    fn apply_simple(&self, messages: &[ChatMessage]) -> String {
        let mut out = String::new();
        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    out.push_str(&format!("### System:\n{}\n\n", msg.content));
                }
                "user" => {
                    out.push_str(&format!("### Instruction:\n{}\n\n", msg.content));
                }
                "assistant" => {
                    out.push_str(&format!("### Response:\n{}\n\n", msg.content));
                }
                _ => {}
            }
        }
        out.push_str("### Response:\n");
        out
    }

    fn apply_raw(&self, messages: &[ChatMessage]) -> String {
        messages.iter().map(|m| m.content.as_str()).collect::<Vec<_>>().join("\n")
    }
}

// ---------------------------------------------------------------------------
// TokenStats
// ---------------------------------------------------------------------------

/// Lightweight statistics tracker for tokenization throughput.
#[derive(Debug, Clone)]
pub struct TokenStats {
    /// Total tokens processed.
    pub total_tokens: u64,
    /// Total time spent encoding / decoding.
    pub total_duration: Duration,
    /// Sum of sequence lengths (for average computation).
    pub total_seq_len: u64,
    /// Number of sequences processed.
    pub num_sequences: u64,
    /// Set of unique token IDs seen.
    unique_ids: HashSet<u32>,
}

impl TokenStats {
    pub fn new() -> Self {
        Self {
            total_tokens: 0,
            total_duration: Duration::ZERO,
            total_seq_len: 0,
            num_sequences: 0,
            unique_ids: HashSet::new(),
        }
    }

    /// Record the encoding of a single sequence.
    pub fn record(&mut self, ids: &[u32], duration: Duration) {
        self.total_tokens += ids.len() as u64;
        self.total_duration += duration;
        self.total_seq_len += ids.len() as u64;
        self.num_sequences += 1;
        self.unique_ids.extend(ids.iter().copied());
    }

    /// Tokens per second (0.0 if no time recorded).
    pub fn tokens_per_sec(&self) -> f64 {
        let secs = self.total_duration.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        self.total_tokens as f64 / secs
    }

    /// Average sequence length.
    pub fn avg_seq_len(&self) -> f64 {
        if self.num_sequences == 0 {
            return 0.0;
        }
        self.total_seq_len as f64 / self.num_sequences as f64
    }

    /// Number of unique token IDs encountered.
    pub fn vocab_coverage(&self) -> usize {
        self.unique_ids.len()
    }

    /// Fraction of the vocabulary used, given total vocab size.
    pub fn vocab_coverage_ratio(&self, vocab_size: usize) -> f64 {
        if vocab_size == 0 {
            return 0.0;
        }
        self.unique_ids.len() as f64 / vocab_size as f64
    }

    /// Merge another stats instance into this one.
    pub fn merge(&mut self, other: &TokenStats) {
        self.total_tokens += other.total_tokens;
        self.total_duration += other.total_duration;
        self.total_seq_len += other.total_seq_len;
        self.num_sequences += other.num_sequences;
        self.unique_ids.extend(&other.unique_ids);
    }
}

impl Default for TokenStats {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TokenStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TokenStats {{ tokens: {}, seqs: {}, tok/s: {:.1}, avg_len: {:.1}, coverage: {} }}",
            self.total_tokens,
            self.num_sequences,
            self.tokens_per_sec(),
            self.avg_seq_len(),
            self.vocab_coverage(),
        )
    }
}

// ---------------------------------------------------------------------------
// CPU reference: encode batch with stats
// ---------------------------------------------------------------------------

/// Encode a batch of texts with statistics collection.
pub fn encode_batch_with_stats(
    encoder: &TokenEncoder,
    texts: &[&str],
) -> Result<(Vec<Vec<u32>>, TokenStats), String> {
    let mut stats = TokenStats::new();
    let mut all_ids = Vec::with_capacity(texts.len());
    for &text in texts {
        let start = Instant::now();
        let ids = encoder.encode(text)?;
        let elapsed = start.elapsed();
        stats.record(&ids, elapsed);
        all_ids.push(ids);
    }
    Ok((all_ids, stats))
}

/// Full CPU pipeline: texts → padded `TokenBatch` + stats.
pub fn prepare_batch(
    encoder: &TokenEncoder,
    texts: &[&str],
) -> Result<(TokenBatch, TokenStats), String> {
    let (sequences, stats) = encode_batch_with_stats(encoder, texts)?;
    let batch = TokenBatch::from_sequences(&sequences, encoder.config())?;
    Ok((batch, stats))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Simple whitespace tokenizer — splits on whitespace and lowercases.
fn tokenize_words(text: &str) -> Vec<&str> {
    text.split_whitespace().collect()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers -----------------------------------------------------------

    fn test_encoder() -> TokenEncoder {
        TokenEncoder::tiny_test()
    }

    fn test_decoder(encoder: &TokenEncoder) -> TokenDecoder {
        TokenDecoder::from_encoder(encoder)
    }

    // ── SpecialTokens ────────────────────────────────────────────

    #[test]
    fn special_tokens_default() {
        let st = SpecialTokens::default();
        assert_eq!(st.bos_id, 1);
        assert_eq!(st.eos_id, 2);
        assert_eq!(st.pad_id, 0);
        assert_eq!(st.unk_id, 3);
        assert!(st.sep_id.is_none());
    }

    #[test]
    fn special_tokens_is_special() {
        let st = SpecialTokens::default().with_sep(4);
        assert!(st.is_special(0)); // PAD
        assert!(st.is_special(1)); // BOS
        assert!(st.is_special(2)); // EOS
        assert!(st.is_special(3)); // UNK
        assert!(st.is_special(4)); // SEP
        assert!(!st.is_special(5));
        assert!(!st.is_special(999));
    }

    #[test]
    fn special_tokens_without_sep() {
        let st = SpecialTokens::new(10, 11, 12, 13);
        assert!(!st.is_special(4));
        assert!(st.is_special(10));
        assert!(st.is_special(13));
    }

    // ── TokenizerConfig ──────────────────────────────────────────

    #[test]
    fn config_validate_ok() {
        let cfg = TokenizerConfig::new(32000, 2048);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_validate_zero_vocab() {
        let cfg = TokenizerConfig::new(0, 2048);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_zero_seq_len() {
        let cfg = TokenizerConfig::new(32000, 0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_special_token_oob() {
        let cfg = TokenizerConfig::new(2, 128); // vocab_size=2 but default unk=3
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_sep_oob() {
        let st = SpecialTokens::default().with_sep(999);
        let cfg = TokenizerConfig::new(10, 128).with_special_tokens(st);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_builder_methods() {
        let cfg = TokenizerConfig::new(1000, 512)
            .with_padding_side(PaddingSide::Right)
            .with_truncation(TruncationStrategy::TruncateStart)
            .with_add_bos(false)
            .with_add_eos(true);
        assert_eq!(cfg.padding_side, PaddingSide::Right);
        assert_eq!(cfg.truncation, TruncationStrategy::TruncateStart);
        assert!(!cfg.add_bos);
        assert!(cfg.add_eos);
    }

    // ── TokenEncoder ─────────────────────────────────────────────

    #[test]
    fn encode_simple() {
        let enc = test_encoder();
        let ids = enc.encode("hello world").unwrap();
        // BOS + hello + world
        assert_eq!(ids.len(), 3);
        assert_eq!(ids[0], enc.config.special_tokens.bos_id);
    }

    #[test]
    fn encode_empty_string() {
        let enc = test_encoder();
        let ids = enc.encode("").unwrap();
        // Only BOS (add_bos is true by default)
        assert_eq!(ids, vec![enc.config.special_tokens.bos_id]);
    }

    #[test]
    fn encode_unknown_word() {
        let enc = test_encoder();
        let ids = enc.encode("xyzzyx").unwrap();
        // BOS + UNK
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[1], enc.config.special_tokens.unk_id);
    }

    #[test]
    fn encode_with_eos() {
        let mut enc = test_encoder();
        enc.config.add_eos = true;
        let ids = enc.encode("hello").unwrap();
        // BOS + hello + EOS
        assert_eq!(ids.len(), 3);
        assert_eq!(*ids.last().unwrap(), enc.config.special_tokens.eos_id);
    }

    #[test]
    fn encode_no_bos() {
        let mut enc = test_encoder();
        enc.config.add_bos = false;
        let ids = enc.encode("hello world").unwrap();
        assert_eq!(ids.len(), 2); // just hello + world
        assert_ne!(ids[0], enc.config.special_tokens.bos_id);
    }

    #[test]
    fn encode_truncate_end() {
        let mut enc = test_encoder();
        enc.config.max_seq_len = 3; // BOS + 2 words max
        let ids = enc.encode("hello world the a is").unwrap();
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn encode_truncate_start() {
        let mut enc = test_encoder();
        enc.config.max_seq_len = 2;
        enc.config.truncation = TruncationStrategy::TruncateStart;
        let ids = enc.encode("hello world the").unwrap();
        // last 2 tokens
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn encode_truncation_error() {
        let mut enc = test_encoder();
        enc.config.max_seq_len = 2;
        enc.config.truncation = TruncationStrategy::Error;
        let result = enc.encode("hello world the");
        assert!(result.is_err());
    }

    #[test]
    fn encode_batch_multiple() {
        let enc = test_encoder();
        let batch = enc.encode_batch(&["hello", "world the"]).unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].len(), 2); // BOS + hello
        assert_eq!(batch[1].len(), 3); // BOS + world + the
    }

    // ── TokenDecoder ─────────────────────────────────────────────

    #[test]
    fn decode_simple() {
        let enc = test_encoder();
        let dec = test_decoder(&enc);
        let ids = enc.encode("hello world").unwrap();
        let text = dec.decode(&ids);
        assert_eq!(text, "hello world");
    }

    #[test]
    fn decode_skips_special_tokens() {
        let enc = test_encoder();
        let dec = test_decoder(&enc);
        let ids = vec![
            enc.config.special_tokens.bos_id,
            *enc.vocab.get("hello").unwrap(),
            enc.config.special_tokens.eos_id,
        ];
        let text = dec.decode(&ids);
        assert_eq!(text, "hello");
    }

    #[test]
    fn decode_with_special_tokens_shown() {
        let enc = test_encoder();
        let dec = test_decoder(&enc);
        let ids = vec![
            enc.config.special_tokens.bos_id,
            *enc.vocab.get("hello").unwrap(),
            enc.config.special_tokens.eos_id,
        ];
        let text = dec.decode_with_special(&ids);
        assert!(text.contains("<BOS>"));
        assert!(text.contains("hello"));
        assert!(text.contains("<EOS>"));
    }

    #[test]
    fn decode_unknown_id() {
        let enc = test_encoder();
        let dec = test_decoder(&enc);
        let text = dec.decode(&[9999]);
        assert_eq!(text, "<unk>");
    }

    #[test]
    fn decode_empty() {
        let enc = test_encoder();
        let dec = test_decoder(&enc);
        let text = dec.decode(&[]);
        assert_eq!(text, "");
    }

    #[test]
    fn decode_batch_multiple() {
        let enc = test_encoder();
        let dec = test_decoder(&enc);
        let ids1 = enc.encode("hello").unwrap();
        let ids2 = enc.encode("world").unwrap();
        let texts = dec.decode_batch(&[ids1, ids2]);
        assert_eq!(texts.len(), 2);
        assert_eq!(texts[0], "hello");
        assert_eq!(texts[1], "world");
    }

    #[test]
    fn decode_pad_token() {
        let enc = test_encoder();
        let dec = test_decoder(&enc);
        let text = dec.decode_with_special(&[enc.config.special_tokens.pad_id]);
        assert_eq!(text, "<PAD>");
    }

    #[test]
    fn decode_unk_token_special() {
        let enc = test_encoder();
        let dec = test_decoder(&enc);
        let text = dec.decode_with_special(&[enc.config.special_tokens.unk_id]);
        assert_eq!(text, "<UNK>");
    }

    // ── Roundtrip ────────────────────────────────────────────────

    #[test]
    fn roundtrip_encode_decode() {
        let enc = test_encoder();
        let dec = test_decoder(&enc);
        let original = "hello world the a is of to and";
        let ids = enc.encode(original).unwrap();
        let recovered = dec.decode(&ids);
        assert_eq!(recovered, original);
    }

    #[test]
    fn roundtrip_single_word() {
        let enc = test_encoder();
        let dec = test_decoder(&enc);
        for word in &["hello", "world", "the", "is", "of"] {
            let ids = enc.encode(word).unwrap();
            let recovered = dec.decode(&ids);
            assert_eq!(&recovered, word);
        }
    }

    #[test]
    fn roundtrip_preserves_word_order() {
        let enc = test_encoder();
        let dec = test_decoder(&enc);
        let texts = ["the world is hello", "a of to and in", "hello hello hello"];
        for text in &texts {
            let ids = enc.encode(text).unwrap();
            let recovered = dec.decode(&ids);
            assert_eq!(&recovered, text);
        }
    }

    // ── TokenBatch / BatchPadder ─────────────────────────────────

    #[test]
    fn batch_empty_sequences() {
        let cfg = TokenizerConfig::new(100, 128);
        let batch = TokenBatch::from_sequences(&[], &cfg).unwrap();
        assert_eq!(batch.batch_size, 0);
        assert_eq!(batch.seq_len, 0);
    }

    #[test]
    fn batch_single_sequence() {
        let cfg = TokenizerConfig::new(100, 128);
        let batch = TokenBatch::from_sequences(&[vec![10, 20, 30]], &cfg).unwrap();
        assert_eq!(batch.batch_size, 1);
        assert_eq!(batch.seq_len, 3);
        assert_eq!(batch.token_ids, vec![10, 20, 30]);
        assert_eq!(batch.attention_mask, vec![1, 1, 1]);
    }

    #[test]
    fn batch_right_padding() {
        let cfg = TokenizerConfig::new(100, 128).with_padding_side(PaddingSide::Right);
        let seqs = vec![vec![10, 20, 30], vec![40, 50]];
        let batch = TokenBatch::from_sequences(&seqs, &cfg).unwrap();
        assert_eq!(batch.seq_len, 3);
        assert_eq!(batch.sequence(0).unwrap(), &[10, 20, 30]);
        assert_eq!(batch.sequence(1).unwrap(), &[40, 50, 0]); // PAD=0
        assert_eq!(batch.mask(1).unwrap(), &[1, 1, 0]);
    }

    #[test]
    fn batch_left_padding() {
        let cfg = TokenizerConfig::new(100, 128).with_padding_side(PaddingSide::Left);
        let seqs = vec![vec![10, 20, 30], vec![40, 50]];
        let batch = TokenBatch::from_sequences(&seqs, &cfg).unwrap();
        assert_eq!(batch.seq_len, 3);
        assert_eq!(batch.sequence(0).unwrap(), &[10, 20, 30]);
        assert_eq!(batch.sequence(1).unwrap(), &[0, 40, 50]); // PAD on left
        assert_eq!(batch.mask(1).unwrap(), &[0, 1, 1]);
    }

    #[test]
    fn batch_truncation_end() {
        let cfg = TokenizerConfig::new(100, 3).with_truncation(TruncationStrategy::TruncateEnd);
        let seqs = vec![vec![1, 2, 3, 4, 5]];
        let batch = TokenBatch::from_sequences(&seqs, &cfg).unwrap();
        assert_eq!(batch.seq_len, 3);
        assert_eq!(batch.sequence(0).unwrap(), &[1, 2, 3]);
    }

    #[test]
    fn batch_truncation_start() {
        let cfg = TokenizerConfig::new(100, 3).with_truncation(TruncationStrategy::TruncateStart);
        let seqs = vec![vec![1, 2, 3, 4, 5]];
        let batch = TokenBatch::from_sequences(&seqs, &cfg).unwrap();
        assert_eq!(batch.seq_len, 3);
        assert_eq!(batch.sequence(0).unwrap(), &[3, 4, 5]);
    }

    #[test]
    fn batch_truncation_error() {
        let cfg = TokenizerConfig::new(100, 3).with_truncation(TruncationStrategy::Error);
        let seqs = vec![vec![1, 2, 3, 4, 5]];
        let result = TokenBatch::from_sequences(&seqs, &cfg);
        assert!(result.is_err());
    }

    #[test]
    fn batch_total_real_tokens() {
        let cfg = TokenizerConfig::new(100, 128).with_padding_side(PaddingSide::Right);
        let seqs = vec![vec![10, 20, 30], vec![40, 50]];
        let batch = TokenBatch::from_sequences(&seqs, &cfg).unwrap();
        assert_eq!(batch.total_real_tokens(), 5);
    }

    #[test]
    fn batch_sequence_oob() {
        let cfg = TokenizerConfig::new(100, 128);
        let batch = TokenBatch::from_sequences(&[vec![1, 2]], &cfg).unwrap();
        assert!(batch.sequence(0).is_some());
        assert!(batch.sequence(1).is_none());
        assert!(batch.mask(1).is_none());
    }

    #[test]
    fn batch_many_variable_lengths() {
        let cfg = TokenizerConfig::new(100, 128).with_padding_side(PaddingSide::Right);
        let seqs = vec![vec![1], vec![2, 3], vec![4, 5, 6], vec![7, 8, 9, 10]];
        let batch = TokenBatch::from_sequences(&seqs, &cfg).unwrap();
        assert_eq!(batch.batch_size, 4);
        assert_eq!(batch.seq_len, 4);
        // shortest gets 3 pad tokens
        assert_eq!(batch.mask(0).unwrap(), &[1, 0, 0, 0]);
        // longest gets 0
        assert_eq!(batch.mask(3).unwrap(), &[1, 1, 1, 1]);
    }

    #[test]
    fn batch_uniform_lengths_no_padding() {
        let cfg = TokenizerConfig::new(100, 128);
        let seqs = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let batch = TokenBatch::from_sequences(&seqs, &cfg).unwrap();
        assert!(batch.attention_mask.iter().all(|&m| m == 1));
    }

    #[test]
    fn batch_max_seq_len_clamps() {
        let cfg = TokenizerConfig::new(100, 4).with_padding_side(PaddingSide::Right);
        let seqs = vec![vec![1, 2, 3, 4, 5, 6], vec![7, 8]];
        let batch = TokenBatch::from_sequences(&seqs, &cfg).unwrap();
        // max_seq_len=4 clamps both
        assert_eq!(batch.seq_len, 4);
        assert_eq!(batch.sequence(0).unwrap(), &[1, 2, 3, 4]);
        assert_eq!(batch.sequence(1).unwrap(), &[7, 8, 0, 0]);
    }

    #[test]
    fn batch_custom_pad_id() {
        let st = SpecialTokens::new(1, 2, 99, 3);
        let cfg = TokenizerConfig::new(100, 128)
            .with_special_tokens(st)
            .with_padding_side(PaddingSide::Right);
        let seqs = vec![vec![10, 20], vec![30]];
        let batch = TokenBatch::from_sequences(&seqs, &cfg).unwrap();
        // seq 1 is padded with 99
        assert_eq!(batch.sequence(1).unwrap(), &[30, 99]);
    }

    // ── PromptTemplate ───────────────────────────────────────────

    #[test]
    fn template_chatml_basic() {
        let tpl = PromptTemplate::new(TemplateFormat::ChatML);
        let out = tpl.format_instruction("What is 2+2?");
        assert!(out.contains("<|im_start|>user"));
        assert!(out.contains("What is 2+2?"));
        assert!(out.contains("<|im_end|>"));
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn template_chatml_with_system() {
        let tpl =
            PromptTemplate::new(TemplateFormat::ChatML).with_system_prompt("You are helpful.");
        let out = tpl.format_instruction("Hi");
        assert!(out.contains("<|im_start|>system"));
        assert!(out.contains("You are helpful."));
    }

    #[test]
    fn template_chatml_multi_turn() {
        let tpl = PromptTemplate::new(TemplateFormat::ChatML);
        let msgs = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there"),
            ChatMessage::user("How are you?"),
        ];
        let out = tpl.apply(&msgs);
        // 3 messages + assistant prompt
        assert_eq!(out.matches("<|im_start|>").count(), 4);
        assert_eq!(out.matches("<|im_end|>").count(), 3);
    }

    #[test]
    fn template_llama2_basic() {
        let tpl = PromptTemplate::new(TemplateFormat::Llama2);
        let out = tpl.format_instruction("What is Rust?");
        assert!(out.contains("[INST]"));
        assert!(out.contains("[/INST]"));
        assert!(out.contains("What is Rust?"));
    }

    #[test]
    fn template_llama2_with_system() {
        let tpl =
            PromptTemplate::new(TemplateFormat::Llama2).with_system_prompt("You are an expert.");
        let out = tpl.format_instruction("Explain monads");
        assert!(out.contains("<<SYS>>"));
        assert!(out.contains("You are an expert."));
        assert!(out.contains("<</SYS>>"));
    }

    #[test]
    fn template_simple_format() {
        let tpl = PromptTemplate::new(TemplateFormat::Simple);
        let out = tpl.format_instruction("Do the thing");
        assert!(out.contains("### Instruction:"));
        assert!(out.contains("Do the thing"));
        assert!(out.ends_with("### Response:\n"));
    }

    #[test]
    fn template_raw() {
        let tpl = PromptTemplate::new(TemplateFormat::Raw);
        let msgs = vec![ChatMessage::user("one"), ChatMessage::user("two")];
        let out = tpl.apply(&msgs);
        assert_eq!(out, "one\ntwo");
    }

    #[test]
    fn template_format_display() {
        assert_eq!(TemplateFormat::ChatML.to_string(), "chatml");
        assert_eq!(TemplateFormat::Llama2.to_string(), "llama2");
        assert_eq!(TemplateFormat::Simple.to_string(), "simple");
        assert_eq!(TemplateFormat::Raw.to_string(), "raw");
    }

    #[test]
    fn template_empty_messages() {
        let tpl = PromptTemplate::new(TemplateFormat::ChatML);
        let out = tpl.apply(&[]);
        assert_eq!(out, "<|im_start|>assistant\n");
    }

    // ── TokenStats ───────────────────────────────────────────────

    #[test]
    fn stats_empty() {
        let stats = TokenStats::new();
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.tokens_per_sec(), 0.0);
        assert_eq!(stats.avg_seq_len(), 0.0);
        assert_eq!(stats.vocab_coverage(), 0);
    }

    #[test]
    fn stats_record() {
        let mut stats = TokenStats::new();
        stats.record(&[1, 2, 3], Duration::from_millis(10));
        assert_eq!(stats.total_tokens, 3);
        assert_eq!(stats.num_sequences, 1);
        assert_eq!(stats.vocab_coverage(), 3);
    }

    #[test]
    fn stats_tokens_per_sec() {
        let mut stats = TokenStats::new();
        stats.record(&[1, 2, 3, 4, 5], Duration::from_secs(1));
        assert!((stats.tokens_per_sec() - 5.0).abs() < 0.01);
    }

    #[test]
    fn stats_avg_seq_len() {
        let mut stats = TokenStats::new();
        stats.record(&[1, 2], Duration::from_millis(1));
        stats.record(&[3, 4, 5, 6], Duration::from_millis(1));
        assert!((stats.avg_seq_len() - 3.0).abs() < 0.01);
    }

    #[test]
    fn stats_vocab_coverage_deduplicates() {
        let mut stats = TokenStats::new();
        stats.record(&[1, 2, 1, 2, 1], Duration::from_millis(1));
        assert_eq!(stats.vocab_coverage(), 2);
    }

    #[test]
    fn stats_vocab_coverage_ratio() {
        let mut stats = TokenStats::new();
        stats.record(&[0, 1, 2, 3, 4], Duration::from_millis(1));
        assert!((stats.vocab_coverage_ratio(10) - 0.5).abs() < 0.01);
        assert_eq!(stats.vocab_coverage_ratio(0), 0.0);
    }

    #[test]
    fn stats_merge() {
        let mut a = TokenStats::new();
        a.record(&[1, 2, 3], Duration::from_millis(10));
        let mut b = TokenStats::new();
        b.record(&[3, 4, 5], Duration::from_millis(20));
        a.merge(&b);
        assert_eq!(a.total_tokens, 6);
        assert_eq!(a.num_sequences, 2);
        assert_eq!(a.vocab_coverage(), 5); // {1,2,3,4,5}
        assert_eq!(a.total_duration, Duration::from_millis(30));
    }

    #[test]
    fn stats_display() {
        let stats = TokenStats::new();
        let s = stats.to_string();
        assert!(s.contains("TokenStats"));
        assert!(s.contains("tokens: 0"));
    }

    // ── encode_batch_with_stats / prepare_batch ──────────────────

    #[test]
    fn encode_batch_with_stats_basic() {
        let enc = test_encoder();
        let (ids, stats) = encode_batch_with_stats(&enc, &["hello", "world the"]).unwrap();
        assert_eq!(ids.len(), 2);
        assert_eq!(stats.num_sequences, 2);
        assert!(stats.total_tokens > 0);
    }

    #[test]
    fn prepare_batch_basic() {
        let enc = test_encoder();
        let (batch, stats) = prepare_batch(&enc, &["hello", "world the"]).unwrap();
        assert_eq!(batch.batch_size, 2);
        assert!(stats.num_sequences == 2);
        // Both sequences same padded length
        assert!(batch.seq_len >= 2);
    }

    // ── PaddingSide display ──────────────────────────────────────

    #[test]
    fn padding_side_display() {
        assert_eq!(PaddingSide::Left.to_string(), "left");
        assert_eq!(PaddingSide::Right.to_string(), "right");
    }

    // ── Edge cases ───────────────────────────────────────────────

    #[test]
    fn encode_whitespace_only() {
        let enc = test_encoder();
        let ids = enc.encode("   ").unwrap();
        // BOS only — no words to encode
        assert_eq!(ids, vec![enc.config.special_tokens.bos_id]);
    }

    #[test]
    fn encode_multiple_spaces() {
        let enc = test_encoder();
        let ids = enc.encode("hello   world").unwrap();
        // BOS + hello + world (split_whitespace collapses spaces)
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn batch_all_same_length() {
        let cfg = TokenizerConfig::new(100, 128).with_padding_side(PaddingSide::Right);
        let seqs = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let batch = TokenBatch::from_sequences(&seqs, &cfg).unwrap();
        // No padding needed
        assert_eq!(batch.total_real_tokens(), 6);
        assert!(batch.attention_mask.iter().all(|&m| m == 1));
    }

    #[test]
    fn batch_single_token_sequences() {
        let cfg = TokenizerConfig::new(100, 128).with_padding_side(PaddingSide::Right);
        let seqs = vec![vec![1], vec![2], vec![3]];
        let batch = TokenBatch::from_sequences(&seqs, &cfg).unwrap();
        assert_eq!(batch.seq_len, 1);
        assert_eq!(batch.total_real_tokens(), 3);
    }

    #[test]
    fn decode_with_special_sep() {
        let st = SpecialTokens::default().with_sep(50);
        let dec = TokenDecoder::new(HashMap::new(), st);
        let text = dec.decode_with_special(&[50]);
        assert_eq!(text, "<SEP>");
    }

    #[test]
    fn template_llama2_multi_turn() {
        let tpl = PromptTemplate::new(TemplateFormat::Llama2);
        let msgs = vec![
            ChatMessage::user("Hi"),
            ChatMessage::assistant("Hello!"),
            ChatMessage::user("How are you?"),
        ];
        let out = tpl.apply(&msgs);
        assert!(out.contains("[INST]"));
        assert!(out.contains("[/INST]"));
        assert!(out.contains("</s>"));
    }

    #[test]
    fn template_simple_with_system() {
        let tpl = PromptTemplate::new(TemplateFormat::Simple).with_system_prompt("Be brief.");
        let out = tpl.format_instruction("Summarize");
        assert!(out.contains("### System:"));
        assert!(out.contains("Be brief."));
        assert!(out.contains("### Instruction:"));
    }

    #[test]
    fn template_format_getter() {
        let tpl = PromptTemplate::new(TemplateFormat::ChatML);
        assert_eq!(tpl.format(), TemplateFormat::ChatML);
    }

    // ── Property-style tests ─────────────────────────────────────

    #[test]
    fn property_decode_encode_preserves_known_words() {
        let enc = test_encoder();
        let dec = test_decoder(&enc);
        let known_words = ["hello", "world", "the", "a", "is", "of", "to"];
        for &w in &known_words {
            let ids = enc.encode(w).unwrap();
            let text = dec.decode(&ids);
            assert_eq!(text, w, "roundtrip failed for '{w}'");
        }
    }

    #[test]
    fn property_batch_mask_length_equals_ids_length() {
        let cfg = TokenizerConfig::new(100, 128).with_padding_side(PaddingSide::Right);
        let seqs: Vec<Vec<u32>> = (1..=5).map(|n| (0..n).collect()).collect();
        let batch = TokenBatch::from_sequences(&seqs, &cfg).unwrap();
        assert_eq!(batch.token_ids.len(), batch.attention_mask.len());
        assert_eq!(batch.token_ids.len(), batch.batch_size * batch.seq_len);
    }

    #[test]
    fn property_attention_mask_sums_to_real_tokens() {
        let cfg = TokenizerConfig::new(100, 128).with_padding_side(PaddingSide::Right);
        let seqs = vec![vec![1, 2, 3], vec![4], vec![5, 6]];
        let total_real: usize = seqs.iter().map(|s| s.len()).sum();
        let batch = TokenBatch::from_sequences(&seqs, &cfg).unwrap();
        assert_eq!(batch.total_real_tokens(), total_real);
    }

    #[test]
    fn property_left_pad_real_tokens_at_end() {
        let cfg = TokenizerConfig::new(100, 128).with_padding_side(PaddingSide::Left);
        let seqs = vec![vec![10, 20, 30], vec![40]];
        let batch = TokenBatch::from_sequences(&seqs, &cfg).unwrap();
        let mask = batch.mask(1).unwrap();
        // Left padding: trailing entries should be 1
        assert_eq!(mask[mask.len() - 1], 1);
        // Leading entries should be 0
        assert_eq!(mask[0], 0);
    }

    #[test]
    fn property_right_pad_real_tokens_at_start() {
        let cfg = TokenizerConfig::new(100, 128).with_padding_side(PaddingSide::Right);
        let seqs = vec![vec![10, 20, 30], vec![40]];
        let batch = TokenBatch::from_sequences(&seqs, &cfg).unwrap();
        let mask = batch.mask(1).unwrap();
        // Right padding: leading entries should be 1
        assert_eq!(mask[0], 1);
        // Trailing entries should be 0
        assert_eq!(mask[mask.len() - 1], 0);
    }

    #[test]
    fn stats_default_impl() {
        let stats = TokenStats::default();
        assert_eq!(stats.total_tokens, 0);
    }

    #[test]
    fn encoder_config_accessor() {
        let enc = test_encoder();
        assert!(enc.config().vocab_size > 0);
        assert!(enc.config().max_seq_len > 0);
    }

    #[test]
    fn encoder_vocab_accessor() {
        let enc = test_encoder();
        assert!(enc.vocab().contains_key("hello"));
        assert!(!enc.vocab().contains_key("nonexistent_xyz"));
    }
}
