//! Full tokenizer pipeline for GPU inference: pre-tokenization,
//! normalization, BPE / `WordPiece` / `SentencePiece` encoding,
//! post-processing, batching, and metrics.
//!
//! Every component runs on the CPU as a reference implementation that can
//! later be off-loaded to GPU kernels behind the HAL traits.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::missing_panics_doc)]

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ── PipelineConfig ──────────────────────────────────────────────────────────

/// Configuration for the full tokenizer pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Maximum sequence length (in tokens) after encoding.
    pub max_length: usize,
    /// Whether to truncate sequences that exceed `max_length`.
    pub truncation: bool,
    /// Whether to pad sequences shorter than `max_length`.
    pub padding: bool,
    /// Token ID used for padding.
    pub pad_token_id: u32,
    /// Beginning-of-sequence token string.
    pub bos_token: Option<String>,
    /// End-of-sequence token string.
    pub eos_token: Option<String>,
    /// Unknown-token string.
    pub unk_token: String,
    /// CLS token string (BERT-style).
    pub cls_token: Option<String>,
    /// SEP token string (BERT-style).
    pub sep_token: Option<String>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_length: 512,
            truncation: true,
            padding: true,
            pad_token_id: 0,
            bos_token: None,
            eos_token: None,
            unk_token: "[UNK]".to_string(),
            cls_token: None,
            sep_token: None,
        }
    }
}

impl PipelineConfig {
    /// Create a BERT-style config with CLS/SEP tokens.
    #[must_use]
    pub fn bert_default() -> Self {
        Self {
            max_length: 512,
            truncation: true,
            padding: true,
            pad_token_id: 0,
            bos_token: None,
            eos_token: None,
            unk_token: "[UNK]".to_string(),
            cls_token: Some("[CLS]".to_string()),
            sep_token: Some("[SEP]".to_string()),
        }
    }

    /// Create a GPT-style config with BOS/EOS tokens.
    #[must_use]
    pub fn gpt_default() -> Self {
        Self {
            max_length: 1024,
            truncation: true,
            padding: false,
            pad_token_id: 0,
            bos_token: Some("<s>".to_string()),
            eos_token: Some("</s>".to_string()),
            unk_token: "<unk>".to_string(),
            cls_token: None,
            sep_token: None,
        }
    }
}

// ── PreTokenizer ────────────────────────────────────────────────────────────

/// Strategy used to split raw text before encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreTokenizeStrategy {
    /// Split on Unicode whitespace boundaries.
    Whitespace,
    /// Byte-level preparation for BPE (no whitespace collapse).
    ByteLevel,
    /// Split on punctuation boundaries.
    Punctuation,
}

/// Pre-tokenizer: splits raw text into candidate tokens
/// before the main encoding step.
#[derive(Debug, Clone)]
pub struct PreTokenizer {
    strategy: PreTokenizeStrategy,
}

impl PreTokenizer {
    /// Create a new pre-tokenizer with the given strategy.
    #[must_use]
    pub const fn new(strategy: PreTokenizeStrategy) -> Self {
        Self { strategy }
    }

    /// Split `text` into pre-tokenized segments.
    #[must_use]
    pub fn pre_tokenize(&self, text: &str) -> Vec<String> {
        match self.strategy {
            PreTokenizeStrategy::Whitespace => text.split_whitespace().map(String::from).collect(),
            PreTokenizeStrategy::ByteLevel => {
                if text.is_empty() {
                    return Vec::new();
                }
                // Byte-level: each character becomes its own token;
                // spaces are replaced with the Ġ marker (U+0120).
                let replaced = text.replace(' ', "\u{0120}");
                replaced.chars().map(|c| c.to_string()).collect()
            }
            PreTokenizeStrategy::Punctuation => split_on_punctuation(text),
        }
    }
}

/// Helper: split text on punctuation boundaries, keeping both
/// punctuation tokens and word tokens.
fn split_on_punctuation(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_ascii_punctuation() {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            tokens.push(ch.to_string());
        } else if ch.is_whitespace() {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

// ── TokenNormalizer ─────────────────────────────────────────────────────────

/// Flags controlling which normalization passes to apply.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NormalizationFlags {
    /// Apply Unicode NFKC-like folding (ASCII approximation).
    pub nfkc: bool,
    /// Lower-case the text.
    pub lowercase: bool,
    /// Strip combining diacritical marks (accent stripping).
    pub strip_accents: bool,
}

/// Text normalizer applied before pre-tokenization.
#[derive(Debug, Clone)]
pub struct TokenNormalizer {
    flags: NormalizationFlags,
}

impl TokenNormalizer {
    /// Create a normalizer with the given flags.
    #[must_use]
    pub const fn new(flags: NormalizationFlags) -> Self {
        Self { flags }
    }

    /// Create a normalizer that lower-cases text.
    #[must_use]
    pub fn lowercase() -> Self {
        Self { flags: NormalizationFlags { lowercase: true, ..Default::default() } }
    }

    /// Normalize `text` according to the configured flags.
    #[must_use]
    pub fn normalize(&self, text: &str) -> String {
        let mut out = text.to_string();
        if self.flags.nfkc {
            out = ascii_nfkc_fold(&out);
        }
        if self.flags.strip_accents {
            out = strip_accents_ascii(&out);
        }
        if self.flags.lowercase {
            out = out.to_lowercase();
        }
        out
    }
}

/// Minimal ASCII NFKC-like folding: curly quotes → straight, em-dash → --, etc.
fn ascii_nfkc_fold(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\u{201C}' | '\u{201D}' => '"',
            '\u{2018}' | '\u{2019}' => '\'',
            '\u{2014}' => '-',
            '\u{2026}' => '.',
            '\u{00A0}' => ' ',
            other => other,
        })
        .collect()
}

/// Strip common accented Latin characters to their base letter.
fn strip_accents_ascii(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'á' | 'à' | 'â' | 'ä' | 'ã' => 'a',
            'é' | 'è' | 'ê' | 'ë' => 'e',
            'í' | 'ì' | 'î' | 'ï' => 'i',
            'ó' | 'ò' | 'ô' | 'ö' | 'õ' => 'o',
            'ú' | 'ù' | 'û' | 'ü' => 'u',
            'ñ' => 'n',
            'ç' => 'c',
            other => other,
        })
        .collect()
}

// ── BPEEncoder ──────────────────────────────────────────────────────────────

/// Byte-pair encoding implementation.
///
/// Holds a merge table (ordered pairs) and a vocabulary mapping.
/// The CPU reference iteratively merges the most frequent pair until
/// no more merges apply.
#[derive(Debug, Clone)]
pub struct BPEEncoder {
    /// Ordered merge rules: `(left, right)` → merged token.
    merges: Vec<(String, String)>,
    /// Token string → token ID.
    vocab: HashMap<String, u32>,
    /// Token ID → token string (reverse lookup).
    id_to_token: HashMap<u32, String>,
    /// Unknown token string.
    unk_token: String,
}

impl BPEEncoder {
    /// Build a BPE encoder from merge rules and a vocabulary.
    #[must_use]
    pub fn new(
        merges: Vec<(String, String)>,
        vocab: HashMap<String, u32>,
        unk_token: &str,
    ) -> Self {
        let id_to_token: HashMap<u32, String> =
            vocab.iter().map(|(k, &v)| (v, k.clone())).collect();
        Self { merges, vocab, id_to_token, unk_token: unk_token.to_string() }
    }

    /// Vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Encode a single pre-tokenized word into token IDs.
    #[must_use]
    pub fn encode_word(&self, word: &str) -> Vec<u32> {
        if word.is_empty() {
            return Vec::new();
        }
        let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();

        for (left, right) in &self.merges {
            let merged = format!("{left}{right}");
            let mut i = 0;
            while i + 1 < symbols.len() {
                if symbols[i] == *left && symbols[i + 1] == *right {
                    symbols[i].clone_from(&merged);
                    symbols.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }

        symbols
            .iter()
            .map(|s| {
                self.vocab
                    .get(s)
                    .copied()
                    .unwrap_or_else(|| self.vocab.get(&self.unk_token).copied().unwrap_or(0))
            })
            .collect()
    }

    /// Encode a sequence of pre-tokenized words.
    #[must_use]
    pub fn encode(&self, words: &[String]) -> Vec<u32> {
        words.iter().flat_map(|w| self.encode_word(w)).collect()
    }

    /// Decode token IDs back to a string.
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter().filter_map(|id| self.id_to_token.get(id)).cloned().collect::<String>()
    }
}

// ── WordPieceEncoder ────────────────────────────────────────────────────────

/// `WordPiece` encoding implementation (BERT-style).
///
/// Greedily matches the longest prefix in the vocabulary, prepending
/// the continuation prefix (`##`) for sub-word tokens.
#[derive(Debug, Clone)]
pub struct WordPieceEncoder {
    /// Token string → token ID.
    vocab: HashMap<String, u32>,
    /// Token ID → token string.
    id_to_token: HashMap<u32, String>,
    /// Unknown token string.
    unk_token: String,
    /// Prefix for continuation sub-words (default `"##"`).
    continuing_prefix: String,
    /// Maximum characters to consider in a single word.
    max_word_len: usize,
}

impl WordPieceEncoder {
    /// Build a `WordPiece` encoder.
    #[must_use]
    pub fn new(vocab: HashMap<String, u32>, unk_token: &str) -> Self {
        let id_to_token: HashMap<u32, String> =
            vocab.iter().map(|(k, &v)| (v, k.clone())).collect();
        Self {
            vocab,
            id_to_token,
            unk_token: unk_token.to_string(),
            continuing_prefix: "##".to_string(),
            max_word_len: 200,
        }
    }

    /// Vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Encode a single word into `WordPiece` token IDs.
    #[must_use]
    pub fn encode_word(&self, word: &str) -> Vec<u32> {
        if word.is_empty() {
            return Vec::new();
        }
        if word.chars().count() > self.max_word_len {
            return vec![self.vocab.get(&self.unk_token).copied().unwrap_or(0)];
        }

        let chars: Vec<char> = word.chars().collect();
        let mut tokens = Vec::new();
        let mut start = 0;

        while start < chars.len() {
            let mut end = chars.len();
            let mut found = false;

            while start < end {
                let substr: String = chars[start..end].iter().collect();
                let candidate = if start == 0 {
                    substr.clone()
                } else {
                    format!("{}{substr}", self.continuing_prefix)
                };

                if let Some(&id) = self.vocab.get(&candidate) {
                    tokens.push(id);
                    found = true;
                    start = end;
                    break;
                }
                end -= 1;
            }

            if !found {
                tokens.push(self.vocab.get(&self.unk_token).copied().unwrap_or(0));
                start += 1;
            }
        }
        tokens
    }

    /// Encode a sequence of pre-tokenized words.
    #[must_use]
    pub fn encode(&self, words: &[String]) -> Vec<u32> {
        words.iter().flat_map(|w| self.encode_word(w)).collect()
    }

    /// Decode token IDs back to a string.
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut parts: Vec<String> = Vec::new();
        for id in ids {
            if let Some(tok) = self.id_to_token.get(id) {
                if let Some(stripped) = tok.strip_prefix(&self.continuing_prefix) {
                    if let Some(last) = parts.last_mut() {
                        last.push_str(stripped);
                    } else {
                        parts.push(stripped.to_string());
                    }
                } else {
                    parts.push(tok.clone());
                }
            }
        }
        parts.join(" ")
    }
}

// ── SentencePieceEncoder ────────────────────────────────────────────────────

/// `SentencePiece` (unigram) encoding implementation.
///
/// Uses a scored vocabulary to find the tokenization that maximises
/// the sum of log-probabilities (Viterbi on a simple unigram model).
#[derive(Debug, Clone)]
pub struct SentencePieceEncoder {
    /// Token string → (token ID, log-probability score).
    vocab: HashMap<String, (u32, f64)>,
    /// Token ID → token string.
    id_to_token: HashMap<u32, String>,
    /// Unknown token string.
    unk_token: String,
    /// Prefix used for the start of a word (e.g. `▁`).
    word_prefix: String,
}

impl SentencePieceEncoder {
    /// Build a `SentencePiece` encoder from scored vocabulary entries.
    ///
    /// Each entry is `(token_string, token_id, log_prob)`.
    #[must_use]
    pub fn new(entries: Vec<(String, u32, f64)>, unk_token: &str) -> Self {
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();
        for (tok, id, score) in entries {
            id_to_token.insert(id, tok.clone());
            vocab.insert(tok, (id, score));
        }
        Self {
            vocab,
            id_to_token,
            unk_token: unk_token.to_string(),
            word_prefix: "\u{2581}".to_string(), // ▁
        }
    }

    /// Vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Encode text using the unigram Viterbi algorithm.
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        // Prepend word-prefix, replacing leading spaces.
        let processed =
            format!("{}{}", self.word_prefix, text.trim_start().replace(' ', &self.word_prefix));
        self.viterbi_encode(&processed)
    }

    /// Viterbi forward pass over `text`.
    fn viterbi_encode(&self, text: &str) -> Vec<u32> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        // best[i] = (best_score, best_token_len) to reach position i
        let mut best: Vec<(f64, usize)> = vec![(f64::NEG_INFINITY, 0); n + 1];
        best[0] = (0.0, 0);

        for i in 0..n {
            if best[i].0 == f64::NEG_INFINITY && i > 0 {
                continue;
            }
            let max_end = n.min(i + 32); // limit token length
            for end in (i + 1)..=max_end {
                let substr: String = chars[i..end].iter().collect();
                if let Some(&(_id, score)) = self.vocab.get(&substr) {
                    let new_score = best[i].0 + score;
                    if new_score > best[end].0 {
                        best[end] = (new_score, end - i);
                    }
                }
            }
        }

        // Back-track to recover the best tokenization.
        let mut ids = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let tok_len = best[pos].1;
            if tok_len == 0 {
                // Unreachable via known vocab — emit UNK and step back.
                ids.push(self.vocab.get(&self.unk_token).map_or(0, |&(id, _)| id));
                pos -= 1;
                continue;
            }
            let start = pos - tok_len;
            let substr: String = chars[start..pos].iter().collect();
            if let Some(&(id, _)) = self.vocab.get(&substr) {
                ids.push(id);
            }
            pos = start;
        }

        ids.reverse();
        ids
    }

    /// Decode token IDs back to text.
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        let raw: String =
            ids.iter().filter_map(|id| self.id_to_token.get(id)).cloned().collect::<String>();
        // Remove word-prefix markers and trim.
        raw.replace(&self.word_prefix, " ").trim().to_string()
    }
}

// ── PostProcessor ───────────────────────────────────────────────────────────

/// Template describing how special tokens are added around encoded IDs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PostProcessTemplate {
    /// No special tokens added.
    None,
    /// BERT-style: `[CLS] <tokens> [SEP]`.
    BertSingle { cls_id: u32, sep_id: u32 },
    /// BERT-style pair: `[CLS] <A> [SEP] <B> [SEP]`.
    BertPair { cls_id: u32, sep_id: u32 },
    /// GPT-style: `<BOS> <tokens> <EOS>`.
    GptSingle { bos_id: u32, eos_id: u32 },
}

/// Post-processor: wraps encoded token IDs with special tokens.
#[derive(Debug, Clone)]
pub struct PostProcessor {
    template: PostProcessTemplate,
}

impl PostProcessor {
    /// Create a post-processor with the given template.
    #[must_use]
    pub const fn new(template: PostProcessTemplate) -> Self {
        Self { template }
    }

    /// Apply post-processing to a single sequence of token IDs.
    #[must_use]
    pub fn process_single(&self, ids: &[u32]) -> Vec<u32> {
        match &self.template {
            PostProcessTemplate::None => ids.to_vec(),
            PostProcessTemplate::BertSingle { cls_id, sep_id }
            | PostProcessTemplate::BertPair { cls_id, sep_id } => {
                let mut out = Vec::with_capacity(ids.len() + 2);
                out.push(*cls_id);
                out.extend_from_slice(ids);
                out.push(*sep_id);
                out
            }
            PostProcessTemplate::GptSingle { bos_id, eos_id } => {
                let mut out = Vec::with_capacity(ids.len() + 2);
                out.push(*bos_id);
                out.extend_from_slice(ids);
                out.push(*eos_id);
                out
            }
        }
    }

    /// Apply post-processing to a pair of sequences (e.g. QA).
    #[must_use]
    pub fn process_pair(&self, ids_a: &[u32], ids_b: &[u32]) -> Vec<u32> {
        if let PostProcessTemplate::BertPair { cls_id, sep_id } = &self.template {
            let mut out = Vec::with_capacity(ids_a.len() + ids_b.len() + 3);
            out.push(*cls_id);
            out.extend_from_slice(ids_a);
            out.push(*sep_id);
            out.extend_from_slice(ids_b);
            out.push(*sep_id);
            out
        } else {
            // Fall back to concatenation for non-pair templates.
            let mut out = self.process_single(ids_a);
            out.extend_from_slice(ids_b);
            out
        }
    }
}

// ── BatchTokenizer ──────────────────────────────────────────────────────────

/// Batched tokenization: encodes multiple sequences with
/// optional padding and truncation.
#[derive(Debug, Clone)]
pub struct BatchTokenizer {
    config: PipelineConfig,
}

/// Output from batched tokenization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchOutput {
    /// Token IDs for each sequence.
    pub input_ids: Vec<Vec<u32>>,
    /// Attention masks (1 = real, 0 = padding).
    pub attention_masks: Vec<Vec<u8>>,
    /// Original lengths before padding/truncation.
    pub original_lengths: Vec<usize>,
}

impl BatchTokenizer {
    /// Create a batch tokenizer from a pipeline config.
    #[must_use]
    pub const fn new(config: PipelineConfig) -> Self {
        Self { config }
    }

    /// Tokenize a batch of already-encoded ID sequences, applying
    /// truncation and padding according to the config.
    #[must_use]
    pub fn process_batch(&self, encoded: &[Vec<u32>]) -> BatchOutput {
        let mut input_ids: Vec<Vec<u32>> = Vec::new();
        let mut original_lengths: Vec<usize> = Vec::new();

        for seq in encoded {
            original_lengths.push(seq.len());
            let mut ids = seq.clone();
            if self.config.truncation && ids.len() > self.config.max_length {
                ids.truncate(self.config.max_length);
            }
            input_ids.push(ids);
        }

        // Determine the target length for padding.
        let target_len = if self.config.padding {
            let max_in_batch = input_ids.iter().map(Vec::len).max().unwrap_or(0);
            max_in_batch.min(self.config.max_length)
        } else {
            0
        };

        let mut attention_masks: Vec<Vec<u8>> = Vec::new();
        for ids in &mut input_ids {
            let real_len = ids.len();
            let mask_len = if self.config.padding { target_len } else { real_len };
            let mut mask = vec![1u8; real_len];
            if self.config.padding && real_len < target_len {
                ids.resize(target_len, self.config.pad_token_id);
                mask.resize(target_len, 0);
            }
            // Ensure mask length matches even if no padding needed.
            mask.resize(mask_len, 0);
            attention_masks.push(mask);
        }

        BatchOutput { input_ids, attention_masks, original_lengths }
    }
}

// ── TokenizerMetrics ────────────────────────────────────────────────────────

/// Accumulated metrics for tokenization operations.
#[derive(Debug, Clone)]
pub struct TokenizerMetrics {
    /// Total sequences processed.
    pub sequences_processed: u64,
    /// Total tokens produced.
    pub tokens_produced: u64,
    /// Total characters consumed.
    pub characters_consumed: u64,
    /// Cumulative encoding time.
    pub encoding_duration: Duration,
    /// Per-token ID histogram (`token_id` → count).
    pub token_histogram: HashMap<u32, u64>,
}

impl Default for TokenizerMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenizerMetrics {
    /// Create empty metrics.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sequences_processed: 0,
            tokens_produced: 0,
            characters_consumed: 0,
            encoding_duration: Duration::ZERO,
            token_histogram: HashMap::new(),
        }
    }

    /// Record a single encoding operation.
    pub fn record(&mut self, input_chars: usize, output_ids: &[u32], elapsed: Duration) {
        self.sequences_processed += 1;
        self.tokens_produced += output_ids.len() as u64;
        self.characters_consumed += input_chars as u64;
        self.encoding_duration += elapsed;
        for &id in output_ids {
            *self.token_histogram.entry(id).or_insert(0) += 1;
        }
    }

    /// Tokens per second (0 if no time elapsed).
    #[must_use]
    pub fn tokens_per_second(&self) -> f64 {
        let secs = self.encoding_duration.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        self.tokens_produced as f64 / secs
    }

    /// Average tokens per sequence.
    #[must_use]
    pub fn avg_tokens_per_sequence(&self) -> f64 {
        if self.sequences_processed == 0 {
            return 0.0;
        }
        self.tokens_produced as f64 / self.sequences_processed as f64
    }

    /// Characters per token ratio.
    #[must_use]
    pub fn chars_per_token(&self) -> f64 {
        if self.tokens_produced == 0 {
            return 0.0;
        }
        self.characters_consumed as f64 / self.tokens_produced as f64
    }

    /// Number of unique token IDs seen.
    #[must_use]
    pub fn unique_tokens(&self) -> usize {
        self.token_histogram.len()
    }

    /// Reset all counters.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

// ── Encoding algorithm selection ────────────────────────────────────────────

/// Which encoding algorithm to use in the pipeline.
#[derive(Debug, Clone)]
pub enum EncoderKind {
    /// Byte-pair encoding.
    Bpe(BPEEncoder),
    /// `WordPiece` (BERT-style).
    WordPiece(WordPieceEncoder),
    /// `SentencePiece` (unigram).
    SentencePiece(SentencePieceEncoder),
}

// ── TokenizerPipelineEngine ─────────────────────────────────────────────────

/// Unified tokenizer pipeline composing normalizer → pre-tokenizer →
/// encoder → post-processor, with optional batching and metrics.
#[derive(Debug, Clone)]
pub struct TokenizerPipelineEngine {
    /// Pipeline configuration.
    pub config: PipelineConfig,
    /// Optional text normalizer.
    normalizer: Option<TokenNormalizer>,
    /// Pre-tokenizer.
    pre_tokenizer: PreTokenizer,
    /// Main encoder.
    encoder: EncoderKind,
    /// Post-processor.
    post_processor: PostProcessor,
    /// Batch handler.
    batch_tokenizer: BatchTokenizer,
    /// Accumulated metrics.
    metrics: TokenizerMetrics,
}

impl TokenizerPipelineEngine {
    /// Build a pipeline from its components.
    #[must_use]
    pub fn new(
        config: PipelineConfig,
        normalizer: Option<TokenNormalizer>,
        pre_tokenizer: PreTokenizer,
        encoder: EncoderKind,
        post_processor: PostProcessor,
    ) -> Self {
        let batch_tokenizer = BatchTokenizer::new(config.clone());
        Self {
            config,
            normalizer,
            pre_tokenizer,
            encoder,
            post_processor,
            batch_tokenizer,
            metrics: TokenizerMetrics::new(),
        }
    }

    /// Encode a single text string through the full pipeline.
    pub fn encode(&mut self, text: &str) -> Vec<u32> {
        let start = Instant::now();

        // 1. Normalize.
        let normalized =
            self.normalizer.as_ref().map_or_else(|| text.to_string(), |n| n.normalize(text));

        // 2. Pre-tokenize.
        let segments = self.pre_tokenizer.pre_tokenize(&normalized);

        // 3. Encode.
        let raw_ids = match &self.encoder {
            EncoderKind::Bpe(enc) => enc.encode(&segments),
            EncoderKind::WordPiece(enc) => enc.encode(&segments),
            EncoderKind::SentencePiece(enc) => enc.encode(&normalized),
        };

        // 4. Post-process.
        let ids = self.post_processor.process_single(&raw_ids);

        self.metrics.record(text.len(), &ids, start.elapsed());
        ids
    }

    /// Decode token IDs back to text.
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        match &self.encoder {
            EncoderKind::Bpe(enc) => enc.decode(ids),
            EncoderKind::WordPiece(enc) => enc.decode(ids),
            EncoderKind::SentencePiece(enc) => enc.decode(ids),
        }
    }

    /// Encode a batch of texts, returning padded/truncated output.
    pub fn encode_batch(&mut self, texts: &[&str]) -> BatchOutput {
        let encoded: Vec<Vec<u32>> = texts.iter().map(|t| self.encode(t)).collect();
        self.batch_tokenizer.process_batch(&encoded)
    }

    /// Reference to the accumulated metrics.
    #[must_use]
    pub const fn metrics(&self) -> &TokenizerMetrics {
        &self.metrics
    }

    /// Reset metrics.
    pub fn reset_metrics(&mut self) {
        self.metrics.reset();
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────

    /// Build a small BPE vocabulary + merges for testing.
    fn test_bpe() -> BPEEncoder {
        let mut vocab = HashMap::new();
        for (i, c) in "abcdefghijklmnopqrstuvwxyz ".chars().enumerate() {
            vocab.insert(c.to_string(), i as u32);
        }
        vocab.insert("[UNK]".to_string(), 100);
        vocab.insert("he".to_string(), 101);
        vocab.insert("ll".to_string(), 102);
        vocab.insert("lo".to_string(), 103);
        vocab.insert("hell".to_string(), 104);
        vocab.insert("hello".to_string(), 105);
        let merges = vec![
            ("h".into(), "e".into()),
            ("l".into(), "l".into()),
            ("l".into(), "o".into()),
            ("he".into(), "ll".into()),
            ("hell".into(), "o".into()),
        ];
        BPEEncoder::new(merges, vocab, "[UNK]")
    }

    /// Build a small WordPiece vocabulary for testing.
    fn test_wordpiece() -> WordPieceEncoder {
        let mut vocab = HashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("[CLS]".to_string(), 1);
        vocab.insert("[SEP]".to_string(), 2);
        vocab.insert("[PAD]".to_string(), 3);
        vocab.insert("hello".to_string(), 100);
        vocab.insert("world".to_string(), 101);
        vocab.insert("the".to_string(), 102);
        vocab.insert("##ing".to_string(), 103);
        vocab.insert("##er".to_string(), 104);
        vocab.insert("##s".to_string(), 105);
        vocab.insert("run".to_string(), 106);
        vocab.insert("walk".to_string(), 107);
        vocab.insert("test".to_string(), 108);
        vocab.insert("un".to_string(), 109);
        vocab.insert("##known".to_string(), 110);
        vocab.insert("##ly".to_string(), 111);
        vocab.insert("##ning".to_string(), 112);
        WordPieceEncoder::new(vocab, "[UNK]")
    }

    /// Build a small SentencePiece vocabulary for testing.
    fn test_sentencepiece() -> SentencePieceEncoder {
        let entries = vec![
            ("<unk>".into(), 0, -100.0),
            ("\u{2581}".into(), 1, -1.0),
            ("\u{2581}hello".into(), 2, -2.0),
            ("\u{2581}world".into(), 3, -2.0),
            ("\u{2581}the".into(), 4, -1.5),
            ("h".into(), 5, -3.0),
            ("e".into(), 6, -3.0),
            ("l".into(), 7, -3.0),
            ("o".into(), 8, -3.0),
            ("w".into(), 9, -3.0),
            ("r".into(), 10, -3.0),
            ("d".into(), 11, -3.0),
            ("t".into(), 12, -3.0),
            ("\u{2581}h".into(), 13, -2.5),
            ("\u{2581}t".into(), 14, -2.5),
        ];
        SentencePieceEncoder::new(entries, "<unk>")
    }

    // ── PipelineConfig ──────────────────────────────────────────────────

    #[test]
    fn config_default_values() {
        let c = PipelineConfig::default();
        assert_eq!(c.max_length, 512);
        assert!(c.truncation);
        assert!(c.padding);
        assert_eq!(c.pad_token_id, 0);
        assert_eq!(c.unk_token, "[UNK]");
    }

    #[test]
    fn config_bert_default() {
        let c = PipelineConfig::bert_default();
        assert_eq!(c.cls_token.as_deref(), Some("[CLS]"));
        assert_eq!(c.sep_token.as_deref(), Some("[SEP]"));
        assert_eq!(c.max_length, 512);
    }

    #[test]
    fn config_gpt_default() {
        let c = PipelineConfig::gpt_default();
        assert_eq!(c.bos_token.as_deref(), Some("<s>"));
        assert_eq!(c.eos_token.as_deref(), Some("</s>"));
        assert_eq!(c.max_length, 1024);
        assert!(!c.padding);
    }

    #[test]
    fn config_clone_and_debug() {
        let c = PipelineConfig::default();
        let c2 = c.clone();
        assert_eq!(c2.max_length, c.max_length);
        let _ = format!("{c:?}");
    }

    // ── PreTokenizer ────────────────────────────────────────────────────

    #[test]
    fn pre_tokenize_whitespace() {
        let pt = PreTokenizer::new(PreTokenizeStrategy::Whitespace);
        assert_eq!(pt.pre_tokenize("hello world"), vec!["hello", "world"]);
    }

    #[test]
    fn pre_tokenize_whitespace_multiple_spaces() {
        let pt = PreTokenizer::new(PreTokenizeStrategy::Whitespace);
        assert_eq!(pt.pre_tokenize("hello   world  test"), vec!["hello", "world", "test"]);
    }

    #[test]
    fn pre_tokenize_whitespace_empty() {
        let pt = PreTokenizer::new(PreTokenizeStrategy::Whitespace);
        assert!(pt.pre_tokenize("").is_empty());
    }

    #[test]
    fn pre_tokenize_whitespace_only_spaces() {
        let pt = PreTokenizer::new(PreTokenizeStrategy::Whitespace);
        assert!(pt.pre_tokenize("   ").is_empty());
    }

    #[test]
    fn pre_tokenize_byte_level() {
        let pt = PreTokenizer::new(PreTokenizeStrategy::ByteLevel);
        let result = pt.pre_tokenize("a b");
        assert_eq!(result, vec!["a", "\u{0120}", "b"]);
    }

    #[test]
    fn pre_tokenize_byte_level_empty() {
        let pt = PreTokenizer::new(PreTokenizeStrategy::ByteLevel);
        assert!(pt.pre_tokenize("").is_empty());
    }

    #[test]
    fn pre_tokenize_punctuation() {
        let pt = PreTokenizer::new(PreTokenizeStrategy::Punctuation);
        let result = pt.pre_tokenize("hello, world!");
        assert_eq!(result, vec!["hello", ",", "world", "!"]);
    }

    #[test]
    fn pre_tokenize_punctuation_no_punct() {
        let pt = PreTokenizer::new(PreTokenizeStrategy::Punctuation);
        assert_eq!(pt.pre_tokenize("hello world"), vec!["hello", "world"]);
    }

    #[test]
    fn pre_tokenize_punctuation_only_punct() {
        let pt = PreTokenizer::new(PreTokenizeStrategy::Punctuation);
        assert_eq!(pt.pre_tokenize("!@#"), vec!["!", "@", "#"]);
    }

    #[test]
    fn pre_tokenize_clone_debug() {
        let pt = PreTokenizer::new(PreTokenizeStrategy::Whitespace);
        let pt2 = pt.clone();
        let _ = format!("{pt2:?}");
    }

    // ── TokenNormalizer ─────────────────────────────────────────────────

    #[test]
    fn normalizer_lowercase() {
        let n = TokenNormalizer::lowercase();
        assert_eq!(n.normalize("Hello WORLD"), "hello world");
    }

    #[test]
    fn normalizer_strip_accents() {
        let n =
            TokenNormalizer::new(NormalizationFlags { strip_accents: true, ..Default::default() });
        assert_eq!(n.normalize("café"), "cafe");
    }

    #[test]
    fn normalizer_nfkc_curly_quotes() {
        let n = TokenNormalizer::new(NormalizationFlags { nfkc: true, ..Default::default() });
        assert_eq!(n.normalize("\u{201C}hello\u{201D}"), "\"hello\"");
    }

    #[test]
    fn normalizer_nfkc_em_dash() {
        let n = TokenNormalizer::new(NormalizationFlags { nfkc: true, ..Default::default() });
        assert_eq!(n.normalize("a\u{2014}b"), "a-b");
    }

    #[test]
    fn normalizer_combined_flags() {
        let n = TokenNormalizer::new(NormalizationFlags {
            nfkc: true,
            lowercase: true,
            strip_accents: true,
        });
        assert_eq!(n.normalize("\u{201C}Café\u{201D}"), "\"cafe\"");
    }

    #[test]
    fn normalizer_no_flags() {
        let n = TokenNormalizer::new(NormalizationFlags::default());
        assert_eq!(n.normalize("Hello"), "Hello");
    }

    #[test]
    fn normalizer_empty_string() {
        let n = TokenNormalizer::lowercase();
        assert_eq!(n.normalize(""), "");
    }

    #[test]
    fn normalizer_clone_debug() {
        let n = TokenNormalizer::lowercase();
        let n2 = n.clone();
        let _ = format!("{n2:?}");
    }

    // ── BPEEncoder ──────────────────────────────────────────────────────

    #[test]
    fn bpe_encode_hello() {
        let bpe = test_bpe();
        let ids = bpe.encode_word("hello");
        assert_eq!(ids, vec![105]); // "hello" is a single merged token
    }

    #[test]
    fn bpe_encode_single_char() {
        let bpe = test_bpe();
        let ids = bpe.encode_word("a");
        assert_eq!(ids, vec![0]); // 'a' = 0
    }

    #[test]
    fn bpe_encode_empty() {
        let bpe = test_bpe();
        assert!(bpe.encode_word("").is_empty());
    }

    #[test]
    fn bpe_encode_unknown() {
        let bpe = test_bpe();
        let ids = bpe.encode_word("!");
        assert_eq!(ids, vec![100]); // UNK
    }

    #[test]
    fn bpe_encode_multiple_words() {
        let bpe = test_bpe();
        let words = vec!["hello".into(), "ab".into()];
        let ids = bpe.encode(&words);
        assert!(!ids.is_empty());
    }

    #[test]
    fn bpe_decode_roundtrip() {
        let bpe = test_bpe();
        let ids = bpe.encode_word("hello");
        let decoded = bpe.decode(&ids);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn bpe_vocab_size() {
        let bpe = test_bpe();
        assert!(bpe.vocab_size() > 26);
    }

    #[test]
    fn bpe_decode_empty() {
        let bpe = test_bpe();
        assert_eq!(bpe.decode(&[]), "");
    }

    #[test]
    fn bpe_clone_debug() {
        let bpe = test_bpe();
        let bpe2 = bpe.clone();
        let _ = format!("{bpe2:?}");
    }

    #[test]
    fn bpe_partial_merge() {
        let bpe = test_bpe();
        // "he" should merge to id 101
        let ids = bpe.encode_word("he");
        assert_eq!(ids, vec![101]);
    }

    #[test]
    fn bpe_multi_merge_steps() {
        let bpe = test_bpe();
        // "hell" should merge h+e→he, l+l→ll, he+ll→hell
        let ids = bpe.encode_word("hell");
        assert_eq!(ids, vec![104]);
    }

    // ── WordPieceEncoder ────────────────────────────────────────────────

    #[test]
    fn wordpiece_known_word() {
        let wp = test_wordpiece();
        let ids = wp.encode_word("hello");
        assert_eq!(ids, vec![100]);
    }

    #[test]
    fn wordpiece_subword_split() {
        let wp = test_wordpiece();
        // "running" → "run" + "##ning"
        let ids = wp.encode_word("running");
        assert_eq!(ids, vec![106, 112]); // run=106, ##ning=112
    }

    #[test]
    fn wordpiece_unknown_char() {
        let wp = test_wordpiece();
        // "xyz" has no match → UNK for each char
        let ids = wp.encode_word("xyz");
        assert!(ids.iter().all(|&id| id == 0)); // all UNK
    }

    #[test]
    fn wordpiece_empty() {
        let wp = test_wordpiece();
        assert!(wp.encode_word("").is_empty());
    }

    #[test]
    fn wordpiece_multiple_words() {
        let wp = test_wordpiece();
        let words = vec!["hello".into(), "world".into()];
        let ids = wp.encode(&words);
        assert_eq!(ids, vec![100, 101]);
    }

    #[test]
    fn wordpiece_decode() {
        let wp = test_wordpiece();
        let ids = wp.encode(&vec!["hello".into(), "world".into()]);
        let decoded = wp.decode(&ids);
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn wordpiece_decode_subword() {
        let wp = test_wordpiece();
        // "run" + "##ning" → "running"
        let decoded = wp.decode(&[106, 112]);
        assert_eq!(decoded, "running");
    }

    #[test]
    fn wordpiece_decode_empty() {
        let wp = test_wordpiece();
        assert_eq!(wp.decode(&[]), "");
    }

    #[test]
    fn wordpiece_vocab_size() {
        let wp = test_wordpiece();
        assert_eq!(wp.vocab_size(), 17);
    }

    #[test]
    fn wordpiece_clone_debug() {
        let wp = test_wordpiece();
        let wp2 = wp.clone();
        let _ = format!("{wp2:?}");
    }

    #[test]
    fn wordpiece_long_word_unk() {
        let mut wp = test_wordpiece();
        wp.max_word_len = 5;
        let ids = wp.encode_word("toolongword");
        assert_eq!(ids, vec![0]); // UNK for exceeding max_word_len
    }

    #[test]
    fn wordpiece_continuation_prefix() {
        let wp = test_wordpiece();
        // "walkers" → "walk" + "##er" + "##s"
        let ids = wp.encode_word("walkers");
        assert_eq!(ids, vec![107, 104, 105]);
    }

    // ── SentencePieceEncoder ────────────────────────────────────────────

    #[test]
    fn sentencepiece_known_word() {
        let sp = test_sentencepiece();
        let ids = sp.encode("hello");
        assert!(ids.contains(&2)); // ▁hello = id 2
    }

    #[test]
    fn sentencepiece_empty() {
        let sp = test_sentencepiece();
        assert!(sp.encode("").is_empty());
    }

    #[test]
    fn sentencepiece_vocab_size() {
        let sp = test_sentencepiece();
        assert_eq!(sp.vocab_size(), 15);
    }

    #[test]
    fn sentencepiece_decode() {
        let sp = test_sentencepiece();
        let decoded = sp.decode(&[2, 3]);
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn sentencepiece_decode_empty() {
        let sp = test_sentencepiece();
        assert_eq!(sp.decode(&[]), "");
    }

    #[test]
    fn sentencepiece_clone_debug() {
        let sp = test_sentencepiece();
        let sp2 = sp.clone();
        let _ = format!("{sp2:?}");
    }

    #[test]
    fn sentencepiece_single_chars() {
        let sp = test_sentencepiece();
        // Characters present in vocab should encode to something.
        let ids = sp.encode("h");
        assert!(!ids.is_empty());
    }

    #[test]
    fn sentencepiece_word_prefix() {
        let sp = test_sentencepiece();
        assert_eq!(sp.word_prefix, "\u{2581}");
    }

    // ── PostProcessor ───────────────────────────────────────────────────

    #[test]
    fn post_process_none() {
        let pp = PostProcessor::new(PostProcessTemplate::None);
        assert_eq!(pp.process_single(&[1, 2, 3]), vec![1, 2, 3]);
    }

    #[test]
    fn post_process_bert_single() {
        let pp = PostProcessor::new(PostProcessTemplate::BertSingle { cls_id: 101, sep_id: 102 });
        assert_eq!(pp.process_single(&[1, 2, 3]), vec![101, 1, 2, 3, 102]);
    }

    #[test]
    fn post_process_bert_pair() {
        let pp = PostProcessor::new(PostProcessTemplate::BertPair { cls_id: 101, sep_id: 102 });
        let result = pp.process_pair(&[1, 2], &[3, 4]);
        assert_eq!(result, vec![101, 1, 2, 102, 3, 4, 102]);
    }

    #[test]
    fn post_process_gpt_single() {
        let pp = PostProcessor::new(PostProcessTemplate::GptSingle { bos_id: 1, eos_id: 2 });
        assert_eq!(pp.process_single(&[10, 20]), vec![1, 10, 20, 2]);
    }

    #[test]
    fn post_process_empty_input() {
        let pp = PostProcessor::new(PostProcessTemplate::BertSingle { cls_id: 101, sep_id: 102 });
        assert_eq!(pp.process_single(&[]), vec![101, 102]);
    }

    #[test]
    fn post_process_pair_non_pair_template() {
        let pp = PostProcessor::new(PostProcessTemplate::GptSingle { bos_id: 1, eos_id: 2 });
        // Falls back to single processing + concat.
        let result = pp.process_pair(&[10], &[20]);
        assert_eq!(result, vec![1, 10, 2, 20]);
    }

    #[test]
    fn post_process_clone_debug() {
        let pp = PostProcessor::new(PostProcessTemplate::None);
        let pp2 = pp.clone();
        let _ = format!("{pp2:?}");
    }

    // ── BatchTokenizer ──────────────────────────────────────────────────

    #[test]
    fn batch_no_padding() {
        let config = PipelineConfig { padding: false, truncation: false, ..Default::default() };
        let bt = BatchTokenizer::new(config);
        let encoded = vec![vec![1, 2, 3], vec![4, 5]];
        let out = bt.process_batch(&encoded);
        assert_eq!(out.input_ids, vec![vec![1, 2, 3], vec![4, 5]]);
    }

    #[test]
    fn batch_padding_to_longest() {
        let config = PipelineConfig { padding: true, truncation: false, ..Default::default() };
        let bt = BatchTokenizer::new(config);
        let encoded = vec![vec![1, 2, 3], vec![4, 5]];
        let out = bt.process_batch(&encoded);
        assert_eq!(out.input_ids, vec![vec![1, 2, 3], vec![4, 5, 0]]);
        assert_eq!(out.attention_masks, vec![vec![1, 1, 1], vec![1, 1, 0]]);
    }

    #[test]
    fn batch_truncation() {
        let config = PipelineConfig {
            padding: false,
            truncation: true,
            max_length: 2,
            ..Default::default()
        };
        let bt = BatchTokenizer::new(config);
        let encoded = vec![vec![1, 2, 3, 4]];
        let out = bt.process_batch(&encoded);
        assert_eq!(out.input_ids, vec![vec![1, 2]]);
    }

    #[test]
    fn batch_padding_and_truncation() {
        let config =
            PipelineConfig { padding: true, truncation: true, max_length: 3, ..Default::default() };
        let bt = BatchTokenizer::new(config);
        let encoded = vec![vec![1, 2, 3, 4, 5], vec![10]];
        let out = bt.process_batch(&encoded);
        assert_eq!(out.input_ids, vec![vec![1, 2, 3], vec![10, 0, 0]]);
        assert_eq!(out.attention_masks, vec![vec![1, 1, 1], vec![1, 0, 0]]);
    }

    #[test]
    fn batch_empty() {
        let bt = BatchTokenizer::new(PipelineConfig::default());
        let out = bt.process_batch(&[]);
        assert!(out.input_ids.is_empty());
    }

    #[test]
    fn batch_single_sequence() {
        let bt = BatchTokenizer::new(PipelineConfig::default());
        let out = bt.process_batch(&[vec![1, 2, 3]]);
        assert_eq!(out.input_ids.len(), 1);
        assert_eq!(out.original_lengths, vec![3]);
    }

    #[test]
    fn batch_original_lengths_preserved() {
        let config =
            PipelineConfig { padding: true, truncation: true, max_length: 5, ..Default::default() };
        let bt = BatchTokenizer::new(config);
        let encoded = vec![vec![1, 2, 3, 4, 5, 6, 7], vec![10, 20]];
        let out = bt.process_batch(&encoded);
        assert_eq!(out.original_lengths, vec![7, 2]);
    }

    #[test]
    fn batch_clone_debug() {
        let bt = BatchTokenizer::new(PipelineConfig::default());
        let bt2 = bt.clone();
        let _ = format!("{bt2:?}");
    }

    // ── TokenizerMetrics ────────────────────────────────────────────────

    #[test]
    fn metrics_initial_state() {
        let m = TokenizerMetrics::new();
        assert_eq!(m.sequences_processed, 0);
        assert_eq!(m.tokens_produced, 0);
        assert_eq!(m.characters_consumed, 0);
        assert_eq!(m.unique_tokens(), 0);
    }

    #[test]
    fn metrics_record() {
        let mut m = TokenizerMetrics::new();
        m.record(10, &[1, 2, 3], Duration::from_millis(5));
        assert_eq!(m.sequences_processed, 1);
        assert_eq!(m.tokens_produced, 3);
        assert_eq!(m.characters_consumed, 10);
        assert_eq!(m.unique_tokens(), 3);
    }

    #[test]
    fn metrics_multiple_records() {
        let mut m = TokenizerMetrics::new();
        m.record(5, &[1, 2], Duration::from_millis(1));
        m.record(10, &[2, 3, 4], Duration::from_millis(2));
        assert_eq!(m.sequences_processed, 2);
        assert_eq!(m.tokens_produced, 5);
        assert_eq!(m.characters_consumed, 15);
        assert_eq!(m.unique_tokens(), 4); // {1,2,3,4}
    }

    #[test]
    fn metrics_avg_tokens() {
        let mut m = TokenizerMetrics::new();
        m.record(5, &[1, 2], Duration::ZERO);
        m.record(5, &[3, 4, 5, 6], Duration::ZERO);
        assert!((m.avg_tokens_per_sequence() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn metrics_chars_per_token() {
        let mut m = TokenizerMetrics::new();
        m.record(12, &[1, 2, 3], Duration::ZERO);
        assert!((m.chars_per_token() - 4.0).abs() < 1e-9);
    }

    #[test]
    fn metrics_tokens_per_second() {
        let mut m = TokenizerMetrics::new();
        m.record(5, &[1, 2, 3, 4, 5], Duration::from_secs(1));
        assert!((m.tokens_per_second() - 5.0).abs() < 1e-3);
    }

    #[test]
    fn metrics_empty_denominators() {
        let m = TokenizerMetrics::new();
        assert_eq!(m.tokens_per_second(), 0.0);
        assert_eq!(m.avg_tokens_per_sequence(), 0.0);
        assert_eq!(m.chars_per_token(), 0.0);
    }

    #[test]
    fn metrics_reset() {
        let mut m = TokenizerMetrics::new();
        m.record(10, &[1, 2], Duration::from_millis(5));
        m.reset();
        assert_eq!(m.sequences_processed, 0);
        assert_eq!(m.tokens_produced, 0);
    }

    #[test]
    fn metrics_default() {
        let m = TokenizerMetrics::default();
        assert_eq!(m.sequences_processed, 0);
    }

    #[test]
    fn metrics_clone_debug() {
        let m = TokenizerMetrics::new();
        let m2 = m.clone();
        let _ = format!("{m2:?}");
    }

    #[test]
    fn metrics_histogram_counts() {
        let mut m = TokenizerMetrics::new();
        m.record(5, &[1, 1, 2], Duration::ZERO);
        assert_eq!(m.token_histogram[&1], 2);
        assert_eq!(m.token_histogram[&2], 1);
    }

    // ── TokenizerPipelineEngine ─────────────────────────────────────────

    fn make_bpe_pipeline() -> TokenizerPipelineEngine {
        let config = PipelineConfig {
            padding: true,
            truncation: true,
            max_length: 64,
            ..Default::default()
        };
        let normalizer = Some(TokenNormalizer::lowercase());
        let pre_tok = PreTokenizer::new(PreTokenizeStrategy::Whitespace);
        let encoder = EncoderKind::Bpe(test_bpe());
        let post = PostProcessor::new(PostProcessTemplate::None);
        TokenizerPipelineEngine::new(config, normalizer, pre_tok, encoder, post)
    }

    fn make_wordpiece_pipeline() -> TokenizerPipelineEngine {
        let config = PipelineConfig::bert_default();
        let normalizer = Some(TokenNormalizer::lowercase());
        let pre_tok = PreTokenizer::new(PreTokenizeStrategy::Whitespace);
        let encoder = EncoderKind::WordPiece(test_wordpiece());
        let post = PostProcessor::new(PostProcessTemplate::BertSingle { cls_id: 1, sep_id: 2 });
        TokenizerPipelineEngine::new(config, normalizer, pre_tok, encoder, post)
    }

    fn make_sentencepiece_pipeline() -> TokenizerPipelineEngine {
        let config = PipelineConfig::gpt_default();
        let normalizer = None;
        let pre_tok = PreTokenizer::new(PreTokenizeStrategy::Whitespace);
        let encoder = EncoderKind::SentencePiece(test_sentencepiece());
        let post = PostProcessor::new(PostProcessTemplate::GptSingle { bos_id: 50, eos_id: 51 });
        TokenizerPipelineEngine::new(config, normalizer, pre_tok, encoder, post)
    }

    #[test]
    fn pipeline_bpe_encode() {
        let mut pipeline = make_bpe_pipeline();
        let ids = pipeline.encode("hello");
        assert!(!ids.is_empty());
    }

    #[test]
    fn pipeline_bpe_encode_multiple_words() {
        let mut pipeline = make_bpe_pipeline();
        let ids = pipeline.encode("hello ab");
        assert!(ids.len() >= 2);
    }

    #[test]
    fn pipeline_bpe_decode() {
        let mut pipeline = make_bpe_pipeline();
        let ids = pipeline.encode("hello");
        let decoded = pipeline.decode(&ids);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn pipeline_wordpiece_encode() {
        let mut pipeline = make_wordpiece_pipeline();
        let ids = pipeline.encode("hello world");
        // Should have CLS + hello + world + SEP
        assert!(ids.contains(&1)); // CLS
        assert!(ids.contains(&100)); // hello
        assert!(ids.contains(&101)); // world
        assert!(ids.contains(&2)); // SEP
    }

    #[test]
    fn pipeline_wordpiece_subword() {
        let mut pipeline = make_wordpiece_pipeline();
        let ids = pipeline.encode("running");
        // CLS + run + ##ning + SEP
        assert_eq!(ids[0], 1); // CLS
        assert_eq!(ids[1], 106); // run
        assert_eq!(ids[2], 112); // ##ning
        assert_eq!(ids[3], 2); // SEP
    }

    #[test]
    fn pipeline_sentencepiece_encode() {
        let mut pipeline = make_sentencepiece_pipeline();
        let ids = pipeline.encode("hello");
        assert!(!ids.is_empty());
        // Should start with BOS=50 and end with EOS=51.
        assert_eq!(*ids.first().unwrap(), 50);
        assert_eq!(*ids.last().unwrap(), 51);
    }

    #[test]
    fn pipeline_sentencepiece_decode() {
        let pipeline = make_sentencepiece_pipeline();
        let decoded = pipeline.decode(&[2, 3]);
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn pipeline_metrics_updated() {
        let mut pipeline = make_bpe_pipeline();
        pipeline.encode("hello");
        pipeline.encode("world");
        let m = pipeline.metrics();
        assert_eq!(m.sequences_processed, 2);
        assert!(m.tokens_produced > 0);
    }

    #[test]
    fn pipeline_metrics_reset() {
        let mut pipeline = make_bpe_pipeline();
        pipeline.encode("hello");
        pipeline.reset_metrics();
        assert_eq!(pipeline.metrics().sequences_processed, 0);
    }

    #[test]
    fn pipeline_batch_encode() {
        let mut pipeline = make_bpe_pipeline();
        let out = pipeline.encode_batch(&["hello", "ab"]);
        assert_eq!(out.input_ids.len(), 2);
        assert_eq!(out.original_lengths.len(), 2);
    }

    #[test]
    fn pipeline_batch_padding() {
        let mut pipeline = make_bpe_pipeline();
        let out = pipeline.encode_batch(&["hello", "a"]);
        // Padded to same length.
        let len0 = out.input_ids[0].len();
        let len1 = out.input_ids[1].len();
        assert_eq!(len0, len1);
    }

    #[test]
    fn pipeline_batch_empty() {
        let mut pipeline = make_bpe_pipeline();
        let out = pipeline.encode_batch(&[]);
        assert!(out.input_ids.is_empty());
    }

    #[test]
    fn pipeline_normalizer_applied() {
        let mut pipeline = make_bpe_pipeline();
        let ids_upper = pipeline.encode("HELLO");
        let mut pipeline2 = make_bpe_pipeline();
        let ids_lower = pipeline2.encode("hello");
        // Both should produce the same IDs because normalizer lowercases.
        assert_eq!(ids_upper, ids_lower);
    }

    #[test]
    fn pipeline_clone_debug() {
        let pipeline = make_bpe_pipeline();
        let p2 = pipeline.clone();
        let _ = format!("{p2:?}");
    }

    // ── Edge cases ──────────────────────────────────────────────────────

    #[test]
    fn edge_unicode_input() {
        let mut pipeline = make_bpe_pipeline();
        // Should not panic on unicode input.
        let ids = pipeline.encode("日本語テスト");
        assert!(!ids.is_empty());
    }

    #[test]
    fn edge_very_long_input() {
        let mut pipeline = make_bpe_pipeline();
        let long_text = "hello ".repeat(1000);
        let ids = pipeline.encode(&long_text);
        assert!(!ids.is_empty());
    }

    #[test]
    fn edge_only_whitespace() {
        let mut pipeline = make_bpe_pipeline();
        let ids = pipeline.encode("   ");
        // After whitespace pre-tokenization, no segments → empty encoding.
        assert!(ids.is_empty());
    }

    #[test]
    fn edge_special_chars() {
        let mut pipeline = make_bpe_pipeline();
        let ids = pipeline.encode("!@#$%^&*()");
        // Unknown chars map to UNK.
        assert!(!ids.is_empty());
    }

    #[test]
    fn edge_newlines_and_tabs() {
        let mut pipeline = make_bpe_pipeline();
        let ids = pipeline.encode("hello\n\tworld");
        assert!(!ids.is_empty());
    }

    #[test]
    fn edge_single_char() {
        let mut pipeline = make_bpe_pipeline();
        let ids = pipeline.encode("a");
        assert!(!ids.is_empty());
    }

    #[test]
    fn edge_batch_all_empty() {
        let mut pipeline = make_bpe_pipeline();
        let out = pipeline.encode_batch(&["", "", ""]);
        assert_eq!(out.input_ids.len(), 3);
    }

    #[test]
    fn edge_batch_mixed_lengths() {
        let mut pipeline = make_bpe_pipeline();
        let out = pipeline.encode_batch(&["hello", "a", "hello hello hello"]);
        assert_eq!(out.input_ids.len(), 3);
    }

    #[test]
    fn edge_truncation_exact_boundary() {
        let config = PipelineConfig {
            max_length: 3,
            truncation: true,
            padding: false,
            ..Default::default()
        };
        let bt = BatchTokenizer::new(config);
        let out = bt.process_batch(&[vec![1, 2, 3]]);
        assert_eq!(out.input_ids[0].len(), 3);
    }

    #[test]
    fn edge_pad_token_id_custom() {
        let config = PipelineConfig {
            padding: true,
            pad_token_id: 999,
            max_length: 5,
            ..Default::default()
        };
        let bt = BatchTokenizer::new(config);
        let out = bt.process_batch(&[vec![1], vec![1, 2]]);
        assert!(out.input_ids[0].contains(&999));
    }

    // ── Roundtrip tests ─────────────────────────────────────────────────

    #[test]
    fn roundtrip_bpe_simple() {
        let bpe = test_bpe();
        let ids = bpe.encode_word("hello");
        let text = bpe.decode(&ids);
        assert_eq!(text, "hello");
    }

    #[test]
    fn roundtrip_wordpiece_simple() {
        let wp = test_wordpiece();
        let words = vec!["hello".into(), "world".into()];
        let ids = wp.encode(&words);
        let text = wp.decode(&ids);
        assert_eq!(text, "hello world");
    }

    #[test]
    fn roundtrip_sentencepiece_simple() {
        let sp = test_sentencepiece();
        let ids = sp.encode("hello world");
        let text = sp.decode(&ids);
        assert_eq!(text, "hello world");
    }

    #[test]
    fn roundtrip_bpe_two_words() {
        let bpe = test_bpe();
        let ids_a = bpe.encode_word("he");
        let ids_b = bpe.encode_word("ab");
        let all: Vec<u32> = ids_a.iter().chain(ids_b.iter()).copied().collect();
        let text = bpe.decode(&all);
        assert_eq!(text, "heab");
    }

    #[test]
    fn roundtrip_wordpiece_subword() {
        let wp = test_wordpiece();
        let ids = wp.encode_word("running");
        let text = wp.decode(&ids);
        assert_eq!(text, "running");
    }

    // ── Encoder kind dispatch ───────────────────────────────────────────

    #[test]
    fn encoder_kind_bpe_debug() {
        let ek = EncoderKind::Bpe(test_bpe());
        let _ = format!("{ek:?}");
    }

    #[test]
    fn encoder_kind_wordpiece_debug() {
        let ek = EncoderKind::WordPiece(test_wordpiece());
        let _ = format!("{ek:?}");
    }

    #[test]
    fn encoder_kind_sentencepiece_debug() {
        let ek = EncoderKind::SentencePiece(test_sentencepiece());
        let _ = format!("{ek:?}");
    }

    // ── Additional coverage ─────────────────────────────────────────────

    #[test]
    fn pre_tokenize_strategy_eq() {
        assert_eq!(PreTokenizeStrategy::Whitespace, PreTokenizeStrategy::Whitespace);
        assert_ne!(PreTokenizeStrategy::Whitespace, PreTokenizeStrategy::ByteLevel);
    }

    #[test]
    fn normalization_flags_default() {
        let f = NormalizationFlags::default();
        assert!(!f.nfkc);
        assert!(!f.lowercase);
        assert!(!f.strip_accents);
    }

    #[test]
    fn post_process_template_eq() {
        assert_eq!(PostProcessTemplate::None, PostProcessTemplate::None);
    }

    #[test]
    fn batch_output_eq() {
        let a = BatchOutput {
            input_ids: vec![vec![1]],
            attention_masks: vec![vec![1]],
            original_lengths: vec![1],
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn nfkc_non_breaking_space() {
        let n = TokenNormalizer::new(NormalizationFlags { nfkc: true, ..Default::default() });
        assert_eq!(n.normalize("a\u{00A0}b"), "a b");
    }

    #[test]
    fn strip_accents_noop_on_ascii() {
        let n =
            TokenNormalizer::new(NormalizationFlags { strip_accents: true, ..Default::default() });
        assert_eq!(n.normalize("hello"), "hello");
    }

    #[test]
    fn nfkc_ellipsis() {
        let n = TokenNormalizer::new(NormalizationFlags { nfkc: true, ..Default::default() });
        assert_eq!(n.normalize("wait\u{2026}"), "wait.");
    }

    #[test]
    fn bpe_decode_unknown_ids() {
        let bpe = test_bpe();
        // IDs that don't exist produce empty decode.
        let decoded = bpe.decode(&[9999]);
        assert_eq!(decoded, "");
    }

    #[test]
    fn wordpiece_decode_unknown_ids() {
        let wp = test_wordpiece();
        let decoded = wp.decode(&[9999]);
        assert_eq!(decoded, "");
    }

    #[test]
    fn batch_attention_mask_correct() {
        let config = PipelineConfig {
            padding: true,
            truncation: false,
            max_length: 512,
            ..Default::default()
        };
        let bt = BatchTokenizer::new(config);
        let out = bt.process_batch(&[vec![1, 2], vec![3]]);
        assert_eq!(out.attention_masks[0], vec![1, 1]);
        assert_eq!(out.attention_masks[1], vec![1, 0]);
    }

    #[test]
    fn pipeline_encode_empty_string() {
        let mut pipeline = make_bpe_pipeline();
        let ids = pipeline.encode("");
        assert!(ids.is_empty());
    }

    #[test]
    fn pipeline_batch_single_item() {
        let mut pipeline = make_wordpiece_pipeline();
        let out = pipeline.encode_batch(&["test"]);
        assert_eq!(out.input_ids.len(), 1);
    }

    #[test]
    fn metrics_encoding_duration() {
        let mut m = TokenizerMetrics::new();
        m.record(5, &[1], Duration::from_millis(100));
        m.record(5, &[2], Duration::from_millis(200));
        assert_eq!(m.encoding_duration, Duration::from_millis(300));
    }
}
