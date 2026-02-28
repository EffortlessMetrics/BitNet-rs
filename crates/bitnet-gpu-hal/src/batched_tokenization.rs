//! Batched and parallel tokenization system for GPU inference pipelines.
//!
//! Provides configurable batch encoding/decoding with padding strategies,
//! truncation policies, streaming decode, and throughput metrics.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ── Configuration ────────────────────────────────────────────────────────────

/// How to pad sequences within a batch to uniform length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingStrategy {
    /// No padding — sequences retain their original lengths.
    NoPadding,
    /// Pad every sequence to `max_seq_length`.
    PadToMax,
    /// Pad every sequence to the longest sequence in the batch.
    PadToLongest,
    /// Pad every sequence to the next multiple of the given value.
    PadToMultipleOf(usize),
}

/// How to truncate sequences that exceed `max_seq_length`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationStrategy {
    /// No truncation — sequences keep their full length.
    None,
    /// Remove tokens from the beginning.
    TruncateLeft,
    /// Remove tokens from the end.
    TruncateRight,
    /// Remove tokens from the middle, keeping start and end.
    TruncateMiddle,
}

/// Configuration for the batch tokenizer.
#[derive(Debug, Clone)]
pub struct BatchTokenizerConfig {
    /// Maximum number of sequences per batch.
    pub max_batch_size: usize,
    /// Maximum sequence length (in tokens) before truncation/padding.
    pub max_seq_length: usize,
    /// Number of threads for parallel encoding/decoding.
    pub num_threads: usize,
    /// Strategy for padding sequences to uniform length.
    pub padding_strategy: PaddingStrategy,
    /// Strategy for truncating over-length sequences.
    pub truncation_strategy: TruncationStrategy,
    /// Token ID used for padding (typically 0).
    pub padding_token_id: u32,
}

impl Default for BatchTokenizerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_seq_length: 512,
            num_threads: 4,
            padding_strategy: PaddingStrategy::PadToLongest,
            truncation_strategy: TruncationStrategy::TruncateRight,
            padding_token_id: 0,
        }
    }
}

// ── Batch output ─────────────────────────────────────────────────────────────

/// Result of encoding a batch of strings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizedBatch {
    /// Token IDs for each sequence in the batch.
    pub input_ids: Vec<Vec<u32>>,
    /// Attention masks: 1 for real tokens, 0 for padding.
    pub attention_masks: Vec<Vec<u8>>,
    /// Original (pre-padding) length of each sequence.
    pub lengths: Vec<usize>,
    /// The token ID used for padding.
    pub padding_token_id: u32,
}

impl TokenizedBatch {
    /// Number of sequences in this batch.
    #[must_use]
    pub const fn batch_size(&self) -> usize {
        self.input_ids.len()
    }

    /// Whether all padded sequences have the same length.
    #[must_use]
    pub fn is_uniform_length(&self) -> bool {
        if self.input_ids.is_empty() {
            return true;
        }
        let first_len = self.input_ids[0].len();
        self.input_ids.iter().all(|seq| seq.len() == first_len)
    }

    /// The padded sequence length (assumes uniform; returns 0 for empty batch).
    #[must_use]
    pub fn padded_length(&self) -> usize {
        self.input_ids.first().map_or(0, Vec::len)
    }

    /// Total number of real (non-padding) tokens across all sequences.
    #[must_use]
    pub fn total_real_tokens(&self) -> usize {
        self.lengths.iter().sum()
    }

    /// Batch utilization ratio: real tokens / total slots.
    #[must_use]
    pub fn utilization(&self) -> f64 {
        let total_slots = self.input_ids.iter().map(Vec::len).sum::<usize>();
        if total_slots == 0 {
            return 0.0;
        }
        self.total_real_tokens() as f64 / total_slots as f64
    }
}

// ── Tokenizer abstraction ────────────────────────────────────────────────────

/// Trait abstracting a tokenizer that can encode text to IDs and decode back.
pub trait TokenizerHandle: Send + Sync {
    /// Size of the vocabulary.
    fn vocab_size(&self) -> usize;

    /// Encode a string into token IDs.
    fn encode(&self, text: &str) -> Vec<u32>;

    /// Decode a sequence of token IDs back to a string.
    fn decode(&self, ids: &[u32]) -> String;

    /// Return special token IDs (e.g. BOS, EOS, PAD).
    fn special_tokens(&self) -> &HashMap<String, u32>;
}

/// Simple mock tokenizer for testing: splits on whitespace, maps words to IDs.
#[derive(Debug, Clone)]
pub struct MockTokenizer {
    word_to_id: HashMap<String, u32>,
    id_to_word: HashMap<u32, String>,
    specials: HashMap<String, u32>,
    next_id: u32,
}

impl MockTokenizer {
    /// Create a new mock tokenizer with the given vocabulary words.
    #[must_use]
    pub fn new(words: &[&str]) -> Self {
        let mut word_to_id = HashMap::new();
        let mut id_to_word = HashMap::new();
        // Reserve 0 for PAD, 1 for UNK
        id_to_word.insert(0, "<pad>".to_string());
        id_to_word.insert(1, "<unk>".to_string());
        word_to_id.insert("<pad>".to_string(), 0);
        word_to_id.insert("<unk>".to_string(), 1);

        let mut next_id = 2u32;
        for &w in words {
            if !word_to_id.contains_key(w) {
                word_to_id.insert(w.to_string(), next_id);
                id_to_word.insert(next_id, w.to_string());
                next_id += 1;
            }
        }

        let mut specials = HashMap::new();
        specials.insert("PAD".to_string(), 0);
        specials.insert("UNK".to_string(), 1);

        Self {
            word_to_id,
            id_to_word,
            specials,
            next_id,
        }
    }

    /// Number of distinct tokens (including specials).
    #[must_use]
    pub fn len(&self) -> usize {
        self.word_to_id.len()
    }

    /// Whether the vocabulary is empty (never true — specials always present).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.word_to_id.is_empty()
    }
}

impl TokenizerHandle for MockTokenizer {
    fn vocab_size(&self) -> usize {
        self.next_id as usize
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        text.split_whitespace()
            .map(|w| *self.word_to_id.get(w).unwrap_or(&1))
            .collect()
    }

    fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|id| self.id_to_word.get(id))
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn special_tokens(&self) -> &HashMap<String, u32> {
        &self.specials
    }
}

// ── Truncation helpers ───────────────────────────────────────────────────────

/// Apply truncation strategy to a token sequence.
fn truncate(tokens: &mut Vec<u32>, max_len: usize, strategy: TruncationStrategy) {
    if tokens.len() <= max_len {
        return;
    }
    match strategy {
        TruncationStrategy::None => {}
        TruncationStrategy::TruncateRight => {
            tokens.truncate(max_len);
        }
        TruncationStrategy::TruncateLeft => {
            let start = tokens.len() - max_len;
            *tokens = tokens[start..].to_vec();
        }
        TruncationStrategy::TruncateMiddle => {
            if max_len == 0 {
                tokens.clear();
                return;
            }
            let keep_start = max_len.div_ceil(2);
            let keep_end = max_len / 2;
            let end_start = tokens.len() - keep_end;
            let mut result = tokens[..keep_start].to_vec();
            result.extend_from_slice(&tokens[end_start..]);
            *tokens = result;
        }
    }
}

// ── Padding helpers ──────────────────────────────────────────────────────────

/// Compute the target padded length for a batch given the strategy and config.
fn compute_pad_length(
    lengths: &[usize],
    strategy: PaddingStrategy,
    max_seq_length: usize,
) -> Option<usize> {
    match strategy {
        PaddingStrategy::NoPadding => None,
        PaddingStrategy::PadToMax => Some(max_seq_length),
        PaddingStrategy::PadToLongest => {
            let max_len = lengths.iter().copied().max().unwrap_or(0);
            Some(max_len)
        }
        PaddingStrategy::PadToMultipleOf(multiple) => {
            if multiple == 0 {
                return None;
            }
            let max_len = lengths.iter().copied().max().unwrap_or(0);
            let padded = max_len.div_ceil(multiple) * multiple;
            Some(padded)
        }
    }
}

/// Pad a sequence in-place and return its attention mask.
fn pad_sequence(tokens: &mut Vec<u32>, target_len: usize, pad_id: u32) -> Vec<u8> {
    let real_len = tokens.len();
    let mut mask = vec![1u8; real_len];
    if real_len < target_len {
        tokens.resize(target_len, pad_id);
        mask.resize(target_len, 0);
    }
    mask
}

// ── Batch encoder ────────────────────────────────────────────────────────────

/// Encodes multiple strings into a `TokenizedBatch` using parallel workers.
pub struct BatchEncoder<T: TokenizerHandle> {
    tokenizer: Arc<T>,
    config: BatchTokenizerConfig,
}

impl<T: TokenizerHandle + 'static> BatchEncoder<T> {
    /// Create a new batch encoder.
    #[must_use]
    pub const fn new(tokenizer: Arc<T>, config: BatchTokenizerConfig) -> Self {
        Self { tokenizer, config }
    }

    /// Encode a slice of strings into a `TokenizedBatch`.
    ///
    /// If `texts` exceeds `max_batch_size`, only the first `max_batch_size`
    /// items are encoded.
    pub fn encode(&self, texts: &[&str]) -> TokenizedBatch {
        let limit = texts.len().min(self.config.max_batch_size);
        let texts = &texts[..limit];

        // Encode all strings (parallel via std threads when num_threads > 1)
        let mut encoded: Vec<Vec<u32>> = if self.config.num_threads <= 1 || texts.len() <= 1 {
            texts.iter().map(|t| self.tokenizer.encode(t)).collect()
        } else {
            self.encode_parallel(texts)
        };

        // Truncate
        for seq in &mut encoded {
            truncate(seq, self.config.max_seq_length, self.config.truncation_strategy);
        }

        let lengths: Vec<usize> = encoded.iter().map(Vec::len).collect();

        // Pad
        let attention_masks = if let Some(target_len) = compute_pad_length(
            &lengths,
            self.config.padding_strategy,
            self.config.max_seq_length,
        ) {
            encoded
                .iter_mut()
                .map(|seq| pad_sequence(seq, target_len, self.config.padding_token_id))
                .collect()
        } else {
            encoded.iter().map(|seq| vec![1u8; seq.len()]).collect()
        };

        TokenizedBatch {
            input_ids: encoded,
            attention_masks,
            lengths,
            padding_token_id: self.config.padding_token_id,
        }
    }

    /// Parallel encoding using std threads.
    fn encode_parallel(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        let chunk_size = texts.len().div_ceil(self.config.num_threads);
        let owned_texts: Vec<String> =
            texts.iter().map(|s| (*s).to_string()).collect();

        let handles: Vec<_> = owned_texts
            .chunks(chunk_size)
            .map(|chunk| {
                let chunk_owned: Vec<String> = chunk.to_vec();
                let tok = Arc::clone(&self.tokenizer);
                std::thread::spawn(move || {
                    chunk_owned.iter().map(|s| tok.encode(s)).collect::<Vec<_>>()
                })
            })
            .collect();

        let mut results = Vec::with_capacity(texts.len());
        for h in handles {
            results.extend(h.join().expect("encoder thread panicked"));
        }
        results
    }
}

// ── Batch decoder ────────────────────────────────────────────────────────────

/// Decodes multiple token sequences back to strings using parallel workers.
pub struct BatchDecoder<T: TokenizerHandle> {
    tokenizer: Arc<T>,
    config: BatchTokenizerConfig,
}

impl<T: TokenizerHandle + 'static> BatchDecoder<T> {
    /// Create a new batch decoder.
    #[must_use]
    pub const fn new(tokenizer: Arc<T>, config: BatchTokenizerConfig) -> Self {
        Self { tokenizer, config }
    }

    /// Decode token ID sequences back to strings.
    ///
    /// If `sequences` exceeds `max_batch_size`, only the first `max_batch_size`
    /// items are decoded.
    pub fn decode(&self, sequences: &[Vec<u32>]) -> Vec<String> {
        let limit = sequences.len().min(self.config.max_batch_size);
        let sequences = &sequences[..limit];

        if self.config.num_threads <= 1 || sequences.len() <= 1 {
            sequences.iter().map(|s| self.tokenizer.decode(s)).collect()
        } else {
            self.decode_parallel(sequences)
        }
    }

    /// Parallel decoding using std threads.
    fn decode_parallel(&self, sequences: &[Vec<u32>]) -> Vec<String> {
        let chunk_size = sequences.len().div_ceil(self.config.num_threads);

        let handles: Vec<_> = sequences
            .chunks(chunk_size)
            .map(|chunk| {
                let chunk_owned: Vec<Vec<u32>> = chunk.to_vec();
                let tok = Arc::clone(&self.tokenizer);
                std::thread::spawn(move || {
                    chunk_owned
                        .iter()
                        .map(|s| tok.decode(s))
                        .collect::<Vec<_>>()
                })
            })
            .collect();

        let mut results = Vec::with_capacity(sequences.len());
        for h in handles {
            results.extend(h.join().expect("decoder thread panicked"));
        }
        results
    }
}

// ── Streaming tokenizer ──────────────────────────────────────────────────────

/// Processes tokens one at a time with incremental decoding.
///
/// Useful for streaming inference where tokens arrive one-by-one and must be
/// decoded incrementally without re-decoding the entire sequence each time.
pub struct StreamingTokenizer<T: TokenizerHandle> {
    tokenizer: Arc<T>,
    /// Accumulated token IDs so far.
    token_buffer: Vec<u32>,
    /// Previously decoded prefix length (in chars) to avoid re-emitting.
    prev_decoded_len: usize,
}

impl<T: TokenizerHandle> StreamingTokenizer<T> {
    /// Create a new streaming tokenizer.
    #[must_use]
    pub const fn new(tokenizer: Arc<T>) -> Self {
        Self {
            tokenizer,
            token_buffer: Vec::new(),
            prev_decoded_len: 0,
        }
    }

    /// Feed a single token and return the new text fragment produced.
    pub fn feed(&mut self, token_id: u32) -> String {
        self.token_buffer.push(token_id);
        let full = self.tokenizer.decode(&self.token_buffer);
        let new_chars: String = full.chars().skip(self.prev_decoded_len).collect();
        self.prev_decoded_len = full.chars().count();
        new_chars
    }

    /// Feed multiple tokens at once, returning the combined new text.
    pub fn feed_many(&mut self, token_ids: &[u32]) -> String {
        let mut combined = String::new();
        for &id in token_ids {
            combined.push_str(&self.feed(id));
        }
        combined
    }

    /// Return the full decoded text so far.
    #[must_use]
    pub fn text(&self) -> String {
        self.tokenizer.decode(&self.token_buffer)
    }

    /// Return the accumulated token buffer.
    #[must_use]
    pub fn tokens(&self) -> &[u32] {
        &self.token_buffer
    }

    /// Reset the streaming state.
    pub fn reset(&mut self) {
        self.token_buffer.clear();
        self.prev_decoded_len = 0;
    }

    /// Number of tokens consumed so far.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.token_buffer.len()
    }

    /// Whether no tokens have been consumed yet.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.token_buffer.is_empty()
    }
}

// ── Metrics ──────────────────────────────────────────────────────────────────

/// Tracks tokenization throughput and latency metrics.
#[derive(Debug)]
pub struct TokenizationMetrics {
    total_strings: AtomicU64,
    total_tokens: AtomicU64,
    total_encode_ns: AtomicU64,
    total_decode_ns: AtomicU64,
    batch_count: AtomicU64,
}

impl Default for TokenizationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenizationMetrics {
    /// Create a new metrics tracker.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            total_strings: AtomicU64::new(0),
            total_tokens: AtomicU64::new(0),
            total_encode_ns: AtomicU64::new(0),
            total_decode_ns: AtomicU64::new(0),
            batch_count: AtomicU64::new(0),
        }
    }

    /// Record an encoding operation.
    pub fn record_encode(&self, num_strings: u64, num_tokens: u64, duration: Duration) {
        self.total_strings.fetch_add(num_strings, Ordering::Relaxed);
        self.total_tokens.fetch_add(num_tokens, Ordering::Relaxed);
        self.total_encode_ns
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
        self.batch_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a decoding operation.
    pub fn record_decode(&self, num_tokens: u64, duration: Duration) {
        self.total_tokens.fetch_add(num_tokens, Ordering::Relaxed);
        self.total_decode_ns
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
    }

    /// Strings encoded per second (0.0 if no time recorded).
    #[must_use]
    pub fn strings_per_sec(&self) -> f64 {
        let ns = self.total_encode_ns.load(Ordering::Relaxed);
        if ns == 0 {
            return 0.0;
        }
        let secs = ns as f64 / 1_000_000_000.0;
        self.total_strings.load(Ordering::Relaxed) as f64 / secs
    }

    /// Tokens processed per second (encode + decode combined).
    #[must_use]
    pub fn tokens_per_sec(&self) -> f64 {
        let total_ns = self.total_encode_ns.load(Ordering::Relaxed)
            + self.total_decode_ns.load(Ordering::Relaxed);
        if total_ns == 0 {
            return 0.0;
        }
        let secs = total_ns as f64 / 1_000_000_000.0;
        self.total_tokens.load(Ordering::Relaxed) as f64 / secs
    }

    /// Average encoding time per batch.
    #[must_use]
    pub fn avg_encode_time(&self) -> Duration {
        let batches = self.batch_count.load(Ordering::Relaxed);
        if batches == 0 {
            return Duration::ZERO;
        }
        let ns = self.total_encode_ns.load(Ordering::Relaxed);
        Duration::from_nanos(ns / batches)
    }

    /// Total strings encoded.
    #[must_use]
    pub fn total_strings(&self) -> u64 {
        self.total_strings.load(Ordering::Relaxed)
    }

    /// Total tokens processed.
    #[must_use]
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens.load(Ordering::Relaxed)
    }

    /// Total number of batches encoded.
    #[must_use]
    pub fn batch_count(&self) -> u64 {
        self.batch_count.load(Ordering::Relaxed)
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.total_strings.store(0, Ordering::Relaxed);
        self.total_tokens.store(0, Ordering::Relaxed);
        self.total_encode_ns.store(0, Ordering::Relaxed);
        self.total_decode_ns.store(0, Ordering::Relaxed);
        self.batch_count.store(0, Ordering::Relaxed);
    }
}

/// Encode a batch while recording metrics.
pub fn encode_with_metrics<T: TokenizerHandle + 'static>(
    encoder: &BatchEncoder<T>,
    texts: &[&str],
    metrics: &TokenizationMetrics,
) -> TokenizedBatch {
    let start = Instant::now();
    let batch = encoder.encode(texts);
    let elapsed = start.elapsed();
    let num_tokens: u64 = batch.lengths.iter().map(|l| *l as u64).sum();
    metrics.record_encode(texts.len() as u64, num_tokens, elapsed);
    batch
}

/// Decode sequences while recording metrics.
pub fn decode_with_metrics<T: TokenizerHandle + 'static>(
    decoder: &BatchDecoder<T>,
    sequences: &[Vec<u32>],
    metrics: &TokenizationMetrics,
) -> Vec<String> {
    let start = Instant::now();
    let results = decoder.decode(sequences);
    let elapsed = start.elapsed();
    let num_tokens: u64 = sequences.iter().map(|s| s.len() as u64).sum();
    metrics.record_decode(num_tokens, elapsed);
    results
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tokenizer() -> Arc<MockTokenizer> {
        Arc::new(MockTokenizer::new(&[
            "hello", "world", "foo", "bar", "baz", "the", "quick", "brown", "fox",
            "jumps", "over", "lazy", "dog", "a", "b", "c", "d", "e",
        ]))
    }

    fn default_config() -> BatchTokenizerConfig {
        BatchTokenizerConfig::default()
    }

    // ── MockTokenizer tests ──────────────────────────────────────────────

    #[test]
    fn mock_tokenizer_encode_known_words() {
        let tok = MockTokenizer::new(&["hello", "world"]);
        let ids = tok.encode("hello world");
        assert_eq!(ids.len(), 2);
        assert_ne!(ids[0], 1); // not UNK
        assert_ne!(ids[1], 1);
    }

    #[test]
    fn mock_tokenizer_encode_unknown_word() {
        let tok = MockTokenizer::new(&["hello"]);
        let ids = tok.encode("hello unknown");
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[1], 1); // UNK
    }

    #[test]
    fn mock_tokenizer_encode_empty() {
        let tok = MockTokenizer::new(&["hello"]);
        let ids = tok.encode("");
        assert!(ids.is_empty());
    }

    #[test]
    fn mock_tokenizer_decode_roundtrip() {
        let tok = MockTokenizer::new(&["hello", "world"]);
        let ids = tok.encode("hello world");
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn mock_tokenizer_decode_empty() {
        let tok = MockTokenizer::new(&["hello"]);
        assert_eq!(tok.decode(&[]), "");
    }

    #[test]
    fn mock_tokenizer_vocab_size() {
        let tok = MockTokenizer::new(&["a", "b", "c"]);
        // 2 specials + 3 words = 5
        assert_eq!(tok.vocab_size(), 5);
    }

    #[test]
    fn mock_tokenizer_special_tokens() {
        let tok = MockTokenizer::new(&[]);
        let specials = tok.special_tokens();
        assert_eq!(specials.get("PAD"), Some(&0));
        assert_eq!(specials.get("UNK"), Some(&1));
    }

    #[test]
    fn mock_tokenizer_duplicate_words() {
        let tok = MockTokenizer::new(&["hello", "hello", "world"]);
        assert_eq!(tok.vocab_size(), 4); // pad, unk, hello, world
    }

    #[test]
    fn mock_tokenizer_len() {
        let tok = MockTokenizer::new(&["a", "b"]);
        assert_eq!(tok.len(), 4); // pad, unk, a, b
        assert!(!tok.is_empty());
    }

    #[test]
    fn mock_tokenizer_multiple_spaces() {
        let tok = MockTokenizer::new(&["hello", "world"]);
        let ids = tok.encode("hello   world");
        assert_eq!(ids.len(), 2); // split_whitespace ignores extra spaces
    }

    // ── Truncation tests ─────────────────────────────────────────────────

    #[test]
    fn truncate_right() {
        let mut tokens = vec![1, 2, 3, 4, 5];
        truncate(&mut tokens, 3, TruncationStrategy::TruncateRight);
        assert_eq!(tokens, vec![1, 2, 3]);
    }

    #[test]
    fn truncate_left() {
        let mut tokens = vec![1, 2, 3, 4, 5];
        truncate(&mut tokens, 3, TruncationStrategy::TruncateLeft);
        assert_eq!(tokens, vec![3, 4, 5]);
    }

    #[test]
    fn truncate_middle_even() {
        let mut tokens = vec![1, 2, 3, 4, 5, 6];
        truncate(&mut tokens, 4, TruncationStrategy::TruncateMiddle);
        // keep_start=2, keep_end=2 → [1,2,5,6]
        assert_eq!(tokens, vec![1, 2, 5, 6]);
    }

    #[test]
    fn truncate_middle_odd() {
        let mut tokens = vec![1, 2, 3, 4, 5, 6, 7];
        truncate(&mut tokens, 3, TruncationStrategy::TruncateMiddle);
        // keep_start=2, keep_end=1 → [1,2,7]
        assert_eq!(tokens, vec![1, 2, 7]);
    }

    #[test]
    fn truncate_none_strategy() {
        let mut tokens = vec![1, 2, 3, 4, 5];
        truncate(&mut tokens, 3, TruncationStrategy::None);
        assert_eq!(tokens, vec![1, 2, 3, 4, 5]); // unchanged
    }

    #[test]
    fn truncate_already_short() {
        let mut tokens = vec![1, 2];
        truncate(&mut tokens, 5, TruncationStrategy::TruncateRight);
        assert_eq!(tokens, vec![1, 2]); // unchanged
    }

    #[test]
    fn truncate_to_zero() {
        let mut tokens = vec![1, 2, 3];
        truncate(&mut tokens, 0, TruncationStrategy::TruncateRight);
        assert!(tokens.is_empty());
    }

    #[test]
    fn truncate_middle_to_zero() {
        let mut tokens = vec![1, 2, 3];
        truncate(&mut tokens, 0, TruncationStrategy::TruncateMiddle);
        assert!(tokens.is_empty());
    }

    #[test]
    fn truncate_exact_length() {
        let mut tokens = vec![1, 2, 3];
        truncate(&mut tokens, 3, TruncationStrategy::TruncateRight);
        assert_eq!(tokens, vec![1, 2, 3]); // unchanged
    }

    #[test]
    fn truncate_to_one() {
        let mut tokens = vec![1, 2, 3, 4, 5];
        truncate(&mut tokens, 1, TruncationStrategy::TruncateRight);
        assert_eq!(tokens, vec![1]);

        let mut tokens2 = vec![1, 2, 3, 4, 5];
        truncate(&mut tokens2, 1, TruncationStrategy::TruncateLeft);
        assert_eq!(tokens2, vec![5]);

        let mut tokens3 = vec![1, 2, 3, 4, 5];
        truncate(&mut tokens3, 1, TruncationStrategy::TruncateMiddle);
        assert_eq!(tokens3, vec![1]);
    }

    // ── Padding tests ────────────────────────────────────────────────────

    #[test]
    fn pad_to_max() {
        let lengths = vec![3, 5, 2];
        let target = compute_pad_length(&lengths, PaddingStrategy::PadToMax, 10);
        assert_eq!(target, Some(10));
    }

    #[test]
    fn pad_to_longest() {
        let lengths = vec![3, 5, 2];
        let target = compute_pad_length(&lengths, PaddingStrategy::PadToLongest, 10);
        assert_eq!(target, Some(5));
    }

    #[test]
    fn pad_to_multiple_of() {
        let lengths = vec![3, 5, 2];
        let target = compute_pad_length(&lengths, PaddingStrategy::PadToMultipleOf(4), 10);
        assert_eq!(target, Some(8)); // next multiple of 4 >= 5
    }

    #[test]
    fn pad_to_multiple_of_exact() {
        let lengths = vec![4, 8];
        let target = compute_pad_length(&lengths, PaddingStrategy::PadToMultipleOf(4), 10);
        assert_eq!(target, Some(8)); // already a multiple
    }

    #[test]
    fn pad_to_multiple_of_zero() {
        let lengths = vec![3];
        let target = compute_pad_length(&lengths, PaddingStrategy::PadToMultipleOf(0), 10);
        assert_eq!(target, None);
    }

    #[test]
    fn no_padding() {
        let lengths = vec![3, 5];
        let target = compute_pad_length(&lengths, PaddingStrategy::NoPadding, 10);
        assert_eq!(target, None);
    }

    #[test]
    fn pad_sequence_adds_tokens() {
        let mut tokens = vec![10, 20, 30];
        let mask = pad_sequence(&mut tokens, 5, 0);
        assert_eq!(tokens, vec![10, 20, 30, 0, 0]);
        assert_eq!(mask, vec![1, 1, 1, 0, 0]);
    }

    #[test]
    fn pad_sequence_already_at_target() {
        let mut tokens = vec![10, 20, 30];
        let mask = pad_sequence(&mut tokens, 3, 0);
        assert_eq!(tokens, vec![10, 20, 30]);
        assert_eq!(mask, vec![1, 1, 1]);
    }

    #[test]
    fn pad_empty_lengths() {
        let lengths: Vec<usize> = vec![];
        assert_eq!(
            compute_pad_length(&lengths, PaddingStrategy::PadToLongest, 10),
            Some(0)
        );
    }

    // ── BatchEncoder tests ───────────────────────────────────────────────

    #[test]
    fn batch_encode_basic() {
        let tok = make_tokenizer();
        let encoder = BatchEncoder::new(tok, default_config());
        let batch = encoder.encode(&["hello world", "foo bar"]);
        assert_eq!(batch.batch_size(), 2);
        assert!(batch.is_uniform_length());
    }

    #[test]
    fn batch_encode_single() {
        let tok = make_tokenizer();
        let encoder = BatchEncoder::new(tok, default_config());
        let batch = encoder.encode(&["hello"]);
        assert_eq!(batch.batch_size(), 1);
        assert_eq!(batch.lengths, vec![1]);
    }

    #[test]
    fn batch_encode_empty_input() {
        let tok = make_tokenizer();
        let encoder = BatchEncoder::new(tok, default_config());
        let batch = encoder.encode(&[]);
        assert_eq!(batch.batch_size(), 0);
        assert!(batch.input_ids.is_empty());
    }

    #[test]
    fn batch_encode_empty_string() {
        let tok = make_tokenizer();
        let encoder = BatchEncoder::new(tok, default_config());
        let batch = encoder.encode(&[""]);
        assert_eq!(batch.batch_size(), 1);
        assert_eq!(batch.lengths[0], 0);
    }

    #[test]
    fn batch_encode_respects_max_batch_size() {
        let tok = make_tokenizer();
        let mut config = default_config();
        config.max_batch_size = 2;
        let encoder = BatchEncoder::new(tok, config);
        let batch = encoder.encode(&["hello", "world", "foo", "bar"]);
        assert_eq!(batch.batch_size(), 2);
    }

    #[test]
    fn batch_encode_truncation_right() {
        let tok = make_tokenizer();
        let mut config = default_config();
        config.max_seq_length = 2;
        config.truncation_strategy = TruncationStrategy::TruncateRight;
        let encoder = BatchEncoder::new(tok, config);
        let batch = encoder.encode(&["hello world foo bar"]);
        assert_eq!(batch.lengths[0], 2);
    }

    #[test]
    fn batch_encode_truncation_left() {
        let tok = make_tokenizer();
        let mut config = default_config();
        config.max_seq_length = 2;
        config.truncation_strategy = TruncationStrategy::TruncateLeft;
        let encoder = BatchEncoder::new(tok.clone(), config);
        let batch = encoder.encode(&["hello world foo bar"]);
        assert_eq!(batch.lengths[0], 2);
        // Last two tokens should be "foo" and "bar"
        let expected = tok.encode("foo bar");
        assert_eq!(batch.input_ids[0][..2], expected[..2]);
    }

    #[test]
    fn batch_encode_no_padding() {
        let tok = make_tokenizer();
        let mut config = default_config();
        config.padding_strategy = PaddingStrategy::NoPadding;
        let encoder = BatchEncoder::new(tok, config);
        let batch = encoder.encode(&["hello", "hello world"]);
        // Different lengths
        assert_eq!(batch.input_ids[0].len(), 1);
        assert_eq!(batch.input_ids[1].len(), 2);
    }

    #[test]
    fn batch_encode_pad_to_max() {
        let tok = make_tokenizer();
        let mut config = default_config();
        config.padding_strategy = PaddingStrategy::PadToMax;
        config.max_seq_length = 10;
        let encoder = BatchEncoder::new(tok, config);
        let batch = encoder.encode(&["hello", "hello world"]);
        assert_eq!(batch.input_ids[0].len(), 10);
        assert_eq!(batch.input_ids[1].len(), 10);
    }

    #[test]
    fn batch_encode_attention_mask_correctness() {
        let tok = make_tokenizer();
        let mut config = default_config();
        config.padding_strategy = PaddingStrategy::PadToLongest;
        let encoder = BatchEncoder::new(tok, config);
        let batch = encoder.encode(&["hello", "hello world foo"]);
        // "hello" → 1 token, "hello world foo" → 3 tokens, padded to 3
        assert_eq!(batch.attention_masks[0], vec![1, 0, 0]);
        assert_eq!(batch.attention_masks[1], vec![1, 1, 1]);
    }

    #[test]
    fn batch_encode_same_as_individual() {
        let tok = make_tokenizer();
        let texts = ["hello world", "foo bar baz", "the quick brown fox"];
        let encoder = BatchEncoder::new(tok.clone(), {
            let mut c = default_config();
            c.padding_strategy = PaddingStrategy::NoPadding;
            c.truncation_strategy = TruncationStrategy::None;
            c
        });
        let batch = encoder.encode(&texts);
        for (i, text) in texts.iter().enumerate() {
            let individual = tok.encode(text);
            assert_eq!(batch.input_ids[i], individual, "mismatch at index {i}");
        }
    }

    #[test]
    fn batch_encode_parallel_same_as_sequential() {
        let tok = make_tokenizer();
        let texts: Vec<&str> = vec![
            "hello world",
            "foo bar",
            "the quick brown fox",
            "a b c d e",
            "baz",
            "jumps over lazy dog",
        ];

        let seq_config = BatchTokenizerConfig {
            num_threads: 1,
            padding_strategy: PaddingStrategy::NoPadding,
            truncation_strategy: TruncationStrategy::None,
            ..default_config()
        };
        let par_config = BatchTokenizerConfig {
            num_threads: 4,
            padding_strategy: PaddingStrategy::NoPadding,
            truncation_strategy: TruncationStrategy::None,
            ..default_config()
        };

        let seq_encoder = BatchEncoder::new(tok.clone(), seq_config);
        let par_encoder = BatchEncoder::new(tok, par_config);

        let seq_batch = seq_encoder.encode(&texts);
        let par_batch = par_encoder.encode(&texts);

        assert_eq!(seq_batch.input_ids, par_batch.input_ids);
        assert_eq!(seq_batch.lengths, par_batch.lengths);
    }

    #[test]
    fn batch_encode_pad_to_multiple_of() {
        let tok = make_tokenizer();
        let mut config = default_config();
        config.padding_strategy = PaddingStrategy::PadToMultipleOf(8);
        let encoder = BatchEncoder::new(tok, config);
        let batch = encoder.encode(&["hello world foo"]); // 3 tokens → pad to 8
        assert_eq!(batch.input_ids[0].len(), 8);
    }

    // ── BatchDecoder tests ───────────────────────────────────────────────

    #[test]
    fn batch_decode_basic() {
        let tok = make_tokenizer();
        let ids1 = tok.encode("hello world");
        let ids2 = tok.encode("foo bar");
        let decoder = BatchDecoder::new(tok, default_config());
        let results = decoder.decode(&[ids1, ids2]);
        assert_eq!(results[0], "hello world");
        assert_eq!(results[1], "foo bar");
    }

    #[test]
    fn batch_decode_empty() {
        let tok = make_tokenizer();
        let decoder = BatchDecoder::new(tok, default_config());
        let results = decoder.decode(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn batch_decode_respects_max_batch_size() {
        let tok = make_tokenizer();
        let mut config = default_config();
        config.max_batch_size = 2;
        let decoder = BatchDecoder::new(tok.clone(), config);
        let seqs: Vec<Vec<u32>> = vec![
            tok.encode("hello"),
            tok.encode("world"),
            tok.encode("foo"),
        ];
        let results = decoder.decode(&seqs);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn batch_decode_parallel_same_as_sequential() {
        let tok = make_tokenizer();
        let seqs: Vec<Vec<u32>> = vec![
            tok.encode("hello world"),
            tok.encode("foo bar"),
            tok.encode("the quick brown fox"),
            tok.encode("jumps over lazy dog"),
        ];

        let seq_decoder = BatchDecoder::new(tok.clone(), {
            let mut c = default_config();
            c.num_threads = 1;
            c
        });
        let par_decoder = BatchDecoder::new(tok, {
            let mut c = default_config();
            c.num_threads = 4;
            c
        });

        let seq_results = seq_decoder.decode(&seqs);
        let par_results = par_decoder.decode(&seqs);
        assert_eq!(seq_results, par_results);
    }

    // ── Encode-decode roundtrip tests ────────────────────────────────────

    #[test]
    fn encode_decode_roundtrip() {
        let tok = make_tokenizer();
        let encoder = BatchEncoder::new(tok.clone(), {
            let mut c = default_config();
            c.padding_strategy = PaddingStrategy::NoPadding;
            c
        });
        let decoder = BatchDecoder::new(tok, default_config());
        let texts = ["hello world", "foo bar baz"];
        let batch = encoder.encode(&texts);
        let roundtrip = decoder.decode(&batch.input_ids);
        assert_eq!(roundtrip[0], "hello world");
        assert_eq!(roundtrip[1], "foo bar baz");
    }

    // ── TokenizedBatch method tests ──────────────────────────────────────

    #[test]
    fn tokenized_batch_is_uniform_empty() {
        let batch = TokenizedBatch {
            input_ids: vec![],
            attention_masks: vec![],
            lengths: vec![],
            padding_token_id: 0,
        };
        assert!(batch.is_uniform_length());
        assert_eq!(batch.padded_length(), 0);
        assert_eq!(batch.total_real_tokens(), 0);
        assert!(batch.utilization().abs() < f64::EPSILON);
    }

    #[test]
    fn tokenized_batch_utilization(){
        let batch = TokenizedBatch {
            input_ids: vec![vec![1, 2, 0, 0], vec![3, 4, 5, 0]],
            attention_masks: vec![vec![1, 1, 0, 0], vec![1, 1, 1, 0]],
            lengths: vec![2, 3],
            padding_token_id: 0,
        };
        // 5 real / 8 total = 0.625
        assert!((batch.utilization() - 0.625).abs() < 1e-9);
    }

    #[test]
    fn tokenized_batch_padded_length() {
        let batch = TokenizedBatch {
            input_ids: vec![vec![1, 2, 0], vec![3, 0, 0]],
            attention_masks: vec![vec![1, 1, 0], vec![1, 0, 0]],
            lengths: vec![2, 1],
            padding_token_id: 0,
        };
        assert_eq!(batch.padded_length(), 3);
        assert!(batch.is_uniform_length());
    }

    // ── StreamingTokenizer tests ─────────────────────────────────────────

    #[test]
    fn streaming_feed_one_at_a_time() {
        let tok = make_tokenizer();
        let ids = tok.encode("hello world foo");
        let mut streamer = StreamingTokenizer::new(tok);

        let mut accumulated= String::new();
        for &id in &ids {
            accumulated.push_str(&streamer.feed(id));
        }
        assert_eq!(accumulated, "hello world foo");
    }

    #[test]
    fn streaming_feed_many() {
        let tok = make_tokenizer();
        let ids = tok.encode("hello world");
        let mut streamer = StreamingTokenizer::new(tok);
        let result = streamer.feed_many(&ids);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn streaming_text_matches_full_decode() {
        let tok = make_tokenizer();
        let ids = tok.encode("the quick brown fox");
        let mut streamer = StreamingTokenizer::new(tok.clone());
        for &id in &ids {
            streamer.feed(id);
        }
        assert_eq!(streamer.text(), tok.decode(&ids));
    }

    #[test]
    fn streaming_tokens_buffer() {
        let tok = make_tokenizer();
        let ids = tok.encode("hello world");
        let mut streamer = StreamingTokenizer::new(tok);
        for &id in &ids {
            streamer.feed(id);
        }
        assert_eq!(streamer.tokens(), &ids);
        assert_eq!(streamer.len(), 2);
        assert!(!streamer.is_empty());
    }

    #[test]
    fn streaming_reset() {
        let tok = make_tokenizer();
        let mut streamer = StreamingTokenizer::new(tok);
        streamer.feed(2);
        streamer.reset();
        assert!(streamer.is_empty());
        assert_eq!(streamer.text(), "");
    }

    #[test]
    fn streaming_empty_initial() {
        let tok = make_tokenizer();
        let streamer = StreamingTokenizer::new(tok);
        assert!(streamer.is_empty());
        assert_eq!(streamer.len(), 0);
        assert_eq!(streamer.text(), "");
    }

    // ── TokenizationMetrics tests ────────────────────────────────────────

    #[test]
    fn metrics_initial_state() {
        let m = TokenizationMetrics::new();
        assert_eq!(m.total_strings(), 0);
        assert_eq!(m.total_tokens(), 0);
        assert_eq!(m.batch_count(), 0);
        assert!(m.strings_per_sec().abs() < f64::EPSILON);
        assert!(m.tokens_per_sec().abs() < f64::EPSILON);
        assert_eq!(m.avg_encode_time(), Duration::ZERO);
    }

    #[test]
    fn metrics_record_encode() {
        let m = TokenizationMetrics::new();
        m.record_encode(10, 50, Duration::from_secs(1));
        assert_eq!(m.total_strings(), 10);
        assert_eq!(m.total_tokens(), 50);
        assert_eq!(m.batch_count(), 1);
        assert!((m.strings_per_sec() - 10.0).abs() < 0.1);
    }

    #[test]
    fn metrics_record_decode() {
        let m = TokenizationMetrics::new();
        m.record_decode(100, Duration::from_secs(2));
        assert_eq!(m.total_tokens(), 100);
    }

    #[test]
    fn metrics_reset() {
        let m = TokenizationMetrics::new();
        m.record_encode(5, 25, Duration::from_millis(100));
        m.reset();
        assert_eq!(m.total_strings(), 0);
        assert_eq!(m.total_tokens(), 0);
        assert_eq!(m.batch_count(), 0);
    }

    #[test]
    fn metrics_multiple_batches_avg() {
        let m = TokenizationMetrics::new();
        m.record_encode(10, 50, Duration::from_secs(1));
        m.record_encode(10, 50, Duration::from_secs(3));
        assert_eq!(m.batch_count(), 2);
        // avg = 4s / 2 = 2s
        let avg = m.avg_encode_time();
        assert!(avg >= Duration::from_secs(1) && avg <= Duration::from_secs(3));
    }

    #[test]
    fn metrics_default_impl() {
        let m = TokenizationMetrics::default();
        assert_eq!(m.total_strings(), 0);
    }

    // ── encode_with_metrics / decode_with_metrics tests ──────────────────

    #[test]
    fn encode_with_metrics_records() {
        let tok = make_tokenizer();
        let encoder = BatchEncoder::new(tok, default_config());
        let metrics = TokenizationMetrics::new();
        let batch = encode_with_metrics(&encoder, &["hello world", "foo"], &metrics);
        assert_eq!(batch.batch_size(), 2);
        assert_eq!(metrics.total_strings(), 2);
        assert!(metrics.total_tokens() > 0);
        assert_eq!(metrics.batch_count(), 1);
    }

    #[test]
    fn decode_with_metrics_records() {
        let tok = make_tokenizer();
        let seqs = vec![tok.encode("hello"), tok.encode("world")];
        let decoder = BatchDecoder::new(tok, default_config());
        let metrics = TokenizationMetrics::new();
        let results = decode_with_metrics(&decoder, &seqs, &metrics);
        assert_eq!(results.len(), 2);
        assert!(metrics.total_tokens() > 0);
    }

    // ── Thread safety tests ──────────────────────────────────────────────

    #[test]
    fn metrics_thread_safe() {
        let m = Arc::new(TokenizationMetrics::new());
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let m = Arc::clone(&m);
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        m.record_encode(1, 5, Duration::from_nanos(1000));
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(m.total_strings(), 800);
        assert_eq!(m.batch_count(), 800);
    }

    #[test]
    fn encoder_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BatchEncoder<MockTokenizer>>();
    }

    #[test]
    fn decoder_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BatchDecoder<MockTokenizer>>();
    }

    // ── Edge-case / property tests ───────────────────────────────────────

    #[test]
    fn padding_invariant_all_same_length() {
        let tok = make_tokenizer();
        let mut config = default_config();
        config.padding_strategy = PaddingStrategy::PadToLongest;
        let encoder = BatchEncoder::new(tok, config);
        let texts = [
            "hello",
            "hello world",
            "hello world foo",
            "a b c d",
        ];
        let batch = encoder.encode(&texts);
        assert!(batch.is_uniform_length());
        let expected_len = batch.padded_length();
        for seq in &batch.input_ids {
            assert_eq!(seq.len(), expected_len);
        }
        for mask in &batch.attention_masks {
            assert_eq!(mask.len(), expected_len);
        }
    }

    #[test]
    fn attention_mask_ones_match_original_length() {
        let tok = make_tokenizer();
        let mut config = default_config();
        config.padding_strategy = PaddingStrategy::PadToLongest;
        let encoder = BatchEncoder::new(tok, config);
        let batch = encoder.encode(&["hello", "hello world foo bar"]);
        for (i, mask) in batch.attention_masks.iter().enumerate() {
            let ones: usize = mask.iter().map(|&x| x as usize).sum();
            assert_eq!(ones, batch.lengths[i]);
        }
    }

    #[test]
    fn padding_tokens_are_pad_id() {
        let tok = make_tokenizer();
        let pad_id = 42u32;
        let mut config = default_config();
        config.padding_strategy = PaddingStrategy::PadToMax;
        config.max_seq_length = 10;
        config.padding_token_id = pad_id;
        let encoder = BatchEncoder::new(tok, config);
        let batch = encoder.encode(&["hello world"]);
        let seq = &batch.input_ids[0];
        let real_len = batch.lengths[0];
        for &token in &seq[real_len..] {
            assert_eq!(token, pad_id);
        }
    }

    #[test]
    fn max_seq_length_enforcement() {
        let tok = make_tokenizer();
        let mut config = default_config();
        config.max_seq_length = 3;
        config.truncation_strategy = TruncationStrategy::TruncateRight;
        config.padding_strategy = PaddingStrategy::NoPadding;
        let encoder = BatchEncoder::new(tok, config);
        let batch = encoder.encode(&["hello world foo bar baz"]);
        assert!(batch.lengths[0] <= 3);
    }

    #[test]
    fn all_unknown_words() {
        let tok = Arc::new(MockTokenizer::new(&[]));
        let encoder = BatchEncoder::new(tok, default_config());
        let batch = encoder.encode(&["xyz abc def"]);
        // All should be UNK (1)
        assert!(batch.input_ids[0].iter().all(|&id| id == 1));
    }

    #[test]
    fn single_token_sequences() {
        let tok = make_tokenizer();
        let encoder = BatchEncoder::new(tok, {
            let mut c = default_config();
            c.padding_strategy = PaddingStrategy::PadToLongest;
            c
        });
        let batch = encoder.encode(&["hello", "world", "foo"]);
        assert!(batch.is_uniform_length());
        assert_eq!(batch.padded_length(), 1);
        // No padding needed since all are length 1
        for mask in &batch.attention_masks {
            assert_eq!(mask, &[1]);
        }
    }

    #[test]
    fn batch_config_default_values() {
        let config = BatchTokenizerConfig::default();
        assert_eq!(config.max_batch_size, 32);
        assert_eq!(config.max_seq_length, 512);
        assert_eq!(config.num_threads, 4);
        assert_eq!(config.padding_strategy, PaddingStrategy::PadToLongest);
        assert_eq!(
            config.truncation_strategy,
            TruncationStrategy::TruncateRight
        );
        assert_eq!(config.padding_token_id, 0);
    }

    #[test]
    fn padding_strategy_equality() {
        assert_eq!(PaddingStrategy::NoPadding, PaddingStrategy::NoPadding);
        assert_eq!(
            PaddingStrategy::PadToMultipleOf(8),
            PaddingStrategy::PadToMultipleOf(8)
        );
        assert_ne!(
            PaddingStrategy::PadToMultipleOf(8),
            PaddingStrategy::PadToMultipleOf(16)
        );
    }

    #[test]
    fn truncation_strategy_equality() {
        assert_eq!(TruncationStrategy::None, TruncationStrategy::None);
        assert_ne!(
            TruncationStrategy::TruncateLeft,
            TruncationStrategy::TruncateRight
        );
    }

    #[test]
    fn large_batch_parallel_correctness() {
        let tok = make_tokenizer();
        let texts: Vec<&str> = (0..100)
            .map(|i| match i % 4 {
                0 => "hello world",
                1 => "foo bar baz",
                2 => "the quick brown fox",
                _ => "jumps over lazy dog",
            })
            .collect();
        let encoder = BatchEncoder::new(tok, {
            let mut c = default_config();
            c.max_batch_size = 200;
            c.num_threads = 8;
            c.padding_strategy = PaddingStrategy::PadToLongest;
            c
        });
        let batch = encoder.encode(&texts);
        assert_eq!(batch.batch_size(), 100);
        assert!(batch.is_uniform_length());
    }

    #[test]
    fn pad_to_multiple_rounds_up() {
        let lengths = vec![1];
        let target = compute_pad_length(&lengths, PaddingStrategy::PadToMultipleOf(16), 512);
        assert_eq!(target, Some(16));
    }

    #[test]
    fn batch_encode_preserves_order() {
        let tok = make_tokenizer();
        let texts = ["hello", "world", "foo", "bar", "baz"];
        let encoder = BatchEncoder::new(tok.clone(), {
            let mut c = default_config();
            c.num_threads = 4;
            c.padding_strategy = PaddingStrategy::NoPadding;
            c
        });
        let batch = encoder.encode(&texts);
        for (i, text) in texts.iter().enumerate() {
            let expected = tok.encode(text);
            assert_eq!(batch.input_ids[i], expected, "order mismatch at {i}");
        }
    }

    #[test]
    fn streaming_incremental_matches_full() {
        let tok = make_tokenizer();
        let text = "the quick brown fox jumps over lazy dog";
        let ids = tok.encode(text);
        let mut streamer = StreamingTokenizer::new(tok.clone());
        let mut incremental = String::new();
        for &id in &ids {
            incremental.push_str(&streamer.feed(id));
        }
        let full = tok.decode(&ids);
        assert_eq!(incremental, full);
    }

    #[test]
    fn decode_with_padding_tokens() {
        let tok = make_tokenizer();
        let ids = tok.encode("hello world");
        let mut padded = ids;
        padded.extend_from_slice(&[0, 0, 0]); // padding
        let decoded = tok.decode(&padded);
        // Decoding padding tokens produces "<pad>" words
        assert!(decoded.starts_with("hello world"));
    }

    #[test]
    fn batch_utilization_full() {
        let tok = make_tokenizer();
        let mut config = default_config();
        config.padding_strategy = PaddingStrategy::NoPadding;
        let encoder = BatchEncoder::new(tok, config);
        let batch = encoder.encode(&["hello world", "foo bar"]);
        // No padding → utilization = 1.0
        assert!((batch.utilization() - 1.0).abs() < 1e-9);
    }
}
