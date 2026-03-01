//! Tokenizer/Detokenizer pipeline for text-to-token and token-to-text conversion.
//!
//! Provides a complete pipeline: normalize → pre-tokenize → BPE encode → post-process,
//! with a streaming [`IncrementalDetokenizer`] for partial UTF-8 safe decoding.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Tokenizer model type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    BPE,
    Unigram,
    WordPiece,
}

/// Top-level tokenizer configuration.
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    pub model_type: ModelType,
    pub vocab_size: u32,
    pub add_bos: bool,
    pub add_eos: bool,
    pub pad_token_id: Option<u32>,
}

impl TokenizerConfig {
    pub const fn new(model_type: ModelType, vocab_size: u32) -> Self {
        Self { model_type, vocab_size, add_bos: false, add_eos: false, pad_token_id: None }
    }
}

// ---------------------------------------------------------------------------
// BPE Encoder
// ---------------------------------------------------------------------------

/// A single BPE merge rule: (left, right) → merged token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeRule {
    pub left: String,
    pub right: String,
    pub merged: String,
    pub priority: u32,
}

/// Byte-pair encoding encoder.
#[derive(Debug, Clone)]
pub struct BPEEncoder {
    vocab: HashMap<String, u32>,
    merges: Vec<MergeRule>,
}

impl BPEEncoder {
    pub const fn new(vocab: HashMap<String, u32>, merges: Vec<MergeRule>) -> Self {
        Self { vocab, merges }
    }

    /// Encode `text` into a sequence of token ids using iterative BPE merging.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Start with individual characters as tokens.
        let mut symbols: Vec<String> = text.chars().map(|c| c.to_string()).collect();

        // Iteratively apply the highest-priority applicable merge.
        loop {
            let mut best: Option<(usize, &MergeRule)> = None;
            for rule in &self.merges {
                for i in 0..symbols.len().saturating_sub(1) {
                    if symbols[i] == rule.left
                        && symbols[i + 1] == rule.right
                        && best.is_none_or(|(_, b)| rule.priority < b.priority)
                    {
                        best = Some((i, rule));
                    }
                }
            }

            let Some((idx, rule)) = best else { break };
            symbols[idx].clone_from(&rule.merged);
            symbols.remove(idx + 1);
        }

        symbols.iter().map(|s| self.vocab.get(s).copied().unwrap_or(0)).collect()
    }
}

// ---------------------------------------------------------------------------
// BPE Decoder
// ---------------------------------------------------------------------------

/// Reverse BPE decoder: token ids → string.
#[derive(Debug, Clone)]
pub struct BPEDecoder {
    id_to_token: HashMap<u32, String>,
}

impl BPEDecoder {
    pub const fn new(id_to_token: HashMap<u32, String>) -> Self {
        Self { id_to_token }
    }

    /// Decode a sequence of token ids back into a string.
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|id| self.id_to_token.get(id).cloned().unwrap_or_else(|| "\u{FFFD}".to_string()))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Incremental Detokenizer
// ---------------------------------------------------------------------------

/// Stateful streaming decoder that handles partial UTF-8 byte sequences.
#[derive(Debug)]
pub struct IncrementalDetokenizer {
    id_to_token: HashMap<u32, String>,
    byte_buffer: Vec<u8>,
}

impl IncrementalDetokenizer {
    pub const fn new(id_to_token: HashMap<u32, String>) -> Self {
        Self { id_to_token, byte_buffer: Vec::new() }
    }

    /// Feed a single token id.  Returns decoded text when complete
    /// codepoints are available, or `None` while buffering partial bytes.
    pub fn add_token(&mut self, id: u32) -> Option<String> {
        let piece = self.id_to_token.get(&id).cloned().unwrap_or_else(|| "\u{FFFD}".to_string());

        // Check for byte-level tokens like <0xHH>.
        if let Some(byte_val) = parse_byte_token(&piece) {
            self.byte_buffer.push(byte_val);
            return self.try_flush();
        }

        // Non-byte token: flush any buffered bytes first, then emit.
        let mut out = String::new();
        if !self.byte_buffer.is_empty() {
            out.push_str(&String::from_utf8_lossy(&self.byte_buffer));
            self.byte_buffer.clear();
        }
        out.push_str(&piece);
        Some(out)
    }

    /// Number of bytes currently buffered awaiting a complete codepoint.
    pub const fn buffered_bytes(&self) -> usize {
        self.byte_buffer.len()
    }

    /// Flush remaining buffered bytes (lossy).
    pub fn flush(&mut self) -> Option<String> {
        if self.byte_buffer.is_empty() {
            return None;
        }
        let s = String::from_utf8_lossy(&self.byte_buffer).into_owned();
        self.byte_buffer.clear();
        Some(s)
    }

    /// Try to decode as many complete UTF-8 codepoints as possible from the
    /// buffer, keeping any trailing incomplete sequence.
    fn try_flush(&mut self) -> Option<String> {
        match std::str::from_utf8(&self.byte_buffer) {
            Ok(s) => {
                let out = s.to_string();
                self.byte_buffer.clear();
                Some(out)
            }
            Err(e) => {
                let valid_up_to = e.valid_up_to();
                if valid_up_to == 0 {
                    // Could still be an incomplete prefix – keep buffering.
                    if self.byte_buffer.len() < 4 {
                        return None;
                    }
                    // Definitely invalid; emit replacement.
                    let s = String::from_utf8_lossy(&self.byte_buffer).into_owned();
                    self.byte_buffer.clear();
                    Some(s)
                } else {
                    let decoded =
                        std::str::from_utf8(&self.byte_buffer[..valid_up_to]).unwrap().to_string();
                    self.byte_buffer = self.byte_buffer[valid_up_to..].to_vec();
                    Some(decoded)
                }
            }
        }
    }
}

/// Parse a byte-level token like `<0xE2>` and return the byte value.
fn parse_byte_token(s: &str) -> Option<u8> {
    let s = s.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(s, 16).ok()
}

// ---------------------------------------------------------------------------
// Special Token Handler
// ---------------------------------------------------------------------------

/// Well-known special token roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecialTokenKind {
    BOS,
    EOS,
    PAD,
    UNK,
    MASK,
}

/// Manages adding and stripping special tokens.
#[derive(Debug, Clone)]
pub struct SpecialTokenHandler {
    tokens: HashMap<SpecialTokenKind, u32>,
}

impl SpecialTokenHandler {
    pub const fn new(tokens: HashMap<SpecialTokenKind, u32>) -> Self {
        Self { tokens }
    }

    /// Get the id for a special token kind, if registered.
    pub fn id_of(&self, kind: SpecialTokenKind) -> Option<u32> {
        self.tokens.get(&kind).copied()
    }

    /// Prepend BOS and/or append EOS based on config flags.
    pub fn add_special(&self, mut ids: Vec<u32>, config: &TokenizerConfig) -> Vec<u32> {
        if config.add_bos
            && let Some(bos) = self.id_of(SpecialTokenKind::BOS)
        {
            ids.insert(0, bos);
        }
        if config.add_eos
            && let Some(eos) = self.id_of(SpecialTokenKind::EOS)
        {
            ids.push(eos);
        }
        ids
    }

    /// Remove all known special token ids from the sequence.
    pub fn strip_special(&self, ids: &[u32]) -> Vec<u32> {
        let special_ids: std::collections::HashSet<u32> = self.tokens.values().copied().collect();
        ids.iter().copied().filter(|id| !special_ids.contains(id)).collect()
    }
}

// ---------------------------------------------------------------------------
// Pre-Tokenizer
// ---------------------------------------------------------------------------

/// Pre-tokenization strategy applied before BPE encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreTokenizerKind {
    /// Split on whitespace boundaries.
    Whitespace,
    /// Byte-level BPE: keep text as-is (individual bytes).
    ByteLevel,
}

/// Pre-tokenizer splits raw text into preliminary chunks.
#[derive(Debug, Clone)]
pub struct PreTokenizer {
    pub kind: PreTokenizerKind,
}

impl PreTokenizer {
    pub const fn new(kind: PreTokenizerKind) -> Self {
        Self { kind }
    }

    /// Split text into pre-token chunks.
    pub fn pre_tokenize(&self, text: &str) -> Vec<String> {
        match self.kind {
            PreTokenizerKind::Whitespace => text.split_whitespace().map(String::from).collect(),
            PreTokenizerKind::ByteLevel => {
                if text.is_empty() {
                    Vec::new()
                } else {
                    vec![text.to_string()]
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Post-Processor
// ---------------------------------------------------------------------------

/// Post-processing mode for encoded sequences.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PostProcessorKind {
    /// No post-processing.
    Identity,
    /// Template-based, e.g. `"[CLS] $A [SEP]"`.
    Template(String),
}

/// Applies post-processing to token id sequences.
#[derive(Debug, Clone)]
pub struct PostProcessor {
    pub kind: PostProcessorKind,
    token_lookup: HashMap<String, u32>,
}

impl PostProcessor {
    pub const fn new(kind: PostProcessorKind, token_lookup: HashMap<String, u32>) -> Self {
        Self { kind, token_lookup }
    }

    /// Apply post-processing to a token sequence.
    pub fn process(&self, ids: Vec<u32>) -> Vec<u32> {
        match &self.kind {
            PostProcessorKind::Identity => ids,
            PostProcessorKind::Template(tmpl) => {
                let mut result = Vec::new();
                for part in tmpl.split_whitespace() {
                    if part == "$A" {
                        result.extend_from_slice(&ids);
                    } else if let Some(&tok_id) = self.token_lookup.get(part) {
                        result.push(tok_id);
                    }
                    // Unknown template tokens are silently skipped.
                }
                result
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Normalizer Pipeline
// ---------------------------------------------------------------------------

/// Individual normalization operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizerOp {
    /// Unicode NFC normalization.
    NFC,
    /// Unicode NFD normalization.
    NFD,
    /// Unicode NFKC normalization.
    NFKC,
    /// Convert to lowercase.
    Lowercase,
    /// Strip leading/trailing whitespace.
    StripWhitespace,
}

/// Applies a sequence of normalization operations to input text.
#[derive(Debug, Clone)]
pub struct NormalizerPipeline {
    ops: Vec<NormalizerOp>,
}

impl NormalizerPipeline {
    pub const fn new(ops: Vec<NormalizerOp>) -> Self {
        Self { ops }
    }

    /// Apply all normalization operations in order.
    pub fn normalize(&self, text: &str) -> String {
        let mut s = text.to_string();
        for op in &self.ops {
            s = match op {
                // Without the `unicode-normalization` crate we do a
                // best-effort identity for NFC/NFD/NFKC.  The structure is
                // in place for downstream to plug in a real implementation.
                NormalizerOp::NFC | NormalizerOp::NFD | NormalizerOp::NFKC => s,
                NormalizerOp::Lowercase => s.to_lowercase(),
                NormalizerOp::StripWhitespace => s.trim().to_string(),
            };
        }
        s
    }
}

// ---------------------------------------------------------------------------
// Tokenizer Pipeline
// ---------------------------------------------------------------------------

/// Full tokenization pipeline: normalize → pre-tokenize → encode → post-process.
#[derive(Debug)]
pub struct TokenizerPipeline {
    pub config: TokenizerConfig,
    pub normalizer: Option<NormalizerPipeline>,
    pub pre_tokenizer: PreTokenizer,
    pub encoder: BPEEncoder,
    pub post_processor: PostProcessor,
    pub special_handler: SpecialTokenHandler,
}

impl TokenizerPipeline {
    /// Run the full encoding pipeline on `text`.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // 1. Normalize
        let normed =
            self.normalizer.as_ref().map_or_else(|| text.to_string(), |n| n.normalize(text));

        // 2. Pre-tokenize
        let chunks = self.pre_tokenizer.pre_tokenize(&normed);

        // 3. BPE-encode each chunk and concatenate.
        let mut ids: Vec<u32> = Vec::new();
        for chunk in &chunks {
            ids.extend(self.encoder.encode(chunk));
        }

        // 4. Add special tokens.
        ids = self.special_handler.add_special(ids, &self.config);

        // 5. Post-process.
        self.post_processor.process(ids)
    }
}

// ---------------------------------------------------------------------------
// Detokenizer Metrics
// ---------------------------------------------------------------------------

/// Counters for the detokenization path.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DetokenizerMetrics {
    pub tokens_decoded: u64,
    pub partial_bytes_buffered: u64,
    pub special_tokens_stripped: u64,
}

impl DetokenizerMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn record_decode(&mut self, count: u64) {
        self.tokens_decoded += count;
    }

    pub const fn record_partial_bytes(&mut self, count: u64) {
        self.partial_bytes_buffered += count;
    }

    pub const fn record_special_stripped(&mut self, count: u64) {
        self.special_tokens_stripped += count;
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----- helpers --------------------------------------------------------

    /// Build a tiny vocab + merge set for testing.
    fn tiny_bpe() -> (HashMap<String, u32>, Vec<MergeRule>, HashMap<u32, String>) {
        let mut vocab = HashMap::new();
        vocab.insert("h".to_string(), 1);
        vocab.insert("e".to_string(), 2);
        vocab.insert("l".to_string(), 3);
        vocab.insert("o".to_string(), 4);
        vocab.insert("he".to_string(), 5);
        vocab.insert("ll".to_string(), 6);
        vocab.insert("lo".to_string(), 7);
        vocab.insert("hel".to_string(), 8);
        vocab.insert("hello".to_string(), 9);

        let merges = vec![
            MergeRule { left: "h".into(), right: "e".into(), merged: "he".into(), priority: 0 },
            MergeRule { left: "l".into(), right: "o".into(), merged: "lo".into(), priority: 1 },
            MergeRule { left: "he".into(), right: "l".into(), merged: "hel".into(), priority: 2 },
            MergeRule {
                left: "hel".into(),
                right: "lo".into(),
                merged: "hello".into(),
                priority: 3,
            },
            MergeRule { left: "l".into(), right: "l".into(), merged: "ll".into(), priority: 4 },
        ];

        let id_to_token: HashMap<u32, String> =
            vocab.iter().map(|(k, &v)| (v, k.clone())).collect();

        (vocab, merges, id_to_token)
    }

    fn default_special_handler() -> SpecialTokenHandler {
        let mut map = HashMap::new();
        map.insert(SpecialTokenKind::BOS, 100);
        map.insert(SpecialTokenKind::EOS, 101);
        map.insert(SpecialTokenKind::PAD, 102);
        map.insert(SpecialTokenKind::UNK, 103);
        map.insert(SpecialTokenKind::MASK, 104);
        SpecialTokenHandler::new(map)
    }

    // =====================================================================
    // TokenizerConfig
    // =====================================================================

    #[test]
    fn config_defaults() {
        let cfg = TokenizerConfig::new(ModelType::BPE, 32_000);
        assert_eq!(cfg.model_type, ModelType::BPE);
        assert_eq!(cfg.vocab_size, 32_000);
        assert!(!cfg.add_bos);
        assert!(!cfg.add_eos);
        assert!(cfg.pad_token_id.is_none());
    }

    #[test]
    fn config_with_bos_eos() {
        let mut cfg = TokenizerConfig::new(ModelType::WordPiece, 50_000);
        cfg.add_bos = true;
        cfg.add_eos = true;
        cfg.pad_token_id = Some(0);
        assert!(cfg.add_bos);
        assert!(cfg.add_eos);
        assert_eq!(cfg.pad_token_id, Some(0));
    }

    #[test]
    fn config_model_types() {
        assert_ne!(ModelType::BPE, ModelType::Unigram);
        assert_ne!(ModelType::Unigram, ModelType::WordPiece);
    }

    // =====================================================================
    // BPE Encoder
    // =====================================================================

    #[test]
    fn bpe_encode_hello() {
        let (vocab, merges, _) = tiny_bpe();
        let enc = BPEEncoder::new(vocab, merges);
        let ids = enc.encode("hello");
        assert_eq!(ids, vec![9]); // fully merged into single token
    }

    #[test]
    fn bpe_encode_empty() {
        let (vocab, merges, _) = tiny_bpe();
        let enc = BPEEncoder::new(vocab, merges);
        assert!(enc.encode("").is_empty());
    }

    #[test]
    fn bpe_encode_single_char() {
        let (vocab, merges, _) = tiny_bpe();
        let enc = BPEEncoder::new(vocab, merges);
        let ids = enc.encode("h");
        assert_eq!(ids, vec![1]);
    }

    #[test]
    fn bpe_encode_unknown_char() {
        let (vocab, merges, _) = tiny_bpe();
        let enc = BPEEncoder::new(vocab, merges);
        let ids = enc.encode("z");
        assert_eq!(ids, vec![0]); // unknown → 0
    }

    #[test]
    fn bpe_encode_partial_merge() {
        let (vocab, merges, _) = tiny_bpe();
        let enc = BPEEncoder::new(vocab, merges);
        let ids = enc.encode("he");
        assert_eq!(ids, vec![5]); // merged "he"
    }

    #[test]
    fn bpe_encode_repeated_chars() {
        let (vocab, merges, _) = tiny_bpe();
        let enc = BPEEncoder::new(vocab, merges);
        let ids = enc.encode("ll");
        assert_eq!(ids, vec![6]); // merged "ll"
    }

    // =====================================================================
    // BPE Decoder
    // =====================================================================

    #[test]
    fn bpe_decode_hello() {
        let (_, _, id_to_tok) = tiny_bpe();
        let dec = BPEDecoder::new(id_to_tok);
        assert_eq!(dec.decode(&[9]), "hello");
    }

    #[test]
    fn bpe_decode_empty() {
        let (_, _, id_to_tok) = tiny_bpe();
        let dec = BPEDecoder::new(id_to_tok);
        assert_eq!(dec.decode(&[]), "");
    }

    #[test]
    fn bpe_decode_unknown_id() {
        let (_, _, id_to_tok) = tiny_bpe();
        let dec = BPEDecoder::new(id_to_tok);
        assert_eq!(dec.decode(&[999]), "\u{FFFD}");
    }

    #[test]
    fn bpe_decode_sequence() {
        let (_, _, id_to_tok) = tiny_bpe();
        let dec = BPEDecoder::new(id_to_tok);
        // h + e + ll + o = "hello"
        assert_eq!(dec.decode(&[1, 2, 6, 4]), "hello");
    }

    #[test]
    fn bpe_roundtrip() {
        let (vocab, merges, id_to_tok) = tiny_bpe();
        let enc = BPEEncoder::new(vocab, merges);
        let dec = BPEDecoder::new(id_to_tok);
        let ids = enc.encode("hello");
        assert_eq!(dec.decode(&ids), "hello");
    }

    // =====================================================================
    // Incremental Detokenizer
    // =====================================================================

    #[test]
    fn incremental_simple() {
        let (_, _, id_to_tok) = tiny_bpe();
        let mut det = IncrementalDetokenizer::new(id_to_tok);
        assert_eq!(det.add_token(1), Some("h".into()));
        assert_eq!(det.add_token(2), Some("e".into()));
    }

    #[test]
    fn incremental_byte_token_ascii() {
        let mut map = HashMap::new();
        map.insert(1u32, "<0x41>".to_string()); // 'A'
        let mut det = IncrementalDetokenizer::new(map);
        assert_eq!(det.add_token(1), Some("A".into()));
    }

    #[test]
    fn incremental_byte_token_multibyte() {
        // UTF-8 for 'é' = 0xC3 0xA9
        let mut map = HashMap::new();
        map.insert(1u32, "<0xC3>".to_string());
        map.insert(2u32, "<0xA9>".to_string());
        let mut det = IncrementalDetokenizer::new(map);
        assert_eq!(det.add_token(1), None); // incomplete
        assert_eq!(det.buffered_bytes(), 1);
        assert_eq!(det.add_token(2), Some("é".into()));
        assert_eq!(det.buffered_bytes(), 0);
    }

    #[test]
    fn incremental_byte_token_3byte() {
        // UTF-8 for '€' = 0xE2 0x82 0xAC
        let mut map = HashMap::new();
        map.insert(1u32, "<0xE2>".to_string());
        map.insert(2u32, "<0x82>".to_string());
        map.insert(3u32, "<0xAC>".to_string());
        let mut det = IncrementalDetokenizer::new(map);
        assert_eq!(det.add_token(1), None);
        assert_eq!(det.add_token(2), None);
        assert_eq!(det.add_token(3), Some("€".into()));
    }

    #[test]
    fn incremental_flush_incomplete() {
        let mut map = HashMap::new();
        map.insert(1u32, "<0xC3>".to_string());
        let mut det = IncrementalDetokenizer::new(map);
        det.add_token(1);
        let flushed = det.flush();
        assert!(flushed.is_some());
        assert_eq!(det.buffered_bytes(), 0);
    }

    #[test]
    fn incremental_flush_empty() {
        let det_map: HashMap<u32, String> = HashMap::new();
        let mut det = IncrementalDetokenizer::new(det_map);
        assert_eq!(det.flush(), None);
    }

    #[test]
    fn incremental_mixed_byte_and_regular() {
        let mut map = HashMap::new();
        map.insert(1u32, "<0xC3>".to_string());
        map.insert(2u32, "<0xA9>".to_string());
        map.insert(3u32, "hello".to_string());
        let mut det = IncrementalDetokenizer::new(map);
        assert_eq!(det.add_token(1), None);
        // Regular token flushes buffered bytes first.
        let out = det.add_token(3).unwrap();
        assert!(out.contains("hello"));
    }

    #[test]
    fn incremental_unknown_token() {
        let map: HashMap<u32, String> = HashMap::new();
        let mut det = IncrementalDetokenizer::new(map);
        assert_eq!(det.add_token(999), Some("\u{FFFD}".into()));
    }

    // =====================================================================
    // Special Token Handler
    // =====================================================================

    #[test]
    fn special_id_lookup() {
        let handler = default_special_handler();
        assert_eq!(handler.id_of(SpecialTokenKind::BOS), Some(100));
        assert_eq!(handler.id_of(SpecialTokenKind::EOS), Some(101));
        assert_eq!(handler.id_of(SpecialTokenKind::PAD), Some(102));
    }

    #[test]
    fn special_add_bos_only() {
        let handler = default_special_handler();
        let mut cfg = TokenizerConfig::new(ModelType::BPE, 100);
        cfg.add_bos = true;
        let ids = handler.add_special(vec![1, 2, 3], &cfg);
        assert_eq!(ids, vec![100, 1, 2, 3]);
    }

    #[test]
    fn special_add_eos_only() {
        let handler = default_special_handler();
        let mut cfg = TokenizerConfig::new(ModelType::BPE, 100);
        cfg.add_eos = true;
        let ids = handler.add_special(vec![1, 2, 3], &cfg);
        assert_eq!(ids, vec![1, 2, 3, 101]);
    }

    #[test]
    fn special_add_bos_and_eos() {
        let handler = default_special_handler();
        let mut cfg = TokenizerConfig::new(ModelType::BPE, 100);
        cfg.add_bos = true;
        cfg.add_eos = true;
        let ids = handler.add_special(vec![1, 2, 3], &cfg);
        assert_eq!(ids, vec![100, 1, 2, 3, 101]);
    }

    #[test]
    fn special_add_none() {
        let handler = default_special_handler();
        let cfg = TokenizerConfig::new(ModelType::BPE, 100);
        let ids = handler.add_special(vec![1, 2, 3], &cfg);
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn special_strip_all() {
        let handler = default_special_handler();
        let ids = vec![100, 1, 2, 101, 3, 102];
        let stripped = handler.strip_special(&ids);
        assert_eq!(stripped, vec![1, 2, 3]);
    }

    #[test]
    fn special_strip_empty() {
        let handler = default_special_handler();
        assert!(handler.strip_special(&[]).is_empty());
    }

    #[test]
    fn special_strip_only_special() {
        let handler = default_special_handler();
        let stripped = handler.strip_special(&[100, 101, 102, 103, 104]);
        assert!(stripped.is_empty());
    }

    #[test]
    fn special_strip_no_special() {
        let handler = default_special_handler();
        let stripped = handler.strip_special(&[1, 2, 3]);
        assert_eq!(stripped, vec![1, 2, 3]);
    }

    // =====================================================================
    // Pre-Tokenizer
    // =====================================================================

    #[test]
    fn pretok_whitespace_basic() {
        let pt = PreTokenizer::new(PreTokenizerKind::Whitespace);
        let chunks = pt.pre_tokenize("hello world foo");
        assert_eq!(chunks, vec!["hello", "world", "foo"]);
    }

    #[test]
    fn pretok_whitespace_empty() {
        let pt = PreTokenizer::new(PreTokenizerKind::Whitespace);
        assert!(pt.pre_tokenize("").is_empty());
    }

    #[test]
    fn pretok_whitespace_only_spaces() {
        let pt = PreTokenizer::new(PreTokenizerKind::Whitespace);
        assert!(pt.pre_tokenize("   ").is_empty());
    }

    #[test]
    fn pretok_whitespace_tabs_newlines() {
        let pt = PreTokenizer::new(PreTokenizerKind::Whitespace);
        let chunks = pt.pre_tokenize("a\tb\nc");
        assert_eq!(chunks, vec!["a", "b", "c"]);
    }

    #[test]
    fn pretok_bytelevel_passthrough() {
        let pt = PreTokenizer::new(PreTokenizerKind::ByteLevel);
        let chunks = pt.pre_tokenize("hello world");
        assert_eq!(chunks, vec!["hello world"]);
    }

    #[test]
    fn pretok_bytelevel_empty() {
        let pt = PreTokenizer::new(PreTokenizerKind::ByteLevel);
        assert!(pt.pre_tokenize("").is_empty());
    }

    // =====================================================================
    // Post-Processor
    // =====================================================================

    #[test]
    fn postproc_identity() {
        let pp = PostProcessor::new(PostProcessorKind::Identity, HashMap::new());
        assert_eq!(pp.process(vec![1, 2, 3]), vec![1, 2, 3]);
    }

    #[test]
    fn postproc_template_cls_sep() {
        let mut lookup = HashMap::new();
        lookup.insert("[CLS]".to_string(), 50);
        lookup.insert("[SEP]".to_string(), 51);
        let pp =
            PostProcessor::new(PostProcessorKind::Template("[CLS] $A [SEP]".to_string()), lookup);
        assert_eq!(pp.process(vec![1, 2, 3]), vec![50, 1, 2, 3, 51]);
    }

    #[test]
    fn postproc_template_only_a() {
        let pp = PostProcessor::new(PostProcessorKind::Template("$A".to_string()), HashMap::new());
        assert_eq!(pp.process(vec![1, 2]), vec![1, 2]);
    }

    #[test]
    fn postproc_template_empty_input() {
        let mut lookup = HashMap::new();
        lookup.insert("[CLS]".to_string(), 50);
        let pp = PostProcessor::new(PostProcessorKind::Template("[CLS] $A".to_string()), lookup);
        assert_eq!(pp.process(vec![]), vec![50]);
    }

    #[test]
    fn postproc_template_unknown_marker() {
        let pp = PostProcessor::new(
            PostProcessorKind::Template("$A [UNKNOWN]".to_string()),
            HashMap::new(),
        );
        // [UNKNOWN] not in lookup → skipped
        assert_eq!(pp.process(vec![1]), vec![1]);
    }

    // =====================================================================
    // Normalizer Pipeline
    // =====================================================================

    #[test]
    fn normalizer_lowercase() {
        let n = NormalizerPipeline::new(vec![NormalizerOp::Lowercase]);
        assert_eq!(n.normalize("HELLO World"), "hello world");
    }

    #[test]
    fn normalizer_strip_whitespace() {
        let n = NormalizerPipeline::new(vec![NormalizerOp::StripWhitespace]);
        assert_eq!(n.normalize("  hello  "), "hello");
    }

    #[test]
    fn normalizer_combined() {
        let n =
            NormalizerPipeline::new(vec![NormalizerOp::StripWhitespace, NormalizerOp::Lowercase]);
        assert_eq!(n.normalize("  HELLO  "), "hello");
    }

    #[test]
    fn normalizer_empty() {
        let n = NormalizerPipeline::new(vec![]);
        assert_eq!(n.normalize("Hello"), "Hello");
    }

    #[test]
    fn normalizer_nfc_passthrough() {
        let n = NormalizerPipeline::new(vec![NormalizerOp::NFC]);
        assert_eq!(n.normalize("café"), "café");
    }

    #[test]
    fn normalizer_nfd_passthrough() {
        let n = NormalizerPipeline::new(vec![NormalizerOp::NFD]);
        assert_eq!(n.normalize("naïve"), "naïve");
    }

    #[test]
    fn normalizer_nfkc_passthrough() {
        let n = NormalizerPipeline::new(vec![NormalizerOp::NFKC]);
        assert_eq!(n.normalize("ﬁ"), "ﬁ");
    }

    // =====================================================================
    // Full Pipeline
    // =====================================================================

    fn build_pipeline(bos: bool, eos: bool) -> TokenizerPipeline {
        let (vocab, merges, _) = tiny_bpe();
        let mut cfg = TokenizerConfig::new(ModelType::BPE, 200);
        cfg.add_bos = bos;
        cfg.add_eos = eos;
        TokenizerPipeline {
            config: cfg,
            normalizer: Some(NormalizerPipeline::new(vec![NormalizerOp::Lowercase])),
            pre_tokenizer: PreTokenizer::new(PreTokenizerKind::ByteLevel),
            encoder: BPEEncoder::new(vocab, merges),
            post_processor: PostProcessor::new(PostProcessorKind::Identity, HashMap::new()),
            special_handler: default_special_handler(),
        }
    }

    #[test]
    fn pipeline_encode_basic() {
        let pipe = build_pipeline(false, false);
        let ids = pipe.encode("hello");
        assert_eq!(ids, vec![9]);
    }

    #[test]
    fn pipeline_encode_with_bos_eos() {
        let pipe = build_pipeline(true, true);
        let ids = pipe.encode("hello");
        assert_eq!(ids, vec![100, 9, 101]);
    }

    #[test]
    fn pipeline_encode_normalizes() {
        let pipe = build_pipeline(false, false);
        // Uppercase input → lowercased by normalizer → encodes as "hello"
        let ids = pipe.encode("HELLO");
        assert_eq!(ids, vec![9]);
    }

    #[test]
    fn pipeline_encode_empty() {
        let pipe = build_pipeline(false, false);
        assert!(pipe.encode("").is_empty());
    }

    #[test]
    fn pipeline_encode_empty_with_bos_eos() {
        let pipe = build_pipeline(true, true);
        let ids = pipe.encode("");
        assert_eq!(ids, vec![100, 101]); // only special tokens
    }

    #[test]
    fn pipeline_with_template_postprocessor() {
        let (vocab, merges, _) = tiny_bpe();
        let mut lookup = HashMap::new();
        lookup.insert("[CLS]".to_string(), 50);
        lookup.insert("[SEP]".to_string(), 51);
        let pipe = TokenizerPipeline {
            config: TokenizerConfig::new(ModelType::BPE, 200),
            normalizer: None,
            pre_tokenizer: PreTokenizer::new(PreTokenizerKind::ByteLevel),
            encoder: BPEEncoder::new(vocab, merges),
            post_processor: PostProcessor::new(
                PostProcessorKind::Template("[CLS] $A [SEP]".to_string()),
                lookup,
            ),
            special_handler: default_special_handler(),
        };
        let ids = pipe.encode("hello");
        assert_eq!(ids, vec![50, 9, 51]);
    }

    // =====================================================================
    // Detokenizer Metrics
    // =====================================================================

    #[test]
    fn metrics_default() {
        let m = DetokenizerMetrics::new();
        assert_eq!(m.tokens_decoded, 0);
        assert_eq!(m.partial_bytes_buffered, 0);
        assert_eq!(m.special_tokens_stripped, 0);
    }

    #[test]
    fn metrics_record() {
        let mut m = DetokenizerMetrics::new();
        m.record_decode(10);
        m.record_partial_bytes(3);
        m.record_special_stripped(2);
        assert_eq!(m.tokens_decoded, 10);
        assert_eq!(m.partial_bytes_buffered, 3);
        assert_eq!(m.special_tokens_stripped, 2);
    }

    #[test]
    fn metrics_accumulate() {
        let mut m = DetokenizerMetrics::new();
        m.record_decode(5);
        m.record_decode(7);
        assert_eq!(m.tokens_decoded, 12);
    }

    // =====================================================================
    // Streaming / Buffering
    // =====================================================================

    #[test]
    fn streaming_sequence() {
        let (_, _, id_to_tok) = tiny_bpe();
        let mut det = IncrementalDetokenizer::new(id_to_tok);
        let mut out = String::new();
        for id in [1, 2, 6, 4] {
            if let Some(s) = det.add_token(id) {
                out.push_str(&s);
            }
        }
        assert_eq!(out, "hello");
    }

    #[test]
    fn streaming_4byte_utf8() {
        // UTF-8 for '𝄞' (U+1D11E) = 0xF0 0x9D 0x84 0x9E
        let mut map = HashMap::new();
        map.insert(1u32, "<0xF0>".to_string());
        map.insert(2u32, "<0x9D>".to_string());
        map.insert(3u32, "<0x84>".to_string());
        map.insert(4u32, "<0x9E>".to_string());
        let mut det = IncrementalDetokenizer::new(map);
        assert_eq!(det.add_token(1), None);
        assert_eq!(det.add_token(2), None);
        assert_eq!(det.add_token(3), None);
        assert_eq!(det.add_token(4), Some("𝄞".into()));
    }

    // =====================================================================
    // Edge cases
    // =====================================================================

    #[test]
    fn edge_single_char_roundtrip() {
        let (vocab, merges, id_to_tok) = tiny_bpe();
        let enc = BPEEncoder::new(vocab, merges);
        let dec = BPEDecoder::new(id_to_tok);
        for ch in ['h', 'e', 'l', 'o'] {
            let s = ch.to_string();
            let ids = enc.encode(&s);
            assert_eq!(dec.decode(&ids), s);
        }
    }

    #[test]
    fn edge_all_special_strip() {
        let handler = default_special_handler();
        let all = vec![100, 101, 102, 103, 104];
        assert!(handler.strip_special(&all).is_empty());
    }

    #[test]
    fn edge_parse_byte_token() {
        assert_eq!(parse_byte_token("<0x41>"), Some(0x41));
        assert_eq!(parse_byte_token("<0xFF>"), Some(0xFF));
        assert_eq!(parse_byte_token("<0x00>"), Some(0x00));
        assert_eq!(parse_byte_token("hello"), None);
        assert_eq!(parse_byte_token("<0xGG>"), None);
    }

    #[test]
    fn edge_empty_normalizer_pipeline() {
        let n = NormalizerPipeline::new(vec![]);
        assert_eq!(n.normalize(""), "");
    }

    #[test]
    fn edge_whitespace_pretok_single_word() {
        let pt = PreTokenizer::new(PreTokenizerKind::Whitespace);
        assert_eq!(pt.pre_tokenize("hello"), vec!["hello"]);
    }

    #[test]
    fn edge_identity_postproc_empty() {
        let pp = PostProcessor::new(PostProcessorKind::Identity, HashMap::new());
        assert!(pp.process(vec![]).is_empty());
    }

    #[test]
    fn edge_decoder_single_token() {
        let (_, _, id_to_tok) = tiny_bpe();
        let dec = BPEDecoder::new(id_to_tok);
        assert_eq!(dec.decode(&[9]), "hello");
    }

    #[test]
    fn edge_config_unigram() {
        let cfg = TokenizerConfig::new(ModelType::Unigram, 128_000);
        assert_eq!(cfg.model_type, ModelType::Unigram);
        assert_eq!(cfg.vocab_size, 128_000);
    }

    #[test]
    fn edge_special_handler_empty() {
        let handler = SpecialTokenHandler::new(HashMap::new());
        assert_eq!(handler.id_of(SpecialTokenKind::BOS), None);
        let cfg = TokenizerConfig::new(ModelType::BPE, 100);
        assert_eq!(handler.add_special(vec![1], &cfg), vec![1]);
    }

    #[test]
    fn edge_multiple_normalizer_ops() {
        let n = NormalizerPipeline::new(vec![
            NormalizerOp::NFC,
            NormalizerOp::Lowercase,
            NormalizerOp::StripWhitespace,
            NormalizerOp::NFD,
        ]);
        assert_eq!(n.normalize("  FOO  "), "foo");
    }

    #[test]
    fn edge_metrics_equality() {
        let a = DetokenizerMetrics::new();
        let b = DetokenizerMetrics::default();
        assert_eq!(a, b);
    }

    #[test]
    fn edge_incremental_consecutive_regular_tokens() {
        let mut map = HashMap::new();
        map.insert(1u32, "ab".to_string());
        map.insert(2u32, "cd".to_string());
        let mut det = IncrementalDetokenizer::new(map);
        assert_eq!(det.add_token(1), Some("ab".into()));
        assert_eq!(det.add_token(2), Some("cd".into()));
    }

    #[test]
    fn edge_bpe_encoder_no_merges() {
        let mut vocab = HashMap::new();
        vocab.insert("a".to_string(), 1);
        vocab.insert("b".to_string(), 2);
        let enc = BPEEncoder::new(vocab, vec![]);
        assert_eq!(enc.encode("ab"), vec![1, 2]);
    }

    #[test]
    fn edge_postproc_template_double_a() {
        let pp =
            PostProcessor::new(PostProcessorKind::Template("$A $A".to_string()), HashMap::new());
        assert_eq!(pp.process(vec![1, 2]), vec![1, 2, 1, 2]);
    }

    #[test]
    fn edge_incremental_byte_then_flush() {
        // Feed partial 2-byte sequence, then flush (lossy).
        let mut map = HashMap::new();
        map.insert(1u32, "<0xC3>".to_string());
        let mut det = IncrementalDetokenizer::new(map);
        det.add_token(1);
        assert_eq!(det.buffered_bytes(), 1);
        let flushed = det.flush().unwrap();
        // Lossy decode of 0xC3 alone → replacement char
        assert!(flushed.contains('\u{FFFD}'));
        assert_eq!(det.buffered_bytes(), 0);
    }

    #[test]
    fn edge_pipeline_whitespace_pretok_multiple_words() {
        let (vocab, merges, _) = tiny_bpe();
        let pipe = TokenizerPipeline {
            config: TokenizerConfig::new(ModelType::BPE, 200),
            normalizer: None,
            pre_tokenizer: PreTokenizer::new(PreTokenizerKind::Whitespace),
            encoder: BPEEncoder::new(vocab, merges),
            post_processor: PostProcessor::new(PostProcessorKind::Identity, HashMap::new()),
            special_handler: default_special_handler(),
        };
        // "hello hello" → split into two "hello" words
        let ids = pipe.encode("hello hello");
        assert_eq!(ids, vec![9, 9]);
    }
}
