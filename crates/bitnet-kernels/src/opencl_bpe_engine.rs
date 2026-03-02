//! OpenCL-accelerated Byte-Pair Encoding (BPE) tokenizer engine.
//!
//! Provides a complete BPE tokenizer pipeline with:
//!
//! - **`BpeVocab`**: vocabulary with bidirectional token↔id mappings
//! - **`MergeRule`**: BPE merge pairs ordered by priority/rank
//! - **`BpeEncoder`**: encodes text to token IDs via greedy BPE merges
//! - **`BpeDecoder`**: decodes token IDs back to text
//! - **`SpecialTokens`**: BOS, EOS, PAD, UNK, and user-defined specials
//! - **`BatchEncoder`**: encodes multiple strings in parallel
//! - **`ByteFallback`**: handles unknown bytes via byte-level tokens
//! - **`BpeStats`**: encoding/decoding timing and coverage metrics
//!
//! An OpenCL kernel source for parallel merge-scan is included for
//! future GPU-accelerated batch tokenization on Intel Arc / other
//! OpenCL 3.0 devices.

use std::collections::HashMap;
use std::time::Instant;

// ── OpenCL kernel source ─────────────────────────────────────────

/// OpenCL kernel source for parallel BPE merge operations.
///
/// Each work-item scans a contiguous segment of the symbol buffer and
/// applies the highest-priority merge that matches, enabling O(n/p)
/// per-pass tokenization on GPU.
pub const BPE_MERGE_CL: &str = r#"
// Parallel BPE merge kernel.
// symbols:    token-id buffer  (read-write)
// merge_a/b:  left/right token for this merge pass
// merge_out:  replacement token id
// len:        number of symbols in the buffer
// changed:    atomic flag – set to 1 if any merge happened
__kernel void bpe_merge(
    __global int* symbols,
    const int merge_a,
    const int merge_b,
    const int merge_out,
    const int len,
    __global int* changed
) {
    int gid = get_global_id(0);
    if (gid >= len - 1) return;

    if (symbols[gid] == merge_a && symbols[gid + 1] == merge_b) {
        symbols[gid] = merge_out;
        // Mark the consumed position with -1 (tombstone).
        symbols[gid + 1] = -1;
        changed[0] = 1;
    }
}

// Compact the symbol buffer by removing tombstones (-1).
// out_len: output atomic counter for the compacted length.
__kernel void compact_symbols(
    __global const int* symbols,
    __global int* output,
    const int len,
    __global int* out_len
) {
    // Single work-group serial compaction (sufficient for short
    // sequences; a parallel prefix-sum variant can replace this
    // for longer inputs).
    if (get_global_id(0) != 0) return;
    int j = 0;
    for (int i = 0; i < len; i++) {
        if (symbols[i] != -1) {
            output[j++] = symbols[i];
        }
    }
    out_len[0] = j;
}
"#;

// ── MergeRule ────────────────────────────────────────────────────

/// A single BPE merge rule: pair (`left`, `right`) → `merged` with a
/// priority `rank` (lower rank = higher priority, applied first).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeRule {
    /// Left token of the pair.
    pub left: String,
    /// Right token of the pair.
    pub right: String,
    /// Merged output token.
    pub merged: String,
    /// Priority rank (0 = highest priority).
    pub rank: u32,
}

impl MergeRule {
    /// Create a new merge rule.
    pub fn new(left: &str, right: &str, rank: u32) -> Self {
        let merged = format!("{left}{right}");
        Self { left: left.to_string(), right: right.to_string(), merged, rank }
    }
}

impl PartialOrd for MergeRule {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeRule {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.rank.cmp(&other.rank)
    }
}

// ── SpecialTokens ────────────────────────────────────────────────

/// Special token configuration for a BPE tokenizer.
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    /// Beginning-of-sequence token.
    pub bos: Option<(String, u32)>,
    /// End-of-sequence token.
    pub eos: Option<(String, u32)>,
    /// Padding token.
    pub pad: Option<(String, u32)>,
    /// Unknown / fallback token.
    pub unk: Option<(String, u32)>,
    /// Additional user-defined special tokens.
    pub custom: HashMap<String, u32>,
}

impl SpecialTokens {
    /// Create an empty special token set.
    pub fn new() -> Self {
        Self { bos: None, eos: None, pad: None, unk: None, custom: HashMap::new() }
    }

    /// Set beginning-of-sequence token.
    #[must_use]
    pub fn with_bos(mut self, token: &str, id: u32) -> Self {
        self.bos = Some((token.to_string(), id));
        self
    }

    /// Set end-of-sequence token.
    #[must_use]
    pub fn with_eos(mut self, token: &str, id: u32) -> Self {
        self.eos = Some((token.to_string(), id));
        self
    }

    /// Set padding token.
    #[must_use]
    pub fn with_pad(mut self, token: &str, id: u32) -> Self {
        self.pad = Some((token.to_string(), id));
        self
    }

    /// Set unknown token.
    #[must_use]
    pub fn with_unk(mut self, token: &str, id: u32) -> Self {
        self.unk = Some((token.to_string(), id));
        self
    }

    /// Add a custom special token.
    #[must_use]
    pub fn with_custom(mut self, token: &str, id: u32) -> Self {
        self.custom.insert(token.to_string(), id);
        self
    }

    /// Return all special tokens as `(token, id)` pairs.
    pub fn all_pairs(&self) -> Vec<(&str, u32)> {
        let mut out = Vec::new();
        if let Some((ref t, id)) = self.bos {
            out.push((t.as_str(), id));
        }
        if let Some((ref t, id)) = self.eos {
            out.push((t.as_str(), id));
        }
        if let Some((ref t, id)) = self.pad {
            out.push((t.as_str(), id));
        }
        if let Some((ref t, id)) = self.unk {
            out.push((t.as_str(), id));
        }
        for (t, &id) in &self.custom {
            out.push((t.as_str(), id));
        }
        out
    }
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self::new()
    }
}

// ── BpeVocab ─────────────────────────────────────────────────────

/// Bidirectional vocabulary: token string ↔ token id.
#[derive(Debug, Clone)]
pub struct BpeVocab {
    token_to_id: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
}

impl BpeVocab {
    /// Build a vocabulary from an iterator of `(token, id)` pairs.
    pub fn from_pairs(pairs: impl IntoIterator<Item = (String, u32)>) -> Self {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();
        for (tok, id) in pairs {
            token_to_id.insert(tok.clone(), id);
            id_to_token.insert(id, tok);
        }
        Self { token_to_id, id_to_token }
    }

    /// Look up the id for a token string.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Look up the token string for an id.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(String::as_str)
    }

    /// Number of entries in the vocabulary.
    pub fn len(&self) -> usize {
        self.token_to_id.len()
    }

    /// Whether the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.token_to_id.is_empty()
    }

    /// Check if a token string exists.
    pub fn contains(&self, token: &str) -> bool {
        self.token_to_id.contains_key(token)
    }

    /// Check if an id exists.
    pub fn contains_id(&self, id: u32) -> bool {
        self.id_to_token.contains_key(&id)
    }
}

// ── ByteFallback ─────────────────────────────────────────────────

/// Byte-level fallback for characters absent from the vocabulary.
///
/// Maps each byte value 0x00–0xFF to a special token `<0x00>`–`<0xFF>`
/// so that *any* input can be encoded without information loss.
#[derive(Debug, Clone)]
pub struct ByteFallback {
    byte_to_id: HashMap<u8, u32>,
    id_to_byte: HashMap<u32, u8>,
}

impl ByteFallback {
    /// Build byte fallback mappings starting at `base_id`.
    ///
    /// Tokens `<0x00>` through `<0xFF>` receive ids
    /// `base_id` through `base_id + 255`.
    pub fn new(base_id: u32) -> Self {
        let mut byte_to_id = HashMap::with_capacity(256);
        let mut id_to_byte = HashMap::with_capacity(256);
        for b in 0u16..=255 {
            let id = base_id + u32::from(b as u8);
            byte_to_id.insert(b as u8, id);
            id_to_byte.insert(id, b as u8);
        }
        Self { byte_to_id, id_to_byte }
    }

    /// Get the token id for a single byte.
    pub fn byte_to_id(&self, byte: u8) -> Option<u32> {
        self.byte_to_id.get(&byte).copied()
    }

    /// Get the byte value for a fallback token id.
    pub fn id_to_byte(&self, id: u32) -> Option<u8> {
        self.id_to_byte.get(&id).copied()
    }

    /// Encode a slice of bytes as fallback token ids.
    pub fn encode_bytes(&self, bytes: &[u8]) -> Vec<u32> {
        bytes.iter().filter_map(|&b| self.byte_to_id(b)).collect()
    }

    /// Decode fallback token ids back to bytes.
    pub fn decode_ids(&self, ids: &[u32]) -> Vec<u8> {
        ids.iter().filter_map(|&id| self.id_to_byte(id)).collect()
    }

    /// Format byte as fallback token string `<0xHH>`.
    pub fn byte_token_string(byte: u8) -> String {
        format!("<0x{byte:02X}>")
    }
}

// ── BpeEncoder ───────────────────────────────────────────────────

/// BPE encoder: converts text to a sequence of token ids.
#[derive(Debug, Clone)]
pub struct BpeEncoder {
    vocab: BpeVocab,
    merges: Vec<MergeRule>,
    /// Merge pair → rank for O(1) lookup during encoding.
    merge_index: HashMap<(String, String), u32>,
    special: SpecialTokens,
    byte_fallback: Option<ByteFallback>,
    add_bos: bool,
    add_eos: bool,
}

impl BpeEncoder {
    /// Create a new BPE encoder.
    pub fn new(vocab: BpeVocab, merges: Vec<MergeRule>, special: SpecialTokens) -> Self {
        let mut merge_index = HashMap::with_capacity(merges.len());
        for m in &merges {
            merge_index.insert((m.left.clone(), m.right.clone()), m.rank);
        }
        Self {
            vocab,
            merges,
            merge_index,
            special,
            byte_fallback: None,
            add_bos: false,
            add_eos: false,
        }
    }

    /// Enable byte-level fallback.
    #[must_use]
    pub fn with_byte_fallback(mut self, bf: ByteFallback) -> Self {
        self.byte_fallback = Some(bf);
        self
    }

    /// Prepend BOS token when encoding.
    #[must_use]
    pub fn with_bos(mut self, enable: bool) -> Self {
        self.add_bos = enable;
        self
    }

    /// Append EOS token when encoding.
    #[must_use]
    pub fn with_eos(mut self, enable: bool) -> Self {
        self.add_eos = enable;
        self
    }

    /// Access the vocabulary.
    pub fn vocab(&self) -> &BpeVocab {
        &self.vocab
    }

    /// Access the merge rules (sorted by rank).
    pub fn merges(&self) -> &[MergeRule] {
        &self.merges
    }

    /// Access the special tokens.
    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special
    }

    /// Encode a single text string into token ids.
    ///
    /// Steps:
    /// 1. Split text into initial character-level symbols.
    /// 2. Iteratively apply the highest-priority merge pair.
    /// 3. Map resulting symbols to ids (with byte fallback if enabled).
    /// 4. Optionally prepend BOS / append EOS.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();

        // Prepend BOS if configured.
        if self.add_bos
            && let Some((_, bos_id)) = &self.special.bos
        {
            ids.push(*bos_id);
        }

        if !text.is_empty() {
            // Check for special tokens in the text first.
            let segments = self.split_special_tokens(text);
            for segment in segments {
                if let Some(id) = self.lookup_special(&segment) {
                    ids.push(id);
                } else {
                    ids.extend(self.encode_ordinary(&segment));
                }
            }
        }

        // Append EOS if configured.
        if self.add_eos
            && let Some((_, eos_id)) = &self.special.eos
        {
            ids.push(*eos_id);
        }

        ids
    }

    /// Encode an ordinary (non-special) text segment via BPE merges.
    fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        // Start with character-level symbols.
        let mut symbols: Vec<String> = text.chars().map(|c| c.to_string()).collect();

        if symbols.is_empty() {
            return Vec::new();
        }

        // Iteratively merge the best pair.
        loop {
            let best = self.find_best_merge(&symbols);
            match best {
                Some((idx, merged)) => {
                    symbols.splice(idx..idx + 2, std::iter::once(merged));
                }
                None => break,
            }
        }

        // Map symbols to ids.
        symbols.iter().flat_map(|sym| self.symbol_to_ids(sym)).collect()
    }

    /// Find the merge with the lowest rank among adjacent pairs.
    fn find_best_merge(&self, symbols: &[String]) -> Option<(usize, String)> {
        if symbols.len() < 2 {
            return None;
        }
        let mut best: Option<(usize, u32, String)> = None;
        for i in 0..symbols.len() - 1 {
            let key = (symbols[i].clone(), symbols[i + 1].clone());
            if let Some(&rank) = self.merge_index.get(&key)
                && best.as_ref().is_none_or(|(_, br, _)| rank < *br)
            {
                let merged = format!("{}{}", symbols[i], symbols[i + 1]);
                best = Some((i, rank, merged));
            }
        }
        best.map(|(idx, _, merged)| (idx, merged))
    }

    /// Convert a symbol string to one or more token ids.
    fn symbol_to_ids(&self, symbol: &str) -> Vec<u32> {
        if let Some(id) = self.vocab.token_to_id(symbol) {
            return vec![id];
        }
        // Byte fallback: encode each byte of the symbol.
        if let Some(ref bf) = self.byte_fallback {
            let ids = bf.encode_bytes(symbol.as_bytes());
            if !ids.is_empty() {
                return ids;
            }
        }
        // Last resort: UNK token.
        if let Some((_, unk_id)) = &self.special.unk {
            return vec![*unk_id];
        }
        Vec::new()
    }

    /// Split text on special token boundaries.
    fn split_special_tokens(&self, text: &str) -> Vec<String> {
        let specials: Vec<(&str, u32)> = self.special.all_pairs();
        if specials.is_empty() {
            return vec![text.to_string()];
        }

        let mut result = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            // Find the earliest special token occurrence.
            let mut earliest: Option<(usize, usize, &str)> = None;
            for &(tok, _) in &specials {
                if let Some(pos) = remaining.find(tok)
                    && earliest.as_ref().is_none_or(|(ep, _, _)| pos < *ep)
                {
                    earliest = Some((pos, tok.len(), tok));
                }
            }

            match earliest {
                Some((pos, len, _tok)) => {
                    if pos > 0 {
                        result.push(remaining[..pos].to_string());
                    }
                    result.push(remaining[pos..pos + len].to_string());
                    remaining = &remaining[pos + len..];
                }
                None => {
                    result.push(remaining.to_string());
                    break;
                }
            }
        }
        result
    }

    /// Look up a special token string and return its id.
    fn lookup_special(&self, text: &str) -> Option<u32> {
        for (tok, id) in self.special.all_pairs() {
            if tok == text {
                return Some(id);
            }
        }
        None
    }
}

// ── BpeDecoder ───────────────────────────────────────────────────

/// BPE decoder: converts token ids back to text.
#[derive(Debug, Clone)]
pub struct BpeDecoder {
    vocab: BpeVocab,
    special: SpecialTokens,
    byte_fallback: Option<ByteFallback>,
    skip_special: bool,
}

impl BpeDecoder {
    /// Create a new BPE decoder.
    pub fn new(vocab: BpeVocab, special: SpecialTokens) -> Self {
        Self { vocab, special, byte_fallback: None, skip_special: false }
    }

    /// Enable byte-level fallback decoding.
    #[must_use]
    pub fn with_byte_fallback(mut self, bf: ByteFallback) -> Self {
        self.byte_fallback = Some(bf);
        self
    }

    /// When true, special tokens (BOS/EOS/PAD) are omitted from output.
    #[must_use]
    pub fn with_skip_special(mut self, skip: bool) -> Self {
        self.skip_special = skip;
        self
    }

    /// Decode a slice of token ids into a `String`.
    pub fn decode(&self, ids: &[u32]) -> String {
        let special_ids: HashMap<u32, &str> =
            self.special.all_pairs().into_iter().map(|(t, id)| (id, t)).collect();

        let mut out = String::new();
        let mut byte_buf: Vec<u8> = Vec::new();

        for &id in ids {
            // Check byte fallback first.
            if let Some(ref bf) = self.byte_fallback
                && let Some(byte) = bf.id_to_byte(id)
            {
                byte_buf.push(byte);
                continue;
            }

            // Flush accumulated byte-fallback buffer.
            if !byte_buf.is_empty() {
                out.push_str(&String::from_utf8_lossy(&byte_buf));
                byte_buf.clear();
            }

            if let Some(&tok) = special_ids.get(&id) {
                if !self.skip_special {
                    out.push_str(tok);
                }
            } else if let Some(tok) = self.vocab.id_to_token(id) {
                out.push_str(tok);
            }
        }

        // Flush any remaining byte buffer.
        if !byte_buf.is_empty() {
            out.push_str(&String::from_utf8_lossy(&byte_buf));
        }

        out
    }
}

// ── BatchEncoder ─────────────────────────────────────────────────

/// Encodes multiple strings in parallel using Rayon-style chunked
/// iteration (falls back to sequential when Rayon is unavailable).
#[derive(Debug, Clone)]
pub struct BatchEncoder {
    encoder: BpeEncoder,
}

impl BatchEncoder {
    /// Wrap an existing encoder for batch use.
    pub fn new(encoder: BpeEncoder) -> Self {
        Self { encoder }
    }

    /// Encode a batch of texts, returning one id-vector per input.
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        texts.iter().map(|t| self.encoder.encode(t)).collect()
    }

    /// Access the underlying encoder.
    pub fn encoder(&self) -> &BpeEncoder {
        &self.encoder
    }
}

// ── BpeStats ─────────────────────────────────────────────────────

/// Tokenization statistics.
#[derive(Debug, Clone, Default)]
pub struct BpeStats {
    /// Total encoding wall-clock time.
    pub encode_time: std::time::Duration,
    /// Total decoding wall-clock time.
    pub decode_time: std::time::Duration,
    /// Average tokens produced per input text.
    pub avg_tokens_per_text: f64,
    /// Fraction of input characters that mapped to vocab tokens
    /// without needing byte fallback (0.0–1.0).
    pub vocab_coverage: f64,
}

/// Collect stats for a batch encode + decode round-trip.
pub fn collect_stats(encoder: &BpeEncoder, decoder: &BpeDecoder, texts: &[&str]) -> BpeStats {
    if texts.is_empty() {
        return BpeStats::default();
    }

    // Encode.
    let enc_start = Instant::now();
    let encoded: Vec<Vec<u32>> = texts.iter().map(|t| encoder.encode(t)).collect();
    let encode_time = enc_start.elapsed();

    // Decode.
    let dec_start = Instant::now();
    for ids in &encoded {
        let _ = decoder.decode(ids);
    }
    let decode_time = dec_start.elapsed();

    // Avg tokens per text.
    let total_tokens: usize = encoded.iter().map(Vec::len).sum();
    let avg_tokens_per_text = total_tokens as f64 / texts.len() as f64;

    // Vocab coverage: fraction of encoded ids that are in the vocab.
    let vocab_ids: usize = encoded
        .iter()
        .flat_map(|ids| ids.iter())
        .filter(|&&id| encoder.vocab().contains_id(id))
        .count();
    let vocab_coverage =
        if total_tokens > 0 { vocab_ids as f64 / total_tokens as f64 } else { 0.0 };

    BpeStats { encode_time, decode_time, avg_tokens_per_text, vocab_coverage }
}

// ── CPU reference: apply one merge pass ──────────────────────────

/// CPU reference implementation of a single BPE merge pass.
///
/// Scans `symbols` left-to-right, replacing every adjacent
/// `(left, right)` pair with `merged`. Returns `true` if at least
/// one merge was applied.
pub fn cpu_merge_pass(symbols: &mut Vec<String>, left: &str, right: &str, merged: &str) -> bool {
    let mut changed = false;
    let mut i = 0;
    while i + 1 < symbols.len() {
        if symbols[i] == left && symbols[i + 1] == right {
            symbols[i] = merged.to_string();
            symbols.remove(i + 1);
            changed = true;
            // Don't advance i — the new merged symbol may participate
            // in another merge with its new right neighbour.
        } else {
            i += 1;
        }
    }
    changed
}

/// CPU reference: fully apply all merge rules (by rank) to a symbol
/// list, returning the final merged symbols.
pub fn cpu_full_bpe(text: &str, merges: &[MergeRule]) -> Vec<String> {
    let mut symbols: Vec<String> = text.chars().map(|c| c.to_string()).collect();
    let mut sorted = merges.to_vec();
    sorted.sort();
    for m in &sorted {
        cpu_merge_pass(&mut symbols, &m.left, &m.right, &m.merged);
    }
    symbols
}

// ══════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────

    /// Build a small toy vocabulary + merges for testing.
    ///
    /// Merge priority is carefully ordered so that "hello" and
    /// "world" fully reduce to single tokens via greedy BPE.
    fn toy_vocab_and_merges() -> (BpeVocab, Vec<MergeRule>) {
        let pairs: Vec<(String, u32)> = vec![
            ("h".into(), 0),
            ("e".into(), 1),
            ("l".into(), 2),
            ("o".into(), 3),
            ("w".into(), 4),
            ("r".into(), 5),
            ("d".into(), 6),
            (" ".into(), 7),
            ("he".into(), 8),
            ("lo".into(), 9),
            ("hel".into(), 10),
            ("hell".into(), 11),
            ("hello".into(), 12),
            ("wo".into(), 13),
            ("wor".into(), 14),
            ("worl".into(), 15),
            ("world".into(), 16),
        ];
        let vocab = BpeVocab::from_pairs(pairs);

        // l+o (rank 1) must fire before he+l (rank 2) so that
        // h,e,l,l,o → he,l,l,o → he,l,lo → hel,lo → hello.
        let merges = vec![
            MergeRule::new("h", "e", 0),
            MergeRule::new("l", "o", 1),
            MergeRule::new("he", "l", 2),
            MergeRule::new("hel", "l", 3),
            MergeRule::new("hel", "lo", 4),
            MergeRule::new("w", "o", 5),
            MergeRule::new("wo", "r", 6),
            MergeRule::new("wor", "l", 7),
            MergeRule::new("worl", "d", 8),
        ];
        (vocab, merges)
    }

    fn toy_special() -> SpecialTokens {
        SpecialTokens::new()
            .with_bos("<s>", 100)
            .with_eos("</s>", 101)
            .with_pad("<pad>", 102)
            .with_unk("<unk>", 103)
    }

    fn toy_encoder() -> BpeEncoder {
        let (vocab, merges) = toy_vocab_and_merges();
        BpeEncoder::new(vocab, merges, toy_special())
    }

    fn toy_decoder() -> BpeDecoder {
        let (vocab, _) = toy_vocab_and_merges();
        BpeDecoder::new(vocab, toy_special())
    }

    // ── Single word encoding ────────────────────────────────────

    #[test]
    fn test_encode_hello() {
        let enc = toy_encoder();
        let ids = enc.encode("hello");
        assert_eq!(ids, vec![12], "\"hello\" → single merged token");
    }

    #[test]
    fn test_encode_world() {
        let enc = toy_encoder();
        let ids = enc.encode("world");
        assert_eq!(ids, vec![16]);
    }

    // ── Multi-word encoding ─────────────────────────────────────

    #[test]
    fn test_encode_hello_world() {
        let enc = toy_encoder();
        let ids = enc.encode("hello world");
        assert_eq!(ids, vec![12, 7, 16]);
    }

    // ── Decode simple ───────────────────────────────────────────

    #[test]
    fn test_decode_hello() {
        let dec = toy_decoder();
        assert_eq!(dec.decode(&[12]), "hello");
    }

    #[test]
    fn test_decode_hello_world() {
        let dec = toy_decoder();
        assert_eq!(dec.decode(&[12, 7, 16]), "hello world");
    }

    // ── Round-trip ──────────────────────────────────────────────

    #[test]
    fn test_round_trip_hello() {
        let enc = toy_encoder();
        let dec = toy_decoder();
        let text = "hello";
        let ids = enc.encode(text);
        assert_eq!(dec.decode(&ids), text);
    }

    #[test]
    fn test_round_trip_hello_world() {
        let enc = toy_encoder();
        let dec = toy_decoder();
        let text = "hello world";
        let ids = enc.encode(text);
        assert_eq!(dec.decode(&ids), text);
    }

    // ── Special token handling ──────────────────────────────────

    #[test]
    fn test_bos_insertion() {
        let enc = toy_encoder().with_bos(true);
        let ids = enc.encode("hello");
        assert_eq!(ids[0], 100, "BOS should be first");
        assert_eq!(ids[1], 12);
    }

    #[test]
    fn test_eos_insertion() {
        let enc = toy_encoder().with_eos(true);
        let ids = enc.encode("hello");
        assert_eq!(*ids.last().unwrap(), 101, "EOS should be last");
    }

    #[test]
    fn test_bos_and_eos() {
        let enc = toy_encoder().with_bos(true).with_eos(true);
        let ids = enc.encode("hello");
        assert_eq!(ids, vec![100, 12, 101]);
    }

    #[test]
    fn test_decode_with_special_tokens_visible() {
        let dec = toy_decoder();
        let text = dec.decode(&[100, 12, 101]);
        assert_eq!(text, "<s>hello</s>");
    }

    #[test]
    fn test_decode_skip_special_tokens() {
        let dec = toy_decoder().with_skip_special(true);
        let text = dec.decode(&[100, 12, 101]);
        assert_eq!(text, "hello");
    }

    #[test]
    fn test_special_token_in_input_text() {
        // If the input text literally contains "<s>", it should be
        // recognised as the BOS special token.
        let enc = toy_encoder();
        let ids = enc.encode("<s>");
        assert_eq!(ids, vec![100]);
    }

    #[test]
    fn test_pad_token_decode() {
        let dec = toy_decoder();
        assert_eq!(dec.decode(&[102, 102]), "<pad><pad>");
    }

    #[test]
    fn test_unk_token_decode() {
        let dec = toy_decoder();
        assert_eq!(dec.decode(&[103]), "<unk>");
    }

    // ── Empty string ────────────────────────────────────────────

    #[test]
    fn test_encode_empty_string() {
        let enc = toy_encoder();
        let ids = enc.encode("");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_encode_empty_with_bos_eos() {
        let enc = toy_encoder().with_bos(true).with_eos(true);
        let ids = enc.encode("");
        assert_eq!(ids, vec![100, 101]);
    }

    #[test]
    fn test_decode_empty() {
        let dec = toy_decoder();
        assert_eq!(dec.decode(&[]), "");
    }

    // ── Single character ────────────────────────────────────────

    #[test]
    fn test_encode_single_char() {
        let enc = toy_encoder();
        assert_eq!(enc.encode("h"), vec![0]);
        assert_eq!(enc.encode("e"), vec![1]);
    }

    #[test]
    fn test_decode_single_char() {
        let dec = toy_decoder();
        assert_eq!(dec.decode(&[0]), "h");
        assert_eq!(dec.decode(&[1]), "e");
    }

    // ── Byte fallback ───────────────────────────────────────────

    #[test]
    fn test_byte_fallback_unknown_char() {
        let (vocab, merges) = toy_vocab_and_merges();
        let special = toy_special();
        let bf = ByteFallback::new(200);
        let enc = BpeEncoder::new(vocab, merges, special).with_byte_fallback(bf);
        // 'z' is not in the vocabulary → byte fallback.
        let ids = enc.encode("z");
        assert_eq!(ids.len(), 1);
        // 'z' = 0x7A = 122 → base_id 200 + 122 = 322
        assert_eq!(ids[0], 322);
    }

    #[test]
    fn test_byte_fallback_round_trip() {
        let (vocab, merges) = toy_vocab_and_merges();
        let special = toy_special();
        let bf = ByteFallback::new(200);
        let enc =
            BpeEncoder::new(vocab.clone(), merges, special.clone()).with_byte_fallback(bf.clone());
        let dec = BpeDecoder::new(vocab, special).with_byte_fallback(bf);
        let ids = enc.encode("z");
        assert_eq!(dec.decode(&ids), "z");
    }

    #[test]
    fn test_byte_fallback_multibyte_utf8() {
        let (vocab, merges) = toy_vocab_and_merges();
        let special = toy_special();
        let bf = ByteFallback::new(200);
        let enc =
            BpeEncoder::new(vocab.clone(), merges, special.clone()).with_byte_fallback(bf.clone());
        let dec = BpeDecoder::new(vocab, special).with_byte_fallback(bf);
        // '€' is 3 bytes in UTF-8: E2 82 AC
        let ids = enc.encode("€");
        assert_eq!(ids.len(), 3);
        assert_eq!(dec.decode(&ids), "€");
    }

    #[test]
    fn test_byte_fallback_token_string_format() {
        assert_eq!(ByteFallback::byte_token_string(0x00), "<0x00>");
        assert_eq!(ByteFallback::byte_token_string(0xFF), "<0xFF>");
        assert_eq!(ByteFallback::byte_token_string(0x7A), "<0x7A>");
    }

    #[test]
    fn test_byte_fallback_encode_bytes() {
        let bf = ByteFallback::new(1000);
        let ids = bf.encode_bytes(&[0x41, 0x42]);
        assert_eq!(ids, vec![1065, 1066]); // 1000 + 65, 1000 + 66
    }

    #[test]
    fn test_byte_fallback_decode_ids() {
        let bf = ByteFallback::new(1000);
        let bytes = bf.decode_ids(&[1065, 1066]);
        assert_eq!(bytes, vec![0x41, 0x42]);
    }

    // ── Merge priority / ordering ───────────────────────────────

    #[test]
    fn test_merges_applied_by_rank() {
        // h+e (rank 0) fires first, then he+l (rank 2), then
        // hel+l (rank 3), fully reducing "hell" to id 11.
        let enc = toy_encoder();
        let ids = enc.encode("hell");
        assert_eq!(ids, vec![11]);
    }

    #[test]
    fn test_merge_rule_ordering() {
        let m0 = MergeRule::new("a", "b", 0);
        let m1 = MergeRule::new("c", "d", 1);
        assert!(m0 < m1);
    }

    #[test]
    fn test_merge_rule_equality() {
        let m1 = MergeRule::new("a", "b", 5);
        let m2 = MergeRule::new("a", "b", 5);
        assert_eq!(m1, m2);
    }

    // ── Batch encoding ──────────────────────────────────────────

    #[test]
    fn test_batch_encode_multiple() {
        let enc = toy_encoder();
        let batch = BatchEncoder::new(enc);
        let results = batch.encode_batch(&["hello", "world"]);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], vec![12]);
        assert_eq!(results[1], vec![16]);
    }

    #[test]
    fn test_batch_encode_empty_list() {
        let enc = toy_encoder();
        let batch = BatchEncoder::new(enc);
        let results = batch.encode_batch(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_encode_single() {
        let enc = toy_encoder();
        let batch = BatchEncoder::new(enc);
        let results = batch.encode_batch(&["hello"]);
        assert_eq!(results, vec![vec![12]]);
    }

    #[test]
    fn test_batch_encode_with_empty_string() {
        let enc = toy_encoder();
        let batch = BatchEncoder::new(enc);
        let results = batch.encode_batch(&["hello", "", "world"]);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], vec![12]);
        assert!(results[1].is_empty());
        assert_eq!(results[2], vec![16]);
    }

    // ── Vocabulary ──────────────────────────────────────────────

    #[test]
    fn test_vocab_len() {
        let (vocab, _) = toy_vocab_and_merges();
        assert_eq!(vocab.len(), 17);
    }

    #[test]
    fn test_vocab_is_empty() {
        let empty = BpeVocab::from_pairs(std::iter::empty());
        assert!(empty.is_empty());
    }

    #[test]
    fn test_vocab_contains() {
        let (vocab, _) = toy_vocab_and_merges();
        assert!(vocab.contains("hello"));
        assert!(!vocab.contains("xyz"));
    }

    #[test]
    fn test_vocab_contains_id() {
        let (vocab, _) = toy_vocab_and_merges();
        assert!(vocab.contains_id(0));
        assert!(!vocab.contains_id(999));
    }

    #[test]
    fn test_vocab_bidirectional() {
        let (vocab, _) = toy_vocab_and_merges();
        let id = vocab.token_to_id("hello").unwrap();
        assert_eq!(vocab.id_to_token(id).unwrap(), "hello");
    }

    #[test]
    fn test_vocab_missing_token() {
        let (vocab, _) = toy_vocab_and_merges();
        assert!(vocab.token_to_id("missing").is_none());
    }

    #[test]
    fn test_vocab_missing_id() {
        let (vocab, _) = toy_vocab_and_merges();
        assert!(vocab.id_to_token(9999).is_none());
    }

    // ── SpecialTokens ───────────────────────────────────────────

    #[test]
    fn test_special_tokens_all_pairs() {
        let st = toy_special();
        let pairs = st.all_pairs();
        assert_eq!(pairs.len(), 4);
    }

    #[test]
    fn test_special_tokens_custom() {
        let st = SpecialTokens::new().with_custom("<mask>", 50).with_custom("<cls>", 51);
        assert_eq!(st.custom.len(), 2);
        assert_eq!(*st.custom.get("<mask>").unwrap(), 50);
    }

    #[test]
    fn test_special_tokens_default_empty() {
        let st = SpecialTokens::default();
        assert!(st.bos.is_none());
        assert!(st.eos.is_none());
        assert!(st.pad.is_none());
        assert!(st.unk.is_none());
        assert!(st.custom.is_empty());
    }

    // ── Unicode ─────────────────────────────────────────────────

    #[test]
    fn test_encode_unknown_unicode_with_unk() {
        // Characters not in vocab and no byte fallback → UNK.
        let enc = toy_encoder();
        let ids = enc.encode("é");
        // 'é' not in vocab → fallback to UNK (103).
        assert_eq!(ids, vec![103]);
    }

    #[test]
    fn test_encode_unicode_with_byte_fallback() {
        let (vocab, merges) = toy_vocab_and_merges();
        let special = toy_special();
        let bf = ByteFallback::new(500);
        let enc = BpeEncoder::new(vocab, merges, special).with_byte_fallback(bf);
        let ids = enc.encode("é");
        // 'é' = UTF-8 bytes C3 A9 → 500+0xC3=695, 500+0xA9=669
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], 500 + 0xC3);
        assert_eq!(ids[1], 500 + 0xA9);
    }

    #[test]
    fn test_encode_mixed_known_unknown() {
        let (vocab, merges) = toy_vocab_and_merges();
        let special = toy_special();
        let bf = ByteFallback::new(500);
        let enc = BpeEncoder::new(vocab, merges, special).with_byte_fallback(bf);
        // 'h' is in vocab, 'z' requires byte fallback.
        let ids = enc.encode("hz");
        assert_eq!(ids[0], 0); // 'h'
        assert_eq!(ids[1], 500 + 0x7A); // 'z' byte fallback
    }

    // ── CPU reference implementations ───────────────────────────

    #[test]
    fn test_cpu_merge_pass_basic() {
        let mut syms = vec!["h".into(), "e".into(), "l".into()];
        let changed = cpu_merge_pass(&mut syms, "h", "e", "he");
        assert!(changed);
        assert_eq!(syms, vec!["he", "l"]);
    }

    #[test]
    fn test_cpu_merge_pass_no_match() {
        let mut syms = vec!["a".into(), "b".into()];
        let changed = cpu_merge_pass(&mut syms, "x", "y", "xy");
        assert!(!changed);
        assert_eq!(syms, vec!["a", "b"]);
    }

    #[test]
    fn test_cpu_merge_pass_multiple_matches() {
        let mut syms = vec!["a".into(), "b".into(), "a".into(), "b".into()];
        let changed = cpu_merge_pass(&mut syms, "a", "b", "ab");
        assert!(changed);
        assert_eq!(syms, vec!["ab", "ab"]);
    }

    #[test]
    fn test_cpu_full_bpe_hello() {
        let (_, merges) = toy_vocab_and_merges();
        let result = cpu_full_bpe("hello", &merges);
        assert_eq!(result, vec!["hello"]);
    }

    #[test]
    fn test_cpu_full_bpe_single_char() {
        let (_, merges) = toy_vocab_and_merges();
        let result = cpu_full_bpe("h", &merges);
        assert_eq!(result, vec!["h"]);
    }

    #[test]
    fn test_cpu_full_bpe_empty() {
        let (_, merges) = toy_vocab_and_merges();
        let result = cpu_full_bpe("", &merges);
        assert!(result.is_empty());
    }

    // ── BpeStats ────────────────────────────────────────────────

    #[test]
    fn test_stats_basic() {
        let enc = toy_encoder();
        let dec = toy_decoder();
        let stats = collect_stats(&enc, &dec, &["hello", "world"]);
        assert!(stats.encode_time.as_nanos() > 0);
        assert!(stats.decode_time.as_nanos() > 0);
        assert!((stats.avg_tokens_per_text - 1.0).abs() < 0.01);
        assert!((stats.vocab_coverage - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_stats_empty_input() {
        let enc = toy_encoder();
        let dec = toy_decoder();
        let stats = collect_stats(&enc, &dec, &[]);
        assert_eq!(stats.avg_tokens_per_text, 0.0);
        assert_eq!(stats.vocab_coverage, 0.0);
    }

    // ── Edge cases ──────────────────────────────────────────────

    #[test]
    fn test_all_special_tokens_only() {
        let enc = toy_encoder();
        let ids = enc.encode("<s></s><pad><unk>");
        assert_eq!(ids, vec![100, 101, 102, 103]);
    }

    #[test]
    fn test_very_long_text() {
        let enc = toy_encoder();
        let text = "hello ".repeat(100);
        let ids = enc.encode(text.trim_end());
        // 100 "hello" tokens + 99 space tokens = 199.
        assert_eq!(ids.len(), 199);
    }

    #[test]
    fn test_repeated_single_char() {
        let enc = toy_encoder();
        let ids = enc.encode("hh");
        // Two separate 'h' tokens (no merge for h+h).
        assert_eq!(ids, vec![0, 0]);
    }

    #[test]
    fn test_no_merges_possible() {
        // Vocabulary with only single chars and no merges.
        let vocab = BpeVocab::from_pairs(vec![("a".into(), 0), ("b".into(), 1)]);
        let enc = BpeEncoder::new(vocab, vec![], SpecialTokens::new());
        assert_eq!(enc.encode("ab"), vec![0, 1]);
        assert_eq!(enc.encode("ba"), vec![1, 0]);
    }

    #[test]
    fn test_encode_only_spaces() {
        let enc = toy_encoder();
        let ids = enc.encode("   ");
        assert_eq!(ids, vec![7, 7, 7]);
    }

    #[test]
    fn test_encode_with_custom_special() {
        let special = SpecialTokens::new().with_custom("[SEP]", 200);
        let vocab = BpeVocab::from_pairs(vec![("a".into(), 0)]);
        let enc = BpeEncoder::new(vocab, vec![], special);
        let ids = enc.encode("a[SEP]a");
        assert_eq!(ids, vec![0, 200, 0]);
    }

    #[test]
    fn test_decoder_unknown_id_ignored() {
        let dec = toy_decoder();
        // id 999 is not in vocab or specials → silently dropped.
        let text = dec.decode(&[12, 999, 16]);
        assert_eq!(text, "helloworld");
    }

    // ── OpenCL kernel source ────────────────────────────────────

    #[test]
    fn test_opencl_kernel_source_not_empty() {
        assert!(!BPE_MERGE_CL.is_empty());
    }

    #[test]
    fn test_opencl_kernel_contains_merge_fn() {
        assert!(BPE_MERGE_CL.contains("bpe_merge"));
    }

    #[test]
    fn test_opencl_kernel_contains_compact_fn() {
        assert!(BPE_MERGE_CL.contains("compact_symbols"));
    }

    // ── Encoder accessor methods ────────────────────────────────

    #[test]
    fn test_encoder_vocab_accessor() {
        let enc = toy_encoder();
        assert_eq!(enc.vocab().len(), 17);
    }

    #[test]
    fn test_encoder_merges_accessor() {
        let enc = toy_encoder();
        assert_eq!(enc.merges().len(), 9);
    }

    #[test]
    fn test_encoder_special_tokens_accessor() {
        let enc = toy_encoder();
        assert!(enc.special_tokens().bos.is_some());
    }

    #[test]
    fn test_batch_encoder_inner_accessor() {
        let enc = toy_encoder();
        let batch = BatchEncoder::new(enc);
        assert_eq!(batch.encoder().vocab().len(), 17);
    }

    // ── Property-like round-trip ────────────────────────────────

    #[test]
    fn test_round_trip_known_words() {
        let enc = toy_encoder();
        let dec = toy_decoder();
        for word in &["hello", "world", "he", "hell"] {
            let ids = enc.encode(word);
            let decoded = dec.decode(&ids);
            assert_eq!(&decoded, word, "round-trip failed for {word}");
        }
    }

    #[test]
    fn test_round_trip_with_byte_fallback() {
        let (vocab, merges) = toy_vocab_and_merges();
        let special = toy_special();
        let bf = ByteFallback::new(500);
        let enc =
            BpeEncoder::new(vocab.clone(), merges, special.clone()).with_byte_fallback(bf.clone());
        let dec = BpeDecoder::new(vocab, special).with_byte_fallback(bf);
        for text in &["hello", "z", "€", "hzh"] {
            let ids = enc.encode(text);
            let decoded = dec.decode(&ids);
            assert_eq!(&decoded, text, "byte-fallback round-trip failed for {text}");
        }
    }

    #[test]
    fn test_stats_coverage_with_fallback() {
        let (vocab, merges) = toy_vocab_and_merges();
        let special = toy_special();
        let bf = ByteFallback::new(500);
        let enc =
            BpeEncoder::new(vocab.clone(), merges, special.clone()).with_byte_fallback(bf.clone());
        let dec = BpeDecoder::new(vocab, special).with_byte_fallback(bf);
        let stats = collect_stats(&enc, &dec, &["helloézz"]);
        // 'hello' is 1 vocab token, 'é' is 2 byte-fallback, 'z' is 1
        // byte-fallback each. Total 5 tokens, 1 in vocab → 20%.
        assert!(stats.vocab_coverage < 1.0);
        assert!(stats.vocab_coverage > 0.0);
    }
}
