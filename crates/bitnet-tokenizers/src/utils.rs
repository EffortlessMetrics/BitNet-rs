//! Tokenizer extension utilities: vocabulary statistics, batch tokenization,
//! benchmarking, and roundtrip validation.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::Tokenizer;

/// Vocabulary statistics computed from a tokenizer.
#[derive(Debug, Clone, PartialEq)]
pub struct VocabStats {
    /// Total vocabulary size (including padding).
    pub vocab_size: usize,
    /// Number of special tokens (BOS, EOS, PAD) that are configured.
    pub special_token_count: usize,
    /// Average token length in bytes (across printable pieces).
    pub avg_token_len: f64,
    /// Maximum token length in bytes (across printable pieces).
    pub max_token_len: usize,
    /// Fraction of single-byte values (0–255) covered by the vocabulary.
    pub byte_coverage: f64,
}

/// Compute vocabulary statistics from a tokenizer.
///
/// Iterates over token IDs `0..vocab_size`, calling `token_to_piece` on each.
/// Tokens that return `None` are counted but excluded from length stats.
pub fn compute_vocab_stats(tokenizer: &dyn Tokenizer) -> VocabStats {
    let vocab_size = tokenizer.vocab_size();

    let mut special_count: usize = 0;
    if tokenizer.bos_token_id().is_some() {
        special_count += 1;
    }
    if tokenizer.eos_token_id().is_some() {
        special_count += 1;
    }
    if tokenizer.pad_token_id().is_some() {
        special_count += 1;
    }

    let mut total_len: usize = 0;
    let mut max_len: usize = 0;
    let mut piece_count: usize = 0;
    let mut byte_set = [false; 256];

    let scan_limit = vocab_size.min(200_000);
    for id in 0..scan_limit as u32 {
        if let Some(piece) = tokenizer.token_to_piece(id) {
            let len = piece.len();
            total_len += len;
            if len > max_len {
                max_len = len;
            }
            piece_count += 1;

            // Track single-byte coverage.
            for &b in piece.as_bytes() {
                byte_set[b as usize] = true;
            }
        }
    }

    let avg_token_len = if piece_count > 0 { total_len as f64 / piece_count as f64 } else { 0.0 };

    let byte_coverage = byte_set.iter().filter(|&&b| b).count() as f64 / 256.0;

    VocabStats {
        vocab_size,
        special_token_count: special_count,
        avg_token_len,
        max_token_len: max_len,
        byte_coverage,
    }
}

/// Tokenize a batch of texts, returning one token-ID vector per input.
pub fn tokenize_batch(
    tokenizer: &dyn Tokenizer,
    texts: &[&str],
) -> Vec<bitnet_common::Result<Vec<u32>>> {
    texts.iter().map(|t| tokenizer.encode(t, false, false)).collect()
}

/// Fast approximate token count (chars / 4, minimum 1 for non-empty text).
///
/// This is a zero-allocation heuristic useful for capacity pre-allocation and
/// cost estimation when an actual tokenizer is unavailable or too expensive.
pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    (text.len() / 4).max(1)
}

/// Timing results from [`TokenizerBenchmark::run`].
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Number of iterations executed.
    pub iterations: usize,
    /// Total wall-clock time for all iterations.
    pub total_time: Duration,
    /// Mean time per encode call.
    pub mean_encode: Duration,
    /// Mean time per decode call.
    pub mean_decode: Duration,
}

/// Micro-benchmark harness for tokenizer encode/decode throughput.
pub struct TokenizerBenchmark {
    text: String,
    iterations: usize,
}

impl TokenizerBenchmark {
    /// Create a new benchmark with the given sample text and iteration count.
    pub fn new(text: impl Into<String>, iterations: usize) -> Self {
        Self { text: text.into(), iterations: iterations.max(1) }
    }

    /// Run the benchmark against `tokenizer`, returning timing results.
    pub fn run(&self, tokenizer: &dyn Tokenizer) -> BenchmarkResult {
        let start = Instant::now();

        let mut encode_total = Duration::ZERO;
        let mut decode_total = Duration::ZERO;

        for _ in 0..self.iterations {
            let t0 = Instant::now();
            let tokens = tokenizer.encode(&self.text, false, false).unwrap_or_default();
            encode_total += t0.elapsed();

            let t1 = Instant::now();
            let _ = tokenizer.decode(&tokens);
            decode_total += t1.elapsed();
        }

        let n = self.iterations as u32;
        BenchmarkResult {
            iterations: self.iterations,
            total_time: start.elapsed(),
            mean_encode: encode_total / n,
            mean_decode: decode_total / n,
        }
    }
}

/// Check whether encode→decode round-trips to the original text.
///
/// Returns `true` when `decode(encode(text)) == text`.
pub fn validate_roundtrip(tokenizer: &dyn Tokenizer, text: &str) -> bool {
    let Ok(tokens) = tokenizer.encode(text, false, false) else {
        return false;
    };
    let Ok(decoded) = tokenizer.decode(&tokens) else {
        return false;
    };
    decoded == text
}

/// Extract a map of known special tokens and their IDs.
///
/// Probes BOS, EOS, PAD via trait methods and additionally looks up common
/// LLaMA-3 / Mistral markers via `token_to_id`.
pub fn special_tokens_map(tokenizer: &dyn Tokenizer) -> HashMap<String, u32> {
    let mut map = HashMap::new();

    if let Some(id) = tokenizer.bos_token_id() {
        map.insert("bos".to_string(), id);
    }
    if let Some(id) = tokenizer.eos_token_id() {
        map.insert("eos".to_string(), id);
    }
    if let Some(id) = tokenizer.pad_token_id() {
        map.insert("pad".to_string(), id);
    }

    // Probe well-known special token strings.
    let probes = [
        "<|eot_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "[INST]",
        "[/INST]",
        "</s>",
        "<s>",
        "<unk>",
    ];
    for token_str in probes {
        if let Some(id) = tokenizer.token_to_id(token_str) {
            map.insert(token_str.to_string(), id);
        }
    }

    map
}

// ──────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BasicTokenizer, MockTokenizer};

    // ── VocabStats ────────────────────────────────────────────────────

    #[test]
    fn stats_basic_tokenizer_default() {
        let tok = BasicTokenizer::new();
        let stats = compute_vocab_stats(&tok);
        assert_eq!(stats.vocab_size, 50257);
        // BasicTokenizer: eos=Some(50256), bos=None, pad=None → 1 special
        assert_eq!(stats.special_token_count, 1);
    }

    #[test]
    fn stats_with_all_specials() {
        let tok = BasicTokenizer::with_config(1000, Some(1), Some(2), Some(3));
        let stats = compute_vocab_stats(&tok);
        assert_eq!(stats.special_token_count, 3);
    }

    #[test]
    fn stats_max_token_len_positive() {
        let tok = BasicTokenizer::new();
        let stats = compute_vocab_stats(&tok);
        assert!(stats.max_token_len >= 1, "byte tokens are at least 1 char");
    }

    #[test]
    fn stats_avg_token_len_positive() {
        let tok = BasicTokenizer::new();
        let stats = compute_vocab_stats(&tok);
        assert!(stats.avg_token_len > 0.0);
    }

    #[test]
    fn stats_byte_coverage_nonzero() {
        let tok = BasicTokenizer::new();
        let stats = compute_vocab_stats(&tok);
        // BasicTokenizer maps 0–255 to single-byte pieces (lossy UTF-8),
        // so roughly half the byte range is covered as printable.
        assert!(
            stats.byte_coverage > 0.4,
            "expected meaningful byte coverage, got {}",
            stats.byte_coverage
        );
    }

    #[test]
    fn stats_mock_tokenizer() {
        let tok = MockTokenizer::new();
        let stats = compute_vocab_stats(&tok);
        assert_eq!(stats.vocab_size, 50257);
        // MockTokenizer: all special getters return None → 0 specials.
        assert_eq!(stats.special_token_count, 0);
    }

    // ── tokenize_batch ────────────────────────────────────────────────

    #[test]
    fn batch_empty_input() {
        let tok = MockTokenizer::new();
        let results = tokenize_batch(&tok, &[]);
        assert!(results.is_empty());
    }

    #[test]
    fn batch_single_text() {
        let tok = MockTokenizer::new();
        let results = tokenize_batch(&tok, &["hi"]);
        assert_eq!(results.len(), 1);
        let tokens = results[0].as_ref().unwrap();
        assert_eq!(tokens.len(), 2); // 'h','i'
    }

    #[test]
    fn batch_multiple_texts() {
        let tok = MockTokenizer::new();
        let results = tokenize_batch(&tok, &["a", "bb", "ccc"]);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].as_ref().unwrap().len(), 1);
        assert_eq!(results[1].as_ref().unwrap().len(), 2);
        assert_eq!(results[2].as_ref().unwrap().len(), 3);
    }

    #[test]
    fn batch_with_empty_string() {
        let tok = MockTokenizer::new();
        let results = tokenize_batch(&tok, &["", "a", ""]);
        assert_eq!(results.len(), 3);
        assert!(results[0].as_ref().unwrap().is_empty());
        assert_eq!(results[1].as_ref().unwrap().len(), 1);
        assert!(results[2].as_ref().unwrap().is_empty());
    }

    // ── estimate_tokens ───────────────────────────────────────────────

    #[test]
    fn estimate_empty() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn estimate_short_text() {
        // "hi" → 2 bytes / 4 = 0, but min is 1
        assert_eq!(estimate_tokens("hi"), 1);
    }

    #[test]
    fn estimate_longer_text() {
        let text = "Hello, world! This is a test.";
        let est = estimate_tokens(text);
        assert!(est >= 1);
        assert_eq!(est, text.len() / 4);
    }

    #[test]
    fn estimate_exact_multiple() {
        // 8 bytes / 4 = 2
        assert_eq!(estimate_tokens("abcdefgh"), 2);
    }

    #[test]
    fn estimate_single_char() {
        assert_eq!(estimate_tokens("x"), 1);
    }

    // ── validate_roundtrip ────────────────────────────────────────────

    #[test]
    fn roundtrip_ascii_mock() {
        let tok = MockTokenizer::new();
        assert!(validate_roundtrip(&tok, "hello world"));
    }

    #[test]
    fn roundtrip_empty_string() {
        let tok = MockTokenizer::new();
        assert!(validate_roundtrip(&tok, ""));
    }

    #[test]
    fn roundtrip_basic_tokenizer_ascii() {
        let tok = BasicTokenizer::new();
        assert!(validate_roundtrip(&tok, "abc123"));
    }

    // ── special_tokens_map ────────────────────────────────────────────

    #[test]
    fn specials_basic_default() {
        let tok = BasicTokenizer::new();
        let map = special_tokens_map(&tok);
        assert!(map.contains_key("eos"));
        assert_eq!(map["eos"], 50256);
        assert!(!map.contains_key("bos"));
        assert!(!map.contains_key("pad"));
    }

    #[test]
    fn specials_all_configured() {
        let tok = BasicTokenizer::with_config(50257, Some(10), Some(20), Some(30));
        let map = special_tokens_map(&tok);
        assert_eq!(map["bos"], 10);
        assert_eq!(map["eos"], 20);
        assert_eq!(map["pad"], 30);
    }

    #[test]
    fn specials_mock_with_llama3_tokens() {
        let tok = MockTokenizer::with_special_tokens(&[
            ("<|eot_id|>", 128009),
            ("<|start_header_id|>", 128006),
        ]);
        let map = special_tokens_map(&tok);
        assert_eq!(map["<|eot_id|>"], 128009);
        assert_eq!(map["<|start_header_id|>"], 128006);
    }

    #[test]
    fn specials_mock_empty() {
        let tok = MockTokenizer::new();
        let map = special_tokens_map(&tok);
        // MockTokenizer returns None for bos/eos/pad and token_to_id.
        assert!(map.is_empty());
    }

    // ── TokenizerBenchmark ────────────────────────────────────────────

    #[test]
    fn benchmark_returns_results() {
        let tok = MockTokenizer::new();
        let bench = TokenizerBenchmark::new("hello", 10);
        let result = bench.run(&tok);
        assert_eq!(result.iterations, 10);
        assert!(result.total_time > Duration::ZERO);
    }

    #[test]
    fn benchmark_single_iteration() {
        let tok = MockTokenizer::new();
        let bench = TokenizerBenchmark::new("test", 1);
        let result = bench.run(&tok);
        assert_eq!(result.iterations, 1);
        assert!(result.mean_encode <= result.total_time);
    }

    #[test]
    fn benchmark_zero_iterations_clamped() {
        let tok = MockTokenizer::new();
        let bench = TokenizerBenchmark::new("x", 0);
        // 0 is clamped to 1
        let result = bench.run(&tok);
        assert_eq!(result.iterations, 1);
    }

    #[test]
    fn benchmark_empty_text() {
        let tok = MockTokenizer::new();
        let bench = TokenizerBenchmark::new("", 5);
        let result = bench.run(&tok);
        assert_eq!(result.iterations, 5);
    }

    // ── cross-function integration ────────────────────────────────────

    #[test]
    fn batch_roundtrip_consistency() {
        let tok = MockTokenizer::new();
        let texts: &[&str] = &["alpha", "beta", "gamma"];
        let results = tokenize_batch(&tok, texts);
        for (i, text) in texts.iter().enumerate() {
            let tokens = results[i].as_ref().unwrap();
            let decoded = tok.decode(tokens).unwrap();
            assert_eq!(&decoded, text);
        }
    }

    #[test]
    fn estimate_vs_actual() {
        let tok = MockTokenizer::new();
        let text = "The quick brown fox jumps over the lazy dog.";
        let est = estimate_tokens(text);
        let actual = tok.encode(text, false, false).unwrap().len();
        // Estimate should be within 5× of actual for byte-level mock.
        assert!(
            est <= actual * 5 && actual <= est * 5,
            "estimate {} too far from actual {}",
            est,
            actual
        );
    }
}
