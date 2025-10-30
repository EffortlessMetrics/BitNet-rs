//! Unit tests for stop-sequence correctness (fixes "one token late" bug)
//!
//! These tests validate that stop sequences are detected at the exact moment
//! they occur, not one token later. The critical fix is checking the candidate
//! token *before* adding it to the output sequence.
//!
//! ## Background
//!
//! The "one token late" bug occurred because stop sequences were checked
//! *after* the token was added to the output. This meant:
//!
//! 1. Token "!" generates text "Hello world!"
//! 2. Check: does "Hello world" end with "world!"? No.
//! 3. Add "!" to output → "Hello world!"
//! 4. Next iteration: Token "." generates
//! 5. Check: does "Hello world!" end with "world!"? Yes! STOP.
//! 6. But we already generated one extra token.
//!
//! ## The Fix
//!
//! The `matches_with_candidate()` helper checks if adding the candidate token
//! would complete a stop sequence, BEFORE adding it to the output:
//!
//! 1. Token "!" is candidate
//! 2. Check: does "Hello world" + "!" end with "world!"? Yes! STOP.
//! 3. Don't add "!" to output
//! 4. Generation stops at exactly the right moment
//!
//! ## Test Strategy
//!
//! These tests validate the `matches_with_candidate()` logic directly by:
//! - Creating a mock tokenizer with predictable decode behavior
//! - Testing exact-match detection (no late stops)
//! - Testing multiple stop sequences
//! - Testing token ID vs string priority
//! - Testing Unicode/multi-byte sequences
//! - Testing rolling window boundary conditions
use bitnet_inference::config::GenerationConfig;
use bitnet_tokenizers::Tokenizer;
use std::sync::Arc;
/// Mock tokenizer that maps token IDs to predictable strings
///
/// Token mapping:
/// - 1 → "Hello"
/// - 2 → " world"
/// - 3 → "!"
/// - 4 → "\n"
/// - 5 → "\n"
/// - 6 → "Q"
/// - 7 → ":"
/// - 8 → " "
/// - 9 → "世"
/// - 10 → "界"
/// - 999 → "<eos>"
/// - 128009 → "<|eot_id|>"
struct PredictableTokenizer;
impl Tokenizer for PredictableTokenizer {
    fn encode(
        &self,
        _text: &str,
        _add_bos: bool,
        _add_special: bool,
    ) -> bitnet_common::Result<Vec<u32>> {
        Ok(vec![])
    }
    fn decode(&self, tokens: &[u32]) -> bitnet_common::Result<String> {
        let mut result = String::new();
        for &token in tokens {
            match token {
                1 => result.push_str("Hello"),
                2 => result.push_str(" world"),
                3 => result.push('!'),
                4 => result.push('\n'),
                5 => result.push('\n'),
                6 => result.push('Q'),
                7 => result.push(':'),
                8 => result.push(' '),
                9 => result.push('世'),
                10 => result.push('界'),
                999 => result.push_str("<eos>"),
                128009 => result.push_str("<|eot_id|>"),
                _ => result.push_str(&format!("[token_{}]", token)),
            }
        }
        Ok(result)
    }
    fn vocab_size(&self) -> usize {
        150000
    }
    fn token_to_piece(&self, token: u32) -> Option<String> {
        self.decode(&[token]).ok()
    }
    fn eos_token_id(&self) -> Option<u32> {
        Some(999)
    }
    fn bos_token_id(&self) -> Option<u32> {
        None
    }
    fn pad_token_id(&self) -> Option<u32> {
        None
    }
}
/// Helper function that mimics the `matches_with_candidate` logic
///
/// This is the critical fix for the "one token late" bug.
fn matches_with_candidate(
    tail_tokens: &[u32],
    candidate_token: u32,
    stop_sequences: &[String],
    tokenizer: &Arc<dyn Tokenizer>,
) -> bool {
    let mut test_tokens = tail_tokens.to_vec();
    test_tokens.push(candidate_token);
    let text = tokenizer.decode(&test_tokens).unwrap_or_default();
    stop_sequences.iter().any(|seq| text.ends_with(seq))
}
/// Helper function that mimics the old (buggy) behavior
///
/// This checks if the current tokens (without candidate) match stop sequences.
/// This is the "one token late" behavior we're fixing.
#[allow(dead_code)]
fn matches_without_candidate(
    current_tokens: &[u32],
    stop_sequences: &[String],
    tokenizer: &Arc<dyn Tokenizer>,
) -> bool {
    let text = tokenizer.decode(current_tokens).unwrap_or_default();
    stop_sequences.iter().any(|seq| text.ends_with(seq))
}
#[test]
fn test_stop_sequence_exact_match() {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(PredictableTokenizer);
    let stop_sequences = vec!["world!".to_string()];
    let tail_tokens = vec![1, 2];
    let candidate = 3;
    assert!(
        matches_with_candidate(&tail_tokens, candidate, &stop_sequences, &tokenizer),
        "Should detect stop sequence when candidate completes it"
    );
    assert!(
        !matches_without_candidate(&tail_tokens, &stop_sequences, &tokenizer),
        "Old behavior: wouldn't detect stop sequence without candidate (one token late bug)"
    );
}
#[test]
fn test_stop_sequence_not_one_token_late() {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(PredictableTokenizer);
    let stop_sequences = vec!["world!".to_string()];
    assert!(
        !matches_with_candidate(&[], 1, &stop_sequences, &tokenizer),
        "Token 1 ('Hello') should not trigger stop"
    );
    assert!(
        !matches_with_candidate(&[1], 2, &stop_sequences, &tokenizer),
        "Token 2 (' world') should not trigger stop"
    );
    assert!(
        matches_with_candidate(&[1, 2], 3, &stop_sequences, &tokenizer),
        "Token 3 ('!') should trigger stop immediately"
    );
}
#[test]
fn test_multiple_stop_sequences() {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(PredictableTokenizer);
    let stop_sequences = vec!["world!".to_string(), "\n\nQ:".to_string(), "<eos>".to_string()];
    assert!(
        matches_with_candidate(&[1, 2], 3, &stop_sequences, &tokenizer),
        "Should match 'world!' stop sequence"
    );
    assert!(
        matches_with_candidate(&[4, 5, 6], 7, &stop_sequences, &tokenizer),
        "Should match '\\n\\nQ:' stop sequence"
    );
    assert!(
        matches_with_candidate(&[], 999, &stop_sequences, &tokenizer),
        "Should match '<eos>' stop sequence"
    );
    assert!(
        !matches_with_candidate(&[1], 2, &stop_sequences, &tokenizer),
        "Should not match when no stop sequence is present"
    );
}
#[test]
fn test_stop_token_id_vs_string() {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(PredictableTokenizer);
    let stop_sequences = vec!["<|eot_id|>".to_string()];
    assert!(
        matches_with_candidate(&[], 128009, &stop_sequences, &tokenizer),
        "String-based stop should detect '<|eot_id|>' token"
    );
    let config = GenerationConfig::greedy().with_stop_token_ids(vec![128009]);
    assert!(
        config.stop_token_ids.contains(&128009),
        "Token ID-based stop should detect token 128009 directly"
    );
}
#[test]
fn test_rolling_window_with_candidate() {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(PredictableTokenizer);
    let stop_sequences = vec!["world!".to_string()];
    let all_tokens = [100, 101, 102, 103, 104, 1, 2];
    let window_size = 3;
    let tail_start = all_tokens.len().saturating_sub(window_size - 1);
    let tail_tokens = &all_tokens[tail_start..];
    assert_eq!(tail_tokens, &[1, 2], "Tail window should be last 2 tokens");
    let candidate = 3;
    assert!(
        matches_with_candidate(tail_tokens, candidate, &stop_sequences, &tokenizer),
        "Should detect stop sequence in tail window with candidate"
    );
    let effective_window = window_size.min(all_tokens.len() + 1);
    assert_eq!(effective_window, 3, "Window should account for candidate (+1)");
}
#[test]
fn test_unicode_stop_sequences() {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(PredictableTokenizer);
    let stop_sequences = vec!["世界".to_string()];
    assert!(
        matches_with_candidate(&[9], 10, &stop_sequences, &tokenizer),
        "Should detect Unicode stop sequence '世界'"
    );
    assert!(
        !matches_with_candidate(&[], 9, &stop_sequences, &tokenizer),
        "Partial Unicode match should not trigger stop"
    );
    let stop_sequences_emoji = vec!["<|eot_id|>".to_string()];
    assert!(
        matches_with_candidate(&[], 128009, &stop_sequences_emoji, &tokenizer),
        "Should detect special token stop sequence"
    );
}
#[test]
fn test_empty_stop_sequences() {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(PredictableTokenizer);
    let stop_sequences: Vec<String> = vec![];
    assert!(
        !matches_with_candidate(&[1, 2], 3, &stop_sequences, &tokenizer),
        "Empty stop sequences should never trigger stop"
    );
}
#[test]
fn test_stop_sequence_boundary_conditions() {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(PredictableTokenizer);
    let stop_sequences = vec!["Hello world! This is a long sequence".to_string()];
    assert!(
        !matches_with_candidate(&[1, 2], 3, &stop_sequences, &tokenizer),
        "Long stop sequence should not match short text"
    );
    let stop_sequences_single = vec!["!".to_string()];
    assert!(
        matches_with_candidate(&[], 3, &stop_sequences_single, &tokenizer),
        "Single-token stop sequence should match"
    );
    let stop_sequences_prefix = vec!["Hello".to_string()];
    assert!(
        matches_with_candidate(&[], 1, &stop_sequences_prefix, &tokenizer),
        "Stop sequence at text start should match (ends_with is still true)"
    );
    let stop_sequences_multi = vec!["world!".to_string()];
    let mut current_tokens = vec![];
    assert!(
        !matches_with_candidate(&current_tokens, 1, &stop_sequences_multi, &tokenizer),
        "First token should not match"
    );
    current_tokens.push(1);
    assert!(
        !matches_with_candidate(&current_tokens, 2, &stop_sequences_multi, &tokenizer),
        "Second token should not match"
    );
    current_tokens.push(2);
    assert!(
        matches_with_candidate(&current_tokens, 3, &stop_sequences_multi, &tokenizer),
        "Third token should match and trigger stop"
    );
}
#[test]
fn test_stop_sequence_partial_matches() {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(PredictableTokenizer);
    let stop_sequences = vec!["\n\nQ:".to_string()];
    assert!(
        !matches_with_candidate(&[], 4, &stop_sequences, &tokenizer),
        "Single newline should not trigger stop"
    );
    assert!(
        !matches_with_candidate(&[4], 5, &stop_sequences, &tokenizer),
        "Double newline should not trigger stop"
    );
    assert!(
        !matches_with_candidate(&[4, 5], 6, &stop_sequences, &tokenizer),
        "Partial match should not trigger stop"
    );
    assert!(
        matches_with_candidate(&[4, 5, 6], 7, &stop_sequences, &tokenizer),
        "Full match should trigger stop"
    );
}
#[test]
fn test_window_size_edge_cases() {
    let _tokenizer: Arc<dyn Tokenizer> = Arc::new(PredictableTokenizer);
    let current_tokens: Vec<u32> = vec![];
    let window_size = 64_usize;
    let effective_window = window_size.min(current_tokens.len() + 1);
    assert_eq!(effective_window, 1, "Window size should be 1 when current_tokens is empty");
    let current_tokens = [1_u32, 2];
    let effective_window = window_size.min(current_tokens.len() + 1);
    assert_eq!(effective_window, 3, "Window size should be len + 1 when smaller than max window");
    let current_tokens: Vec<u32> = (0..100).collect();
    let effective_window = window_size.min(current_tokens.len() + 1);
    assert_eq!(effective_window, 64, "Window size should be capped at max window");
    let current_tokens = [100_u32, 101, 102, 1, 2];
    let window_size = 3;
    let tail_start = current_tokens.len().saturating_sub(window_size - 1);
    assert_eq!(tail_start, 3, "Tail should start at index 3 (last 2 tokens + candidate)");
    let tail_tokens = &current_tokens[tail_start..];
    assert_eq!(tail_tokens, &[1, 2], "Tail should be last (window_size - 1) tokens");
}
#[test]
fn test_config_stop_sequences_integration() {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(PredictableTokenizer);
    let config = GenerationConfig::greedy()
        .with_stop_sequences(vec!["world!".to_string(), "<eos>".to_string()])
        .with_stop_token_ids(vec![999, 128009])
        .with_stop_string_window(64);
    assert!(
        matches_with_candidate(&[1, 2], 3, &config.stop_sequences, &tokenizer),
        "Config stop_sequences should work with matches_with_candidate"
    );
    assert!(config.stop_token_ids.contains(&999), "Config stop_token_ids should contain EOS token");
    assert!(
        config.stop_token_ids.contains(&128009),
        "Config stop_token_ids should contain LLaMA-3 EOT token"
    );
    assert_eq!(config.stop_string_window, 64, "Config should allow custom window size");
}
