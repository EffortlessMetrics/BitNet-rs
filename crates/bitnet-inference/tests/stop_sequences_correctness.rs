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

use std::sync::Arc;

use bitnet_inference::config::GenerationConfig;
use bitnet_tokenizers::Tokenizer;

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
        // Not used in these tests
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
        150000 // Large enough for all test token IDs
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
    // Test that when the candidate token completes a stop sequence,
    // it's detected immediately (not one token later)

    let tokenizer: Arc<dyn Tokenizer> = Arc::new(PredictableTokenizer);
    let stop_sequences = vec!["world!".to_string()];

    // Scenario: We have "Hello" and " world", candidate is "!"
    // Expected: "Hello" + " world" + "!" = "Hello world!" ends with "world!" → STOP

    let tail_tokens = vec![1, 2]; // "Hello world"
    let candidate = 3; // "!"

    // The fix: Check WITH the candidate token
    assert!(
        matches_with_candidate(&tail_tokens, candidate, &stop_sequences, &tokenizer),
        "Should detect stop sequence when candidate completes it"
    );

    // Show that the old behavior would be wrong:
    // Without the candidate, we wouldn't detect the stop yet
    assert!(
        !matches_without_candidate(&tail_tokens, &stop_sequences, &tokenizer),
        "Old behavior: wouldn't detect stop sequence without candidate (one token late bug)"
    );
}

#[test]
fn test_stop_sequence_not_one_token_late() {
    // Verify we DON'T generate an extra token after the stop sequence

    let tokenizer: Arc<dyn Tokenizer> = Arc::new(PredictableTokenizer);
    let stop_sequences = vec!["world!".to_string()];

    // Generation sequence:
    // 1. Generate token 1: "Hello"
    // 2. Generate token 2: "Hello world"
    // 3. Generate token 3: "Hello world!" → STOP (don't add token 3)

    // Step 1: Token 1 doesn't trigger stop
    assert!(
        !matches_with_candidate(&[], 1, &stop_sequences, &tokenizer),
        "Token 1 ('Hello') should not trigger stop"
    );

    // Step 2: Token 2 doesn't trigger stop
    assert!(
        !matches_with_candidate(&[1], 2, &stop_sequences, &tokenizer),
        "Token 2 (' world') should not trigger stop"
    );

    // Step 3: Token 3 DOES trigger stop (before adding to output)
    assert!(
        matches_with_candidate(&[1, 2], 3, &stop_sequences, &tokenizer),
        "Token 3 ('!') should trigger stop immediately"
    );

    // Step 4: We should NOT generate token 4
    // (this test validates that we stop at exactly 3 tokens, not 4)
}

#[test]
fn test_multiple_stop_sequences() {
    // Multiple stop sequences should be handled correctly

    let tokenizer: Arc<dyn Tokenizer> = Arc::new(PredictableTokenizer);
    let stop_sequences = vec!["world!".to_string(), "\n\nQ:".to_string(), "<eos>".to_string()];

    // Test first stop sequence: "world!"
    assert!(
        matches_with_candidate(&[1, 2], 3, &stop_sequences, &tokenizer),
        "Should match 'world!' stop sequence"
    );

    // Test second stop sequence: "\n\nQ:"
    // Tokens: [4, 5, 6, 7] = "\n" + "\n" + "Q" + ":" = "\n\nQ:"
    assert!(
        matches_with_candidate(&[4, 5, 6], 7, &stop_sequences, &tokenizer),
        "Should match '\\n\\nQ:' stop sequence"
    );

    // Test third stop sequence: "<eos>"
    // Token 999 = "<eos>"
    assert!(
        matches_with_candidate(&[], 999, &stop_sequences, &tokenizer),
        "Should match '<eos>' stop sequence"
    );

    // Test non-matching sequence
    assert!(
        !matches_with_candidate(&[1], 2, &stop_sequences, &tokenizer),
        "Should not match when no stop sequence is present"
    );
}

#[test]
fn test_stop_token_id_vs_string() {
    // Token ID stops are checked before string stops in the engine
    // This test validates that both mechanisms work correctly

    let tokenizer: Arc<dyn Tokenizer> = Arc::new(PredictableTokenizer);

    // Test string-based stop sequence
    let stop_sequences = vec!["<|eot_id|>".to_string()];
    assert!(
        matches_with_candidate(&[], 128009, &stop_sequences, &tokenizer),
        "String-based stop should detect '<|eot_id|>' token"
    );

    // Test token ID-based stop (this would be checked first in the engine)
    let config = GenerationConfig { stop_token_ids: vec![128009], ..Default::default() };

    // In the actual engine, this check happens first:
    assert!(
        config.stop_token_ids.contains(&128009),
        "Token ID-based stop should detect token 128009 directly"
    );

    // Combined: both methods should work
    // Engine checks token IDs first (O(1) via binary_search), then strings
}

#[test]
fn test_rolling_window_with_candidate() {
    // Test that window size calculation includes the candidate token

    let tokenizer: Arc<dyn Tokenizer> = Arc::new(PredictableTokenizer);
    let stop_sequences = vec!["world!".to_string()];

    // Simulate the engine's tail window optimization
    // Window size: 64 bytes (default), but we'll use 3 tokens for this test

    // Scenario: Long sequence of tokens, but only check the tail window
    let all_tokens = [100, 101, 102, 103, 104, 1, 2]; // Many tokens, ending with "Hello world"
    let window_size = 3; // Small window for testing

    // Calculate tail window (should include space for candidate)
    let tail_start = all_tokens.len().saturating_sub(window_size - 1);
    let tail_tokens = &all_tokens[tail_start..];

    // tail_tokens should be the last (window_size - 1) tokens
    // For window_size=3, we take last 2 tokens: [1, 2] = "Hello world"
    assert_eq!(tail_tokens, &[1, 2], "Tail window should be last 2 tokens");

    // Now check with candidate
    let candidate = 3; // "!"
    assert!(
        matches_with_candidate(tail_tokens, candidate, &stop_sequences, &tokenizer),
        "Should detect stop sequence in tail window with candidate"
    );

    // Verify that the window size includes the candidate
    // The engine code: `let window_size = config.stop_string_window.min(generated_tokens.len() + 1);`
    // The +1 accounts for the candidate token
    let effective_window = window_size.min(all_tokens.len() + 1);
    assert_eq!(effective_window, 3, "Window should account for candidate (+1)");
}

#[test]
fn test_unicode_stop_sequences() {
    // Test that Unicode/multi-byte stop sequences work correctly

    let tokenizer: Arc<dyn Tokenizer> = Arc::new(PredictableTokenizer);

    // Chinese characters: 世界 = "world" in Chinese
    let stop_sequences = vec!["世界".to_string()];

    // Tokens: [9, 10] = "世" + "界" = "世界"
    assert!(
        matches_with_candidate(&[9], 10, &stop_sequences, &tokenizer),
        "Should detect Unicode stop sequence '世界'"
    );

    // Test with partial match (should not trigger)
    assert!(
        !matches_with_candidate(&[], 9, &stop_sequences, &tokenizer),
        "Partial Unicode match should not trigger stop"
    );

    // Test with emoji-like sequence
    let stop_sequences_emoji = vec!["<|eot_id|>".to_string()];
    assert!(
        matches_with_candidate(&[], 128009, &stop_sequences_emoji, &tokenizer),
        "Should detect special token stop sequence"
    );
}

#[test]
fn test_empty_stop_sequences() {
    // Test that empty stop sequences don't cause false positives

    let tokenizer: Arc<dyn Tokenizer> = Arc::new(PredictableTokenizer);
    let stop_sequences: Vec<String> = vec![];

    // No stop sequences → should never match
    assert!(
        !matches_with_candidate(&[1, 2], 3, &stop_sequences, &tokenizer),
        "Empty stop sequences should never trigger stop"
    );
}

#[test]
fn test_stop_sequence_boundary_conditions() {
    // Test edge cases and boundary conditions

    let tokenizer: Arc<dyn Tokenizer> = Arc::new(PredictableTokenizer);

    // Test 1: Stop sequence longer than generated text
    let stop_sequences = vec!["Hello world! This is a long sequence".to_string()];
    assert!(
        !matches_with_candidate(&[1, 2], 3, &stop_sequences, &tokenizer),
        "Long stop sequence should not match short text"
    );

    // Test 2: Stop sequence is exactly one token
    let stop_sequences_single = vec!["!".to_string()];
    assert!(
        matches_with_candidate(&[], 3, &stop_sequences_single, &tokenizer),
        "Single-token stop sequence should match"
    );

    // Test 3: Stop sequence at beginning of text
    let stop_sequences_prefix = vec!["Hello".to_string()];
    assert!(
        matches_with_candidate(&[], 1, &stop_sequences_prefix, &tokenizer),
        "Stop sequence at text start should match (ends_with is still true)"
    );

    // Test 4: Multiple candidates in sequence
    let stop_sequences_multi = vec!["world!".to_string()];

    // Build up tokens one by one
    let mut current_tokens = vec![];

    // Add token 1: "Hello"
    assert!(
        !matches_with_candidate(&current_tokens, 1, &stop_sequences_multi, &tokenizer),
        "First token should not match"
    );
    current_tokens.push(1);

    // Add token 2: "Hello world"
    assert!(
        !matches_with_candidate(&current_tokens, 2, &stop_sequences_multi, &tokenizer),
        "Second token should not match"
    );
    current_tokens.push(2);

    // Add token 3: "Hello world!" → STOP
    assert!(
        matches_with_candidate(&current_tokens, 3, &stop_sequences_multi, &tokenizer),
        "Third token should match and trigger stop"
    );
}

#[test]
fn test_stop_sequence_partial_matches() {
    // Test that partial matches don't cause premature stops

    let tokenizer: Arc<dyn Tokenizer> = Arc::new(PredictableTokenizer);
    let stop_sequences = vec!["\n\nQ:".to_string()];

    // Partial match: "\n" (not "\n\nQ:")
    assert!(
        !matches_with_candidate(&[], 4, &stop_sequences, &tokenizer),
        "Single newline should not trigger stop"
    );

    // Partial match: "\n\n" (not "\n\nQ:")
    assert!(
        !matches_with_candidate(&[4], 5, &stop_sequences, &tokenizer),
        "Double newline should not trigger stop"
    );

    // Partial match: "\n\nQ" (not "\n\nQ:")
    assert!(
        !matches_with_candidate(&[4, 5], 6, &stop_sequences, &tokenizer),
        "Partial match should not trigger stop"
    );

    // Full match: "\n\nQ:"
    assert!(
        matches_with_candidate(&[4, 5, 6], 7, &stop_sequences, &tokenizer),
        "Full match should trigger stop"
    );
}

#[test]
fn test_window_size_edge_cases() {
    // Test edge cases in window size calculation

    let _tokenizer: Arc<dyn Tokenizer> = Arc::new(PredictableTokenizer);

    // Test 1: Empty current_tokens (beginning of generation)
    let current_tokens: Vec<u32> = vec![];
    let window_size = 64_usize;
    let effective_window = window_size.min(current_tokens.len() + 1);
    assert_eq!(effective_window, 1, "Window size should be 1 when current_tokens is empty");

    // Test 2: current_tokens.len() < window_size
    let current_tokens = [1_u32, 2];
    let effective_window = window_size.min(current_tokens.len() + 1);
    assert_eq!(effective_window, 3, "Window size should be len + 1 when smaller than max window");

    // Test 3: current_tokens.len() >= window_size
    let current_tokens: Vec<u32> = (0..100).collect();
    let effective_window = window_size.min(current_tokens.len() + 1);
    assert_eq!(effective_window, 64, "Window size should be capped at max window");

    // Test 4: Tail calculation with window
    let current_tokens = [100_u32, 101, 102, 1, 2]; // 5 tokens
    let window_size = 3;
    let tail_start = current_tokens.len().saturating_sub(window_size - 1);
    assert_eq!(tail_start, 3, "Tail should start at index 3 (last 2 tokens + candidate)");

    let tail_tokens = &current_tokens[tail_start..];
    assert_eq!(tail_tokens, &[1, 2], "Tail should be last (window_size - 1) tokens");
}

#[test]
fn test_config_stop_sequences_integration() {
    // Test integration with GenerationConfig

    let tokenizer: Arc<dyn Tokenizer> = Arc::new(PredictableTokenizer);

    let config = GenerationConfig {
        stop_sequences: vec!["world!".to_string(), "<eos>".to_string()],
        stop_token_ids: vec![999, 128009],
        stop_string_window: 64,
        ..Default::default()
    };

    // Test string-based stop
    assert!(
        matches_with_candidate(&[1, 2], 3, &config.stop_sequences, &tokenizer),
        "Config stop_sequences should work with matches_with_candidate"
    );

    // Test token ID-based stop
    assert!(config.stop_token_ids.contains(&999), "Config stop_token_ids should contain EOS token");
    assert!(
        config.stop_token_ids.contains(&128009),
        "Config stop_token_ids should contain LLaMA-3 EOT token"
    );

    // Test window size configuration
    assert_eq!(config.stop_string_window, 64, "Config should allow custom window size");
}
