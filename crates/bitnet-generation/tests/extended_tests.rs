//! Extended integration tests for `bitnet-generation`.
//!
//! Covers:
//! - Unicode / UTF-8 safety in stop-string matching
//! - Very long context (10 k tokens) – no panic

use bitnet_generation::{StopCriteria, StopReason, check_stop};
use proptest::prelude::*;

// ── Unicode / UTF-8 safety ────────────────────────────────────────────────

/// A stop string made entirely of multi-byte UTF-8 characters is found
/// correctly in a tail that contains that stop string.
#[test]
fn unicode_stop_string_found_in_tail() {
    let stop = "こんにちは".to_string(); // 5 three-byte UTF-8 characters
    let criteria = StopCriteria {
        stop_token_ids: vec![],
        eos_token_id: None,
        max_tokens: 0,
        stop_strings: vec![stop.clone()],
    };
    let tail = format!("prefix text {stop} suffix text");
    let reason = check_stop(&criteria, 1, &[], &tail);
    assert_eq!(reason, Some(StopReason::StopString(stop)));
}

/// When the tail contains only ASCII and the stop string is multi-byte UTF-8,
/// no false positive is produced.
#[test]
fn unicode_stop_string_absent_no_false_positive() {
    let criteria = StopCriteria {
        stop_token_ids: vec![],
        eos_token_id: None,
        max_tokens: 0,
        stop_strings: vec!["こんにちは".to_string()],
    };
    let reason = check_stop(&criteria, 1, &[], "hello world, no japanese here");
    assert_eq!(reason, None);
}

/// Emoji and other 4-byte UTF-8 code-points in the tail don't cause panics
/// or false matches when the stop string is not present.
#[test]
fn emoji_in_tail_no_panic() {
    let criteria = StopCriteria {
        stop_token_ids: vec![],
        eos_token_id: None,
        max_tokens: 0,
        stop_strings: vec!["</s>".to_string()],
    };
    let tail = "Hello 🌍🎉🦀 world";
    let reason = check_stop(&criteria, 1, &[], tail);
    assert_eq!(reason, None);
}

/// A stop string that is itself emoji is detected correctly in a tail.
#[test]
fn emoji_stop_string_detected() {
    let stop = "🛑".to_string();
    let criteria = StopCriteria {
        stop_token_ids: vec![],
        eos_token_id: None,
        max_tokens: 0,
        stop_strings: vec![stop.clone()],
    };
    let tail = format!("some text {stop} and more");
    let reason = check_stop(&criteria, 1, &[], &tail);
    assert_eq!(reason, Some(StopReason::StopString(stop)));
}

proptest! {
    /// Arbitrary valid UTF-8 strings in the tail never cause a panic in
    /// `check_stop`, even when the stop string is a different Unicode sequence.
    #[test]
    fn unicode_tail_never_panics(
        tail in "\\PC{0,64}", // proptest's "any printable Unicode char", 0-64 chars
    ) {
        let criteria = StopCriteria {
            stop_token_ids: vec![],
            eos_token_id: None,
            max_tokens: 0,
            stop_strings: vec!["[STOP]".to_string()],
        };
        // We only care that this doesn't panic; we don't assert a specific reason.
        let _ = check_stop(&criteria, 1, &[], &tail);
        prop_assert!(true);
    }

    /// Arbitrary UTF-8 stop strings that appear literally in the tail are
    /// always detected.
    #[test]
    fn unicode_stop_string_always_detected_when_present(
        stop in "\\PC{1,8}", // non-empty printable Unicode, up to 8 chars
        prefix in "\\PC{0,16}",
        suffix in "\\PC{0,16}",
    ) {
        let criteria = StopCriteria {
            stop_token_ids: vec![],
            eos_token_id: None,
            max_tokens: 0,
            stop_strings: vec![stop.clone()],
        };
        let tail = format!("{prefix}{stop}{suffix}");
        let reason = check_stop(&criteria, 1, &[], &tail);
        prop_assert_eq!(reason, Some(StopReason::StopString(stop)));
    }
}

// ── Very long context (no panic) ──────────────────────────────────────────

/// `check_stop` must not panic when `generated` contains 10 000 token IDs.
#[test]
fn very_long_context_no_panic() {
    let criteria = StopCriteria {
        stop_token_ids: vec![],
        eos_token_id: None,
        max_tokens: 0,
        stop_strings: vec![],
    };
    let generated: Vec<u32> = (0..10_000).collect();
    let result = check_stop(&criteria, 99_999, &generated, "some output text");
    // With all conditions disabled the function must return None.
    assert_eq!(result, None);
}

/// With a budget cap set, `check_stop` still terminates correctly at exactly
/// 10 000 tokens – no panic, correct reason.
#[test]
fn very_long_context_with_budget_cap_no_panic() {
    const BUDGET: usize = 10_000;
    let criteria = StopCriteria {
        stop_token_ids: vec![],
        eos_token_id: None,
        max_tokens: BUDGET,
        stop_strings: vec![],
    };
    let generated: Vec<u32> = vec![1u32; BUDGET];
    let result = check_stop(&criteria, 99_999, &generated, "");
    assert_eq!(result, Some(StopReason::MaxTokens));
}

/// `check_stop` does not panic when presented with a very long decoded tail
/// (simulating a large accumulated output string).
#[test]
fn very_long_tail_no_panic() {
    let criteria = StopCriteria {
        stop_token_ids: vec![],
        eos_token_id: None,
        max_tokens: 0,
        stop_strings: vec!["[STOP]".to_string()],
    };
    let long_tail = "token ".repeat(10_000);
    let result = check_stop(&criteria, 1, &[], &long_tail);
    assert_eq!(result, None);
}

proptest! {
    /// `check_stop` never panics regardless of `generated` length up to 10 000.
    #[test]
    fn long_context_never_panics(
        context_len in 0usize..10_001usize,
        token in 0u32..100_000u32,
    ) {
        let criteria = StopCriteria {
            stop_token_ids: vec![],
            eos_token_id: None,
            max_tokens: 0,
            stop_strings: vec![],
        };
        let generated = vec![0u32; context_len];
        let _ = check_stop(&criteria, token, &generated, "");
        prop_assert!(true);
    }
}
