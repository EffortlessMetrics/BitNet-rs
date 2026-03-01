//! Edge case and boundary tests for generation stop criteria.
//!
//! Tests exercise unusual inputs, priority ordering, and boundary conditions
//! for the `check_stop` function and related types.

use bitnet_generation::{
    GenerationConfig, GenerationStats, StopCriteria, StopReason, StreamEvent, TokenEvent,
    check_stop,
};

// --- check_stop priority and ordering ---

#[test]
fn stop_token_id_takes_priority_over_max_tokens() {
    let criteria = StopCriteria {
        stop_token_ids: vec![42],
        stop_strings: vec![],
        max_tokens: 1,
        eos_token_id: None,
    };
    // generated has 1 token (at max), AND token_id is a stop token
    let result = check_stop(&criteria, 42, &[42], "hello");
    assert!(matches!(result, Some(StopReason::StopTokenId(42))));
}

#[test]
fn stop_token_id_takes_priority_over_eos() {
    let criteria = StopCriteria {
        stop_token_ids: vec![42],
        stop_strings: vec![],
        max_tokens: 100,
        eos_token_id: Some(42),
    };
    // token_id matches both stop_token_ids and eos_token_id
    let result = check_stop(&criteria, 42, &[42], "");
    assert!(matches!(result, Some(StopReason::StopTokenId(42))));
}

#[test]
fn eos_takes_priority_over_max_tokens() {
    let criteria = StopCriteria {
        stop_token_ids: vec![],
        stop_strings: vec![],
        max_tokens: 1,
        eos_token_id: Some(99),
    };
    let result = check_stop(&criteria, 99, &[99], "");
    assert!(matches!(result, Some(StopReason::EosToken)));
}

#[test]
fn max_tokens_takes_priority_over_stop_strings() {
    let criteria = StopCriteria {
        stop_token_ids: vec![],
        stop_strings: vec!["stop".to_string()],
        max_tokens: 1,
        eos_token_id: None,
    };
    let result = check_stop(&criteria, 1, &[1], "stop");
    assert!(matches!(result, Some(StopReason::MaxTokens)));
}

// --- Empty criteria ---

#[test]
fn empty_criteria_never_stops() {
    let criteria = StopCriteria {
        stop_token_ids: vec![],
        stop_strings: vec![],
        max_tokens: 0,
        eos_token_id: None,
    };
    let result = check_stop(&criteria, 42, &[1, 2, 3], "hello world");
    assert!(result.is_none());
}

#[test]
fn zero_max_tokens_means_no_limit() {
    let criteria = StopCriteria {
        stop_token_ids: vec![],
        stop_strings: vec![],
        max_tokens: 0,
        eos_token_id: None,
    };
    let generated: Vec<u32> = (0..10000).collect();
    let result = check_stop(&criteria, 9999, &generated, "");
    assert!(result.is_none());
}

// --- Stop token ID edge cases ---

#[test]
fn multiple_stop_token_ids_first_match_wins() {
    let criteria = StopCriteria {
        stop_token_ids: vec![10, 20, 30],
        stop_strings: vec![],
        max_tokens: 100,
        eos_token_id: None,
    };
    let result = check_stop(&criteria, 20, &[20], "");
    assert!(matches!(result, Some(StopReason::StopTokenId(20))));
}

#[test]
fn stop_token_id_zero_works() {
    let criteria = StopCriteria {
        stop_token_ids: vec![0],
        stop_strings: vec![],
        max_tokens: 100,
        eos_token_id: None,
    };
    let result = check_stop(&criteria, 0, &[0], "");
    assert!(matches!(result, Some(StopReason::StopTokenId(0))));
}

#[test]
fn stop_token_id_u32_max_works() {
    let criteria = StopCriteria {
        stop_token_ids: vec![u32::MAX],
        stop_strings: vec![],
        max_tokens: 100,
        eos_token_id: None,
    };
    let result = check_stop(&criteria, u32::MAX, &[u32::MAX], "");
    assert!(matches!(result, Some(StopReason::StopTokenId(id)) if id == u32::MAX));
}

// --- Stop string edge cases ---

#[test]
fn stop_string_substring_match() {
    let criteria = StopCriteria {
        stop_token_ids: vec![],
        stop_strings: vec!["end".to_string()],
        max_tokens: 100,
        eos_token_id: None,
    };
    let result = check_stop(&criteria, 1, &[], "the end is near");
    assert!(matches!(result, Some(StopReason::StopString(s)) if s == "end"));
}

#[test]
fn stop_string_empty_string_always_matches() {
    let criteria = StopCriteria {
        stop_token_ids: vec![],
        stop_strings: vec!["".to_string()],
        max_tokens: 100,
        eos_token_id: None,
    };
    // Empty string is contained in any string
    let result = check_stop(&criteria, 1, &[], "anything");
    assert!(matches!(result, Some(StopReason::StopString(_))));
}

#[test]
fn stop_string_unicode_match() {
    let criteria = StopCriteria {
        stop_token_ids: vec![],
        stop_strings: vec!["终".to_string()],
        max_tokens: 100,
        eos_token_id: None,
    };
    let result = check_stop(&criteria, 1, &[], "这是终点");
    assert!(matches!(result, Some(StopReason::StopString(s)) if s == "终"));
}

#[test]
fn stop_string_no_match_returns_none() {
    let criteria = StopCriteria {
        stop_token_ids: vec![],
        stop_strings: vec!["stop".to_string()],
        max_tokens: 100,
        eos_token_id: None,
    };
    let result = check_stop(&criteria, 1, &[], "this does not match");
    assert!(result.is_none());
}

#[test]
fn multiple_stop_strings_first_match_wins() {
    let criteria = StopCriteria {
        stop_token_ids: vec![],
        stop_strings: vec!["alpha".to_string(), "beta".to_string()],
        max_tokens: 100,
        eos_token_id: None,
    };
    let result = check_stop(&criteria, 1, &[], "contains beta and alpha");
    // "alpha" is checked first in the list
    assert!(matches!(result, Some(StopReason::StopString(s)) if s == "alpha"));
}

// --- Max tokens boundary ---

#[test]
fn max_tokens_exact_boundary() {
    let criteria = StopCriteria {
        stop_token_ids: vec![],
        stop_strings: vec![],
        max_tokens: 5,
        eos_token_id: None,
    };
    // generated has exactly 4 tokens (one less than max)
    let result_under = check_stop(&criteria, 1, &[1, 2, 3, 4], "");
    assert!(result_under.is_none());

    // generated has exactly 5 tokens (at max)
    let result_at = check_stop(&criteria, 1, &[1, 2, 3, 4, 5], "");
    assert!(matches!(result_at, Some(StopReason::MaxTokens)));
}

#[test]
fn max_tokens_one() {
    let criteria = StopCriteria {
        stop_token_ids: vec![],
        stop_strings: vec![],
        max_tokens: 1,
        eos_token_id: None,
    };
    let result = check_stop(&criteria, 1, &[1], "");
    assert!(matches!(result, Some(StopReason::MaxTokens)));
}

// --- Type construction and cloning ---

#[test]
fn stop_criteria_default_does_not_stop() {
    let criteria = StopCriteria {
        stop_token_ids: vec![],
        stop_strings: vec![],
        max_tokens: 0,
        eos_token_id: None,
    };
    let result = check_stop(&criteria, 42, &[1, 2, 3, 4, 5], "hello world");
    assert!(result.is_none());
}

#[test]
fn generation_config_with_seed() {
    let config = GenerationConfig {
        max_new_tokens: 100,
        seed: Some(42),
        stop_criteria: StopCriteria {
            stop_token_ids: vec![],
            stop_strings: vec![],
            max_tokens: 100,
            eos_token_id: None,
        },
    };
    assert_eq!(config.seed, Some(42));
    assert_eq!(config.max_new_tokens, 100);
}

#[test]
fn generation_config_without_seed() {
    let config = GenerationConfig {
        max_new_tokens: 50,
        seed: None,
        stop_criteria: StopCriteria {
            stop_token_ids: vec![1, 2],
            stop_strings: vec!["stop".to_string()],
            max_tokens: 50,
            eos_token_id: Some(0),
        },
    };
    assert!(config.seed.is_none());
    assert_eq!(config.stop_criteria.stop_token_ids.len(), 2);
}

#[test]
fn token_event_with_empty_text() {
    let event = TokenEvent { id: 42, text: String::new() };
    assert_eq!(event.id, 42);
    assert!(event.text.is_empty());
}

#[test]
fn token_event_with_unicode() {
    let event = TokenEvent { id: 1, text: "🎉".to_string() };
    assert_eq!(event.text, "🎉");
}

#[test]
fn generation_stats_zero_values() {
    let stats = GenerationStats { tokens_generated: 0, tokens_per_second: 0.0 };
    assert_eq!(stats.tokens_generated, 0);
    assert_eq!(stats.tokens_per_second, 0.0);
}

#[test]
fn generation_stats_high_throughput() {
    let stats = GenerationStats { tokens_generated: 1_000_000, tokens_per_second: 50000.0 };
    assert_eq!(stats.tokens_generated, 1_000_000);
    assert!(stats.tokens_per_second > 0.0);
}

#[test]
fn stream_event_token_variant() {
    let event = StreamEvent::Token(TokenEvent { id: 1, text: "hello".to_string() });
    match event {
        StreamEvent::Token(t) => {
            assert_eq!(t.id, 1);
            assert_eq!(t.text, "hello");
        }
        _ => panic!("Expected Token variant"),
    }
}

#[test]
fn stream_event_done_variant() {
    let event = StreamEvent::Done {
        reason: StopReason::MaxTokens,
        stats: GenerationStats { tokens_generated: 10, tokens_per_second: 5.0 },
    };
    match event {
        StreamEvent::Done { reason, stats } => {
            assert!(matches!(reason, StopReason::MaxTokens));
            assert_eq!(stats.tokens_generated, 10);
        }
        _ => panic!("Expected Done variant"),
    }
}

// --- StopReason equality and debug ---

#[test]
fn stop_reason_variants_are_distinct() {
    let reasons = [
        StopReason::MaxTokens,
        StopReason::StopTokenId(1),
        StopReason::StopString("s".to_string()),
        StopReason::EosToken,
    ];
    for (i, a) in reasons.iter().enumerate() {
        for (j, b) in reasons.iter().enumerate() {
            if i == j {
                assert_eq!(a, b);
            } else {
                assert_ne!(a, b);
            }
        }
    }
}

#[test]
fn stop_reason_debug_is_non_empty() {
    let reasons = [
        StopReason::MaxTokens,
        StopReason::StopTokenId(42),
        StopReason::StopString("test".to_string()),
        StopReason::EosToken,
    ];
    for reason in &reasons {
        let debug = format!("{reason:?}");
        assert!(!debug.is_empty());
    }
}

// --- Combined scenario tests ---

#[test]
fn realistic_generation_scenario() {
    let criteria = StopCriteria {
        stop_token_ids: vec![2], // EOS-like
        stop_strings: vec!["<|endoftext|>".to_string()],
        max_tokens: 128,
        eos_token_id: Some(1),
    };

    // Normal token, no stop
    let r1 = check_stop(&criteria, 100, &[100], "Hello");
    assert!(r1.is_none());

    // Still going
    let r2 = check_stop(&criteria, 200, &[100, 200], "Hello world");
    assert!(r2.is_none());

    // Hit stop token
    let r3 = check_stop(&criteria, 2, &[100, 200, 2], "Hello world.");
    assert!(matches!(r3, Some(StopReason::StopTokenId(2))));
}

#[test]
fn eos_in_middle_of_generation() {
    let criteria = StopCriteria {
        stop_token_ids: vec![],
        stop_strings: vec![],
        max_tokens: 100,
        eos_token_id: Some(50256),
    };
    let result = check_stop(&criteria, 50256, &[1, 2, 3], "some text");
    assert!(matches!(result, Some(StopReason::EosToken)));
}
