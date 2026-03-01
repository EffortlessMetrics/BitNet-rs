//! Edge-case tests for `bitnet-generation` config, stop criteria, and stream events.
//!
//! Targets gaps not covered by the existing test suite:
//! - Duplicate stop IDs/strings, case sensitivity, overlapping patterns
//! - GenerationConfig clone independence, zero max_new_tokens
//! - GenerationStats edge values (NaN, infinity, f64 extremes)
//! - Full stream-sequence invariants
//! - Serde with every StopReason variant inside StreamEvent::Done
//! - Whitespace-only and very-long stop strings/token text

use bitnet_generation::{
    GenerationConfig, GenerationStats, StopCriteria, StopReason, StreamEvent, TokenEvent,
    check_stop,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_criteria(
    stop_ids: &[u32],
    stop_strings: &[&str],
    max: usize,
    eos: Option<u32>,
) -> StopCriteria {
    StopCriteria {
        stop_token_ids: stop_ids.to_vec(),
        stop_strings: stop_strings.iter().map(|s| s.to_string()).collect(),
        max_tokens: max,
        eos_token_id: eos,
    }
}

// ---------------------------------------------------------------------------
// 1. GenerationConfig defaults — verify all default field values
// ---------------------------------------------------------------------------

#[test]
fn config_default_max_new_tokens_is_128() {
    let cfg = GenerationConfig::default();
    assert_eq!(cfg.max_new_tokens, 128);
}

#[test]
fn config_default_seed_is_none() {
    assert!(GenerationConfig::default().seed.is_none());
}

#[test]
fn config_default_stop_criteria_is_default() {
    let cfg = GenerationConfig::default();
    assert!(cfg.stop_criteria.stop_token_ids.is_empty());
    assert!(cfg.stop_criteria.stop_strings.is_empty());
    assert_eq!(cfg.stop_criteria.max_tokens, 0);
    assert!(cfg.stop_criteria.eos_token_id.is_none());
}

#[test]
fn config_with_zero_max_new_tokens() {
    let cfg = GenerationConfig {
        max_new_tokens: 0,
        seed: None,
        stop_criteria: StopCriteria::default(),
    };
    assert_eq!(cfg.max_new_tokens, 0);
}

#[test]
fn config_with_very_large_max_new_tokens() {
    let cfg = GenerationConfig {
        max_new_tokens: usize::MAX,
        seed: Some(u64::MAX),
        stop_criteria: StopCriteria::default(),
    };
    assert_eq!(cfg.max_new_tokens, usize::MAX);
    assert_eq!(cfg.seed, Some(u64::MAX));
}

// ---------------------------------------------------------------------------
// 2. GenerationConfig clone independence
// ---------------------------------------------------------------------------

#[test]
fn config_clone_is_independent() {
    let original = GenerationConfig {
        max_new_tokens: 64,
        seed: Some(42),
        stop_criteria: make_criteria(&[1, 2], &["stop"], 64, Some(2)),
    };
    let mut cloned = original.clone();
    cloned.max_new_tokens = 999;
    cloned.seed = Some(0);
    cloned.stop_criteria.stop_token_ids.push(99);
    // Original must be unchanged.
    assert_eq!(original.max_new_tokens, 64);
    assert_eq!(original.seed, Some(42));
    assert_eq!(original.stop_criteria.stop_token_ids, vec![1, 2]);
}

// ---------------------------------------------------------------------------
// 3. Duplicate stop token IDs and stop strings
// ---------------------------------------------------------------------------

#[test]
fn duplicate_stop_token_ids_still_fires() {
    let c = make_criteria(&[42, 42, 42], &[], 0, None);
    assert_eq!(check_stop(&c, 42, &[], ""), Some(StopReason::StopTokenId(42)));
}

#[test]
fn duplicate_stop_strings_returns_first_match() {
    let c = make_criteria(&[], &["end", "end"], 0, None);
    let result = check_stop(&c, 1, &[], "the end");
    assert_eq!(result, Some(StopReason::StopString("end".to_string())));
}

// ---------------------------------------------------------------------------
// 4. Stop string case sensitivity
// ---------------------------------------------------------------------------

#[test]
fn stop_string_is_case_sensitive_no_match() {
    let c = make_criteria(&[], &["STOP"], 0, None);
    assert!(check_stop(&c, 1, &[], "stop").is_none());
}

#[test]
fn stop_string_is_case_sensitive_exact_match() {
    let c = make_criteria(&[], &["STOP"], 0, None);
    assert_eq!(
        check_stop(&c, 1, &[], "STOP"),
        Some(StopReason::StopString("STOP".to_string()))
    );
}

#[test]
fn stop_string_mixed_case_no_match() {
    let c = make_criteria(&[], &["Stop"], 0, None);
    assert!(check_stop(&c, 1, &[], "STOP").is_none());
    assert!(check_stop(&c, 1, &[], "stop").is_none());
    assert_eq!(
        check_stop(&c, 1, &[], "Stop"),
        Some(StopReason::StopString("Stop".to_string()))
    );
}

// ---------------------------------------------------------------------------
// 5. Overlapping stop string patterns
// ---------------------------------------------------------------------------

#[test]
fn overlapping_pattern_aa_in_aaa() {
    let c = make_criteria(&[], &["aa"], 0, None);
    // "aaa" contains "aa" starting at index 0
    assert_eq!(
        check_stop(&c, 1, &[], "aaa"),
        Some(StopReason::StopString("aa".to_string()))
    );
}

#[test]
fn nested_stop_strings_shorter_first_in_list() {
    // "end" is checked before "ending" in the list
    let c = make_criteria(&[], &["end", "ending"], 0, None);
    assert_eq!(
        check_stop(&c, 1, &[], "the ending"),
        Some(StopReason::StopString("end".to_string()))
    );
}

#[test]
fn nested_stop_strings_longer_first_in_list() {
    // "ending" is checked before "end" — "ending" appears first
    let c = make_criteria(&[], &["ending", "end"], 0, None);
    assert_eq!(
        check_stop(&c, 1, &[], "the ending"),
        Some(StopReason::StopString("ending".to_string()))
    );
}

#[test]
fn stop_string_partial_match_does_not_trigger() {
    let c = make_criteria(&[], &["</s>"], 0, None);
    // Partial: only "</s" without the closing ">"
    assert!(check_stop(&c, 1, &[], "text</s text").is_none());
}

// ---------------------------------------------------------------------------
// 6. Whitespace-only stop strings
// ---------------------------------------------------------------------------

#[test]
fn stop_string_single_newline() {
    let c = make_criteria(&[], &["\n"], 0, None);
    assert_eq!(
        check_stop(&c, 1, &[], "line1\nline2"),
        Some(StopReason::StopString("\n".to_string()))
    );
}

#[test]
fn stop_string_tab_character() {
    let c = make_criteria(&[], &["\t"], 0, None);
    assert_eq!(
        check_stop(&c, 1, &[], "col1\tcol2"),
        Some(StopReason::StopString("\t".to_string()))
    );
}

#[test]
fn stop_string_whitespace_only_spaces() {
    let c = make_criteria(&[], &["   "], 0, None);
    assert_eq!(
        check_stop(&c, 1, &[], "word   word"),
        Some(StopReason::StopString("   ".to_string()))
    );
    // Two spaces should NOT match three-space stop
    assert!(check_stop(&c, 1, &[], "word  word").is_none());
}

// ---------------------------------------------------------------------------
// 7. Multi-byte UTF-8 stop strings — emoji sequences
// ---------------------------------------------------------------------------

#[test]
fn stop_string_emoji_sequence() {
    let c = make_criteria(&[], &["🔥🔥"], 0, None);
    assert_eq!(
        check_stop(&c, 1, &[], "hot 🔥🔥 stuff"),
        Some(StopReason::StopString("🔥🔥".to_string()))
    );
    // Single emoji should not match double
    assert!(check_stop(&c, 1, &[], "hot 🔥 stuff").is_none());
}

#[test]
fn stop_string_mixed_ascii_and_unicode() {
    let c = make_criteria(&[], &["end终"], 0, None);
    assert_eq!(
        check_stop(&c, 1, &[], "the end终 here"),
        Some(StopReason::StopString("end终".to_string()))
    );
}

#[test]
fn stop_string_zero_width_joiner_emoji() {
    // Family emoji (compound emoji with ZWJ)
    let stop = "👨\u{200D}👩\u{200D}👧";
    let c = make_criteria(&[], &[stop], 0, None);
    assert_eq!(
        check_stop(&c, 1, &[], &format!("family: {stop} !")),
        Some(StopReason::StopString(stop.to_string()))
    );
}

// ---------------------------------------------------------------------------
// 8. StopReason serde roundtrip — all variants
// ---------------------------------------------------------------------------

#[test]
fn stop_reason_max_tokens_serde() {
    let r = StopReason::MaxTokens;
    let json = serde_json::to_string(&r).unwrap();
    let r2: StopReason = serde_json::from_str(&json).unwrap();
    assert_eq!(r, r2);
}

#[test]
fn stop_reason_eos_token_serde() {
    let r = StopReason::EosToken;
    let json = serde_json::to_string(&r).unwrap();
    let r2: StopReason = serde_json::from_str(&json).unwrap();
    assert_eq!(r, r2);
}

#[test]
fn stop_reason_stop_token_id_serde_boundary_values() {
    for id in [0u32, 1, u32::MAX / 2, u32::MAX] {
        let r = StopReason::StopTokenId(id);
        let json = serde_json::to_string(&r).unwrap();
        let r2: StopReason = serde_json::from_str(&json).unwrap();
        assert_eq!(r, r2, "roundtrip failed for StopTokenId({id})");
    }
}

#[test]
fn stop_reason_stop_string_serde_with_special_chars() {
    for s in ["", "</s>", "\n\n", "🛑", "end终", "a\"b\\c"] {
        let r = StopReason::StopString(s.to_string());
        let json = serde_json::to_string(&r).unwrap();
        let r2: StopReason = serde_json::from_str(&json).unwrap();
        assert_eq!(r, r2, "roundtrip failed for StopString({s:?})");
    }
}

// ---------------------------------------------------------------------------
// 9. StreamEvent variants — serde with each StopReason in Done
// ---------------------------------------------------------------------------

#[test]
fn stream_event_done_with_each_stop_reason_serde() {
    let reasons = [
        StopReason::MaxTokens,
        StopReason::EosToken,
        StopReason::StopTokenId(42),
        StopReason::StopString("</s>".to_string()),
    ];
    let stats = GenerationStats { tokens_generated: 5, tokens_per_second: 10.0 };
    for reason in &reasons {
        let ev = StreamEvent::Done { reason: reason.clone(), stats: stats.clone() };
        let json = serde_json::to_string(&ev).unwrap();
        let ev2: StreamEvent = serde_json::from_str(&json).unwrap();
        match ev2 {
            StreamEvent::Done { reason: r2, stats: s2 } => {
                assert_eq!(*reason, r2, "reason mismatch for {reason:?}");
                assert_eq!(s2.tokens_generated, 5);
            }
            StreamEvent::Token(_) => panic!("expected Done, got Token"),
        }
    }
}

#[test]
fn stream_event_token_serde_with_empty_text() {
    let ev = StreamEvent::Token(TokenEvent { id: 0, text: String::new() });
    let json = serde_json::to_string(&ev).unwrap();
    let ev2: StreamEvent = serde_json::from_str(&json).unwrap();
    match ev2 {
        StreamEvent::Token(t) => {
            assert_eq!(t.id, 0);
            assert!(t.text.is_empty());
        }
        StreamEvent::Done { .. } => panic!("expected Token"),
    }
}

#[test]
fn stream_event_token_serde_with_unicode_text() {
    let ev = StreamEvent::Token(TokenEvent { id: 99, text: "héllo 世界 🦀".to_string() });
    let json = serde_json::to_string(&ev).unwrap();
    let ev2: StreamEvent = serde_json::from_str(&json).unwrap();
    match ev2 {
        StreamEvent::Token(t) => assert_eq!(t.text, "héllo 世界 🦀"),
        StreamEvent::Done { .. } => panic!("expected Token"),
    }
}

// ---------------------------------------------------------------------------
// 10. GenerationStats edge values
// ---------------------------------------------------------------------------

#[test]
fn stats_default_is_zeroed() {
    let s = GenerationStats::default();
    assert_eq!(s.tokens_generated, 0);
    assert_eq!(s.tokens_per_second, 0.0);
}

#[test]
fn stats_very_high_throughput() {
    let s = GenerationStats {
        tokens_generated: usize::MAX,
        tokens_per_second: f64::MAX,
    };
    assert_eq!(s.tokens_generated, usize::MAX);
    assert_eq!(s.tokens_per_second, f64::MAX);
}

#[test]
fn stats_nan_tokens_per_second_serde() {
    let s = GenerationStats { tokens_generated: 0, tokens_per_second: f64::NAN };
    let json = serde_json::to_string(&s).unwrap();
    // NaN serializes to null in serde_json — deserialization may fail or produce null.
    // The key invariant: serialization does not panic.
    assert!(!json.is_empty());
}

#[test]
fn stats_infinity_tokens_per_second_serde() {
    let s = GenerationStats {
        tokens_generated: 1,
        tokens_per_second: f64::INFINITY,
    };
    let json = serde_json::to_string(&s).unwrap();
    assert!(!json.is_empty());
}

#[test]
fn stats_clone_independence() {
    let s1 = GenerationStats { tokens_generated: 42, tokens_per_second: 10.5 };
    let s2 = s1.clone();
    assert_eq!(s1.tokens_generated, s2.tokens_generated);
    assert!((s1.tokens_per_second - s2.tokens_per_second).abs() < f64::EPSILON);
}

// ---------------------------------------------------------------------------
// 11. Full stream sequence invariant
// ---------------------------------------------------------------------------

#[test]
fn stream_sequence_tokens_then_done() {
    let events: Vec<StreamEvent> = vec![
        StreamEvent::Token(TokenEvent { id: 10, text: "Hello".to_string() }),
        StreamEvent::Token(TokenEvent { id: 20, text: " world".to_string() }),
        StreamEvent::Token(TokenEvent { id: 30, text: "!".to_string() }),
        StreamEvent::Done {
            reason: StopReason::EosToken,
            stats: GenerationStats { tokens_generated: 3, tokens_per_second: 100.0 },
        },
    ];

    // Count variants
    let token_count = events.iter().filter(|e| matches!(e, StreamEvent::Token(_))).count();
    let done_count = events.iter().filter(|e| matches!(e, StreamEvent::Done { .. })).count();
    assert_eq!(token_count, 3);
    assert_eq!(done_count, 1);

    // Done must be last
    assert!(matches!(events.last(), Some(StreamEvent::Done { .. })));

    // Stats match token count
    if let StreamEvent::Done { stats, .. } = &events[3] {
        assert_eq!(stats.tokens_generated, token_count);
    }
}

#[test]
fn stream_sequence_empty_generation_only_done() {
    let events: Vec<StreamEvent> = vec![StreamEvent::Done {
        reason: StopReason::MaxTokens,
        stats: GenerationStats { tokens_generated: 0, tokens_per_second: 0.0 },
    }];
    assert_eq!(events.len(), 1);
    assert!(matches!(&events[0], StreamEvent::Done { .. }));
}

// ---------------------------------------------------------------------------
// 12. StopCriteria combinations — stop tokens + stop strings together
// ---------------------------------------------------------------------------

#[test]
fn stop_tokens_and_strings_together_token_wins_when_matched() {
    let c = make_criteria(&[42], &["end"], 0, None);
    // Token 42 matches stop_token_ids
    assert_eq!(
        check_stop(&c, 42, &[], "end"),
        Some(StopReason::StopTokenId(42))
    );
}

#[test]
fn stop_tokens_and_strings_together_string_wins_when_token_unmatched() {
    let c = make_criteria(&[42], &["end"], 0, None);
    // Token 99 does not match stop_token_ids, but "end" is in the tail
    assert_eq!(
        check_stop(&c, 99, &[], "the end"),
        Some(StopReason::StopString("end".to_string()))
    );
}

#[test]
fn stop_tokens_and_eos_and_strings_and_max_all_set() {
    let c = make_criteria(&[100], &["<stop>"], 10, Some(50));

    // No match at all
    assert!(check_stop(&c, 1, &[1, 2], "hello").is_none());

    // Stop token match
    assert_eq!(
        check_stop(&c, 100, &[1, 2], "hello"),
        Some(StopReason::StopTokenId(100))
    );

    // EOS match (no stop token match)
    assert_eq!(
        check_stop(&c, 50, &[1, 2], "hello"),
        Some(StopReason::EosToken)
    );

    // Max tokens (no token/EOS match)
    assert_eq!(
        check_stop(&c, 1, &vec![0u32; 10], "hello"),
        Some(StopReason::MaxTokens)
    );

    // String match (no token/EOS/max match)
    assert_eq!(
        check_stop(&c, 1, &[1], "text<stop>more"),
        Some(StopReason::StopString("<stop>".to_string()))
    );
}

// ---------------------------------------------------------------------------
// 13. TokenEvent with very long text
// ---------------------------------------------------------------------------

#[test]
fn token_event_with_very_long_text() {
    let long_text = "a".repeat(100_000);
    let ev = TokenEvent { id: 1, text: long_text.clone() };
    assert_eq!(ev.text.len(), 100_000);

    // Serde roundtrip with long text
    let json = serde_json::to_string(&ev).unwrap();
    let ev2: TokenEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(ev2.text, long_text);
}

// ---------------------------------------------------------------------------
// 14. GenerationConfig serde roundtrip with complex criteria
// ---------------------------------------------------------------------------

#[test]
fn config_serde_roundtrip_complex() {
    let cfg = GenerationConfig {
        max_new_tokens: 512,
        seed: Some(u64::MAX),
        stop_criteria: StopCriteria {
            stop_token_ids: vec![0, 1, u32::MAX],
            stop_strings: vec![
                "</s>".to_string(),
                "\n\n".to_string(),
                "🛑".to_string(),
            ],
            max_tokens: 512,
            eos_token_id: Some(2),
        },
    };
    let json = serde_json::to_string(&cfg).unwrap();
    let cfg2: GenerationConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg.max_new_tokens, cfg2.max_new_tokens);
    assert_eq!(cfg.seed, cfg2.seed);
    assert_eq!(cfg.stop_criteria.stop_token_ids, cfg2.stop_criteria.stop_token_ids);
    assert_eq!(cfg.stop_criteria.stop_strings, cfg2.stop_criteria.stop_strings);
    assert_eq!(cfg.stop_criteria.max_tokens, cfg2.stop_criteria.max_tokens);
    assert_eq!(cfg.stop_criteria.eos_token_id, cfg2.stop_criteria.eos_token_id);
}

#[test]
fn config_serde_roundtrip_seed_none() {
    let cfg = GenerationConfig { max_new_tokens: 1, seed: None, stop_criteria: StopCriteria::default() };
    let json = serde_json::to_string(&cfg).unwrap();
    let cfg2: GenerationConfig = serde_json::from_str(&json).unwrap();
    assert!(cfg2.seed.is_none());
}

// ---------------------------------------------------------------------------
// 15. StopCriteria with empty generated slice and empty tail
// ---------------------------------------------------------------------------

#[test]
fn empty_generated_and_empty_tail_with_max_tokens_zero() {
    let c = make_criteria(&[], &[], 0, None);
    assert!(check_stop(&c, 0, &[], "").is_none());
}

#[test]
fn empty_generated_with_max_tokens_one_does_not_fire() {
    // generated is empty (0 tokens), max_tokens=1 → not yet reached
    let c = make_criteria(&[], &[], 1, None);
    assert!(check_stop(&c, 1, &[], "").is_none());
}

// ---------------------------------------------------------------------------
// 16. Stop string that equals the entire tail exactly
// ---------------------------------------------------------------------------

#[test]
fn stop_string_is_exact_tail() {
    let c = make_criteria(&[], &["hello world"], 0, None);
    assert_eq!(
        check_stop(&c, 1, &[], "hello world"),
        Some(StopReason::StopString("hello world".to_string()))
    );
}

// ---------------------------------------------------------------------------
// 17. Debug formatting doesn't panic
// ---------------------------------------------------------------------------

#[test]
fn all_types_debug_format_no_panic() {
    let _ = format!("{:?}", StopCriteria::default());
    let _ = format!("{:?}", StopReason::MaxTokens);
    let _ = format!("{:?}", StopReason::EosToken);
    let _ = format!("{:?}", StopReason::StopTokenId(42));
    let _ = format!("{:?}", StopReason::StopString("test".to_string()));
    let _ = format!("{:?}", GenerationConfig::default());
    let _ = format!("{:?}", GenerationStats::default());
    let _ = format!("{:?}", TokenEvent { id: 0, text: String::new() });
    let _ = format!(
        "{:?}",
        StreamEvent::Token(TokenEvent { id: 0, text: String::new() })
    );
    let _ = format!(
        "{:?}",
        StreamEvent::Done {
            reason: StopReason::MaxTokens,
            stats: GenerationStats::default(),
        }
    );
}
