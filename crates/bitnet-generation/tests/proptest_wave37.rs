//! Property-based tests — wave 37: stop sequence detection, token stream
//! ordering, generation budget tracking, and config invariants.

use bitnet_generation::{
    GenerationConfig, GenerationStats, StopCriteria, StopReason, StreamEvent, TokenEvent,
    check_stop,
};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn arb_stop_criteria() -> impl Strategy<Value = StopCriteria> {
    (
        proptest::collection::vec(0u32..100_000, 0..5),
        proptest::collection::vec("[a-z]{1,8}", 0..3),
        0usize..200,
        proptest::option::of(0u32..100_000),
    )
        .prop_map(|(ids, strings, max, eos)| StopCriteria {
            stop_token_ids: ids,
            stop_strings: strings,
            max_tokens: max,
            eos_token_id: eos,
        })
}

// ---------------------------------------------------------------------------
// Stop-sequence detection properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// If token_id is in stop_token_ids, check_stop always returns StopTokenId.
    #[test]
    fn stop_token_id_always_fires(
        token in 0u32..100_000,
        extra_ids in proptest::collection::vec(0u32..100_000, 0..4),
        max_tokens in 0usize..100,
        eos in proptest::option::of(0u32..100_000),
    ) {
        let mut ids = extra_ids;
        ids.push(token);
        let criteria = StopCriteria {
            stop_token_ids: ids,
            stop_strings: vec![],
            max_tokens,
            eos_token_id: eos,
        };
        let result = check_stop(&criteria, token, &[], "");
        prop_assert_eq!(result, Some(StopReason::StopTokenId(token)));
    }

    /// If token_id == eos and NOT in stop_token_ids, returns EosToken.
    #[test]
    fn eos_fires_when_not_in_stop_ids(token in 0u32..100_000) {
        let criteria = StopCriteria {
            stop_token_ids: vec![],
            stop_strings: vec![],
            max_tokens: 0,
            eos_token_id: Some(token),
        };
        let result = check_stop(&criteria, token, &[], "");
        prop_assert_eq!(result, Some(StopReason::EosToken));
    }

    /// StopTokenId takes priority over EosToken even when both match.
    #[test]
    fn stop_id_priority_over_eos(token in 0u32..100_000) {
        let criteria = StopCriteria {
            stop_token_ids: vec![token],
            stop_strings: vec![],
            max_tokens: 0,
            eos_token_id: Some(token),
        };
        let result = check_stop(&criteria, token, &[], "");
        prop_assert_eq!(result, Some(StopReason::StopTokenId(token)));
    }

    /// max_tokens fires when generated.len() >= max_tokens and max_tokens > 0.
    #[test]
    fn max_tokens_fires_at_boundary(budget in 1usize..=200) {
        let generated: Vec<u32> = (0..budget as u32).collect();
        let criteria = StopCriteria {
            stop_token_ids: vec![],
            stop_strings: vec![],
            max_tokens: budget,
            eos_token_id: None,
        };
        let result = check_stop(&criteria, 99999, &generated, "");
        prop_assert_eq!(result, Some(StopReason::MaxTokens));
    }

    /// max_tokens == 0 never triggers budget stop.
    #[test]
    fn zero_max_tokens_never_fires(gen_len in 0usize..100) {
        let generated: Vec<u32> = (0..gen_len as u32).collect();
        let criteria = StopCriteria {
            stop_token_ids: vec![],
            stop_strings: vec![],
            max_tokens: 0,
            eos_token_id: None,
        };
        let result = check_stop(&criteria, 42, &generated, "no stop");
        prop_assert!(result.is_none());
    }

    /// Stop string present in decoded_tail triggers StopString.
    #[test]
    fn stop_string_in_tail_fires(
        prefix in "[a-z]{0,20}",
        stop in "[a-z]{1,8}",
        suffix in "[a-z]{0,20}",
    ) {
        let tail = format!("{prefix}{stop}{suffix}");
        let criteria = StopCriteria {
            stop_token_ids: vec![],
            stop_strings: vec![stop.clone()],
            max_tokens: 0,
            eos_token_id: None,
        };
        let result = check_stop(&criteria, 42, &[], &tail);
        prop_assert_eq!(result, Some(StopReason::StopString(stop)));
    }

    /// When no criteria match, check_stop returns None.
    #[test]
    fn no_match_returns_none(
        token in 1000u32..2000,
        gen_len in 0usize..50,
    ) {
        let generated: Vec<u32> = (0..gen_len as u32).collect();
        let criteria = StopCriteria {
            stop_token_ids: vec![99999],
            stop_strings: vec!["XYZZY".to_string()],
            max_tokens: 10000,
            eos_token_id: Some(99998),
        };
        let result = check_stop(&criteria, token, &generated, "hello world");
        prop_assert!(result.is_none());
    }

    /// check_stop is deterministic: same inputs always produce same output.
    #[test]
    fn check_stop_deterministic(
        criteria in arb_stop_criteria(),
        token in 0u32..100_000,
        gen_len in 0usize..20,
    ) {
        let generated: Vec<u32> = (0..gen_len as u32).collect();
        let r1 = check_stop(&criteria, token, &generated, "tail");
        let r2 = check_stop(&criteria, token, &generated, "tail");
        prop_assert_eq!(r1, r2);
    }
}

// ---------------------------------------------------------------------------
// Generation budget tracking
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Below-budget generation never fires MaxTokens.
    #[test]
    fn below_budget_no_max_tokens(budget in 2usize..=100, deficit in 1usize..=100) {
        prop_assume!(deficit < budget);
        let gen_len = budget - deficit;
        let generated: Vec<u32> = (0..gen_len as u32).collect();
        let criteria = StopCriteria {
            stop_token_ids: vec![],
            stop_strings: vec![],
            max_tokens: budget,
            eos_token_id: None,
        };
        let result = check_stop(&criteria, 42, &generated, "");
        prop_assert!(result.is_none());
    }

    /// Over-budget generation always fires MaxTokens.
    #[test]
    fn over_budget_always_fires(budget in 1usize..=100, excess in 0usize..=50) {
        let gen_len = budget + excess;
        let generated: Vec<u32> = (0..gen_len as u32).collect();
        let criteria = StopCriteria {
            stop_token_ids: vec![],
            stop_strings: vec![],
            max_tokens: budget,
            eos_token_id: None,
        };
        let result = check_stop(&criteria, 42, &generated, "");
        prop_assert_eq!(result, Some(StopReason::MaxTokens));
    }
}

// ---------------------------------------------------------------------------
// GenerationConfig defaults
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Default GenerationConfig has positive max_new_tokens.
    #[test]
    fn default_config_positive_max_tokens(_seed in 0u32..10) {
        let cfg = GenerationConfig::default();
        prop_assert!(cfg.max_new_tokens > 0);
    }

    /// Default GenerationConfig seed is None.
    #[test]
    fn default_config_seed_none(_seed in 0u32..10) {
        let cfg = GenerationConfig::default();
        prop_assert!(cfg.seed.is_none());
    }
}

// ---------------------------------------------------------------------------
// GenerationStats TPS
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// tokens_per_second is non-negative.
    #[test]
    fn tps_non_negative(tokens in 0usize..=10_000, tps in 0.0f64..=1e6) {
        let stats = GenerationStats { tokens_generated: tokens, tokens_per_second: tps };
        prop_assert!(stats.tokens_per_second >= 0.0);
    }

    /// GenerationStats default is zero tokens and zero TPS.
    #[test]
    fn stats_default_is_zero(_seed in 0u32..10) {
        let stats = GenerationStats::default();
        prop_assert_eq!(stats.tokens_generated, 0);
        prop_assert!((stats.tokens_per_second - 0.0).abs() < 1e-10);
    }
}

// ---------------------------------------------------------------------------
// StopReason serde round-trip
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// StopReason survives JSON round-trip.
    #[test]
    fn stop_reason_json_roundtrip(variant in prop_oneof![
        Just(StopReason::MaxTokens),
        Just(StopReason::EosToken),
        (0u32..100_000).prop_map(StopReason::StopTokenId),
        "[a-z]{1,16}".prop_map(StopReason::StopString),
    ]) {
        let json = serde_json::to_string(&variant).unwrap();
        let back: StopReason = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(back, variant);
    }

    /// StopCriteria survives JSON round-trip.
    #[test]
    fn stop_criteria_json_roundtrip(criteria in arb_stop_criteria()) {
        let json = serde_json::to_string(&criteria).unwrap();
        let back: StopCriteria = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(back.stop_token_ids, criteria.stop_token_ids);
        prop_assert_eq!(back.stop_strings, criteria.stop_strings);
        prop_assert_eq!(back.max_tokens, criteria.max_tokens);
        prop_assert_eq!(back.eos_token_id, criteria.eos_token_id);
    }

    /// GenerationConfig survives JSON round-trip.
    #[test]
    fn generation_config_json_roundtrip(
        max_tokens in 1usize..=10_000,
        seed in proptest::option::of(0u64..u64::MAX),
    ) {
        let cfg = GenerationConfig {
            max_new_tokens: max_tokens,
            seed,
            stop_criteria: StopCriteria::default(),
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: GenerationConfig = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(back.max_new_tokens, max_tokens);
        prop_assert_eq!(back.seed, seed);
    }
}

// ---------------------------------------------------------------------------
// StreamEvent / TokenEvent properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// TokenEvent preserves id and text.
    #[test]
    fn token_event_preserves_fields(id in 0u32..100_000, text in "[a-z ]{0,20}") {
        let ev = TokenEvent { id, text: text.clone() };
        prop_assert_eq!(ev.id, id);
        prop_assert_eq!(ev.text, text);
    }

    /// StreamEvent::Token wraps a TokenEvent correctly.
    #[test]
    fn stream_event_token_wraps(id in 0u32..100_000, text in "[a-z]{1,8}") {
        let ev = StreamEvent::Token(TokenEvent { id, text: text.clone() });
        match ev {
            StreamEvent::Token(te) => {
                prop_assert_eq!(te.id, id);
                prop_assert_eq!(te.text, text);
            }
            _ => prop_assert!(false, "expected Token variant"),
        }
    }

    /// StreamEvent::Done carries the stop reason.
    #[test]
    fn stream_event_done_carries_reason(tokens in 0usize..1000, tps in 0.0f64..=1e6) {
        let ev = StreamEvent::Done {
            reason: StopReason::MaxTokens,
            stats: GenerationStats { tokens_generated: tokens, tokens_per_second: tps },
        };
        match ev {
            StreamEvent::Done { reason, stats } => {
                prop_assert_eq!(reason, StopReason::MaxTokens);
                prop_assert_eq!(stats.tokens_generated, tokens);
            }
            _ => prop_assert!(false, "expected Done variant"),
        }
    }
}
