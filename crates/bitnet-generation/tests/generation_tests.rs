//! Named integration tests for `bitnet-generation` covering the five SRP invariants
//! required by the extraction task.

use bitnet_generation::{
    GenerationConfig, GenerationStats, StopCriteria, StopReason, StreamEvent, TokenEvent,
    check_stop,
};
use proptest::prelude::*;

// ── helpers ───────────────────────────────────────────────────────────────

fn empty_criteria() -> StopCriteria {
    StopCriteria::default()
}

fn criteria_with_max(max: usize) -> StopCriteria {
    StopCriteria { max_tokens: max, ..Default::default() }
}

fn criteria_with_stop_id(id: u32) -> StopCriteria {
    StopCriteria { stop_token_ids: vec![id], ..Default::default() }
}

fn criteria_with_stop_string(s: &str) -> StopCriteria {
    StopCriteria { stop_strings: vec![s.to_string()], ..Default::default() }
}

// ── named proptest invariants ─────────────────────────────────────────────

proptest! {
    /// Stops exactly when generated token count reaches max_tokens.
    #[test]
    fn stop_checker_triggers_on_max_tokens(budget in 1usize..128usize) {
        let criteria = criteria_with_max(budget);
        let generated = vec![0u32; budget];
        let reason = check_stop(&criteria, 1, &generated, "");
        prop_assert_eq!(reason, Some(StopReason::MaxTokens));
    }

    /// Fires StopTokenId when the produced token is in the stop list.
    #[test]
    fn stop_checker_triggers_on_stop_token_id(id in 1u32..100_000u32) {
        let criteria = criteria_with_stop_id(id);
        let reason = check_stop(&criteria, id, &[], "");
        prop_assert_eq!(reason, Some(StopReason::StopTokenId(id)));
    }

    /// Returns None for any token when no stopping condition is set.
    #[test]
    fn stop_checker_never_triggers_on_empty_conditions(
        token in 0u32..100_000u32,
        gen_len in 0usize..64usize,
    ) {
        let criteria = empty_criteria(); // max_tokens=0, no stop IDs, no EOS, no strings
        let generated = vec![0u32; gen_len];
        let reason = check_stop(&criteria, token, &generated, "no-stop-here");
        prop_assert_eq!(reason, None);
    }

    /// All valid GenerationConfig fields must pass trivial range invariants.
    #[test]
    fn generation_config_valid_ranges(
        max_new_tokens in 1usize..4096usize,
        seed in prop::option::of(0u64..u64::MAX),
    ) {
        let cfg = GenerationConfig {
            max_new_tokens,
            seed,
            stop_criteria: StopCriteria {
                max_tokens: max_new_tokens,
                ..Default::default()
            },
        };
        prop_assert!(cfg.max_new_tokens > 0, "max_new_tokens must be positive");
        prop_assert_eq!(cfg.max_new_tokens, max_new_tokens);
        prop_assert_eq!(cfg.seed, seed);
    }

    /// A stop string present anywhere in the decoded tail triggers StopString.
    #[test]
    fn stop_checker_detects_string_sequence(
        prefix in "[a-z]{0,20}",
        suffix in "[a-z]{0,20}",
    ) {
        let stop = "</s>".to_string();
        let criteria = criteria_with_stop_string(&stop);
        let tail = format!("{prefix}</s>{suffix}");
        let reason = check_stop(&criteria, 1, &[], &tail);
        prop_assert_eq!(reason, Some(StopReason::StopString(stop)));
    }

    /// A token strictly below budget with no other triggers returns None.
    #[test]
    fn stop_checker_does_not_stop_below_budget(
        budget in 2usize..128usize,
        token in 1u32..50_000u32,
    ) {
        let criteria = criteria_with_max(budget);
        let generated = vec![0u32; budget - 1]; // one short of budget
        let reason = check_stop(&criteria, token, &generated, "");
        prop_assert_ne!(reason, Some(StopReason::MaxTokens));
    }
}

// ── TokenEvent / GenerationStats / StreamEvent unit tests ─────────────────

#[test]
fn token_event_fields_accessible() {
    let ev = TokenEvent { id: 42, text: "world".to_string() };
    assert_eq!(ev.id, 42);
    assert_eq!(ev.text, "world");
}

#[test]
fn token_event_empty_text() {
    let ev = TokenEvent { id: 0, text: String::new() };
    assert_eq!(ev.id, 0);
    assert!(ev.text.is_empty());
}

#[test]
fn generation_stats_fields_accessible() {
    let stats = GenerationStats { tokens_generated: 32, tokens_per_second: 8.0 };
    assert_eq!(stats.tokens_generated, 32);
    assert!((stats.tokens_per_second - 8.0).abs() < f64::EPSILON);
}

#[test]
fn generation_stats_default_is_zero() {
    let stats = GenerationStats::default();
    assert_eq!(stats.tokens_generated, 0);
    assert_eq!(stats.tokens_per_second, 0.0);
}

#[test]
fn stream_event_token_variant_holds_token_event() {
    let ev = StreamEvent::Token(TokenEvent { id: 7, text: "hi".to_string() });
    match ev {
        StreamEvent::Token(t) => {
            assert_eq!(t.id, 7);
            assert_eq!(t.text, "hi");
        }
        StreamEvent::Done { .. } => panic!("expected Token"),
    }
}

#[test]
fn stream_event_done_variant_holds_reason_and_stats() {
    let ev = StreamEvent::Done {
        reason: StopReason::MaxTokens,
        stats: GenerationStats { tokens_generated: 5, tokens_per_second: 2.5 },
    };
    match ev {
        StreamEvent::Done { reason, stats } => {
            assert_eq!(reason, StopReason::MaxTokens);
            assert_eq!(stats.tokens_generated, 5);
        }
        StreamEvent::Token(_) => panic!("expected Done"),
    }
}

// ── Multiple stop strings: first match in list wins ───────────────────────

#[test]
fn first_stop_string_in_list_triggers_first() {
    let criteria = StopCriteria {
        stop_token_ids: vec![],
        eos_token_id: None,
        max_tokens: 0,
        stop_strings: vec!["STOP".to_string(), "END".to_string()],
    };
    // Both strings are present in the tail; the one that appears first in the
    // stop_strings list should be the returned reason.
    let reason = check_stop(&criteria, 1, &[], "textSTOPandEND");
    assert_eq!(reason, Some(StopReason::StopString("STOP".to_string())));
}

#[test]
fn second_stop_string_fires_when_first_absent() {
    let criteria = StopCriteria {
        stop_token_ids: vec![],
        eos_token_id: None,
        max_tokens: 0,
        stop_strings: vec!["STOP".to_string(), "END".to_string()],
    };
    let reason = check_stop(&criteria, 1, &[], "textEND");
    assert_eq!(reason, Some(StopReason::StopString("END".to_string())));
}

#[test]
fn empty_stop_strings_list_never_fires() {
    let criteria = StopCriteria {
        stop_token_ids: vec![],
        eos_token_id: None,
        max_tokens: 0,
        stop_strings: vec![],
    };
    let reason = check_stop(&criteria, 1, &[], "anything goes here");
    assert_eq!(reason, None);
}

// ── GenerationConfig: seed is stored correctly ────────────────────────────

#[test]
fn generation_config_seed_some_stored() {
    let cfg = GenerationConfig {
        max_new_tokens: 10,
        seed: Some(999),
        stop_criteria: StopCriteria::default(),
    };
    assert_eq!(cfg.seed, Some(999));
}

#[test]
fn generation_config_seed_none_by_default() {
    assert!(GenerationConfig::default().seed.is_none());
}

// ── StopCriteria clone behaviour ──────────────────────────────────────────

#[test]
fn stop_criteria_clone_is_independent() {
    let original = StopCriteria {
        stop_token_ids: vec![1, 2, 3],
        stop_strings: vec!["</s>".to_string()],
        max_tokens: 10,
        eos_token_id: Some(2),
    };
    let mut cloned = original.clone();
    cloned.stop_token_ids.push(99);
    cloned.max_tokens = 999;
    // Original must be unchanged.
    assert_eq!(original.stop_token_ids, vec![1, 2, 3]);
    assert_eq!(original.max_tokens, 10);
}

// ── StopCriteria serde ────────────────────────────────────────────────────

#[test]
fn stop_criteria_serde_roundtrip_unit() {
    let criteria = StopCriteria {
        stop_token_ids: vec![128009, 2],
        stop_strings: vec!["</s>".to_string(), "[STOP]".to_string()],
        max_tokens: 64,
        eos_token_id: Some(2),
    };
    let json = serde_json::to_string(&criteria).unwrap();
    let restored: StopCriteria = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.stop_token_ids, criteria.stop_token_ids);
    assert_eq!(restored.stop_strings, criteria.stop_strings);
    assert_eq!(restored.max_tokens, criteria.max_tokens);
    assert_eq!(restored.eos_token_id, criteria.eos_token_id);
}

#[test]
fn generation_config_serde_roundtrip() {
    let cfg = GenerationConfig {
        max_new_tokens: 256,
        seed: Some(42),
        stop_criteria: StopCriteria {
            stop_token_ids: vec![128009],
            stop_strings: vec!["</s>".to_string()],
            max_tokens: 256,
            eos_token_id: Some(2),
        },
    };
    let json = serde_json::to_string(&cfg).unwrap();
    let back: GenerationConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.max_new_tokens, 256);
    assert_eq!(back.seed, Some(42));
    assert_eq!(back.stop_criteria.max_tokens, 256);
}

// ── StopReason equality and cloning ──────────────────────────────────────

#[test]
fn stop_reason_equality() {
    assert_eq!(StopReason::MaxTokens, StopReason::MaxTokens);
    assert_eq!(StopReason::EosToken, StopReason::EosToken);
    assert_eq!(StopReason::StopTokenId(42), StopReason::StopTokenId(42));
    assert_ne!(StopReason::StopTokenId(1), StopReason::StopTokenId(2));
    assert_eq!(
        StopReason::StopString("</s>".to_string()),
        StopReason::StopString("</s>".to_string()),
    );
    assert_ne!(StopReason::MaxTokens, StopReason::EosToken);
}

#[test]
fn stop_reason_clone_is_equal() {
    let r = StopReason::StopString("abc".to_string());
    assert_eq!(r.clone(), r);
}

// ── proptest: GenerationConfig serde preserves seed ──────────────────────

proptest! {
    #[test]
    fn generation_config_serde_preserves_seed(
        seed in prop::option::of(any::<u64>()),
        max_new_tokens in 1usize..512,
    ) {
        let cfg = GenerationConfig {
            max_new_tokens,
            seed,
            stop_criteria: StopCriteria::default(),
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: GenerationConfig = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(back.seed, seed);
        prop_assert_eq!(back.max_new_tokens, max_new_tokens);
    }

    /// TokenEvent serde preserves id and text.
    #[test]
    fn token_event_serde_roundtrip(id in any::<u32>(), text in "[a-zA-Z0-9 ]{0,32}") {
        let ev = TokenEvent { id, text: text.clone() };
        let json = serde_json::to_string(&ev).unwrap();
        let back: TokenEvent = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(back.id, id);
        prop_assert_eq!(back.text, text);
    }
}
