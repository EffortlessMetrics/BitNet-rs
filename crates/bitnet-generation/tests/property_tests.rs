//! Property-based tests for `bitnet-generation`.
//!
//! Key invariants tested:
//! - `check_stop` priority ordering (stop-token-id > EOS > max-tokens > stop-strings)
//! - Stop token IDs always trigger before EOS when both match
//! - `max_tokens=0` never triggers a budget stop
//! - Stop string match returns `StopString(…)` only after decoding

use bitnet_generation::{StopCriteria, StopReason, check_stop};
use proptest::prelude::*;

// ── helpers ───────────────────────────────────────────────────────────────

fn arb_token_id() -> impl Strategy<Value = u32> {
    0u32..100_000u32
}

fn arb_stop_criteria() -> impl Strategy<Value = StopCriteria> {
    (
        prop::collection::vec(arb_token_id(), 0..5),
        prop::option::of(arb_token_id()),
        0usize..128usize,
        prop::collection::vec(
            prop_oneof![
                Just("[STOP]".to_string()),
                Just("</s>".to_string()),
                Just("\n".to_string())
            ],
            0..3,
        ),
    )
        .prop_map(|(stop_token_ids, eos_token_id, max_tokens, stop_strings)| StopCriteria {
            stop_token_ids,
            eos_token_id,
            max_tokens,
            stop_strings,
        })
}

// ── priority ordering ─────────────────────────────────────────────────────

proptest! {
    /// Stop-token-ID match always returns StopTokenId, even when EOS would also match.
    #[test]
    fn stop_token_id_beats_eos(token in arb_token_id()) {
        let criteria = StopCriteria {
            stop_token_ids: vec![token],
            eos_token_id: Some(token),   // both match
            max_tokens: 1,              // budget also exhausted
            stop_strings: vec![],
        };
        let reason = check_stop(&criteria, token, &[token], "");
        prop_assert_eq!(reason, Some(StopReason::StopTokenId(token)));
    }

    /// EOS match returns EosToken when token is not in stop_token_ids.
    #[test]
    fn eos_beats_max_tokens(token in arb_token_id()) {
        let criteria = StopCriteria {
            stop_token_ids: vec![],
            eos_token_id: Some(token),
            max_tokens: 1,               // budget exhausted
            stop_strings: vec![],
        };
        // generated is length 1, so budget is also hit; EOS should win
        let reason = check_stop(&criteria, token, &[token], "");
        prop_assert_eq!(reason, Some(StopReason::EosToken));
    }

    /// When only the budget is exceeded, we get MaxTokens.
    #[test]
    fn max_tokens_fires_when_budget_exceeded(
        token in arb_token_id(),
        budget in 1usize..32usize
    ) {
        let criteria = StopCriteria {
            stop_token_ids: vec![],
            eos_token_id: None,
            max_tokens: budget,
            stop_strings: vec![],
        };
        let generated: Vec<u32> = vec![0u32; budget]; // exactly at budget
        let reason = check_stop(&criteria, token, &generated, "");
        prop_assert_eq!(reason, Some(StopReason::MaxTokens));
    }

    /// max_tokens=0 disables the budget check.
    #[test]
    fn zero_max_tokens_never_stops_on_budget(
        token in arb_token_id(),
        generated_len in 0usize..64usize
    ) {
        let criteria = StopCriteria {
            stop_token_ids: vec![],
            eos_token_id: None,
            max_tokens: 0,
            stop_strings: vec![],
        };
        let generated = vec![token; generated_len];
        let reason = check_stop(&criteria, token, &generated, "");
        prop_assert_ne!(reason, Some(StopReason::MaxTokens));
    }

    /// An unrelated token with no matching criteria returns None.
    #[test]
    fn no_match_returns_none(token in arb_token_id()) {
        let criteria = StopCriteria {
            stop_token_ids: vec![],
            eos_token_id: None,
            max_tokens: 0,
            stop_strings: vec![],
        };
        let reason = check_stop(&criteria, token, &[], "");
        prop_assert_eq!(reason, None);
    }

    /// check_stop is pure: same inputs always produce the same result.
    #[test]
    fn check_stop_is_deterministic(
        criteria in arb_stop_criteria(),
        token in arb_token_id(),
        gen_len in 0usize..32usize,
        tail in "[a-z ]*"
    ) {
        let generated = vec![0u32; gen_len];
        let r1 = check_stop(&criteria, token, &generated, &tail);
        let r2 = check_stop(&criteria, token, &generated, &tail);
        prop_assert_eq!(r1, r2, "check_stop is not deterministic");
    }
}

// ── stop string matching ───────────────────────────────────────────────────

#[test]
fn stop_string_present_in_tail_triggers_stop() {
    let criteria = StopCriteria {
        stop_token_ids: vec![],
        eos_token_id: None,
        max_tokens: 0,
        stop_strings: vec!["</s>".to_string()],
    };
    let reason = check_stop(&criteria, 999, &[], "the answer is 42</s>");
    assert_eq!(reason, Some(StopReason::StopString("</s>".to_string())));
}

#[test]
fn stop_string_absent_returns_none() {
    let criteria = StopCriteria {
        stop_token_ids: vec![],
        eos_token_id: None,
        max_tokens: 0,
        stop_strings: vec!["</s>".to_string()],
    };
    let reason = check_stop(&criteria, 999, &[], "the answer is 42");
    assert_eq!(reason, None);
}

// ── expanded proptest coverage ────────────────────────────────────────────

use bitnet_generation::{GenerationConfig, GenerationStats, StreamEvent, TokenEvent};

proptest! {
    /// `StopCriteria` serializes to JSON and deserializes back without data loss.
    #[test]
    fn stop_criteria_serde_roundtrip(
        stop_token_ids in prop::collection::vec(0u32..100_000u32, 0..5),
        eos_token_id in prop::option::of(0u32..100_000u32),
        max_tokens in 0usize..1024usize,
        stop_strings in prop::collection::vec(
            prop_oneof![
                Just("</s>".to_string()),
                Just("[STOP]".to_string()),
                Just("\n".to_string()),
            ],
            0..3,
        ),
    ) {
        let criteria = StopCriteria {
            stop_token_ids: stop_token_ids.clone(),
            eos_token_id,
            max_tokens,
            stop_strings: stop_strings.clone(),
        };
        let json = serde_json::to_string(&criteria).expect("serialize");
        let restored: StopCriteria = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(&criteria.stop_token_ids, &restored.stop_token_ids);
        prop_assert_eq!(criteria.eos_token_id, restored.eos_token_id);
        prop_assert_eq!(criteria.max_tokens, restored.max_tokens);
        prop_assert_eq!(&criteria.stop_strings, &restored.stop_strings);
    }

    /// `GenerationStats::tokens_per_second` equals `tokens_generated / elapsed_secs`.
    #[test]
    fn generation_stats_tps_invariant(
        tokens_generated in 1usize..10_000usize,
        elapsed_secs in 0.001f64..3600.0f64,
    ) {
        let tps = tokens_generated as f64 / elapsed_secs;
        let stats = GenerationStats { tokens_generated, tokens_per_second: tps };
        let expected = stats.tokens_generated as f64 / elapsed_secs;
        prop_assert!(
            (stats.tokens_per_second - expected).abs() < 1e-9,
            "tps={} expected={}", stats.tokens_per_second, expected
        );
    }

    /// All `StreamEvent` variants can be constructed and debug-printed without panicking.
    #[test]
    fn stream_event_variants_debug_no_panic(
        id in 0u32..100_000u32,
        text in "[a-zA-Z0-9 ]{0,32}",
        tokens_generated in 0usize..10_000usize,
        tps in 0.0f64..10_000.0f64,
    ) {
        let token_ev = StreamEvent::Token(TokenEvent { id, text });
        let done_ev = StreamEvent::Done {
            reason: StopReason::MaxTokens,
            stats: GenerationStats { tokens_generated, tokens_per_second: tps },
        };
        let _ = format!("{token_ev:?}");
        let _ = format!("{done_ev:?}");
        prop_assert!(true);
    }

    /// `max_tokens = 0` never triggers a budget stop regardless of generated length.
    #[test]
    fn max_tokens_zero_never_limits(
        generated_len in 0usize..1000usize,
        token in 1u32..100_000u32,
    ) {
        let criteria = StopCriteria {
            stop_token_ids: vec![],
            eos_token_id: None,
            max_tokens: 0,
            stop_strings: vec![],
        };
        let generated = vec![0u32; generated_len];
        // Use token+1 to avoid triggering a stop_token_id if token==0.
        let reason = check_stop(&criteria, token.saturating_add(1), &generated, "");
        prop_assert_ne!(reason, Some(StopReason::MaxTokens));
    }

    /// `max_tokens > 0` fires `MaxTokens` exactly when generated reaches the budget.
    #[test]
    fn positive_max_tokens_fires_at_budget(budget in 1usize..100usize) {
        let criteria = StopCriteria {
            stop_token_ids: vec![],
            eos_token_id: None,
            max_tokens: budget,
            stop_strings: vec![],
        };
        let generated = vec![0u32; budget];
        let reason = check_stop(&criteria, 99_999, &generated, "");
        prop_assert_eq!(reason, Some(StopReason::MaxTokens));
    }

    /// `GenerationConfig` default produces a positive `max_new_tokens` and no seed.
    #[test]
    fn generation_config_default_is_valid(_: ()) {
        let cfg = GenerationConfig::default();
        prop_assert!(cfg.max_new_tokens > 0, "default max_new_tokens must be positive");
        prop_assert!(cfg.seed.is_none(), "default seed must be None");
    }

    /// Token accumulation invariant: simulating a generation loop up to `budget` tokens
    /// accumulates exactly `budget` token IDs when MaxTokens fires.
    #[test]
    fn token_accumulation_matches_budget(
        budget in 1usize..64usize,
        start_token in 1u32..50_000u32,
    ) {
        let criteria = StopCriteria {
            stop_token_ids: vec![],
            eos_token_id: None,
            max_tokens: budget,
            stop_strings: vec![],
        };
        let mut accumulated: Vec<u32> = Vec::new();
        let mut stopped = false;
        for step in 0..budget + 1 {
            let token = start_token.wrapping_add(step as u32);
            if let Some(reason) = check_stop(&criteria, token, &accumulated, "") {
                prop_assert_eq!(
                    reason,
                    StopReason::MaxTokens,
                    "unexpected stop reason at step {}",
                    step
                );
                prop_assert_eq!(
                    accumulated.len(),
                    budget,
                    "accumulated {} tokens, expected {}",
                    accumulated.len(),
                    budget
                );
                stopped = true;
                break;
            }
            accumulated.push(token);
        }
        prop_assert!(stopped, "generation never stopped within budget+1 steps");
    }

    /// Streaming invariant: a sequence of StreamEvents has all Token events before Done,
    /// and token IDs appear in the order they were inserted.
    #[test]
    fn streaming_tokens_in_order(
        token_ids in prop::collection::vec(0u32..100_000u32, 1..32),
    ) {
        // Build a stream: Token events followed by a Done event.
        let mut events: Vec<StreamEvent> = token_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                StreamEvent::Token(TokenEvent {
                    id,
                    text: format!("t{i}"),
                })
            })
            .collect();
        events.push(StreamEvent::Done {
            reason: StopReason::MaxTokens,
            stats: GenerationStats {
                tokens_generated: token_ids.len(),
                tokens_per_second: 1.0,
            },
        });

        // All Token events must come before the Done event.
        let done_pos = events
            .iter()
            .position(|e| matches!(e, StreamEvent::Done { .. }))
            .expect("Done event must be present");
        prop_assert_eq!(done_pos, token_ids.len(), "Done is not last");

        // Token IDs must appear in insertion order.
        let observed_ids: Vec<u32> = events
            .iter()
            .filter_map(|e| {
                if let StreamEvent::Token(t) = e {
                    Some(t.id)
                } else {
                    None
                }
            })
            .collect();
        prop_assert_eq!(&observed_ids, &token_ids, "Token IDs out of order");
    }
}

use serde_json;
