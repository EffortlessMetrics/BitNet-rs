//! BDD-style scenario tests for `bitnet-generation` stopping criteria.
//!
//! Each test follows the **Given / When / Then** structure.  All scenarios are
//! fast (no I/O, no external models) and complete in milliseconds.
//!
//! # Covered scenarios
//! - Given `stop_token_id` in criteria, When generated token matches, Then `StopTokenId` returned
//! - Given `stop_token_id` in criteria, When generated token differs, Then None returned
//! - Given `stop_sequence` = "\n\n", When decoded tail contains "\n\n", Then `StopString` returned
//! - Given `stop_sequence` present, When tail does not contain it, Then None returned
//! - Given `max_tokens` = 5, When exactly 5 tokens generated, Then `MaxTokens` returned
//! - Given `max_tokens` = 5, When only 4 tokens generated, Then None returned
//! - Given `max_tokens` = 0, When any number of tokens generated, Then no budget limit fires
//! - Given `eos_token_id` set, When produced token matches EOS, Then `EosToken` returned
//! - Given `stop_token_id` has priority over EOS, When both match, Then `StopTokenId` returned
//! - Given multiple stop conditions, When first applicable fires, Then returns its reason
//! - Given no conditions, When any token produced, Then None always returned
//! - Given `GenerationConfig` default, When inspected, Then sensible values

use bitnet_generation::{
    GenerationConfig, GenerationStats, StopCriteria, StopReason, StreamEvent, TokenEvent,
    check_stop,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn criteria(
    stop_ids: &[u32],
    stop_strings: &[&str],
    max_tokens: usize,
    eos: Option<u32>,
) -> StopCriteria {
    StopCriteria {
        stop_token_ids: stop_ids.to_vec(),
        stop_strings: stop_strings.iter().map(std::string::ToString::to_string).collect(),
        max_tokens,
        eos_token_id: eos,
    }
}

// ── Stop token ID ─────────────────────────────────────────────────────────────

/// Given: `stop_token_ids` = [`128_009`] (LLaMA-3 <|`eot_id`|>)
/// When: generated token is `128_009`
/// Then: `check_stop` returns `StopTokenId(128_009)`
#[test]
fn given_stop_token_id_when_generated_token_matches_then_stop_token_id_returned() {
    let c = criteria(&[128_009], &[], 100, None);
    let reason = check_stop(&c, 128_009, &[], "");
    assert_eq!(
        reason,
        Some(StopReason::StopTokenId(128_009)),
        "stop_token_id must fire when the produced token matches"
    );
}

/// Given: `stop_token_ids` = [`128_009`]
/// When: generated token is 42 (does not match)
/// Then: `check_stop` returns None
#[test]
fn given_stop_token_id_when_generated_token_differs_then_none_returned() {
    let c = criteria(&[128_009], &[], 100, None);
    let reason = check_stop(&c, 42, &[1, 2, 3], "some text");
    assert_eq!(reason, None, "stop_token_id must not fire when the token does not match");
}

/// Given: multiple `stop_token_ids` = [1, 2, 3]
/// When: generated token is 2
/// Then: `check_stop` returns `StopTokenId(2)`
#[test]
fn given_multiple_stop_token_ids_when_one_matches_then_fires() {
    let c = criteria(&[1, 2, 3], &[], 100, None);
    let reason = check_stop(&c, 2, &[], "");
    assert_eq!(reason, Some(StopReason::StopTokenId(2)));
}

/// Given: `stop_token_ids` = [5]
/// When: generated token is 5 and generated list already has many tokens
/// Then: `stop_token_id` has highest priority (fires before `max_tokens` check)
#[test]
fn given_stop_token_id_when_also_at_max_then_stop_token_id_has_priority() {
    // Both stop_token_id and max_tokens would fire; stop_token_id (priority 1) must win.
    let generated = vec![0u32; 10]; // max_tokens=10 would also fire
    let c = criteria(&[5], &[], 10, None);
    let reason = check_stop(&c, 5, &generated, "");
    assert_eq!(
        reason,
        Some(StopReason::StopTokenId(5)),
        "stop_token_id has higher priority than max_tokens"
    );
}

// ── EOS token ────────────────────────────────────────────────────────────────

/// Given: `eos_token_id` = 2
/// When: generated token is 2
/// Then: `check_stop` returns `EosToken`
#[test]
fn given_eos_token_id_when_generated_token_matches_then_eos_token_returned() {
    let c = criteria(&[], &[], 100, Some(2));
    let reason = check_stop(&c, 2, &[1], "some text");
    assert_eq!(reason, Some(StopReason::EosToken), "EOS token must trigger EosToken reason");
}

/// Given: `eos_token_id` = 2
/// When: generated token is 3
/// Then: `check_stop` returns None
#[test]
fn given_eos_token_id_when_token_differs_then_none_returned() {
    let c = criteria(&[], &[], 100, Some(2));
    let reason = check_stop(&c, 3, &[], "");
    assert_eq!(reason, None, "EOS check must not fire when token does not match eos_token_id");
}

/// Given: `stop_token_ids` = [`128_009`] AND `eos_token_id` = `128_009` (same token)
/// When: generated token is `128_009`
/// Then: `StopTokenId` fires (`stop_token_id` list checked before EOS)
#[test]
fn given_stop_token_id_same_as_eos_when_fires_then_stop_token_id_returned() {
    let c = criteria(&[128_009], &[], 100, Some(128_009));
    let reason = check_stop(&c, 128_009, &[], "");
    // stop_token_ids is checked before EOS in the priority order.
    assert_eq!(
        reason,
        Some(StopReason::StopTokenId(128_009)),
        "stop_token_ids list takes priority over eos_token_id"
    );
}

// ── Max tokens budget ─────────────────────────────────────────────────────────

/// Given: `max_tokens` = 5
/// When: generated slice has exactly 5 tokens
/// Then: `check_stop` returns `MaxTokens`
#[test]
fn given_max_tokens_5_when_5_tokens_generated_then_max_tokens_returned() {
    let c = criteria(&[], &[], 5, None);
    let generated = vec![1_u32, 2, 3, 4, 5];
    let reason = check_stop(&c, 99, &generated, "");
    assert_eq!(
        reason,
        Some(StopReason::MaxTokens),
        "MaxTokens must fire when generated.len() >= max_tokens"
    );
}

/// Given: `max_tokens` = 5
/// When: generated slice has only 4 tokens
/// Then: `check_stop` returns None (budget not exhausted)
#[test]
fn given_max_tokens_5_when_4_tokens_generated_then_none_returned() {
    let c = criteria(&[], &[], 5, None);
    let generated = vec![1_u32, 2, 3, 4];
    let reason = check_stop(&c, 99, &generated, "");
    assert_eq!(reason, None, "MaxTokens must not fire until generated.len() >= max_tokens");
}

/// Given: `max_tokens` = 1
/// When: the very first token is generated (generated now has 1 entry)
/// Then: `check_stop` returns `MaxTokens` immediately
#[test]
fn given_max_tokens_1_when_first_token_generated_then_stops_immediately() {
    let c = criteria(&[], &[], 1, None);
    let reason = check_stop(&c, 7, &[7], "text");
    assert_eq!(
        reason,
        Some(StopReason::MaxTokens),
        "max_tokens=1 must stop after the very first generated token"
    );
}

/// Given: `max_tokens` = 0 (no budget limit)
/// When: a very large number of tokens has been generated
/// Then: `check_stop` returns None (0 means unlimited)
#[test]
fn given_max_tokens_zero_when_many_tokens_generated_then_no_budget_limit() {
    let c = criteria(&[], &[], 0, None);
    let large_generated = vec![1_u32; 10_000];
    let reason = check_stop(&c, 99, &large_generated, "");
    assert_eq!(reason, None, "max_tokens=0 must mean unlimited; no MaxTokens should fire");
}

// ── Stop strings ─────────────────────────────────────────────────────────────

/// Given: `stop_strings` = `["\n\n"]`
/// When: `decoded_tail` contains `"\n\n"`
/// Then: `check_stop` returns `StopString`(`"\n\n"`)
#[test]
fn given_stop_string_double_newline_when_tail_contains_it_then_stop_string_returned() {
    let c = criteria(&[], &["\n\n"], 100, None);
    let reason = check_stop(&c, 10, &[], "first paragraph\n\nnext paragraph");
    assert_eq!(
        reason,
        Some(StopReason::StopString("\n\n".to_string())),
        "stop_string must fire when the decoded tail contains the stop sequence"
    );
}

/// Given: `stop_strings` = `["\n\n"]`
/// When: `decoded_tail` does NOT contain `"\n\n"`
/// Then: `check_stop` returns None
#[test]
fn given_stop_string_when_tail_does_not_contain_it_then_none_returned() {
    let c = criteria(&[], &["\n\n"], 100, None);
    let reason = check_stop(&c, 10, &[], "single line text");
    assert_eq!(reason, None, "stop_string must not fire when sequence is absent from the tail");
}

/// Given: `stop_strings` = `["</s>"]`
/// When: `decoded_tail` ends with `"</s>"`
/// Then: `check_stop` returns `StopString`(`"</s>"`)
#[test]
fn given_stop_string_eos_tag_when_tail_ends_with_it_then_stops() {
    let c = criteria(&[], &["</s>"], 100, None);
    let reason = check_stop(&c, 10, &[], "The answer is 42</s>");
    assert_eq!(reason, Some(StopReason::StopString("</s>".to_string())));
}

/// Given: `stop_strings` = `["Q:"]` (common instruct stop)
/// When: `decoded_tail` is `"A: Paris\n\nQ:"` (contains stop)
/// Then: `check_stop` returns `StopString`(`"Q:"`)
#[test]
fn given_stop_string_q_colon_when_tail_contains_it_then_stops() {
    let c = criteria(&[], &["Q:"], 100, None);
    let reason = check_stop(&c, 5, &[], "A: Paris\n\nQ:");
    assert_eq!(reason, Some(StopReason::StopString("Q:".to_string())));
}

/// Given: multiple stop strings `["STOP", "END"]`
/// When: tail contains "END" (second stop)
/// Then: `check_stop` returns StopString("END")
#[test]
fn given_multiple_stop_strings_when_second_matches_then_fires() {
    let c = criteria(&[], &["STOP", "END"], 100, None);
    let reason = check_stop(&c, 5, &[], "output END here");
    assert_eq!(reason, Some(StopReason::StopString("END".to_string())));
}

// ── No stopping conditions ────────────────────────────────────────────────────

/// Given: no stop conditions set (all defaults)
/// When: any token is produced with any context
/// Then: `check_stop` always returns None
#[test]
fn given_no_conditions_when_any_token_produced_then_none_always() {
    let c = StopCriteria::default(); // max_tokens=0, no stop IDs, no EOS, no strings
    let generated = vec![0_u32; 50];
    let reason = check_stop(&c, 12345, &generated, "some long decoded text");
    assert_eq!(reason, None, "no stopping conditions set means generation never stops");
}

// ── Priority ordering ─────────────────────────────────────────────────────────

/// Given: `stop_token_id=5`, EOS=6, `max_tokens=1`, `stop_string` present in tail
/// When: `token_id=5` is produced (`stop_token_id` matches first)
/// Then: StopTokenId(5) is returned (highest priority)
#[test]
fn given_all_conditions_met_when_stop_token_id_present_then_it_has_highest_priority() {
    let generated = vec![0_u32]; // max_tokens=1 also fires
    let c = criteria(&[5], &["STOP"], 1, Some(6));
    // Token 5 matches stop_token_ids; STOP is in the tail; EOS is 6; max_tokens is 1.
    let reason = check_stop(&c, 5, &generated, "some STOP text");
    assert_eq!(
        reason,
        Some(StopReason::StopTokenId(5)),
        "stop_token_id list must be checked before all other conditions"
    );
}

/// Given: EOS=7 and `max_tokens=0` (disabled) and no `stop_token_ids`
/// When: `token_id=7` is produced
/// Then: `EosToken` is returned (EOS fires when `stop_token_ids` does not match)
#[test]
fn given_eos_and_no_stop_ids_when_eos_token_produced_then_eos_token_returned() {
    let c = criteria(&[], &[], 0, Some(7));
    let reason = check_stop(&c, 7, &[], "");
    assert_eq!(reason, Some(StopReason::EosToken));
}

// ── GenerationConfig ─────────────────────────────────────────────────────────

/// Given: `GenerationConfig::default()`
/// When: inspected
/// Then: `max_new_tokens` is a sensible positive value, seed is None
#[test]
fn given_default_generation_config_when_inspected_then_sensible_values() {
    let cfg = GenerationConfig::default();
    assert!(cfg.max_new_tokens > 0, "default max_new_tokens must be positive");
    assert!(cfg.seed.is_none(), "default seed must be None (random)");
}

/// Given: `GenerationConfig` with `max_new_tokens` = 32 and seed = Some(42)
/// When: constructed and inspected
/// Then: fields match the given values
#[test]
fn given_custom_generation_config_when_constructed_then_fields_are_correct() {
    let cfg = GenerationConfig {
        max_new_tokens: 32,
        seed: Some(42),
        stop_criteria: StopCriteria {
            stop_token_ids: vec![128_009],
            max_tokens: 32,
            ..Default::default()
        },
    };
    assert_eq!(cfg.max_new_tokens, 32);
    assert_eq!(cfg.seed, Some(42));
    assert_eq!(cfg.stop_criteria.stop_token_ids, vec![128_009]);
}

// ── StreamEvent ───────────────────────────────────────────────────────────────

/// Given: a `StreamEvent::Token` variant
/// When: inspected
/// Then: carries the expected `TokenEvent` data
#[test]
fn given_stream_event_token_when_constructed_then_carries_token_data() {
    let event = StreamEvent::Token(TokenEvent { id: 42, text: "hello".to_string() });
    match event {
        StreamEvent::Token(t) => {
            assert_eq!(t.id, 42);
            assert_eq!(t.text, "hello");
        }
        StreamEvent::Done { .. } => panic!("expected Token, got Done"),
    }
}

/// Given: a `StreamEvent::Done` variant with `StopReason::MaxTokens`
/// When: inspected
/// Then: the reason and stats are accessible
#[test]
fn given_stream_event_done_when_constructed_then_reason_and_stats_accessible() {
    let event = StreamEvent::Done {
        reason: StopReason::MaxTokens,
        stats: GenerationStats { tokens_generated: 5, tokens_per_second: 10.0 },
    };
    match event {
        StreamEvent::Done { reason, stats } => {
            assert_eq!(reason, StopReason::MaxTokens);
            assert_eq!(stats.tokens_generated, 5);
        }
        StreamEvent::Token(_) => panic!("expected Done, got Token"),
    }
}

// ── Edge cases ────────────────────────────────────────────────────────────────

/// Given: `stop_strings` = `[""]`  (empty string - always matches any tail)
/// When: `check_stop` is called with any tail
/// Then: `StopString`("") is returned (empty string is a substring of everything)
#[test]
fn given_empty_stop_string_when_any_tail_then_always_matches() {
    let c = criteria(&[], &[""], 100, None);
    let reason = check_stop(&c, 10, &[], "any text at all");
    assert_eq!(
        reason,
        Some(StopReason::StopString(String::new())),
        "an empty stop string is a substring of every decoded tail and must always match"
    );
}

/// Given: `stop_token_ids` contains 0 (token 0 is a valid token)
/// When: generated token is 0
/// Then: StopTokenId(0) is returned
#[test]
fn given_stop_on_token_zero_when_token_zero_generated_then_fires() {
    let c = criteria(&[0], &[], 100, None);
    let reason = check_stop(&c, 0, &[], "");
    assert_eq!(reason, Some(StopReason::StopTokenId(0)));
}

/// Given: criteria with `max_tokens` = `usize::MAX`
/// When: generation length is very large
/// Then: no `MaxTokens` fires (it would require generating MAX tokens first)
#[test]
fn given_max_tokens_usize_max_when_large_generation_then_no_overflow() {
    // We just call with a moderately-large generated slice; no panic expected.
    let c = criteria(&[], &[], usize::MAX, None);
    let generated = vec![1_u32; 1000];
    let reason = check_stop(&c, 99, &generated, "");
    assert_eq!(reason, None, "usize::MAX max_tokens must not fire for a small generated slice");
}
