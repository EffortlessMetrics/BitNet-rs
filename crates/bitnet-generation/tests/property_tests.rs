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
            prop_oneof![Just("[STOP]".to_string()), Just("</s>".to_string()), Just("\n".to_string())],
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
