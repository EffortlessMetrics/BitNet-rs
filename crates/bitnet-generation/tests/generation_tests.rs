//! Named integration tests for `bitnet-generation` covering the five SRP invariants
//! required by the extraction task.

use bitnet_generation::{GenerationConfig, StopCriteria, StopReason, check_stop};
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
