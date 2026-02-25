//! Property-based tests for bitnet-inference configuration invariants.
//!
//! Tests key invariants of GenerationConfig and InferenceConfig that must hold
//! across all possible inputs — validation rules, stop-token semantics, and builder methods.

use bitnet_inference::config::{GenerationConfig, InferenceConfig};
use proptest::prelude::*;

// ── GenerationConfig builders ─────────────────────────────────────────────────

proptest! {
    /// greedy() always produces temperature=0.0 (deterministic).
    #[test]
    fn prop_generation_config_greedy_is_deterministic(_dummy in 0u8..1) {
        let cfg = GenerationConfig::greedy();
        prop_assert_eq!(cfg.temperature, 0.0);
        prop_assert!(cfg.top_k <= 1 || cfg.temperature == 0.0);
    }

    /// creative() always produces temperature > 0 (stochastic).
    #[test]
    fn prop_generation_config_creative_is_stochastic(_dummy in 0u8..1) {
        let cfg = GenerationConfig::creative();
        prop_assert!(cfg.temperature > 0.0);
        prop_assert!(cfg.top_k > 1);
        prop_assert!(cfg.top_p < 1.0);
    }

    /// with_max_tokens preserves the given value.
    #[test]
    fn prop_with_max_tokens_preserved(max in 1u32..4096) {
        let cfg = GenerationConfig::default().with_max_tokens(max);
        prop_assert_eq!(cfg.max_new_tokens, max);
    }

    /// with_temperature preserves the given value exactly.
    #[test]
    fn prop_with_temperature_preserved(temp in 0.0f32..2.0) {
        let cfg = GenerationConfig::default().with_temperature(temp);
        prop_assert!((cfg.temperature - temp).abs() < 1e-7);
    }

    /// with_seed preserves the seed value.
    #[test]
    fn prop_with_seed_preserved(seed in any::<u64>()) {
        let cfg = GenerationConfig::default().with_seed(seed);
        prop_assert_eq!(cfg.seed, Some(seed));
    }

    /// with_top_p preserves the value.
    #[test]
    fn prop_with_top_p_preserved(top_p in 0.01f32..1.0) {
        let cfg = GenerationConfig::default().with_top_p(top_p);
        prop_assert!((cfg.top_p - top_p).abs() < 1e-7);
    }

    /// with_repetition_penalty preserves the value.
    #[test]
    fn prop_with_repetition_penalty_preserved(penalty in 1.0f32..2.0) {
        let cfg = GenerationConfig::default().with_repetition_penalty(penalty);
        prop_assert!((cfg.repetition_penalty - penalty).abs() < 1e-7);
    }
}

// ── Stop token invariants ─────────────────────────────────────────────────────

proptest! {
    /// is_stop_token returns true for all IDs added via with_stop_token_id.
    #[test]
    fn prop_stop_token_id_is_detected(id in any::<u32>()) {
        let cfg = GenerationConfig::default().with_stop_token_id(id);
        prop_assert!(cfg.is_stop_token(id));
    }

    /// is_stop_token returns false for IDs NOT in the stop set.
    #[test]
    fn prop_non_stop_token_not_detected(
        stop_id in 1u32..10,
        other_id in 11u32..10000,
    ) {
        let cfg = GenerationConfig::default().with_stop_token_id(stop_id);
        prop_assert!(!cfg.is_stop_token(other_id));
    }

    /// with_stop_token_ids sets all IDs as detectable stop tokens.
    #[test]
    fn prop_stop_token_ids_all_detected(
        ids in proptest::collection::vec(any::<u32>(), 1..10),
    ) {
        let cfg = GenerationConfig::default().with_stop_token_ids(ids.clone());
        for id in &ids {
            prop_assert!(cfg.is_stop_token(*id), "ID {} not detected", id);
        }
    }

    /// rebuild_stop_token_set syncs Vec to HashSet so is_stop_token works.
    #[test]
    fn prop_rebuild_stop_token_set_syncs(id in any::<u32>()) {
        let mut cfg = GenerationConfig::default();
        cfg.stop_token_ids = vec![id];
        // Before rebuild: HashSet may be out of sync
        cfg.rebuild_stop_token_set();
        prop_assert!(cfg.is_stop_token(id));
    }
}

// ── GenerationConfig::validate ────────────────────────────────────────────────

proptest! {
    /// Valid configs always pass validate().
    #[test]
    fn prop_valid_generation_config_passes_validation(
        max_tokens in 1u32..4096,
        temp in 0.0f32..2.0,
        top_p in 0.01f32..1.0,
        rep_penalty in 0.1f32..3.0,
    ) {
        let cfg = GenerationConfig::default()
            .with_max_tokens(max_tokens)
            .with_temperature(temp)
            .with_top_p(top_p)
            .with_repetition_penalty(rep_penalty);
        let result = cfg.validate();
        prop_assert!(result.is_ok(), "validate failed: {:?}", result);
    }

    /// max_new_tokens=0 always fails validate().
    #[test]
    fn prop_zero_max_tokens_fails_validation(_dummy in 0u8..1) {
        let cfg = GenerationConfig::default().with_max_tokens(0);
        prop_assert!(cfg.validate().is_err());
    }
}

// ── InferenceConfig ───────────────────────────────────────────────────────────

proptest! {
    /// Default InferenceConfig has sensible non-zero values.
    #[test]
    fn prop_inference_config_default_valid(_dummy in 0u8..1) {
        let cfg = InferenceConfig::default();
        prop_assert!(cfg.max_context_length > 0);
        prop_assert!(cfg.num_threads > 0);
        prop_assert!(cfg.batch_size > 0);
        prop_assert!(cfg.memory_pool_size > 0);
    }

    /// with_threads preserves the given thread count.
    #[test]
    fn prop_inference_config_with_threads(threads in 1usize..32) {
        let cfg = InferenceConfig::default().with_threads(threads);
        prop_assert_eq!(cfg.num_threads, threads);
    }

    /// with_batch_size preserves the given batch size.
    #[test]
    fn prop_inference_config_with_batch_size(batch in 1usize..64) {
        let cfg = InferenceConfig::default().with_batch_size(batch);
        prop_assert_eq!(cfg.batch_size, batch);
    }
}
