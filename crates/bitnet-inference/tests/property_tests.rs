//! Property-based tests for bitnet-inference configuration invariants.
//!
//! Tests key invariants of GenerationConfig, InferenceConfig, SamplingConfig,
//! and StopCriteria that must hold across all possible inputs — validation rules,
//! stop-token semantics, builder methods, and round-trip correctness.

use bitnet_generation::StopCriteria;
use bitnet_inference::config::{GenerationConfig, InferenceConfig};
use bitnet_inference::sampling::SamplingConfig;
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

// ── InferenceConfig validation — zero fields must fail ───────────────────────

proptest! {
    /// InferenceConfig with max_context_length=0 fails validate().
    #[test]
    fn prop_inference_config_zero_context_fails(_dummy in 0u8..1) {
        let cfg = InferenceConfig { max_context_length: 0, ..InferenceConfig::default() };
        prop_assert!(cfg.validate().is_err());
    }

    /// InferenceConfig with num_threads=0 fails validate().
    #[test]
    fn prop_inference_config_zero_threads_fails(_dummy in 0u8..1) {
        let cfg = InferenceConfig { num_threads: 0, ..InferenceConfig::default() };
        prop_assert!(cfg.validate().is_err());
    }

    /// InferenceConfig with batch_size=0 fails validate().
    #[test]
    fn prop_inference_config_zero_batch_fails(_dummy in 0u8..1) {
        let cfg = InferenceConfig { batch_size: 0, ..InferenceConfig::default() };
        prop_assert!(cfg.validate().is_err());
    }

    /// InferenceConfig with memory_pool_size=0 fails validate().
    #[test]
    fn prop_inference_config_zero_memory_fails(_dummy in 0u8..1) {
        let cfg = InferenceConfig { memory_pool_size: 0, ..InferenceConfig::default() };
        prop_assert!(cfg.validate().is_err());
    }

    /// InferenceConfig with all positive fields passes validate().
    #[test]
    fn prop_inference_config_positive_fields_pass(
        ctx in 1usize..65536,
        threads in 1usize..64,
        batch in 1usize..32,
        pool in 1usize..usize::MAX,
    ) {
        let cfg = InferenceConfig {
            max_context_length: ctx,
            num_threads: threads,
            batch_size: batch,
            memory_pool_size: pool,
            ..InferenceConfig::default()
        };
        prop_assert!(cfg.validate().is_ok(), "expected Ok, got {:?}", cfg.validate());
    }
}

// ── SamplingConfig invariants ─────────────────────────────────────────────────

proptest! {
    /// SamplingConfig defaults satisfy expected field-range invariants.
    #[test]
    fn prop_sampling_config_defaults_in_range(_dummy in 0u8..1) {
        let cfg = SamplingConfig::default();
        prop_assert!(cfg.temperature >= 0.0);
        prop_assert!(cfg.top_p > 0.0 && cfg.top_p <= 1.0);
        prop_assert!(cfg.repetition_penalty >= 0.0);
    }

    /// Any temperature ≥ 0.0 is stored unchanged.
    #[test]
    fn prop_sampling_config_temperature_preserved(temp in 0.0f32..10.0) {
        let cfg = SamplingConfig { temperature: temp, ..SamplingConfig::default() };
        prop_assert!((cfg.temperature - temp).abs() < f32::EPSILON);
    }

    /// top_p stored unchanged for values in (0, 1].
    #[test]
    fn prop_sampling_config_top_p_preserved(top_p in 0.001f32..=1.0) {
        let cfg = SamplingConfig { top_p, ..SamplingConfig::default() };
        prop_assert!((cfg.top_p - top_p).abs() < f32::EPSILON);
    }

    /// repetition_penalty stored unchanged.
    #[test]
    fn prop_sampling_config_repetition_penalty_preserved(penalty in 0.5f32..3.0) {
        let cfg = SamplingConfig { repetition_penalty: penalty, ..SamplingConfig::default() };
        prop_assert!((cfg.repetition_penalty - penalty).abs() < f32::EPSILON);
    }

    /// seed is round-tripped without modification.
    #[test]
    fn prop_sampling_config_seed_roundtrip(seed in any::<u64>()) {
        let cfg = SamplingConfig { seed: Some(seed), ..SamplingConfig::default() };
        prop_assert_eq!(cfg.seed, Some(seed));
    }
}

// ── Token ID round-trips ──────────────────────────────────────────────────────

proptest! {
    /// Any u32 token ID pushed into stop_token_ids is retrievable unchanged.
    #[test]
    fn prop_token_id_roundtrip(id in any::<u32>()) {
        let cfg = GenerationConfig::default().with_stop_token_id(id);
        prop_assert!(cfg.stop_token_ids.contains(&id));
        prop_assert_eq!(cfg.stop_token_ids.iter().filter(|&&x| x == id).count(), 1);
    }

    /// A batch of arbitrary token IDs survives with_stop_token_ids unchanged.
    #[test]
    fn prop_token_id_batch_roundtrip(ids in proptest::collection::vec(any::<u32>(), 1..20)) {
        let cfg = GenerationConfig::default().with_stop_token_ids(ids.clone());
        prop_assert_eq!(cfg.stop_token_ids, ids);
    }
}

// ── GenerationConfig defaults are self-consistent ────────────────────────────

proptest! {
    /// The default GenerationConfig always passes validate().
    #[test]
    fn prop_generation_config_default_passes_validation(_dummy in 0u8..1) {
        prop_assert!(GenerationConfig::default().validate().is_ok());
    }

    /// negative temperature always fails validate().
    #[test]
    fn prop_negative_temperature_fails_validation(temp in f32::MIN..-f32::EPSILON) {
        let cfg = GenerationConfig::default().with_temperature(temp);
        prop_assert!(cfg.validate().is_err());
    }

    /// top_p > 1.0 always fails validate().
    #[test]
    fn prop_top_p_above_one_fails_validation(top_p in 1.001f32..=2.0) {
        let cfg = GenerationConfig::default().with_top_p(top_p);
        prop_assert!(cfg.validate().is_err());
    }
}

// ── StopCriteria — no panics, empty strings are valid ────────────────────────

proptest! {
    /// StopCriteria with stop strings up to 256 bytes never panics.
    #[test]
    fn prop_stop_criteria_strings_no_panic(
        s in proptest::string::string_regex(".{0,256}").unwrap(),
    ) {
        let criteria = StopCriteria {
            stop_token_ids: vec![],
            stop_strings: vec![s],
            max_tokens: 64,
            eos_token_id: None,
        };
        // Access fields to ensure no lazy-init panics.
        let _ = criteria.stop_strings.len();
        let _ = criteria.max_tokens;
    }

    /// StopCriteria with an empty stop_strings vec is valid (no panic).
    #[test]
    fn prop_stop_criteria_empty_strings_valid(max in 0usize..4096) {
        let criteria = StopCriteria {
            stop_token_ids: vec![],
            stop_strings: vec![],
            max_tokens: max,
            eos_token_id: None,
        };
        prop_assert_eq!(criteria.stop_strings.len(), 0);
    }

    /// StopCriteria stop_token_ids are stored without modification.
    #[test]
    fn prop_stop_criteria_token_ids_roundtrip(
        ids in proptest::collection::vec(any::<u32>(), 0..16),
    ) {
        let criteria = StopCriteria {
            stop_token_ids: ids.clone(),
            stop_strings: vec![],
            max_tokens: 1,
            eos_token_id: None,
        };
        prop_assert_eq!(criteria.stop_token_ids, ids);
    }
}
