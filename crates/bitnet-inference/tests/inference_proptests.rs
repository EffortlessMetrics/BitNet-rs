//! Property-based tests for `bitnet-inference` invariants.
//!
//! Covers:
//! - [`SamplingConfig`] field round-trips and greedy-mode invariants
//! - [`GenerationConfig`] builder correctness and validation contracts
//! - [`StopCriteria`] / [`check_stop`] stop-token and stop-string semantics
//! - [`StreamingConfig`] construction and validation contracts
//! - [`InferenceReceipt`] schema: `compute_path="real"` with non-mock kernels passes
//!   validation; `compute_path="mock"` always fails

use bitnet_generation::{StopCriteria, StopReason, check_stop};
use bitnet_inference::{GenerationConfig, InferenceReceipt, SamplingConfig, StreamingConfig};
use proptest::prelude::*;

// ── SamplingConfig ────────────────────────────────────────────────────────────

proptest! {
    /// temperature=0.0 is a valid (greedy) sampling config.
    #[test]
    fn prop_sampling_temperature_zero_valid(_dummy in 0u8..1) {
        let cfg = SamplingConfig { temperature: 0.0, ..SamplingConfig::default() };
        prop_assert_eq!(cfg.temperature, 0.0);
    }

    /// Any temperature ≥ 0.0 is stored without modification.
    #[test]
    fn prop_sampling_temperature_roundtrip(temp in 0.0f32..=10.0f32) {
        let cfg = SamplingConfig { temperature: temp, ..SamplingConfig::default() };
        prop_assert!((cfg.temperature - temp).abs() < f32::EPSILON);
    }

    /// top_p values in (0, 1] are stored unchanged.
    #[test]
    fn prop_sampling_top_p_roundtrip(top_p in 0.001f32..=1.0f32) {
        let cfg = SamplingConfig { top_p, ..SamplingConfig::default() };
        prop_assert!((cfg.top_p - top_p).abs() < f32::EPSILON);
    }

    /// Arbitrary seed values are preserved exactly.
    #[test]
    fn prop_sampling_seed_roundtrip(seed in any::<u64>()) {
        let cfg = SamplingConfig { seed: Some(seed), ..SamplingConfig::default() };
        prop_assert_eq!(cfg.seed, Some(seed));
    }

    /// Default SamplingConfig has temperature > 0 (stochastic by default).
    #[test]
    fn prop_sampling_default_stochastic(_dummy in 0u8..1) {
        let cfg = SamplingConfig::default();
        prop_assert!(cfg.temperature > 0.0, "default temperature should be > 0");
        prop_assert!(cfg.top_p > 0.0 && cfg.top_p <= 1.0);
        prop_assert!(cfg.repetition_penalty >= 1.0);
    }
}

// ── GenerationConfig greedy / temperature invariants ─────────────────────────

proptest! {
    /// GenerationConfig::greedy() always sets temperature=0.0.
    #[test]
    fn prop_greedy_config_temperature_zero(_dummy in 0u8..1) {
        let cfg = GenerationConfig::greedy();
        prop_assert_eq!(cfg.temperature, 0.0,
            "greedy() must set temperature to 0.0");
    }

    /// with_max_tokens(n) stores exactly n.
    #[test]
    fn prop_max_tokens_roundtrip(n in 1u32..8192) {
        let cfg = GenerationConfig::default().with_max_tokens(n);
        prop_assert_eq!(cfg.max_new_tokens, n);
    }

    /// max_new_tokens=0 always fails validate().
    #[test]
    fn prop_zero_max_tokens_fails_validate(_dummy in 0u8..1) {
        let cfg = GenerationConfig::default().with_max_tokens(0);
        prop_assert!(
            cfg.validate().is_err(),
            "max_new_tokens=0 must fail validation"
        );
    }

    /// Any positive max_tokens with temperature in [0,2) passes validate().
    #[test]
    fn prop_valid_generation_config_validates(
        max in 1u32..4096,
        temp in 0.0f32..2.0f32,
        top_p in 0.01f32..1.0f32,
    ) {
        let cfg = GenerationConfig::default()
            .with_max_tokens(max)
            .with_temperature(temp)
            .with_top_p(top_p);
        prop_assert!(cfg.validate().is_ok(), "expected Ok, got {:?}", cfg.validate());
    }
}

// ── Stop token lookup: is_stop_token ─────────────────────────────────────────

proptest! {
    /// A token ID explicitly added as a stop token is always detected.
    #[test]
    fn prop_stop_token_id_is_detected(id in any::<u32>()) {
        let cfg = GenerationConfig::default().with_stop_token_id(id);
        prop_assert!(cfg.is_stop_token(id), "stop token {} not detected", id);
    }

    /// A token ID that was NOT added is never detected as a stop token.
    #[test]
    fn prop_non_stop_token_not_detected(
        stop_id in 1u32..1000,
        other_id in 1001u32..u32::MAX,
    ) {
        let cfg = GenerationConfig::default().with_stop_token_id(stop_id);
        prop_assert!(
            !cfg.is_stop_token(other_id),
            "token {} should not be a stop token", other_id
        );
    }

    /// Batch stop token registration: every ID in the batch is detectable.
    #[test]
    fn prop_batch_stop_token_ids_all_detected(
        ids in proptest::collection::vec(0u32..100_000, 1..16),
    ) {
        let cfg = GenerationConfig::default().with_stop_token_ids(ids.clone());
        for &id in &ids {
            prop_assert!(cfg.is_stop_token(id), "batch stop ID {} not detected", id);
        }
    }
}

// ── check_stop: StopCriteria semantics ───────────────────────────────────────

proptest! {
    /// A token in stop_token_ids always triggers StopTokenId, regardless of
    /// generated-sequence length or decoded tail content.
    #[test]
    fn prop_check_stop_explicit_id_wins(
        id in any::<u32>(),
        tail in ".{0,64}",
        extra_ids in proptest::collection::vec(1u32..u32::MAX, 0..4),
    ) {
        let criteria = StopCriteria {
            stop_token_ids: vec![id],
            stop_strings: vec![],
            max_tokens: 1000,
            eos_token_id: None,
        };
        let result = check_stop(&criteria, id, &extra_ids, &tail);
        prop_assert_eq!(result, Some(StopReason::StopTokenId(id)));
    }

    /// A token that is NOT in stop_token_ids, is NOT eos, and has not
    /// exhausted the budget returns None (generation should continue).
    #[test]
    fn prop_check_stop_none_for_non_stop_token(
        stop_id in 1u32..1000,
        other_id in 1001u32..10_000,
    ) {
        let criteria = StopCriteria {
            stop_token_ids: vec![stop_id],
            stop_strings: vec![],
            max_tokens: 1000,
            eos_token_id: None,
        };
        // generated slice is short, so budget is not exhausted
        let result = check_stop(&criteria, other_id, &[1, 2, 3], "hello world");
        prop_assert!(result.is_none(),
            "check_stop should return None for non-stop token {}", other_id);
    }

    /// Once the generated slice reaches max_tokens, MaxTokens is returned.
    #[test]
    fn prop_check_stop_max_tokens_triggers(
        budget in 1usize..64,
        non_stop in 50_000u32..u32::MAX,
    ) {
        let criteria = StopCriteria {
            stop_token_ids: vec![],
            stop_strings: vec![],
            max_tokens: budget,
            eos_token_id: None,
        };
        let generated: Vec<u32> = vec![1u32; budget]; // exactly at limit
        let result = check_stop(&criteria, non_stop, &generated, "");
        prop_assert_eq!(result, Some(StopReason::MaxTokens));
    }

    /// EOS token triggers EosToken stop reason.
    #[test]
    fn prop_check_stop_eos_triggers(eos_id in any::<u32>()) {
        let criteria = StopCriteria {
            stop_token_ids: vec![],
            stop_strings: vec![],
            max_tokens: 1000,
            eos_token_id: Some(eos_id),
        };
        let result = check_stop(&criteria, eos_id, &[], "");
        prop_assert_eq!(result, Some(StopReason::EosToken));
    }
}

// ── StreamingConfig invariants ────────────────────────────────────────────────

proptest! {
    /// StreamingConfig with all-zero fields fails validate().
    #[test]
    fn prop_streaming_config_zero_buffer_fails(_dummy in 0u8..1) {
        let cfg = StreamingConfig { buffer_size: 0, ..StreamingConfig::default() };
        prop_assert!(cfg.validate().is_err());
    }

    /// Positive buffer_size / flush_interval / token_timeout passes validate().
    #[test]
    fn prop_streaming_config_positive_fields_pass(
        buf in 1usize..256,
        flush in 1u64..1000,
        timeout in 1u64..30_000,
    ) {
        let cfg = StreamingConfig {
            buffer_size: buf,
            flush_interval_ms: flush,
            token_timeout_ms: timeout,
            ..StreamingConfig::default()
        };
        prop_assert!(cfg.validate().is_ok(), "expected Ok, got {:?}", cfg.validate());
    }
}

// ── InferenceReceipt schema contracts ────────────────────────────────────────

proptest! {
    /// A receipt with compute_path="real" and non-mock kernel IDs passes validate().
    #[test]
    fn prop_receipt_real_compute_path_valid(
        backend in prop_oneof![Just("cpu"), Just("metal")],
        kernel_suffix in "[a-z][a-z0-9_]{1,15}",
    ) {
        let kernel = format!("real_{}", kernel_suffix);
        let result = InferenceReceipt::generate(backend, vec![kernel], None);
        match result {
            Ok(receipt) => {
                prop_assert_eq!(&receipt.compute_path, "real");
                prop_assert!(receipt.validate().is_ok(),
                    "receipt with real kernel failed validate: {:?}", receipt.validate());
            }
            Err(e) => prop_assert!(false, "generate() failed unexpectedly: {}", e),
        }
    }

    /// A receipt with a mock kernel in the list has compute_path="mock"
    /// and its validate() returns an error.
    #[test]
    fn prop_receipt_mock_kernel_fails_validate(_dummy in 0u8..1) {
        let result = InferenceReceipt::generate("cpu", vec!["mock_gemv".to_string()], None);
        match result {
            Ok(receipt) => {
                prop_assert_eq!(&receipt.compute_path, "mock");
                prop_assert!(
                    receipt.validate().is_err(),
                    "receipt with mock kernel must fail validate()"
                );
            }
            Err(e) => prop_assert!(false, "generate() failed unexpectedly: {}", e),
        }
    }
}
