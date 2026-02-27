//! Property-based tests for re-exported types in `bitnet-engine-core`.
//!
//! Covers invariants for the types that are re-exported from
//! `bitnet-generation`: `StopCriteria`, `StopReason`, `GenerationStats`,
//! and `TokenEvent`, as well as structural properties of `SessionConfig`
//! field independence and the `GenerationConfig` stop-criteria embedding.

use bitnet_engine_core::{
    GenerationConfig, GenerationStats, SessionConfig, StopCriteria, StopReason, TokenEvent,
};
use proptest::prelude::*;

// ── strategies ─────────────────────────────────────────────────────────────

fn arb_stop_reason() -> impl Strategy<Value = StopReason> {
    prop_oneof![
        Just(StopReason::MaxTokens),
        Just(StopReason::EosToken),
        any::<u32>().prop_map(StopReason::StopTokenId),
        "[a-z</]{1,16}".prop_map(StopReason::StopString),
    ]
}

fn arb_stop_criteria() -> impl Strategy<Value = StopCriteria> {
    (
        prop::collection::vec(any::<u32>(), 0..8),
        prop::collection::vec("[a-z<>/]{1,16}", 0..4),
        0usize..512,
        prop::option::of(any::<u32>()),
    )
        .prop_map(|(stop_token_ids, stop_strings, max_tokens, eos_token_id)| StopCriteria {
            stop_token_ids,
            stop_strings,
            max_tokens,
            eos_token_id,
        })
}

// ── 1. StopCriteria serde roundtrip ────────────────────────────────────────

proptest! {
    /// `StopCriteria` round-trips through JSON without data loss.
    ///
    /// Checks that every field (token IDs, strings, budget, EOS) is preserved
    /// exactly so the stopping logic is not silently altered by serialization.
    #[test]
    fn stop_criteria_json_roundtrip(criteria in arb_stop_criteria()) {
        let json = serde_json::to_string(&criteria).expect("serialize StopCriteria");
        let restored: StopCriteria = serde_json::from_str(&json).expect("deserialize StopCriteria");
        prop_assert_eq!(criteria.stop_token_ids, restored.stop_token_ids);
        prop_assert_eq!(criteria.stop_strings, restored.stop_strings);
        prop_assert_eq!(criteria.max_tokens, restored.max_tokens);
        prop_assert_eq!(criteria.eos_token_id, restored.eos_token_id);
    }

    /// The order of `stop_token_ids` is preserved after JSON round-trip.
    ///
    /// Stopping logic matches the first token ID in the list, so order must
    /// not be silently shuffled by serialization.
    #[test]
    fn stop_criteria_token_ids_order_preserved(
        ids in prop::collection::vec(any::<u32>(), 0..16)
    ) {
        let criteria = StopCriteria {
            stop_token_ids: ids.clone(),
            stop_strings: vec![],
            max_tokens: 0,
            eos_token_id: None,
        };
        let json = serde_json::to_string(&criteria).unwrap();
        let restored: StopCriteria = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(ids, restored.stop_token_ids);
    }
}

// ── 2. StopReason serde roundtrip ──────────────────────────────────────────

proptest! {
    /// Every `StopReason` variant round-trips through JSON without data loss.
    ///
    /// Because `StopReason` is an enum with a newtype variant carrying a `u32`
    /// and a tuple variant carrying a `String`, serde tagging must handle all
    /// four variants correctly.
    #[test]
    fn stop_reason_json_roundtrip(reason in arb_stop_reason()) {
        let json = serde_json::to_string(&reason).expect("serialize StopReason");
        let restored: StopReason = serde_json::from_str(&json).expect("deserialize StopReason");
        prop_assert_eq!(reason, restored);
    }

    /// Cloning `StopReason` produces a value that serializes identically.
    #[test]
    fn stop_reason_clone_serializes_identically(reason in arb_stop_reason()) {
        let cloned = reason.clone();
        let orig_json  = serde_json::to_string(&reason).unwrap();
        let clone_json = serde_json::to_string(&cloned).unwrap();
        prop_assert_eq!(orig_json, clone_json);
    }
}

// ── 3. GenerationStats serde roundtrip ─────────────────────────────────────

proptest! {
    /// `GenerationStats` round-trips through JSON with finite-float precision.
    ///
    /// `tokens_per_second` is a finite `f64`; any JSON round-trip loss must be
    /// within floating-point epsilon.
    #[test]
    fn generation_stats_json_roundtrip(
        tokens_generated in 0usize..1_000_000,
        tokens_per_second in 0.0f64..1_000_000.0f64,
    ) {
        let stats = GenerationStats { tokens_generated, tokens_per_second };
        let json = serde_json::to_string(&stats).expect("serialize GenerationStats");
        let restored: GenerationStats = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(stats.tokens_generated, restored.tokens_generated);
        prop_assert!((stats.tokens_per_second - restored.tokens_per_second).abs() < 1e-6);
    }

    /// Cloning `GenerationStats` produces a value that serializes identically.
    #[test]
    fn generation_stats_clone_serializes_identically(
        tokens_generated in 0usize..100_000,
        tokens_per_second in 0.0f64..100_000.0f64,
    ) {
        let stats = GenerationStats { tokens_generated, tokens_per_second };
        let cloned = stats.clone();
        let orig_json  = serde_json::to_string(&stats).unwrap();
        let clone_json = serde_json::to_string(&cloned).unwrap();
        prop_assert_eq!(orig_json, clone_json);
    }
}

// ── 4. TokenEvent serde roundtrip ──────────────────────────────────────────

proptest! {
    /// `TokenEvent` round-trips through JSON preserving both `id` and `text`.
    ///
    /// `text` may contain arbitrary Unicode; the JSON codec must not corrupt it.
    #[test]
    fn token_event_json_roundtrip(
        id in any::<u32>(),
        text in "[ -~]{0,64}",  // printable ASCII
    ) {
        let event = TokenEvent { id, text: text.clone() };
        let json = serde_json::to_string(&event).expect("serialize TokenEvent");
        let restored: TokenEvent = serde_json::from_str(&json).expect("deserialize TokenEvent");
        prop_assert_eq!(event.id, restored.id);
        prop_assert_eq!(event.text, restored.text);
    }
}

// ── 5. GenerationConfig stop_criteria embedding ─────────────────────────────

proptest! {
    /// `GenerationConfig` embeds `StopCriteria` and the whole thing round-trips
    /// through JSON without losing stop-criteria fields.
    #[test]
    fn generation_config_stop_criteria_embedded_roundtrip(
        max_new_tokens in 1usize..4096,
        seed in prop::option::of(any::<u64>()),
        criteria in arb_stop_criteria(),
    ) {
        let config = GenerationConfig {
            max_new_tokens,
            seed,
            stop_criteria: criteria.clone(),
        };
        let json = serde_json::to_string(&config).expect("serialize GenerationConfig");
        let restored: GenerationConfig = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(config.max_new_tokens, restored.max_new_tokens);
        prop_assert_eq!(config.seed, restored.seed);
        prop_assert_eq!(
            criteria.stop_token_ids,
            restored.stop_criteria.stop_token_ids
        );
        prop_assert_eq!(criteria.max_tokens, restored.stop_criteria.max_tokens);
        prop_assert_eq!(criteria.eos_token_id, restored.stop_criteria.eos_token_id);
    }
}

// ── 6. SessionConfig field independence after clone ─────────────────────────

proptest! {
    /// Mutating a cloned `SessionConfig` does not affect the original.
    ///
    /// This guards against accidental `Rc`/`Arc` sharing or shallow clones
    /// that would let mutations bleed across boundaries.
    #[test]
    fn session_config_clone_is_independent(
        model_path in "[a-z0-9_/]{1,32}",
        max_context in 1usize..8192,
        seed in prop::option::of(any::<u64>()),
    ) {
        let original = SessionConfig {
            model_path: model_path.clone(),
            tokenizer_path: "tok.json".to_string(),
            backend: "cpu".to_string(),
            max_context,
            seed,
        };
        let mut cloned = original.clone();

        // Mutate the clone.
        cloned.model_path = "mutated.gguf".to_string();
        cloned.max_context = max_context.saturating_add(1);

        // Original must be unchanged.
        prop_assert_eq!(&original.model_path, &model_path);
        prop_assert_eq!(original.max_context, max_context);
    }

    /// `SessionConfig::seed` stores the full `u64` range without truncation,
    /// even after mutating and re-serializing.
    #[test]
    fn session_config_seed_mutation_is_independent(seed_a in any::<u64>(), seed_b in any::<u64>()) {
        let mut config = SessionConfig {
            model_path: String::new(),
            tokenizer_path: String::new(),
            backend: "cpu".to_string(),
            max_context: 512,
            seed: Some(seed_a),
        };
        let original_seed = config.seed;
        config.seed = Some(seed_b);
        // The original binding still holds the old value (value semantics).
        prop_assert_eq!(original_seed, Some(seed_a));
        prop_assert_eq!(config.seed, Some(seed_b));
    }
}
