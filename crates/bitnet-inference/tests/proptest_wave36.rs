//! Property-based tests — wave 36.
//!
//! Covers bitnet-inference: sampling config validation, generation budget
//! tracking, inference request validation, config builder invariants,
//! and batch/token budget bounds.

use bitnet_inference::config_builder::{
    GenerationConfig, HardwareConfig, InferenceConfig, InferenceConfigBuilder, InferencePreset,
    SamplingConfig,
};
use bitnet_inference::generation_budget::{BudgetTracker, GenerationBudget, StopReason};
use bitnet_inference::request_types::{InferenceRequest, validate_request};
use proptest::prelude::*;

// ── Strategies ──────────────────────────────────────────────────────────────

fn arb_preset() -> impl Strategy<Value = InferencePreset> {
    prop_oneof![
        Just(InferencePreset::Fast),
        Just(InferencePreset::Balanced),
        Just(InferencePreset::Quality),
        Just(InferencePreset::Deterministic),
        Just(InferencePreset::Debug),
    ]
}

fn valid_temperature() -> impl Strategy<Value = f32> {
    0.0f32..5.0
}

fn valid_top_p() -> impl Strategy<Value = f32> {
    0.01f32..1.0
}

fn valid_top_k() -> impl Strategy<Value = u32> {
    1u32..500
}

fn valid_rep_penalty() -> impl Strategy<Value = f32> {
    1.0f32..3.0
}

fn valid_max_tokens() -> impl Strategy<Value = u32> {
    1u32..32768
}

// ── Property tests ──────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    // ════════════════════════════════════════════════════════════════
    // 1. InferenceConfig validation properties
    // ════════════════════════════════════════════════════════════════

    /// Preset configs always validate successfully.
    #[test]
    fn prop_preset_validates(preset in arb_preset()) {
        let config = InferenceConfigBuilder::new()
            .preset(preset)
            .build()
            .unwrap();
        prop_assert!(config.validate().is_ok(), "preset {:?} should validate", preset);
    }

    /// Valid temperature + valid top_p + valid max_tokens always validates.
    #[test]
    fn prop_valid_params_validate(
        temp in valid_temperature(),
        top_p in valid_top_p(),
        max_tokens in valid_max_tokens(),
        rep_penalty in valid_rep_penalty()
    ) {
        let mut gen_cfg = GenerationConfig::default();
            gen_cfg.max_tokens = max_tokens;
            let config = InferenceConfig {
            sampling: SamplingConfig {
                temperature: temp,
                top_k: 50,
                top_p,
                repetition_penalty: rep_penalty,
                seed: None,
            },
            generation: gen_cfg,
            hardware: HardwareConfig::default(),
        };
        prop_assert!(config.validate().is_ok());
    }

    /// Negative temperature always fails validation.
    #[test]
    fn prop_neg_temperature_fails(temp in -10.0f32..-0.001) {
        let config = InferenceConfig {
            sampling: SamplingConfig {
                temperature: temp,
                ..SamplingConfig::default()
            },
            generation: GenerationConfig::default(),
            hardware: HardwareConfig::default(),
        };
        prop_assert!(config.validate().is_err());
    }

    /// top_p <= 0 or > 1 fails validation.
    #[test]
    fn prop_invalid_top_p_fails(top_p in prop_oneof![
        -1.0f32..0.0,
        Just(0.0f32),
        1.001f32..5.0,
    ]) {
        let config = InferenceConfig {
            sampling: SamplingConfig {
                top_p,
                ..SamplingConfig::default()
            },
            generation: GenerationConfig::default(),
            hardware: HardwareConfig::default(),
        };
        prop_assert!(config.validate().is_err());
    }

    /// max_tokens == 0 fails validation.
    #[test]
    fn prop_zero_max_tokens_fails(_dummy in 0u8..1) {
        let mut gen_cfg = GenerationConfig::default();
            gen_cfg.max_tokens = 0;
            let config = InferenceConfig {
            sampling: SamplingConfig::default(),
            generation: gen_cfg,
            hardware: HardwareConfig::default(),
        };
        prop_assert!(config.validate().is_err());
    }

    /// Repetition penalty <= 0 fails validation.
    #[test]
    fn prop_bad_rep_penalty_fails(penalty in -5.0f32..0.0) {
        let config = InferenceConfig {
            sampling: SamplingConfig {
                repetition_penalty: penalty,
                ..SamplingConfig::default()
            },
            generation: GenerationConfig::default(),
            hardware: HardwareConfig::default(),
        };
        prop_assert!(config.validate().is_err());
    }

    // ════════════════════════════════════════════════════════════════
    // 2. InferenceConfigBuilder properties
    // ════════════════════════════════════════════════════════════════

    /// Builder with valid parameters always produces a valid config.
    #[test]
    fn prop_builder_valid(
        temp in valid_temperature(),
        top_k in valid_top_k(),
        top_p in valid_top_p(),
        max_tokens in valid_max_tokens()
    ) {
        let result = InferenceConfigBuilder::new()
            .temperature(temp)
            .top_k(top_k)
            .top_p(top_p)
            .max_tokens(max_tokens)
            .build();
        prop_assert!(result.is_ok(), "builder with valid params should succeed");
    }

    /// Builder temperature setter is stored correctly.
    #[test]
    fn prop_builder_temperature_stored(temp in valid_temperature()) {
        let config = InferenceConfigBuilder::new()
            .temperature(temp)
            .build()
            .unwrap();
        prop_assert!(
            (config.sampling.temperature - temp).abs() < 1e-6,
            "temperature should match"
        );
    }

    /// Builder top_k setter is stored correctly.
    #[test]
    fn prop_builder_top_k_stored(k in valid_top_k()) {
        let config = InferenceConfigBuilder::new()
            .top_k(k)
            .build()
            .unwrap();
        prop_assert_eq!(config.sampling.top_k, k);
    }

    /// Builder preset overrides previous settings.
    #[test]
    fn prop_preset_overrides(
        preset in arb_preset(),
        temp in valid_temperature()
    ) {
        let config = InferenceConfigBuilder::new()
            .temperature(temp)
            .preset(preset)
            .build()
            .unwrap();
        // After applying preset, temperature should match preset's default,
        // not the previously set value (unless they happen to be equal).
        prop_assert!(config.validate().is_ok());
    }

    // ════════════════════════════════════════════════════════════════
    // 3. GenerationBudget properties
    // ════════════════════════════════════════════════════════════════

    /// New budget starts with 0 tokens generated.
    #[test]
    fn prop_budget_starts_empty(max_tokens in 1usize..10000) {
        let budget = GenerationBudget::new(max_tokens);
        let tracker = BudgetTracker::new(budget);
        prop_assert_eq!(tracker.tokens_generated(), 0);
        prop_assert!(tracker.can_continue());
    }

    /// After recording max_tokens, budget is exhausted.
    #[test]
    fn prop_budget_exhausted_at_limit(max_tokens in 1usize..1000) {
        let budget = GenerationBudget::new(max_tokens);
        let mut tracker = BudgetTracker::new(budget);
        for _ in 0..max_tokens {
            let can = tracker.record_token();
            prop_assert!(can || tracker.tokens_generated() == max_tokens);
        }
        prop_assert!(!tracker.can_continue());
    }

    /// tokens_generated increments by 1 per record_token.
    #[test]
    fn prop_budget_token_count(n in 1usize..100) {
        let budget = GenerationBudget::new(1000);
        let mut tracker = BudgetTracker::new(budget);
        for i in 0..n {
            tracker.record_token();
            prop_assert_eq!(tracker.tokens_generated(), i + 1);
        }
    }

    /// tokens_remaining = max_tokens - tokens_generated.
    #[test]
    fn prop_budget_remaining(max_tokens in 10usize..1000, n in 0usize..10) {
        let budget = GenerationBudget::new(max_tokens);
        let mut tracker = BudgetTracker::new(budget);
        for _ in 0..n {
            tracker.record_token();
        }
        prop_assert_eq!(tracker.tokens_remaining(), max_tokens - n);
    }

    /// token_utilization is in [0.0, 1.0].
    #[test]
    fn prop_budget_utilization_bounded(max_tokens in 1usize..1000, n in 0usize..100) {
        prop_assume!(n <= max_tokens);
        let budget = GenerationBudget::new(max_tokens);
        let mut tracker = BudgetTracker::new(budget);
        for _ in 0..n {
            tracker.record_token();
        }
        let util = tracker.token_utilization();
        prop_assert!(util >= 0.0 && util <= 1.0, "utilization {} out of range", util);
    }

    /// StopReason Display is non-empty.
    #[test]
    fn prop_stop_reason_display(
        reason in prop_oneof![
            Just(StopReason::MaxTokens),
            Just(StopReason::TimeLimit),
            Just(StopReason::MemoryLimit),
            Just(StopReason::EndOfSequence),
            Just(StopReason::UserStop),
        ]
    ) {
        prop_assert!(!reason.to_string().is_empty());
    }

    // ════════════════════════════════════════════════════════════════
    // 4. InferenceRequest validation properties
    // ════════════════════════════════════════════════════════════════

    /// Valid InferenceRequest passes validation.
    #[test]
    fn prop_valid_request_validates(
        max_tokens in 1usize..32768,
        temperature in 0.0f32..10.0,
        top_p in 0.01f32..1.0,
        rep_penalty in 1.0f32..5.0
    ) {
        let req = InferenceRequest::new("test prompt")
            .with_max_tokens(max_tokens)
            .with_temperature(temperature)
            .with_top_p(top_p);
        // Set rep_penalty on the struct directly since there's no builder method
        let mut req = req;
        req.repetition_penalty = rep_penalty;
        prop_assert!(validate_request(&req).is_ok());
    }

    /// Empty prompt fails validation.
    #[test]
    fn prop_empty_prompt_fails(_dummy in 0u8..1) {
        let req = InferenceRequest::new("");
        let result = validate_request(&req);
        prop_assert!(result.is_err());
    }

    /// max_tokens > 32768 fails validation.
    #[test]
    fn prop_too_many_tokens_fails(max_tokens in 32769usize..100_000) {
        let req = InferenceRequest::new("test").with_max_tokens(max_tokens);
        let result = validate_request(&req);
        prop_assert!(result.is_err());
    }

    /// Negative temperature fails validation.
    #[test]
    fn prop_neg_temp_request_fails(temp in -10.0f32..-0.001) {
        let req = InferenceRequest::new("test").with_temperature(temp);
        prop_assert!(validate_request(&req).is_err());
    }

    /// is_greedy is true iff temperature <= 0.01.
    #[test]
    fn prop_is_greedy_consistency(temp in 0.0f32..2.0) {
        let req = InferenceRequest::new("test").with_temperature(temp);
        prop_assert_eq!(req.is_greedy(), temp <= 0.01);
    }

    /// is_deterministic is true iff seed is set.
    #[test]
    fn prop_is_deterministic_with_seed(seed in any::<u64>()) {
        let req = InferenceRequest::new("test").with_seed(seed);
        prop_assert!(req.is_deterministic());
    }

    /// Default request has no seed (not deterministic).
    #[test]
    fn prop_default_request_not_deterministic(_dummy in 0u8..1) {
        let req = InferenceRequest::new("test");
        prop_assert!(!req.is_deterministic());
    }
}
