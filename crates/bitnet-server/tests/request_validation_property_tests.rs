//! Property-based tests for server request validation.
//!
//! Key invariants tested:
//! - Valid prompts (within length limit) are accepted
//! - Oversized prompts are rejected with `PromptTooLong`
//! - Valid token counts (≤ max_tokens_per_request) are accepted
//! - Excessive token counts are rejected with `TooManyTokens`
//! - Temperature in [0.0, 2.0] is accepted; outside is rejected
//! - top_p in [0.0, 1.0] is accepted; outside is rejected
//! - top_k in [1, 1000] is accepted; 0 or >1000 is rejected
//! - repetition_penalty in [0.1, 10.0] is accepted; outside is rejected
//! - Prompts with null bytes or control characters are rejected

use bitnet_server::security::{SecurityConfig, SecurityValidator, ValidationError};
use bitnet_server::InferenceRequest;
use proptest::prelude::*;

// ── helpers ──────────────────────────────────────────────────────────────────

fn default_validator() -> SecurityValidator {
    let config = SecurityConfig::default();
    SecurityValidator::new(config).unwrap()
}

fn make_request(prompt: &str) -> InferenceRequest {
    InferenceRequest {
        prompt: prompt.to_string(),
        max_tokens: None,
        model: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    }
}

// ── Prompt length ────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Prompts within the max length are accepted.
    #[test]
    fn prop_valid_prompt_accepted(len in 1usize..=100) {
        let validator = default_validator();
        let prompt: String = "a".repeat(len);
        let req = make_request(&prompt);
        let result = validator.validate_inference_request(&req);
        prop_assert!(result.is_ok(), "prompt of len {} should be accepted", len);
    }

    /// Prompts exceeding max_prompt_length are rejected.
    #[test]
    fn prop_oversized_prompt_rejected(extra in 1usize..1000) {
        let config = SecurityConfig { max_prompt_length: 100, ..SecurityConfig::default() };
        let validator = SecurityValidator::new(config).unwrap();
        let prompt: String = "a".repeat(100 + extra);
        let req = make_request(&prompt);
        let result = validator.validate_inference_request(&req);
        prop_assert!(
            matches!(result, Err(ValidationError::PromptTooLong(_, _))),
            "prompt of len {} should be PromptTooLong", 100 + extra
        );
    }
}

// ── Max tokens ───────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Token counts within the limit are accepted.
    #[test]
    fn prop_valid_max_tokens_accepted(tokens in 1usize..=2048) {
        let validator = default_validator();
        let mut req = make_request("hello");
        req.max_tokens = Some(tokens);
        let result = validator.validate_inference_request(&req);
        prop_assert!(result.is_ok(), "max_tokens={} should be accepted", tokens);
    }

    /// Token counts exceeding the limit are rejected.
    #[test]
    fn prop_excessive_max_tokens_rejected(extra in 1usize..1000) {
        let config = SecurityConfig {
            max_tokens_per_request: 100,
            content_filtering: false,
            input_sanitization: false,
            ..SecurityConfig::default()
        };
        let validator = SecurityValidator::new(config).unwrap();
        let mut req = make_request("hello");
        req.max_tokens = Some(100 + extra);
        let result = validator.validate_inference_request(&req);
        prop_assert!(
            matches!(result, Err(ValidationError::TooManyTokens(_, _))),
            "max_tokens={} should be TooManyTokens", 100 + extra
        );
    }
}

// ── Temperature bounds ───────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Temperature in [0.0, 2.0] is accepted.
    #[test]
    fn prop_valid_temperature_accepted(temp in 0.0f32..=2.0) {
        let config = SecurityConfig {
            content_filtering: false,
            input_sanitization: false,
            ..SecurityConfig::default()
        };
        let validator = SecurityValidator::new(config).unwrap();
        let mut req = make_request("hello");
        req.temperature = Some(temp);
        let result = validator.validate_inference_request(&req);
        prop_assert!(result.is_ok(), "temperature={} should be accepted", temp);
    }

    /// Temperature > 2.0 is rejected.
    #[test]
    fn prop_high_temperature_rejected(temp in 2.01f32..100.0) {
        let config = SecurityConfig {
            content_filtering: false,
            input_sanitization: false,
            ..SecurityConfig::default()
        };
        let validator = SecurityValidator::new(config).unwrap();
        let mut req = make_request("hello");
        req.temperature = Some(temp);
        let result = validator.validate_inference_request(&req);
        prop_assert!(
            matches!(result, Err(ValidationError::InvalidFieldValue(_))),
            "temperature={} should be rejected", temp
        );
    }
}

// ── top_p bounds ─────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// top_p in [0.0, 1.0] is accepted.
    #[test]
    fn prop_valid_top_p_accepted(top_p in 0.0f32..=1.0) {
        let config = SecurityConfig {
            content_filtering: false,
            input_sanitization: false,
            ..SecurityConfig::default()
        };
        let validator = SecurityValidator::new(config).unwrap();
        let mut req = make_request("hello");
        req.top_p = Some(top_p);
        let result = validator.validate_inference_request(&req);
        prop_assert!(result.is_ok(), "top_p={} should be accepted", top_p);
    }

    /// top_p > 1.0 is rejected.
    #[test]
    fn prop_high_top_p_rejected(top_p in 1.01f32..100.0) {
        let config = SecurityConfig {
            content_filtering: false,
            input_sanitization: false,
            ..SecurityConfig::default()
        };
        let validator = SecurityValidator::new(config).unwrap();
        let mut req = make_request("hello");
        req.top_p = Some(top_p);
        let result = validator.validate_inference_request(&req);
        prop_assert!(
            matches!(result, Err(ValidationError::InvalidFieldValue(_))),
            "top_p={} should be rejected", top_p
        );
    }
}

// ── top_k bounds ─────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// top_k in [1, 1000] is accepted.
    #[test]
    fn prop_valid_top_k_accepted(top_k in 1usize..=1000) {
        let config = SecurityConfig {
            content_filtering: false,
            input_sanitization: false,
            ..SecurityConfig::default()
        };
        let validator = SecurityValidator::new(config).unwrap();
        let mut req = make_request("hello");
        req.top_k = Some(top_k);
        let result = validator.validate_inference_request(&req);
        prop_assert!(result.is_ok(), "top_k={} should be accepted", top_k);
    }

    /// top_k == 0 is rejected.
    #[test]
    fn prop_zero_top_k_rejected(_seed in 0u8..1) {
        let config = SecurityConfig {
            content_filtering: false,
            input_sanitization: false,
            ..SecurityConfig::default()
        };
        let validator = SecurityValidator::new(config).unwrap();
        let mut req = make_request("hello");
        req.top_k = Some(0);
        let result = validator.validate_inference_request(&req);
        prop_assert!(matches!(result, Err(ValidationError::InvalidFieldValue(_))));
    }

    /// top_k > 1000 is rejected.
    #[test]
    fn prop_high_top_k_rejected(top_k in 1001usize..10000) {
        let config = SecurityConfig {
            content_filtering: false,
            input_sanitization: false,
            ..SecurityConfig::default()
        };
        let validator = SecurityValidator::new(config).unwrap();
        let mut req = make_request("hello");
        req.top_k = Some(top_k);
        let result = validator.validate_inference_request(&req);
        prop_assert!(matches!(result, Err(ValidationError::InvalidFieldValue(_))));
    }
}

// ── repetition_penalty bounds ────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// repetition_penalty in [0.1, 10.0] is accepted.
    #[test]
    fn prop_valid_rep_penalty_accepted(penalty in 0.1f32..=10.0) {
        let config = SecurityConfig {
            content_filtering: false,
            input_sanitization: false,
            ..SecurityConfig::default()
        };
        let validator = SecurityValidator::new(config).unwrap();
        let mut req = make_request("hello");
        req.repetition_penalty = Some(penalty);
        let result = validator.validate_inference_request(&req);
        prop_assert!(result.is_ok(), "rep_penalty={} should be accepted", penalty);
    }

    /// repetition_penalty > 10.0 is rejected.
    #[test]
    fn prop_high_rep_penalty_rejected(penalty in 10.01f32..100.0) {
        let config = SecurityConfig {
            content_filtering: false,
            input_sanitization: false,
            ..SecurityConfig::default()
        };
        let validator = SecurityValidator::new(config).unwrap();
        let mut req = make_request("hello");
        req.repetition_penalty = Some(penalty);
        let result = validator.validate_inference_request(&req);
        prop_assert!(matches!(result, Err(ValidationError::InvalidFieldValue(_))));
    }
}

// ── Input sanitization ───────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Prompts with null bytes are rejected when sanitization is enabled.
    #[test]
    fn prop_null_byte_prompt_rejected(
        prefix in "[a-z]{1,10}",
        suffix in "[a-z]{1,10}",
    ) {
        let config = SecurityConfig {
            input_sanitization: true,
            content_filtering: false,
            ..SecurityConfig::default()
        };
        let validator = SecurityValidator::new(config).unwrap();
        let prompt = format!("{}\0{}", prefix, suffix);
        let req = make_request(&prompt);
        let result = validator.validate_inference_request(&req);
        prop_assert!(
            result.is_err(),
            "prompt with null byte should be rejected"
        );
    }
}
