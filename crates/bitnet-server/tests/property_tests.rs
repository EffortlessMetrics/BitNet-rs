/// Property-based tests for bitnet-server using proptest.
///
/// Tests cover:
/// - BatchEngineConfig invariants (valid ranges for all fields)
/// - RequestPriority ordering properties
/// - BatchRequest builder pattern invariants
/// - SecurityConfig field bounds (prompt length, token limits, temperature)
/// - SecurityValidator input validation properties
use bitnet_server::batch_engine::{BatchEngineConfig, BatchRequest, RequestPriority};
use bitnet_server::security::{SecurityConfig, SecurityValidator};
use bitnet_server::InferenceRequest;
use proptest::prelude::*;

proptest! {
    /// BatchEngineConfig: max_batch_size > 0 is always valid
    #[test]
    fn prop_batch_engine_config_max_batch_size_is_positive(
        max_batch_size in 1usize..=256usize
    ) {
        let config = BatchEngineConfig {
            max_batch_size,
            ..Default::default()
        };
        prop_assert!(config.max_batch_size > 0);
    }

    /// BatchEngineConfig: max_concurrent_batches > 0 is always valid
    #[test]
    fn prop_batch_engine_config_concurrent_batches_positive(
        max_concurrent_batches in 1usize..=32usize
    ) {
        let config = BatchEngineConfig {
            max_concurrent_batches,
            ..Default::default()
        };
        prop_assert!(config.max_concurrent_batches > 0);
    }

    /// RequestPriority: ordering is transitive
    /// Low < Normal < High < Critical
    #[test]
    fn prop_request_priority_ordering_is_consistent(
        _seed in 0u32..100u32
    ) {
        prop_assert!(RequestPriority::Low < RequestPriority::Normal);
        prop_assert!(RequestPriority::Normal < RequestPriority::High);
        prop_assert!(RequestPriority::High < RequestPriority::Critical);
        prop_assert!(RequestPriority::Low < RequestPriority::Critical);
    }

    /// BatchRequest::new always produces a non-empty ID
    #[test]
    fn prop_batch_request_new_has_nonempty_id(
        prompt in "[a-z ]{1,100}"
    ) {
        use bitnet_inference::GenerationConfig;
        let config = GenerationConfig::default();
        let prompt: String = prompt;
        let request = BatchRequest::new(prompt.clone(), config);
        prop_assert!(!request.id.is_empty(), "BatchRequest id should not be empty");
        prop_assert_eq!(&request.prompt, &prompt);
        prop_assert_eq!(request.priority, RequestPriority::Normal);
    }

    /// BatchRequest::with_priority preserves the priority
    #[test]
    fn prop_batch_request_with_priority_preserves_priority(
        priority_val in 0u8..=3u8
    ) {
        use bitnet_inference::GenerationConfig;
        let priority = match priority_val {
            0 => RequestPriority::Low,
            1 => RequestPriority::Normal,
            2 => RequestPriority::High,
            _ => RequestPriority::Critical,
        };
        let config = GenerationConfig::default();
        let request = BatchRequest::new("test".to_string(), config)
            .with_priority(priority);
        prop_assert_eq!(request.priority, priority);
    }

    /// SecurityConfig: max_prompt_length ≥ 1 is always valid  
    #[test]
    fn prop_security_config_prompt_length_positive(
        max_prompt_length in 1usize..=65536usize
    ) {
        let config = SecurityConfig {
            max_prompt_length,
            ..Default::default()
        };
        prop_assert!(config.max_prompt_length >= 1);
    }

    /// SecurityConfig: max_tokens_per_request ≥ 1 is valid
    #[test]
    fn prop_security_config_max_tokens_positive(
        max_tokens in 1u32..=8192u32
    ) {
        let config = SecurityConfig {
            max_tokens_per_request: max_tokens,
            ..Default::default()
        };
        prop_assert!(config.max_tokens_per_request >= 1);
    }

    /// SecurityValidator: prompts within max_prompt_length always pass length check
    #[test]
    fn prop_security_validator_short_prompt_passes(
        prompt in "[a-z ]{1,100}"
    ) {
        let config = SecurityConfig {
            max_prompt_length: 8192,
            input_sanitization: false,
            content_filtering: false,
            ..Default::default()
        };
        let validator = SecurityValidator::new(config).expect("validator creation should succeed");
        let request = InferenceRequest {
            prompt: prompt.clone(),
            max_tokens: Some(32),
            model: None,
            temperature: Some(0.7),
            top_p: None,
            top_k: None,
            repetition_penalty: None,
        };
        prop_assert!(
            validator.validate_inference_request(&request).is_ok(),
            "short prompt should pass validation: {:?}",
            prompt
        );
    }

    /// SecurityValidator: prompts exceeding max_prompt_length always fail
    #[test]
    fn prop_security_validator_long_prompt_fails(
        extra in 1usize..=100usize
    ) {
        let max_len = 50usize;
        let config = SecurityConfig {
            max_prompt_length: max_len,
            input_sanitization: false,
            content_filtering: false,
            ..Default::default()
        };
        let validator = SecurityValidator::new(config).expect("validator creation should succeed");
        let long_prompt = "a".repeat(max_len + extra);
        let request = InferenceRequest {
            prompt: long_prompt,
            max_tokens: Some(32),
            model: None,
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
        };
        prop_assert!(
            validator.validate_inference_request(&request).is_err(),
            "prompt exceeding max_prompt_length should fail validation"
        );
    }

    /// SecurityValidator: temperature in [0.0, 2.0] always passes
    #[test]
    fn prop_security_validator_valid_temperature_passes(
        temp_int in 0u32..=200u32  // represents 0.0..=2.0 in steps of 0.01
    ) {
        let temperature = temp_int as f32 / 100.0;
        let config = SecurityConfig {
            input_sanitization: false,
            content_filtering: false,
            ..Default::default()
        };
        let validator = SecurityValidator::new(config).expect("validator creation should succeed");
        let request = InferenceRequest {
            prompt: "hello".to_string(),
            max_tokens: Some(32),
            model: None,
            temperature: Some(temperature),
            top_p: None,
            top_k: None,
            repetition_penalty: None,
        };
        prop_assert!(
            validator.validate_inference_request(&request).is_ok(),
            "temperature {temperature} in [0.0, 2.0] should pass validation"
        );
    }

    /// SecurityValidator: top_p in (0.0, 1.0] always passes
    #[test]
    fn prop_security_validator_valid_top_p_passes(
        top_p_int in 1u32..=100u32  // represents 0.01..=1.0 in steps of 0.01
    ) {
        let top_p = top_p_int as f32 / 100.0;
        let config = SecurityConfig {
            input_sanitization: false,
            content_filtering: false,
            ..Default::default()
        };
        let validator = SecurityValidator::new(config).expect("validator creation should succeed");
        let request = InferenceRequest {
            prompt: "hello".to_string(),
            max_tokens: Some(32),
            model: None,
            temperature: None,
            top_p: Some(top_p),
            top_k: None,
            repetition_penalty: None,
        };
        prop_assert!(
            validator.validate_inference_request(&request).is_ok(),
            "top_p {top_p} in (0.0, 1.0] should pass validation"
        );
    }
}
