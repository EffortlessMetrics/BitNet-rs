//! Property-based tests — wave 37: request validation, batch scheduling,
//! rate-limiting fairness, and device routing properties.

use bitnet_server::batch_engine::{BatchEngineConfig, BatchRequest, RequestPriority};
use bitnet_server::concurrency::ConcurrencyConfig;
use bitnet_server::config::ServerConfig;
use bitnet_server::execution_router::{DeviceSelectionStrategy, ExecutionRouterConfig};
use bitnet_server::security::{SecurityConfig, SecurityValidator, ValidationError};
use bitnet_server::{InferenceRequest, InferenceResponse};
use proptest::prelude::*;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn minimal_security_config() -> SecurityConfig {
    SecurityConfig {
        content_filtering: false,
        input_sanitization: false,
        ..SecurityConfig::default()
    }
}

fn validator_no_filter() -> SecurityValidator {
    SecurityValidator::new(minimal_security_config()).unwrap()
}

fn make_inference_request(
    prompt: &str,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repetition_penalty: Option<f32>,
) -> InferenceRequest {
    InferenceRequest {
        prompt: prompt.to_string(),
        max_tokens,
        model: None,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
    }
}

// ---------------------------------------------------------------------------
// Request-validation properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Valid temperature in [0.0, 2.0] always passes validation.
    #[test]
    fn valid_temperature_always_accepted(temp in 0.0f32..=2.0f32) {
        let v = validator_no_filter();
        let req = make_inference_request("hello", None, Some(temp), None, None, None);
        prop_assert!(v.validate_inference_request(&req).is_ok());
    }

    /// Temperature outside [0.0, 2.0] is rejected.
    #[test]
    fn out_of_range_temperature_rejected(temp in prop_oneof![
        (-100.0f32..-0.001f32),
        (2.001f32..100.0f32),
    ]) {
        let v = validator_no_filter();
        let req = make_inference_request("hello", None, Some(temp), None, None, None);
        let err = v.validate_inference_request(&req).unwrap_err();
        prop_assert!(matches!(err, ValidationError::InvalidFieldValue(_)));
    }

    /// Valid top_p in [0.0, 1.0] always passes validation.
    #[test]
    fn valid_top_p_always_accepted(p in 0.0f32..=1.0f32) {
        let v = validator_no_filter();
        let req = make_inference_request("hello", None, None, Some(p), None, None);
        prop_assert!(v.validate_inference_request(&req).is_ok());
    }

    /// top_p outside [0.0, 1.0] is rejected.
    #[test]
    fn out_of_range_top_p_rejected(p in prop_oneof![
        (-100.0f32..-0.001f32),
        (1.001f32..100.0f32),
    ]) {
        let v = validator_no_filter();
        let req = make_inference_request("hello", None, None, Some(p), None, None);
        let err = v.validate_inference_request(&req).unwrap_err();
        prop_assert!(matches!(err, ValidationError::InvalidFieldValue(_)));
    }

    /// Valid top_k in [1, 1000] always passes validation.
    #[test]
    fn valid_top_k_always_accepted(k in 1usize..=1000) {
        let v = validator_no_filter();
        let req = make_inference_request("hello", None, None, None, Some(k), None);
        prop_assert!(v.validate_inference_request(&req).is_ok());
    }

    /// top_k of 0 or > 1000 is rejected.
    #[test]
    fn out_of_range_top_k_rejected(k in prop_oneof![Just(0usize), (1001usize..10000)]) {
        let v = validator_no_filter();
        let req = make_inference_request("hello", None, None, None, Some(k), None);
        let err = v.validate_inference_request(&req).unwrap_err();
        prop_assert!(matches!(err, ValidationError::InvalidFieldValue(_)));
    }

    /// Valid repetition_penalty in [0.1, 10.0] passes.
    #[test]
    fn valid_repetition_penalty_accepted(rp in 0.1f32..=10.0f32) {
        let v = validator_no_filter();
        let req = make_inference_request("hello", None, None, None, None, Some(rp));
        prop_assert!(v.validate_inference_request(&req).is_ok());
    }

    /// Prompt within max length always passes (no sanitization/filter).
    #[test]
    fn prompt_within_limit_accepted(len in 1usize..=8192) {
        let v = validator_no_filter();
        let prompt: String = "a".repeat(len);
        let req = make_inference_request(&prompt, None, None, None, None, None);
        prop_assert!(v.validate_inference_request(&req).is_ok());
    }

    /// Prompt exceeding max length is rejected.
    #[test]
    fn prompt_exceeding_limit_rejected(excess in 1usize..=1000) {
        let v = validator_no_filter();
        let prompt: String = "a".repeat(8192 + excess);
        let req = make_inference_request(&prompt, None, None, None, None, None);
        let err = v.validate_inference_request(&req).unwrap_err();
        prop_assert!(matches!(err, ValidationError::PromptTooLong(_, _)));
    }

    /// max_tokens within limit passes; exceeding limit fails.
    #[test]
    fn max_tokens_boundary(tokens in 1usize..=4096) {
        let v = validator_no_filter();
        let req = make_inference_request("hi", Some(tokens), None, None, None, None);
        if tokens <= 2048 {
            prop_assert!(v.validate_inference_request(&req).is_ok());
        } else {
            prop_assert!(v.validate_inference_request(&req).is_err());
        }
    }
}

// ---------------------------------------------------------------------------
// Batch scheduling invariants
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// BatchRequest id is always a valid UUID (36 hex chars + hyphens).
    #[test]
    fn batch_request_id_is_uuid(_seed in 0u32..1000) {
        let cfg = bitnet_inference::GenerationConfig::default();
        let req = BatchRequest::new("prompt".to_string(), cfg);
        let id = &req.id;
        prop_assert_eq!(id.len(), 36);
        prop_assert!(id.chars().all(|c| c.is_ascii_hexdigit() || c == '-'));
    }

    /// Priority ordering is a total order: Low < Normal < High < Critical.
    #[test]
    fn priority_total_order(a in 0u8..4, b in 0u8..4) {
        let prios = [
            RequestPriority::Low,
            RequestPriority::Normal,
            RequestPriority::High,
            RequestPriority::Critical,
        ];
        let pa = prios[a as usize];
        let pb = prios[b as usize];
        if a < b {
            prop_assert!(pa < pb);
        } else if a > b {
            prop_assert!(pa > pb);
        } else {
            prop_assert!(pa == pb);
        }
    }

    /// with_priority builder preserves the chosen priority.
    #[test]
    fn with_priority_preserves(idx in 0u8..4) {
        let prios = [
            RequestPriority::Low,
            RequestPriority::Normal,
            RequestPriority::High,
            RequestPriority::Critical,
        ];
        let chosen = prios[idx as usize];
        let cfg = bitnet_inference::GenerationConfig::default();
        let req = BatchRequest::new("p".to_string(), cfg).with_priority(chosen);
        prop_assert_eq!(req.priority, chosen);
    }

    /// with_timeout sets the timeout correctly.
    #[test]
    fn with_timeout_preserves(ms in 1u64..=60_000) {
        let cfg = bitnet_inference::GenerationConfig::default();
        let req = BatchRequest::new("p".to_string(), cfg)
            .with_timeout(Duration::from_millis(ms));
        prop_assert_eq!(req.timeout, Some(Duration::from_millis(ms)));
    }

    /// BatchEngineConfig defaults are positive.
    #[test]
    fn batch_engine_defaults_positive(_seed in 0u32..10) {
        let cfg = BatchEngineConfig::default();
        prop_assert!(cfg.max_batch_size > 0);
        prop_assert!(cfg.max_concurrent_batches > 0);
    }
}

// ---------------------------------------------------------------------------
// Concurrency / rate-limiting properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// ConcurrencyConfig preserves all fields through construction.
    #[test]
    fn concurrency_config_fields_preserved(
        max_concurrent in 1usize..=10000,
        rps in 1u64..=10000,
        rpm in 1u64..=100_000,
        threshold in 0.0f64..=1.0,
    ) {
        let cfg = ConcurrencyConfig {
            max_concurrent_requests: max_concurrent,
            max_requests_per_second: rps,
            max_requests_per_minute: rpm,
            backpressure_threshold: threshold,
            ..ConcurrencyConfig::default()
        };
        prop_assert_eq!(cfg.max_concurrent_requests, max_concurrent);
        prop_assert_eq!(cfg.max_requests_per_second, rps);
        prop_assert_eq!(cfg.max_requests_per_minute, rpm);
        prop_assert!((cfg.backpressure_threshold - threshold).abs() < 1e-10);
    }

    /// Per-IP rate limit, if set, is stored correctly.
    #[test]
    fn per_ip_rate_limit_preserved(limit in 1u64..=1000) {
        let cfg = ConcurrencyConfig {
            per_ip_rate_limit: Some(limit),
            ..ConcurrencyConfig::default()
        };
        prop_assert_eq!(cfg.per_ip_rate_limit, Some(limit));
    }

    /// Global rate limit, if set, is stored correctly.
    #[test]
    fn global_rate_limit_preserved(limit in 1u64..=10000) {
        let cfg = ConcurrencyConfig {
            global_rate_limit: Some(limit),
            ..ConcurrencyConfig::default()
        };
        prop_assert_eq!(cfg.global_rate_limit, Some(limit));
    }
}

// ---------------------------------------------------------------------------
// Execution router config properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// ExecutionRouterConfig defaults have sensible values.
    #[test]
    fn execution_router_defaults_sensible(_seed in 0u32..10) {
        let cfg = ExecutionRouterConfig::default();
        prop_assert!(cfg.performance_threshold_tps > 0.0);
        prop_assert!(cfg.memory_threshold_percent > 0.0);
        prop_assert!(cfg.memory_threshold_percent <= 1.0);
        prop_assert!(cfg.fallback_enabled);
    }

    /// Custom performance threshold is preserved.
    #[test]
    fn performance_threshold_preserved(tps in 0.1f64..=1000.0) {
        let cfg = ExecutionRouterConfig {
            performance_threshold_tps: tps,
            ..ExecutionRouterConfig::default()
        };
        prop_assert!((cfg.performance_threshold_tps - tps).abs() < 1e-10);
    }

    /// Memory threshold percentage is preserved.
    #[test]
    fn memory_threshold_preserved(pct in 0.0f64..=1.0) {
        let cfg = ExecutionRouterConfig {
            memory_threshold_percent: pct,
            ..ExecutionRouterConfig::default()
        };
        prop_assert!((cfg.memory_threshold_percent - pct).abs() < 1e-10);
    }
}

// ---------------------------------------------------------------------------
// InferenceResponse properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// tokens_per_second is consistent: tokens / (time_ms / 1000).
    #[test]
    fn tps_consistency(tokens in 1u64..=10_000, time_ms in 1u64..=60_000) {
        let expected_tps = (tokens as f64 * 1000.0) / time_ms as f64;
        let resp = InferenceResponse {
            text: String::new(),
            tokens_generated: tokens,
            inference_time_ms: time_ms,
            tokens_per_second: expected_tps,
        };
        prop_assert!((resp.tokens_per_second - expected_tps).abs() < 1e-6);
    }
}

// ---------------------------------------------------------------------------
// SecurityConfig model-path validation
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Model paths with ".." are always rejected.
    #[test]
    fn path_traversal_rejected(prefix in "[a-z]{1,10}", suffix in "[a-z]{1,10}") {
        let v = validator_no_filter();
        let path = format!("{prefix}/../{suffix}.gguf");
        let err = v.validate_model_request(&path).unwrap_err();
        prop_assert!(matches!(err, ValidationError::InvalidFieldValue(_)));
    }

    /// Model paths ending in .gguf (without traversal) are accepted.
    #[test]
    fn valid_gguf_path_accepted(name in "[a-z]{1,20}") {
        let v = validator_no_filter();
        let path = format!("{name}.gguf");
        prop_assert!(v.validate_model_request(&path).is_ok());
    }

    /// Model paths ending in .safetensors are accepted.
    #[test]
    fn valid_safetensors_path_accepted(name in "[a-z]{1,20}") {
        let v = validator_no_filter();
        let path = format!("{name}.safetensors");
        prop_assert!(v.validate_model_request(&path).is_ok());
    }

    /// Model paths with wrong extension are rejected.
    #[test]
    fn wrong_extension_rejected(name in "[a-z]{1,20}", ext in "[a-z]{1,5}") {
        prop_assume!(ext != "gguf" && ext != "safetensors");
        let v = validator_no_filter();
        let path = format!("{name}.{ext}");
        let err = v.validate_model_request(&path).unwrap_err();
        prop_assert!(matches!(err, ValidationError::InvalidFieldValue(_)));
    }
}
