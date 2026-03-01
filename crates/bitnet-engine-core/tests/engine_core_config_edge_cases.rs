//! Edge-case tests for bitnet-engine-core config validation, session management,
//! serde boundaries, and state-machine invariants.
//!
//! Focuses on boundary values, unusual inputs, and invariants not covered by
//! the existing property-based and unit test suites.

use std::collections::HashSet;

use bitnet_engine_core::{
    BackendInfo, ConcurrencyConfig, ConfigError, EngineState, EngineStateError,
    EngineStateTracker, SessionConfig, SessionId, SessionMetrics, VALID_BACKENDS,
};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn base_config() -> SessionConfig {
    SessionConfig {
        model_path: "m.gguf".into(),
        tokenizer_path: "t.json".into(),
        backend: "cpu".into(),
        max_context: 2048,
        seed: None,
    }
}

// ===========================================================================
// 1. SessionConfig validation — boundary & unusual inputs
// ===========================================================================

#[test]
fn whitespace_only_model_path_is_accepted() {
    // Whitespace is non-empty; validate() only checks emptiness.
    let cfg = SessionConfig { model_path: "   ".into(), ..base_config() };
    assert!(cfg.validate().is_ok());
}

#[test]
fn whitespace_only_tokenizer_path_is_accepted() {
    let cfg = SessionConfig { tokenizer_path: " \t ".into(), ..base_config() };
    assert!(cfg.validate().is_ok());
}

#[test]
fn backend_is_case_sensitive_uppercase_rejected() {
    let cfg = SessionConfig { backend: "CPU".into(), ..base_config() };
    assert_eq!(cfg.validate(), Err(ConfigError::UnsupportedBackend("CPU".into())));
}

#[test]
fn backend_with_trailing_space_rejected() {
    let cfg = SessionConfig { backend: "cpu ".into(), ..base_config() };
    assert_eq!(cfg.validate(), Err(ConfigError::UnsupportedBackend("cpu ".into())));
}

#[test]
fn max_context_one_is_valid() {
    let cfg = SessionConfig { max_context: 1, ..base_config() };
    assert!(cfg.validate().is_ok());
}

#[test]
fn max_context_usize_max_is_valid() {
    let cfg = SessionConfig { max_context: usize::MAX, ..base_config() };
    assert!(cfg.validate().is_ok());
}

#[test]
fn seed_zero_is_valid() {
    let cfg = SessionConfig { seed: Some(0), ..base_config() };
    assert!(cfg.validate().is_ok());
}

#[test]
fn seed_u64_max_is_valid() {
    let cfg = SessionConfig { seed: Some(u64::MAX), ..base_config() };
    assert!(cfg.validate().is_ok());
}

#[test]
fn unicode_model_path_is_accepted() {
    let cfg = SessionConfig { model_path: "модель/模型.gguf".into(), ..base_config() };
    assert!(cfg.validate().is_ok());
}

#[test]
fn very_long_path_is_accepted() {
    let long = "a".repeat(10_000);
    let cfg = SessionConfig { model_path: long, ..base_config() };
    assert!(cfg.validate().is_ok());
}

// ===========================================================================
// 2. ConcurrencyConfig — boundary values
// ===========================================================================

#[test]
fn concurrency_zero_max_denies_zero_active() {
    let cfg = ConcurrencyConfig { max_concurrent: 0 };
    // 0 < 0 is false
    assert!(!cfg.allows(0));
}

#[test]
fn concurrency_usize_max_allows_large_active() {
    let cfg = ConcurrencyConfig { max_concurrent: usize::MAX };
    assert!(cfg.allows(usize::MAX - 1));
    assert!(!cfg.allows(usize::MAX));
}

#[test]
fn concurrency_default_is_four() {
    let cfg = ConcurrencyConfig::default();
    assert_eq!(cfg.max_concurrent, 4);
}

#[test]
fn concurrency_boundary_exactly_one_below_limit() {
    for max in [1usize, 2, 10, 100] {
        let cfg = ConcurrencyConfig { max_concurrent: max };
        assert!(cfg.allows(max - 1), "allows(max-1) should be true for max={max}");
        assert!(!cfg.allows(max), "allows(max) should be false for max={max}");
    }
}

// ===========================================================================
// 3. BackendInfo — construction and serde edge cases
// ===========================================================================

#[test]
fn backend_info_empty_kernel_ids_roundtrip() {
    let info = BackendInfo {
        backend_name: "test".into(),
        kernel_ids: vec![],
        backend_summary: String::new(),
    };
    let json = serde_json::to_string(&info).unwrap();
    let back: BackendInfo = serde_json::from_str(&json).unwrap();
    assert!(back.kernel_ids.is_empty());
    assert_eq!(back.backend_name, "test");
}

#[test]
fn backend_info_many_kernel_ids() {
    let ids: Vec<String> = (0..1000).map(|i| format!("kernel_{i}")).collect();
    let info = BackendInfo {
        backend_name: "stress".into(),
        kernel_ids: ids.clone(),
        backend_summary: "stress test".into(),
    };
    let json = serde_json::to_string(&info).unwrap();
    let back: BackendInfo = serde_json::from_str(&json).unwrap();
    assert_eq!(back.kernel_ids.len(), 1000);
    assert_eq!(back.kernel_ids, ids);
}

#[test]
fn backend_info_special_chars_in_name() {
    let info = BackendInfo {
        backend_name: "cpu-avx2/512 (v3)".into(),
        kernel_ids: vec!["kern\"el".into()],
        backend_summary: "has \"quotes\" & <angles>".into(),
    };
    let json = serde_json::to_string(&info).unwrap();
    let back: BackendInfo = serde_json::from_str(&json).unwrap();
    assert_eq!(back.backend_name, info.backend_name);
    assert_eq!(back.kernel_ids, info.kernel_ids);
    assert_eq!(back.backend_summary, info.backend_summary);
}

#[test]
fn backend_info_debug_includes_name() {
    let info = BackendInfo {
        backend_name: "cpu-rust".into(),
        kernel_ids: vec![],
        backend_summary: String::new(),
    };
    let dbg = format!("{info:?}");
    assert!(dbg.contains("cpu-rust"));
}

// ===========================================================================
// 4. EngineStateTracker — state invariants after failed transitions
// ===========================================================================

#[test]
fn state_unchanged_after_failed_start_from_running() {
    let mut tracker = EngineStateTracker::new();
    tracker.start().unwrap();
    assert!(tracker.start().is_err());
    // State must still be Running after the failed call
    assert_eq!(*tracker.state(), EngineState::Running);
}

#[test]
fn state_unchanged_after_failed_finish_from_idle() {
    let mut tracker = EngineStateTracker::new();
    assert!(tracker.finish().is_err());
    assert_eq!(*tracker.state(), EngineState::Idle);
}

#[test]
fn state_unchanged_after_failed_start_from_done() {
    let mut tracker = EngineStateTracker::new();
    tracker.start().unwrap();
    tracker.finish().unwrap();
    assert!(tracker.start().is_err());
    assert_eq!(*tracker.state(), EngineState::Done);
}

#[test]
fn state_unchanged_after_failed_finish_from_done() {
    let mut tracker = EngineStateTracker::new();
    tracker.start().unwrap();
    tracker.finish().unwrap();
    assert!(tracker.finish().is_err());
    assert_eq!(*tracker.state(), EngineState::Done);
}

#[test]
fn error_message_from_running_mentions_running() {
    let mut tracker = EngineStateTracker::new();
    tracker.start().unwrap();
    let err = tracker.start().unwrap_err();
    assert!(
        err.to_string().contains("Running"),
        "expected 'Running' in error: {}",
        err
    );
}

#[test]
fn error_message_from_done_mentions_done() {
    let mut tracker = EngineStateTracker::new();
    tracker.start().unwrap();
    tracker.finish().unwrap();
    let err = tracker.start().unwrap_err();
    assert!(
        err.to_string().contains("Done"),
        "expected 'Done' in error: {}",
        err
    );
}

#[test]
fn engine_state_error_implements_std_error() {
    let err = EngineStateError("test error".into());
    let dyn_err: &dyn std::error::Error = &err;
    assert_eq!(dyn_err.to_string(), "test error");
}

#[test]
fn engine_state_serde_all_variants() {
    for state in [EngineState::Idle, EngineState::Running, EngineState::Done] {
        let json = serde_json::to_string(&state).unwrap();
        let back: EngineState = serde_json::from_str(&json).unwrap();
        assert_eq!(state, back);
    }
}

#[test]
fn engine_state_clone_equals_original() {
    let states = [EngineState::Idle, EngineState::Running, EngineState::Done];
    for s in &states {
        assert_eq!(*s, s.clone());
    }
}

// ===========================================================================
// 5. SessionMetrics — serde edge cases and extreme values
// ===========================================================================

#[test]
fn session_metrics_default_all_zero() {
    let m = SessionMetrics::default();
    assert_eq!(m.total_tokens, 0);
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(m.tokens_per_second, 0.0);
        assert_eq!(m.time_to_first_token_ms, 0.0);
    }
}

#[test]
fn session_metrics_large_values_roundtrip() {
    let m = SessionMetrics {
        tokens_per_second: 1_000_000.0,
        time_to_first_token_ms: 0.001,
        total_tokens: usize::MAX,
    };
    let json = serde_json::to_string(&m).unwrap();
    let back: SessionMetrics = serde_json::from_str(&json).unwrap();
    assert!((m.tokens_per_second - back.tokens_per_second).abs() < 1e-6);
    assert!((m.time_to_first_token_ms - back.time_to_first_token_ms).abs() < 1e-12);
    assert_eq!(m.total_tokens, back.total_tokens);
}

#[test]
fn session_metrics_fractional_tps_roundtrip() {
    let m = SessionMetrics {
        tokens_per_second: 0.1,
        time_to_first_token_ms: 5432.789,
        total_tokens: 7,
    };
    let json = serde_json::to_string(&m).unwrap();
    let back: SessionMetrics = serde_json::from_str(&json).unwrap();
    assert!((m.tokens_per_second - back.tokens_per_second).abs() < 1e-15);
    assert_eq!(m.total_tokens, back.total_tokens);
}

#[test]
fn session_metrics_json_has_expected_keys() {
    let m = SessionMetrics {
        tokens_per_second: 10.0,
        time_to_first_token_ms: 50.0,
        total_tokens: 100,
    };
    let json = serde_json::to_string(&m).unwrap();
    assert!(json.contains("tokens_per_second"));
    assert!(json.contains("time_to_first_token_ms"));
    assert!(json.contains("total_tokens"));
}

#[test]
fn session_metrics_clone_matches() {
    let m = SessionMetrics {
        tokens_per_second: 42.0,
        time_to_first_token_ms: 100.0,
        total_tokens: 256,
    };
    let m2 = m.clone();
    assert_eq!(m.total_tokens, m2.total_tokens);
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(m.tokens_per_second, m2.tokens_per_second);
    }
}

// ===========================================================================
// 6. ConfigError — variant messages and trait impls
// ===========================================================================

#[test]
fn config_error_display_empty_model() {
    let msg = ConfigError::EmptyModelPath.to_string();
    assert!(msg.contains("model_path"), "message should mention model_path: {msg}");
}

#[test]
fn config_error_display_empty_tokenizer() {
    let msg = ConfigError::EmptyTokenizerPath.to_string();
    assert!(msg.contains("tokenizer_path"), "message should mention tokenizer_path: {msg}");
}

#[test]
fn config_error_display_zero_context() {
    let msg = ConfigError::ZeroContextWindow.to_string();
    assert!(msg.contains("max_context"), "message should mention max_context: {msg}");
}

#[test]
fn config_error_unsupported_backend_includes_name() {
    let msg = ConfigError::UnsupportedBackend("vulkan".into()).to_string();
    assert!(msg.contains("vulkan"), "message should include the backend name: {msg}");
}

#[test]
fn config_error_unsupported_backend_empty_string() {
    let err = ConfigError::UnsupportedBackend(String::new());
    let msg = err.to_string();
    // Should still produce a meaningful message even for empty backend
    assert!(msg.contains("unsupported") || msg.contains("backend"), "msg: {msg}");
}

#[test]
fn config_error_implements_std_error() {
    let err = ConfigError::EmptyModelPath;
    let dyn_err: &dyn std::error::Error = &err;
    assert!(!dyn_err.to_string().is_empty());
}

#[test]
fn config_error_eq_same_variant() {
    assert_eq!(ConfigError::EmptyModelPath, ConfigError::EmptyModelPath);
    assert_eq!(ConfigError::ZeroContextWindow, ConfigError::ZeroContextWindow);
}

#[test]
fn config_error_ne_different_unsupported_backends() {
    let a = ConfigError::UnsupportedBackend("metal".into());
    let b = ConfigError::UnsupportedBackend("vulkan".into());
    assert_ne!(a, b);
}

#[test]
fn config_error_clone_equals() {
    let err = ConfigError::UnsupportedBackend("rocm".into());
    assert_eq!(err, err.clone());
}

// ===========================================================================
// 7. SessionId — uniqueness and formatting
// ===========================================================================

#[test]
fn session_id_format_prefix() {
    let id = SessionId::generate();
    assert!(id.as_str().starts_with("session-"), "got: {}", id.as_str());
}

#[test]
fn session_id_500_unique() {
    let ids: HashSet<String> = (0..500).map(|_| SessionId::generate().as_str().to_owned()).collect();
    assert_eq!(ids.len(), 500, "all 500 session IDs must be unique");
}

#[test]
fn session_id_serde_roundtrip() {
    let id = SessionId::generate();
    let json = serde_json::to_string(&id).unwrap();
    let back: SessionId = serde_json::from_str(&json).unwrap();
    assert_eq!(id, back);
    assert_eq!(id.as_str(), back.as_str());
}

#[test]
fn session_id_debug_contains_session_prefix() {
    let id = SessionId::generate();
    let dbg = format!("{id:?}");
    assert!(dbg.contains("session-"), "debug output: {dbg}");
}

#[test]
fn session_id_hash_consistent() {
    use std::collections::HashMap;
    let id = SessionId::generate();
    let mut map = HashMap::new();
    map.insert(id.clone(), "value");
    // Same key must retrieve the same value
    assert_eq!(map.get(&id), Some(&"value"));
    // Cloned key must also work
    let id2 = id.clone();
    assert_eq!(map.get(&id2), Some(&"value"));
}

#[test]
fn session_id_as_str_is_stable() {
    let id = SessionId::generate();
    let s1 = id.as_str();
    let s2 = id.as_str();
    assert_eq!(s1, s2);
}

// ===========================================================================
// 8. SessionConfig serde — edge-case payloads
// ===========================================================================

#[test]
fn session_config_serde_with_seed_none() {
    let cfg = SessionConfig { seed: None, ..base_config() };
    let json = serde_json::to_string(&cfg).unwrap();
    assert!(json.contains("null") || !json.contains("\"seed\":42"));
    let back: SessionConfig = serde_json::from_str(&json).unwrap();
    assert!(back.seed.is_none());
}

#[test]
fn session_config_serde_with_seed_some() {
    let cfg = SessionConfig { seed: Some(12345), ..base_config() };
    let json = serde_json::to_string(&cfg).unwrap();
    let back: SessionConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.seed, Some(12345));
}

#[test]
fn session_config_default_serde_roundtrip() {
    let cfg = SessionConfig::default();
    let json = serde_json::to_string(&cfg).unwrap();
    let back: SessionConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg.backend, back.backend);
    assert_eq!(cfg.max_context, back.max_context);
    assert_eq!(cfg.seed, back.seed);
}

#[test]
fn session_config_deserialize_from_json_literal() {
    let json = r#"{
        "model_path": "test.gguf",
        "tokenizer_path": "tok.json",
        "backend": "cuda",
        "max_context": 512,
        "seed": 99
    }"#;
    let cfg: SessionConfig = serde_json::from_str(json).unwrap();
    assert_eq!(cfg.model_path, "test.gguf");
    assert_eq!(cfg.backend, "cuda");
    assert_eq!(cfg.max_context, 512);
    assert_eq!(cfg.seed, Some(99));
}

// ===========================================================================
// 9. VALID_BACKENDS constant
// ===========================================================================

#[test]
fn valid_backends_has_at_least_cpu_and_cuda() {
    assert!(VALID_BACKENDS.contains(&"cpu"));
    assert!(VALID_BACKENDS.contains(&"cuda"));
}

#[test]
fn valid_backends_no_duplicates() {
    let set: HashSet<&&str> = VALID_BACKENDS.iter().collect();
    assert_eq!(set.len(), VALID_BACKENDS.len(), "VALID_BACKENDS has duplicates");
}

#[test]
fn valid_backends_all_lowercase() {
    for b in VALID_BACKENDS {
        assert_eq!(*b, b.to_lowercase(), "backend {b:?} should be lowercase");
    }
}

// ===========================================================================
// 10. ConcurrencyConfig serde edge cases
// ===========================================================================

#[test]
fn concurrency_config_serde_zero() {
    let cfg = ConcurrencyConfig { max_concurrent: 0 };
    let json = serde_json::to_string(&cfg).unwrap();
    let back: ConcurrencyConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.max_concurrent, 0);
}

#[test]
fn concurrency_config_deserialize_from_literal() {
    let json = r#"{"max_concurrent": 16}"#;
    let cfg: ConcurrencyConfig = serde_json::from_str(json).unwrap();
    assert_eq!(cfg.max_concurrent, 16);
}

// ===========================================================================
// 11. Combined: validate after serde roundtrip
// ===========================================================================

#[test]
fn valid_config_stays_valid_after_serde_roundtrip() {
    let cfg = base_config();
    assert!(cfg.validate().is_ok());
    let json = serde_json::to_string(&cfg).unwrap();
    let back: SessionConfig = serde_json::from_str(&json).unwrap();
    assert!(back.validate().is_ok());
}

#[test]
fn invalid_config_stays_invalid_after_serde_roundtrip() {
    let cfg = SessionConfig { max_context: 0, ..base_config() };
    assert!(cfg.validate().is_err());
    let json = serde_json::to_string(&cfg).unwrap();
    let back: SessionConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.validate(), Err(ConfigError::ZeroContextWindow));
}
