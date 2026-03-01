//! Edge-case tests for `bitnet-engine-core` config and session management.
//!
//! Focuses on boundary values, corner cases, and behaviours NOT covered by
//! existing test suites (engine_core_edge_cases, session_management_edge_cases,
//! validation_and_state_tests, engine_core_tests).

use bitnet_engine_core::{
    BackendInfo, ConcurrencyConfig, ConfigError, EngineState, EngineStateError, EngineStateTracker,
    SessionConfig, SessionId, SessionMetrics, VALID_BACKENDS,
};
use std::collections::HashSet;

// =========================================================================
// SessionConfig — boundary & corner-case validation
// =========================================================================

#[test]
fn whitespace_only_model_path_passes_validation() {
    // Whitespace is not empty—validate() only checks `.is_empty()`.
    let cfg = SessionConfig {
        model_path: "   ".into(),
        tokenizer_path: "t.json".into(),
        backend: "cpu".into(),
        max_context: 1,
        seed: None,
    };
    assert!(cfg.validate().is_ok());
}

#[test]
fn whitespace_only_tokenizer_path_passes_validation() {
    let cfg = SessionConfig {
        model_path: "m.gguf".into(),
        tokenizer_path: " \t ".into(),
        backend: "cpu".into(),
        max_context: 1,
        seed: None,
    };
    assert!(cfg.validate().is_ok());
}

#[test]
fn max_context_usize_max_is_valid() {
    let cfg = SessionConfig {
        model_path: "m.gguf".into(),
        tokenizer_path: "t.json".into(),
        backend: "cpu".into(),
        max_context: usize::MAX,
        seed: None,
    };
    assert!(cfg.validate().is_ok());
}

#[test]
fn seed_boundary_values() {
    for seed in [Some(0), Some(1), Some(u64::MAX), None] {
        let cfg = SessionConfig {
            model_path: "m.gguf".into(),
            tokenizer_path: "t.json".into(),
            backend: "cpu".into(),
            max_context: 512,
            seed,
        };
        assert!(cfg.validate().is_ok(), "seed={seed:?} should be valid");
    }
}

#[test]
fn backend_is_case_sensitive() {
    for name in ["CPU", "Cpu", "CUDA", "GPU", "FFI"] {
        let cfg = SessionConfig {
            model_path: "m.gguf".into(),
            tokenizer_path: "t.json".into(),
            backend: name.into(),
            max_context: 1,
            seed: None,
        };
        assert_eq!(
            cfg.validate(),
            Err(ConfigError::UnsupportedBackend(name.into())),
            "{name:?} should be rejected (case sensitive)"
        );
    }
}

#[test]
fn empty_backend_string_is_unsupported() {
    let cfg = SessionConfig {
        model_path: "m.gguf".into(),
        tokenizer_path: "t.json".into(),
        backend: String::new(),
        max_context: 1,
        seed: None,
    };
    assert_eq!(cfg.validate(), Err(ConfigError::UnsupportedBackend(String::new())));
}

#[test]
fn unicode_paths_pass_non_empty_check() {
    let cfg = SessionConfig {
        model_path: "模型/model.gguf".into(),
        tokenizer_path: "トークナイザ/tok.json".into(),
        backend: "cpu".into(),
        max_context: 256,
        seed: None,
    };
    assert!(cfg.validate().is_ok());
}

#[test]
fn session_config_clone_is_independent() {
    let original = SessionConfig {
        model_path: "m.gguf".into(),
        tokenizer_path: "t.json".into(),
        backend: "cuda".into(),
        max_context: 4096,
        seed: Some(42),
    };
    let mut cloned = original.clone();
    cloned.backend = "cpu".into();
    cloned.max_context = 1;
    // Original is unchanged.
    assert_eq!(original.backend, "cuda");
    assert_eq!(original.max_context, 4096);
}

#[test]
fn session_config_serde_preserves_unicode_paths() {
    let cfg = SessionConfig {
        model_path: "données/modèle.gguf".into(),
        tokenizer_path: "datos/tokenizador.json".into(),
        backend: "cpu".into(),
        max_context: 128,
        seed: Some(0),
    };
    let json = serde_json::to_string(&cfg).unwrap();
    let back: SessionConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.model_path, cfg.model_path);
    assert_eq!(back.tokenizer_path, cfg.tokenizer_path);
}

#[test]
fn session_config_serde_with_max_seed() {
    let cfg = SessionConfig {
        model_path: "m.gguf".into(),
        tokenizer_path: "t.json".into(),
        backend: "ffi".into(),
        max_context: 1,
        seed: Some(u64::MAX),
    };
    let json = serde_json::to_string(&cfg).unwrap();
    let back: SessionConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.seed, Some(u64::MAX));
}

// =========================================================================
// ConcurrencyConfig — boundary behaviour
// =========================================================================

#[test]
fn concurrency_usize_max_allows_all_below() {
    let cfg = ConcurrencyConfig { max_concurrent: usize::MAX };
    assert!(cfg.allows(0));
    assert!(cfg.allows(usize::MAX - 1));
    // Exactly usize::MAX active is NOT allowed (active < max).
    assert!(!cfg.allows(usize::MAX));
}

#[test]
fn concurrency_one_allows_only_zero() {
    let cfg = ConcurrencyConfig { max_concurrent: 1 };
    assert!(cfg.allows(0));
    assert!(!cfg.allows(1));
    assert!(!cfg.allows(2));
}

#[test]
fn concurrency_zero_denies_everything() {
    let cfg = ConcurrencyConfig { max_concurrent: 0 };
    assert!(!cfg.allows(0));
    assert!(!cfg.allows(1));
    assert!(!cfg.allows(usize::MAX));
}

#[test]
fn concurrency_serde_with_extreme_value() {
    let cfg = ConcurrencyConfig { max_concurrent: usize::MAX };
    let json = serde_json::to_string(&cfg).unwrap();
    let back: ConcurrencyConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.max_concurrent, usize::MAX);
}

#[test]
fn concurrency_clone_is_independent() {
    let original = ConcurrencyConfig { max_concurrent: 8 };
    let cloned = original.clone();
    assert_eq!(cloned.max_concurrent, 8);
    assert_eq!(original.max_concurrent, 8);
}

// =========================================================================
// BackendInfo — edge cases
// =========================================================================

#[test]
fn backend_info_with_many_kernel_ids() {
    let ids: Vec<String> = (0..1000).map(|i| format!("kernel_{i}")).collect();
    let info = BackendInfo {
        backend_name: "stress-test".into(),
        kernel_ids: ids.clone(),
        backend_summary: "1000 kernels".into(),
    };
    assert_eq!(info.kernel_ids.len(), 1000);
    assert_eq!(info.kernel_ids[999], "kernel_999");
}

#[test]
fn backend_info_kernel_ids_preserve_order_after_serde() {
    let ids = vec!["z_last".into(), "a_first".into(), "m_middle".into()];
    let info = BackendInfo {
        backend_name: "test".into(),
        kernel_ids: ids.clone(),
        backend_summary: String::new(),
    };
    let json = serde_json::to_string(&info).unwrap();
    let back: BackendInfo = serde_json::from_str(&json).unwrap();
    assert_eq!(back.kernel_ids, ids, "kernel_ids order must be preserved");
}

#[test]
fn backend_info_allows_empty_strings_in_kernel_ids() {
    let info = BackendInfo {
        backend_name: String::new(),
        kernel_ids: vec![String::new(), String::new()],
        backend_summary: String::new(),
    };
    let json = serde_json::to_string(&info).unwrap();
    let back: BackendInfo = serde_json::from_str(&json).unwrap();
    assert_eq!(back.kernel_ids, vec!["", ""]);
}

#[test]
fn backend_info_unicode_fields() {
    let info = BackendInfo {
        backend_name: "バックエンド".into(),
        kernel_ids: vec!["カーネル_1".into()],
        backend_summary: "概要テスト".into(),
    };
    let json = serde_json::to_string(&info).unwrap();
    let back: BackendInfo = serde_json::from_str(&json).unwrap();
    assert_eq!(back.backend_name, "バックエンド");
    assert_eq!(back.backend_summary, "概要テスト");
}

// =========================================================================
// EngineStateTracker — failed transitions preserve state
// =========================================================================

#[test]
fn failed_start_from_running_preserves_running() {
    let mut t = EngineStateTracker::new();
    t.start().unwrap();
    assert!(t.start().is_err());
    assert_eq!(t.state(), &EngineState::Running);
}

#[test]
fn failed_finish_from_idle_preserves_idle() {
    let mut t = EngineStateTracker::new();
    assert!(t.finish().is_err());
    assert_eq!(t.state(), &EngineState::Idle);
}

#[test]
fn failed_start_from_done_preserves_done() {
    let mut t = EngineStateTracker::new();
    t.start().unwrap();
    t.finish().unwrap();
    assert!(t.start().is_err());
    assert_eq!(t.state(), &EngineState::Done);
}

#[test]
fn failed_finish_from_done_preserves_done() {
    let mut t = EngineStateTracker::new();
    t.start().unwrap();
    t.finish().unwrap();
    assert!(t.finish().is_err());
    assert_eq!(t.state(), &EngineState::Done);
}

#[test]
fn multiple_failures_do_not_corrupt_state() {
    let mut t = EngineStateTracker::new();
    // Repeatedly fail finish from Idle—state stays Idle.
    for _ in 0..10 {
        assert!(t.finish().is_err());
    }
    assert_eq!(t.state(), &EngineState::Idle);
    // Transition through the happy path still works.
    t.start().unwrap();
    t.finish().unwrap();
    assert_eq!(t.state(), &EngineState::Done);
}

#[test]
fn error_message_from_running_contains_running() {
    let mut t = EngineStateTracker::new();
    t.start().unwrap();
    let err = t.start().unwrap_err();
    assert!(err.to_string().contains("Running"), "error should mention current state: {}", err);
}

#[test]
fn error_message_from_done_contains_done() {
    let mut t = EngineStateTracker::new();
    t.start().unwrap();
    t.finish().unwrap();
    let err = t.start().unwrap_err();
    assert!(err.to_string().contains("Done"), "error should mention Done: {err}");
}

#[test]
fn error_message_finish_from_idle_contains_idle() {
    let mut t = EngineStateTracker::new();
    let err = t.finish().unwrap_err();
    assert!(err.to_string().contains("Idle"), "error should mention Idle: {err}");
}

#[test]
fn engine_state_error_implements_std_error() {
    let err = EngineStateError("test error".into());
    let std_err: &dyn std::error::Error = &err;
    assert!(!std_err.to_string().is_empty());
    // source() returns None (no chained cause).
    assert!(std_err.source().is_none());
}

// =========================================================================
// EngineState — serde edge cases
// =========================================================================

#[test]
fn engine_state_all_variants_serde_roundtrip() {
    for state in [EngineState::Idle, EngineState::Running, EngineState::Done] {
        let json = serde_json::to_string(&state).unwrap();
        let back: EngineState = serde_json::from_str(&json).unwrap();
        assert_eq!(back, state);
    }
}

#[test]
fn engine_state_invalid_json_fails_deserialization() {
    let result = serde_json::from_str::<EngineState>(r#""InvalidState""#);
    assert!(result.is_err());
}

// =========================================================================
// SessionMetrics — special float values in serde
// =========================================================================

#[test]
fn session_metrics_nan_serializes_to_null() {
    let m = SessionMetrics {
        tokens_per_second: f64::NAN,
        time_to_first_token_ms: 0.0,
        total_tokens: 0,
    };
    // NaN serializes (to JSON null) but cannot roundtrip back to f64.
    let json = serde_json::to_string(&m).unwrap();
    assert!(json.contains("null"), "NaN should serialize as null: {json}");
    assert!(serde_json::from_str::<SessionMetrics>(&json).is_err());
}

#[test]
fn session_metrics_infinity_serializes_to_null() {
    let m = SessionMetrics {
        tokens_per_second: f64::INFINITY,
        time_to_first_token_ms: f64::NEG_INFINITY,
        total_tokens: 0,
    };
    // Infinities serialize (to JSON null) but cannot roundtrip.
    let json = serde_json::to_string(&m).unwrap();
    assert!(json.contains("null"), "Infinity should serialize as null: {json}");
    assert!(serde_json::from_str::<SessionMetrics>(&json).is_err());
}

#[test]
fn session_metrics_large_values_roundtrip() {
    let m = SessionMetrics {
        tokens_per_second: 1e308,
        time_to_first_token_ms: 1e15,
        total_tokens: usize::MAX,
    };
    let json = serde_json::to_string(&m).unwrap();
    let back: SessionMetrics = serde_json::from_str(&json).unwrap();
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(back.tokens_per_second, 1e308);
        assert_eq!(back.time_to_first_token_ms, 1e15);
    }
    assert_eq!(back.total_tokens, usize::MAX);
}

#[test]
fn session_metrics_negative_floats_roundtrip() {
    // Negative metrics are semantically wrong but structurally valid.
    let m =
        SessionMetrics { tokens_per_second: -1.0, time_to_first_token_ms: -0.001, total_tokens: 0 };
    let json = serde_json::to_string(&m).unwrap();
    let back: SessionMetrics = serde_json::from_str(&json).unwrap();
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(back.tokens_per_second, -1.0);
        assert_eq!(back.time_to_first_token_ms, -0.001);
    }
}

#[test]
fn session_metrics_zero_is_distinct_from_negative_zero() {
    let pos = SessionMetrics { tokens_per_second: 0.0, ..Default::default() };
    let neg = SessionMetrics { tokens_per_second: -0.0, ..Default::default() };
    // IEEE 754: 0.0 == -0.0, but bits differ.
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(pos.tokens_per_second, neg.tokens_per_second);
    }
    assert_ne!(pos.tokens_per_second.to_bits(), neg.tokens_per_second.to_bits());
}

// =========================================================================
// ConfigError — display messages & equality
// =========================================================================

#[test]
fn config_error_display_exact_messages() {
    assert_eq!(ConfigError::EmptyModelPath.to_string(), "model_path must not be empty");
    assert_eq!(ConfigError::EmptyTokenizerPath.to_string(), "tokenizer_path must not be empty");
    assert_eq!(ConfigError::ZeroContextWindow.to_string(), "max_context must be greater than zero");
    assert_eq!(
        ConfigError::UnsupportedBackend("xyz".into()).to_string(),
        r#"unsupported backend: "xyz""#
    );
}

#[test]
fn config_error_unsupported_backend_preserves_input_verbatim() {
    let weird = "  spaced  ";
    let err = ConfigError::UnsupportedBackend(weird.into());
    assert!(err.to_string().contains(weird));
}

#[test]
fn config_error_implements_std_error() {
    let err = ConfigError::EmptyModelPath;
    let std_err: &dyn std::error::Error = &err;
    assert!(std_err.source().is_none());
}

#[test]
fn config_error_different_variants_are_not_equal() {
    assert_ne!(ConfigError::EmptyModelPath, ConfigError::EmptyTokenizerPath);
    assert_ne!(ConfigError::EmptyModelPath, ConfigError::ZeroContextWindow);
    assert_ne!(
        ConfigError::UnsupportedBackend("a".into()),
        ConfigError::UnsupportedBackend("b".into())
    );
}

#[test]
fn config_error_same_variant_same_data_is_equal() {
    assert_eq!(
        ConfigError::UnsupportedBackend("metal".into()),
        ConfigError::UnsupportedBackend("metal".into())
    );
}

// =========================================================================
// SessionId — uniqueness, format, serde
// =========================================================================

#[test]
fn session_id_format_is_session_dash_number() {
    let id = SessionId::generate();
    let s = id.as_str();
    assert!(s.starts_with("session-"), "unexpected prefix: {s}");
    let num_part = &s["session-".len()..];
    assert!(num_part.parse::<u64>().is_ok(), "suffix should be numeric: {num_part}");
}

#[test]
fn session_id_1000_unique() {
    let ids: HashSet<String> =
        (0..1000).map(|_| SessionId::generate().as_str().to_string()).collect();
    assert_eq!(ids.len(), 1000, "all 1000 session IDs should be unique");
}

#[test]
fn session_id_serde_preserves_exact_string() {
    let id = SessionId::generate();
    let json = serde_json::to_string(&id).unwrap();
    let back: SessionId = serde_json::from_str(&json).unwrap();
    assert_eq!(id.as_str(), back.as_str());
    assert_eq!(id, back);
}

#[test]
fn session_id_hash_consistent_with_eq() {
    use std::hash::{Hash, Hasher};
    let id = SessionId::generate();
    let cloned = id.clone();
    assert_eq!(id, cloned);

    let mut h1 = std::collections::hash_map::DefaultHasher::new();
    let mut h2 = std::collections::hash_map::DefaultHasher::new();
    id.hash(&mut h1);
    cloned.hash(&mut h2);
    assert_eq!(h1.finish(), h2.finish());
}

#[test]
fn session_id_debug_contains_inner_string() {
    let id = SessionId::generate();
    let dbg = format!("{id:?}");
    assert!(dbg.contains(id.as_str()), "Debug should contain the ID string: {dbg}");
}

// =========================================================================
// VALID_BACKENDS — exhaustive coverage
// =========================================================================

#[test]
fn valid_backends_exactly_four_known_entries() {
    assert_eq!(VALID_BACKENDS.len(), 4);
    assert!(VALID_BACKENDS.contains(&"cpu"));
    assert!(VALID_BACKENDS.contains(&"cuda"));
    assert!(VALID_BACKENDS.contains(&"gpu"));
    assert!(VALID_BACKENDS.contains(&"ffi"));
}

#[test]
fn each_valid_backend_passes_validation() {
    for &b in VALID_BACKENDS {
        let cfg = SessionConfig {
            model_path: "m".into(),
            tokenizer_path: "t".into(),
            backend: b.into(),
            max_context: 1,
            seed: None,
        };
        assert!(cfg.validate().is_ok(), "backend {b:?} should be valid");
    }
}

// =========================================================================
// Cross-type integration: SessionConfig -> validate -> ConfigError cycle
// =========================================================================

#[test]
fn validation_returns_first_error_only() {
    // All fields are invalid; model_path is checked first.
    let cfg = SessionConfig {
        model_path: String::new(),
        tokenizer_path: String::new(),
        backend: "nope".into(),
        max_context: 0,
        seed: None,
    };
    assert_eq!(cfg.validate(), Err(ConfigError::EmptyModelPath));
}

#[test]
fn validation_error_priority_tokenizer_before_backend() {
    let cfg = SessionConfig {
        model_path: "m.gguf".into(),
        tokenizer_path: String::new(),
        backend: "nope".into(),
        max_context: 0,
        seed: None,
    };
    assert_eq!(cfg.validate(), Err(ConfigError::EmptyTokenizerPath));
}

#[test]
fn validation_error_priority_backend_before_context() {
    let cfg = SessionConfig {
        model_path: "m.gguf".into(),
        tokenizer_path: "t.json".into(),
        backend: "nope".into(),
        max_context: 0,
        seed: None,
    };
    assert_eq!(cfg.validate(), Err(ConfigError::UnsupportedBackend("nope".into())));
}

// =========================================================================
// EngineStateTracker — Default trait
// =========================================================================

#[test]
fn engine_state_tracker_default_equals_new() {
    let from_new = EngineStateTracker::new();
    let from_default = EngineStateTracker::default();
    assert_eq!(from_new.state(), from_default.state());
    assert_eq!(from_new.state(), &EngineState::Idle);
}
