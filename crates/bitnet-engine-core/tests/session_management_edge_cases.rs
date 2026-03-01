//! Edge case and boundary tests for engine-core session management.
//!
//! Tests exercise validation, state machine transitions, concurrency limits,
//! and session ID generation.

use bitnet_engine_core::{
    BackendInfo, ConcurrencyConfig, ConfigError, EngineState, EngineStateTracker, SessionConfig,
    SessionId, SessionMetrics, VALID_BACKENDS,
};

// --- SessionConfig validation ---

#[test]
fn valid_config_passes_validation() {
    let config = SessionConfig {
        model_path: "model.gguf".to_string(),
        tokenizer_path: "tokenizer.json".to_string(),
        backend: "cpu".to_string(),
        max_context: 4096,
        seed: Some(42),
    };
    assert!(config.validate().is_ok());
}

#[test]
fn empty_model_path_fails() {
    let config = SessionConfig {
        model_path: String::new(),
        tokenizer_path: "tokenizer.json".to_string(),
        backend: "cpu".to_string(),
        max_context: 4096,
        seed: None,
    };
    assert_eq!(config.validate().unwrap_err(), ConfigError::EmptyModelPath);
}

#[test]
fn empty_tokenizer_path_fails() {
    let config = SessionConfig {
        model_path: "model.gguf".to_string(),
        tokenizer_path: String::new(),
        backend: "cpu".to_string(),
        max_context: 4096,
        seed: None,
    };
    assert_eq!(config.validate().unwrap_err(), ConfigError::EmptyTokenizerPath);
}

#[test]
fn unsupported_backend_fails() {
    let config = SessionConfig {
        model_path: "model.gguf".to_string(),
        tokenizer_path: "tokenizer.json".to_string(),
        backend: "tpu".to_string(),
        max_context: 4096,
        seed: None,
    };
    assert!(matches!(
        config.validate().unwrap_err(),
        ConfigError::UnsupportedBackend(b) if b == "tpu"
    ));
}

#[test]
fn zero_context_window_fails() {
    let config = SessionConfig {
        model_path: "model.gguf".to_string(),
        tokenizer_path: "tokenizer.json".to_string(),
        backend: "cpu".to_string(),
        max_context: 0,
        seed: None,
    };
    assert_eq!(config.validate().unwrap_err(), ConfigError::ZeroContextWindow);
}

#[test]
fn all_valid_backends_accepted() {
    for backend in VALID_BACKENDS {
        let config = SessionConfig {
            model_path: "m".to_string(),
            tokenizer_path: "t".to_string(),
            backend: backend.to_string(),
            max_context: 1,
            seed: None,
        };
        assert!(config.validate().is_ok(), "Backend '{backend}' should be valid");
    }
}

#[test]
fn validation_priority_model_path_first() {
    // Both model_path and tokenizer_path are empty
    let config = SessionConfig {
        model_path: String::new(),
        tokenizer_path: String::new(),
        backend: "invalid".to_string(),
        max_context: 0,
        seed: None,
    };
    // Should report model_path error first
    assert_eq!(config.validate().unwrap_err(), ConfigError::EmptyModelPath);
}

// --- EngineStateTracker transitions ---

#[test]
fn tracker_starts_idle() {
    let tracker = EngineStateTracker::new();
    assert_eq!(*tracker.state(), EngineState::Idle);
}

#[test]
fn idle_to_running_succeeds() {
    let mut tracker = EngineStateTracker::new();
    assert!(tracker.start().is_ok());
    assert_eq!(*tracker.state(), EngineState::Running);
}

#[test]
fn running_to_done_succeeds() {
    let mut tracker = EngineStateTracker::new();
    tracker.start().unwrap();
    assert!(tracker.finish().is_ok());
    assert_eq!(*tracker.state(), EngineState::Done);
}

#[test]
fn double_start_fails() {
    let mut tracker = EngineStateTracker::new();
    tracker.start().unwrap();
    assert!(tracker.start().is_err());
}

#[test]
fn finish_without_start_fails() {
    let mut tracker = EngineStateTracker::new();
    assert!(tracker.finish().is_err());
}

#[test]
fn double_finish_fails() {
    let mut tracker = EngineStateTracker::new();
    tracker.start().unwrap();
    tracker.finish().unwrap();
    assert!(tracker.finish().is_err());
}

// --- SessionId ---

#[test]
fn session_id_generate_is_unique() {
    let id1 = SessionId::generate();
    let id2 = SessionId::generate();
    assert_ne!(id1, id2);
}

#[test]
fn session_id_starts_with_session_prefix() {
    let id = SessionId::generate();
    assert!(id.as_str().starts_with("session-"));
}

#[test]
fn session_id_clone_equals_original() {
    let id = SessionId::generate();
    let cloned = id.clone();
    assert_eq!(id, cloned);
}

#[test]
fn session_id_debug_non_empty() {
    let id = SessionId::generate();
    assert!(!format!("{id:?}").is_empty());
}

// --- ConcurrencyConfig ---

#[test]
fn concurrency_allows_below_limit() {
    let config = ConcurrencyConfig { max_concurrent: 4 };
    assert!(config.allows(0));
    assert!(config.allows(1));
    assert!(config.allows(3));
}

#[test]
fn concurrency_denies_at_limit() {
    let config = ConcurrencyConfig { max_concurrent: 4 };
    assert!(!config.allows(4));
}

#[test]
fn concurrency_denies_above_limit() {
    let config = ConcurrencyConfig { max_concurrent: 4 };
    assert!(!config.allows(5));
    assert!(!config.allows(100));
}

#[test]
fn concurrency_zero_max_denies_all() {
    let config = ConcurrencyConfig { max_concurrent: 0 };
    assert!(!config.allows(0));
    assert!(!config.allows(1));
}

#[test]
fn concurrency_one_allows_only_zero_active() {
    let config = ConcurrencyConfig { max_concurrent: 1 };
    assert!(config.allows(0));
    assert!(!config.allows(1));
}

// --- BackendInfo ---

#[test]
fn backend_info_default_is_empty() {
    let info = BackendInfo::default();
    assert!(info.backend_name.is_empty());
    assert!(info.kernel_ids.is_empty());
    assert!(info.backend_summary.is_empty());
}

#[test]
fn backend_info_construction() {
    let info = BackendInfo {
        backend_name: "cuda".to_string(),
        kernel_ids: vec!["matmul_f16".to_string(), "softmax".to_string()],
        backend_summary: "CUDA 12.0".to_string(),
    };
    assert_eq!(info.backend_name, "cuda");
    assert_eq!(info.kernel_ids.len(), 2);
}

#[test]
fn backend_info_clone() {
    let info = BackendInfo {
        backend_name: "cpu".to_string(),
        kernel_ids: vec!["avx2".to_string()],
        backend_summary: "AVX2 optimized".to_string(),
    };
    let cloned = info.clone();
    assert_eq!(info.backend_name, cloned.backend_name);
}

// --- SessionMetrics ---

#[test]
fn session_metrics_default_is_zero() {
    let metrics = SessionMetrics::default();
    assert_eq!(metrics.tokens_per_second, 0.0);
    assert_eq!(metrics.time_to_first_token_ms, 0.0);
    assert_eq!(metrics.total_tokens, 0);
}

#[test]
fn session_metrics_construction() {
    let metrics = SessionMetrics {
        tokens_per_second: 45.5,
        time_to_first_token_ms: 120.0,
        total_tokens: 256,
    };
    assert!(metrics.tokens_per_second > 0.0);
    assert_eq!(metrics.total_tokens, 256);
}

// --- EngineState enum ---

#[test]
fn engine_state_equality() {
    assert_eq!(EngineState::Idle, EngineState::Idle);
    assert_eq!(EngineState::Running, EngineState::Running);
    assert_eq!(EngineState::Done, EngineState::Done);
    assert_ne!(EngineState::Idle, EngineState::Running);
    assert_ne!(EngineState::Running, EngineState::Done);
}

#[test]
fn engine_state_debug_non_empty() {
    for state in [EngineState::Idle, EngineState::Running, EngineState::Done] {
        assert!(!format!("{state:?}").is_empty());
    }
}

// --- ConfigError ---

#[test]
fn config_error_variants_are_distinct() {
    let errors = [
        ConfigError::EmptyModelPath,
        ConfigError::EmptyTokenizerPath,
        ConfigError::UnsupportedBackend("x".to_string()),
        ConfigError::ZeroContextWindow,
    ];
    for (i, a) in errors.iter().enumerate() {
        for (j, b) in errors.iter().enumerate() {
            if i == j {
                assert_eq!(a, b);
            } else {
                assert_ne!(a, b);
            }
        }
    }
}

#[test]
fn config_error_debug_non_empty() {
    let errors = [
        ConfigError::EmptyModelPath,
        ConfigError::EmptyTokenizerPath,
        ConfigError::UnsupportedBackend("test".to_string()),
        ConfigError::ZeroContextWindow,
    ];
    for error in &errors {
        assert!(!format!("{error:?}").is_empty());
    }
}

// --- VALID_BACKENDS ---

#[test]
fn valid_backends_contains_cpu() {
    assert!(VALID_BACKENDS.contains(&"cpu"));
}

#[test]
fn valid_backends_contains_cuda() {
    assert!(VALID_BACKENDS.contains(&"cuda"));
}

#[test]
fn valid_backends_non_empty() {
    assert!(!VALID_BACKENDS.is_empty());
}
