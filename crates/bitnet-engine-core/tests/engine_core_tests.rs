//! Comprehensive unit tests for `bitnet-engine-core` public API.
//!
//! Coverage matrix
//! ───────────────
//! 1.  `InferenceSession` trait – implement, call, inspect events
//! 2.  `BackendInfo` – construction, field access, kernel list, clone
//! 3.  `SessionMetrics` – non-default values, JSON roundtrip
//! 4.  `EngineState` – all variants, Clone, PartialEq
//! 5.  `EngineStateError` – Display message content
//! 6.  `ConfigError` Display messages embed the invalid value
//! 7.  `ConcurrencyConfig` – boundary (max-1 allows, max disallows), max=1
//! 8.  `SessionId` – clone equals original, as_str prefix, 50 unique IDs
//! 9.  `EngineStateTracker` – full cycle, all illegal transitions
//! 10. `VALID_BACKENDS` – contains all four documented identifiers
//! 11. `SessionConfig::validate` – backend is third priority field
//! 12. `SessionConfig::validate` – max_context=1 is minimum valid
//! 13. JSON roundtrip for `SessionMetrics`
//! 14. `BackendInfo` kernel_ids order preserved after clone

use anyhow::Result;
use bitnet_engine_core::{
    BackendInfo, ConcurrencyConfig, ConfigError, EngineState, EngineStateTracker, GenerationConfig,
    GenerationStats, InferenceSession, SessionConfig, SessionId, SessionMetrics, StopReason,
    StreamEvent, TokenEvent, VALID_BACKENDS,
};
use std::collections::HashSet;

// ── 1. InferenceSession trait ─────────────────────────────────────────────

/// Minimal implementation that echoes the prompt back as a single Token event.
struct EchoSession;

impl InferenceSession for EchoSession {
    fn generate(&self, prompt: &str, _config: &GenerationConfig) -> Result<Vec<StreamEvent>> {
        Ok(vec![
            StreamEvent::Token(TokenEvent { id: 0, text: prompt.to_string() }),
            StreamEvent::Done { reason: StopReason::MaxTokens, stats: GenerationStats::default() },
        ])
    }
}

#[test]
fn inference_session_returns_token_then_done() {
    let session = EchoSession;
    let events = session.generate("hello", &GenerationConfig::default()).unwrap();
    assert_eq!(events.len(), 2, "expected exactly Token + Done");
    assert!(matches!(&events[0], StreamEvent::Token(t) if t.text == "hello"));
    assert!(matches!(&events[1], StreamEvent::Done { .. }));
}

#[test]
fn inference_session_last_event_is_done() {
    let session = EchoSession;
    let events = session.generate("test", &GenerationConfig::default()).unwrap();
    assert!(matches!(events.last(), Some(StreamEvent::Done { .. })), "final event must be Done");
}

#[test]
fn inference_session_empty_prompt() {
    let session = EchoSession;
    let events = session.generate("", &GenerationConfig::default()).unwrap();
    assert!(matches!(&events[0], StreamEvent::Token(t) if t.text.is_empty()));
}

// ── 2. BackendInfo ────────────────────────────────────────────────────────

#[test]
fn backend_info_fields_accessible() {
    let info = BackendInfo {
        backend_name: "cpu-rust".to_string(),
        kernel_ids: vec!["i2s_cpu_matmul".to_string(), "layernorm_cpu".to_string()],
        backend_summary: "CPU-Rust (AVX2)".to_string(),
    };
    assert_eq!(info.backend_name, "cpu-rust");
    assert_eq!(info.kernel_ids.len(), 2);
    assert_eq!(info.kernel_ids[0], "i2s_cpu_matmul");
    assert_eq!(info.backend_summary, "CPU-Rust (AVX2)");
}

#[test]
fn backend_info_clone_kernel_ids_order_preserved() {
    let info = BackendInfo {
        backend_name: "test".to_string(),
        kernel_ids: vec!["k3".to_string(), "k1".to_string(), "k2".to_string()],
        backend_summary: String::new(),
    };
    let cloned = info.clone();
    assert_eq!(cloned.kernel_ids, vec!["k3", "k1", "k2"]);
}

#[test]
fn backend_info_empty_kernel_ids_is_valid() {
    let info = BackendInfo {
        backend_name: "cpu".to_string(),
        kernel_ids: vec![],
        backend_summary: "none".to_string(),
    };
    assert!(info.kernel_ids.is_empty());
}

#[test]
fn backend_info_json_roundtrip() {
    let info = BackendInfo {
        backend_name: "cuda".to_string(),
        kernel_ids: vec!["gemm_fp16".to_string()],
        backend_summary: "CUDA backend".to_string(),
    };
    let json = serde_json::to_string(&info).unwrap();
    let back: BackendInfo = serde_json::from_str(&json).unwrap();
    assert_eq!(back.backend_name, info.backend_name);
    assert_eq!(back.kernel_ids, info.kernel_ids);
    assert_eq!(back.backend_summary, info.backend_summary);
}

// ── 3. SessionMetrics ─────────────────────────────────────────────────────

#[test]
fn session_metrics_non_default_values() {
    let m =
        SessionMetrics { tokens_per_second: 42.5, time_to_first_token_ms: 100.0, total_tokens: 64 };
    assert!((m.tokens_per_second - 42.5).abs() < f64::EPSILON);
    assert!((m.time_to_first_token_ms - 100.0).abs() < f64::EPSILON);
    assert_eq!(m.total_tokens, 64);
}

#[test]
fn session_metrics_json_roundtrip() {
    let m =
        SessionMetrics { tokens_per_second: 12.3, time_to_first_token_ms: 50.0, total_tokens: 128 };
    let json = serde_json::to_string(&m).unwrap();
    let back: SessionMetrics = serde_json::from_str(&json).unwrap();
    assert!((back.tokens_per_second - m.tokens_per_second).abs() < 1e-9);
    assert!((back.time_to_first_token_ms - m.time_to_first_token_ms).abs() < 1e-9);
    assert_eq!(back.total_tokens, m.total_tokens);
}

// ── 4. EngineState ────────────────────────────────────────────────────────

#[test]
fn engine_state_variants_equality() {
    assert_eq!(EngineState::Idle, EngineState::Idle);
    assert_eq!(EngineState::Running, EngineState::Running);
    assert_eq!(EngineState::Done, EngineState::Done);
    assert_ne!(EngineState::Idle, EngineState::Running);
    assert_ne!(EngineState::Running, EngineState::Done);
    assert_ne!(EngineState::Idle, EngineState::Done);
}

#[test]
fn engine_state_clone_equals_original() {
    for state in [EngineState::Idle, EngineState::Running, EngineState::Done] {
        assert_eq!(state.clone(), state);
    }
}

// ── 5. EngineStateError ───────────────────────────────────────────────────

#[test]
fn engine_state_error_double_start_message_non_empty() {
    let mut tracker = EngineStateTracker::new();
    tracker.start().unwrap();
    let err = tracker.start().unwrap_err();
    assert!(!err.to_string().is_empty());
}

#[test]
fn engine_state_error_finish_from_idle_message_non_empty() {
    let mut tracker = EngineStateTracker::new();
    let err = tracker.finish().unwrap_err();
    let msg = err.to_string();
    assert!(!msg.is_empty());
    // The message should mention the current state.
    assert!(msg.contains("Idle"), "expected 'Idle' in message: {msg}");
}

#[test]
fn engine_state_error_finish_from_done_fails() {
    let mut tracker = EngineStateTracker::new();
    tracker.start().unwrap();
    tracker.finish().unwrap();
    let err = tracker.finish().unwrap_err();
    let msg = err.to_string();
    assert!(!msg.is_empty());
    assert!(msg.contains("Done"), "expected 'Done' in message: {msg}");
}

#[test]
fn engine_state_error_start_from_done_fails() {
    let mut tracker = EngineStateTracker::new();
    tracker.start().unwrap();
    tracker.finish().unwrap();
    let err = tracker.start().unwrap_err();
    assert!(!err.to_string().is_empty());
}

// ── 6. ConfigError Display messages ──────────────────────────────────────

#[test]
fn config_error_unsupported_backend_display_contains_backend() {
    let err = ConfigError::UnsupportedBackend("tpu".to_string());
    let msg = err.to_string();
    assert!(msg.contains("tpu"), "expected backend name in message: {msg}");
}

#[test]
fn config_error_empty_model_path_display_non_empty() {
    assert!(!ConfigError::EmptyModelPath.to_string().is_empty());
}

#[test]
fn config_error_empty_tokenizer_path_display_non_empty() {
    assert!(!ConfigError::EmptyTokenizerPath.to_string().is_empty());
}

#[test]
fn config_error_zero_context_window_display_non_empty() {
    assert!(!ConfigError::ZeroContextWindow.to_string().is_empty());
}

// ── 7. ConcurrencyConfig boundary behaviour ───────────────────────────────

#[test]
fn concurrency_config_max_one_allows_only_zero() {
    let cfg = ConcurrencyConfig { max_concurrent: 1 };
    assert!(cfg.allows(0), "0 active should be allowed when max=1");
    assert!(!cfg.allows(1), "1 active should NOT be allowed when max=1");
    assert!(!cfg.allows(2), "2 active should NOT be allowed when max=1");
}

#[test]
fn concurrency_config_boundary_exact() {
    let cfg = ConcurrencyConfig { max_concurrent: 8 };
    assert!(cfg.allows(7));
    assert!(!cfg.allows(8));
    assert!(!cfg.allows(9));
}

#[test]
fn concurrency_config_default_max_concurrent_is_four() {
    assert_eq!(ConcurrencyConfig::default().max_concurrent, 4);
}

// ── 8. SessionId ──────────────────────────────────────────────────────────

#[test]
fn session_id_as_str_starts_with_session_prefix() {
    let id = SessionId::generate();
    assert!(id.as_str().starts_with("session-"), "id={}", id.as_str());
}

#[test]
fn session_id_clone_equals_original() {
    let id = SessionId::generate();
    assert_eq!(id.clone(), id);
}

#[test]
fn session_id_fifty_consecutive_all_distinct() {
    let ids: HashSet<String> =
        (0..50).map(|_| SessionId::generate().as_str().to_string()).collect();
    assert_eq!(ids.len(), 50, "expected 50 distinct session IDs");
}

#[test]
fn session_id_as_str_is_stable() {
    let id = SessionId::generate();
    assert_eq!(id.as_str(), id.as_str());
}

// ── 9. EngineStateTracker full cycle and all illegal transitions ──────────

#[test]
fn engine_state_full_cycle_succeeds() {
    let mut tracker = EngineStateTracker::new();
    assert_eq!(tracker.state(), &EngineState::Idle);
    tracker.start().expect("Idle → Running");
    assert_eq!(tracker.state(), &EngineState::Running);
    tracker.finish().expect("Running → Done");
    assert_eq!(tracker.state(), &EngineState::Done);
}

#[test]
fn engine_state_start_from_running_fails() {
    let mut tracker = EngineStateTracker::new();
    tracker.start().unwrap();
    assert!(tracker.start().is_err(), "start from Running must fail");
}

#[test]
fn engine_state_finish_from_idle_fails() {
    let mut tracker = EngineStateTracker::new();
    assert!(tracker.finish().is_err(), "finish from Idle must fail");
}

#[test]
fn engine_state_tracker_default_equals_new() {
    let a = EngineStateTracker::new();
    let b = EngineStateTracker::default();
    assert_eq!(a.state(), b.state());
}

// ── 10. VALID_BACKENDS ───────────────────────────────────────────────────

#[test]
fn valid_backends_contains_all_documented_identifiers() {
    for expected in ["cpu", "cuda", "gpu", "ffi"] {
        assert!(VALID_BACKENDS.contains(&expected), "{expected:?} must be in VALID_BACKENDS");
    }
}

#[test]
fn valid_backends_count_is_four() {
    assert_eq!(VALID_BACKENDS.len(), 4);
}

// ── 11. SessionConfig::validate priority: backend is third ───────────────

#[test]
fn validate_backend_checked_after_paths() {
    // model_path and tokenizer_path are valid; bad backend should now fail.
    let cfg = SessionConfig {
        model_path: "m.gguf".to_string(),
        tokenizer_path: "t.json".to_string(),
        backend: "unknown-backend".to_string(),
        max_context: 512,
        seed: None,
    };
    assert_eq!(cfg.validate(), Err(ConfigError::UnsupportedBackend("unknown-backend".to_string())));
}

// ── 12. SessionConfig::validate – max_context=1 is minimum valid ──────────

#[test]
fn session_config_min_context_one_is_valid() {
    let cfg = SessionConfig {
        model_path: "m.gguf".to_string(),
        tokenizer_path: "t.json".to_string(),
        backend: "cpu".to_string(),
        max_context: 1,
        seed: None,
    };
    assert!(cfg.validate().is_ok());
}

// ── 13. SessionConfig JSON roundtrip with seed ────────────────────────────

#[test]
fn session_config_json_roundtrip_with_seed() {
    let cfg = SessionConfig {
        model_path: "models/model.gguf".to_string(),
        tokenizer_path: "models/tokenizer.json".to_string(),
        backend: "gpu".to_string(),
        max_context: 4096,
        seed: Some(12345),
    };
    let json = serde_json::to_string(&cfg).unwrap();
    let back: SessionConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.model_path, cfg.model_path);
    assert_eq!(back.tokenizer_path, cfg.tokenizer_path);
    assert_eq!(back.backend, cfg.backend);
    assert_eq!(back.max_context, cfg.max_context);
    assert_eq!(back.seed, cfg.seed);
}

#[test]
fn session_config_json_roundtrip_no_seed() {
    let cfg = SessionConfig {
        model_path: "m.gguf".to_string(),
        tokenizer_path: "t.json".to_string(),
        backend: "ffi".to_string(),
        max_context: 2048,
        seed: None,
    };
    let json = serde_json::to_string(&cfg).unwrap();
    let back: SessionConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.seed, None);
    assert_eq!(back.backend, "ffi");
}
