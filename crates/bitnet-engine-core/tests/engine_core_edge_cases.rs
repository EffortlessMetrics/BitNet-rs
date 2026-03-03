//! Edge-case tests for bitnet-engine-core session management, config validation,
//! state machine transitions, concurrency, and serde roundtrips.

use bitnet_engine_core::*;

// ---------------------------------------------------------------------------
// SessionConfig validation
// ---------------------------------------------------------------------------

fn valid_config() -> SessionConfig {
    SessionConfig {
        model_path: "models/model.gguf".to_string(),
        tokenizer_path: "models/tokenizer.json".to_string(),
        backend: "cpu".to_string(),
        max_context: 4096,
        seed: Some(42),
    }
}

#[test]
fn valid_config_passes_validation() {
    assert!(valid_config().validate().is_ok());
}

#[test]
fn validate_rejects_empty_model_path() {
    let mut cfg = valid_config();
    cfg.model_path = String::new();
    assert_eq!(cfg.validate(), Err(ConfigError::EmptyModelPath));
}

#[test]
fn validate_rejects_empty_tokenizer_path() {
    let mut cfg = valid_config();
    cfg.tokenizer_path = String::new();
    assert_eq!(cfg.validate(), Err(ConfigError::EmptyTokenizerPath));
}

#[test]
fn validate_rejects_unsupported_backend() {
    let mut cfg = valid_config();
    cfg.backend = "metal".to_string();
    assert_eq!(cfg.validate(), Err(ConfigError::UnsupportedBackend("metal".to_string())));
}

#[test]
fn validate_rejects_zero_context() {
    let mut cfg = valid_config();
    cfg.max_context = 0;
    assert_eq!(cfg.validate(), Err(ConfigError::ZeroContextWindow));
}

#[test]
fn validate_empty_backend() {
    let mut cfg = valid_config();
    cfg.backend = String::new();
    assert_eq!(cfg.validate(), Err(ConfigError::UnsupportedBackend(String::new())));
}

#[test]
fn validate_all_valid_backends() {
    for backend in VALID_BACKENDS {
        let mut cfg = valid_config();
        cfg.backend = backend.to_string();
        assert!(cfg.validate().is_ok(), "backend {backend:?} should be valid");
    }
}

#[test]
fn validate_first_error_wins_model_path_over_tokenizer() {
    let cfg = SessionConfig {
        model_path: String::new(),
        tokenizer_path: String::new(),
        backend: "cpu".to_string(),
        max_context: 4096,
        seed: None,
    };
    // model_path is checked first
    assert_eq!(cfg.validate(), Err(ConfigError::EmptyModelPath));
}

#[test]
fn validate_first_error_wins_tokenizer_over_backend() {
    let cfg = SessionConfig {
        model_path: "m.gguf".to_string(),
        tokenizer_path: String::new(),
        backend: "invalid".to_string(),
        max_context: 4096,
        seed: None,
    };
    assert_eq!(cfg.validate(), Err(ConfigError::EmptyTokenizerPath));
}

#[test]
fn validate_first_error_wins_backend_over_context() {
    let cfg = SessionConfig {
        model_path: "m.gguf".to_string(),
        tokenizer_path: "t.json".to_string(),
        backend: "invalid".to_string(),
        max_context: 0,
        seed: None,
    };
    assert_eq!(cfg.validate(), Err(ConfigError::UnsupportedBackend("invalid".to_string())));
}

#[test]
fn validate_max_context_one_is_valid() {
    let mut cfg = valid_config();
    cfg.max_context = 1;
    assert!(cfg.validate().is_ok());
}

// ---------------------------------------------------------------------------
// ConfigError display
// ---------------------------------------------------------------------------

#[test]
fn config_error_display_messages() {
    assert_eq!(ConfigError::EmptyModelPath.to_string(), "model_path must not be empty");
    assert_eq!(ConfigError::EmptyTokenizerPath.to_string(), "tokenizer_path must not be empty");
    assert_eq!(ConfigError::ZeroContextWindow.to_string(), "max_context must be greater than zero");
    let msg = ConfigError::UnsupportedBackend("metal".to_string()).to_string();
    assert!(msg.contains("metal"));
}

// ---------------------------------------------------------------------------
// EngineStateTracker
// ---------------------------------------------------------------------------

#[test]
fn tracker_starts_idle() {
    let tracker = EngineStateTracker::new();
    assert_eq!(*tracker.state(), EngineState::Idle);
}

#[test]
fn tracker_default_is_idle() {
    let tracker = EngineStateTracker::default();
    assert_eq!(*tracker.state(), EngineState::Idle);
}

#[test]
fn tracker_happy_path() {
    let mut tracker = EngineStateTracker::new();
    tracker.start().unwrap();
    assert_eq!(*tracker.state(), EngineState::Running);
    tracker.finish().unwrap();
    assert_eq!(*tracker.state(), EngineState::Done);
}

#[test]
fn tracker_cannot_start_from_running() {
    let mut tracker = EngineStateTracker::new();
    tracker.start().unwrap();
    let err = tracker.start().unwrap_err();
    assert!(err.to_string().contains("Running"));
}

#[test]
fn tracker_cannot_start_from_done() {
    let mut tracker = EngineStateTracker::new();
    tracker.start().unwrap();
    tracker.finish().unwrap();
    let err = tracker.start().unwrap_err();
    assert!(err.to_string().contains("Done"));
}

#[test]
fn tracker_cannot_finish_from_idle() {
    let mut tracker = EngineStateTracker::new();
    let err = tracker.finish().unwrap_err();
    assert!(err.to_string().contains("Idle"));
}

#[test]
fn tracker_cannot_finish_from_done() {
    let mut tracker = EngineStateTracker::new();
    tracker.start().unwrap();
    tracker.finish().unwrap();
    let err = tracker.finish().unwrap_err();
    assert!(err.to_string().contains("Done"));
}

#[test]
fn engine_state_error_is_error_trait() {
    let err = EngineStateError("test".to_string());
    let _: &dyn std::error::Error = &err;
    assert_eq!(err.to_string(), "test");
}

// ---------------------------------------------------------------------------
// SessionId
// ---------------------------------------------------------------------------

#[test]
fn session_id_non_empty() {
    let id = SessionId::generate();
    assert!(!id.as_str().is_empty());
}

#[test]
fn session_id_monotonic() {
    let id1 = SessionId::generate();
    let id2 = SessionId::generate();
    assert_ne!(id1, id2);
    // Both should start with "session-"
    assert!(id1.as_str().starts_with("session-"));
    assert!(id2.as_str().starts_with("session-"));
}

#[test]
fn session_id_many_unique() {
    let ids: Vec<SessionId> = (0..100).map(|_| SessionId::generate()).collect();
    let unique: std::collections::HashSet<_> =
        ids.iter().map(|id| id.as_str().to_string()).collect();
    assert_eq!(unique.len(), 100);
}

#[test]
fn session_id_clone_eq() {
    let id = SessionId::generate();
    let id2 = id.clone();
    assert_eq!(id, id2);
}

#[test]
fn session_id_hash() {
    use std::collections::HashMap;
    let id = SessionId::generate();
    let mut map = HashMap::new();
    map.insert(id.clone(), 42);
    assert_eq!(map[&id], 42);
}

// ---------------------------------------------------------------------------
// ConcurrencyConfig
// ---------------------------------------------------------------------------

#[test]
fn concurrency_default() {
    let cfg = ConcurrencyConfig::default();
    assert_eq!(cfg.max_concurrent, 4);
}

#[test]
fn concurrency_allows_zero_active() {
    let cfg = ConcurrencyConfig { max_concurrent: 4 };
    assert!(cfg.allows(0));
}

#[test]
fn concurrency_allows_below_limit() {
    let cfg = ConcurrencyConfig { max_concurrent: 4 };
    assert!(cfg.allows(3));
}

#[test]
fn concurrency_denies_at_limit() {
    let cfg = ConcurrencyConfig { max_concurrent: 4 };
    assert!(!cfg.allows(4));
}

#[test]
fn concurrency_denies_above_limit() {
    let cfg = ConcurrencyConfig { max_concurrent: 4 };
    assert!(!cfg.allows(5));
}

#[test]
fn concurrency_one_slot() {
    let cfg = ConcurrencyConfig { max_concurrent: 1 };
    assert!(cfg.allows(0));
    assert!(!cfg.allows(1));
}

// ---------------------------------------------------------------------------
// Serde roundtrips
// ---------------------------------------------------------------------------

#[test]
fn session_config_serde_roundtrip() {
    let cfg = valid_config();
    let json = serde_json::to_string(&cfg).unwrap();
    let cfg2: SessionConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg.model_path, cfg2.model_path);
    assert_eq!(cfg.tokenizer_path, cfg2.tokenizer_path);
    assert_eq!(cfg.backend, cfg2.backend);
    assert_eq!(cfg.max_context, cfg2.max_context);
    assert_eq!(cfg.seed, cfg2.seed);
}

#[test]
fn backend_info_serde_roundtrip() {
    let info = BackendInfo {
        backend_name: "cpu-rust".to_string(),
        kernel_ids: vec!["avx2_matmul".to_string(), "simd_norm".to_string()],
        backend_summary: "CPU with AVX2".to_string(),
    };
    let json = serde_json::to_string(&info).unwrap();
    let info2: BackendInfo = serde_json::from_str(&json).unwrap();
    assert_eq!(info.backend_name, info2.backend_name);
    assert_eq!(info.kernel_ids, info2.kernel_ids);
    assert_eq!(info.backend_summary, info2.backend_summary);
}

#[test]
fn session_metrics_serde_roundtrip() {
    let m = SessionMetrics {
        tokens_per_second: 42.5,
        time_to_first_token_ms: 100.0,
        total_tokens: 256,
    };
    let json = serde_json::to_string(&m).unwrap();
    let m2: SessionMetrics = serde_json::from_str(&json).unwrap();
    assert!((m.tokens_per_second - m2.tokens_per_second).abs() < 1e-10);
    assert!((m.time_to_first_token_ms - m2.time_to_first_token_ms).abs() < 1e-10);
    assert_eq!(m.total_tokens, m2.total_tokens);
}

#[test]
fn engine_state_serde_roundtrip() {
    for state in &[EngineState::Idle, EngineState::Running, EngineState::Done] {
        let json = serde_json::to_string(state).unwrap();
        let s2: EngineState = serde_json::from_str(&json).unwrap();
        assert_eq!(*state, s2);
    }
}

#[test]
fn concurrency_config_serde_roundtrip() {
    let cfg = ConcurrencyConfig { max_concurrent: 8 };
    let json = serde_json::to_string(&cfg).unwrap();
    let cfg2: ConcurrencyConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg.max_concurrent, cfg2.max_concurrent);
}

#[test]
fn session_id_serde_roundtrip() {
    let id = SessionId::generate();
    let json = serde_json::to_string(&id).unwrap();
    let id2: SessionId = serde_json::from_str(&json).unwrap();
    assert_eq!(id, id2);
}

// ---------------------------------------------------------------------------
// Multi-SLM context window configs
// ---------------------------------------------------------------------------

#[test]
fn phi4_context_config() {
    let mut cfg = valid_config();
    cfg.max_context = 16384; // Phi-4
    assert!(cfg.validate().is_ok());
}

#[test]
fn llama3_context_config() {
    let mut cfg = valid_config();
    cfg.max_context = 8192; // LLaMA-3
    assert!(cfg.validate().is_ok());
}

#[test]
fn gemma_context_config() {
    let mut cfg = valid_config();
    cfg.max_context = 8192; // Gemma
    assert!(cfg.validate().is_ok());
}

#[test]
fn qwen_context_config() {
    let mut cfg = valid_config();
    cfg.max_context = 32768; // Qwen2.5
    assert!(cfg.validate().is_ok());
}

// ---------------------------------------------------------------------------
// InferenceSession trait object
// ---------------------------------------------------------------------------

struct MockSession {
    response: Vec<StreamEvent>,
}

impl InferenceSession for MockSession {
    fn generate(
        &self,
        _prompt: &str,
        _config: &GenerationConfig,
    ) -> anyhow::Result<Vec<StreamEvent>> {
        Ok(self.response.clone())
    }
}

#[test]
fn mock_session_returns_done() {
    let session = MockSession {
        response: vec![StreamEvent::Done {
            reason: StopReason::MaxTokens,
            stats: GenerationStats::default(),
        }],
    };
    let events = session.generate("hello", &GenerationConfig::default()).unwrap();
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], StreamEvent::Done { .. }));
}

#[test]
fn mock_session_with_tokens() {
    let session = MockSession {
        response: vec![
            StreamEvent::Token(TokenEvent { id: 1, text: "Hello".to_string() }),
            StreamEvent::Token(TokenEvent { id: 2, text: " world".to_string() }),
            StreamEvent::Done {
                reason: StopReason::EosToken,
                stats: GenerationStats { tokens_generated: 2, tokens_per_second: 10.0 },
            },
        ],
    };
    let events = session.generate("prompt", &GenerationConfig::default()).unwrap();
    assert_eq!(events.len(), 3);
    let tokens: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::Token(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(tokens, vec!["Hello", " world"]);
}

#[test]
fn trait_object_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Box<dyn InferenceSession>>();
}
