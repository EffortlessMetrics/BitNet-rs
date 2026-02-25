//! Property-based tests for `bitnet-engine-core` orchestration contracts.
//!
//! Key invariants:
//! - `SessionConfig` JSON round-trips without data loss
//! - `BackendInfo` JSON round-trips without data loss
//! - `SessionMetrics` values are always non-negative after deserialization

use bitnet_engine_core::{BackendInfo, SessionConfig, SessionMetrics};
use proptest::prelude::*;

// ── strategies ────────────────────────────────────────────────────────────

fn arb_session_config() -> impl Strategy<Value = SessionConfig> {
    (
        "[a-z0-9_/\\.]{1,64}", // model_path
        "[a-z0-9_/\\.]{1,64}", // tokenizer_path
        prop_oneof![Just("cpu"), Just("cuda"), Just("ffi")],
        1usize..16384usize, // max_context
        prop::option::of(any::<u64>()),
    )
        .prop_map(|(model_path, tokenizer_path, backend, max_context, seed)| SessionConfig {
            model_path,
            tokenizer_path,
            backend: backend.to_string(),
            max_context,
            seed,
        })
}

fn arb_backend_info() -> impl Strategy<Value = BackendInfo> {
    (
        "[a-z\\-]{1,32}",
        prop::collection::vec("[a-z0-9_]{1,32}", 0..8),
        "[a-zA-Z0-9 \\-\\(\\)]{0,128}",
    )
        .prop_map(|(backend_name, kernel_ids, backend_summary)| BackendInfo {
            backend_name,
            kernel_ids,
            backend_summary,
        })
}

fn arb_session_metrics() -> impl Strategy<Value = SessionMetrics> {
    (0.0f64..10000.0f64, 0.0f64..10000.0f64, 0usize..100_000usize).prop_map(
        |(tokens_per_second, time_to_first_token_ms, total_tokens)| SessionMetrics {
            tokens_per_second,
            time_to_first_token_ms,
            total_tokens,
        },
    )
}

// ── round-trip properties ─────────────────────────────────────────────────

proptest! {
    /// SessionConfig serializes to JSON and deserializes back to the same value.
    #[test]
    fn session_config_json_round_trip(config in arb_session_config()) {
        let json = serde_json::to_string(&config).unwrap();
        let restored: SessionConfig = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(config.model_path, restored.model_path);
        prop_assert_eq!(config.tokenizer_path, restored.tokenizer_path);
        prop_assert_eq!(config.backend, restored.backend);
        prop_assert_eq!(config.max_context, restored.max_context);
        prop_assert_eq!(config.seed, restored.seed);
    }

    /// BackendInfo serializes to JSON and deserializes back without data loss.
    #[test]
    fn backend_info_json_round_trip(info in arb_backend_info()) {
        let json = serde_json::to_string(&info).unwrap();
        let restored: BackendInfo = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(info.backend_name, restored.backend_name);
        prop_assert_eq!(info.kernel_ids, restored.kernel_ids);
        prop_assert_eq!(info.backend_summary, restored.backend_summary);
    }

    /// SessionMetrics serializes to JSON and deserializes without loss.
    #[test]
    fn session_metrics_json_round_trip(metrics in arb_session_metrics()) {
        let json = serde_json::to_string(&metrics).unwrap();
        let restored: SessionMetrics = serde_json::from_str(&json).unwrap();
        prop_assert!((metrics.tokens_per_second - restored.tokens_per_second).abs() < 1e-6);
        prop_assert!((metrics.time_to_first_token_ms - restored.time_to_first_token_ms).abs() < 1e-6);
        prop_assert_eq!(metrics.total_tokens, restored.total_tokens);
    }

    /// SessionMetrics non-negativity invariant: constructed values are non-negative.
    #[test]
    fn session_metrics_values_non_negative(metrics in arb_session_metrics()) {
        prop_assert!(metrics.tokens_per_second >= 0.0);
        prop_assert!(metrics.time_to_first_token_ms >= 0.0);
        // total_tokens is usize so always >= 0
    }

    /// Default SessionConfig has non-empty backend identifier.
    #[test]
    fn default_session_config_has_backend(_: ()) {
        let config = SessionConfig::default();
        prop_assert!(!config.backend.is_empty(), "Default backend must not be empty");
    }
}

// ── unit tests ────────────────────────────────────────────────────────────

#[test]
fn session_config_default_backend_is_cpu() {
    let config = SessionConfig::default();
    assert_eq!(config.backend, "cpu");
}

#[test]
fn session_metrics_default_is_zero() {
    let m = SessionMetrics::default();
    assert_eq!(m.tokens_per_second, 0.0);
    assert_eq!(m.time_to_first_token_ms, 0.0);
    assert_eq!(m.total_tokens, 0);
}
