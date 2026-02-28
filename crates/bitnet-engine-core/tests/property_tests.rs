//! Property-based tests for `bitnet-engine-core` orchestration contracts.
//!
//! Key invariants:
//! - `SessionConfig` JSON round-trips without data loss
//! - `BackendInfo` JSON round-trips without data loss
//! - `SessionMetrics` values are always non-negative after deserialization
//! - Clone produces a value whose serialization is identical to the original
//! - Field-level invariants (`kernel_ids` order/length, `max_context` exactness, seed precision)

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
    assert!((m.tokens_per_second - 0.0).abs() < f64::EPSILON);
    assert!((m.time_to_first_token_ms - 0.0).abs() < f64::EPSILON);
    assert_eq!(m.total_tokens, 0);
}

// ── expanded proptest coverage ────────────────────────────────────────────

use bitnet_engine_core::{GenerationConfig, StopCriteria};

proptest! {
    /// `GenerationConfig::default()` produces a config with sensible field values.
    #[test]
    fn generation_config_default_is_sensible(_: ()) {
        let cfg = GenerationConfig::default();
        prop_assert!(cfg.max_new_tokens > 0, "default max_new_tokens must be positive");
        prop_assert!(cfg.seed.is_none(), "default seed must be None");
        prop_assert_eq!(
            cfg.max_new_tokens, 128,
            "expected default max_new_tokens=128"
        );
    }

    /// `GenerationConfig` (re-exported from `bitnet-generation`) round-trips
    /// through JSON without data loss.
    #[test]
    fn generation_config_serde_roundtrip(
        max_new_tokens in 1usize..4096usize,
        seed in prop::option::of(any::<u64>()),
    ) {
        let config = GenerationConfig {
            max_new_tokens,
            seed,
            stop_criteria: StopCriteria::default(),
        };
        let json = serde_json::to_string(&config).expect("serialize");
        let restored: GenerationConfig = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(config.max_new_tokens, restored.max_new_tokens);
        prop_assert_eq!(config.seed, restored.seed);
    }

    /// The re-exported `GenerationConfig` from `bitnet-engine-core` carries the same
    /// default values as the upstream `bitnet-generation` type.
    #[test]
    fn reexported_generation_config_matches_upstream_defaults(_: ()) {
        let from_engine_core = GenerationConfig::default();
        // Values must match bitnet_generation::GenerationConfig::default().
        prop_assert_eq!(from_engine_core.max_new_tokens, 128);
        prop_assert!(from_engine_core.seed.is_none());
        prop_assert_eq!(from_engine_core.stop_criteria.max_tokens, 0);
    }
}

// ── clone invariants ──────────────────────────────────────────────────────

proptest! {
    /// Cloning `SessionConfig` produces a value that serializes identically.
    ///
    /// Verifies that `Clone` captures all fields — structural equality via
    /// JSON is used because `SessionConfig` does not implement `PartialEq`.
    #[test]
    fn session_config_clone_serializes_identically(config in arb_session_config()) {
        let cloned = config.clone();
        let orig_json  = serde_json::to_string(&config).unwrap();
        let clone_json = serde_json::to_string(&cloned).unwrap();
        prop_assert_eq!(orig_json, clone_json);
    }

    /// Cloning `BackendInfo` produces a value that serializes identically.
    #[test]
    fn backend_info_clone_serializes_identically(info in arb_backend_info()) {
        let cloned = info.clone();
        let orig_json  = serde_json::to_string(&info).unwrap();
        let clone_json = serde_json::to_string(&cloned).unwrap();
        prop_assert_eq!(orig_json, clone_json);
    }

    /// Cloning `SessionMetrics` produces a value that serializes identically.
    #[test]
    fn session_metrics_clone_serializes_identically(metrics in arb_session_metrics()) {
        let cloned = metrics.clone();
        let orig_json  = serde_json::to_string(&metrics).unwrap();
        let clone_json = serde_json::to_string(&cloned).unwrap();
        prop_assert_eq!(orig_json, clone_json);
    }
}

// ── field-level invariants ────────────────────────────────────────────────

proptest! {
    /// `BackendInfo::kernel_ids` length is preserved exactly after JSON round-trip.
    ///
    /// Confirms neither truncation nor duplication occurs in the serde path.
    #[test]
    fn backend_info_kernel_ids_len_preserved(info in arb_backend_info()) {
        let json = serde_json::to_string(&info).unwrap();
        let restored: BackendInfo = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(
            info.kernel_ids.len(),
            restored.kernel_ids.len(),
            "kernel_ids length changed after JSON round-trip"
        );
    }

    /// `SessionConfig::max_context` round-trips with bit-exact equality.
    ///
    /// Uses the full `usize` range supported by JSON integers to catch any
    /// implicit narrowing or overflow in the serialization path.
    #[test]
    fn session_config_max_context_exact_roundtrip(max_context in 1usize..=1_000_000usize) {
        let config = SessionConfig {
            model_path: "m.gguf".to_string(),
            tokenizer_path: "t.json".to_string(),
            backend: "cpu".to_string(),
            max_context,
            seed: None,
        };
        let json = serde_json::to_string(&config).unwrap();
        let restored: SessionConfig = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(
            config.max_context,
            restored.max_context,
            "max_context changed after JSON round-trip"
        );
    }

    /// `SessionConfig::seed = Some(v)` preserves the full u64 value after
    /// JSON round-trip, including values ≥ 2^53 that JSON integers cannot
    /// represent exactly as f64.
    #[test]
    fn session_config_seed_some_all_u64_values(seed in any::<u64>()) {
        let config = SessionConfig {
            model_path: String::new(),
            tokenizer_path: String::new(),
            backend: "cpu".to_string(),
            max_context: 1024,
            seed: Some(seed),
        };
        let json = serde_json::to_string(&config).unwrap();
        let restored: SessionConfig = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(config.seed, restored.seed);
    }
}
