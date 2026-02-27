//! Snapshot tests for `bitnet-engine-core` public API surface.
//!
//! Pins the JSON serialization format of key orchestration types so that
//! changes to the wire format are visible in code review.

use bitnet_engine_core::{BackendInfo, SessionConfig, SessionMetrics};

#[test]
fn session_config_default_json_snapshot() {
    let config = SessionConfig::default();
    let json = serde_json::to_string_pretty(&config).unwrap();
    insta::assert_snapshot!("session_config_default_json", json);
}

#[test]
fn session_config_with_values_json_snapshot() {
    let config = SessionConfig {
        model_path: "/models/model.gguf".to_string(),
        tokenizer_path: "/models/tokenizer.json".to_string(),
        backend: "cpu".to_string(),
        max_context: 4096,
        seed: Some(42),
    };
    let json = serde_json::to_string_pretty(&config).unwrap();
    insta::assert_snapshot!("session_config_with_values_json", json);
}

#[test]
fn backend_info_default_json_snapshot() {
    let info = BackendInfo::default();
    let json = serde_json::to_string_pretty(&info).unwrap();
    insta::assert_snapshot!("backend_info_default_json", json);
}

#[test]
fn backend_info_cpu_rust_json_snapshot() {
    let info = BackendInfo {
        backend_name: "cpu-rust".to_string(),
        kernel_ids: vec!["i2s_cpu_matmul".to_string(), "layernorm_cpu".to_string()],
        backend_summary: "CPU-Rust (AVX2) — 2 kernels".to_string(),
    };
    let json = serde_json::to_string_pretty(&info).unwrap();
    insta::assert_snapshot!("backend_info_cpu_rust_json", json);
}

#[test]
fn session_metrics_default_json_snapshot() {
    let metrics = SessionMetrics::default();
    let json = serde_json::to_string_pretty(&metrics).unwrap();
    insta::assert_snapshot!("session_metrics_default_json", json);
}

#[test]
fn generation_stats_default_json_snapshot() {
    use bitnet_engine_core::GenerationStats;
    let stats = GenerationStats::default();
    let json = serde_json::to_string_pretty(&stats).unwrap();
    insta::assert_snapshot!("generation_stats_default_json", json);
}

#[test]
fn generation_stats_with_values_json_snapshot() {
    use bitnet_engine_core::GenerationStats;
    let stats = GenerationStats { tokens_generated: 64, tokens_per_second: 12.3 };
    let json = serde_json::to_string_pretty(&stats).unwrap();
    insta::assert_snapshot!("generation_stats_with_values_json", json);
}
