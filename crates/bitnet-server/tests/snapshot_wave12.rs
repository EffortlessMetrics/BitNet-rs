#![allow(clippy::all, clippy::pedantic, clippy::nursery, unused_imports)]
//! Snapshot wave 12 — bitnet-server
//!
//! Covers: ServerConfig JSON, ServerSettings debug, DeviceConfig variants/parsing,
//! BatchEngineConfig, ConcurrencyConfig, SecurityConfig, ModelManagerConfig,
//! ExecutionRouterConfig, DeviceSelectionStrategy, DeviceHealth, ModelState,
//! ModelEntry JSON, ErrorResponse JSON, InferenceResponse JSON, DeviceCapabilities.

use std::time::Duration;

use bitnet_server::batch_engine::BatchEngineConfig;
use bitnet_server::concurrency::ConcurrencyConfig;
use bitnet_server::config::{DeviceConfig, ServerConfig};
use bitnet_server::execution_router::{
    DeviceCapabilities, DeviceHealth, DeviceSelectionStrategy, ExecutionRouterConfig,
};
use bitnet_server::model_manager::ModelManagerConfig;
use bitnet_server::model_registry::{ModelEntry, ModelState};
use bitnet_server::security::SecurityConfig;
use bitnet_server::{ErrorResponse, InferenceResponse};

// ── ServerConfig ────────────────────────────────────────────────────────────

#[test]
fn server_config_default_json() {
    let cfg = ServerConfig::default();
    insta::assert_json_snapshot!(cfg);
}

#[test]
fn server_config_default_debug() {
    let cfg = ServerConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

// ── DeviceConfig ────────────────────────────────────────────────────────────

#[test]
fn device_config_auto_debug() {
    insta::assert_debug_snapshot!(DeviceConfig::Auto);
}

#[test]
fn device_config_cpu_debug() {
    insta::assert_debug_snapshot!(DeviceConfig::Cpu);
}

#[test]
fn device_config_gpu0_debug() {
    insta::assert_debug_snapshot!(DeviceConfig::Gpu(0));
}

#[test]
fn device_config_gpu3_debug() {
    insta::assert_debug_snapshot!(DeviceConfig::Gpu(3));
}

#[test]
fn device_config_parse_auto() {
    let d: DeviceConfig = "auto".parse().unwrap();
    insta::assert_debug_snapshot!(d);
}

#[test]
fn device_config_parse_cpu() {
    let d: DeviceConfig = "cpu".parse().unwrap();
    insta::assert_debug_snapshot!(d);
}

#[test]
fn device_config_parse_gpu() {
    let d: DeviceConfig = "gpu".parse().unwrap();
    insta::assert_debug_snapshot!(d);
}

#[test]
fn device_config_parse_cuda_id() {
    let d: DeviceConfig = "cuda:2".parse().unwrap();
    insta::assert_debug_snapshot!(d);
}

#[test]
fn device_config_parse_invalid() {
    let err = "foobar".parse::<DeviceConfig>().unwrap_err();
    insta::assert_snapshot!(format!("{err}"));
}

// ── BatchEngineConfig ───────────────────────────────────────────────────────

#[test]
fn batch_engine_config_default_json() {
    let c = BatchEngineConfig::default();
    insta::assert_json_snapshot!(c);
}

#[test]
fn batch_engine_config_default_debug() {
    let c = BatchEngineConfig::default();
    insta::assert_debug_snapshot!(c);
}

// ── ConcurrencyConfig ───────────────────────────────────────────────────────

#[test]
fn concurrency_config_default_json() {
    let c = ConcurrencyConfig::default();
    insta::assert_json_snapshot!(c);
}

#[test]
fn concurrency_config_default_debug() {
    let c = ConcurrencyConfig::default();
    insta::assert_debug_snapshot!(c);
}

// ── SecurityConfig ──────────────────────────────────────────────────────────

#[test]
fn security_config_default_debug() {
    let c = SecurityConfig::default();
    insta::assert_debug_snapshot!(c);
}

#[test]
fn security_config_default_json() {
    let c = SecurityConfig::default();
    insta::assert_json_snapshot!(c);
}

// ── ModelManagerConfig ──────────────────────────────────────────────────────

#[test]
fn model_manager_config_default_debug() {
    let c = ModelManagerConfig::default();
    insta::assert_debug_snapshot!(c);
}

#[test]
fn model_manager_config_default_json() {
    let c = ModelManagerConfig::default();
    insta::assert_json_snapshot!(c);
}

// ── ExecutionRouterConfig ───────────────────────────────────────────────────

#[test]
fn execution_router_config_default_debug() {
    let c = ExecutionRouterConfig::default();
    insta::assert_debug_snapshot!(c);
}

#[test]
fn execution_router_config_default_json() {
    let c = ExecutionRouterConfig::default();
    insta::assert_json_snapshot!(c);
}

// ── DeviceSelectionStrategy ─────────────────────────────────────────────────

#[test]
fn device_selection_strategy_all_variants() {
    let variants: Vec<String> = [
        DeviceSelectionStrategy::PreferGpu,
        DeviceSelectionStrategy::CpuOnly,
        DeviceSelectionStrategy::PerformanceBased,
        DeviceSelectionStrategy::LoadBalance,
    ]
    .iter()
    .map(|v| format!("{v:?}"))
    .collect();
    insta::assert_debug_snapshot!(variants);
}

// ── DeviceHealth ────────────────────────────────────────────────────────────

#[test]
fn device_health_healthy() {
    insta::assert_debug_snapshot!(DeviceHealth::Healthy);
}

#[test]
fn device_health_degraded() {
    insta::assert_debug_snapshot!(DeviceHealth::Degraded { reason: "high memory usage".into() });
}

#[test]
fn device_health_unavailable() {
    insta::assert_debug_snapshot!(DeviceHealth::Unavailable { reason: "device not found".into() });
}

// ── ModelState ──────────────────────────────────────────────────────────────

#[test]
fn model_state_all_variants_debug() {
    let variants: Vec<String> =
        [ModelState::Loading, ModelState::Ready, ModelState::Serving, ModelState::Unloading]
            .iter()
            .map(|v| format!("{v:?}"))
            .collect();
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn model_state_all_variants_display() {
    let variants: Vec<String> =
        [ModelState::Loading, ModelState::Ready, ModelState::Serving, ModelState::Unloading]
            .iter()
            .map(|v| format!("{v}"))
            .collect();
    insta::assert_debug_snapshot!(variants);
}

// ── ModelEntry ──────────────────────────────────────────────────────────────

#[test]
fn model_entry_json() {
    let e = ModelEntry {
        model_id: "bitnet-2b".into(),
        device_id: "cuda:0".into(),
        state: ModelState::Ready,
        memory_bytes: 4_000_000_000,
        state_changed_at: None,
    };
    insta::assert_json_snapshot!(e);
}

// ── ErrorResponse / InferenceResponse ───────────────────────────────────────

#[test]
fn error_response_json_snapshot() {
    let e = ErrorResponse {
        error: "model not found".into(),
        error_code: "MODEL_NOT_FOUND".into(),
        request_id: Some("req-123".into()),
        details: None,
    };
    insta::assert_json_snapshot!(e);
}

#[test]
fn inference_response_json_snapshot() {
    let r = InferenceResponse {
        text: "Hello world".into(),
        tokens_generated: 3,
        inference_time_ms: 150,
        tokens_per_second: 20.0,
    };
    insta::assert_json_snapshot!(r);
}

// ── DeviceCapabilities ──────────────────────────────────────────────────────

#[test]
fn device_capabilities_cpu_json() {
    let c = DeviceCapabilities {
        device: bitnet_common::Device::Cpu,
        available: true,
        memory_total_mb: 32768,
        memory_free_mb: 16384,
        compute_capability: None,
        simd_support: vec!["AVX2".into(), "AVX-512".into()],
        avg_tokens_per_second: 15.0,
        last_benchmark: None,
    };
    insta::assert_json_snapshot!(c);
}
