//! Snapshot wave 27 — bitnet-inference config/profiler output stability.

use bitnet_inference::config_builder::{
    GenerationConfig, HardwareConfig, InferenceConfigBuilder, InferencePreset, SamplingConfig,
};
use bitnet_inference::profiler::{
    LayerProfile, LayerStats, MemorySnapshot, ProfileReport, ProfilerConfig,
};

// ---------------------------------------------------------------------------
// InferencePreset — all variants
// ---------------------------------------------------------------------------

#[test]
fn snapshot_preset_fast_config_json() {
    let cfg = InferenceConfigBuilder::new().preset(InferencePreset::Fast).build().unwrap();
    insta::assert_json_snapshot!("preset_fast_config", cfg);
}

#[test]
fn snapshot_preset_balanced_config_json() {
    let cfg = InferenceConfigBuilder::new().preset(InferencePreset::Balanced).build().unwrap();
    insta::assert_json_snapshot!("preset_balanced_config", cfg);
}

#[test]
fn snapshot_preset_quality_config_json() {
    let cfg = InferenceConfigBuilder::new().preset(InferencePreset::Quality).build().unwrap();
    insta::assert_json_snapshot!("preset_quality_config", cfg);
}

#[test]
fn snapshot_preset_deterministic_config_json() {
    let cfg = InferenceConfigBuilder::new().preset(InferencePreset::Deterministic).build().unwrap();
    insta::assert_json_snapshot!("preset_deterministic_config", cfg);
}

#[test]
fn snapshot_preset_debug_config_json() {
    let cfg = InferenceConfigBuilder::new().preset(InferencePreset::Debug).build().unwrap();
    insta::assert_json_snapshot!("preset_debug_config", cfg);
}

// ---------------------------------------------------------------------------
// SamplingConfig defaults
// ---------------------------------------------------------------------------

#[test]
fn snapshot_sampling_config_default_json() {
    let cfg = SamplingConfig::default();
    insta::assert_json_snapshot!("sampling_config_default", cfg);
}

#[test]
fn snapshot_sampling_config_debug() {
    let cfg = SamplingConfig {
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(42),
    };
    insta::assert_debug_snapshot!("sampling_config_greedy", cfg);
}

// ---------------------------------------------------------------------------
// GenerationConfig
// ---------------------------------------------------------------------------

#[test]
fn snapshot_generation_config_default_json() {
    let cfg = GenerationConfig::default();
    insta::assert_json_snapshot!("generation_config_default", cfg);
}

#[test]
fn snapshot_generation_config_with_stops_json() {
    let cfg = InferenceConfigBuilder::new()
        .max_tokens(64)
        .stop_sequence("</s>")
        .stop_sequence("\n\nQ:")
        .stop_token_id(128009)
        .stop_token_id(128001)
        .stream(true)
        .build()
        .unwrap();
    insta::assert_json_snapshot!("generation_config_with_stops", cfg.generation);
}

// ---------------------------------------------------------------------------
// HardwareConfig
// ---------------------------------------------------------------------------

#[test]
fn snapshot_hardware_config_default_json() {
    let cfg = HardwareConfig::default();
    insta::assert_json_snapshot!("hardware_config_default", cfg);
}

// ---------------------------------------------------------------------------
// InferenceConfig — custom build
// ---------------------------------------------------------------------------

#[test]
fn snapshot_custom_inference_config_json() {
    let cfg = InferenceConfigBuilder::new()
        .temperature(0.8)
        .top_k(40)
        .top_p(0.95)
        .repetition_penalty(1.1)
        .seed(12345)
        .max_tokens(256)
        .stop_sequence("</s>")
        .stop_token_id(128009)
        .stream(true)
        .num_threads(4)
        .memory_limit_mb(4096)
        .build()
        .unwrap();
    insta::assert_json_snapshot!("custom_inference_config", cfg);
}

#[test]
fn snapshot_inference_config_debug() {
    let cfg = InferenceConfigBuilder::new().preset(InferencePreset::Fast).build().unwrap();
    insta::assert_debug_snapshot!("inference_config_debug_fast", cfg);
}

// ---------------------------------------------------------------------------
// InferencePreset serde roundtrip
// ---------------------------------------------------------------------------

#[test]
fn snapshot_preset_enum_json_all() {
    let presets = vec![
        InferencePreset::Fast,
        InferencePreset::Balanced,
        InferencePreset::Quality,
        InferencePreset::Deterministic,
        InferencePreset::Debug,
    ];
    insta::assert_json_snapshot!("preset_enum_all_json", presets);
}

// ---------------------------------------------------------------------------
// ProfilerConfig
// ---------------------------------------------------------------------------

#[test]
fn snapshot_profiler_config_default_json() {
    let cfg = ProfilerConfig::default();
    insta::assert_json_snapshot!("profiler_config_default", cfg);
}

#[test]
fn snapshot_profiler_config_disabled_json() {
    let cfg = ProfilerConfig::disabled();
    insta::assert_json_snapshot!("profiler_config_disabled", cfg);
}

#[test]
fn snapshot_profiler_config_custom_json() {
    let cfg = ProfilerConfig::default().with_warmup(3).with_sample_size(10).with_memory(true);
    insta::assert_json_snapshot!("profiler_config_custom", cfg);
}

// ---------------------------------------------------------------------------
// ProfileReport
// ---------------------------------------------------------------------------

#[test]
fn snapshot_profile_report_empty_json() {
    let report = ProfileReport {
        total_time_us: 0.0,
        per_layer_breakdown: vec![],
        bottleneck_layers: vec![],
        memory_peak: 0,
        estimated_flops: 0,
        memory_snapshots: vec![],
        layer_profiles: vec![],
    };
    insta::assert_json_snapshot!("profile_report_empty", report);
}

#[test]
fn snapshot_profile_report_with_layers_json() {
    let report = ProfileReport {
        total_time_us: 2500.0,
        per_layer_breakdown: vec![
            LayerStats {
                layer_name: "layer_0.attention".into(),
                layer_type: "attention".into(),
                mean_time_us: 1200.0,
                std_time_us: 50.0,
                min_time_us: 1150.0,
                max_time_us: 1250.0,
                count: 5,
                total_memory_bytes: 4_194_304,
                total_flops: 1_000_000,
            },
            LayerStats {
                layer_name: "layer_0.ffn".into(),
                layer_type: "ffn".into(),
                mean_time_us: 800.0,
                std_time_us: 30.0,
                min_time_us: 770.0,
                max_time_us: 830.0,
                count: 5,
                total_memory_bytes: 2_097_152,
                total_flops: 500_000,
            },
        ],
        bottleneck_layers: vec!["layer_0.attention".into()],
        memory_peak: 8_388_608,
        estimated_flops: 1_500_000,
        memory_snapshots: vec![MemorySnapshot {
            label: "post_attention".into(),
            timestamp_us: 1200.0,
            memory_bytes: 8_388_608,
        }],
        layer_profiles: vec![
            LayerProfile {
                layer_name: "layer_0.attention".into(),
                layer_type: "attention".into(),
                forward_time_us: 1200.0,
                backward_time_us: 0.0,
                memory_bytes: 4_194_304,
                flops_estimate: 1_000_000,
            },
            LayerProfile {
                layer_name: "layer_0.ffn".into(),
                layer_type: "ffn".into(),
                forward_time_us: 800.0,
                backward_time_us: 0.0,
                memory_bytes: 2_097_152,
                flops_estimate: 500_000,
            },
        ],
    };
    insta::assert_json_snapshot!("profile_report_with_layers", report);
}

#[test]
fn snapshot_profile_report_chrome_trace() {
    let report = ProfileReport {
        total_time_us: 500.0,
        per_layer_breakdown: vec![],
        bottleneck_layers: vec![],
        memory_peak: 0,
        estimated_flops: 0,
        memory_snapshots: vec![],
        layer_profiles: vec![LayerProfile {
            layer_name: "layer_0.norm".into(),
            layer_type: "norm".into(),
            forward_time_us: 500.0,
            backward_time_us: 0.0,
            memory_bytes: 0,
            flops_estimate: 0,
        }],
    };
    insta::assert_snapshot!("profile_report_chrome_trace", report.export_chrome_trace());
}
