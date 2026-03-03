//! Wave 10 snapshot tests for bitnet-inference.
//!
//! Covers: SamplingConfig, config_builder presets, GenerationConfig Debug,
//! InferenceConfig JSON serialization, streaming config, and validation errors.
#![allow(clippy::field_reassign_with_default)]

use bitnet_inference::config::{GenerationConfig, InferenceConfig};
use bitnet_inference::config_builder::{
    GenerationConfig as BuilderGenConfig, HardwareConfig, InferenceConfigBuilder, InferencePreset,
    SamplingConfig,
};
use bitnet_inference::streaming::StreamingConfig;

// -- SamplingConfig defaults -------------------------------------------------

#[test]
fn sampling_config_default_debug() {
    let cfg = SamplingConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn sampling_config_default_snapshot() {
    let cfg = SamplingConfig::default();
    insta::assert_snapshot!(format!(
        "temperature={} top_k={} top_p={} repetition_penalty={} seed={:?}",
        cfg.temperature, cfg.top_k, cfg.top_p, cfg.repetition_penalty, cfg.seed
    ));
}

// -- Builder GenerationConfig defaults ---------------------------------------

#[test]
fn builder_generation_config_default_debug() {
    let cfg = BuilderGenConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn builder_generation_config_default_snapshot() {
    let cfg = BuilderGenConfig::default();
    insta::assert_snapshot!(format!(
        "max_tokens={} stream={} stop_sequences={} stop_token_ids={}",
        cfg.max_tokens,
        cfg.stream,
        cfg.stop_sequences.len(),
        cfg.stop_token_ids.len()
    ));
}

// -- HardwareConfig defaults -------------------------------------------------

#[test]
fn hardware_config_default_debug() {
    let cfg = HardwareConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

// -- InferencePreset Debug ---------------------------------------------------

#[test]
fn inference_preset_all_variants_debug() {
    let presets = [
        InferencePreset::Fast,
        InferencePreset::Balanced,
        InferencePreset::Quality,
        InferencePreset::Deterministic,
        InferencePreset::Debug,
    ];
    let debug: Vec<String> = presets.iter().map(|p| format!("{p:?}")).collect();
    insta::assert_debug_snapshot!(debug);
}

// -- Built configs from each preset ------------------------------------------

#[test]
fn builder_preset_fast_debug() {
    let cfg = InferenceConfigBuilder::new().preset(InferencePreset::Fast).build().unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn builder_preset_balanced_debug() {
    let cfg = InferenceConfigBuilder::new().preset(InferencePreset::Balanced).build().unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn builder_preset_quality_debug() {
    let cfg = InferenceConfigBuilder::new().preset(InferencePreset::Quality).build().unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn builder_preset_deterministic_debug() {
    let cfg = InferenceConfigBuilder::new().preset(InferencePreset::Deterministic).build().unwrap();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn builder_preset_debug_debug() {
    let cfg = InferenceConfigBuilder::new().preset(InferencePreset::Debug).build().unwrap();
    insta::assert_debug_snapshot!(cfg);
}

// -- GenerationConfig Debug (from config module) -----------------------------

#[test]
fn generation_config_default_debug() {
    let cfg = GenerationConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn generation_config_greedy_debug() {
    let cfg = GenerationConfig::greedy();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn generation_config_creative_debug() {
    let cfg = GenerationConfig::creative();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn generation_config_balanced_debug() {
    let cfg = GenerationConfig::balanced();
    insta::assert_debug_snapshot!(cfg);
}

// -- InferenceConfig presets -------------------------------------------------

#[test]
fn inference_config_cpu_optimized_debug() {
    let mut cfg = InferenceConfig::cpu_optimized();
    // Pin num_threads to avoid machine-dependent snapshots
    cfg.num_threads = 4;
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn inference_config_gpu_optimized_debug() {
    let cfg = InferenceConfig::gpu_optimized();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn inference_config_memory_efficient_debug() {
    let cfg = InferenceConfig::memory_efficient();
    insta::assert_debug_snapshot!(cfg);
}

// -- InferenceConfig JSON serialization round-trip ---------------------------

#[test]
fn inference_config_default_json() {
    let mut cfg = InferenceConfig::default();
    cfg.num_threads = 4; // Pin for deterministic output
    insta::assert_json_snapshot!(cfg);
}

// -- GenerationConfig JSON serialization ------------------------------------

#[test]
fn generation_config_greedy_json() {
    let cfg = GenerationConfig::greedy();
    insta::assert_json_snapshot!(cfg);
}

// -- Validation error messages -----------------------------------------------

#[test]
fn generation_config_validate_negative_temperature_error() {
    let cfg = GenerationConfig::default().with_temperature(-1.0);
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(err);
}

#[test]
fn generation_config_validate_top_p_zero_error() {
    let cfg = GenerationConfig::default().with_top_p(0.0);
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(err);
}

#[test]
fn generation_config_validate_top_p_above_one_error() {
    let cfg = GenerationConfig::default().with_top_p(1.5);
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(err);
}

#[test]
fn generation_config_validate_repetition_penalty_zero_error() {
    let cfg = GenerationConfig::default().with_repetition_penalty(0.0);
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(err);
}

// -- Builder validation errors -----------------------------------------------

#[test]
fn builder_validate_negative_temperature_error() {
    let err = InferenceConfigBuilder::new().temperature(-0.1).build().unwrap_err();
    insta::assert_snapshot!(err);
}

#[test]
fn builder_validate_max_tokens_zero_error() {
    let err = InferenceConfigBuilder::new().max_tokens(0).build().unwrap_err();
    insta::assert_snapshot!(err);
}

#[test]
fn builder_validate_num_threads_too_large_error() {
    let err = InferenceConfigBuilder::new().num_threads(2000).build().unwrap_err();
    insta::assert_snapshot!(err);
}

// -- StreamingConfig ---------------------------------------------------------

#[test]
fn streaming_config_default_debug() {
    let cfg = StreamingConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn streaming_config_low_latency_debug() {
    let cfg = StreamingConfig::low_latency();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn streaming_config_high_throughput_debug() {
    let cfg = StreamingConfig::high_throughput();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn streaming_config_validate_zero_buffer_error() {
    let mut cfg = StreamingConfig::default();
    cfg.buffer_size = 0;
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn streaming_config_validate_zero_flush_error() {
    let mut cfg = StreamingConfig::default();
    cfg.flush_interval_ms = 0;
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(err.to_string());
}

#[test]
fn streaming_config_validate_zero_timeout_error() {
    let mut cfg = StreamingConfig::default();
    cfg.token_timeout_ms = 0;
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(err.to_string());
}

// -- Builder with stop sequences JSON ----------------------------------------

#[test]
fn builder_config_with_stops_json() {
    let cfg = InferenceConfigBuilder::new()
        .preset(InferencePreset::Quality)
        .stop_sequence("</s>")
        .stop_sequence("\n\nQ:")
        .stop_token_id(128009)
        .stop_token_id(128001)
        .seed(42)
        .build()
        .unwrap();
    insta::assert_json_snapshot!(cfg);
}
