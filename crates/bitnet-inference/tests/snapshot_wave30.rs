//! Snapshot wave 30 — bitnet-inference
//!
//! Pins Debug/Display representations of inference configuration types,
//! sampling strategies, stop reasons, and generation budgets.

use bitnet_inference::config_builder::{
    GenerationConfig as BuilderGenConfig, HardwareConfig, InferenceConfigBuilder, InferencePreset,
    SamplingConfig as BuilderSamplingConfig,
};
use bitnet_inference::generation_budget::{GenerationBudget, StopReason};
use std::time::Duration;

// =========================================================================
// Section 1 — StopReason
// =========================================================================

#[test]
fn snapshot_wave30__stop_reason_all_display() {
    let reasons = [
        StopReason::MaxTokens,
        StopReason::TimeLimit,
        StopReason::MemoryLimit,
        StopReason::EndOfSequence,
        StopReason::UserStop,
    ];
    let output: Vec<String> = reasons.iter().map(|r| format!("{r}")).collect();
    insta::assert_snapshot!(output.join("\n"));
}

#[test]
fn snapshot_wave30__stop_reason_all_debug() {
    let reasons = [
        StopReason::MaxTokens,
        StopReason::TimeLimit,
        StopReason::MemoryLimit,
        StopReason::EndOfSequence,
        StopReason::UserStop,
    ];
    insta::assert_debug_snapshot!(reasons);
}

#[test]
fn snapshot_wave30__stop_reason_eos_display() {
    insta::assert_snapshot!(format!("{}", StopReason::EndOfSequence));
}

// =========================================================================
// Section 2 — GenerationBudget
// =========================================================================

#[test]
fn snapshot_wave30__generation_budget_default() {
    let budget = GenerationBudget::default();
    insta::assert_debug_snapshot!(budget);
}

#[test]
fn snapshot_wave30__generation_budget_with_limits() {
    let budget = GenerationBudget::new(512)
        .with_time_limit(Duration::from_secs(30))
        .with_memory_limit(1024 * 1024 * 256);
    insta::assert_debug_snapshot!(budget);
}

#[test]
fn snapshot_wave30__generation_budget_unlimited() {
    let budget = GenerationBudget::unlimited();
    insta::assert_debug_snapshot!(budget);
}

// =========================================================================
// Section 3 — config_builder types
// =========================================================================

#[test]
fn snapshot_wave30__sampling_config_default() {
    let cfg = BuilderSamplingConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn snapshot_wave30__generation_config_default() {
    let cfg = BuilderGenConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn snapshot_wave30__hardware_config_default() {
    let cfg = HardwareConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn snapshot_wave30__inference_preset_all_debug() {
    let presets = [
        InferencePreset::Fast,
        InferencePreset::Balanced,
        InferencePreset::Quality,
        InferencePreset::Deterministic,
        InferencePreset::Debug,
    ];
    insta::assert_debug_snapshot!(presets);
}

#[test]
fn snapshot_wave30__inference_config_fast_preset_json() {
    let cfg = InferenceConfigBuilder::new().preset(InferencePreset::Fast).build().unwrap();
    insta::assert_json_snapshot!(cfg);
}

#[test]
fn snapshot_wave30__inference_config_balanced_preset_json() {
    let cfg = InferenceConfigBuilder::new().preset(InferencePreset::Balanced).build().unwrap();
    insta::assert_json_snapshot!(cfg);
}

#[test]
fn snapshot_wave30__inference_config_quality_preset_json() {
    let cfg = InferenceConfigBuilder::new().preset(InferencePreset::Quality).build().unwrap();
    insta::assert_json_snapshot!(cfg);
}

#[test]
fn snapshot_wave30__inference_config_deterministic_preset_json() {
    let cfg = InferenceConfigBuilder::new().preset(InferencePreset::Deterministic).build().unwrap();
    insta::assert_json_snapshot!(cfg);
}

#[test]
fn snapshot_wave30__inference_config_debug_preset_json() {
    let cfg = InferenceConfigBuilder::new().preset(InferencePreset::Debug).build().unwrap();
    insta::assert_json_snapshot!(cfg);
}
