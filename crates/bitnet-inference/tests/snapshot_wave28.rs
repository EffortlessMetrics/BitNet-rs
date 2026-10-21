//! Wave 28 insta snapshot tests for bitnet-inference types.

use bitnet_inference::generation::autoregressive::GenerationConfig;
use bitnet_inference::generation::sampling::{SamplingConfig, SamplingStrategy};
use bitnet_inference::generation_budget::{GenerationBudget, StopReason};

// ── SamplingConfig / SamplingStrategy ────────────────────────────────────────

#[test]
fn snapshot_sampling_config_default() {
    let cfg = SamplingConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
fn snapshot_sampling_strategy_deterministic() {
    let s = SamplingStrategy::deterministic();
    insta::assert_debug_snapshot!(s);
}

#[test]
fn snapshot_sampling_strategy_creative() {
    let s = SamplingStrategy::creative();
    insta::assert_debug_snapshot!(s);
}

#[test]
fn snapshot_sampling_strategy_balanced() {
    let s = SamplingStrategy::balanced();
    insta::assert_debug_snapshot!(s);
}

#[test]
fn snapshot_sampling_strategy_conservative() {
    let s = SamplingStrategy::conservative();
    insta::assert_debug_snapshot!(s);
}

#[test]
fn snapshot_sampling_config_custom_temperature() {
    let cfg = SamplingConfig {
        temperature: 0.42,
        top_k: Some(10),
        top_p: Some(0.85),
        repetition_penalty: 1.3,
        do_sample: true,
    };
    insta::assert_debug_snapshot!(cfg);
}

// ── GenerationConfig ─────────────────────────────────────────────────────────

#[test]
fn snapshot_generation_config_default() {
    let cfg = GenerationConfig::default();
    insta::assert_debug_snapshot!(cfg);
}

#[test]
#[allow(clippy::field_reassign_with_default)]
fn snapshot_generation_config_custom() {
    let mut cfg = GenerationConfig::default();
    cfg.max_new_tokens = 128;
    cfg.temperature = 0.7;
    cfg.top_k = Some(20);
    cfg.top_p = Some(0.95);
    cfg.repetition_penalty = 1.05;
    cfg.do_sample = false;
    cfg.seed = Some(42);
    cfg.eos_token_id = 128009;
    cfg.pad_token_id = 0;
    cfg.min_length = 1;
    cfg.max_length = 4096;
    insta::assert_debug_snapshot!(cfg);
}

// ── GenerationBudget ─────────────────────────────────────────────────────────

#[test]
fn snapshot_generation_budget_default() {
    let budget = GenerationBudget::default();
    insta::assert_debug_snapshot!(budget);
}

#[test]
fn snapshot_generation_budget_with_limits() {
    let budget = GenerationBudget::new(1024)
        .with_time_limit(std::time::Duration::from_mins(1))
        .with_memory_limit(1024 * 1024 * 512);
    insta::assert_debug_snapshot!(budget);
}

#[test]
fn snapshot_generation_budget_unlimited() {
    let budget = GenerationBudget::unlimited();
    insta::assert_debug_snapshot!(budget);
}

// ── StopReason ───────────────────────────────────────────────────────────────

#[test]
fn snapshot_stop_reason_all_variants_debug() {
    let variants = vec![
        StopReason::MaxTokens,
        StopReason::TimeLimit,
        StopReason::MemoryLimit,
        StopReason::EndOfSequence,
        StopReason::UserStop,
    ];
    insta::assert_debug_snapshot!(variants);
}

#[test]
fn snapshot_stop_reason_max_tokens_display() {
    insta::assert_snapshot!(StopReason::MaxTokens.to_string());
}

#[test]
fn snapshot_stop_reason_end_of_sequence_display() {
    insta::assert_snapshot!(StopReason::EndOfSequence.to_string());
}

#[test]
fn snapshot_stop_reason_user_stop_display() {
    insta::assert_snapshot!(StopReason::UserStop.to_string());
}
