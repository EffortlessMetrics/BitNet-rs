//! Wave 8 snapshot tests — GenerationConfig and sampling presets.
//!
//! Extends the existing snapshot_tests.rs and snapshot_wave5.rs by pinning
//! builder edge-cases, validation error messages, stop-token configuration,
//! and the SamplingStrategy preset surfaces.

use bitnet_inference::config::{GenerationConfig, InferenceConfig};
use bitnet_inference::generation::sampling::{SamplingConfig, SamplingStrategy};

// ---------------------------------------------------------------------------
// GenerationConfig — stop-token configuration
// ---------------------------------------------------------------------------

#[test]
fn generation_config_with_stop_sequences_snapshot() {
    let cfg = GenerationConfig::default()
        .with_stop_sequence("</s>".into())
        .with_stop_sequence("\n\nQ:".into());
    insta::assert_snapshot!("gen_config_stop_sequences", format!("{:?}", cfg.stop_sequences));
}

#[test]
fn generation_config_with_stop_token_ids_snapshot() {
    let cfg =
        GenerationConfig::default().with_stop_token_ids(vec![128009, 128001]).with_stop_token_id(2);
    insta::assert_snapshot!("gen_config_stop_token_ids", format!("{:?}", cfg.stop_token_ids));
}

#[test]
fn generation_config_stop_string_window_default() {
    let cfg = GenerationConfig::default();
    insta::assert_snapshot!(
        "gen_config_stop_window_default",
        format!("{}", cfg.stop_string_window)
    );
}

#[test]
fn generation_config_custom_stop_window() {
    let cfg = GenerationConfig::default().with_stop_string_window(256);
    insta::assert_snapshot!("gen_config_stop_window_custom", format!("{}", cfg.stop_string_window));
}

// ---------------------------------------------------------------------------
// GenerationConfig — EOS and special token handling
// ---------------------------------------------------------------------------

#[test]
fn generation_config_eos_token_id_snapshot() {
    let cfg = GenerationConfig::default().with_eos_token_id(Some(128001));
    insta::assert_snapshot!(
        "gen_config_eos_token_id",
        format!("eos_token_id={:?}", cfg.eos_token_id)
    );
}

#[test]
fn generation_config_skip_special_tokens_default() {
    let cfg = GenerationConfig::default();
    insta::assert_snapshot!(
        "gen_config_skip_special_default",
        format!("skip_special_tokens={}", cfg.skip_special_tokens)
    );
}

#[test]
fn generation_config_add_bos_override() {
    let cfg = GenerationConfig::default().with_add_bos(true);
    insta::assert_snapshot!("gen_config_add_bos_true", format!("add_bos={}", cfg.add_bos));
}

// ---------------------------------------------------------------------------
// GenerationConfig — logits tap configuration
// ---------------------------------------------------------------------------

#[test]
fn generation_config_logits_tap_defaults() {
    let cfg = GenerationConfig::default();
    insta::assert_snapshot!(
        "gen_config_logits_tap_defaults",
        format!("tap_steps={}, topk={}", cfg.logits_tap_steps, cfg.logits_topk)
    );
}

#[test]
fn generation_config_logits_tap_custom() {
    let cfg = GenerationConfig::default().with_logits_tap_steps(10).with_logits_topk(5);
    insta::assert_snapshot!(
        "gen_config_logits_tap_custom",
        format!("tap_steps={}, topk={}", cfg.logits_tap_steps, cfg.logits_topk)
    );
}

// ---------------------------------------------------------------------------
// GenerationConfig — validation error messages
// ---------------------------------------------------------------------------

#[test]
fn generation_config_validate_zero_repetition_penalty() {
    let cfg = GenerationConfig::default().with_repetition_penalty(0.0);
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!("gen_config_validate_zero_rep_penalty", err);
}

#[test]
fn generation_config_validate_negative_repetition_penalty() {
    let cfg = GenerationConfig::default().with_repetition_penalty(-1.0);
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!("gen_config_validate_neg_rep_penalty", err);
}

#[test]
fn generation_config_validate_top_p_over_one() {
    let cfg = GenerationConfig::default().with_top_p(1.5);
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!("gen_config_validate_top_p_over_one", err);
}

#[test]
fn generation_config_validate_ok_snapshot() {
    let cfg = GenerationConfig::balanced();
    let result = cfg.validate();
    insta::assert_snapshot!("gen_config_validate_ok", format!("{result:?}"));
}

// ---------------------------------------------------------------------------
// GenerationConfig — JSON round-trip
// ---------------------------------------------------------------------------

#[test]
fn generation_config_greedy_json_snapshot() {
    let cfg = GenerationConfig::greedy().with_max_tokens(16).with_seed(42);
    insta::assert_json_snapshot!("gen_config_greedy_json", cfg);
}

#[test]
fn generation_config_creative_json_snapshot() {
    let cfg = GenerationConfig::creative().with_max_tokens(256);
    insta::assert_json_snapshot!("gen_config_creative_json", cfg);
}

// ---------------------------------------------------------------------------
// InferenceConfig — field stability
// ---------------------------------------------------------------------------

#[test]
fn inference_config_default_json_snapshot() {
    let cfg = InferenceConfig { num_threads: 0, ..Default::default() };
    insta::assert_json_snapshot!("inference_config_default_json", cfg);
}

// ---------------------------------------------------------------------------
// SamplingConfig — presets
// ---------------------------------------------------------------------------

#[test]
fn sampling_config_default_snapshot() {
    let cfg = SamplingConfig::default();
    insta::assert_debug_snapshot!("sampling_config_default", cfg);
}

// ---------------------------------------------------------------------------
// SamplingStrategy — preset Debug snapshots
// ---------------------------------------------------------------------------

#[test]
fn sampling_strategy_deterministic_snapshot() {
    let strat = SamplingStrategy::deterministic();
    insta::assert_debug_snapshot!("sampling_strategy_deterministic", strat);
}

#[test]
fn sampling_strategy_creative_snapshot() {
    let strat = SamplingStrategy::creative();
    insta::assert_debug_snapshot!("sampling_strategy_creative", strat);
}

#[test]
fn sampling_strategy_balanced_snapshot() {
    let strat = SamplingStrategy::balanced();
    insta::assert_debug_snapshot!("sampling_strategy_balanced", strat);
}

#[test]
fn sampling_strategy_conservative_snapshot() {
    let strat = SamplingStrategy::conservative();
    insta::assert_debug_snapshot!("sampling_strategy_conservative", strat);
}
