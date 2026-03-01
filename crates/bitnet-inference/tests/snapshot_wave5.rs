//! Wave 5 snapshot tests for `bitnet-inference` configuration and strategy types.
//!
//! Pins the Debug/Display formats and default values of types that must remain
//! stable across releases.

use bitnet_inference::config::{GenerationConfig, InferenceConfig};

// ---------------------------------------------------------------------------
// GenerationConfig — balanced preset
// ---------------------------------------------------------------------------

#[test]
fn generation_config_balanced_snapshot() {
    let cfg = GenerationConfig::balanced();
    insta::assert_snapshot!(format!(
        "temperature={} top_k={} top_p={} repetition_penalty={}",
        cfg.temperature, cfg.top_k, cfg.top_p, cfg.repetition_penalty
    ));
}

// ---------------------------------------------------------------------------
// GenerationConfig — default fields
// ---------------------------------------------------------------------------

#[test]
fn generation_config_default_all_fields() {
    let cfg = GenerationConfig::default();
    insta::assert_snapshot!(format!(
        "max_new_tokens={} temperature={} top_k={} top_p={} \
         repetition_penalty={} skip_special_tokens={} add_bos={}",
        cfg.max_new_tokens,
        cfg.temperature,
        cfg.top_k,
        cfg.top_p,
        cfg.repetition_penalty,
        cfg.skip_special_tokens,
        cfg.add_bos,
    ));
}

#[test]
fn generation_config_greedy_fields() {
    let cfg = GenerationConfig::greedy();
    insta::assert_snapshot!(format!(
        "temperature={} top_k={} top_p={} repetition_penalty={}",
        cfg.temperature, cfg.top_k, cfg.top_p, cfg.repetition_penalty,
    ));
}

// ---------------------------------------------------------------------------
// GenerationConfig — builder chain snapshot
// ---------------------------------------------------------------------------

#[test]
fn generation_config_builder_chain_fields() {
    let cfg = GenerationConfig::greedy()
        .with_max_tokens(16)
        .with_temperature(0.5)
        .with_top_k(10)
        .with_top_p(0.8)
        .with_seed(42);
    insta::assert_snapshot!(format!(
        "max_new_tokens={} temperature={} top_k={} top_p={} seed={:?}",
        cfg.max_new_tokens, cfg.temperature, cfg.top_k, cfg.top_p, cfg.seed,
    ));
}

// ---------------------------------------------------------------------------
// GenerationConfig — validation error messages
// ---------------------------------------------------------------------------

#[test]
fn generation_config_validate_negative_temperature_error() {
    let cfg = GenerationConfig::default().with_temperature(-1.0);
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(err);
}

#[test]
fn generation_config_validate_invalid_top_p_error() {
    let cfg = GenerationConfig::default().with_top_p(0.0);
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(err);
}

// ---------------------------------------------------------------------------
// InferenceConfig — full Debug output
// ---------------------------------------------------------------------------

#[test]
fn inference_config_default_debug() {
    let cfg = InferenceConfig::default();
    insta::with_settings!({filters => vec![
        (r"num_threads: \d+", "num_threads: [N]"),
    ]}, {
        insta::assert_debug_snapshot!(cfg);
    });
}

// ---------------------------------------------------------------------------
// SamplingConfig — Debug output
// ---------------------------------------------------------------------------

#[test]
fn sampling_config_default_debug() {
    use bitnet_inference::generation::sampling::SamplingConfig;

    let cfg = SamplingConfig::default();
    insta::assert_debug_snapshot!(cfg);
}
