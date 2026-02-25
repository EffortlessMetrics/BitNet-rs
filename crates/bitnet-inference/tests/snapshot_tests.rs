//! Snapshot tests for stable bitnet-inference configuration defaults and error formats.
//! These test key structural constants that should never silently change.

use bitnet_inference::config::{GenerationConfig, InferenceConfig};

#[test]
fn generation_config_default_max_new_tokens() {
    let cfg = GenerationConfig::default();
    insta::assert_snapshot!(format!("max_new_tokens={}", cfg.max_new_tokens));
}

#[test]
fn generation_config_default_temperature() {
    let cfg = GenerationConfig::default();
    insta::assert_snapshot!(format!("temperature={}", cfg.temperature));
}

#[test]
fn generation_config_greedy_temperature_is_zero() {
    let cfg = GenerationConfig::greedy();
    insta::assert_snapshot!(format!("temperature={} top_k={}", cfg.temperature, cfg.top_k));
}

#[test]
fn generation_config_creative_temperature_and_top_k() {
    let cfg = GenerationConfig::creative();
    insta::assert_snapshot!(format!(
        "temperature={} top_k={} top_p={}",
        cfg.temperature, cfg.top_k, cfg.top_p
    ));
}

#[test]
fn inference_config_default_context_length() {
    let cfg = InferenceConfig::default();
    insta::assert_snapshot!(format!("max_context_length={}", cfg.max_context_length));
}

#[test]
fn inference_config_default_batch_size() {
    let cfg = InferenceConfig::default();
    insta::assert_snapshot!(format!("batch_size={}", cfg.batch_size));
}

#[test]
fn generation_config_validate_zero_tokens_error() {
    let cfg = GenerationConfig::default().with_max_tokens(0);
    let err = cfg.validate().unwrap_err();
    insta::assert_snapshot!(err);
}
