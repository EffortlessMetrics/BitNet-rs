//! Snapshot tests for `bitnet-generation` public API surface.
//!
//! Pins the serialized forms of generation contracts — stopping criteria,
//! stream events, and configuration — to catch unintended wire-format changes.

use bitnet_generation::{GenerationConfig, StopCriteria, StopReason};

#[test]
fn stop_reason_debug_all_variants() {
    let reasons = [
        StopReason::MaxTokens,
        StopReason::StopTokenId(128_009),
        StopReason::StopString("</s>".to_string()),
        StopReason::EosToken,
    ];
    let debug: Vec<String> = reasons.iter().map(|r| format!("{r:?}")).collect();
    insta::assert_debug_snapshot!("stop_reason_debug_variants", debug);
}

#[test]
fn generation_config_default_debug_snapshot() {
    let config = GenerationConfig::default();
    insta::assert_debug_snapshot!("generation_config_default", config);
}

#[test]
fn stop_criteria_default_debug_snapshot() {
    let criteria = StopCriteria::default();
    insta::assert_debug_snapshot!("stop_criteria_default", criteria);
}

#[test]
fn stop_criteria_with_token_ids_debug_snapshot() {
    let criteria = StopCriteria {
        stop_token_ids: vec![128_009, 2],
        stop_strings: vec!["</s>".to_string()],
        max_tokens: 64,
        eos_token_id: Some(2),
    };
    insta::assert_debug_snapshot!("stop_criteria_with_values", criteria);
}
