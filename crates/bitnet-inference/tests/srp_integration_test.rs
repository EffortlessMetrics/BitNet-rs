//! Cross-crate SRP integration tests.
//!
//! Verifies that the SRP microcrate wiring works end-to-end through the
//! `bitnet-inference` dependency graph:
//!
//! - `bitnet-logits`          – pure logit transforms
//! - `bitnet-generation`      – stop criteria and generation events
//! - `bitnet-prompt-templates`– prompt formatting
//! - `bitnet-engine-core`     – `SessionConfig` serde round-trip

use bitnet_engine_core::SessionConfig;
use bitnet_generation::{StopCriteria, StopReason, check_stop};
use bitnet_logits::{apply_temperature, apply_top_k, argmax, softmax_in_place};
use bitnet_prompt_templates::{PromptTemplate, TemplateType};

// ---------------------------------------------------------------------------
// bitnet-logits
// ---------------------------------------------------------------------------

#[test]
fn logits_temperature_scales_values() {
    let mut logits = vec![1.0f32, 2.0, 4.0];
    apply_temperature(&mut logits, 2.0);
    // Each value should be halved (multiplied by 1/temperature = 0.5)
    assert!((logits[0] - 0.5).abs() < 1e-6);
    assert!((logits[1] - 1.0).abs() < 1e-6);
    assert!((logits[2] - 2.0).abs() < 1e-6);
}

#[test]
fn logits_softmax_sums_to_one() {
    let mut logits = vec![1.0f32, 2.0, 3.0, 0.5];
    softmax_in_place(&mut logits);
    let sum: f32 = logits.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum}");
    // All probabilities must be non-negative
    assert!(logits.iter().all(|&p| p >= 0.0));
}

#[test]
fn logits_top_k_one_leaves_single_candidate() {
    let mut logits = vec![1.0f32, 5.0, 3.0, 2.0];
    let kept = apply_top_k(&mut logits, 1);
    assert_eq!(kept, 1, "only one token should survive top-k=1");
    // The highest-value token (index 1, value 5.0) must survive.
    let best = argmax(&logits);
    assert_eq!(best, 1);
    // All other positions must be NEG_INFINITY.
    for (i, &v) in logits.iter().enumerate() {
        if i != best {
            assert_eq!(v, f32::NEG_INFINITY, "logits[{i}] should be NEG_INFINITY");
        }
    }
}

// ---------------------------------------------------------------------------
// bitnet-generation
// ---------------------------------------------------------------------------

#[test]
fn check_stop_eos_token_id() {
    let criteria = StopCriteria { eos_token_id: Some(2), ..Default::default() };
    let result = check_stop(&criteria, 2, &[2], "hello");
    assert_eq!(result, Some(StopReason::EosToken));
}

#[test]
fn check_stop_max_tokens_budget() {
    let criteria = StopCriteria { max_tokens: 3, ..Default::default() };
    // generated has 3 tokens – budget exhausted
    let result = check_stop(&criteria, 99, &[10, 20, 30], "some text");
    assert_eq!(result, Some(StopReason::MaxTokens));
}

#[test]
fn check_stop_string_in_tail() {
    let criteria = StopCriteria { stop_strings: vec!["\n\nQ:".to_string()], ..Default::default() };
    let result = check_stop(&criteria, 42, &[42], "previous answer\n\nQ: next");
    assert_eq!(result, Some(StopReason::StopString("\n\nQ:".to_string())));
}

#[test]
fn check_stop_no_condition_returns_none() {
    let criteria = StopCriteria { max_tokens: 10, ..Default::default() };
    // Only 2 tokens generated – well under the budget.
    let result = check_stop(&criteria, 7, &[5, 7], "hi");
    assert_eq!(result, None);
}

// ---------------------------------------------------------------------------
// bitnet-prompt-templates
// ---------------------------------------------------------------------------

#[test]
fn prompt_template_raw_passes_through_unchanged() {
    let tpl = PromptTemplate::new(TemplateType::Raw);
    let input = "What is 2+2?";
    assert_eq!(tpl.format(input), input);
}

#[test]
fn prompt_template_instruct_wraps_in_qa_format() {
    let tpl = PromptTemplate::new(TemplateType::Instruct);
    let formatted = tpl.format("What is the capital of France?");
    assert!(formatted.starts_with("Q: "), "should start with 'Q: '");
    assert!(formatted.contains("What is the capital of France?"));
    assert!(formatted.contains("\nA:"), "should contain '\\nA:'");
}

#[test]
fn prompt_template_instruct_with_system_prompt() {
    let tpl = PromptTemplate::new(TemplateType::Instruct)
        .with_system_prompt("You are a helpful assistant.");
    let formatted = tpl.format("Hello");
    assert!(formatted.starts_with("System: You are a helpful assistant."));
    assert!(formatted.contains("Q: Hello"));
    assert!(formatted.contains("\nA:"));
}

// ---------------------------------------------------------------------------
// bitnet-engine-core
// ---------------------------------------------------------------------------

#[test]
fn session_config_serde_round_trip() {
    let original = SessionConfig {
        model_path: "/models/bitnet.gguf".to_string(),
        tokenizer_path: "/models/tokenizer.json".to_string(),
        backend: "cpu".to_string(),
        max_context: 4096,
        seed: Some(42),
    };

    let json = serde_json::to_string(&original).expect("serialize SessionConfig");
    let restored: SessionConfig = serde_json::from_str(&json).expect("deserialize SessionConfig");

    assert_eq!(restored.model_path, original.model_path);
    assert_eq!(restored.tokenizer_path, original.tokenizer_path);
    assert_eq!(restored.backend, original.backend);
    assert_eq!(restored.max_context, original.max_context);
    assert_eq!(restored.seed, original.seed);
}

#[test]
fn session_config_default_values() {
    let cfg = SessionConfig::default();
    assert_eq!(cfg.backend, "cpu");
    assert_eq!(cfg.max_context, 2048);
    assert!(cfg.seed.is_none());
}
