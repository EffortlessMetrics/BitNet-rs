//! Edge-case tests for `bitnet-inference` config module:
//! GenerationConfig, InferenceConfig, builders, validation, serde, stop tokens.

use bitnet_inference::config::{GenerationConfig, InferenceConfig};

// ---------------------------------------------------------------------------
// InferenceConfig defaults
// ---------------------------------------------------------------------------

#[test]
fn inference_config_defaults() {
    let cfg = InferenceConfig::default();
    assert_eq!(cfg.max_context_length, 2048);
    assert!(cfg.num_threads > 0);
    assert_eq!(cfg.batch_size, 1);
    assert!(!cfg.mixed_precision);
    assert_eq!(cfg.memory_pool_size, 512 * 1024 * 1024);
}

#[test]
fn inference_config_serde_roundtrip() {
    let cfg = InferenceConfig::default();
    let json = serde_json::to_string(&cfg).unwrap();
    let cfg2: InferenceConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg2.max_context_length, cfg.max_context_length);
    assert_eq!(cfg2.batch_size, cfg.batch_size);
}

// ---------------------------------------------------------------------------
// GenerationConfig defaults
// ---------------------------------------------------------------------------

#[test]
fn generation_config_default_values() {
    let cfg = GenerationConfig::default();
    assert_eq!(cfg.max_new_tokens, 100);
    assert!((cfg.temperature - 0.7).abs() < f32::EPSILON);
    assert_eq!(cfg.top_k, 50);
    assert!((cfg.top_p - 0.9).abs() < f32::EPSILON);
    assert!((cfg.repetition_penalty - 1.0).abs() < f32::EPSILON);
    assert!(cfg.stop_sequences.is_empty());
    assert!(cfg.stop_token_ids.is_empty());
    assert_eq!(cfg.stop_string_window, 64);
    assert!(cfg.seed.is_none());
    assert!(cfg.skip_special_tokens);
    assert!(cfg.eos_token_id.is_none());
    assert_eq!(cfg.logits_tap_steps, 0);
    assert_eq!(cfg.logits_topk, 10);
    assert!(cfg.logits_cb.is_none());
    assert!(!cfg.add_bos);
}

// ---------------------------------------------------------------------------
// Named constructors
// ---------------------------------------------------------------------------

#[test]
fn greedy_config() {
    let cfg = GenerationConfig::greedy();
    assert!((cfg.temperature - 0.0).abs() < f32::EPSILON);
    assert_eq!(cfg.top_k, 1);
    assert!((cfg.top_p - 1.0).abs() < f32::EPSILON);
}

#[test]
fn creative_config() {
    let cfg = GenerationConfig::creative();
    assert!(cfg.temperature > 0.5);
    assert!(cfg.top_k > 50);
    assert!(cfg.repetition_penalty > 1.0);
}

#[test]
fn balanced_config() {
    let cfg = GenerationConfig::balanced();
    assert!((cfg.temperature - 0.7).abs() < f32::EPSILON);
    assert_eq!(cfg.top_k, 50);
    assert!((cfg.top_p - 0.9).abs() < f32::EPSILON);
    assert!(cfg.repetition_penalty > 1.0);
}

// ---------------------------------------------------------------------------
// Builder fluent API
// ---------------------------------------------------------------------------

#[test]
fn builder_with_max_tokens() {
    let cfg = GenerationConfig::default().with_max_tokens(256);
    assert_eq!(cfg.max_new_tokens, 256);
}

#[test]
fn builder_with_temperature() {
    let cfg = GenerationConfig::default().with_temperature(1.5);
    assert!((cfg.temperature - 1.5).abs() < f32::EPSILON);
}

#[test]
fn builder_with_top_k() {
    let cfg = GenerationConfig::default().with_top_k(10);
    assert_eq!(cfg.top_k, 10);
}

#[test]
fn builder_with_top_p() {
    let cfg = GenerationConfig::default().with_top_p(0.5);
    assert!((cfg.top_p - 0.5).abs() < f32::EPSILON);
}

#[test]
fn builder_with_repetition_penalty() {
    let cfg = GenerationConfig::default().with_repetition_penalty(1.2);
    assert!((cfg.repetition_penalty - 1.2).abs() < f32::EPSILON);
}

#[test]
fn builder_with_seed() {
    let cfg = GenerationConfig::default().with_seed(42);
    assert_eq!(cfg.seed, Some(42));
}

#[test]
fn builder_with_stop_sequence() {
    let cfg = GenerationConfig::default().with_stop_sequence("</s>".to_string());
    assert_eq!(cfg.stop_sequences, vec!["</s>"]);
}

#[test]
fn builder_chained_stop_sequences() {
    let cfg = GenerationConfig::default()
        .with_stop_sequence("</s>".to_string())
        .with_stop_sequence("\n\n".to_string());
    assert_eq!(cfg.stop_sequences.len(), 2);
}

#[test]
fn builder_with_stop_sequences_replaces() {
    let cfg = GenerationConfig::default()
        .with_stop_sequence("old".to_string())
        .with_stop_sequences(vec!["new1".to_string(), "new2".to_string()]);
    assert_eq!(cfg.stop_sequences.len(), 2);
    assert!(!cfg.stop_sequences.contains(&"old".to_string()));
}

#[test]
fn builder_with_stop_string_window() {
    let cfg = GenerationConfig::default().with_stop_string_window(128);
    assert_eq!(cfg.stop_string_window, 128);
}

#[test]
fn builder_with_skip_special_tokens() {
    let cfg = GenerationConfig::default().with_skip_special_tokens(false);
    assert!(!cfg.skip_special_tokens);
}

#[test]
fn builder_with_add_bos() {
    let cfg = GenerationConfig::default().with_add_bos(true);
    assert!(cfg.add_bos);
}

#[test]
fn builder_with_eos_token_id() {
    let cfg = GenerationConfig::default().with_eos_token_id(Some(2));
    assert_eq!(cfg.eos_token_id, Some(2));
}

#[test]
fn builder_with_logits_tap() {
    let cfg = GenerationConfig::default().with_logits_tap_steps(5).with_logits_topk(20);
    assert_eq!(cfg.logits_tap_steps, 5);
    assert_eq!(cfg.logits_topk, 20);
}

// ---------------------------------------------------------------------------
// Stop token IDs
// ---------------------------------------------------------------------------

#[test]
fn stop_token_ids_via_builder() {
    let cfg = GenerationConfig::default().with_stop_token_ids(vec![128009, 128001]);
    assert!(cfg.is_stop_token(128009));
    assert!(cfg.is_stop_token(128001));
    assert!(!cfg.is_stop_token(999));
}

#[test]
fn stop_token_id_single() {
    let cfg = GenerationConfig::default().with_stop_token_id(128009);
    assert!(cfg.is_stop_token(128009));
    assert!(!cfg.is_stop_token(0));
}

#[test]
fn stop_token_id_chained() {
    let cfg = GenerationConfig::default().with_stop_token_id(1).with_stop_token_id(2);
    assert!(cfg.is_stop_token(1));
    assert!(cfg.is_stop_token(2));
    assert!(!cfg.is_stop_token(3));
}

#[test]
fn stop_token_direct_modify_needs_rebuild() {
    let mut cfg = GenerationConfig::default();
    cfg.stop_token_ids = vec![128009];
    // Without rebuild, HashSet is stale
    assert!(!cfg.is_stop_token(128009));
    cfg.rebuild_stop_token_set();
    assert!(cfg.is_stop_token(128009));
}

#[test]
fn no_stop_tokens_by_default() {
    let cfg = GenerationConfig::default();
    assert!(!cfg.is_stop_token(0));
    assert!(!cfg.is_stop_token(u32::MAX));
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

#[test]
fn validate_default_passes() {
    assert!(GenerationConfig::default().validate().is_ok());
}

#[test]
fn validate_greedy_passes() {
    assert!(GenerationConfig::greedy().validate().is_ok());
}

#[test]
fn validate_creative_passes() {
    assert!(GenerationConfig::creative().validate().is_ok());
}

#[test]
fn validate_rejects_zero_max_tokens() {
    let cfg = GenerationConfig::default().with_max_tokens(0);
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_negative_temperature() {
    let cfg = GenerationConfig::default().with_temperature(-0.1);
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_accepts_zero_temperature() {
    let cfg = GenerationConfig::default().with_temperature(0.0);
    assert!(cfg.validate().is_ok());
}

#[test]
fn validate_rejects_zero_top_p() {
    let cfg = GenerationConfig::default().with_top_p(0.0);
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_top_p_above_1() {
    let cfg = GenerationConfig::default().with_top_p(1.1);
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_accepts_top_p_exactly_1() {
    let cfg = GenerationConfig::default().with_top_p(1.0);
    assert!(cfg.validate().is_ok());
}

#[test]
fn validate_rejects_zero_repetition_penalty() {
    let cfg = GenerationConfig::default().with_repetition_penalty(0.0);
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_negative_repetition_penalty() {
    let cfg = GenerationConfig::default().with_repetition_penalty(-1.0);
    assert!(cfg.validate().is_err());
}

// ---------------------------------------------------------------------------
// Serde roundtrip
// ---------------------------------------------------------------------------

#[test]
fn generation_config_serde_roundtrip() {
    let cfg = GenerationConfig::greedy()
        .with_max_tokens(32)
        .with_seed(42)
        .with_stop_sequence("</s>".to_string())
        .with_stop_token_ids(vec![128009])
        .with_eos_token_id(Some(2));
    let json = serde_json::to_string(&cfg).unwrap();
    let mut cfg2: GenerationConfig = serde_json::from_str(&json).unwrap();
    // After deserialization, stop token set needs rebuild
    cfg2.rebuild_stop_token_set();
    assert_eq!(cfg2.max_new_tokens, 32);
    assert_eq!(cfg2.seed, Some(42));
    assert_eq!(cfg2.stop_sequences, vec!["</s>"]);
    assert!(cfg2.is_stop_token(128009));
    assert_eq!(cfg2.eos_token_id, Some(2));
}

#[test]
fn generation_config_serde_skips_callback() {
    use std::sync::Arc;
    let cfg = GenerationConfig::default().with_logits_cb(Some(Arc::new(|_, _, _| {})));
    assert!(cfg.logits_cb.is_some());
    let json = serde_json::to_string(&cfg).unwrap();
    let cfg2: GenerationConfig = serde_json::from_str(&json).unwrap();
    // logits_cb is #[serde(skip)]
    assert!(cfg2.logits_cb.is_none());
}

// ---------------------------------------------------------------------------
// Clone semantics
// ---------------------------------------------------------------------------

#[test]
fn generation_config_clone() {
    let cfg = GenerationConfig::greedy().with_seed(42).with_max_tokens(16);
    let clone = cfg.clone();
    assert_eq!(clone.max_new_tokens, 16);
    assert_eq!(clone.seed, Some(42));
}

// ---------------------------------------------------------------------------
// Chained builder
// ---------------------------------------------------------------------------

#[test]
fn full_builder_chain() {
    let cfg = GenerationConfig::greedy()
        .with_max_tokens(64)
        .with_temperature(0.8)
        .with_top_k(40)
        .with_top_p(0.95)
        .with_repetition_penalty(1.1)
        .with_seed(123)
        .with_stop_sequence("###".to_string())
        .with_stop_token_ids(vec![1, 2])
        .with_stop_string_window(32)
        .with_skip_special_tokens(false)
        .with_add_bos(true)
        .with_eos_token_id(Some(2))
        .with_logits_tap_steps(3)
        .with_logits_topk(5);

    assert_eq!(cfg.max_new_tokens, 64);
    assert!((cfg.temperature - 0.8).abs() < f32::EPSILON);
    assert_eq!(cfg.top_k, 40);
    assert_eq!(cfg.seed, Some(123));
    assert!(cfg.is_stop_token(1));
    assert!(cfg.is_stop_token(2));
    assert!(!cfg.skip_special_tokens);
    assert!(cfg.add_bos);
    assert_eq!(cfg.logits_tap_steps, 3);
    assert!(cfg.validate().is_ok());
}
