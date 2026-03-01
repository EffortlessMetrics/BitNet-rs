//! Edge-case tests for inference configuration types: GenerationConfig,
//! InferenceConfig, and StreamingConfig.
//!
//! Tests cover: defaults, presets (greedy/creative/balanced/cpu/gpu/memory/
//! low_latency/high_throughput), builder chaining, validation (valid/invalid
//! boundaries), serde roundtrip, stop token management, and debug formatting.

use bitnet_inference::StreamingConfig;
use bitnet_inference::config::{GenerationConfig, InferenceConfig};

// ===========================================================================
// GenerationConfig — defaults & presets
// ===========================================================================

#[test]
fn gen_config_default_values() {
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

#[test]
fn gen_config_greedy_preset() {
    let cfg = GenerationConfig::greedy();
    assert!((cfg.temperature - 0.0).abs() < f32::EPSILON);
    assert_eq!(cfg.top_k, 1);
    assert!((cfg.top_p - 1.0).abs() < f32::EPSILON);
}

#[test]
fn gen_config_creative_preset() {
    let cfg = GenerationConfig::creative();
    assert!((cfg.temperature - 0.9).abs() < f32::EPSILON);
    assert_eq!(cfg.top_k, 100);
    assert!((cfg.top_p - 0.95).abs() < f32::EPSILON);
    assert!((cfg.repetition_penalty - 1.1).abs() < f32::EPSILON);
}

#[test]
fn gen_config_balanced_preset() {
    let cfg = GenerationConfig::balanced();
    assert!((cfg.temperature - 0.7).abs() < f32::EPSILON);
    assert_eq!(cfg.top_k, 50);
    assert!((cfg.top_p - 0.9).abs() < f32::EPSILON);
    assert!((cfg.repetition_penalty - 1.05).abs() < f32::EPSILON);
}

// ===========================================================================
// GenerationConfig — builder chaining
// ===========================================================================

#[test]
fn gen_config_builder_chain() {
    let cfg = GenerationConfig::default()
        .with_max_tokens(256)
        .with_temperature(0.5)
        .with_top_k(40)
        .with_top_p(0.85)
        .with_repetition_penalty(1.2)
        .with_seed(42)
        .with_stop_string_window(128)
        .with_skip_special_tokens(false)
        .with_add_bos(true)
        .with_eos_token_id(Some(2))
        .with_logits_tap_steps(5)
        .with_logits_topk(20);

    assert_eq!(cfg.max_new_tokens, 256);
    assert!((cfg.temperature - 0.5).abs() < f32::EPSILON);
    assert_eq!(cfg.top_k, 40);
    assert!((cfg.top_p - 0.85).abs() < f32::EPSILON);
    assert!((cfg.repetition_penalty - 1.2).abs() < f32::EPSILON);
    assert_eq!(cfg.seed, Some(42));
    assert_eq!(cfg.stop_string_window, 128);
    assert!(!cfg.skip_special_tokens);
    assert!(cfg.add_bos);
    assert_eq!(cfg.eos_token_id, Some(2));
    assert_eq!(cfg.logits_tap_steps, 5);
    assert_eq!(cfg.logits_topk, 20);
}

#[test]
fn gen_config_with_stop_sequence() {
    let cfg = GenerationConfig::default()
        .with_stop_sequence("</s>".into())
        .with_stop_sequence("\n\n".into());
    assert_eq!(cfg.stop_sequences.len(), 2);
    assert_eq!(cfg.stop_sequences[0], "</s>");
    assert_eq!(cfg.stop_sequences[1], "\n\n");
}

#[test]
fn gen_config_with_stop_sequences_batch() {
    let cfg =
        GenerationConfig::default().with_stop_sequences(vec!["<|end|>".into(), "<|eot|>".into()]);
    assert_eq!(cfg.stop_sequences.len(), 2);
}

// ===========================================================================
// GenerationConfig — stop token management
// ===========================================================================

#[test]
fn gen_config_stop_token_ids_builder() {
    let cfg = GenerationConfig::default().with_stop_token_ids(vec![128009, 128001]);
    assert!(cfg.is_stop_token(128009));
    assert!(cfg.is_stop_token(128001));
    assert!(!cfg.is_stop_token(999));
}

#[test]
fn gen_config_stop_token_id_single() {
    let cfg = GenerationConfig::default().with_stop_token_id(42).with_stop_token_id(99);
    assert!(cfg.is_stop_token(42));
    assert!(cfg.is_stop_token(99));
    assert!(!cfg.is_stop_token(0));
}

#[test]
fn gen_config_stop_token_empty() {
    let cfg = GenerationConfig::default();
    assert!(!cfg.is_stop_token(0));
    assert!(!cfg.is_stop_token(u32::MAX));
}

#[test]
fn gen_config_rebuild_stop_token_set() {
    let mut cfg = GenerationConfig::default();
    cfg.stop_token_ids = vec![100, 200];
    // Before rebuild, set is out of sync
    assert!(!cfg.is_stop_token(100));
    cfg.rebuild_stop_token_set();
    assert!(cfg.is_stop_token(100));
    assert!(cfg.is_stop_token(200));
}

// ===========================================================================
// GenerationConfig — validation
// ===========================================================================

#[test]
fn gen_config_validate_default_ok() {
    assert!(GenerationConfig::default().validate().is_ok());
}

#[test]
fn gen_config_validate_greedy_ok() {
    assert!(GenerationConfig::greedy().validate().is_ok());
}

#[test]
fn gen_config_validate_creative_ok() {
    assert!(GenerationConfig::creative().validate().is_ok());
}

#[test]
fn gen_config_validate_zero_max_tokens() {
    let cfg = GenerationConfig::default().with_max_tokens(0);
    let result = cfg.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("max_new_tokens"));
}

#[test]
fn gen_config_validate_negative_temperature() {
    let cfg = GenerationConfig::default().with_temperature(-0.1);
    assert!(cfg.validate().is_err());
}

#[test]
fn gen_config_validate_zero_temperature_ok() {
    // Zero temperature = deterministic, should be valid
    let cfg = GenerationConfig::default().with_temperature(0.0);
    assert!(cfg.validate().is_ok());
}

#[test]
fn gen_config_validate_top_p_zero() {
    let cfg = GenerationConfig::default().with_top_p(0.0);
    assert!(cfg.validate().is_err());
}

#[test]
fn gen_config_validate_top_p_one_ok() {
    let cfg = GenerationConfig::default().with_top_p(1.0);
    assert!(cfg.validate().is_ok());
}

#[test]
fn gen_config_validate_top_p_over_one() {
    let cfg = GenerationConfig::default().with_top_p(1.01);
    assert!(cfg.validate().is_err());
}

#[test]
fn gen_config_validate_top_p_negative() {
    let cfg = GenerationConfig::default().with_top_p(-0.5);
    assert!(cfg.validate().is_err());
}

#[test]
fn gen_config_validate_repetition_penalty_zero() {
    let cfg = GenerationConfig::default().with_repetition_penalty(0.0);
    assert!(cfg.validate().is_err());
}

#[test]
fn gen_config_validate_repetition_penalty_negative() {
    let cfg = GenerationConfig::default().with_repetition_penalty(-1.0);
    assert!(cfg.validate().is_err());
}

#[test]
fn gen_config_validate_repetition_penalty_small_positive_ok() {
    let cfg = GenerationConfig::default().with_repetition_penalty(0.01);
    assert!(cfg.validate().is_ok());
}

// ===========================================================================
// GenerationConfig — serde
// ===========================================================================

#[test]
fn gen_config_serde_roundtrip() {
    let cfg = GenerationConfig::default()
        .with_max_tokens(50)
        .with_seed(42)
        .with_stop_token_ids(vec![128009]);
    let json = serde_json::to_string(&cfg).unwrap();
    let mut cfg2: GenerationConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg2.max_new_tokens, 50);
    assert_eq!(cfg2.seed, Some(42));
    assert_eq!(cfg2.stop_token_ids, vec![128009]);
    // stop_token_ids_set is skipped in serde, so rebuild
    assert!(!cfg2.is_stop_token(128009));
    cfg2.rebuild_stop_token_set();
    assert!(cfg2.is_stop_token(128009));
}

#[test]
fn gen_config_clone() {
    let cfg = GenerationConfig::default().with_seed(99);
    let cloned = cfg.clone();
    assert_eq!(cloned.seed, Some(99));
}

#[test]
fn gen_config_debug_output() {
    let cfg = GenerationConfig::default();
    let dbg = format!("{cfg:?}");
    assert!(dbg.contains("GenerationConfig"));
    assert!(dbg.contains("max_new_tokens"));
    assert!(dbg.contains("temperature"));
}

// ===========================================================================
// GenerationConfig — logits callback
// ===========================================================================

#[test]
fn gen_config_with_logits_cb() {
    use std::sync::Arc;
    let cfg = GenerationConfig::default()
        .with_logits_cb(Some(Arc::new(|_step: usize, _tokens: Vec<(u32, f32)>, _chosen: u32| {})));
    assert!(cfg.logits_cb.is_some());
}

#[test]
fn gen_config_without_logits_cb() {
    let cfg = GenerationConfig::default().with_logits_cb(None);
    assert!(cfg.logits_cb.is_none());
}

// ===========================================================================
// InferenceConfig — defaults & presets
// ===========================================================================

#[test]
fn inf_config_defaults() {
    let cfg = InferenceConfig::default();
    assert_eq!(cfg.max_context_length, 2048);
    assert!(cfg.num_threads >= 1);
    assert_eq!(cfg.batch_size, 1);
    assert!(!cfg.mixed_precision);
    assert_eq!(cfg.memory_pool_size, 512 * 1024 * 1024);
}

#[test]
fn inf_config_cpu_optimized() {
    let cfg = InferenceConfig::cpu_optimized();
    assert!(cfg.num_threads >= 1);
    assert!(!cfg.mixed_precision);
    assert_eq!(cfg.batch_size, 1);
}

#[test]
fn inf_config_gpu_optimized() {
    let cfg = InferenceConfig::gpu_optimized();
    assert!(cfg.mixed_precision);
    assert_eq!(cfg.batch_size, 4);
    assert_eq!(cfg.memory_pool_size, 1024 * 1024 * 1024);
}

#[test]
fn inf_config_memory_efficient() {
    let cfg = InferenceConfig::memory_efficient();
    assert_eq!(cfg.max_context_length, 1024);
    assert_eq!(cfg.batch_size, 1);
    assert_eq!(cfg.memory_pool_size, 256 * 1024 * 1024);
}

// ===========================================================================
// InferenceConfig — builders
// ===========================================================================

#[test]
fn inf_config_builder_chain() {
    let cfg = InferenceConfig::default()
        .with_threads(8)
        .with_batch_size(16)
        .with_mixed_precision(true)
        .with_memory_pool_size(2 * 1024 * 1024 * 1024);
    assert_eq!(cfg.num_threads, 8);
    assert_eq!(cfg.batch_size, 16);
    assert!(cfg.mixed_precision);
    assert_eq!(cfg.memory_pool_size, 2 * 1024 * 1024 * 1024);
}

// ===========================================================================
// InferenceConfig — validation
// ===========================================================================

#[test]
fn inf_config_validate_default_ok() {
    assert!(InferenceConfig::default().validate().is_ok());
}

#[test]
fn inf_config_validate_zero_context() {
    let mut cfg = InferenceConfig::default();
    cfg.max_context_length = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn inf_config_validate_zero_threads() {
    let cfg = InferenceConfig::default().with_threads(0);
    assert!(cfg.validate().is_err());
}

#[test]
fn inf_config_validate_zero_batch() {
    let cfg = InferenceConfig::default().with_batch_size(0);
    assert!(cfg.validate().is_err());
}

#[test]
fn inf_config_validate_zero_memory_pool() {
    let cfg = InferenceConfig::default().with_memory_pool_size(0);
    assert!(cfg.validate().is_err());
}

// ===========================================================================
// InferenceConfig — serde
// ===========================================================================

#[test]
fn inf_config_serde_roundtrip() {
    let cfg = InferenceConfig::default().with_threads(4).with_batch_size(2);
    let json = serde_json::to_string(&cfg).unwrap();
    let cfg2: InferenceConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg2.num_threads, 4);
    assert_eq!(cfg2.batch_size, 2);
}

#[test]
fn inf_config_clone() {
    let cfg = InferenceConfig::gpu_optimized();
    let cloned = cfg.clone();
    assert_eq!(cloned.batch_size, cfg.batch_size);
}

#[test]
fn inf_config_debug() {
    let cfg = InferenceConfig::default();
    let dbg = format!("{cfg:?}");
    assert!(dbg.contains("InferenceConfig"));
}

// ===========================================================================
// StreamingConfig — defaults & presets
// ===========================================================================

#[test]
fn stream_config_defaults() {
    let cfg = StreamingConfig::default();
    assert_eq!(cfg.buffer_size, 10);
    assert_eq!(cfg.flush_interval_ms, 50);
    assert_eq!(cfg.max_retries, 3);
    assert_eq!(cfg.token_timeout_ms, 5000);
    assert!(cfg.cancellable);
}

#[test]
fn stream_config_low_latency() {
    let cfg = StreamingConfig::low_latency();
    assert_eq!(cfg.buffer_size, 1);
    assert_eq!(cfg.flush_interval_ms, 10);
    assert_eq!(cfg.max_retries, 1);
    assert_eq!(cfg.token_timeout_ms, 1000);
    assert!(cfg.cancellable);
}

#[test]
fn stream_config_high_throughput() {
    let cfg = StreamingConfig::high_throughput();
    assert_eq!(cfg.buffer_size, 50);
    assert_eq!(cfg.flush_interval_ms, 200);
    assert_eq!(cfg.max_retries, 5);
    assert_eq!(cfg.token_timeout_ms, 10000);
    assert!(!cfg.cancellable);
}

// ===========================================================================
// StreamingConfig — validation
// ===========================================================================

#[test]
fn stream_config_validate_default_ok() {
    assert!(StreamingConfig::default().validate().is_ok());
}

#[test]
fn stream_config_validate_low_latency_ok() {
    assert!(StreamingConfig::low_latency().validate().is_ok());
}

#[test]
fn stream_config_validate_high_throughput_ok() {
    assert!(StreamingConfig::high_throughput().validate().is_ok());
}

#[test]
fn stream_config_validate_zero_buffer() {
    let mut cfg = StreamingConfig::default();
    cfg.buffer_size = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn stream_config_validate_zero_flush_interval() {
    let mut cfg = StreamingConfig::default();
    cfg.flush_interval_ms = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn stream_config_validate_zero_token_timeout() {
    let mut cfg = StreamingConfig::default();
    cfg.token_timeout_ms = 0;
    assert!(cfg.validate().is_err());
}

// ===========================================================================
// StreamingConfig — clone & debug
// ===========================================================================

#[test]
fn stream_config_clone() {
    let cfg = StreamingConfig::low_latency();
    let cloned = cfg.clone();
    assert_eq!(cloned.buffer_size, 1);
}

#[test]
fn stream_config_debug() {
    let cfg = StreamingConfig::default();
    let dbg = format!("{cfg:?}");
    assert!(dbg.contains("StreamingConfig"));
    assert!(dbg.contains("buffer_size"));
}
