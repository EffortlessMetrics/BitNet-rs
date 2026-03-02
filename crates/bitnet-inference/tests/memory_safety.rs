//! Memory safety and allocation sanity tests for bitnet-inference.
//!
//! These tests verify that edge-case inputs do not cause panics, overflows,
//! division by zero, or unbounded memory allocation. All tests are CPU-only,
//! deterministic, and require no model files.

use bitnet_inference::batch::{BatchConfig, BatchRequest, BatchResult, BatchScheduler};
use bitnet_inference::config::{GenerationConfig, InferenceConfig};
use bitnet_sampling::{SamplingConfig, SamplingStrategy};
use std::time::Duration;

// ============================================================================
// GenerationConfig — extreme values
// ============================================================================

#[test]
fn generation_config_u32_max_tokens_does_not_panic() {
    let config = GenerationConfig::greedy().with_max_tokens(u32::MAX);
    assert_eq!(config.max_new_tokens, u32::MAX);
    // Validation should still succeed (the value is > 0)
    assert!(config.validate().is_ok());
}

#[test]
fn generation_config_extreme_temperature_does_not_panic() {
    let config = GenerationConfig::default().with_temperature(f32::MAX);
    assert_eq!(config.temperature, f32::MAX);
    assert!(config.validate().is_ok());
}

#[test]
fn generation_config_zero_temperature_is_valid() {
    let config = GenerationConfig::greedy();
    assert_eq!(config.temperature, 0.0);
    assert!(config.validate().is_ok());
}

#[test]
fn generation_config_negative_temperature_rejected() {
    let config = GenerationConfig::default().with_temperature(-1.0);
    assert!(config.validate().is_err());
}

#[test]
fn generation_config_zero_max_tokens_rejected() {
    let config = GenerationConfig::default().with_max_tokens(0);
    assert!(config.validate().is_err());
}

#[test]
fn generation_config_top_p_boundary_values() {
    // top_p = 1.0 is valid (disabled)
    let config = GenerationConfig::default().with_top_p(1.0);
    assert!(config.validate().is_ok());

    // top_p = 0.0 is invalid
    let config = GenerationConfig::default().with_top_p(0.0);
    assert!(config.validate().is_err());

    // top_p > 1.0 is invalid
    let config = GenerationConfig::default().with_top_p(1.001);
    assert!(config.validate().is_err());
}

#[test]
fn generation_config_zero_repetition_penalty_rejected() {
    let config = GenerationConfig::default().with_repetition_penalty(0.0);
    assert!(config.validate().is_err());
}

#[test]
fn generation_config_extreme_repetition_penalty_does_not_panic() {
    let config = GenerationConfig::default().with_repetition_penalty(f32::MAX);
    assert!(config.validate().is_ok());
}

// ============================================================================
// InferenceConfig — extreme values
// ============================================================================

#[test]
fn inference_config_zero_batch_size_rejected() {
    let config = InferenceConfig::default().with_batch_size(0);
    assert!(config.validate().is_err());
}

#[test]
fn inference_config_large_batch_size_does_not_overflow() {
    let config = InferenceConfig::default().with_batch_size(usize::MAX);
    assert!(config.validate().is_ok());
    assert_eq!(config.batch_size, usize::MAX);
}

#[test]
fn inference_config_zero_threads_rejected() {
    let config = InferenceConfig::default().with_threads(0);
    assert!(config.validate().is_err());
}

#[test]
fn inference_config_zero_memory_pool_rejected() {
    let config = InferenceConfig::default().with_memory_pool_size(0);
    assert!(config.validate().is_err());
}

// ============================================================================
// SamplingStrategy — edge cases
// ============================================================================

#[test]
fn sampling_empty_logits_returns_error() {
    let config = SamplingConfig { temperature: 0.7, seed: Some(42), ..Default::default() };
    let mut strategy = SamplingStrategy::new(config);
    let result = strategy.sample(&[], &[]);
    assert!(result.is_err(), "empty logits must return an error");
}

#[test]
fn sampling_single_token_logits() {
    let config = SamplingConfig { temperature: 0.0, seed: Some(42), ..Default::default() };
    let mut strategy = SamplingStrategy::new(config);
    let token = strategy.sample(&[1.0], &[]).unwrap();
    assert_eq!(token, 0, "single-element logits must return index 0");
}

#[test]
fn sampling_temperature_zero_is_greedy() {
    let config = SamplingConfig { temperature: 0.0, seed: Some(42), ..Default::default() };
    let mut strategy = SamplingStrategy::new(config);
    let logits = vec![0.1, 0.9, 0.3, 0.5];
    let token = strategy.sample(&logits, &[]).unwrap();
    assert_eq!(token, 1, "temperature=0 must select argmax (index 1)");
}

#[test]
fn sampling_negative_temperature_treated_as_greedy_or_safe() {
    // Negative temperature should not cause division by zero or panic.
    // The SamplingStrategy uses temperature directly; at temperature <= 0
    // the greedy path triggers.
    let config = SamplingConfig { temperature: -1.0, seed: Some(42), ..Default::default() };
    let mut strategy = SamplingStrategy::new(config);
    let logits = vec![0.1, 0.9, 0.3];
    // Should not panic — greedy path handles temperature <= 0
    let result = strategy.sample(&logits, &[]);
    assert!(result.is_ok(), "negative temperature must not panic");
}

#[test]
fn sampling_very_large_temperature_does_not_panic() {
    let config = SamplingConfig {
        temperature: 1e10,
        seed: Some(42),
        top_k: 0,
        top_p: 1.0,
        ..Default::default()
    };
    let mut strategy = SamplingStrategy::new(config);
    let logits = vec![1.0, 2.0, 3.0, 4.0];
    let token = strategy.sample(&logits, &[]).unwrap();
    assert!((token as usize) < logits.len(), "sampled token must be within vocab range");
}

#[test]
fn sampling_all_equal_logits() {
    let config = SamplingConfig { temperature: 0.0, seed: Some(42), ..Default::default() };
    let mut strategy = SamplingStrategy::new(config);
    let logits = vec![1.0; 100];
    let token = strategy.sample(&logits, &[]).unwrap();
    // Greedy with ties: should pick lowest index (llama.cpp compat)
    assert_eq!(token, 0, "greedy on equal logits should pick index 0");
}

#[test]
fn sampling_all_negative_infinity_logits() {
    let config = SamplingConfig { temperature: 0.7, seed: Some(42), ..Default::default() };
    let mut strategy = SamplingStrategy::new(config);
    let logits = vec![f32::NEG_INFINITY; 10];
    // Should not panic; softmax of all-NEG_INF may produce NaN but the
    // fallback in sample_from_distribution handles sum <= 0.
    let result = strategy.sample(&logits, &[]);
    assert!(result.is_ok(), "all-NEG_INFINITY logits must not panic");
}

#[test]
fn sampling_all_zero_logits() {
    let config = SamplingConfig { temperature: 0.7, seed: Some(42), ..Default::default() };
    let mut strategy = SamplingStrategy::new(config);
    let logits = vec![0.0; 50];
    let token = strategy.sample(&logits, &[]).unwrap();
    assert!((token as usize) < 50, "sampled token must be within vocab range");
}

#[test]
fn sampling_top_k_larger_than_vocab_is_safe() {
    // top_k = 10_000 on a 5-element vocab should not panic
    let config = SamplingConfig {
        temperature: 0.7,
        top_k: 10_000,
        top_p: 1.0,
        seed: Some(42),
        ..Default::default()
    };
    let mut strategy = SamplingStrategy::new(config);
    let logits = vec![0.1, 0.5, 0.2, 0.1, 0.1];
    let token = strategy.sample(&logits, &[]).unwrap();
    assert!((token as usize) < logits.len(), "top_k > vocab must be clamped safely");
}

#[test]
fn sampling_top_k_zero_disables_filtering() {
    let config = SamplingConfig {
        temperature: 0.7,
        top_k: 0,
        top_p: 1.0,
        seed: Some(42),
        ..Default::default()
    };
    let mut strategy = SamplingStrategy::new(config);
    let logits = vec![1.0, 2.0, 3.0];
    let token = strategy.sample(&logits, &[]).unwrap();
    assert!((token as usize) < logits.len());
}

#[test]
fn sampling_top_k_one_selects_argmax() {
    let config = SamplingConfig {
        temperature: 0.7,
        top_k: 1,
        top_p: 1.0,
        seed: Some(42),
        ..Default::default()
    };
    let mut strategy = SamplingStrategy::new(config);
    let logits = vec![0.1, 0.9, 0.3];
    let token = strategy.sample(&logits, &[]).unwrap();
    assert_eq!(token, 1, "top_k=1 should always select the highest logit");
}

// ============================================================================
// SamplingStrategy — determinism
// ============================================================================

#[test]
fn sampling_deterministic_with_same_seed() {
    let make = || {
        SamplingStrategy::new(SamplingConfig {
            temperature: 0.8,
            seed: Some(42),
            ..Default::default()
        })
    };
    let logits = vec![0.2, 0.5, 0.1, 0.2];

    let mut s1 = make();
    let mut s2 = make();
    let t1 = s1.sample(&logits, &[]).unwrap();
    let t2 = s2.sample(&logits, &[]).unwrap();
    assert_eq!(t1, t2, "same seed must produce identical tokens");
}

// ============================================================================
// Batch — overflow and edge cases
// ============================================================================

#[test]
fn batch_empty_request_schedules_nothing() {
    let batch = BatchRequest::new();
    let scheduler = BatchScheduler::new(BatchConfig::default());
    let ids = scheduler.schedule(&batch);
    assert!(ids.is_empty());
}

#[test]
fn batch_zero_length_prompt() {
    let mut batch = BatchRequest::new();
    let id = batch.add(String::new(), GenerationConfig::greedy());
    assert_eq!(id, 0);

    let scheduler = BatchScheduler::new(BatchConfig::default());
    let ids = scheduler.schedule(&batch);
    assert_eq!(ids, vec![0], "zero-length prompt should still be scheduled");
}

#[test]
fn batch_result_insert_beyond_capacity() {
    use bitnet_inference::batch::SingleResult;

    let mut result = BatchResult::with_capacity(2);
    // Insert at index beyond initial capacity
    result.insert(SingleResult { id: 10, text: "hello".into(), tokens_generated: 1 });
    assert_eq!(result.get(10).unwrap().text, "hello");
    assert_eq!(result.completed_count(), 1);
}

#[test]
fn batch_config_max_total_tokens_zero_rejected() {
    let config = BatchConfig::default().with_max_total_tokens(0);
    assert!(config.validate().is_err());
}

#[test]
#[should_panic(expected = "max_batch_size must be > 0")]
fn batch_config_zero_batch_size_panics() {
    let _ = BatchConfig::new(0, Duration::from_secs(1));
}

#[test]
fn batch_large_max_new_tokens_does_not_overflow_scheduler() {
    let mut batch = BatchRequest::new();
    // Add a request with u32::MAX max_new_tokens — the scheduler uses
    // saturating_add so it must not overflow.
    batch.add("test".into(), GenerationConfig::greedy().with_max_tokens(u32::MAX));
    let config = BatchConfig::new(8, Duration::from_secs(30)).with_max_total_tokens(usize::MAX);
    let scheduler = BatchScheduler::new(config);
    let ids = scheduler.schedule(&batch);
    assert_eq!(ids.len(), 1, "request must be scheduled despite u32::MAX tokens");
}

// ============================================================================
// GenerationConfig — serialization round-trip with extreme values
// ============================================================================

#[test]
fn generation_config_serde_round_trip_extreme() {
    let config = GenerationConfig::greedy()
        .with_max_tokens(u32::MAX)
        .with_temperature(0.0)
        .with_top_k(u32::MAX)
        .with_top_p(1.0)
        .with_repetition_penalty(f32::MAX)
        .with_stop_string_window(usize::MAX);

    let json = serde_json::to_string(&config).expect("serialization must not fail");
    let deserialized: GenerationConfig =
        serde_json::from_str(&json).expect("deserialization must not fail");

    assert_eq!(deserialized.max_new_tokens, u32::MAX);
    assert_eq!(deserialized.temperature, 0.0);
    assert_eq!(deserialized.top_k, u32::MAX);
}

// ============================================================================
// Large vocab — sampling does not allocate unbounded memory
// ============================================================================

#[test]
fn sampling_large_vocab_returns_valid_index() {
    let vocab_size = 128_000; // typical LLM vocab
    let mut logits = vec![0.0f32; vocab_size];
    logits[42_000] = 10.0; // spike at one token

    let config = SamplingConfig { temperature: 0.0, seed: Some(1), ..Default::default() };
    let mut strategy = SamplingStrategy::new(config);
    let token = strategy.sample(&logits, &[]).unwrap();
    assert_eq!(token, 42_000, "greedy must pick the spike token");
}

#[test]
fn sampling_with_heavy_repetition_context() {
    let config = SamplingConfig {
        temperature: 0.7,
        repetition_penalty: 1.5,
        seed: Some(42),
        ..Default::default()
    };
    let mut strategy = SamplingStrategy::new(config);
    let logits = vec![1.0; 100];
    // Context with many repeated tokens
    let context: Vec<u32> = (0..100).flat_map(|i| std::iter::repeat_n(i, 10)).collect();
    let token = strategy.sample(&logits, &context).unwrap();
    assert!((token as usize) < 100);
}

// ============================================================================
// SamplingStrategy — reset clears state
// ============================================================================

#[test]
fn sampling_reset_clears_token_counts() {
    let config = SamplingConfig {
        temperature: 0.0,
        repetition_penalty: 2.0,
        seed: Some(42),
        ..Default::default()
    };
    let mut strategy = SamplingStrategy::new(config);
    let logits = vec![0.5, 0.5, 0.1];

    // Sample once — builds internal state
    let _ = strategy.sample(&logits, &[0, 0, 0]).unwrap();
    strategy.reset();

    // After reset, the strategy should behave as fresh
    let token = strategy.sample(&logits, &[]).unwrap();
    // Greedy on [0.5, 0.5, 0.1] with no context → index 0 (tie-break lowest)
    assert_eq!(token, 0);
}
