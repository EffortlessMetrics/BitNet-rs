//! Edge-case tests for SamplingStrategy state management, StreamingConfig,
//! GenerationStats, and GenerationConfig stop-token mechanics.

use std::sync::atomic::Ordering;

// ---------------------------------------------------------------------------
// generation::sampling  (Candle-based SamplingStrategy)
// ---------------------------------------------------------------------------
use bitnet_inference::generation::sampling::{
    SamplingConfig as GenSamplingConfig, SamplingStrategy as GenSamplingStrategy,
};

// ---------------------------------------------------------------------------
// streaming types
// ---------------------------------------------------------------------------
use bitnet_inference::streaming::{GenerationStats, StreamingConfig};

// ---------------------------------------------------------------------------
// config types
// ---------------------------------------------------------------------------
use bitnet_inference::config::GenerationConfig;

// ===== SamplingStrategy: track_token =====

#[test]
fn track_token_increments_count() {
    let mut s = GenSamplingStrategy::new(GenSamplingConfig::default());
    s.track_token(42);
    s.track_token(42);
    s.track_token(42);
    // After tracking, penalty should still reflect the base config value
    assert!(
        (s.effective_repetition_penalty() - GenSamplingConfig::default().repetition_penalty).abs()
            < f32::EPSILON
    );
}

#[test]
fn track_token_distinct_tokens() {
    let mut s = GenSamplingStrategy::new(GenSamplingConfig::default());
    for id in 0..50 {
        s.track_token(id);
    }
    // Effective penalty unchanged by mere tracking
    assert!(
        (s.effective_repetition_penalty() - GenSamplingConfig::default().repetition_penalty).abs()
            < f32::EPSILON
    );
}

#[test]
fn track_token_clears_at_capacity_boundary() {
    let mut s = GenSamplingStrategy::new(GenSamplingConfig::default());
    // Fill to exactly 1000 unique tokens
    for id in 0..1000 {
        s.track_token(id);
    }
    // The 1001st distinct token triggers the `> 1000` clear
    s.track_token(9999);
    // After clearing, token 0 no longer tracked — penalty state resets
    // (indirectly tested: no panic, effective penalty still valid)
    assert!(s.effective_repetition_penalty() > 0.0);
}

// ===== SamplingStrategy: increase / reset repetition penalty =====

#[test]
fn increase_repetition_penalty_applies_multiplier() {
    let cfg = GenSamplingConfig { repetition_penalty: 1.0, ..Default::default() };
    let mut s = GenSamplingStrategy::new(cfg);
    s.increase_repetition_penalty();
    assert!((s.effective_repetition_penalty() - 1.1).abs() < f32::EPSILON);
}

#[test]
fn increase_repetition_penalty_caps_at_two() {
    let cfg = GenSamplingConfig { repetition_penalty: 1.9, ..Default::default() };
    let mut s = GenSamplingStrategy::new(cfg);
    // 1.9 * 1.1 = 2.09 → capped to 2.0
    s.increase_repetition_penalty();
    assert!((s.effective_repetition_penalty() - 2.0).abs() < f32::EPSILON);
}

#[test]
fn increase_repetition_penalty_successive_calls() {
    let cfg = GenSamplingConfig { repetition_penalty: 1.0, ..Default::default() };
    let mut s = GenSamplingStrategy::new(cfg);
    for _ in 0..20 {
        s.increase_repetition_penalty();
    }
    // Must never exceed cap
    assert!(s.effective_repetition_penalty() <= 2.0);
    assert!(s.effective_repetition_penalty() >= 1.0);
}

#[test]
fn reset_repetition_penalty_restores_config_value() {
    let cfg = GenSamplingConfig { repetition_penalty: 1.3, ..Default::default() };
    let mut s = GenSamplingStrategy::new(cfg);
    s.increase_repetition_penalty();
    s.increase_repetition_penalty();
    assert!(s.effective_repetition_penalty() > 1.3);
    s.reset_repetition_penalty();
    assert!((s.effective_repetition_penalty() - 1.3).abs() < f32::EPSILON);
}

#[test]
fn reset_clears_repetition_counts() {
    let mut s = GenSamplingStrategy::new(GenSamplingConfig::default());
    s.track_token(1);
    s.track_token(1);
    s.reset_repetition_penalty();
    // Indirectly: no panic and penalty is back to base
    assert!(
        (s.effective_repetition_penalty() - GenSamplingConfig::default().repetition_penalty).abs()
            < f32::EPSILON
    );
}

// ===== SamplingStrategy: effective_temperature =====

#[test]
fn effective_temperature_matches_config() {
    let cfg = GenSamplingConfig { temperature: 0.42, ..Default::default() };
    let s = GenSamplingStrategy::new(cfg);
    assert!((s.effective_temperature() - 0.42).abs() < f32::EPSILON);
}

#[test]
fn effective_temperature_zero() {
    let cfg = GenSamplingConfig { temperature: 0.0, ..Default::default() };
    let s = GenSamplingStrategy::new(cfg);
    assert!((s.effective_temperature() - 0.0).abs() < f32::EPSILON);
}

// ===== SamplingStrategy: update_config =====

#[test]
fn update_config_replaces_penalty_and_temperature() {
    let mut s = GenSamplingStrategy::new(GenSamplingConfig::default());
    s.increase_repetition_penalty();

    let new_cfg =
        GenSamplingConfig { temperature: 0.5, repetition_penalty: 1.7, ..Default::default() };
    s.update_config(new_cfg);

    assert!((s.effective_temperature() - 0.5).abs() < f32::EPSILON);
    assert!((s.effective_repetition_penalty() - 1.7).abs() < f32::EPSILON);
}

// ===== SamplingStrategy: preset constructors =====

#[test]
fn deterministic_preset_disables_sampling() {
    let s = GenSamplingStrategy::deterministic();
    assert!((s.effective_temperature() - 1.0).abs() < f32::EPSILON);
    assert!((s.effective_repetition_penalty() - 1.0).abs() < f32::EPSILON);
}

#[test]
fn creative_preset_values() {
    let s = GenSamplingStrategy::creative();
    assert!((s.effective_temperature() - 1.2).abs() < f32::EPSILON);
    assert!((s.effective_repetition_penalty() - 1.2).abs() < f32::EPSILON);
}

#[test]
fn balanced_preset_values() {
    let s = GenSamplingStrategy::balanced();
    assert!((s.effective_temperature() - 0.8).abs() < f32::EPSILON);
    assert!((s.effective_repetition_penalty() - 1.1).abs() < f32::EPSILON);
}

#[test]
fn conservative_preset_values() {
    let s = GenSamplingStrategy::conservative();
    assert!((s.effective_temperature() - 0.3).abs() < f32::EPSILON);
    assert!((s.effective_repetition_penalty() - 1.05).abs() < f32::EPSILON);
}

// ===== StreamingConfig: validate =====

#[test]
fn streaming_config_default_is_valid() {
    assert!(StreamingConfig::default().validate().is_ok());
}

#[test]
fn streaming_config_zero_buffer_rejected() {
    let mut cfg = StreamingConfig::default();
    cfg.buffer_size = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn streaming_config_zero_flush_interval_rejected() {
    let mut cfg = StreamingConfig::default();
    cfg.flush_interval_ms = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn streaming_config_zero_timeout_rejected() {
    let mut cfg = StreamingConfig::default();
    cfg.token_timeout_ms = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn streaming_config_low_latency_valid() {
    let cfg = StreamingConfig::low_latency();
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.buffer_size, 1);
    assert!(cfg.cancellable);
}

#[test]
fn streaming_config_high_throughput_valid() {
    let cfg = StreamingConfig::high_throughput();
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.buffer_size, 50);
    assert!(!cfg.cancellable);
}

#[test]
fn streaming_config_large_values_accepted() {
    let cfg = StreamingConfig {
        buffer_size: usize::MAX,
        flush_interval_ms: u64::MAX,
        max_retries: usize::MAX,
        token_timeout_ms: u64::MAX,
        cancellable: false,
    };
    assert!(cfg.validate().is_ok());
}

// ===== GenerationStats: atomic counters =====

#[test]
fn generation_stats_defaults_to_zero() {
    let stats = GenerationStats::default();
    assert_eq!(stats.tokens_generated(), 0);
    assert_eq!(stats.errors_encountered(), 0);
    assert_eq!(stats.retries_attempted(), 0);
    assert!(!stats.is_cancelled());
}

#[test]
fn generation_stats_atomic_increments() {
    let stats = GenerationStats::default();
    stats.tokens_generated.fetch_add(5, Ordering::Relaxed);
    stats.errors_encountered.fetch_add(2, Ordering::Relaxed);
    stats.retries_attempted.fetch_add(3, Ordering::Relaxed);
    assert_eq!(stats.tokens_generated(), 5);
    assert_eq!(stats.errors_encountered(), 2);
    assert_eq!(stats.retries_attempted(), 3);
}

#[test]
fn generation_stats_cancellation_flag() {
    let stats = GenerationStats::default();
    assert!(!stats.is_cancelled());
    stats.cancelled.store(true, Ordering::Relaxed);
    assert!(stats.is_cancelled());
}

#[test]
fn generation_stats_concurrent_updates() {
    use std::sync::Arc;
    let stats = Arc::new(GenerationStats::default());

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let s = Arc::clone(&stats);
            std::thread::spawn(move || {
                for _ in 0..100 {
                    s.tokens_generated.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(stats.tokens_generated(), 800);
}

// ===== GenerationConfig: is_stop_token edge cases =====

#[test]
fn is_stop_token_empty_set() {
    let config = GenerationConfig::default();
    assert!(!config.is_stop_token(0));
    assert!(!config.is_stop_token(u32::MAX));
}

#[test]
fn is_stop_token_after_with_stop_token_ids() {
    let config = GenerationConfig::default().with_stop_token_ids(vec![0, u32::MAX, 128009]);
    assert!(config.is_stop_token(0));
    assert!(config.is_stop_token(u32::MAX));
    assert!(config.is_stop_token(128009));
    assert!(!config.is_stop_token(1));
}

#[test]
fn is_stop_token_single_id_builder() {
    let config = GenerationConfig::default()
        .with_stop_token_id(100)
        .with_stop_token_id(200)
        .with_stop_token_id(100); // duplicate
    assert!(config.is_stop_token(100));
    assert!(config.is_stop_token(200));
    // Duplicates are benign (vec grows but set deduplicates)
    assert_eq!(config.stop_token_ids.len(), 3);
}

#[test]
fn rebuild_stop_token_set_after_direct_mutation() {
    let mut config = GenerationConfig::default();
    config.stop_token_ids = vec![42, 99];
    // Without rebuild, the HashSet is stale
    assert!(!config.is_stop_token(42));
    config.rebuild_stop_token_set();
    assert!(config.is_stop_token(42));
    assert!(config.is_stop_token(99));
}

#[test]
fn rebuild_stop_token_set_after_clearing_vec() {
    let mut config = GenerationConfig::default().with_stop_token_ids(vec![1, 2, 3]);
    assert!(config.is_stop_token(1));
    config.stop_token_ids.clear();
    config.rebuild_stop_token_set();
    assert!(!config.is_stop_token(1));
}

#[test]
fn rebuild_stop_token_set_after_deserialization() {
    let json = serde_json::json!({
        "max_new_tokens": 10,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "stop_sequences": [],
        "stop_token_ids": [128009, 128001],
        "stop_string_window": 64,
        "seed": null,
        "skip_special_tokens": true,
        "eos_token_id": null,
        "logits_tap_steps": 0,
        "logits_topk": 10,
        "add_bos": false
    });
    let mut config: GenerationConfig = serde_json::from_value(json).unwrap();
    // Before rebuild, HashSet is empty (serde skip)
    assert!(!config.is_stop_token(128009));
    config.rebuild_stop_token_set();
    assert!(config.is_stop_token(128009));
    assert!(config.is_stop_token(128001));
}

// ===== GenerationConfig: validation edge cases =====

#[test]
fn generation_config_top_p_boundary_one_is_valid() {
    let config = GenerationConfig::default().with_top_p(1.0);
    assert!(config.validate().is_ok());
}

#[test]
fn generation_config_repetition_penalty_just_above_zero() {
    let config = GenerationConfig::default().with_repetition_penalty(f32::MIN_POSITIVE);
    assert!(config.validate().is_ok());
}

#[test]
fn generation_config_negative_repetition_penalty_invalid() {
    let config = GenerationConfig::default().with_repetition_penalty(-0.1);
    assert!(config.validate().is_err());
}

#[test]
fn generation_config_stop_string_window_builder() {
    let config = GenerationConfig::default().with_stop_string_window(128);
    assert_eq!(config.stop_string_window, 128);
}
