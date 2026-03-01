//! Edge-case tests for bitnet-sampling: SamplingConfig, SamplingStrategy,
//! greedy/stochastic sampling, repetition penalty, seed reproducibility.

use bitnet_sampling::{SamplingConfig, SamplingStrategy};

// ---------------------------------------------------------------------------
// SamplingConfig — defaults
// ---------------------------------------------------------------------------

#[test]
fn config_default_values() {
    let cfg = SamplingConfig::default();
    assert!((cfg.temperature - 0.7).abs() < f32::EPSILON);
    assert_eq!(cfg.top_k, 50);
    assert!((cfg.top_p - 0.9).abs() < f32::EPSILON);
    assert!((cfg.repetition_penalty - 1.0).abs() < f32::EPSILON);
    assert!(cfg.seed.is_none());
}

#[test]
fn config_clone() {
    let cfg = SamplingConfig { temperature: 0.5, seed: Some(42), ..Default::default() };
    let cloned = cfg.clone();
    assert!((cloned.temperature - 0.5).abs() < f32::EPSILON);
    assert_eq!(cloned.seed, Some(42));
}

#[test]
fn config_debug() {
    let cfg = SamplingConfig::default();
    let dbg = format!("{cfg:?}");
    assert!(dbg.contains("SamplingConfig"));
}

// ---------------------------------------------------------------------------
// Greedy sampling (temperature=0.0)
// ---------------------------------------------------------------------------

#[test]
fn greedy_picks_highest() {
    let cfg = SamplingConfig { temperature: 0.0, seed: Some(0), ..Default::default() };
    let mut strategy = SamplingStrategy::new(cfg);
    let logits = vec![0.1f32, 0.9, 0.3];
    assert_eq!(strategy.sample(&logits, &[]).unwrap(), 1);
}

#[test]
fn greedy_single_token() {
    let cfg = SamplingConfig { temperature: 0.0, seed: Some(0), ..Default::default() };
    let mut strategy = SamplingStrategy::new(cfg);
    let logits = vec![42.0f32];
    assert_eq!(strategy.sample(&logits, &[]).unwrap(), 0);
}

#[test]
fn greedy_negative_logits() {
    let cfg = SamplingConfig { temperature: 0.0, seed: Some(0), ..Default::default() };
    let mut strategy = SamplingStrategy::new(cfg);
    let logits = vec![-5.0f32, -2.0, -10.0];
    assert_eq!(strategy.sample(&logits, &[]).unwrap(), 1); // -2.0 is highest
}

#[test]
fn greedy_deterministic_across_calls() {
    let cfg = SamplingConfig { temperature: 0.0, seed: Some(42), ..Default::default() };
    let mut s1 = SamplingStrategy::new(cfg.clone());
    let mut s2 = SamplingStrategy::new(cfg);
    let logits = vec![1.0f32, 5.0, 3.0];
    assert_eq!(s1.sample(&logits, &[]).unwrap(), s2.sample(&logits, &[]).unwrap());
}

// ---------------------------------------------------------------------------
// Stochastic sampling
// ---------------------------------------------------------------------------

#[test]
fn stochastic_with_seed_reproducible() {
    let cfg = SamplingConfig { temperature: 0.8, seed: Some(42), ..Default::default() };
    let mut s1 = SamplingStrategy::new(cfg.clone());
    let mut s2 = SamplingStrategy::new(cfg);
    let logits = vec![0.2f32, 0.5, 0.3];
    assert_eq!(s1.sample(&logits, &[]).unwrap(), s2.sample(&logits, &[]).unwrap());
}

#[test]
fn stochastic_returns_valid_token_id() {
    let cfg = SamplingConfig { temperature: 1.0, seed: Some(123), ..Default::default() };
    let mut strategy = SamplingStrategy::new(cfg);
    let logits = vec![1.0f32; 100];
    let token = strategy.sample(&logits, &[]).unwrap();
    assert!(token < 100);
}

#[test]
fn stochastic_high_temperature() {
    let cfg = SamplingConfig { temperature: 10.0, seed: Some(42), ..Default::default() };
    let mut strategy = SamplingStrategy::new(cfg);
    let logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let token = strategy.sample(&logits, &[]).unwrap();
    assert!(token < 5);
}

#[test]
fn stochastic_low_temperature() {
    let cfg = SamplingConfig { temperature: 0.01, seed: Some(42), ..Default::default() };
    let mut strategy = SamplingStrategy::new(cfg);
    let logits = vec![1.0f32, 10.0, 3.0]; // 10.0 dominates
    let token = strategy.sample(&logits, &[]).unwrap();
    assert_eq!(token, 1); // Almost greedy
}

// ---------------------------------------------------------------------------
// Empty logits error
// ---------------------------------------------------------------------------

#[test]
fn sample_empty_logits_error() {
    let cfg = SamplingConfig { temperature: 0.0, seed: Some(0), ..Default::default() };
    let mut strategy = SamplingStrategy::new(cfg);
    let result = strategy.sample(&[], &[]);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Top-k interaction
// ---------------------------------------------------------------------------

#[test]
fn top_k_zero_disabled() {
    let cfg = SamplingConfig { temperature: 1.0, top_k: 0, seed: Some(42), ..Default::default() };
    let mut strategy = SamplingStrategy::new(cfg);
    let logits = vec![1.0f32; 10];
    let token = strategy.sample(&logits, &[]).unwrap();
    assert!(token < 10);
}

#[test]
fn top_k_one_like_greedy() {
    let cfg = SamplingConfig { temperature: 1.0, top_k: 1, seed: Some(42), ..Default::default() };
    let mut strategy = SamplingStrategy::new(cfg);
    let logits = vec![1.0f32, 5.0, 3.0];
    // With top_k=1, only the highest is kept
    assert_eq!(strategy.sample(&logits, &[]).unwrap(), 1);
}

// ---------------------------------------------------------------------------
// Top-p interaction
// ---------------------------------------------------------------------------

#[test]
fn top_p_one_disabled() {
    let cfg = SamplingConfig { temperature: 0.8, top_p: 1.0, seed: Some(42), ..Default::default() };
    let mut strategy = SamplingStrategy::new(cfg);
    let logits = vec![1.0f32; 5];
    let token = strategy.sample(&logits, &[]).unwrap();
    assert!(token < 5);
}

// ---------------------------------------------------------------------------
// Repetition penalty
// ---------------------------------------------------------------------------

#[test]
fn repetition_penalty_no_penalty() {
    let cfg = SamplingConfig {
        temperature: 0.0,
        repetition_penalty: 1.0,
        seed: Some(0),
        ..Default::default()
    };
    let mut strategy = SamplingStrategy::new(cfg);
    let logits = vec![1.0f32, 5.0, 3.0];
    // No penalty, greedy → picks 1
    assert_eq!(strategy.sample(&logits, &[]).unwrap(), 1);
}

#[test]
fn repetition_penalty_reduces_repeated() {
    let cfg = SamplingConfig {
        temperature: 0.0,
        repetition_penalty: 10.0,
        seed: Some(0),
        ..Default::default()
    };
    let mut strategy = SamplingStrategy::new(cfg);
    let logits = vec![3.0f32, 5.0, 4.5];
    // Token 1 has already appeared in context, will be penalized
    let token = strategy.sample(&logits, &[1]).unwrap();
    // With penalty 10.0, token 1 (5.0) → 5.0/10.0 = 0.5
    // Token 2 (4.5) becomes highest → expect token 2
    assert_eq!(token, 2);
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

#[test]
fn reset_clears_state() {
    let cfg = SamplingConfig { temperature: 0.0, seed: Some(0), ..Default::default() };
    let mut strategy = SamplingStrategy::new(cfg);
    let logits = vec![1.0f32, 5.0, 3.0];
    let _ = strategy.sample(&logits, &[]).unwrap();
    strategy.reset();
    // After reset, can sample again
    let token = strategy.sample(&logits, &[]).unwrap();
    assert_eq!(token, 1);
}

// ---------------------------------------------------------------------------
// Multiple sequential samples
// ---------------------------------------------------------------------------

#[test]
fn multiple_samples_all_valid() {
    let cfg = SamplingConfig { temperature: 0.8, seed: Some(42), ..Default::default() };
    let mut strategy = SamplingStrategy::new(cfg);
    let logits = vec![1.0f32; 50];
    for _ in 0..20 {
        let token = strategy.sample(&logits, &[]).unwrap();
        assert!(token < 50);
    }
}
