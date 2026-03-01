//! Edge-case tests for bitnet-sampling strategy, greedy, temperature, and config.
//!
//! Covers boundary conditions, reproducibility, reset behavior, config update,
//! multi-SLM sampling configurations, and the SamplerChain builder.

use bitnet_sampling::*;

// ---------------------------------------------------------------------------
// SamplingConfig defaults
// ---------------------------------------------------------------------------

#[test]
fn config_default_values() {
    let cfg = SamplingConfig::default();
    assert_eq!(cfg.temperature, 0.7);
    assert_eq!(cfg.top_k, 50);
    assert_eq!(cfg.top_p, 0.9);
    assert_eq!(cfg.repetition_penalty, 1.0);
    assert!(cfg.seed.is_none());
}

#[test]
fn config_clone() {
    let cfg = SamplingConfig { temperature: 0.5, seed: Some(42), ..Default::default() };
    let cfg2 = cfg.clone();
    assert_eq!(cfg.temperature, cfg2.temperature);
    assert_eq!(cfg.seed, cfg2.seed);
}

// ---------------------------------------------------------------------------
// greedy_sample
// ---------------------------------------------------------------------------

#[test]
fn greedy_picks_highest() {
    assert_eq!(greedy_sample(&[1.0, 5.0, 3.0]).unwrap(), 1);
}

#[test]
fn greedy_tie_lowest_id() {
    // On tie, greedy_sample picks lowest index
    assert_eq!(greedy_sample(&[1.0, 1.0, 0.5]).unwrap(), 0);
}

#[test]
fn greedy_tie_all_equal() {
    assert_eq!(greedy_sample(&[3.0, 3.0, 3.0]).unwrap(), 0);
}

#[test]
fn greedy_single_element() {
    assert_eq!(greedy_sample(&[42.0]).unwrap(), 0);
}

#[test]
fn greedy_empty_errors() {
    assert!(greedy_sample(&[]).is_err());
}

#[test]
fn greedy_negative_logits() {
    assert_eq!(greedy_sample(&[-5.0, -1.0, -3.0]).unwrap(), 1);
}

#[test]
fn greedy_with_neg_infinity() {
    assert_eq!(greedy_sample(&[f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY]).unwrap(), 1);
}

// ---------------------------------------------------------------------------
// SamplingStrategy: greedy path (temperature=0)
// ---------------------------------------------------------------------------

#[test]
fn strategy_greedy_picks_max() {
    let cfg = SamplingConfig { temperature: 0.0, seed: Some(0), ..Default::default() };
    let mut s = SamplingStrategy::new(cfg);
    assert_eq!(s.sample(&[0.1, 0.9, 0.3], &[]).unwrap(), 1);
}

#[test]
fn strategy_greedy_empty_errors() {
    let cfg = SamplingConfig { temperature: 0.0, seed: Some(0), ..Default::default() };
    let mut s = SamplingStrategy::new(cfg);
    assert!(s.sample(&[], &[]).is_err());
}

#[test]
fn strategy_greedy_deterministic_across_calls() {
    let cfg = SamplingConfig { temperature: 0.0, seed: Some(42), ..Default::default() };
    let mut s = SamplingStrategy::new(cfg);
    let logits = vec![0.1, 0.5, 0.3, 0.4];
    let t1 = s.sample(&logits, &[]).unwrap();
    let t2 = s.sample(&logits, &[]).unwrap();
    assert_eq!(t1, t2); // Greedy is always deterministic
    assert_eq!(t1, 1);
}

// ---------------------------------------------------------------------------
// SamplingStrategy: stochastic path
// ---------------------------------------------------------------------------

#[test]
fn strategy_stochastic_reproducible() {
    let cfg = SamplingConfig { temperature: 0.8, seed: Some(42), ..Default::default() };
    let mut s1 = SamplingStrategy::new(cfg.clone());
    let mut s2 = SamplingStrategy::new(cfg);
    let logits = vec![0.2, 0.5, 0.3];
    assert_eq!(s1.sample(&logits, &[]).unwrap(), s2.sample(&logits, &[]).unwrap());
}

#[test]
fn strategy_stochastic_returns_valid_index() {
    let cfg = SamplingConfig { temperature: 1.0, seed: Some(99), ..Default::default() };
    let mut s = SamplingStrategy::new(cfg);
    let logits = vec![0.1; 100];
    for _ in 0..20 {
        let t = s.sample(&logits, &[]).unwrap();
        assert!((t as usize) < 100);
    }
}

#[test]
fn strategy_high_temperature_varies() {
    let cfg = SamplingConfig {
        temperature: 2.0,
        top_k: 0,
        top_p: 1.0,
        seed: Some(123),
        repetition_penalty: 1.0,
    };
    let mut s = SamplingStrategy::new(cfg);
    let logits = vec![0.5, 0.3, 0.2, 0.4, 0.35];
    let mut tokens = std::collections::HashSet::new();
    for _ in 0..50 {
        tokens.insert(s.sample(&logits, &[]).unwrap());
    }
    // With high temperature and 50 samples, we should see variety
    assert!(tokens.len() > 1, "high temp should produce varied tokens");
}

// ---------------------------------------------------------------------------
// Repetition penalty
// ---------------------------------------------------------------------------

#[test]
fn strategy_repetition_penalty_effect() {
    let cfg = SamplingConfig {
        temperature: 0.0,
        repetition_penalty: 5.0,
        seed: Some(0),
        ..Default::default()
    };
    let mut s = SamplingStrategy::new(cfg);
    // Token 1 has highest logit but is in context
    let t = s.sample(&[3.0, 5.0, 4.0], &[1]).unwrap();
    // With heavy penalty on token 1, token 2 (4.0) should win
    assert_eq!(t, 2);
}

#[test]
fn strategy_repetition_penalty_one_is_noop() {
    let cfg = SamplingConfig {
        temperature: 0.0,
        repetition_penalty: 1.0,
        seed: Some(0),
        ..Default::default()
    };
    let mut s = SamplingStrategy::new(cfg);
    assert_eq!(s.sample(&[0.1, 0.9, 0.3], &[1, 1, 1]).unwrap(), 1);
}

#[test]
fn strategy_repetition_penalty_count_aware() {
    let cfg = SamplingConfig {
        temperature: 0.0,
        repetition_penalty: 1.5,
        seed: Some(0),
        ..Default::default()
    };
    let mut s = SamplingStrategy::new(cfg);
    // Token 0 appears 3 times → penalty^3 = 1.5^3 = 3.375
    // Token 1 appears 1 time → penalty^1 = 1.5
    // Logits: [10.0, 8.0, 5.0]
    // After penalty: [10.0/3.375=2.96, 8.0/1.5=5.33, 5.0]
    let t = s.sample(&[10.0, 8.0, 5.0], &[0, 0, 0, 1]).unwrap();
    assert_eq!(t, 1); // 5.33 > 5.0 > 2.96
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

#[test]
fn strategy_reset_clears_state() {
    let cfg = SamplingConfig { temperature: 0.0, seed: Some(0), ..Default::default() };
    let mut s = SamplingStrategy::new(cfg);
    let _ = s.sample(&[0.1, 0.9, 0.3], &[]).unwrap();
    s.reset();
    // After reset, internal counts are cleared
    let t = s.sample(&[0.1, 0.9, 0.3], &[]).unwrap();
    assert_eq!(t, 1); // Same result, no stale state
}

// ---------------------------------------------------------------------------
// update_config
// ---------------------------------------------------------------------------

#[test]
fn strategy_update_config_changes_behavior() {
    let cfg = SamplingConfig { temperature: 0.0, seed: Some(42), ..Default::default() };
    let mut s = SamplingStrategy::new(cfg);
    // Greedy picks index 1
    assert_eq!(s.sample(&[0.1, 0.9, 0.3], &[]).unwrap(), 1);
    // Switch to stochastic
    s.update_config(SamplingConfig {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        seed: Some(42),
        repetition_penalty: 1.0,
    });
    // Should still produce valid output
    let t = s.sample(&[0.1, 0.9, 0.3], &[]).unwrap();
    assert!((t as usize) < 3);
}

// ---------------------------------------------------------------------------
// temperature_sample
// ---------------------------------------------------------------------------

#[test]
fn temperature_sample_zero_is_greedy() {
    let mut rng = rand::rng();
    assert_eq!(temperature_sample(&[0.1, 0.9, 0.3], 0.0, &mut rng).unwrap(), 1);
}

#[test]
fn temperature_sample_negative_is_greedy() {
    let mut rng = rand::rng();
    assert_eq!(temperature_sample(&[0.1, 0.9, 0.3], -1.0, &mut rng).unwrap(), 1);
}

#[test]
fn temperature_sample_empty_errors() {
    let mut rng = rand::rng();
    assert!(temperature_sample(&[], 0.7, &mut rng).is_err());
}

#[test]
fn temperature_sample_positive_valid() {
    let mut rng = rand::rng();
    let t = temperature_sample(&[0.1, 0.9, 0.3], 0.7, &mut rng).unwrap();
    assert!((t as usize) < 3);
}

// ---------------------------------------------------------------------------
// SamplerChain builder
// ---------------------------------------------------------------------------

#[test]
fn sampler_chain_builder_empty() {
    let chain = SamplerChain::builder().build(None);
    assert!(chain.stages().is_empty());
}

#[test]
fn sampler_chain_builder_with_temperature() {
    let chain = SamplerChain::builder().temperature(0.8).build(Some(42));
    assert_eq!(chain.stages().len(), 1);
    assert!(matches!(chain.stages()[0], SamplerStage::Temperature(t) if (t - 0.8).abs() < 1e-6));
}

#[test]
fn sampler_chain_builder_with_top_k() {
    let chain = SamplerChain::builder().top_k(50).build(Some(42));
    assert_eq!(chain.stages().len(), 1);
    assert!(matches!(chain.stages()[0], SamplerStage::TopK(50)));
}

#[test]
fn sampler_chain_builder_with_top_p() {
    let chain = SamplerChain::builder().top_p(0.9).build(Some(42));
    assert_eq!(chain.stages().len(), 1);
    assert!(matches!(chain.stages()[0], SamplerStage::TopP(p) if (p - 0.9).abs() < 1e-6));
}

#[test]
fn sampler_chain_builder_full_pipeline() {
    let chain = SamplerChain::builder().temperature(0.8).top_k(50).top_p(0.9).build(Some(42));
    assert_eq!(chain.stages().len(), 3);
}

// ---------------------------------------------------------------------------
// Multi-SLM sampling configurations
// ---------------------------------------------------------------------------

#[test]
fn phi4_sampling_config() {
    let cfg = SamplingConfig {
        temperature: 0.7,
        top_k: 50,
        top_p: 0.95,
        repetition_penalty: 1.1,
        seed: Some(42),
    };
    let mut s = SamplingStrategy::new(cfg);
    let logits = vec![0.1; 100];
    let t = s.sample(&logits, &[]).unwrap();
    assert!((t as usize) < 100);
}

#[test]
fn greedy_decoding_for_deterministic_eval() {
    let cfg = SamplingConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(0),
    };
    let mut s = SamplingStrategy::new(cfg);
    let logits = vec![0.1, 0.9, 0.3, 0.5, 0.2];
    let t = s.sample(&logits, &[]).unwrap();
    assert_eq!(t, 1);
}

#[test]
fn creative_writing_config() {
    let cfg = SamplingConfig {
        temperature: 1.5,
        top_k: 0,
        top_p: 0.7,
        repetition_penalty: 1.2,
        seed: Some(42),
    };
    let mut s = SamplingStrategy::new(cfg);
    let logits = vec![0.3; 50];
    let t = s.sample(&logits, &[]).unwrap();
    assert!((t as usize) < 50);
}
