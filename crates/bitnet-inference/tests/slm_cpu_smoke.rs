//! SLM CPU Smoke Regression Fixtures
//!
//! Deterministic tests for CPU inference math primitives that exercise the
//! SLM hot path without requiring model files.  Every test is seed-pinned
//! (seed=42 where randomness is involved) and must produce identical results
//! on every run.
#![cfg(feature = "cpu")]

use bitnet_inference::config::GenerationConfig;
use bitnet_inference::sampling::{
    SamplingConfig, SamplingStrategy, apply_repetition_penalty, apply_temperature, apply_top_k,
    argmax, greedy_sample, softmax_in_place,
};

// ── Greedy sampling ────────────────────────────────────────────────────

#[test]
fn greedy_picks_highest_logit() {
    let logits = vec![0.1_f32, 0.9, 0.3, 0.5];
    assert_eq!(greedy_sample(&logits).unwrap(), 1);
}

#[test]
fn greedy_tie_breaks_to_lowest_id() {
    // Equal logits → lowest index wins (llama.cpp compat).
    let logits = vec![1.0_f32, 1.0, 1.0];
    assert_eq!(greedy_sample(&logits).unwrap(), 0);
}

#[test]
fn greedy_single_element() {
    let logits = vec![42.0_f32];
    assert_eq!(greedy_sample(&logits).unwrap(), 0);
}

#[test]
fn greedy_negative_logits() {
    let logits = vec![-5.0_f32, -1.0, -3.0];
    assert_eq!(greedy_sample(&logits).unwrap(), 1); // -1.0 is the highest
}

// ── SamplingStrategy: greedy (temperature=0) ───────────────────────────

#[test]
fn strategy_greedy_deterministic() {
    let config = SamplingConfig { temperature: 0.0, seed: Some(42), ..Default::default() };
    let mut strategy = SamplingStrategy::new(config);

    let logits = vec![0.2_f32, 0.8, 0.5, 0.1];
    let token = strategy.sample(&logits, &[]).unwrap();
    assert_eq!(token, 1, "greedy must pick argmax");
}

#[test]
fn strategy_greedy_matches_greedy_sample() {
    let config = SamplingConfig { temperature: 0.0, seed: Some(42), ..Default::default() };
    let mut strategy = SamplingStrategy::new(config);

    let logits = vec![0.3_f32, 0.1, 0.7, 0.4, 0.6];
    let via_strategy = strategy.sample(&logits, &[]).unwrap();
    let via_greedy = greedy_sample(&logits).unwrap();
    assert_eq!(via_strategy, via_greedy);
}

// ── SamplingStrategy: stochastic with fixed seed ───────────────────────

#[test]
fn strategy_seeded_is_reproducible() {
    let logits = vec![0.2_f32, 0.5, 0.3, 0.1];

    let make = || {
        let config = SamplingConfig { temperature: 0.8, seed: Some(42), ..Default::default() };
        SamplingStrategy::new(config)
    };

    let mut s1 = make();
    let mut s2 = make();

    // Run 10 samples; every pair must match.
    for _ in 0..10 {
        let t1 = s1.sample(&logits, &[]).unwrap();
        let t2 = s2.sample(&logits, &[]).unwrap();
        assert_eq!(t1, t2, "same seed must produce same sequence");
    }
}

#[test]
fn strategy_different_seeds_diverge() {
    let logits = vec![0.25_f32; 100]; // uniform → seed controls outcome

    let mut s1 = SamplingStrategy::new(SamplingConfig {
        temperature: 1.0,
        seed: Some(1),
        top_k: 0,
        top_p: 1.0,
        ..Default::default()
    });
    let mut s2 = SamplingStrategy::new(SamplingConfig {
        temperature: 1.0,
        seed: Some(999),
        top_k: 0,
        top_p: 1.0,
        ..Default::default()
    });

    // Collect 20 tokens from each; at least one pair should differ.
    let tokens1: Vec<u32> = (0..20).map(|_| s1.sample(&logits, &[]).unwrap()).collect();
    let tokens2: Vec<u32> = (0..20).map(|_| s2.sample(&logits, &[]).unwrap()).collect();
    assert_ne!(tokens1, tokens2, "different seeds should produce different sequences");
}

// ── SamplingStrategy: top-k ────────────────────────────────────────────

#[test]
fn strategy_top_k_limits_candidates() {
    // With top_k=1, the strategy should always pick the argmax regardless
    // of temperature, because only the top-1 logit survives filtering.
    let config = SamplingConfig {
        temperature: 1.0,
        top_k: 1,
        top_p: 1.0,
        seed: Some(42),
        ..Default::default()
    };
    let mut strategy = SamplingStrategy::new(config);

    let logits = vec![0.1_f32, 0.9, 0.3];
    for _ in 0..20 {
        let token = strategy.sample(&logits, &[]).unwrap();
        assert_eq!(token, 1, "top_k=1 must always select the argmax");
    }
}

// ── Temperature scaling ────────────────────────────────────────────────

#[test]
fn temperature_zero_is_noop() {
    let original = vec![1.0_f32, 2.0, 3.0];
    let mut logits = original.clone();
    apply_temperature(&mut logits, 0.0);
    assert_eq!(logits, original, "temperature 0.0 must be a no-op");
}

#[test]
fn temperature_one_is_passthrough() {
    let original = vec![1.0_f32, 2.0, 3.0];
    let mut logits = original.clone();
    apply_temperature(&mut logits, 1.0);
    assert_eq!(logits, original, "temperature 1.0 must be a passthrough");
}

#[test]
fn temperature_scales_by_inverse() {
    let mut logits = vec![2.0_f32, 4.0, 6.0];
    apply_temperature(&mut logits, 2.0);
    // 1/2.0 = 0.5 scaling
    assert!((logits[0] - 1.0).abs() < 1e-6);
    assert!((logits[1] - 2.0).abs() < 1e-6);
    assert!((logits[2] - 3.0).abs() < 1e-6);
}

#[test]
fn temperature_preserves_argmax() {
    let logits = vec![0.5_f32, 3.0, 1.5, 2.0];
    for temp in [0.1, 0.5, 1.0, 2.0, 5.0] {
        let mut scaled = logits.clone();
        apply_temperature(&mut scaled, temp);
        assert_eq!(argmax(&scaled), 1, "temperature {temp} must preserve argmax");
    }
}

// ── Repetition penalty ─────────────────────────────────────────────────

#[test]
fn repetition_penalty_reduces_positive_logit() {
    let mut logits = vec![0.0_f32, 4.0, -2.0];
    apply_repetition_penalty(&mut logits, &[1], 2.0);
    assert!((logits[1] - 2.0).abs() < 1e-6, "4.0 / 2.0 = 2.0");
}

#[test]
fn repetition_penalty_amplifies_negative_logit() {
    let mut logits = vec![0.0_f32, 4.0, -2.0];
    apply_repetition_penalty(&mut logits, &[2], 2.0);
    assert!((logits[2] - (-4.0)).abs() < 1e-6, "-2.0 * 2.0 = -4.0");
}

#[test]
fn repetition_penalty_one_is_noop() {
    let original = vec![1.0_f32, 2.0, 3.0];
    let mut logits = original.clone();
    apply_repetition_penalty(&mut logits, &[0, 1, 2], 1.0);
    assert_eq!(logits, original, "penalty=1.0 must be a no-op");
}

#[test]
fn repetition_penalty_deterministic_across_runs() {
    let run = || {
        let mut logits = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        apply_repetition_penalty(&mut logits, &[0, 2, 4], 1.5);
        logits
    };
    assert_eq!(run(), run(), "repetition penalty must be deterministic");
}

#[test]
fn repetition_penalty_multiple_tokens() {
    let mut logits = vec![3.0_f32, 3.0, 3.0];
    apply_repetition_penalty(&mut logits, &[0, 2], 1.5);
    // Tokens 0 and 2 penalized: 3.0 / 1.5 = 2.0
    assert!((logits[0] - 2.0).abs() < 1e-6);
    assert!((logits[1] - 3.0).abs() < 1e-6, "unseen token unchanged");
    assert!((logits[2] - 2.0).abs() < 1e-6);
}

// ── Softmax ────────────────────────────────────────────────────────────

#[test]
fn softmax_sums_to_one() {
    let mut logits = vec![1.0_f32, 2.0, 3.0, 4.0];
    softmax_in_place(&mut logits);
    let sum: f32 = logits.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum}");
}

#[test]
fn softmax_preserves_ordering() {
    let mut logits = vec![1.0_f32, 3.0, 2.0];
    softmax_in_place(&mut logits);
    assert!(logits[1] > logits[2], "3.0 > 2.0 in logits → prob[1] > prob[2]");
    assert!(logits[2] > logits[0], "2.0 > 1.0 in logits → prob[2] > prob[0]");
}

#[test]
fn softmax_uniform_input_gives_uniform_output() {
    let mut logits = vec![1.0_f32; 4];
    softmax_in_place(&mut logits);
    for &p in &logits {
        assert!((p - 0.25).abs() < 1e-6, "uniform logits → uniform probs");
    }
}

// ── Top-k filtering ────────────────────────────────────────────────────

#[test]
fn top_k_filters_non_top_entries() {
    let mut logits = vec![1.0_f32, 5.0, 3.0, 2.0, 4.0];
    apply_top_k(&mut logits, 2);
    // Top-2: index 1 (5.0), index 4 (4.0)
    assert!(logits[1].is_finite());
    assert!(logits[4].is_finite());
    assert!(logits[0] == f32::NEG_INFINITY);
    assert!(logits[2] == f32::NEG_INFINITY);
    assert!(logits[3] == f32::NEG_INFINITY);
}

#[test]
fn top_k_then_softmax_valid_distribution() {
    let mut logits = vec![1.0_f32, 5.0, 3.0, 2.0, 4.0];
    apply_top_k(&mut logits, 3);
    softmax_in_place(&mut logits);

    let sum: f32 = logits.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "top-k + softmax must sum to 1");

    // Filtered entries become 0.0 probability
    assert_eq!(logits[0], 0.0);
    assert_eq!(logits[3], 0.0);
}

// ── Argmax ─────────────────────────────────────────────────────────────

#[test]
fn argmax_finds_maximum() {
    let logits = vec![0.1_f32, 0.5, 0.9, 0.2];
    assert_eq!(argmax(&logits), 2);
}

#[test]
fn argmax_empty_returns_zero() {
    assert_eq!(argmax(&[]), 0);
}

// ── GenerationConfig builder ───────────────────────────────────────────

#[test]
fn generation_config_greedy_preset() {
    let config = GenerationConfig::greedy();
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_k, 1);
    assert_eq!(config.top_p, 1.0);
}

#[test]
fn generation_config_creative_preset() {
    let config = GenerationConfig::creative();
    assert_eq!(config.temperature, 0.9);
    assert_eq!(config.top_k, 100);
    assert_eq!(config.repetition_penalty, 1.1);
}

#[test]
fn generation_config_builder_chain() {
    let config = GenerationConfig::greedy()
        .with_max_tokens(16)
        .with_temperature(0.5)
        .with_top_k(10)
        .with_top_p(0.95)
        .with_repetition_penalty(1.2)
        .with_seed(42);

    assert_eq!(config.max_new_tokens, 16);
    assert_eq!(config.temperature, 0.5);
    assert_eq!(config.top_k, 10);
    assert_eq!(config.top_p, 0.95);
    assert_eq!(config.repetition_penalty, 1.2);
    assert_eq!(config.seed, Some(42));
}

#[test]
fn generation_config_validation_accepts_valid() {
    assert!(GenerationConfig::greedy().validate().is_ok());
    assert!(GenerationConfig::creative().validate().is_ok());
    assert!(GenerationConfig::balanced().validate().is_ok());
}

#[test]
fn generation_config_validation_rejects_invalid() {
    let bad_tokens = GenerationConfig::greedy().with_max_tokens(0);
    assert!(bad_tokens.validate().is_err());

    let bad_temp = GenerationConfig::greedy().with_temperature(-1.0);
    assert!(bad_temp.validate().is_err());

    let bad_top_p = GenerationConfig::greedy().with_top_p(0.0);
    assert!(bad_top_p.validate().is_err());

    let bad_penalty = GenerationConfig::greedy().with_repetition_penalty(0.0);
    assert!(bad_penalty.validate().is_err());
}

#[test]
fn generation_config_stop_tokens() {
    let config = GenerationConfig::greedy()
        .with_stop_token_ids(vec![128009, 128001])
        .with_stop_sequence("</s>".to_string());

    assert!(config.is_stop_token(128009));
    assert!(config.is_stop_token(128001));
    assert!(!config.is_stop_token(999));
    assert_eq!(config.stop_sequences, vec!["</s>"]);
}

// ── End-to-end: fixed logits → deterministic token ─────────────────────

#[test]
fn e2e_fixed_logits_greedy_pipeline() {
    // Simulate the full pipeline: logits → temperature → top-k → softmax → argmax
    let mut logits = vec![1.0_f32, 4.0, 2.0, 3.0];

    apply_temperature(&mut logits, 0.5); // sharpen
    apply_top_k(&mut logits, 2); // keep top-2
    softmax_in_place(&mut logits);

    let selected = argmax(&logits);
    assert_eq!(selected, 1, "token 1 (logit 4.0) must win");

    // Verify filtered tokens have zero probability
    assert_eq!(logits[0], 0.0);
    assert_eq!(logits[2], 0.0);
}

#[test]
fn e2e_fixed_logits_with_penalty() {
    let mut logits = vec![5.0_f32, 3.0, 3.0, 1.0];
    // Penalize token 0 so token 1 or 2 should win
    apply_repetition_penalty(&mut logits, &[0], 10.0);
    // 5.0 / 10.0 = 0.5; token 1 and 2 are now highest at 3.0
    let selected = greedy_sample(&logits).unwrap();
    assert_eq!(selected, 1, "after penalty, token 1 (first of tied 3.0s) wins");
}

#[test]
fn e2e_pipeline_deterministic_10_runs() {
    let run = || {
        let mut logits = vec![0.5_f32, 2.0, 1.5, 0.8, 3.0];
        apply_repetition_penalty(&mut logits, &[4], 1.3);
        apply_temperature(&mut logits, 0.7);
        apply_top_k(&mut logits, 3);
        softmax_in_place(&mut logits);
        argmax(&logits)
    };

    let expected = run();
    for _ in 0..10 {
        assert_eq!(run(), expected, "pipeline must be deterministic");
    }
}
