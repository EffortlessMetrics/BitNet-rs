//! Snapshot wave 12 — bitnet-sampling
//!
//! Covers: MinPSampler, TypicalSampler, MirostatSampler, RepetitionPenaltyConfig,
//! SamplerStage, SamplerChain/Builder, SamplingConfig presets, SamplingStrategy
//! multi-step, greedy_sample edge cases.

use bitnet_sampling::{
    MinPSampler, MirostatSampler, RepetitionPenaltyConfig, SamplerChain, SamplerStage,
    SamplingConfig, SamplingStrategy, TypicalSampler, greedy_sample,
};

// ── MinPSampler ─────────────────────────────────────────────────────────────

#[test]
fn min_p_sampler_default() {
    let s = MinPSampler::new(0.05);
    insta::assert_debug_snapshot!(s);
}

#[test]
fn min_p_sampler_clamped_high() {
    let s = MinPSampler::new(2.0);
    insta::assert_debug_snapshot!(s);
}

#[test]
fn min_p_sampler_clamped_low() {
    let s = MinPSampler::new(-1.0);
    insta::assert_debug_snapshot!(s);
}

#[test]
fn min_p_filter_uniform() {
    let s = MinPSampler::new(0.3);
    let mut probs = vec![0.25, 0.25, 0.25, 0.25];
    s.filter(&mut probs);
    insta::assert_debug_snapshot!(probs);
}

#[test]
fn min_p_filter_skewed() {
    let s = MinPSampler::new(0.1);
    let mut probs = vec![0.7, 0.2, 0.05, 0.05];
    s.filter(&mut probs);
    insta::assert_debug_snapshot!(probs);
}

// ── TypicalSampler ──────────────────────────────────────────────────────────

#[test]
fn typical_sampler_default() {
    let s = TypicalSampler::new(0.95);
    insta::assert_debug_snapshot!(s);
}

#[test]
fn typical_sampler_clamped() {
    let s = TypicalSampler::new(5.0);
    insta::assert_debug_snapshot!(s);
}

#[test]
fn typical_filter_uniform() {
    let s = TypicalSampler::new(0.5);
    let mut probs = vec![0.25, 0.25, 0.25, 0.25];
    s.filter(&mut probs);
    insta::assert_debug_snapshot!(probs);
}

// ── MirostatSampler ─────────────────────────────────────────────────────────

#[test]
fn mirostat_sampler_debug() {
    let s = MirostatSampler::new(5.0, 0.1, Some(42));
    insta::assert_debug_snapshot!(s);
}

#[test]
fn mirostat_sample_deterministic() {
    let mut s = MirostatSampler::new(5.0, 0.1, Some(42));
    let logits: Vec<f32> = (0..100).map(|i| (i as f32) * 0.1).collect();
    let tok = s.sample(&logits).unwrap();
    insta::assert_snapshot!(format!("token_id={tok}"));
}

#[test]
fn mirostat_sample_twice_state_changes() {
    let mut s = MirostatSampler::new(5.0, 0.1, Some(42));
    let logits: Vec<f32> = (0..50).map(|i| (i as f32) * 0.2).collect();
    let t1 = s.sample(&logits).unwrap();
    let t2 = s.sample(&logits).unwrap();
    insta::assert_snapshot!(format!("t1={t1} t2={t2}"));
}

#[test]
fn mirostat_reset() {
    let mut s = MirostatSampler::new(3.0, 0.2, Some(99));
    let logits: Vec<f32> = (0..20).map(|i| i as f32).collect();
    let _ = s.sample(&logits);
    s.reset();
    insta::assert_debug_snapshot!(s);
}

// ── RepetitionPenaltyConfig ─────────────────────────────────────────────────

#[test]
fn repetition_penalty_config_default() {
    let c = RepetitionPenaltyConfig::default();
    insta::assert_debug_snapshot!(c);
}

#[test]
fn repetition_penalty_config_custom() {
    let c = RepetitionPenaltyConfig {
        frequency_penalty: 0.5,
        presence_penalty: 0.3,
        count_penalty: 1.2,
    };
    insta::assert_debug_snapshot!(c);
}

#[test]
fn repetition_penalty_apply_no_effect() {
    let c = RepetitionPenaltyConfig::default();
    let mut logits = vec![1.0, 2.0, 3.0, 4.0];
    c.apply(&mut logits, &[]);
    insta::assert_debug_snapshot!(logits);
}

#[test]
fn repetition_penalty_apply_with_counts() {
    let c = RepetitionPenaltyConfig {
        frequency_penalty: 0.5,
        presence_penalty: 0.3,
        count_penalty: 1.1,
    };
    let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let counts = vec![(1u32, 3usize), (3u32, 1usize)];
    c.apply(&mut logits, &counts);
    insta::assert_debug_snapshot!(logits);
}

// ── SamplerStage ────────────────────────────────────────────────────────────

#[test]
fn sampler_stage_temperature() {
    let s = SamplerStage::Temperature(0.7);
    insta::assert_debug_snapshot!(s);
}

#[test]
fn sampler_stage_top_k() {
    let s = SamplerStage::TopK(50);
    insta::assert_debug_snapshot!(s);
}

#[test]
fn sampler_stage_top_p() {
    let s = SamplerStage::TopP(0.9);
    insta::assert_debug_snapshot!(s);
}

#[test]
fn sampler_stage_min_p() {
    let s = SamplerStage::MinP(0.05);
    insta::assert_debug_snapshot!(s);
}

#[test]
fn sampler_stage_typical() {
    let s = SamplerStage::Typical(0.95);
    insta::assert_debug_snapshot!(s);
}

#[test]
fn sampler_stage_repetition_penalty() {
    let s = SamplerStage::RepetitionPenalty(
        RepetitionPenaltyConfig {
            frequency_penalty: 0.4,
            presence_penalty: 0.2,
            count_penalty: 1.1,
        },
        vec![(5, 2), (10, 1)],
    );
    insta::assert_debug_snapshot!(s);
}

// ── SamplerChain ────────────────────────────────────────────────────────────

#[test]
fn sampler_chain_builder_empty() {
    let chain = SamplerChain::builder().build(Some(42));
    insta::assert_debug_snapshot!(chain.stages());
}

#[test]
fn sampler_chain_builder_full_pipeline() {
    let chain = SamplerChain::builder()
        .temperature(0.8)
        .top_k(40)
        .top_p(0.9)
        .min_p(0.05)
        .typical(0.95)
        .build(Some(42));
    insta::assert_debug_snapshot!(chain.stages());
}

#[test]
fn sampler_chain_deterministic() {
    let chain = SamplerChain::builder().temperature(0.7).top_k(10).build(Some(42));
    let logits: Vec<f32> = (0..20).map(|i| (i as f32) * 0.5).collect();
    let t1 = chain.sample(&logits).unwrap();
    let t2 = chain.sample(&logits).unwrap();
    insta::assert_snapshot!(format!("t1={t1} t2={t2}"));
}

// ── SamplingConfig presets ──────────────────────────────────────────────────

#[test]
fn sampling_config_low_temp() {
    let c = SamplingConfig {
        temperature: 0.1,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(0),
    };
    insta::assert_debug_snapshot!(c);
}

#[test]
fn sampling_config_high_temp() {
    let c = SamplingConfig {
        temperature: 2.0,
        top_k: 200,
        top_p: 0.99,
        repetition_penalty: 1.3,
        seed: None,
    };
    insta::assert_debug_snapshot!(c);
}

// ── SamplingStrategy multi-step ─────────────────────────────────────────────

#[test]
fn sampling_strategy_multi_step_deterministic() {
    let config = SamplingConfig {
        temperature: 0.5,
        top_k: 10,
        top_p: 0.9,
        repetition_penalty: 1.0,
        seed: Some(42),
    };
    let mut strategy = SamplingStrategy::new(config);
    let logits: Vec<f32> = (0..50).map(|i| (i as f32) * 0.1 - 2.5).collect();
    let mut tokens = Vec::new();
    for _ in 0..4 {
        tokens.push(strategy.sample(&logits, &tokens).unwrap());
    }
    insta::assert_debug_snapshot!(tokens);
}

#[test]
fn sampling_strategy_reset_restores_state() {
    let config = SamplingConfig {
        temperature: 0.7,
        top_k: 5,
        top_p: 0.9,
        repetition_penalty: 1.0,
        seed: Some(99),
    };
    let mut s1 = SamplingStrategy::new(config.clone());
    let logits: Vec<f32> = (0..30).map(|i| i as f32).collect();
    let before = s1.sample(&logits, &[]).unwrap();
    s1.reset();
    let after = s1.sample(&logits, &[]).unwrap();
    insta::assert_snapshot!(format!("before={before} after={after}"));
}

// ── greedy_sample edge cases ────────────────────────────────────────────────

#[test]
fn greedy_sample_single_element() {
    let tok = greedy_sample(&[42.0]).unwrap();
    insta::assert_snapshot!(format!("token_id={tok}"));
}

#[test]
fn greedy_sample_ties_lowest_id() {
    let tok = greedy_sample(&[1.0, 1.0, 1.0]).unwrap();
    insta::assert_snapshot!(format!("token_id={tok}"));
}

#[test]
fn greedy_sample_negative_logits() {
    let tok = greedy_sample(&[-5.0, -1.0, -3.0, -0.5]).unwrap();
    insta::assert_snapshot!(format!("token_id={tok}"));
}

#[test]
fn greedy_sample_large_vocab() {
    let mut logits = vec![0.0f32; 32000];
    logits[12345] = 100.0;
    let tok = greedy_sample(&logits).unwrap();
    insta::assert_snapshot!(format!("token_id={tok}"));
}
