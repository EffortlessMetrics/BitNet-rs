//! Snapshot tests for bitnet-sampling.
//!
//! Pins the debug representation of key sampling types and the output
//! of sampling operations with fixed seeds for regression detection.

use bitnet_sampling::{SamplingConfig, SamplingStrategy, greedy_sample};

#[test]
fn snapshot_default_sampling_config() {
    let config = SamplingConfig::default();
    insta::assert_debug_snapshot!("default_sampling_config", config);
}

#[test]
fn snapshot_greedy_config() {
    let config = SamplingConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(42),
    };
    insta::assert_debug_snapshot!("greedy_sampling_config", config);
}

#[test]
fn snapshot_creative_config() {
    let config = SamplingConfig {
        temperature: 1.5,
        top_k: 100,
        top_p: 0.95,
        repetition_penalty: 1.2,
        seed: Some(123),
    };
    insta::assert_debug_snapshot!("creative_sampling_config", config);
}

#[test]
fn snapshot_greedy_sample_output() {
    // Fixed logits: highest value at index 3
    let logits = vec![0.1f32, 0.2, 0.3, 2.5, 0.5, 0.1, 0.0];
    let result = greedy_sample(&logits).unwrap();
    insta::assert_debug_snapshot!("greedy_sample_index3", result);
}

#[test]
fn snapshot_seeded_sample_output() {
    // Fixed seed + fixed logits → deterministic token output
    let logits = vec![1.0f32, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
    let config = SamplingConfig {
        temperature: 0.7,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(42),
    };
    let mut strategy = SamplingStrategy::new(config);
    let token = strategy.sample(&logits, &[]).unwrap();
    insta::assert_debug_snapshot!("seeded_sample_seed42", token);
}

#[test]
fn snapshot_repetition_penalty_effect() {
    // With high repetition penalty for already-seen tokens,
    // the sampler should avoid index 3 (already seen)
    let logits = vec![0.1f32, 0.2, 0.3, 10.0, 0.5, 0.1]; // 10.0 at index 3
    let config = SamplingConfig {
        temperature: 0.0, // greedy
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.5,
        seed: Some(0),
    };
    let mut strategy = SamplingStrategy::new(config);
    // With no context, should pick index 3 (highest logit)
    let first = strategy.sample(&logits, &[]).unwrap();
    // With context containing index 3, repetition penalty should push selection
    let second = strategy.sample(&logits, &[3]).unwrap();
    insta::assert_debug_snapshot!("rep_penalty_first_no_context", first);
    insta::assert_debug_snapshot!("rep_penalty_second_with_context", second);
}
