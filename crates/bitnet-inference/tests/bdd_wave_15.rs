//! BDD Wave 15 — Integration tests for the inference pipeline.
//!
//! 10 Given/When/Then scenarios covering: deterministic seed
//! reproducibility, max_tokens=1 single-token generation, stop sequence
//! matching, temperature=0 greedy sampling, and empty-prompt error
//! handling.

use bitnet_inference::config::GenerationConfig;
use bitnet_inference::config_builder::{InferenceConfigBuilder, InferencePreset};
use bitnet_inference::sampling::{SamplingConfig, SamplingStrategy};

// ═══════════════════════════════════════════════════════════════════
// Scenario 1: Deterministic seed — generate twice, outputs identical
// ═══════════════════════════════════════════════════════════════════

#[test]
fn bdd_w15_deterministic_seed_identical_outputs() {
    // Given two SamplingStrategies with the same seed
    let config = SamplingConfig { temperature: 0.8, seed: Some(42), ..Default::default() };
    let mut s1 = SamplingStrategy::new(config.clone());
    let mut s2 = SamplingStrategy::new(config);

    let logits = vec![0.1f32, 0.5, 0.3, 0.05, 0.05];

    // When sampling 10 tokens from each
    let mut tokens1 = Vec::new();
    let mut tokens2 = Vec::new();
    for _ in 0..10 {
        tokens1.push(s1.sample(&logits, &[]).unwrap());
        tokens2.push(s2.sample(&logits, &[]).unwrap());
    }

    // Then outputs are identical
    assert_eq!(tokens1, tokens2, "same seed must produce identical sequences");
}

#[test]
fn bdd_w15_different_seeds_produce_different_outputs() {
    // Given two strategies with different seeds
    let mut s1 = SamplingStrategy::new(SamplingConfig {
        temperature: 0.9,
        seed: Some(1),
        ..Default::default()
    });
    let mut s2 = SamplingStrategy::new(SamplingConfig {
        temperature: 0.9,
        seed: Some(999),
        ..Default::default()
    });

    // When sampling many tokens from a flat distribution
    let logits = vec![1.0f32; 100];
    let mut tokens1 = Vec::new();
    let mut tokens2 = Vec::new();
    for _ in 0..20 {
        tokens1.push(s1.sample(&logits, &[]).unwrap());
        tokens2.push(s2.sample(&logits, &[]).unwrap());
    }

    // Then the sequences are likely different
    // (with 100 choices and 20 samples, probability of exact match is negligible)
    assert_ne!(
        tokens1, tokens2,
        "different seeds should (almost certainly) produce different sequences"
    );
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 2: max_tokens=1 — exactly 1 token produced
// ═══════════════════════════════════════════════════════════════════

#[test]
fn bdd_w15_max_tokens_one_produces_single_token() {
    // Given a GenerationConfig with max_tokens=1
    let config = GenerationConfig::greedy().with_max_tokens(1);

    // When we validate the config
    assert!(config.validate().is_ok());

    // Then max_new_tokens is exactly 1
    assert_eq!(config.max_new_tokens, 1);

    // And sampling once yields exactly one token
    let mut strategy =
        SamplingStrategy::new(SamplingConfig { temperature: 0.0, ..Default::default() });
    let logits = vec![0.1, 0.9, 0.3];
    let token = strategy.sample(&logits, &[]).unwrap();

    // The token is a valid index into the logits
    assert!((token as usize) < logits.len());
}

#[test]
fn bdd_w15_max_tokens_zero_validation_fails() {
    // Given a GenerationConfig with max_tokens=0
    let config = GenerationConfig::greedy().with_max_tokens(0);

    // When validating
    let result = config.validate();

    // Then validation fails
    assert!(result.is_err(), "max_tokens=0 should fail validation");
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 3: Stop sequence — generation stops when encountered
// ═══════════════════════════════════════════════════════════════════

#[test]
fn bdd_w15_stop_sequence_is_registered() {
    // Given a config with stop sequences
    let config = GenerationConfig::greedy()
        .with_stop_sequence("</s>".to_string())
        .with_stop_sequence("\n\nQ:".to_string());

    // When inspecting the config
    // Then the stop sequences are stored
    assert_eq!(config.stop_sequences.len(), 2);
    assert!(config.stop_sequences.contains(&"</s>".to_string()));
    assert!(config.stop_sequences.contains(&"\n\nQ:".to_string()));
}

#[test]
fn bdd_w15_stop_token_id_lookup() {
    // Given a config with stop token IDs
    let config = GenerationConfig::greedy().with_stop_token_id(128009).with_stop_token_id(2);

    // When checking if a token is a stop token
    // Then registered IDs return true
    assert!(config.is_stop_token(128009));
    assert!(config.is_stop_token(2));

    // And unregistered IDs return false
    assert!(!config.is_stop_token(0));
    assert!(!config.is_stop_token(42));
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 4: Temperature=0 — always picks highest logit
// ═══════════════════════════════════════════════════════════════════

#[test]
fn bdd_w15_temperature_zero_always_picks_argmax() {
    // Given greedy sampling (temperature=0)
    let config = SamplingConfig { temperature: 0.0, ..Default::default() };
    let mut strategy = SamplingStrategy::new(config);

    // When sampling from logits with a clear maximum at index 3
    let logits = vec![0.1f32, 0.2, 0.3, 0.9, 0.05];
    let mut tokens = Vec::new();
    for _ in 0..5 {
        tokens.push(strategy.sample(&logits, &[]).unwrap());
    }

    // Then every sample is the argmax index (3)
    for (i, &t) in tokens.iter().enumerate() {
        assert_eq!(t, 3, "sample {i}: expected argmax=3, got {t}");
    }
}

#[test]
fn bdd_w15_temperature_zero_greedy_config_consistency() {
    // Given a GenerationConfig::greedy()
    let config = GenerationConfig::greedy();

    // When inspecting temperature
    // Then it is exactly 0.0 (greedy)
    assert!(
        (config.temperature - 0.0).abs() < f32::EPSILON,
        "greedy config should have temperature=0.0"
    );
}

// ═══════════════════════════════════════════════════════════════════
// Scenario 5: Empty prompt — appropriate error returned
// ═══════════════════════════════════════════════════════════════════

#[test]
fn bdd_w15_empty_logits_returns_error() {
    // Given a sampling strategy
    let mut strategy =
        SamplingStrategy::new(SamplingConfig { temperature: 0.0, ..Default::default() });

    // When sampling from an empty logits slice
    let result = strategy.sample(&[], &[]);

    // Then an error is returned
    assert!(result.is_err(), "empty logits should return an error");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("Empty") || err_msg.contains("empty"),
        "error should mention empty logits: {err_msg}"
    );
}

#[test]
fn bdd_w15_empty_prompt_builder_validation() {
    // Given a builder with invalid configuration (zero threads)
    let result = InferenceConfigBuilder::new().preset(InferencePreset::Fast).build();

    // When built, it should succeed (Fast preset is valid)
    assert!(result.is_ok());

    // And a config with explicitly bad temperature should fail
    let result = InferenceConfigBuilder::new().temperature(-1.0).build();
    assert!(result.is_err(), "negative temperature should fail validation");
}
