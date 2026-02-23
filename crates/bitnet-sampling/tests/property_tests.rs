//! Property-based tests for bitnet-sampling.
//!
//! Key invariants:
//! - Greedy sampling always returns the argmax token
//! - Deterministic seeded sampling is reproducible
//! - Repetition penalty reduces repeated tokens' probability
//! - Sampled token index is always within vocab range

use bitnet_sampling::{SamplingConfig, SamplingStrategy, greedy_sample};
use proptest::prelude::*;

proptest! {
    /// Greedy sampling always returns the token with the maximum logit.
    #[test]
    fn greedy_returns_argmax(logits in prop::collection::vec(-10.0f32..10.0f32, 2..100)) {
        let result = greedy_sample(&logits).unwrap();
        let expected = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap();
        prop_assert_eq!(result, expected);
    }

    /// Sampled token is always within the vocab range.
    #[test]
    fn sampled_token_in_range(
        logits in prop::collection::vec(-5.0f32..5.0f32, 2..200),
        temperature in 0.01f32..2.0f32,
        seed in any::<u64>()
    ) {
        let config = SamplingConfig {
            temperature,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(seed),
        };
        let mut strategy = SamplingStrategy::new(config);
        let token = strategy.sample(&logits, &[]).unwrap();
        prop_assert!((token as usize) < logits.len(), "Token {} out of range {}", token, logits.len());
    }

    /// Seeded deterministic sampling produces the same result every time.
    #[test]
    fn seeded_sampling_is_deterministic(
        logits in prop::collection::vec(-3.0f32..3.0f32, 2..50),
        seed in any::<u64>()
    ) {
        let make_config = || SamplingConfig {
            temperature: 0.7,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(seed),
        };
        let t1 = SamplingStrategy::new(make_config()).sample(&logits, &[]).unwrap();
        let t2 = SamplingStrategy::new(make_config()).sample(&logits, &[]).unwrap();
        prop_assert_eq!(t1, t2, "Same seed produced different tokens");
    }

    /// Temperature=0 (greedy) produces the same result as greedy_sample().
    #[test]
    fn zero_temperature_matches_greedy(logits in prop::collection::vec(-5.0f32..5.0f32, 2..50)) {
        let greedy = greedy_sample(&logits).unwrap();
        let config = SamplingConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(42),
        };
        let mut strategy = SamplingStrategy::new(config);
        let sampled = strategy.sample(&logits, &[]).unwrap();
        prop_assert_eq!(greedy, sampled);
    }
}

#[test]
fn greedy_on_single_token() {
    let logits = vec![1.0f32];
    assert_eq!(greedy_sample(&logits).unwrap(), 0);
}

#[test]
fn snapshot_default_config() {
    let config = SamplingConfig::default();
    insta::assert_debug_snapshot!("sampling_config_default", config);
}
