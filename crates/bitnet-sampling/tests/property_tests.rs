//! Property-based tests for bitnet-sampling.
//!
//! Key invariants:
//! - Greedy sampling always returns the argmax token
//! - Deterministic seeded sampling is reproducible
//! - Repetition penalty reduces repeated tokens' probability
//! - Sampled token index is always within vocab range
//! - top_k=1 always returns the same token as greedy
//! - Higher temperature produces a more uniform distribution (lower max probability)
//! - Different seeds always return valid tokens (no panics)
//! - Multi-step sampling always stays within vocab bounds

use bitnet_sampling::{
    SamplingConfig, SamplingStrategy, apply_temperature, greedy_sample, softmax_in_place,
};
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

proptest! {
    /// top_k=1 always returns the same token as greedy regardless of temperature.
    ///
    /// Rationale: temperature scaling multiplies logits by a positive constant
    /// (1/temp), which preserves ordering.  After apply_top_k(1) only the argmax
    /// survives; softmax maps it to probability 1.0; the sampler picks it
    /// deterministically.
    #[test]
    fn top_k_one_always_returns_argmax(
        logits in prop::collection::vec(-10.0f32..10.0f32, 2..64),
        temperature in 0.1f32..5.0f32,
        seed in any::<u64>(),
    ) {
        let greedy = greedy_sample(&logits).unwrap();
        let config = SamplingConfig {
            temperature,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(seed),
        };
        let mut strategy = SamplingStrategy::new(config);
        let token = strategy.sample(&logits, &[]).unwrap();
        prop_assert_eq!(token, greedy, "top_k=1 must always return the greedy argmax");
    }

    /// Repetition penalty > 1 can displace the argmax when the penalty is large
    /// enough relative to the gap between the top two logits.
    ///
    /// Construction: token 0 has logit `second * 1.5`, penalty=2.0.
    /// After penalty: token 0 becomes `second * 1.5 / 2.0 = second * 0.75 < second`.
    /// Temperature=0 (greedy) must therefore choose token 1.
    #[test]
    fn repetition_penalty_displaces_argmax(
        second_logit in 1.0f32..8.0f32,
    ) {
        let first_logit = second_logit * 1.5;
        let logits = vec![first_logit, second_logit, second_logit * 0.5];
        let config = SamplingConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 2.0,
            seed: Some(0),
        };
        let mut strategy = SamplingStrategy::new(config);
        // Context says token 0 has been seen once; penalty halves its logit.
        let token = strategy.sample(&logits, &[0]).unwrap();
        prop_assert_eq!(
            token,
            1u32,
            "penalised argmax (logit={}) should be displaced by token 1 (logit={})",
            first_logit,
            second_logit
        );
    }

    /// Higher temperature yields a less-peaked distribution (lower max probability).
    ///
    /// `apply_temperature` scales logits by `1/T`; larger T → smaller scaling →
    /// softer softmax distribution.  So max_prob(high_T) ≤ max_prob(low_T).
    #[test]
    fn higher_temperature_lowers_max_probability(
        logits in prop::collection::vec(0.0f32..10.0f32, 2..50),
        low_temp in 0.1f32..0.8f32,
        temp_delta in 0.3f32..4.0f32,
    ) {
        let high_temp = low_temp + temp_delta;

        let mut low_probs = logits.clone();
        apply_temperature(&mut low_probs, low_temp);
        softmax_in_place(&mut low_probs);
        let low_max = low_probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let mut high_probs = logits.clone();
        apply_temperature(&mut high_probs, high_temp);
        softmax_in_place(&mut high_probs);
        let high_max = high_probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        prop_assert!(
            high_max <= low_max + 1e-4,
            "high_temp={} max_prob={} should be ≤ low_temp={} max_prob={}",
            high_temp,
            high_max,
            low_temp,
            low_max
        );
    }

    /// Two different seeds produce valid tokens without panicking.
    ///
    /// This is an anti-flakiness / no-panic check: we cannot assert the tokens
    /// differ (they may coincidentally be equal), but both must be in range.
    #[test]
    fn different_seeds_produce_valid_tokens(
        logits in prop::collection::vec(-5.0f32..5.0f32, 2..50),
        seed1 in any::<u64>(),
        seed2 in any::<u64>(),
    ) {
        let make_config = |seed| SamplingConfig {
            temperature: 0.7,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(seed),
        };
        let t1 = SamplingStrategy::new(make_config(seed1)).sample(&logits, &[]).unwrap();
        let t2 = SamplingStrategy::new(make_config(seed2)).sample(&logits, &[]).unwrap();
        prop_assert!((t1 as usize) < logits.len(), "seed1 token {} out of range", t1);
        prop_assert!((t2 as usize) < logits.len(), "seed2 token {} out of range", t2);
    }

    /// Multi-step sampling: every token across N steps remains within vocab bounds.
    #[test]
    fn multi_step_sampling_stays_in_range(
        logits in prop::collection::vec(-3.0f32..3.0f32, 2..30),
        temperature in 0.5f32..2.0f32,
        seed in any::<u64>(),
        steps in 2usize..8,
    ) {
        let config = SamplingConfig {
            temperature,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(seed),
        };
        let mut strategy = SamplingStrategy::new(config);
        let mut context: Vec<u32> = Vec::new();
        for _ in 0..steps {
            let token = strategy.sample(&logits, &context).unwrap();
            prop_assert!(
                (token as usize) < logits.len(),
                "token {} out of range {}",
                token,
                logits.len()
            );
            context.push(token);
        }
    }

    /// Repetition penalty with no prior context leaves all logits unchanged.
    ///
    /// If `context_tokens` is empty, no penalty should be applied and the output
    /// equals the penalty-free result (greedy path for clarity).
    #[test]
    fn repetition_penalty_with_empty_context_leaves_result_unchanged(
        logits in prop::collection::vec(-5.0f32..5.0f32, 2..50),
        penalty in 1.01f32..3.0f32,
        seed in any::<u64>(),
    ) {
        let make_config = |rep_pen| SamplingConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: rep_pen,
            seed: Some(seed),
        };
        let t_no_pen = SamplingStrategy::new(make_config(1.0)).sample(&logits, &[]).unwrap();
        let t_with_pen = SamplingStrategy::new(make_config(penalty)).sample(&logits, &[]).unwrap();
        prop_assert_eq!(
            t_no_pen,
            t_with_pen,
            "empty context → penalty has no effect; expected same token"
        );
    }
}

#[test]
fn greedy_on_single_token() {
    let logits = vec![1.0f32];
    assert_eq!(greedy_sample(&logits).unwrap(), 0);
}

/// reset() clears internal state without breaking subsequent sampling.
#[test]
fn reset_leaves_strategy_functional() {
    let logits = vec![0.5f32, 1.0, 2.0, 0.3];
    let config = SamplingConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(7),
    };
    let mut strategy = SamplingStrategy::new(config);
    let before = strategy.sample(&logits, &[]).unwrap();
    strategy.reset();
    let after = strategy.sample(&logits, &[]).unwrap();
    // Greedy output should be stable across reset
    assert_eq!(before, after);
    assert!((after as usize) < logits.len());
}

#[test]
fn snapshot_default_config() {
    let config = SamplingConfig::default();
    insta::assert_debug_snapshot!("sampling_config_default", config);
}
