//! Wave 13 property tests: sampling strategy invariants for temperature
//! scaling, top-k/top-p filtering, repetition penalty, min-p, typical,
//! mirostat, and sampler chain composition.
//!
//! Key invariants tested (15 properties):
//! - Temperature: higher temperature increases entropy of softmax output
//! - Top-k: after apply_top_k, at most k entries are finite
//! - Top-p: surviving probabilities sum to >= top_p
//! - Repetition penalty: penalty > 1 reduces positive logit of penalized token
//! - Min-p: all surviving probabilities >= min_p * max_prob
//! - Mirostat: sample output is always a valid token index
//! - SamplerChain: builder produces correct stage count, sample in range
//! - RepetitionPenaltyConfig: frequency penalty reduces logit proportionally
//! - SamplingStrategy: greedy with same seed is deterministic, output in range

use bitnet_logits::{
    apply_min_p, apply_repetition_penalty, apply_temperature, apply_top_k, apply_top_p, argmax,
    softmax_in_place,
};
use bitnet_sampling::{
    MinPSampler, MirostatSampler, RepetitionPenaltyConfig, SamplerChain, SamplingConfig,
    SamplingStrategy, TypicalSampler,
};
use proptest::prelude::*;

// -------------------------------------------------------------------
// Strategy helpers
// -------------------------------------------------------------------

/// Non-empty f32 vector with finite values in [-5, 5] for logits.
fn logits_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-5.0f32..5.0f32, 2..=max_len)
}

/// Probability-like vector (non-negative, sums roughly to 1).
fn prob_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(0.01f32..1.0f32, 2..=max_len).prop_map(|v| {
        let sum: f32 = v.iter().sum();
        v.into_iter().map(|x| x / sum).collect::<Vec<f32>>()
    })
}

// ===================================================================
// 1. Temperature scaling: higher temp → more uniform distribution
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Higher temperature produces a more uniform (higher entropy) distribution.
    #[test]
    fn prop_higher_temp_higher_entropy(
        logits in logits_vec(32),
    ) {
        let mut low = logits.clone();
        apply_temperature(&mut low, 0.5);
        softmax_in_place(&mut low);
        let entropy_low: f32 = low.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();

        let mut high = logits.clone();
        apply_temperature(&mut high, 2.0);
        softmax_in_place(&mut high);
        let entropy_high: f32 = high.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();

        prop_assert!(
            entropy_high >= entropy_low - 1e-5,
            "entropy_high={entropy_high} < entropy_low={entropy_low}"
        );
    }

    /// Temperature 1.0 is a no-op on logits.
    #[test]
    fn prop_temperature_one_noop(logits in logits_vec(32)) {
        let mut buf = logits.clone();
        apply_temperature(&mut buf, 1.0);
        for (i, (&orig, &after)) in logits.iter().zip(buf.iter()).enumerate() {
            prop_assert!(
                (orig - after).abs() < 1e-6,
                "index {i}: orig={orig}, after={after}"
            );
        }
    }
}

// ===================================================================
// 2. Top-k: at most k entries remain finite
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// After apply_top_k, the number of finite entries is at most k.
    #[test]
    fn prop_top_k_keeps_at_most_k(
        logits in logits_vec(32),
        k in 1usize..=10,
    ) {
        let mut buf = logits.clone();
        let kept = apply_top_k(&mut buf, k);
        prop_assert!(
            kept <= k,
            "kept={kept} > k={k}"
        );
        let finite_count = buf.iter().filter(|x| x.is_finite()).count();
        prop_assert!(
            finite_count <= k,
            "finite_count={finite_count} > k={k}"
        );
    }

    /// Top-k with k >= len is a no-op (all entries preserved).
    #[test]
    fn prop_top_k_large_k_noop(logits in logits_vec(16)) {
        let mut buf = logits.clone();
        let kept = apply_top_k(&mut buf, logits.len() + 10);
        prop_assert_eq!(kept, logits.len());
        for (i, (&orig, &after)) in logits.iter().zip(buf.iter()).enumerate() {
            prop_assert!(
                (orig - after).abs() < 1e-6,
                "index {i}: changed from {orig} to {after}"
            );
        }
    }
}

// ===================================================================
// 3. Top-p: surviving mass >= top_p
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// After apply_top_p, the sum of surviving probabilities >= top_p.
    #[test]
    fn prop_top_p_surviving_mass(
        probs in prob_vec(16),
        top_p in 0.1f32..0.99f32,
    ) {
        let mut buf = probs.clone();
        apply_top_p(&mut buf, top_p);
        let surviving_sum: f32 = buf.iter().sum();
        prop_assert!(
            surviving_sum >= top_p - 1e-5,
            "surviving_sum={surviving_sum} < top_p={top_p}"
        );
    }

    /// apply_top_p with top_p=1.0 is a no-op.
    #[test]
    fn prop_top_p_one_noop(probs in prob_vec(16)) {
        let mut buf = probs.clone();
        apply_top_p(&mut buf, 1.0);
        for (i, (&orig, &after)) in probs.iter().zip(buf.iter()).enumerate() {
            prop_assert!(
                (orig - after).abs() < 1e-6,
                "index {i}: changed from {orig} to {after}"
            );
        }
    }
}

// ===================================================================
// 4. Repetition penalty: penalized tokens get reduced logits
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// apply_repetition_penalty with penalty > 1 reduces positive logits.
    #[test]
    fn prop_repetition_penalty_reduces_positive(
        logits in prop::collection::vec(0.1f32..5.0f32, 4..=16),
        penalty in 1.1f32..3.0f32,
    ) {
        let token_id = 0u32;
        let orig = logits[0];
        let mut buf = logits.clone();
        apply_repetition_penalty(&mut buf, &[token_id], penalty);
        prop_assert!(
            buf[0] < orig + 1e-6,
            "penalized logit {} not reduced from original {}", buf[0], orig
        );
    }

    /// RepetitionPenaltyConfig frequency penalty is proportional to count.
    #[test]
    fn prop_rep_config_frequency_proportional(
        base_logit in 1.0f32..5.0f32,
        freq_penalty in 0.1f32..1.0f32,
    ) {
        let config = RepetitionPenaltyConfig {
            frequency_penalty: freq_penalty,
            presence_penalty: 0.0,
            count_penalty: 1.0,
        };
        let mut logits1 = vec![base_logit; 4];
        let mut logits2 = vec![base_logit; 4];
        config.apply(&mut logits1, &[(0, 1)]);
        config.apply(&mut logits2, &[(0, 2)]);
        prop_assert!(
            logits2[0] < logits1[0] + 1e-6,
            "count=2 logit {} not less than count=1 logit {}", logits2[0], logits1[0]
        );
    }
}

// ===================================================================
// 5. Min-p: surviving probabilities above threshold
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// After apply_min_p, all surviving probabilities >= min_p * max_prob.
    #[test]
    fn prop_min_p_threshold_respected(
        probs in prob_vec(16),
        min_p in 0.01f32..0.5f32,
    ) {
        let max_prob = probs.iter().copied().fold(0.0f32, f32::max);
        let threshold = min_p * max_prob;
        let mut buf = probs.clone();
        apply_min_p(&mut buf, min_p);
        for (i, &p) in buf.iter().enumerate() {
            if p > 0.0 {
                prop_assert!(
                    p >= threshold - 1e-6,
                    "probs[{i}]={p} < threshold={threshold}"
                );
            }
        }
    }

    /// MinPSampler.filter zeroes the same entries as apply_min_p.
    #[test]
    fn prop_min_p_sampler_matches_raw(
        probs in prob_vec(16),
        min_p in 0.01f32..0.5f32,
    ) {
        let sampler = MinPSampler::new(min_p);
        let mut via_sampler = probs.clone();
        sampler.filter(&mut via_sampler);
        let mut via_raw = probs.clone();
        apply_min_p(&mut via_raw, min_p);
        for (i, (&a, &b)) in via_sampler.iter().zip(via_raw.iter()).enumerate() {
            prop_assert!(
                (a - b).abs() < 1e-6,
                "index {i}: sampler={a}, raw={b}"
            );
        }
    }
}

// ===================================================================
// 6. Mirostat: output is always a valid token index
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// MirostatSampler always returns an index within vocab size.
    #[test]
    fn prop_mirostat_output_in_range(
        logits in logits_vec(32),
        tau in 1.0f32..10.0f32,
    ) {
        let mut sampler = MirostatSampler::new(tau, 0.1, Some(42));
        let token = sampler.sample(&logits).unwrap();
        prop_assert!(
            (token as usize) < logits.len(),
            "token={token} >= vocab_size={}", logits.len()
        );
    }

    /// MirostatSampler.reset restores mu to 2*tau.
    #[test]
    fn prop_mirostat_reset_restores_mu(
        tau in 1.0f32..10.0f32,
    ) {
        let mut sampler = MirostatSampler::new(tau, 0.1, Some(0));
        let logits = vec![1.0, 2.0, 0.5, -1.0];
        let _ = sampler.sample(&logits);
        let _ = sampler.sample(&logits);
        sampler.reset();
        prop_assert!(
            (sampler.mu - 2.0 * tau).abs() < 1e-6,
            "mu={} != 2*tau={}", sampler.mu, 2.0 * tau
        );
    }
}

// ===================================================================
// 7. TypicalSampler: at least one token survives
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// TypicalSampler.filter keeps at least one non-zero probability.
    #[test]
    fn prop_typical_keeps_at_least_one(
        probs in prob_vec(16),
        typical_p in 0.1f32..0.99f32,
    ) {
        let sampler = TypicalSampler::new(typical_p);
        let mut buf = probs.clone();
        sampler.filter(&mut buf);
        let nonzero = buf.iter().filter(|&&p| p > 0.0).count();
        prop_assert!(
            nonzero >= 1,
            "typical filter zeroed all probabilities"
        );
    }
}

// ===================================================================
// 8. SamplerChain: builder, sample in range
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// SamplerChain.sample always returns an index within vocab size.
    #[test]
    fn prop_sampler_chain_output_in_range(
        logits in logits_vec(32),
    ) {
        let chain = SamplerChain::builder()
            .temperature(0.8)
            .top_k(10)
            .top_p(0.9)
            .build(Some(42));
        let token = chain.sample(&logits).unwrap();
        prop_assert!(
            (token as usize) < logits.len(),
            "token={token} >= vocab_size={}", logits.len()
        );
    }

    /// SamplerChain builder accumulates stages correctly.
    #[test]
    fn prop_sampler_chain_stage_count(
        use_top_k in proptest::bool::ANY,
        use_min_p in proptest::bool::ANY,
    ) {
        let mut builder = SamplerChain::builder().temperature(0.7);
        let mut expected = 1; // temperature
        if use_top_k {
            builder = builder.top_k(10);
            expected += 1;
        }
        if use_min_p {
            builder = builder.min_p(0.05);
            expected += 1;
        }
        let chain = builder.build(Some(42));
        prop_assert_eq!(
            chain.stages().len(),
            expected,
            "expected {} stages", expected
        );
    }
}

// ===================================================================
// 9. SamplingStrategy: determinism, output range
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// SamplingStrategy with same seed produces same token.
    #[test]
    fn prop_sampling_strategy_deterministic_with_seed(
        logits in logits_vec(16),
        seed in 0u64..1000,
    ) {
        let config1 = SamplingConfig {
            temperature: 0.7,
            seed: Some(seed),
            ..Default::default()
        };
        let config2 = SamplingConfig {
            temperature: 0.7,
            seed: Some(seed),
            ..Default::default()
        };
        let mut s1 = SamplingStrategy::new(config1);
        let mut s2 = SamplingStrategy::new(config2);
        let t1 = s1.sample(&logits, &[]).unwrap();
        let t2 = s2.sample(&logits, &[]).unwrap();
        prop_assert_eq!(t1, t2, "same seed should produce same token");
    }

    /// SamplingStrategy output is always within vocab range.
    #[test]
    fn prop_sampling_strategy_output_in_range(
        logits in logits_vec(16),
    ) {
        let config = SamplingConfig {
            temperature: 0.7,
            seed: Some(42),
            ..Default::default()
        };
        let mut strategy = SamplingStrategy::new(config);
        let token = strategy.sample(&logits, &[]).unwrap();
        prop_assert!(
            (token as usize) < logits.len(),
            "token={token} >= vocab_size={}", logits.len()
        );
    }

    /// Greedy sampling (temp=0) always returns argmax.
    #[test]
    fn prop_greedy_returns_argmax(logits in logits_vec(16)) {
        let config = SamplingConfig {
            temperature: 0.0,
            seed: Some(0),
            ..Default::default()
        };
        let mut strategy = SamplingStrategy::new(config);
        let token = strategy.sample(&logits, &[]).unwrap();
        let expected = argmax(&logits) as u32;
        prop_assert_eq!(token, expected, "greedy should pick argmax");
    }
}
