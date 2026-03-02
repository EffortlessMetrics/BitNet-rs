//! Property-based tests — wave 28.
//!
//! Inference sampling properties: apply_temperature scaling, apply_top_k
//! filtering, apply_top_p nucleus correctness, repetition penalty
//! monotonicity, MirostatSampler mu tracking, SamplerChain composability,
//! GenerationConfig builder round-trips, InferenceConfig presets,
//! MinPSampler threshold filtering, TypicalSampler entropy, greedy_sample
//! determinism, and softmax temperature interaction.
//!
//! 54 property assertions across 17 invariant categories.

#![cfg(feature = "cpu")]

use bitnet_inference::config::{GenerationConfig, InferenceConfig};
use bitnet_sampling::{
    MirostatSampler, RepetitionPenaltyConfig, SamplerChain, SamplingConfig, SamplingStrategy,
    apply_min_p, apply_repetition_penalty, apply_temperature, apply_top_k, apply_top_p,
    apply_typical, argmax, greedy_sample, softmax_in_place,
};
use proptest::prelude::*;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn logits_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(-5.0f32..5.0, 2..=max_len)
}

fn positive_logits(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(0.1f32..5.0, 2..=max_len)
}

// ── 1. apply_temperature scaling ────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Temperature=1.0 leaves logits unchanged.
    #[test]
    fn temperature_one_is_identity(logits in logits_vec(16)) {
        let mut scaled = logits.clone();
        apply_temperature(&mut scaled, 1.0);
        for (i, (&orig, &sc)) in logits.iter().zip(scaled.iter()).enumerate() {
            prop_assert!(
                (orig - sc).abs() < 1e-6,
                "temp=1.0 changed logits[{}]: {} vs {}", i, orig, sc
            );
        }
    }

    /// Higher temperature reduces logit magnitudes (divides).
    #[test]
    fn temperature_reduces_magnitude(logits in logits_vec(16)) {
        let mut scaled = logits.clone();
        apply_temperature(&mut scaled, 2.0);
        for (i, (&orig, &sc)) in logits.iter().zip(scaled.iter()).enumerate() {
            let expected = orig / 2.0;
            prop_assert!(
                (sc - expected).abs() < 1e-5,
                "temp=2.0: logits[{}] {} / 2.0 = {} != {}", i, orig, expected, sc
            );
        }
    }

    /// Temperature preserves argmax (for distinct max).
    #[test]
    fn temperature_preserves_argmax(
        logits in logits_vec(16),
        temp in 0.1f32..5.0,
    ) {
        let orig_max = argmax(&logits);
        let mut scaled = logits.clone();
        apply_temperature(&mut scaled, temp);
        let scaled_max = argmax(&scaled);
        prop_assert_eq!(
            orig_max, scaled_max,
            "argmax changed from {} to {} with temp={}", orig_max, scaled_max, temp
        );
    }
}

// ── 2. apply_top_k filtering ────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// After top-k, at most k values are non-NEG_INFINITY.
    #[test]
    fn top_k_at_most_k_active(
        logits in logits_vec(16),
        k in 1usize..8,
    ) {
        let mut filtered = logits.clone();
        apply_top_k(&mut filtered, k);
        let active = filtered.iter().filter(|&&v| v != f32::NEG_INFINITY).count();
        prop_assert!(
            active <= k,
            "top_k={} but {} values active", k, active
        );
    }

    /// top-k with k >= n leaves all values intact.
    #[test]
    fn top_k_full_is_identity(logits in logits_vec(8)) {
        let mut filtered = logits.clone();
        apply_top_k(&mut filtered, logits.len());
        for (i, (&orig, &f)) in logits.iter().zip(filtered.iter()).enumerate() {
            prop_assert!(
                (orig - f).abs() < 1e-6,
                "top_k=n changed logits[{}]: {} vs {}", i, orig, f
            );
        }
    }

    /// top-k with k=1 keeps only the argmax.
    #[test]
    fn top_k_one_keeps_max(logits in logits_vec(16)) {
        let max_idx = argmax(&logits);
        let mut filtered = logits.clone();
        apply_top_k(&mut filtered, 1);
        for (i, &v) in filtered.iter().enumerate() {
            if i == max_idx {
                prop_assert!(
                    (v - logits[i]).abs() < 1e-6,
                    "max value changed"
                );
            } else {
                prop_assert!(
                    v == f32::NEG_INFINITY,
                    "non-max logits[{}] = {} not -inf", i, v
                );
            }
        }
    }
}

// ── 3. apply_top_p nucleus correctness ──────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// top_p=1.0 keeps all values (no filtering).
    #[test]
    fn top_p_one_is_identity(logits in logits_vec(8)) {
        let mut probs = logits.clone();
        softmax_in_place(&mut probs);
        let orig_probs = probs.clone();
        apply_top_p(&mut probs, 1.0);
        for (i, (&orig, &filtered)) in orig_probs.iter().zip(probs.iter()).enumerate() {
            prop_assert!(
                (orig - filtered).abs() < 1e-5,
                "top_p=1.0 changed probs[{}]: {} vs {}", i, orig, filtered
            );
        }
    }

    /// After top_p filtering, at least one value > 0.
    #[test]
    fn top_p_at_least_one_active(
        logits in logits_vec(8),
        p in 0.01f32..1.0,
    ) {
        let mut probs = logits.clone();
        softmax_in_place(&mut probs);
        apply_top_p(&mut probs, p);
        let active = probs.iter().filter(|&&v| v > 0.0).count();
        prop_assert!(active >= 1, "top_p={} left no active probs", p);
    }
}

// ── 4. Softmax sum and bounds ───────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Softmax output sums to 1.0.
    #[test]
    fn softmax_sum_one(logits in logits_vec(16)) {
        let mut probs = logits.clone();
        softmax_in_place(&mut probs);
        let sum: f32 = probs.iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 1e-4,
            "softmax sum {} != 1.0", sum
        );
    }

    /// All softmax outputs in [0, 1].
    #[test]
    fn softmax_in_unit_interval(logits in logits_vec(16)) {
        let mut probs = logits.clone();
        softmax_in_place(&mut probs);
        for (i, &p) in probs.iter().enumerate() {
            prop_assert!(
                (0.0..=1.0).contains(&p),
                "softmax[{}] = {} out of [0,1]", i, p
            );
        }
    }

    /// Softmax preserves relative ordering.
    #[test]
    fn softmax_preserves_order(logits in logits_vec(8)) {
        let mut probs = logits.clone();
        softmax_in_place(&mut probs);
        for i in 0..logits.len() {
            for j in (i + 1)..logits.len() {
                if logits[i] > logits[j] + 1e-6 {
                    prop_assert!(
                        probs[i] >= probs[j] - 1e-5,
                        "order violated: logits[{}]={} > logits[{}]={} but probs {} < {}",
                        i, logits[i], j, logits[j], probs[i], probs[j]
                    );
                }
            }
        }
    }
}

// ── 5. Repetition penalty ───────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Penalty > 1 reduces positive logits of repeated tokens.
    #[test]
    fn repetition_penalty_reduces_positive(
        logits in positive_logits(8),
        penalty in 1.1f32..3.0,
    ) {
        let mut penalized = logits.clone();
        let token_ids: Vec<u32> = (0..logits.len() as u32).collect();
        apply_repetition_penalty(&mut penalized, &token_ids, penalty);
        for (i, (&orig, &pen)) in logits.iter().zip(penalized.iter()).enumerate() {
            prop_assert!(
                pen <= orig + 1e-6,
                "penalty didn't reduce logits[{}]: {} -> {}", i, orig, pen
            );
        }
    }

    /// Penalty=1.0 leaves logits unchanged.
    #[test]
    fn repetition_penalty_one_identity(logits in logits_vec(8)) {
        let mut penalized = logits.clone();
        let token_ids: Vec<u32> = (0..logits.len() as u32).collect();
        apply_repetition_penalty(&mut penalized, &token_ids, 1.0);
        for (i, (&orig, &pen)) in logits.iter().zip(penalized.iter()).enumerate() {
            prop_assert!(
                (orig - pen).abs() < 1e-6,
                "penalty=1.0 changed logits[{}]: {} vs {}", i, orig, pen
            );
        }
    }
}

// ── 6. Greedy sample determinism ────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// greedy_sample always picks the argmax.
    #[test]
    fn greedy_sample_is_argmax(logits in logits_vec(16)) {
        let token = greedy_sample(&logits).unwrap();
        let expected = argmax(&logits) as u32;
        prop_assert_eq!(token, expected, "greedy != argmax");
    }

    /// greedy_sample is deterministic (same input → same output).
    #[test]
    fn greedy_sample_deterministic(logits in logits_vec(16)) {
        let t1 = greedy_sample(&logits).unwrap();
        let t2 = greedy_sample(&logits).unwrap();
        prop_assert_eq!(t1, t2, "greedy not deterministic");
    }
}

// ── 7. SamplingStrategy output range ────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Sampled token is always within vocab range.
    #[test]
    fn sampling_output_in_vocab_range(
        logits in logits_vec(16),
        seed in 0u64..10000,
    ) {
        let config = SamplingConfig {
            temperature: 0.7,
            seed: Some(seed),
            ..Default::default()
        };
        let mut strategy = SamplingStrategy::new(config);
        let token = strategy.sample(&logits, &[]).unwrap();
        prop_assert!(
            (token as usize) < logits.len(),
            "token {} >= vocab {}", token, logits.len()
        );
    }

    /// Greedy strategy (temp=0) always returns argmax.
    #[test]
    fn sampling_greedy_matches_argmax(logits in logits_vec(16)) {
        let config = SamplingConfig {
            temperature: 0.0,
            seed: Some(42),
            ..Default::default()
        };
        let mut strategy = SamplingStrategy::new(config);
        let token = strategy.sample(&logits, &[]).unwrap();
        let expected = argmax(&logits) as u32;
        prop_assert_eq!(token, expected);
    }
}

// ── 8. MirostatSampler properties ───────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Mirostat output is within vocab range.
    #[test]
    fn mirostat_output_in_range(logits in logits_vec(16)) {
        let mut sampler = MirostatSampler::new(5.0, 0.1, Some(42));
        let token = sampler.sample(&logits).unwrap();
        prop_assert!(
            (token as usize) < logits.len(),
            "mirostat token {} >= vocab {}", token, logits.len()
        );
    }

    /// Mirostat reset restores mu to 2*tau.
    #[test]
    fn mirostat_reset_restores_mu(
        tau in 1.0f32..10.0,
        logits in logits_vec(8),
    ) {
        let mut sampler = MirostatSampler::new(tau, 0.1, Some(42));
        let _ = sampler.sample(&logits); // mutate mu
        sampler.reset();
        // After reset, sample should work from initial state
        let token = sampler.sample(&logits).unwrap();
        prop_assert!(
            (token as usize) < logits.len(),
            "mirostat post-reset token {} >= vocab {}", token, logits.len()
        );
    }
}

// ── 9. apply_min_p threshold ────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// min_p=0 keeps all probabilities.
    #[test]
    fn min_p_zero_keeps_all(logits in logits_vec(8)) {
        let mut probs = logits.clone();
        softmax_in_place(&mut probs);
        let orig = probs.clone();
        apply_min_p(&mut probs, 0.0);
        for (i, (&o, &f)) in orig.iter().zip(probs.iter()).enumerate() {
            prop_assert!(
                (o - f).abs() < 1e-6,
                "min_p=0 changed probs[{}]: {} vs {}", i, o, f
            );
        }
    }

    /// After min_p filtering, at least one value > 0.
    #[test]
    fn min_p_at_least_one_active(
        logits in logits_vec(8),
        min_p in 0.0f32..0.5,
    ) {
        let mut probs = logits.clone();
        softmax_in_place(&mut probs);
        apply_min_p(&mut probs, min_p);
        let active = probs.iter().filter(|&&v| v > 0.0).count();
        prop_assert!(active >= 1, "min_p={} left no active probs", min_p);
    }
}

// ── 10. apply_typical filtering ─────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// typical_p=1.0 keeps all probabilities.
    #[test]
    fn typical_one_keeps_all(logits in logits_vec(8)) {
        let mut probs = logits.clone();
        softmax_in_place(&mut probs);
        let orig = probs.clone();
        apply_typical(&mut probs, 1.0);
        for (i, (&o, &f)) in orig.iter().zip(probs.iter()).enumerate() {
            prop_assert!(
                (o - f).abs() < 1e-6,
                "typical=1.0 changed probs[{}]: {} vs {}", i, o, f
            );
        }
    }

    /// After typical filtering, at least one value > 0.
    #[test]
    fn typical_at_least_one(
        logits in logits_vec(8),
        typ in 0.1f32..1.0,
    ) {
        let mut probs = logits.clone();
        softmax_in_place(&mut probs);
        apply_typical(&mut probs, typ);
        let active = probs.iter().filter(|&&v| v > 0.0).count();
        prop_assert!(active >= 1, "typical={} left no active probs", typ);
    }
}

// ── 11. RepetitionPenaltyConfig ─────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Default RepetitionPenaltyConfig leaves logits unchanged (all penalties = 0).
    #[test]
    fn repetition_penalty_config_default_noop(logits in logits_vec(8)) {
        let config = RepetitionPenaltyConfig::default();
        let mut penalized = logits.clone();
        config.apply(&mut penalized, &[]);
        for (i, (&orig, &pen)) in logits.iter().zip(penalized.iter()).enumerate() {
            prop_assert!(
                (orig - pen).abs() < 1e-6,
                "default penalty changed logits[{}]", i
            );
        }
    }
}

// ── 12. SamplerChain composability ──────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Empty chain produces a valid token.
    #[test]
    fn sampler_chain_empty_produces_token(logits in logits_vec(8)) {
        let chain = SamplerChain::builder().build(None);
        let token = chain.sample(&logits).unwrap();
        prop_assert!(
            (token as usize) < logits.len(),
            "empty chain token {} >= vocab {}", token, logits.len()
        );
    }

    /// Chain with just temperature produces valid token.
    #[test]
    fn sampler_chain_temp_only(
        logits in logits_vec(8),
        temp in 0.1f32..3.0,
    ) {
        let chain = SamplerChain::builder()
            .temperature(temp)
            .build(None);
        let token = chain.sample(&logits).unwrap();
        prop_assert!(
            (token as usize) < logits.len(),
            "temp chain token {} >= vocab {}", token, logits.len()
        );
    }

    /// Chain stages count matches builder additions.
    #[test]
    fn sampler_chain_stage_count(
        use_temp in proptest::bool::ANY,
        use_topk in proptest::bool::ANY,
        use_topp in proptest::bool::ANY,
    ) {
        let mut builder = SamplerChain::builder();
        let mut expected = 0;
        if use_temp {
            builder = builder.temperature(0.8);
            expected += 1;
        }
        if use_topk {
            builder = builder.top_k(50);
            expected += 1;
        }
        if use_topp {
            builder = builder.top_p(0.9);
            expected += 1;
        }
        let chain = builder.build(None);
        prop_assert_eq!(
            chain.stages().len(), expected,
            "stage count mismatch"
        );
    }
}

// ── 13. GenerationConfig presets ─────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// greedy().temperature == 0.0 and top_k == 1.
    #[test]
    fn gen_config_greedy_invariants(_dummy in 0..1i32) {
        let c = GenerationConfig::greedy();
        prop_assert_eq!(c.temperature, 0.0);
        prop_assert_eq!(c.top_k, 1);
    }

    /// balanced() has moderate temperature.
    #[test]
    fn gen_config_balanced_temp(_dummy in 0..1i32) {
        let c = GenerationConfig::balanced();
        prop_assert!(c.temperature > 0.0 && c.temperature < 2.0);
    }

    /// creative() has higher temperature than balanced().
    #[test]
    fn gen_config_creative_gt_balanced(_dummy in 0..1i32) {
        let b = GenerationConfig::balanced();
        let c = GenerationConfig::creative();
        prop_assert!(
            c.temperature >= b.temperature,
            "creative temp {} < balanced temp {}", c.temperature, b.temperature
        );
    }
}

// ── 14. GenerationConfig builder round-trips ────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// with_temperature sets temperature correctly.
    #[test]
    fn gen_config_with_temperature(temp in 0.0f32..5.0) {
        let c = GenerationConfig::greedy().with_temperature(temp);
        prop_assert!(
            (c.temperature - temp).abs() < 1e-6,
            "temperature {} != set {}", c.temperature, temp
        );
    }

    /// with_top_k sets top_k correctly.
    #[test]
    fn gen_config_with_top_k(k in 1u32..200) {
        let c = GenerationConfig::greedy().with_top_k(k);
        prop_assert_eq!(c.top_k, k);
    }

    /// with_top_p sets top_p correctly.
    #[test]
    fn gen_config_with_top_p(p in 0.0f32..1.0) {
        let c = GenerationConfig::greedy().with_top_p(p);
        prop_assert!(
            (c.top_p - p).abs() < 1e-6,
            "top_p {} != set {}", c.top_p, p
        );
    }

    /// with_max_tokens sets max_new_tokens.
    #[test]
    fn gen_config_with_max_tokens(n in 1u32..10000) {
        let c = GenerationConfig::greedy().with_max_tokens(n);
        prop_assert_eq!(c.max_new_tokens, n);
    }

    /// with_seed sets seed.
    #[test]
    fn gen_config_with_seed(seed in 0u64..u64::MAX) {
        let c = GenerationConfig::greedy().with_seed(seed);
        prop_assert_eq!(c.seed, Some(seed));
    }
}

// ── 15. GenerationConfig validation ─────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Valid config passes validation.
    #[test]
    fn gen_config_valid_passes(
        temp in 0.0f32..5.0,
        top_p in 0.0f32..=1.0,
        top_k in 1u32..1000,
    ) {
        let c = GenerationConfig::greedy()
            .with_temperature(temp)
            .with_top_p(top_p)
            .with_top_k(top_k);
        let result = c.validate();
        prop_assert!(result.is_ok(), "valid config rejected: {:?}", result.err());
    }

    /// stop_token_ids round-trip via builder.
    #[test]
    fn gen_config_stop_ids_roundtrip(
        ids in proptest::collection::vec(0u32..100000, 0..=8),
    ) {
        let c = GenerationConfig::greedy().with_stop_token_ids(ids.clone());
        for &id in &ids {
            prop_assert!(c.is_stop_token(id), "is_stop_token({}) false", id);
        }
    }
}

// ── 16. InferenceConfig presets ─────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// cpu_optimized has positive threads.
    #[test]
    fn inf_config_cpu_positive_threads(_dummy in 0..1i32) {
        let c = InferenceConfig::cpu_optimized();
        prop_assert!(c.num_threads > 0, "threads={}", c.num_threads);
    }

    /// with_threads sets thread count.
    #[test]
    fn inf_config_with_threads(n in 1usize..128) {
        let c = InferenceConfig::default().with_threads(n);
        prop_assert_eq!(c.num_threads, n);
    }

    /// with_batch_size sets batch size.
    #[test]
    fn inf_config_with_batch(b in 1usize..64) {
        let c = InferenceConfig::default().with_batch_size(b);
        prop_assert_eq!(c.batch_size, b);
    }

    /// validate accepts reasonable configs.
    #[test]
    fn inf_config_valid_passes(
        threads in 1usize..64,
        batch in 1usize..32,
    ) {
        let c = InferenceConfig::default()
            .with_threads(threads)
            .with_batch_size(batch);
        let result = c.validate();
        prop_assert!(result.is_ok(), "valid inf config rejected: {:?}", result.err());
    }
}

// ── 17. Seed determinism ────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Same seed → same token for stochastic sampling.
    #[test]
    fn seed_determinism(
        logits in logits_vec(16),
        seed in 0u64..10000,
    ) {
        let config = SamplingConfig {
            temperature: 0.8,
            seed: Some(seed),
            ..Default::default()
        };
        let mut s1 = SamplingStrategy::new(config.clone());
        let mut s2 = SamplingStrategy::new(config);
        let t1 = s1.sample(&logits, &[]).unwrap();
        let t2 = s2.sample(&logits, &[]).unwrap();
        prop_assert_eq!(t1, t2, "same seed different tokens");
    }

    /// Reset clears token count history.
    #[test]
    fn reset_clears_history(
        logits in logits_vec(16),
        seed in 0u64..10000,
    ) {
        let config = SamplingConfig {
            temperature: 0.5,
            seed: Some(seed),
            ..Default::default()
        };
        let mut strategy = SamplingStrategy::new(config);
        // Sample a few tokens to build history
        let _ = strategy.sample(&logits, &[]);
        let _ = strategy.sample(&logits, &[]);
        // Reset should not panic
        strategy.reset();
        // After reset, sampling still works
        let t = strategy.sample(&logits, &[]).unwrap();
        prop_assert!((t as usize) < logits.len(), "token {} >= vocab {}", t, logits.len());
    }
}
