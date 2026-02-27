//! Property-based tests for `bitnet-sampling` – new invariant coverage.
//!
//! Each test exercises a distinct invariant:
//!
//!   1. `apply_temperature(1.0)` is an identity transform.
//!   2. Top-K: at most K non-zero probabilities after filtering + softmax.
//!   3. Top-P: retained nucleus cumulative mass is >= p.
//!   4. Repetition penalty lowers the effective logit of penalised tokens.
//!   5. Same seed + same logits -> same sampled token (reproducibility).
//!   6. All-NEG_INFINITY-except-one -> that single token is always chosen.

use bitnet_sampling::{
    SamplingConfig, SamplingStrategy, apply_repetition_penalty, apply_temperature, apply_top_k,
    apply_top_p, softmax_in_place,
};
use proptest::prelude::*;

proptest! {
    #[test]
    fn temperature_one_is_identity(
        logits in prop::collection::vec(-20.0f32..20.0f32, 2..=64),
    ) {
        let mut scaled = logits.clone();
        apply_temperature(&mut scaled, 1.0);
        prop_assert_eq!(&scaled, &logits, "apply_temperature(1.0) must leave logits unchanged");

        let mut probs_orig = logits.clone();
        let mut probs_scaled = scaled.clone();
        softmax_in_place(&mut probs_orig);
        softmax_in_place(&mut probs_scaled);
        for (a, b) in probs_orig.iter().zip(probs_scaled.iter()) {
            prop_assert!(
                (a - b).abs() < 1e-5,
                "softmax probability {} vs {} differ after temp=1.0 no-op", a, b
            );
        }
    }

    #[test]
    fn top_k_at_most_k_nonzero_probs(
        logits in prop::collection::vec(-10.0f32..10.0f32, 2..=64),
        k in 1usize..=32,
    ) {
        let effective_k = k.min(logits.len());
        let mut buf = logits.clone();
        apply_top_k(&mut buf, effective_k);
        softmax_in_place(&mut buf);
        let nonzero = buf.iter().filter(|&&p| p > 0.0).count();
        prop_assert!(
            nonzero <= effective_k,
            "non-zero count {} exceeds top_k={}",
            nonzero,
            effective_k
        );
    }

    #[test]
    fn top_p_nucleus_covers_at_least_p_mass(
        logits in prop::collection::vec(-10.0f32..10.0f32, 2..=64),
        p in 0.05f32..0.99f32,
    ) {
        let mut probs = logits.clone();
        softmax_in_place(&mut probs);
        apply_top_p(&mut probs, p);
        let nucleus_mass: f32 = probs.iter().filter(|&&x| x > 0.0).sum();
        prop_assert!(
            nucleus_mass >= p - 1e-4,
            "nucleus mass {} < top_p={}", nucleus_mass, p
        );
    }

    #[test]
    fn repetition_penalty_lowers_penalised_token_logit(
        pos_logit in 0.01f32..20.0f32,
        neg_logit in -20.0f32..-0.01f32,
        penalty in 1.01f32..5.0f32,
    ) {
        let mut buf_pos = vec![pos_logit, 0.5f32];
        apply_repetition_penalty(&mut buf_pos, &[0u32], penalty);
        prop_assert!(
            buf_pos[0] < pos_logit,
            "positive logit {} should decrease after penalty {}; got {}",
            pos_logit, penalty, buf_pos[0]
        );

        let mut buf_neg = vec![neg_logit, 0.5f32];
        apply_repetition_penalty(&mut buf_neg, &[0u32], penalty);
        prop_assert!(
            buf_neg[0] < neg_logit,
            "negative logit {} should become more negative after penalty {}; got {}",
            neg_logit, penalty, buf_neg[0]
        );
    }

    #[test]
    fn same_seed_and_logits_give_same_token(
        logits in prop::collection::vec(-5.0f32..5.0f32, 2..=64),
        seed in any::<u64>(),
        temperature in 0.1f32..3.0f32,
    ) {
        let make = || {
            SamplingStrategy::new(SamplingConfig {
                temperature,
                top_k: 0,
                top_p: 1.0,
                repetition_penalty: 1.0,
                seed: Some(seed),
            })
        };
        let t1 = make().sample(&logits, &[]).unwrap();
        let t2 = make().sample(&logits, &[]).unwrap();
        prop_assert_eq!(
            t1, t2,
            "seed={} must yield identical tokens; got {} vs {}",
            seed, t1, t2
        );
    }

    #[test]
    fn single_finite_logit_always_selected(
        n_tokens in 2usize..=64,
        hot_idx in 0usize..64,
        hot_logit in -10.0f32..10.0f32,
        temperature in 0.1f32..3.0f32,
        seed in any::<u64>(),
    ) {
        let hot = hot_idx % n_tokens;
        let mut logits = vec![f32::NEG_INFINITY; n_tokens];
        logits[hot] = hot_logit;

        let config = SamplingConfig {
            temperature,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(seed),
        };
        let mut strategy = SamplingStrategy::new(config);
        let token = strategy.sample(&logits, &[]).unwrap();
        prop_assert_eq!(
            token, hot as u32,
            "with only token {} finite (logit={}), sampler must select it; got {}",
            hot, hot_logit, token
        );
    }
}
