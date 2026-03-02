//! Wave 33 property tests: inference sampling invariants.
//!
//! Properties tested (5):
//! 1. SamplingStrategy::sample always returns a valid token index (< vocab_size)
//! 2. Temperature=0 always selects argmax
//! 3. top_k filtering keeps at most k candidates
//! 4. Repetition penalty decreases probability of repeated tokens
//! 5. Greedy decode is deterministic with same seed

use bitnet_sampling::{
    SamplingConfig, SamplingStrategy, apply_repetition_penalty, apply_top_k, greedy_sample,
};
use proptest::prelude::*;

/// Non-empty f32 logits vector with finite values.
fn logits_vec(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-10.0f32..10.0f32, 2..=max_len)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    // ── 1. sample always returns a valid token index ────────────────────

    /// Stochastic sampling always returns a token ID strictly less than
    /// the vocabulary size (logits length).
    #[test]
    fn prop_sample_returns_valid_index(
        logits in logits_vec(64),
        seed in 0u64..10_000,
    ) {
        let vocab_size = logits.len();
        let config = SamplingConfig {
            temperature: 0.8,
            seed: Some(seed),
            ..Default::default()
        };
        let mut strategy = SamplingStrategy::new(config);
        let token = strategy.sample(&logits, &[]).unwrap();
        prop_assert!(
            (token as usize) < vocab_size,
            "token {} >= vocab_size {}",
            token,
            vocab_size,
        );
    }

    // ── 2. Temperature=0 always selects argmax ──────────────────────────

    /// Greedy decoding (temperature=0) must agree with `argmax`.
    #[test]
    fn prop_temp_zero_is_argmax(
        logits in logits_vec(32),
        seed in 0u64..10_000,
    ) {
        let config = SamplingConfig {
            temperature: 0.0,
            seed: Some(seed),
            ..Default::default()
        };
        let mut strategy = SamplingStrategy::new(config);
        let token = strategy.sample(&logits, &[]).unwrap();
        let expected = greedy_sample(&logits).unwrap();
        prop_assert_eq!(
            token, expected,
            "temperature=0 should always pick argmax",
        );
    }

    // ── 3. top_k filtering keeps at most k candidates ───────────────────

    /// After `apply_top_k`, at most `k` entries remain finite;
    /// the rest are `NEG_INFINITY`.
    #[test]
    fn prop_top_k_keeps_at_most_k(
        logits in logits_vec(64),
        k in 1usize..=32,
    ) {
        let mut filtered = logits.clone();
        let effective_k = k.min(filtered.len());
        apply_top_k(&mut filtered, effective_k);

        let finite_count = filtered.iter().filter(|v| v.is_finite()).count();
        prop_assert!(
            finite_count <= effective_k,
            "finite_count {} > k {}",
            finite_count,
            effective_k,
        );

        // Everything that survived must be one of the original top-k values.
        let mut sorted = logits.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let threshold = sorted[effective_k.min(sorted.len()) - 1];

        for (i, &v) in filtered.iter().enumerate() {
            if v.is_finite() {
                prop_assert!(
                    logits[i] >= threshold,
                    "surviving logit {} at idx {} < threshold {}",
                    logits[i],
                    i,
                    threshold,
                );
            }
        }
    }

    // ── 4. Repetition penalty decreases probability of repeated tokens ──

    /// Applying repetition penalty > 1.0 to positive logits makes them
    /// smaller (less likely).
    #[test]
    fn prop_repetition_penalty_decreases_positive_logits(
        base_logit in 0.01f32..10.0,
        penalty in 1.01f32..3.0,
        count in 1u32..5,
    ) {
        let vocab_size = 8usize;
        let mut logits = vec![base_logit; vocab_size];
        let repeated_token = 0u32;

        // Build context with `count` occurrences of token 0.
        let context: Vec<u32> = std::iter::repeat_n(repeated_token, count as usize).collect();

        let original = logits[repeated_token as usize];
        apply_repetition_penalty(&mut logits, &context, penalty);

        prop_assert!(
            logits[repeated_token as usize] < original,
            "penalty should decrease positive logit: {} -> {}",
            original,
            logits[repeated_token as usize],
        );

        // Unpenalized tokens should be unchanged.
        for &v in &logits[1..] {
            prop_assert_eq!(v, base_logit, "unpenalized token should stay the same");
        }
    }

    // ── 5. Greedy decode is deterministic with same seed ────────────────

    /// Two `SamplingStrategy` instances with the same seed and temperature=0
    /// must produce identical tokens for the same logits.
    #[test]
    fn prop_greedy_deterministic_same_seed(
        logits in logits_vec(32),
        seed in 0u64..100_000,
    ) {
        let make = || {
            SamplingStrategy::new(SamplingConfig {
                temperature: 0.0,
                seed: Some(seed),
                ..Default::default()
            })
        };

        let mut s1 = make();
        let mut s2 = make();

        let t1 = s1.sample(&logits, &[]).unwrap();
        let t2 = s2.sample(&logits, &[]).unwrap();
        prop_assert_eq!(t1, t2, "same seed + greedy must be deterministic");
    }
}
