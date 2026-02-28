//! Comprehensive property-based tests for `bitnet-sampling`.
//!
//! Covers the 12 required invariants:
//!   1.  `temperature = 0.0` → greedy argmax (short-circuits before softmax)
//!   2.  Lower temperature → more-peaked distribution (higher max probability)
//!   3.  Top-k: sampled token is always one of the top-k tokens by logit
//!   4.  Top-p: retained nucleus cumulative mass ≥ p
//!   5.  Repetition penalty: penalised token logits are strictly reduced
//!   6.  Output token index is always in `[0, vocab_size)`
//!   7.  Empty logits → `Err` (no panic) for both `greedy_sample` and `SamplingStrategy`
//!   8.  Single-token vocabulary always returns token 0
//!   9.  Determinism: same seed + same logits → same output token
//!  10.  All-equal logits + any temperature → exactly uniform softmax (each prob = 1/n)
//!  11.  Greedy tie-breaking: equal-logit tokens → lowest index wins (llama.cpp compat)
//!  12.  `apply_repetition_penalty` with penalty = 1.0 is a no-op

use bitnet_sampling::{
    SamplingConfig, SamplingStrategy, apply_repetition_penalty, apply_temperature, apply_top_p,
    greedy_sample, softmax_in_place,
};
use proptest::prelude::*;

// ── Test 7: empty logits ─────────────────────────────────────────────────────
// These are unit tests (no parameters to vary), but they exercise a required
// error path that property tests cannot cover meaningfully.

/// `greedy_sample(&[])` must return `Err`, not panic.
#[test]
fn empty_logits_greedy_returns_error() {
    assert!(greedy_sample(&[]).is_err(), "greedy_sample on empty slice must return Err");
}

/// `SamplingStrategy::sample(&[], &[])` must return `Err`, not panic.
#[test]
fn empty_logits_strategy_returns_error() {
    let config = SamplingConfig { temperature: 0.7, seed: Some(0), ..Default::default() };
    let mut strategy = SamplingStrategy::new(config);
    assert!(
        strategy.sample(&[], &[]).is_err(),
        "SamplingStrategy::sample on empty logits must return Err"
    );
}

// ── Property tests ───────────────────────────────────────────────────────────

proptest! {
    // ── Test 1: temperature = 0.0 always produces the greedy argmax ──────────
    //
    // The implementation must short-circuit to `greedy_sample` before any
    // softmax when `temperature == 0.0`, making the result deterministic and
    // independent of the RNG seed.
    #[test]
    fn prop_temp_zero_is_greedy(
        logits in prop::collection::vec(-50.0f32..50.0f32, 1..=128),
        seed in any::<u64>(),
    ) {
        let config = SamplingConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(seed),
        };
        let mut strategy = SamplingStrategy::new(config);
        let sampled = strategy.sample(&logits, &[]).unwrap();
        let greedy = greedy_sample(&logits).unwrap();
        prop_assert_eq!(
            sampled, greedy,
            "temperature=0 must equal greedy argmax; got {} vs {}",
            sampled, greedy
        );
    }

    // ── Test 2: lower temperature → more peaked distribution ─────────────────
    //
    // `apply_temperature(T)` scales logits by `1/T`.  A larger T produces a
    // smaller scale factor, which compresses the spread of logits and leads to
    // a softer (less peaked) softmax distribution.  Therefore:
    //   max_prob(high_T) ≤ max_prob(low_T)
    #[test]
    fn prop_lower_temp_more_peaked(
        logits in prop::collection::vec(0.01f32..10.0f32, 2..=32),
        low_temp in 0.1f32..0.5f32,
        delta in 0.5f32..4.0f32,
    ) {
        let high_temp = low_temp + delta;

        let mut low_probs = logits.clone();
        apply_temperature(&mut low_probs, low_temp);
        softmax_in_place(&mut low_probs);
        let low_max = low_probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let mut high_probs = logits.clone();
        apply_temperature(&mut high_probs, high_temp);
        softmax_in_place(&mut high_probs);
        let high_max = high_probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        prop_assert!(
            high_max <= low_max + 1e-5,
            "higher temp={} max_prob={} must be ≤ lower temp={} max_prob={}",
            high_temp, high_max, low_temp, low_max
        );
    }

    // ── Test 3: top-k: sampled token is always in the top-k set ──────────────
    //
    // After `apply_top_k(k)` only the k tokens with the highest logits
    // survive (the rest become NEG_INFINITY and thus probability 0 after
    // softmax).  Any token sampled from the resulting distribution must
    // therefore come from those k positions.
    //
    // Ties at the boundary are handled by including every token whose logit
    // is ≥ the k-th-largest value, so the set may be slightly larger than k
    // when values are equal.
    #[test]
    fn prop_top_k_sampled_in_top_k_set(
        logits in prop::collection::vec(-10.0f32..10.0f32, 2..=64),
        k in 1usize..=16,
        seed in any::<u64>(),
    ) {
        let effective_k = k.min(logits.len());

        // Determine the minimum logit value that survives top-k filtering.
        // We sort descending and take the value at position effective_k - 1.
        let mut sorted = logits.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted[effective_k - 1];

        // Collect every index whose logit is ≥ threshold (handles ties).
        let top_k_indices: std::collections::HashSet<usize> = logits
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v >= threshold)
            .map(|(i, _)| i)
            .collect();

        let config = SamplingConfig {
            temperature: 0.7,
            top_k: effective_k as u32,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(seed),
        };
        let mut strategy = SamplingStrategy::new(config);
        let token = strategy.sample(&logits, &[]).unwrap();

        prop_assert!(
            top_k_indices.contains(&(token as usize)),
            "sampled token {} not in top-{} set (threshold={:.4})",
            token, effective_k, threshold
        );
    }

    // ── Test 4: top-p: nucleus cumulative mass ≥ p ───────────────────────────
    //
    // `apply_top_p(p)` zeroes out low-probability tokens until the remaining
    // tokens together cover at least `p` of the total probability mass.
    // Summing the surviving (non-zero) probabilities must yield ≥ p.
    #[test]
    fn prop_top_p_nucleus_mass_gte_p(
        logits in prop::collection::vec(-10.0f32..10.0f32, 2..=64),
        p in 0.1f32..0.99f32,
    ) {
        let mut probs = logits.clone();
        softmax_in_place(&mut probs);
        apply_top_p(&mut probs, p);
        let nucleus_mass: f32 = probs.iter().filter(|&&x| x > 0.0).sum();
        prop_assert!(
            nucleus_mass >= p - 1e-4,
            "nucleus mass {:.6} < top_p={:.4}", nucleus_mass, p
        );
    }

    // ── Test 5: repetition penalty strictly reduces penalised token logits ────
    //
    // For a token in the context:
    //   positive logit → divided by penalty  → smaller positive  (< original)
    //   negative logit → multiplied by penalty → more negative   (< original)
    // The unpenalised token (index 1) must remain unchanged.
    #[test]
    fn prop_rep_penalty_reduces_penalized_logits(
        pos_logit in 0.01f32..20.0f32,
        neg_logit in -20.0f32..-0.01f32,
        penalty in 1.01f32..5.0f32,
    ) {
        // Positive logit: penalty divides → value decreases.
        let mut buf_pos = vec![pos_logit, 0.5f32];
        apply_repetition_penalty(&mut buf_pos, &[0u32], penalty);
        prop_assert!(
            buf_pos[0] < pos_logit,
            "positive logit {:.4} not reduced by penalty {:.4}; got {:.4}",
            pos_logit, penalty, buf_pos[0]
        );

        // Negative logit: penalty multiplies → value becomes more negative.
        let mut buf_neg = vec![neg_logit, 0.5f32];
        apply_repetition_penalty(&mut buf_neg, &[0u32], penalty);
        prop_assert!(
            buf_neg[0] < neg_logit,
            "negative logit {:.4} not pushed more negative by penalty {:.4}; got {:.4}",
            neg_logit, penalty, buf_neg[0]
        );
    }

    // ── Test 6: output token always in [0, vocab_size) ───────────────────────
    //
    // No matter what combination of temperature, top-k, top-p, and seed is
    // used, the returned token index must be a valid index into the logit
    // slice — never equal to or greater than vocab_size.
    #[test]
    fn prop_output_always_in_vocab_range(
        logits in prop::collection::vec(-10.0f32..10.0f32, 1..=128),
        temperature in 0.0f32..3.0f32,
        top_k in 0u32..=32,
        top_p in 0.5f32..1.0f32,
        seed in any::<u64>(),
    ) {
        let config = SamplingConfig {
            temperature,
            top_k,
            top_p,
            repetition_penalty: 1.0,
            seed: Some(seed),
        };
        let mut strategy = SamplingStrategy::new(config);
        let token = strategy.sample(&logits, &[]).unwrap();
        prop_assert!(
            (token as usize) < logits.len(),
            "token {} out of range [0, {})",
            token, logits.len()
        );
    }

    // ── Test 7b: softmax of a single-element vector equals [1.0] ─────────────
    //
    // A vocabulary of size 1 has exactly one token with probability 1.0
    // regardless of its logit value.  This is a cornerstone property of
    // softmax and validates the degenerate-vocab code path.
    #[test]
    fn prop_single_element_softmax_is_one(
        v in -1000.0f32..1000.0f32,
    ) {
        let mut buf = vec![v];
        softmax_in_place(&mut buf);
        prop_assert!(
            (buf[0] - 1.0).abs() < 1e-6,
            "softmax([{}]) = {} != 1.0",
            v, buf[0]
        );
    }

    // ── Test 8: single-token vocabulary always returns token 0 ───────────────
    //
    // When there is only one possible token, every sampler — greedy or
    // stochastic — must return 0 regardless of the logit value, temperature,
    // or seed.
    #[test]
    fn prop_single_token_vocab_returns_zero(
        logit in -100.0f32..100.0f32,
        temperature in 0.0f32..3.0f32,
        seed in any::<u64>(),
    ) {
        // Greedy path
        prop_assert_eq!(
            greedy_sample(&[logit]).unwrap(),
            0u32,
            "greedy_sample on single-token vocab must return 0"
        );

        // Stochastic path via SamplingStrategy
        let config = SamplingConfig {
            temperature,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(seed),
        };
        let mut strategy = SamplingStrategy::new(config);
        let token = strategy.sample(&[logit], &[]).unwrap();
        prop_assert_eq!(token, 0u32, "single-token vocab must return 0, got {}", token);
    }

    // ── Test 9: determinism — same seed + same logits → same token ───────────
    //
    // Two `SamplingStrategy` instances created with the same configuration
    // (including the same seed) must produce the same token when presented
    // with the same logits.  This must hold for any positive temperature
    // (i.e., the stochastic code path).
    #[test]
    fn prop_determinism_same_seed_same_token(
        logits in prop::collection::vec(-10.0f32..10.0f32, 1..=64),
        temperature in 0.1f32..2.0f32,
        seed in any::<u64>(),
    ) {
        let make = || SamplingStrategy::new(SamplingConfig {
            temperature,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(seed),
        });
        let t1 = make().sample(&logits, &[]).unwrap();
        let t2 = make().sample(&logits, &[]).unwrap();
        prop_assert_eq!(
            t1, t2,
            "seed={} must produce identical tokens; got {} vs {}",
            seed, t1, t2
        );
    }

    // ── Test 10: all-equal logits → uniform softmax ───────────────────────────
    //
    // When all logits are identical, temperature scaling multiplies every
    // entry by the same constant.  `softmax` of a constant vector is the
    // uniform distribution `[1/n, 1/n, ..., 1/n]` exactly.
    #[test]
    fn prop_all_equal_logits_uniform_softmax(
        n in 1usize..=64,
        v in -50.0f32..50.0f32,
        temperature in 0.1f32..5.0f32,
    ) {
        let logits = vec![v; n];
        let mut probs = logits.clone();
        apply_temperature(&mut probs, temperature);
        softmax_in_place(&mut probs);
        let expected = 1.0f32 / n as f32;
        for (i, &p) in probs.iter().enumerate() {
            prop_assert!(
                (p - expected).abs() < 1e-5,
                "all-equal logits[{}]: prob {:.8} != 1/{} = {:.8}",
                i, p, n, expected
            );
        }
    }

    // ── Test 11: greedy tie-breaking — equal logits → lowest index wins ───────
    //
    // The llama.cpp-compatible greedy decoder must break logit ties by
    // returning the **lowest** token index.  We construct a vector where the
    // first `n_ties` entries share the maximum value and the remaining entries
    // are strictly lower, then assert the result is always index 0.
    #[test]
    fn prop_greedy_tie_breaking_lowest_index(
        max_val in 0.0f32..10.0f32,
        n_ties in 2usize..=5,
        extra in 0usize..=4,
    ) {
        // n_ties tokens at max_val, `extra` tokens at max_val - 1 (strictly lower).
        let mut logits = vec![max_val; n_ties];
        logits.extend(std::iter::repeat_n(max_val - 1.0, extra));

        let result = greedy_sample(&logits).unwrap();
        prop_assert_eq!(
            result, 0u32,
            "tie-break: first of {} equal-max tokens should be index 0, got {}",
            n_ties, result
        );
    }

    // ── Test 12: apply_repetition_penalty with penalty = 1.0 is a no-op ──────
    //
    // A penalty of exactly 1.0 means "no penalty": dividing or multiplying by 1
    // leaves every logit unchanged regardless of the context tokens present.
    #[test]
    fn prop_rep_penalty_one_is_noop(
        logits in prop::collection::vec(-20.0f32..20.0f32, 1..=64),
        context in prop::collection::vec(0u32..64u32, 0..=8),
    ) {
        let original = logits.clone();
        let mut penalized = logits.clone();
        apply_repetition_penalty(&mut penalized, &context, 1.0);
        prop_assert_eq!(
            &penalized, &original,
            "apply_repetition_penalty with penalty=1.0 must be a no-op"
        );
    }
}
