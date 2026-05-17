//! Integration tests for `bitnet-scoring-core`.
//!
//! These tests exercise the public surface of the crate end-to-end to give us
//! code-coverage signal that is independent of the inline unit tests in
//! `src/lib.rs`. They intentionally encode the *actual* behavior of the crate
//! (verified against the source) rather than aspirational behavior.

#![allow(clippy::float_cmp)]

use bitnet_scoring_core::{NllStats, observe_target_nll, sanitize_logits_in_place};

/// Tolerance for `f64` comparisons in this file. Tight enough to catch real
/// regressions while still allowing `f32 -> f64` softmax round-tripping.
const TOL: f64 = 1e-9;

/// Looser tolerance for paths that go through `f32` log-softmax before being
/// promoted to `f64`. `log_softmax_stable` accumulates in `f32`, so we cannot
/// expect tighter than ~1e-5 agreement with a pure-`f64` oracle.
const SOFTMAX_TOL: f64 = 1e-5;

// ---------------------------------------------------------------------------
// NllStats: construction / empty state
// ---------------------------------------------------------------------------

#[test]
fn nll_stats_default_starts_at_zero() {
    let stats = NllStats::default();
    assert_eq!(stats.sum, 0.0);
    assert_eq!(stats.tokens, 0);
}

#[test]
fn nll_stats_empty_mean_is_zero_not_nan() {
    // Documented behavior: `mean()` returns 0.0 when tokens == 0 to avoid NaN.
    let stats = NllStats::default();
    let m = stats.mean();
    assert!(!m.is_nan(), "mean() must not be NaN on empty stats");
    assert_eq!(m, 0.0);
}

#[test]
fn nll_stats_empty_perplexity_is_one() {
    // perplexity = exp(mean) and mean is 0.0 on empty, so perplexity is exp(0)=1.
    let stats = NllStats::default();
    let p = stats.perplexity();
    assert!(!p.is_nan());
    assert_eq!(p, 1.0);
}

// ---------------------------------------------------------------------------
// NllStats: observe_logprob
// ---------------------------------------------------------------------------

#[test]
fn observe_logprob_single_observation_negates_log_prob() {
    // The struct accumulates *negative* log-likelihood, so observing a
    // log-prob of -0.75 should add +0.75 to `sum`.
    let mut stats = NllStats::default();
    stats.observe_logprob(-0.75);
    assert_eq!(stats.tokens, 1);
    assert!((stats.sum - 0.75).abs() < TOL);
}

#[test]
fn observe_logprob_multiple_observations_accumulate() {
    let mut stats = NllStats::default();
    stats.observe_logprob(-0.25);
    stats.observe_logprob(-0.5);
    stats.observe_logprob(-1.25);
    assert_eq!(stats.tokens, 3);
    assert!((stats.sum - 2.0).abs() < TOL);
}

#[test]
fn observe_logprob_with_zero_log_prob_only_increments_tokens() {
    let mut stats = NllStats::default();
    stats.observe_logprob(0.0);
    assert_eq!(stats.tokens, 1);
    assert_eq!(stats.sum, 0.0);
}

#[test]
fn observe_logprob_with_positive_log_prob_decreases_sum() {
    // While log-probs are normally <= 0, the function is purely arithmetic,
    // so a positive input should subtract from `sum`.
    let mut stats = NllStats::default();
    stats.observe_logprob(1.5);
    assert_eq!(stats.tokens, 1);
    assert!((stats.sum + 1.5).abs() < TOL);
}

// ---------------------------------------------------------------------------
// NllStats: mean / perplexity over real samples
// ---------------------------------------------------------------------------

#[test]
fn mean_after_multiple_observations() {
    let mut stats = NllStats::default();
    stats.observe_logprob(-2.0);
    stats.observe_logprob(-4.0);
    stats.observe_logprob(-6.0);
    // sum = 12.0, tokens = 3 -> mean = 4.0
    assert!((stats.mean() - 4.0).abs() < TOL);
}

#[test]
fn perplexity_matches_exp_of_mean() {
    let mut stats = NllStats::default();
    stats.observe_logprob(-0.5);
    stats.observe_logprob(-1.5);
    // mean = 1.0, perplexity = e^1
    let expected = stats.mean().exp();
    assert!((stats.perplexity() - expected).abs() < TOL);
    assert!((stats.perplexity() - std::f64::consts::E).abs() < TOL);
}

// ---------------------------------------------------------------------------
// NllStats: add
// ---------------------------------------------------------------------------

#[test]
fn add_combines_sum_and_tokens() {
    let mut a = NllStats { sum: 1.25, tokens: 4 };
    let b = NllStats { sum: 2.75, tokens: 6 };
    a.add(b);
    assert_eq!(a.tokens, 10);
    assert!((a.sum - 4.0).abs() < TOL);
}

#[test]
fn add_is_equivalent_to_sum_then_construct() {
    let mut a = NllStats::default();
    a.observe_logprob(-0.5);
    a.observe_logprob(-1.0);
    let mut b = NllStats::default();
    b.observe_logprob(-2.0);
    b.observe_logprob(-3.0);

    let mut combined = a;
    combined.add(b);

    let constructed = NllStats { sum: a.sum + b.sum, tokens: a.tokens + b.tokens };
    assert_eq!(combined, constructed);
}

#[test]
fn add_default_is_identity() {
    let mut a = NllStats { sum: 7.5, tokens: 11 };
    let before = a;
    a.add(NllStats::default());
    assert_eq!(a, before);
}

// ---------------------------------------------------------------------------
// sanitize_logits_in_place
// ---------------------------------------------------------------------------

#[test]
fn sanitize_replaces_nan_with_neg_infinity() {
    let mut logits = vec![f32::NAN];
    sanitize_logits_in_place(&mut logits);
    assert_eq!(logits[0], f32::NEG_INFINITY);
}

#[test]
fn sanitize_replaces_positive_infinity_with_neg_infinity() {
    let mut logits = vec![f32::INFINITY];
    sanitize_logits_in_place(&mut logits);
    assert_eq!(logits[0], f32::NEG_INFINITY);
}

#[test]
fn sanitize_replaces_negative_infinity_with_neg_infinity() {
    // -inf is non-finite, so the implementation still rewrites it (to itself).
    let mut logits = vec![f32::NEG_INFINITY];
    sanitize_logits_in_place(&mut logits);
    assert_eq!(logits[0], f32::NEG_INFINITY);
}

#[test]
fn sanitize_preserves_finite_values_bitwise() {
    // Finite values should be left exactly as-is, including signed zero.
    let inputs: [f32; 6] = [-3.5, -0.0, 0.0, 1.0, 2.5, f32::MAX];
    let mut logits = inputs.to_vec();
    sanitize_logits_in_place(&mut logits);
    for (i, (got, want)) in logits.iter().zip(inputs.iter()).enumerate() {
        assert_eq!(got.to_bits(), want.to_bits(), "element {i} mutated unexpectedly");
    }
}

#[test]
fn sanitize_on_empty_slice_is_noop() {
    let mut logits: Vec<f32> = Vec::new();
    sanitize_logits_in_place(&mut logits);
    assert!(logits.is_empty());
}

#[test]
fn sanitize_mixed_slice_replaces_only_non_finite() {
    let mut logits = vec![1.0_f32, f32::NAN, 2.0, f32::INFINITY, -3.0, f32::NEG_INFINITY, 0.0];
    sanitize_logits_in_place(&mut logits);
    assert_eq!(logits[0], 1.0);
    assert_eq!(logits[1], f32::NEG_INFINITY);
    assert_eq!(logits[2], 2.0);
    assert_eq!(logits[3], f32::NEG_INFINITY);
    assert_eq!(logits[4], -3.0);
    assert_eq!(logits[5], f32::NEG_INFINITY);
    assert_eq!(logits[6], 0.0);
}

// ---------------------------------------------------------------------------
// observe_target_nll
// ---------------------------------------------------------------------------

#[test]
fn observe_target_nll_uniform_logits_yields_ln_vocab_size() {
    // Equal logits -> uniform softmax -> NLL == ln(vocab_size).
    let logits = vec![0.0_f32; 8];
    let mut stats = NllStats::default();
    observe_target_nll(&mut stats, &logits, 3);
    assert_eq!(stats.tokens, 1);
    let expected = (logits.len() as f64).ln();
    assert!((stats.sum - expected).abs() < SOFTMAX_TOL);
}

#[test]
fn observe_target_nll_accepts_target_at_first_middle_and_last_indices() {
    // Uniform distribution: the NLL contribution is the same regardless of
    // which index we pick, but we want to make sure all three indices work
    // (i.e. no off-by-one panic).
    for target in [0usize, 2, 4] {
        let logits = vec![0.0_f32; 5];
        let mut stats = NllStats::default();
        observe_target_nll(&mut stats, &logits, target);
        let expected = 5_f64.ln();
        assert!(
            (stats.sum - expected).abs() < SOFTMAX_TOL,
            "target={target} sum={} expected={expected}",
            stats.sum
        );
        assert_eq!(stats.tokens, 1);
    }
}

#[test]
fn observe_target_nll_matches_hand_computed_softmax() {
    // logits = [1.0, 2.0, 3.0], target = 2 (the largest).
    // Numerically stable softmax: shift by max (=3.0) -> [-2, -1, 0]
    // exps: [e^-2, e^-1, 1], sum = e^-2 + e^-1 + 1
    // log p_2 = 0 - ln(sum). NLL = -log p_2 = ln(sum).
    let logits = vec![1.0_f32, 2.0, 3.0];
    let mut stats = NllStats::default();
    observe_target_nll(&mut stats, &logits, 2);
    let sum = (-2.0_f64).exp() + (-1.0_f64).exp() + 1.0;
    let expected = sum.ln();
    assert!((stats.sum - expected).abs() < SOFTMAX_TOL, "got {} expected {}", stats.sum, expected);
    assert_eq!(stats.tokens, 1);
}

#[test]
fn observe_target_nll_is_numerically_stable_with_huge_logit() {
    // A very large logit should not overflow thanks to the max-shift inside
    // `log_softmax_stable`. Target = the huge logit -> NLL ~ 0.
    let logits = vec![0.0_f32, 1e30, 0.0, 0.0];
    let mut stats = NllStats::default();
    observe_target_nll(&mut stats, &logits, 1);
    assert_eq!(stats.tokens, 1);
    assert!(stats.sum.is_finite(), "sum should be finite, got {}", stats.sum);
    // log p ~ -log(1 + 3 * exp(-1e30)) ~ 0
    assert!(stats.sum.abs() < 1e-4, "expected ~0 NLL for dominant logit, got {}", stats.sum);
}

#[test]
fn observe_target_nll_accumulates_across_calls() {
    // Two observations on a 2-class uniform distribution should give 2 * ln(2).
    let logits = vec![0.0_f32, 0.0];
    let mut stats = NllStats::default();
    observe_target_nll(&mut stats, &logits, 0);
    observe_target_nll(&mut stats, &logits, 1);
    assert_eq!(stats.tokens, 2);
    let expected = 2.0 * 2_f64.ln();
    assert!((stats.sum - expected).abs() < SOFTMAX_TOL);
}

#[test]
fn observe_target_nll_does_not_sanitize_nan_input() {
    // The function does NOT call `sanitize_logits_in_place`; a NaN logit
    // therefore propagates through the softmax and yields a NaN sum.
    let logits = vec![1.0_f32, f32::NAN, 0.0];
    let mut stats = NllStats::default();
    observe_target_nll(&mut stats, &logits, 0);
    assert_eq!(stats.tokens, 1);
    assert!(stats.sum.is_nan(), "expected NaN sum from un-sanitized NaN logit, got {}", stats.sum);
}

#[test]
fn observe_target_nll_panics_on_out_of_range_target() {
    // `logp[target]` is a direct index, so out-of-range targets panic.
    let logits = vec![0.0_f32; 3];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut stats = NllStats::default();
        observe_target_nll(&mut stats, &logits, 99);
    }));
    assert!(result.is_err(), "expected panic on out-of-range target");
}

#[test]
fn observe_target_nll_panics_on_empty_logits() {
    // An empty logits slice gives an empty softmax vector, so any target
    // index (including 0) will panic on indexing.
    let logits: Vec<f32> = Vec::new();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut stats = NllStats::default();
        observe_target_nll(&mut stats, &logits, 0);
    }));
    assert!(result.is_err(), "expected panic on empty logits");
}

#[test]
fn observe_target_nll_after_sanitize_handles_nan() {
    // The recommended usage pattern: sanitize first, then observe.
    // After sanitization, the NaN becomes -inf, so it contributes ~0 to
    // softmax denominator and the remaining mass is split between indices 0
    // and 2 (both equal to 1.0) -> p_0 = 0.5 -> NLL = ln(2).
    let mut logits = vec![1.0_f32, f32::NAN, 1.0];
    sanitize_logits_in_place(&mut logits);
    let mut stats = NllStats::default();
    observe_target_nll(&mut stats, &logits, 0);
    assert_eq!(stats.tokens, 1);
    let expected = 2_f64.ln();
    assert!((stats.sum - expected).abs() < SOFTMAX_TOL, "got {} expected {}", stats.sum, expected);
}
