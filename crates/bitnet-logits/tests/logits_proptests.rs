//! Comprehensive property-based tests for `bitnet-logits`.
//!
//! Covers the invariants specified in the task:
//! - Softmax: sum ≈ 1.0, all outputs in [0, 1]
//! - Softmax numerical stability: very large inputs produce no NaN/inf
//! - Log-softmax monotonicity: log(p) preserves the same order as p
//! - Repetition penalty: penalized tokens have lower logits
//! - Temperature scaling: >1 → more uniform, <1 → more peaked
//! - Top-k masking: at most k elements remain non-NEG_INFINITY
//! - Top-p masking: surviving nucleus has cumulative probability ≥ p
//! - -inf inputs handled gracefully (no NaN, valid output)
//! - Empty input: no panic, returns empty
//! - Single element: probability = 1.0

use bitnet_logits::{
    apply_repetition_penalty, apply_temperature, apply_top_k, apply_top_p, argmax, softmax_in_place,
};
use proptest::prelude::*;

// ── helpers ───────────────────────────────────────────────────────────────────

/// Generate a vec of finite f32 values in [min, max].
fn finite_vec(
    min: f32,
    max: f32,
    len: impl Into<prop::collection::SizeRange>,
) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(min..=max, len)
}

/// Generate a vec that may contain some `NEG_INFINITY` entries alongside finite values.
fn mixed_neginf_vec(
    len: impl Into<prop::collection::SizeRange>,
) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(
        prop_oneof![
            Just(f32::NEG_INFINITY),
            (-20.0f32..=20.0f32),
        ],
        len,
    )
    // Ensure at least one finite entry so softmax has something to work with.
    .prop_filter("at least one finite", |v| v.iter().any(|x| x.is_finite()))
}

// ── 1. Softmax: all outputs in [0, 1] ────────────────────────────────────────

proptest! {
    /// Every element produced by softmax must lie in the closed interval [0, 1].
    #[test]
    fn softmax_outputs_in_zero_one(logits in finite_vec(-50.0, 50.0, 1..200)) {
        let mut probs = logits;
        softmax_in_place(&mut probs);
        for &p in &probs {
            prop_assert!(
                (0.0..=1.0).contains(&p),
                "softmax output {p} not in [0, 1]"
            );
        }
    }
}

// ── 2. Softmax numerical stability: very large inputs ────────────────────────

proptest! {
    /// Softmax with very large (±1e30) inputs must not produce NaN or +inf.
    #[test]
    fn softmax_no_nan_inf_on_huge_inputs(
        logits in finite_vec(-1e30, 1e30, 1..50)
    ) {
        let mut probs = logits;
        softmax_in_place(&mut probs);
        for &p in &probs {
            prop_assert!(!p.is_nan(), "softmax produced NaN for huge input");
            prop_assert!(!p.is_infinite(), "softmax produced Inf for huge input");
        }
    }

    /// Softmax applied to all-equal huge values yields a uniform distribution.
    #[test]
    fn softmax_uniform_on_constant_huge_input(
        val in prop_oneof![Just(1e30f32), Just(-1e30f32), Just(1e20f32)],
        n in 2usize..=32usize,
    ) {
        let mut logits = vec![val; n];
        softmax_in_place(&mut logits);
        let expected = 1.0 / n as f32;
        for &p in &logits {
            prop_assert!(
                (p - expected).abs() < 1e-4,
                "expected uniform {expected}, got {p}"
            );
        }
    }
}

// ── 3. Log-softmax monotonicity ───────────────────────────────────────────────

proptest! {
    /// log(p_i) and p_i induce the same total order (log is strictly monotone increasing).
    /// Consequently, the argmax of log-probabilities equals the argmax of probabilities.
    #[test]
    fn log_softmax_argmax_equals_softmax_argmax(
        logits in finite_vec(-10.0, 10.0, 2..100)
    ) {
        // Require a unique maximum so argmax is unambiguous.
        let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let max_count = logits.iter().filter(|&&x| (x - max_val).abs() < f32::EPSILON).count();
        prop_assume!(max_count == 1);

        let mut probs = logits;
        softmax_in_place(&mut probs);

        // log-probs: apply ln to every non-zero probability.
        let log_probs: Vec<f32> = probs.iter().map(|&p| if p > 0.0 { p.ln() } else { f32::NEG_INFINITY }).collect();

        prop_assert_eq!(
            argmax(&probs),
            argmax(&log_probs),
            "argmax of probs and log-probs must agree"
        );
    }

    /// For any two indices i, j: if p[i] > p[j] > 0 then log(p[i]) > log(p[j]).
    #[test]
    fn log_softmax_order_matches_softmax_order(
        logits in finite_vec(-10.0, 10.0, 2..50)
    ) {
        let mut probs = logits;
        softmax_in_place(&mut probs);

        // Pick the two highest-probability entries; they must be ordered the same in log-space.
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if indexed.len() >= 2 && indexed[0].1 > 0.0 && indexed[1].1 > 0.0 {
            let log_top = indexed[0].1.ln();
            let log_second = indexed[1].1.ln();
            prop_assert!(
                log_top >= log_second,
                "log-prob order violated: ln({}) < ln({})",
                indexed[0].1,
                indexed[1].1
            );
        }
    }
}

// ── 4. Repetition penalty: penalized tokens have lower logits ─────────────────

proptest! {
    /// After applying a penalty > 1, the penalized logit is always ≤ the original
    /// (regardless of sign): positive logits shrink toward 0, negative logits grow
    /// more negative — both reduce the token's relative likelihood.
    #[test]
    fn repetition_penalty_moves_logit_toward_or_below_zero(
        base_logit in prop_oneof![0.01f32..10.0f32, -10.0f32..-0.01f32],
        penalty in 1.01f32..5.0f32,
        vocab_size in 2usize..=64usize,
    ) {
        let mut logits = vec![0.0f32; vocab_size];
        logits[0] = base_logit;
        let original = logits[0];
        apply_repetition_penalty(&mut logits, &[0u32], penalty);
        // For positive logit: logit/penalty < logit  → new ≤ original
        // For negative logit: logit*penalty < logit  → new ≤ original
        prop_assert!(
            logits[0] <= original + 1e-6,
            "penalty={penalty}: {} → {} should be ≤ original",
            original,
            logits[0]
        );
    }

    /// After penalty, the penalized token is less likely than a token that
    /// started with the same logit but was not penalized.
    #[test]
    fn repetition_penalty_makes_token_less_likely(
        base_positive in 0.1f32..5.0f32,
        penalty in 1.01f32..4.0f32,
    ) {
        // Two tokens start at the same positive logit; token 0 is penalized.
        let mut logits = vec![base_positive, base_positive];
        apply_repetition_penalty(&mut logits, &[0u32], penalty);
        prop_assert!(
            logits[0] < logits[1],
            "penalized logit {} must be < unpenalized {}",
            logits[0],
            logits[1]
        );
    }
}

// ── 5. Temperature scaling: >1 → more uniform, <1 → more peaked ──────────────

proptest! {
    /// Temperature > 1 must produce a distribution with a lower maximum probability
    /// than the original (more uniform).
    #[test]
    fn temperature_gt1_lowers_max_prob(
        logits in finite_vec(-5.0, 5.0, 3..30),
        temp in 1.01f32..5.0f32,
    ) {
        // Require at least two distinct logit values so the distribution is non-trivial.
        let unique: std::collections::HashSet<u32> = logits.iter().map(|&x| x.to_bits()).collect();
        prop_assume!(unique.len() >= 2);

        let mut base = logits.clone();
        softmax_in_place(&mut base);
        let max_base = base.iter().copied().fold(0.0f32, f32::max);

        let mut scaled = logits;
        apply_temperature(&mut scaled, temp);
        softmax_in_place(&mut scaled);
        let max_scaled = scaled.iter().copied().fold(0.0f32, f32::max);

        prop_assert!(
            max_scaled <= max_base + 1e-5,
            "temp={temp}: max_prob went from {max_base} to {max_scaled} (should decrease or stay)"
        );
    }

    /// Temperature < 1 must produce a distribution with a higher maximum probability
    /// than the original (more peaked).
    #[test]
    fn temperature_lt1_raises_max_prob(
        logits in finite_vec(-5.0, 5.0, 3..30),
        temp in 0.01f32..0.99f32,
    ) {
        let unique: std::collections::HashSet<u32> = logits.iter().map(|&x| x.to_bits()).collect();
        prop_assume!(unique.len() >= 2);

        let mut base = logits.clone();
        softmax_in_place(&mut base);
        let max_base = base.iter().copied().fold(0.0f32, f32::max);

        let mut scaled = logits;
        apply_temperature(&mut scaled, temp);
        softmax_in_place(&mut scaled);
        let max_scaled = scaled.iter().copied().fold(0.0f32, f32::max);

        prop_assert!(
            max_scaled >= max_base - 1e-5,
            "temp={temp}: max_prob went from {max_base} to {max_scaled} (should increase or stay)"
        );
    }
}

// ── 6. Top-k masking: at most k elements non-NEG_INFINITY ────────────────────

proptest! {
    /// After `apply_top_k(k)`, the number of finite elements is exactly
    /// min(k, len) when all logits are distinct.
    #[test]
    fn top_k_finite_count_exactly_k(
        logits in finite_vec(-10.0, 10.0, 2..80),
        k in 1usize..30,
    ) {
        // All distinct → no tie-breaking ambiguity.
        let unique: std::collections::HashSet<u32> = logits.iter().map(|&x| x.to_bits()).collect();
        prop_assume!(unique.len() == logits.len());

        let k_eff = k.min(logits.len());
        let mut l = logits;
        apply_top_k(&mut l, k_eff);
        let kept = l.iter().filter(|&&x| x != f32::NEG_INFINITY).count();
        prop_assert_eq!(kept, k_eff, "expected exactly {} finite elements, got {}", k_eff, kept);
    }

    /// The highest logit in the input is always among the surviving top-k elements.
    #[test]
    fn top_k_always_keeps_global_max(
        logits in finite_vec(-10.0, 10.0, 2..80),
        k in 1usize..30,
    ) {
        let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let max_count = logits.iter().filter(|&&x| (x - max_val).abs() < f32::EPSILON).count();
        prop_assume!(max_count == 1);

        let k_eff = k.min(logits.len());
        let max_idx = argmax(&logits);
        let mut l = logits;
        apply_top_k(&mut l, k_eff);
        prop_assert!(
            l[max_idx] != f32::NEG_INFINITY,
            "top_k({k_eff}) removed the global maximum at index {max_idx}"
        );
    }
}

// ── 7. Top-p masking: nucleus has cumulative probability ≥ p ──────────────────

proptest! {
    /// The tokens that survive `apply_top_p(p)` must sum to at least `p`.
    #[test]
    fn top_p_nucleus_cum_prob_ge_p(
        logits in finite_vec(-10.0, 10.0, 2..100),
        top_p in 0.01f32..0.99f32,
    ) {
        let mut probs = logits;
        softmax_in_place(&mut probs);
        apply_top_p(&mut probs, top_p);
        let surviving: f32 = probs.iter().copied().sum();
        prop_assert!(
            surviving >= top_p - 1e-4,
            "nucleus sum {surviving} < top_p {top_p}"
        );
    }

    /// `apply_top_p(1.0)` is always a no-op.
    #[test]
    fn top_p_one_is_noop(logits in finite_vec(-10.0, 10.0, 1..50)) {
        let mut probs = logits;
        softmax_in_place(&mut probs);
        let before = probs.clone();
        apply_top_p(&mut probs, 1.0);
        prop_assert_eq!(probs, before, "top_p=1.0 must not change probabilities");
    }

    /// After `apply_top_p`, no element may be negative.
    #[test]
    fn top_p_output_non_negative(
        logits in finite_vec(-10.0, 10.0, 2..100),
        top_p in 0.01f32..0.99f32,
    ) {
        let mut probs = logits;
        softmax_in_place(&mut probs);
        apply_top_p(&mut probs, top_p);
        for &p in &probs {
            prop_assert!(p >= 0.0, "top_p produced negative probability {p}");
        }
    }
}

// ── 8. Inputs with -inf values: handled gracefully ────────────────────────────

proptest! {
    /// When some logits are NEG_INFINITY (e.g., from top-k), softmax must
    /// produce no NaN values and must still sum to ≈ 1.0.
    #[test]
    fn softmax_handles_neginf_inputs(logits in mixed_neginf_vec(1..100)) {
        let mut probs = logits;
        softmax_in_place(&mut probs);
        for &p in &probs {
            prop_assert!(!p.is_nan(), "softmax produced NaN with NEG_INFINITY input");
        }
        let sum: f32 = probs.iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 1e-4,
            "softmax with NEG_INFINITY inputs: sum={sum} ≠ 1"
        );
    }
}

// ── 9. Empty input: no panic ──────────────────────────────────────────────────

/// `softmax_in_place` on an empty slice must not panic.
#[test]
fn empty_softmax_no_panic() {
    let mut logits: Vec<f32> = vec![];
    softmax_in_place(&mut logits);
    assert!(logits.is_empty());
}

/// `apply_temperature` on an empty slice must not panic.
#[test]
fn empty_temperature_no_panic() {
    let mut logits: Vec<f32> = vec![];
    apply_temperature(&mut logits, 0.5);
    assert!(logits.is_empty());
}

/// `apply_top_k` on an empty slice must not panic and returns 0.
#[test]
fn empty_top_k_no_panic() {
    let mut logits: Vec<f32> = vec![];
    let kept = apply_top_k(&mut logits, 5);
    assert_eq!(kept, 0);
}

/// `apply_top_p` on an empty slice must not panic.
#[test]
fn empty_top_p_no_panic() {
    let mut probs: Vec<f32> = vec![];
    apply_top_p(&mut probs, 0.9);
    assert!(probs.is_empty());
}

/// `apply_repetition_penalty` on an empty slice must not panic.
#[test]
fn empty_repetition_penalty_no_panic() {
    let mut logits: Vec<f32> = vec![];
    apply_repetition_penalty(&mut logits, &[0u32, 1, 2], 2.0);
    assert!(logits.is_empty());
}

// ── 10. Single element: probability = 1.0 ────────────────────────────────────

proptest! {
    /// A single-element logit vector always softmaxes to [1.0].
    #[test]
    fn single_element_softmax_is_one(val in -1e6f32..=1e6f32) {
        prop_assume!(val.is_finite());
        let mut logits = vec![val];
        softmax_in_place(&mut logits);
        prop_assert!(
            (logits[0] - 1.0).abs() < 1e-6,
            "single-element softmax expected 1.0, got {}",
            logits[0]
        );
    }

    /// A single-element NEG_INFINITY input (degenerate): softmax falls back to uniform (= 1.0).
    #[test]
    fn single_element_neginf_softmax_is_one(_seed in 0u32..=100u32) {
        let mut logits = vec![f32::NEG_INFINITY];
        softmax_in_place(&mut logits);
        prop_assert!(
            (logits[0] - 1.0).abs() < 1e-6,
            "single NEG_INFINITY softmax expected 1.0 (uniform fallback), got {}",
            logits[0]
        );
    }
}

// ── 11. Top-k then softmax: still a valid distribution ───────────────────────

proptest! {
    /// After top-k masking, softmax must still produce a valid probability
    /// distribution (sum ≈ 1.0, all values ≥ 0).
    #[test]
    fn top_k_then_softmax_valid_distribution(
        logits in finite_vec(-10.0, 10.0, 2..80),
        k in 1usize..20,
    ) {
        let k_eff = k.min(logits.len());
        let mut l = logits;
        apply_top_k(&mut l, k_eff);
        softmax_in_place(&mut l);
        let sum: f32 = l.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-4, "sum after top_k+softmax = {sum}");
        for &p in &l {
            prop_assert!(p >= 0.0, "negative probability {p} after top_k+softmax");
            prop_assert!(!p.is_nan(), "NaN after top_k+softmax");
        }
    }
}

// ── 12. Repetition penalty does not affect unseen tokens ─────────────────────

proptest! {
    /// Tokens whose IDs are NOT in `token_ids` must have unchanged logits.
    #[test]
    fn repetition_penalty_leaves_unseen_tokens_unchanged(
        logits in finite_vec(-10.0, 10.0, 4..50),
        penalty in 1.01f32..4.0f32,
    ) {
        // Penalize only index 0; verify indices 1..n are unchanged.
        let original = logits.clone();
        let mut l = logits;
        apply_repetition_penalty(&mut l, &[0u32], penalty);
        for i in 1..l.len() {
            prop_assert!((l[i] - original[i]).abs() < f32::EPSILON,
                "token {} was not in penalty list but logit changed",
                i
            );
        }
    }
}
