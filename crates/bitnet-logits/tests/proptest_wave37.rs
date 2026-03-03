//! Property-based tests — wave 37: logit filtering preserves relative ordering,
//! temperature scaling invariants, top-k/top-p selection properties, and
//! numerical edge cases.

use bitnet_logits::*;
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn finite_logits(
    lo: f32,
    hi: f32,
    len: impl Into<prop::collection::SizeRange>,
) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(lo..hi, len)
}

fn normalized_probs(
    len: impl Into<prop::collection::SizeRange>,
) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(0.01f32..1.0, len).prop_map(|v| {
        let sum: f32 = v.iter().sum();
        v.iter().map(|&x| x / sum).collect()
    })
}

// ---------------------------------------------------------------------------
// Temperature scaling invariants
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Temperature == 0 is a no-op.
    #[test]
    fn temperature_zero_is_noop(logits in finite_logits(-10.0, 10.0, 1..50)) {
        let original = logits.clone();
        let mut copy = logits;
        apply_temperature(&mut copy, 0.0);
        prop_assert_eq!(copy, original);
    }

    /// Temperature == 1.0 is a no-op.
    #[test]
    fn temperature_one_is_noop(logits in finite_logits(-10.0, 10.0, 1..50)) {
        let original = logits.clone();
        let mut copy = logits;
        apply_temperature(&mut copy, 1.0);
        prop_assert_eq!(copy, original);
    }

    /// Temperature > 0 preserves the relative ordering of logits.
    #[test]
    fn temperature_preserves_ordering(
        logits in finite_logits(-10.0, 10.0, 2..30),
        temp in 0.01f32..5.0,
    ) {
        let mut scaled = logits.clone();
        apply_temperature(&mut scaled, temp);
        for i in 0..logits.len() {
            for j in (i + 1)..logits.len() {
                if logits[i] > logits[j] {
                    prop_assert!(scaled[i] > scaled[j],
                        "ordering violated at ({}, {})", i, j);
                } else if logits[i] < logits[j] {
                    prop_assert!(scaled[i] < scaled[j],
                        "ordering violated at ({}, {})", i, j);
                }
            }
        }
    }

    /// After temperature scaling, logit[i] == original[i] / temperature.
    #[test]
    fn temperature_divides_by_temp(
        logits in finite_logits(-10.0, 10.0, 1..20),
        temp in 0.01f32..5.0,
    ) {
        let mut scaled = logits.clone();
        apply_temperature(&mut scaled, temp);
        for (i, &orig) in logits.iter().enumerate() {
            let expected = orig / temp;
            prop_assert!((scaled[i] - expected).abs() < 1e-4,
                "scaled[{}] = {}, expected {}", i, scaled[i], expected);
        }
    }

    /// Temperature < 1 increases the spread (makes max prob higher after softmax).
    #[test]
    fn low_temperature_sharpens(
        logits in finite_logits(0.1, 5.0, 3..20),
        temp in 0.01f32..0.99,
    ) {
        let mut base = logits.clone();
        softmax_in_place(&mut base);
        let max_base = base.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let mut sharp = logits;
        apply_temperature(&mut sharp, temp);
        softmax_in_place(&mut sharp);
        let max_sharp = sharp.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        prop_assert!(max_sharp >= max_base - 1e-6,
            "low temp should sharpen: max_sharp={}, max_base={}", max_sharp, max_base);
    }

    /// Temperature > 1 flattens the distribution (max prob decreases or equal).
    #[test]
    fn high_temperature_flattens(
        logits in finite_logits(0.1, 5.0, 3..20),
        temp in 1.01f32..5.0,
    ) {
        let mut base = logits.clone();
        softmax_in_place(&mut base);
        let max_base = base.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let mut flat = logits;
        apply_temperature(&mut flat, temp);
        softmax_in_place(&mut flat);
        let max_flat = flat.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        prop_assert!(max_flat <= max_base + 1e-6,
            "high temp should flatten: max_flat={}, max_base={}", max_flat, max_base);
    }
}

// ---------------------------------------------------------------------------
// Softmax properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Softmax output sums to 1.0 (within tolerance).
    #[test]
    fn softmax_sums_to_one(logits in finite_logits(-50.0, 50.0, 1..100)) {
        let mut probs = logits;
        softmax_in_place(&mut probs);
        let sum: f32 = probs.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-4, "sum = {}", sum);
    }

    /// All softmax outputs are non-negative.
    #[test]
    fn softmax_all_non_negative(logits in finite_logits(-50.0, 50.0, 1..100)) {
        let mut probs = logits;
        softmax_in_place(&mut probs);
        for (i, &p) in probs.iter().enumerate() {
            prop_assert!(p >= 0.0, "probs[{}] = {} < 0", i, p);
        }
    }

    /// Softmax preserves the argmax position.
    #[test]
    fn softmax_preserves_argmax(logits in finite_logits(-10.0, 10.0, 2..50)) {
        let best_before = argmax(&logits);
        let mut probs = logits;
        softmax_in_place(&mut probs);
        let best_after = argmax(&probs);
        prop_assert_eq!(best_before, best_after);
    }

    /// Softmax on a single element yields 1.0.
    #[test]
    fn softmax_single_element(val in -1e6f32..=1e6) {
        let mut v = vec![val];
        softmax_in_place(&mut v);
        prop_assert!((v[0] - 1.0).abs() < 1e-6);
    }

    /// Softmax on uniform input yields uniform output.
    #[test]
    fn softmax_uniform_input_gives_uniform(val in -50.0f32..50.0, len in 2usize..50) {
        let mut v = vec![val; len];
        softmax_in_place(&mut v);
        #[allow(clippy::cast_precision_loss)]
        let expected = 1.0 / len as f32;
        for (i, &p) in v.iter().enumerate() {
            prop_assert!((p - expected).abs() < 1e-4,
                "probs[{}] = {}, expected {}", i, p, expected);
        }
    }
}

// ---------------------------------------------------------------------------
// Top-k properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// apply_top_k returns exactly k finite values (when k < len).
    #[test]
    fn top_k_returns_exactly_k(
        logits in finite_logits(-10.0, 10.0, 5..50),
        k in 1usize..5,
    ) {
        let mut copy = logits;
        let kept = apply_top_k(&mut copy, k);
        prop_assert_eq!(kept, k);
        let finite_count = copy.iter().filter(|v| v.is_finite()).count();
        prop_assert_eq!(finite_count, k);
    }

    /// apply_top_k with k == 0 is a no-op.
    #[test]
    fn top_k_zero_is_noop(logits in finite_logits(-10.0, 10.0, 1..20)) {
        let original = logits.clone();
        let mut copy = logits;
        apply_top_k(&mut copy, 0);
        prop_assert_eq!(copy, original);
    }

    /// apply_top_k with k >= len is a no-op.
    #[test]
    fn top_k_ge_len_is_noop(logits in finite_logits(-10.0, 10.0, 1..20)) {
        let original = logits.clone();
        let mut copy = logits.clone();
        apply_top_k(&mut copy, logits.len());
        prop_assert_eq!(copy, original.clone());

        let mut copy2 = logits.clone();
        apply_top_k(&mut copy2, logits.len() + 10);
        prop_assert_eq!(copy2, original);
    }

    /// apply_top_k always keeps the argmax.
    #[test]
    fn top_k_keeps_argmax(
        logits in finite_logits(-10.0, 10.0, 3..30),
        k in 1usize..3,
    ) {
        let best = argmax(&logits);
        let mut copy = logits;
        apply_top_k(&mut copy, k);
        prop_assert!(copy[best].is_finite(),
            "argmax position {} was filtered out", best);
    }

    /// Non-top-k entries are NEG_INFINITY.
    #[test]
    fn top_k_sets_non_top_to_neginf(
        logits in finite_logits(-10.0, 10.0, 5..30),
        k in 1usize..5,
    ) {
        let mut copy = logits;
        apply_top_k(&mut copy, k);
        for &v in &copy {
            prop_assert!(v.is_finite() || v == f32::NEG_INFINITY,
                "unexpected value: {}", v);
        }
    }
}

// ---------------------------------------------------------------------------
// Top-p properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// apply_top_p with p >= 1.0 is a no-op.
    #[test]
    fn top_p_one_is_noop(probs in normalized_probs(2..30)) {
        let original = probs.clone();
        let mut copy = probs;
        apply_top_p(&mut copy, 1.0);
        prop_assert_eq!(copy, original);
    }

    /// apply_top_p kept tokens' sum >= p.
    #[test]
    fn top_p_surviving_sum_ge_p(
        probs in normalized_probs(3..30),
        p in 0.1f32..0.99,
    ) {
        let mut copy = probs;
        apply_top_p(&mut copy, p);
        let surviving_sum: f32 = copy.iter().filter(|&&v| v > 0.0).sum();
        prop_assert!(surviving_sum >= p - 1e-4,
            "surviving sum {} < p {}", surviving_sum, p);
    }

    /// apply_top_p never removes the highest-probability token.
    #[test]
    fn top_p_keeps_max(
        probs in normalized_probs(2..30),
        p in 0.01f32..0.99,
    ) {
        let max_idx = probs.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap();
        let mut copy = probs;
        apply_top_p(&mut copy, p);
        prop_assert!(copy[max_idx] > 0.0,
            "max token at {} was zeroed", max_idx);
    }

    /// Zeroed tokens stay zero (no negative probabilities).
    #[test]
    fn top_p_no_negatives(
        probs in normalized_probs(3..30),
        p in 0.1f32..0.99,
    ) {
        let mut copy = probs;
        apply_top_p(&mut copy, p);
        for (i, &v) in copy.iter().enumerate() {
            prop_assert!(v >= 0.0, "probs[{}] = {} < 0", i, v);
        }
    }
}

// ---------------------------------------------------------------------------
// Repetition penalty properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Penalty == 1.0 is a no-op.
    #[test]
    fn rep_penalty_one_is_noop(
        logits in finite_logits(-10.0, 10.0, 1..20),
        ids in proptest::collection::vec(0u32..20, 1..5),
    ) {
        let original = logits.clone();
        let mut copy = logits;
        apply_repetition_penalty(&mut copy, &ids, 1.0);
        prop_assert_eq!(copy, original);
    }

    /// Penalty > 1 reduces positive logits and increases magnitude of negative logits.
    #[test]
    fn rep_penalty_direction(
        logits in proptest::collection::vec(
            prop_oneof![(-10.0f32..-0.01), (0.01f32..10.0)], 5..20
        ),
        ids in proptest::collection::vec(0u32..5, 1..3),
        penalty in 1.1f32..5.0,
    ) {
        let mut copy = logits.clone();
        apply_repetition_penalty(&mut copy, &ids, penalty);
        for &id in &ids {
            let idx = id as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    prop_assert!(copy[idx] <= logits[idx] + 1e-6,
                        "positive logit at {} increased", idx);
                } else if logits[idx] < 0.0 {
                    prop_assert!(copy[idx] <= logits[idx] + 1e-6,
                        "negative logit at {} didn't decrease", idx);
                }
            }
        }
    }

    /// Unseen tokens are not affected by repetition penalty.
    #[test]
    fn rep_penalty_unseen_unchanged(
        logits in finite_logits(-10.0, 10.0, 10..20),
        penalty in 1.1f32..5.0,
    ) {
        // Only penalize tokens 0..3; check that 5..9 are unchanged.
        let ids: Vec<u32> = (0..3).collect();
        let mut copy = logits.clone();
        apply_repetition_penalty(&mut copy, &ids, penalty);
        for i in 5..logits.len().min(10) {
            prop_assert!((copy[i] - logits[i]).abs() < 1e-6,
                "unseen token {} was modified", i);
        }
    }
}

// ---------------------------------------------------------------------------
// Min-p properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// min_p == 0 is a no-op.
    #[test]
    fn min_p_zero_is_noop(probs in normalized_probs(2..20)) {
        let original = probs.clone();
        let mut copy = probs;
        apply_min_p(&mut copy, 0.0);
        prop_assert_eq!(copy, original);
    }

    /// min_p never removes the highest-probability token.
    #[test]
    fn min_p_keeps_max(
        probs in normalized_probs(2..20),
        min_p in 0.0f32..1.0,
    ) {
        let max_idx = probs.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap();
        let mut copy = probs;
        apply_min_p(&mut copy, min_p);
        prop_assert!(copy[max_idx] > 0.0);
    }
}

// ---------------------------------------------------------------------------
// Argmax properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// argmax returns a valid index.
    #[test]
    fn argmax_valid_index(logits in finite_logits(-10.0, 10.0, 1..100)) {
        let idx = argmax(&logits);
        prop_assert!(idx < logits.len());
    }

    /// argmax value is the global maximum.
    #[test]
    fn argmax_is_global_max(logits in finite_logits(-10.0, 10.0, 1..100)) {
        let idx = argmax(&logits);
        let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        prop_assert!((logits[idx] - max_val).abs() < 1e-6);
    }
}
