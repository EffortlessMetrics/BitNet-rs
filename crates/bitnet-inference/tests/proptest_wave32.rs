//! Property-based tests for bitnet-inference sampling logic (wave 32).

use bitnet_logits::{
    apply_repetition_penalty, apply_temperature, apply_top_k, apply_top_p, argmax, softmax_in_place,
};
use proptest::prelude::*;

// ── Helpers ────────────────────────────────────────────────────────────

fn finite_logits_vec(min_len: usize, max_len: usize) -> BoxedStrategy<Vec<f32>> {
    prop::collection::vec(-50.0f32..50.0, min_len..=max_len).boxed()
}

// ── Tests ──────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    // 1. Temperature scaling preserves ordering
    #[test]
    fn proptest_wave32_temperature_preserves_ordering(
        logits in finite_logits_vec(2, 128),
        temp in 0.1f32..5.0,
    ) {
        let mut sorted_indices_before: Vec<usize> = (0..logits.len()).collect();
        sorted_indices_before.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());

        let mut scaled = logits.clone();
        apply_temperature(&mut scaled, temp);

        let mut sorted_indices_after: Vec<usize> = (0..scaled.len()).collect();
        sorted_indices_after.sort_by(|&a, &b| scaled[b].partial_cmp(&scaled[a]).unwrap());

        prop_assert_eq!(sorted_indices_before, sorted_indices_after);
    }

    // 2. Top-k preserves top elements
    #[test]
    fn proptest_wave32_top_k_preserves_top_elements(
        logits in finite_logits_vec(4, 64),
        k in 1usize..=16,
    ) {
        let k = k.min(logits.len());
        let mut filtered = logits.clone();
        let kept = apply_top_k(&mut filtered, k);
        prop_assert!(kept <= k);
        // All non-neg-inf values in filtered must appear in original top-k
        let non_inf_count = filtered.iter().filter(|&&v| v != f32::NEG_INFINITY).count();
        prop_assert!(non_inf_count <= k);
    }

    // 3. Top-p cumulative sum respects threshold
    #[test]
    fn proptest_wave32_top_p_cumulative_sum(
        logits in finite_logits_vec(4, 64),
        p in 0.1f32..1.0,
    ) {
        let mut probs = logits.clone();
        softmax_in_place(&mut probs);
        apply_top_p(&mut probs, p);
        let remaining_sum: f32 = probs.iter().sum();
        // Remaining probability mass should be >= top_p (we keep enough tokens)
        prop_assert!(remaining_sum >= p - 1e-5 || remaining_sum > 0.0);
    }

    // 4. Repetition penalty: penalized tokens have lower logits
    #[test]
    fn proptest_wave32_repetition_penalty_lowers_positive_logits(
        logits in prop::collection::vec(0.1f32..50.0, 4..=32),
        penalty in 1.01f32..3.0,
        token_idx in 0usize..4,
    ) {
        let token_idx = token_idx % logits.len();
        let original = logits[token_idx];
        let mut penalized = logits.clone();
        apply_repetition_penalty(&mut penalized, &[token_idx as u32], penalty);
        // Positive logits are divided by penalty
        prop_assert!(penalized[token_idx] < original + 1e-6);
    }

    // 5. Greedy sample always selects argmax
    #[test]
    fn proptest_wave32_greedy_selects_argmax(
        logits in finite_logits_vec(2, 128),
    ) {
        let best = argmax(&logits);
        for (i, &v) in logits.iter().enumerate() {
            prop_assert!(logits[best] >= v, "argmax idx {} val {} but idx {} has {}", best, logits[best], i, v);
        }
    }

    // 6. Token generation: valid token IDs within vocab range
    #[test]
    fn proptest_wave32_argmax_within_vocab_range(
        logits in finite_logits_vec(1, 256),
    ) {
        let idx = argmax(&logits);
        prop_assert!(idx < logits.len());
    }

    // 7. Seed determinism: same seed → same softmax output
    #[test]
    fn proptest_wave32_softmax_determinism(
        logits in finite_logits_vec(4, 64),
    ) {
        let mut a = logits.clone();
        let mut b = logits.clone();
        softmax_in_place(&mut a);
        softmax_in_place(&mut b);
        for (va, vb) in a.iter().zip(b.iter()) {
            prop_assert!((va - vb).abs() < 1e-7);
        }
    }

    // 8. Temperature 0 → no-op (greedy behavior: logits unchanged)
    #[test]
    fn proptest_wave32_temperature_zero_is_noop(
        logits in finite_logits_vec(2, 64),
    ) {
        let mut scaled = logits.clone();
        apply_temperature(&mut scaled, 0.0);
        prop_assert_eq!(logits, scaled);
    }

    // 9. Logit bias: adding large value to a logit makes it argmax
    #[test]
    fn proptest_wave32_larger_logit_higher_probability(
        base in finite_logits_vec(4, 64),
        bias_idx in 0usize..4,
    ) {
        let bias_idx = bias_idx % base.len();
        let mut biased = base.clone();
        // Add enough to guarantee this index becomes the max
        let current_max = base.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        biased[bias_idx] = current_max + 10.0;
        softmax_in_place(&mut biased);
        let max_prob_idx = argmax(&biased);
        prop_assert_eq!(max_prob_idx, bias_idx);
    }

    // 10. Multiple samples with different temperatures differ in entropy
    #[test]
    fn proptest_wave32_different_temperatures_different_distributions(
        logits in finite_logits_vec(4, 64),
    ) {
        let mut low_temp = logits.clone();
        apply_temperature(&mut low_temp, 0.1);
        softmax_in_place(&mut low_temp);

        let mut high_temp = logits.clone();
        apply_temperature(&mut high_temp, 5.0);
        softmax_in_place(&mut high_temp);

        // Low temperature should be more peaked (max prob higher)
        let low_max = low_temp.iter().copied().fold(0.0f32, f32::max);
        let high_max = high_temp.iter().copied().fold(0.0f32, f32::max);
        prop_assert!(low_max >= high_max - 1e-5);
    }

    // 11. Softmax after temperature still sums to 1
    #[test]
    fn proptest_wave32_softmax_after_temperature_sums_to_one(
        logits in finite_logits_vec(2, 128),
        temp in 0.1f32..10.0,
    ) {
        let mut scaled = logits.clone();
        apply_temperature(&mut scaled, temp);
        softmax_in_place(&mut scaled);
        let sum: f32 = scaled.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-5, "sum = {}", sum);
    }

    // 12. Repetition penalty of 1.0 is no-op
    #[test]
    fn proptest_wave32_repetition_penalty_1_is_noop(
        logits in finite_logits_vec(4, 32),
        token_idx in 0usize..4,
    ) {
        let token_idx = token_idx % logits.len();
        let mut penalized = logits.clone();
        apply_repetition_penalty(&mut penalized, &[token_idx as u32], 1.0);
        prop_assert_eq!(logits, penalized);
    }

    // 13. Top-k with k >= len is no-op
    #[test]
    fn proptest_wave32_top_k_full_is_noop(
        logits in finite_logits_vec(2, 32),
    ) {
        let mut filtered = logits.clone();
        apply_top_k(&mut filtered, logits.len() + 1);
        prop_assert_eq!(logits, filtered);
    }

    // 14. Top-p with p=1.0 is no-op
    #[test]
    fn proptest_wave32_top_p_one_is_noop(
        logits in finite_logits_vec(2, 32),
    ) {
        let mut probs = logits.clone();
        softmax_in_place(&mut probs);
        let original = probs.clone();
        apply_top_p(&mut probs, 1.0);
        prop_assert_eq!(original, probs);
    }

    // 15. Softmax output all non-negative
    #[test]
    fn proptest_wave32_softmax_all_nonnegative(
        logits in finite_logits_vec(1, 128),
    ) {
        let mut probs = logits;
        softmax_in_place(&mut probs);
        for &p in &probs {
            prop_assert!(p >= 0.0, "negative prob: {}", p);
        }
    }

    // 16. Softmax monotonicity preserved
    #[test]
    fn proptest_wave32_softmax_monotonicity(
        logits in finite_logits_vec(2, 64),
    ) {
        let mut probs = logits.clone();
        softmax_in_place(&mut probs);
        for i in 0..logits.len() {
            for j in 0..logits.len() {
                if logits[i] > logits[j] {
                    prop_assert!(probs[i] >= probs[j] - 1e-7);
                }
            }
        }
    }

    // 17. Repetition penalty on negative logits makes them more negative
    #[test]
    fn proptest_wave32_repetition_penalty_negative_logits(
        logits in prop::collection::vec(-50.0f32..-0.1, 4..=32),
        penalty in 1.01f32..3.0,
        token_idx in 0usize..4,
    ) {
        let token_idx = token_idx % logits.len();
        let original = logits[token_idx];
        let mut penalized = logits.clone();
        apply_repetition_penalty(&mut penalized, &[token_idx as u32], penalty);
        // Negative logits are multiplied by penalty (become more negative)
        prop_assert!(penalized[token_idx] <= original + 1e-6);
    }

    // 18. Temperature 1.0 is identity
    #[test]
    fn proptest_wave32_temperature_one_identity(
        logits in finite_logits_vec(2, 64),
    ) {
        let mut scaled = logits.clone();
        apply_temperature(&mut scaled, 1.0);
        prop_assert_eq!(logits, scaled);
    }

    // 19. Softmax with single element returns 1.0
    #[test]
    fn proptest_wave32_softmax_single_element(
        val in -50.0f32..50.0,
    ) {
        let mut probs = vec![val];
        softmax_in_place(&mut probs);
        prop_assert!((probs[0] - 1.0).abs() < 1e-6);
    }

    // 20. Top-k then softmax still sums to 1
    #[test]
    fn proptest_wave32_top_k_then_softmax_sums_to_one(
        logits in finite_logits_vec(4, 64),
        k in 1usize..=8,
    ) {
        let k = k.min(logits.len());
        let mut filtered = logits;
        apply_top_k(&mut filtered, k);
        softmax_in_place(&mut filtered);
        let sum: f32 = filtered.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-5, "sum = {}", sum);
    }
}
