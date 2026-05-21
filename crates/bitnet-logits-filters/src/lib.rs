//! Logit/probability filtering transforms extracted from `bitnet-logits`.
//!
//! This crate keeps filtering responsibilities isolated so downstream users can
//! depend only on top-k/top-p/min-p/typical transforms without pulling in the
//! rest of the logits pipeline surface.

use std::cmp::Ordering;

/// Zero out all but the top-`top_k` logits (by value).
///
/// Entries outside the top-k are set to `f32::NEG_INFINITY` so that a
/// subsequent softmax maps them to probability `0.0`.
///
/// Returns the number of non-`NEG_INFINITY` entries remaining.
/// If `top_k == 0` or `top_k >= logits.len()`, the slice is unchanged.
pub fn apply_top_k(logits: &mut [f32], top_k: usize) -> usize {
    let unmasked = logits.iter().filter(|&&x| x > f32::NEG_INFINITY).count();
    if top_k == 0 || top_k >= logits.len() || unmasked <= top_k {
        return unmasked;
    }

    let mut vals = Vec::with_capacity(unmasked);
    for &logit in logits.iter() {
        if logit > f32::NEG_INFINITY {
            vals.push(logit);
        }
    }

    let partition_idx = vals.len() - top_k;
    vals.select_nth_unstable_by(partition_idx, |a, b| f32_ascending(*a, *b));
    let threshold = vals[partition_idx];

    let mut kept = 0usize;
    for l in logits.iter_mut() {
        if *l >= threshold && kept < top_k {
            kept += 1;
        } else if *l > f32::NEG_INFINITY {
            *l = f32::NEG_INFINITY;
        }
    }
    kept
}

/// Nucleus (top-p) filtering on a **probability** slice (post-softmax).
///
/// Tokens are ranked by probability (descending). The smallest set whose
/// cumulative probability ≥ `top_p` is kept; all others are zeroed.
pub fn apply_top_p(probs: &mut [f32], top_p: f32) {
    if top_p >= 1.0 || probs.is_empty() {
        return;
    }

    let positive_count = probs.iter().filter(|&&p| p > 0.0).count();
    if positive_count <= 1 {
        return;
    }

    let mut indexed = Vec::with_capacity(positive_count);
    for (idx, &probability) in probs.iter().enumerate() {
        if probability > 0.0 {
            indexed.push((idx, probability));
        }
    }

    indexed.sort_unstable_by(|a, b| f32_descending(a.1, b.1));

    let mut cumsum = 0.0f64;
    let mut cutoff = indexed.len();
    for (rank, (_, p)) in indexed.iter().enumerate() {
        cumsum += f64::from(*p);
        if cumsum >= f64::from(top_p) {
            cutoff = rank + 1;
            break;
        }
    }

    for (_, (idx, _)) in indexed.iter().enumerate().skip(cutoff) {
        probs[*idx] = 0.0;
    }
}

/// Min-p filtering on a **probability** slice (post-softmax).
///
/// Zeroes out all tokens whose probability is below `min_p * max_probability`.
pub fn apply_min_p(probs: &mut [f32], min_p: f32) {
    if min_p <= 0.0 || probs.is_empty() {
        return;
    }

    let max_prob = probs.iter().copied().fold(0.0f32, f32::max);
    let threshold = min_p * max_prob;
    for p in probs.iter_mut() {
        // Skip writing zero over an already-zero slot; this avoids a needless
        // store in the common case of sparse probability vectors after a
        // top-k/top-p stage. The branch is cheap relative to the store, and
        // the visible result is identical for valid probabilities while still
        // preserving the old behavior for out-of-contract negative inputs.
        if *p != 0.0 && *p < threshold {
            *p = 0.0;
        }
    }
}

/// Locally typical sampling filter on a **probability** slice (post-softmax).
///
/// Keeps tokens whose "surprise" (negative log probability) is closest to
/// the expected surprise (entropy), until cumulative kept probability reaches
/// `typical_p`.
pub fn apply_typical(probs: &mut [f32], typical_p: f32) {
    if typical_filter::should_skip(typical_p, probs) {
        return;
    }

    let Some(mut deviations) = typical_filter::collect_deviations(probs) else {
        return;
    };

    typical_filter::normalize_deviations(&mut deviations);
    typical_filter::mask_tail_by_cumulative_probability(probs, &deviations, typical_p);
}

mod typical_filter {
    use super::f32_ascending;

    pub(super) type TypicalEntry = (usize, f32, f32);

    #[inline]
    pub(super) fn should_skip(typical_p: f32, probs: &[f32]) -> bool {
        typical_p >= 1.0 || probs.is_empty()
    }

    pub(super) fn collect_deviations(probs: &[f32]) -> Option<Vec<TypicalEntry>> {
        let mut entropy = 0.0f64;
        let mut entries: Vec<TypicalEntry> = Vec::with_capacity(probs.len());

        for (index, &probability) in probs.iter().enumerate() {
            if probability <= 0.0 {
                continue;
            }

            let surprise = -probability.ln();
            entropy += f64::from(probability * surprise);
            entries.push((index, probability, surprise));
        }

        if entries.is_empty() {
            return None;
        }

        for entry in &mut entries {
            entry.2 = (f64::from(entry.2) - entropy).abs() as f32;
        }

        Some(entries)
    }

    pub(super) fn normalize_deviations(entries: &mut [TypicalEntry]) {
        entries.sort_unstable_by(|left, right| f32_ascending(left.2, right.2));
    }

    pub(super) fn mask_tail_by_cumulative_probability(
        probs: &mut [f32],
        entries: &[TypicalEntry],
        typical_p: f32,
    ) {
        let cutoff = cutoff_index(entries, typical_p);
        for &(index, _, _) in entries.iter().skip(cutoff) {
            probs[index] = 0.0;
        }
    }

    fn cutoff_index(entries: &[TypicalEntry], typical_p: f32) -> usize {
        let mut cumulative_sum = 0.0f64;
        let mut cutoff = entries.len();

        for (rank, &(_, probability, _)) in entries.iter().enumerate() {
            cumulative_sum += f64::from(probability);
            if cumulative_sum >= f64::from(typical_p) {
                cutoff = rank + 1;
                break;
            }
        }

        cutoff
    }
}

#[inline]
fn f32_descending(a: f32, b: f32) -> Ordering {
    b.partial_cmp(&a).unwrap_or(Ordering::Equal)
}

#[inline]
fn f32_ascending(a: f32, b: f32) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_p_uses_accurate_cumulative_sum_for_large_vocab() {
        let mut probs = vec![1.0e-5f32; 100_000];
        apply_top_p(&mut probs, 0.5);

        let kept = probs.iter().filter(|&&p| p > 0.0).count();
        assert_eq!(kept, 50_001);
    }

    #[test]
    fn top_k_keeps_k_largest() {
        let mut logits = vec![1.0f32, 5.0, 3.0, 2.0, 4.0];
        let kept = apply_top_k(&mut logits, 2);
        assert_eq!(kept, 2);
        assert!(logits[1].is_finite());
        assert!(logits[4].is_finite());
        assert!(logits[0].is_infinite());
        assert!(logits[2].is_infinite());
        assert!(logits[3].is_infinite());
    }

    #[test]
    fn top_k_preserves_already_masked_logits_when_finite_count_is_within_k() {
        let mut logits = vec![f32::NEG_INFINITY, 5.0, f32::NEG_INFINITY, 2.0, 4.0];
        let kept = apply_top_k(&mut logits, 3);
        assert_eq!(kept, 3);
        assert!(logits[0].is_infinite());
        assert!(logits[1].is_finite());
        assert!(logits[2].is_infinite());
        assert!(logits[3].is_finite());
        assert!(logits[4].is_finite());
    }

    #[test]
    fn top_k_all_masked_input_keeps_zero_entries() {
        let mut logits = vec![f32::NEG_INFINITY; 4];
        let kept = apply_top_k(&mut logits, 2);
        assert_eq!(kept, 0);
        assert!(logits.iter().all(|value| *value == f32::NEG_INFINITY));
    }

    #[test]
    fn top_k_zero_returns_unmasked_count_not_slice_len() {
        // top_k == 0 is a no-op for the slice, but the return value must still
        // reflect the contract: "number of non-NEG_INFINITY entries remaining".
        let original = vec![f32::NEG_INFINITY, 5.0, f32::NEG_INFINITY, 2.0, 4.0];
        let mut logits = original.clone();
        let kept = apply_top_k(&mut logits, 0);
        assert_eq!(kept, 3, "kept must count unmasked entries, not slice length");
        assert_eq!(logits, original, "slice must be unchanged when top_k == 0");
    }

    #[test]
    fn top_k_at_or_above_len_returns_unmasked_count_not_slice_len() {
        // top_k >= len is a no-op for the slice, but the return value must
        // still equal the unmasked-entry count.
        let original = vec![f32::NEG_INFINITY, 5.0, f32::NEG_INFINITY, 2.0, 4.0];
        let mut logits = original.clone();
        let kept = apply_top_k(&mut logits, original.len());
        assert_eq!(kept, 3, "kept must count unmasked entries when top_k == len");
        assert_eq!(logits, original);

        let mut logits = original.clone();
        let kept = apply_top_k(&mut logits, original.len() + 10);
        assert_eq!(kept, 3, "kept must count unmasked entries when top_k > len");
        assert_eq!(logits, original);
    }

    #[test]
    fn top_p_removes_low_prob_tokens() {
        let mut probs = vec![0.5f32, 0.3, 0.2];
        apply_top_p(&mut probs, 0.8);
        assert!(probs[0] > 0.0);
        assert!(probs[1] > 0.0);
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(probs[2], 0.0);
        }
    }

    #[test]
    fn top_p_single_positive_probability_is_noop() {
        let mut probs = vec![0.0f32, 1.0, 0.0];
        apply_top_p(&mut probs, 0.5);
        assert_eq!(probs, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn min_p_filters_below_threshold() {
        let mut probs = vec![0.5f32, 0.3, 0.1, 0.05, 0.05];
        apply_min_p(&mut probs, 0.2);
        assert!(probs[0] > 0.0);
        assert!(probs[1] > 0.0);
        assert!(probs[2] > 0.0);
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(probs[3], 0.0);
            assert_eq!(probs[4], 0.0);
        }
    }

    #[test]
    fn typical_keeps_at_least_one_token() {
        let mut probs = vec![0.5f32, 0.25, 0.15, 0.07, 0.03];
        apply_typical(&mut probs, 0.5);
        let non_zero = probs.iter().filter(|&&p| p > 0.0).count();
        assert!(non_zero >= 1);
        assert!(non_zero < probs.len());
    }

    #[test]
    fn typical_handles_all_zero_distribution() {
        // The fused implementation must early-return without dividing by zero
        // or sorting an empty deviation vector.
        let mut probs = vec![0.0f32; 8];
        apply_typical(&mut probs, 0.5);
        assert!(probs.iter().all(|&p| p == 0.0));
    }

    #[test]
    fn typical_handles_single_nonzero_distribution() {
        // Only one token has positive probability — entropy collapses to
        // p * (-ln p), the single deviation is zero, so the token is kept.
        let mut probs = vec![0.0f32, 1.0, 0.0, 0.0];
        apply_typical(&mut probs, 0.5);
        assert!(probs[1] > 0.0);
        assert!(probs.iter().enumerate().filter(|&(_, &p)| p > 0.0).count() == 1);
    }

    #[test]
    fn typical_handles_sparse_post_topk_distribution() {
        // After a hypothetical top-k stage many entries are exactly zero. The
        // surprise cache must skip those without including them in the
        // entropy or sorting them as ties.
        let mut probs = vec![0.0f32, 0.0, 0.6, 0.0, 0.4, 0.0];
        apply_typical(&mut probs, 0.95);
        let kept: Vec<usize> =
            probs.iter().enumerate().filter(|&(_, &p)| p > 0.0).map(|(i, _)| i).collect();
        // The two non-zero entries must remain available; zeros stay zero.
        assert!(!kept.is_empty());
        for &i in &[0_usize, 1, 3, 5] {
            assert_eq!(probs[i], 0.0, "untouched zero at index {i}");
        }
    }

    #[test]
    fn typical_no_op_when_threshold_is_one() {
        // typical_p == 1.0 is the documented no-op short-circuit.
        let mut probs = vec![0.5f32, 0.25, 0.25];
        let original = probs.clone();
        apply_typical(&mut probs, 1.0);
        assert_eq!(probs, original);
    }

    #[test]
    fn min_p_skip_zero_writes_does_not_change_outputs() {
        // The "skip writing zero over an already-zero slot" optimization must
        // produce the same final vector as a naive pass that writes zero
        // unconditionally.
        let mut probs = vec![0.0f32, 0.5, 0.3, 0.0, 0.05, 0.0, -0.01];
        let mut reference = probs.clone();
        apply_min_p(&mut probs, 0.2);

        // Reference implementation: always write.
        let max_prob = reference.iter().copied().fold(0.0f32, f32::max);
        let threshold = 0.2 * max_prob;
        for p in reference.iter_mut() {
            if *p < threshold {
                *p = 0.0;
            }
        }
        assert_eq!(probs, reference);
    }

    #[test]
    fn min_p_keeps_max_token() {
        // With min_p > 0 the maximum token must always be kept (its value is
        // the threshold's reference point).
        let mut probs = vec![0.6f32, 0.3, 0.05, 0.05];
        apply_min_p(&mut probs, 0.5);
        assert!(probs[0] > 0.0);
    }

    proptest::proptest! {
        #[test]
        fn min_p_never_removes_max_token(
            probs in proptest::collection::vec(0.01f32..1.0f32, 2..32),
            min_p in 0.0f32..1.0f32,
        ) {
            let max_idx = probs.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            let mut filtered = probs;
            apply_min_p(&mut filtered, min_p);
            proptest::prop_assert!(filtered[max_idx] > 0.0);
        }
    }
}
