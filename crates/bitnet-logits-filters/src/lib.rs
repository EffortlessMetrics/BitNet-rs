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
    if top_k == 0 || top_k >= logits.len() {
        return logits.len();
    }

    let mut vals: Vec<f32> = logits.iter().copied().filter(|&x| x > f32::NEG_INFINITY).collect();
    if vals.len() <= top_k {
        return vals.len();
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

    let mut non_zero_count = 0;
    for &p in probs.iter() {
        if p > 0.0 {
            non_zero_count += 1;
        }
    }

    if non_zero_count <= 1 {
        return;
    }

    let mut indexed: Vec<(usize, f32)> = Vec::with_capacity(non_zero_count);
    for (idx, &p) in probs.iter().enumerate() {
        if p > 0.0 {
            indexed.push((idx, p));
        }
    }

    indexed.sort_unstable_by(|a, b| f32_descending(a.1, b.1));

    let mut cumsum = 0.0f32;
    let mut cutoff = indexed.len();
    for (rank, (_, p)) in indexed.iter().enumerate() {
        cumsum += p;
        if cumsum >= top_p {
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
        if *p < threshold {
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
    if typical_p >= 1.0 || probs.is_empty() {
        return;
    }

    let mut non_zero_count = 0;
    for &p in probs.iter() {
        if p > 0.0 {
            non_zero_count += 1;
        }
    }

    if non_zero_count <= 1 {
        return;
    }

    let mut indexed: Vec<(usize, f32)> = Vec::with_capacity(non_zero_count);
    for (idx, &p) in probs.iter().enumerate() {
        if p > 0.0 {
            indexed.push((idx, p));
        }
    }

    let entropy: f32 = indexed.iter().map(|&(_, p)| -p * p.ln()).sum();

    let mut deviations: Vec<(usize, f32, f32)> = Vec::with_capacity(non_zero_count);
    for (i, p) in indexed {
        let surprise = -p.ln();
        let deviation = (surprise - entropy).abs();
        deviations.push((i, p, deviation));
    }

    deviations.sort_unstable_by(|a, b| f32_ascending(a.2, b.2));

    let mut cumsum = 0.0f32;
    let mut cutoff = deviations.len();
    for (rank, &(_, p, _)) in deviations.iter().enumerate() {
        cumsum += p;
        if cumsum >= typical_p {
            cutoff = rank + 1;
            break;
        }
    }

    for &(idx, _, _) in deviations.iter().skip(cutoff) {
        probs[idx] = 0.0;
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
