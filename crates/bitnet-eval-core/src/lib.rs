//! Shared, deterministic evaluation helpers used by CLI scoring/eval commands.

use core::cmp::Ordering;

/// Deterministic top-k index selection over logits.
///
/// Sorts by logit descending and then by index ascending for ties.
#[must_use]
pub fn topk_stable_indices(logits: &[f32], k: usize) -> Vec<usize> {
    if k == 0 {
        return Vec::new();
    }

    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| match logits[b].partial_cmp(&logits[a]) {
        Some(Ordering::Less) => Ordering::Less,
        Some(Ordering::Greater) => Ordering::Greater,
        _ => a.cmp(&b),
    });

    idx.truncate(k);
    idx
}

/// Numerically stable log-softmax.
#[must_use]
pub fn log_softmax_stable(xs: &[f32]) -> Vec<f32> {
    let mut m = f32::NEG_INFINITY;
    for &v in xs {
        if v > m {
            m = v;
        }
    }
    let mut sum = 0.0f32;
    for &v in xs {
        sum += (v - m).exp();
    }
    let lse = m + sum.ln();
    xs.iter().map(|&v| v - lse).collect()
}

/// L2 divergence (Euclidean distance) between two vectors.
///
/// Returns `f64::INFINITY` when vector lengths do not match.
#[must_use]
pub fn l2_divergence(baseline: &[f32], canary: &[f32]) -> f64 {
    if baseline.len() != canary.len() {
        return f64::INFINITY;
    }

    let sum_sq: f64 =
        baseline.iter().zip(canary.iter()).map(|(a, b)| ((*a as f64) - (*b as f64)).powi(2)).sum();

    sum_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::{l2_divergence, log_softmax_stable, topk_stable_indices};

    #[test]
    fn topk_is_deterministic_for_ties() {
        let logits = vec![0.5f32, 1.0, 1.0, 0.2];
        let top2 = topk_stable_indices(&logits, 2);
        assert_eq!(top2, vec![1, 2]);
    }

    #[test]
    fn log_softmax_outputs_normalized_distribution_when_expd() {
        let logits = vec![1.0f32, 2.0, 3.0];
        let logp = log_softmax_stable(&logits);
        let p_sum: f32 = logp.iter().map(|v| v.exp()).sum();
        assert!((p_sum - 1.0).abs() < 1e-5, "sum was {p_sum}");
    }

    #[test]
    fn l2_divergence_zero_for_identical_vectors() {
        let a = vec![0.1f32, -2.5, 3.0, 9.75];
        assert_eq!(l2_divergence(&a, &a), 0.0);
    }

    #[test]
    fn l2_divergence_returns_infinity_for_mismatched_lengths() {
        assert!(l2_divergence(&[1.0f32, 2.0], &[1.0]).is_infinite());
    }
}
