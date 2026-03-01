//! Shared scoring primitives for negative log-likelihood evaluation.

/// Running sum of NLL and the number of predicted tokens (T-1).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NllStats {
    /// Total negative log-likelihood over predicted tokens.
    pub sum: f64,
    /// Number of predicted tokens (T-1), padding excluded.
    pub tokens: usize,
}

impl NllStats {
    /// Mean negative log-likelihood.
    #[must_use]
    #[inline]
    pub fn mean(self) -> f64 {
        if self.tokens > 0 { self.sum / self.tokens as f64 } else { 0.0 }
    }

    /// Perplexity computed as `exp(mean_nll)`.
    #[must_use]
    #[inline]
    pub fn perplexity(self) -> f64 {
        self.mean().exp()
    }

    /// Accumulate another stats sample.
    #[inline]
    pub fn add(&mut self, other: NllStats) {
        self.sum += other.sum;
        self.tokens += other.tokens;
    }

    /// Add one observed log-probability for a target token.
    #[inline]
    pub fn observe_logprob(&mut self, log_prob: f32) {
        self.sum -= log_prob as f64;
        self.tokens += 1;
    }
}

/// Replaces any non-finite logit value with `NEG_INFINITY`.
#[inline]
pub fn sanitize_logits_in_place(logits: &mut [f32]) {
    for v in logits {
        if !v.is_finite() {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Accumulate NLL from a target token index using stable log-softmax.
#[inline]
pub fn observe_target_nll(stats: &mut NllStats, logits: &[f32], target: usize) {
    let logp = bitnet_eval_core::log_softmax_stable(logits);
    stats.observe_logprob(logp[target]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nll_stats_mean_and_perplexity_are_stable_on_empty() {
        let stats = NllStats::default();
        assert_eq!(stats.mean(), 0.0);
        assert_eq!(stats.perplexity(), 1.0);
    }

    #[test]
    fn nll_stats_accumulates_observations() {
        let mut stats = NllStats::default();
        stats.observe_logprob(-0.5);
        stats.observe_logprob(-1.5);

        assert_eq!(stats.tokens, 2);
        assert!((stats.sum - 2.0).abs() < 1e-9);
        assert!((stats.mean() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn sanitize_logits_maps_nan_and_infinity_to_neg_inf() {
        let mut logits = vec![1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -2.0];
        sanitize_logits_in_place(&mut logits);

        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[3], f32::NEG_INFINITY);
        assert_eq!(logits[4], -2.0);
    }
}
