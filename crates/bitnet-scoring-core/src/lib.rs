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
    #[allow(clippy::cast_precision_loss)]
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
    pub fn add(&mut self, other: Self) {
        self.sum += other.sum;
        self.tokens += other.tokens;
    }

    /// Add one observed log-probability for a target token.
    #[inline]
    pub fn observe_logprob(&mut self, log_prob: f32) {
        self.sum -= f64::from(log_prob);
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
#[allow(clippy::float_cmp)]
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

    #[test]
    fn sanitize_logits_no_op_on_finite_slice() {
        let mut logits = vec![-3.0, 0.0, 2.5, 100.0];
        let before = logits.clone();
        sanitize_logits_in_place(&mut logits);
        assert_eq!(logits, before);
    }

    #[test]
    fn sanitize_logits_on_empty_slice() {
        let mut logits: Vec<f32> = Vec::new();
        sanitize_logits_in_place(&mut logits);
        assert!(logits.is_empty());
    }

    #[test]
    fn nll_stats_add_combines_sums_and_tokens() {
        let mut a = NllStats { sum: 1.0, tokens: 2 };
        let b = NllStats { sum: 3.5, tokens: 5 };
        a.add(b);
        assert_eq!(a.tokens, 7);
        assert!((a.sum - 4.5).abs() < 1e-9);
    }

    #[test]
    fn nll_stats_add_with_default_is_identity() {
        let mut a = NllStats { sum: 2.0, tokens: 3 };
        let before = a;
        a.add(NllStats::default());
        assert_eq!(a, before);
    }

    #[test]
    fn nll_stats_perplexity_reflects_mean() {
        let mut stats = NllStats::default();
        stats.observe_logprob(-1.0);
        stats.observe_logprob(-1.0);
        // mean NLL == 1.0, so perplexity == e.
        assert!((stats.perplexity() - std::f64::consts::E).abs() < 1e-9);
    }

    #[test]
    fn observe_target_nll_picks_target_log_softmax() {
        // Equal logits -> uniform distribution -> -log(1/n) for any target.
        let logits = vec![0.0_f32; 4];
        let mut stats = NllStats::default();
        observe_target_nll(&mut stats, &logits, 2);
        assert_eq!(stats.tokens, 1);
        let expected = 4_f64.ln();
        assert!((stats.sum - expected).abs() < 1e-5);
    }

    #[test]
    fn observe_target_nll_accumulates_multiple_observations() {
        let logits = vec![0.0_f32; 2];
        let mut stats = NllStats::default();
        observe_target_nll(&mut stats, &logits, 0);
        observe_target_nll(&mut stats, &logits, 1);
        assert_eq!(stats.tokens, 2);
        let expected = 2.0 * 2_f64.ln();
        assert!((stats.sum - expected).abs() < 1e-5);
    }
}
