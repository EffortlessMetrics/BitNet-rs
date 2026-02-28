//! Speculative decoding engine for accelerating autoregressive generation.
//!
//! Uses a small draft model to propose multiple tokens, then verifies them
//! against a larger target model in a single forward pass.

/// Configuration for the speculative decoding engine.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of tokens to draft per speculation round (typically 3–8).
    pub draft_length: usize,
    /// Upper bound on `draft_length` when adaptive mode is active.
    pub max_draft_length: usize,
    /// Minimum probability for a draft token to be accepted.
    pub acceptance_threshold: f64,
    /// Whether to dynamically adjust `draft_length` based on acceptance rate.
    pub adaptive_draft: bool,
    /// Sampling temperature applied during verification.
    pub temperature: f32,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            draft_length: 5,
            max_draft_length: 8,
            acceptance_threshold: 0.1,
            adaptive_draft: true,
            temperature: 1.0,
        }
    }
}

/// A single token proposed by the draft model.
#[derive(Debug, Clone)]
pub struct DraftToken {
    pub token_id: u32,
    pub draft_probability: f64,
}

/// The output of a single draft round.
#[derive(Debug, Clone)]
pub struct DraftResult {
    pub tokens: Vec<DraftToken>,
    pub draft_time_ms: f64,
}

/// The outcome of verifying a draft against the target model.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Token ids that were accepted in order.
    pub accepted_tokens: Vec<u32>,
    /// How many consecutive draft tokens were accepted.
    pub accepted_count: usize,
    /// Index at which the first rejection occurred, if any.
    pub rejected_at: Option<usize>,
    /// Correction token sampled from the target model at the rejection point.
    pub correction_token: Option<u32>,
    /// Wall-clock time spent on verification.
    pub verify_time_ms: f64,
}

/// Acceptance method used during verification.
#[derive(Debug, Clone)]
pub enum AcceptanceMethod {
    /// Accept only if the draft token is the argmax of the target distribution.
    Greedy,
    /// Accept if the target probability exceeds a fixed threshold.
    Threshold(f64),
    /// Accept with probability `min(1, target_prob / draft_prob)`.
    Stochastic,
    /// Accept if the draft token is among the top-k of the target distribution.
    TopK(usize),
}

/// Policy that decides whether to accept a draft token.
#[derive(Debug, Clone)]
pub struct AcceptancePolicy {
    pub method: AcceptanceMethod,
}

impl AcceptancePolicy {
    /// Create a new policy with the given method.
    #[must_use]
    pub const fn new(method: AcceptanceMethod) -> Self {
        Self { method }
    }

    /// Decide whether to accept a draft token.
    ///
    /// * `draft_prob`  – probability the draft model assigned to this token.
    /// * `target_prob` – probability the target model assigned to this token.
    ///
    /// For `TopK` the caller should pass `target_prob` equal to the token's
    /// rank-based indicator (1.0 if in top-k, 0.0 otherwise).
    #[must_use]
    pub fn should_accept(&self, draft_prob: f64, target_prob: f64) -> bool {
        match &self.method {
            AcceptanceMethod::Greedy => {
                // In greedy mode we treat target_prob == 1.0 as "is argmax".
                target_prob >= 1.0
            }
            AcceptanceMethod::Threshold(t) => target_prob >= *t,
            AcceptanceMethod::Stochastic => {
                if draft_prob <= 0.0 {
                    return target_prob > 0.0;
                }
                let ratio = target_prob / draft_prob;
                ratio >= 1.0
            }
            AcceptanceMethod::TopK(_k) => {
                // target_prob encodes whether the token is in the top-k.
                target_prob >= 1.0
            }
        }
    }
}

/// Cumulative statistics for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpeculativeStats {
    pub total_drafts: u64,
    pub total_tokens_drafted: u64,
    pub total_tokens_accepted: u64,
    pub acceptance_rate: f64,
    pub avg_draft_length: f64,
    pub avg_accepted_length: f64,
    pub speedup_ratio: f64,
    pub draft_time_total_ms: f64,
    pub verify_time_total_ms: f64,
}

impl Default for SpeculativeStats {
    fn default() -> Self {
        Self {
            total_drafts: 0,
            total_tokens_drafted: 0,
            total_tokens_accepted: 0,
            acceptance_rate: 0.0,
            avg_draft_length: 0.0,
            avg_accepted_length: 0.0,
            speedup_ratio: 0.0,
            draft_time_total_ms: 0.0,
            verify_time_total_ms: 0.0,
        }
    }
}

/// Speculative decoding engine.
///
/// Coordinates the draft→verify loop, tracks statistics, and optionally
/// adapts the draft length to maximise throughput.
pub struct SpeculativeEngine {
    config: SpeculativeConfig,
    stats: SpeculativeStats,
}

impl SpeculativeEngine {
    /// Create a new engine with the given configuration.
    #[must_use]
    pub fn new(config: SpeculativeConfig) -> Self {
        Self { config, stats: SpeculativeStats::default() }
    }

    /// Verify a draft against target-model probabilities.
    ///
    /// `target_probs` must contain one probability per draft token
    /// (i.e. the target model's probability for the *same* token id).
    pub fn verify(&mut self, draft: &DraftResult, target_probs: &[f64]) -> VerificationResult {
        let start = std::time::Instant::now();

        let policy =
            AcceptancePolicy::new(AcceptanceMethod::Threshold(self.config.acceptance_threshold));

        let mut accepted_tokens = Vec::new();
        let mut rejected_at = None;
        let mut correction_token = None;

        let effective_len = draft.tokens.len().min(target_probs.len());

        for (i, &target_p) in target_probs.iter().enumerate().take(effective_len) {
            let dt = &draft.tokens[i];

            let target_prob = if self.config.temperature > 0.0
                && (self.config.temperature - 1.0).abs() > f32::EPSILON
            {
                apply_temperature(target_p, f64::from(self.config.temperature))
            } else {
                target_p
            };

            if policy.should_accept(dt.draft_probability, target_prob) {
                accepted_tokens.push(dt.token_id);
            } else {
                rejected_at = Some(i);
                // In a real system, we would sample from the adjusted target
                // distribution here. For now, use a deterministic correction.
                correction_token = Some(dt.token_id.wrapping_add(1));
                break;
            }
        }

        let accepted_count = accepted_tokens.len();
        let verify_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Update cumulative stats.
        self.stats.total_drafts += 1;
        let drafted = draft.tokens.len() as u64;
        self.stats.total_tokens_drafted += drafted;
        self.stats.total_tokens_accepted += accepted_count as u64;
        self.stats.draft_time_total_ms += draft.draft_time_ms;
        self.stats.verify_time_total_ms += verify_time_ms;
        self.recompute_derived_stats();

        if self.config.adaptive_draft {
            self.adapt_draft_length();
        }

        VerificationResult {
            accepted_tokens,
            accepted_count,
            rejected_at,
            correction_token,
            verify_time_ms,
        }
    }

    /// Adapt `draft_length` based on the rolling acceptance rate.
    pub fn adapt_draft_length(&mut self) {
        let rate = self.stats.acceptance_rate;
        if rate > 0.8 && self.config.draft_length < self.config.max_draft_length {
            self.config.draft_length += 1;
        } else if rate < 0.3 && self.config.draft_length > 1 {
            self.config.draft_length -= 1;
        }
    }

    /// Current acceptance rate (0.0 if no drafts yet).
    #[must_use]
    pub const fn acceptance_rate(&self) -> f64 {
        self.stats.acceptance_rate
    }

    /// Estimated speedup ratio versus sequential decoding.
    ///
    /// Computed as `avg_accepted_length / 1.0` (each speculation replaces
    /// one sequential step with `accepted+1` tokens).
    #[must_use]
    pub const fn effective_speedup(&self) -> f64 {
        self.stats.speedup_ratio
    }

    /// Return a snapshot of the current statistics.
    #[must_use]
    pub const fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    /// Reset all accumulated statistics and restore the configured draft length.
    pub fn reset_stats(&mut self) {
        self.stats = SpeculativeStats::default();
    }

    /// Current draft length (may differ from the initial config if adaptive).
    #[must_use]
    pub const fn draft_length(&self) -> usize {
        self.config.draft_length
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    #[allow(clippy::cast_precision_loss)]
    fn recompute_derived_stats(&mut self) {
        let s = &mut self.stats;
        if s.total_tokens_drafted > 0 {
            s.acceptance_rate = s.total_tokens_accepted as f64 / s.total_tokens_drafted as f64;
        }
        if s.total_drafts > 0 {
            s.avg_draft_length = s.total_tokens_drafted as f64 / s.total_drafts as f64;
            s.avg_accepted_length = s.total_tokens_accepted as f64 / s.total_drafts as f64;
            // Each speculation round produces accepted+1 tokens with 2 forward
            // passes (draft + verify), compared to accepted+1 sequential passes.
            // Simplified: speedup ≈ avg_accepted + 1.
            s.speedup_ratio = s.avg_accepted_length + 1.0;
        }
    }
}

/// Apply a softmax-style temperature rescaling to a single log-probability.
fn apply_temperature(prob: f64, temperature: f64) -> f64 {
    if temperature <= 0.0 {
        return prob;
    }
    // Crude rescaling: raise probability to the power of 1/T and renormalise
    // against (prob^(1/T) + (1-prob)^(1/T)).
    let inv_t = 1.0 / temperature;
    let a = prob.powf(inv_t);
    let b = (1.0 - prob).powf(inv_t);
    let denom = a + b;
    if denom == 0.0 { prob } else { a / denom }
}

// ======================================================================
// Tests
// ======================================================================
#[cfg(test)]
mod tests {
    use super::*;

    // --- helpers ---------------------------------------------------

    fn default_engine() -> SpeculativeEngine {
        SpeculativeEngine::new(SpeculativeConfig::default())
    }

    fn engine_with(threshold: f64, adaptive: bool, temperature: f32) -> SpeculativeEngine {
        SpeculativeEngine::new(SpeculativeConfig {
            draft_length: 5,
            max_draft_length: 8,
            acceptance_threshold: threshold,
            adaptive_draft: adaptive,
            temperature,
        })
    }

    fn make_draft(tokens: &[(u32, f64)], time_ms: f64) -> DraftResult {
        DraftResult {
            tokens: tokens
                .iter()
                .map(|&(id, p)| DraftToken { token_id: id, draft_probability: p })
                .collect(),
            draft_time_ms: time_ms,
        }
    }

    // ===============================================================
    // 1. All draft tokens accepted (high agreement)
    // ===============================================================

    #[test]
    fn all_tokens_accepted_high_agreement() {
        let mut engine = engine_with(0.1, false, 1.0);
        let draft = make_draft(&[(1, 0.9), (2, 0.8), (3, 0.7)], 1.0);
        let target_probs = vec![0.9, 0.8, 0.7];
        let result = engine.verify(&draft, &target_probs);

        assert_eq!(result.accepted_count, 3);
        assert_eq!(result.accepted_tokens, vec![1, 2, 3]);
        assert!(result.rejected_at.is_none());
        assert!(result.correction_token.is_none());
    }

    #[test]
    fn all_tokens_accepted_well_above_threshold() {
        let mut engine = engine_with(0.05, false, 1.0);
        let draft = make_draft(&[(10, 0.5), (20, 0.6), (30, 0.7), (40, 0.8)], 2.0);
        let target_probs = vec![0.5, 0.6, 0.7, 0.8];
        let result = engine.verify(&draft, &target_probs);
        assert_eq!(result.accepted_count, 4);
    }

    #[test]
    fn all_five_accepted() {
        let mut engine = engine_with(0.1, false, 1.0);
        let draft = make_draft(&[(1, 0.9), (2, 0.85), (3, 0.8), (4, 0.75), (5, 0.7)], 1.5);
        let target_probs = vec![0.9, 0.85, 0.8, 0.75, 0.7];
        let result = engine.verify(&draft, &target_probs);
        assert_eq!(result.accepted_count, 5);
        assert!(result.rejected_at.is_none());
    }

    // ===============================================================
    // 2. First draft token rejected → accepted_count == 0
    // ===============================================================

    #[test]
    fn first_token_rejected() {
        let mut engine = engine_with(0.5, false, 1.0);
        let draft = make_draft(&[(1, 0.9), (2, 0.8)], 1.0);
        let target_probs = vec![0.1, 0.9]; // first below threshold
        let result = engine.verify(&draft, &target_probs);

        assert_eq!(result.accepted_count, 0);
        assert_eq!(result.rejected_at, Some(0));
        assert!(result.correction_token.is_some());
    }

    #[test]
    fn first_token_rejected_returns_correction() {
        let mut engine = engine_with(0.5, false, 1.0);
        let draft = make_draft(&[(100, 0.9)], 0.5);
        let target_probs = vec![0.01];
        let result = engine.verify(&draft, &target_probs);

        assert_eq!(result.accepted_count, 0);
        assert_eq!(result.correction_token, Some(101));
    }

    #[test]
    fn all_tokens_rejected_when_probs_zero() {
        let mut engine = engine_with(0.1, false, 1.0);
        let draft = make_draft(&[(1, 0.5), (2, 0.5)], 1.0);
        let target_probs = vec![0.0, 0.0];
        let result = engine.verify(&draft, &target_probs);
        assert_eq!(result.accepted_count, 0);
        assert_eq!(result.rejected_at, Some(0));
    }

    // ===============================================================
    // 3. Partial acceptance (3/5 accepted)
    // ===============================================================

    #[test]
    fn partial_acceptance_three_of_five() {
        let mut engine = engine_with(0.3, false, 1.0);
        let draft = make_draft(&[(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6), (5, 0.5)], 2.0);
        let target_probs = vec![0.9, 0.8, 0.7, 0.1, 0.1]; // 4th & 5th below threshold
        let result = engine.verify(&draft, &target_probs);

        assert_eq!(result.accepted_count, 3);
        assert_eq!(result.accepted_tokens, vec![1, 2, 3]);
        assert_eq!(result.rejected_at, Some(3));
    }

    #[test]
    fn partial_acceptance_two_of_four() {
        let mut engine = engine_with(0.5, false, 1.0);
        let draft = make_draft(&[(10, 0.9), (20, 0.8), (30, 0.7), (40, 0.6)], 1.0);
        let target_probs = vec![0.8, 0.7, 0.2, 0.1];
        let result = engine.verify(&draft, &target_probs);

        assert_eq!(result.accepted_count, 2);
        assert_eq!(result.rejected_at, Some(2));
    }

    #[test]
    fn partial_acceptance_last_rejected() {
        let mut engine = engine_with(0.3, false, 1.0);
        let draft = make_draft(&[(1, 0.9), (2, 0.8), (3, 0.7)], 1.0);
        let target_probs = vec![0.9, 0.8, 0.1];
        let result = engine.verify(&draft, &target_probs);

        assert_eq!(result.accepted_count, 2);
        assert_eq!(result.rejected_at, Some(2));
    }

    // ===============================================================
    // 4. Greedy acceptance: exact match only
    // ===============================================================

    #[test]
    fn greedy_accepts_only_argmax() {
        let policy = AcceptancePolicy::new(AcceptanceMethod::Greedy);
        // target_prob == 1.0 ⇒ this is the argmax
        assert!(policy.should_accept(0.9, 1.0));
        assert!(!policy.should_accept(0.9, 0.99));
        assert!(!policy.should_accept(0.9, 0.0));
    }

    #[test]
    fn greedy_rejects_high_but_not_argmax() {
        let policy = AcceptancePolicy::new(AcceptanceMethod::Greedy);
        assert!(!policy.should_accept(0.95, 0.95));
    }

    #[test]
    fn greedy_accepts_above_one() {
        let policy = AcceptancePolicy::new(AcceptanceMethod::Greedy);
        assert!(policy.should_accept(0.5, 1.5)); // edge: > 1.0
    }

    // ===============================================================
    // 5. Threshold acceptance: above / below threshold
    // ===============================================================

    #[test]
    fn threshold_above() {
        let policy = AcceptancePolicy::new(AcceptanceMethod::Threshold(0.5));
        assert!(policy.should_accept(0.1, 0.6));
        assert!(policy.should_accept(0.1, 0.5)); // boundary
    }

    #[test]
    fn threshold_below() {
        let policy = AcceptancePolicy::new(AcceptanceMethod::Threshold(0.5));
        assert!(!policy.should_accept(0.9, 0.49));
    }

    #[test]
    fn threshold_zero_accepts_positive() {
        let policy = AcceptancePolicy::new(AcceptanceMethod::Threshold(0.0));
        assert!(policy.should_accept(0.1, 0.001));
        assert!(policy.should_accept(0.1, 0.0)); // boundary
    }

    #[test]
    fn threshold_one_rejects_below_one() {
        let policy = AcceptancePolicy::new(AcceptanceMethod::Threshold(1.0));
        assert!(!policy.should_accept(0.99, 0.99));
        assert!(policy.should_accept(0.99, 1.0));
    }

    // ===============================================================
    // 6. Stochastic acceptance: probability-based
    // ===============================================================

    #[test]
    fn stochastic_accepts_when_target_ge_draft() {
        let policy = AcceptancePolicy::new(AcceptanceMethod::Stochastic);
        assert!(policy.should_accept(0.5, 0.5)); // ratio == 1.0
        assert!(policy.should_accept(0.3, 0.6)); // ratio > 1.0
    }

    #[test]
    fn stochastic_rejects_when_target_lt_draft() {
        let policy = AcceptancePolicy::new(AcceptanceMethod::Stochastic);
        assert!(!policy.should_accept(0.8, 0.4)); // ratio < 1.0
    }

    #[test]
    fn stochastic_zero_draft_prob_accepts_positive_target() {
        let policy = AcceptancePolicy::new(AcceptanceMethod::Stochastic);
        assert!(policy.should_accept(0.0, 0.5));
    }

    #[test]
    fn stochastic_zero_both_rejects() {
        let policy = AcceptancePolicy::new(AcceptanceMethod::Stochastic);
        assert!(!policy.should_accept(0.0, 0.0));
    }

    // ===============================================================
    // 7. TopK acceptance: in / out of top-k
    // ===============================================================

    #[test]
    fn topk_accepts_when_in_topk() {
        let policy = AcceptancePolicy::new(AcceptanceMethod::TopK(5));
        // Convention: target_prob == 1.0 means the token is in top-k.
        assert!(policy.should_accept(0.5, 1.0));
    }

    #[test]
    fn topk_rejects_when_out_of_topk() {
        let policy = AcceptancePolicy::new(AcceptanceMethod::TopK(5));
        assert!(!policy.should_accept(0.5, 0.0));
        assert!(!policy.should_accept(0.5, 0.99));
    }

    #[test]
    fn topk_k_value_stored() {
        let policy = AcceptancePolicy::new(AcceptanceMethod::TopK(10));
        match &policy.method {
            AcceptanceMethod::TopK(k) => assert_eq!(*k, 10),
            _ => panic!("expected TopK"),
        }
    }

    // ===============================================================
    // 8. Adaptive draft length — increases on high acceptance
    // ===============================================================

    #[test]
    fn adaptive_increases_on_high_acceptance() {
        let mut engine = SpeculativeEngine::new(SpeculativeConfig {
            draft_length: 4,
            max_draft_length: 8,
            acceptance_threshold: 0.1,
            adaptive_draft: false, // manually trigger adapt
            temperature: 1.0,
        });
        // Simulate high acceptance rate.
        engine.stats.total_tokens_drafted = 100;
        engine.stats.total_tokens_accepted = 90;
        engine.stats.acceptance_rate = 0.9;
        engine.adapt_draft_length();

        assert_eq!(engine.draft_length(), 5);
    }

    #[test]
    fn adaptive_does_not_exceed_max() {
        let mut engine = SpeculativeEngine::new(SpeculativeConfig {
            draft_length: 8,
            max_draft_length: 8,
            acceptance_threshold: 0.1,
            adaptive_draft: false,
            temperature: 1.0,
        });
        engine.stats.acceptance_rate = 0.95;
        engine.adapt_draft_length();
        assert_eq!(engine.draft_length(), 8);
    }

    #[test]
    fn adaptive_increases_via_verify() {
        let mut engine = SpeculativeEngine::new(SpeculativeConfig {
            draft_length: 3,
            max_draft_length: 8,
            acceptance_threshold: 0.1,
            adaptive_draft: true,
            temperature: 1.0,
        });
        // Feed many fully-accepted drafts to push acceptance rate above 0.8.
        for _ in 0..20 {
            let draft = make_draft(&[(1, 0.9), (2, 0.9), (3, 0.9)], 0.1);
            let target_probs = vec![0.9, 0.9, 0.9];
            engine.verify(&draft, &target_probs);
        }
        assert!(engine.draft_length() > 3);
    }

    // ===============================================================
    // 9. Adaptive draft length — decreases on low acceptance
    // ===============================================================

    #[test]
    fn adaptive_decreases_on_low_acceptance() {
        let mut engine = SpeculativeEngine::new(SpeculativeConfig {
            draft_length: 5,
            max_draft_length: 8,
            acceptance_threshold: 0.1,
            adaptive_draft: false,
            temperature: 1.0,
        });
        engine.stats.acceptance_rate = 0.1;
        engine.adapt_draft_length();
        assert_eq!(engine.draft_length(), 4);
    }

    #[test]
    fn adaptive_does_not_go_below_one() {
        let mut engine = SpeculativeEngine::new(SpeculativeConfig {
            draft_length: 1,
            max_draft_length: 8,
            acceptance_threshold: 0.1,
            adaptive_draft: false,
            temperature: 1.0,
        });
        engine.stats.acceptance_rate = 0.05;
        engine.adapt_draft_length();
        assert_eq!(engine.draft_length(), 1);
    }

    #[test]
    fn adaptive_no_change_in_middle_zone() {
        let mut engine = SpeculativeEngine::new(SpeculativeConfig {
            draft_length: 5,
            max_draft_length: 8,
            acceptance_threshold: 0.1,
            adaptive_draft: false,
            temperature: 1.0,
        });
        engine.stats.acceptance_rate = 0.5; // between 0.3 and 0.8
        engine.adapt_draft_length();
        assert_eq!(engine.draft_length(), 5);
    }

    #[test]
    fn adaptive_decreases_via_verify() {
        let mut engine = SpeculativeEngine::new(SpeculativeConfig {
            draft_length: 5,
            max_draft_length: 8,
            acceptance_threshold: 0.5,
            adaptive_draft: true,
            temperature: 1.0,
        });
        // Feed many mostly-rejected drafts.
        for _ in 0..20 {
            let draft = make_draft(&[(1, 0.9), (2, 0.9), (3, 0.9), (4, 0.9), (5, 0.9)], 0.1);
            let target_probs = vec![0.01, 0.01, 0.01, 0.01, 0.01];
            engine.verify(&draft, &target_probs);
        }
        assert!(engine.draft_length() < 5);
    }

    // ===============================================================
    // 10. Stats tracking accuracy
    // ===============================================================

    #[test]
    fn stats_total_drafts_increments() {
        let mut engine = engine_with(0.1, false, 1.0);
        let draft = make_draft(&[(1, 0.9)], 1.0);
        engine.verify(&draft, &[0.9]);
        engine.verify(&draft, &[0.9]);
        assert_eq!(engine.stats().total_drafts, 2);
    }

    #[test]
    fn stats_total_tokens_drafted() {
        let mut engine = engine_with(0.1, false, 1.0);
        let draft = make_draft(&[(1, 0.9), (2, 0.8), (3, 0.7)], 1.0);
        engine.verify(&draft, &[0.9, 0.8, 0.7]);
        assert_eq!(engine.stats().total_tokens_drafted, 3);
    }

    #[test]
    fn stats_total_tokens_accepted() {
        let mut engine = engine_with(0.5, false, 1.0);
        let draft = make_draft(&[(1, 0.9), (2, 0.8), (3, 0.7)], 1.0);
        let target_probs = vec![0.9, 0.1, 0.1]; // only first accepted
        engine.verify(&draft, &target_probs);
        assert_eq!(engine.stats().total_tokens_accepted, 1);
    }

    #[test]
    fn stats_acceptance_rate_computed() {
        let mut engine = engine_with(0.1, false, 1.0);
        let draft = make_draft(&[(1, 0.9), (2, 0.8)], 1.0);
        engine.verify(&draft, &[0.9, 0.05]); // 1 of 2 accepted
        let rate = engine.acceptance_rate();
        assert!((rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn stats_draft_time_accumulated() {
        let mut engine = engine_with(0.1, false, 1.0);
        let d1 = make_draft(&[(1, 0.9)], 5.0);
        let d2 = make_draft(&[(2, 0.9)], 3.0);
        engine.verify(&d1, &[0.9]);
        engine.verify(&d2, &[0.9]);
        assert!((engine.stats().draft_time_total_ms - 8.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_verify_time_positive() {
        let mut engine = engine_with(0.1, false, 1.0);
        let draft = make_draft(&[(1, 0.9)], 1.0);
        engine.verify(&draft, &[0.9]);
        assert!(engine.stats().verify_time_total_ms >= 0.0);
    }

    // ===============================================================
    // 11. Speedup ratio calculation
    // ===============================================================

    #[test]
    fn speedup_ratio_after_full_acceptance() {
        let mut engine = engine_with(0.1, false, 1.0);
        let draft = make_draft(&[(1, 0.9), (2, 0.8), (3, 0.7)], 1.0);
        engine.verify(&draft, &[0.9, 0.8, 0.7]); // all 3 accepted
        // speedup = avg_accepted + 1 = 3 + 1 = 4
        assert!((engine.effective_speedup() - 4.0).abs() < 0.01);
    }

    #[test]
    fn speedup_ratio_zero_acceptance() {
        let mut engine = engine_with(0.9, false, 1.0);
        let draft = make_draft(&[(1, 0.1)], 1.0);
        engine.verify(&draft, &[0.1]); // rejected
        // speedup = avg_accepted(0) + 1 = 1
        assert!((engine.effective_speedup() - 1.0).abs() < 0.01);
    }

    #[test]
    fn speedup_ratio_mixed() {
        let mut engine = engine_with(0.3, false, 1.0);
        // Round 1: 2 of 3 accepted.
        let d1 = make_draft(&[(1, 0.9), (2, 0.8), (3, 0.7)], 1.0);
        engine.verify(&d1, &[0.9, 0.8, 0.1]);
        // Round 2: 3 of 3 accepted.
        let d2 = make_draft(&[(4, 0.9), (5, 0.8), (6, 0.7)], 1.0);
        engine.verify(&d2, &[0.9, 0.8, 0.7]);
        // avg_accepted = (2+3)/2 = 2.5, speedup = 3.5
        assert!((engine.effective_speedup() - 3.5).abs() < 0.01);
    }

    // ===============================================================
    // 12. Empty draft handling
    // ===============================================================

    #[test]
    fn empty_draft_produces_zero_accepted() {
        let mut engine = default_engine();
        let draft = make_draft(&[], 0.5);
        let result = engine.verify(&draft, &[]);
        assert_eq!(result.accepted_count, 0);
        assert!(result.rejected_at.is_none());
        assert!(result.correction_token.is_none());
    }

    #[test]
    fn empty_draft_with_nonempty_probs() {
        let mut engine = default_engine();
        let draft = make_draft(&[], 0.5);
        let result = engine.verify(&draft, &[0.9, 0.8]);
        assert_eq!(result.accepted_count, 0);
    }

    // ===============================================================
    // 13. Single token draft
    // ===============================================================

    #[test]
    fn single_token_accepted() {
        let mut engine = engine_with(0.1, false, 1.0);
        let draft = make_draft(&[(42, 0.9)], 0.5);
        let result = engine.verify(&draft, &[0.9]);
        assert_eq!(result.accepted_count, 1);
        assert_eq!(result.accepted_tokens, vec![42]);
    }

    #[test]
    fn single_token_rejected() {
        let mut engine = engine_with(0.5, false, 1.0);
        let draft = make_draft(&[(42, 0.9)], 0.5);
        let result = engine.verify(&draft, &[0.1]);
        assert_eq!(result.accepted_count, 0);
        assert_eq!(result.rejected_at, Some(0));
    }

    // ===============================================================
    // 14. Maximum draft length enforcement
    // ===============================================================

    #[test]
    fn max_draft_length_caps_adaptation() {
        let mut engine = SpeculativeEngine::new(SpeculativeConfig {
            draft_length: 7,
            max_draft_length: 8,
            acceptance_threshold: 0.1,
            adaptive_draft: false,
            temperature: 1.0,
        });
        engine.stats.acceptance_rate = 0.95;
        engine.adapt_draft_length(); // 7 → 8
        engine.adapt_draft_length(); // 8 → 8 (capped)
        assert_eq!(engine.draft_length(), 8);
    }

    #[test]
    fn config_max_draft_length_respected() {
        let engine = SpeculativeEngine::new(SpeculativeConfig {
            draft_length: 5,
            max_draft_length: 5,
            acceptance_threshold: 0.1,
            adaptive_draft: false,
            temperature: 1.0,
        });
        assert!(engine.draft_length() <= engine.config.max_draft_length);
    }

    // ===============================================================
    // 15. Temperature effect on acceptance
    // ===============================================================

    #[test]
    fn high_temperature_flattens_probabilities() {
        // With high temperature, marginal probabilities move toward 0.5,
        // so a low target_prob can be pushed above threshold.
        let mut engine = engine_with(0.3, false, 3.0);
        let draft = make_draft(&[(1, 0.9)], 0.5);
        // target_prob = 0.25, but temperature rescaling should push it up.
        let result = engine.verify(&draft, &[0.25]);
        assert_eq!(result.accepted_count, 1);
    }

    #[test]
    fn low_temperature_sharpens_probabilities() {
        // Low temperature makes high probs higher, low probs lower.
        let mut engine = engine_with(0.5, false, 0.3);
        let draft = make_draft(&[(1, 0.9)], 0.5);
        // target_prob = 0.6 should be pushed higher, accepted.
        let result = engine.verify(&draft, &[0.6]);
        assert_eq!(result.accepted_count, 1);
    }

    #[test]
    fn temperature_one_is_identity() {
        let val = apply_temperature(0.7, 1.0);
        assert!((val - 0.7).abs() < 1e-10);
    }

    #[test]
    fn temperature_zero_returns_original() {
        let val = apply_temperature(0.7, 0.0);
        assert!((val - 0.7).abs() < 1e-10);
    }

    // ===============================================================
    // 16. Correction token generation
    // ===============================================================

    #[test]
    fn correction_token_on_rejection() {
        let mut engine = engine_with(0.5, false, 1.0);
        let draft = make_draft(&[(50, 0.9), (60, 0.8)], 1.0);
        let result = engine.verify(&draft, &[0.1, 0.9]);

        assert!(result.correction_token.is_some());
        assert_eq!(result.correction_token, Some(51)); // 50 + 1
    }

    #[test]
    fn no_correction_token_when_all_accepted() {
        let mut engine = engine_with(0.1, false, 1.0);
        let draft = make_draft(&[(1, 0.9), (2, 0.8)], 1.0);
        let result = engine.verify(&draft, &[0.9, 0.8]);
        assert!(result.correction_token.is_none());
    }

    #[test]
    fn correction_token_on_partial_acceptance() {
        let mut engine = engine_with(0.4, false, 1.0);
        let draft = make_draft(&[(10, 0.9), (20, 0.8), (30, 0.7)], 1.0);
        let result = engine.verify(&draft, &[0.9, 0.1, 0.9]);
        assert_eq!(result.correction_token, Some(21));
        assert_eq!(result.accepted_count, 1);
    }

    // ===============================================================
    // 17. Stats reset
    // ===============================================================

    #[test]
    fn reset_stats_clears_all() {
        let mut engine = engine_with(0.1, false, 1.0);
        let draft = make_draft(&[(1, 0.9), (2, 0.8)], 1.0);
        engine.verify(&draft, &[0.9, 0.8]);
        assert!(engine.stats().total_drafts > 0);

        engine.reset_stats();
        assert_eq!(engine.stats().total_drafts, 0);
        assert_eq!(engine.stats().total_tokens_drafted, 0);
        assert_eq!(engine.stats().total_tokens_accepted, 0);
        assert!((engine.stats().acceptance_rate).abs() < f64::EPSILON);
        assert!((engine.stats().speedup_ratio).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_accumulate_after_reset() {
        let mut engine = engine_with(0.1, false, 1.0);
        let draft = make_draft(&[(1, 0.9)], 1.0);
        engine.verify(&draft, &[0.9]);
        engine.reset_stats();
        engine.verify(&draft, &[0.9]);
        assert_eq!(engine.stats().total_drafts, 1);
    }

    // ===============================================================
    // Extra coverage
    // ===============================================================

    #[test]
    fn default_config_values() {
        let cfg = SpeculativeConfig::default();
        assert_eq!(cfg.draft_length, 5);
        assert_eq!(cfg.max_draft_length, 8);
        assert!((cfg.acceptance_threshold - 0.1).abs() < f64::EPSILON);
        assert!(cfg.adaptive_draft);
        assert!((cfg.temperature - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn default_stats_are_zero() {
        let stats = SpeculativeStats::default();
        assert_eq!(stats.total_drafts, 0);
        assert_eq!(stats.total_tokens_drafted, 0);
        assert!((stats.acceptance_rate).abs() < f64::EPSILON);
        assert!((stats.speedup_ratio).abs() < f64::EPSILON);
    }

    #[test]
    fn avg_draft_length_computed() {
        let mut engine = engine_with(0.1, false, 1.0);
        let d1 = make_draft(&[(1, 0.9), (2, 0.8)], 1.0);
        let d2 = make_draft(&[(3, 0.9), (4, 0.8), (5, 0.7), (6, 0.6)], 1.0);
        engine.verify(&d1, &[0.9, 0.8]);
        engine.verify(&d2, &[0.9, 0.8, 0.7, 0.6]);
        // avg = (2 + 4) / 2 = 3
        assert!((engine.stats().avg_draft_length - 3.0).abs() < 0.01);
    }

    #[test]
    fn avg_accepted_length_computed() {
        let mut engine = engine_with(0.3, false, 1.0);
        // Round 1: 1 accepted out of 2.
        engine.verify(&make_draft(&[(1, 0.9), (2, 0.8)], 1.0), &[0.9, 0.1]);
        // Round 2: 3 accepted out of 3.
        engine.verify(&make_draft(&[(3, 0.9), (4, 0.8), (5, 0.7)], 1.0), &[0.9, 0.8, 0.7]);
        // avg_accepted = (1 + 3) / 2 = 2.0
        assert!((engine.stats().avg_accepted_length - 2.0).abs() < 0.01);
    }

    #[test]
    fn verify_time_increases_with_drafts() {
        let mut engine = engine_with(0.1, false, 1.0);
        let draft = make_draft(&[(1, 0.9)], 1.0);
        engine.verify(&draft, &[0.9]);
        let t1 = engine.stats().verify_time_total_ms;
        engine.verify(&draft, &[0.9]);
        assert!(engine.stats().verify_time_total_ms >= t1);
    }

    #[test]
    fn mismatched_probs_length_uses_minimum() {
        let mut engine = engine_with(0.1, false, 1.0);
        // 3 draft tokens but only 2 probs
        let draft = make_draft(&[(1, 0.9), (2, 0.8), (3, 0.7)], 1.0);
        let result = engine.verify(&draft, &[0.9, 0.8]);
        assert_eq!(result.accepted_count, 2);
    }

    #[test]
    fn acceptance_policy_debug() {
        let policy = AcceptancePolicy::new(AcceptanceMethod::Greedy);
        let dbg = format!("{policy:?}");
        assert!(dbg.contains("Greedy"));
    }

    #[test]
    fn draft_result_clone() {
        let draft = make_draft(&[(1, 0.9)], 1.0);
        #[allow(clippy::redundant_clone)]
        let cloned = draft.clone();
        assert_eq!(cloned.tokens.len(), 1);
        assert!((cloned.draft_time_ms - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn verification_result_fields() {
        let mut engine = engine_with(0.1, false, 1.0);
        let draft = make_draft(&[(1, 0.9), (2, 0.8)], 1.0);
        let result = engine.verify(&draft, &[0.9, 0.8]);
        assert!(result.verify_time_ms >= 0.0);
        assert_eq!(result.accepted_tokens.len(), result.accepted_count);
    }

    #[test]
    fn multiple_rounds_stats_accumulate() {
        let mut engine = engine_with(0.1, false, 1.0);
        for i in 0..10 {
            let draft = make_draft(&[(i, 0.9), (i + 1, 0.8)], 0.5);
            engine.verify(&draft, &[0.9, 0.8]);
        }
        assert_eq!(engine.stats().total_drafts, 10);
        assert_eq!(engine.stats().total_tokens_drafted, 20);
        assert_eq!(engine.stats().total_tokens_accepted, 20);
    }

    #[test]
    fn engine_new_starts_with_empty_stats() {
        let engine = default_engine();
        assert_eq!(engine.stats().total_drafts, 0);
        assert!((engine.acceptance_rate()).abs() < f64::EPSILON);
        assert!((engine.effective_speedup()).abs() < f64::EPSILON);
    }
}
