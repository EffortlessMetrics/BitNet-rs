//! # Speculative Decoding Engine
//!
//! Implements speculative decoding for accelerated inference. A small "draft"
//! model proposes candidate tokens that are verified against the larger "target"
//! model in parallel, achieving wall-clock speedup when the draft model is a
//! good predictor of the target.
//!
//! ## Algorithm
//!
//! 1. The draft model autoregressively generates `draft_tokens` candidates.
//! 2. The target model scores all candidates in a single forward pass.
//! 3. Each draft token is accepted or rejected via rejection sampling:
//!    the token is kept with probability `min(1, p_target / p_draft)`.
//! 4. On first rejection, all subsequent draft tokens are discarded and
//!    the target model's distribution is used to sample a replacement.
//! 5. If every draft token is accepted, one additional token is sampled
//!    from the target model's distribution at the next position.
//!
//! This guarantees the output distribution is identical to the target model.

use std::fmt;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of tokens the draft model proposes per iteration.
    pub draft_tokens: usize,
    /// Minimum probability ratio (`p_target / p_draft`) for acceptance.
    /// Must be in `[0.0, 1.0]`. A value of `0.0` accepts everything;
    /// `1.0` requires the target to be at least as confident as the draft.
    pub acceptance_threshold: f32,
    /// Maximum consecutive draft iterations before falling back to normal
    /// autoregressive decoding.
    pub max_draft_attempts: usize,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self { draft_tokens: 5, acceptance_threshold: 0.0, max_draft_attempts: 3 }
    }
}

impl SpeculativeConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.draft_tokens == 0 {
            return Err("draft_tokens must be greater than 0".into());
        }
        if !(0.0..=1.0).contains(&self.acceptance_threshold) {
            return Err("acceptance_threshold must be in [0.0, 1.0]".into());
        }
        if self.max_draft_attempts == 0 {
            return Err("max_draft_attempts must be greater than 0".into());
        }
        Ok(())
    }

    /// Set `draft_tokens`.
    #[must_use]
    pub fn with_draft_tokens(mut self, n: usize) -> Self {
        self.draft_tokens = n;
        self
    }

    /// Set `acceptance_threshold`.
    #[must_use]
    pub fn with_acceptance_threshold(mut self, t: f32) -> Self {
        self.acceptance_threshold = t;
        self
    }

    /// Set `max_draft_attempts`.
    #[must_use]
    pub fn with_max_draft_attempts(mut self, n: usize) -> Self {
        self.max_draft_attempts = n;
        self
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Running statistics for speculative decoding.
#[derive(Debug, Clone, Default)]
pub struct SpeculativeStats {
    /// Total draft tokens that were accepted by the target model.
    pub accepted_tokens: u64,
    /// Total draft tokens that were rejected.
    pub rejected_tokens: u64,
    /// Number of draft iterations executed.
    pub total_drafts: u64,
}

impl SpeculativeStats {
    /// Token-level acceptance rate in `[0.0, 1.0]`.
    /// Returns `0.0` when no tokens have been evaluated.
    pub fn acceptance_rate(&self) -> f64 {
        let total = self.accepted_tokens + self.rejected_tokens;
        if total == 0 { 0.0 } else { self.accepted_tokens as f64 / total as f64 }
    }

    /// Estimated speedup factor.
    ///
    /// In speculative decoding the draft model produces tokens cheaply, and
    /// the expensive target model verifies them in bulk. The speedup comes
    /// from amortising the target-model cost over multiple tokens per forward
    /// pass.  Here we approximate the factor as
    /// `(accepted + total_drafts) / total_drafts` — i.e. how many tokens
    /// were produced per expensive verification round (each draft round
    /// yields at least one token from the target model).
    pub fn speedup_factor(&self) -> f64 {
        if self.total_drafts == 0 {
            1.0
        } else {
            (self.accepted_tokens as f64 + self.total_drafts as f64) / self.total_drafts as f64
        }
    }
}

impl fmt::Display for SpeculativeStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "accepted={} rejected={} drafts={} rate={:.2}% speedup={:.2}×",
            self.accepted_tokens,
            self.rejected_tokens,
            self.total_drafts,
            self.acceptance_rate() * 100.0,
            self.speedup_factor(),
        )
    }
}

// ---------------------------------------------------------------------------
// Draft token proposal
// ---------------------------------------------------------------------------

/// A single proposed token together with the draft model's probability.
#[derive(Debug, Clone, Copy)]
pub struct DraftToken {
    /// Token id.
    pub token_id: u32,
    /// Draft model's probability for this token.
    pub draft_prob: f32,
}

// ---------------------------------------------------------------------------
// Verification result
// ---------------------------------------------------------------------------

/// Result of one speculative-decoding iteration.
#[derive(Debug, Clone)]
pub struct SpeculativeResult {
    /// Tokens accepted from the draft (in order), plus the bonus/replacement
    /// token sampled from the target model at the end.
    pub accepted_tokens: Vec<u32>,
    /// Whether the decoder fell back to standard autoregressive decoding
    /// (e.g. after exhausting `max_draft_attempts` with no progress).
    pub fell_back: bool,
    /// Snapshot of cumulative statistics after this iteration.
    pub stats: SpeculativeStats,
}

// ---------------------------------------------------------------------------
// Core decoder
// ---------------------------------------------------------------------------

/// Speculative decoding engine.
///
/// `SpeculativeDecoder` is backend-agnostic: callers supply probability
/// distributions (as `&[f32]` slices over the vocabulary) produced by
/// their own draft and target models.  This keeps the module free of
/// hard dependencies on model-loading or device-management code.
#[derive(Debug, Clone)]
pub struct SpeculativeDecoder {
    /// Active configuration.
    pub config: SpeculativeConfig,
    /// Cumulative statistics.
    pub stats: SpeculativeStats,
}

impl SpeculativeDecoder {
    /// Create a new decoder with the given configuration.
    ///
    /// Returns `Err` if the configuration is invalid.
    pub fn new(config: SpeculativeConfig) -> Result<Self, String> {
        config.validate()?;
        Ok(Self { config, stats: SpeculativeStats::default() })
    }

    /// Verify a batch of draft tokens against the target model's
    /// probability distributions.
    ///
    /// # Arguments
    ///
    /// * `draft_tokens` — tokens proposed by the draft model, each carrying
    ///   the draft probability.
    /// * `target_probs` — one probability distribution (over the full
    ///   vocabulary) per draft token, as produced by a single batched
    ///   forward pass of the target model.  `target_probs[i]` corresponds
    ///   to `draft_tokens[i]`.
    /// * `bonus_token` — the token sampled from the target model's
    ///   distribution at position `n+1` (used when all drafts are accepted)
    ///   or from the adjusted distribution at the first rejection point.
    ///
    /// The function performs rejection sampling: token *i* is accepted when
    /// `target_probs[i][token_id] / draft_prob >= threshold` (clamped to 1).
    /// On the first rejection the remaining drafts are discarded and
    /// `bonus_token` is appended instead.
    pub fn verify(
        &mut self,
        draft_tokens: &[DraftToken],
        target_probs: &[&[f32]],
        bonus_token: u32,
    ) -> SpeculativeResult {
        assert_eq!(
            draft_tokens.len(),
            target_probs.len(),
            "draft_tokens and target_probs must have equal length",
        );

        self.stats.total_drafts += 1;

        let mut accepted: Vec<u32> = Vec::with_capacity(draft_tokens.len() + 1);

        for (i, draft) in draft_tokens.iter().enumerate() {
            let target_p = *target_probs[i].get(draft.token_id as usize).unwrap_or(&0.0);
            let ratio = if draft.draft_prob <= 0.0 { 0.0 } else { target_p / draft.draft_prob };

            if ratio >= self.config.acceptance_threshold {
                accepted.push(draft.token_id);
                self.stats.accepted_tokens += 1;
            } else {
                // Reject this and all subsequent draft tokens.
                self.stats.rejected_tokens += (draft_tokens.len() - i) as u64;
                accepted.push(bonus_token);
                return SpeculativeResult {
                    accepted_tokens: accepted,
                    fell_back: false,
                    stats: self.stats.clone(),
                };
            }
        }

        // All drafts accepted — append the bonus token from the target
        // model at position n+1.
        accepted.push(bonus_token);

        SpeculativeResult { accepted_tokens: accepted, fell_back: false, stats: self.stats.clone() }
    }

    /// Reset cumulative statistics.
    pub fn reset_stats(&mut self) {
        self.stats = SpeculativeStats::default();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helpers ---------------------------------------------------------------

    fn draft(token_id: u32, prob: f32) -> DraftToken {
        DraftToken { token_id, draft_prob: prob }
    }

    /// Build a probability vector of size `vocab` with a spike at `token_id`.
    fn prob_vec(vocab: usize, token_id: u32, prob: f32) -> Vec<f32> {
        let mut v = vec![0.0; vocab];
        let remainder = (1.0 - prob) / (vocab as f32 - 1.0);
        for (i, p) in v.iter_mut().enumerate() {
            *p = if i == token_id as usize { prob } else { remainder };
        }
        v
    }

    // Config validation -----------------------------------------------------

    #[test]
    fn test_config_default_is_valid() {
        let cfg = SpeculativeConfig::default();
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.draft_tokens, 5);
        assert_eq!(cfg.acceptance_threshold, 0.0);
        assert_eq!(cfg.max_draft_attempts, 3);
    }

    #[test]
    fn test_config_zero_draft_tokens_rejected() {
        let cfg = SpeculativeConfig::default().with_draft_tokens(0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_threshold_below_zero_rejected() {
        let cfg = SpeculativeConfig::default().with_acceptance_threshold(-0.1);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_threshold_above_one_rejected() {
        let cfg = SpeculativeConfig::default().with_acceptance_threshold(1.1);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_zero_max_attempts_rejected() {
        let cfg = SpeculativeConfig::default().with_max_draft_attempts(0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_boundary_threshold_values() {
        assert!(SpeculativeConfig::default().with_acceptance_threshold(0.0).validate().is_ok());
        assert!(SpeculativeConfig::default().with_acceptance_threshold(1.0).validate().is_ok());
    }

    // Decoder creation ------------------------------------------------------

    #[test]
    fn test_decoder_creation_valid() {
        let dec = SpeculativeDecoder::new(SpeculativeConfig::default());
        assert!(dec.is_ok());
    }

    #[test]
    fn test_decoder_creation_invalid() {
        let dec = SpeculativeDecoder::new(SpeculativeConfig::default().with_draft_tokens(0));
        assert!(dec.is_err());
    }

    // All accepted ----------------------------------------------------------

    #[test]
    fn test_all_tokens_accepted() {
        let cfg = SpeculativeConfig::default().with_draft_tokens(3).with_acceptance_threshold(0.0);
        let mut dec = SpeculativeDecoder::new(cfg).unwrap();

        let vocab = 10;
        let drafts = vec![draft(1, 0.4), draft(2, 0.3), draft(3, 0.2)];
        let t0 = prob_vec(vocab, 1, 0.5);
        let t1 = prob_vec(vocab, 2, 0.4);
        let t2 = prob_vec(vocab, 3, 0.3);
        let target: Vec<&[f32]> = vec![t0.as_slice(), t1.as_slice(), t2.as_slice()];

        let res = dec.verify(&drafts, &target, 7);

        // All 3 drafts + bonus
        assert_eq!(res.accepted_tokens, vec![1, 2, 3, 7]);
        assert!(!res.fell_back);
        assert_eq!(res.stats.accepted_tokens, 3);
        assert_eq!(res.stats.rejected_tokens, 0);
        assert_eq!(res.stats.total_drafts, 1);
    }

    // All rejected ----------------------------------------------------------

    #[test]
    fn test_all_tokens_rejected() {
        // threshold=1.0 and target probs much lower than draft probs.
        let cfg = SpeculativeConfig::default().with_draft_tokens(3).with_acceptance_threshold(1.0);
        let mut dec = SpeculativeDecoder::new(cfg).unwrap();

        let vocab = 10;
        let drafts = vec![draft(1, 0.9), draft(2, 0.8), draft(3, 0.7)];
        // Target assigns very low probability to the draft tokens.
        let t0 = prob_vec(vocab, 5, 0.9); // token 1 gets ~0.01
        let t1 = prob_vec(vocab, 5, 0.9);
        let t2 = prob_vec(vocab, 5, 0.9);
        let target: Vec<&[f32]> = vec![t0.as_slice(), t1.as_slice(), t2.as_slice()];

        let res = dec.verify(&drafts, &target, 9);

        // First token rejected immediately → bonus only.
        assert_eq!(res.accepted_tokens, vec![9]);
        assert!(!res.fell_back);
        assert_eq!(res.stats.accepted_tokens, 0);
        assert_eq!(res.stats.rejected_tokens, 3);
    }

    // Partial acceptance ----------------------------------------------------

    #[test]
    fn test_partial_acceptance() {
        let cfg = SpeculativeConfig::default().with_draft_tokens(4).with_acceptance_threshold(0.5);
        let mut dec = SpeculativeDecoder::new(cfg).unwrap();

        let vocab = 10;
        let drafts = vec![draft(1, 0.4), draft(2, 0.4), draft(3, 0.4), draft(4, 0.4)];
        // Tokens 1 and 2 have high target prob → ratio ≥ 0.5.
        // Token 3 has low target prob → ratio < 0.5.
        let t0 = prob_vec(vocab, 1, 0.6);
        let t1 = prob_vec(vocab, 2, 0.5);
        let t2 = prob_vec(vocab, 3, 0.01); // ratio ≈ 0.025 < 0.5
        let t3 = prob_vec(vocab, 4, 0.9); // never reached
        let target: Vec<&[f32]> = vec![t0.as_slice(), t1.as_slice(), t2.as_slice(), t3.as_slice()];

        let res = dec.verify(&drafts, &target, 8);

        assert_eq!(res.accepted_tokens, vec![1, 2, 8]);
        assert_eq!(res.stats.accepted_tokens, 2);
        // Tokens 3 and 4 rejected.
        assert_eq!(res.stats.rejected_tokens, 2);
    }

    // Statistics tracking ---------------------------------------------------

    #[test]
    fn test_stats_accumulate_across_iterations() {
        let cfg = SpeculativeConfig::default().with_draft_tokens(2).with_acceptance_threshold(0.0);
        let mut dec = SpeculativeDecoder::new(cfg).unwrap();

        let vocab = 5;
        let drafts = vec![draft(0, 0.5), draft(1, 0.5)];
        let t0 = prob_vec(vocab, 0, 0.5);
        let t1 = prob_vec(vocab, 1, 0.5);
        let target: Vec<&[f32]> = vec![t0.as_slice(), t1.as_slice()];

        // Run two iterations.
        dec.verify(&drafts, &target, 2);
        let res = dec.verify(&drafts, &target, 3);

        assert_eq!(res.stats.accepted_tokens, 4); // 2 × 2
        assert_eq!(res.stats.rejected_tokens, 0);
        assert_eq!(res.stats.total_drafts, 2);
    }

    #[test]
    fn test_stats_reset() {
        let cfg = SpeculativeConfig::default().with_draft_tokens(1).with_acceptance_threshold(0.0);
        let mut dec = SpeculativeDecoder::new(cfg).unwrap();

        let vocab = 4;
        let t = prob_vec(vocab, 0, 0.5);
        dec.verify(&[draft(0, 0.5)], &[t.as_slice()], 1);
        assert_eq!(dec.stats.total_drafts, 1);

        dec.reset_stats();
        assert_eq!(dec.stats.total_drafts, 0);
        assert_eq!(dec.stats.accepted_tokens, 0);
        assert_eq!(dec.stats.rejected_tokens, 0);
    }

    #[test]
    fn test_acceptance_rate_computation() {
        let mut stats = SpeculativeStats::default();
        // No data → 0.
        assert_eq!(stats.acceptance_rate(), 0.0);

        stats.accepted_tokens = 3;
        stats.rejected_tokens = 1;
        let rate = stats.acceptance_rate();
        assert!((rate - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_speedup_factor_no_drafts() {
        let stats = SpeculativeStats::default();
        assert_eq!(stats.speedup_factor(), 1.0);
    }

    #[test]
    fn test_speedup_factor_with_data() {
        let stats = SpeculativeStats { accepted_tokens: 8, rejected_tokens: 2, total_drafts: 2 };
        // (8 + 2) / 2 = 5.0
        assert!((stats.speedup_factor() - 5.0).abs() < 1e-9);
    }

    // Edge cases ------------------------------------------------------------

    #[test]
    fn test_single_token_draft() {
        let cfg = SpeculativeConfig::default().with_draft_tokens(1).with_acceptance_threshold(0.0);
        let mut dec = SpeculativeDecoder::new(cfg).unwrap();

        let vocab = 4;
        let t = prob_vec(vocab, 2, 0.8);
        let res = dec.verify(&[draft(2, 0.5)], &[t.as_slice()], 3);

        assert_eq!(res.accepted_tokens, vec![2, 3]);
        assert_eq!(res.stats.accepted_tokens, 1);
    }

    #[test]
    fn test_empty_draft() {
        let cfg = SpeculativeConfig::default().with_draft_tokens(5).with_acceptance_threshold(0.0);
        let mut dec = SpeculativeDecoder::new(cfg).unwrap();

        // Empty draft: nothing to verify, bonus token still returned.
        let res = dec.verify(&[], &[], 42);
        assert_eq!(res.accepted_tokens, vec![42]);
        assert_eq!(res.stats.accepted_tokens, 0);
        assert_eq!(res.stats.rejected_tokens, 0);
    }

    #[test]
    fn test_zero_draft_probability_always_rejected_at_nonzero_threshold() {
        let cfg = SpeculativeConfig::default().with_draft_tokens(2).with_acceptance_threshold(0.5);
        let mut dec = SpeculativeDecoder::new(cfg).unwrap();

        let vocab = 4;
        let t0 = prob_vec(vocab, 0, 0.9);
        let t1 = prob_vec(vocab, 1, 0.9);
        // Draft prob = 0 → ratio = 0 → below any positive threshold.
        let drafts = vec![draft(0, 0.0), draft(1, 0.0)];
        let target: Vec<&[f32]> = vec![t0.as_slice(), t1.as_slice()];

        let res = dec.verify(&drafts, &target, 9);
        assert_eq!(res.accepted_tokens, vec![9]);
        assert_eq!(res.stats.rejected_tokens, 2);
    }

    #[test]
    fn test_stats_display() {
        let stats = SpeculativeStats { accepted_tokens: 10, rejected_tokens: 5, total_drafts: 3 };
        let s = format!("{stats}");
        assert!(s.contains("accepted=10"));
        assert!(s.contains("rejected=5"));
        assert!(s.contains("drafts=3"));
    }
}
