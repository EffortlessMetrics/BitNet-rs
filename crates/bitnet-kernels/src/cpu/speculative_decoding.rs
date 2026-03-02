//! CPU speculative decoding kernel.
//!
//! Implements speculative decoding for autoregressive inference where a
//! fast *draft* model proposes candidate tokens and a larger *target*
//! model verifies them in a single forward pass.  Correct tokens are
//! accepted, and the first rejected token is resampled from an adjusted
//! distribution, giving exact target-model quality with higher
//! throughput.
//!
//! The module also supports **adaptive draft length**: the number of
//! draft tokens proposed per step is increased when acceptance rates
//! are high and decreased when they are low.

use std::fmt;

// ── Error ──────────────────────────────────────────────────────────

/// Errors specific to the speculative decoding kernel.
#[derive(Debug, Clone, PartialEq)]
pub enum SpeculativeDecodingError {
    /// Draft and target logit vectors have different vocabulary sizes.
    VocabMismatch { draft: usize, target: usize },
    /// An empty logit vector was supplied.
    EmptyLogits,
    /// A probability distribution does not sum to ≈1.
    InvalidDistribution { sum: f32 },
    /// The requested draft length is zero or exceeds the configured
    /// maximum.
    InvalidDraftLength { requested: usize, max: usize },
    /// Temperature must be strictly positive.
    InvalidTemperature { value: f32 },
}

impl fmt::Display for SpeculativeDecodingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VocabMismatch { draft, target } => {
                write!(f, "vocab size mismatch: draft={draft}, target={target}")
            }
            Self::EmptyLogits => write!(f, "logit vector must not be empty"),
            Self::InvalidDistribution { sum } => {
                write!(f, "invalid probability distribution (sum={sum})")
            }
            Self::InvalidDraftLength { requested, max } => {
                write!(f, "invalid draft length {requested} (max {max})")
            }
            Self::InvalidTemperature { value } => {
                write!(f, "temperature must be > 0, got {value}")
            }
        }
    }
}

impl std::error::Error for SpeculativeDecodingError {}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, SpeculativeDecodingError>;

// ── Configuration ──────────────────────────────────────────────────

/// Tunable parameters for the speculative decoding loop.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativeDecodingConfig {
    /// Number of tokens the draft model proposes per step.
    pub draft_length: usize,
    /// Minimum acceptance probability below which a draft is rejected.
    pub verification_threshold: f32,
    /// Hard upper bound on `draft_length` (for adaptive mode).
    pub max_draft_tokens: usize,
    /// Softmax temperature applied to logits before sampling.
    pub temperature: f32,
}

impl Default for SpeculativeDecodingConfig {
    fn default() -> Self {
        Self {
            draft_length: 5,
            verification_threshold: 0.0,
            max_draft_tokens: 16,
            temperature: 1.0,
        }
    }
}

impl SpeculativeDecodingConfig {
    /// Validate the configuration, returning an error if any field is
    /// out of range.
    pub fn validate(&self) -> Result<()> {
        if self.temperature <= 0.0 {
            return Err(SpeculativeDecodingError::InvalidTemperature { value: self.temperature });
        }
        if self.draft_length == 0 || self.draft_length > self.max_draft_tokens {
            return Err(SpeculativeDecodingError::InvalidDraftLength {
                requested: self.draft_length,
                max: self.max_draft_tokens,
            });
        }
        Ok(())
    }
}

// ── Core types ─────────────────────────────────────────────────────

/// A single token proposed by the draft model.
#[derive(Debug, Clone, PartialEq)]
pub struct DraftToken {
    /// Token id in the shared vocabulary.
    pub token_id: u32,
    /// Probability assigned by the draft model.
    pub probability: f32,
    /// Running product of draft probabilities up to and including this
    /// token.
    pub cumulative_prob: f32,
}

/// Outcome of verifying a batch of draft tokens against the target
/// model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationResult {
    /// All draft tokens were accepted.
    Accept,
    /// The first rejection occurred at `position` (0-indexed into the
    /// draft batch).
    Reject { position: usize },
}

/// Mutable state tracked across speculative decoding steps.
#[derive(Debug, Clone)]
pub struct SpeculativeDecodingState {
    /// Draft tokens from the most recent step.
    pub draft_tokens: Vec<DraftToken>,
    /// Tokens accepted by the target model so far.
    pub verified_tokens: Vec<u32>,
    /// Lifetime rejection count.
    pub rejection_count: u64,
    /// Exponential moving average of the acceptance rate.
    pub acceptance_rate: f32,
}

impl SpeculativeDecodingState {
    /// Create a fresh state with no history.
    pub fn new() -> Self {
        Self {
            draft_tokens: Vec::new(),
            verified_tokens: Vec::new(),
            rejection_count: 0,
            acceptance_rate: 1.0,
        }
    }

    /// Update the acceptance rate EMA after a verification round.
    fn update_acceptance_rate(&mut self, accepted: usize, total: usize) {
        if total == 0 {
            return;
        }
        let alpha = 0.3_f32;
        let batch_rate = accepted as f32 / total as f32;
        self.acceptance_rate = alpha * batch_rate + (1.0 - alpha) * self.acceptance_rate;
    }
}

impl Default for SpeculativeDecodingState {
    fn default() -> Self {
        Self::new()
    }
}

// ── Metrics ────────────────────────────────────────────────────────

/// Aggregate metrics for a speculative decoding session.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativeDecodingMetrics {
    /// Total tokens that entered the output sequence.
    pub tokens_generated: u64,
    /// Tokens accepted from draft batches (excludes resampled tokens).
    pub tokens_accepted: u64,
    /// Ratio of wall-clock tokens produced per target-model forward
    /// pass.
    pub speedup_ratio: f32,
    /// Running mean of draft-batch sizes actually used.
    pub avg_draft_length: f32,
}

impl SpeculativeDecodingMetrics {
    /// Fresh zero-valued metrics.
    pub fn new() -> Self {
        Self { tokens_generated: 0, tokens_accepted: 0, speedup_ratio: 0.0, avg_draft_length: 0.0 }
    }

    /// Record one speculative step.
    pub fn record_step(&mut self, accepted: u64, draft_len: usize, target_passes: u64) {
        self.tokens_accepted += accepted;
        // +1 because a rejection still produces one resampled token.
        self.tokens_generated += accepted + 1;
        if target_passes > 0 {
            self.speedup_ratio = self.tokens_generated as f32
                / (self.tokens_generated - self.tokens_accepted + target_passes) as f32;
        }
        let alpha = 0.2_f32;
        self.avg_draft_length = alpha * draft_len as f32 + (1.0 - alpha) * self.avg_draft_length;
    }
}

impl Default for SpeculativeDecodingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helper: softmax ────────────────────────────────────────────────

/// Numerically-stable softmax with temperature scaling.
fn softmax_with_temperature(logits: &[f32], temperature: f32) -> Result<Vec<f32>> {
    if logits.is_empty() {
        return Err(SpeculativeDecodingError::EmptyLogits);
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| ((x - max) / temperature).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 || !sum.is_finite() {
        return Err(SpeculativeDecodingError::InvalidDistribution { sum });
    }
    Ok(exps.into_iter().map(|e| e / sum).collect())
}

// ── Public API ─────────────────────────────────────────────────────

/// Generate draft tokens by sampling from `draft_logits` sequences.
///
/// Each entry in `draft_logits` is the logit vector produced by the
/// draft model for one successive position.  Returns at most
/// `config.draft_length` [`DraftToken`]s.
pub fn generate_draft_tokens(
    draft_logits: &[Vec<f32>],
    config: &SpeculativeDecodingConfig,
    rng_seed: u64,
) -> Result<Vec<DraftToken>> {
    config.validate()?;
    let n = draft_logits.len().min(config.draft_length);
    let mut tokens = Vec::with_capacity(n);
    let mut cumulative = 1.0_f32;
    let mut rng_state = rng_seed;

    for logits in draft_logits.iter().take(n) {
        let probs = softmax_with_temperature(logits, config.temperature)?;
        let (token_id, prob) = sample_from_distribution(&probs, &mut rng_state);
        cumulative *= prob;
        tokens.push(DraftToken {
            token_id: token_id as u32,
            probability: prob,
            cumulative_prob: cumulative,
        });
    }
    Ok(tokens)
}

/// Verify a batch of draft tokens against the target model's logits.
///
/// For each draft position we compute the acceptance probability
/// `min(1, p_target / p_draft)` and accept with that probability
/// (using deterministic randomness derived from `rng_seed`).
///
/// `target_logits` must contain one logit vector per draft token.
pub fn verify_draft_tokens(
    draft_tokens: &[DraftToken],
    target_logits: &[Vec<f32>],
    config: &SpeculativeDecodingConfig,
    state: &mut SpeculativeDecodingState,
    rng_seed: u64,
) -> Result<VerificationResult> {
    if draft_tokens.is_empty() {
        return Ok(VerificationResult::Accept);
    }
    if draft_tokens.len() != target_logits.len() {
        return Err(SpeculativeDecodingError::VocabMismatch {
            draft: draft_tokens.len(),
            target: target_logits.len(),
        });
    }

    let mut rng_state = rng_seed;
    let mut accepted = 0_usize;

    for (i, draft) in draft_tokens.iter().enumerate() {
        let target_probs = softmax_with_temperature(&target_logits[i], config.temperature)?;

        if (draft.token_id as usize) >= target_probs.len() {
            state.rejection_count += 1;
            state.update_acceptance_rate(accepted, draft_tokens.len());
            return Ok(VerificationResult::Reject { position: i });
        }

        let p_target = target_probs[draft.token_id as usize];
        let accept_prob = compute_acceptance_probability(p_target, draft.probability);

        let r = xorshift_uniform(&mut rng_state);
        if r >= accept_prob {
            state.rejection_count += 1;
            state.update_acceptance_rate(accepted, draft_tokens.len());
            return Ok(VerificationResult::Reject { position: i });
        }
        state.verified_tokens.push(draft.token_id);
        accepted += 1;
    }
    state.update_acceptance_rate(accepted, draft_tokens.len());
    Ok(VerificationResult::Accept)
}

/// Compute `min(1, p_target / p_draft)`.
///
/// Handles the edge-case where `p_draft` is zero (returns 0.0 when
/// `p_target` is also zero, 1.0 otherwise).
pub fn compute_acceptance_probability(p_target: f32, p_draft: f32) -> f32 {
    if p_draft <= 0.0 {
        if p_target <= 0.0 {
            return 0.0;
        }
        return 1.0;
    }
    (p_target / p_draft).min(1.0)
}

/// Resample one token from the *adjusted* distribution when a draft
/// token is rejected.
///
/// The adjusted distribution is
/// `max(0, p_target - p_draft)` normalised,
/// which guarantees the overall output distribution matches the target
/// model exactly.
pub fn speculative_sample(
    target_logits: &[f32],
    draft_logits: &[f32],
    temperature: f32,
    rng_seed: u64,
) -> Result<u32> {
    if target_logits.len() != draft_logits.len() {
        return Err(SpeculativeDecodingError::VocabMismatch {
            draft: draft_logits.len(),
            target: target_logits.len(),
        });
    }
    let p_target = softmax_with_temperature(target_logits, temperature)?;
    let p_draft = softmax_with_temperature(draft_logits, temperature)?;

    let adjusted: Vec<f32> =
        p_target.iter().zip(p_draft.iter()).map(|(&pt, &pd)| (pt - pd).max(0.0)).collect();

    let sum: f32 = adjusted.iter().sum();
    if sum <= 0.0 {
        // Distributions identical – fall back to target sampling.
        let mut rng = rng_seed;
        let (id, _) = sample_from_distribution(&p_target, &mut rng);
        return Ok(id as u32);
    }
    let normed: Vec<f32> = adjusted.iter().map(|&v| v / sum).collect();
    let mut rng = rng_seed;
    let (id, _) = sample_from_distribution(&normed, &mut rng);
    Ok(id as u32)
}

/// Adaptively adjust `config.draft_length` based on the recent
/// acceptance rate stored in `state`.
///
/// * High acceptance → increase draft length (up to
///   `config.max_draft_tokens`).
/// * Low acceptance  → decrease draft length (minimum 1).
pub fn adjust_draft_length(
    config: &mut SpeculativeDecodingConfig,
    state: &SpeculativeDecodingState,
) {
    let rate = state.acceptance_rate;
    if rate > 0.8 {
        config.draft_length = (config.draft_length + 1).min(config.max_draft_tokens);
    } else if rate < 0.4 {
        config.draft_length = config.draft_length.saturating_sub(1).max(1);
    }
    // Between 0.4 and 0.8 keep unchanged.
}

// ── Internal helpers ───────────────────────────────────────────────

/// Lightweight xorshift64 PRNG returning a value in `[0, 1)`.
fn xorshift_uniform(state: &mut u64) -> f32 {
    if *state == 0 {
        *state = 0xDEAD_BEEF;
    }
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f32) / (u64::MAX as f32)
}

/// Sample a token index from a probability distribution.  Returns
/// `(index, probability)`.
fn sample_from_distribution(probs: &[f32], rng: &mut u64) -> (usize, f32) {
    let r = xorshift_uniform(rng);
    let mut cumulative = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return (i, p);
        }
    }
    // Numerical rounding – return the last token.
    let last = probs.len() - 1;
    (last, probs[last])
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Config validation ──────────────────────────────────────────

    #[test]
    fn test_default_config_is_valid() {
        let cfg = SpeculativeDecodingConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_zero_draft_length_rejected() {
        let cfg = SpeculativeDecodingConfig { draft_length: 0, ..Default::default() };
        assert!(matches!(cfg.validate(), Err(SpeculativeDecodingError::InvalidDraftLength { .. })));
    }

    #[test]
    fn test_config_draft_exceeds_max_rejected() {
        let cfg = SpeculativeDecodingConfig {
            draft_length: 20,
            max_draft_tokens: 10,
            ..Default::default()
        };
        assert!(matches!(cfg.validate(), Err(SpeculativeDecodingError::InvalidDraftLength { .. })));
    }

    #[test]
    fn test_config_zero_temperature_rejected() {
        let cfg = SpeculativeDecodingConfig { temperature: 0.0, ..Default::default() };
        assert!(matches!(cfg.validate(), Err(SpeculativeDecodingError::InvalidTemperature { .. })));
    }

    #[test]
    fn test_config_negative_temperature_rejected() {
        let cfg = SpeculativeDecodingConfig { temperature: -1.0, ..Default::default() };
        assert!(matches!(cfg.validate(), Err(SpeculativeDecodingError::InvalidTemperature { .. })));
    }

    #[test]
    fn test_config_max_equals_draft_is_valid() {
        let cfg = SpeculativeDecodingConfig {
            draft_length: 8,
            max_draft_tokens: 8,
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    // ── Softmax helper ─────────────────────────────────────────────

    #[test]
    fn test_softmax_uniform() {
        let logits = vec![0.0; 4];
        let probs = softmax_with_temperature(&logits, 1.0).unwrap();
        for &p in &probs {
            assert!((p - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_empty_rejected() {
        assert!(matches!(
            softmax_with_temperature(&[], 1.0),
            Err(SpeculativeDecodingError::EmptyLogits)
        ));
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let probs = softmax_with_temperature(&logits, 1.0).unwrap();
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_high_temperature_flattens() {
        let logits = vec![0.0, 10.0];
        let probs = softmax_with_temperature(&logits, 100.0).unwrap();
        assert!((probs[0] - probs[1]).abs() < 0.05);
    }

    #[test]
    fn test_softmax_low_temperature_sharpens() {
        let logits = vec![0.0, 10.0];
        let probs = softmax_with_temperature(&logits, 0.01).unwrap();
        assert!(probs[1] > 0.99);
    }

    // ── Draft token generation ─────────────────────────────────────

    #[test]
    fn test_generate_draft_tokens_basic() {
        let logits = vec![vec![-100.0, -100.0, 100.0]; 3];
        let cfg = SpeculativeDecodingConfig { draft_length: 3, ..Default::default() };
        let tokens = generate_draft_tokens(&logits, &cfg, 42).unwrap();
        assert_eq!(tokens.len(), 3);
        for t in &tokens {
            assert_eq!(t.token_id, 2);
        }
    }

    #[test]
    fn test_generate_draft_tokens_clamps_to_available() {
        let logits = vec![vec![1.0, 2.0]; 2];
        let cfg = SpeculativeDecodingConfig { draft_length: 5, ..Default::default() };
        let tokens = generate_draft_tokens(&logits, &cfg, 99).unwrap();
        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn test_generate_cumulative_prob_decreases() {
        let logits = vec![vec![1.0, 1.0]; 4];
        let cfg = SpeculativeDecodingConfig { draft_length: 4, ..Default::default() };
        let tokens = generate_draft_tokens(&logits, &cfg, 7).unwrap();
        for w in tokens.windows(2) {
            assert!(w[1].cumulative_prob <= w[0].cumulative_prob + 1e-7);
        }
    }

    #[test]
    fn test_generate_empty_logits_returns_empty() {
        let cfg = SpeculativeDecodingConfig::default();
        let tokens = generate_draft_tokens(&[], &cfg, 1).unwrap();
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_generate_single_logit_vector() {
        let logits = vec![vec![5.0]];
        let cfg = SpeculativeDecodingConfig {
            draft_length: 1,
            max_draft_tokens: 1,
            ..Default::default()
        };
        let tokens = generate_draft_tokens(&logits, &cfg, 1).unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].token_id, 0);
        assert!((tokens[0].probability - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_generate_deterministic_same_seed() {
        let logits = vec![vec![1.0, 2.0, 3.0]; 3];
        let cfg = SpeculativeDecodingConfig { draft_length: 3, ..Default::default() };
        let a = generate_draft_tokens(&logits, &cfg, 1234).unwrap();
        let b = generate_draft_tokens(&logits, &cfg, 1234).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_generate_different_seeds_may_differ() {
        let logits = vec![vec![0.0; 1000]; 5];
        let cfg = SpeculativeDecodingConfig { draft_length: 5, ..Default::default() };
        let a = generate_draft_tokens(&logits, &cfg, 1).unwrap();
        let b = generate_draft_tokens(&logits, &cfg, 999_999).unwrap();
        let same = a.iter().zip(b.iter()).all(|(x, y)| x.token_id == y.token_id);
        assert!(!same, "different seeds on uniform distribution should almost certainly differ");
    }

    // ── Verification ───────────────────────────────────────────────

    #[test]
    fn test_verify_all_accept_identical_logits() {
        let logits = vec![vec![-100.0, -100.0, 100.0]; 3];
        let cfg = SpeculativeDecodingConfig { draft_length: 3, ..Default::default() };
        let drafts = generate_draft_tokens(&logits, &cfg, 42).unwrap();
        let mut state = SpeculativeDecodingState::new();
        let result = verify_draft_tokens(&drafts, &logits, &cfg, &mut state, 42).unwrap();
        assert_eq!(result, VerificationResult::Accept);
        assert_eq!(state.verified_tokens.len(), 3);
    }

    #[test]
    fn test_verify_reject_mismatched_logits() {
        let draft_logits = vec![vec![10.0, 0.0]];
        let cfg = SpeculativeDecodingConfig {
            draft_length: 1,
            max_draft_tokens: 1,
            ..Default::default()
        };
        let drafts = generate_draft_tokens(&draft_logits, &cfg, 42).unwrap();
        assert_eq!(drafts[0].token_id, 0);

        let target_logits = vec![vec![0.0, 100.0]];
        let mut state = SpeculativeDecodingState::new();
        let result = verify_draft_tokens(&drafts, &target_logits, &cfg, &mut state, 42).unwrap();
        assert_eq!(result, VerificationResult::Reject { position: 0 });
    }

    #[test]
    fn test_verify_empty_drafts_accept() {
        let cfg = SpeculativeDecodingConfig::default();
        let mut state = SpeculativeDecodingState::new();
        let result = verify_draft_tokens(&[], &[], &cfg, &mut state, 1).unwrap();
        assert_eq!(result, VerificationResult::Accept);
    }

    #[test]
    fn test_verify_length_mismatch_error() {
        let drafts = vec![DraftToken { token_id: 0, probability: 1.0, cumulative_prob: 1.0 }];
        let target_logits: Vec<Vec<f32>> = vec![];
        let cfg = SpeculativeDecodingConfig::default();
        let mut state = SpeculativeDecodingState::new();
        let err = verify_draft_tokens(&drafts, &target_logits, &cfg, &mut state, 1);
        assert!(err.is_err());
    }

    #[test]
    fn test_verify_updates_rejection_count() {
        let draft = vec![DraftToken { token_id: 0, probability: 0.99, cumulative_prob: 0.99 }];
        let target = vec![vec![-100.0, 100.0]];
        let cfg = SpeculativeDecodingConfig {
            draft_length: 1,
            max_draft_tokens: 1,
            ..Default::default()
        };
        let mut state = SpeculativeDecodingState::new();
        let _ = verify_draft_tokens(&draft, &target, &cfg, &mut state, 42);
        assert!(state.rejection_count >= 1);
    }

    #[test]
    fn test_verify_multiple_rounds_accumulate() {
        let logits = vec![vec![-100.0, -100.0, 100.0]; 2];
        let cfg = SpeculativeDecodingConfig { draft_length: 2, ..Default::default() };
        let mut state = SpeculativeDecodingState::new();
        for _ in 0..3 {
            let drafts = generate_draft_tokens(&logits, &cfg, 42).unwrap();
            let _ = verify_draft_tokens(&drafts, &logits, &cfg, &mut state, 42);
        }
        assert_eq!(state.verified_tokens.len(), 6);
    }

    // ── Acceptance probability ──────────────────────────────────────

    #[test]
    fn test_acceptance_prob_equal() {
        assert!((compute_acceptance_probability(0.5, 0.5) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_acceptance_prob_target_higher() {
        assert!((compute_acceptance_probability(0.8, 0.4) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_acceptance_prob_target_lower() {
        let p = compute_acceptance_probability(0.2, 0.4);
        assert!((p - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_acceptance_prob_zero_draft_zero_target() {
        assert_eq!(compute_acceptance_probability(0.0, 0.0), 0.0);
    }

    #[test]
    fn test_acceptance_prob_zero_draft_nonzero_target() {
        assert_eq!(compute_acceptance_probability(0.5, 0.0), 1.0);
    }

    #[test]
    fn test_acceptance_prob_never_exceeds_one() {
        for n in 1..=100 {
            let pt = n as f32 / 100.0;
            let pd = (101 - n) as f32 / 100.0;
            assert!(compute_acceptance_probability(pt, pd) <= 1.0);
        }
    }

    #[test]
    fn test_acceptance_prob_never_negative() {
        assert!(compute_acceptance_probability(0.0, 1.0) >= 0.0);
        assert!(compute_acceptance_probability(0.01, 1.0) >= 0.0);
    }

    // ── Speculative sample (rejection resampling) ──────────────────

    #[test]
    fn test_speculative_sample_basic() {
        let target = vec![0.0, 10.0, 0.0];
        let draft = vec![10.0, 0.0, 0.0];
        let id = speculative_sample(&target, &draft, 1.0, 42).unwrap();
        assert_eq!(id, 1);
    }

    #[test]
    fn test_speculative_sample_identical_distributions() {
        let logits = vec![1.0, 2.0, 3.0];
        let id = speculative_sample(&logits, &logits, 1.0, 42).unwrap();
        assert!((id as usize) < logits.len());
    }

    #[test]
    fn test_speculative_sample_vocab_mismatch() {
        let target = vec![1.0, 2.0];
        let draft = vec![1.0];
        assert!(matches!(
            speculative_sample(&target, &draft, 1.0, 1),
            Err(SpeculativeDecodingError::VocabMismatch { .. })
        ));
    }

    #[test]
    fn test_speculative_sample_deterministic() {
        let target = vec![1.0, 2.0, 3.0, 4.0];
        let draft = vec![4.0, 3.0, 2.0, 1.0];
        let a = speculative_sample(&target, &draft, 1.0, 77).unwrap();
        let b = speculative_sample(&target, &draft, 1.0, 77).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_speculative_sample_single_vocab() {
        let target = vec![5.0];
        let draft = vec![3.0];
        let id = speculative_sample(&target, &draft, 1.0, 1).unwrap();
        assert_eq!(id, 0);
    }

    // ── Adaptive draft length ──────────────────────────────────────

    #[test]
    fn test_adjust_increases_on_high_acceptance() {
        let mut cfg = SpeculativeDecodingConfig { draft_length: 4, ..Default::default() };
        let state = SpeculativeDecodingState { acceptance_rate: 0.95, ..Default::default() };
        adjust_draft_length(&mut cfg, &state);
        assert_eq!(cfg.draft_length, 5);
    }

    #[test]
    fn test_adjust_decreases_on_low_acceptance() {
        let mut cfg = SpeculativeDecodingConfig { draft_length: 4, ..Default::default() };
        let state = SpeculativeDecodingState { acceptance_rate: 0.2, ..Default::default() };
        adjust_draft_length(&mut cfg, &state);
        assert_eq!(cfg.draft_length, 3);
    }

    #[test]
    fn test_adjust_unchanged_mid_acceptance() {
        let mut cfg = SpeculativeDecodingConfig { draft_length: 5, ..Default::default() };
        let state = SpeculativeDecodingState { acceptance_rate: 0.6, ..Default::default() };
        adjust_draft_length(&mut cfg, &state);
        assert_eq!(cfg.draft_length, 5);
    }

    #[test]
    fn test_adjust_clamps_at_max() {
        let mut cfg = SpeculativeDecodingConfig {
            draft_length: 16,
            max_draft_tokens: 16,
            ..Default::default()
        };
        let state = SpeculativeDecodingState { acceptance_rate: 0.99, ..Default::default() };
        adjust_draft_length(&mut cfg, &state);
        assert_eq!(cfg.draft_length, 16);
    }

    #[test]
    fn test_adjust_clamps_at_min() {
        let mut cfg = SpeculativeDecodingConfig {
            draft_length: 1,
            max_draft_tokens: 16,
            ..Default::default()
        };
        let state = SpeculativeDecodingState { acceptance_rate: 0.1, ..Default::default() };
        adjust_draft_length(&mut cfg, &state);
        assert_eq!(cfg.draft_length, 1);
    }

    #[test]
    fn test_adjust_boundary_0_8_no_increase() {
        let mut cfg = SpeculativeDecodingConfig { draft_length: 5, ..Default::default() };
        let state = SpeculativeDecodingState { acceptance_rate: 0.8, ..Default::default() };
        adjust_draft_length(&mut cfg, &state);
        assert_eq!(cfg.draft_length, 5);
    }

    #[test]
    fn test_adjust_boundary_0_4_no_decrease() {
        let mut cfg = SpeculativeDecodingConfig { draft_length: 5, ..Default::default() };
        let state = SpeculativeDecodingState { acceptance_rate: 0.4, ..Default::default() };
        adjust_draft_length(&mut cfg, &state);
        assert_eq!(cfg.draft_length, 5);
    }

    // ── Metrics ────────────────────────────────────────────────────

    #[test]
    fn test_metrics_new_zeroed() {
        let m = SpeculativeDecodingMetrics::new();
        assert_eq!(m.tokens_generated, 0);
        assert_eq!(m.tokens_accepted, 0);
        assert_eq!(m.speedup_ratio, 0.0);
        assert_eq!(m.avg_draft_length, 0.0);
    }

    #[test]
    fn test_metrics_record_step() {
        let mut m = SpeculativeDecodingMetrics::new();
        m.record_step(4, 5, 1);
        assert_eq!(m.tokens_generated, 5);
        assert_eq!(m.tokens_accepted, 4);
        assert!(m.speedup_ratio > 1.0);
    }

    #[test]
    fn test_metrics_record_multiple_steps() {
        let mut m = SpeculativeDecodingMetrics::new();
        m.record_step(3, 4, 1);
        m.record_step(5, 6, 1);
        assert_eq!(m.tokens_accepted, 8);
        assert_eq!(m.tokens_generated, 10);
    }

    #[test]
    fn test_metrics_avg_draft_length_ema() {
        let mut m = SpeculativeDecodingMetrics::new();
        m.record_step(3, 5, 1);
        let first = m.avg_draft_length;
        m.record_step(3, 10, 1);
        assert!(m.avg_draft_length > first);
    }

    // ── State ──────────────────────────────────────────────────────

    #[test]
    fn test_state_new_defaults() {
        let s = SpeculativeDecodingState::new();
        assert!(s.draft_tokens.is_empty());
        assert!(s.verified_tokens.is_empty());
        assert_eq!(s.rejection_count, 0);
        assert!((s.acceptance_rate - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_state_acceptance_rate_ema() {
        let mut s = SpeculativeDecodingState::new();
        s.update_acceptance_rate(1, 2);
        assert!((s.acceptance_rate - 0.85).abs() < 1e-5);
    }

    #[test]
    fn test_state_acceptance_rate_zero_total() {
        let mut s = SpeculativeDecodingState::new();
        s.update_acceptance_rate(0, 0);
        assert!((s.acceptance_rate - 1.0).abs() < 1e-6);
    }

    // ── Error display ──────────────────────────────────────────────

    #[test]
    fn test_error_display_vocab_mismatch() {
        let e = SpeculativeDecodingError::VocabMismatch { draft: 3, target: 5 };
        assert!(e.to_string().contains("vocab size mismatch"));
    }

    #[test]
    fn test_error_display_empty_logits() {
        let e = SpeculativeDecodingError::EmptyLogits;
        assert!(e.to_string().contains("must not be empty"));
    }

    #[test]
    fn test_error_display_invalid_distribution() {
        let e = SpeculativeDecodingError::InvalidDistribution { sum: 0.0 };
        assert!(e.to_string().contains("invalid probability"));
    }

    #[test]
    fn test_error_display_invalid_draft_length() {
        let e = SpeculativeDecodingError::InvalidDraftLength { requested: 0, max: 16 };
        assert!(e.to_string().contains("invalid draft length"));
    }

    #[test]
    fn test_error_display_invalid_temperature() {
        let e = SpeculativeDecodingError::InvalidTemperature { value: -1.0 };
        assert!(e.to_string().contains("temperature"));
    }

    #[test]
    fn test_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(SpeculativeDecodingError::EmptyLogits);
        assert!(!e.to_string().is_empty());
    }

    // ── xorshift helper ────────────────────────────────────────────

    #[test]
    fn test_xorshift_in_range() {
        let mut s = 42_u64;
        for _ in 0..1000 {
            let v = xorshift_uniform(&mut s);
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_xorshift_zero_seed_handled() {
        let mut s = 0_u64;
        let v = xorshift_uniform(&mut s);
        assert!(v.is_finite());
    }

    // ── Edge cases ─────────────────────────────────────────────────

    #[test]
    fn test_single_token_draft_accept() {
        let logits = vec![vec![10.0, 0.0]];
        let cfg = SpeculativeDecodingConfig {
            draft_length: 1,
            max_draft_tokens: 1,
            ..Default::default()
        };
        let drafts = generate_draft_tokens(&logits, &cfg, 42).unwrap();
        let mut state = SpeculativeDecodingState::new();
        let res = verify_draft_tokens(&drafts, &logits, &cfg, &mut state, 42).unwrap();
        assert_eq!(res, VerificationResult::Accept);
    }

    #[test]
    fn test_all_reject_token_out_of_vocab() {
        let draft = vec![DraftToken { token_id: 999, probability: 0.5, cumulative_prob: 0.5 }];
        let target = vec![vec![1.0, 2.0]];
        let cfg = SpeculativeDecodingConfig {
            draft_length: 1,
            max_draft_tokens: 1,
            ..Default::default()
        };
        let mut state = SpeculativeDecodingState::new();
        let res = verify_draft_tokens(&draft, &target, &cfg, &mut state, 1).unwrap();
        assert_eq!(res, VerificationResult::Reject { position: 0 });
    }

    #[test]
    fn test_large_vocab_softmax_stability() {
        let logits: Vec<f32> = (0..50_000).map(|i| (i as f32) * 0.01).collect();
        let probs = softmax_with_temperature(&logits, 1.0).unwrap();
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_draft_token_debug_display() {
        let t = DraftToken { token_id: 7, probability: 0.5, cumulative_prob: 0.25 };
        let dbg = format!("{t:?}");
        assert!(dbg.contains("token_id: 7"));
    }

    #[test]
    fn test_verification_result_eq() {
        assert_eq!(VerificationResult::Accept, VerificationResult::Accept);
        assert_eq!(
            VerificationResult::Reject { position: 3 },
            VerificationResult::Reject { position: 3 }
        );
        assert_ne!(VerificationResult::Accept, VerificationResult::Reject { position: 0 });
    }
}

// ── Property tests ─────────────────────────────────────────────────

#[cfg(test)]
mod prop_tests {
    use super::*;
    use proptest::prelude::*;

    prop_compose! {
        fn arb_logits(len: usize)(v in proptest::collection::vec(-10.0f32..10.0f32, len..=len)) -> Vec<f32> {
            v
        }
    }

    proptest! {
        #[test]
        fn softmax_sums_to_one(logits in arb_logits(8)) {
            let probs = softmax_with_temperature(&logits, 1.0).unwrap();
            let sum: f32 = probs.iter().sum();
            prop_assert!((sum - 1.0).abs() < 1e-4, "sum was {sum}");
        }

        #[test]
        fn softmax_all_non_negative(logits in arb_logits(16)) {
            let probs = softmax_with_temperature(&logits, 1.0).unwrap();
            for &p in &probs {
                prop_assert!(p >= 0.0, "negative prob {p}");
            }
        }

        #[test]
        fn acceptance_prob_in_unit_interval(
            pt in 0.0f32..1.0f32,
            pd in 0.001f32..1.0f32,
        ) {
            let a = compute_acceptance_probability(pt, pd);
            prop_assert!((0.0..=1.0).contains(&a), "acceptance={a}");
        }

        #[test]
        fn draft_count_le_config(
            n in 1_usize..=10,
            seed in 1u64..1_000_000,
        ) {
            let logits = vec![vec![0.0_f32; 4]; n];
            let cfg = SpeculativeDecodingConfig {
                draft_length: n,
                max_draft_tokens: n.max(1),
                ..Default::default()
            };
            let tokens = generate_draft_tokens(&logits, &cfg, seed).unwrap();
            prop_assert!(tokens.len() <= n);
        }

        #[test]
        fn cumulative_prob_monotonic_decreasing(
            seed in 1u64..1_000_000,
        ) {
            let logits = vec![vec![1.0, 1.0, 1.0]; 6];
            let cfg = SpeculativeDecodingConfig {
                draft_length: 6,
                ..Default::default()
            };
            let tokens = generate_draft_tokens(&logits, &cfg, seed).unwrap();
            for w in tokens.windows(2) {
                prop_assert!(w[1].cumulative_prob <= w[0].cumulative_prob + 1e-7);
            }
        }

        #[test]
        fn speculative_sample_in_vocab(
            seed in 1u64..1_000_000,
        ) {
            let target = vec![1.0, 2.0, 3.0, 4.0];
            let draft  = vec![4.0, 3.0, 2.0, 1.0];
            let id = speculative_sample(&target, &draft, 1.0, seed).unwrap();
            prop_assert!((id as usize) < target.len());
        }

        #[test]
        fn adjust_never_exceeds_bounds(
            rate in 0.0f32..1.0f32,
            dl in 1_usize..=16,
        ) {
            let mut cfg = SpeculativeDecodingConfig {
                draft_length: dl,
                max_draft_tokens: 16,
                ..Default::default()
            };
            let state = SpeculativeDecodingState {
                acceptance_rate: rate,
                ..Default::default()
            };
            adjust_draft_length(&mut cfg, &state);
            prop_assert!(cfg.draft_length >= 1);
            prop_assert!(cfg.draft_length <= 16);
        }
    }
}
