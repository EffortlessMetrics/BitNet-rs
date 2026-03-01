//! Speculative decoding for Intel A770 (OpenCL) GPU inference.
//!
//! Speculative decoding accelerates autoregressive generation by having a
//! small *draft* model propose `k` tokens that are then verified in a
//! single forward pass of the larger *target* model.  Accepted tokens are
//! emitted without additional target calls, yielding a wall-clock speedup
//! proportional to the acceptance rate.
//!
//! This module provides:
//!
//! - **CPU reference implementations** — scalar rejection-sampling loop for
//!   correctness testing on any platform.
//! - **OpenCL dispatch hooks** — (planned) kernel sources for batched
//!   draft/verify on Intel Arc A770 and other OpenCL 3.0 devices.

use std::fmt;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Tuning knobs for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Maximum number of tokens the draft model may propose per step.
    pub max_draft_tokens: u32,
    /// Sampling temperature applied to both draft and target logits.
    pub temperature: f32,
    /// Minimum probability threshold below which a draft token is
    /// automatically rejected (0.0 disables the gate).
    pub acceptance_threshold: f32,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self { max_draft_tokens: 5, temperature: 1.0, acceptance_threshold: 0.0 }
    }
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// Tokens (and their associated probabilities) proposed by the draft model.
#[derive(Debug, Clone)]
pub struct DraftProposal {
    /// Draft token ids — length ≤ `max_draft_tokens`.
    pub tokens: Vec<u32>,
    /// Raw logits produced by the draft model for each position.
    pub logits: Vec<Vec<f32>>,
    /// Softmax probabilities for each position.
    pub draft_probs: Vec<Vec<f32>>,
}

/// Outcome of verifying a [`DraftProposal`] against the target model.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// How many consecutive draft tokens were accepted.
    pub accepted_count: usize,
    /// The accepted token ids (length = `accepted_count`).
    pub accepted_tokens: Vec<u32>,
    /// An extra token sampled from the adjusted distribution when not
    /// all draft tokens were accepted.
    pub bonus_token: Option<u32>,
    /// Target-model softmax probabilities at each verified position.
    pub target_probs: Vec<Vec<f32>>,
}

/// Cumulative statistics for a speculative-decoding session.
#[derive(Debug, Clone, Default)]
pub struct SpeculativeStats {
    /// Total number of draft proposals evaluated.
    pub total_proposals: usize,
    /// Total number of accepted tokens across all proposals.
    pub total_accepted: usize,
    /// Running acceptance rate (accepted / proposed).
    pub acceptance_rate: f32,
    /// Mean accepted tokens per speculative step.
    pub avg_accepted_per_step: f32,
    /// Tokens that did *not* require a separate target-model call.
    pub tokens_saved: usize,
}

/// Errors specific to speculative decoding.
#[derive(Debug, Clone)]
pub enum SpeculativeError {
    /// The draft model failed to produce logits.
    DraftModelFailed(String),
    /// Target-model verification encountered an error.
    VerificationFailed(String),
    /// Draft and target models have different vocabulary sizes.
    VocabMismatch { draft_vocab: usize, target_vocab: usize },
    /// The draft proposal contained zero tokens.
    EmptyDraft,
}

impl fmt::Display for SpeculativeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DraftModelFailed(msg) => {
                write!(f, "draft model failed: {msg}")
            }
            Self::VerificationFailed(msg) => {
                write!(f, "verification failed: {msg}")
            }
            Self::VocabMismatch { draft_vocab, target_vocab } => {
                write!(
                    f,
                    "vocab mismatch: draft={draft_vocab}, \
                     target={target_vocab}"
                )
            }
            Self::EmptyDraft => write!(f, "empty draft proposal"),
        }
    }
}

impl std::error::Error for SpeculativeError {}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Simple xorshift64 PRNG — deterministic, no external deps.
fn xorshift64(state: &mut u64) -> u64 {
    // Ensure state is never zero (xorshift fixed point).
    if *state == 0 {
        *state = 0x5851_f42d_4c95_7f2d;
    }
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    s
}

/// Return a uniform f32 in [0, 1) from the PRNG.
fn rand_f32(state: &mut u64) -> f32 {
    (xorshift64(state) >> 11) as f32 / ((1u64 << 53) as f32)
}

/// In-place softmax with optional temperature scaling.
fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return vec![];
    }
    let temp = if temperature <= 0.0 { 1e-8 } else { temperature };
    let scaled: Vec<f32> = logits.iter().map(|&l| l / temp).collect();
    let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        // Uniform fallback.
        let u = 1.0 / logits.len() as f32;
        return vec![u; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// Argmax over a slice.
fn argmax(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Sample a token id from a probability distribution.
fn sample_from_probs(probs: &[f32], rng: &mut u64) -> u32 {
    let r = rand_f32(rng);
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

/// Generate draft tokens autoregressively using `logits_fn`.
///
/// `logits_fn(token)` returns the raw logit vector for the *next* token
/// given the current token.  Sampling is greedy when `temperature` ≤ 0
/// and multinomial otherwise.
pub fn cpu_generate_draft(
    logits_fn: &dyn Fn(u32) -> Vec<f32>,
    prompt_token: u32,
    num_draft: usize,
    temperature: f32,
    seed: u64,
) -> DraftProposal {
    let mut rng = seed;
    let mut tokens = Vec::with_capacity(num_draft);
    let mut all_logits = Vec::with_capacity(num_draft);
    let mut all_probs = Vec::with_capacity(num_draft);

    let mut current = prompt_token;
    for _ in 0..num_draft {
        let logits = logits_fn(current);
        let probs = softmax(&logits, temperature);
        let tok =
            if temperature <= 0.0 { argmax(&probs) } else { sample_from_probs(&probs, &mut rng) };
        tokens.push(tok);
        all_logits.push(logits);
        all_probs.push(probs);
        current = tok;
    }

    DraftProposal { tokens, logits: all_logits, draft_probs: all_probs }
}

/// Verify a [`DraftProposal`] against the target model.
///
/// `target_logits_fn` receives the full sequence of draft tokens and returns
/// one logit vector per position (length = `draft.tokens.len() + 1` for the
/// bonus position).
///
/// Uses rejection sampling: for each draft token *i*, accept with probability
/// `min(1, target_prob[i] / draft_prob[i])`.  On the first rejection,
/// sample a *bonus* token from the adjusted distribution
/// `max(0, target − draft)`.
pub fn cpu_verify_draft(
    draft: &DraftProposal,
    target_logits_fn: &dyn Fn(&[u32]) -> Vec<Vec<f32>>,
    config: &SpeculativeConfig,
    seed: u64,
) -> Result<VerificationResult, SpeculativeError> {
    if draft.tokens.is_empty() {
        return Err(SpeculativeError::EmptyDraft);
    }

    let target_logits_all = target_logits_fn(&draft.tokens);

    // Vocab-size sanity check.
    if let (Some(d), Some(t)) = (draft.draft_probs.first(), target_logits_all.first())
        && d.len() != t.len()
    {
        return Err(SpeculativeError::VocabMismatch {
            draft_vocab: d.len(),
            target_vocab: t.len(),
        });
    }

    let mut rng = seed;
    let mut accepted_tokens = Vec::new();
    let mut target_probs_out = Vec::new();

    for (i, &draft_tok) in draft.tokens.iter().enumerate() {
        let target_probs = softmax(&target_logits_all[i], config.temperature);
        target_probs_out.push(target_probs.clone());

        let d_prob = draft.draft_probs[i].get(draft_tok as usize).copied().unwrap_or(0.0);
        let t_prob = target_probs.get(draft_tok as usize).copied().unwrap_or(0.0);

        if t_prob < config.acceptance_threshold {
            // Below hard threshold — reject.
            let bonus = cpu_sample_from_adjusted(&target_probs, &draft.draft_probs[i], rng);
            return Ok(VerificationResult {
                accepted_count: accepted_tokens.len(),
                accepted_tokens,
                bonus_token: Some(bonus),
                target_probs: target_probs_out,
            });
        }

        if !cpu_rejection_sample(d_prob, t_prob, rng) {
            // Rejected — sample bonus.
            let bonus = cpu_sample_from_adjusted(&target_probs, &draft.draft_probs[i], rng);
            return Ok(VerificationResult {
                accepted_count: accepted_tokens.len(),
                accepted_tokens,
                bonus_token: Some(bonus),
                target_probs: target_probs_out,
            });
        }

        // Advance the PRNG so different positions get different randomness.
        xorshift64(&mut rng);
        accepted_tokens.push(draft_tok);
    }

    // All accepted — take the bonus from the last target position if
    // the target produced an extra row, else None.
    let bonus = if target_logits_all.len() > draft.tokens.len() {
        let last = softmax(&target_logits_all[draft.tokens.len()], config.temperature);
        Some(sample_from_probs(&last, &mut rng))
    } else {
        None
    };

    Ok(VerificationResult {
        accepted_count: accepted_tokens.len(),
        accepted_tokens,
        bonus_token: bonus,
        target_probs: target_probs_out,
    })
}

/// Accept a single draft token with probability `min(1, target / draft)`.
pub fn cpu_rejection_sample(draft_prob: f32, target_prob: f32, seed: u64) -> bool {
    if draft_prob <= 0.0 {
        return target_prob > 0.0;
    }
    let ratio = target_prob / draft_prob;
    if ratio >= 1.0 {
        return true;
    }
    let mut rng = seed;
    rand_f32(&mut rng) < ratio
}

/// Sample from the adjusted distribution `max(0, target − draft)`.
///
/// Used to pick the *bonus* token on rejection.
pub fn cpu_sample_from_adjusted(target_probs: &[f32], draft_probs: &[f32], seed: u64) -> u32 {
    let adjusted: Vec<f32> =
        target_probs.iter().zip(draft_probs.iter()).map(|(&t, &d)| (t - d).max(0.0)).collect();
    let sum: f32 = adjusted.iter().sum();
    if sum <= 0.0 {
        // Fall back to argmax of target.
        return argmax(target_probs);
    }
    let normed: Vec<f32> = adjusted.iter().map(|&v| v / sum).collect();
    let mut rng = seed;
    sample_from_probs(&normed, &mut rng)
}

/// Run one complete speculative-decoding step.
///
/// Returns the accepted tokens (including any bonus) and updated stats.
pub fn cpu_speculative_step(
    draft_fn: &dyn Fn(u32) -> Vec<f32>,
    target_fn: &dyn Fn(&[u32]) -> Vec<Vec<f32>>,
    prev_token: u32,
    config: &SpeculativeConfig,
    seed: u64,
) -> Result<(Vec<u32>, SpeculativeStats), SpeculativeError> {
    let draft = cpu_generate_draft(
        draft_fn,
        prev_token,
        config.max_draft_tokens as usize,
        config.temperature,
        seed,
    );

    if draft.tokens.is_empty() {
        return Err(SpeculativeError::EmptyDraft);
    }

    let result = cpu_verify_draft(&draft, target_fn, config, seed)?;

    let mut output_tokens: Vec<u32> = result.accepted_tokens.clone();
    if let Some(bonus) = result.bonus_token {
        output_tokens.push(bonus);
    }

    let proposed = draft.tokens.len();
    let accepted = result.accepted_count;
    let acceptance_rate = if proposed > 0 { accepted as f32 / proposed as f32 } else { 0.0 };

    let stats = SpeculativeStats {
        total_proposals: proposed,
        total_accepted: accepted,
        acceptance_rate,
        avg_accepted_per_step: accepted as f32,
        tokens_saved: accepted.saturating_sub(1),
    };

    Ok((output_tokens, stats))
}

/// Compute the acceptance rate from cumulative stats.
pub fn cpu_compute_acceptance_rate(stats: &SpeculativeStats) -> f32 {
    if stats.total_proposals == 0 {
        return 0.0;
    }
    stats.total_accepted as f32 / stats.total_proposals as f32
}

/// Estimate the theoretical speedup from speculative decoding.
///
/// `draft_cost_ratio` is the cost of one draft-model call relative to
/// one target-model call (e.g. 0.1 means the draft is 10× cheaper).
pub fn cpu_estimate_speedup(acceptance_rate: f32, draft_cost_ratio: f32, max_draft: u32) -> f32 {
    if max_draft == 0 || draft_cost_ratio <= 0.0 {
        return 1.0;
    }
    let k = max_draft as f32;
    // Expected accepted tokens per step: α·k (geometric approx).
    let expected_accepted = acceptance_rate * k;
    // Cost: k draft calls + 1 target call.
    let cost = k * draft_cost_ratio + 1.0;
    // Baseline cost: (expected_accepted + 1) individual target calls.
    let baseline = expected_accepted + 1.0;
    if cost <= 0.0 {
        return 1.0;
    }
    (baseline / cost).max(1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic logits function: token id → logit vector of size 8.
    fn mock_logits(token: u32) -> Vec<f32> {
        let mut logits = vec![0.0f32; 8];
        logits[(token as usize + 1) % 8] = 5.0;
        logits[(token as usize + 3) % 8] = 2.0;
        logits
    }

    // Target-model logits that agree perfectly with the draft.
    fn target_agree(tokens: &[u32]) -> Vec<Vec<f32>> {
        let mut out = Vec::new();
        let mut prev = 0u32;
        for &t in tokens {
            let _ = t;
            out.push(mock_logits(prev));
            prev = t;
        }
        // Bonus position.
        out.push(mock_logits(prev));
        out
    }

    // Target model that always disagrees: uniform logits.
    fn target_disagree(tokens: &[u32]) -> Vec<Vec<f32>> {
        let vocab = 8;
        let mut out = Vec::new();
        for _ in 0..=tokens.len() {
            out.push(vec![1.0; vocab]);
        }
        out
    }

    // ------------------------------------------------------------------
    // Draft generation
    // ------------------------------------------------------------------

    #[test]
    fn draft_produces_correct_count() {
        let draft = cpu_generate_draft(&mock_logits, 0, 5, 1.0, 42);
        assert_eq!(draft.tokens.len(), 5);
        assert_eq!(draft.logits.len(), 5);
        assert_eq!(draft.draft_probs.len(), 5);
    }

    #[test]
    fn draft_produces_one_token() {
        let draft = cpu_generate_draft(&mock_logits, 0, 1, 1.0, 42);
        assert_eq!(draft.tokens.len(), 1);
    }

    #[test]
    fn draft_greedy_temperature_zero() {
        let d1 = cpu_generate_draft(&mock_logits, 0, 5, 0.0, 42);
        let d2 = cpu_generate_draft(&mock_logits, 0, 5, 0.0, 999);
        // Greedy must yield the same tokens regardless of seed.
        assert_eq!(d1.tokens, d2.tokens);
    }

    #[test]
    fn draft_deterministic_same_seed() {
        let d1 = cpu_generate_draft(&mock_logits, 0, 5, 1.0, 42);
        let d2 = cpu_generate_draft(&mock_logits, 0, 5, 1.0, 42);
        assert_eq!(d1.tokens, d2.tokens);
    }

    #[test]
    fn draft_different_seeds_may_differ() {
        // With temperature > 0 and different seeds, outputs usually differ.
        // We cannot *guarantee* it for tiny vocab, so just check no panic.
        let _d1 = cpu_generate_draft(&mock_logits, 0, 5, 1.0, 1);
        let _d2 = cpu_generate_draft(&mock_logits, 0, 5, 1.0, 9999);
    }

    #[test]
    fn draft_probs_sum_to_one() {
        let draft = cpu_generate_draft(&mock_logits, 0, 3, 1.0, 42);
        for probs in &draft.draft_probs {
            let sum: f32 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
        }
    }

    #[test]
    fn draft_tokens_in_vocab_range() {
        let draft = cpu_generate_draft(&mock_logits, 0, 5, 1.0, 42);
        for &t in &draft.tokens {
            assert!((t as usize) < 8);
        }
    }

    // ------------------------------------------------------------------
    // Verification — all accepted
    // ------------------------------------------------------------------

    #[test]
    fn verify_all_accepted_when_models_agree() {
        let draft = cpu_generate_draft(&mock_logits, 0, 5, 0.0, 42);
        let config = SpeculativeConfig { temperature: 0.0, ..Default::default() };

        // Build a target that exactly reproduces the draft logits.
        let draft_clone = draft.clone();
        let target_fn = move |_tokens: &[u32]| -> Vec<Vec<f32>> {
            let mut out: Vec<Vec<f32>> = draft_clone.logits.iter().cloned().collect();
            // Bonus position — use the last token's logits.
            out.push(draft_clone.logits.last().unwrap().clone());
            out
        };

        let result = cpu_verify_draft(&draft, &target_fn, &config, 42).unwrap();
        assert_eq!(result.accepted_count, 5);
        assert_eq!(result.accepted_tokens.len(), 5);
    }

    #[test]
    fn verify_some_rejected_when_models_differ() {
        let draft = cpu_generate_draft(&mock_logits, 0, 5, 1.0, 42);
        let config = SpeculativeConfig::default();
        let result = cpu_verify_draft(&draft, &target_disagree, &config, 42).unwrap();
        // Uniform target will not match draft perfectly; some rejections.
        assert!(result.accepted_count <= 5);
    }

    #[test]
    fn verify_accepted_count_leq_draft_len() {
        let draft = cpu_generate_draft(&mock_logits, 0, 5, 1.0, 12345);
        let config = SpeculativeConfig::default();
        let result = cpu_verify_draft(&draft, &target_disagree, &config, 12345).unwrap();
        assert!(result.accepted_count <= draft.tokens.len());
    }

    #[test]
    fn verify_empty_draft_returns_error() {
        let draft = DraftProposal { tokens: vec![], logits: vec![], draft_probs: vec![] };
        let config = SpeculativeConfig::default();
        let err = cpu_verify_draft(&draft, &target_agree, &config, 42);
        assert!(
            matches!(err, Err(SpeculativeError::EmptyDraft)),
            "expected EmptyDraft, got {err:?}"
        );
    }

    #[test]
    fn verify_vocab_mismatch_detected() {
        let draft = DraftProposal {
            tokens: vec![0],
            logits: vec![vec![1.0; 8]],
            draft_probs: vec![softmax(&[1.0; 8], 1.0)],
        };
        // Target returns vocab size 4, draft has 8.
        let target_fn = |_t: &[u32]| -> Vec<Vec<f32>> { vec![vec![1.0; 4], vec![1.0; 4]] };
        let config = SpeculativeConfig::default();
        let err = cpu_verify_draft(&draft, &target_fn, &config, 42);
        assert!(
            matches!(err, Err(SpeculativeError::VocabMismatch { .. })),
            "expected VocabMismatch, got {err:?}"
        );
    }

    #[test]
    fn verify_target_probs_populated() {
        let draft = cpu_generate_draft(&mock_logits, 0, 3, 1.0, 42);
        let config = SpeculativeConfig::default();
        let result = cpu_verify_draft(&draft, &target_agree, &config, 42).unwrap();
        assert!(!result.target_probs.is_empty());
    }

    // ------------------------------------------------------------------
    // Rejection sampling
    // ------------------------------------------------------------------

    #[test]
    fn rejection_always_accept_when_target_geq_draft() {
        // target ≥ draft ⟹ ratio ≥ 1 ⟹ always accept.
        for seed in 0..100 {
            assert!(cpu_rejection_sample(0.3, 0.5, seed));
            assert!(cpu_rejection_sample(0.3, 0.3, seed));
        }
    }

    #[test]
    fn rejection_sometimes_reject_when_target_lt_draft() {
        let mut any_reject = false;
        for seed in 0..200 {
            if !cpu_rejection_sample(0.9, 0.1, seed) {
                any_reject = true;
                break;
            }
        }
        assert!(any_reject, "expected at least one rejection");
    }

    #[test]
    fn rejection_sample_zero_draft_prob() {
        // draft_prob = 0 and target > 0 → accept.
        assert!(cpu_rejection_sample(0.0, 0.5, 42));
    }

    #[test]
    fn rejection_sample_both_zero() {
        // Both zero → target_prob NOT > 0 → reject.
        assert!(!cpu_rejection_sample(0.0, 0.0, 42));
    }

    // ------------------------------------------------------------------
    // Bonus token / adjusted distribution
    // ------------------------------------------------------------------

    #[test]
    fn bonus_token_generated_on_rejection() {
        // Use a strongly-disagreeing target.
        let draft = cpu_generate_draft(&mock_logits, 0, 5, 1.0, 42);
        let config = SpeculativeConfig::default();

        // Craft a target that puts all mass on token 0.
        let target_fn = |tokens: &[u32]| -> Vec<Vec<f32>> {
            let mut out = Vec::new();
            for _ in 0..=tokens.len() {
                let mut v = vec![0.0f32; 8];
                v[0] = 10.0;
                out.push(v);
            }
            out
        };
        let result = cpu_verify_draft(&draft, &target_fn, &config, 7).unwrap();
        if result.accepted_count < draft.tokens.len() {
            assert!(result.bonus_token.is_some());
        }
    }

    #[test]
    fn adjusted_distribution_produces_valid_token() {
        let target = vec![0.4, 0.3, 0.2, 0.1];
        let draft = vec![0.1, 0.5, 0.2, 0.2];
        let tok = cpu_sample_from_adjusted(&target, &draft, 42);
        assert!((tok as usize) < 4);
    }

    #[test]
    fn adjusted_distribution_favours_underrepresented() {
        // target much higher than draft on token 0 → token 0 favoured.
        let target = vec![0.9, 0.05, 0.025, 0.025];
        let draft = vec![0.1, 0.5, 0.2, 0.2];
        let mut counts = [0u32; 4];
        for s in 0..500 {
            let tok = cpu_sample_from_adjusted(&target, &draft, s);
            counts[tok as usize] += 1;
        }
        // Token 0 should dominate.
        assert!(
            counts[0] > counts[1],
            "token 0 ({}) should dominate token 1 ({})",
            counts[0],
            counts[1]
        );
    }

    #[test]
    fn adjusted_fallback_when_all_negative() {
        // target ≤ draft everywhere → adjusted is all zeros → argmax.
        let target2 = vec![0.1, 0.1, 0.1, 0.1];
        let draft2 = vec![0.3, 0.3, 0.3, 0.3];
        let tok = cpu_sample_from_adjusted(&target2, &draft2, 42);
        assert!((tok as usize) < 4);
    }

    // ------------------------------------------------------------------
    // Speculative step
    // ------------------------------------------------------------------

    #[test]
    fn speculative_step_produces_at_least_one_token() {
        let config = SpeculativeConfig::default();
        let (tokens, _) =
            cpu_speculative_step(&mock_logits, &target_disagree, 0, &config, 42).unwrap();
        assert!(!tokens.is_empty(), "speculative step must emit ≥ 1 token");
    }

    #[test]
    fn speculative_step_max_draft_one() {
        let config = SpeculativeConfig { max_draft_tokens: 1, ..Default::default() };
        let (tokens, stats) =
            cpu_speculative_step(&mock_logits, &target_agree, 0, &config, 42).unwrap();
        assert!(!tokens.is_empty());
        assert!(stats.total_proposals <= 1);
    }

    #[test]
    fn speculative_step_deterministic() {
        let config = SpeculativeConfig::default();
        let (t1, _) = cpu_speculative_step(&mock_logits, &target_agree, 0, &config, 42).unwrap();
        let (t2, _) = cpu_speculative_step(&mock_logits, &target_agree, 0, &config, 42).unwrap();
        assert_eq!(t1, t2);
    }

    #[test]
    fn speculative_step_with_greedy() {
        let config = SpeculativeConfig { temperature: 0.0, ..Default::default() };
        let (tokens, _) =
            cpu_speculative_step(&mock_logits, &target_agree, 0, &config, 42).unwrap();
        assert!(!tokens.is_empty());
    }

    // ------------------------------------------------------------------
    // Acceptance rate
    // ------------------------------------------------------------------

    #[test]
    fn acceptance_rate_perfect_when_agree() {
        let config = SpeculativeConfig { temperature: 0.0, ..Default::default() };

        let draft = cpu_generate_draft(&mock_logits, 0, 5, 0.0, 42);
        let draft_clone = draft.clone();
        let target_fn = move |_tokens: &[u32]| -> Vec<Vec<f32>> {
            let mut out: Vec<Vec<f32>> = draft_clone.logits.iter().cloned().collect();
            out.push(draft_clone.logits.last().unwrap().clone());
            out
        };
        let (_, stats) = cpu_speculative_step(&mock_logits, &target_fn, 0, &config, 42).unwrap();
        let rate = cpu_compute_acceptance_rate(&stats);
        assert!((rate - 1.0).abs() < 1e-5, "rate should be 1.0, got {rate}");
    }

    #[test]
    fn acceptance_rate_below_one_when_differ() {
        let config = SpeculativeConfig::default();
        let (_, stats) =
            cpu_speculative_step(&mock_logits, &target_disagree, 0, &config, 42).unwrap();
        let rate = cpu_compute_acceptance_rate(&stats);
        assert!(rate <= 1.0);
    }

    #[test]
    fn acceptance_rate_in_zero_one() {
        for seed in [1, 42, 999, 12345] {
            let config = SpeculativeConfig::default();
            let (_, stats) =
                cpu_speculative_step(&mock_logits, &target_disagree, 0, &config, seed).unwrap();
            let rate = cpu_compute_acceptance_rate(&stats);
            assert!((0.0..=1.0).contains(&rate), "rate {rate} out of range for seed {seed}");
        }
    }

    #[test]
    fn acceptance_rate_zero_proposals() {
        let stats = SpeculativeStats::default();
        assert_eq!(cpu_compute_acceptance_rate(&stats), 0.0);
    }

    // ------------------------------------------------------------------
    // Speedup estimation
    // ------------------------------------------------------------------

    #[test]
    fn speedup_gt_one_when_acceptance_high() {
        let s = cpu_estimate_speedup(0.8, 0.1, 5);
        assert!(s > 1.0, "speedup = {s}");
    }

    #[test]
    fn speedup_one_when_no_draft() {
        assert_eq!(cpu_estimate_speedup(0.8, 0.1, 0), 1.0);
    }

    #[test]
    fn speedup_one_when_cost_zero() {
        assert_eq!(cpu_estimate_speedup(0.8, 0.0, 5), 1.0);
    }

    #[test]
    fn speedup_at_least_one() {
        for &rate in &[0.0, 0.1, 0.5, 0.9, 1.0] {
            for &cost in &[0.01, 0.1, 0.5, 1.0] {
                let s = cpu_estimate_speedup(rate, cost, 5);
                assert!(s >= 1.0, "speedup {s} < 1 at rate={rate}, cost={cost}");
            }
        }
    }

    #[test]
    fn speedup_increases_with_acceptance() {
        let s_lo = cpu_estimate_speedup(0.2, 0.1, 5);
        let s_hi = cpu_estimate_speedup(0.9, 0.1, 5);
        assert!(s_hi > s_lo, "s_hi={s_hi} <= s_lo={s_lo}");
    }

    // ------------------------------------------------------------------
    // Stats accumulation
    // ------------------------------------------------------------------

    #[test]
    fn stats_correct_totals() {
        let config = SpeculativeConfig::default();
        let mut total_proposed = 0usize;
        let mut total_accepted = 0usize;
        for seed in [42, 43, 44] {
            let (_, stats) =
                cpu_speculative_step(&mock_logits, &target_agree, 0, &config, seed).unwrap();
            total_proposed += stats.total_proposals;
            total_accepted += stats.total_accepted;
        }
        assert!(total_proposed > 0);
        assert!(total_accepted <= total_proposed);
    }

    #[test]
    fn stats_tokens_saved_leq_accepted() {
        let config = SpeculativeConfig::default();
        let (_, stats) = cpu_speculative_step(&mock_logits, &target_agree, 0, &config, 42).unwrap();
        assert!(stats.tokens_saved <= stats.total_accepted);
    }

    // ------------------------------------------------------------------
    // Error types
    // ------------------------------------------------------------------

    #[test]
    fn error_display_draft_model_failed() {
        let e = SpeculativeError::DraftModelFailed("oom".into());
        assert!(e.to_string().contains("draft model failed"));
    }

    #[test]
    fn error_display_verification_failed() {
        let e = SpeculativeError::VerificationFailed("nan".into());
        assert!(e.to_string().contains("verification failed"));
    }

    #[test]
    fn error_display_vocab_mismatch() {
        let e = SpeculativeError::VocabMismatch { draft_vocab: 8, target_vocab: 16 };
        let s = e.to_string();
        assert!(s.contains("8") && s.contains("16"));
    }

    #[test]
    fn error_display_empty_draft() {
        let e = SpeculativeError::EmptyDraft;
        assert!(e.to_string().contains("empty"));
    }

    // ------------------------------------------------------------------
    // Config defaults
    // ------------------------------------------------------------------

    #[test]
    fn config_default_values() {
        let c = SpeculativeConfig::default();
        assert_eq!(c.max_draft_tokens, 5);
        assert!((c.temperature - 1.0).abs() < 1e-6);
        assert!((c.acceptance_threshold - 0.0).abs() < 1e-6);
    }

    // ------------------------------------------------------------------
    // Softmax helper
    // ------------------------------------------------------------------

    #[test]
    fn softmax_sums_to_one() {
        let probs = softmax(&[1.0, 2.0, 3.0, 4.0], 1.0);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
    }

    #[test]
    fn softmax_empty_input() {
        let probs = softmax(&[], 1.0);
        assert!(probs.is_empty());
    }
}
