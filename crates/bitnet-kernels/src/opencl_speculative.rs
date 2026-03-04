//! OpenCL-accelerated speculative decoding for Intel Arc A770.
//!
//! Speculative decoding speeds up autoregressive generation by using a small
//! *draft* model to propose `γ` (gamma) candidate tokens at once, then
//! verifying them in a single forward pass through the larger *target* model.
//! Accepted tokens skip individual target-model forward passes, yielding a
//! wall-clock speedup proportional to the acceptance rate.
//!
//! # Components
//!
//! | Type | Purpose |
//! |------|---------|
//! | [`SpecConfig`] | Draft model, gamma, acceptance method |
//! | [`DraftProposal`] | Token ids + log-probs from the draft model |
//! | [`VerificationResult`] | How many tokens were accepted / corrected |
//! | [`AcceptanceMethod`] | Greedy, Stochastic, or Typical comparison |
//! | [`SpeculativeDecoder`] | Main draft → verify → accept loop |
//! | [`TokenVerifier`] | Per-position draft-vs-target comparison |
//! | [`SpecStats`] | Runtime statistics (acceptance rate, speedup) |
//! | [`AdaptiveGamma`] | Dynamic γ adjustment from recent history |
//!
//! # OpenCL kernel
//!
//! [`SPECULATIVE_CL`] contains an OpenCL C kernel that verifies all `γ`
//! candidate tokens in parallel on the GPU.

use std::fmt;

// ── Configuration ──────────────────────────────────────────────────────

/// How draft-vs-target distributions are compared.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AcceptanceMethod {
    /// Accept iff `argmax(target) == draft_token`.
    #[default]
    Greedy,
    /// Rejection-sampling: accept with probability
    /// `min(1, target_prob / draft_prob)`.
    Stochastic,
    /// Accept if the token falls within the *typical set* of the target
    /// distribution (entropy-based threshold).
    Typical {
        /// Entropy-ratio threshold in `(0, 1]`.
        threshold: f32,
    },
}

/// Configuration for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpecConfig {
    /// Identifier of the draft model (informational).
    pub draft_model_id: String,
    /// Number of speculative tokens to propose per step (`γ`).
    pub num_speculative_tokens: usize,
    /// Acceptance strategy.
    pub acceptance_method: AcceptanceMethod,
}

impl SpecConfig {
    pub fn new(draft_model_id: impl Into<String>, gamma: usize, method: AcceptanceMethod) -> Self {
        Self {
            draft_model_id: draft_model_id.into(),
            num_speculative_tokens: gamma.max(1),
            acceptance_method: method,
        }
    }
}

impl Default for SpecConfig {
    fn default() -> Self {
        Self {
            draft_model_id: String::from("draft-default"),
            num_speculative_tokens: 4,
            acceptance_method: AcceptanceMethod::Greedy,
        }
    }
}

// ── Draft proposal ─────────────────────────────────────────────────────

/// A batch of tokens proposed by the draft model.
#[derive(Debug, Clone)]
pub struct DraftProposal {
    /// Proposed token ids (length ≤ γ).
    pub token_ids: Vec<u32>,
    /// Log-probabilities assigned by the draft model (parallel to `token_ids`).
    pub log_probs: Vec<f32>,
}

impl DraftProposal {
    pub fn new(token_ids: Vec<u32>, log_probs: Vec<f32>) -> Self {
        assert_eq!(token_ids.len(), log_probs.len(), "token_ids/log_probs length mismatch");
        Self { token_ids, log_probs }
    }

    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }
}

// ── Verification result ────────────────────────────────────────────────

/// Outcome of verifying a [`DraftProposal`] against the target model.
#[derive(Debug, Clone, PartialEq)]
pub struct VerificationResult {
    /// Number of draft tokens accepted (0 … γ).
    pub accepted_count: usize,
    /// Position of the first rejection, or `None` if all accepted.
    pub first_rejected_pos: Option<usize>,
    /// The corrected token sampled from the target at the rejection point
    /// (or the bonus token when all drafts are accepted).
    pub corrected_token: u32,
    /// Final list of tokens to emit this step (accepted drafts + corrected).
    pub output_tokens: Vec<u32>,
}

impl VerificationResult {
    /// Total tokens produced (accepted + 1 corrected/bonus).
    pub fn total_tokens(&self) -> usize {
        self.output_tokens.len()
    }
}

// ── Token verifier ─────────────────────────────────────────────────────

/// Compares draft tokens against target-model distributions.
pub struct TokenVerifier {
    method: AcceptanceMethod,
    rng_state: u64,
}

impl TokenVerifier {
    pub fn new(method: AcceptanceMethod, seed: u64) -> Self {
        Self { method, rng_state: if seed == 0 { 0x5851_F42D_4C95_7F2D } else { seed } }
    }

    /// Verify a single position.
    ///
    /// * `draft_token` – token proposed by the draft model.
    /// * `draft_log_prob` – draft model's log-prob for the token.
    /// * `target_logits` – raw logits from the target model at this position.
    /// * `vocab_size` – vocabulary size.
    ///
    /// Returns `true` if the token is accepted.
    pub fn verify_position(
        &mut self,
        draft_token: u32,
        draft_log_prob: f32,
        target_logits: &[f32],
        vocab_size: usize,
    ) -> bool {
        if target_logits.is_empty() || vocab_size == 0 {
            return false;
        }
        let effective = &target_logits[..vocab_size.min(target_logits.len())];
        match self.method {
            AcceptanceMethod::Greedy => {
                let target_argmax = cpu_argmax(effective);
                target_argmax == draft_token as usize
            }
            AcceptanceMethod::Stochastic => {
                let target_probs = cpu_softmax(effective);
                let target_prob = target_probs.get(draft_token as usize).copied().unwrap_or(0.0);
                let draft_prob = draft_log_prob.exp();
                let accept_prob =
                    if draft_prob > 0.0 { (target_prob / draft_prob).min(1.0) } else { 0.0 };
                let r = xorshift64(&mut self.rng_state);
                r < accept_prob
            }
            AcceptanceMethod::Typical { threshold } => {
                let target_probs = cpu_softmax(effective);
                let target_prob = target_probs.get(draft_token as usize).copied().unwrap_or(0.0);
                if target_prob <= 0.0 {
                    return false;
                }
                let entropy = cpu_entropy(&target_probs);
                if entropy <= 0.0 {
                    // Degenerate distribution (single token with all mass)
                    return cpu_argmax(effective) == draft_token as usize;
                }
                let surprise = -target_prob.ln();
                let deviation = (surprise - entropy).abs() / entropy;
                deviation < threshold
            }
        }
    }

    /// Sample a corrected token from target logits.
    pub fn sample_corrected(&mut self, target_logits: &[f32]) -> u32 {
        if target_logits.is_empty() {
            return 0;
        }
        match self.method {
            AcceptanceMethod::Greedy => cpu_argmax(target_logits) as u32,
            AcceptanceMethod::Stochastic | AcceptanceMethod::Typical { .. } => {
                let probs = cpu_softmax(target_logits);
                cpu_sample_categorical(&probs, &mut self.rng_state)
            }
        }
    }
}

// ── Speculative decoder ────────────────────────────────────────────────

/// Main speculative-decoding loop.
///
/// Coordinates the draft model, target-model verification, and adaptive γ.
pub struct SpeculativeDecoder {
    #[allow(dead_code)]
    config: SpecConfig,
    verifier: TokenVerifier,
    stats: SpecStats,
    adaptive_gamma: AdaptiveGamma,
}

impl SpeculativeDecoder {
    pub fn new(config: SpecConfig, seed: u64) -> Self {
        let verifier = TokenVerifier::new(config.acceptance_method, seed);
        let gamma = config.num_speculative_tokens;
        Self {
            config,
            verifier,
            stats: SpecStats::default(),
            adaptive_gamma: AdaptiveGamma::new(gamma),
        }
    }

    /// Current (possibly adapted) γ.
    pub fn current_gamma(&self) -> usize {
        self.adaptive_gamma.current()
    }

    /// Read-only reference to cumulative statistics.
    pub fn stats(&self) -> &SpecStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = SpecStats::default();
    }

    /// Run one speculative-decoding step.
    ///
    /// * `draft` – tokens + log-probs from the draft model.
    /// * `target_logits_per_pos` – target model logits for each drafted
    ///   position **plus** the bonus position.  Length must be
    ///   `draft.len() + 1`, each inner slice being `vocab_size` wide.
    /// * `vocab_size` – vocabulary size.
    /// * `draft_time_ms` – time spent generating the draft.
    /// * `verify_time_ms` – time spent in the target forward pass.
    pub fn step(
        &mut self,
        draft: &DraftProposal,
        target_logits_per_pos: &[Vec<f32>],
        vocab_size: usize,
        draft_time_ms: f64,
        verify_time_ms: f64,
    ) -> VerificationResult {
        let gamma = draft.len();

        // We need γ+1 sets of target logits (one per draft pos + bonus).
        assert!(
            target_logits_per_pos.len() > gamma,
            "need {} target-logit rows, got {}",
            gamma + 1,
            target_logits_per_pos.len()
        );

        let mut accepted: Vec<u32> = Vec::with_capacity(gamma);
        let mut first_rejected_pos: Option<usize> = None;

        for (i, target_logits_row) in target_logits_per_pos.iter().enumerate().take(gamma) {
            let ok = self.verifier.verify_position(
                draft.token_ids[i],
                draft.log_probs[i],
                target_logits_row,
                vocab_size,
            );
            if ok {
                accepted.push(draft.token_ids[i]);
            } else {
                first_rejected_pos = Some(i);
                break;
            }
        }

        let accepted_count = accepted.len();

        // Corrected / bonus token comes from the position after the last
        // accepted token.
        let correction_pos = accepted_count;
        let corrected_token =
            self.verifier.sample_corrected(&target_logits_per_pos[correction_pos]);

        let mut output_tokens = accepted.clone();
        output_tokens.push(corrected_token);

        // Update statistics.
        self.stats.record_step(accepted_count, gamma, draft_time_ms, verify_time_ms);

        // Adapt γ for the next step.
        self.adaptive_gamma.update(accepted_count, gamma);

        VerificationResult { accepted_count, first_rejected_pos, corrected_token, output_tokens }
    }
}

// ── Statistics ─────────────────────────────────────────────────────────

/// Cumulative statistics for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpecStats {
    pub total_steps: u64,
    pub total_accepted: u64,
    pub total_proposed: u64,
    pub draft_time_ms: f64,
    pub verify_time_ms: f64,
}

impl Default for SpecStats {
    fn default() -> Self {
        Self {
            total_steps: 0,
            total_accepted: 0,
            total_proposed: 0,
            draft_time_ms: 0.0,
            verify_time_ms: 0.0,
        }
    }
}

impl SpecStats {
    /// Record one speculative step.
    pub fn record_step(&mut self, accepted: usize, proposed: usize, draft_ms: f64, verify_ms: f64) {
        self.total_steps += 1;
        self.total_accepted += accepted as u64;
        self.total_proposed += proposed as u64;
        self.draft_time_ms += draft_ms;
        self.verify_time_ms += verify_ms;
    }

    /// Fraction of proposed tokens that were accepted.
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_proposed == 0 {
            return 0.0;
        }
        self.total_accepted as f64 / self.total_proposed as f64
    }

    /// Average number of accepted tokens per step.
    pub fn avg_accepted(&self) -> f64 {
        if self.total_steps == 0 {
            return 0.0;
        }
        self.total_accepted as f64 / self.total_steps as f64
    }

    /// Estimated speedup ratio versus single-token decoding.
    ///
    /// Assumes the cost of a verification pass equals the cost of one
    /// single-token forward pass in the target model, so the speedup is
    /// roughly `(accepted + 1) / 1` per step.
    pub fn speedup_ratio(&self) -> f64 {
        if self.total_steps == 0 {
            return 1.0;
        }
        // total emitted tokens = total_accepted + total_steps (one corrected per step)
        let total_emitted = self.total_accepted + self.total_steps;
        total_emitted as f64 / self.total_steps as f64
    }
}

impl fmt::Display for SpecStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "steps={} accept={:.1}% avg_acc={:.2} speedup={:.2}x \
             draft={:.1}ms verify={:.1}ms",
            self.total_steps,
            self.acceptance_rate() * 100.0,
            self.avg_accepted(),
            self.speedup_ratio(),
            self.draft_time_ms,
            self.verify_time_ms,
        )
    }
}

// ── Adaptive gamma ─────────────────────────────────────────────────────

/// Dynamically adjusts `γ` based on a sliding window of acceptance rates.
#[derive(Debug, Clone)]
pub struct AdaptiveGamma {
    /// Current speculation depth.
    gamma: usize,
    /// Initial / baseline gamma.
    base_gamma: usize,
    /// Minimum γ (always ≥ 1).
    min_gamma: usize,
    /// Maximum γ.
    max_gamma: usize,
    /// Sliding window of (accepted, proposed) per step.
    history: Vec<(usize, usize)>,
    /// Window size.
    window: usize,
    /// If recent acceptance rate exceeds this, increase γ.
    increase_threshold: f64,
    /// If recent acceptance rate drops below this, decrease γ.
    decrease_threshold: f64,
}

impl AdaptiveGamma {
    pub fn new(base_gamma: usize) -> Self {
        let base = base_gamma.max(1);
        Self {
            gamma: base,
            base_gamma: base,
            min_gamma: 1,
            max_gamma: base * 3,
            history: Vec::new(),
            window: 8,
            increase_threshold: 0.8,
            decrease_threshold: 0.3,
        }
    }

    pub fn with_bounds(mut self, min: usize, max: usize) -> Self {
        self.min_gamma = min.max(1);
        self.max_gamma = max.max(self.min_gamma);
        self.gamma = self.gamma.clamp(self.min_gamma, self.max_gamma);
        self
    }

    pub fn with_thresholds(mut self, increase: f64, decrease: f64) -> Self {
        self.increase_threshold = increase;
        self.decrease_threshold = decrease;
        self
    }

    pub fn with_window(mut self, window: usize) -> Self {
        self.window = window.max(1);
        self
    }

    /// Current γ.
    pub fn current(&self) -> usize {
        self.gamma
    }

    /// Update γ after observing one step.
    pub fn update(&mut self, accepted: usize, proposed: usize) {
        self.history.push((accepted, proposed));
        if self.history.len() > self.window {
            self.history.remove(0);
        }

        let rate = self.recent_acceptance_rate();

        if rate >= self.increase_threshold && self.gamma < self.max_gamma {
            self.gamma += 1;
        } else if rate <= self.decrease_threshold && self.gamma > self.min_gamma {
            self.gamma -= 1;
        }
    }

    /// Acceptance rate over the sliding window.
    pub fn recent_acceptance_rate(&self) -> f64 {
        let (total_acc, total_prop) =
            self.history.iter().fold((0usize, 0usize), |(a, p), &(ai, pi)| (a + ai, p + pi));
        if total_prop == 0 {
            return 0.0;
        }
        total_acc as f64 / total_prop as f64
    }

    /// Reset to the base gamma.
    pub fn reset(&mut self) {
        self.gamma = self.base_gamma;
        self.history.clear();
    }
}

// ── CPU reference helpers ──────────────────────────────────────────────

/// Argmax over a slice of f32.
pub fn cpu_argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Numerically-stable softmax.
pub fn cpu_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return vec![0.0; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// Shannon entropy of a probability distribution (nats).
pub fn cpu_entropy(probs: &[f32]) -> f32 {
    probs.iter().fold(0.0f32, |acc, &p| if p > 0.0 { acc - p * p.ln() } else { acc })
}

/// Sample from a categorical distribution using pre-computed probabilities.
pub fn cpu_sample_categorical(probs: &[f32], rng_state: &mut u64) -> u32 {
    let r = xorshift64(rng_state);
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i as u32;
        }
    }
    // Fallback to last token (numerical precision).
    (probs.len().saturating_sub(1)) as u32
}

/// Parallel verification of draft tokens against target distributions (CPU).
///
/// Returns a boolean mask indicating acceptance for each position.
pub fn cpu_verify_batch(
    draft_tokens: &[u32],
    draft_log_probs: &[f32],
    target_logits: &[Vec<f32>],
    vocab_size: usize,
    method: AcceptanceMethod,
    seed: u64,
) -> Vec<bool> {
    let mut verifier = TokenVerifier::new(method, seed);
    let mut results = Vec::with_capacity(draft_tokens.len());
    for i in 0..draft_tokens.len() {
        let logits = if i < target_logits.len() { &target_logits[i] } else { &[] as &[f32] };
        results.push(verifier.verify_position(
            draft_tokens[i],
            draft_log_probs[i],
            logits,
            vocab_size,
        ));
    }
    results
}

/// Xorshift64 PRNG — returns a value in `[0, 1)`.
fn xorshift64(state: &mut u64) -> f32 {
    if *state == 0 {
        *state = 0x5851_F42D_4C95_7F2D;
    }
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f32) / (u64::MAX as f32)
}

// ── OpenCL kernel source ───────────────────────────────────────────────

/// OpenCL C kernel for parallel speculative-token verification.
///
/// Each work-item handles one candidate position, computing the target
/// softmax and comparing against the draft token to produce an accept/reject
/// flag. The host reads back the boolean mask and performs the sequential
/// first-rejection scan on the CPU (a single O(γ) pass).
pub const SPECULATIVE_CL: &str = r#"
/// Parallel speculative-token verification kernel.
///
/// One work-item per candidate position [0..gamma).
///
/// Args:
///   draft_tokens   – uint[gamma]   proposed token ids
///   draft_logprobs – float[gamma]  draft model log-probs
///   target_logits  – float[gamma * vocab_size]  row-major target logits
///   accept_mask    – int[gamma]    output: 1 = accepted, 0 = rejected
///   vocab_size     – uint
///   gamma          – uint
__kernel void verify_speculative(
    __global const uint  *draft_tokens,
    __global const float *draft_logprobs,
    __global const float *target_logits,
    __global       int   *accept_mask,
    const uint vocab_size,
    const uint gamma)
{
    uint pos = get_global_id(0);
    if (pos >= gamma) return;

    // Pointer to this position's row of target logits.
    __global const float *row = target_logits + pos * vocab_size;

    // 1. Find argmax of target logits (greedy acceptance).
    float max_val = row[0];
    uint  argmax  = 0;
    for (uint i = 1; i < vocab_size; i++) {
        if (row[i] > max_val) {
            max_val = row[i];
            argmax  = i;
        }
    }

    accept_mask[pos] = (argmax == draft_tokens[pos]) ? 1 : 0;
}

/// Stochastic verification kernel — computes acceptance probability.
///
/// Writes the acceptance probability into `accept_probs`; the host draws
/// a uniform random number and compares.
__kernel void verify_speculative_stochastic(
    __global const uint  *draft_tokens,
    __global const float *draft_logprobs,
    __global const float *target_logits,
    __global       float *accept_probs,
    const uint vocab_size,
    const uint gamma)
{
    uint pos = get_global_id(0);
    if (pos >= gamma) return;

    __global const float *row = target_logits + pos * vocab_size;
    uint tok = draft_tokens[pos];

    // Numerically-stable softmax for this row.
    float row_max = row[0];
    for (uint i = 1; i < vocab_size; i++) {
        row_max = fmax(row_max, row[i]);
    }
    float sum_exp = 0.0f;
    for (uint i = 0; i < vocab_size; i++) {
        sum_exp += exp(row[i] - row_max);
    }
    float target_prob = exp(row[tok] - row_max) / sum_exp;
    float draft_prob  = exp(draft_logprobs[pos]);

    accept_probs[pos] = (draft_prob > 0.0f)
        ? fmin(target_prob / draft_prob, 1.0f)
        : 0.0f;
}
"#;

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helpers ────────────────────────────────────────────────────────────

    /// Build a flat logits vector that has its max at `token`.
    fn logits_with_max(vocab_size: usize, token: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; vocab_size];
        if token < vocab_size {
            v[token] = 10.0;
        }
        v
    }

    /// Build uniform logits (all equal).
    #[allow(dead_code)]
    fn uniform_logits(vocab_size: usize) -> Vec<f32> {
        vec![1.0f32; vocab_size]
    }

    /// Build a one-hot probability distribution (for stochastic tests).
    fn one_hot_logits(vocab_size: usize, token: usize) -> Vec<f32> {
        let mut v = vec![-100.0f32; vocab_size];
        if token < vocab_size {
            v[token] = 100.0;
        }
        v
    }

    // ── AcceptanceMethod defaults ──────────────────────────────────────

    #[test]
    fn test_acceptance_method_default_is_greedy() {
        assert_eq!(AcceptanceMethod::default(), AcceptanceMethod::Greedy);
    }

    // ── SpecConfig ─────────────────────────────────────────────────────

    #[test]
    fn test_spec_config_default() {
        let cfg = SpecConfig::default();
        assert_eq!(cfg.num_speculative_tokens, 4);
        assert_eq!(cfg.acceptance_method, AcceptanceMethod::Greedy);
    }

    #[test]
    fn test_spec_config_gamma_at_least_one() {
        let cfg = SpecConfig::new("tiny", 0, AcceptanceMethod::Greedy);
        assert_eq!(cfg.num_speculative_tokens, 1);
    }

    #[test]
    fn test_spec_config_custom() {
        let cfg = SpecConfig::new("draft-7b", 8, AcceptanceMethod::Stochastic);
        assert_eq!(cfg.draft_model_id, "draft-7b");
        assert_eq!(cfg.num_speculative_tokens, 8);
        assert_eq!(cfg.acceptance_method, AcceptanceMethod::Stochastic);
    }

    // ── DraftProposal ──────────────────────────────────────────────────

    #[test]
    fn test_draft_proposal_len() {
        let d = DraftProposal::new(vec![1, 2, 3], vec![-0.1, -0.2, -0.3]);
        assert_eq!(d.len(), 3);
        assert!(!d.is_empty());
    }

    #[test]
    fn test_draft_proposal_empty() {
        let d = DraftProposal::new(vec![], vec![]);
        assert!(d.is_empty());
        assert_eq!(d.len(), 0);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn test_draft_proposal_mismatched_lengths() {
        DraftProposal::new(vec![1, 2], vec![-0.1]);
    }

    // ── VerificationResult ─────────────────────────────────────────────

    #[test]
    fn test_verification_result_total_tokens() {
        let vr = VerificationResult {
            accepted_count: 3,
            first_rejected_pos: None,
            corrected_token: 99,
            output_tokens: vec![1, 2, 3, 99],
        };
        assert_eq!(vr.total_tokens(), 4);
    }

    // ── CPU helpers ────────────────────────────────────────────────────

    #[test]
    fn test_cpu_argmax_basic() {
        assert_eq!(cpu_argmax(&[1.0, 3.0, 2.0]), 1);
    }

    #[test]
    fn test_cpu_argmax_single() {
        assert_eq!(cpu_argmax(&[42.0]), 0);
    }

    #[test]
    fn test_cpu_argmax_negative() {
        assert_eq!(cpu_argmax(&[-5.0, -1.0, -3.0]), 1);
    }

    #[test]
    fn test_cpu_softmax_sums_to_one() {
        let probs = cpu_softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_softmax_empty() {
        assert!(cpu_softmax(&[]).is_empty());
    }

    #[test]
    fn test_cpu_softmax_uniform() {
        let probs = cpu_softmax(&[1.0, 1.0, 1.0]);
        for &p in &probs {
            assert!((p - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_cpu_softmax_peaked() {
        let probs = cpu_softmax(&[100.0, 0.0, 0.0]);
        assert!(probs[0] > 0.99);
    }

    #[test]
    fn test_cpu_entropy_uniform() {
        let n = 4;
        let probs = vec![1.0 / n as f32; n];
        let h = cpu_entropy(&probs);
        let expected = (n as f32).ln();
        assert!((h - expected).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_entropy_peaked() {
        let probs = vec![1.0, 0.0, 0.0];
        let h = cpu_entropy(&probs);
        assert!(h.abs() < 1e-6, "entropy of peaked dist should be ~0");
    }

    #[test]
    fn test_cpu_sample_categorical_deterministic() {
        let probs = vec![0.0, 0.0, 1.0];
        let mut rng = 42u64;
        // With all mass on token 2, every sample should be 2.
        for _ in 0..10 {
            assert_eq!(cpu_sample_categorical(&probs, &mut rng), 2);
        }
    }

    #[test]
    fn test_cpu_sample_categorical_returns_valid_token() {
        let probs = cpu_softmax(&[1.0, 2.0, 3.0, 4.0]);
        let mut rng = 123u64;
        for _ in 0..100 {
            let tok = cpu_sample_categorical(&probs, &mut rng);
            assert!((tok as usize) < probs.len());
        }
    }

    // ── Greedy acceptance ──────────────────────────────────────────────

    #[test]
    fn test_greedy_accept_matching_token() {
        let mut v = TokenVerifier::new(AcceptanceMethod::Greedy, 42);
        let logits = logits_with_max(100, 7);
        assert!(v.verify_position(7, -0.1, &logits, 100));
    }

    #[test]
    fn test_greedy_reject_non_matching_token() {
        let mut v = TokenVerifier::new(AcceptanceMethod::Greedy, 42);
        let logits = logits_with_max(100, 7);
        assert!(!v.verify_position(5, -0.1, &logits, 100));
    }

    #[test]
    fn test_greedy_empty_logits_rejected() {
        let mut v = TokenVerifier::new(AcceptanceMethod::Greedy, 42);
        assert!(!v.verify_position(0, 0.0, &[], 0));
    }

    // ── Stochastic acceptance ──────────────────────────────────────────

    #[test]
    fn test_stochastic_high_target_prob_accepted() {
        // Draft has low prob, target has high prob → ratio ≥ 1 → always accept.
        let mut v = TokenVerifier::new(AcceptanceMethod::Stochastic, 42);
        let logits = one_hot_logits(10, 3);
        // draft_log_prob = ln(0.01) ≈ -4.6, target_prob ≈ 1.0 → ratio >> 1
        assert!(v.verify_position(3, -4.6f32, &logits, 10));
    }

    #[test]
    fn test_stochastic_zero_draft_prob_rejected() {
        let mut v = TokenVerifier::new(AcceptanceMethod::Stochastic, 42);
        let logits = logits_with_max(10, 5);
        // draft_log_prob = -inf → exp = 0 → always reject
        assert!(!v.verify_position(5, f32::NEG_INFINITY, &logits, 10));
    }

    #[test]
    fn test_stochastic_identical_distributions() {
        // When draft ≈ target, acceptance should be very high.
        let logits = vec![1.0f32; 4];
        let probs = cpu_softmax(&logits);
        let draft_lp = probs[0].ln();
        let mut accepted = 0;
        for seed in 1..=100 {
            let mut v = TokenVerifier::new(AcceptanceMethod::Stochastic, seed);
            if v.verify_position(0, draft_lp, &logits, 4) {
                accepted += 1;
            }
        }
        // Should accept most of the time (ratio ≈ 1.0).
        assert!(accepted > 80, "expected high acceptance, got {accepted}/100");
    }

    // ── Typical acceptance ─────────────────────────────────────────────

    #[test]
    fn test_typical_accepts_high_prob_token() {
        let mut v = TokenVerifier::new(AcceptanceMethod::Typical { threshold: 0.9 }, 42);
        // Peaked distribution: token 2 has most mass → low surprise → accepted.
        let logits = one_hot_logits(10, 2);
        assert!(v.verify_position(2, -0.01, &logits, 10));
    }

    #[test]
    fn test_typical_rejects_low_prob_token() {
        let mut v = TokenVerifier::new(AcceptanceMethod::Typical { threshold: 0.1 }, 42);
        // Uniform logits → entropy = ln(10). Token prob = 0.1, surprise = ln(10).
        // deviation = |ln(10) - ln(10)| / ln(10) = 0 → accepted actually.
        // Use skewed distribution instead.
        let mut logits = vec![0.0f32; 10];
        logits[0] = 10.0; // peaked at 0
        // Token 5 has very low prob under this distribution.
        let accepted = v.verify_position(5, -10.0, &logits, 10);
        // Token 5's surprise is high relative to entropy → deviation > threshold.
        assert!(!accepted);
    }

    // ── Speculative decoder: γ=1 ──────────────────────────────────────

    #[test]
    fn test_gamma_1_accept() {
        let cfg = SpecConfig::new("draft", 1, AcceptanceMethod::Greedy);
        let mut dec = SpeculativeDecoder::new(cfg, 42);
        let draft = DraftProposal::new(vec![7], vec![-0.1]);
        let target = vec![logits_with_max(10, 7), logits_with_max(10, 3)];
        let res = dec.step(&draft, &target, 10, 1.0, 2.0);
        assert_eq!(res.accepted_count, 1);
        assert!(res.first_rejected_pos.is_none());
        assert_eq!(res.output_tokens, vec![7, 3]); // accepted + bonus
    }

    #[test]
    fn test_gamma_1_reject() {
        let cfg = SpecConfig::new("draft", 1, AcceptanceMethod::Greedy);
        let mut dec = SpeculativeDecoder::new(cfg, 42);
        let draft = DraftProposal::new(vec![5], vec![-0.1]);
        let target = vec![logits_with_max(10, 7), logits_with_max(10, 3)];
        let res = dec.step(&draft, &target, 10, 1.0, 2.0);
        assert_eq!(res.accepted_count, 0);
        assert_eq!(res.first_rejected_pos, Some(0));
        assert_eq!(res.corrected_token, 7);
        assert_eq!(res.output_tokens, vec![7]);
    }

    // ── Speculative decoder: γ=4 ──────────────────────────────────────

    #[test]
    fn test_gamma_4_all_accepted() {
        let cfg = SpecConfig::new("draft", 4, AcceptanceMethod::Greedy);
        let mut dec = SpeculativeDecoder::new(cfg, 42);
        let draft = DraftProposal::new(vec![1, 2, 3, 4], vec![-0.1; 4]);
        // Target argmax matches all draft tokens + bonus at pos 4.
        let target = vec![
            logits_with_max(10, 1),
            logits_with_max(10, 2),
            logits_with_max(10, 3),
            logits_with_max(10, 4),
            logits_with_max(10, 9), // bonus
        ];
        let res = dec.step(&draft, &target, 10, 1.0, 2.0);
        assert_eq!(res.accepted_count, 4);
        assert!(res.first_rejected_pos.is_none());
        assert_eq!(res.output_tokens, vec![1, 2, 3, 4, 9]);
    }

    #[test]
    fn test_gamma_4_all_rejected() {
        let cfg = SpecConfig::new("draft", 4, AcceptanceMethod::Greedy);
        let mut dec = SpeculativeDecoder::new(cfg, 42);
        let draft = DraftProposal::new(vec![1, 2, 3, 4], vec![-0.1; 4]);
        // Target argmax is 0 everywhere → no match.
        let target = vec![logits_with_max(10, 0); 5];
        let res = dec.step(&draft, &target, 10, 1.0, 2.0);
        assert_eq!(res.accepted_count, 0);
        assert_eq!(res.first_rejected_pos, Some(0));
        assert_eq!(res.corrected_token, 0);
        assert_eq!(res.output_tokens, vec![0]);
    }

    #[test]
    fn test_gamma_4_partial_accept() {
        let cfg = SpecConfig::new("draft", 4, AcceptanceMethod::Greedy);
        let mut dec = SpeculativeDecoder::new(cfg, 42);
        let draft = DraftProposal::new(vec![1, 2, 3, 4], vec![-0.1; 4]);
        // First two match, third doesn't.
        let target = vec![
            logits_with_max(10, 1), // match
            logits_with_max(10, 2), // match
            logits_with_max(10, 9), // mismatch (draft=3)
            logits_with_max(10, 4), // never reached
            logits_with_max(10, 5), // never reached
        ];
        let res = dec.step(&draft, &target, 10, 1.0, 2.0);
        assert_eq!(res.accepted_count, 2);
        assert_eq!(res.first_rejected_pos, Some(2));
        assert_eq!(res.corrected_token, 9);
        assert_eq!(res.output_tokens, vec![1, 2, 9]);
    }

    // ── Statistics tracking ────────────────────────────────────────────

    #[test]
    fn test_stats_initial() {
        let s = SpecStats::default();
        assert_eq!(s.total_steps, 0);
        assert_eq!(s.acceptance_rate(), 0.0);
        assert_eq!(s.avg_accepted(), 0.0);
        assert_eq!(s.speedup_ratio(), 1.0);
    }

    #[test]
    fn test_stats_after_one_step() {
        let mut s = SpecStats::default();
        s.record_step(3, 4, 1.0, 2.0);
        assert_eq!(s.total_steps, 1);
        assert_eq!(s.total_accepted, 3);
        assert_eq!(s.total_proposed, 4);
        assert!((s.acceptance_rate() - 0.75).abs() < 1e-9);
        assert!((s.avg_accepted() - 3.0).abs() < 1e-9);
        // speedup = (3 + 1) / 1 = 4.0
        assert!((s.speedup_ratio() - 4.0).abs() < 1e-9);
        assert!((s.draft_time_ms - 1.0).abs() < 1e-9);
        assert!((s.verify_time_ms - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_multiple_steps() {
        let mut s = SpecStats::default();
        s.record_step(4, 4, 1.0, 1.0);
        s.record_step(2, 4, 1.0, 1.0);
        s.record_step(0, 4, 1.0, 1.0);
        assert_eq!(s.total_steps, 3);
        assert_eq!(s.total_accepted, 6);
        assert_eq!(s.total_proposed, 12);
        assert!((s.acceptance_rate() - 0.5).abs() < 1e-9);
        assert!((s.avg_accepted() - 2.0).abs() < 1e-9);
        // speedup = (6 + 3) / 3 = 3.0
        assert!((s.speedup_ratio() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_display() {
        let s = SpecStats::default();
        let text = format!("{s}");
        assert!(text.contains("steps="));
        assert!(text.contains("accept="));
    }

    #[test]
    fn test_stats_reset() {
        let cfg = SpecConfig::new("d", 2, AcceptanceMethod::Greedy);
        let mut dec = SpeculativeDecoder::new(cfg, 1);
        let draft = DraftProposal::new(vec![0, 0], vec![-0.1; 2]);
        let target = vec![logits_with_max(4, 0); 3];
        dec.step(&draft, &target, 4, 1.0, 1.0);
        assert!(dec.stats().total_steps > 0);
        dec.reset_stats();
        assert_eq!(dec.stats().total_steps, 0);
    }

    // ── Decoder integration with stats ─────────────────────────────────

    #[test]
    fn test_decoder_stats_after_steps() {
        let cfg = SpecConfig::new("draft", 4, AcceptanceMethod::Greedy);
        let mut dec = SpeculativeDecoder::new(cfg, 42);

        // Step 1: all accepted.
        let draft = DraftProposal::new(vec![1, 2, 3, 4], vec![-0.1; 4]);
        let target = vec![
            logits_with_max(10, 1),
            logits_with_max(10, 2),
            logits_with_max(10, 3),
            logits_with_max(10, 4),
            logits_with_max(10, 0),
        ];
        dec.step(&draft, &target, 10, 5.0, 10.0);

        // Step 2: none accepted.
        let draft2 = DraftProposal::new(vec![9, 8, 7, 6], vec![-0.1; 4]);
        let target2 = vec![logits_with_max(10, 0); 5];
        dec.step(&draft2, &target2, 10, 3.0, 8.0);

        assert_eq!(dec.stats().total_steps, 2);
        assert_eq!(dec.stats().total_accepted, 4);
        assert_eq!(dec.stats().total_proposed, 8);
        assert!((dec.stats().acceptance_rate() - 0.5).abs() < 1e-9);
        assert!((dec.stats().draft_time_ms - 8.0).abs() < 1e-9);
        assert!((dec.stats().verify_time_ms - 18.0).abs() < 1e-9);
    }

    // ── Adaptive gamma ─────────────────────────────────────────────────

    #[test]
    fn test_adaptive_gamma_initial() {
        let ag = AdaptiveGamma::new(4);
        assert_eq!(ag.current(), 4);
    }

    #[test]
    fn test_adaptive_gamma_increases_on_high_acceptance() {
        let mut ag =
            AdaptiveGamma::new(4).with_bounds(1, 12).with_thresholds(0.8, 0.3).with_window(4);
        // Feed all-accepted steps.
        for _ in 0..4 {
            ag.update(4, 4); // 100% acceptance
        }
        assert!(ag.current() > 4, "gamma should increase, got {}", ag.current());
    }

    #[test]
    fn test_adaptive_gamma_decreases_on_low_acceptance() {
        let mut ag =
            AdaptiveGamma::new(4).with_bounds(1, 12).with_thresholds(0.8, 0.3).with_window(4);
        // Feed zero-accepted steps.
        for _ in 0..4 {
            ag.update(0, 4); // 0% acceptance
        }
        assert!(ag.current() < 4, "gamma should decrease, got {}", ag.current());
    }

    #[test]
    fn test_adaptive_gamma_respects_min_bound() {
        let mut ag =
            AdaptiveGamma::new(2).with_bounds(2, 8).with_thresholds(0.8, 0.3).with_window(2);
        for _ in 0..10 {
            ag.update(0, 4);
        }
        assert!(ag.current() >= 2, "gamma below min bound: {}", ag.current());
    }

    #[test]
    fn test_adaptive_gamma_respects_max_bound() {
        let mut ag =
            AdaptiveGamma::new(4).with_bounds(1, 6).with_thresholds(0.8, 0.3).with_window(2);
        for _ in 0..20 {
            ag.update(4, 4);
        }
        assert!(ag.current() <= 6, "gamma above max bound: {}", ag.current());
    }

    #[test]
    fn test_adaptive_gamma_stable_at_moderate_rate() {
        let mut ag =
            AdaptiveGamma::new(4).with_bounds(1, 12).with_thresholds(0.8, 0.3).with_window(4);
        // 50% acceptance – between thresholds → no change.
        for _ in 0..4 {
            ag.update(2, 4);
        }
        assert_eq!(ag.current(), 4);
    }

    #[test]
    fn test_adaptive_gamma_reset() {
        let mut ag = AdaptiveGamma::new(4);
        ag.update(4, 4);
        ag.update(4, 4);
        ag.reset();
        assert_eq!(ag.current(), 4);
        assert!((ag.recent_acceptance_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_adaptive_gamma_window_slides() {
        let mut ag =
            AdaptiveGamma::new(4).with_bounds(1, 12).with_thresholds(0.8, 0.3).with_window(3);
        // Three high steps.
        ag.update(4, 4);
        ag.update(4, 4);
        ag.update(4, 4);
        assert!(ag.recent_acceptance_rate() > 0.9);
        // Three low steps — old high ones should slide out.
        ag.update(0, 4);
        ag.update(0, 4);
        ag.update(0, 4);
        assert!(ag.recent_acceptance_rate() < 0.1);
    }

    // ── Decoder uses adaptive gamma ────────────────────────────────────

    #[test]
    fn test_decoder_gamma_adapts() {
        let cfg = SpecConfig::new("draft", 4, AcceptanceMethod::Greedy);
        let mut dec = SpeculativeDecoder::new(cfg, 42);
        let initial = dec.current_gamma();

        // Feed 10 all-accepted steps.
        for _ in 0..10 {
            let draft =
                DraftProposal::new(vec![1; dec.current_gamma()], vec![-0.1; dec.current_gamma()]);
            let gamma = dec.current_gamma();
            let target: Vec<Vec<f32>> = (0..gamma + 1).map(|_| logits_with_max(10, 1)).collect();
            dec.step(&draft, &target, 10, 1.0, 1.0);
        }
        assert!(
            dec.current_gamma() > initial,
            "gamma should have increased from {initial}: got {}",
            dec.current_gamma()
        );
    }

    // ── Batch verification (CPU) ───────────────────────────────────────

    #[test]
    fn test_cpu_verify_batch_greedy_all_match() {
        let draft_tokens = vec![1, 2, 3];
        let draft_lps = vec![-0.1; 3];
        let target = vec![logits_with_max(10, 1), logits_with_max(10, 2), logits_with_max(10, 3)];
        let mask =
            cpu_verify_batch(&draft_tokens, &draft_lps, &target, 10, AcceptanceMethod::Greedy, 42);
        assert_eq!(mask, vec![true, true, true]);
    }

    #[test]
    fn test_cpu_verify_batch_greedy_none_match() {
        let draft_tokens = vec![1, 2, 3];
        let draft_lps = vec![-0.1; 3];
        let target = vec![logits_with_max(10, 0); 3];
        let mask =
            cpu_verify_batch(&draft_tokens, &draft_lps, &target, 10, AcceptanceMethod::Greedy, 42);
        assert_eq!(mask, vec![false, false, false]);
    }

    #[test]
    fn test_cpu_verify_batch_partial() {
        let draft_tokens = vec![1, 2, 3];
        let draft_lps = vec![-0.1; 3];
        let target = vec![
            logits_with_max(10, 1), // match
            logits_with_max(10, 0), // mismatch
            logits_with_max(10, 3), // match but sequential scan stops earlier
        ];
        let mask =
            cpu_verify_batch(&draft_tokens, &draft_lps, &target, 10, AcceptanceMethod::Greedy, 42);
        assert_eq!(mask, vec![true, false, true]);
    }

    // ── OpenCL kernel source ───────────────────────────────────────────

    #[test]
    fn test_kernel_source_not_empty() {
        assert!(!SPECULATIVE_CL.is_empty());
    }

    #[test]
    fn test_kernel_contains_greedy_kernel() {
        assert!(SPECULATIVE_CL.contains("__kernel void verify_speculative"));
    }

    #[test]
    fn test_kernel_contains_stochastic_kernel() {
        assert!(SPECULATIVE_CL.contains("verify_speculative_stochastic"));
    }

    #[test]
    fn test_kernel_contains_global_id() {
        assert!(SPECULATIVE_CL.contains("get_global_id"));
    }

    #[test]
    fn test_kernel_contains_softmax_computation() {
        assert!(SPECULATIVE_CL.contains("exp("));
    }

    #[test]
    fn test_kernel_contains_accept_mask_output() {
        assert!(SPECULATIVE_CL.contains("accept_mask"));
    }

    // ── Edge cases ─────────────────────────────────────────────────────

    #[test]
    fn test_single_vocab_token() {
        let cfg = SpecConfig::new("draft", 2, AcceptanceMethod::Greedy);
        let mut dec = SpeculativeDecoder::new(cfg, 42);
        // vocab_size=1: only token 0 exists.
        let draft = DraftProposal::new(vec![0, 0], vec![0.0; 2]);
        let target = vec![vec![5.0]; 3]; // three rows, each with single logit
        let res = dec.step(&draft, &target, 1, 1.0, 1.0);
        assert_eq!(res.accepted_count, 2);
        assert_eq!(res.output_tokens, vec![0, 0, 0]);
    }

    #[test]
    fn test_large_vocab() {
        let vocab = 50_000;
        let cfg = SpecConfig::new("draft", 1, AcceptanceMethod::Greedy);
        let mut dec = SpeculativeDecoder::new(cfg, 42);
        let draft = DraftProposal::new(vec![49_999], vec![-0.1]);
        let target = vec![logits_with_max(vocab, 49_999), logits_with_max(vocab, 0)];
        let res = dec.step(&draft, &target, vocab, 1.0, 1.0);
        assert_eq!(res.accepted_count, 1);
        assert_eq!(res.output_tokens[0], 49_999);
    }

    #[test]
    fn test_identical_distributions_stochastic() {
        // When draft and target are identical, acceptance rate → 100%.
        let logits = vec![1.0f32; 8];
        let prob = cpu_softmax(&logits);
        let draft_lp = prob[0].ln();
        let cfg = SpecConfig::new("draft", 1, AcceptanceMethod::Stochastic);
        let mut accepted_total = 0;
        for seed in 1..=50u64 {
            let mut dec = SpeculativeDecoder::new(cfg.clone(), seed);
            let draft = DraftProposal::new(vec![0], vec![draft_lp]);
            let target = vec![logits.clone(), logits.clone()];
            let res = dec.step(&draft, &target, 8, 0.0, 0.0);
            accepted_total += res.accepted_count;
        }
        assert!(
            accepted_total > 40,
            "expected high acceptance for identical dists, got {accepted_total}/50"
        );
    }

    // ── Property-style tests ───────────────────────────────────────────

    #[test]
    fn test_output_always_has_at_least_one_token() {
        for gamma in 1..=6 {
            let cfg = SpecConfig::new("d", gamma, AcceptanceMethod::Greedy);
            let mut dec = SpeculativeDecoder::new(cfg, 42);
            let draft = DraftProposal::new(vec![0; gamma], vec![-0.1; gamma]);
            let target = vec![logits_with_max(10, 5); gamma + 1];
            let res = dec.step(&draft, &target, 10, 0.0, 0.0);
            assert!(!res.output_tokens.is_empty(), "output must never be empty (gamma={gamma})");
        }
    }

    #[test]
    fn test_accepted_count_never_exceeds_gamma() {
        for gamma in 1..=8 {
            let cfg = SpecConfig::new("d", gamma, AcceptanceMethod::Greedy);
            let mut dec = SpeculativeDecoder::new(cfg, 42);
            let draft = DraftProposal::new(vec![1; gamma], vec![-0.1; gamma]);
            let target: Vec<Vec<f32>> = (0..gamma + 1).map(|_| logits_with_max(10, 1)).collect();
            let res = dec.step(&draft, &target, 10, 0.0, 0.0);
            assert!(
                res.accepted_count <= gamma,
                "accepted {} > gamma {} ",
                res.accepted_count,
                gamma,
            );
        }
    }

    #[test]
    fn test_output_length_equals_accepted_plus_one() {
        for gamma in 1..=6 {
            for target_tok in 0..5 {
                let cfg = SpecConfig::new("d", gamma, AcceptanceMethod::Greedy);
                let mut dec = SpeculativeDecoder::new(cfg, 42);
                let draft = DraftProposal::new(vec![0; gamma], vec![-0.1; gamma]);
                let target = vec![logits_with_max(10, target_tok); gamma + 1];
                let res = dec.step(&draft, &target, 10, 0.0, 0.0);
                assert_eq!(
                    res.output_tokens.len(),
                    res.accepted_count + 1,
                    "output len should be accepted+1 (gamma={gamma}, target_tok={target_tok})"
                );
            }
        }
    }

    #[test]
    fn test_all_output_tokens_within_vocab() {
        let vocab = 16;
        for gamma in 1..=4 {
            let cfg = SpecConfig::new("d", gamma, AcceptanceMethod::Greedy);
            let mut dec = SpeculativeDecoder::new(cfg, 42);
            let draft = DraftProposal::new(vec![1; gamma], vec![-0.1; gamma]);
            let target: Vec<Vec<f32>> = (0..gamma + 1).map(|_| logits_with_max(vocab, 1)).collect();
            let res = dec.step(&draft, &target, vocab, 0.0, 0.0);
            for &tok in &res.output_tokens {
                assert!((tok as usize) < vocab, "token {tok} out of vocab range [0, {vocab})");
            }
        }
    }

    #[test]
    fn test_corrected_token_always_present() {
        // Even when all are accepted, a bonus token is always appended.
        let cfg = SpecConfig::new("d", 3, AcceptanceMethod::Greedy);
        let mut dec = SpeculativeDecoder::new(cfg, 42);
        let draft = DraftProposal::new(vec![1, 2, 3], vec![-0.1; 3]);
        let target = vec![
            logits_with_max(10, 1),
            logits_with_max(10, 2),
            logits_with_max(10, 3),
            logits_with_max(10, 7), // bonus
        ];
        let res = dec.step(&draft, &target, 10, 0.0, 0.0);
        assert_eq!(res.corrected_token, 7);
        assert_eq!(*res.output_tokens.last().unwrap(), 7);
    }

    // ── Verifier with different methods ────────────────────────────────

    #[test]
    fn test_verifier_sample_corrected_greedy() {
        let mut v = TokenVerifier::new(AcceptanceMethod::Greedy, 42);
        let logits = logits_with_max(10, 5);
        assert_eq!(v.sample_corrected(&logits), 5);
    }

    #[test]
    fn test_verifier_sample_corrected_stochastic() {
        let mut v = TokenVerifier::new(AcceptanceMethod::Stochastic, 42);
        let logits = one_hot_logits(10, 3);
        // Nearly all mass on token 3.
        assert_eq!(v.sample_corrected(&logits), 3);
    }

    #[test]
    fn test_verifier_sample_corrected_empty() {
        let mut v = TokenVerifier::new(AcceptanceMethod::Greedy, 42);
        assert_eq!(v.sample_corrected(&[]), 0);
    }
}
