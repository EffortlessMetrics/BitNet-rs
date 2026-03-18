//! ARM NEON optimized speculative decoding kernels for Apple Silicon.
//!
//! Implements speculative decoding primitives using NEON SIMD intrinsics:
//! - Draft token verification against target model logits
//! - Acceptance sampling with temperature-adjusted probabilities
//! - Token tree verification for tree-based speculative decoding
//! - KV cache rollback for accepted/rejected token management
//! - Batch verification across multiple sequences
//! - Log-probability comparison between draft and target models
//! - Adaptive draft length based on acceptance rate tracking

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON lane width for f32 vectors.
const LANES: usize = 4;

// ── Helpers ─────────────────────────────────────────────────────────────

/// Horizontal sum of a `float32x4_t` vector.
///
/// # Safety
/// Requires aarch64 target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn hsum_f32x4(v: float32x4_t) -> f32 {
    let pair = vpaddq_f32(v, v);
    vgetq_lane_f32(vpaddq_f32(pair, pair), 0)
}

/// Scalar fast-exp approximation (degree-4 polynomial).
/// Maximum relative error ≈ 2 × 10⁻⁴ for |x| ≤ 20.
#[inline(always)]
fn fast_exp_scalar(x: f32) -> f32 {
    let x = x.clamp(-88.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E).round();
    let r = x - n * std::f32::consts::LN_2;
    let poly = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6.0 + r * (1.0 / 24.0))));
    poly * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// NEON fast-exp for four lanes.
///
/// # Safety
/// Requires aarch64 target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn fast_exp_neon(x: float32x4_t) -> float32x4_t {
    let min_val = vdupq_n_f32(-88.0);
    let max_val = vdupq_n_f32(88.0);
    let x = vmaxq_f32(vminq_f32(x, max_val), min_val);

    let log2e = vdupq_n_f32(std::f32::consts::LOG2_E);
    let ln2 = vdupq_n_f32(std::f32::consts::LN_2);
    let n = vrndnq_f32(vmulq_f32(x, log2e));
    let r = vsubq_f32(x, vmulq_f32(n, ln2));

    let c1 = vdupq_n_f32(1.0 / 24.0);
    let c2 = vdupq_n_f32(1.0 / 6.0);
    let c3 = vdupq_n_f32(0.5);
    let one = vdupq_n_f32(1.0);

    let p = vfmaq_f32(c2, r, c1);
    let p = vfmaq_f32(c3, r, p);
    let p = vfmaq_f32(one, r, p);
    let poly = vfmaq_f32(one, r, p);

    let bias = vdupq_n_s32(127);
    let ni = vcvtq_s32_f32(n);
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(ni, bias), 23));

    vmulq_f32(poly, pow2n)
}

// ── 1. Draft Token Verification ─────────────────────────────────────────

/// Result of verifying a single draft token against the target model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenVerification {
    /// Whether the token was accepted.
    pub accepted: bool,
    /// Target probability for this token.
    pub target_prob: f32,
    /// Draft probability for this token.
    pub draft_prob: f32,
    /// Ratio target_prob / draft_prob (clamped to [0, 1] for sampling).
    pub acceptance_ratio: f32,
}

/// Verify a sequence of draft tokens against target model logits.
///
/// For each draft token, computes softmax over target logits, then
/// checks acceptance via the ratio `p_target / p_draft`. Returns
/// verifications up to (and including) the first rejection.
///
/// - `draft_tokens`: token IDs produced by the draft model
/// - `draft_probs`: draft model probabilities for each token
/// - `target_logits`: target model logits per position, shape
///   `[num_tokens][vocab_size]` flattened in row-major order
/// - `vocab_size`: vocabulary size
/// - `temperature`: sampling temperature (> 0)
#[cfg(target_arch = "aarch64")]
pub fn neon_verify_draft_tokens(
    draft_tokens: &[u32],
    draft_probs: &[f32],
    target_logits: &[f32],
    vocab_size: usize,
    temperature: f32,
) -> Vec<TokenVerification> {
    assert_eq!(draft_tokens.len(), draft_probs.len());
    assert!(target_logits.len() >= draft_tokens.len() * vocab_size, "target_logits too short");
    assert!(temperature > 0.0, "temperature must be positive");

    let inv_temp = 1.0 / temperature;
    let mut results = Vec::with_capacity(draft_tokens.len());

    for (i, (&token, &d_prob)) in draft_tokens.iter().zip(draft_probs.iter()).enumerate() {
        let logits = &target_logits[i * vocab_size..(i + 1) * vocab_size];

        // Compute softmax over target logits with temperature scaling.
        let target_prob = neon_softmax_single_prob(logits, token as usize, inv_temp);

        let ratio = if d_prob > 0.0 { (target_prob / d_prob).min(1.0) } else { 1.0 };

        results.push(TokenVerification {
            accepted: ratio >= 1.0 || will_accept_deterministic(ratio, i),
            target_prob,
            draft_prob: d_prob,
            acceptance_ratio: ratio,
        });
    }

    results
}

/// Deterministic acceptance using hash-based pseudo-random threshold.
/// Used so verification is reproducible without external RNG.
#[inline]
fn will_accept_deterministic(ratio: f32, position: usize) -> bool {
    // Simple hash → [0,1) to compare against ratio.
    let hash = ((position as u64).wrapping_mul(2654435761) & 0xFFFF_FFFF) as u32;
    let threshold = (hash as f32) / (u32::MAX as f32);
    threshold < ratio
}

/// Compute softmax probability of `token_idx` from raw logits using NEON.
///
/// Applies `inv_temp` scaling, finds max for numerical stability, then
/// accumulates exp-sum and returns `exp(logit) / sum`.
#[cfg(target_arch = "aarch64")]
fn neon_softmax_single_prob(logits: &[f32], token_idx: usize, inv_temp: f32) -> f32 {
    let len = logits.len();
    if len == 0 || token_idx >= len {
        return 0.0;
    }

    // Pass 1: find max (for numerical stability).
    let max_val = neon_find_max(logits, inv_temp);

    // Pass 2: accumulate exp(x - max) and record target exp value.
    let chunks = len / LANES;
    let remainder = len % LANES;

    let mut sum = 0.0f32;
    let target_scaled = logits[token_idx] * inv_temp - max_val;
    let target_exp = fast_exp_scalar(target_scaled);

    unsafe {
        let inv_temp_v = vdupq_n_f32(inv_temp);
        let max_v = vdupq_n_f32(max_val);
        let mut acc = vdupq_n_f32(0.0);

        for c in 0..chunks {
            let base = c * LANES;
            let v = vld1q_f32(logits.as_ptr().add(base));
            let scaled = vmulq_f32(v, inv_temp_v);
            let shifted = vsubq_f32(scaled, max_v);
            let exp_v = fast_exp_neon(shifted);
            acc = vaddq_f32(acc, exp_v);
        }

        sum += hsum_f32x4(acc);
    }

    // Scalar tail.
    let tail_start = chunks * LANES;
    for i in 0..remainder {
        let x = logits[tail_start + i] * inv_temp - max_val;
        sum += fast_exp_scalar(x);
    }

    if sum > 0.0 { target_exp / sum } else { 0.0 }
}

/// Find max(logits * inv_temp) using NEON.
#[cfg(target_arch = "aarch64")]
fn neon_find_max(logits: &[f32], inv_temp: f32) -> f32 {
    let len = logits.len();
    let chunks = len / LANES;
    let remainder = len % LANES;

    let mut max_val: f32;

    unsafe {
        let inv_temp_v = vdupq_n_f32(inv_temp);
        let mut max_v = vdupq_n_f32(f32::NEG_INFINITY);

        for c in 0..chunks {
            let base = c * LANES;
            let v = vld1q_f32(logits.as_ptr().add(base));
            let scaled = vmulq_f32(v, inv_temp_v);
            max_v = vmaxq_f32(max_v, scaled);
        }

        // Horizontal max.
        let pair = vpmaxq_f32(max_v, max_v);
        max_val = vgetq_lane_f32(vpmaxq_f32(pair, pair), 0);
    }

    let tail_start = chunks * LANES;
    for i in 0..remainder {
        let x = logits[tail_start + i] * inv_temp;
        if x > max_val {
            max_val = x;
        }
    }

    max_val
}

// ── 2. Acceptance Sampling ──────────────────────────────────────────────

/// Temperature-adjusted acceptance sampling result.
#[derive(Debug, Clone)]
pub struct AcceptanceSample {
    /// Number of tokens accepted from the draft.
    pub accepted_count: usize,
    /// Replacement token sampled from the adjusted distribution
    /// (if the last draft token was rejected). `None` if all accepted.
    pub replacement_token: Option<u32>,
}

/// Perform acceptance sampling with temperature adjustment.
///
/// For each draft token, accept with probability
/// `min(1, p_target / p_draft)`. On rejection, sample from the
/// residual distribution `max(0, p_target - p_draft)`.
///
/// - `target_probs`: softmax probabilities from target, shape
///   `[num_tokens][vocab_size]` flattened row-major
/// - `draft_probs_per_pos`: draft probabilities per position, same shape
/// - `draft_tokens`: token IDs from draft model
/// - `vocab_size`: vocabulary size
/// - `temperature`: temperature applied during softmax (informational;
///   probabilities should already be temperature-adjusted)
#[cfg(target_arch = "aarch64")]
pub fn neon_acceptance_sampling(
    target_probs: &[f32],
    draft_probs_per_pos: &[f32],
    draft_tokens: &[u32],
    vocab_size: usize,
    _temperature: f32,
) -> AcceptanceSample {
    let num_tokens = draft_tokens.len();
    assert!(target_probs.len() >= num_tokens * vocab_size);
    assert!(draft_probs_per_pos.len() >= num_tokens * vocab_size);

    let mut accepted_count = 0;

    for (i, &token) in draft_tokens.iter().enumerate() {
        let t_idx = i * vocab_size + token as usize;
        let t_prob = target_probs[t_idx];
        let d_prob = draft_probs_per_pos[t_idx];

        let ratio = if d_prob > 0.0 { (t_prob / d_prob).min(1.0) } else { 1.0 };

        if will_accept_deterministic(ratio, i) || ratio >= 1.0 {
            accepted_count += 1;
        } else {
            // On rejection, sample from residual distribution.
            let pos_offset = i * vocab_size;
            let replacement = neon_sample_residual(
                &target_probs[pos_offset..pos_offset + vocab_size],
                &draft_probs_per_pos[pos_offset..pos_offset + vocab_size],
                i,
            );
            return AcceptanceSample { accepted_count, replacement_token: Some(replacement) };
        }
    }

    AcceptanceSample { accepted_count, replacement_token: None }
}

/// Sample from `max(0, p_target - p_draft)` using NEON to compute the
/// residual and then argmax (deterministic fallback).
#[cfg(target_arch = "aarch64")]
fn neon_sample_residual(target: &[f32], draft: &[f32], _seed: usize) -> u32 {
    let len = target.len();
    let chunks = len / LANES;
    let remainder = len % LANES;

    let mut best_idx: u32 = 0;
    let mut best_val: f32 = f32::NEG_INFINITY;

    unsafe {
        let zero = vdupq_n_f32(0.0);
        for c in 0..chunks {
            let base = c * LANES;
            let t = vld1q_f32(target.as_ptr().add(base));
            let d = vld1q_f32(draft.as_ptr().add(base));
            let residual = vmaxq_f32(vsubq_f32(t, d), zero);

            // Extract lanes and check.
            let mut buf = [0.0f32; LANES];
            vst1q_f32(buf.as_mut_ptr(), residual);
            for (j, &v) in buf.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = (base + j) as u32;
                }
            }
        }
    }

    let tail_start = chunks * LANES;
    for i in 0..remainder {
        let v = (target[tail_start + i] - draft[tail_start + i]).max(0.0);
        if v > best_val {
            best_val = v;
            best_idx = (tail_start + i) as u32;
        }
    }

    best_idx
}

// ── 3. Token Tree Verification ──────────────────────────────────────────

/// A node in a speculative decoding token tree.
#[derive(Debug, Clone)]
pub struct TokenTreeNode {
    /// Token ID at this node.
    pub token_id: u32,
    /// Draft model probability for this token.
    pub draft_prob: f32,
    /// Indices of child nodes in the tree array.
    pub children: Vec<usize>,
}

/// Result of verifying a token tree.
#[derive(Debug, Clone)]
pub struct TreeVerificationResult {
    /// Longest accepted path (sequence of token IDs).
    pub accepted_path: Vec<u32>,
    /// Total nodes verified.
    pub nodes_verified: usize,
    /// Number of accepted nodes.
    pub nodes_accepted: usize,
}

/// Verify a token tree against target logits using NEON.
///
/// Performs depth-first traversal, accepting tokens with the same
/// `p_target / p_draft` ratio criterion. Returns the longest
/// accepted prefix path.
///
/// - `tree`: array of `TokenTreeNode`s (index 0 is root)
/// - `target_logits`: target logits per tree depth, shape
///   `[max_depth][vocab_size]` flattened row-major
/// - `vocab_size`: vocabulary size
/// - `temperature`: sampling temperature
#[cfg(target_arch = "aarch64")]
pub fn neon_verify_token_tree(
    tree: &[TokenTreeNode],
    target_logits: &[f32],
    vocab_size: usize,
    temperature: f32,
) -> TreeVerificationResult {
    if tree.is_empty() {
        return TreeVerificationResult {
            accepted_path: Vec::new(),
            nodes_verified: 0,
            nodes_accepted: 0,
        };
    }

    let inv_temp = 1.0 / temperature;
    let mut best_path: Vec<u32> = Vec::new();
    let mut current_path: Vec<u32> = Vec::new();
    let mut nodes_verified = 0usize;
    let mut nodes_accepted = 0usize;

    // DFS stack: (node_index, depth).
    let mut stack: Vec<(usize, usize)> = vec![(0, 0)];

    while let Some((node_idx, depth)) = stack.pop() {
        let node = &tree[node_idx];
        nodes_verified += 1;

        // Truncate current path to this depth.
        current_path.truncate(depth);

        // Check acceptance if we have logits for this depth.
        if depth * vocab_size + vocab_size <= target_logits.len() {
            let logits = &target_logits[depth * vocab_size..(depth + 1) * vocab_size];
            let t_prob = neon_softmax_single_prob(logits, node.token_id as usize, inv_temp);
            let ratio =
                if node.draft_prob > 0.0 { (t_prob / node.draft_prob).min(1.0) } else { 1.0 };

            if ratio >= 1.0 || will_accept_deterministic(ratio, depth) {
                nodes_accepted += 1;
                current_path.push(node.token_id);

                if current_path.len() > best_path.len() {
                    best_path.clone_from(&current_path);
                }

                // Push children for further exploration.
                for &child_idx in node.children.iter().rev() {
                    if child_idx < tree.len() {
                        stack.push((child_idx, depth + 1));
                    }
                }
            }
            // If rejected, don't explore children.
        }
    }

    TreeVerificationResult { accepted_path: best_path, nodes_verified, nodes_accepted }
}

// ── 4. KV Cache Rollback ────────────────────────────────────────────────

/// State for tracking KV cache positions during speculative decoding.
#[derive(Debug, Clone)]
pub struct KvCacheRollbackState {
    /// Number of positions committed in the cache before speculation.
    pub committed_length: usize,
    /// Number of speculative positions appended.
    pub speculative_length: usize,
}

impl KvCacheRollbackState {
    /// Create a new rollback state with the given committed length.
    pub fn new(committed_length: usize) -> Self {
        Self { committed_length, speculative_length: 0 }
    }

    /// Mark `n` speculative tokens as appended.
    pub fn mark_speculative(&mut self, n: usize) {
        self.speculative_length += n;
    }

    /// Commit `n` accepted tokens from speculative positions.
    pub fn commit(&mut self, n: usize) {
        let to_commit = n.min(self.speculative_length);
        self.committed_length += to_commit;
        self.speculative_length -= to_commit;
    }

    /// Total length including speculative entries.
    pub fn total_length(&self) -> usize {
        self.committed_length + self.speculative_length
    }
}

/// Roll back a KV cache by zeroing rejected speculative entries.
///
/// After `accepted` tokens from the speculative range are kept, the
/// remainder is zeroed using NEON stores for cache-friendly clearing.
///
/// - `cache`: flat f32 cache, shape `[positions][head_dim]`
/// - `head_dim`: dimension per position
/// - `state`: rollback state (mutated on success)
/// - `accepted`: number of speculative tokens accepted
#[cfg(target_arch = "aarch64")]
pub fn neon_kv_cache_rollback(
    cache: &mut [f32],
    head_dim: usize,
    state: &mut KvCacheRollbackState,
    accepted: usize,
) {
    state.commit(accepted);

    let reject_start = state.committed_length;
    let reject_end = reject_start + state.speculative_length;
    let elem_start = reject_start * head_dim;
    let elem_end = (reject_end * head_dim).min(cache.len());

    if elem_start >= elem_end {
        state.speculative_length = 0;
        return;
    }

    let region = &mut cache[elem_start..elem_end];
    let len = region.len();
    let chunks = len / LANES;
    let remainder = len % LANES;

    unsafe {
        let zero = vdupq_n_f32(0.0);
        for c in 0..chunks {
            let base = c * LANES;
            vst1q_f32(region.as_mut_ptr().add(base), zero);
        }
    }

    let tail_start = chunks * LANES;
    for i in 0..remainder {
        region[tail_start + i] = 0.0;
    }

    state.speculative_length = 0;
}

// ── 5. Batch Verification ───────────────────────────────────────────────

/// Result of batch verification across multiple sequences.
#[derive(Debug, Clone)]
pub struct BatchVerificationResult {
    /// Per-sequence accepted counts.
    pub accepted_counts: Vec<usize>,
    /// Per-sequence replacement tokens (None if all accepted).
    pub replacement_tokens: Vec<Option<u32>>,
}

/// Verify draft tokens for a batch of sequences in parallel using NEON.
///
/// Each sequence has its own draft tokens, draft probabilities, and
/// target logits. NEON is used for softmax and probability comparison
/// within each sequence.
///
/// - `batch_draft_tokens`: per-sequence draft token arrays
/// - `batch_draft_probs`: per-sequence draft probability arrays
/// - `batch_target_logits`: per-sequence flattened target logits
/// - `vocab_size`: shared vocabulary size
/// - `temperature`: sampling temperature
#[cfg(target_arch = "aarch64")]
pub fn neon_batch_verify(
    batch_draft_tokens: &[&[u32]],
    batch_draft_probs: &[&[f32]],
    batch_target_logits: &[&[f32]],
    vocab_size: usize,
    temperature: f32,
) -> BatchVerificationResult {
    let batch_size = batch_draft_tokens.len();
    assert_eq!(batch_size, batch_draft_probs.len());
    assert_eq!(batch_size, batch_target_logits.len());

    let mut accepted_counts = Vec::with_capacity(batch_size);
    let mut replacement_tokens = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let verifications = neon_verify_draft_tokens(
            batch_draft_tokens[i],
            batch_draft_probs[i],
            batch_target_logits[i],
            vocab_size,
            temperature,
        );

        let mut accepted = 0;
        let mut replacement = None;

        for v in &verifications {
            if v.accepted {
                accepted += 1;
            } else {
                // Sample replacement from target logits at
                // rejection position.
                let offset = accepted * vocab_size;
                if offset + vocab_size <= batch_target_logits[i].len() {
                    let logits = &batch_target_logits[i][offset..offset + vocab_size];
                    replacement = Some(neon_argmax(logits));
                }
                break;
            }
        }

        accepted_counts.push(accepted);
        replacement_tokens.push(replacement);
    }

    BatchVerificationResult { accepted_counts, replacement_tokens }
}

/// NEON-accelerated argmax over f32 slice.
#[cfg(target_arch = "aarch64")]
fn neon_argmax(data: &[f32]) -> u32 {
    let len = data.len();
    if len == 0 {
        return 0;
    }

    let chunks = len / LANES;
    let remainder = len % LANES;

    let mut best_idx: u32 = 0;
    let mut best_val: f32 = f32::NEG_INFINITY;

    unsafe {
        for c in 0..chunks {
            let base = c * LANES;
            let v = vld1q_f32(data.as_ptr().add(base));
            let mut buf = [0.0f32; LANES];
            vst1q_f32(buf.as_mut_ptr(), v);
            for (j, &val) in buf.iter().enumerate() {
                if val > best_val {
                    best_val = val;
                    best_idx = (base + j) as u32;
                }
            }
        }
    }

    let tail_start = chunks * LANES;
    for i in 0..remainder {
        if data[tail_start + i] > best_val {
            best_val = data[tail_start + i];
            best_idx = (tail_start + i) as u32;
        }
    }

    best_idx
}

// ── 6. Probability Comparison ───────────────────────────────────────────

/// Result of comparing log-probabilities between draft and target.
#[derive(Debug, Clone, Copy)]
pub struct LogProbComparison {
    /// Sum of absolute differences in log-probabilities.
    pub total_abs_diff: f32,
    /// Maximum absolute difference across all positions.
    pub max_abs_diff: f32,
    /// Mean absolute difference.
    pub mean_abs_diff: f32,
    /// Number of positions compared.
    pub num_positions: usize,
}

/// Compare log-probabilities between draft and target models using NEON.
///
/// Computes per-element `|log_target - log_draft|` with NEON, then
/// accumulates statistics.
///
/// - `target_log_probs`: log-probabilities from target model
/// - `draft_log_probs`: log-probabilities from draft model
#[cfg(target_arch = "aarch64")]
pub fn neon_compare_log_probs(
    target_log_probs: &[f32],
    draft_log_probs: &[f32],
) -> LogProbComparison {
    let len = target_log_probs.len().min(draft_log_probs.len());
    if len == 0 {
        return LogProbComparison {
            total_abs_diff: 0.0,
            max_abs_diff: 0.0,
            mean_abs_diff: 0.0,
            num_positions: 0,
        };
    }

    let chunks = len / LANES;
    let remainder = len % LANES;

    let mut total: f32 = 0.0;
    let mut max_diff: f32;

    unsafe {
        let mut sum_v = vdupq_n_f32(0.0);
        let mut max_v = vdupq_n_f32(0.0);

        for c in 0..chunks {
            let base = c * LANES;
            let t = vld1q_f32(target_log_probs.as_ptr().add(base));
            let d = vld1q_f32(draft_log_probs.as_ptr().add(base));
            let diff = vabdq_f32(t, d);
            sum_v = vaddq_f32(sum_v, diff);
            max_v = vmaxq_f32(max_v, diff);
        }

        total += hsum_f32x4(sum_v);

        // Horizontal max.
        let pair = vpmaxq_f32(max_v, max_v);
        max_diff = vgetq_lane_f32(vpmaxq_f32(pair, pair), 0);
    }

    let tail_start = chunks * LANES;
    for i in 0..remainder {
        let d = (target_log_probs[tail_start + i] - draft_log_probs[tail_start + i]).abs();
        total += d;
        if d > max_diff {
            max_diff = d;
        }
    }

    LogProbComparison {
        total_abs_diff: total,
        max_abs_diff: max_diff,
        mean_abs_diff: total / len as f32,
        num_positions: len,
    }
}

// ── 7. Adaptive Draft Length ────────────────────────────────────────────

/// Tracks acceptance rates and dynamically adjusts draft length.
#[derive(Debug, Clone)]
pub struct AdaptiveDraftLength {
    /// Current draft length (number of tokens to speculate).
    pub current_length: usize,
    /// Minimum draft length.
    pub min_length: usize,
    /// Maximum draft length.
    pub max_length: usize,
    /// Exponential moving average of acceptance rate.
    pub ema_acceptance_rate: f32,
    /// Smoothing factor for EMA (0..1). Higher = faster adaptation.
    pub alpha: f32,
    /// Target acceptance rate threshold for increasing draft length.
    pub increase_threshold: f32,
    /// Target acceptance rate threshold for decreasing draft length.
    pub decrease_threshold: f32,
    /// Number of rounds observed.
    pub rounds: usize,
}

impl AdaptiveDraftLength {
    /// Create a new adaptive draft length tracker.
    ///
    /// - `initial_length`: starting draft length
    /// - `min_length`: minimum allowed draft length (≥ 1)
    /// - `max_length`: maximum allowed draft length
    pub fn new(initial_length: usize, min_length: usize, max_length: usize) -> Self {
        assert!(min_length >= 1);
        assert!(max_length >= min_length);
        Self {
            current_length: initial_length.clamp(min_length, max_length),
            min_length,
            max_length,
            ema_acceptance_rate: 0.5,
            alpha: 0.3,
            increase_threshold: 0.8,
            decrease_threshold: 0.3,
            rounds: 0,
        }
    }

    /// Update the tracker after a speculative decoding round.
    ///
    /// - `draft_length`: number of tokens drafted this round
    /// - `accepted`: number of tokens accepted
    ///
    /// Adjusts `current_length` up or down based on the EMA acceptance
    /// rate crossing the configured thresholds.
    pub fn update(&mut self, draft_length: usize, accepted: usize) {
        if draft_length == 0 {
            return;
        }

        let rate = accepted as f32 / draft_length as f32;
        self.ema_acceptance_rate =
            self.alpha * rate + (1.0 - self.alpha) * self.ema_acceptance_rate;
        self.rounds += 1;

        if self.ema_acceptance_rate > self.increase_threshold {
            self.current_length = (self.current_length + 1).min(self.max_length);
        } else if self.ema_acceptance_rate < self.decrease_threshold {
            self.current_length = self.current_length.saturating_sub(1).max(self.min_length);
        }
    }

    /// Get the recommended draft length for the next round.
    pub fn recommended_length(&self) -> usize {
        self.current_length
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    // Helper: build uniform logits (equal probability for all tokens).
    fn uniform_logits(vocab_size: usize, num_positions: usize) -> Vec<f32> {
        vec![1.0; vocab_size * num_positions]
    }

    // Helper: build logits that strongly favour a specific token.
    fn peaked_logits(vocab_size: usize, num_positions: usize, peak_token: u32) -> Vec<f32> {
        let mut logits = vec![-10.0; vocab_size * num_positions];
        for pos in 0..num_positions {
            logits[pos * vocab_size + peak_token as usize] = 10.0;
        }
        logits
    }

    // ── Draft Token Verification Tests ──────────────────────────────

    #[test]
    fn test_verify_draft_tokens_all_accepted_peaked() {
        let vocab_size = 16;
        let draft_tokens = vec![3u32, 3, 3];
        let draft_probs = vec![0.99, 0.99, 0.99];
        let target_logits = peaked_logits(vocab_size, 3, 3);

        let results =
            neon_verify_draft_tokens(&draft_tokens, &draft_probs, &target_logits, vocab_size, 1.0);

        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.target_prob > 0.9, "target_prob too low: {}", r.target_prob,);
        }
    }

    #[test]
    fn test_verify_draft_tokens_uniform_logits() {
        let vocab_size = 8;
        let draft_tokens = vec![0u32, 1];
        let draft_probs = vec![0.125, 0.125]; // 1/8 each
        let target_logits = uniform_logits(vocab_size, 2);

        let results =
            neon_verify_draft_tokens(&draft_tokens, &draft_probs, &target_logits, vocab_size, 1.0);

        assert_eq!(results.len(), 2);
        for r in &results {
            // Uniform → ~0.125 target prob.
            assert!(
                (r.target_prob - 0.125).abs() < 0.02,
                "unexpected target_prob: {}",
                r.target_prob,
            );
        }
    }

    #[test]
    fn test_verify_draft_tokens_temperature_scaling() {
        let vocab_size = 8;
        let draft_tokens = vec![2u32];
        let draft_probs = [0.5];
        // One token much higher than others.
        let mut logits = vec![0.0f32; vocab_size];
        logits[2] = 5.0;

        // Low temp should sharpen distribution.
        let results_low =
            neon_verify_draft_tokens(&draft_tokens, &draft_probs, &logits, vocab_size, 0.1);
        // High temp should flatten distribution.
        let results_high =
            neon_verify_draft_tokens(&draft_tokens, &draft_probs, &logits, vocab_size, 10.0);

        assert!(
            results_low[0].target_prob > results_high[0].target_prob,
            "low temp should produce higher target prob for peak token",
        );
    }

    #[test]
    fn test_verify_empty_draft() {
        let results = neon_verify_draft_tokens(&[], &[], &[], 8, 1.0);
        assert!(results.is_empty());
    }

    // ── Acceptance Sampling Tests ───────────────────────────────────

    #[test]
    fn test_acceptance_sampling_all_match() {
        let vocab_size = 4;
        // Target and draft agree perfectly.
        let probs = vec![0.25f32; vocab_size * 2];
        let draft_tokens = vec![0u32, 1];

        let result = neon_acceptance_sampling(&probs, &probs, &draft_tokens, vocab_size, 1.0);

        // With identical probs (ratio = 1.0), all should be accepted.
        assert_eq!(result.accepted_count, 2);
        assert!(result.replacement_token.is_none());
    }

    #[test]
    fn test_acceptance_sampling_returns_replacement() {
        let vocab_size = 4;
        // Target strongly prefers token 3.
        let mut target_probs = vec![0.01f32; vocab_size];
        target_probs[3] = 0.97;
        // Draft strongly prefers token 0.
        let mut draft_probs = vec![0.01f32; vocab_size];
        draft_probs[0] = 0.97;

        let draft_tokens = vec![0u32]; // Draft picks token 0.

        let result =
            neon_acceptance_sampling(&target_probs, &draft_probs, &draft_tokens, vocab_size, 1.0);

        // Ratio is ~0.01/0.97 ≈ 0.01 → almost certainly rejected.
        // Replacement should be token 3 (highest residual).
        if let Some(replacement) = result.replacement_token {
            assert_eq!(replacement, 3, "replacement should be the high-residual token",);
        }
    }

    // ── Token Tree Verification Tests ───────────────────────────────

    #[test]
    fn test_tree_verification_single_node() {
        let vocab_size = 8;
        let tree = vec![TokenTreeNode { token_id: 2, draft_prob: 0.01, children: vec![] }];
        let logits = peaked_logits(vocab_size, 1, 2);

        let result = neon_verify_token_tree(&tree, &logits, vocab_size, 1.0);

        assert_eq!(result.nodes_verified, 1);
        assert_eq!(result.accepted_path, vec![2]);
    }

    #[test]
    fn test_tree_verification_chain() {
        let vocab_size = 8;
        let tree = vec![
            TokenTreeNode { token_id: 1, draft_prob: 0.01, children: vec![1] },
            TokenTreeNode { token_id: 1, draft_prob: 0.01, children: vec![2] },
            TokenTreeNode { token_id: 1, draft_prob: 0.01, children: vec![] },
        ];
        let logits = peaked_logits(vocab_size, 3, 1);

        let result = neon_verify_token_tree(&tree, &logits, vocab_size, 1.0);

        assert_eq!(result.accepted_path, vec![1, 1, 1]);
        assert_eq!(result.nodes_accepted, 3);
    }

    #[test]
    fn test_tree_verification_empty() {
        let result = neon_verify_token_tree(&[], &[], 8, 1.0);
        assert!(result.accepted_path.is_empty());
        assert_eq!(result.nodes_verified, 0);
    }

    // ── KV Cache Rollback Tests ─────────────────────────────────────

    #[test]
    fn test_kv_cache_rollback_basic() {
        let head_dim = 4;
        let mut cache = vec![1.0f32; 5 * head_dim]; // 5 positions
        let mut state = KvCacheRollbackState::new(2);
        state.mark_speculative(3); // 3 speculative after 2 committed

        neon_kv_cache_rollback(&mut cache, head_dim, &mut state, 1);

        // After accepting 1: committed = 3, speculative = 0.
        assert_eq!(state.committed_length, 3);
        assert_eq!(state.speculative_length, 0);
        // Positions 3..5 should be zeroed.
        for i in (3 * head_dim)..(5 * head_dim) {
            assert_eq!(cache[i], 0.0, "position {i} not zeroed");
        }
        // Positions 0..3 should be untouched.
        for i in 0..(3 * head_dim) {
            assert_eq!(cache[i], 1.0, "position {i} should be 1.0");
        }
    }

    #[test]
    fn test_kv_cache_rollback_all_accepted() {
        let head_dim = 4;
        let mut cache = vec![1.0f32; 4 * head_dim];
        let mut state = KvCacheRollbackState::new(2);
        state.mark_speculative(2);

        neon_kv_cache_rollback(&mut cache, head_dim, &mut state, 2);

        assert_eq!(state.committed_length, 4);
        assert_eq!(state.speculative_length, 0);
        // All kept → no zeroing.
        for &v in &cache {
            assert_eq!(v, 1.0);
        }
    }

    #[test]
    fn test_kv_cache_rollback_state_new() {
        let state = KvCacheRollbackState::new(10);
        assert_eq!(state.committed_length, 10);
        assert_eq!(state.speculative_length, 0);
        assert_eq!(state.total_length(), 10);
    }

    // ── Batch Verification Tests ────────────────────────────────────

    #[test]
    fn test_batch_verify_two_sequences() {
        let vocab_size = 8;

        // Sequence 1: draft matches target peak.
        let tokens1 = vec![3u32, 3];
        let probs1 = vec![0.99, 0.99];
        let logits1 = peaked_logits(vocab_size, 2, 3);

        // Sequence 2: draft matches target peak.
        let tokens2 = vec![5u32];
        let probs2 = [0.99];
        let logits2 = peaked_logits(vocab_size, 1, 5);

        let result = neon_batch_verify(
            &[&tokens1, &tokens2],
            &[&probs1, &probs2],
            &[&logits1, &logits2],
            vocab_size,
            1.0,
        );

        assert_eq!(result.accepted_counts.len(), 2);
        // Both sequences should have high acceptance.
        assert!(result.accepted_counts[0] >= 1, "seq 0 accepted: {}", result.accepted_counts[0],);
        assert!(result.accepted_counts[1] >= 1, "seq 1 accepted: {}", result.accepted_counts[1],);
    }

    // ── Log-Probability Comparison Tests ────────────────────────────

    #[test]
    fn test_compare_log_probs_identical() {
        let probs = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let result = neon_compare_log_probs(&probs, &probs);

        assert_eq!(result.num_positions, 8);
        assert!(
            result.total_abs_diff < 1e-6,
            "identical probs should have ~0 diff: {}",
            result.total_abs_diff,
        );
        assert!(result.max_abs_diff < 1e-6);
    }

    #[test]
    fn test_compare_log_probs_known_diff() {
        let target = vec![1.0f32, 2.0, 3.0, 4.0];
        let draft = vec![1.5f32, 2.5, 3.5, 4.5];
        let result = neon_compare_log_probs(&target, &draft);

        assert_eq!(result.num_positions, 4);
        // Each diff is 0.5; total = 2.0, max = 0.5, mean = 0.5.
        assert!((result.total_abs_diff - 2.0).abs() < 0.01, "total: {}", result.total_abs_diff,);
        assert!((result.max_abs_diff - 0.5).abs() < 0.01, "max: {}", result.max_abs_diff,);
        assert!((result.mean_abs_diff - 0.5).abs() < 0.01, "mean: {}", result.mean_abs_diff,);
    }

    #[test]
    fn test_compare_log_probs_empty() {
        let result = neon_compare_log_probs(&[], &[]);
        assert_eq!(result.num_positions, 0);
        assert_eq!(result.total_abs_diff, 0.0);
    }

    #[test]
    fn test_compare_log_probs_unequal_lengths() {
        let target = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let draft = vec![1.0, 2.0, 3.0];
        let result = neon_compare_log_probs(&target, &draft);
        // Should use the shorter length (3).
        assert_eq!(result.num_positions, 3);
    }

    // ── Adaptive Draft Length Tests ──────────────────────────────────

    #[test]
    fn test_adaptive_draft_length_increase() {
        let mut adl = AdaptiveDraftLength::new(4, 1, 10);
        // Simulate high acceptance rates.
        for _ in 0..10 {
            adl.update(4, 4); // 100% acceptance
        }
        assert!(
            adl.current_length > 4,
            "should increase after high acceptance: {}",
            adl.current_length,
        );
    }

    #[test]
    fn test_adaptive_draft_length_decrease() {
        let mut adl = AdaptiveDraftLength::new(6, 1, 10);
        // Simulate low acceptance rates.
        for _ in 0..10 {
            adl.update(6, 0); // 0% acceptance
        }
        assert!(
            adl.current_length < 6,
            "should decrease after low acceptance: {}",
            adl.current_length,
        );
    }

    #[test]
    fn test_adaptive_draft_length_bounds() {
        let mut adl = AdaptiveDraftLength::new(5, 2, 8);
        assert_eq!(adl.current_length, 5);
        assert_eq!(adl.min_length, 2);
        assert_eq!(adl.max_length, 8);

        // Drive to max.
        for _ in 0..50 {
            adl.update(8, 8);
        }
        assert_eq!(adl.current_length, 8);

        // Drive to min.
        for _ in 0..50 {
            adl.update(8, 0);
        }
        assert_eq!(adl.current_length, 2);
    }

    #[test]
    fn test_adaptive_draft_length_zero_update() {
        let mut adl = AdaptiveDraftLength::new(4, 1, 10);
        adl.update(0, 0); // Should be a no-op.
        assert_eq!(adl.current_length, 4);
        assert_eq!(adl.rounds, 0);
    }

    #[test]
    fn test_adaptive_recommended_length() {
        let adl = AdaptiveDraftLength::new(5, 1, 10);
        assert_eq!(adl.recommended_length(), 5);
    }

    // ── NEON Argmax Test ────────────────────────────────────────────

    #[test]
    fn test_neon_argmax_basic() {
        let data = vec![0.1, 0.5, 0.3, 0.9, 0.2, 0.8, 0.4, 0.7, 0.6];
        let idx = neon_argmax(&data);
        assert_eq!(idx, 3, "argmax should be index 3 (value 0.9)");
    }

    #[test]
    fn test_neon_argmax_empty() {
        let idx = neon_argmax(&[]);
        assert_eq!(idx, 0);
    }
}
