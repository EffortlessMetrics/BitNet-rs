//! ARM NEON optimized speculative decoding v2 kernel for Apple Silicon.
//!
//! Provides NEON-accelerated speculative decoding operations for aarch64:
//! - `draft_token_verify` — vectorized comparison of draft tokens against target logits
//! - `acceptance_probability` — compute acceptance probability with NEON exp/log approximations
//! - `parallel_draft_scoring` — score multiple draft tokens simultaneously
//! - `rejection_sampling` — NEON-optimized rejection sampling for draft acceptance
//! - `token_tree_verify` — verify token trees (multi-path speculative decoding)
//! - `kl_divergence_check` — check KL divergence between draft and target distributions
//!
//! Each function has a NEON fast-path and a scalar fallback.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Lane count for `float32x4_t` NEON vectors.
const LANES: usize = 4;

// ── Fast math approximations ────────────────────────────────────────────

/// Scalar fast exp approximation (degree-4 Cody–Waite polynomial).
/// Maximum relative error ≈ 2 × 10⁻⁴ for |x| ≤ 20.
#[inline(always)]
fn fast_exp_scalar(x: f32) -> f32 {
    let x = x.clamp(-88.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E).round();
    let r = x - n * std::f32::consts::LN_2;
    let poly = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6.0 + r * (1.0 / 24.0))));
    poly * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// Scalar fast natural logarithm approximation.
/// Uses the identity `ln(x) = (exponent - 127) * ln(2) + ln(mantissa)` with a
/// polynomial for `ln(mantissa)` on [1, 2).
#[inline(always)]
fn fast_ln_scalar(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }
    let bits = x.to_bits() as i32;
    let exponent = ((bits >> 23) & 0xFF) - 127;
    let mantissa = f32::from_bits((bits & 0x007F_FFFF) as u32 | 0x3F80_0000);
    // Polynomial approximation for ln(m) on [1, 2).
    let m = mantissa - 1.0;
    let ln_m = m * (1.0 + m * (-0.5 + m * (1.0 / 3.0 + m * (-1.0 / 4.0))));
    (exponent as f32) * std::f32::consts::LN_2 + ln_m
}

/// NEON vectorised fast exp for four lanes.
///
/// # Safety
/// Requires `aarch64` target with NEON.
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

/// NEON vectorised fast natural log for four lanes.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn fast_ln_neon(x: float32x4_t) -> float32x4_t {
    let bits = vreinterpretq_s32_f32(x);
    let exponent = vsubq_s32(
        vandq_s32(vshrq_n_s32(bits, 23), vdupq_n_s32(0xFF)),
        vdupq_n_s32(127),
    );
    let mantissa_bits = vorrq_s32(
        vandq_s32(bits, vdupq_n_s32(0x007F_FFFF)),
        vdupq_n_s32(0x3F80_0000_u32 as i32),
    );
    let mantissa = vreinterpretq_f32_s32(mantissa_bits);
    let one = vdupq_n_f32(1.0);
    let m = vsubq_f32(mantissa, one);

    // Polynomial: m * (1 + m * (-0.5 + m * (1/3 + m * (-1/4))))
    let c4 = vdupq_n_f32(-0.25);
    let c3 = vdupq_n_f32(1.0 / 3.0);
    let c2 = vdupq_n_f32(-0.5);

    let p = vfmaq_f32(c3, m, c4);
    let p = vfmaq_f32(c2, m, p);
    let p = vfmaq_f32(one, m, p);
    let ln_m = vmulq_f32(m, p);

    let ln2 = vdupq_n_f32(std::f32::consts::LN_2);
    let exp_f = vcvtq_f32_s32(exponent);
    vfmaq_f32(ln_m, exp_f, ln2)
}

// ── 1. draft_token_verify ───────────────────────────────────────────────

/// Result of draft token verification.
#[derive(Debug, Clone, PartialEq)]
pub struct DraftVerifyResult {
    /// Number of accepted draft tokens (contiguous from start).
    pub accepted_count: usize,
    /// Per-token acceptance flags.
    pub accepted: Vec<bool>,
}

/// NEON-accelerated draft token verification.
///
/// Compares each draft token's probability against the corresponding target
/// probability. A draft token at position `i` is accepted when
/// `target_probs[draft_tokens[i]] >= draft_probs[i]`.
///
/// # Safety
/// Requires `aarch64` target with NEON. `draft_probs` and `target_probs_per_token`
/// must have length `draft_tokens.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn draft_token_verify_neon(
    draft_probs: &[f32],
    target_probs_per_token: &[f32],
) -> Vec<bool> {
    let len = draft_probs.len();
    let mut accepted = vec![false; len];
    let chunks = len / LANES;
    let remainder = len % LANES;

    let dp = draft_probs.as_ptr();
    let tp = target_probs_per_token.as_ptr();

    for i in 0..chunks {
        let offset = i * LANES;
        let d = vld1q_f32(dp.add(offset));
        let t = vld1q_f32(tp.add(offset));
        let cmp = vcgeq_f32(t, d);
        // Extract comparison results per lane.
        let mask: [u32; 4] = std::mem::transmute(cmp);
        for j in 0..LANES {
            accepted[offset + j] = mask[j] != 0;
        }
    }

    for i in 0..remainder {
        let idx = chunks * LANES + i;
        accepted[idx] = target_probs_per_token[idx] >= draft_probs[idx];
    }

    accepted
}

fn draft_token_verify_scalar(
    draft_probs: &[f32],
    target_probs_per_token: &[f32],
) -> Vec<bool> {
    draft_probs
        .iter()
        .zip(target_probs_per_token.iter())
        .map(|(&d, &t)| t >= d)
        .collect()
}

/// Verify draft tokens against target model logits.
///
/// For each draft token, checks whether the target probability meets or exceeds
/// the draft probability. Returns how many contiguous tokens from the start
/// were accepted and per-token flags.
///
/// `draft_tokens` contains token ids, `draft_probs` contains the draft model
/// probability for each drafted token, and `target_probs_per_token` contains
/// the target model probability for the same token at the same position.
pub fn draft_token_verify(
    draft_tokens: &[u32],
    draft_probs: &[f32],
    target_probs_per_token: &[f32],
) -> DraftVerifyResult {
    let len = draft_tokens.len();
    assert_eq!(draft_probs.len(), len);
    assert_eq!(target_probs_per_token.len(), len);

    if len == 0 {
        return DraftVerifyResult {
            accepted_count: 0,
            accepted: vec![],
        };
    }

    let accepted;
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            accepted =
                unsafe { draft_token_verify_neon(draft_probs, target_probs_per_token) };
        } else {
            accepted = draft_token_verify_scalar(draft_probs, target_probs_per_token);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        accepted = draft_token_verify_scalar(draft_probs, target_probs_per_token);
    }

    let accepted_count = accepted.iter().take_while(|&&a| a).count();
    DraftVerifyResult {
        accepted_count,
        accepted,
    }
}

// ── 2. acceptance_probability ───────────────────────────────────────────

/// NEON-accelerated acceptance probability computation.
///
/// Computes `min(1, target_prob / draft_prob)` for each token pair using
/// NEON division and min operations.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn acceptance_probability_neon(
    draft_probs: &[f32],
    target_probs: &[f32],
) -> Vec<f32> {
    let len = draft_probs.len();
    let mut result = vec![0.0f32; len];
    let chunks = len / LANES;
    let remainder = len % LANES;

    let dp = draft_probs.as_ptr();
    let tp = target_probs.as_ptr();
    let rp = result.as_mut_ptr();
    let one = vdupq_n_f32(1.0);
    let eps = vdupq_n_f32(1e-10);

    for i in 0..chunks {
        let offset = i * LANES;
        let d = vld1q_f32(dp.add(offset));
        let t = vld1q_f32(tp.add(offset));
        // Avoid division by zero.
        let d_safe = vmaxq_f32(d, eps);
        let ratio = vdivq_f32(t, d_safe);
        let clamped = vminq_f32(ratio, one);
        let clamped = vmaxq_f32(clamped, vdupq_n_f32(0.0));
        vst1q_f32(rp.add(offset), clamped);
    }

    for i in 0..remainder {
        let idx = chunks * LANES + i;
        let d = draft_probs[idx].max(1e-10);
        result[idx] = (target_probs[idx] / d).clamp(0.0, 1.0);
    }

    result
}

fn acceptance_probability_scalar(
    draft_probs: &[f32],
    target_probs: &[f32],
) -> Vec<f32> {
    draft_probs
        .iter()
        .zip(target_probs.iter())
        .map(|(&d, &t)| {
            let d = d.max(1e-10);
            (t / d).clamp(0.0, 1.0)
        })
        .collect()
}

/// Compute acceptance probabilities for speculative decoding.
///
/// Returns `min(1, target_prob / draft_prob)` for each position, clamped to
/// `[0, 1]`.
pub fn acceptance_probability(
    draft_probs: &[f32],
    target_probs: &[f32],
) -> Vec<f32> {
    let len = draft_probs.len();
    assert_eq!(target_probs.len(), len);

    if len == 0 {
        return vec![];
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe {
                acceptance_probability_neon(draft_probs, target_probs)
            };
        }
    }
    acceptance_probability_scalar(draft_probs, target_probs)
}

// ── 3. parallel_draft_scoring ───────────────────────────────────────────

/// Score for a single draft candidate.
#[derive(Debug, Clone)]
pub struct DraftScore {
    /// Token id.
    pub token_id: u32,
    /// Log-probability from the draft model.
    pub log_prob: f32,
}

/// NEON-accelerated log-softmax for scoring draft tokens.
///
/// Computes `logits[token_id] - log(sum(exp(logits)))` for each candidate.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn parallel_draft_scoring_neon(
    logits: &[f32],
    candidate_ids: &[u32],
) -> Vec<DraftScore> {
    let len = logits.len();
    if len == 0 || candidate_ids.is_empty() {
        return candidate_ids
            .iter()
            .map(|&id| DraftScore {
                token_id: id,
                log_prob: f32::NEG_INFINITY,
            })
            .collect();
    }

    // Find max for numerical stability.
    let chunks = len / LANES;
    let remainder = len % LANES;
    let ptr = logits.as_ptr();

    let mut max_vec = vdupq_n_f32(f32::NEG_INFINITY);
    for i in 0..chunks {
        let v = vld1q_f32(ptr.add(i * LANES));
        max_vec = vmaxq_f32(max_vec, v);
    }
    let mut max_val = vmaxvq_f32(max_vec);
    for i in 0..remainder {
        let val = *ptr.add(chunks * LANES + i);
        if val > max_val {
            max_val = val;
        }
    }

    // Compute sum of exp(logits - max).
    let max_v = vdupq_n_f32(max_val);
    let mut sum_vec = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let v = vld1q_f32(ptr.add(i * LANES));
        let shifted = vsubq_f32(v, max_v);
        let e = fast_exp_neon(shifted);
        sum_vec = vaddq_f32(sum_vec, e);
    }
    let mut sum_exp = vaddvq_f32(sum_vec);
    for i in 0..remainder {
        let val = *ptr.add(chunks * LANES + i);
        sum_exp += fast_exp_scalar(val - max_val);
    }

    let log_sum_exp = max_val + fast_ln_scalar(sum_exp);

    candidate_ids
        .iter()
        .map(|&id| {
            let log_prob = if (id as usize) < len {
                logits[id as usize] - log_sum_exp
            } else {
                f32::NEG_INFINITY
            };
            DraftScore {
                token_id: id,
                log_prob,
            }
        })
        .collect()
}

fn parallel_draft_scoring_scalar(
    logits: &[f32],
    candidate_ids: &[u32],
) -> Vec<DraftScore> {
    if logits.is_empty() || candidate_ids.is_empty() {
        return candidate_ids
            .iter()
            .map(|&id| DraftScore {
                token_id: id,
                log_prob: f32::NEG_INFINITY,
            })
            .collect();
    }

    let max_val = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits
        .iter()
        .map(|&l| fast_exp_scalar(l - max_val))
        .sum();
    let log_sum_exp = max_val + fast_ln_scalar(sum_exp);

    candidate_ids
        .iter()
        .map(|&id| {
            let log_prob = if (id as usize) < logits.len() {
                logits[id as usize] - log_sum_exp
            } else {
                f32::NEG_INFINITY
            };
            DraftScore {
                token_id: id,
                log_prob,
            }
        })
        .collect()
}

/// Score multiple draft candidate tokens in parallel against a logits vector.
///
/// Computes the log-softmax of the logits and returns the log-probability for
/// each candidate token id.
pub fn parallel_draft_scoring(
    logits: &[f32],
    candidate_ids: &[u32],
) -> Vec<DraftScore> {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe {
                parallel_draft_scoring_neon(logits, candidate_ids)
            };
        }
    }
    parallel_draft_scoring_scalar(logits, candidate_ids)
}

// ── 4. rejection_sampling ───────────────────────────────────────────────

/// Result of rejection sampling.
#[derive(Debug, Clone)]
pub struct RejectionSampleResult {
    /// Number of accepted tokens (contiguous from start).
    pub accepted_count: usize,
    /// Per-position acceptance flags.
    pub accepted: Vec<bool>,
    /// Adjusted probability distribution for the first rejected position,
    /// or `None` if all accepted.
    pub adjusted_distribution: Option<Vec<f32>>,
}

/// NEON-accelerated rejection sampling.
///
/// For each position, accepts the draft token with probability
/// `min(1, target_prob / draft_prob)`. Uses deterministic random values
/// from `random_values` for reproducibility.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn rejection_sampling_neon(
    draft_probs: &[f32],
    target_probs: &[f32],
    random_values: &[f32],
) -> RejectionSampleResult {
    let len = draft_probs.len();
    let mut accepted = vec![false; len];
    let chunks = len / LANES;
    let remainder = len % LANES;

    let dp = draft_probs.as_ptr();
    let tp = target_probs.as_ptr();
    let rp = random_values.as_ptr();
    let one = vdupq_n_f32(1.0);
    let eps = vdupq_n_f32(1e-10);
    let zero = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let offset = i * LANES;
        let d = vld1q_f32(dp.add(offset));
        let t = vld1q_f32(tp.add(offset));
        let r = vld1q_f32(rp.add(offset));

        let d_safe = vmaxq_f32(d, eps);
        let ratio = vdivq_f32(t, d_safe);
        let prob = vminq_f32(ratio, one);
        let prob = vmaxq_f32(prob, zero);

        // Accept if random < prob.
        let cmp = vcgtq_f32(prob, r);
        let mask: [u32; 4] = std::mem::transmute(cmp);
        for j in 0..LANES {
            accepted[offset + j] = mask[j] != 0;
        }
    }

    for i in 0..remainder {
        let idx = chunks * LANES + i;
        let d = draft_probs[idx].max(1e-10);
        let prob = (target_probs[idx] / d).clamp(0.0, 1.0);
        accepted[idx] = random_values[idx] < prob;
    }

    let accepted_count = accepted.iter().take_while(|&&a| a).count();

    // Compute adjusted distribution at first rejection point.
    let adjusted_distribution = if accepted_count < len {
        let adj: Vec<f32> = draft_probs
            .iter()
            .zip(target_probs.iter())
            .map(|(&d, &t)| (t - d).max(0.0))
            .collect();
        let sum: f32 = adj.iter().sum();
        if sum > 0.0 {
            Some(adj.iter().map(|&v| v / sum).collect())
        } else {
            Some(target_probs.to_vec())
        }
    } else {
        None
    };

    RejectionSampleResult {
        accepted_count,
        accepted,
        adjusted_distribution,
    }
}

fn rejection_sampling_scalar(
    draft_probs: &[f32],
    target_probs: &[f32],
    random_values: &[f32],
) -> RejectionSampleResult {
    let len = draft_probs.len();
    let mut accepted = vec![false; len];

    for i in 0..len {
        let d = draft_probs[i].max(1e-10);
        let prob = (target_probs[i] / d).clamp(0.0, 1.0);
        accepted[i] = random_values[i] < prob;
    }

    let accepted_count = accepted.iter().take_while(|&&a| a).count();

    let adjusted_distribution = if accepted_count < len {
        let adj: Vec<f32> = draft_probs
            .iter()
            .zip(target_probs.iter())
            .map(|(&d, &t)| (t - d).max(0.0))
            .collect();
        let sum: f32 = adj.iter().sum();
        if sum > 0.0 {
            Some(adj.iter().map(|&v| v / sum).collect())
        } else {
            Some(target_probs.to_vec())
        }
    } else {
        None
    };

    RejectionSampleResult {
        accepted_count,
        accepted,
        adjusted_distribution,
    }
}

/// Perform rejection sampling for speculative decoding.
///
/// Uses the standard speculative decoding acceptance rule: accept token at
/// position `i` with probability `min(1, target / draft)`. `random_values`
/// should contain uniform `[0, 1)` random numbers for deterministic testing.
pub fn rejection_sampling(
    draft_probs: &[f32],
    target_probs: &[f32],
    random_values: &[f32],
) -> RejectionSampleResult {
    let len = draft_probs.len();
    assert_eq!(target_probs.len(), len);
    assert_eq!(random_values.len(), len);

    if len == 0 {
        return RejectionSampleResult {
            accepted_count: 0,
            accepted: vec![],
            adjusted_distribution: None,
        };
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe {
                rejection_sampling_neon(draft_probs, target_probs, random_values)
            };
        }
    }
    rejection_sampling_scalar(draft_probs, target_probs, random_values)
}

// ── 5. token_tree_verify ────────────────────────────────────────────────

/// A node in a speculative decoding token tree.
#[derive(Debug, Clone)]
pub struct TokenTreeNode {
    /// Token id at this node.
    pub token_id: u32,
    /// Draft model probability for this token.
    pub draft_prob: f32,
    /// Target model probability for this token.
    pub target_prob: f32,
    /// Indices of child nodes in the flat tree array.
    pub children: Vec<usize>,
}

/// Result of token tree verification.
#[derive(Debug, Clone)]
pub struct TokenTreeResult {
    /// Best accepted path (sequence of token ids).
    pub best_path: Vec<u32>,
    /// Acceptance flags per node (indexed by node position in the tree array).
    pub node_accepted: Vec<bool>,
    /// Total number of accepted nodes.
    pub total_accepted: usize,
}

/// NEON-accelerated batch comparison for tree node probabilities.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn batch_compare_probs_neon(
    draft_probs: &[f32],
    target_probs: &[f32],
) -> Vec<bool> {
    let len = draft_probs.len();
    let mut result = vec![false; len];
    let chunks = len / LANES;
    let remainder = len % LANES;

    let dp = draft_probs.as_ptr();
    let tp = target_probs.as_ptr();

    for i in 0..chunks {
        let offset = i * LANES;
        let d = vld1q_f32(dp.add(offset));
        let t = vld1q_f32(tp.add(offset));
        let cmp = vcgeq_f32(t, d);
        let mask: [u32; 4] = std::mem::transmute(cmp);
        for j in 0..LANES {
            result[offset + j] = mask[j] != 0;
        }
    }

    for i in 0..remainder {
        let idx = chunks * LANES + i;
        result[idx] = target_probs[idx] >= draft_probs[idx];
    }

    result
}

fn batch_compare_probs_scalar(
    draft_probs: &[f32],
    target_probs: &[f32],
) -> Vec<bool> {
    draft_probs
        .iter()
        .zip(target_probs.iter())
        .map(|(&d, &t)| t >= d)
        .collect()
}

/// Verify a token tree for multi-path speculative decoding.
///
/// Traverses the tree depth-first, accepting nodes where
/// `target_prob >= draft_prob`. Returns the longest accepted path and
/// per-node acceptance flags.
pub fn token_tree_verify(tree: &[TokenTreeNode]) -> TokenTreeResult {
    if tree.is_empty() {
        return TokenTreeResult {
            best_path: vec![],
            node_accepted: vec![],
            total_accepted: 0,
        };
    }

    // Batch compare all node probabilities.
    let draft_probs: Vec<f32> = tree.iter().map(|n| n.draft_prob).collect();
    let target_probs: Vec<f32> = tree.iter().map(|n| n.target_prob).collect();

    let node_accepted;
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            node_accepted = unsafe {
                batch_compare_probs_neon(&draft_probs, &target_probs)
            };
        } else {
            node_accepted =
                batch_compare_probs_scalar(&draft_probs, &target_probs);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        node_accepted =
            batch_compare_probs_scalar(&draft_probs, &target_probs);
    }

    let total_accepted = node_accepted.iter().filter(|&&a| a).count();

    // DFS to find the longest accepted path.
    let best_path = find_best_path(tree, &node_accepted, 0);

    TokenTreeResult {
        best_path,
        node_accepted,
        total_accepted,
    }
}

/// DFS helper to find the longest accepted path starting from `node_idx`.
fn find_best_path(
    tree: &[TokenTreeNode],
    accepted: &[bool],
    node_idx: usize,
) -> Vec<u32> {
    if node_idx >= tree.len() || !accepted[node_idx] {
        return vec![];
    }

    let mut best = vec![tree[node_idx].token_id];

    let mut longest_child_path: Vec<u32> = vec![];
    for &child_idx in &tree[node_idx].children {
        let child_path = find_best_path(tree, accepted, child_idx);
        if child_path.len() > longest_child_path.len() {
            longest_child_path = child_path;
        }
    }

    best.extend(longest_child_path);
    best
}

// ── 6. kl_divergence_check ──────────────────────────────────────────────

/// Result of KL divergence check.
#[derive(Debug, Clone)]
pub struct KlDivergenceResult {
    /// KL(target || draft) divergence value.
    pub kl_divergence: f32,
    /// Whether the divergence is below the threshold.
    pub within_threshold: bool,
}

/// NEON-accelerated KL divergence computation.
///
/// Computes `sum(target * ln(target / draft))` with NEON log approximation.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn kl_divergence_neon(
    target_dist: &[f32],
    draft_dist: &[f32],
) -> f32 {
    let len = target_dist.len();
    if len == 0 {
        return 0.0;
    }

    let chunks = len / LANES;
    let remainder = len % LANES;

    let tp = target_dist.as_ptr();
    let dp = draft_dist.as_ptr();
    let eps = vdupq_n_f32(1e-10);
    let mut acc = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let offset = i * LANES;
        let t = vld1q_f32(tp.add(offset));
        let d = vld1q_f32(dp.add(offset));

        let t_safe = vmaxq_f32(t, eps);
        let d_safe = vmaxq_f32(d, eps);

        let ln_t = fast_ln_neon(t_safe);
        let ln_d = fast_ln_neon(d_safe);
        let ln_ratio = vsubq_f32(ln_t, ln_d);

        // Only accumulate where t > eps to handle zeros in distribution.
        let contrib = vmulq_f32(t_safe, ln_ratio);
        acc = vaddq_f32(acc, contrib);
    }

    let mut kl = vaddvq_f32(acc);

    for i in 0..remainder {
        let idx = chunks * LANES + i;
        let t = target_dist[idx].max(1e-10);
        let d = draft_dist[idx].max(1e-10);
        kl += t * fast_ln_scalar(t / d);
    }

    kl.max(0.0)
}

fn kl_divergence_scalar(target_dist: &[f32], draft_dist: &[f32]) -> f32 {
    let mut kl = 0.0f32;
    for i in 0..target_dist.len() {
        let t = target_dist[i].max(1e-10);
        let d = draft_dist[i].max(1e-10);
        kl += t * fast_ln_scalar(t / d);
    }
    kl.max(0.0)
}

/// Check KL divergence between target and draft distributions.
///
/// Computes `KL(target || draft) = sum(target * ln(target / draft))` and
/// compares against the given threshold.
pub fn kl_divergence_check(
    target_dist: &[f32],
    draft_dist: &[f32],
    threshold: f32,
) -> KlDivergenceResult {
    let len = target_dist.len();
    assert_eq!(draft_dist.len(), len);

    if len == 0 {
        return KlDivergenceResult {
            kl_divergence: 0.0,
            within_threshold: true,
        };
    }

    let kl;
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            kl = unsafe { kl_divergence_neon(target_dist, draft_dist) };
        } else {
            kl = kl_divergence_scalar(target_dist, draft_dist);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        kl = kl_divergence_scalar(target_dist, draft_dist);
    }

    KlDivergenceResult {
        kl_divergence: kl,
        within_threshold: kl <= threshold,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── draft_token_verify tests ────────────────────────────────────────

    #[test]
    fn test_draft_verify_empty() {
        let result = draft_token_verify(&[], &[], &[]);
        assert_eq!(result.accepted_count, 0);
        assert!(result.accepted.is_empty());
    }

    #[test]
    fn test_draft_verify_single_accepted() {
        let result = draft_token_verify(&[1], &[0.3], &[0.5]);
        assert_eq!(result.accepted_count, 1);
        assert_eq!(result.accepted, vec![true]);
    }

    #[test]
    fn test_draft_verify_single_rejected() {
        let result = draft_token_verify(&[1], &[0.8], &[0.3]);
        assert_eq!(result.accepted_count, 0);
        assert_eq!(result.accepted, vec![false]);
    }

    #[test]
    fn test_draft_verify_all_accepted() {
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let draft_probs = vec![0.1, 0.2, 0.3, 0.1, 0.2, 0.1, 0.15, 0.05];
        let target_probs = vec![0.5, 0.4, 0.6, 0.3, 0.5, 0.2, 0.3, 0.1];
        let result = draft_token_verify(&tokens, &draft_probs, &target_probs);
        assert_eq!(result.accepted_count, 8);
        assert!(result.accepted.iter().all(|&a| a));
    }

    #[test]
    fn test_draft_verify_all_rejected() {
        let tokens = vec![1, 2, 3, 4];
        let draft_probs = vec![0.9, 0.8, 0.7, 0.6];
        let target_probs = vec![0.1, 0.2, 0.3, 0.4];
        let result = draft_token_verify(&tokens, &draft_probs, &target_probs);
        assert_eq!(result.accepted_count, 0);
        assert!(result.accepted.iter().all(|&a| !a));
    }

    #[test]
    fn test_draft_verify_partial_contiguous() {
        let tokens = vec![1, 2, 3, 4, 5];
        let draft_probs = vec![0.1, 0.2, 0.3, 0.8, 0.1];
        let target_probs = vec![0.5, 0.4, 0.6, 0.1, 0.5];
        let result = draft_token_verify(&tokens, &draft_probs, &target_probs);
        assert_eq!(result.accepted_count, 3);
        assert_eq!(result.accepted, vec![true, true, true, false, true]);
    }

    #[test]
    fn test_draft_verify_equal_probs() {
        let tokens = vec![1, 2, 3, 4];
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let result = draft_token_verify(&tokens, &probs, &probs);
        assert_eq!(result.accepted_count, 4);
    }

    #[test]
    fn test_draft_verify_neon_alignment_5_elements() {
        let tokens = vec![1, 2, 3, 4, 5];
        let draft = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let target = vec![0.5, 0.4, 0.3, 0.2, 0.1];
        let result = draft_token_verify(&tokens, &draft, &target);
        assert_eq!(result.accepted, vec![true, true, true, false, false]);
    }

    #[test]
    fn test_draft_verify_neon_alignment_7_elements() {
        let tokens = vec![10, 20, 30, 40, 50, 60, 70];
        let draft = vec![0.1; 7];
        let target = vec![0.5; 7];
        let result = draft_token_verify(&tokens, &draft, &target);
        assert_eq!(result.accepted_count, 7);
    }

    #[test]
    fn test_draft_verify_neon_alignment_9_elements() {
        let tokens: Vec<u32> = (0..9).collect();
        let draft = vec![0.5; 9];
        let target = vec![0.1; 9];
        let result = draft_token_verify(&tokens, &draft, &target);
        assert_eq!(result.accepted_count, 0);
    }

    #[test]
    fn test_draft_verify_large_batch() {
        let n = 128;
        let tokens: Vec<u32> = (0..n as u32).collect();
        let draft_probs = vec![0.1; n];
        let target_probs = vec![0.9; n];
        let result = draft_token_verify(&tokens, &draft_probs, &target_probs);
        assert_eq!(result.accepted_count, n);
    }

    // ── acceptance_probability tests ────────────────────────────────────

    #[test]
    fn test_acceptance_prob_empty() {
        let result = acceptance_probability(&[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_acceptance_prob_target_higher() {
        let draft = vec![0.3, 0.2, 0.1, 0.4];
        let target = vec![0.6, 0.5, 0.3, 0.8];
        let result = acceptance_probability(&draft, &target);
        for &p in &result {
            assert!((0.0..=1.0).contains(&p));
        }
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_acceptance_prob_target_lower() {
        let draft = vec![0.8, 0.7];
        let target = vec![0.4, 0.35];
        let result = acceptance_probability(&draft, &target);
        assert!((result[0] - 0.5).abs() < 1e-5);
        assert!((result[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_acceptance_prob_clamped_to_one() {
        let draft = vec![0.1];
        let target = vec![0.9];
        let result = acceptance_probability(&draft, &target);
        assert!((result[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_acceptance_prob_zero_draft() {
        let draft = vec![0.0, 0.0];
        let target = vec![0.5, 0.0];
        let result = acceptance_probability(&draft, &target);
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!(result[1] >= 0.0 && result[1] <= 1.0);
    }

    #[test]
    fn test_acceptance_prob_identical_distributions() {
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let result = acceptance_probability(&probs, &probs);
        for &p in &result {
            assert!((p - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_acceptance_prob_neon_alignment_5() {
        let draft = vec![0.2; 5];
        let target = vec![0.1; 5];
        let result = acceptance_probability(&draft, &target);
        assert_eq!(result.len(), 5);
        for &p in &result {
            assert!((p - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn test_acceptance_prob_neon_alignment_9() {
        let draft = vec![0.5; 9];
        let target = vec![0.5; 9];
        let result = acceptance_probability(&draft, &target);
        for &p in &result {
            assert!((p - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_acceptance_prob_large_batch() {
        let n = 256;
        let draft = vec![0.4; n];
        let target = vec![0.2; n];
        let result = acceptance_probability(&draft, &target);
        assert_eq!(result.len(), n);
        for &p in &result {
            assert!((p - 0.5).abs() < 1e-5);
        }
    }

    // ── NEON vs scalar parity tests ─────────────────────────────────────

    #[test]
    fn test_draft_verify_neon_scalar_parity() {
        let tokens: Vec<u32> = (0..17).collect();
        let draft: Vec<f32> = (0..17).map(|i| 0.05 * (i as f32 + 1.0)).collect();
        let target: Vec<f32> = (0..17).map(|i| 0.9 - 0.05 * i as f32).collect();
        let neon_result = draft_token_verify(&tokens, &draft, &target);
        let scalar_result = draft_token_verify_scalar(&draft, &target);
        for i in 0..17 {
            assert_eq!(
                neon_result.accepted[i], scalar_result[i],
                "mismatch at {i}"
            );
        }
    }

    #[test]
    fn test_acceptance_prob_neon_scalar_parity() {
        let draft: Vec<f32> =
            (0..19).map(|i| 0.05 * (i as f32 + 1.0)).collect();
        let target: Vec<f32> = (0..19).map(|i| 0.9 - 0.04 * i as f32).collect();
        let neon = acceptance_probability(&draft, &target);
        let scalar = acceptance_probability_scalar(&draft, &target);
        for i in 0..19 {
            assert!(
                (neon[i] - scalar[i]).abs() < 1e-5,
                "mismatch at {i}: neon={}, scalar={}",
                neon[i],
                scalar[i]
            );
        }
    }

    #[test]
    fn test_parallel_scoring_neon_scalar_parity() {
        let logits: Vec<f32> =
            (0..32).map(|i| -2.0 + 0.2 * i as f32).collect();
        let candidates: Vec<u32> = vec![0, 5, 15, 31];
        let neon = parallel_draft_scoring(&logits, &candidates);
        let scalar = parallel_draft_scoring_scalar(&logits, &candidates);
        for i in 0..candidates.len() {
            assert!(
                (neon[i].log_prob - scalar[i].log_prob).abs() < 0.02,
                "mismatch at {i}: neon={}, scalar={}",
                neon[i].log_prob,
                scalar[i].log_prob
            );
        }
    }

    #[test]
    fn test_rejection_sampling_neon_scalar_parity() {
        let draft = vec![0.3, 0.4, 0.2, 0.5, 0.1];
        let target = vec![0.5, 0.3, 0.4, 0.2, 0.6];
        let random = vec![0.5, 0.5, 0.5, 0.5, 0.5];
        let neon = rejection_sampling(&draft, &target, &random);
        let scalar = rejection_sampling_scalar(&draft, &target, &random);
        assert_eq!(neon.accepted_count, scalar.accepted_count);
        assert_eq!(neon.accepted, scalar.accepted);
    }

    #[test]
    fn test_kl_divergence_neon_scalar_parity() {
        let target = vec![0.4, 0.3, 0.2, 0.1];
        let draft = vec![0.25, 0.25, 0.25, 0.25];
        let neon =
            kl_divergence_check(&target, &draft, 1.0).kl_divergence;
        let scalar = kl_divergence_scalar(&target, &draft);
        assert!(
            (neon - scalar).abs() < 0.01,
            "neon={neon}, scalar={scalar}"
        );
    }

    #[test]
    fn test_kl_divergence_neon_scalar_parity_large() {
        let n = 64;
        let target: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0)).collect();
        let sum_t: f32 = target.iter().sum();
        let target: Vec<f32> = target.iter().map(|&v| v / sum_t).collect();
        let draft = vec![1.0 / n as f32; n];
        let neon =
            kl_divergence_check(&target, &draft, 10.0).kl_divergence;
        let scalar = kl_divergence_scalar(&target, &draft);
        assert!(
            (neon - scalar).abs() < 0.05,
            "neon={neon}, scalar={scalar}"
        );
    }

    // ── parallel_draft_scoring tests ────────────────────────────────────

    #[test]
    fn test_scoring_empty_logits() {
        let scores = parallel_draft_scoring(&[], &[1, 2, 3]);
        for s in &scores {
            assert!(s.log_prob == f32::NEG_INFINITY);
        }
    }

    #[test]
    fn test_scoring_empty_candidates() {
        let scores = parallel_draft_scoring(&[1.0, 2.0, 3.0], &[]);
        assert!(scores.is_empty());
    }

    #[test]
    fn test_scoring_single_token() {
        let logits = vec![0.0; 10];
        let scores = parallel_draft_scoring(&logits, &[5]);
        assert_eq!(scores.len(), 1);
        let expected = -(10.0f32).ln();
        assert!(
            (scores[0].log_prob - expected).abs() < 0.05,
            "got {}",
            scores[0].log_prob
        );
    }

    #[test]
    fn test_scoring_out_of_bounds_candidate() {
        let logits = vec![1.0, 2.0, 3.0];
        let scores = parallel_draft_scoring(&logits, &[100]);
        assert!(scores[0].log_prob == f32::NEG_INFINITY);
    }

    #[test]
    fn test_scoring_multiple_candidates() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let scores = parallel_draft_scoring(&logits, &[0, 1, 2, 3]);
        assert_eq!(scores.len(), 4);
        // Higher logit → higher log-prob.
        assert!(scores[3].log_prob > scores[0].log_prob);
    }

    #[test]
    fn test_scoring_preserves_token_ids() {
        let logits = vec![1.0; 8];
        let ids = vec![42, 99, 7, 0];
        let scores = parallel_draft_scoring(&logits, &ids);
        for (i, s) in scores.iter().enumerate() {
            assert_eq!(s.token_id, ids[i]);
        }
    }

    #[test]
    fn test_scoring_negative_logits() {
        let logits = vec![-10.0, -5.0, -1.0, 0.0];
        let scores = parallel_draft_scoring(&logits, &[0, 3]);
        assert!(scores[1].log_prob > scores[0].log_prob);
    }

    #[test]
    fn test_scoring_large_vocab() {
        let n = 512;
        let logits: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let candidates: Vec<u32> = (0..16).map(|i| i * 32).collect();
        let scores = parallel_draft_scoring(&logits, &candidates);
        assert_eq!(scores.len(), 16);
        for s in &scores {
            assert!(s.log_prob.is_finite());
        }
    }

    // ── rejection_sampling tests ────────────────────────────────────────

    #[test]
    fn test_rejection_empty() {
        let result = rejection_sampling(&[], &[], &[]);
        assert_eq!(result.accepted_count, 0);
        assert!(result.accepted.is_empty());
        assert!(result.adjusted_distribution.is_none());
    }

    #[test]
    fn test_rejection_all_accepted() {
        let draft = vec![0.1, 0.1, 0.1, 0.1];
        let target = vec![0.9, 0.9, 0.9, 0.9];
        let random = vec![0.5, 0.5, 0.5, 0.5];
        let result = rejection_sampling(&draft, &target, &random);
        assert_eq!(result.accepted_count, 4);
        assert!(result.adjusted_distribution.is_none());
    }

    #[test]
    fn test_rejection_all_rejected() {
        let draft = vec![0.9, 0.9, 0.9, 0.9];
        let target = vec![0.1, 0.1, 0.1, 0.1];
        let random = vec![0.5, 0.5, 0.5, 0.5];
        let result = rejection_sampling(&draft, &target, &random);
        assert_eq!(result.accepted_count, 0);
        assert!(result.adjusted_distribution.is_some());
    }

    #[test]
    fn test_rejection_partial() {
        // ratio = target/draft. First two have ratio >= 1 (always accept).
        // Third has ratio 0.25 < random 0.5 → reject.
        let draft = vec![0.1, 0.2, 0.8, 0.1];
        let target = vec![0.9, 0.8, 0.2, 0.9];
        let random = vec![0.5, 0.5, 0.5, 0.5];
        let result = rejection_sampling(&draft, &target, &random);
        assert_eq!(result.accepted_count, 2);
        assert!(result.accepted[0]);
        assert!(result.accepted[1]);
        assert!(!result.accepted[2]);
    }

    #[test]
    fn test_rejection_adjusted_dist_sums_to_one() {
        let draft = vec![0.5, 0.3, 0.2];
        let target = vec![0.2, 0.5, 0.3];
        let random = vec![0.99, 0.99, 0.99];
        let result = rejection_sampling(&draft, &target, &random);
        if let Some(ref adj) = result.adjusted_distribution {
            let sum: f32 = adj.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "adjusted dist sum = {sum}"
            );
        }
    }

    #[test]
    fn test_rejection_deterministic_with_random_zero() {
        // random = 0 → always accept (since prob > 0).
        let draft = vec![0.5, 0.5, 0.5, 0.5];
        let target = vec![0.01, 0.01, 0.01, 0.01];
        let random = vec![0.0, 0.0, 0.0, 0.0];
        let result = rejection_sampling(&draft, &target, &random);
        assert_eq!(result.accepted_count, 4);
    }

    #[test]
    fn test_rejection_deterministic_with_random_one() {
        // random = 1.0 → never accept (since prob < 1 when target < draft).
        let draft = vec![0.5, 0.5, 0.5, 0.5];
        let target = vec![0.3, 0.3, 0.3, 0.3];
        let random = vec![1.0, 1.0, 1.0, 1.0];
        let result = rejection_sampling(&draft, &target, &random);
        assert_eq!(result.accepted_count, 0);
    }

    #[test]
    fn test_rejection_neon_alignment_5() {
        let draft = vec![0.2; 5];
        let target = vec![0.8; 5];
        let random = vec![0.5; 5];
        let result = rejection_sampling(&draft, &target, &random);
        assert_eq!(result.accepted_count, 5);
    }

    // ── token_tree_verify tests ─────────────────────────────────────────

    #[test]
    fn test_tree_verify_empty() {
        let result = token_tree_verify(&[]);
        assert!(result.best_path.is_empty());
        assert_eq!(result.total_accepted, 0);
    }

    #[test]
    fn test_tree_verify_single_accepted_node() {
        let tree = vec![TokenTreeNode {
            token_id: 42,
            draft_prob: 0.3,
            target_prob: 0.5,
            children: vec![],
        }];
        let result = token_tree_verify(&tree);
        assert_eq!(result.best_path, vec![42]);
        assert_eq!(result.total_accepted, 1);
    }

    #[test]
    fn test_tree_verify_single_rejected_node() {
        let tree = vec![TokenTreeNode {
            token_id: 42,
            draft_prob: 0.8,
            target_prob: 0.2,
            children: vec![],
        }];
        let result = token_tree_verify(&tree);
        assert!(result.best_path.is_empty());
        assert_eq!(result.total_accepted, 0);
    }

    #[test]
    fn test_tree_verify_linear_chain_all_accepted() {
        let tree = vec![
            TokenTreeNode {
                token_id: 1,
                draft_prob: 0.2,
                target_prob: 0.5,
                children: vec![1],
            },
            TokenTreeNode {
                token_id: 2,
                draft_prob: 0.3,
                target_prob: 0.4,
                children: vec![2],
            },
            TokenTreeNode {
                token_id: 3,
                draft_prob: 0.1,
                target_prob: 0.6,
                children: vec![],
            },
        ];
        let result = token_tree_verify(&tree);
        assert_eq!(result.best_path, vec![1, 2, 3]);
        assert_eq!(result.total_accepted, 3);
    }

    #[test]
    fn test_tree_verify_branching() {
        // Root → child 1 (accepted) → grandchild (accepted)
        //      → child 2 (rejected)
        let tree = vec![
            TokenTreeNode {
                token_id: 10,
                draft_prob: 0.1,
                target_prob: 0.5,
                children: vec![1, 2],
            },
            TokenTreeNode {
                token_id: 20,
                draft_prob: 0.2,
                target_prob: 0.4,
                children: vec![3],
            },
            TokenTreeNode {
                token_id: 30,
                draft_prob: 0.9,
                target_prob: 0.1,
                children: vec![],
            },
            TokenTreeNode {
                token_id: 40,
                draft_prob: 0.1,
                target_prob: 0.3,
                children: vec![],
            },
        ];
        let result = token_tree_verify(&tree);
        assert_eq!(result.best_path, vec![10, 20, 40]);
        assert_eq!(result.total_accepted, 3);
        assert!(!result.node_accepted[2]); // node 30 rejected
    }

    #[test]
    fn test_tree_verify_picks_longest_branch() {
        // Root → branch A (1 node) vs branch B (2 nodes).
        let tree = vec![
            TokenTreeNode {
                token_id: 1,
                draft_prob: 0.1,
                target_prob: 0.5,
                children: vec![1, 2],
            },
            TokenTreeNode {
                token_id: 2,
                draft_prob: 0.1,
                target_prob: 0.5,
                children: vec![],
            },
            TokenTreeNode {
                token_id: 3,
                draft_prob: 0.1,
                target_prob: 0.5,
                children: vec![3],
            },
            TokenTreeNode {
                token_id: 4,
                draft_prob: 0.1,
                target_prob: 0.5,
                children: vec![],
            },
        ];
        let result = token_tree_verify(&tree);
        assert_eq!(result.best_path, vec![1, 3, 4]);
    }

    #[test]
    fn test_tree_verify_root_rejected() {
        let tree = vec![
            TokenTreeNode {
                token_id: 1,
                draft_prob: 0.9,
                target_prob: 0.1,
                children: vec![1],
            },
            TokenTreeNode {
                token_id: 2,
                draft_prob: 0.1,
                target_prob: 0.9,
                children: vec![],
            },
        ];
        let result = token_tree_verify(&tree);
        assert!(result.best_path.is_empty());
    }

    #[test]
    fn test_tree_verify_large_tree() {
        let n = 32;
        let mut tree: Vec<TokenTreeNode> = Vec::with_capacity(n);
        for i in 0..n {
            tree.push(TokenTreeNode {
                token_id: i as u32,
                draft_prob: 0.1,
                target_prob: 0.5,
                children: if i + 1 < n { vec![i + 1] } else { vec![] },
            });
        }
        let result = token_tree_verify(&tree);
        assert_eq!(result.best_path.len(), n);
        assert_eq!(result.total_accepted, n);
    }

    // ── kl_divergence_check tests ───────────────────────────────────────

    #[test]
    fn test_kl_empty() {
        let result = kl_divergence_check(&[], &[], 0.1);
        assert!((result.kl_divergence - 0.0).abs() < 1e-10);
        assert!(result.within_threshold);
    }

    #[test]
    fn test_kl_identical_distributions() {
        let dist = vec![0.25, 0.25, 0.25, 0.25];
        let result = kl_divergence_check(&dist, &dist, 0.01);
        assert!(
            result.kl_divergence < 0.01,
            "KL = {}",
            result.kl_divergence
        );
        assert!(result.within_threshold);
    }

    #[test]
    fn test_kl_different_distributions() {
        let target = vec![0.9, 0.05, 0.025, 0.025];
        let draft = vec![0.25, 0.25, 0.25, 0.25];
        let result = kl_divergence_check(&target, &draft, 0.5);
        assert!(result.kl_divergence > 0.0);
        // KL(peaked || uniform) should be sizable.
        assert!(result.kl_divergence > 0.3);
    }

    #[test]
    fn test_kl_within_threshold() {
        let target = vec![0.3, 0.3, 0.2, 0.2];
        let draft = vec![0.25, 0.25, 0.25, 0.25];
        let result = kl_divergence_check(&target, &draft, 1.0);
        assert!(result.within_threshold);
    }

    #[test]
    fn test_kl_exceeds_threshold() {
        let target = vec![0.99, 0.003, 0.003, 0.004];
        let draft = vec![0.25, 0.25, 0.25, 0.25];
        let result = kl_divergence_check(&target, &draft, 0.01);
        assert!(!result.within_threshold);
    }

    #[test]
    fn test_kl_non_negative() {
        let target = vec![0.1, 0.2, 0.3, 0.4];
        let draft = vec![0.4, 0.3, 0.2, 0.1];
        let result = kl_divergence_check(&target, &draft, 10.0);
        assert!(result.kl_divergence >= 0.0);
    }

    #[test]
    fn test_kl_neon_alignment_5() {
        let target = vec![0.2, 0.2, 0.2, 0.2, 0.2];
        let draft = vec![0.2, 0.2, 0.2, 0.2, 0.2];
        let result = kl_divergence_check(&target, &draft, 0.01);
        assert!(result.kl_divergence < 0.01);
    }

    #[test]
    fn test_kl_neon_alignment_9() {
        let n = 9;
        let val = 1.0 / n as f32;
        let dist = vec![val; n];
        let result = kl_divergence_check(&dist, &dist, 0.01);
        assert!(result.kl_divergence < 0.01);
    }

    #[test]
    fn test_kl_large_vocab() {
        let n = 256;
        let target: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0)).collect();
        let sum_t: f32 = target.iter().sum();
        let target: Vec<f32> = target.iter().map(|&v| v / sum_t).collect();
        let draft = vec![1.0 / n as f32; n];
        let result = kl_divergence_check(&target, &draft, 10.0);
        assert!(result.kl_divergence > 0.0);
        assert!(result.kl_divergence.is_finite());
    }

    #[test]
    fn test_kl_zero_in_target() {
        let target = vec![0.0, 0.0, 0.5, 0.5];
        let draft = vec![0.25, 0.25, 0.25, 0.25];
        let result = kl_divergence_check(&target, &draft, 10.0);
        assert!(result.kl_divergence.is_finite());
    }

    // ── Additional edge-case and stress tests ───────────────────────────

    #[test]
    fn test_fast_exp_scalar_bounds() {
        assert!((fast_exp_scalar(0.0) - 1.0).abs() < 1e-4);
        assert!(fast_exp_scalar(-100.0) >= 0.0);
        assert!(fast_exp_scalar(100.0).is_finite());
    }

    #[test]
    fn test_fast_ln_scalar_basic() {
        assert!((fast_ln_scalar(1.0) - 0.0).abs() < 1e-3);
        assert!(fast_ln_scalar(0.0) == f32::NEG_INFINITY);
        assert!(fast_ln_scalar(-1.0) == f32::NEG_INFINITY);
    }

    #[test]
    fn test_fast_ln_scalar_e() {
        let result = fast_ln_scalar(std::f32::consts::E);
        assert!((result - 1.0).abs() < 0.05, "ln(e) = {result}");
    }

    #[test]
    fn test_draft_verify_boundary_equal_probs() {
        // Edge: target == draft should be accepted (>=).
        let tokens = vec![1, 2];
        let probs = vec![0.5, 0.5];
        let result = draft_token_verify(&tokens, &probs, &probs);
        assert_eq!(result.accepted_count, 2);
    }

    #[test]
    fn test_acceptance_prob_very_small_draft() {
        let draft = vec![1e-30];
        let target = vec![0.5];
        let result = acceptance_probability(&draft, &target);
        assert!((result[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rejection_single_token() {
        let result =
            rejection_sampling(&[0.3], &[0.6], &[0.5]);
        assert_eq!(result.accepted_count, 1);
    }

    #[test]
    fn test_tree_verify_all_leaves_rejected() {
        let tree = vec![
            TokenTreeNode {
                token_id: 1,
                draft_prob: 0.1,
                target_prob: 0.5,
                children: vec![1, 2],
            },
            TokenTreeNode {
                token_id: 2,
                draft_prob: 0.9,
                target_prob: 0.1,
                children: vec![],
            },
            TokenTreeNode {
                token_id: 3,
                draft_prob: 0.9,
                target_prob: 0.1,
                children: vec![],
            },
        ];
        let result = token_tree_verify(&tree);
        assert_eq!(result.best_path, vec![1]);
    }

    #[test]
    fn test_scoring_uniform_logits() {
        let logits = vec![0.0; 100];
        let scores = parallel_draft_scoring(&logits, &[0, 50, 99]);
        let expected = -(100.0f32).ln();
        for s in &scores {
            assert!(
                (s.log_prob - expected).abs() < 0.1,
                "got {}",
                s.log_prob
            );
        }
    }

    #[test]
    fn test_kl_symmetry_check() {
        let p = vec![0.6, 0.3, 0.1];
        let q = vec![0.3, 0.4, 0.3];
        let kl_pq = kl_divergence_check(&p, &q, 10.0).kl_divergence;
        let kl_qp = kl_divergence_check(&q, &p, 10.0).kl_divergence;
        // KL is NOT symmetric, so they should differ.
        assert!((kl_pq - kl_qp).abs() > 1e-4);
    }

    #[test]
    fn test_draft_verify_exactly_4_elements() {
        let tokens = vec![1, 2, 3, 4];
        let draft = vec![0.3, 0.3, 0.3, 0.3];
        let target = vec![0.5, 0.5, 0.5, 0.5];
        let result = draft_token_verify(&tokens, &draft, &target);
        assert_eq!(result.accepted_count, 4);
    }

    #[test]
    fn test_draft_verify_exactly_8_elements() {
        let tokens: Vec<u32> = (0..8).collect();
        let draft = vec![0.3; 8];
        let target = vec![0.5; 8];
        let result = draft_token_verify(&tokens, &draft, &target);
        assert_eq!(result.accepted_count, 8);
    }

    #[test]
    fn test_acceptance_prob_single() {
        let result = acceptance_probability(&[0.5], &[0.25]);
        assert!((result[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_rejection_neon_alignment_9() {
        let n = 9;
        let draft = vec![0.1; n];
        let target = vec![0.9; n];
        let random = vec![0.5; n];
        let result = rejection_sampling(&draft, &target, &random);
        assert_eq!(result.accepted_count, n);
    }
}
