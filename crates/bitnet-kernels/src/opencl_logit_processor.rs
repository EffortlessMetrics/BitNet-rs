//! OpenCL logit-processing pipeline for controlling text generation output.
//!
//! Provides CPU reference implementations and OpenCL kernel sources for the
//! full logit-warping stack used during autoregressive decoding:
//!
//! - **Temperature scaling** — sharpen or flatten the distribution
//! - **Top-k filtering** — retain only the k highest logits
//! - **Top-p (nucleus) filtering** — retain tokens within cumulative probability p
//! - **Min-p filtering** — dynamic threshold relative to max probability
//! - **Logit bias** — per-token additive bias (prompt engineering)
//! - **N-gram blocking** — prevent repeated N-grams
//! - **Forced tokens** — override output at specific positions
//! - **Warper pipeline** — ordered composition of the above
//! - **Logit statistics** — entropy, confidence, effective vocab size
//!
//! All implementations are pure CPU reference code with no OpenCL runtime
//! dependency, ready for future GPU acceleration via `include_str!` kernels.

use std::collections::{HashMap, HashSet};

// ── OpenCL kernel source (placeholder) ──────────────────────────

/// OpenCL kernel source for logit processing operations.
///
/// Will contain GPU-accelerated versions of temperature scaling, top-k,
/// top-p, and softmax once the OpenCL backend is wired.
pub const LOGIT_PROCESSOR_CL: &str = r#"
// Placeholder: logit_processor.cl
// GPU kernels for logit warping will be added when OpenCL runtime is available.

__kernel void temperature_scale(
    __global float* logits,
    const uint len,
    const float inv_temp
) {
    uint gid = get_global_id(0);
    if (gid < len) {
        logits[gid] *= inv_temp;
    }
}

__kernel void apply_mask_neg_inf(
    __global float* logits,
    __global const uchar* mask,
    const uint len
) {
    uint gid = get_global_id(0);
    if (gid < len && mask[gid] == 0) {
        logits[gid] = -INFINITY;
    }
}
"#;

// ── Trait ────────────────────────────────────────────────────────

/// A logit transformation applied before sampling.
///
/// Implementations modify the logit slice in-place. The `token_ids` slice
/// contains previously generated token IDs (needed by N-gram blocking).
/// `position` is the current generation step (needed by forced-token).
pub trait LogitWarper: std::fmt::Debug {
    /// Apply the transformation to `logits` in-place.
    fn warp(&self, logits: &mut [f32], token_ids: &[u32], position: usize);

    /// Human-readable name for diagnostics.
    fn name(&self) -> &str;
}

// ── TemperatureWarper ───────────────────────────────────────────

/// Scale logits by `1 / temperature`.
///
/// * `temperature == 0.0` → greedy (set all but argmax to `NEG_INFINITY`)
/// * `temperature == 1.0` → identity (no-op)
/// * `(0, 1)` → sharpen  
/// * `> 1` → flatten
#[derive(Debug, Clone)]
pub struct TemperatureWarper {
    pub temperature: f32,
}

impl TemperatureWarper {
    pub fn new(temperature: f32) -> Self {
        assert!(temperature >= 0.0, "temperature must be non-negative");
        Self { temperature }
    }
}

impl LogitWarper for TemperatureWarper {
    fn warp(&self, logits: &mut [f32], _token_ids: &[u32], _position: usize) {
        if logits.is_empty() {
            return;
        }
        #[allow(clippy::float_cmp)]
        if self.temperature == 0.0 {
            // Greedy: keep only argmax
            let max_idx = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            for (i, l) in logits.iter_mut().enumerate() {
                if i != max_idx {
                    *l = f32::NEG_INFINITY;
                }
            }
            return;
        }
        #[allow(clippy::float_cmp)]
        if self.temperature == 1.0 {
            return;
        }
        let inv = 1.0 / self.temperature;
        for l in logits.iter_mut() {
            *l *= inv;
        }
    }

    fn name(&self) -> &str {
        "TemperatureWarper"
    }
}

// ── TopKWarper ──────────────────────────────────────────────────

/// Keep only the top-k logits; set the rest to `NEG_INFINITY`.
///
/// `k == 0` or `k >= vocab_size` is a no-op.
#[derive(Debug, Clone)]
pub struct TopKWarper {
    pub k: usize,
}

impl TopKWarper {
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl LogitWarper for TopKWarper {
    fn warp(&self, logits: &mut [f32], _token_ids: &[u32], _position: usize) {
        if self.k == 0 || self.k >= logits.len() || logits.is_empty() {
            return;
        }
        // Find the k-th largest value via partial sort.
        let mut vals: Vec<f32> = logits.to_vec();
        let partition_idx = vals.len() - self.k;
        vals.select_nth_unstable_by(partition_idx, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let threshold = vals[partition_idx];

        let mut kept = 0usize;
        for l in logits.iter_mut() {
            if *l >= threshold && kept < self.k {
                kept += 1;
            } else {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    fn name(&self) -> &str {
        "TopKWarper"
    }
}

// ── TopPWarper ──────────────────────────────────────────────────

/// Nucleus (top-p) sampling: after softmax, zero tokens beyond cumulative
/// probability `p`.
///
/// This warper operates on **logits** (pre-softmax). It internally computes
/// softmax, determines the cutoff, then masks logits below the threshold
/// to `NEG_INFINITY`.
#[derive(Debug, Clone)]
pub struct TopPWarper {
    pub p: f32,
}

impl TopPWarper {
    pub fn new(p: f32) -> Self {
        assert!((0.0..=1.0).contains(&p), "p must be in [0.0, 1.0]");
        Self { p }
    }
}

impl LogitWarper for TopPWarper {
    fn warp(&self, logits: &mut [f32], _token_ids: &[u32], _position: usize) {
        #[allow(clippy::float_cmp)]
        if self.p >= 1.0 || logits.is_empty() {
            return;
        }

        // Compute softmax probabilities for ranking.
        let probs = softmax(logits);

        // Sort indices by probability descending.
        let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0f32;
        let mut keep_set = HashSet::new();
        for &(idx, prob) in &indexed {
            keep_set.insert(idx);
            cumsum += prob;
            if cumsum >= self.p {
                break;
            }
        }

        for (i, l) in logits.iter_mut().enumerate() {
            if !keep_set.contains(&i) {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    fn name(&self) -> &str {
        "TopPWarper"
    }
}

// ── MinPWarper ──────────────────────────────────────────────────

/// Min-p filtering: mask tokens whose probability is below
/// `min_p × max_probability`.
///
/// Operates on logits by internally computing softmax to determine
/// probabilities, then masking low-probability tokens.
#[derive(Debug, Clone)]
pub struct MinPWarper {
    pub min_p: f32,
}

impl MinPWarper {
    pub fn new(min_p: f32) -> Self {
        assert!((0.0..=1.0).contains(&min_p), "min_p must be in [0.0, 1.0]");
        Self { min_p }
    }
}

impl LogitWarper for MinPWarper {
    fn warp(&self, logits: &mut [f32], _token_ids: &[u32], _position: usize) {
        if self.min_p <= 0.0 || logits.is_empty() {
            return;
        }

        let probs = softmax(logits);
        let max_prob = probs.iter().copied().fold(0.0f32, f32::max);
        let threshold = self.min_p * max_prob;

        for (i, l) in logits.iter_mut().enumerate() {
            if probs[i] < threshold {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    fn name(&self) -> &str {
        "MinPWarper"
    }
}

// ── LogitBias ───────────────────────────────────────────────────

/// Add per-token bias to logits. Positive biases encourage tokens;
/// negative biases discourage them. `-inf` bans a token entirely.
#[derive(Debug, Clone)]
pub struct LogitBias {
    /// Map from token ID to additive bias.
    pub biases: HashMap<u32, f32>,
}

impl LogitBias {
    pub fn new(biases: HashMap<u32, f32>) -> Self {
        Self { biases }
    }

    /// Convenience: create from an iterator of `(token_id, bias)` pairs.
    pub fn from_pairs(iter: impl IntoIterator<Item = (u32, f32)>) -> Self {
        Self { biases: iter.into_iter().collect() }
    }
}

impl LogitWarper for LogitBias {
    fn warp(&self, logits: &mut [f32], _token_ids: &[u32], _position: usize) {
        for (&token_id, &bias) in &self.biases {
            let idx = token_id as usize;
            if idx < logits.len() {
                logits[idx] += bias;
            }
        }
    }

    fn name(&self) -> &str {
        "LogitBias"
    }
}

// ── NoRepeatNgramWarper ────────────────────────────────────────

/// Prevent repetition of N-grams that already appear in the token history.
///
/// If the last `n-1` tokens of `token_ids` form a prefix that appeared
/// earlier followed by some token `t`, then `t` is masked to `NEG_INFINITY`.
#[derive(Debug, Clone)]
pub struct NoRepeatNgramWarper {
    pub n: usize,
}

impl NoRepeatNgramWarper {
    pub fn new(n: usize) -> Self {
        assert!(n >= 2, "n-gram size must be at least 2");
        Self { n }
    }
}

impl LogitWarper for NoRepeatNgramWarper {
    fn warp(&self, logits: &mut [f32], token_ids: &[u32], _position: usize) {
        if token_ids.len() < self.n - 1 || logits.is_empty() {
            return;
        }

        // The current suffix of length n-1.
        let suffix = &token_ids[token_ids.len() - (self.n - 1)..];

        // Scan history for matching (n-1)-grams and collect the tokens that follow.
        let mut banned = HashSet::new();
        if token_ids.len() >= self.n {
            for start in 0..=token_ids.len() - self.n {
                let window = &token_ids[start..start + self.n - 1];
                if window == suffix {
                    let next_token = token_ids[start + self.n - 1];
                    banned.insert(next_token);
                }
            }
        }

        for token_id in banned {
            let idx = token_id as usize;
            if idx < logits.len() {
                logits[idx] = f32::NEG_INFINITY;
            }
        }
    }

    fn name(&self) -> &str {
        "NoRepeatNgramWarper"
    }
}

// ── ForcedTokenWarper ──────────────────────────────────────────

/// Force a specific token at a specific generation position.
///
/// At `position == target_position`, all logits except the forced token
/// are set to `NEG_INFINITY`.
#[derive(Debug, Clone)]
pub struct ForcedTokenWarper {
    /// Map from generation position to the token ID that must be produced.
    pub forced: HashMap<usize, u32>,
}

impl ForcedTokenWarper {
    pub fn new(forced: HashMap<usize, u32>) -> Self {
        Self { forced }
    }

    /// Convenience: force a single token at a given position.
    pub fn single(position: usize, token_id: u32) -> Self {
        let mut forced = HashMap::new();
        forced.insert(position, token_id);
        Self { forced }
    }
}

impl LogitWarper for ForcedTokenWarper {
    fn warp(&self, logits: &mut [f32], _token_ids: &[u32], position: usize) {
        if let Some(&forced_id) = self.forced.get(&position) {
            let idx = forced_id as usize;
            if idx < logits.len() {
                for (i, l) in logits.iter_mut().enumerate() {
                    if i != idx {
                        *l = f32::NEG_INFINITY;
                    }
                }
            }
        }
    }

    fn name(&self) -> &str {
        "ForcedTokenWarper"
    }
}

// ── WarperPipeline ─────────────────────────────────────────────

/// Ordered composition of logit warpers.
///
/// Warpers are applied in the order they were added. Order matters:
/// e.g. temperature before top-k yields different results than the reverse.
#[derive(Debug)]
pub struct WarperPipeline {
    warpers: Vec<Box<dyn LogitWarper>>,
}

impl WarperPipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self { warpers: Vec::new() }
    }

    /// Append a warper to the end of the pipeline.
    pub fn push(&mut self, warper: Box<dyn LogitWarper>) {
        self.warpers.push(warper);
    }

    /// Number of warpers in the pipeline.
    pub fn len(&self) -> usize {
        self.warpers.len()
    }

    /// Whether the pipeline is empty.
    pub fn is_empty(&self) -> bool {
        self.warpers.is_empty()
    }

    /// Apply all warpers in order.
    pub fn apply(&self, logits: &mut [f32], token_ids: &[u32], position: usize) {
        for warper in &self.warpers {
            warper.warp(logits, token_ids, position);
        }
    }

    /// Return the names of all warpers in order.
    pub fn warper_names(&self) -> Vec<&str> {
        self.warpers.iter().map(|w| w.name()).collect()
    }
}

impl Default for WarperPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ── LogitStats ─────────────────────────────────────────────────

/// Statistics computed from a logit distribution.
#[derive(Debug, Clone)]
pub struct LogitStats {
    /// Shannon entropy of the softmax distribution (nats).
    pub entropy: f32,
    /// Probability of the most likely token.
    pub top1_confidence: f32,
    /// Token ID of the most likely token.
    pub top1_token: u32,
    /// Effective vocabulary size: `exp(entropy)`.
    pub effective_vocab_size: f32,
    /// Maximum logit value.
    pub max_logit: f32,
    /// Minimum finite logit value.
    pub min_logit: f32,
}

impl LogitStats {
    /// Compute statistics from a raw logit slice.
    pub fn compute(logits: &[f32]) -> Self {
        if logits.is_empty() {
            return Self {
                entropy: 0.0,
                top1_confidence: 0.0,
                top1_token: 0,
                effective_vocab_size: 0.0,
                max_logit: 0.0,
                min_logit: 0.0,
            };
        }

        let probs = softmax(logits);

        // Entropy: H = -Σ p * ln(p), skipping zero probabilities.
        let entropy: f32 = probs.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();

        let (top1_idx, &top1_prob) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min_logit =
            logits.iter().copied().filter(|v| v.is_finite()).fold(f32::INFINITY, f32::min);

        Self {
            entropy,
            top1_confidence: top1_prob,
            #[allow(clippy::cast_possible_truncation)]
            top1_token: top1_idx as u32,
            effective_vocab_size: entropy.exp(),
            max_logit,
            min_logit: if min_logit == f32::INFINITY { 0.0 } else { min_logit },
        }
    }
}

// ── Helpers ────────────────────────────────────────────────────

/// Numerically-stable softmax returning a new vector.
fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = logits
        .iter()
        .map(|&l| if l == f32::NEG_INFINITY { 0.0 } else { (l - max).exp() })
        .collect();
    let sum: f32 = out.iter().sum();
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in &mut out {
            *v *= inv;
        }
    }
    out
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Temperature ─────────────────────────────────────────

    #[test]
    fn temperature_scales_by_inverse() {
        let mut logits = vec![2.0, 4.0, 6.0];
        TemperatureWarper::new(2.0).warp(&mut logits, &[], 0);
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn temperature_one_is_noop() {
        let original = vec![1.0, 2.0, 3.0];
        let mut logits = original.clone();
        TemperatureWarper::new(1.0).warp(&mut logits, &[], 0);
        assert_eq!(logits, original);
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0];
        TemperatureWarper::new(0.0).warp(&mut logits, &[], 0);
        assert!(logits[1].is_finite()); // max kept
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    #[test]
    fn temperature_zero_preserves_max_value() {
        let mut logits = vec![1.0, 5.0, 3.0];
        TemperatureWarper::new(0.0).warp(&mut logits, &[], 0);
        assert!((logits[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn temperature_low_sharpens() {
        let mut logits = vec![1.0, 2.0, 3.0];
        TemperatureWarper::new(0.5).warp(&mut logits, &[], 0);
        // 1/0.5 = 2.0 scaling
        assert!((logits[0] - 2.0).abs() < 1e-6);
        assert!((logits[1] - 4.0).abs() < 1e-6);
        assert!((logits[2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn temperature_high_flattens() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let range_before = logits[2] - logits[0];
        TemperatureWarper::new(3.0).warp(&mut logits, &[], 0);
        let range_after = logits[2] - logits[0];
        assert!(range_after < range_before);
    }

    #[test]
    fn temperature_empty_logits() {
        let mut logits: Vec<f32> = vec![];
        TemperatureWarper::new(0.5).warp(&mut logits, &[], 0);
        assert!(logits.is_empty());
    }

    #[test]
    fn temperature_preserves_argmax() {
        let mut logits = vec![0.5, 3.0, 1.5, 2.5];
        TemperatureWarper::new(0.7).warp(&mut logits, &[], 0);
        let max_idx =
            logits.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
        assert_eq!(max_idx, 1);
    }

    // ─── TopK ────────────────────────────────────────────────

    #[test]
    fn topk_keeps_k_largest() {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        TopKWarper::new(2).warp(&mut logits, &[], 0);
        assert!(logits[1].is_finite()); // 5.0
        assert!(logits[4].is_finite()); // 4.0
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    #[test]
    fn topk_zero_is_noop() {
        let original = vec![1.0, 2.0, 3.0];
        let mut logits = original.clone();
        TopKWarper::new(0).warp(&mut logits, &[], 0);
        assert_eq!(logits, original);
    }

    #[test]
    fn topk_ge_vocab_is_noop() {
        let original = vec![1.0, 2.0, 3.0];
        let mut logits = original.clone();
        TopKWarper::new(3).warp(&mut logits, &[], 0);
        assert_eq!(logits, original);

        let mut logits2 = original.clone();
        TopKWarper::new(100).warp(&mut logits2, &[], 0);
        assert_eq!(logits2, original);
    }

    #[test]
    fn topk_one_keeps_max_only() {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0];
        TopKWarper::new(1).warp(&mut logits, &[], 0);
        assert!(logits[1].is_finite());
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    #[test]
    fn topk_preserves_original_values() {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        TopKWarper::new(3).warp(&mut logits, &[], 0);
        // Top 3 are 5.0, 4.0, 3.0
        assert!((logits[1] - 5.0).abs() < 1e-6);
        assert!((logits[4] - 4.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn topk_empty_logits() {
        let mut logits: Vec<f32> = vec![];
        TopKWarper::new(5).warp(&mut logits, &[], 0);
        assert!(logits.is_empty());
    }

    #[test]
    fn topk_with_negative_logits() {
        let mut logits = vec![-3.0, -1.0, -5.0, -2.0];
        TopKWarper::new(2).warp(&mut logits, &[], 0);
        assert!(logits[1].is_finite()); // -1.0
        assert!(logits[3].is_finite()); // -2.0
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    // ─── TopP ────────────────────────────────────────────────

    #[test]
    fn topp_filters_low_prob_tokens() {
        // logits: [1.0, 3.0, 0.5] → after softmax, token 1 dominates.
        let mut logits = vec![1.0, 3.0, 0.5];
        TopPWarper::new(0.8).warp(&mut logits, &[], 0);
        // Token 1 (highest prob) must survive.
        assert!(logits[1].is_finite());
        // At least one token should be masked.
        let masked = logits.iter().filter(|l| **l == f32::NEG_INFINITY).count();
        assert!(masked >= 1);
    }

    #[test]
    fn topp_one_is_noop() {
        let original = vec![1.0, 2.0, 3.0];
        let mut logits = original.clone();
        TopPWarper::new(1.0).warp(&mut logits, &[], 0);
        assert_eq!(logits, original);
    }

    #[test]
    fn topp_small_p_keeps_top_token() {
        let mut logits = vec![0.1, 10.0, 0.2, 0.3];
        TopPWarper::new(0.01).warp(&mut logits, &[], 0);
        // Token 1 (10.0) has nearly all probability mass.
        assert!(logits[1].is_finite());
    }

    #[test]
    fn topp_preserves_kept_values() {
        let mut logits = vec![1.0, 5.0, 0.5];
        let saved = logits[1];
        TopPWarper::new(0.5).warp(&mut logits, &[], 0);
        if logits[1].is_finite() {
            assert!((logits[1] - saved).abs() < 1e-6);
        }
    }

    #[test]
    fn topp_empty_logits() {
        let mut logits: Vec<f32> = vec![];
        TopPWarper::new(0.9).warp(&mut logits, &[], 0);
        assert!(logits.is_empty());
    }

    #[test]
    fn topp_uniform_distribution() {
        // 4 equal logits → equal probs ≈ 0.25 each.
        let mut logits = vec![1.0, 1.0, 1.0, 1.0];
        TopPWarper::new(0.5).warp(&mut logits, &[], 0);
        // Should keep about 2 tokens (0.25 + 0.25 = 0.50).
        let finite_count = logits.iter().filter(|l| l.is_finite()).count();
        assert!(finite_count >= 2);
    }

    // ─── MinP ────────────────────────────────────────────────

    #[test]
    fn minp_filters_low_prob() {
        let mut logits = vec![0.1, 5.0, 0.2, 4.5];
        MinPWarper::new(0.3).warp(&mut logits, &[], 0);
        // Tokens with prob < 0.3 * max_prob should be masked.
        assert!(logits[1].is_finite()); // high logit
        assert!(logits[3].is_finite()); // high logit
    }

    #[test]
    fn minp_zero_is_noop() {
        let original = vec![1.0, 2.0, 3.0];
        let mut logits = original.clone();
        MinPWarper::new(0.0).warp(&mut logits, &[], 0);
        assert_eq!(logits, original);
    }

    #[test]
    fn minp_never_removes_max_token() {
        let mut logits = vec![1.0, 5.0, 2.0, 0.1];
        MinPWarper::new(0.99).warp(&mut logits, &[], 0);
        assert!(logits[1].is_finite()); // 5.0 is the max logit
    }

    #[test]
    fn minp_empty_logits() {
        let mut logits: Vec<f32> = vec![];
        MinPWarper::new(0.5).warp(&mut logits, &[], 0);
        assert!(logits.is_empty());
    }

    #[test]
    fn minp_preserves_high_prob_values() {
        let mut logits = vec![0.1, 5.0, 4.8];
        let v1 = logits[1];
        let v2 = logits[2];
        MinPWarper::new(0.5).warp(&mut logits, &[], 0);
        if logits[1].is_finite() {
            assert!((logits[1] - v1).abs() < 1e-6);
        }
        if logits[2].is_finite() {
            assert!((logits[2] - v2).abs() < 1e-6);
        }
    }

    // ─── LogitBias ───────────────────────────────────────────

    #[test]
    fn bias_adds_to_logits() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        let bias = LogitBias::from_pairs([(1, 10.0), (3, -5.0)]);
        bias.warp(&mut logits, &[], 0);
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 12.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
        assert!((logits[3] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn bias_neg_inf_bans_token() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let bias = LogitBias::from_pairs([(1, f32::NEG_INFINITY)]);
        bias.warp(&mut logits, &[], 0);
        assert_eq!(logits[1], f32::NEG_INFINITY);
    }

    #[test]
    fn bias_out_of_range_ignored() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let bias = LogitBias::from_pairs([(100, 5.0)]);
        bias.warp(&mut logits, &[], 0);
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn bias_empty_map_is_noop() {
        let original = vec![1.0, 2.0, 3.0];
        let mut logits = original.clone();
        let bias = LogitBias::new(HashMap::new());
        bias.warp(&mut logits, &[], 0);
        assert_eq!(logits, original);
    }

    #[test]
    fn bias_zero_bias_is_noop() {
        let original = vec![1.0, 2.0, 3.0];
        let mut logits = original.clone();
        let bias = LogitBias::from_pairs([(0, 0.0), (1, 0.0)]);
        bias.warp(&mut logits, &[], 0);
        assert_eq!(logits, original);
    }

    #[test]
    fn bias_multiple_tokens() {
        let mut logits = [0.0; 10];
        let bias = LogitBias::from_pairs([(0, 1.0), (5, 2.0), (9, 3.0)]);
        bias.warp(&mut logits, &[], 0);
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[5] - 2.0).abs() < 1e-6);
        assert!((logits[9] - 3.0).abs() < 1e-6);
        assert!((logits[3] - 0.0).abs() < 1e-6);
    }

    // ─── NoRepeatNgram ───────────────────────────────────────

    #[test]
    fn ngram_blocks_repeated_bigram() {
        let mut logits = [1.0; 10];
        // History: [3, 5, 7, 3, 5] — bigram [3,5] appeared, so 7 should be banned
        let token_ids = vec![3, 5, 7, 3, 5];
        NoRepeatNgramWarper::new(2).warp(&mut logits, &token_ids, 0);
        // After [3,5] the next was 7, so 7 should be banned now that suffix is [5]
        // Wait — n=2, suffix is last 1 token = [5].
        // Scan: [3,5] at start=0: window=[3], but suffix is [5]. No match.
        //        [5,7] at start=1: window=[5], suffix=[5] → match! ban token 7.
        //        [7,3] at start=2: window=[7], suffix=[5]. No match.
        //        [3,5] at start=3: window=[3], suffix=[5]. No match.
        assert_eq!(logits[7], f32::NEG_INFINITY);
        // Others remain.
        assert!(logits[0].is_finite());
        assert!(logits[3].is_finite());
    }

    #[test]
    fn ngram_blocks_repeated_trigram() {
        let mut logits = [1.0; 10];
        // History: [1, 2, 3, 1, 2] — trigram [1,2,3] appeared, suffix=[1,2], ban 3
        let token_ids = vec![1, 2, 3, 1, 2];
        NoRepeatNgramWarper::new(3).warp(&mut logits, &token_ids, 0);
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    #[test]
    fn ngram_no_history_is_noop() {
        let original = vec![1.0, 2.0, 3.0];
        let mut logits = original.clone();
        NoRepeatNgramWarper::new(2).warp(&mut logits, &[], 0);
        assert_eq!(logits, original);
    }

    #[test]
    fn ngram_short_history_is_noop() {
        let original = vec![1.0, 2.0, 3.0];
        let mut logits = original.clone();
        // n=3 requires 2-token suffix, but history has only 1 token.
        NoRepeatNgramWarper::new(3).warp(&mut logits, &[5], 0);
        assert_eq!(logits, original);
    }

    #[test]
    fn ngram_no_repeat_found_is_noop() {
        let original = [1.0; 10];
        let mut logits = original.clone();
        // All unique tokens.
        let token_ids = vec![1, 2, 3, 4, 5];
        NoRepeatNgramWarper::new(2).warp(&mut logits, &token_ids, 0);
        // Last token is 5, no previous bigram starting with 5.
        assert_eq!(logits, original);
    }

    #[test]
    fn ngram_multiple_continuations_banned() {
        let mut logits = [1.0; 10];
        // History: [5, 3, 5, 7, 5] — suffix=[5], after 5 we saw 3 and 7.
        let token_ids = vec![5, 3, 5, 7, 5];
        NoRepeatNgramWarper::new(2).warp(&mut logits, &token_ids, 0);
        assert_eq!(logits[3], f32::NEG_INFINITY);
        assert_eq!(logits[7], f32::NEG_INFINITY);
    }

    #[test]
    fn ngram_does_not_ban_unrelated_tokens() {
        let mut logits = [1.0; 10];
        let token_ids = vec![1, 2, 3, 1, 2];
        NoRepeatNgramWarper::new(3).warp(&mut logits, &token_ids, 0);
        // Only 3 should be banned (trigram [1,2,3] repeat).
        assert_eq!(logits[3], f32::NEG_INFINITY);
        assert!(logits[0].is_finite());
        assert!(logits[1].is_finite());
        assert!(logits[2].is_finite());
        assert!(logits[4].is_finite());
    }

    // ─── ForcedToken ─────────────────────────────────────────

    #[test]
    fn forced_token_at_position() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        ForcedTokenWarper::single(5, 2).warp(&mut logits, &[], 5);
        assert!(logits[2].is_finite());
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    #[test]
    fn forced_token_wrong_position_is_noop() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let mut logits = original.clone();
        ForcedTokenWarper::single(5, 2).warp(&mut logits, &[], 3);
        assert_eq!(logits, original);
    }

    #[test]
    fn forced_token_preserves_value() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        ForcedTokenWarper::single(0, 3).warp(&mut logits, &[], 0);
        assert!((logits[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn forced_token_out_of_range_is_noop() {
        let original = vec![1.0, 2.0, 3.0];
        let mut logits = original.clone();
        ForcedTokenWarper::single(0, 100).warp(&mut logits, &[], 0);
        assert_eq!(logits, original);
    }

    #[test]
    fn forced_token_multiple_positions() {
        let forced = ForcedTokenWarper::new(HashMap::from([(0, 1), (3, 2)]));

        let mut logits1 = vec![1.0, 2.0, 3.0, 4.0];
        forced.warp(&mut logits1, &[], 0);
        assert!(logits1[1].is_finite());
        assert_eq!(logits1[0], f32::NEG_INFINITY);

        let mut logits2 = vec![1.0, 2.0, 3.0, 4.0];
        forced.warp(&mut logits2, &[], 3);
        assert!(logits2[2].is_finite());
        assert_eq!(logits2[0], f32::NEG_INFINITY);
    }

    // ─── Pipeline ────────────────────────────────────────────

    #[test]
    fn pipeline_empty_is_noop() {
        let original = vec![1.0, 2.0, 3.0];
        let mut logits = original.clone();
        let pipeline = WarperPipeline::new();
        pipeline.apply(&mut logits, &[], 0);
        assert_eq!(logits, original);
    }

    #[test]
    fn pipeline_applies_in_order() {
        let mut pipeline = WarperPipeline::new();
        // Temperature 0.5 → multiply by 2.
        pipeline.push(Box::new(TemperatureWarper::new(0.5)));
        // Then top-k=1 → keep only max.
        pipeline.push(Box::new(TopKWarper::new(1)));

        let mut logits = vec![1.0, 3.0, 2.0];
        pipeline.apply(&mut logits, &[], 0);

        // After temp 0.5: [2.0, 6.0, 4.0]. Top-k=1 keeps idx 1.
        assert!(logits[1].is_finite());
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn pipeline_order_matters() {
        // Demonstrate that bias + top-k order matters.
        // Pipeline A: bias on token 0 (+10), THEN top-k=2.
        let mut pipeline_a = WarperPipeline::new();
        pipeline_a.push(Box::new(LogitBias::from_pairs([(0, 10.0)])));
        pipeline_a.push(Box::new(TopKWarper::new(2)));

        // Pipeline B: top-k=2 THEN bias on token 0.
        let mut pipeline_b = WarperPipeline::new();
        pipeline_b.push(Box::new(TopKWarper::new(2)));
        pipeline_b.push(Box::new(LogitBias::from_pairs([(0, 10.0)])));

        let mut logits_a = vec![1.0, 5.0, 3.0, 2.0];
        pipeline_a.apply(&mut logits_a, &[], 0);

        let mut logits_b = vec![1.0, 5.0, 3.0, 2.0];
        pipeline_b.apply(&mut logits_b, &[], 0);

        // Pipeline A: bias → [11.0, 5.0, 3.0, 2.0] → topk=2 → keeps idx 0, 1.
        assert!(logits_a[0].is_finite());
        assert!(logits_a[1].is_finite());
        assert_eq!(logits_a[2], f32::NEG_INFINITY);

        // Pipeline B: topk=2 → keeps idx 1, 2 → bias → idx 0 stays -inf.
        assert_eq!(logits_b[0], f32::NEG_INFINITY);
        assert!(logits_b[1].is_finite());
        assert!(logits_b[2].is_finite());

        // Different sets of tokens are kept.
        let finite_a: Vec<usize> =
            logits_a.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(i, _)| i).collect();
        let finite_b: Vec<usize> =
            logits_b.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(i, _)| i).collect();
        assert_ne!(finite_a, finite_b);
    }

    #[test]
    fn pipeline_warper_names() {
        let mut pipeline = WarperPipeline::new();
        pipeline.push(Box::new(TemperatureWarper::new(0.8)));
        pipeline.push(Box::new(TopKWarper::new(50)));
        pipeline.push(Box::new(TopPWarper::new(0.9)));
        assert_eq!(pipeline.warper_names(), vec!["TemperatureWarper", "TopKWarper", "TopPWarper"]);
    }

    #[test]
    fn pipeline_len_and_is_empty() {
        let mut pipeline = WarperPipeline::new();
        assert!(pipeline.is_empty());
        assert_eq!(pipeline.len(), 0);
        pipeline.push(Box::new(TemperatureWarper::new(1.0)));
        assert!(!pipeline.is_empty());
        assert_eq!(pipeline.len(), 1);
    }

    #[test]
    fn pipeline_with_bias_and_topk() {
        let mut pipeline = WarperPipeline::new();
        pipeline.push(Box::new(LogitBias::from_pairs([(0, 100.0)])));
        pipeline.push(Box::new(TopKWarper::new(1)));

        let mut logits = vec![0.0, 5.0, 3.0];
        pipeline.apply(&mut logits, &[], 0);

        // Bias boosted idx 0 to 100.0, which is now the top-1.
        assert!(logits[0].is_finite());
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn pipeline_forced_overrides_everything() {
        // ForcedToken applied alone guarantees only the forced token survives.
        let mut pipeline = WarperPipeline::new();
        pipeline.push(Box::new(ForcedTokenWarper::single(0, 2)));

        let mut logits = vec![1.0, 5.0, 3.0, 4.0];
        pipeline.apply(&mut logits, &[], 0);

        assert!(logits[2].is_finite());
        assert!((logits[2] - 3.0).abs() < 1e-6);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    // ─── LogitStats ──────────────────────────────────────────

    #[test]
    fn stats_entropy_uniform() {
        // Uniform distribution of 4 tokens → entropy = ln(4) ≈ 1.386.
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        let stats = LogitStats::compute(&logits);
        assert!((stats.entropy - (4.0f32).ln()).abs() < 1e-4);
    }

    #[test]
    fn stats_entropy_peaked() {
        // Peaked distribution → low entropy.
        let logits = vec![0.0, 0.0, 100.0, 0.0];
        let stats = LogitStats::compute(&logits);
        assert!(stats.entropy < 0.1);
        assert!(stats.top1_confidence > 0.99);
        assert_eq!(stats.top1_token, 2);
    }

    #[test]
    fn stats_effective_vocab_size() {
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        let stats = LogitStats::compute(&logits);
        // effective_vocab_size = exp(entropy) ≈ 4.0 for uniform.
        assert!((stats.effective_vocab_size - 4.0).abs() < 0.1);
    }

    #[test]
    fn stats_max_min_logit() {
        let logits = vec![-2.0, 5.0, 3.0, -1.0];
        let stats = LogitStats::compute(&logits);
        assert!((stats.max_logit - 5.0).abs() < 1e-6);
        assert!((stats.min_logit - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn stats_empty_logits() {
        let stats = LogitStats::compute(&[]);
        assert!((stats.entropy - 0.0).abs() < 1e-6);
        assert_eq!(stats.top1_token, 0);
        assert!((stats.effective_vocab_size - 0.0).abs() < 1e-6);
    }

    #[test]
    fn stats_single_token() {
        let logits = [5.0];
        let stats = LogitStats::compute(&logits);
        assert!((stats.entropy - 0.0).abs() < 1e-5);
        assert!((stats.top1_confidence - 1.0).abs() < 1e-5);
        assert_eq!(stats.top1_token, 0);
        assert!((stats.effective_vocab_size - 1.0).abs() < 0.1);
    }

    #[test]
    fn stats_with_neg_inf_logits() {
        let logits = vec![f32::NEG_INFINITY, 3.0, f32::NEG_INFINITY, 3.0];
        let stats = LogitStats::compute(&logits);
        // Two equal finite tokens → entropy = ln(2).
        assert!((stats.entropy - (2.0f32).ln()).abs() < 1e-4);
        assert!((stats.top1_confidence - 0.5).abs() < 1e-4);
    }

    // ─── Softmax helper ──────────────────────────────────────

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_preserves_order() {
        let logits = vec![1.0, 3.0, 2.0];
        let probs = softmax(&logits);
        assert!(probs[1] > probs[2]);
        assert!(probs[2] > probs[0]);
    }

    #[test]
    fn softmax_neg_inf_becomes_zero() {
        let logits = vec![f32::NEG_INFINITY, 2.0, f32::NEG_INFINITY];
        let probs = softmax(&logits);
        assert!((probs[0] - 0.0).abs() < 1e-6);
        assert!((probs[1] - 1.0).abs() < 1e-5);
        assert!((probs[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_empty() {
        let probs = softmax(&[]);
        assert!(probs.is_empty());
    }

    // ─── Edge cases ──────────────────────────────────────────

    #[test]
    fn all_same_logits_topk() {
        let mut logits = [2.0; 5];
        TopKWarper::new(3).warp(&mut logits, &[], 0);
        let finite = logits.iter().filter(|v| v.is_finite()).count();
        assert!(finite >= 3);
    }

    #[test]
    fn all_same_logits_temperature() {
        let mut logits = [2.0; 5];
        TemperatureWarper::new(0.5).warp(&mut logits, &[], 0);
        // All should be 4.0.
        for l in &logits {
            assert!((*l - 4.0).abs() < 1e-6);
        }
    }

    #[test]
    fn single_logit_temperature_zero() {
        let mut logits = [3.0];
        TemperatureWarper::new(0.0).warp(&mut logits, &[], 0);
        // Single token stays finite.
        assert!(logits[0].is_finite());
    }

    #[test]
    fn large_vocab_topk() {
        let mut logits: Vec<f32> = (0..10000).map(|i| i as f32).collect();
        TopKWarper::new(10).warp(&mut logits, &[], 0);
        let finite = logits.iter().filter(|v| v.is_finite()).count();
        assert_eq!(finite, 10);
    }

    #[test]
    fn pipeline_ngram_then_topk() {
        let mut pipeline = WarperPipeline::new();
        pipeline.push(Box::new(NoRepeatNgramWarper::new(2)));
        pipeline.push(Box::new(TopKWarper::new(2)));

        let mut logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        // History: [1, 3, 1] → suffix=[1], ban 3.
        let token_ids = vec![1, 3, 1];
        pipeline.apply(&mut logits, &token_ids, 0);

        // Token 3 banned by ngram. Then top-k=2 of remaining.
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    // ─── Property tests ──────────────────────────────────────

    #[test]
    fn property_filtered_logits_le_original() {
        let original = vec![1.0f32, 5.0, 3.0, 2.0, 4.0, 0.5, 6.0, 1.5];
        let mut logits = original.clone();

        let mut pipeline = WarperPipeline::new();
        pipeline.push(Box::new(TopKWarper::new(4)));

        pipeline.apply(&mut logits, &[], 0);

        for (filtered, &orig) in logits.iter().zip(original.iter()) {
            assert!(*filtered <= orig, "filtered {filtered} > original {orig}");
        }
    }

    #[test]
    fn property_temperature_preserves_argmax_positive() {
        let logits = vec![0.5, 3.0, 1.5, 2.5, 0.1, 4.0];
        let orig_max =
            logits.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;

        for &temp in &[0.1, 0.5, 1.0, 2.0, 5.0] {
            let mut warped = logits.clone();
            TemperatureWarper::new(temp).warp(&mut warped, &[], 0);
            let new_max = warped
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
            assert_eq!(orig_max, new_max, "temp={temp} changed argmax");
        }
    }

    #[test]
    fn property_topk_keeps_exactly_k_finite() {
        for k in 1..=5 {
            let mut logits = vec![1.0, 5.0, 3.0, 2.0, 4.0, 0.5];
            TopKWarper::new(k).warp(&mut logits, &[], 0);
            let finite = logits.iter().filter(|v| v.is_finite()).count();
            assert_eq!(finite, k, "k={k} but got {finite} finite");
        }
    }

    #[test]
    fn property_topp_keeps_at_least_one() {
        for &p in &[0.01, 0.1, 0.5, 0.9, 0.99] {
            let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
            TopPWarper::new(p).warp(&mut logits, &[], 0);
            let finite = logits.iter().filter(|v| v.is_finite()).count();
            assert!(finite >= 1, "p={p} but no finite tokens left");
        }
    }

    #[test]
    fn property_minp_keeps_at_least_one() {
        for &min_p in &[0.01, 0.1, 0.5, 0.9, 0.99] {
            let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
            MinPWarper::new(min_p).warp(&mut logits, &[], 0);
            let finite = logits.iter().filter(|v| v.is_finite()).count();
            assert!(finite >= 1, "min_p={min_p} but no finite tokens left");
        }
    }

    #[test]
    fn property_forced_token_guarantees_selection() {
        for pos in 0..5 {
            for token in 0..4u32 {
                let mut logits = vec![1.0, 2.0, 3.0, 4.0];
                ForcedTokenWarper::single(pos, token).warp(&mut logits, &[], pos);
                let finite: Vec<usize> = logits
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v.is_finite())
                    .map(|(i, _)| i)
                    .collect();
                assert_eq!(finite, vec![token as usize]);
            }
        }
    }

    #[test]
    fn property_bias_is_additive() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let biases: Vec<(u32, f32)> = vec![(0, 1.5), (2, -0.5), (4, 3.0)];
        let bias = LogitBias::from_pairs(biases.clone());

        let mut logits = original.clone();
        bias.warp(&mut logits, &[], 0);

        for (id, b) in &biases {
            let idx = *id as usize;
            assert!(
                (logits[idx] - (original[idx] + b)).abs() < 1e-6,
                "bias not additive at idx {idx}"
            );
        }
    }

    #[test]
    fn property_ngram_idempotent() {
        let mut logits = [1.0; 10];
        let token_ids = vec![1, 2, 3, 1, 2];
        let warper = NoRepeatNgramWarper::new(3);

        warper.warp(&mut logits, &token_ids, 0);
        let after_first = logits.clone();

        warper.warp(&mut logits, &token_ids, 0);
        assert_eq!(logits, after_first, "ngram warp is not idempotent");
    }

    // ─── Trait object ────────────────────────────────────────

    #[test]
    fn warper_trait_is_object_safe() {
        let warpers: Vec<Box<dyn LogitWarper>> = vec![
            Box::new(TemperatureWarper::new(0.8)),
            Box::new(TopKWarper::new(50)),
            Box::new(TopPWarper::new(0.9)),
            Box::new(MinPWarper::new(0.1)),
            Box::new(LogitBias::new(HashMap::new())),
            Box::new(NoRepeatNgramWarper::new(3)),
            Box::new(ForcedTokenWarper::single(0, 1)),
        ];
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        for w in &warpers {
            w.warp(&mut logits, &[], 0);
        }
        // Just verify it doesn't panic; trait objects work.
    }

    #[test]
    fn warper_debug_output() {
        let w = TemperatureWarper::new(0.5);
        let debug = format!("{w:?}");
        assert!(debug.contains("TemperatureWarper"));
    }
}
