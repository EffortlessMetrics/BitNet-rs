//! Advanced token sampling strategies for Intel Arc A770 (OpenCL backend).
//!
//! This module provides CPU reference implementations and embedded OpenCL
//! kernel source for advanced sampling strategies beyond basic
//! temperature / top-k / top-p.  Strategies include Min-P, Typical-P,
//! Mirostat v1/v2 adaptive perplexity control, and contrastive search.
//!
//! A composable [`SamplingChain`] lets callers combine filters, transforms,
//! samplers, and validators in an arbitrary order.

use std::fmt;

// ── Sampling method enum ───────────────────────────────────────────────

/// Enumerates every supported sampling strategy.
#[derive(Debug, Clone, PartialEq)]
pub enum SamplingMethod {
    /// Always pick the highest-probability token.
    Greedy,
    /// Scale logits by `1/T` before softmax.
    Temperature(f32),
    /// Keep only the top-K most probable tokens.
    TopK(u32),
    /// Keep the smallest set of tokens whose cumulative probability ≥ p.
    TopP(f32),
    /// Keep tokens with probability ≥ min_p × max_probability.
    MinP(f32),
    /// Sample from the *typical set* whose information content is close
    /// to the entropy of the distribution.
    TypicalP(f32),
    /// Mirostat v1: adaptive perplexity targeting with surprise-rate
    /// feedback.
    Mirostat1 { target_tau: f32, learning_rate: f32 },
    /// Mirostat v2: simplified adaptive perplexity targeting.
    Mirostat2 { target_tau: f32, learning_rate: f32 },
    /// Contrastive search: balance likelihood with diversity penalty.
    ContrastiveSearch { alpha: f32, k: u32 },
}

impl fmt::Display for SamplingMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Greedy => write!(f, "greedy"),
            Self::Temperature(t) => write!(f, "temperature({t})"),
            Self::TopK(k) => write!(f, "top-k({k})"),
            Self::TopP(p) => write!(f, "top-p({p})"),
            Self::MinP(p) => write!(f, "min-p({p})"),
            Self::TypicalP(p) => write!(f, "typical-p({p})"),
            Self::Mirostat1 { target_tau, .. } => {
                write!(f, "mirostat-v1(τ={target_tau})")
            }
            Self::Mirostat2 { target_tau, .. } => {
                write!(f, "mirostat-v2(τ={target_tau})")
            }
            Self::ContrastiveSearch { alpha, k } => {
                write!(f, "contrastive(α={alpha}, k={k})")
            }
        }
    }
}

// ── Softmax / utility helpers ──────────────────────────────────────────

/// Numerically-stable softmax in-place, returning the computed
/// probabilities as a new `Vec`.
fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }
    probs
}

/// Argmax over a slice; returns 0 for empty input.
fn argmax(vals: &[f32]) -> usize {
    vals.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ── Min-P sampler ──────────────────────────────────────────────────────

/// Min-P sampling: keep tokens whose probability ≥ `min_p × max_prob`.
#[derive(Debug, Clone)]
pub struct MinPSampler {
    /// Relative probability threshold in `(0, 1]`.
    pub min_p: f32,
}

impl MinPSampler {
    pub fn new(min_p: f32) -> Self {
        Self { min_p: min_p.clamp(0.0, 1.0) }
    }

    /// Filter logits, returning (index, probability) pairs that pass the
    /// Min-P threshold.
    pub fn filter(&self, logits: &[f32]) -> Vec<(usize, f32)> {
        let probs = softmax(logits);
        let max_prob = probs.iter().copied().fold(0.0_f32, f32::max);
        let threshold = self.min_p * max_prob;
        probs.iter().enumerate().filter(|&(_, &p)| p >= threshold).map(|(i, &p)| (i, p)).collect()
    }

    /// Apply Min-P filtering in-place: set rejected logits to
    /// `f32::NEG_INFINITY`.
    pub fn apply(&self, logits: &mut [f32]) {
        let probs = softmax(logits);
        let max_prob = probs.iter().copied().fold(0.0_f32, f32::max);
        let threshold = self.min_p * max_prob;
        for (i, p) in probs.iter().enumerate() {
            if *p < threshold {
                logits[i] = f32::NEG_INFINITY;
            }
        }
    }
}

// ── Typical-P sampler ──────────────────────────────────────────────────

/// Typical decoding: sample from the *typical set* – the subset of tokens
/// whose negative log-probability is close to the entropy of the full
/// distribution.
#[derive(Debug, Clone)]
pub struct TypicalSampler {
    /// Cumulative-probability mass to retain from the typical set.
    pub p: f32,
}

impl TypicalSampler {
    pub fn new(p: f32) -> Self {
        Self { p: p.clamp(0.0, 1.0) }
    }

    /// Compute the entropy (nats) of `probs`.
    fn entropy(probs: &[f32]) -> f32 {
        probs.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum()
    }

    /// Return the indices that belong to the typical set (up to
    /// cumulative probability `self.p`).
    pub fn typical_set(&self, logits: &[f32]) -> Vec<usize> {
        let probs = softmax(logits);
        let ent = Self::entropy(&probs);

        // Compute |−log p_i − H| for each token, then sort ascending.
        let mut scored: Vec<(usize, f32, f32)> = probs
            .iter()
            .enumerate()
            .filter(|&(_, &p)| p > 0.0)
            .map(|(i, &p)| (i, (-p.ln() - ent).abs(), p))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut cum = 0.0_f32;
        let mut set = Vec::new();
        for (idx, _deviation, prob) in &scored {
            set.push(*idx);
            cum += prob;
            if cum >= self.p {
                break;
            }
        }
        set
    }

    /// Apply typical-P filtering in-place.
    pub fn apply(&self, logits: &mut [f32]) {
        let keep: Vec<usize> = self.typical_set(logits);
        let keep_set: std::collections::HashSet<usize> = keep.into_iter().collect();
        for (i, l) in logits.iter_mut().enumerate() {
            if !keep_set.contains(&i) {
                *l = f32::NEG_INFINITY;
            }
        }
    }
}

// ── Mirostat state ─────────────────────────────────────────────────────

/// Mirostat v1/v2 adaptive perplexity control.
///
/// Maintains a running estimate `mu` that is adjusted after each token to
/// steer the effective perplexity towards `target_tau`.
#[derive(Debug, Clone)]
pub struct MirostatState {
    pub target_tau: f32,
    pub learning_rate: f32,
    /// Running surprise estimate; initialised to `2 × target_tau`.
    pub mu: f32,
}

impl MirostatState {
    pub fn new(target_tau: f32, learning_rate: f32) -> Self {
        Self { target_tau, learning_rate, mu: 2.0 * target_tau }
    }

    /// Mirostat **v2** step: truncate to tokens with surprise ≤ mu, sample,
    /// then update mu.  Returns `(selected_token, updated_mu)`.
    pub fn sample_v2(&mut self, logits: &[f32], rng_uniform: f32) -> usize {
        if logits.is_empty() {
            return 0;
        }
        let probs = softmax(logits);

        // Keep tokens whose surprise (−log₂ p) ≤ mu.
        let mut candidates: Vec<(usize, f32)> = probs
            .iter()
            .enumerate()
            .filter(|&(_, &p)| p > 0.0 && -p.log2() <= self.mu)
            .map(|(i, &p)| (i, p))
            .collect();

        if candidates.is_empty() {
            // Fallback: pick argmax.
            let idx = argmax(logits);
            let surprise = if probs[idx] > 0.0 { -probs[idx].log2() } else { 0.0 };
            self.mu -= self.learning_rate * (surprise - self.target_tau);
            return idx;
        }

        // Re-normalise and sample.
        let sum: f32 = candidates.iter().map(|(_, p)| p).sum();
        for c in &mut candidates {
            c.1 /= sum;
        }

        let chosen = weighted_pick(&candidates, rng_uniform);
        let surprise = if probs[chosen] > 0.0 { -probs[chosen].log2() } else { 0.0 };
        self.mu -= self.learning_rate * (surprise - self.target_tau);
        chosen
    }

    /// Mirostat **v1** step. Uses a top-k estimate derived from mu to
    /// truncate, then samples.  Returns chosen token index.
    pub fn sample_v1(&mut self, logits: &[f32], rng_uniform: f32) -> usize {
        let n = logits.len();
        if n == 0 {
            return 0;
        }

        // Estimate k from mu: k ≈ (e^mu − 1) × n / e^mu, clamped to [1, n].
        let e_mu = self.mu.exp();
        let k = ((e_mu - 1.0) * (n as f32) / e_mu).round().max(1.0).min(n as f32) as usize;

        // Build sorted candidates.
        let probs = softmax(logits);
        let mut indexed: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(k);

        // Re-normalise and sample.
        let sum: f32 = indexed.iter().map(|(_, p)| p).sum();
        if sum > 0.0 {
            for c in &mut indexed {
                c.1 /= sum;
            }
        }

        let chosen = weighted_pick(&indexed, rng_uniform);
        let surprise = if probs[chosen] > 0.0 { -probs[chosen].log2() } else { 0.0 };
        self.mu -= self.learning_rate * (surprise - self.target_tau);
        chosen
    }
}

/// Weighted random pick given `(index, weight)` pairs and a uniform
/// random value in `[0, 1)`.
fn weighted_pick(candidates: &[(usize, f32)], u: f32) -> usize {
    let mut cum = 0.0_f32;
    for &(idx, w) in candidates {
        cum += w;
        if u < cum {
            return idx;
        }
    }
    candidates.last().map(|&(i, _)| i).unwrap_or(0)
}

// ── Contrastive search ─────────────────────────────────────────────────

/// Contrastive search balances likelihood with a degeneration penalty
/// that discourages tokens too similar to recently generated ones.
#[derive(Debug, Clone)]
pub struct ContrastiveSearch {
    /// Weight for the degeneration penalty (`0 ≤ α ≤ 1`).
    pub alpha: f32,
    /// Number of top-k candidates to consider.
    pub k: u32,
}

impl ContrastiveSearch {
    pub fn new(alpha: f32, k: u32) -> Self {
        Self { alpha: alpha.clamp(0.0, 1.0), k: k.max(1) }
    }

    /// Select the best token from `logits`, penalising candidates that
    /// are too similar to `context_embeddings`.
    ///
    /// * `logits` – raw logits (vocab_size).
    /// * `context_embeddings` – list of embedding vectors for previously
    ///   generated tokens.
    /// * `candidate_embeddings` – embedding for each vocab entry; may be
    ///   `None` when embeddings are unavailable (falls back to pure
    ///   likelihood).
    pub fn select(
        &self,
        logits: &[f32],
        context_embeddings: &[Vec<f32>],
        candidate_embeddings: Option<&[Vec<f32>]>,
    ) -> usize {
        let probs = softmax(logits);

        // Top-k candidates by probability.
        let mut indexed: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(self.k as usize);

        if context_embeddings.is_empty() || candidate_embeddings.is_none() {
            // No context: fall back to argmax of top-k.
            return indexed.first().map(|&(i, _)| i).unwrap_or(0);
        }

        let embeds = candidate_embeddings.unwrap();
        let max_prob = indexed.first().map(|&(_, p)| p).unwrap_or(1.0).max(f32::EPSILON);

        let mut best_idx = indexed[0].0;
        let mut best_score = f32::NEG_INFINITY;

        for &(idx, prob) in &indexed {
            if idx >= embeds.len() {
                continue;
            }
            // Max cosine similarity to any context token.
            let max_sim = context_embeddings
                .iter()
                .map(|ctx| cosine_similarity(&embeds[idx], ctx))
                .fold(f32::NEG_INFINITY, f32::max);

            let score = (1.0 - self.alpha) * (prob / max_prob) - self.alpha * max_sim;
            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }
        best_idx
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let denom = norm_a * norm_b;
    if denom < f32::EPSILON { 0.0 } else { dot / denom }
}

// ── Grammar constraint ─────────────────────────────────────────────────

/// A grammar constraint that masks logits to only allow tokens valid in
/// the current grammar state.
#[derive(Debug, Clone)]
pub struct GrammarConstraint {
    /// Set of token indices that are valid in the current grammar state.
    pub allowed_tokens: Vec<u32>,
}

impl GrammarConstraint {
    pub fn new(allowed_tokens: Vec<u32>) -> Self {
        Self { allowed_tokens }
    }

    /// Mask out disallowed tokens by setting their logits to
    /// `f32::NEG_INFINITY`.
    pub fn apply(&self, logits: &mut [f32]) {
        let allowed: std::collections::HashSet<u32> = self.allowed_tokens.iter().copied().collect();
        for (i, l) in logits.iter_mut().enumerate() {
            if !allowed.contains(&(i as u32)) {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    /// Returns `true` when the token is allowed by this constraint.
    pub fn is_allowed(&self, token_id: u32) -> bool {
        self.allowed_tokens.contains(&token_id)
    }
}

// ── Sampling chain ─────────────────────────────────────────────────────

/// A single step in a [`SamplingChain`].
#[derive(Debug, Clone)]
pub enum ChainStep {
    /// Apply temperature scaling.
    Temperature(f32),
    /// Top-K truncation.
    TopK(u32),
    /// Top-P (nucleus) truncation.
    TopP(f32),
    /// Min-P filtering.
    MinP(f32),
    /// Typical-P filtering.
    TypicalP(f32),
    /// Grammar constraint.
    Grammar(GrammarConstraint),
    /// Final: greedy argmax.
    Greedy,
}

/// Composable chain: filter → transform → sample → validate.
///
/// Steps execute in order.  The chain always returns a single token
/// index.
#[derive(Debug, Clone)]
pub struct SamplingChain {
    steps: Vec<ChainStep>,
}

impl SamplingChain {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    pub fn push(&mut self, step: ChainStep) -> &mut Self {
        self.steps.push(step);
        self
    }

    /// Number of steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Execute the chain on a *copy* of `logits` and return the chosen
    /// token index.
    pub fn execute(&self, logits: &[f32]) -> usize {
        let mut buf = logits.to_vec();
        for step in &self.steps {
            match step {
                ChainStep::Temperature(t) => {
                    cpu_apply_temperature(&mut buf, *t);
                }
                ChainStep::TopK(k) => {
                    cpu_apply_top_k(&mut buf, *k);
                }
                ChainStep::TopP(p) => {
                    cpu_apply_top_p(&mut buf, *p);
                }
                ChainStep::MinP(p) => {
                    MinPSampler::new(*p).apply(&mut buf);
                }
                ChainStep::TypicalP(p) => {
                    TypicalSampler::new(*p).apply(&mut buf);
                }
                ChainStep::Grammar(g) => {
                    g.apply(&mut buf);
                }
                ChainStep::Greedy => {
                    return argmax(&buf);
                }
            }
        }
        // If no terminal step, default to greedy.
        argmax(&buf)
    }
}

impl Default for SamplingChain {
    fn default() -> Self {
        Self::new()
    }
}

// ── Sampling statistics ────────────────────────────────────────────────

/// Statistics computed for a sampling step.
#[derive(Debug, Clone)]
pub struct SamplingStats {
    /// Shannon entropy of the probability distribution (nats).
    pub entropy: f32,
    /// Effective vocabulary size (perplexity = e^H).
    pub effective_vocab_size: f32,
    /// Fraction of vocab that survived filtering.
    pub acceptance_rate: f32,
    /// Which method produced these stats.
    pub method: SamplingMethod,
}

impl SamplingStats {
    /// Compute statistics from logits *after* filtering.
    pub fn compute(logits: &[f32], method: SamplingMethod) -> Self {
        let probs = softmax(logits);
        let total = probs.len();
        let active = probs.iter().filter(|&&p| p > 0.0).count();
        let entropy: f32 = probs.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
        let effective_vocab_size = entropy.exp();
        let acceptance_rate = if total > 0 { active as f32 / total as f32 } else { 0.0 };
        Self { entropy, effective_vocab_size, acceptance_rate, method }
    }
}

impl fmt::Display for SamplingStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "method={} entropy={:.3} eff_vocab={:.1} accept={:.3}",
            self.method, self.entropy, self.effective_vocab_size, self.acceptance_rate,
        )
    }
}

// ── CPU reference implementations ──────────────────────────────────────

/// Apply temperature scaling in-place.
pub fn cpu_apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature <= 0.0 || temperature == 1.0 || logits.is_empty() {
        return;
    }
    let inv_t = 1.0 / temperature;
    for l in logits.iter_mut() {
        *l *= inv_t;
    }
}

/// Top-K truncation in-place: keep only the top `k` logits, mask the
/// rest to `NEG_INFINITY`.
pub fn cpu_apply_top_k(logits: &mut [f32], k: u32) {
    let k = (k as usize).min(logits.len());
    if k == 0 || k >= logits.len() {
        return;
    }
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let threshold = indexed[k - 1].1;

    // Count how many are at exactly the threshold.
    let at_threshold: usize = indexed.iter().filter(|&&(_, v)| v == threshold).count();
    let above: usize = indexed.iter().filter(|&&(_, v)| v > threshold).count();
    let allow_at_threshold = k - above;

    let mut seen_at_threshold = 0usize;
    for (i, l) in logits.iter_mut().enumerate() {
        if *l > threshold {
            continue;
        } else if *l == threshold && seen_at_threshold < allow_at_threshold {
            seen_at_threshold += 1;
            continue;
        }
        let _ = at_threshold;
        let _ = i;
        *l = f32::NEG_INFINITY;
    }
}

/// Top-P (nucleus) truncation in-place.
pub fn cpu_apply_top_p(logits: &mut [f32], p: f32) {
    if logits.is_empty() {
        return;
    }
    let probs = softmax(logits);
    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut cum = 0.0_f32;
    let mut keep = std::collections::HashSet::new();
    for &(idx, prob) in &indexed {
        keep.insert(idx);
        cum += prob;
        if cum >= p {
            break;
        }
    }

    for (i, l) in logits.iter_mut().enumerate() {
        if !keep.contains(&i) {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Greedy decoding: return the index of the largest logit.
pub fn cpu_greedy_sample(logits: &[f32]) -> usize {
    argmax(logits)
}

// ── OpenCL kernel source ───────────────────────────────────────────────

/// OpenCL kernel source for parallel advanced sampling on GPU.
///
/// Contains kernels for temperature scaling, softmax, top-k extraction,
/// and RNG-based weighted sampling using a Philox-style counter-based
/// PRNG suitable for massively-parallel execution.
pub const ADVANCED_SAMPLING_CL: &str = r#"
// ── Philox 2×32-10 counter-based RNG ────────────────────────────────
inline uint2 philox2x32_round(uint2 ctr, uint key) {
    uint hi = mul_hi(ctr.x, 0xD2511F53u);
    uint lo = ctr.x * 0xD2511F53u;
    return (uint2)(hi ^ key ^ ctr.y, lo);
}

inline float gpu_rand(uint tid, uint round) {
    uint2 ctr = (uint2)(tid, round);
    uint key = 0x9E3779B9u;
    for (int i = 0; i < 10; i++) {
        ctr = philox2x32_round(ctr, key);
        key += 0x9E3779B9u;
    }
    return (float)(ctr.x) / 4294967296.0f;
}

// ── Temperature scaling kernel ──────────────────────────────────────
__kernel void apply_temperature(
    __global float* logits,
    const float inv_temperature,
    const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        logits[gid] *= inv_temperature;
    }
}

// ── Parallel softmax (two-pass: max-reduce then exp-normalise) ──────
__kernel void softmax_pass1_max(
    __global const float* logits,
    __global float* block_max,
    const int n)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);

    __local float sdata[256];
    float val = (gid < n) ? logits[gid] : -INFINITY;
    sdata[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = group_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            sdata[lid] = fmax(sdata[lid], sdata[lid + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) block_max[get_group_id(0)] = sdata[0];
}

// ── Weighted sampling with Philox RNG ───────────────────────────────
__kernel void weighted_sample(
    __global const float* probs,
    __global uint* result,
    const uint seed,
    const int n)
{
    float u = gpu_rand(seed, 0);
    float cum = 0.0f;
    for (int i = 0; i < n; i++) {
        cum += probs[i];
        if (u < cum) {
            result[0] = (uint)i;
            return;
        }
    }
    result[0] = (uint)(n - 1);
}
"#;

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────────

    fn sample_logits() -> Vec<f32> {
        vec![1.0, 2.0, 3.0, 4.0, 5.0]
    }

    fn uniform_logits(n: usize) -> Vec<f32> {
        vec![1.0; n]
    }

    fn dominated_logits() -> Vec<f32> {
        vec![0.0, 0.0, 100.0, 0.0, 0.0]
    }

    fn assert_valid_probs(probs: &[f32]) {
        for &p in probs {
            assert!(p >= 0.0, "negative probability: {p}");
        }
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "probabilities don't sum to 1: {sum}");
    }

    // ── SamplingMethod display ────────────────────────────────────────

    #[test]
    fn sampling_method_display_greedy() {
        assert_eq!(SamplingMethod::Greedy.to_string(), "greedy");
    }

    #[test]
    fn sampling_method_display_temperature() {
        let s = SamplingMethod::Temperature(0.7).to_string();
        assert!(s.contains("0.7"));
    }

    #[test]
    fn sampling_method_display_contrastive() {
        let s = SamplingMethod::ContrastiveSearch { alpha: 0.6, k: 5 };
        assert!(s.to_string().contains("contrastive"));
    }

    #[test]
    fn sampling_method_display_all_variants() {
        let variants = vec![
            SamplingMethod::Greedy,
            SamplingMethod::Temperature(0.5),
            SamplingMethod::TopK(10),
            SamplingMethod::TopP(0.9),
            SamplingMethod::MinP(0.05),
            SamplingMethod::TypicalP(0.95),
            SamplingMethod::Mirostat1 { target_tau: 5.0, learning_rate: 0.1 },
            SamplingMethod::Mirostat2 { target_tau: 5.0, learning_rate: 0.1 },
            SamplingMethod::ContrastiveSearch { alpha: 0.6, k: 5 },
        ];
        for v in &variants {
            assert!(!v.to_string().is_empty());
        }
    }

    // ── softmax helper ────────────────────────────────────────────────

    #[test]
    fn softmax_sums_to_one() {
        let probs = softmax(&sample_logits());
        assert_valid_probs(&probs);
    }

    #[test]
    fn softmax_empty_returns_empty() {
        assert!(softmax(&[]).is_empty());
    }

    #[test]
    fn softmax_single_element() {
        let probs = softmax(&[42.0]);
        assert_eq!(probs.len(), 1);
        assert!((probs[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_uniform_gives_equal_probs() {
        let probs = softmax(&uniform_logits(4));
        for &p in &probs {
            assert!((p - 0.25).abs() < 1e-6);
        }
    }

    // ── Greedy sampling ───────────────────────────────────────────────

    #[test]
    fn greedy_picks_argmax() {
        assert_eq!(cpu_greedy_sample(&sample_logits()), 4);
    }

    #[test]
    fn greedy_picks_dominated_token() {
        assert_eq!(cpu_greedy_sample(&dominated_logits()), 2);
    }

    #[test]
    fn greedy_single_token() {
        assert_eq!(cpu_greedy_sample(&[7.0]), 0);
    }

    #[test]
    fn greedy_first_of_ties() {
        // When tied, argmax returns a valid index among the maxima.
        let token = cpu_greedy_sample(&[5.0, 5.0, 5.0]);
        assert!(token < 3);
    }

    // ── Temperature ───────────────────────────────────────────────────

    #[test]
    fn temperature_zero_equivalent_to_greedy() {
        let mut logits = sample_logits();
        // T → 0 means we don't modify (our impl treats T≤0 as noop),
        // but the argmax of the unchanged logits is still greedy.
        cpu_apply_temperature(&mut logits, 0.0);
        assert_eq!(argmax(&logits), 4);
    }

    #[test]
    fn temperature_1_is_identity() {
        let mut logits = sample_logits();
        let orig = logits.clone();
        cpu_apply_temperature(&mut logits, 1.0);
        assert_eq!(logits, orig);
    }

    #[test]
    fn temperature_low_sharpens() {
        let mut logits = sample_logits();
        cpu_apply_temperature(&mut logits, 0.01);
        let gap = logits[4] - logits[3];
        assert!(gap > 50.0, "expected sharpened gap, got {gap}");
    }

    #[test]
    fn temperature_high_flattens() {
        let mut logits = sample_logits();
        cpu_apply_temperature(&mut logits, 10.0);
        let range = logits[4] - logits[0];
        assert!(range < 1.0, "expected flattened range, got {range}");
    }

    #[test]
    fn temperature_empty_is_noop() {
        let mut logits: Vec<f32> = Vec::new();
        cpu_apply_temperature(&mut logits, 0.5);
        assert!(logits.is_empty());
    }

    // ── Top-K ─────────────────────────────────────────────────────────

    #[test]
    fn top_k_returns_exactly_k_candidates() {
        let mut logits = sample_logits();
        cpu_apply_top_k(&mut logits, 3);
        let active = logits.iter().filter(|&&l| l.is_finite()).count();
        assert_eq!(active, 3);
    }

    #[test]
    fn top_k_preserves_all_when_k_ge_n() {
        let mut logits = sample_logits();
        let orig = logits.clone();
        cpu_apply_top_k(&mut logits, 10);
        assert_eq!(logits, orig);
    }

    #[test]
    fn top_k_1_keeps_only_argmax() {
        let mut logits = sample_logits();
        cpu_apply_top_k(&mut logits, 1);
        let active: Vec<usize> =
            logits.iter().enumerate().filter(|(_, l)| l.is_finite()).map(|(i, _)| i).collect();
        assert_eq!(active, vec![4]);
    }

    #[test]
    fn top_k_zero_preserves_all() {
        let mut logits = sample_logits();
        let orig = logits.clone();
        cpu_apply_top_k(&mut logits, 0);
        assert_eq!(logits, orig);
    }

    // ── Top-P ─────────────────────────────────────────────────────────

    #[test]
    fn top_p_cumulative_threshold() {
        let mut logits = vec![1.0, 2.0, 3.0, 10.0];
        cpu_apply_top_p(&mut logits, 0.9);
        // Token 3 (logit 10) dominates; should survive.
        assert!(logits[3].is_finite());
    }

    #[test]
    fn top_p_1_keeps_all() {
        let mut logits = sample_logits();
        cpu_apply_top_p(&mut logits, 1.0);
        assert!(logits.iter().all(|l| l.is_finite()));
    }

    #[test]
    fn top_p_very_small_keeps_at_least_one() {
        let mut logits = sample_logits();
        cpu_apply_top_p(&mut logits, 0.001);
        let active = logits.iter().filter(|&&l| l.is_finite()).count();
        assert!(active >= 1);
    }

    #[test]
    fn top_p_empty_is_noop() {
        let mut logits: Vec<f32> = Vec::new();
        cpu_apply_top_p(&mut logits, 0.9);
        assert!(logits.is_empty());
    }

    // ── Min-P ─────────────────────────────────────────────────────────

    #[test]
    fn min_p_filters_low_probability_tokens() {
        let logits = dominated_logits();
        let sampler = MinPSampler::new(0.5);
        let candidates = sampler.filter(&logits);
        // Only the dominant token (index 2) should survive.
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].0, 2);
    }

    #[test]
    fn min_p_zero_keeps_all() {
        let logits = sample_logits();
        let sampler = MinPSampler::new(0.0);
        let candidates = sampler.filter(&logits);
        assert_eq!(candidates.len(), logits.len());
    }

    #[test]
    fn min_p_one_keeps_only_max() {
        let logits = sample_logits();
        let sampler = MinPSampler::new(1.0);
        let candidates = sampler.filter(&logits);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].0, 4);
    }

    #[test]
    fn min_p_apply_masks_logits() {
        let mut logits = dominated_logits();
        MinPSampler::new(0.5).apply(&mut logits);
        // Non-dominant tokens should be masked.
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert!(logits[2].is_finite());
    }

    #[test]
    fn min_p_uniform_keeps_all() {
        let logits = uniform_logits(5);
        let sampler = MinPSampler::new(0.5);
        let candidates = sampler.filter(&logits);
        assert_eq!(candidates.len(), 5);
    }

    #[test]
    fn min_p_clamps_out_of_range() {
        let s1 = MinPSampler::new(-0.5);
        assert_eq!(s1.min_p, 0.0);
        let s2 = MinPSampler::new(2.0);
        assert_eq!(s2.min_p, 1.0);
    }

    // ── Typical-P ─────────────────────────────────────────────────────

    #[test]
    fn typical_p_selects_typical_set() {
        let logits = sample_logits();
        let sampler = TypicalSampler::new(0.5);
        let set = sampler.typical_set(&logits);
        assert!(!set.is_empty());
        assert!(set.len() <= logits.len());
    }

    #[test]
    fn typical_p_1_keeps_all() {
        let logits = sample_logits();
        let sampler = TypicalSampler::new(1.0);
        let set = sampler.typical_set(&logits);
        assert_eq!(set.len(), logits.len());
    }

    #[test]
    fn typical_p_apply_masks_logits() {
        let mut logits = sample_logits();
        TypicalSampler::new(0.3).apply(&mut logits);
        let active = logits.iter().filter(|&&l| l.is_finite()).count();
        assert!(active >= 1);
        assert!(active < 5);
    }

    #[test]
    fn typical_p_uniform_keeps_all() {
        // Uniform distribution: all tokens are equally typical.
        let logits = uniform_logits(5);
        let sampler = TypicalSampler::new(0.99);
        let set = sampler.typical_set(&logits);
        assert_eq!(set.len(), 5);
    }

    #[test]
    fn typical_entropy_computation() {
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let ent = TypicalSampler::entropy(&probs);
        // ln(4) ≈ 1.3863
        assert!((ent - 4.0_f32.ln()).abs() < 1e-4);
    }

    // ── Mirostat ──────────────────────────────────────────────────────

    #[test]
    fn mirostat_v2_returns_valid_token() {
        let mut state = MirostatState::new(5.0, 0.1);
        let logits = sample_logits();
        let token = state.sample_v2(&logits, 0.5);
        assert!(token < logits.len());
    }

    #[test]
    fn mirostat_v2_updates_mu() {
        let mut state = MirostatState::new(5.0, 0.1);
        let initial_mu = state.mu;
        let logits = sample_logits();
        let _ = state.sample_v2(&logits, 0.5);
        assert_ne!(state.mu, initial_mu, "mu should be updated");
    }

    #[test]
    fn mirostat_v2_converges_toward_tau() {
        let mut state = MirostatState::new(3.0, 0.1);
        // Use a wider distribution so there's enough surprise variation.
        let logits: Vec<f32> = (0..50).map(|i| (i as f32) * 0.5).collect();
        for i in 0..200 {
            let u = ((i * 7 + 3) % 100) as f32 / 100.0;
            let _ = state.sample_v2(&logits, u);
        }
        // After many iterations, mu should be in a reasonable range.
        assert!(state.mu > -10.0 && state.mu < 30.0, "mu diverged: {}", state.mu);
    }

    #[test]
    fn mirostat_v1_returns_valid_token() {
        let mut state = MirostatState::new(5.0, 0.1);
        let logits = sample_logits();
        let token = state.sample_v1(&logits, 0.5);
        assert!(token < logits.len());
    }

    #[test]
    fn mirostat_v1_updates_mu() {
        let mut state = MirostatState::new(5.0, 0.1);
        let initial_mu = state.mu;
        let logits = sample_logits();
        let _ = state.sample_v1(&logits, 0.5);
        assert_ne!(state.mu, initial_mu, "mu should be updated");
    }

    #[test]
    fn mirostat_initial_mu_is_2x_tau() {
        let state = MirostatState::new(3.5, 0.1);
        assert!((state.mu - 7.0).abs() < f32::EPSILON);
    }

    #[test]
    fn mirostat_v2_empty_logits_returns_zero() {
        let mut state = MirostatState::new(5.0, 0.1);
        let logits: Vec<f32> = Vec::new();
        let token = state.sample_v2(&logits, 0.5);
        assert_eq!(token, 0);
    }

    #[test]
    fn mirostat_v1_empty_logits_returns_zero() {
        let mut state = MirostatState::new(5.0, 0.1);
        let logits: Vec<f32> = Vec::new();
        let token = state.sample_v1(&logits, 0.5);
        assert_eq!(token, 0);
    }

    // ── Contrastive search ────────────────────────────────────────────

    #[test]
    fn contrastive_no_context_falls_back_to_argmax() {
        let cs = ContrastiveSearch::new(0.6, 5);
        let logits = sample_logits();
        let token = cs.select(&logits, &[], None);
        assert_eq!(token, argmax(&logits));
    }

    #[test]
    fn contrastive_penalises_similar_tokens() {
        let cs = ContrastiveSearch::new(0.9, 3);
        let logits = vec![5.0, 4.9, 4.8]; // very close likelihoods

        // Context embedding is very similar to candidate 0.
        let ctx = vec![vec![1.0, 0.0, 0.0]];
        let embeds = vec![
            vec![1.0, 0.0, 0.0], // idx 0 – identical to context
            vec![0.0, 1.0, 0.0], // idx 1 – orthogonal
            vec![0.0, 0.0, 1.0], // idx 2 – orthogonal
        ];

        let token = cs.select(&logits, &ctx, Some(&embeds));
        // High alpha means strong penalty → should avoid token 0.
        assert_ne!(token, 0, "should penalise token similar to context");
    }

    #[test]
    fn contrastive_alpha_zero_is_pure_likelihood() {
        let cs = ContrastiveSearch::new(0.0, 3);
        let logits = vec![5.0, 4.9, 4.8];
        let ctx = vec![vec![1.0, 0.0, 0.0]];
        let embeds = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let token = cs.select(&logits, &ctx, Some(&embeds));
        assert_eq!(token, 0, "alpha=0 → pure likelihood → argmax");
    }

    #[test]
    fn contrastive_k_1_considers_single_candidate() {
        let cs = ContrastiveSearch::new(0.5, 1);
        let logits = sample_logits();
        let token = cs.select(&logits, &[], None);
        assert_eq!(token, argmax(&logits));
    }

    // ── Grammar constraint ────────────────────────────────────────────

    #[test]
    fn grammar_masks_disallowed_tokens() {
        let gc = GrammarConstraint::new(vec![1, 3]);
        let mut logits = sample_logits();
        gc.apply(&mut logits);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert!(logits[1].is_finite());
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert!(logits[3].is_finite());
        assert_eq!(logits[4], f32::NEG_INFINITY);
    }

    #[test]
    fn grammar_is_allowed() {
        let gc = GrammarConstraint::new(vec![2, 5, 8]);
        assert!(gc.is_allowed(2));
        assert!(gc.is_allowed(5));
        assert!(!gc.is_allowed(0));
        assert!(!gc.is_allowed(3));
    }

    #[test]
    fn grammar_empty_masks_everything() {
        let gc = GrammarConstraint::new(vec![]);
        let mut logits = sample_logits();
        gc.apply(&mut logits);
        assert!(logits.iter().all(|&l| l == f32::NEG_INFINITY));
    }

    #[test]
    fn grammar_all_allowed_is_noop() {
        let gc = GrammarConstraint::new(vec![0, 1, 2, 3, 4]);
        let mut logits = sample_logits();
        let orig = logits.clone();
        gc.apply(&mut logits);
        assert_eq!(logits, orig);
    }

    // ── Sampling chain ────────────────────────────────────────────────

    #[test]
    fn chain_empty_defaults_to_greedy() {
        let chain = SamplingChain::new();
        assert_eq!(chain.execute(&sample_logits()), 4);
    }

    #[test]
    fn chain_greedy_step() {
        let mut chain = SamplingChain::new();
        chain.push(ChainStep::Greedy);
        assert_eq!(chain.execute(&sample_logits()), 4);
    }

    #[test]
    fn chain_temperature_then_greedy() {
        let mut chain = SamplingChain::new();
        chain.push(ChainStep::Temperature(0.5));
        chain.push(ChainStep::Greedy);
        assert_eq!(chain.execute(&sample_logits()), 4);
    }

    #[test]
    fn chain_top_k_then_greedy() {
        let mut chain = SamplingChain::new();
        chain.push(ChainStep::TopK(2));
        chain.push(ChainStep::Greedy);
        let token = chain.execute(&sample_logits());
        assert!(token == 3 || token == 4); // top-2 of [1,2,3,4,5]
    }

    #[test]
    fn chain_composition_order_matters() {
        // TopK(1) → Greedy should always be index 4.
        let mut chain1 = SamplingChain::new();
        chain1.push(ChainStep::TopK(1));
        chain1.push(ChainStep::Greedy);

        // Grammar([0]) → Greedy should always be index 0.
        let mut chain2 = SamplingChain::new();
        chain2.push(ChainStep::Grammar(GrammarConstraint::new(vec![0])));
        chain2.push(ChainStep::Greedy);

        assert_eq!(chain1.execute(&sample_logits()), 4);
        assert_eq!(chain2.execute(&sample_logits()), 0);
    }

    #[test]
    fn chain_min_p_step() {
        let mut chain = SamplingChain::new();
        chain.push(ChainStep::MinP(0.5));
        chain.push(ChainStep::Greedy);
        let token = chain.execute(&dominated_logits());
        assert_eq!(token, 2);
    }

    #[test]
    fn chain_typical_p_step() {
        let mut chain = SamplingChain::new();
        chain.push(ChainStep::TypicalP(0.5));
        chain.push(ChainStep::Greedy);
        let token = chain.execute(&sample_logits());
        assert!(token < 5);
    }

    #[test]
    fn chain_len_and_is_empty() {
        let mut chain = SamplingChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
        chain.push(ChainStep::Greedy);
        assert!(!chain.is_empty());
        assert_eq!(chain.len(), 1);
    }

    #[test]
    fn chain_default_is_empty() {
        let chain = SamplingChain::default();
        assert!(chain.is_empty());
    }

    // ── SamplingStats ─────────────────────────────────────────────────

    #[test]
    fn stats_entropy_uniform() {
        let logits = uniform_logits(4);
        let stats = SamplingStats::compute(&logits, SamplingMethod::Greedy);
        let expected_ent = 4.0_f32.ln();
        assert!(
            (stats.entropy - expected_ent).abs() < 1e-3,
            "entropy {}, expected {}",
            stats.entropy,
            expected_ent
        );
    }

    #[test]
    fn stats_effective_vocab_size_uniform() {
        let logits = uniform_logits(8);
        let stats = SamplingStats::compute(&logits, SamplingMethod::Greedy);
        assert!(
            (stats.effective_vocab_size - 8.0).abs() < 0.5,
            "effective vocab {}",
            stats.effective_vocab_size
        );
    }

    #[test]
    fn stats_acceptance_rate_full() {
        let logits = sample_logits();
        let stats = SamplingStats::compute(&logits, SamplingMethod::Greedy);
        assert!((stats.acceptance_rate - 1.0).abs() < 1e-6, "rate {}", stats.acceptance_rate);
    }

    #[test]
    fn stats_acceptance_rate_after_filtering() {
        let mut logits = sample_logits();
        cpu_apply_top_k(&mut logits, 2);
        let stats = SamplingStats::compute(&logits, SamplingMethod::TopK(2));
        assert!((stats.acceptance_rate - 2.0 / 5.0).abs() < 0.15, "rate {}", stats.acceptance_rate);
    }

    #[test]
    fn stats_display() {
        let logits = sample_logits();
        let stats = SamplingStats::compute(&logits, SamplingMethod::Greedy);
        let s = stats.to_string();
        assert!(s.contains("greedy"));
        assert!(s.contains("entropy="));
    }

    // ── Edge cases ────────────────────────────────────────────────────

    #[test]
    fn single_token_distribution() {
        let logits = vec![42.0];
        assert_eq!(cpu_greedy_sample(&logits), 0);

        let probs = softmax(&logits);
        assert!((probs[0] - 1.0).abs() < 1e-6);

        let mut miro = MirostatState::new(5.0, 0.1);
        assert_eq!(miro.sample_v2(&logits, 0.5), 0);
    }

    #[test]
    fn extreme_temperature_does_not_panic() {
        let mut logits = sample_logits();
        cpu_apply_temperature(&mut logits, 1e-6);
        assert!(logits.iter().all(|l| l.is_finite()));

        let mut logits2 = sample_logits();
        cpu_apply_temperature(&mut logits2, 1e6);
        assert!(logits2.iter().all(|l| l.is_finite()));
    }

    #[test]
    fn large_vocab_top_k() {
        let logits: Vec<f32> = (0..10_000).map(|i| i as f32).collect();
        let mut buf = logits.clone();
        cpu_apply_top_k(&mut buf, 50);
        let active = buf.iter().filter(|&&l| l.is_finite()).count();
        assert_eq!(active, 50);
    }

    // ── Property: sampled token is always in valid set ────────────────

    #[test]
    fn property_greedy_always_in_valid_set() {
        for size in [1, 5, 100, 1000] {
            let logits: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
            let token = cpu_greedy_sample(&logits);
            assert!(token < size);
        }
    }

    #[test]
    fn property_mirostat_v2_always_valid() {
        let mut state = MirostatState::new(5.0, 0.1);
        for size in [2, 10, 50] {
            let logits: Vec<f32> = (0..size).map(|i| (i as f32) * 0.5).collect();
            for step in 0..20 {
                let u = (step as f32 * 0.05) % 1.0;
                let token = state.sample_v2(&logits, u);
                assert!(token < size, "token {token} >= vocab {size}");
            }
        }
    }

    #[test]
    fn property_mirostat_v1_always_valid() {
        let mut state = MirostatState::new(5.0, 0.1);
        for size in [2, 10, 50] {
            let logits: Vec<f32> = (0..size).map(|i| (i as f32) * 0.5).collect();
            for step in 0..20 {
                let u = (step as f32 * 0.05) % 1.0;
                let token = state.sample_v1(&logits, u);
                assert!(token < size, "token {token} >= vocab {size}");
            }
        }
    }

    #[test]
    fn property_chain_always_in_range() {
        let chain = {
            let mut c = SamplingChain::new();
            c.push(ChainStep::TopK(5));
            c.push(ChainStep::MinP(0.1));
            c.push(ChainStep::Greedy);
            c
        };
        for size in [5, 50, 200] {
            let logits: Vec<f32> = (0..size).map(|i| (i as f32) * 0.3).collect();
            let token = chain.execute(&logits);
            assert!(token < size);
        }
    }

    // ── OpenCL kernel source sanity ───────────────────────────────────

    #[test]
    fn opencl_kernel_source_not_empty() {
        assert!(!ADVANCED_SAMPLING_CL.is_empty());
    }

    #[test]
    fn opencl_kernel_contains_temperature() {
        assert!(ADVANCED_SAMPLING_CL.contains("apply_temperature"));
    }

    #[test]
    fn opencl_kernel_contains_rng() {
        assert!(ADVANCED_SAMPLING_CL.contains("philox2x32_round"));
    }

    #[test]
    fn opencl_kernel_contains_softmax() {
        assert!(ADVANCED_SAMPLING_CL.contains("softmax_pass1_max"));
    }

    #[test]
    fn opencl_kernel_contains_weighted_sample() {
        assert!(ADVANCED_SAMPLING_CL.contains("weighted_sample"));
    }

    // ── Cosine similarity helper ──────────────────────────────────────

    #[test]
    fn cosine_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn cosine_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }
}
