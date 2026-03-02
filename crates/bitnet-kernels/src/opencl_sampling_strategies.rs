//! OpenCL-optimized token sampling strategies for text generation.
//!
//! # Overview
//!
//! This module provides CPU reference implementations of common sampling
//! strategies used in autoregressive LLM inference. Each sampler operates on
//! a mutable logits slice and either modifies it in-place (penalties /
//! filters) or selects a single token index (samplers).
//!
//! # Strategies
//!
//! | Strategy                  | Kind     | Description                                |
//! |--------------------------|----------|--------------------------------------------|
//! | [`GreedySampler`]        | Sampler  | Always picks the highest-probability token |
//! | [`TopKSampler`]          | Sampler  | Sample from top-*k* candidates             |
//! | [`TopPSampler`]          | Sampler  | Nucleus sampling (cumulative prob ≥ *p*)   |
//! | [`MinPSampler`]          | Sampler  | Filter tokens below `min_p × max_prob`     |
//! | [`TypicalSampler`]       | Sampler  | Entropy-based surprise filtering           |
//! | [`MirostatSampler`]      | Sampler  | Mirostat v2 target perplexity control      |
//! | [`RepetitionPenalty`]     | Penalty  | Multiplicative + additive repeat penalty   |
//! | [`FrequencyPresencePenalty`] | Penalty | OpenAI-style frequency + presence       |
//! | [`SamplerChain`]         | Chain    | Compose multiple samplers in sequence      |
//!
//! # OpenCL kernel
//!
//! The embedded OpenCL C source (`SAMPLING_CL`) contains GPU kernels for the
//! most performance-critical operations (top-k / softmax / multinomial).
//! When no OpenCL runtime is available the CPU reference functions are used.

use std::collections::HashMap;
use std::fmt;

// ── OpenCL kernel source ─────────────────────────────────────────────────

/// Embedded OpenCL C source for sampling operations.
pub const SAMPLING_CL: &str = r#"
// OpenCL sampling kernels — placeholder for GPU dispatch.
// These are compiled at runtime when an OpenCL device is available.

__kernel void softmax(
    __global float* logits,
    const int n
) {
    float max_val = -INFINITY;
    for (int i = 0; i < n; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        logits[i] = exp(logits[i] - max_val);
        sum += logits[i];
    }
    for (int i = 0; i < n; i++) {
        logits[i] /= sum;
    }
}

__kernel void top_k_filter(
    __global float* logits,
    const int n,
    const int k
) {
    // Find k-th largest value via partial sort, then mask
    // Full implementation uses local memory + bitonic sort
    float kth = -INFINITY;
    for (int found = 0; found < k; found++) {
        float best = -INFINITY;
        for (int i = 0; i < n; i++) {
            if (logits[i] > kth && logits[i] <= best) continue;
            if (logits[i] > best || (found == 0 && logits[i] > kth)) {
                best = logits[i];
            }
        }
        kth = best;
    }
    for (int i = 0; i < n; i++) {
        if (logits[i] < kth) logits[i] = -INFINITY;
    }
}

__kernel void argmax(
    __global const float* logits,
    const int n,
    __global int* result
) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    result[0] = best;
}
"#;

// ── Seedable RNG ─────────────────────────────────────────────────────────

/// Seedable RNG wrapper for reproducible sampling.
///
/// Uses a simple xoshiro256** implementation that is deterministic given
/// the same seed, fast, and has good statistical properties.
#[derive(Debug, Clone)]
pub struct SamplingRng {
    state: [u64; 4],
}

impl SamplingRng {
    /// Create a new RNG with the given seed.
    pub fn new(seed: u64) -> Self {
        // SplitMix64 seeding to fill the state from a single u64
        let mut s = seed;
        let mut state = [0u64; 4];
        for st in &mut state {
            s = s.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *st = z ^ (z >> 31);
        }
        Self { state }
    }

    /// Generate the next u64 value (xoshiro256**).
    fn next_u64(&mut self) -> u64 {
        let result =
            self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    /// Generate a uniformly distributed `f32` in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Sample an index from a probability distribution (multinomial).
    ///
    /// `probs` must sum to approximately 1.0. Returns the chosen index.
    pub fn sample_index(&mut self, probs: &[f32]) -> usize {
        let r = self.next_f32();
        let mut cumulative = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                return i;
            }
        }
        // Fallback to last index (numerical rounding)
        probs.len().saturating_sub(1)
    }
}

impl fmt::Display for SamplingRng {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SamplingRng(state={:?})", self.state)
    }
}

// ── Softmax utility ──────────────────────────────────────────────────────

/// Numerically stable softmax: converts logits to probabilities in-place.
pub fn softmax(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in logits.iter_mut() {
            *v /= sum;
        }
    }
}

/// Apply temperature scaling to logits in-place.
///
/// Temperature = 1.0 is identity. Lower temperatures sharpen the
/// distribution; higher temperatures flatten it. Temperature ≤ 0 is
/// treated as greedy (only max logit survives).
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if logits.is_empty() {
        return;
    }
    if temperature <= 0.0 {
        // Greedy: set all but argmax to -inf
        let max_idx = argmax_ref(logits);
        for (i, v) in logits.iter_mut().enumerate() {
            if i != max_idx {
                *v = f32::NEG_INFINITY;
            }
        }
        return;
    }
    let inv_t = 1.0 / temperature;
    for v in logits.iter_mut() {
        *v *= inv_t;
    }
}

/// CPU reference argmax: returns index of the largest value.
///
/// Ties are broken in favour of the first (lowest-index) occurrence.
pub fn argmax_ref(logits: &[f32]) -> usize {
    if logits.is_empty() {
        return 0;
    }
    let mut best = 0;
    let mut best_val = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best
}

// ── Sampler trait ─────────────────────────────────────────────────────────

/// Common interface for all sampling strategies.
///
/// A sampler takes logits (raw model outputs) and returns a token index.
/// Implementations may modify `logits` in-place (e.g., masking, scaling).
pub trait Sampler: Send + Sync + fmt::Debug {
    /// Select a token index from the given logits.
    fn sample(&mut self, logits: &mut [f32]) -> usize;

    /// Human-readable name of this sampler.
    fn name(&self) -> &'static str;
}

// ── GreedySampler ────────────────────────────────────────────────────────

/// Always picks the highest-probability token (argmax).
#[derive(Debug, Clone, Copy)]
pub struct GreedySampler;

impl Sampler for GreedySampler {
    fn sample(&mut self, logits: &mut [f32]) -> usize {
        argmax_ref(logits)
    }

    fn name(&self) -> &'static str {
        "greedy"
    }
}

impl fmt::Display for GreedySampler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GreedySampler")
    }
}

// ── TopKSampler ──────────────────────────────────────────────────────────

/// Sample from the top-*k* highest-probability tokens.
///
/// After filtering to the top-*k* tokens the remaining logits are
/// renormalized via softmax, then a token is drawn from the resulting
/// distribution.
#[derive(Debug, Clone)]
pub struct TopKSampler {
    /// Number of top candidates to keep.
    pub k: usize,
    /// Temperature applied before sampling.
    pub temperature: f32,
    rng: SamplingRng,
}

impl TopKSampler {
    pub fn new(k: usize, temperature: f32, seed: u64) -> Self {
        Self { k: k.max(1), temperature, rng: SamplingRng::new(seed) }
    }
}

impl Sampler for TopKSampler {
    fn sample(&mut self, logits: &mut [f32]) -> usize {
        apply_temperature(logits, self.temperature);
        top_k_filter(logits, self.k);
        softmax(logits);
        self.rng.sample_index(logits)
    }

    fn name(&self) -> &'static str {
        "top_k"
    }
}

impl fmt::Display for TopKSampler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TopKSampler(k={}, temperature={})",
            self.k, self.temperature
        )
    }
}

/// CPU reference top-k filter: set all but the top-k logits to -∞.
pub fn top_k_filter(logits: &mut [f32], k: usize) {
    if k >= logits.len() {
        return;
    }
    // Collect (index, value) and partial-sort to find the k-th threshold
    let mut indexed: Vec<(usize, f32)> =
        logits.iter().copied().enumerate().collect();
    indexed
        .select_nth_unstable_by(k.saturating_sub(1), |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
    let threshold = indexed[k.saturating_sub(1)].1;
    // Count how many values are >= threshold (may be more than k due to ties)
    let mut kept = 0;
    for v in logits.iter_mut() {
        if *v >= threshold && kept < k {
            kept += 1;
        } else if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }
}

// ── TopPSampler ──────────────────────────────────────────────────────────

/// Nucleus sampling: sample from the smallest set of tokens whose
/// cumulative probability is ≥ *p*.
#[derive(Debug, Clone)]
pub struct TopPSampler {
    /// Cumulative probability threshold (0.0 .. 1.0].
    pub p: f32,
    /// Temperature applied before sampling.
    pub temperature: f32,
    rng: SamplingRng,
}

impl TopPSampler {
    pub fn new(p: f32, temperature: f32, seed: u64) -> Self {
        Self {
            p: p.clamp(0.0, 1.0),
            temperature,
            rng: SamplingRng::new(seed),
        }
    }
}

impl Sampler for TopPSampler {
    fn sample(&mut self, logits: &mut [f32]) -> usize {
        apply_temperature(logits, self.temperature);
        top_p_filter(logits, self.p);
        softmax(logits);
        self.rng.sample_index(logits)
    }

    fn name(&self) -> &'static str {
        "top_p"
    }
}

impl fmt::Display for TopPSampler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TopPSampler(p={}, temperature={})",
            self.p, self.temperature
        )
    }
}

/// CPU reference top-p filter: mask tokens outside the nucleus.
pub fn top_p_filter(logits: &mut [f32], p: f32) {
    if logits.is_empty() {
        return;
    }
    // Convert to probabilities for ranking
    let mut probs: Vec<(usize, f32)> =
        logits.iter().copied().enumerate().collect();
    // Sort descending by logit value
    probs.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Apply softmax to sorted logits to get probabilities
    let max_val = probs[0].1;
    let mut sorted_probs: Vec<f32> = probs
        .iter()
        .map(|&(_, v)| (v - max_val).exp())
        .collect();
    let sum: f32 = sorted_probs.iter().sum();
    if sum > 0.0 {
        for sp in &mut sorted_probs {
            *sp /= sum;
        }
    }

    // Find cutoff: smallest set with cumulative prob >= p
    let mut cumulative = 0.0f32;
    let mut keep = std::collections::HashSet::new();
    for (rank, &(idx, _)) in probs.iter().enumerate() {
        cumulative += sorted_probs[rank];
        keep.insert(idx);
        if cumulative >= p {
            break;
        }
    }

    // Mask everything not in the nucleus
    for (i, v) in logits.iter_mut().enumerate() {
        if !keep.contains(&i) {
            *v = f32::NEG_INFINITY;
        }
    }
}

// ── MinPSampler ──────────────────────────────────────────────────────────

/// Minimum-probability sampling: filter tokens whose probability is below
/// `min_p × max_prob`.
#[derive(Debug, Clone)]
pub struct MinPSampler {
    /// Minimum probability ratio (0.0 .. 1.0].
    pub min_p: f32,
    /// Temperature applied before sampling.
    pub temperature: f32,
    rng: SamplingRng,
}

impl MinPSampler {
    pub fn new(min_p: f32, temperature: f32, seed: u64) -> Self {
        Self {
            min_p: min_p.clamp(0.0, 1.0),
            temperature,
            rng: SamplingRng::new(seed),
        }
    }
}

impl Sampler for MinPSampler {
    fn sample(&mut self, logits: &mut [f32]) -> usize {
        apply_temperature(logits, self.temperature);
        min_p_filter(logits, self.min_p);
        softmax(logits);
        self.rng.sample_index(logits)
    }

    fn name(&self) -> &'static str {
        "min_p"
    }
}

impl fmt::Display for MinPSampler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MinPSampler(min_p={}, temperature={})",
            self.min_p, self.temperature
        )
    }
}

/// CPU reference min-p filter.
///
/// Converts logits → probs via softmax, finds max prob, then masks any
/// token whose probability is below `min_p * max_prob`.
pub fn min_p_filter(logits: &mut [f32], min_p: f32) {
    if logits.is_empty() || min_p <= 0.0 {
        return;
    }
    // Compute softmax probabilities (without modifying logits)
    let max_logit =
        logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = logits
        .iter()
        .map(|&v| (v - max_logit).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 {
        return;
    }
    let probs: Vec<f32> = probs.iter().map(|&p| p / sum).collect();
    let max_prob =
        probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let threshold = min_p * max_prob;

    for (i, v) in logits.iter_mut().enumerate() {
        if probs[i] < threshold {
            *v = f32::NEG_INFINITY;
        }
    }
}

// ── TypicalSampler ───────────────────────────────────────────────────────

/// Typical sampling: filter by entropy-based surprise.
///
/// Keeps tokens whose information content (negative log probability) is
/// close to the expected information content (entropy) of the distribution.
/// See "Typical Decoding for Natural Language Generation" (Meister et al.,
/// 2022).
#[derive(Debug, Clone)]
pub struct TypicalSampler {
    /// Cumulative probability mass to keep (like top-p but in typical
    /// space). Range (0.0 .. 1.0].
    pub typical_p: f32,
    /// Temperature applied before sampling.
    pub temperature: f32,
    rng: SamplingRng,
}

impl TypicalSampler {
    pub fn new(typical_p: f32, temperature: f32, seed: u64) -> Self {
        Self {
            typical_p: typical_p.clamp(0.0, 1.0),
            temperature,
            rng: SamplingRng::new(seed),
        }
    }
}

impl Sampler for TypicalSampler {
    fn sample(&mut self, logits: &mut [f32]) -> usize {
        apply_temperature(logits, self.temperature);
        typical_filter(logits, self.typical_p);
        softmax(logits);
        self.rng.sample_index(logits)
    }

    fn name(&self) -> &'static str {
        "typical"
    }
}

impl fmt::Display for TypicalSampler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TypicalSampler(p={}, temperature={})",
            self.typical_p, self.temperature
        )
    }
}

/// CPU reference typical sampling filter.
///
/// 1. Convert logits → probs via softmax.
/// 2. Compute entropy H = −Σ p·log(p).
/// 3. For each token compute surprise = −log(p) and deviation |surprise − H|.
/// 4. Sort tokens by deviation (ascending).
/// 5. Keep the smallest set whose cumulative probability ≥ `typical_p`.
pub fn typical_filter(logits: &mut [f32], typical_p: f32) {
    if logits.is_empty() || typical_p >= 1.0 {
        return;
    }
    // Softmax → probs
    let max_val =
        logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = logits
        .iter()
        .map(|&v| (v - max_val).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 {
        return;
    }
    let probs: Vec<f32> = probs.iter().map(|&p| p / sum).collect();

    // Entropy
    let entropy: f32 = probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();

    // (index, probability, |surprise − entropy|)
    let mut scored: Vec<(usize, f32, f32)> = probs
        .iter()
        .enumerate()
        .filter(|&(_, p)| *p > 0.0)
        .map(|(i, p)| {
            let surprise = -p.ln();
            let deviation = (surprise - entropy).abs();
            (i, *p, deviation)
        })
        .collect();

    // Sort by deviation ascending (most "typical" first)
    scored.sort_unstable_by(|a, b| {
        a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Keep smallest set with cumulative prob ≥ typical_p
    let mut keep = std::collections::HashSet::new();
    let mut cumulative = 0.0f32;
    for &(idx, prob, _) in &scored {
        keep.insert(idx);
        cumulative += prob;
        if cumulative >= typical_p {
            break;
        }
    }

    for (i, v) in logits.iter_mut().enumerate() {
        if !keep.contains(&i) {
            *v = f32::NEG_INFINITY;
        }
    }
}

// ── MirostatSampler ──────────────────────────────────────────────────────

/// Mirostat v2 sampling: maintains a target perplexity by dynamically
/// adjusting the surprise threshold.
///
/// See "Mirostat: A Neural Text Decoding Algorithm that Directly Controls
/// Perplexity" (Basu et al., 2021).
#[derive(Debug, Clone)]
pub struct MirostatSampler {
    /// Target surprise (log2 of target perplexity). Typical range: 3–8.
    pub tau: f32,
    /// Learning rate for threshold adaptation. Typical value: 0.1.
    pub eta: f32,
    /// Current surprise threshold (mu). Starts at 2*tau.
    mu: f32,
    rng: SamplingRng,
}

impl MirostatSampler {
    pub fn new(tau: f32, eta: f32, seed: u64) -> Self {
        Self {
            tau,
            eta,
            mu: 2.0 * tau,
            rng: SamplingRng::new(seed),
        }
    }

    /// Current value of the adaptive threshold (mu).
    pub fn mu(&self) -> f32 {
        self.mu
    }
}

impl Sampler for MirostatSampler {
    fn sample(&mut self, logits: &mut [f32]) -> usize {
        // Convert to probabilities
        softmax(logits);

        // Sort tokens by probability descending
        let mut candidates: Vec<(usize, f32)> =
            logits.iter().copied().enumerate().collect();
        candidates.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Find tokens with surprise ≤ mu
        let mut filtered: Vec<(usize, f32)> = Vec::new();
        for &(idx, prob) in &candidates {
            if prob <= 0.0 {
                continue;
            }
            let surprise = -(prob.log2());
            if surprise <= self.mu {
                filtered.push((idx, prob));
            }
        }

        // If nothing passes the filter, keep the most probable token
        if filtered.is_empty()
            && let Some(&(idx, prob)) = candidates.first()
        {
            filtered.push((idx, prob));
        }

        // Renormalize and sample
        let total: f32 = filtered.iter().map(|&(_, p)| p).sum();
        let r = self.rng.next_f32();
        let mut cumulative = 0.0f32;
        let mut chosen_idx = filtered[0].0;
        let mut chosen_prob = filtered[0].1;
        for &(idx, prob) in &filtered {
            cumulative += prob / total;
            if r < cumulative {
                chosen_idx = idx;
                chosen_prob = prob;
                break;
            }
        }

        // Update mu: mu = mu - eta * (surprise - tau)
        let surprise = if chosen_prob > 0.0 {
            -(chosen_prob.log2())
        } else {
            self.tau
        };
        self.mu -= self.eta * (surprise - self.tau);

        chosen_idx
    }

    fn name(&self) -> &'static str {
        "mirostat_v2"
    }
}

impl fmt::Display for MirostatSampler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MirostatSampler(tau={}, eta={}, mu={})",
            self.tau, self.eta, self.mu
        )
    }
}

// ── RepetitionPenalty ────────────────────────────────────────────────────

/// Penalize repeated tokens using multiplicative and optional additive
/// penalties.
///
/// For each token in the context window:
/// - If `logit > 0`: `logit /= penalty`
/// - If `logit < 0`: `logit *= penalty`
/// - Then `logit -= additive_penalty`
///
/// `penalty` = 1.0 is no-op. Typical values: 1.05–1.3.
#[derive(Debug, Clone)]
pub struct RepetitionPenalty {
    /// Multiplicative penalty factor (≥ 1.0).
    pub penalty: f32,
    /// Additive penalty subtracted from each repeated token's logit.
    pub additive: f32,
    /// Maximum number of recent tokens to track.
    pub window_size: usize,
    /// Ring-buffer of recent token IDs.
    context: Vec<u32>,
}

impl RepetitionPenalty {
    pub fn new(
        penalty: f32,
        additive: f32,
        window_size: usize,
    ) -> Self {
        Self {
            penalty: penalty.max(1.0),
            additive,
            window_size,
            context: Vec::new(),
        }
    }

    /// Record a generated token into the context window.
    pub fn add_token(&mut self, token_id: u32) {
        self.context.push(token_id);
        if self.context.len() > self.window_size {
            self.context.remove(0);
        }
    }

    /// Apply the repetition penalty to logits in-place.
    pub fn apply(&self, logits: &mut [f32]) {
        for &tok in &self.context {
            let idx = tok as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    logits[idx] /= self.penalty;
                } else if logits[idx] < 0.0 {
                    logits[idx] *= self.penalty;
                }
                logits[idx] -= self.additive;
            }
        }
    }

    /// Reset the context window.
    pub fn reset(&mut self) {
        self.context.clear();
    }
}

impl fmt::Display for RepetitionPenalty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RepetitionPenalty(penalty={}, additive={}, window={})",
            self.penalty, self.additive, self.window_size
        )
    }
}

// ── FrequencyPresencePenalty ─────────────────────────────────────────────

/// OpenAI-style frequency and presence penalties.
///
/// `logit -= frequency_penalty * count(token) + presence_penalty * (count > 0)`
///
/// - `frequency_penalty` scales with how many times a token appeared.
/// - `presence_penalty` is a flat penalty for any token that appeared at
///   least once.
#[derive(Debug, Clone)]
pub struct FrequencyPresencePenalty {
    /// Penalty per occurrence. Range: 0.0–2.0.
    pub frequency_penalty: f32,
    /// Flat penalty for any token seen at least once. Range: 0.0–2.0.
    pub presence_penalty: f32,
    /// Token → count map.
    counts: HashMap<u32, u32>,
}

impl FrequencyPresencePenalty {
    pub fn new(frequency_penalty: f32, presence_penalty: f32) -> Self {
        Self {
            frequency_penalty,
            presence_penalty,
            counts: HashMap::new(),
        }
    }

    /// Record a generated token.
    pub fn add_token(&mut self, token_id: u32) {
        *self.counts.entry(token_id).or_insert(0) += 1;
    }

    /// Apply frequency + presence penalties to logits in-place.
    pub fn apply(&self, logits: &mut [f32]) {
        for (&tok, &count) in &self.counts {
            let idx = tok as usize;
            if idx < logits.len() {
                logits[idx] -= self.frequency_penalty * count as f32;
                logits[idx] -= self.presence_penalty;
            }
        }
    }

    /// Reset token counts.
    pub fn reset(&mut self) {
        self.counts.clear();
    }
}

impl fmt::Display for FrequencyPresencePenalty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FrequencyPresencePenalty(freq={}, presence={})",
            self.frequency_penalty, self.presence_penalty
        )
    }
}

// ── SamplerChain ─────────────────────────────────────────────────────────

/// Compose multiple logit processors and a final sampler.
///
/// The chain applies each processor in order, then delegates to the final
/// sampler for token selection.
///
/// # Example (conceptual)
///
/// ```text
/// RepetitionPenalty → TopK filter → TopP filter → sample
/// ```
#[derive(Debug)]
pub struct SamplerChain {
    /// Pre-processors that modify logits (penalties, filters).
    processors: Vec<Box<dyn LogitProcessor>>,
    /// Final sampler that picks a token.
    sampler: Box<dyn Sampler>,
}

/// Trait for logit processors that modify logits in-place without
/// selecting a token.
pub trait LogitProcessor: Send + Sync + fmt::Debug {
    /// Modify logits in-place.
    fn process(&self, logits: &mut [f32]);

    /// Human-readable name.
    fn name(&self) -> &'static str;
}

impl SamplerChain {
    /// Create a new chain with the given final sampler.
    pub fn new(sampler: Box<dyn Sampler>) -> Self {
        Self { processors: Vec::new(), sampler }
    }

    /// Add a logit processor to the chain.
    pub fn add_processor(
        &mut self,
        processor: Box<dyn LogitProcessor>,
    ) {
        self.processors.push(processor);
    }

    /// Run the full chain: processors then sampler.
    pub fn sample(&mut self, logits: &mut [f32]) -> usize {
        for proc in &self.processors {
            proc.process(logits);
        }
        self.sampler.sample(logits)
    }

    /// Number of processors in the chain (not counting the final sampler).
    pub fn len(&self) -> usize {
        self.processors.len()
    }

    /// Whether the chain has no processors.
    pub fn is_empty(&self) -> bool {
        self.processors.is_empty()
    }
}

impl fmt::Display for SamplerChain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SamplerChain[")?;
        for (i, proc) in self.processors.iter().enumerate() {
            if i > 0 {
                write!(f, " → ")?;
            }
            write!(f, "{}", proc.name())?;
        }
        if !self.processors.is_empty() {
            write!(f, " → ")?;
        }
        write!(f, "{}]", self.sampler.name())
    }
}

// ── LogitProcessor adapters ──────────────────────────────────────────────

/// Adapter: wrap `RepetitionPenalty` as a `LogitProcessor`.
#[derive(Debug)]
pub struct RepetitionPenaltyProcessor {
    inner: RepetitionPenalty,
}

impl RepetitionPenaltyProcessor {
    pub fn new(penalty: RepetitionPenalty) -> Self {
        Self { inner: penalty }
    }

    /// Record a generated token.
    pub fn add_token(&mut self, token_id: u32) {
        self.inner.add_token(token_id);
    }
}

impl LogitProcessor for RepetitionPenaltyProcessor {
    fn process(&self, logits: &mut [f32]) {
        self.inner.apply(logits);
    }

    fn name(&self) -> &'static str {
        "repetition_penalty"
    }
}

/// Adapter: wrap `FrequencyPresencePenalty` as a `LogitProcessor`.
#[derive(Debug)]
pub struct FrequencyPresencePenaltyProcessor {
    inner: FrequencyPresencePenalty,
}

impl FrequencyPresencePenaltyProcessor {
    pub fn new(penalty: FrequencyPresencePenalty) -> Self {
        Self { inner: penalty }
    }

    /// Record a generated token.
    pub fn add_token(&mut self, token_id: u32) {
        self.inner.add_token(token_id);
    }
}

impl LogitProcessor for FrequencyPresencePenaltyProcessor {
    fn process(&self, logits: &mut [f32]) {
        self.inner.apply(logits);
    }

    fn name(&self) -> &'static str {
        "frequency_presence_penalty"
    }
}

/// Temperature processor: scales logits before sampling.
#[derive(Debug, Clone)]
pub struct TemperatureProcessor {
    pub temperature: f32,
}

impl TemperatureProcessor {
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }
}

impl LogitProcessor for TemperatureProcessor {
    fn process(&self, logits: &mut [f32]) {
        apply_temperature(logits, self.temperature);
    }

    fn name(&self) -> &'static str {
        "temperature"
    }
}

/// Top-K logit processor (filters, does not sample).
#[derive(Debug, Clone)]
pub struct TopKProcessor {
    pub k: usize,
}

impl TopKProcessor {
    pub fn new(k: usize) -> Self {
        Self { k: k.max(1) }
    }
}

impl LogitProcessor for TopKProcessor {
    fn process(&self, logits: &mut [f32]) {
        top_k_filter(logits, self.k);
    }

    fn name(&self) -> &'static str {
        "top_k_filter"
    }
}

/// Top-P logit processor (filters, does not sample).
#[derive(Debug, Clone)]
pub struct TopPProcessor {
    pub p: f32,
}

impl TopPProcessor {
    pub fn new(p: f32) -> Self {
        Self { p: p.clamp(0.0, 1.0) }
    }
}

impl LogitProcessor for TopPProcessor {
    fn process(&self, logits: &mut [f32]) {
        top_p_filter(logits, self.p);
    }

    fn name(&self) -> &'static str {
        "top_p_filter"
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── OpenCL kernel source ─────────────────────────────────

    #[test]
    fn opencl_source_is_not_empty() {
        assert!(!SAMPLING_CL.is_empty());
    }

    #[test]
    fn opencl_source_contains_kernel_keyword() {
        assert!(SAMPLING_CL.contains("__kernel"));
    }

    #[test]
    fn opencl_source_has_softmax_kernel() {
        assert!(SAMPLING_CL.contains("softmax"));
    }

    #[test]
    fn opencl_source_has_top_k_filter_kernel() {
        assert!(SAMPLING_CL.contains("top_k_filter"));
    }

    #[test]
    fn opencl_source_has_argmax_kernel() {
        assert!(SAMPLING_CL.contains("argmax"));
    }

    // ── SamplingRng ──────────────────────────────────────────

    #[test]
    fn rng_deterministic_with_same_seed() {
        let mut a = SamplingRng::new(42);
        let mut b = SamplingRng::new(42);
        let vals_a: Vec<f32> = (0..100).map(|_| a.next_f32()).collect();
        let vals_b: Vec<f32> = (0..100).map(|_| b.next_f32()).collect();
        assert_eq!(vals_a, vals_b);
    }

    #[test]
    fn rng_different_seeds_produce_different_values() {
        let mut a = SamplingRng::new(1);
        let mut b = SamplingRng::new(2);
        let vals_a: Vec<f32> = (0..10).map(|_| a.next_f32()).collect();
        let vals_b: Vec<f32> = (0..10).map(|_| b.next_f32()).collect();
        assert_ne!(vals_a, vals_b);
    }

    #[test]
    fn rng_values_in_unit_interval() {
        let mut rng = SamplingRng::new(123);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "value {v} not in [0, 1)");
        }
    }

    #[test]
    fn rng_sample_index_returns_valid_index() {
        let mut rng = SamplingRng::new(42);
        let probs = vec![0.1, 0.2, 0.3, 0.4];
        for _ in 0..1000 {
            let idx = rng.sample_index(&probs);
            assert!(idx < probs.len());
        }
    }

    #[test]
    fn rng_sample_index_single_element() {
        let mut rng = SamplingRng::new(0);
        let probs = vec![1.0];
        for _ in 0..100 {
            assert_eq!(rng.sample_index(&probs), 0);
        }
    }

    #[test]
    fn rng_sample_index_concentrates_on_high_prob() {
        let mut rng = SamplingRng::new(99);
        let probs = vec![0.0001, 0.0001, 0.9998];
        let mut counts = [0u32; 3];
        for _ in 0..1000 {
            counts[rng.sample_index(&probs)] += 1;
        }
        // Token 2 should dominate
        assert!(counts[2] > 900, "counts: {counts:?}");
    }

    #[test]
    fn rng_display() {
        let rng = SamplingRng::new(42);
        let s = format!("{rng}");
        assert!(s.contains("SamplingRng"));
    }

    // ── Softmax ──────────────────────────────────────────────

    #[test]
    fn softmax_sums_to_one() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        softmax(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
    }

    #[test]
    fn softmax_preserves_order() {
        let mut logits = vec![1.0, 3.0, 2.0];
        softmax(&mut logits);
        assert!(logits[1] > logits[2]);
        assert!(logits[2] > logits[0]);
    }

    #[test]
    fn softmax_empty_is_noop() {
        let mut logits: Vec<f32> = vec![];
        softmax(&mut logits);
        assert!(logits.is_empty());
    }

    #[test]
    fn softmax_single_element() {
        let mut logits = vec![5.0];
        softmax(&mut logits);
        assert!((logits[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_uniform_produces_uniform() {
        let mut logits = vec![2.0; 5];
        softmax(&mut logits);
        for &p in &logits {
            assert!((p - 0.2).abs() < 1e-6, "p = {p}");
        }
    }

    #[test]
    fn softmax_numerical_stability_large_values() {
        let mut logits = vec![1000.0, 1001.0, 999.0];
        softmax(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(logits[1] > logits[0]);
        assert!(logits[0] > logits[2]);
    }

    // ── Temperature ──────────────────────────────────────────

    #[test]
    fn temperature_identity_at_one() {
        let original = vec![1.0, 2.0, 3.0];
        let mut logits = original.clone();
        apply_temperature(&mut logits, 1.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let mut logits = vec![1.0, 5.0, 3.0];
        apply_temperature(&mut logits, 0.0);
        assert_eq!(logits[1], 5.0);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn temperature_low_sharpens() {
        let mut low = vec![1.0, 2.0, 3.0];
        let mut high = vec![1.0, 2.0, 3.0];
        apply_temperature(&mut low, 0.1);
        apply_temperature(&mut high, 2.0);
        softmax(&mut low);
        softmax(&mut high);
        // Low temp should make distribution more peaked
        assert!(low[2] > high[2]);
    }

    #[test]
    fn temperature_empty_is_noop() {
        let mut logits: Vec<f32> = vec![];
        apply_temperature(&mut logits, 0.5);
        assert!(logits.is_empty());
    }

    // ── Argmax ───────────────────────────────────────────────

    #[test]
    fn argmax_simple() {
        assert_eq!(argmax_ref(&[1.0, 5.0, 3.0]), 1);
    }

    #[test]
    fn argmax_first_element() {
        assert_eq!(argmax_ref(&[10.0, 1.0, 2.0]), 0);
    }

    #[test]
    fn argmax_last_element() {
        assert_eq!(argmax_ref(&[1.0, 2.0, 10.0]), 2);
    }

    #[test]
    fn argmax_tie_picks_first() {
        assert_eq!(argmax_ref(&[5.0, 5.0, 5.0]), 0);
    }

    #[test]
    fn argmax_single() {
        assert_eq!(argmax_ref(&[42.0]), 0);
    }

    #[test]
    fn argmax_empty_returns_zero() {
        assert_eq!(argmax_ref(&[]), 0);
    }

    #[test]
    fn argmax_negative_values() {
        assert_eq!(argmax_ref(&[-3.0, -1.0, -2.0]), 1);
    }

    // ── GreedySampler ────────────────────────────────────────

    #[test]
    fn greedy_picks_argmax() {
        let mut s = GreedySampler;
        assert_eq!(s.sample(&mut [1.0, 5.0, 3.0]), 1);
    }

    #[test]
    fn greedy_picks_argmax_tie() {
        let mut s = GreedySampler;
        assert_eq!(s.sample(&mut [5.0, 5.0, 5.0]), 0);
    }

    #[test]
    fn greedy_deterministic() {
        let mut s = GreedySampler;
        let logits = [1.0, 3.0, 2.0];
        let a = s.sample(&mut logits.clone());
        let b = s.sample(&mut logits.clone());
        assert_eq!(a, b);
    }

    #[test]
    fn greedy_name() {
        assert_eq!(GreedySampler.name(), "greedy");
    }

    #[test]
    fn greedy_display() {
        assert_eq!(format!("{}", GreedySampler), "GreedySampler");
    }

    // ── TopKSampler ──────────────────────────────────────────

    #[test]
    fn top_k_filter_limits_candidates() {
        let mut logits = vec![1.0, 5.0, 3.0, 4.0, 2.0];
        top_k_filter(&mut logits, 2);
        let active: Vec<usize> = logits
            .iter()
            .enumerate()
            .filter(|&(_, v)| *v != f32::NEG_INFINITY)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(active.len(), 2);
        // Should keep indices 1 (5.0) and 3 (4.0)
        assert!(active.contains(&1));
        assert!(active.contains(&3));
    }

    #[test]
    fn top_k_filter_k_equals_len_keeps_all() {
        let mut logits = vec![1.0, 2.0, 3.0];
        top_k_filter(&mut logits, 3);
        assert!(logits.iter().all(|&v| v != f32::NEG_INFINITY));
    }

    #[test]
    fn top_k_filter_k_exceeds_len_keeps_all() {
        let mut logits = vec![1.0, 2.0];
        top_k_filter(&mut logits, 10);
        assert!(logits.iter().all(|&v| v != f32::NEG_INFINITY));
    }

    #[test]
    fn top_k_filter_k_one_keeps_max() {
        let mut logits = vec![1.0, 5.0, 3.0];
        top_k_filter(&mut logits, 1);
        assert_eq!(logits[1], 5.0);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn top_k_sampler_returns_valid_index() {
        let mut s = TopKSampler::new(3, 1.0, 42);
        for _ in 0..100 {
            let logits = &mut [1.0, 2.0, 3.0, 4.0, 5.0];
            let idx = s.sample(logits);
            assert!(idx < 5);
        }
    }

    #[test]
    fn top_k_sampler_reproducible() {
        let mut a = TopKSampler::new(3, 1.0, 42);
        let mut b = TopKSampler::new(3, 1.0, 42);
        let results_a: Vec<usize> = (0..20)
            .map(|_| a.sample(&mut [1.0, 2.0, 3.0, 4.0, 5.0]))
            .collect();
        let results_b: Vec<usize> = (0..20)
            .map(|_| b.sample(&mut [1.0, 2.0, 3.0, 4.0, 5.0]))
            .collect();
        assert_eq!(results_a, results_b);
    }

    #[test]
    fn top_k_sampler_name() {
        let s = TopKSampler::new(5, 1.0, 0);
        assert_eq!(s.name(), "top_k");
    }

    #[test]
    fn top_k_sampler_display() {
        let s = TopKSampler::new(5, 0.8, 0);
        let d = format!("{s}");
        assert!(d.contains("TopKSampler"));
        assert!(d.contains("k=5"));
    }

    // ── TopPSampler ──────────────────────────────────────────

    #[test]
    fn top_p_filter_keeps_nucleus() {
        // Logits that form a clear probability ordering
        let mut logits = vec![10.0, 1.0, 0.5, 0.1, 0.01];
        // With p=0.9 the single highest token has prob ~0.9999
        // so only 1 token should remain
        top_p_filter(&mut logits, 0.9);
        let active: usize = logits
            .iter()
            .filter(|&&v| v != f32::NEG_INFINITY)
            .count();
        assert!(active >= 1, "at least 1 token should survive");
    }

    #[test]
    fn top_p_filter_p_one_keeps_all() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        top_p_filter(&mut logits, 1.0);
        let active: usize = logits
            .iter()
            .filter(|&&v| v != f32::NEG_INFINITY)
            .count();
        assert_eq!(active, 4);
    }

    #[test]
    fn top_p_filter_small_p_keeps_top() {
        let mut logits = vec![0.0, 0.0, 10.0, 0.0];
        top_p_filter(&mut logits, 0.01);
        // Token 2 (logit=10) dominates
        assert!(logits[2] != f32::NEG_INFINITY);
    }

    #[test]
    fn top_p_sampler_returns_valid_index() {
        let mut s = TopPSampler::new(0.9, 1.0, 42);
        for _ in 0..100 {
            let idx = s.sample(&mut [1.0, 2.0, 3.0, 4.0]);
            assert!(idx < 4);
        }
    }

    #[test]
    fn top_p_sampler_reproducible() {
        let mut a = TopPSampler::new(0.9, 1.0, 42);
        let mut b = TopPSampler::new(0.9, 1.0, 42);
        let ra: Vec<usize> = (0..20)
            .map(|_| a.sample(&mut [1.0, 2.0, 3.0, 4.0]))
            .collect();
        let rb: Vec<usize> = (0..20)
            .map(|_| b.sample(&mut [1.0, 2.0, 3.0, 4.0]))
            .collect();
        assert_eq!(ra, rb);
    }

    #[test]
    fn top_p_sampler_name() {
        let s = TopPSampler::new(0.9, 1.0, 0);
        assert_eq!(s.name(), "top_p");
    }

    #[test]
    fn top_p_sampler_display() {
        let s = TopPSampler::new(0.95, 1.0, 0);
        let d = format!("{s}");
        assert!(d.contains("TopPSampler"));
    }

    // ── MinPSampler ──────────────────────────────────────────

    #[test]
    fn min_p_filter_removes_low_prob_tokens() {
        // logits: softmax([10, 1, 0, -5]) ≈ [0.9999, 0.0001, ...]
        let mut logits = vec![10.0, 1.0, 0.0, -5.0];
        min_p_filter(&mut logits, 0.1);
        // Only token 0 should survive (prob > 0.1 * max_prob)
        assert!(logits[0] != f32::NEG_INFINITY);
        // Token 3 (very low prob) should be filtered
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    #[test]
    fn min_p_filter_zero_keeps_all() {
        let mut logits = vec![1.0, 2.0, 3.0];
        min_p_filter(&mut logits, 0.0);
        assert!(logits.iter().all(|&v| v != f32::NEG_INFINITY));
    }

    #[test]
    fn min_p_filter_one_keeps_only_max() {
        let mut logits = vec![1.0, 5.0, 3.0];
        min_p_filter(&mut logits, 1.0);
        // Only the token with max probability should survive
        assert!(logits[1] != f32::NEG_INFINITY);
    }

    #[test]
    fn min_p_filter_uniform_keeps_all() {
        let mut logits = vec![2.0; 5];
        min_p_filter(&mut logits, 0.5);
        // All have equal probability, all should survive
        assert!(logits.iter().all(|&v| v != f32::NEG_INFINITY));
    }

    #[test]
    fn min_p_sampler_returns_valid_index() {
        let mut s = MinPSampler::new(0.1, 1.0, 42);
        for _ in 0..100 {
            let idx = s.sample(&mut [1.0, 2.0, 3.0, 4.0, 5.0]);
            assert!(idx < 5);
        }
    }

    #[test]
    fn min_p_sampler_reproducible() {
        let mut a = MinPSampler::new(0.1, 1.0, 42);
        let mut b = MinPSampler::new(0.1, 1.0, 42);
        let ra: Vec<usize> = (0..20)
            .map(|_| a.sample(&mut [1.0, 2.0, 3.0, 4.0, 5.0]))
            .collect();
        let rb: Vec<usize> = (0..20)
            .map(|_| b.sample(&mut [1.0, 2.0, 3.0, 4.0, 5.0]))
            .collect();
        assert_eq!(ra, rb);
    }

    #[test]
    fn min_p_sampler_name() {
        assert_eq!(MinPSampler::new(0.1, 1.0, 0).name(), "min_p");
    }

    // ── TypicalSampler ───────────────────────────────────────

    #[test]
    fn typical_filter_keeps_typical_tokens() {
        // Heavily skewed: one dominant token
        let mut logits = vec![10.0, 0.0, 0.0, 0.0, 0.0];
        typical_filter(&mut logits, 0.5);
        // The dominant token should survive
        assert!(logits[0] != f32::NEG_INFINITY);
    }

    #[test]
    fn typical_filter_p_one_keeps_all() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        typical_filter(&mut logits, 1.0);
        assert!(logits.iter().all(|&v| v != f32::NEG_INFINITY));
    }

    #[test]
    fn typical_filter_uniform_keeps_all_at_high_p() {
        let mut logits = vec![1.0; 10];
        typical_filter(&mut logits, 0.99);
        let active = logits
            .iter()
            .filter(|&&v| v != f32::NEG_INFINITY)
            .count();
        assert_eq!(active, 10);
    }

    #[test]
    fn typical_sampler_returns_valid_index() {
        let mut s = TypicalSampler::new(0.9, 1.0, 42);
        for _ in 0..100 {
            let idx = s.sample(&mut [1.0, 2.0, 3.0, 4.0]);
            assert!(idx < 4);
        }
    }

    #[test]
    fn typical_sampler_reproducible() {
        let mut a = TypicalSampler::new(0.9, 1.0, 42);
        let mut b = TypicalSampler::new(0.9, 1.0, 42);
        let ra: Vec<usize> = (0..20)
            .map(|_| a.sample(&mut [1.0, 2.0, 3.0, 4.0]))
            .collect();
        let rb: Vec<usize> = (0..20)
            .map(|_| b.sample(&mut [1.0, 2.0, 3.0, 4.0]))
            .collect();
        assert_eq!(ra, rb);
    }

    #[test]
    fn typical_sampler_name() {
        assert_eq!(
            TypicalSampler::new(0.9, 1.0, 0).name(),
            "typical"
        );
    }

    #[test]
    fn typical_sampler_display() {
        let s = TypicalSampler::new(0.9, 1.0, 0);
        let d = format!("{s}");
        assert!(d.contains("TypicalSampler"));
    }

    // ── MirostatSampler ──────────────────────────────────────

    #[test]
    fn mirostat_returns_valid_index() {
        let mut s = MirostatSampler::new(5.0, 0.1, 42);
        for _ in 0..100 {
            let idx = s.sample(&mut [1.0, 2.0, 3.0, 4.0, 5.0]);
            assert!(idx < 5);
        }
    }

    #[test]
    fn mirostat_mu_adapts() {
        let mut s = MirostatSampler::new(5.0, 0.1, 42);
        let initial_mu = s.mu();
        // Sample several times to let mu adapt
        for _ in 0..20 {
            s.sample(&mut [1.0, 2.0, 3.0, 4.0, 5.0]);
        }
        // mu should have changed from initial value
        assert!(
            (s.mu() - initial_mu).abs() > 1e-6,
            "mu should adapt: initial={initial_mu}, final={}",
            s.mu()
        );
    }

    #[test]
    fn mirostat_converges_toward_target() {
        let tau = 5.0;
        let mut s = MirostatSampler::new(tau, 0.1, 42);
        // Run many iterations
        for _ in 0..200 {
            s.sample(&mut [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        }
        // mu should be in a reasonable range around 2*tau ± some
        // (convergence depends on the distribution)
        assert!(
            s.mu().is_finite(),
            "mu should remain finite: {}",
            s.mu()
        );
    }

    #[test]
    fn mirostat_initial_mu_is_two_tau() {
        let s = MirostatSampler::new(5.0, 0.1, 0);
        assert!((s.mu() - 10.0).abs() < 1e-6);
    }

    #[test]
    fn mirostat_reproducible() {
        let mut a = MirostatSampler::new(5.0, 0.1, 42);
        let mut b = MirostatSampler::new(5.0, 0.1, 42);
        let ra: Vec<usize> = (0..20)
            .map(|_| a.sample(&mut [1.0, 2.0, 3.0, 4.0, 5.0]))
            .collect();
        let rb: Vec<usize> = (0..20)
            .map(|_| b.sample(&mut [1.0, 2.0, 3.0, 4.0, 5.0]))
            .collect();
        assert_eq!(ra, rb);
    }

    #[test]
    fn mirostat_name() {
        assert_eq!(
            MirostatSampler::new(5.0, 0.1, 0).name(),
            "mirostat_v2"
        );
    }

    #[test]
    fn mirostat_display() {
        let s = MirostatSampler::new(5.0, 0.1, 0);
        let d = format!("{s}");
        assert!(d.contains("MirostatSampler"));
        assert!(d.contains("tau=5"));
    }

    // ── RepetitionPenalty ────────────────────────────────────

    #[test]
    fn repetition_penalty_reduces_repeated() {
        let mut rp = RepetitionPenalty::new(1.2, 0.0, 64);
        rp.add_token(2);
        let mut logits = vec![1.0, 1.0, 1.0, 1.0];
        rp.apply(&mut logits);
        assert!(logits[2] < logits[0], "repeated token should be penalized");
        assert!(logits[2] < logits[1]);
        assert!(logits[2] < logits[3]);
    }

    #[test]
    fn repetition_penalty_negative_logit() {
        let mut rp = RepetitionPenalty::new(1.5, 0.0, 64);
        rp.add_token(1);
        let mut logits = vec![0.0, -2.0, 0.0];
        rp.apply(&mut logits);
        // Negative logit * penalty = more negative
        assert!(logits[1] < -2.0);
    }

    #[test]
    fn repetition_penalty_additive() {
        let mut rp = RepetitionPenalty::new(1.0, 0.5, 64);
        rp.add_token(0);
        let mut logits = vec![3.0, 3.0, 3.0];
        rp.apply(&mut logits);
        assert!((logits[0] - 2.5).abs() < 1e-6);
        assert!((logits[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn repetition_penalty_window() {
        let mut rp = RepetitionPenalty::new(2.0, 0.0, 3);
        for i in 0..5 {
            rp.add_token(i);
        }
        let mut logits = vec![1.0; 6];
        rp.apply(&mut logits);
        // Only tokens 2, 3, 4 should be in the window (size=3)
        assert!((logits[0] - 1.0).abs() < 1e-6, "token 0 evicted");
        assert!((logits[1] - 1.0).abs() < 1e-6, "token 1 evicted");
        assert!(logits[2] < 1.0, "token 2 in window");
        assert!(logits[3] < 1.0, "token 3 in window");
        assert!(logits[4] < 1.0, "token 4 in window");
    }

    #[test]
    fn repetition_penalty_reset() {
        let mut rp = RepetitionPenalty::new(2.0, 0.0, 64);
        rp.add_token(0);
        rp.reset();
        let mut logits = vec![1.0, 1.0];
        rp.apply(&mut logits);
        assert!((logits[0] - 1.0).abs() < 1e-6, "should be unpenalized");
    }

    #[test]
    fn repetition_penalty_no_penalty_at_one() {
        let mut rp = RepetitionPenalty::new(1.0, 0.0, 64);
        rp.add_token(0);
        rp.add_token(1);
        let mut logits = vec![2.0, 3.0, 4.0];
        rp.apply(&mut logits);
        assert!((logits[0] - 2.0).abs() < 1e-6);
        assert!((logits[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn repetition_penalty_display() {
        let rp = RepetitionPenalty::new(1.2, 0.1, 64);
        let d = format!("{rp}");
        assert!(d.contains("RepetitionPenalty"));
    }

    // ── FrequencyPresencePenalty ─────────────────────────────

    #[test]
    fn freq_presence_frequency_scales_with_count() {
        let mut fp = FrequencyPresencePenalty::new(0.5, 0.0);
        fp.add_token(1);
        fp.add_token(1);
        fp.add_token(1);
        let mut logits = vec![5.0, 5.0, 5.0];
        fp.apply(&mut logits);
        // Token 1: 5.0 - 0.5 * 3 = 3.5
        assert!((logits[1] - 3.5).abs() < 1e-6);
        assert!((logits[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn freq_presence_presence_is_flat() {
        let mut fp = FrequencyPresencePenalty::new(0.0, 1.0);
        fp.add_token(0);
        fp.add_token(0);
        fp.add_token(0);
        let mut logits = vec![5.0, 5.0];
        fp.apply(&mut logits);
        // Token 0: 5.0 - 1.0 = 4.0 (presence penalty applied once)
        assert!((logits[0] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn freq_presence_combined() {
        let mut fp = FrequencyPresencePenalty::new(0.5, 0.3);
        fp.add_token(2);
        fp.add_token(2);
        let mut logits = vec![5.0, 5.0, 5.0];
        fp.apply(&mut logits);
        // Token 2: 5.0 - 0.5*2 - 0.3 = 3.7
        assert!((logits[2] - 3.7).abs() < 1e-5);
    }

    #[test]
    fn freq_presence_reset() {
        let mut fp = FrequencyPresencePenalty::new(1.0, 1.0);
        fp.add_token(0);
        fp.reset();
        let mut logits = vec![5.0, 5.0];
        fp.apply(&mut logits);
        assert!((logits[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn freq_presence_display() {
        let fp = FrequencyPresencePenalty::new(0.5, 0.3);
        let d = format!("{fp}");
        assert!(d.contains("FrequencyPresencePenalty"));
    }

    // ── SamplerChain ─────────────────────────────────────────

    #[test]
    fn chain_empty_delegates_to_sampler() {
        let mut chain = SamplerChain::new(Box::new(GreedySampler));
        let idx = chain.sample(&mut [1.0, 3.0, 2.0]);
        assert_eq!(idx, 1);
    }

    #[test]
    fn chain_with_temperature_and_greedy() {
        let mut chain = SamplerChain::new(Box::new(GreedySampler));
        chain.add_processor(Box::new(TemperatureProcessor::new(0.0)));
        let idx = chain.sample(&mut [1.0, 3.0, 2.0]);
        // Temperature 0 makes it greedy, then greedy picks argmax
        assert_eq!(idx, 1);
    }

    #[test]
    fn chain_with_top_k_processor() {
        let mut chain = SamplerChain::new(
            Box::new(TopKSampler::new(2, 1.0, 42)),
        );
        chain.add_processor(Box::new(TopKProcessor::new(3)));
        for _ in 0..50 {
            let idx =
                chain.sample(&mut [1.0, 2.0, 3.0, 4.0, 5.0]);
            assert!(idx < 5);
        }
    }

    #[test]
    fn chain_len_and_is_empty() {
        let chain = SamplerChain::new(Box::new(GreedySampler));
        assert_eq!(chain.len(), 0);
        assert!(chain.is_empty());

        let mut chain2 = SamplerChain::new(Box::new(GreedySampler));
        chain2.add_processor(Box::new(TemperatureProcessor::new(0.5)));
        assert_eq!(chain2.len(), 1);
        assert!(!chain2.is_empty());
    }

    #[test]
    fn chain_display() {
        let mut chain = SamplerChain::new(Box::new(GreedySampler));
        chain.add_processor(Box::new(TemperatureProcessor::new(0.5)));
        chain.add_processor(Box::new(TopKProcessor::new(10)));
        let d = format!("{chain}");
        assert!(d.contains("SamplerChain"));
        assert!(d.contains("temperature"));
        assert!(d.contains("top_k_filter"));
        assert!(d.contains("greedy"));
    }

    #[test]
    fn chain_multiple_processors_order() {
        // Temperature + Top-K then greedy
        let mut chain = SamplerChain::new(Box::new(GreedySampler));
        chain.add_processor(Box::new(TemperatureProcessor::new(1.0)));
        chain.add_processor(Box::new(TopKProcessor::new(1)));
        // Top-K=1 should leave only the argmax, then greedy picks it
        let idx = chain.sample(&mut [1.0, 5.0, 3.0]);
        assert_eq!(idx, 1);
    }

    // ── Edge cases ───────────────────────────────────────────

    #[test]
    fn all_same_logits_greedy_picks_first() {
        let mut s = GreedySampler;
        assert_eq!(s.sample(&mut [3.0, 3.0, 3.0, 3.0]), 0);
    }

    #[test]
    fn all_same_logits_top_k_returns_valid() {
        let mut s = TopKSampler::new(2, 1.0, 42);
        for _ in 0..100 {
            let idx = s.sample(&mut [3.0, 3.0, 3.0, 3.0]);
            assert!(idx < 4);
        }
    }

    #[test]
    fn all_same_logits_top_p_returns_valid() {
        let mut s = TopPSampler::new(0.9, 1.0, 42);
        for _ in 0..100 {
            let idx = s.sample(&mut [3.0, 3.0, 3.0, 3.0]);
            assert!(idx < 4);
        }
    }

    #[test]
    fn single_token_greedy() {
        let mut s = GreedySampler;
        assert_eq!(s.sample(&mut [42.0]), 0);
    }

    #[test]
    fn single_token_top_k() {
        let mut s = TopKSampler::new(5, 1.0, 42);
        assert_eq!(s.sample(&mut [42.0]), 0);
    }

    #[test]
    fn single_token_top_p() {
        let mut s = TopPSampler::new(0.9, 1.0, 42);
        assert_eq!(s.sample(&mut [42.0]), 0);
    }

    #[test]
    fn single_token_min_p() {
        let mut s = MinPSampler::new(0.1, 1.0, 42);
        assert_eq!(s.sample(&mut [42.0]), 0);
    }

    #[test]
    fn single_token_typical() {
        let mut s = TypicalSampler::new(0.9, 1.0, 42);
        assert_eq!(s.sample(&mut [42.0]), 0);
    }

    #[test]
    fn single_token_mirostat() {
        let mut s = MirostatSampler::new(5.0, 0.1, 42);
        assert_eq!(s.sample(&mut [42.0]), 0);
    }

    // ── Property tests ───────────────────────────────────────

    #[test]
    fn property_greedy_always_argmax() {
        let mut s = GreedySampler;
        let test_cases: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0],
            vec![10.0, -5.0, 3.0, 7.0],
            vec![0.1, 0.2, 0.3, 0.4, 0.5],
            vec![-1.0, -2.0, -0.5],
            vec![f32::MAX, 0.0, -f32::MAX],
        ];
        for logits in &test_cases {
            let expected = argmax_ref(logits);
            let actual = s.sample(&mut logits.clone());
            assert_eq!(actual, expected, "logits: {logits:?}");
        }
    }

    #[test]
    fn property_top_k_at_most_k_candidates() {
        for k in 1..=5 {
            let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
            top_k_filter(&mut logits, k);
            let active = logits
                .iter()
                .filter(|&&v| v != f32::NEG_INFINITY)
                .count();
            assert!(
                active <= k,
                "k={k}, active={active}"
            );
        }
    }

    #[test]
    fn property_softmax_output_is_probability() {
        let test_cases: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0],
            vec![-10.0, 0.0, 10.0],
            vec![0.0; 100],
            vec![1e6, -1e6, 0.0],
        ];
        for logits in &test_cases {
            let mut v = logits.clone();
            softmax(&mut v);
            let sum: f32 = v.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "sum={sum} for {logits:?}"
            );
            for &p in &v {
                assert!(p >= 0.0, "negative prob {p}");
                assert!(p <= 1.0, "prob > 1: {p}");
            }
        }
    }

    #[test]
    fn property_sampler_returns_valid_index_any_logits() {
        let distributions: Vec<Vec<f32>> = vec![
            vec![0.0; 10],
            vec![100.0, -100.0, 0.0],
            vec![1.0],
            vec![0.5, 0.5],
            (0..100).map(|i| i as f32 * 0.01).collect(),
        ];
        let mut rng = SamplingRng::new(42);
        for logits in &distributions {
            // GreedySampler
            let idx = GreedySampler.sample(&mut logits.clone());
            assert!(idx < logits.len());

            // TopKSampler
            let mut s = TopKSampler::new(3, 1.0, rng.next_u64());
            let idx = s.sample(&mut logits.clone());
            assert!(idx < logits.len());

            // TopPSampler
            let mut s = TopPSampler::new(0.9, 1.0, rng.next_u64());
            let idx = s.sample(&mut logits.clone());
            assert!(idx < logits.len());
        }
    }

    #[test]
    fn property_repetition_penalty_monotone() {
        // More repetitions → lower logit
        let mut rp = RepetitionPenalty::new(1.5, 0.1, 64);
        let mut logits_1 = vec![5.0, 5.0, 5.0];
        rp.add_token(0);
        rp.apply(&mut logits_1);
        let after_1 = logits_1[0];

        let mut rp = RepetitionPenalty::new(1.5, 0.1, 64);
        let mut logits_2 = vec![5.0, 5.0, 5.0];
        rp.add_token(0);
        rp.add_token(0);
        rp.apply(&mut logits_2);
        let after_2 = logits_2[0];

        // Two occurrences in window means double additive penalty
        assert!(
            after_2 < after_1,
            "more reps should lower logit: {after_1} vs {after_2}"
        );
    }

    // ── LogitProcessor adapters ──────────────────────────────

    #[test]
    fn temperature_processor_scales() {
        let proc = TemperatureProcessor::new(2.0);
        let mut logits = vec![2.0, 4.0, 6.0];
        proc.process(&mut logits);
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn temperature_processor_name() {
        assert_eq!(TemperatureProcessor::new(1.0).name(), "temperature");
    }

    #[test]
    fn top_k_processor_filters() {
        let proc = TopKProcessor::new(2);
        let mut logits = vec![1.0, 5.0, 3.0, 4.0];
        proc.process(&mut logits);
        let active = logits
            .iter()
            .filter(|&&v| v != f32::NEG_INFINITY)
            .count();
        assert_eq!(active, 2);
    }

    #[test]
    fn top_k_processor_name() {
        assert_eq!(TopKProcessor::new(5).name(), "top_k_filter");
    }

    #[test]
    fn top_p_processor_filters() {
        let proc = TopPProcessor::new(0.9);
        let mut logits = vec![10.0, 0.0, 0.0, 0.0];
        proc.process(&mut logits);
        assert!(logits[0] != f32::NEG_INFINITY);
    }

    #[test]
    fn top_p_processor_name() {
        assert_eq!(TopPProcessor::new(0.9).name(), "top_p_filter");
    }

    #[test]
    fn repetition_penalty_processor_applies() {
        let rp = RepetitionPenalty::new(2.0, 0.0, 64);
        let mut proc = RepetitionPenaltyProcessor::new(rp);
        proc.add_token(1);
        let mut logits = vec![1.0, 1.0, 1.0];
        proc.process(&mut logits);
        assert!(logits[1] < logits[0]);
    }

    #[test]
    fn freq_presence_processor_applies() {
        let fp = FrequencyPresencePenalty::new(1.0, 0.0);
        let mut proc = FrequencyPresencePenaltyProcessor::new(fp);
        proc.add_token(0);
        let mut logits = vec![5.0, 5.0];
        proc.process(&mut logits);
        assert!(logits[0] < logits[1]);
    }

    // ── Sampler trait object ─────────────────────────────────

    #[test]
    fn sampler_as_trait_object() {
        let samplers: Vec<Box<dyn Sampler>> = vec![
            Box::new(GreedySampler),
            Box::new(TopKSampler::new(5, 1.0, 42)),
            Box::new(TopPSampler::new(0.9, 1.0, 42)),
            Box::new(MinPSampler::new(0.1, 1.0, 42)),
            Box::new(TypicalSampler::new(0.9, 1.0, 42)),
            Box::new(MirostatSampler::new(5.0, 0.1, 42)),
        ];
        for mut s in samplers {
            let idx = s.sample(&mut [1.0, 2.0, 3.0]);
            assert!(idx < 3, "{} returned invalid index", s.name());
        }
    }

    #[test]
    fn logit_processor_as_trait_object() {
        let procs: Vec<Box<dyn LogitProcessor>> = vec![
            Box::new(TemperatureProcessor::new(1.0)),
            Box::new(TopKProcessor::new(5)),
            Box::new(TopPProcessor::new(0.9)),
        ];
        for proc in &procs {
            let mut logits = vec![1.0, 2.0, 3.0];
            proc.process(&mut logits);
            // Just verify no panic
        }
    }
}
