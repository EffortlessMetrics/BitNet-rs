//! Dense (non-quantized) generation pipeline with token sampling.
//!
//! Provides streaming token generation for dense SLM models (Phi-4, Qwen,
//! Gemma, Mistral, LLaMA) with configurable sampling strategies including
//! temperature scaling, top-k/top-p filtering, and repetition penalty.

use std::time::Instant;

// ── Finish reason ────────────────────────────────────────────────────────────

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Hit the configured `max_tokens` limit.
    MaxTokens,
    /// Encountered a token in the `stop_tokens` set.
    StopToken,
    /// Model emitted an end-of-sequence signal.
    EndOfSequence,
}

// ── Generation config ────────────────────────────────────────────────────────

/// Configuration for dense model token generation.
#[derive(Debug, Clone)]
pub struct DenseGenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: f32,
    pub stop_tokens: Vec<u32>,
    pub seed: Option<u64>,
}

impl Default for DenseGenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
            stop_tokens: Vec::new(),
            seed: None,
        }
    }
}

impl DenseGenerationConfig {
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = penalty;
        self
    }

    pub fn with_stop_tokens(mut self, tokens: Vec<u32>) -> Self {
        self.stop_tokens = tokens;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Returns `true` when `token` is in the configured stop set.
    pub fn is_stop_token(&self, token: u32) -> bool {
        self.stop_tokens.contains(&token)
    }
}

// ── Generation state ─────────────────────────────────────────────────────────

/// Tracks progress of an in-flight generation.
#[derive(Debug, Clone)]
pub struct DenseGenerationState {
    pub tokens_generated: Vec<u32>,
    pub total_time_ms: u64,
    pub is_finished: bool,
    pub finish_reason: Option<FinishReason>,
}

impl DenseGenerationState {
    pub fn new() -> Self {
        Self {
            tokens_generated: Vec::new(),
            total_time_ms: 0,
            is_finished: false,
            finish_reason: None,
        }
    }

    /// Record a new token and optionally finish generation.
    pub fn push_token(&mut self, token: u32, config: &DenseGenerationConfig) {
        self.tokens_generated.push(token);

        if config.is_stop_token(token) {
            self.is_finished = true;
            self.finish_reason = Some(FinishReason::StopToken);
        } else if self.tokens_generated.len() >= config.max_tokens {
            self.is_finished = true;
            self.finish_reason = Some(FinishReason::MaxTokens);
        }
    }

    /// Mark generation as finished due to EOS.
    pub fn finish_eos(&mut self) {
        self.is_finished = true;
        self.finish_reason = Some(FinishReason::EndOfSequence);
    }
}

impl Default for DenseGenerationState {
    fn default() -> Self {
        Self::new()
    }
}

// ── Generation step ──────────────────────────────────────────────────────────

/// Metadata for a single generated token.
#[derive(Debug, Clone)]
pub struct DenseGenerationStep {
    pub token_id: u32,
    pub logit: f32,
    pub probability: f32,
    pub step_time_us: u64,
}

impl DenseGenerationStep {
    pub fn new(token_id: u32, logit: f32, probability: f32, step_time_us: u64) -> Self {
        Self { token_id, logit, probability, step_time_us }
    }

    /// Build a step by timing the provided closure.
    pub fn timed<F>(f: F) -> Self
    where
        F: FnOnce() -> (u32, f32, f32),
    {
        let start = Instant::now();
        let (token_id, logit, probability) = f();
        let step_time_us = start.elapsed().as_micros() as u64;
        Self { token_id, logit, probability, step_time_us }
    }
}

// ── Reusable sampling buffer ─────────────────────────────────────────────────

/// Pre-allocated buffers for [`DenseTokenSampler`] to avoid per-token
/// allocations in the sampling hot path. Create once, pass to
/// [`DenseTokenSampler::sample_token_with_buffer`] on every token.
#[derive(Debug, Clone, Default)]
pub struct SamplingBuffer {
    /// Working copy of logits (modified in-place through the pipeline).
    logits: Vec<f32>,
    /// Scratch space for indexed sorts (top-k / top-p).
    indexed: Vec<(usize, f32)>,
    /// Scratch space for probability distributions.
    probs: Vec<f32>,
}

impl SamplingBuffer {
    /// Create a new, empty buffer. It will grow to vocab size on first use.
    pub fn new() -> Self {
        Self::default()
    }
}

// ── Token sampler ────────────────────────────────────────────────────────────

/// Stateless sampler operating on raw `f32` logit slices.
pub struct DenseTokenSampler;

impl DenseTokenSampler {
    // ── temperature ──────────────────────────────────────────────────────

    /// Scale logits by temperature. `temp <= 0` is treated as greedy (returns
    /// the argmax logit as-is, all others set to `f32::NEG_INFINITY`).
    pub fn temperature_scale(logits: &[f32], temp: f32) -> Vec<f32> {
        if logits.is_empty() {
            return Vec::new();
        }
        if temp <= 0.0 {
            return Self::greedy_mask(logits);
        }
        logits.iter().map(|&l| l / temp).collect()
    }

    // ── top-k ────────────────────────────────────────────────────────────

    /// Keep only the `k` largest logits; set the rest to `NEG_INFINITY`.
    /// `k == 0` or `k >= logits.len()` is a no-op.
    pub fn top_k_filter(logits: &[f32], k: usize) -> Vec<f32> {
        if k == 0 || k >= logits.len() {
            return logits.to_vec();
        }
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut out = vec![f32::NEG_INFINITY; logits.len()];
        for &(idx, val) in &indexed[..k] {
            out[idx] = val;
        }
        out
    }

    // ── top-p (nucleus) ──────────────────────────────────────────────────

    /// Keep the smallest set of tokens whose cumulative probability ≥ `p`.
    /// Operates on raw logits (applies softmax internally).
    pub fn top_p_filter(logits: &[f32], p: f32) -> Vec<f32> {
        if logits.is_empty() {
            return Vec::new();
        }
        if p >= 1.0 {
            return logits.to_vec();
        }
        if p <= 0.0 {
            // Keep only the top-1 token.
            return Self::greedy_mask(logits);
        }

        let probs = Self::softmax(logits);
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0f32;
        let mut keep = vec![false; logits.len()];
        for &(idx, prob) in &indexed {
            keep[idx] = true;
            cumsum += prob;
            if cumsum >= p {
                break;
            }
        }

        logits
            .iter()
            .enumerate()
            .map(|(i, &l)| if keep[i] { l } else { f32::NEG_INFINITY })
            .collect()
    }

    // ── repetition penalty ───────────────────────────────────────────────

    /// Penalise tokens that appeared in `past_tokens`.
    /// Positive logits are divided by `penalty`, negative logits are multiplied.
    pub fn apply_repetition_penalty(logits: &[f32], past_tokens: &[u32], penalty: f32) -> Vec<f32> {
        if penalty == 1.0 || past_tokens.is_empty() {
            return logits.to_vec();
        }
        let mut out = logits.to_vec();
        for &tok in past_tokens {
            let idx = tok as usize;
            if idx < out.len() {
                if out[idx] > 0.0 {
                    out[idx] /= penalty;
                } else {
                    out[idx] *= penalty;
                }
            }
        }
        out
    }

    // ── sampling ─────────────────────────────────────────────────────────

    /// Full sampling pipeline: temperature → top-k → top-p → repetition
    /// penalty → categorical sample (or greedy if temp ≤ 0).
    pub fn sample_token(
        logits: &[f32],
        config: &DenseGenerationConfig,
        past_tokens: &[u32],
    ) -> Option<u32> {
        if logits.is_empty() {
            return None;
        }

        let mut l = Self::temperature_scale(logits, config.temperature);

        if let Some(k) = config.top_k {
            l = Self::top_k_filter(&l, k);
        }
        if let Some(p) = config.top_p {
            l = Self::top_p_filter(&l, p);
        }

        l = Self::apply_repetition_penalty(&l, past_tokens, config.repetition_penalty);

        if config.temperature <= 0.0 {
            return Some(Self::greedy_sample(&l));
        }

        // Categorical sample from softmax probabilities.
        let probs = Self::softmax(&l);
        let r = Self::seeded_random(config.seed);
        let mut cumsum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= r {
                return Some(i as u32);
            }
        }
        // Fallback to last token (rounding).
        Some((probs.len() - 1) as u32)
    }

    /// Argmax (greedy) sampling — always picks the highest logit.
    pub fn greedy_sample(logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }

    // ── helpers (pub for testing) ────────────────────────────────────────

    /// Numerically-stable softmax over a logit slice.
    pub fn softmax(logits: &[f32]) -> Vec<f32> {
        if logits.is_empty() {
            return Vec::new();
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum == 0.0 {
            // All -inf logits → uniform over non-neg-inf entries.
            return vec![0.0; logits.len()];
        }
        exps.iter().map(|&e| e / sum).collect()
    }

    /// Greedy mask: max element keeps its value, rest → NEG_INFINITY.
    fn greedy_mask(logits: &[f32]) -> Vec<f32> {
        let max_idx = Self::greedy_sample(logits) as usize;
        logits
            .iter()
            .enumerate()
            .map(|(i, &l)| if i == max_idx { l } else { f32::NEG_INFINITY })
            .collect()
    }

    /// Deterministic pseudo-random in `[0, 1)` from an optional seed.
    fn seeded_random(seed: Option<u64>) -> f32 {
        match seed {
            Some(s) => {
                // Simple splitmix-style hash for determinism.
                let mut z = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                z ^= z >> 31;
                (z as f32) / (u64::MAX as f32)
            }
            None => rand::random::<f32>(),
        }
    }

    // ── Buffer-reusing pipeline (zero per-token allocation) ──────────────

    /// Full sampling pipeline that reuses `buf` to avoid per-token heap
    /// allocations. Semantically identical to [`Self::sample_token`].
    pub fn sample_token_with_buffer(
        logits: &[f32],
        config: &DenseGenerationConfig,
        past_tokens: &[u32],
        buf: &mut SamplingBuffer,
    ) -> Option<u32> {
        if logits.is_empty() {
            return None;
        }

        // Copy logits into reusable working buffer (one memcpy, no alloc
        // after the first call because the Vec capacity is retained).
        buf.logits.clear();
        buf.logits.extend_from_slice(logits);

        // Temperature
        Self::temperature_scale_in_place(&mut buf.logits, config.temperature);

        // Top-k
        if let Some(k) = config.top_k {
            Self::top_k_filter_in_place(&mut buf.logits, k, &mut buf.indexed);
        }

        // Top-p
        if let Some(p) = config.top_p {
            Self::top_p_filter_in_place(&mut buf.logits, p, &mut buf.indexed, &mut buf.probs);
        }

        // Repetition penalty
        Self::apply_repetition_penalty_in_place(
            &mut buf.logits,
            past_tokens,
            config.repetition_penalty,
        );

        if config.temperature <= 0.0 {
            return Some(Self::greedy_sample(&buf.logits));
        }

        // Softmax → categorical sample
        Self::softmax_into(&buf.logits, &mut buf.probs);
        let r = Self::seeded_random(config.seed);
        let mut cumsum = 0.0f32;
        for (i, &p) in buf.probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= r {
                return Some(i as u32);
            }
        }
        Some((buf.probs.len() - 1) as u32)
    }

    // ── In-place helpers ─────────────────────────────────────────────────

    fn temperature_scale_in_place(logits: &mut [f32], temp: f32) {
        if logits.is_empty() {
            return;
        }
        if temp <= 0.0 {
            let max_idx = Self::greedy_sample(logits) as usize;
            for (i, l) in logits.iter_mut().enumerate() {
                if i != max_idx {
                    *l = f32::NEG_INFINITY;
                }
            }
            return;
        }
        for l in logits.iter_mut() {
            *l /= temp;
        }
    }

    fn top_k_filter_in_place(logits: &mut [f32], k: usize, indexed: &mut Vec<(usize, f32)>) {
        if k == 0 || k >= logits.len() {
            return;
        }
        indexed.clear();
        indexed.extend(logits.iter().copied().enumerate());
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        // Blank everything, then restore top-k.
        for l in logits.iter_mut() {
            *l = f32::NEG_INFINITY;
        }
        for &(idx, val) in &indexed[..k] {
            logits[idx] = val;
        }
    }

    fn top_p_filter_in_place(
        logits: &mut [f32],
        p: f32,
        indexed: &mut Vec<(usize, f32)>,
        probs: &mut Vec<f32>,
    ) {
        if logits.is_empty() || p >= 1.0 {
            return;
        }
        if p <= 0.0 {
            let max_idx = Self::greedy_sample(logits) as usize;
            for (i, l) in logits.iter_mut().enumerate() {
                if i != max_idx {
                    *l = f32::NEG_INFINITY;
                }
            }
            return;
        }

        Self::softmax_into(logits, probs);
        indexed.clear();
        indexed.extend(probs.iter().copied().enumerate());
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0f32;
        let mut cutoff = indexed.len();
        for (rank, &(_idx, prob)) in indexed.iter().enumerate() {
            cumsum += prob;
            if cumsum >= p {
                cutoff = rank + 1;
                break;
            }
        }
        for &(idx, _) in &indexed[cutoff..] {
            logits[idx] = f32::NEG_INFINITY;
        }
    }

    fn apply_repetition_penalty_in_place(logits: &mut [f32], past_tokens: &[u32], penalty: f32) {
        if penalty == 1.0 || past_tokens.is_empty() {
            return;
        }
        for &tok in past_tokens {
            let idx = tok as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    logits[idx] /= penalty;
                } else {
                    logits[idx] *= penalty;
                }
            }
        }
    }

    /// Write softmax of `logits` into `out`, reusing `out`'s allocation.
    fn softmax_into(logits: &[f32], out: &mut Vec<f32>) {
        out.clear();
        if logits.is_empty() {
            return;
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        out.extend(logits.iter().map(|&l| (l - max).exp()));
        let sum: f32 = out.iter().sum();
        if sum == 0.0 {
            for v in out.iter_mut() {
                *v = 0.0;
            }
            return;
        }
        for v in out.iter_mut() {
            *v /= sum;
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── DenseGenerationConfig defaults & builder ─────────────────────────

    #[test]
    fn test_config_defaults() {
        let cfg = DenseGenerationConfig::default();
        assert_eq!(cfg.max_tokens, 256);
        assert!((cfg.temperature - 1.0).abs() < f32::EPSILON);
        assert!(cfg.top_k.is_none());
        assert!(cfg.top_p.is_none());
        assert!((cfg.repetition_penalty - 1.0).abs() < f32::EPSILON);
        assert!(cfg.stop_tokens.is_empty());
        assert!(cfg.seed.is_none());
    }

    #[test]
    fn test_config_builder() {
        let cfg = DenseGenerationConfig::default()
            .with_max_tokens(128)
            .with_temperature(0.7)
            .with_top_k(50)
            .with_top_p(0.9)
            .with_repetition_penalty(1.2)
            .with_stop_tokens(vec![2, 3])
            .with_seed(42);

        assert_eq!(cfg.max_tokens, 128);
        assert!((cfg.temperature - 0.7).abs() < 1e-6);
        assert_eq!(cfg.top_k, Some(50));
        assert_eq!(cfg.top_p, Some(0.9));
        assert!((cfg.repetition_penalty - 1.2).abs() < 1e-6);
        assert_eq!(cfg.stop_tokens, vec![2, 3]);
        assert_eq!(cfg.seed, Some(42));
    }

    // ── Temperature scaling ──────────────────────────────────────────────

    #[test]
    fn test_temperature_zero_is_greedy() {
        let logits = vec![1.0, 3.0, 2.0];
        let scaled = DenseTokenSampler::temperature_scale(&logits, 0.0);
        // Only the max (index 1) should remain finite.
        assert!(scaled[1].is_finite());
        assert!(scaled[0] == f32::NEG_INFINITY);
        assert!(scaled[2] == f32::NEG_INFINITY);
    }

    #[test]
    fn test_temperature_one_is_identity() {
        let logits = vec![1.0, 2.0, 3.0];
        let scaled = DenseTokenSampler::temperature_scale(&logits, 1.0);
        assert_eq!(logits, scaled);
    }

    #[test]
    fn test_temperature_high_flattens() {
        let logits = vec![1.0, 5.0, 2.0];
        let scaled = DenseTokenSampler::temperature_scale(&logits, 2.0);
        // Differences shrink: (5-1)/2 = 2 vs original 4.
        let orig_spread = logits[1] - logits[0];
        let new_spread = scaled[1] - scaled[0];
        assert!(new_spread < orig_spread);
    }

    // ── Top-k filtering ──────────────────────────────────────────────────

    #[test]
    fn test_top_k_one_is_greedy() {
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let filtered = DenseTokenSampler::top_k_filter(&logits, 1);
        let finite: Vec<usize> =
            filtered.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(i, _)| i).collect();
        assert_eq!(finite, vec![1]); // index of max
    }

    #[test]
    fn test_top_k_five_keeps_five() {
        let logits: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let filtered = DenseTokenSampler::top_k_filter(&logits, 5);
        let finite_count = filtered.iter().filter(|v| v.is_finite()).count();
        assert_eq!(finite_count, 5);
    }

    #[test]
    fn test_top_k_exceeds_vocab_is_identity() {
        let logits = vec![1.0, 2.0, 3.0];
        let filtered = DenseTokenSampler::top_k_filter(&logits, 100);
        assert_eq!(logits, filtered);
    }

    // ── Top-p filtering ──────────────────────────────────────────────────

    #[test]
    fn test_top_p_zero_keeps_top_token() {
        let logits = vec![1.0, 5.0, 2.0];
        let filtered = DenseTokenSampler::top_p_filter(&logits, 0.0);
        let finite: Vec<usize> =
            filtered.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(i, _)| i).collect();
        assert_eq!(finite, vec![1]);
    }

    #[test]
    fn test_top_p_one_keeps_all() {
        let logits = vec![1.0, 2.0, 3.0];
        let filtered = DenseTokenSampler::top_p_filter(&logits, 1.0);
        assert_eq!(logits, filtered);
    }

    #[test]
    fn test_top_p_nucleus() {
        let logits = vec![0.0, 0.0, 10.0]; // softmax ≈ [~0, ~0, ~1]
        let filtered = DenseTokenSampler::top_p_filter(&logits, 0.9);
        // The dominant token (index 2) alone exceeds 0.9.
        assert!(filtered[2].is_finite());
        // At least one other should be masked.
        let finite_count = filtered.iter().filter(|v| v.is_finite()).count();
        assert!(finite_count <= 2);
    }

    // ── Repetition penalty ───────────────────────────────────────────────

    #[test]
    fn test_repetition_penalty_identity() {
        let logits = vec![1.0, 2.0, 3.0];
        let result = DenseTokenSampler::apply_repetition_penalty(&logits, &[0, 1, 2], 1.0);
        assert_eq!(logits, result);
    }

    #[test]
    fn test_repetition_penalty_reduces_repeated() {
        let logits = vec![4.0, 2.0, 3.0];
        let past = vec![0u32]; // token 0 was seen
        let result = DenseTokenSampler::apply_repetition_penalty(&logits, &past, 2.0);
        // Positive logit divided by penalty.
        assert!((result[0] - 2.0).abs() < 1e-6);
        // Others unchanged.
        assert!((result[1] - 2.0).abs() < 1e-6);
        assert!((result[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_repetition_penalty_negative_logit() {
        let logits = vec![-2.0, 1.0];
        let past = vec![0u32];
        let result = DenseTokenSampler::apply_repetition_penalty(&logits, &past, 2.0);
        // Negative logit multiplied by penalty (becomes more negative).
        assert!((result[0] - (-4.0)).abs() < 1e-6);
    }

    // ── Greedy sampling determinism ──────────────────────────────────────

    #[test]
    fn test_greedy_determinism() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let a = DenseTokenSampler::greedy_sample(&logits);
        let b = DenseTokenSampler::greedy_sample(&logits);
        assert_eq!(a, b);
        assert_eq!(a, 3); // index of 0.9
    }

    // ── Stop token detection ─────────────────────────────────────────────

    #[test]
    fn test_stop_token_detected() {
        let cfg = DenseGenerationConfig::default().with_stop_tokens(vec![50256]);
        assert!(cfg.is_stop_token(50256));
        assert!(!cfg.is_stop_token(1));
    }

    // ── FinishReason variants ────────────────────────────────────────────

    #[test]
    fn test_finish_reason_variants() {
        assert_ne!(FinishReason::MaxTokens, FinishReason::StopToken);
        assert_ne!(FinishReason::StopToken, FinishReason::EndOfSequence);
        assert_eq!(FinishReason::MaxTokens, FinishReason::MaxTokens);
    }

    // ── DenseGenerationState tracking ────────────────────────────────────

    #[test]
    fn test_state_token_accumulation() {
        let cfg = DenseGenerationConfig::default().with_max_tokens(3);
        let mut state = DenseGenerationState::new();

        state.push_token(10, &cfg);
        assert_eq!(state.tokens_generated, vec![10]);
        assert!(!state.is_finished);

        state.push_token(20, &cfg);
        assert_eq!(state.tokens_generated.len(), 2);

        state.push_token(30, &cfg);
        assert!(state.is_finished);
        assert_eq!(state.finish_reason, Some(FinishReason::MaxTokens));
    }

    #[test]
    fn test_state_stop_token_finishes() {
        let cfg = DenseGenerationConfig::default().with_stop_tokens(vec![99]);
        let mut state = DenseGenerationState::new();

        state.push_token(1, &cfg);
        assert!(!state.is_finished);

        state.push_token(99, &cfg);
        assert!(state.is_finished);
        assert_eq!(state.finish_reason, Some(FinishReason::StopToken));
    }

    #[test]
    fn test_state_eos_finish() {
        let mut state = DenseGenerationState::new();
        state.finish_eos();
        assert!(state.is_finished);
        assert_eq!(state.finish_reason, Some(FinishReason::EndOfSequence));
    }

    // ── DenseGenerationStep construction ─────────────────────────────────

    #[test]
    fn test_generation_step_new() {
        let step = DenseGenerationStep::new(42, 3.5, 0.85, 120);
        assert_eq!(step.token_id, 42);
        assert!((step.logit - 3.5).abs() < 1e-6);
        assert!((step.probability - 0.85).abs() < 1e-6);
        assert_eq!(step.step_time_us, 120);
    }

    #[test]
    fn test_generation_step_timed() {
        let step = DenseGenerationStep::timed(|| (7, 1.0, 0.5));
        assert_eq!(step.token_id, 7);
        // Timing should be non-negative (could be 0 on fast machines).
        assert!(step.step_time_us < 1_000_000);
    }

    // ── Edge cases ───────────────────────────────────────────────────────

    #[test]
    fn test_empty_logits() {
        let empty: Vec<f32> = vec![];
        assert!(DenseTokenSampler::temperature_scale(&empty, 1.0).is_empty());
        assert!(DenseTokenSampler::top_p_filter(&empty, 0.9).is_empty());
        assert!(DenseTokenSampler::softmax(&empty).is_empty());
        assert_eq!(
            DenseTokenSampler::sample_token(&empty, &DenseGenerationConfig::default(), &[]),
            None
        );
    }

    #[test]
    fn test_single_element_logits() {
        let logits = vec![5.0];
        assert_eq!(DenseTokenSampler::greedy_sample(&logits), 0);
        let scaled = DenseTokenSampler::temperature_scale(&logits, 0.5);
        assert!((scaled[0] - 10.0).abs() < 1e-6);
        let probs = DenseTokenSampler::softmax(&logits);
        assert!((probs[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_all_same_logits() {
        let logits = vec![2.0; 5];
        let probs = DenseTokenSampler::softmax(&logits);
        for &p in &probs {
            assert!((p - 0.2).abs() < 1e-6);
        }
    }

    #[test]
    fn test_nan_handling_in_greedy() {
        let logits = vec![1.0, f32::NAN, 3.0];
        // Should not panic; greedy returns a valid index.
        let idx = DenseTokenSampler::greedy_sample(&logits);
        assert!(idx < logits.len() as u32);
    }

    // ── sample_token full pipeline ───────────────────────────────────────

    #[test]
    fn test_sample_token_greedy() {
        let logits = vec![1.0, 5.0, 3.0];
        let cfg = DenseGenerationConfig::default().with_temperature(0.0);
        let tok = DenseTokenSampler::sample_token(&logits, &cfg, &[]);
        assert_eq!(tok, Some(1));
    }

    #[test]
    fn test_sample_token_deterministic_with_seed() {
        let logits = vec![1.0, 2.0, 3.0, 2.5];
        let cfg = DenseGenerationConfig::default().with_seed(42).with_temperature(1.0);
        let a = DenseTokenSampler::sample_token(&logits, &cfg, &[]);
        let b = DenseTokenSampler::sample_token(&logits, &cfg, &[]);
        assert_eq!(a, b, "same seed should produce same token");
    }

    // ── sample_token_with_buffer parity ──────────────────────────────────

    #[test]
    fn test_buffered_greedy_matches_original() {
        let logits = vec![1.0, 5.0, 3.0];
        let cfg = DenseGenerationConfig::default().with_temperature(0.0);
        let mut buf = SamplingBuffer::new();
        let original = DenseTokenSampler::sample_token(&logits, &cfg, &[]);
        let buffered = DenseTokenSampler::sample_token_with_buffer(&logits, &cfg, &[], &mut buf);
        assert_eq!(original, buffered);
    }

    #[test]
    fn test_buffered_seeded_matches_original() {
        let logits = vec![1.0, 2.0, 3.0, 2.5];
        let cfg = DenseGenerationConfig::default().with_seed(42).with_temperature(1.0);
        let mut buf = SamplingBuffer::new();
        let original = DenseTokenSampler::sample_token(&logits, &cfg, &[]);
        let buffered = DenseTokenSampler::sample_token_with_buffer(&logits, &cfg, &[], &mut buf);
        assert_eq!(original, buffered);
    }

    #[test]
    fn test_buffered_with_top_k() {
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        let cfg = DenseGenerationConfig::default().with_temperature(0.0).with_top_k(2);
        let mut buf = SamplingBuffer::new();
        let original = DenseTokenSampler::sample_token(&logits, &cfg, &[]);
        let buffered = DenseTokenSampler::sample_token_with_buffer(&logits, &cfg, &[], &mut buf);
        assert_eq!(original, buffered);
    }

    #[test]
    fn test_buffered_with_repetition_penalty() {
        let logits = vec![4.0, 2.0, 3.0];
        let past = vec![0u32];
        let cfg =
            DenseGenerationConfig::default().with_temperature(0.0).with_repetition_penalty(2.0);
        let mut buf = SamplingBuffer::new();
        let original = DenseTokenSampler::sample_token(&logits, &cfg, &past);
        let buffered = DenseTokenSampler::sample_token_with_buffer(&logits, &cfg, &past, &mut buf);
        assert_eq!(original, buffered);
    }

    #[test]
    fn test_buffered_empty_logits() {
        let empty: Vec<f32> = vec![];
        let cfg = DenseGenerationConfig::default();
        let mut buf = SamplingBuffer::new();
        assert_eq!(DenseTokenSampler::sample_token_with_buffer(&empty, &cfg, &[], &mut buf), None);
    }

    #[test]
    fn test_buffer_reuse_across_calls() {
        let mut buf = SamplingBuffer::new();
        let cfg = DenseGenerationConfig::default().with_temperature(0.0);

        let tok1 =
            DenseTokenSampler::sample_token_with_buffer(&[1.0, 5.0, 3.0], &cfg, &[], &mut buf);
        assert_eq!(tok1, Some(1));

        // Second call reuses the same buffer — no new allocation.
        let tok2 =
            DenseTokenSampler::sample_token_with_buffer(&[9.0, 2.0, 3.0], &cfg, &[], &mut buf);
        assert_eq!(tok2, Some(0));
    }
}
