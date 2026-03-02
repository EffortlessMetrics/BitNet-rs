//! Logits processing pipeline for Intel Arc A770 (OpenCL backend).
//!
//! Post-processes raw logits before sampling: temperature scaling, repetition
//! penalty, frequency penalty, presence penalty, and token banning. All
//! processors implement the [`LogitsProcessor`] trait and can be composed
//! via [`LogitsProcessorChain`].
//!
//! This module provides CPU reference implementations; the OpenCL device
//! kernels will be added when the A770 runtime integration lands.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ── ProcessContext ─────────────────────────────────────────────────────

/// Contextual information passed to each logits processor.
#[derive(Debug, Clone, Default)]
pub struct ProcessContext {
    /// Tokens generated so far (in order).
    pub generated_tokens: Vec<u32>,
    /// Per-token frequency counts (token_id → count).
    pub token_frequencies: HashMap<u32, u32>,
    /// Tokens that must be forced to `-inf`.
    pub banned_tokens: HashSet<u32>,
}

impl ProcessContext {
    /// Create an empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a newly generated token, updating both the token list and
    /// frequency map.
    pub fn record_token(&mut self, token: u32) {
        self.generated_tokens.push(token);
        *self.token_frequencies.entry(token).or_insert(0) += 1;
    }

    /// Return the last `window` generated tokens (or fewer if not enough
    /// history).
    pub fn recent_tokens(&self, window: usize) -> &[u32] {
        let start = self.generated_tokens.len().saturating_sub(window);
        &self.generated_tokens[start..]
    }
}

// ── LogitsProcessor trait ──────────────────────────────────────────────

/// Trait for a single logits post-processing step.
pub trait LogitsProcessor: fmt::Debug + Send + Sync {
    /// Apply the processing step to `logits` given `context`.
    fn process(&self, logits: &mut [f32], context: &ProcessContext);

    /// Human-readable name for diagnostics.
    fn name(&self) -> &'static str;
}

// ── TemperatureProcessor ───────────────────────────────────────────────

/// Scales logits by `1 / temperature`.
///
/// * `T = 1.0` — identity (no-op).
/// * `T < 1.0` — sharpens distribution (more confident).
/// * `T > 1.0` — flattens distribution (more random).
/// * `T = 0.0` — treated as argmax: the maximum logit gets 0.0, all
///   others get `-inf`.
#[derive(Debug, Clone)]
pub struct TemperatureProcessor {
    pub temperature: f32,
}

impl TemperatureProcessor {
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }
}

impl LogitsProcessor for TemperatureProcessor {
    fn process(&self, logits: &mut [f32], _context: &ProcessContext) {
        if logits.is_empty() {
            return;
        }
        if self.temperature == 0.0 {
            // Argmax mode: keep the maximum, set everything else to
            // -inf.
            let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for l in logits.iter_mut() {
                if *l < max_val {
                    *l = f32::NEG_INFINITY;
                }
            }
            return;
        }
        if self.temperature == 1.0 {
            return;
        }
        let inv = 1.0 / self.temperature;
        for l in logits.iter_mut() {
            *l *= inv;
        }
    }

    fn name(&self) -> &'static str {
        "TemperatureProcessor"
    }
}

// ── RepetitionPenaltyProcessor ─────────────────────────────────────────

/// Penalises repeated tokens within a configurable window.
///
/// For each token seen in the recent window, its logit is divided by
/// `penalty` when positive and multiplied by `penalty` when negative.
/// This biases sampling away from tokens that have already appeared.
#[derive(Debug, Clone)]
pub struct RepetitionPenaltyProcessor {
    pub penalty: f32,
    pub window: usize,
}

impl RepetitionPenaltyProcessor {
    /// `penalty` ≥ 1.0.  `window = 0` means consider all history.
    pub fn new(penalty: f32, window: usize) -> Self {
        Self { penalty, window }
    }
}

impl LogitsProcessor for RepetitionPenaltyProcessor {
    fn process(&self, logits: &mut [f32], context: &ProcessContext) {
        if self.penalty == 1.0 {
            return;
        }
        let tokens = if self.window == 0 {
            context.generated_tokens.as_slice()
        } else {
            context.recent_tokens(self.window)
        };
        // Deduplicate so each token is penalised at most once.
        let unique: HashSet<u32> = tokens.iter().copied().collect();
        for tok in unique {
            let idx = tok as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    logits[idx] /= self.penalty;
                } else {
                    logits[idx] *= self.penalty;
                }
            }
        }
    }

    fn name(&self) -> &'static str {
        "RepetitionPenaltyProcessor"
    }
}

// ── FrequencyPenaltyProcessor ──────────────────────────────────────────

/// Penalises tokens proportionally to how often they have appeared.
///
/// `logit -= frequency_count * penalty`
///
/// Higher `penalty` values discourage high-frequency tokens more
/// aggressively.
#[derive(Debug, Clone)]
pub struct FrequencyPenaltyProcessor {
    pub penalty: f32,
}

impl FrequencyPenaltyProcessor {
    pub fn new(penalty: f32) -> Self {
        Self { penalty }
    }
}

impl LogitsProcessor for FrequencyPenaltyProcessor {
    fn process(&self, logits: &mut [f32], context: &ProcessContext) {
        if self.penalty == 0.0 {
            return;
        }
        for (&tok, &count) in &context.token_frequencies {
            let idx = tok as usize;
            if idx < logits.len() {
                logits[idx] -= (count as f32) * self.penalty;
            }
        }
    }

    fn name(&self) -> &'static str {
        "FrequencyPenaltyProcessor"
    }
}

// ── PresencePenaltyProcessor ───────────────────────────────────────────

/// Applies a flat penalty to any token that has appeared at least once.
///
/// `logit -= penalty` for every token present in context.
#[derive(Debug, Clone)]
pub struct PresencePenaltyProcessor {
    pub penalty: f32,
}

impl PresencePenaltyProcessor {
    pub fn new(penalty: f32) -> Self {
        Self { penalty }
    }
}

impl LogitsProcessor for PresencePenaltyProcessor {
    fn process(&self, logits: &mut [f32], context: &ProcessContext) {
        if self.penalty == 0.0 {
            return;
        }
        for &tok in context.token_frequencies.keys() {
            let idx = tok as usize;
            if idx < logits.len() {
                logits[idx] -= self.penalty;
            }
        }
    }

    fn name(&self) -> &'static str {
        "PresencePenaltyProcessor"
    }
}

// ── TokenBanProcessor ──────────────────────────────────────────────────

/// Forces specified tokens to `-inf` so they can never be sampled.
#[derive(Debug, Clone)]
pub struct TokenBanProcessor;

impl LogitsProcessor for TokenBanProcessor {
    fn process(&self, logits: &mut [f32], context: &ProcessContext) {
        for &tok in &context.banned_tokens {
            let idx = tok as usize;
            if idx < logits.len() {
                logits[idx] = f32::NEG_INFINITY;
            }
        }
    }

    fn name(&self) -> &'static str {
        "TokenBanProcessor"
    }
}

// ── LogitsProcessorChain ───────────────────────────────────────────────

/// Applies a sequence of [`LogitsProcessor`]s in order.
#[derive(Debug, Default)]
pub struct LogitsProcessorChain {
    processors: Vec<Box<dyn LogitsProcessor>>,
}

impl LogitsProcessorChain {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a processor to the end of the chain.
    pub fn push(&mut self, processor: Box<dyn LogitsProcessor>) {
        self.processors.push(processor);
    }

    /// Number of processors in the chain.
    pub fn len(&self) -> usize {
        self.processors.len()
    }

    /// Whether the chain contains any processors.
    pub fn is_empty(&self) -> bool {
        self.processors.is_empty()
    }

    /// Execute every processor in order.
    pub fn process(&self, logits: &mut [f32], context: &ProcessContext) {
        for p in &self.processors {
            p.process(logits, context);
        }
    }

    /// Return the names of all processors, in order.
    pub fn names(&self) -> Vec<&'static str> {
        self.processors.iter().map(|p| p.name()).collect()
    }
}

// ── CPU helper: softmax ────────────────────────────────────────────────

/// Numerically-stable softmax in-place. Returns the log-sum-exp for
/// verification.
pub fn cpu_softmax(logits: &mut [f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for l in logits.iter_mut() {
        *l = (*l - max).exp();
        sum += *l;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for l in logits.iter_mut() {
            *l *= inv;
        }
    }
    max + sum.ln()
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // Helpers ───────────────────────────────────────────────────────────

    fn sample_logits() -> Vec<f32> {
        vec![1.0, 2.0, 3.0, 4.0, 5.0]
    }

    fn uniform_logits(n: usize) -> Vec<f32> {
        vec![1.0; n]
    }

    fn context_with_tokens(tokens: &[u32]) -> ProcessContext {
        let mut ctx = ProcessContext::new();
        for &t in tokens {
            ctx.record_token(t);
        }
        ctx
    }

    fn assert_close(a: f32, b: f32, eps: f32) {
        assert!((a - b).abs() < eps, "expected {a} ≈ {b} (eps={eps})");
    }

    fn prob_mass(logits: &[f32]) -> f32 {
        let mut copy = logits.to_vec();
        cpu_softmax(&mut copy);
        copy.iter().sum()
    }

    // ── TemperatureProcessor ──────────────────────────────────────────

    #[test]
    fn temperature_identity_at_1() {
        let mut logits = sample_logits();
        let orig = logits.clone();
        let p = TemperatureProcessor::new(1.0);
        p.process(&mut logits, &ProcessContext::new());
        assert_eq!(logits, orig);
    }

    #[test]
    fn temperature_half_doubles_logits() {
        let mut logits = sample_logits();
        let p = TemperatureProcessor::new(0.5);
        p.process(&mut logits, &ProcessContext::new());
        assert_eq!(logits, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn temperature_two_halves_logits() {
        let mut logits = sample_logits();
        let p = TemperatureProcessor::new(2.0);
        p.process(&mut logits, &ProcessContext::new());
        assert_eq!(logits, vec![0.5, 1.0, 1.5, 2.0, 2.5]);
    }

    #[test]
    fn temperature_zero_is_argmax() {
        let mut logits = sample_logits();
        let p = TemperatureProcessor::new(0.0);
        p.process(&mut logits, &ProcessContext::new());
        // Only the max (index 4) should survive.
        assert_eq!(logits[4], 5.0);
        for i in 0..4 {
            assert_eq!(logits[i], f32::NEG_INFINITY);
        }
    }

    #[test]
    fn temperature_zero_keeps_ties() {
        let mut logits = vec![1.0, 5.0, 5.0, 2.0];
        let p = TemperatureProcessor::new(0.0);
        p.process(&mut logits, &ProcessContext::new());
        assert_eq!(logits[1], 5.0);
        assert_eq!(logits[2], 5.0);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    #[test]
    fn temperature_empty_logits() {
        let mut logits: Vec<f32> = vec![];
        TemperatureProcessor::new(0.5).process(&mut logits, &ProcessContext::new());
        assert!(logits.is_empty());
    }

    #[test]
    fn temperature_very_small() {
        let mut logits = vec![1.0, 2.0];
        TemperatureProcessor::new(0.01).process(&mut logits, &ProcessContext::new());
        assert_close(logits[0], 100.0, 0.01);
        assert_close(logits[1], 200.0, 0.01);
    }

    #[test]
    fn temperature_very_large() {
        let mut logits = vec![100.0, 200.0];
        TemperatureProcessor::new(100.0).process(&mut logits, &ProcessContext::new());
        assert_close(logits[0], 1.0, 0.01);
        assert_close(logits[1], 2.0, 0.01);
    }

    // ── RepetitionPenaltyProcessor ────────────────────────────────────

    #[test]
    fn repetition_penalty_no_history() {
        let mut logits = sample_logits();
        let orig = logits.clone();
        let p = RepetitionPenaltyProcessor::new(2.0, 0);
        p.process(&mut logits, &ProcessContext::new());
        assert_eq!(logits, orig);
    }

    #[test]
    fn repetition_penalty_positive_logit() {
        let mut logits = vec![4.0, 2.0, 1.0];
        let ctx = context_with_tokens(&[0]);
        RepetitionPenaltyProcessor::new(2.0, 0).process(&mut logits, &ctx);
        assert_close(logits[0], 2.0, 1e-6);
        assert_close(logits[1], 2.0, 1e-6); // untouched
    }

    #[test]
    fn repetition_penalty_negative_logit() {
        let mut logits = vec![-4.0, 2.0];
        let ctx = context_with_tokens(&[0]);
        RepetitionPenaltyProcessor::new(2.0, 0).process(&mut logits, &ctx);
        assert_close(logits[0], -8.0, 1e-6);
    }

    #[test]
    fn repetition_penalty_identity() {
        let mut logits = sample_logits();
        let orig = logits.clone();
        let ctx = context_with_tokens(&[0, 1, 2]);
        RepetitionPenaltyProcessor::new(1.0, 0).process(&mut logits, &ctx);
        assert_eq!(logits, orig);
    }

    #[test]
    fn repetition_penalty_window_respected() {
        let mut logits = vec![10.0, 10.0, 10.0, 10.0];
        let ctx = context_with_tokens(&[0, 1, 2, 3]);
        // Window of 2: only tokens 2, 3 should be penalised.
        RepetitionPenaltyProcessor::new(2.0, 2).process(&mut logits, &ctx);
        assert_close(logits[0], 10.0, 1e-6);
        assert_close(logits[1], 10.0, 1e-6);
        assert_close(logits[2], 5.0, 1e-6);
        assert_close(logits[3], 5.0, 1e-6);
    }

    #[test]
    fn repetition_penalty_window_larger_than_history() {
        let mut logits = vec![10.0, 10.0];
        let ctx = context_with_tokens(&[0]);
        RepetitionPenaltyProcessor::new(2.0, 100).process(&mut logits, &ctx);
        assert_close(logits[0], 5.0, 1e-6);
    }

    #[test]
    fn repetition_penalty_duplicate_tokens() {
        let mut logits = vec![8.0, 4.0];
        // Token 0 appears twice — should still only be penalised once.
        let ctx = context_with_tokens(&[0, 0]);
        RepetitionPenaltyProcessor::new(2.0, 0).process(&mut logits, &ctx);
        assert_close(logits[0], 4.0, 1e-6);
    }

    #[test]
    fn repetition_penalty_out_of_range_token() {
        let mut logits = vec![1.0, 2.0];
        let ctx = context_with_tokens(&[99]); // out of range
        RepetitionPenaltyProcessor::new(2.0, 0).process(&mut logits, &ctx);
        assert_eq!(logits, vec![1.0, 2.0]);
    }

    // ── FrequencyPenaltyProcessor ─────────────────────────────────────

    #[test]
    fn frequency_penalty_basic() {
        let mut logits = vec![5.0, 5.0, 5.0];
        let ctx = context_with_tokens(&[0, 0, 0, 1]);
        FrequencyPenaltyProcessor::new(1.0).process(&mut logits, &ctx);
        // token 0 appeared 3 times → 5 - 3*1 = 2
        // token 1 appeared 1 time  → 5 - 1*1 = 4
        assert_close(logits[0], 2.0, 1e-6);
        assert_close(logits[1], 4.0, 1e-6);
        assert_close(logits[2], 5.0, 1e-6);
    }

    #[test]
    fn frequency_penalty_zero_is_noop() {
        let mut logits = sample_logits();
        let orig = logits.clone();
        let ctx = context_with_tokens(&[0, 0, 0]);
        FrequencyPenaltyProcessor::new(0.0).process(&mut logits, &ctx);
        assert_eq!(logits, orig);
    }

    #[test]
    fn frequency_penalty_fractional() {
        let mut logits = vec![10.0, 10.0];
        let ctx = context_with_tokens(&[0, 0, 1]);
        FrequencyPenaltyProcessor::new(0.5).process(&mut logits, &ctx);
        // token 0: 10 - 2*0.5 = 9
        // token 1: 10 - 1*0.5 = 9.5
        assert_close(logits[0], 9.0, 1e-6);
        assert_close(logits[1], 9.5, 1e-6);
    }

    #[test]
    fn frequency_penalty_empty_context() {
        let mut logits = sample_logits();
        let orig = logits.clone();
        FrequencyPenaltyProcessor::new(2.0).process(&mut logits, &ProcessContext::new());
        assert_eq!(logits, orig);
    }

    #[test]
    fn frequency_penalty_large_count() {
        let mut logits = vec![100.0];
        let mut ctx = ProcessContext::new();
        for _ in 0..1000 {
            ctx.record_token(0);
        }
        FrequencyPenaltyProcessor::new(0.1).process(&mut logits, &ctx);
        assert_close(logits[0], 0.0, 1e-3);
    }

    // ── PresencePenaltyProcessor ──────────────────────────────────────

    #[test]
    fn presence_penalty_basic() {
        let mut logits = vec![5.0, 5.0, 5.0];
        let ctx = context_with_tokens(&[0, 0, 0, 1]);
        PresencePenaltyProcessor::new(2.0).process(&mut logits, &ctx);
        // Both 0 and 1 are present → each gets -2.0.
        assert_close(logits[0], 3.0, 1e-6);
        assert_close(logits[1], 3.0, 1e-6);
        assert_close(logits[2], 5.0, 1e-6);
    }

    #[test]
    fn presence_penalty_zero_is_noop() {
        let mut logits = sample_logits();
        let orig = logits.clone();
        let ctx = context_with_tokens(&[0, 1]);
        PresencePenaltyProcessor::new(0.0).process(&mut logits, &ctx);
        assert_eq!(logits, orig);
    }

    #[test]
    fn presence_penalty_ignores_frequency() {
        let mut logits = vec![10.0, 10.0];
        let ctx = context_with_tokens(&[0, 0, 0, 1]);
        PresencePenaltyProcessor::new(1.0).process(&mut logits, &ctx);
        // Both get the same flat penalty regardless of frequency.
        assert_close(logits[0], 9.0, 1e-6);
        assert_close(logits[1], 9.0, 1e-6);
    }

    #[test]
    fn presence_penalty_empty_context() {
        let mut logits = sample_logits();
        let orig = logits.clone();
        PresencePenaltyProcessor::new(5.0).process(&mut logits, &ProcessContext::new());
        assert_eq!(logits, orig);
    }

    // ── TokenBanProcessor ─────────────────────────────────────────────

    #[test]
    fn ban_single_token() {
        let mut logits = sample_logits();
        let mut ctx = ProcessContext::new();
        ctx.banned_tokens.insert(2);
        TokenBanProcessor.process(&mut logits, &ctx);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[0], 1.0);
    }

    #[test]
    fn ban_multiple_tokens() {
        let mut logits = sample_logits();
        let mut ctx = ProcessContext::new();
        ctx.banned_tokens.insert(0);
        ctx.banned_tokens.insert(4);
        TokenBanProcessor.process(&mut logits, &ctx);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[4], f32::NEG_INFINITY);
        assert_eq!(logits[2], 3.0);
    }

    #[test]
    fn ban_out_of_range_token() {
        let mut logits = sample_logits();
        let mut ctx = ProcessContext::new();
        ctx.banned_tokens.insert(999);
        TokenBanProcessor.process(&mut logits, &ctx);
        assert_eq!(logits, sample_logits());
    }

    #[test]
    fn ban_all_tokens() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let mut ctx = ProcessContext::new();
        ctx.banned_tokens.extend([0, 1, 2]);
        TokenBanProcessor.process(&mut logits, &ctx);
        assert!(logits.iter().all(|&l| l == f32::NEG_INFINITY));
    }

    #[test]
    fn ban_empty_ban_set() {
        let mut logits = sample_logits();
        let orig = logits.clone();
        TokenBanProcessor.process(&mut logits, &ProcessContext::new());
        assert_eq!(logits, orig);
    }

    // ── LogitsProcessorChain ──────────────────────────────────────────

    #[test]
    fn chain_empty() {
        let chain = LogitsProcessorChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
        let mut logits = sample_logits();
        let orig = logits.clone();
        chain.process(&mut logits, &ProcessContext::new());
        assert_eq!(logits, orig);
    }

    #[test]
    fn chain_single_processor() {
        let mut chain = LogitsProcessorChain::new();
        chain.push(Box::new(TemperatureProcessor::new(0.5)));
        assert_eq!(chain.len(), 1);
        let mut logits = sample_logits();
        chain.process(&mut logits, &ProcessContext::new());
        assert_eq!(logits, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn chain_multiple_processors() {
        let mut chain = LogitsProcessorChain::new();
        chain.push(Box::new(TemperatureProcessor::new(0.5)));
        chain.push(Box::new(PresencePenaltyProcessor::new(1.0)));
        let ctx = context_with_tokens(&[0]);
        let mut logits = vec![4.0, 2.0];
        chain.process(&mut logits, &ctx);
        // After temperature 0.5: [8.0, 4.0]
        // After presence -1.0 on token 0: [7.0, 4.0]
        assert_close(logits[0], 7.0, 1e-6);
        assert_close(logits[1], 4.0, 1e-6);
    }

    #[test]
    fn chain_names() {
        let mut chain = LogitsProcessorChain::new();
        chain.push(Box::new(TemperatureProcessor::new(1.0)));
        chain.push(Box::new(TokenBanProcessor));
        assert_eq!(chain.names(), vec!["TemperatureProcessor", "TokenBanProcessor"]);
    }

    #[test]
    fn chain_ordering_matters() {
        // Demonstrate that order of processors affects the result.
        let ctx = context_with_tokens(&[0]);

        // Order A: temperature then presence
        let mut chain_a = LogitsProcessorChain::new();
        chain_a.push(Box::new(TemperatureProcessor::new(0.5)));
        chain_a.push(Box::new(PresencePenaltyProcessor::new(1.0)));
        let mut logits_a = vec![4.0, 2.0];
        chain_a.process(&mut logits_a, &ctx);

        // Order B: presence then temperature
        let mut chain_b = LogitsProcessorChain::new();
        chain_b.push(Box::new(PresencePenaltyProcessor::new(1.0)));
        chain_b.push(Box::new(TemperatureProcessor::new(0.5)));
        let mut logits_b = vec![4.0, 2.0];
        chain_b.process(&mut logits_b, &ctx);

        // Results should differ.
        assert_ne!(logits_a, logits_b);
    }

    // ── Softmax after processing ──────────────────────────────────────

    #[test]
    fn softmax_preserves_probability_mass() {
        let mut logits = sample_logits();
        cpu_softmax(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert_close(sum, 1.0, 1e-5);
    }

    #[test]
    fn softmax_after_temperature() {
        let mut logits = sample_logits();
        TemperatureProcessor::new(0.5).process(&mut logits, &ProcessContext::new());
        cpu_softmax(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert_close(sum, 1.0, 1e-5);
    }

    #[test]
    fn softmax_after_chain() {
        let mut chain = LogitsProcessorChain::new();
        chain.push(Box::new(TemperatureProcessor::new(0.5)));
        chain.push(Box::new(RepetitionPenaltyProcessor::new(1.5, 0)));
        let ctx = context_with_tokens(&[0, 1]);
        let mut logits = sample_logits();
        chain.process(&mut logits, &ctx);
        cpu_softmax(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert_close(sum, 1.0, 1e-5);
    }

    #[test]
    fn softmax_with_banned_tokens() {
        let mut logits = sample_logits();
        let mut ctx = ProcessContext::new();
        ctx.banned_tokens.insert(0);
        ctx.banned_tokens.insert(1);
        TokenBanProcessor.process(&mut logits, &ctx);
        cpu_softmax(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert_close(sum, 1.0, 1e-5);
        assert_close(logits[0], 0.0, 1e-6);
        assert_close(logits[1], 0.0, 1e-6);
    }

    #[test]
    fn softmax_empty() {
        let mut logits: Vec<f32> = vec![];
        let lse = cpu_softmax(&mut logits);
        assert_eq!(lse, 0.0);
    }

    // ── ProcessContext ─────────────────────────────────────────────────

    #[test]
    fn context_record_updates_frequency() {
        let mut ctx = ProcessContext::new();
        ctx.record_token(5);
        ctx.record_token(5);
        ctx.record_token(3);
        assert_eq!(ctx.token_frequencies[&5], 2);
        assert_eq!(ctx.token_frequencies[&3], 1);
        assert_eq!(ctx.generated_tokens, vec![5, 5, 3]);
    }

    #[test]
    fn context_recent_tokens_window() {
        let ctx = context_with_tokens(&[10, 20, 30, 40, 50]);
        assert_eq!(ctx.recent_tokens(3), &[30, 40, 50]);
        assert_eq!(ctx.recent_tokens(10), &[10, 20, 30, 40, 50]);
        assert_eq!(ctx.recent_tokens(0), &[] as &[u32]);
    }

    // ── Edge cases ────────────────────────────────────────────────────

    #[test]
    fn temperature_on_single_element() {
        let mut logits = vec![3.0];
        TemperatureProcessor::new(0.5).process(&mut logits, &ProcessContext::new());
        assert_close(logits[0], 6.0, 1e-6);
    }

    #[test]
    fn argmax_single_element() {
        let mut logits = vec![3.0];
        TemperatureProcessor::new(0.0).process(&mut logits, &ProcessContext::new());
        assert_eq!(logits[0], 3.0);
    }

    #[test]
    fn all_banned_then_softmax() {
        let mut logits = vec![1.0, 2.0];
        let mut ctx = ProcessContext::new();
        ctx.banned_tokens.extend([0, 1]);
        TokenBanProcessor.process(&mut logits, &ctx);
        cpu_softmax(&mut logits);
        // All -inf → all NaN or 0 after softmax; sum is NaN or 0.
        // We accept either: the important thing is we don't panic.
        assert!(logits.iter().all(|l| l.is_nan() || *l == 0.0), "unexpected values: {logits:?}");
    }

    #[test]
    fn probability_mass_after_full_chain() {
        let mut chain = LogitsProcessorChain::new();
        chain.push(Box::new(TemperatureProcessor::new(0.8)));
        chain.push(Box::new(RepetitionPenaltyProcessor::new(1.2, 5)));
        chain.push(Box::new(FrequencyPenaltyProcessor::new(0.5)));
        chain.push(Box::new(PresencePenaltyProcessor::new(0.3)));
        chain.push(Box::new(TokenBanProcessor));

        let mut ctx = context_with_tokens(&[0, 1, 1, 2]);
        ctx.banned_tokens.insert(4);

        let mut logits = vec![2.0, 3.0, 1.0, 5.0, 4.0, 0.5];
        chain.process(&mut logits, &ctx);
        assert_close(prob_mass(&logits), 1.0, 1e-4);
    }

    // ── Property-style tests ──────────────────────────────────────────

    #[test]
    fn temperature_monotonicity_preserved() {
        // Temperature should preserve the relative ordering of logits.
        for &temp in &[0.1, 0.5, 1.0, 2.0, 10.0] {
            let mut logits = sample_logits();
            TemperatureProcessor::new(temp).process(&mut logits, &ProcessContext::new());
            for i in 1..logits.len() {
                assert!(logits[i] >= logits[i - 1], "monotonicity broken at T={temp}");
            }
        }
    }

    #[test]
    fn repetition_penalty_reduces_repeated_token_probability() {
        let mut logits = uniform_logits(10);
        let ctx = context_with_tokens(&[3]);
        RepetitionPenaltyProcessor::new(2.0, 0).process(&mut logits, &ctx);
        // Token 3 should now have a lower logit.
        assert!(logits[3] < logits[0]);
    }

    #[test]
    fn frequency_penalty_proportional_to_count() {
        let mut logits = vec![10.0, 10.0, 10.0];
        // Token 0 appears 3x, token 1 appears 1x, token 2 appears 0x.
        let ctx = context_with_tokens(&[0, 0, 0, 1]);
        FrequencyPenaltyProcessor::new(1.0).process(&mut logits, &ctx);
        assert!(logits[0] < logits[1]);
        assert!(logits[1] < logits[2]);
    }

    #[test]
    fn chain_composition_is_associative() {
        // (A then B) then C == A then (B then C) when applied
        // sequentially. This is trivially true because we process in
        // order, but verify the output matches.
        let ctx = context_with_tokens(&[0]);
        let temp = TemperatureProcessor::new(0.5);
        let pres = PresencePenaltyProcessor::new(1.0);
        let freq = FrequencyPenaltyProcessor::new(0.5);

        let mut logits_seq = vec![4.0, 2.0, 6.0];
        temp.process(&mut logits_seq, &ctx);
        pres.process(&mut logits_seq, &ctx);
        freq.process(&mut logits_seq, &ctx);

        let mut chain = LogitsProcessorChain::new();
        chain.push(Box::new(TemperatureProcessor::new(0.5)));
        chain.push(Box::new(PresencePenaltyProcessor::new(1.0)));
        chain.push(Box::new(FrequencyPenaltyProcessor::new(0.5)));
        let mut logits_chain = vec![4.0, 2.0, 6.0];
        chain.process(&mut logits_chain, &ctx);

        assert_eq!(logits_seq, logits_chain);
    }

    #[test]
    fn processor_name_strings() {
        assert_eq!(TemperatureProcessor::new(1.0).name(), "TemperatureProcessor");
        assert_eq!(RepetitionPenaltyProcessor::new(1.0, 0).name(), "RepetitionPenaltyProcessor");
        assert_eq!(FrequencyPenaltyProcessor::new(0.0).name(), "FrequencyPenaltyProcessor");
        assert_eq!(PresencePenaltyProcessor::new(0.0).name(), "PresencePenaltyProcessor");
        assert_eq!(TokenBanProcessor.name(), "TokenBanProcessor");
    }
}
