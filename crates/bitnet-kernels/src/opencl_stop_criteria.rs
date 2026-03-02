//! Token generation stopping criteria for Intel Arc A770 inference.
//!
//! # Overview
//!
//! Determines when to stop generating tokens based on various conditions:
//!
//! - **EOS detection** — stops when an end-of-sequence token is generated.
//! - **Max length** — caps the total number of generated tokens.
//! - **Stop strings** — halts when a decoded substring matches a stop pattern.
//! - **Repetition** — detects degenerate n-gram loops.
//! - **Quality threshold** — entropy-based quality gate for logits distributions.
//! - **Timeout** — wall-clock time limit for generation.
//! - **User cancel** — external cancellation signal.
//!
//! # Architecture
//!
//! All criteria implement the [`StopCriteria`] trait. [`CompositeCriteria`]
//! combines multiple criteria with configurable AND/OR logic.
//! CPU reference implementations are provided; GPU-side early-exit is planned
//! for v0.3.

use std::collections::HashMap;
use std::fmt;

// ── Stop reasons ─────────────────────────────────────────────────

/// Why token generation was stopped.
#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    /// An end-of-sequence token was generated.
    EosToken,
    /// The maximum token count was reached.
    MaxLength,
    /// A stop string was detected in the decoded output.
    StopString(String),
    /// Excessive n-gram repetition was detected.
    RepetitionLimit,
    /// Logits entropy fell below the quality threshold.
    QualityThreshold,
    /// Generation was cancelled by the user.
    UserCancel,
    /// Wall-clock timeout expired.
    Timeout,
}

impl fmt::Display for StopReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EosToken => write!(f, "eos_token"),
            Self::MaxLength => write!(f, "max_length"),
            Self::StopString(s) => write!(f, "stop_string({s})"),
            Self::RepetitionLimit => write!(f, "repetition_limit"),
            Self::QualityThreshold => write!(f, "quality_threshold"),
            Self::UserCancel => write!(f, "user_cancel"),
            Self::Timeout => write!(f, "timeout"),
        }
    }
}

// ── Stop context ─────────────────────────────────────────────────

/// Runtime context passed to each stopping criterion.
#[derive(Debug, Clone)]
pub struct StopContext {
    /// Number of tokens generated so far.
    pub generated_count: usize,
    /// Elapsed wall-clock time in milliseconds since generation started.
    pub elapsed_ms: u64,
    /// Tail of the decoded text (for stop-string matching).
    pub decoded_text_tail: String,
    /// Shannon entropy of the most recent logits distribution (nats).
    /// `None` if not yet computed.
    pub logits_entropy: Option<f32>,
}

impl StopContext {
    /// Create a new context with the given generated count.
    pub fn new(generated_count: usize) -> Self {
        Self {
            generated_count,
            elapsed_ms: 0,
            decoded_text_tail: String::new(),
            logits_entropy: None,
        }
    }

    /// Builder: set elapsed time.
    pub fn with_elapsed_ms(mut self, ms: u64) -> Self {
        self.elapsed_ms = ms;
        self
    }

    /// Builder: set decoded text tail.
    pub fn with_decoded_text_tail(mut self, tail: String) -> Self {
        self.decoded_text_tail = tail;
        self
    }

    /// Builder: set logits entropy.
    pub fn with_logits_entropy(mut self, entropy: f32) -> Self {
        self.logits_entropy = Some(entropy);
        self
    }
}

impl Default for StopContext {
    fn default() -> Self {
        Self::new(0)
    }
}

// ── Trait ─────────────────────────────────────────────────────────

/// Trait for token generation stopping criteria.
pub trait StopCriteria: Send + Sync {
    /// Returns `Some(reason)` if generation should stop, `None` otherwise.
    fn should_stop(&self, tokens: &[u32], context: &StopContext) -> Option<StopReason>;

    /// Human-readable name of this criterion.
    fn name(&self) -> &str;
}

// ── EOS criteria ─────────────────────────────────────────────────

/// Stops when the last generated token is an end-of-sequence token.
#[derive(Debug, Clone)]
pub struct EosCriteria {
    /// Set of token IDs that signal end-of-sequence.
    pub eos_token_ids: Vec<u32>,
}

impl EosCriteria {
    pub fn new(eos_token_ids: Vec<u32>) -> Self {
        Self { eos_token_ids }
    }

    pub fn single(eos_id: u32) -> Self {
        Self { eos_token_ids: vec![eos_id] }
    }
}

impl StopCriteria for EosCriteria {
    fn should_stop(&self, tokens: &[u32], _context: &StopContext) -> Option<StopReason> {
        tokens.last().filter(|t| self.eos_token_ids.contains(t)).map(|_| StopReason::EosToken)
    }

    fn name(&self) -> &str {
        "eos"
    }
}

// ── Max length criteria ──────────────────────────────────────────

/// Stops when the generated token count reaches `max_tokens`.
#[derive(Debug, Clone)]
pub struct MaxLengthCriteria {
    pub max_tokens: usize,
}

impl MaxLengthCriteria {
    pub fn new(max_tokens: usize) -> Self {
        Self { max_tokens }
    }
}

impl StopCriteria for MaxLengthCriteria {
    fn should_stop(&self, _tokens: &[u32], context: &StopContext) -> Option<StopReason> {
        if context.generated_count >= self.max_tokens { Some(StopReason::MaxLength) } else { None }
    }

    fn name(&self) -> &str {
        "max_length"
    }
}

// ── Stop string criteria ─────────────────────────────────────────

/// Stops when any of the configured strings appear in the decoded output.
#[derive(Debug, Clone)]
pub struct StopStringCriteria {
    /// Strings that trigger generation stop.
    pub stop_strings: Vec<String>,
}

impl StopStringCriteria {
    pub fn new(stop_strings: Vec<String>) -> Self {
        Self { stop_strings }
    }

    pub fn single(s: impl Into<String>) -> Self {
        Self { stop_strings: vec![s.into()] }
    }
}

impl StopCriteria for StopStringCriteria {
    fn should_stop(&self, _tokens: &[u32], context: &StopContext) -> Option<StopReason> {
        for s in &self.stop_strings {
            if !s.is_empty() && context.decoded_text_tail.contains(s.as_str()) {
                return Some(StopReason::StopString(s.clone()));
            }
        }
        None
    }

    fn name(&self) -> &str {
        "stop_string"
    }
}

// ── Repetition criteria ──────────────────────────────────────────

/// Detects degenerate n-gram repetition in the token stream.
///
/// Scans the last `window_size` tokens for any n-gram (of size `ngram_size`)
/// that appears more than `max_repetitions` times.
#[derive(Debug, Clone)]
pub struct RepetitionCriteria {
    /// Number of trailing tokens to inspect.
    pub window_size: usize,
    /// Maximum allowed occurrences of any single n-gram.
    pub max_repetitions: usize,
    /// Size of the n-gram to track.
    pub ngram_size: usize,
}

impl RepetitionCriteria {
    pub fn new(window_size: usize, max_repetitions: usize, ngram_size: usize) -> Self {
        Self { window_size, max_repetitions, ngram_size }
    }

    /// Count n-gram occurrences in the given token window.
    fn count_ngrams(tokens: &[u32], ngram_size: usize) -> HashMap<Vec<u32>, usize> {
        let mut counts: HashMap<Vec<u32>, usize> = HashMap::new();
        if tokens.len() < ngram_size || ngram_size == 0 {
            return counts;
        }
        for window in tokens.windows(ngram_size) {
            *counts.entry(window.to_vec()).or_insert(0) += 1;
        }
        counts
    }
}

impl StopCriteria for RepetitionCriteria {
    fn should_stop(&self, tokens: &[u32], _context: &StopContext) -> Option<StopReason> {
        if self.ngram_size == 0 || self.window_size == 0 {
            return None;
        }
        let start = tokens.len().saturating_sub(self.window_size);
        let window = &tokens[start..];
        let counts = Self::count_ngrams(window, self.ngram_size);
        if counts.values().any(|&c| c > self.max_repetitions) {
            Some(StopReason::RepetitionLimit)
        } else {
            None
        }
    }

    fn name(&self) -> &str {
        "repetition"
    }
}

// ── Quality threshold criteria ───────────────────────────────────

/// Stops when the logits entropy drops below a configured threshold,
/// indicating the model is producing low-quality / degenerate output.
#[derive(Debug)]
pub struct QualityThresholdCriteria {
    /// Minimum acceptable Shannon entropy (nats).
    pub min_entropy: f32,
    /// Number of consecutive low-entropy steps before triggering.
    pub consecutive_required: usize,
    /// Internal counter is not stored here; use with [`CompositeCriteria`]
    /// or wrap in a stateful adapter.
    low_entropy_count: std::sync::atomic::AtomicUsize,
}

impl QualityThresholdCriteria {
    pub fn new(min_entropy: f32) -> Self {
        Self {
            min_entropy,
            consecutive_required: 1,
            low_entropy_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn with_consecutive(mut self, n: usize) -> Self {
        self.consecutive_required = n;
        self
    }
}

impl StopCriteria for QualityThresholdCriteria {
    fn should_stop(&self, _tokens: &[u32], context: &StopContext) -> Option<StopReason> {
        if let Some(entropy) = context.logits_entropy {
            if entropy < self.min_entropy {
                let count =
                    self.low_entropy_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if count >= self.consecutive_required {
                    return Some(StopReason::QualityThreshold);
                }
            } else {
                self.low_entropy_count.store(0, std::sync::atomic::Ordering::Relaxed);
            }
        }
        None
    }

    fn name(&self) -> &str {
        "quality_threshold"
    }
}

// ── Timeout criteria ─────────────────────────────────────────────

/// Stops when elapsed generation time exceeds the configured timeout.
#[derive(Debug, Clone)]
pub struct TimeoutCriteria {
    /// Maximum allowed generation time in milliseconds.
    pub timeout_ms: u64,
}

impl TimeoutCriteria {
    pub fn new(timeout_ms: u64) -> Self {
        Self { timeout_ms }
    }
}

impl StopCriteria for TimeoutCriteria {
    fn should_stop(&self, _tokens: &[u32], context: &StopContext) -> Option<StopReason> {
        if context.elapsed_ms >= self.timeout_ms { Some(StopReason::Timeout) } else { None }
    }

    fn name(&self) -> &str {
        "timeout"
    }
}

// ── Composite criteria ───────────────────────────────────────────

/// How multiple criteria are combined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeMode {
    /// Stop if **any** criterion fires (logical OR, default).
    Any,
    /// Stop only if **all** criteria fire simultaneously (logical AND).
    All,
}

/// Combines multiple stopping criteria with configurable logic.
pub struct CompositeCriteria {
    criteria: Vec<Box<dyn StopCriteria>>,
    mode: CompositeMode,
}

impl CompositeCriteria {
    pub fn new(mode: CompositeMode) -> Self {
        Self { criteria: Vec::new(), mode }
    }

    /// Shorthand for OR-mode composite.
    pub fn any() -> Self {
        Self::new(CompositeMode::Any)
    }

    /// Shorthand for AND-mode composite.
    pub fn all() -> Self {
        Self::new(CompositeMode::All)
    }

    pub fn with(mut self, criterion: impl StopCriteria + 'static) -> Self {
        self.criteria.push(Box::new(criterion));
        self
    }

    /// Number of contained criteria.
    pub fn len(&self) -> usize {
        self.criteria.len()
    }

    /// Whether this composite has no criteria.
    pub fn is_empty(&self) -> bool {
        self.criteria.is_empty()
    }
}

impl StopCriteria for CompositeCriteria {
    fn should_stop(&self, tokens: &[u32], context: &StopContext) -> Option<StopReason> {
        if self.criteria.is_empty() {
            return None;
        }

        match self.mode {
            CompositeMode::Any => {
                // Return the first reason that fires.
                for c in &self.criteria {
                    if let Some(reason) = c.should_stop(tokens, context) {
                        return Some(reason);
                    }
                }
                None
            }
            CompositeMode::All => {
                // Collect reasons; only stop if *every* criterion fires.
                let mut reasons: Vec<StopReason> = Vec::with_capacity(self.criteria.len());
                for c in &self.criteria {
                    match c.should_stop(tokens, context) {
                        Some(r) => reasons.push(r),
                        None => return None,
                    }
                }
                // Return the first reason as representative.
                reasons.into_iter().next()
            }
        }
    }

    fn name(&self) -> &str {
        "composite"
    }
}

impl fmt::Debug for CompositeCriteria {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompositeCriteria")
            .field("mode", &self.mode)
            .field("num_criteria", &self.criteria.len())
            .finish()
    }
}

// ── Utility: compute Shannon entropy ─────────────────────────────

/// Compute Shannon entropy (in nats) from a logits slice.
///
/// Applies softmax internally before computing `-Σ p·ln(p)`.
pub fn compute_entropy(logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return 0.0;
    }
    let mut entropy = 0.0f32;
    for &e in &exps {
        let p = e / sum;
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }
    entropy
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers --------------------------------------------------------

    fn ctx(generated: usize) -> StopContext {
        StopContext::new(generated)
    }

    fn ctx_with_text(generated: usize, text: &str) -> StopContext {
        StopContext::new(generated).with_decoded_text_tail(text.to_string())
    }

    fn ctx_with_time(generated: usize, ms: u64) -> StopContext {
        StopContext::new(generated).with_elapsed_ms(ms)
    }

    fn ctx_with_entropy(generated: usize, entropy: f32) -> StopContext {
        StopContext::new(generated).with_logits_entropy(entropy)
    }

    // =================================================================
    // StopReason
    // =================================================================

    #[test]
    fn stop_reason_display() {
        assert_eq!(StopReason::EosToken.to_string(), "eos_token");
        assert_eq!(StopReason::MaxLength.to_string(), "max_length");
        assert_eq!(StopReason::StopString("</s>".into()).to_string(), "stop_string(</s>)");
        assert_eq!(StopReason::RepetitionLimit.to_string(), "repetition_limit");
        assert_eq!(StopReason::QualityThreshold.to_string(), "quality_threshold");
        assert_eq!(StopReason::UserCancel.to_string(), "user_cancel");
        assert_eq!(StopReason::Timeout.to_string(), "timeout");
    }

    #[test]
    fn stop_reason_equality() {
        assert_eq!(StopReason::EosToken, StopReason::EosToken);
        assert_ne!(StopReason::EosToken, StopReason::MaxLength);
        assert_eq!(StopReason::StopString("a".into()), StopReason::StopString("a".into()));
        assert_ne!(StopReason::StopString("a".into()), StopReason::StopString("b".into()));
    }

    // =================================================================
    // StopContext
    // =================================================================

    #[test]
    fn context_default() {
        let c = StopContext::default();
        assert_eq!(c.generated_count, 0);
        assert_eq!(c.elapsed_ms, 0);
        assert!(c.decoded_text_tail.is_empty());
        assert!(c.logits_entropy.is_none());
    }

    #[test]
    fn context_builders() {
        let c = StopContext::new(10)
            .with_elapsed_ms(500)
            .with_decoded_text_tail("hello".into())
            .with_logits_entropy(2.5);
        assert_eq!(c.generated_count, 10);
        assert_eq!(c.elapsed_ms, 500);
        assert_eq!(c.decoded_text_tail, "hello");
        assert_eq!(c.logits_entropy, Some(2.5));
    }

    // =================================================================
    // EosCriteria
    // =================================================================

    #[test]
    fn eos_single_token_match() {
        let crit = EosCriteria::single(2);
        assert_eq!(crit.should_stop(&[10, 20, 2], &ctx(3)), Some(StopReason::EosToken));
    }

    #[test]
    fn eos_single_token_no_match() {
        let crit = EosCriteria::single(2);
        assert_eq!(crit.should_stop(&[10, 20, 30], &ctx(3)), None);
    }

    #[test]
    fn eos_multiple_token_ids() {
        let crit = EosCriteria::new(vec![2, 3, 50256]);
        assert_eq!(crit.should_stop(&[100, 50256], &ctx(2)), Some(StopReason::EosToken));
        assert_eq!(crit.should_stop(&[100, 3], &ctx(2)), Some(StopReason::EosToken));
    }

    #[test]
    fn eos_empty_tokens() {
        let crit = EosCriteria::single(2);
        assert_eq!(crit.should_stop(&[], &ctx(0)), None);
    }

    #[test]
    fn eos_only_last_token_matters() {
        let crit = EosCriteria::single(2);
        // EOS in the middle but not at end — should not stop.
        assert_eq!(crit.should_stop(&[2, 10, 20], &ctx(3)), None);
    }

    #[test]
    fn eos_name() {
        assert_eq!(EosCriteria::single(0).name(), "eos");
    }

    // =================================================================
    // MaxLengthCriteria
    // =================================================================

    #[test]
    fn max_length_under_limit() {
        let crit = MaxLengthCriteria::new(10);
        assert_eq!(crit.should_stop(&[1, 2, 3], &ctx(3)), None);
    }

    #[test]
    fn max_length_at_limit() {
        let crit = MaxLengthCriteria::new(5);
        assert_eq!(crit.should_stop(&[1, 2, 3, 4, 5], &ctx(5)), Some(StopReason::MaxLength));
    }

    #[test]
    fn max_length_over_limit() {
        let crit = MaxLengthCriteria::new(3);
        assert_eq!(crit.should_stop(&[1, 2, 3, 4], &ctx(4)), Some(StopReason::MaxLength));
    }

    #[test]
    fn max_length_zero() {
        let crit = MaxLengthCriteria::new(0);
        assert_eq!(crit.should_stop(&[], &ctx(0)), Some(StopReason::MaxLength));
    }

    #[test]
    fn max_length_name() {
        assert_eq!(MaxLengthCriteria::new(1).name(), "max_length");
    }

    // =================================================================
    // StopStringCriteria
    // =================================================================

    #[test]
    fn stop_string_exact_match() {
        let crit = StopStringCriteria::single("</s>");
        let c = ctx_with_text(5, "Hello world</s>");
        assert_eq!(crit.should_stop(&[1, 2, 3], &c), Some(StopReason::StopString("</s>".into())));
    }

    #[test]
    fn stop_string_no_match() {
        let crit = StopStringCriteria::single("</s>");
        let c = ctx_with_text(5, "Hello world");
        assert_eq!(crit.should_stop(&[1, 2, 3], &c), None);
    }

    #[test]
    fn stop_string_substring_match() {
        let crit = StopStringCriteria::single("world");
        let c = ctx_with_text(3, "Hello world!");
        assert_eq!(crit.should_stop(&[], &c), Some(StopReason::StopString("world".into())));
    }

    #[test]
    fn stop_string_multi_string() {
        let crit = StopStringCriteria::new(vec!["STOP".into(), "END".into(), "QUIT".into()]);
        let c = ctx_with_text(10, "output text END more");
        let reason = crit.should_stop(&[], &c);
        assert_eq!(reason, Some(StopReason::StopString("END".into())));
    }

    #[test]
    fn stop_string_empty_string_ignored() {
        let crit = StopStringCriteria::new(vec!["".into(), "ok".into()]);
        let c = ctx_with_text(1, "anything");
        // Empty stop string should not match.
        assert_eq!(crit.should_stop(&[], &c), None);
    }

    #[test]
    fn stop_string_empty_text() {
        let crit = StopStringCriteria::single("stop");
        let c = ctx_with_text(0, "");
        assert_eq!(crit.should_stop(&[], &c), None);
    }

    #[test]
    fn stop_string_at_boundary() {
        let crit = StopStringCriteria::single("end");
        let c = ctx_with_text(3, "end");
        assert_eq!(crit.should_stop(&[], &c), Some(StopReason::StopString("end".into())));
    }

    #[test]
    fn stop_string_name() {
        assert_eq!(StopStringCriteria::single("x").name(), "stop_string");
    }

    // =================================================================
    // RepetitionCriteria
    // =================================================================

    #[test]
    fn repetition_no_repeat() {
        let crit = RepetitionCriteria::new(10, 2, 2);
        // All unique bigrams.
        assert_eq!(crit.should_stop(&[1, 2, 3, 4, 5], &ctx(5)), None);
    }

    #[test]
    fn repetition_bigram_detected() {
        let crit = RepetitionCriteria::new(20, 2, 2);
        // Bigram [1,2] appears 3 times (> max_repetitions=2).
        let tokens = vec![1, 2, 1, 2, 1, 2, 3];
        assert_eq!(crit.should_stop(&tokens, &ctx(7)), Some(StopReason::RepetitionLimit));
    }

    #[test]
    fn repetition_trigram_detected() {
        let crit = RepetitionCriteria::new(30, 1, 3);
        // Trigram [5,6,7] appears 2 times (> max_repetitions=1).
        let tokens = vec![5, 6, 7, 5, 6, 7];
        assert_eq!(crit.should_stop(&tokens, &ctx(6)), Some(StopReason::RepetitionLimit));
    }

    #[test]
    fn repetition_within_window() {
        let crit = RepetitionCriteria::new(4, 1, 2);
        // Window is last 4 tokens: [1,2,1,2] → bigram [1,2] appears 2×.
        let tokens = vec![99, 98, 97, 1, 2, 1, 2];
        assert_eq!(crit.should_stop(&tokens, &ctx(7)), Some(StopReason::RepetitionLimit));
    }

    #[test]
    fn repetition_outside_window() {
        let crit = RepetitionCriteria::new(3, 1, 2);
        // Window is last 3 tokens: [2,1,2] → bigram [2,1] 1×, [1,2] 1×.
        let tokens = vec![1, 2, 1, 2];
        assert_eq!(crit.should_stop(&tokens, &ctx(4)), None);
    }

    #[test]
    fn repetition_empty_tokens() {
        let crit = RepetitionCriteria::new(10, 1, 2);
        assert_eq!(crit.should_stop(&[], &ctx(0)), None);
    }

    #[test]
    fn repetition_single_token() {
        let crit = RepetitionCriteria::new(10, 1, 2);
        assert_eq!(crit.should_stop(&[42], &ctx(1)), None);
    }

    #[test]
    fn repetition_zero_ngram_size() {
        let crit = RepetitionCriteria::new(10, 1, 0);
        assert_eq!(crit.should_stop(&[1, 2, 3], &ctx(3)), None);
    }

    #[test]
    fn repetition_zero_window() {
        let crit = RepetitionCriteria::new(0, 1, 2);
        assert_eq!(crit.should_stop(&[1, 2, 1, 2], &ctx(4)), None);
    }

    #[test]
    fn repetition_unigram() {
        let crit = RepetitionCriteria::new(10, 2, 1);
        // Token 7 appears 4 times (> max_repetitions=2).
        let tokens = vec![7, 7, 7, 7];
        assert_eq!(crit.should_stop(&tokens, &ctx(4)), Some(StopReason::RepetitionLimit));
    }

    #[test]
    fn repetition_name() {
        assert_eq!(RepetitionCriteria::new(1, 1, 1).name(), "repetition");
    }

    // =================================================================
    // QualityThresholdCriteria
    // =================================================================

    #[test]
    fn quality_below_threshold() {
        let crit = QualityThresholdCriteria::new(1.0);
        let c = ctx_with_entropy(5, 0.5);
        assert_eq!(crit.should_stop(&[1], &c), Some(StopReason::QualityThreshold));
    }

    #[test]
    fn quality_above_threshold() {
        let crit = QualityThresholdCriteria::new(1.0);
        let c = ctx_with_entropy(5, 2.0);
        assert_eq!(crit.should_stop(&[1], &c), None);
    }

    #[test]
    fn quality_no_entropy() {
        let crit = QualityThresholdCriteria::new(1.0);
        let c = ctx(5); // No entropy set.
        assert_eq!(crit.should_stop(&[1], &c), None);
    }

    #[test]
    fn quality_consecutive_required() {
        let crit = QualityThresholdCriteria::new(1.0).with_consecutive(3);
        let low = ctx_with_entropy(1, 0.1);
        // First two low-entropy calls should not trigger.
        assert_eq!(crit.should_stop(&[1], &low), None);
        assert_eq!(crit.should_stop(&[1, 2], &low), None);
        // Third consecutive triggers.
        assert_eq!(crit.should_stop(&[1, 2, 3], &low), Some(StopReason::QualityThreshold));
    }

    #[test]
    fn quality_consecutive_reset_on_high() {
        let crit = QualityThresholdCriteria::new(1.0).with_consecutive(2);
        let low = ctx_with_entropy(1, 0.1);
        let high = ctx_with_entropy(2, 5.0);
        assert_eq!(crit.should_stop(&[1], &low), None);
        // High entropy resets counter.
        assert_eq!(crit.should_stop(&[1], &high), None);
        // Need 2 consecutive again.
        assert_eq!(crit.should_stop(&[1], &low), None);
        assert_eq!(crit.should_stop(&[1], &low), Some(StopReason::QualityThreshold));
    }

    #[test]
    fn quality_name() {
        assert_eq!(QualityThresholdCriteria::new(1.0).name(), "quality_threshold");
    }

    // =================================================================
    // TimeoutCriteria
    // =================================================================

    #[test]
    fn timeout_not_expired() {
        let crit = TimeoutCriteria::new(5000);
        let c = ctx_with_time(10, 3000);
        assert_eq!(crit.should_stop(&[1], &c), None);
    }

    #[test]
    fn timeout_exactly_at_limit() {
        let crit = TimeoutCriteria::new(5000);
        let c = ctx_with_time(10, 5000);
        assert_eq!(crit.should_stop(&[1], &c), Some(StopReason::Timeout));
    }

    #[test]
    fn timeout_expired() {
        let crit = TimeoutCriteria::new(1000);
        let c = ctx_with_time(5, 2000);
        assert_eq!(crit.should_stop(&[1], &c), Some(StopReason::Timeout));
    }

    #[test]
    fn timeout_name() {
        assert_eq!(TimeoutCriteria::new(0).name(), "timeout");
    }

    // =================================================================
    // CompositeCriteria — OR mode
    // =================================================================

    #[test]
    fn composite_or_first_fires() {
        let comp =
            CompositeCriteria::any().with(EosCriteria::single(2)).with(MaxLengthCriteria::new(100));
        assert_eq!(comp.should_stop(&[1, 2], &ctx(2)), Some(StopReason::EosToken));
    }

    #[test]
    fn composite_or_second_fires() {
        let comp =
            CompositeCriteria::any().with(EosCriteria::single(2)).with(MaxLengthCriteria::new(3));
        assert_eq!(comp.should_stop(&[1, 3, 4], &ctx(3)), Some(StopReason::MaxLength));
    }

    #[test]
    fn composite_or_none_fires() {
        let comp =
            CompositeCriteria::any().with(EosCriteria::single(2)).with(MaxLengthCriteria::new(100));
        assert_eq!(comp.should_stop(&[1, 3], &ctx(2)), None);
    }

    #[test]
    fn composite_or_empty() {
        let comp = CompositeCriteria::any();
        assert_eq!(comp.should_stop(&[1], &ctx(1)), None);
    }

    // =================================================================
    // CompositeCriteria — AND mode
    // =================================================================

    #[test]
    fn composite_and_all_fire() {
        let comp =
            CompositeCriteria::all().with(EosCriteria::single(2)).with(MaxLengthCriteria::new(3));
        // EOS fires (last=2) AND max_length fires (count=3 >= 3).
        assert_eq!(comp.should_stop(&[1, 3, 2], &ctx(3)), Some(StopReason::EosToken));
    }

    #[test]
    fn composite_and_partial_fire() {
        let comp =
            CompositeCriteria::all().with(EosCriteria::single(2)).with(MaxLengthCriteria::new(10));
        // EOS fires but max_length doesn't (2 < 10).
        assert_eq!(comp.should_stop(&[1, 2], &ctx(2)), None);
    }

    #[test]
    fn composite_and_empty() {
        let comp = CompositeCriteria::all();
        assert_eq!(comp.should_stop(&[1], &ctx(1)), None);
    }

    // =================================================================
    // CompositeCriteria — structural
    // =================================================================

    #[test]
    fn composite_len_and_empty() {
        let comp = CompositeCriteria::any();
        assert!(comp.is_empty());
        assert_eq!(comp.len(), 0);
        let comp = comp.with(EosCriteria::single(0));
        assert!(!comp.is_empty());
        assert_eq!(comp.len(), 1);
    }

    #[test]
    fn composite_debug_format() {
        let comp =
            CompositeCriteria::any().with(EosCriteria::single(2)).with(MaxLengthCriteria::new(10));
        let dbg = format!("{comp:?}");
        assert!(dbg.contains("CompositeCriteria"));
        assert!(dbg.contains("Any"));
        assert!(dbg.contains("2"));
    }

    #[test]
    fn composite_name() {
        assert_eq!(CompositeCriteria::any().name(), "composite");
    }

    // =================================================================
    // compute_entropy utility
    // =================================================================

    #[test]
    fn entropy_empty_logits() {
        assert_eq!(compute_entropy(&[]), 0.0);
    }

    #[test]
    fn entropy_single_logit() {
        // Single class → probability 1.0 → entropy 0.
        let e = compute_entropy(&[5.0]);
        assert!((e - 0.0).abs() < 1e-6, "got {e}");
    }

    #[test]
    fn entropy_uniform_distribution() {
        // Uniform over 4 classes → entropy = ln(4) ≈ 1.386.
        let logits = vec![0.0; 4];
        let e = compute_entropy(&logits);
        let expected = (4.0f32).ln();
        assert!((e - expected).abs() < 1e-5, "expected ~{expected}, got {e}");
    }

    #[test]
    fn entropy_peaked_distribution() {
        // Very peaked: one large logit dominates.
        let logits = vec![100.0, 0.0, 0.0, 0.0];
        let e = compute_entropy(&logits);
        assert!(e < 0.01, "expected near-zero entropy, got {e}");
    }

    #[test]
    fn entropy_increases_with_uniformity() {
        let peaked = compute_entropy(&[10.0, 0.0]);
        let uniform = compute_entropy(&[0.0, 0.0]);
        assert!(uniform > peaked, "uniform ({uniform}) should be > peaked ({peaked})");
    }

    // =================================================================
    // Property-like tests
    // =================================================================

    #[test]
    fn property_composite_or_superset_of_individuals() {
        let eos = EosCriteria::single(2);
        let max_len = MaxLengthCriteria::new(5);

        let tokens = &[1, 2];
        let c = ctx(2);

        let eos_result = eos.should_stop(tokens, &c);
        let max_result = max_len.should_stop(tokens, &c);

        let comp =
            CompositeCriteria::any().with(EosCriteria::single(2)).with(MaxLengthCriteria::new(5));
        let comp_result = comp.should_stop(tokens, &c);

        // If either individual fires, composite must also fire.
        if eos_result.is_some() || max_result.is_some() {
            assert!(comp_result.is_some());
        }
    }

    #[test]
    fn property_composite_and_subset_of_individuals() {
        let tokens = &[1, 2];
        let c = ctx(2);

        let comp =
            CompositeCriteria::all().with(EosCriteria::single(2)).with(MaxLengthCriteria::new(5));
        let comp_result = comp.should_stop(tokens, &c);

        // AND composite can only fire if ALL individuals fire.
        // max_length(5) doesn't fire at count=2, so composite must not.
        assert!(comp_result.is_none());
    }

    #[test]
    fn property_eos_never_fires_without_eos_token() {
        let crit = EosCriteria::new(vec![2, 3, 4]);
        // Generate sequences without any EOS token.
        for last in [0_u32, 1, 5, 100, 999] {
            let tokens = vec![10, 20, last];
            assert!(
                crit.should_stop(&tokens, &ctx(3)).is_none(),
                "should not fire for last={last}"
            );
        }
    }

    #[test]
    fn property_max_length_monotonic() {
        let crit = MaxLengthCriteria::new(5);
        // Once count >= max, it stays triggered.
        for count in 0..10 {
            let result = crit.should_stop(&[], &ctx(count));
            if count >= 5 {
                assert!(result.is_some(), "count={count} should trigger");
            } else {
                assert!(result.is_none(), "count={count} should not trigger");
            }
        }
    }

    #[test]
    fn property_timeout_monotonic() {
        let crit = TimeoutCriteria::new(100);
        for ms in [0, 50, 99, 100, 101, 500] {
            let result = crit.should_stop(&[], &ctx_with_time(1, ms));
            if ms >= 100 {
                assert!(result.is_some(), "ms={ms} should trigger");
            } else {
                assert!(result.is_none(), "ms={ms} should not trigger");
            }
        }
    }
}
