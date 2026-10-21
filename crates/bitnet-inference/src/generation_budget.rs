//! Generation budget tracking.
//!
//! Tracks and enforces token generation budgets including
//! max tokens, time limits, and memory constraints.

use std::time::{Duration, Instant};

/// Budget limits for generation.
#[derive(Debug, Clone)]
pub struct GenerationBudget {
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Maximum wall-clock time.
    pub max_time: Option<Duration>,
    /// Maximum memory (bytes) for KV cache.
    pub max_memory_bytes: Option<u64>,
}

impl Default for GenerationBudget {
    fn default() -> Self {
        Self { max_tokens: 256, max_time: None, max_memory_bytes: None }
    }
}

impl GenerationBudget {
    pub fn new(max_tokens: usize) -> Self {
        Self { max_tokens, ..Default::default() }
    }

    pub fn with_time_limit(mut self, limit: Duration) -> Self {
        self.max_time = Some(limit);
        self
    }

    pub fn with_memory_limit(mut self, bytes: u64) -> Self {
        self.max_memory_bytes = Some(bytes);
        self
    }

    pub fn unlimited() -> Self {
        Self { max_tokens: usize::MAX, max_time: None, max_memory_bytes: None }
    }
}

/// Tracks budget consumption during generation.
#[derive(Debug)]
pub struct BudgetTracker {
    budget: GenerationBudget,
    tokens_generated: usize,
    start_time: Instant,
    current_memory: u64,
    stop_reason: Option<StopReason>,
}

/// Reason generation was stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Reached max token count.
    MaxTokens,
    /// Exceeded time limit.
    TimeLimit,
    /// Exceeded memory limit.
    MemoryLimit,
    /// End-of-sequence token generated.
    EndOfSequence,
    /// User-requested stop.
    UserStop,
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MaxTokens => write!(f, "max_tokens"),
            Self::TimeLimit => write!(f, "time_limit"),
            Self::MemoryLimit => write!(f, "memory_limit"),
            Self::EndOfSequence => write!(f, "end_of_sequence"),
            Self::UserStop => write!(f, "user_stop"),
        }
    }
}

impl BudgetTracker {
    pub fn new(budget: GenerationBudget) -> Self {
        Self {
            budget,
            tokens_generated: 0,
            start_time: Instant::now(),
            current_memory: 0,
            stop_reason: None,
        }
    }

    /// Record a generated token. Returns true if budget allows continuing.
    pub fn record_token(&mut self) -> bool {
        self.tokens_generated += 1;
        if self.tokens_generated >= self.budget.max_tokens {
            self.stop_reason = Some(StopReason::MaxTokens);
            return false;
        }
        if let Some(limit) = self.budget.max_time
            && self.start_time.elapsed() >= limit
        {
            self.stop_reason = Some(StopReason::TimeLimit);
            return false;
        }
        if let Some(limit) = self.budget.max_memory_bytes
            && self.current_memory > limit
        {
            self.stop_reason = Some(StopReason::MemoryLimit);
            return false;
        }
        true
    }

    /// Record end-of-sequence detection.
    pub fn record_eos(&mut self) {
        self.stop_reason = Some(StopReason::EndOfSequence);
    }

    /// Record user-initiated stop.
    pub fn record_user_stop(&mut self) {
        self.stop_reason = Some(StopReason::UserStop);
    }

    /// Update current memory usage estimate.
    pub fn update_memory(&mut self, bytes: u64) {
        self.current_memory = bytes;
    }

    /// Whether generation should continue.
    pub fn can_continue(&self) -> bool {
        self.stop_reason.is_none()
    }

    /// Get the stop reason, if any.
    pub fn stop_reason(&self) -> Option<StopReason> {
        self.stop_reason
    }

    /// Tokens generated so far.
    pub fn tokens_generated(&self) -> usize {
        self.tokens_generated
    }

    /// Remaining token budget.
    pub fn tokens_remaining(&self) -> usize {
        self.budget.max_tokens.saturating_sub(self.tokens_generated)
    }

    /// Elapsed time since generation started.
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Remaining time budget.
    pub fn time_remaining(&self) -> Option<Duration> {
        self.budget.max_time.map(|limit| limit.saturating_sub(self.start_time.elapsed()))
    }

    /// Fraction of token budget consumed (0.0 to 1.0).
    pub fn token_utilization(&self) -> f64 {
        if self.budget.max_tokens == 0 {
            return 1.0;
        }
        self.tokens_generated as f64 / self.budget.max_tokens as f64
    }

    /// Tokens per second so far.
    pub fn tokens_per_second(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed == 0.0 {
            return 0.0;
        }
        self.tokens_generated as f64 / elapsed
    }

    /// Get a snapshot summary.
    pub fn summary(&self) -> BudgetSummary {
        BudgetSummary {
            tokens_generated: self.tokens_generated,
            max_tokens: self.budget.max_tokens,
            elapsed: self.start_time.elapsed(),
            stop_reason: self.stop_reason,
            tokens_per_second: self.tokens_per_second(),
        }
    }
}

/// Summary of generation budget state.
#[derive(Debug, Clone)]
pub struct BudgetSummary {
    pub tokens_generated: usize,
    pub max_tokens: usize,
    pub elapsed: Duration,
    pub stop_reason: Option<StopReason>,
    pub tokens_per_second: f64,
}

impl std::fmt::Display for BudgetSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}/{} tokens in {:.2?} ({:.1} tok/s, stop={:?})",
            self.tokens_generated,
            self.max_tokens,
            self.elapsed,
            self.tokens_per_second,
            self.stop_reason,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_budget() {
        let b = GenerationBudget::default();
        assert_eq!(b.max_tokens, 256);
        assert!(b.max_time.is_none());
    }

    #[test]
    fn test_budget_new() {
        let b = GenerationBudget::new(100);
        assert_eq!(b.max_tokens, 100);
    }

    #[test]
    fn test_budget_unlimited() {
        let b = GenerationBudget::unlimited();
        assert_eq!(b.max_tokens, usize::MAX);
    }

    #[test]
    fn test_budget_builder() {
        let b = GenerationBudget::new(50)
            .with_time_limit(Duration::from_secs(30))
            .with_memory_limit(1_000_000);
        assert_eq!(b.max_tokens, 50);
        assert_eq!(b.max_time, Some(Duration::from_secs(30)));
        assert_eq!(b.max_memory_bytes, Some(1_000_000));
    }

    #[test]
    fn test_tracker_basic() {
        let budget = GenerationBudget::new(3);
        let tracker = BudgetTracker::new(budget);
        assert!(tracker.can_continue());
        assert_eq!(tracker.tokens_generated(), 0);
    }

    #[test]
    fn test_tracker_record_tokens() {
        let budget = GenerationBudget::new(3);
        let mut tracker = BudgetTracker::new(budget);
        assert!(tracker.record_token()); // 1
        assert!(tracker.record_token()); // 2
        assert!(!tracker.record_token()); // 3 = max, stop
        assert_eq!(tracker.stop_reason(), Some(StopReason::MaxTokens));
    }

    #[test]
    fn test_tokens_remaining() {
        let budget = GenerationBudget::new(10);
        let mut tracker = BudgetTracker::new(budget);
        assert_eq!(tracker.tokens_remaining(), 10);
        tracker.record_token();
        assert_eq!(tracker.tokens_remaining(), 9);
    }

    #[test]
    fn test_record_eos() {
        let budget = GenerationBudget::new(100);
        let mut tracker = BudgetTracker::new(budget);
        tracker.record_eos();
        assert!(!tracker.can_continue());
        assert_eq!(tracker.stop_reason(), Some(StopReason::EndOfSequence));
    }

    #[test]
    fn test_record_user_stop() {
        let budget = GenerationBudget::new(100);
        let mut tracker = BudgetTracker::new(budget);
        tracker.record_user_stop();
        assert_eq!(tracker.stop_reason(), Some(StopReason::UserStop));
    }

    #[test]
    fn test_memory_limit() {
        let budget = GenerationBudget::new(100).with_memory_limit(1000);
        let mut tracker = BudgetTracker::new(budget);
        tracker.update_memory(2000); // over limit
        assert!(!tracker.record_token()); // should stop
        assert_eq!(tracker.stop_reason(), Some(StopReason::MemoryLimit));
    }

    #[test]
    fn test_token_utilization() {
        let budget = GenerationBudget::new(10);
        let mut tracker = BudgetTracker::new(budget);
        tracker.record_token();
        tracker.record_token();
        assert!((tracker.token_utilization() - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_token_utilization_zero_budget() {
        let budget = GenerationBudget::new(0);
        let tracker = BudgetTracker::new(budget);
        assert_eq!(tracker.token_utilization(), 1.0);
    }

    #[test]
    fn test_summary() {
        let budget = GenerationBudget::new(100);
        let mut tracker = BudgetTracker::new(budget);
        tracker.record_token();
        let s = tracker.summary();
        assert_eq!(s.tokens_generated, 1);
        assert_eq!(s.max_tokens, 100);
    }

    #[test]
    fn test_summary_display() {
        let s = BudgetSummary {
            tokens_generated: 5,
            max_tokens: 100,
            elapsed: Duration::from_millis(500),
            stop_reason: None,
            tokens_per_second: 10.0,
        };
        let text = format!("{s}");
        assert!(text.contains("5/100"));
        assert!(text.contains("tok/s"));
    }

    #[test]
    fn test_stop_reason_display() {
        assert_eq!(format!("{}", StopReason::MaxTokens), "max_tokens");
        assert_eq!(format!("{}", StopReason::EndOfSequence), "end_of_sequence");
    }

    #[test]
    fn test_time_remaining_none() {
        let budget = GenerationBudget::new(100);
        let tracker = BudgetTracker::new(budget);
        assert!(tracker.time_remaining().is_none());
    }

    #[test]
    fn test_time_remaining_some() {
        let budget = GenerationBudget::new(100).with_time_limit(Duration::from_mins(1));
        let tracker = BudgetTracker::new(budget);
        let remaining = tracker.time_remaining().unwrap();
        assert!(remaining.as_secs() >= 59);
    }
}
