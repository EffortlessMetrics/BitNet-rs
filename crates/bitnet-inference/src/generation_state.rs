//! State machine for autoregressive text generation.
//!
//! Track generation progress: token counts, stop conditions,
//! timing, and state transitions.

use std::time::{Duration, Instant};

/// Generation state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenState {
    /// Not started.
    Idle,
    /// Processing prompt tokens (prefill).
    Prefill,
    /// Generating tokens autoregressively.
    Generating,
    /// Reached a stop condition.
    Stopped(StopReason),
    /// Failed with an error.
    Failed,
}

impl GenState {
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Prefill | Self::Generating)
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Stopped(_) | Self::Failed)
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Prefill => "prefill",
            Self::Generating => "generating",
            Self::Stopped(_) => "stopped",
            Self::Failed => "failed",
        }
    }
}

/// Reason generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Reached max_tokens.
    MaxTokens,
    /// Generated an EOS token.
    EndOfSequence,
    /// Client cancellation.
    Cancelled,
    /// Repetition loop detected.
    RepetitionLoop,
}

impl StopReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::MaxTokens => "max_tokens",
            Self::EndOfSequence => "end_of_sequence",
            Self::Cancelled => "cancelled",
            Self::RepetitionLoop => "repetition_loop",
        }
    }
}

/// Tracks the state of a generation request.
#[derive(Debug)]
pub struct GenerationTracker {
    state: GenState,
    prompt_tokens: usize,
    generated_tokens: usize,
    max_tokens: usize,
    eos_token_id: Option<u32>,
    start_time: Option<Instant>,
    prefill_duration: Option<Duration>,
    last_token_ids: Vec<u32>,
    repetition_window: usize,
}

impl GenerationTracker {
    pub fn new(max_tokens: usize) -> Self {
        Self {
            state: GenState::Idle,
            prompt_tokens: 0,
            generated_tokens: 0,
            max_tokens,
            eos_token_id: None,
            start_time: None,
            prefill_duration: None,
            last_token_ids: Vec::new(),
            repetition_window: 8,
        }
    }

    pub fn with_eos(mut self, eos_id: u32) -> Self {
        self.eos_token_id = Some(eos_id);
        self
    }

    pub fn with_repetition_window(mut self, window: usize) -> Self {
        self.repetition_window = window;
        self
    }

    /// Begin prefill phase.
    pub fn start_prefill(&mut self, prompt_len: usize) {
        self.state = GenState::Prefill;
        self.prompt_tokens = prompt_len;
        self.start_time = Some(Instant::now());
    }

    /// Transition from prefill to generation.
    pub fn start_generation(&mut self) {
        if let Some(start) = self.start_time {
            self.prefill_duration = Some(start.elapsed());
        }
        self.state = GenState::Generating;
    }

    /// Record a generated token and check stop conditions.
    pub fn on_token(&mut self, token_id: u32) -> GenState {
        if self.state != GenState::Generating {
            return self.state;
        }

        self.generated_tokens += 1;
        self.last_token_ids.push(token_id);

        // Check EOS
        if let Some(eos) = self.eos_token_id
            && token_id == eos
        {
            self.state = GenState::Stopped(StopReason::EndOfSequence);
            return self.state;
        }

        // Check max tokens
        if self.generated_tokens >= self.max_tokens {
            self.state = GenState::Stopped(StopReason::MaxTokens);
            return self.state;
        }

        // Check repetition
        if self.detect_repetition() {
            self.state = GenState::Stopped(StopReason::RepetitionLoop);
            return self.state;
        }

        self.state
    }

    /// Cancel generation.
    pub fn cancel(&mut self) {
        self.state = GenState::Stopped(StopReason::Cancelled);
    }

    /// Mark as failed.
    pub fn fail(&mut self) {
        self.state = GenState::Failed;
    }

    fn detect_repetition(&self) -> bool {
        let w = self.repetition_window;
        if self.last_token_ids.len() < w * 2 {
            return false;
        }
        let n = self.last_token_ids.len();
        let recent = &self.last_token_ids[n - w..];
        let prior = &self.last_token_ids[n - 2 * w..n - w];
        recent == prior
    }

    pub fn state(&self) -> GenState {
        self.state
    }

    pub fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    pub fn generated_tokens(&self) -> usize {
        self.generated_tokens
    }

    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens + self.generated_tokens
    }

    pub fn elapsed(&self) -> Duration {
        self.start_time.map(|s| s.elapsed()).unwrap_or(Duration::ZERO)
    }

    pub fn tokens_per_second(&self) -> f64 {
        let secs = self.elapsed().as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        self.generated_tokens as f64 / secs
    }

    pub fn summary(&self) -> String {
        format!(
            "state={}, prompt={}, generated={}/{}, tps={:.1}",
            self.state.name(),
            self.prompt_tokens,
            self.generated_tokens,
            self.max_tokens,
            self.tokens_per_second(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let tracker = GenerationTracker::new(100);
        assert_eq!(tracker.state(), GenState::Idle);
        assert!(!tracker.state().is_active());
    }

    #[test]
    fn test_prefill_to_generating() {
        let mut tracker = GenerationTracker::new(100);
        tracker.start_prefill(10);
        assert_eq!(tracker.state(), GenState::Prefill);
        assert!(tracker.state().is_active());
        tracker.start_generation();
        assert_eq!(tracker.state(), GenState::Generating);
    }

    #[test]
    fn test_max_tokens_stop() {
        let mut tracker = GenerationTracker::new(3);
        tracker.start_prefill(5);
        tracker.start_generation();
        tracker.on_token(1);
        tracker.on_token(2);
        let state = tracker.on_token(3);
        assert_eq!(state, GenState::Stopped(StopReason::MaxTokens));
    }

    #[test]
    fn test_eos_stop() {
        let mut tracker = GenerationTracker::new(100).with_eos(2);
        tracker.start_prefill(1);
        tracker.start_generation();
        tracker.on_token(1);
        let state = tracker.on_token(2); // EOS
        assert_eq!(state, GenState::Stopped(StopReason::EndOfSequence));
    }

    #[test]
    fn test_cancel() {
        let mut tracker = GenerationTracker::new(100);
        tracker.start_prefill(1);
        tracker.start_generation();
        tracker.cancel();
        assert_eq!(tracker.state(), GenState::Stopped(StopReason::Cancelled));
    }

    #[test]
    fn test_fail() {
        let mut tracker = GenerationTracker::new(100);
        tracker.start_prefill(1);
        tracker.fail();
        assert_eq!(tracker.state(), GenState::Failed);
        assert!(tracker.state().is_terminal());
    }

    #[test]
    fn test_token_counts() {
        let mut tracker = GenerationTracker::new(100);
        tracker.start_prefill(10);
        tracker.start_generation();
        tracker.on_token(1);
        tracker.on_token(2);
        assert_eq!(tracker.prompt_tokens(), 10);
        assert_eq!(tracker.generated_tokens(), 2);
        assert_eq!(tracker.total_tokens(), 12);
    }

    #[test]
    fn test_repetition_detection() {
        let mut tracker = GenerationTracker::new(100).with_repetition_window(3);
        tracker.start_prefill(1);
        tracker.start_generation();
        // Generate: 1,2,3,1,2,3 — should detect repetition
        for &id in &[1, 2, 3, 1, 2] {
            tracker.on_token(id);
        }
        let state = tracker.on_token(3);
        assert_eq!(state, GenState::Stopped(StopReason::RepetitionLoop));
    }

    #[test]
    fn test_stop_reason_str() {
        assert_eq!(StopReason::MaxTokens.as_str(), "max_tokens");
        assert_eq!(StopReason::EndOfSequence.as_str(), "end_of_sequence");
    }

    #[test]
    fn test_state_name() {
        assert_eq!(GenState::Idle.name(), "idle");
        assert_eq!(GenState::Generating.name(), "generating");
    }

    #[test]
    fn test_summary() {
        let mut tracker = GenerationTracker::new(100);
        tracker.start_prefill(5);
        tracker.start_generation();
        let s = tracker.summary();
        assert!(s.contains("generating"));
        assert!(s.contains("prompt=5"));
    }

    #[test]
    fn test_is_terminal() {
        assert!(GenState::Stopped(StopReason::MaxTokens).is_terminal());
        assert!(GenState::Failed.is_terminal());
        assert!(!GenState::Generating.is_terminal());
    }
}
