//! Generation output tracking and statistics.
//!
//! Collects token-level details during generation for diagnostics.

use std::time::Instant;

/// A single generated token with metadata.
#[derive(Debug, Clone)]
pub struct TokenOutput {
    pub token_id: u32,
    pub text: String,
    pub logit: f32,
    pub probability: f32,
    pub latency_us: u64,
    pub position: usize,
}

/// Tracks the output of a generation run.
#[derive(Debug)]
pub struct GenerationOutput {
    tokens: Vec<TokenOutput>,
    start: Instant,
    prompt_tokens: usize,
    stop_reason: Option<StopReason>,
}

/// Why generation stopped.
#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    MaxTokens,
    EosToken,
    StopSequence(String),
    UserAbort,
    Error(String),
}

impl Default for GenerationOutput {
    fn default() -> Self {
        Self::new(0)
    }
}

impl GenerationOutput {
    pub fn new(prompt_tokens: usize) -> Self {
        Self { tokens: Vec::new(), start: Instant::now(), prompt_tokens, stop_reason: None }
    }

    pub fn add_token(&mut self, token: TokenOutput) {
        self.tokens.push(token);
    }

    pub fn set_stop_reason(&mut self, reason: StopReason) {
        self.stop_reason = Some(reason);
    }

    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }
    pub fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }
    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens + self.tokens.len()
    }
    pub fn tokens(&self) -> &[TokenOutput] {
        &self.tokens
    }
    pub fn stop_reason(&self) -> Option<&StopReason> {
        self.stop_reason.as_ref()
    }

    /// Full generated text.
    pub fn text(&self) -> String {
        self.tokens.iter().map(|t| t.text.as_str()).collect()
    }

    /// All token IDs generated.
    pub fn token_ids(&self) -> Vec<u32> {
        self.tokens.iter().map(|t| t.token_id).collect()
    }

    /// Average token probability.
    pub fn avg_probability(&self) -> f32 {
        if self.tokens.is_empty() {
            return 0.0;
        }
        self.tokens.iter().map(|t| t.probability).sum::<f32>() / self.tokens.len() as f32
    }

    /// Per-token average latency in microseconds.
    pub fn avg_latency_us(&self) -> u64 {
        if self.tokens.is_empty() {
            return 0;
        }
        self.tokens.iter().map(|t| t.latency_us).sum::<u64>() / self.tokens.len() as u64
    }

    /// Total generation time in milliseconds.
    pub fn total_time_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    /// Tokens per second.
    pub fn tokens_per_sec(&self) -> f64 {
        let ms = self.total_time_ms();
        if ms == 0 {
            return 0.0;
        }
        self.tokens.len() as f64 / (ms as f64 / 1000.0)
    }

    /// Perplexity estimate from token probabilities.
    pub fn perplexity(&self) -> f64 {
        if self.tokens.is_empty() {
            return 0.0;
        }
        let log_sum: f64 = self.tokens.iter().map(|t| (t.probability as f64).max(1e-10).ln()).sum();
        (-log_sum / self.tokens.len() as f64).exp()
    }

    /// Summary statistics.
    pub fn summary(&self) -> OutputSummary {
        OutputSummary {
            generated_tokens: self.tokens.len(),
            prompt_tokens: self.prompt_tokens,
            total_tokens: self.total_tokens(),
            avg_probability: self.avg_probability(),
            avg_latency_us: self.avg_latency_us(),
            perplexity: self.perplexity(),
            stop_reason: self.stop_reason.clone(),
        }
    }
}

/// Summary of generation output.
#[derive(Debug, Clone)]
pub struct OutputSummary {
    pub generated_tokens: usize,
    pub prompt_tokens: usize,
    pub total_tokens: usize,
    pub avg_probability: f32,
    pub avg_latency_us: u64,
    pub perplexity: f64,
    pub stop_reason: Option<StopReason>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_token(id: u32, text: &str, prob: f32) -> TokenOutput {
        TokenOutput {
            token_id: id,
            text: text.to_string(),
            logit: 1.0,
            probability: prob,
            latency_us: 100,
            position: id as usize,
        }
    }

    #[test]
    fn test_new() {
        let g = GenerationOutput::new(10);
        assert_eq!(g.prompt_tokens(), 10);
        assert_eq!(g.token_count(), 0);
    }

    #[test]
    fn test_add_token() {
        let mut g = GenerationOutput::new(5);
        g.add_token(make_token(1, "hello", 0.9));
        assert_eq!(g.token_count(), 1);
        assert_eq!(g.total_tokens(), 6);
    }

    #[test]
    fn test_text() {
        let mut g = GenerationOutput::new(0);
        g.add_token(make_token(1, "Hello", 0.9));
        g.add_token(make_token(2, " world", 0.8));
        assert_eq!(g.text(), "Hello world");
    }

    #[test]
    fn test_token_ids() {
        let mut g = GenerationOutput::new(0);
        g.add_token(make_token(10, "a", 0.9));
        g.add_token(make_token(20, "b", 0.8));
        assert_eq!(g.token_ids(), vec![10, 20]);
    }

    #[test]
    fn test_avg_probability() {
        let mut g = GenerationOutput::new(0);
        g.add_token(make_token(1, "a", 0.8));
        g.add_token(make_token(2, "b", 0.6));
        assert!((g.avg_probability() - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_avg_probability_empty() {
        let g = GenerationOutput::new(0);
        assert_eq!(g.avg_probability(), 0.0);
    }

    #[test]
    fn test_avg_latency() {
        let mut g = GenerationOutput::new(0);
        g.add_token(make_token(1, "a", 0.9));
        g.add_token(make_token(2, "b", 0.8));
        assert_eq!(g.avg_latency_us(), 100);
    }

    #[test]
    fn test_stop_reason() {
        let mut g = GenerationOutput::new(0);
        assert!(g.stop_reason().is_none());
        g.set_stop_reason(StopReason::EosToken);
        assert_eq!(g.stop_reason(), Some(&StopReason::EosToken));
    }

    #[test]
    fn test_stop_sequence() {
        let s = StopReason::StopSequence("</s>".into());
        assert_eq!(s, StopReason::StopSequence("</s>".into()));
    }

    #[test]
    fn test_perplexity() {
        let mut g = GenerationOutput::new(0);
        g.add_token(make_token(1, "a", 0.5));
        g.add_token(make_token(2, "b", 0.5));
        let ppl = g.perplexity();
        assert!(ppl > 1.0);
    }

    #[test]
    fn test_perplexity_empty() {
        let g = GenerationOutput::new(0);
        assert_eq!(g.perplexity(), 0.0);
    }

    #[test]
    fn test_summary() {
        let mut g = GenerationOutput::new(5);
        g.add_token(make_token(1, "a", 0.9));
        g.set_stop_reason(StopReason::MaxTokens);
        let s = g.summary();
        assert_eq!(s.generated_tokens, 1);
        assert_eq!(s.prompt_tokens, 5);
        assert_eq!(s.total_tokens, 6);
        assert!(matches!(s.stop_reason, Some(StopReason::MaxTokens)));
    }

    #[test]
    fn test_default() {
        let g = GenerationOutput::default();
        assert_eq!(g.prompt_tokens(), 0);
    }

    #[test]
    fn test_tokens_accessor() {
        let mut g = GenerationOutput::new(0);
        g.add_token(make_token(1, "x", 0.5));
        assert_eq!(g.tokens().len(), 1);
        assert_eq!(g.tokens()[0].token_id, 1);
    }
}
