//! Generation pipeline configuration and orchestration.
//!
//! Manages the full text generation pipeline: prompt → encode → generate → decode.

use std::time::{Duration, Instant};

/// Stop condition for generation.
#[derive(Debug, Clone, PartialEq)]
pub enum StopCondition {
    MaxTokens(usize),
    EosToken(u32),
    StopSequence(String),
    MaxTime(Duration),
}

/// Pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub stop_conditions: Vec<StopCondition>,
    pub stream: bool,
    pub seed: Option<u64>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            stop_conditions: vec![StopCondition::MaxTokens(256)],
            stream: false,
            seed: None,
        }
    }
}

impl PipelineConfig {
    pub fn greedy(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            temperature: 0.0,
            stop_conditions: vec![StopCondition::MaxTokens(max_tokens)],
            ..Default::default()
        }
    }

    pub fn creative(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            temperature: 0.9,
            top_p: 0.95,
            stop_conditions: vec![StopCondition::MaxTokens(max_tokens)],
            ..Default::default()
        }
    }

    pub fn with_eos(mut self, eos: u32) -> Self {
        self.stop_conditions.push(StopCondition::EosToken(eos));
        self
    }

    pub fn with_stop_seq(mut self, seq: &str) -> Self {
        self.stop_conditions.push(StopCondition::StopSequence(seq.to_string()));
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }

    pub fn is_deterministic(&self) -> bool {
        self.is_greedy() || self.seed.is_some()
    }
}

/// Generation result.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub tokens: Vec<u32>,
    pub text: String,
    pub stop_reason: StopReason,
    pub stats: GenerationStats,
}

/// Why generation stopped.
#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    MaxTokens,
    EosToken,
    StopSequence(String),
    Timeout,
    Error(String),
}

impl StopReason {
    pub fn as_str(&self) -> &str {
        match self {
            Self::MaxTokens => "max_tokens",
            Self::EosToken => "eos_token",
            Self::StopSequence(_) => "stop_sequence",
            Self::Timeout => "timeout",
            Self::Error(_) => "error",
        }
    }

    pub fn is_natural(&self) -> bool {
        matches!(self, Self::EosToken | Self::StopSequence(_))
    }
}

/// Generation statistics.
#[derive(Debug, Clone)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub total_duration: Duration,
    pub first_token_latency: Duration,
}

impl GenerationStats {
    pub fn tokens_per_second(&self) -> f64 {
        let secs = self.total_duration.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        self.generated_tokens as f64 / secs
    }

    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens + self.generated_tokens
    }
}

/// Pipeline stage tracking.
#[derive(Debug)]
pub struct PipelineStages {
    stages: Vec<(String, Duration)>,
    start: Instant,
}

impl Default for PipelineStages {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineStages {
    pub fn new() -> Self {
        Self { stages: Vec::new(), start: Instant::now() }
    }

    pub fn record(&mut self, name: &str, duration: Duration) {
        self.stages.push((name.to_string(), duration));
    }

    pub fn total(&self) -> Duration {
        self.start.elapsed()
    }

    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    pub fn slowest(&self) -> Option<(&str, Duration)> {
        self.stages.iter().max_by_key(|(_, d)| *d).map(|(n, d)| (n.as_str(), *d))
    }

    pub fn summary(&self) -> Vec<(&str, Duration)> {
        self.stages.iter().map(|(n, d)| (n.as_str(), *d)).collect()
    }
}

/// Check if generation should stop.
pub fn should_stop(
    config: &PipelineConfig,
    generated: &[u32],
    text: &str,
    start: Instant,
) -> Option<StopReason> {
    for cond in &config.stop_conditions {
        match cond {
            StopCondition::MaxTokens(max) => {
                if generated.len() >= *max {
                    return Some(StopReason::MaxTokens);
                }
            }
            StopCondition::EosToken(eos) => {
                if generated.last() == Some(eos) {
                    return Some(StopReason::EosToken);
                }
            }
            StopCondition::StopSequence(seq) => {
                if text.ends_with(seq.as_str()) {
                    return Some(StopReason::StopSequence(seq.clone()));
                }
            }
            StopCondition::MaxTime(dur) => {
                if start.elapsed() >= *dur {
                    return Some(StopReason::Timeout);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = PipelineConfig::default();
        assert_eq!(cfg.max_tokens, 256);
        assert_eq!(cfg.temperature, 1.0);
    }

    #[test]
    fn test_greedy_config() {
        let cfg = PipelineConfig::greedy(32);
        assert!(cfg.is_greedy());
        assert!(cfg.is_deterministic());
    }

    #[test]
    fn test_creative_config() {
        let cfg = PipelineConfig::creative(64);
        assert!(!cfg.is_greedy());
        assert!(!cfg.is_deterministic());
    }

    #[test]
    fn test_with_seed() {
        let cfg = PipelineConfig::creative(64).with_seed(42);
        assert!(cfg.is_deterministic());
    }

    #[test]
    fn test_with_eos() {
        let cfg = PipelineConfig::greedy(32).with_eos(2);
        assert!(cfg.stop_conditions.contains(&StopCondition::EosToken(2)));
    }

    #[test]
    fn test_stop_reason_str() {
        assert_eq!(StopReason::MaxTokens.as_str(), "max_tokens");
        assert_eq!(StopReason::EosToken.as_str(), "eos_token");
        assert!(StopReason::EosToken.is_natural());
        assert!(!StopReason::MaxTokens.is_natural());
    }

    #[test]
    fn test_generation_stats() {
        let stats = GenerationStats {
            prompt_tokens: 10,
            generated_tokens: 20,
            total_duration: Duration::from_secs(2),
            first_token_latency: Duration::from_millis(100),
        };
        assert!((stats.tokens_per_second() - 10.0).abs() < 0.1);
        assert_eq!(stats.total_tokens(), 30);
    }

    #[test]
    fn test_should_stop_max_tokens() {
        let cfg = PipelineConfig::greedy(3);
        let start = Instant::now();
        assert!(should_stop(&cfg, &[1, 2, 3], "", start).is_some());
        assert!(should_stop(&cfg, &[1, 2], "", start).is_none());
    }

    #[test]
    fn test_should_stop_eos() {
        let cfg = PipelineConfig::greedy(100).with_eos(2);
        let start = Instant::now();
        assert_eq!(should_stop(&cfg, &[1, 2], "", start), Some(StopReason::EosToken));
    }

    #[test]
    fn test_should_stop_sequence() {
        let cfg = PipelineConfig::greedy(100).with_stop_seq("</s>");
        let start = Instant::now();
        let r = should_stop(&cfg, &[1], "Hello</s>", start);
        assert!(matches!(r, Some(StopReason::StopSequence(_))));
    }

    #[test]
    fn test_pipeline_stages() {
        let mut stages = PipelineStages::new();
        stages.record("encode", Duration::from_millis(5));
        stages.record("generate", Duration::from_millis(100));
        assert_eq!(stages.stage_count(), 2);
        let (name, _) = stages.slowest().unwrap();
        assert_eq!(name, "generate");
    }

    #[test]
    fn test_pipeline_stages_summary() {
        let mut stages = PipelineStages::new();
        stages.record("a", Duration::from_millis(1));
        assert_eq!(stages.summary().len(), 1);
    }
}
