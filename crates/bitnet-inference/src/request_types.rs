//! Inference request and response types.
//!
//! Standardized types for submitting and receiving inference results.

use std::collections::HashMap;

/// An inference request.
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub id: String,
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub stop_sequences: Vec<String>,
    pub seed: Option<u64>,
    pub stream: bool,
    pub metadata: HashMap<String, String>,
}

impl Default for InferenceRequest {
    fn default() -> Self {
        Self {
            id: String::new(),
            prompt: String::new(),
            max_tokens: 128,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            repetition_penalty: 1.1,
            stop_sequences: Vec::new(),
            seed: None,
            stream: false,
            metadata: HashMap::new(),
        }
    }
}

impl InferenceRequest {
    pub fn new(prompt: &str) -> Self {
        Self { prompt: prompt.to_string(), ..Default::default() }
    }

    pub fn with_id(mut self, id: &str) -> Self {
        self.id = id.to_string();
        self
    }
    pub fn with_max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }
    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }
    pub fn with_stream(mut self, s: bool) -> Self {
        self.stream = s;
        self
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.01
    }
    pub fn is_deterministic(&self) -> bool {
        self.seed.is_some()
    }
}

/// Inference response.
#[derive(Debug, Clone)]
pub struct InferenceResponse {
    pub id: String,
    pub text: String,
    pub token_ids: Vec<u32>,
    pub token_count: usize,
    pub prompt_tokens: usize,
    pub finish_reason: FinishReason,
    pub timing: TimingInfo,
    pub usage: UsageInfo,
}

/// Why inference finished.
#[derive(Debug, Clone, PartialEq)]
pub enum FinishReason {
    MaxTokens,
    StopSequence,
    EosToken,
    Error(String),
}

/// Timing information.
#[derive(Debug, Clone, Default)]
pub struct TimingInfo {
    pub prompt_eval_ms: u64,
    pub generation_ms: u64,
    pub total_ms: u64,
    pub tokens_per_sec: f64,
}

/// Token usage information.
#[derive(Debug, Clone, Default)]
pub struct UsageInfo {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl UsageInfo {
    pub fn new(prompt: usize, completion: usize) -> Self {
        Self {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
        }
    }
}

/// Validate an inference request.
pub fn validate_request(req: &InferenceRequest) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    if req.prompt.is_empty() {
        errors.push("prompt is empty".into());
    }
    if req.max_tokens == 0 {
        errors.push("max_tokens must be > 0".into());
    }
    if req.max_tokens > 32768 {
        errors.push("max_tokens exceeds 32768".into());
    }
    if req.temperature < 0.0 {
        errors.push("temperature must be >= 0".into());
    }
    if req.temperature > 10.0 {
        errors.push("temperature exceeds 10.0".into());
    }
    if req.top_p <= 0.0 || req.top_p > 1.0 {
        errors.push("top_p must be in (0, 1]".into());
    }
    if req.repetition_penalty < 1.0 {
        errors.push("repetition_penalty must be >= 1.0".into());
    }

    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_request() {
        let r = InferenceRequest::default();
        assert_eq!(r.max_tokens, 128);
        assert!((r.temperature - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_new_request() {
        let r = InferenceRequest::new("Hello");
        assert_eq!(r.prompt, "Hello");
    }

    #[test]
    fn test_builder_chain() {
        let r = InferenceRequest::new("Hi").with_max_tokens(64).with_temperature(0.0).with_seed(42);
        assert_eq!(r.max_tokens, 64);
        assert!(r.is_greedy());
        assert!(r.is_deterministic());
    }

    #[test]
    fn test_is_greedy() {
        let r = InferenceRequest::new("x").with_temperature(0.0);
        assert!(r.is_greedy());
        let r2 = InferenceRequest::new("x").with_temperature(0.5);
        assert!(!r2.is_greedy());
    }

    #[test]
    fn test_validate_valid() {
        let r = InferenceRequest::new("Hello");
        assert!(validate_request(&r).is_ok());
    }

    #[test]
    fn test_validate_empty_prompt() {
        let r = InferenceRequest::default();
        let e = validate_request(&r);
        assert!(e.is_err());
    }

    #[test]
    fn test_validate_max_tokens_zero() {
        let r = InferenceRequest::new("x").with_max_tokens(0);
        assert!(validate_request(&r).is_err());
    }

    #[test]
    fn test_validate_temp_negative() {
        let mut r = InferenceRequest::new("x");
        r.temperature = -1.0;
        assert!(validate_request(&r).is_err());
    }

    #[test]
    fn test_validate_top_p() {
        let mut r = InferenceRequest::new("x");
        r.top_p = 0.0;
        assert!(validate_request(&r).is_err());
    }

    #[test]
    fn test_validate_rep_penalty() {
        let mut r = InferenceRequest::new("x");
        r.repetition_penalty = 0.5;
        assert!(validate_request(&r).is_err());
    }

    #[test]
    fn test_usage_info() {
        let u = UsageInfo::new(10, 5);
        assert_eq!(u.total_tokens, 15);
    }

    #[test]
    fn test_finish_reason() {
        assert_eq!(FinishReason::MaxTokens, FinishReason::MaxTokens);
        assert_ne!(FinishReason::MaxTokens, FinishReason::EosToken);
    }

    #[test]
    fn test_timing_default() {
        let t = TimingInfo::default();
        assert_eq!(t.total_ms, 0);
    }

    #[test]
    fn test_with_stream() {
        let r = InferenceRequest::new("x").with_stream(true);
        assert!(r.stream);
    }

    #[test]
    fn test_with_top_k() {
        let r = InferenceRequest::new("x").with_top_k(10);
        assert_eq!(r.top_k, 10);
    }
}
