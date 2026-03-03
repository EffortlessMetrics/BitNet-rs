//! HTTP response builder for inference API.
//!
//! Constructs standardized JSON responses for generation, health,
//! and error endpoints with consistent formatting.

use std::time::Duration;

/// Completion choice in a response.
#[derive(Debug, Clone)]
pub struct Choice {
    pub index: usize,
    pub text: String,
    pub finish_reason: FinishReason,
    pub tokens_generated: usize,
}

/// Reason generation finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    MaxTokens,
    Error,
}

impl FinishReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::MaxTokens => "max_tokens",
            FinishReason::Error => "error",
        }
    }
}

/// Token usage statistics.
#[derive(Debug, Clone, Default)]
pub struct UsageStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

impl UsageStats {
    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens + self.completion_tokens
    }
}

/// A complete generation response.
#[derive(Debug)]
pub struct GenerationResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: UsageStats,
    pub duration: Duration,
}

/// Builder for constructing generation responses.
#[derive(Debug)]
pub struct ResponseBuilder {
    id: String,
    model: String,
    choices: Vec<Choice>,
    usage: UsageStats,
    duration: Duration,
}

impl ResponseBuilder {
    pub fn new(id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            model: model.into(),
            choices: Vec::new(),
            usage: UsageStats::default(),
            duration: Duration::ZERO,
        }
    }

    pub fn add_choice(mut self, text: impl Into<String>, finish_reason: FinishReason) -> Self {
        let idx = self.choices.len();
        let text = text.into();
        let tokens = text.split_whitespace().count();
        self.choices.push(Choice {
            index: idx,
            text,
            finish_reason,
            tokens_generated: tokens,
        });
        self
    }

    pub fn with_usage(mut self, prompt_tokens: usize, completion_tokens: usize) -> Self {
        self.usage = UsageStats {
            prompt_tokens,
            completion_tokens,
        };
        self
    }

    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }

    pub fn build(self) -> GenerationResponse {
        GenerationResponse {
            id: self.id,
            model: self.model,
            choices: self.choices,
            usage: self.usage,
            duration: self.duration,
        }
    }
}

impl GenerationResponse {
    pub fn to_json(&self) -> String {
        let choices_json: Vec<String> = self
            .choices
            .iter()
            .map(|c| {
                format!(
                    r#"{{"index":{},"text":"{}","finish_reason":"{}","tokens":{}}}"#,
                    c.index,
                    escape_json(&c.text),
                    c.finish_reason.as_str(),
                    c.tokens_generated,
                )
            })
            .collect();

        format!(
            r#"{{"id":"{}","model":"{}","choices":[{}],"usage":{{"prompt_tokens":{},"completion_tokens":{},"total_tokens":{}}},"duration_ms":{}}}"#,
            escape_json(&self.id),
            escape_json(&self.model),
            choices_json.join(","),
            self.usage.prompt_tokens,
            self.usage.completion_tokens,
            self.usage.total_tokens(),
            self.duration.as_millis(),
        )
    }

    pub fn tokens_per_second(&self) -> f64 {
        let secs = self.duration.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        self.usage.completion_tokens as f64 / secs
    }
}

/// Build a health check response.
pub fn health_response(status: &str, model_loaded: bool) -> String {
    format!(
        r#"{{"status":"{}","model_loaded":{}}}"#,
        escape_json(status),
        model_loaded,
    )
}

/// Build an error response.
pub fn error_response(code: u16, message: &str) -> String {
    format!(
        r#"{{"error":{{"code":{},"message":"{}"}}}}"#,
        code,
        escape_json(message),
    )
}

/// Escape special characters for JSON strings.
fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_basic() {
        let resp = ResponseBuilder::new("req-1", "phi-4")
            .add_choice("Hello world", FinishReason::Stop)
            .with_usage(5, 2)
            .build();
        assert_eq!(resp.id, "req-1");
        assert_eq!(resp.choices.len(), 1);
    }

    #[test]
    fn test_multiple_choices() {
        let resp = ResponseBuilder::new("req-2", "test")
            .add_choice("Hello", FinishReason::Stop)
            .add_choice("World", FinishReason::MaxTokens)
            .build();
        assert_eq!(resp.choices.len(), 2);
        assert_eq!(resp.choices[0].index, 0);
        assert_eq!(resp.choices[1].index, 1);
    }

    #[test]
    fn test_finish_reason() {
        assert_eq!(FinishReason::Stop.as_str(), "stop");
        assert_eq!(FinishReason::MaxTokens.as_str(), "max_tokens");
        assert_eq!(FinishReason::Error.as_str(), "error");
    }

    #[test]
    fn test_usage_total() {
        let usage = UsageStats {
            prompt_tokens: 10,
            completion_tokens: 20,
        };
        assert_eq!(usage.total_tokens(), 30);
    }

    #[test]
    fn test_to_json() {
        let resp = ResponseBuilder::new("id1", "model1")
            .add_choice("hi", FinishReason::Stop)
            .with_usage(3, 1)
            .with_duration(Duration::from_millis(100))
            .build();
        let json = resp.to_json();
        assert!(json.contains("\"id\":\"id1\""));
        assert!(json.contains("\"model\":\"model1\""));
        assert!(json.contains("\"total_tokens\":4"));
    }

    #[test]
    fn test_tokens_per_second() {
        let resp = ResponseBuilder::new("id", "m")
            .with_usage(0, 100)
            .with_duration(Duration::from_secs(2))
            .build();
        assert!((resp.tokens_per_second() - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_tokens_per_second_zero() {
        let resp = ResponseBuilder::new("id", "m").build();
        assert_eq!(resp.tokens_per_second(), 0.0);
    }

    #[test]
    fn test_health_response() {
        let json = health_response("healthy", true);
        assert!(json.contains("\"status\":\"healthy\""));
        assert!(json.contains("\"model_loaded\":true"));
    }

    #[test]
    fn test_error_response() {
        let json = error_response(400, "Bad request");
        assert!(json.contains("\"code\":400"));
        assert!(json.contains("\"message\":\"Bad request\""));
    }

    #[test]
    fn test_escape_json() {
        let json = error_response(500, "line1\nline2");
        assert!(json.contains("\\n"));
    }

    #[test]
    fn test_escape_quotes() {
        let json = error_response(500, "say \"hello\"");
        assert!(json.contains("\\\"hello\\\""));
    }

    #[test]
    fn test_empty_response() {
        let resp = ResponseBuilder::new("empty", "none").build();
        let json = resp.to_json();
        assert!(json.contains("\"choices\":[]"));
    }
}
