//! OpenAI-compatible API types for Chat Completions, Embeddings,
//! Model List, and streaming SSE chunks.
//!
//! All types are fully typed, `serde`-serializable, and include request
//! validation suitable for an `/v1/…` HTTP façade.

use serde::{Deserialize, Serialize};
use std::fmt;

// ── Chat Completions ─────────────────────────────────────────────────────

/// Role in a chat conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

impl fmt::Display for ChatRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::System => f.write_str("system"),
            Self::User => f.write_str("user"),
            Self::Assistant => f.write_str("assistant"),
            Self::Tool => f.write_str("tool"),
        }
    }
}

/// A single message in the chat conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    /// Tool-call ID, present only when `role == Tool`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Name of the participant (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    /// Convenience constructor for a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: ChatRole::User, content: content.into(), tool_call_id: None, name: None }
    }

    /// Convenience constructor for a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: ChatRole::System, content: content.into(), tool_call_id: None, name: None }
    }

    /// Convenience constructor for an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: ChatRole::Assistant, content: content.into(), tool_call_id: None, name: None }
    }
}

/// OpenAI-compatible `/v1/chat/completions` request body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

impl ChatCompletionRequest {
    /// Validate the request, returning the first error found.
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.model.is_empty() {
            return Err(ValidationError::EmptyModel);
        }
        if self.messages.is_empty() {
            return Err(ValidationError::EmptyMessages);
        }
        if let Some(t) = self.temperature {
            if !(0.0..=2.0).contains(&t) {
                return Err(ValidationError::OutOfRange {
                    field: "temperature",
                    min: 0.0,
                    max: 2.0,
                    actual: f64::from(t),
                });
            }
        }
        if let Some(p) = self.top_p {
            if !(0.0..=1.0).contains(&p) {
                return Err(ValidationError::OutOfRange {
                    field: "top_p",
                    min: 0.0,
                    max: 1.0,
                    actual: f64::from(p),
                });
            }
        }
        if let Some(fp) = self.frequency_penalty {
            if !(-2.0..=2.0).contains(&fp) {
                return Err(ValidationError::OutOfRange {
                    field: "frequency_penalty",
                    min: -2.0,
                    max: 2.0,
                    actual: f64::from(fp),
                });
            }
        }
        if let Some(pp) = self.presence_penalty {
            if !(-2.0..=2.0).contains(&pp) {
                return Err(ValidationError::OutOfRange {
                    field: "presence_penalty",
                    min: -2.0,
                    max: 2.0,
                    actual: f64::from(pp),
                });
            }
        }
        if let Some(n) = self.n {
            if n == 0 {
                return Err(ValidationError::ZeroN);
            }
        }
        if let Some(mt) = self.max_tokens {
            if mt == 0 {
                return Err(ValidationError::ZeroMaxTokens);
            }
        }
        Ok(())
    }
}

/// Reason the model stopped generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
}

impl fmt::Display for FinishReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stop => f.write_str("stop"),
            Self::Length => f.write_str("length"),
            Self::ContentFilter => f.write_str("content_filter"),
            Self::ToolCalls => f.write_str("tool_calls"),
        }
    }
}

/// A single choice in the chat completion response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: FinishReason,
}

/// Token usage statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct UsageStats {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl UsageStats {
    /// Build usage stats; `total_tokens` is computed automatically.
    pub const fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self { prompt_tokens, completion_tokens, total_tokens: prompt_tokens + completion_tokens }
    }
}

/// Response from `/v1/chat/completions`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: UsageStats,
}

impl ChatCompletionResponse {
    /// Build a minimal single-choice response.
    pub fn single(
        id: impl Into<String>,
        model: impl Into<String>,
        created: u64,
        message: ChatMessage,
        finish_reason: FinishReason,
        usage: UsageStats,
    ) -> Self {
        Self {
            id: id.into(),
            object: "chat.completion".to_owned(),
            created,
            model: model.into(),
            choices: vec![ChatChoice { index: 0, message, finish_reason }],
            usage,
        }
    }
}

// ── Streaming (SSE) ──────────────────────────────────────────────────────

/// Delta content inside a streaming chunk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeltaContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<ChatRole>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// A single choice inside a streaming chunk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: DeltaContent,
    pub finish_reason: Option<FinishReason>,
}

/// SSE streaming chunk (`data: {…}` line in the event stream).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

impl StreamChunk {
    /// Build the initial chunk that carries the assistant role.
    pub fn role_chunk(id: impl Into<String>, model: impl Into<String>, created: u64) -> Self {
        Self {
            id: id.into(),
            object: "chat.completion.chunk".to_owned(),
            created,
            model: model.into(),
            choices: vec![StreamChoice {
                index: 0,
                delta: DeltaContent { role: Some(ChatRole::Assistant), content: None },
                finish_reason: None,
            }],
        }
    }

    /// Build a content-delta chunk.
    pub fn content_chunk(
        id: impl Into<String>,
        model: impl Into<String>,
        created: u64,
        text: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            object: "chat.completion.chunk".to_owned(),
            created,
            model: model.into(),
            choices: vec![StreamChoice {
                index: 0,
                delta: DeltaContent { role: None, content: Some(text.into()) },
                finish_reason: None,
            }],
        }
    }

    /// Build the final chunk that carries the finish reason.
    pub fn done_chunk(
        id: impl Into<String>,
        model: impl Into<String>,
        created: u64,
        finish_reason: FinishReason,
    ) -> Self {
        Self {
            id: id.into(),
            object: "chat.completion.chunk".to_owned(),
            created,
            model: model.into(),
            choices: vec![StreamChoice {
                index: 0,
                delta: DeltaContent { role: None, content: None },
                finish_reason: Some(finish_reason),
            }],
        }
    }

    /// Serialize to an SSE `data:` line (without trailing newlines).
    pub fn to_sse_line(&self) -> String {
        format!(
            "data: {}",
            serde_json::to_string(self).expect("StreamChunk is always serializable")
        )
    }
}

// ── Embeddings ───────────────────────────────────────────────────────────

/// Embedding input encoding format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EncodingFormat {
    Float,
    Base64,
}

/// Request body for `/v1/embeddings`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<EncodingFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// The input field can be a single string or a batch of strings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

impl EmbeddingInput {
    /// Return the total number of input texts.
    pub fn len(&self) -> usize {
        match self {
            Self::Single(_) => 1,
            Self::Batch(v) => v.len(),
        }
    }

    /// Whether the input is empty.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Single(s) => s.is_empty(),
            Self::Batch(v) => v.is_empty(),
        }
    }
}

impl EmbeddingRequest {
    /// Validate the request.
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.model.is_empty() {
            return Err(ValidationError::EmptyModel);
        }
        if self.input.is_empty() {
            return Err(ValidationError::EmptyInput);
        }
        if let Some(d) = self.dimensions {
            if d == 0 {
                return Err(ValidationError::ZeroDimensions);
            }
        }
        Ok(())
    }
}

/// A single embedding vector in the response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub index: u32,
    pub embedding: Vec<f32>,
}

/// Response from `/v1/embeddings`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

/// Token usage in an embedding response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

// ── Models ───────────────────────────────────────────────────────────────

/// A single model descriptor returned by `/v1/models`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

impl ModelInfo {
    /// Build a model info entry.
    pub fn new(id: impl Into<String>, created: u64, owned_by: impl Into<String>) -> Self {
        Self { id: id.into(), object: "model".to_owned(), created, owned_by: owned_by.into() }
    }
}

/// Response from `/v1/models`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

impl ModelListResponse {
    /// Build from an iterator of model infos.
    pub fn from_models(models: impl IntoIterator<Item = ModelInfo>) -> Self {
        Self { object: "list".to_owned(), data: models.into_iter().collect() }
    }
}

// ── Error response ───────────────────────────────────────────────────────

/// OpenAI-style error envelope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

/// Inner error detail.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl ErrorResponse {
    /// Build an error response.
    pub fn new(
        message: impl Into<String>,
        error_type: impl Into<String>,
        param: Option<String>,
        code: Option<String>,
    ) -> Self {
        Self {
            error: ErrorDetail {
                message: message.into(),
                error_type: error_type.into(),
                param,
                code,
            },
        }
    }

    /// Shorthand for a 400-class validation error.
    pub fn invalid_request(message: impl Into<String>, param: Option<String>) -> Self {
        Self::new(message, "invalid_request_error", param, Some("invalid_value".to_owned()))
    }

    /// Shorthand for model-not-found.
    pub fn model_not_found(model: &str) -> Self {
        Self::new(
            format!("The model `{model}` does not exist"),
            "invalid_request_error",
            Some("model".to_owned()),
            Some("model_not_found".to_owned()),
        )
    }
}

// ── Validation error ─────────────────────────────────────────────────────

/// Validation errors raised by `validate()` methods.
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    EmptyModel,
    EmptyMessages,
    EmptyInput,
    ZeroN,
    ZeroMaxTokens,
    ZeroDimensions,
    OutOfRange { field: &'static str, min: f64, max: f64, actual: f64 },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyModel => f.write_str("`model` must not be empty"),
            Self::EmptyMessages => f.write_str("`messages` must not be empty"),
            Self::EmptyInput => f.write_str("`input` must not be empty"),
            Self::ZeroN => f.write_str("`n` must be >= 1"),
            Self::ZeroMaxTokens => f.write_str("`max_tokens` must be >= 1"),
            Self::ZeroDimensions => f.write_str("`dimensions` must be >= 1"),
            Self::OutOfRange { field, min, max, actual } => {
                write!(f, "`{field}` must be between {min} and {max}, got {actual}")
            }
        }
    }
}

impl std::error::Error for ValidationError {}

impl From<ValidationError> for ErrorResponse {
    fn from(ve: ValidationError) -> Self {
        let param = match &ve {
            ValidationError::EmptyModel => Some("model".to_owned()),
            ValidationError::EmptyMessages => Some("messages".to_owned()),
            ValidationError::EmptyInput => Some("input".to_owned()),
            ValidationError::ZeroN => Some("n".to_owned()),
            ValidationError::ZeroMaxTokens => Some("max_tokens".to_owned()),
            ValidationError::ZeroDimensions => Some("dimensions".to_owned()),
            ValidationError::OutOfRange { field, .. } => Some((*field).to_owned()),
        };
        ErrorResponse::invalid_request(ve.to_string(), param)
    }
}

// ── Request converter ────────────────────────────────────────────────────

/// Intermediate inference parameters produced by [`RequestConverter`].
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceParams {
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub stop_sequences: Vec<String>,
    pub seed: Option<u64>,
    pub n: u32,
    pub stream: bool,
}

/// Converts OpenAI-format requests into internal inference parameters.
pub struct RequestConverter {
    default_max_tokens: u32,
}

impl Default for RequestConverter {
    fn default() -> Self {
        Self { default_max_tokens: 256 }
    }
}

impl RequestConverter {
    /// Create a converter with a custom default for `max_tokens`.
    pub const fn new(default_max_tokens: u32) -> Self {
        Self { default_max_tokens }
    }

    /// Convert a [`ChatCompletionRequest`] to [`InferenceParams`].
    ///
    /// The prompt is built by concatenating messages with role prefixes.
    pub fn convert(&self, req: &ChatCompletionRequest) -> Result<InferenceParams, ValidationError> {
        req.validate()?;

        let prompt = req
            .messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");

        Ok(InferenceParams {
            prompt,
            max_tokens: req.max_tokens.unwrap_or(self.default_max_tokens),
            temperature: req.temperature.unwrap_or(1.0),
            top_p: req.top_p.unwrap_or(1.0),
            frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
            presence_penalty: req.presence_penalty.unwrap_or(0.0),
            stop_sequences: req.stop.clone().unwrap_or_default(),
            seed: req.seed,
            n: req.n.unwrap_or(1),
            stream: req.stream.unwrap_or(false),
        })
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────

    fn sample_request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "bitnet-2b".into(),
            messages: vec![ChatMessage::system("You are helpful."), ChatMessage::user("Hi")],
            temperature: Some(0.7),
            max_tokens: Some(64),
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stream: None,
            stop: None,
            n: None,
            user: None,
            seed: None,
        }
    }

    fn roundtrip<T: Serialize + for<'de> Deserialize<'de> + PartialEq + fmt::Debug>(val: &T) {
        let json = serde_json::to_string(val).expect("serialize");
        let back: T = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(*val, back);
    }

    // ── ChatRole ────────────────────────────────────────────────────

    #[test]
    fn role_serialize_lowercase() {
        assert_eq!(serde_json::to_string(&ChatRole::System).unwrap(), "\"system\"");
        assert_eq!(serde_json::to_string(&ChatRole::User).unwrap(), "\"user\"");
        assert_eq!(serde_json::to_string(&ChatRole::Assistant).unwrap(), "\"assistant\"");
        assert_eq!(serde_json::to_string(&ChatRole::Tool).unwrap(), "\"tool\"");
    }

    #[test]
    fn role_deserialize() {
        let r: ChatRole = serde_json::from_str("\"tool\"").unwrap();
        assert_eq!(r, ChatRole::Tool);
    }

    #[test]
    fn role_display() {
        assert_eq!(ChatRole::Assistant.to_string(), "assistant");
    }

    // ── ChatMessage ─────────────────────────────────────────────────

    #[test]
    fn message_user_convenience() {
        let m = ChatMessage::user("hello");
        assert_eq!(m.role, ChatRole::User);
        assert_eq!(m.content, "hello");
        assert!(m.tool_call_id.is_none());
    }

    #[test]
    fn message_system_convenience() {
        let m = ChatMessage::system("be nice");
        assert_eq!(m.role, ChatRole::System);
    }

    #[test]
    fn message_assistant_convenience() {
        let m = ChatMessage::assistant("sure");
        assert_eq!(m.role, ChatRole::Assistant);
    }

    #[test]
    fn message_roundtrip() {
        roundtrip(&ChatMessage::user("test"));
    }

    #[test]
    fn message_with_tool_call_id() {
        let m = ChatMessage {
            role: ChatRole::Tool,
            content: "result".into(),
            tool_call_id: Some("call_123".into()),
            name: None,
        };
        let json = serde_json::to_string(&m).unwrap();
        assert!(json.contains("\"tool_call_id\":\"call_123\""));
        roundtrip(&m);
    }

    #[test]
    fn message_omits_none_fields() {
        let json = serde_json::to_string(&ChatMessage::user("hi")).unwrap();
        assert!(!json.contains("tool_call_id"));
        assert!(!json.contains("name"));
    }

    // ── ChatCompletionRequest ───────────────────────────────────────

    #[test]
    fn request_roundtrip() {
        roundtrip(&sample_request());
    }

    #[test]
    fn request_minimal_json() {
        let json = r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "m");
        assert_eq!(req.messages.len(), 1);
        assert!(req.temperature.is_none());
    }

    #[test]
    fn request_full_json_roundtrip() {
        let req = ChatCompletionRequest {
            model: "m".into(),
            messages: vec![ChatMessage::user("x")],
            temperature: Some(0.5),
            max_tokens: Some(100),
            top_p: Some(0.9),
            frequency_penalty: Some(0.1),
            presence_penalty: Some(-0.2),
            stream: Some(true),
            stop: Some(vec!["END".into()]),
            n: Some(2),
            user: Some("u1".into()),
            seed: Some(42),
        };
        roundtrip(&req);
    }

    // ── Validation ──────────────────────────────────────────────────

    #[test]
    fn validate_ok() {
        assert!(sample_request().validate().is_ok());
    }

    #[test]
    fn validate_empty_model() {
        let mut r = sample_request();
        r.model = String::new();
        assert_eq!(r.validate().unwrap_err(), ValidationError::EmptyModel);
    }

    #[test]
    fn validate_empty_messages() {
        let mut r = sample_request();
        r.messages.clear();
        assert_eq!(r.validate().unwrap_err(), ValidationError::EmptyMessages);
    }

    #[test]
    fn validate_temperature_too_low() {
        let mut r = sample_request();
        r.temperature = Some(-0.1);
        assert!(matches!(
            r.validate().unwrap_err(),
            ValidationError::OutOfRange { field: "temperature", .. }
        ));
    }

    #[test]
    fn validate_temperature_too_high() {
        let mut r = sample_request();
        r.temperature = Some(2.1);
        assert!(matches!(
            r.validate().unwrap_err(),
            ValidationError::OutOfRange { field: "temperature", .. }
        ));
    }

    #[test]
    fn validate_temperature_boundary() {
        let mut r = sample_request();
        r.temperature = Some(0.0);
        assert!(r.validate().is_ok());
        r.temperature = Some(2.0);
        assert!(r.validate().is_ok());
    }

    #[test]
    fn validate_top_p_out_of_range() {
        let mut r = sample_request();
        r.top_p = Some(1.5);
        assert!(matches!(
            r.validate().unwrap_err(),
            ValidationError::OutOfRange { field: "top_p", .. }
        ));
    }

    #[test]
    fn validate_frequency_penalty_bounds() {
        let mut r = sample_request();
        r.frequency_penalty = Some(-2.1);
        assert!(r.validate().is_err());
        r.frequency_penalty = Some(2.0);
        assert!(r.validate().is_ok());
    }

    #[test]
    fn validate_presence_penalty_bounds() {
        let mut r = sample_request();
        r.presence_penalty = Some(2.5);
        assert!(r.validate().is_err());
    }

    #[test]
    fn validate_zero_n() {
        let mut r = sample_request();
        r.n = Some(0);
        assert_eq!(r.validate().unwrap_err(), ValidationError::ZeroN);
    }

    #[test]
    fn validate_zero_max_tokens() {
        let mut r = sample_request();
        r.max_tokens = Some(0);
        assert_eq!(r.validate().unwrap_err(), ValidationError::ZeroMaxTokens);
    }

    #[test]
    fn validate_none_optionals_ok() {
        let r = ChatCompletionRequest {
            model: "m".into(),
            messages: vec![ChatMessage::user("x")],
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stream: None,
            stop: None,
            n: None,
            user: None,
            seed: None,
        };
        assert!(r.validate().is_ok());
    }

    // ── FinishReason ────────────────────────────────────────────────

    #[test]
    fn finish_reason_serialize() {
        assert_eq!(serde_json::to_string(&FinishReason::Stop).unwrap(), "\"stop\"");
        assert_eq!(serde_json::to_string(&FinishReason::Length).unwrap(), "\"length\"");
        assert_eq!(
            serde_json::to_string(&FinishReason::ContentFilter).unwrap(),
            "\"content_filter\""
        );
        assert_eq!(serde_json::to_string(&FinishReason::ToolCalls).unwrap(), "\"tool_calls\"");
    }

    #[test]
    fn finish_reason_display() {
        assert_eq!(FinishReason::ContentFilter.to_string(), "content_filter");
    }

    // ── ChatChoice / Response ───────────────────────────────────────

    #[test]
    fn choice_roundtrip() {
        let c = ChatChoice {
            index: 0,
            message: ChatMessage::assistant("hi"),
            finish_reason: FinishReason::Stop,
        };
        roundtrip(&c);
    }

    #[test]
    fn usage_stats_new() {
        let u = UsageStats::new(10, 20);
        assert_eq!(u.total_tokens, 30);
    }

    #[test]
    fn usage_stats_roundtrip() {
        roundtrip(&UsageStats::new(5, 10));
    }

    #[test]
    fn response_single_builder() {
        let resp = ChatCompletionResponse::single(
            "chatcmpl-1",
            "bitnet-2b",
            1_700_000_000,
            ChatMessage::assistant("Hello!"),
            FinishReason::Stop,
            UsageStats::new(3, 1),
        );
        assert_eq!(resp.object, "chat.completion");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].index, 0);
    }

    #[test]
    fn response_roundtrip() {
        let resp = ChatCompletionResponse::single(
            "id",
            "m",
            0,
            ChatMessage::assistant("ok"),
            FinishReason::Stop,
            UsageStats::new(1, 1),
        );
        roundtrip(&resp);
    }

    // ── Streaming ───────────────────────────────────────────────────

    #[test]
    fn stream_role_chunk() {
        let c = StreamChunk::role_chunk("id", "m", 0);
        assert_eq!(c.object, "chat.completion.chunk");
        assert_eq!(c.choices[0].delta.role, Some(ChatRole::Assistant));
        assert!(c.choices[0].finish_reason.is_none());
    }

    #[test]
    fn stream_content_chunk() {
        let c = StreamChunk::content_chunk("id", "m", 0, "hello");
        assert_eq!(c.choices[0].delta.content.as_deref(), Some("hello"));
        assert!(c.choices[0].delta.role.is_none());
    }

    #[test]
    fn stream_done_chunk() {
        let c = StreamChunk::done_chunk("id", "m", 0, FinishReason::Stop);
        assert_eq!(c.choices[0].finish_reason, Some(FinishReason::Stop));
        assert!(c.choices[0].delta.content.is_none());
    }

    #[test]
    fn stream_chunk_roundtrip() {
        roundtrip(&StreamChunk::content_chunk("id", "m", 0, "hi"));
    }

    #[test]
    fn sse_line_format() {
        let c = StreamChunk::content_chunk("id", "m", 0, "x");
        let line = c.to_sse_line();
        assert!(line.starts_with("data: {"));
        let payload = &line["data: ".len()..];
        let parsed: StreamChunk = serde_json::from_str(payload).unwrap();
        assert_eq!(parsed, c);
    }

    #[test]
    fn stream_full_sequence() {
        let chunks = vec![
            StreamChunk::role_chunk("id", "m", 0),
            StreamChunk::content_chunk("id", "m", 0, "He"),
            StreamChunk::content_chunk("id", "m", 0, "llo"),
            StreamChunk::done_chunk("id", "m", 0, FinishReason::Stop),
        ];
        assert_eq!(chunks.len(), 4);
        assert!(chunks[0].choices[0].delta.role.is_some());
        assert_eq!(chunks[3].choices[0].finish_reason, Some(FinishReason::Stop));
    }

    // ── Embeddings ──────────────────────────────────────────────────

    #[test]
    fn embedding_input_single() {
        let i = EmbeddingInput::Single("hello".into());
        assert_eq!(i.len(), 1);
        assert!(!i.is_empty());
    }

    #[test]
    fn embedding_input_batch() {
        let i = EmbeddingInput::Batch(vec!["a".into(), "b".into()]);
        assert_eq!(i.len(), 2);
    }

    #[test]
    fn embedding_input_empty_batch() {
        let i = EmbeddingInput::Batch(vec![]);
        assert!(i.is_empty());
    }

    #[test]
    fn embedding_request_roundtrip() {
        let req = EmbeddingRequest {
            model: "m".into(),
            input: EmbeddingInput::Single("hi".into()),
            encoding_format: Some(EncodingFormat::Float),
            dimensions: Some(512),
            user: None,
        };
        roundtrip(&req);
    }

    #[test]
    fn embedding_request_batch_roundtrip() {
        let req = EmbeddingRequest {
            model: "m".into(),
            input: EmbeddingInput::Batch(vec!["a".into(), "b".into()]),
            encoding_format: None,
            dimensions: None,
            user: None,
        };
        roundtrip(&req);
    }

    #[test]
    fn embedding_validate_ok() {
        let req = EmbeddingRequest {
            model: "m".into(),
            input: EmbeddingInput::Single("hi".into()),
            encoding_format: None,
            dimensions: None,
            user: None,
        };
        assert!(req.validate().is_ok());
    }

    #[test]
    fn embedding_validate_empty_model() {
        let req = EmbeddingRequest {
            model: String::new(),
            input: EmbeddingInput::Single("hi".into()),
            encoding_format: None,
            dimensions: None,
            user: None,
        };
        assert_eq!(req.validate().unwrap_err(), ValidationError::EmptyModel);
    }

    #[test]
    fn embedding_validate_empty_input() {
        let req = EmbeddingRequest {
            model: "m".into(),
            input: EmbeddingInput::Batch(vec![]),
            encoding_format: None,
            dimensions: None,
            user: None,
        };
        assert_eq!(req.validate().unwrap_err(), ValidationError::EmptyInput);
    }

    #[test]
    fn embedding_validate_zero_dimensions() {
        let req = EmbeddingRequest {
            model: "m".into(),
            input: EmbeddingInput::Single("hi".into()),
            encoding_format: None,
            dimensions: Some(0),
            user: None,
        };
        assert_eq!(req.validate().unwrap_err(), ValidationError::ZeroDimensions);
    }

    #[test]
    fn embedding_response_roundtrip() {
        let resp = EmbeddingResponse {
            object: "list".into(),
            data: vec![EmbeddingData {
                object: "embedding".into(),
                index: 0,
                embedding: vec![0.1, 0.2, 0.3],
            }],
            model: "m".into(),
            usage: EmbeddingUsage { prompt_tokens: 5, total_tokens: 5 },
        };
        roundtrip(&resp);
    }

    #[test]
    fn encoding_format_roundtrip() {
        roundtrip(&EncodingFormat::Float);
        roundtrip(&EncodingFormat::Base64);
    }

    // ── Model list ──────────────────────────────────────────────────

    #[test]
    fn model_info_new() {
        let m = ModelInfo::new("bitnet-2b", 1_700_000_000, "bitnet-rs");
        assert_eq!(m.object, "model");
        assert_eq!(m.id, "bitnet-2b");
    }

    #[test]
    fn model_info_roundtrip() {
        roundtrip(&ModelInfo::new("m", 0, "o"));
    }

    #[test]
    fn model_list_from_models() {
        let list = ModelListResponse::from_models(vec![
            ModelInfo::new("a", 0, "o"),
            ModelInfo::new("b", 1, "o"),
        ]);
        assert_eq!(list.object, "list");
        assert_eq!(list.data.len(), 2);
    }

    #[test]
    fn model_list_empty() {
        let list = ModelListResponse::from_models(std::iter::empty());
        assert!(list.data.is_empty());
    }

    #[test]
    fn model_list_roundtrip() {
        roundtrip(&ModelListResponse::from_models(vec![ModelInfo::new("m", 0, "o")]));
    }

    // ── Error response ──────────────────────────────────────────────

    #[test]
    fn error_response_roundtrip() {
        let e = ErrorResponse::new("bad", "invalid_request_error", None, None);
        roundtrip(&e);
    }

    #[test]
    fn error_invalid_request() {
        let e = ErrorResponse::invalid_request("oops", Some("temperature".into()));
        assert_eq!(e.error.error_type, "invalid_request_error");
        assert_eq!(e.error.param.as_deref(), Some("temperature"));
        assert_eq!(e.error.code.as_deref(), Some("invalid_value"));
    }

    #[test]
    fn error_model_not_found() {
        let e = ErrorResponse::model_not_found("gpt-99");
        assert!(e.error.message.contains("gpt-99"));
        assert_eq!(e.error.code.as_deref(), Some("model_not_found"));
    }

    #[test]
    fn error_response_json_shape() {
        let e = ErrorResponse::new("msg", "type", Some("p".into()), Some("c".into()));
        let json: serde_json::Value = serde_json::to_value(&e).unwrap();
        assert!(json.get("error").is_some());
        assert_eq!(json["error"]["type"], "type");
    }

    // ── ValidationError ─────────────────────────────────────────────

    #[test]
    fn validation_error_display() {
        let e = ValidationError::EmptyModel;
        assert!(e.to_string().contains("model"));
    }

    #[test]
    fn validation_error_out_of_range_display() {
        let e =
            ValidationError::OutOfRange { field: "temperature", min: 0.0, max: 2.0, actual: 3.0 };
        let s = e.to_string();
        assert!(s.contains("temperature"));
        assert!(s.contains("0"));
        assert!(s.contains("2"));
        assert!(s.contains("3"));
    }

    #[test]
    fn validation_error_into_error_response() {
        let e: ErrorResponse = ValidationError::EmptyModel.into();
        assert_eq!(e.error.param.as_deref(), Some("model"));
    }

    #[test]
    fn validation_error_out_of_range_into_response() {
        let e: ErrorResponse =
            ValidationError::OutOfRange { field: "top_p", min: 0.0, max: 1.0, actual: 5.0 }.into();
        assert_eq!(e.error.param.as_deref(), Some("top_p"));
    }

    // ── RequestConverter ────────────────────────────────────────────

    #[test]
    fn converter_defaults() {
        let c = RequestConverter::default();
        let params = c.convert(&sample_request()).unwrap();
        assert_eq!(params.max_tokens, 64);
        assert!((params.temperature - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn converter_uses_custom_default_max_tokens() {
        let c = RequestConverter::new(128);
        let mut req = sample_request();
        req.max_tokens = None;
        let params = c.convert(&req).unwrap();
        assert_eq!(params.max_tokens, 128);
    }

    #[test]
    fn converter_prompt_format() {
        let c = RequestConverter::default();
        let params = c.convert(&sample_request()).unwrap();
        assert!(params.prompt.contains("system: You are helpful."));
        assert!(params.prompt.contains("user: Hi"));
    }

    #[test]
    fn converter_defaults_for_omitted_fields() {
        let c = RequestConverter::default();
        let req = ChatCompletionRequest {
            model: "m".into(),
            messages: vec![ChatMessage::user("x")],
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stream: None,
            stop: None,
            n: None,
            user: None,
            seed: None,
        };
        let params = c.convert(&req).unwrap();
        assert!((params.temperature - 1.0).abs() < f32::EPSILON);
        assert!((params.top_p - 1.0).abs() < f32::EPSILON);
        assert!((params.frequency_penalty).abs() < f32::EPSILON);
        assert!((params.presence_penalty).abs() < f32::EPSILON);
        assert_eq!(params.n, 1);
        assert!(!params.stream);
        assert!(params.stop_sequences.is_empty());
        assert!(params.seed.is_none());
    }

    #[test]
    fn converter_stream_flag() {
        let c = RequestConverter::default();
        let mut req = sample_request();
        req.stream = Some(true);
        assert!(c.convert(&req).unwrap().stream);
    }

    #[test]
    fn converter_stop_sequences() {
        let c = RequestConverter::default();
        let mut req = sample_request();
        req.stop = Some(vec!["END".into(), "STOP".into()]);
        let params = c.convert(&req).unwrap();
        assert_eq!(params.stop_sequences, vec!["END", "STOP"]);
    }

    #[test]
    fn converter_seed_passthrough() {
        let c = RequestConverter::default();
        let mut req = sample_request();
        req.seed = Some(42);
        assert_eq!(c.convert(&req).unwrap().seed, Some(42));
    }

    #[test]
    fn converter_rejects_invalid() {
        let c = RequestConverter::default();
        let mut req = sample_request();
        req.model = String::new();
        assert!(c.convert(&req).is_err());
    }

    #[test]
    fn converter_multi_message_prompt() {
        let c = RequestConverter::default();
        let req = ChatCompletionRequest {
            model: "m".into(),
            messages: vec![
                ChatMessage::system("sys"),
                ChatMessage::user("u1"),
                ChatMessage::assistant("a1"),
                ChatMessage::user("u2"),
            ],
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stream: None,
            stop: None,
            n: None,
            user: None,
            seed: None,
        };
        let params = c.convert(&req).unwrap();
        let lines: Vec<&str> = params.prompt.lines().collect();
        assert_eq!(lines.len(), 4);
        assert!(lines[2].starts_with("assistant:"));
    }

    #[test]
    fn converter_penalties_passthrough() {
        let c = RequestConverter::default();
        let mut req = sample_request();
        req.frequency_penalty = Some(0.5);
        req.presence_penalty = Some(-0.3);
        let params = c.convert(&req).unwrap();
        assert!((params.frequency_penalty - 0.5).abs() < f32::EPSILON);
        assert!((params.presence_penalty - (-0.3)).abs() < f32::EPSILON);
    }

    #[test]
    fn converter_n_passthrough() {
        let c = RequestConverter::default();
        let mut req = sample_request();
        req.n = Some(3);
        assert_eq!(c.convert(&req).unwrap().n, 3);
    }
}
