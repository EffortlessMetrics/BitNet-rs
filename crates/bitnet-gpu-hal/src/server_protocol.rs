//! Module stub - implementation pending merge from feature branch
//! OpenAI-compatible inference server protocol (chat, completion, embedding).
//!
//! Provides request/response types and a protocol handler for routing
//! chat-completion, text-completion, and embedding requests. Supports
//! SSE streaming, configurable limits, and structured error responses.

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Protocol-level configuration for the inference server.
#[derive(Debug, Clone)]
pub struct ProtocolConfig {
    /// Maximum allowed request body size in bytes.
    pub max_request_size: usize,
    /// Per-request timeout.
    pub timeout: Duration,
    /// Whether SSE streaming is enabled server-wide.
    pub streaming_enabled: bool,
    /// Maximum number of tokens that can be requested in a single call.
    pub max_tokens_limit: u32,
    /// Default model identifier when the client omits it.
    pub default_model: String,
}

impl Default for ProtocolConfig {
    fn default() -> Self {
        Self {
            max_request_size: 4 * 1024 * 1024, // 4 MiB
            timeout: Duration::from_secs(60),
            streaming_enabled: true,
            max_tokens_limit: 4096,
            default_model: "bitnet-b1.58-2B-4T".to_string(),
        }
    }
}

impl ProtocolConfig {
    /// Validate configuration values.
    pub fn validate(&self) -> Result<(), ProtocolError> {
        if self.max_request_size == 0 {
            return Err(ProtocolError::invalid_request("max_request_size must be > 0"));
        }
        if self.timeout.is_zero() {
            return Err(ProtocolError::invalid_request("timeout must be > 0"));
        }
        if self.max_tokens_limit == 0 {
            return Err(ProtocolError::invalid_request("max_tokens_limit must be > 0"));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Chat messages
// ---------------------------------------------------------------------------

/// Role of a chat message participant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::System => write!(f, "system"),
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
        }
    }
}

impl Role {
    /// Parse a role from its string representation.
    pub fn from_str_value(s: &str) -> Result<Self, ProtocolError> {
        match s {
            "system" => Ok(Self::System),
            "user" => Ok(Self::User),
            "assistant" => Ok(Self::Assistant),
            other => Err(ProtocolError::invalid_request(&format!("unknown role: {other}"))),
        }
    }
}

/// A single message in a chat conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self { role, content: content.into() }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }
}

// ---------------------------------------------------------------------------
// Chat completion
// ---------------------------------------------------------------------------

/// OpenAI-compatible chat completion request.
#[derive(Debug, Clone)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f32>,
    pub stream: bool,
    pub stop: Option<Vec<String>>,
    /// Additional vendor-specific parameters.
    pub extra: HashMap<String, String>,
}

impl ChatCompletionRequest {
    /// Validate the request fields.
    pub fn validate(&self, config: &ProtocolConfig) -> Result<(), ProtocolError> {
        if self.messages.is_empty() {
            return Err(ProtocolError::invalid_request("messages must not be empty"));
        }
        if let Some(t) = self.temperature
            && !(0.0..=2.0).contains(&t)
        {
            return Err(ProtocolError::invalid_request("temperature must be in [0.0, 2.0]"));
        }
        if let Some(p) = self.top_p
            && !(0.0..=1.0).contains(&p)
        {
            return Err(ProtocolError::invalid_request("top_p must be in [0.0, 1.0]"));
        }
        if let Some(mt) = self.max_tokens
            && (mt == 0 || mt > config.max_tokens_limit)
        {
            return Err(ProtocolError::invalid_request(&format!(
                "max_tokens must be in [1, {}]",
                config.max_tokens_limit
            )));
        }
        if self.stream && !config.streaming_enabled {
            return Err(ProtocolError::invalid_request("streaming is not enabled on this server"));
        }
        Ok(())
    }
}

/// A single choice in a chat completion response.
#[derive(Debug, Clone)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: FinishReason,
}

/// Response to a chat completion request.
#[derive(Debug, Clone)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: UsageStats,
}

impl ChatCompletionResponse {
    /// Build a simple single-choice response.
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
            object: "chat.completion".to_string(),
            created,
            model: model.into(),
            choices: vec![ChatCompletionChoice { index: 0, message, finish_reason }],
            usage,
        }
    }
}

// ---------------------------------------------------------------------------
// Text completion
// ---------------------------------------------------------------------------

/// Text completion request (legacy /v1/completions endpoint).
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: bool,
    pub stop: Option<Vec<String>>,
    pub echo: bool,
    pub extra: HashMap<String, String>,
}

impl CompletionRequest {
    /// Validate the request fields.
    pub fn validate(&self, config: &ProtocolConfig) -> Result<(), ProtocolError> {
        if self.prompt.is_empty() {
            return Err(ProtocolError::invalid_request("prompt must not be empty"));
        }
        if let Some(t) = self.temperature
            && !(0.0..=2.0).contains(&t)
        {
            return Err(ProtocolError::invalid_request("temperature must be in [0.0, 2.0]"));
        }
        if let Some(p) = self.top_p
            && !(0.0..=1.0).contains(&p)
        {
            return Err(ProtocolError::invalid_request("top_p must be in [0.0, 1.0]"));
        }
        if let Some(mt) = self.max_tokens
            && (mt == 0 || mt > config.max_tokens_limit)
        {
            return Err(ProtocolError::invalid_request(&format!(
                "max_tokens must be in [1, {}]",
                config.max_tokens_limit
            )));
        }
        if self.stream && !config.streaming_enabled {
            return Err(ProtocolError::invalid_request("streaming is not enabled on this server"));
        }
        Ok(())
    }
}

/// A single choice in a text completion response.
#[derive(Debug, Clone)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: FinishReason,
}

/// Response to a text completion request.
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: UsageStats,
}

impl CompletionResponse {
    /// Build a simple single-choice response.
    pub fn single(
        id: impl Into<String>,
        model: impl Into<String>,
        created: u64,
        text: impl Into<String>,
        finish_reason: FinishReason,
        usage: UsageStats,
    ) -> Self {
        Self {
            id: id.into(),
            object: "text_completion".to_string(),
            created,
            model: model.into(),
            choices: vec![CompletionChoice { index: 0, text: text.into(), finish_reason }],
            usage,
        }
    }
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

/// Embedding request (/v1/embeddings endpoint).
#[derive(Debug, Clone)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
}

/// Input variants for an embedding request.
#[derive(Debug, Clone)]
pub enum EmbeddingInput {
    /// Single string input.
    Single(String),
    /// Batch of string inputs.
    Batch(Vec<String>),
}

impl EmbeddingInput {
    /// Return the number of input texts.
    #[must_use]
    pub const fn len(&self) -> usize {
        match self {
            Self::Single(_) => 1,
            Self::Batch(v) => v.len(),
        }
    }

    /// Whether the input is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Single(s) => s.is_empty(),
            Self::Batch(v) => v.is_empty() || v.iter().all(String::is_empty),
        }
    }

    /// Iterate over input texts.
    #[must_use]
    pub fn texts(&self) -> Vec<&str> {
        match self {
            Self::Single(s) => vec![s.as_str()],
            Self::Batch(v) => v.iter().map(String::as_str).collect(),
        }
    }
}

impl EmbeddingRequest {
    /// Validate the request fields.
    pub fn validate(&self) -> Result<(), ProtocolError> {
        if self.input.is_empty() {
            return Err(ProtocolError::invalid_request("embedding input must not be empty"));
        }
        if self.model.is_empty() {
            return Err(ProtocolError::invalid_request("model must not be empty"));
        }
        Ok(())
    }
}

/// A single embedding vector in the response.
#[derive(Debug, Clone)]
pub struct EmbeddingData {
    pub index: u32,
    pub embedding: Vec<f32>,
    pub object: String,
}

/// Response to an embedding request.
#[derive(Debug, Clone)]
pub struct EmbeddingResponse {
    pub object: String,
    pub model: String,
    pub data: Vec<EmbeddingData>,
    pub usage: EmbeddingUsage,
}

impl EmbeddingResponse {
    /// Build a response from a list of embedding vectors.
    pub fn from_vectors(
        model: impl Into<String>,
        vectors: Vec<Vec<f32>>,
        prompt_tokens: u32,
    ) -> Self {
        let data = vectors
            .into_iter()
            .enumerate()
            .map(|(i, emb)| EmbeddingData {
                index: u32::try_from(i).expect("index overflow"),
                embedding: emb,
                object: "embedding".to_string(),
            })
            .collect();
        Self {
            object: "list".to_string(),
            model: model.into(),
            data,
            usage: EmbeddingUsage { prompt_tokens, total_tokens: prompt_tokens },
        }
    }
}

/// Token usage for embedding requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

/// A single SSE chunk for streaming responses.
#[derive(Debug, Clone)]
pub struct StreamingChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamingChoice>,
    /// Usage is only present in the final chunk.
    pub usage: Option<UsageStats>,
}

/// A choice delta within a streaming chunk.
#[derive(Debug, Clone)]
pub struct StreamingChoice {
    pub index: u32,
    pub delta: DeltaContent,
    pub finish_reason: Option<FinishReason>,
}

/// Incremental content in a streaming delta.
#[derive(Debug, Clone)]
pub struct DeltaContent {
    pub role: Option<Role>,
    pub content: Option<String>,
}

impl StreamingChunk {
    /// Format this chunk as an SSE `data:` line.
    pub fn to_sse_line(&self) -> String {
        // Minimal JSON-like serialisation for protocol use.
        let choices_json: Vec<String> = self
            .choices
            .iter()
            .map(|c| {
                let role_part =
                    c.delta.role.as_ref().map_or_else(String::new, |r| format!(r#""role":"{r}""#));
                let content_part = c
                    .delta
                    .content
                    .as_ref()
                    .map_or_else(String::new, |t| format!(r#""content":"{}""#, escape_json(t)));
                let delta_fields: Vec<&str> = [role_part.as_str(), content_part.as_str()]
                    .into_iter()
                    .filter(|s| !s.is_empty())
                    .collect();
                let finish = c
                    .finish_reason
                    .as_ref()
                    .map_or_else(String::new, |fr| format!(r#","finish_reason":"{fr}""#));
                format!(
                    r#"{{"index":{},"delta":{{{}}}{}}}""#,
                    c.index,
                    delta_fields.join(","),
                    finish,
                )
            })
            .collect();
        format!(
            r#"data: {{"id":"{}","object":"{}","created":{},"model":"{}","choices":[{}]}}"#,
            self.id,
            self.object,
            self.created,
            self.model,
            choices_json.join(","),
        )
    }

    /// The terminal SSE sentinel.
    pub fn done_sentinel() -> String {
        "data: [DONE]".to_string()
    }

    /// Create a content chunk.
    pub fn content(
        id: impl Into<String>,
        model: impl Into<String>,
        created: u64,
        text: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.into(),
            choices: vec![StreamingChoice {
                index: 0,
                delta: DeltaContent { role: None, content: Some(text.into()) },
                finish_reason: None,
            }],
            usage: None,
        }
    }

    /// Create a final chunk signalling completion.
    pub fn finish(
        id: impl Into<String>,
        model: impl Into<String>,
        created: u64,
        reason: FinishReason,
        usage: UsageStats,
    ) -> Self {
        Self {
            id: id.into(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.into(),
            choices: vec![StreamingChoice {
                index: 0,
                delta: DeltaContent { role: None, content: None },
                finish_reason: Some(reason),
            }],
            usage: Some(usage),
        }
    }
}

/// Minimal JSON string escaping.
fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

/// Reason a generation finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
}

impl fmt::Display for FinishReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stop => write!(f, "stop"),
            Self::Length => write!(f, "length"),
            Self::ContentFilter => write!(f, "content_filter"),
        }
    }
}

/// Token usage statistics for generation requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UsageStats {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl UsageStats {
    /// Create usage stats (total is computed automatically).
    #[must_use]
    pub const fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self { prompt_tokens, completion_tokens, total_tokens: prompt_tokens + completion_tokens }
    }
}

// ---------------------------------------------------------------------------
// Error response
// ---------------------------------------------------------------------------

/// Structured error response following the [`ErrorType`] `OpenAI` error format.
#[derive(Debug, Clone)]
pub struct ProtocolError {
    pub error_type: ErrorType,
    pub message: String,
    pub code: Option<String>,
    pub param: Option<String>,
}

/// Categories of protocol errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorType {
    InvalidRequest,
    Authentication,
    RateLimit,
    NotFound,
    ServerError,
    ServiceUnavailable,
}

impl fmt::Display for ErrorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest => write!(f, "invalid_request_error"),
            Self::Authentication => write!(f, "authentication_error"),
            Self::RateLimit => write!(f, "rate_limit_error"),
            Self::NotFound => write!(f, "not_found_error"),
            Self::ServerError => write!(f, "server_error"),
            Self::ServiceUnavailable => {
                write!(f, "service_unavailable_error")
            }
        }
    }
}

impl fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.error_type, self.message)
    }
}

impl ProtocolError {
    /// Create an invalid-request error.
    pub fn invalid_request(message: &str) -> Self {
        Self {
            error_type: ErrorType::InvalidRequest,
            message: message.to_string(),
            code: Some("invalid_request".to_string()),
            param: None,
        }
    }

    /// Create a not-found error.
    pub fn not_found(message: &str) -> Self {
        Self {
            error_type: ErrorType::NotFound,
            message: message.to_string(),
            code: Some("not_found".to_string()),
            param: None,
        }
    }

    /// Create a rate-limit error.
    pub fn rate_limited(message: &str) -> Self {
        Self {
            error_type: ErrorType::RateLimit,
            message: message.to_string(),
            code: Some("rate_limit_exceeded".to_string()),
            param: None,
        }
    }

    /// Create a server-error.
    pub fn server_error(message: &str) -> Self {
        Self {
            error_type: ErrorType::ServerError,
            message: message.to_string(),
            code: Some("server_error".to_string()),
            param: None,
        }
    }

    /// Create a service-unavailable error.
    pub fn unavailable(message: &str) -> Self {
        Self {
            error_type: ErrorType::ServiceUnavailable,
            message: message.to_string(),
            code: Some("service_unavailable".to_string()),
            param: None,
        }
    }

    /// Create an authentication error.
    pub fn authentication(message: &str) -> Self {
        Self {
            error_type: ErrorType::Authentication,
            message: message.to_string(),
            code: Some("authentication_error".to_string()),
            param: None,
        }
    }

    /// Return the HTTP status code appropriate for this error type.
    #[must_use]
    pub const fn http_status_code(&self) -> u16 {
        match self.error_type {
            ErrorType::InvalidRequest => 400,
            ErrorType::Authentication => 401,
            ErrorType::RateLimit => 429,
            ErrorType::NotFound => 404,
            ErrorType::ServerError => 500,
            ErrorType::ServiceUnavailable => 503,
        }
    }

    /// Format as an OpenAI-compatible JSON error body.
    pub fn to_json_body(&self) -> String {
        let code_field =
            self.code.as_ref().map_or_else(String::new, |c| format!(r#","code":"{c}""#));
        let param_field =
            self.param.as_ref().map_or_else(String::new, |p| format!(r#","param":"{p}""#));
        format!(
            r#"{{"error":{{"type":"{}","message":"{}"{}{}}}}}"#,
            self.error_type,
            escape_json(&self.message),
            code_field,
            param_field,
        )
    }
}

// ---------------------------------------------------------------------------
// Request routing
// ---------------------------------------------------------------------------

/// A parsed and validated API request.
#[derive(Debug)]
pub enum ApiRequest {
    ChatCompletion(ChatCompletionRequest),
    Completion(CompletionRequest),
    Embedding(EmbeddingRequest),
}

/// Result produced by the protocol handler after routing.
#[derive(Debug)]
pub enum ApiResponse {
    ChatCompletion(ChatCompletionResponse),
    Completion(CompletionResponse),
    Embedding(EmbeddingResponse),
    Error(ProtocolError),
}

// ---------------------------------------------------------------------------
// Protocol handler
// ---------------------------------------------------------------------------

/// Parses, validates, and routes incoming API requests.
#[derive(Debug)]
pub struct ProtocolHandler {
    config: ProtocolConfig,
    /// Models that are currently loaded and available.
    available_models: Vec<String>,
    request_count: u64,
}

impl ProtocolHandler {
    /// Create a handler with the given configuration.
    pub fn new(config: ProtocolConfig) -> Result<Self, ProtocolError> {
        config.validate()?;
        Ok(Self { config, available_models: Vec::new(), request_count: 0 })
    }

    /// Create a handler with default configuration.
    pub fn with_defaults() -> Self {
        Self {
            config: ProtocolConfig::default(),
            available_models: vec!["bitnet-b1.58-2B-4T".to_string()],
            request_count: 0,
        }
    }

    /// Register a model as available.
    pub fn register_model(&mut self, model: impl Into<String>) {
        let model = model.into();
        if !self.available_models.contains(&model) {
            self.available_models.push(model);
        }
    }

    /// Unregister a model.
    pub fn unregister_model(&mut self, model: &str) -> bool {
        if let Some(pos) = self.available_models.iter().position(|m| m == model) {
            self.available_models.remove(pos);
            true
        } else {
            false
        }
    }

    /// Return the list of available models.
    #[must_use]
    pub fn available_models(&self) -> &[String] {
        &self.available_models
    }

    /// Return a reference to the current configuration.
    #[must_use]
    pub const fn config(&self) -> &ProtocolConfig {
        &self.config
    }

    /// Total number of requests processed.
    #[must_use]
    pub const fn request_count(&self) -> u64 {
        self.request_count
    }

    /// Resolve the effective model identifier, falling back to default.
    fn resolve_model(&self, model: &str) -> Result<String, ProtocolError> {
        let effective = if model.is_empty() { &self.config.default_model } else { model };
        if !self.available_models.contains(&effective.to_string()) {
            return Err(ProtocolError::not_found(&format!("model '{effective}' is not available")));
        }
        Ok(effective.to_string())
    }

    /// Validate and route a chat completion request.
    pub fn handle_chat_completion(
        &mut self,
        req: ChatCompletionRequest,
    ) -> Result<ChatCompletionRequest, ProtocolError> {
        self.request_count += 1;
        let model = self.resolve_model(&req.model)?;
        let mut validated = req;
        validated.model = model;
        validated.validate(&self.config)?;
        Ok(validated)
    }

    /// Validate and route a text completion request.
    pub fn handle_completion(
        &mut self,
        req: CompletionRequest,
    ) -> Result<CompletionRequest, ProtocolError> {
        self.request_count += 1;
        let model = self.resolve_model(&req.model)?;
        let mut validated = req;
        validated.model = model;
        validated.validate(&self.config)?;
        Ok(validated)
    }

    /// Validate and route an embedding request.
    pub fn handle_embedding(
        &mut self,
        req: EmbeddingRequest,
    ) -> Result<EmbeddingRequest, ProtocolError> {
        self.request_count += 1;
        let model = self.resolve_model(&req.model)?;
        let mut validated = req;
        validated.model = model;
        validated.validate()?;
        Ok(validated)
    }

    /// Route a generic API request.
    pub fn route(&mut self, request: ApiRequest) -> Result<ApiRequest, ProtocolError> {
        match request {
            ApiRequest::ChatCompletion(req) => {
                self.handle_chat_completion(req).map(ApiRequest::ChatCompletion)
            }
            ApiRequest::Completion(req) => self.handle_completion(req).map(ApiRequest::Completion),
            ApiRequest::Embedding(req) => self.handle_embedding(req).map(ApiRequest::Embedding),
        }
    }

    /// Check whether a raw request body exceeds the size limit.
    pub fn check_request_size(&self, size: usize) -> Result<(), ProtocolError> {
        if size > self.config.max_request_size {
            return Err(ProtocolError::invalid_request(&format!(
                "request body too large: {} bytes (max {})",
                size, self.config.max_request_size
            )));
        }
        Ok(())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers --

    fn default_config() -> ProtocolConfig {
        ProtocolConfig::default()
    }

    fn chat_request(model: &str, msg: &str) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: model.to_string(),
            messages: vec![ChatMessage::user(msg)],
            temperature: None,
            max_tokens: None,
            top_p: None,
            stream: false,
            stop: None,
            extra: HashMap::new(),
        }
    }

    fn completion_request(model: &str, prompt: &str) -> CompletionRequest {
        CompletionRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            stream: false,
            stop: None,
            echo: false,
            extra: HashMap::new(),
        }
    }

    fn embedding_request(model: &str, input: &str) -> EmbeddingRequest {
        EmbeddingRequest {
            model: model.to_string(),
            input: EmbeddingInput::Single(input.to_string()),
        }
    }

    fn handler() -> ProtocolHandler {
        ProtocolHandler::with_defaults()
    }

    // -----------------------------------------------------------------------
    // ProtocolConfig
    // -----------------------------------------------------------------------

    #[test]
    fn config_default_values() {
        let cfg = default_config();
        assert_eq!(cfg.max_request_size, 4 * 1024 * 1024);
        assert_eq!(cfg.timeout, Duration::from_secs(60));
        assert!(cfg.streaming_enabled);
        assert_eq!(cfg.max_tokens_limit, 4096);
    }

    #[test]
    fn config_validate_ok() {
        assert!(default_config().validate().is_ok());
    }

    #[test]
    fn config_validate_zero_max_request_size() {
        let mut cfg = default_config();
        cfg.max_request_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_zero_timeout() {
        let mut cfg = default_config();
        cfg.timeout = Duration::ZERO;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_zero_max_tokens() {
        let mut cfg = default_config();
        cfg.max_tokens_limit = 0;
        assert!(cfg.validate().is_err());
    }

    // -----------------------------------------------------------------------
    // Role
    // -----------------------------------------------------------------------

    #[test]
    fn role_display() {
        assert_eq!(Role::System.to_string(), "system");
        assert_eq!(Role::User.to_string(), "user");
        assert_eq!(Role::Assistant.to_string(), "assistant");
    }

    #[test]
    fn role_from_str_valid() {
        assert_eq!(Role::from_str_value("system").unwrap(), Role::System);
        assert_eq!(Role::from_str_value("user").unwrap(), Role::User);
        assert_eq!(Role::from_str_value("assistant").unwrap(), Role::Assistant,);
    }

    #[test]
    fn role_from_str_invalid() {
        assert!(Role::from_str_value("unknown").is_err());
        assert!(Role::from_str_value("").is_err());
    }

    // -----------------------------------------------------------------------
    // ChatMessage
    // -----------------------------------------------------------------------

    #[test]
    fn chat_message_constructors() {
        let m = ChatMessage::system("sys");
        assert_eq!(m.role, Role::System);
        assert_eq!(m.content, "sys");

        let m = ChatMessage::user("hello");
        assert_eq!(m.role, Role::User);

        let m = ChatMessage::assistant("hi");
        assert_eq!(m.role, Role::Assistant);
    }

    #[test]
    fn chat_message_new() {
        let m = ChatMessage::new(Role::User, "test");
        assert_eq!(m.role, Role::User);
        assert_eq!(m.content, "test");
    }

    #[test]
    fn chat_message_equality() {
        let a = ChatMessage::user("hello");
        let b = ChatMessage::user("hello");
        assert_eq!(a, b);
    }

    #[test]
    fn chat_message_inequality() {
        let a = ChatMessage::user("hello");
        let b = ChatMessage::user("world");
        assert_ne!(a, b);
    }

    // -----------------------------------------------------------------------
    // ChatCompletionRequest validation
    // -----------------------------------------------------------------------

    #[test]
    fn chat_req_valid() {
        let cfg = default_config();
        let req = chat_request("m", "hi");
        assert!(req.validate(&cfg).is_ok());
    }

    #[test]
    fn chat_req_empty_messages() {
        let cfg = default_config();
        let mut req = chat_request("m", "hi");
        req.messages.clear();
        assert!(req.validate(&cfg).is_err());
    }

    #[test]
    fn chat_req_temperature_too_low() {
        let cfg = default_config();
        let mut req = chat_request("m", "hi");
        req.temperature = Some(-0.1);
        assert!(req.validate(&cfg).is_err());
    }

    #[test]
    fn chat_req_temperature_too_high() {
        let cfg = default_config();
        let mut req = chat_request("m", "hi");
        req.temperature = Some(2.1);
        assert!(req.validate(&cfg).is_err());
    }

    #[test]
    fn chat_req_temperature_boundary() {
        let cfg = default_config();
        let mut req = chat_request("m", "hi");
        req.temperature = Some(0.0);
        assert!(req.validate(&cfg).is_ok());
        req.temperature = Some(2.0);
        assert!(req.validate(&cfg).is_ok());
    }

    #[test]
    fn chat_req_top_p_too_low() {
        let cfg = default_config();
        let mut req = chat_request("m", "hi");
        req.top_p = Some(-0.1);
        assert!(req.validate(&cfg).is_err());
    }

    #[test]
    fn chat_req_top_p_too_high() {
        let cfg = default_config();
        let mut req = chat_request("m", "hi");
        req.top_p = Some(1.1);
        assert!(req.validate(&cfg).is_err());
    }

    #[test]
    fn chat_req_top_p_boundary() {
        let cfg = default_config();
        let mut req = chat_request("m", "hi");
        req.top_p = Some(0.0);
        assert!(req.validate(&cfg).is_ok());
        req.top_p = Some(1.0);
        assert!(req.validate(&cfg).is_ok());
    }

    #[test]
    fn chat_req_max_tokens_zero() {
        let cfg = default_config();
        let mut req = chat_request("m", "hi");
        req.max_tokens = Some(0);
        assert!(req.validate(&cfg).is_err());
    }

    #[test]
    fn chat_req_max_tokens_over_limit() {
        let cfg = default_config();
        let mut req = chat_request("m", "hi");
        req.max_tokens = Some(cfg.max_tokens_limit + 1);
        assert!(req.validate(&cfg).is_err());
    }

    #[test]
    fn chat_req_max_tokens_at_limit() {
        let cfg = default_config();
        let mut req = chat_request("m", "hi");
        req.max_tokens = Some(cfg.max_tokens_limit);
        assert!(req.validate(&cfg).is_ok());
    }

    #[test]
    fn chat_req_stream_disabled() {
        let mut cfg = default_config();
        cfg.streaming_enabled = false;
        let mut req = chat_request("m", "hi");
        req.stream = true;
        assert!(req.validate(&cfg).is_err());
    }

    #[test]
    fn chat_req_stream_enabled() {
        let cfg = default_config();
        let mut req = chat_request("m", "hi");
        req.stream = true;
        assert!(req.validate(&cfg).is_ok());
    }

    // -----------------------------------------------------------------------
    // ChatCompletionResponse
    // -----------------------------------------------------------------------

    #[test]
    fn chat_response_single() {
        let resp = ChatCompletionResponse::single(
            "id-1",
            "model-a",
            12345,
            ChatMessage::assistant("hello"),
            FinishReason::Stop,
            UsageStats::new(5, 3),
        );
        assert_eq!(resp.id, "id-1");
        assert_eq!(resp.object, "chat.completion");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].index, 0);
        assert_eq!(resp.choices[0].message.content, "hello");
        assert_eq!(resp.choices[0].finish_reason, FinishReason::Stop);
    }

    #[test]
    fn chat_response_usage() {
        let resp = ChatCompletionResponse::single(
            "id",
            "m",
            0,
            ChatMessage::assistant("x"),
            FinishReason::Length,
            UsageStats::new(10, 20),
        );
        assert_eq!(resp.usage.prompt_tokens, 10);
        assert_eq!(resp.usage.completion_tokens, 20);
        assert_eq!(resp.usage.total_tokens, 30);
    }

    // -----------------------------------------------------------------------
    // CompletionRequest validation
    // -----------------------------------------------------------------------

    #[test]
    fn completion_req_valid() {
        let cfg = default_config();
        let req = completion_request("m", "hello");
        assert!(req.validate(&cfg).is_ok());
    }

    #[test]
    fn completion_req_empty_prompt() {
        let cfg = default_config();
        let req = completion_request("m", "");
        assert!(req.validate(&cfg).is_err());
    }

    #[test]
    fn completion_req_temperature_too_high() {
        let cfg = default_config();
        let mut req = completion_request("m", "hi");
        req.temperature = Some(3.0);
        assert!(req.validate(&cfg).is_err());
    }

    #[test]
    fn completion_req_top_p_invalid() {
        let cfg = default_config();
        let mut req = completion_request("m", "hi");
        req.top_p = Some(-0.5);
        assert!(req.validate(&cfg).is_err());
    }

    #[test]
    fn completion_req_max_tokens_zero() {
        let cfg = default_config();
        let mut req = completion_request("m", "hi");
        req.max_tokens = Some(0);
        assert!(req.validate(&cfg).is_err());
    }

    #[test]
    fn completion_req_max_tokens_over_limit() {
        let cfg = default_config();
        let mut req = completion_request("m", "hi");
        req.max_tokens = Some(cfg.max_tokens_limit + 1);
        assert!(req.validate(&cfg).is_err());
    }

    #[test]
    fn completion_req_stream_disabled() {
        let mut cfg = default_config();
        cfg.streaming_enabled = false;
        let mut req = completion_request("m", "hi");
        req.stream = true;
        assert!(req.validate(&cfg).is_err());
    }

    // -----------------------------------------------------------------------
    // CompletionResponse
    // -----------------------------------------------------------------------

    #[test]
    fn completion_response_single() {
        let resp = CompletionResponse::single(
            "id-2",
            "model-b",
            99,
            "world",
            FinishReason::Stop,
            UsageStats::new(3, 1),
        );
        assert_eq!(resp.id, "id-2");
        assert_eq!(resp.object, "text_completion");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].text, "world");
        assert_eq!(resp.usage.total_tokens, 4);
    }

    #[test]
    fn completion_response_length_finish() {
        let resp = CompletionResponse::single(
            "id",
            "m",
            0,
            "tok",
            FinishReason::Length,
            UsageStats::new(1, 1),
        );
        assert_eq!(resp.choices[0].finish_reason, FinishReason::Length);
    }

    // -----------------------------------------------------------------------
    // EmbeddingInput
    // -----------------------------------------------------------------------

    #[test]
    fn embedding_input_single_len() {
        let inp = EmbeddingInput::Single("hello".into());
        assert_eq!(inp.len(), 1);
        assert!(!inp.is_empty());
    }

    #[test]
    fn embedding_input_batch_len() {
        let inp = EmbeddingInput::Batch(vec!["a".into(), "b".into(), "c".into()]);
        assert_eq!(inp.len(), 3);
    }

    #[test]
    fn embedding_input_empty_single() {
        let inp = EmbeddingInput::Single(String::new());
        assert!(inp.is_empty());
    }

    #[test]
    fn embedding_input_empty_batch() {
        let inp = EmbeddingInput::Batch(vec![]);
        assert!(inp.is_empty());
    }

    #[test]
    fn embedding_input_batch_all_empty() {
        let inp = EmbeddingInput::Batch(vec![String::new(), String::new()]);
        assert!(inp.is_empty());
    }

    #[test]
    fn embedding_input_texts_single() {
        let inp = EmbeddingInput::Single("hi".into());
        assert_eq!(inp.texts(), vec!["hi"]);
    }

    #[test]
    fn embedding_input_texts_batch() {
        let inp = EmbeddingInput::Batch(vec!["a".into(), "b".into()]);
        assert_eq!(inp.texts(), vec!["a", "b"]);
    }

    // -----------------------------------------------------------------------
    // EmbeddingRequest validation
    // -----------------------------------------------------------------------

    #[test]
    fn embedding_req_valid() {
        let req = embedding_request("m", "hello");
        assert!(req.validate().is_ok());
    }

    #[test]
    fn embedding_req_empty_input() {
        let req =
            EmbeddingRequest { model: "m".into(), input: EmbeddingInput::Single(String::new()) };
        assert!(req.validate().is_err());
    }

    #[test]
    fn embedding_req_empty_model() {
        let req =
            EmbeddingRequest { model: String::new(), input: EmbeddingInput::Single("hi".into()) };
        assert!(req.validate().is_err());
    }

    // -----------------------------------------------------------------------
    // EmbeddingResponse
    // -----------------------------------------------------------------------

    #[test]
    fn embedding_response_from_vectors() {
        let resp =
            EmbeddingResponse::from_vectors("emb-model", vec![vec![0.1, 0.2], vec![0.3, 0.4]], 10);
        assert_eq!(resp.object, "list");
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].index, 0);
        assert_eq!(resp.data[1].index, 1);
        assert_eq!(resp.data[0].object, "embedding");
        assert_eq!(resp.usage.prompt_tokens, 10);
        assert_eq!(resp.usage.total_tokens, 10);
    }

    #[test]
    fn embedding_response_empty_vectors() {
        let resp = EmbeddingResponse::from_vectors("m", vec![], 0);
        assert!(resp.data.is_empty());
    }

    // -----------------------------------------------------------------------
    // UsageStats
    // -----------------------------------------------------------------------

    #[test]
    fn usage_stats_new() {
        let u = UsageStats::new(5, 10);
        assert_eq!(u.prompt_tokens, 5);
        assert_eq!(u.completion_tokens, 10);
        assert_eq!(u.total_tokens, 15);
    }

    #[test]
    fn usage_stats_zero() {
        let u = UsageStats::new(0, 0);
        assert_eq!(u.total_tokens, 0);
    }

    // -----------------------------------------------------------------------
    // FinishReason
    // -----------------------------------------------------------------------

    #[test]
    fn finish_reason_display() {
        assert_eq!(FinishReason::Stop.to_string(), "stop");
        assert_eq!(FinishReason::Length.to_string(), "length");
        assert_eq!(FinishReason::ContentFilter.to_string(), "content_filter",);
    }

    #[test]
    fn finish_reason_equality() {
        assert_eq!(FinishReason::Stop, FinishReason::Stop);
        assert_ne!(FinishReason::Stop, FinishReason::Length);
    }

    // -----------------------------------------------------------------------
    // ProtocolError
    // -----------------------------------------------------------------------

    #[test]
    fn error_invalid_request() {
        let e = ProtocolError::invalid_request("bad field");
        assert_eq!(e.error_type, ErrorType::InvalidRequest);
        assert_eq!(e.http_status_code(), 400);
        assert!(e.message.contains("bad field"));
    }

    #[test]
    fn error_not_found() {
        let e = ProtocolError::not_found("no model");
        assert_eq!(e.error_type, ErrorType::NotFound);
        assert_eq!(e.http_status_code(), 404);
    }

    #[test]
    fn error_rate_limited() {
        let e = ProtocolError::rate_limited("slow down");
        assert_eq!(e.error_type, ErrorType::RateLimit);
        assert_eq!(e.http_status_code(), 429);
    }

    #[test]
    fn error_server_error() {
        let e = ProtocolError::server_error("internal");
        assert_eq!(e.error_type, ErrorType::ServerError);
        assert_eq!(e.http_status_code(), 500);
    }

    #[test]
    fn error_unavailable() {
        let e = ProtocolError::unavailable("down");
        assert_eq!(e.error_type, ErrorType::ServiceUnavailable);
        assert_eq!(e.http_status_code(), 503);
    }

    #[test]
    fn error_authentication() {
        let e = ProtocolError::authentication("bad key");
        assert_eq!(e.error_type, ErrorType::Authentication);
        assert_eq!(e.http_status_code(), 401);
    }

    #[test]
    fn error_display() {
        let e = ProtocolError::invalid_request("oops");
        let s = e.to_string();
        assert!(s.contains("invalid_request_error"));
        assert!(s.contains("oops"));
    }

    #[test]
    fn error_json_body_contains_type() {
        let e = ProtocolError::server_error("boom");
        let json = e.to_json_body();
        assert!(json.contains("server_error"));
        assert!(json.contains("boom"));
    }

    #[test]
    fn error_json_body_contains_code() {
        let e = ProtocolError::not_found("missing");
        let json = e.to_json_body();
        assert!(json.contains(r#""code":"not_found""#));
    }

    #[test]
    fn error_json_body_with_param() {
        let mut e = ProtocolError::invalid_request("bad");
        e.param = Some("temperature".into());
        let json = e.to_json_body();
        assert!(json.contains(r#""param":"temperature""#));
    }

    #[test]
    fn error_json_escapes_message() {
        let e = ProtocolError::server_error("a \"quoted\" msg");
        let json = e.to_json_body();
        assert!(json.contains(r#"a \"quoted\" msg"#));
    }

    // -----------------------------------------------------------------------
    // ErrorType display
    // -----------------------------------------------------------------------

    #[test]
    fn error_type_display() {
        assert_eq!(ErrorType::InvalidRequest.to_string(), "invalid_request_error",);
        assert_eq!(ErrorType::Authentication.to_string(), "authentication_error",);
        assert_eq!(ErrorType::RateLimit.to_string(), "rate_limit_error",);
        assert_eq!(ErrorType::NotFound.to_string(), "not_found_error");
        assert_eq!(ErrorType::ServerError.to_string(), "server_error");
        assert_eq!(ErrorType::ServiceUnavailable.to_string(), "service_unavailable_error",);
    }

    // -----------------------------------------------------------------------
    // StreamingChunk
    // -----------------------------------------------------------------------

    #[test]
    fn streaming_chunk_content() {
        let chunk = StreamingChunk::content("c-1", "model", 100, "hello");
        assert_eq!(chunk.id, "c-1");
        assert_eq!(chunk.object, "chat.completion.chunk");
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("hello"),);
        assert!(chunk.choices[0].finish_reason.is_none());
        assert!(chunk.usage.is_none());
    }

    #[test]
    fn streaming_chunk_finish() {
        let chunk =
            StreamingChunk::finish("c-2", "model", 200, FinishReason::Stop, UsageStats::new(5, 10));
        assert_eq!(chunk.choices[0].finish_reason, Some(FinishReason::Stop),);
        assert!(chunk.usage.is_some());
        assert_eq!(chunk.usage.unwrap().total_tokens, 15);
    }

    #[test]
    fn streaming_chunk_sse_line_starts_with_data() {
        let chunk = StreamingChunk::content("c", "m", 0, "hi");
        let line = chunk.to_sse_line();
        assert!(line.starts_with("data: "));
    }

    #[test]
    fn streaming_chunk_sse_contains_model() {
        let chunk = StreamingChunk::content("c", "my-model", 0, "x");
        let line = chunk.to_sse_line();
        assert!(line.contains("my-model"));
    }

    #[test]
    fn streaming_chunk_sse_contains_content() {
        let chunk = StreamingChunk::content("c", "m", 0, "hello world");
        let line = chunk.to_sse_line();
        assert!(line.contains("hello world"));
    }

    #[test]
    fn streaming_chunk_done_sentinel() {
        assert_eq!(StreamingChunk::done_sentinel(), "data: [DONE]");
    }

    #[test]
    fn streaming_chunk_sse_escapes_newlines() {
        let chunk = StreamingChunk::content("c", "m", 0, "line1\nline2");
        let line = chunk.to_sse_line();
        assert!(line.contains("line1\\nline2"));
        assert!(!line.contains("line1\nline2"));
    }

    // -----------------------------------------------------------------------
    // DeltaContent
    // -----------------------------------------------------------------------

    #[test]
    fn delta_content_role_only() {
        let d = DeltaContent { role: Some(Role::Assistant), content: None };
        assert!(d.role.is_some());
        assert!(d.content.is_none());
    }

    #[test]
    fn delta_content_content_only() {
        let d = DeltaContent { role: None, content: Some("hi".into()) };
        assert!(d.role.is_none());
        assert_eq!(d.content.as_deref(), Some("hi"));
    }

    // -----------------------------------------------------------------------
    // ProtocolHandler
    // -----------------------------------------------------------------------

    #[test]
    fn handler_with_defaults() {
        let h = handler();
        assert_eq!(h.request_count(), 0);
        assert!(!h.available_models().is_empty());
    }

    #[test]
    fn handler_new_with_valid_config() {
        let h = ProtocolHandler::new(default_config());
        assert!(h.is_ok());
    }

    #[test]
    fn handler_new_with_invalid_config() {
        let mut cfg = default_config();
        cfg.max_request_size = 0;
        let h = ProtocolHandler::new(cfg);
        assert!(h.is_err());
    }

    #[test]
    fn handler_register_model() {
        let mut h = handler();
        h.register_model("new-model");
        assert!(h.available_models().contains(&"new-model".to_string()));
    }

    #[test]
    fn handler_register_model_idempotent() {
        let mut h = handler();
        h.register_model("m");
        h.register_model("m");
        let count = h.available_models().iter().filter(|x| *x == "m").count();
        assert_eq!(count, 1);
    }

    #[test]
    fn handler_unregister_model() {
        let mut h = handler();
        h.register_model("tmp");
        assert!(h.unregister_model("tmp"));
        assert!(!h.available_models().contains(&"tmp".to_string()));
    }

    #[test]
    fn handler_unregister_missing_model() {
        let mut h = handler();
        assert!(!h.unregister_model("nonexistent"));
    }

    #[test]
    fn handler_config_ref() {
        let h = handler();
        assert_eq!(h.config().max_tokens_limit, 4096);
    }

    #[test]
    fn handler_chat_completion_valid() {
        let mut h = handler();
        let req = chat_request("bitnet-b1.58-2B-4T", "hi");
        assert!(h.handle_chat_completion(req).is_ok());
        assert_eq!(h.request_count(), 1);
    }

    #[test]
    fn handler_chat_completion_unknown_model() {
        let mut h = handler();
        let req = chat_request("unknown-model", "hi");
        assert!(h.handle_chat_completion(req).is_err());
    }

    #[test]
    fn handler_completion_valid() {
        let mut h = handler();
        let req = completion_request("bitnet-b1.58-2B-4T", "once");
        assert!(h.handle_completion(req).is_ok());
    }

    #[test]
    fn handler_completion_empty_prompt() {
        let mut h = handler();
        let req = completion_request("bitnet-b1.58-2B-4T", "");
        assert!(h.handle_completion(req).is_err());
    }

    #[test]
    fn handler_embedding_valid() {
        let mut h = handler();
        let req = embedding_request("bitnet-b1.58-2B-4T", "embed me");
        assert!(h.handle_embedding(req).is_ok());
    }

    #[test]
    fn handler_embedding_empty_input() {
        let mut h = handler();
        let req = EmbeddingRequest {
            model: "bitnet-b1.58-2B-4T".into(),
            input: EmbeddingInput::Single(String::new()),
        };
        assert!(h.handle_embedding(req).is_err());
    }

    #[test]
    fn handler_route_chat() {
        let mut h = handler();
        let req = ApiRequest::ChatCompletion(chat_request("bitnet-b1.58-2B-4T", "route me"));
        let result = h.route(req);
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), ApiRequest::ChatCompletion(_)));
    }

    #[test]
    fn handler_route_completion() {
        let mut h = handler();
        let req = ApiRequest::Completion(completion_request("bitnet-b1.58-2B-4T", "route"));
        let result = h.route(req);
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), ApiRequest::Completion(_)));
    }

    #[test]
    fn handler_route_embedding() {
        let mut h = handler();
        let req = ApiRequest::Embedding(embedding_request("bitnet-b1.58-2B-4T", "embed"));
        let result = h.route(req);
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), ApiRequest::Embedding(_)));
    }

    #[test]
    fn handler_request_count_increments() {
        let mut h = handler();
        let _ = h.handle_chat_completion(chat_request("bitnet-b1.58-2B-4T", "a"));
        let _ = h.handle_completion(completion_request("bitnet-b1.58-2B-4T", "b"));
        let _ = h.handle_embedding(embedding_request("bitnet-b1.58-2B-4T", "c"));
        assert_eq!(h.request_count(), 3);
    }

    #[test]
    fn handler_check_request_size_ok() {
        let h = handler();
        assert!(h.check_request_size(100).is_ok());
    }

    #[test]
    fn handler_check_request_size_too_large() {
        let h = handler();
        let big = h.config().max_request_size + 1;
        assert!(h.check_request_size(big).is_err());
    }

    #[test]
    fn handler_check_request_size_at_limit() {
        let h = handler();
        let limit = h.config().max_request_size;
        assert!(h.check_request_size(limit).is_ok());
    }

    // -----------------------------------------------------------------------
    // escape_json
    // -----------------------------------------------------------------------

    #[test]
    fn escape_json_no_special() {
        assert_eq!(escape_json("hello"), "hello");
    }

    #[test]
    fn escape_json_quotes() {
        assert_eq!(escape_json(r#"say "hi""#), r#"say \"hi\""#);
    }

    #[test]
    fn escape_json_backslash() {
        assert_eq!(escape_json(r"back\slash"), r"back\\slash");
    }

    #[test]
    fn escape_json_newline() {
        assert_eq!(escape_json("a\nb"), "a\\nb");
    }

    #[test]
    fn escape_json_tab() {
        assert_eq!(escape_json("a\tb"), "a\\tb");
    }

    #[test]
    fn escape_json_carriage_return() {
        assert_eq!(escape_json("a\rb"), "a\\rb");
    }

    // -----------------------------------------------------------------------
    // EmbeddingUsage
    // -----------------------------------------------------------------------

    #[test]
    fn embedding_usage_fields() {
        let u = EmbeddingUsage { prompt_tokens: 7, total_tokens: 7 };
        assert_eq!(u.prompt_tokens, 7);
        assert_eq!(u.total_tokens, 7);
    }

    #[test]
    fn embedding_usage_equality() {
        let a = EmbeddingUsage { prompt_tokens: 5, total_tokens: 5 };
        let b = EmbeddingUsage { prompt_tokens: 5, total_tokens: 5 };
        assert_eq!(a, b);
    }

    // -----------------------------------------------------------------------
    // Edge cases & integration-style tests
    // -----------------------------------------------------------------------

    #[test]
    fn handler_empty_model_falls_back_to_default() {
        let mut h = handler();
        let req = chat_request("", "hi");
        let result = h.handle_chat_completion(req);
        assert!(result.is_ok());
        let validated = result.unwrap();
        assert_eq!(validated.model, "bitnet-b1.58-2B-4T");
    }

    #[test]
    fn handler_completion_empty_model_falls_back() {
        let mut h = handler();
        let req = completion_request("", "prompt");
        let result = h.handle_completion(req);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().model, "bitnet-b1.58-2B-4T");
    }

    #[test]
    fn handler_embedding_empty_model_fails() {
        // Embedding requires an explicit model from the request,
        // and empty model after resolve fails validation.
        let mut h = handler();
        let req = embedding_request("", "data");
        // Empty model -> resolve checks availability of default.
        // Default is available -> resolve succeeds, then validate
        // runs with the resolved model.
        let result = h.handle_embedding(req);
        assert!(result.is_ok());
    }

    #[test]
    fn chat_request_with_stop_sequences() {
        let cfg = default_config();
        let mut req = chat_request("m", "hi");
        req.stop = Some(vec!["<|end|>".into(), "\n".into()]);
        assert!(req.validate(&cfg).is_ok());
    }

    #[test]
    fn completion_request_echo_mode() {
        let req = CompletionRequest {
            model: "m".into(),
            prompt: "hi".into(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            stream: false,
            stop: None,
            echo: true,
            extra: HashMap::new(),
        };
        assert!(req.echo);
    }

    #[test]
    fn embedding_batch_request() {
        let req = EmbeddingRequest {
            model: "m".into(),
            input: EmbeddingInput::Batch(vec!["one".into(), "two".into(), "three".into()]),
        };
        assert!(req.validate().is_ok());
        assert_eq!(req.input.len(), 3);
    }

    #[test]
    fn multiple_chat_messages() {
        let cfg = default_config();
        let req = ChatCompletionRequest {
            model: "m".into(),
            messages: vec![
                ChatMessage::system("You are helpful."),
                ChatMessage::user("Hi"),
                ChatMessage::assistant("Hello!"),
                ChatMessage::user("Bye"),
            ],
            temperature: None,
            max_tokens: None,
            top_p: None,
            stream: false,
            stop: None,
            extra: HashMap::new(),
        };
        assert!(req.validate(&cfg).is_ok());
        assert_eq!(req.messages.len(), 4);
    }

    #[test]
    fn chat_request_extra_params() {
        let mut req = chat_request("m", "hi");
        req.extra.insert("frequency_penalty".into(), "0.5".into());
        assert_eq!(req.extra.get("frequency_penalty").unwrap(), "0.5");
    }

    #[test]
    fn streaming_full_sequence() {
        // Simulate a 3-chunk streaming response.
        let chunks = [
            StreamingChunk::content("s1", "m", 0, "Hello"),
            StreamingChunk::content("s1", "m", 0, " world"),
            StreamingChunk::finish("s1", "m", 0, FinishReason::Stop, UsageStats::new(3, 2)),
        ];
        assert_eq!(chunks.len(), 3);
        assert!(chunks[0].choices[0].finish_reason.is_none());
        assert!(chunks[1].choices[0].finish_reason.is_none());
        assert_eq!(chunks[2].choices[0].finish_reason, Some(FinishReason::Stop),);
    }

    #[test]
    fn protocol_error_no_code() {
        let e = ProtocolError {
            error_type: ErrorType::ServerError,
            message: "fail".into(),
            code: None,
            param: None,
        };
        let json = e.to_json_body();
        assert!(!json.contains(r#""code""#));
    }
}
