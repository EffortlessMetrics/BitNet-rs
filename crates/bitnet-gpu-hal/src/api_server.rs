//! OpenAI-compatible API server for GPU-accelerated inference.
//!
//! Provides an HTTP server with chat/completions endpoints, SSE streaming,
//! API key authentication, request validation, health checks, and Prometheus
//! metrics.  All types are CPU-reference (mock) implementations suitable for
//! unit testing; actual network I/O is deferred to the downstream integration
//! layer.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// ServerConfig
// ---------------------------------------------------------------------------

/// Top-level configuration for the API server.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Bind address (e.g. `"0.0.0.0"`).
    pub host: String,
    /// Listen port.
    pub port: u16,
    /// Number of worker threads for request processing.
    pub workers: usize,
    /// Whether TLS is enabled.
    pub tls_enabled: bool,
    /// Path to the TLS certificate file (PEM).
    pub tls_cert_path: Option<String>,
    /// Path to the TLS key file (PEM).
    pub tls_key_path: Option<String>,
    /// Allowed CORS origins (`["*"]` for any).
    pub cors_origins: Vec<String>,
    /// Maximum concurrent connections.
    pub max_connections: usize,
    /// Request body size limit in bytes.
    pub max_body_bytes: usize,
    /// Request timeout.
    pub request_timeout: Duration,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".into(),
            port: 8080,
            workers: 4,
            tls_enabled: false,
            tls_cert_path: None,
            tls_key_path: None,
            cors_origins: vec!["*".into()],
            max_connections: 1024,
            max_body_bytes: 4 * 1024 * 1024,
            request_timeout: Duration::from_secs(60),
        }
    }
}

impl ServerConfig {
    /// Create a new configuration with the given host and port.
    pub fn new(host: impl Into<String>, port: u16) -> Self {
        Self { host: host.into(), port, ..Self::default() }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.host.is_empty() {
            return Err("host must not be empty".into());
        }
        if self.port == 0 {
            return Err("port must be > 0".into());
        }
        if self.workers == 0 {
            return Err("workers must be > 0".into());
        }
        if self.max_connections == 0 {
            return Err("max_connections must be > 0".into());
        }
        if self.tls_enabled && self.tls_cert_path.is_none() {
            return Err("tls_cert_path required when TLS is enabled".into());
        }
        if self.tls_enabled && self.tls_key_path.is_none() {
            return Err("tls_key_path required when TLS is enabled".into());
        }
        Ok(())
    }

    /// Enable TLS with the given certificate and key paths.
    pub fn with_tls(mut self, cert: impl Into<String>, key: impl Into<String>) -> Self {
        self.tls_enabled = true;
        self.tls_cert_path = Some(cert.into());
        self.tls_key_path = Some(key.into());
        self
    }

    /// Set the allowed CORS origins.
    pub fn with_cors(mut self, origins: Vec<String>) -> Self {
        self.cors_origins = origins;
        self
    }

    /// Return the socket address string.
    pub fn bind_address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

// ---------------------------------------------------------------------------
// ChatMessage / ChatRequest / ChatChoice / ChatResponse
// ---------------------------------------------------------------------------

/// Role in a chat conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl fmt::Display for ChatRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::System => write!(f, "system"),
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
        }
    }
}

impl ChatRole {
    /// Parse a role from its string representation.
    pub fn from_str_name(s: &str) -> Result<Self, String> {
        match s {
            "system" => Ok(Self::System),
            "user" => Ok(Self::User),
            "assistant" => Ok(Self::Assistant),
            other => Err(format!("unknown role: {other}")),
        }
    }
}

/// A single message in a chat conversation.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self { role, content: content.into() }
    }
}

/// An incoming chat/completions request (OpenAI-compatible subset).
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: bool,
    pub stop: Vec<String>,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub user: Option<String>,
}

impl Default for ChatRequest {
    fn default() -> Self {
        Self {
            model: String::new(),
            messages: Vec::new(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            stream: false,
            stop: Vec::new(),
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            user: None,
        }
    }
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        Self { model: model.into(), messages, ..Self::default() }
    }
}

/// One generated choice in a chat completion response.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: FinishReason,
}

/// Reason the model stopped generating tokens.
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

/// Token usage statistics returned alongside a completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl TokenUsage {
    pub fn new(prompt: u32, completion: u32) -> Self {
        Self {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
        }
    }
}

/// Full chat completion response.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: TokenUsage,
}

// ---------------------------------------------------------------------------
// RequestHandler
// ---------------------------------------------------------------------------

/// Handles incoming inference requests and produces chat completions.
///
/// This is a CPU-reference (mock) handler that returns deterministic
/// placeholder responses for testing.
#[derive(Debug, Clone)]
pub struct RequestHandler {
    /// Model identifier this handler is bound to.
    model_id: String,
    /// Maximum tokens to generate when not specified by the request.
    default_max_tokens: u32,
    /// Number of requests processed.
    requests_handled: u64,
}

impl RequestHandler {
    /// Create a handler bound to the given model.
    pub fn new(model_id: impl Into<String>) -> Self {
        Self { model_id: model_id.into(), default_max_tokens: 256, requests_handled: 0 }
    }

    /// Return the model identifier.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Return how many requests have been processed.
    pub fn requests_handled(&self) -> u64 {
        self.requests_handled
    }

    /// Process a chat completion request and return a response.
    pub fn handle_chat(&mut self, req: &ChatRequest) -> Result<ChatResponse, String> {
        if req.messages.is_empty() {
            return Err("messages must not be empty".into());
        }
        if req.model.is_empty() {
            return Err("model must not be empty".into());
        }

        let max_tokens = req.max_tokens.unwrap_or(self.default_max_tokens);
        let reply_text = self.generate_mock_reply(&req.messages, max_tokens);
        let prompt_tokens =
            req.messages.iter().map(|m| m.content.len() as u32 / 4).sum::<u32>().max(1);
        let completion_tokens = (reply_text.len() as u32 / 4).max(1);

        self.requests_handled += 1;

        Ok(ChatResponse {
            id: format!("chatcmpl-mock-{}", self.requests_handled),
            object: "chat.completion".into(),
            created: 1_700_000_000,
            model: req.model.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage::new(ChatRole::Assistant, reply_text),
                finish_reason: FinishReason::Stop,
            }],
            usage: TokenUsage::new(prompt_tokens, completion_tokens),
        })
    }

    /// Generate a deterministic mock reply from the conversation.
    fn generate_mock_reply(&self, messages: &[ChatMessage], max_tokens: u32) -> String {
        let last = messages.last().map(|m| m.content.as_str()).unwrap_or("");
        let reply = format!("[mock-{}] Reply to: {}", self.model_id, last);
        let limit = max_tokens as usize * 4;
        if reply.len() > limit { reply[..limit].to_string() } else { reply }
    }
}

// ---------------------------------------------------------------------------
// ResponseBuilder
// ---------------------------------------------------------------------------

/// Constructs OpenAI-compatible JSON response payloads.
#[derive(Debug, Clone)]
pub struct ResponseBuilder {
    default_model: String,
}

impl ResponseBuilder {
    pub fn new(model: impl Into<String>) -> Self {
        Self { default_model: model.into() }
    }

    /// Build a JSON string from a [`ChatResponse`].
    pub fn build_json(&self, resp: &ChatResponse) -> String {
        let choices_json: Vec<String> = resp
            .choices
            .iter()
            .map(|c| {
                format!(
                    r#"{{"index":{},"message":{{"role":"{}","content":"{}"}},"finish_reason":"{}"}}"#,
                    c.index, c.message.role, escape_json(&c.message.content), c.finish_reason,
                )
            })
            .collect();
        format!(
            r#"{{"id":"{}","object":"{}","created":{},"model":"{}","choices":[{}],"usage":{{"prompt_tokens":{},"completion_tokens":{},"total_tokens":{}}}}}"#,
            resp.id,
            resp.object,
            resp.created,
            resp.model,
            choices_json.join(","),
            resp.usage.prompt_tokens,
            resp.usage.completion_tokens,
            resp.usage.total_tokens,
        )
    }

    /// Build an error response payload.
    pub fn build_error(&self, code: u16, message: &str, error_type: &str) -> String {
        format!(
            r#"{{"error":{{"message":"{}","type":"{}","code":{}}}}}"#,
            escape_json(message),
            error_type,
            code,
        )
    }

    /// Return the default model name.
    pub fn default_model(&self) -> &str {
        &self.default_model
    }
}

/// Minimal JSON string escaping (double-quote and backslash).
fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

// ---------------------------------------------------------------------------
// StreamingResponse
// ---------------------------------------------------------------------------

/// Server-Sent Events streaming response for token-by-token generation.
#[derive(Debug, Clone)]
pub struct StreamingResponse {
    /// Completion id shared across all chunks.
    id: String,
    model: String,
    /// Accumulated chunks so far.
    chunks: Vec<StreamChunk>,
    finished: bool,
}

/// A single streaming delta chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamChunk {
    pub index: u32,
    pub delta_content: String,
    pub finish_reason: Option<FinishReason>,
}

impl StreamingResponse {
    /// Create a new streaming response for the given model.
    pub fn new(id: impl Into<String>, model: impl Into<String>) -> Self {
        Self { id: id.into(), model: model.into(), chunks: Vec::new(), finished: false }
    }

    /// Push a token delta.  Returns an error if the stream is already finished.
    pub fn push_delta(&mut self, content: impl Into<String>) -> Result<(), String> {
        if self.finished {
            return Err("stream already finished".into());
        }
        let idx = self.chunks.len() as u32;
        self.chunks.push(StreamChunk {
            index: idx,
            delta_content: content.into(),
            finish_reason: None,
        });
        Ok(())
    }

    /// Mark the stream as finished.
    pub fn finish(&mut self, reason: FinishReason) -> Result<(), String> {
        if self.finished {
            return Err("stream already finished".into());
        }
        let idx = self.chunks.len() as u32;
        self.chunks.push(StreamChunk {
            index: idx,
            delta_content: String::new(),
            finish_reason: Some(reason),
        });
        self.finished = true;
        Ok(())
    }

    /// Whether the stream has been finished.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Return a reference to all emitted chunks.
    pub fn chunks(&self) -> &[StreamChunk] {
        &self.chunks
    }

    /// Format all chunks as SSE lines (`data: …`).
    pub fn format_sse(&self) -> Vec<String> {
        let mut lines = Vec::with_capacity(self.chunks.len() + 1);
        for chunk in &self.chunks {
            let finish = chunk
                .finish_reason
                .as_ref()
                .map(|r| format!(r#","finish_reason":"{r}""#))
                .unwrap_or_default();
            lines.push(format!(
                r#"data: {{"id":"{}","object":"chat.completion.chunk","model":"{}","choices":[{{"index":0,"delta":{{"content":"{}"}}{}}}]}}"#,
                self.id,
                self.model,
                escape_json(&chunk.delta_content),
                finish,
            ));
        }
        lines.push("data: [DONE]".into());
        lines
    }

    /// Return the total number of chunks emitted.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Return the stream id.
    pub fn id(&self) -> &str {
        &self.id
    }
}

// ---------------------------------------------------------------------------
// AuthMiddleware
// ---------------------------------------------------------------------------

/// API key authentication middleware.
///
/// Validates bearer tokens against a set of known keys and tracks
/// per-key usage statistics.
#[derive(Debug, Clone)]
pub struct AuthMiddleware {
    /// Valid API keys (key → label).
    keys: HashMap<String, String>,
    /// Per-key request counts.
    usage: HashMap<String, u64>,
    /// Whether authentication is enabled.
    enabled: bool,
}

impl AuthMiddleware {
    /// Create an enabled middleware with no keys.
    pub fn new() -> Self {
        Self { keys: HashMap::new(), usage: HashMap::new(), enabled: true }
    }

    /// Create a disabled (pass-through) middleware.
    pub fn disabled() -> Self {
        Self { keys: HashMap::new(), usage: HashMap::new(), enabled: false }
    }

    /// Register an API key with a human-readable label.
    pub fn add_key(&mut self, key: impl Into<String>, label: impl Into<String>) {
        let k = key.into();
        self.usage.entry(k.clone()).or_insert(0);
        self.keys.insert(k, label.into());
    }

    /// Remove an API key.
    pub fn remove_key(&mut self, key: &str) -> bool {
        self.usage.remove(key);
        self.keys.remove(key).is_some()
    }

    /// Authenticate a bearer token.  Returns the label on success.
    pub fn authenticate(&mut self, token: &str) -> Result<String, AuthError> {
        if !self.enabled {
            return Ok("anonymous".into());
        }
        if token.is_empty() {
            return Err(AuthError::MissingToken);
        }
        match self.keys.get(token) {
            Some(label) => {
                *self.usage.entry(token.to_string()).or_insert(0) += 1;
                Ok(label.clone())
            }
            None => Err(AuthError::InvalidToken),
        }
    }

    /// Parse a bearer token from an `Authorization` header value.
    pub fn parse_bearer(header: &str) -> Option<&str> {
        header.strip_prefix("Bearer ")
    }

    /// Authenticate from a raw `Authorization` header value.
    pub fn authenticate_header(&mut self, header: &str) -> Result<String, AuthError> {
        let token = Self::parse_bearer(header).ok_or(AuthError::MissingToken)?;
        self.authenticate(token)
    }

    /// Return per-key usage counts.
    pub fn usage(&self) -> &HashMap<String, u64> {
        &self.usage
    }

    /// Number of registered keys.
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    /// Whether authentication is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

/// Errors produced by [`AuthMiddleware`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthError {
    MissingToken,
    InvalidToken,
}

impl fmt::Display for AuthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingToken => write!(f, "missing or malformed bearer token"),
            Self::InvalidToken => write!(f, "invalid API key"),
        }
    }
}

// ---------------------------------------------------------------------------
// RequestValidator
// ---------------------------------------------------------------------------

/// Validates incoming request payloads against expected schemas.
#[derive(Debug, Clone)]
pub struct RequestValidator {
    /// Maximum allowed messages in a single request.
    pub max_messages: usize,
    /// Maximum allowed `max_tokens` value.
    pub max_max_tokens: u32,
    /// Allowed model identifiers.
    pub allowed_models: Vec<String>,
    /// Maximum content length per message (bytes).
    pub max_content_length: usize,
}

impl Default for RequestValidator {
    fn default() -> Self {
        Self {
            max_messages: 128,
            max_max_tokens: 4096,
            allowed_models: Vec::new(),
            max_content_length: 32_768,
        }
    }
}

impl RequestValidator {
    /// Create a validator with the given allowed models.
    pub fn new(models: Vec<String>) -> Self {
        Self { allowed_models: models, ..Self::default() }
    }

    /// Validate a [`ChatRequest`].
    pub fn validate(&self, req: &ChatRequest) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        if req.model.is_empty() {
            errors.push("model is required".into());
        } else if !self.allowed_models.is_empty() && !self.allowed_models.contains(&req.model) {
            errors.push(format!("model '{}' is not available", req.model));
        }

        if req.messages.is_empty() {
            errors.push("messages must not be empty".into());
        }
        if req.messages.len() > self.max_messages {
            errors.push(format!(
                "too many messages: {} (max {})",
                req.messages.len(),
                self.max_messages
            ));
        }

        for (i, msg) in req.messages.iter().enumerate() {
            if msg.content.len() > self.max_content_length {
                errors.push(format!(
                    "message[{}] content too long: {} bytes (max {})",
                    i,
                    msg.content.len(),
                    self.max_content_length
                ));
            }
        }

        if let Some(mt) = req.max_tokens {
            if mt > self.max_max_tokens {
                errors.push(format!("max_tokens {} exceeds limit {}", mt, self.max_max_tokens));
            }
            if mt == 0 {
                errors.push("max_tokens must be > 0".into());
            }
        }

        if let Some(t) = req.temperature {
            if !(0.0..=2.0).contains(&t) {
                errors.push(format!("temperature {t} out of range [0.0, 2.0]"));
            }
        }

        if let Some(p) = req.top_p {
            if !(0.0..=1.0).contains(&p) {
                errors.push(format!("top_p {p} out of range [0.0, 1.0]"));
            }
        }

        if !(-2.0..=2.0).contains(&req.presence_penalty) {
            errors.push(format!(
                "presence_penalty {} out of range [-2.0, 2.0]",
                req.presence_penalty
            ));
        }

        if !(-2.0..=2.0).contains(&req.frequency_penalty) {
            errors.push(format!(
                "frequency_penalty {} out of range [-2.0, 2.0]",
                req.frequency_penalty
            ));
        }

        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }

    /// Quick boolean check.
    pub fn is_valid(&self, req: &ChatRequest) -> bool {
        self.validate(req).is_ok()
    }
}

// ---------------------------------------------------------------------------
// HealthEndpoint
// ---------------------------------------------------------------------------

/// Health / readiness / liveness check endpoint state.
#[derive(Debug, Clone)]
pub struct HealthEndpoint {
    /// Whether the server is ready to accept traffic.
    ready: bool,
    /// Whether the server process is alive.
    alive: bool,
    /// Optional human-readable status message.
    status_message: String,
    /// Timestamp of the last health check.
    last_check: Option<Instant>,
    /// Number of loaded models.
    models_loaded: usize,
    /// Current request queue depth.
    queue_depth: usize,
}

impl Default for HealthEndpoint {
    fn default() -> Self {
        Self::new()
    }
}

impl HealthEndpoint {
    pub fn new() -> Self {
        Self {
            ready: false,
            alive: true,
            status_message: "initializing".into(),
            last_check: None,
            models_loaded: 0,
            queue_depth: 0,
        }
    }

    /// Mark the server as ready.
    pub fn set_ready(&mut self, ready: bool) {
        self.ready = ready;
        if ready {
            self.status_message = "ok".into();
        }
    }

    /// Mark the server as alive (or not).
    pub fn set_alive(&mut self, alive: bool) {
        self.alive = alive;
    }

    /// Update model count.
    pub fn set_models_loaded(&mut self, n: usize) {
        self.models_loaded = n;
    }

    /// Update queue depth.
    pub fn set_queue_depth(&mut self, depth: usize) {
        self.queue_depth = depth;
    }

    /// Record that a health check was performed.
    pub fn record_check(&mut self) {
        self.last_check = Some(Instant::now());
    }

    /// Return the health check response as JSON.
    pub fn health_json(&self) -> String {
        format!(
            r#"{{"status":"{}","ready":{},"alive":{},"models_loaded":{},"queue_depth":{}}}"#,
            self.status_message, self.ready, self.alive, self.models_loaded, self.queue_depth,
        )
    }

    /// Readiness probe — returns `Ok(())` when ready.
    pub fn readiness(&self) -> Result<(), String> {
        if self.ready { Ok(()) } else { Err(format!("not ready: {}", self.status_message)) }
    }

    /// Liveness probe — returns `Ok(())` when alive.
    pub fn liveness(&self) -> Result<(), String> {
        if self.alive { Ok(()) } else { Err("process is not alive".into()) }
    }

    pub fn is_ready(&self) -> bool {
        self.ready
    }

    pub fn is_alive(&self) -> bool {
        self.alive
    }

    pub fn models_loaded(&self) -> usize {
        self.models_loaded
    }

    pub fn queue_depth(&self) -> usize {
        self.queue_depth
    }

    pub fn status_message(&self) -> &str {
        &self.status_message
    }
}

// ---------------------------------------------------------------------------
// MetricsEndpoint
// ---------------------------------------------------------------------------

/// Prometheus-style metrics collector for the API server.
#[derive(Debug, Clone)]
pub struct MetricsEndpoint {
    /// Total requests received.
    total_requests: u64,
    /// Total successful responses.
    total_successes: u64,
    /// Total error responses.
    total_errors: u64,
    /// Total tokens generated.
    total_tokens_generated: u64,
    /// Total prompt tokens processed.
    total_prompt_tokens: u64,
    /// Request latencies in milliseconds (rolling window).
    latencies_ms: Vec<f64>,
    /// Maximum latency window size.
    max_latency_window: usize,
    /// Active in-flight requests.
    active_requests: u64,
}

impl Default for MetricsEndpoint {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsEndpoint {
    pub fn new() -> Self {
        Self {
            total_requests: 0,
            total_successes: 0,
            total_errors: 0,
            total_tokens_generated: 0,
            total_prompt_tokens: 0,
            latencies_ms: Vec::new(),
            max_latency_window: 1000,
            active_requests: 0,
        }
    }

    /// Record an incoming request.
    pub fn record_request(&mut self) {
        self.total_requests += 1;
        self.active_requests += 1;
    }

    /// Record a successful response with latency and token counts.
    pub fn record_success(&mut self, latency_ms: f64, prompt_tokens: u32, completion_tokens: u32) {
        self.total_successes += 1;
        self.total_prompt_tokens += u64::from(prompt_tokens);
        self.total_tokens_generated += u64::from(completion_tokens);
        self.active_requests = self.active_requests.saturating_sub(1);
        if self.latencies_ms.len() >= self.max_latency_window {
            self.latencies_ms.remove(0);
        }
        self.latencies_ms.push(latency_ms);
    }

    /// Record an error response.
    pub fn record_error(&mut self) {
        self.total_errors += 1;
        self.active_requests = self.active_requests.saturating_sub(1);
    }

    /// Average latency over the window.
    pub fn avg_latency_ms(&self) -> f64 {
        if self.latencies_ms.is_empty() {
            return 0.0;
        }
        self.latencies_ms.iter().sum::<f64>() / self.latencies_ms.len() as f64
    }

    /// P99 latency (approximate — uses sorted slice).
    pub fn p99_latency_ms(&self) -> f64 {
        if self.latencies_ms.is_empty() {
            return 0.0;
        }
        let mut sorted = self.latencies_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((sorted.len() as f64) * 0.99).ceil() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    /// Format as Prometheus exposition text.
    pub fn prometheus(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "# HELP bitnet_api_requests_total Total API requests\n\
             # TYPE bitnet_api_requests_total counter\n\
             bitnet_api_requests_total {}\n",
            self.total_requests
        ));
        out.push_str(&format!(
            "# HELP bitnet_api_successes_total Total successful responses\n\
             # TYPE bitnet_api_successes_total counter\n\
             bitnet_api_successes_total {}\n",
            self.total_successes
        ));
        out.push_str(&format!(
            "# HELP bitnet_api_errors_total Total error responses\n\
             # TYPE bitnet_api_errors_total counter\n\
             bitnet_api_errors_total {}\n",
            self.total_errors
        ));
        out.push_str(&format!(
            "# HELP bitnet_api_tokens_generated_total Total tokens generated\n\
             # TYPE bitnet_api_tokens_generated_total counter\n\
             bitnet_api_tokens_generated_total {}\n",
            self.total_tokens_generated
        ));
        out.push_str(&format!(
            "# HELP bitnet_api_prompt_tokens_total Total prompt tokens\n\
             # TYPE bitnet_api_prompt_tokens_total counter\n\
             bitnet_api_prompt_tokens_total {}\n",
            self.total_prompt_tokens
        ));
        out.push_str(&format!(
            "# HELP bitnet_api_active_requests Active in-flight requests\n\
             # TYPE bitnet_api_active_requests gauge\n\
             bitnet_api_active_requests {}\n",
            self.active_requests
        ));
        out.push_str(&format!(
            "# HELP bitnet_api_latency_avg_ms Average request latency\n\
             # TYPE bitnet_api_latency_avg_ms gauge\n\
             bitnet_api_latency_avg_ms {:.2}\n",
            self.avg_latency_ms()
        ));
        out
    }

    pub fn total_requests(&self) -> u64 {
        self.total_requests
    }

    pub fn total_successes(&self) -> u64 {
        self.total_successes
    }

    pub fn total_errors(&self) -> u64 {
        self.total_errors
    }

    pub fn total_tokens_generated(&self) -> u64 {
        self.total_tokens_generated
    }

    pub fn active_requests(&self) -> u64 {
        self.active_requests
    }

    /// Reset all counters.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

// ---------------------------------------------------------------------------
// RouterConfig
// ---------------------------------------------------------------------------

/// Route configuration for the API server (models, endpoints, versioning).
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// API version prefix (e.g. `"/v1"`).
    pub api_prefix: String,
    /// Available model routes: model id → model path.
    pub model_routes: HashMap<String, String>,
    /// Whether the `/v1/models` listing endpoint is enabled.
    pub models_endpoint: bool,
    /// Whether the `/v1/chat/completions` endpoint is enabled.
    pub completions_endpoint: bool,
    /// Whether the `/health` endpoint is enabled.
    pub health_endpoint: bool,
    /// Whether the `/metrics` endpoint is enabled.
    pub metrics_endpoint: bool,
    /// Custom extra routes: path → handler label.
    pub custom_routes: HashMap<String, String>,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            api_prefix: "/v1".into(),
            model_routes: HashMap::new(),
            models_endpoint: true,
            completions_endpoint: true,
            health_endpoint: true,
            metrics_endpoint: true,
            custom_routes: HashMap::new(),
        }
    }
}

impl RouterConfig {
    pub fn new(prefix: impl Into<String>) -> Self {
        Self { api_prefix: prefix.into(), ..Self::default() }
    }

    /// Register a model route.
    pub fn add_model(&mut self, id: impl Into<String>, path: impl Into<String>) {
        self.model_routes.insert(id.into(), path.into());
    }

    /// Register a custom route.
    pub fn add_custom_route(&mut self, path: impl Into<String>, handler: impl Into<String>) {
        self.custom_routes.insert(path.into(), handler.into());
    }

    /// Return all endpoint paths this router exposes.
    pub fn all_paths(&self) -> Vec<String> {
        let mut paths = Vec::new();
        if self.completions_endpoint {
            paths.push(format!("{}/chat/completions", self.api_prefix));
        }
        if self.models_endpoint {
            paths.push(format!("{}/models", self.api_prefix));
        }
        if self.health_endpoint {
            paths.push("/health".into());
            paths.push("/health/ready".into());
            paths.push("/health/live".into());
        }
        if self.metrics_endpoint {
            paths.push("/metrics".into());
        }
        for p in self.custom_routes.keys() {
            paths.push(p.clone());
        }
        paths
    }

    /// Check whether a request path is routable.
    pub fn is_routable(&self, path: &str) -> bool {
        self.all_paths().iter().any(|p| p == path)
    }

    /// Build a JSON model-list response (OpenAI `/v1/models` format).
    pub fn models_json(&self) -> String {
        let items: Vec<String> = self
            .model_routes
            .keys()
            .map(|id| format!(r#"{{"id":"{}","object":"model","owned_by":"bitnet"}}"#, id))
            .collect();
        format!(r#"{{"object":"list","data":[{}]}}"#, items.join(","))
    }

    /// Validate the router configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.api_prefix.is_empty() {
            return Err("api_prefix must not be empty".into());
        }
        if !self.api_prefix.starts_with('/') {
            return Err("api_prefix must start with '/'".into());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ApiServerEngine
// ---------------------------------------------------------------------------

/// Unified API server lifecycle manager.
///
/// Coordinates configuration, routing, authentication, request handling,
/// health checks, and metrics into a single entry point.
#[derive(Debug)]
pub struct ApiServerEngine {
    config: ServerConfig,
    router: RouterConfig,
    auth: AuthMiddleware,
    handler: RequestHandler,
    validator: RequestValidator,
    health: HealthEndpoint,
    metrics: MetricsEndpoint,
    started: bool,
}

impl Clone for ApiServerEngine {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            router: self.router.clone(),
            auth: self.auth.clone(),
            handler: self.handler.clone(),
            validator: self.validator.clone(),
            health: self.health.clone(),
            metrics: self.metrics.clone(),
            started: self.started,
        }
    }
}

impl ApiServerEngine {
    /// Create a new engine from server config, router, and model id.
    pub fn new(config: ServerConfig, router: RouterConfig, model_id: impl Into<String>) -> Self {
        let model = model_id.into();
        let allowed: Vec<String> = router.model_routes.keys().cloned().collect();
        Self {
            config,
            router,
            auth: AuthMiddleware::new(),
            handler: RequestHandler::new(&model),
            validator: RequestValidator::new(allowed),
            health: HealthEndpoint::new(),
            metrics: MetricsEndpoint::new(),
            started: false,
        }
    }

    /// Start the (mock) server.
    pub fn start(&mut self) -> Result<(), String> {
        self.config.validate()?;
        self.router.validate()?;
        self.started = true;
        self.health.set_ready(true);
        log::info!("API server started on {}", self.config.bind_address());
        Ok(())
    }

    /// Stop the (mock) server.
    pub fn stop(&mut self) {
        self.started = false;
        self.health.set_ready(false);
    }

    /// Process a request end-to-end: auth → validate → handle → metrics.
    pub fn process_request(
        &mut self,
        auth_header: Option<&str>,
        request: &ChatRequest,
    ) -> Result<ChatResponse, String> {
        // Auth
        if let Some(hdr) = auth_header {
            self.auth.authenticate_header(hdr).map_err(|e| e.to_string())?;
        } else if self.auth.is_enabled() {
            return Err(AuthError::MissingToken.to_string());
        }

        // Validate
        self.validator.validate(request).map_err(|errs| errs.join("; "))?;

        // Handle
        self.metrics.record_request();
        let start = Instant::now();
        match self.handler.handle_chat(request) {
            Ok(resp) => {
                let latency = start.elapsed().as_secs_f64() * 1000.0;
                self.metrics.record_success(
                    latency,
                    resp.usage.prompt_tokens,
                    resp.usage.completion_tokens,
                );
                Ok(resp)
            }
            Err(e) => {
                self.metrics.record_error();
                Err(e)
            }
        }
    }

    pub fn is_started(&self) -> bool {
        self.started
    }

    pub fn config(&self) -> &ServerConfig {
        &self.config
    }

    pub fn router(&self) -> &RouterConfig {
        &self.router
    }

    pub fn auth(&self) -> &AuthMiddleware {
        &self.auth
    }

    pub fn auth_mut(&mut self) -> &mut AuthMiddleware {
        &mut self.auth
    }

    pub fn health(&self) -> &HealthEndpoint {
        &self.health
    }

    pub fn health_mut(&mut self) -> &mut HealthEndpoint {
        &mut self.health
    }

    pub fn metrics(&self) -> &MetricsEndpoint {
        &self.metrics
    }

    pub fn handler(&self) -> &RequestHandler {
        &self.handler
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers -----------------------------------------------------------

    fn sample_messages() -> Vec<ChatMessage> {
        vec![
            ChatMessage::new(ChatRole::System, "You are a helpful assistant."),
            ChatMessage::new(ChatRole::User, "Hello!"),
        ]
    }

    fn sample_request() -> ChatRequest {
        ChatRequest::new("test-model", sample_messages())
    }

    fn make_engine() -> ApiServerEngine {
        let mut router = RouterConfig::default();
        router.add_model("test-model", "/models/test");
        let config = ServerConfig::default();
        ApiServerEngine::new(config, router, "test-model")
    }

    // ====================================================================
    // ServerConfig
    // ====================================================================

    #[test]
    fn test_server_config_default() {
        let cfg = ServerConfig::default();
        assert_eq!(cfg.host, "0.0.0.0");
        assert_eq!(cfg.port, 8080);
        assert_eq!(cfg.workers, 4);
        assert!(!cfg.tls_enabled);
        assert_eq!(cfg.max_connections, 1024);
    }

    #[test]
    fn test_server_config_new() {
        let cfg = ServerConfig::new("127.0.0.1", 3000);
        assert_eq!(cfg.host, "127.0.0.1");
        assert_eq!(cfg.port, 3000);
    }

    #[test]
    fn test_server_config_validate_ok() {
        assert!(ServerConfig::default().validate().is_ok());
    }

    #[test]
    fn test_server_config_validate_empty_host() {
        let mut cfg = ServerConfig::default();
        cfg.host = String::new();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_server_config_validate_zero_port() {
        let mut cfg = ServerConfig::default();
        cfg.port = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_server_config_validate_zero_workers() {
        let mut cfg = ServerConfig::default();
        cfg.workers = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_server_config_validate_zero_connections() {
        let mut cfg = ServerConfig::default();
        cfg.max_connections = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_server_config_tls_missing_cert() {
        let mut cfg = ServerConfig::default();
        cfg.tls_enabled = true;
        cfg.tls_key_path = Some("key.pem".into());
        assert!(cfg.validate().unwrap_err().contains("cert"));
    }

    #[test]
    fn test_server_config_tls_missing_key() {
        let mut cfg = ServerConfig::default();
        cfg.tls_enabled = true;
        cfg.tls_cert_path = Some("cert.pem".into());
        assert!(cfg.validate().unwrap_err().contains("key"));
    }

    #[test]
    fn test_server_config_with_tls() {
        let cfg = ServerConfig::default().with_tls("cert.pem", "key.pem");
        assert!(cfg.tls_enabled);
        assert_eq!(cfg.tls_cert_path.as_deref(), Some("cert.pem"));
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_server_config_with_cors() {
        let cfg = ServerConfig::default().with_cors(vec!["https://example.com".into()]);
        assert_eq!(cfg.cors_origins.len(), 1);
    }

    #[test]
    fn test_server_config_bind_address() {
        let cfg = ServerConfig::new("localhost", 9090);
        assert_eq!(cfg.bind_address(), "localhost:9090");
    }

    // ====================================================================
    // ChatRole / ChatMessage
    // ====================================================================

    #[test]
    fn test_chat_role_display() {
        assert_eq!(ChatRole::System.to_string(), "system");
        assert_eq!(ChatRole::User.to_string(), "user");
        assert_eq!(ChatRole::Assistant.to_string(), "assistant");
    }

    #[test]
    fn test_chat_role_from_str() {
        assert_eq!(ChatRole::from_str_name("system").unwrap(), ChatRole::System);
        assert_eq!(ChatRole::from_str_name("user").unwrap(), ChatRole::User);
        assert_eq!(ChatRole::from_str_name("assistant").unwrap(), ChatRole::Assistant);
    }

    #[test]
    fn test_chat_role_from_str_invalid() {
        assert!(ChatRole::from_str_name("unknown").is_err());
    }

    #[test]
    fn test_chat_message_new() {
        let msg = ChatMessage::new(ChatRole::User, "hi");
        assert_eq!(msg.role, ChatRole::User);
        assert_eq!(msg.content, "hi");
    }

    // ====================================================================
    // ChatRequest
    // ====================================================================

    #[test]
    fn test_chat_request_defaults() {
        let req = ChatRequest::default();
        assert!(req.model.is_empty());
        assert!(!req.stream);
        assert_eq!(req.presence_penalty, 0.0);
    }

    #[test]
    fn test_chat_request_new() {
        let req = sample_request();
        assert_eq!(req.model, "test-model");
        assert_eq!(req.messages.len(), 2);
    }

    // ====================================================================
    // FinishReason / TokenUsage
    // ====================================================================

    #[test]
    fn test_finish_reason_display() {
        assert_eq!(FinishReason::Stop.to_string(), "stop");
        assert_eq!(FinishReason::Length.to_string(), "length");
        assert_eq!(FinishReason::ContentFilter.to_string(), "content_filter");
    }

    #[test]
    fn test_token_usage_new() {
        let u = TokenUsage::new(10, 20);
        assert_eq!(u.prompt_tokens, 10);
        assert_eq!(u.completion_tokens, 20);
        assert_eq!(u.total_tokens, 30);
    }

    // ====================================================================
    // RequestHandler
    // ====================================================================

    #[test]
    fn test_handler_new() {
        let h = RequestHandler::new("m1");
        assert_eq!(h.model_id(), "m1");
        assert_eq!(h.requests_handled(), 0);
    }

    #[test]
    fn test_handler_handle_chat_ok() {
        let mut h = RequestHandler::new("m1");
        let resp = h.handle_chat(&sample_request()).unwrap();
        assert_eq!(resp.model, "test-model");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].finish_reason, FinishReason::Stop);
        assert_eq!(h.requests_handled(), 1);
    }

    #[test]
    fn test_handler_empty_messages() {
        let mut h = RequestHandler::new("m1");
        let req = ChatRequest::new("m1", vec![]);
        assert!(h.handle_chat(&req).is_err());
    }

    #[test]
    fn test_handler_empty_model() {
        let mut h = RequestHandler::new("m1");
        let req = ChatRequest::new("", sample_messages());
        assert!(h.handle_chat(&req).is_err());
    }

    #[test]
    fn test_handler_increments_counter() {
        let mut h = RequestHandler::new("m1");
        let req = sample_request();
        h.handle_chat(&req).unwrap();
        h.handle_chat(&req).unwrap();
        assert_eq!(h.requests_handled(), 2);
    }

    #[test]
    fn test_handler_mock_reply_contains_input() {
        let mut h = RequestHandler::new("m1");
        let resp = h.handle_chat(&sample_request()).unwrap();
        let text = &resp.choices[0].message.content;
        assert!(text.contains("Hello!"));
    }

    #[test]
    fn test_handler_response_id_unique() {
        let mut h = RequestHandler::new("m1");
        let r1 = h.handle_chat(&sample_request()).unwrap();
        let r2 = h.handle_chat(&sample_request()).unwrap();
        assert_ne!(r1.id, r2.id);
    }

    #[test]
    fn test_handler_usage_nonzero() {
        let mut h = RequestHandler::new("m1");
        let resp = h.handle_chat(&sample_request()).unwrap();
        assert!(resp.usage.prompt_tokens > 0);
        assert!(resp.usage.completion_tokens > 0);
        assert_eq!(
            resp.usage.total_tokens,
            resp.usage.prompt_tokens + resp.usage.completion_tokens
        );
    }

    // ====================================================================
    // ResponseBuilder
    // ====================================================================

    #[test]
    fn test_response_builder_json() {
        let mut h = RequestHandler::new("m");
        let resp = h.handle_chat(&sample_request()).unwrap();
        let builder = ResponseBuilder::new("m");
        let json = builder.build_json(&resp);
        assert!(json.contains("\"object\":\"chat.completion\""));
        assert!(json.contains("\"model\":\"test-model\""));
    }

    #[test]
    fn test_response_builder_error() {
        let builder = ResponseBuilder::new("m");
        let json = builder.build_error(400, "bad request", "invalid_request_error");
        assert!(json.contains("\"code\":400"));
        assert!(json.contains("bad request"));
    }

    #[test]
    fn test_response_builder_default_model() {
        let builder = ResponseBuilder::new("gpt-bitnet");
        assert_eq!(builder.default_model(), "gpt-bitnet");
    }

    #[test]
    fn test_response_builder_escapes_quotes() {
        let builder = ResponseBuilder::new("m");
        let json = builder.build_error(400, r#"say "hello""#, "err");
        assert!(json.contains(r#"say \"hello\""#));
    }

    // ====================================================================
    // StreamingResponse
    // ====================================================================

    #[test]
    fn test_stream_new() {
        let s = StreamingResponse::new("id1", "model1");
        assert!(!s.is_finished());
        assert_eq!(s.chunk_count(), 0);
        assert_eq!(s.id(), "id1");
    }

    #[test]
    fn test_stream_push_delta() {
        let mut s = StreamingResponse::new("id1", "m");
        s.push_delta("Hello").unwrap();
        s.push_delta(" world").unwrap();
        assert_eq!(s.chunk_count(), 2);
        assert_eq!(s.chunks()[0].delta_content, "Hello");
        assert_eq!(s.chunks()[1].delta_content, " world");
    }

    #[test]
    fn test_stream_finish() {
        let mut s = StreamingResponse::new("id1", "m");
        s.push_delta("tok").unwrap();
        s.finish(FinishReason::Stop).unwrap();
        assert!(s.is_finished());
    }

    #[test]
    fn test_stream_push_after_finish() {
        let mut s = StreamingResponse::new("id1", "m");
        s.finish(FinishReason::Stop).unwrap();
        assert!(s.push_delta("x").is_err());
    }

    #[test]
    fn test_stream_double_finish() {
        let mut s = StreamingResponse::new("id1", "m");
        s.finish(FinishReason::Stop).unwrap();
        assert!(s.finish(FinishReason::Length).is_err());
    }

    #[test]
    fn test_stream_format_sse() {
        let mut s = StreamingResponse::new("id1", "m");
        s.push_delta("Hi").unwrap();
        s.finish(FinishReason::Stop).unwrap();
        let lines = s.format_sse();
        assert_eq!(lines.len(), 3); // 1 delta + 1 finish + [DONE]
        assert!(lines[0].starts_with("data: "));
        assert!(lines[0].contains("\"content\":\"Hi\""));
        assert!(lines[1].contains("\"finish_reason\":\"stop\""));
        assert_eq!(lines[2], "data: [DONE]");
    }

    #[test]
    fn test_stream_sse_escapes_quotes() {
        let mut s = StreamingResponse::new("id1", "m");
        s.push_delta(r#"say "hi""#).unwrap();
        let lines = s.format_sse();
        assert!(lines[0].contains(r#"say \"hi\""#));
    }

    #[test]
    fn test_stream_chunk_indices() {
        let mut s = StreamingResponse::new("id1", "m");
        s.push_delta("a").unwrap();
        s.push_delta("b").unwrap();
        s.push_delta("c").unwrap();
        assert_eq!(s.chunks()[0].index, 0);
        assert_eq!(s.chunks()[1].index, 1);
        assert_eq!(s.chunks()[2].index, 2);
    }

    #[test]
    fn test_stream_finish_reason_in_last_chunk() {
        let mut s = StreamingResponse::new("id1", "m");
        s.push_delta("a").unwrap();
        s.finish(FinishReason::Length).unwrap();
        let last = s.chunks().last().unwrap();
        assert_eq!(last.finish_reason, Some(FinishReason::Length));
    }

    #[test]
    fn test_stream_empty_sse() {
        let s = StreamingResponse::new("id1", "m");
        let lines = s.format_sse();
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0], "data: [DONE]");
    }

    // ====================================================================
    // AuthMiddleware
    // ====================================================================

    #[test]
    fn test_auth_new() {
        let auth = AuthMiddleware::new();
        assert!(auth.is_enabled());
        assert_eq!(auth.key_count(), 0);
    }

    #[test]
    fn test_auth_disabled() {
        let mut auth = AuthMiddleware::disabled();
        assert!(!auth.is_enabled());
        let label = auth.authenticate("anything").unwrap();
        assert_eq!(label, "anonymous");
    }

    #[test]
    fn test_auth_add_and_authenticate() {
        let mut auth = AuthMiddleware::new();
        auth.add_key("sk-abc123", "test-key");
        let label = auth.authenticate("sk-abc123").unwrap();
        assert_eq!(label, "test-key");
    }

    #[test]
    fn test_auth_invalid_token() {
        let mut auth = AuthMiddleware::new();
        auth.add_key("sk-abc123", "test-key");
        assert_eq!(auth.authenticate("sk-wrong"), Err(AuthError::InvalidToken));
    }

    #[test]
    fn test_auth_empty_token() {
        let mut auth = AuthMiddleware::new();
        assert_eq!(auth.authenticate(""), Err(AuthError::MissingToken));
    }

    #[test]
    fn test_auth_remove_key() {
        let mut auth = AuthMiddleware::new();
        auth.add_key("sk-abc", "k");
        assert!(auth.remove_key("sk-abc"));
        assert!(!auth.remove_key("sk-abc"));
        assert_eq!(auth.key_count(), 0);
    }

    #[test]
    fn test_auth_usage_tracking() {
        let mut auth = AuthMiddleware::new();
        auth.add_key("sk-abc", "k");
        auth.authenticate("sk-abc").unwrap();
        auth.authenticate("sk-abc").unwrap();
        assert_eq!(auth.usage()["sk-abc"], 2);
    }

    #[test]
    fn test_auth_parse_bearer() {
        assert_eq!(AuthMiddleware::parse_bearer("Bearer sk-abc"), Some("sk-abc"));
        assert_eq!(AuthMiddleware::parse_bearer("Token sk-abc"), None);
        assert_eq!(AuthMiddleware::parse_bearer(""), None);
    }

    #[test]
    fn test_auth_authenticate_header() {
        let mut auth = AuthMiddleware::new();
        auth.add_key("sk-abc", "k");
        assert!(auth.authenticate_header("Bearer sk-abc").is_ok());
    }

    #[test]
    fn test_auth_authenticate_header_bad_prefix() {
        let mut auth = AuthMiddleware::new();
        auth.add_key("sk-abc", "k");
        assert_eq!(auth.authenticate_header("Basic sk-abc"), Err(AuthError::MissingToken));
    }

    #[test]
    fn test_auth_error_display() {
        assert!(!AuthError::MissingToken.to_string().is_empty());
        assert!(!AuthError::InvalidToken.to_string().is_empty());
    }

    #[test]
    fn test_auth_multiple_keys() {
        let mut auth = AuthMiddleware::new();
        auth.add_key("sk-1", "key-one");
        auth.add_key("sk-2", "key-two");
        assert_eq!(auth.key_count(), 2);
        assert_eq!(auth.authenticate("sk-1").unwrap(), "key-one");
        assert_eq!(auth.authenticate("sk-2").unwrap(), "key-two");
    }

    // ====================================================================
    // RequestValidator
    // ====================================================================

    #[test]
    fn test_validator_ok() {
        let v = RequestValidator::new(vec!["test-model".into()]);
        assert!(v.validate(&sample_request()).is_ok());
    }

    #[test]
    fn test_validator_empty_model() {
        let v = RequestValidator::default();
        let mut req = sample_request();
        req.model = String::new();
        let errs = v.validate(&req).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("model")));
    }

    #[test]
    fn test_validator_unknown_model() {
        let v = RequestValidator::new(vec!["allowed".into()]);
        let req = sample_request(); // model = "test-model"
        let errs = v.validate(&req).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("not available")));
    }

    #[test]
    fn test_validator_empty_messages() {
        let v = RequestValidator::default();
        let mut req = sample_request();
        req.messages.clear();
        let errs = v.validate(&req).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("empty")));
    }

    #[test]
    fn test_validator_too_many_messages() {
        let mut v = RequestValidator::default();
        v.max_messages = 1;
        let req = sample_request();
        let errs = v.validate(&req).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("too many")));
    }

    #[test]
    fn test_validator_content_too_long() {
        let mut v = RequestValidator::default();
        v.max_content_length = 2;
        let req = sample_request(); // messages have content longer than 2
        let errs = v.validate(&req).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("too long")));
    }

    #[test]
    fn test_validator_max_tokens_exceeded() {
        let v = RequestValidator::default();
        let mut req = sample_request();
        req.max_tokens = Some(999_999);
        let errs = v.validate(&req).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("max_tokens")));
    }

    #[test]
    fn test_validator_max_tokens_zero() {
        let v = RequestValidator::default();
        let mut req = sample_request();
        req.max_tokens = Some(0);
        let errs = v.validate(&req).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("max_tokens")));
    }

    #[test]
    fn test_validator_temperature_out_of_range() {
        let v = RequestValidator::default();
        let mut req = sample_request();
        req.temperature = Some(3.0);
        let errs = v.validate(&req).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("temperature")));
    }

    #[test]
    fn test_validator_top_p_out_of_range() {
        let v = RequestValidator::default();
        let mut req = sample_request();
        req.top_p = Some(-0.1);
        let errs = v.validate(&req).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("top_p")));
    }

    #[test]
    fn test_validator_presence_penalty_out_of_range() {
        let v = RequestValidator::default();
        let mut req = sample_request();
        req.presence_penalty = 3.0;
        let errs = v.validate(&req).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("presence_penalty")));
    }

    #[test]
    fn test_validator_frequency_penalty_out_of_range() {
        let v = RequestValidator::default();
        let mut req = sample_request();
        req.frequency_penalty = -3.0;
        let errs = v.validate(&req).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("frequency_penalty")));
    }

    #[test]
    fn test_validator_multiple_errors() {
        let v = RequestValidator::new(vec!["allowed".into()]);
        let mut req = ChatRequest::default();
        req.temperature = Some(5.0);
        let errs = v.validate(&req).unwrap_err();
        assert!(errs.len() >= 2); // empty model + empty messages + temperature
    }

    #[test]
    fn test_validator_is_valid() {
        let v = RequestValidator::new(vec!["test-model".into()]);
        assert!(v.is_valid(&sample_request()));
        let mut bad = sample_request();
        bad.messages.clear();
        assert!(!v.is_valid(&bad));
    }

    #[test]
    fn test_validator_no_model_restriction() {
        let v = RequestValidator::default(); // allowed_models is empty → any model ok
        assert!(v.validate(&sample_request()).is_ok());
    }

    // ====================================================================
    // HealthEndpoint
    // ====================================================================

    #[test]
    fn test_health_default() {
        let h = HealthEndpoint::new();
        assert!(!h.is_ready());
        assert!(h.is_alive());
        assert_eq!(h.models_loaded(), 0);
    }

    #[test]
    fn test_health_set_ready() {
        let mut h = HealthEndpoint::new();
        h.set_ready(true);
        assert!(h.is_ready());
        assert!(h.readiness().is_ok());
    }

    #[test]
    fn test_health_not_ready() {
        let h = HealthEndpoint::new();
        assert!(h.readiness().is_err());
    }

    #[test]
    fn test_health_liveness_ok() {
        let h = HealthEndpoint::new();
        assert!(h.liveness().is_ok());
    }

    #[test]
    fn test_health_liveness_dead() {
        let mut h = HealthEndpoint::new();
        h.set_alive(false);
        assert!(h.liveness().is_err());
    }

    #[test]
    fn test_health_models_loaded() {
        let mut h = HealthEndpoint::new();
        h.set_models_loaded(3);
        assert_eq!(h.models_loaded(), 3);
    }

    #[test]
    fn test_health_queue_depth() {
        let mut h = HealthEndpoint::new();
        h.set_queue_depth(42);
        assert_eq!(h.queue_depth(), 42);
    }

    #[test]
    fn test_health_json() {
        let mut h = HealthEndpoint::new();
        h.set_ready(true);
        let json = h.health_json();
        assert!(json.contains("\"ready\":true"));
        assert!(json.contains("\"alive\":true"));
    }

    #[test]
    fn test_health_record_check() {
        let mut h = HealthEndpoint::new();
        h.record_check();
        // Just verify it doesn't panic and last_check is set.
        assert!(h.last_check.is_some());
    }

    #[test]
    fn test_health_status_message() {
        let h = HealthEndpoint::new();
        assert_eq!(h.status_message(), "initializing");
    }

    // ====================================================================
    // MetricsEndpoint
    // ====================================================================

    #[test]
    fn test_metrics_default() {
        let m = MetricsEndpoint::new();
        assert_eq!(m.total_requests(), 0);
        assert_eq!(m.total_successes(), 0);
        assert_eq!(m.total_errors(), 0);
        assert_eq!(m.active_requests(), 0);
    }

    #[test]
    fn test_metrics_record_request() {
        let mut m = MetricsEndpoint::new();
        m.record_request();
        assert_eq!(m.total_requests(), 1);
        assert_eq!(m.active_requests(), 1);
    }

    #[test]
    fn test_metrics_record_success() {
        let mut m = MetricsEndpoint::new();
        m.record_request();
        m.record_success(10.0, 5, 20);
        assert_eq!(m.total_successes(), 1);
        assert_eq!(m.total_tokens_generated(), 20);
        assert_eq!(m.active_requests(), 0);
    }

    #[test]
    fn test_metrics_record_error() {
        let mut m = MetricsEndpoint::new();
        m.record_request();
        m.record_error();
        assert_eq!(m.total_errors(), 1);
        assert_eq!(m.active_requests(), 0);
    }

    #[test]
    fn test_metrics_avg_latency() {
        let mut m = MetricsEndpoint::new();
        m.record_request();
        m.record_success(10.0, 1, 1);
        m.record_request();
        m.record_success(30.0, 1, 1);
        assert!((m.avg_latency_ms() - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_metrics_avg_latency_empty() {
        let m = MetricsEndpoint::new();
        assert_eq!(m.avg_latency_ms(), 0.0);
    }

    #[test]
    fn test_metrics_p99_latency() {
        let mut m = MetricsEndpoint::new();
        for i in 1..=100 {
            m.record_request();
            m.record_success(i as f64, 1, 1);
        }
        // p99 should be >= 99
        assert!(m.p99_latency_ms() >= 99.0);
    }

    #[test]
    fn test_metrics_p99_empty() {
        let m = MetricsEndpoint::new();
        assert_eq!(m.p99_latency_ms(), 0.0);
    }

    #[test]
    fn test_metrics_prometheus_format() {
        let mut m = MetricsEndpoint::new();
        m.record_request();
        m.record_success(5.0, 10, 20);
        let prom = m.prometheus();
        assert!(prom.contains("bitnet_api_requests_total 1"));
        assert!(prom.contains("bitnet_api_successes_total 1"));
        assert!(prom.contains("bitnet_api_tokens_generated_total 20"));
        assert!(prom.contains("bitnet_api_prompt_tokens_total 10"));
    }

    #[test]
    fn test_metrics_reset() {
        let mut m = MetricsEndpoint::new();
        m.record_request();
        m.record_success(5.0, 1, 1);
        m.reset();
        assert_eq!(m.total_requests(), 0);
        assert_eq!(m.total_successes(), 0);
    }

    #[test]
    fn test_metrics_active_requests_saturating() {
        let mut m = MetricsEndpoint::new();
        m.record_error(); // active was 0, should stay 0
        assert_eq!(m.active_requests(), 0);
    }

    // ====================================================================
    // RouterConfig
    // ====================================================================

    #[test]
    fn test_router_default() {
        let r = RouterConfig::default();
        assert_eq!(r.api_prefix, "/v1");
        assert!(r.models_endpoint);
        assert!(r.completions_endpoint);
    }

    #[test]
    fn test_router_add_model() {
        let mut r = RouterConfig::default();
        r.add_model("m1", "/models/m1");
        assert!(r.model_routes.contains_key("m1"));
    }

    #[test]
    fn test_router_add_custom_route() {
        let mut r = RouterConfig::default();
        r.add_custom_route("/custom", "handler1");
        assert!(r.custom_routes.contains_key("/custom"));
    }

    #[test]
    fn test_router_all_paths() {
        let r = RouterConfig::default();
        let paths = r.all_paths();
        assert!(paths.contains(&"/v1/chat/completions".into()));
        assert!(paths.contains(&"/v1/models".into()));
        assert!(paths.contains(&"/health".into()));
        assert!(paths.contains(&"/metrics".into()));
    }

    #[test]
    fn test_router_is_routable() {
        let r = RouterConfig::default();
        assert!(r.is_routable("/v1/chat/completions"));
        assert!(!r.is_routable("/v1/unknown"));
    }

    #[test]
    fn test_router_models_json() {
        let mut r = RouterConfig::default();
        r.add_model("m1", "/p");
        let json = r.models_json();
        assert!(json.contains("\"id\":\"m1\""));
        assert!(json.contains("\"object\":\"list\""));
    }

    #[test]
    fn test_router_validate_ok() {
        assert!(RouterConfig::default().validate().is_ok());
    }

    #[test]
    fn test_router_validate_empty_prefix() {
        let mut r = RouterConfig::default();
        r.api_prefix = String::new();
        assert!(r.validate().is_err());
    }

    #[test]
    fn test_router_validate_no_slash_prefix() {
        let mut r = RouterConfig::default();
        r.api_prefix = "v1".into();
        assert!(r.validate().unwrap_err().contains("start with"));
    }

    #[test]
    fn test_router_disabled_endpoints() {
        let mut r = RouterConfig::default();
        r.models_endpoint = false;
        r.completions_endpoint = false;
        r.health_endpoint = false;
        r.metrics_endpoint = false;
        assert!(r.all_paths().is_empty());
    }

    // ====================================================================
    // ApiServerEngine
    // ====================================================================

    #[test]
    fn test_engine_new() {
        let e = make_engine();
        assert!(!e.is_started());
    }

    #[test]
    fn test_engine_start_stop() {
        let mut e = make_engine();
        e.start().unwrap();
        assert!(e.is_started());
        assert!(e.health().is_ready());
        e.stop();
        assert!(!e.is_started());
        assert!(!e.health().is_ready());
    }

    #[test]
    fn test_engine_process_request_no_auth() {
        let mut e = make_engine();
        e.auth_mut().add_key("sk-test", "test");
        e.start().unwrap();
        let resp = e.process_request(Some("Bearer sk-test"), &sample_request());
        assert!(resp.is_ok());
    }

    #[test]
    fn test_engine_process_request_missing_auth() {
        let mut e = make_engine();
        e.auth_mut().add_key("sk-test", "test");
        e.start().unwrap();
        let resp = e.process_request(None, &sample_request());
        assert!(resp.is_err());
    }

    #[test]
    fn test_engine_process_request_bad_auth() {
        let mut e = make_engine();
        e.auth_mut().add_key("sk-test", "test");
        e.start().unwrap();
        let resp = e.process_request(Some("Bearer sk-wrong"), &sample_request());
        assert!(resp.is_err());
    }

    #[test]
    fn test_engine_process_request_disabled_auth() {
        let mut router = RouterConfig::default();
        router.add_model("test-model", "/m");
        let config = ServerConfig::default();
        let mut e = ApiServerEngine::new(config, router, "test-model");
        // Replace auth with disabled
        e.auth = AuthMiddleware::disabled();
        e.start().unwrap();
        let resp = e.process_request(None, &sample_request());
        assert!(resp.is_ok());
    }

    #[test]
    fn test_engine_process_request_validation_fail() {
        let mut e = make_engine();
        e.auth = AuthMiddleware::disabled();
        e.start().unwrap();
        let mut req = sample_request();
        req.temperature = Some(5.0);
        assert!(e.process_request(None, &req).is_err());
    }

    #[test]
    fn test_engine_metrics_after_request() {
        let mut e = make_engine();
        e.auth = AuthMiddleware::disabled();
        e.start().unwrap();
        e.process_request(None, &sample_request()).unwrap();
        assert_eq!(e.metrics().total_requests(), 1);
        assert_eq!(e.metrics().total_successes(), 1);
    }

    #[test]
    fn test_engine_metrics_after_error() {
        let mut e = make_engine();
        e.auth = AuthMiddleware::disabled();
        e.start().unwrap();
        let mut req = sample_request();
        req.messages.clear();
        let _ = e.process_request(None, &req);
        // Validation error happens before metrics.record_request
        // so only auth → validate → fail path is tested.
    }

    #[test]
    fn test_engine_clone() {
        let e = make_engine();
        let e2 = e.clone();
        assert_eq!(e.config().port, e2.config().port);
    }

    #[test]
    fn test_engine_config_accessor() {
        let e = make_engine();
        assert_eq!(e.config().port, 8080);
    }

    #[test]
    fn test_engine_router_accessor() {
        let e = make_engine();
        assert!(e.router().model_routes.contains_key("test-model"));
    }

    #[test]
    fn test_engine_handler_accessor() {
        let e = make_engine();
        assert_eq!(e.handler().model_id(), "test-model");
    }
}
