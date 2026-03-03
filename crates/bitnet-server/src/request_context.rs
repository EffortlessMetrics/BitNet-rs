//! Request context for inference API.
//!
//! Carries per-request metadata through the inference pipeline:
//! request ID, timing, model selection, and client info.

use std::time::{Duration, Instant};

/// Unique request identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RequestId(pub String);

impl RequestId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Generate a simple sequential ID.
    pub fn sequential(n: u64) -> Self {
        Self(format!("req-{n}"))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Client information.
#[derive(Debug, Clone, Default)]
pub struct ClientInfo {
    pub ip: Option<String>,
    pub user_agent: Option<String>,
    pub api_key_id: Option<String>,
}

/// Request context carrying metadata through the pipeline.
#[derive(Debug)]
pub struct RequestContext {
    pub id: RequestId,
    pub created_at: Instant,
    pub model_id: Option<String>,
    pub client: ClientInfo,
    pub max_tokens: usize,
    pub temperature: f32,
    pub stream: bool,
    deadlines: Option<Duration>,
}

impl RequestContext {
    pub fn new(id: RequestId) -> Self {
        Self {
            id,
            created_at: Instant::now(),
            model_id: None,
            client: ClientInfo::default(),
            max_tokens: 256,
            temperature: 1.0,
            stream: false,
            deadlines: None,
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model_id = Some(model.into());
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

    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    pub fn with_deadline(mut self, timeout: Duration) -> Self {
        self.deadlines = Some(timeout);
        self
    }

    pub fn with_client(mut self, client: ClientInfo) -> Self {
        self.client = client;
        self
    }

    /// Time elapsed since request creation.
    pub fn elapsed(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Check if the request has exceeded its deadline.
    pub fn is_expired(&self) -> bool {
        if let Some(deadline) = self.deadlines { self.elapsed() > deadline } else { false }
    }

    /// Remaining time before deadline (None if no deadline or expired).
    pub fn remaining(&self) -> Option<Duration> {
        self.deadlines.and_then(|d| d.checked_sub(self.elapsed()))
    }
}

/// Builder for batch request contexts.
#[derive(Debug)]
pub struct RequestBatch {
    requests: Vec<RequestContext>,
}

impl RequestBatch {
    pub fn new() -> Self {
        Self { requests: Vec::new() }
    }

    pub fn add(&mut self, ctx: RequestContext) {
        self.requests.push(ctx);
    }

    pub fn len(&self) -> usize {
        self.requests.len()
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    pub fn total_max_tokens(&self) -> usize {
        self.requests.iter().map(|r| r.max_tokens).sum()
    }

    pub fn has_streaming(&self) -> bool {
        self.requests.iter().any(|r| r.stream)
    }

    pub fn iter(&self) -> impl Iterator<Item = &RequestContext> {
        self.requests.iter()
    }

    pub fn expired_count(&self) -> usize {
        self.requests.iter().filter(|r| r.is_expired()).count()
    }
}

impl Default for RequestBatch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_id() {
        let id = RequestId::new("test-123");
        assert_eq!(id.as_str(), "test-123");
        assert_eq!(format!("{id}"), "test-123");
    }

    #[test]
    fn test_sequential_id() {
        let id = RequestId::sequential(42);
        assert_eq!(id.as_str(), "req-42");
    }

    #[test]
    fn test_context_builder() {
        let ctx = RequestContext::new(RequestId::new("r1"))
            .with_model("phi-4")
            .with_max_tokens(100)
            .with_temperature(0.7)
            .with_stream(true);
        assert_eq!(ctx.model_id.as_deref(), Some("phi-4"));
        assert_eq!(ctx.max_tokens, 100);
        assert!(ctx.stream);
    }

    #[test]
    fn test_elapsed() {
        let ctx = RequestContext::new(RequestId::new("r"));
        std::thread::sleep(Duration::from_millis(10));
        assert!(ctx.elapsed() >= Duration::from_millis(5));
    }

    #[test]
    fn test_not_expired() {
        let ctx = RequestContext::new(RequestId::new("r")).with_deadline(Duration::from_secs(60));
        assert!(!ctx.is_expired());
    }

    #[test]
    fn test_expired() {
        let ctx = RequestContext::new(RequestId::new("r")).with_deadline(Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(10));
        assert!(ctx.is_expired());
    }

    #[test]
    fn test_no_deadline() {
        let ctx = RequestContext::new(RequestId::new("r"));
        assert!(!ctx.is_expired());
        assert!(ctx.remaining().is_none());
    }

    #[test]
    fn test_remaining() {
        let ctx = RequestContext::new(RequestId::new("r")).with_deadline(Duration::from_secs(60));
        assert!(ctx.remaining().is_some());
    }

    #[test]
    fn test_batch() {
        let mut batch = RequestBatch::new();
        batch.add(RequestContext::new(RequestId::new("r1")).with_max_tokens(50));
        batch.add(RequestContext::new(RequestId::new("r2")).with_max_tokens(100).with_stream(true));
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.total_max_tokens(), 150);
        assert!(batch.has_streaming());
    }

    #[test]
    fn test_batch_empty() {
        let batch = RequestBatch::new();
        assert!(batch.is_empty());
        assert_eq!(batch.total_max_tokens(), 0);
        assert!(!batch.has_streaming());
    }

    #[test]
    fn test_client_info() {
        let client = ClientInfo {
            ip: Some("127.0.0.1".into()),
            user_agent: Some("test/1.0".into()),
            api_key_id: None,
        };
        let ctx = RequestContext::new(RequestId::new("r")).with_client(client);
        assert_eq!(ctx.client.ip.as_deref(), Some("127.0.0.1"));
    }

    #[test]
    fn test_default_values() {
        let ctx = RequestContext::new(RequestId::new("r"));
        assert_eq!(ctx.max_tokens, 256);
        assert_eq!(ctx.temperature, 1.0);
        assert!(!ctx.stream);
    }
}
