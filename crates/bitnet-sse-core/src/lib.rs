//! Core Server-Sent Events (SSE) primitives shared across server flows.

use serde::{Deserialize, Serialize};

/// SSE event for token streaming.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SseToken {
    /// Optional event id for resumable streams.
    pub id: Option<String>,
    /// SSE event type (e.g. `"token"`, `"metadata"`, `"error"`).
    pub event: String,
    /// JSON-encoded event payload.
    pub data: String,
    /// Retry interval hint for the client (milliseconds).
    pub retry: Option<u64>,
}

/// SSE stream configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SseConfig {
    /// Default retry interval hint sent to clients (milliseconds).
    pub retry_ms: u64,
    /// Interval for SSE keep-alive comments (seconds).
    pub keep_alive_secs: u64,
}

impl Default for SseConfig {
    fn default() -> Self {
        Self { retry_ms: 3000, keep_alive_secs: 15 }
    }
}

/// Build a standards-compliant SSE event string with a trailing `\n\n`.
pub fn format_sse_event(event: &str, data: &str) -> String {
    format!("event: {event}\ndata: {data}\n\n")
}

/// Build an [`SseToken`] from event/data with optional id and retry hint.
pub fn build_sse_token(
    event: impl Into<String>,
    data: impl Into<String>,
    id: Option<String>,
    retry: Option<u64>,
) -> SseToken {
    SseToken { id, event: event.into(), data: data.into(), retry }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane() {
        let cfg = SseConfig::default();
        assert_eq!(cfg.retry_ms, 3000);
        assert_eq!(cfg.keep_alive_secs, 15);
    }

    #[test]
    fn event_format_has_required_terminator() {
        let event = format_sse_event("token", "{\"x\":1}");
        assert_eq!(event, "event: token\ndata: {\"x\":1}\n\n");
    }

    #[test]
    fn token_builder_sets_all_fields() {
        let token =
            build_sse_token("error", "{\"message\":\"oops\"}", Some("e-1".into()), Some(5000));
        assert_eq!(token.event, "error");
        assert_eq!(token.data, "{\"message\":\"oops\"}");
        assert_eq!(token.id.as_deref(), Some("e-1"));
        assert_eq!(token.retry, Some(5000));
    }
}
