//! Server-Sent Events (SSE) fallback for streaming inference.
//!
//! When a client does not support WebSocket, the server can fall back
//! to SSE for token-by-token delivery.

use crate::websocket::WsMessage;

/// SSE event for token streaming.
#[derive(Debug, Clone)]
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

/// Format a [`WsMessage`] as a standards-compliant SSE event string.
///
/// The returned string includes the trailing double-newline required by
/// the SSE specification.
pub fn format_sse_event(msg: &WsMessage) -> String {
    match msg {
        WsMessage::Token { .. } => {
            let data = serde_json::to_string(msg).unwrap_or_default();
            format!("event: token\ndata: {data}\n\n")
        }
        WsMessage::Metadata { .. } => {
            let data = serde_json::to_string(msg).unwrap_or_default();
            format!("event: metadata\ndata: {data}\n\n")
        }
        WsMessage::Error { .. } => {
            let data = serde_json::to_string(msg).unwrap_or_default();
            format!("event: error\ndata: {data}\n\n")
        }
        WsMessage::Ping => "event: ping\ndata: \n\n".to_string(),
        WsMessage::Pong => "event: pong\ndata: \n\n".to_string(),
        WsMessage::Request { .. } => {
            // Requests are client→server; formatting as SSE is
            // atypical but supported for debugging.
            let data = serde_json::to_string(msg).unwrap_or_default();
            format!("event: request\ndata: {data}\n\n")
        }
    }
}

/// Build an [`SseToken`] from a [`WsMessage`] with optional id and
/// retry hint.
pub fn build_sse_token(msg: &WsMessage, id: Option<String>, retry: Option<u64>) -> SseToken {
    let (event, data) = match msg {
        WsMessage::Token { .. } => {
            ("token".to_string(), serde_json::to_string(msg).unwrap_or_default())
        }
        WsMessage::Metadata { .. } => {
            ("metadata".to_string(), serde_json::to_string(msg).unwrap_or_default())
        }
        WsMessage::Error { .. } => {
            ("error".to_string(), serde_json::to_string(msg).unwrap_or_default())
        }
        WsMessage::Ping => ("ping".to_string(), String::new()),
        WsMessage::Pong => ("pong".to_string(), String::new()),
        WsMessage::Request { .. } => {
            ("request".to_string(), serde_json::to_string(msg).unwrap_or_default())
        }
    };
    SseToken { id, event, data, retry }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_config_defaults_are_sane() {
        let cfg = SseConfig::default();
        assert_eq!(cfg.retry_ms, 3000);
        assert_eq!(cfg.keep_alive_secs, 15);
    }

    #[test]
    fn format_token_event() {
        let msg = WsMessage::Token {
            text: "hello".into(),
            token_id: 42,
            logprob: None,
            finish_reason: None,
        };
        let sse = format_sse_event(&msg);
        assert!(sse.starts_with("event: token\n"));
        assert!(sse.contains("\"token_id\":42"));
        assert!(sse.ends_with("\n\n"));
    }

    #[test]
    fn format_ping_event() {
        let sse = format_sse_event(&WsMessage::Ping);
        assert_eq!(sse, "event: ping\ndata: \n\n");
    }

    #[test]
    fn build_sse_token_captures_fields() {
        let msg = WsMessage::Error { message: "oops".into(), code: 500 };
        let tok = build_sse_token(&msg, Some("id-1".into()), Some(5000));
        assert_eq!(tok.event, "error");
        assert_eq!(tok.id.as_deref(), Some("id-1"));
        assert_eq!(tok.retry, Some(5000));
        assert!(tok.data.contains("oops"));
    }
}
