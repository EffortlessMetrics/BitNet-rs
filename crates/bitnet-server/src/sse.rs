//! Server-Sent Events (SSE) fallback for streaming inference.
//!
//! When a client does not support WebSocket, the server can fall back
//! to SSE for token-by-token delivery.

pub use bitnet_sse_core::{SseConfig, SseToken};
use bitnet_sse_core::{
    build_sse_token as build_sse_token_core, format_sse_event as format_sse_event_core,
};

use crate::websocket::WsMessage;

/// Format a [`WsMessage`] as a standards-compliant SSE event string.
///
/// The returned string includes the trailing double-newline required by
/// the SSE specification.
pub fn format_sse_event(msg: &WsMessage) -> String {
    match msg {
        WsMessage::Token { .. } => {
            let data = serde_json::to_string(msg).unwrap_or_default();
            format_sse_event_core("token", &data)
        }
        WsMessage::Metadata { .. } => {
            let data = serde_json::to_string(msg).unwrap_or_default();
            format_sse_event_core("metadata", &data)
        }
        WsMessage::Error { .. } => {
            let data = serde_json::to_string(msg).unwrap_or_default();
            format_sse_event_core("error", &data)
        }
        WsMessage::Ping => format_sse_event_core("ping", ""),
        WsMessage::Pong => format_sse_event_core("pong", ""),
        WsMessage::Request { .. } => {
            // Requests are client→server; formatting as SSE is
            // atypical but supported for debugging.
            let data = serde_json::to_string(msg).unwrap_or_default();
            format_sse_event_core("request", &data)
        }
    }
}

/// Build an [`SseToken`] from a [`WsMessage`] with optional id and
/// retry hint.
pub fn build_sse_token(msg: &WsMessage, id: Option<String>, retry: Option<u64>) -> SseToken {
    let (event, data): (&str, String) = match msg {
        WsMessage::Token { .. } => ("token", serde_json::to_string(msg).unwrap_or_default()),
        WsMessage::Metadata { .. } => ("metadata", serde_json::to_string(msg).unwrap_or_default()),
        WsMessage::Error { .. } => ("error", serde_json::to_string(msg).unwrap_or_default()),
        WsMessage::Ping => ("ping", String::new()),
        WsMessage::Pong => ("pong", String::new()),
        WsMessage::Request { .. } => ("request", serde_json::to_string(msg).unwrap_or_default()),
    };
    build_sse_token_core(event, data, id, retry)
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
