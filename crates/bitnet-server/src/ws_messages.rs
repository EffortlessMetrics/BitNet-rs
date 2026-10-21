//! WebSocket message types for streaming chat.
//!
//! Defines request/response message types for WebSocket-based
//! inference streaming, compatible with OpenAI-style protocols.

use std::collections::HashMap;

/// Client-to-server WebSocket message types.
#[derive(Debug, Clone, PartialEq)]
pub enum ClientMessage {
    /// Start a new generation request.
    Generate { prompt: String, max_tokens: usize, temperature: f32, stream: bool },
    /// Cancel an in-progress generation.
    Cancel { request_id: String },
    /// Ping for keepalive.
    Ping { seq: u64 },
    /// Client metadata.
    Hello { client_id: String, protocol_version: u32 },
}

/// Server-to-client WebSocket message types.
#[derive(Debug, Clone, PartialEq)]
pub enum ServerMessage {
    /// A generated token.
    Token {
        request_id: String,
        token: String,
        token_id: u32,
        index: usize,
        finish_reason: Option<FinishReason>,
    },
    /// Generation complete.
    Done { request_id: String, total_tokens: usize, elapsed_ms: u64 },
    /// Error response.
    Error { request_id: Option<String>, code: u16, message: String },
    /// Pong response to client ping.
    Pong { seq: u64 },
    /// Server welcome.
    Welcome { server_version: String, capabilities: Vec<String> },
}

/// Reason generation finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Hit max_tokens limit.
    Length,
    /// Generated a stop token.
    Stop,
    /// Client cancelled.
    Cancelled,
    /// Internal error.
    Error,
}

impl FinishReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Length => "length",
            Self::Stop => "stop",
            Self::Cancelled => "cancelled",
            Self::Error => "error",
        }
    }
}

/// WebSocket connection state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    Connecting,
    Connected,
    Generating,
    Closing,
    Closed,
}

impl ConnectionState {
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Connected | Self::Generating)
    }

    pub fn can_send(&self) -> bool {
        !matches!(self, Self::Closing | Self::Closed)
    }
}

/// Format a token message as JSON-like string.
pub fn format_token_event(request_id: &str, token: &str, index: usize) -> String {
    format!(
        r#"{{"type":"token","request_id":"{}","token":"{}","index":{}}}"#,
        request_id, token, index,
    )
}

/// Format a done event.
pub fn format_done_event(request_id: &str, total_tokens: usize, elapsed_ms: u64) -> String {
    format!(
        r#"{{"type":"done","request_id":"{}","total_tokens":{},"elapsed_ms":{}}}"#,
        request_id, total_tokens, elapsed_ms,
    )
}

/// Format an error event.
pub fn format_error_event(code: u16, message: &str) -> String {
    format!(r#"{{"type":"error","code":{},"message":"{}"}}"#, code, message)
}

/// Track active generation sessions.
#[derive(Debug, Default)]
pub struct SessionTracker {
    sessions: HashMap<String, ConnectionState>,
}

impl SessionTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, id: String, state: ConnectionState) {
        self.sessions.insert(id, state);
    }

    pub fn update(&mut self, id: &str, state: ConnectionState) -> bool {
        if let Some(s) = self.sessions.get_mut(id) {
            *s = state;
            true
        } else {
            false
        }
    }

    pub fn remove(&mut self, id: &str) -> bool {
        self.sessions.remove(id).is_some()
    }

    pub fn active_count(&self) -> usize {
        self.sessions.values().filter(|s| s.is_active()).count()
    }

    pub fn total_count(&self) -> usize {
        self.sessions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_generate() {
        let msg = ClientMessage::Generate {
            prompt: "hello".into(),
            max_tokens: 32,
            temperature: 0.7,
            stream: true,
        };
        if let ClientMessage::Generate { max_tokens, .. } = msg {
            assert_eq!(max_tokens, 32);
        }
    }

    #[test]
    fn test_server_token() {
        let msg = ServerMessage::Token {
            request_id: "r1".into(),
            token: "the".into(),
            token_id: 42,
            index: 0,
            finish_reason: None,
        };
        if let ServerMessage::Token { token_id, .. } = msg {
            assert_eq!(token_id, 42);
        }
    }

    #[test]
    fn test_finish_reason() {
        assert_eq!(FinishReason::Length.as_str(), "length");
        assert_eq!(FinishReason::Stop.as_str(), "stop");
        assert_eq!(FinishReason::Cancelled.as_str(), "cancelled");
    }

    #[test]
    fn test_connection_state() {
        assert!(ConnectionState::Connected.is_active());
        assert!(ConnectionState::Generating.is_active());
        assert!(!ConnectionState::Closed.is_active());
        assert!(ConnectionState::Connected.can_send());
        assert!(!ConnectionState::Closed.can_send());
    }

    #[test]
    fn test_format_token_event() {
        let s = format_token_event("r1", "hello", 0);
        assert!(s.contains("\"type\":\"token\""));
        assert!(s.contains("\"request_id\":\"r1\""));
    }

    #[test]
    fn test_format_done_event() {
        let s = format_done_event("r1", 10, 500);
        assert!(s.contains("\"type\":\"done\""));
        assert!(s.contains("\"total_tokens\":10"));
    }

    #[test]
    fn test_format_error_event() {
        let s = format_error_event(500, "internal error");
        assert!(s.contains("\"code\":500"));
    }

    #[test]
    fn test_session_tracker_add_remove() {
        let mut tracker = SessionTracker::new();
        tracker.add("s1".into(), ConnectionState::Connected);
        assert_eq!(tracker.total_count(), 1);
        assert_eq!(tracker.active_count(), 1);
        tracker.remove("s1");
        assert_eq!(tracker.total_count(), 0);
    }

    #[test]
    fn test_session_tracker_update() {
        let mut tracker = SessionTracker::new();
        tracker.add("s1".into(), ConnectionState::Connected);
        assert!(tracker.update("s1", ConnectionState::Generating));
        assert!(!tracker.update("nonexistent", ConnectionState::Closed));
    }

    #[test]
    fn test_server_welcome() {
        let msg = ServerMessage::Welcome {
            server_version: "0.2.1".into(),
            capabilities: vec!["streaming".into(), "batch".into()],
        };
        if let ServerMessage::Welcome { capabilities, .. } = msg {
            assert_eq!(capabilities.len(), 2);
        }
    }

    #[test]
    fn test_client_hello() {
        let msg = ClientMessage::Hello { client_id: "cli-1".into(), protocol_version: 1 };
        if let ClientMessage::Hello { protocol_version, .. } = msg {
            assert_eq!(protocol_version, 1);
        }
    }

    #[test]
    fn test_done_with_finish_reason() {
        let msg = ServerMessage::Token {
            request_id: "r1".into(),
            token: "".into(),
            token_id: 0,
            index: 5,
            finish_reason: Some(FinishReason::Stop),
        };
        if let ServerMessage::Token { finish_reason, .. } = msg {
            assert_eq!(finish_reason, Some(FinishReason::Stop));
        }
    }
}
