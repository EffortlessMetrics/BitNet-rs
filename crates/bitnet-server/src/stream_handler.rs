//! Streaming output handler.
//!
//! Server-Sent Events (SSE) formatting for streaming inference responses.

use std::time::Instant;

/// SSE event type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SseEventType {
    Token,
    Done,
    Error,
    Heartbeat,
}

impl SseEventType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Token => "token",
            Self::Done => "done",
            Self::Error => "error",
            Self::Heartbeat => "heartbeat",
        }
    }
}

/// A single SSE event.
#[derive(Debug, Clone)]
pub struct SseEvent {
    pub event_type: SseEventType,
    pub data: String,
    pub id: Option<String>,
}

impl SseEvent {
    pub fn token(data: &str) -> Self {
        Self { event_type: SseEventType::Token, data: data.to_string(), id: None }
    }

    pub fn done() -> Self {
        Self { event_type: SseEventType::Done, data: "[DONE]".to_string(), id: None }
    }

    pub fn error(msg: &str) -> Self {
        Self { event_type: SseEventType::Error, data: msg.to_string(), id: None }
    }

    pub fn heartbeat() -> Self {
        Self { event_type: SseEventType::Heartbeat, data: String::new(), id: None }
    }

    /// Format as SSE wire format.
    pub fn format(&self) -> String {
        let mut out = String::new();
        if let Some(ref id) = self.id {
            out.push_str(&format!("id: {id}\n"));
        }
        out.push_str(&format!("event: {}\n", self.event_type.as_str()));
        for line in self.data.lines() {
            out.push_str(&format!("data: {line}\n"));
        }
        if self.data.is_empty() {
            out.push_str("data: \n");
        }
        out.push('\n');
        out
    }
}

/// Stream state tracker.
#[derive(Debug)]
pub struct StreamState {
    pub tokens_sent: usize,
    pub bytes_sent: usize,
    pub started_at: Instant,
    pub finished: bool,
    pub error: Option<String>,
    events: Vec<SseEvent>,
}

impl Default for StreamState {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamState {
    pub fn new() -> Self {
        Self {
            tokens_sent: 0,
            bytes_sent: 0,
            started_at: Instant::now(),
            finished: false,
            error: None,
            events: Vec::new(),
        }
    }

    pub fn send_token(&mut self, text: &str) -> SseEvent {
        let event = SseEvent::token(text);
        self.tokens_sent += 1;
        self.bytes_sent += text.len();
        self.events.push(event.clone());
        event
    }

    pub fn finish(&mut self) -> SseEvent {
        self.finished = true;
        let event = SseEvent::done();
        self.events.push(event.clone());
        event
    }

    pub fn send_error(&mut self, msg: &str) -> SseEvent {
        self.error = Some(msg.to_string());
        self.finished = true;
        let event = SseEvent::error(msg);
        self.events.push(event.clone());
        event
    }

    pub fn elapsed_ms(&self) -> u128 {
        self.started_at.elapsed().as_millis()
    }

    pub fn tokens_per_second(&self) -> f64 {
        let secs = self.started_at.elapsed().as_secs_f64();
        if secs > 0.0 { self.tokens_sent as f64 / secs } else { 0.0 }
    }

    pub fn event_count(&self) -> usize {
        self.events.len()
    }
}

/// OpenAI-compatible chunk format.
pub fn format_chat_chunk(token: &str, model: &str, idx: usize) -> String {
    format!(
        r#"{{"id":"chatcmpl-{idx}","object":"chat.completion.chunk","model":"{model}","choices":[{{"index":0,"delta":{{"content":"{}"}},"finish_reason":null}}]}}"#,
        escape_json(token)
    )
}

pub fn format_chat_done(model: &str, idx: usize) -> String {
    format!(
        r#"{{"id":"chatcmpl-{idx}","object":"chat.completion.chunk","model":"{model}","choices":[{{"index":0,"delta":{{}},"finish_reason":"stop"}}]}}"#
    )
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_event() {
        let e = SseEvent::token("hello");
        assert_eq!(e.event_type, SseEventType::Token);
        let formatted = e.format();
        assert!(formatted.contains("event: token"));
        assert!(formatted.contains("data: hello"));
    }

    #[test]
    fn test_done_event() {
        let e = SseEvent::done();
        let formatted = e.format();
        assert!(formatted.contains("[DONE]"));
    }

    #[test]
    fn test_error_event() {
        let e = SseEvent::error("bad input");
        assert_eq!(e.event_type, SseEventType::Error);
    }

    #[test]
    fn test_heartbeat() {
        let e = SseEvent::heartbeat();
        let formatted = e.format();
        assert!(formatted.contains("event: heartbeat"));
    }

    #[test]
    fn test_stream_state() {
        let mut state = StreamState::new();
        state.send_token("hello");
        state.send_token(" world");
        assert_eq!(state.tokens_sent, 2);
        assert_eq!(state.bytes_sent, 11);
    }

    #[test]
    fn test_stream_finish() {
        let mut state = StreamState::new();
        state.send_token("hi");
        state.finish();
        assert!(state.finished);
        assert_eq!(state.event_count(), 2);
    }

    #[test]
    fn test_stream_error() {
        let mut state = StreamState::new();
        state.send_error("timeout");
        assert!(state.finished);
        assert!(state.error.is_some());
    }

    #[test]
    fn test_chat_chunk() {
        let chunk = format_chat_chunk("Hello", "phi-4", 1);
        assert!(chunk.contains("phi-4"));
        assert!(chunk.contains("Hello"));
        assert!(chunk.contains("chat.completion.chunk"));
    }

    #[test]
    fn test_chat_done() {
        let done = format_chat_done("phi-4", 1);
        assert!(done.contains("stop"));
    }

    #[test]
    fn test_escape_json() {
        let chunk = format_chat_chunk("line1\nline2", "m", 1);
        assert!(chunk.contains("\\n"));
    }

    #[test]
    fn test_sse_with_id() {
        let mut e = SseEvent::token("data");
        e.id = Some("42".into());
        let formatted = e.format();
        assert!(formatted.contains("id: 42"));
    }

    #[test]
    fn test_default_state() {
        let s = StreamState::default();
        assert_eq!(s.tokens_sent, 0);
        assert!(!s.finished);
    }
}
