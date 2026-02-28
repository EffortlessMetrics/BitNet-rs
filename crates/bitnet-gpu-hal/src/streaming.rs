//! Streaming token generation system for SSE/WebSocket delivery.

use std::fmt;

/// Unique identifier for a generation stream.
pub type StreamId = u64;

/// Manages concurrent streaming generation sessions.
pub struct StreamingGenerator {
    active_streams: Vec<GenerationStream>,
    config: StreamingConfig,
    next_id: StreamId,
}

/// Configuration for the streaming generator.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    pub max_concurrent_streams: usize,
    pub token_buffer_size: usize,
    pub flush_interval_ms: u64,
    pub heartbeat_interval_ms: u64,
}

/// A single generation stream that produces tokens over time.
pub struct GenerationStream {
    pub id: StreamId,
    pub state: StreamState,
    pub tokens: Vec<StreamToken>,
    pub created_at: u64,
    pub last_token_at: Option<u64>,
    pub finish_reason: Option<FinishReason>,
    pending_events: Vec<StreamEvent>,
}

/// State machine for a generation stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamState {
    Created,
    Generating,
    Paused,
    Completed,
    Error(String),
    Cancelled,
}

impl fmt::Display for StreamState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Created => write!(f, "Created"),
            Self::Generating => write!(f, "Generating"),
            Self::Paused => write!(f, "Paused"),
            Self::Completed => write!(f, "Completed"),
            Self::Error(e) => write!(f, "Error: {e}"),
            Self::Cancelled => write!(f, "Cancelled"),
        }
    }
}

/// A single generated token with metadata.
#[derive(Debug, Clone)]
pub struct StreamToken {
    pub token_id: u32,
    pub text: String,
    pub logprob: Option<f32>,
    pub timestamp_ms: u64,
    pub is_special: bool,
}

/// Reason a stream finished generating.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    MaxTokens,
    Error(String),
    Cancelled,
}

/// An event emitted by a generation stream.
#[derive(Debug, Clone)]
pub struct StreamEvent {
    pub stream_id: StreamId,
    pub event_type: StreamEventType,
    pub data: String,
}

/// Type of stream event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEventType {
    TokenGenerated,
    StreamStarted,
    StreamCompleted,
    StreamError,
    Heartbeat,
}

/// Formats stream events as Server-Sent Events.
pub struct SseFormatter;

/// Formats stream events as JSON lines (newline-delimited JSON).
pub struct JsonStreamFormatter;

/// Errors that can occur during streaming operations.
#[derive(Debug, PartialEq, Eq)]
pub enum StreamError {
    StreamNotFound(StreamId),
    MaxConcurrentStreams,
    InvalidStateTransition { from: StreamState, to: StreamState },
    BufferOverflow(StreamId),
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StreamNotFound(id) => {
                write!(f, "stream {id} not found")
            }
            Self::MaxConcurrentStreams => {
                write!(f, "maximum concurrent streams reached")
            }
            Self::InvalidStateTransition { from, to } => {
                write!(f, "invalid transition from {from} to {to}")
            }
            Self::BufferOverflow(id) => {
                write!(f, "token buffer overflow on stream {id}")
            }
        }
    }
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            max_concurrent_streams: 16,
            token_buffer_size: 1024,
            flush_interval_ms: 50,
            heartbeat_interval_ms: 15000,
        }
    }
}

impl StreamingGenerator {
    /// Create a new streaming generator with the given configuration.
    pub const fn new(config: StreamingConfig) -> Self {
        Self { active_streams: Vec::new(), config, next_id: 1 }
    }

    /// Create a new generation stream, returning its unique ID.
    pub fn create_stream(&mut self) -> Result<StreamId, StreamError> {
        let active = self.active_count();
        if active >= self.config.max_concurrent_streams {
            return Err(StreamError::MaxConcurrentStreams);
        }
        let id = self.next_id;
        self.next_id += 1;
        let mut stream = GenerationStream {
            id,
            state: StreamState::Created,
            tokens: Vec::new(),
            created_at: current_time_ms(),
            last_token_at: None,
            finish_reason: None,
            pending_events: Vec::new(),
        };
        stream.pending_events.push(StreamEvent {
            stream_id: id,
            event_type: StreamEventType::StreamStarted,
            data: format!("{{\"stream_id\":{id}}}"),
        });
        stream.state = StreamState::Generating;
        self.active_streams.push(stream);
        Ok(id)
    }

    /// Push a token into a stream's buffer.
    pub fn push_token(
        &mut self,
        stream_id: StreamId,
        token: StreamToken,
    ) -> Result<(), StreamError> {
        let buf_limit = self.config.token_buffer_size;
        let stream =
            self.find_stream_mut(stream_id).ok_or(StreamError::StreamNotFound(stream_id))?;

        match &stream.state {
            StreamState::Generating => {}
            other => {
                return Err(StreamError::InvalidStateTransition {
                    from: other.clone(),
                    to: StreamState::Generating,
                });
            }
        }

        if stream.tokens.len() >= buf_limit {
            return Err(StreamError::BufferOverflow(stream_id));
        }

        let data = if let Some(lp) = token.logprob {
            format!(
                "{{\"token\":\"{}\",\"token_id\":{},\"logprob\":{:.4},\
                 \"is_special\":{}}}",
                escape_json(&token.text),
                token.token_id,
                lp,
                token.is_special,
            )
        } else {
            format!(
                "{{\"token\":\"{}\",\"token_id\":{},\"is_special\":{}}}",
                escape_json(&token.text),
                token.token_id,
                token.is_special,
            )
        };

        stream.pending_events.push(StreamEvent {
            stream_id,
            event_type: StreamEventType::TokenGenerated,
            data,
        });

        stream.last_token_at = Some(token.timestamp_ms);
        stream.tokens.push(token);
        Ok(())
    }

    /// Poll pending events for a stream, draining the buffer.
    pub fn poll_events(&mut self, stream_id: StreamId) -> Result<Vec<StreamEvent>, StreamError> {
        let stream =
            self.find_stream_mut(stream_id).ok_or(StreamError::StreamNotFound(stream_id))?;
        let events: Vec<StreamEvent> = stream.pending_events.drain(..).collect();
        Ok(events)
    }

    /// Mark a stream as completed with the given finish reason.
    pub fn complete_stream(
        &mut self,
        stream_id: StreamId,
        reason: FinishReason,
    ) -> Result<(), StreamError> {
        let stream =
            self.find_stream_mut(stream_id).ok_or(StreamError::StreamNotFound(stream_id))?;

        match &stream.state {
            StreamState::Generating | StreamState::Paused => {}
            other => {
                return Err(StreamError::InvalidStateTransition {
                    from: other.clone(),
                    to: StreamState::Completed,
                });
            }
        }

        let reason_str = match &reason {
            FinishReason::Stop => "stop",
            FinishReason::MaxTokens => "max_tokens",
            FinishReason::Error(_) => "error",
            FinishReason::Cancelled => "cancelled",
        };

        stream.pending_events.push(StreamEvent {
            stream_id,
            event_type: StreamEventType::StreamCompleted,
            data: format!("{{\"stream_id\":{stream_id},\"finish_reason\":\"{reason_str}\"}}"),
        });

        stream.finish_reason = Some(reason);
        stream.state = StreamState::Completed;
        Ok(())
    }

    /// Cancel an active stream.
    pub fn cancel_stream(&mut self, stream_id: StreamId) -> Result<(), StreamError> {
        let stream =
            self.find_stream_mut(stream_id).ok_or(StreamError::StreamNotFound(stream_id))?;

        match &stream.state {
            StreamState::Generating | StreamState::Paused => {}
            other => {
                return Err(StreamError::InvalidStateTransition {
                    from: other.clone(),
                    to: StreamState::Cancelled,
                });
            }
        }

        stream.pending_events.push(StreamEvent {
            stream_id,
            event_type: StreamEventType::StreamCompleted,
            data: format!(
                "{{\"stream_id\":{stream_id},\
                 \"finish_reason\":\"cancelled\"}}"
            ),
        });

        stream.finish_reason = Some(FinishReason::Cancelled);
        stream.state = StreamState::Cancelled;
        Ok(())
    }

    /// Pause an active stream.
    pub fn pause_stream(&mut self, stream_id: StreamId) -> Result<(), StreamError> {
        let stream =
            self.find_stream_mut(stream_id).ok_or(StreamError::StreamNotFound(stream_id))?;

        match &stream.state {
            StreamState::Generating => {}
            other => {
                return Err(StreamError::InvalidStateTransition {
                    from: other.clone(),
                    to: StreamState::Paused,
                });
            }
        }

        stream.state = StreamState::Paused;
        Ok(())
    }

    /// Resume a paused stream.
    pub fn resume_stream(&mut self, stream_id: StreamId) -> Result<(), StreamError> {
        let stream =
            self.find_stream_mut(stream_id).ok_or(StreamError::StreamNotFound(stream_id))?;

        match &stream.state {
            StreamState::Paused => {}
            other => {
                return Err(StreamError::InvalidStateTransition {
                    from: other.clone(),
                    to: StreamState::Generating,
                });
            }
        }

        stream.state = StreamState::Generating;
        Ok(())
    }

    /// Set a stream into an error state.
    pub fn error_stream(
        &mut self,
        stream_id: StreamId,
        message: String,
    ) -> Result<(), StreamError> {
        let stream =
            self.find_stream_mut(stream_id).ok_or(StreamError::StreamNotFound(stream_id))?;

        match &stream.state {
            StreamState::Completed | StreamState::Error(_) | StreamState::Cancelled => {
                return Err(StreamError::InvalidStateTransition {
                    from: stream.state.clone(),
                    to: StreamState::Error(message),
                });
            }
            _ => {}
        }

        stream.pending_events.push(StreamEvent {
            stream_id,
            event_type: StreamEventType::StreamError,
            data: format!("{{\"stream_id\":{stream_id},\"error\":\"{}\"}}", escape_json(&message),),
        });

        stream.state = StreamState::Error(message);
        Ok(())
    }

    /// Generate a heartbeat event for the given stream.
    pub fn heartbeat(&mut self, stream_id: StreamId) -> Result<StreamEvent, StreamError> {
        let stream =
            self.find_stream_mut(stream_id).ok_or(StreamError::StreamNotFound(stream_id))?;

        let event = StreamEvent {
            stream_id,
            event_type: StreamEventType::Heartbeat,
            data: format!("{{\"stream_id\":{},\"timestamp\":{}}}", stream.id, current_time_ms(),),
        };

        stream.pending_events.push(event.clone());
        Ok(event)
    }

    /// Return the number of active (non-terminal) streams.
    pub fn active_count(&self) -> usize {
        self.active_streams
            .iter()
            .filter(|s| {
                matches!(
                    s.state,
                    StreamState::Created | StreamState::Generating | StreamState::Paused
                )
            })
            .count()
    }

    /// Get the state of a stream by ID.
    pub fn stream_state(&self, stream_id: StreamId) -> Option<StreamState> {
        self.active_streams.iter().find(|s| s.id == stream_id).map(|s| s.state.clone())
    }

    /// Check if a stream has timed out based on the heartbeat interval.
    pub fn is_stream_timed_out(&self, stream_id: StreamId, now_ms: u64) -> Option<bool> {
        self.active_streams.iter().find(|s| s.id == stream_id).map(|s| {
            let last_activity = s.last_token_at.unwrap_or(s.created_at);
            now_ms.saturating_sub(last_activity) > self.config.heartbeat_interval_ms
        })
    }

    /// Get the configuration.
    pub const fn config(&self) -> &StreamingConfig {
        &self.config
    }

    /// Get a reference to a stream by ID.
    pub fn get_stream(&self, stream_id: StreamId) -> Option<&GenerationStream> {
        self.active_streams.iter().find(|s| s.id == stream_id)
    }

    fn find_stream_mut(&mut self, stream_id: StreamId) -> Option<&mut GenerationStream> {
        self.active_streams.iter_mut().find(|s| s.id == stream_id)
    }
}

impl SseFormatter {
    /// Format a stream event as an SSE message.
    pub fn format(event: &StreamEvent) -> String {
        let event_name = match event.event_type {
            StreamEventType::TokenGenerated => "token",
            StreamEventType::StreamStarted => "stream_start",
            StreamEventType::StreamCompleted => "stream_complete",
            StreamEventType::StreamError => "error",
            StreamEventType::Heartbeat => "heartbeat",
        };
        format!("event: {event_name}\ndata: {}\n\n", event.data)
    }
}

impl JsonStreamFormatter {
    /// Format a stream event as a JSON line.
    pub fn format(event: &StreamEvent) -> String {
        let event_type = match event.event_type {
            StreamEventType::TokenGenerated => "token_generated",
            StreamEventType::StreamStarted => "stream_started",
            StreamEventType::StreamCompleted => "stream_completed",
            StreamEventType::StreamError => "stream_error",
            StreamEventType::Heartbeat => "heartbeat",
        };
        format!(
            "{{\"stream_id\":{},\"event\":\"{event_type}\",\
             \"data\":{}}}\n",
            event.stream_id, event.data,
        )
    }
}

fn current_time_ms() -> u64 {
    u64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis(),
    )
    .unwrap_or(u64::MAX)
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

    fn default_config() -> StreamingConfig {
        StreamingConfig::default()
    }

    fn make_token(text: &str, id: u32) -> StreamToken {
        StreamToken {
            token_id: id,
            text: text.to_string(),
            logprob: None,
            timestamp_ms: current_time_ms(),
            is_special: false,
        }
    }

    fn small_config() -> StreamingConfig {
        StreamingConfig {
            max_concurrent_streams: 2,
            token_buffer_size: 4,
            flush_interval_ms: 10,
            heartbeat_interval_ms: 100,
        }
    }

    // ── Stream lifecycle ────────────────────────────────────────

    #[test]
    fn test_create_stream() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        assert_eq!(id, 1);
        assert_eq!(sg.stream_state(id), Some(StreamState::Generating));
    }

    #[test]
    fn test_create_multiple_streams() {
        let mut sg = StreamingGenerator::new(default_config());
        let id1 = sg.create_stream().unwrap();
        let id2 = sg.create_stream().unwrap();
        let id3 = sg.create_stream().unwrap();
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
        assert_eq!(sg.active_count(), 3);
    }

    #[test]
    fn test_complete_stream_lifecycle() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();

        sg.push_token(id, make_token("hello", 1)).unwrap();
        sg.push_token(id, make_token(" world", 2)).unwrap();
        sg.complete_stream(id, FinishReason::Stop).unwrap();

        assert_eq!(sg.stream_state(id), Some(StreamState::Completed));
    }

    #[test]
    fn test_complete_stream_max_tokens() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        sg.complete_stream(id, FinishReason::MaxTokens).unwrap();
        let stream = sg.get_stream(id).unwrap();
        assert_eq!(stream.finish_reason, Some(FinishReason::MaxTokens));
    }

    #[test]
    fn test_complete_stream_error_reason() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        sg.complete_stream(id, FinishReason::Error("oom".into())).unwrap();
        assert_eq!(sg.stream_state(id), Some(StreamState::Completed));
    }

    // ── Push tokens & poll events ───────────────────────────────

    #[test]
    fn test_push_token_and_poll() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        sg.push_token(id, make_token("hello", 42)).unwrap();

        let events = sg.poll_events(id).unwrap();
        // StreamStarted + TokenGenerated
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event_type, StreamEventType::StreamStarted);
        assert_eq!(events[1].event_type, StreamEventType::TokenGenerated);
    }

    #[test]
    fn test_push_multiple_tokens() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        sg.push_token(id, make_token("a", 1)).unwrap();
        sg.push_token(id, make_token("b", 2)).unwrap();
        sg.push_token(id, make_token("c", 3)).unwrap();

        let events = sg.poll_events(id).unwrap();
        assert_eq!(events.len(), 4); // start + 3 tokens

        let stream = sg.get_stream(id).unwrap();
        assert_eq!(stream.tokens.len(), 3);
    }

    #[test]
    fn test_poll_clears_events() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        sg.push_token(id, make_token("x", 1)).unwrap();

        let first = sg.poll_events(id).unwrap();
        assert_eq!(first.len(), 2);

        let second = sg.poll_events(id).unwrap();
        assert!(second.is_empty());
    }

    #[test]
    fn test_empty_poll_returns_no_events() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        // Drain the start event
        let _ = sg.poll_events(id).unwrap();
        let events = sg.poll_events(id).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn test_push_token_nonexistent_stream() {
        let mut sg = StreamingGenerator::new(default_config());
        let result = sg.push_token(999, make_token("x", 1));
        assert_eq!(result.unwrap_err(), StreamError::StreamNotFound(999));
    }

    // ── SSE format ──────────────────────────────────────────────

    #[test]
    fn test_sse_format_token() {
        let event = StreamEvent {
            stream_id: 1,
            event_type: StreamEventType::TokenGenerated,
            data: r#"{"token":"hello"}"#.to_string(),
        };
        let sse = SseFormatter::format(&event);
        assert_eq!(sse, "event: token\ndata: {\"token\":\"hello\"}\n\n");
    }

    #[test]
    fn test_sse_format_stream_start() {
        let event = StreamEvent {
            stream_id: 1,
            event_type: StreamEventType::StreamStarted,
            data: r#"{"stream_id":1}"#.to_string(),
        };
        let sse = SseFormatter::format(&event);
        assert!(sse.starts_with("event: stream_start\n"));
        assert!(sse.ends_with("\n\n"));
    }

    #[test]
    fn test_sse_format_stream_complete() {
        let event = StreamEvent {
            stream_id: 1,
            event_type: StreamEventType::StreamCompleted,
            data: r#"{"finish_reason":"stop"}"#.to_string(),
        };
        let sse = SseFormatter::format(&event);
        assert!(sse.contains("event: stream_complete\n"));
    }

    #[test]
    fn test_sse_format_error() {
        let event = StreamEvent {
            stream_id: 1,
            event_type: StreamEventType::StreamError,
            data: r#"{"error":"timeout"}"#.to_string(),
        };
        let sse = SseFormatter::format(&event);
        assert!(sse.starts_with("event: error\n"));
    }

    #[test]
    fn test_sse_format_heartbeat() {
        let event = StreamEvent {
            stream_id: 1,
            event_type: StreamEventType::Heartbeat,
            data: r#"{"stream_id":1,"timestamp":123}"#.to_string(),
        };
        let sse = SseFormatter::format(&event);
        assert!(sse.starts_with("event: heartbeat\n"));
        assert!(sse.ends_with("\n\n"));
    }

    #[test]
    fn test_sse_double_newline_terminator() {
        let event = StreamEvent {
            stream_id: 1,
            event_type: StreamEventType::TokenGenerated,
            data: "{}".to_string(),
        };
        let sse = SseFormatter::format(&event);
        assert!(sse.ends_with("\n\n"));
        // Must not end with three newlines
        assert!(!sse.ends_with("\n\n\n"));
    }

    // ── JSON lines format ───────────────────────────────────────

    #[test]
    fn test_json_lines_format_token() {
        let event = StreamEvent {
            stream_id: 42,
            event_type: StreamEventType::TokenGenerated,
            data: r#"{"token":"world"}"#.to_string(),
        };
        let line = JsonStreamFormatter::format(&event);
        assert!(line.contains("\"stream_id\":42"));
        assert!(line.contains("\"event\":\"token_generated\""));
        assert!(line.ends_with('\n'));
    }

    #[test]
    fn test_json_lines_format_started() {
        let event = StreamEvent {
            stream_id: 1,
            event_type: StreamEventType::StreamStarted,
            data: r#"{"stream_id":1}"#.to_string(),
        };
        let line = JsonStreamFormatter::format(&event);
        assert!(line.contains("\"event\":\"stream_started\""));
    }

    #[test]
    fn test_json_lines_format_completed() {
        let event = StreamEvent {
            stream_id: 1,
            event_type: StreamEventType::StreamCompleted,
            data: r#"{"finish":"stop"}"#.to_string(),
        };
        let line = JsonStreamFormatter::format(&event);
        assert!(line.contains("\"event\":\"stream_completed\""));
    }

    #[test]
    fn test_json_lines_format_error() {
        let event = StreamEvent {
            stream_id: 1,
            event_type: StreamEventType::StreamError,
            data: r#"{"error":"bad"}"#.to_string(),
        };
        let line = JsonStreamFormatter::format(&event);
        assert!(line.contains("\"event\":\"stream_error\""));
    }

    #[test]
    fn test_json_lines_format_heartbeat() {
        let event = StreamEvent {
            stream_id: 5,
            event_type: StreamEventType::Heartbeat,
            data: r#"{"ts":100}"#.to_string(),
        };
        let line = JsonStreamFormatter::format(&event);
        assert!(line.contains("\"stream_id\":5"));
        assert!(line.contains("\"event\":\"heartbeat\""));
    }

    #[test]
    fn test_json_lines_single_newline() {
        let event = StreamEvent {
            stream_id: 1,
            event_type: StreamEventType::TokenGenerated,
            data: "{}".to_string(),
        };
        let line = JsonStreamFormatter::format(&event);
        assert!(line.ends_with('\n'));
        assert!(!line.ends_with("\n\n"));
    }

    // ── Stream state transitions ────────────────────────────────

    #[test]
    fn test_state_created_to_generating() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        // create_stream auto-transitions to Generating
        assert_eq!(sg.stream_state(id), Some(StreamState::Generating));
    }

    #[test]
    fn test_state_generating_to_completed() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        sg.complete_stream(id, FinishReason::Stop).unwrap();
        assert_eq!(sg.stream_state(id), Some(StreamState::Completed));
    }

    #[test]
    fn test_state_generating_to_cancelled() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        sg.cancel_stream(id).unwrap();
        assert_eq!(sg.stream_state(id), Some(StreamState::Cancelled));
    }

    #[test]
    fn test_state_generating_to_paused() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        sg.pause_stream(id).unwrap();
        assert_eq!(sg.stream_state(id), Some(StreamState::Paused));
    }

    #[test]
    fn test_state_paused_to_generating() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        sg.pause_stream(id).unwrap();
        sg.resume_stream(id).unwrap();
        assert_eq!(sg.stream_state(id), Some(StreamState::Generating));
    }

    #[test]
    fn test_state_paused_to_completed() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        sg.pause_stream(id).unwrap();
        sg.complete_stream(id, FinishReason::Stop).unwrap();
        assert_eq!(sg.stream_state(id), Some(StreamState::Completed));
    }

    #[test]
    fn test_state_paused_to_cancelled() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        sg.pause_stream(id).unwrap();
        sg.cancel_stream(id).unwrap();
        assert_eq!(sg.stream_state(id), Some(StreamState::Cancelled));
    }

    #[test]
    fn test_cannot_resume_completed_stream() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        sg.complete_stream(id, FinishReason::Stop).unwrap();
        let err = sg.resume_stream(id).unwrap_err();
        assert!(matches!(err, StreamError::InvalidStateTransition { .. }));
    }

    #[test]
    fn test_cannot_pause_completed_stream() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        sg.complete_stream(id, FinishReason::Stop).unwrap();
        let err = sg.pause_stream(id).unwrap_err();
        assert!(matches!(err, StreamError::InvalidStateTransition { .. }));
    }

    #[test]
    fn test_cannot_complete_already_completed() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        sg.complete_stream(id, FinishReason::Stop).unwrap();
        let err = sg.complete_stream(id, FinishReason::Stop).unwrap_err();
        assert!(matches!(err, StreamError::InvalidStateTransition { .. }));
    }

    #[test]
    fn test_cannot_cancel_completed_stream() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        sg.complete_stream(id, FinishReason::Stop).unwrap();
        let err = sg.cancel_stream(id).unwrap_err();
        assert!(matches!(err, StreamError::InvalidStateTransition { .. }));
    }

    #[test]
    fn test_cannot_push_to_paused_stream() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        sg.pause_stream(id).unwrap();
        let err = sg.push_token(id, make_token("x", 1)).unwrap_err();
        assert!(matches!(err, StreamError::InvalidStateTransition { .. }));
    }

    #[test]
    fn test_cannot_push_to_completed_stream() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        sg.complete_stream(id, FinishReason::Stop).unwrap();
        let err = sg.push_token(id, make_token("x", 1)).unwrap_err();
        assert!(matches!(err, StreamError::InvalidStateTransition { .. }));
    }

    // ── Pause / resume ──────────────────────────────────────────

    #[test]
    fn test_pause_and_resume_preserves_tokens() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        sg.push_token(id, make_token("a", 1)).unwrap();
        sg.pause_stream(id).unwrap();
        sg.resume_stream(id).unwrap();
        sg.push_token(id, make_token("b", 2)).unwrap();

        let stream = sg.get_stream(id).unwrap();
        assert_eq!(stream.tokens.len(), 2);
    }

    #[test]
    fn test_pause_resume_cycle() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        for _ in 0..5 {
            sg.pause_stream(id).unwrap();
            sg.resume_stream(id).unwrap();
        }
        assert_eq!(sg.stream_state(id), Some(StreamState::Generating));
    }

    // ── Cancel mid-generation ───────────────────────────────────

    #[test]
    fn test_cancel_mid_generation() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        sg.push_token(id, make_token("hello", 1)).unwrap();
        sg.push_token(id, make_token(" world", 2)).unwrap();
        sg.cancel_stream(id).unwrap();

        assert_eq!(sg.stream_state(id), Some(StreamState::Cancelled));
        let stream = sg.get_stream(id).unwrap();
        assert_eq!(stream.finish_reason, Some(FinishReason::Cancelled));
        assert_eq!(stream.tokens.len(), 2);
    }

    #[test]
    fn test_cancel_emits_event() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        let _ = sg.poll_events(id).unwrap(); // drain start
        sg.cancel_stream(id).unwrap();
        let events = sg.poll_events(id).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, StreamEventType::StreamCompleted);
        assert!(events[0].data.contains("cancelled"));
    }

    // ── Max concurrent streams ──────────────────────────────────

    #[test]
    fn test_max_concurrent_streams_limit() {
        let mut sg = StreamingGenerator::new(small_config());
        sg.create_stream().unwrap();
        sg.create_stream().unwrap();
        let err = sg.create_stream().unwrap_err();
        assert_eq!(err, StreamError::MaxConcurrentStreams);
    }

    #[test]
    fn test_completed_stream_frees_slot() {
        let mut sg = StreamingGenerator::new(small_config());
        let id1 = sg.create_stream().unwrap();
        sg.create_stream().unwrap();
        sg.complete_stream(id1, FinishReason::Stop).unwrap();
        // Should succeed because id1 is now completed
        let id3 = sg.create_stream().unwrap();
        assert!(id3 > 0);
    }

    #[test]
    fn test_cancelled_stream_frees_slot() {
        let mut sg = StreamingGenerator::new(small_config());
        let id1 = sg.create_stream().unwrap();
        sg.create_stream().unwrap();
        sg.cancel_stream(id1).unwrap();
        assert!(sg.create_stream().is_ok());
    }

    // ── Heartbeat ───────────────────────────────────────────────

    #[test]
    fn test_heartbeat_generation() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        let hb = sg.heartbeat(id).unwrap();
        assert_eq!(hb.event_type, StreamEventType::Heartbeat);
        assert_eq!(hb.stream_id, id);
        assert!(hb.data.contains("timestamp"));
    }

    #[test]
    fn test_heartbeat_appears_in_poll() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        let _ = sg.poll_events(id).unwrap(); // drain
        sg.heartbeat(id).unwrap();
        let events = sg.poll_events(id).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, StreamEventType::Heartbeat);
    }

    #[test]
    fn test_heartbeat_nonexistent_stream() {
        let mut sg = StreamingGenerator::new(default_config());
        let err = sg.heartbeat(999).unwrap_err();
        assert_eq!(err, StreamError::StreamNotFound(999));
    }

    // ── Multiple concurrent streams ─────────────────────────────

    #[test]
    fn test_concurrent_streams_independent() {
        let mut sg = StreamingGenerator::new(default_config());
        let id1 = sg.create_stream().unwrap();
        let id2 = sg.create_stream().unwrap();

        sg.push_token(id1, make_token("a", 1)).unwrap();
        sg.push_token(id2, make_token("x", 10)).unwrap();
        sg.push_token(id2, make_token("y", 11)).unwrap();

        let s1 = sg.get_stream(id1).unwrap();
        let s2 = sg.get_stream(id2).unwrap();
        assert_eq!(s1.tokens.len(), 1);
        assert_eq!(s2.tokens.len(), 2);
    }

    #[test]
    fn test_concurrent_streams_independent_state() {
        let mut sg = StreamingGenerator::new(default_config());
        let id1 = sg.create_stream().unwrap();
        let id2 = sg.create_stream().unwrap();

        sg.pause_stream(id1).unwrap();
        assert_eq!(sg.stream_state(id1), Some(StreamState::Paused));
        assert_eq!(sg.stream_state(id2), Some(StreamState::Generating));
    }

    // ── Token logprob inclusion ─────────────────────────────────

    #[test]
    fn test_token_logprob_included() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        let token = StreamToken {
            token_id: 5,
            text: "hi".to_string(),
            logprob: Some(-0.5),
            timestamp_ms: 1000,
            is_special: false,
        };
        sg.push_token(id, token).unwrap();
        let _ = sg.poll_events(id).unwrap();
        let stream = sg.get_stream(id).unwrap();
        assert_eq!(stream.tokens[0].logprob, Some(-0.5));
    }

    #[test]
    fn test_token_logprob_in_event_data() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        let _ = sg.poll_events(id).unwrap(); // drain start
        let token = StreamToken {
            token_id: 5,
            text: "hi".to_string(),
            logprob: Some(-1.23),
            timestamp_ms: 1000,
            is_special: false,
        };
        sg.push_token(id, token).unwrap();
        let events = sg.poll_events(id).unwrap();
        assert!(events[0].data.contains("logprob"));
    }

    #[test]
    fn test_token_no_logprob_omitted() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        let _ = sg.poll_events(id).unwrap();
        sg.push_token(id, make_token("hi", 1)).unwrap();
        let events = sg.poll_events(id).unwrap();
        assert!(!events[0].data.contains("logprob"));
    }

    // ── Special token handling ──────────────────────────────────

    #[test]
    fn test_special_token_flag() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        let _ = sg.poll_events(id).unwrap();
        let token = StreamToken {
            token_id: 0,
            text: "<eos>".to_string(),
            logprob: None,
            timestamp_ms: 1000,
            is_special: true,
        };
        sg.push_token(id, token).unwrap();
        let events = sg.poll_events(id).unwrap();
        assert!(events[0].data.contains("\"is_special\":true"));
    }

    #[test]
    fn test_non_special_token_flag() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        let _ = sg.poll_events(id).unwrap();
        sg.push_token(id, make_token("word", 10)).unwrap();
        let events = sg.poll_events(id).unwrap();
        assert!(events[0].data.contains("\"is_special\":false"));
    }

    // ── Stream timeout detection ────────────────────────────────

    #[test]
    fn test_stream_not_timed_out() {
        let mut sg = StreamingGenerator::new(small_config());
        let id = sg.create_stream().unwrap();
        let now = current_time_ms();
        assert_eq!(sg.is_stream_timed_out(id, now), Some(false));
    }

    #[test]
    fn test_stream_timed_out() {
        let mut sg = StreamingGenerator::new(small_config());
        let id = sg.create_stream().unwrap();
        // heartbeat_interval_ms = 100; simulate far future
        let stream = sg.find_stream_mut(id).unwrap();
        stream.created_at = 1000;
        stream.last_token_at = None;
        assert_eq!(sg.is_stream_timed_out(id, 1200), Some(true));
    }

    #[test]
    fn test_timeout_uses_last_token_time() {
        let mut sg = StreamingGenerator::new(small_config());
        let id = sg.create_stream().unwrap();
        let stream = sg.find_stream_mut(id).unwrap();
        stream.created_at = 1000;
        stream.last_token_at = Some(1150);
        // 1200 - 1150 = 50 < 100 (heartbeat interval)
        assert_eq!(sg.is_stream_timed_out(id, 1200), Some(false));
    }

    #[test]
    fn test_timeout_nonexistent_stream() {
        let sg = StreamingGenerator::new(default_config());
        assert_eq!(sg.is_stream_timed_out(999, 1000), None);
    }

    // ── Error state ─────────────────────────────────────────────

    #[test]
    fn test_error_state() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        sg.error_stream(id, "gpu fault".into()).unwrap();
        assert_eq!(sg.stream_state(id), Some(StreamState::Error("gpu fault".into())));
    }

    #[test]
    fn test_error_emits_event() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        let _ = sg.poll_events(id).unwrap();
        sg.error_stream(id, "timeout".into()).unwrap();
        let events = sg.poll_events(id).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, StreamEventType::StreamError);
        assert!(events[0].data.contains("timeout"));
    }

    #[test]
    fn test_cannot_error_completed_stream() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        sg.complete_stream(id, FinishReason::Stop).unwrap();
        let err = sg.error_stream(id, "late error".into()).unwrap_err();
        assert!(matches!(err, StreamError::InvalidStateTransition { .. }));
    }

    #[test]
    fn test_cannot_push_to_errored_stream() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        sg.error_stream(id, "broken".into()).unwrap();
        let err = sg.push_token(id, make_token("x", 1)).unwrap_err();
        assert!(matches!(err, StreamError::InvalidStateTransition { .. }));
    }

    // ── Buffer overflow ─────────────────────────────────────────

    #[test]
    fn test_buffer_overflow() {
        let mut sg = StreamingGenerator::new(small_config());
        let id = sg.create_stream().unwrap();
        for i in 0..4 {
            sg.push_token(id, make_token("t", i)).unwrap();
        }
        let err = sg.push_token(id, make_token("t", 5)).unwrap_err();
        assert_eq!(err, StreamError::BufferOverflow(id));
    }

    #[test]
    fn test_buffer_exactly_at_limit() {
        let mut sg = StreamingGenerator::new(small_config());
        let id = sg.create_stream().unwrap();
        for i in 0..4 {
            sg.push_token(id, make_token("t", i)).unwrap();
        }
        let stream = sg.get_stream(id).unwrap();
        assert_eq!(stream.tokens.len(), 4);
    }

    // ── Flush interval ──────────────────────────────────────────

    #[test]
    fn test_flush_interval_config() {
        let cfg = StreamingConfig { flush_interval_ms: 25, ..StreamingConfig::default() };
        let sg = StreamingGenerator::new(cfg);
        assert_eq!(sg.config().flush_interval_ms, 25);
    }

    #[test]
    fn test_default_flush_interval() {
        let cfg = StreamingConfig::default();
        assert_eq!(cfg.flush_interval_ms, 50);
    }

    // ── Config defaults ─────────────────────────────────────────

    #[test]
    fn test_default_config_values() {
        let cfg = StreamingConfig::default();
        assert_eq!(cfg.max_concurrent_streams, 16);
        assert_eq!(cfg.token_buffer_size, 1024);
        assert_eq!(cfg.flush_interval_ms, 50);
        assert_eq!(cfg.heartbeat_interval_ms, 15000);
    }

    // ── Active count ────────────────────────────────────────────

    #[test]
    fn test_active_count_empty() {
        let sg = StreamingGenerator::new(default_config());
        assert_eq!(sg.active_count(), 0);
    }

    #[test]
    fn test_active_count_with_completed() {
        let mut sg = StreamingGenerator::new(default_config());
        let id1 = sg.create_stream().unwrap();
        sg.create_stream().unwrap();
        sg.complete_stream(id1, FinishReason::Stop).unwrap();
        assert_eq!(sg.active_count(), 1);
    }

    #[test]
    fn test_active_count_includes_paused() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        sg.pause_stream(id).unwrap();
        assert_eq!(sg.active_count(), 1);
    }

    #[test]
    fn test_active_count_excludes_errored() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        sg.error_stream(id, "fail".into()).unwrap();
        assert_eq!(sg.active_count(), 0);
    }

    // ── stream_state ────────────────────────────────────────────

    #[test]
    fn test_stream_state_nonexistent() {
        let sg = StreamingGenerator::new(default_config());
        assert_eq!(sg.stream_state(999), None);
    }

    // ── Edge cases ──────────────────────────────────────────────

    #[test]
    fn test_escape_json_in_token_text() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        let _ = sg.poll_events(id).unwrap();
        let token = StreamToken {
            token_id: 1,
            text: "he said \"hi\"".to_string(),
            logprob: None,
            timestamp_ms: 1000,
            is_special: false,
        };
        sg.push_token(id, token).unwrap();
        let events = sg.poll_events(id).unwrap();
        assert!(events[0].data.contains(r#"he said \"hi\""#));
    }

    #[test]
    fn test_empty_token_text() {
        let mut sg = StreamingGenerator::new(default_config());
        let id = sg.create_stream().unwrap();
        sg.push_token(id, make_token("", 0)).unwrap();
        let stream = sg.get_stream(id).unwrap();
        assert_eq!(stream.tokens[0].text, "");
    }

    #[test]
    fn test_stream_error_display() {
        let err = StreamError::StreamNotFound(42);
        assert_eq!(format!("{err}"), "stream 42 not found");
    }

    #[test]
    fn test_stream_error_max_display() {
        let err = StreamError::MaxConcurrentStreams;
        assert_eq!(format!("{err}"), "maximum concurrent streams reached");
    }

    #[test]
    fn test_stream_state_display() {
        assert_eq!(format!("{}", StreamState::Created), "Created");
        assert_eq!(format!("{}", StreamState::Generating), "Generating");
        assert_eq!(format!("{}", StreamState::Error("x".into())), "Error: x");
    }
}
