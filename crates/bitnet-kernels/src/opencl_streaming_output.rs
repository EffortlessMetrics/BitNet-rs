//! Streaming token output for Intel Arc A770 (OpenCL backend).
//!
//! Manages token-by-token output delivery with buffering, SSE/WebSocket
//! formatting, and backpressure handling for real-time inference. All
//! implementations are CPU reference paths — actual OpenCL device integration
//! is deferred to a future hardware-specific pass.

use std::collections::VecDeque;
use std::fmt;
use std::time::Instant;

// ── StreamFormat ───────────────────────────────────────────────────

/// Wire format used to serialise streaming events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamFormat {
    /// Bare token ids / text with no framing.
    RawTokens,
    /// Server-Sent Events (`text/event-stream`).
    SSE,
    /// WebSocket text frames (JSON payloads).
    WebSocket,
    /// Newline-delimited JSON.
    NDJSON,
    /// OpenAI-compatible `data: {…}\n\n` streaming format.
    OpenAICompat,
}

impl fmt::Display for StreamFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::RawTokens => "RawTokens",
            Self::SSE => "SSE",
            Self::WebSocket => "WebSocket",
            Self::NDJSON => "NDJSON",
            Self::OpenAICompat => "OpenAICompat",
        };
        write!(f, "{label}")
    }
}

// ── StreamConfig ──────────────────────────────────────────────────

/// Configuration for the streaming output pipeline.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Wire format.
    pub format: StreamFormat,
    /// Maximum number of events buffered before a flush is required.
    pub buffer_size: usize,
    /// Target flush interval in milliseconds.
    pub flush_interval_ms: u64,
    /// Whether to include log-probabilities in each event.
    pub include_logprobs: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            format: StreamFormat::SSE,
            buffer_size: 32,
            flush_interval_ms: 50,
            include_logprobs: false,
        }
    }
}

// ── StreamEvent ───────────────────────────────────────────────────

/// A single token-level event emitted by the inference loop.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamEvent {
    /// Vocabulary token id.
    pub token_id: u32,
    /// Decoded text fragment.
    pub text: String,
    /// Optional log-probability of the selected token.
    pub logprob: Option<f32>,
    /// Reason the stream ended, if applicable.
    pub finish_reason: Option<FinishReason>,
    /// Wall-clock timestamp in milliseconds since some epoch.
    pub timestamp_ms: u64,
}

/// Why the generation stream terminated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FinishReason {
    /// `max_tokens` limit reached.
    Length,
    /// EOS / stop token encountered.
    Stop,
    /// Client cancelled.
    Cancelled,
}

impl fmt::Display for FinishReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Length => "length",
            Self::Stop => "stop",
            Self::Cancelled => "cancelled",
        };
        write!(f, "{s}")
    }
}

// ── BackpressurePolicy ────────────────────────────────────────────

/// Strategy applied when the stream buffer is full and a new event arrives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackpressurePolicy {
    /// Discard the oldest buffered event to make room.
    DropOldest,
    /// Discard the incoming (newest) event.
    DropNewest,
    /// Block the producer until space is available (simulated via flag).
    Block,
    /// Double the buffer capacity.
    Resize,
}

// ── StreamBuffer ──────────────────────────────────────────────────

/// Ring buffer that accumulates [`StreamEvent`]s until a flush threshold.
#[derive(Debug)]
pub struct StreamBuffer {
    events: VecDeque<StreamEvent>,
    capacity: usize,
    policy: BackpressurePolicy,
    last_flush: Instant,
    flush_interval_ms: u64,
    dropped_count: u64,
}

impl StreamBuffer {
    /// Create a new buffer with the given capacity, backpressure policy and
    /// flush interval.
    pub fn new(capacity: usize, policy: BackpressurePolicy, flush_interval_ms: u64) -> Self {
        Self {
            events: VecDeque::with_capacity(capacity),
            capacity,
            policy,
            last_flush: Instant::now(),
            flush_interval_ms,
            dropped_count: 0,
        }
    }

    /// Build a buffer directly from a [`StreamConfig`] and policy.
    pub fn from_config(config: &StreamConfig, policy: BackpressurePolicy) -> Self {
        Self::new(config.buffer_size, policy, config.flush_interval_ms)
    }

    /// Push an event into the buffer, applying the backpressure policy when
    /// the buffer is full.
    ///
    /// Returns `true` if the event was accepted, `false` if it was dropped
    /// (only possible with [`BackpressurePolicy::DropNewest`]).
    pub fn push(&mut self, event: StreamEvent) -> bool {
        if self.events.len() < self.capacity {
            self.events.push_back(event);
            return true;
        }

        match self.policy {
            BackpressurePolicy::DropOldest => {
                self.events.pop_front();
                self.dropped_count += 1;
                self.events.push_back(event);
                true
            }
            BackpressurePolicy::DropNewest => {
                self.dropped_count += 1;
                false
            }
            BackpressurePolicy::Block => {
                // In a real async runtime this would await a notify /
                // semaphore. For the CPU reference we simply grow to accept
                // the event — no data is lost.
                self.events.push_back(event);
                true
            }
            BackpressurePolicy::Resize => {
                self.capacity *= 2;
                self.events.push_back(event);
                true
            }
        }
    }

    /// Returns `true` when a flush should happen — either the buffer is full
    /// or the flush interval has elapsed.
    pub fn should_flush(&self) -> bool {
        if self.events.is_empty() {
            return false;
        }
        self.events.len() >= self.capacity
            || self.last_flush.elapsed().as_millis() as u64 >= self.flush_interval_ms
    }

    /// Drain all buffered events, resetting the flush timer.
    pub fn flush(&mut self) -> Vec<StreamEvent> {
        self.last_flush = Instant::now();
        self.events.drain(..).collect()
    }

    /// Number of events currently buffered.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Current capacity (may grow under [`BackpressurePolicy::Resize`]).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Total events dropped since creation.
    pub fn dropped_count(&self) -> u64 {
        self.dropped_count
    }
}

// ── StreamFormatter ───────────────────────────────────────────────

/// Formats [`StreamEvent`]s into the chosen wire format.
pub struct StreamFormatter {
    format: StreamFormat,
    include_logprobs: bool,
}

impl StreamFormatter {
    pub fn new(format: StreamFormat, include_logprobs: bool) -> Self {
        Self { format, include_logprobs }
    }

    pub fn from_config(config: &StreamConfig) -> Self {
        Self::new(config.format, config.include_logprobs)
    }

    /// Render a single event to the configured wire format.
    pub fn format_event(&self, event: &StreamEvent) -> String {
        match self.format {
            StreamFormat::RawTokens => self.format_raw(event),
            StreamFormat::SSE => self.format_sse(event),
            StreamFormat::WebSocket => self.format_websocket(event),
            StreamFormat::NDJSON => self.format_ndjson(event),
            StreamFormat::OpenAICompat => self.format_openai(event),
        }
    }

    /// Render a batch of events.
    pub fn format_batch(&self, events: &[StreamEvent]) -> String {
        events.iter().map(|e| self.format_event(e)).collect::<Vec<_>>().join("")
    }

    // ── private helpers ────────────────────────────────────────────

    fn format_raw(&self, event: &StreamEvent) -> String {
        event.text.clone()
    }

    fn format_sse(&self, event: &StreamEvent) -> String {
        let mut out = String::new();
        if let Some(reason) = &event.finish_reason {
            out.push_str(&format!("event: done\ndata: {{\"finish_reason\":\"{reason}\"}}\n\n"));
        } else {
            let data = self.json_payload(event);
            out.push_str(&format!("data: {data}\n\n"));
        }
        out
    }

    fn format_websocket(&self, event: &StreamEvent) -> String {
        let payload = self.json_payload(event);
        format!("{payload}\n")
    }

    fn format_ndjson(&self, event: &StreamEvent) -> String {
        let payload = self.json_payload(event);
        format!("{payload}\n")
    }

    fn format_openai(&self, event: &StreamEvent) -> String {
        if let Some(reason) = &event.finish_reason {
            return format!(
                "data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"{reason}\"}}]}}\n\n\
                 data: [DONE]\n\n"
            );
        }
        let mut delta = format!("\"content\":\"{}\"", escape_json_str(&event.text));
        if self.include_logprobs
            && let Some(lp) = event.logprob
        {
            delta.push_str(&format!(",\"logprob\":{lp}"));
        }
        format!("data: {{\"choices\":[{{\"delta\":{{{delta}}},\"finish_reason\":null}}]}}\n\n")
    }

    fn json_payload(&self, event: &StreamEvent) -> String {
        let mut parts: Vec<String> = vec![
            format!("\"token_id\":{}", event.token_id),
            format!("\"text\":\"{}\"", escape_json_str(&event.text)),
            format!("\"timestamp_ms\":{}", event.timestamp_ms),
        ];
        if self.include_logprobs
            && let Some(lp) = event.logprob
        {
            parts.push(format!("\"logprob\":{lp}"));
        }
        if let Some(reason) = &event.finish_reason {
            parts.push(format!("\"finish_reason\":\"{reason}\""));
        }
        format!("{{{}}}", parts.join(","))
    }
}

// ── StreamMetrics ─────────────────────────────────────────────────

/// Delivery-level metrics accumulated during a streaming session.
#[derive(Debug, Clone, Default)]
pub struct StreamMetrics {
    /// Total tokens sent to the client.
    pub tokens_sent: u64,
    /// Total bytes sent (after formatting).
    pub bytes_sent: u64,
    /// Running sum of per-token latency in milliseconds (divide by
    /// `tokens_sent` to get the average).
    pub total_latency_ms: f64,
    /// Number of events dropped due to backpressure.
    pub dropped_count: u64,
}

impl StreamMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successfully sent event with its formatted byte size and
    /// delivery latency.
    pub fn record_send(&mut self, bytes: u64, latency_ms: f64) {
        self.tokens_sent += 1;
        self.bytes_sent += bytes;
        self.total_latency_ms += latency_ms;
    }

    /// Record a dropped event.
    pub fn record_drop(&mut self) {
        self.dropped_count += 1;
    }

    /// Average per-token latency, or `0.0` when no tokens have been sent.
    pub fn avg_latency_ms(&self) -> f64 {
        if self.tokens_sent == 0 { 0.0 } else { self.total_latency_ms / self.tokens_sent as f64 }
    }
}

// ── helpers ───────────────────────────────────────────────────────

/// Minimal JSON string escaping (quotes, backslashes, control chars).
fn escape_json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

// ===================================================================
// Tests
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────

    fn make_event(token_id: u32, text: &str) -> StreamEvent {
        StreamEvent {
            token_id,
            text: text.to_string(),
            logprob: None,
            finish_reason: None,
            timestamp_ms: 1000 + u64::from(token_id),
        }
    }

    fn make_event_with_logprob(token_id: u32, text: &str, lp: f32) -> StreamEvent {
        StreamEvent {
            token_id,
            text: text.to_string(),
            logprob: Some(lp),
            finish_reason: None,
            timestamp_ms: 1000 + u64::from(token_id),
        }
    }

    fn finish_event(reason: FinishReason) -> StreamEvent {
        StreamEvent {
            token_id: 0,
            text: String::new(),
            logprob: None,
            finish_reason: Some(reason),
            timestamp_ms: 9999,
        }
    }

    // ── StreamFormat Display ──────────────────────────────────────

    #[test]
    fn test_stream_format_display() {
        assert_eq!(StreamFormat::RawTokens.to_string(), "RawTokens");
        assert_eq!(StreamFormat::SSE.to_string(), "SSE");
        assert_eq!(StreamFormat::WebSocket.to_string(), "WebSocket");
        assert_eq!(StreamFormat::NDJSON.to_string(), "NDJSON");
        assert_eq!(StreamFormat::OpenAICompat.to_string(), "OpenAICompat");
    }

    // ── StreamConfig defaults ─────────────────────────────────────

    #[test]
    fn test_stream_config_default() {
        let cfg = StreamConfig::default();
        assert_eq!(cfg.format, StreamFormat::SSE);
        assert_eq!(cfg.buffer_size, 32);
        assert_eq!(cfg.flush_interval_ms, 50);
        assert!(!cfg.include_logprobs);
    }

    // ── FinishReason Display ──────────────────────────────────────

    #[test]
    fn test_finish_reason_display() {
        assert_eq!(FinishReason::Length.to_string(), "length");
        assert_eq!(FinishReason::Stop.to_string(), "stop");
        assert_eq!(FinishReason::Cancelled.to_string(), "cancelled");
    }

    // ── RawTokens format ──────────────────────────────────────────

    #[test]
    fn test_format_raw_tokens_single() {
        let fmt = StreamFormatter::new(StreamFormat::RawTokens, false);
        let ev = make_event(42, "hello");
        assert_eq!(fmt.format_event(&ev), "hello");
    }

    #[test]
    fn test_format_raw_tokens_batch() {
        let fmt = StreamFormatter::new(StreamFormat::RawTokens, false);
        let evs = vec![make_event(1, "a"), make_event(2, "b"), make_event(3, "c")];
        assert_eq!(fmt.format_batch(&evs), "abc");
    }

    #[test]
    fn test_format_raw_empty_text() {
        let fmt = StreamFormatter::new(StreamFormat::RawTokens, false);
        let ev = make_event(0, "");
        assert_eq!(fmt.format_event(&ev), "");
    }

    // ── SSE format ────────────────────────────────────────────────

    #[test]
    fn test_format_sse_basic() {
        let fmt = StreamFormatter::new(StreamFormat::SSE, false);
        let ev = make_event(1, "Hi");
        let out = fmt.format_event(&ev);
        assert!(out.starts_with("data: "));
        assert!(out.ends_with("\n\n"));
        assert!(out.contains("\"token_id\":1"));
        assert!(out.contains("\"text\":\"Hi\""));
    }

    #[test]
    fn test_format_sse_finish_event() {
        let fmt = StreamFormatter::new(StreamFormat::SSE, false);
        let ev = finish_event(FinishReason::Stop);
        let out = fmt.format_event(&ev);
        assert!(out.contains("event: done"));
        assert!(out.contains("\"finish_reason\":\"stop\""));
    }

    #[test]
    fn test_format_sse_with_logprob() {
        let fmt = StreamFormatter::new(StreamFormat::SSE, true);
        let ev = make_event_with_logprob(5, "x", -0.5);
        let out = fmt.format_event(&ev);
        assert!(out.contains("\"logprob\":-0.5"));
    }

    #[test]
    fn test_format_sse_without_logprob_flag() {
        let fmt = StreamFormatter::new(StreamFormat::SSE, false);
        let ev = make_event_with_logprob(5, "x", -0.5);
        let out = fmt.format_event(&ev);
        assert!(!out.contains("logprob"));
    }

    // ── WebSocket format ──────────────────────────────────────────

    #[test]
    fn test_format_websocket_basic() {
        let fmt = StreamFormatter::new(StreamFormat::WebSocket, false);
        let ev = make_event(10, "ws");
        let out = fmt.format_event(&ev);
        assert!(out.ends_with('\n'));
        assert!(out.contains("\"token_id\":10"));
        assert!(out.contains("\"text\":\"ws\""));
    }

    #[test]
    fn test_format_websocket_finish() {
        let fmt = StreamFormatter::new(StreamFormat::WebSocket, false);
        let ev = finish_event(FinishReason::Length);
        let out = fmt.format_event(&ev);
        assert!(out.contains("\"finish_reason\":\"length\""));
    }

    // ── NDJSON format ─────────────────────────────────────────────

    #[test]
    fn test_format_ndjson_basic() {
        let fmt = StreamFormatter::new(StreamFormat::NDJSON, false);
        let ev = make_event(7, "nd");
        let out = fmt.format_event(&ev);
        assert!(out.ends_with('\n'));
        assert!(out.contains("\"token_id\":7"));
    }

    #[test]
    fn test_format_ndjson_batch_lines() {
        let fmt = StreamFormatter::new(StreamFormat::NDJSON, false);
        let evs = vec![make_event(1, "a"), make_event(2, "b")];
        let out = fmt.format_batch(&evs);
        let lines: Vec<&str> = out.trim_end().split('\n').collect();
        assert_eq!(lines.len(), 2);
    }

    // ── OpenAICompat format ───────────────────────────────────────

    #[test]
    fn test_format_openai_basic() {
        let fmt = StreamFormatter::new(StreamFormat::OpenAICompat, false);
        let ev = make_event(3, "oa");
        let out = fmt.format_event(&ev);
        assert!(out.starts_with("data: "));
        assert!(out.contains("\"choices\""));
        assert!(out.contains("\"delta\""));
        assert!(out.contains("\"content\":\"oa\""));
        assert!(out.contains("\"finish_reason\":null"));
    }

    #[test]
    fn test_format_openai_finish() {
        let fmt = StreamFormatter::new(StreamFormat::OpenAICompat, false);
        let ev = finish_event(FinishReason::Stop);
        let out = fmt.format_event(&ev);
        assert!(out.contains("\"finish_reason\":\"stop\""));
        assert!(out.contains("[DONE]"));
    }

    #[test]
    fn test_format_openai_with_logprob() {
        let fmt = StreamFormatter::new(StreamFormat::OpenAICompat, true);
        let ev = make_event_with_logprob(8, "lp", -1.2);
        let out = fmt.format_event(&ev);
        assert!(out.contains("\"logprob\":-1.2"));
    }

    #[test]
    fn test_format_openai_no_logprob_without_flag() {
        let fmt = StreamFormatter::new(StreamFormat::OpenAICompat, false);
        let ev = make_event_with_logprob(8, "lp", -1.2);
        let out = fmt.format_event(&ev);
        assert!(!out.contains("logprob"));
    }

    #[test]
    fn test_format_openai_finish_length() {
        let fmt = StreamFormatter::new(StreamFormat::OpenAICompat, false);
        let ev = finish_event(FinishReason::Length);
        let out = fmt.format_event(&ev);
        assert!(out.contains("\"finish_reason\":\"length\""));
        assert!(out.contains("[DONE]"));
    }

    // ── StreamBuffer basic ────────────────────────────────────────

    #[test]
    fn test_buffer_new_empty() {
        let buf = StreamBuffer::new(8, BackpressurePolicy::DropOldest, 50);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.capacity(), 8);
        assert_eq!(buf.dropped_count(), 0);
    }

    #[test]
    fn test_buffer_from_config() {
        let cfg = StreamConfig { buffer_size: 16, ..Default::default() };
        let buf = StreamBuffer::from_config(&cfg, BackpressurePolicy::Block);
        assert_eq!(buf.capacity(), 16);
    }

    #[test]
    fn test_buffer_push_and_len() {
        let mut buf = StreamBuffer::new(4, BackpressurePolicy::Block, 50);
        assert!(buf.push(make_event(1, "a")));
        assert!(buf.push(make_event(2, "b")));
        assert_eq!(buf.len(), 2);
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_buffer_flush_drains() {
        let mut buf = StreamBuffer::new(4, BackpressurePolicy::Block, 50);
        buf.push(make_event(1, "a"));
        buf.push(make_event(2, "b"));
        let events = buf.flush();
        assert_eq!(events.len(), 2);
        assert!(buf.is_empty());
        assert_eq!(events[0].token_id, 1);
        assert_eq!(events[1].token_id, 2);
    }

    #[test]
    fn test_buffer_should_flush_when_full() {
        let mut buf = StreamBuffer::new(2, BackpressurePolicy::Block, 10_000);
        buf.push(make_event(1, "a"));
        assert!(!buf.should_flush()); // 1/2 full, timer not elapsed
        buf.push(make_event(2, "b"));
        assert!(buf.should_flush()); // 2/2 full
    }

    #[test]
    fn test_buffer_should_flush_empty_never() {
        let buf = StreamBuffer::new(2, BackpressurePolicy::Block, 0);
        assert!(!buf.should_flush());
    }

    // ── Backpressure: DropOldest ──────────────────────────────────

    #[test]
    fn test_backpressure_drop_oldest() {
        let mut buf = StreamBuffer::new(2, BackpressurePolicy::DropOldest, 10_000);
        buf.push(make_event(1, "a"));
        buf.push(make_event(2, "b"));
        // Buffer full — pushing should drop oldest (token_id=1)
        assert!(buf.push(make_event(3, "c")));
        assert_eq!(buf.len(), 2);
        assert_eq!(buf.dropped_count(), 1);
        let events = buf.flush();
        assert_eq!(events[0].token_id, 2);
        assert_eq!(events[1].token_id, 3);
    }

    #[test]
    fn test_backpressure_drop_oldest_multiple() {
        let mut buf = StreamBuffer::new(2, BackpressurePolicy::DropOldest, 10_000);
        for i in 0..5 {
            buf.push(make_event(i, "x"));
        }
        assert_eq!(buf.dropped_count(), 3);
        let events = buf.flush();
        assert_eq!(events[0].token_id, 3);
        assert_eq!(events[1].token_id, 4);
    }

    // ── Backpressure: DropNewest ──────────────────────────────────

    #[test]
    fn test_backpressure_drop_newest() {
        let mut buf = StreamBuffer::new(2, BackpressurePolicy::DropNewest, 10_000);
        buf.push(make_event(1, "a"));
        buf.push(make_event(2, "b"));
        assert!(!buf.push(make_event(3, "c")));
        assert_eq!(buf.len(), 2);
        assert_eq!(buf.dropped_count(), 1);
        let events = buf.flush();
        assert_eq!(events[0].token_id, 1);
        assert_eq!(events[1].token_id, 2);
    }

    #[test]
    fn test_backpressure_drop_newest_preserves_first() {
        let mut buf = StreamBuffer::new(1, BackpressurePolicy::DropNewest, 10_000);
        assert!(buf.push(make_event(1, "a")));
        assert!(!buf.push(make_event(2, "b")));
        assert!(!buf.push(make_event(3, "c")));
        assert_eq!(buf.dropped_count(), 2);
        let events = buf.flush();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].token_id, 1);
    }

    // ── Backpressure: Block ───────────────────────────────────────

    #[test]
    fn test_backpressure_block_accepts_all() {
        let mut buf = StreamBuffer::new(2, BackpressurePolicy::Block, 10_000);
        for i in 0..10 {
            assert!(buf.push(make_event(i, "x")));
        }
        assert_eq!(buf.len(), 10);
        assert_eq!(buf.dropped_count(), 0);
    }

    #[test]
    fn test_backpressure_block_no_data_loss() {
        let mut buf = StreamBuffer::new(1, BackpressurePolicy::Block, 10_000);
        for i in 0..50 {
            buf.push(make_event(i, &format!("t{i}")));
        }
        let events = buf.flush();
        assert_eq!(events.len(), 50);
        for (i, ev) in events.iter().enumerate() {
            assert_eq!(ev.token_id, i as u32);
        }
    }

    // ── Backpressure: Resize ──────────────────────────────────────

    #[test]
    fn test_backpressure_resize_doubles_capacity() {
        let mut buf = StreamBuffer::new(2, BackpressurePolicy::Resize, 10_000);
        buf.push(make_event(1, "a"));
        buf.push(make_event(2, "b"));
        assert_eq!(buf.capacity(), 2);
        buf.push(make_event(3, "c"));
        assert_eq!(buf.capacity(), 4);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.dropped_count(), 0);
    }

    #[test]
    fn test_backpressure_resize_multiple() {
        let mut buf = StreamBuffer::new(1, BackpressurePolicy::Resize, 10_000);
        for i in 0..5 {
            buf.push(make_event(i, "x"));
        }
        // 1 → 2 → 4 → 8
        assert!(buf.capacity() >= 5);
        assert_eq!(buf.len(), 5);
        assert_eq!(buf.dropped_count(), 0);
    }

    // ── StreamMetrics ─────────────────────────────────────────────

    #[test]
    fn test_metrics_default() {
        let m = StreamMetrics::new();
        assert_eq!(m.tokens_sent, 0);
        assert_eq!(m.bytes_sent, 0);
        assert_eq!(m.dropped_count, 0);
        assert_eq!(m.avg_latency_ms(), 0.0);
    }

    #[test]
    fn test_metrics_record_send() {
        let mut m = StreamMetrics::new();
        m.record_send(100, 5.0);
        m.record_send(200, 15.0);
        assert_eq!(m.tokens_sent, 2);
        assert_eq!(m.bytes_sent, 300);
        assert!((m.avg_latency_ms() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metrics_record_drop() {
        let mut m = StreamMetrics::new();
        m.record_drop();
        m.record_drop();
        assert_eq!(m.dropped_count, 2);
        assert_eq!(m.tokens_sent, 0);
    }

    #[test]
    fn test_metrics_avg_latency_single() {
        let mut m = StreamMetrics::new();
        m.record_send(50, 7.5);
        assert!((m.avg_latency_ms() - 7.5).abs() < f64::EPSILON);
    }

    // ── JSON escaping ─────────────────────────────────────────────

    #[test]
    fn test_escape_json_str_basic() {
        assert_eq!(escape_json_str("hello"), "hello");
    }

    #[test]
    fn test_escape_json_str_quotes() {
        assert_eq!(escape_json_str(r#"he said "hi""#), r#"he said \"hi\""#);
    }

    #[test]
    fn test_escape_json_str_backslash() {
        assert_eq!(escape_json_str(r"a\b"), r"a\\b");
    }

    #[test]
    fn test_escape_json_str_newlines() {
        assert_eq!(escape_json_str("a\nb\r\tc"), r"a\nb\r\tc");
    }

    #[test]
    fn test_escape_json_str_control_char() {
        let input = String::from("a\x01b");
        let escaped = escape_json_str(&input);
        assert!(escaped.contains("\\u0001"));
    }

    // ── Edge cases ────────────────────────────────────────────────

    #[test]
    fn test_empty_stream_flush() {
        let mut buf = StreamBuffer::new(4, BackpressurePolicy::Block, 50);
        let events = buf.flush();
        assert!(events.is_empty());
    }

    #[test]
    fn test_single_token_stream() {
        let mut buf = StreamBuffer::new(4, BackpressurePolicy::Block, 50);
        buf.push(make_event(42, "only"));
        let events = buf.flush();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].text, "only");
    }

    #[test]
    fn test_formatter_from_config() {
        let cfg = StreamConfig {
            format: StreamFormat::NDJSON,
            include_logprobs: true,
            ..Default::default()
        };
        let fmt = StreamFormatter::from_config(&cfg);
        let ev = make_event_with_logprob(1, "tok", -0.3);
        let out = fmt.format_event(&ev);
        assert!(out.contains("\"logprob\":-0.3"));
    }

    #[test]
    fn test_finish_reason_propagation_all_formats() {
        for format in [
            StreamFormat::RawTokens,
            StreamFormat::SSE,
            StreamFormat::WebSocket,
            StreamFormat::NDJSON,
            StreamFormat::OpenAICompat,
        ] {
            let fmt = StreamFormatter::new(format, false);
            let ev = finish_event(FinishReason::Cancelled);
            let out = fmt.format_event(&ev);
            // RawTokens gives empty text; others should contain "cancelled"
            if format != StreamFormat::RawTokens {
                assert!(
                    out.contains("cancelled"),
                    "format {format} did not propagate finish_reason"
                );
            }
        }
    }

    #[test]
    fn test_buffer_flush_resets_timer() {
        let mut buf = StreamBuffer::new(4, BackpressurePolicy::Block, 10_000);
        buf.push(make_event(1, "a"));
        buf.flush();
        // Immediately after flush, should_flush is false because timer reset
        // and buffer is empty.
        assert!(!buf.should_flush());
    }

    #[test]
    fn test_openai_compat_cancelled() {
        let fmt = StreamFormatter::new(StreamFormat::OpenAICompat, false);
        let ev = finish_event(FinishReason::Cancelled);
        let out = fmt.format_event(&ev);
        assert!(out.contains("\"finish_reason\":\"cancelled\""));
        assert!(out.contains("[DONE]"));
    }

    // ── Property-style: no tokens lost under Block policy ─────────

    #[test]
    fn test_property_block_no_loss_varying_sizes() {
        for cap in [1, 2, 3, 5, 8, 16] {
            let n = cap * 4;
            let mut buf = StreamBuffer::new(cap, BackpressurePolicy::Block, 10_000);
            for i in 0..n {
                assert!(buf.push(make_event(i as u32, &format!("t{i}"))));
            }
            let events = buf.flush();
            assert_eq!(events.len(), n, "lost tokens with cap={cap}");
            for (i, ev) in events.iter().enumerate() {
                assert_eq!(ev.token_id, i as u32);
            }
        }
    }

    #[test]
    fn test_property_drop_oldest_keeps_newest() {
        for cap in [1, 2, 4, 8] {
            let n = cap * 3;
            let mut buf = StreamBuffer::new(cap, BackpressurePolicy::DropOldest, 10_000);
            for i in 0..n {
                buf.push(make_event(i as u32, "x"));
            }
            let events = buf.flush();
            assert_eq!(events.len(), cap);
            // The last `cap` tokens should be present.
            for (j, ev) in events.iter().enumerate() {
                assert_eq!(ev.token_id, (n - cap + j) as u32);
            }
        }
    }

    #[test]
    fn test_property_drop_newest_keeps_oldest() {
        for cap in [1, 2, 4, 8] {
            let n = cap * 3;
            let mut buf = StreamBuffer::new(cap, BackpressurePolicy::DropNewest, 10_000);
            for i in 0..n {
                buf.push(make_event(i as u32, "x"));
            }
            let events = buf.flush();
            assert_eq!(events.len(), cap);
            for (j, ev) in events.iter().enumerate() {
                assert_eq!(ev.token_id, j as u32);
            }
        }
    }

    #[test]
    fn test_property_resize_no_loss() {
        for initial_cap in [1, 2, 3] {
            let n = 20;
            let mut buf = StreamBuffer::new(initial_cap, BackpressurePolicy::Resize, 10_000);
            for i in 0..n {
                buf.push(make_event(i, "x"));
            }
            let events = buf.flush();
            assert_eq!(events.len(), n as usize);
            for (i, ev) in events.iter().enumerate() {
                assert_eq!(ev.token_id, i as u32);
            }
        }
    }

    // ── Metrics integration ───────────────────────────────────────

    #[test]
    fn test_metrics_integration_with_buffer() {
        let mut buf = StreamBuffer::new(4, BackpressurePolicy::DropNewest, 10_000);
        let mut metrics = StreamMetrics::new();
        let fmt = StreamFormatter::new(StreamFormat::SSE, false);

        for i in 0..6 {
            let ev = make_event(i, &format!("t{i}"));
            if buf.push(ev.clone()) {
                let rendered = fmt.format_event(&ev);
                metrics.record_send(rendered.len() as u64, 1.0);
            } else {
                metrics.record_drop();
            }
        }

        assert_eq!(metrics.tokens_sent, 4);
        assert_eq!(metrics.dropped_count, 2);
        assert!(metrics.bytes_sent > 0);
    }
}
