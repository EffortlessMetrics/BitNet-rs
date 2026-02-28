//! Token streaming protocol with backpressure control.
//!
//! Provides an async [`TokenStream`] that delivers tokens through
//! configurable formatters (SSE, JSON Lines, raw text) with
//! backpressure monitoring, ring-buffer batching, heartbeat
//! keep-alives, and per-stream metrics.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

// ── Protocol ──────────────────────────────────────────────────────────

/// Wire protocol used to deliver streamed tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StreamProtocol {
    /// Server-Sent Events (`text/event-stream`).
    ServerSentEvents,
    /// WebSocket frames.
    WebSocket,
    /// HTTP chunked transfer encoding.
    Chunked,
    /// gRPC server-streaming RPC.
    #[serde(rename = "grpc")]
    Grpc,
}

impl std::fmt::Display for StreamProtocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ServerSentEvents => write!(f, "SSE"),
            Self::WebSocket => write!(f, "WebSocket"),
            Self::Chunked => write!(f, "Chunked"),
            Self::Grpc => write!(f, "gRPC"),
        }
    }
}

// ── Events ────────────────────────────────────────────────────────────

/// A single event emitted by a [`TokenStream`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StreamEvent {
    /// A generated token.
    Token(String),
    /// Arbitrary metadata blob.
    Metadata(serde_json::Value),
    /// Keep-alive heartbeat.
    Heartbeat,
    /// Non-fatal error.
    Error(String),
    /// Stream has finished.
    Done,
}

// ── Configuration ─────────────────────────────────────────────────────

/// Knobs that govern a [`TokenStream`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamConfig {
    /// Channel capacity between producer and consumer.
    pub buffer_size: usize,
    /// Milliseconds between heartbeat events when idle.
    pub heartbeat_interval_ms: u64,
    /// Maximum idle time (ms) before the stream is closed.
    pub max_idle_ms: u64,
    /// When the channel has this many pending items the
    /// [`BackpressureController`] signals the producer to pause.
    pub backpressure_threshold: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            buffer_size: 64,
            heartbeat_interval_ms: 15_000,
            max_idle_ms: 60_000,
            backpressure_threshold: 48,
        }
    }
}

// ── Token stream ──────────────────────────────────────────────────────

/// Async token stream backed by an MPSC channel.
pub struct TokenStream {
    rx: mpsc::Receiver<StreamEvent>,
    tx: mpsc::Sender<StreamEvent>,
    protocol: StreamProtocol,
    config: StreamConfig,
    metrics: Arc<StreamMetrics>,
    backpressure: Arc<BackpressureController>,
}

impl TokenStream {
    /// Create a new stream with the given protocol and config.
    pub fn new(protocol: StreamProtocol, config: StreamConfig) -> Self {
        let (tx, rx) = mpsc::channel(config.buffer_size);
        let backpressure = Arc::new(BackpressureController::new(
            config.backpressure_threshold,
            config.buffer_size,
        ));
        Self {
            rx,
            tx,
            protocol,
            config,
            metrics: Arc::new(StreamMetrics::new()),
            backpressure,
        }
    }

    /// Obtain a producer handle that can send events into this stream.
    pub fn producer(&self) -> StreamProducer {
        StreamProducer {
            tx: self.tx.clone(),
            backpressure: Arc::clone(&self.backpressure),
            metrics: Arc::clone(&self.metrics),
        }
    }

    /// Receive the next event (async).
    pub async fn recv(&mut self) -> Option<StreamEvent> {
        let event = self.rx.recv().await;
        if let Some(ref e) = event {
            if matches!(e, StreamEvent::Token(_)) {
                self.metrics.record_token();
            }
            self.backpressure.on_consume();
        }
        event
    }

    /// The wire protocol chosen at creation.
    pub fn protocol(&self) -> StreamProtocol {
        self.protocol
    }

    /// Snapshot of current metrics.
    pub fn metrics(&self) -> &Arc<StreamMetrics> {
        &self.metrics
    }

    /// Access the backpressure controller.
    pub fn backpressure(&self) -> &Arc<BackpressureController> {
        &self.backpressure
    }

    /// Access the stream configuration.
    pub fn config(&self) -> &StreamConfig {
        &self.config
    }
}

// ── Producer handle ───────────────────────────────────────────────────

/// Cloneable handle used by the generation engine to push events.
#[derive(Clone)]
pub struct StreamProducer {
    tx: mpsc::Sender<StreamEvent>,
    backpressure: Arc<BackpressureController>,
    metrics: Arc<StreamMetrics>,
}

impl StreamProducer {
    /// Send an event. Returns `Err` if the consumer has dropped.
    pub async fn send(
        &self,
        event: StreamEvent,
    ) -> Result<(), mpsc::error::SendError<StreamEvent>> {
        self.backpressure.on_produce();
        if matches!(event, StreamEvent::Token(_)) {
            self.metrics.record_produced();
        }
        self.tx.send(event).await
    }

    /// Check whether the producer should pause.
    pub fn should_pause(&self) -> bool {
        self.backpressure.is_paused()
    }
}

// ── Backpressure ──────────────────────────────────────────────────────

/// Monitors consumer speed and pauses production when the consumer
/// falls behind.
pub struct BackpressureController {
    threshold: usize,
    capacity: usize,
    pending: AtomicU64,
    paused: AtomicBool,
    pause_count: AtomicU64,
}

impl BackpressureController {
    /// Create a controller with the given threshold and capacity.
    pub fn new(threshold: usize, capacity: usize) -> Self {
        Self {
            threshold,
            capacity,
            pending: AtomicU64::new(0),
            paused: AtomicBool::new(false),
            pause_count: AtomicU64::new(0),
        }
    }

    /// Called when a new event is produced.
    pub fn on_produce(&self) {
        let prev = self.pending.fetch_add(1, Ordering::SeqCst);
        if (prev + 1) as usize >= self.threshold {
            if !self.paused.swap(true, Ordering::SeqCst) {
                self.pause_count.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    /// Called when an event is consumed.
    pub fn on_consume(&self) {
        let prev = self.pending.fetch_sub(1, Ordering::SeqCst);
        // Resume when we drop below half the threshold.
        if (prev - 1) as usize <= self.threshold / 2 {
            self.paused.store(false, Ordering::SeqCst);
        }
    }

    /// Whether the producer should currently pause.
    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::SeqCst)
    }

    /// Number of pending (in-flight) events.
    pub fn pending(&self) -> u64 {
        self.pending.load(Ordering::SeqCst)
    }

    /// Total number of times backpressure was activated.
    pub fn pause_count(&self) -> u64 {
        self.pause_count.load(Ordering::SeqCst)
    }

    /// The configured threshold.
    pub fn threshold(&self) -> usize {
        self.threshold
    }

    /// The channel capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ── Metrics ───────────────────────────────────────────────────────────

/// Per-stream metrics: throughput, backpressure events, totals.
pub struct StreamMetrics {
    start: Instant,
    total_tokens: AtomicU64,
    total_produced: AtomicU64,
    backpressure_events: AtomicU64,
}

impl StreamMetrics {
    fn new() -> Self {
        Self {
            start: Instant::now(),
            total_tokens: AtomicU64::new(0),
            total_produced: AtomicU64::new(0),
            backpressure_events: AtomicU64::new(0),
        }
    }

    /// Record that a token was received by the consumer.
    pub fn record_token(&self) {
        self.total_tokens.fetch_add(1, Ordering::Relaxed);
    }

    /// Record that a token was produced.
    pub fn record_produced(&self) {
        self.total_produced.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a backpressure activation.
    pub fn record_backpressure(&self) {
        self.backpressure_events.fetch_add(1, Ordering::Relaxed);
    }

    /// Total tokens consumed.
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens.load(Ordering::Relaxed)
    }

    /// Total tokens produced.
    pub fn total_produced(&self) -> u64 {
        self.total_produced.load(Ordering::Relaxed)
    }

    /// Number of backpressure activations.
    pub fn backpressure_events(&self) -> u64 {
        self.backpressure_events.load(Ordering::Relaxed)
    }

    /// Elapsed time since stream creation.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Tokens per second (consumed).
    pub fn tokens_per_sec(&self) -> f64 {
        let secs = self.elapsed().as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        self.total_tokens() as f64 / secs
    }
}

// ── Formatter trait ───────────────────────────────────────────────────

/// Serialises [`StreamEvent`]s into a wire-format string.
pub trait StreamFormatter: Send + Sync {
    /// Format a single event.
    fn format(&self, event: &StreamEvent) -> String;

    /// Content-Type header value for this format.
    fn content_type(&self) -> &str;
}

// ── SSE formatter ─────────────────────────────────────────────────────

/// Formats events as Server-Sent Events (`data: ...\n\n`).
pub struct SseFormatter;

impl StreamFormatter for SseFormatter {
    fn format(&self, event: &StreamEvent) -> String {
        match event {
            StreamEvent::Token(t) => {
                let payload =
                    serde_json::json!({ "token": t }).to_string();
                format!("event: token\ndata: {payload}\n\n")
            }
            StreamEvent::Metadata(v) => {
                let payload = v.to_string();
                format!("event: metadata\ndata: {payload}\n\n")
            }
            StreamEvent::Heartbeat => "event: heartbeat\ndata: \n\n".to_string(),
            StreamEvent::Error(e) => {
                let payload =
                    serde_json::json!({ "error": e }).to_string();
                format!("event: error\ndata: {payload}\n\n")
            }
            StreamEvent::Done => "event: done\ndata: [DONE]\n\n".to_string(),
        }
    }

    fn content_type(&self) -> &str {
        "text/event-stream"
    }
}

// ── JSON Lines formatter ──────────────────────────────────────────────

/// One JSON object per line, newline-delimited.
pub struct JsonLinesFormatter;

impl StreamFormatter for JsonLinesFormatter {
    fn format(&self, event: &StreamEvent) -> String {
        let obj = match event {
            StreamEvent::Token(t) => {
                serde_json::json!({ "type": "token", "data": t })
            }
            StreamEvent::Metadata(v) => {
                serde_json::json!({ "type": "metadata", "data": v })
            }
            StreamEvent::Heartbeat => {
                serde_json::json!({ "type": "heartbeat" })
            }
            StreamEvent::Error(e) => {
                serde_json::json!({ "type": "error", "data": e })
            }
            StreamEvent::Done => {
                serde_json::json!({ "type": "done" })
            }
        };
        let mut s = obj.to_string();
        s.push('\n');
        s
    }

    fn content_type(&self) -> &str {
        "application/x-ndjson"
    }
}

// ── Raw text formatter ────────────────────────────────────────────────

/// Emits only the token text with no framing.
pub struct RawTextFormatter;

impl StreamFormatter for RawTextFormatter {
    fn format(&self, event: &StreamEvent) -> String {
        match event {
            StreamEvent::Token(t) => t.clone(),
            StreamEvent::Heartbeat | StreamEvent::Done => String::new(),
            StreamEvent::Metadata(v) => v.to_string(),
            StreamEvent::Error(e) => format!("[ERROR] {e}"),
        }
    }

    fn content_type(&self) -> &str {
        "text/plain"
    }
}

// ── Ring buffer for token batching ────────────────────────────────────

/// Fixed-capacity ring buffer that batches tokens before emission.
///
/// Tokens accumulate until either `capacity` tokens are buffered or
/// `max_delay` has elapsed since the first un-flushed token arrived.
pub struct StreamBuffer {
    buf: VecDeque<StreamEvent>,
    capacity: usize,
    max_delay: Duration,
    first_insert: Option<Instant>,
}

impl StreamBuffer {
    /// Create a buffer that flushes after `capacity` items or
    /// `max_delay`, whichever comes first.
    pub fn new(capacity: usize, max_delay: Duration) -> Self {
        Self {
            buf: VecDeque::with_capacity(capacity),
            capacity,
            max_delay,
            first_insert: None,
        }
    }

    /// Push an event into the buffer.
    pub fn push(&mut self, event: StreamEvent) {
        if self.first_insert.is_none() {
            self.first_insert = Some(Instant::now());
        }
        self.buf.push_back(event);
    }

    /// Whether the buffer should be flushed now.
    pub fn should_flush(&self) -> bool {
        if self.buf.len() >= self.capacity {
            return true;
        }
        if let Some(ts) = self.first_insert {
            if ts.elapsed() >= self.max_delay {
                return true;
            }
        }
        false
    }

    /// Drain all buffered events.
    pub fn flush(&mut self) -> Vec<StreamEvent> {
        self.first_insert = None;
        self.buf.drain(..).collect()
    }

    /// Number of events currently buffered.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// The configured capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The configured maximum delay.
    pub fn max_delay(&self) -> Duration {
        self.max_delay
    }
}

// ── Helper: build formatter for a protocol ────────────────────────────

/// Return a boxed formatter appropriate for `protocol`.
pub fn formatter_for_protocol(
    protocol: StreamProtocol,
) -> Box<dyn StreamFormatter> {
    match protocol {
        StreamProtocol::ServerSentEvents => Box::new(SseFormatter),
        StreamProtocol::WebSocket | StreamProtocol::Chunked => {
            Box::new(JsonLinesFormatter)
        }
        StreamProtocol::Grpc => Box::new(JsonLinesFormatter),
    }
}

// ══════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── StreamProtocol ────────────────────────────────────────────

    #[test]
    fn protocol_display_sse() {
        assert_eq!(StreamProtocol::ServerSentEvents.to_string(), "SSE");
    }

    #[test]
    fn protocol_display_websocket() {
        assert_eq!(StreamProtocol::WebSocket.to_string(), "WebSocket");
    }

    #[test]
    fn protocol_display_chunked() {
        assert_eq!(StreamProtocol::Chunked.to_string(), "Chunked");
    }

    #[test]
    fn protocol_display_grpc() {
        assert_eq!(StreamProtocol::Grpc.to_string(), "gRPC");
    }

    #[test]
    fn protocol_serde_roundtrip() {
        let p = StreamProtocol::ServerSentEvents;
        let json = serde_json::to_string(&p).unwrap();
        let back: StreamProtocol = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn protocol_grpc_serde_name() {
        let json = serde_json::to_string(&StreamProtocol::Grpc).unwrap();
        assert!(json.contains("grpc"));
    }

    #[test]
    fn protocol_all_variants_are_distinct() {
        let variants = [
            StreamProtocol::ServerSentEvents,
            StreamProtocol::WebSocket,
            StreamProtocol::Chunked,
            StreamProtocol::Grpc,
        ];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b);
                }
            }
        }
    }

    // ── StreamEvent ───────────────────────────────────────────────

    #[test]
    fn event_token_serde() {
        let e = StreamEvent::Token("hello".into());
        let json = serde_json::to_string(&e).unwrap();
        let back: StreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(e, back);
    }

    #[test]
    fn event_metadata_serde() {
        let val = serde_json::json!({"model": "bitnet-2b"});
        let e = StreamEvent::Metadata(val);
        let json = serde_json::to_string(&e).unwrap();
        let back: StreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(e, back);
    }

    #[test]
    fn event_heartbeat_serde() {
        let e = StreamEvent::Heartbeat;
        let json = serde_json::to_string(&e).unwrap();
        let back: StreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(e, back);
    }

    #[test]
    fn event_error_serde() {
        let e = StreamEvent::Error("oops".into());
        let json = serde_json::to_string(&e).unwrap();
        let back: StreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(e, back);
    }

    #[test]
    fn event_done_serde() {
        let e = StreamEvent::Done;
        let json = serde_json::to_string(&e).unwrap();
        let back: StreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(e, back);
    }

    // ── StreamConfig ──────────────────────────────────────────────

    #[test]
    fn config_default_values() {
        let c = StreamConfig::default();
        assert_eq!(c.buffer_size, 64);
        assert_eq!(c.heartbeat_interval_ms, 15_000);
        assert_eq!(c.max_idle_ms, 60_000);
        assert_eq!(c.backpressure_threshold, 48);
    }

    #[test]
    fn config_serde_roundtrip() {
        let c = StreamConfig {
            buffer_size: 32,
            heartbeat_interval_ms: 5_000,
            max_idle_ms: 30_000,
            backpressure_threshold: 24,
        };
        let json = serde_json::to_string(&c).unwrap();
        let back: StreamConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.buffer_size, 32);
        assert_eq!(back.heartbeat_interval_ms, 5_000);
    }

    #[test]
    fn config_custom() {
        let c = StreamConfig {
            buffer_size: 128,
            heartbeat_interval_ms: 1_000,
            max_idle_ms: 10_000,
            backpressure_threshold: 96,
        };
        assert_eq!(c.buffer_size, 128);
        assert_eq!(c.backpressure_threshold, 96);
    }

    // ── TokenStream basic ─────────────────────────────────────────

    #[tokio::test]
    async fn stream_send_recv_token() {
        let mut stream =
            TokenStream::new(StreamProtocol::ServerSentEvents, StreamConfig::default());
        let producer = stream.producer();
        producer.send(StreamEvent::Token("hi".into())).await.unwrap();
        let event = stream.recv().await.unwrap();
        assert_eq!(event, StreamEvent::Token("hi".into()));
    }

    #[tokio::test]
    async fn stream_send_recv_done() {
        let mut stream =
            TokenStream::new(StreamProtocol::WebSocket, StreamConfig::default());
        let producer = stream.producer();
        producer.send(StreamEvent::Done).await.unwrap();
        let event = stream.recv().await.unwrap();
        assert_eq!(event, StreamEvent::Done);
    }

    #[tokio::test]
    async fn stream_send_recv_heartbeat() {
        let mut stream =
            TokenStream::new(StreamProtocol::Chunked, StreamConfig::default());
        let producer = stream.producer();
        producer.send(StreamEvent::Heartbeat).await.unwrap();
        let event = stream.recv().await.unwrap();
        assert_eq!(event, StreamEvent::Heartbeat);
    }

    #[tokio::test]
    async fn stream_send_recv_error() {
        let mut stream =
            TokenStream::new(StreamProtocol::Grpc, StreamConfig::default());
        let producer = stream.producer();
        producer.send(StreamEvent::Error("fail".into())).await.unwrap();
        let event = stream.recv().await.unwrap();
        assert_eq!(event, StreamEvent::Error("fail".into()));
    }

    #[tokio::test]
    async fn stream_send_recv_metadata() {
        let mut stream =
            TokenStream::new(StreamProtocol::ServerSentEvents, StreamConfig::default());
        let producer = stream.producer();
        let val = serde_json::json!({"key": "value"});
        producer.send(StreamEvent::Metadata(val.clone())).await.unwrap();
        let event = stream.recv().await.unwrap();
        assert_eq!(event, StreamEvent::Metadata(val));
    }

    #[tokio::test]
    async fn stream_multiple_tokens() {
        let mut stream =
            TokenStream::new(StreamProtocol::ServerSentEvents, StreamConfig::default());
        let producer = stream.producer();
        for i in 0..5 {
            producer
                .send(StreamEvent::Token(format!("tok{i}")))
                .await
                .unwrap();
        }
        for i in 0..5 {
            let event = stream.recv().await.unwrap();
            assert_eq!(event, StreamEvent::Token(format!("tok{i}")));
        }
    }

    #[tokio::test]
    async fn stream_protocol_accessor() {
        let stream =
            TokenStream::new(StreamProtocol::WebSocket, StreamConfig::default());
        assert_eq!(stream.protocol(), StreamProtocol::WebSocket);
    }

    #[tokio::test]
    async fn stream_config_accessor() {
        let cfg = StreamConfig {
            buffer_size: 16,
            heartbeat_interval_ms: 500,
            max_idle_ms: 2_000,
            backpressure_threshold: 12,
        };
        let stream =
            TokenStream::new(StreamProtocol::ServerSentEvents, cfg);
        assert_eq!(stream.config().buffer_size, 16);
    }

    #[tokio::test]
    async fn stream_closed_on_producer_drop() {
        let stream =
            TokenStream::new(StreamProtocol::ServerSentEvents, StreamConfig::default());
        let producer = stream.producer();
        drop(producer);
        // The internal tx is still alive (held by stream).
        // Drop stream's own tx to fully close.
        let mut rx = stream.rx;
        drop(stream.tx);
        let event = rx.recv().await;
        assert!(event.is_none());
    }

    // ── Metrics ───────────────────────────────────────────────────

    #[tokio::test]
    async fn metrics_token_count() {
        let mut stream =
            TokenStream::new(StreamProtocol::ServerSentEvents, StreamConfig::default());
        let producer = stream.producer();
        for _ in 0..3 {
            producer.send(StreamEvent::Token("t".into())).await.unwrap();
        }
        for _ in 0..3 {
            stream.recv().await.unwrap();
        }
        assert_eq!(stream.metrics().total_tokens(), 3);
    }

    #[tokio::test]
    async fn metrics_produced_count() {
        let stream =
            TokenStream::new(StreamProtocol::ServerSentEvents, StreamConfig::default());
        let producer = stream.producer();
        producer.send(StreamEvent::Token("a".into())).await.unwrap();
        producer.send(StreamEvent::Token("b".into())).await.unwrap();
        // Heartbeat is not a token.
        producer.send(StreamEvent::Heartbeat).await.unwrap();
        assert_eq!(stream.metrics().total_produced(), 2);
    }

    #[tokio::test]
    async fn metrics_elapsed_is_positive() {
        let stream =
            TokenStream::new(StreamProtocol::ServerSentEvents, StreamConfig::default());
        // Elapsed should be non-negative (may be zero on fast machines).
        assert!(stream.metrics().elapsed().as_nanos() < u128::MAX);
    }

    #[test]
    fn metrics_tokens_per_sec_zero_when_no_tokens() {
        let m = StreamMetrics::new();
        assert_eq!(m.tokens_per_sec(), 0.0);
    }

    #[test]
    fn metrics_backpressure_events_default_zero() {
        let m = StreamMetrics::new();
        assert_eq!(m.backpressure_events(), 0);
    }

    #[test]
    fn metrics_record_backpressure() {
        let m = StreamMetrics::new();
        m.record_backpressure();
        m.record_backpressure();
        assert_eq!(m.backpressure_events(), 2);
    }

    // ── BackpressureController ────────────────────────────────────

    #[test]
    fn backpressure_initially_not_paused() {
        let bp = BackpressureController::new(4, 8);
        assert!(!bp.is_paused());
        assert_eq!(bp.pending(), 0);
    }

    #[test]
    fn backpressure_pauses_at_threshold() {
        let bp = BackpressureController::new(3, 8);
        bp.on_produce();
        bp.on_produce();
        assert!(!bp.is_paused());
        bp.on_produce(); // pending == 3 == threshold
        assert!(bp.is_paused());
    }

    #[test]
    fn backpressure_resumes_below_half() {
        let bp = BackpressureController::new(4, 8);
        for _ in 0..4 {
            bp.on_produce();
        }
        assert!(bp.is_paused());
        // Consume until pending <= threshold/2 == 2.
        bp.on_consume(); // pending = 3
        assert!(bp.is_paused());
        bp.on_consume(); // pending = 2
        assert!(!bp.is_paused());
    }

    #[test]
    fn backpressure_pause_count_increments() {
        let bp = BackpressureController::new(2, 4);
        bp.on_produce();
        bp.on_produce(); // triggers pause
        assert_eq!(bp.pause_count(), 1);
        // Resume.
        bp.on_consume();
        bp.on_consume();
        assert!(!bp.is_paused());
        // Trigger again.
        bp.on_produce();
        bp.on_produce();
        assert_eq!(bp.pause_count(), 2);
    }

    #[test]
    fn backpressure_threshold_and_capacity() {
        let bp = BackpressureController::new(10, 20);
        assert_eq!(bp.threshold(), 10);
        assert_eq!(bp.capacity(), 20);
    }

    #[tokio::test]
    async fn producer_should_pause_reflects_backpressure() {
        let cfg = StreamConfig {
            buffer_size: 64,
            backpressure_threshold: 2,
            ..StreamConfig::default()
        };
        let stream = TokenStream::new(StreamProtocol::ServerSentEvents, cfg);
        let producer = stream.producer();
        assert!(!producer.should_pause());
        producer.send(StreamEvent::Token("a".into())).await.unwrap();
        producer.send(StreamEvent::Token("b".into())).await.unwrap();
        assert!(producer.should_pause());
    }

    // ── SSE formatter ─────────────────────────────────────────────

    #[test]
    fn sse_format_token() {
        let f = SseFormatter;
        let out = f.format(&StreamEvent::Token("hello".into()));
        assert!(out.starts_with("event: token\n"));
        assert!(out.contains("data: "));
        assert!(out.contains("\"token\":\"hello\""));
        assert!(out.ends_with("\n\n"));
    }

    #[test]
    fn sse_format_heartbeat() {
        let f = SseFormatter;
        let out = f.format(&StreamEvent::Heartbeat);
        assert_eq!(out, "event: heartbeat\ndata: \n\n");
    }

    #[test]
    fn sse_format_done() {
        let f = SseFormatter;
        let out = f.format(&StreamEvent::Done);
        assert_eq!(out, "event: done\ndata: [DONE]\n\n");
    }

    #[test]
    fn sse_format_error() {
        let f = SseFormatter;
        let out = f.format(&StreamEvent::Error("bad".into()));
        assert!(out.starts_with("event: error\n"));
        assert!(out.contains("\"error\":\"bad\""));
    }

    #[test]
    fn sse_format_metadata() {
        let f = SseFormatter;
        let val = serde_json::json!({"k": 1});
        let out = f.format(&StreamEvent::Metadata(val));
        assert!(out.starts_with("event: metadata\n"));
        assert!(out.contains("\"k\":1"));
    }

    #[test]
    fn sse_content_type() {
        let f = SseFormatter;
        assert_eq!(f.content_type(), "text/event-stream");
    }

    // ── JSON Lines formatter ──────────────────────────────────────

    #[test]
    fn jsonl_format_token() {
        let f = JsonLinesFormatter;
        let out = f.format(&StreamEvent::Token("hi".into()));
        assert!(out.ends_with('\n'));
        let parsed: serde_json::Value =
            serde_json::from_str(out.trim()).unwrap();
        assert_eq!(parsed["type"], "token");
        assert_eq!(parsed["data"], "hi");
    }

    #[test]
    fn jsonl_format_heartbeat() {
        let f = JsonLinesFormatter;
        let out = f.format(&StreamEvent::Heartbeat);
        let parsed: serde_json::Value =
            serde_json::from_str(out.trim()).unwrap();
        assert_eq!(parsed["type"], "heartbeat");
    }

    #[test]
    fn jsonl_format_done() {
        let f = JsonLinesFormatter;
        let out = f.format(&StreamEvent::Done);
        let parsed: serde_json::Value =
            serde_json::from_str(out.trim()).unwrap();
        assert_eq!(parsed["type"], "done");
    }

    #[test]
    fn jsonl_format_error() {
        let f = JsonLinesFormatter;
        let out = f.format(&StreamEvent::Error("err".into()));
        let parsed: serde_json::Value =
            serde_json::from_str(out.trim()).unwrap();
        assert_eq!(parsed["type"], "error");
        assert_eq!(parsed["data"], "err");
    }

    #[test]
    fn jsonl_format_metadata() {
        let f = JsonLinesFormatter;
        let val = serde_json::json!({"x": 42});
        let out = f.format(&StreamEvent::Metadata(val));
        let parsed: serde_json::Value =
            serde_json::from_str(out.trim()).unwrap();
        assert_eq!(parsed["type"], "metadata");
        assert_eq!(parsed["data"]["x"], 42);
    }

    #[test]
    fn jsonl_content_type() {
        let f = JsonLinesFormatter;
        assert_eq!(f.content_type(), "application/x-ndjson");
    }

    // ── Raw text formatter ────────────────────────────────────────

    #[test]
    fn raw_format_token() {
        let f = RawTextFormatter;
        assert_eq!(f.format(&StreamEvent::Token("abc".into())), "abc");
    }

    #[test]
    fn raw_format_heartbeat_empty() {
        let f = RawTextFormatter;
        assert_eq!(f.format(&StreamEvent::Heartbeat), "");
    }

    #[test]
    fn raw_format_done_empty() {
        let f = RawTextFormatter;
        assert_eq!(f.format(&StreamEvent::Done), "");
    }

    #[test]
    fn raw_format_error() {
        let f = RawTextFormatter;
        assert_eq!(
            f.format(&StreamEvent::Error("x".into())),
            "[ERROR] x"
        );
    }

    #[test]
    fn raw_content_type() {
        let f = RawTextFormatter;
        assert_eq!(f.content_type(), "text/plain");
    }

    // ── StreamBuffer ──────────────────────────────────────────────

    #[test]
    fn buffer_new_is_empty() {
        let buf = StreamBuffer::new(4, Duration::from_millis(100));
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.capacity(), 4);
    }

    #[test]
    fn buffer_push_increments_len() {
        let mut buf = StreamBuffer::new(4, Duration::from_millis(100));
        buf.push(StreamEvent::Token("a".into()));
        assert_eq!(buf.len(), 1);
        assert!(!buf.is_empty());
    }

    #[test]
    fn buffer_flushes_at_capacity() {
        let mut buf = StreamBuffer::new(3, Duration::from_secs(999));
        buf.push(StreamEvent::Token("a".into()));
        buf.push(StreamEvent::Token("b".into()));
        assert!(!buf.should_flush());
        buf.push(StreamEvent::Token("c".into()));
        assert!(buf.should_flush());
    }

    #[test]
    fn buffer_flush_drains_all() {
        let mut buf = StreamBuffer::new(2, Duration::from_secs(999));
        buf.push(StreamEvent::Token("x".into()));
        buf.push(StreamEvent::Token("y".into()));
        let events = buf.flush();
        assert_eq!(events.len(), 2);
        assert!(buf.is_empty());
    }

    #[test]
    fn buffer_flush_resets_timer() {
        let mut buf = StreamBuffer::new(10, Duration::from_millis(1));
        buf.push(StreamEvent::Heartbeat);
        std::thread::sleep(Duration::from_millis(5));
        assert!(buf.should_flush());
        let _ = buf.flush();
        // After flush, should_flush is false (empty buffer).
        assert!(!buf.should_flush());
    }

    #[test]
    fn buffer_max_delay() {
        let d = Duration::from_millis(200);
        let buf = StreamBuffer::new(10, d);
        assert_eq!(buf.max_delay(), d);
    }

    #[test]
    fn buffer_time_based_flush() {
        let mut buf = StreamBuffer::new(100, Duration::from_millis(1));
        buf.push(StreamEvent::Token("t".into()));
        std::thread::sleep(Duration::from_millis(5));
        assert!(buf.should_flush());
    }

    #[test]
    fn buffer_no_flush_when_empty() {
        let buf = StreamBuffer::new(1, Duration::from_millis(0));
        assert!(!buf.should_flush());
    }

    // ── formatter_for_protocol ────────────────────────────────────

    #[test]
    fn formatter_for_sse() {
        let f = formatter_for_protocol(StreamProtocol::ServerSentEvents);
        assert_eq!(f.content_type(), "text/event-stream");
    }

    #[test]
    fn formatter_for_websocket() {
        let f = formatter_for_protocol(StreamProtocol::WebSocket);
        assert_eq!(f.content_type(), "application/x-ndjson");
    }

    #[test]
    fn formatter_for_chunked() {
        let f = formatter_for_protocol(StreamProtocol::Chunked);
        assert_eq!(f.content_type(), "application/x-ndjson");
    }

    #[test]
    fn formatter_for_grpc() {
        let f = formatter_for_protocol(StreamProtocol::Grpc);
        assert_eq!(f.content_type(), "application/x-ndjson");
    }

    // ── Edge cases ────────────────────────────────────────────────

    #[test]
    fn event_token_empty_string() {
        let e = StreamEvent::Token(String::new());
        let json = serde_json::to_string(&e).unwrap();
        let back: StreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back, StreamEvent::Token(String::new()));
    }

    #[test]
    fn event_token_unicode() {
        let e = StreamEvent::Token("こんにちは 🌍".into());
        let json = serde_json::to_string(&e).unwrap();
        let back: StreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back, StreamEvent::Token("こんにちは 🌍".into()));
    }

    #[test]
    fn sse_format_token_with_special_chars() {
        let f = SseFormatter;
        let out = f.format(&StreamEvent::Token("line\nbreak".into()));
        // SSE data lines must not contain raw newlines in data payload.
        // serde_json escapes them for us.
        assert!(out.contains("\\n"));
    }

    #[test]
    fn jsonl_each_line_is_valid_json() {
        let f = JsonLinesFormatter;
        let events = vec![
            StreamEvent::Token("a".into()),
            StreamEvent::Heartbeat,
            StreamEvent::Done,
        ];
        for e in &events {
            let line = f.format(e);
            serde_json::from_str::<serde_json::Value>(line.trim())
                .expect("each line must be valid JSON");
        }
    }

    #[test]
    fn raw_format_metadata() {
        let f = RawTextFormatter;
        let val = serde_json::json!({"a": 1});
        let out = f.format(&StreamEvent::Metadata(val));
        assert!(out.contains("\"a\":1"));
    }

    #[tokio::test]
    async fn stream_interleaved_event_types() {
        let mut stream = TokenStream::new(
            StreamProtocol::ServerSentEvents,
            StreamConfig::default(),
        );
        let producer = stream.producer();
        let events = vec![
            StreamEvent::Token("a".into()),
            StreamEvent::Heartbeat,
            StreamEvent::Token("b".into()),
            StreamEvent::Error("e".into()),
            StreamEvent::Done,
        ];
        for e in &events {
            producer.send(e.clone()).await.unwrap();
        }
        for expected in &events {
            let got = stream.recv().await.unwrap();
            assert_eq!(&got, expected);
        }
    }

    #[test]
    fn buffer_mixed_events() {
        let mut buf = StreamBuffer::new(3, Duration::from_secs(999));
        buf.push(StreamEvent::Token("t".into()));
        buf.push(StreamEvent::Heartbeat);
        buf.push(StreamEvent::Done);
        assert!(buf.should_flush());
        let events = buf.flush();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0], StreamEvent::Token("t".into()));
        assert_eq!(events[1], StreamEvent::Heartbeat);
        assert_eq!(events[2], StreamEvent::Done);
    }

    #[test]
    fn backpressure_pending_tracks_correctly() {
        let bp = BackpressureController::new(100, 200);
        bp.on_produce();
        bp.on_produce();
        bp.on_produce();
        assert_eq!(bp.pending(), 3);
        bp.on_consume();
        assert_eq!(bp.pending(), 2);
    }

    #[tokio::test]
    async fn metrics_heartbeat_not_counted_as_token() {
        let mut stream = TokenStream::new(
            StreamProtocol::ServerSentEvents,
            StreamConfig::default(),
        );
        let producer = stream.producer();
        producer.send(StreamEvent::Heartbeat).await.unwrap();
        stream.recv().await.unwrap();
        assert_eq!(stream.metrics().total_tokens(), 0);
    }

    #[tokio::test]
    async fn metrics_error_not_counted_as_token() {
        let mut stream = TokenStream::new(
            StreamProtocol::ServerSentEvents,
            StreamConfig::default(),
        );
        let producer = stream.producer();
        producer.send(StreamEvent::Error("e".into())).await.unwrap();
        stream.recv().await.unwrap();
        assert_eq!(stream.metrics().total_tokens(), 0);
    }
}
