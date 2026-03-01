//! # Token Streaming
//!
//! Efficient incremental text generation output. Buffers raw token IDs until
//! they form valid UTF-8 text, then emits [`StreamEvent::Text`] chunks
//! according to the [`StreamConfig`] policy.
//!
//! This module bridges the inference engine to real-time output consumers
//! (CLI, HTTP SSE, WebSocket) without requiring async or channel machinery.

use tracing::debug;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Controls when buffered tokens are flushed as text events.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Maximum tokens to buffer before forcing a flush.
    pub buffer_size: usize,
    /// Emit a text event whenever decoded text ends with ASCII whitespace.
    pub flush_on_whitespace: bool,
    /// Emit a text event whenever decoded text contains a newline.
    pub flush_on_newline: bool,
    /// Hard ceiling on pending (un-flushed) tokens. Once reached the buffer
    /// is drained regardless of UTF-8 validity.
    pub max_pending_tokens: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            buffer_size: 8,
            flush_on_whitespace: true,
            flush_on_newline: true,
            max_pending_tokens: 64,
        }
    }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// Discrete events emitted by [`TokenStream`].
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// A single token was accepted (informational – always emitted before any
    /// text event for the same push).
    Token(u32),
    /// One or more buffered tokens decoded to valid UTF-8 text.
    Text(String),
    /// The stream has been marked complete.
    EndOfStream,
    /// A non-fatal error (e.g. invalid UTF-8 bytes were skipped).
    Error(String),
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Cumulative statistics for a [`TokenStream`] lifetime.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StreamStats {
    /// Total tokens pushed into the stream.
    pub tokens_generated: usize,
    /// Number of [`StreamEvent::Text`] chunks emitted.
    pub text_chunks_emitted: usize,
    /// Running average: `tokens_generated / text_chunks_emitted`.
    pub avg_tokens_per_chunk: f64,
    /// Total bytes across all emitted text chunks.
    pub total_bytes: usize,
}

// ---------------------------------------------------------------------------
// Token buffer (UTF-8 accumulator)
// ---------------------------------------------------------------------------

/// Accumulates raw bytes produced by a decode callback until they form valid
/// UTF-8, then yields the decoded [`String`].
#[derive(Debug, Clone)]
pub struct TokenBuffer {
    bytes: Vec<u8>,
}

impl TokenBuffer {
    pub fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    /// Append raw bytes (typically the output of `tokenizer.decode` for a
    /// single token).
    pub fn push_bytes(&mut self, data: &[u8]) {
        self.bytes.extend_from_slice(data);
    }

    /// Try to drain the longest valid UTF-8 prefix from the buffer.
    ///
    /// Returns `Some(text)` when at least one valid character can be
    /// extracted, leaving any trailing incomplete sequence in the buffer
    /// for the next call.
    pub fn try_decode(&mut self) -> Option<String> {
        if self.bytes.is_empty() {
            return None;
        }
        match std::str::from_utf8(&self.bytes) {
            Ok(s) => {
                let text = s.to_owned();
                self.bytes.clear();
                Some(text)
            }
            Err(e) => {
                let valid_up_to = e.valid_up_to();
                if valid_up_to > 0 {
                    let text =
                        String::from_utf8(self.bytes[..valid_up_to].to_vec()).expect("valid utf8");
                    self.bytes.drain(..valid_up_to);
                    Some(text)
                } else {
                    // Leading bytes are not valid. If the error length is
                    // known (i.e. an unexpected continuation byte or
                    // overlong sequence), skip those bytes so we don't get
                    // stuck. An unknown error_len means the sequence is
                    // simply incomplete — wait for more data.
                    if e.error_len().is_some() {
                        // Skip the single invalid byte.
                        self.bytes.remove(0);
                        // Recurse to try the remainder.
                        return self.try_decode();
                    }
                    None
                }
            }
        }
    }

    /// Drain *all* remaining bytes, replacing invalid sequences with U+FFFD.
    pub fn drain_lossy(&mut self) -> String {
        let text = String::from_utf8_lossy(&self.bytes).into_owned();
        self.bytes.clear();
        text
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }
}

impl Default for TokenBuffer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TokenStream
// ---------------------------------------------------------------------------

/// Manages incremental text output from token-by-token generation.
///
/// Feed tokens via [`push_token`](Self::push_token). The stream buffers them,
/// decodes to UTF-8, and emits [`StreamEvent`]s according to the
/// [`StreamConfig`] policy.
///
/// The caller supplies a *decode* closure (`Fn(u32) -> Option<Vec<u8>>`) at
/// construction time so that `TokenStream` remains independent of any
/// particular tokenizer implementation.
pub struct TokenStream<F: Fn(u32) -> Option<Vec<u8>>> {
    config: StreamConfig,
    buffer: TokenBuffer,
    pending_token_count: usize,
    complete: bool,
    stats: StreamStats,
    decode_fn: F,
}

impl<F: Fn(u32) -> Option<Vec<u8>>> TokenStream<F> {
    /// Create a new stream.
    ///
    /// `decode_fn` converts a single token ID to its raw byte representation
    /// (e.g. via `tokenizer.token_to_piece`). Return `None` for unknown /
    /// special tokens.
    pub fn new(config: StreamConfig, decode_fn: F) -> Self {
        Self {
            config,
            buffer: TokenBuffer::new(),
            pending_token_count: 0,
            complete: false,
            stats: StreamStats::default(),
            decode_fn,
        }
    }

    /// Push a token into the stream.
    ///
    /// Returns a [`StreamEvent::Text`] when the buffer can be decoded and the
    /// flush policy triggers, or `None` when tokens are still being buffered.
    pub fn push_token(&mut self, token_id: u32) -> Option<StreamEvent> {
        if self.complete {
            return Some(StreamEvent::Error("stream already complete".into()));
        }

        self.stats.tokens_generated += 1;

        if let Some(raw) = (self.decode_fn)(token_id) {
            self.buffer.push_bytes(&raw);
        }
        self.pending_token_count += 1;

        // Try to extract valid UTF-8.
        if let Some(text) = self.buffer.try_decode() {
            if self.should_flush(&text) {
                return Some(self.emit_text(text));
            }
            // Not flushing yet – push the decoded text back as bytes so we
            // can re-decode later together with subsequent tokens.
            self.buffer.push_bytes(text.as_bytes());
        }

        // Force flush when we hit the hard ceiling.
        if self.pending_token_count >= self.config.max_pending_tokens {
            return Some(self.force_flush());
        }

        None
    }

    /// Drain any remaining buffered content and mark the stream complete.
    pub fn flush(&mut self) -> Vec<StreamEvent> {
        let mut events = Vec::new();

        if !self.buffer.is_empty() {
            let text = self.buffer.drain_lossy();
            if !text.is_empty() {
                events.push(self.emit_text(text));
            }
        }
        self.pending_token_count = 0;

        if !self.complete {
            self.complete = true;
            events.push(StreamEvent::EndOfStream);
        }

        events
    }

    /// Whether the stream has been marked complete (via [`flush`](Self::flush)).
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// Current cumulative statistics.
    pub fn stats(&self) -> &StreamStats {
        &self.stats
    }

    // -- private helpers ----------------------------------------------------

    fn should_flush(&self, text: &str) -> bool {
        if self.pending_token_count >= self.config.buffer_size {
            return true;
        }
        if self.config.flush_on_newline && text.contains('\n') {
            return true;
        }
        if self.config.flush_on_whitespace && text.ends_with(|c: char| c.is_ascii_whitespace()) {
            return true;
        }
        false
    }

    fn emit_text(&mut self, text: String) -> StreamEvent {
        self.stats.total_bytes += text.len();
        self.stats.text_chunks_emitted += 1;
        self.stats.avg_tokens_per_chunk =
            self.stats.tokens_generated as f64 / self.stats.text_chunks_emitted as f64;
        self.pending_token_count = 0;
        debug!(
            bytes = text.len(),
            chunks = self.stats.text_chunks_emitted,
            "token_stream: emitting text chunk"
        );
        StreamEvent::Text(text)
    }

    fn force_flush(&mut self) -> StreamEvent {
        let text = self.buffer.drain_lossy();
        if text.is_empty() {
            self.pending_token_count = 0;
            return StreamEvent::Error("max_pending_tokens reached with no decodable text".into());
        }
        self.emit_text(text)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    type DecodeFn = fn(u32) -> Option<Vec<u8>>;

    /// Simple ASCII decode: every token maps to a known ASCII byte sequence.
    fn ascii_decode(id: u32) -> Option<Vec<u8>> {
        match id {
            0 => Some(b"hello".to_vec()),
            1 => Some(b" ".to_vec()),
            2 => Some(b"world".to_vec()),
            3 => Some(b"\n".to_vec()),
            4 => Some(b"foo".to_vec()),
            5 => Some(b"bar".to_vec()),
            // Token 100 = unknown/special
            100 => None,
            other => Some(format!("<{other}>").into_bytes()),
        }
    }

    fn default_stream() -> TokenStream<DecodeFn> {
        TokenStream::new(StreamConfig::default(), ascii_decode as DecodeFn)
    }

    // -- Single token streaming --

    #[test]
    fn single_token_with_whitespace_flush() {
        let config = StreamConfig { flush_on_whitespace: true, ..Default::default() };
        let mut stream = TokenStream::new(config, ascii_decode as DecodeFn);

        // "hello" – no trailing whitespace, no flush yet.
        assert_eq!(stream.push_token(0), None);

        // " " – trailing whitespace triggers flush of accumulated "hello ".
        let event = stream.push_token(1);
        assert_eq!(event, Some(StreamEvent::Text("hello ".into())));
    }

    #[test]
    fn single_token_emits_on_buffer_size() {
        let config = StreamConfig {
            buffer_size: 1,
            flush_on_whitespace: false,
            flush_on_newline: false,
            ..Default::default()
        };
        let mut stream = TokenStream::new(config, ascii_decode as DecodeFn);

        // Buffer size = 1, so every token triggers a flush.
        let event = stream.push_token(0);
        assert_eq!(event, Some(StreamEvent::Text("hello".into())));
    }

    // -- Multi-byte UTF-8 handling --

    #[test]
    fn multibyte_utf8_buffered_until_valid() {
        // é = 0xC3 0xA9 (2-byte UTF-8). Feed one byte at a time.
        let bytes: Vec<Vec<u8>> = vec![vec![0xC3], vec![0xA9]];
        let bytes_clone = bytes.clone();
        let decode = move |id: u32| -> Option<Vec<u8>> { bytes_clone.get(id as usize).cloned() };

        let config = StreamConfig {
            buffer_size: 1,
            flush_on_whitespace: false,
            flush_on_newline: false,
            ..Default::default()
        };
        let mut stream = TokenStream::new(config, decode);

        // First byte alone is an incomplete sequence — no event.
        assert_eq!(stream.push_token(0), None);

        // Second byte completes the character.
        let event = stream.push_token(1);
        assert_eq!(event, Some(StreamEvent::Text("é".into())));
    }

    #[test]
    fn three_byte_utf8_buffered() {
        // ✓ = U+2713 = 0xE2 0x9C 0x93
        let chunks: Vec<Vec<u8>> = vec![vec![0xE2], vec![0x9C], vec![0x93]];
        let chunks_clone = chunks.clone();
        let decode = move |id: u32| -> Option<Vec<u8>> { chunks_clone.get(id as usize).cloned() };

        let config = StreamConfig {
            buffer_size: 1,
            flush_on_whitespace: false,
            flush_on_newline: false,
            ..Default::default()
        };
        let mut stream = TokenStream::new(config, decode);

        assert_eq!(stream.push_token(0), None);
        assert_eq!(stream.push_token(1), None);
        let event = stream.push_token(2);
        assert_eq!(event, Some(StreamEvent::Text("✓".into())));
    }

    #[test]
    fn four_byte_utf8_emoji() {
        // 🦀 = U+1F980 = 0xF0 0x9F 0xA6 0x80
        let chunks: Vec<Vec<u8>> = vec![vec![0xF0], vec![0x9F], vec![0xA6], vec![0x80]];
        let chunks_clone = chunks.clone();
        let decode = move |id: u32| -> Option<Vec<u8>> { chunks_clone.get(id as usize).cloned() };

        let config = StreamConfig {
            buffer_size: 1,
            flush_on_whitespace: false,
            flush_on_newline: false,
            ..Default::default()
        };
        let mut stream = TokenStream::new(config, decode);

        assert_eq!(stream.push_token(0), None);
        assert_eq!(stream.push_token(1), None);
        assert_eq!(stream.push_token(2), None);
        let event = stream.push_token(3);
        assert_eq!(event, Some(StreamEvent::Text("🦀".into())));
    }

    // -- Flush on whitespace / newline config --

    #[test]
    fn flush_on_newline() {
        let config = StreamConfig {
            flush_on_whitespace: false,
            flush_on_newline: true,
            ..Default::default()
        };
        let mut stream = TokenStream::new(config, ascii_decode as DecodeFn);

        assert_eq!(stream.push_token(0), None); // "hello"
        let event = stream.push_token(3); // "\n"
        assert_eq!(event, Some(StreamEvent::Text("hello\n".into())));
    }

    #[test]
    fn no_auto_flush_when_disabled() {
        let config = StreamConfig {
            buffer_size: 100,
            flush_on_whitespace: false,
            flush_on_newline: false,
            max_pending_tokens: 100,
        };
        let mut stream = TokenStream::new(config, ascii_decode as DecodeFn);

        // Push several tokens without hitting buffer_size — nothing should flush.
        for id in 0..5 {
            assert_eq!(stream.push_token(id), None);
        }
        // Explicit flush drains everything.
        let events = stream.flush();
        assert!(events.iter().any(|e| matches!(e, StreamEvent::Text(_))));
        assert!(events.iter().any(|e| matches!(e, StreamEvent::EndOfStream)));
    }

    // -- Buffer size limits --

    #[test]
    fn buffer_size_triggers_flush() {
        let config = StreamConfig {
            buffer_size: 3,
            flush_on_whitespace: false,
            flush_on_newline: false,
            max_pending_tokens: 64,
        };
        let mut stream = TokenStream::new(config, ascii_decode as DecodeFn);

        assert_eq!(stream.push_token(0), None); // 1 pending
        assert_eq!(stream.push_token(4), None); // 2 pending
        // 3rd token hits buffer_size → flush.
        let event = stream.push_token(5);
        assert_eq!(event, Some(StreamEvent::Text("hellofoobar".into())));
    }

    #[test]
    fn max_pending_tokens_forces_flush() {
        let config = StreamConfig {
            buffer_size: 100,
            flush_on_whitespace: false,
            flush_on_newline: false,
            max_pending_tokens: 2,
        };
        let mut stream = TokenStream::new(config, ascii_decode as DecodeFn);

        assert_eq!(stream.push_token(0), None); // 1 pending
        // 2nd token hits max_pending → forced flush.
        let event = stream.push_token(4);
        assert_eq!(event, Some(StreamEvent::Text("hellofoo".into())));
    }

    // -- End of stream handling --

    #[test]
    fn flush_marks_complete() {
        let mut stream = default_stream();
        stream.push_token(0);
        let events = stream.flush();
        assert!(stream.is_complete());
        assert!(events.last() == Some(&StreamEvent::EndOfStream));
    }

    #[test]
    fn push_after_complete_returns_error() {
        let mut stream = default_stream();
        stream.flush();
        assert!(stream.is_complete());
        let event = stream.push_token(0);
        assert_eq!(event, Some(StreamEvent::Error("stream already complete".into())));
    }

    #[test]
    fn double_flush_only_one_eos() {
        let mut stream = default_stream();
        let first = stream.flush();
        let second = stream.flush();
        let eos_count = first
            .iter()
            .chain(second.iter())
            .filter(|e| matches!(e, StreamEvent::EndOfStream))
            .count();
        assert_eq!(eos_count, 1);
    }

    // -- Statistics tracking --

    #[test]
    fn stats_updated_on_flush() {
        let config = StreamConfig {
            buffer_size: 2,
            flush_on_whitespace: false,
            flush_on_newline: false,
            max_pending_tokens: 64,
        };
        let mut stream = TokenStream::new(config, ascii_decode as DecodeFn);

        stream.push_token(0); // "hello" – 1 pending
        stream.push_token(2); // "world" – 2 pending → flush "helloworld" (10 bytes)

        let stats = stream.stats();
        assert_eq!(stats.tokens_generated, 2);
        assert_eq!(stats.text_chunks_emitted, 1);
        assert_eq!(stats.total_bytes, 10);
        assert!((stats.avg_tokens_per_chunk - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_accumulate_across_chunks() {
        let config = StreamConfig {
            buffer_size: 1,
            flush_on_whitespace: false,
            flush_on_newline: false,
            max_pending_tokens: 64,
        };
        let mut stream = TokenStream::new(config, ascii_decode as DecodeFn);

        stream.push_token(0); // "hello" → flush
        stream.push_token(1); // " " → flush
        stream.push_token(2); // "world" → flush

        let stats = stream.stats();
        assert_eq!(stats.tokens_generated, 3);
        assert_eq!(stats.text_chunks_emitted, 3);
        // 5 + 1 + 5 = 11
        assert_eq!(stats.total_bytes, 11);
    }

    // -- Empty stream --

    #[test]
    fn empty_stream_flush() {
        let mut stream = default_stream();
        let events = stream.flush();
        // Only EndOfStream, no Text event for empty stream.
        assert_eq!(events, vec![StreamEvent::EndOfStream]);
        assert!(stream.is_complete());
    }

    #[test]
    fn empty_stream_stats() {
        let stream = default_stream();
        let stats = stream.stats();
        assert_eq!(stats.tokens_generated, 0);
        assert_eq!(stats.text_chunks_emitted, 0);
        assert_eq!(stats.total_bytes, 0);
    }

    // -- Rapid token succession --

    #[test]
    fn rapid_tokens_respect_buffer_size() {
        let config = StreamConfig {
            buffer_size: 4,
            flush_on_whitespace: false,
            flush_on_newline: false,
            max_pending_tokens: 64,
        };
        let mut stream = TokenStream::new(config, ascii_decode as DecodeFn);

        let mut flush_count = 0;
        for id in 0..20 {
            if stream.push_token(id).is_some() {
                flush_count += 1;
            }
        }
        // 20 tokens / buffer_size 4 = 5 flushes.
        assert_eq!(flush_count, 5);
    }

    // -- Invalid UTF-8 recovery --

    #[test]
    fn invalid_utf8_skipped_on_push() {
        // 0xFF is never valid in UTF-8. It gets skipped by try_decode during
        // push_token, so only the valid "ok" text survives to flush.
        let decode = |id: u32| -> Option<Vec<u8>> {
            match id {
                0 => Some(vec![0xFF]),
                1 => Some(b"ok".to_vec()),
                _ => None,
            }
        };
        let config = StreamConfig {
            buffer_size: 100,
            flush_on_whitespace: false,
            flush_on_newline: false,
            max_pending_tokens: 100,
        };
        let mut stream = TokenStream::new(config, decode);
        stream.push_token(0); // invalid byte — silently skipped
        stream.push_token(1); // valid "ok"
        let events = stream.flush();

        let text_events: Vec<_> =
            events.iter().filter(|e| matches!(e, StreamEvent::Text(_))).collect();
        assert_eq!(text_events.len(), 1);
        if let StreamEvent::Text(t) = &text_events[0] {
            assert_eq!(t, "ok");
        }
    }

    #[test]
    fn invalid_utf8_replaced_on_forced_drain() {
        // When invalid bytes are still in the buffer at forced drain (e.g.
        // incomplete sequence that never gets more bytes), drain_lossy
        // replaces them with U+FFFD.
        let mut buf = TokenBuffer::new();
        buf.push_bytes(&[0xFF, b'A']);
        let text = buf.drain_lossy();
        assert!(text.contains('\u{FFFD}'));
        assert!(text.contains('A'));
    }

    #[test]
    fn invalid_leading_byte_skipped_during_decode() {
        // Feed: [0xFF, b'A']. The 0xFF should be skipped and 'A' decoded.
        let decode = |id: u32| -> Option<Vec<u8>> {
            match id {
                0 => Some(vec![0xFF, b'A']),
                _ => None,
            }
        };
        let config = StreamConfig {
            buffer_size: 1,
            flush_on_whitespace: false,
            flush_on_newline: false,
            max_pending_tokens: 64,
        };
        let mut stream = TokenStream::new(config, decode);
        let event = stream.push_token(0);
        assert_eq!(event, Some(StreamEvent::Text("A".into())));
    }

    // -- Unknown / special tokens --

    #[test]
    fn unknown_token_does_not_crash() {
        let mut stream = default_stream();
        // Token 100 returns None from ascii_decode.
        assert_eq!(stream.push_token(100), None);
        assert_eq!(stream.stats().tokens_generated, 1);
    }

    // -- TokenBuffer unit tests --

    #[test]
    fn token_buffer_empty() {
        let mut buf = TokenBuffer::new();
        assert!(buf.is_empty());
        assert_eq!(buf.try_decode(), None);
        assert_eq!(buf.drain_lossy(), "");
    }

    #[test]
    fn token_buffer_valid_ascii() {
        let mut buf = TokenBuffer::new();
        buf.push_bytes(b"hello");
        assert_eq!(buf.try_decode(), Some("hello".into()));
        assert!(buf.is_empty());
    }

    #[test]
    fn token_buffer_partial_utf8() {
        let mut buf = TokenBuffer::new();
        // First byte of 'é' (0xC3 0xA9).
        buf.push_bytes(&[0xC3]);
        assert_eq!(buf.try_decode(), None); // incomplete
        buf.push_bytes(&[0xA9]);
        assert_eq!(buf.try_decode(), Some("é".into()));
    }

    #[test]
    fn token_buffer_drain_lossy_replaces_invalid() {
        let mut buf = TokenBuffer::new();
        buf.push_bytes(&[0xFF, b'A']);
        let text = buf.drain_lossy();
        assert!(text.contains('\u{FFFD}'));
        assert!(text.contains('A'));
    }

    #[test]
    fn token_buffer_len() {
        let mut buf = TokenBuffer::new();
        assert_eq!(buf.len(), 0);
        buf.push_bytes(b"abc");
        assert_eq!(buf.len(), 3);
    }
}
