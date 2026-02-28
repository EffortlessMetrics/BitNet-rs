//! Streaming output aggregator for combining and buffering streaming outputs.
//!
//! Provides chunk ordering, token accumulation at word/sentence boundaries,
//! stream fan-out/fan-in, and configurable backpressure strategies.

use std::collections::{BTreeMap, HashMap, VecDeque};

// ---------------------------------------------------------------------------
// StreamConfig
// ---------------------------------------------------------------------------

/// Configuration for streaming aggregation behaviour.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Maximum number of chunks the buffer will hold before applying
    /// backpressure.
    pub buffer_size: usize,
    /// Flush interval in milliseconds (advisory – callers drive flushing).
    pub flush_interval_ms: u64,
    /// Delimiter inserted between chunks when joining text.
    pub chunk_delimiter: String,
    /// Hard cap on pending (un-drained) chunks.  Zero means unlimited.
    pub max_pending_chunks: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1024,
            flush_interval_ms: 50,
            chunk_delimiter: String::new(),
            max_pending_chunks: 0,
        }
    }
}

impl StreamConfig {
    /// Validate the configuration, returning an error string on failure.
    pub fn validate(&self) -> Result<(), String> {
        if self.buffer_size == 0 {
            return Err("buffer_size must be > 0".into());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// StreamChunk
// ---------------------------------------------------------------------------

/// A single output chunk produced by a stream.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamChunk {
    /// Identifies the originating sequence / request.
    pub sequence_id: u64,
    /// Zero-based position inside the sequence.
    pub chunk_index: u64,
    /// Decoded token text (may be empty for non-text tokens).
    pub token_text: String,
    /// Vocabulary token id.
    pub token_id: u32,
    /// Log-probability of this token (NaN when unavailable).
    pub logprob: f64,
    /// `true` when this is the last chunk of the sequence.
    pub is_final: bool,
    /// Wall-clock timestamp in milliseconds.
    pub timestamp_ms: u64,
}

// ---------------------------------------------------------------------------
// ChunkBuffer
// ---------------------------------------------------------------------------

/// Ordered buffer that accepts out-of-order [`StreamChunk`]s and drains them
/// in chunk-index order.
#[derive(Debug)]
pub struct ChunkBuffer {
    /// Chunks keyed by `chunk_index` for O(log n) ordered iteration.
    pending: BTreeMap<u64, StreamChunk>,
    /// The next chunk_index we expect to drain.
    next_drain_index: u64,
    /// Whether we have received the final chunk.
    seen_final: bool,
    /// Maximum capacity (0 = unlimited).
    capacity: usize,
}

impl ChunkBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            pending: BTreeMap::new(),
            next_drain_index: 0,
            seen_final: false,
            capacity,
        }
    }

    /// Insert a chunk.  Returns `Err` if the buffer is full (capacity > 0).
    pub fn insert(&mut self, chunk: StreamChunk) -> Result<(), StreamChunk> {
        if self.capacity > 0 && self.pending.len() >= self.capacity {
            return Err(chunk);
        }
        if chunk.is_final {
            self.seen_final = true;
        }
        // Duplicate indices silently overwrite (last-writer-wins).
        self.pending.insert(chunk.chunk_index, chunk);
        Ok(())
    }

    /// Drain all chunks that are contiguous starting from `next_drain_index`.
    pub fn drain_ordered(&mut self) -> Vec<StreamChunk> {
        let mut out = Vec::new();
        while let Some(chunk) = self.pending.remove(&self.next_drain_index) {
            self.next_drain_index += 1;
            out.push(chunk);
        }
        out
    }

    /// Returns `true` when the final chunk has been drained (buffer fully
    /// consumed).
    pub fn is_complete(&self) -> bool {
        self.seen_final && self.pending.is_empty()
    }

    /// Number of buffered (not-yet-drained) chunks.
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

// ---------------------------------------------------------------------------
// StreamAggregator
// ---------------------------------------------------------------------------

/// Combines chunks from multiple sequences into per-sequence ordered output.
#[derive(Debug)]
pub struct StreamAggregator {
    buffers: HashMap<u64, ChunkBuffer>,
    config: StreamConfig,
}

impl StreamAggregator {
    pub fn new(config: StreamConfig) -> Self {
        Self {
            buffers: HashMap::new(),
            config,
        }
    }

    /// Push a chunk into the aggregator.  Creates a buffer for new sequences
    /// automatically.
    pub fn push(&mut self, chunk: StreamChunk) -> Result<(), StreamChunk> {
        let seq = chunk.sequence_id;
        let buf = self
            .buffers
            .entry(seq)
            .or_insert_with(|| ChunkBuffer::new(self.config.buffer_size));
        buf.insert(chunk)
    }

    /// Drain ordered chunks for **all** sequences.
    pub fn drain_all(&mut self) -> HashMap<u64, Vec<StreamChunk>> {
        let mut out = HashMap::new();
        for (&seq, buf) in &mut self.buffers {
            let drained = buf.drain_ordered();
            if !drained.is_empty() {
                out.insert(seq, drained);
            }
        }
        out
    }

    /// Drain ordered chunks for a single sequence.
    pub fn drain_sequence(&mut self, sequence_id: u64) -> Vec<StreamChunk> {
        self.buffers
            .get_mut(&sequence_id)
            .map(|b| b.drain_ordered())
            .unwrap_or_default()
    }

    /// Returns `true` when the given sequence has been fully consumed.
    pub fn is_sequence_complete(&self, sequence_id: u64) -> bool {
        self.buffers
            .get(&sequence_id)
            .is_some_and(|b| b.is_complete())
    }

    /// Number of active (non-complete) sequences.
    pub fn active_sequences(&self) -> usize {
        self.buffers.values().filter(|b| !b.is_complete()).count()
    }
}

// ---------------------------------------------------------------------------
// TokenAccumulator
// ---------------------------------------------------------------------------

/// Accumulates token text and flushes at word or sentence boundaries.
#[derive(Debug, Default)]
pub struct TokenAccumulator {
    buffer: String,
}

impl TokenAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append token text to the internal buffer.
    pub fn accumulate(&mut self, text: &str) {
        self.buffer.push_str(text);
    }

    /// Flush up to the last word boundary (whitespace).  Returns the flushed
    /// text (if any) and leaves the remainder in the buffer.
    pub fn flush_word(&mut self) -> Option<String> {
        if let Some(pos) = self.buffer.rfind(|c: char| c.is_whitespace()) {
            let flushed = self.buffer[..=pos].to_string();
            self.buffer = self.buffer[pos + 1..].to_string();
            Some(flushed)
        } else {
            None
        }
    }

    /// Flush up to the last sentence-ending punctuation (`.`, `!`, `?`).
    pub fn flush_sentence(&mut self) -> Option<String> {
        let end = self
            .buffer
            .rfind(|c: char| c == '.' || c == '!' || c == '?');
        if let Some(pos) = end {
            let flushed = self.buffer[..=pos].to_string();
            self.buffer = self.buffer[pos + 1..].to_string();
            Some(flushed)
        } else {
            None
        }
    }

    /// Flush everything remaining in the buffer.
    pub fn flush_all(&mut self) -> String {
        std::mem::take(&mut self.buffer)
    }

    /// Current buffered content (non-consuming peek).
    pub fn peek(&self) -> &str {
        &self.buffer
    }

    /// Whether the accumulator is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

// ---------------------------------------------------------------------------
// StreamSplitter  (fan-out)
// ---------------------------------------------------------------------------

/// Fans a single stream out to multiple consumers by cloning each chunk.
#[derive(Debug)]
pub struct StreamSplitter {
    consumer_count: usize,
    /// Per-consumer outbox.
    outboxes: Vec<VecDeque<StreamChunk>>,
}

impl StreamSplitter {
    pub fn new(consumer_count: usize) -> Self {
        Self {
            consumer_count,
            outboxes: (0..consumer_count).map(|_| VecDeque::new()).collect(),
        }
    }

    /// Broadcast a chunk to every consumer.
    pub fn broadcast(&mut self, chunk: StreamChunk) {
        for i in 0..self.consumer_count {
            self.outboxes[i].push_back(chunk.clone());
        }
    }

    /// Drain pending chunks for a specific consumer.
    pub fn drain_consumer(&mut self, index: usize) -> Vec<StreamChunk> {
        self.outboxes
            .get_mut(index)
            .map(|q| q.drain(..).collect())
            .unwrap_or_default()
    }

    /// Number of consumers.
    pub fn consumer_count(&self) -> usize {
        self.consumer_count
    }
}

// ---------------------------------------------------------------------------
// StreamMerger  (fan-in)
// ---------------------------------------------------------------------------

/// Merges multiple input streams into a single ordered output stream.
///
/// Ordering is by `(sequence_id, chunk_index)`.
#[derive(Debug, Default)]
pub struct StreamMerger {
    buffer: Vec<StreamChunk>,
}

impl StreamMerger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a chunk from any input stream.
    pub fn push(&mut self, chunk: StreamChunk) {
        self.buffer.push(chunk);
    }

    /// Drain all buffered chunks in `(sequence_id, chunk_index)` order.
    pub fn drain_ordered(&mut self) -> Vec<StreamChunk> {
        self.buffer
            .sort_by_key(|c| (c.sequence_id, c.chunk_index));
        std::mem::take(&mut self.buffer)
    }

    /// Number of buffered chunks.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether the merger buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

// ---------------------------------------------------------------------------
// BackpressureStrategy / BackpressureHandler
// ---------------------------------------------------------------------------

/// Strategy applied when a consumer cannot keep up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackpressureStrategy {
    /// Drop the oldest buffered chunk to make room.
    DropOldest,
    /// Drop the incoming (newest) chunk.
    DropNewest,
    /// Block the producer (represented by returning `Err`).
    Block,
    /// Buffer without limit (may grow unbounded).
    BufferUnlimited,
}

/// Applies a [`BackpressureStrategy`] to a bounded chunk queue.
#[derive(Debug)]
pub struct BackpressureHandler {
    strategy: BackpressureStrategy,
    queue: VecDeque<StreamChunk>,
    capacity: usize,
    dropped_count: u64,
}

impl BackpressureHandler {
    pub fn new(strategy: BackpressureStrategy, capacity: usize) -> Self {
        Self {
            strategy,
            queue: VecDeque::new(),
            capacity,
            dropped_count: 0,
        }
    }

    /// Offer a chunk to the handler.
    ///
    /// - `DropOldest` – evicts the front of the queue.
    /// - `DropNewest` – silently drops the incoming chunk.
    /// - `Block` – returns `Err(chunk)` so the caller can retry.
    /// - `BufferUnlimited` – always accepts.
    pub fn offer(&mut self, chunk: StreamChunk) -> Result<(), StreamChunk> {
        if self.strategy == BackpressureStrategy::BufferUnlimited
            || self.queue.len() < self.capacity
        {
            self.queue.push_back(chunk);
            return Ok(());
        }
        match self.strategy {
            BackpressureStrategy::DropOldest => {
                self.queue.pop_front();
                self.dropped_count += 1;
                self.queue.push_back(chunk);
                Ok(())
            }
            BackpressureStrategy::DropNewest => {
                self.dropped_count += 1;
                Ok(())
            }
            BackpressureStrategy::Block => Err(chunk),
            BackpressureStrategy::BufferUnlimited => unreachable!(),
        }
    }

    /// Drain up to `n` chunks from the front of the queue.
    pub fn drain(&mut self, n: usize) -> Vec<StreamChunk> {
        let take = n.min(self.queue.len());
        self.queue.drain(..take).collect()
    }

    /// Total chunks dropped since creation.
    pub fn dropped_count(&self) -> u64 {
        self.dropped_count
    }

    /// Current queue length.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// The active strategy.
    pub fn strategy(&self) -> BackpressureStrategy {
        self.strategy
    }
}

// ---------------------------------------------------------------------------
// StreamMetrics
// ---------------------------------------------------------------------------

/// Tracks streaming performance counters.
#[derive(Debug, Clone)]
pub struct StreamMetrics {
    /// Total chunks received.
    pub total_chunks: u64,
    /// Total chunks successfully delivered.
    pub delivered_chunks: u64,
    /// Total chunks dropped (by backpressure).
    pub dropped_chunks: u64,
    /// Timestamp (ms) of the first chunk received.
    pub first_chunk_ts: Option<u64>,
    /// Timestamp (ms) of the most recent chunk received.
    pub last_chunk_ts: Option<u64>,
    /// Sum of per-chunk latencies (for averaging).
    latency_sum_ms: u64,
    latency_count: u64,
    /// Peak buffer utilisation observed.
    pub peak_buffer_len: usize,
}

impl Default for StreamMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamMetrics {
    pub fn new() -> Self {
        Self {
            total_chunks: 0,
            delivered_chunks: 0,
            dropped_chunks: 0,
            first_chunk_ts: None,
            last_chunk_ts: None,
            latency_sum_ms: 0,
            latency_count: 0,
            peak_buffer_len: 0,
        }
    }

    /// Record a received chunk.
    pub fn record_received(&mut self, timestamp_ms: u64) {
        self.total_chunks += 1;
        if self.first_chunk_ts.is_none() {
            self.first_chunk_ts = Some(timestamp_ms);
        }
        self.last_chunk_ts = Some(timestamp_ms);
    }

    /// Record a delivered chunk.
    pub fn record_delivered(&mut self) {
        self.delivered_chunks += 1;
    }

    /// Record a dropped chunk.
    pub fn record_dropped(&mut self) {
        self.dropped_chunks += 1;
    }

    /// Record a per-chunk latency sample.
    pub fn record_latency(&mut self, latency_ms: u64) {
        self.latency_sum_ms += latency_ms;
        self.latency_count += 1;
    }

    /// Update peak buffer utilisation.
    pub fn update_buffer_len(&mut self, current_len: usize) {
        if current_len > self.peak_buffer_len {
            self.peak_buffer_len = current_len;
        }
    }

    /// Average latency in milliseconds, or `None` if no samples recorded.
    pub fn avg_latency_ms(&self) -> Option<f64> {
        if self.latency_count == 0 {
            None
        } else {
            Some(self.latency_sum_ms as f64 / self.latency_count as f64)
        }
    }

    /// Throughput in chunks per second, or `None` if fewer than two
    /// timestamps.
    pub fn chunks_per_sec(&self) -> Option<f64> {
        match (self.first_chunk_ts, self.last_chunk_ts) {
            (Some(first), Some(last)) if last > first => {
                let elapsed_s = (last - first) as f64 / 1000.0;
                Some(self.total_chunks as f64 / elapsed_s)
            }
            _ => None,
        }
    }

    /// Time to first token in milliseconds relative to a given start time.
    pub fn time_to_first_token(&self, start_ms: u64) -> Option<u64> {
        self.first_chunk_ts.map(|ts| ts.saturating_sub(start_ms))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- helpers -----------------------------------------------------------

    fn make_chunk(seq: u64, idx: u64, text: &str, is_final: bool) -> StreamChunk {
        StreamChunk {
            sequence_id: seq,
            chunk_index: idx,
            token_text: text.to_string(),
            token_id: idx as u32,
            logprob: 0.0,
            is_final,
            timestamp_ms: 100 + idx,
        }
    }

    // -----------------------------------------------------------------------
    // StreamConfig
    // -----------------------------------------------------------------------

    #[test]
    fn config_default_is_valid() {
        assert!(StreamConfig::default().validate().is_ok());
    }

    #[test]
    fn config_zero_buffer_is_invalid() {
        let mut cfg = StreamConfig::default();
        cfg.buffer_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_custom_delimiter() {
        let cfg = StreamConfig {
            chunk_delimiter: "|".into(),
            ..Default::default()
        };
        assert_eq!(cfg.chunk_delimiter, "|");
    }

    #[test]
    fn config_max_pending_zero_means_unlimited() {
        let cfg = StreamConfig::default();
        assert_eq!(cfg.max_pending_chunks, 0);
    }

    // -----------------------------------------------------------------------
    // ChunkBuffer – ordering
    // -----------------------------------------------------------------------

    #[test]
    fn buffer_ordered_insert_drain() {
        let mut buf = ChunkBuffer::new(0);
        buf.insert(make_chunk(1, 0, "a", false)).unwrap();
        buf.insert(make_chunk(1, 1, "b", false)).unwrap();
        buf.insert(make_chunk(1, 2, "c", true)).unwrap();

        let out = buf.drain_ordered();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].token_text, "a");
        assert_eq!(out[2].token_text, "c");
    }

    #[test]
    fn buffer_out_of_order_insert_drain() {
        let mut buf = ChunkBuffer::new(0);
        buf.insert(make_chunk(1, 2, "c", true)).unwrap();
        buf.insert(make_chunk(1, 0, "a", false)).unwrap();
        buf.insert(make_chunk(1, 1, "b", false)).unwrap();

        let out = buf.drain_ordered();
        let texts: Vec<&str> = out.iter().map(|c| c.token_text.as_str()).collect();
        assert_eq!(texts, vec!["a", "b", "c"]);
    }

    #[test]
    fn buffer_partial_drain_waits_for_gap() {
        let mut buf = ChunkBuffer::new(0);
        buf.insert(make_chunk(1, 0, "a", false)).unwrap();
        // skip index 1
        buf.insert(make_chunk(1, 2, "c", false)).unwrap();

        let out = buf.drain_ordered();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].token_text, "a");

        // fill the gap
        buf.insert(make_chunk(1, 1, "b", false)).unwrap();
        let out2 = buf.drain_ordered();
        assert_eq!(out2.len(), 2);
        assert_eq!(out2[0].token_text, "b");
        assert_eq!(out2[1].token_text, "c");
    }

    #[test]
    fn buffer_is_complete_after_final_drained() {
        let mut buf = ChunkBuffer::new(0);
        buf.insert(make_chunk(1, 0, "x", true)).unwrap();
        assert!(!buf.is_complete()); // not drained yet
        buf.drain_ordered();
        assert!(buf.is_complete());
    }

    #[test]
    fn buffer_not_complete_without_final() {
        let mut buf = ChunkBuffer::new(0);
        buf.insert(make_chunk(1, 0, "x", false)).unwrap();
        buf.drain_ordered();
        assert!(!buf.is_complete());
    }

    #[test]
    fn buffer_empty_drain_returns_empty() {
        let mut buf = ChunkBuffer::new(0);
        assert!(buf.drain_ordered().is_empty());
    }

    #[test]
    fn buffer_len_and_is_empty() {
        let mut buf = ChunkBuffer::new(0);
        assert!(buf.is_empty());
        buf.insert(make_chunk(1, 0, "a", false)).unwrap();
        assert_eq!(buf.len(), 1);
        assert!(!buf.is_empty());
    }

    // -----------------------------------------------------------------------
    // ChunkBuffer – overflow
    // -----------------------------------------------------------------------

    #[test]
    fn buffer_overflow_rejects_when_full() {
        let mut buf = ChunkBuffer::new(2);
        buf.insert(make_chunk(1, 0, "a", false)).unwrap();
        buf.insert(make_chunk(1, 1, "b", false)).unwrap();
        let res = buf.insert(make_chunk(1, 2, "c", false));
        assert!(res.is_err());
    }

    #[test]
    fn buffer_overflow_unlimited_accepts() {
        let mut buf = ChunkBuffer::new(0);
        for i in 0..1000 {
            buf.insert(make_chunk(1, i, "x", false)).unwrap();
        }
        assert_eq!(buf.len(), 1000);
    }

    // -----------------------------------------------------------------------
    // ChunkBuffer – duplicate indices
    // -----------------------------------------------------------------------

    #[test]
    fn buffer_duplicate_index_overwrites() {
        let mut buf = ChunkBuffer::new(0);
        buf.insert(make_chunk(1, 0, "old", false)).unwrap();
        buf.insert(make_chunk(1, 0, "new", false)).unwrap();
        let out = buf.drain_ordered();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].token_text, "new");
    }

    // -----------------------------------------------------------------------
    // StreamAggregator
    // -----------------------------------------------------------------------

    #[test]
    fn aggregator_single_sequence() {
        let mut agg = StreamAggregator::new(StreamConfig::default());
        agg.push(make_chunk(1, 0, "hello", false)).unwrap();
        agg.push(make_chunk(1, 1, " world", true)).unwrap();
        let out = agg.drain_sequence(1);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].token_text, "hello");
    }

    #[test]
    fn aggregator_multiple_sequences() {
        let mut agg = StreamAggregator::new(StreamConfig::default());
        agg.push(make_chunk(1, 0, "a", false)).unwrap();
        agg.push(make_chunk(2, 0, "x", false)).unwrap();
        agg.push(make_chunk(1, 1, "b", true)).unwrap();
        agg.push(make_chunk(2, 1, "y", true)).unwrap();

        let all = agg.drain_all();
        assert_eq!(all[&1].len(), 2);
        assert_eq!(all[&2].len(), 2);
    }

    #[test]
    fn aggregator_sequence_complete() {
        let mut agg = StreamAggregator::new(StreamConfig::default());
        agg.push(make_chunk(1, 0, "a", true)).unwrap();
        assert!(!agg.is_sequence_complete(1)); // not drained
        agg.drain_sequence(1);
        assert!(agg.is_sequence_complete(1));
    }

    #[test]
    fn aggregator_active_sequences() {
        let mut agg = StreamAggregator::new(StreamConfig::default());
        agg.push(make_chunk(1, 0, "a", true)).unwrap();
        agg.push(make_chunk(2, 0, "b", false)).unwrap();
        agg.drain_sequence(1);
        assert_eq!(agg.active_sequences(), 1); // seq 2 still active
    }

    #[test]
    fn aggregator_drain_unknown_sequence() {
        let mut agg = StreamAggregator::new(StreamConfig::default());
        assert!(agg.drain_sequence(999).is_empty());
    }

    #[test]
    fn aggregator_out_of_order_multi_sequence() {
        let mut agg = StreamAggregator::new(StreamConfig::default());
        agg.push(make_chunk(1, 2, "c", true)).unwrap();
        agg.push(make_chunk(1, 0, "a", false)).unwrap();
        agg.push(make_chunk(1, 1, "b", false)).unwrap();

        let out = agg.drain_sequence(1);
        let texts: Vec<&str> = out.iter().map(|c| c.token_text.as_str()).collect();
        assert_eq!(texts, vec!["a", "b", "c"]);
    }

    // -----------------------------------------------------------------------
    // TokenAccumulator – word boundaries
    // -----------------------------------------------------------------------

    #[test]
    fn accumulator_flush_word_basic() {
        let mut acc = TokenAccumulator::new();
        acc.accumulate("hello world foo");
        let flushed = acc.flush_word().unwrap();
        assert_eq!(flushed, "hello world ");
        assert_eq!(acc.peek(), "foo");
    }

    #[test]
    fn accumulator_flush_word_no_boundary() {
        let mut acc = TokenAccumulator::new();
        acc.accumulate("hello");
        assert!(acc.flush_word().is_none());
    }

    #[test]
    fn accumulator_flush_word_trailing_space() {
        let mut acc = TokenAccumulator::new();
        acc.accumulate("hello ");
        let flushed = acc.flush_word().unwrap();
        assert_eq!(flushed, "hello ");
        assert!(acc.peek().is_empty());
    }

    // -----------------------------------------------------------------------
    // TokenAccumulator – sentence boundaries
    // -----------------------------------------------------------------------

    #[test]
    fn accumulator_flush_sentence_period() {
        let mut acc = TokenAccumulator::new();
        acc.accumulate("Hello world. Next");
        let flushed = acc.flush_sentence().unwrap();
        assert_eq!(flushed, "Hello world.");
        assert_eq!(acc.peek(), " Next");
    }

    #[test]
    fn accumulator_flush_sentence_exclamation() {
        let mut acc = TokenAccumulator::new();
        acc.accumulate("Wow! More");
        let flushed = acc.flush_sentence().unwrap();
        assert_eq!(flushed, "Wow!");
        assert_eq!(acc.peek(), " More");
    }

    #[test]
    fn accumulator_flush_sentence_question() {
        let mut acc = TokenAccumulator::new();
        acc.accumulate("Really? Yes");
        let flushed = acc.flush_sentence().unwrap();
        assert_eq!(flushed, "Really?");
        assert_eq!(acc.peek(), " Yes");
    }

    #[test]
    fn accumulator_flush_sentence_none() {
        let mut acc = TokenAccumulator::new();
        acc.accumulate("no sentence ending");
        assert!(acc.flush_sentence().is_none());
    }

    #[test]
    fn accumulator_flush_all() {
        let mut acc = TokenAccumulator::new();
        acc.accumulate("leftover");
        assert_eq!(acc.flush_all(), "leftover");
        assert!(acc.is_empty());
    }

    #[test]
    fn accumulator_empty() {
        let acc = TokenAccumulator::new();
        assert!(acc.is_empty());
        assert_eq!(acc.peek(), "");
    }

    #[test]
    fn accumulator_multiple_accumulate() {
        let mut acc = TokenAccumulator::new();
        acc.accumulate("one ");
        acc.accumulate("two ");
        acc.accumulate("three");
        assert_eq!(acc.peek(), "one two three");
    }

    // -----------------------------------------------------------------------
    // StreamSplitter (fan-out)
    // -----------------------------------------------------------------------

    #[test]
    fn splitter_all_consumers_get_chunk() {
        let mut sp = StreamSplitter::new(3);
        sp.broadcast(make_chunk(1, 0, "a", false));
        for i in 0..3 {
            let out = sp.drain_consumer(i);
            assert_eq!(out.len(), 1);
            assert_eq!(out[0].token_text, "a");
        }
    }

    #[test]
    fn splitter_multiple_chunks() {
        let mut sp = StreamSplitter::new(2);
        sp.broadcast(make_chunk(1, 0, "a", false));
        sp.broadcast(make_chunk(1, 1, "b", true));
        let out = sp.drain_consumer(0);
        assert_eq!(out.len(), 2);
        assert_eq!(out[1].token_text, "b");
    }

    #[test]
    fn splitter_consumer_count() {
        let sp = StreamSplitter::new(5);
        assert_eq!(sp.consumer_count(), 5);
    }

    #[test]
    fn splitter_drain_invalid_consumer() {
        let mut sp = StreamSplitter::new(1);
        sp.broadcast(make_chunk(1, 0, "a", false));
        assert!(sp.drain_consumer(99).is_empty());
    }

    #[test]
    fn splitter_independent_drain() {
        let mut sp = StreamSplitter::new(2);
        sp.broadcast(make_chunk(1, 0, "a", false));
        // drain consumer 0, leave consumer 1
        let _ = sp.drain_consumer(0);
        let out1 = sp.drain_consumer(1);
        assert_eq!(out1.len(), 1);
    }

    // -----------------------------------------------------------------------
    // StreamMerger (fan-in)
    // -----------------------------------------------------------------------

    #[test]
    fn merger_interleaved_ordering() {
        let mut m = StreamMerger::new();
        m.push(make_chunk(2, 0, "x", false));
        m.push(make_chunk(1, 1, "b", false));
        m.push(make_chunk(1, 0, "a", false));
        m.push(make_chunk(2, 1, "y", true));

        let out = m.drain_ordered();
        let keys: Vec<(u64, u64)> = out.iter().map(|c| (c.sequence_id, c.chunk_index)).collect();
        assert_eq!(keys, vec![(1, 0), (1, 1), (2, 0), (2, 1)]);
    }

    #[test]
    fn merger_empty() {
        let mut m = StreamMerger::new();
        assert!(m.is_empty());
        assert!(m.drain_ordered().is_empty());
    }

    #[test]
    fn merger_single_chunk() {
        let mut m = StreamMerger::new();
        m.push(make_chunk(1, 0, "only", true));
        let out = m.drain_ordered();
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn merger_len() {
        let mut m = StreamMerger::new();
        m.push(make_chunk(1, 0, "a", false));
        m.push(make_chunk(1, 1, "b", false));
        assert_eq!(m.len(), 2);
    }

    // -----------------------------------------------------------------------
    // BackpressureHandler – DropOldest
    // -----------------------------------------------------------------------

    #[test]
    fn bp_drop_oldest_evicts_front() {
        let mut h = BackpressureHandler::new(BackpressureStrategy::DropOldest, 2);
        h.offer(make_chunk(1, 0, "a", false)).unwrap();
        h.offer(make_chunk(1, 1, "b", false)).unwrap();
        h.offer(make_chunk(1, 2, "c", false)).unwrap(); // evicts "a"

        assert_eq!(h.dropped_count(), 1);
        let out = h.drain(10);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].token_text, "b");
        assert_eq!(out[1].token_text, "c");
    }

    // -----------------------------------------------------------------------
    // BackpressureHandler – DropNewest
    // -----------------------------------------------------------------------

    #[test]
    fn bp_drop_newest_discards_incoming() {
        let mut h = BackpressureHandler::new(BackpressureStrategy::DropNewest, 2);
        h.offer(make_chunk(1, 0, "a", false)).unwrap();
        h.offer(make_chunk(1, 1, "b", false)).unwrap();
        h.offer(make_chunk(1, 2, "c", false)).unwrap(); // dropped

        assert_eq!(h.dropped_count(), 1);
        let out = h.drain(10);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].token_text, "a");
    }

    // -----------------------------------------------------------------------
    // BackpressureHandler – Block
    // -----------------------------------------------------------------------

    #[test]
    fn bp_block_returns_err() {
        let mut h = BackpressureHandler::new(BackpressureStrategy::Block, 1);
        h.offer(make_chunk(1, 0, "a", false)).unwrap();
        let res = h.offer(make_chunk(1, 1, "b", false));
        assert!(res.is_err());
        assert_eq!(h.dropped_count(), 0);
    }

    #[test]
    fn bp_block_accepts_after_drain() {
        let mut h = BackpressureHandler::new(BackpressureStrategy::Block, 1);
        h.offer(make_chunk(1, 0, "a", false)).unwrap();
        h.drain(1);
        h.offer(make_chunk(1, 1, "b", false)).unwrap();
        assert_eq!(h.len(), 1);
    }

    // -----------------------------------------------------------------------
    // BackpressureHandler – BufferUnlimited
    // -----------------------------------------------------------------------

    #[test]
    fn bp_unlimited_never_drops() {
        let mut h = BackpressureHandler::new(BackpressureStrategy::BufferUnlimited, 1);
        for i in 0..100 {
            h.offer(make_chunk(1, i, "x", false)).unwrap();
        }
        assert_eq!(h.len(), 100);
        assert_eq!(h.dropped_count(), 0);
    }

    // -----------------------------------------------------------------------
    // BackpressureHandler – misc
    // -----------------------------------------------------------------------

    #[test]
    fn bp_strategy_accessor() {
        let h = BackpressureHandler::new(BackpressureStrategy::DropOldest, 4);
        assert_eq!(h.strategy(), BackpressureStrategy::DropOldest);
    }

    #[test]
    fn bp_drain_partial() {
        let mut h = BackpressureHandler::new(BackpressureStrategy::BufferUnlimited, 0);
        for i in 0..5 {
            h.offer(make_chunk(1, i, "x", false)).unwrap();
        }
        let out = h.drain(3);
        assert_eq!(out.len(), 3);
        assert_eq!(h.len(), 2);
    }

    #[test]
    fn bp_empty() {
        let h = BackpressureHandler::new(BackpressureStrategy::Block, 10);
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
    }

    // -----------------------------------------------------------------------
    // StreamMetrics
    // -----------------------------------------------------------------------

    #[test]
    fn metrics_initial_state() {
        let m = StreamMetrics::new();
        assert_eq!(m.total_chunks, 0);
        assert_eq!(m.delivered_chunks, 0);
        assert_eq!(m.dropped_chunks, 0);
        assert!(m.first_chunk_ts.is_none());
        assert!(m.avg_latency_ms().is_none());
        assert!(m.chunks_per_sec().is_none());
    }

    #[test]
    fn metrics_record_received() {
        let mut m = StreamMetrics::new();
        m.record_received(1000);
        m.record_received(1050);
        assert_eq!(m.total_chunks, 2);
        assert_eq!(m.first_chunk_ts, Some(1000));
        assert_eq!(m.last_chunk_ts, Some(1050));
    }

    #[test]
    fn metrics_record_delivered() {
        let mut m = StreamMetrics::new();
        m.record_delivered();
        m.record_delivered();
        assert_eq!(m.delivered_chunks, 2);
    }

    #[test]
    fn metrics_record_dropped() {
        let mut m = StreamMetrics::new();
        m.record_dropped();
        assert_eq!(m.dropped_chunks, 1);
    }

    #[test]
    fn metrics_avg_latency() {
        let mut m = StreamMetrics::new();
        m.record_latency(10);
        m.record_latency(20);
        assert!((m.avg_latency_ms().unwrap() - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_chunks_per_sec() {
        let mut m = StreamMetrics::new();
        m.record_received(1000);
        m.record_received(2000);
        m.record_received(3000);
        // 3 chunks over 2 seconds = 1.5 c/s
        let cps = m.chunks_per_sec().unwrap();
        assert!((cps - 1.5).abs() < 0.01);
    }

    #[test]
    fn metrics_chunks_per_sec_single_ts() {
        let mut m = StreamMetrics::new();
        m.record_received(1000);
        assert!(m.chunks_per_sec().is_none());
    }

    #[test]
    fn metrics_time_to_first_token() {
        let mut m = StreamMetrics::new();
        m.record_received(150);
        assert_eq!(m.time_to_first_token(100), Some(50));
    }

    #[test]
    fn metrics_time_to_first_token_none() {
        let m = StreamMetrics::new();
        assert!(m.time_to_first_token(100).is_none());
    }

    #[test]
    fn metrics_peak_buffer_len() {
        let mut m = StreamMetrics::new();
        m.update_buffer_len(5);
        m.update_buffer_len(10);
        m.update_buffer_len(3);
        assert_eq!(m.peak_buffer_len, 10);
    }

    #[test]
    fn metrics_default() {
        let m = StreamMetrics::default();
        assert_eq!(m.total_chunks, 0);
    }

    // -----------------------------------------------------------------------
    // Empty / single-chunk edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn single_final_chunk_buffer() {
        let mut buf = ChunkBuffer::new(0);
        buf.insert(make_chunk(1, 0, "only", true)).unwrap();
        let out = buf.drain_ordered();
        assert_eq!(out.len(), 1);
        assert!(buf.is_complete());
    }

    #[test]
    fn single_final_chunk_aggregator() {
        let mut agg = StreamAggregator::new(StreamConfig::default());
        agg.push(make_chunk(1, 0, "only", true)).unwrap();
        let out = agg.drain_sequence(1);
        assert_eq!(out.len(), 1);
        assert!(agg.is_sequence_complete(1));
    }

    #[test]
    fn empty_stream_aggregator() {
        let mut agg = StreamAggregator::new(StreamConfig::default());
        let all = agg.drain_all();
        assert!(all.is_empty());
    }

    // -----------------------------------------------------------------------
    // Property-style ordering invariant tests
    // -----------------------------------------------------------------------

    #[test]
    fn property_drain_always_sorted() {
        let mut buf = ChunkBuffer::new(0);
        // Insert in reverse order
        for i in (0..20).rev() {
            buf.insert(make_chunk(1, i, "", i == 0)).unwrap();
        }
        let out = buf.drain_ordered();
        assert_eq!(out.len(), 20);
        for (i, c) in out.iter().enumerate() {
            assert_eq!(c.chunk_index, i as u64);
        }
    }

    #[test]
    fn property_merger_stable_order_within_sequence() {
        let mut m = StreamMerger::new();
        for i in (0..10).rev() {
            m.push(make_chunk(1, i, "", false));
        }
        let out = m.drain_ordered();
        for i in 0..10u64 {
            assert_eq!(out[i as usize].chunk_index, i);
        }
    }

    #[test]
    fn property_splitter_preserves_order() {
        let mut sp = StreamSplitter::new(3);
        for i in 0..10 {
            sp.broadcast(make_chunk(1, i, "", false));
        }
        for consumer in 0..3 {
            let out = sp.drain_consumer(consumer);
            assert_eq!(out.len(), 10);
            for (i, c) in out.iter().enumerate() {
                assert_eq!(c.chunk_index, i as u64);
            }
        }
    }

    #[test]
    fn property_backpressure_drop_oldest_keeps_newest() {
        let mut h = BackpressureHandler::new(BackpressureStrategy::DropOldest, 5);
        for i in 0..20 {
            h.offer(make_chunk(1, i, &format!("{i}"), false)).unwrap();
        }
        let out = h.drain(10);
        assert_eq!(out.len(), 5);
        // Should keep the last 5: 15..20
        for (j, c) in out.iter().enumerate() {
            assert_eq!(c.chunk_index, 15 + j as u64);
        }
    }

    #[test]
    fn property_drop_newest_keeps_oldest() {
        let mut h = BackpressureHandler::new(BackpressureStrategy::DropNewest, 5);
        for i in 0..20 {
            h.offer(make_chunk(1, i, &format!("{i}"), false)).unwrap();
        }
        let out = h.drain(10);
        assert_eq!(out.len(), 5);
        // Should keep the first 5: 0..5
        for (j, c) in out.iter().enumerate() {
            assert_eq!(c.chunk_index, j as u64);
        }
    }

    // -----------------------------------------------------------------------
    // Additional edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn chunk_logprob_nan() {
        let c = StreamChunk {
            sequence_id: 1,
            chunk_index: 0,
            token_text: String::new(),
            token_id: 0,
            logprob: f64::NAN,
            is_final: false,
            timestamp_ms: 0,
        };
        assert!(c.logprob.is_nan());
    }

    #[test]
    fn aggregator_interleaved_push_drain() {
        let mut agg = StreamAggregator::new(StreamConfig::default());
        agg.push(make_chunk(1, 0, "a", false)).unwrap();
        let out1 = agg.drain_sequence(1);
        assert_eq!(out1.len(), 1);

        agg.push(make_chunk(1, 1, "b", true)).unwrap();
        let out2 = agg.drain_sequence(1);
        assert_eq!(out2.len(), 1);
        assert!(agg.is_sequence_complete(1));
    }

    #[test]
    fn accumulator_unicode() {
        let mut acc = TokenAccumulator::new();
        acc.accumulate("こんにちは 世界");
        let flushed = acc.flush_word().unwrap();
        assert_eq!(flushed, "こんにちは ");
        assert_eq!(acc.peek(), "世界");
    }

    #[test]
    fn splitter_zero_consumers() {
        let mut sp = StreamSplitter::new(0);
        sp.broadcast(make_chunk(1, 0, "a", false));
        assert_eq!(sp.consumer_count(), 0);
    }

    #[test]
    fn metrics_time_to_first_token_saturating() {
        let mut m = StreamMetrics::new();
        m.record_received(50);
        // start_ms > first_chunk_ts → saturating_sub → 0
        assert_eq!(m.time_to_first_token(100), Some(0));
    }

    #[test]
    fn bp_drain_more_than_available() {
        let mut h = BackpressureHandler::new(BackpressureStrategy::BufferUnlimited, 0);
        h.offer(make_chunk(1, 0, "a", false)).unwrap();
        let out = h.drain(100);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn buffer_final_in_middle_still_detected() {
        let mut buf = ChunkBuffer::new(0);
        buf.insert(make_chunk(1, 0, "a", false)).unwrap();
        buf.insert(make_chunk(1, 1, "b", true)).unwrap();
        buf.insert(make_chunk(1, 2, "c", false)).unwrap();
        assert!(buf.seen_final);
    }

    #[test]
    fn merger_same_sequence_reverse() {
        let mut m = StreamMerger::new();
        m.push(make_chunk(1, 4, "e", true));
        m.push(make_chunk(1, 3, "d", false));
        m.push(make_chunk(1, 2, "c", false));
        m.push(make_chunk(1, 1, "b", false));
        m.push(make_chunk(1, 0, "a", false));
        let out = m.drain_ordered();
        let indices: Vec<u64> = out.iter().map(|c| c.chunk_index).collect();
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn aggregator_many_sequences() {
        let mut agg = StreamAggregator::new(StreamConfig::default());
        for seq in 0..50 {
            agg.push(make_chunk(seq, 0, "x", true)).unwrap();
        }
        let all = agg.drain_all();
        assert_eq!(all.len(), 50);
    }
}
