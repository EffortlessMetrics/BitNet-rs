//! Data pipeline for efficient batched data loading and transformation.
//!
//! Provides a composable pipeline: source → transforms → batch → output,
//! with support for shuffling, filtering, mapping, prefetch, and metrics.

use std::collections::VecDeque;
use std::fmt;
use std::time::{Duration, Instant};

// ── Configuration ───────────────────────────────────────────────────────────

/// Settings controlling pipeline behaviour.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Maximum items held in internal buffers.
    pub buffer_size: usize,
    /// Number of logical worker slots (informational; no threading here).
    pub num_workers: usize,
    /// How many items to prefetch ahead of consumption.
    pub prefetch_count: usize,
    /// Items per batch emitted by [`BatchCollector`].
    pub batch_size: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self { buffer_size: 1024, num_workers: 1, prefetch_count: 2, batch_size: 32 }
    }
}

impl PipelineConfig {
    /// Create a config with the given batch size, other fields default.
    #[must_use]
    pub const fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the prefetch count.
    #[must_use]
    pub const fn with_prefetch_count(mut self, prefetch_count: usize) -> Self {
        self.prefetch_count = prefetch_count;
        self
    }

    /// Set the buffer size.
    #[must_use]
    pub const fn with_buffer_size(mut self, buffer_size: usize) -> Self {
        self.buffer_size = buffer_size;
        self
    }

    /// Set the number of workers.
    #[must_use]
    pub const fn with_num_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = num_workers;
        self
    }
}

// ── DataSource trait ────────────────────────────────────────────────────────

/// Interface for data sources that produce items sequentially.
pub trait DataSource<T> {
    /// Return the next item, or `None` when exhausted.
    fn next_item(&mut self) -> Option<T>;
    /// Whether more items remain.
    fn has_next(&self) -> bool;
    /// Reset the source to the beginning.
    fn reset(&mut self);
    /// Total number of items (if known).
    fn len(&self) -> Option<usize>;
    /// Whether the source is empty.
    fn is_empty(&self) -> bool {
        self.len() == Some(0)
    }
}

/// A simple in-memory data source backed by a `Vec`.
pub struct VecSource<T: Clone> {
    items: Vec<T>,
    pos: usize,
}

impl<T: Clone> VecSource<T> {
    pub const fn new(items: Vec<T>) -> Self {
        Self { items, pos: 0 }
    }
}

impl<T: Clone> DataSource<T> for VecSource<T> {
    fn next_item(&mut self) -> Option<T> {
        if self.pos < self.items.len() {
            let item = self.items[self.pos].clone();
            self.pos += 1;
            Some(item)
        } else {
            None
        }
    }

    fn has_next(&self) -> bool {
        self.pos < self.items.len()
    }

    fn reset(&mut self) {
        self.pos = 0;
    }

    fn len(&self) -> Option<usize> {
        Some(self.items.len())
    }
}

/// A data source that yields items from a range `[start, end)`.
pub struct RangeSource {
    start: usize,
    end: usize,
    current: usize,
}

impl RangeSource {
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end, current: start }
    }
}

impl DataSource<usize> for RangeSource {
    fn next_item(&mut self) -> Option<usize> {
        if self.current < self.end {
            let v = self.current;
            self.current += 1;
            Some(v)
        } else {
            None
        }
    }

    fn has_next(&self) -> bool {
        self.current < self.end
    }

    fn reset(&mut self) {
        self.current = self.start;
    }

    fn len(&self) -> Option<usize> {
        Some(self.end.saturating_sub(self.start))
    }
}

// ── Transform trait ─────────────────────────────────────────────────────────

/// A named transformation applied to individual items.
pub trait Transform<T> {
    /// Apply the transformation, returning `Some(result)` to keep or `None` to drop.
    fn transform(&self, item: T) -> Option<T>;
    /// Human-readable name for diagnostics.
    fn name(&self) -> &str;
}

// ── FilterTransform ─────────────────────────────────────────────────────────

/// Keeps only items satisfying a predicate.
pub struct FilterTransform<F> {
    predicate: F,
    name: String,
}

impl<F> FilterTransform<F> {
    pub fn new(predicate: F, name: impl Into<String>) -> Self {
        Self { predicate, name: name.into() }
    }
}

impl<T, F: Fn(&T) -> bool> Transform<T> for FilterTransform<F> {
    fn transform(&self, item: T) -> Option<T> {
        if (self.predicate)(&item) { Some(item) } else { None }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ── MapTransform ────────────────────────────────────────────────────────────

/// Maps each item through a function.
pub struct MapTransform<F> {
    func: F,
    name: String,
}

impl<F> MapTransform<F> {
    pub fn new(func: F, name: impl Into<String>) -> Self {
        Self { func, name: name.into() }
    }
}

impl<T, F: Fn(T) -> T> Transform<T> for MapTransform<F> {
    fn transform(&self, item: T) -> Option<T> {
        Some((self.func)(item))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ── BatchCollector ──────────────────────────────────────────────────────────

/// Collects individual items into fixed-size batches.
pub struct BatchCollector<T> {
    batch_size: usize,
    current_batch: Vec<T>,
    drop_last: bool,
}

impl<T> BatchCollector<T> {
    pub fn new(batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size must be > 0");
        Self { batch_size, current_batch: Vec::with_capacity(batch_size), drop_last: false }
    }

    /// When true, the final incomplete batch is discarded.
    #[must_use]
    pub const fn with_drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Push an item; returns a full batch when ready.
    pub fn push(&mut self, item: T) -> Option<Vec<T>> {
        self.current_batch.push(item);
        if self.current_batch.len() == self.batch_size {
            let batch =
                std::mem::replace(&mut self.current_batch, Vec::with_capacity(self.batch_size));
            Some(batch)
        } else {
            None
        }
    }

    /// Flush any remaining items as a partial batch (unless `drop_last`).
    pub fn flush(&mut self) -> Option<Vec<T>> {
        if self.current_batch.is_empty() || self.drop_last {
            self.current_batch.clear();
            None
        } else {
            let batch =
                std::mem::replace(&mut self.current_batch, Vec::with_capacity(self.batch_size));
            Some(batch)
        }
    }

    /// Number of items in the current incomplete batch.
    pub const fn pending(&self) -> usize {
        self.current_batch.len()
    }

    /// Configured batch size.
    pub const fn batch_size(&self) -> usize {
        self.batch_size
    }
}

// ── ShuffleBuffer ───────────────────────────────────────────────────────────

/// Shuffles incoming items using a fixed-size reservoir.
///
/// Items are inserted into a buffer of `capacity` slots. When the buffer is
/// full, a random slot is evicted and returned, replaced by the new item.
/// This provides approximate shuffling with bounded memory.
pub struct ShuffleBuffer<T> {
    buffer: Vec<T>,
    capacity: usize,
    rng_state: u64,
}

impl<T> ShuffleBuffer<T> {
    pub fn new(capacity: usize, seed: u64) -> Self {
        assert!(capacity > 0, "shuffle buffer capacity must be > 0");
        Self { buffer: Vec::with_capacity(capacity), capacity, rng_state: seed }
    }

    /// Simple xorshift64 PRNG (deterministic, non-cryptographic).
    const fn next_rand(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    /// Add an item. If the buffer is full, evict and return a random item.
    pub fn push(&mut self, item: T) -> Option<T> {
        if self.buffer.len() < self.capacity {
            self.buffer.push(item);
            None
        } else {
            #[allow(clippy::cast_possible_truncation)]
            let idx = (self.next_rand() as usize) % self.capacity;
            let evicted = std::mem::replace(&mut self.buffer[idx], item);
            Some(evicted)
        }
    }

    /// Drain remaining items (order is buffer-order, not shuffled further).
    pub fn drain(&mut self) -> Vec<T> {
        std::mem::take(&mut self.buffer)
    }

    pub const fn len(&self) -> usize {
        self.buffer.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub const fn capacity(&self) -> usize {
        self.capacity
    }
}

// ── PrefetchBuffer ──────────────────────────────────────────────────────────

/// Eagerly fills an internal queue from a source to reduce stalls.
pub struct PrefetchBuffer<T> {
    buffer: VecDeque<T>,
    prefetch_count: usize,
}

impl<T> PrefetchBuffer<T> {
    pub fn new(prefetch_count: usize) -> Self {
        Self { buffer: VecDeque::with_capacity(prefetch_count), prefetch_count }
    }

    /// Fill the buffer from `source` up to `prefetch_count`.
    pub fn fill<S: DataSource<T> + ?Sized>(&mut self, source: &mut S) {
        while self.buffer.len() < self.prefetch_count {
            match source.next_item() {
                Some(item) => self.buffer.push_back(item),
                None => break,
            }
        }
    }

    /// Take the next prefetched item.
    pub fn pop(&mut self) -> Option<T> {
        self.buffer.pop_front()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub const fn prefetch_count(&self) -> usize {
        self.prefetch_count
    }
}

// ── DataPipelineMetrics ─────────────────────────────────────────────────────

/// Runtime metrics collected by the pipeline.
#[derive(Debug, Clone)]
pub struct DataPipelineMetrics {
    pub items_processed: u64,
    pub batches_produced: u64,
    pub items_dropped: u64,
    pub total_transform_time: Duration,
    pub total_wall_time: Duration,
    start_time: Option<Instant>,
}

impl Default for DataPipelineMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl DataPipelineMetrics {
    pub const fn new() -> Self {
        Self {
            items_processed: 0,
            batches_produced: 0,
            items_dropped: 0,
            total_transform_time: Duration::ZERO,
            total_wall_time: Duration::ZERO,
            start_time: None,
        }
    }

    /// Start the wall-clock timer.
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Stop the wall-clock timer and accumulate elapsed time.
    pub fn stop(&mut self) {
        if let Some(t) = self.start_time.take() {
            self.total_wall_time += t.elapsed();
        }
    }

    /// Items processed per second (wall time).
    #[allow(clippy::cast_precision_loss)]
    pub fn items_per_sec(&self) -> f64 {
        let secs = self.total_wall_time.as_secs_f64();
        if secs > 0.0 { self.items_processed as f64 / secs } else { 0.0 }
    }

    /// Average transform time per item.
    pub fn avg_transform_time(&self) -> Duration {
        if self.items_processed > 0 {
            #[allow(clippy::cast_possible_truncation)]
            let divisor = self.items_processed as u32;
            self.total_transform_time / divisor
        } else {
            Duration::ZERO
        }
    }

    /// Buffer utilisation: `items_processed / (items_processed + items_dropped)`.
    #[allow(clippy::cast_precision_loss)]
    pub fn buffer_utilization(&self) -> f64 {
        let total = self.items_processed + self.items_dropped;
        if total > 0 { self.items_processed as f64 / total as f64 } else { 0.0 }
    }
}

impl fmt::Display for DataPipelineMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "items={} batches={} dropped={} throughput={:.1} items/s avg_transform={:?}",
            self.items_processed,
            self.batches_produced,
            self.items_dropped,
            self.items_per_sec(),
            self.avg_transform_time(),
        )
    }
}

// ── DataPipeline ────────────────────────────────────────────────────────────

/// Composable data pipeline: source → transforms → batch → output.
pub struct DataPipeline<T: Clone> {
    config: PipelineConfig,
    source: Box<dyn DataSource<T>>,
    transforms: Vec<Box<dyn Transform<T>>>,
    batch_collector: BatchCollector<T>,
    shuffle_buffer: Option<ShuffleBuffer<T>>,
    prefetch: PrefetchBuffer<T>,
    metrics: DataPipelineMetrics,
    exhausted: bool,
    /// Items drained from the shuffle buffer awaiting transform.
    drain_queue: VecDeque<T>,
}

impl<T: Clone + 'static> DataPipeline<T> {
    /// Build a new pipeline from a source and config.
    pub fn new(source: Box<dyn DataSource<T>>, config: PipelineConfig) -> Self {
        let batch_collector = BatchCollector::new(config.batch_size);
        let prefetch = PrefetchBuffer::new(config.prefetch_count);
        Self {
            config,
            source,
            transforms: Vec::new(),
            batch_collector,
            shuffle_buffer: None,
            prefetch,
            metrics: DataPipelineMetrics::new(),
            exhausted: false,
            drain_queue: VecDeque::new(),
        }
    }

    /// Add a transform stage.
    pub fn add_transform(&mut self, transform: Box<dyn Transform<T>>) {
        self.transforms.push(transform);
    }

    /// Enable shuffling with the given buffer capacity and seed.
    pub fn enable_shuffle(&mut self, capacity: usize, seed: u64) {
        self.shuffle_buffer = Some(ShuffleBuffer::new(capacity, seed));
    }

    /// Apply all transforms to one item. Returns `None` if any transform drops it.
    fn apply_transforms(&self, mut item: T) -> Option<T> {
        for t in &self.transforms {
            match t.transform(item) {
                Some(v) => item = v,
                None => return None,
            }
        }
        Some(item)
    }

    /// Pull the next raw item through prefetch + shuffle + transform.
    fn next_transformed_item(&mut self) -> Option<T> {
        loop {
            // First, drain any queued items from a previous shuffle drain.
            if let Some(item) = self.drain_queue.pop_front() {
                let t_start = Instant::now();
                if let Some(transformed) = self.apply_transforms(item) {
                    self.metrics.total_transform_time += t_start.elapsed();
                    self.metrics.items_processed += 1;
                    return Some(transformed);
                }
                self.metrics.total_transform_time += t_start.elapsed();
                self.metrics.items_dropped += 1;
                continue;
            }

            // Refill prefetch buffer.
            self.prefetch.fill(&mut *self.source);
            let raw = self.prefetch.pop();

            // Route through shuffle buffer if enabled.
            let candidate = if let Some(ref mut sb) = self.shuffle_buffer {
                if let Some(item) = raw {
                    sb.push(item)
                } else {
                    // Source exhausted—drain shuffle buffer into queue.
                    let remaining = sb.drain();
                    if remaining.is_empty() {
                        return None;
                    }
                    self.drain_queue.extend(remaining);
                    continue;
                }
            } else {
                raw
            };

            if let Some(item) = candidate {
                let t_start = Instant::now();
                if let Some(transformed) = self.apply_transforms(item) {
                    self.metrics.total_transform_time += t_start.elapsed();
                    self.metrics.items_processed += 1;
                    return Some(transformed);
                }
                self.metrics.total_transform_time += t_start.elapsed();
                self.metrics.items_dropped += 1;
            } else if self.shuffle_buffer.is_some() {
                // Shuffle buffer absorbed the item without evicting; loop.
            } else {
                return None;
            }
        }
    }

    /// Produce the next batch, or `None` when the source is fully consumed.
    pub fn next_batch(&mut self) -> Option<Vec<T>> {
        if self.metrics.start_time.is_none() {
            self.metrics.start();
        }

        loop {
            if self.exhausted {
                let flushed = self.batch_collector.flush();
                if flushed.is_some() {
                    self.metrics.batches_produced += 1;
                }
                return flushed;
            }

            if let Some(item) = self.next_transformed_item() {
                if let Some(batch) = self.batch_collector.push(item) {
                    self.metrics.batches_produced += 1;
                    return Some(batch);
                }
            } else {
                self.exhausted = true;
                let flushed = self.batch_collector.flush();
                if flushed.is_some() {
                    self.metrics.batches_produced += 1;
                }
                return flushed;
            }
        }
    }

    /// Collect all batches.
    pub fn collect_all(&mut self) -> Vec<Vec<T>> {
        let mut batches = Vec::new();
        while let Some(b) = self.next_batch() {
            batches.push(b);
        }
        self.metrics.stop();
        batches
    }

    /// Reset the pipeline for another pass.
    pub fn reset(&mut self) {
        self.source.reset();
        self.exhausted = false;
        self.metrics = DataPipelineMetrics::new();
        self.batch_collector = BatchCollector::new(self.config.batch_size);
        self.prefetch = PrefetchBuffer::new(self.config.prefetch_count);
        self.drain_queue.clear();
        if let Some(ref sb) = self.shuffle_buffer {
            let cap = sb.capacity();
            self.shuffle_buffer = Some(ShuffleBuffer::new(cap, 42));
        }
    }

    /// Current pipeline metrics (snapshot).
    pub const fn metrics(&self) -> &DataPipelineMetrics {
        &self.metrics
    }

    /// Pipeline configuration.
    pub const fn config(&self) -> &PipelineConfig {
        &self.config
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── PipelineConfig tests ────────────────────────────────────────────

    #[test]
    fn config_default_values() {
        let cfg = PipelineConfig::default();
        assert_eq!(cfg.buffer_size, 1024);
        assert_eq!(cfg.num_workers, 1);
        assert_eq!(cfg.prefetch_count, 2);
        assert_eq!(cfg.batch_size, 32);
    }

    #[test]
    fn config_builder_batch_size() {
        let cfg = PipelineConfig::default().with_batch_size(64);
        assert_eq!(cfg.batch_size, 64);
    }

    #[test]
    fn config_builder_prefetch() {
        let cfg = PipelineConfig::default().with_prefetch_count(8);
        assert_eq!(cfg.prefetch_count, 8);
    }

    #[test]
    fn config_builder_buffer_size() {
        let cfg = PipelineConfig::default().with_buffer_size(512);
        assert_eq!(cfg.buffer_size, 512);
    }

    #[test]
    fn config_builder_num_workers() {
        let cfg = PipelineConfig::default().with_num_workers(4);
        assert_eq!(cfg.num_workers, 4);
    }

    #[test]
    fn config_builder_chained() {
        let cfg = PipelineConfig::default()
            .with_batch_size(16)
            .with_prefetch_count(4)
            .with_buffer_size(256)
            .with_num_workers(2);
        assert_eq!(cfg.batch_size, 16);
        assert_eq!(cfg.prefetch_count, 4);
        assert_eq!(cfg.buffer_size, 256);
        assert_eq!(cfg.num_workers, 2);
    }

    // ── VecSource tests ─────────────────────────────────────────────────

    #[test]
    fn vec_source_iterates_all() {
        let mut src = VecSource::new(vec![10, 20, 30]);
        assert_eq!(src.next_item(), Some(10));
        assert_eq!(src.next_item(), Some(20));
        assert_eq!(src.next_item(), Some(30));
        assert_eq!(src.next_item(), None);
    }

    #[test]
    fn vec_source_has_next() {
        let mut src = VecSource::new(vec![1]);
        assert!(src.has_next());
        src.next_item();
        assert!(!src.has_next());
    }

    #[test]
    fn vec_source_len() {
        let src = VecSource::new(vec![1, 2, 3]);
        assert_eq!(src.len(), Some(3));
    }

    #[test]
    fn vec_source_is_empty() {
        let src: VecSource<i32> = VecSource::new(vec![]);
        assert!(src.is_empty());
    }

    #[test]
    fn vec_source_reset() {
        let mut src = VecSource::new(vec![1, 2]);
        src.next_item();
        src.next_item();
        assert!(!src.has_next());
        src.reset();
        assert!(src.has_next());
        assert_eq!(src.next_item(), Some(1));
    }

    // ── RangeSource tests ───────────────────────────────────────────────

    #[test]
    fn range_source_basic() {
        let mut src = RangeSource::new(0, 3);
        assert_eq!(src.next_item(), Some(0));
        assert_eq!(src.next_item(), Some(1));
        assert_eq!(src.next_item(), Some(2));
        assert_eq!(src.next_item(), None);
    }

    #[test]
    fn range_source_len() {
        let src = RangeSource::new(5, 10);
        assert_eq!(src.len(), Some(5));
    }

    #[test]
    fn range_source_reset() {
        let mut src = RangeSource::new(0, 2);
        src.next_item();
        src.next_item();
        assert!(!src.has_next());
        src.reset();
        assert!(src.has_next());
        assert_eq!(src.next_item(), Some(0));
    }

    #[test]
    fn range_source_empty() {
        let src = RangeSource::new(5, 5);
        assert!(!src.has_next());
        assert_eq!(src.len(), Some(0));
    }

    #[test]
    fn range_source_single() {
        let mut src = RangeSource::new(42, 43);
        assert_eq!(src.next_item(), Some(42));
        assert_eq!(src.next_item(), None);
    }

    // ── FilterTransform tests ───────────────────────────────────────────

    #[test]
    fn filter_keeps_matching() {
        let f = FilterTransform::new(|x: &i32| *x > 5, "gt5");
        assert_eq!(f.transform(10), Some(10));
    }

    #[test]
    fn filter_drops_non_matching() {
        let f = FilterTransform::new(|x: &i32| *x > 5, "gt5");
        assert_eq!(f.transform(3), None);
    }

    #[test]
    fn filter_name() {
        let f = FilterTransform::new(|_: &i32| true, "keep_all");
        assert_eq!(f.name(), "keep_all");
    }

    #[test]
    fn filter_boundary() {
        let f = FilterTransform::new(|x: &i32| *x % 2 == 0, "even");
        assert_eq!(f.transform(0), Some(0));
        assert_eq!(f.transform(1), None);
        assert_eq!(f.transform(2), Some(2));
    }

    // ── MapTransform tests ──────────────────────────────────────────────

    #[test]
    fn map_doubles() {
        let m = MapTransform::new(|x: i32| x * 2, "double");
        assert_eq!(m.transform(5), Some(10));
    }

    #[test]
    fn map_name() {
        let m = MapTransform::new(|x: i32| x, "identity");
        assert_eq!(m.name(), "identity");
    }

    #[test]
    fn map_always_returns_some() {
        let m = MapTransform::new(|x: i32| x + 1, "inc");
        for i in 0..100 {
            assert!(m.transform(i).is_some());
        }
    }

    #[test]
    fn map_identity() {
        let m = MapTransform::new(|x: i32| x, "id");
        assert_eq!(m.transform(42), Some(42));
    }

    // ── BatchCollector tests ────────────────────────────────────────────

    #[test]
    fn batch_exact_fill() {
        let mut bc = BatchCollector::new(3);
        assert!(bc.push(1).is_none());
        assert!(bc.push(2).is_none());
        let batch = bc.push(3);
        assert_eq!(batch, Some(vec![1, 2, 3]));
    }

    #[test]
    fn batch_flush_partial() {
        let mut bc = BatchCollector::new(4);
        bc.push(1);
        bc.push(2);
        let flushed = bc.flush();
        assert_eq!(flushed, Some(vec![1, 2]));
    }

    #[test]
    fn batch_flush_empty() {
        let mut bc: BatchCollector<i32> = BatchCollector::new(4);
        assert!(bc.flush().is_none());
    }

    #[test]
    fn batch_drop_last() {
        let mut bc = BatchCollector::new(3).with_drop_last(true);
        bc.push(1);
        bc.push(2);
        assert!(bc.flush().is_none());
    }

    #[test]
    fn batch_pending_count() {
        let mut bc = BatchCollector::new(5);
        assert_eq!(bc.pending(), 0);
        bc.push(10);
        bc.push(20);
        assert_eq!(bc.pending(), 2);
    }

    #[test]
    fn batch_size_accessor() {
        let bc: BatchCollector<i32> = BatchCollector::new(7);
        assert_eq!(bc.batch_size(), 7);
    }

    #[test]
    fn batch_multiple_full_batches() {
        let mut bc = BatchCollector::new(2);
        assert!(bc.push(1).is_none());
        assert_eq!(bc.push(2), Some(vec![1, 2]));
        assert!(bc.push(3).is_none());
        assert_eq!(bc.push(4), Some(vec![3, 4]));
    }

    #[test]
    #[should_panic(expected = "batch_size must be > 0")]
    fn batch_zero_panics() {
        let _bc: BatchCollector<i32> = BatchCollector::new(0);
    }

    // ── ShuffleBuffer tests ─────────────────────────────────────────────

    #[test]
    fn shuffle_fills_before_evicting() {
        let mut sb = ShuffleBuffer::new(3, 42);
        assert!(sb.push(1).is_none());
        assert!(sb.push(2).is_none());
        assert!(sb.push(3).is_none());
        // Buffer now full, next push evicts.
        assert!(sb.push(4).is_some());
    }

    #[test]
    fn shuffle_drain() {
        let mut sb = ShuffleBuffer::new(5, 1);
        sb.push(10);
        sb.push(20);
        let drained = sb.drain();
        assert_eq!(drained.len(), 2);
        assert!(sb.is_empty());
    }

    #[test]
    fn shuffle_len_and_capacity() {
        let mut sb = ShuffleBuffer::new(4, 0);
        assert_eq!(sb.capacity(), 4);
        assert_eq!(sb.len(), 0);
        sb.push(1);
        assert_eq!(sb.len(), 1);
    }

    #[test]
    fn shuffle_deterministic_with_seed() {
        let run = |seed: u64| -> Vec<i32> {
            let mut sb = ShuffleBuffer::new(3, seed);
            let mut out = Vec::new();
            for i in 0..6 {
                if let Some(v) = sb.push(i) {
                    out.push(v);
                }
            }
            out.extend(sb.drain());
            out
        };
        assert_eq!(run(42), run(42));
    }

    #[test]
    fn shuffle_different_seeds_differ() {
        let run = |seed: u64| -> Vec<i32> {
            let mut sb = ShuffleBuffer::new(3, seed);
            let mut out = Vec::new();
            for i in 0..20 {
                if let Some(v) = sb.push(i) {
                    out.push(v);
                }
            }
            out.extend(sb.drain());
            out
        };
        // With sufficiently many items, different seeds should produce
        // different orderings.
        assert_ne!(run(1), run(999));
    }

    #[test]
    fn shuffle_preserves_all_items() {
        let mut sb = ShuffleBuffer::new(4, 7);
        let mut out = Vec::new();
        for i in 0..10 {
            if let Some(v) = sb.push(i) {
                out.push(v);
            }
        }
        out.extend(sb.drain());
        out.sort_unstable();
        assert_eq!(out, (0..10).collect::<Vec<_>>());
    }

    #[test]
    #[should_panic(expected = "shuffle buffer capacity must be > 0")]
    fn shuffle_zero_capacity_panics() {
        let _sb: ShuffleBuffer<i32> = ShuffleBuffer::new(0, 0);
    }

    // ── PrefetchBuffer tests ────────────────────────────────────────────

    #[test]
    fn prefetch_fills_from_source() {
        let mut src = VecSource::new(vec![1, 2, 3, 4, 5]);
        let mut pb = PrefetchBuffer::new(3);
        pb.fill(&mut src);
        assert_eq!(pb.len(), 3);
    }

    #[test]
    fn prefetch_pop_order() {
        let mut src = VecSource::new(vec![10, 20, 30]);
        let mut pb = PrefetchBuffer::new(5);
        pb.fill(&mut src);
        assert_eq!(pb.pop(), Some(10));
        assert_eq!(pb.pop(), Some(20));
        assert_eq!(pb.pop(), Some(30));
        assert_eq!(pb.pop(), None);
    }

    #[test]
    fn prefetch_partial_fill() {
        let mut src = VecSource::new(vec![1]);
        let mut pb = PrefetchBuffer::new(10);
        pb.fill(&mut src);
        assert_eq!(pb.len(), 1);
    }

    #[test]
    fn prefetch_empty_source() {
        let mut src: VecSource<i32> = VecSource::new(vec![]);
        let mut pb = PrefetchBuffer::new(5);
        pb.fill(&mut src);
        assert!(pb.is_empty());
    }

    #[test]
    fn prefetch_count_accessor() {
        let pb: PrefetchBuffer<i32> = PrefetchBuffer::new(7);
        assert_eq!(pb.prefetch_count(), 7);
    }

    // ── DataPipelineMetrics tests ───────────────────────────────────────

    #[test]
    fn metrics_default() {
        let m = DataPipelineMetrics::new();
        assert_eq!(m.items_processed, 0);
        assert_eq!(m.batches_produced, 0);
        assert_eq!(m.items_dropped, 0);
    }

    #[test]
    fn metrics_items_per_sec_zero_time() {
        let m = DataPipelineMetrics::new();
        assert!(m.items_per_sec().abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_avg_transform_zero_items() {
        let m = DataPipelineMetrics::new();
        assert_eq!(m.avg_transform_time(), Duration::ZERO);
    }

    #[test]
    fn metrics_buffer_utilization_no_items() {
        let m = DataPipelineMetrics::new();
        assert!(m.buffer_utilization().abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_buffer_utilization_all_kept() {
        let mut m = DataPipelineMetrics::new();
        m.items_processed = 100;
        m.items_dropped = 0;
        assert!((m.buffer_utilization() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_buffer_utilization_half_dropped() {
        let mut m = DataPipelineMetrics::new();
        m.items_processed = 50;
        m.items_dropped = 50;
        assert!((m.buffer_utilization() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_display() {
        let m = DataPipelineMetrics::new();
        let s = format!("{m}");
        assert!(s.contains("items=0"));
        assert!(s.contains("batches=0"));
    }

    #[test]
    fn metrics_start_stop() {
        let mut m = DataPipelineMetrics::new();
        m.start();
        std::thread::sleep(Duration::from_millis(10));
        m.stop();
        assert!(m.total_wall_time >= Duration::from_millis(5));
    }

    // ── DataPipeline integration tests ──────────────────────────────────

    fn make_pipeline(n: usize, batch_size: usize) -> DataPipeline<usize> {
        let source = Box::new(RangeSource::new(0, n));
        let config = PipelineConfig::default().with_batch_size(batch_size);
        DataPipeline::new(source, config)
    }

    #[test]
    fn pipeline_basic_batching() {
        let mut p = make_pipeline(10, 3);
        let batches = p.collect_all();
        // 10 items / 3 = 3 full + 1 partial
        assert_eq!(batches.len(), 4);
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[3].len(), 1);
    }

    #[test]
    fn pipeline_exact_batches() {
        let mut p = make_pipeline(9, 3);
        let batches = p.collect_all();
        assert_eq!(batches.len(), 3);
        for b in &batches {
            assert_eq!(b.len(), 3);
        }
    }

    #[test]
    fn pipeline_single_item() {
        let mut p = make_pipeline(1, 5);
        let batches = p.collect_all();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0], vec![0]);
    }

    #[test]
    fn pipeline_empty_source() {
        let mut p = make_pipeline(0, 4);
        let batches = p.collect_all();
        assert!(batches.is_empty());
    }

    #[test]
    fn pipeline_with_map() {
        let source = Box::new(RangeSource::new(0, 6));
        let config = PipelineConfig::default().with_batch_size(3);
        let mut p = DataPipeline::new(source, config);
        p.add_transform(Box::new(MapTransform::new(|x: usize| x * 10, "mul10")));
        let batches = p.collect_all();
        assert_eq!(batches[0], vec![0, 10, 20]);
        assert_eq!(batches[1], vec![30, 40, 50]);
    }

    #[test]
    fn pipeline_with_filter() {
        let source = Box::new(RangeSource::new(0, 10));
        let config = PipelineConfig::default().with_batch_size(3);
        let mut p = DataPipeline::new(source, config);
        p.add_transform(Box::new(FilterTransform::new(|x: &usize| (*x).is_multiple_of(2), "even")));
        let batches = p.collect_all();
        let all: Vec<usize> = batches.into_iter().flatten().collect();
        assert_eq!(all, vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn pipeline_filter_then_map() {
        let source = Box::new(RangeSource::new(0, 10));
        let config = PipelineConfig::default().with_batch_size(5);
        let mut p = DataPipeline::new(source, config);
        p.add_transform(Box::new(FilterTransform::new(|x: &usize| *x < 5, "lt5")));
        p.add_transform(Box::new(MapTransform::new(|x: usize| x + 100, "add100")));
        let all: Vec<usize> = p.collect_all().into_iter().flatten().collect();
        assert_eq!(all, vec![100, 101, 102, 103, 104]);
    }

    #[test]
    fn pipeline_map_then_filter() {
        let source = Box::new(RangeSource::new(0, 10));
        let config = PipelineConfig::default().with_batch_size(10);
        let mut p = DataPipeline::new(source, config);
        p.add_transform(Box::new(MapTransform::new(|x: usize| x * 3, "mul3")));
        p.add_transform(Box::new(FilterTransform::new(|x: &usize| *x > 10, "gt10")));
        let all: Vec<usize> = p.collect_all().into_iter().flatten().collect();
        // 0*3=0, 1*3=3, 2*3=6, 3*3=9, 4*3=12✓, 5*3=15✓, ...9*3=27✓
        assert_eq!(all, vec![12, 15, 18, 21, 24, 27]);
    }

    #[test]
    fn pipeline_filter_drops_all() {
        let source = Box::new(RangeSource::new(0, 10));
        let config = PipelineConfig::default().with_batch_size(5);
        let mut p = DataPipeline::new(source, config);
        p.add_transform(Box::new(FilterTransform::new(|_: &usize| false, "none")));
        let batches = p.collect_all();
        assert!(batches.is_empty());
    }

    #[test]
    fn pipeline_with_shuffle() {
        let source = Box::new(RangeSource::new(0, 20));
        let config = PipelineConfig::default().with_batch_size(20);
        let mut p = DataPipeline::new(source, config);
        p.enable_shuffle(5, 42);
        let all: Vec<usize> = p.collect_all().into_iter().flatten().collect();
        // All items present.
        let mut sorted = all;
        sorted.sort_unstable();
        assert_eq!(sorted, (0..20).collect::<Vec<_>>());
    }

    #[test]
    fn pipeline_shuffle_deterministic() {
        let run = || {
            let source = Box::new(RangeSource::new(0, 30));
            let config = PipelineConfig::default().with_batch_size(30);
            let mut p = DataPipeline::new(source, config);
            p.enable_shuffle(5, 123);
            p.collect_all().into_iter().flatten().collect::<Vec<usize>>()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn pipeline_metrics_after_run() {
        let mut p = make_pipeline(10, 3);
        let _ = p.collect_all();
        let m = p.metrics();
        assert_eq!(m.items_processed, 10);
        assert_eq!(m.batches_produced, 4); // 3+3+3+1
        assert_eq!(m.items_dropped, 0);
    }

    #[test]
    fn pipeline_metrics_with_filter() {
        let source = Box::new(RangeSource::new(0, 10));
        let config = PipelineConfig::default().with_batch_size(10);
        let mut p = DataPipeline::new(source, config);
        p.add_transform(Box::new(FilterTransform::new(|x: &usize| (*x).is_multiple_of(2), "even")));
        let _ = p.collect_all();
        let m = p.metrics();
        assert_eq!(m.items_processed, 5);
        assert_eq!(m.items_dropped, 5);
    }

    #[test]
    fn pipeline_config_accessor() {
        let p = make_pipeline(5, 3);
        assert_eq!(p.config().batch_size, 3);
    }

    #[test]
    fn pipeline_reset_and_rerun() {
        let mut p = make_pipeline(6, 3);
        let b1 = p.collect_all();
        p.reset();
        let b2 = p.collect_all();
        assert_eq!(b1, b2);
    }

    #[test]
    fn pipeline_incremental_next_batch() {
        let mut p = make_pipeline(7, 3);
        let b1 = p.next_batch().unwrap();
        assert_eq!(b1.len(), 3);
        let b2 = p.next_batch().unwrap();
        assert_eq!(b2.len(), 3);
        let b3 = p.next_batch().unwrap();
        assert_eq!(b3.len(), 1); // remaining
        assert!(p.next_batch().is_none());
    }

    #[test]
    fn pipeline_large_batch_size() {
        let mut p = make_pipeline(5, 100);
        let batches = p.collect_all();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 5);
    }

    #[test]
    fn pipeline_batch_size_one() {
        let mut p = make_pipeline(4, 1);
        let batches = p.collect_all();
        assert_eq!(batches.len(), 4);
        for b in &batches {
            assert_eq!(b.len(), 1);
        }
    }

    #[test]
    fn pipeline_with_prefetch() {
        let source = Box::new(RangeSource::new(0, 10));
        let config = PipelineConfig::default().with_batch_size(5).with_prefetch_count(4);
        let mut p = DataPipeline::new(source, config);
        let batches = p.collect_all();
        let all: Vec<usize> = batches.into_iter().flatten().collect();
        assert_eq!(all, (0..10).collect::<Vec<usize>>());
    }

    #[test]
    fn pipeline_prefetch_zero() {
        let source = Box::new(RangeSource::new(0, 5));
        let config = PipelineConfig::default().with_batch_size(5).with_prefetch_count(0);
        let mut p = DataPipeline::new(source, config);
        let batches = p.collect_all();
        // prefetch_count=0 means fill does nothing, so no items flow.
        assert!(batches.is_empty());
    }

    #[test]
    fn pipeline_vec_source_strings() {
        let source = Box::new(VecSource::new(vec![
            "hello".to_string(),
            "world".to_string(),
            "foo".to_string(),
        ]));
        let config = PipelineConfig::default().with_batch_size(2);
        let mut p = DataPipeline::new(source, config);
        let batches = p.collect_all();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0], vec!["hello", "world"]);
        assert_eq!(batches[1], vec!["foo"]);
    }

    #[test]
    fn pipeline_multiple_maps() {
        let source = Box::new(RangeSource::new(1, 4));
        let config = PipelineConfig::default().with_batch_size(10);
        let mut p = DataPipeline::new(source, config);
        p.add_transform(Box::new(MapTransform::new(|x: usize| x + 1, "inc")));
        p.add_transform(Box::new(MapTransform::new(|x: usize| x * x, "sq")));
        let all: Vec<usize> = p.collect_all().into_iter().flatten().collect();
        // 1→2→4, 2→3→9, 3→4→16
        assert_eq!(all, vec![4, 9, 16]);
    }

    #[test]
    fn pipeline_filter_keeps_all() {
        let source = Box::new(RangeSource::new(0, 5));
        let config = PipelineConfig::default().with_batch_size(5);
        let mut p = DataPipeline::new(source, config);
        p.add_transform(Box::new(FilterTransform::new(|_: &usize| true, "all")));
        let all: Vec<usize> = p.collect_all().into_iter().flatten().collect();
        assert_eq!(all, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn pipeline_wall_time_recorded() {
        let mut p = make_pipeline(100, 10);
        let _ = p.collect_all();
        // Wall time should be non-negative (could be zero on very fast runs).
        assert!(p.metrics().total_wall_time >= Duration::ZERO);
    }
}
