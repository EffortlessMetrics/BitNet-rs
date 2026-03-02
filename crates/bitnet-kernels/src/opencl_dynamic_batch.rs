//! Dynamic batching with priority queues for serving multiple concurrent
//! inference requests on Intel Arc A770 via OpenCL.
//!
//! Provides a [`BatchScheduler`] that collects [`InferenceRequest`]s into a
//! priority-aware [`RequestQueue`], groups them by compatible sequence lengths
//! via [`BatchFormer`], and emits [`DynamicBatch`]es when a timeout or
//! capacity threshold is reached. All operations have CPU reference
//! implementations so the module compiles and tests without GPU hardware.

use std::collections::BinaryHeap;
use std::fmt;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Strategy for padding sequences within a batch to a uniform length.
#[derive(Debug, Clone, PartialEq)]
pub enum PaddingStrategy {
    /// No padding — only batch sequences of identical length.
    NoPad,
    /// Pad every sequence to the longest in the batch.
    PadToMax,
    /// Pad to the next power of two that fits.
    PadToBucket,
    /// Pad to the next multiple of `n`.
    PadToNearest(usize),
}

/// Priority level for an inference request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RequestPriority {
    Critical,
    High,
    Normal,
    Low,
    Background,
}

impl RequestPriority {
    /// Numeric weight (higher = more important).
    fn base_weight(self) -> u64 {
        match self {
            Self::Critical => 1000,
            Self::High => 100,
            Self::Normal => 10,
            Self::Low => 1,
            Self::Background => 0,
        }
    }
}

/// Configuration for the dynamic batching scheduler.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum requests in a single batch.
    pub max_batch_size: usize,
    /// Maximum milliseconds to wait before forming a partial batch.
    pub max_wait_ms: u64,
    /// How sequences are padded to uniform length.
    pub padding_strategy: PaddingStrategy,
    /// Number of distinct priority levels actually used.
    pub priority_levels: usize,
    /// Milliseconds of wait time that add one unit of effective priority
    /// (priority aging). Zero disables aging.
    pub aging_interval_ms: u64,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_wait_ms: 50,
            padding_strategy: PaddingStrategy::PadToMax,
            priority_levels: 5,
            aging_interval_ms: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------

/// A single inference request waiting to be batched.
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    /// Unique identifier.
    pub request_id: u64,
    /// Input token sequence.
    pub token_ids: Vec<u32>,
    /// Scheduling priority.
    pub priority: RequestPriority,
    /// When the request arrived.
    pub arrival_time: Instant,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
}

/// Internal wrapper that implements priority ordering.
#[derive(Debug, Clone)]
struct PrioritizedRequest {
    request: InferenceRequest,
    effective_priority: u64,
}

impl PartialEq for PrioritizedRequest {
    fn eq(&self, other: &Self) -> bool {
        self.effective_priority == other.effective_priority
            && self.request.request_id == other.request.request_id
    }
}

impl Eq for PrioritizedRequest {}

impl PartialOrd for PrioritizedRequest {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedRequest {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.effective_priority
            .cmp(&other.effective_priority)
            // Break ties: older request first (lower id = earlier).
            .then_with(|| other.request.request_id.cmp(&self.request.request_id))
    }
}

// ---------------------------------------------------------------------------
// Request queue
// ---------------------------------------------------------------------------

/// Priority queue of pending [`InferenceRequest`]s.
#[derive(Debug)]
pub struct RequestQueue {
    heap: BinaryHeap<PrioritizedRequest>,
    aging_interval_ms: u64,
}

impl RequestQueue {
    /// Create a new empty queue.
    pub fn new(aging_interval_ms: u64) -> Self {
        Self { heap: BinaryHeap::new(), aging_interval_ms }
    }

    /// Enqueue a request, computing its initial effective priority.
    pub fn push(&mut self, request: InferenceRequest) {
        let effective_priority = request.priority.base_weight();
        self.heap.push(PrioritizedRequest { request, effective_priority });
    }

    /// Dequeue the highest-effective-priority request, applying aging first.
    pub fn pop(&mut self) -> Option<InferenceRequest> {
        // Re-heapify with updated priorities.
        self.refresh_priorities();
        self.heap.pop().map(|pr| pr.request)
    }

    /// Peek at the highest-priority request without removing it.
    pub fn peek(&self) -> Option<&InferenceRequest> {
        self.heap.peek().map(|pr| &pr.request)
    }

    /// Number of pending requests.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Drain up to `n` requests in priority order with aging applied.
    pub fn drain_up_to(&mut self, n: usize) -> Vec<InferenceRequest> {
        self.refresh_priorities();
        let mut out = Vec::with_capacity(n.min(self.heap.len()));
        for _ in 0..n {
            match self.heap.pop() {
                Some(pr) => out.push(pr.request),
                None => break,
            }
        }
        out
    }

    /// Recalculate effective priorities with aging.
    fn refresh_priorities(&mut self) {
        if self.aging_interval_ms == 0 {
            return;
        }
        let now = Instant::now();
        let items: Vec<_> = self.heap.drain().collect();
        for mut pr in items {
            let waited_ms = now.duration_since(pr.request.arrival_time).as_millis() as u64;
            let aging_bonus = waited_ms / self.aging_interval_ms;
            pr.effective_priority = pr.request.priority.base_weight() + aging_bonus;
            self.heap.push(pr);
        }
    }
}

// ---------------------------------------------------------------------------
// Padding helpers
// ---------------------------------------------------------------------------

/// Compute the padded sequence length according to the strategy.
pub fn compute_padded_length(lengths: &[usize], strategy: &PaddingStrategy) -> usize {
    if lengths.is_empty() {
        return 0;
    }
    let max_len = lengths.iter().copied().max().unwrap_or(0);
    match strategy {
        PaddingStrategy::NoPad => max_len,
        PaddingStrategy::PadToMax => max_len,
        PaddingStrategy::PadToBucket => max_len.next_power_of_two(),
        PaddingStrategy::PadToNearest(n) => {
            let n = (*n).max(1);
            max_len.div_ceil(n) * n
        }
    }
}

/// Build a padding mask: `true` where the position is real, `false` where
/// it is padding.
pub fn build_padding_mask(
    token_lengths: &[usize],
    padded_length: usize,
) -> Vec<Vec<bool>> {
    token_lengths
        .iter()
        .map(|&len| {
            let mut mask = vec![true; len];
            mask.resize(padded_length, false);
            mask
        })
        .collect()
}

/// Calculate the ratio of padding tokens to total tokens (0.0 = no waste).
pub fn padding_waste_ratio(token_lengths: &[usize], padded_length: usize) -> f64 {
    if token_lengths.is_empty() || padded_length == 0 {
        return 0.0;
    }
    let real: usize = token_lengths.iter().sum();
    let total = token_lengths.len() * padded_length;
    if total == 0 {
        return 0.0;
    }
    1.0 - (real as f64 / total as f64)
}

// ---------------------------------------------------------------------------
// Dynamic batch
// ---------------------------------------------------------------------------

/// A fully formed batch ready for execution.
#[derive(Debug, Clone)]
pub struct DynamicBatch {
    /// Requests included in this batch.
    pub requests: Vec<InferenceRequest>,
    /// Padded token ids — `requests.len() × padded_seq_len`.
    pub padded_tokens: Vec<Vec<u32>>,
    /// Padding mask (true = real token).
    pub padding_mask: Vec<Vec<bool>>,
    /// Uniform sequence length after padding.
    pub padded_seq_len: usize,
    /// How many requests are in this batch.
    pub effective_batch_size: usize,
}

/// Map batch-level outputs back to individual requests.
pub fn map_outputs_to_requests(
    batch: &DynamicBatch,
    batch_outputs: &[Vec<u32>],
) -> Vec<(u64, Vec<u32>)> {
    batch
        .requests
        .iter()
        .zip(batch_outputs.iter())
        .map(|(req, out)| (req.request_id, out.clone()))
        .collect()
}

// ---------------------------------------------------------------------------
// Batch former
// ---------------------------------------------------------------------------

/// Groups requests into batches of compatible sequence lengths.
pub struct BatchFormer {
    config: BatchConfig,
}

impl BatchFormer {
    pub fn new(config: BatchConfig) -> Self {
        Self { config }
    }

    /// Form a single batch from the provided requests.
    pub fn form_batch(&self, requests: Vec<InferenceRequest>) -> Option<DynamicBatch> {
        if requests.is_empty() {
            return None;
        }

        let lengths: Vec<usize> = requests.iter().map(|r| r.token_ids.len()).collect();
        let padded_seq_len = compute_padded_length(&lengths, &self.config.padding_strategy);

        let padded_tokens: Vec<Vec<u32>> = requests
            .iter()
            .map(|r| {
                let mut toks = r.token_ids.clone();
                toks.resize(padded_seq_len, 0);
                toks
            })
            .collect();

        let padding_mask = build_padding_mask(&lengths, padded_seq_len);
        let effective_batch_size = requests.len();

        Some(DynamicBatch {
            requests,
            padded_tokens,
            padding_mask,
            padded_seq_len,
            effective_batch_size,
        })
    }

    /// Form a batch respecting the `NoPad` strategy: only group requests
    /// whose token lengths are identical.
    pub fn form_compatible_batches(
        &self,
        mut requests: Vec<InferenceRequest>,
    ) -> Vec<DynamicBatch> {
        if requests.is_empty() {
            return Vec::new();
        }

        // Sort by token length so equal-length runs are adjacent.
        requests.sort_by_key(|r| r.token_ids.len());

        let mut batches = Vec::new();
        let mut current_group: Vec<InferenceRequest> = Vec::new();
        let mut current_len: Option<usize> = None;

        for req in requests {
            let len = req.token_ids.len();
            let same_group = match (&self.config.padding_strategy, current_len) {
                (PaddingStrategy::NoPad, Some(cl)) => len == cl,
                (_, Some(_)) => true,
                (_, None) => true,
            };

            if (!same_group || current_group.len() >= self.config.max_batch_size)
                && !current_group.is_empty()
                && let Some(b) = self.form_batch(std::mem::take(&mut current_group))
            {
                batches.push(b);
            }

            current_len = Some(len);
            current_group.push(req);

            if current_group.len() >= self.config.max_batch_size {
                if let Some(b) = self.form_batch(std::mem::take(&mut current_group)) {
                    batches.push(b);
                }
                current_len = None;
            }
        }

        if !current_group.is_empty()
            && let Some(b) = self.form_batch(current_group)
        {
            batches.push(b);
        }

        batches
    }
}

// ---------------------------------------------------------------------------
// Batch statistics
// ---------------------------------------------------------------------------

/// Aggregate statistics for the dynamic batcher.
#[derive(Debug, Clone)]
pub struct BatchStats {
    /// Total batches formed.
    pub batches_formed: u64,
    /// Running sum of batch sizes.
    pub total_requests_batched: u64,
    /// Running sum of padding waste ratios.
    pub total_padding_waste: f64,
    /// Running sum of request wait times in milliseconds.
    pub total_wait_ms: f64,
    /// Total tokens produced across all batches.
    pub total_tokens_produced: u64,
    /// Timestamp of first batch (for throughput calculation).
    pub first_batch_time: Option<Instant>,
    /// Timestamp of most recent batch.
    pub last_batch_time: Option<Instant>,
}

impl Default for BatchStats {
    fn default() -> Self {
        Self {
            batches_formed: 0,
            total_requests_batched: 0,
            total_padding_waste: 0.0,
            total_wait_ms: 0.0,
            total_tokens_produced: 0,
            first_batch_time: None,
            last_batch_time: None,
        }
    }
}

impl BatchStats {
    /// Average batch size (requests per batch).
    pub fn avg_batch_size(&self) -> f64 {
        if self.batches_formed == 0 {
            return 0.0;
        }
        self.total_requests_batched as f64 / self.batches_formed as f64
    }

    /// Average padding waste ratio across batches.
    pub fn avg_padding_waste(&self) -> f64 {
        if self.batches_formed == 0 {
            return 0.0;
        }
        self.total_padding_waste / self.batches_formed as f64
    }

    /// Average wait time per request in milliseconds.
    pub fn avg_wait_ms(&self) -> f64 {
        if self.total_requests_batched == 0 {
            return 0.0;
        }
        self.total_wait_ms / self.total_requests_batched as f64
    }

    /// Throughput in tokens per second (wall-clock).
    pub fn throughput_tok_per_sec(&self) -> f64 {
        match (self.first_batch_time, self.last_batch_time) {
            (Some(first), Some(last)) => {
                let elapsed = last.duration_since(first).as_secs_f64();
                if elapsed < 1e-9 {
                    return 0.0;
                }
                self.total_tokens_produced as f64 / elapsed
            }
            _ => 0.0,
        }
    }

    /// Record one formed batch.
    pub fn record_batch(&mut self, batch: &DynamicBatch, now: Instant) {
        self.batches_formed += 1;
        self.total_requests_batched += batch.effective_batch_size as u64;

        let lengths: Vec<usize> =
            batch.requests.iter().map(|r| r.token_ids.len()).collect();
        self.total_padding_waste +=
            padding_waste_ratio(&lengths, batch.padded_seq_len);

        for req in &batch.requests {
            let waited = now.duration_since(req.arrival_time).as_millis() as f64;
            self.total_wait_ms += waited;
        }

        if self.first_batch_time.is_none() {
            self.first_batch_time = Some(now);
        }
        self.last_batch_time = Some(now);
    }
}

impl fmt::Display for BatchStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "batches={} avg_size={:.1} waste={:.2}% avg_wait={:.1}ms",
            self.batches_formed,
            self.avg_batch_size(),
            self.avg_padding_waste() * 100.0,
            self.avg_wait_ms(),
        )
    }
}

// ---------------------------------------------------------------------------
// Batch scheduler
// ---------------------------------------------------------------------------

/// Decides when to form a batch (timeout **or** full) and dispatches it.
pub struct BatchScheduler {
    queue: RequestQueue,
    former: BatchFormer,
    config: BatchConfig,
    stats: BatchStats,
    /// Time when the first un-batched request was received after the last
    /// batch was formed. Used for timeout tracking.
    first_pending_time: Option<Instant>,
}

impl BatchScheduler {
    pub fn new(config: BatchConfig) -> Self {
        let aging = config.aging_interval_ms;
        let former = BatchFormer::new(config.clone());
        Self {
            queue: RequestQueue::new(aging),
            former,
            config,
            stats: BatchStats::default(),
            first_pending_time: None,
        }
    }

    /// Submit a new request to the scheduler.
    pub fn submit(&mut self, request: InferenceRequest) {
        if self.first_pending_time.is_none() {
            self.first_pending_time = Some(Instant::now());
        }
        self.queue.push(request);
    }

    /// Check whether a batch should be formed right now.
    pub fn should_form_batch(&self) -> bool {
        if self.queue.is_empty() {
            return false;
        }
        // Capacity trigger.
        if self.queue.len() >= self.config.max_batch_size {
            return true;
        }
        // Timeout trigger.
        if let Some(first) = self.first_pending_time
            && Instant::now().duration_since(first)
                >= Duration::from_millis(self.config.max_wait_ms)
        {
            return true;
        }
        false
    }

    /// Force-form a batch with whatever is currently queued, returning
    /// `None` if the queue is empty.
    pub fn force_form_batch(&mut self) -> Option<DynamicBatch> {
        let requests = self.queue.drain_up_to(self.config.max_batch_size);
        if requests.is_empty() {
            return None;
        }
        let batch = self.former.form_batch(requests)?;
        let now = Instant::now();
        self.stats.record_batch(&batch, now);
        self.first_pending_time = None;
        Some(batch)
    }

    /// Try to form a batch if either trigger fires.
    pub fn try_form_batch(&mut self) -> Option<DynamicBatch> {
        if self.should_form_batch() {
            self.force_form_batch()
        } else {
            None
        }
    }

    /// Number of pending requests.
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }

    /// Accumulated statistics.
    pub fn stats(&self) -> &BatchStats {
        &self.stats
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from the dynamic batching subsystem.
#[derive(Debug, Clone, PartialEq)]
pub enum DynamicBatchError {
    /// Queue is empty, nothing to batch.
    EmptyQueue,
    /// Configuration is invalid.
    InvalidConfig(String),
}

impl fmt::Display for DynamicBatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyQueue => write!(f, "queue is empty"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for DynamicBatchError {}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    // Helpers ---------------------------------------------------------------

    fn make_request(id: u64, len: usize, priority: RequestPriority) -> InferenceRequest {
        InferenceRequest {
            request_id: id,
            token_ids: vec![1; len],
            priority,
            arrival_time: Instant::now(),
            max_tokens: 64,
        }
    }

    fn make_request_at(
        id: u64,
        len: usize,
        priority: RequestPriority,
        arrival: Instant,
    ) -> InferenceRequest {
        InferenceRequest {
            request_id: id,
            token_ids: vec![1; len],
            priority,
            arrival_time: arrival,
            max_tokens: 64,
        }
    }

    fn default_config() -> BatchConfig {
        BatchConfig { max_batch_size: 4, ..Default::default() }
    }

    // ---- PaddingStrategy --------------------------------------------------

    #[test]
    fn padding_no_pad_returns_max() {
        assert_eq!(compute_padded_length(&[3, 5, 7], &PaddingStrategy::NoPad), 7);
    }

    #[test]
    fn padding_pad_to_max() {
        assert_eq!(compute_padded_length(&[3, 5, 7], &PaddingStrategy::PadToMax), 7);
    }

    #[test]
    fn padding_pad_to_bucket_power_of_two() {
        assert_eq!(compute_padded_length(&[3, 5, 7], &PaddingStrategy::PadToBucket), 8);
    }

    #[test]
    fn padding_pad_to_bucket_already_power() {
        assert_eq!(compute_padded_length(&[4, 8], &PaddingStrategy::PadToBucket), 8);
    }

    #[test]
    fn padding_pad_to_nearest_multiple() {
        assert_eq!(compute_padded_length(&[3, 5, 7], &PaddingStrategy::PadToNearest(4)), 8);
    }

    #[test]
    fn padding_pad_to_nearest_exact() {
        assert_eq!(compute_padded_length(&[4, 8], &PaddingStrategy::PadToNearest(4)), 8);
    }

    #[test]
    fn padding_empty_lengths() {
        assert_eq!(compute_padded_length(&[], &PaddingStrategy::PadToMax), 0);
    }

    #[test]
    fn padding_single_length() {
        assert_eq!(compute_padded_length(&[5], &PaddingStrategy::PadToBucket), 8);
    }

    #[test]
    fn padding_nearest_with_zero_rounds_to_1() {
        // PadToNearest(0) treated as PadToNearest(1).
        assert_eq!(compute_padded_length(&[3], &PaddingStrategy::PadToNearest(0)), 3);
    }

    // ---- Padding mask -----------------------------------------------------

    #[test]
    fn mask_correct_shape() {
        let mask = build_padding_mask(&[3, 5], 8);
        assert_eq!(mask.len(), 2);
        assert_eq!(mask[0].len(), 8);
        assert_eq!(mask[1].len(), 8);
    }

    #[test]
    fn mask_true_for_real_tokens() {
        let mask = build_padding_mask(&[3], 5);
        assert_eq!(mask[0], vec![true, true, true, false, false]);
    }

    #[test]
    fn mask_no_padding_when_equal() {
        let mask = build_padding_mask(&[4], 4);
        assert!(mask[0].iter().all(|&v| v));
    }

    // ---- Padding waste ratio ----------------------------------------------

    #[test]
    fn waste_zero_when_no_padding() {
        assert!((padding_waste_ratio(&[8, 8], 8) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn waste_correct_calculation() {
        // 2 sequences: lengths 4 and 8, padded to 8.
        // Real = 12, total = 16, waste = 4/16 = 0.25.
        let ratio = padding_waste_ratio(&[4, 8], 8);
        assert!((ratio - 0.25).abs() < 1e-9);
    }

    #[test]
    fn waste_empty_is_zero() {
        assert!((padding_waste_ratio(&[], 8) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn waste_zero_padded_len() {
        assert!((padding_waste_ratio(&[3], 0) - 0.0).abs() < 1e-9);
    }

    // ---- RequestPriority --------------------------------------------------

    #[test]
    fn priority_ordering_weights() {
        assert!(RequestPriority::Critical.base_weight() > RequestPriority::High.base_weight());
        assert!(RequestPriority::High.base_weight() > RequestPriority::Normal.base_weight());
        assert!(RequestPriority::Normal.base_weight() > RequestPriority::Low.base_weight());
        assert!(RequestPriority::Low.base_weight() > RequestPriority::Background.base_weight());
    }

    // ---- RequestQueue — basic ---------------------------------------------

    #[test]
    fn queue_push_and_pop() {
        let mut q = RequestQueue::new(0);
        q.push(make_request(1, 10, RequestPriority::Normal));
        assert_eq!(q.len(), 1);
        let req = q.pop().unwrap();
        assert_eq!(req.request_id, 1);
        assert!(q.is_empty());
    }

    #[test]
    fn queue_pop_empty_returns_none() {
        let mut q = RequestQueue::new(0);
        assert!(q.pop().is_none());
    }

    #[test]
    fn queue_priority_order_high_before_low() {
        let mut q = RequestQueue::new(0);
        q.push(make_request(1, 10, RequestPriority::Low));
        q.push(make_request(2, 10, RequestPriority::High));
        q.push(make_request(3, 10, RequestPriority::Normal));

        assert_eq!(q.pop().unwrap().request_id, 2); // High
        assert_eq!(q.pop().unwrap().request_id, 3); // Normal
        assert_eq!(q.pop().unwrap().request_id, 1); // Low
    }

    #[test]
    fn queue_critical_before_all() {
        let mut q = RequestQueue::new(0);
        q.push(make_request(1, 10, RequestPriority::Background));
        q.push(make_request(2, 10, RequestPriority::Critical));
        assert_eq!(q.pop().unwrap().request_id, 2);
    }

    #[test]
    fn queue_fifo_within_same_priority() {
        let mut q = RequestQueue::new(0);
        q.push(make_request(1, 10, RequestPriority::Normal));
        q.push(make_request(2, 10, RequestPriority::Normal));
        q.push(make_request(3, 10, RequestPriority::Normal));
        // Within the same priority, lower id (older) comes first.
        assert_eq!(q.pop().unwrap().request_id, 1);
        assert_eq!(q.pop().unwrap().request_id, 2);
        assert_eq!(q.pop().unwrap().request_id, 3);
    }

    #[test]
    fn queue_drain_up_to_limits() {
        let mut q = RequestQueue::new(0);
        for i in 0..10 {
            q.push(make_request(i, 10, RequestPriority::Normal));
        }
        let drained = q.drain_up_to(3);
        assert_eq!(drained.len(), 3);
        assert_eq!(q.len(), 7);
    }

    #[test]
    fn queue_drain_up_to_more_than_available() {
        let mut q = RequestQueue::new(0);
        q.push(make_request(1, 10, RequestPriority::Normal));
        let drained = q.drain_up_to(100);
        assert_eq!(drained.len(), 1);
        assert!(q.is_empty());
    }

    #[test]
    fn queue_peek_does_not_remove() {
        let mut q = RequestQueue::new(0);
        q.push(make_request(1, 10, RequestPriority::Normal));
        assert!(q.peek().is_some());
        assert_eq!(q.len(), 1);
    }

    // ---- Priority aging ---------------------------------------------------

    #[test]
    fn aging_boosts_old_low_priority() {
        let past = Instant::now() - Duration::from_millis(500);
        let mut q = RequestQueue::new(100); // +1 every 100 ms

        // Old Background request: base 0 + 500/100 = 5
        q.push(make_request_at(1, 10, RequestPriority::Background, past));
        // Fresh Normal request: base 10 + ~0 = 10
        q.push(make_request(2, 10, RequestPriority::Normal));

        // Normal still wins (10 > 5) but aging narrowed the gap.
        let first = q.pop().unwrap();
        assert_eq!(first.request_id, 2);
    }

    #[test]
    fn aging_old_low_overtakes_fresh_normal() {
        let past = Instant::now() - Duration::from_secs(2);
        let mut q = RequestQueue::new(100);

        // Old Low: base 1 + 2000/100 = 21
        q.push(make_request_at(1, 10, RequestPriority::Low, past));
        // Fresh Normal: base 10 + ~0 = 10
        q.push(make_request(2, 10, RequestPriority::Normal));

        assert_eq!(q.pop().unwrap().request_id, 1); // aged Low wins
    }

    #[test]
    fn aging_disabled_when_zero() {
        let past = Instant::now() - Duration::from_secs(100);
        let mut q = RequestQueue::new(0);

        q.push(make_request_at(1, 10, RequestPriority::Low, past));
        q.push(make_request(2, 10, RequestPriority::Normal));

        assert_eq!(q.pop().unwrap().request_id, 2); // Normal still first
    }

    // ---- BatchFormer — single batch --------------------------------------

    #[test]
    fn former_single_request_batch_of_one() {
        let cfg = default_config();
        let former = BatchFormer::new(cfg);
        let reqs = vec![make_request(1, 10, RequestPriority::Normal)];
        let batch = former.form_batch(reqs).unwrap();
        assert_eq!(batch.effective_batch_size, 1);
        assert_eq!(batch.requests[0].request_id, 1);
    }

    #[test]
    fn former_empty_returns_none() {
        let former = BatchFormer::new(default_config());
        assert!(former.form_batch(Vec::new()).is_none());
    }

    #[test]
    fn former_padded_tokens_correct_length() {
        let cfg = BatchConfig {
            padding_strategy: PaddingStrategy::PadToMax,
            ..default_config()
        };
        let former = BatchFormer::new(cfg);
        let reqs = vec![
            make_request(1, 3, RequestPriority::Normal),
            make_request(2, 7, RequestPriority::Normal),
        ];
        let batch = former.form_batch(reqs).unwrap();
        assert_eq!(batch.padded_seq_len, 7);
        assert!(batch.padded_tokens.iter().all(|t| t.len() == 7));
    }

    #[test]
    fn former_pad_to_bucket() {
        let cfg = BatchConfig {
            padding_strategy: PaddingStrategy::PadToBucket,
            ..default_config()
        };
        let former = BatchFormer::new(cfg);
        let reqs = vec![
            make_request(1, 5, RequestPriority::Normal),
            make_request(2, 7, RequestPriority::Normal),
        ];
        let batch = former.form_batch(reqs).unwrap();
        assert_eq!(batch.padded_seq_len, 8);
    }

    #[test]
    fn former_padding_mask_matches() {
        let cfg = BatchConfig {
            padding_strategy: PaddingStrategy::PadToMax,
            ..default_config()
        };
        let former = BatchFormer::new(cfg);
        let reqs = vec![
            make_request(1, 3, RequestPriority::Normal),
            make_request(2, 5, RequestPriority::Normal),
        ];
        let batch = former.form_batch(reqs).unwrap();
        // First request: 3 real, 2 padding.
        assert!(batch.padding_mask[0][2]);
        assert!(!batch.padding_mask[0][3]);
        // Second request: all real.
        assert!(batch.padding_mask[1].iter().all(|&v| v));
    }

    // ---- BatchFormer — compatible batches (NoPad) -------------------------

    #[test]
    fn compatible_batches_no_pad_groups_by_length() {
        let cfg = BatchConfig {
            max_batch_size: 10,
            padding_strategy: PaddingStrategy::NoPad,
            ..Default::default()
        };
        let former = BatchFormer::new(cfg);
        let reqs = vec![
            make_request(1, 5, RequestPriority::Normal),
            make_request(2, 5, RequestPriority::Normal),
            make_request(3, 10, RequestPriority::Normal),
            make_request(4, 10, RequestPriority::Normal),
        ];
        let batches = former.form_compatible_batches(reqs);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].effective_batch_size, 2);
        assert_eq!(batches[1].effective_batch_size, 2);
    }

    #[test]
    fn compatible_batches_all_same_length() {
        let cfg = BatchConfig {
            max_batch_size: 10,
            padding_strategy: PaddingStrategy::NoPad,
            ..Default::default()
        };
        let former = BatchFormer::new(cfg);
        let reqs: Vec<_> = (0..5).map(|i| make_request(i, 8, RequestPriority::Normal)).collect();
        let batches = former.form_compatible_batches(reqs);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].effective_batch_size, 5);
    }

    #[test]
    fn compatible_batches_all_different_lengths() {
        let cfg = BatchConfig {
            max_batch_size: 10,
            padding_strategy: PaddingStrategy::NoPad,
            ..Default::default()
        };
        let former = BatchFormer::new(cfg);
        let reqs = vec![
            make_request(1, 3, RequestPriority::Normal),
            make_request(2, 5, RequestPriority::Normal),
            make_request(3, 7, RequestPriority::Normal),
        ];
        let batches = former.form_compatible_batches(reqs);
        // Each gets its own batch.
        assert_eq!(batches.len(), 3);
    }

    #[test]
    fn compatible_batches_pad_to_max_groups_all() {
        let cfg = BatchConfig {
            max_batch_size: 10,
            padding_strategy: PaddingStrategy::PadToMax,
            ..Default::default()
        };
        let former = BatchFormer::new(cfg);
        let reqs = vec![
            make_request(1, 3, RequestPriority::Normal),
            make_request(2, 5, RequestPriority::Normal),
            make_request(3, 7, RequestPriority::Normal),
        ];
        let batches = former.form_compatible_batches(reqs);
        assert_eq!(batches.len(), 1);
    }

    #[test]
    fn compatible_batches_respects_max_batch_size() {
        let cfg = BatchConfig {
            max_batch_size: 2,
            padding_strategy: PaddingStrategy::PadToMax,
            ..Default::default()
        };
        let former = BatchFormer::new(cfg);
        let reqs: Vec<_> = (0..5).map(|i| make_request(i, 4, RequestPriority::Normal)).collect();
        let batches = former.form_compatible_batches(reqs);
        // 5 requests / max 2 = 3 batches (2+2+1).
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].effective_batch_size, 2);
        assert_eq!(batches[1].effective_batch_size, 2);
        assert_eq!(batches[2].effective_batch_size, 1);
    }

    #[test]
    fn compatible_batches_empty_input() {
        let former = BatchFormer::new(default_config());
        assert!(former.form_compatible_batches(Vec::new()).is_empty());
    }

    // ---- DynamicBatch — output mapping ------------------------------------

    #[test]
    fn output_mapping_correct_ids() {
        let former = BatchFormer::new(default_config());
        let reqs = vec![
            make_request(10, 4, RequestPriority::Normal),
            make_request(20, 4, RequestPriority::Normal),
        ];
        let batch = former.form_batch(reqs).unwrap();
        let outputs = vec![vec![100, 101], vec![200, 201]];
        let mapped = map_outputs_to_requests(&batch, &outputs);
        assert_eq!(mapped[0], (10, vec![100, 101]));
        assert_eq!(mapped[1], (20, vec![200, 201]));
    }

    #[test]
    fn output_mapping_empty_outputs() {
        let former = BatchFormer::new(default_config());
        let reqs = vec![make_request(1, 4, RequestPriority::Normal)];
        let batch = former.form_batch(reqs).unwrap();
        let outputs = vec![vec![]];
        let mapped = map_outputs_to_requests(&batch, &outputs);
        assert_eq!(mapped[0], (1, vec![]));
    }

    // ---- BatchScheduler ---------------------------------------------------

    #[test]
    fn scheduler_empty_no_batch() {
        let mut sched = BatchScheduler::new(default_config());
        assert!(!sched.should_form_batch());
        assert!(sched.try_form_batch().is_none());
    }

    #[test]
    fn scheduler_capacity_trigger() {
        let cfg = BatchConfig { max_batch_size: 2, ..Default::default() };
        let mut sched = BatchScheduler::new(cfg);
        sched.submit(make_request(1, 5, RequestPriority::Normal));
        assert!(!sched.should_form_batch()); // 1 < 2
        sched.submit(make_request(2, 5, RequestPriority::Normal));
        assert!(sched.should_form_batch()); // 2 >= 2
    }

    #[test]
    fn scheduler_try_form_batch_returns_batch() {
        let cfg = BatchConfig { max_batch_size: 2, ..Default::default() };
        let mut sched = BatchScheduler::new(cfg);
        sched.submit(make_request(1, 5, RequestPriority::Normal));
        sched.submit(make_request(2, 5, RequestPriority::Normal));
        let batch = sched.try_form_batch().unwrap();
        assert_eq!(batch.effective_batch_size, 2);
        assert_eq!(sched.pending_count(), 0);
    }

    #[test]
    fn scheduler_force_form_partial_batch() {
        let mut sched = BatchScheduler::new(default_config());
        sched.submit(make_request(1, 5, RequestPriority::Normal));
        // Force even though below capacity.
        let batch = sched.force_form_batch().unwrap();
        assert_eq!(batch.effective_batch_size, 1);
    }

    #[test]
    fn scheduler_force_form_empty_returns_none() {
        let mut sched = BatchScheduler::new(default_config());
        assert!(sched.force_form_batch().is_none());
    }

    #[test]
    fn scheduler_timeout_trigger() {
        let cfg = BatchConfig { max_wait_ms: 0, max_batch_size: 100, ..Default::default() };
        let mut sched = BatchScheduler::new(cfg);
        sched.submit(make_request(1, 5, RequestPriority::Normal));
        // max_wait_ms = 0 means "form immediately after any request".
        assert!(sched.should_form_batch());
    }

    #[test]
    fn scheduler_stats_updated() {
        let cfg = BatchConfig { max_batch_size: 2, ..Default::default() };
        let mut sched = BatchScheduler::new(cfg);
        sched.submit(make_request(1, 5, RequestPriority::Normal));
        sched.submit(make_request(2, 5, RequestPriority::Normal));
        sched.try_form_batch();
        assert_eq!(sched.stats().batches_formed, 1);
        assert_eq!(sched.stats().total_requests_batched, 2);
    }

    #[test]
    fn scheduler_pending_count() {
        let mut sched = BatchScheduler::new(default_config());
        sched.submit(make_request(1, 5, RequestPriority::Normal));
        sched.submit(make_request(2, 5, RequestPriority::Normal));
        assert_eq!(sched.pending_count(), 2);
    }

    #[test]
    fn scheduler_priority_respected_in_batch() {
        let cfg = BatchConfig { max_batch_size: 3, ..Default::default() };
        let mut sched = BatchScheduler::new(cfg);
        sched.submit(make_request(1, 5, RequestPriority::Low));
        sched.submit(make_request(2, 5, RequestPriority::Critical));
        sched.submit(make_request(3, 5, RequestPriority::Normal));
        let batch = sched.force_form_batch().unwrap();
        // Requests are drained in priority order.
        assert_eq!(batch.requests[0].request_id, 2); // Critical
        assert_eq!(batch.requests[1].request_id, 3); // Normal
        assert_eq!(batch.requests[2].request_id, 1); // Low
    }

    // ---- BatchStats -------------------------------------------------------

    #[test]
    fn stats_default_zeros() {
        let s = BatchStats::default();
        assert!((s.avg_batch_size() - 0.0).abs() < 1e-9);
        assert!((s.avg_padding_waste() - 0.0).abs() < 1e-9);
        assert!((s.avg_wait_ms() - 0.0).abs() < 1e-9);
        assert!((s.throughput_tok_per_sec() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn stats_avg_batch_size() {
        let mut s = BatchStats::default();
        s.batches_formed = 3;
        s.total_requests_batched = 12;
        assert!((s.avg_batch_size() - 4.0).abs() < 1e-9);
    }

    #[test]
    fn stats_avg_padding_waste() {
        let mut s = BatchStats::default();
        s.batches_formed = 2;
        s.total_padding_waste = 0.5;
        assert!((s.avg_padding_waste() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn stats_display() {
        let s = BatchStats::default();
        let text = format!("{s}");
        assert!(text.contains("batches=0"));
    }

    #[test]
    fn stats_record_batch_increments() {
        let mut s = BatchStats::default();
        let former = BatchFormer::new(default_config());
        let reqs = vec![
            make_request(1, 3, RequestPriority::Normal),
            make_request(2, 5, RequestPriority::Normal),
        ];
        let batch = former.form_batch(reqs).unwrap();
        let now = Instant::now();
        s.record_batch(&batch, now);
        assert_eq!(s.batches_formed, 1);
        assert_eq!(s.total_requests_batched, 2);
    }

    // ---- Edge cases -------------------------------------------------------

    #[test]
    fn single_token_request() {
        let former = BatchFormer::new(default_config());
        let reqs = vec![make_request(1, 1, RequestPriority::Normal)];
        let batch = former.form_batch(reqs).unwrap();
        assert_eq!(batch.padded_seq_len, 1);
    }

    #[test]
    fn large_batch_all_same_length() {
        let cfg = BatchConfig { max_batch_size: 128, ..Default::default() };
        let former = BatchFormer::new(cfg);
        let reqs: Vec<_> =
            (0..64).map(|i| make_request(i, 32, RequestPriority::Normal)).collect();
        let batch = former.form_batch(reqs).unwrap();
        assert_eq!(batch.effective_batch_size, 64);
        assert_eq!(batch.padded_seq_len, 32);
    }

    #[test]
    fn batch_preserves_request_ids() {
        let former = BatchFormer::new(default_config());
        let reqs = vec![
            make_request(42, 4, RequestPriority::Normal),
            make_request(99, 4, RequestPriority::Normal),
        ];
        let batch = former.form_batch(reqs).unwrap();
        let ids: Vec<u64> = batch.requests.iter().map(|r| r.request_id).collect();
        assert!(ids.contains(&42));
        assert!(ids.contains(&99));
    }

    // ---- Property-like tests -----------------------------------------------

    #[test]
    fn all_requests_eventually_served() {
        let cfg = BatchConfig { max_batch_size: 3, ..Default::default() };
        let mut sched = BatchScheduler::new(cfg);
        let total = 10u64;
        for i in 0..total {
            sched.submit(make_request(i, 5, RequestPriority::Normal));
        }
        let mut served = Vec::new();
        while sched.pending_count() > 0 {
            if let Some(batch) = sched.force_form_batch() {
                for req in &batch.requests {
                    served.push(req.request_id);
                }
            }
        }
        served.sort();
        let expected: Vec<u64> = (0..total).collect();
        assert_eq!(served, expected);
    }

    #[test]
    fn no_request_lost_with_mixed_priorities() {
        let mut sched = BatchScheduler::new(default_config());
        let priorities = [
            RequestPriority::Critical,
            RequestPriority::High,
            RequestPriority::Normal,
            RequestPriority::Low,
            RequestPriority::Background,
        ];
        for (i, &prio) in priorities.iter().enumerate() {
            sched.submit(make_request(i as u64, 4, prio));
        }
        let mut served = Vec::new();
        while sched.pending_count() > 0 {
            if let Some(batch) = sched.force_form_batch() {
                for req in &batch.requests {
                    served.push(req.request_id);
                }
            }
        }
        served.sort();
        assert_eq!(served, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn no_request_lost_drain_multiple_batches() {
        let cfg = BatchConfig { max_batch_size: 2, ..Default::default() };
        let mut sched = BatchScheduler::new(cfg);
        for i in 0..7u64 {
            sched.submit(make_request(i, 4, RequestPriority::Normal));
        }
        let mut served = Vec::new();
        while sched.pending_count() > 0 {
            if let Some(batch) = sched.force_form_batch() {
                for req in &batch.requests {
                    served.push(req.request_id);
                }
            }
        }
        served.sort();
        assert_eq!(served, (0..7).collect::<Vec<_>>());
    }

    #[test]
    fn scheduler_multiple_cycles() {
        let cfg = BatchConfig { max_batch_size: 2, ..Default::default() };
        let mut sched = BatchScheduler::new(cfg);

        // Cycle 1.
        sched.submit(make_request(1, 5, RequestPriority::Normal));
        sched.submit(make_request(2, 5, RequestPriority::Normal));
        let b1 = sched.try_form_batch().unwrap();
        assert_eq!(b1.effective_batch_size, 2);

        // Cycle 2.
        sched.submit(make_request(3, 5, RequestPriority::Normal));
        sched.submit(make_request(4, 5, RequestPriority::Normal));
        let b2 = sched.try_form_batch().unwrap();
        assert_eq!(b2.effective_batch_size, 2);

        assert_eq!(sched.stats().batches_formed, 2);
    }

    #[test]
    fn padding_waste_ratio_all_same() {
        let ratio = padding_waste_ratio(&[8, 8, 8], 8);
        assert!((ratio - 0.0).abs() < 1e-9);
    }

    #[test]
    fn padding_waste_ratio_extreme() {
        // 1 token padded to 100: waste = (100-1)/100 = 0.99
        let ratio = padding_waste_ratio(&[1], 100);
        assert!((ratio - 0.99).abs() < 1e-9);
    }

    #[test]
    fn batch_config_default_sane() {
        let cfg = BatchConfig::default();
        assert!(cfg.max_batch_size > 0);
        assert!(cfg.max_wait_ms > 0);
        assert!(cfg.priority_levels > 0);
    }

    #[test]
    fn request_priority_all_variants() {
        let variants = [
            RequestPriority::Critical,
            RequestPriority::High,
            RequestPriority::Normal,
            RequestPriority::Low,
            RequestPriority::Background,
        ];
        // Each variant has a distinct weight.
        let weights: Vec<u64> = variants.iter().map(|p| p.base_weight()).collect();
        let mut sorted = weights.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), weights.len());
    }

    #[test]
    fn dynamic_batch_error_display() {
        let e = DynamicBatchError::EmptyQueue;
        assert_eq!(format!("{e}"), "queue is empty");

        let e2 = DynamicBatchError::InvalidConfig("bad".into());
        assert!(format!("{e2}").contains("bad"));
    }

    #[test]
    fn scheduler_stats_avg_wait() {
        let cfg = BatchConfig { max_batch_size: 2, ..Default::default() };
        let mut sched = BatchScheduler::new(cfg);
        sched.submit(make_request(1, 5, RequestPriority::Normal));
        sched.submit(make_request(2, 5, RequestPriority::Normal));
        sched.force_form_batch();
        // avg_wait should be non-negative.
        assert!(sched.stats().avg_wait_ms() >= 0.0);
    }

    #[test]
    fn pad_to_nearest_large_multiple() {
        assert_eq!(compute_padded_length(&[100], &PaddingStrategy::PadToNearest(64)), 128);
    }

    #[test]
    fn pad_to_nearest_already_multiple() {
        assert_eq!(compute_padded_length(&[64], &PaddingStrategy::PadToNearest(64)), 64);
    }
}
