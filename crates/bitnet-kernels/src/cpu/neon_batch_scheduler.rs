//! ARM NEON optimized batch scheduling and management for Apple Silicon inference.
//!
//! Provides batch token scheduling with priority queuing, dynamic batch size
//! adjustment, batch padding/unpadding, throughput and latency statistics, and
//! batch completion tracking. NEON intrinsics accelerate hot-path aggregation
//! while all public APIs remain safe Rust.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
use std::collections::BinaryHeap;
use std::time::Instant;

/// NEON lane count for `float32x4_t`.
const LANES: usize = 4;

/// Default maximum batch size.
const DEFAULT_MAX_BATCH: usize = 64;

/// Minimum allowed batch size.
const MIN_BATCH_SIZE: usize = 1;

// ── Priority ────────────────────────────────────────────────────────────

/// Priority level for batch requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    /// Low priority — best-effort scheduling.
    Low = 0,
    /// Normal priority — default level.
    Normal = 1,
    /// High priority — processed before lower priorities.
    High = 2,
    /// Critical priority — always scheduled first.
    Critical = 3,
}

impl Default for Priority {
    fn default() -> Self {
        Priority::Normal
    }
}

// ── BatchRequest ────────────────────────────────────────────────────────

/// A single request submitted for batched inference.
#[derive(Debug, Clone)]
pub struct BatchRequest {
    /// Unique request identifier.
    pub id: u64,
    /// Token IDs to process.
    pub token_ids: Vec<u32>,
    /// Scheduling priority.
    pub priority: Priority,
    /// Submission timestamp.
    pub submitted_at: Instant,
}

impl BatchRequest {
    /// Create a new request with the given id, tokens, and priority.
    pub fn new(id: u64, token_ids: Vec<u32>, priority: Priority) -> Self {
        Self { id, token_ids, priority, submitted_at: Instant::now() }
    }

    /// Sequence length (number of tokens).
    pub fn seq_len(&self) -> usize {
        self.token_ids.len()
    }
}

impl PartialEq for BatchRequest {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for BatchRequest {}

impl PartialOrd for BatchRequest {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BatchRequest {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority.cmp(&other.priority).then_with(|| other.submitted_at.cmp(&self.submitted_at))
    }
}

// ── CompletionStatus ────────────────────────────────────────────────────

/// Completion status for a processed request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionStatus {
    /// Successfully processed.
    Success,
    /// Failed during processing.
    Failed,
    /// Timed out before completion.
    TimedOut,
}

/// A completed request record.
#[derive(Debug, Clone)]
pub struct CompletedRequest {
    /// Request identifier.
    pub id: u64,
    /// Outcome.
    pub status: CompletionStatus,
    /// Number of tokens that were processed.
    pub tokens_processed: usize,
    /// Wall-clock latency in microseconds.
    pub latency_us: u64,
}

// ── BatchStatistics ─────────────────────────────────────────────────────

/// Running statistics for batch scheduling throughput and latency.
#[derive(Debug, Clone)]
pub struct BatchStatistics {
    /// Total tokens processed across all completed batches.
    pub total_tokens: u64,
    /// Total batches that have been dispatched.
    pub total_batches: u64,
    /// Total wall-clock time spent processing in microseconds.
    pub total_time_us: u64,
    /// Per-request latencies in microseconds (ring buffer, last N entries).
    latencies: Vec<f32>,
    /// Maximum entries retained in `latencies`.
    max_latency_entries: usize,
}

impl BatchStatistics {
    /// Create empty statistics with the given ring-buffer capacity.
    pub fn new(max_latency_entries: usize) -> Self {
        Self {
            total_tokens: 0,
            total_batches: 0,
            total_time_us: 0,
            latencies: Vec::with_capacity(max_latency_entries),
            max_latency_entries,
        }
    }

    /// Record one completed batch.
    pub fn record_batch(&mut self, tokens: usize, elapsed_us: u64) {
        self.total_tokens += tokens as u64;
        self.total_batches += 1;
        self.total_time_us += elapsed_us;
        if self.latencies.len() >= self.max_latency_entries {
            self.latencies.remove(0);
        }
        self.latencies.push(elapsed_us as f32);
    }

    /// Average throughput in tokens per second (0.0 if no time recorded).
    pub fn throughput_tps(&self) -> f64 {
        if self.total_time_us == 0 {
            return 0.0;
        }
        (self.total_tokens as f64) / (self.total_time_us as f64 / 1_000_000.0)
    }

    /// Mean latency in microseconds using NEON-accelerated summation when
    /// available, falling back to scalar otherwise.
    pub fn mean_latency_us(&self) -> f32 {
        let n = self.latencies.len();
        if n == 0 {
            return 0.0;
        }
        #[cfg(target_arch = "aarch64")]
        {
            let sum = neon_sum_f32(&self.latencies);
            sum / n as f32
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let sum: f32 = self.latencies.iter().sum();
            sum / n as f32
        }
    }

    /// p50 latency (median) from the retained window.
    pub fn p50_latency_us(&self) -> f32 {
        percentile_latency(&self.latencies, 50)
    }

    /// p99 latency from the retained window.
    pub fn p99_latency_us(&self) -> f32 {
        percentile_latency(&self.latencies, 99)
    }

    /// Number of latency samples currently retained.
    pub fn latency_sample_count(&self) -> usize {
        self.latencies.len()
    }

    /// Reset all counters to zero.
    pub fn reset(&mut self) {
        self.total_tokens = 0;
        self.total_batches = 0;
        self.total_time_us = 0;
        self.latencies.clear();
    }
}

/// Compute percentile from an unsorted slice (copies and sorts internally).
fn percentile_latency(data: &[f32], pct: u32) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f32> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((pct as f64 / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ── NEON helpers ────────────────────────────────────────────────────────

/// NEON-accelerated horizontal sum of an f32 slice.
#[cfg(target_arch = "aarch64")]
pub fn neon_sum_f32(data: &[f32]) -> f32 {
    let len = data.len();
    let chunks = len / LANES;
    let remainder = len % LANES;

    let mut acc = unsafe { vdupq_n_f32(0.0) };
    for i in 0..chunks {
        let base = i * LANES;
        let v = unsafe { vld1q_f32(data.as_ptr().add(base)) };
        acc = unsafe { vaddq_f32(acc, v) };
    }

    // Horizontal reduction.
    let mut sum: f32 = unsafe {
        let pair = vpadd_f32(vget_low_f32(acc), vget_high_f32(acc));
        let pair2 = vpadd_f32(pair, pair);
        vget_lane_f32(pair2, 0)
    };

    let tail_start = chunks * LANES;
    for i in 0..remainder {
        sum += data[tail_start + i];
    }
    sum
}

/// Scalar fallback for sum (used on non-aarch64 or in tests).
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_sum_f32(data: &[f32]) -> f32 {
    data.iter().sum()
}

/// NEON-accelerated element-wise multiply of two f32 slices into `out`.
/// `out.len()` must be >= `len`; reads `len` elements from `a` and `b`.
#[cfg(target_arch = "aarch64")]
pub fn neon_mul_f32(a: &[f32], b: &[f32], out: &mut [f32], len: usize) {
    assert!(a.len() >= len && b.len() >= len && out.len() >= len);
    let chunks = len / LANES;
    let remainder = len % LANES;

    for i in 0..chunks {
        let base = i * LANES;
        unsafe {
            let va = vld1q_f32(a.as_ptr().add(base));
            let vb = vld1q_f32(b.as_ptr().add(base));
            let vr = vmulq_f32(va, vb);
            vst1q_f32(out.as_mut_ptr().add(base), vr);
        }
    }

    let tail = chunks * LANES;
    for i in 0..remainder {
        out[tail + i] = a[tail + i] * b[tail + i];
    }
}

/// Scalar fallback for element-wise multiply.
#[cfg(not(target_arch = "aarch64"))]
pub fn neon_mul_f32(a: &[f32], b: &[f32], out: &mut [f32], len: usize) {
    assert!(a.len() >= len && b.len() >= len && out.len() >= len);
    for i in 0..len {
        out[i] = a[i] * b[i];
    }
}

// ── Padding / Unpadding ─────────────────────────────────────────────────

/// Pad token sequences in `batch` to `target_len` with `pad_token`,
/// returning a flattened vec of length `batch.len() * target_len` and
/// a mask of the same length (1.0 = real, 0.0 = padding).
pub fn pad_batch(batch: &[Vec<u32>], target_len: usize, pad_token: u32) -> (Vec<u32>, Vec<f32>) {
    let total = batch.len() * target_len;
    let mut padded = vec![pad_token; total];
    let mut mask = vec![0.0f32; total];

    for (i, seq) in batch.iter().enumerate() {
        let copy_len = seq.len().min(target_len);
        let offset = i * target_len;
        padded[offset..offset + copy_len].copy_from_slice(&seq[..copy_len]);
        for j in 0..copy_len {
            mask[offset + j] = 1.0;
        }
    }
    (padded, mask)
}

/// Remove padding from a padded batch, returning per-sequence token vecs
/// trimmed to `original_lengths[i]`.
pub fn unpad_batch(padded: &[u32], target_len: usize, original_lengths: &[usize]) -> Vec<Vec<u32>> {
    original_lengths
        .iter()
        .enumerate()
        .map(|(i, &orig)| {
            let offset = i * target_len;
            let end = (offset + orig).min(offset + target_len).min(padded.len());
            padded[offset..end].to_vec()
        })
        .collect()
}

/// Apply an attention mask (multiply element-wise) to a logits slice using
/// NEON-accelerated multiply. `logits` and `mask` must have the same length.
pub fn apply_padding_mask(logits: &mut [f32], mask: &[f32]) {
    let len = logits.len().min(mask.len());
    let mut out = vec![0.0f32; len];
    neon_mul_f32(logits, mask, &mut out, len);
    logits[..len].copy_from_slice(&out[..len]);
}

// ── Dynamic batch sizing ────────────────────────────────────────────────

/// Configuration for dynamic batch size adjustment.
#[derive(Debug, Clone)]
pub struct DynamicBatchConfig {
    /// Minimum batch size.
    pub min_batch: usize,
    /// Maximum batch size.
    pub max_batch: usize,
    /// Target latency in microseconds. The scheduler will shrink the batch
    /// if observed latency exceeds this and grow if below.
    pub target_latency_us: u64,
    /// Growth factor (> 1.0) when latency is below target.
    pub grow_factor: f64,
    /// Shrink factor (< 1.0) when latency exceeds target.
    pub shrink_factor: f64,
}

impl Default for DynamicBatchConfig {
    fn default() -> Self {
        Self {
            min_batch: MIN_BATCH_SIZE,
            max_batch: DEFAULT_MAX_BATCH,
            target_latency_us: 50_000, // 50 ms
            grow_factor: 1.25,
            shrink_factor: 0.75,
        }
    }
}

/// Compute the next batch size given the current size and observed latency.
pub fn adjust_batch_size(
    current: usize,
    observed_latency_us: u64,
    cfg: &DynamicBatchConfig,
) -> usize {
    let next = if observed_latency_us > cfg.target_latency_us {
        ((current as f64) * cfg.shrink_factor).floor() as usize
    } else {
        ((current as f64) * cfg.grow_factor).ceil() as usize
    };
    next.clamp(cfg.min_batch, cfg.max_batch)
}

// ── BatchScheduler ──────────────────────────────────────────────────────

/// NEON-optimized batch scheduler for Apple Silicon inference.
///
/// Manages a priority queue of [`BatchRequest`]s, dynamically sizes batches
/// based on observed latency, tracks completion, and maintains running
/// throughput / latency statistics accelerated by NEON intrinsics.
pub struct BatchScheduler {
    /// Priority queue of pending requests.
    queue: BinaryHeap<BatchRequest>,
    /// Current effective batch size.
    current_batch_size: usize,
    /// Dynamic sizing configuration.
    config: DynamicBatchConfig,
    /// Running statistics.
    stats: BatchStatistics,
    /// Completed request log.
    completed: Vec<CompletedRequest>,
    /// Next auto-assigned request ID.
    next_id: u64,
}

impl BatchScheduler {
    /// Create a scheduler with the given configuration.
    pub fn new(config: DynamicBatchConfig) -> Self {
        let initial = config.max_batch;
        Self {
            queue: BinaryHeap::new(),
            current_batch_size: initial,
            config,
            stats: BatchStatistics::new(1024),
            completed: Vec::new(),
            next_id: 0,
        }
    }

    /// Create a scheduler with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(DynamicBatchConfig::default())
    }

    /// Submit a request with explicit priority.
    pub fn submit(&mut self, token_ids: Vec<u32>, priority: Priority) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.queue.push(BatchRequest::new(id, token_ids, priority));
        id
    }

    /// Submit a normal-priority request.
    pub fn submit_normal(&mut self, token_ids: Vec<u32>) -> u64 {
        self.submit(token_ids, Priority::Normal)
    }

    /// Number of pending requests in the queue.
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Current effective batch size.
    pub fn current_batch_size(&self) -> usize {
        self.current_batch_size
    }

    /// Drain up to `current_batch_size` highest-priority requests from the
    /// queue and return them as the next batch.
    pub fn schedule_batch(&mut self) -> Vec<BatchRequest> {
        let n = self.current_batch_size.min(self.queue.len());
        let mut batch = Vec::with_capacity(n);
        for _ in 0..n {
            if let Some(req) = self.queue.pop() {
                batch.push(req);
            }
        }
        batch
    }

    /// Record that a batch of the given token count took `elapsed_us` and
    /// adjust the batch size accordingly.
    pub fn record_batch_completion(&mut self, tokens: usize, elapsed_us: u64) {
        self.stats.record_batch(tokens, elapsed_us);
        self.current_batch_size =
            adjust_batch_size(self.current_batch_size, elapsed_us, &self.config);
    }

    /// Record an individual request completion.
    pub fn record_request_completion(
        &mut self,
        id: u64,
        status: CompletionStatus,
        tokens_processed: usize,
        latency_us: u64,
    ) {
        self.completed.push(CompletedRequest { id, status, tokens_processed, latency_us });
    }

    /// Number of completed requests.
    pub fn completed_count(&self) -> usize {
        self.completed.len()
    }

    /// Retrieve completed request records.
    pub fn completed_requests(&self) -> &[CompletedRequest] {
        &self.completed
    }

    /// Count of completed requests with a given status.
    pub fn count_by_status(&self, status: CompletionStatus) -> usize {
        self.completed.iter().filter(|c| c.status == status).count()
    }

    /// Reference to running statistics.
    pub fn statistics(&self) -> &BatchStatistics {
        &self.stats
    }

    /// Mutable reference to running statistics (e.g. for reset).
    pub fn statistics_mut(&mut self) -> &mut BatchStatistics {
        &mut self.stats
    }

    /// Drain all pending requests (e.g. on shutdown).
    pub fn drain_all(&mut self) -> Vec<BatchRequest> {
        let mut out = Vec::with_capacity(self.queue.len());
        while let Some(req) = self.queue.pop() {
            out.push(req);
        }
        out
    }

    /// Maximum batch size from configuration.
    pub fn max_batch_size(&self) -> usize {
        self.config.max_batch
    }

    /// Minimum batch size from configuration.
    pub fn min_batch_size(&self) -> usize {
        self.config.min_batch
    }

    /// Compute the maximum sequence length across a batch of requests.
    pub fn max_seq_len(batch: &[BatchRequest]) -> usize {
        batch.iter().map(|r| r.seq_len()).max().unwrap_or(0)
    }

    /// Pad a batch of requests to uniform length for batched inference and
    /// return (padded_tokens, mask, original_lengths).
    pub fn pad_requests(
        batch: &[BatchRequest],
        pad_token: u32,
    ) -> (Vec<u32>, Vec<f32>, Vec<usize>) {
        let target_len = Self::max_seq_len(batch);
        if target_len == 0 {
            return (Vec::new(), Vec::new(), Vec::new());
        }
        let seqs: Vec<Vec<u32>> = batch.iter().map(|r| r.token_ids.clone()).collect();
        let original_lengths: Vec<usize> = seqs.iter().map(|s| s.len()).collect();
        let (padded, mask) = pad_batch(&seqs, target_len, pad_token);
        (padded, mask, original_lengths)
    }

    /// Total tokens across a batch of requests, computed with NEON
    /// acceleration when the lengths are cast to f32.
    pub fn total_tokens_in_batch(batch: &[BatchRequest]) -> usize {
        if batch.is_empty() {
            return 0;
        }
        let lens: Vec<f32> = batch.iter().map(|r| r.seq_len() as f32).collect();
        neon_sum_f32(&lens) as usize
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- Priority tests --------------------------------------------------

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }

    #[test]
    fn test_priority_default_is_normal() {
        assert_eq!(Priority::default(), Priority::Normal);
    }

    #[test]
    fn test_priority_equality() {
        assert_eq!(Priority::High, Priority::High);
        assert_ne!(Priority::High, Priority::Low);
    }

    // -- BatchRequest tests ----------------------------------------------

    #[test]
    fn test_batch_request_seq_len() {
        let req = BatchRequest::new(1, vec![10, 20, 30], Priority::Normal);
        assert_eq!(req.seq_len(), 3);
    }

    #[test]
    fn test_batch_request_empty_tokens() {
        let req = BatchRequest::new(0, vec![], Priority::Low);
        assert_eq!(req.seq_len(), 0);
    }

    #[test]
    fn test_batch_request_equality_by_id() {
        let a = BatchRequest::new(42, vec![1], Priority::Low);
        let b = BatchRequest::new(42, vec![9, 9], Priority::Critical);
        assert_eq!(a, b); // equality is by id only
    }

    #[test]
    fn test_batch_request_ordering_by_priority() {
        let low = BatchRequest::new(1, vec![1], Priority::Low);
        let high = BatchRequest::new(2, vec![1], Priority::High);
        assert!(high > low);
    }

    // -- Padding / Unpadding tests ---------------------------------------

    #[test]
    fn test_pad_batch_uniform() {
        let seqs = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let (padded, mask) = pad_batch(&seqs, 3, 0);
        assert_eq!(padded, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(mask, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_pad_batch_with_shorter_seq() {
        let seqs = vec![vec![1, 2], vec![3, 4, 5]];
        let (padded, mask) = pad_batch(&seqs, 3, 0);
        assert_eq!(padded, vec![1, 2, 0, 3, 4, 5]);
        assert_eq!(mask, vec![1.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_pad_batch_empty() {
        let seqs: Vec<Vec<u32>> = vec![];
        let (padded, mask) = pad_batch(&seqs, 4, 0);
        assert!(padded.is_empty());
        assert!(mask.is_empty());
    }

    #[test]
    fn test_pad_batch_single_token() {
        let seqs = vec![vec![7]];
        let (padded, mask) = pad_batch(&seqs, 4, 99);
        assert_eq!(padded, vec![7, 99, 99, 99]);
        assert_eq!(mask, vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_unpad_batch_restores_originals() {
        let seqs = vec![vec![1, 2], vec![3, 4, 5]];
        let (padded, _mask) = pad_batch(&seqs, 3, 0);
        let restored = unpad_batch(&padded, 3, &[2, 3]);
        assert_eq!(restored, seqs);
    }

    #[test]
    fn test_unpad_batch_empty() {
        let restored = unpad_batch(&[], 4, &[]);
        assert!(restored.is_empty());
    }

    #[test]
    fn test_pad_unpad_roundtrip() {
        let seqs = vec![vec![10, 20], vec![30], vec![40, 50, 60, 70]];
        let target = 4;
        let orig_lens: Vec<usize> = seqs.iter().map(|s| s.len()).collect();
        let (padded, _mask) = pad_batch(&seqs, target, 0);
        let restored = unpad_batch(&padded, target, &orig_lens);
        assert_eq!(restored, seqs);
    }

    // -- apply_padding_mask tests ----------------------------------------

    #[test]
    fn test_apply_padding_mask_zeroes_padding() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![1.0, 1.0, 0.0, 0.0];
        apply_padding_mask(&mut logits, &mask);
        assert_eq!(logits, vec![1.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn test_apply_padding_mask_all_real() {
        let mut logits = vec![5.0; 8];
        let mask = vec![1.0; 8];
        apply_padding_mask(&mut logits, &mask);
        assert_eq!(logits, vec![5.0; 8]);
    }

    #[test]
    fn test_apply_padding_mask_empty() {
        let mut logits: Vec<f32> = vec![];
        let mask: Vec<f32> = vec![];
        apply_padding_mask(&mut logits, &mask);
        assert!(logits.is_empty());
    }

    // -- neon_sum_f32 tests ----------------------------------------------

    #[test]
    fn test_neon_sum_empty() {
        assert_eq!(neon_sum_f32(&[]), 0.0);
    }

    #[test]
    fn test_neon_sum_single() {
        assert!((neon_sum_f32(&[3.5]) - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_neon_sum_exact_lanes() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!((neon_sum_f32(&data) - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_neon_sum_with_remainder() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        assert!((neon_sum_f32(&data) - 28.0).abs() < 1e-5);
    }

    #[test]
    fn test_neon_sum_large() {
        let data: Vec<f32> = (1..=100).map(|x| x as f32).collect();
        let expected = 5050.0f32;
        assert!((neon_sum_f32(&data) - expected).abs() < 1.0);
    }

    // -- neon_mul_f32 tests ----------------------------------------------

    #[test]
    fn test_neon_mul_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let mut out = vec![0.0; 4];
        neon_mul_f32(&a, &b, &mut out, 4);
        assert_eq!(out, vec![2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    fn test_neon_mul_with_remainder() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 2.0, 2.0, 2.0, 2.0];
        let mut out = vec![0.0; 5];
        neon_mul_f32(&a, &b, &mut out, 5);
        assert_eq!(out, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_neon_mul_zeros() {
        let a = vec![1.0; 8];
        let b = vec![0.0; 8];
        let mut out = vec![9.0; 8];
        neon_mul_f32(&a, &b, &mut out, 8);
        assert_eq!(out, vec![0.0; 8]);
    }

    // -- BatchStatistics tests -------------------------------------------

    #[test]
    fn test_stats_initial() {
        let s = BatchStatistics::new(16);
        assert_eq!(s.total_tokens, 0);
        assert_eq!(s.total_batches, 0);
        assert_eq!(s.throughput_tps(), 0.0);
        assert_eq!(s.mean_latency_us(), 0.0);
    }

    #[test]
    fn test_stats_record_single() {
        let mut s = BatchStatistics::new(16);
        s.record_batch(100, 1_000_000); // 100 tokens in 1 s
        assert_eq!(s.total_tokens, 100);
        assert_eq!(s.total_batches, 1);
        assert!((s.throughput_tps() - 100.0).abs() < 1e-3);
    }

    #[test]
    fn test_stats_mean_latency() {
        let mut s = BatchStatistics::new(16);
        s.record_batch(10, 100);
        s.record_batch(10, 200);
        s.record_batch(10, 300);
        assert!((s.mean_latency_us() - 200.0).abs() < 1e-3);
    }

    #[test]
    fn test_stats_p50_p99() {
        let mut s = BatchStatistics::new(128);
        for i in 1..=100 {
            s.record_batch(1, i as u64);
        }
        let p50 = s.p50_latency_us();
        let p99 = s.p99_latency_us();
        assert!(p50 >= 49.0 && p50 <= 51.0, "p50={p50}");
        assert!(p99 >= 98.0 && p99 <= 100.0, "p99={p99}");
    }

    #[test]
    fn test_stats_ring_buffer_eviction() {
        let mut s = BatchStatistics::new(4);
        for i in 0..10 {
            s.record_batch(1, (i + 1) * 10);
        }
        assert_eq!(s.latency_sample_count(), 4);
        // Only the last 4 entries: 70, 80, 90, 100
        assert!((s.mean_latency_us() - 85.0).abs() < 1e-3);
    }

    #[test]
    fn test_stats_reset() {
        let mut s = BatchStatistics::new(16);
        s.record_batch(50, 500);
        s.reset();
        assert_eq!(s.total_tokens, 0);
        assert_eq!(s.total_batches, 0);
        assert_eq!(s.latency_sample_count(), 0);
    }

    // -- DynamicBatchConfig / adjust_batch_size tests --------------------

    #[test]
    fn test_adjust_grows_under_target() {
        let cfg = DynamicBatchConfig {
            min_batch: 1,
            max_batch: 64,
            target_latency_us: 50_000,
            grow_factor: 2.0,
            shrink_factor: 0.5,
        };
        let next = adjust_batch_size(8, 20_000, &cfg); // well under target
        assert_eq!(next, 16); // 8 * 2.0
    }

    #[test]
    fn test_adjust_shrinks_over_target() {
        let cfg = DynamicBatchConfig {
            min_batch: 1,
            max_batch: 64,
            target_latency_us: 50_000,
            grow_factor: 2.0,
            shrink_factor: 0.5,
        };
        let next = adjust_batch_size(16, 80_000, &cfg); // over target
        assert_eq!(next, 8); // 16 * 0.5
    }

    #[test]
    fn test_adjust_clamps_to_min() {
        let cfg = DynamicBatchConfig {
            min_batch: 4,
            max_batch: 64,
            target_latency_us: 50_000,
            grow_factor: 2.0,
            shrink_factor: 0.1,
        };
        let next = adjust_batch_size(4, 100_000, &cfg);
        assert_eq!(next, 4); // floor(4*0.1)=0 → clamped to 4
    }

    #[test]
    fn test_adjust_clamps_to_max() {
        let cfg = DynamicBatchConfig {
            min_batch: 1,
            max_batch: 32,
            target_latency_us: 50_000,
            grow_factor: 10.0,
            shrink_factor: 0.5,
        };
        let next = adjust_batch_size(10, 1_000, &cfg);
        assert_eq!(next, 32); // ceil(10*10)=100 → clamped to 32
    }

    #[test]
    fn test_default_dynamic_config() {
        let cfg = DynamicBatchConfig::default();
        assert_eq!(cfg.min_batch, MIN_BATCH_SIZE);
        assert_eq!(cfg.max_batch, DEFAULT_MAX_BATCH);
        assert!(cfg.grow_factor > 1.0);
        assert!(cfg.shrink_factor < 1.0);
    }

    // -- BatchScheduler core tests ---------------------------------------

    #[test]
    fn test_scheduler_starts_empty() {
        let s = BatchScheduler::with_defaults();
        assert!(s.is_empty());
        assert_eq!(s.pending_count(), 0);
    }

    #[test]
    fn test_submit_increments_pending() {
        let mut s = BatchScheduler::with_defaults();
        s.submit_normal(vec![1, 2, 3]);
        s.submit_normal(vec![4, 5]);
        assert_eq!(s.pending_count(), 2);
    }

    #[test]
    fn test_submit_returns_unique_ids() {
        let mut s = BatchScheduler::with_defaults();
        let a = s.submit_normal(vec![1]);
        let b = s.submit_normal(vec![2]);
        let c = s.submit(vec![3], Priority::High);
        assert_ne!(a, b);
        assert_ne!(b, c);
    }

    #[test]
    fn test_schedule_batch_drains_queue() {
        let mut s = BatchScheduler::new(DynamicBatchConfig { max_batch: 2, ..Default::default() });
        s.submit_normal(vec![1]);
        s.submit_normal(vec![2]);
        s.submit_normal(vec![3]);
        let batch = s.schedule_batch();
        assert_eq!(batch.len(), 2);
        assert_eq!(s.pending_count(), 1);
    }

    #[test]
    fn test_schedule_respects_priority() {
        let mut s = BatchScheduler::new(DynamicBatchConfig { max_batch: 1, ..Default::default() });
        s.submit(vec![1], Priority::Low);
        s.submit(vec![2], Priority::Critical);
        let batch = s.schedule_batch();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].token_ids, vec![2]); // Critical first
    }

    #[test]
    fn test_schedule_batch_empty_queue() {
        let mut s = BatchScheduler::with_defaults();
        let batch = s.schedule_batch();
        assert!(batch.is_empty());
    }

    #[test]
    fn test_drain_all() {
        let mut s = BatchScheduler::with_defaults();
        s.submit_normal(vec![1]);
        s.submit_normal(vec![2]);
        let drained = s.drain_all();
        assert_eq!(drained.len(), 2);
        assert!(s.is_empty());
    }

    // -- Completion tracking tests ---------------------------------------

    #[test]
    fn test_completion_tracking() {
        let mut s = BatchScheduler::with_defaults();
        s.record_request_completion(0, CompletionStatus::Success, 10, 500);
        s.record_request_completion(1, CompletionStatus::Failed, 5, 300);
        assert_eq!(s.completed_count(), 2);
        assert_eq!(s.count_by_status(CompletionStatus::Success), 1);
        assert_eq!(s.count_by_status(CompletionStatus::Failed), 1);
        assert_eq!(s.count_by_status(CompletionStatus::TimedOut), 0);
    }

    #[test]
    fn test_completed_requests_returns_slice() {
        let mut s = BatchScheduler::with_defaults();
        s.record_request_completion(42, CompletionStatus::Success, 8, 1000);
        let c = s.completed_requests();
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].id, 42);
        assert_eq!(c[0].tokens_processed, 8);
    }

    // -- Batch size adjustment integration -------------------------------

    #[test]
    fn test_record_batch_adjusts_size() {
        let cfg = DynamicBatchConfig {
            min_batch: 1,
            max_batch: 64,
            target_latency_us: 50_000,
            grow_factor: 2.0,
            shrink_factor: 0.5,
        };
        let mut s = BatchScheduler::new(cfg);
        let initial = s.current_batch_size();
        // Record a very slow batch → should shrink
        s.record_batch_completion(10, 100_000);
        assert!(s.current_batch_size() < initial);
    }

    #[test]
    fn test_record_batch_grows_when_fast() {
        let cfg = DynamicBatchConfig {
            min_batch: 1,
            max_batch: 128,
            target_latency_us: 50_000,
            grow_factor: 2.0,
            shrink_factor: 0.5,
        };
        let mut s = BatchScheduler::new(cfg);
        // Force a small starting size by shrinking first
        s.record_batch_completion(10, 100_000); // shrink
        let after_shrink = s.current_batch_size();
        s.record_batch_completion(10, 1_000); // fast → grow
        assert!(s.current_batch_size() > after_shrink);
    }

    // -- Static helpers (max_seq_len, pad_requests, total_tokens) --------

    #[test]
    fn test_max_seq_len() {
        let batch = vec![
            BatchRequest::new(0, vec![1, 2, 3], Priority::Normal),
            BatchRequest::new(1, vec![4], Priority::Normal),
            BatchRequest::new(2, vec![5, 6], Priority::Normal),
        ];
        assert_eq!(BatchScheduler::max_seq_len(&batch), 3);
    }

    #[test]
    fn test_max_seq_len_empty() {
        assert_eq!(BatchScheduler::max_seq_len(&[]), 0);
    }

    #[test]
    fn test_pad_requests() {
        let batch = vec![
            BatchRequest::new(0, vec![1, 2], Priority::Normal),
            BatchRequest::new(1, vec![3, 4, 5], Priority::Normal),
        ];
        let (padded, mask, orig) = BatchScheduler::pad_requests(&batch, 0);
        assert_eq!(padded.len(), 6); // 2 * 3
        assert_eq!(mask.len(), 6);
        assert_eq!(orig, vec![2, 3]);
        assert_eq!(padded[2], 0); // padding token
        assert_eq!(mask[2], 0.0);
    }

    #[test]
    fn test_pad_requests_empty_batch() {
        let (padded, mask, orig) = BatchScheduler::pad_requests(&[], 0);
        assert!(padded.is_empty());
        assert!(mask.is_empty());
        assert!(orig.is_empty());
    }

    #[test]
    fn test_total_tokens_in_batch() {
        let batch = vec![
            BatchRequest::new(0, vec![1, 2, 3], Priority::Normal),
            BatchRequest::new(1, vec![4, 5], Priority::Normal),
        ];
        assert_eq!(BatchScheduler::total_tokens_in_batch(&batch), 5);
    }

    #[test]
    fn test_total_tokens_empty_batch() {
        assert_eq!(BatchScheduler::total_tokens_in_batch(&[]), 0);
    }

    // -- Config accessors ------------------------------------------------

    #[test]
    fn test_max_min_batch_accessors() {
        let cfg = DynamicBatchConfig { min_batch: 2, max_batch: 48, ..Default::default() };
        let s = BatchScheduler::new(cfg);
        assert_eq!(s.max_batch_size(), 48);
        assert_eq!(s.min_batch_size(), 2);
    }

    // -- Statistics integration ------------------------------------------

    #[test]
    fn test_statistics_accessible() {
        let mut s = BatchScheduler::with_defaults();
        s.record_batch_completion(50, 500_000);
        assert_eq!(s.statistics().total_tokens, 50);
        assert_eq!(s.statistics().total_batches, 1);
    }

    #[test]
    fn test_statistics_reset_via_mut() {
        let mut s = BatchScheduler::with_defaults();
        s.record_batch_completion(50, 500_000);
        s.statistics_mut().reset();
        assert_eq!(s.statistics().total_tokens, 0);
    }
}
