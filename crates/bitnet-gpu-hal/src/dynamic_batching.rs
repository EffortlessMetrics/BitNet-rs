//! Module stub - implementation pending merge from feature branch
//! Dynamic request batching with efficiency-aware scheduling.
//!
//! Groups incoming inference requests into efficient batches using
//! configurable formation strategies (greedy, first-fit, best-fit),
//! padding-aware scheduling, and real-time efficiency tracking.

use std::collections::VecDeque;
use std::fmt;
use std::time::{Duration, Instant};

// ── Configuration ───────────────────────────────────────────────────────────

/// Padding strategy for aligning sequences within a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingStrategy {
    /// Pad all sequences to the length of the longest in the batch.
    PadToLongest,
    /// Pad all sequences to a fixed maximum length.
    PadToMax(usize),
    /// No padding — each sequence keeps its original length.
    NoPadding,
}

/// Configuration for the dynamic batcher.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of requests per batch.
    pub max_batch_size: usize,
    /// Maximum time (ms) to wait before forming a batch.
    pub max_wait_ms: u64,
    /// How sequences are padded within a batch.
    pub padding_strategy: PaddingStrategy,
    /// Whether to sort requests by token length before batching.
    pub sort_by_length: bool,
    /// Maximum total tokens (prompt + generation) across all requests in a batch.
    pub max_total_tokens: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_wait_ms: 50,
            padding_strategy: PaddingStrategy::PadToLongest,
            sort_by_length: true,
            max_total_tokens: 8192,
        }
    }
}

// ── Request ─────────────────────────────────────────────────────────────────

/// An incoming inference request awaiting batching.
#[derive(Debug, Clone)]
pub struct BatchRequest {
    /// Unique request identifier.
    pub id: u64,
    /// Input token IDs.
    pub tokens: Vec<u32>,
    /// Scheduling priority (higher = more urgent).
    pub priority: u32,
    /// When this request arrived.
    pub arrival_time: Instant,
    /// Maximum tokens this request may generate.
    pub max_generation_tokens: usize,
}

impl BatchRequest {
    /// Total token budget: prompt length + max generation tokens.
    #[must_use]
    pub const fn total_token_budget(&self) -> usize {
        self.tokens.len() + self.max_generation_tokens
    }
}

// ── Dynamic batch ───────────────────────────────────────────────────────────

/// A formed batch ready for dispatch.
#[derive(Debug, Clone)]
pub struct DynamicBatch {
    /// Requests assigned to this batch.
    pub requests: Vec<BatchRequest>,
    /// Total padding tokens inserted.
    pub padding_tokens: usize,
    /// Total tokens across all requests (including padding).
    pub total_tokens: usize,
    /// Batch efficiency (useful / total).
    pub efficiency: f64,
}

impl DynamicBatch {
    /// Number of requests in this batch.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.requests.len()
    }

    /// Whether this batch contains no requests.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }
}

// ── Formation strategy ──────────────────────────────────────────────────────

/// Strategy for forming batches from the request queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchFormationStrategy {
    /// Take requests in FIFO order up to the batch-size limit.
    Greedy,
    /// Scan the queue and add the first request that fits within token limits.
    FirstFit,
    /// Scan the entire queue and pick the request that best fills remaining capacity.
    BestFit,
    /// Sort by token length, then batch greedily.
    SortedGreedy,
}

impl fmt::Display for BatchFormationStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Greedy => write!(f, "Greedy"),
            Self::FirstFit => write!(f, "FirstFit"),
            Self::BestFit => write!(f, "BestFit"),
            Self::SortedGreedy => write!(f, "SortedGreedy"),
        }
    }
}

// ── Padding calculator ──────────────────────────────────────────────────────

/// Calculates padding overhead for different batch compositions.
pub struct PaddingCalculator;

impl PaddingCalculator {
    /// Compute the number of padding tokens for a set of sequence lengths.
    #[must_use]
    pub fn compute(lengths: &[usize], strategy: PaddingStrategy) -> usize {
        if lengths.is_empty() {
            return 0;
        }

        let max_len = match strategy {
            PaddingStrategy::PadToLongest => lengths.iter().copied().max().unwrap_or(0),
            PaddingStrategy::PadToMax(m) => m,
            PaddingStrategy::NoPadding => return 0,
        };

        lengths.iter().map(|&l| max_len.saturating_sub(l)).sum()
    }

    /// Compute the total tokens (useful + padding) for the given lengths.
    #[must_use]
    pub fn total_with_padding(lengths: &[usize], strategy: PaddingStrategy) -> usize {
        let useful: usize = lengths.iter().sum();
        useful + Self::compute(lengths, strategy)
    }
}

// ── Batch scheduler ─────────────────────────────────────────────────────────

/// Decides when to form and dispatch batches.
pub struct BatchScheduler {
    config: BatchConfig,
}

impl BatchScheduler {
    /// Create a new scheduler with the given config.
    #[must_use]
    pub const fn new(config: BatchConfig) -> Self {
        Self { config }
    }

    /// Whether a batch should be dispatched now based on size.
    #[must_use]
    pub const fn should_dispatch_by_size(&self, queue_len: usize) -> bool {
        queue_len >= self.config.max_batch_size
    }

    /// Whether a batch should be dispatched now based on timeout.
    #[must_use]
    pub fn should_dispatch_by_timeout(&self, oldest_arrival: Instant) -> bool {
        oldest_arrival.elapsed() >= Duration::from_millis(self.config.max_wait_ms)
    }

    /// Whether a batch should be dispatched based on efficiency threshold.
    #[must_use]
    pub fn should_dispatch_by_efficiency(&self, queue: &[BatchRequest], threshold: f64) -> bool {
        if queue.is_empty() {
            return false;
        }
        let efficiency = BatchEfficiency::estimate(queue, self.config.padding_strategy);
        efficiency >= threshold
    }

    /// Reference to the inner config.
    #[must_use]
    pub const fn config(&self) -> &BatchConfig {
        &self.config
    }
}

// ── Batch efficiency ────────────────────────────────────────────────────────

/// Computes batch efficiency: `useful_tokens / total_tokens`.
pub struct BatchEfficiency;

impl BatchEfficiency {
    /// Compute efficiency for a formed batch.
    #[must_use]
    pub fn compute(useful_tokens: usize, total_tokens: usize) -> f64 {
        if total_tokens == 0 {
            return 1.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let eff = useful_tokens as f64 / total_tokens as f64;
        eff
    }

    /// Estimate efficiency for a set of pending requests.
    #[must_use]
    pub fn estimate(requests: &[BatchRequest], strategy: PaddingStrategy) -> f64 {
        if requests.is_empty() {
            return 1.0;
        }
        let lengths: Vec<usize> = requests.iter().map(|r| r.tokens.len()).collect();
        let useful: usize = lengths.iter().sum();
        let total = PaddingCalculator::total_with_padding(&lengths, strategy);
        Self::compute(useful, total)
    }
}

// ── In-flight batch tracker ─────────────────────────────────────────────────

/// Tracks a single in-flight batch.
#[derive(Debug, Clone)]
pub struct InflightEntry {
    /// Unique batch identifier.
    pub batch_id: u64,
    /// Number of requests in this batch.
    pub request_count: usize,
    /// Total tokens allocated for this batch.
    pub total_tokens: usize,
    /// When the batch was dispatched.
    pub dispatched_at: Instant,
}

/// Tracks batches currently being processed.
pub struct InflightBatchTracker {
    entries: Vec<InflightEntry>,
    next_batch_id: u64,
}

impl InflightBatchTracker {
    /// Create a new, empty tracker.
    #[must_use]
    pub const fn new() -> Self {
        Self { entries: Vec::new(), next_batch_id: 0 }
    }

    /// Register a batch as in-flight. Returns the assigned batch ID.
    pub fn register(&mut self, request_count: usize, total_tokens: usize) -> u64 {
        let id = self.next_batch_id;
        self.next_batch_id += 1;
        self.entries.push(InflightEntry {
            batch_id: id,
            request_count,
            total_tokens,
            dispatched_at: Instant::now(),
        });
        id
    }

    /// Mark a batch as completed and remove it from tracking.
    pub fn complete(&mut self, batch_id: u64) -> Option<InflightEntry> {
        if let Some(pos) = self.entries.iter().position(|e| e.batch_id == batch_id) {
            Some(self.entries.swap_remove(pos))
        } else {
            None
        }
    }

    /// Number of batches currently in flight.
    #[must_use]
    pub const fn active_count(&self) -> usize {
        self.entries.len()
    }

    /// Total tokens currently allocated across all in-flight batches.
    #[must_use]
    pub fn total_inflight_tokens(&self) -> usize {
        self.entries.iter().map(|e| e.total_tokens).sum()
    }

    /// Whether no batches are in flight.
    #[must_use]
    pub const fn is_idle(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for InflightBatchTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Batch metrics ───────────────────────────────────────────────────────────

/// Aggregate metrics for the dynamic batcher.
#[derive(Debug, Clone)]
pub struct BatchMetrics {
    /// Total batches formed.
    pub batches_formed: u64,
    /// Total requests processed.
    pub requests_processed: u64,
    /// Sum of all batch sizes (for averaging).
    pub total_batch_sizes: u64,
    /// Sum of all request wait times in microseconds.
    pub total_wait_us: u64,
    /// Sum of padding tokens across all batches.
    pub total_padding_tokens: u64,
    /// Sum of useful tokens across all batches.
    pub total_useful_tokens: u64,
    /// Sum of batch formation durations in microseconds.
    pub total_formation_us: u64,
}

impl BatchMetrics {
    /// Create zeroed-out metrics.
    pub const fn new() -> Self {
        Self {
            batches_formed: 0,
            requests_processed: 0,
            total_batch_sizes: 0,
            total_wait_us: 0,
            total_padding_tokens: 0,
            total_useful_tokens: 0,
            total_formation_us: 0,
        }
    }

    /// Average batch size.
    #[must_use]
    pub fn avg_batch_size(&self) -> f64 {
        if self.batches_formed == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let avg = self.total_batch_sizes as f64 / self.batches_formed as f64;
        avg
    }

    /// Average request wait time in microseconds.
    #[must_use]
    pub fn avg_wait_us(&self) -> f64 {
        if self.requests_processed == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let avg = self.total_wait_us as f64 / self.requests_processed as f64;
        avg
    }

    /// Overall padding overhead ratio.
    #[must_use]
    pub fn padding_overhead(&self) -> f64 {
        let total = self.total_useful_tokens + self.total_padding_tokens;
        if total == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let ratio = self.total_padding_tokens as f64 / total as f64;
        ratio
    }

    /// Requests processed per second, given a wall-clock duration.
    #[must_use]
    pub fn throughput(&self, elapsed: Duration) -> f64 {
        let secs = elapsed.as_secs_f64();
        if secs <= 0.0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let tput = self.requests_processed as f64 / secs;
        tput
    }

    /// Average batch formation time in microseconds.
    #[must_use]
    pub fn avg_formation_us(&self) -> f64 {
        if self.batches_formed == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let avg = self.total_formation_us as f64 / self.batches_formed as f64;
        avg
    }

    /// Record one formed batch.
    pub const fn record_batch(
        &mut self,
        batch_size: usize,
        useful_tokens: usize,
        padding_tokens: usize,
        formation_us: u64,
        wait_us_sum: u64,
    ) {
        self.batches_formed += 1;
        #[allow(clippy::cast_possible_truncation)]
        {
            self.requests_processed += batch_size as u64;
            self.total_batch_sizes += batch_size as u64;
            self.total_useful_tokens += useful_tokens as u64;
            self.total_padding_tokens += padding_tokens as u64;
        }
        self.total_formation_us += formation_us;
        self.total_wait_us += wait_us_sum;
    }
}

impl Default for BatchMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ── Dynamic batcher ─────────────────────────────────────────────────────────

/// Main batcher: accepts requests, forms batches, tracks metrics.
pub struct DynamicBatcher {
    config: BatchConfig,
    strategy: BatchFormationStrategy,
    queue: VecDeque<BatchRequest>,
    metrics: BatchMetrics,
    inflight: InflightBatchTracker,
}

impl DynamicBatcher {
    /// Create a new batcher with the given config and strategy.
    #[must_use]
    pub const fn new(config: BatchConfig, strategy: BatchFormationStrategy) -> Self {
        Self {
            config,
            strategy,
            queue: VecDeque::new(),
            metrics: BatchMetrics::new(),
            inflight: InflightBatchTracker::new(),
        }
    }

    /// Submit a request to the batcher queue.
    pub fn submit(&mut self, request: BatchRequest) {
        self.queue.push_back(request);
    }

    /// Number of requests waiting in the queue.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }

    /// Try to form a batch from the current queue.
    ///
    /// Returns `None` if the queue is empty.
    pub fn try_form_batch(&mut self) -> Option<DynamicBatch> {
        if self.queue.is_empty() {
            return None;
        }

        let start = Instant::now();

        let mut candidates: Vec<BatchRequest> = self.queue.drain(..).collect();

        // Pre-sort if configured or using SortedGreedy strategy.
        if self.config.sort_by_length || self.strategy == BatchFormationStrategy::SortedGreedy {
            candidates.sort_by_key(|r| r.tokens.len());
        }

        let selected = match self.strategy {
            BatchFormationStrategy::Greedy | BatchFormationStrategy::SortedGreedy => {
                self.form_greedy(&mut candidates)
            }
            BatchFormationStrategy::FirstFit => self.form_first_fit(&mut candidates),
            BatchFormationStrategy::BestFit => self.form_best_fit(&mut candidates),
        };

        // Return unselected candidates to the queue.
        for c in candidates {
            self.queue.push_back(c);
        }

        if selected.is_empty() {
            return None;
        }

        let lengths: Vec<usize> = selected.iter().map(|r| r.tokens.len()).collect();
        let useful_tokens: usize = lengths.iter().sum();
        let padding_tokens = PaddingCalculator::compute(&lengths, self.config.padding_strategy);
        let total_tokens = useful_tokens + padding_tokens;
        let efficiency = BatchEfficiency::compute(useful_tokens, total_tokens);

        #[allow(clippy::cast_possible_truncation)]
        let formation_us = start.elapsed().as_micros() as u64;
        let now = Instant::now();
        #[allow(clippy::cast_possible_truncation)]
        let wait_us_sum: u64 =
            selected.iter().map(|r| now.duration_since(r.arrival_time).as_micros() as u64).sum();

        self.metrics.record_batch(
            selected.len(),
            useful_tokens,
            padding_tokens,
            formation_us,
            wait_us_sum,
        );

        let batch_id = self.inflight.register(selected.len(), total_tokens);
        log::debug!(
            "Formed batch {batch_id}: {} requests, {total_tokens} tokens, efficiency {efficiency:.2}",
            selected.len()
        );

        Some(DynamicBatch { requests: selected, padding_tokens, total_tokens, efficiency })
    }

    /// Mark an in-flight batch as completed.
    pub fn complete_batch(&mut self, batch_id: u64) -> Option<InflightEntry> {
        self.inflight.complete(batch_id)
    }

    /// Get a snapshot of the current metrics.
    #[must_use]
    pub const fn metrics(&self) -> &BatchMetrics {
        &self.metrics
    }

    /// Reference to the in-flight tracker.
    #[must_use]
    pub const fn inflight(&self) -> &InflightBatchTracker {
        &self.inflight
    }

    /// The active formation strategy.
    #[must_use]
    pub const fn strategy(&self) -> BatchFormationStrategy {
        self.strategy
    }

    // ── Private helpers ─────────────────────────────────────────────────────

    fn form_greedy(&self, candidates: &mut Vec<BatchRequest>) -> Vec<BatchRequest> {
        let take = candidates.len().min(self.config.max_batch_size);
        let mut selected = Vec::with_capacity(take);
        let mut total = 0usize;

        let mut i = 0;
        while i < candidates.len() && selected.len() < self.config.max_batch_size {
            let budget = candidates[i].total_token_budget();
            if total + budget <= self.config.max_total_tokens {
                total += budget;
                selected.push(candidates.remove(i));
            } else {
                i += 1;
            }
        }
        selected
    }

    fn form_first_fit(&self, candidates: &mut Vec<BatchRequest>) -> Vec<BatchRequest> {
        let mut selected = Vec::new();
        let mut total = 0usize;

        let mut i = 0;
        while i < candidates.len() && selected.len() < self.config.max_batch_size {
            let budget = candidates[i].total_token_budget();
            if total + budget <= self.config.max_total_tokens {
                total += budget;
                selected.push(candidates.remove(i));
            } else {
                i += 1;
            }
        }
        selected
    }

    fn form_best_fit(&self, candidates: &mut Vec<BatchRequest>) -> Vec<BatchRequest> {
        let mut selected = Vec::new();
        let mut remaining = self.config.max_total_tokens;

        while !candidates.is_empty() && selected.len() < self.config.max_batch_size {
            // Find the candidate whose budget is closest to remaining capacity.
            let best_idx = candidates
                .iter()
                .enumerate()
                .filter(|(_, r)| r.total_token_budget() <= remaining)
                .min_by_key(|(_, r)| remaining - r.total_token_budget())
                .map(|(i, _)| i);

            match best_idx {
                Some(idx) => {
                    let req = candidates.remove(idx);
                    remaining = remaining.saturating_sub(req.total_token_budget());
                    selected.push(req);
                }
                None => break,
            }
        }
        selected
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn now() -> Instant {
        Instant::now()
    }

    fn req(id: u64, token_count: usize, max_gen: usize) -> BatchRequest {
        BatchRequest {
            id,
            tokens: vec![0; token_count],
            priority: 0,
            arrival_time: now(),
            max_generation_tokens: max_gen,
        }
    }

    fn req_with_priority(id: u64, token_count: usize, priority: u32) -> BatchRequest {
        BatchRequest {
            id,
            tokens: vec![0; token_count],
            priority,
            arrival_time: now(),
            max_generation_tokens: 16,
        }
    }

    fn req_with_tokens(id: u64, tokens: Vec<u32>, max_gen: usize) -> BatchRequest {
        BatchRequest {
            id,
            tokens,
            priority: 0,
            arrival_time: now(),
            max_generation_tokens: max_gen,
        }
    }

    fn default_batcher() -> DynamicBatcher {
        DynamicBatcher::new(BatchConfig::default(), BatchFormationStrategy::Greedy)
    }

    // -----------------------------------------------------------------------
    // BatchConfig defaults
    // -----------------------------------------------------------------------

    #[test]
    fn default_config_max_batch_size() {
        let cfg = BatchConfig::default();
        assert_eq!(cfg.max_batch_size, 32);
    }

    #[test]
    fn default_config_max_wait_ms() {
        let cfg = BatchConfig::default();
        assert_eq!(cfg.max_wait_ms, 50);
    }

    #[test]
    fn default_config_padding_strategy() {
        let cfg = BatchConfig::default();
        assert_eq!(cfg.padding_strategy, PaddingStrategy::PadToLongest);
    }

    #[test]
    fn default_config_sort_by_length() {
        let cfg = BatchConfig::default();
        assert!(cfg.sort_by_length);
    }

    #[test]
    fn default_config_max_total_tokens() {
        let cfg = BatchConfig::default();
        assert_eq!(cfg.max_total_tokens, 8192);
    }

    // -----------------------------------------------------------------------
    // BatchRequest
    // -----------------------------------------------------------------------

    #[test]
    fn request_total_token_budget() {
        let r = req(1, 10, 20);
        assert_eq!(r.total_token_budget(), 30);
    }

    #[test]
    fn request_zero_generation() {
        let r = req(1, 5, 0);
        assert_eq!(r.total_token_budget(), 5);
    }

    #[test]
    fn request_empty_tokens() {
        let r = req(1, 0, 10);
        assert_eq!(r.total_token_budget(), 10);
    }

    #[test]
    fn request_clone() {
        let r = req(42, 10, 5);
        let r2 = r.clone();
        assert_eq!(r.id, r2.id);
        assert_eq!(r.tokens.len(), r2.tokens.len());
    }

    #[test]
    fn request_priority_ordering() {
        let a = req_with_priority(1, 10, 5);
        let b = req_with_priority(2, 10, 10);
        assert!(b.priority > a.priority);
    }

    // -----------------------------------------------------------------------
    // DynamicBatch
    // -----------------------------------------------------------------------

    #[test]
    fn batch_size_and_empty() {
        let batch =
            DynamicBatch { requests: vec![], padding_tokens: 0, total_tokens: 0, efficiency: 1.0 };
        assert!(batch.is_empty());
        assert_eq!(batch.size(), 0);
    }

    #[test]
    fn batch_non_empty() {
        let batch = DynamicBatch {
            requests: vec![req(1, 5, 5)],
            padding_tokens: 0,
            total_tokens: 5,
            efficiency: 1.0,
        };
        assert!(!batch.is_empty());
        assert_eq!(batch.size(), 1);
    }

    // -----------------------------------------------------------------------
    // BatchFormationStrategy display
    // -----------------------------------------------------------------------

    #[test]
    fn strategy_display_greedy() {
        assert_eq!(format!("{}", BatchFormationStrategy::Greedy), "Greedy");
    }

    #[test]
    fn strategy_display_first_fit() {
        assert_eq!(format!("{}", BatchFormationStrategy::FirstFit), "FirstFit");
    }

    #[test]
    fn strategy_display_best_fit() {
        assert_eq!(format!("{}", BatchFormationStrategy::BestFit), "BestFit");
    }

    #[test]
    fn strategy_display_sorted_greedy() {
        assert_eq!(format!("{}", BatchFormationStrategy::SortedGreedy), "SortedGreedy");
    }

    #[test]
    fn strategy_equality() {
        assert_eq!(BatchFormationStrategy::Greedy, BatchFormationStrategy::Greedy);
        assert_ne!(BatchFormationStrategy::Greedy, BatchFormationStrategy::BestFit);
    }

    #[test]
    fn strategy_copy() {
        let a = BatchFormationStrategy::FirstFit;
        let b = a;
        assert_eq!(a, b);
    }

    // -----------------------------------------------------------------------
    // PaddingCalculator
    // -----------------------------------------------------------------------

    #[test]
    fn padding_pad_to_longest_uniform() {
        let padding = PaddingCalculator::compute(&[10, 10, 10], PaddingStrategy::PadToLongest);
        assert_eq!(padding, 0);
    }

    #[test]
    fn padding_pad_to_longest_varying() {
        let padding = PaddingCalculator::compute(&[5, 10, 3], PaddingStrategy::PadToLongest);
        // max=10, padding = (10-5) + (10-10) + (10-3) = 5+0+7 = 12
        assert_eq!(padding, 12);
    }

    #[test]
    fn padding_pad_to_max() {
        let padding = PaddingCalculator::compute(&[3, 5], PaddingStrategy::PadToMax(8));
        // padding = (8-3) + (8-5) = 5+3 = 8
        assert_eq!(padding, 8);
    }

    #[test]
    fn padding_no_padding() {
        let padding = PaddingCalculator::compute(&[3, 7, 1], PaddingStrategy::NoPadding);
        assert_eq!(padding, 0);
    }

    #[test]
    fn padding_empty_input() {
        assert_eq!(PaddingCalculator::compute(&[], PaddingStrategy::PadToLongest), 0);
    }

    #[test]
    fn padding_single_element() {
        assert_eq!(PaddingCalculator::compute(&[5], PaddingStrategy::PadToLongest), 0);
    }

    #[test]
    fn padding_total_with_padding() {
        let total = PaddingCalculator::total_with_padding(&[3, 7], PaddingStrategy::PadToLongest);
        // useful = 10, padding = (7-3) = 4, total = 14
        assert_eq!(total, 14);
    }

    #[test]
    fn padding_total_no_padding() {
        let total = PaddingCalculator::total_with_padding(&[3, 7], PaddingStrategy::NoPadding);
        assert_eq!(total, 10);
    }

    #[test]
    fn padding_pad_to_max_smaller_than_sequences() {
        // When max is smaller than some sequences, saturating_sub yields 0 for those.
        let padding = PaddingCalculator::compute(&[10, 5], PaddingStrategy::PadToMax(7));
        // (7-10).saturating = 0, (7-5) = 2
        assert_eq!(padding, 2);
    }

    // -----------------------------------------------------------------------
    // BatchScheduler
    // -----------------------------------------------------------------------

    #[test]
    fn scheduler_dispatch_by_size_at_limit() {
        let sched =
            BatchScheduler::new(BatchConfig { max_batch_size: 4, ..BatchConfig::default() });
        assert!(sched.should_dispatch_by_size(4));
    }

    #[test]
    fn scheduler_dispatch_by_size_below() {
        let sched =
            BatchScheduler::new(BatchConfig { max_batch_size: 4, ..BatchConfig::default() });
        assert!(!sched.should_dispatch_by_size(3));
    }

    #[test]
    fn scheduler_dispatch_by_size_above() {
        let sched =
            BatchScheduler::new(BatchConfig { max_batch_size: 4, ..BatchConfig::default() });
        assert!(sched.should_dispatch_by_size(10));
    }

    #[test]
    fn scheduler_dispatch_by_timeout() {
        let arrival = Instant::now().checked_sub(Duration::from_millis(100)).unwrap();
        let sched = BatchScheduler::new(BatchConfig { max_wait_ms: 50, ..BatchConfig::default() });
        assert!(sched.should_dispatch_by_timeout(arrival));
    }

    #[test]
    fn scheduler_no_dispatch_before_timeout() {
        let arrival = Instant::now();
        let sched =
            BatchScheduler::new(BatchConfig { max_wait_ms: 5000, ..BatchConfig::default() });
        assert!(!sched.should_dispatch_by_timeout(arrival));
    }

    #[test]
    fn scheduler_efficiency_empty_queue() {
        let sched = BatchScheduler::new(BatchConfig::default());
        assert!(!sched.should_dispatch_by_efficiency(&[], 0.5));
    }

    #[test]
    fn scheduler_efficiency_uniform_lengths() {
        let reqs = vec![req(1, 10, 0), req(2, 10, 0)];
        let sched = BatchScheduler::new(BatchConfig::default());
        // Uniform lengths → efficiency = 1.0
        assert!(sched.should_dispatch_by_efficiency(&reqs, 0.9));
    }

    #[test]
    fn scheduler_config_ref() {
        let cfg = BatchConfig { max_batch_size: 7, ..BatchConfig::default() };
        let sched = BatchScheduler::new(cfg);
        assert_eq!(sched.config().max_batch_size, 7);
    }

    // -----------------------------------------------------------------------
    // BatchEfficiency
    // -----------------------------------------------------------------------

    #[test]
    fn efficiency_perfect() {
        assert!((BatchEfficiency::compute(100, 100) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn efficiency_half() {
        assert!((BatchEfficiency::compute(50, 100) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn efficiency_zero_total() {
        assert!((BatchEfficiency::compute(0, 0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn efficiency_zero_useful() {
        assert!((BatchEfficiency::compute(0, 100) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn efficiency_estimate_uniform() {
        let reqs = vec![req(1, 10, 0), req(2, 10, 0)];
        let eff = BatchEfficiency::estimate(&reqs, PaddingStrategy::PadToLongest);
        assert!((eff - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn efficiency_estimate_varied() {
        let reqs = vec![req(1, 5, 0), req(2, 10, 0)];
        let eff = BatchEfficiency::estimate(&reqs, PaddingStrategy::PadToLongest);
        // useful=15, total=15+5=20, efficiency=0.75
        assert!((eff - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn efficiency_estimate_empty() {
        assert!(
            (BatchEfficiency::estimate(&[], PaddingStrategy::PadToLongest) - 1.0).abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn efficiency_estimate_no_padding_strategy() {
        let reqs = vec![req(1, 5, 0), req(2, 15, 0)];
        let eff = BatchEfficiency::estimate(&reqs, PaddingStrategy::NoPadding);
        assert!((eff - 1.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // InflightBatchTracker
    // -----------------------------------------------------------------------

    #[test]
    fn tracker_starts_idle() {
        let tracker = InflightBatchTracker::new();
        assert!(tracker.is_idle());
        assert_eq!(tracker.active_count(), 0);
        assert_eq!(tracker.total_inflight_tokens(), 0);
    }

    #[test]
    fn tracker_register_and_count() {
        let mut tracker = InflightBatchTracker::new();
        tracker.register(4, 100);
        assert_eq!(tracker.active_count(), 1);
        assert!(!tracker.is_idle());
    }

    #[test]
    fn tracker_register_sequential_ids() {
        let mut tracker = InflightBatchTracker::new();
        let id0 = tracker.register(1, 10);
        let id1 = tracker.register(2, 20);
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
    }

    #[test]
    fn tracker_complete_returns_entry() {
        let mut tracker = InflightBatchTracker::new();
        let id = tracker.register(3, 50);
        let entry = tracker.complete(id).unwrap();
        assert_eq!(entry.batch_id, id);
        assert_eq!(entry.request_count, 3);
        assert_eq!(entry.total_tokens, 50);
    }

    #[test]
    fn tracker_complete_unknown_returns_none() {
        let mut tracker = InflightBatchTracker::new();
        assert!(tracker.complete(999).is_none());
    }

    #[test]
    fn tracker_double_complete_returns_none() {
        let mut tracker = InflightBatchTracker::new();
        let id = tracker.register(1, 10);
        assert!(tracker.complete(id).is_some());
        assert!(tracker.complete(id).is_none());
    }

    #[test]
    fn tracker_total_inflight_tokens() {
        let mut tracker = InflightBatchTracker::new();
        tracker.register(1, 100);
        tracker.register(2, 200);
        assert_eq!(tracker.total_inflight_tokens(), 300);
    }

    #[test]
    fn tracker_tokens_decrease_after_complete() {
        let mut tracker = InflightBatchTracker::new();
        let id = tracker.register(1, 100);
        tracker.register(2, 200);
        tracker.complete(id);
        assert_eq!(tracker.total_inflight_tokens(), 200);
    }

    #[test]
    fn tracker_default_is_idle() {
        let tracker = InflightBatchTracker::default();
        assert!(tracker.is_idle());
    }

    // -----------------------------------------------------------------------
    // BatchMetrics
    // -----------------------------------------------------------------------

    #[test]
    fn metrics_new_zeroed() {
        let m = BatchMetrics::new();
        assert_eq!(m.batches_formed, 0);
        assert_eq!(m.requests_processed, 0);
    }

    #[test]
    fn metrics_default_equals_new() {
        let a = BatchMetrics::new();
        let b = BatchMetrics::default();
        assert_eq!(a.batches_formed, b.batches_formed);
    }

    #[test]
    fn metrics_avg_batch_size_no_batches() {
        let m = BatchMetrics::new();
        assert!((m.avg_batch_size() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_avg_batch_size() {
        let mut m = BatchMetrics::new();
        m.record_batch(4, 40, 0, 10, 0);
        m.record_batch(6, 60, 0, 10, 0);
        // (4+6)/2 = 5.0
        assert!((m.avg_batch_size() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_avg_wait_us_no_requests() {
        let m = BatchMetrics::new();
        assert!((m.avg_wait_us() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_avg_wait_us() {
        let mut m = BatchMetrics::new();
        m.record_batch(2, 20, 0, 0, 100); // 2 reqs, wait_sum=100us
        // avg = 100/2 = 50
        assert!((m.avg_wait_us() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_padding_overhead_no_data() {
        let m = BatchMetrics::new();
        assert!((m.padding_overhead() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_padding_overhead() {
        let mut m = BatchMetrics::new();
        m.record_batch(2, 80, 20, 0, 0); // useful=80, padding=20 → overhead=20/100=0.2
        assert!((m.padding_overhead() - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_throughput_zero_elapsed() {
        let m = BatchMetrics::new();
        assert!((m.throughput(Duration::ZERO) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_throughput() {
        let mut m = BatchMetrics::new();
        m.record_batch(10, 100, 0, 0, 0);
        // 10 requests in 2 seconds → 5 req/s
        assert!((m.throughput(Duration::from_secs(2)) - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_avg_formation_us() {
        let mut m = BatchMetrics::new();
        m.record_batch(1, 10, 0, 100, 0);
        m.record_batch(1, 10, 0, 200, 0);
        // (100+200)/2 = 150
        assert!((m.avg_formation_us() - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_avg_formation_us_no_batches() {
        let m = BatchMetrics::new();
        assert!((m.avg_formation_us() - 0.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // DynamicBatcher — greedy strategy
    // -----------------------------------------------------------------------

    #[test]
    fn batcher_empty_queue_returns_none() {
        let mut batcher = default_batcher();
        assert!(batcher.try_form_batch().is_none());
    }

    #[test]
    fn batcher_single_request() {
        let mut batcher = default_batcher();
        batcher.submit(req(1, 10, 10));
        let batch = batcher.try_form_batch().unwrap();
        assert_eq!(batch.size(), 1);
        assert_eq!(batch.requests[0].id, 1);
    }

    #[test]
    fn batcher_respects_max_batch_size() {
        let config = BatchConfig { max_batch_size: 2, ..BatchConfig::default() };
        let mut batcher = DynamicBatcher::new(config, BatchFormationStrategy::Greedy);
        for i in 0..5 {
            batcher.submit(req(i, 10, 10));
        }
        let batch = batcher.try_form_batch().unwrap();
        assert!(batch.size() <= 2);
        assert_eq!(batcher.pending_count(), 3);
    }

    #[test]
    fn batcher_respects_max_total_tokens() {
        let config = BatchConfig {
            max_batch_size: 100,
            max_total_tokens: 50,
            sort_by_length: false,
            ..BatchConfig::default()
        };
        let mut batcher = DynamicBatcher::new(config, BatchFormationStrategy::Greedy);
        // Each request budget = 10+10=20. Limit 50 → can fit 2 (40) but not 3 (60).
        for i in 0..5 {
            batcher.submit(req(i, 10, 10));
        }
        let batch = batcher.try_form_batch().unwrap();
        assert!(batch.size() <= 2);
    }

    #[test]
    fn batcher_pending_count() {
        let mut batcher = default_batcher();
        assert_eq!(batcher.pending_count(), 0);
        batcher.submit(req(1, 5, 5));
        batcher.submit(req(2, 5, 5));
        assert_eq!(batcher.pending_count(), 2);
    }

    #[test]
    fn batcher_strategy_accessor() {
        let batcher = DynamicBatcher::new(BatchConfig::default(), BatchFormationStrategy::BestFit);
        assert_eq!(batcher.strategy(), BatchFormationStrategy::BestFit);
    }

    #[test]
    fn batcher_metrics_updated_after_batch() {
        let mut batcher = default_batcher();
        batcher.submit(req(1, 10, 10));
        batcher.try_form_batch();
        assert_eq!(batcher.metrics().batches_formed, 1);
        assert_eq!(batcher.metrics().requests_processed, 1);
    }

    #[test]
    fn batcher_inflight_tracking() {
        let mut batcher = default_batcher();
        batcher.submit(req(1, 10, 10));
        batcher.try_form_batch();
        assert_eq!(batcher.inflight().active_count(), 1);
        batcher.complete_batch(0);
        assert!(batcher.inflight().is_idle());
    }

    #[test]
    fn batcher_multiple_batches() {
        let config = BatchConfig { max_batch_size: 1, ..BatchConfig::default() };
        let mut batcher = DynamicBatcher::new(config, BatchFormationStrategy::Greedy);
        batcher.submit(req(1, 5, 5));
        batcher.submit(req(2, 5, 5));
        let b1 = batcher.try_form_batch().unwrap();
        let b2 = batcher.try_form_batch().unwrap();
        assert_eq!(b1.size(), 1);
        assert_eq!(b2.size(), 1);
        assert!(batcher.try_form_batch().is_none());
    }

    // -----------------------------------------------------------------------
    // DynamicBatcher — first-fit strategy
    // -----------------------------------------------------------------------

    #[test]
    fn first_fit_skips_oversized() {
        let config = BatchConfig {
            max_batch_size: 10,
            max_total_tokens: 50,
            sort_by_length: false,
            ..BatchConfig::default()
        };
        let mut batcher = DynamicBatcher::new(config, BatchFormationStrategy::FirstFit);
        batcher.submit(req(1, 40, 10)); // budget=50, fits
        batcher.submit(req(2, 5, 5)); // budget=10, fits remaining (but total already 50)
        let batch = batcher.try_form_batch().unwrap();
        // First request (budget=50) fills the limit entirely.
        assert_eq!(batch.size(), 1);
        assert_eq!(batch.requests[0].id, 1);
    }

    #[test]
    fn first_fit_fills_gap() {
        let config = BatchConfig {
            max_batch_size: 10,
            max_total_tokens: 100,
            sort_by_length: false,
            ..BatchConfig::default()
        };
        let mut batcher = DynamicBatcher::new(config, BatchFormationStrategy::FirstFit);
        batcher.submit(req(1, 30, 30)); // budget=60
        batcher.submit(req(2, 50, 50)); // budget=100 — doesn't fit after first
        batcher.submit(req(3, 10, 10)); // budget=20 — fits after first
        let batch = batcher.try_form_batch().unwrap();
        let ids: Vec<u64> = batch.requests.iter().map(|r| r.id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
        assert!(!ids.contains(&2));
    }

    // -----------------------------------------------------------------------
    // DynamicBatcher — best-fit strategy
    // -----------------------------------------------------------------------

    #[test]
    fn best_fit_selects_tightest() {
        let config = BatchConfig {
            max_batch_size: 1,
            max_total_tokens: 50,
            sort_by_length: false,
            ..BatchConfig::default()
        };
        let mut batcher = DynamicBatcher::new(config, BatchFormationStrategy::BestFit);
        batcher.submit(req(1, 10, 10)); // budget=20
        batcher.submit(req(2, 20, 20)); // budget=40
        batcher.submit(req(3, 25, 24)); // budget=49 — closest to 50
        let batch = batcher.try_form_batch().unwrap();
        assert_eq!(batch.requests[0].id, 3);
    }

    #[test]
    fn best_fit_fills_multiple() {
        let config = BatchConfig {
            max_batch_size: 10,
            max_total_tokens: 100,
            sort_by_length: false,
            ..BatchConfig::default()
        };
        let mut batcher = DynamicBatcher::new(config, BatchFormationStrategy::BestFit);
        batcher.submit(req(1, 20, 20)); // budget=40
        batcher.submit(req(2, 30, 29)); // budget=59
        batcher.submit(req(3, 5, 5)); // budget=10
        let batch = batcher.try_form_batch().unwrap();
        // Best-fit: first picks budget=59 (closest to 100), then budget=40 (closest to 41), then budget=10 fits in 1
        assert!(batch.size() >= 2);
    }

    // -----------------------------------------------------------------------
    // DynamicBatcher — sorted-greedy strategy
    // -----------------------------------------------------------------------

    #[test]
    fn sorted_greedy_orders_by_length() {
        let config = BatchConfig {
            max_batch_size: 10,
            sort_by_length: false, // SortedGreedy overrides this
            ..BatchConfig::default()
        };
        let mut batcher = DynamicBatcher::new(config, BatchFormationStrategy::SortedGreedy);
        batcher.submit(req_with_tokens(1, vec![1; 20], 10));
        batcher.submit(req_with_tokens(2, vec![1; 5], 10));
        batcher.submit(req_with_tokens(3, vec![1; 10], 10));
        let batch = batcher.try_form_batch().unwrap();
        // Should be sorted by prompt length: 5, 10, 20
        assert_eq!(batch.requests[0].tokens.len(), 5);
        assert_eq!(batch.requests[1].tokens.len(), 10);
        assert_eq!(batch.requests[2].tokens.len(), 20);
    }

    // -----------------------------------------------------------------------
    // Padding in batches
    // -----------------------------------------------------------------------

    #[test]
    fn batch_padding_uniform_lengths() {
        let config = BatchConfig {
            max_batch_size: 10,
            padding_strategy: PaddingStrategy::PadToLongest,
            ..BatchConfig::default()
        };
        let mut batcher = DynamicBatcher::new(config, BatchFormationStrategy::Greedy);
        batcher.submit(req(1, 10, 5));
        batcher.submit(req(2, 10, 5));
        let batch = batcher.try_form_batch().unwrap();
        assert_eq!(batch.padding_tokens, 0);
        assert!((batch.efficiency - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn batch_padding_varying_lengths() {
        let config = BatchConfig {
            max_batch_size: 10,
            padding_strategy: PaddingStrategy::PadToLongest,
            sort_by_length: false,
            ..BatchConfig::default()
        };
        let mut batcher = DynamicBatcher::new(config, BatchFormationStrategy::Greedy);
        batcher.submit(req_with_tokens(1, vec![0; 5], 0));
        batcher.submit(req_with_tokens(2, vec![0; 10], 0));
        let batch = batcher.try_form_batch().unwrap();
        // padding = 10-5 = 5
        assert_eq!(batch.padding_tokens, 5);
        // efficiency = 15/20 = 0.75
        assert!((batch.efficiency - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn batch_no_padding_strategy() {
        let config = BatchConfig {
            max_batch_size: 10,
            padding_strategy: PaddingStrategy::NoPadding,
            ..BatchConfig::default()
        };
        let mut batcher = DynamicBatcher::new(config, BatchFormationStrategy::Greedy);
        batcher.submit(req_with_tokens(1, vec![0; 5], 0));
        batcher.submit(req_with_tokens(2, vec![0; 10], 0));
        let batch = batcher.try_form_batch().unwrap();
        assert_eq!(batch.padding_tokens, 0);
        assert!((batch.efficiency - 1.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn batcher_single_large_request_exceeds_limit() {
        let config = BatchConfig {
            max_batch_size: 10,
            max_total_tokens: 10,
            sort_by_length: false,
            ..BatchConfig::default()
        };
        let mut batcher = DynamicBatcher::new(config, BatchFormationStrategy::Greedy);
        batcher.submit(req(1, 100, 100)); // budget=200, exceeds limit=10
        // All candidates are too large → none selected → returns None
        assert!(batcher.try_form_batch().is_none());
        // Request should be returned to queue
        assert_eq!(batcher.pending_count(), 1);
    }

    #[test]
    fn batcher_max_batch_size_one() {
        let config = BatchConfig { max_batch_size: 1, ..BatchConfig::default() };
        let mut batcher = DynamicBatcher::new(config, BatchFormationStrategy::Greedy);
        batcher.submit(req(1, 5, 5));
        batcher.submit(req(2, 5, 5));
        let batch = batcher.try_form_batch().unwrap();
        assert_eq!(batch.size(), 1);
        assert_eq!(batcher.pending_count(), 1);
    }

    #[test]
    fn batcher_drain_all() {
        let mut batcher = default_batcher();
        for i in 0..5 {
            batcher.submit(req(i, 5, 5));
        }
        let batch = batcher.try_form_batch().unwrap();
        assert_eq!(batch.size(), 5);
        assert_eq!(batcher.pending_count(), 0);
        assert!(batcher.try_form_batch().is_none());
    }

    #[test]
    fn batcher_zero_length_tokens() {
        let mut batcher = default_batcher();
        batcher.submit(req(1, 0, 0));
        let batch = batcher.try_form_batch().unwrap();
        assert_eq!(batch.size(), 1);
        assert_eq!(batch.total_tokens, 0);
    }

    #[test]
    fn batcher_complete_unknown_batch() {
        let mut batcher = default_batcher();
        assert!(batcher.complete_batch(42).is_none());
    }

    // -----------------------------------------------------------------------
    // Priority ordering (manual sort verification)
    // -----------------------------------------------------------------------

    #[test]
    fn requests_sorted_by_priority() {
        let mut reqs =
            [req_with_priority(1, 10, 1), req_with_priority(2, 10, 5), req_with_priority(3, 10, 3)];
        reqs.sort_by(|a, b| b.priority.cmp(&a.priority));
        assert_eq!(reqs[0].id, 2); // priority 5
        assert_eq!(reqs[1].id, 3); // priority 3
        assert_eq!(reqs[2].id, 1); // priority 1
    }

    // -----------------------------------------------------------------------
    // Metrics accumulation across batches
    // -----------------------------------------------------------------------

    #[test]
    fn metrics_accumulate_over_batches() {
        let config = BatchConfig { max_batch_size: 2, ..BatchConfig::default() };
        let mut batcher = DynamicBatcher::new(config, BatchFormationStrategy::Greedy);
        for i in 0..6 {
            batcher.submit(req(i, 5, 5));
        }
        batcher.try_form_batch();
        batcher.try_form_batch();
        batcher.try_form_batch();
        assert_eq!(batcher.metrics().batches_formed, 3);
        assert_eq!(batcher.metrics().requests_processed, 6);
    }

    // -----------------------------------------------------------------------
    // Inflight entry timing
    // -----------------------------------------------------------------------

    #[test]
    fn inflight_entry_has_dispatch_time() {
        let mut tracker = InflightBatchTracker::new();
        let before = Instant::now();
        tracker.register(1, 10);
        let entry = tracker.complete(0).unwrap();
        assert!(entry.dispatched_at >= before);
    }

    // -----------------------------------------------------------------------
    // PaddingStrategy equality & debug
    // -----------------------------------------------------------------------

    #[test]
    fn padding_strategy_equality() {
        assert_eq!(PaddingStrategy::NoPadding, PaddingStrategy::NoPadding);
        assert_eq!(PaddingStrategy::PadToMax(10), PaddingStrategy::PadToMax(10));
        assert_ne!(PaddingStrategy::PadToMax(10), PaddingStrategy::PadToMax(20));
        assert_ne!(PaddingStrategy::NoPadding, PaddingStrategy::PadToLongest);
    }

    #[test]
    fn padding_strategy_debug() {
        let dbg = format!("{:?}", PaddingStrategy::PadToLongest);
        assert!(dbg.contains("PadToLongest"));
    }

    // -----------------------------------------------------------------------
    // proptest
    // -----------------------------------------------------------------------

    proptest::proptest! {
        #[test]
        fn batch_efficiency_bounded(useful in 0usize..10000, extra in 0usize..10000) {
            let total = useful + extra;
            let eff = BatchEfficiency::compute(useful, total);
            proptest::prop_assert!(eff >= 0.0);
            proptest::prop_assert!(eff <= 1.0);
        }

        #[test]
        fn padding_never_negative(
            a in 1usize..100,
            b in 1usize..100,
            c in 1usize..100,
        ) {
            let lengths = vec![a, b, c];
            let pad = PaddingCalculator::compute(&lengths, PaddingStrategy::PadToLongest);
            let useful: usize = lengths.iter().sum();
            let total = PaddingCalculator::total_with_padding(&lengths, PaddingStrategy::PadToLongest);
            proptest::prop_assert!(total >= useful);
            proptest::prop_assert_eq!(total, useful + pad);
        }

        #[test]
        fn greedy_respects_max_batch_size(
            n in 1usize..20,
            max_bs in 1usize..10,
        ) {
            let config = BatchConfig {
                max_batch_size: max_bs,
                max_total_tokens: 1_000_000,
                ..BatchConfig::default()
            };
            let mut batcher = DynamicBatcher::new(config, BatchFormationStrategy::Greedy);
            for i in 0..n {
                batcher.submit(req(i as u64, 5, 5));
            }
            if let Some(batch) = batcher.try_form_batch() {
                proptest::prop_assert!(batch.size() <= max_bs);
            }
        }

        #[test]
        fn total_token_budget_consistent(prompt in 0usize..500, max_gen in 0usize..500) {
            let r = req(0, prompt, max_gen);
            proptest::prop_assert_eq!(r.total_token_budget(), prompt + max_gen);
        }
    }
}
