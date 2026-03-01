//! Batch inference engine for processing multiple requests concurrently.
//!
//! Provides dynamic batching, priority scheduling, and multiple padding
//! strategies for efficient GPU inference across concurrent requests.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]

// ── Types ────────────────────────────────────────────────────────────────────

/// Unique identifier for an inference request.
pub type RequestId = u64;

/// Configuration for the batch inference engine.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of requests in a single batch.
    pub max_batch_size: usize,
    /// Maximum allowed sequence length (tokens).
    pub max_sequence_length: usize,
    /// Enable dynamic batch sizing based on sequence lengths.
    pub dynamic_batching: bool,
    /// How sequences are padded within a batch.
    pub padding_strategy: PaddingStrategy,
    /// Timeout in milliseconds before a pending request is discarded.
    pub timeout_ms: u64,
    /// Whether to schedule higher-priority requests first.
    pub priority_scheduling: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_sequence_length: 2048,
            dynamic_batching: false,
            padding_strategy: PaddingStrategy::PadToLongest,
            timeout_ms: 30_000,
            priority_scheduling: false,
        }
    }
}

/// Strategy used to pad sequences within a batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PaddingStrategy {
    /// Pad every sequence to `max_sequence_length`.
    PadToMax,
    /// Pad every sequence to the longest sequence in the batch.
    PadToLongest,
    /// No padding — only same-length sequences are batched together.
    NoPadding,
    /// Pad to the nearest bucket boundary.
    BucketedPadding { buckets: Vec<usize> },
}

/// An inference request submitted to the engine.
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub id: RequestId,
    pub prompt_tokens: Vec<u32>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub priority: RequestPriority,
    pub submitted_at: u64,
}

/// Priority level for a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RequestPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Realtime = 3,
}

/// A batch of requests currently being processed.
#[derive(Debug, Clone)]
pub struct ActiveBatch {
    pub requests: Vec<RequestId>,
    pub batch_size: usize,
    pub padded_length: usize,
    pub started_at: u64,
}

/// Result of a completed inference request.
#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub request_id: RequestId,
    pub output_tokens: Vec<u32>,
    pub time_to_first_token_ms: f64,
    pub total_time_ms: f64,
    pub tokens_per_second: f64,
    pub finish_reason: FinishReason,
}

/// Reason an inference request completed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    MaxTokens,
    EosToken,
    Timeout,
    Error(String),
}

/// Aggregate statistics for the batch engine.
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    pub total_requests: u64,
    pub total_batches: u64,
    pub avg_batch_size: f64,
    pub avg_padding_waste: f64,
    pub avg_wait_time_ms: f64,
    pub throughput_tokens_per_sec: f64,
    pub p50_latency_ms: f64,
    pub p99_latency_ms: f64,
}

// ── Engine ───────────────────────────────────────────────────────────────────

/// Batch inference engine that collects requests, forms batches, and
/// tracks completion statistics.
pub struct BatchEngine {
    config: BatchConfig,
    pending: Vec<InferenceRequest>,
    active_batch: Option<ActiveBatch>,
    completed: Vec<InferenceResult>,
    stats: BatchStats,
    next_id: RequestId,
    all_latencies_ms: Vec<f64>,
    total_output_tokens: u64,
    total_padding_waste: f64,
    total_wait_time_ms: f64,
    wait_count: u64,
}

impl BatchEngine {
    /// Create a new engine with the given configuration.
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config,
            pending: Vec::new(),
            active_batch: None,
            completed: Vec::new(),
            stats: BatchStats::default(),
            next_id: 1,
            all_latencies_ms: Vec::new(),
            total_output_tokens: 0,
            total_padding_waste: 0.0,
            total_wait_time_ms: 0.0,
            wait_count: 0,
        }
    }

    /// Submit a request and return its assigned `RequestId`.
    pub fn submit(&mut self, mut request: InferenceRequest) -> RequestId {
        let id = self.next_id;
        self.next_id += 1;
        request.id = id;
        self.pending.push(request);
        id
    }

    /// Number of pending (not yet batched) requests.
    pub const fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Current aggregate statistics.
    pub const fn stats(&self) -> &BatchStats {
        &self.stats
    }

    /// Drain and return all completed results.
    pub fn take_completed(&mut self) -> Vec<InferenceResult> {
        std::mem::take(&mut self.completed)
    }

    /// Reference to the currently active batch, if any.
    pub const fn active_batch(&self) -> Option<&ActiveBatch> {
        self.active_batch.as_ref()
    }

    // ── Timeout handling ─────────────────────────────────────────────────

    /// Remove pending requests older than `timeout_ms` and return them as
    /// timed-out results.
    pub fn expire_stale(&mut self, now_ms: u64) {
        let timeout = self.config.timeout_ms;
        let (stale, fresh): (Vec<_>, Vec<_>) =
            self.pending.drain(..).partition(|r| now_ms.saturating_sub(r.submitted_at) >= timeout);

        self.pending = fresh;

        for req in stale {
            self.completed.push(InferenceResult {
                request_id: req.id,
                output_tokens: Vec::new(),
                time_to_first_token_ms: 0.0,
                total_time_ms: now_ms.saturating_sub(req.submitted_at) as f64,
                tokens_per_second: 0.0,
                finish_reason: FinishReason::Timeout,
            });
        }
    }

    // ── Batch formation ──────────────────────────────────────────────────

    /// Select pending requests and form the next batch.
    ///
    /// Returns `None` when the pending queue is empty.
    pub fn form_batch(&mut self) -> Option<ActiveBatch> {
        if self.pending.is_empty() {
            return None;
        }

        // Sort by priority (descending) if enabled.
        if self.config.priority_scheduling {
            self.pending.sort_by(|a, b| b.priority.cmp(&a.priority));
        }

        let selected = self.select_for_batch();
        if selected.is_empty() {
            return None;
        }

        let padded_length = self.compute_padded_length(&selected);
        let ids: Vec<RequestId> = selected.iter().map(|r| r.id).collect();
        let batch_size = ids.len();

        // Record wait times.
        let now_ms = selected.iter().map(|r| r.submitted_at).max().unwrap_or(0);
        for req in &selected {
            let wait = now_ms.saturating_sub(req.submitted_at) as f64;
            self.total_wait_time_ms += wait;
            self.wait_count += 1;
        }

        // Track padding waste.
        let total_tokens: usize = selected.iter().map(|r| r.prompt_tokens.len()).sum();
        let padded_total = batch_size * padded_length;
        if padded_total > 0 {
            let waste = (padded_total - total_tokens) as f64 / padded_total as f64;
            self.total_padding_waste += waste;
        }

        // Remove selected requests from pending.
        let selected_ids: Vec<RequestId> = selected.iter().map(|r| r.id).collect();
        self.pending.retain(|r| !selected_ids.contains(&r.id));

        let batch = ActiveBatch { requests: ids, batch_size, padded_length, started_at: now_ms };
        self.active_batch = Some(batch.clone());
        Some(batch)
    }

    /// Mark the active batch as complete, absorb results, and update stats.
    pub fn complete_batch(&mut self, results: Vec<InferenceResult>) {
        for result in &results {
            self.all_latencies_ms.push(result.total_time_ms);
            self.total_output_tokens += result.output_tokens.len() as u64;
        }

        self.stats.total_batches += 1;
        self.stats.total_requests += results.len() as u64;

        // Avg batch size.
        self.stats.avg_batch_size =
            self.stats.total_requests as f64 / self.stats.total_batches as f64;

        // Avg padding waste.
        if self.stats.total_batches > 0 {
            self.stats.avg_padding_waste =
                self.total_padding_waste / self.stats.total_batches as f64;
        }

        // Avg wait time.
        if self.wait_count > 0 {
            self.stats.avg_wait_time_ms = self.total_wait_time_ms / self.wait_count as f64;
        }

        // Throughput: total output tokens / total elapsed time (s).
        if !self.all_latencies_ms.is_empty() {
            let total_time_s: f64 = self.all_latencies_ms.iter().sum::<f64>() / 1000.0;
            if total_time_s > 0.0 {
                self.stats.throughput_tokens_per_sec =
                    self.total_output_tokens as f64 / total_time_s;
            }
        }

        // Latency percentiles.
        self.stats.p50_latency_ms = percentile(&self.all_latencies_ms, 50.0);
        self.stats.p99_latency_ms = percentile(&self.all_latencies_ms, 99.0);

        self.completed.extend(results);
        self.active_batch = None;
    }

    // ── Internal helpers ─────────────────────────────────────────────────

    /// Choose which pending requests go into the next batch.
    fn select_for_batch(&self) -> Vec<InferenceRequest> {
        let limit = if self.config.dynamic_batching {
            self.dynamic_batch_limit()
        } else {
            self.config.max_batch_size
        };

        match &self.config.padding_strategy {
            PaddingStrategy::NoPadding => self.select_same_length(limit),
            _ => self.pending.iter().take(limit).cloned().collect(),
        }
    }

    /// For `NoPadding`, find the biggest group of same-length sequences
    /// (up to `limit`).
    fn select_same_length(&self, limit: usize) -> Vec<InferenceRequest> {
        let mut groups: std::collections::HashMap<usize, Vec<&InferenceRequest>> =
            std::collections::HashMap::new();
        for req in &self.pending {
            groups.entry(req.prompt_tokens.len()).or_default().push(req);
        }

        groups
            .into_values()
            .max_by_key(Vec::len)
            .unwrap_or_default()
            .into_iter()
            .take(limit)
            .cloned()
            .collect()
    }

    /// Heuristic: reduce batch size when average sequence length is large.
    fn dynamic_batch_limit(&self) -> usize {
        if self.pending.is_empty() {
            return self.config.max_batch_size;
        }
        let avg_len: usize =
            self.pending.iter().map(|r| r.prompt_tokens.len()).sum::<usize>() / self.pending.len();

        let max = self.config.max_batch_size;
        if avg_len == 0 {
            return max;
        }
        // Scale inversely with average length relative to max_sequence_length.
        let ratio = self.config.max_sequence_length as f64 / avg_len as f64;
        (ratio as usize).min(max).max(1)
    }

    /// Compute the padded sequence length for a candidate batch.
    fn compute_padded_length(&self, requests: &[InferenceRequest]) -> usize {
        let longest = requests.iter().map(|r| r.prompt_tokens.len()).max().unwrap_or(0);

        match &self.config.padding_strategy {
            PaddingStrategy::PadToMax => self.config.max_sequence_length,
            PaddingStrategy::PadToLongest | PaddingStrategy::NoPadding => longest,
            PaddingStrategy::BucketedPadding { buckets } => find_bucket(buckets, longest),
        }
    }
}

/// Compute a percentile from an unsorted slice using nearest-rank.
fn percentile(values: &[f64], pct: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((pct / 100.0) * sorted.len() as f64).ceil() as usize;
    let idx = idx.min(sorted.len()).saturating_sub(1);
    sorted[idx]
}

/// Find the smallest bucket ≥ `len`. Falls back to `len` itself.
fn find_bucket(buckets: &[usize], len: usize) -> usize {
    let mut sorted = buckets.to_vec();
    sorted.sort_unstable();
    sorted.into_iter().find(|&b| b >= len).unwrap_or(len)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::cast_sign_loss, clippy::cast_precision_loss)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────

    fn default_config() -> BatchConfig {
        BatchConfig::default()
    }

    fn make_request(tokens: &[u32]) -> InferenceRequest {
        InferenceRequest {
            id: 0,
            prompt_tokens: tokens.to_vec(),
            max_tokens: 64,
            temperature: 1.0,
            priority: RequestPriority::Normal,
            submitted_at: 0,
        }
    }

    fn make_request_with_priority(tokens: &[u32], priority: RequestPriority) -> InferenceRequest {
        InferenceRequest {
            id: 0,
            prompt_tokens: tokens.to_vec(),
            max_tokens: 64,
            temperature: 1.0,
            priority,
            submitted_at: 0,
        }
    }

    fn make_request_at(tokens: &[u32], submitted_at: u64) -> InferenceRequest {
        InferenceRequest {
            id: 0,
            prompt_tokens: tokens.to_vec(),
            max_tokens: 64,
            temperature: 1.0,
            priority: RequestPriority::Normal,
            submitted_at,
        }
    }

    fn make_result(
        request_id: RequestId,
        output_tokens: Vec<u32>,
        total_time_ms: f64,
    ) -> InferenceResult {
        let tps = if total_time_ms > 0.0 {
            output_tokens.len() as f64 / (total_time_ms / 1000.0)
        } else {
            0.0
        };
        InferenceResult {
            request_id,
            output_tokens,
            time_to_first_token_ms: total_time_ms * 0.1,
            total_time_ms,
            tokens_per_second: tps,
            finish_reason: FinishReason::MaxTokens,
        }
    }

    // ── Basic operations ─────────────────────────────────────────────────

    #[test]
    fn new_engine_has_no_pending() {
        let engine = BatchEngine::new(default_config());
        assert_eq!(engine.pending_count(), 0);
    }

    #[test]
    fn new_engine_has_no_active_batch() {
        let engine = BatchEngine::new(default_config());
        assert!(engine.active_batch().is_none());
    }

    #[test]
    fn new_engine_stats_are_zero() {
        let engine = BatchEngine::new(default_config());
        let s = engine.stats();
        assert_eq!(s.total_requests, 0);
        assert_eq!(s.total_batches, 0);
    }

    #[test]
    fn submit_returns_unique_ids() {
        let mut engine = BatchEngine::new(default_config());
        let id1 = engine.submit(make_request(&[1, 2, 3]));
        let id2 = engine.submit(make_request(&[4, 5]));
        assert_ne!(id1, id2);
    }

    #[test]
    fn submit_increments_pending_count() {
        let mut engine = BatchEngine::new(default_config());
        engine.submit(make_request(&[1]));
        assert_eq!(engine.pending_count(), 1);
        engine.submit(make_request(&[2]));
        assert_eq!(engine.pending_count(), 2);
    }

    #[test]
    fn take_completed_on_empty_returns_empty() {
        let mut engine = BatchEngine::new(default_config());
        assert!(engine.take_completed().is_empty());
    }

    #[test]
    fn form_batch_on_empty_returns_none() {
        let mut engine = BatchEngine::new(default_config());
        assert!(engine.form_batch().is_none());
    }

    // ── Single request → batch of 1 ─────────────────────────────────────

    #[test]
    fn single_request_forms_batch_of_one() {
        let mut engine = BatchEngine::new(default_config());
        engine.submit(make_request(&[1, 2, 3]));
        let batch = engine.form_batch().unwrap();
        assert_eq!(batch.batch_size, 1);
        assert_eq!(batch.requests.len(), 1);
    }

    #[test]
    fn single_request_batch_padded_length() {
        let mut engine = BatchEngine::new(default_config());
        engine.submit(make_request(&[1, 2, 3]));
        let batch = engine.form_batch().unwrap();
        // PadToLongest → length of the only request
        assert_eq!(batch.padded_length, 3);
    }

    #[test]
    fn single_request_clears_pending() {
        let mut engine = BatchEngine::new(default_config());
        engine.submit(make_request(&[1]));
        engine.form_batch();
        assert_eq!(engine.pending_count(), 0);
    }

    // ── Multiple requests batched together ───────────────────────────────

    #[test]
    fn multiple_requests_form_single_batch() {
        let mut engine = BatchEngine::new(default_config());
        engine.submit(make_request(&[1, 2]));
        engine.submit(make_request(&[3, 4, 5]));
        engine.submit(make_request(&[6]));
        let batch = engine.form_batch().unwrap();
        assert_eq!(batch.batch_size, 3);
    }

    #[test]
    fn multiple_requests_padded_to_longest() {
        let mut engine = BatchEngine::new(default_config());
        engine.submit(make_request(&[1]));
        engine.submit(make_request(&[1, 2, 3, 4, 5]));
        let batch = engine.form_batch().unwrap();
        assert_eq!(batch.padded_length, 5);
    }

    // ── Max batch size enforced ──────────────────────────────────────────

    #[test]
    fn max_batch_size_limits_batch() {
        let config = BatchConfig { max_batch_size: 2, ..default_config() };
        let mut engine = BatchEngine::new(config);
        engine.submit(make_request(&[1]));
        engine.submit(make_request(&[2]));
        engine.submit(make_request(&[3]));
        let batch = engine.form_batch().unwrap();
        assert_eq!(batch.batch_size, 2);
        assert_eq!(engine.pending_count(), 1);
    }

    #[test]
    fn remaining_requests_stay_pending() {
        let config = BatchConfig { max_batch_size: 1, ..default_config() };
        let mut engine = BatchEngine::new(config);
        engine.submit(make_request(&[1]));
        engine.submit(make_request(&[2]));
        engine.form_batch();
        assert_eq!(engine.pending_count(), 1);
    }

    #[test]
    fn second_batch_picks_up_remaining() {
        let config = BatchConfig { max_batch_size: 1, ..default_config() };
        let mut engine = BatchEngine::new(config);
        engine.submit(make_request(&[1]));
        engine.submit(make_request(&[2]));
        let b1 = engine.form_batch().unwrap();
        engine.complete_batch(vec![make_result(b1.requests[0], vec![10], 50.0)]);
        let b2 = engine.form_batch().unwrap();
        assert_eq!(b2.batch_size, 1);
        assert_eq!(engine.pending_count(), 0);
    }

    // ── Priority scheduling ──────────────────────────────────────────────

    #[test]
    fn priority_scheduling_realtime_first() {
        let config =
            BatchConfig { max_batch_size: 2, priority_scheduling: true, ..default_config() };
        let mut engine = BatchEngine::new(config);
        let low_id = engine.submit(make_request_with_priority(&[1], RequestPriority::Low));
        let rt_id = engine.submit(make_request_with_priority(&[2], RequestPriority::Realtime));
        let _normal_id = engine.submit(make_request_with_priority(&[3], RequestPriority::Normal));
        let batch = engine.form_batch().unwrap();
        assert_eq!(batch.requests[0], rt_id);
        assert!(!batch.requests.contains(&low_id));
    }

    #[test]
    fn priority_scheduling_high_before_normal() {
        let config =
            BatchConfig { max_batch_size: 1, priority_scheduling: true, ..default_config() };
        let mut engine = BatchEngine::new(config);
        engine.submit(make_request_with_priority(&[1], RequestPriority::Normal));
        let high_id = engine.submit(make_request_with_priority(&[2], RequestPriority::High));
        let batch = engine.form_batch().unwrap();
        assert_eq!(batch.requests[0], high_id);
    }

    #[test]
    fn priority_scheduling_all_levels_ordered() {
        let config =
            BatchConfig { max_batch_size: 4, priority_scheduling: true, ..default_config() };
        let mut engine = BatchEngine::new(config);
        let low = engine.submit(make_request_with_priority(&[1], RequestPriority::Low));
        let normal = engine.submit(make_request_with_priority(&[2], RequestPriority::Normal));
        let high = engine.submit(make_request_with_priority(&[3], RequestPriority::High));
        let rt = engine.submit(make_request_with_priority(&[4], RequestPriority::Realtime));
        let batch = engine.form_batch().unwrap();
        assert_eq!(batch.requests, vec![rt, high, normal, low]);
    }

    #[test]
    fn no_priority_scheduling_preserves_order() {
        let config =
            BatchConfig { max_batch_size: 4, priority_scheduling: false, ..default_config() };
        let mut engine = BatchEngine::new(config);
        let id1 = engine.submit(make_request_with_priority(&[1], RequestPriority::Low));
        let id2 = engine.submit(make_request_with_priority(&[2], RequestPriority::Realtime));
        let batch = engine.form_batch().unwrap();
        assert_eq!(batch.requests[0], id1);
        assert_eq!(batch.requests[1], id2);
    }

    #[test]
    fn mixed_priorities_batch_formation() {
        let config =
            BatchConfig { max_batch_size: 3, priority_scheduling: true, ..default_config() };
        let mut engine = BatchEngine::new(config);
        engine.submit(make_request_with_priority(&[1], RequestPriority::Normal));
        engine.submit(make_request_with_priority(&[2], RequestPriority::Low));
        let high_id = engine.submit(make_request_with_priority(&[3], RequestPriority::High));
        engine.submit(make_request_with_priority(&[4], RequestPriority::Low));
        let batch = engine.form_batch().unwrap();
        assert_eq!(batch.batch_size, 3);
        assert_eq!(batch.requests[0], high_id);
    }

    // ── PadToMax ─────────────────────────────────────────────────────────

    #[test]
    fn pad_to_max_uses_max_sequence_length() {
        let config = BatchConfig {
            max_sequence_length: 512,
            padding_strategy: PaddingStrategy::PadToMax,
            ..default_config()
        };
        let mut engine = BatchEngine::new(config);
        engine.submit(make_request(&[1, 2]));
        let batch = engine.form_batch().unwrap();
        assert_eq!(batch.padded_length, 512);
    }

    #[test]
    fn pad_to_max_independent_of_actual_lengths() {
        let config = BatchConfig {
            max_sequence_length: 1024,
            padding_strategy: PaddingStrategy::PadToMax,
            ..default_config()
        };
        let mut engine = BatchEngine::new(config);
        engine.submit(make_request(&[1]));
        engine.submit(make_request(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
        let batch = engine.form_batch().unwrap();
        assert_eq!(batch.padded_length, 1024);
    }

    // ── PadToLongest ─────────────────────────────────────────────────────

    #[test]
    fn pad_to_longest_uses_longest_sequence() {
        let config =
            BatchConfig { padding_strategy: PaddingStrategy::PadToLongest, ..default_config() };
        let mut engine = BatchEngine::new(config);
        engine.submit(make_request(&[1, 2]));
        engine.submit(make_request(&[1, 2, 3, 4, 5, 6, 7]));
        engine.submit(make_request(&[1]));
        let batch = engine.form_batch().unwrap();
        assert_eq!(batch.padded_length, 7);
    }

    #[test]
    fn pad_to_longest_single_request() {
        let config =
            BatchConfig { padding_strategy: PaddingStrategy::PadToLongest, ..default_config() };
        let mut engine = BatchEngine::new(config);
        engine.submit(make_request(&[1, 2, 3]));
        let batch = engine.form_batch().unwrap();
        assert_eq!(batch.padded_length, 3);
    }

    // ── BucketedPadding ──────────────────────────────────────────────────

    #[test]
    fn bucketed_padding_selects_correct_bucket() {
        let config = BatchConfig {
            padding_strategy: PaddingStrategy::BucketedPadding { buckets: vec![64, 128, 256, 512] },
            ..default_config()
        };
        let mut engine = BatchEngine::new(config);
        // Longest is 5 → should select bucket 64
        engine.submit(make_request(&[1, 2, 3, 4, 5]));
        let batch = engine.form_batch().unwrap();
        assert_eq!(batch.padded_length, 64);
    }

    #[test]
    fn bucketed_padding_exact_match() {
        let config = BatchConfig {
            padding_strategy: PaddingStrategy::BucketedPadding { buckets: vec![4, 8, 16] },
            ..default_config()
        };
        let mut engine = BatchEngine::new(config);
        engine.submit(make_request(&[1, 2, 3, 4]));
        let batch = engine.form_batch().unwrap();
        assert_eq!(batch.padded_length, 4);
    }

    #[test]
    fn bucketed_padding_falls_back_to_length_when_too_large() {
        let config = BatchConfig {
            padding_strategy: PaddingStrategy::BucketedPadding { buckets: vec![4, 8] },
            ..default_config()
        };
        let mut engine = BatchEngine::new(config);
        engine.submit(make_request(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
        let batch = engine.form_batch().unwrap();
        // No bucket ≥ 10, falls back to 10
        assert_eq!(batch.padded_length, 10);
    }

    #[test]
    fn bucketed_padding_unsorted_buckets() {
        let config = BatchConfig {
            padding_strategy: PaddingStrategy::BucketedPadding { buckets: vec![256, 64, 128] },
            ..default_config()
        };
        let mut engine = BatchEngine::new(config);
        engine.submit(make_request(&[1; 100]));
        let batch = engine.form_batch().unwrap();
        assert_eq!(batch.padded_length, 128);
    }

    #[test]
    fn bucketed_padding_multiple_requests_uses_longest() {
        let config = BatchConfig {
            padding_strategy: PaddingStrategy::BucketedPadding { buckets: vec![8, 16, 32] },
            ..default_config()
        };
        let mut engine = BatchEngine::new(config);
        engine.submit(make_request(&[1, 2]));
        engine.submit(make_request(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
        let batch = engine.form_batch().unwrap();
        assert_eq!(batch.padded_length, 16);
    }

    // ── NoPadding ────────────────────────────────────────────────────────

    #[test]
    fn no_padding_groups_same_length() {
        let config = BatchConfig {
            padding_strategy: PaddingStrategy::NoPadding,
            max_batch_size: 10,
            ..default_config()
        };
        let mut engine = BatchEngine::new(config);
        engine.submit(make_request(&[1, 2, 3])); // len 3
        engine.submit(make_request(&[4, 5])); // len 2
        engine.submit(make_request(&[6, 7, 8])); // len 3
        engine.submit(make_request(&[9, 10])); // len 2
        engine.submit(make_request(&[11, 12, 13])); // len 3
        let batch = engine.form_batch().unwrap();
        // Largest group is length-3 with 3 members
        assert_eq!(batch.batch_size, 3);
        assert_eq!(batch.padded_length, 3);
    }

    #[test]
    fn no_padding_all_same_length() {
        let config = BatchConfig {
            padding_strategy: PaddingStrategy::NoPadding,
            max_batch_size: 10,
            ..default_config()
        };
        let mut engine = BatchEngine::new(config);
        engine.submit(make_request(&[1, 2]));
        engine.submit(make_request(&[3, 4]));
        engine.submit(make_request(&[5, 6]));
        let batch = engine.form_batch().unwrap();
        assert_eq!(batch.batch_size, 3);
    }

    #[test]
    fn no_padding_respects_max_batch_size() {
        let config = BatchConfig {
            padding_strategy: PaddingStrategy::NoPadding,
            max_batch_size: 2,
            ..default_config()
        };
        let mut engine = BatchEngine::new(config);
        engine.submit(make_request(&[1, 2]));
        engine.submit(make_request(&[3, 4]));
        engine.submit(make_request(&[5, 6]));
        let batch = engine.form_batch().unwrap();
        assert_eq!(batch.batch_size, 2);
    }

    // ── Timeout / stale requests ─────────────────────────────────────────

    #[test]
    fn expire_stale_removes_old_requests() {
        let config = BatchConfig { timeout_ms: 100, ..default_config() };
        let mut engine = BatchEngine::new(config);
        engine.submit(make_request_at(&[1], 0));
        engine.submit(make_request_at(&[2], 50));
        engine.expire_stale(150);
        // First request (submitted at 0) is 150ms old → expired.
        // Second request (submitted at 50) is 100ms old → also expired.
        assert_eq!(engine.pending_count(), 0);
    }

    #[test]
    fn expire_stale_keeps_fresh_requests() {
        let config = BatchConfig { timeout_ms: 100, ..default_config() };
        let mut engine = BatchEngine::new(config);
        engine.submit(make_request_at(&[1], 0));
        engine.submit(make_request_at(&[2], 80));
        engine.expire_stale(99);
        // At t=99: first (99ms old) < 100 → kept; second (19ms old) → kept.
        assert_eq!(engine.pending_count(), 2);
    }

    #[test]
    fn expire_stale_produces_timeout_results() {
        let config = BatchConfig { timeout_ms: 50, ..default_config() };
        let mut engine = BatchEngine::new(config);
        engine.submit(make_request_at(&[1], 0));
        engine.expire_stale(100);
        let results = engine.take_completed();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].finish_reason, FinishReason::Timeout);
    }

    #[test]
    fn expire_stale_with_no_stale_is_noop() {
        let mut engine = BatchEngine::new(default_config());
        engine.submit(make_request_at(&[1], 100));
        engine.expire_stale(100);
        assert_eq!(engine.pending_count(), 1);
    }

    #[test]
    fn expire_stale_result_has_correct_time() {
        let config = BatchConfig { timeout_ms: 10, ..default_config() };
        let mut engine = BatchEngine::new(config);
        engine.submit(make_request_at(&[1], 5));
        engine.expire_stale(100);
        let results = engine.take_completed();
        assert_eq!(results[0].total_time_ms, 95.0);
    }

    // ── Dynamic batching ─────────────────────────────────────────────────

    #[test]
    fn dynamic_batching_reduces_size_for_long_sequences() {
        let config = BatchConfig {
            max_batch_size: 16,
            max_sequence_length: 1024,
            dynamic_batching: true,
            ..default_config()
        };
        let mut engine = BatchEngine::new(config);
        // All sequences are 512 tokens → ratio = 1024/512 = 2
        for _ in 0..16 {
            engine.submit(make_request(&[0; 512]));
        }
        let batch = engine.form_batch().unwrap();
        assert!(batch.batch_size <= 2);
    }

    #[test]
    fn dynamic_batching_allows_full_size_for_short_sequences() {
        let config = BatchConfig {
            max_batch_size: 8,
            max_sequence_length: 1024,
            dynamic_batching: true,
            ..default_config()
        };
        let mut engine = BatchEngine::new(config);
        // All sequences are 1 token → ratio = 1024/1 = 1024, clamped to 8
        for _ in 0..8 {
            engine.submit(make_request(&[1]));
        }
        let batch = engine.form_batch().unwrap();
        assert_eq!(batch.batch_size, 8);
    }

    #[test]
    fn dynamic_batching_never_zero() {
        let config = BatchConfig {
            max_batch_size: 4,
            max_sequence_length: 4,
            dynamic_batching: true,
            ..default_config()
        };
        let mut engine = BatchEngine::new(config);
        // Sequences exactly at max length → ratio = 1.0 → limit = 1
        for _ in 0..4 {
            engine.submit(make_request(&[0; 4]));
        }
        let batch = engine.form_batch().unwrap();
        assert!(batch.batch_size >= 1);
    }

    // ── Stats tracking ───────────────────────────────────────────────────

    #[test]
    fn complete_batch_updates_total_requests() {
        let mut engine = BatchEngine::new(default_config());
        engine.submit(make_request(&[1]));
        engine.submit(make_request(&[2]));
        let batch = engine.form_batch().unwrap();
        let results =
            batch.requests.iter().map(|&id| make_result(id, vec![10, 20], 100.0)).collect();
        engine.complete_batch(results);
        assert_eq!(engine.stats().total_requests, 2);
    }

    #[test]
    fn complete_batch_updates_total_batches() {
        let mut engine = BatchEngine::new(default_config());
        engine.submit(make_request(&[1]));
        let batch = engine.form_batch().unwrap();
        engine.complete_batch(vec![make_result(batch.requests[0], vec![10], 50.0)]);
        assert_eq!(engine.stats().total_batches, 1);
    }

    #[test]
    fn complete_batch_clears_active_batch() {
        let mut engine = BatchEngine::new(default_config());
        engine.submit(make_request(&[1]));
        let batch = engine.form_batch().unwrap();
        assert!(engine.active_batch().is_some());
        engine.complete_batch(vec![make_result(batch.requests[0], vec![10], 50.0)]);
        assert!(engine.active_batch().is_none());
    }

    #[test]
    fn avg_batch_size_computed_correctly() {
        let config = BatchConfig { max_batch_size: 2, ..default_config() };
        let mut engine = BatchEngine::new(config);

        // Batch 1: 2 requests
        engine.submit(make_request(&[1]));
        engine.submit(make_request(&[2]));
        let b = engine.form_batch().unwrap();
        let results: Vec<_> =
            b.requests.iter().map(|&id| make_result(id, vec![10], 50.0)).collect();
        engine.complete_batch(results);

        // Batch 2: 1 request
        engine.submit(make_request(&[3]));
        let b = engine.form_batch().unwrap();
        engine.complete_batch(vec![make_result(b.requests[0], vec![10], 50.0)]);

        // Total 3 requests / 2 batches = 1.5
        assert!((engine.stats().avg_batch_size - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_after_multiple_batches() {
        let mut engine = BatchEngine::new(default_config());
        for i in 0..3 {
            engine.submit(make_request(&[i as u32]));
            let b = engine.form_batch().unwrap();
            engine.complete_batch(vec![make_result(b.requests[0], vec![10, 20], 100.0)]);
        }
        assert_eq!(engine.stats().total_batches, 3);
        assert_eq!(engine.stats().total_requests, 3);
    }

    // ── Throughput ────────────────────────────────────────────────────────

    #[test]
    fn throughput_calculated_from_output_tokens() {
        let mut engine = BatchEngine::new(default_config());
        engine.submit(make_request(&[1]));
        let b = engine.form_batch().unwrap();
        // 10 output tokens in 1000ms = 10 tok/s
        engine.complete_batch(vec![make_result(b.requests[0], vec![0; 10], 1000.0)]);
        assert!((engine.stats().throughput_tokens_per_sec - 10.0).abs() < 0.01);
    }

    #[test]
    fn throughput_accumulates_across_batches() {
        let mut engine = BatchEngine::new(default_config());

        engine.submit(make_request(&[1]));
        let b = engine.form_batch().unwrap();
        engine.complete_batch(vec![make_result(b.requests[0], vec![0; 10], 1000.0)]);

        engine.submit(make_request(&[2]));
        let b = engine.form_batch().unwrap();
        engine.complete_batch(vec![make_result(b.requests[0], vec![0; 10], 1000.0)]);

        // 20 tokens / 2.0s = 10 tok/s
        assert!((engine.stats().throughput_tokens_per_sec - 10.0).abs() < 0.01);
    }

    // ── Latency percentiles ──────────────────────────────────────────────

    #[test]
    fn p50_latency_single_result() {
        let mut engine = BatchEngine::new(default_config());
        engine.submit(make_request(&[1]));
        let b = engine.form_batch().unwrap();
        engine.complete_batch(vec![make_result(b.requests[0], vec![10], 42.0)]);
        assert!((engine.stats().p50_latency_ms - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn p99_latency_identifies_tail() {
        let config = BatchConfig { max_batch_size: 100, ..default_config() };
        let mut engine = BatchEngine::new(config);
        // Submit 100 requests with latencies 1..=100
        for i in 1..=100u32 {
            engine.submit(make_request(&[i]));
        }
        let b = engine.form_batch().unwrap();
        let results: Vec<_> = b
            .requests
            .iter()
            .enumerate()
            .map(|(i, &id)| make_result(id, vec![0], i as f64 + 1.0))
            .collect();
        engine.complete_batch(results);
        // p99 should be 100.0 (the highest)
        assert!(engine.stats().p99_latency_ms >= 99.0);
    }

    #[test]
    fn p50_latency_with_even_count() {
        let mut engine = BatchEngine::new(default_config());
        // 4 requests: latencies 10, 20, 30, 40
        for i in 0..4 {
            engine.submit(make_request(&[i as u32]));
        }
        let b = engine.form_batch().unwrap();
        let results: Vec<_> = b
            .requests
            .iter()
            .enumerate()
            .map(|(i, &id)| make_result(id, vec![0], (i + 1) as f64 * 10.0))
            .collect();
        engine.complete_batch(results);
        // Nearest-rank p50 of [10,20,30,40]: ceil(0.5*4)=2 → idx 1 → 20.0
        assert!((engine.stats().p50_latency_ms - 20.0).abs() < f64::EPSILON);
    }

    // ── Take completed / drain ───────────────────────────────────────────

    #[test]
    fn take_completed_returns_results() {
        let mut engine = BatchEngine::new(default_config());
        engine.submit(make_request(&[1]));
        let b = engine.form_batch().unwrap();
        engine.complete_batch(vec![make_result(b.requests[0], vec![10], 50.0)]);
        let results = engine.take_completed();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].output_tokens, vec![10]);
    }

    #[test]
    fn take_completed_drains_buffer() {
        let mut engine = BatchEngine::new(default_config());
        engine.submit(make_request(&[1]));
        let b = engine.form_batch().unwrap();
        engine.complete_batch(vec![make_result(b.requests[0], vec![10], 50.0)]);
        engine.take_completed();
        assert!(engine.take_completed().is_empty());
    }

    #[test]
    fn take_completed_preserves_request_id() {
        let mut engine = BatchEngine::new(default_config());
        let id = engine.submit(make_request(&[1]));
        let b = engine.form_batch().unwrap();
        engine.complete_batch(vec![make_result(b.requests[0], vec![10], 50.0)]);
        let results = engine.take_completed();
        assert_eq!(results[0].request_id, id);
    }

    // ── Padding waste ────────────────────────────────────────────────────

    #[test]
    fn pad_to_max_has_high_waste() {
        let config = BatchConfig {
            max_sequence_length: 100,
            padding_strategy: PaddingStrategy::PadToMax,
            ..default_config()
        };
        let mut engine = BatchEngine::new(config);
        engine.submit(make_request(&[1])); // 1 token padded to 100
        let b = engine.form_batch().unwrap();
        engine.complete_batch(vec![make_result(b.requests[0], vec![10], 50.0)]);
        // waste = (100-1)/100 = 0.99
        assert!(engine.stats().avg_padding_waste > 0.9);
    }

    #[test]
    fn pad_to_longest_with_equal_lengths_no_waste() {
        let config =
            BatchConfig { padding_strategy: PaddingStrategy::PadToLongest, ..default_config() };
        let mut engine = BatchEngine::new(config);
        engine.submit(make_request(&[1, 2, 3]));
        engine.submit(make_request(&[4, 5, 6]));
        let b = engine.form_batch().unwrap();
        engine
            .complete_batch(b.requests.iter().map(|&id| make_result(id, vec![10], 50.0)).collect());
        assert!(engine.stats().avg_padding_waste.abs() < f64::EPSILON);
    }

    #[test]
    fn padding_waste_accumulates_across_batches() {
        let config = BatchConfig {
            max_sequence_length: 10,
            max_batch_size: 1,
            padding_strategy: PaddingStrategy::PadToMax,
            ..default_config()
        };
        let mut engine = BatchEngine::new(config);

        // Batch 1: 1 token → waste = 9/10 = 0.9
        engine.submit(make_request(&[1]));
        let b = engine.form_batch().unwrap();
        engine.complete_batch(vec![make_result(b.requests[0], vec![10], 50.0)]);

        // Batch 2: 5 tokens → waste = 5/10 = 0.5
        engine.submit(make_request(&[1, 2, 3, 4, 5]));
        let b = engine.form_batch().unwrap();
        engine.complete_batch(vec![make_result(b.requests[0], vec![10], 50.0)]);

        // avg waste = (0.9 + 0.5) / 2 = 0.7
        assert!((engine.stats().avg_padding_waste - 0.7).abs() < 0.01);
    }

    // ── Edge cases ───────────────────────────────────────────────────────

    #[test]
    fn empty_prompt_tokens() {
        let mut engine = BatchEngine::new(default_config());
        engine.submit(make_request(&[]));
        let batch = engine.form_batch().unwrap();
        assert_eq!(batch.padded_length, 0);
    }

    #[test]
    fn request_priority_ordering() {
        assert!(RequestPriority::Low < RequestPriority::Normal);
        assert!(RequestPriority::Normal < RequestPriority::High);
        assert!(RequestPriority::High < RequestPriority::Realtime);
    }

    #[test]
    fn finish_reason_eq() {
        assert_eq!(FinishReason::MaxTokens, FinishReason::MaxTokens);
        assert_eq!(FinishReason::EosToken, FinishReason::EosToken);
        assert_eq!(FinishReason::Timeout, FinishReason::Timeout);
        assert_eq!(FinishReason::Error("x".into()), FinishReason::Error("x".into()));
        assert_ne!(FinishReason::MaxTokens, FinishReason::Timeout);
    }

    #[test]
    fn default_config_values() {
        let c = BatchConfig::default();
        assert_eq!(c.max_batch_size, 32);
        assert_eq!(c.max_sequence_length, 2048);
        assert!(!c.dynamic_batching);
        assert_eq!(c.timeout_ms, 30_000);
        assert!(!c.priority_scheduling);
    }

    #[test]
    fn batch_engine_sequential_ids() {
        let mut engine = BatchEngine::new(default_config());
        let id1 = engine.submit(make_request(&[1]));
        let id2 = engine.submit(make_request(&[2]));
        let id3 = engine.submit(make_request(&[3]));
        assert_eq!(id2, id1 + 1);
        assert_eq!(id3, id2 + 1);
    }

    #[test]
    fn form_batch_sets_active_batch() {
        let mut engine = BatchEngine::new(default_config());
        engine.submit(make_request(&[1]));
        engine.form_batch();
        assert!(engine.active_batch().is_some());
    }

    #[test]
    fn active_batch_has_correct_batch_size() {
        let mut engine = BatchEngine::new(default_config());
        engine.submit(make_request(&[1]));
        engine.submit(make_request(&[2]));
        engine.form_batch();
        assert_eq!(engine.active_batch().unwrap().batch_size, 2);
    }

    // ── Percentile helper unit tests ─────────────────────────────────────

    #[test]
    fn percentile_empty() {
        assert_eq!(percentile(&[], 50.0), 0.0);
    }

    #[test]
    fn percentile_single_value() {
        assert!((percentile(&[42.0], 50.0) - 42.0).abs() < f64::EPSILON);
        assert!((percentile(&[42.0], 99.0) - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn percentile_sorted_input() {
        let v = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        assert!((percentile(&v, 50.0) - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn percentile_unsorted_input() {
        let v = vec![50.0, 10.0, 30.0, 40.0, 20.0];
        assert!((percentile(&v, 50.0) - 30.0).abs() < f64::EPSILON);
    }

    // ── find_bucket helper ───────────────────────────────────────────────

    #[test]
    fn find_bucket_exact() {
        assert_eq!(find_bucket(&[8, 16, 32], 16), 16);
    }

    #[test]
    fn find_bucket_rounds_up() {
        assert_eq!(find_bucket(&[8, 16, 32], 10), 16);
    }

    #[test]
    fn find_bucket_no_fit() {
        assert_eq!(find_bucket(&[8, 16], 20), 20);
    }

    #[test]
    fn find_bucket_unsorted() {
        assert_eq!(find_bucket(&[32, 8, 16], 10), 16);
    }

    // ── Avg wait time ────────────────────────────────────────────────────

    #[test]
    fn avg_wait_time_tracked() {
        let mut engine = BatchEngine::new(default_config());
        engine.submit(make_request_at(&[1], 10));
        engine.submit(make_request_at(&[2], 20));
        let b = engine.form_batch().unwrap();
        let results: Vec<_> =
            b.requests.iter().map(|&id| make_result(id, vec![10], 50.0)).collect();
        engine.complete_batch(results);
        // Wait times: first = 20-10=10, second = 20-20=0 → avg=5
        assert!((engine.stats().avg_wait_time_ms - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn large_batch_stress() {
        let config = BatchConfig { max_batch_size: 1000, ..default_config() };
        let mut engine = BatchEngine::new(config);
        for i in 0..500 {
            engine.submit(make_request(&[i as u32; 10]));
        }
        let batch = engine.form_batch().unwrap();
        assert_eq!(batch.batch_size, 500);
        assert_eq!(engine.pending_count(), 0);
    }
}
