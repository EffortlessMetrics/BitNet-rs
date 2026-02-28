//! Multi-request GPU batching for the inference server.
//!
//! [`GpuBatchScheduler`] collects incoming inference requests and groups them
//! into GPU-friendly batches.  Batch sizes adapt dynamically based on
//! available GPU memory, and a priority queue ensures interactive requests
//! are served before bulk workloads.

use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{Mutex, Notify, oneshot};

// ---------------------------------------------------------------------------
// Priority
// ---------------------------------------------------------------------------

/// Priority class for incoming requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuRequestPriority {
    /// Lowest priority – background / batch jobs.
    Batch = 0,
    /// Default priority for API requests.
    Normal = 1,
    /// Elevated priority for latency-sensitive requests.
    Interactive = 2,
}

impl Default for GpuRequestPriority {
    fn default() -> Self {
        Self::Normal
    }
}

// ---------------------------------------------------------------------------
// Request / Response
// ---------------------------------------------------------------------------

/// A single inference request entering the GPU batch scheduler.
#[derive(Debug)]
pub struct GpuBatchRequest {
    /// Opaque caller-assigned identifier.
    pub id: String,
    /// Token IDs to process.
    pub token_ids: Vec<u32>,
    /// Maximum new tokens to generate.
    pub max_tokens: usize,
    /// Priority class.
    pub priority: GpuRequestPriority,
    /// Timestamp at which the request was submitted.
    pub submitted_at: Instant,
}

/// Result returned to the caller after a batch has been processed.
#[derive(Debug, Clone)]
pub struct GpuBatchResponse {
    /// Echoed request id.
    pub id: String,
    /// Generated output token IDs.
    pub output_token_ids: Vec<u32>,
    /// Wall-clock time the request spent in the queue.
    pub queue_duration: Duration,
    /// Wall-clock time spent in GPU execution.
    pub execution_duration: Duration,
}

// ---------------------------------------------------------------------------
// Internal priority-queue wrapper
// ---------------------------------------------------------------------------

/// Wraps a request together with its reply channel so the heap can order by
/// priority (descending) and submission time (ascending – FIFO within the same
/// priority).
struct PendingRequest {
    request: GpuBatchRequest,
    response_tx: oneshot::Sender<GpuBatchResponse>,
    sequence: u64,
}

impl PartialEq for PendingRequest {
    fn eq(&self, other: &Self) -> bool {
        self.request.priority == other.request.priority && self.sequence == other.sequence
    }
}

impl Eq for PendingRequest {}

impl PartialOrd for PendingRequest {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PendingRequest {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.request
            .priority
            .cmp(&other.request.priority)
            .then_with(|| other.sequence.cmp(&self.sequence)) // lower seq = earlier
    }
}

// ---------------------------------------------------------------------------
// GPU memory estimator
// ---------------------------------------------------------------------------

/// Snapshot of GPU memory state used to size batches dynamically.
#[derive(Debug, Clone, Copy)]
pub struct GpuMemorySnapshot {
    /// Total device memory in bytes.
    pub total_bytes: u64,
    /// Currently free device memory in bytes.
    pub free_bytes: u64,
}

impl GpuMemorySnapshot {
    /// Estimate how many requests can fit given `per_request_bytes`.
    pub fn estimate_capacity(&self, per_request_bytes: u64) -> usize {
        if per_request_bytes == 0 {
            return 0;
        }
        // Use 90 % of free memory as a safety margin.
        let usable = (self.free_bytes as f64 * 0.9) as u64;
        (usable / per_request_bytes) as usize
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration knobs for [`GpuBatchScheduler`].
#[derive(Debug, Clone)]
pub struct GpuBatchConfig {
    /// Hard upper limit on batch size.
    pub max_batch_size: usize,
    /// Maximum time to wait for a full batch before flushing a partial one.
    pub batch_timeout: Duration,
    /// Estimated per-request GPU memory consumption in bytes.
    pub per_request_memory_bytes: u64,
    /// Maximum number of pending requests in the submission queue.
    pub max_queue_depth: usize,
}

impl Default for GpuBatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            batch_timeout: Duration::from_millis(50),
            per_request_memory_bytes: 64 * 1024 * 1024, // 64 MiB
            max_queue_depth: 256,
        }
    }
}

impl GpuBatchConfig {
    /// Validate configuration values, returning an error message on failure.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_batch_size == 0 {
            return Err("max_batch_size must be > 0".into());
        }
        if self.max_queue_depth == 0 {
            return Err("max_queue_depth must be > 0".into());
        }
        if self.per_request_memory_bytes == 0 {
            return Err("per_request_memory_bytes must be > 0".into());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Scheduler statistics
// ---------------------------------------------------------------------------

/// Runtime statistics exposed by the scheduler.
#[derive(Debug, Clone)]
pub struct GpuBatchStats {
    pub total_requests_submitted: u64,
    pub total_batches_executed: u64,
    pub total_requests_completed: u64,
    pub current_queue_depth: usize,
}

// ---------------------------------------------------------------------------
// GpuBatchScheduler
// ---------------------------------------------------------------------------

/// Collects individual inference requests, groups them into GPU-friendly
/// batches (sized dynamically based on memory availability), and dispatches
/// them respecting a priority queue.
pub struct GpuBatchScheduler {
    config: GpuBatchConfig,
    queue: Arc<Mutex<BinaryHeap<PendingRequest>>>,
    notify: Arc<Notify>,
    sequence: AtomicU64,
    running: AtomicBool,
    stats_submitted: AtomicU64,
    stats_batches: AtomicU64,
    stats_completed: AtomicU64,
}

impl GpuBatchScheduler {
    /// Create a new scheduler with the given configuration.
    pub fn new(config: GpuBatchConfig) -> Result<Self, String> {
        config.validate()?;
        Ok(Self {
            config,
            queue: Arc::new(Mutex::new(BinaryHeap::new())),
            notify: Arc::new(Notify::new()),
            sequence: AtomicU64::new(0),
            running: AtomicBool::new(false),
            stats_submitted: AtomicU64::new(0),
            stats_batches: AtomicU64::new(0),
            stats_completed: AtomicU64::new(0),
        })
    }

    /// Submit a request and receive a [`oneshot::Receiver`] that will
    /// eventually contain the response.
    pub async fn submit(
        &self,
        request: GpuBatchRequest,
    ) -> Result<oneshot::Receiver<GpuBatchResponse>, String> {
        let mut queue = self.queue.lock().await;
        if queue.len() >= self.config.max_queue_depth {
            return Err("GPU batch queue is full".into());
        }
        let (tx, rx) = oneshot::channel();
        let seq = self.sequence.fetch_add(1, Ordering::Relaxed);
        queue.push(PendingRequest {
            request,
            response_tx: tx,
            sequence: seq,
        });
        drop(queue);
        self.stats_submitted.fetch_add(1, Ordering::Relaxed);
        self.notify.notify_one();
        Ok(rx)
    }

    /// Drain up to `max_count` highest-priority requests from the queue.
    pub async fn drain_batch(&self, max_count: usize) -> Vec<(GpuBatchRequest, oneshot::Sender<GpuBatchResponse>)> {
        let mut queue = self.queue.lock().await;
        let n = max_count.min(queue.len());
        let mut batch = Vec::with_capacity(n);
        for _ in 0..n {
            if let Some(pending) = queue.pop() {
                batch.push((pending.request, pending.response_tx));
            }
        }
        batch
    }

    /// Compute the effective batch size given a GPU memory snapshot.
    pub fn effective_batch_size(&self, mem: &GpuMemorySnapshot) -> usize {
        let mem_limited = mem.estimate_capacity(self.config.per_request_memory_bytes);
        mem_limited.min(self.config.max_batch_size).max(1)
    }

    /// Start a background processing loop.
    ///
    /// The provided callback `execute_batch` is invoked for each batch and
    /// must return a `Vec<GpuBatchResponse>` matching the input requests in
    /// order.
    pub async fn run<F, Fut>(&self, gpu_memory: GpuMemorySnapshot, execute_batch: F)
    where
        F: Fn(Vec<GpuBatchRequest>) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Vec<GpuBatchResponse>> + Send,
    {
        self.running.store(true, Ordering::SeqCst);
        while self.running.load(Ordering::SeqCst) {
            // Wait until notified or timeout.
            let _ = tokio::time::timeout(self.config.batch_timeout, self.notify.notified()).await;

            let batch_size = self.effective_batch_size(&gpu_memory);
            let batch = self.drain_batch(batch_size).await;
            if batch.is_empty() {
                continue;
            }

            let (requests, senders): (Vec<_>, Vec<_>) = batch.into_iter().unzip();
            let count = requests.len();
            let responses = execute_batch(requests).await;

            for (resp, tx) in responses.into_iter().zip(senders) {
                let _ = tx.send(resp);
            }

            self.stats_batches.fetch_add(1, Ordering::Relaxed);
            self.stats_completed
                .fetch_add(count as u64, Ordering::Relaxed);
        }
    }

    /// Signal the processing loop to stop.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        self.notify.notify_one();
    }

    /// Return `true` if the scheduler is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Current queue depth.
    pub async fn queue_depth(&self) -> usize {
        self.queue.lock().await.len()
    }

    /// Snapshot of cumulative statistics.
    pub async fn stats(&self) -> GpuBatchStats {
        GpuBatchStats {
            total_requests_submitted: self.stats_submitted.load(Ordering::Relaxed),
            total_batches_executed: self.stats_batches.load(Ordering::Relaxed),
            total_requests_completed: self.stats_completed.load(Ordering::Relaxed),
            current_queue_depth: self.queue.lock().await.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_request(id: &str, priority: GpuRequestPriority) -> GpuBatchRequest {
        GpuBatchRequest {
            id: id.to_string(),
            token_ids: vec![1, 2, 3],
            max_tokens: 16,
            priority,
            submitted_at: Instant::now(),
        }
    }

    // -- Config validation ---------------------------------------------------

    #[test]
    fn test_config_default_is_valid() {
        assert!(GpuBatchConfig::default().validate().is_ok());
    }

    #[test]
    fn test_config_zero_batch_size_rejected() {
        let mut cfg = GpuBatchConfig::default();
        cfg.max_batch_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_zero_queue_depth_rejected() {
        let mut cfg = GpuBatchConfig::default();
        cfg.max_queue_depth = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_zero_per_request_memory_rejected() {
        let mut cfg = GpuBatchConfig::default();
        cfg.per_request_memory_bytes = 0;
        assert!(cfg.validate().is_err());
    }

    // -- Memory estimation ---------------------------------------------------

    #[test]
    fn test_memory_snapshot_estimate_capacity() {
        let snap = GpuMemorySnapshot {
            total_bytes: 8 * 1024 * 1024 * 1024, // 8 GiB
            free_bytes: 4 * 1024 * 1024 * 1024,  // 4 GiB free
        };
        let per_req = 64 * 1024 * 1024; // 64 MiB
        let cap = snap.estimate_capacity(per_req);
        // 4 GiB * 0.9 / 64 MiB ≈ 57
        assert!(cap > 50 && cap < 60, "got {cap}");
    }

    #[test]
    fn test_memory_snapshot_zero_per_request() {
        let snap = GpuMemorySnapshot {
            total_bytes: 8_000_000_000,
            free_bytes: 4_000_000_000,
        };
        assert_eq!(snap.estimate_capacity(0), 0);
    }

    // -- Effective batch size ------------------------------------------------

    #[test]
    fn test_effective_batch_size_memory_limited() {
        let cfg = GpuBatchConfig {
            max_batch_size: 64,
            per_request_memory_bytes: 512 * 1024 * 1024, // 512 MiB
            ..Default::default()
        };
        let sched = GpuBatchScheduler::new(cfg).unwrap();
        let mem = GpuMemorySnapshot {
            total_bytes: 4 * 1024 * 1024 * 1024,
            free_bytes: 2 * 1024 * 1024 * 1024, // 2 GiB free
        };
        let bs = sched.effective_batch_size(&mem);
        // 2 GiB * 0.9 / 512 MiB ≈ 3
        assert!(bs >= 1 && bs <= 4, "got {bs}");
    }

    #[test]
    fn test_effective_batch_size_config_limited() {
        let cfg = GpuBatchConfig {
            max_batch_size: 4,
            per_request_memory_bytes: 1024, // tiny
            ..Default::default()
        };
        let sched = GpuBatchScheduler::new(cfg).unwrap();
        let mem = GpuMemorySnapshot {
            total_bytes: 8_000_000_000,
            free_bytes: 8_000_000_000,
        };
        assert_eq!(sched.effective_batch_size(&mem), 4);
    }

    // -- Submit & drain ------------------------------------------------------

    #[tokio::test]
    async fn test_submit_and_drain_fifo() {
        let sched = GpuBatchScheduler::new(GpuBatchConfig::default()).unwrap();
        let _rx1 = sched.submit(make_request("a", GpuRequestPriority::Normal)).await.unwrap();
        let _rx2 = sched.submit(make_request("b", GpuRequestPriority::Normal)).await.unwrap();
        let _rx3 = sched.submit(make_request("c", GpuRequestPriority::Normal)).await.unwrap();

        let batch = sched.drain_batch(10).await;
        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0].0.id, "a");
        assert_eq!(batch[1].0.id, "b");
        assert_eq!(batch[2].0.id, "c");
    }

    #[tokio::test]
    async fn test_priority_ordering() {
        let sched = GpuBatchScheduler::new(GpuBatchConfig::default()).unwrap();
        let _rx1 = sched.submit(make_request("batch", GpuRequestPriority::Batch)).await.unwrap();
        let _rx2 = sched.submit(make_request("interactive", GpuRequestPriority::Interactive)).await.unwrap();
        let _rx3 = sched.submit(make_request("normal", GpuRequestPriority::Normal)).await.unwrap();

        let batch = sched.drain_batch(10).await;
        assert_eq!(batch[0].0.id, "interactive");
        assert_eq!(batch[1].0.id, "normal");
        assert_eq!(batch[2].0.id, "batch");
    }

    #[tokio::test]
    async fn test_queue_full_rejected() {
        let cfg = GpuBatchConfig {
            max_queue_depth: 2,
            ..Default::default()
        };
        let sched = GpuBatchScheduler::new(cfg).unwrap();
        let _ = sched.submit(make_request("1", GpuRequestPriority::Normal)).await.unwrap();
        let _ = sched.submit(make_request("2", GpuRequestPriority::Normal)).await.unwrap();
        let res = sched.submit(make_request("3", GpuRequestPriority::Normal)).await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let sched = GpuBatchScheduler::new(GpuBatchConfig::default()).unwrap();
        let _ = sched.submit(make_request("x", GpuRequestPriority::Normal)).await.unwrap();
        let stats = sched.stats().await;
        assert_eq!(stats.total_requests_submitted, 1);
        assert_eq!(stats.current_queue_depth, 1);
    }

    #[tokio::test]
    async fn test_drain_respects_limit() {
        let sched = GpuBatchScheduler::new(GpuBatchConfig::default()).unwrap();
        for i in 0..10 {
            let _ = sched
                .submit(make_request(&format!("r{i}"), GpuRequestPriority::Normal))
                .await
                .unwrap();
        }
        let batch = sched.drain_batch(3).await;
        assert_eq!(batch.len(), 3);
        assert_eq!(sched.queue_depth().await, 7);
    }
}
