//! GPU-aware server inference handler with priority queuing and
//! load balancing.

use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

/// Priority levels for inference requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RequestPriority {
    /// Lowest priority – background/offline jobs.
    Background = 0,
    /// Normal interactive traffic.
    Interactive = 1,
    /// Highest priority – batch workloads.
    Batch = 2,
}

/// Sampling configuration carried with a request.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self { temperature: 1.0, top_k: 50, top_p: 1.0 }
    }
}

/// An inference request submitted to the handler.
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub sampling_config: SamplingConfig,
    pub priority: RequestPriority,
    /// Maximum wall-clock time (ms) before the request should be
    /// considered timed-out. `0` means no deadline.
    pub deadline_ms: u64,
}

/// The response returned after processing an inference request.
#[derive(Debug, Clone)]
pub struct InferenceResponse {
    pub output_text: String,
    pub tokens: Vec<u32>,
    pub latency_ms: u64,
    pub device_used: usize,
    pub queue_wait_ms: u64,
}

/// Error type for handler operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HandlerError {
    QueueFull,
    Timeout,
    ShuttingDown,
    NoHealthyDevice,
}

impl std::fmt::Display for HandlerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QueueFull => write!(f, "request queue is full"),
            Self::Timeout => write!(f, "request exceeded deadline"),
            Self::ShuttingDown => write!(f, "handler is shutting down"),
            Self::NoHealthyDevice => {
                write!(f, "no healthy device available")
            }
        }
    }
}

impl std::error::Error for HandlerError {}

// ---------------------------------------------------------------------------
// Queued wrapper (for priority ordering)
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct QueuedRequest {
    request: InferenceRequest,
    enqueued_at: Instant,
    seq: u64,
}

impl PartialEq for QueuedRequest {
    fn eq(&self, other: &Self) -> bool {
        self.request.priority == other.request.priority && self.seq == other.seq
    }
}

impl Eq for QueuedRequest {}

impl PartialOrd for QueuedRequest {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueuedRequest {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.request
            .priority
            .cmp(&other.request.priority)
            // Tie-break: lower seq (earlier arrival) wins.
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

// ---------------------------------------------------------------------------
// RequestQueue
// ---------------------------------------------------------------------------

/// Priority-ordered, capacity-limited request queue.
pub struct RequestQueue {
    heap: BinaryHeap<QueuedRequest>,
    capacity: usize,
    next_seq: u64,
}

impl RequestQueue {
    pub fn new(capacity: usize) -> Self {
        Self { heap: BinaryHeap::new(), capacity, next_seq: 0 }
    }

    /// Push a request. Returns `Err(HandlerError::QueueFull)` when at
    /// capacity.
    pub fn push(&mut self, request: InferenceRequest) -> Result<(), HandlerError> {
        if self.heap.len() >= self.capacity {
            return Err(HandlerError::QueueFull);
        }
        let seq = self.next_seq;
        self.next_seq += 1;
        self.heap.push(QueuedRequest { request, enqueued_at: Instant::now(), seq });
        Ok(())
    }

    /// Pop the highest-priority request.
    pub fn pop(&mut self) -> Option<(InferenceRequest, Instant)> {
        self.heap.pop().map(|q| (q.request, q.enqueued_at))
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Load-balancing strategies
// ---------------------------------------------------------------------------

/// Strategy used to pick the next device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BalancingStrategy {
    RoundRobin,
    LeastLoaded,
    MemoryAware,
}

/// Per-device bookkeeping used by the load balancer.
#[derive(Debug)]
struct DeviceSlot {
    active_requests: AtomicUsize,
    healthy: AtomicBool,
    available_memory_mb: AtomicU64,
}

impl DeviceSlot {
    fn new() -> Self {
        Self {
            active_requests: AtomicUsize::new(0),
            healthy: AtomicBool::new(true),
            available_memory_mb: AtomicU64::new(1024),
        }
    }
}

/// Routes requests across multiple devices.
pub struct LoadBalancer {
    strategy: BalancingStrategy,
    devices: Vec<DeviceSlot>,
    round_robin_idx: AtomicUsize,
}

impl LoadBalancer {
    pub fn new(device_count: usize, strategy: BalancingStrategy) -> Self {
        let devices = (0..device_count).map(|_| DeviceSlot::new()).collect();
        Self { strategy, devices, round_robin_idx: AtomicUsize::new(0) }
    }

    /// Pick the next device id, skipping unhealthy ones.
    pub fn select_device(&self) -> Result<usize, HandlerError> {
        match self.strategy {
            BalancingStrategy::RoundRobin => self.select_round_robin(),
            BalancingStrategy::LeastLoaded => self.select_least_loaded(),
            BalancingStrategy::MemoryAware => self.select_memory_aware(),
        }
    }

    /// Mark a device as having one more active request.
    pub fn begin_request(&self, device_id: usize) {
        if let Some(slot) = self.devices.get(device_id) {
            slot.active_requests.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Mark a device as having completed one request.
    pub fn end_request(&self, device_id: usize) {
        if let Some(slot) = self.devices.get(device_id) {
            slot.active_requests.fetch_sub(1, Ordering::Relaxed);
        }
    }

    pub fn active_requests(&self, device_id: usize) -> usize {
        self.devices.get(device_id).map(|s| s.active_requests.load(Ordering::Relaxed)).unwrap_or(0)
    }

    pub fn set_device_healthy(&self, device_id: usize, healthy: bool) {
        if let Some(slot) = self.devices.get(device_id) {
            slot.healthy.store(healthy, Ordering::Relaxed);
        }
    }

    pub fn is_device_healthy(&self, device_id: usize) -> bool {
        self.devices.get(device_id).map(|s| s.healthy.load(Ordering::Relaxed)).unwrap_or(false)
    }

    pub fn set_available_memory(&self, device_id: usize, mb: u64) {
        if let Some(slot) = self.devices.get(device_id) {
            slot.available_memory_mb.store(mb, Ordering::Relaxed);
        }
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    // -- private selection helpers ----------------------------------------

    fn healthy_ids(&self) -> Vec<usize> {
        self.devices
            .iter()
            .enumerate()
            .filter(|(_, s)| s.healthy.load(Ordering::Relaxed))
            .map(|(i, _)| i)
            .collect()
    }

    fn select_round_robin(&self) -> Result<usize, HandlerError> {
        let healthy = self.healthy_ids();
        if healthy.is_empty() {
            return Err(HandlerError::NoHealthyDevice);
        }
        let idx = self.round_robin_idx.fetch_add(1, Ordering::Relaxed);
        Ok(healthy[idx % healthy.len()])
    }

    fn select_least_loaded(&self) -> Result<usize, HandlerError> {
        self.healthy_ids()
            .into_iter()
            .min_by_key(|&id| self.devices[id].active_requests.load(Ordering::Relaxed))
            .ok_or(HandlerError::NoHealthyDevice)
    }

    fn select_memory_aware(&self) -> Result<usize, HandlerError> {
        self.healthy_ids()
            .into_iter()
            .max_by_key(|&id| self.devices[id].available_memory_mb.load(Ordering::Relaxed))
            .ok_or(HandlerError::NoHealthyDevice)
    }
}

// ---------------------------------------------------------------------------
// GpuInferenceHandler
// ---------------------------------------------------------------------------

/// Configuration for [`GpuInferenceHandler`].
#[derive(Debug, Clone)]
pub struct HandlerConfig {
    pub queue_capacity: usize,
    pub device_count: usize,
    pub strategy: BalancingStrategy,
    pub default_deadline_ms: u64,
}

impl Default for HandlerConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 128,
            device_count: 1,
            strategy: BalancingStrategy::RoundRobin,
            default_deadline_ms: 30_000,
        }
    }
}

/// Type alias for the mock token-generation callback.
type MockGenerateFn = dyn Fn(&str, usize) -> Vec<u32> + Send + Sync;

/// Top-level handler that accepts inference requests, queues them
/// with priority ordering, picks a device via the [`LoadBalancer`],
/// and returns an [`InferenceResponse`].
pub struct GpuInferenceHandler {
    queue: Mutex<RequestQueue>,
    balancer: LoadBalancer,
    shutting_down: AtomicBool,
    config: HandlerConfig,
    /// Mock token generator for testing without a real model.
    mock_generate: Option<Arc<MockGenerateFn>>,
}

impl GpuInferenceHandler {
    pub fn new(config: HandlerConfig) -> Self {
        let balancer = LoadBalancer::new(config.device_count, config.strategy);
        let queue = Mutex::new(RequestQueue::new(config.queue_capacity));
        Self { queue, balancer, shutting_down: AtomicBool::new(false), config, mock_generate: None }
    }

    /// Provide a mock token generator (useful for tests).
    pub fn with_mock_generate<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, usize) -> Vec<u32> + Send + Sync + 'static,
    {
        self.mock_generate = Some(Arc::new(f));
        self
    }

    /// Submit a request and process it synchronously.
    pub fn handle_request(&self, req: InferenceRequest) -> Result<InferenceResponse, HandlerError> {
        if self.shutting_down.load(Ordering::SeqCst) {
            return Err(HandlerError::ShuttingDown);
        }

        // Enqueue then immediately dequeue (single-threaded path
        // mirrors what a real async runtime would do).
        let enqueued_at = Instant::now();
        {
            let mut q = self.queue.lock().unwrap();
            q.push(req.clone())?;
        }

        let (dequeued, dequeued_at) = {
            let mut q = self.queue.lock().unwrap();
            q.pop().expect("just pushed")
        };

        let _queue_wait = dequeued_at.elapsed();
        let queue_wait_ms = enqueued_at.elapsed().as_millis() as u64;

        // Check deadline.
        let deadline = if dequeued.deadline_ms > 0 {
            dequeued.deadline_ms
        } else {
            self.config.default_deadline_ms
        };
        let elapsed_total = enqueued_at.elapsed().as_millis() as u64;
        if elapsed_total > deadline {
            return Err(HandlerError::Timeout);
        }

        // Select a device.
        let device_id = self.balancer.select_device()?;
        self.balancer.begin_request(device_id);

        // "Inference" – use mock generator or produce dummy tokens.
        let infer_start = Instant::now();
        let tokens = if let Some(ref gfn) = self.mock_generate {
            gfn(&dequeued.prompt, dequeued.max_tokens)
        } else {
            // Placeholder: return sequential token ids.
            (0..dequeued.max_tokens as u32).collect()
        };

        let output_text = format!("[generated {} tokens on device {}]", tokens.len(), device_id);
        let latency_ms = infer_start.elapsed().as_millis() as u64;

        self.balancer.end_request(device_id);

        // Final deadline check after inference.
        let total_ms = enqueued_at.elapsed().as_millis() as u64;
        if total_ms > deadline {
            return Err(HandlerError::Timeout);
        }

        Ok(InferenceResponse {
            output_text,
            tokens,
            latency_ms,
            device_used: device_id,
            queue_wait_ms,
        })
    }

    /// Access the underlying load balancer.
    pub fn balancer(&self) -> &LoadBalancer {
        &self.balancer
    }

    /// Number of requests currently sitting in the queue.
    pub fn queued_requests(&self) -> usize {
        self.queue.lock().unwrap().len()
    }

    /// Initiate graceful shutdown – new requests are rejected, but
    /// already-queued work can still be drained.
    pub fn shutdown(&self) {
        self.shutting_down.store(true, Ordering::SeqCst);
        log::info!("GpuInferenceHandler: shutting down");
    }

    pub fn is_shutting_down(&self) -> bool {
        self.shutting_down.load(Ordering::SeqCst)
    }

    /// Drain remaining requests from the queue, processing each one.
    /// Returns the number of requests drained.
    pub fn drain_queue(&self) -> Vec<InferenceResponse> {
        let mut results = Vec::new();
        loop {
            let item = {
                let mut q = self.queue.lock().unwrap();
                q.pop()
            };
            let Some((req, enqueued_at)) = item else {
                break;
            };
            let queue_wait_ms = enqueued_at.elapsed().as_millis() as u64;
            let device_id = self.balancer.select_device().unwrap_or(0);
            self.balancer.begin_request(device_id);

            let infer_start = Instant::now();
            let tokens = if let Some(ref gfn) = self.mock_generate {
                gfn(&req.prompt, req.max_tokens)
            } else {
                (0..req.max_tokens as u32).collect()
            };
            let output_text =
                format!("[generated {} tokens on device {}]", tokens.len(), device_id);
            let latency_ms = infer_start.elapsed().as_millis() as u64;
            self.balancer.end_request(device_id);

            results.push(InferenceResponse {
                output_text,
                tokens,
                latency_ms,
                device_used: device_id,
                queue_wait_ms,
            });
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_request(priority: RequestPriority, prompt: &str) -> InferenceRequest {
        InferenceRequest {
            prompt: prompt.to_string(),
            max_tokens: 4,
            sampling_config: SamplingConfig::default(),
            priority,
            deadline_ms: 0,
        }
    }

    #[test]
    fn queue_push_and_pop_respects_priority() {
        let mut q = RequestQueue::new(8);
        q.push(make_request(RequestPriority::Background, "bg")).unwrap();
        q.push(make_request(RequestPriority::Batch, "batch")).unwrap();
        q.push(make_request(RequestPriority::Interactive, "inter")).unwrap();

        let (r1, _) = q.pop().unwrap();
        let (r2, _) = q.pop().unwrap();
        let (r3, _) = q.pop().unwrap();

        assert_eq!(r1.priority, RequestPriority::Batch);
        assert_eq!(r2.priority, RequestPriority::Interactive);
        assert_eq!(r3.priority, RequestPriority::Background);
    }

    #[test]
    fn queue_capacity_enforced() {
        let mut q = RequestQueue::new(1);
        q.push(make_request(RequestPriority::Batch, "a")).unwrap();
        let err = q.push(make_request(RequestPriority::Batch, "b")).unwrap_err();
        assert_eq!(err, HandlerError::QueueFull);
    }

    #[test]
    fn round_robin_distributes_evenly() {
        let lb = LoadBalancer::new(3, BalancingStrategy::RoundRobin);
        let mut counts = [0usize; 3];
        for _ in 0..9 {
            let id = lb.select_device().unwrap();
            counts[id] += 1;
        }
        assert!(counts.iter().all(|&c| c == 3));
    }

    #[test]
    fn least_loaded_selects_idle_device() {
        let lb = LoadBalancer::new(2, BalancingStrategy::LeastLoaded);
        lb.begin_request(0);
        lb.begin_request(0);
        let id = lb.select_device().unwrap();
        assert_eq!(id, 1);
    }

    #[test]
    fn memory_aware_picks_most_memory() {
        let lb = LoadBalancer::new(2, BalancingStrategy::MemoryAware);
        lb.set_available_memory(0, 512);
        lb.set_available_memory(1, 2048);
        let id = lb.select_device().unwrap();
        assert_eq!(id, 1);
    }

    #[test]
    fn unhealthy_device_skipped() {
        let lb = LoadBalancer::new(2, BalancingStrategy::RoundRobin);
        lb.set_device_healthy(0, false);
        for _ in 0..5 {
            assert_eq!(lb.select_device().unwrap(), 1);
        }
    }

    #[test]
    fn all_unhealthy_returns_error() {
        let lb = LoadBalancer::new(2, BalancingStrategy::RoundRobin);
        lb.set_device_healthy(0, false);
        lb.set_device_healthy(1, false);
        assert_eq!(lb.select_device().unwrap_err(), HandlerError::NoHealthyDevice,);
    }
}
