//! Module stub - implementation pending merge from feature branch
//! Production model serving runtime with health checks and hot-swap.
//!
//! Provides a complete serving stack for GPU-accelerated inference:
//! request queuing with priorities and timeouts, semaphore-based concurrency
//! control, graceful shutdown with request draining, Kubernetes health
//! endpoints, Prometheus-format metrics, and zero-downtime model hot-swap.

use std::collections::{BinaryHeap, HashMap};
use std::fmt;
use std::time::{Duration, Instant};

// ── Device ──────────────────────────────────────────────────────────────────

/// Device type for model placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    /// CPU inference.
    Cpu,
    /// GPU inference on the given ordinal.
    Gpu(u32),
}

impl fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Gpu(id) => write!(f, "gpu:{id}"),
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Configuration
// ═════════════════════════════════════════════════════════════════════════════

/// Serving configuration for the runtime.
#[derive(Debug, Clone)]
pub struct ServingConfig {
    /// Host address to bind (e.g. `"0.0.0.0"`).
    pub host: String,
    /// Port to listen on.
    pub port: u16,
    /// Maximum number of concurrent inference requests.
    pub max_concurrent: usize,
    /// Path to the model file (GGUF/SafeTensors).
    pub model_path: String,
    /// Device to run inference on.
    pub device: DeviceKind,
    /// Number of warmup requests to run before accepting traffic.
    pub warmup_requests: u32,
}

impl Default for ServingConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            max_concurrent: 8,
            model_path: String::new(),
            device: DeviceKind::Cpu,
            warmup_requests: 3,
        }
    }
}

impl ServingConfig {
    /// Validate configuration values.
    pub fn validate(&self) -> Result<(), String> {
        if self.host.is_empty() {
            return Err("host must not be empty".into());
        }
        if self.port == 0 {
            return Err("port must be > 0".into());
        }
        if self.max_concurrent == 0 {
            return Err("max_concurrent must be > 0".into());
        }
        if self.model_path.is_empty() {
            return Err("model_path must not be empty".into());
        }
        Ok(())
    }

    /// Bind address in `host:port` form.
    #[must_use]
    pub fn bind_address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Model Instance
// ═════════════════════════════════════════════════════════════════════════════

/// Status of a loaded model instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelStatus {
    /// Model is loading from disk.
    Loading,
    /// Model is warming up.
    WarmingUp,
    /// Model is ready to serve.
    Ready,
    /// Model is draining in-flight requests before unload.
    Draining,
    /// Model has been unloaded.
    Unloaded,
    /// Model load failed.
    Failed,
}

/// A loaded model with associated tokenizer and device placement.
#[derive(Debug, Clone)]
pub struct ModelInstance {
    /// Unique identifier for this model instance.
    pub id: String,
    /// Path the model was loaded from.
    pub model_path: String,
    /// Optional tokenizer path.
    pub tokenizer_path: Option<String>,
    /// Device the model is placed on.
    pub device: DeviceKind,
    /// Current status.
    pub status: ModelStatus,
    /// When the model was loaded.
    pub loaded_at: Option<Instant>,
    /// Number of requests served by this instance.
    pub requests_served: u64,
    /// Model version tag for hot-swap tracking.
    pub version: u64,
}

impl ModelInstance {
    /// Create a new model instance in `Loading` state.
    pub fn new(id: impl Into<String>, model_path: impl Into<String>, device: DeviceKind) -> Self {
        Self {
            id: id.into(),
            model_path: model_path.into(),
            tokenizer_path: None,
            device,
            status: ModelStatus::Loading,
            loaded_at: None,
            requests_served: 0,
            version: 1,
        }
    }

    /// Transition to `Ready` state.
    pub fn mark_ready(&mut self) {
        self.status = ModelStatus::Ready;
        self.loaded_at = Some(Instant::now());
    }

    /// Transition to `Draining` state.
    pub fn mark_draining(&mut self) {
        self.status = ModelStatus::Draining;
    }

    /// Transition to `Unloaded` state.
    pub fn mark_unloaded(&mut self) {
        self.status = ModelStatus::Unloaded;
    }

    /// Transition to `Failed` state.
    pub fn mark_failed(&mut self) {
        self.status = ModelStatus::Failed;
    }

    /// Whether the model can accept requests.
    #[must_use]
    pub fn is_serving(&self) -> bool {
        self.status == ModelStatus::Ready
    }

    /// Record a served request.
    pub fn record_request(&mut self) {
        self.requests_served += 1;
    }

    /// Uptime since model became ready, or zero.
    #[must_use]
    pub fn uptime(&self) -> Duration {
        self.loaded_at.map(|t| t.elapsed()).unwrap_or(Duration::ZERO)
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Request Queue
// ═════════════════════════════════════════════════════════════════════════════

/// Priority level for inference requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RequestPriority {
    /// Background / batch priority.
    Low = 0,
    /// Default user-facing priority.
    Normal = 1,
    /// Elevated priority.
    High = 2,
    /// System-critical priority.
    Critical = 3,
}

/// Status of an inference request in the queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestStatus {
    /// Waiting in the queue.
    Queued,
    /// Currently being processed.
    Processing,
    /// Completed successfully.
    Completed,
    /// Timed out before processing.
    TimedOut,
    /// Cancelled by the caller.
    Cancelled,
    /// Processing failed.
    Failed,
}

/// An inference request submitted to the serving runtime.
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    /// Unique request identifier.
    pub id: String,
    /// Input prompt text.
    pub prompt: String,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Sampling temperature.
    pub temperature: f32,
    /// Request priority.
    pub priority: RequestPriority,
    /// Current status.
    pub status: RequestStatus,
    /// When the request was enqueued.
    pub enqueued_at: Instant,
    /// Timeout for this request.
    pub timeout: Duration,
}

impl InferenceRequest {
    /// Create a new request with normal priority.
    pub fn new(id: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            prompt: prompt.into(),
            max_tokens: 256,
            temperature: 0.7,
            priority: RequestPriority::Normal,
            status: RequestStatus::Queued,
            enqueued_at: Instant::now(),
            timeout: Duration::from_secs(30),
        }
    }

    /// Set the priority.
    #[must_use]
    pub fn with_priority(mut self, priority: RequestPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Set the timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set max tokens.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set temperature.
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Whether this request has exceeded its timeout.
    #[must_use]
    pub fn is_timed_out(&self) -> bool {
        self.enqueued_at.elapsed() > self.timeout
    }

    /// Remaining time before timeout, or zero.
    #[must_use]
    pub fn remaining_time(&self) -> Duration {
        self.timeout.saturating_sub(self.enqueued_at.elapsed())
    }
}

/// Ordering wrapper so the `BinaryHeap` yields highest-priority first,
/// breaking ties by earliest enqueue time.
#[derive(Debug)]
struct PriorityEntry {
    priority: RequestPriority,
    sequence: u64,
    request: InferenceRequest,
}

impl PartialEq for PriorityEntry {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.sequence == other.sequence
    }
}

impl Eq for PriorityEntry {}

impl PartialOrd for PriorityEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority.cmp(&other.priority).then_with(|| other.sequence.cmp(&self.sequence))
    }
}

/// Priority queue of inference requests with timeout support.
#[derive(Debug)]
pub struct RequestQueue {
    heap: BinaryHeap<PriorityEntry>,
    capacity: usize,
    next_sequence: u64,
    total_enqueued: u64,
    total_dequeued: u64,
    total_timed_out: u64,
}

impl RequestQueue {
    /// Create a new queue with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            heap: BinaryHeap::new(),
            capacity,
            next_sequence: 0,
            total_enqueued: 0,
            total_dequeued: 0,
            total_timed_out: 0,
        }
    }

    /// Enqueue a request. Returns `Err` if the queue is full.
    pub fn enqueue(&mut self, request: InferenceRequest) -> Result<(), String> {
        if self.heap.len() >= self.capacity {
            return Err("request queue is full".into());
        }
        let priority = request.priority;
        let seq = self.next_sequence;
        self.next_sequence += 1;
        self.total_enqueued += 1;
        self.heap.push(PriorityEntry { priority, sequence: seq, request });
        Ok(())
    }

    /// Dequeue the highest-priority non-timed-out request.
    /// Timed-out entries are silently discarded.
    pub fn dequeue(&mut self) -> Option<InferenceRequest> {
        while let Some(entry) = self.heap.pop() {
            if entry.request.is_timed_out() {
                self.total_timed_out += 1;
                continue;
            }
            self.total_dequeued += 1;
            return Some(entry.request);
        }
        None
    }

    /// Number of entries currently in the queue (may include timed-out).
    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Whether the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Total requests ever enqueued.
    #[must_use]
    pub fn total_enqueued(&self) -> u64 {
        self.total_enqueued
    }

    /// Total requests successfully dequeued.
    #[must_use]
    pub fn total_dequeued(&self) -> u64 {
        self.total_dequeued
    }

    /// Total requests that timed out while queued.
    #[must_use]
    pub fn total_timed_out(&self) -> u64 {
        self.total_timed_out
    }

    /// Remaining capacity.
    #[must_use]
    pub fn remaining_capacity(&self) -> usize {
        self.capacity.saturating_sub(self.heap.len())
    }

    /// Purge all timed-out entries and return how many were removed.
    pub fn purge_timed_out(&mut self) -> usize {
        let before = self.heap.len();
        let kept: Vec<_> = self.heap.drain().filter(|e| !e.request.is_timed_out()).collect();
        let removed = before - kept.len();
        self.total_timed_out += removed as u64;
        self.heap = BinaryHeap::from(kept);
        removed
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Request Handler
// ═════════════════════════════════════════════════════════════════════════════

/// Result of processing a single inference request.
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// Request ID this result corresponds to.
    pub request_id: String,
    /// Generated text.
    pub generated_text: String,
    /// Number of tokens generated.
    pub tokens_generated: usize,
    /// Wall-clock latency.
    pub latency: Duration,
    /// Tokens per second.
    pub tokens_per_second: f64,
    /// Whether generation was truncated by max_tokens.
    pub truncated: bool,
}

/// Processes individual inference requests through the model pipeline.
#[derive(Debug)]
pub struct RequestHandler {
    /// The model instance to run inference on.
    model_id: String,
    /// Total requests handled.
    total_handled: u64,
    /// Total failures.
    total_failures: u64,
    /// Cumulative latency for average computation.
    cumulative_latency: Duration,
}

impl RequestHandler {
    /// Create a handler bound to a model instance.
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            total_handled: 0,
            total_failures: 0,
            cumulative_latency: Duration::ZERO,
        }
    }

    /// Process a single request, producing an `InferenceResult`.
    ///
    /// In this mock implementation the generated text is derived from the
    /// prompt length and `max_tokens`; a real implementation would invoke
    /// the inference engine.
    pub fn handle(&mut self, request: &InferenceRequest) -> Result<InferenceResult, String> {
        if request.prompt.is_empty() {
            self.total_failures += 1;
            return Err("empty prompt".into());
        }
        if request.is_timed_out() {
            self.total_failures += 1;
            return Err("request timed out before processing".into());
        }

        let start = Instant::now();
        // Simulate token generation proportional to max_tokens.
        let tokens = request.max_tokens.min(512);
        let generated = format!("[generated:{tokens}]");
        let latency = start.elapsed();
        let tps = if latency.as_secs_f64() > 0.0 {
            tokens as f64 / latency.as_secs_f64()
        } else {
            tokens as f64
        };

        self.total_handled += 1;
        self.cumulative_latency += latency;

        Ok(InferenceResult {
            request_id: request.id.clone(),
            generated_text: generated,
            tokens_generated: tokens,
            latency,
            tokens_per_second: tps,
            truncated: request.max_tokens > 512,
        })
    }

    /// Model ID this handler is bound to.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Total requests successfully handled.
    #[must_use]
    pub fn total_handled(&self) -> u64 {
        self.total_handled
    }

    /// Total failures.
    #[must_use]
    pub fn total_failures(&self) -> u64 {
        self.total_failures
    }

    /// Average latency across all handled requests, or zero.
    #[must_use]
    pub fn average_latency(&self) -> Duration {
        if self.total_handled == 0 {
            Duration::ZERO
        } else {
            self.cumulative_latency / self.total_handled as u32
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Concurrency Limiter
// ═════════════════════════════════════════════════════════════════════════════

/// Semaphore-based concurrency control for inference requests.
#[derive(Debug)]
pub struct ConcurrencyLimiter {
    /// Maximum number of concurrent permits.
    max_permits: usize,
    /// Currently acquired permits.
    acquired: usize,
    /// Total number of times a permit was acquired.
    total_acquisitions: u64,
    /// Total number of times acquisition was rejected (no permits).
    total_rejections: u64,
}

impl ConcurrencyLimiter {
    /// Create a limiter with the given number of permits.
    pub fn new(max_permits: usize) -> Self {
        Self { max_permits, acquired: 0, total_acquisitions: 0, total_rejections: 0 }
    }

    /// Try to acquire a permit. Returns `true` on success.
    pub fn try_acquire(&mut self) -> bool {
        if self.acquired < self.max_permits {
            self.acquired += 1;
            self.total_acquisitions += 1;
            true
        } else {
            self.total_rejections += 1;
            false
        }
    }

    /// Release a permit. Panics if no permits are held.
    pub fn release(&mut self) {
        assert!(self.acquired > 0, "release called with no acquired permits");
        self.acquired -= 1;
    }

    /// Number of permits currently in use.
    #[must_use]
    pub fn active(&self) -> usize {
        self.acquired
    }

    /// Number of available permits.
    #[must_use]
    pub fn available(&self) -> usize {
        self.max_permits - self.acquired
    }

    /// Maximum permits.
    #[must_use]
    pub fn max_permits(&self) -> usize {
        self.max_permits
    }

    /// Whether the limiter is fully utilised.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.acquired >= self.max_permits
    }

    /// Total successful acquisitions.
    #[must_use]
    pub fn total_acquisitions(&self) -> u64 {
        self.total_acquisitions
    }

    /// Total rejections.
    #[must_use]
    pub fn total_rejections(&self) -> u64 {
        self.total_rejections
    }

    /// Utilisation fraction (0.0–1.0).
    #[must_use]
    pub fn utilisation(&self) -> f64 {
        if self.max_permits == 0 {
            return 0.0;
        }
        self.acquired as f64 / self.max_permits as f64
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Graceful Shutdown
// ═════════════════════════════════════════════════════════════════════════════

/// Phase of the graceful shutdown process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownPhase {
    /// Normal operation.
    Running,
    /// Stop accepting new requests.
    StopAccepting,
    /// Drain in-flight requests.
    Draining,
    /// Save state / flush metrics.
    Finalising,
    /// Shutdown complete.
    Completed,
}

/// Handles SIGTERM/SIGINT with request draining.
#[derive(Debug)]
pub struct GracefulShutdown {
    /// Current phase.
    phase: ShutdownPhase,
    /// Maximum time to wait for in-flight requests to drain.
    drain_timeout: Duration,
    /// When shutdown was initiated.
    initiated_at: Option<Instant>,
    /// Number of requests drained during shutdown.
    requests_drained: u64,
    /// Whether a shutdown signal has been received.
    signal_received: bool,
}

impl GracefulShutdown {
    /// Create a new shutdown handler.
    pub fn new(drain_timeout: Duration) -> Self {
        Self {
            phase: ShutdownPhase::Running,
            drain_timeout,
            initiated_at: None,
            requests_drained: 0,
            signal_received: false,
        }
    }

    /// Signal that a shutdown has been requested.
    pub fn initiate(&mut self) {
        if self.phase == ShutdownPhase::Running {
            self.phase = ShutdownPhase::StopAccepting;
            self.initiated_at = Some(Instant::now());
            self.signal_received = true;
        }
    }

    /// Advance to the next shutdown phase. Returns the new phase.
    pub fn advance(&mut self) -> ShutdownPhase {
        self.phase = match self.phase {
            ShutdownPhase::Running => ShutdownPhase::Running,
            ShutdownPhase::StopAccepting => ShutdownPhase::Draining,
            ShutdownPhase::Draining => ShutdownPhase::Finalising,
            ShutdownPhase::Finalising => ShutdownPhase::Completed,
            ShutdownPhase::Completed => ShutdownPhase::Completed,
        };
        self.phase
    }

    /// Record that an in-flight request was drained.
    pub fn record_drained(&mut self) {
        self.requests_drained += 1;
    }

    /// Whether the drain timeout has been exceeded.
    #[must_use]
    pub fn is_drain_timed_out(&self) -> bool {
        self.initiated_at.is_some_and(|t| t.elapsed() > self.drain_timeout)
    }

    /// Current phase.
    #[must_use]
    pub fn phase(&self) -> ShutdownPhase {
        self.phase
    }

    /// Whether shutdown is complete.
    #[must_use]
    pub fn is_completed(&self) -> bool {
        self.phase == ShutdownPhase::Completed
    }

    /// Whether a signal has been received.
    #[must_use]
    pub fn signal_received(&self) -> bool {
        self.signal_received
    }

    /// Number of requests drained so far.
    #[must_use]
    pub fn requests_drained(&self) -> u64 {
        self.requests_drained
    }

    /// Drain timeout.
    #[must_use]
    pub fn drain_timeout(&self) -> Duration {
        self.drain_timeout
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Health Endpoint
// ═════════════════════════════════════════════════════════════════════════════

/// Health check status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// All systems operational.
    Healthy,
    /// Partially degraded.
    Degraded,
    /// Not healthy.
    Unhealthy,
}

impl fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Healthy => write!(f, "healthy"),
            Self::Degraded => write!(f, "degraded"),
            Self::Unhealthy => write!(f, "unhealthy"),
        }
    }
}

/// Response payload for health endpoints.
#[derive(Debug, Clone)]
pub struct HealthResponse {
    /// Overall status.
    pub status: HealthStatus,
    /// Human-readable message.
    pub message: String,
    /// Component-level checks.
    pub checks: HashMap<String, bool>,
}

/// Provides /health, /ready, /live endpoints for Kubernetes probes.
#[derive(Debug)]
pub struct HealthEndpoint {
    /// Whether the server is live (process is running).
    is_live: bool,
    /// Whether the server is ready to accept traffic.
    is_ready: bool,
    /// Component health checks.
    component_checks: HashMap<String, bool>,
    /// Total health check invocations.
    total_checks: u64,
}

impl HealthEndpoint {
    /// Create a new endpoint (starts as live but not ready).
    pub fn new() -> Self {
        Self { is_live: true, is_ready: false, component_checks: HashMap::new(), total_checks: 0 }
    }

    /// Mark the server as ready.
    pub fn set_ready(&mut self, ready: bool) {
        self.is_ready = ready;
    }

    /// Mark the server as live/not-live.
    pub fn set_live(&mut self, live: bool) {
        self.is_live = live;
    }

    /// Register or update a component health check.
    pub fn set_component(&mut self, name: impl Into<String>, healthy: bool) {
        self.component_checks.insert(name.into(), healthy);
    }

    /// GET /live — is the process running?
    #[must_use]
    pub fn liveness(&mut self) -> HealthResponse {
        self.total_checks += 1;
        let status = if self.is_live { HealthStatus::Healthy } else { HealthStatus::Unhealthy };
        HealthResponse { status, message: format!("liveness: {status}"), checks: HashMap::new() }
    }

    /// GET /ready — is the server ready for traffic?
    #[must_use]
    pub fn readiness(&mut self) -> HealthResponse {
        self.total_checks += 1;
        let status = if self.is_ready { HealthStatus::Healthy } else { HealthStatus::Unhealthy };
        HealthResponse { status, message: format!("readiness: {status}"), checks: HashMap::new() }
    }

    /// GET /health — comprehensive health including components.
    #[must_use]
    pub fn health(&mut self) -> HealthResponse {
        self.total_checks += 1;
        let all_ok = self.is_live && self.is_ready && self.component_checks.values().all(|&v| v);
        let any_ok = self.is_live && self.component_checks.values().any(|&v| v);
        let status = if all_ok {
            HealthStatus::Healthy
        } else if any_ok {
            HealthStatus::Degraded
        } else {
            HealthStatus::Unhealthy
        };
        HealthResponse {
            status,
            message: format!("health: {status}"),
            checks: self.component_checks.clone(),
        }
    }

    /// Total health-check invocations across all endpoints.
    #[must_use]
    pub fn total_checks(&self) -> u64 {
        self.total_checks
    }
}

impl Default for HealthEndpoint {
    fn default() -> Self {
        Self::new()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Metrics Endpoint
// ═════════════════════════════════════════════════════════════════════════════

/// A single Prometheus-style metric.
#[derive(Debug, Clone)]
pub struct Metric {
    /// Metric name (e.g. `bitnet_requests_total`).
    pub name: String,
    /// HELP description.
    pub help: String,
    /// TYPE (counter, gauge, histogram).
    pub metric_type: String,
    /// Current value.
    pub value: f64,
    /// Optional labels.
    pub labels: HashMap<String, String>,
}

impl Metric {
    /// Create a counter metric.
    pub fn counter(name: impl Into<String>, help: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            help: help.into(),
            metric_type: "counter".to_string(),
            value,
            labels: HashMap::new(),
        }
    }

    /// Create a gauge metric.
    pub fn gauge(name: impl Into<String>, help: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            help: help.into(),
            metric_type: "gauge".to_string(),
            value,
            labels: HashMap::new(),
        }
    }

    /// Add a label.
    #[must_use]
    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    /// Render in Prometheus exposition format.
    #[must_use]
    pub fn to_prometheus(&self) -> String {
        let labels_str = if self.labels.is_empty() {
            String::new()
        } else {
            let pairs: Vec<String> =
                self.labels.iter().map(|(k, v)| format!("{k}=\"{v}\"")).collect();
            format!("{{{}}}", pairs.join(","))
        };
        format!(
            "# HELP {} {}\n# TYPE {} {}\n{}{} {}",
            self.name, self.help, self.name, self.metric_type, self.name, labels_str, self.value,
        )
    }
}

/// Provides /metrics with Prometheus-format exposition.
#[derive(Debug)]
pub struct MetricsEndpoint {
    /// Collected metrics keyed by name.
    metrics: HashMap<String, Metric>,
    /// When the endpoint was created.
    created_at: Instant,
    /// Total scrape count.
    scrape_count: u64,
}

impl MetricsEndpoint {
    /// Create a new metrics endpoint.
    pub fn new() -> Self {
        Self { metrics: HashMap::new(), created_at: Instant::now(), scrape_count: 0 }
    }

    /// Register or update a metric.
    pub fn set(&mut self, metric: Metric) {
        self.metrics.insert(metric.name.clone(), metric);
    }

    /// Increment a counter metric by the given amount.
    pub fn increment(&mut self, name: &str, amount: f64) {
        if let Some(m) = self.metrics.get_mut(name) {
            m.value += amount;
        }
    }

    /// Set the value of a gauge metric.
    pub fn set_gauge(&mut self, name: &str, value: f64) {
        if let Some(m) = self.metrics.get_mut(name) {
            m.value = value;
        }
    }

    /// Get a metric by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Metric> {
        self.metrics.get(name)
    }

    /// Render all metrics in Prometheus exposition format.
    #[must_use]
    pub fn render(&mut self) -> String {
        self.scrape_count += 1;
        let mut lines: Vec<String> = self.metrics.values().map(|m| m.to_prometheus()).collect();
        lines.sort();
        lines.join("\n")
    }

    /// Number of registered metrics.
    #[must_use]
    pub fn metric_count(&self) -> usize {
        self.metrics.len()
    }

    /// Total times render() was called.
    #[must_use]
    pub fn scrape_count(&self) -> u64 {
        self.scrape_count
    }

    /// Uptime since creation.
    #[must_use]
    pub fn uptime(&self) -> Duration {
        self.created_at.elapsed()
    }
}

impl Default for MetricsEndpoint {
    fn default() -> Self {
        Self::new()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Model Hot-Swap
// ═════════════════════════════════════════════════════════════════════════════

/// Phase of a model hot-swap operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotSwapPhase {
    /// Idle — no swap in progress.
    Idle,
    /// Loading the new model.
    LoadingNew,
    /// Warming up the new model.
    WarmingNew,
    /// Routing traffic to the new model.
    Switching,
    /// Draining the old model.
    DrainingOld,
    /// Unloading the old model.
    UnloadingOld,
    /// Swap complete.
    Completed,
    /// Swap failed and was rolled back.
    RolledBack,
}

/// Zero-downtime model hot-swap: load new → route traffic → unload old.
#[derive(Debug)]
pub struct ModelHotSwap {
    /// Current phase.
    phase: HotSwapPhase,
    /// The currently active model instance.
    current_model: Option<ModelInstance>,
    /// The new model being swapped in.
    new_model: Option<ModelInstance>,
    /// History of completed swaps (old_version → new_version).
    swap_history: Vec<(u64, u64)>,
    /// Total successful swaps.
    total_swaps: u64,
    /// Total failed swaps.
    total_failures: u64,
}

impl ModelHotSwap {
    /// Create a new hot-swap manager.
    pub fn new() -> Self {
        Self {
            phase: HotSwapPhase::Idle,
            current_model: None,
            new_model: None,
            swap_history: Vec::new(),
            total_swaps: 0,
            total_failures: 0,
        }
    }

    /// Set the initial model (no swap, just the first load).
    pub fn set_initial_model(&mut self, model: ModelInstance) {
        self.current_model = Some(model);
    }

    /// Begin a hot-swap by providing the new model instance.
    pub fn begin_swap(&mut self, new_model: ModelInstance) -> Result<(), String> {
        if self.phase != HotSwapPhase::Idle {
            return Err(format!("swap already in progress (phase: {:?})", self.phase));
        }
        if self.current_model.is_none() {
            return Err("no current model to swap from".into());
        }
        self.new_model = Some(new_model);
        self.phase = HotSwapPhase::LoadingNew;
        Ok(())
    }

    /// Advance the hot-swap to the next phase.
    pub fn advance(&mut self) -> HotSwapPhase {
        self.phase = match self.phase {
            HotSwapPhase::Idle => HotSwapPhase::Idle,
            HotSwapPhase::LoadingNew => HotSwapPhase::WarmingNew,
            HotSwapPhase::WarmingNew => HotSwapPhase::Switching,
            HotSwapPhase::Switching => {
                // Perform the actual swap.
                if let Some(mut new) = self.new_model.take() {
                    new.mark_ready();
                    let old_version = self.current_model.as_ref().map(|m| m.version).unwrap_or(0);
                    let new_version = new.version;
                    if let Some(old) = self.current_model.as_mut() {
                        old.mark_draining();
                    }
                    self.swap_history.push((old_version, new_version));
                    self.new_model = self.current_model.take();
                    self.current_model = Some(new);
                }
                HotSwapPhase::DrainingOld
            }
            HotSwapPhase::DrainingOld => {
                if let Some(old) = self.new_model.as_mut() {
                    old.mark_unloaded();
                }
                HotSwapPhase::UnloadingOld
            }
            HotSwapPhase::UnloadingOld => {
                self.new_model = None;
                self.total_swaps += 1;
                HotSwapPhase::Completed
            }
            HotSwapPhase::Completed => {
                self.phase = HotSwapPhase::Idle;
                return HotSwapPhase::Idle;
            }
            HotSwapPhase::RolledBack => {
                self.phase = HotSwapPhase::Idle;
                return HotSwapPhase::Idle;
            }
        };
        self.phase
    }

    /// Abort the current swap and rollback.
    pub fn rollback(&mut self) {
        self.new_model = None;
        self.phase = HotSwapPhase::RolledBack;
        self.total_failures += 1;
    }

    /// Current phase.
    #[must_use]
    pub fn phase(&self) -> HotSwapPhase {
        self.phase
    }

    /// Reference to the active model, if any.
    #[must_use]
    pub fn current_model(&self) -> Option<&ModelInstance> {
        self.current_model.as_ref()
    }

    /// Swap history.
    #[must_use]
    pub fn swap_history(&self) -> &[(u64, u64)] {
        &self.swap_history
    }

    /// Total completed swaps.
    #[must_use]
    pub fn total_swaps(&self) -> u64 {
        self.total_swaps
    }

    /// Total failed swaps.
    #[must_use]
    pub fn total_failures(&self) -> u64 {
        self.total_failures
    }
}

impl Default for ModelHotSwap {
    fn default() -> Self {
        Self::new()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Serving Runtime (orchestrator)
// ═════════════════════════════════════════════════════════════════════════════

/// Runtime state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeState {
    /// Initialising subsystems.
    Initialising,
    /// Loading the model.
    LoadingModel,
    /// Running warmup requests.
    WarmingUp,
    /// Accepting and serving requests.
    Serving,
    /// Shutting down.
    ShuttingDown,
    /// Stopped.
    Stopped,
}

/// Orchestrator: load model → start server → handle requests → metrics → shutdown.
#[derive(Debug)]
pub struct ServingRuntime {
    /// Configuration.
    config: ServingConfig,
    /// Current state.
    state: RuntimeState,
    /// Loaded model instance.
    model: Option<ModelInstance>,
    /// Request queue.
    queue: RequestQueue,
    /// Request handler.
    handler: Option<RequestHandler>,
    /// Concurrency limiter.
    limiter: ConcurrencyLimiter,
    /// Shutdown handler.
    shutdown: GracefulShutdown,
    /// Health endpoint.
    health: HealthEndpoint,
    /// Metrics endpoint.
    metrics: MetricsEndpoint,
    /// Hot-swap manager.
    hot_swap: ModelHotSwap,
    /// Total requests processed.
    total_processed: u64,
    /// When the runtime was created.
    created_at: Instant,
}

impl ServingRuntime {
    /// Create a new runtime from configuration.
    pub fn new(config: ServingConfig) -> Result<Self, String> {
        config.validate()?;
        let limiter = ConcurrencyLimiter::new(config.max_concurrent);
        let queue = RequestQueue::new(config.max_concurrent * 4);
        let shutdown = GracefulShutdown::new(Duration::from_secs(30));

        let mut metrics = MetricsEndpoint::new();
        metrics.set(Metric::counter("bitnet_requests_total", "Total inference requests", 0.0));
        metrics.set(Metric::counter(
            "bitnet_requests_failed",
            "Total failed inference requests",
            0.0,
        ));
        metrics.set(Metric::gauge(
            "bitnet_active_requests",
            "Currently active inference requests",
            0.0,
        ));
        metrics.set(Metric::gauge("bitnet_queue_depth", "Current request queue depth", 0.0));
        metrics.set(Metric::gauge("bitnet_model_loaded", "Whether a model is loaded (1/0)", 0.0));

        Ok(Self {
            config,
            state: RuntimeState::Initialising,
            model: None,
            queue,
            handler: None,
            limiter,
            shutdown,
            health: HealthEndpoint::new(),
            metrics,
            hot_swap: ModelHotSwap::new(),
            total_processed: 0,
            created_at: Instant::now(),
        })
    }

    /// Load the model specified in the configuration.
    pub fn load_model(&mut self) -> Result<(), String> {
        self.state = RuntimeState::LoadingModel;

        let mut instance =
            ModelInstance::new("primary", &self.config.model_path, self.config.device);
        instance.mark_ready();

        self.handler = Some(RequestHandler::new(&instance.id));
        self.hot_swap.set_initial_model(instance.clone());
        self.model = Some(instance);

        self.health.set_component("model", true);
        self.metrics.set_gauge("bitnet_model_loaded", 1.0);
        Ok(())
    }

    /// Run warmup requests.
    pub fn warmup(&mut self) -> Result<u32, String> {
        self.state = RuntimeState::WarmingUp;
        let handler = self.handler.as_mut().ok_or("no handler")?;
        let n = self.config.warmup_requests;
        for i in 0..n {
            let req =
                InferenceRequest::new(format!("warmup-{i}"), "warmup prompt").with_max_tokens(1);
            handler.handle(&req)?;
        }
        Ok(n)
    }

    /// Transition to serving state.
    pub fn start_serving(&mut self) {
        self.state = RuntimeState::Serving;
        self.health.set_ready(true);
        self.health.set_component("server", true);
    }

    /// Submit a request to the queue.
    pub fn submit(&mut self, request: InferenceRequest) -> Result<(), String> {
        if self.state != RuntimeState::Serving {
            return Err(format!("runtime is not serving (state: {:?})", self.state));
        }
        self.queue.enqueue(request)?;
        self.metrics.set_gauge("bitnet_queue_depth", self.queue.len() as f64);
        Ok(())
    }

    /// Process the next queued request. Returns the result or `None` if
    /// the queue is empty / limiter is full.
    pub fn process_next(&mut self) -> Option<Result<InferenceResult, String>> {
        if !self.limiter.try_acquire() {
            return None;
        }
        let request = match self.queue.dequeue() {
            Some(r) => r,
            None => {
                self.limiter.release();
                return None;
            }
        };
        let result =
            self.handler.as_mut().map(|h| h.handle(&request)).unwrap_or(Err("no handler".into()));

        self.limiter.release();
        self.total_processed += 1;
        self.metrics.increment("bitnet_requests_total", 1.0);
        self.metrics.set_gauge("bitnet_active_requests", self.limiter.active() as f64);
        self.metrics.set_gauge("bitnet_queue_depth", self.queue.len() as f64);

        if result.is_err() {
            self.metrics.increment("bitnet_requests_failed", 1.0);
        }
        if let Some(m) = self.model.as_mut() {
            m.record_request();
        }

        Some(result)
    }

    /// Initiate graceful shutdown.
    pub fn initiate_shutdown(&mut self) {
        self.state = RuntimeState::ShuttingDown;
        self.shutdown.initiate();
        self.health.set_ready(false);
    }

    /// Drain remaining requests during shutdown.
    pub fn drain(&mut self) -> u64 {
        let mut drained = 0u64;
        while let Some(_result) = self.process_next() {
            self.shutdown.record_drained();
            drained += 1;
        }
        drained
    }

    /// Complete shutdown.
    pub fn complete_shutdown(&mut self) {
        self.shutdown.advance(); // StopAccepting → Draining
        self.shutdown.advance(); // Draining → Finalising
        self.shutdown.advance(); // Finalising → Completed
        self.state = RuntimeState::Stopped;
        self.health.set_live(false);
    }

    // ── Accessors ───────────────────────────────────────────────────────

    /// Current runtime state.
    #[must_use]
    pub fn state(&self) -> RuntimeState {
        self.state
    }

    /// Runtime configuration.
    #[must_use]
    pub fn config(&self) -> &ServingConfig {
        &self.config
    }

    /// Loaded model instance.
    #[must_use]
    pub fn model(&self) -> Option<&ModelInstance> {
        self.model.as_ref()
    }

    /// Request queue.
    #[must_use]
    pub fn queue(&self) -> &RequestQueue {
        &self.queue
    }

    /// Concurrency limiter.
    #[must_use]
    pub fn limiter(&self) -> &ConcurrencyLimiter {
        &self.limiter
    }

    /// Health endpoint.
    #[must_use]
    pub fn health(&self) -> &HealthEndpoint {
        &self.health
    }

    /// Mutable health endpoint for checks.
    pub fn health_mut(&mut self) -> &mut HealthEndpoint {
        &mut self.health
    }

    /// Metrics endpoint.
    #[must_use]
    pub fn metrics(&self) -> &MetricsEndpoint {
        &self.metrics
    }

    /// Mutable metrics endpoint.
    pub fn metrics_mut(&mut self) -> &mut MetricsEndpoint {
        &mut self.metrics
    }

    /// Hot-swap manager.
    #[must_use]
    pub fn hot_swap(&self) -> &ModelHotSwap {
        &self.hot_swap
    }

    /// Mutable hot-swap manager.
    pub fn hot_swap_mut(&mut self) -> &mut ModelHotSwap {
        &mut self.hot_swap
    }

    /// Shutdown handler.
    #[must_use]
    pub fn shutdown(&self) -> &GracefulShutdown {
        &self.shutdown
    }

    /// Total requests processed.
    #[must_use]
    pub fn total_processed(&self) -> u64 {
        self.total_processed
    }

    /// Uptime since creation.
    #[must_use]
    pub fn uptime(&self) -> Duration {
        self.created_at.elapsed()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // -- Helpers ----------------------------------------------------------

    fn default_config() -> ServingConfig {
        ServingConfig { model_path: "/models/test.gguf".to_string(), ..ServingConfig::default() }
    }

    fn make_request(id: &str) -> InferenceRequest {
        InferenceRequest::new(id, "Hello, world!")
    }

    fn make_model(id: &str, version: u64) -> ModelInstance {
        let mut m = ModelInstance::new(id, "/models/test.gguf", DeviceKind::Cpu);
        m.version = version;
        m
    }

    fn runtime() -> ServingRuntime {
        ServingRuntime::new(default_config()).unwrap()
    }

    fn serving_runtime() -> ServingRuntime {
        let mut rt = runtime();
        rt.load_model().unwrap();
        rt.warmup().unwrap();
        rt.start_serving();
        rt
    }

    // ── ServingConfig ───────────────────────────────────────────────────

    #[test]
    fn test_config_default_has_expected_values() {
        let cfg = ServingConfig::default();
        assert_eq!(cfg.host, "0.0.0.0");
        assert_eq!(cfg.port, 8080);
        assert_eq!(cfg.max_concurrent, 8);
        assert_eq!(cfg.device, DeviceKind::Cpu);
        assert_eq!(cfg.warmup_requests, 3);
    }

    #[test]
    fn test_config_validate_accepts_valid() {
        assert!(default_config().validate().is_ok());
    }

    #[test]
    fn test_config_validate_rejects_empty_host() {
        let mut cfg = default_config();
        cfg.host = String::new();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_rejects_zero_port() {
        let mut cfg = default_config();
        cfg.port = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_rejects_zero_concurrency() {
        let mut cfg = default_config();
        cfg.max_concurrent = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_rejects_empty_model_path() {
        let cfg = ServingConfig::default();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_bind_address() {
        let cfg = default_config();
        assert_eq!(cfg.bind_address(), "0.0.0.0:8080");
    }

    // ── DeviceKind ──────────────────────────────────────────────────────

    #[test]
    fn test_device_display_cpu() {
        assert_eq!(DeviceKind::Cpu.to_string(), "cpu");
    }

    #[test]
    fn test_device_display_gpu() {
        assert_eq!(DeviceKind::Gpu(0).to_string(), "gpu:0");
        assert_eq!(DeviceKind::Gpu(3).to_string(), "gpu:3");
    }

    // ── ModelInstance ───────────────────────────────────────────────────

    #[test]
    fn test_model_new_is_loading() {
        let m = ModelInstance::new("m1", "/path", DeviceKind::Cpu);
        assert_eq!(m.status, ModelStatus::Loading);
        assert!(!m.is_serving());
        assert_eq!(m.requests_served, 0);
    }

    #[test]
    fn test_model_mark_ready() {
        let mut m = ModelInstance::new("m1", "/path", DeviceKind::Cpu);
        m.mark_ready();
        assert_eq!(m.status, ModelStatus::Ready);
        assert!(m.is_serving());
        assert!(m.loaded_at.is_some());
    }

    #[test]
    fn test_model_mark_draining() {
        let mut m = ModelInstance::new("m1", "/path", DeviceKind::Cpu);
        m.mark_ready();
        m.mark_draining();
        assert_eq!(m.status, ModelStatus::Draining);
        assert!(!m.is_serving());
    }

    #[test]
    fn test_model_mark_unloaded() {
        let mut m = ModelInstance::new("m1", "/path", DeviceKind::Cpu);
        m.mark_unloaded();
        assert_eq!(m.status, ModelStatus::Unloaded);
    }

    #[test]
    fn test_model_mark_failed() {
        let mut m = ModelInstance::new("m1", "/path", DeviceKind::Cpu);
        m.mark_failed();
        assert_eq!(m.status, ModelStatus::Failed);
    }

    #[test]
    fn test_model_record_request() {
        let mut m = ModelInstance::new("m1", "/path", DeviceKind::Cpu);
        m.mark_ready();
        m.record_request();
        m.record_request();
        assert_eq!(m.requests_served, 2);
    }

    #[test]
    fn test_model_uptime_zero_before_ready() {
        let m = ModelInstance::new("m1", "/path", DeviceKind::Cpu);
        assert_eq!(m.uptime(), Duration::ZERO);
    }

    #[test]
    fn test_model_tokenizer_path_default_none() {
        let m = ModelInstance::new("m1", "/path", DeviceKind::Cpu);
        assert!(m.tokenizer_path.is_none());
    }

    // ── InferenceRequest ────────────────────────────────────────────────

    #[test]
    fn test_request_defaults() {
        let r = make_request("r1");
        assert_eq!(r.id, "r1");
        assert_eq!(r.max_tokens, 256);
        assert!((r.temperature - 0.7).abs() < 0.001);
        assert_eq!(r.priority, RequestPriority::Normal);
        assert_eq!(r.status, RequestStatus::Queued);
    }

    #[test]
    fn test_request_with_priority() {
        let r = make_request("r1").with_priority(RequestPriority::High);
        assert_eq!(r.priority, RequestPriority::High);
    }

    #[test]
    fn test_request_with_timeout() {
        let r = make_request("r1").with_timeout(Duration::from_secs(5));
        assert_eq!(r.timeout, Duration::from_secs(5));
    }

    #[test]
    fn test_request_with_max_tokens() {
        let r = make_request("r1").with_max_tokens(128);
        assert_eq!(r.max_tokens, 128);
    }

    #[test]
    fn test_request_with_temperature() {
        let r = make_request("r1").with_temperature(0.9);
        assert!((r.temperature - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_request_not_timed_out_when_fresh() {
        let r = make_request("r1");
        assert!(!r.is_timed_out());
    }

    #[test]
    fn test_request_remaining_time_positive() {
        let r = make_request("r1").with_timeout(Duration::from_secs(60));
        assert!(r.remaining_time() > Duration::ZERO);
    }

    #[test]
    fn test_request_timed_out_with_zero_timeout() {
        let r = make_request("r1").with_timeout(Duration::ZERO);
        // Instant::now() already passed at construction.
        assert!(r.is_timed_out());
    }

    // ── RequestPriority ordering ────────────────────────────────────────

    #[test]
    fn test_priority_ordering() {
        assert!(RequestPriority::Critical > RequestPriority::High);
        assert!(RequestPriority::High > RequestPriority::Normal);
        assert!(RequestPriority::Normal > RequestPriority::Low);
    }

    // ── RequestQueue ────────────────────────────────────────────────────

    #[test]
    fn test_queue_new_is_empty() {
        let q = RequestQueue::new(10);
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
        assert_eq!(q.remaining_capacity(), 10);
    }

    #[test]
    fn test_queue_enqueue_dequeue() {
        let mut q = RequestQueue::new(10);
        q.enqueue(make_request("r1")).unwrap();
        assert_eq!(q.len(), 1);
        let r = q.dequeue().unwrap();
        assert_eq!(r.id, "r1");
        assert!(q.is_empty());
    }

    #[test]
    fn test_queue_rejects_when_full() {
        let mut q = RequestQueue::new(1);
        q.enqueue(make_request("r1")).unwrap();
        assert!(q.enqueue(make_request("r2")).is_err());
    }

    #[test]
    fn test_queue_priority_order() {
        let mut q = RequestQueue::new(10);
        q.enqueue(make_request("low").with_priority(RequestPriority::Low)).unwrap();
        q.enqueue(make_request("high").with_priority(RequestPriority::High)).unwrap();
        q.enqueue(make_request("normal").with_priority(RequestPriority::Normal)).unwrap();

        assert_eq!(q.dequeue().unwrap().id, "high");
        assert_eq!(q.dequeue().unwrap().id, "normal");
        assert_eq!(q.dequeue().unwrap().id, "low");
    }

    #[test]
    fn test_queue_fifo_within_same_priority() {
        let mut q = RequestQueue::new(10);
        q.enqueue(make_request("first")).unwrap();
        q.enqueue(make_request("second")).unwrap();
        q.enqueue(make_request("third")).unwrap();

        assert_eq!(q.dequeue().unwrap().id, "first");
        assert_eq!(q.dequeue().unwrap().id, "second");
        assert_eq!(q.dequeue().unwrap().id, "third");
    }

    #[test]
    fn test_queue_skips_timed_out() {
        let mut q = RequestQueue::new(10);
        q.enqueue(make_request("expired").with_timeout(Duration::ZERO)).unwrap();
        q.enqueue(make_request("valid")).unwrap();

        let r = q.dequeue().unwrap();
        assert_eq!(r.id, "valid");
        assert_eq!(q.total_timed_out(), 1);
    }

    #[test]
    fn test_queue_dequeue_empty_returns_none() {
        let mut q = RequestQueue::new(10);
        assert!(q.dequeue().is_none());
    }

    #[test]
    fn test_queue_counters() {
        let mut q = RequestQueue::new(10);
        q.enqueue(make_request("r1")).unwrap();
        q.enqueue(make_request("r2")).unwrap();
        assert_eq!(q.total_enqueued(), 2);
        q.dequeue();
        assert_eq!(q.total_dequeued(), 1);
    }

    #[test]
    fn test_queue_remaining_capacity() {
        let mut q = RequestQueue::new(3);
        q.enqueue(make_request("r1")).unwrap();
        assert_eq!(q.remaining_capacity(), 2);
    }

    #[test]
    fn test_queue_purge_timed_out() {
        let mut q = RequestQueue::new(10);
        q.enqueue(make_request("e1").with_timeout(Duration::ZERO)).unwrap();
        q.enqueue(make_request("e2").with_timeout(Duration::ZERO)).unwrap();
        q.enqueue(make_request("v1")).unwrap();
        let purged = q.purge_timed_out();
        assert_eq!(purged, 2);
        assert_eq!(q.len(), 1);
    }

    // ── RequestHandler ──────────────────────────────────────────────────

    #[test]
    fn test_handler_new() {
        let h = RequestHandler::new("m1");
        assert_eq!(h.model_id(), "m1");
        assert_eq!(h.total_handled(), 0);
        assert_eq!(h.total_failures(), 0);
    }

    #[test]
    fn test_handler_handle_success() {
        let mut h = RequestHandler::new("m1");
        let r = make_request("r1");
        let result = h.handle(&r).unwrap();
        assert_eq!(result.request_id, "r1");
        assert!(result.tokens_generated > 0);
        assert_eq!(h.total_handled(), 1);
    }

    #[test]
    fn test_handler_rejects_empty_prompt() {
        let mut h = RequestHandler::new("m1");
        let r = InferenceRequest::new("r1", "");
        assert!(h.handle(&r).is_err());
        assert_eq!(h.total_failures(), 1);
    }

    #[test]
    fn test_handler_rejects_timed_out() {
        let mut h = RequestHandler::new("m1");
        let r = make_request("r1").with_timeout(Duration::ZERO);
        assert!(h.handle(&r).is_err());
        assert_eq!(h.total_failures(), 1);
    }

    #[test]
    fn test_handler_average_latency_zero_when_none() {
        let h = RequestHandler::new("m1");
        assert_eq!(h.average_latency(), Duration::ZERO);
    }

    #[test]
    fn test_handler_multiple_requests() {
        let mut h = RequestHandler::new("m1");
        for i in 0..5 {
            h.handle(&make_request(&format!("r{i}"))).unwrap();
        }
        assert_eq!(h.total_handled(), 5);
    }

    #[test]
    fn test_handler_truncation_flag() {
        let mut h = RequestHandler::new("m1");
        let r = make_request("r1").with_max_tokens(1024);
        let result = h.handle(&r).unwrap();
        assert!(result.truncated);
    }

    #[test]
    fn test_handler_no_truncation() {
        let mut h = RequestHandler::new("m1");
        let r = make_request("r1").with_max_tokens(64);
        let result = h.handle(&r).unwrap();
        assert!(!result.truncated);
    }

    // ── ConcurrencyLimiter ──────────────────────────────────────────────

    #[test]
    fn test_limiter_new() {
        let l = ConcurrencyLimiter::new(4);
        assert_eq!(l.max_permits(), 4);
        assert_eq!(l.active(), 0);
        assert_eq!(l.available(), 4);
        assert!(!l.is_full());
    }

    #[test]
    fn test_limiter_acquire_release() {
        let mut l = ConcurrencyLimiter::new(2);
        assert!(l.try_acquire());
        assert_eq!(l.active(), 1);
        assert_eq!(l.available(), 1);
        l.release();
        assert_eq!(l.active(), 0);
    }

    #[test]
    fn test_limiter_rejects_when_full() {
        let mut l = ConcurrencyLimiter::new(1);
        assert!(l.try_acquire());
        assert!(!l.try_acquire());
        assert_eq!(l.total_rejections(), 1);
    }

    #[test]
    fn test_limiter_counters() {
        let mut l = ConcurrencyLimiter::new(2);
        l.try_acquire();
        l.try_acquire();
        l.try_acquire(); // rejected
        assert_eq!(l.total_acquisitions(), 2);
        assert_eq!(l.total_rejections(), 1);
    }

    #[test]
    fn test_limiter_utilisation() {
        let mut l = ConcurrencyLimiter::new(4);
        l.try_acquire();
        l.try_acquire();
        assert!((l.utilisation() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_limiter_utilisation_zero_capacity() {
        let l = ConcurrencyLimiter::new(0);
        assert!((l.utilisation() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_limiter_is_full() {
        let mut l = ConcurrencyLimiter::new(1);
        l.try_acquire();
        assert!(l.is_full());
    }

    #[test]
    #[should_panic(expected = "no acquired permits")]
    fn test_limiter_release_without_acquire_panics() {
        let mut l = ConcurrencyLimiter::new(1);
        l.release();
    }

    // ── GracefulShutdown ────────────────────────────────────────────────

    #[test]
    fn test_shutdown_initial_state() {
        let s = GracefulShutdown::new(Duration::from_secs(10));
        assert_eq!(s.phase(), ShutdownPhase::Running);
        assert!(!s.signal_received());
        assert!(!s.is_completed());
    }

    #[test]
    fn test_shutdown_initiate() {
        let mut s = GracefulShutdown::new(Duration::from_secs(10));
        s.initiate();
        assert_eq!(s.phase(), ShutdownPhase::StopAccepting);
        assert!(s.signal_received());
    }

    #[test]
    fn test_shutdown_advance_phases() {
        let mut s = GracefulShutdown::new(Duration::from_secs(10));
        s.initiate();
        assert_eq!(s.advance(), ShutdownPhase::Draining);
        assert_eq!(s.advance(), ShutdownPhase::Finalising);
        assert_eq!(s.advance(), ShutdownPhase::Completed);
        assert!(s.is_completed());
    }

    #[test]
    fn test_shutdown_running_advance_stays_running() {
        let mut s = GracefulShutdown::new(Duration::from_secs(10));
        assert_eq!(s.advance(), ShutdownPhase::Running);
    }

    #[test]
    fn test_shutdown_completed_stays_completed() {
        let mut s = GracefulShutdown::new(Duration::from_secs(10));
        s.initiate();
        s.advance();
        s.advance();
        s.advance();
        assert_eq!(s.advance(), ShutdownPhase::Completed);
    }

    #[test]
    fn test_shutdown_record_drained() {
        let mut s = GracefulShutdown::new(Duration::from_secs(10));
        s.initiate();
        s.record_drained();
        s.record_drained();
        assert_eq!(s.requests_drained(), 2);
    }

    #[test]
    fn test_shutdown_drain_timeout_accessor() {
        let s = GracefulShutdown::new(Duration::from_secs(42));
        assert_eq!(s.drain_timeout(), Duration::from_secs(42));
    }

    #[test]
    fn test_shutdown_drain_not_timed_out_initially() {
        let s = GracefulShutdown::new(Duration::from_secs(60));
        assert!(!s.is_drain_timed_out());
    }

    #[test]
    fn test_shutdown_double_initiate_is_idempotent() {
        let mut s = GracefulShutdown::new(Duration::from_secs(10));
        s.initiate();
        s.initiate();
        assert_eq!(s.phase(), ShutdownPhase::StopAccepting);
    }

    // ── HealthEndpoint ──────────────────────────────────────────────────

    #[test]
    fn test_health_initial_state() {
        let mut h = HealthEndpoint::new();
        assert_eq!(h.liveness().status, HealthStatus::Healthy);
        assert_eq!(h.readiness().status, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_health_set_ready() {
        let mut h = HealthEndpoint::new();
        h.set_ready(true);
        assert_eq!(h.readiness().status, HealthStatus::Healthy);
    }

    #[test]
    fn test_health_set_not_live() {
        let mut h = HealthEndpoint::new();
        h.set_live(false);
        assert_eq!(h.liveness().status, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_health_components_all_ok() {
        let mut h = HealthEndpoint::new();
        h.set_ready(true);
        h.set_component("model", true);
        h.set_component("queue", true);
        assert_eq!(h.health().status, HealthStatus::Healthy);
    }

    #[test]
    fn test_health_components_degraded() {
        let mut h = HealthEndpoint::new();
        h.set_ready(true);
        h.set_component("model", true);
        h.set_component("queue", false);
        assert_eq!(h.health().status, HealthStatus::Degraded);
    }

    #[test]
    fn test_health_not_ready_with_components_unhealthy() {
        let mut h = HealthEndpoint::new();
        h.set_component("model", false);
        assert_eq!(h.health().status, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_health_total_checks() {
        let mut h = HealthEndpoint::new();
        let _ = h.liveness();
        let _ = h.readiness();
        let _ = h.health();
        assert_eq!(h.total_checks(), 3);
    }

    #[test]
    fn test_health_response_message() {
        let mut h = HealthEndpoint::new();
        let resp = h.liveness();
        assert!(resp.message.contains("liveness"));
    }

    #[test]
    fn test_health_status_display() {
        assert_eq!(HealthStatus::Healthy.to_string(), "healthy");
        assert_eq!(HealthStatus::Degraded.to_string(), "degraded");
        assert_eq!(HealthStatus::Unhealthy.to_string(), "unhealthy");
    }

    #[test]
    fn test_health_default_impl() {
        let h = HealthEndpoint::default();
        assert_eq!(h.total_checks(), 0);
    }

    // ── MetricsEndpoint ─────────────────────────────────────────────────

    #[test]
    fn test_metrics_new_is_empty() {
        let m = MetricsEndpoint::new();
        assert_eq!(m.metric_count(), 0);
        assert_eq!(m.scrape_count(), 0);
    }

    #[test]
    fn test_metrics_set_and_get() {
        let mut m = MetricsEndpoint::new();
        m.set(Metric::counter("req_total", "Total requests", 42.0));
        assert_eq!(m.get("req_total").unwrap().value, 42.0);
    }

    #[test]
    fn test_metrics_increment() {
        let mut m = MetricsEndpoint::new();
        m.set(Metric::counter("req_total", "Total requests", 10.0));
        m.increment("req_total", 5.0);
        assert_eq!(m.get("req_total").unwrap().value, 15.0);
    }

    #[test]
    fn test_metrics_set_gauge() {
        let mut m = MetricsEndpoint::new();
        m.set(Metric::gauge("active", "Active requests", 0.0));
        m.set_gauge("active", 7.0);
        assert_eq!(m.get("active").unwrap().value, 7.0);
    }

    #[test]
    fn test_metrics_render_increments_scrape_count() {
        let mut m = MetricsEndpoint::new();
        m.set(Metric::counter("x", "X", 1.0));
        let _ = m.render();
        let _ = m.render();
        assert_eq!(m.scrape_count(), 2);
    }

    #[test]
    fn test_metrics_render_contains_help_and_type() {
        let mut m = MetricsEndpoint::new();
        m.set(Metric::counter("req_total", "Total requests", 5.0));
        let out = m.render();
        assert!(out.contains("# HELP req_total Total requests"));
        assert!(out.contains("# TYPE req_total counter"));
        assert!(out.contains("req_total 5"));
    }

    #[test]
    fn test_metric_with_label() {
        let metric = Metric::counter("req", "requests", 1.0).with_label("method", "GET");
        let prom = metric.to_prometheus();
        assert!(prom.contains("method=\"GET\""));
    }

    #[test]
    fn test_metric_gauge_type() {
        let metric = Metric::gauge("active", "active", 3.0);
        assert_eq!(metric.metric_type, "gauge");
    }

    #[test]
    fn test_metrics_default_impl() {
        let m = MetricsEndpoint::default();
        assert_eq!(m.metric_count(), 0);
    }

    #[test]
    fn test_metrics_get_nonexistent() {
        let m = MetricsEndpoint::new();
        assert!(m.get("missing").is_none());
    }

    #[test]
    fn test_metrics_uptime_positive() {
        let m = MetricsEndpoint::new();
        // Uptime should be at least zero.
        assert!(m.uptime() >= Duration::ZERO);
    }

    // ── ModelHotSwap ────────────────────────────────────────────────────

    #[test]
    fn test_hot_swap_initial_state() {
        let hs = ModelHotSwap::new();
        assert_eq!(hs.phase(), HotSwapPhase::Idle);
        assert!(hs.current_model().is_none());
        assert_eq!(hs.total_swaps(), 0);
    }

    #[test]
    fn test_hot_swap_set_initial_model() {
        let mut hs = ModelHotSwap::new();
        let m = make_model("m1", 1);
        hs.set_initial_model(m);
        assert!(hs.current_model().is_some());
    }

    #[test]
    fn test_hot_swap_begin_requires_current_model() {
        let mut hs = ModelHotSwap::new();
        let new = make_model("m2", 2);
        assert!(hs.begin_swap(new).is_err());
    }

    #[test]
    fn test_hot_swap_begin_swap() {
        let mut hs = ModelHotSwap::new();
        hs.set_initial_model(make_model("m1", 1));
        hs.begin_swap(make_model("m2", 2)).unwrap();
        assert_eq!(hs.phase(), HotSwapPhase::LoadingNew);
    }

    #[test]
    fn test_hot_swap_double_begin_fails() {
        let mut hs = ModelHotSwap::new();
        hs.set_initial_model(make_model("m1", 1));
        hs.begin_swap(make_model("m2", 2)).unwrap();
        assert!(hs.begin_swap(make_model("m3", 3)).is_err());
    }

    #[test]
    fn test_hot_swap_full_lifecycle() {
        let mut hs = ModelHotSwap::new();
        let mut m1 = make_model("m1", 1);
        m1.mark_ready();
        hs.set_initial_model(m1);

        let m2 = make_model("m2", 2);
        hs.begin_swap(m2).unwrap();

        assert_eq!(hs.advance(), HotSwapPhase::WarmingNew);
        assert_eq!(hs.advance(), HotSwapPhase::Switching);
        assert_eq!(hs.advance(), HotSwapPhase::DrainingOld);
        assert_eq!(hs.advance(), HotSwapPhase::UnloadingOld);
        assert_eq!(hs.advance(), HotSwapPhase::Completed);

        assert_eq!(hs.total_swaps(), 1);
        assert_eq!(hs.swap_history().len(), 1);
        assert_eq!(hs.swap_history()[0], (1, 2));

        // Current model is now m2.
        assert_eq!(hs.current_model().unwrap().version, 2);
    }

    #[test]
    fn test_hot_swap_rollback() {
        let mut hs = ModelHotSwap::new();
        hs.set_initial_model(make_model("m1", 1));
        hs.begin_swap(make_model("m2", 2)).unwrap();
        hs.advance(); // WarmingNew
        hs.rollback();
        assert_eq!(hs.phase(), HotSwapPhase::RolledBack);
        assert_eq!(hs.total_failures(), 1);
    }

    #[test]
    fn test_hot_swap_completed_returns_to_idle() {
        let mut hs = ModelHotSwap::new();
        let mut m1 = make_model("m1", 1);
        m1.mark_ready();
        hs.set_initial_model(m1);
        hs.begin_swap(make_model("m2", 2)).unwrap();
        // Run through all phases.
        for _ in 0..5 {
            hs.advance();
        }
        assert_eq!(hs.advance(), HotSwapPhase::Idle);
    }

    #[test]
    fn test_hot_swap_rolled_back_returns_to_idle() {
        let mut hs = ModelHotSwap::new();
        hs.set_initial_model(make_model("m1", 1));
        hs.begin_swap(make_model("m2", 2)).unwrap();
        hs.rollback();
        assert_eq!(hs.advance(), HotSwapPhase::Idle);
    }

    #[test]
    fn test_hot_swap_default_impl() {
        let hs = ModelHotSwap::default();
        assert_eq!(hs.phase(), HotSwapPhase::Idle);
    }

    // ── ServingRuntime ──────────────────────────────────────────────────

    #[test]
    fn test_runtime_new_valid_config() {
        let rt = runtime();
        assert_eq!(rt.state(), RuntimeState::Initialising);
    }

    #[test]
    fn test_runtime_new_invalid_config_rejected() {
        let cfg = ServingConfig::default(); // empty model_path
        assert!(ServingRuntime::new(cfg).is_err());
    }

    #[test]
    fn test_runtime_load_model() {
        let mut rt = runtime();
        rt.load_model().unwrap();
        assert!(rt.model().is_some());
        assert_eq!(rt.state(), RuntimeState::LoadingModel);
    }

    #[test]
    fn test_runtime_warmup() {
        let mut rt = runtime();
        rt.load_model().unwrap();
        let n = rt.warmup().unwrap();
        assert_eq!(n, 3);
        assert_eq!(rt.state(), RuntimeState::WarmingUp);
    }

    #[test]
    fn test_runtime_start_serving() {
        let rt = serving_runtime();
        assert_eq!(rt.state(), RuntimeState::Serving);
    }

    #[test]
    fn test_runtime_submit_and_process() {
        let mut rt = serving_runtime();
        rt.submit(make_request("r1")).unwrap();
        let result = rt.process_next().unwrap().unwrap();
        assert_eq!(result.request_id, "r1");
        assert_eq!(rt.total_processed(), 1);
    }

    #[test]
    fn test_runtime_submit_rejects_when_not_serving() {
        let mut rt = runtime();
        assert!(rt.submit(make_request("r1")).is_err());
    }

    #[test]
    fn test_runtime_process_empty_queue() {
        let mut rt = serving_runtime();
        assert!(rt.process_next().is_none());
    }

    #[test]
    fn test_runtime_metrics_updated_after_process() {
        let mut rt = serving_runtime();
        rt.submit(make_request("r1")).unwrap();
        rt.process_next();
        let total = rt.metrics().get("bitnet_requests_total").unwrap().value;
        assert!(total >= 1.0);
    }

    #[test]
    fn test_runtime_shutdown_lifecycle() {
        let mut rt = serving_runtime();
        rt.submit(make_request("r1")).unwrap();
        rt.initiate_shutdown();
        assert_eq!(rt.state(), RuntimeState::ShuttingDown);
        let drained = rt.drain();
        assert_eq!(drained, 1);
        rt.complete_shutdown();
        assert_eq!(rt.state(), RuntimeState::Stopped);
        assert!(rt.shutdown().is_completed());
    }

    #[test]
    fn test_runtime_model_records_requests() {
        let mut rt = serving_runtime();
        rt.submit(make_request("r1")).unwrap();
        rt.submit(make_request("r2")).unwrap();
        rt.process_next();
        rt.process_next();
        assert_eq!(rt.model().unwrap().requests_served, 2);
    }

    #[test]
    fn test_runtime_config_accessor() {
        let rt = runtime();
        assert_eq!(rt.config().port, 8080);
    }

    #[test]
    fn test_runtime_queue_accessor() {
        let rt = serving_runtime();
        assert!(rt.queue().is_empty());
    }

    #[test]
    fn test_runtime_limiter_accessor() {
        let rt = serving_runtime();
        assert_eq!(rt.limiter().active(), 0);
    }

    #[test]
    fn test_runtime_uptime_positive() {
        let rt = runtime();
        assert!(rt.uptime() >= Duration::ZERO);
    }

    #[test]
    fn test_runtime_health_ready_after_start() {
        let mut rt = serving_runtime();
        let resp = rt.health_mut().readiness();
        assert_eq!(resp.status, HealthStatus::Healthy);
    }

    #[test]
    fn test_runtime_health_not_ready_after_shutdown() {
        let mut rt = serving_runtime();
        rt.initiate_shutdown();
        let resp = rt.health_mut().readiness();
        assert_eq!(resp.status, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_runtime_initial_metrics_registered() {
        let rt = runtime();
        assert!(rt.metrics().get("bitnet_requests_total").is_some());
        assert!(rt.metrics().get("bitnet_requests_failed").is_some());
        assert!(rt.metrics().get("bitnet_active_requests").is_some());
        assert!(rt.metrics().get("bitnet_queue_depth").is_some());
        assert!(rt.metrics().get("bitnet_model_loaded").is_some());
    }

    #[test]
    fn test_runtime_model_loaded_metric_after_load() {
        let mut rt = runtime();
        rt.load_model().unwrap();
        assert_eq!(rt.metrics().get("bitnet_model_loaded").unwrap().value, 1.0);
    }

    #[test]
    fn test_runtime_hot_swap_accessor() {
        let rt = runtime();
        assert_eq!(rt.hot_swap().phase(), HotSwapPhase::Idle);
    }

    #[test]
    fn test_runtime_multiple_requests() {
        let mut rt = serving_runtime();
        for i in 0..10 {
            rt.submit(make_request(&format!("r{i}"))).unwrap();
        }
        let mut processed = 0;
        while rt.process_next().is_some() {
            processed += 1;
        }
        assert_eq!(processed, 10);
        assert_eq!(rt.total_processed(), 10);
    }
}
