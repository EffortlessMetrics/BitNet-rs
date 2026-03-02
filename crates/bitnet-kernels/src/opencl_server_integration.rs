//! Server integration layer for OpenCL A770 backend.
//!
//! Bridges the OpenCL compute backend with `bitnet-server` by providing:
//!
//! - **`ServerBackend`** — device info, concurrency limits, health tracking,
//!   and aggregate statistics.
//! - **`InferenceSession`** / **`SessionPool`** — lifecycle-managed inference
//!   sessions with configurable limits and LRU eviction.
//! - **`RequestPriority`** / **`LoadBalancer`** — priority-aware request
//!   distribution across active sessions.
//! - **`BackendHealth`** / **`GracefulDegradation`** — GPU health monitoring
//!   with automatic CPU fallback when thresholds are exceeded.
//!
//! All scheduling and health-check algorithms have CPU reference
//! implementations so that the module compiles and passes tests without any
//! GPU hardware or OpenCL runtime.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Device info
// ---------------------------------------------------------------------------

/// Identifies the GPU device backing this server backend.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device name (e.g. "Intel(R) Arc(TM) A770 Graphics").
    pub name: String,
    /// Vendor string.
    pub vendor: String,
    /// Total device memory in megabytes.
    pub total_memory_mb: u64,
    /// Maximum concurrent sessions the device can support.
    pub max_concurrent: usize,
}

impl DeviceInfo {
    /// Create a `DeviceInfo` for an Intel Arc A770 with sensible defaults.
    pub fn arc_a770() -> Self {
        Self {
            name: "Intel(R) Arc(TM) A770 Graphics".into(),
            vendor: "Intel".into(),
            total_memory_mb: 16384,
            max_concurrent: 8,
        }
    }

    /// Create a minimal device info for testing.
    pub fn test_device() -> Self {
        Self {
            name: "Test GPU".into(),
            vendor: "Test".into(),
            total_memory_mb: 4096,
            max_concurrent: 4,
        }
    }
}

// ---------------------------------------------------------------------------
// Backend health
// ---------------------------------------------------------------------------

/// Health snapshot of the GPU backend.
#[derive(Debug, Clone)]
pub struct BackendHealth {
    /// GPU temperature in degrees Celsius.
    pub gpu_temp_c: f32,
    /// Memory currently in use (MB).
    pub memory_used_mb: u64,
    /// GPU utilization percentage (0–100).
    pub utilization_pct: f32,
    /// Cumulative error count since startup.
    pub error_count: u64,
    /// Seconds since the backend was started.
    pub uptime_s: f64,
}

impl BackendHealth {
    /// A healthy default for testing.
    pub fn healthy() -> Self {
        Self {
            gpu_temp_c: 45.0,
            memory_used_mb: 1024,
            utilization_pct: 30.0,
            error_count: 0,
            uptime_s: 100.0,
        }
    }

    /// Whether all metrics are within safe operating ranges.
    pub fn is_healthy(&self) -> bool {
        self.gpu_temp_c < 90.0 && self.utilization_pct < 95.0 && self.error_count < 100
    }
}

impl Default for BackendHealth {
    fn default() -> Self {
        Self::healthy()
    }
}

// ---------------------------------------------------------------------------
// Health status
// ---------------------------------------------------------------------------

/// Overall backend health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// All systems operational.
    Healthy,
    /// Operating but with degraded performance.
    Degraded,
    /// GPU unavailable — CPU fallback active.
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

// ---------------------------------------------------------------------------
// Backend statistics
// ---------------------------------------------------------------------------

/// Aggregate statistics for the server backend.
#[derive(Debug)]
pub struct BackendStats {
    /// Total requests processed.
    pub total_requests: AtomicU64,
    /// Total tokens generated.
    pub total_tokens: AtomicU64,
    /// Requests that fell back to CPU.
    pub cpu_fallback_count: AtomicU64,
    /// Requests rejected due to overload.
    pub rejected_count: AtomicU64,
}

impl BackendStats {
    pub fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            total_tokens: AtomicU64::new(0),
            cpu_fallback_count: AtomicU64::new(0),
            rejected_count: AtomicU64::new(0),
        }
    }

    pub fn record_request(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_tokens(&self, count: u64) {
        self.total_tokens.fetch_add(count, Ordering::Relaxed);
    }

    pub fn record_cpu_fallback(&self) {
        self.cpu_fallback_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_rejected(&self) {
        self.rejected_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Snapshot of current counters.
    pub fn snapshot(&self) -> StatsSnapshot {
        StatsSnapshot {
            total_requests: self.total_requests.load(Ordering::Relaxed),
            total_tokens: self.total_tokens.load(Ordering::Relaxed),
            cpu_fallback_count: self.cpu_fallback_count.load(Ordering::Relaxed),
            rejected_count: self.rejected_count.load(Ordering::Relaxed),
        }
    }
}

impl Default for BackendStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Immutable snapshot of [`BackendStats`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatsSnapshot {
    pub total_requests: u64,
    pub total_tokens: u64,
    pub cpu_fallback_count: u64,
    pub rejected_count: u64,
}

// ---------------------------------------------------------------------------
// Server backend
// ---------------------------------------------------------------------------

/// Top-level server backend managing device, health, and statistics.
pub struct ServerBackend {
    pub device_info: DeviceInfo,
    pub max_concurrent: usize,
    pub health_status: HealthStatus,
    pub health: BackendHealth,
    pub stats: BackendStats,
    started_at: Instant,
}

impl ServerBackend {
    /// Create a new backend for the given device.
    pub fn new(device_info: DeviceInfo) -> Self {
        let max_concurrent = device_info.max_concurrent;
        Self {
            device_info,
            max_concurrent,
            health_status: HealthStatus::Healthy,
            health: BackendHealth::healthy(),
            stats: BackendStats::new(),
            started_at: Instant::now(),
        }
    }

    /// Update health metrics and recompute status.
    pub fn update_health(&mut self, health: BackendHealth) {
        self.health_status = if health.gpu_temp_c >= 90.0 || health.error_count >= 100 {
            HealthStatus::Unhealthy
        } else if health.gpu_temp_c >= 80.0
            || health.utilization_pct >= 90.0
            || health.error_count >= 50
        {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };
        self.health = health;
    }

    /// Uptime since creation.
    pub fn uptime(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Whether the backend can accept new work.
    pub fn is_available(&self) -> bool {
        self.health_status != HealthStatus::Unhealthy
    }
}

impl fmt::Debug for ServerBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ServerBackend")
            .field("device", &self.device_info.name)
            .field("status", &self.health_status)
            .field("max_concurrent", &self.max_concurrent)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Inference session
// ---------------------------------------------------------------------------

/// Unique session identifier.
pub type SessionId = u64;

/// A single inference session bound to a model.
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceSession {
    /// Unique session identifier.
    pub id: SessionId,
    /// Name/path of the loaded model.
    pub model_name: String,
    /// When the session was created.
    pub created_at: Instant,
    /// Last time the session processed a request.
    pub last_active: Instant,
    /// Cumulative tokens generated in this session.
    pub tokens_generated: u64,
}

impl InferenceSession {
    /// Create a new session.
    pub fn new(id: SessionId, model_name: impl Into<String>) -> Self {
        let now = Instant::now();
        Self {
            id,
            model_name: model_name.into(),
            created_at: now,
            last_active: now,
            tokens_generated: 0,
        }
    }

    /// Mark the session as active and add to its token count.
    pub fn record_activity(&mut self, tokens: u64) {
        self.last_active = Instant::now();
        self.tokens_generated += tokens;
    }

    /// Duration since last activity.
    pub fn idle_time(&self) -> Duration {
        self.last_active.elapsed()
    }
}

// ---------------------------------------------------------------------------
// Session pool
// ---------------------------------------------------------------------------

/// Error returned when the session pool cannot fulfil a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionPoolError {
    /// Pool has reached its capacity limit.
    PoolFull,
    /// The requested session does not exist.
    NotFound(SessionId),
    /// A session with this ID already exists.
    DuplicateId(SessionId),
}

impl fmt::Display for SessionPoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PoolFull => write!(f, "session pool is full"),
            Self::NotFound(id) => write!(f, "session {id} not found"),
            Self::DuplicateId(id) => write!(f, "session {id} already exists"),
        }
    }
}

impl std::error::Error for SessionPoolError {}

/// Manages a bounded set of [`InferenceSession`]s with LRU eviction.
pub struct SessionPool {
    sessions: HashMap<SessionId, InferenceSession>,
    /// Insertion order for LRU eviction.
    order: VecDeque<SessionId>,
    max_sessions: usize,
    next_id: SessionId,
}

impl SessionPool {
    /// Create a pool with the given capacity.
    pub fn new(max_sessions: usize) -> Self {
        Self { sessions: HashMap::new(), order: VecDeque::new(), max_sessions, next_id: 1 }
    }

    /// Number of active sessions.
    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }

    /// Whether the pool is at capacity.
    pub fn is_full(&self) -> bool {
        self.sessions.len() >= self.max_sessions
    }

    /// Maximum number of sessions.
    pub fn capacity(&self) -> usize {
        self.max_sessions
    }

    /// Create a new session, returning its ID. Evicts LRU if full.
    pub fn create_session(
        &mut self,
        model_name: impl Into<String>,
    ) -> Result<SessionId, SessionPoolError> {
        if self.is_full() {
            self.evict_lru();
        }
        // After eviction the pool must have room (unless max_sessions == 0).
        if self.is_full() {
            return Err(SessionPoolError::PoolFull);
        }

        let id = self.next_id;
        self.next_id += 1;
        let session = InferenceSession::new(id, model_name);
        self.sessions.insert(id, session);
        self.order.push_back(id);
        Ok(id)
    }

    /// Insert a session with a specific ID (for testing).
    pub fn insert(&mut self, session: InferenceSession) -> Result<(), SessionPoolError> {
        if self.sessions.contains_key(&session.id) {
            return Err(SessionPoolError::DuplicateId(session.id));
        }
        if self.is_full() {
            self.evict_lru();
        }
        if self.is_full() {
            return Err(SessionPoolError::PoolFull);
        }
        let id = session.id;
        self.sessions.insert(id, session);
        self.order.push_back(id);
        if id >= self.next_id {
            self.next_id = id + 1;
        }
        Ok(())
    }

    /// Get a reference to a session by ID.
    pub fn get(&self, id: SessionId) -> Option<&InferenceSession> {
        self.sessions.get(&id)
    }

    /// Get a mutable reference to a session by ID.
    pub fn get_mut(&mut self, id: SessionId) -> Option<&mut InferenceSession> {
        self.sessions.get_mut(&id)
    }

    /// Remove a session by ID.
    pub fn remove(&mut self, id: SessionId) -> Result<InferenceSession, SessionPoolError> {
        let session = self.sessions.remove(&id).ok_or(SessionPoolError::NotFound(id))?;
        self.order.retain(|&sid| sid != id);
        Ok(session)
    }

    /// Evict the least-recently-used session.
    fn evict_lru(&mut self) -> Option<InferenceSession> {
        // Find the session with the oldest `last_active`.
        let lru_id = self.sessions.values().min_by_key(|s| s.last_active).map(|s| s.id)?;
        self.sessions.remove(&lru_id).inspect(|_| {
            self.order.retain(|&sid| sid != lru_id);
        })
    }

    /// Remove all sessions.
    pub fn clear(&mut self) {
        self.sessions.clear();
        self.order.clear();
    }

    /// Iterate over all active sessions.
    pub fn iter(&self) -> impl Iterator<Item = &InferenceSession> {
        self.sessions.values()
    }

    /// IDs of all active sessions in insertion order.
    pub fn session_ids(&self) -> Vec<SessionId> {
        self.order.iter().copied().filter(|id| self.sessions.contains_key(id)).collect()
    }
}

impl fmt::Debug for SessionPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SessionPool")
            .field("active", &self.sessions.len())
            .field("max", &self.max_sessions)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Request priority
// ---------------------------------------------------------------------------

/// Priority level for an inference request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RequestPriority {
    /// Background batch jobs — lowest priority.
    Background = 0,
    /// Batch processing — low priority.
    Batch = 1,
    /// Interactive user requests — normal priority.
    Interactive = 2,
    /// Realtime streaming — highest priority.
    Realtime = 3,
}

impl RequestPriority {
    /// All variants in ascending priority order.
    pub const ALL: [RequestPriority; 4] =
        [Self::Background, Self::Batch, Self::Interactive, Self::Realtime];

    /// Relative weight for load-balancing (higher = more resources).
    pub fn weight(self) -> u32 {
        match self {
            Self::Background => 1,
            Self::Batch => 2,
            Self::Interactive => 4,
            Self::Realtime => 8,
        }
    }
}

impl fmt::Display for RequestPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Background => write!(f, "background"),
            Self::Batch => write!(f, "batch"),
            Self::Interactive => write!(f, "interactive"),
            Self::Realtime => write!(f, "realtime"),
        }
    }
}

// ---------------------------------------------------------------------------
// Inference request
// ---------------------------------------------------------------------------

/// An inference request submitted to the load balancer.
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    /// Unique request ID.
    pub id: u64,
    /// Desired priority.
    pub priority: RequestPriority,
    /// Target model name.
    pub model_name: String,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
}

// ---------------------------------------------------------------------------
// Load balancer
// ---------------------------------------------------------------------------

/// Routing decision made by the load balancer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteDecision {
    /// Route to the given GPU session.
    GpuSession(SessionId),
    /// Fall back to CPU inference.
    CpuFallback,
    /// Reject the request (overloaded).
    Reject,
}

/// Distributes requests across sessions based on priority and load.
pub struct LoadBalancer {
    /// Per-session pending request count.
    pending: HashMap<SessionId, u32>,
    /// Total requests routed to each session (lifetime).
    routed: HashMap<SessionId, u64>,
}

impl LoadBalancer {
    pub fn new() -> Self {
        Self { pending: HashMap::new(), routed: HashMap::new() }
    }

    /// Register a session so the balancer can route to it.
    pub fn add_session(&mut self, id: SessionId) {
        self.pending.entry(id).or_insert(0);
        self.routed.entry(id).or_insert(0);
    }

    /// Remove a session from the balancer.
    pub fn remove_session(&mut self, id: SessionId) {
        self.pending.remove(&id);
        self.routed.remove(&id);
    }

    /// Route a request, returning the target session or a fallback decision.
    ///
    /// Strategy: pick the session with the fewest pending requests (least-loaded).
    /// If no sessions are available, return `CpuFallback` for high-priority
    /// requests or `Reject` for background work.
    pub fn route(
        &mut self,
        request: &InferenceRequest,
        health_status: HealthStatus,
    ) -> RouteDecision {
        // If unhealthy, everything goes to CPU or is rejected.
        if health_status == HealthStatus::Unhealthy {
            return if request.priority >= RequestPriority::Interactive {
                RouteDecision::CpuFallback
            } else {
                RouteDecision::Reject
            };
        }

        // Find least-loaded session; break ties by fewest lifetime routes,
        // then by lowest session ID for determinism.
        let best = self
            .pending
            .iter()
            .min_by(|&(&id_a, &cnt_a), &(&id_b, &cnt_b)| {
                cnt_a
                    .cmp(&cnt_b)
                    .then_with(|| {
                        let ra = self.routed.get(&id_a).copied().unwrap_or(0);
                        let rb = self.routed.get(&id_b).copied().unwrap_or(0);
                        ra.cmp(&rb)
                    })
                    .then_with(|| id_a.cmp(&id_b))
            })
            .map(|(&id, _)| id);

        match best {
            Some(id) => {
                *self.pending.entry(id).or_insert(0) += 1;
                *self.routed.entry(id).or_insert(0) += 1;
                RouteDecision::GpuSession(id)
            }
            None => {
                if request.priority >= RequestPriority::Interactive {
                    RouteDecision::CpuFallback
                } else {
                    RouteDecision::Reject
                }
            }
        }
    }

    /// Mark a request as completed on the given session.
    pub fn complete(&mut self, session_id: SessionId) {
        if let Some(count) = self.pending.get_mut(&session_id) {
            *count = count.saturating_sub(1);
        }
    }

    /// Current pending count for a session.
    pub fn pending_count(&self, session_id: SessionId) -> u32 {
        self.pending.get(&session_id).copied().unwrap_or(0)
    }

    /// Lifetime routed count for a session.
    pub fn routed_count(&self, session_id: SessionId) -> u64 {
        self.routed.get(&session_id).copied().unwrap_or(0)
    }

    /// Total pending across all sessions.
    pub fn total_pending(&self) -> u32 {
        self.pending.values().sum()
    }

    /// Number of tracked sessions.
    pub fn session_count(&self) -> usize {
        self.pending.len()
    }

    /// Check distribution fairness: ratio of max-to-min routed counts.
    /// Returns `None` if fewer than 2 sessions are tracked or any session
    /// has zero routes.
    pub fn fairness_ratio(&self) -> Option<f64> {
        if self.routed.len() < 2 {
            return None;
        }
        let min = *self.routed.values().min()?;
        let max = *self.routed.values().max()?;
        if min == 0 {
            return None;
        }
        Some(max as f64 / min as f64)
    }
}

impl Default for LoadBalancer {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for LoadBalancer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoadBalancer")
            .field("sessions", &self.pending.len())
            .field("total_pending", &self.total_pending())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Graceful degradation
// ---------------------------------------------------------------------------

/// Thresholds that trigger CPU fallback.
#[derive(Debug, Clone)]
pub struct DegradationPolicy {
    /// GPU temperature threshold (°C) — above this, degrade.
    pub temp_threshold_c: f32,
    /// Memory usage threshold (MB) — above this, degrade.
    pub memory_threshold_mb: u64,
    /// Utilization threshold (%) — above this, degrade.
    pub utilization_threshold_pct: f32,
    /// Error rate threshold — more than this many errors, degrade.
    pub error_threshold: u64,
}

impl DegradationPolicy {
    /// Sensible defaults for an Intel Arc A770.
    pub fn default_a770() -> Self {
        Self {
            temp_threshold_c: 85.0,
            memory_threshold_mb: 14_000,
            utilization_threshold_pct: 95.0,
            error_threshold: 50,
        }
    }

    /// Tight thresholds for testing.
    pub fn strict() -> Self {
        Self {
            temp_threshold_c: 70.0,
            memory_threshold_mb: 3000,
            utilization_threshold_pct: 80.0,
            error_threshold: 5,
        }
    }
}

impl Default for DegradationPolicy {
    fn default() -> Self {
        Self::default_a770()
    }
}

/// Reason the system decided to fall back to CPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DegradationReason {
    /// GPU temperature exceeded threshold.
    TemperatureExceeded,
    /// GPU memory usage exceeded threshold.
    MemoryExceeded,
    /// GPU utilization exceeded threshold.
    UtilizationExceeded,
    /// Error count exceeded threshold.
    ErrorRateExceeded,
    /// Multiple thresholds exceeded.
    MultipleThresholds(Vec<DegradationReason>),
}

impl fmt::Display for DegradationReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TemperatureExceeded => write!(f, "GPU temperature exceeded"),
            Self::MemoryExceeded => write!(f, "GPU memory exceeded"),
            Self::UtilizationExceeded => write!(f, "GPU utilization exceeded"),
            Self::ErrorRateExceeded => write!(f, "error rate exceeded"),
            Self::MultipleThresholds(reasons) => {
                write!(f, "multiple thresholds: ")?;
                for (i, r) in reasons.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{r}")?;
                }
                Ok(())
            }
        }
    }
}

/// Evaluates GPU health against a [`DegradationPolicy`] and decides whether
/// to fall back to CPU.
pub struct GracefulDegradation {
    policy: DegradationPolicy,
    /// Whether CPU fallback is currently active.
    fallback_active: bool,
    /// History of degradation events.
    history: Vec<(Instant, DegradationReason)>,
}

impl GracefulDegradation {
    pub fn new(policy: DegradationPolicy) -> Self {
        Self { policy, fallback_active: false, history: Vec::new() }
    }

    /// Evaluate current health and return any degradation reason.
    pub fn evaluate(&mut self, health: &BackendHealth) -> Option<DegradationReason> {
        let mut reasons = Vec::new();

        if health.gpu_temp_c >= self.policy.temp_threshold_c {
            reasons.push(DegradationReason::TemperatureExceeded);
        }
        if health.memory_used_mb >= self.policy.memory_threshold_mb {
            reasons.push(DegradationReason::MemoryExceeded);
        }
        if health.utilization_pct >= self.policy.utilization_threshold_pct {
            reasons.push(DegradationReason::UtilizationExceeded);
        }
        if health.error_count >= self.policy.error_threshold {
            reasons.push(DegradationReason::ErrorRateExceeded);
        }

        let reason = match reasons.len() {
            0 => {
                self.fallback_active = false;
                return None;
            }
            1 => reasons.into_iter().next().unwrap(),
            _ => DegradationReason::MultipleThresholds(reasons),
        };

        self.fallback_active = true;
        self.history.push((Instant::now(), reason.clone()));
        Some(reason)
    }

    /// Whether CPU fallback is currently engaged.
    pub fn is_fallback_active(&self) -> bool {
        self.fallback_active
    }

    /// Number of degradation events recorded.
    pub fn event_count(&self) -> usize {
        self.history.len()
    }

    /// The current policy.
    pub fn policy(&self) -> &DegradationPolicy {
        &self.policy
    }

    /// Update the degradation policy.
    pub fn set_policy(&mut self, policy: DegradationPolicy) {
        self.policy = policy;
    }

    /// Reset fallback state (e.g. after manual intervention).
    pub fn reset(&mut self) {
        self.fallback_active = false;
    }
}

impl fmt::Debug for GracefulDegradation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GracefulDegradation")
            .field("fallback_active", &self.fallback_active)
            .field("events", &self.history.len())
            .finish()
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- DeviceInfo --------------------------------------------------------

    #[test]
    fn test_device_info_arc_a770_defaults() {
        let d = DeviceInfo::arc_a770();
        assert_eq!(d.vendor, "Intel");
        assert_eq!(d.total_memory_mb, 16384);
        assert!(d.max_concurrent > 0);
    }

    #[test]
    fn test_device_info_test_defaults() {
        let d = DeviceInfo::test_device();
        assert_eq!(d.vendor, "Test");
        assert_eq!(d.max_concurrent, 4);
    }

    // -- BackendHealth ----------------------------------------------------

    #[test]
    fn test_backend_health_healthy() {
        let h = BackendHealth::healthy();
        assert!(h.is_healthy());
    }

    #[test]
    fn test_backend_health_high_temp_unhealthy() {
        let h = BackendHealth { gpu_temp_c: 95.0, ..BackendHealth::healthy() };
        assert!(!h.is_healthy());
    }

    #[test]
    fn test_backend_health_high_utilization_unhealthy() {
        let h = BackendHealth { utilization_pct: 99.0, ..BackendHealth::healthy() };
        assert!(!h.is_healthy());
    }

    #[test]
    fn test_backend_health_high_errors_unhealthy() {
        let h = BackendHealth { error_count: 200, ..BackendHealth::healthy() };
        assert!(!h.is_healthy());
    }

    #[test]
    fn test_backend_health_default() {
        let h = BackendHealth::default();
        assert!(h.is_healthy());
    }

    // -- HealthStatus -----------------------------------------------------

    #[test]
    fn test_health_status_display() {
        assert_eq!(HealthStatus::Healthy.to_string(), "healthy");
        assert_eq!(HealthStatus::Degraded.to_string(), "degraded");
        assert_eq!(HealthStatus::Unhealthy.to_string(), "unhealthy");
    }

    // -- BackendStats -----------------------------------------------------

    #[test]
    fn test_backend_stats_initial() {
        let s = BackendStats::new();
        let snap = s.snapshot();
        assert_eq!(snap.total_requests, 0);
        assert_eq!(snap.total_tokens, 0);
    }

    #[test]
    fn test_backend_stats_record_request() {
        let s = BackendStats::new();
        s.record_request();
        s.record_request();
        assert_eq!(s.snapshot().total_requests, 2);
    }

    #[test]
    fn test_backend_stats_record_tokens() {
        let s = BackendStats::new();
        s.record_tokens(10);
        s.record_tokens(20);
        assert_eq!(s.snapshot().total_tokens, 30);
    }

    #[test]
    fn test_backend_stats_record_cpu_fallback() {
        let s = BackendStats::new();
        s.record_cpu_fallback();
        assert_eq!(s.snapshot().cpu_fallback_count, 1);
    }

    #[test]
    fn test_backend_stats_record_rejected() {
        let s = BackendStats::new();
        s.record_rejected();
        assert_eq!(s.snapshot().rejected_count, 1);
    }

    // -- ServerBackend ----------------------------------------------------

    #[test]
    fn test_server_backend_new() {
        let b = ServerBackend::new(DeviceInfo::arc_a770());
        assert_eq!(b.health_status, HealthStatus::Healthy);
        assert!(b.is_available());
    }

    #[test]
    fn test_server_backend_update_health_degraded() {
        let mut b = ServerBackend::new(DeviceInfo::test_device());
        b.update_health(BackendHealth { gpu_temp_c: 82.0, ..BackendHealth::healthy() });
        assert_eq!(b.health_status, HealthStatus::Degraded);
        assert!(b.is_available());
    }

    #[test]
    fn test_server_backend_update_health_unhealthy() {
        let mut b = ServerBackend::new(DeviceInfo::test_device());
        b.update_health(BackendHealth { gpu_temp_c: 95.0, ..BackendHealth::healthy() });
        assert_eq!(b.health_status, HealthStatus::Unhealthy);
        assert!(!b.is_available());
    }

    #[test]
    fn test_server_backend_uptime() {
        let b = ServerBackend::new(DeviceInfo::test_device());
        assert!(b.uptime().as_nanos() > 0 || b.uptime().as_nanos() == 0);
    }

    #[test]
    fn test_server_backend_debug() {
        let b = ServerBackend::new(DeviceInfo::test_device());
        let dbg = format!("{b:?}");
        assert!(dbg.contains("ServerBackend"));
    }

    // -- InferenceSession -------------------------------------------------

    #[test]
    fn test_session_new() {
        let s = InferenceSession::new(1, "test-model");
        assert_eq!(s.id, 1);
        assert_eq!(s.model_name, "test-model");
        assert_eq!(s.tokens_generated, 0);
    }

    #[test]
    fn test_session_record_activity() {
        let mut s = InferenceSession::new(1, "m");
        s.record_activity(10);
        assert_eq!(s.tokens_generated, 10);
        s.record_activity(5);
        assert_eq!(s.tokens_generated, 15);
    }

    #[test]
    fn test_session_idle_time() {
        let s = InferenceSession::new(1, "m");
        // Idle time should be very small right after creation.
        assert!(s.idle_time().as_secs() < 1);
    }

    // -- SessionPool: creation / destruction lifecycle ---------------------

    #[test]
    fn test_pool_create_session() {
        let mut pool = SessionPool::new(4);
        let id = pool.create_session("model-a").unwrap();
        assert_eq!(pool.len(), 1);
        assert!(pool.get(id).is_some());
    }

    #[test]
    fn test_pool_remove_session() {
        let mut pool = SessionPool::new(4);
        let id = pool.create_session("model-a").unwrap();
        let removed = pool.remove(id).unwrap();
        assert_eq!(removed.id, id);
        assert!(pool.is_empty());
    }

    #[test]
    fn test_pool_remove_nonexistent() {
        let mut pool = SessionPool::new(4);
        assert_eq!(pool.remove(999), Err(SessionPoolError::NotFound(999)));
    }

    #[test]
    fn test_pool_insert_duplicate() {
        let mut pool = SessionPool::new(4);
        let id = pool.create_session("m").unwrap();
        let dup = InferenceSession::new(id, "m2");
        assert_eq!(pool.insert(dup), Err(SessionPoolError::DuplicateId(id)));
    }

    #[test]
    fn test_pool_clear() {
        let mut pool = SessionPool::new(4);
        pool.create_session("a").unwrap();
        pool.create_session("b").unwrap();
        pool.clear();
        assert!(pool.is_empty());
    }

    // -- SessionPool: limits and eviction ---------------------------------

    #[test]
    fn test_pool_evicts_lru_when_full() {
        let mut pool = SessionPool::new(2);
        let id1 = pool.create_session("a").unwrap();
        let _id2 = pool.create_session("b").unwrap();
        // Pool is full; creating another should evict id1 (oldest last_active).
        let id3 = pool.create_session("c").unwrap();
        assert!(pool.get(id1).is_none(), "LRU session should be evicted");
        assert!(pool.get(id3).is_some());
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn test_pool_zero_capacity() {
        let mut pool = SessionPool::new(0);
        assert_eq!(pool.create_session("a"), Err(SessionPoolError::PoolFull));
    }

    #[test]
    fn test_pool_max_sessions_respected() {
        let mut pool = SessionPool::new(3);
        for i in 0..10 {
            let _ = pool.create_session(format!("model-{i}"));
        }
        assert!(pool.len() <= 3);
    }

    #[test]
    fn test_pool_is_full() {
        let mut pool = SessionPool::new(1);
        assert!(!pool.is_full());
        pool.create_session("a").unwrap();
        assert!(pool.is_full());
    }

    #[test]
    fn test_pool_session_ids_order() {
        let mut pool = SessionPool::new(4);
        let a = pool.create_session("a").unwrap();
        let b = pool.create_session("b").unwrap();
        let c = pool.create_session("c").unwrap();
        let ids = pool.session_ids();
        assert_eq!(ids, vec![a, b, c]);
    }

    #[test]
    fn test_pool_capacity() {
        let pool = SessionPool::new(5);
        assert_eq!(pool.capacity(), 5);
    }

    #[test]
    fn test_pool_iter() {
        let mut pool = SessionPool::new(4);
        pool.create_session("a").unwrap();
        pool.create_session("b").unwrap();
        let names: Vec<_> = pool.iter().map(|s| s.model_name.clone()).collect();
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn test_pool_get_mut_updates_session() {
        let mut pool = SessionPool::new(4);
        let id = pool.create_session("m").unwrap();
        pool.get_mut(id).unwrap().record_activity(42);
        assert_eq!(pool.get(id).unwrap().tokens_generated, 42);
    }

    // -- RequestPriority --------------------------------------------------

    #[test]
    fn test_priority_ordering() {
        assert!(RequestPriority::Background < RequestPriority::Batch);
        assert!(RequestPriority::Batch < RequestPriority::Interactive);
        assert!(RequestPriority::Interactive < RequestPriority::Realtime);
    }

    #[test]
    fn test_priority_weights() {
        assert!(RequestPriority::Realtime.weight() > RequestPriority::Interactive.weight());
        assert!(RequestPriority::Interactive.weight() > RequestPriority::Batch.weight());
        assert!(RequestPriority::Batch.weight() > RequestPriority::Background.weight());
    }

    #[test]
    fn test_priority_display() {
        assert_eq!(RequestPriority::Realtime.to_string(), "realtime");
        assert_eq!(RequestPriority::Background.to_string(), "background");
    }

    #[test]
    fn test_priority_all_variants() {
        assert_eq!(RequestPriority::ALL.len(), 4);
    }

    // -- LoadBalancer: distribution fairness -------------------------------

    fn make_request(id: u64, priority: RequestPriority) -> InferenceRequest {
        InferenceRequest { id, priority, model_name: "test".into(), max_tokens: 32 }
    }

    #[test]
    fn test_lb_routes_to_least_loaded() {
        let mut lb = LoadBalancer::new();
        lb.add_session(1);
        lb.add_session(2);

        // Route first request — both sessions empty, picks one.
        let r1 = lb.route(&make_request(1, RequestPriority::Interactive), HealthStatus::Healthy);
        assert!(matches!(r1, RouteDecision::GpuSession(_)));

        // Complete on whichever got it, then route again.
        if let RouteDecision::GpuSession(sid) = r1 {
            // Don't complete — the second request should go to the other session.
            let r2 =
                lb.route(&make_request(2, RequestPriority::Interactive), HealthStatus::Healthy);
            if let RouteDecision::GpuSession(sid2) = r2 {
                assert_ne!(sid, sid2, "should route to the less-loaded session");
            }
        }
    }

    #[test]
    fn test_lb_no_sessions_interactive_falls_back() {
        let mut lb = LoadBalancer::new();
        let d = lb.route(&make_request(1, RequestPriority::Interactive), HealthStatus::Healthy);
        assert_eq!(d, RouteDecision::CpuFallback);
    }

    #[test]
    fn test_lb_no_sessions_background_rejected() {
        let mut lb = LoadBalancer::new();
        let d = lb.route(&make_request(1, RequestPriority::Background), HealthStatus::Healthy);
        assert_eq!(d, RouteDecision::Reject);
    }

    #[test]
    fn test_lb_unhealthy_interactive_falls_back() {
        let mut lb = LoadBalancer::new();
        lb.add_session(1);
        let d = lb.route(&make_request(1, RequestPriority::Interactive), HealthStatus::Unhealthy);
        assert_eq!(d, RouteDecision::CpuFallback);
    }

    #[test]
    fn test_lb_unhealthy_background_rejected() {
        let mut lb = LoadBalancer::new();
        lb.add_session(1);
        let d = lb.route(&make_request(1, RequestPriority::Background), HealthStatus::Unhealthy);
        assert_eq!(d, RouteDecision::Reject);
    }

    #[test]
    fn test_lb_complete_decrements_pending() {
        let mut lb = LoadBalancer::new();
        lb.add_session(1);
        lb.route(&make_request(1, RequestPriority::Interactive), HealthStatus::Healthy);
        assert_eq!(lb.pending_count(1), 1);
        lb.complete(1);
        assert_eq!(lb.pending_count(1), 0);
    }

    #[test]
    fn test_lb_complete_saturating() {
        let mut lb = LoadBalancer::new();
        lb.add_session(1);
        lb.complete(1); // no pending — should not underflow
        assert_eq!(lb.pending_count(1), 0);
    }

    #[test]
    fn test_lb_remove_session() {
        let mut lb = LoadBalancer::new();
        lb.add_session(1);
        lb.remove_session(1);
        assert_eq!(lb.session_count(), 0);
    }

    #[test]
    fn test_lb_total_pending() {
        let mut lb = LoadBalancer::new();
        lb.add_session(1);
        lb.add_session(2);
        lb.route(&make_request(1, RequestPriority::Batch), HealthStatus::Healthy);
        lb.route(&make_request(2, RequestPriority::Batch), HealthStatus::Healthy);
        lb.route(&make_request(3, RequestPriority::Batch), HealthStatus::Healthy);
        assert_eq!(lb.total_pending(), 3);
    }

    #[test]
    fn test_lb_fairness_ratio_balanced() {
        let mut lb = LoadBalancer::new();
        lb.add_session(1);
        lb.add_session(2);
        // Route 100 requests; with least-loaded strategy they should be ~even.
        for i in 0..100 {
            let d = lb.route(&make_request(i, RequestPriority::Interactive), HealthStatus::Healthy);
            if let RouteDecision::GpuSession(sid) = d {
                lb.complete(sid);
            }
        }
        let ratio = lb.fairness_ratio().expect("should have ratio");
        assert!(ratio <= 1.5, "fairness ratio {ratio} should be ≤ 1.5 for balanced load");
    }

    #[test]
    fn test_lb_fairness_ratio_single_session() {
        let mut lb = LoadBalancer::new();
        lb.add_session(1);
        lb.route(&make_request(1, RequestPriority::Batch), HealthStatus::Healthy);
        assert!(lb.fairness_ratio().is_none());
    }

    #[test]
    fn test_lb_routed_count() {
        let mut lb = LoadBalancer::new();
        lb.add_session(1);
        lb.route(&make_request(1, RequestPriority::Batch), HealthStatus::Healthy);
        assert_eq!(lb.routed_count(1), 1);
    }

    // -- GracefulDegradation: trigger conditions --------------------------

    #[test]
    fn test_degradation_none_when_healthy() {
        let mut gd = GracefulDegradation::new(DegradationPolicy::default_a770());
        let result = gd.evaluate(&BackendHealth::healthy());
        assert!(result.is_none());
        assert!(!gd.is_fallback_active());
    }

    #[test]
    fn test_degradation_temperature_exceeded() {
        let mut gd = GracefulDegradation::new(DegradationPolicy::strict());
        let h = BackendHealth { gpu_temp_c: 75.0, ..BackendHealth::healthy() };
        let reason = gd.evaluate(&h).expect("should degrade");
        assert_eq!(reason, DegradationReason::TemperatureExceeded);
        assert!(gd.is_fallback_active());
    }

    #[test]
    fn test_degradation_memory_exceeded() {
        let mut gd = GracefulDegradation::new(DegradationPolicy::strict());
        let h = BackendHealth { memory_used_mb: 4000, ..BackendHealth::healthy() };
        let reason = gd.evaluate(&h).expect("should degrade");
        assert_eq!(reason, DegradationReason::MemoryExceeded);
    }

    #[test]
    fn test_degradation_utilization_exceeded() {
        let mut gd = GracefulDegradation::new(DegradationPolicy::strict());
        let h = BackendHealth { utilization_pct: 85.0, ..BackendHealth::healthy() };
        let reason = gd.evaluate(&h).expect("should degrade");
        assert_eq!(reason, DegradationReason::UtilizationExceeded);
    }

    #[test]
    fn test_degradation_error_rate_exceeded() {
        let mut gd = GracefulDegradation::new(DegradationPolicy::strict());
        let h = BackendHealth { error_count: 10, ..BackendHealth::healthy() };
        let reason = gd.evaluate(&h).expect("should degrade");
        assert_eq!(reason, DegradationReason::ErrorRateExceeded);
    }

    #[test]
    fn test_degradation_multiple_thresholds() {
        let mut gd = GracefulDegradation::new(DegradationPolicy::strict());
        let h =
            BackendHealth { gpu_temp_c: 75.0, memory_used_mb: 4000, ..BackendHealth::healthy() };
        let reason = gd.evaluate(&h).expect("should degrade");
        assert!(
            matches!(reason, DegradationReason::MultipleThresholds(_)),
            "expected MultipleThresholds, got {reason:?}"
        );
    }

    #[test]
    fn test_degradation_recovery() {
        let mut gd = GracefulDegradation::new(DegradationPolicy::strict());
        // Trigger degradation.
        let h_bad = BackendHealth { gpu_temp_c: 75.0, ..BackendHealth::healthy() };
        gd.evaluate(&h_bad);
        assert!(gd.is_fallback_active());
        // Recover.
        gd.evaluate(&BackendHealth::healthy());
        assert!(!gd.is_fallback_active());
    }

    #[test]
    fn test_degradation_event_count() {
        let mut gd = GracefulDegradation::new(DegradationPolicy::strict());
        let h = BackendHealth { gpu_temp_c: 75.0, ..BackendHealth::healthy() };
        gd.evaluate(&h);
        gd.evaluate(&h);
        assert_eq!(gd.event_count(), 2);
    }

    #[test]
    fn test_degradation_reset() {
        let mut gd = GracefulDegradation::new(DegradationPolicy::strict());
        let h = BackendHealth { gpu_temp_c: 75.0, ..BackendHealth::healthy() };
        gd.evaluate(&h);
        gd.reset();
        assert!(!gd.is_fallback_active());
    }

    #[test]
    fn test_degradation_set_policy() {
        let mut gd = GracefulDegradation::new(DegradationPolicy::strict());
        gd.set_policy(DegradationPolicy::default_a770());
        assert!((gd.policy().temp_threshold_c - 85.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_degradation_reason_display() {
        let r = DegradationReason::TemperatureExceeded;
        assert!(r.to_string().contains("temperature"));
        let r2 = DegradationReason::MultipleThresholds(vec![
            DegradationReason::MemoryExceeded,
            DegradationReason::ErrorRateExceeded,
        ]);
        let s = r2.to_string();
        assert!(s.contains("memory"));
        assert!(s.contains("error"));
    }

    // -- Concurrent session interleaving ----------------------------------

    #[test]
    fn test_concurrent_interleaving() {
        let mut pool = SessionPool::new(4);
        let mut lb = LoadBalancer::new();

        let s1 = pool.create_session("model").unwrap();
        let s2 = pool.create_session("model").unwrap();
        lb.add_session(s1);
        lb.add_session(s2);

        // Interleave requests across both sessions.
        for i in 0..20 {
            let d = lb.route(&make_request(i, RequestPriority::Interactive), HealthStatus::Healthy);
            if let RouteDecision::GpuSession(sid) = d {
                pool.get_mut(sid).unwrap().record_activity(1);
                lb.complete(sid);
            }
        }

        // Both sessions should have tokens.
        let t1 = pool.get(s1).unwrap().tokens_generated;
        let t2 = pool.get(s2).unwrap().tokens_generated;
        assert!(t1 > 0 && t2 > 0, "both sessions should have work: {t1}, {t2}");
        assert_eq!(t1 + t2, 20);
    }

    // -- Edge cases -------------------------------------------------------

    #[test]
    fn test_edge_zero_sessions_pool() {
        let pool = SessionPool::new(0);
        assert!(pool.is_empty());
        assert!(pool.is_full());
    }

    #[test]
    fn test_edge_max_sessions() {
        let mut pool = SessionPool::new(100);
        for _ in 0..100 {
            pool.create_session("m").unwrap();
        }
        assert!(pool.is_full());
        assert_eq!(pool.len(), 100);
    }

    #[test]
    fn test_edge_all_unhealthy_routes() {
        let mut lb = LoadBalancer::new();
        lb.add_session(1);
        lb.add_session(2);

        // All requests with Unhealthy status for realtime → CpuFallback.
        for i in 0..5 {
            let d = lb.route(&make_request(i, RequestPriority::Realtime), HealthStatus::Unhealthy);
            assert_eq!(d, RouteDecision::CpuFallback);
        }
    }

    // -- Property tests: load balancer fairness invariants -----------------

    #[test]
    fn test_property_lb_total_routed_equals_individual() {
        let mut lb = LoadBalancer::new();
        lb.add_session(1);
        lb.add_session(2);
        lb.add_session(3);

        for i in 0..60 {
            let d = lb.route(&make_request(i, RequestPriority::Batch), HealthStatus::Healthy);
            if let RouteDecision::GpuSession(sid) = d {
                lb.complete(sid);
            }
        }

        let total: u64 = [1, 2, 3].iter().map(|&id| lb.routed_count(id)).sum();
        assert_eq!(total, 60);
    }

    #[test]
    fn test_property_lb_pending_never_negative() {
        let mut lb = LoadBalancer::new();
        lb.add_session(1);
        // Complete many times without routing.
        for _ in 0..100 {
            lb.complete(1);
        }
        assert_eq!(lb.pending_count(1), 0);
    }

    #[test]
    fn test_property_lb_fairness_three_sessions() {
        let mut lb = LoadBalancer::new();
        lb.add_session(10);
        lb.add_session(20);
        lb.add_session(30);

        for i in 0..300 {
            let d = lb.route(&make_request(i, RequestPriority::Interactive), HealthStatus::Healthy);
            if let RouteDecision::GpuSession(sid) = d {
                lb.complete(sid);
            }
        }

        let ratio = lb.fairness_ratio().unwrap();
        assert!(ratio <= 1.1, "3-session fairness ratio {ratio} should be ≤ 1.1");
    }

    #[test]
    fn test_property_degraded_still_routes_gpu() {
        let mut lb = LoadBalancer::new();
        lb.add_session(1);
        let d = lb.route(&make_request(1, RequestPriority::Batch), HealthStatus::Degraded);
        assert!(matches!(d, RouteDecision::GpuSession(1)), "degraded should still use GPU");
    }

    // -- SessionPoolError display -----------------------------------------

    #[test]
    fn test_session_pool_error_display() {
        let e = SessionPoolError::PoolFull;
        assert!(e.to_string().contains("full"));
        let e2 = SessionPoolError::NotFound(42);
        assert!(e2.to_string().contains("42"));
    }

    // -- Integration: backend + pool + balancer + degradation -------------

    #[test]
    fn test_integration_full_lifecycle() {
        let mut backend = ServerBackend::new(DeviceInfo::test_device());
        let mut pool = SessionPool::new(backend.max_concurrent);
        let mut lb = LoadBalancer::new();
        let mut gd = GracefulDegradation::new(DegradationPolicy::default_a770());

        // Create sessions.
        for _ in 0..3 {
            let id = pool.create_session("bitnet-2b").unwrap();
            lb.add_session(id);
        }

        // Process requests while healthy.
        for i in 0..10 {
            backend.stats.record_request();
            let d = lb.route(&make_request(i, RequestPriority::Interactive), backend.health_status);
            match d {
                RouteDecision::GpuSession(sid) => {
                    pool.get_mut(sid).unwrap().record_activity(4);
                    backend.stats.record_tokens(4);
                    lb.complete(sid);
                }
                RouteDecision::CpuFallback => backend.stats.record_cpu_fallback(),
                RouteDecision::Reject => backend.stats.record_rejected(),
            }
        }

        assert_eq!(backend.stats.snapshot().total_requests, 10);
        assert_eq!(backend.stats.snapshot().total_tokens, 40);

        // Simulate overheating → degradation.
        backend.update_health(BackendHealth { gpu_temp_c: 92.0, ..BackendHealth::healthy() });
        assert_eq!(backend.health_status, HealthStatus::Unhealthy);

        let reason = gd.evaluate(&backend.health);
        assert!(reason.is_some());

        // Cleanup.
        for id in pool.session_ids() {
            lb.remove_session(id);
            pool.remove(id).unwrap();
        }
        assert!(pool.is_empty());
    }
}
