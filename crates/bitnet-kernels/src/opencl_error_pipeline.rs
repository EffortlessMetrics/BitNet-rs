//! Comprehensive error recovery pipeline for Intel Arc A770 GPU production reliability.
//!
//! Provides GPU error classification, retry strategies, circuit breaker pattern,
//! and health probing to handle OpenCL failures gracefully. All logic is
//! CPU-reference — no OpenCL runtime required.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// GpuError — classified GPU failure modes
// ---------------------------------------------------------------------------

/// GPU error types encountered during OpenCL kernel execution on Intel Arc A770.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuError {
    /// OpenCL kernel failed to launch (e.g. `CL_INVALID_WORK_GROUP_SIZE`).
    KernelLaunchFailed,
    /// Device ran out of global or local memory.
    OutOfMemory,
    /// Device was lost or disconnected (TDR, driver reset).
    DeviceLost,
    /// Kernel execution exceeded the configured timeout.
    Timeout,
    /// Driver crashed or returned an unrecoverable error.
    DriverCrash,
    /// Internal state inconsistency (e.g. stale command queue).
    InvalidState,
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KernelLaunchFailed => write!(f, "kernel launch failed"),
            Self::OutOfMemory => write!(f, "out of memory"),
            Self::DeviceLost => write!(f, "device lost"),
            Self::Timeout => write!(f, "timeout"),
            Self::DriverCrash => write!(f, "driver crash"),
            Self::InvalidState => write!(f, "invalid state"),
        }
    }
}

// ---------------------------------------------------------------------------
// RecoveryAction — what to do when an error occurs
// ---------------------------------------------------------------------------

/// Action to take in response to a [`GpuError`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryAction {
    /// Retry the operation up to `attempts` times.
    Retry(u32),
    /// Fall back to CPU kernel execution.
    FallbackCPU,
    /// Reduce the batch size and retry.
    ReduceBatch,
    /// Reset the OpenCL device / recreate context.
    ResetDevice,
    /// Abort — error is unrecoverable.
    Abort,
}

impl fmt::Display for RecoveryAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Retry(n) => write!(f, "retry({n})"),
            Self::FallbackCPU => write!(f, "fallback_cpu"),
            Self::ReduceBatch => write!(f, "reduce_batch"),
            Self::ResetDevice => write!(f, "reset_device"),
            Self::Abort => write!(f, "abort"),
        }
    }
}

// ---------------------------------------------------------------------------
// RecoveryPolicy — error→action mapping with priorities
// ---------------------------------------------------------------------------

/// Maps each [`GpuError`] to a priority-ordered list of [`RecoveryAction`]s.
#[derive(Debug, Clone)]
pub struct RecoveryPolicy {
    rules: HashMap<GpuError, Vec<RecoveryAction>>,
}

impl RecoveryPolicy {
    /// Create a policy with sensible defaults for Intel Arc A770.
    pub fn default_a770() -> Self {
        let mut rules = HashMap::new();
        rules.insert(
            GpuError::KernelLaunchFailed,
            vec![RecoveryAction::Retry(3), RecoveryAction::FallbackCPU],
        );
        rules.insert(
            GpuError::OutOfMemory,
            vec![RecoveryAction::ReduceBatch, RecoveryAction::FallbackCPU],
        );
        rules.insert(
            GpuError::DeviceLost,
            vec![RecoveryAction::ResetDevice, RecoveryAction::FallbackCPU],
        );
        rules
            .insert(GpuError::Timeout, vec![RecoveryAction::Retry(2), RecoveryAction::ReduceBatch]);
        rules.insert(
            GpuError::DriverCrash,
            vec![RecoveryAction::ResetDevice, RecoveryAction::Abort],
        );
        rules.insert(GpuError::InvalidState, vec![RecoveryAction::ResetDevice]);
        Self { rules }
    }

    /// Create an empty policy (no rules).
    pub fn empty() -> Self {
        Self { rules: HashMap::new() }
    }

    /// Set the actions for a given error type.
    pub fn set(&mut self, error: GpuError, actions: Vec<RecoveryAction>) {
        self.rules.insert(error, actions);
    }

    /// Look up recovery actions for an error, in priority order.
    pub fn actions_for(&self, error: GpuError) -> &[RecoveryAction] {
        self.rules.get(&error).map_or(&[], |v| v.as_slice())
    }
}

impl Default for RecoveryPolicy {
    fn default() -> Self {
        Self::default_a770()
    }
}

// ---------------------------------------------------------------------------
// RetryStrategy — delay computation for retries
// ---------------------------------------------------------------------------

/// Strategy for computing inter-retry delays.
#[derive(Debug, Clone, PartialEq)]
pub enum RetryStrategy {
    /// Fixed delay between retries.
    Fixed(Duration),
    /// Exponential backoff with a cap.
    Exponential { base: Duration, max: Duration },
    /// Exponential backoff with random jitter (±50%).
    Jittered { base: Duration },
}

impl RetryStrategy {
    /// Compute the delay for the `attempt`-th retry (0-indexed).
    pub fn delay(&self, attempt: u32) -> Duration {
        match self {
            Self::Fixed(d) => *d,
            Self::Exponential { base, max } => {
                let factor = 2u64.saturating_pow(attempt);
                let raw = base.saturating_mul(factor as u32);
                if raw > *max { *max } else { raw }
            }
            Self::Jittered { base } => {
                let factor = 2u64.saturating_pow(attempt);
                let raw_ms = base.as_millis() as u64 * factor;
                // Deterministic "jitter" based on attempt for reproducibility.
                let jitter_pct = 50i64 - (attempt as i64 * 17 % 100);
                let jittered = raw_ms as i64 + raw_ms as i64 * jitter_pct / 100;
                Duration::from_millis(jittered.max(1) as u64)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CircuitBreaker — state machine to prevent cascading failures
// ---------------------------------------------------------------------------

/// Circuit breaker states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation — requests pass through.
    Closed,
    /// Failures exceeded threshold — requests are rejected immediately.
    Open,
    /// Probing — one trial request allowed to test recovery.
    HalfOpen,
}

/// Circuit breaker that trips after repeated failures and resets after a cooldown.
#[derive(Debug)]
pub struct CircuitBreaker {
    state: CircuitState,
    failure_count: u32,
    success_count: u32,
    failure_threshold: u32,
    success_threshold: u32,
    reset_timeout: Duration,
    last_failure_time: Option<Instant>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker.
    ///
    /// * `failure_threshold` — consecutive failures to trip open.
    /// * `success_threshold` — consecutive successes in half-open to close.
    /// * `reset_timeout` — how long to stay open before transitioning to half-open.
    pub fn new(failure_threshold: u32, success_threshold: u32, reset_timeout: Duration) -> Self {
        Self {
            state: CircuitState::Closed,
            failure_count: 0,
            success_count: 0,
            failure_threshold,
            success_threshold,
            reset_timeout,
            last_failure_time: None,
        }
    }

    /// Current state.
    pub fn state(&self) -> CircuitState {
        self.state
    }

    /// Whether a request is allowed through.
    pub fn allow_request(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                if let Some(t) = self.last_failure_time
                    && t.elapsed() >= self.reset_timeout
                {
                    self.state = CircuitState::HalfOpen;
                    self.success_count = 0;
                    return true;
                }
                false
            }
            CircuitState::HalfOpen => true,
        }
    }

    /// Record a successful operation.
    pub fn record_success(&mut self) {
        match self.state {
            CircuitState::Closed => {
                self.failure_count = 0;
            }
            CircuitState::HalfOpen => {
                self.success_count += 1;
                if self.success_count >= self.success_threshold {
                    self.state = CircuitState::Closed;
                    self.failure_count = 0;
                    self.success_count = 0;
                }
            }
            CircuitState::Open => {}
        }
    }

    /// Record a failed operation.
    pub fn record_failure(&mut self) {
        self.last_failure_time = Some(Instant::now());
        match self.state {
            CircuitState::Closed => {
                self.failure_count += 1;
                if self.failure_count >= self.failure_threshold {
                    self.state = CircuitState::Open;
                }
            }
            CircuitState::HalfOpen => {
                self.state = CircuitState::Open;
                self.success_count = 0;
            }
            CircuitState::Open => {}
        }
    }

    /// Reset the breaker to closed.
    pub fn reset(&mut self) {
        self.state = CircuitState::Closed;
        self.failure_count = 0;
        self.success_count = 0;
        self.last_failure_time = None;
    }

    /// Number of consecutive failures recorded.
    pub fn failure_count(&self) -> u32 {
        self.failure_count
    }
}

// ---------------------------------------------------------------------------
// ErrorStats — aggregate error telemetry
// ---------------------------------------------------------------------------

/// Accumulated error statistics for observability.
#[derive(Debug)]
pub struct ErrorStats {
    total_errors: AtomicU64,
    total_recoveries: AtomicU64,
    by_type: std::sync::Mutex<HashMap<GpuError, u64>>,
    recovery_durations_ms: std::sync::Mutex<Vec<f64>>,
}

impl ErrorStats {
    /// Create empty stats.
    pub fn new() -> Self {
        Self {
            total_errors: AtomicU64::new(0),
            total_recoveries: AtomicU64::new(0),
            by_type: std::sync::Mutex::new(HashMap::new()),
            recovery_durations_ms: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Record an error occurrence.
    pub fn record_error(&self, error: GpuError) {
        self.total_errors.fetch_add(1, Ordering::Relaxed);
        *self.by_type.lock().unwrap().entry(error).or_insert(0) += 1;
    }

    /// Record a successful recovery and its latency.
    pub fn record_recovery(&self, duration_ms: f64) {
        self.total_recoveries.fetch_add(1, Ordering::Relaxed);
        self.recovery_durations_ms.lock().unwrap().push(duration_ms);
    }

    /// Total number of errors recorded.
    pub fn total_errors(&self) -> u64 {
        self.total_errors.load(Ordering::Relaxed)
    }

    /// Total number of successful recoveries.
    pub fn total_recoveries(&self) -> u64 {
        self.total_recoveries.load(Ordering::Relaxed)
    }

    /// Error count by type.
    pub fn errors_by_type(&self) -> HashMap<GpuError, u64> {
        self.by_type.lock().unwrap().clone()
    }

    /// Recovery success rate (0.0–1.0). Returns 0.0 if no errors recorded.
    pub fn recovery_success_rate(&self) -> f64 {
        let total = self.total_errors.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        self.total_recoveries.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// Average recovery latency in milliseconds.
    pub fn avg_recovery_ms(&self) -> f64 {
        let durations = self.recovery_durations_ms.lock().unwrap();
        if durations.is_empty() {
            return 0.0;
        }
        durations.iter().sum::<f64>() / durations.len() as f64
    }
}

impl Default for ErrorStats {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// HealthProbe — periodic device health check
// ---------------------------------------------------------------------------

/// Health status from a device probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// Device is responsive and healthy.
    Healthy,
    /// Device is responding slowly (possible thermal throttle / TDR risk).
    Degraded,
    /// Device is unresponsive.
    Unresponsive,
}

/// Periodic health checker that can preempt failures.
#[derive(Debug)]
pub struct HealthProbe {
    interval: Duration,
    timeout: Duration,
    last_check: Option<Instant>,
    last_status: HealthStatus,
    consecutive_degraded: u32,
    degraded_threshold: u32,
}

impl HealthProbe {
    /// Create a new health probe.
    ///
    /// * `interval` — how often to probe.
    /// * `timeout` — probe response deadline.
    /// * `degraded_threshold` — consecutive degraded checks before reporting unresponsive.
    pub fn new(interval: Duration, timeout: Duration, degraded_threshold: u32) -> Self {
        Self {
            interval,
            timeout,
            last_check: None,
            last_status: HealthStatus::Healthy,
            consecutive_degraded: 0,
            degraded_threshold,
        }
    }

    /// Whether a check is due based on the configured interval.
    pub fn is_check_due(&self) -> bool {
        match self.last_check {
            None => true,
            Some(t) => t.elapsed() >= self.interval,
        }
    }

    /// Record the result of a health probe.
    ///
    /// `response_time` — how long the device took to respond (if at all).
    pub fn record_check(&mut self, response_time: Option<Duration>) -> HealthStatus {
        self.last_check = Some(Instant::now());
        let status = match response_time {
            None => HealthStatus::Unresponsive,
            Some(d) if d >= self.timeout => {
                self.consecutive_degraded += 1;
                if self.consecutive_degraded >= self.degraded_threshold {
                    HealthStatus::Unresponsive
                } else {
                    HealthStatus::Degraded
                }
            }
            Some(_) => {
                self.consecutive_degraded = 0;
                HealthStatus::Healthy
            }
        };
        self.last_status = status;
        status
    }

    /// Last recorded health status.
    pub fn last_status(&self) -> HealthStatus {
        self.last_status
    }

    /// Configured probe timeout.
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Configured probe interval.
    pub fn interval(&self) -> Duration {
        self.interval
    }
}

// ---------------------------------------------------------------------------
// ErrorHandler — a single handler in the pipeline chain
// ---------------------------------------------------------------------------

/// A handler that can inspect an error and optionally produce a recovery action.
pub trait ErrorHandler: Send + Sync + fmt::Debug {
    /// Inspect the error and return an action if this handler can deal with it.
    fn handle(&self, error: GpuError, attempt: u32) -> Option<RecoveryAction>;

    /// Human-readable name for logging.
    fn name(&self) -> &str;
}

/// Handler that delegates to a [`RecoveryPolicy`].
#[derive(Debug, Clone)]
pub struct PolicyHandler {
    policy: RecoveryPolicy,
}

impl PolicyHandler {
    pub fn new(policy: RecoveryPolicy) -> Self {
        Self { policy }
    }
}

impl ErrorHandler for PolicyHandler {
    fn handle(&self, error: GpuError, attempt: u32) -> Option<RecoveryAction> {
        let actions = self.policy.actions_for(error);
        actions.get(attempt as usize).cloned()
    }

    fn name(&self) -> &str {
        "PolicyHandler"
    }
}

/// Handler that always returns [`RecoveryAction::FallbackCPU`] as a last resort.
#[derive(Debug, Clone, Copy)]
pub struct CpuFallbackHandler;

impl ErrorHandler for CpuFallbackHandler {
    fn handle(&self, _error: GpuError, _attempt: u32) -> Option<RecoveryAction> {
        Some(RecoveryAction::FallbackCPU)
    }

    fn name(&self) -> &str {
        "CpuFallbackHandler"
    }
}

/// Handler that always aborts — catch-all terminal.
#[derive(Debug, Clone, Copy)]
pub struct AbortHandler;

impl ErrorHandler for AbortHandler {
    fn handle(&self, _error: GpuError, _attempt: u32) -> Option<RecoveryAction> {
        Some(RecoveryAction::Abort)
    }

    fn name(&self) -> &str {
        "AbortHandler"
    }
}

// ---------------------------------------------------------------------------
// ErrorPipeline — chains handlers in priority order
// ---------------------------------------------------------------------------

/// Chains [`ErrorHandler`]s and returns the first matching recovery action.
#[derive(Debug)]
pub struct ErrorPipeline {
    handlers: Vec<Box<dyn ErrorHandler>>,
    stats: ErrorStats,
}

impl ErrorPipeline {
    /// Create a pipeline with the given handlers (evaluated in order).
    pub fn new(handlers: Vec<Box<dyn ErrorHandler>>) -> Self {
        Self { handlers, stats: ErrorStats::new() }
    }

    /// Create a default pipeline: policy → CPU fallback → abort.
    pub fn default_a770() -> Self {
        Self::new(vec![
            Box::new(PolicyHandler::new(RecoveryPolicy::default_a770())),
            Box::new(CpuFallbackHandler),
            Box::new(AbortHandler),
        ])
    }

    /// Process an error through the handler chain.
    pub fn handle(&self, error: GpuError, attempt: u32) -> RecoveryAction {
        let start = Instant::now();
        self.stats.record_error(error);
        for handler in &self.handlers {
            if let Some(action) = handler.handle(error, attempt) {
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                self.stats.record_recovery(elapsed_ms);
                return action;
            }
        }
        // Should be unreachable if AbortHandler is in the chain.
        RecoveryAction::Abort
    }

    /// Access accumulated error statistics.
    pub fn stats(&self) -> &ErrorStats {
        &self.stats
    }

    /// Number of handlers in the chain.
    pub fn handler_count(&self) -> usize {
        self.handlers.len()
    }
}

impl Default for ErrorPipeline {
    fn default() -> Self {
        Self::default_a770()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // GpuError display
    // -----------------------------------------------------------------------

    #[test]
    fn test_gpu_error_display_kernel_launch_failed() {
        assert_eq!(GpuError::KernelLaunchFailed.to_string(), "kernel launch failed");
    }

    #[test]
    fn test_gpu_error_display_out_of_memory() {
        assert_eq!(GpuError::OutOfMemory.to_string(), "out of memory");
    }

    #[test]
    fn test_gpu_error_display_device_lost() {
        assert_eq!(GpuError::DeviceLost.to_string(), "device lost");
    }

    #[test]
    fn test_gpu_error_display_timeout() {
        assert_eq!(GpuError::Timeout.to_string(), "timeout");
    }

    #[test]
    fn test_gpu_error_display_driver_crash() {
        assert_eq!(GpuError::DriverCrash.to_string(), "driver crash");
    }

    #[test]
    fn test_gpu_error_display_invalid_state() {
        assert_eq!(GpuError::InvalidState.to_string(), "invalid state");
    }

    // -----------------------------------------------------------------------
    // RecoveryAction display
    // -----------------------------------------------------------------------

    #[test]
    fn test_recovery_action_display() {
        assert_eq!(RecoveryAction::Retry(3).to_string(), "retry(3)");
        assert_eq!(RecoveryAction::FallbackCPU.to_string(), "fallback_cpu");
        assert_eq!(RecoveryAction::ReduceBatch.to_string(), "reduce_batch");
        assert_eq!(RecoveryAction::ResetDevice.to_string(), "reset_device");
        assert_eq!(RecoveryAction::Abort.to_string(), "abort");
    }

    // -----------------------------------------------------------------------
    // RecoveryPolicy — error→action mapping
    // -----------------------------------------------------------------------

    #[test]
    fn test_policy_kernel_launch_failed_maps_to_retry_then_fallback() {
        let policy = RecoveryPolicy::default_a770();
        let actions = policy.actions_for(GpuError::KernelLaunchFailed);
        assert_eq!(actions, &[RecoveryAction::Retry(3), RecoveryAction::FallbackCPU]);
    }

    #[test]
    fn test_policy_out_of_memory_maps_to_reduce_then_fallback() {
        let policy = RecoveryPolicy::default_a770();
        let actions = policy.actions_for(GpuError::OutOfMemory);
        assert_eq!(actions, &[RecoveryAction::ReduceBatch, RecoveryAction::FallbackCPU]);
    }

    #[test]
    fn test_policy_device_lost_maps_to_reset_then_fallback() {
        let policy = RecoveryPolicy::default_a770();
        let actions = policy.actions_for(GpuError::DeviceLost);
        assert_eq!(actions, &[RecoveryAction::ResetDevice, RecoveryAction::FallbackCPU]);
    }

    #[test]
    fn test_policy_timeout_maps_to_retry_then_reduce() {
        let policy = RecoveryPolicy::default_a770();
        let actions = policy.actions_for(GpuError::Timeout);
        assert_eq!(actions, &[RecoveryAction::Retry(2), RecoveryAction::ReduceBatch]);
    }

    #[test]
    fn test_policy_driver_crash_maps_to_reset_then_abort() {
        let policy = RecoveryPolicy::default_a770();
        let actions = policy.actions_for(GpuError::DriverCrash);
        assert_eq!(actions, &[RecoveryAction::ResetDevice, RecoveryAction::Abort]);
    }

    #[test]
    fn test_policy_invalid_state_maps_to_reset() {
        let policy = RecoveryPolicy::default_a770();
        let actions = policy.actions_for(GpuError::InvalidState);
        assert_eq!(actions, &[RecoveryAction::ResetDevice]);
    }

    #[test]
    fn test_policy_empty_returns_no_actions() {
        let policy = RecoveryPolicy::empty();
        assert!(policy.actions_for(GpuError::Timeout).is_empty());
    }

    #[test]
    fn test_policy_set_overwrites_existing() {
        let mut policy = RecoveryPolicy::default_a770();
        policy.set(GpuError::Timeout, vec![RecoveryAction::Abort]);
        assert_eq!(policy.actions_for(GpuError::Timeout), &[RecoveryAction::Abort]);
    }

    // -----------------------------------------------------------------------
    // RetryStrategy
    // -----------------------------------------------------------------------

    #[test]
    fn test_retry_fixed_delay_is_constant() {
        let s = RetryStrategy::Fixed(Duration::from_millis(100));
        assert_eq!(s.delay(0), Duration::from_millis(100));
        assert_eq!(s.delay(1), Duration::from_millis(100));
        assert_eq!(s.delay(5), Duration::from_millis(100));
    }

    #[test]
    fn test_retry_exponential_doubles() {
        let s = RetryStrategy::Exponential {
            base: Duration::from_millis(100),
            max: Duration::from_secs(10),
        };
        assert_eq!(s.delay(0), Duration::from_millis(100));
        assert_eq!(s.delay(1), Duration::from_millis(200));
        assert_eq!(s.delay(2), Duration::from_millis(400));
        assert_eq!(s.delay(3), Duration::from_millis(800));
    }

    #[test]
    fn test_retry_exponential_caps_at_max() {
        let s = RetryStrategy::Exponential {
            base: Duration::from_millis(100),
            max: Duration::from_millis(500),
        };
        assert_eq!(s.delay(0), Duration::from_millis(100));
        assert_eq!(s.delay(3), Duration::from_millis(500));
        assert_eq!(s.delay(10), Duration::from_millis(500));
    }

    #[test]
    fn test_retry_jittered_is_nonzero() {
        let s = RetryStrategy::Jittered { base: Duration::from_millis(100) };
        for attempt in 0..5 {
            assert!(s.delay(attempt).as_millis() >= 1);
        }
    }

    #[test]
    fn test_retry_jittered_grows_with_attempt() {
        let s = RetryStrategy::Jittered { base: Duration::from_millis(100) };
        // Jittered delays should generally increase (base doubles each attempt).
        let d0 = s.delay(0).as_millis();
        let d2 = s.delay(2).as_millis();
        assert!(d2 > d0, "d2={d2} should be > d0={d0}");
    }

    // -----------------------------------------------------------------------
    // CircuitBreaker — state machine
    // -----------------------------------------------------------------------

    #[test]
    fn test_circuit_breaker_starts_closed() {
        let cb = CircuitBreaker::new(3, 1, Duration::from_secs(10));
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_allows_request_when_closed() {
        let mut cb = CircuitBreaker::new(3, 1, Duration::from_secs(10));
        assert!(cb.allow_request());
    }

    #[test]
    fn test_circuit_breaker_opens_after_threshold_failures() {
        let mut cb = CircuitBreaker::new(3, 1, Duration::from_secs(10));
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_rejects_when_open() {
        let mut cb = CircuitBreaker::new(1, 1, Duration::from_mins(1));
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.allow_request());
    }

    #[test]
    fn test_circuit_breaker_transitions_to_half_open_after_timeout() {
        let mut cb = CircuitBreaker::new(1, 1, Duration::from_millis(1));
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        std::thread::sleep(Duration::from_millis(5));
        assert!(cb.allow_request());
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn test_circuit_breaker_closes_on_half_open_success() {
        let mut cb = CircuitBreaker::new(1, 1, Duration::from_millis(1));
        cb.record_failure();
        std::thread::sleep(Duration::from_millis(5));
        cb.allow_request();
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_reopens_on_half_open_failure() {
        let mut cb = CircuitBreaker::new(1, 1, Duration::from_millis(1));
        cb.record_failure();
        std::thread::sleep(Duration::from_millis(5));
        cb.allow_request();
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_success_resets_failure_count() {
        let mut cb = CircuitBreaker::new(3, 1, Duration::from_secs(10));
        cb.record_failure();
        cb.record_failure();
        cb.record_success();
        assert_eq!(cb.failure_count(), 0);
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_reset() {
        let mut cb = CircuitBreaker::new(1, 1, Duration::from_secs(10));
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.failure_count(), 0);
    }

    #[test]
    fn test_circuit_breaker_half_open_needs_n_successes() {
        let mut cb = CircuitBreaker::new(1, 3, Duration::from_millis(1));
        cb.record_failure();
        std::thread::sleep(Duration::from_millis(5));
        cb.allow_request();
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    // -----------------------------------------------------------------------
    // ErrorStats
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_stats_initial_values() {
        let stats = ErrorStats::new();
        assert_eq!(stats.total_errors(), 0);
        assert_eq!(stats.total_recoveries(), 0);
        assert_eq!(stats.recovery_success_rate(), 0.0);
        assert_eq!(stats.avg_recovery_ms(), 0.0);
    }

    #[test]
    fn test_error_stats_record_errors() {
        let stats = ErrorStats::new();
        stats.record_error(GpuError::Timeout);
        stats.record_error(GpuError::Timeout);
        stats.record_error(GpuError::OutOfMemory);
        assert_eq!(stats.total_errors(), 3);
        let by_type = stats.errors_by_type();
        assert_eq!(by_type[&GpuError::Timeout], 2);
        assert_eq!(by_type[&GpuError::OutOfMemory], 1);
    }

    #[test]
    fn test_error_stats_recovery_rate() {
        let stats = ErrorStats::new();
        stats.record_error(GpuError::Timeout);
        stats.record_error(GpuError::Timeout);
        stats.record_recovery(1.5);
        assert!((stats.recovery_success_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_error_stats_avg_recovery_ms() {
        let stats = ErrorStats::new();
        stats.record_recovery(2.0);
        stats.record_recovery(4.0);
        assert!((stats.avg_recovery_ms() - 3.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // HealthProbe
    // -----------------------------------------------------------------------

    #[test]
    fn test_health_probe_initial_check_due() {
        let probe = HealthProbe::new(Duration::from_secs(1), Duration::from_millis(100), 3);
        assert!(probe.is_check_due());
    }

    #[test]
    fn test_health_probe_healthy_response() {
        let mut probe = HealthProbe::new(Duration::from_secs(1), Duration::from_millis(100), 3);
        let status = probe.record_check(Some(Duration::from_millis(10)));
        assert_eq!(status, HealthStatus::Healthy);
        assert_eq!(probe.last_status(), HealthStatus::Healthy);
    }

    #[test]
    fn test_health_probe_degraded_response() {
        let mut probe = HealthProbe::new(Duration::from_secs(1), Duration::from_millis(100), 3);
        let status = probe.record_check(Some(Duration::from_millis(200)));
        assert_eq!(status, HealthStatus::Degraded);
    }

    #[test]
    fn test_health_probe_unresponsive_on_none() {
        let mut probe = HealthProbe::new(Duration::from_secs(1), Duration::from_millis(100), 3);
        let status = probe.record_check(None);
        assert_eq!(status, HealthStatus::Unresponsive);
    }

    #[test]
    fn test_health_probe_consecutive_degraded_becomes_unresponsive() {
        let mut probe = HealthProbe::new(Duration::from_secs(1), Duration::from_millis(100), 2);
        probe.record_check(Some(Duration::from_millis(200)));
        let status = probe.record_check(Some(Duration::from_millis(200)));
        assert_eq!(status, HealthStatus::Unresponsive);
    }

    #[test]
    fn test_health_probe_healthy_resets_degraded_count() {
        let mut probe = HealthProbe::new(Duration::from_secs(1), Duration::from_millis(100), 2);
        probe.record_check(Some(Duration::from_millis(200)));
        probe.record_check(Some(Duration::from_millis(10)));
        let status = probe.record_check(Some(Duration::from_millis(200)));
        assert_eq!(status, HealthStatus::Degraded);
    }

    #[test]
    fn test_health_probe_not_due_after_recent_check() {
        let mut probe = HealthProbe::new(Duration::from_mins(1), Duration::from_millis(100), 3);
        probe.record_check(Some(Duration::from_millis(10)));
        assert!(!probe.is_check_due());
    }

    #[test]
    fn test_health_probe_accessors() {
        let probe = HealthProbe::new(Duration::from_secs(5), Duration::from_millis(200), 3);
        assert_eq!(probe.interval(), Duration::from_secs(5));
        assert_eq!(probe.timeout(), Duration::from_millis(200));
    }

    // -----------------------------------------------------------------------
    // ErrorHandler implementations
    // -----------------------------------------------------------------------

    #[test]
    fn test_policy_handler_returns_first_action() {
        let handler = PolicyHandler::new(RecoveryPolicy::default_a770());
        assert_eq!(handler.handle(GpuError::KernelLaunchFailed, 0), Some(RecoveryAction::Retry(3)),);
    }

    #[test]
    fn test_policy_handler_returns_second_action_on_attempt_1() {
        let handler = PolicyHandler::new(RecoveryPolicy::default_a770());
        assert_eq!(
            handler.handle(GpuError::KernelLaunchFailed, 1),
            Some(RecoveryAction::FallbackCPU),
        );
    }

    #[test]
    fn test_policy_handler_returns_none_when_exhausted() {
        let handler = PolicyHandler::new(RecoveryPolicy::default_a770());
        assert_eq!(handler.handle(GpuError::KernelLaunchFailed, 99), None);
    }

    #[test]
    fn test_cpu_fallback_handler_always_returns_fallback() {
        let handler = CpuFallbackHandler;
        assert_eq!(handler.handle(GpuError::DriverCrash, 0), Some(RecoveryAction::FallbackCPU));
        assert_eq!(handler.handle(GpuError::Timeout, 10), Some(RecoveryAction::FallbackCPU));
    }

    #[test]
    fn test_abort_handler_always_returns_abort() {
        let handler = AbortHandler;
        assert_eq!(handler.handle(GpuError::DeviceLost, 0), Some(RecoveryAction::Abort));
    }

    // -----------------------------------------------------------------------
    // ErrorPipeline
    // -----------------------------------------------------------------------

    #[test]
    fn test_pipeline_default_handler_count() {
        let pipeline = ErrorPipeline::default_a770();
        assert_eq!(pipeline.handler_count(), 3);
    }

    #[test]
    fn test_pipeline_returns_policy_action_first() {
        let pipeline = ErrorPipeline::default_a770();
        let action = pipeline.handle(GpuError::KernelLaunchFailed, 0);
        assert_eq!(action, RecoveryAction::Retry(3));
    }

    #[test]
    fn test_pipeline_falls_through_to_cpu_fallback() {
        let pipeline = ErrorPipeline::default_a770();
        // Attempt 99 exhausts the policy handler; CPU fallback catches it.
        let action = pipeline.handle(GpuError::KernelLaunchFailed, 99);
        assert_eq!(action, RecoveryAction::FallbackCPU);
    }

    #[test]
    fn test_pipeline_empty_policy_falls_to_cpu() {
        let pipeline = ErrorPipeline::new(vec![
            Box::new(PolicyHandler::new(RecoveryPolicy::empty())),
            Box::new(CpuFallbackHandler),
        ]);
        let action = pipeline.handle(GpuError::Timeout, 0);
        assert_eq!(action, RecoveryAction::FallbackCPU);
    }

    #[test]
    fn test_pipeline_abort_only() {
        let pipeline = ErrorPipeline::new(vec![Box::new(AbortHandler)]);
        assert_eq!(pipeline.handle(GpuError::DeviceLost, 0), RecoveryAction::Abort);
    }

    #[test]
    fn test_pipeline_stats_accumulate() {
        let pipeline = ErrorPipeline::default_a770();
        pipeline.handle(GpuError::Timeout, 0);
        pipeline.handle(GpuError::OutOfMemory, 0);
        assert_eq!(pipeline.stats().total_errors(), 2);
        assert_eq!(pipeline.stats().total_recoveries(), 2);
    }

    #[test]
    fn test_pipeline_stats_recovery_rate() {
        let pipeline = ErrorPipeline::default_a770();
        pipeline.handle(GpuError::Timeout, 0);
        pipeline.handle(GpuError::Timeout, 0);
        // Both errors should be recovered (pipeline always finds an action).
        assert!((pipeline.stats().recovery_success_rate() - 1.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_retries_exhausted_falls_through() {
        let pipeline = ErrorPipeline::new(vec![
            Box::new(PolicyHandler::new(RecoveryPolicy::default_a770())),
            Box::new(AbortHandler),
        ]);
        // InvalidState has only one action (ResetDevice at attempt 0).
        // Attempt 1 exhausts the policy, falls through to AbortHandler.
        let action = pipeline.handle(GpuError::InvalidState, 1);
        assert_eq!(action, RecoveryAction::Abort);
    }

    #[test]
    fn test_device_lost_first_action_is_reset() {
        let pipeline = ErrorPipeline::default_a770();
        assert_eq!(pipeline.handle(GpuError::DeviceLost, 0), RecoveryAction::ResetDevice);
    }

    #[test]
    fn test_device_lost_second_action_is_fallback() {
        let pipeline = ErrorPipeline::default_a770();
        assert_eq!(pipeline.handle(GpuError::DeviceLost, 1), RecoveryAction::FallbackCPU);
    }

    #[test]
    fn test_concurrent_stats_recording() {
        use std::sync::Arc;
        let stats = Arc::new(ErrorStats::new());
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let s = Arc::clone(&stats);
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        s.record_error(GpuError::Timeout);
                        s.record_recovery(1.0);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(stats.total_errors(), 400);
        assert_eq!(stats.total_recoveries(), 400);
    }

    // -----------------------------------------------------------------------
    // Property-style: circuit breaker state transitions are valid
    // -----------------------------------------------------------------------

    #[test]
    fn test_circuit_breaker_state_transitions_are_valid() {
        // Enumerate many failure/success sequences and assert no invalid state.
        let mut cb = CircuitBreaker::new(2, 2, Duration::from_millis(1));
        let ops = [true, false, false, true, true, false, false, false, true, true, true, false];
        for &success in &ops {
            let prev = cb.state();
            if success {
                cb.record_success();
            } else {
                cb.record_failure();
            }
            let next = cb.state();
            // Valid transitions:
            // Closed  → Closed | Open
            // Open    → Open
            // HalfOpen→ Closed | Open
            match prev {
                CircuitState::Closed => {
                    assert!(
                        next == CircuitState::Closed || next == CircuitState::Open,
                        "invalid transition from Closed to {next:?}"
                    );
                }
                CircuitState::Open => {
                    assert_eq!(next, CircuitState::Open, "Open should stay Open via record_*");
                }
                CircuitState::HalfOpen => {
                    assert!(
                        next == CircuitState::Closed
                            || next == CircuitState::Open
                            || next == CircuitState::HalfOpen,
                        "invalid transition from HalfOpen to {next:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_circuit_breaker_allow_request_transitions() {
        // allow_request can move Open→HalfOpen (after timeout).
        let mut cb = CircuitBreaker::new(1, 1, Duration::from_millis(1));
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        std::thread::sleep(Duration::from_millis(5));
        let allowed = cb.allow_request();
        assert!(allowed);
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn test_gpu_error_equality_and_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(GpuError::Timeout);
        set.insert(GpuError::Timeout);
        set.insert(GpuError::OutOfMemory);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_recovery_action_equality() {
        assert_eq!(RecoveryAction::Retry(3), RecoveryAction::Retry(3));
        assert_ne!(RecoveryAction::Retry(3), RecoveryAction::Retry(2));
        assert_ne!(RecoveryAction::FallbackCPU, RecoveryAction::Abort);
    }
}
