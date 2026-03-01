//! GPU resource limiter with memory, compute, and kernel budgets.
//!
//! Provides a [`ResourceLimiter`] that tracks usage against configurable
//! limits and supports a reservation system with timeout/expiry,
//! overcommit control, and priority-based admission.

use std::fmt;

// ── Types ─────────────────────────────────────────────────────────────────

/// Hard caps for GPU resource consumption.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceLimits {
    pub max_memory_bytes: u64,
    pub max_compute_time_ms: u64,
    pub max_concurrent_kernels: u32,
    pub max_batch_size: u32,
    pub max_sequence_length: u32,
    pub max_active_requests: u32,
}

/// Current (and peak) resource usage snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceUsage {
    pub memory_bytes: u64,
    pub compute_time_ms: u64,
    pub active_kernels: u32,
    pub active_requests: u32,
    pub peak_memory_bytes: u64,
    pub peak_kernels: u32,
}

impl ResourceUsage {
    const fn new() -> Self {
        Self {
            memory_bytes: 0,
            compute_time_ms: 0,
            active_kernels: 0,
            active_requests: 0,
            peak_memory_bytes: 0,
            peak_kernels: 0,
        }
    }
}

/// A reservation holding resources against the budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reservation {
    pub id: u64,
    pub memory_bytes: u64,
    pub compute_budget_ms: u64,
    pub created_at: u64,
    pub expires_at: Option<u64>,
}

/// Behaviour knobs for the limiter.
#[derive(Debug, Clone, PartialEq)]
pub struct LimiterConfig {
    pub enforce_strict: bool,
    pub allow_overcommit: bool,
    pub overcommit_ratio: f64,
    pub reservation_timeout_ms: Option<u64>,
}

impl Default for LimiterConfig {
    fn default() -> Self {
        Self {
            enforce_strict: false,
            allow_overcommit: false,
            overcommit_ratio: 1.0,
            reservation_timeout_ms: None,
        }
    }
}

/// A request for resources.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceRequest {
    pub memory_bytes: u64,
    pub compute_budget_ms: u64,
    pub priority: RequestPriority,
}

/// Priority tier for admission control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RequestPriority {
    Low,
    Normal,
    High,
    System,
}

/// Outcome of a limit check.
#[derive(Debug, Clone, PartialEq)]
pub enum LimitResult {
    Allowed,
    WaitRequired { estimated_wait_ms: u64 },
    Denied { reason: String },
    OvercommitAllowed { current_usage_pct: f64 },
}

/// Errors returned by reservation operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceError {
    LimitExceeded(String),
    ReservationNotFound(u64),
}

impl fmt::Display for ResourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LimitExceeded(msg) => write!(f, "limit exceeded: {msg}"),
            Self::ReservationNotFound(id) => {
                write!(f, "reservation {id} not found")
            }
        }
    }
}

impl std::error::Error for ResourceError {}

// ── Core limiter ──────────────────────────────────────────────────────────

/// Tracks GPU resource consumption against configurable limits.
pub struct ResourceLimiter {
    limits: ResourceLimits,
    usage: ResourceUsage,
    reservations: Vec<Reservation>,
    config: LimiterConfig,
    next_reservation_id: u64,
}

impl ResourceLimiter {
    /// Create a limiter with the given caps and behaviour config.
    pub const fn new(limits: ResourceLimits, config: LimiterConfig) -> Self {
        Self {
            limits,
            usage: ResourceUsage::new(),
            reservations: Vec::new(),
            config,
            next_reservation_id: 1,
        }
    }

    // ── queries ───────────────────────────────────────────────────────

    /// Snapshot of current usage (including peaks).
    pub fn current_usage(&self) -> ResourceUsage {
        self.usage.clone()
    }

    /// Unreserved memory still available.
    pub const fn available_memory(&self) -> u64 {
        self.limits.max_memory_bytes.saturating_sub(self.usage.memory_bytes)
    }

    /// Overall utilisation as a percentage of the memory limit.
    pub fn utilization_percent(&self) -> f64 {
        if self.limits.max_memory_bytes == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        {
            self.usage.memory_bytes as f64 / self.limits.max_memory_bytes as f64 * 100.0
        }
    }

    // ── admission ─────────────────────────────────────────────────────

    /// Pure admission check — does **not** mutate usage.
    pub fn check(&self, request: &ResourceRequest) -> LimitResult {
        // System priority always passes.
        if request.priority == RequestPriority::System {
            return LimitResult::Allowed;
        }

        // Memory check.
        let total_committed = self.committed_memory();
        let needed = total_committed + request.memory_bytes;

        if needed > self.limits.max_memory_bytes {
            if self.config.allow_overcommit {
                #[allow(clippy::cast_precision_loss)]
                let effective_limit =
                    self.limits.max_memory_bytes as f64 * self.config.overcommit_ratio;
                #[allow(clippy::cast_precision_loss)]
                if (needed as f64) <= effective_limit {
                    #[allow(clippy::cast_precision_loss)]
                    let pct = total_committed as f64 / self.limits.max_memory_bytes as f64 * 100.0;
                    return LimitResult::OvercommitAllowed { current_usage_pct: pct };
                }
            }

            if self.config.enforce_strict {
                return LimitResult::Denied { reason: "memory limit exceeded".into() };
            }

            // Non-strict: suggest waiting.
            let excess = needed - self.limits.max_memory_bytes;
            return LimitResult::WaitRequired { estimated_wait_ms: excess.min(10_000) };
        }

        // Kernel concurrency check.
        if self.usage.active_kernels >= self.limits.max_concurrent_kernels {
            if self.config.enforce_strict {
                return LimitResult::Denied { reason: "concurrent kernel limit reached".into() };
            }
            return LimitResult::WaitRequired { estimated_wait_ms: 100 };
        }

        // Active-request check.
        if self.usage.active_requests >= self.limits.max_active_requests {
            if self.config.enforce_strict {
                return LimitResult::Denied { reason: "active request limit reached".into() };
            }
            return LimitResult::WaitRequired { estimated_wait_ms: 50 };
        }

        // Compute budget check.
        if request.compute_budget_ms > self.limits.max_compute_time_ms {
            return LimitResult::Denied { reason: "compute budget exceeds limit".into() };
        }

        LimitResult::Allowed
    }

    /// Check + optionally block. Equivalent to [`check`](Self::check)
    /// in this synchronous implementation.
    pub fn enforce(&self, request: &ResourceRequest) -> LimitResult {
        self.check(request)
    }

    // ── reservations ──────────────────────────────────────────────────

    /// Reserve resources. Returns a [`Reservation`] on success.
    pub fn reserve(
        &mut self,
        request: &ResourceRequest,
        current_time_ms: u64,
    ) -> Result<Reservation, ResourceError> {
        let result = self.check(request);
        match result {
            LimitResult::Allowed | LimitResult::OvercommitAllowed { .. } => {}
            LimitResult::Denied { reason } => {
                return Err(ResourceError::LimitExceeded(reason));
            }
            LimitResult::WaitRequired { .. } => {
                return Err(ResourceError::LimitExceeded(
                    "resources not immediately available".into(),
                ));
            }
        }

        let id = self.next_reservation_id;
        self.next_reservation_id += 1;

        let expires_at = self.config.reservation_timeout_ms.map(|t| current_time_ms + t);

        let reservation = Reservation {
            id,
            memory_bytes: request.memory_bytes,
            compute_budget_ms: request.compute_budget_ms,
            created_at: current_time_ms,
            expires_at,
        };
        self.reservations.push(reservation.clone());

        // Update usage tracking.
        self.usage.memory_bytes += request.memory_bytes;
        self.usage.active_requests += 1;
        self.update_peaks();

        Ok(reservation)
    }

    /// Release a reservation by id, freeing its resources.
    pub fn release(&mut self, reservation_id: u64) -> Result<(), ResourceError> {
        let pos = self
            .reservations
            .iter()
            .position(|r| r.id == reservation_id)
            .ok_or(ResourceError::ReservationNotFound(reservation_id))?;

        let reservation = self.reservations.remove(pos);
        self.usage.memory_bytes = self.usage.memory_bytes.saturating_sub(reservation.memory_bytes);
        self.usage.active_requests = self.usage.active_requests.saturating_sub(1);

        Ok(())
    }

    /// Remove reservations whose `expires_at` ≤ `current_time_ms`.
    pub fn expire_stale_reservations(&mut self, current_time_ms: u64) -> u32 {
        let mut expired_count = 0u32;
        let mut i = 0;
        while i < self.reservations.len() {
            if let Some(exp) = self.reservations[i].expires_at
                && exp <= current_time_ms
            {
                let r = self.reservations.remove(i);
                self.usage.memory_bytes = self.usage.memory_bytes.saturating_sub(r.memory_bytes);
                self.usage.active_requests = self.usage.active_requests.saturating_sub(1);
                expired_count += 1;
                continue;
            }
            i += 1;
        }
        expired_count
    }

    // ── mutations ─────────────────────────────────────────────────────

    /// Record that a kernel has started.
    pub const fn add_active_kernel(&mut self) {
        self.usage.active_kernels += 1;
        self.update_peaks();
    }

    /// Record that a kernel has finished.
    pub const fn remove_active_kernel(&mut self) {
        self.usage.active_kernels = self.usage.active_kernels.saturating_sub(1);
    }

    /// Reset all counters (but keep limits and config).
    pub fn reset_usage(&mut self) {
        self.usage = ResourceUsage::new();
        self.reservations.clear();
        self.next_reservation_id = 1;
    }

    // ── internal ──────────────────────────────────────────────────────

    fn committed_memory(&self) -> u64 {
        let reserved: u64 = self.reservations.iter().map(|r| r.memory_bytes).sum();
        self.usage.memory_bytes + reserved
            - self.reservations.iter().map(|r| r.memory_bytes).sum::<u64>()
    }

    const fn update_peaks(&mut self) {
        if self.usage.memory_bytes > self.usage.peak_memory_bytes {
            self.usage.peak_memory_bytes = self.usage.memory_bytes;
        }
        if self.usage.active_kernels > self.usage.peak_kernels {
            self.usage.peak_kernels = self.usage.active_kernels;
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────

    fn default_limits() -> ResourceLimits {
        ResourceLimits {
            max_memory_bytes: 1_073_741_824, // 1 GiB
            max_compute_time_ms: 10_000,
            max_concurrent_kernels: 8,
            max_batch_size: 32,
            max_sequence_length: 2048,
            max_active_requests: 16,
        }
    }

    fn default_config() -> LimiterConfig {
        LimiterConfig::default()
    }

    fn strict_config() -> LimiterConfig {
        LimiterConfig { enforce_strict: true, ..LimiterConfig::default() }
    }

    fn overcommit_config(ratio: f64) -> LimiterConfig {
        LimiterConfig {
            allow_overcommit: true,
            overcommit_ratio: ratio,
            ..LimiterConfig::default()
        }
    }

    fn normal_request(mem: u64, compute: u64) -> ResourceRequest {
        ResourceRequest {
            memory_bytes: mem,
            compute_budget_ms: compute,
            priority: RequestPriority::Normal,
        }
    }

    fn system_request(mem: u64, compute: u64) -> ResourceRequest {
        ResourceRequest {
            memory_bytes: mem,
            compute_budget_ms: compute,
            priority: RequestPriority::System,
        }
    }

    fn high_request(mem: u64, compute: u64) -> ResourceRequest {
        ResourceRequest {
            memory_bytes: mem,
            compute_budget_ms: compute,
            priority: RequestPriority::High,
        }
    }

    fn low_request(mem: u64, compute: u64) -> ResourceRequest {
        ResourceRequest {
            memory_bytes: mem,
            compute_budget_ms: compute,
            priority: RequestPriority::Low,
        }
    }

    // ── basic admission ──────────────────────────────────────────────

    #[test]
    fn request_within_limits_allowed() {
        let limiter = ResourceLimiter::new(default_limits(), default_config());
        let result = limiter.check(&normal_request(1024, 100));
        assert_eq!(result, LimitResult::Allowed);
    }

    #[test]
    fn zero_resource_request_allowed() {
        let limiter = ResourceLimiter::new(default_limits(), default_config());
        assert_eq!(limiter.check(&normal_request(0, 0)), LimitResult::Allowed);
    }

    #[test]
    fn request_exactly_at_limit_allowed() {
        let limits = ResourceLimits { max_memory_bytes: 1024, ..default_limits() };
        let limiter = ResourceLimiter::new(limits, default_config());
        assert_eq!(limiter.check(&normal_request(1024, 100)), LimitResult::Allowed);
    }

    #[test]
    fn memory_limit_exceeded_strict_denied() {
        let limits = ResourceLimits { max_memory_bytes: 512, ..default_limits() };
        let limiter = ResourceLimiter::new(limits, strict_config());
        let result = limiter.check(&normal_request(1024, 100));
        assert!(matches!(result, LimitResult::Denied { .. }));
    }

    #[test]
    fn memory_limit_exceeded_nonstrict_wait() {
        let limits = ResourceLimits { max_memory_bytes: 512, ..default_limits() };
        let limiter = ResourceLimiter::new(limits, default_config());
        let result = limiter.check(&normal_request(1024, 100));
        assert!(matches!(result, LimitResult::WaitRequired { .. }));
    }

    #[test]
    fn compute_budget_exceeds_limit_denied() {
        let limiter = ResourceLimiter::new(default_limits(), default_config());
        let result = limiter.check(&normal_request(1024, 999_999));
        assert!(matches!(result, LimitResult::Denied { .. }));
    }

    // ── kernel concurrency ───────────────────────────────────────────

    #[test]
    fn concurrent_kernel_limit_strict() {
        let limits = ResourceLimits { max_concurrent_kernels: 2, ..default_limits() };
        let mut limiter = ResourceLimiter::new(limits, strict_config());
        limiter.add_active_kernel();
        limiter.add_active_kernel();
        let result = limiter.check(&normal_request(0, 100));
        assert!(matches!(result, LimitResult::Denied { .. }));
    }

    #[test]
    fn concurrent_kernel_limit_nonstrict_wait() {
        let limits = ResourceLimits { max_concurrent_kernels: 1, ..default_limits() };
        let mut limiter = ResourceLimiter::new(limits, default_config());
        limiter.add_active_kernel();
        let result = limiter.check(&normal_request(0, 100));
        assert!(matches!(result, LimitResult::WaitRequired { .. }));
    }

    #[test]
    fn kernel_release_allows_new_request() {
        let limits = ResourceLimits { max_concurrent_kernels: 1, ..default_limits() };
        let mut limiter = ResourceLimiter::new(limits, strict_config());
        limiter.add_active_kernel();
        assert!(matches!(limiter.check(&normal_request(0, 100)), LimitResult::Denied { .. }));
        limiter.remove_active_kernel();
        assert_eq!(limiter.check(&normal_request(0, 100)), LimitResult::Allowed);
    }

    // ── active-request limit ─────────────────────────────────────────

    #[test]
    fn active_request_limit_strict() {
        let limits = ResourceLimits { max_active_requests: 1, ..default_limits() };
        let mut limiter = ResourceLimiter::new(limits, strict_config());
        limiter.reserve(&normal_request(0, 100), 0).expect("first reserve");
        let result = limiter.check(&normal_request(0, 100));
        assert!(matches!(result, LimitResult::Denied { .. }));
    }

    #[test]
    fn active_request_limit_nonstrict_wait() {
        let limits = ResourceLimits { max_active_requests: 1, ..default_limits() };
        let mut limiter = ResourceLimiter::new(limits, default_config());
        limiter.reserve(&normal_request(0, 100), 0).expect("first reserve");
        let result = limiter.check(&normal_request(0, 100));
        assert!(matches!(result, LimitResult::WaitRequired { .. }));
    }

    // ── overcommit ───────────────────────────────────────────────────

    #[test]
    fn overcommit_allowed_when_configured() {
        let limits = ResourceLimits { max_memory_bytes: 1000, ..default_limits() };
        let limiter = ResourceLimiter::new(limits, overcommit_config(1.5));
        let result = limiter.check(&normal_request(1200, 100));
        assert!(matches!(result, LimitResult::OvercommitAllowed { .. }));
    }

    #[test]
    fn overcommit_denied_when_disabled() {
        let limits = ResourceLimits { max_memory_bytes: 1000, ..default_limits() };
        let limiter = ResourceLimiter::new(limits, strict_config());
        let result = limiter.check(&normal_request(1200, 100));
        assert!(matches!(result, LimitResult::Denied { .. }));
    }

    #[test]
    fn overcommit_beyond_ratio_denied_strict() {
        let limits = ResourceLimits { max_memory_bytes: 1000, ..default_limits() };
        let cfg = LimiterConfig {
            enforce_strict: true,
            allow_overcommit: true,
            overcommit_ratio: 1.2,
            reservation_timeout_ms: None,
        };
        let limiter = ResourceLimiter::new(limits, cfg);
        // 1300 > 1000 * 1.2 = 1200 → denied
        let result = limiter.check(&normal_request(1300, 100));
        assert!(matches!(result, LimitResult::Denied { .. }));
    }

    #[test]
    fn overcommit_within_ratio_returns_usage_pct() {
        let limits = ResourceLimits { max_memory_bytes: 1000, ..default_limits() };
        let limiter = ResourceLimiter::new(limits, overcommit_config(2.0));
        if let LimitResult::OvercommitAllowed { current_usage_pct } =
            limiter.check(&normal_request(1500, 100))
        {
            assert!(current_usage_pct >= 0.0);
        } else {
            panic!("expected OvercommitAllowed");
        }
    }

    // ── reservations ─────────────────────────────────────────────────

    #[test]
    fn reservation_creation() {
        let mut limiter = ResourceLimiter::new(default_limits(), default_config());
        let r = limiter.reserve(&normal_request(1024, 100), 1000).expect("reserve");
        assert_eq!(r.id, 1);
        assert_eq!(r.memory_bytes, 1024);
        assert_eq!(r.created_at, 1000);
    }

    #[test]
    fn reservation_ids_increment() {
        let mut limiter = ResourceLimiter::new(default_limits(), default_config());
        let r1 = limiter.reserve(&normal_request(0, 0), 0).expect("r1");
        let r2 = limiter.reserve(&normal_request(0, 0), 0).expect("r2");
        assert_eq!(r2.id, r1.id + 1);
    }

    #[test]
    fn reservation_release() {
        let mut limiter = ResourceLimiter::new(default_limits(), default_config());
        let r = limiter.reserve(&normal_request(1024, 100), 0).expect("reserve");
        assert!(limiter.release(r.id).is_ok());
        assert_eq!(limiter.current_usage().memory_bytes, 0);
    }

    #[test]
    fn release_nonexistent_reservation_errors() {
        let mut limiter = ResourceLimiter::new(default_limits(), default_config());
        assert!(matches!(limiter.release(999), Err(ResourceError::ReservationNotFound(999))));
    }

    #[test]
    fn reservation_timeout_set_from_config() {
        let cfg = LimiterConfig { reservation_timeout_ms: Some(5000), ..LimiterConfig::default() };
        let mut limiter = ResourceLimiter::new(default_limits(), cfg);
        let r = limiter.reserve(&normal_request(0, 0), 1000).expect("reserve");
        assert_eq!(r.expires_at, Some(6000));
    }

    #[test]
    fn reservation_no_timeout_when_config_none() {
        let mut limiter = ResourceLimiter::new(default_limits(), default_config());
        let r = limiter.reserve(&normal_request(0, 0), 1000).expect("reserve");
        assert_eq!(r.expires_at, None);
    }

    #[test]
    fn expire_stale_reservations_removes_expired() {
        let cfg = LimiterConfig { reservation_timeout_ms: Some(100), ..LimiterConfig::default() };
        let mut limiter = ResourceLimiter::new(default_limits(), cfg);
        limiter.reserve(&normal_request(512, 0), 0).expect("reserve");
        let expired = limiter.expire_stale_reservations(200);
        assert_eq!(expired, 1);
        assert_eq!(limiter.current_usage().memory_bytes, 0);
    }

    #[test]
    fn expire_stale_keeps_non_expired() {
        let cfg = LimiterConfig { reservation_timeout_ms: Some(1000), ..LimiterConfig::default() };
        let mut limiter = ResourceLimiter::new(default_limits(), cfg);
        limiter.reserve(&normal_request(512, 0), 0).expect("reserve");
        let expired = limiter.expire_stale_reservations(500);
        assert_eq!(expired, 0);
        assert_eq!(limiter.current_usage().memory_bytes, 512);
    }

    #[test]
    fn expire_stale_mixed_expiry() {
        let cfg = LimiterConfig { reservation_timeout_ms: Some(100), ..LimiterConfig::default() };
        let mut limiter = ResourceLimiter::new(default_limits(), cfg);
        limiter.reserve(&normal_request(256, 0), 0).expect("r1");
        limiter.reserve(&normal_request(256, 0), 200).expect("r2");
        // At t=150 only the first should expire (expires_at=100).
        let expired = limiter.expire_stale_reservations(150);
        assert_eq!(expired, 1);
        assert_eq!(limiter.current_usage().memory_bytes, 256);
    }

    #[test]
    fn reservation_reduces_available_memory() {
        let limits = ResourceLimits { max_memory_bytes: 2048, ..default_limits() };
        let mut limiter = ResourceLimiter::new(limits, default_config());
        let before = limiter.available_memory();
        limiter.reserve(&normal_request(512, 0), 0).expect("reserve");
        assert!(limiter.available_memory() < before);
    }

    #[test]
    fn multiple_concurrent_reservations() {
        let mut limiter = ResourceLimiter::new(default_limits(), default_config());
        let ids: Vec<u64> = (0u64..5)
            .map(|i| limiter.reserve(&normal_request(100, 10), i).expect("reserve").id)
            .collect();
        assert_eq!(ids.len(), 5);
        assert_eq!(limiter.current_usage().active_requests, 5);
        assert_eq!(limiter.current_usage().memory_bytes, 500);
    }

    #[test]
    fn release_all_reservations_restores_memory() {
        let mut limiter = ResourceLimiter::new(default_limits(), default_config());
        let ids: Vec<u64> = (0..3)
            .map(|_| limiter.reserve(&normal_request(100, 0), 0).expect("reserve").id)
            .collect();
        for id in ids {
            limiter.release(id).expect("release");
        }
        assert_eq!(limiter.current_usage().memory_bytes, 0);
        assert_eq!(limiter.current_usage().active_requests, 0);
    }

    // ── peak tracking ────────────────────────────────────────────────

    #[test]
    fn peak_memory_tracking() {
        let mut limiter = ResourceLimiter::new(default_limits(), default_config());
        let r1 = limiter.reserve(&normal_request(500, 0), 0).expect("r1");
        let r2 = limiter.reserve(&normal_request(500, 0), 0).expect("r2");
        assert_eq!(limiter.current_usage().peak_memory_bytes, 1000);
        limiter.release(r1.id).expect("release r1");
        // Peak should still reflect the high-water mark.
        assert_eq!(limiter.current_usage().peak_memory_bytes, 1000);
        limiter.release(r2.id).expect("release r2");
        assert_eq!(limiter.current_usage().peak_memory_bytes, 1000);
    }

    #[test]
    fn peak_kernel_tracking() {
        let mut limiter = ResourceLimiter::new(default_limits(), default_config());
        limiter.add_active_kernel();
        limiter.add_active_kernel();
        limiter.add_active_kernel();
        assert_eq!(limiter.current_usage().peak_kernels, 3);
        limiter.remove_active_kernel();
        limiter.remove_active_kernel();
        assert_eq!(limiter.current_usage().peak_kernels, 3);
    }

    // ── utilisation ──────────────────────────────────────────────────

    #[test]
    fn utilization_percent_zero_when_idle() {
        let limiter = ResourceLimiter::new(default_limits(), default_config());
        assert!((limiter.utilization_percent() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn utilization_percent_after_reservation() {
        let limits = ResourceLimits { max_memory_bytes: 1000, ..default_limits() };
        let mut limiter = ResourceLimiter::new(limits, default_config());
        limiter.reserve(&normal_request(500, 0), 0).expect("reserve");
        let pct = limiter.utilization_percent();
        assert!((pct - 50.0).abs() < 0.01);
    }

    #[test]
    fn utilization_percent_zero_limit() {
        let limits = ResourceLimits { max_memory_bytes: 0, ..default_limits() };
        let limiter = ResourceLimiter::new(limits, default_config());
        assert!((limiter.utilization_percent() - 0.0).abs() < f64::EPSILON);
    }

    // ── available memory ─────────────────────────────────────────────

    #[test]
    fn available_memory_full_when_idle() {
        let limiter = ResourceLimiter::new(default_limits(), default_config());
        assert_eq!(limiter.available_memory(), default_limits().max_memory_bytes);
    }

    #[test]
    fn available_memory_decreases_with_usage() {
        let limits = ResourceLimits { max_memory_bytes: 2048, ..default_limits() };
        let mut limiter = ResourceLimiter::new(limits, default_config());
        limiter.reserve(&normal_request(512, 0), 0).expect("reserve");
        assert_eq!(limiter.available_memory(), 2048 - 512);
    }

    #[test]
    fn available_memory_saturates_at_zero() {
        let limits = ResourceLimits { max_memory_bytes: 100, ..default_limits() };
        let mut limiter = ResourceLimiter::new(limits, overcommit_config(5.0));
        limiter.reserve(&normal_request(100, 0), 0).expect("reserve");
        assert_eq!(limiter.available_memory(), 0);
    }

    // ── priority ─────────────────────────────────────────────────────

    #[test]
    fn system_priority_bypasses_memory_limit() {
        let limits = ResourceLimits { max_memory_bytes: 100, ..default_limits() };
        let limiter = ResourceLimiter::new(limits, strict_config());
        let result = limiter.check(&system_request(9999, 100));
        assert_eq!(result, LimitResult::Allowed);
    }

    #[test]
    fn system_priority_bypasses_kernel_limit() {
        let limits = ResourceLimits { max_concurrent_kernels: 0, ..default_limits() };
        let mut limiter = ResourceLimiter::new(limits, strict_config());
        limiter.add_active_kernel();
        let result = limiter.check(&system_request(0, 100));
        assert_eq!(result, LimitResult::Allowed);
    }

    #[test]
    fn high_priority_still_checked() {
        let limits = ResourceLimits { max_memory_bytes: 100, ..default_limits() };
        let limiter = ResourceLimiter::new(limits, strict_config());
        let result = limiter.check(&high_request(200, 100));
        assert!(matches!(result, LimitResult::Denied { .. }));
    }

    #[test]
    fn low_priority_still_checked() {
        let limits = ResourceLimits { max_memory_bytes: 100, ..default_limits() };
        let limiter = ResourceLimiter::new(limits, strict_config());
        let result = limiter.check(&low_request(200, 100));
        assert!(matches!(result, LimitResult::Denied { .. }));
    }

    #[test]
    fn priority_ordering() {
        assert!(RequestPriority::Low < RequestPriority::Normal);
        assert!(RequestPriority::Normal < RequestPriority::High);
        assert!(RequestPriority::High < RequestPriority::System);
    }

    // ── enforce ──────────────────────────────────────────────────────

    #[test]
    fn enforce_mirrors_check_allowed() {
        let limiter = ResourceLimiter::new(default_limits(), default_config());
        let req = normal_request(100, 100);
        assert_eq!(limiter.enforce(&req), limiter.check(&req));
    }

    #[test]
    fn enforce_mirrors_check_denied() {
        let limits = ResourceLimits { max_memory_bytes: 50, ..default_limits() };
        let limiter = ResourceLimiter::new(limits, strict_config());
        let req = normal_request(100, 100);
        let e = limiter.enforce(&req);
        let c = limiter.check(&req);
        assert!(matches!(e, LimitResult::Denied { .. }));
        assert!(matches!(c, LimitResult::Denied { .. }));
    }

    // ── reset ────────────────────────────────────────────────────────

    #[test]
    fn reset_clears_usage() {
        let mut limiter = ResourceLimiter::new(default_limits(), default_config());
        limiter.reserve(&normal_request(1024, 100), 0).expect("reserve");
        limiter.add_active_kernel();
        limiter.reset_usage();
        let usage = limiter.current_usage();
        assert_eq!(usage.memory_bytes, 0);
        assert_eq!(usage.active_kernels, 0);
        assert_eq!(usage.active_requests, 0);
        assert_eq!(usage.peak_memory_bytes, 0);
    }

    #[test]
    fn reset_clears_reservations() {
        let mut limiter = ResourceLimiter::new(default_limits(), default_config());
        let r = limiter.reserve(&normal_request(100, 0), 0).expect("reserve");
        limiter.reset_usage();
        assert!(matches!(limiter.release(r.id), Err(ResourceError::ReservationNotFound(_))));
    }

    #[test]
    fn reset_reservation_ids_restart() {
        let mut limiter = ResourceLimiter::new(default_limits(), default_config());
        limiter.reserve(&normal_request(0, 0), 0).expect("first");
        limiter.reset_usage();
        let r = limiter.reserve(&normal_request(0, 0), 0).expect("after reset");
        assert_eq!(r.id, 1);
    }

    // ── empty reservations ───────────────────────────────────────────

    #[test]
    fn empty_reservations_on_new_limiter() {
        let limiter = ResourceLimiter::new(default_limits(), default_config());
        assert_eq!(limiter.current_usage().active_requests, 0);
        assert_eq!(limiter.available_memory(), default_limits().max_memory_bytes);
    }

    #[test]
    fn expire_on_empty_reservations_returns_zero() {
        let mut limiter = ResourceLimiter::new(default_limits(), default_config());
        assert_eq!(limiter.expire_stale_reservations(9999), 0);
    }

    // ── wait estimation ──────────────────────────────────────────────

    #[test]
    fn wait_estimation_proportional_to_excess() {
        let limits = ResourceLimits { max_memory_bytes: 1000, ..default_limits() };
        let limiter = ResourceLimiter::new(limits, default_config());
        if let LimitResult::WaitRequired { estimated_wait_ms } =
            limiter.check(&normal_request(1500, 100))
        {
            assert_eq!(estimated_wait_ms, 500);
        } else {
            panic!("expected WaitRequired");
        }
    }

    #[test]
    fn wait_estimation_capped_at_10s() {
        let limits = ResourceLimits { max_memory_bytes: 100, ..default_limits() };
        let limiter = ResourceLimiter::new(limits, default_config());
        if let LimitResult::WaitRequired { estimated_wait_ms } =
            limiter.check(&normal_request(1_000_000, 100))
        {
            assert_eq!(estimated_wait_ms, 10_000);
        } else {
            panic!("expected WaitRequired");
        }
    }

    // ── resource error display ───────────────────────────────────────

    #[test]
    fn resource_error_display_limit_exceeded() {
        let e = ResourceError::LimitExceeded("oom".into());
        assert_eq!(e.to_string(), "limit exceeded: oom");
    }

    #[test]
    fn resource_error_display_not_found() {
        let e = ResourceError::ReservationNotFound(42);
        assert_eq!(e.to_string(), "reservation 42 not found");
    }

    // ── batch size and sequence length fields ────────────────────────

    #[test]
    fn limits_store_batch_size() {
        let limits = ResourceLimits { max_batch_size: 64, ..default_limits() };
        assert_eq!(limits.max_batch_size, 64);
    }

    #[test]
    fn limits_store_sequence_length() {
        let limits = ResourceLimits { max_sequence_length: 4096, ..default_limits() };
        assert_eq!(limits.max_sequence_length, 4096);
    }

    // ── kernel add / remove boundary ─────────────────────────────────

    #[test]
    fn remove_kernel_saturates_at_zero() {
        let mut limiter = ResourceLimiter::new(default_limits(), default_config());
        limiter.remove_active_kernel();
        assert_eq!(limiter.current_usage().active_kernels, 0);
    }

    #[test]
    fn add_then_remove_kernel_round_trip() {
        let mut limiter = ResourceLimiter::new(default_limits(), default_config());
        limiter.add_active_kernel();
        limiter.add_active_kernel();
        limiter.remove_active_kernel();
        assert_eq!(limiter.current_usage().active_kernels, 1);
    }

    // ── config defaults ──────────────────────────────────────────────

    #[test]
    fn default_config_not_strict() {
        let cfg = LimiterConfig::default();
        assert!(!cfg.enforce_strict);
        assert!(!cfg.allow_overcommit);
        assert!((cfg.overcommit_ratio - 1.0).abs() < f64::EPSILON);
        assert_eq!(cfg.reservation_timeout_ms, None);
    }

    // ── reservation denied propagates ────────────────────────────────

    #[test]
    fn reserve_denied_returns_error() {
        let limits = ResourceLimits { max_memory_bytes: 100, ..default_limits() };
        let mut limiter = ResourceLimiter::new(limits, strict_config());
        let result = limiter.reserve(&normal_request(500, 0), 0);
        assert!(matches!(result, Err(ResourceError::LimitExceeded(_))));
    }

    #[test]
    fn reserve_wait_required_returns_error() {
        let limits = ResourceLimits { max_memory_bytes: 100, ..default_limits() };
        let mut limiter = ResourceLimiter::new(limits, default_config());
        let result = limiter.reserve(&normal_request(500, 0), 0);
        assert!(matches!(result, Err(ResourceError::LimitExceeded(_))));
    }
}
