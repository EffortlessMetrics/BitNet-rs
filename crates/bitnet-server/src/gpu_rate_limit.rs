//! GPU-aware rate limiting with per-device token buckets.
//!
//! [`GpuRateLimiter`] enforces configurable concurrency and throughput
//! limits per GPU device, with dynamic rate adjustment based on memory
//! pressure. When a device is under heavy load the limiter rejects new
//! requests and suggests a `Retry-After` interval.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Per-device rate limiting configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Maximum concurrent requests allowed on the device.
    pub max_concurrent: u32,
    /// Maximum tokens generated per second (0 = unlimited).
    pub max_tokens_per_sec: f64,
    /// Memory utilisation fraction (0.0–1.0) above which rates are
    /// dynamically reduced.
    pub memory_threshold: f64,
    /// How many tokens the bucket holds at most.
    pub bucket_capacity: f64,
    /// Seconds to suggest in `Retry-After` when rejecting.
    pub default_retry_after: Duration,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 8,
            max_tokens_per_sec: 100.0,
            memory_threshold: 0.85,
            bucket_capacity: 200.0,
            default_retry_after: Duration::from_secs(2),
        }
    }
}

// ---------------------------------------------------------------------------
// Decision
// ---------------------------------------------------------------------------

/// Outcome of a rate-limit check.
#[derive(Debug, Clone, PartialEq)]
pub enum RateLimitDecision {
    /// Request is allowed.
    Allowed,
    /// Request is rejected; the caller should wait before retrying.
    Rejected {
        /// Suggested wait duration for the `Retry-After` header.
        retry_after: Duration,
        reason: String,
    },
}

// ---------------------------------------------------------------------------
// Per-device bucket state
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct DeviceBucket {
    config: RateLimitConfig,
    /// Tokens currently in the bucket.
    tokens: f64,
    /// Last time tokens were replenished.
    last_refill: Instant,
    /// Current concurrent requests.
    active: u32,
    /// Current memory utilisation (0.0–1.0).
    memory_utilisation: f64,
}

impl DeviceBucket {
    fn new(config: RateLimitConfig) -> Self {
        Self {
            tokens: config.bucket_capacity,
            last_refill: Instant::now(),
            active: 0,
            memory_utilisation: 0.0,
            config,
        }
    }

    /// Refill tokens based on elapsed time and effective rate.
    fn refill(&mut self, now: Instant) {
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        if elapsed <= 0.0 {
            return;
        }
        let rate = self.effective_rate();
        self.tokens = (self.tokens + rate * elapsed)
            .min(self.config.bucket_capacity);
        self.last_refill = now;
    }

    /// Token generation rate, reduced when memory is above threshold.
    fn effective_rate(&self) -> f64 {
        if self.memory_utilisation > self.config.memory_threshold {
            // Linear reduction: at threshold → full rate,
            // at 1.0 → 10% of rate.
            let excess = (self.memory_utilisation
                - self.config.memory_threshold)
                / (1.0 - self.config.memory_threshold);
            let factor = 1.0 - 0.9 * excess.clamp(0.0, 1.0);
            self.config.max_tokens_per_sec * factor
        } else {
            self.config.max_tokens_per_sec
        }
    }

    /// Try to acquire one request token.
    fn try_acquire(&mut self, now: Instant) -> RateLimitDecision {
        // Concurrency check first
        if self.active >= self.config.max_concurrent {
            return RateLimitDecision::Rejected {
                retry_after: self.config.default_retry_after,
                reason: format!(
                    "max concurrent requests ({}) reached",
                    self.config.max_concurrent
                ),
            };
        }

        // Token bucket check
        self.refill(now);
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            self.active += 1;
            RateLimitDecision::Allowed
        } else {
            // Compute how long until 1 token is available
            let rate = self.effective_rate();
            let wait = if rate > 0.0 {
                Duration::from_secs_f64((1.0 - self.tokens) / rate)
            } else {
                self.config.default_retry_after
            };
            RateLimitDecision::Rejected {
                retry_after: wait,
                reason: "token bucket exhausted".into(),
            }
        }
    }

    fn release(&mut self) {
        self.active = self.active.saturating_sub(1);
    }
}

// ---------------------------------------------------------------------------
// GpuRateLimiter
// ---------------------------------------------------------------------------

/// Thread-safe, per-device GPU rate limiter.
#[derive(Clone)]
pub struct GpuRateLimiter {
    inner: Arc<RwLock<HashMap<String, DeviceBucket>>>,
}

impl GpuRateLimiter {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a device with the given rate limit config.
    pub fn register_device(
        &self,
        device_id: &str,
        config: RateLimitConfig,
    ) {
        let mut inner = self.inner.write().unwrap();
        inner.insert(
            device_id.to_string(),
            DeviceBucket::new(config),
        );
    }

    /// Remove a device from the limiter.
    pub fn remove_device(&self, device_id: &str) -> bool {
        let mut inner = self.inner.write().unwrap();
        inner.remove(device_id).is_some()
    }

    /// Update the reported memory utilisation for a device (0.0–1.0).
    pub fn update_memory_utilisation(
        &self,
        device_id: &str,
        utilisation: f64,
    ) {
        let mut inner = self.inner.write().unwrap();
        if let Some(bucket) = inner.get_mut(device_id) {
            bucket.memory_utilisation = utilisation.clamp(0.0, 1.0);
        }
    }

    /// Check whether a request to `device_id` is allowed.
    pub fn check(
        &self,
        device_id: &str,
    ) -> RateLimitDecision {
        self.check_at(device_id, Instant::now())
    }

    /// Testable variant that accepts an explicit timestamp.
    pub fn check_at(
        &self,
        device_id: &str,
        now: Instant,
    ) -> RateLimitDecision {
        let mut inner = self.inner.write().unwrap();
        match inner.get_mut(device_id) {
            Some(bucket) => bucket.try_acquire(now),
            // Unknown device → allow (no limit configured)
            None => RateLimitDecision::Allowed,
        }
    }

    /// Release a request slot after completion.
    pub fn release(&self, device_id: &str) {
        let mut inner = self.inner.write().unwrap();
        if let Some(bucket) = inner.get_mut(device_id) {
            bucket.release();
        }
    }

    /// Current number of active requests on a device.
    pub fn active_count(&self, device_id: &str) -> u32 {
        let inner = self.inner.read().unwrap();
        inner.get(device_id).map(|b| b.active).unwrap_or(0)
    }

    /// Number of registered devices.
    pub fn device_count(&self) -> usize {
        self.inner.read().unwrap().len()
    }
}

impl Default for GpuRateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(max_conc: u32, tps: f64) -> RateLimitConfig {
        RateLimitConfig {
            max_concurrent: max_conc,
            max_tokens_per_sec: tps,
            memory_threshold: 0.85,
            bucket_capacity: 10.0,
            default_retry_after: Duration::from_secs(1),
        }
    }

    #[test]
    fn allow_under_limit() {
        let rl = GpuRateLimiter::new();
        rl.register_device("g0", cfg(4, 100.0));
        assert_eq!(rl.check("g0"), RateLimitDecision::Allowed);
    }

    #[test]
    fn reject_at_max_concurrent() {
        let rl = GpuRateLimiter::new();
        rl.register_device("g0", cfg(2, 100.0));
        assert_eq!(rl.check("g0"), RateLimitDecision::Allowed);
        assert_eq!(rl.check("g0"), RateLimitDecision::Allowed);
        let d = rl.check("g0");
        assert!(
            matches!(d, RateLimitDecision::Rejected { .. }),
            "3rd request should be rejected"
        );
    }

    #[test]
    fn release_frees_slot() {
        let rl = GpuRateLimiter::new();
        rl.register_device("g0", cfg(1, 100.0));
        assert_eq!(rl.check("g0"), RateLimitDecision::Allowed);
        assert!(matches!(
            rl.check("g0"),
            RateLimitDecision::Rejected { .. }
        ));
        rl.release("g0");
        assert_eq!(rl.check("g0"), RateLimitDecision::Allowed);
    }

    #[test]
    fn token_bucket_exhaustion() {
        let rl = GpuRateLimiter::new();
        // bucket_capacity = 10, max_concurrent = 100
        rl.register_device("g0", cfg(100, 1.0));
        let now = Instant::now();
        // Exhaust all 10 tokens
        for _ in 0..10 {
            assert_eq!(
                rl.check_at("g0", now),
                RateLimitDecision::Allowed,
            );
            rl.release("g0");
        }
        // 11th should be rejected
        let d = rl.check_at("g0", now);
        assert!(
            matches!(d, RateLimitDecision::Rejected { .. }),
            "bucket should be exhausted"
        );
    }

    #[test]
    fn tokens_refill_over_time() {
        let rl = GpuRateLimiter::new();
        // rate = 10 tok/s, bucket = 10
        rl.register_device("g0", cfg(100, 10.0));
        let t0 = Instant::now();
        // Drain bucket
        for _ in 0..10 {
            assert_eq!(
                rl.check_at("g0", t0),
                RateLimitDecision::Allowed,
            );
            rl.release("g0");
        }
        // After 1 second at 10 tok/s, 10 tokens should refill
        let t1 = t0 + Duration::from_secs(1);
        assert_eq!(
            rl.check_at("g0", t1),
            RateLimitDecision::Allowed,
        );
    }

    #[test]
    fn memory_pressure_reduces_rate() {
        let rl = GpuRateLimiter::new();
        // rate = 100, threshold = 0.85
        rl.register_device("g0", cfg(100, 100.0));
        let t0 = Instant::now();
        // Drain bucket
        for _ in 0..10 {
            rl.check_at("g0", t0);
            rl.release("g0");
        }
        // Set memory to 100% (max pressure)
        rl.update_memory_utilisation("g0", 1.0);
        // After 0.15s at full rate we'd get 15 tokens back,
        // but under pressure (rate * 0.1 factor) only ~1.5 tokens.
        let t1 = t0 + Duration::from_millis(150);
        let d = rl.check_at("g0", t1);
        assert_eq!(d, RateLimitDecision::Allowed);
        // Should have very few tokens left (≈0.5)
        let d2 = rl.check_at("g0", t1);
        assert!(
            matches!(d2, RateLimitDecision::Rejected { .. }),
            "memory pressure should reduce refill rate"
        );
    }

    #[test]
    fn unknown_device_allowed() {
        let rl = GpuRateLimiter::new();
        assert_eq!(rl.check("unknown"), RateLimitDecision::Allowed);
    }

    #[test]
    fn retry_after_in_rejection() {
        let rl = GpuRateLimiter::new();
        rl.register_device("g0", cfg(1, 100.0));
        rl.check("g0"); // consume the slot
        match rl.check("g0") {
            RateLimitDecision::Rejected { retry_after, .. } => {
                assert!(
                    retry_after.as_millis() > 0,
                    "retry_after should be non-zero"
                );
            }
            _ => panic!("expected Rejected"),
        }
    }

    #[test]
    fn register_and_remove() {
        let rl = GpuRateLimiter::new();
        rl.register_device("g0", RateLimitConfig::default());
        assert_eq!(rl.device_count(), 1);
        assert!(rl.remove_device("g0"));
        assert_eq!(rl.device_count(), 0);
    }

    #[test]
    fn active_count_tracking() {
        let rl = GpuRateLimiter::new();
        rl.register_device("g0", cfg(4, 100.0));
        assert_eq!(rl.active_count("g0"), 0);
        rl.check("g0");
        rl.check("g0");
        assert_eq!(rl.active_count("g0"), 2);
        rl.release("g0");
        assert_eq!(rl.active_count("g0"), 1);
    }
}
