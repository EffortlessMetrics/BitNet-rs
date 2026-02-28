//! Rate limiting for inference requests.
//!
//! Provides multiple algorithms (token bucket, sliding window, fixed window,
//! leaky bucket) with per-user quotas, burst handling, and metrics collection.

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Rate limiter configuration.
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum sustained requests per second.
    pub requests_per_second: f64,
    /// Maximum tokens (LLM output) per minute.
    pub tokens_per_minute: u64,
    /// Burst capacity beyond sustained rate.
    pub burst_size: u32,
    /// Per-user limit overrides keyed by user ID.
    pub per_user_limits: HashMap<String, UserLimitOverride>,
    /// Algorithm to use.
    pub algorithm: RateLimitAlgorithm,
}

/// Per-user override of rate limits.
#[derive(Debug, Clone)]
pub struct UserLimitOverride {
    /// Override for requests per second.
    pub requests_per_second: Option<f64>,
    /// Override for tokens per minute.
    pub tokens_per_minute: Option<u64>,
    /// Override for burst size.
    pub burst_size: Option<u32>,
}

impl RateLimitConfig {
    /// Create a new config with the given sustained rate and burst size.
    pub fn new(requests_per_second: f64, burst_size: u32) -> Self {
        Self {
            requests_per_second,
            tokens_per_minute: 60_000,
            burst_size,
            per_user_limits: HashMap::new(),
            algorithm: RateLimitAlgorithm::TokenBucket,
        }
    }

    /// Validate that the configuration is well-formed.
    pub fn validate(&self) -> Result<(), RateLimitError> {
        if self.requests_per_second < 0.0 {
            return Err(RateLimitError::InvalidConfig(
                "requests_per_second must be non-negative".into(),
            ));
        }
        if self.requests_per_second.is_nan() {
            return Err(RateLimitError::InvalidConfig(
                "requests_per_second must not be NaN".into(),
            ));
        }
        if self.burst_size == 0 && self.requests_per_second > 0.0 {
            return Err(RateLimitError::InvalidConfig(
                "burst_size must be > 0 when rate > 0".into(),
            ));
        }
        Ok(())
    }

    /// Effective requests per second for a user, considering overrides.
    pub fn effective_rps(&self, user_id: Option<&str>) -> f64 {
        user_id
            .and_then(|uid| self.per_user_limits.get(uid))
            .and_then(|o| o.requests_per_second)
            .unwrap_or(self.requests_per_second)
    }

    /// Effective burst size for a user, considering overrides.
    pub fn effective_burst(&self, user_id: Option<&str>) -> u32 {
        user_id
            .and_then(|uid| self.per_user_limits.get(uid))
            .and_then(|o| o.burst_size)
            .unwrap_or(self.burst_size)
    }
}

// ---------------------------------------------------------------------------
// Algorithm selection
// ---------------------------------------------------------------------------

/// Algorithm used for rate limiting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateLimitAlgorithm {
    /// Token bucket algorithm.
    TokenBucket,
    /// Sliding window counter.
    SlidingWindow,
    /// Fixed (tumbling) window counter.
    FixedWindow,
    /// Leaky bucket algorithm.
    LeakyBucket,
}

// ---------------------------------------------------------------------------
// Rate limit key
// ---------------------------------------------------------------------------

/// Key used to identify a rate-limited entity.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum RateLimitKey {
    /// Limit by user ID.
    User(String),
    /// Limit by API key.
    ApiKey(String),
    /// Limit by IP address.
    Ip(String),
    /// Global (singleton) limiter.
    Global,
}

// ---------------------------------------------------------------------------
// Rate limit result
// ---------------------------------------------------------------------------

/// Result of a rate-limit check.
#[derive(Debug, Clone, PartialEq)]
pub struct RateLimitResult {
    /// Whether the request is allowed.
    pub allowed: bool,
    /// Milliseconds until the client should retry (0 if allowed).
    pub retry_after_ms: u64,
    /// Remaining tokens in the current window/bucket.
    pub remaining_tokens: f64,
    /// The maximum limit for reference.
    pub limit: f64,
}

impl RateLimitResult {
    /// Convenience constructor for an allowed result.
    pub const fn allow(remaining: f64, limit: f64) -> Self {
        Self {
            allowed: true,
            retry_after_ms: 0,
            remaining_tokens: remaining,
            limit,
        }
    }

    /// Convenience constructor for a denied result.
    pub const fn deny(retry_after_ms: u64, remaining: f64, limit: f64) -> Self {
        Self {
            allowed: false,
            retry_after_ms,
            remaining_tokens: remaining,
            limit,
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the rate limiter subsystem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RateLimitError {
    /// Configuration is invalid.
    InvalidConfig(String),
    /// Quota exhausted.
    QuotaExhausted {
        /// Remaining daily tokens.
        daily_remaining: u64,
        /// Remaining monthly tokens.
        monthly_remaining: u64,
    },
}

impl std::fmt::Display for RateLimitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid rate-limit config: {msg}"),
            Self::QuotaExhausted {
                daily_remaining,
                monthly_remaining,
            } => {
                write!(
                    f,
                    "quota exhausted (daily_remaining={daily_remaining}, \
                     monthly_remaining={monthly_remaining})"
                )
            }
        }
    }
}

impl std::error::Error for RateLimitError {}

// ---------------------------------------------------------------------------
// Token bucket limiter
// ---------------------------------------------------------------------------

/// Classic token-bucket rate limiter.
///
/// Tokens are added at a fixed rate up to `capacity`. Each request consumes
/// one token. Bursts up to `capacity` are allowed.
#[derive(Debug)]
pub struct TokenBucketLimiter {
    capacity: f64,
    rate: f64,
    tokens: f64,
    last_refill: Instant,
}

impl TokenBucketLimiter {
    /// Create a new bucket starting at full capacity.
    pub fn new(capacity: f64, rate: f64) -> Self {
        Self {
            capacity,
            rate,
            tokens: capacity,
            last_refill: Instant::now(),
        }
    }

    /// Create with a specific starting instant (for deterministic testing).
    pub const fn with_instant(capacity: f64, rate: f64, now: Instant) -> Self {
        Self {
            capacity,
            rate,
            tokens: capacity,
            last_refill: now,
        }
    }

    /// Refill tokens based on elapsed time.
    pub fn refill(&mut self, now: Instant) {
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        let added = elapsed * self.rate;
        self.tokens = (self.tokens + added).min(self.capacity);
        self.last_refill = now;
    }

    /// Try to consume one token. Returns the check result.
    pub fn try_acquire(&mut self, now: Instant) -> RateLimitResult {
        self.refill(now);
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            RateLimitResult::allow(self.tokens, self.capacity)
        } else {
            let deficit = 1.0 - self.tokens;
            let wait_secs = if self.rate > 0.0 {
                deficit / self.rate
            } else {
                f64::INFINITY
            };
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss
            )]
            let retry_ms = (wait_secs * 1000.0).ceil() as u64;
            RateLimitResult::deny(retry_ms, self.tokens, self.capacity)
        }
    }

    /// Current token count (after refill at `now`).
    pub fn available(&mut self, now: Instant) -> f64 {
        self.refill(now);
        self.tokens
    }

    /// Current token count without refill (snapshot).
    pub const fn tokens_snapshot(&self) -> f64 {
        self.tokens
    }

    /// Capacity of this bucket.
    pub const fn capacity(&self) -> f64 {
        self.capacity
    }

    /// Rate of token replenishment per second.
    pub const fn rate(&self) -> f64 {
        self.rate
    }
}

// ---------------------------------------------------------------------------
// Sliding window limiter
// ---------------------------------------------------------------------------

/// Sliding-window rate limiter.
///
/// Tracks request timestamps within a rolling window and denies requests
/// once the window count reaches the limit.
#[derive(Debug)]
pub struct SlidingWindowLimiter {
    window: Duration,
    max_requests: u64,
    timestamps: Vec<Instant>,
}

impl SlidingWindowLimiter {
    /// Create a new sliding window limiter.
    pub const fn new(window: Duration, max_requests: u64) -> Self {
        Self {
            window,
            max_requests,
            timestamps: Vec::new(),
        }
    }

    /// Prune timestamps outside the window.
    fn prune(&mut self, now: Instant) {
        self.timestamps
            .retain(|&t| now.duration_since(t) < self.window);
    }

    /// Try to record a request. Returns the check result.
    pub fn try_acquire(&mut self, now: Instant) -> RateLimitResult {
        self.prune(now);
        let count = self.timestamps.len() as u64;
        #[allow(clippy::cast_precision_loss)]
        let limit = self.max_requests as f64;
        if count < self.max_requests {
            self.timestamps.push(now);
            #[allow(clippy::cast_precision_loss)]
            let remaining = (self.max_requests - count - 1) as f64;
            RateLimitResult::allow(remaining, limit)
        } else {
            // Earliest timestamp expiry determines retry delay.
            let earliest = self.timestamps[0];
            let expires_at = earliest + self.window;
            let wait = expires_at
                .checked_duration_since(now)
                .unwrap_or(Duration::ZERO);
            #[allow(clippy::cast_possible_truncation)]
            let retry_ms = wait.as_millis() as u64 + 1;
            RateLimitResult::deny(retry_ms, 0.0, limit)
        }
    }

    /// Current count of requests in the window.
    pub fn current_count(&mut self, now: Instant) -> u64 {
        self.prune(now);
        self.timestamps.len() as u64
    }

    /// Maximum requests allowed in the window.
    pub const fn max_requests(&self) -> u64 {
        self.max_requests
    }
}

// ---------------------------------------------------------------------------
// Fixed window limiter
// ---------------------------------------------------------------------------

/// Fixed-window rate limiter.
///
/// Counts requests within discrete time windows and resets when a new window
/// starts.
#[derive(Debug)]
pub struct FixedWindowLimiter {
    window: Duration,
    max_requests: u64,
    window_start: Instant,
    count: u64,
}

impl FixedWindowLimiter {
    /// Create a new fixed window limiter.
    pub fn new(window: Duration, max_requests: u64) -> Self {
        Self {
            window,
            max_requests,
            window_start: Instant::now(),
            count: 0,
        }
    }

    /// Create with a specific starting instant (for deterministic testing).
    pub const fn with_instant(window: Duration, max_requests: u64, now: Instant) -> Self {
        Self {
            window,
            max_requests,
            window_start: now,
            count: 0,
        }
    }

    /// Try to record a request.
    pub fn try_acquire(&mut self, now: Instant) -> RateLimitResult {
        if now.duration_since(self.window_start) >= self.window {
            self.window_start = now;
            self.count = 0;
        }
        #[allow(clippy::cast_precision_loss)]
        let limit = self.max_requests as f64;
        if self.count < self.max_requests {
            self.count += 1;
            #[allow(clippy::cast_precision_loss)]
            let remaining = (self.max_requests - self.count) as f64;
            RateLimitResult::allow(remaining, limit)
        } else {
            let window_end = self.window_start + self.window;
            let wait = window_end
                .checked_duration_since(now)
                .unwrap_or(Duration::ZERO);
            #[allow(clippy::cast_possible_truncation)]
            let retry_ms = wait.as_millis() as u64 + 1;
            RateLimitResult::deny(retry_ms, 0.0, limit)
        }
    }

    /// Current request count in the active window.
    pub const fn current_count(&self) -> u64 {
        self.count
    }
}

// ---------------------------------------------------------------------------
// Leaky bucket limiter
// ---------------------------------------------------------------------------

/// Leaky-bucket rate limiter.
///
/// Requests fill a bucket that drains at a constant rate. When the bucket
/// is full, requests are denied.
#[derive(Debug)]
pub struct LeakyBucketLimiter {
    capacity: f64,
    drain_rate: f64,
    level: f64,
    last_drain: Instant,
}

impl LeakyBucketLimiter {
    /// Create a new leaky bucket limiter.
    pub fn new(capacity: f64, drain_rate: f64) -> Self {
        Self {
            capacity,
            drain_rate,
            level: 0.0,
            last_drain: Instant::now(),
        }
    }

    /// Create with a specific starting instant (for deterministic testing).
    pub const fn with_instant(capacity: f64, drain_rate: f64, now: Instant) -> Self {
        Self {
            capacity,
            drain_rate,
            level: 0.0,
            last_drain: now,
        }
    }

    /// Drain the bucket based on elapsed time.
    fn drain(&mut self, now: Instant) {
        let elapsed = now.duration_since(self.last_drain).as_secs_f64();
        let drained = elapsed * self.drain_rate;
        self.level = (self.level - drained).max(0.0);
        self.last_drain = now;
    }

    /// Try to add a request to the bucket.
    pub fn try_acquire(&mut self, now: Instant) -> RateLimitResult {
        self.drain(now);
        if self.level + 1.0 <= self.capacity {
            self.level += 1.0;
            let remaining = self.capacity - self.level;
            RateLimitResult::allow(remaining, self.capacity)
        } else {
            let excess = (self.level + 1.0) - self.capacity;
            let wait_secs = if self.drain_rate > 0.0 {
                excess / self.drain_rate
            } else {
                f64::INFINITY
            };
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss
            )]
            let retry_ms = (wait_secs * 1000.0).ceil() as u64;
            let remaining = (self.capacity - self.level).max(0.0);
            RateLimitResult::deny(retry_ms, remaining, self.capacity)
        }
    }

    /// Current level (after drain).
    pub fn current_level(&mut self, now: Instant) -> f64 {
        self.drain(now);
        self.level
    }

    /// Maximum capacity.
    pub const fn capacity(&self) -> f64 {
        self.capacity
    }

    /// Drain rate (requests per second).
    pub const fn drain_rate(&self) -> f64 {
        self.drain_rate
    }
}

// ---------------------------------------------------------------------------
// Quota manager
// ---------------------------------------------------------------------------

/// Per-user token quota tracking.
#[derive(Debug, Clone)]
pub struct UserQuota {
    /// Maximum tokens per day.
    pub daily_limit: u64,
    /// Maximum tokens per month.
    pub monthly_limit: u64,
    /// Tokens consumed today.
    pub daily_used: u64,
    /// Tokens consumed this month.
    pub monthly_used: u64,
}

impl UserQuota {
    /// Create a new quota with zero usage.
    pub const fn new(daily_limit: u64, monthly_limit: u64) -> Self {
        Self {
            daily_limit,
            monthly_limit,
            daily_used: 0,
            monthly_used: 0,
        }
    }

    /// Remaining daily tokens.
    pub const fn daily_remaining(&self) -> u64 {
        self.daily_limit.saturating_sub(self.daily_used)
    }

    /// Remaining monthly tokens.
    pub const fn monthly_remaining(&self) -> u64 {
        self.monthly_limit.saturating_sub(self.monthly_used)
    }

    /// Try to consume `count` tokens. Returns `Ok` if within limits.
    pub const fn try_consume(&mut self, count: u64) -> Result<(), RateLimitError> {
        if self.daily_used + count > self.daily_limit
            || self.monthly_used + count > self.monthly_limit
        {
            return Err(RateLimitError::QuotaExhausted {
                daily_remaining: self.daily_remaining(),
                monthly_remaining: self.monthly_remaining(),
            });
        }
        self.daily_used += count;
        self.monthly_used += count;
        Ok(())
    }

    /// Reset daily counter.
    pub const fn reset_daily(&mut self) {
        self.daily_used = 0;
    }

    /// Reset monthly counter (also resets daily).
    pub const fn reset_monthly(&mut self) {
        self.monthly_used = 0;
        self.daily_used = 0;
    }
}

/// Manages per-user quotas.
#[derive(Debug)]
pub struct QuotaManager {
    quotas: HashMap<String, UserQuota>,
    default_daily: u64,
    default_monthly: u64,
}

impl QuotaManager {
    /// Create a new quota manager with default limits.
    pub fn new(default_daily: u64, default_monthly: u64) -> Self {
        Self {
            quotas: HashMap::new(),
            default_daily,
            default_monthly,
        }
    }

    /// Get or create quota for a user.
    pub fn get_quota(&mut self, user_id: &str) -> &mut UserQuota {
        let daily = self.default_daily;
        let monthly = self.default_monthly;
        self.quotas
            .entry(user_id.to_string())
            .or_insert_with(|| UserQuota::new(daily, monthly))
    }

    /// Set a custom quota for a user.
    pub fn set_quota(&mut self, user_id: &str, quota: UserQuota) {
        self.quotas.insert(user_id.to_string(), quota);
    }

    /// Try to consume tokens for a user.
    pub fn try_consume(&mut self, user_id: &str, count: u64) -> Result<(), RateLimitError> {
        self.get_quota(user_id).try_consume(count)
    }

    /// Reset all daily counters.
    pub fn reset_all_daily(&mut self) {
        for quota in self.quotas.values_mut() {
            quota.reset_daily();
        }
    }

    /// Reset all monthly counters.
    pub fn reset_all_monthly(&mut self) {
        for quota in self.quotas.values_mut() {
            quota.reset_monthly();
        }
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Aggregate metrics for rate limiting.
#[derive(Debug, Clone, Default)]
pub struct RateLimitMetrics {
    /// Total requests checked.
    pub total_requests: u64,
    /// Total requests allowed.
    pub allowed: u64,
    /// Total requests denied.
    pub denied: u64,
    /// Cumulative wait time (ms) for denied requests.
    pub total_wait_ms: u64,
}

impl RateLimitMetrics {
    /// Create empty metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an allowed request.
    pub const fn record_allowed(&mut self) {
        self.total_requests += 1;
        self.allowed += 1;
    }

    /// Record a denied request with its retry-after delay.
    pub const fn record_denied(&mut self, retry_after_ms: u64) {
        self.total_requests += 1;
        self.denied += 1;
        self.total_wait_ms += retry_after_ms;
    }

    /// Average wait time in milliseconds for denied requests.
    pub fn avg_wait_ms(&self) -> f64 {
        if self.denied == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let avg = self.total_wait_ms as f64 / self.denied as f64;
        avg
    }

    /// Denial rate as a fraction in `[0, 1]`.
    pub fn denial_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let rate = self.denied as f64 / self.total_requests as f64;
        rate
    }

    /// Reset all counters.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

// ---------------------------------------------------------------------------
// Main rate limiter
// ---------------------------------------------------------------------------

/// Unified rate limiter combining per-key rate limiting, quota management,
/// and metrics collection.
#[derive(Debug)]
pub struct RateLimiter {
    config: RateLimitConfig,
    token_buckets: HashMap<RateLimitKey, TokenBucketLimiter>,
    sliding_windows: HashMap<RateLimitKey, SlidingWindowLimiter>,
    fixed_windows: HashMap<RateLimitKey, FixedWindowLimiter>,
    leaky_buckets: HashMap<RateLimitKey, LeakyBucketLimiter>,
    quota_manager: QuotaManager,
    metrics: RateLimitMetrics,
}

impl RateLimiter {
    /// Create a new rate limiter from the given config.
    pub fn new(config: RateLimitConfig) -> Result<Self, RateLimitError> {
        config.validate()?;
        Ok(Self {
            config,
            token_buckets: HashMap::new(),
            sliding_windows: HashMap::new(),
            fixed_windows: HashMap::new(),
            leaky_buckets: HashMap::new(),
            quota_manager: QuotaManager::new(100_000, 3_000_000),
            metrics: RateLimitMetrics::new(),
        })
    }

    /// Check whether a request identified by `key` is allowed.
    pub fn check(&mut self, key: &RateLimitKey, now: Instant) -> RateLimitResult {
        let user_id = match key {
            RateLimitKey::User(id) => Some(id.as_str()),
            _ => None,
        };
        let rps = self.config.effective_rps(user_id);
        let burst = f64::from(self.config.effective_burst(user_id));

        let result = match self.config.algorithm {
            RateLimitAlgorithm::TokenBucket => {
                let bucket = self
                    .token_buckets
                    .entry(key.clone())
                    .or_insert_with(|| TokenBucketLimiter::with_instant(burst, rps, now));
                bucket.try_acquire(now)
            }
            RateLimitAlgorithm::SlidingWindow => {
                let window_dur = Duration::from_secs(1);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let max_req = burst as u64;
                let limiter = self
                    .sliding_windows
                    .entry(key.clone())
                    .or_insert_with(|| SlidingWindowLimiter::new(window_dur, max_req));
                limiter.try_acquire(now)
            }
            RateLimitAlgorithm::FixedWindow => {
                let window_dur = Duration::from_secs(1);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let max_req = burst as u64;
                let limiter = self
                    .fixed_windows
                    .entry(key.clone())
                    .or_insert_with(|| FixedWindowLimiter::with_instant(window_dur, max_req, now));
                limiter.try_acquire(now)
            }
            RateLimitAlgorithm::LeakyBucket => {
                let limiter = self
                    .leaky_buckets
                    .entry(key.clone())
                    .or_insert_with(|| LeakyBucketLimiter::with_instant(burst, rps, now));
                limiter.try_acquire(now)
            }
        };

        if result.allowed {
            self.metrics.record_allowed();
        } else {
            self.metrics.record_denied(result.retry_after_ms);
        }
        result
    }

    /// Consume quota tokens for a user (separate from rate limiting).
    pub fn consume_quota(
        &mut self,
        user_id: &str,
        token_count: u64,
    ) -> Result<(), RateLimitError> {
        self.quota_manager.try_consume(user_id, token_count)
    }

    /// Get a mutable reference to the quota for a user.
    pub fn get_quota(&mut self, user_id: &str) -> &mut UserQuota {
        self.quota_manager.get_quota(user_id)
    }

    /// Reset the limiter state for a specific key.
    pub fn reset(&mut self, key: &RateLimitKey) {
        self.token_buckets.remove(key);
        self.sliding_windows.remove(key);
        self.fixed_windows.remove(key);
        self.leaky_buckets.remove(key);
    }

    /// Reset all limiter state.
    pub fn reset_all(&mut self) {
        self.token_buckets.clear();
        self.sliding_windows.clear();
        self.fixed_windows.clear();
        self.leaky_buckets.clear();
        self.metrics.reset();
    }

    /// Current aggregate metrics.
    pub const fn metrics(&self) -> &RateLimitMetrics {
        &self.metrics
    }

    /// Current config.
    pub const fn config(&self) -> &RateLimitConfig {
        &self.config
    }

    /// Access the quota manager.
    pub const fn quota_manager(&mut self) -> &mut QuotaManager {
        &mut self.quota_manager
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn now() -> Instant {
        Instant::now()
    }

    fn advance(base: Instant, millis: u64) -> Instant {
        base + Duration::from_millis(millis)
    }

    // ---- RateLimitConfig --------------------------------------------------

    #[test]
    fn config_new_defaults() {
        let cfg = RateLimitConfig::new(10.0, 5);
        assert_eq!(cfg.requests_per_second, 10.0);
        assert_eq!(cfg.burst_size, 5);
        assert_eq!(cfg.tokens_per_minute, 60_000);
        assert_eq!(cfg.algorithm, RateLimitAlgorithm::TokenBucket);
    }

    #[test]
    fn config_validate_ok() {
        let cfg = RateLimitConfig::new(10.0, 5);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_validate_negative_rps() {
        let cfg = RateLimitConfig::new(-1.0, 5);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_nan_rps() {
        let cfg = RateLimitConfig::new(f64::NAN, 5);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_zero_burst_with_positive_rate() {
        let cfg = RateLimitConfig::new(10.0, 0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_zero_rate_zero_burst_ok() {
        let cfg = RateLimitConfig::new(0.0, 0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_effective_rps_default() {
        let cfg = RateLimitConfig::new(10.0, 5);
        assert_eq!(cfg.effective_rps(None), 10.0);
        assert_eq!(cfg.effective_rps(Some("unknown")), 10.0);
    }

    #[test]
    fn config_effective_rps_override() {
        let mut cfg = RateLimitConfig::new(10.0, 5);
        cfg.per_user_limits.insert(
            "vip".into(),
            UserLimitOverride {
                requests_per_second: Some(100.0),
                tokens_per_minute: None,
                burst_size: None,
            },
        );
        assert_eq!(cfg.effective_rps(Some("vip")), 100.0);
        assert_eq!(cfg.effective_rps(Some("normal")), 10.0);
    }

    #[test]
    fn config_effective_burst_override() {
        let mut cfg = RateLimitConfig::new(10.0, 5);
        cfg.per_user_limits.insert(
            "vip".into(),
            UserLimitOverride {
                requests_per_second: None,
                tokens_per_minute: None,
                burst_size: Some(50),
            },
        );
        assert_eq!(cfg.effective_burst(Some("vip")), 50);
        assert_eq!(cfg.effective_burst(Some("other")), 5);
    }

    // ---- RateLimitResult --------------------------------------------------

    #[test]
    fn result_allow() {
        let r = RateLimitResult::allow(4.0, 5.0);
        assert!(r.allowed);
        assert_eq!(r.retry_after_ms, 0);
        assert_eq!(r.remaining_tokens, 4.0);
        assert_eq!(r.limit, 5.0);
    }

    #[test]
    fn result_deny() {
        let r = RateLimitResult::deny(500, 0.0, 5.0);
        assert!(!r.allowed);
        assert_eq!(r.retry_after_ms, 500);
    }

    // ---- TokenBucketLimiter -----------------------------------------------

    #[test]
    fn token_bucket_starts_full() {
        let t = now();
        let mut b = TokenBucketLimiter::with_instant(5.0, 1.0, t);
        assert_eq!(b.available(t), 5.0);
    }

    #[test]
    fn token_bucket_consume_one() {
        let t = now();
        let mut b = TokenBucketLimiter::with_instant(5.0, 1.0, t);
        let r = b.try_acquire(t);
        assert!(r.allowed);
        assert_eq!(r.remaining_tokens, 4.0);
    }

    #[test]
    fn token_bucket_drain_all() {
        let t = now();
        let mut b = TokenBucketLimiter::with_instant(3.0, 1.0, t);
        assert!(b.try_acquire(t).allowed);
        assert!(b.try_acquire(t).allowed);
        assert!(b.try_acquire(t).allowed);
        let r = b.try_acquire(t);
        assert!(!r.allowed);
    }

    #[test]
    fn token_bucket_refill_after_wait() {
        let t = now();
        let mut b = TokenBucketLimiter::with_instant(2.0, 2.0, t);
        assert!(b.try_acquire(t).allowed);
        assert!(b.try_acquire(t).allowed);
        assert!(!b.try_acquire(t).allowed);
        let t2 = advance(t, 500);
        let r = b.try_acquire(t2);
        assert!(r.allowed);
    }

    #[test]
    fn token_bucket_does_not_exceed_capacity() {
        let t = now();
        let mut b = TokenBucketLimiter::with_instant(5.0, 100.0, t);
        let t2 = advance(t, 10_000);
        assert_eq!(b.available(t2), 5.0);
    }

    #[test]
    fn token_bucket_retry_after_calculation() {
        let t = now();
        let mut b = TokenBucketLimiter::with_instant(1.0, 2.0, t);
        assert!(b.try_acquire(t).allowed);
        let r = b.try_acquire(t);
        assert!(!r.allowed);
        assert!(r.retry_after_ms > 0);
        assert!(r.retry_after_ms <= 500);
    }

    #[test]
    fn token_bucket_zero_rate_never_refills() {
        let t = now();
        let mut b = TokenBucketLimiter::with_instant(1.0, 0.0, t);
        assert!(b.try_acquire(t).allowed);
        let t2 = advance(t, 60_000);
        let r = b.try_acquire(t2);
        assert!(!r.allowed);
    }

    #[test]
    fn token_bucket_high_burst() {
        let t = now();
        let mut b = TokenBucketLimiter::with_instant(1000.0, 1.0, t);
        for _ in 0..1000 {
            assert!(b.try_acquire(t).allowed);
        }
        assert!(!b.try_acquire(t).allowed);
    }

    #[test]
    fn token_bucket_fractional_refill() {
        let t = now();
        let mut b = TokenBucketLimiter::with_instant(1.0, 10.0, t);
        assert!(b.try_acquire(t).allowed);
        let t2 = advance(t, 50);
        assert!(!b.try_acquire(t2).allowed);
        let t3 = advance(t, 150);
        assert!(b.try_acquire(t3).allowed);
    }

    #[test]
    fn token_bucket_capacity_accessor() {
        let b = TokenBucketLimiter::with_instant(7.0, 3.0, now());
        assert_eq!(b.capacity(), 7.0);
        assert_eq!(b.rate(), 3.0);
    }

    #[test]
    fn token_bucket_snapshot_no_refill() {
        let t = now();
        let mut b = TokenBucketLimiter::with_instant(5.0, 1.0, t);
        b.try_acquire(t);
        assert_eq!(b.tokens_snapshot(), 4.0);
    }

    // ---- SlidingWindowLimiter ---------------------------------------------

    #[test]
    fn sliding_window_allows_up_to_limit() {
        let t = now();
        let mut sw = SlidingWindowLimiter::new(Duration::from_secs(1), 3);
        assert!(sw.try_acquire(t).allowed);
        assert!(sw.try_acquire(t).allowed);
        assert!(sw.try_acquire(t).allowed);
        assert!(!sw.try_acquire(t).allowed);
    }

    #[test]
    fn sliding_window_expires_old_entries() {
        let t = now();
        let mut sw = SlidingWindowLimiter::new(Duration::from_secs(1), 2);
        assert!(sw.try_acquire(t).allowed);
        assert!(sw.try_acquire(t).allowed);
        assert!(!sw.try_acquire(t).allowed);
        let t2 = advance(t, 1001);
        assert!(sw.try_acquire(t2).allowed);
    }

    #[test]
    fn sliding_window_remaining_decreases() {
        let t = now();
        let mut sw = SlidingWindowLimiter::new(Duration::from_secs(1), 5);
        let r1 = sw.try_acquire(t);
        assert_eq!(r1.remaining_tokens, 4.0);
        let r2 = sw.try_acquire(t);
        assert_eq!(r2.remaining_tokens, 3.0);
    }

    #[test]
    fn sliding_window_retry_after_nonzero() {
        let t = now();
        let mut sw = SlidingWindowLimiter::new(Duration::from_secs(1), 1);
        sw.try_acquire(t);
        let r = sw.try_acquire(t);
        assert!(!r.allowed);
        assert!(r.retry_after_ms > 0);
    }

    #[test]
    fn sliding_window_current_count() {
        let t = now();
        let mut sw = SlidingWindowLimiter::new(Duration::from_secs(1), 10);
        sw.try_acquire(t);
        sw.try_acquire(t);
        assert_eq!(sw.current_count(t), 2);
    }

    #[test]
    fn sliding_window_max_requests_accessor() {
        let sw = SlidingWindowLimiter::new(Duration::from_secs(1), 42);
        assert_eq!(sw.max_requests(), 42);
    }

    // ---- FixedWindowLimiter -----------------------------------------------

    #[test]
    fn fixed_window_allows_up_to_limit() {
        let t = now();
        let mut fw = FixedWindowLimiter::with_instant(Duration::from_secs(1), 3, t);
        assert!(fw.try_acquire(t).allowed);
        assert!(fw.try_acquire(t).allowed);
        assert!(fw.try_acquire(t).allowed);
        assert!(!fw.try_acquire(t).allowed);
    }

    #[test]
    fn fixed_window_resets_on_new_window() {
        let t = now();
        let mut fw = FixedWindowLimiter::with_instant(Duration::from_secs(1), 2, t);
        assert!(fw.try_acquire(t).allowed);
        assert!(fw.try_acquire(t).allowed);
        assert!(!fw.try_acquire(t).allowed);
        let t2 = advance(t, 1001);
        assert!(fw.try_acquire(t2).allowed);
        assert_eq!(fw.current_count(), 1);
    }

    #[test]
    fn fixed_window_remaining_tokens() {
        let t = now();
        let mut fw = FixedWindowLimiter::with_instant(Duration::from_secs(1), 5, t);
        let r = fw.try_acquire(t);
        assert_eq!(r.remaining_tokens, 4.0);
    }

    #[test]
    fn fixed_window_retry_after_points_to_window_end() {
        let t = now();
        let mut fw = FixedWindowLimiter::with_instant(Duration::from_secs(1), 1, t);
        fw.try_acquire(t);
        let r = fw.try_acquire(advance(t, 200));
        assert!(!r.allowed);
        assert!(r.retry_after_ms > 0);
    }

    // ---- LeakyBucketLimiter -----------------------------------------------

    #[test]
    fn leaky_bucket_allows_within_capacity() {
        let t = now();
        let mut lb = LeakyBucketLimiter::with_instant(3.0, 1.0, t);
        assert!(lb.try_acquire(t).allowed);
        assert!(lb.try_acquire(t).allowed);
        assert!(lb.try_acquire(t).allowed);
        assert!(!lb.try_acquire(t).allowed);
    }

    #[test]
    fn leaky_bucket_drains_over_time() {
        let t = now();
        let mut lb = LeakyBucketLimiter::with_instant(2.0, 2.0, t);
        assert!(lb.try_acquire(t).allowed);
        assert!(lb.try_acquire(t).allowed);
        assert!(!lb.try_acquire(t).allowed);
        let t2 = advance(t, 500);
        assert!(lb.try_acquire(t2).allowed);
    }

    #[test]
    fn leaky_bucket_constant_drain() {
        let t = now();
        let mut lb = LeakyBucketLimiter::with_instant(5.0, 10.0, t);
        for _ in 0..5 {
            assert!(lb.try_acquire(t).allowed);
        }
        assert!(!lb.try_acquire(t).allowed);
        let t2 = advance(t, 100);
        let level = lb.current_level(t2);
        assert!((level - 4.0).abs() < 0.01);
    }

    #[test]
    fn leaky_bucket_retry_after() {
        let t = now();
        let mut lb = LeakyBucketLimiter::with_instant(1.0, 1.0, t);
        lb.try_acquire(t);
        let r = lb.try_acquire(t);
        assert!(!r.allowed);
        assert!(r.retry_after_ms > 0);
    }

    #[test]
    fn leaky_bucket_accessors() {
        let lb = LeakyBucketLimiter::with_instant(10.0, 5.0, now());
        assert_eq!(lb.capacity(), 10.0);
        assert_eq!(lb.drain_rate(), 5.0);
    }

    #[test]
    fn leaky_bucket_zero_drain_never_empties() {
        let t = now();
        let mut lb = LeakyBucketLimiter::with_instant(1.0, 0.0, t);
        lb.try_acquire(t);
        let t2 = advance(t, 60_000);
        let r = lb.try_acquire(t2);
        assert!(!r.allowed);
    }

    // ---- RateLimitKey -----------------------------------------------------

    #[test]
    fn key_user_equality() {
        let a = RateLimitKey::User("alice".into());
        let b = RateLimitKey::User("alice".into());
        assert_eq!(a, b);
    }

    #[test]
    fn key_different_types_not_equal() {
        let u = RateLimitKey::User("x".into());
        let k = RateLimitKey::ApiKey("x".into());
        assert_ne!(u, k);
    }

    #[test]
    fn key_global_singleton() {
        assert_eq!(RateLimitKey::Global, RateLimitKey::Global);
    }

    // ---- UserQuota --------------------------------------------------------

    #[test]
    fn quota_new_full() {
        let q = UserQuota::new(1000, 30_000);
        assert_eq!(q.daily_remaining(), 1000);
        assert_eq!(q.monthly_remaining(), 30_000);
    }

    #[test]
    fn quota_consume_ok() {
        let mut q = UserQuota::new(100, 1000);
        assert!(q.try_consume(50).is_ok());
        assert_eq!(q.daily_remaining(), 50);
        assert_eq!(q.monthly_remaining(), 950);
    }

    #[test]
    fn quota_consume_exceeds_daily() {
        let mut q = UserQuota::new(10, 1000);
        assert!(q.try_consume(11).is_err());
    }

    #[test]
    fn quota_consume_exceeds_monthly() {
        let mut q = UserQuota::new(1000, 10);
        assert!(q.try_consume(11).is_err());
    }

    #[test]
    fn quota_reset_daily() {
        let mut q = UserQuota::new(100, 1000);
        q.try_consume(80).unwrap();
        q.reset_daily();
        assert_eq!(q.daily_remaining(), 100);
        assert_eq!(q.monthly_remaining(), 920);
    }

    #[test]
    fn quota_reset_monthly() {
        let mut q = UserQuota::new(100, 1000);
        q.try_consume(80).unwrap();
        q.reset_monthly();
        assert_eq!(q.daily_remaining(), 100);
        assert_eq!(q.monthly_remaining(), 1000);
    }

    #[test]
    fn quota_exhaustion_error_contains_remaining() {
        let mut q = UserQuota::new(5, 1000);
        q.try_consume(3).unwrap();
        let err = q.try_consume(5).unwrap_err();
        match err {
            RateLimitError::QuotaExhausted {
                daily_remaining,
                monthly_remaining,
            } => {
                assert_eq!(daily_remaining, 2);
                assert_eq!(monthly_remaining, 997);
            }
            RateLimitError::InvalidConfig(_) => panic!("expected QuotaExhausted"),
        }
    }

    // ---- QuotaManager -----------------------------------------------------

    #[test]
    fn quota_manager_default_quota() {
        let mut qm = QuotaManager::new(100, 3000);
        let q = qm.get_quota("new_user");
        assert_eq!(q.daily_limit, 100);
        assert_eq!(q.monthly_limit, 3000);
    }

    #[test]
    fn quota_manager_custom_quota() {
        let mut qm = QuotaManager::new(100, 3000);
        qm.set_quota("vip", UserQuota::new(10_000, 300_000));
        let q = qm.get_quota("vip");
        assert_eq!(q.daily_limit, 10_000);
    }

    #[test]
    fn quota_manager_try_consume() {
        let mut qm = QuotaManager::new(100, 3000);
        assert!(qm.try_consume("user1", 50).is_ok());
        assert!(qm.try_consume("user1", 60).is_err());
    }

    #[test]
    fn quota_manager_reset_daily() {
        let mut qm = QuotaManager::new(100, 3000);
        qm.try_consume("u1", 100).unwrap();
        qm.reset_all_daily();
        assert!(qm.try_consume("u1", 1).is_ok());
    }

    #[test]
    fn quota_manager_reset_monthly() {
        let mut qm = QuotaManager::new(100, 200);
        qm.try_consume("u1", 100).unwrap();
        qm.reset_all_daily();
        qm.try_consume("u1", 100).unwrap();
        assert!(qm.try_consume("u1", 1).is_err());
        qm.reset_all_monthly();
        assert!(qm.try_consume("u1", 1).is_ok());
    }

    // ---- RateLimitMetrics -------------------------------------------------

    #[test]
    fn metrics_default_empty() {
        let m = RateLimitMetrics::new();
        assert_eq!(m.total_requests, 0);
        assert_eq!(m.allowed, 0);
        assert_eq!(m.denied, 0);
        assert_eq!(m.avg_wait_ms(), 0.0);
        assert_eq!(m.denial_rate(), 0.0);
    }

    #[test]
    fn metrics_record_allowed() {
        let mut m = RateLimitMetrics::new();
        m.record_allowed();
        m.record_allowed();
        assert_eq!(m.total_requests, 2);
        assert_eq!(m.allowed, 2);
        assert_eq!(m.denied, 0);
    }

    #[test]
    fn metrics_record_denied() {
        let mut m = RateLimitMetrics::new();
        m.record_denied(100);
        m.record_denied(200);
        assert_eq!(m.denied, 2);
        assert_eq!(m.avg_wait_ms(), 150.0);
    }

    #[test]
    fn metrics_denial_rate() {
        let mut m = RateLimitMetrics::new();
        m.record_allowed();
        m.record_denied(50);
        assert_eq!(m.denial_rate(), 0.5);
    }

    #[test]
    fn metrics_reset() {
        let mut m = RateLimitMetrics::new();
        m.record_allowed();
        m.record_denied(100);
        m.reset();
        assert_eq!(m.total_requests, 0);
    }

    // ---- RateLimiter (integrated) -----------------------------------------

    #[test]
    fn limiter_token_bucket_basic() {
        let cfg = RateLimitConfig::new(10.0, 3);
        let mut rl = RateLimiter::new(cfg).unwrap();
        let t = now();
        let key = RateLimitKey::Global;
        assert!(rl.check(&key, t).allowed);
        assert!(rl.check(&key, t).allowed);
        assert!(rl.check(&key, t).allowed);
        assert!(!rl.check(&key, t).allowed);
    }

    #[test]
    fn limiter_sliding_window_basic() {
        let mut cfg = RateLimitConfig::new(10.0, 3);
        cfg.algorithm = RateLimitAlgorithm::SlidingWindow;
        let mut rl = RateLimiter::new(cfg).unwrap();
        let t = now();
        let key = RateLimitKey::Global;
        assert!(rl.check(&key, t).allowed);
        assert!(rl.check(&key, t).allowed);
        assert!(rl.check(&key, t).allowed);
        assert!(!rl.check(&key, t).allowed);
    }

    #[test]
    fn limiter_fixed_window_basic() {
        let mut cfg = RateLimitConfig::new(10.0, 3);
        cfg.algorithm = RateLimitAlgorithm::FixedWindow;
        let mut rl = RateLimiter::new(cfg).unwrap();
        let t = now();
        let key = RateLimitKey::Global;
        assert!(rl.check(&key, t).allowed);
        assert!(rl.check(&key, t).allowed);
        assert!(rl.check(&key, t).allowed);
        assert!(!rl.check(&key, t).allowed);
    }

    #[test]
    fn limiter_leaky_bucket_basic() {
        let mut cfg = RateLimitConfig::new(10.0, 3);
        cfg.algorithm = RateLimitAlgorithm::LeakyBucket;
        let mut rl = RateLimiter::new(cfg).unwrap();
        let t = now();
        let key = RateLimitKey::Global;
        assert!(rl.check(&key, t).allowed);
        assert!(rl.check(&key, t).allowed);
        assert!(rl.check(&key, t).allowed);
        assert!(!rl.check(&key, t).allowed);
    }

    #[test]
    fn limiter_per_user_isolation() {
        let cfg = RateLimitConfig::new(10.0, 2);
        let mut rl = RateLimiter::new(cfg).unwrap();
        let t = now();
        let alice = RateLimitKey::User("alice".into());
        let bob = RateLimitKey::User("bob".into());

        assert!(rl.check(&alice, t).allowed);
        assert!(rl.check(&alice, t).allowed);
        assert!(!rl.check(&alice, t).allowed);

        assert!(rl.check(&bob, t).allowed);
        assert!(rl.check(&bob, t).allowed);
        assert!(!rl.check(&bob, t).allowed);
    }

    #[test]
    fn limiter_per_user_override() {
        let mut cfg = RateLimitConfig::new(10.0, 2);
        cfg.per_user_limits.insert(
            "vip".into(),
            UserLimitOverride {
                requests_per_second: None,
                tokens_per_minute: None,
                burst_size: Some(5),
            },
        );
        let mut rl = RateLimiter::new(cfg).unwrap();
        let t = now();
        let vip = RateLimitKey::User("vip".into());
        for _ in 0..5 {
            assert!(rl.check(&vip, t).allowed);
        }
        assert!(!rl.check(&vip, t).allowed);
    }

    #[test]
    fn limiter_metrics_collected() {
        let cfg = RateLimitConfig::new(10.0, 2);
        let mut rl = RateLimiter::new(cfg).unwrap();
        let t = now();
        let key = RateLimitKey::Global;
        rl.check(&key, t);
        rl.check(&key, t);
        rl.check(&key, t);
        assert_eq!(rl.metrics().allowed, 2);
        assert_eq!(rl.metrics().denied, 1);
    }

    #[test]
    fn limiter_quota_integration() {
        let cfg = RateLimitConfig::new(10.0, 5);
        let mut rl = RateLimiter::new(cfg).unwrap();
        assert!(rl.consume_quota("user1", 50_000).is_ok());
        assert!(rl.consume_quota("user1", 60_000).is_err());
    }

    #[test]
    fn limiter_reset_key() {
        let cfg = RateLimitConfig::new(10.0, 2);
        let mut rl = RateLimiter::new(cfg).unwrap();
        let t = now();
        let key = RateLimitKey::Global;
        rl.check(&key, t);
        rl.check(&key, t);
        assert!(!rl.check(&key, t).allowed);
        rl.reset(&key);
        assert!(rl.check(&key, t).allowed);
    }

    #[test]
    fn limiter_reset_all() {
        let cfg = RateLimitConfig::new(10.0, 1);
        let mut rl = RateLimiter::new(cfg).unwrap();
        let t = now();
        let k1 = RateLimitKey::User("a".into());
        let k2 = RateLimitKey::User("b".into());
        rl.check(&k1, t);
        rl.check(&k2, t);
        rl.reset_all();
        assert!(rl.check(&k1, t).allowed);
        assert!(rl.check(&k2, t).allowed);
    }

    #[test]
    fn limiter_invalid_config_rejected() {
        let cfg = RateLimitConfig::new(-5.0, 5);
        assert!(RateLimiter::new(cfg).is_err());
    }

    // ---- burst handling ---------------------------------------------------

    #[test]
    fn burst_then_throttle_token_bucket() {
        let t = now();
        let cfg = RateLimitConfig::new(1.0, 5);
        let mut rl = RateLimiter::new(cfg).unwrap();
        let key = RateLimitKey::Global;
        for _ in 0..5 {
            assert!(rl.check(&key, t).allowed);
        }
        assert!(!rl.check(&key, t).allowed);
        let t2 = advance(t, 1000);
        assert!(rl.check(&key, t2).allowed);
        assert!(!rl.check(&key, t2).allowed);
    }

    #[test]
    fn burst_then_throttle_leaky_bucket() {
        let t = now();
        let mut cfg = RateLimitConfig::new(1.0, 5);
        cfg.algorithm = RateLimitAlgorithm::LeakyBucket;
        let mut rl = RateLimiter::new(cfg).unwrap();
        let key = RateLimitKey::Global;
        for _ in 0..5 {
            assert!(rl.check(&key, t).allowed);
        }
        assert!(!rl.check(&key, t).allowed);
        let t2 = advance(t, 1000);
        assert!(rl.check(&key, t2).allowed);
    }

    // ---- edge cases -------------------------------------------------------

    #[test]
    fn edge_max_burst_large() {
        let cfg = RateLimitConfig::new(1.0, u32::MAX);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn edge_zero_rate_allows_initial_burst() {
        let cfg = RateLimitConfig::new(0.0, 0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn edge_quota_exactly_at_limit() {
        let mut q = UserQuota::new(10, 100);
        assert!(q.try_consume(10).is_ok());
        assert!(q.try_consume(1).is_err());
    }

    #[test]
    fn edge_quota_zero_consume() {
        let mut q = UserQuota::new(10, 100);
        assert!(q.try_consume(0).is_ok());
        assert_eq!(q.daily_remaining(), 10);
    }

    // ---- error display ----------------------------------------------------

    #[test]
    fn error_display_invalid_config() {
        let e = RateLimitError::InvalidConfig("bad".into());
        assert!(format!("{e}").contains("bad"));
    }

    #[test]
    fn error_display_quota_exhausted() {
        let e = RateLimitError::QuotaExhausted {
            daily_remaining: 0,
            monthly_remaining: 5,
        };
        let s = format!("{e}");
        assert!(s.contains("daily_remaining=0"));
        assert!(s.contains("monthly_remaining=5"));
    }

    // ---- property-style tests ---------------------------------------------

    #[test]
    fn prop_token_bucket_tokens_never_exceed_capacity() {
        let t = now();
        let mut b = TokenBucketLimiter::with_instant(10.0, 5.0, t);
        for i in 0..100 {
            let ti = advance(t, i * 50);
            b.try_acquire(ti);
            let _ = b.available(ti);
            assert!(b.tokens_snapshot() <= b.capacity());
        }
    }

    #[test]
    fn prop_leaky_bucket_level_never_negative() {
        let t = now();
        let mut lb = LeakyBucketLimiter::with_instant(5.0, 10.0, t);
        for i in 0..100 {
            let ti = advance(t, i * 200);
            lb.try_acquire(ti);
            assert!(lb.current_level(ti) >= 0.0);
        }
    }

    #[test]
    fn prop_metrics_total_equals_allowed_plus_denied() {
        let mut m = RateLimitMetrics::new();
        for i in 0..50 {
            if i % 3 == 0 {
                m.record_denied(10);
            } else {
                m.record_allowed();
            }
        }
        assert_eq!(m.total_requests, m.allowed + m.denied);
    }

    #[test]
    fn prop_denial_rate_bounded() {
        let mut m = RateLimitMetrics::new();
        for _ in 0..10 {
            m.record_allowed();
        }
        for _ in 0..5 {
            m.record_denied(1);
        }
        let rate = m.denial_rate();
        assert!((0.0..=1.0).contains(&rate));
    }
}
