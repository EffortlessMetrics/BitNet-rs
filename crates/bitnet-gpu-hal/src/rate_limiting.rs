//! Rate limiting middleware for inference requests.
//!
//! Provides configurable rate limiting with four algorithms:
//! token bucket, sliding window, fixed window, and leaky bucket.
//! Includes concurrency limiting and per-user quota management.

use std::collections::HashMap;
use std::fmt;

// ── Configuration ─────────────────────────────────────────────────────────

/// Rate limiting algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Algorithm {
    /// Classic token bucket: smooth burst handling.
    TokenBucket,
    /// Sliding window counter: accurate over rolling intervals.
    SlidingWindow,
    /// Fixed window counter: simple per-interval limits.
    FixedWindow,
    /// Leaky bucket: constant drain rate.
    LeakyBucket,
}

impl fmt::Display for Algorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TokenBucket => write!(f, "TokenBucket"),
            Self::SlidingWindow => write!(f, "SlidingWindow"),
            Self::FixedWindow => write!(f, "FixedWindow"),
            Self::LeakyBucket => write!(f, "LeakyBucket"),
        }
    }
}

/// Key that identifies the scope of a rate limit.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RateLimitKey {
    /// Per-user limit, keyed by user ID.
    User(String),
    /// Per-IP address limit.
    Ip(String),
    /// Per-API-key limit.
    ApiKey(String),
    /// Global (shared across all callers).
    Global,
}

impl fmt::Display for RateLimitKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::User(id) => write!(f, "user:{id}"),
            Self::Ip(ip) => write!(f, "ip:{ip}"),
            Self::ApiKey(key) => write!(f, "api_key:{key}"),
            Self::Global => write!(f, "global"),
        }
    }
}

/// Result of a rate limit check.
#[derive(Debug, Clone, PartialEq)]
pub enum RateLimitResult {
    /// Request is allowed through.
    Allowed,
    /// Request is rate limited; retry after the given milliseconds.
    RateLimited { retry_after_ms: u64 },
    /// Request is queued at the given position.
    Queued { position: usize },
}

/// Configuration for rate limiting.
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum requests per second.
    pub max_requests_per_second: f64,
    /// Burst size (peak tokens available).
    pub burst_size: u32,
    /// Per-user limits (overrides global for matched keys).
    pub per_user_limits: HashMap<String, f64>,
    /// Global limit applied when no per-user override matches.
    pub global_limit: f64,
    /// Algorithm to use.
    pub algorithm: Algorithm,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_requests_per_second: 100.0,
            burst_size: 50,
            per_user_limits: HashMap::new(),
            global_limit: 1000.0,
            algorithm: Algorithm::TokenBucket,
        }
    }
}

// ── Token Bucket ──────────────────────────────────────────────────────────

/// Token bucket rate limiter.
///
/// Tokens refill at a steady rate up to a maximum burst capacity.
pub struct TokenBucketLimiter {
    /// Current available tokens.
    tokens: f64,
    /// Maximum tokens (burst capacity).
    max_tokens: f64,
    /// Tokens added per millisecond.
    refill_rate: f64,
    /// Timestamp of last refill (ms since epoch).
    last_refill_ms: u64,
}

impl TokenBucketLimiter {
    /// Create a new token bucket.
    ///
    /// - `rate_per_second`: sustained request rate.
    /// - `burst_size`: peak burst capacity.
    pub fn new(rate_per_second: f64, burst_size: u32) -> Self {
        let max_tokens = f64::from(burst_size);
        Self {
            tokens: max_tokens,
            max_tokens,
            refill_rate: rate_per_second / 1000.0,
            last_refill_ms: 0,
        }
    }

    /// Refill tokens based on elapsed time.
    pub fn refill(&mut self, now_ms: u64) {
        if now_ms > self.last_refill_ms {
            let elapsed = now_ms - self.last_refill_ms;
            #[allow(clippy::cast_precision_loss)]
            let added = elapsed as f64 * self.refill_rate;
            self.tokens = (self.tokens + added).min(self.max_tokens);
            self.last_refill_ms = now_ms;
        }
    }

    /// Try to acquire one token. Returns the rate-limit result.
    pub fn try_acquire(&mut self, now_ms: u64) -> RateLimitResult {
        self.refill(now_ms);
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            RateLimitResult::Allowed
        } else {
            let deficit = 1.0 - self.tokens;
            let wait_ms = if self.refill_rate > 0.0 {
                (deficit / self.refill_rate).ceil() as u64
            } else {
                u64::MAX
            };
            RateLimitResult::RateLimited { retry_after_ms: wait_ms }
        }
    }

    /// Current available tokens.
    pub fn available_tokens(&self) -> f64 {
        self.tokens
    }
}

// ── Sliding Window ────────────────────────────────────────────────────────

/// Sliding window rate limiter.
///
/// Tracks request timestamps and counts within a rolling window.
pub struct SlidingWindowLimiter {
    /// Timestamps of requests (ms since epoch).
    timestamps: Vec<u64>,
    /// Window duration in milliseconds.
    window_ms: u64,
    /// Maximum requests allowed per window.
    max_requests: u32,
}

impl SlidingWindowLimiter {
    /// Create a new sliding window limiter.
    pub fn new(max_requests: u32, window_ms: u64) -> Self {
        Self { timestamps: Vec::new(), window_ms, max_requests }
    }

    /// Evict timestamps older than the window boundary.
    fn evict(&mut self, now_ms: u64) {
        let cutoff = now_ms.saturating_sub(self.window_ms);
        self.timestamps.retain(|&ts| ts >= cutoff);
    }

    /// Try to record a request. Returns the rate-limit result.
    pub fn try_acquire(&mut self, now_ms: u64) -> RateLimitResult {
        self.evict(now_ms);
        if (self.timestamps.len() as u32) < self.max_requests {
            self.timestamps.push(now_ms);
            RateLimitResult::Allowed
        } else {
            // Earliest timestamp in window determines when a slot opens.
            let earliest = self.timestamps.first().copied().unwrap_or(now_ms);
            let retry_after = (earliest + self.window_ms).saturating_sub(now_ms);
            RateLimitResult::RateLimited { retry_after_ms: retry_after.max(1) }
        }
    }

    /// Current request count within the window.
    pub fn current_count(&self) -> usize {
        self.timestamps.len()
    }
}

// ── Fixed Window ──────────────────────────────────────────────────────────

/// Fixed window rate limiter.
///
/// Counts requests in discrete time windows that reset on boundary.
pub struct FixedWindowLimiter {
    /// Current window start (ms since epoch).
    window_start: u64,
    /// Window duration in milliseconds.
    window_ms: u64,
    /// Requests counted in the current window.
    count: u32,
    /// Maximum requests per window.
    max_requests: u32,
}

impl FixedWindowLimiter {
    /// Create a new fixed window limiter.
    pub fn new(max_requests: u32, window_ms: u64) -> Self {
        Self { window_start: 0, window_ms, count: 0, max_requests }
    }

    /// Try to record a request.
    pub fn try_acquire(&mut self, now_ms: u64) -> RateLimitResult {
        // Reset window if we've passed the boundary.
        if now_ms >= self.window_start + self.window_ms {
            self.window_start = now_ms - (now_ms % self.window_ms.max(1));
            self.count = 0;
        }
        if self.count < self.max_requests {
            self.count += 1;
            RateLimitResult::Allowed
        } else {
            let window_end = self.window_start + self.window_ms;
            let retry_after = window_end.saturating_sub(now_ms).max(1);
            RateLimitResult::RateLimited { retry_after_ms: retry_after }
        }
    }

    /// Current request count in this window.
    pub fn current_count(&self) -> u32 {
        self.count
    }
}

// ── Leaky Bucket ──────────────────────────────────────────────────────────

/// Leaky bucket rate limiter.
///
/// Requests fill a bucket that drains at a constant rate.
pub struct LeakyBucketLimiter {
    /// Current water level in the bucket.
    water: f64,
    /// Maximum bucket capacity.
    capacity: f64,
    /// Drain rate (units per millisecond).
    drain_rate: f64,
    /// Last drain timestamp (ms since epoch).
    last_drain_ms: u64,
}

impl LeakyBucketLimiter {
    /// Create a new leaky bucket.
    ///
    /// - `drain_rate_per_second`: constant drain rate.
    /// - `capacity`: maximum bucket size.
    pub fn new(drain_rate_per_second: f64, capacity: u32) -> Self {
        Self {
            water: 0.0,
            capacity: f64::from(capacity),
            drain_rate: drain_rate_per_second / 1000.0,
            last_drain_ms: 0,
        }
    }

    /// Drain water based on elapsed time.
    fn drain(&mut self, now_ms: u64) {
        if now_ms > self.last_drain_ms {
            let elapsed = now_ms - self.last_drain_ms;
            #[allow(clippy::cast_precision_loss)]
            let drained = elapsed as f64 * self.drain_rate;
            self.water = (self.water - drained).max(0.0);
            self.last_drain_ms = now_ms;
        }
    }

    /// Try to add a request to the bucket.
    pub fn try_acquire(&mut self, now_ms: u64) -> RateLimitResult {
        self.drain(now_ms);
        if self.water + 1.0 <= self.capacity {
            self.water += 1.0;
            RateLimitResult::Allowed
        } else {
            let excess = (self.water + 1.0) - self.capacity;
            let wait_ms = if self.drain_rate > 0.0 {
                (excess / self.drain_rate).ceil() as u64
            } else {
                u64::MAX
            };
            RateLimitResult::RateLimited { retry_after_ms: wait_ms }
        }
    }

    /// Current water level.
    pub fn current_level(&self) -> f64 {
        self.water
    }
}

// ── Concurrency Limiter ───────────────────────────────────────────────────

/// Limits the number of concurrent in-flight requests.
pub struct ConcurrencyLimiter {
    /// Maximum concurrent requests.
    max_concurrent: usize,
    /// Currently in-flight request count.
    in_flight: usize,
    /// Queue of waiting requests (positions).
    queue: Vec<u64>,
    next_ticket: u64,
}

impl ConcurrencyLimiter {
    /// Create a new concurrency limiter.
    pub fn new(max_concurrent: usize) -> Self {
        Self { max_concurrent, in_flight: 0, queue: Vec::new(), next_ticket: 0 }
    }

    /// Try to acquire a concurrency slot.
    pub fn try_acquire(&mut self) -> RateLimitResult {
        if self.in_flight < self.max_concurrent {
            self.in_flight += 1;
            RateLimitResult::Allowed
        } else {
            let ticket = self.next_ticket;
            self.next_ticket += 1;
            self.queue.push(ticket);
            RateLimitResult::Queued { position: self.queue.len() }
        }
    }

    /// Release a concurrency slot and admit next queued request.
    ///
    /// Returns `true` if a queued request was promoted.
    pub fn release(&mut self) -> bool {
        if self.in_flight > 0 {
            self.in_flight -= 1;
        }
        if !self.queue.is_empty() && self.in_flight < self.max_concurrent {
            self.queue.remove(0);
            self.in_flight += 1;
            return true;
        }
        false
    }

    /// Number of currently in-flight requests.
    pub fn in_flight(&self) -> usize {
        self.in_flight
    }

    /// Number of queued requests.
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }
}

// ── Quota Manager ─────────────────────────────────────────────────────────

/// Per-user token quota.
#[derive(Debug, Clone)]
pub struct UserQuota {
    /// Maximum tokens allowed in the quota period.
    pub max_tokens: u64,
    /// Tokens consumed so far.
    pub used_tokens: u64,
    /// Quota period in milliseconds.
    pub period_ms: u64,
    /// Period start timestamp (ms since epoch).
    pub period_start: u64,
}

/// Manages per-user quotas with token counting.
pub struct QuotaManager {
    quotas: HashMap<String, UserQuota>,
    default_max_tokens: u64,
    default_period_ms: u64,
}

impl QuotaManager {
    /// Create a new quota manager with defaults.
    pub fn new(default_max_tokens: u64, default_period_ms: u64) -> Self {
        Self { quotas: HashMap::new(), default_max_tokens, default_period_ms }
    }

    /// Set a custom quota for a user.
    pub fn set_quota(&mut self, user_id: &str, max_tokens: u64) {
        let entry = self.quotas.entry(user_id.to_owned()).or_insert_with(|| UserQuota {
            max_tokens,
            used_tokens: 0,
            period_ms: self.default_period_ms,
            period_start: 0,
        });
        entry.max_tokens = max_tokens;
    }

    /// Try to consume `tokens` from a user's quota.
    pub fn try_consume(&mut self, user_id: &str, tokens: u64, now_ms: u64) -> RateLimitResult {
        let default_max = self.default_max_tokens;
        let default_period = self.default_period_ms;
        let quota = self.quotas.entry(user_id.to_owned()).or_insert_with(|| UserQuota {
            max_tokens: default_max,
            used_tokens: 0,
            period_ms: default_period,
            period_start: now_ms,
        });

        // Reset period if expired.
        if now_ms >= quota.period_start + quota.period_ms {
            quota.used_tokens = 0;
            quota.period_start = now_ms;
        }

        if quota.used_tokens + tokens <= quota.max_tokens {
            quota.used_tokens += tokens;
            RateLimitResult::Allowed
        } else {
            let period_end = quota.period_start + quota.period_ms;
            let retry_after = period_end.saturating_sub(now_ms).max(1);
            RateLimitResult::RateLimited { retry_after_ms: retry_after }
        }
    }

    /// Get remaining tokens for a user.
    pub fn remaining(&self, user_id: &str) -> u64 {
        self.quotas
            .get(user_id)
            .map_or(self.default_max_tokens, |q| q.max_tokens.saturating_sub(q.used_tokens))
    }

    /// Get quota info for a user (if exists).
    pub fn get_quota(&self, user_id: &str) -> Option<&UserQuota> {
        self.quotas.get(user_id)
    }
}

// ── Metrics ───────────────────────────────────────────────────────────────

/// Rate limiting metrics.
#[derive(Debug, Clone, Default)]
pub struct RateLimitMetrics {
    /// Total requests allowed.
    pub allowed: u64,
    /// Total requests denied (rate limited).
    pub denied: u64,
    /// Total requests queued.
    pub queued: u64,
    /// Cumulative wait time in milliseconds (for average calculation).
    pub total_wait_ms: u64,
    /// Number of requests that had a wait time recorded.
    pub wait_count: u64,
}

impl RateLimitMetrics {
    /// Record an allowed request.
    pub fn record_allowed(&mut self) {
        self.allowed += 1;
    }

    /// Record a denied request.
    pub fn record_denied(&mut self) {
        self.denied += 1;
    }

    /// Record a queued request.
    pub fn record_queued(&mut self) {
        self.queued += 1;
    }

    /// Record a wait time in milliseconds.
    pub fn record_wait(&mut self, wait_ms: u64) {
        self.total_wait_ms += wait_ms;
        self.wait_count += 1;
    }

    /// Average wait time in milliseconds, or 0 if no waits recorded.
    pub fn avg_wait_ms(&self) -> f64 {
        if self.wait_count == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            {
                self.total_wait_ms as f64 / self.wait_count as f64
            }
        }
    }

    /// Total requests processed (allowed + denied + queued).
    pub fn total_requests(&self) -> u64 {
        self.allowed + self.denied + self.queued
    }

    /// Reset all metrics to zero.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

// ── Unified Rate Limiter ──────────────────────────────────────────────────

/// Configurable rate limiter that delegates to the chosen algorithm.
pub struct RateLimiter {
    config: RateLimitConfig,
    token_buckets: HashMap<String, TokenBucketLimiter>,
    sliding_windows: HashMap<String, SlidingWindowLimiter>,
    fixed_windows: HashMap<String, FixedWindowLimiter>,
    leaky_buckets: HashMap<String, LeakyBucketLimiter>,
    metrics: RateLimitMetrics,
}

impl RateLimiter {
    /// Create a new rate limiter with the given configuration.
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            token_buckets: HashMap::new(),
            sliding_windows: HashMap::new(),
            fixed_windows: HashMap::new(),
            leaky_buckets: HashMap::new(),
            metrics: RateLimitMetrics::default(),
        }
    }

    /// Check if a request identified by `key` should be allowed.
    pub fn check(&mut self, key: &RateLimitKey, now_ms: u64) -> RateLimitResult {
        let key_str = key.to_string();

        let rate = match key {
            RateLimitKey::User(id) => self
                .config
                .per_user_limits
                .get(id)
                .copied()
                .unwrap_or(self.config.max_requests_per_second),
            _ => self.config.max_requests_per_second,
        };

        let result = match self.config.algorithm {
            Algorithm::TokenBucket => {
                let limiter = self
                    .token_buckets
                    .entry(key_str)
                    .or_insert_with(|| TokenBucketLimiter::new(rate, self.config.burst_size));
                limiter.try_acquire(now_ms)
            }
            Algorithm::SlidingWindow => {
                let window_ms = (1000.0 / rate * f64::from(self.config.burst_size)) as u64;
                let limiter = self.sliding_windows.entry(key_str).or_insert_with(|| {
                    SlidingWindowLimiter::new(self.config.burst_size, window_ms.max(1))
                });
                limiter.try_acquire(now_ms)
            }
            Algorithm::FixedWindow => {
                let window_ms = (1000.0 / rate * f64::from(self.config.burst_size)) as u64;
                let limiter = self.fixed_windows.entry(key_str).or_insert_with(|| {
                    FixedWindowLimiter::new(self.config.burst_size, window_ms.max(1))
                });
                limiter.try_acquire(now_ms)
            }
            Algorithm::LeakyBucket => {
                let limiter = self
                    .leaky_buckets
                    .entry(key_str)
                    .or_insert_with(|| LeakyBucketLimiter::new(rate, self.config.burst_size));
                limiter.try_acquire(now_ms)
            }
        };

        match &result {
            RateLimitResult::Allowed => self.metrics.record_allowed(),
            RateLimitResult::RateLimited { retry_after_ms } => {
                self.metrics.record_denied();
                self.metrics.record_wait(*retry_after_ms);
            }
            RateLimitResult::Queued { .. } => self.metrics.record_queued(),
        }

        result
    }

    /// Get a reference to current metrics.
    pub fn metrics(&self) -> &RateLimitMetrics {
        &self.metrics
    }

    /// Reset metrics.
    pub fn reset_metrics(&mut self) {
        self.metrics.reset();
    }

    /// Get the current configuration.
    pub fn config(&self) -> &RateLimitConfig {
        &self.config
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Algorithm Display ─────────────────────────────────────────────

    #[test]
    fn algorithm_display() {
        assert_eq!(Algorithm::TokenBucket.to_string(), "TokenBucket");
        assert_eq!(Algorithm::SlidingWindow.to_string(), "SlidingWindow");
        assert_eq!(Algorithm::FixedWindow.to_string(), "FixedWindow");
        assert_eq!(Algorithm::LeakyBucket.to_string(), "LeakyBucket");
    }

    // ── RateLimitKey Display ──────────────────────────────────────────

    #[test]
    fn key_display_user() {
        let k = RateLimitKey::User("alice".into());
        assert_eq!(k.to_string(), "user:alice");
    }

    #[test]
    fn key_display_ip() {
        let k = RateLimitKey::Ip("127.0.0.1".into());
        assert_eq!(k.to_string(), "ip:127.0.0.1");
    }

    #[test]
    fn key_display_api_key() {
        let k = RateLimitKey::ApiKey("sk-abc".into());
        assert_eq!(k.to_string(), "api_key:sk-abc");
    }

    #[test]
    fn key_display_global() {
        assert_eq!(RateLimitKey::Global.to_string(), "global");
    }

    // ── RateLimitConfig default ───────────────────────────────────────

    #[test]
    fn config_default_values() {
        let cfg = RateLimitConfig::default();
        assert!((cfg.max_requests_per_second - 100.0).abs() < f64::EPSILON);
        assert_eq!(cfg.burst_size, 50);
        assert!(cfg.per_user_limits.is_empty());
        assert!((cfg.global_limit - 1000.0).abs() < f64::EPSILON);
        assert_eq!(cfg.algorithm, Algorithm::TokenBucket);
    }

    // ── Token Bucket ──────────────────────────────────────────────────

    #[test]
    fn token_bucket_allows_within_burst() {
        let mut tb = TokenBucketLimiter::new(10.0, 5);
        for _ in 0..5 {
            assert_eq!(tb.try_acquire(0), RateLimitResult::Allowed);
        }
    }

    #[test]
    fn token_bucket_denies_over_burst() {
        let mut tb = TokenBucketLimiter::new(10.0, 3);
        for _ in 0..3 {
            assert_eq!(tb.try_acquire(0), RateLimitResult::Allowed);
        }
        match tb.try_acquire(0) {
            RateLimitResult::RateLimited { retry_after_ms } => {
                assert!(retry_after_ms > 0);
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[test]
    fn token_bucket_refills_over_time() {
        let mut tb = TokenBucketLimiter::new(10.0, 2);
        // Drain all tokens.
        assert_eq!(tb.try_acquire(0), RateLimitResult::Allowed);
        assert_eq!(tb.try_acquire(0), RateLimitResult::Allowed);
        assert!(matches!(tb.try_acquire(0), RateLimitResult::RateLimited { .. }));
        // Advance 200ms → should have ~2 tokens (10/s * 0.2s = 2).
        assert_eq!(tb.try_acquire(200), RateLimitResult::Allowed);
    }

    #[test]
    fn token_bucket_available_tokens() {
        let tb = TokenBucketLimiter::new(10.0, 5);
        assert!((tb.available_tokens() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn token_bucket_refill_caps_at_max() {
        let mut tb = TokenBucketLimiter::new(10.0, 5);
        tb.refill(10_000); // Long time, but should cap at 5.
        assert!(tb.available_tokens() <= 5.0 + f64::EPSILON);
    }

    #[test]
    fn token_bucket_no_refill_backward_time() {
        let mut tb = TokenBucketLimiter::new(10.0, 5);
        assert_eq!(tb.try_acquire(100), RateLimitResult::Allowed);
        let before = tb.available_tokens();
        tb.refill(50); // time goes backward — no change
        assert!((tb.available_tokens() - before).abs() < f64::EPSILON);
    }

    // ── Sliding Window ────────────────────────────────────────────────

    #[test]
    fn sliding_window_allows_within_limit() {
        let mut sw = SlidingWindowLimiter::new(3, 1000);
        for i in 0..3 {
            assert_eq!(sw.try_acquire(i * 100), RateLimitResult::Allowed);
        }
    }

    #[test]
    fn sliding_window_denies_over_limit() {
        let mut sw = SlidingWindowLimiter::new(2, 1000);
        assert_eq!(sw.try_acquire(0), RateLimitResult::Allowed);
        assert_eq!(sw.try_acquire(100), RateLimitResult::Allowed);
        match sw.try_acquire(200) {
            RateLimitResult::RateLimited { retry_after_ms } => {
                assert!(retry_after_ms > 0);
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[test]
    fn sliding_window_allows_after_expiry() {
        let mut sw = SlidingWindowLimiter::new(1, 1000);
        assert_eq!(sw.try_acquire(0), RateLimitResult::Allowed);
        assert!(matches!(sw.try_acquire(500), RateLimitResult::RateLimited { .. }));
        // Window slides past the first request.
        assert_eq!(sw.try_acquire(1001), RateLimitResult::Allowed);
    }

    #[test]
    fn sliding_window_current_count() {
        let mut sw = SlidingWindowLimiter::new(10, 1000);
        sw.try_acquire(0);
        sw.try_acquire(100);
        assert_eq!(sw.current_count(), 2);
    }

    // ── Fixed Window ──────────────────────────────────────────────────

    #[test]
    fn fixed_window_allows_within_limit() {
        let mut fw = FixedWindowLimiter::new(3, 1000);
        for _ in 0..3 {
            assert_eq!(fw.try_acquire(500), RateLimitResult::Allowed);
        }
    }

    #[test]
    fn fixed_window_denies_over_limit() {
        let mut fw = FixedWindowLimiter::new(2, 1000);
        assert_eq!(fw.try_acquire(100), RateLimitResult::Allowed);
        assert_eq!(fw.try_acquire(200), RateLimitResult::Allowed);
        match fw.try_acquire(300) {
            RateLimitResult::RateLimited { retry_after_ms } => {
                assert!(retry_after_ms > 0);
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[test]
    fn fixed_window_resets_at_boundary() {
        let mut fw = FixedWindowLimiter::new(1, 1000);
        assert_eq!(fw.try_acquire(100), RateLimitResult::Allowed);
        assert!(matches!(fw.try_acquire(200), RateLimitResult::RateLimited { .. }));
        // Next window boundary.
        assert_eq!(fw.try_acquire(1000), RateLimitResult::Allowed);
    }

    #[test]
    fn fixed_window_current_count() {
        let mut fw = FixedWindowLimiter::new(10, 1000);
        fw.try_acquire(100);
        fw.try_acquire(200);
        assert_eq!(fw.current_count(), 2);
    }

    // ── Leaky Bucket ──────────────────────────────────────────────────

    #[test]
    fn leaky_bucket_allows_within_capacity() {
        let mut lb = LeakyBucketLimiter::new(10.0, 3);
        for _ in 0..3 {
            assert_eq!(lb.try_acquire(0), RateLimitResult::Allowed);
        }
    }

    #[test]
    fn leaky_bucket_denies_over_capacity() {
        let mut lb = LeakyBucketLimiter::new(10.0, 2);
        assert_eq!(lb.try_acquire(0), RateLimitResult::Allowed);
        assert_eq!(lb.try_acquire(0), RateLimitResult::Allowed);
        match lb.try_acquire(0) {
            RateLimitResult::RateLimited { retry_after_ms } => {
                assert!(retry_after_ms > 0);
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[test]
    fn leaky_bucket_drains_over_time() {
        let mut lb = LeakyBucketLimiter::new(10.0, 2);
        assert_eq!(lb.try_acquire(0), RateLimitResult::Allowed);
        assert_eq!(lb.try_acquire(0), RateLimitResult::Allowed);
        // Bucket full. Wait 200ms → drains 2 units at 10/s.
        assert_eq!(lb.try_acquire(200), RateLimitResult::Allowed);
    }

    #[test]
    fn leaky_bucket_current_level() {
        let mut lb = LeakyBucketLimiter::new(10.0, 5);
        lb.try_acquire(0);
        assert!((lb.current_level() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn leaky_bucket_no_drain_backward_time() {
        let mut lb = LeakyBucketLimiter::new(10.0, 5);
        lb.try_acquire(100);
        let before = lb.current_level();
        lb.drain(50); // backward — no drain
        assert!((lb.current_level() - before).abs() < f64::EPSILON);
    }

    // ── Concurrency Limiter ───────────────────────────────────────────

    #[test]
    fn concurrency_allows_within_limit() {
        let mut cl = ConcurrencyLimiter::new(2);
        assert_eq!(cl.try_acquire(), RateLimitResult::Allowed);
        assert_eq!(cl.try_acquire(), RateLimitResult::Allowed);
        assert_eq!(cl.in_flight(), 2);
    }

    #[test]
    fn concurrency_queues_over_limit() {
        let mut cl = ConcurrencyLimiter::new(1);
        assert_eq!(cl.try_acquire(), RateLimitResult::Allowed);
        match cl.try_acquire() {
            RateLimitResult::Queued { position } => {
                assert_eq!(position, 1);
            }
            other => panic!("expected Queued, got {other:?}"),
        }
    }

    #[test]
    fn concurrency_release_promotes_queued() {
        let mut cl = ConcurrencyLimiter::new(1);
        assert_eq!(cl.try_acquire(), RateLimitResult::Allowed);
        assert!(matches!(cl.try_acquire(), RateLimitResult::Queued { .. }));
        assert!(cl.release()); // promotes queued request
        assert_eq!(cl.in_flight(), 1);
        assert_eq!(cl.queue_len(), 0);
    }

    #[test]
    fn concurrency_release_no_queue() {
        let mut cl = ConcurrencyLimiter::new(2);
        assert_eq!(cl.try_acquire(), RateLimitResult::Allowed);
        assert!(!cl.release()); // nothing queued, no promotion
        assert_eq!(cl.in_flight(), 0);
    }

    #[test]
    fn concurrency_queue_ordering() {
        let mut cl = ConcurrencyLimiter::new(1);
        assert_eq!(cl.try_acquire(), RateLimitResult::Allowed);
        // Queue three more.
        for expected_pos in 1..=3 {
            match cl.try_acquire() {
                RateLimitResult::Queued { position } => {
                    assert_eq!(position, expected_pos);
                }
                other => panic!("expected Queued, got {other:?}"),
            }
        }
        assert_eq!(cl.queue_len(), 3);
    }

    // ── Quota Manager ─────────────────────────────────────────────────

    #[test]
    fn quota_allows_within_limit() {
        let mut qm = QuotaManager::new(100, 60_000);
        assert_eq!(qm.try_consume("alice", 50, 0), RateLimitResult::Allowed);
        assert_eq!(qm.remaining("alice"), 50);
    }

    #[test]
    fn quota_denies_over_limit() {
        let mut qm = QuotaManager::new(100, 60_000);
        assert_eq!(qm.try_consume("alice", 80, 0), RateLimitResult::Allowed);
        match qm.try_consume("alice", 30, 1000) {
            RateLimitResult::RateLimited { retry_after_ms } => {
                assert!(retry_after_ms > 0);
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[test]
    fn quota_resets_after_period() {
        let mut qm = QuotaManager::new(100, 1000);
        assert_eq!(qm.try_consume("alice", 100, 0), RateLimitResult::Allowed);
        assert!(matches!(qm.try_consume("alice", 1, 500), RateLimitResult::RateLimited { .. }));
        // Period expired.
        assert_eq!(qm.try_consume("alice", 1, 1000), RateLimitResult::Allowed);
    }

    #[test]
    fn quota_custom_per_user() {
        let mut qm = QuotaManager::new(100, 60_000);
        qm.set_quota("vip", 1000);
        assert_eq!(qm.try_consume("vip", 500, 0), RateLimitResult::Allowed);
        assert_eq!(qm.remaining("vip"), 500);
    }

    #[test]
    fn quota_default_remaining() {
        let qm = QuotaManager::new(100, 60_000);
        assert_eq!(qm.remaining("unknown_user"), 100);
    }

    #[test]
    fn quota_get_existing() {
        let mut qm = QuotaManager::new(100, 60_000);
        qm.try_consume("alice", 10, 0);
        let q = qm.get_quota("alice").unwrap();
        assert_eq!(q.used_tokens, 10);
    }

    #[test]
    fn quota_get_nonexistent() {
        let qm = QuotaManager::new(100, 60_000);
        assert!(qm.get_quota("unknown").is_none());
    }

    // ── Metrics ───────────────────────────────────────────────────────

    #[test]
    fn metrics_record_allowed() {
        let mut m = RateLimitMetrics::default();
        m.record_allowed();
        m.record_allowed();
        assert_eq!(m.allowed, 2);
        assert_eq!(m.total_requests(), 2);
    }

    #[test]
    fn metrics_record_denied() {
        let mut m = RateLimitMetrics::default();
        m.record_denied();
        assert_eq!(m.denied, 1);
    }

    #[test]
    fn metrics_record_queued() {
        let mut m = RateLimitMetrics::default();
        m.record_queued();
        assert_eq!(m.queued, 1);
    }

    #[test]
    fn metrics_avg_wait_no_waits() {
        let m = RateLimitMetrics::default();
        assert!((m.avg_wait_ms() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_avg_wait() {
        let mut m = RateLimitMetrics::default();
        m.record_wait(100);
        m.record_wait(200);
        assert!((m.avg_wait_ms() - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_total_requests() {
        let mut m = RateLimitMetrics::default();
        m.record_allowed();
        m.record_denied();
        m.record_queued();
        assert_eq!(m.total_requests(), 3);
    }

    #[test]
    fn metrics_reset() {
        let mut m = RateLimitMetrics::default();
        m.record_allowed();
        m.record_denied();
        m.record_wait(50);
        m.reset();
        assert_eq!(m.allowed, 0);
        assert_eq!(m.denied, 0);
        assert_eq!(m.total_wait_ms, 0);
    }

    // ── Unified RateLimiter ───────────────────────────────────────────

    #[test]
    fn rate_limiter_token_bucket_default() {
        let cfg = RateLimitConfig::default();
        let mut rl = RateLimiter::new(cfg);
        let key = RateLimitKey::Global;
        assert_eq!(rl.check(&key, 0), RateLimitResult::Allowed);
    }

    #[test]
    fn rate_limiter_sliding_window() {
        let cfg = RateLimitConfig {
            algorithm: Algorithm::SlidingWindow,
            max_requests_per_second: 10.0,
            burst_size: 2,
            ..Default::default()
        };
        let mut rl = RateLimiter::new(cfg);
        let key = RateLimitKey::Global;
        assert_eq!(rl.check(&key, 0), RateLimitResult::Allowed);
        assert_eq!(rl.check(&key, 10), RateLimitResult::Allowed);
        assert!(matches!(rl.check(&key, 20), RateLimitResult::RateLimited { .. }));
    }

    #[test]
    fn rate_limiter_fixed_window() {
        let cfg = RateLimitConfig {
            algorithm: Algorithm::FixedWindow,
            max_requests_per_second: 10.0,
            burst_size: 2,
            ..Default::default()
        };
        let mut rl = RateLimiter::new(cfg);
        let key = RateLimitKey::Global;
        assert_eq!(rl.check(&key, 0), RateLimitResult::Allowed);
        assert_eq!(rl.check(&key, 50), RateLimitResult::Allowed);
        assert!(matches!(rl.check(&key, 100), RateLimitResult::RateLimited { .. }));
    }

    #[test]
    fn rate_limiter_leaky_bucket() {
        let cfg = RateLimitConfig {
            algorithm: Algorithm::LeakyBucket,
            max_requests_per_second: 10.0,
            burst_size: 2,
            ..Default::default()
        };
        let mut rl = RateLimiter::new(cfg);
        let key = RateLimitKey::Global;
        assert_eq!(rl.check(&key, 0), RateLimitResult::Allowed);
        assert_eq!(rl.check(&key, 0), RateLimitResult::Allowed);
        assert!(matches!(rl.check(&key, 0), RateLimitResult::RateLimited { .. }));
    }

    #[test]
    fn rate_limiter_per_user_limit() {
        let mut per_user = HashMap::new();
        per_user.insert("slow_user".to_owned(), 1.0);
        let cfg = RateLimitConfig {
            algorithm: Algorithm::TokenBucket,
            max_requests_per_second: 100.0,
            burst_size: 1,
            per_user_limits: per_user,
            ..Default::default()
        };
        let mut rl = RateLimiter::new(cfg);
        let key = RateLimitKey::User("slow_user".into());
        assert_eq!(rl.check(&key, 0), RateLimitResult::Allowed);
        // burst_size=1 so immediately limited
        assert!(matches!(rl.check(&key, 0), RateLimitResult::RateLimited { .. }));
    }

    #[test]
    fn rate_limiter_different_keys_independent() {
        let cfg = RateLimitConfig { burst_size: 1, ..Default::default() };
        let mut rl = RateLimiter::new(cfg);
        let k1 = RateLimitKey::User("alice".into());
        let k2 = RateLimitKey::User("bob".into());
        assert_eq!(rl.check(&k1, 0), RateLimitResult::Allowed);
        assert_eq!(rl.check(&k2, 0), RateLimitResult::Allowed);
        // Both should be limited (burst=1 each).
        assert!(matches!(rl.check(&k1, 0), RateLimitResult::RateLimited { .. }));
        assert!(matches!(rl.check(&k2, 0), RateLimitResult::RateLimited { .. }));
    }

    #[test]
    fn rate_limiter_metrics_integration() {
        let cfg = RateLimitConfig { burst_size: 1, ..Default::default() };
        let mut rl = RateLimiter::new(cfg);
        let key = RateLimitKey::Global;
        rl.check(&key, 0); // allowed
        rl.check(&key, 0); // denied
        assert_eq!(rl.metrics().allowed, 1);
        assert_eq!(rl.metrics().denied, 1);
    }

    #[test]
    fn rate_limiter_reset_metrics() {
        let cfg = RateLimitConfig::default();
        let mut rl = RateLimiter::new(cfg);
        rl.check(&RateLimitKey::Global, 0);
        rl.reset_metrics();
        assert_eq!(rl.metrics().total_requests(), 0);
    }

    #[test]
    fn rate_limiter_config_accessor() {
        let cfg = RateLimitConfig { max_requests_per_second: 42.0, ..Default::default() };
        let rl = RateLimiter::new(cfg);
        assert!((rl.config().max_requests_per_second - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn rate_limiter_ip_key() {
        let cfg = RateLimitConfig { burst_size: 2, ..Default::default() };
        let mut rl = RateLimiter::new(cfg);
        let key = RateLimitKey::Ip("10.0.0.1".into());
        assert_eq!(rl.check(&key, 0), RateLimitResult::Allowed);
        assert_eq!(rl.check(&key, 0), RateLimitResult::Allowed);
    }

    #[test]
    fn rate_limiter_api_key() {
        let cfg = RateLimitConfig { burst_size: 1, ..Default::default() };
        let mut rl = RateLimiter::new(cfg);
        let key = RateLimitKey::ApiKey("sk-test".into());
        assert_eq!(rl.check(&key, 0), RateLimitResult::Allowed);
        assert!(matches!(rl.check(&key, 0), RateLimitResult::RateLimited { .. }));
    }

    // ── Edge cases ────────────────────────────────────────────────────

    #[test]
    fn token_bucket_zero_rate() {
        let mut tb = TokenBucketLimiter::new(0.0, 1);
        assert_eq!(tb.try_acquire(0), RateLimitResult::Allowed);
        match tb.try_acquire(1000) {
            RateLimitResult::RateLimited { retry_after_ms } => {
                assert_eq!(retry_after_ms, u64::MAX);
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[test]
    fn leaky_bucket_zero_drain_rate() {
        let mut lb = LeakyBucketLimiter::new(0.0, 1);
        assert_eq!(lb.try_acquire(0), RateLimitResult::Allowed);
        match lb.try_acquire(1000) {
            RateLimitResult::RateLimited { retry_after_ms } => {
                assert_eq!(retry_after_ms, u64::MAX);
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[test]
    fn concurrency_limiter_zero_max() {
        let mut cl = ConcurrencyLimiter::new(0);
        match cl.try_acquire() {
            RateLimitResult::Queued { position } => {
                assert_eq!(position, 1);
            }
            other => panic!("expected Queued, got {other:?}"),
        }
    }

    #[test]
    fn sliding_window_single_slot() {
        let mut sw = SlidingWindowLimiter::new(1, 100);
        assert_eq!(sw.try_acquire(0), RateLimitResult::Allowed);
        assert!(matches!(sw.try_acquire(50), RateLimitResult::RateLimited { .. }));
        assert_eq!(sw.try_acquire(101), RateLimitResult::Allowed);
    }

    #[test]
    fn fixed_window_zero_time() {
        let mut fw = FixedWindowLimiter::new(2, 1000);
        assert_eq!(fw.try_acquire(0), RateLimitResult::Allowed);
        assert_eq!(fw.try_acquire(0), RateLimitResult::Allowed);
        assert!(matches!(fw.try_acquire(0), RateLimitResult::RateLimited { .. }));
    }

    #[test]
    fn quota_zero_tokens() {
        let mut qm = QuotaManager::new(0, 60_000);
        match qm.try_consume("alice", 1, 0) {
            RateLimitResult::RateLimited { .. } => {}
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[test]
    fn quota_exact_limit() {
        let mut qm = QuotaManager::new(100, 60_000);
        assert_eq!(qm.try_consume("alice", 100, 0), RateLimitResult::Allowed);
        assert_eq!(qm.remaining("alice"), 0);
        assert!(matches!(qm.try_consume("alice", 1, 0), RateLimitResult::RateLimited { .. }));
    }

    #[test]
    fn concurrency_release_when_empty() {
        let mut cl = ConcurrencyLimiter::new(2);
        // Release without any acquired — should not underflow.
        cl.release();
        assert_eq!(cl.in_flight(), 0);
    }

    #[test]
    fn rate_limit_result_partial_eq() {
        assert_eq!(RateLimitResult::Allowed, RateLimitResult::Allowed);
        assert_ne!(RateLimitResult::Allowed, RateLimitResult::Queued { position: 1 });
    }
}
