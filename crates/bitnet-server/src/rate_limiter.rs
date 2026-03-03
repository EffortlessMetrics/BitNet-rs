//! Token bucket rate limiter.
//!
//! Rate limit API requests using a token bucket algorithm
//! with per-client tracking and configurable limits.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Rate limit configuration.
#[derive(Debug, Clone, Copy)]
pub struct RateLimitConfig {
    pub requests_per_second: f64,
    pub burst_size: usize,
    pub refill_interval: Duration,
}

impl RateLimitConfig {
    pub fn new(rps: f64, burst: usize) -> Self {
        let refill = Duration::from_secs_f64(1.0 / rps);
        Self { requests_per_second: rps, burst_size: burst, refill_interval: refill }
    }

    pub fn permissive() -> Self {
        Self::new(100.0, 200)
    }
    pub fn moderate() -> Self {
        Self::new(10.0, 20)
    }
    pub fn strict() -> Self {
        Self::new(1.0, 5)
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self::moderate()
    }
}

/// A token bucket for a single client.
#[derive(Debug, Clone)]
pub struct TokenBucket {
    tokens: f64,
    max_tokens: f64,
    refill_rate: f64,
    last_refill: Instant,
}

impl TokenBucket {
    pub fn new(config: &RateLimitConfig) -> Self {
        Self {
            tokens: config.burst_size as f64,
            max_tokens: config.burst_size as f64,
            refill_rate: config.requests_per_second,
            last_refill: Instant::now(),
        }
    }

    /// Refill tokens based on elapsed time.
    pub fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.max_tokens);
        self.last_refill = now;
    }

    /// Try to consume a token. Returns true if allowed.
    pub fn try_acquire(&mut self) -> bool {
        self.refill();
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Remaining tokens.
    pub fn available(&self) -> usize {
        self.tokens as usize
    }

    /// Time until next token available.
    pub fn retry_after(&self) -> Duration {
        if self.tokens >= 1.0 {
            return Duration::ZERO;
        }
        let needed = 1.0 - self.tokens;
        Duration::from_secs_f64(needed / self.refill_rate)
    }
}

/// Rate limit decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RateLimitResult {
    Allowed,
    Limited,
}

impl RateLimitResult {
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allowed)
    }
}

/// Per-client rate limiter.
#[derive(Debug)]
pub struct RateLimiter {
    config: RateLimitConfig,
    clients: HashMap<String, TokenBucket>,
    global_bucket: TokenBucket,
    enabled: bool,
    total_allowed: u64,
    total_limited: u64,
}

impl RateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        let global = TokenBucket::new(&config);
        Self {
            config,
            clients: HashMap::new(),
            global_bucket: global,
            enabled: true,
            total_allowed: 0,
            total_limited: 0,
        }
    }

    pub fn disabled() -> Self {
        let mut limiter = Self::new(RateLimitConfig::permissive());
        limiter.enabled = false;
        limiter
    }

    /// Check rate limit for a client.
    pub fn check(&mut self, client_id: &str) -> RateLimitResult {
        if !self.enabled {
            self.total_allowed += 1;
            return RateLimitResult::Allowed;
        }

        // Check global limit first
        if !self.global_bucket.try_acquire() {
            self.total_limited += 1;
            return RateLimitResult::Limited;
        }

        // Check per-client limit
        let bucket = self
            .clients
            .entry(client_id.to_string())
            .or_insert_with(|| TokenBucket::new(&self.config));

        if bucket.try_acquire() {
            self.total_allowed += 1;
            RateLimitResult::Allowed
        } else {
            self.total_limited += 1;
            RateLimitResult::Limited
        }
    }

    pub fn client_count(&self) -> usize {
        self.clients.len()
    }
    pub fn total_allowed(&self) -> u64 {
        self.total_allowed
    }
    pub fn total_limited(&self) -> u64 {
        self.total_limited
    }

    pub fn limit_rate(&self) -> f64 {
        let total = self.total_allowed + self.total_limited;
        if total == 0 {
            return 0.0;
        }
        self.total_limited as f64 / total as f64
    }

    /// Clean up inactive client buckets.
    pub fn cleanup_inactive(&mut self, max_age: Duration) {
        let now = Instant::now();
        self.clients.retain(|_, bucket| now.duration_since(bucket.last_refill) < max_age);
    }

    pub fn reset_stats(&mut self) {
        self.total_allowed = 0;
        self.total_limited = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_initial() {
        let config = RateLimitConfig::new(10.0, 5);
        let bucket = TokenBucket::new(&config);
        assert_eq!(bucket.available(), 5);
    }

    #[test]
    fn test_bucket_acquire() {
        let config = RateLimitConfig::new(10.0, 2);
        let mut bucket = TokenBucket::new(&config);
        assert!(bucket.try_acquire());
        assert!(bucket.try_acquire());
        assert!(!bucket.try_acquire());
    }

    #[test]
    fn test_limiter_allows() {
        let mut limiter = RateLimiter::new(RateLimitConfig::new(100.0, 10));
        assert!(limiter.check("client1").is_allowed());
    }

    #[test]
    fn test_limiter_limits() {
        let mut limiter = RateLimiter::new(RateLimitConfig::new(100.0, 2));
        limiter.check("c1");
        limiter.check("c1");
        // 3rd request should be limited (bucket exhausted)
        let result = limiter.check("c1");
        assert_eq!(result, RateLimitResult::Limited);
    }

    #[test]
    fn test_different_clients() {
        let mut limiter = RateLimiter::new(RateLimitConfig::new(100.0, 2));
        assert!(limiter.check("c1").is_allowed());
        assert!(limiter.check("c2").is_allowed());
        assert_eq!(limiter.client_count(), 2);
    }

    #[test]
    fn test_disabled() {
        let mut limiter = RateLimiter::disabled();
        for _ in 0..100 {
            assert!(limiter.check("c1").is_allowed());
        }
    }

    #[test]
    fn test_stats() {
        let mut limiter = RateLimiter::new(RateLimitConfig::new(100.0, 1));
        limiter.check("c1");
        limiter.check("c1");
        assert_eq!(limiter.total_allowed(), 1);
        assert_eq!(limiter.total_limited(), 1);
    }

    #[test]
    fn test_limit_rate() {
        let mut limiter = RateLimiter::new(RateLimitConfig::new(100.0, 1));
        limiter.check("c1");
        limiter.check("c1");
        assert!(limiter.limit_rate() > 0.0);
    }

    #[test]
    fn test_reset_stats() {
        let mut limiter = RateLimiter::new(RateLimitConfig::new(100.0, 1));
        limiter.check("c1");
        limiter.reset_stats();
        assert_eq!(limiter.total_allowed(), 0);
    }

    #[test]
    fn test_config_presets() {
        let p = RateLimitConfig::permissive();
        let m = RateLimitConfig::moderate();
        let s = RateLimitConfig::strict();
        assert!(p.requests_per_second > m.requests_per_second);
        assert!(m.requests_per_second > s.requests_per_second);
    }

    #[test]
    fn test_retry_after() {
        let config = RateLimitConfig::new(1.0, 1);
        let mut bucket = TokenBucket::new(&config);
        bucket.try_acquire();
        let retry = bucket.retry_after();
        assert!(retry > Duration::ZERO);
    }

    #[test]
    fn test_empty_limit_rate() {
        let limiter = RateLimiter::new(RateLimitConfig::moderate());
        assert_eq!(limiter.limit_rate(), 0.0);
    }
}
