//! Concurrency management with backpressure and rate limiting

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock, Semaphore};
use tracing::{debug, info, warn};

/// Concurrency management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrencyConfig {
    pub max_concurrent_requests: usize,
    pub max_requests_per_second: u64,
    pub max_requests_per_minute: u64,
    pub rate_limit_window: Duration,
    pub backpressure_threshold: f64,
    pub circuit_breaker_enabled: bool,
    pub circuit_breaker_failure_threshold: u64,
    pub circuit_breaker_timeout: Duration,
    pub per_ip_rate_limit: Option<u64>,
    pub global_rate_limit: Option<u64>,
}

impl Default for ConcurrencyConfig {
    fn default() -> Self {
        Self {
            max_concurrent_requests: 100,
            max_requests_per_second: 50,
            max_requests_per_minute: 1000,
            rate_limit_window: Duration::from_secs(60),
            backpressure_threshold: 0.8, // 80% of max concurrent requests
            circuit_breaker_enabled: true,
            circuit_breaker_failure_threshold: 10,
            circuit_breaker_timeout: Duration::from_secs(30),
            per_ip_rate_limit: Some(10),  // 10 requests per second per IP
            global_rate_limit: Some(100), // 100 requests per second globally
        }
    }
}

/// Rate limiting bucket for token bucket algorithm
#[derive(Debug)]
struct RateLimitBucket {
    tokens: AtomicU64,
    last_refill: Mutex<Instant>,
    capacity: u64,
    refill_rate: u64, // tokens per second
}

impl RateLimitBucket {
    fn new(capacity: u64, refill_rate: u64) -> Self {
        Self {
            tokens: AtomicU64::new(capacity),
            last_refill: Mutex::new(Instant::now()),
            capacity,
            refill_rate,
        }
    }

    async fn try_consume(&self, tokens: u64) -> bool {
        self.refill().await;

        let current_tokens = self.tokens.load(Ordering::Relaxed);
        if current_tokens >= tokens {
            self.tokens.fetch_sub(tokens, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    async fn refill(&self) {
        let mut last_refill = self.last_refill.lock().await;
        let now = Instant::now();
        let elapsed = now.duration_since(*last_refill);

        if elapsed.as_secs() > 0 {
            let tokens_to_add = elapsed.as_secs() * self.refill_rate;
            let current_tokens = self.tokens.load(Ordering::Relaxed);
            let new_tokens = (current_tokens + tokens_to_add).min(self.capacity);

            self.tokens.store(new_tokens, Ordering::Relaxed);
            *last_refill = now;
        }
    }
}

/// Circuit breaker states
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum CircuitBreakerState {
    Closed,   // Normal operation
    Open,     // Blocking requests
    HalfOpen, // Testing if service recovered
}

/// Circuit breaker for fault tolerance
#[derive(Debug)]
pub struct CircuitBreaker {
    state: RwLock<CircuitBreakerState>,
    failure_count: AtomicU64,
    success_count: AtomicU64,
    last_failure_time: RwLock<Option<Instant>>,
    failure_threshold: u64,
    timeout: Duration,
    half_open_max_requests: u64,
}

impl CircuitBreaker {
    fn new(failure_threshold: u64, timeout: Duration) -> Self {
        Self {
            state: RwLock::new(CircuitBreakerState::Closed),
            failure_count: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            last_failure_time: RwLock::new(None),
            failure_threshold,
            timeout,
            half_open_max_requests: 3,
        }
    }

    async fn can_execute(&self) -> bool {
        let state = self.state.read().await;

        match *state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open => {
                drop(state);
                self.check_timeout().await
            }
            CircuitBreakerState::HalfOpen => {
                // Allow limited requests in half-open state
                self.success_count.load(Ordering::Relaxed) < self.half_open_max_requests
            }
        }
    }

    async fn record_success(&self) {
        self.success_count.fetch_add(1, Ordering::Relaxed);

        let state = self.state.read().await;
        if matches!(*state, CircuitBreakerState::HalfOpen)
            && self.success_count.load(Ordering::Relaxed) >= self.half_open_max_requests
        {
            drop(state);
            let mut state = self.state.write().await;
            *state = CircuitBreakerState::Closed;
            self.failure_count.store(0, Ordering::Relaxed);
            self.success_count.store(0, Ordering::Relaxed);
            info!("Circuit breaker closed - service recovered");
        }
    }

    async fn record_failure(&self) {
        let failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;

        if failures >= self.failure_threshold {
            let mut state = self.state.write().await;
            if !matches!(*state, CircuitBreakerState::Open) {
                *state = CircuitBreakerState::Open;
                let mut last_failure = self.last_failure_time.write().await;
                *last_failure = Some(Instant::now());
                warn!(failures = failures, "Circuit breaker opened - too many failures");
            }
        }
    }

    async fn check_timeout(&self) -> bool {
        let last_failure = self.last_failure_time.read().await;
        if let Some(last_failure_time) = *last_failure {
            if last_failure_time.elapsed() >= self.timeout {
                drop(last_failure);
                let mut state = self.state.write().await;
                *state = CircuitBreakerState::HalfOpen;
                self.success_count.store(0, Ordering::Relaxed);
                info!("Circuit breaker half-open - testing service recovery");
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    async fn get_state(&self) -> CircuitBreakerState {
        self.state.read().await.clone()
    }
}

/// Request metadata for tracking
#[derive(Debug, Clone)]
pub struct RequestMetadata {
    pub id: String,
    pub client_ip: IpAddr,
    pub user_agent: Option<String>,
    pub start_time: Instant,
    pub priority: crate::batch_engine::RequestPriority,
}

/// Concurrency manager with backpressure and rate limiting
pub struct ConcurrencyManager {
    config: ConcurrencyConfig,
    request_semaphore: Arc<Semaphore>,
    global_rate_limiter: Arc<RateLimitBucket>,
    per_ip_rate_limiters: Arc<RwLock<HashMap<IpAddr, Arc<RateLimitBucket>>>>,
    circuit_breaker: Arc<CircuitBreaker>,
    active_requests: Arc<RwLock<HashMap<String, RequestMetadata>>>,
    request_counter: AtomicU64,
    rejected_requests: AtomicU64,
    backpressure_active: AtomicU64,
}

impl ConcurrencyManager {
    /// Create a new concurrency manager
    pub fn new(config: ConcurrencyConfig) -> Self {
        let global_rate_limiter = Arc::new(RateLimitBucket::new(
            config.global_rate_limit.unwrap_or(1000),
            config.global_rate_limit.unwrap_or(100),
        ));

        let circuit_breaker = if config.circuit_breaker_enabled {
            Arc::new(CircuitBreaker::new(
                config.circuit_breaker_failure_threshold,
                config.circuit_breaker_timeout,
            ))
        } else {
            Arc::new(CircuitBreaker::new(u64::MAX, Duration::from_secs(0)))
        };

        Self {
            request_semaphore: Arc::new(Semaphore::new(config.max_concurrent_requests)),
            config,
            global_rate_limiter,
            per_ip_rate_limiters: Arc::new(RwLock::new(HashMap::new())),
            circuit_breaker,
            active_requests: Arc::new(RwLock::new(HashMap::new())),
            request_counter: AtomicU64::new(0),
            rejected_requests: AtomicU64::new(0),
            backpressure_active: AtomicU64::new(0),
        }
    }

    /// Check if request should be admitted (rate limiting + circuit breaker)
    pub async fn should_admit_request(
        &self,
        metadata: &RequestMetadata,
    ) -> Result<RequestAdmission> {
        // Check circuit breaker
        if !self.circuit_breaker.can_execute().await {
            self.rejected_requests.fetch_add(1, Ordering::Relaxed);
            return Ok(RequestAdmission::Rejected {
                reason: "Circuit breaker open".to_string(),
                retry_after: Some(self.config.circuit_breaker_timeout),
            });
        }

        // Check global rate limit
        if let Some(_global_limit) = self.config.global_rate_limit
            && !self.global_rate_limiter.try_consume(1).await
        {
            self.rejected_requests.fetch_add(1, Ordering::Relaxed);
            return Ok(RequestAdmission::Rejected {
                reason: "Global rate limit exceeded".to_string(),
                retry_after: Some(Duration::from_secs(1)),
            });
        }

        // Check per-IP rate limit
        if let Some(per_ip_limit) = self.config.per_ip_rate_limit {
            let ip_limiter = self.get_or_create_ip_limiter(metadata.client_ip, per_ip_limit).await;
            if !ip_limiter.try_consume(1).await {
                self.rejected_requests.fetch_add(1, Ordering::Relaxed);
                return Ok(RequestAdmission::Rejected {
                    reason: format!("Rate limit exceeded for IP {}", metadata.client_ip),
                    retry_after: Some(Duration::from_secs(1)),
                });
            }
        }

        Ok(RequestAdmission::Admitted)
    }

    /// Acquire request slot with backpressure
    pub async fn acquire_request_slot(&self, metadata: RequestMetadata) -> Result<RequestSlot<'_>> {
        // Check admission first
        match self.should_admit_request(&metadata).await? {
            RequestAdmission::Rejected { reason, retry_after } => {
                return Err(anyhow::anyhow!(
                    "Request rejected: {}. Retry after: {:?}",
                    reason,
                    retry_after
                ));
            }
            RequestAdmission::Admitted => {}
        }

        // Check backpressure
        let current_load = self.get_current_load().await;
        if current_load > self.config.backpressure_threshold {
            self.backpressure_active.fetch_add(1, Ordering::Relaxed);
            warn!(
                current_load = current_load,
                threshold = self.config.backpressure_threshold,
                "Backpressure activated"
            );

            // Apply backpressure delay based on load
            let delay_ms = ((current_load - self.config.backpressure_threshold) * 1000.0) as u64;
            tokio::time::sleep(Duration::from_millis(delay_ms.min(5000))).await;
        }

        // Try to acquire semaphore permit
        let permit = self.request_semaphore.acquire().await?;

        // Register active request
        {
            let mut active = self.active_requests.write().await;
            active.insert(metadata.id.clone(), metadata.clone());
        }

        self.request_counter.fetch_add(1, Ordering::Relaxed);

        debug!(
            request_id = %metadata.id,
            client_ip = %metadata.client_ip,
            "Request slot acquired"
        );

        Ok(RequestSlot { _permit: permit, metadata, manager: self.clone() })
    }

    /// Get current system load (0.0 to 1.0)
    async fn get_current_load(&self) -> f64 {
        let available_permits = self.request_semaphore.available_permits();
        let max_permits = self.config.max_concurrent_requests;
        let used_permits = max_permits - available_permits;

        used_permits as f64 / max_permits as f64
    }

    /// Get or create rate limiter for IP
    async fn get_or_create_ip_limiter(&self, ip: IpAddr, limit: u64) -> Arc<RateLimitBucket> {
        {
            let limiters = self.per_ip_rate_limiters.read().await;
            if let Some(limiter) = limiters.get(&ip) {
                return Arc::clone(limiter);
            }
        }

        let new_limiter = Arc::new(RateLimitBucket::new(limit, limit));
        {
            let mut limiters = self.per_ip_rate_limiters.write().await;
            limiters.insert(ip, Arc::clone(&new_limiter));
        }

        new_limiter
    }

    /// Record successful request completion
    pub async fn record_success(&self, request_id: &str) {
        self.circuit_breaker.record_success().await;

        // Remove from active requests
        {
            let mut active = self.active_requests.write().await;
            active.remove(request_id);
        }

        debug!(request_id = %request_id, "Request completed successfully");
    }

    /// Record failed request
    pub async fn record_failure(&self, request_id: &str, _error: &str) {
        self.circuit_breaker.record_failure().await;

        // Remove from active requests
        {
            let mut active = self.active_requests.write().await;
            active.remove(request_id);
        }

        warn!(request_id = %request_id, "Request failed");
    }

    /// Get concurrency statistics
    pub async fn get_stats(&self) -> ConcurrencyStats {
        let active_requests = {
            let active = self.active_requests.read().await;
            active.len()
        };

        let per_ip_limiter_count = {
            let limiters = self.per_ip_rate_limiters.read().await;
            limiters.len()
        };

        let current_load = self.get_current_load().await;
        let circuit_breaker_state = self.circuit_breaker.get_state().await;

        ConcurrencyStats {
            active_requests,
            max_concurrent_requests: self.config.max_concurrent_requests,
            current_load,
            total_requests: self.request_counter.load(Ordering::Relaxed),
            rejected_requests: self.rejected_requests.load(Ordering::Relaxed),
            backpressure_activations: self.backpressure_active.load(Ordering::Relaxed),
            circuit_breaker_state: format!("{:?}", circuit_breaker_state),
            available_permits: self.request_semaphore.available_permits(),
            per_ip_limiter_count,
        }
    }

    /// Get health status
    pub async fn get_health(&self) -> ConcurrencyHealth {
        let stats = self.get_stats().await;
        let circuit_breaker_state = self.circuit_breaker.get_state().await;

        let healthy =
            stats.current_load < 0.9 && circuit_breaker_state != CircuitBreakerState::Open;

        let mut issues = Vec::new();
        if stats.current_load > 0.9 {
            issues.push("High system load".to_string());
        }
        if circuit_breaker_state == CircuitBreakerState::Open {
            issues.push("Circuit breaker open".to_string());
        }
        if stats.backpressure_activations > 0 {
            issues.push("Backpressure active".to_string());
        }

        ConcurrencyHealth {
            healthy,
            current_load: stats.current_load,
            circuit_breaker_state,
            issues,
        }
    }

    /// Cleanup old rate limiters (should be called periodically)
    pub async fn cleanup_rate_limiters(&self, max_idle: Duration) {
        let mut limiters = self.per_ip_rate_limiters.write().await;
        let now = Instant::now();
        let mut keys_to_remove = Vec::new();

        for (ip, bucket) in limiters.iter() {
            // Check if bucket is idle
            // We use try_lock to avoid blocking if the bucket is currently in use
            if let Ok(last_refill) = bucket.last_refill.try_lock() {
                if now.duration_since(*last_refill) > max_idle {
                    keys_to_remove.push(*ip);
                }
            }
        }

        if !keys_to_remove.is_empty() {
            let count = keys_to_remove.len();
            for ip in keys_to_remove {
                limiters.remove(&ip);
            }
            debug!(removed = count, "Cleaned up stale rate limiters");
        }
    }
}

impl Clone for ConcurrencyManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            request_semaphore: Arc::clone(&self.request_semaphore),
            global_rate_limiter: Arc::clone(&self.global_rate_limiter),
            per_ip_rate_limiters: Arc::clone(&self.per_ip_rate_limiters),
            circuit_breaker: Arc::clone(&self.circuit_breaker),
            active_requests: Arc::clone(&self.active_requests),
            request_counter: AtomicU64::new(self.request_counter.load(Ordering::Relaxed)),
            rejected_requests: AtomicU64::new(self.rejected_requests.load(Ordering::Relaxed)),
            backpressure_active: AtomicU64::new(self.backpressure_active.load(Ordering::Relaxed)),
        }
    }
}

/// Request admission result
#[derive(Debug)]
pub enum RequestAdmission {
    Admitted,
    Rejected { reason: String, retry_after: Option<Duration> },
}

/// Request slot with automatic cleanup
pub struct RequestSlot<'a> {
    _permit: tokio::sync::SemaphorePermit<'a>,
    metadata: RequestMetadata,
    manager: ConcurrencyManager,
}

impl<'a> RequestSlot<'a> {
    pub fn metadata(&self) -> &RequestMetadata {
        &self.metadata
    }

    /// Mark request as successful
    pub async fn mark_success(self) {
        self.manager.record_success(&self.metadata.id).await;
    }

    /// Mark request as failed
    pub async fn mark_failure(self, error: &str) {
        self.manager.record_failure(&self.metadata.id, error).await;
    }
}

/// Concurrency statistics
#[derive(Debug, Clone, Serialize)]
pub struct ConcurrencyStats {
    pub active_requests: usize,
    pub max_concurrent_requests: usize,
    pub current_load: f64,
    pub total_requests: u64,
    pub rejected_requests: u64,
    pub backpressure_activations: u64,
    pub circuit_breaker_state: String,
    pub available_permits: usize,
    pub per_ip_limiter_count: usize,
}

/// Concurrency health status
#[derive(Debug, Clone, Serialize)]
pub struct ConcurrencyHealth {
    pub healthy: bool,
    pub current_load: f64,
    pub circuit_breaker_state: CircuitBreakerState,
    pub issues: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_rate_limiter_cleanup() {
        // Create configuration with per-IP rate limiting enabled
        let mut config = ConcurrencyConfig::default();
        config.per_ip_rate_limit = Some(10);

        let manager = ConcurrencyManager::new(config);

        // Initial state should be empty (but per_ip_limiter_count isn't exposed yet via get_stats for newly created manager until acquire is called)
        let stats = manager.get_stats().await;
        assert_eq!(stats.per_ip_limiter_count, 0);

        // Create a request metadata for a specific IP
        let metadata = RequestMetadata {
            id: "test-req".to_string(),
            client_ip: "127.0.0.1".parse().unwrap(),
            user_agent: None,
            start_time: Instant::now(),
            priority: crate::batch_engine::RequestPriority::Normal,
        };

        // Acquire a slot, which should create a rate limiter for the IP
        let _slot = manager.acquire_request_slot(metadata.clone()).await.unwrap();

        // Check that rate limiter was created
        let stats = manager.get_stats().await;
        assert_eq!(stats.per_ip_limiter_count, 1);

        // Cleanup with very long duration - should NOT remove it (it was just used)
        manager.cleanup_rate_limiters(Duration::from_secs(3600)).await;
        let stats = manager.get_stats().await;
        assert_eq!(stats.per_ip_limiter_count, 1);

        // Wait a tiny bit to ensure time advances
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Cleanup with very short duration - SHOULD remove it
        // The last_refill is updated when acquire_request_slot calls try_consume -> refill
        // So it's effectively "now"
        // But since we slept for 10ms, the last_refill (from acquire) is > 1ns old.
        manager.cleanup_rate_limiters(Duration::from_nanos(1)).await;

        let stats = manager.get_stats().await;
        assert_eq!(stats.per_ip_limiter_count, 0);
    }
}
