//! Reusable async circuit breaker primitives.

use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Circuit breaker states.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum CircuitBreakerState {
    /// Normal operation.
    Closed,
    /// Blocking requests.
    Open,
    /// Testing whether service has recovered.
    HalfOpen,
}

/// Configurable async circuit breaker.
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
    /// Create a new circuit breaker.
    #[must_use]
    pub fn new(failure_threshold: u64, timeout: Duration) -> Self {
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

    /// Return true if the circuit breaker currently allows execution.
    pub async fn can_execute(&self) -> bool {
        let state = self.state.read().await;

        match *state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open => {
                drop(state);
                self.check_timeout().await
            }
            CircuitBreakerState::HalfOpen => {
                self.success_count.load(Ordering::Relaxed) < self.half_open_max_requests
            }
        }
    }

    /// Record a successful execution.
    pub async fn record_success(&self) {
        self.success_count.fetch_add(1, Ordering::Relaxed);

        let state = self.state.read().await;
        if matches!(*state, CircuitBreakerState::HalfOpen)
            && self.success_count.load(Ordering::Relaxed) >= self.half_open_max_requests
        {
            drop(state);
            *self.state.write().await = CircuitBreakerState::Closed;
            self.failure_count.store(0, Ordering::Relaxed);
            self.success_count.store(0, Ordering::Relaxed);
            info!("Circuit breaker closed - service recovered");
        }
    }

    /// Record a failed execution.
    pub async fn record_failure(&self) {
        let failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;

        if failures >= self.failure_threshold {
            let mut state = self.state.write().await;
            if !matches!(*state, CircuitBreakerState::Open) {
                *state = CircuitBreakerState::Open;
                drop(state);
                *self.last_failure_time.write().await = Some(Instant::now());
                warn!(failures = failures, "Circuit breaker opened - too many failures");
            }
        }
    }

    /// Get the current state.
    pub async fn get_state(&self) -> CircuitBreakerState {
        self.state.read().await.clone()
    }

    async fn check_timeout(&self) -> bool {
        let last_failure = self.last_failure_time.read().await;
        if let Some(last_failure_time) = *last_failure {
            if last_failure_time.elapsed() >= self.timeout {
                drop(last_failure);
                *self.state.write().await = CircuitBreakerState::HalfOpen;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn opens_after_failures_and_transitions_half_open_after_timeout() {
        let breaker = CircuitBreaker::new(2, Duration::from_millis(5));
        assert!(breaker.can_execute().await);

        breaker.record_failure().await;
        assert_eq!(breaker.get_state().await, CircuitBreakerState::Closed);

        breaker.record_failure().await;
        assert_eq!(breaker.get_state().await, CircuitBreakerState::Open);
        assert!(!breaker.can_execute().await);

        tokio::time::sleep(Duration::from_millis(6)).await;
        assert!(breaker.can_execute().await);
        assert_eq!(breaker.get_state().await, CircuitBreakerState::HalfOpen);
    }

    #[tokio::test]
    async fn closes_after_half_open_success_budget() {
        let breaker = CircuitBreaker::new(1, Duration::from_millis(0));

        breaker.record_failure().await;
        assert_eq!(breaker.get_state().await, CircuitBreakerState::Open);

        assert!(breaker.can_execute().await);
        assert_eq!(breaker.get_state().await, CircuitBreakerState::HalfOpen);

        breaker.record_success().await;
        breaker.record_success().await;
        assert_eq!(breaker.get_state().await, CircuitBreakerState::HalfOpen);

        breaker.record_success().await;
        assert_eq!(breaker.get_state().await, CircuitBreakerState::Closed);
    }
}
