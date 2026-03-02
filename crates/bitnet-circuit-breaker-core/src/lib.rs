use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{info, warn};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u64,
    pub timeout: Duration,
    pub half_open_max_requests: u64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self { failure_threshold: 10, timeout: Duration::from_secs(30), half_open_max_requests: 3 }
    }
}

#[derive(Debug)]
pub struct CircuitBreaker {
    state: RwLock<CircuitBreakerState>,
    failure_count: AtomicU64,
    success_count: AtomicU64,
    last_failure_time: RwLock<Option<Instant>>,
    config: CircuitBreakerConfig,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: RwLock::new(CircuitBreakerState::Closed),
            failure_count: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            last_failure_time: RwLock::new(None),
            config,
        }
    }

    pub fn disabled() -> Self {
        Self::new(CircuitBreakerConfig {
            failure_threshold: u64::MAX,
            timeout: Duration::ZERO,
            half_open_max_requests: 1,
        })
    }

    pub async fn can_execute(&self) -> bool {
        let state = self.state.read().await;
        match *state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open => {
                drop(state);
                self.check_timeout().await
            }
            CircuitBreakerState::HalfOpen => {
                self.success_count.load(Ordering::Relaxed) < self.config.half_open_max_requests
            }
        }
    }

    pub async fn record_success(&self) {
        self.success_count.fetch_add(1, Ordering::Relaxed);

        let state = self.state.read().await;
        if matches!(*state, CircuitBreakerState::HalfOpen)
            && self.success_count.load(Ordering::Relaxed) >= self.config.half_open_max_requests
        {
            drop(state);
            let mut state = self.state.write().await;
            *state = CircuitBreakerState::Closed;
            self.failure_count.store(0, Ordering::Relaxed);
            self.success_count.store(0, Ordering::Relaxed);
            info!("Circuit breaker closed - service recovered");
        }
    }

    pub async fn record_failure(&self) {
        let failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;

        if failures >= self.config.failure_threshold {
            let mut state = self.state.write().await;
            if !matches!(*state, CircuitBreakerState::Open) {
                *state = CircuitBreakerState::Open;
                let mut last_failure = self.last_failure_time.write().await;
                *last_failure = Some(Instant::now());
                warn!(failures, "Circuit breaker opened - too many failures");
            }
        }
    }

    pub async fn state(&self) -> CircuitBreakerState {
        self.state.read().await.clone()
    }

    async fn check_timeout(&self) -> bool {
        let last_failure = self.last_failure_time.read().await;
        if let Some(last_failure_time) = *last_failure {
            if last_failure_time.elapsed() >= self.config.timeout {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn opens_after_failure_threshold() {
        let breaker = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 2,
            timeout: Duration::from_millis(20),
            half_open_max_requests: 1,
        });

        breaker.record_failure().await;
        assert_eq!(breaker.state().await, CircuitBreakerState::Closed);

        breaker.record_failure().await;
        assert_eq!(breaker.state().await, CircuitBreakerState::Open);
        assert!(!breaker.can_execute().await);
    }

    #[tokio::test]
    async fn transitions_to_half_open_after_timeout() {
        let breaker = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 1,
            timeout: Duration::from_millis(5),
            half_open_max_requests: 1,
        });

        breaker.record_failure().await;
        tokio::time::sleep(Duration::from_millis(10)).await;

        assert!(breaker.can_execute().await);
        assert_eq!(breaker.state().await, CircuitBreakerState::HalfOpen);

        breaker.record_success().await;
        assert_eq!(breaker.state().await, CircuitBreakerState::Closed);
    }
}
