//! Edge-case tests for concurrency management: ConcurrencyConfig, CircuitBreakerState,
//! RequestMetadata, ConcurrencyManager, ConcurrencyStats, ConcurrencyHealth, and
//! request admission/rate-limiting behaviors.

use bitnet_server::batch_engine::RequestPriority;
use bitnet_server::concurrency::{
    CircuitBreakerState, ConcurrencyConfig, ConcurrencyHealth, ConcurrencyManager,
    ConcurrencyStats, RequestAdmission, RequestMetadata,
};
use std::net::IpAddr;
use std::time::{Duration, Instant};

// ─── ConcurrencyConfig ─────────────────────────────────────────────

#[test]
fn concurrency_config_defaults() {
    let config = ConcurrencyConfig::default();
    assert_eq!(config.max_concurrent_requests, 100);
    assert_eq!(config.max_requests_per_second, 50);
    assert_eq!(config.max_requests_per_minute, 1000);
    assert_eq!(config.rate_limit_window, Duration::from_secs(60));
    assert_eq!(config.backpressure_threshold, 0.8);
    assert!(config.circuit_breaker_enabled);
    assert_eq!(config.circuit_breaker_failure_threshold, 10);
    assert_eq!(config.circuit_breaker_timeout, Duration::from_secs(30));
    assert_eq!(config.per_ip_rate_limit, Some(10));
    assert_eq!(config.global_rate_limit, Some(100));
}

#[test]
fn concurrency_config_clone() {
    let config = ConcurrencyConfig::default();
    let cloned = config.clone();
    assert_eq!(cloned.max_concurrent_requests, 100);
    assert_eq!(cloned.per_ip_rate_limit, Some(10));
}

#[test]
fn concurrency_config_debug() {
    let config = ConcurrencyConfig::default();
    let debug = format!("{:?}", config);
    assert!(debug.contains("ConcurrencyConfig"));
    assert!(debug.contains("max_concurrent_requests"));
}

#[test]
fn concurrency_config_serde_roundtrip() {
    let config = ConcurrencyConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    let deser: ConcurrencyConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(deser.max_concurrent_requests, 100);
    assert_eq!(deser.backpressure_threshold, 0.8);
}

#[test]
fn concurrency_config_custom_values() {
    let config = ConcurrencyConfig {
        max_concurrent_requests: 10,
        max_requests_per_second: 5,
        max_requests_per_minute: 100,
        rate_limit_window: Duration::from_secs(120),
        backpressure_threshold: 0.5,
        circuit_breaker_enabled: false,
        circuit_breaker_failure_threshold: 3,
        circuit_breaker_timeout: Duration::from_secs(10),
        per_ip_rate_limit: None,
        global_rate_limit: None,
    };
    assert_eq!(config.max_concurrent_requests, 10);
    assert!(!config.circuit_breaker_enabled);
    assert!(config.per_ip_rate_limit.is_none());
    assert!(config.global_rate_limit.is_none());
}

#[test]
fn concurrency_config_zero_limits() {
    let config = ConcurrencyConfig {
        max_concurrent_requests: 0,
        max_requests_per_second: 0,
        max_requests_per_minute: 0,
        rate_limit_window: Duration::from_secs(0),
        backpressure_threshold: 0.0,
        circuit_breaker_enabled: false,
        circuit_breaker_failure_threshold: 0,
        circuit_breaker_timeout: Duration::from_secs(0),
        per_ip_rate_limit: Some(0),
        global_rate_limit: Some(0),
    };
    assert_eq!(config.max_concurrent_requests, 0);
    assert_eq!(config.per_ip_rate_limit, Some(0));
}

// ─── CircuitBreakerState ────────────────────────────────────────────

#[test]
fn circuit_breaker_state_debug() {
    assert!(format!("{:?}", CircuitBreakerState::Closed).contains("Closed"));
    assert!(format!("{:?}", CircuitBreakerState::Open).contains("Open"));
    assert!(format!("{:?}", CircuitBreakerState::HalfOpen).contains("HalfOpen"));
}

#[test]
fn circuit_breaker_state_clone() {
    let state = CircuitBreakerState::HalfOpen;
    let cloned = state.clone();
    assert_eq!(cloned, CircuitBreakerState::HalfOpen);
}

#[test]
fn circuit_breaker_state_eq() {
    assert_eq!(CircuitBreakerState::Closed, CircuitBreakerState::Closed);
    assert_ne!(CircuitBreakerState::Closed, CircuitBreakerState::Open);
    assert_ne!(CircuitBreakerState::Open, CircuitBreakerState::HalfOpen);
}

#[test]
fn circuit_breaker_state_serialize() {
    let json = serde_json::to_string(&CircuitBreakerState::Closed).unwrap();
    assert!(json.contains("Closed"));
    let json2 = serde_json::to_string(&CircuitBreakerState::Open).unwrap();
    assert!(json2.contains("Open"));
}

// ─── RequestMetadata ────────────────────────────────────────────────

fn make_metadata(id: &str) -> RequestMetadata {
    RequestMetadata {
        id: id.to_string(),
        client_ip: "127.0.0.1".parse().unwrap(),
        user_agent: None,
        start_time: Instant::now(),
        priority: RequestPriority::Normal,
    }
}

fn make_metadata_ip(id: &str, ip: &str) -> RequestMetadata {
    RequestMetadata {
        id: id.to_string(),
        client_ip: ip.parse().unwrap(),
        user_agent: None,
        start_time: Instant::now(),
        priority: RequestPriority::Normal,
    }
}

#[test]
fn request_metadata_debug() {
    let meta = make_metadata("test-1");
    let debug = format!("{:?}", meta);
    assert!(debug.contains("test-1"));
}

#[test]
fn request_metadata_clone() {
    let meta = make_metadata("test-2");
    let cloned = meta.clone();
    assert_eq!(cloned.id, "test-2");
    assert_eq!(cloned.client_ip, meta.client_ip);
}

#[test]
fn request_metadata_with_user_agent() {
    let meta = RequestMetadata {
        id: "req-1".to_string(),
        client_ip: "10.0.0.1".parse().unwrap(),
        user_agent: Some("test-agent/1.0".to_string()),
        start_time: Instant::now(),
        priority: RequestPriority::High,
    };
    assert_eq!(meta.user_agent.as_deref(), Some("test-agent/1.0"));
}

#[test]
fn request_metadata_ipv6() {
    let meta = RequestMetadata {
        id: "req-ipv6".to_string(),
        client_ip: "::1".parse().unwrap(),
        user_agent: None,
        start_time: Instant::now(),
        priority: RequestPriority::Low,
    };
    assert!(meta.client_ip.is_loopback());
}

#[test]
fn request_metadata_all_priorities() {
    let priorities = [
        RequestPriority::Low,
        RequestPriority::Normal,
        RequestPriority::High,
        RequestPriority::Critical,
    ];
    for p in priorities {
        let meta = RequestMetadata {
            id: "test".to_string(),
            client_ip: "127.0.0.1".parse().unwrap(),
            user_agent: None,
            start_time: Instant::now(),
            priority: p,
        };
        let _ = format!("{:?}", meta);
    }
}

// ─── ConcurrencyManager ────────────────────────────────────────────

#[tokio::test]
async fn concurrency_manager_initial_stats() {
    let manager = ConcurrencyManager::new(ConcurrencyConfig::default());
    let stats = manager.get_stats().await;
    assert_eq!(stats.active_requests, 0);
    assert_eq!(stats.max_concurrent_requests, 100);
    assert_eq!(stats.current_load, 0.0);
    assert_eq!(stats.total_requests, 0);
    assert_eq!(stats.rejected_requests, 0);
    assert_eq!(stats.backpressure_activations, 0);
    assert_eq!(stats.available_permits, 100);
    assert_eq!(stats.per_ip_limiter_count, 0);
}

#[tokio::test]
async fn concurrency_manager_initial_health() {
    let manager = ConcurrencyManager::new(ConcurrencyConfig::default());
    let health = manager.get_health().await;
    assert!(health.healthy);
    assert_eq!(health.current_load, 0.0);
    assert_eq!(health.circuit_breaker_state, CircuitBreakerState::Closed);
    assert!(health.issues.is_empty());
}

#[tokio::test]
async fn concurrency_manager_should_admit_request() {
    let config = ConcurrencyConfig {
        per_ip_rate_limit: None,
        global_rate_limit: None,
        ..ConcurrencyConfig::default()
    };
    let manager = ConcurrencyManager::new(config);
    let meta = make_metadata("req-1");
    let admission = manager.should_admit_request(&meta).await.unwrap();
    assert!(matches!(admission, RequestAdmission::Admitted));
}

#[tokio::test]
async fn concurrency_manager_acquire_and_release() {
    let config = ConcurrencyConfig {
        max_concurrent_requests: 2,
        per_ip_rate_limit: None,
        global_rate_limit: None,
        backpressure_threshold: 1.0, // Disable backpressure
        ..ConcurrencyConfig::default()
    };
    let manager = ConcurrencyManager::new(config);
    let meta = make_metadata("req-acquire");

    let slot = manager.acquire_request_slot(meta).await.unwrap();
    let stats = manager.get_stats().await;
    assert_eq!(stats.active_requests, 1);
    assert_eq!(stats.total_requests, 1);

    slot.mark_success().await;
    let stats = manager.get_stats().await;
    assert_eq!(stats.active_requests, 0);
}

#[tokio::test]
async fn concurrency_manager_record_success() {
    let config = ConcurrencyConfig {
        per_ip_rate_limit: None,
        global_rate_limit: None,
        ..ConcurrencyConfig::default()
    };
    let manager = ConcurrencyManager::new(config);
    manager.record_success("req-1").await;
    // Should not panic, removes from active (noop if not present)
}

#[tokio::test]
async fn concurrency_manager_record_failure() {
    let config = ConcurrencyConfig {
        per_ip_rate_limit: None,
        global_rate_limit: None,
        ..ConcurrencyConfig::default()
    };
    let manager = ConcurrencyManager::new(config);
    manager.record_failure("req-1", "test error").await;
    // Should not panic
}

#[tokio::test]
async fn concurrency_manager_slot_mark_failure() {
    let config = ConcurrencyConfig {
        max_concurrent_requests: 10,
        per_ip_rate_limit: None,
        global_rate_limit: None,
        backpressure_threshold: 1.0,
        ..ConcurrencyConfig::default()
    };
    let manager = ConcurrencyManager::new(config);
    let slot = manager.acquire_request_slot(make_metadata("fail-req")).await.unwrap();
    slot.mark_failure("something went wrong").await;
    let stats = manager.get_stats().await;
    assert_eq!(stats.active_requests, 0);
}

#[tokio::test]
async fn concurrency_manager_slot_metadata() {
    let config = ConcurrencyConfig {
        max_concurrent_requests: 10,
        per_ip_rate_limit: None,
        global_rate_limit: None,
        backpressure_threshold: 1.0,
        ..ConcurrencyConfig::default()
    };
    let manager = ConcurrencyManager::new(config);
    let slot = manager.acquire_request_slot(make_metadata("meta-req")).await.unwrap();
    assert_eq!(slot.metadata().id, "meta-req");
    assert_eq!(slot.metadata().client_ip, "127.0.0.1".parse::<IpAddr>().unwrap());
    slot.mark_success().await;
}

#[tokio::test]
async fn concurrency_manager_cleanup_rate_limiters_empty() {
    let manager = ConcurrencyManager::new(ConcurrencyConfig::default());
    // Should not panic on empty state
    manager.cleanup_rate_limiters(Duration::from_secs(1)).await;
    let stats = manager.get_stats().await;
    assert_eq!(stats.per_ip_limiter_count, 0);
}

#[tokio::test]
async fn concurrency_manager_clone_shares_state() {
    let config = ConcurrencyConfig {
        per_ip_rate_limit: None,
        global_rate_limit: None,
        ..ConcurrencyConfig::default()
    };
    let manager = ConcurrencyManager::new(config);
    let cloned = manager.clone();

    // Recording on original should affect cloned (shared Arc state)
    manager.record_success("shared-req").await;
    let stats = cloned.get_stats().await;
    // Stats counters are cloned (not shared via Arc) so values may differ
    // But the circuit breaker and active requests are shared
    assert_eq!(stats.active_requests, 0);
}

// ─── ConcurrencyStats ──────────────────────────────────────────────

#[tokio::test]
async fn concurrency_stats_serialize() {
    let manager = ConcurrencyManager::new(ConcurrencyConfig::default());
    let stats = manager.get_stats().await;
    let json = serde_json::to_string(&stats).unwrap();
    assert!(json.contains("active_requests"));
    assert!(json.contains("max_concurrent_requests"));
    assert!(json.contains("circuit_breaker_state"));
    assert!(json.contains("per_ip_limiter_count"));
}

#[tokio::test]
async fn concurrency_stats_debug() {
    let manager = ConcurrencyManager::new(ConcurrencyConfig::default());
    let stats = manager.get_stats().await;
    let debug = format!("{:?}", stats);
    assert!(debug.contains("ConcurrencyStats"));
}

#[tokio::test]
async fn concurrency_stats_clone() {
    let manager = ConcurrencyManager::new(ConcurrencyConfig::default());
    let stats = manager.get_stats().await;
    let cloned = stats.clone();
    assert_eq!(cloned.active_requests, stats.active_requests);
    assert_eq!(cloned.max_concurrent_requests, stats.max_concurrent_requests);
}

// ─── ConcurrencyHealth ─────────────────────────────────────────────

#[tokio::test]
async fn concurrency_health_serialize() {
    let manager = ConcurrencyManager::new(ConcurrencyConfig::default());
    let health = manager.get_health().await;
    let json = serde_json::to_string(&health).unwrap();
    assert!(json.contains("healthy"));
    assert!(json.contains("current_load"));
    assert!(json.contains("circuit_breaker_state"));
}

#[tokio::test]
async fn concurrency_health_debug() {
    let manager = ConcurrencyManager::new(ConcurrencyConfig::default());
    let health = manager.get_health().await;
    let debug = format!("{:?}", health);
    assert!(debug.contains("ConcurrencyHealth"));
}

#[tokio::test]
async fn concurrency_health_clone() {
    let manager = ConcurrencyManager::new(ConcurrencyConfig::default());
    let health = manager.get_health().await;
    let cloned = health.clone();
    assert_eq!(cloned.healthy, health.healthy);
    assert_eq!(cloned.circuit_breaker_state, health.circuit_breaker_state);
}

// ─── RequestAdmission ───────────────────────────────────────────────

#[test]
fn request_admission_debug() {
    let admitted = RequestAdmission::Admitted;
    assert!(format!("{:?}", admitted).contains("Admitted"));

    let rejected = RequestAdmission::Rejected {
        reason: "rate limited".to_string(),
        retry_after: Some(Duration::from_secs(1)),
    };
    assert!(format!("{:?}", rejected).contains("Rejected"));
    assert!(format!("{:?}", rejected).contains("rate limited"));
}

#[test]
fn request_admission_rejected_no_retry() {
    let rejected =
        RequestAdmission::Rejected { reason: "permanently banned".to_string(), retry_after: None };
    let debug = format!("{:?}", rejected);
    assert!(debug.contains("permanently banned"));
}

// ─── Circuit breaker behavior through ConcurrencyManager ────────────

#[tokio::test]
async fn circuit_breaker_opens_after_threshold_failures() {
    let config = ConcurrencyConfig {
        circuit_breaker_enabled: true,
        circuit_breaker_failure_threshold: 3,
        circuit_breaker_timeout: Duration::from_secs(60),
        per_ip_rate_limit: None,
        global_rate_limit: None,
        ..ConcurrencyConfig::default()
    };
    let manager = ConcurrencyManager::new(config);

    // Record failures up to threshold
    for i in 0..3 {
        manager.record_failure(&format!("fail-{}", i), "error").await;
    }

    // Health should reflect circuit breaker is open
    let health = manager.get_health().await;
    assert_eq!(health.circuit_breaker_state, CircuitBreakerState::Open);
    assert!(!health.healthy);
    assert!(health.issues.iter().any(|i| i.contains("Circuit breaker")));
}

#[tokio::test]
async fn circuit_breaker_rejects_when_open() {
    let config = ConcurrencyConfig {
        circuit_breaker_enabled: true,
        circuit_breaker_failure_threshold: 2,
        circuit_breaker_timeout: Duration::from_secs(60),
        per_ip_rate_limit: None,
        global_rate_limit: None,
        ..ConcurrencyConfig::default()
    };
    let manager = ConcurrencyManager::new(config);

    // Trip the circuit breaker
    manager.record_failure("f1", "err").await;
    manager.record_failure("f2", "err").await;

    // New requests should be rejected
    let meta = make_metadata("should-reject");
    let admission = manager.should_admit_request(&meta).await.unwrap();
    assert!(matches!(admission, RequestAdmission::Rejected { .. }));

    if let RequestAdmission::Rejected { reason, retry_after } = admission {
        assert!(reason.contains("Circuit breaker"));
        assert!(retry_after.is_some());
    }
}

#[tokio::test]
async fn circuit_breaker_disabled_allows_all() {
    let config = ConcurrencyConfig {
        circuit_breaker_enabled: false,
        per_ip_rate_limit: None,
        global_rate_limit: None,
        ..ConcurrencyConfig::default()
    };
    let manager = ConcurrencyManager::new(config);

    // Even with many failures, requests should be admitted
    for i in 0..20 {
        manager.record_failure(&format!("fail-{}", i), "error").await;
    }

    let meta = make_metadata("should-admit");
    let admission = manager.should_admit_request(&meta).await.unwrap();
    assert!(matches!(admission, RequestAdmission::Admitted));
}

// ─── Multiple IP tracking ───────────────────────────────────────────

#[tokio::test]
async fn concurrency_manager_tracks_multiple_ips() {
    let config = ConcurrencyConfig {
        max_concurrent_requests: 100,
        per_ip_rate_limit: Some(100), // High limit so we don't get rate limited
        global_rate_limit: None,
        backpressure_threshold: 1.0,
        ..ConcurrencyConfig::default()
    };
    let manager = ConcurrencyManager::new(config);

    // Send requests from different IPs
    let slot1 = manager.acquire_request_slot(make_metadata_ip("r1", "10.0.0.1")).await.unwrap();
    let slot2 = manager.acquire_request_slot(make_metadata_ip("r2", "10.0.0.2")).await.unwrap();
    let slot3 = manager.acquire_request_slot(make_metadata_ip("r3", "10.0.0.3")).await.unwrap();

    let stats = manager.get_stats().await;
    assert_eq!(stats.active_requests, 3);
    assert_eq!(stats.per_ip_limiter_count, 3);

    slot1.mark_success().await;
    slot2.mark_success().await;
    slot3.mark_success().await;

    let stats = manager.get_stats().await;
    assert_eq!(stats.active_requests, 0);
    // Rate limiters persist even after requests complete
    assert_eq!(stats.per_ip_limiter_count, 3);
}
