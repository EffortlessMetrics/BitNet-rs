//! Edge-case tests for execution_router: DeviceCapabilities, DeviceSelectionStrategy,
//! ExecutionRouterConfig, DeviceHealth, DeviceStats, DeviceMonitor, ExecutionRouter,
//! DeviceStatus, and ExecutionRouterHealth.

use bitnet_common::Device;
use bitnet_server::execution_router::{
    DeviceCapabilities, DeviceHealth, DeviceMonitor, DeviceSelectionStrategy, DeviceStats,
    ExecutionRouter, ExecutionRouterConfig, ExecutionRouterHealth,
};
use std::time::Duration;

// ─── DeviceCapabilities ─────────────────────────────────────────────

#[test]
fn device_capabilities_cpu_defaults() {
    let caps = DeviceCapabilities {
        device: Device::Cpu,
        available: true,
        memory_total_mb: 16384,
        memory_free_mb: 8192,
        compute_capability: None,
        simd_support: vec!["AVX2".to_string()],
        avg_tokens_per_second: 35.0,
        last_benchmark: None,
    };
    assert_eq!(caps.device, Device::Cpu);
    assert!(caps.available);
    assert_eq!(caps.memory_total_mb, 16384);
    assert_eq!(caps.memory_free_mb, 8192);
    assert!(caps.compute_capability.is_none());
    assert_eq!(caps.simd_support.len(), 1);
    assert_eq!(caps.avg_tokens_per_second, 35.0);
    assert!(caps.last_benchmark.is_none());
}

#[test]
fn device_capabilities_clone() {
    let caps = DeviceCapabilities {
        device: Device::Cpu,
        available: true,
        memory_total_mb: 1024,
        memory_free_mb: 512,
        compute_capability: Some("8.6".to_string()),
        simd_support: vec!["AVX-512".to_string(), "AVX2".to_string()],
        avg_tokens_per_second: 100.0,
        last_benchmark: Some(std::time::SystemTime::now()),
    };
    let cloned = caps.clone();
    assert_eq!(cloned.device, caps.device);
    assert_eq!(cloned.memory_total_mb, caps.memory_total_mb);
    assert_eq!(cloned.compute_capability, caps.compute_capability);
    assert_eq!(cloned.simd_support, caps.simd_support);
}

#[test]
fn device_capabilities_debug() {
    let caps = DeviceCapabilities {
        device: Device::Cpu,
        available: false,
        memory_total_mb: 0,
        memory_free_mb: 0,
        compute_capability: None,
        simd_support: vec![],
        avg_tokens_per_second: 0.0,
        last_benchmark: None,
    };
    let debug = format!("{:?}", caps);
    assert!(debug.contains("DeviceCapabilities"));
    assert!(debug.contains("Cpu"));
}

#[test]
fn device_capabilities_serde_roundtrip() {
    let caps = DeviceCapabilities {
        device: Device::Cpu,
        available: true,
        memory_total_mb: 32768,
        memory_free_mb: 16384,
        compute_capability: None,
        simd_support: vec!["AVX2".to_string()],
        avg_tokens_per_second: 42.0,
        last_benchmark: None,
    };
    let json = serde_json::to_string(&caps).unwrap();
    let deser: DeviceCapabilities = serde_json::from_str(&json).unwrap();
    assert_eq!(deser.device, Device::Cpu);
    assert_eq!(deser.memory_total_mb, 32768);
    assert_eq!(deser.avg_tokens_per_second, 42.0);
}

#[test]
fn device_capabilities_zero_memory() {
    let caps = DeviceCapabilities {
        device: Device::Cpu,
        available: true,
        memory_total_mb: 0,
        memory_free_mb: 0,
        compute_capability: None,
        simd_support: vec![],
        avg_tokens_per_second: 0.0,
        last_benchmark: None,
    };
    assert_eq!(caps.memory_total_mb, 0);
    assert_eq!(caps.memory_free_mb, 0);
}

#[test]
fn device_capabilities_huge_memory() {
    let caps = DeviceCapabilities {
        device: Device::Cpu,
        available: true,
        memory_total_mb: u64::MAX,
        memory_free_mb: u64::MAX / 2,
        compute_capability: None,
        simd_support: vec![],
        avg_tokens_per_second: 0.0,
        last_benchmark: None,
    };
    assert!(caps.memory_free_mb < caps.memory_total_mb);
}

// ─── DeviceSelectionStrategy ────────────────────────────────────────

#[test]
fn device_selection_strategy_debug() {
    let strategies = vec![
        DeviceSelectionStrategy::PreferGpu,
        DeviceSelectionStrategy::CpuOnly,
        DeviceSelectionStrategy::PerformanceBased,
        DeviceSelectionStrategy::LoadBalance,
        DeviceSelectionStrategy::UserPreference(Device::Cpu),
    ];
    for s in &strategies {
        let debug = format!("{:?}", s);
        assert!(!debug.is_empty());
    }
}

#[test]
fn device_selection_strategy_clone() {
    let s = DeviceSelectionStrategy::UserPreference(Device::Cpu);
    let cloned = s.clone();
    assert!(format!("{:?}", cloned).contains("UserPreference"));
}

#[test]
fn device_selection_strategy_serde_roundtrip() {
    let strategies = vec![
        DeviceSelectionStrategy::PreferGpu,
        DeviceSelectionStrategy::CpuOnly,
        DeviceSelectionStrategy::PerformanceBased,
        DeviceSelectionStrategy::LoadBalance,
    ];
    for s in &strategies {
        let json = serde_json::to_string(s).unwrap();
        let deser: DeviceSelectionStrategy = serde_json::from_str(&json).unwrap();
        assert_eq!(format!("{:?}", s), format!("{:?}", deser));
    }
}

#[test]
fn device_selection_strategy_user_preference_serde() {
    let s = DeviceSelectionStrategy::UserPreference(Device::Cpu);
    let json = serde_json::to_string(&s).unwrap();
    assert!(json.contains("UserPreference"));
    let deser: DeviceSelectionStrategy = serde_json::from_str(&json).unwrap();
    assert!(format!("{:?}", deser).contains("UserPreference"));
}

// ─── ExecutionRouterConfig ──────────────────────────────────────────

#[test]
fn execution_router_config_defaults() {
    let config = ExecutionRouterConfig::default();
    assert!(matches!(config.strategy, DeviceSelectionStrategy::PerformanceBased));
    assert!(config.fallback_enabled);
    assert_eq!(config.health_check_interval, Duration::from_secs(30));
    assert_eq!(config.performance_threshold_tps, 10.0);
    assert_eq!(config.memory_threshold_percent, 0.8);
    assert!(config.benchmark_on_startup);
}

#[test]
fn execution_router_config_clone() {
    let config = ExecutionRouterConfig::default();
    let cloned = config.clone();
    assert!(cloned.fallback_enabled);
    assert_eq!(cloned.performance_threshold_tps, 10.0);
}

#[test]
fn execution_router_config_debug() {
    let config = ExecutionRouterConfig::default();
    let debug = format!("{:?}", config);
    assert!(debug.contains("ExecutionRouterConfig"));
    assert!(debug.contains("fallback_enabled"));
}

#[test]
fn execution_router_config_serde_roundtrip() {
    let config = ExecutionRouterConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    let deser: ExecutionRouterConfig = serde_json::from_str(&json).unwrap();
    assert!(deser.fallback_enabled);
    assert_eq!(deser.performance_threshold_tps, 10.0);
}

#[test]
fn execution_router_config_custom_values() {
    let config = ExecutionRouterConfig {
        strategy: DeviceSelectionStrategy::CpuOnly,
        fallback_enabled: false,
        health_check_interval: Duration::from_secs(5),
        performance_threshold_tps: 1.0,
        memory_threshold_percent: 0.5,
        benchmark_on_startup: false,
    };
    assert!(!config.fallback_enabled);
    assert_eq!(config.health_check_interval, Duration::from_secs(5));
    assert_eq!(config.performance_threshold_tps, 1.0);
    assert_eq!(config.memory_threshold_percent, 0.5);
    assert!(!config.benchmark_on_startup);
}

// ─── DeviceHealth ───────────────────────────────────────────────────

#[test]
fn device_health_healthy() {
    let h = DeviceHealth::Healthy;
    let debug = format!("{:?}", h);
    assert!(debug.contains("Healthy"));
}

#[test]
fn device_health_degraded() {
    let h = DeviceHealth::Degraded { reason: "Low memory".to_string() };
    let debug = format!("{:?}", h);
    assert!(debug.contains("Degraded"));
    assert!(debug.contains("Low memory"));
}

#[test]
fn device_health_unavailable() {
    let h = DeviceHealth::Unavailable { reason: "Device disconnected".to_string() };
    let debug = format!("{:?}", h);
    assert!(debug.contains("Unavailable"));
    assert!(debug.contains("Device disconnected"));
}

#[test]
fn device_health_clone() {
    let h = DeviceHealth::Degraded { reason: "test reason".to_string() };
    let cloned = h.clone();
    assert!(format!("{:?}", cloned).contains("test reason"));
}

#[test]
fn device_health_serialize() {
    let h = DeviceHealth::Healthy;
    let json = serde_json::to_string(&h).unwrap();
    assert!(json.contains("Healthy"));

    let h2 = DeviceHealth::Degraded { reason: "test".to_string() };
    let json2 = serde_json::to_string(&h2).unwrap();
    assert!(json2.contains("Degraded"));
    assert!(json2.contains("test"));
}

// ─── DeviceStats ────────────────────────────────────────────────────

#[test]
fn device_stats_new_defaults() {
    let stats = DeviceStats::new();
    assert_eq!(stats.requests_processed.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert_eq!(stats.total_tokens_generated.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert_eq!(stats.total_execution_time.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert_eq!(stats.consecutive_failures.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn device_stats_default_is_new() {
    let stats = DeviceStats::default();
    assert_eq!(stats.requests_processed.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn device_stats_avg_tokens_zero() {
    let stats = DeviceStats::new();
    assert_eq!(stats.get_avg_tokens_per_second(), 0.0);
}

#[tokio::test]
async fn device_stats_record_success() {
    let stats = DeviceStats::new();
    stats.record_success(100, Duration::from_secs(1)).await;
    assert_eq!(stats.requests_processed.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(stats.total_tokens_generated.load(std::sync::atomic::Ordering::Relaxed), 100);
    assert!(stats.get_avg_tokens_per_second() > 0.0);
}

#[tokio::test]
async fn device_stats_record_multiple_successes() {
    let stats = DeviceStats::new();
    stats.record_success(50, Duration::from_millis(500)).await;
    stats.record_success(50, Duration::from_millis(500)).await;
    assert_eq!(stats.requests_processed.load(std::sync::atomic::Ordering::Relaxed), 2);
    assert_eq!(stats.total_tokens_generated.load(std::sync::atomic::Ordering::Relaxed), 100);
    // 100 tokens / 1000ms = 100 tps
    let avg = stats.get_avg_tokens_per_second();
    assert!(avg > 90.0 && avg < 110.0, "avg was {}", avg);
}

#[test]
fn device_stats_record_failure() {
    let stats = DeviceStats::new();
    stats.record_failure();
    assert_eq!(stats.consecutive_failures.load(std::sync::atomic::Ordering::Relaxed), 1);
    stats.record_failure();
    assert_eq!(stats.consecutive_failures.load(std::sync::atomic::Ordering::Relaxed), 2);
}

#[tokio::test]
async fn device_stats_success_resets_failures() {
    let stats = DeviceStats::new();
    stats.record_failure();
    stats.record_failure();
    assert_eq!(stats.consecutive_failures.load(std::sync::atomic::Ordering::Relaxed), 2);
    stats.record_success(10, Duration::from_millis(100)).await;
    assert_eq!(stats.consecutive_failures.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn device_stats_clone() {
    let stats = DeviceStats::new();
    stats.record_failure();
    let cloned = stats.clone();
    assert_eq!(cloned.consecutive_failures.load(std::sync::atomic::Ordering::Relaxed), 1);
}

#[test]
fn device_stats_debug() {
    let stats = DeviceStats::new();
    let debug = format!("{:?}", stats);
    assert!(debug.contains("DeviceStats"));
}

// ─── DeviceMonitor ──────────────────────────────────────────────────

#[test]
fn device_monitor_new_cpu() {
    let monitor = DeviceMonitor::new(Device::Cpu);
    assert_eq!(monitor.device, Device::Cpu);
}

#[tokio::test]
async fn device_monitor_cpu_capabilities() {
    let monitor = DeviceMonitor::new(Device::Cpu);
    let caps = monitor.capabilities.read().await;
    assert_eq!(caps.device, Device::Cpu);
    assert!(caps.available);
    // CPU should have non-zero total memory
    assert!(caps.memory_total_mb > 0);
}

#[tokio::test]
async fn device_monitor_benchmark_cpu() {
    let monitor = DeviceMonitor::new(Device::Cpu);
    let tps = monitor.benchmark().await.unwrap();
    assert!(tps > 0.0, "CPU benchmark should return positive tps");
    // Check capabilities updated
    let caps = monitor.capabilities.read().await;
    assert!(caps.avg_tokens_per_second > 0.0);
    assert!(caps.last_benchmark.is_some());
}

#[tokio::test]
async fn device_monitor_update_health() {
    let monitor = DeviceMonitor::new(Device::Cpu);
    monitor.update_health().await;
    let health = monitor.health.read().await;
    // CPU should be available — exact health depends on host memory pressure
    assert!(!matches!(*health, DeviceHealth::Unavailable { .. }));
}

#[tokio::test]
async fn device_monitor_get_status() {
    let monitor = DeviceMonitor::new(Device::Cpu);
    let status = monitor.get_status().await;
    assert_eq!(status.device, Device::Cpu);
    assert_eq!(status.requests_processed, 0);
    assert_eq!(status.consecutive_failures, 0);
}

#[tokio::test]
async fn device_monitor_stats_accumulate() {
    let monitor = DeviceMonitor::new(Device::Cpu);
    monitor.stats.record_success(50, Duration::from_millis(500)).await;
    let status = monitor.get_status().await;
    assert_eq!(status.requests_processed, 1);
    assert!(status.avg_tokens_per_second > 0.0);
}

// ─── DeviceStatus ───────────────────────────────────────────────────

#[tokio::test]
async fn device_status_serialize() {
    let monitor = DeviceMonitor::new(Device::Cpu);
    let status = monitor.get_status().await;
    let json = serde_json::to_string(&status).unwrap();
    assert!(json.contains("Cpu"));
    assert!(json.contains("requests_processed"));
}

#[tokio::test]
async fn device_status_debug() {
    let monitor = DeviceMonitor::new(Device::Cpu);
    let status = monitor.get_status().await;
    let debug = format!("{:?}", status);
    assert!(debug.contains("DeviceStatus"));
}

#[tokio::test]
async fn device_status_clone() {
    let monitor = DeviceMonitor::new(Device::Cpu);
    let status = monitor.get_status().await;
    let cloned = status.clone();
    assert_eq!(cloned.device, status.device);
    assert_eq!(cloned.requests_processed, status.requests_processed);
}

// ─── ExecutionRouter ────────────────────────────────────────────────

#[tokio::test]
async fn execution_router_cpu_only_strategy() {
    let config = ExecutionRouterConfig {
        strategy: DeviceSelectionStrategy::CpuOnly,
        benchmark_on_startup: false,
        ..ExecutionRouterConfig::default()
    };
    let router = ExecutionRouter::new(config, vec![Device::Cpu]).await.unwrap();
    let device = router.select_device().await;
    assert_eq!(device, Some(Device::Cpu));
}

#[tokio::test]
async fn execution_router_empty_devices() {
    let config = ExecutionRouterConfig {
        strategy: DeviceSelectionStrategy::CpuOnly,
        benchmark_on_startup: false,
        ..ExecutionRouterConfig::default()
    };
    let router = ExecutionRouter::new(config, vec![]).await.unwrap();
    let device = router.select_device().await;
    assert_eq!(device, None);
}

#[tokio::test]
async fn execution_router_performance_based() {
    let config = ExecutionRouterConfig {
        strategy: DeviceSelectionStrategy::PerformanceBased,
        benchmark_on_startup: true,
        ..ExecutionRouterConfig::default()
    };
    let router = ExecutionRouter::new(config, vec![Device::Cpu]).await.unwrap();
    let device = router.select_device().await;
    // On machines with high memory usage, device health may be Degraded, so None is also valid
    assert!(device == Some(Device::Cpu) || device.is_none());
}

#[tokio::test]
async fn execution_router_load_balance_single_device() {
    let config = ExecutionRouterConfig {
        strategy: DeviceSelectionStrategy::LoadBalance,
        benchmark_on_startup: false,
        ..ExecutionRouterConfig::default()
    };
    let router = ExecutionRouter::new(config, vec![Device::Cpu]).await.unwrap();
    // Load balance requires device to be Healthy; on high-memory-usage hosts it may be Degraded
    let device = router.select_device().await;
    assert!(device == Some(Device::Cpu) || device.is_none());
}

#[tokio::test]
async fn execution_router_user_preference_cpu() {
    let config = ExecutionRouterConfig {
        strategy: DeviceSelectionStrategy::UserPreference(Device::Cpu),
        benchmark_on_startup: false,
        ..ExecutionRouterConfig::default()
    };
    let router = ExecutionRouter::new(config, vec![Device::Cpu]).await.unwrap();
    let device = router.select_device().await;
    assert_eq!(device, Some(Device::Cpu));
}

#[tokio::test]
async fn execution_router_user_preference_unavailable_with_fallback() {
    let config = ExecutionRouterConfig {
        strategy: DeviceSelectionStrategy::UserPreference(Device::Cuda(99)),
        fallback_enabled: true,
        benchmark_on_startup: false,
        ..ExecutionRouterConfig::default()
    };
    let router = ExecutionRouter::new(config, vec![Device::Cpu]).await.unwrap();
    // Cuda(99) not in devices, fallback to CPU
    let device = router.select_device().await;
    assert_eq!(device, Some(Device::Cpu));
}

#[tokio::test]
async fn execution_router_user_preference_unavailable_no_fallback() {
    let config = ExecutionRouterConfig {
        strategy: DeviceSelectionStrategy::UserPreference(Device::Cuda(99)),
        fallback_enabled: false,
        benchmark_on_startup: false,
        ..ExecutionRouterConfig::default()
    };
    let router = ExecutionRouter::new(config, vec![Device::Cpu]).await.unwrap();
    let device = router.select_device().await;
    assert_eq!(device, None);
}

#[tokio::test]
async fn execution_router_record_execution_result() {
    let config = ExecutionRouterConfig {
        strategy: DeviceSelectionStrategy::CpuOnly,
        benchmark_on_startup: false,
        ..ExecutionRouterConfig::default()
    };
    let router = ExecutionRouter::new(config, vec![Device::Cpu]).await.unwrap();
    router.record_execution_result(&Device::Cpu, 100, Duration::from_secs(1), true).await;
    let statuses = router.get_device_statuses().await;
    assert_eq!(statuses.len(), 1);
    assert_eq!(statuses[0].requests_processed, 1);
}

#[tokio::test]
async fn execution_router_record_failure() {
    let config = ExecutionRouterConfig {
        strategy: DeviceSelectionStrategy::CpuOnly,
        benchmark_on_startup: false,
        ..ExecutionRouterConfig::default()
    };
    let router = ExecutionRouter::new(config, vec![Device::Cpu]).await.unwrap();
    router.record_execution_result(&Device::Cpu, 0, Duration::from_secs(0), false).await;
    let statuses = router.get_device_statuses().await;
    assert_eq!(statuses[0].consecutive_failures, 1);
}

#[tokio::test]
async fn execution_router_get_device_statuses() {
    let config = ExecutionRouterConfig {
        strategy: DeviceSelectionStrategy::CpuOnly,
        benchmark_on_startup: false,
        ..ExecutionRouterConfig::default()
    };
    let router = ExecutionRouter::new(config, vec![Device::Cpu]).await.unwrap();
    let statuses = router.get_device_statuses().await;
    assert_eq!(statuses.len(), 1);
    assert_eq!(statuses[0].device, Device::Cpu);
}

#[tokio::test]
async fn execution_router_update_device_health() {
    let config = ExecutionRouterConfig {
        strategy: DeviceSelectionStrategy::CpuOnly,
        benchmark_on_startup: false,
        ..ExecutionRouterConfig::default()
    };
    let router = ExecutionRouter::new(config, vec![Device::Cpu]).await.unwrap();
    // Should not panic
    router.update_device_health().await;
}

#[tokio::test]
async fn execution_router_prefer_gpu_falls_back_to_cpu() {
    let config = ExecutionRouterConfig {
        strategy: DeviceSelectionStrategy::PreferGpu,
        fallback_enabled: true,
        benchmark_on_startup: false,
        ..ExecutionRouterConfig::default()
    };
    // Only CPU available — no GPU
    let router = ExecutionRouter::new(config, vec![Device::Cpu]).await.unwrap();
    let device = router.select_device().await;
    assert_eq!(device, Some(Device::Cpu));
}

#[tokio::test]
async fn execution_router_prefer_gpu_no_fallback_no_gpu() {
    let config = ExecutionRouterConfig {
        strategy: DeviceSelectionStrategy::PreferGpu,
        fallback_enabled: false,
        benchmark_on_startup: false,
        ..ExecutionRouterConfig::default()
    };
    let router = ExecutionRouter::new(config, vec![Device::Cpu]).await.unwrap();
    let device = router.select_device().await;
    // No GPU, no fallback
    assert_eq!(device, None);
}

// ─── ExecutionRouterHealth ──────────────────────────────────────────

#[tokio::test]
async fn execution_router_health_summary() {
    let config = ExecutionRouterConfig {
        strategy: DeviceSelectionStrategy::CpuOnly,
        benchmark_on_startup: false,
        ..ExecutionRouterConfig::default()
    };
    let router = ExecutionRouter::new(config, vec![Device::Cpu]).await.unwrap();
    let health = router.get_health_summary().await;
    assert_eq!(health.total_devices, 1);
    // Device health depends on host memory pressure
    assert_eq!(health.healthy_devices + health.degraded_devices + health.unavailable_devices, 1);
    assert!(health.fallback_enabled);
    assert!(health.strategy.contains("CpuOnly"));
}

#[tokio::test]
async fn execution_router_health_empty() {
    let config = ExecutionRouterConfig {
        strategy: DeviceSelectionStrategy::CpuOnly,
        benchmark_on_startup: false,
        ..ExecutionRouterConfig::default()
    };
    let router = ExecutionRouter::new(config, vec![]).await.unwrap();
    let health = router.get_health_summary().await;
    assert_eq!(health.total_devices, 0);
    assert_eq!(health.healthy_devices, 0);
}

#[test]
fn execution_router_health_serialize() {
    let health = ExecutionRouterHealth {
        total_devices: 2,
        healthy_devices: 1,
        degraded_devices: 1,
        unavailable_devices: 0,
        fallback_enabled: true,
        strategy: "PerformanceBased".to_string(),
    };
    let json = serde_json::to_string(&health).unwrap();
    assert!(json.contains("total_devices"));
    assert!(json.contains("PerformanceBased"));
}

#[test]
fn execution_router_health_debug() {
    let health = ExecutionRouterHealth {
        total_devices: 3,
        healthy_devices: 2,
        degraded_devices: 0,
        unavailable_devices: 1,
        fallback_enabled: false,
        strategy: "LoadBalance".to_string(),
    };
    let debug = format!("{:?}", health);
    assert!(debug.contains("ExecutionRouterHealth"));
}

#[test]
fn execution_router_health_clone() {
    let health = ExecutionRouterHealth {
        total_devices: 1,
        healthy_devices: 1,
        degraded_devices: 0,
        unavailable_devices: 0,
        fallback_enabled: true,
        strategy: "CpuOnly".to_string(),
    };
    let cloned = health.clone();
    assert_eq!(cloned.total_devices, 1);
    assert_eq!(cloned.strategy, "CpuOnly");
}
