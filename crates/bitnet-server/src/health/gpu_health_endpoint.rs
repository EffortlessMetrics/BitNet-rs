//! GPU health check REST endpoint (`/api/v1/health/gpu`)
//!
//! Provides detailed GPU device information including memory, utilization,
//! temperature, and alerting thresholds. A background poller periodically
//! refreshes the cached snapshot so the endpoint returns near-instantly.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Router;
use axum::extract::State;
use axum::response::Json;
use axum::routing::get;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Per-device health information returned by the endpoint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GpuDeviceHealth {
    pub device_name: String,
    pub driver_version: String,
    pub memory_total: u64,
    pub memory_used: u64,
    pub utilization_percent: f64,
    pub temperature_c: f64,
}

/// An alert raised when a threshold is exceeded.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GpuAlert {
    pub device_name: String,
    pub kind: GpuAlertKind,
    pub message: String,
}

/// Discriminant for alert types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum GpuAlertKind {
    HighMemory,
    HighTemperature,
}

/// Top-level JSON response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GpuHealthResponse {
    pub status: String,
    pub devices: Vec<GpuDeviceHealth>,
    pub alerts: Vec<GpuAlert>,
    pub polled_at_epoch_ms: u64,
}

/// Configurable alerting thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuHealthThresholds {
    /// Fire alert when memory usage exceeds this percentage (0–100).
    pub memory_percent: f64,
    /// Fire alert when temperature exceeds this value (°C).
    pub temperature_c: f64,
}

impl Default for GpuHealthThresholds {
    fn default() -> Self {
        Self { memory_percent: 90.0, temperature_c: 85.0 }
    }
}

/// Configuration for the background health poller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuHealthPollerConfig {
    /// Interval between background polls.
    pub poll_interval: Duration,
    /// Alerting thresholds.
    pub thresholds: GpuHealthThresholds,
}

impl Default for GpuHealthPollerConfig {
    fn default() -> Self {
        Self { poll_interval: Duration::from_secs(10), thresholds: GpuHealthThresholds::default() }
    }
}

// ---------------------------------------------------------------------------
// Collector trait (allows test injection)
// ---------------------------------------------------------------------------

/// Abstracts GPU metric collection so tests can inject fake data.
#[async_trait::async_trait]
pub trait GpuMetricsCollector: Send + Sync + 'static {
    async fn collect(&self) -> Vec<GpuDeviceHealth>;
}

/// Default collector that delegates to the existing `GpuMetrics` helper.
#[derive(Default)]
pub struct DefaultGpuCollector;

#[async_trait::async_trait]
impl GpuMetricsCollector for DefaultGpuCollector {
    async fn collect(&self) -> Vec<GpuDeviceHealth> {
        let m = super::gpu_monitor::GpuMetrics::collect().await;
        if m.error_message.is_some() {
            return Vec::new();
        }
        vec![GpuDeviceHealth {
            device_name: "gpu-0".to_string(),
            driver_version: "unknown".to_string(),
            memory_total: (m.memory_total_mb * 1_048_576.0) as u64,
            memory_used: (m.memory_used_mb * 1_048_576.0) as u64,
            utilization_percent: m.utilization_percent,
            temperature_c: m.temperature_celsius,
        }]
    }
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

struct GpuHealthInner {
    devices: Vec<GpuDeviceHealth>,
    polled_at: Instant,
    polled_at_epoch_ms: u64,
}

/// Shared, lock-protected health snapshot.
pub struct GpuHealthState<C: GpuMetricsCollector = DefaultGpuCollector> {
    inner: RwLock<GpuHealthInner>,
    thresholds: GpuHealthThresholds,
    collector: C,
}

impl<C: GpuMetricsCollector> GpuHealthState<C> {
    pub fn new(thresholds: GpuHealthThresholds, collector: C) -> Self {
        Self {
            inner: RwLock::new(GpuHealthInner {
                devices: Vec::new(),
                polled_at: Instant::now(),
                polled_at_epoch_ms: epoch_ms(),
            }),
            thresholds,
            collector,
        }
    }

    /// Poll GPU devices and update the cached snapshot.
    pub async fn poll(&self) {
        let devices = self.collector.collect().await;
        let mut guard = self.inner.write().await;
        guard.devices = devices;
        guard.polled_at = Instant::now();
        guard.polled_at_epoch_ms = epoch_ms();
    }

    /// Build the JSON response from the cached snapshot.
    pub async fn snapshot(&self) -> GpuHealthResponse {
        let guard = self.inner.read().await;
        let alerts = evaluate_alerts(&guard.devices, &self.thresholds);
        let status = if alerts.is_empty() { "healthy".to_string() } else { "warning".to_string() };
        GpuHealthResponse {
            status,
            devices: guard.devices.clone(),
            alerts,
            polled_at_epoch_ms: guard.polled_at_epoch_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// Alert evaluation
// ---------------------------------------------------------------------------

/// Evaluate alerting thresholds against a list of devices.
pub fn evaluate_alerts(
    devices: &[GpuDeviceHealth],
    thresholds: &GpuHealthThresholds,
) -> Vec<GpuAlert> {
    let mut alerts = Vec::new();
    for dev in devices {
        if dev.memory_total > 0 {
            let pct = dev.memory_used as f64 / dev.memory_total as f64 * 100.0;
            if pct > thresholds.memory_percent {
                alerts.push(GpuAlert {
                    device_name: dev.device_name.clone(),
                    kind: GpuAlertKind::HighMemory,
                    message: format!(
                        "Memory usage {:.1}% exceeds threshold {:.1}%",
                        pct, thresholds.memory_percent
                    ),
                });
            }
        }
        if dev.temperature_c > thresholds.temperature_c {
            alerts.push(GpuAlert {
                device_name: dev.device_name.clone(),
                kind: GpuAlertKind::HighTemperature,
                message: format!(
                    "Temperature {:.1}°C exceeds threshold {:.1}°C",
                    dev.temperature_c, thresholds.temperature_c
                ),
            });
        }
    }
    alerts
}

// ---------------------------------------------------------------------------
// Axum handler + router factory
// ---------------------------------------------------------------------------

async fn gpu_health_handler<C: GpuMetricsCollector>(
    State(state): State<Arc<GpuHealthState<C>>>,
) -> Json<GpuHealthResponse> {
    Json(state.snapshot().await)
}

/// Create the `/api/v1/health/gpu` route.
pub fn create_gpu_health_route<C: GpuMetricsCollector>(state: Arc<GpuHealthState<C>>) -> Router {
    Router::new().route("/api/v1/health/gpu", get(gpu_health_handler::<C>)).with_state(state)
}

/// Spawn the background poller task. Returns a `JoinHandle` the caller can
/// abort on shutdown.
pub fn spawn_gpu_health_poller<C: GpuMetricsCollector>(
    state: Arc<GpuHealthState<C>>,
    interval: Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        loop {
            ticker.tick().await;
            state.poll().await;
        }
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    // Fake collector for deterministic tests
    struct FakeCollector(Vec<GpuDeviceHealth>);

    #[async_trait::async_trait]
    impl GpuMetricsCollector for FakeCollector {
        async fn collect(&self) -> Vec<GpuDeviceHealth> {
            self.0.clone()
        }
    }

    fn sample_device(name: &str, mem_used: u64, mem_total: u64, temp: f64) -> GpuDeviceHealth {
        GpuDeviceHealth {
            device_name: name.to_string(),
            driver_version: "535.129.03".to_string(),
            memory_total: mem_total,
            memory_used: mem_used,
            utilization_percent: 42.0,
            temperature_c: temp,
        }
    }

    #[test]
    fn test_evaluate_no_alerts_when_within_thresholds() {
        let devices = vec![sample_device("gpu-0", 4_000, 10_000, 70.0)];
        let thresholds = GpuHealthThresholds::default();
        let alerts = evaluate_alerts(&devices, &thresholds);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_evaluate_high_memory_alert() {
        let devices = vec![sample_device("gpu-0", 9_500, 10_000, 70.0)];
        let thresholds = GpuHealthThresholds::default(); // 90%
        let alerts = evaluate_alerts(&devices, &thresholds);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].kind, GpuAlertKind::HighMemory);
    }

    #[test]
    fn test_evaluate_high_temperature_alert() {
        let devices = vec![sample_device("gpu-0", 1_000, 10_000, 90.0)];
        let thresholds = GpuHealthThresholds::default(); // 85°C
        let alerts = evaluate_alerts(&devices, &thresholds);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].kind, GpuAlertKind::HighTemperature);
    }

    #[test]
    fn test_evaluate_multiple_alerts() {
        let devices = vec![
            sample_device("gpu-0", 9_500, 10_000, 90.0),
            sample_device("gpu-1", 1_000, 10_000, 30.0),
        ];
        let thresholds = GpuHealthThresholds::default();
        let alerts = evaluate_alerts(&devices, &thresholds);
        // gpu-0: high memory + high temp, gpu-1: clean
        assert_eq!(alerts.len(), 2);
    }

    #[test]
    fn test_custom_thresholds() {
        let devices = vec![sample_device("gpu-0", 8_000, 10_000, 80.0)];
        let thresholds = GpuHealthThresholds { memory_percent: 75.0, temperature_c: 75.0 };
        let alerts = evaluate_alerts(&devices, &thresholds);
        assert_eq!(alerts.len(), 2);
    }

    #[tokio::test]
    async fn test_snapshot_healthy() {
        let devices = vec![sample_device("gpu-0", 4_000, 10_000, 60.0)];
        let state =
            Arc::new(GpuHealthState::new(GpuHealthThresholds::default(), FakeCollector(devices)));
        state.poll().await;
        let resp = state.snapshot().await;
        assert_eq!(resp.status, "healthy");
        assert!(resp.alerts.is_empty());
        assert_eq!(resp.devices.len(), 1);
    }

    #[tokio::test]
    async fn test_snapshot_warning() {
        let devices = vec![sample_device("gpu-0", 9_500, 10_000, 60.0)];
        let state =
            Arc::new(GpuHealthState::new(GpuHealthThresholds::default(), FakeCollector(devices)));
        state.poll().await;
        let resp = state.snapshot().await;
        assert_eq!(resp.status, "warning");
        assert_eq!(resp.alerts.len(), 1);
    }

    #[tokio::test]
    async fn test_endpoint_returns_200() {
        let devices = vec![sample_device("gpu-0", 2_000, 10_000, 55.0)];
        let state =
            Arc::new(GpuHealthState::new(GpuHealthThresholds::default(), FakeCollector(devices)));
        state.poll().await;
        let app = create_gpu_health_route(state);
        let req = Request::builder().uri("/api/v1/health/gpu").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1_000_000).await.unwrap();
        let parsed: GpuHealthResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed.status, "healthy");
        assert_eq!(parsed.devices.len(), 1);
        assert_eq!(parsed.devices[0].device_name, "gpu-0");
    }

    #[tokio::test]
    async fn test_poll_updates_snapshot() {
        let state = Arc::new(GpuHealthState::new(
            GpuHealthThresholds::default(),
            FakeCollector(vec![sample_device("gpu-0", 1_000, 10_000, 40.0)]),
        ));
        // Before first poll, devices list is empty.
        let resp = state.snapshot().await;
        assert!(resp.devices.is_empty());

        // After poll, devices are populated.
        state.poll().await;
        let resp = state.snapshot().await;
        assert_eq!(resp.devices.len(), 1);
    }
}
