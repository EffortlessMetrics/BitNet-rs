//! GPU fleet monitoring dashboard data.
//!
//! Provides [`FleetMonitor`] for aggregating metrics across multiple GPU
//! servers, a device registry with heartbeat tracking, and fleet-wide
//! statistics suitable for dashboard consumption via the
//! `/api/v1/fleet/status` JSON endpoint.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

// ---------------------------------------------------------------------------
// Device info & status
// ---------------------------------------------------------------------------

/// Unique identifier for a GPU device in the fleet.
pub type DeviceUid = String;

/// Static metadata about a GPU device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceInfo {
    /// Unique device identifier (e.g. hostname + GPU index).
    pub uid: DeviceUid,
    /// Human-readable device name (e.g. "Intel Arc A770").
    pub name: String,
    /// Server hostname where this device resides.
    pub hostname: String,
    /// Total GPU memory in bytes.
    pub total_memory_bytes: u64,
    /// Backend type (e.g. "opencl", "vulkan", "cuda").
    pub backend: String,
}

/// Health status of a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceHealth {
    /// Device is healthy and accepting work.
    Healthy,
    /// Device has not sent a heartbeat within the expected interval.
    Stale,
    /// Device has been explicitly marked offline.
    Offline,
}

/// Live status of a registered device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceStatus {
    /// Static device metadata.
    pub info: GpuDeviceInfo,
    /// Current GPU utilization as a fraction in `[0.0, 1.0]`.
    pub utilization: f64,
    /// Currently used GPU memory in bytes.
    pub used_memory_bytes: u64,
    /// Health status derived from heartbeat.
    pub health: DeviceHealth,
    /// Timestamp of last heartbeat (seconds since UNIX epoch).
    pub last_heartbeat_epoch: u64,
}

// ---------------------------------------------------------------------------
// Fleet-wide statistics
// ---------------------------------------------------------------------------

/// Aggregated fleet-wide GPU statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetStats {
    /// Total number of registered devices.
    pub device_count: usize,
    /// Number of devices currently healthy.
    pub healthy_count: usize,
    /// Total GPU memory across all devices (bytes).
    pub total_memory_bytes: u64,
    /// Total used GPU memory across all devices (bytes).
    pub used_memory_bytes: u64,
    /// Average utilization across healthy devices.
    pub avg_utilization: f64,
}

// ---------------------------------------------------------------------------
// Fleet status (JSON API response)
// ---------------------------------------------------------------------------

/// Complete fleet status returned by `/api/v1/fleet/status`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetStatusResponse {
    /// Timestamp when the snapshot was taken (seconds since UNIX epoch).
    pub timestamp_epoch: u64,
    /// Aggregated fleet statistics.
    pub stats: FleetStats,
    /// Per-device status list.
    pub devices: Vec<DeviceStatus>,
}

// ---------------------------------------------------------------------------
// Fleet monitor
// ---------------------------------------------------------------------------

/// Internal entry in the device registry.
#[derive(Debug, Clone)]
struct DeviceEntry {
    info: GpuDeviceInfo,
    utilization: f64,
    used_memory_bytes: u64,
    health: DeviceHealth,
    last_heartbeat: Instant,
    last_heartbeat_epoch: u64,
}

/// Fleet monitor aggregating metrics across multiple GPU servers.
///
/// Thread-safe via internal `RwLock`. Call [`FleetMonitor::heartbeat`]
/// periodically from each device agent, and [`FleetMonitor::status`] to
/// retrieve the dashboard snapshot.
#[derive(Debug, Clone)]
pub struct FleetMonitor {
    devices: Arc<RwLock<HashMap<DeviceUid, DeviceEntry>>>,
    /// Duration after which a device without a heartbeat is marked stale.
    stale_threshold: Duration,
}

impl FleetMonitor {
    /// Create a new fleet monitor.
    ///
    /// `stale_threshold` controls how long a device can go without a
    /// heartbeat before being marked [`DeviceHealth::Stale`].
    pub fn new(stale_threshold: Duration) -> Self {
        Self {
            devices: Arc::new(RwLock::new(HashMap::new())),
            stale_threshold,
        }
    }

    /// Register a new device in the fleet.
    ///
    /// If a device with the same UID already exists it is replaced.
    pub fn register_device(&self, info: GpuDeviceInfo) {
        let uid = info.uid.clone();
        let entry = DeviceEntry {
            info,
            utilization: 0.0,
            used_memory_bytes: 0,
            health: DeviceHealth::Healthy,
            last_heartbeat: Instant::now(),
            last_heartbeat_epoch: epoch_secs(),
        };
        self.devices.write().unwrap().insert(uid, entry);
    }

    /// Remove a device from the fleet registry.
    pub fn remove_device(&self, uid: &str) -> bool {
        self.devices.write().unwrap().remove(uid).is_some()
    }

    /// Record a heartbeat from a device, updating its utilization and memory.
    pub fn heartbeat(
        &self,
        uid: &str,
        utilization: f64,
        used_memory_bytes: u64,
    ) -> bool {
        let mut devices = self.devices.write().unwrap();
        if let Some(entry) = devices.get_mut(uid) {
            entry.utilization = utilization.clamp(0.0, 1.0);
            entry.used_memory_bytes = used_memory_bytes;
            entry.health = DeviceHealth::Healthy;
            entry.last_heartbeat = Instant::now();
            entry.last_heartbeat_epoch = epoch_secs();
            true
        } else {
            false
        }
    }

    /// Mark a device as offline.
    pub fn mark_offline(&self, uid: &str) -> bool {
        let mut devices = self.devices.write().unwrap();
        if let Some(entry) = devices.get_mut(uid) {
            entry.health = DeviceHealth::Offline;
            true
        } else {
            false
        }
    }

    /// Scan all devices and mark those whose heartbeat is older than
    /// `stale_threshold` as [`DeviceHealth::Stale`].
    pub fn check_stale_devices(&self) -> usize {
        let now = Instant::now();
        let threshold = self.stale_threshold;
        let mut count = 0;
        let mut devices = self.devices.write().unwrap();
        for entry in devices.values_mut() {
            if entry.health == DeviceHealth::Healthy
                && now.duration_since(entry.last_heartbeat) > threshold
            {
                entry.health = DeviceHealth::Stale;
                count += 1;
            }
        }
        count
    }

    /// Return the number of registered devices.
    pub fn device_count(&self) -> usize {
        self.devices.read().unwrap().len()
    }

    /// Build an aggregated fleet status snapshot.
    pub fn status(&self) -> FleetStatusResponse {
        let devices = self.devices.read().unwrap();

        let mut device_statuses = Vec::with_capacity(devices.len());
        let mut total_mem: u64 = 0;
        let mut used_mem: u64 = 0;
        let mut util_sum: f64 = 0.0;
        let mut healthy: usize = 0;

        for entry in devices.values() {
            total_mem += entry.info.total_memory_bytes;
            used_mem += entry.used_memory_bytes;
            if entry.health == DeviceHealth::Healthy {
                util_sum += entry.utilization;
                healthy += 1;
            }
            device_statuses.push(DeviceStatus {
                info: entry.info.clone(),
                utilization: entry.utilization,
                used_memory_bytes: entry.used_memory_bytes,
                health: entry.health,
                last_heartbeat_epoch: entry.last_heartbeat_epoch,
            });
        }

        let avg_util = if healthy > 0 {
            util_sum / healthy as f64
        } else {
            0.0
        };

        FleetStatusResponse {
            timestamp_epoch: epoch_secs(),
            stats: FleetStats {
                device_count: devices.len(),
                healthy_count: healthy,
                total_memory_bytes: total_mem,
                used_memory_bytes: used_mem,
                avg_utilization: avg_util,
            },
            devices: device_statuses,
        }
    }
}

/// Create an axum [`Router`] exposing the fleet status endpoint.
///
/// Mounts `GET /api/v1/fleet/status` returning a [`FleetStatusResponse`]
/// as JSON.
pub fn fleet_routes(
    monitor: Arc<FleetMonitor>,
) -> axum::Router {
    use axum::extract::State;
    use axum::routing::get;

    async fn fleet_status_handler(
        State(monitor): State<Arc<FleetMonitor>>,
    ) -> axum::Json<FleetStatusResponse> {
        axum::Json(monitor.status())
    }

    axum::Router::new()
        .route("/api/v1/fleet/status", get(fleet_status_handler))
        .with_state(monitor)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_device(uid: &str, mem_gb: u64) -> GpuDeviceInfo {
        GpuDeviceInfo {
            uid: uid.to_string(),
            name: format!("GPU-{uid}"),
            hostname: "server-1".to_string(),
            total_memory_bytes: mem_gb * 1024 * 1024 * 1024,
            backend: "opencl".to_string(),
        }
    }

    #[test]
    fn test_register_device() {
        let monitor = FleetMonitor::new(Duration::from_secs(30));
        monitor.register_device(test_device("gpu0", 16));
        assert_eq!(monitor.device_count(), 1);
    }

    #[test]
    fn test_remove_device() {
        let monitor = FleetMonitor::new(Duration::from_secs(30));
        monitor.register_device(test_device("gpu0", 16));
        assert!(monitor.remove_device("gpu0"));
        assert_eq!(monitor.device_count(), 0);
        assert!(!monitor.remove_device("gpu0"));
    }

    #[test]
    fn test_heartbeat_updates_device() {
        let monitor = FleetMonitor::new(Duration::from_secs(30));
        monitor.register_device(test_device("gpu0", 16));

        assert!(monitor.heartbeat("gpu0", 0.75, 8 * 1024 * 1024 * 1024));
        let status = monitor.status();
        let dev = &status.devices[0];
        assert_eq!(dev.utilization, 0.75);
        assert_eq!(dev.used_memory_bytes, 8 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_heartbeat_unknown_device_returns_false() {
        let monitor = FleetMonitor::new(Duration::from_secs(30));
        assert!(!monitor.heartbeat("unknown", 0.5, 0));
    }

    #[test]
    fn test_utilization_clamped() {
        let monitor = FleetMonitor::new(Duration::from_secs(30));
        monitor.register_device(test_device("gpu0", 16));
        monitor.heartbeat("gpu0", 1.5, 0);

        let status = monitor.status();
        assert_eq!(status.devices[0].utilization, 1.0);
    }

    #[test]
    fn test_mark_offline() {
        let monitor = FleetMonitor::new(Duration::from_secs(30));
        monitor.register_device(test_device("gpu0", 16));
        assert!(monitor.mark_offline("gpu0"));
        assert!(!monitor.mark_offline("nonexistent"));

        let status = monitor.status();
        assert_eq!(status.devices[0].health, DeviceHealth::Offline);
    }

    #[test]
    fn test_fleet_stats_aggregation() {
        let monitor = FleetMonitor::new(Duration::from_secs(60));
        monitor.register_device(test_device("gpu0", 16));
        monitor.register_device(test_device("gpu1", 8));
        monitor.heartbeat("gpu0", 0.80, 10 * 1024 * 1024 * 1024);
        monitor.heartbeat("gpu1", 0.40, 2 * 1024 * 1024 * 1024);

        let status = monitor.status();
        assert_eq!(status.stats.device_count, 2);
        assert_eq!(status.stats.healthy_count, 2);
        assert_eq!(
            status.stats.total_memory_bytes,
            24 * 1024 * 1024 * 1024
        );
        assert_eq!(
            status.stats.used_memory_bytes,
            12 * 1024 * 1024 * 1024
        );
        assert!((status.stats.avg_utilization - 0.60).abs() < 1e-6);
    }

    #[test]
    fn test_fleet_status_json_serialization() {
        let monitor = FleetMonitor::new(Duration::from_secs(30));
        monitor.register_device(test_device("gpu0", 16));
        monitor.heartbeat("gpu0", 0.5, 4 * 1024 * 1024 * 1024);

        let status = monitor.status();
        let json = serde_json::to_string(&status).expect("serialize");
        assert!(json.contains("\"device_count\":1"));
        assert!(json.contains("\"healthy_count\":1"));
        assert!(json.contains("\"backend\":\"opencl\""));
    }

    #[test]
    fn test_check_stale_devices() {
        // Use a very short threshold so devices become stale immediately.
        let monitor = FleetMonitor::new(Duration::from_millis(0));
        monitor.register_device(test_device("gpu0", 16));
        // The device was registered with Instant::now(), so with 0ms
        // threshold it should be stale after any delay.
        std::thread::sleep(Duration::from_millis(1));
        let stale_count = monitor.check_stale_devices();
        assert_eq!(stale_count, 1);

        let status = monitor.status();
        assert_eq!(status.devices[0].health, DeviceHealth::Stale);
        assert_eq!(status.stats.healthy_count, 0);
    }

    #[test]
    fn test_offline_device_excluded_from_avg_util() {
        let monitor = FleetMonitor::new(Duration::from_secs(60));
        monitor.register_device(test_device("gpu0", 16));
        monitor.register_device(test_device("gpu1", 8));
        monitor.heartbeat("gpu0", 0.80, 0);
        monitor.heartbeat("gpu1", 0.40, 0);
        monitor.mark_offline("gpu1");

        let status = monitor.status();
        assert_eq!(status.stats.healthy_count, 1);
        // Only gpu0's 0.80 should count
        assert!((status.stats.avg_utilization - 0.80).abs() < 1e-6);
    }
}
