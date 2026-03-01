//! Smart request routing to GPU backends with load-aware device selection.
//!
//! [`GpuRouter`] maintains a registry of available GPU devices, tracks their
//! health and utilisation, and selects the best device for each incoming
//! inference request based on model size, available memory, current load, and
//! device capabilities.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Device descriptor
// ---------------------------------------------------------------------------

/// Unique identifier for a GPU device in the router.
pub type DeviceId = String;

/// Static capabilities of a registered GPU device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceInfo {
    pub id: DeviceId,
    pub name: String,
    /// Total VRAM in bytes.
    pub memory_total: u64,
    /// Backend kind (e.g. "cuda", "opencl", "vulkan").
    pub backend: String,
    /// Compute capability string (e.g. "8.6" for Ampere).
    pub compute_capability: Option<String>,
    /// Maximum concurrent requests the device should handle.
    pub max_concurrent: u32,
}

/// Mutable runtime state tracked per device.
#[derive(Debug)]
struct DeviceState {
    info: GpuDeviceInfo,
    healthy: bool,
    /// Currently used VRAM in bytes.
    memory_used: u64,
    /// Number of in-flight requests.
    active_requests: AtomicU64,
    /// Cumulative requests routed to this device.
    total_requests: AtomicU64,
    /// Last time a health check succeeded.
    last_health_check: Option<Instant>,
}

// ---------------------------------------------------------------------------
// Routing criteria
// ---------------------------------------------------------------------------

/// Information about an incoming request that guides routing.
#[derive(Debug, Clone)]
pub struct RouteRequest {
    /// Estimated model VRAM requirement in bytes.
    pub model_memory_bytes: u64,
    /// Optional preferred backend (e.g. "cuda").
    pub preferred_backend: Option<String>,
    /// Whether the request requires a specific compute capability.
    pub min_compute_capability: Option<String>,
}

/// Result returned by the router on successful device selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteDecision {
    pub device_id: DeviceId,
    pub device_name: String,
    pub backend: String,
    pub available_memory: u64,
}

/// Errors that can occur during routing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RouteError {
    /// No devices registered in the router.
    NoDevicesRegistered,
    /// No healthy device has sufficient memory for the request.
    InsufficientMemory { required: u64, best_available: u64 },
    /// All registered devices are unhealthy.
    AllDevicesUnhealthy,
    /// No device matches the requested backend.
    NoMatchingBackend { requested: String },
}

impl std::fmt::Display for RouteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RouteError::NoDevicesRegistered => {
                write!(f, "no GPU devices registered")
            }
            RouteError::InsufficientMemory {
                required,
                best_available,
            } => write!(
                f,
                "insufficient GPU memory: need {required} B, \
                 best available {best_available} B",
            ),
            RouteError::AllDevicesUnhealthy => {
                write!(f, "all GPU devices are unhealthy")
            }
            RouteError::NoMatchingBackend { requested } => {
                write!(f, "no device with backend '{requested}'")
            }
        }
    }
}

impl std::error::Error for RouteError {}

// ---------------------------------------------------------------------------
// GpuRouter
// ---------------------------------------------------------------------------

/// Smart GPU request router with device registry and load tracking.
///
/// Thread-safe: the inner state is behind an `Arc<RwLock<..>>` so the
/// router can be shared across Axum handler tasks.
#[derive(Clone)]
pub struct GpuRouter {
    inner: Arc<RwLock<RouterInner>>,
}

struct RouterInner {
    devices: HashMap<DeviceId, DeviceState>,
    /// Round-robin counter used when devices are equally suited.
    rr_counter: u64,
}

impl GpuRouter {
    /// Create a new, empty router.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(RouterInner {
                devices: HashMap::new(),
                rr_counter: 0,
            })),
        }
    }

    // -- device registration -----------------------------------------------

    /// Register a GPU device. Replaces any existing device with the
    /// same id.
    pub fn register_device(&self, info: GpuDeviceInfo) {
        let mut inner = self.inner.write().unwrap();
        let id = info.id.clone();
        inner.devices.insert(
            id,
            DeviceState {
                info,
                healthy: true,
                memory_used: 0,
                active_requests: AtomicU64::new(0),
                total_requests: AtomicU64::new(0),
                last_health_check: None,
            },
        );
    }

    /// Remove a device from the registry.
    pub fn remove_device(&self, id: &str) -> bool {
        let mut inner = self.inner.write().unwrap();
        inner.devices.remove(id).is_some()
    }

    /// Number of registered devices.
    pub fn device_count(&self) -> usize {
        self.inner.read().unwrap().devices.len()
    }

    // -- health management -------------------------------------------------

    /// Mark a device as healthy or unhealthy.
    pub fn set_device_health(&self, id: &str, healthy: bool) {
        let mut inner = self.inner.write().unwrap();
        if let Some(state) = inner.devices.get_mut(id) {
            state.healthy = healthy;
            if healthy {
                state.last_health_check = Some(Instant::now());
            }
        }
    }

    /// Returns `true` if the device exists and is healthy.
    pub fn is_device_healthy(&self, id: &str) -> Option<bool> {
        let inner = self.inner.read().unwrap();
        inner.devices.get(id).map(|s| s.healthy)
    }

    /// Return the number of healthy devices.
    pub fn healthy_device_count(&self) -> usize {
        let inner = self.inner.read().unwrap();
        inner.devices.values().filter(|s| s.healthy).count()
    }

    // -- memory accounting -------------------------------------------------

    /// Update the reported memory usage for a device.
    pub fn update_memory_usage(&self, id: &str, used_bytes: u64) {
        let mut inner = self.inner.write().unwrap();
        if let Some(state) = inner.devices.get_mut(id) {
            state.memory_used = used_bytes;
        }
    }

    /// Free memory for a given device (total − used).
    pub fn free_memory(&self, id: &str) -> Option<u64> {
        let inner = self.inner.read().unwrap();
        inner
            .devices
            .get(id)
            .map(|s| s.info.memory_total.saturating_sub(s.memory_used))
    }

    // -- load tracking -----------------------------------------------------

    /// Notify the router that a request has been dispatched to `id`.
    pub fn begin_request(&self, id: &str) {
        let inner = self.inner.read().unwrap();
        if let Some(state) = inner.devices.get(id) {
            state.active_requests.fetch_add(1, Ordering::Relaxed);
            state.total_requests.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Notify the router that a request on `id` has completed.
    pub fn end_request(&self, id: &str) {
        let inner = self.inner.read().unwrap();
        if let Some(state) = inner.devices.get(id) {
            state.active_requests.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Current number of in-flight requests for a device.
    pub fn active_requests(&self, id: &str) -> Option<u64> {
        let inner = self.inner.read().unwrap();
        inner
            .devices
            .get(id)
            .map(|s| s.active_requests.load(Ordering::Relaxed))
    }

    // -- routing -----------------------------------------------------------

    /// Select the best device for the given request.
    ///
    /// Strategy:
    /// 1. Filter to healthy devices.
    /// 2. If `preferred_backend` is set, restrict to matching devices.
    /// 3. Exclude devices with insufficient free memory.
    /// 4. Among remaining candidates pick the one with the most free
    ///    memory weighted by current utilisation. On a tie, round-robin.
    pub fn route(
        &self,
        req: &RouteRequest,
    ) -> Result<RouteDecision, RouteError> {
        let mut inner = self.inner.write().unwrap();
        if inner.devices.is_empty() {
            return Err(RouteError::NoDevicesRegistered);
        }

        // Snapshot candidate data so we don't hold borrows across
        // the mutable `rr_counter` update below.
        struct Snap {
            id: String,
            name: String,
            backend: String,
            free: u64,
            score: u64,
        }

        let has_healthy =
            inner.devices.values().any(|s| s.healthy);
        if !has_healthy {
            return Err(RouteError::AllDevicesUnhealthy);
        }

        // 1+2. Healthy + backend filter
        if let Some(ref be) = req.preferred_backend {
            let any_match = inner
                .devices
                .values()
                .any(|s| s.healthy && s.info.backend == *be);
            if !any_match {
                return Err(RouteError::NoMatchingBackend {
                    requested: be.clone(),
                });
            }
        }

        // 3. Memory filter → collect owned snapshots
        let mut best_available: u64 = 0;
        let candidates: Vec<Snap> = inner
            .devices
            .values()
            .filter(|s| {
                if !s.healthy {
                    return false;
                }
                if let Some(ref be) = req.preferred_backend {
                    if s.info.backend != *be {
                        return false;
                    }
                }
                true
            })
            .filter_map(|s| {
                let free =
                    s.info.memory_total.saturating_sub(s.memory_used);
                if free > best_available {
                    best_available = free;
                }
                if free < req.model_memory_bytes {
                    return None;
                }
                let active =
                    s.active_requests.load(Ordering::Relaxed);
                let score = free.saturating_sub(active * 1_000_000);
                Some(Snap {
                    id: s.info.id.clone(),
                    name: s.info.name.clone(),
                    backend: s.info.backend.clone(),
                    free,
                    score,
                })
            })
            .collect();

        if candidates.is_empty() {
            return Err(RouteError::InsufficientMemory {
                required: req.model_memory_bytes,
                best_available,
            });
        }

        // 4. Pick best score; round-robin on tie.
        let max_score =
            candidates.iter().map(|c| c.score).max().unwrap_or(0);
        let top: Vec<usize> = candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| c.score == max_score)
            .map(|(i, _)| i)
            .collect();

        let pick_idx = if top.len() == 1 {
            top[0]
        } else {
            let rr = inner.rr_counter;
            inner.rr_counter = rr.wrapping_add(1);
            top[rr as usize % top.len()]
        };

        let chosen = &candidates[pick_idx];
        Ok(RouteDecision {
            device_id: chosen.id.clone(),
            device_name: chosen.name.clone(),
            backend: chosen.backend.clone(),
            available_memory: chosen.free,
        })
    }

    /// Convenience: list all registered device ids.
    pub fn device_ids(&self) -> Vec<DeviceId> {
        let inner = self.inner.read().unwrap();
        inner.devices.keys().cloned().collect()
    }
}

impl Default for GpuRouter {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(id: &str, mem: u64, be: &str) -> GpuDeviceInfo {
        GpuDeviceInfo {
            id: id.into(),
            name: format!("Test GPU {id}"),
            memory_total: mem,
            backend: be.into(),
            compute_capability: None,
            max_concurrent: 8,
        }
    }

    fn req(mem: u64) -> RouteRequest {
        RouteRequest {
            model_memory_bytes: mem,
            preferred_backend: None,
            min_compute_capability: None,
        }
    }

    #[test]
    fn register_and_count() {
        let r = GpuRouter::new();
        assert_eq!(r.device_count(), 0);
        r.register_device(dev("g0", 8_000_000_000, "cuda"));
        assert_eq!(r.device_count(), 1);
        r.register_device(dev("g1", 16_000_000_000, "cuda"));
        assert_eq!(r.device_count(), 2);
    }

    #[test]
    fn remove_device() {
        let r = GpuRouter::new();
        r.register_device(dev("g0", 8_000_000_000, "cuda"));
        assert!(r.remove_device("g0"));
        assert_eq!(r.device_count(), 0);
        assert!(!r.remove_device("g0"));
    }

    #[test]
    fn route_single_device() {
        let r = GpuRouter::new();
        r.register_device(dev("g0", 8_000_000_000, "cuda"));
        let d = r.route(&req(1_000_000_000)).unwrap();
        assert_eq!(d.device_id, "g0");
    }

    #[test]
    fn route_prefers_more_free_memory() {
        let r = GpuRouter::new();
        r.register_device(dev("small", 4_000_000_000, "cuda"));
        r.register_device(dev("large", 16_000_000_000, "cuda"));
        let d = r.route(&req(1_000_000_000)).unwrap();
        assert_eq!(d.device_id, "large");
    }

    #[test]
    fn route_skips_unhealthy() {
        let r = GpuRouter::new();
        r.register_device(dev("sick", 16_000_000_000, "cuda"));
        r.register_device(dev("ok", 8_000_000_000, "cuda"));
        r.set_device_health("sick", false);
        let d = r.route(&req(1_000_000_000)).unwrap();
        assert_eq!(d.device_id, "ok");
    }

    #[test]
    fn all_unhealthy_error() {
        let r = GpuRouter::new();
        r.register_device(dev("g0", 8_000_000_000, "cuda"));
        r.set_device_health("g0", false);
        let e = r.route(&req(1_000_000_000)).unwrap_err();
        assert_eq!(e, RouteError::AllDevicesUnhealthy);
    }

    #[test]
    fn no_devices_error() {
        let r = GpuRouter::new();
        let e = r.route(&req(1_000_000_000)).unwrap_err();
        assert_eq!(e, RouteError::NoDevicesRegistered);
    }

    #[test]
    fn insufficient_memory_error() {
        let r = GpuRouter::new();
        r.register_device(dev("g0", 4_000_000_000, "cuda"));
        let e = r.route(&req(8_000_000_000)).unwrap_err();
        assert!(matches!(e, RouteError::InsufficientMemory { .. }));
    }

    #[test]
    fn backend_filter() {
        let r = GpuRouter::new();
        r.register_device(dev("cu0", 8_000_000_000, "cuda"));
        r.register_device(dev("ocl0", 16_000_000_000, "opencl"));
        let rr = RouteRequest {
            model_memory_bytes: 1_000_000_000,
            preferred_backend: Some("opencl".into()),
            min_compute_capability: None,
        };
        let d = r.route(&rr).unwrap();
        assert_eq!(d.device_id, "ocl0");
    }

    #[test]
    fn no_matching_backend_error() {
        let r = GpuRouter::new();
        r.register_device(dev("cu0", 8_000_000_000, "cuda"));
        let rr = RouteRequest {
            model_memory_bytes: 1_000_000_000,
            preferred_backend: Some("vulkan".into()),
            min_compute_capability: None,
        };
        let e = r.route(&rr).unwrap_err();
        assert_eq!(
            e,
            RouteError::NoMatchingBackend {
                requested: "vulkan".into()
            }
        );
    }

    #[test]
    fn round_robin_fallback() {
        let r = GpuRouter::new();
        r.register_device(dev("a", 8_000_000_000, "cuda"));
        r.register_device(dev("b", 8_000_000_000, "cuda"));
        let mut seen = std::collections::HashSet::new();
        for _ in 0..10 {
            let d = r.route(&req(1_000_000_000)).unwrap();
            seen.insert(d.device_id);
        }
        assert_eq!(seen.len(), 2, "round-robin should hit both");
    }

    #[test]
    fn load_tracking_shifts_preference() {
        let r = GpuRouter::new();
        r.register_device(dev("a", 8_000_000_000, "cuda"));
        r.register_device(dev("b", 8_000_000_000, "cuda"));
        for _ in 0..20 {
            r.begin_request("a");
        }
        let d = r.route(&req(1_000_000_000)).unwrap();
        assert_eq!(d.device_id, "b");
        for _ in 0..20 {
            r.end_request("a");
        }
    }

    #[test]
    fn memory_accounting_affects_routing() {
        let r = GpuRouter::new();
        r.register_device(dev("a", 8_000_000_000, "cuda"));
        r.register_device(dev("b", 8_000_000_000, "cuda"));
        r.update_memory_usage("a", 7_500_000_000);
        let d = r.route(&req(1_000_000_000)).unwrap();
        assert_eq!(d.device_id, "b");
    }

    #[test]
    fn health_recovery() {
        let r = GpuRouter::new();
        r.register_device(dev("g0", 8_000_000_000, "cuda"));
        r.set_device_health("g0", false);
        assert_eq!(
            r.route(&req(1_000_000_000)).unwrap_err(),
            RouteError::AllDevicesUnhealthy,
        );
        r.set_device_health("g0", true);
        assert!(r.route(&req(1_000_000_000)).is_ok());
    }

    #[test]
    fn free_memory_helper() {
        let r = GpuRouter::new();
        r.register_device(dev("g0", 8_000_000_000, "cuda"));
        assert_eq!(r.free_memory("g0"), Some(8_000_000_000));
        r.update_memory_usage("g0", 3_000_000_000);
        assert_eq!(r.free_memory("g0"), Some(5_000_000_000));
        assert_eq!(r.free_memory("nonexistent"), None);
    }
}
