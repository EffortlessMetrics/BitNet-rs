//! Device health checking for GPU inference routing.

use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Health status
// ---------------------------------------------------------------------------

/// Health verdict for a single device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Degraded(String),
    Unhealthy(String),
}

impl HealthStatus {
    pub fn is_healthy(&self) -> bool {
        matches!(self, Self::Healthy)
    }

    pub fn is_usable(&self) -> bool {
        !matches!(self, Self::Unhealthy(_))
    }
}

// ---------------------------------------------------------------------------
// Per-device error tracking
// ---------------------------------------------------------------------------

/// Lightweight error-rate monitor for a single device.
#[derive(Debug)]
pub struct ErrorRateMonitor {
    total: AtomicU64,
    errors: AtomicU64,
}

impl ErrorRateMonitor {
    pub fn new() -> Self {
        Self { total: AtomicU64::new(0), errors: AtomicU64::new(0) }
    }

    pub fn record_success(&self) {
        self.total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_error(&self) {
        self.total.fetch_add(1, Ordering::Relaxed);
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    pub fn error_rate(&self) -> f64 {
        let t = self.total.load(Ordering::Relaxed);
        if t == 0 {
            return 0.0;
        }
        self.errors.load(Ordering::Relaxed) as f64 / t as f64
    }

    pub fn total(&self) -> u64 {
        self.total.load(Ordering::Relaxed)
    }

    pub fn errors(&self) -> u64 {
        self.errors.load(Ordering::Relaxed)
    }

    pub fn reset(&self) {
        self.total.store(0, Ordering::Relaxed);
        self.errors.store(0, Ordering::Relaxed);
    }
}

impl Default for ErrorRateMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// DeviceHealthChecker
// ---------------------------------------------------------------------------

/// Configuration thresholds for health evaluation.
#[derive(Debug, Clone)]
pub struct HealthConfig {
    /// Error-rate above which a device is considered degraded.
    pub degraded_error_rate: f64,
    /// Error-rate above which a device is unhealthy.
    pub unhealthy_error_rate: f64,
    /// Minimum available memory (MB) before marking degraded.
    pub min_memory_mb: u64,
    /// Memory below which the device is unhealthy.
    pub critical_memory_mb: u64,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            degraded_error_rate: 0.05,
            unhealthy_error_rate: 0.25,
            min_memory_mb: 256,
            critical_memory_mb: 64,
        }
    }
}

/// Checks health of logical GPU devices.
///
/// In production, the smoke test would allocate a small OpenCL
/// buffer and run a trivial kernel.  Here we provide a
/// configurable mock layer so that tests can run without hardware.
pub struct DeviceHealthChecker {
    config: HealthConfig,
    monitors: Vec<ErrorRateMonitor>,
    /// Per-device available memory (MB) – set externally or by a
    /// real runtime query.
    available_memory: Vec<AtomicU64>,
    /// Optional mock: returns `true` if the smoke test passes.
    mock_smoke: Option<Box<dyn Fn(usize) -> bool + Send + Sync>>,
}

impl DeviceHealthChecker {
    pub fn new(device_count: usize, config: HealthConfig) -> Self {
        let monitors = (0..device_count).map(|_| ErrorRateMonitor::new()).collect();
        let available_memory = (0..device_count).map(|_| AtomicU64::new(1024)).collect();
        Self { config, monitors, available_memory, mock_smoke: None }
    }

    /// Install a mock smoke-test function.
    pub fn with_mock_smoke<F>(mut self, f: F) -> Self
    where
        F: Fn(usize) -> bool + Send + Sync + 'static,
    {
        self.mock_smoke = Some(Box::new(f));
        self
    }

    /// Set the reported available memory for a device.
    pub fn set_available_memory(&self, device_id: usize, mb: u64) {
        if let Some(m) = self.available_memory.get(device_id) {
            m.store(mb, Ordering::Relaxed);
        }
    }

    /// Return the error-rate monitor for a device.
    pub fn monitor(&self, device_id: usize) -> Option<&ErrorRateMonitor> {
        self.monitors.get(device_id)
    }

    /// Evaluate the health of `device_id`.
    pub fn check_device(&self, device_id: usize) -> HealthStatus {
        // Smoke test.
        let smoke_ok = if let Some(ref f) = self.mock_smoke {
            f(device_id)
        } else {
            // Without real hardware, assume pass.
            true
        };
        if !smoke_ok {
            return HealthStatus::Unhealthy("smoke test failed".to_string());
        }

        // Memory check.
        let mem =
            self.available_memory.get(device_id).map(|m| m.load(Ordering::Relaxed)).unwrap_or(0);
        if mem < self.config.critical_memory_mb {
            return HealthStatus::Unhealthy(format!(
                "available memory {mem} MB below critical \
                 threshold {} MB",
                self.config.critical_memory_mb,
            ));
        }
        if mem < self.config.min_memory_mb {
            return HealthStatus::Degraded(format!(
                "available memory {mem} MB below minimum \
                 {} MB",
                self.config.min_memory_mb,
            ));
        }

        // Error rate.
        if let Some(mon) = self.monitors.get(device_id) {
            let rate = mon.error_rate();
            if rate >= self.config.unhealthy_error_rate {
                return HealthStatus::Unhealthy(format!(
                    "error rate {rate:.2} exceeds threshold \
                     {}",
                    self.config.unhealthy_error_rate,
                ));
            }
            if rate >= self.config.degraded_error_rate {
                return HealthStatus::Degraded(format!(
                    "error rate {rate:.2} exceeds degraded \
                     threshold {}",
                    self.config.degraded_error_rate,
                ));
            }
        }

        HealthStatus::Healthy
    }

    /// Check all devices and produce an aggregate report.
    pub fn check_all(&self) -> HealthReport {
        let statuses = (0..self.monitors.len()).map(|id| (id, self.check_device(id))).collect();
        HealthReport { statuses }
    }
}

// ---------------------------------------------------------------------------
// HealthReport
// ---------------------------------------------------------------------------

/// Aggregate health status of every device.
#[derive(Debug, Clone)]
pub struct HealthReport {
    pub statuses: Vec<(usize, HealthStatus)>,
}

impl HealthReport {
    /// Device IDs considered usable (Healthy or Degraded).
    pub fn usable_devices(&self) -> Vec<usize> {
        self.statuses.iter().filter(|(_, s)| s.is_usable()).map(|(id, _)| *id).collect()
    }

    /// All devices that are fully healthy.
    pub fn healthy_devices(&self) -> Vec<usize> {
        self.statuses.iter().filter(|(_, s)| s.is_healthy()).map(|(id, _)| *id).collect()
    }

    /// `true` if at least one device is usable.
    pub fn any_usable(&self) -> bool {
        self.statuses.iter().any(|(_, s)| s.is_usable())
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_rate_starts_at_zero() {
        let m = ErrorRateMonitor::new();
        assert_eq!(m.error_rate(), 0.0);
    }

    #[test]
    fn error_rate_computed_correctly() {
        let m = ErrorRateMonitor::new();
        for _ in 0..8 {
            m.record_success();
        }
        for _ in 0..2 {
            m.record_error();
        }
        let rate = m.error_rate();
        assert!((rate - 0.2).abs() < 1e-9);
    }

    #[test]
    fn healthy_when_all_good() {
        let hc = DeviceHealthChecker::new(1, HealthConfig::default());
        assert_eq!(hc.check_device(0), HealthStatus::Healthy);
    }

    #[test]
    fn unhealthy_on_smoke_failure() {
        let hc = DeviceHealthChecker::new(1, HealthConfig::default()).with_mock_smoke(|_| false);
        assert!(matches!(hc.check_device(0), HealthStatus::Unhealthy(_)));
    }

    #[test]
    fn degraded_on_low_memory() {
        let hc = DeviceHealthChecker::new(1, HealthConfig::default());
        // Between critical (64) and min (256).
        hc.set_available_memory(0, 128);
        assert!(matches!(hc.check_device(0), HealthStatus::Degraded(_)));
    }

    #[test]
    fn unhealthy_on_critical_memory() {
        let hc = DeviceHealthChecker::new(1, HealthConfig::default());
        hc.set_available_memory(0, 32);
        assert!(matches!(hc.check_device(0), HealthStatus::Unhealthy(_)));
    }

    #[test]
    fn unhealthy_on_high_error_rate() {
        let hc = DeviceHealthChecker::new(1, HealthConfig::default());
        let mon = hc.monitor(0).unwrap();
        // 3 errors out of 4 → 75 %
        mon.record_error();
        mon.record_error();
        mon.record_error();
        mon.record_success();
        assert!(matches!(hc.check_device(0), HealthStatus::Unhealthy(_)));
    }

    #[test]
    fn degraded_on_elevated_error_rate() {
        let hc = DeviceHealthChecker::new(1, HealthConfig::default());
        let mon = hc.monitor(0).unwrap();
        // 1 error out of 10 = 10 % → above degraded (5 %)
        for _ in 0..9 {
            mon.record_success();
        }
        mon.record_error();
        assert!(matches!(hc.check_device(0), HealthStatus::Degraded(_)));
    }

    #[test]
    fn check_all_produces_report() {
        let hc = DeviceHealthChecker::new(3, HealthConfig::default());
        hc.set_available_memory(1, 32); // unhealthy
        let report = hc.check_all();
        assert_eq!(report.statuses.len(), 3);
        assert_eq!(report.healthy_devices(), vec![0, 2]);
        assert_eq!(report.usable_devices(), vec![0, 2]);
        assert!(report.any_usable());
    }

    #[test]
    fn error_monitor_reset() {
        let m = ErrorRateMonitor::new();
        m.record_error();
        m.record_error();
        assert_eq!(m.errors(), 2);
        m.reset();
        assert_eq!(m.errors(), 0);
        assert_eq!(m.total(), 0);
    }
}
