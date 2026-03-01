//! GPU health monitoring and diagnostics.
//!
//! Provides [`HealthMonitor`] for running periodic health checks against GPU
//! devices, recording history snapshots, and generating diagnostic reports
//! with actionable recommendations.

// ── Core types ──────────────────────────────────────────────────────────────

/// Configuration for the health monitor.
#[derive(Debug, Clone)]
pub struct HealthConfig {
    /// Minimum interval between check runs, in milliseconds.
    pub check_interval_ms: u64,
    /// Maximum number of history snapshots to retain.
    pub max_history: usize,
    /// Whether to emit alerts when any check enters `Degraded` status.
    pub alert_on_degraded: bool,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self { check_interval_ms: 1000, max_history: 100, alert_on_degraded: true }
    }
}

/// The type of health check to evaluate.
#[derive(Debug, Clone, PartialEq)]
pub enum CheckType {
    /// Whether the device is reachable / available.
    DeviceAvailable,
    /// Memory-usage check — fires when usage exceeds `threshold_percent`.
    MemoryUsage { threshold_percent: f64 },
    /// Whether GPU kernels can be compiled successfully.
    KernelCompilation,
    /// Driver version compatibility check.
    DriverVersion,
    /// GPU temperature guard.
    Temperature { max_celsius: f64 },
    /// Command-queue depth guard.
    QueueDepth { max_depth: usize },
    /// Error-rate guard (errors per minute).
    ErrorRate { max_errors_per_minute: u32 },
    /// Latency guard (milliseconds).
    Latency { max_ms: f64 },
    /// User-defined check type.
    Custom(String),
}

/// Current health status of a single check or the overall monitor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthStatus {
    /// Everything is within acceptable limits.
    Healthy,
    /// Operating but outside ideal parameters.
    Degraded(String),
    /// Critical failure or limit exceeded.
    Unhealthy(String),
    /// Check has not yet been evaluated.
    Unknown,
}

impl HealthStatus {
    /// Returns a numeric severity used for ordering.
    const fn severity(&self) -> u8 {
        match self {
            Self::Degraded(_) => 1,
            Self::Unhealthy(_) => 2,
            Self::Healthy | Self::Unknown => 0,
        }
    }

    /// `true` when the status is `Healthy` or `Unknown`.
    pub const fn is_ok(&self) -> bool {
        matches!(self, Self::Healthy | Self::Unknown)
    }
}

/// A single named health check with its most recent result.
#[derive(Debug, Clone)]
pub struct HealthCheck {
    /// Human-readable name.
    pub name: String,
    /// What kind of check this is.
    pub check_type: CheckType,
    /// Most-recently evaluated status.
    pub status: HealthStatus,
    /// Timestamp (ms) of the last evaluation, if any.
    pub last_check_ms: Option<u64>,
    /// Optional explanatory message.
    pub message: Option<String>,
}

impl HealthCheck {
    /// Create a new check that starts in `Unknown` status.
    pub fn new(name: impl Into<String>, check_type: CheckType) -> Self {
        Self {
            name: name.into(),
            check_type,
            status: HealthStatus::Unknown,
            last_check_ms: None,
            message: None,
        }
    }

    /// Evaluate the check against a provided `value` and return the
    /// resulting [`HealthStatus`].  The check's internal status is
    /// **not** mutated — call [`HealthMonitor::run_checks`] for that.
    #[allow(clippy::too_many_lines)]
    pub fn evaluate(&self, value: f64) -> HealthStatus {
        match &self.check_type {
            CheckType::DeviceAvailable => {
                if value > 0.0 {
                    HealthStatus::Healthy
                } else {
                    HealthStatus::Unhealthy("device not available".into())
                }
            }
            CheckType::MemoryUsage { threshold_percent } => {
                if value <= *threshold_percent {
                    HealthStatus::Healthy
                } else if value <= threshold_percent + 10.0 {
                    HealthStatus::Degraded(format!(
                        "memory usage {value:.1}% exceeds {threshold_percent:.1}%"
                    ))
                } else {
                    HealthStatus::Unhealthy(format!(
                        "memory usage {value:.1}% critically exceeds \
                         {threshold_percent:.1}%"
                    ))
                }
            }
            CheckType::KernelCompilation => {
                if value > 0.0 {
                    HealthStatus::Healthy
                } else {
                    HealthStatus::Unhealthy("kernel compilation failed".into())
                }
            }
            CheckType::DriverVersion => {
                if value >= 1.0 {
                    HealthStatus::Healthy
                } else if value > 0.0 {
                    HealthStatus::Degraded("driver version outdated".into())
                } else {
                    HealthStatus::Unhealthy("driver not found".into())
                }
            }
            CheckType::Temperature { max_celsius } => {
                let warn_threshold = max_celsius * 0.85;
                if value <= warn_threshold {
                    HealthStatus::Healthy
                } else if value <= *max_celsius {
                    HealthStatus::Degraded(format!(
                        "temperature {value:.1}°C approaching limit \
                         {max_celsius:.1}°C"
                    ))
                } else {
                    HealthStatus::Unhealthy(format!(
                        "temperature {value:.1}°C exceeds limit \
                         {max_celsius:.1}°C"
                    ))
                }
            }
            CheckType::QueueDepth { max_depth } => {
                #[allow(clippy::cast_precision_loss)]
                let max_f = *max_depth as f64;
                let warn_threshold = max_f * 0.75;
                if value <= warn_threshold {
                    HealthStatus::Healthy
                } else if value <= max_f {
                    HealthStatus::Degraded(format!(
                        "queue depth {value:.0} approaching limit {max_depth}"
                    ))
                } else {
                    HealthStatus::Unhealthy(format!(
                        "queue depth {value:.0} exceeds limit {max_depth}"
                    ))
                }
            }
            CheckType::ErrorRate { max_errors_per_minute } => {
                let max_f = f64::from(*max_errors_per_minute);
                let warn_threshold = max_f * 0.5;
                if value <= warn_threshold {
                    HealthStatus::Healthy
                } else if value <= max_f {
                    HealthStatus::Degraded(format!(
                        "error rate {value:.0}/min approaching limit \
                         {max_errors_per_minute}/min"
                    ))
                } else {
                    HealthStatus::Unhealthy(format!(
                        "error rate {value:.0}/min exceeds limit \
                         {max_errors_per_minute}/min"
                    ))
                }
            }
            CheckType::Latency { max_ms } => {
                let warn_threshold = max_ms * 0.75;
                if value <= warn_threshold {
                    HealthStatus::Healthy
                } else if value <= *max_ms {
                    HealthStatus::Degraded(format!(
                        "latency {value:.1}ms approaching limit {max_ms:.1}ms"
                    ))
                } else {
                    HealthStatus::Unhealthy(format!(
                        "latency {value:.1}ms exceeds limit {max_ms:.1}ms"
                    ))
                }
            }
            CheckType::Custom(name) => {
                if value > 0.0 {
                    HealthStatus::Healthy
                } else {
                    HealthStatus::Unhealthy(format!("custom check '{name}' failed"))
                }
            }
        }
    }
}

// ── History ─────────────────────────────────────────────────────────────────

/// A point-in-time snapshot of every check's status.
#[derive(Debug, Clone)]
pub struct HealthSnapshot {
    /// Millisecond timestamp when the snapshot was taken.
    pub timestamp_ms: u64,
    /// Aggregate status (worst of all checks).
    pub overall_status: HealthStatus,
    /// Per-check name → status pairs.
    pub checks: Vec<(String, HealthStatus)>,
}

// ── Diagnostics ─────────────────────────────────────────────────────────────

/// Memory diagnostic data.
#[derive(Debug, Clone, Default)]
pub struct MemoryDiagnostics {
    /// Total device memory in bytes.
    pub total: u64,
    /// Currently used memory in bytes.
    pub used: u64,
    /// Free memory in bytes.
    pub free: u64,
    /// Fragmentation ratio in `[0.0, 1.0]`.
    pub fragmentation: f64,
    /// Largest contiguous free block in bytes.
    pub largest_free_block: u64,
}

/// A comprehensive diagnostic report.
#[derive(Debug, Clone)]
pub struct DiagnosticReport {
    /// Key-value pairs describing the device.
    pub device_info: Vec<(String, String)>,
    /// Key-value pairs describing the driver.
    pub driver_info: Vec<(String, String)>,
    /// Memory diagnostics snapshot.
    pub memory_info: MemoryDiagnostics,
    /// Clone of every check with its current status.
    pub check_results: Vec<HealthCheck>,
    /// Actionable recommendations.
    pub recommendations: Vec<String>,
}

// ── Monitor ─────────────────────────────────────────────────────────────────

/// Central health monitor that owns checks, records history, and generates
/// diagnostics.
#[derive(Debug)]
pub struct HealthMonitor {
    checks: Vec<HealthCheck>,
    history: Vec<HealthSnapshot>,
    config: HealthConfig,
}

impl HealthMonitor {
    /// Create a new monitor with the given configuration.
    pub const fn new(config: HealthConfig) -> Self {
        Self { checks: Vec::new(), history: Vec::new(), config }
    }

    /// Register a new health check.
    pub fn add_check(&mut self, check: HealthCheck) {
        self.checks.push(check);
    }

    /// Return a reference to all registered checks.
    pub fn checks(&self) -> &[HealthCheck] {
        &self.checks
    }

    /// Return a reference to the recorded history.
    pub fn history(&self) -> &[HealthSnapshot] {
        &self.history
    }

    /// Return a reference to the configuration.
    pub const fn config(&self) -> &HealthConfig {
        &self.config
    }

    /// Evaluate every check with the given `values` map (check-name → value),
    /// record a [`HealthSnapshot`], and return it.
    ///
    /// Checks whose name does not appear in `values` keep their current
    /// status and are still included in the snapshot.
    pub fn run_checks(&mut self, current_ms: u64, values: &[(&str, f64)]) -> HealthSnapshot {
        let value_map: std::collections::HashMap<&str, f64> = values.iter().copied().collect();

        for check in &mut self.checks {
            if let Some(&val) = value_map.get(check.name.as_str()) {
                check.status = check.evaluate(val);
                check.last_check_ms = Some(current_ms);
                check.message = match &check.status {
                    HealthStatus::Degraded(m) | HealthStatus::Unhealthy(m) => Some(m.clone()),
                    HealthStatus::Healthy | HealthStatus::Unknown => None,
                };
            }
        }

        let overall = self.overall_status();
        let snapshot = HealthSnapshot {
            timestamp_ms: current_ms,
            overall_status: overall,
            checks: self.checks.iter().map(|c| (c.name.clone(), c.status.clone())).collect(),
        };

        self.history.push(snapshot);
        if self.history.len() > self.config.max_history {
            self.history.drain(..self.history.len() - self.config.max_history);
        }

        self.history.last().unwrap().clone()
    }

    /// Compute the aggregate status — the *worst* severity across all
    /// checks.  Returns `Healthy` when there are no checks.
    pub fn overall_status(&self) -> HealthStatus {
        self.checks
            .iter()
            .map(|c| &c.status)
            .max_by_key(|s| s.severity())
            .cloned()
            .unwrap_or(HealthStatus::Healthy)
    }

    /// Generate a full diagnostic report with memory info and
    /// recommendations.
    pub fn generate_diagnostics(&self) -> DiagnosticReport {
        DiagnosticReport {
            device_info: vec![
                ("vendor".into(), "unknown".into()),
                ("model".into(), "unknown".into()),
            ],
            driver_info: vec![("version".into(), "unknown".into())],
            memory_info: MemoryDiagnostics::default(),
            check_results: self.checks.clone(),
            recommendations: self.recommendations(),
        }
    }

    /// Produce actionable recommendations based on the current check
    /// statuses.
    pub fn recommendations(&self) -> Vec<String> {
        let mut recs = Vec::new();
        for check in &self.checks {
            match (&check.check_type, &check.status) {
                (
                    CheckType::MemoryUsage { .. },
                    HealthStatus::Degraded(_) | HealthStatus::Unhealthy(_),
                ) => {
                    recs.push(
                        "Reduce memory usage or increase GPU memory \
                         allocation"
                            .into(),
                    );
                }
                (
                    CheckType::Temperature { .. },
                    HealthStatus::Degraded(_) | HealthStatus::Unhealthy(_),
                ) => {
                    recs.push("Check GPU cooling; consider reducing workload".into());
                }
                (CheckType::KernelCompilation, HealthStatus::Unhealthy(_)) => {
                    recs.push("Verify GPU driver and SDK installation".into());
                }
                (
                    CheckType::DriverVersion,
                    HealthStatus::Degraded(_) | HealthStatus::Unhealthy(_),
                ) => {
                    recs.push("Update GPU driver to latest version".into());
                }
                (CheckType::DeviceAvailable, HealthStatus::Unhealthy(_)) => {
                    recs.push("Verify GPU device is connected and powered on".into());
                }
                (
                    CheckType::ErrorRate { .. },
                    HealthStatus::Degraded(_) | HealthStatus::Unhealthy(_),
                ) => {
                    recs.push("Investigate error sources; check driver logs".into());
                }
                (
                    CheckType::Latency { .. },
                    HealthStatus::Degraded(_) | HealthStatus::Unhealthy(_),
                ) => {
                    recs.push("Check PCIe bandwidth; reduce concurrent workloads".into());
                }
                (
                    CheckType::QueueDepth { .. },
                    HealthStatus::Degraded(_) | HealthStatus::Unhealthy(_),
                ) => {
                    recs.push("Reduce command submission rate or batch commands".into());
                }
                _ => {}
            }
        }
        recs
    }

    /// Detect trend direction from history for a named check.
    ///
    /// Returns `Some(true)` if the check was previously non-healthy and is
    /// now healthy (improving), `Some(false)` if it was healthy and is now
    /// non-healthy (degrading), or `None` when there is insufficient data.
    pub fn trend(&self, check_name: &str) -> Option<bool> {
        if self.history.len() < 2 {
            return None;
        }
        let prev = self.history[self.history.len() - 2]
            .checks
            .iter()
            .find(|(n, _)| n == check_name)
            .map(|(_, s)| s);
        let curr = self
            .history
            .last()
            .and_then(|snap| snap.checks.iter().find(|(n, _)| n == check_name).map(|(_, s)| s));

        match (prev, curr) {
            (Some(p), Some(c)) => match p.severity().cmp(&c.severity()) {
                std::cmp::Ordering::Greater => Some(true),
                std::cmp::Ordering::Less => Some(false),
                std::cmp::Ordering::Equal => None,
            },
            _ => None,
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_monitor() -> HealthMonitor {
        HealthMonitor::new(HealthConfig::default())
    }

    // -- overall status aggregation ------------------------------------

    #[test]
    fn all_healthy_overall_healthy() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("a", CheckType::DeviceAvailable));
        mon.add_check(HealthCheck::new("b", CheckType::KernelCompilation));
        mon.run_checks(0, &[("a", 1.0), ("b", 1.0)]);
        assert_eq!(mon.overall_status(), HealthStatus::Healthy);
    }

    #[test]
    fn one_degraded_overall_degraded() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("mem", CheckType::MemoryUsage { threshold_percent: 80.0 }));
        mon.add_check(HealthCheck::new("dev", CheckType::DeviceAvailable));
        mon.run_checks(0, &[("mem", 85.0), ("dev", 1.0)]);
        assert!(matches!(mon.overall_status(), HealthStatus::Degraded(_)));
    }

    #[test]
    fn one_unhealthy_overall_unhealthy() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("dev", CheckType::DeviceAvailable));
        mon.add_check(HealthCheck::new("kern", CheckType::KernelCompilation));
        mon.run_checks(0, &[("dev", 0.0), ("kern", 1.0)]);
        assert!(matches!(mon.overall_status(), HealthStatus::Unhealthy(_)));
    }

    #[test]
    fn unhealthy_beats_degraded_in_overall() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("mem", CheckType::MemoryUsage { threshold_percent: 80.0 }));
        mon.add_check(HealthCheck::new("dev", CheckType::DeviceAvailable));
        mon.run_checks(0, &[("mem", 85.0), ("dev", 0.0)]);
        assert!(matches!(mon.overall_status(), HealthStatus::Unhealthy(_)));
    }

    // -- HealthCheck::evaluate -----------------------------------------

    #[test]
    fn device_available_healthy() {
        let c = HealthCheck::new("d", CheckType::DeviceAvailable);
        assert_eq!(c.evaluate(1.0), HealthStatus::Healthy);
    }

    #[test]
    fn device_available_unhealthy() {
        let c = HealthCheck::new("d", CheckType::DeviceAvailable);
        assert!(matches!(c.evaluate(0.0), HealthStatus::Unhealthy(_)));
    }

    #[test]
    fn memory_below_threshold_healthy() {
        let c = HealthCheck::new("m", CheckType::MemoryUsage { threshold_percent: 80.0 });
        assert_eq!(c.evaluate(50.0), HealthStatus::Healthy);
    }

    #[test]
    fn memory_at_threshold_healthy() {
        let c = HealthCheck::new("m", CheckType::MemoryUsage { threshold_percent: 80.0 });
        assert_eq!(c.evaluate(80.0), HealthStatus::Healthy);
    }

    #[test]
    fn memory_slightly_over_threshold_degraded() {
        let c = HealthCheck::new("m", CheckType::MemoryUsage { threshold_percent: 80.0 });
        assert!(matches!(c.evaluate(85.0), HealthStatus::Degraded(_)));
    }

    #[test]
    fn memory_far_over_threshold_unhealthy() {
        let c = HealthCheck::new("m", CheckType::MemoryUsage { threshold_percent: 80.0 });
        assert!(matches!(c.evaluate(95.0), HealthStatus::Unhealthy(_)));
    }

    #[test]
    fn temperature_cool_healthy() {
        let c = HealthCheck::new("t", CheckType::Temperature { max_celsius: 100.0 });
        assert_eq!(c.evaluate(60.0), HealthStatus::Healthy);
    }

    #[test]
    fn temperature_warm_degraded() {
        let c = HealthCheck::new("t", CheckType::Temperature { max_celsius: 100.0 });
        // 85% of 100 = 85; 90 > 85 → Degraded
        assert!(matches!(c.evaluate(90.0), HealthStatus::Degraded(_)));
    }

    #[test]
    fn temperature_hot_unhealthy() {
        let c = HealthCheck::new("t", CheckType::Temperature { max_celsius: 100.0 });
        assert!(matches!(c.evaluate(105.0), HealthStatus::Unhealthy(_)));
    }

    #[test]
    fn error_rate_zero_healthy() {
        let c = HealthCheck::new("e", CheckType::ErrorRate { max_errors_per_minute: 10 });
        assert_eq!(c.evaluate(0.0), HealthStatus::Healthy);
    }

    #[test]
    fn error_rate_moderate_degraded() {
        let c = HealthCheck::new("e", CheckType::ErrorRate { max_errors_per_minute: 10 });
        // 50% of 10 = 5; 7 > 5 → Degraded
        assert!(matches!(c.evaluate(7.0), HealthStatus::Degraded(_)));
    }

    #[test]
    fn error_rate_high_unhealthy() {
        let c = HealthCheck::new("e", CheckType::ErrorRate { max_errors_per_minute: 10 });
        assert!(matches!(c.evaluate(15.0), HealthStatus::Unhealthy(_)));
    }

    #[test]
    fn latency_low_healthy() {
        let c = HealthCheck::new("l", CheckType::Latency { max_ms: 100.0 });
        assert_eq!(c.evaluate(30.0), HealthStatus::Healthy);
    }

    #[test]
    fn latency_high_degraded() {
        let c = HealthCheck::new("l", CheckType::Latency { max_ms: 100.0 });
        // 75% of 100 = 75; 80 > 75 → Degraded
        assert!(matches!(c.evaluate(80.0), HealthStatus::Degraded(_)));
    }

    #[test]
    fn latency_over_limit_unhealthy() {
        let c = HealthCheck::new("l", CheckType::Latency { max_ms: 100.0 });
        assert!(matches!(c.evaluate(120.0), HealthStatus::Unhealthy(_)));
    }

    #[test]
    fn queue_depth_low_healthy() {
        let c = HealthCheck::new("q", CheckType::QueueDepth { max_depth: 100 });
        assert_eq!(c.evaluate(10.0), HealthStatus::Healthy);
    }

    #[test]
    fn queue_depth_high_degraded() {
        let c = HealthCheck::new("q", CheckType::QueueDepth { max_depth: 100 });
        // 75% of 100 = 75; 80 > 75 → Degraded
        assert!(matches!(c.evaluate(80.0), HealthStatus::Degraded(_)));
    }

    #[test]
    fn queue_depth_over_limit_unhealthy() {
        let c = HealthCheck::new("q", CheckType::QueueDepth { max_depth: 100 });
        assert!(matches!(c.evaluate(110.0), HealthStatus::Unhealthy(_)));
    }

    #[test]
    fn kernel_compilation_success_healthy() {
        let c = HealthCheck::new("k", CheckType::KernelCompilation);
        assert_eq!(c.evaluate(1.0), HealthStatus::Healthy);
    }

    #[test]
    fn kernel_compilation_failure_unhealthy() {
        let c = HealthCheck::new("k", CheckType::KernelCompilation);
        assert!(matches!(c.evaluate(0.0), HealthStatus::Unhealthy(_)));
    }

    #[test]
    fn driver_version_current_healthy() {
        let c = HealthCheck::new("dv", CheckType::DriverVersion);
        assert_eq!(c.evaluate(1.0), HealthStatus::Healthy);
    }

    #[test]
    fn driver_version_outdated_degraded() {
        let c = HealthCheck::new("dv", CheckType::DriverVersion);
        assert!(matches!(c.evaluate(0.5), HealthStatus::Degraded(_)));
    }

    #[test]
    fn driver_version_missing_unhealthy() {
        let c = HealthCheck::new("dv", CheckType::DriverVersion);
        assert!(matches!(c.evaluate(0.0), HealthStatus::Unhealthy(_)));
    }

    #[test]
    fn custom_check_pass() {
        let c = HealthCheck::new("x", CheckType::Custom("my-check".into()));
        assert_eq!(c.evaluate(1.0), HealthStatus::Healthy);
    }

    #[test]
    fn custom_check_fail() {
        let c = HealthCheck::new("x", CheckType::Custom("my-check".into()));
        assert!(matches!(c.evaluate(0.0), HealthStatus::Unhealthy(_)));
    }

    // -- unknown status ------------------------------------------------

    #[test]
    fn new_check_starts_unknown() {
        let c = HealthCheck::new("u", CheckType::DeviceAvailable);
        assert_eq!(c.status, HealthStatus::Unknown);
    }

    #[test]
    fn unchecked_monitor_overall_unknown() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("a", CheckType::DeviceAvailable));
        assert_eq!(mon.overall_status(), HealthStatus::Unknown);
    }

    // -- empty monitor -------------------------------------------------

    #[test]
    fn empty_monitor_overall_healthy() {
        let mon = default_monitor();
        assert_eq!(mon.overall_status(), HealthStatus::Healthy);
    }

    #[test]
    fn empty_monitor_no_history() {
        let mon = default_monitor();
        assert!(mon.history().is_empty());
    }

    #[test]
    fn empty_monitor_no_checks() {
        let mon = default_monitor();
        assert!(mon.checks().is_empty());
    }

    // -- run_checks / history ------------------------------------------

    #[test]
    fn run_checks_records_snapshot() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("a", CheckType::DeviceAvailable));
        mon.run_checks(100, &[("a", 1.0)]);
        assert_eq!(mon.history().len(), 1);
        assert_eq!(mon.history()[0].timestamp_ms, 100);
    }

    #[test]
    fn run_checks_updates_last_check_ms() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("a", CheckType::DeviceAvailable));
        mon.run_checks(42, &[("a", 1.0)]);
        assert_eq!(mon.checks()[0].last_check_ms, Some(42));
    }

    #[test]
    fn run_checks_preserves_unchecked_status() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("a", CheckType::DeviceAvailable));
        mon.add_check(HealthCheck::new("b", CheckType::KernelCompilation));
        mon.run_checks(0, &[("a", 1.0)]); // b not in values
        assert_eq!(mon.checks()[1].status, HealthStatus::Unknown);
    }

    #[test]
    fn run_checks_returns_snapshot() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("a", CheckType::DeviceAvailable));
        let snap = mon.run_checks(10, &[("a", 1.0)]);
        assert_eq!(snap.timestamp_ms, 10);
        assert_eq!(snap.overall_status, HealthStatus::Healthy);
    }

    #[test]
    fn max_history_enforced() {
        let mut mon =
            HealthMonitor::new(HealthConfig { max_history: 3, ..HealthConfig::default() });
        mon.add_check(HealthCheck::new("a", CheckType::DeviceAvailable));
        for i in 0..10 {
            mon.run_checks(i, &[("a", 1.0)]);
        }
        assert_eq!(mon.history().len(), 3);
        // Oldest surviving should be timestamp 7
        assert_eq!(mon.history()[0].timestamp_ms, 7);
    }

    #[test]
    fn multiple_check_updates() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("a", CheckType::DeviceAvailable));
        mon.run_checks(0, &[("a", 0.0)]);
        assert!(matches!(mon.overall_status(), HealthStatus::Unhealthy(_)));
        mon.run_checks(1, &[("a", 1.0)]);
        assert_eq!(mon.overall_status(), HealthStatus::Healthy);
    }

    // -- snapshot contents ---------------------------------------------

    #[test]
    fn snapshot_contains_all_checks() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("a", CheckType::DeviceAvailable));
        mon.add_check(HealthCheck::new("b", CheckType::KernelCompilation));
        let snap = mon.run_checks(0, &[("a", 1.0), ("b", 1.0)]);
        assert_eq!(snap.checks.len(), 2);
    }

    #[test]
    fn snapshot_check_names_match() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("alpha", CheckType::DeviceAvailable));
        let snap = mon.run_checks(0, &[("alpha", 1.0)]);
        assert_eq!(snap.checks[0].0, "alpha");
    }

    // -- diagnostics ---------------------------------------------------

    #[test]
    fn diagnostic_report_has_device_info() {
        let mon = default_monitor();
        let report = mon.generate_diagnostics();
        assert!(!report.device_info.is_empty());
    }

    #[test]
    fn diagnostic_report_has_driver_info() {
        let mon = default_monitor();
        let report = mon.generate_diagnostics();
        assert!(!report.driver_info.is_empty());
    }

    #[test]
    fn diagnostic_report_includes_checks() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("a", CheckType::DeviceAvailable));
        let report = mon.generate_diagnostics();
        assert_eq!(report.check_results.len(), 1);
    }

    #[test]
    fn diagnostic_report_default_memory() {
        let mon = default_monitor();
        let report = mon.generate_diagnostics();
        assert_eq!(report.memory_info.total, 0);
        assert_eq!(report.memory_info.free, 0);
    }

    // -- recommendations -----------------------------------------------

    #[test]
    fn no_recommendations_when_healthy() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("a", CheckType::DeviceAvailable));
        mon.run_checks(0, &[("a", 1.0)]);
        assert!(mon.recommendations().is_empty());
    }

    #[test]
    fn recommendation_for_memory_issue() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("m", CheckType::MemoryUsage { threshold_percent: 80.0 }));
        mon.run_checks(0, &[("m", 95.0)]);
        let recs = mon.recommendations();
        assert!(recs.iter().any(|r| r.contains("memory")));
    }

    #[test]
    fn recommendation_for_temperature_issue() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("t", CheckType::Temperature { max_celsius: 100.0 }));
        mon.run_checks(0, &[("t", 105.0)]);
        let recs = mon.recommendations();
        assert!(recs.iter().any(|r| r.contains("cooling")));
    }

    #[test]
    fn recommendation_for_kernel_compilation() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("k", CheckType::KernelCompilation));
        mon.run_checks(0, &[("k", 0.0)]);
        let recs = mon.recommendations();
        assert!(recs.iter().any(|r| r.contains("driver")));
    }

    #[test]
    fn recommendation_for_driver_version() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("dv", CheckType::DriverVersion));
        mon.run_checks(0, &[("dv", 0.5)]);
        let recs = mon.recommendations();
        assert!(recs.iter().any(|r| r.contains("Update")));
    }

    #[test]
    fn recommendation_for_device_unavailable() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("dev", CheckType::DeviceAvailable));
        mon.run_checks(0, &[("dev", 0.0)]);
        let recs = mon.recommendations();
        assert!(recs.iter().any(|r| r.contains("connected")));
    }

    #[test]
    fn recommendation_for_error_rate() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("e", CheckType::ErrorRate { max_errors_per_minute: 10 }));
        mon.run_checks(0, &[("e", 15.0)]);
        let recs = mon.recommendations();
        assert!(recs.iter().any(|r| r.contains("error")));
    }

    #[test]
    fn recommendation_for_latency() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("l", CheckType::Latency { max_ms: 100.0 }));
        mon.run_checks(0, &[("l", 120.0)]);
        let recs = mon.recommendations();
        assert!(recs.iter().any(|r| r.contains("PCIe")));
    }

    #[test]
    fn recommendation_for_queue_depth() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("q", CheckType::QueueDepth { max_depth: 100 }));
        mon.run_checks(0, &[("q", 110.0)]);
        let recs = mon.recommendations();
        assert!(recs.iter().any(|r| r.contains("batch")));
    }

    // -- trend detection -----------------------------------------------

    #[test]
    fn trend_improving() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("a", CheckType::DeviceAvailable));
        mon.run_checks(0, &[("a", 0.0)]); // unhealthy
        mon.run_checks(1, &[("a", 1.0)]); // healthy
        assert_eq!(mon.trend("a"), Some(true));
    }

    #[test]
    fn trend_degrading() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("a", CheckType::DeviceAvailable));
        mon.run_checks(0, &[("a", 1.0)]); // healthy
        mon.run_checks(1, &[("a", 0.0)]); // unhealthy
        assert_eq!(mon.trend("a"), Some(false));
    }

    #[test]
    fn trend_stable_returns_none() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("a", CheckType::DeviceAvailable));
        mon.run_checks(0, &[("a", 1.0)]);
        mon.run_checks(1, &[("a", 1.0)]);
        assert_eq!(mon.trend("a"), None);
    }

    #[test]
    fn trend_insufficient_history() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("a", CheckType::DeviceAvailable));
        mon.run_checks(0, &[("a", 1.0)]);
        assert_eq!(mon.trend("a"), None);
    }

    #[test]
    fn trend_unknown_check_name() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("a", CheckType::DeviceAvailable));
        mon.run_checks(0, &[("a", 1.0)]);
        mon.run_checks(1, &[("a", 1.0)]);
        assert_eq!(mon.trend("nonexistent"), None);
    }

    // -- HealthStatus helpers ------------------------------------------

    #[test]
    fn healthy_is_ok() {
        assert!(HealthStatus::Healthy.is_ok());
    }

    #[test]
    fn unknown_is_ok() {
        assert!(HealthStatus::Unknown.is_ok());
    }

    #[test]
    fn degraded_is_not_ok() {
        assert!(!HealthStatus::Degraded("x".into()).is_ok());
    }

    #[test]
    fn unhealthy_is_not_ok() {
        assert!(!HealthStatus::Unhealthy("x".into()).is_ok());
    }

    #[test]
    fn severity_ordering() {
        assert!(
            HealthStatus::Unhealthy("x".into()).severity()
                > HealthStatus::Degraded("x".into()).severity()
        );
        assert!(HealthStatus::Degraded("x".into()).severity() > HealthStatus::Healthy.severity());
    }

    // -- config --------------------------------------------------------

    #[test]
    fn default_config_values() {
        let cfg = HealthConfig::default();
        assert_eq!(cfg.check_interval_ms, 1000);
        assert_eq!(cfg.max_history, 100);
        assert!(cfg.alert_on_degraded);
    }

    #[test]
    fn custom_config_respected() {
        let cfg =
            HealthConfig { check_interval_ms: 500, max_history: 10, alert_on_degraded: false };
        let mon = HealthMonitor::new(cfg);
        assert_eq!(mon.config().check_interval_ms, 500);
        assert_eq!(mon.config().max_history, 10);
        assert!(!mon.config().alert_on_degraded);
    }

    // -- HealthCheck construction / message ----------------------------

    #[test]
    fn check_message_set_on_degraded() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("m", CheckType::MemoryUsage { threshold_percent: 80.0 }));
        mon.run_checks(0, &[("m", 85.0)]);
        assert!(mon.checks()[0].message.is_some());
    }

    #[test]
    fn check_message_none_when_healthy() {
        let mut mon = default_monitor();
        mon.add_check(HealthCheck::new("a", CheckType::DeviceAvailable));
        mon.run_checks(0, &[("a", 1.0)]);
        assert!(mon.checks()[0].message.is_none());
    }

    // -- MemoryDiagnostics --------------------------------------------

    #[test]
    fn memory_diagnostics_default() {
        let m = MemoryDiagnostics::default();
        assert_eq!(m.total, 0);
        assert_eq!(m.used, 0);
        assert_eq!(m.free, 0);
        assert!((m.fragmentation - 0.0).abs() < f64::EPSILON);
        assert_eq!(m.largest_free_block, 0);
    }

    #[test]
    fn memory_diagnostics_custom() {
        let m = MemoryDiagnostics {
            total: 1024,
            used: 512,
            free: 512,
            fragmentation: 0.25,
            largest_free_block: 256,
        };
        assert_eq!(m.total, 1024);
        assert_eq!(m.free, 512);
        assert!((m.fragmentation - 0.25).abs() < f64::EPSILON);
    }
}
