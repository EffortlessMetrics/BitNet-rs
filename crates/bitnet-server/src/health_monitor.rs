//! Server health monitoring and diagnostics.
//!
//! Track system health: memory, request rates, error rates,
//! model status, and readiness probes.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Overall health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Starting,
}

impl HealthStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
            Self::Unhealthy => "unhealthy",
            Self::Starting => "starting",
        }
    }

    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Healthy | Self::Degraded)
    }
}

/// Individual health check result.
#[derive(Debug, Clone)]
pub struct HealthCheck {
    pub name: String,
    pub status: HealthStatus,
    pub message: Option<String>,
    pub duration: Duration,
}

impl HealthCheck {
    pub fn ok(name: impl Into<String>, duration: Duration) -> Self {
        Self { name: name.into(), status: HealthStatus::Healthy, message: None, duration }
    }

    pub fn degraded(name: impl Into<String>, msg: impl Into<String>, duration: Duration) -> Self {
        Self {
            name: name.into(),
            status: HealthStatus::Degraded,
            message: Some(msg.into()),
            duration,
        }
    }

    pub fn unhealthy(name: impl Into<String>, msg: impl Into<String>, duration: Duration) -> Self {
        Self {
            name: name.into(),
            status: HealthStatus::Unhealthy,
            message: Some(msg.into()),
            duration,
        }
    }
}

/// Request rate tracker using a sliding window.
#[derive(Debug)]
pub struct RateTracker {
    window: Duration,
    timestamps: VecDeque<Instant>,
}

impl RateTracker {
    pub fn new(window: Duration) -> Self {
        Self { window, timestamps: VecDeque::new() }
    }

    pub fn record(&mut self) {
        let now = Instant::now();
        self.timestamps.push_back(now);
        self.prune(now);
    }

    fn prune(&mut self, now: Instant) {
        while let Some(&front) = self.timestamps.front() {
            if now.duration_since(front) > self.window {
                self.timestamps.pop_front();
            } else {
                break;
            }
        }
    }

    pub fn count(&mut self) -> usize {
        self.prune(Instant::now());
        self.timestamps.len()
    }

    pub fn rate_per_second(&mut self) -> f64 {
        let c = self.count();
        let secs = self.window.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        c as f64 / secs
    }
}

/// Health report aggregating multiple checks.
#[derive(Debug)]
pub struct HealthReport {
    pub checks: Vec<HealthCheck>,
    pub overall: HealthStatus,
    pub uptime: Duration,
}

impl HealthReport {
    pub fn from_checks(checks: Vec<HealthCheck>, uptime: Duration) -> Self {
        let overall = if checks.iter().any(|c| c.status == HealthStatus::Unhealthy) {
            HealthStatus::Unhealthy
        } else if checks.iter().any(|c| c.status == HealthStatus::Degraded) {
            HealthStatus::Degraded
        } else if checks.is_empty() {
            HealthStatus::Starting
        } else {
            HealthStatus::Healthy
        };
        Self { checks, overall, uptime }
    }

    pub fn is_ready(&self) -> bool {
        self.overall.is_ready()
    }

    pub fn summary(&self) -> String {
        format!(
            "status={}, checks={}, uptime={:.0}s",
            self.overall.as_str(),
            self.checks.len(),
            self.uptime.as_secs_f64(),
        )
    }
}

/// Error rate tracker.
#[derive(Debug, Default)]
pub struct ErrorTracker {
    pub total_requests: u64,
    pub total_errors: u64,
    pub last_error: Option<String>,
}

impl ErrorTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_success(&mut self) {
        self.total_requests += 1;
    }

    pub fn record_error(&mut self, msg: impl Into<String>) {
        self.total_requests += 1;
        self.total_errors += 1;
        self.last_error = Some(msg.into());
    }

    pub fn error_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        self.total_errors as f64 / self.total_requests as f64
    }

    pub fn success_rate(&self) -> f64 {
        1.0 - self.error_rate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status() {
        assert!(HealthStatus::Healthy.is_ready());
        assert!(HealthStatus::Degraded.is_ready());
        assert!(!HealthStatus::Unhealthy.is_ready());
        assert!(!HealthStatus::Starting.is_ready());
    }

    #[test]
    fn test_health_check_ok() {
        let check = HealthCheck::ok("model", Duration::from_millis(5));
        assert_eq!(check.status, HealthStatus::Healthy);
        assert!(check.message.is_none());
    }

    #[test]
    fn test_health_check_degraded() {
        let check = HealthCheck::degraded("memory", "high usage", Duration::from_millis(1));
        assert_eq!(check.status, HealthStatus::Degraded);
        assert_eq!(check.message.as_deref(), Some("high usage"));
    }

    #[test]
    fn test_health_report_healthy() {
        let checks =
            vec![HealthCheck::ok("a", Duration::ZERO), HealthCheck::ok("b", Duration::ZERO)];
        let report = HealthReport::from_checks(checks, Duration::from_mins(1));
        assert_eq!(report.overall, HealthStatus::Healthy);
        assert!(report.is_ready());
    }

    #[test]
    fn test_health_report_degraded() {
        let checks = vec![
            HealthCheck::ok("a", Duration::ZERO),
            HealthCheck::degraded("b", "slow", Duration::ZERO),
        ];
        let report = HealthReport::from_checks(checks, Duration::from_mins(1));
        assert_eq!(report.overall, HealthStatus::Degraded);
    }

    #[test]
    fn test_health_report_unhealthy() {
        let checks = vec![HealthCheck::unhealthy("model", "not loaded", Duration::ZERO)];
        let report = HealthReport::from_checks(checks, Duration::from_secs(10));
        assert_eq!(report.overall, HealthStatus::Unhealthy);
        assert!(!report.is_ready());
    }

    #[test]
    fn test_rate_tracker() {
        let mut tracker = RateTracker::new(Duration::from_mins(1));
        tracker.record();
        tracker.record();
        tracker.record();
        assert_eq!(tracker.count(), 3);
    }

    #[test]
    fn test_error_tracker_success() {
        let mut et = ErrorTracker::new();
        et.record_success();
        et.record_success();
        assert_eq!(et.total_requests, 2);
        assert_eq!(et.error_rate(), 0.0);
        assert_eq!(et.success_rate(), 1.0);
    }

    #[test]
    fn test_error_tracker_errors() {
        let mut et = ErrorTracker::new();
        et.record_success();
        et.record_error("timeout");
        assert_eq!(et.total_errors, 1);
        assert!((et.error_rate() - 0.5).abs() < 1e-6);
        assert_eq!(et.last_error.as_deref(), Some("timeout"));
    }

    #[test]
    fn test_report_summary() {
        let report = HealthReport::from_checks(vec![], Duration::from_mins(2));
        let s = report.summary();
        assert!(s.contains("starting"));
        assert!(s.contains("120"));
    }

    #[test]
    fn test_status_as_str() {
        assert_eq!(HealthStatus::Healthy.as_str(), "healthy");
        assert_eq!(HealthStatus::Unhealthy.as_str(), "unhealthy");
    }

    #[test]
    fn test_empty_error_tracker() {
        let et = ErrorTracker::new();
        assert_eq!(et.error_rate(), 0.0);
    }
}
