//! Server health and readiness endpoints.
//!
//! Provides health status, readiness checks, and version information.

use std::time::{Duration, Instant};

/// Health status of a component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

impl HealthStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
            Self::Unhealthy => "unhealthy",
        }
    }

    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Healthy | Self::Degraded)
    }
}

/// Individual component check.
#[derive(Debug, Clone)]
pub struct ComponentCheck {
    pub name: String,
    pub status: HealthStatus,
    pub message: Option<String>,
    pub latency: Duration,
}

/// Health check result.
#[derive(Debug, Clone)]
pub struct HealthResponse {
    pub status: HealthStatus,
    pub version: String,
    pub uptime: Duration,
    pub components: Vec<ComponentCheck>,
}

impl HealthResponse {
    pub fn is_ready(&self) -> bool {
        self.status.is_ok() && self.components.iter().all(|c| c.status.is_ok())
    }

    pub fn unhealthy_components(&self) -> Vec<&ComponentCheck> {
        self.components.iter().filter(|c| !c.status.is_ok()).collect()
    }
}

/// Health checker that tracks server state.
#[derive(Debug)]
pub struct ServerHealthChecker {
    start_time: Instant,
    version: String,
    checks: Vec<Box<dyn HealthCheckFn>>,
}

/// Trait for custom health checks.
pub trait HealthCheckFn: std::fmt::Debug + Send + Sync {
    fn name(&self) -> &str;
    fn check(&self) -> ComponentCheck;
}

/// Simple check that always passes.
#[derive(Debug)]
pub struct AlwaysHealthy {
    name: String,
}

impl AlwaysHealthy {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string() }
    }
}

impl HealthCheckFn for AlwaysHealthy {
    fn name(&self) -> &str {
        &self.name
    }
    fn check(&self) -> ComponentCheck {
        ComponentCheck {
            name: self.name.clone(),
            status: HealthStatus::Healthy,
            message: None,
            latency: Duration::ZERO,
        }
    }
}

/// Model loaded check.
#[derive(Debug)]
pub struct ModelLoadedCheck {
    is_loaded: bool,
}

impl ModelLoadedCheck {
    pub fn new(loaded: bool) -> Self {
        Self { is_loaded: loaded }
    }
}

impl HealthCheckFn for ModelLoadedCheck {
    fn name(&self) -> &str {
        "model"
    }
    fn check(&self) -> ComponentCheck {
        let start = Instant::now();
        let status = if self.is_loaded { HealthStatus::Healthy } else { HealthStatus::Unhealthy };
        let msg = if self.is_loaded { None } else { Some("No model loaded".to_string()) };
        ComponentCheck { name: "model".into(), status, message: msg, latency: start.elapsed() }
    }
}

impl ServerHealthChecker {
    pub fn new(version: &str) -> Self {
        Self { start_time: Instant::now(), version: version.to_string(), checks: Vec::new() }
    }

    pub fn add_check(&mut self, check: Box<dyn HealthCheckFn>) {
        self.checks.push(check);
    }

    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    pub fn check(&self) -> HealthResponse {
        let components: Vec<_> = self.checks.iter().map(|c| c.check()).collect();
        let status = if components.iter().any(|c| c.status == HealthStatus::Unhealthy) {
            HealthStatus::Unhealthy
        } else if components.iter().any(|c| c.status == HealthStatus::Degraded) {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };

        HealthResponse { status, version: self.version.clone(), uptime: self.uptime(), components }
    }

    /// Process is alive.
    pub fn liveness(&self) -> bool {
        true
    }

    pub fn readiness(&self) -> bool {
        self.check().is_ready()
    }
}

/// Readiness gate that blocks until conditions are met.
#[derive(Debug)]
pub struct ReadinessGate {
    model_loaded: bool,
    tokenizer_loaded: bool,
}

impl Default for ReadinessGate {
    fn default() -> Self {
        Self::new()
    }
}

impl ReadinessGate {
    pub fn new() -> Self {
        Self { model_loaded: false, tokenizer_loaded: false }
    }

    pub fn set_model_loaded(&mut self) {
        self.model_loaded = true;
    }
    pub fn set_tokenizer_loaded(&mut self) {
        self.tokenizer_loaded = true;
    }

    pub fn is_ready(&self) -> bool {
        self.model_loaded && self.tokenizer_loaded
    }

    pub fn missing(&self) -> Vec<&'static str> {
        let mut m = Vec::new();
        if !self.model_loaded {
            m.push("model");
        }
        if !self.tokenizer_loaded {
            m.push("tokenizer");
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status_str() {
        assert_eq!(HealthStatus::Healthy.as_str(), "healthy");
        assert_eq!(HealthStatus::Unhealthy.as_str(), "unhealthy");
    }

    #[test]
    fn test_health_status_ok() {
        assert!(HealthStatus::Healthy.is_ok());
        assert!(HealthStatus::Degraded.is_ok());
        assert!(!HealthStatus::Unhealthy.is_ok());
    }

    #[test]
    fn test_health_checker_empty() {
        let checker = ServerHealthChecker::new("0.1.0");
        let resp = checker.check();
        assert_eq!(resp.status, HealthStatus::Healthy);
        assert!(resp.is_ready());
    }

    #[test]
    fn test_health_checker_with_checks() {
        let mut checker = ServerHealthChecker::new("0.1.0");
        checker.add_check(Box::new(AlwaysHealthy::new("test")));
        let resp = checker.check();
        assert_eq!(resp.status, HealthStatus::Healthy);
        assert_eq!(resp.components.len(), 1);
    }

    #[test]
    fn test_unhealthy_model() {
        let mut checker = ServerHealthChecker::new("0.1.0");
        checker.add_check(Box::new(ModelLoadedCheck::new(false)));
        let resp = checker.check();
        assert_eq!(resp.status, HealthStatus::Unhealthy);
        assert!(!resp.is_ready());
    }

    #[test]
    fn test_healthy_model() {
        let mut checker = ServerHealthChecker::new("0.1.0");
        checker.add_check(Box::new(ModelLoadedCheck::new(true)));
        assert!(checker.readiness());
    }

    #[test]
    fn test_liveness() {
        let checker = ServerHealthChecker::new("0.1.0");
        assert!(checker.liveness());
    }

    #[test]
    fn test_uptime() {
        let checker = ServerHealthChecker::new("0.1.0");
        std::thread::sleep(Duration::from_millis(10));
        assert!(checker.uptime() >= Duration::from_millis(5));
    }

    #[test]
    fn test_readiness_gate_new() {
        let gate = ReadinessGate::new();
        assert!(!gate.is_ready());
        assert_eq!(gate.missing().len(), 2);
    }

    #[test]
    fn test_readiness_gate_partial() {
        let mut gate = ReadinessGate::new();
        gate.set_model_loaded();
        assert!(!gate.is_ready());
        assert_eq!(gate.missing(), vec!["tokenizer"]);
    }

    #[test]
    fn test_readiness_gate_full() {
        let mut gate = ReadinessGate::new();
        gate.set_model_loaded();
        gate.set_tokenizer_loaded();
        assert!(gate.is_ready());
        assert!(gate.missing().is_empty());
    }

    #[test]
    fn test_unhealthy_components() {
        let mut checker = ServerHealthChecker::new("0.1.0");
        checker.add_check(Box::new(AlwaysHealthy::new("ok")));
        checker.add_check(Box::new(ModelLoadedCheck::new(false)));
        let resp = checker.check();
        assert_eq!(resp.unhealthy_components().len(), 1);
    }
}
