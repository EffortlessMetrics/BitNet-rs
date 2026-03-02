//! Model health check command for validating model files.

use std::path::PathBuf;

/// Health check status.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Healthy => write!(f, "HEALTHY"),
            Self::Degraded => write!(f, "DEGRADED"),
            Self::Unhealthy => write!(f, "UNHEALTHY"),
        }
    }
}

/// A single health check item.
#[derive(Debug, Clone)]
pub struct CheckItem {
    pub name: String,
    pub status: HealthStatus,
    pub detail: String,
}

impl CheckItem {
    pub fn pass(name: &str, detail: &str) -> Self {
        Self { name: name.to_string(), status: HealthStatus::Healthy, detail: detail.to_string() }
    }
    pub fn warn(name: &str, detail: &str) -> Self {
        Self { name: name.to_string(), status: HealthStatus::Degraded, detail: detail.to_string() }
    }
    pub fn fail(name: &str, detail: &str) -> Self {
        Self { name: name.to_string(), status: HealthStatus::Unhealthy, detail: detail.to_string() }
    }
}

/// Results of a health check.
#[derive(Debug, Clone)]
pub struct HealthReport {
    pub model_path: PathBuf,
    pub checks: Vec<CheckItem>,
    pub overall: HealthStatus,
}

impl HealthReport {
    pub fn new(path: PathBuf) -> Self {
        Self { model_path: path, checks: Vec::new(), overall: HealthStatus::Healthy }
    }

    pub fn add(&mut self, item: CheckItem) {
        if item.status == HealthStatus::Unhealthy {
            self.overall = HealthStatus::Unhealthy;
        } else if item.status == HealthStatus::Degraded && self.overall == HealthStatus::Healthy {
            self.overall = HealthStatus::Degraded;
        }
        self.checks.push(item);
    }

    pub fn passed_count(&self) -> usize {
        self.checks.iter().filter(|c| c.status == HealthStatus::Healthy).count()
    }

    pub fn failed_count(&self) -> usize {
        self.checks.iter().filter(|c| c.status == HealthStatus::Unhealthy).count()
    }

    pub fn degraded_count(&self) -> usize {
        self.checks.iter().filter(|c| c.status == HealthStatus::Degraded).count()
    }
}

/// Check if a file exists and has content.
pub fn check_file_exists(path: &std::path::Path) -> CheckItem {
    if path.exists() {
        if let Ok(meta) = std::fs::metadata(path) {
            if meta.len() > 0 {
                CheckItem::pass(
                    "file_exists",
                    &format!("{} ({} bytes)", path.display(), meta.len()),
                )
            } else {
                CheckItem::fail("file_exists", &format!("{} exists but is empty", path.display()))
            }
        } else {
            CheckItem::warn(
                "file_exists",
                &format!("{} exists but metadata unavailable", path.display()),
            )
        }
    } else {
        CheckItem::fail("file_exists", &format!("{} not found", path.display()))
    }
}

/// Check if file extension matches expected model format.
pub fn check_file_format(path: &std::path::Path) -> CheckItem {
    match path.extension().and_then(|e| e.to_str()) {
        Some("gguf") => CheckItem::pass("file_format", "GGUF format detected"),
        Some("safetensors") => CheckItem::pass("file_format", "SafeTensors format detected"),
        Some("bin") => CheckItem::warn("file_format", "PyTorch .bin format (may need conversion)"),
        Some("onnx") => CheckItem::warn("file_format", "ONNX format (limited support)"),
        Some(ext) => CheckItem::fail("file_format", &format!("Unknown extension: .{ext}")),
        None => CheckItem::fail("file_format", "No file extension"),
    }
}

/// Check file size against expected ranges.
pub fn check_file_size(path: &std::path::Path, min_mb: u64, max_mb: u64) -> CheckItem {
    match std::fs::metadata(path) {
        Ok(meta) => {
            let size_mb = meta.len() / (1024 * 1024);
            if size_mb >= min_mb && size_mb <= max_mb {
                CheckItem::pass(
                    "file_size",
                    &format!("{size_mb} MB (expected {min_mb}-{max_mb} MB)"),
                )
            } else if size_mb < min_mb {
                CheckItem::warn(
                    "file_size",
                    &format!("{size_mb} MB (smaller than expected {min_mb} MB)"),
                )
            } else {
                CheckItem::warn(
                    "file_size",
                    &format!("{size_mb} MB (larger than expected {max_mb} MB)"),
                )
            }
        }
        Err(e) => CheckItem::fail("file_size", &format!("Cannot read: {e}")),
    }
}

/// Format a health report for display.
pub fn format_health_report(report: &HealthReport) -> String {
    let mut out = format!("=== Health Check: {} ===\n", report.model_path.display());
    out.push_str(&format!("Overall: {}\n\n", report.overall));
    for check in &report.checks {
        let icon = match check.status {
            HealthStatus::Healthy => "✓",
            HealthStatus::Degraded => "⚠",
            HealthStatus::Unhealthy => "✗",
        };
        out.push_str(&format!("{icon} {}: {}\n", check.name, check.detail));
    }
    out.push_str(&format!(
        "\nSummary: {} passed, {} degraded, {} failed\n",
        report.passed_count(),
        report.degraded_count(),
        report.failed_count()
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status_display() {
        assert_eq!(format!("{}", HealthStatus::Healthy), "HEALTHY");
        assert_eq!(format!("{}", HealthStatus::Degraded), "DEGRADED");
        assert_eq!(format!("{}", HealthStatus::Unhealthy), "UNHEALTHY");
    }

    #[test]
    fn test_check_item_pass() {
        let c = CheckItem::pass("test", "ok");
        assert_eq!(c.status, HealthStatus::Healthy);
    }

    #[test]
    fn test_check_item_warn() {
        let c = CheckItem::warn("test", "meh");
        assert_eq!(c.status, HealthStatus::Degraded);
    }

    #[test]
    fn test_check_item_fail() {
        let c = CheckItem::fail("test", "bad");
        assert_eq!(c.status, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_report_new() {
        let r = HealthReport::new(PathBuf::from("test.gguf"));
        assert_eq!(r.overall, HealthStatus::Healthy);
        assert!(r.checks.is_empty());
    }

    #[test]
    fn test_report_add_pass() {
        let mut r = HealthReport::new(PathBuf::from("test.gguf"));
        r.add(CheckItem::pass("test", "ok"));
        assert_eq!(r.overall, HealthStatus::Healthy);
        assert_eq!(r.passed_count(), 1);
    }

    #[test]
    fn test_report_add_fail() {
        let mut r = HealthReport::new(PathBuf::from("test.gguf"));
        r.add(CheckItem::fail("test", "bad"));
        assert_eq!(r.overall, HealthStatus::Unhealthy);
        assert_eq!(r.failed_count(), 1);
    }

    #[test]
    fn test_report_degraded_not_override_unhealthy() {
        let mut r = HealthReport::new(PathBuf::from("test.gguf"));
        r.add(CheckItem::fail("a", "bad"));
        r.add(CheckItem::warn("b", "meh"));
        assert_eq!(r.overall, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_report_healthy_then_degraded() {
        let mut r = HealthReport::new(PathBuf::from("test.gguf"));
        r.add(CheckItem::pass("a", "ok"));
        r.add(CheckItem::warn("b", "meh"));
        assert_eq!(r.overall, HealthStatus::Degraded);
    }

    #[test]
    fn test_report_counts() {
        let mut r = HealthReport::new(PathBuf::from("test.gguf"));
        r.add(CheckItem::pass("a", "ok"));
        r.add(CheckItem::pass("b", "ok"));
        r.add(CheckItem::warn("c", "meh"));
        r.add(CheckItem::fail("d", "bad"));
        assert_eq!(r.passed_count(), 2);
        assert_eq!(r.degraded_count(), 1);
        assert_eq!(r.failed_count(), 1);
    }

    #[test]
    fn test_check_file_format_gguf() {
        let c = check_file_format(std::path::Path::new("model.gguf"));
        assert_eq!(c.status, HealthStatus::Healthy);
    }

    #[test]
    fn test_check_file_format_safetensors() {
        let c = check_file_format(std::path::Path::new("model.safetensors"));
        assert_eq!(c.status, HealthStatus::Healthy);
    }

    #[test]
    fn test_check_file_format_bin() {
        let c = check_file_format(std::path::Path::new("model.bin"));
        assert_eq!(c.status, HealthStatus::Degraded);
    }

    #[test]
    fn test_check_file_format_unknown() {
        let c = check_file_format(std::path::Path::new("model.xyz"));
        assert_eq!(c.status, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_check_file_format_no_ext() {
        let c = check_file_format(std::path::Path::new("modelfile"));
        assert_eq!(c.status, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_check_file_exists_missing() {
        let c = check_file_exists(std::path::Path::new("/nonexistent/model.gguf"));
        assert_eq!(c.status, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_check_file_size_missing() {
        let c = check_file_size(std::path::Path::new("/nonexistent"), 100, 1000);
        assert_eq!(c.status, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_format_report() {
        let mut r = HealthReport::new(PathBuf::from("test.gguf"));
        r.add(CheckItem::pass("format", "GGUF"));
        r.add(CheckItem::fail("size", "too small"));
        let out = format_health_report(&r);
        assert!(out.contains("test.gguf"));
        assert!(out.contains("UNHEALTHY"));
        assert!(out.contains("format"));
        assert!(out.contains("Summary"));
    }

    #[test]
    fn test_format_report_all_pass() {
        let mut r = HealthReport::new(PathBuf::from("good.gguf"));
        r.add(CheckItem::pass("a", "ok"));
        r.add(CheckItem::pass("b", "ok"));
        let out = format_health_report(&r);
        assert!(out.contains("HEALTHY"));
        assert!(out.contains("2 passed"));
    }

    #[test]
    fn test_health_status_equality() {
        assert_eq!(HealthStatus::Healthy, HealthStatus::Healthy);
        assert_ne!(HealthStatus::Healthy, HealthStatus::Unhealthy);
    }
}
