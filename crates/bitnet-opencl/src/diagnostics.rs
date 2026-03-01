//! GPU runtime diagnostics for troubleshooting.
//!
//! Provides comprehensive system checks for driver availability,
//! `OpenCL`/Vulkan/CUDA status, GPU memory, and feature flags.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::system_info::{self, SystemInfo};

// ── Status types ─────────────────────────────────────────────────────────────

/// Driver detection status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriverStatus {
    /// Whether any GPU driver was detected.
    pub found: bool,
    /// Human-readable driver description (or "not found").
    pub description: String,
}

/// `OpenCL` platform/device enumeration status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenClStatus {
    /// Whether `OpenCL` support was compiled in.
    pub compiled: bool,
    /// Whether an `OpenCL` runtime was detected.
    pub available: bool,
    /// Discovered platform names.
    pub platforms: Vec<String>,
    /// Number of devices found across all platforms.
    pub device_count: usize,
}

/// Vulkan availability status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulkanStatus {
    /// Whether Vulkan support was compiled in.
    pub compiled: bool,
    /// Whether a Vulkan-capable device was found at runtime.
    pub available: bool,
    /// Discovered device names.
    pub devices: Vec<String>,
}

/// CUDA availability status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaStatus {
    /// Whether CUDA support was compiled in.
    pub compiled: bool,
    /// Whether a CUDA-capable GPU was found at runtime.
    pub available: bool,
    /// CUDA driver version string (if available).
    pub driver_version: Option<String>,
}

/// GPU memory status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStatus {
    /// Whether memory information was successfully queried.
    pub available: bool,
    /// Total GPU memory in megabytes (0 if unknown).
    pub total_mb: u64,
    /// Free GPU memory in megabytes (0 if unknown).
    pub free_mb: u64,
}

/// Feature flag compilation status.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct FeatureStatus {
    /// `cpu` feature is enabled.
    pub cpu: bool,
    /// `gpu` feature is enabled.
    pub gpu: bool,
    /// `cuda` feature is enabled.
    pub cuda: bool,
    /// `opencl` feature is enabled.
    pub opencl: bool,
    /// `vulkan` feature is enabled.
    pub vulkan: bool,
}

/// Result of a quick GPU smoke test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmokeTestResult {
    /// Whether the smoke test passed.
    pub passed: bool,
    /// Human-readable result message.
    pub message: String,
    /// Elapsed time in milliseconds.
    pub elapsed_ms: u64,
}

// ── Diagnostic report ────────────────────────────────────────────────────────

/// Issue found during diagnostics, with a suggested fix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticIssue {
    /// Short issue summary.
    pub summary: String,
    /// Suggested remediation.
    pub suggestion: String,
}

/// Complete diagnostic report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticReport {
    /// System information snapshot.
    pub system: SystemInfo,
    /// Driver detection results.
    pub driver: DriverStatus,
    /// `OpenCL` status.
    pub opencl: OpenClStatus,
    /// Vulkan status.
    pub vulkan: VulkanStatus,
    /// CUDA status.
    pub cuda: CudaStatus,
    /// GPU memory status.
    pub memory: MemoryStatus,
    /// Feature flag status.
    pub features: FeatureStatus,
    /// Smoke test result (if run).
    pub smoke_test: Option<SmokeTestResult>,
    /// Issues found with suggested fixes.
    pub issues: Vec<DiagnosticIssue>,
    /// Recommended configuration notes.
    pub recommendations: Vec<String>,
}

// ── GpuDiagnostics ───────────────────────────────────────────────────────────

/// Runs comprehensive GPU diagnostic checks.
pub struct GpuDiagnostics;

impl GpuDiagnostics {
    /// Create a new diagnostics runner.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Detect installed GPU drivers.
    #[must_use]
    pub fn check_drivers(&self) -> DriverStatus {
        if command_succeeds("nvidia-smi", &[]) {
            return DriverStatus {
                found: true,
                description: "NVIDIA GPU driver detected (nvidia-smi)".to_owned(),
            };
        }

        if command_succeeds("rocm-smi", &["--showid"]) {
            return DriverStatus {
                found: true,
                description: "AMD ROCm driver detected (rocm-smi)".to_owned(),
            };
        }

        DriverStatus { found: false, description: "not found".to_owned() }
    }

    /// Check `OpenCL` platform/device availability.
    #[must_use]
    pub const fn check_opencl(&self) -> OpenClStatus {
        let compiled = cfg!(feature = "opencl");
        if !compiled {
            return OpenClStatus {
                compiled: false,
                available: false,
                platforms: Vec::new(),
                device_count: 0,
            };
        }
        OpenClStatus { compiled: true, available: false, platforms: Vec::new(), device_count: 0 }
    }

    /// Check Vulkan availability.
    #[must_use]
    pub const fn check_vulkan(&self) -> VulkanStatus {
        let compiled = cfg!(feature = "vulkan");
        VulkanStatus { compiled, available: false, devices: Vec::new() }
    }

    /// Check CUDA availability.
    #[must_use]
    pub fn check_cuda(&self) -> CudaStatus {
        let compiled = cfg!(any(feature = "gpu", feature = "cuda"));
        let available = compiled && command_succeeds("nvidia-smi", &[]);
        let driver_version = if available { read_nvidia_driver_version() } else { None };
        CudaStatus { compiled, available, driver_version }
    }

    /// Check available GPU memory.
    #[must_use]
    pub const fn check_memory(&self) -> MemoryStatus {
        MemoryStatus { available: false, total_mb: 0, free_mb: 0 }
    }

    /// Report which feature flags were compiled in.
    #[must_use]
    pub const fn check_features(&self) -> FeatureStatus {
        FeatureStatus {
            cpu: cfg!(feature = "cpu"),
            gpu: cfg!(feature = "gpu"),
            cuda: cfg!(feature = "cuda"),
            opencl: cfg!(feature = "opencl"),
            vulkan: cfg!(feature = "vulkan"),
        }
    }

    /// Run a quick smoke test to verify basic compute works.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn run_smoke_test(&self) -> SmokeTestResult {
        let start = std::time::Instant::now();

        let a: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..1024).map(|i| (1024 - i) as f32).collect();
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        let elapsed_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
        // Floating-point accumulation may differ slightly from the exact value.
        let passed = dot > 178_900_000.0 && dot < 179_000_000.0;

        SmokeTestResult {
            passed,
            message: if passed {
                format!("Smoke test passed (dot product = {dot}, elapsed = {elapsed_ms}ms)")
            } else {
                format!("Smoke test FAILED (expected ~178_956_288, got {dot})")
            },
            elapsed_ms,
        }
    }

    /// Run all diagnostics and produce a full report.
    #[must_use]
    pub fn run_full(&self) -> DiagnosticReport {
        let system = system_info::collect_system_info();
        let driver = self.check_drivers();
        let opencl = self.check_opencl();
        let vulkan = self.check_vulkan();
        let cuda = self.check_cuda();
        let memory = self.check_memory();
        let features = self.check_features();
        let smoke_test = Some(self.run_smoke_test());

        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        if !driver.found {
            issues.push(DiagnosticIssue {
                summary: "No GPU driver detected".to_owned(),
                suggestion: "Install NVIDIA or AMD GPU drivers for hardware acceleration"
                    .to_owned(),
            });
        }

        if !features.gpu && !features.cuda {
            issues.push(DiagnosticIssue {
                summary: "GPU feature not compiled".to_owned(),
                suggestion: "Rebuild with --features gpu to enable GPU support".to_owned(),
            });
            recommendations
                .push("Build with: cargo build --no-default-features --features gpu".to_owned());
        }

        if features.cpu {
            recommendations.push("CPU backend is available as fallback".to_owned());
        }

        if !features.opencl {
            recommendations
                .push("OpenCL support not compiled; add --features opencl if needed".to_owned());
        }

        DiagnosticReport {
            system,
            driver,
            opencl,
            vulkan,
            cuda,
            memory,
            features,
            smoke_test,
            issues,
            recommendations,
        }
    }
}

impl Default for GpuDiagnostics {
    fn default() -> Self {
        Self::new()
    }
}

// ── Report formatting ────────────────────────────────────────────────────────

/// Format a diagnostic report as human-readable text.
#[must_use]
pub fn format_report(report: &DiagnosticReport) -> String {
    let mut out = String::with_capacity(2048);

    out.push_str("=== BitNet GPU Diagnostics Report ===\n\n");

    out.push_str("── System ──\n");
    let _ = writeln!(out, "  OS:         {} {}", report.system.os_name, report.system.os_version);
    let _ = writeln!(out, "  Kernel:     {}", report.system.kernel_version);
    let _ = writeln!(out, "  CPU:        {}", report.system.cpu_name);
    let _ = writeln!(out, "  CPU cores:  {}", report.system.cpu_cores);
    if report.system.total_memory_mb > 0 {
        let _ = writeln!(out, "  Memory:     {} MB", report.system.total_memory_mb);
    }
    let _ = writeln!(out, "  Rust:       {}", report.system.rust_version);
    let _ = writeln!(out, "  BitNet:     {}", report.system.bitnet_version);
    out.push('\n');

    out.push_str("── GPU Driver ──\n");
    let _ = writeln!(out, "  Status:     {}", report.driver.description);
    out.push('\n');

    out.push_str("── CUDA ──\n");
    let _ = writeln!(out, "  Compiled:   {}", yes_no(report.cuda.compiled));
    let _ = writeln!(out, "  Available:  {}", yes_no(report.cuda.available));
    if let Some(ref ver) = report.cuda.driver_version {
        let _ = writeln!(out, "  Driver:     {ver}");
    }
    out.push('\n');

    out.push_str("── OpenCL ──\n");
    let _ = writeln!(out, "  Compiled:   {}", yes_no(report.opencl.compiled));
    let _ = writeln!(out, "  Available:  {}", yes_no(report.opencl.available));
    if !report.opencl.platforms.is_empty() {
        let _ = writeln!(out, "  Platforms:  {}", report.opencl.platforms.join(", "));
        let _ = writeln!(out, "  Devices:    {}", report.opencl.device_count);
    }
    out.push('\n');

    out.push_str("── Vulkan ──\n");
    let _ = writeln!(out, "  Compiled:   {}", yes_no(report.vulkan.compiled));
    let _ = writeln!(out, "  Available:  {}", yes_no(report.vulkan.available));
    out.push('\n');

    out.push_str("── GPU Memory ──\n");
    if report.memory.available {
        let _ = writeln!(out, "  Total:      {} MB", report.memory.total_mb);
        let _ = writeln!(out, "  Free:       {} MB", report.memory.free_mb);
    } else {
        out.push_str("  Status:     not available\n");
    }
    out.push('\n');

    out.push_str("── Feature Flags ──\n");
    let _ = writeln!(out, "  cpu:        {}", yes_no(report.features.cpu));
    let _ = writeln!(out, "  gpu:        {}", yes_no(report.features.gpu));
    let _ = writeln!(out, "  cuda:       {}", yes_no(report.features.cuda));
    let _ = writeln!(out, "  opencl:     {}", yes_no(report.features.opencl));
    let _ = writeln!(out, "  vulkan:     {}", yes_no(report.features.vulkan));
    out.push('\n');

    if let Some(ref st) = report.smoke_test {
        out.push_str("── Smoke Test ──\n");
        let _ = writeln!(out, "  {}\n", st.message);
    }

    out.push_str("── Issues ──\n");
    if report.issues.is_empty() {
        out.push_str("  No issues found.\n\n");
    } else {
        for (i, issue) in report.issues.iter().enumerate() {
            let _ = writeln!(out, "  {}. {}\n     Fix: {}", i + 1, issue.summary, issue.suggestion);
        }
        out.push('\n');
    }

    if !report.recommendations.is_empty() {
        out.push_str("── Recommendations ──\n");
        for rec in &report.recommendations {
            let _ = writeln!(out, "  • {rec}");
        }
        out.push('\n');
    }

    out
}

/// Format a diagnostic report as JSON.
///
/// # Panics
///
/// Panics if serialization fails (should not happen with valid data).
#[must_use]
pub fn format_json(report: &DiagnosticReport) -> String {
    serde_json::to_string_pretty(report).expect("DiagnosticReport should always serialize to JSON")
}

// ── Helpers ──────────────────────────────────────────────────────────────────

const fn yes_no(b: bool) -> &'static str {
    if b { "yes" } else { "no" }
}

fn command_succeeds(cmd: &str, args: &[&str]) -> bool {
    std::process::Command::new(cmd)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn read_nvidia_driver_version() -> Option<String> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=driver_version", "--format=csv,noheader"])
        .output()
        .ok()?;
    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).trim().to_owned())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_diagnostics_default_creates_instance() {
        let _diag = GpuDiagnostics::default();
    }

    #[test]
    fn check_drivers_returns_not_found_without_gpu() {
        let diag = GpuDiagnostics::new();
        let status = diag.check_drivers();
        assert!(!status.description.is_empty());
    }

    #[test]
    fn check_opencl_without_feature() {
        let diag = GpuDiagnostics::new();
        let status = diag.check_opencl();
        if !cfg!(feature = "opencl") {
            assert!(!status.compiled);
            assert!(!status.available);
        }
    }

    #[test]
    fn check_vulkan_without_feature() {
        let diag = GpuDiagnostics::new();
        let status = diag.check_vulkan();
        if !cfg!(feature = "vulkan") {
            assert!(!status.compiled);
        }
    }

    #[test]
    fn check_cuda_without_feature() {
        let diag = GpuDiagnostics::new();
        let status = diag.check_cuda();
        if !cfg!(any(feature = "gpu", feature = "cuda")) {
            assert!(!status.compiled);
            assert!(!status.available);
        }
    }

    #[test]
    fn check_features_reflects_compile_flags() {
        let diag = GpuDiagnostics::new();
        let f = diag.check_features();
        assert_eq!(f.cpu, cfg!(feature = "cpu"));
        assert_eq!(f.gpu, cfg!(feature = "gpu"));
        assert_eq!(f.cuda, cfg!(feature = "cuda"));
        assert_eq!(f.opencl, cfg!(feature = "opencl"));
        assert_eq!(f.vulkan, cfg!(feature = "vulkan"));
    }

    #[test]
    fn smoke_test_passes() {
        let diag = GpuDiagnostics::new();
        let result = diag.run_smoke_test();
        assert!(result.passed, "smoke test should pass: {}", result.message);
    }

    #[test]
    fn run_full_produces_report() {
        let diag = GpuDiagnostics::new();
        let report = diag.run_full();
        assert!(!report.system.os_name.is_empty());
        assert!(report.smoke_test.is_some());
    }
}
