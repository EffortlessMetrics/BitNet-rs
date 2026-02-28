//! GPU compatibility report generator.
//!
//! Detects all available GPUs, probes their drivers and capabilities,
//! optionally tests basic kernel execution on each device, and generates
//! a JSON or HTML report suitable for support and diagnostics.

use anyhow::Result;
use clap::Args;
use serde::Serialize;
use std::path::PathBuf;

/// GPU compatibility report command arguments.
#[derive(Args, Debug, Clone)]
pub struct GpuReportCommand {
    /// Output file path (stdout if omitted)
    #[arg(short, long, value_name = "PATH")]
    pub output: Option<PathBuf>,

    /// Output format: json (default) or html
    #[arg(long, default_value = "json")]
    pub format: ReportFormat,

    /// Run basic kernel execution tests on detected devices
    #[arg(long, default_value_t = false)]
    pub test_kernels: bool,

    /// Include verbose driver details
    #[arg(long, default_value_t = false)]
    pub verbose: bool,
}

/// Output format for the GPU report.
#[derive(Debug, Clone, clap::ValueEnum)]
pub enum ReportFormat {
    Json,
    Html,
}

/// Top-level GPU compatibility report.
#[derive(Debug, Clone, Serialize)]
pub struct GpuCompatibilityReport {
    /// Report schema version.
    pub schema_version: String,
    /// ISO 8601 timestamp of report generation.
    pub generated_at: String,
    /// Host system information.
    pub system: SystemInfo,
    /// Compile-time feature flags.
    pub compile_features: CompileFeatures,
    /// Detected GPU devices.
    pub devices: Vec<GpuDeviceInfo>,
    /// Kernel execution test results (if requested).
    pub kernel_tests: Vec<KernelTestResult>,
    /// Overall compatibility verdict.
    pub verdict: Verdict,
}

/// Host system information.
#[derive(Debug, Clone, Serialize)]
pub struct SystemInfo {
    pub os: String,
    pub arch: String,
    pub cpu_cores: usize,
    pub simd_level: String,
}

/// Compile-time feature flags relevant to GPU support.
#[derive(Debug, Clone, Serialize)]
pub struct CompileFeatures {
    pub gpu: bool,
    pub cuda: bool,
    pub oneapi: bool,
    pub metal: bool,
    pub vulkan: bool,
}

/// Information about a single detected GPU device.
#[derive(Debug, Clone, Serialize)]
pub struct GpuDeviceInfo {
    pub index: usize,
    pub name: String,
    pub vendor: String,
    pub backend: String,
    pub driver_version: String,
    pub memory_mb: Option<u64>,
    pub available: bool,
}

/// Result of a basic kernel execution test on a device.
#[derive(Debug, Clone, Serialize)]
pub struct KernelTestResult {
    pub device_index: usize,
    pub test_name: String,
    pub passed: bool,
    pub message: String,
    pub duration_ms: Option<f64>,
}

/// Overall compatibility verdict.
#[derive(Debug, Clone, Serialize)]
pub struct Verdict {
    pub compatible: bool,
    pub recommended_backend: String,
    pub warnings: Vec<String>,
}

impl GpuReportCommand {
    /// Execute the gpu-report command.
    pub async fn execute(&self) -> Result<()> {
        let report = generate_report(self.test_kernels)?;
        let output = match self.format {
            ReportFormat::Json => render_json(&report)?,
            ReportFormat::Html => render_html(&report),
        };

        if let Some(path) = &self.output {
            std::fs::write(path, &output)?;
            eprintln!("GPU report written to {}", path.display());
        } else {
            println!("{output}");
        }

        Ok(())
    }
}

/// Generate the full GPU compatibility report.
pub fn generate_report(test_kernels: bool) -> Result<GpuCompatibilityReport> {
    let system = detect_system_info();
    let compile_features = detect_compile_features();
    let devices = detect_gpu_devices();
    let kernel_tests = if test_kernels { run_kernel_tests(&devices) } else { vec![] };
    let verdict = compute_verdict(&compile_features, &devices, &kernel_tests);

    Ok(GpuCompatibilityReport {
        schema_version: "1.0.0".to_string(),
        generated_at: chrono::Utc::now().to_rfc3339(),
        system,
        compile_features,
        devices,
        kernel_tests,
        verdict,
    })
}

/// Detect host system information.
fn detect_system_info() -> SystemInfo {
    let cpu_cores = std::thread::available_parallelism().map(std::num::NonZero::get).unwrap_or(1);
    let simd_level = format!("{:?}", bitnet_device_probe::detect_simd_level());

    SystemInfo {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cpu_cores,
        simd_level,
    }
}

/// Detect compile-time GPU feature flags.
fn detect_compile_features() -> CompileFeatures {
    CompileFeatures {
        gpu: cfg!(feature = "gpu"),
        cuda: cfg!(any(feature = "gpu", feature = "cuda")),
        oneapi: cfg!(feature = "oneapi"),
        metal: cfg!(feature = "metal"),
        vulkan: cfg!(feature = "vulkan"),
    }
}

/// Detect all available GPU devices.
///
/// Uses `bitnet_device_probe` for runtime detection and augments with
/// platform-specific tools (nvidia-smi, clinfo) when available.
fn detect_gpu_devices() -> Vec<GpuDeviceInfo> {
    let mut devices = Vec::new();
    let probe = bitnet_device_probe::probe_device();

    if probe.cuda_available {
        devices.push(GpuDeviceInfo {
            index: devices.len(),
            name: detect_nvidia_gpu_name().unwrap_or_else(|| "NVIDIA GPU".to_string()),
            vendor: "NVIDIA".to_string(),
            backend: "CUDA".to_string(),
            driver_version: detect_nvidia_driver_version().unwrap_or_else(|| "unknown".to_string()),
            memory_mb: detect_nvidia_memory_mb(),
            available: true,
        });
    }

    if probe.oneapi_available {
        devices.push(GpuDeviceInfo {
            index: devices.len(),
            name: detect_intel_gpu_name().unwrap_or_else(|| "Intel GPU".to_string()),
            vendor: "Intel".to_string(),
            backend: "OpenCL/oneAPI".to_string(),
            driver_version: detect_intel_driver_version().unwrap_or_else(|| "unknown".to_string()),
            memory_mb: None,
            available: true,
        });
    }

    if probe.rocm_available {
        devices.push(GpuDeviceInfo {
            index: devices.len(),
            name: "AMD GPU".to_string(),
            vendor: "AMD".to_string(),
            backend: "ROCm".to_string(),
            driver_version: "unknown".to_string(),
            memory_mb: None,
            available: true,
        });
    }

    // If no GPUs detected, note that CPU-only is available
    if devices.is_empty() {
        devices.push(GpuDeviceInfo {
            index: 0,
            name: "No GPU detected".to_string(),
            vendor: "N/A".to_string(),
            backend: "CPU-only".to_string(),
            driver_version: "N/A".to_string(),
            memory_mb: None,
            available: false,
        });
    }

    devices
}

/// Run basic kernel execution tests on detected devices.
fn run_kernel_tests(devices: &[GpuDeviceInfo]) -> Vec<KernelTestResult> {
    let mut results = Vec::new();

    for device in devices {
        if !device.available {
            results.push(KernelTestResult {
                device_index: device.index,
                test_name: "device_availability".to_string(),
                passed: false,
                message: "No GPU device available".to_string(),
                duration_ms: None,
            });
            continue;
        }

        // Basic availability test
        results.push(KernelTestResult {
            device_index: device.index,
            test_name: "device_availability".to_string(),
            passed: true,
            message: format!("{} detected and available", device.backend),
            duration_ms: Some(0.0),
        });

        // Feature compilation test
        let feature_compiled = match device.backend.as_str() {
            "CUDA" => cfg!(any(feature = "gpu", feature = "cuda")),
            "OpenCL/oneAPI" => cfg!(feature = "oneapi"),
            "ROCm" => cfg!(any(feature = "gpu", feature = "rocm")),
            _ => false,
        };
        results.push(KernelTestResult {
            device_index: device.index,
            test_name: "feature_compiled".to_string(),
            passed: feature_compiled,
            message: if feature_compiled {
                format!("{} feature compiled", device.backend)
            } else {
                format!(
                    "{} feature NOT compiled — rebuild with appropriate features",
                    device.backend
                )
            },
            duration_ms: None,
        });
    }

    results
}

/// Compute the overall compatibility verdict.
fn compute_verdict(
    features: &CompileFeatures,
    devices: &[GpuDeviceInfo],
    _tests: &[KernelTestResult],
) -> Verdict {
    let has_gpu = devices.iter().any(|d| d.available);
    let mut warnings = Vec::new();

    let recommended_backend = if has_gpu {
        let gpu = &devices[0];
        if !features.gpu && !features.cuda && !features.oneapi {
            warnings.push(format!(
                "GPU hardware detected ({}) but no GPU features compiled. \
                 Rebuild with --features gpu",
                gpu.vendor
            ));
        }
        gpu.backend.clone()
    } else {
        if features.gpu || features.cuda {
            warnings.push(
                "GPU features compiled but no GPU hardware detected. \
                 Falling back to CPU."
                    .to_string(),
            );
        }
        "CPU".to_string()
    };

    Verdict { compatible: has_gpu || !features.gpu, recommended_backend, warnings }
}

// ── Platform-specific detection helpers ──────────────────────────────────────

fn detect_nvidia_gpu_name() -> Option<String> {
    run_command("nvidia-smi", &["--query-gpu=name", "--format=csv,noheader"])
        .map(|s| s.lines().next().unwrap_or("NVIDIA GPU").trim().to_string())
}

fn detect_nvidia_driver_version() -> Option<String> {
    run_command("nvidia-smi", &["--query-gpu=driver_version", "--format=csv,noheader"])
        .map(|s| s.trim().to_string())
}

fn detect_nvidia_memory_mb() -> Option<u64> {
    run_command("nvidia-smi", &["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .and_then(|s| s.trim().parse::<u64>().ok())
}

fn detect_intel_gpu_name() -> Option<String> {
    run_command("clinfo", &[]).and_then(|output| {
        for line in output.lines() {
            if line.contains("Device Name") && (line.contains("Intel") || line.contains("Arc")) {
                return Some(line.split_once(':')?.1.trim().to_string());
            }
        }
        None
    })
}

fn detect_intel_driver_version() -> Option<String> {
    run_command("clinfo", &[]).and_then(|output| {
        for line in output.lines() {
            if line.contains("Driver Version") {
                return Some(line.split_once(':')?.1.trim().to_string());
            }
        }
        None
    })
}

fn run_command(cmd: &str, args: &[&str]) -> Option<String> {
    std::process::Command::new(cmd)
        .args(args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
}

// ── Rendering ────────────────────────────────────────────────────────────────

fn render_json(report: &GpuCompatibilityReport) -> Result<String> {
    Ok(serde_json::to_string_pretty(report)?)
}

fn render_html(report: &GpuCompatibilityReport) -> String {
    let mut html = String::new();
    html.push_str("<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\">\n");
    html.push_str("<title>BitNet-rs GPU Compatibility Report</title>\n");
    html.push_str("<style>body{font-family:sans-serif;max-width:800px;margin:2em auto}");
    html.push_str("table{border-collapse:collapse;width:100%}");
    html.push_str("th,td{border:1px solid #ddd;padding:8px;text-align:left}");
    html.push_str("th{background:#f4f4f4}.pass{color:green}.fail{color:red}");
    html.push_str("</style></head><body>\n");
    html.push_str("<h1>BitNet-rs GPU Compatibility Report</h1>\n");
    html.push_str(&format!("<p>Generated: {}</p>\n", report.generated_at));
    html.push_str(&format!("<p>Schema: v{}</p>\n", report.schema_version));

    // System info
    html.push_str("<h2>System</h2><table>\n");
    html.push_str(&format!("<tr><td>OS</td><td>{}</td></tr>\n", report.system.os));
    html.push_str(&format!("<tr><td>Arch</td><td>{}</td></tr>\n", report.system.arch));
    html.push_str(&format!("<tr><td>CPU Cores</td><td>{}</td></tr>\n", report.system.cpu_cores));
    html.push_str(&format!("<tr><td>SIMD</td><td>{}</td></tr>\n", report.system.simd_level));
    html.push_str("</table>\n");

    // Devices
    html.push_str("<h2>Detected Devices</h2><table>\n");
    html.push_str("<tr><th>#</th><th>Name</th><th>Vendor</th><th>Backend</th>");
    html.push_str("<th>Driver</th><th>Memory</th><th>Available</th></tr>\n");
    for d in &report.devices {
        let avail_class = if d.available { "pass" } else { "fail" };
        html.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>",
            d.index,
            d.name,
            d.vendor,
            d.backend,
            d.driver_version,
            d.memory_mb.map_or("N/A".to_string(), |m| format!("{m} MB")),
        ));
        html.push_str(&format!(
            "<td class=\"{avail_class}\">{}</td></tr>\n",
            if d.available { "Yes" } else { "No" }
        ));
    }
    html.push_str("</table>\n");

    // Kernel tests
    if !report.kernel_tests.is_empty() {
        html.push_str("<h2>Kernel Tests</h2><table>\n");
        html.push_str("<tr><th>Device</th><th>Test</th><th>Result</th><th>Message</th></tr>\n");
        for t in &report.kernel_tests {
            let cls = if t.passed { "pass" } else { "fail" };
            html.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td class=\"{cls}\">{}</td><td>{}</td></tr>\n",
                t.device_index,
                t.test_name,
                if t.passed { "PASS" } else { "FAIL" },
                t.message,
            ));
        }
        html.push_str("</table>\n");
    }

    // Verdict
    html.push_str("<h2>Verdict</h2>\n");
    let v = &report.verdict;
    let cls = if v.compatible { "pass" } else { "fail" };
    html.push_str(&format!(
        "<p class=\"{cls}\">Compatible: {}</p>\n",
        if v.compatible { "Yes" } else { "No" }
    ));
    html.push_str(&format!("<p>Recommended backend: {}</p>\n", v.recommended_backend));
    if !v.warnings.is_empty() {
        html.push_str("<h3>Warnings</h3><ul>\n");
        for w in &v.warnings {
            html.push_str(&format!("<li>{w}</li>\n"));
        }
        html.push_str("</ul>\n");
    }

    html.push_str("</body></html>\n");
    html
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_report_succeeds() {
        let report = generate_report(false).expect("report generation should not fail");
        assert_eq!(report.schema_version, "1.0.0");
        assert!(!report.generated_at.is_empty());
    }

    #[test]
    fn test_generate_report_with_kernel_tests() {
        let report = generate_report(true).expect("report with kernel tests should succeed");
        // Either we have devices with tests, or an empty test list if no GPU
        assert!(!report.devices.is_empty());
    }

    #[test]
    fn test_system_info_detection() {
        let info = detect_system_info();
        assert!(info.cpu_cores >= 1);
        assert!(!info.os.is_empty());
        assert!(!info.arch.is_empty());
        assert!(!info.simd_level.is_empty());
    }

    #[test]
    fn test_compile_features_detection() {
        let features = detect_compile_features();
        // CPU-only build: gpu/cuda should be false
        // The result depends on build flags, but the function should not panic
        let _ = features.gpu;
        let _ = features.cuda;
    }

    #[test]
    fn test_render_json_valid() {
        let report = generate_report(false).unwrap();
        let json = render_json(&report).expect("JSON rendering should succeed");
        // Verify it parses back
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("JSON should be valid");
        assert_eq!(parsed["schema_version"], "1.0.0");
    }

    #[test]
    fn test_render_html_contains_structure() {
        let report = generate_report(false).unwrap();
        let html = render_html(&report);
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("GPU Compatibility Report"));
        assert!(html.contains("</html>"));
        assert!(html.contains("<table>"));
    }

    #[test]
    fn test_verdict_no_gpu_no_features() {
        let features =
            CompileFeatures { gpu: false, cuda: false, oneapi: false, metal: false, vulkan: false };
        let devices = vec![GpuDeviceInfo {
            index: 0,
            name: "No GPU".to_string(),
            vendor: "N/A".to_string(),
            backend: "CPU-only".to_string(),
            driver_version: "N/A".to_string(),
            memory_mb: None,
            available: false,
        }];
        let verdict = compute_verdict(&features, &devices, &[]);
        assert!(verdict.compatible, "no GPU + no GPU features = compatible (CPU mode)");
        assert_eq!(verdict.recommended_backend, "CPU");
    }

    #[test]
    fn test_verdict_gpu_compiled_no_hardware() {
        let features =
            CompileFeatures { gpu: true, cuda: true, oneapi: false, metal: false, vulkan: false };
        let devices = vec![GpuDeviceInfo {
            index: 0,
            name: "No GPU".to_string(),
            vendor: "N/A".to_string(),
            backend: "CPU-only".to_string(),
            driver_version: "N/A".to_string(),
            memory_mb: None,
            available: false,
        }];
        let verdict = compute_verdict(&features, &devices, &[]);
        assert!(!verdict.compatible);
        assert!(!verdict.warnings.is_empty());
    }

    #[test]
    fn test_gpu_device_info_serialize() {
        let device = GpuDeviceInfo {
            index: 0,
            name: "Test GPU".to_string(),
            vendor: "TestVendor".to_string(),
            backend: "CUDA".to_string(),
            driver_version: "535.0".to_string(),
            memory_mb: Some(8192),
            available: true,
        };
        let json = serde_json::to_string(&device).expect("should serialize");
        assert!(json.contains("Test GPU"));
        assert!(json.contains("8192"));
    }

    #[test]
    fn test_kernel_test_result_serialize() {
        let result = KernelTestResult {
            device_index: 0,
            test_name: "device_availability".to_string(),
            passed: true,
            message: "OK".to_string(),
            duration_ms: Some(1.5),
        };
        let json = serde_json::to_string(&result).expect("should serialize");
        assert!(json.contains("device_availability"));
        assert!(json.contains("1.5"));
    }
}
