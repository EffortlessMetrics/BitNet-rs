//! GPU device information and discovery command.

use anyhow::Result;
use clap::Args;
use console::style;
use serde::Serialize;

use bitnet_device_probe::{detect_simd_level, probe_cpu, probe_gpu};
use bitnet_kernels::device_features::{
    current_kernel_capabilities, oneapi_available_runtime, oneapi_compiled,
};
use bitnet_kernels::gpu_utils::get_gpu_info;

/// List available GPU devices and capabilities.
#[derive(Args, Debug)]
pub struct GpuInfoCommand {
    /// Output as JSON instead of table
    #[arg(long)]
    pub json: bool,
}

/// Serialisable representation of a detected GPU device.
#[derive(Debug, Serialize)]
pub struct DeviceEntry {
    pub index: usize,
    pub name: String,
    pub backend: String,
    pub driver_version: Option<String>,
    pub memory_mb: Option<u64>,
    pub compute_units: Option<u32>,
}

/// Full report produced by gpu-info.
#[derive(Debug, Serialize)]
pub struct GpuInfoReport {
    pub devices: Vec<DeviceEntry>,
    pub compiled_backends: Vec<String>,
    pub runtime_backends: Vec<String>,
    pub recommended: Option<String>,
    pub simd_level: String,
    pub cpu_cores: usize,
}

impl GpuInfoCommand {
    pub async fn execute(&self) -> Result<()> {
        let report = build_report();

        if self.json {
            println!("{}", serde_json::to_string_pretty(&report)?);
        } else {
            print_table(&report);
        }

        Ok(())
    }
}

/// Collect device information from all available probes.
pub fn build_report() -> GpuInfoReport {
    let gpu_info = get_gpu_info();
    let cpu_caps = probe_cpu();
    let gpu_caps = probe_gpu();
    let simd = detect_simd_level();
    let caps = current_kernel_capabilities();

    let mut devices = Vec::new();
    let mut idx = 0usize;

    if gpu_caps.cuda_available {
        devices.push(DeviceEntry {
            index: idx,
            name: "NVIDIA GPU".to_string(),
            backend: "cuda".to_string(),
            driver_version: gpu_info.cuda_version.clone(),
            memory_mb: None,
            compute_units: None,
        });
        idx += 1;
    }

    if gpu_caps.rocm_available {
        devices.push(DeviceEntry {
            index: idx,
            name: "AMD GPU".to_string(),
            backend: "rocm".to_string(),
            driver_version: gpu_info.rocm_version.clone(),
            memory_mb: None,
            compute_units: None,
        });
        idx += 1;
    }

    if oneapi_available_runtime() {
        devices.push(DeviceEntry {
            index: idx,
            name: "Intel GPU".to_string(),
            backend: "oneapi".to_string(),
            driver_version: None,
            memory_mb: None,
            compute_units: None,
        });
        idx += 1;
    }

    if gpu_info.metal {
        devices.push(DeviceEntry {
            index: idx,
            name: "Apple GPU".to_string(),
            backend: "metal".to_string(),
            driver_version: None,
            memory_mb: None,
            compute_units: None,
        });
        let _ = idx;
    }

    let mut compiled = vec!["cpu".to_string()];
    if caps.cuda_compiled {
        compiled.push("cuda".to_string());
    }
    if oneapi_compiled() {
        compiled.push("oneapi".to_string());
    }
    if cfg!(feature = "metal") {
        compiled.push("metal".to_string());
    }

    let mut runtime = vec!["cpu".to_string()];
    if caps.cuda_runtime {
        runtime.push("cuda".to_string());
    }
    if oneapi_available_runtime() {
        runtime.push("oneapi".to_string());
    }
    if gpu_info.metal {
        runtime.push("metal".to_string());
    }

    let recommended = recommend_device(&devices, &runtime);

    GpuInfoReport {
        devices,
        compiled_backends: compiled,
        runtime_backends: runtime,
        recommended,
        simd_level: format!("{:?}", simd),
        cpu_cores: cpu_caps.core_count,
    }
}

/// Pick the best device for inference (GPU backends take priority over CPU).
fn recommend_device(
    devices: &[DeviceEntry],
    runtime: &[String],
) -> Option<String> {
    let priority = ["cuda", "oneapi", "rocm", "metal"];
    for backend in &priority {
        if runtime.contains(&(*backend).to_string()) {
            if let Some(dev) = devices.iter().find(|d| d.backend == *backend) {
                return Some(format!("{} ({})", backend, dev.name));
            }
            return Some(backend.to_string());
        }
    }
    Some("cpu".to_string())
}

/// Pretty-print the report as a human-readable table.
fn print_table(report: &GpuInfoReport) {
    println!("{}", style("GPU Devices Detected:").bold().cyan());

    if report.devices.is_empty() {
        println!("  (none)");
    } else {
        for dev in &report.devices {
            println!(
                "  #{} {} ({})",
                dev.index,
                style(&dev.name).bold(),
                dev.backend
            );
            if let Some(ref drv) = dev.driver_version {
                println!("     Driver: {}", drv);
            }
            if let Some(mem) = dev.memory_mb {
                println!("     Memory: {} MB", mem);
            }
            if let Some(cu) = dev.compute_units {
                println!("     Compute: {} units", cu);
            }
        }
    }

    println!();
    println!(
        "{} {}",
        style("Compiled Backends:").bold(),
        report.compiled_backends.join(", ")
    );
    println!(
        "{} {}",
        style("Runtime Available:").bold(),
        report.runtime_backends.join(", ")
    );

    if let Some(ref rec) = report.recommended {
        println!("{} {}", style("Recommended:").bold(), rec);
    }

    println!();
    println!("{} {}", style("SIMD Level:").bold(), report.simd_level);
    println!("{} {}", style("CPU Cores:").bold(), report.cpu_cores);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_report_returns_cpu_backend() {
        let report = build_report();
        assert!(
            report.compiled_backends.contains(&"cpu".to_string()),
            "cpu must always be in compiled backends"
        );
        assert!(
            report.runtime_backends.contains(&"cpu".to_string()),
            "cpu must always be in runtime backends"
        );
    }

    #[test]
    fn test_recommend_device_cpu_fallback() {
        let devices = vec![];
        let runtime = vec!["cpu".to_string()];
        let rec = recommend_device(&devices, &runtime);
        assert_eq!(rec, Some("cpu".to_string()));
    }

    #[test]
    fn test_recommend_device_prefers_cuda() {
        let devices = vec![
            DeviceEntry {
                index: 0,
                name: "NVIDIA GPU".to_string(),
                backend: "cuda".to_string(),
                driver_version: None,
                memory_mb: None,
                compute_units: None,
            },
            DeviceEntry {
                index: 1,
                name: "Intel GPU".to_string(),
                backend: "oneapi".to_string(),
                driver_version: None,
                memory_mb: None,
                compute_units: None,
            },
        ];
        let runtime = vec![
            "cpu".to_string(),
            "cuda".to_string(),
            "oneapi".to_string(),
        ];
        let rec = recommend_device(&devices, &runtime);
        assert_eq!(rec, Some("cuda (NVIDIA GPU)".to_string()));
    }

    #[test]
    fn test_recommend_device_prefers_oneapi_over_rocm() {
        let devices = vec![
            DeviceEntry {
                index: 0,
                name: "AMD GPU".to_string(),
                backend: "rocm".to_string(),
                driver_version: None,
                memory_mb: None,
                compute_units: None,
            },
            DeviceEntry {
                index: 1,
                name: "Intel GPU".to_string(),
                backend: "oneapi".to_string(),
                driver_version: None,
                memory_mb: None,
                compute_units: None,
            },
        ];
        let runtime = vec![
            "cpu".to_string(),
            "rocm".to_string(),
            "oneapi".to_string(),
        ];
        let rec = recommend_device(&devices, &runtime);
        assert_eq!(rec, Some("oneapi (Intel GPU)".to_string()));
    }

    #[test]
    fn test_report_json_serialization() {
        let report = build_report();
        let json = serde_json::to_string(&report);
        assert!(json.is_ok(), "report must be JSON-serialisable");
        let parsed: serde_json::Value =
            serde_json::from_str(&json.unwrap()).unwrap();
        assert!(parsed["compiled_backends"].is_array());
        assert!(parsed["cpu_cores"].is_number());
    }

    #[test]
    fn test_print_table_no_panic() {
        let report = build_report();
        print_table(&report);
    }

    #[test]
    fn test_empty_devices_table() {
        let report = GpuInfoReport {
            devices: vec![],
            compiled_backends: vec!["cpu".to_string()],
            runtime_backends: vec!["cpu".to_string()],
            recommended: Some("cpu".to_string()),
            simd_level: "Avx2".to_string(),
            cpu_cores: 8,
        };
        print_table(&report);
    }
}
