//! GPU availability and preflight check utilities
//!
//! This module provides utilities to check GPU availability, driver versions,
//! and perform preflight checks before running GPU-accelerated code.

use std::env;
use std::process::Command;
use sysinfo::System;

/// Check if any GPU backend is available
pub fn gpu_available() -> bool {
    get_gpu_info().any_available()
}

/// Get information about available GPU backends
pub fn get_gpu_info() -> GpuInfo {
    if let Ok(fake) = env::var("BITNET_GPU_FAKE") {
        let lower = fake.to_lowercase();
        return GpuInfo {
            cuda: lower.contains("cuda"),
            cuda_version: None,
            metal: lower.contains("metal"),
            rocm: lower.contains("rocm"),
            rocm_version: None,
            wgpu: lower.contains("wgpu")
                || lower.contains("cuda")
                || lower.contains("rocm")
                || lower.contains("metal"),
        };
    }

    let _sys = System::new_all();

    let mut metal = System::name().unwrap_or_default().to_lowercase().contains("mac");

    let cuda = Command::new("nvidia-smi")
        .arg("--query-gpu=gpu_name")
        .arg("--format=csv,noheader")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);

    let cuda_version = if cuda { get_cuda_version() } else { None };

    let rocm = Command::new("rocm-smi")
        .arg("--showid")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);

    let rocm_version = if rocm { get_rocm_version() } else { None };

    if cfg!(target_os = "macos") {
        metal = true;
    }

    let wgpu = cuda || rocm || metal;

    GpuInfo { cuda, cuda_version, metal, rocm, rocm_version, wgpu }
}

/// Information about available GPU backends
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub cuda: bool,
    pub cuda_version: Option<String>,
    pub metal: bool,
    pub rocm: bool,
    pub rocm_version: Option<String>,
    pub wgpu: bool,
}

impl GpuInfo {
    /// Check if any GPU backend is available
    pub fn any_available(&self) -> bool {
        self.cuda || self.metal || self.rocm || self.wgpu
    }

    /// Get a human-readable summary of available backends
    pub fn summary(&self) -> String {
        let mut backends = Vec::new();

        if self.cuda {
            if let Some(version) = &self.cuda_version {
                backends.push(format!("CUDA {}", version));
            } else {
                backends.push("CUDA".to_string());
            }
        }

        if self.metal {
            backends.push("Metal".to_string());
        }

        if self.rocm {
            if let Some(version) = &self.rocm_version {
                backends.push(format!("ROCm {}", version));
            } else {
                backends.push("ROCm".to_string());
            }
        }

        if self.wgpu {
            backends.push("WebGPU".to_string());
        }

        if backends.is_empty() {
            "No GPU backends available".to_string()
        } else {
            format!("Available GPU backends: {}", backends.join(", "))
        }
    }
}

/// Get CUDA version if available
fn get_cuda_version() -> Option<String> {
    Command::new("nvcc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() { String::from_utf8(output.stdout).ok() } else { None }
        })
        .and_then(|output| {
            // Parse version from nvcc output
            output.lines().find(|line| line.contains("release")).and_then(|line| {
                line.split("release")
                    .nth(1)
                    .and_then(|s| s.split(',').next())
                    .map(|s| s.trim().to_string())
            })
        })
}

/// Get ROCm version if available
fn get_rocm_version() -> Option<String> {
    Command::new("rocm-smi")
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() { String::from_utf8(output.stdout).ok() } else { None }
        })
        .and_then(|output| {
            // Parse version from rocm-smi output
            output
                .lines()
                .find(|line| line.contains("Version"))
                .and_then(|line| line.split(':').nth(1).map(|s| s.trim().to_string()))
        })
}

/// Perform a preflight check for GPU operations
/// Returns Ok(()) if GPU is available, Err with helpful message otherwise
pub fn preflight_check() -> Result<(), String> {
    let info = get_gpu_info();

    if !info.any_available() {
        return Err(format!("No GPU backend detected. {}", preflight_help_message()));
    }

    eprintln!("{}", info.summary());
    Ok(())
}

/// Get a helpful message for GPU setup
fn preflight_help_message() -> &'static str {
    "To enable GPU acceleration:
    - NVIDIA GPUs: Install CUDA toolkit from https://developer.nvidia.com/cuda-downloads
    - AMD GPUs: Install ROCm from https://rocm.docs.amd.com
    - Apple Silicon: Metal support is built-in on macOS
    - Other GPUs: WebGPU backend provides compatibility
    
    Set CUDA_HOME or ROCM_PATH environment variables after installation."
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_info_summary() {
        let info = GpuInfo {
            cuda: true,
            cuda_version: Some("12.0".to_string()),
            metal: false,
            rocm: false,
            rocm_version: None,
            wgpu: true,
        };

        assert!(info.any_available());
        assert!(info.summary().contains("CUDA 12.0"));
        assert!(info.summary().contains("WebGPU"));
    }

    #[test]
    fn test_no_gpu_info() {
        let info = GpuInfo {
            cuda: false,
            cuda_version: None,
            metal: false,
            rocm: false,
            rocm_version: None,
            wgpu: false,
        };

        assert!(!info.any_available());
        assert_eq!(info.summary(), "No GPU backends available");
    }
}
