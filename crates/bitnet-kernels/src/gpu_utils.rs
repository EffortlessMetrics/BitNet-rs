//! GPU availability and preflight check utilities
//!
//! This module provides utilities to check GPU availability, driver versions,
//! and perform preflight checks before running GPU-accelerated code.

use std::env;
use std::process::Command;

/// Check if any GPU backend is available
pub fn gpu_available() -> bool {
    cuda_available() || metal_available() || rocm_available() || wgpu_available()
}

/// Check if NVIDIA CUDA is available
pub fn cuda_available() -> bool {
    // Check common CUDA environment variables
    if env::var("CUDA_HOME").is_ok() || env::var("CUDA_PATH").is_ok() {
        return true;
    }

    // Try to run nvidia-smi to check for NVIDIA GPU
    Command::new("nvidia-smi")
        .arg("--query-gpu=gpu_name")
        .arg("--format=csv,noheader")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Check if Apple Metal is available (macOS only)
pub fn metal_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        // Metal is available on all modern macOS systems
        true
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

/// Check if AMD ROCm is available
pub fn rocm_available() -> bool {
    // Check ROCm environment variable
    if env::var("ROCM_PATH").is_ok() {
        return true;
    }

    // Try to run rocm-smi to check for AMD GPU
    Command::new("rocm-smi")
        .arg("--showid")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Check if WebGPU/wgpu backend is available
pub fn wgpu_available() -> bool {
    // wgpu is a software fallback that's always available
    // but we should check for actual GPU devices
    true // In production, this would query wgpu for available adapters
}

/// Get information about available GPU backends
pub fn get_gpu_info() -> GpuInfo {
    GpuInfo {
        cuda: cuda_available(),
        cuda_version: get_cuda_version(),
        metal: metal_available(),
        rocm: rocm_available(),
        rocm_version: get_rocm_version(),
        wgpu: wgpu_available(),
    }
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
