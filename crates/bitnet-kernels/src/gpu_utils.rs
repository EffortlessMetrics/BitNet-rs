//! GPU availability and preflight check utilities
//!
//! This module provides utilities to check GPU availability, driver versions,
//! and perform preflight checks before running GPU-accelerated code.

use std::env;
use std::process::Command;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

const PROBE_TIMEOUT: Duration = Duration::from_secs(5);
const PROBE_POLL_INTERVAL: Duration = Duration::from_millis(50);
static REAL_GPU_INFO_CACHE: OnceLock<GpuInfo> = OnceLock::new();

/// Run a shell command with a hard-kill timeout. Returns `false` on timeout or failure.
///
/// Unlike the previous `recv_timeout` approach, this actually kills the subprocess
/// when the deadline is exceeded so no zombie processes or stuck threads are left behind.
fn probe_command(cmd: &str, args: &[&str]) -> bool {
    let mut child = match Command::new(cmd)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    let deadline = Instant::now() + PROBE_TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return status.success(),
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait(); // reap to avoid zombie
                    return false;
                }
                std::thread::sleep(PROBE_POLL_INTERVAL);
            }
            Err(_) => return false,
        }
    }
}

/// Run a shell command with a hard-kill timeout and capture its stdout.
/// Returns `None` on timeout, failure, or non-UTF8 output.
fn probe_command_output(cmd: &str, args: &[&str]) -> Option<String> {
    use std::io::Read;
    let mut child = match Command::new(cmd)
        .args(args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => return None,
    };
    let deadline = Instant::now() + PROBE_TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait(); // reap to avoid zombie
                    return None;
                }
                std::thread::sleep(PROBE_POLL_INTERVAL);
            }
            Err(_) => return None,
        }
    };
    if !status.success() {
        return None;
    }
    // Process already exited; drain its piped stdout (at EOF, so this is fast).
    let mut stdout = child.stdout.take()?;
    let mut output = String::new();
    stdout.read_to_string(&mut output).ok()?;
    Some(output)
}

/// Check if any GPU backend is available
pub fn gpu_available() -> bool {
    get_gpu_info().any_available()
}

/// Get information about available GPU backends
pub fn get_gpu_info() -> GpuInfo {
    if let Ok(fake) = env::var("BITNET_GPU_FAKE") {
        if env::var("BITNET_STRICT_NO_FAKE_GPU").as_deref() == Ok("1") {
            panic!(
                "BITNET_GPU_FAKE is set but strict mode forbids fake GPU (BITNET_STRICT_NO_FAKE_GPU=1)"
            );
        }
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

    // Fake GPU selection is intentionally not cached, so tests and tooling can
    // change BITNET_GPU_FAKE across calls and get deterministic responses.
    if env::var("BITNET_GPU_CACHE").as_deref() == Ok("0") {
        return detect_real_gpu_info();
    }

    REAL_GPU_INFO_CACHE.get_or_init(detect_real_gpu_info).clone()
}

fn detect_real_gpu_info() -> GpuInfo {
    let metal = cfg!(target_os = "macos");

    let cuda = probe_command("nvidia-smi", &["--query-gpu=gpu_name", "--format=csv,noheader"]);

    let cuda_version = if cuda { get_cuda_version() } else { None };

    let rocm = probe_command("rocm-smi", &["--showid"]);

    let rocm_version = if rocm { get_rocm_version() } else { None };

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
    probe_command_output("nvcc", &["--version"]).and_then(|output| {
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
    probe_command_output("rocm-smi", &["--version"]).and_then(|output| {
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
