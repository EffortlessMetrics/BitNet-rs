//! Device feature detection and capability queries for Issue #439
//!
//! This module provides unified device capability checks for GPU/CPU selection
//! across the BitNet-rs workspace. It consolidates compile-time feature gates
//! with runtime hardware detection.
//!
//! # Architecture Decision
//!
//! This module lives in `bitnet-kernels` rather than `bitnet-common` to avoid
//! circular dependencies, since `bitnet-common` depends on `bitnet-kernels`
//! for GPU availability checks.
//!
//! # Specification
//!
//! Tests specification: docs/explanation/issue-439-spec.md#device-feature-detection-api

/// Check if GPU support was compiled into this binary
///
/// Returns `true` if either `feature="gpu"` or `feature="cuda"` was enabled
/// at compile time. This does NOT check runtime GPU availability.
///
/// # Example
///
/// ```
/// use bitnet_kernels::device_features::gpu_compiled;
///
/// if gpu_compiled() {
///     println!("GPU support compiled into binary");
///     // Can attempt GPU operations (may still fail at runtime)
/// } else {
///     println!("GPU support NOT compiled - CPU only");
///     // Must use CPU-only operations
/// }
/// ```
///
/// # Specification
///
/// Tests specification: docs/explanation/issue-439-spec.md#ac3-shared-helpers
#[inline]
pub fn gpu_compiled() -> bool {
    cfg!(any(feature = "gpu", feature = "cuda", feature = "hip"))
}

/// Check if HIP/ROCm support was compiled into this binary.
#[inline]
pub fn hip_compiled() -> bool {
    cfg!(feature = "hip")
}

/// Check if Intel oneAPI/`OpenCL` support was compiled into this binary.
#[inline]
pub fn oneapi_compiled() -> bool {
    cfg!(feature = "oneapi")
}

/// Check if GPU is available at runtime
///
/// Returns `false` if:
/// - GPU not compiled (`gpu_compiled() == false`)
/// - CUDA runtime unavailable (`nvidia-smi` fails)
/// - `BITNET_GPU_FAKE=none` environment variable set
///
/// Returns `true` if:
/// - GPU compiled AND CUDA runtime detected
/// - `BITNET_GPU_FAKE=cuda` environment variable set (overrides real detection)
///   (unless `BITNET_STRICT_MODE=1` is set, which forces real detection)
///
/// # Strict Mode
///
/// When `BITNET_STRICT_MODE=1` is set, `BITNET_GPU_FAKE` is ignored and only
/// real GPU detection is used. This prevents fake GPU simulation in strict mode.
///
/// # Example
///
/// ```
/// use bitnet_kernels::device_features::{gpu_compiled, gpu_available_runtime};
///
/// // Check both compile-time and runtime GPU availability
/// if gpu_compiled() && gpu_available_runtime() {
///     println!("GPU available: use CUDA acceleration");
///     // Safe to use GPU operations
/// } else if gpu_compiled() {
///     println!("GPU compiled but not available at runtime - fallback to CPU");
/// } else {
///     println!("GPU not compiled - CPU only");
/// }
/// ```
///
/// # Specification
///
/// Tests specification: docs/explanation/issue-439-spec.md#ac3-shared-helpers
#[cfg(any(feature = "gpu", feature = "cuda", feature = "hip"))]
#[inline]
pub fn gpu_available_runtime() -> bool {
    use std::env;

    // In strict mode, refuse BITNET_GPU_FAKE and always use real detection
    let strict_mode = env::var("BITNET_STRICT_MODE")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);

    if strict_mode {
        // Strict mode: only real GPU detection
        return crate::gpu_utils::get_gpu_info().cuda
            || bitnet_device_probe::gpu_available_runtime();
    }

    // Check BITNET_GPU_FAKE first (deterministic testing)
    if let Ok(fake) = env::var("BITNET_GPU_FAKE") {
        return fake.eq_ignore_ascii_case("cuda")
            || fake.eq_ignore_ascii_case("rocm")
            || fake.eq_ignore_ascii_case("hip")
            || fake.eq_ignore_ascii_case("gpu");
    }

    // Fall back to real GPU detection
    crate::gpu_utils::get_gpu_info().cuda || bitnet_device_probe::gpu_available_runtime()
}

/// Check if Intel oneAPI GPU runtime is available.
///
/// Detection is best-effort via `sycl-ls`. Tests can force deterministic
/// outcomes with `BITNET_GPU_FAKE=oneapi` / `BITNET_GPU_FAKE=none` unless
/// strict mode is enabled.
#[cfg(feature = "oneapi")]
#[inline]
pub fn oneapi_available_runtime() -> bool {
    use std::env;
    use std::process::{Command, Stdio};

    let strict_mode = env::var("BITNET_STRICT_MODE")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);

    if !strict_mode {
        if let Ok(fake) = env::var("BITNET_GPU_FAKE") {
            if fake.eq_ignore_ascii_case("oneapi") {
                return true;
            }
            if fake.eq_ignore_ascii_case("none") {
                return false;
            }
        }
    }

    Command::new("sycl-ls")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(not(feature = "oneapi"))]
#[inline]
pub fn oneapi_available_runtime() -> bool {
    false
}

/// Stub implementation when GPU not compiled
#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "hip")))]
#[inline]
pub fn gpu_available_runtime() -> bool {
    // Correct implementation: GPU never available if not compiled
    false
}

/// Get device capability summary for diagnostics
///
/// Returns a human-readable summary of compile-time and runtime capabilities.
///
/// # Example
///
/// ```
/// use bitnet_kernels::device_features::device_capability_summary;
///
/// // Print diagnostic information
/// println!("{}", device_capability_summary());
///
/// // Example output when GPU compiled and available:
/// // Device Capabilities:
/// //   Compiled: GPU ✓, CPU ✓
/// //   Runtime: CUDA 12.1 ✓, CPU ✓
/// ```
///
/// # Specification
///
/// Tests specification: docs/explanation/issue-439-spec.md#device-feature-detection-api
pub fn device_capability_summary() -> String {
    let mut summary = String::from("Device Capabilities:\n");

    // Compile-time capabilities
    summary.push_str("  Compiled: ");
    if gpu_compiled() {
        summary.push_str("GPU ✓, ");
    }
    summary.push_str("CPU ✓\n");

    // Runtime capabilities
    summary.push_str("  Runtime: ");

    #[cfg(any(feature = "gpu", feature = "cuda", feature = "hip"))]
    {
        if gpu_available_runtime() {
            let info = crate::gpu_utils::get_gpu_info();
            if info.cuda {
                if let Some(version) = &info.cuda_version {
                    summary.push_str(&format!("CUDA {} ✓", version));
                } else {
                    summary.push_str("CUDA ✓");
                }
            } else {
                summary.push_str("HIP/ROCm ✓");
            }
        } else {
            summary.push_str("CUDA ✗");
        }
        summary.push_str(", ");
    }

    summary.push_str("CPU ✓");

    summary
}

// ---------------------------------------------------------------------------
// Intel GPU detection
// ---------------------------------------------------------------------------

/// Intel GPU device information detected at runtime.
#[derive(Debug, Clone, Default)]
pub struct IntelGpuInfo {
    /// Whether any Intel GPU was detected
    pub detected: bool,
    /// Device name (e.g., "Intel Arc A770")
    pub device_name: String,
    /// Driver version
    pub driver_version: String,
    /// OpenCL version supported
    pub opencl_version: String,
    /// Device memory in bytes
    pub memory_bytes: u64,
    /// Number of compute units (Xe-cores for Arc)
    pub compute_units: u32,
    /// Maximum work group size
    pub max_work_group_size: usize,
    /// Whether Level Zero is available
    pub level_zero_available: bool,
}

/// Probe for Intel GPU devices.
///
/// Checks for Intel Arc/Xe GPUs via:
/// 1. `BITNET_GPU_FAKE=opencl` or `BITNET_GPU_FAKE=intel` environment variable
///    (for testing)
/// 2. `sycl-ls` command (if oneAPI installed)
/// 3. `clinfo` command (if OpenCL runtime installed)
pub fn probe_intel_gpu() -> IntelGpuInfo {
    // Check fake GPU environment first (unless strict mode)
    if std::env::var("BITNET_STRICT_MODE").unwrap_or_default() != "1" {
        if let Ok(val) = std::env::var("BITNET_GPU_FAKE") {
            if val.contains("opencl") || val.contains("intel") {
                return IntelGpuInfo {
                    detected: true,
                    device_name: "Intel Arc A770 (simulated)".to_string(),
                    driver_version: "simulated".to_string(),
                    opencl_version: "OpenCL 3.0".to_string(),
                    memory_bytes: 16 * 1024 * 1024 * 1024, // 16 GB
                    compute_units: 32,
                    max_work_group_size: 1024,
                    level_zero_available: true,
                };
            }
        }
    }

    probe_intel_gpu_real()
}

fn probe_intel_gpu_real() -> IntelGpuInfo {
    let mut info = IntelGpuInfo::default();

    // Try sycl-ls first (Intel oneAPI)
    if let Ok(output) = std::process::Command::new("sycl-ls").output() {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if stdout.contains("Intel")
                && (stdout.contains("Arc") || stdout.contains("gpu"))
            {
                info.detected = true;
                info.device_name = "Intel GPU (via sycl-ls)".to_string();
            }
        }
    }

    // Fallback to clinfo
    if !info.detected {
        if let Ok(output) =
            std::process::Command::new("clinfo").arg("--list").output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if stdout.contains("Intel") {
                    info.detected = true;
                    for line in stdout.lines() {
                        if line.contains("Arc")
                            || (line.contains("Intel") && line.contains("Graphics"))
                        {
                            info.device_name = line.trim().to_string();
                            break;
                        }
                    }
                    if info.device_name.is_empty() {
                        info.device_name = "Intel GPU (via clinfo)".to_string();
                    }
                }
            }
        }
    }

    info
}

/// Returns true if an Intel GPU is available for OpenCL compute.
///
/// Respects `BITNET_GPU_FAKE` for testing and `BITNET_STRICT_MODE` for
/// production.
pub fn intel_gpu_available() -> bool {
    probe_intel_gpu().detected
}

/// Returns a human-readable summary of Intel GPU status.
pub fn intel_gpu_status_string() -> String {
    let info = probe_intel_gpu();
    if info.detected {
        format!(
            "Intel GPU: {} ({}MB, {} CUs, {})",
            info.device_name,
            info.memory_bytes / (1024 * 1024),
            info.compute_units,
            info.opencl_version,
        )
    } else {
        "Intel GPU: not detected".to_string()
    }
}

// Re-export the dedicated crate so callers can use either path.
pub use bitnet_device_probe::DeviceCapabilities;

/// Detect the best SIMD level available at runtime.
///
/// Delegates to [`bitnet_device_probe::detect_simd_level`].
#[inline]
pub fn detect_simd_level() -> bitnet_common::kernel_registry::SimdLevel {
    bitnet_device_probe::detect_simd_level()
}

/// Build a fully-populated [`KernelCapabilities`] by combining compile-time
/// feature flags with a live runtime GPU probe.
///
/// This is the canonical way for binaries (CLI, server) to determine what
/// backend to use at startup. Combine with [`bitnet_common::select_backend`]
/// to get the startup log line:
/// `requested=auto detected=[cpu-rust] selected=cpu-rust`.
///
/// Note: uses the `bitnet-kernels` crate's own feature flags (`cpu`, `cuda`)
/// which are the authoritative source for kernel availability.
pub fn current_kernel_capabilities() -> bitnet_common::kernel_registry::KernelCapabilities {
    bitnet_common::kernel_registry::KernelCapabilities {
        cpu_rust: cfg!(feature = "cpu"),
        cuda_compiled: cfg!(any(feature = "gpu", feature = "cuda")),
        cuda_runtime: crate::gpu_utils::get_gpu_info().cuda,
        hip_compiled: cfg!(feature = "hip"),
        hip_runtime: bitnet_device_probe::probe_gpu().rocm_available,
        oneapi_compiled: cfg!(feature = "oneapi"),
        oneapi_runtime: oneapi_available_runtime(),
        cpp_ffi: false,
        simd_level: detect_simd_level(),
    }
}

#[cfg(test)]
mod intel_tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn intel_gpu_info_default() {
        let info = IntelGpuInfo::default();
        assert!(!info.detected);
        assert!(info.device_name.is_empty());
    }

    #[test]
    #[serial(bitnet_env)]
    fn intel_gpu_fake_detection() {
        temp_env::with_var("BITNET_GPU_FAKE", Some("opencl"), || {
            let info = probe_intel_gpu();
            assert!(info.detected);
            assert!(info.device_name.contains("simulated"));
            assert_eq!(info.compute_units, 32);
            assert!(intel_gpu_available());
        });
    }

    #[test]
    #[serial(bitnet_env)]
    fn intel_gpu_fake_intel_variant() {
        temp_env::with_var("BITNET_GPU_FAKE", Some("intel"), || {
            assert!(intel_gpu_available());
        });
    }

    #[test]
    #[serial(bitnet_env)]
    fn intel_gpu_no_fake() {
        temp_env::with_var("BITNET_GPU_FAKE", Some("none"), || {
            // Without real hardware, should not detect
            // (may or may not detect depending on system)
            let _ = probe_intel_gpu();
        });
    }

    #[test]
    fn intel_gpu_status_string_format() {
        let status = intel_gpu_status_string();
        assert!(status.contains("Intel GPU"));
    }

    #[test]
    fn intel_gpu_info_clone() {
        let info = IntelGpuInfo {
            detected: true,
            device_name: "Arc A770".to_string(),
            ..Default::default()
        };
        let clone = info.clone();
        assert_eq!(clone.device_name, "Arc A770");
    }

    #[test]
    fn intel_gpu_info_debug() {
        let info = IntelGpuInfo::default();
        let debug = format!("{info:?}");
        assert!(debug.contains("IntelGpuInfo"));
    }
}
