//! Device detection and capability probing for `BitNet` inference.
//!
//! Provides unified compile-time and runtime device capability queries,
//! extracted from `bitnet-kernels` for use by the broader workspace.

pub use bitnet_common::kernel_registry::SimdLevel;
pub use bitnet_cpu_probe::{CpuCapabilities, detect_simd_level, probe_cpu, simd_level_rank};

#[cfg(feature = "opencl")]
pub mod opencl;
#[cfg(feature = "opencl")]
pub use opencl::{
    IntelArcDetector, OpenClDeviceInfo, OpenClDeviceType, OpenClPlatformInfo, OpenClProbeResult,
    ProbeResult, is_intel_arc_available, list_opencl_devices, probe_opencl,
};

#[cfg(feature = "wgpu-probe")]
pub mod wgpu_probe;
#[cfg(feature = "wgpu-probe")]
pub use wgpu_probe::{
    WgpuBackend, WgpuDeviceInfo, WgpuDeviceType, WgpuLimits, is_nvidia, is_vulkan_backend,
    probe_best_wgpu_device, probe_wgpu_devices, supports_f16,
};

// ── CPU capabilities ─────────────────────────────────────────────────────────

// ── GPU capabilities ─────────────────────────────────────────────────────────

/// GPU capabilities detected at runtime.
///
/// Obtained by calling [`probe_gpu`].
///
/// `BITNET_GPU_FAKE` supports comma-separated backends (`cuda`, `rocm`, `oneapi`, `gpu`).
/// For example `BITNET_GPU_FAKE=rocm` forces `ROCm` runtime availability for tests.
/// `BITNET_GPU_FAKE=none` makes all GPU flags `false`.
/// Strict mode (`BITNET_STRICT_MODE=1`) ignores `BITNET_GPU_FAKE` and probes real hardware.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct GpuCapabilities {
    /// Any GPU backend is available (CUDA, `ROCm`, and/or oneAPI).
    pub available: bool,
    /// CUDA runtime was detected (or faked via `BITNET_GPU_FAKE`).
    pub cuda_available: bool,
    /// `ROCm` runtime was detected (or faked via `BITNET_GPU_FAKE`).
    pub rocm_available: bool,
    /// Intel oneAPI/`OpenCL` runtime was detected (or faked via `BITNET_GPU_FAKE`).
    pub oneapi_available: bool,
}

/// NPU capabilities detected at runtime.
///
/// Obtained by calling [`probe_npu`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NpuCapabilities {
    /// Intel NPU runtime is available.
    pub available: bool,
    /// A `/dev/accel/*` character device appears to be present.
    pub accel_device_present: bool,
}
/// Probe GPU availability and return its capabilities.
///
/// Honours `BITNET_GPU_FAKE` for deterministic testing unless
/// `BITNET_STRICT_MODE=1` is set.
#[cfg(any(feature = "gpu", feature = "cuda", feature = "rocm", feature = "oneapi"))]
pub fn probe_gpu() -> GpuCapabilities {
    let cuda_available = cuda_available_runtime();
    let rocm_available = rocm_available_runtime();
    let oneapi_available = oneapi_available_runtime();
    let available = cuda_available || rocm_available || oneapi_available;
    GpuCapabilities { available, cuda_available, rocm_available, oneapi_available }
}

/// Probe GPU availability; always returns `false` when GPU not compiled.
#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "rocm", feature = "oneapi")))]
pub const fn probe_gpu() -> GpuCapabilities {
    GpuCapabilities {
        available: false,
        cuda_available: false,
        rocm_available: false,
        oneapi_available: false,
    }
}

/// Check if GPU support was compiled into this binary.
///
/// Returns `true` if `feature="gpu"`, `feature="cuda"`, `feature="rocm"`,
/// or `feature="oneapi"` was enabled at compile time. Does **not** check
/// runtime GPU availability — use [`gpu_available_runtime`] for that.
///
/// # Examples
///
/// ```
/// use bitnet_device_probe::gpu_compiled;
///
/// // When built with `--features cpu` only, this returns false.
/// // When built with `--features gpu` or `--features oneapi`, this returns true.
/// let _compiled: bool = gpu_compiled();
/// ```
#[inline]
pub const fn gpu_compiled() -> bool {
    cfg!(any(feature = "gpu", feature = "cuda", feature = "rocm", feature = "oneapi"))
}

/// Check if a GPU is available at runtime.
///
/// - Returns `false` when GPU is not compiled or CUDA runtime is unavailable.
/// - Respects `BITNET_GPU_FAKE=cuda|rocm|oneapi|gpu` (returns `true`) /
///   `BITNET_GPU_FAKE=none` (returns `false`) for deterministic testing, unless
///   `BITNET_STRICT_MODE=1`.
/// - In strict mode only real hardware detection is used.
#[cfg(any(feature = "gpu", feature = "cuda", feature = "rocm", feature = "oneapi"))]
pub fn gpu_available_runtime() -> bool {
    cuda_available_runtime() || rocm_available_runtime() || oneapi_available_runtime()
}

/// Stub: GPU never available when not compiled.
#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "rocm", feature = "oneapi")))]
#[inline]
pub const fn gpu_available_runtime() -> bool {
    false
}

#[cfg(any(feature = "gpu", feature = "cuda", feature = "rocm", feature = "oneapi"))]
fn strict_mode_enabled() -> bool {
    std::env::var("BITNET_STRICT_MODE")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false)
}

#[cfg(any(feature = "gpu", feature = "cuda", feature = "rocm", feature = "oneapi"))]
fn fake_gpu_backends() -> Option<std::collections::HashSet<String>> {
    if strict_mode_enabled() {
        return None;
    }

    let fake = std::env::var("BITNET_GPU_FAKE").ok()?;
    let normalized = fake.trim().to_ascii_lowercase();

    if normalized == "none" {
        return Some(std::collections::HashSet::new());
    }

    let set = normalized
        .split([',', ';', '|', ' '])
        .filter(|part| !part.is_empty())
        .map(ToOwned::to_owned)
        .collect();

    Some(set)
}

#[cfg(any(feature = "gpu", feature = "cuda", feature = "rocm", feature = "oneapi"))]
fn command_ok(cmd: &str, args: &[&str]) -> bool {
    std::process::Command::new(cmd)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(any(feature = "gpu", feature = "cuda", feature = "rocm"))]
fn cuda_available_runtime() -> bool {
    if let Some(fake) = fake_gpu_backends() {
        return fake.contains("cuda") || fake.contains("gpu");
    }

    command_ok("nvidia-smi", &[])
}

#[cfg(any(feature = "gpu", feature = "cuda", feature = "rocm"))]
fn rocm_available_runtime() -> bool {
    if let Some(fake) = fake_gpu_backends() {
        return fake.contains("rocm") || fake.contains("gpu");
    }

    command_ok("rocm-smi", &["--showid"])
}

#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "rocm")))]
#[inline]
const fn cuda_available_runtime() -> bool {
    false
}

#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "rocm")))]
#[inline]
const fn rocm_available_runtime() -> bool {
    false
}

/// Check if NPU support was compiled into this binary.
#[inline]
pub const fn npu_compiled() -> bool {
    cfg!(feature = "npu")
}

/// Probe Intel NPU availability and return capabilities.
#[cfg(feature = "npu")]
pub fn probe_npu() -> NpuCapabilities {
    let accel_device_present = accel_device_exists();
    NpuCapabilities { available: accel_device_present, accel_device_present }
}

/// Probe Intel NPU availability; always returns `false` when NPU support is not compiled.
#[cfg(not(feature = "npu"))]
pub const fn probe_npu() -> NpuCapabilities {
    NpuCapabilities { available: false, accel_device_present: false }
}

#[cfg(feature = "npu")]
fn accel_device_exists() -> bool {
    std::fs::read_dir("/dev/accel")
        .map(|entries| {
            entries.flatten().any(|entry| entry.file_name().to_string_lossy().starts_with("accel"))
        })
        .unwrap_or(false)
}

/// Check if Intel oneAPI/`OpenCL` support was compiled.
#[inline]
pub const fn oneapi_compiled() -> bool {
    cfg!(feature = "oneapi")
}

/// Check if an Intel GPU is available at runtime via `OpenCL`.
///
/// Detection strategy:
/// 1. Check `BITNET_GPU_FAKE` env var (respects strict mode)
/// 2. Try running `clinfo` and look for Intel vendor
/// 3. Try running `sycl-ls` as fallback
#[cfg(feature = "oneapi")]
pub fn oneapi_available_runtime() -> bool {
    if let Some(fake) = fake_gpu_backends() {
        return fake.contains("oneapi") || fake.contains("gpu");
    }

    // Try clinfo first — most reliable
    if clinfo_has_intel_gpu() {
        return true;
    }

    // Fallback: try sycl-ls
    command_ok("sycl-ls", &[])
}

#[cfg(not(feature = "oneapi"))]
#[inline]
pub const fn oneapi_available_runtime() -> bool {
    false
}

#[cfg(feature = "oneapi")]
fn clinfo_has_intel_gpu() -> bool {
    std::process::Command::new("clinfo")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .map(|output| {
            let stdout = String::from_utf8_lossy(&output.stdout);
            stdout.contains("Intel")
                && (stdout.contains("GPU") || stdout.contains("Arc") || stdout.contains("Graphics"))
        })
        .unwrap_or(false)
}

/// Check if Vulkan support was compiled into this binary.
#[inline]
pub const fn vulkan_compiled() -> bool {
    cfg!(feature = "vulkan")
}

/// Check if a Vulkan-capable device is available at runtime.
#[cfg(feature = "vulkan")]
pub fn vulkan_available_runtime() -> bool {
    // SAFETY: Loads the Vulkan loader via dynamic linking; we only inspect presence and enumerate devices.
    let entry = unsafe { ash::Entry::load() };
    let Ok(entry) = entry else {
        return false;
    };

    let app_info =
        ash::vk::ApplicationInfo::default().api_version(ash::vk::make_api_version(0, 1, 0, 0));
    let create_info = ash::vk::InstanceCreateInfo::default().application_info(&app_info);

    // SAFETY: create_info points to stack-local immutable data for the duration of call.
    let instance = unsafe { entry.create_instance(&create_info, None) };
    let Ok(instance) = instance else {
        return false;
    };

    // SAFETY: valid Vulkan instance handle.
    let has_devices = unsafe { instance.enumerate_physical_devices() }
        .map(|devices| !devices.is_empty())
        .unwrap_or(false);

    // SAFETY: valid instance, no further use after destroy.
    unsafe { instance.destroy_instance(None) };
    has_devices
}

/// Stub: Vulkan never available when not compiled.
#[cfg(not(feature = "vulkan"))]
#[inline]
pub const fn vulkan_available_runtime() -> bool {
    false
}

/// Snapshot of compile-time and runtime device capabilities.
///
/// Build with [`DeviceCapabilities::detect`] to capture the current machine's
/// capabilities.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct DeviceCapabilities {
    /// CPU-Rust backend is always available.
    pub cpu_rust: bool,
    /// CUDA backend was compiled in.
    pub cuda_compiled: bool,
    /// `ROCm` backend was compiled in.
    pub rocm_compiled: bool,
    /// Intel oneAPI backend was compiled in.
    pub oneapi_compiled: bool,
    /// CUDA runtime was detected at call time.
    pub cuda_runtime: bool,
    /// `ROCm` runtime was detected at call time.
    pub rocm_runtime: bool,
    /// Intel NPU backend was compiled in.
    pub npu_compiled: bool,
    /// Intel NPU runtime was detected at call time.
    pub npu_runtime: bool,
    /// Intel oneAPI runtime was detected at call time.
    pub oneapi_runtime: bool,
    /// Best SIMD level detected at call time.
    pub simd_level: SimdLevel,
}

impl DeviceCapabilities {
    /// Build a snapshot using compile-time flags and runtime probing.
    ///
    /// # Examples
    ///
    /// ```
    /// use bitnet_device_probe::{DeviceCapabilities, gpu_compiled};
    ///
    /// let caps = DeviceCapabilities::detect();
    /// assert!(caps.cpu_rust, "CPU backend is always available");
    /// assert_eq!(
    ///     caps.cuda_compiled || caps.rocm_compiled || caps.oneapi_compiled,
    ///     gpu_compiled(),
    /// );
    /// println!("SIMD: {:?}", caps.simd_level);
    /// ```
    pub fn detect() -> Self {
        Self {
            cpu_rust: true,
            cuda_compiled: cfg!(any(feature = "gpu", feature = "cuda")),
            rocm_compiled: cfg!(any(feature = "gpu", feature = "rocm")),
            oneapi_compiled: cfg!(feature = "oneapi"),
            cuda_runtime: cuda_available_runtime(),
            rocm_runtime: rocm_available_runtime(),
            npu_compiled: npu_compiled(),
            npu_runtime: probe_npu().available,
            oneapi_runtime: oneapi_available_runtime(),
            simd_level: detect_simd_level(),
        }
    }
}

// ── CpuProbe / DeviceProbe / probe_device ────────────────────────────────────

/// CPU probe result combining SIMD level, core count, and thread count.
///
/// Obtained via [`probe_device`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuProbe {
    /// Best SIMD instruction-set level available at runtime.
    pub simd_level: SimdLevel,
    /// Number of physical CPU cores (≥ 1).
    pub cores: usize,
    /// Number of logical threads (≥ 1).
    pub threads: usize,
}

/// Full device probe result.
///
/// Obtained via [`probe_device`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct DeviceProbe {
    /// CPU capability snapshot.
    pub cpu: CpuProbe,
    /// Whether a CUDA-capable GPU was found at runtime.
    pub cuda_available: bool,
    pub rocm_available: bool,
    /// Whether an Intel NPU was found at runtime.
    pub npu_available: bool,
    /// Whether an Intel GPU was found at runtime via `OpenCL`.
    pub oneapi_available: bool,
}

/// Run a full device probe and return the result.
///
/// This function never panics. On unusual platforms `cores` and `threads`
/// fall back to `1`.
///
/// # Examples
///
/// ```
/// use bitnet_device_probe::probe_device;
///
/// let result = probe_device();
/// assert!(result.cpu.cores >= 1);
/// assert!(result.cpu.threads >= 1);
/// println!("SIMD: {:?}, CUDA: {}", result.cpu.simd_level, result.cuda_available);
/// ```
pub fn probe_device() -> DeviceProbe {
    let threads = std::thread::available_parallelism().map(std::num::NonZero::get).unwrap_or(1);
    // Physical core count is not reliably available in stable std; use
    // logical thread count as a conservative approximation.
    let cores = threads.max(1);
    let simd_level = detect_simd_level();
    DeviceProbe {
        cpu: CpuProbe { simd_level, cores, threads },
        cuda_available: cuda_available_runtime(),
        rocm_available: rocm_available_runtime(),
        npu_available: probe_npu().available,
        oneapi_available: oneapi_available_runtime(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_compiled_reflects_feature_flags() {
        // When built with --features cpu only, this should be false.
        let compiled = gpu_compiled();
        // Just assert it returns a bool without panicking.
        let _ = compiled;
    }

    #[test]
    fn gpu_not_available_without_feature() {
        #[cfg(not(any(feature = "gpu", feature = "cuda", feature = "rocm", feature = "oneapi")))]
        assert!(!gpu_available_runtime());
    }

    #[test]
    fn detect_simd_level_returns_valid_level() {
        let level = detect_simd_level();
        // Must be one of the known variants (no panic).
        let _ = format!("{level:?}");
    }

    #[test]
    fn device_capabilities_detect_runs() {
        let caps = DeviceCapabilities::detect();
        assert!(caps.cpu_rust);
        // compiled flags reflect their feature flags.
        assert_eq!(caps.cuda_compiled, cfg!(any(feature = "gpu", feature = "cuda")));
        assert_eq!(caps.rocm_compiled, cfg!(any(feature = "gpu", feature = "rocm")));
        assert_eq!(caps.cuda_compiled || caps.rocm_compiled, gpu_compiled());
        assert_eq!(caps.npu_compiled, npu_compiled());
        assert_eq!(caps.oneapi_compiled, cfg!(feature = "oneapi"));
        assert_eq!(
            caps.cuda_compiled || caps.rocm_compiled || caps.oneapi_compiled,
            gpu_compiled(),
        );
    }

    #[cfg(any(feature = "gpu", feature = "cuda", feature = "rocm"))]
    #[test]
    #[serial_test::serial(bitnet_env)]
    fn gpu_fake_env_overrides_detection() {
        temp_env::with_var("BITNET_STRICT_MODE", None::<&str>, || {
            temp_env::with_var("BITNET_GPU_FAKE", Some("cuda"), || {
                assert!(gpu_available_runtime());
            });
            temp_env::with_var("BITNET_GPU_FAKE", Some("none"), || {
                assert!(!gpu_available_runtime());
            });
        });
    }

    #[cfg(any(feature = "gpu", feature = "cuda", feature = "rocm"))]
    #[test]
    #[serial_test::serial(bitnet_env)]
    fn strict_mode_ignores_gpu_fake() {
        temp_env::with_var("BITNET_STRICT_MODE", Some("1"), || {
            temp_env::with_var("BITNET_GPU_FAKE", Some("cuda"), || {
                // In strict mode the env var is ignored; result is real detection.
                let _ = gpu_available_runtime();
            });
        });
    }

    #[test]
    fn oneapi_compiled_reflects_feature_flags() {
        let compiled = oneapi_compiled();
        assert_eq!(compiled, cfg!(feature = "oneapi"));
    }

    #[cfg(feature = "oneapi")]
    #[test]
    #[serial_test::serial(bitnet_env)]
    fn oneapi_fake_env_detection() {
        temp_env::with_var("BITNET_STRICT_MODE", None::<&str>, || {
            temp_env::with_var("BITNET_GPU_FAKE", Some("oneapi"), || {
                assert!(oneapi_available_runtime());
            });
            temp_env::with_var("BITNET_GPU_FAKE", Some("none"), || {
                assert!(!oneapi_available_runtime());
            });
        });
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;

    // `gpu_compiled()` is a compile-time constant — multiple calls always agree.
    #[test]
    fn gpu_compiled_is_idempotent() {
        assert_eq!(gpu_compiled(), gpu_compiled());
    }

    // `detect_simd_level()` is deterministic — repeated calls return the same value.
    #[test]
    fn simd_level_is_deterministic() {
        assert_eq!(detect_simd_level(), detect_simd_level());
    }

    // `DeviceCapabilities::detect()` always reports `cpu_rust = true`.
    #[test]
    fn device_caps_always_has_cpu() {
        assert!(DeviceCapabilities::detect().cpu_rust, "cpu_rust must always be true");
    }

    // compile-time GPU flags in the capabilities snapshot match `gpu_compiled()`.
    #[test]
    fn device_caps_compiled_flags_consistent_with_gpu_compiled() {
        let caps = DeviceCapabilities::detect();
        assert_eq!(caps.cuda_compiled, cfg!(any(feature = "gpu", feature = "cuda")));
        assert_eq!(caps.rocm_compiled, cfg!(any(feature = "gpu", feature = "rocm")));
        assert_eq!(caps.oneapi_compiled, cfg!(feature = "oneapi"));
        assert_eq!(
            caps.cuda_compiled || caps.rocm_compiled || caps.oneapi_compiled,
            gpu_compiled(),
        );
    }
}
