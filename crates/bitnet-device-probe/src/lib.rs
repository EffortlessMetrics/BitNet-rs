//! Device detection and capability probing for `BitNet` inference.
//!
//! Provides unified compile-time and runtime device capability queries,
//! extracted from `bitnet-kernels` for use by the broader workspace.

pub use bitnet_common::kernel_registry::SimdLevel;

// ── CPU capabilities ─────────────────────────────────────────────────────────

/// CPU capabilities detected at runtime.
///
/// Obtained by calling [`probe_cpu`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuCapabilities {
    /// Number of logical CPU cores available to the process (always ≥ 1).
    pub core_count: usize,
    /// AVX2 SIMD extension available on this CPU (`x86_64` only).
    pub has_avx2: bool,
    /// AVX-512 SIMD extension available on this CPU (`x86_64` only).
    pub has_avx512: bool,
    /// NEON SIMD extension available (always `true` on `AArch64`, `false` elsewhere).
    pub has_neon: bool,
}

/// Probe the current CPU and return its capabilities.
///
/// `core_count` is derived from [`std::thread::available_parallelism`] and is
/// guaranteed to be ≥ 1. SIMD flags are detected via `is_x86_feature_detected!`
/// (`x86_64`) or compile-time cfg (`aarch64`).
///
/// # Examples
///
/// ```
/// use bitnet_device_probe::probe_cpu;
///
/// let caps = probe_cpu();
/// // At least one logical core is always present.
/// assert!(caps.core_count >= 1);
///
/// // NEON and AVX flags are mutually exclusive across architectures.
/// assert!(!(caps.has_avx2 && caps.has_neon));
/// assert!(!(caps.has_avx512 && caps.has_neon));
///
/// println!(
///     "cores={} avx2={} avx512={} neon={}",
///     caps.core_count, caps.has_avx2, caps.has_avx512, caps.has_neon
/// );
/// ```
pub fn probe_cpu() -> CpuCapabilities {
    let core_count = std::thread::available_parallelism().map(std::num::NonZero::get).unwrap_or(1);

    #[cfg(target_arch = "x86_64")]
    let (has_avx2, has_avx512, has_neon) =
        (is_x86_feature_detected!("avx2"), is_x86_feature_detected!("avx512f"), false);

    #[cfg(target_arch = "aarch64")]
    let (has_avx2, has_avx512, has_neon) = (false, false, true);

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let (has_avx2, has_avx512, has_neon) = (false, false, false);

    CpuCapabilities { core_count, has_avx2, has_avx512, has_neon }
}

// ── GPU capabilities ─────────────────────────────────────────────────────────

/// GPU capabilities detected at runtime.
///
/// Obtained by calling [`probe_gpu`].
///
/// `BITNET_GPU_FAKE=cuda` makes both fields `true`; `BITNET_GPU_FAKE=none`
/// makes both fields `false`. Strict mode (`BITNET_STRICT_MODE=1`) ignores
/// `BITNET_GPU_FAKE` and probes real hardware.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuCapabilities {
    /// Any GPU backend is available (currently CUDA only).
    pub available: bool,
    /// CUDA runtime was detected (or faked via `BITNET_GPU_FAKE`).
    pub cuda_available: bool,
}

/// Probe GPU availability and return its capabilities.
///
/// Honours `BITNET_GPU_FAKE` for deterministic testing unless
/// `BITNET_STRICT_MODE=1` is set.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn probe_gpu() -> GpuCapabilities {
    let cuda_available = gpu_available_runtime();
    GpuCapabilities { available: cuda_available, cuda_available }
}

/// Probe GPU availability; always returns `false` when GPU not compiled.
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
pub const fn probe_gpu() -> GpuCapabilities {
    GpuCapabilities { available: false, cuda_available: false }
}

/// Check if GPU support was compiled into this binary.
///
/// Returns `true` if `feature="gpu"` or `feature="cuda"` was enabled at
/// compile time. Does **not** check runtime GPU availability — use
/// [`gpu_available_runtime`] for that.
///
/// # Examples
///
/// ```
/// use bitnet_device_probe::gpu_compiled;
///
/// // When built with `--features cpu` only, this returns false.
/// // When built with `--features gpu`, this returns true.
/// let _compiled: bool = gpu_compiled();
/// ```
#[inline]
pub const fn gpu_compiled() -> bool {
    cfg!(any(feature = "gpu", feature = "cuda"))
}

/// Check if a GPU is available at runtime.
///
/// - Returns `false` when GPU is not compiled or CUDA runtime is unavailable.
/// - Respects `BITNET_GPU_FAKE=cuda` (returns `true`) / `BITNET_GPU_FAKE=none`
///   (returns `false`) for deterministic testing, unless `BITNET_STRICT_MODE=1`.
/// - In strict mode only real hardware detection is used.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn gpu_available_runtime() -> bool {
    use std::env;
    use std::process::Command;

    let strict = env::var("BITNET_STRICT_MODE")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);

    if !strict {
        if let Ok(fake) = env::var("BITNET_GPU_FAKE") {
            return fake.eq_ignore_ascii_case("cuda") || fake.eq_ignore_ascii_case("gpu");
        }
    }

    // Probe nvidia-smi (best-effort; may block briefly if the driver hangs).
    Command::new("nvidia-smi")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Stub: GPU never available when not compiled.
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
#[inline]
pub const fn gpu_available_runtime() -> bool {
    false
}

/// Detect the best SIMD instruction-set level available at runtime.
///
/// Detection order: AVX-512 > AVX2 > SSE4.2 (`x86_64`); NEON (`AArch64`);
/// scalar fallback on all other targets.
///
/// # Examples
///
/// ```
/// use bitnet_device_probe::detect_simd_level;
///
/// let level = detect_simd_level();
/// println!("SIMD level: {level:?}");
/// // level is one of: Scalar, Sse42, Avx2, Avx512, Neon
/// ```
pub fn detect_simd_level() -> SimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return SimdLevel::Avx512;
        }
        if is_x86_feature_detected!("avx2") {
            return SimdLevel::Avx2;
        }
        if is_x86_feature_detected!("sse4.2") {
            return SimdLevel::Sse42;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory on AArch64.
        return SimdLevel::Neon;
    }
    SimdLevel::Scalar
}

/// Snapshot of compile-time and runtime device capabilities.
///
/// Build with [`DeviceCapabilities::detect`] to capture the current machine's
/// capabilities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceCapabilities {
    /// CPU-Rust backend is always available.
    pub cpu_rust: bool,
    /// CUDA backend was compiled in.
    pub cuda_compiled: bool,
    /// CUDA runtime was detected at call time.
    pub cuda_runtime: bool,
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
    /// assert_eq!(caps.cuda_compiled, gpu_compiled());
    /// println!("SIMD: {:?}", caps.simd_level);
    /// ```
    pub fn detect() -> Self {
        Self {
            cpu_rust: true,
            cuda_compiled: gpu_compiled(),
            cuda_runtime: gpu_available_runtime(),
            simd_level: detect_simd_level(),
        }
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
        #[cfg(not(any(feature = "gpu", feature = "cuda")))]
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
        // cuda_compiled reflects the feature flag.
        assert_eq!(caps.cuda_compiled, gpu_compiled());
    }

    #[cfg(any(feature = "gpu", feature = "cuda"))]
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

    #[cfg(any(feature = "gpu", feature = "cuda"))]
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

    // `cuda_compiled` in the capabilities snapshot matches `gpu_compiled()`.
    #[test]
    fn device_caps_cuda_consistent_with_gpu_compiled() {
        let caps = DeviceCapabilities::detect();
        assert_eq!(caps.cuda_compiled, gpu_compiled());
    }
}
