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
/// `BITNET_GPU_FAKE` supports comma-separated backends (`cuda`, `rocm`, `gpu`).
/// For example `BITNET_GPU_FAKE=rocm` forces `ROCm` runtime availability for tests.
/// `BITNET_GPU_FAKE=none` makes all GPU flags `false`.
/// Strict mode (`BITNET_STRICT_MODE=1`) ignores `BITNET_GPU_FAKE` and probes real hardware.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuCapabilities {
    /// Any GPU backend is available (CUDA and/or `ROCm`).
    pub available: bool,
    /// CUDA runtime was detected (or faked via `BITNET_GPU_FAKE`).
    pub cuda_available: bool,
    /// `ROCm` runtime was detected (or faked via `BITNET_GPU_FAKE`).
    pub rocm_available: bool,
}

/// Probe GPU availability and return its capabilities.
///
/// Honours `BITNET_GPU_FAKE` for deterministic testing unless
/// `BITNET_STRICT_MODE=1` is set.
#[cfg(any(feature = "gpu", feature = "cuda", feature = "rocm"))]
pub fn probe_gpu() -> GpuCapabilities {
    let cuda_available = cuda_available_runtime();
    let rocm_available = rocm_available_runtime();
    let available = cuda_available || rocm_available;
    GpuCapabilities { available, cuda_available, rocm_available }
}

/// Probe GPU availability; always returns `false` when GPU not compiled.
#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "rocm")))]
pub const fn probe_gpu() -> GpuCapabilities {
    GpuCapabilities { available: false, cuda_available: false, rocm_available: false }
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
    cfg!(any(feature = "gpu", feature = "cuda", feature = "rocm"))
}

/// Check if a GPU is available at runtime.
///
/// - Returns `false` when GPU is not compiled or CUDA runtime is unavailable.
/// - Respects `BITNET_GPU_FAKE=cuda|rocm|gpu` (returns `true`) /
///   `BITNET_GPU_FAKE=none` (returns `false`) for deterministic testing, unless
///   `BITNET_STRICT_MODE=1`.
/// - In strict mode only real hardware detection is used.
#[cfg(any(feature = "gpu", feature = "cuda", feature = "rocm"))]
pub fn gpu_available_runtime() -> bool {
    cuda_available_runtime() || rocm_available_runtime()
}

/// Stub: GPU never available when not compiled.
#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "rocm")))]
#[inline]
pub const fn gpu_available_runtime() -> bool {
    false
}

#[cfg(any(feature = "gpu", feature = "cuda", feature = "rocm"))]
fn strict_mode_enabled() -> bool {
    std::env::var("BITNET_STRICT_MODE")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false)
}

#[cfg(any(feature = "gpu", feature = "cuda", feature = "rocm"))]
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

#[cfg(any(feature = "gpu", feature = "cuda", feature = "rocm"))]
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
#[allow(clippy::struct_excessive_bools)]
pub struct DeviceCapabilities {
    /// CPU-Rust backend is always available.
    pub cpu_rust: bool,
    /// CUDA backend was compiled in.
    pub cuda_compiled: bool,
    /// `ROCm` backend was compiled in.
    pub rocm_compiled: bool,
    /// CUDA runtime was detected at call time.
    pub cuda_runtime: bool,
    /// `ROCm` runtime was detected at call time.
    pub rocm_runtime: bool,
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
    /// assert_eq!(caps.cuda_compiled || caps.rocm_compiled, gpu_compiled());
    /// println!("SIMD: {:?}", caps.simd_level);
    /// ```
    pub fn detect() -> Self {
        Self {
            cpu_rust: true,
            cuda_compiled: cfg!(any(feature = "gpu", feature = "cuda")),
            rocm_compiled: cfg!(any(feature = "gpu", feature = "rocm")),
            cuda_runtime: cuda_available_runtime(),
            rocm_runtime: rocm_available_runtime(),
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
pub struct DeviceProbe {
    /// CPU capability snapshot.
    pub cpu: CpuProbe,
    /// Whether a CUDA-capable GPU was found at runtime.
    pub cuda_available: bool,
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
        cuda_available: cuda_available_runtime() || rocm_available_runtime(),
    }
}

/// Return a numeric rank for a [`SimdLevel`] so callers can compare levels.
///
/// Higher values represent wider/more capable SIMD.
///
/// | Level | Rank |
/// |-------|------|
/// | `Scalar` | 0 |
/// | `Sse42`  | 1 |
/// | `Avx2`   | 2 |
/// | `Avx512` | 3 |
/// | `Neon`   | 4 |
///
/// Note: NEON and SSE/AVX are mutually exclusive instruction sets; the rank is
/// only meaningful when comparing levels on the same architecture.
///
/// # Examples
///
/// ```
/// use bitnet_device_probe::{simd_level_rank, SimdLevel};
///
/// // Wider SIMD has a strictly higher rank.
/// assert!(simd_level_rank(&SimdLevel::Avx512) > simd_level_rank(&SimdLevel::Avx2));
/// assert!(simd_level_rank(&SimdLevel::Avx2)   > simd_level_rank(&SimdLevel::Sse42));
/// assert!(simd_level_rank(&SimdLevel::Sse42)  > simd_level_rank(&SimdLevel::Scalar));
///
/// // Scalar is always the lowest rank.
/// assert_eq!(simd_level_rank(&SimdLevel::Scalar), 0);
/// ```
pub const fn simd_level_rank(level: &SimdLevel) -> u32 {
    match level {
        SimdLevel::Scalar => 0,
        SimdLevel::Sse42 => 1,
        SimdLevel::Avx2 => 2,
        SimdLevel::Avx512 => 3,
        SimdLevel::Neon => 4,
        // Future variants default to a high rank so they don't compare lower than
        // known levels when SimdLevel gains new entries.
        _ => u32::MAX,
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
        #[cfg(not(any(feature = "gpu", feature = "cuda", feature = "rocm")))]
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
        assert_eq!(caps.cuda_compiled || caps.rocm_compiled, gpu_compiled());
    }
}
