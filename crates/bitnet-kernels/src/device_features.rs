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
    cfg!(any(feature = "gpu", feature = "cuda"))
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
#[cfg(any(feature = "gpu", feature = "cuda"))]
#[inline]
pub fn gpu_available_runtime() -> bool {
    use std::env;

    // In strict mode, refuse BITNET_GPU_FAKE and always use real detection
    let strict_mode = env::var("BITNET_STRICT_MODE")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);

    if strict_mode {
        // Strict mode: only real GPU detection
        return crate::gpu_utils::get_gpu_info().cuda;
    }

    // Check BITNET_GPU_FAKE first (deterministic testing)
    if let Ok(fake) = env::var("BITNET_GPU_FAKE") {
        return fake.eq_ignore_ascii_case("cuda") || fake.eq_ignore_ascii_case("gpu");
    }

    // Fall back to real GPU detection
    crate::gpu_utils::get_gpu_info().cuda
}

/// Stub implementation when GPU not compiled
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
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

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        if gpu_available_runtime() {
            let info = crate::gpu_utils::get_gpu_info();
            if let Some(version) = &info.cuda_version {
                summary.push_str(&format!("CUDA {} ✓", version));
            } else {
                summary.push_str("CUDA ✓");
            }
        } else {
            summary.push_str("CUDA ✗");
        }
        summary.push_str(", ");
    }

    summary.push_str("CPU ✓");

    summary
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
        cuda_runtime: gpu_available_runtime(),
        cpp_ffi: false,
        simd_level: detect_simd_level(),
    }
}
