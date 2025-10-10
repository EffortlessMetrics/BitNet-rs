//! Device feature detection and capability queries for Issue #439
//!
//! This module provides unified device capability checks for GPU/CPU selection
//! across the BitNet.rs workspace. It consolidates compile-time feature gates
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
/// # Implementation Status
///
/// STUB - Implementation required for AC:3
///
/// # Specification
///
/// Tests specification: docs/explanation/issue-439-spec.md#ac3-shared-helpers
#[inline]
pub fn gpu_compiled() -> bool {
    // TODO: Implement unified GPU feature detection
    // Expected implementation:
    // cfg!(any(feature = "gpu", feature = "cuda"))
    unimplemented!("AC:3 - gpu_compiled() requires implementation")
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
///
/// # Implementation Status
///
/// STUB - Implementation required for AC:3
///
/// # Specification
///
/// Tests specification: docs/explanation/issue-439-spec.md#ac3-shared-helpers
#[cfg(any(feature = "gpu", feature = "cuda"))]
#[inline]
pub fn gpu_available_runtime() -> bool {
    // TODO: Implement GPU runtime detection with BITNET_GPU_FAKE precedence
    // Expected implementation:
    // 1. Check BITNET_GPU_FAKE environment variable first (deterministic testing)
    // 2. Fall back to real CUDA detection via gpu_utils::get_gpu_info()
    unimplemented!("AC:3 - gpu_available_runtime() requires implementation")
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
/// # Implementation Status
///
/// STUB - Implementation required for AC:3
///
/// # Example Output
///
/// ```text
/// Device Capabilities:
///   Compiled: GPU ✓, CPU ✓
///   Runtime: CUDA 12.1 ✓, cuBLAS ✓
/// ```
///
/// # Specification
///
/// Tests specification: docs/explanation/issue-439-spec.md#device-feature-detection-api
pub fn device_capability_summary() -> String {
    // TODO: Implement diagnostic summary
    // Expected implementation:
    // 1. Report compile-time capabilities (gpu_compiled())
    // 2. Report runtime capabilities (gpu_available_runtime())
    // 3. Include CUDA version if available
    unimplemented!("AC:3 - device_capability_summary() requires implementation")
}
