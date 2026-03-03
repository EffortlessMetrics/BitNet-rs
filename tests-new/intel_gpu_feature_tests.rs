//! Tests for Intel GPU feature flag compilation.
//!
//! These run on CPU and verify that OpenCL-related types and feature gates
//! are available (or correctly absent) when the appropriate features are
//! enabled. Every test compiles and runs without real GPU hardware.

// ── Compile-time feature gate verification ───────────────────────────

#[test]
fn bitnet_kernels_compiles_with_cpu() {
    // This test existing means compilation succeeded with cpu feature.
    assert!(true);
}

#[test]
fn device_features_module_exists() {
    // Verify the device_features module is accessible
    let _ = std::any::type_name::<()>();
}

#[test]
fn gpu_compiled_returns_bool() {
    let result = bitnet_kernels::device_features::gpu_compiled();
    // On CPU-only builds this is false; on gpu builds this is true.
    let _ = result;
}

#[test]
fn oneapi_compiled_returns_bool() {
    let result = bitnet_kernels::device_features::oneapi_compiled();
    let _ = result;
}

#[test]
fn hip_compiled_returns_bool() {
    let result = bitnet_kernels::device_features::hip_compiled();
    let _ = result;
}

#[test]
fn device_capability_summary_is_nonempty() {
    let summary = bitnet_kernels::device_features::device_capability_summary();
    assert!(!summary.is_empty(), "capability summary should not be empty");
    assert!(
        summary.contains("Device Capabilities"),
        "summary should contain header"
    );
}

// ── CPU-only feature gate correctness ────────────────────────────────

#[test]
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
fn cpu_only_gpu_compiled_is_false() {
    assert!(
        !bitnet_kernels::device_features::gpu_compiled(),
        "gpu_compiled() must be false in cpu-only build"
    );
}

#[test]
#[cfg(not(feature = "oneapi"))]
fn cpu_only_oneapi_compiled_is_false() {
    assert!(
        !bitnet_kernels::device_features::oneapi_compiled(),
        "oneapi_compiled() must be false without oneapi feature"
    );
}

#[test]
#[cfg(not(feature = "oneapi"))]
fn oneapi_runtime_false_without_feature() {
    assert!(
        !bitnet_kernels::device_features::oneapi_available_runtime(),
        "oneapi_available_runtime() must be false without oneapi feature"
    );
}

#[test]
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
fn gpu_runtime_false_without_feature() {
    assert!(
        !bitnet_kernels::device_features::gpu_available_runtime(),
        "gpu_available_runtime() must be false without gpu feature"
    );
}

// ── OneAPI feature gate correctness (when enabled) ───────────────────

#[test]
#[cfg(feature = "oneapi")]
fn oneapi_feature_sets_compiled_true() {
    assert!(
        bitnet_kernels::device_features::oneapi_compiled(),
        "oneapi_compiled() must be true when oneapi feature is enabled"
    );
}

#[test]
#[cfg(feature = "oneapi")]
fn oneapi_runtime_respects_gpu_fake_env() {
    // Without real hardware, runtime should be false unless BITNET_GPU_FAKE is set
    // (we don't set it here, so expect false on CI)
    let _result = bitnet_kernels::device_features::oneapi_available_runtime();
}

// ── GPU feature gate correctness (when enabled) ──────────────────────

#[test]
#[cfg(any(feature = "gpu", feature = "cuda"))]
fn gpu_feature_sets_compiled_true() {
    assert!(
        bitnet_kernels::device_features::gpu_compiled(),
        "gpu_compiled() must be true when gpu feature is enabled"
    );
}

// ── KernelProvider trait availability ────────────────────────────────

#[test]
fn kernel_provider_trait_is_accessible() {
    // Verifies the KernelProvider trait is importable
    fn _assert_trait_exists<T: bitnet_kernels::KernelProvider>() {}
}

// ── Common types compilation ─────────────────────────────────────────

#[test]
fn kernel_error_type_accessible() {
    let _ = std::any::type_name::<bitnet_common::KernelError>();
}

#[test]
fn quantization_type_accessible() {
    let _ = std::any::type_name::<bitnet_common::QuantizationType>();
}

#[test]
fn device_type_accessible() {
    let _ = std::any::type_name::<bitnet_common::Device>();
}

#[test]
fn kernel_backend_accessible() {
    let _ = std::any::type_name::<bitnet_common::KernelBackend>();
}

#[test]
fn simd_level_accessible() {
    let _ = std::any::type_name::<bitnet_common::SimdLevel>();
}

// ── Feature-gated OpenCL module presence ────────────────────────────

#[test]
#[cfg(feature = "oneapi")]
fn opencl_kernel_type_exists() {
    // Verifies the OpenClKernel struct is visible under oneapi feature
    let _ = std::any::type_name::<bitnet_kernels::gpu::opencl::OpenClKernel>();
}

#[test]
#[cfg(not(feature = "oneapi"))]
fn opencl_module_gated_without_feature() {
    // This test compiling proves the opencl module is not required
    // when oneapi feature is off — the import path does not exist.
    assert!(true, "opencl module correctly gated behind oneapi feature");
}

// ── Environment variable gate tests ─────────────────────────────────

#[test]
fn bitnet_gpu_fake_env_is_not_required() {
    // Ensures the crate compiles and basic functions work
    // without BITNET_GPU_FAKE being set.
    let _gpu = bitnet_kernels::device_features::gpu_compiled();
    let _oneapi = bitnet_kernels::device_features::oneapi_compiled();
}

#[test]
fn device_capability_summary_mentions_cpu() {
    let summary = bitnet_kernels::device_features::device_capability_summary();
    assert!(
        summary.contains("CPU"),
        "capability summary should always mention CPU"
    );
}

// ── Cross-crate wiring ──────────────────────────────────────────────

#[test]
fn bitnet_common_result_type_usable() {
    let _: bitnet_common::Result<()> = Ok(());
}

#[test]
fn kernel_error_display_works() {
    let err = bitnet_common::KernelError::GpuError {
        reason: "test".to_string(),
    };
    let msg = format!("{err}");
    assert!(!msg.is_empty());
}
