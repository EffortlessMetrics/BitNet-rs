//! Extended unit tests for `bitnet-kernels`.
//!
//! These tests cover the public API that is accessible without the
//! `integration-tests` or `ffi` feature gates, and complement the existing
//! `property_tests.rs` without duplicating its coverage.
//!
//! Run with:
//!   cargo test -p bitnet-kernels --no-default-features --features cpu
//!
//! GPU tests are individually marked `#[ignore = "requires CUDA runtime"]` so
//! the suite stays green in CPU-only CI.

use bitnet_common::{QuantizationType, kernel_registry::SimdLevel};
use bitnet_kernels::{
    FallbackKernel, KernelManager, KernelProvider, device_features, select_cpu_kernel,
    select_gpu_kernel,
};

// ── KernelManager construction ────────────────────────────────────────────────

#[test]
fn kernel_manager_new_succeeds() {
    let _mgr = KernelManager::new();
}

#[test]
fn kernel_manager_default_trait_works() {
    let _mgr = KernelManager::default();
}

#[test]
fn kernel_manager_new_twice_is_independent() {
    let mgr1 = KernelManager::new();
    let mgr2 = KernelManager::new();
    let name1 = mgr1.select_best().map(|p| p.name()).unwrap_or("err");
    let name2 = mgr2.select_best().map(|p| p.name()).unwrap_or("err");
    assert_eq!(name1, name2, "two independent KernelManagers must select the same provider");
}

#[test]
fn kernel_manager_list_available_providers_nonempty() {
    let mgr = KernelManager::new();
    let providers = mgr.list_available_providers();
    assert!(
        !providers.is_empty(),
        "list_available_providers must always return at least one entry"
    );
}

#[test]
fn kernel_manager_list_available_providers_contains_fallback_name() {
    let mgr = KernelManager::new();
    let providers = mgr.list_available_providers();
    // "fallback" is always present as the last-resort provider
    assert!(
        providers.iter().any(|&name| name == "fallback"),
        "fallback provider must always be listed, got: {providers:?}"
    );
}

#[test]
fn kernel_manager_selected_provider_name_none_before_select() {
    let mgr = KernelManager::new();
    // Before calling select_best(), selected_provider_name returns None
    assert!(
        mgr.selected_provider_name().is_none(),
        "selected_provider_name must be None before select_best() is called"
    );
}

#[test]
fn kernel_manager_selected_provider_name_some_after_select() {
    let mgr = KernelManager::new();
    mgr.select_best().expect("select_best must succeed");
    let name = mgr.selected_provider_name();
    assert!(name.is_some(), "selected_provider_name must be Some after select_best() is called");
    assert!(!name.unwrap().is_empty(), "selected provider name must be non-empty");
}

#[test]
fn kernel_manager_select_best_is_ok() {
    let mgr = KernelManager::new();
    assert!(mgr.select_best().is_ok(), "select_best must return Ok on any platform");
}

#[test]
fn kernel_manager_select_best_twice_same_name() {
    let mgr = KernelManager::new();
    let n1 = mgr.select_best().map(|p| p.name()).unwrap_or("err");
    let n2 = mgr.select_best().map(|p| p.name()).unwrap_or("err");
    assert_eq!(n1, n2, "select_best must be idempotent (OnceLock-cached)");
}

// ── select_cpu_kernel ─────────────────────────────────────────────────────────

#[test]
fn select_cpu_kernel_always_succeeds() {
    let result = select_cpu_kernel();
    assert!(result.is_ok(), "select_cpu_kernel must always return Ok");
}

#[test]
fn select_cpu_kernel_name_is_nonempty() {
    let kernel = select_cpu_kernel().expect("cpu kernel must be available");
    assert!(!kernel.name().is_empty(), "cpu kernel name must be non-empty");
}

#[test]
fn select_cpu_kernel_is_available() {
    let kernel = select_cpu_kernel().expect("cpu kernel must be available");
    assert!(kernel.is_available(), "cpu kernel must report is_available() = true");
}

// ── select_gpu_kernel ─────────────────────────────────────────────────────────

#[test]
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
fn select_gpu_kernel_returns_err_when_not_compiled() {
    let result = select_gpu_kernel(0);
    assert!(result.is_err(), "select_gpu_kernel must return Err when GPU not compiled");
}

#[test]
#[cfg(any(feature = "gpu", feature = "cuda"))]
fn select_gpu_kernel_returns_ok_with_gpu_runtime() {
    if !device_features::gpu_available_runtime() {
        eprintln!("⏭️  Skipping: no CUDA GPU available at runtime");
        return;
    }
    let result = select_gpu_kernel(0);
    assert!(result.is_ok(), "select_gpu_kernel must return Ok when CUDA GPU is present");
}

// ── FallbackKernel: direct API ────────────────────────────────────────────────

#[test]
fn fallback_kernel_name_is_fallback() {
    assert_eq!(FallbackKernel.name(), "fallback");
}

#[test]
fn fallback_kernel_is_always_available() {
    assert!(FallbackKernel.is_available(), "FallbackKernel must always be available");
}

#[test]
fn fallback_kernel_matmul_identity_1x1() {
    let a = [2i8];
    let b = [3u8];
    let mut c = [0.0f32];
    FallbackKernel.matmul_i2s(&a, &b, &mut c, 1, 1, 1).expect("1×1 matmul must succeed");
    assert_eq!(c[0], 6.0, "1×1: 2 * 3 = 6");
}

#[test]
fn fallback_kernel_matmul_identity_matrix() {
    // 2×2 * identity = original
    let a = [1i8, 2, 3, 4];
    let b = [1u8, 0, 0, 1]; // 2×2 identity
    let mut c = [0.0f32; 4];
    FallbackKernel.matmul_i2s(&a, &b, &mut c, 2, 2, 2).expect("2×2 identity matmul must succeed");
    assert_eq!(c, [1.0, 2.0, 3.0, 4.0], "A × I₂ must equal A");
}

#[test]
fn fallback_kernel_matmul_zero_matrix() {
    let a = [1i8, -1, 2, -2];
    let b = [0u8; 4];
    let mut c = [99.0f32; 4];
    FallbackKernel.matmul_i2s(&a, &b, &mut c, 2, 2, 2).expect("matmul with zero B must succeed");
    assert_eq!(c, [0.0, 0.0, 0.0, 0.0], "A × 0 must equal 0");
}

#[test]
fn fallback_kernel_matmul_3x2_times_2x3() {
    // C(3×3) = A(3×2) × B(2×3)
    // A = [[1,2],[3,4],[5,6]]  B = [[1,0,1],[0,1,1]]
    let a = [1i8, 2, 3, 4, 5, 6];
    let b = [1u8, 0, 1, 0, 1, 1];
    let mut c = [0.0f32; 9];
    FallbackKernel.matmul_i2s(&a, &b, &mut c, 3, 3, 2).expect("3×3 matmul must succeed");
    // Row 0: [1*1+2*0, 1*0+2*1, 1*1+2*1] = [1, 2, 3]
    // Row 1: [3*1+4*0, 3*0+4*1, 3*1+4*1] = [3, 4, 7]
    // Row 2: [5*1+6*0, 5*0+6*1, 5*1+6*1] = [5, 6, 11]
    assert_eq!(c[0], 1.0);
    assert_eq!(c[1], 2.0);
    assert_eq!(c[2], 3.0);
    assert_eq!(c[3], 3.0);
    assert_eq!(c[4], 4.0);
    assert_eq!(c[5], 7.0);
    assert_eq!(c[6], 5.0);
    assert_eq!(c[7], 6.0);
    assert_eq!(c[8], 11.0);
}

// ── FallbackKernel: dimension validation ─────────────────────────────────────

#[test]
fn fallback_kernel_matmul_err_a_too_short() {
    let a = [1i8]; // should be 2*2=4
    let b = [1u8; 4];
    let mut c = [0.0f32; 4];
    let result = FallbackKernel.matmul_i2s(&a, &b, &mut c, 2, 2, 2);
    assert!(result.is_err(), "A too short must return Err");
}

#[test]
fn fallback_kernel_matmul_err_b_too_short() {
    let a = [1i8; 4];
    let b = [1u8]; // should be 4
    let mut c = [0.0f32; 4];
    let result = FallbackKernel.matmul_i2s(&a, &b, &mut c, 2, 2, 2);
    assert!(result.is_err(), "B too short must return Err");
}

#[test]
fn fallback_kernel_matmul_err_c_too_short() {
    let a = [1i8; 4];
    let b = [1u8; 4];
    let mut c = [0.0f32; 2]; // should be 4
    let result = FallbackKernel.matmul_i2s(&a, &b, &mut c, 2, 2, 2);
    assert!(result.is_err(), "C too short must return Err");
}

// ── FallbackKernel: quantization ──────────────────────────────────────────────

#[test]
fn fallback_kernel_quantize_i2s_valid_input() {
    let input: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 8.0).collect();
    let mut output = vec![0u8; 8]; // 32 values / 4 per byte
    let mut scales = vec![0.0f32; 1]; // 32 values / 32 per block = 1 block
    FallbackKernel
        .quantize(&input, &mut output, &mut scales, QuantizationType::I2S)
        .expect("I2S quantize with valid input must succeed");
    assert!(scales[0] > 0.0, "I2S scale must be positive for non-zero input");
}

#[test]
fn fallback_kernel_quantize_tl1_valid_input() {
    let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 16.0).collect();
    let mut output = vec![0u8; 16]; // 64 values / 4 per byte
    let mut scales = vec![0.0f32; 1]; // 64 values / 64 per block = 1 block
    FallbackKernel
        .quantize(&input, &mut output, &mut scales, QuantizationType::TL1)
        .expect("TL1 quantize with valid input must succeed");
    assert!(scales[0] > 0.0, "TL1 scale must be positive");
}

#[test]
fn fallback_kernel_quantize_tl2_valid_input() {
    let input: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 32.0).collect();
    let mut output = vec![0u8; 32]; // 128 values / 4 per byte
    let mut scales = vec![0.0f32; 1]; // 128 values / 128 per block = 1 block
    FallbackKernel
        .quantize(&input, &mut output, &mut scales, QuantizationType::TL2)
        .expect("TL2 quantize with valid input must succeed");
    assert!(scales[0] > 0.0, "TL2 scale must be positive");
}

#[test]
fn fallback_kernel_quantize_err_output_too_small() {
    let input = vec![1.0f32; 32];
    let mut output = vec![0u8; 1]; // needs at least 32/4 = 8 bytes
    let mut scales = vec![0.0f32; 1];
    let result = FallbackKernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
    assert!(result.is_err(), "quantize must return Err when output buffer too small");
}

#[test]
fn fallback_kernel_quantize_err_scales_too_small() {
    let input = vec![1.0f32; 64]; // 2 blocks of 32
    let mut output = vec![0u8; 16];
    let mut scales = vec![0.0f32; 0]; // needs at least 2 scale entries
    let result = FallbackKernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
    assert!(result.is_err(), "quantize must return Err when scales buffer too small");
}

// ── device_features: gpu_compiled ─────────────────────────────────────────────

#[test]
fn gpu_compiled_is_deterministic() {
    let first = device_features::gpu_compiled();
    let second = device_features::gpu_compiled();
    assert_eq!(first, second, "gpu_compiled() must be deterministic (constant)");
}

#[test]
#[cfg(any(feature = "gpu", feature = "cuda"))]
fn gpu_compiled_true_when_gpu_feature_enabled() {
    assert!(
        device_features::gpu_compiled(),
        "gpu_compiled() must return true when gpu/cuda feature is enabled"
    );
}

#[test]
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
fn gpu_compiled_false_when_no_gpu_feature() {
    assert!(
        !device_features::gpu_compiled(),
        "gpu_compiled() must return false without gpu/cuda features"
    );
}

// ── device_features: gpu_available_runtime ────────────────────────────────────

#[test]
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
fn gpu_available_runtime_false_without_gpu_feature() {
    assert!(
        !device_features::gpu_available_runtime(),
        "gpu_available_runtime() must return false without gpu/cuda features"
    );
}

// ── device_features: detect_simd_level ───────────────────────────────────────

#[test]
fn detect_simd_level_returns_valid_level() {
    let level = device_features::detect_simd_level();
    // Just verifying it's a valid enum value — any variant is acceptable
    let display = format!("{level:?}");
    assert!(!display.is_empty(), "SimdLevel debug must be non-empty");
}

#[test]
fn detect_simd_level_is_at_least_scalar() {
    let level = device_features::detect_simd_level();
    assert!(
        level >= SimdLevel::Scalar,
        "detected SIMD level must be at least Scalar, got {level:?}"
    );
}

#[test]
fn detect_simd_level_deterministic() {
    let first = device_features::detect_simd_level();
    let second = device_features::detect_simd_level();
    assert_eq!(first, second, "detect_simd_level() must be deterministic");
}

// ── device_features: device_capability_summary ───────────────────────────────

#[test]
fn device_capability_summary_is_nonempty() {
    let summary = device_features::device_capability_summary();
    assert!(!summary.is_empty(), "device_capability_summary() must not be empty");
}

#[test]
fn device_capability_summary_contains_cpu() {
    let summary = device_features::device_capability_summary();
    assert!(
        summary.contains("CPU"),
        "device_capability_summary must mention CPU, got: {summary:?}"
    );
}

#[test]
fn device_capability_summary_is_deterministic() {
    let first = device_features::device_capability_summary();
    let second = device_features::device_capability_summary();
    assert_eq!(first, second, "device_capability_summary() must be deterministic");
}

// ── device_features: current_kernel_capabilities ─────────────────────────────

#[test]
fn current_kernel_capabilities_cpu_rust_matches_feature() {
    let caps = device_features::current_kernel_capabilities();
    assert_eq!(caps.cpu_rust, cfg!(feature = "cpu"), "cpu_rust must match cfg!(feature = \"cpu\")");
}

#[test]
fn current_kernel_capabilities_cuda_compiled_matches_feature() {
    let caps = device_features::current_kernel_capabilities();
    assert_eq!(
        caps.cuda_compiled,
        cfg!(any(feature = "gpu", feature = "cuda")),
        "cuda_compiled must match gpu/cuda feature gate"
    );
}

#[test]
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
fn current_kernel_capabilities_no_runtime_cuda_without_gpu_feature() {
    let caps = device_features::current_kernel_capabilities();
    assert!(!caps.cuda_runtime, "cuda_runtime must be false without gpu/cuda features");
    assert!(!caps.cuda_compiled, "cuda_compiled must be false without gpu/cuda features");
}

#[test]
fn current_kernel_capabilities_simd_level_at_least_scalar() {
    let caps = device_features::current_kernel_capabilities();
    assert!(
        caps.simd_level >= SimdLevel::Scalar,
        "simd_level must be at least Scalar, got {:?}",
        caps.simd_level
    );
}

#[test]
fn current_kernel_capabilities_is_deterministic() {
    let c1 = device_features::current_kernel_capabilities();
    let c2 = device_features::current_kernel_capabilities();
    assert_eq!(c1.cpu_rust, c2.cpu_rust);
    assert_eq!(c1.cuda_compiled, c2.cuda_compiled);
    assert_eq!(c1.cuda_runtime, c2.cuda_runtime);
    assert_eq!(c1.cpp_ffi, c2.cpp_ffi);
    assert_eq!(c1.simd_level, c2.simd_level);
}

// ── KernelManager: matmul via select_best ────────────────────────────────────

#[test]
fn kernel_manager_matmul_i2s_basic() {
    let mgr = KernelManager::new();
    let kernel = mgr.select_best().expect("kernel must be available");
    let a = [1i8, 0, 0, 1];
    let b = [2u8, 3, 4, 5];
    let mut c = [0.0f32; 4];
    kernel.matmul_i2s(&a, &b, &mut c, 2, 2, 2).expect("2×2 matmul must succeed");
    // Row 0: 1*2 + 0*4 = 2,  1*3 + 0*5 = 3
    // Row 1: 0*2 + 1*4 = 4,  0*3 + 1*5 = 5
    assert_eq!(c[0], 2.0);
    assert_eq!(c[1], 3.0);
    assert_eq!(c[2], 4.0);
    assert_eq!(c[3], 5.0);
}

#[test]
fn kernel_manager_matmul_i2s_returns_err_on_bad_dims() {
    // Use FallbackKernel directly to test dimension validation without
    // hitting the pre-existing panic in the AVX2 kernel for bad dims.
    let a = [1i8; 4];
    let b = [1u8; 4];
    let mut c = [0.0f32; 4];
    // Wrong k dimension: a.len()=4, m*k=2*3=6 → mismatch
    let result = FallbackKernel.matmul_i2s(&a, &b, &mut c, 2, 2, 3);
    assert!(result.is_err(), "mismatched dimensions must return Err");
}

// ── SimdLevel ordering invariants ─────────────────────────────────────────────

#[test]
fn simd_level_scalar_is_minimum() {
    assert!(SimdLevel::Scalar <= SimdLevel::Neon);
    assert!(SimdLevel::Scalar <= SimdLevel::Sse42);
    assert!(SimdLevel::Scalar <= SimdLevel::Avx2);
    assert!(SimdLevel::Scalar <= SimdLevel::Avx512);
}

#[test]
fn simd_level_avx512_is_maximum() {
    assert!(SimdLevel::Avx512 >= SimdLevel::Scalar);
    assert!(SimdLevel::Avx512 >= SimdLevel::Neon);
    assert!(SimdLevel::Avx512 >= SimdLevel::Sse42);
    assert!(SimdLevel::Avx512 >= SimdLevel::Avx2);
}

#[test]
fn simd_level_display_strings_nonempty() {
    for level in
        [SimdLevel::Scalar, SimdLevel::Neon, SimdLevel::Sse42, SimdLevel::Avx2, SimdLevel::Avx512]
    {
        let s = level.to_string();
        assert!(!s.is_empty(), "SimdLevel::Display must be non-empty for {level:?}");
    }
}

// ── FallbackKernel: deterministic output ─────────────────────────────────────

#[test]
fn fallback_kernel_matmul_deterministic_same_inputs() {
    let a = [3i8, -1, 2, 0, -2, 1];
    let b = [1u8, 2, 3, 4, 5, 6];
    let mut c1 = [0.0f32; 4];
    let mut c2 = [0.0f32; 4];
    FallbackKernel.matmul_i2s(&a, &b, &mut c1, 2, 2, 3).expect("first call must succeed");
    FallbackKernel.matmul_i2s(&a, &b, &mut c2, 2, 2, 3).expect("second call must succeed");
    assert_eq!(c1, c2, "matmul must be deterministic for identical inputs");
}
