//! BDD-style and property-based specifications for `bitnet-kernels`.
//!
//! Each test function is named `given_<context>_when_<action>_then_<outcome>`.
//! These tests act as executable specifications describing the observable
//! behaviour of `KernelManager`, `KernelProvider`, SIMD detection, and the
//! scalar math path.

use bitnet_common::{QuantizationType, kernel_registry::SimdLevel};
use bitnet_kernels::{
    FallbackKernel, KernelManager, KernelProvider, device_features, select_cpu_kernel,
};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Section 1 – KernelManager initialization
// ---------------------------------------------------------------------------

/// Without GPU support compiled in, `KernelManager::new()` must still build
/// successfully and expose at least the scalar fallback provider.
#[test]
fn given_no_gpu_when_creating_manager_then_cpu_fallback_is_available() {
    let mgr = KernelManager::new();
    let providers = mgr.list_available_providers();

    assert!(!providers.is_empty(), "manager must expose at least one provider");
    assert!(
        providers.contains(&"fallback"),
        "fallback provider must always be present; got {providers:?}"
    );
}

/// Regardless of hardware, `KernelManager::new()` must succeed and the
/// selected provider must report itself as available.
#[test]
fn given_any_hardware_when_creating_manager_then_selected_provider_is_available() {
    let mgr = KernelManager::new();
    let provider = mgr.select_best().expect("select_best must succeed");
    assert!(provider.is_available(), "selected provider must be available");
}

/// The `cpu` feature compiled without GPU must yield a provider whose name
/// does NOT start with "cuda".
#[test]
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
fn given_cpu_only_build_when_creating_manager_then_provider_name_is_not_cuda() {
    let mgr = KernelManager::new();
    let provider = mgr.select_best().expect("select_best must succeed");
    assert!(
        !provider.name().starts_with("cuda"),
        "CPU-only build must not select a CUDA provider; got {:?}",
        provider.name()
    );
}

// ---------------------------------------------------------------------------
// Section 2 – Kernel capability reporting
// ---------------------------------------------------------------------------

/// A freshly constructed `KernelManager` must return a non-empty capability list.
#[test]
fn given_a_manager_when_querying_capabilities_then_returns_non_empty_list() {
    let mgr = KernelManager::new();
    let providers = mgr.list_available_providers();
    assert!(!providers.is_empty(), "capability list must not be empty");
}

/// Every provider name reported by the manager must be a non-empty string.
#[test]
fn given_a_manager_when_listing_providers_then_all_names_are_nonempty_strings() {
    let mgr = KernelManager::new();
    for name in mgr.list_available_providers() {
        assert!(!name.is_empty(), "provider name must not be an empty string");
    }
}

/// `FallbackKernel::name()` must always return the static string `"fallback"`.
#[test]
fn given_fallback_kernel_when_querying_name_then_returns_fallback() {
    let kernel = FallbackKernel;
    assert_eq!(kernel.name(), "fallback");
}

/// `FallbackKernel::is_available()` must always return `true`.
#[test]
fn given_fallback_kernel_when_checking_availability_then_always_true() {
    let kernel = FallbackKernel;
    assert!(kernel.is_available(), "fallback kernel must always be available");
}

/// `select_cpu_kernel()` must succeed and the returned provider must be available.
#[test]
fn given_cpu_feature_when_selecting_cpu_kernel_then_available_provider_returned() {
    let provider = select_cpu_kernel().expect("select_cpu_kernel must succeed");
    assert!(provider.is_available());
    assert!(!provider.name().is_empty());
}

// ---------------------------------------------------------------------------
// Section 3 – SIMD level detection
// ---------------------------------------------------------------------------

/// Hardware probing must return a value that is a member of the `SimdLevel`
/// discriminant set (i.e. the result must be a valid enum variant).
#[test]
fn given_current_hardware_when_probing_simd_then_returns_valid_simd_level() {
    let level = device_features::detect_simd_level();
    // Verify the result is one of the known variants by checking Display output.
    let s = level.to_string();
    assert!(
        ["scalar", "neon", "sse4.2", "avx2", "avx512"].contains(&s.as_str()),
        "detect_simd_level returned unexpected variant string: {s}"
    );
}

/// The detected SIMD level must be at least `Scalar` (the minimum valid level).
#[test]
fn given_current_hardware_when_probing_simd_then_level_is_at_least_scalar() {
    let level = device_features::detect_simd_level();
    assert!(level >= SimdLevel::Scalar, "SIMD level must be ≥ Scalar");
}

/// `current_kernel_capabilities()` must reflect `cpu` feature gate in
/// `cpu_rust` field.
#[test]
fn given_cpu_feature_flag_when_querying_capabilities_then_cpu_rust_matches_feature() {
    let caps = device_features::current_kernel_capabilities();
    let expected_cpu_rust = cfg!(feature = "cpu");
    assert_eq!(
        caps.cpu_rust, expected_cpu_rust,
        "cpu_rust must match the 'cpu' feature flag"
    );
}

/// Without GPU compiled in, `gpu_compiled()` must return `false`.
#[test]
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
fn given_no_gpu_feature_when_checking_gpu_compiled_then_returns_false() {
    assert!(!device_features::gpu_compiled(), "gpu_compiled must be false without GPU features");
}

/// Without GPU compiled in, `gpu_available_runtime()` must return `false`.
#[test]
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
fn given_no_gpu_feature_when_checking_runtime_gpu_then_returns_false() {
    assert!(
        !device_features::gpu_available_runtime(),
        "gpu_available_runtime must return false when not compiled with GPU support"
    );
}

// ---------------------------------------------------------------------------
// Section 4 – Kernel selection invariants
// ---------------------------------------------------------------------------

/// Calling `select_best()` twice on the same manager must return providers with
/// the same name (cached selection via `OnceLock`).
#[test]
fn given_same_manager_when_selecting_kernel_twice_then_same_kernel_returned() {
    let mgr = KernelManager::new();
    let first = mgr.select_best().expect("first select_best must succeed").name();
    let second = mgr.select_best().expect("second select_best must succeed").name();
    assert_eq!(first, second, "kernel selection must be idempotent (cached)");
}

/// After `select_best()` is called, `selected_provider_name()` must return the
/// same name.
#[test]
fn given_manager_when_select_best_called_then_selected_provider_name_matches() {
    let mgr = KernelManager::new();
    let selected_name = mgr.select_best().expect("select_best must succeed").name();
    let reported_name =
        mgr.selected_provider_name().expect("selected_provider_name must be Some after selection");
    assert_eq!(selected_name, reported_name);
}

/// Before `select_best()` is ever called, `selected_provider_name()` must
/// return `None`.
#[test]
fn given_fresh_manager_when_querying_selected_name_before_any_selection_then_none() {
    let mgr = KernelManager::new();
    // A brand-new manager has not had `select_best()` called yet.
    assert!(
        mgr.selected_provider_name().is_none(),
        "selected_provider_name must be None before select_best() is first called"
    );
}

// ---------------------------------------------------------------------------
// Section 5 – Scalar math correctness
// ---------------------------------------------------------------------------

/// Multiplying a matrix by the identity matrix must return the original values.
#[test]
fn given_identity_matrix_when_matmul_i2s_then_output_equals_input() {
    let kernel = FallbackKernel;

    // 3×3 * 3×3 identity
    let a: Vec<i8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9]; // 3×3
    #[rustfmt::skip]
    let b: Vec<u8> = vec![
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    ]; // identity
    let mut c = vec![0.0f32; 9];

    kernel.matmul_i2s(&a, &b, &mut c, 3, 3, 3).expect("matmul_i2s must succeed");

    let expected: Vec<f32> = a.iter().map(|&v| v as f32).collect();
    assert_eq!(c, expected, "A × I₃ must equal A element-wise");
}

/// Multiplying any matrix by a zero matrix must produce an all-zero output.
#[test]
fn given_zero_weight_matrix_when_matmul_i2s_then_output_is_all_zeros() {
    let kernel = FallbackKernel;

    let a: Vec<i8> = vec![1, -1, 2, -3, 5, 7];  // 2×3
    let b: Vec<u8> = vec![0; 3 * 4];             // zero 3×4
    let mut c = vec![99.0f32; 2 * 4];

    kernel.matmul_i2s(&a, &b, &mut c, 2, 4, 3).expect("matmul_i2s must succeed");

    assert!(
        c.iter().all(|&v| v == 0.0),
        "zero weight matrix must produce all-zero output; got {c:?}"
    );
}

/// A simple 1×k · k×1 dot-product (row × column) must equal the manual sum.
#[test]
fn given_row_and_column_vectors_when_matmul_i2s_then_result_is_correct_dot_product() {
    let kernel = FallbackKernel;

    // row = [1, 2, 3], col = [4, 5, 6]  →  dot = 1·4 + 2·5 + 3·6 = 32
    let a: Vec<i8> = vec![1, 2, 3];
    let b: Vec<u8> = vec![4, 5, 6];
    let mut c = vec![0.0f32; 1];

    kernel.matmul_i2s(&a, &b, &mut c, 1, 1, 3).expect("matmul_i2s must succeed");

    assert_eq!(c[0], 32.0, "dot product [1,2,3]·[4,5,6] must equal 32");
}

/// Scalar-by-scalar single-element multiplication: a=[v], b=[w] → c=[v*w].
#[test]
fn given_single_element_inputs_when_matmul_i2s_then_result_is_product() {
    let kernel = FallbackKernel;

    let a: Vec<i8> = vec![7];
    let b: Vec<u8> = vec![3];
    let mut c = vec![0.0f32; 1];

    kernel.matmul_i2s(&a, &b, &mut c, 1, 1, 1).expect("matmul_i2s must succeed");

    assert_eq!(c[0], 21.0, "1×1 matmul must equal scalar product");
}

/// Dimension mismatch (A too short) must return `Err`.
#[test]
fn given_mismatched_dimensions_when_matmul_i2s_then_returns_error() {
    let kernel = FallbackKernel;

    let a: Vec<i8> = vec![1, 2]; // only 2 elements; need m*k = 2*3 = 6
    let b: Vec<u8> = vec![0; 3 * 4];
    let mut c = vec![0.0f32; 2 * 4];

    let result = kernel.matmul_i2s(&a, &b, &mut c, 2, 4, 3);
    assert!(result.is_err(), "dimension mismatch must return Err");
}

/// I2S quantization of non-zero inputs must produce a positive scale.
#[test]
fn given_nonzero_input_when_quantize_i2s_then_scale_is_positive() {
    let kernel = FallbackKernel;

    let input: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 + 0.1).collect();
    let mut output = vec![0u8; 8]; // 32 / 4
    let mut scales = vec![0.0f32; 1];

    kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S).unwrap();

    assert!(scales[0] > 0.0, "non-zero input must produce positive scale; got {}", scales[0]);
}

/// I2S quantization of all-zero input must complete without error.
#[test]
fn given_all_zero_input_when_quantize_i2s_then_succeeds_without_panic() {
    let kernel = FallbackKernel;

    let input = vec![0.0f32; 32];
    let mut output = vec![0u8; 8];
    let mut scales = vec![0.0f32; 1];

    kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S).unwrap();
    // all-zero input is valid; no assertion on numeric values
}

/// Undersized output buffer must return `Err`.
#[test]
fn given_undersized_output_buffer_when_quantize_i2s_then_returns_error() {
    let kernel = FallbackKernel;

    let input = vec![1.0f32; 32];
    let mut output = vec![0u8; 1]; // too small (need 8)
    let mut scales = vec![0.0f32; 1];

    let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
    assert!(result.is_err(), "undersized output buffer must return Err");
}

// ---------------------------------------------------------------------------
// Section 6 – Property-based specifications
// ---------------------------------------------------------------------------

proptest! {
    /// For any valid m×k · k×n matmul the output must contain exactly m×n elements
    /// and the call must succeed.
    #[test]
    fn prop_given_valid_dimensions_when_matmul_then_output_has_correct_shape(
        m in 1usize..=8,
        n in 1usize..=8,
        k in 1usize..=8,
    ) {
        let kernel = FallbackKernel;
        let a: Vec<i8> = vec![1i8; m * k];
        let b: Vec<u8> = vec![1u8; k * n];
        let mut c = vec![0.0f32; m * n];

        kernel.matmul_i2s(&a, &b, &mut c, m, n, k)
              .expect("valid dimensions must succeed");

        prop_assert_eq!(c.len(), m * n,
            "output must contain m×n = {} elements", m * n);
    }

    /// Multiplying a row vector by a zero column must always yield zero, for any
    /// dimension k.
    #[test]
    fn prop_given_zero_b_when_matmul_then_output_is_always_zero(
        k in 1usize..=16,
        a_val in -127i8..=127i8,
    ) {
        let kernel = FallbackKernel;
        let a = vec![a_val; k];
        let b = vec![0u8; k];
        let mut c = vec![99.0f32; 1];

        kernel.matmul_i2s(&a, &b, &mut c, 1, 1, k).expect("must succeed");
        prop_assert_eq!(c[0], 0.0, "zero B must yield zero output");
    }

    /// The fallback kernel must always be present in the provider list,
    /// regardless of how many times the manager is recreated.
    #[test]
    fn prop_given_any_manager_when_listing_providers_then_fallback_always_present(
        _seed in 0u32..=100u32,
    ) {
        let mgr = KernelManager::new();
        let providers = mgr.list_available_providers();
        prop_assert!(providers.contains(&"fallback"),
            "fallback must always be listed; got {providers:?}");
    }

    /// `select_best()` must be deterministic: calling it n times must always
    /// return the same provider name.
    #[test]
    fn prop_given_manager_when_select_best_called_n_times_then_always_same_name(
        n in 1usize..=5,
    ) {
        let mgr = KernelManager::new();
        let first = mgr.select_best().expect("first call must succeed").name();
        for _ in 1..n {
            let name = mgr.select_best().expect("repeated call must succeed").name();
            prop_assert_eq!(first, name, "select_best must return the same name on every call");
        }
    }

    /// I2S quantization of any non-trivially-small input with valid buffer sizes
    /// must succeed.
    #[test]
    fn prop_given_valid_buffers_when_quantize_i2s_then_always_succeeds(
        len_blocks in 1usize..=4,
    ) {
        let kernel = FallbackKernel;
        let len = len_blocks * 32; // multiples of block size
        let input: Vec<f32> = (0..len).map(|i| (i as f32) * 0.01).collect();
        let mut output = vec![0u8; len / 4];
        let mut scales = vec![0.0f32; len_blocks];

        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
        prop_assert!(result.is_ok(), "quantize must succeed for valid buffers");
    }

    /// `detect_simd_level()` must be idempotent: two calls on the same thread
    /// must return equal values.
    #[test]
    fn prop_given_current_hardware_when_probing_simd_twice_then_results_are_equal(
        _seed in 0u32..=10u32,
    ) {
        let first = device_features::detect_simd_level();
        let second = device_features::detect_simd_level();
        prop_assert_eq!(first, second, "detect_simd_level must be deterministic");
    }
}

// ---------------------------------------------------------------------------
// Section 7 – GPU guard: GPU-only specifications
// ---------------------------------------------------------------------------

#[cfg(any(feature = "gpu", feature = "cuda"))]
mod gpu_specs {
    use bitnet_kernels::device_features;

    /// When GPU features are compiled in, `gpu_compiled()` must return `true`.
    #[test]
    fn given_gpu_feature_when_checking_gpu_compiled_then_returns_true() {
        assert!(device_features::gpu_compiled(), "gpu_compiled must be true when GPU feature enabled");
    }
}
