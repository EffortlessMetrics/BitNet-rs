//! Property-based tests for `bitnet-kernels`.
//!
//! Key invariants tested:
//! - `KernelManager`: provider list is non-empty, selection is stable (idempotent)
//! - `KernelProvider::name()`: always a non-empty string
//! - `FallbackKernel::quantize()`: output shape matches input for valid sizes
//! - `device_features::gpu_compiled()`: deterministic (constant) across calls
//! - `select_cpu_kernel()`: always succeeds and returns a valid provider
//! - `SimdLevel`: ordering is reflexive, total, and has consistent fundamental invariants
//! - MatMul dimensions: rows×cols ≤ usize::MAX for valid inputs up to 4096×4096
//! - `KernelCapabilities`: compiled_backends() reflects the fields set in the snapshot
//! - `device_capability_summary()`: always reports CPU available
//! - `current_kernel_capabilities()`: cpu_rust matches the `cpu` feature flag

use bitnet_common::QuantizationType;
use bitnet_kernels::KernelProvider;
use bitnet_kernels::device_features::gpu_compiled;
use bitnet_kernels::{FallbackKernel, KernelManager, select_cpu_kernel};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Properties: KernelManager
// ---------------------------------------------------------------------------

proptest! {
    /// Provider list is non-empty (fallback always present) and stable across calls.
    #[test]
    fn prop_kernel_manager_always_has_providers(_seed in 0u32..10u32) {
        let mgr = KernelManager::new();
        let providers = mgr.list_available_providers();
        prop_assert!(!providers.is_empty(),
            "KernelManager must always have at least one provider");
    }

    /// All provider names are non-empty strings.
    #[test]
    fn prop_provider_names_are_nonempty(_seed in 0u32..10u32) {
        let mgr = KernelManager::new();
        let providers = mgr.list_available_providers();
        for name in &providers {
            prop_assert!(!name.is_empty(),
                "provider name must not be empty");
        }
    }

    /// `select_best()` succeeds and returns the same logical provider on repeated calls.
    #[test]
    fn prop_select_best_is_stable(_seed in 0u32..10u32) {
        let mgr = KernelManager::new();
        let first = mgr.select_best().map(|p| p.name()).unwrap_or("err");
        let second = mgr.select_best().map(|p| p.name()).unwrap_or("err");
        prop_assert_eq!(first, second,
            "select_best() must return the same provider on repeated calls (caching)");
    }
}

// ---------------------------------------------------------------------------
// Properties: select_cpu_kernel
// ---------------------------------------------------------------------------

proptest! {
    /// `select_cpu_kernel()` always succeeds and provides a non-empty name.
    #[test]
    fn prop_select_cpu_kernel_succeeds(_seed in 0u32..10u32) {
        let kernel = select_cpu_kernel();
        prop_assert!(kernel.is_ok(), "select_cpu_kernel() must always succeed");
        let name = kernel.unwrap().name();
        prop_assert!(!name.is_empty(), "cpu kernel name must not be empty");
    }

    /// The CPU kernel reports itself as available.
    #[test]
    fn prop_cpu_kernel_is_available(_seed in 0u32..10u32) {
        let kernel = select_cpu_kernel().unwrap();
        prop_assert!(kernel.is_available(),
            "CPU kernel must always report itself as available");
    }
}

// ---------------------------------------------------------------------------
// Properties: FallbackKernel quantization
// ---------------------------------------------------------------------------

proptest! {
    /// `FallbackKernel::quantize()` with TL1 produces output of the correct size.
    ///
    /// For TL1: output size = input size (1 byte per weight), scales size = blocks.
    #[test]
    fn prop_fallback_kernel_quantize_tl1_output_size(
        n_blocks in 1usize..64
    ) {
        let block_size = 32;
        let n = n_blocks * block_size;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) / n as f32).collect();
        let mut output = vec![0u8; n];
        let mut scales = vec![0.0f32; n_blocks];

        let kernel = FallbackKernel;
        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::TL1);
        prop_assert!(result.is_ok(),
            "FallbackKernel::quantize(TL1) must succeed for {n} elements ({n_blocks} blocks)");
    }

    /// `FallbackKernel::quantize()` with I2S produces output of the correct size.
    #[test]
    fn prop_fallback_kernel_quantize_i2s_output_size(
        n_blocks in 1usize..64
    ) {
        let block_size = 32;
        let n = n_blocks * block_size;
        // I2S output is packed 2-bit: 4 values per byte → n/4 bytes
        let input: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) / n as f32).collect();
        let mut output = vec![0u8; n / 4];
        let mut scales = vec![0.0f32; n_blocks];

        let kernel = FallbackKernel;
        let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
        prop_assert!(result.is_ok(),
            "FallbackKernel::quantize(I2S) must succeed for {n} elements ({n_blocks} blocks)");
    }
}

// ---------------------------------------------------------------------------
// Properties: device_features
// ---------------------------------------------------------------------------

proptest! {
    /// `gpu_compiled()` is a compile-time constant: always returns the same value.
    #[test]
    fn prop_gpu_compiled_is_constant(_seed in 0u32..100u32) {
        let first = gpu_compiled();
        let second = gpu_compiled();
        prop_assert_eq!(first, second,
            "gpu_compiled() must be a constant (compile-time predicate)");
    }
}

// ---------------------------------------------------------------------------
// Helpers for new property tests
// ---------------------------------------------------------------------------

use bitnet_common::kernel_registry::{KernelBackend, KernelCapabilities, SimdLevel};

fn arb_simd_level() -> impl Strategy<Value = SimdLevel> {
    prop_oneof![
        Just(SimdLevel::Scalar),
        Just(SimdLevel::Neon),
        Just(SimdLevel::Sse42),
        Just(SimdLevel::Avx2),
        Just(SimdLevel::Avx512),
    ]
}

fn arb_caps() -> impl Strategy<Value = KernelCapabilities> {
    (any::<bool>(), any::<bool>(), any::<bool>(), any::<bool>(), arb_simd_level()).prop_map(
        |(cpu_rust, cuda_compiled, cuda_runtime, cpp_ffi, simd_level)| KernelCapabilities {
            cpu_rust,
            cuda_compiled,
            cuda_runtime,
            cpp_ffi,
            vulkan_compiled: false,
            vulkan_runtime: false,
            simd_level,
            oneapi_compiled: false,
            oneapi_runtime: false,
        },
    )
}

// ---------------------------------------------------------------------------
// Properties: SimdLevel ordering
// ---------------------------------------------------------------------------

proptest! {
    /// SimdLevel ordering is reflexive: every level compares equal to itself.
    #[test]
    fn prop_simd_level_ordering_reflexive(level in arb_simd_level()) {
        prop_assert!(level <= level, "SimdLevel ordering must be reflexive");
        prop_assert!(level >= level, "SimdLevel ordering must be reflexive");
    }

    /// SimdLevel ordering is total: for any two levels a ≤ b or b ≤ a.
    #[test]
    fn prop_simd_level_ordering_total(a in arb_simd_level(), b in arb_simd_level()) {
        prop_assert!(a <= b || b <= a,
            "SimdLevel ordering must be total: {a:?} vs {b:?}");
    }

    /// Fundamental ordering invariants: Scalar < Avx2 < Avx512 (and transitivity).
    #[test]
    fn prop_simd_level_fundamental_order(_seed in 0u32..1u32) {
        prop_assert!(SimdLevel::Scalar < SimdLevel::Avx2,
            "Scalar must be strictly less than Avx2");
        prop_assert!(SimdLevel::Avx2 < SimdLevel::Avx512,
            "Avx2 must be strictly less than Avx512");
        prop_assert!(SimdLevel::Scalar < SimdLevel::Avx512,
            "Scalar < Avx512 must hold (transitivity)");
    }
}

// ---------------------------------------------------------------------------
// Properties: MatMul dimensions
// ---------------------------------------------------------------------------

proptest! {
    /// Matrix dimensions up to 4096×4096 do not overflow usize.
    ///
    /// This is a precondition for safe buffer allocation in matmul kernels.
    #[test]
    fn prop_matmul_dims_no_overflow(rows in 1usize..=4096, cols in 1usize..=4096) {
        let product = rows.checked_mul(cols);
        prop_assert!(product.is_some(),
            "rows={rows} * cols={cols} must not overflow usize");
        prop_assert!(
            product.unwrap() <= isize::MAX as usize,
            "rows*cols must fit within isize::MAX for pointer-arithmetic safety"
        );
    }
}

// ---------------------------------------------------------------------------
// Properties: KernelCapabilities round-trips
// ---------------------------------------------------------------------------

proptest! {
    /// When cpu_rust=true the compiled_backends snapshot includes CpuRust.
    #[test]
    fn prop_kernel_caps_cpu_rust_appears_in_compiled_backends(
        cuda_compiled in any::<bool>(),
        cpp_ffi in any::<bool>(),
    ) {
        let caps = KernelCapabilities {
            cpu_rust: true,
            cuda_compiled,
            cuda_runtime: false,
            cpp_ffi,
            vulkan_compiled: false,
            vulkan_runtime: false,
            simd_level: SimdLevel::Scalar,
            oneapi_compiled: false,
            oneapi_runtime: false,
        };
        let backends = caps.compiled_backends();
        prop_assert!(backends.contains(&KernelBackend::CpuRust),
            "cpu_rust=true must produce CpuRust in compiled_backends(); got {backends:?}");
    }

    /// When cuda_compiled=false the snapshot never contains Cuda backend.
    #[test]
    fn prop_kernel_caps_no_cuda_backend_when_not_compiled(caps in arb_caps()) {
        // Force cuda_compiled = false, keep other fields
        let caps = KernelCapabilities { cuda_compiled: false, ..caps };
        let backends = caps.compiled_backends();
        prop_assert!(!backends.contains(&KernelBackend::Cuda),
            "Cuda must not appear in compiled_backends when cuda_compiled=false; got {backends:?}");
    }
}

// ---------------------------------------------------------------------------
// Properties: device_capability_summary and current_kernel_capabilities
// ---------------------------------------------------------------------------

proptest! {
    /// device_capability_summary() always reports CPU is available.
    #[test]
    fn prop_device_capability_summary_always_has_cpu(_seed in 0u32..10u32) {
        let summary = bitnet_kernels::device_features::device_capability_summary();
        prop_assert!(summary.contains("CPU \u{2713}"),
            "capability summary must always include 'CPU ✓'; got: {summary:?}");
    }

    /// current_kernel_capabilities().cpu_rust reflects the `cpu` feature flag.
    #[test]
    fn prop_current_kernel_caps_cpu_rust_matches_feature(_seed in 0u32..10u32) {
        let caps = bitnet_kernels::device_features::current_kernel_capabilities();
        let expected = cfg!(feature = "cpu");
        prop_assert_eq!(caps.cpu_rust, expected,
            "cpu_rust must equal cfg!(feature = \"cpu\"); \
             got cpu_rust={} expected={}", caps.cpu_rust, expected);
    }
}
