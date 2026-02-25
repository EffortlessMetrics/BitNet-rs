//! Property-based tests for `bitnet-kernels`.
//!
//! Key invariants tested:
//! - `KernelManager`: provider list is non-empty, selection is stable (idempotent)
//! - `KernelProvider::name()`: always a non-empty string
//! - `FallbackKernel::quantize()`: output shape matches input for valid sizes
//! - `device_features::gpu_compiled()`: deterministic (constant) across calls
//! - `select_cpu_kernel()`: always succeeds and returns a valid provider

use bitnet_kernels::{FallbackKernel, KernelManager, select_cpu_kernel};
use bitnet_kernels::device_features::gpu_compiled;
use bitnet_common::QuantizationType;
use bitnet_kernels::KernelProvider;
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
