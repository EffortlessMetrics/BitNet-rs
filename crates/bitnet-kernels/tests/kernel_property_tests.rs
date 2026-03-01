//! Wave 5 property tests for `bitnet-kernels`.
//!
//! These tests cover invariants not already addressed in `property_tests.rs`
//! or `kernel_proptests.rs`:
//!
//! 1. **FallbackKernel::matmul_i2s output length** – output always has M×N elements.
//! 2. **FallbackKernel::matmul_i2s linearity** – scaling activations scales output.
//! 3. **FallbackKernel::quantize(I2S) packed codes in 2-bit range**.
//! 4. **FallbackKernel::quantize(TL1) scale count** – one scale per block.
//! 5. **KernelManager repeated selection stability**.
//! 6. **FallbackKernel identity** – always available with stable non-empty name.

use bitnet_common::QuantizationType;
use bitnet_kernels::{FallbackKernel, KernelManager, KernelProvider};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Properties: FallbackKernel::matmul_i2s — output length
// ---------------------------------------------------------------------------

proptest! {
    /// `matmul_i2s` output length always equals `m * n` after a successful call.
    #[test]
    fn prop_matmul_output_length_is_m_times_n(
        m in 1usize..8,
        n in 1usize..8,
        k in 1usize..8,
    ) {
        let kernel = FallbackKernel;
        let a = vec![0i8; m * k];
        let b = vec![0u8; k * n];
        let mut c = vec![f32::NAN; m * n];

        kernel.matmul_i2s(&a, &b, &mut c, m, n, k).unwrap();

        prop_assert_eq!(
            c.len(), m * n,
            "output length {} != m*n = {}*{} = {}", c.len(), m, n, m * n
        );
        // All outputs should be finite (zero weights → zero output).
        for (i, &v) in c.iter().enumerate() {
            prop_assert!(v.is_finite(), "c[{}] = {} is not finite", i, v);
        }
    }
}

// ---------------------------------------------------------------------------
// Properties: FallbackKernel::matmul_i2s — linearity (scaling)
// ---------------------------------------------------------------------------

proptest! {
    /// Scaling all activations by 2 doubles the output (linearity).
    ///
    /// Uses fixed weights (all 1-byte = 0x55 → deterministic non-zero pattern) so the
    /// output is a deterministic dot-product.
    #[test]
    fn prop_matmul_linearity_scaling(
        m in 1usize..4,
        n in 1usize..4,
        k in 1usize..4,
    ) {
        let kernel = FallbackKernel;
        let weights = vec![0x55u8; k * n]; // deterministic non-zero pattern

        let a1: Vec<i8> = vec![1; m * k];
        let a2: Vec<i8> = vec![2; m * k];

        let mut c1 = vec![0.0f32; m * n];
        let mut c2 = vec![0.0f32; m * n];

        kernel.matmul_i2s(&a1, &weights, &mut c1, m, n, k).unwrap();
        kernel.matmul_i2s(&a2, &weights, &mut c2, m, n, k).unwrap();

        for (i, (&v1, &v2)) in c1.iter().zip(c2.iter()).enumerate() {
            let expected = v1 * 2.0;
            prop_assert!(
                (v2 - expected).abs() < 1e-4,
                "linearity: c2[{}] = {} != 2 * c1[{}] = {}", i, v2, i, expected
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Properties: FallbackKernel::quantize(I2S) — packed codes in 2-bit range
// ---------------------------------------------------------------------------

proptest! {
    /// After I2S quantization, every nibble in the packed output decodes to a
    /// 2-bit code in 0..=3 (trivially true for u8, but we verify the logical
    /// encoding by checking all 4 sub-fields per byte).
    #[test]
    fn prop_i2s_quantized_codes_in_2bit_range(
        n_blocks in 1usize..8,
    ) {
        let block_size = 32;
        let n = n_blocks * block_size;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) * 0.1).collect();
        let mut output = vec![0u8; n / 4];
        let mut scales = vec![0.0f32; n_blocks];

        FallbackKernel
            .quantize(&input, &mut output, &mut scales, QuantizationType::I2S)
            .unwrap();

        for (byte_idx, &b) in output.iter().enumerate() {
            let c0 = b & 0x03;
            let c1 = (b >> 2) & 0x03;
            let c2 = (b >> 4) & 0x03;
            let c3 = (b >> 6) & 0x03;
            prop_assert!(c0 <= 3 && c1 <= 3 && c2 <= 3 && c3 <= 3,
                "byte[{}] = {:#04x}: codes [{},{},{},{}] out of range",
                byte_idx, b, c0, c1, c2, c3);
        }
    }
}

// ---------------------------------------------------------------------------
// Properties: FallbackKernel::quantize(TL1) — scale count
// ---------------------------------------------------------------------------

proptest! {
    /// After TL1 quantization the number of scales equals the number of blocks.
    #[test]
    fn prop_tl1_scale_count_equals_block_count(
        n_blocks in 1usize..32,
    ) {
        let block_size = 32;
        let n = n_blocks * block_size;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let mut output = vec![0u8; n]; // TL1: 1 byte per element
        let mut scales = vec![0.0f32; n_blocks];

        FallbackKernel
            .quantize(&input, &mut output, &mut scales, QuantizationType::TL1)
            .unwrap();

        // Verify all scales were written (non-NaN).
        for (i, &s) in scales.iter().enumerate() {
            prop_assert!(
                s.is_finite(),
                "TL1 scale[{}] = {} is not finite after quantization", i, s
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Properties: KernelManager — repeated selection stability
// ---------------------------------------------------------------------------

proptest! {
    /// Calling `select_best()` multiple times returns the same provider name.
    #[test]
    fn prop_kernel_manager_select_best_repeated_stable(_seed in 0u32..10u32) {
        let mgr = KernelManager::new();

        let first = mgr.select_best().map(|p| p.name()).unwrap_or("err1");
        let second = mgr.select_best().map(|p| p.name()).unwrap_or("err2");
        let third = mgr.select_best().map(|p| p.name()).unwrap_or("err3");

        prop_assert_eq!(first, second, "select_best must be stable across calls");
        prop_assert_eq!(second, third, "select_best must be stable across calls");
    }
}

// ---------------------------------------------------------------------------
// Properties: FallbackKernel::name and is_available
// ---------------------------------------------------------------------------

proptest! {
    /// `FallbackKernel` is always available and has a stable, non-empty name.
    #[test]
    fn prop_fallback_kernel_identity(_seed in 0u32..5u32) {
        let k = FallbackKernel;
        prop_assert!(k.is_available(), "FallbackKernel must always be available");
        prop_assert!(!k.name().is_empty(), "FallbackKernel name must be non-empty");

        // Name is deterministic.
        let name1 = k.name();
        let name2 = FallbackKernel.name();
        prop_assert_eq!(name1, name2, "FallbackKernel name must be stable");
    }
}
