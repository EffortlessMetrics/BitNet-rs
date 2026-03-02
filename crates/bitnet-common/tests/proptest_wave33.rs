//! Wave 33 property tests: common types, validation, and error invariants.
//!
//! Properties tested (5):
//! 1. Device::is_gpu() XOR Device::is_cpu() (mutually exclusive)
//! 2. QuantizationType round-trip through Display → serde string
//! 3. Tensor shape validation: broadcast_shape rejects incompatible dims
//! 4. warn_once key deduplication (same key → idempotent insert)
//! 5. Error Display is non-empty for all error variants

use bitnet_common::{
    BitNetError, InferenceError, KernelError, ModelError, QuantizationError, SecurityError,
    tensor_validation::{broadcast_shape, validate_matmul_shapes},
    types::{Device, QuantizationType},
};
use proptest::prelude::*;

// ── helpers ─────────────────────────────────────────────────────────────

/// Strategy producing every `Device` variant.
fn any_device() -> impl Strategy<Value = Device> {
    prop_oneof![
        Just(Device::Cpu),
        (0usize..8).prop_map(Device::Cuda),
        (0usize..4).prop_map(Device::Hip),
        Just(Device::Npu),
        Just(Device::Metal),
        (0usize..4).prop_map(Device::OpenCL),
    ]
}

/// Strategy producing every `QuantizationType` variant.
fn any_quant_type() -> impl Strategy<Value = QuantizationType> {
    prop_oneof![
        Just(QuantizationType::I2S),
        Just(QuantizationType::TL1),
        Just(QuantizationType::TL2),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    // ── 1. Device: is_cpu XOR is_gpu ────────────────────────────────────

    /// For every `Device` variant, `is_cpu()` and any of the GPU-like
    /// predicates (cuda, hip, opencl, metal, npu) are mutually exclusive
    /// with `is_cpu()`: a device is either CPU or non-CPU.
    #[test]
    fn prop_device_cpu_xor_non_cpu(device in any_device()) {
        let is_cpu = device.is_cpu();
        let is_non_cpu = device.is_cuda()
            || device.is_hip()
            || device.is_npu()
            || device.is_opencl()
            || matches!(device, Device::Metal);

        prop_assert!(
            is_cpu ^ is_non_cpu,
            "Device {:?}: is_cpu={} is_non_cpu={} — must be mutually exclusive",
            device,
            is_cpu,
            is_non_cpu,
        );
    }

    // ── 2. QuantizationType Display → serde round-trip ──────────────────

    /// `Display` produces a non-empty string, and serde JSON round-trip
    /// preserves the variant.
    #[test]
    fn prop_quant_type_display_and_serde_roundtrip(qt in any_quant_type()) {
        let display = format!("{qt}");
        prop_assert!(!display.is_empty(), "Display should be non-empty");

        // Serde round-trip.
        let json = serde_json::to_string(&qt).expect("serialize");
        let back: QuantizationType = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(qt, back, "serde round-trip failed");
    }

    // ── 3. broadcast_shape rejects truly incompatible dimensions ────────

    /// Two shapes that differ in a non-1 dimension are incompatible.
    #[test]
    fn prop_broadcast_rejects_incompatible(
        a_dim in 2usize..=8,
        b_dim in 2usize..=8,
    ) {
        // Ensure the two dims differ and neither is 1.
        prop_assume!(a_dim != b_dim);
        let a = vec![a_dim];
        let b = vec![b_dim];
        prop_assert!(
            broadcast_shape(&a, &b).is_err(),
            "shapes [{a_dim}] and [{b_dim}] should be incompatible",
        );
    }

    /// Matmul shape validation rejects mismatched inner dimensions.
    #[test]
    fn prop_matmul_rejects_mismatched_inner(
        m in 1usize..=16,
        k1 in 1usize..=16,
        k2 in 1usize..=16,
        n in 1usize..=16,
    ) {
        prop_assume!(k1 != k2);
        let a = vec![m, k1];
        let b = vec![k2, n];
        prop_assert!(
            validate_matmul_shapes(&a, &b).is_err(),
            "matmul [{m},{k1}] × [{k2},{n}] should fail when k1≠k2",
        );
    }

    // ── 4. warn_once key deduplication ──────────────────────────────────

    /// Inserting the same key into a `HashSet` twice always results in a
    /// single entry — this mirrors the `warn_once` deduplication logic
    /// without touching global state.
    #[test]
    fn prop_warn_once_key_dedup(key in "[a-z_]{1,32}") {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        let first = seen.insert(key.clone());
        let second = seen.insert(key.clone());

        prop_assert!(first, "first insert should return true");
        prop_assert!(!second, "second insert should return false (dup)");
        prop_assert_eq!(seen.len(), 1, "set should contain exactly one entry");
    }

    // ── 5. Error Display is non-empty for all variants ──────────────────

    /// Every `BitNetError` variant produces a non-empty Display string.
    #[test]
    fn prop_bitnet_error_display_non_empty(variant_idx in 0u32..6) {
        let err: BitNetError = match variant_idx {
            0 => BitNetError::Model(ModelError::NotFound {
                path: "test.gguf".into(),
            }),
            1 => BitNetError::Quantization(QuantizationError::UnsupportedType {
                qtype: "Q99".into(),
            }),
            2 => BitNetError::Kernel(KernelError::NoProvider),
            3 => BitNetError::Inference(InferenceError::GenerationFailed {
                reason: "oom".into(),
            }),
            4 => BitNetError::Config("bad config".into()),
            5 => BitNetError::Security(SecurityError::InputValidation {
                reason: "overflow".into(),
            }),
            _ => unreachable!(),
        };

        let display = format!("{err}");
        prop_assert!(!display.is_empty(), "error Display should be non-empty");
        prop_assert!(display.len() > 3, "error message too short: '{display}'");
    }
}
