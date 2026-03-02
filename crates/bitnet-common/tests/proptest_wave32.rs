//! Property-based tests for bitnet-common (wave 32).

use bitnet_common::tensor_validation::{broadcast_shape, can_broadcast};
use bitnet_common::{
    BitNetError, ConfigBuilder, Device, InferenceError, KernelError, ModelError, QuantizationError,
    QuantizationType, SecurityError,
};
use proptest::prelude::*;

// ── Helpers ────────────────────────────────────────────────────────────

fn arb_device() -> BoxedStrategy<Device> {
    prop_oneof![
        Just(Device::Cpu),
        (0usize..4).prop_map(Device::Cuda),
        (0usize..4).prop_map(Device::Hip),
        Just(Device::Npu),
        Just(Device::Metal),
        (0usize..4).prop_map(Device::OpenCL),
    ]
    .boxed()
}

fn arb_quantization_type() -> BoxedStrategy<QuantizationType> {
    prop_oneof![
        Just(QuantizationType::I2S),
        Just(QuantizationType::TL1),
        Just(QuantizationType::TL2),
    ]
    .boxed()
}

fn arb_shape(min_dims: usize, max_dims: usize) -> BoxedStrategy<Vec<usize>> {
    prop::collection::vec(1usize..=16, min_dims..=max_dims).boxed()
}

// ── Tests ──────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    // 1. Device serde roundtrip
    #[test]
    fn proptest_wave32_device_serde_roundtrip(device in arb_device()) {
        let json = serde_json::to_string(&device).unwrap();
        let recovered: Device = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(device, recovered);
    }

    // 2. QuantizationType serde roundtrip
    #[test]
    fn proptest_wave32_quantization_type_serde_roundtrip(qt in arb_quantization_type()) {
        let json = serde_json::to_string(&qt).unwrap();
        let recovered: QuantizationType = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(qt, recovered);
    }

    // 3. QuantizationType display never panics
    #[test]
    fn proptest_wave32_quantization_type_display(qt in arb_quantization_type()) {
        let s = format!("{}", qt);
        prop_assert!(!s.is_empty());
    }

    // 4. Tensor shape validation: reject zero dims via broadcast
    #[test]
    fn proptest_wave32_zero_dim_broadcast(
        good_shape in arb_shape(1, 3),
        zero_pos in 0usize..3,
    ) {
        let zero_pos = zero_pos % good_shape.len();
        let mut bad_shape = good_shape.clone();
        bad_shape[zero_pos] = 0;
        // Broadcasting with a zero dim should fail (dims 0 vs non-zero are incompatible)
        // unless the other shape also has 0 at that position
        if good_shape[zero_pos] != 1 {
            let result = broadcast_shape(&bad_shape, &good_shape);
            // Zero is not 1 and not equal to the other dim, so it should fail
            prop_assert!(result.is_err(),
                "expected broadcast failure for shapes {:?} vs {:?}", bad_shape, good_shape);
        }
    }

    // 5. Broadcast shape commutativity
    #[test]
    fn proptest_wave32_broadcast_commutativity(
        a in arb_shape(1, 4),
        b in arb_shape(1, 4),
    ) {
        let ab = broadcast_shape(&a, &b);
        let ba = broadcast_shape(&b, &a);
        match (ab, ba) {
            (Ok(ab_shape), Ok(ba_shape)) => {
                prop_assert_eq!(ab_shape, ba_shape,
                    "broadcast not commutative: {:?} vs {:?}", a, b);
            }
            (Err(_), Err(_)) => {} // Both fail — OK
            (Ok(_), Err(_)) | (Err(_), Ok(_)) => {
                prop_assert!(false,
                    "broadcast commutativity mismatch for {:?} vs {:?}", a, b);
            }
        }
    }

    // 6. Shape product overflow detection: large dims don't panic
    #[test]
    fn proptest_wave32_shape_product_no_panic(
        dims in prop::collection::vec(1usize..=1_000_000, 1..=4),
    ) {
        // Just ensure product computation doesn't panic
        let product: Option<usize> = dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d));
        // If overflow, checked_mul returns None — that's fine
        if let Some(p) = product {
            prop_assert!(p > 0);
        }
    }

    // 7. Error display never panics — BitNetError variants
    #[test]
    fn proptest_wave32_error_display_model(
        msg in "[a-zA-Z0-9 ]{1,50}",
    ) {
        let err = BitNetError::Model(ModelError::NotFound { path: msg.clone() });
        let s = format!("{}", err);
        prop_assert!(!s.is_empty());
    }

    // 8. Error display — QuantizationError
    #[test]
    fn proptest_wave32_error_display_quantization(
        msg in "[a-zA-Z0-9 ]{1,50}",
    ) {
        let err = BitNetError::Quantization(QuantizationError::UnsupportedType { qtype: msg });
        let s = format!("{}", err);
        prop_assert!(!s.is_empty());
    }

    // 9. Error display — KernelError
    #[test]
    fn proptest_wave32_error_display_kernel(
        msg in "[a-zA-Z0-9 ]{1,50}",
    ) {
        let err = BitNetError::Kernel(KernelError::ExecutionFailed { reason: msg });
        let s = format!("{}", err);
        prop_assert!(!s.is_empty());
    }

    // 10. Error display — InferenceError
    #[test]
    fn proptest_wave32_error_display_inference(
        msg in "[a-zA-Z0-9 ]{1,50}",
    ) {
        let err = BitNetError::Inference(InferenceError::GenerationFailed { reason: msg });
        let s = format!("{}", err);
        prop_assert!(!s.is_empty());
    }

    // 11. Error display — SecurityError
    #[test]
    fn proptest_wave32_error_display_security(
        msg in "[a-zA-Z0-9 ]{1,50}",
    ) {
        let err = BitNetError::Security(SecurityError::InputValidation { reason: msg });
        let s = format!("{}", err);
        prop_assert!(!s.is_empty());
    }

    // 12. Config validation: reject zero vocab_size
    #[test]
    fn proptest_wave32_config_reject_zero_vocab(
        _dummy in 0u8..1,
    ) {
        // Default config has valid fields; overriding vocab_size to 0 should fail
        let result = ConfigBuilder::new()
            .vocab_size(0)
            .build();
        prop_assert!(result.is_err());
    }

    // 13. Config validation: reject zero hidden_size
    #[test]
    fn proptest_wave32_config_reject_zero_hidden(
        _dummy in 0u8..1,
    ) {
        let result = ConfigBuilder::new()
            .hidden_size(0)
            .build();
        prop_assert!(result.is_err());
    }

    // 14. Broadcast: self-broadcast is identity
    #[test]
    fn proptest_wave32_broadcast_self_identity(
        shape in arb_shape(1, 4),
    ) {
        let result = broadcast_shape(&shape, &shape).unwrap();
        prop_assert_eq!(result, shape);
    }

    // 15. can_broadcast consistent with broadcast_shape
    #[test]
    fn proptest_wave32_can_broadcast_consistent(
        a in arb_shape(1, 4),
        b in arb_shape(1, 4),
    ) {
        let result = broadcast_shape(&a, &b);
        let can = can_broadcast(&a, &b);
        prop_assert_eq!(result.is_ok(), can,
            "inconsistency: broadcast_shape={:?}, can_broadcast={} for {:?} vs {:?}",
            result, can, a, b);
    }
}
