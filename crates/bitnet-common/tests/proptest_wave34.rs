//! Wave 34 property tests for `bitnet-common`.
//!
//! 15 properties covering:
//! - Device enum serde round-trip
//! - QuantizationType display is non-empty
//! - BitNetError display is non-empty
//! - Broadcast shape computation: commutative
//! - Tensor shape validation: positive dims always valid
//! - Config serialization round-trip (JSON)

use bitnet_common::config::InferenceConfig;
use bitnet_common::error::{BitNetError, InferenceError, KernelError, ModelError};
use bitnet_common::tensor_validation::{broadcast_shape, can_broadcast};
use bitnet_common::types::{Device, QuantizationType};
use proptest::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ── Helpers ─────────────────────────────────────────────────────────────────

fn all_devices() -> impl Strategy<Value = Device> {
    prop_oneof![
        Just(Device::Cpu),
        (0usize..4).prop_map(Device::Cuda),
        (0usize..4).prop_map(Device::Hip),
        Just(Device::Npu),
        Just(Device::Metal),
        (0usize..4).prop_map(Device::OpenCL),
    ]
}

fn all_qtypes() -> impl Strategy<Value = QuantizationType> {
    prop_oneof![
        Just(QuantizationType::I2S),
        Just(QuantizationType::TL1),
        Just(QuantizationType::TL2),
    ]
}

fn hash_of<T: Hash>(t: &T) -> u64 {
    let mut h = DefaultHasher::new();
    t.hash(&mut h);
    h.finish()
}

/// Strategy for a small shape (1-4 dimensions, each 1..16).
fn small_shape(max_dims: usize) -> impl Strategy<Value = Vec<usize>> {
    proptest::collection::vec(1usize..16, 1..=max_dims)
}

// ── 1. Device serde round-trip ──────────────────────────────────────────────

proptest! {
    /// JSON serialize → deserialize preserves the Device variant.
    #[test]
    fn prop_device_serde_roundtrip(dev in all_devices()) {
        let json = serde_json::to_string(&dev).unwrap();
        let back: Device = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(dev, back);
    }

    /// Device hash is consistent: equal values produce equal hashes.
    #[test]
    fn prop_device_hash_consistency(dev in all_devices()) {
        let h1 = hash_of(&dev);
        let h2 = hash_of(&dev);
        prop_assert_eq!(h1, h2);
    }
}

// ── 2. QuantizationType display is non-empty ────────────────────────────────

proptest! {
    /// Display output for every QuantizationType variant is non-empty ASCII.
    #[test]
    fn prop_qtype_display_non_empty(qt in all_qtypes()) {
        let s = format!("{qt}");
        prop_assert!(!s.is_empty(), "QuantizationType display is empty for {qt:?}");
        prop_assert!(s.is_ascii(), "QuantizationType display is not ASCII: {s}");
    }

    /// QuantizationType serde round-trip.
    #[test]
    fn prop_qtype_serde_roundtrip(qt in all_qtypes()) {
        let json = serde_json::to_string(&qt).unwrap();
        let back: QuantizationType = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(qt, back);
    }
}

// ── 3. BitNetError display is non-empty ─────────────────────────────────────

proptest! {
    /// Every BitNetError variant has a non-empty Display string.
    #[test]
    fn prop_bitnet_error_display_non_empty(variant in 0u8..4) {
        let err: BitNetError = match variant {
            0 => BitNetError::Model(ModelError::NotFound { path: "x".into() }),
            1 => BitNetError::Kernel(KernelError::NoProvider),
            2 => BitNetError::Inference(InferenceError::InvalidInput { reason: "bad".into() }),
            _ => BitNetError::Config("cfg err".into()),
        };
        let s = format!("{err}");
        prop_assert!(!s.is_empty(), "BitNetError display is empty for variant {variant}");
    }

    /// BitNetError Debug output is non-empty.
    #[test]
    fn prop_bitnet_error_debug_non_empty(variant in 0u8..3) {
        let err: BitNetError = match variant {
            0 => BitNetError::Model(ModelError::InvalidFormat { format: "fmt".into() }),
            1 => BitNetError::Kernel(KernelError::ExecutionFailed { reason: "exec".into() }),
            _ => BitNetError::Config("c".into()),
        };
        let s = format!("{err:?}");
        prop_assert!(!s.is_empty());
    }
}

// ── 4. Broadcast shape: commutative ─────────────────────────────────────────

proptest! {
    /// broadcast(a, b) == broadcast(b, a) — commutativity.
    #[test]
    fn prop_broadcast_commutative(
        a in small_shape(4),
        b in small_shape(4),
    ) {
        let ab = broadcast_shape(&a, &b);
        let ba = broadcast_shape(&b, &a);
        match (ab, ba) {
            (Ok(ref s1), Ok(ref s2)) => {
                prop_assert_eq!(s1, s2);
            }
            (Err(_), Err(_)) => { /* both fail — ok */ }
            (ref l, ref r) => {
                prop_assert!(false,
                    "broadcast symmetry broken: {:?},{:?} → {:?} vs {:?}", a, b, l, r);
            }
        }
    }

    /// can_broadcast agrees with broadcast_shape.
    #[test]
    fn prop_can_broadcast_consistent(
        a in small_shape(3),
        b in small_shape(3),
    ) {
        let result = broadcast_shape(&a, &b);
        let can = can_broadcast(&a, &b);
        prop_assert_eq!(can, result.is_ok());
    }

    /// Broadcasting a shape with itself always succeeds and returns itself.
    #[test]
    fn prop_broadcast_self(shape in small_shape(4)) {
        let result = broadcast_shape(&shape, &shape).unwrap();
        prop_assert_eq!(result, shape);
    }
}

// ── 5. Tensor shape validation: positive dims ───────────────────────────────

proptest! {
    /// Shapes with all positive dimensions are always broadcast-compatible
    /// with themselves.
    #[test]
    fn prop_positive_dims_self_compatible(shape in small_shape(4)) {
        prop_assert!(can_broadcast(&shape, &shape),
            "shape {shape:?} should be self-compatible");
    }

    /// A scalar (empty shape) is broadcast-compatible with any shape.
    #[test]
    fn prop_scalar_broadcasts_with_anything(shape in small_shape(4)) {
        let scalar: Vec<usize> = vec![];
        prop_assert!(can_broadcast(&scalar, &shape),
            "scalar should broadcast with {shape:?}");
    }
}

// ── 6. Config serialization round-trip ──────────────────────────────────────

proptest! {
    /// InferenceConfig JSON round-trip preserves all fields.
    #[test]
    fn prop_inference_config_serde_roundtrip(
        max_len in 1usize..8192,
        temp in 0.0f32..2.0,
        top_k in 0usize..100,
    ) {
        let cfg = InferenceConfig {
            max_length: max_len,
            temperature: temp,
            top_k: Some(top_k),
            ..InferenceConfig::default()
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: InferenceConfig = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(back.max_length, cfg.max_length);
        prop_assert!((back.temperature - cfg.temperature).abs() < 1e-6);
        prop_assert_eq!(back.top_k, cfg.top_k);
    }

    /// Device serialization is compact (no extraneous whitespace).
    #[test]
    fn prop_device_json_compact(dev in all_devices()) {
        let json = serde_json::to_string(&dev).unwrap();
        prop_assert!(!json.contains('\n'), "Device JSON should be single-line");
        prop_assert!(!json.is_empty());
    }

    /// QuantizationType Clone produces an equal value.
    #[test]
    fn prop_qtype_clone_eq(qt in all_qtypes()) {
        #[allow(clippy::clone_on_copy)]
        let cloned = qt.clone();
        prop_assert_eq!(qt, cloned);
    }

    /// Device ordering is total: for any two devices, a ≤ b or b ≤ a.
    #[test]
    fn prop_device_ordering_total(a in all_devices(), b in all_devices()) {
        prop_assert!(a <= b || b <= a);
    }
}
