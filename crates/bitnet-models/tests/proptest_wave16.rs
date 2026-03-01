//! Property-based tests — wave 16.
//!
//! GGUF tensor type invariants, naming predicate consistency, model config
//! properties, and qk256 tolerance re-export identity.

#![cfg(all(test, feature = "cpu"))]

use bitnet_gguf::{GGUF_MAGIC, check_magic, read_version};
use bitnet_models::formats::gguf::GgufTensorType;
use bitnet_models::names::{is_layernorm_weight, is_projection_weight};
use bitnet_models::{QK256_SIZE_TOLERANCE_PERCENT, qk256_tolerance_bytes};
use proptest::prelude::*;

/// Strategy that produces every known `GgufTensorType` variant.
fn any_tensor_type() -> impl Strategy<Value = GgufTensorType> {
    prop_oneof![
        Just(GgufTensorType::F32),
        Just(GgufTensorType::F16),
        Just(GgufTensorType::F64),
        Just(GgufTensorType::Q4_0),
        Just(GgufTensorType::Q4_1),
        Just(GgufTensorType::Q5_0),
        Just(GgufTensorType::Q5_1),
        Just(GgufTensorType::Q8_0),
        Just(GgufTensorType::Q8_1),
        Just(GgufTensorType::Q2_K),
        Just(GgufTensorType::Q3_K),
        Just(GgufTensorType::Q4_K),
        Just(GgufTensorType::Q5_K),
        Just(GgufTensorType::Q6_K),
        Just(GgufTensorType::Q8_K),
        Just(GgufTensorType::IQ2_S),
        Just(GgufTensorType::I2_S),
    ]
}

/// Strategy for quantized-only types.
fn quantized_tensor_type() -> impl Strategy<Value = GgufTensorType> {
    prop_oneof![
        Just(GgufTensorType::Q4_0),
        Just(GgufTensorType::Q4_1),
        Just(GgufTensorType::Q5_0),
        Just(GgufTensorType::Q5_1),
        Just(GgufTensorType::Q8_0),
        Just(GgufTensorType::Q8_1),
        Just(GgufTensorType::Q2_K),
        Just(GgufTensorType::Q3_K),
        Just(GgufTensorType::Q4_K),
        Just(GgufTensorType::Q5_K),
        Just(GgufTensorType::Q6_K),
        Just(GgufTensorType::Q8_K),
        Just(GgufTensorType::IQ2_S),
        Just(GgufTensorType::I2_S),
    ]
}

// ── GgufTensorType size invariants ──────────────────────────────────────────

proptest! {
    /// element_size is always > 0 for all known tensor types.
    #[test]
    fn gguf_tensor_type_element_size_positive(t in any_tensor_type()) {
        let size = t.element_size();
        prop_assert!(size > 0, "{:?} element_size is 0", t);
    }

    /// F32 element_size is 4 bytes.
    #[test]
    fn gguf_f32_is_4_bytes(_dummy in 0u8..1) {
        prop_assert_eq!(GgufTensorType::F32.element_size(), 4);
    }

    /// F16 element_size is 2 bytes.
    #[test]
    fn gguf_f16_is_2_bytes(_dummy in 0u8..1) {
        prop_assert_eq!(GgufTensorType::F16.element_size(), 2);
    }

    /// F64 element_size is 8 bytes.
    #[test]
    fn gguf_f64_is_8_bytes(_dummy in 0u8..1) {
        prop_assert_eq!(GgufTensorType::F64.element_size(), 8);
    }

    /// block_size >= 1 for all types.
    #[test]
    fn gguf_block_size_positive(t in any_tensor_type()) {
        let bs = t.block_size();
        prop_assert!(bs >= 1, "{:?} block_size is {}", t, bs);
    }

    /// Quantized types have block_size > 1.
    #[test]
    fn gguf_quantized_block_size_gt1(t in quantized_tensor_type()) {
        prop_assert!(t.block_size() > 1, "{:?} block_size should be > 1", t);
    }

    /// For unquantized types, block_size == 1.
    #[test]
    fn gguf_unquantized_block_size_one(
        t in prop_oneof![
            Just(GgufTensorType::F32),
            Just(GgufTensorType::F16),
            Just(GgufTensorType::F64),
        ]
    ) {
        prop_assert_eq!(t.block_size(), 1, "{:?} block_size should be 1", t);
    }

    /// is_quantized is true iff block_size > 1.
    #[test]
    fn gguf_is_quantized_matches_block_size(t in any_tensor_type()) {
        prop_assert_eq!(t.is_quantized(), t.block_size() > 1,
            "{:?}: is_quantized={} but block_size={}",
            t, t.is_quantized(), t.block_size());
    }
}

// ── Naming predicate properties ─────────────────────────────────────────────

proptest! {
    /// Random alphabetic strings are neither LN nor projection.
    #[test]
    fn random_alpha_not_ln_or_proj(name in "[a-zA-Z]{1,30}") {
        prop_assert!(!is_layernorm_weight(&name),
            "random string '{}' wrongly classified as LN", name);
        prop_assert!(!is_projection_weight(&name),
            "random string '{}' wrongly classified as projection", name);
    }

    /// Predicates are stable: calling twice gives the same answer.
    #[test]
    fn naming_predicates_stable(name in "\\PC{0,80}") {
        let ln1 = is_layernorm_weight(&name);
        let ln2 = is_layernorm_weight(&name);
        prop_assert_eq!(ln1, ln2);

        let proj1 = is_projection_weight(&name);
        let proj2 = is_projection_weight(&name);
        prop_assert_eq!(proj1, proj2);
    }

    /// LN and projection are always mutually exclusive.
    #[test]
    fn ln_proj_mutually_exclusive(name in "\\PC{0,80}") {
        let is_ln = is_layernorm_weight(&name);
        let is_proj = is_projection_weight(&name);
        prop_assert!(!(is_ln && is_proj),
            "'{}' classified as both LN and projection", name);
    }
}

// ── qk256 tolerance re-export identity ──────────────────────────────────────

proptest! {
    /// Re-exported qk256_tolerance_bytes matches the quantization crate.
    #[test]
    fn qk256_tolerance_reexport_identity(n in 0usize..10_000_000) {
        let models_val = qk256_tolerance_bytes(n);
        let quant_val = bitnet_quantization::qk256_tolerance_bytes(n);
        prop_assert_eq!(models_val, quant_val);
    }

    /// Re-exported QK256_SIZE_TOLERANCE_PERCENT matches.
    #[test]
    fn qk256_percent_reexport_identity(_dummy in 0u8..1) {
        prop_assert_eq!(
            QK256_SIZE_TOLERANCE_PERCENT,
            bitnet_quantization::QK256_SIZE_TOLERANCE_PERCENT
        );
    }
}

// ── GGUF magic and version ──────────────────────────────────────────────────

proptest! {
    /// check_magic passes for the real GGUF magic bytes.
    #[test]
    fn gguf_magic_valid(_dummy in 0u8..1) {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC);
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&[0u8; 100]);
        prop_assert!(check_magic(&data));
    }

    /// check_magic rejects random non-magic bytes.
    #[test]
    fn gguf_magic_random_fails(
        b0 in 0u8..=255u8,
        b1 in 0u8..=255u8,
        b2 in 0u8..=255u8,
        b3 in 0u8..=255u8,
    ) {
        let bytes = [b0, b1, b2, b3];
        prop_assume!(bytes != GGUF_MAGIC);
        let mut data = Vec::from(bytes);
        data.extend_from_slice(&[0u8; 100]);
        prop_assert!(!check_magic(&data));
    }

    /// read_version succeeds for version 3.
    #[test]
    fn gguf_version_3_valid(_dummy in 0u8..1) {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC);
        data.extend_from_slice(&3u32.to_le_bytes());
        let version = read_version(&data).unwrap();
        prop_assert_eq!(version, 3);
    }

    /// read_version returns None for data shorter than 8 bytes.
    #[test]
    fn gguf_version_short_data(len in 0usize..8) {
        let data = vec![0u8; len];
        prop_assert!(read_version(&data).is_none());
    }
}

// ── GgufTensorType Display / Debug ──────────────────────────────────────────

proptest! {
    /// Debug output is non-empty for all tensor types.
    #[test]
    fn gguf_tensor_type_debug_non_empty(t in any_tensor_type()) {
        let s = format!("{:?}", t);
        prop_assert!(!s.is_empty());
    }

    /// Clone produces equal value.
    #[test]
    fn gguf_tensor_type_clone_eq(t in any_tensor_type()) {
        let t2 = t;
        prop_assert_eq!(t, t2);
    }

    /// from_quant_string roundtrips for canonical names.
    #[test]
    fn gguf_from_quant_string_known(
        (name, expected) in prop_oneof![
            Just(("i2_s", GgufTensorType::I2_S)),
            Just(("iq2_s", GgufTensorType::IQ2_S)),
            Just(("q4_0", GgufTensorType::Q4_0)),
            Just(("q4_1", GgufTensorType::Q4_1)),
            Just(("q8_0", GgufTensorType::Q8_0)),
            Just(("q8_1", GgufTensorType::Q8_1)),
        ]
    ) {
        let parsed = GgufTensorType::from_quant_string(name);
        prop_assert_eq!(parsed, Some(expected), "parsing '{}' failed", name);
    }
}
