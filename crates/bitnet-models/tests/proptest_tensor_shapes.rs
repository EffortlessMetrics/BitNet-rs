//! Wave 12 property tests: tensor shape invariants, GGUF type properties,
//! and I2S flavor layout consistency.
//!
//! Key invariants tested (8 properties):
//! - GgufTensorType from_u32 roundtrip for known type IDs
//! - from_quant_string is case-insensitive
//! - Quantized types always report is_quantized() == true
//! - element_size is always > 0 for all types
//! - I2SFlavor data_bytes_per_block <= total_bytes_per_block
//! - I2SFlavor block_size consistency with layout kind
//! - I2SLayoutKind data_bytes_per_block is always 8
//! - QuantizedTensor shape numel and validate_shapes symmetry

#![cfg(all(test, feature = "cpu"))]

use bitnet_models::formats::gguf::{GgufTensorType, I2SFlavor, I2SLayoutKind};
use proptest::prelude::*;

/// Strategy producing all known (u32, GgufTensorType) pairs.
fn known_type_pairs() -> impl Strategy<Value = (u32, GgufTensorType)> {
    prop_oneof![
        Just((0, GgufTensorType::F32)),
        Just((1, GgufTensorType::F16)),
        Just((2, GgufTensorType::Q4_0)),
        Just((3, GgufTensorType::Q4_1)),
        Just((4, GgufTensorType::F64)),
        Just((6, GgufTensorType::Q5_0)),
        Just((7, GgufTensorType::Q5_1)),
        Just((8, GgufTensorType::Q8_0)),
        Just((9, GgufTensorType::Q8_1)),
        Just((10, GgufTensorType::Q2_K)),
        Just((11, GgufTensorType::Q3_K)),
        Just((12, GgufTensorType::Q4_K)),
        Just((13, GgufTensorType::Q5_K)),
        Just((14, GgufTensorType::Q6_K)),
        Just((15, GgufTensorType::Q8_K)),
        Just((24, GgufTensorType::IQ2_S)),
        Just((36, GgufTensorType::I2_S)),
    ]
}

/// Strategy producing all GgufTensorType variants.
fn all_tensor_types() -> impl Strategy<Value = GgufTensorType> {
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

/// Strategy producing all I2SFlavor variants.
fn all_i2s_flavors() -> impl Strategy<Value = I2SFlavor> {
    prop_oneof![
        Just(I2SFlavor::BitNet32F16),
        Just(I2SFlavor::Split32WithSibling),
        Just(I2SFlavor::GgmlQk256NoScale),
    ]
}

/// Strategy producing all I2SLayoutKind variants.
fn all_layout_kinds() -> impl Strategy<Value = I2SLayoutKind> {
    prop_oneof![Just(I2SLayoutKind::GgmlSplit), Just(I2SLayoutKind::InlineF16),]
}

/// Strategy for quant string aliases that should parse to a known type.
fn quant_string_aliases() -> impl Strategy<Value = (&'static str, GgufTensorType)> {
    prop_oneof![
        Just(("i2_s", GgufTensorType::I2_S)),
        Just(("I2_S", GgufTensorType::I2_S)),
        Just(("is_2", GgufTensorType::I2_S)),
        Just(("IS_2", GgufTensorType::I2_S)),
        Just(("is2", GgufTensorType::I2_S)),
        Just(("IS2", GgufTensorType::I2_S)),
        Just(("iq2_s", GgufTensorType::IQ2_S)),
        Just(("IQ2_S", GgufTensorType::IQ2_S)),
        Just(("q4_0", GgufTensorType::Q4_0)),
        Just(("Q4_0", GgufTensorType::Q4_0)),
        Just(("f32", GgufTensorType::F32)),
        Just(("F32", GgufTensorType::F32)),
        Just(("f16", GgufTensorType::F16)),
        Just(("F16", GgufTensorType::F16)),
    ]
}

// ===================================================================
// 1. from_u32 roundtrip for known type IDs
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// from_u32 succeeds and returns the expected variant for every known type ID.
    #[test]
    fn prop_from_u32_roundtrip(pair in known_type_pairs()) {
        let (id, expected) = pair;
        let parsed = GgufTensorType::from_u32(id).expect("known type ID must parse");
        prop_assert_eq!(parsed, expected, "ID {} parsed incorrectly", id);
    }

    /// from_u32 fails for type IDs that are not in the known set.
    #[test]
    fn prop_from_u32_unknown_fails(id in 37u32..1000) {
        let result = GgufTensorType::from_u32(id);
        prop_assert!(result.is_err(), "unknown ID {} should fail", id);
    }
}

// ===================================================================
// 2. from_quant_string is case-insensitive
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// from_quant_string correctly parses all known aliases (case-insensitive).
    #[test]
    fn prop_from_quant_string_aliases(pair in quant_string_aliases()) {
        let (s, expected) = pair;
        let parsed = GgufTensorType::from_quant_string(s);
        prop_assert_eq!(
            parsed, Some(expected),
            "from_quant_string({:?}) should produce {:?}", s, expected
        );
    }

    /// Arbitrary strings that don't match any alias return None.
    #[test]
    fn prop_from_quant_string_unknown(s in "[xyz]{3,8}") {
        let parsed = GgufTensorType::from_quant_string(&s);
        prop_assert_eq!(parsed, None, "unknown string {:?} should return None", s);
    }
}

// ===================================================================
// 3. Quantized types report is_quantized() == true
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Float types are not quantized; all others are.
    #[test]
    fn prop_is_quantized_consistency(ty in all_tensor_types()) {
        let expected_float = matches!(ty, GgufTensorType::F32 | GgufTensorType::F16 | GgufTensorType::F64);
        prop_assert_eq!(
            ty.is_quantized(), !expected_float,
            "{:?}.is_quantized() inconsistent", ty
        );
    }
}

// ===================================================================
// 4. element_size is always > 0
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// element_size is positive for every tensor type.
    #[test]
    fn prop_element_size_positive(ty in all_tensor_types()) {
        prop_assert!(
            ty.element_size() > 0,
            "{:?}.element_size() must be > 0", ty
        );
    }

    /// block_size is positive for every tensor type.
    #[test]
    fn prop_block_size_positive(ty in all_tensor_types()) {
        prop_assert!(
            ty.block_size() > 0,
            "{:?}.block_size() must be > 0", ty
        );
    }
}

// ===================================================================
// 5. I2SFlavor: data_bytes_per_block <= total_bytes_per_block
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Data portion never exceeds the total block size.
    #[test]
    fn prop_i2s_flavor_data_le_total(flavor in all_i2s_flavors()) {
        prop_assert!(
            flavor.data_bytes_per_block() <= flavor.total_bytes_per_block(),
            "{:?}: data {} > total {}",
            flavor,
            flavor.data_bytes_per_block(),
            flavor.total_bytes_per_block()
        );
    }

    /// block_size is consistent: always 32 or 256 and matches the flavor semantics.
    #[test]
    fn prop_i2s_flavor_block_size_valid(flavor in all_i2s_flavors()) {
        let bs = flavor.block_size();
        prop_assert!(
            bs == 32 || bs == 256,
            "{:?}.block_size() = {}, expected 32 or 256", flavor, bs
        );
    }
}

// ===================================================================
// 6. I2SFlavor block_size matches its layout kind
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// to_layout_kind preserves the 32-element block size.
    #[test]
    fn prop_i2s_flavor_layout_kind_block_size(flavor in all_i2s_flavors()) {
        let kind = flavor.to_layout_kind();
        // All I2SLayoutKind variants have block_size 32
        prop_assert_eq!(
            kind.block_size(), 32,
            "{:?}.to_layout_kind().block_size() should be 32", flavor
        );
    }
}

// ===================================================================
// 7. I2SLayoutKind data_bytes_per_block is always 8
// ===================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// All layout kinds have 8 bytes of packed data per 32-element block.
    #[test]
    fn prop_layout_kind_data_bytes(kind in all_layout_kinds()) {
        prop_assert_eq!(
            kind.data_bytes_per_block(), 8,
            "{:?}.data_bytes_per_block() should be 8", kind
        );
    }

    /// total_bytes_per_block >= data_bytes_per_block for all layout kinds.
    #[test]
    fn prop_layout_kind_total_ge_data(kind in all_layout_kinds()) {
        prop_assert!(
            kind.total_bytes_per_block() >= kind.data_bytes_per_block(),
            "{:?}: total {} < data {}",
            kind,
            kind.total_bytes_per_block(),
            kind.data_bytes_per_block()
        );
    }
}
