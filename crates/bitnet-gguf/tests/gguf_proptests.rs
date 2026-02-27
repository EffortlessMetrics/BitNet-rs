//! Extended property-based tests for `bitnet-gguf`.
//!
//! Covers areas not exercised by `property_tests.rs`:
//!   - Remaining signed/unsigned `GgufValue` numeric variants
//!   - `GgufValue::Array` construction invariants
//!   - `GgufFileInfo` clone round-trip
//!   - v3 header count fields preserved end-to-end
//!   - `GgufValueType` discriminant values match the GGUF spec
//!   - Metadata key character-class invariants

use bitnet_gguf::{GgufFileInfo, GgufMetadataKv, GgufValue, GgufValueType, parse_header};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn make_header(version: u32, tensor_count: u64, metadata_count: u64) -> Vec<u8> {
    let mut d = Vec::with_capacity(24);
    d.extend_from_slice(b"GGUF");
    d.extend_from_slice(&version.to_le_bytes());
    d.extend_from_slice(&tensor_count.to_le_bytes());
    d.extend_from_slice(&metadata_count.to_le_bytes());
    d
}

// ---------------------------------------------------------------------------
// 1. Signed integer GgufValue variants preserve their bit-patterns exactly
// ---------------------------------------------------------------------------

proptest! {
    /// GgufValue::Int8 stores the provided value verbatim.
    #[test]
    fn prop_gguf_value_int8_round_trip(v in any::<i8>()) {
        let GgufValue::Int8(stored) = GgufValue::Int8(v) else {
            panic!("unexpected variant");
        };
        prop_assert_eq!(stored, v);
    }

    /// GgufValue::Int16 stores the provided value verbatim.
    #[test]
    fn prop_gguf_value_int16_round_trip(v in any::<i16>()) {
        let GgufValue::Int16(stored) = GgufValue::Int16(v) else {
            panic!("unexpected variant");
        };
        prop_assert_eq!(stored, v);
    }

    /// GgufValue::Int32 stores the provided value verbatim.
    #[test]
    fn prop_gguf_value_int32_round_trip(v in any::<i32>()) {
        let GgufValue::Int32(stored) = GgufValue::Int32(v) else {
            panic!("unexpected variant");
        };
        prop_assert_eq!(stored, v);
    }

    /// GgufValue::Int64 stores the provided value verbatim.
    #[test]
    fn prop_gguf_value_int64_round_trip(v in any::<i64>()) {
        let GgufValue::Int64(stored) = GgufValue::Int64(v) else {
            panic!("unexpected variant");
        };
        prop_assert_eq!(stored, v);
    }

    /// GgufValue::Uint16 stores the provided value verbatim.
    #[test]
    fn prop_gguf_value_uint16_round_trip(v in any::<u16>()) {
        let GgufValue::Uint16(stored) = GgufValue::Uint16(v) else {
            panic!("unexpected variant");
        };
        prop_assert_eq!(stored, v);
    }

    /// GgufValue::Uint64 stores the provided value verbatim.
    #[test]
    fn prop_gguf_value_uint64_round_trip(v in any::<u64>()) {
        let GgufValue::Uint64(stored) = GgufValue::Uint64(v) else {
            panic!("unexpected variant");
        };
        prop_assert_eq!(stored, v);
    }

    /// GgufValue::Float64 stores arbitrary finite float64 values verbatim.
    #[test]
    fn prop_gguf_value_float64_finite_round_trip(
        v in proptest::num::f64::NORMAL
            | proptest::num::f64::ZERO
            | proptest::num::f64::NEGATIVE
    ) {
        let GgufValue::Float64(stored) = GgufValue::Float64(v) else {
            panic!("unexpected variant");
        };
        prop_assert_eq!(stored.to_bits(), v.to_bits(), "Float64 must be bit-identical");
    }

    /// GgufValue::Float32 preserves IEEE 754 bit-patterns including ±Inf and NaN.
    #[test]
    fn prop_gguf_value_float32_bit_pattern_preserved(bits in any::<u32>()) {
        let v = f32::from_bits(bits);
        let GgufValue::Float32(stored) = GgufValue::Float32(v) else {
            panic!("unexpected variant");
        };
        prop_assert_eq!(
            stored.to_bits(), bits,
            "Float32 bit-pattern must survive a GgufValue round-trip"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. GgufValue::Array construction invariants
// ---------------------------------------------------------------------------

proptest! {
    /// An Array wrapping uint32 elements has the correct element count and type tag.
    #[test]
    fn prop_gguf_value_array_uint32_count_and_type(
        values in prop::collection::vec(any::<u32>(), 0..=64)
    ) {
        let elems: Vec<GgufValue> = values.iter().map(|&v| GgufValue::Uint32(v)).collect();
        let expected_len = elems.len();
        let arr = GgufValue::Array(GgufValueType::Uint32, elems);
        let GgufValue::Array(elem_type, stored) = arr else {
            panic!("unexpected variant");
        };
        prop_assert_eq!(elem_type, GgufValueType::Uint32);
        prop_assert_eq!(stored.len(), expected_len);
    }

    /// An empty Array retains its element-type discriminant.
    #[test]
    fn prop_gguf_value_empty_array_preserves_type_tag(disc in 0u32..=12u32) {
        let elem_type = GgufValueType::from_u32(disc).unwrap();
        let arr = GgufValue::Array(elem_type, vec![]);
        let GgufValue::Array(stored_type, stored_elems) = arr else {
            panic!("unexpected variant");
        };
        prop_assert_eq!(stored_type, elem_type);
        prop_assert!(stored_elems.is_empty());
    }
}

// ---------------------------------------------------------------------------
// 3. GgufFileInfo clone round-trip
// ---------------------------------------------------------------------------

proptest! {
    /// Cloning a GgufFileInfo yields a struct that is field-for-field identical.
    #[test]
    fn prop_gguf_file_info_clone_is_identical(
        version in 2u32..=3u32,
        tensor_count in 0u64..1_000_000u64,
        metadata_count in 0u64..1_000_000u64,
        // Alignment must be a power of two; pick exponents 0..=16 → 1..=65536.
        exp in 0u32..=16u32,
    ) {
        let alignment = 1u32 << exp;
        let info = GgufFileInfo { version, tensor_count, metadata_count, alignment };
        let cloned = info.clone();
        prop_assert_eq!(info.version, cloned.version);
        prop_assert_eq!(info.tensor_count, cloned.tensor_count);
        prop_assert_eq!(info.metadata_count, cloned.metadata_count);
        prop_assert_eq!(info.alignment, cloned.alignment);
    }
}

// ---------------------------------------------------------------------------
// 4. parse_header preserves tensor_count and metadata_count (v2 and v3)
// ---------------------------------------------------------------------------

proptest! {
    /// For any valid v2 header the parsed tensor_count and metadata_count must
    /// exactly match the values written into the header bytes.
    #[test]
    fn prop_parse_header_v2_counts_round_trip(
        tensor_count in 0u64..=u64::MAX,
        metadata_count in 0u64..=u64::MAX,
    ) {
        let data = make_header(2, tensor_count, metadata_count);
        let info = parse_header(&data).expect("valid v2 header must parse");
        prop_assert_eq!(info.tensor_count, tensor_count);
        prop_assert_eq!(info.metadata_count, metadata_count);
    }

    /// For any valid v3 header the parsed counts must exactly match the written values.
    #[test]
    fn prop_parse_header_v3_counts_round_trip(
        tensor_count in 0u64..=u64::MAX,
        metadata_count in 0u64..=u64::MAX,
        exp in 0u32..=16u32,
    ) {
        let alignment = 1u32 << exp;
        let mut data = make_header(3, tensor_count, metadata_count);
        data.extend_from_slice(&alignment.to_le_bytes());
        let info = parse_header(&data).expect("valid v3 header must parse");
        prop_assert_eq!(info.tensor_count, tensor_count);
        prop_assert_eq!(info.metadata_count, metadata_count);
        prop_assert_eq!(info.alignment, alignment);
    }
}

// ---------------------------------------------------------------------------
// 5. GgufValueType discriminant values match the GGUF specification
// ---------------------------------------------------------------------------

// Each variant's u32 repr must equal the value mandated by the GGUF spec.
// These are unit tests expressed with proptest's `prop_assert_eq!` to make
// any future mismatch easy to diagnose in test output.
proptest! {
    #[test]
    fn prop_value_type_discriminants_match_spec(
        // Iterate over all 13 defined discriminants.
        disc in 0u32..=12u32
    ) {
        let vt = GgufValueType::from_u32(disc).expect("all discriminants 0..=12 must be valid");
        // Verify the round-trip: casting back to u32 via the repr gives the
        // same numeric value we started with.
        let roundtripped = vt as u32;
        prop_assert_eq!(
            roundtripped, disc,
            "GgufValueType discriminant {} must round-trip through as u32", disc
        );
    }
}

// ---------------------------------------------------------------------------
// 6. Metadata key character-class invariants
// ---------------------------------------------------------------------------

proptest! {
    /// A metadata key whose content is ASCII printable text is preserved verbatim.
    #[test]
    fn prop_metadata_kv_ascii_printable_key_preserved(
        key in "[!-~]{1,64}",
        v in any::<i64>(),
    ) {
        let kv = GgufMetadataKv { key: key.clone(), value: GgufValue::Int64(v) };
        prop_assert_eq!(&kv.key, &key, "ASCII printable key must survive GgufMetadataKv construction");
        prop_assert!(kv.key.is_ascii(), "key must remain valid ASCII");
    }

    /// A metadata key containing Unicode characters outside ASCII is stored without
    /// truncation (GgufMetadataKv imposes no encoding restriction).
    #[test]
    fn prop_metadata_kv_unicode_key_no_truncation(
        key in prop::string::string_regex("[\\p{L}][\\p{L}0-9_.]{0,63}").unwrap(),
        v in any::<bool>(),
    ) {
        let kv = GgufMetadataKv { key: key.clone(), value: GgufValue::Bool(v) };
        prop_assert_eq!(kv.key.len(), key.len(), "Unicode key byte length must be preserved");
        prop_assert_eq!(&kv.key, &key);
    }

    /// Cloning a GgufMetadataKv with a Bool value preserves both key and value.
    #[test]
    fn prop_metadata_kv_bool_clone_consistency(
        key in "[a-zA-Z][a-zA-Z0-9_.]{0,31}",
        v in any::<bool>(),
    ) {
        let kv = GgufMetadataKv { key: key.clone(), value: GgufValue::Bool(v) };
        let cloned = kv.clone();
        prop_assert_eq!(&kv.key, &cloned.key);
        match (&kv.value, &cloned.value) {
            (GgufValue::Bool(a), GgufValue::Bool(b)) => prop_assert_eq!(a, b),
            _ => prop_assert!(false, "clone changed GgufValue variant"),
        }
    }

    /// Cloning a GgufMetadataKv with a String value preserves both key and value exactly.
    #[test]
    fn prop_metadata_kv_string_value_clone_consistency(
        key in "[a-z][a-z0-9_.]{0,31}",
        val in any::<String>(),
    ) {
        let kv = GgufMetadataKv { key: key.clone(), value: GgufValue::String(val.clone()) };
        let cloned = kv.clone();
        prop_assert_eq!(&kv.key, &cloned.key);
        match (&kv.value, &cloned.value) {
            (GgufValue::String(a), GgufValue::String(b)) => {
                prop_assert_eq!(a, b, "cloned String value must be byte-identical");
            }
            _ => prop_assert!(false, "clone changed GgufValue variant"),
        }
    }
}
